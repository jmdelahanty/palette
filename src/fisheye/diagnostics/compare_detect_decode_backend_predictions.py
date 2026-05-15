"""Compare fixed-frame YOLO predictions from two decode backends."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np
import torch

from fisheye.shared.pynvvc_luma_rgb import BACKEND_PYNVVC_LUMA_RGB
from fisheye.shared.pynvvc_luma_rgb import BACKEND_PYNVVC_NV12_RGB
from fisheye.shared.pynvvc_luma_rgb import PYNVVC_BACKENDS
from fisheye.shared.pynvvc_luma_rgb import PynvvcLumaRgbReader
from fisheye.shared.pynvvc_luma_rgb import preprocess_luma_rgb
from fisheye.shared.pynvvc_luma_rgb import preprocess_nv12_rgb

def _default_cache_root() -> Path:
    explicit = os.environ.get("PALETTE_JOB_CACHE")
    if explicit:
        return Path(explicit).expanduser()

    user = os.environ.get("USER")
    job_id = os.environ.get("LSB_JOBID")
    if user and job_id:
        return Path("/scratch") / user / job_id / "palette_cache"

    xdg_cache = os.environ.get("XDG_CACHE_HOME")
    if xdg_cache:
        return Path(xdg_cache).expanduser() / "palette"

    return Path(tempfile.gettempdir()) / f"palette-{user or 'unknown'}-cache"


def _ensure_headless_cache_env() -> None:
    cache_root = _default_cache_root()
    yolo_config = cache_root / "ultralytics"
    try:
        yolo_config.mkdir(parents=True, exist_ok=True)
    except OSError:
        fallback = Path(tempfile.gettempdir()) / "palette-ultralytics-cache"
        fallback.mkdir(parents=True, exist_ok=True)
        yolo_config = fallback
    os.environ.setdefault("YOLO_CONFIG_DIR", str(yolo_config))


_ensure_headless_cache_env()

from fisheye.diagnostics import benchmark_detect_stage_split as stage
from fisheye.diagnostics.detect_compute_smoke import _apply_model_runtime_optimizations
from fisheye.diagnostics.detect_compute_smoke import _build_predict_kwargs
from fisheye.diagnostics.detect_compute_smoke import _resolve_smoke_resize
from fisheye.diagnostics.detect_compute_smoke import _resize_to_imgsz


BACKEND_CHOICES = (*stage.BACKEND_CHOICES, *PYNVVC_BACKENDS)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be > 0")
    return parsed


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be >= 0")
    return parsed


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video_path", type=Path)
    parser.add_argument("--model", type=Path, default=None, help="YOLO model path.")
    parser.add_argument("--config", type=Path, default=None, help="Optional YOLO detect config.")
    parser.add_argument("--backend-a", choices=BACKEND_CHOICES, default=stage.BACKEND_DECORD_GPU)
    parser.add_argument("--backend-b", choices=BACKEND_CHOICES, default=BACKEND_PYNVVC_NV12_RGB)
    parser.add_argument("--frames", nargs="+", type=_non_negative_int, required=True)
    parser.add_argument("--batch-size", type=_positive_int, default=16)
    parser.add_argument(
        "--resize",
        nargs=2,
        type=_positive_int,
        default=None,
        metavar=("WIDTH", "HEIGHT"),
        help="Resize before inference. Defaults to config video.resize, then detection.resize_dims.",
    )
    parser.add_argument("--conf", type=float, default=None)
    parser.add_argument("--iou", type=float, default=None)
    parser.add_argument("--max-det", type=int, default=None)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--force-fp32", action="store_true")
    parser.add_argument("--max-bbox-diff", type=float, default=None)
    parser.add_argument("--max-score-diff", type=float, default=None)
    parser.add_argument("--fail-on-count-mismatch", action="store_true")
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args(argv)


def _selected_frames(frames: Sequence[int]) -> list[int]:
    selected = sorted({int(frame) for frame in frames})
    if not selected:
        raise ValueError("At least one frame is required.")
    return selected


def _decode_pynvvc_frames(
    *,
    video_path: Path,
    backend: str,
    frames: Sequence[int],
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    resize_hw: tuple[int, int],
) -> tuple[torch.Tensor, dict[str, Any]]:
    if min(frames) < 0:
        raise ValueError("Frame indices must be >= 0.")

    reader = PynvvcLumaRgbReader(video_path, start_frame=0, gpu_id=0)
    selected = set(int(frame) for frame in frames)
    selected_raw: dict[int, torch.Tensor] = {}
    current_frame = 0
    max_frame = max(selected)
    decode_seconds = 0.0

    try:
        while current_frame <= max_frame:
            request = min(int(batch_size), max_frame - current_frame + 1)
            t0 = time.perf_counter()
            decoded = reader.decode_next(request)
            decode_seconds += time.perf_counter() - t0
            if not decoded:
                break
            for offset, tensor in enumerate(decoded):
                frame_index = current_frame + offset
                if frame_index in selected:
                    selected_raw[frame_index] = tensor
            current_frame += len(decoded)

        missing = [frame for frame in frames if frame not in selected_raw]
        if missing:
            raise RuntimeError(f"{backend} did not decode selected frames: {missing}")

        ordered = [selected_raw[int(frame)] for frame in frames]
        t0 = time.perf_counter()
        if backend == BACKEND_PYNVVC_NV12_RGB:
            processed = preprocess_nv12_rgb(
                ordered,
                source_height=reader.source_height,
                device=device,
                dtype=dtype,
                resize_hw=resize_hw,
            )
            preprocess_mode = "nv12_rgb_bt601_limited"
        else:
            processed = preprocess_luma_rgb(
                ordered,
                source_height=reader.source_height,
                device=device,
                dtype=dtype,
                resize_hw=resize_hw,
            )
            preprocess_mode = "luma_rgb"
        preprocess_seconds = time.perf_counter() - t0
        return processed, {
            "decode_seconds": float(decode_seconds),
            "preprocess_seconds": float(preprocess_seconds),
            "source_width": int(reader.source_width),
            "source_height": int(reader.source_height),
            "frame_rate": float(reader.frame_rate),
            "codec": reader.codec,
            "preprocess_mode": preprocess_mode,
        }
    finally:
        reader.close()


def _decode_standard_frames(
    *,
    video_path: Path,
    backend: str,
    frames: Sequence[int],
    device: torch.device,
    dtype: torch.dtype,
    resize: tuple[int, int] | None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if backend == stage.BACKEND_OPENCV:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"OpenCV cannot open video: {video_path}")
        decoded: list[np.ndarray] = []
        t0 = time.perf_counter()
        try:
            for frame in frames:
                cap.set(cv2.CAP_PROP_POS_FRAMES, float(frame))
                ok, image = cap.read()
                if not ok:
                    raise RuntimeError(f"OpenCV failed to decode frame {frame}")
                decoded.append(image)
        finally:
            cap.release()
        decode_seconds = time.perf_counter() - t0
    else:
        t0 = time.perf_counter()
        reader_info = stage._resolve_backend_reader(video_path, backend, 0)
        try:
            decoded = stage._decode_batch(reader_info, frames)
        finally:
            stage._release_reader(reader_info)
        decode_seconds = time.perf_counter() - t0

    t0 = time.perf_counter()
    processed = stage._preprocess_batch(
        decoded=decoded,
        backend=backend,
        device=device,
        dtype=dtype,
        resize=resize,
    )
    preprocess_seconds = time.perf_counter() - t0
    return processed, {
        "decode_seconds": float(decode_seconds),
        "preprocess_seconds": float(preprocess_seconds),
    }


def _decode_backend_frames(
    *,
    video_path: Path,
    backend: str,
    frames: Sequence[int],
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    resize: tuple[int, int] | None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if backend in PYNVVC_BACKENDS:
        if device.type != "cuda":
            raise RuntimeError(f"{backend} requires CUDA.")
        if resize is None:
            raise RuntimeError(f"{backend} requires a resolved resize.")
        width, height = resize
        return _decode_pynvvc_frames(
            video_path=video_path,
            backend=backend,
            frames=frames,
            batch_size=batch_size,
            device=device,
            dtype=dtype,
            resize_hw=(int(height), int(width)),
        )
    return _decode_standard_frames(
        video_path=video_path,
        backend=backend,
        frames=frames,
        device=device,
        dtype=dtype,
        resize=resize,
    )


def _prediction_rows(
    predictions: Sequence[Any],
    *,
    frames: Sequence[int],
    inference_height: int,
    inference_width: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for frame, result in zip(frames, predictions):
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            rows.append(
                {
                    "frame": int(frame),
                    "count": 0,
                    "boxes_norm": [],
                    "scores": [],
                    "class_ids": [],
                }
            )
            continue

        boxes_xyxy = boxes.xyxy.detach().cpu().numpy().astype(np.float64, copy=False)
        scores = boxes.conf.detach().cpu().numpy().astype(np.float64, copy=False)
        class_ids = (
            boxes.cls.detach().cpu().numpy().astype(np.int64, copy=False)
            if getattr(boxes, "cls", None) is not None
            else np.zeros(boxes_xyxy.shape[0], dtype=np.int64)
        )
        boxes_norm = boxes_xyxy.copy()
        boxes_norm[:, [0, 2]] /= float(inference_width)
        boxes_norm[:, [1, 3]] /= float(inference_height)
        rows.append(
            {
                "frame": int(frame),
                "count": int(boxes_xyxy.shape[0]),
                "boxes_norm": boxes_norm.tolist(),
                "scores": scores.tolist(),
                "class_ids": class_ids.tolist(),
            }
        )
    return rows


def _compare_rows(rows_a: Sequence[dict[str, Any]], rows_b: Sequence[dict[str, Any]]) -> dict[str, Any]:
    count_mismatch_frames: list[int] = []
    bbox_diffs: list[np.ndarray] = []
    score_diffs: list[np.ndarray] = []
    class_mismatches = 0
    detections_a = 0
    detections_b = 0

    for row_a, row_b in zip(rows_a, rows_b):
        count_a = int(row_a["count"])
        count_b = int(row_b["count"])
        detections_a += count_a
        detections_b += count_b
        if count_a != count_b:
            count_mismatch_frames.append(int(row_a["frame"]))
            continue
        if count_a == 0:
            continue

        boxes_a = np.asarray(row_a["boxes_norm"], dtype=np.float64)
        boxes_b = np.asarray(row_b["boxes_norm"], dtype=np.float64)
        scores_a = np.asarray(row_a["scores"], dtype=np.float64)
        scores_b = np.asarray(row_b["scores"], dtype=np.float64)
        classes_a = np.asarray(row_a["class_ids"], dtype=np.int64)
        classes_b = np.asarray(row_b["class_ids"], dtype=np.int64)

        bbox_diffs.append(np.abs(boxes_a - boxes_b).reshape(-1))
        score_diffs.append(np.abs(scores_a - scores_b).reshape(-1))
        class_mismatches += int(np.sum(classes_a != classes_b))

    bbox_diff = np.concatenate(bbox_diffs) if bbox_diffs else np.array([], dtype=np.float64)
    score_diff = np.concatenate(score_diffs) if score_diffs else np.array([], dtype=np.float64)
    frames_compared = len(rows_a)
    return {
        "frames_compared": int(frames_compared),
        "detections_a": int(detections_a),
        "detections_b": int(detections_b),
        "count_mismatch_frames": int(len(count_mismatch_frames)),
        "count_exact_match_fraction": (
            float(1.0 - (len(count_mismatch_frames) / frames_compared))
            if frames_compared
            else 0.0
        ),
        "first_count_mismatch_frames": count_mismatch_frames[:20],
        "bbox_abs_diff_max": float(np.max(bbox_diff)) if bbox_diff.size else 0.0,
        "bbox_abs_diff_mean": float(np.mean(bbox_diff)) if bbox_diff.size else 0.0,
        "score_abs_diff_max": float(np.max(score_diff)) if score_diff.size else 0.0,
        "score_abs_diff_mean": float(np.mean(score_diff)) if score_diff.size else 0.0,
        "class_mismatches": int(class_mismatches),
    }


def _run_backend(
    *,
    video_path: Path,
    backend: str,
    frames: Sequence[int],
    batch_size: int,
    model: Any,
    device: torch.device,
    dtype: torch.dtype,
    resize: tuple[int, int] | None,
    predict_kwargs: dict[str, Any],
) -> dict[str, Any]:
    processed, decode_payload = _decode_backend_frames(
        video_path=video_path,
        backend=backend,
        frames=frames,
        batch_size=batch_size,
        device=device,
        dtype=dtype,
        resize=resize,
    )
    inference_height = int(processed.shape[2])
    inference_width = int(processed.shape[3])
    t0 = time.perf_counter()
    with torch.inference_mode():
        predictions = model.predict(processed, **predict_kwargs)
    if device.type == "cuda":
        torch.cuda.synchronize()
    inference_seconds = time.perf_counter() - t0
    rows = _prediction_rows(
        predictions,
        frames=frames,
        inference_height=inference_height,
        inference_width=inference_width,
    )
    return {
        "backend": backend,
        "frames": [int(frame) for frame in frames],
        "tensor_shape": [int(v) for v in processed.shape],
        "tensor_device": str(processed.device),
        "tensor_dtype": str(processed.dtype),
        "decode": decode_payload,
        "inference_seconds": float(inference_seconds),
        "predictions": rows,
    }


def run_comparison(args: argparse.Namespace) -> dict[str, Any]:
    video_path = args.video_path.expanduser().resolve()
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    config = stage._load_config(args.config.expanduser().resolve() if args.config else None)
    model_path = stage._resolve_model_path(args.model, config)
    resize, resize_source = _resolve_smoke_resize(args.resize, config)
    imgsz_applied = _resize_to_imgsz(resize)
    frames = _selected_frames(args.frames)

    if args.device == "auto":
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device_str = args.device
    if device_str == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA but torch.cuda.is_available() is false")
    device = torch.device(device_str)
    use_fp16 = device.type == "cuda" and not args.force_fp32
    dtype = torch.float16 if use_fp16 else torch.float32

    detect_cfg = config.get("detection") or {}
    conf = float(args.conf if args.conf is not None else detect_cfg.get("conf_threshold", 0.40))
    iou = float(args.iou if args.iou is not None else detect_cfg.get("iou_threshold", 0.45))
    max_det = int(args.max_det if args.max_det is not None else detect_cfg.get("max_det", 20))

    model = stage.YOLO(str(model_path))
    try:
        model.fuse()
    except Exception:
        pass
    model.to(device_str)
    model_optimization = _apply_model_runtime_optimizations(model, device)
    if use_fp16:
        model.half()
    if device.type == "cuda":
        torch.cuda.synchronize()

    predict_kwargs = _build_predict_kwargs(
        conf=conf,
        iou=iou,
        max_det=max_det,
        device_str=device_str,
        use_fp16=use_fp16,
        imgsz_applied=imgsz_applied,
    )

    backend_a = _run_backend(
        video_path=video_path,
        backend=args.backend_a,
        frames=frames,
        batch_size=int(args.batch_size),
        model=model,
        device=device,
        dtype=dtype,
        resize=resize,
        predict_kwargs=predict_kwargs,
    )
    backend_b = _run_backend(
        video_path=video_path,
        backend=args.backend_b,
        frames=frames,
        batch_size=int(args.batch_size),
        model=model,
        device=device,
        dtype=dtype,
        resize=resize,
        predict_kwargs=predict_kwargs,
    )
    comparison = _compare_rows(backend_a["predictions"], backend_b["predictions"])
    failed = False
    if args.fail_on_count_mismatch and comparison["count_mismatch_frames"]:
        failed = True
    if args.max_bbox_diff is not None and comparison["bbox_abs_diff_max"] > float(args.max_bbox_diff):
        failed = True
    if args.max_score_diff is not None and comparison["score_abs_diff_max"] > float(args.max_score_diff):
        failed = True

    return {
        "status": "failed" if failed else "ok",
        "canonical_outputs_written": False,
        "video_path": str(video_path),
        "model_path": str(model_path),
        "frames": frames,
        "backend_a": args.backend_a,
        "backend_b": args.backend_b,
        "resize": [int(resize[0]), int(resize[1])] if resize else None,
        "resize_source": resize_source,
        "imgsz_applied": imgsz_applied,
        "device": device_str,
        "fp16": bool(use_fp16),
        "conf_threshold": conf,
        "iou_threshold": iou,
        "max_det": max_det,
        "model_optimization": model_optimization,
        "comparison": comparison,
        "backend_results": {
            "a": backend_a,
            "b": backend_b,
        },
    }


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(argv)
    payload = run_comparison(args)
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.output_json is not None:
        output_json = args.output_json.expanduser()
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(text + "\n", encoding="utf-8")
        print(f"Wrote backend parity JSON: {output_json}")
    print(text)
    return 0 if payload["status"] == "ok" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
