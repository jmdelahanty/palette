#!/usr/bin/env python3
"""
Benchmark split detection stages: decode+preprocess vs inference-only.

This script is designed as the next step after decode-only backend benchmarking.
It measures:
  1) decode + preprocess time per batch
  2) inference-only time per batch on cached preprocessed tensors
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

os.environ.setdefault("DECORD_EOF_RETRY_MAX", "65536")

# Keep Decord import before OpenCV to avoid FFmpeg symbol conflicts.
try:
    import decord  # type: ignore
    from decord import VideoReader, cpu, gpu  # type: ignore
    _DECORD_IMPORT_ERROR: Optional[Exception] = None
except Exception as exc:  # pragma: no cover - environment dependent
    decord = None  # type: ignore[assignment]
    VideoReader = None  # type: ignore[assignment]
    cpu = None  # type: ignore[assignment]
    gpu = None  # type: ignore[assignment]
    _DECORD_IMPORT_ERROR = exc

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from ultralytics import YOLO

BACKEND_DECORD_GPU = "decord_gpu"
BACKEND_DECORD_CPU = "decord_cpu"
BACKEND_OPENCV = "opencv"
BACKEND_CHOICES = (BACKEND_DECORD_GPU, BACKEND_DECORD_CPU, BACKEND_OPENCV)


class BackendUnavailable(RuntimeError):
    """Raised when a decode backend cannot run in current environment."""


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _series_stats(values: Sequence[float]) -> Dict[str, Optional[float]]:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p90": None,
            "min": None,
            "max": None,
        }
    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def _default_config_path() -> Optional[Path]:
    candidates = [
        Path("yolo_detect_config.yaml"),
        Path("configs/fisheye/yolo_detect_config.yaml"),
        Path(__file__).resolve().parents[3] / "configs/fisheye/yolo_detect_config.yaml",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def _load_config(config_path: Optional[Path]) -> Dict[str, Any]:
    target = config_path if config_path is not None else _default_config_path()
    if target is None:
        return {}
    with target.open("r", encoding="utf-8") as fh:
        loaded = yaml.safe_load(fh) or {}
    if not isinstance(loaded, dict):
        return {}
    return loaded


def _video_meta(video_path: Path) -> Dict[str, Any]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        cap.release()
    return {
        "total_frames": total_frames,
        "fps": fps,
        "width": width,
        "height": height,
    }


def _resolve_window(
    total_frames: int,
    start_frame: int,
    max_frames: int,
    max_batches: int,
    batch_size: int,
) -> Tuple[int, int]:
    if start_frame < 0:
        raise ValueError("--start-frame must be >= 0")
    if batch_size <= 0:
        raise ValueError("--batch-size must be > 0")
    if max_frames < 0:
        raise ValueError("--max-frames must be >= 0")
    if max_batches < 0:
        raise ValueError("--max-batches must be >= 0")
    if start_frame >= total_frames:
        raise ValueError(
            f"start frame {start_frame} beyond total frames {total_frames}"
        )

    limits: List[int] = [total_frames]
    if max_frames > 0:
        limits.append(start_frame + max_frames)
    if max_batches > 0:
        limits.append(start_frame + max_batches * batch_size)
    end_frame = min(limits)
    if end_frame <= start_frame:
        raise ValueError("No frames selected after applying frame/batch limits")
    return start_frame, end_frame


def _collect_environment() -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "hostname": platform.node(),
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "decord_imported": bool(decord is not None),
        "decord_import_error": str(_DECORD_IMPORT_ERROR) if _DECORD_IMPORT_ERROR else None,
    }
    if torch.cuda.is_available():
        payload["cuda_device_count"] = int(torch.cuda.device_count())
        if torch.cuda.device_count() > 0:
            payload["cuda_device_name"] = torch.cuda.get_device_name(0)
    if decord is not None:
        payload["decord_version"] = getattr(decord, "__version__", None)
    return payload


def _resolve_model_path(model_arg: Optional[Path], config: Dict[str, Any]) -> Path:
    if model_arg is not None:
        model_path = model_arg.expanduser().resolve()
    else:
        model_cfg = (config.get("model") or {}).get("path")
        if not model_cfg:
            raise ValueError(
                "Model path required via --model or config model.path"
            )
        model_path = Path(model_cfg).expanduser().resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    return model_path


def _resolve_resize(
    resize_arg: Optional[Sequence[int]],
    config: Dict[str, Any],
) -> Optional[Tuple[int, int]]:
    if resize_arg is not None:
        if len(resize_arg) != 2:
            raise ValueError("--resize requires WIDTH HEIGHT")
        return int(resize_arg[0]), int(resize_arg[1])
    cfg_resize = (config.get("video") or {}).get("resize")
    if cfg_resize is None:
        return None
    if not isinstance(cfg_resize, (list, tuple)) or len(cfg_resize) != 2:
        return None
    return int(cfg_resize[0]), int(cfg_resize[1])


def _resolve_backend_reader(video_path: Path, backend: str, start_frame: int):
    if backend == BACKEND_OPENCV:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise BackendUnavailable(f"OpenCV cannot open video: {video_path}")
        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, float(start_frame))
        return {"backend": backend, "reader": cap}

    if decord is None or VideoReader is None or cpu is None:
        if _DECORD_IMPORT_ERROR:
            raise BackendUnavailable(f"Decord import failed: {_DECORD_IMPORT_ERROR}")
        raise BackendUnavailable("Decord unavailable")

    if backend == BACKEND_DECORD_GPU:
        if gpu is None:
            raise BackendUnavailable("Decord GPU context unavailable")
        if not torch.cuda.is_available():
            raise BackendUnavailable("CUDA unavailable for decord_gpu backend")
        decord.bridge.set_bridge("torch")
        vr = VideoReader(str(video_path), ctx=gpu(0))
        return {"backend": backend, "reader": vr}

    if backend == BACKEND_DECORD_CPU:
        decord.bridge.set_bridge("native")
        vr = VideoReader(str(video_path), ctx=cpu())
        return {"backend": backend, "reader": vr}

    raise ValueError(f"Unsupported backend: {backend}")


def _release_reader(reader_info: Dict[str, Any]) -> None:
    backend = reader_info["backend"]
    reader = reader_info["reader"]
    if backend == BACKEND_OPENCV:
        reader.release()
    else:
        del reader


def _decode_batch(
    reader_info: Dict[str, Any],
    indices: Sequence[int],
) -> Any:
    backend = reader_info["backend"]
    reader = reader_info["reader"]
    if backend == BACKEND_OPENCV:
        frames: List[np.ndarray] = []
        for _ in indices:
            ok, frame = reader.read()
            if not ok:
                break
            frames.append(frame)
        if not frames:
            return None
        return frames
    return reader.get_batch(list(indices))


def _preprocess_batch(
    decoded: Any,
    backend: str,
    device: torch.device,
    dtype: torch.dtype,
    resize: Optional[Tuple[int, int]],
) -> torch.Tensor:
    if backend == BACKEND_DECORD_GPU:
        # decoded is torch [B, H, W, C] uint8 on device.
        assert isinstance(decoded, torch.Tensor)
        frames_chw = decoded.permute(0, 3, 1, 2).contiguous()
        tensor = frames_chw.to(
            device=device,
            dtype=dtype,
            non_blocking=True,
        ).contiguous(memory_format=torch.channels_last)
    elif backend == BACKEND_DECORD_CPU:
        if hasattr(decoded, "asnumpy"):
            frames_nd = decoded.asnumpy()
        else:
            frames_nd = np.asarray(decoded)
        tensor = torch.from_numpy(frames_nd).permute(0, 3, 1, 2)
        tensor = tensor.to(device=device, dtype=dtype, non_blocking=True).contiguous(
            memory_format=torch.channels_last
        )
    elif backend == BACKEND_OPENCV:
        assert isinstance(decoded, list)
        rgb_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in decoded]
        frames_nd = np.stack(rgb_frames, axis=0)
        tensor = torch.from_numpy(frames_nd).permute(0, 3, 1, 2)
        tensor = tensor.to(device=device, dtype=dtype, non_blocking=True).contiguous(
            memory_format=torch.channels_last
        )
    else:
        raise ValueError(f"Unsupported backend: {backend}")

    if resize is not None:
        width, height = resize
        tensor = F.interpolate(
            tensor,
            size=(int(height), int(width)),
            mode="bilinear",
            align_corners=False,
        )
    tensor = tensor.mul_(1.0 / 255.0)
    return tensor


def _run_decode_preprocess_once(
    *,
    video_path: Path,
    backend: str,
    frame_start: int,
    frame_end: int,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    resize: Optional[Tuple[int, int]],
    cache_batches: bool,
) -> Dict[str, Any]:
    reader_info = _resolve_backend_reader(video_path, backend, frame_start)
    decode_ms: List[float] = []
    preprocess_ms: List[float] = []
    cached_batches: List[torch.Tensor] = []
    frames_processed = 0

    t_run = time.perf_counter()
    try:
        for batch_start in range(frame_start, frame_end, batch_size):
            batch_end = min(batch_start + batch_size, frame_end)
            indices = list(range(batch_start, batch_end))
            t0 = time.perf_counter()
            decoded = _decode_batch(reader_info, indices)
            if backend == BACKEND_DECORD_GPU and torch.cuda.is_available():
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            if decoded is None:
                break

            actual_count: int
            if backend == BACKEND_OPENCV:
                actual_count = len(decoded)
            else:
                actual_count = int(decoded.shape[0])
            if actual_count <= 0:
                break

            t2 = time.perf_counter()
            processed = _preprocess_batch(
                decoded=decoded,
                backend=backend,
                device=device,
                dtype=dtype,
                resize=resize,
            )
            if device.type == "cuda":
                torch.cuda.synchronize()
            t3 = time.perf_counter()

            decode_ms.append((t1 - t0) * 1000.0)
            preprocess_ms.append((t3 - t2) * 1000.0)
            frames_processed += actual_count

            if cache_batches:
                cached_batches.append(processed.detach())
            else:
                del processed

            if decoded is not None:
                del decoded
    finally:
        _release_reader(reader_info)

    duration_s = time.perf_counter() - t_run
    fps = float(frames_processed / duration_s) if duration_s > 0 else 0.0
    return {
        "frames_processed": int(frames_processed),
        "batches_processed": int(len(decode_ms)),
        "duration_seconds": float(duration_s),
        "stage_fps": fps,
        "decode_ms": decode_ms,
        "preprocess_ms": preprocess_ms,
        "total_stage_ms": [float(d + p) for d, p in zip(decode_ms, preprocess_ms)],
        "cached_batches": cached_batches,
    }


def _count_detections(predictions: Sequence[Any]) -> int:
    total = 0
    for pred in predictions:
        boxes = getattr(pred, "boxes", None)
        if boxes is None:
            continue
        try:
            total += int(len(boxes))
        except Exception:
            continue
    return int(total)


def _run_inference_only_once(
    *,
    model: YOLO,
    batches: Sequence[torch.Tensor],
    device_str: str,
    half: bool,
    conf: float,
    iou: float,
    max_det: int,
) -> Dict[str, Any]:
    batch_ms: List[float] = []
    detections_total = 0
    frames_processed = 0
    t_run = time.perf_counter()

    with torch.inference_mode():
        for batch in batches:
            frames_processed += int(batch.shape[0])
            t0 = time.perf_counter()
            predictions = model.predict(
                batch,
                conf=conf,
                iou=iou,
                max_det=max_det,
                verbose=False,
                device=device_str,
                half=half,
            )
            if device_str == "cuda" and torch.cuda.is_available():
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            batch_ms.append((t1 - t0) * 1000.0)
            detections_total += _count_detections(predictions)
            del predictions

    duration_s = time.perf_counter() - t_run
    fps = float(frames_processed / duration_s) if duration_s > 0 else 0.0
    return {
        "frames_processed": int(frames_processed),
        "batches_processed": int(len(batch_ms)),
        "duration_seconds": float(duration_s),
        "inference_fps": fps,
        "inference_ms": batch_ms,
        "detections_total": int(detections_total),
    }


def _summarize_decode_reps(repetitions: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    fps_values = [float(rep["stage_fps"]) for rep in repetitions]
    duration_values = [float(rep["duration_seconds"]) for rep in repetitions]
    decode_ms_values: List[float] = []
    preprocess_ms_values: List[float] = []
    total_ms_values: List[float] = []
    for rep in repetitions:
        decode_ms_values.extend(float(v) for v in rep.get("decode_ms", []))
        preprocess_ms_values.extend(float(v) for v in rep.get("preprocess_ms", []))
        total_ms_values.extend(float(v) for v in rep.get("total_stage_ms", []))
    return {
        "stage_fps": _series_stats(fps_values),
        "duration_seconds": _series_stats(duration_values),
        "decode_ms": _series_stats(decode_ms_values),
        "preprocess_ms": _series_stats(preprocess_ms_values),
        "total_stage_ms": _series_stats(total_ms_values),
    }


def _summarize_inference_reps(repetitions: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    fps_values = [float(rep["inference_fps"]) for rep in repetitions]
    duration_values = [float(rep["duration_seconds"]) for rep in repetitions]
    ms_values: List[float] = []
    detections_values = [float(rep["detections_total"]) for rep in repetitions]
    for rep in repetitions:
        ms_values.extend(float(v) for v in rep.get("inference_ms", []))
    return {
        "inference_fps": _series_stats(fps_values),
        "duration_seconds": _series_stats(duration_values),
        "inference_ms": _series_stats(ms_values),
        "detections_total": _series_stats(detections_values),
    }


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark detection decode+preprocess stage separately from inference stage."
    )
    parser.add_argument("video_path", type=Path, help="Input video path.")
    parser.add_argument("--model", type=Path, default=None, help="YOLO model path (.pt).")
    parser.add_argument("--config", type=Path, default=None, help="Optional YAML config path.")
    parser.add_argument(
        "--decode-backend",
        choices=BACKEND_CHOICES,
        default=BACKEND_DECORD_GPU,
        help="Backend for decode+preprocess stage.",
    )
    parser.add_argument("--start-frame", type=int, default=0, help="Start frame index.")
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Max frames to process (0 = unlimited).",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=0,
        help="Max batches to process (0 = unlimited).",
    )
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size.")
    parser.add_argument(
        "--resize",
        type=int,
        nargs=2,
        default=None,
        metavar=("WIDTH", "HEIGHT"),
        help="Resize before inference (default from config video.resize).",
    )
    parser.add_argument("--conf", type=float, default=None, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=None, help="IoU threshold.")
    parser.add_argument("--max-det", type=int, default=None, help="Max detections per frame.")
    parser.add_argument("--warmup-reps", type=int, default=1, help="Warmup repetitions per stage.")
    parser.add_argument("--timed-reps", type=int, default=3, help="Timed repetitions per stage.")
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="Inference device.",
    )
    parser.add_argument(
        "--force-fp32",
        action="store_true",
        help="Disable FP16 even on CUDA.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Output JSON path (default: runs/benchmarks/detect_stage_split_<timestamp>.json).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = parse_args(argv)
    if args.warmup_reps < 0:
        raise ValueError("--warmup-reps must be >= 0")
    if args.timed_reps <= 0:
        raise ValueError("--timed-reps must be > 0")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0")

    video_path = args.video_path.expanduser().resolve()
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    config = _load_config(args.config.expanduser().resolve() if args.config else None)
    model_path = _resolve_model_path(args.model, config)
    resize = _resolve_resize(args.resize, config)

    detect_cfg = config.get("detection") or {}
    conf = float(args.conf if args.conf is not None else detect_cfg.get("conf_threshold", 0.40))
    iou = float(args.iou if args.iou is not None else detect_cfg.get("iou_threshold", 0.45))
    max_det = int(args.max_det if args.max_det is not None else detect_cfg.get("max_det", 20))

    if args.device == "auto":
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device_str = args.device
    if device_str == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA but torch.cuda.is_available() is false")
    device = torch.device(device_str)
    use_fp16 = device.type == "cuda" and not args.force_fp32
    dtype = torch.float16 if use_fp16 else torch.float32

    meta = _video_meta(video_path)
    frame_start, frame_end = _resolve_window(
        total_frames=int(meta["total_frames"]),
        start_frame=int(args.start_frame),
        max_frames=int(args.max_frames),
        max_batches=int(args.max_batches),
        batch_size=int(args.batch_size),
    )
    frames_target = frame_end - frame_start

    print("Stage-split benchmark fixture:")
    print(
        f"- video={video_path} range=[{frame_start}, {frame_end}) frames={frames_target} "
        f"batch={args.batch_size} backend={args.decode_backend}"
    )
    print(
        f"- model={model_path} device={device_str} fp16={use_fp16} "
        f"resize={resize if resize else 'None'} conf={conf} iou={iou} max_det={max_det}"
    )

    model = YOLO(str(model_path))
    try:
        model.fuse()
    except Exception:
        pass
    model.to(device_str)
    if use_fp16:
        model.half()

    decode_stage: Dict[str, Any] = {
        "backend": args.decode_backend,
        "status": "ok",
        "warmup_repetitions": [],
        "timed_repetitions": [],
    }

    cached_batches: List[torch.Tensor] = []
    try:
        for rep in range(args.warmup_reps):
            run = _run_decode_preprocess_once(
                video_path=video_path,
                backend=args.decode_backend,
                frame_start=frame_start,
                frame_end=frame_end,
                batch_size=args.batch_size,
                device=device,
                dtype=dtype,
                resize=resize,
                cache_batches=False,
            )
            run["repetition_index"] = rep
            run.pop("cached_batches", None)
            decode_stage["warmup_repetitions"].append(run)
            print(
                f"  decode+prep warmup {rep + 1}/{args.warmup_reps}: "
                f"{run['stage_fps']:.2f} fps"
            )

        for rep in range(args.timed_reps):
            should_cache = rep == 0
            run = _run_decode_preprocess_once(
                video_path=video_path,
                backend=args.decode_backend,
                frame_start=frame_start,
                frame_end=frame_end,
                batch_size=args.batch_size,
                device=device,
                dtype=dtype,
                resize=resize,
                cache_batches=should_cache,
            )
            if should_cache:
                cached_batches = run.get("cached_batches", [])
            run["repetition_index"] = rep
            run.pop("cached_batches", None)
            decode_stage["timed_repetitions"].append(run)
            print(
                f"  decode+prep timed {rep + 1}/{args.timed_reps}: "
                f"{run['stage_fps']:.2f} fps"
            )

        decode_stage["summary"] = _summarize_decode_reps(decode_stage["timed_repetitions"])
    except BackendUnavailable as exc:
        decode_stage["status"] = "skipped"
        decode_stage["reason"] = str(exc)
    except Exception as exc:
        decode_stage["status"] = "error"
        decode_stage["reason"] = str(exc)

    if decode_stage["status"] == "ok":
        summ = decode_stage["summary"]
        print(
            "- decode+prep: fps median={:.2f}, p90={:.2f}; decode_ms median={:.2f}; preprocess_ms median={:.2f}".format(
                summ["stage_fps"]["median"] or 0.0,
                summ["stage_fps"]["p90"] or 0.0,
                summ["decode_ms"]["median"] or 0.0,
                summ["preprocess_ms"]["median"] or 0.0,
            )
        )
    else:
        print(f"- decode+prep: {decode_stage['status']} ({decode_stage.get('reason', 'unknown')})")

    inference_stage: Dict[str, Any] = {
        "status": "ok",
        "warmup_repetitions": [],
        "timed_repetitions": [],
        "cache_source": "decode_preprocess timed repetition 0",
    }

    if decode_stage["status"] != "ok" or not cached_batches:
        inference_stage["status"] = "skipped"
        inference_stage["reason"] = "no cached preprocessed batches available"
    else:
        for rep in range(args.warmup_reps):
            run = _run_inference_only_once(
                model=model,
                batches=cached_batches,
                device_str=device_str,
                half=use_fp16,
                conf=conf,
                iou=iou,
                max_det=max_det,
            )
            run["repetition_index"] = rep
            inference_stage["warmup_repetitions"].append(run)
            print(
                f"  inference warmup {rep + 1}/{args.warmup_reps}: "
                f"{run['inference_fps']:.2f} fps"
            )

        for rep in range(args.timed_reps):
            run = _run_inference_only_once(
                model=model,
                batches=cached_batches,
                device_str=device_str,
                half=use_fp16,
                conf=conf,
                iou=iou,
                max_det=max_det,
            )
            run["repetition_index"] = rep
            inference_stage["timed_repetitions"].append(run)
            print(
                f"  inference timed {rep + 1}/{args.timed_reps}: "
                f"{run['inference_fps']:.2f} fps"
            )

        inference_stage["summary"] = _summarize_inference_reps(
            inference_stage["timed_repetitions"]
        )
        summ = inference_stage["summary"]
        print(
            "- inference-only: fps median={:.2f}, p90={:.2f}; inference_ms median={:.2f}, p90={:.2f}".format(
                summ["inference_fps"]["median"] or 0.0,
                summ["inference_fps"]["p90"] or 0.0,
                summ["inference_ms"]["median"] or 0.0,
                summ["inference_ms"]["p90"] or 0.0,
            )
        )

    run_payload: Dict[str, Any] = {
        "schema_version": 1,
        "created_at_utc": _utc_now_iso(),
        "fixture": {
            "video_path": str(video_path),
            "video_metadata": meta,
            "start_frame": int(frame_start),
            "end_frame_exclusive": int(frame_end),
            "frames_target": int(frames_target),
            "batch_size": int(args.batch_size),
            "max_frames_arg": int(args.max_frames),
            "max_batches_arg": int(args.max_batches),
            "decode_backend": args.decode_backend,
            "warmup_repetitions": int(args.warmup_reps),
            "timed_repetitions": int(args.timed_reps),
            "model_path": str(model_path),
            "resize": [int(resize[0]), int(resize[1])] if resize else None,
            "conf_threshold": float(conf),
            "iou_threshold": float(iou),
            "max_det": int(max_det),
            "device": device_str,
            "fp16": bool(use_fp16),
        },
        "environment": _collect_environment(),
        "decode_preprocess_stage": decode_stage,
        "inference_only_stage": inference_stage,
    }

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_json = (
        args.output_json
        if args.output_json is not None
        else Path("runs/benchmarks") / f"detect_stage_split_{timestamp}.json"
    )
    output_json = output_json.expanduser()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(run_payload, indent=2), encoding="utf-8")
    print(f"\nWrote benchmark JSON: {output_json}")

    for batch in cached_batches:
        del batch
    if device.type == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":  # pragma: no cover
    main()
