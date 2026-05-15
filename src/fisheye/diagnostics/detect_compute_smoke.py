#!/usr/bin/env python3
"""
Compute-only detection smoke for cluster migration.

This command verifies that a compute node can open a video, decode a bounded
frame batch, preprocess it, load a YOLO model, and run inference. It
intentionally does not create detect_runs or write canonical Zarr outputs.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence

import torch
import torch.nn.functional as F


def _default_cache_root() -> Path:
    explicit = os.environ.get("PALETTE_JOB_CACHE")
    if explicit:
        return Path(explicit).expanduser()

    scratch_user = os.environ.get("USER")
    lsf_job_id = os.environ.get("LSB_JOBID")
    if scratch_user and lsf_job_id:
        return Path("/scratch") / scratch_user / lsf_job_id / "palette_cache"

    xdg_cache = os.environ.get("XDG_CACHE_HOME")
    if xdg_cache:
        return Path(xdg_cache).expanduser() / "palette"

    return Path(tempfile.gettempdir()) / f"palette-{scratch_user or 'unknown'}-cache"


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


BACKEND_PYNVVC_LUMA_RGB = "pynvvc_luma_rgb"
BACKEND_CHOICES = (*stage.BACKEND_CHOICES, BACKEND_PYNVVC_LUMA_RGB)


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


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a bounded detection compute smoke without writing canonical "
            "detect_runs or Zarr chunks."
        )
    )
    parser.add_argument("video_path", type=Path, help="Input video path.")
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="YOLO model path (.pt). Required unless config model.path is set.",
    )
    parser.add_argument("--config", type=Path, default=None, help="Optional YAML config path.")
    parser.add_argument(
        "--decode-backend",
        choices=BACKEND_CHOICES,
        default=stage.BACKEND_DECORD_GPU,
        help="Decode backend to smoke.",
    )
    parser.add_argument("--start-frame", type=_non_negative_int, default=0)
    parser.add_argument(
        "--max-frames",
        type=_non_negative_int,
        default=0,
        help="Maximum frames to process. 0 means use max-batches * batch-size.",
    )
    parser.add_argument(
        "--max-batches",
        type=_non_negative_int,
        default=1,
        help="Maximum batches to process. 0 means use max-frames only.",
    )
    parser.add_argument("--batch-size", type=_positive_int, default=4)
    parser.add_argument(
        "--resize",
        type=int,
        nargs=2,
        default=None,
        metavar=("WIDTH", "HEIGHT"),
        help=(
            "Resize before inference. Defaults to config video.resize, then "
            "detection.resize_dims when present."
        ),
    )
    parser.add_argument("--conf", type=float, default=None, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=None, help="IoU threshold.")
    parser.add_argument("--max-det", type=int, default=None, help="Max detections per frame.")
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
        help="Output JSON path. Defaults to runs/diagnostics/detect_compute_smoke_<timestamp>.json.",
    )
    return parser.parse_args(argv)


def _resolve_frame_end(start_frame: int, max_frames: int, max_batches: int, batch_size: int) -> int:
    if max_frames == 0 and max_batches == 0:
        raise ValueError(
            "Compute smoke must be bounded; set --max-frames or --max-batches."
        )

    limits: list[int] = []
    if max_frames > 0:
        limits.append(start_frame + max_frames)
    if max_batches > 0:
        limits.append(start_frame + max_batches * batch_size)
    frame_end = min(limits)
    if frame_end <= start_frame:
        raise ValueError("No frames selected after applying frame limits.")
    return frame_end


def _decoded_count(decoded: Any, backend: str) -> int:
    if decoded is None:
        return 0
    if backend == stage.BACKEND_OPENCV:
        return int(len(decoded))
    return int(decoded.shape[0])


class _PynvvcLumaRgbReader:
    """Sequential PyNvVideoCodec reader that emits raw NV12 CUDA tensors."""

    def __init__(self, video_path: Path, *, start_frame: int, gpu_id: int = 0) -> None:
        if start_frame != 0:
            raise ValueError(
                f"{BACKEND_PYNVVC_LUMA_RGB} is sequential-only in this smoke; "
                "--start-frame must be 0."
            )
        try:
            import PyNvVideoCodec as nvc  # type: ignore
        except Exception as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                f"PyNvVideoCodec import failed; cannot use {BACKEND_PYNVVC_LUMA_RGB}: {exc}"
            ) from exc

        self.nvc = nvc
        self.demuxer = nvc.CreateDemuxer(filename=str(video_path))
        self.decoder = nvc.CreateDecoder(
            gpuid=int(gpu_id),
            codec=self.demuxer.GetNvCodecId(),
            usedevicememory=True,
        )
        self.packet_iter = iter(self.demuxer)
        self.source_height = int(self.demuxer.Height())
        self.source_width = int(self.demuxer.Width())
        self.codec = str(self.demuxer.GetNvCodecId())
        self.frame_rate = float(self.demuxer.FrameRate())
        self._eof = False
        self._pending_frames: list[torch.Tensor] = []

    def decode_next(self, count: int) -> list[torch.Tensor]:
        frames: list[torch.Tensor] = []
        if self._pending_frames:
            take = min(count, len(self._pending_frames))
            frames.extend(self._pending_frames[:take])
            self._pending_frames = self._pending_frames[take:]
        while len(frames) < count and not self._eof:
            try:
                packet = next(self.packet_iter)
            except StopIteration:
                self._eof = True
                break
            for frame in self.decoder.Decode(packet):
                tensor = torch.from_dlpack(frame)
                if len(frames) < count:
                    frames.append(tensor)
                else:
                    self._pending_frames.append(tensor)
        return frames

    def close(self) -> None:
        self._pending_frames = []
        self.packet_iter = iter(())
        del self.decoder
        del self.demuxer


def _preprocess_pynvvc_luma_rgb(
    raw_frames: Sequence[torch.Tensor],
    *,
    source_height: int,
    device: torch.device,
    dtype: torch.dtype,
    resize: Optional[tuple[int, int]],
) -> torch.Tensor:
    if resize is None:
        raise ValueError(f"{BACKEND_PYNVVC_LUMA_RGB} requires a resolved resize.")

    width, height = int(resize[0]), int(resize[1])
    y_planes = [frame[:source_height, :].contiguous() for frame in raw_frames]
    luma = torch.stack(y_planes, dim=0).unsqueeze(1).to(
        device=device,
        dtype=dtype,
        non_blocking=True,
    )
    resized = F.interpolate(
        luma,
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    )
    rgb = resized.expand(-1, 3, -1, -1).mul(1.0 / 255.0)
    return rgb.contiguous(memory_format=torch.channels_last)


def _release_reader_info(reader_info: Dict[str, Any]) -> None:
    if reader_info["backend"] == BACKEND_PYNVVC_LUMA_RGB:
        reader_info["reader"].close()
        return
    stage._release_reader(reader_info)


def _count_batches(frame_start: int, frame_end: int, batch_size: int) -> int:
    total = max(0, frame_end - frame_start)
    return int((total + batch_size - 1) // batch_size)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _collect_cluster_context() -> Dict[str, Any]:
    keys = (
        "LSB_JOBID",
        "LSB_JOBNAME",
        "LSB_QUEUE",
        "LSB_DJOB_NUMPROC",
        "LSB_HOSTS",
        "HOSTNAME",
        "CUDA_VISIBLE_DEVICES",
        "PALETTE_JOB_CACHE",
        "YOLO_CONFIG_DIR",
        "MPLBACKEND",
        "TMPDIR",
        "USER",
    )
    payload: Dict[str, Any] = {key: os.environ.get(key) for key in keys}
    user = os.environ.get("USER")
    job_id = os.environ.get("LSB_JOBID")
    payload["scratch_job_dir"] = (
        str(Path("/scratch") / user / job_id) if user and job_id else None
    )
    return payload


def _start_span(stages: Dict[str, Any], name: str) -> float:
    stage_payload = stages.setdefault(name, {})
    stage_payload["start_utc"] = _now_iso()
    return time.perf_counter()


def _end_span(stages: Dict[str, Any], name: str, start: float) -> float:
    elapsed = time.perf_counter() - start
    stage_payload = stages.setdefault(name, {})
    stage_payload["end_utc"] = _now_iso()
    stage_payload["seconds"] = float(elapsed)
    return elapsed


def _split_first_and_steady_state(
    batches: Sequence[Dict[str, Any]],
    field: str,
) -> Dict[str, Any]:
    values = [float(batch[field]) for batch in batches if field in batch]
    first = values[0] if values else None
    steady = values[1:]
    return {
        "first_batch_seconds": first,
        "steady_state_batches": len(steady),
        "steady_state_seconds_total": float(sum(steady)),
        "steady_state_seconds_mean": (
            float(sum(steady) / len(steady)) if steady else None
        ),
    }


def _resolve_smoke_resize(
    resize_arg: Optional[Sequence[int]],
    config: Dict[str, Any],
) -> tuple[Optional[tuple[int, int]], str]:
    """Resolve preprocessing resize as (width, height)."""

    if resize_arg is not None:
        if len(resize_arg) != 2:
            raise ValueError("--resize requires WIDTH HEIGHT")
        return (int(resize_arg[0]), int(resize_arg[1])), "cli_resize"

    video_resize = (config.get("video") or {}).get("resize")
    if isinstance(video_resize, (list, tuple)) and len(video_resize) == 2:
        return (int(video_resize[0]), int(video_resize[1])), "config_video_resize"

    detection_resize = (config.get("detection") or {}).get("resize_dims")
    if isinstance(detection_resize, (list, tuple)) and len(detection_resize) == 2:
        # Config uses [height, width]; preprocessing helper expects [width, height].
        height, width = int(detection_resize[0]), int(detection_resize[1])
        return (width, height), "config_detection_resize_dims"

    return None, "none"


def _resize_to_imgsz(resize: Optional[tuple[int, int]]) -> Optional[int | list[int]]:
    """Convert preprocessing resize (width, height) to Ultralytics imgsz."""

    if resize is None:
        return None
    width, height = int(resize[0]), int(resize[1])
    if width == height:
        return height
    return [height, width]


def _apply_model_runtime_optimizations(model: Any, device: torch.device) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "cudnn_benchmark_enabled": None,
        "model_channels_last": False,
        "model_channels_last_error": None,
    }
    if device.type != "cuda":
        return payload

    torch.backends.cudnn.benchmark = True
    payload["cudnn_benchmark_enabled"] = bool(torch.backends.cudnn.benchmark)

    inner_model = getattr(model, "model", None)
    if inner_model is None:
        return payload
    try:
        model.model = inner_model.to(memory_format=torch.channels_last)
        payload["model_channels_last"] = True
    except Exception as exc:  # pragma: no cover - model implementation dependent
        payload["model_channels_last_error"] = str(exc)
    return payload


def _default_output_json() -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return Path("runs/diagnostics") / f"detect_compute_smoke_{timestamp}.json"


def run_smoke(args: argparse.Namespace) -> Dict[str, Any]:
    video_path = args.video_path.expanduser().resolve()
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    config = stage._load_config(args.config.expanduser().resolve() if args.config else None)
    model_path = stage._resolve_model_path(args.model, config)
    resize, resize_source = _resolve_smoke_resize(args.resize, config)
    imgsz_applied = _resize_to_imgsz(resize)

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
    if args.decode_backend == BACKEND_PYNVVC_LUMA_RGB:
        if device.type != "cuda":
            raise RuntimeError(f"{BACKEND_PYNVVC_LUMA_RGB} requires CUDA device inference.")
        if resize is None:
            raise RuntimeError(
                f"{BACKEND_PYNVVC_LUMA_RGB} requires detection.resize_dims or --resize."
            )

    frame_start = int(args.start_frame)
    frame_end = _resolve_frame_end(
        start_frame=frame_start,
        max_frames=int(args.max_frames),
        max_batches=int(args.max_batches),
        batch_size=int(args.batch_size),
    )

    payload: Dict[str, Any] = {
        "schema_version": 1,
        "created_at_utc": stage._utc_now_iso(),
        "status": "ok",
        "canonical_outputs_written": False,
        "canonical_zarr_write_policy": "compute_only_no_detect_runs_or_zarr_chunks",
        "inputs": {
            "video_path": str(video_path),
            "video_size_bytes": int(video_path.stat().st_size),
            "model_path": str(model_path),
            "decode_backend": args.decode_backend,
            "start_frame": frame_start,
            "end_frame_exclusive": frame_end,
            "frames_requested": int(frame_end - frame_start),
            "batch_size": int(args.batch_size),
            "batches_requested": _count_batches(frame_start, frame_end, int(args.batch_size)),
            "resize": [int(resize[0]), int(resize[1])] if resize else None,
            "resize_source": resize_source,
            "imgsz_applied": imgsz_applied,
            "conf_threshold": float(conf),
            "iou_threshold": float(iou),
            "max_det": int(max_det),
            "device": device_str,
            "fp16": bool(use_fp16),
        },
        "environment": stage._collect_environment(),
        "cluster": _collect_cluster_context(),
        "stage_spans": {},
        "stages": {},
        "batches": [],
    }

    t_total = time.perf_counter()
    total_start_utc = _now_iso()
    payload["stage_spans"]["total"] = {"start_utc": total_start_utc}

    print("Detection compute smoke:")
    print(
        f"- video={video_path} frames=[{frame_start}, {frame_end}) "
        f"batch={args.batch_size} backend={args.decode_backend}"
    )
    print(
        f"- model={model_path} device={device_str} fp16={use_fp16} "
        f"resize={resize if resize else 'None'}"
    )
    print("- canonical Zarr writes: disabled")

    t0 = _start_span(payload["stage_spans"], "model_load")
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
    model_load_seconds = _end_span(payload["stage_spans"], "model_load", t0)
    payload["stages"]["model_load_seconds"] = float(model_load_seconds)
    payload["model_optimization"] = model_optimization

    t0 = _start_span(payload["stage_spans"], "video_open")
    if args.decode_backend == BACKEND_PYNVVC_LUMA_RGB:
        pynvvc_reader = _PynvvcLumaRgbReader(video_path, start_frame=frame_start, gpu_id=0)
        reader_info = {"backend": BACKEND_PYNVVC_LUMA_RGB, "reader": pynvvc_reader}
        payload["inputs"]["pynvvc"] = {
            "codec": pynvvc_reader.codec,
            "frame_rate": pynvvc_reader.frame_rate,
            "source_width": pynvvc_reader.source_width,
            "source_height": pynvvc_reader.source_height,
            "preprocess_mode": "luma_rgb",
        }
    else:
        reader_info = stage._resolve_backend_reader(video_path, args.decode_backend, frame_start)
    video_open_seconds = _end_span(payload["stage_spans"], "video_open", t0)
    payload["stages"]["video_open_seconds"] = float(video_open_seconds)

    frames_processed = 0
    detections_total = 0
    decode_seconds_total = 0.0
    preprocess_seconds_total = 0.0
    inference_seconds_total = 0.0
    predict_return_seconds_total = 0.0
    inference_cuda_sync_seconds_total = 0.0

    try:
        for batch_start in range(frame_start, frame_end, int(args.batch_size)):
            batch_end = min(batch_start + int(args.batch_size), frame_end)
            indices = list(range(batch_start, batch_end))

            t_decode = time.perf_counter()
            if args.decode_backend == BACKEND_PYNVVC_LUMA_RGB:
                decoded = reader_info["reader"].decode_next(len(indices))
            else:
                decoded = stage._decode_batch(reader_info, indices)
            if (
                args.decode_backend in {stage.BACKEND_DECORD_GPU, BACKEND_PYNVVC_LUMA_RGB}
                and torch.cuda.is_available()
            ):
                torch.cuda.synchronize()
            decode_seconds = time.perf_counter() - t_decode
            if args.decode_backend == BACKEND_PYNVVC_LUMA_RGB:
                actual_count = len(decoded)
            else:
                actual_count = _decoded_count(decoded, args.decode_backend)
            if actual_count <= 0:
                break

            t_preprocess = time.perf_counter()
            if args.decode_backend == BACKEND_PYNVVC_LUMA_RGB:
                processed = _preprocess_pynvvc_luma_rgb(
                    decoded,
                    source_height=reader_info["reader"].source_height,
                    device=device,
                    dtype=dtype,
                    resize=resize,
                )
            else:
                processed = stage._preprocess_batch(
                    decoded=decoded,
                    backend=args.decode_backend,
                    device=device,
                    dtype=dtype,
                    resize=resize,
                )
            if device.type == "cuda":
                torch.cuda.synchronize()
            preprocess_seconds = time.perf_counter() - t_preprocess

            predict_kwargs = {
                "conf": conf,
                "iou": iou,
                "max_det": max_det,
                "verbose": False,
                "device": device_str,
                "half": use_fp16,
            }
            if imgsz_applied is not None:
                predict_kwargs["imgsz"] = imgsz_applied

            t_inference = time.perf_counter()
            with torch.inference_mode():
                predictions = model.predict(processed, **predict_kwargs)
            predict_return_seconds = time.perf_counter() - t_inference
            t_sync = time.perf_counter()
            if device.type == "cuda":
                torch.cuda.synchronize()
            inference_cuda_sync_seconds = time.perf_counter() - t_sync
            inference_seconds = time.perf_counter() - t_inference
            detections = stage._count_detections(predictions)

            frames_processed += actual_count
            detections_total += detections
            decode_seconds_total += decode_seconds
            preprocess_seconds_total += preprocess_seconds
            inference_seconds_total += inference_seconds
            predict_return_seconds_total += predict_return_seconds
            inference_cuda_sync_seconds_total += inference_cuda_sync_seconds

            batch_payload = {
                "batch_index": len(payload["batches"]),
                "frame_start": int(batch_start),
                "frame_end_exclusive": int(batch_start + actual_count),
                "frames_processed": int(actual_count),
                "decode_seconds": float(decode_seconds),
                "preprocess_seconds": float(preprocess_seconds),
                "inference_seconds": float(inference_seconds),
                "predict_return_seconds": float(predict_return_seconds),
                "inference_cuda_sync_seconds": float(inference_cuda_sync_seconds),
                "detections_total": int(detections),
                "tensor_shape": [int(v) for v in processed.shape],
                "tensor_device": str(processed.device),
                "tensor_dtype": str(processed.dtype),
            }
            payload["batches"].append(batch_payload)
            print(
                "  batch {idx}: frames={frames} detections={detections} "
                "decode={decode:.3f}s preprocess={prep:.3f}s inference={infer:.3f}s".format(
                    idx=batch_payload["batch_index"],
                    frames=actual_count,
                    detections=detections,
                    decode=decode_seconds,
                    prep=preprocess_seconds,
                    infer=inference_seconds,
                )
            )

            del predictions
            del processed
            del decoded
    finally:
        _release_reader_info(reader_info)

    total_seconds = time.perf_counter() - t_total
    total_end_utc = _now_iso()
    payload["stage_spans"]["total"]["end_utc"] = total_end_utc
    payload["stage_spans"]["total"]["seconds"] = float(total_seconds)
    batch_rows = payload["batches"]
    decode_split = _split_first_and_steady_state(batch_rows, "decode_seconds")
    preprocess_split = _split_first_and_steady_state(batch_rows, "preprocess_seconds")
    inference_split = _split_first_and_steady_state(batch_rows, "inference_seconds")
    predict_return_split = _split_first_and_steady_state(batch_rows, "predict_return_seconds")
    inference_sync_split = _split_first_and_steady_state(batch_rows, "inference_cuda_sync_seconds")
    frames_excluding_first = sum(
        int(batch.get("frames_processed", 0)) for batch in batch_rows[1:]
    )
    steady_inference_total = float(inference_split["steady_state_seconds_total"])
    payload["summary"] = {
        "frames_processed": int(frames_processed),
        "batches_processed": int(len(payload["batches"])),
        "detections_total": int(detections_total),
        "decode_seconds_total": float(decode_seconds_total),
        "preprocess_seconds_total": float(preprocess_seconds_total),
        "inference_seconds_total": float(inference_seconds_total),
        "predict_return_seconds_total": float(predict_return_seconds_total),
        "inference_cuda_sync_seconds_total": float(inference_cuda_sync_seconds_total),
        "total_seconds": float(total_seconds),
        "end_to_end_fps": float(frames_processed / total_seconds) if total_seconds > 0 else 0.0,
        "inference_fps": (
            float(frames_processed / inference_seconds_total)
            if inference_seconds_total > 0
            else 0.0
        ),
        "first_batch": {
            "decode_seconds": decode_split["first_batch_seconds"],
            "preprocess_seconds": preprocess_split["first_batch_seconds"],
            "inference_seconds": inference_split["first_batch_seconds"],
            "predict_return_seconds": predict_return_split["first_batch_seconds"],
            "inference_cuda_sync_seconds": inference_sync_split["first_batch_seconds"],
        },
        "steady_state_excluding_first_batch": {
            "batches_processed": int(len(batch_rows[1:])),
            "frames_processed": int(frames_excluding_first),
            "decode_seconds_total": decode_split["steady_state_seconds_total"],
            "decode_seconds_mean": decode_split["steady_state_seconds_mean"],
            "preprocess_seconds_total": preprocess_split["steady_state_seconds_total"],
            "preprocess_seconds_mean": preprocess_split["steady_state_seconds_mean"],
            "inference_seconds_total": steady_inference_total,
            "inference_seconds_mean": inference_split["steady_state_seconds_mean"],
            "predict_return_seconds_total": predict_return_split["steady_state_seconds_total"],
            "predict_return_seconds_mean": predict_return_split["steady_state_seconds_mean"],
            "inference_cuda_sync_seconds_total": inference_sync_split["steady_state_seconds_total"],
            "inference_cuda_sync_seconds_mean": inference_sync_split["steady_state_seconds_mean"],
            "inference_fps": (
                float(frames_excluding_first / steady_inference_total)
                if steady_inference_total > 0
                else None
            ),
        },
    }

    if device.type == "cuda":
        torch.cuda.empty_cache()

    print(
        "- summary: frames={frames} detections={detections} total={total:.3f}s "
        "end_to_end_fps={fps:.2f}".format(
            frames=frames_processed,
            detections=detections_total,
            total=total_seconds,
            fps=payload["summary"]["end_to_end_fps"],
        )
    )
    return payload


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    payload = run_smoke(args)

    output_json = args.output_json if args.output_json is not None else _default_output_json()
    output_json = output_json.expanduser()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote compute smoke JSON: {output_json}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
