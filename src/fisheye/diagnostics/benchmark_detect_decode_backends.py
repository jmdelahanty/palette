#!/usr/bin/env python3
"""
Decode-only backend benchmark for detection pipeline planning.

This script intentionally excludes model inference so decode/read performance can
be measured independently. It supports warmup passes, timed repetitions, and
writes a JSON summary suitable for handoff comparisons.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import shlex
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

os.environ.setdefault("DECORD_EOF_RETRY_MAX", "65536")

# Keep Decord import before OpenCV. OpenCV may preload a different FFmpeg
# stack from conda, which can break Decord symbol resolution.
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

try:
    import torch
except Exception:  # pragma: no cover - environment dependent
    torch = None  # type: ignore[assignment]

BACKEND_DECORD_GPU = "decord_gpu"
BACKEND_DECORD_CPU = "decord_cpu"
BACKEND_OPENCV = "opencv"
BACKEND_NATIVE_CMD = "native_cmd"
BACKEND_CHOICES = (
    BACKEND_DECORD_GPU,
    BACKEND_DECORD_CPU,
    BACKEND_OPENCV,
    BACKEND_NATIVE_CMD,
)
DEFAULT_NATIVE_CMD_TEMPLATE = (
    "scripts/py -m fisheye.diagnostics.native_decode_ffmpeg_nvdec_adapter "
    '--video "{video_path}" '
    "--start-frame {start_frame} "
    "--max-frames {max_frames} "
    "--batch-size {batch_size} "
    '--output-json "{output_json}" '
    "--repeat-index {repeat_index} "
    "--phase {phase} "
    "--hwaccel cuda "
    "--hwaccel-output-format cuda"
)


class BackendUnavailable(RuntimeError):
    """Raised when a backend cannot run in the current environment."""


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


def _resolve_native_cmd_template(
    explicit_native_cmd: Optional[str],
    native_backend_requested: bool,
) -> Tuple[Optional[str], str, Optional[str]]:
    if explicit_native_cmd:
        return explicit_native_cmd, "user", None

    if not native_backend_requested:
        return None, "none", None

    adapter_path = Path(__file__).with_name("native_decode_ffmpeg_nvdec_adapter.py")
    if not adapter_path.exists():
        return None, "none", f"Native adapter not found: {adapter_path}"

    return DEFAULT_NATIVE_CMD_TEMPLATE, "auto_default", None


def _parse_numeric_token(token: Any) -> Optional[float]:
    text = str(token).strip()
    if not text:
        return None
    if text in {"-", "N/A", "n/a", "NA", "na", "?"}:
        return None
    try:
        value = float(text)
    except (TypeError, ValueError):
        return None
    if np.isnan(value) or np.isinf(value):
        return None
    return value


def _parse_nvidia_dmon_output(stdout_text: str, gpu_index: int) -> Dict[str, Any]:
    """
    Parse `nvidia-smi dmon` output for one GPU index.

    Expected columns include entries such as: gpu pwr gtemp mtemp sm mem enc dec.
    """
    payload: Dict[str, Any] = {
        "status": "ok",
        "gpu_index": int(gpu_index),
        "sample_count": 0,
        "metrics": {},
    }
    header_tokens: Optional[List[str]] = None
    rows: List[Dict[str, str]] = []

    for raw_line in stdout_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("#"):
            candidate = line.lstrip("#").strip().split()
            if "gpu" in candidate and len(candidate) >= 2:
                header_tokens = candidate
            continue
        if header_tokens is None:
            continue
        tokens = line.split()
        if len(tokens) < len(header_tokens):
            continue
        row = {header_tokens[i]: tokens[i] for i in range(len(header_tokens))}
        gpu_token = row.get("gpu")
        gpu_value = _parse_numeric_token(gpu_token) if gpu_token is not None else None
        if gpu_value is None:
            continue
        if int(gpu_value) != int(gpu_index):
            continue
        rows.append(row)

    if header_tokens is not None:
        payload["header"] = header_tokens
    payload["sample_count"] = int(len(rows))

    if not rows:
        payload["status"] = "no_samples"
        return payload

    numeric_by_key: Dict[str, List[float]] = {}
    for row in rows:
        for key, raw_value in row.items():
            if key == "gpu":
                continue
            value = _parse_numeric_token(raw_value)
            if value is None:
                continue
            numeric_by_key.setdefault(key, []).append(value)

    for key, values in numeric_by_key.items():
        payload["metrics"][key] = _series_stats(values)

    if not payload["metrics"]:
        payload["status"] = "no_numeric_metrics"

    return payload


def _start_nvidia_dmon(
    enabled: bool,
    interval_seconds: int,
    gpu_index: int,
) -> Tuple[Optional[subprocess.Popen], Optional[Dict[str, Any]]]:
    if not enabled:
        return None, None

    payload: Dict[str, Any] = {
        "enabled": True,
        "gpu_index": int(gpu_index),
        "interval_seconds": int(interval_seconds),
    }
    nvidia_smi = shutil.which("nvidia-smi")
    payload["nvidia_smi_path"] = nvidia_smi
    if nvidia_smi is None:
        payload["status"] = "unavailable"
        payload["reason"] = "nvidia-smi not found in PATH"
        return None, payload

    command = [nvidia_smi, "dmon", "-s", "u", "-d", str(interval_seconds)]
    payload["command"] = command
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except Exception as exc:
        payload["status"] = "error"
        payload["reason"] = str(exc)
        return None, payload

    payload["status"] = "running"
    return process, payload


def _stop_nvidia_dmon(
    process: Optional[subprocess.Popen],
    payload: Optional[Dict[str, Any]],
    gpu_index: int,
) -> Optional[Dict[str, Any]]:
    if payload is None:
        return None
    if process is None:
        return payload

    stdout_text = ""
    stderr_text = ""
    try:
        process.terminate()
        stdout_text, stderr_text = process.communicate(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        stdout_text, stderr_text = process.communicate(timeout=5)
        payload["force_killed"] = True
    except Exception as exc:
        payload["status"] = "error"
        payload["reason"] = f"failed stopping nvidia-smi dmon: {exc}"
        return payload

    parsed = _parse_nvidia_dmon_output(stdout_text=stdout_text, gpu_index=gpu_index)
    payload.update(parsed)
    if stderr_text.strip():
        tail = stderr_text.strip().splitlines()[-10:]
        payload["stderr_tail"] = "\n".join(tail)
    return payload


def _get_video_metadata(video_path: Path) -> Dict[str, Any]:
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


def _resolve_frame_window(total_frames: int, start_frame: int, max_frames: int) -> Tuple[int, int]:
    if start_frame < 0:
        raise ValueError("--start-frame must be >= 0")
    if max_frames < 0:
        raise ValueError("--max-frames must be >= 0")
    if start_frame >= total_frames:
        raise ValueError(
            f"Start frame {start_frame} is beyond total frames {total_frames}"
        )
    end_frame = total_frames if max_frames == 0 else min(total_frames, start_frame + max_frames)
    if end_frame <= start_frame:
        raise ValueError("No frames selected for benchmark")
    return start_frame, end_frame


def _decode_once_opencv(
    video_path: Path,
    start_frame: int,
    end_frame: int,
    batch_size: int,
) -> Dict[str, Any]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video with OpenCV: {video_path}")
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(start_frame))

    frame_idx = start_frame
    frames_processed = 0
    batch_decode_ms: List[float] = []
    benchmark_start = time.perf_counter()

    try:
        while frame_idx < end_frame:
            batch_start = time.perf_counter()
            batch_count = 0

            while batch_count < batch_size and frame_idx < end_frame:
                ok, frame = cap.read()
                if not ok:
                    frame_idx = end_frame
                    break
                _ = frame
                batch_count += 1
                frame_idx += 1

            if batch_count == 0:
                break

            elapsed_ms = (time.perf_counter() - batch_start) * 1000.0
            batch_decode_ms.append(elapsed_ms)
            frames_processed += batch_count
    finally:
        cap.release()

    duration_s = time.perf_counter() - benchmark_start
    decode_fps = float(frames_processed / duration_s) if duration_s > 0 else 0.0
    return {
        "frames_processed": int(frames_processed),
        "batches_processed": int(len(batch_decode_ms)),
        "duration_seconds": float(duration_s),
        "decode_fps": decode_fps,
        "batch_decode_ms": batch_decode_ms,
    }


def _decode_once_decord(
    video_path: Path,
    start_frame: int,
    end_frame: int,
    batch_size: int,
    use_gpu: bool,
) -> Dict[str, Any]:
    if decord is None or VideoReader is None or cpu is None:
        if _DECORD_IMPORT_ERROR:
            raise BackendUnavailable(f"Decord import failed: {_DECORD_IMPORT_ERROR}")
        raise BackendUnavailable("Decord is unavailable in this environment")

    if use_gpu:
        if gpu is None:
            raise BackendUnavailable("Decord GPU context is unavailable")
        if torch is None or not torch.cuda.is_available():
            raise BackendUnavailable("CUDA is unavailable for decord_gpu backend")

    if use_gpu:
        decord.bridge.set_bridge("torch")
        reader = VideoReader(str(video_path), ctx=gpu(0))
    else:
        decord.bridge.set_bridge("native")
        reader = VideoReader(str(video_path), ctx=cpu())

    total_frames = len(reader)
    if start_frame >= total_frames:
        raise ValueError(
            f"Start frame {start_frame} is beyond decord frame count {total_frames}"
        )
    local_end = min(end_frame, total_frames)
    if local_end <= start_frame:
        raise ValueError("No frames selected for decord benchmark")

    frames_processed = 0
    batch_decode_ms: List[float] = []
    benchmark_start = time.perf_counter()

    try:
        for batch_start in range(start_frame, local_end, batch_size):
            batch_end = min(batch_start + batch_size, local_end)
            indices = list(range(batch_start, batch_end))
            tick = time.perf_counter()
            batch = reader.get_batch(indices)
            if use_gpu and torch is not None:
                torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - tick) * 1000.0
            batch_decode_ms.append(elapsed_ms)
            frames_processed += len(indices)
            del batch
    finally:
        del reader
        if use_gpu and torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()

    duration_s = time.perf_counter() - benchmark_start
    decode_fps = float(frames_processed / duration_s) if duration_s > 0 else 0.0
    return {
        "frames_processed": int(frames_processed),
        "batches_processed": int(len(batch_decode_ms)),
        "duration_seconds": float(duration_s),
        "decode_fps": decode_fps,
        "batch_decode_ms": batch_decode_ms,
    }


def _decode_once_native_cmd(
    native_cmd_template: str,
    video_path: Path,
    start_frame: int,
    end_frame: int,
    batch_size: int,
    repetition_index: int,
    phase: str,
    timeout_seconds: float,
) -> Dict[str, Any]:
    frames_requested = end_frame - start_frame
    with tempfile.NamedTemporaryFile(prefix="decode_native_", suffix=".json", delete=False) as tmp:
        output_json = Path(tmp.name)

    format_kwargs = {
        "video_path": str(video_path),
        "start_frame": str(start_frame),
        "max_frames": str(frames_requested),
        "batch_size": str(batch_size),
        "output_json": str(output_json),
        "repeat_index": str(repetition_index),
        "phase": phase,
    }
    cmd_text = native_cmd_template.format(**format_kwargs)

    try:
        completed = subprocess.run(
            shlex.split(cmd_text),
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"native_cmd timed out after {timeout_seconds}s: {exc}"
        ) from exc

    try:
        if completed.returncode != 0:
            stderr_tail = (completed.stderr or "").strip()
            raise RuntimeError(
                f"native_cmd failed with exit code {completed.returncode}: {stderr_tail}"
            )

        payload: Optional[Dict[str, Any]] = None
        if output_json.exists() and output_json.stat().st_size > 0:
            payload = json.loads(output_json.read_text(encoding="utf-8"))
        else:
            stdout = (completed.stdout or "").strip()
            if stdout:
                payload = json.loads(stdout)

        if payload is None:
            raise RuntimeError(
                "native_cmd produced no JSON (stdout empty and output_json not written)"
            )

        frames_processed = int(
            payload.get("frames_processed", payload.get("frames", payload.get("decoded_frames", frames_requested)))
        )
        duration_s: Optional[float] = payload.get("duration_seconds", payload.get("total_seconds"))
        decode_fps: Optional[float] = payload.get("decode_fps")
        batch_decode_ms = payload.get("batch_decode_ms", payload.get("batch_times_ms", []))
        if batch_decode_ms is None:
            batch_decode_ms = []

        batch_decode_ms = [float(v) for v in batch_decode_ms]

        if duration_s is None and decode_fps is not None and decode_fps > 0:
            duration_s = float(frames_processed / decode_fps)
        if duration_s is None:
            raise RuntimeError(
                "native_cmd JSON must include duration_seconds (or decode_fps with frames_processed)"
            )

        duration_s = float(duration_s)
        if decode_fps is None:
            decode_fps = float(frames_processed / duration_s) if duration_s > 0 else 0.0

        return {
            "frames_processed": int(frames_processed),
            "batches_processed": int(len(batch_decode_ms)),
            "duration_seconds": duration_s,
            "decode_fps": float(decode_fps),
            "batch_decode_ms": batch_decode_ms,
            "native_payload": payload,
        }
    finally:
        try:
            output_json.unlink(missing_ok=True)
        except Exception:
            pass


def _run_backend_once(
    backend: str,
    video_path: Path,
    start_frame: int,
    end_frame: int,
    batch_size: int,
    native_cmd_template: Optional[str],
    native_timeout_seconds: float,
    repetition_index: int,
    phase: str,
) -> Dict[str, Any]:
    if backend == BACKEND_OPENCV:
        return _decode_once_opencv(video_path, start_frame, end_frame, batch_size)
    if backend == BACKEND_DECORD_CPU:
        return _decode_once_decord(video_path, start_frame, end_frame, batch_size, use_gpu=False)
    if backend == BACKEND_DECORD_GPU:
        return _decode_once_decord(video_path, start_frame, end_frame, batch_size, use_gpu=True)
    if backend == BACKEND_NATIVE_CMD:
        if not native_cmd_template:
            raise BackendUnavailable(
                "native_cmd backend requested but --native-cmd template was not provided"
            )
        return _decode_once_native_cmd(
            native_cmd_template=native_cmd_template,
            video_path=video_path,
            start_frame=start_frame,
            end_frame=end_frame,
            batch_size=batch_size,
            repetition_index=repetition_index,
            phase=phase,
            timeout_seconds=native_timeout_seconds,
        )
    raise ValueError(f"Unsupported backend: {backend}")


def _run_backend_once_with_optional_dmon(
    *,
    backend: str,
    video_path: Path,
    start_frame: int,
    end_frame: int,
    batch_size: int,
    native_cmd_template: Optional[str],
    native_timeout_seconds: float,
    repetition_index: int,
    phase: str,
    collect_nvidia_dmon: bool,
    dmon_interval_seconds: int,
    dmon_gpu_index: int,
) -> Dict[str, Any]:
    dmon_process, dmon_payload = _start_nvidia_dmon(
        enabled=collect_nvidia_dmon,
        interval_seconds=dmon_interval_seconds,
        gpu_index=dmon_gpu_index,
    )
    run_error: Optional[Exception] = None
    metrics: Optional[Dict[str, Any]] = None
    try:
        metrics = _run_backend_once(
            backend=backend,
            video_path=video_path,
            start_frame=start_frame,
            end_frame=end_frame,
            batch_size=batch_size,
            native_cmd_template=native_cmd_template,
            native_timeout_seconds=native_timeout_seconds,
            repetition_index=repetition_index,
            phase=phase,
        )
    except Exception as exc:
        run_error = exc
    finally:
        dmon_payload = _stop_nvidia_dmon(
            process=dmon_process,
            payload=dmon_payload,
            gpu_index=dmon_gpu_index,
        )

    if metrics is not None and dmon_payload is not None:
        metrics["nvidia_dmon"] = dmon_payload
    if run_error is not None:
        raise run_error
    if metrics is None:
        raise RuntimeError("backend run produced no metrics")
    return metrics


def _summarize_backend_runs(repetitions: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    fps_values = [float(rep["decode_fps"]) for rep in repetitions]
    duration_values = [float(rep["duration_seconds"]) for rep in repetitions]
    frames_values = [float(rep["frames_processed"]) for rep in repetitions]
    batch_ms_values: List[float] = []
    for rep in repetitions:
        batch_ms_values.extend(float(v) for v in rep.get("batch_decode_ms", []))
    return {
        "decode_fps": _series_stats(fps_values),
        "duration_seconds": _series_stats(duration_values),
        "frames_processed": _series_stats(frames_values),
        "batch_decode_ms": _series_stats(batch_ms_values),
    }


def _collect_environment() -> Dict[str, Any]:
    torch_cuda = bool(torch is not None and torch.cuda.is_available())
    env: Dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "hostname": platform.node(),
        "nvidia_smi_path": shutil.which("nvidia-smi"),
        "decord_imported": bool(decord is not None),
        "decord_import_error": str(_DECORD_IMPORT_ERROR) if _DECORD_IMPORT_ERROR else None,
        "torch_available": bool(torch is not None),
        "cuda_available": torch_cuda,
    }
    if torch is not None:
        env["torch_version"] = getattr(torch, "__version__", None)
    if decord is not None:
        env["decord_version"] = getattr(decord, "__version__", None)
    if torch_cuda and torch is not None:
        env["cuda_device_count"] = int(torch.cuda.device_count())
        if torch.cuda.device_count() > 0:
            env["cuda_device_name"] = torch.cuda.get_device_name(0)
    return env


def _print_backend_summary(backend_result: Dict[str, Any]) -> None:
    backend = backend_result["backend"]
    status = backend_result["status"]
    if status != "ok":
        reason = backend_result.get("reason", "unknown")
        print(f"- {backend}: {status} ({reason})")
        return
    summary = backend_result["summary"]
    fps_stats = summary["decode_fps"]
    batch_stats = summary["batch_decode_ms"]
    print(
        "- {backend}: fps median={fps_med:.2f}, fps p90={fps_p90:.2f}, "
        "batch_ms median={batch_med:.2f}, batch_ms p90={batch_p90:.2f}".format(
            backend=backend,
            fps_med=fps_stats["median"] or 0.0,
            fps_p90=fps_stats["p90"] or 0.0,
            batch_med=batch_stats["median"] or 0.0,
            batch_p90=batch_stats["p90"] or 0.0,
        )
    )


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark decode backends only (no model inference) for detection pipeline planning."
        )
    )
    parser.add_argument("video_path", type=Path, help="Input video file path.")
    parser.add_argument(
        "--backends",
        nargs="+",
        default=None,
        choices=BACKEND_CHOICES,
        help=(
            "Backends to benchmark in order. "
            "Default: decord_gpu decord_cpu opencv "
            "(or preset-selected order if --cpu-default-test/--full-suite)."
        ),
    )
    parser.add_argument(
        "--cpu-default-test",
        action="store_true",
        help=(
            "Apply a quick CPU-focused preset. If --backends is omitted, uses "
            "`decord_cpu opencv`. If --max-frames is omitted, uses "
            "`batch_size * --cpu-default-batches`."
        ),
    )
    parser.add_argument(
        "--cpu-default-batches",
        type=int,
        default=5,
        help="Batch count used by --cpu-default-test when --max-frames is not set (default: 5).",
    )
    parser.add_argument(
        "--full-suite",
        action="store_true",
        help=(
            "Apply a combined CPU+GPU preset. If --backends is omitted, uses "
            "`decord_gpu decord_cpu opencv native_cmd` and auto-wires the built-in "
            "FFmpeg/NVDEC adapter when --native-cmd is not set. "
            "If --max-frames is omitted, uses `batch_size * --full-suite-batches`."
        ),
    )
    parser.add_argument(
        "--full-suite-batches",
        type=int,
        default=5,
        help="Batch count used by --full-suite when --max-frames is not set (default: 5).",
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        default=0,
        help="First frame index to include (default: 0).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Number of frames to process; 0 means until EOF (default: 0).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Decode batch size (default: 64).",
    )
    parser.add_argument(
        "--warmup-reps",
        type=int,
        default=1,
        help="Warmup repetitions per backend (discarded, default: 1).",
    )
    parser.add_argument(
        "--timed-reps",
        type=int,
        default=3,
        help="Timed repetitions per backend (default: 3).",
    )
    parser.add_argument(
        "--native-cmd",
        type=str,
        default=None,
        help=(
            "Template command for native backend integration. Supports placeholders: "
            "{video_path}, {start_frame}, {max_frames}, {batch_size}, "
            "{output_json}, {repeat_index}, {phase}."
        ),
    )
    parser.add_argument(
        "--native-timeout-seconds",
        type=float,
        default=3600.0,
        help="Timeout for each native-cmd invocation (default: 3600).",
    )
    parser.add_argument(
        "--collect-nvidia-dmon",
        action="store_true",
        help="Collect nvidia-smi dmon utilization samples during each repetition.",
    )
    parser.add_argument(
        "--dmon-interval-seconds",
        type=int,
        default=1,
        help="Sampling interval for nvidia-smi dmon (default: 1 second).",
    )
    parser.add_argument(
        "--dmon-gpu-index",
        type=int,
        default=0,
        help="GPU index to extract from dmon samples (default: 0).",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Output JSON path. Defaults to runs/benchmarks/detect_decode_<timestamp>.json",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = parse_args(argv)

    video_path = args.video_path.expanduser().resolve()
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0")
    if args.warmup_reps < 0:
        raise ValueError("--warmup-reps must be >= 0")
    if args.timed_reps <= 0:
        raise ValueError("--timed-reps must be > 0")
    if args.dmon_interval_seconds <= 0:
        raise ValueError("--dmon-interval-seconds must be > 0")
    if args.dmon_gpu_index < 0:
        raise ValueError("--dmon-gpu-index must be >= 0")
    if args.cpu_default_batches <= 0:
        raise ValueError("--cpu-default-batches must be > 0")
    if args.full_suite_batches <= 0:
        raise ValueError("--full-suite-batches must be > 0")
    if args.cpu_default_test and args.full_suite:
        raise ValueError("--cpu-default-test and --full-suite are mutually exclusive")

    selected_backends = (
        list(args.backends)
        if args.backends is not None
        else [BACKEND_DECORD_GPU, BACKEND_DECORD_CPU, BACKEND_OPENCV]
    )
    resolved_max_frames = int(args.max_frames)
    preset_applied: Optional[str] = None
    if args.cpu_default_test:
        preset_applied = "cpu_default_test"
        if args.backends is None:
            selected_backends = [BACKEND_DECORD_CPU, BACKEND_OPENCV]
        if args.max_frames == 0:
            resolved_max_frames = int(args.batch_size) * int(args.cpu_default_batches)
    elif args.full_suite:
        preset_applied = "full_suite"
        if args.backends is None:
            selected_backends = [
                BACKEND_DECORD_GPU,
                BACKEND_DECORD_CPU,
                BACKEND_OPENCV,
                BACKEND_NATIVE_CMD,
            ]
        if args.max_frames == 0:
            resolved_max_frames = int(args.batch_size) * int(args.full_suite_batches)

    native_backend_requested = BACKEND_NATIVE_CMD in selected_backends
    resolved_native_cmd, native_cmd_source, native_cmd_unavailable_reason = _resolve_native_cmd_template(
        explicit_native_cmd=args.native_cmd,
        native_backend_requested=native_backend_requested,
    )

    video_meta = _get_video_metadata(video_path)
    start_frame, end_frame = _resolve_frame_window(
        total_frames=video_meta["total_frames"],
        start_frame=int(args.start_frame),
        max_frames=resolved_max_frames,
    )
    frames_target = end_frame - start_frame

    fixture = {
        "video_path": str(video_path),
        "video_metadata": video_meta,
        "start_frame": int(start_frame),
        "end_frame_exclusive": int(end_frame),
        "frames_target": int(frames_target),
        "max_frames_arg": int(args.max_frames),
        "resolved_max_frames": int(resolved_max_frames),
        "cpu_default_test": bool(args.cpu_default_test),
        "cpu_default_batches": int(args.cpu_default_batches),
        "full_suite": bool(args.full_suite),
        "full_suite_batches": int(args.full_suite_batches),
        "preset_applied": preset_applied,
        "native_cmd_requested": bool(native_backend_requested),
        "native_cmd_source": native_cmd_source,
        "native_cmd_template": resolved_native_cmd,
        "native_cmd_unavailable_reason": native_cmd_unavailable_reason,
        "batch_size": int(args.batch_size),
        "warmup_repetitions": int(args.warmup_reps),
        "timed_repetitions": int(args.timed_reps),
        "collect_nvidia_dmon": bool(args.collect_nvidia_dmon),
        "dmon_interval_seconds": int(args.dmon_interval_seconds),
        "dmon_gpu_index": int(args.dmon_gpu_index),
        "backends": list(selected_backends),
    }

    run_payload: Dict[str, Any] = {
        "schema_version": 1,
        "created_at_utc": _utc_now_iso(),
        "fixture": fixture,
        "environment": _collect_environment(),
        "nvidia_dmon": {
            "enabled": bool(args.collect_nvidia_dmon),
            "interval_seconds": int(args.dmon_interval_seconds),
            "gpu_index": int(args.dmon_gpu_index),
        },
        "results": [],
    }

    print("Decode benchmark fixture:")
    print(
        f"- video={video_path} frames={frames_target} range=[{start_frame}, {end_frame}) "
        f"batch={args.batch_size} warmup={args.warmup_reps} timed={args.timed_reps}"
    )
    if preset_applied is not None:
        print(f"- preset={preset_applied} backends={selected_backends} resolved_max_frames={resolved_max_frames}")
    if native_backend_requested:
        if resolved_native_cmd is not None:
            print(f"- native_cmd source={native_cmd_source}")
        else:
            print(
                "- native_cmd source={source} (unavailable: {reason})".format(
                    source=native_cmd_source,
                    reason=native_cmd_unavailable_reason or "unknown",
                )
            )

    for backend in selected_backends:
        backend_result: Dict[str, Any] = {
            "backend": backend,
            "status": "ok",
            "warmup_repetitions": [],
            "timed_repetitions": [],
        }
        print(f"\nRunning backend: {backend}")
        try:
            for rep in range(args.warmup_reps):
                metrics = _run_backend_once_with_optional_dmon(
                    backend=backend,
                    video_path=video_path,
                    start_frame=start_frame,
                    end_frame=end_frame,
                    batch_size=args.batch_size,
                    native_cmd_template=resolved_native_cmd,
                    native_timeout_seconds=args.native_timeout_seconds,
                    repetition_index=rep,
                    phase="warmup",
                    collect_nvidia_dmon=bool(args.collect_nvidia_dmon),
                    dmon_interval_seconds=int(args.dmon_interval_seconds),
                    dmon_gpu_index=int(args.dmon_gpu_index),
                )
                metrics["repetition_index"] = rep
                backend_result["warmup_repetitions"].append(metrics)
                print(
                    f"  warmup rep {rep + 1}/{args.warmup_reps}: "
                    f"{metrics['decode_fps']:.2f} fps"
                )

            for rep in range(args.timed_reps):
                metrics = _run_backend_once_with_optional_dmon(
                    backend=backend,
                    video_path=video_path,
                    start_frame=start_frame,
                    end_frame=end_frame,
                    batch_size=args.batch_size,
                    native_cmd_template=resolved_native_cmd,
                    native_timeout_seconds=args.native_timeout_seconds,
                    repetition_index=rep,
                    phase="timed",
                    collect_nvidia_dmon=bool(args.collect_nvidia_dmon),
                    dmon_interval_seconds=int(args.dmon_interval_seconds),
                    dmon_gpu_index=int(args.dmon_gpu_index),
                )
                metrics["repetition_index"] = rep
                backend_result["timed_repetitions"].append(metrics)
                print(
                    f"  timed rep {rep + 1}/{args.timed_reps}: "
                    f"{metrics['decode_fps']:.2f} fps"
                )

            backend_result["summary"] = _summarize_backend_runs(
                backend_result["timed_repetitions"]
            )
        except BackendUnavailable as exc:
            backend_result["status"] = "skipped"
            backend_result["reason"] = str(exc)
            backend_result.pop("summary", None)
        except Exception as exc:
            if backend == BACKEND_NATIVE_CMD:
                backend_result["status"] = "skipped"
                backend_result["reason"] = f"native_cmd failed; skipping backend: {exc}"
            else:
                backend_result["status"] = "error"
                backend_result["reason"] = str(exc)
            backend_result.pop("summary", None)

        run_payload["results"].append(backend_result)
        _print_backend_summary(backend_result)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_json = (
        args.output_json
        if args.output_json is not None
        else Path("runs/benchmarks") / f"detect_decode_{timestamp}.json"
    )
    output_json = output_json.expanduser()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(run_payload, indent=2), encoding="utf-8")

    print(f"\nWrote benchmark JSON: {output_json}")


if __name__ == "__main__":  # pragma: no cover
    main()
