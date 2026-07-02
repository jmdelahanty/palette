#!/usr/bin/env python3
"""
Direct video inference with YOLO - no import required.

The preferred CLI entry point is :mod:`fisheye.inference.predict_detections`,
which wraps this module and keeps inference scripts in one namespace. This
module still provides the core implementation and legacy CLI.
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, Mapping

os.environ.setdefault("DECORD_EOF_RETRY_MAX", "65536")

# Try to import decord for faster video decoding (GPU or CPU) before other FFmpeg users.
try:
    import decord  # type: ignore
    from decord import VideoReader, cpu, gpu  # type: ignore
    _DECORD_IMPORT_ERROR: Optional[Exception] = None
except Exception as exc:  # pragma: no cover - environment dependent
    decord = None  # type: ignore
    VideoReader = None  # type: ignore
    cpu = None  # type: ignore
    gpu = None  # type: ignore
    _DECORD_IMPORT_ERROR = exc

import sys
import time
import yaml
import zarr
import numpy as np
import torch
import cv2
import imageio.v3 as iio
from datetime import datetime, timezone
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn
from rich.panel import Panel
from rich.markup import escape
from ultralytics import YOLO

from fisheye.shared.pynvvc_luma_rgb import BACKEND_PYNVVC_LUMA_RGB
from fisheye.shared.pynvvc_luma_rgb import BACKEND_PYNVVC_NV12_RGB
from fisheye.shared.pynvvc_luma_rgb import PYNVVC_BACKENDS
from fisheye.shared.pynvvc_luma_rgb import PynvvcLumaRgbReader
from fisheye.shared.pynvvc_luma_rgb import preprocess_luma_rgb
from fisheye.shared.pynvvc_luma_rgb import preprocess_nv12_rgb
from fisheye.shared.stage_provenance import build_stage_provenance, write_stage_provenance
from fisheye.shared.zarr.schema import get_run_group
from fisheye.shared.zarr.chunk_profiles import create_geometry_preload_array
from fisheye.shared.zarr_run_completion import (
    mark_run_complete,
    mark_run_started,
    note_pending_latest,
    require_runs_parent,
)
from fisheye.utils.import_video_metadata import write_video_metadata
from fisheye.utils.system import get_environment_info, get_git_info


DECODE_BACKEND_AUTO = "auto"
DECODE_BACKEND_DECORD_GPU = "decord_gpu"
DECODE_BACKEND_DECORD_CPU = "decord_cpu"
DECODE_BACKEND_OPENCV = "opencv"
DECODE_BACKEND_CHOICES = (
    DECODE_BACKEND_AUTO,
    *PYNVVC_BACKENDS,
    DECODE_BACKEND_DECORD_GPU,
    DECODE_BACKEND_DECORD_CPU,
    DECODE_BACKEND_OPENCV,
)
PYNVVC_SURFACE_MATERIALIZATION = "stream_event_owned_batch_v2"


def _decord_available() -> bool:
    return decord is not None and VideoReader is not None and cpu is not None


def _normalize_decode_backend(value: Optional[str]) -> str:
    if value is None:
        return DECODE_BACKEND_AUTO
    normalized = str(value).strip().lower()
    if not normalized:
        return DECODE_BACKEND_AUTO
    if normalized not in DECODE_BACKEND_CHOICES:
        choices = ", ".join(DECODE_BACKEND_CHOICES)
        raise ValueError(f"Unsupported decode backend {value!r}; expected one of: {choices}")
    return normalized


def _collect_zarr_metadata(zarr_path: Path, console: Optional[Console] = None) -> Dict[str, Any]:
    """
    Return palette metadata (width, height, fps, etc.) from an existing Zarr archive.

    The lookup is best-effort: we consult root attributes, then raw_video attrs,
    and finally dataset shapes. Missing entries are omitted from the result.
    """
    metadata: Dict[str, Any] = {}
    try:
        root = zarr.open(str(zarr_path), mode="r")
    except Exception as exc:  # pragma: no cover - best-effort metadata lookup
        if console:
            console.print(f"[yellow]Warning:[/yellow] Unable to open source Zarr '{zarr_path}': {escape(str(exc))}")
        return metadata

    def _as_int(value: Any) -> Optional[int]:
        try:
            return int(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    def _get_attr(source, *names) -> Optional[Any]:
        for name in names:
            if name in source:
                return source.get(name)
        return None

    # Root-level metadata
    width_attr = _as_int(_get_attr(root.attrs, "width", "video_width"))
    height_attr = _as_int(_get_attr(root.attrs, "height", "video_height"))
    fps = _get_attr(root.attrs, "fps", "video_fps")
    total_frames = _get_attr(root.attrs, "total_frames", "n_frames")
    duration_seconds = _get_attr(root.attrs, "duration_seconds", "video_duration_seconds")
    video_codec = _get_attr(root.attrs, "video_codec")
    video_pix_fmt = _get_attr(root.attrs, "video_pix_fmt")

    # raw_video group metadata provides additional detail
    raw_group = root.get("raw_video")
    if raw_group is not None:
        raw_attrs = raw_group.attrs
        width_attr = width_attr or _as_int(_get_attr(raw_attrs, "video_width"))
        height_attr = height_attr or _as_int(_get_attr(raw_attrs, "video_height"))
        fps = fps or raw_attrs.get("fps")
        total_frames = total_frames or raw_attrs.get("total_frames")
        duration_seconds = duration_seconds or raw_attrs.get("video_duration_seconds")
        video_codec = video_codec or raw_attrs.get("video_codec")
        video_pix_fmt = video_pix_fmt or raw_attrs.get("video_pix_fmt")

        original_resolution = raw_attrs.get("original_resolution")
        if original_resolution and len(original_resolution) == 2:
            metadata["original_resolution"] = [int(original_resolution[0]), int(original_resolution[1])]

        if "has_full_resolution" in raw_attrs:
            metadata["has_full_resolution"] = bool(raw_attrs.get("has_full_resolution"))
        else:
            metadata["has_full_resolution"] = "images_full" in raw_group

        if "has_downsampled" in raw_attrs:
            metadata["has_downsampled"] = bool(raw_attrs.get("has_downsampled"))
        else:
            metadata["has_downsampled"] = "images_ds" in raw_group

        downsampled_resolution = raw_attrs.get("downsampled_resolution")
        if downsampled_resolution and len(downsampled_resolution) == 2:
            metadata["downsampled_resolution"] = [int(downsampled_resolution[0]), int(downsampled_resolution[1])]

        downsample_method = raw_attrs.get("downsample_method")
        if downsample_method:
            metadata["downsample_method"] = downsample_method

        for key in ("images_full", "images", "images_ds"):
            if key in raw_group:
                shape = raw_group[key].shape
                if shape and len(shape) >= 3:
                    height = int(shape[-2])
                    width = int(shape[-1])
                    if key == "images_full":
                        width_attr = width_attr or width
                        height_attr = height_attr or height
                        metadata.setdefault("original_resolution", [height, width])
                    if key == "images_ds":
                        metadata.setdefault("downsampled_resolution", [height, width])

    if width_attr is not None:
        metadata["full_width"] = int(width_attr)
    if height_attr is not None:
        metadata["full_height"] = int(height_attr)
    if fps is not None:
        metadata["fps"] = float(fps)
    if total_frames is not None:
        metadata["total_frames"] = int(total_frames)
    if duration_seconds is not None:
        metadata["duration_seconds"] = float(duration_seconds)
    if video_codec:
        metadata["video_codec"] = video_codec
    if video_pix_fmt:
        metadata["video_pix_fmt"] = video_pix_fmt

    return metadata


def _gpu_decode_unavailable(reason: str) -> RuntimeError:
    return RuntimeError(
        "GPU decode unavailable; refusing CPU fallback - pixels would differ from the "
        f"production path ({reason})"
    )


def _init_decord_reader(video_path: Path, prefer_gpu: bool, console: Console) -> Optional[Dict[str, Any]]:
    """Initialise an explicitly requested Decord VideoReader without backend fallback."""
    if not _decord_available():
        if _DECORD_IMPORT_ERROR:
            console.print(f"[yellow]Decord unavailable: {escape(str(_DECORD_IMPORT_ERROR))}[/yellow]")
        return None

    if prefer_gpu:
        if not torch.cuda.is_available():
            raise _gpu_decode_unavailable("Decord GPU requested but CUDA is unavailable")
        try:
            decord.bridge.set_bridge('torch')
            vr = VideoReader(str(video_path), ctx=gpu(0))
            first = vr[0]
            height, width = int(first.shape[0]), int(first.shape[1])
            fps = vr.get_avg_fps()
            console.print(f"[green]✓[/green] Using Decord GPU decoder")
            return {
                'reader': vr,
                'type': 'decord_gpu',
                'on_gpu': True,
                'width': width,
                'height': height,
                'fps': fps,
            }
        except Exception as exc:
            raise _gpu_decode_unavailable(f"Decord GPU decoder failed: {exc}") from exc

    try:
        decord.bridge.set_bridge('native')
        vr = VideoReader(str(video_path), ctx=cpu())
        first = vr[0]
        height, width = int(first.shape[0]), int(first.shape[1])
        fps = vr.get_avg_fps()
        console.print(f"[green]✓[/green] Using Decord CPU decoder")
        return {
            'reader': vr,
            'type': 'decord_cpu',
            'on_gpu': False,
            'width': width,
            'height': height,
            'fps': fps,
        }
    except Exception as exc:
        raise RuntimeError(f"Requested Decord CPU decoder failed: {exc}") from exc


def get_video_metadata(video_path: Path, cap: Optional[cv2.VideoCapture], width: int, height: int, n_frames: int, fps: float) -> Dict[str, Any]:
    """
    Get comprehensive video metadata similar to import_video.py.
    Uses both cv2 and imageio for maximum compatibility.
    """
    cap_owner = False
    if cap is None:
        cap = cv2.VideoCapture(str(video_path))
        cap_owner = True
        if not cap.isOpened():
            cap = None
    # Try to get codec info from imageio
    try:
        iio_meta = iio.immeta(str(video_path))
    except Exception:
        iio_meta = {}
    
    # Build metadata dictionary matching import_video structure
    meta = {
        "source_video": str(video_path.name),
        "source_path": str(video_path.absolute()),
        "width": width,
        "height": height,
        "total_frames": n_frames,
        "fps": fps,
        "duration_seconds": n_frames / fps if fps > 0 else 0,
    }
    
    # Add codec information if available
    if iio_meta:
        meta["codec"] = iio_meta.get("codec", "unknown")
        meta["pix_fmt"] = iio_meta.get("pix_fmt", "unknown")
        meta["imageio_metadata"] = iio_meta
    else:
        # Fallback to cv2 fourcc
        if cap is not None:
            fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
            if fourcc > 0:
                codec_str = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
                meta["codec"] = codec_str
            else:
                meta["codec"] = "unknown"
        else:
            meta["codec"] = "unknown"
        meta["pix_fmt"] = "unknown"
    
    if cap_owner and cap is not None:
        cap.release()
    return meta


def _read_cv2_video_properties(video_path: Path) -> Tuple[int, float, int, int]:
    """Return (n_frames, fps, width, height) using OpenCV metadata only."""

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video metadata: {video_path}")
    try:
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        cap.release()
    return n_frames, fps, width, height


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    import yaml
    
    if config_path is None:
        # Try default locations
        default_paths = [
            Path('yolo_detect_config.yaml'),  # Current directory
            Path('configs/fisheye/yolo_detect_config.yaml'),  # Standard location
            Path(__file__).parent.parent.parent / 'configs/fisheye/yolo_detect_config.yaml',  # Relative to module
            Path.home() / 'gitrepos/palette/configs/fisheye/yolo_detect_config.yaml',  # Absolute
            Path('src/fisheye/yolo_detect_config.yaml'),  # Old location
            Path(__file__).parent / 'yolo_detect_config.yaml',  # Same dir as this script
        ]
        
        console = Console()
        console.print("[dim]Searching for config file...[/dim]")
        
        for path in default_paths:
            console.print(f"[dim]  Checking: {path}[/dim]")
            if path.exists():
                console.print(f"[green]  ✓ Found config: {path}[/green]")
                config_path = path
                break
        
        if config_path is None:
            console.print("[yellow]  No config file found, using defaults[/yellow]")
    
    if config_path and Path(config_path).exists():
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    return {}


def _normalize_imgsz(value: Optional[Any]) -> Optional[int | list[int]]:
    """Normalize imgsz to Ultralytics-compatible int or [h, w] list."""
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        ints: list[int] = []
        for item in value:
            if item is None:
                continue
            try:
                ints.append(int(item))
            except (TypeError, ValueError):
                continue
        if not ints:
            return None
        if len(ints) == 1:
            return ints[0]
        return [ints[0], ints[1]]
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _imgsz_to_resize_dims(value: Optional[int | list[int]]) -> Optional[list[int]]:
    """Normalize imgsz-like values to canonical [h, w] resize dims."""
    if value is None:
        return None
    if isinstance(value, int):
        if value <= 0:
            return None
        return [value, value]
    if not value:
        return None
    if len(value) == 1:
        edge = int(value[0])
        if edge <= 0:
            return None
        return [edge, edge]
    height = int(value[0])
    width = int(value[1])
    if height <= 0 or width <= 0:
        return None
    return [height, width]


def _normalize_resize_dims(value: Optional[Any]) -> Optional[list[int]]:
    """Normalize canonical resize dims to [h, w]."""
    return _imgsz_to_resize_dims(_normalize_imgsz(value))


def _normalize_legacy_video_resize(value: Optional[Any]) -> Optional[list[int]]:
    """
    Normalize legacy video.resize [w, h] to canonical [h, w].

    Historical detect configs use video.resize as [width, height].
    """
    parsed = _normalize_imgsz(value)
    if parsed is None:
        return None
    if isinstance(parsed, int):
        return _imgsz_to_resize_dims(parsed)
    if len(parsed) == 1:
        return _imgsz_to_resize_dims(parsed)
    width = int(parsed[0])
    height = int(parsed[1])
    if width <= 0 or height <= 0:
        return None
    return [height, width]


def _resize_dims_to_imgsz(value: Optional[list[int]]) -> Optional[int | list[int]]:
    """Convert canonical [h, w] dims to an Ultralytics imgsz value."""
    if value is None:
        return None
    if value[0] == value[1]:
        return int(value[0])
    return [int(value[0]), int(value[1])]


def _actual_input_resize_dims(
    requested_resize_dims: Optional[list[int]],
    pre_resize_dims: Optional[list[int]],
    *,
    decord_on_gpu: bool,
    tensor_input: bool = False,
) -> Optional[list[int]]:
    """
    Return the frame dimensions that will be explicitly fed to YOLO.

    Ultralytics applies imgsz preprocessing for numpy/list inputs, but torch
    tensor inputs are treated as already prepared. Tensor-backed decoder paths
    must apply canonical detection.resize_dims explicitly or inference silently
    runs at source-video resolution.
    """
    if decord_on_gpu or tensor_input:
        if requested_resize_dims is None:
            raise RuntimeError(
                "GPU tensor decode paths require detection.resize_dims or --resize-dims. "
                "Without an explicit resize, Ultralytics treats tensor inputs as already "
                "prepared and can run inference at source-video resolution."
            )
        return [int(requested_resize_dims[0]), int(requested_resize_dims[1])]
    if pre_resize_dims is not None:
        return [int(pre_resize_dims[0]), int(pre_resize_dims[1])]
    return None


def _record_timing(timings: Dict[str, float], key: str, start: float) -> float:
    """Accumulate a stage duration from a perf-counter start timestamp."""
    elapsed = time.perf_counter() - start
    timings[key] = float(timings.get(key, 0.0) + elapsed)
    return elapsed


def _read_and_preprocess_pynvvc_batch(
    *,
    frame_iter: Any,
    max_batch_frames: int,
    decode_backend_effective: str,
    source_height: int,
    device: torch.device,
    dtype: torch.dtype,
    resize_hw: list[int] | tuple[int, int],
) -> tuple[Optional[torch.Tensor], int, float, float]:
    """Read PyNvVC frames into an owned preallocated model-input batch.

    PyNvVideoCodec frames can be backed by decoder-owned reusable surfaces. Do
    not retain raw decoded tensors across further decode calls; preprocess each
    yielded frame into owned resized RGB tensor memory before advancing. CUDA
    events scope the wait to the materialization work that reads the decoder
    surface, rather than synchronizing the whole current stream unconditionally.
    """

    height, width = int(resize_hw[0]), int(resize_hw[1])
    max_batch_frames = max(0, int(max_batch_frames))
    if max_batch_frames <= 0:
        return None, 0, 0.0, 0.0

    processed_batch = torch.empty(
        (max_batch_frames, 3, height, width),
        device=device,
        dtype=dtype,
    ).contiguous(memory_format=torch.channels_last)
    read_seconds = 0.0
    preprocess_seconds = 0.0
    actual_count = 0
    for batch_index in range(max_batch_frames):
        read_start = time.perf_counter()
        try:
            frame_tensor = next(frame_iter)
        except StopIteration:
            break
        read_seconds += time.perf_counter() - read_start

        preprocess_start = time.perf_counter()
        if decode_backend_effective == BACKEND_PYNVVC_NV12_RGB:
            processed_frame = preprocess_nv12_rgb(
                [frame_tensor],
                source_height=source_height,
                device=device,
                dtype=dtype,
                resize_hw=resize_hw,
            )
        else:
            processed_frame = preprocess_luma_rgb(
                [frame_tensor],
                source_height=source_height,
                device=device,
                dtype=dtype,
                resize_hw=resize_hw,
            )
        processed_batch[batch_index : batch_index + 1].copy_(
            processed_frame,
            non_blocking=bool(getattr(processed_frame, "is_cuda", False)),
        )
        if getattr(processed_batch, "is_cuda", False):
            # PyNvVideoCodec may recycle the decoded surface on the next decode.
            # Wait only for the event marking completion of this frame's
            # materialization into owned tensor memory before advancing.
            stream = torch.cuda.current_stream(device=device)
            materialized = torch.cuda.Event(blocking=False)
            materialized.record(stream)
            materialized.synchronize()
        preprocess_seconds += time.perf_counter() - preprocess_start
        actual_count += 1
        del frame_tensor
        del processed_frame

    if actual_count <= 0:
        return None, 0, read_seconds, preprocess_seconds
    processed = processed_batch[:actual_count]
    if actual_count != max_batch_frames:
        processed = processed.contiguous(memory_format=torch.channels_last)
    return (
        processed,
        actual_count,
        read_seconds,
        preprocess_seconds,
    )


def detect_yolo(
    video_path: str,
    model_path: Optional[str] = None,
    output_zarr: str = None,
    config_path: Optional[str] = None,
    conf_threshold: Optional[float] = None,
    iou_threshold: Optional[float] = None,
    max_det: Optional[int] = None,
    batch_size: Optional[int] = None,
    resize_dims: Optional[list[int] | tuple[int, int]] = None,
    imgsz: Optional[int | list[int] | tuple[int, int]] = None,
    decode_backend: Optional[str] = None,
    console: Optional[Console] = None,
    use_gpu: Optional[bool] = None,
    write_raw_video_metadata: bool = False,
    overwrite_raw_video_metadata: bool = False,
    run_name: Optional[str] = None,
    cli_provenance: Optional[Mapping[str, Any]] = None,
) -> str:
    """
    Run YOLO inference directly on video file, creating minimal zarr output.
    
    This is the INFERENCE pathway - for getting detections from a trained model.
    Does NOT import full video, only saves detection results.
    
    Args:
        video_path: Path to input video file
        model_path: Path to trained YOLO model (.pt) - optional if in config
        output_zarr: Path for output zarr - optional, will auto-generate if None
        config_path: Path to YAML config file (optional)
        conf_threshold: Confidence threshold (overrides config)
        iou_threshold: IoU threshold for NMS (overrides config)
        max_det: Max detections per frame (overrides config)
        batch_size: Frames to process at once (overrides config)
        resize_dims: Canonical inference size [h, w]; mapped to YOLO imgsz
        imgsz: Legacy YOLO inference size alias; normalized into resize_dims
        decode_backend: Decoder backend (`auto`, `pynvvc_nv12_rgb`,
            `pynvvc_luma_rgb`, `decord_gpu`, `decord_cpu`, or `opencv`)
        console: Rich console
        use_gpu: Use GPU for inference (overrides config)
        write_raw_video_metadata: Create/update raw_video attrs (no frames) for registry/provenance
        overwrite_raw_video_metadata: Overwrite existing raw_video attrs when writing metadata
        run_name: Optional explicit detect run group name. Used by cluster planners
            when downstream jobs need deterministic paths.
        cli_provenance: Optional Palette CLI provenance block stamped before completion.
        
    Returns:
        Name of detect_runs group
    """
    if console is None:
        console = Console()
    
    console.rule("[bold]YOLO Video Inference[/bold]")
    
    # Load config
    config = load_config(config_path)
    
    # Get parameters from config with CLI overrides
    model_path = model_path or config.get('model', {}).get('path')
    if model_path is None:
        raise ValueError("model_path required (via argument or config file)")
    
    # Auto-generate output path if not provided
    video_path = Path(video_path)
    if output_zarr is None:
        output_zarr = video_path.parent / f"{video_path.stem}_detections.zarr"
    
    # Get detection parameters with CLI overrides taking precedence
    detect_config = config.get('detection', {})
    conf_threshold = conf_threshold if conf_threshold is not None else detect_config.get('conf_threshold', 0.40)
    iou_threshold = iou_threshold if iou_threshold is not None else detect_config.get('iou_threshold', 0.45)
    max_det = max_det if max_det is not None else detect_config.get('max_det', 20)
    batch_size = batch_size if batch_size is not None else detect_config.get('batch_size', 32)

    cli_resize_dims = _normalize_resize_dims(resize_dims)
    cli_imgsz_legacy = _normalize_imgsz(imgsz)
    cli_imgsz_as_resize = _imgsz_to_resize_dims(cli_imgsz_legacy)
    if cli_resize_dims is not None and cli_imgsz_as_resize is not None and cli_resize_dims != cli_imgsz_as_resize:
        raise ValueError(
            f"Conflicting CLI overrides: resize_dims={cli_resize_dims} and imgsz={cli_imgsz_as_resize}. "
            "Set one, or make them equal."
        )

    config_resize_dims = _normalize_resize_dims(detect_config.get("resize_dims"))
    config_imgsz_legacy = _normalize_imgsz(detect_config.get("imgsz"))
    config_imgsz_as_resize = _imgsz_to_resize_dims(config_imgsz_legacy)
    if config_resize_dims is not None and config_imgsz_as_resize is not None and config_resize_dims != config_imgsz_as_resize:
        raise ValueError(
            "Conflicting config values: detection.resize_dims and detection.imgsz differ. "
            "Use detection.resize_dims as canonical, or keep them equal."
        )

    requested_resize_dims = (
        cli_resize_dims
        or cli_imgsz_as_resize
        or config_resize_dims
        or config_imgsz_as_resize
    )
    if cli_resize_dims is not None:
        resize_dims_source = "cli:resize_dims"
    elif cli_imgsz_as_resize is not None:
        resize_dims_source = "cli:imgsz"
    elif config_resize_dims is not None:
        resize_dims_source = "config:detection.resize_dims"
    elif config_imgsz_as_resize is not None:
        resize_dims_source = "config:detection.imgsz"
    else:
        resize_dims_source = "none"

    # Get video processing parameters
    video_config = config.get('video', {})
    decode_backend_requested = _normalize_decode_backend(
        decode_backend
        if decode_backend is not None
        else detect_config.get("decode_backend", video_config.get("decode_backend"))
    )
    legacy_video_resize_dims = _normalize_legacy_video_resize(video_config.get("resize"))
    legacy_video_resize_ignored = False
    if requested_resize_dims is not None:
        pre_resize_dims = None
        if legacy_video_resize_dims is not None:
            legacy_video_resize_ignored = True
    else:
        requested_resize_dims = legacy_video_resize_dims
        pre_resize_dims = legacy_video_resize_dims
        if legacy_video_resize_dims is not None:
            resize_dims_source = "config:video.resize"

    imgsz_applied = _resize_dims_to_imgsz(requested_resize_dims)

    source_zarr_config = (
        video_config.get('source_zarr')
        or video_config.get('source_zarr_path')
        or video_config.get('zarr_path')
        or video_config.get('palette_zarr')
    )
    source_zarr_path: Optional[Path] = None
    source_zarr_meta: Dict[str, Any] = {}
    if source_zarr_config:
        source_zarr_path = Path(source_zarr_config).expanduser()
        if source_zarr_path.exists():
            source_zarr_meta = _collect_zarr_metadata(source_zarr_path, console)
        else:
            console.print(
                f"[yellow]Warning:[/yellow] source Zarr path '{source_zarr_path}' not found; continuing without full-resolution metadata."
            )
    
    # GPU/device configuration
    device_config = config.get('model', {}).get('device', 'auto')
    if use_gpu is None:
        use_gpu = device_config != 'cpu'
    
    # Validate inputs
    model_path = Path(model_path).expanduser()
    output_zarr = Path(output_zarr)
    
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    console.print(f"Video: [cyan]{video_path}[/cyan]")
    console.print(f"Model: [cyan]{model_path}[/cyan]")
    console.print(f"Output: [cyan]{output_zarr}[/cyan]")
    
    # Print parameters
    console.print(f"\n[bold]Detection Parameters:[/bold]")
    console.print(f"  Confidence threshold: {conf_threshold}")
    console.print(f"  IoU threshold: {iou_threshold}")
    console.print(f"  Max detections: {max_det}")
    console.print(f"  Batch size: {batch_size}")
    console.print(f"  Decode backend requested: {decode_backend_requested}")
    if requested_resize_dims is not None:
        console.print(
            f"  Resize dims (canonical): {requested_resize_dims[0]}×{requested_resize_dims[1]} "
            f"[{resize_dims_source}]"
        )
    else:
        console.print("  Resize dims (canonical): None")
    if imgsz_applied is not None:
        console.print(f"  YOLO imgsz applied: {imgsz_applied}")
    if pre_resize_dims:
        console.print(
            f"  Legacy frame pre-resize (video.resize): {pre_resize_dims[1]}×{pre_resize_dims[0]}"
        )
    else:
        console.print("  Legacy frame pre-resize: None")
    if legacy_video_resize_ignored:
        console.print(
            "[yellow]  Note:[/yellow] Ignored legacy video.resize because canonical detection resize_dims/imgsz was set."
        )

    # Load model
    console.print("\n[bold]Loading model...[/bold]")
    model = YOLO(str(model_path))
    try:
        model.fuse()
    except AttributeError:
        pass  # Older versions may not expose fuse; Ultralytics will handle it.
    
    # Check device and move model
    model_fp16 = False
    
    if not use_gpu:
        model.to('cpu')
        console.print(f"[green]✓[/green] Model loaded on CPU")
    else:
        import torch
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            model.to('cuda')
            model.model = model.model.to(memory_format=torch.channels_last)
            model.half()
            model_fp16 = True
            console.print(f"[green]✓[/green] Model loaded on GPU: {torch.cuda.get_device_name(0)}")
            console.print(f"[cyan]  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB[/cyan]")
        else:
            console.print(f"[yellow]⚠[/yellow]  CUDA not available, using CPU")
            use_gpu = False

    predict_kwargs: Dict[str, Any] = {
        "conf": conf_threshold,
        "iou": iou_threshold,
        "max_det": max_det,
        "verbose": False,
        "device": 'cuda' if use_gpu else 'cpu',
        "half": model_fp16,
    }
    if imgsz_applied is not None:
        predict_kwargs["imgsz"] = imgsz_applied
    
    # Open video to get metadata
    console.print("\n[bold]Opening video...[/bold]")
    
    decord_info: Optional[Dict[str, Any]] = None
    vr = None
    cap = None
    pynvvc_reader: Optional[PynvvcLumaRgbReader] = None
    use_pynvvc = False
    use_decord = False
    decode_backend_effective = DECODE_BACKEND_OPENCV

    def _open_pynvvc_reader(backend: str) -> PynvvcLumaRgbReader:
        if not use_gpu or not torch.cuda.is_available():
            raise RuntimeError(f"{backend} requires CUDA inference.")
        if requested_resize_dims is None:
            raise RuntimeError(
                f"{backend} requires detection.resize_dims or --resize-dims; "
                "it emits tensor inputs that must be explicitly resized before YOLO."
            )
        return PynvvcLumaRgbReader(video_path, start_frame=0, gpu_id=0)

    auto_pynvvc_candidate = decode_backend_requested == DECODE_BACKEND_AUTO
    if decode_backend_requested in PYNVVC_BACKENDS or auto_pynvvc_candidate:
        pynvvc_backend = (
            BACKEND_PYNVVC_LUMA_RGB
            if decode_backend_requested == DECODE_BACKEND_AUTO
            else decode_backend_requested
        )
        try:
            pynvvc_reader = _open_pynvvc_reader(pynvvc_backend)
            n_frames, fps_cv2, width_cv2, height_cv2 = _read_cv2_video_properties(video_path)
            width = int(pynvvc_reader.source_width or width_cv2)
            height = int(pynvvc_reader.source_height or height_cv2)
            fps = float(pynvvc_reader.frame_rate or fps_cv2)
            if n_frames <= 0:
                raise RuntimeError("OpenCV metadata did not report a positive frame count.")
            video_reader_type = pynvvc_backend
            decode_backend_effective = pynvvc_backend
            use_pynvvc = True
            console.print(f"[green]✓[/green] Using PyNvVideoCodec {pynvvc_backend} CUDA decoder")
        except Exception as exc:
            if pynvvc_reader is not None:
                pynvvc_reader.close()
                pynvvc_reader = None
            if decode_backend_requested != DECODE_BACKEND_AUTO:
                raise
            raise _gpu_decode_unavailable(
                f"auto requires PyNvVideoCodec {pynvvc_backend}; initialization failed: {exc}"
            ) from exc

    if not use_pynvvc and decode_backend_requested != DECODE_BACKEND_OPENCV:
        if decode_backend_requested == DECODE_BACKEND_AUTO:
            raise _gpu_decode_unavailable("auto did not initialize PyNvVideoCodec luma decode")
        prefer_decord_gpu = decode_backend_requested == DECODE_BACKEND_DECORD_GPU
        decord_info = _init_decord_reader(video_path, prefer_gpu=prefer_decord_gpu, console=console)
        if (
            decode_backend_requested == DECODE_BACKEND_DECORD_GPU
            and decord_info is not None
            and not bool(decord_info.get("on_gpu"))
        ):
            raise RuntimeError("Requested decord_gpu, but Decord fell back to CPU.")
        if decord_info is None and decode_backend_requested in {
            DECODE_BACKEND_DECORD_GPU,
            DECODE_BACKEND_DECORD_CPU,
        }:
            raise RuntimeError(f"Requested {decode_backend_requested}, but Decord is unavailable.")

    if not use_pynvvc and decord_info:
        vr = decord_info['reader']
        n_frames = len(vr)
        width = decord_info['width']
        height = decord_info['height']
        fps = decord_info['fps']
        video_reader_type = decord_info['type']
        decode_backend_effective = str(decord_info['type'])
        use_decord = True
    elif not use_pynvvc:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        console.print(f"[green]✓[/green] Using OpenCV decoder")
        video_reader_type = 'opencv'
        decode_backend_effective = DECODE_BACKEND_OPENCV

    decord_on_gpu = bool(decord_info and decord_info.get('on_gpu'))
    tensor_input_path = bool(decord_on_gpu or use_pynvvc)
    effective_input_resize_dims = _actual_input_resize_dims(
        requested_resize_dims,
        pre_resize_dims,
        decord_on_gpu=decord_on_gpu,
        tensor_input=use_pynvvc,
    )
    
    # Determine dimensions for normalization (actual or resized)
    if effective_input_resize_dims:
        inference_height, inference_width = effective_input_resize_dims
        console.print(f"[green]✓[/green] Video: {n_frames} frames, {fps:.1f} fps, {width}×{height}")
        console.print(f"[cyan]  Will feed {inference_width}×{inference_height} frames to YOLO[/cyan]")
        if tensor_input_path and requested_resize_dims is not None:
            console.print("[cyan]  Tensor decoder path: applying canonical resize before predict[/cyan]")
    else:
        inference_width, inference_height = width, height
        console.print(f"[green]✓[/green] Video: {n_frames} frames, {fps:.1f} fps, {width}×{height}")
    
    console.print("\n[bold]Preparing output Zarr...[/bold]")
    if output_zarr.exists():
        root = zarr.open_group(str(output_zarr), mode='r+')
        created_new_root = False
        console.print(f"[cyan]Appending detect run to existing archive:[/cyan] {output_zarr}")
    else:
        root = zarr.open_group(str(output_zarr), mode='w')
        created_new_root = True
        console.print(f"[cyan]Created new detection archive:[/cyan] {output_zarr}")

    source_video_width = int(width)
    source_video_height = int(height)
    source_full_width = int(source_zarr_meta.get("full_width", source_video_width))
    source_full_height = int(source_zarr_meta.get("full_height", source_video_height))

    # Get comprehensive video metadata
    vid_meta = get_video_metadata(video_path, cap, width, height, n_frames, fps)
    
    # Get git info for reproducibility (matching import_video.py)
    git_info = get_git_info()
    
    # Get full environment info for provenance
    env_info = get_environment_info(
        include_all_packages=False,
        disk_path=str(output_zarr),
        collect_ip=False
    )
    
    # Always keep core video metadata on the root so downstream consumers (and diagnostics) can read it.
    root.attrs.update({
        'source_video': vid_meta['source_video'],
        'source_video_path': vid_meta['source_path'],
        'source_path': vid_meta['source_path'],
        'video_width': int(width),
        'video_height': int(height),
        'width': int(width),
        'height': int(height),
        'fps': float(fps) if fps and fps > 0 else fps,
        'n_frames': int(n_frames),
        'total_frames': int(n_frames),
        'duration_seconds': float(vid_meta['duration_seconds']),
        'video_codec': vid_meta.get('codec', root.attrs.get('video_codec', 'unknown')),
        'video_pix_fmt': vid_meta.get('pix_fmt', root.attrs.get('video_pix_fmt', 'unknown')),
    })

    if created_new_root:
        root.attrs.update({
            # Video source info
            'source_video': vid_meta['source_video'],
            'source_video_path': vid_meta['source_path'],
            'source_path': vid_meta['source_path'],  # Alias for compatibility
            
            # Video properties
            'video_width': width,
            'video_height': height,
            'width': width,  # Alias
            'height': height,  # Alias
            'fps': fps,
            'n_frames': n_frames,
            'total_frames': n_frames,  # Alias
            'duration_seconds': vid_meta['duration_seconds'],
            
            # Codec info
            'video_codec': vid_meta.get('codec', 'unknown'),
            'video_pix_fmt': vid_meta.get('pix_fmt', 'unknown'),
            
            # Pipeline info
            'created_at_utc': datetime.now(timezone.utc).isoformat(),
            'pipeline_type': 'yolo_inference',
            'zarr_purpose': 'production',
            'has_raw_video': False,
            'detection_method': 'yolo',
            
            # Model info
            'model_path': str(model_path.absolute()),
            'model_name': model_path.name,
            
            # Processing info
            'inference_width': inference_width,
            'inference_height': inference_height,
            'resized_for_inference': effective_input_resize_dims is not None,
            
            # Git provenance
            'git_commit_hash': git_info.get('commit_hash', 'unknown'),
            'git_short_hash': git_info.get('short_hash', 'unknown'),
            'git_branch': git_info.get('branch', 'unknown'),
            'git_is_dirty': git_info.get('is_dirty', False),
            'git_remote_url': git_info.get('remote_url', 'unknown'),
            
            # System provenance
            'system_hostname': env_info['platform']['hostname'],
            'system_fqdn': env_info['platform']['fqdn'],
            'system_os': env_info['platform']['system'],
            'system_os_release': env_info['platform']['release'],
            'system_machine': env_info['platform']['machine'],
            'system_python_version': env_info['platform']['python_version'],
            'system_username': env_info['platform']['username'],
            'system_cpu_cores': env_info['platform']['cpu_cores'],
        })

        if 'imageio_metadata' in vid_meta:
            root.attrs['imageio_metadata'] = vid_meta['imageio_metadata']

        if 'cpu_details' in env_info['platform']:
            cpu = env_info['platform']['cpu_details']
            root.attrs.update({
                'cpu_model': cpu.get('model', 'unknown'),
                'cpu_arch': cpu.get('arch', 'unknown'),
            })

        if 'memory' in env_info['platform']:
            mem = env_info['platform']['memory']
            root.attrs.update({
                'memory_total_gb': mem.get('total_gb', 0),
                'memory_available_gb': mem.get('available_gb', 0),
                'memory_percent_used': mem.get('percent_used', 0),
            })

        if 'disk' in env_info['platform']:
            disk = env_info['platform']['disk']
            root.attrs.update({
                'disk_path': disk.get('path', str(output_zarr)),
                'disk_total_gb': disk.get('total_gb', 0),
                'disk_available_gb': disk.get('available_gb', 0),
                'disk_percent_used': disk.get('percent_used', 0),
            })

        if 'lsf' in env_info['platform']:
            lsf = env_info['platform']['lsf']
            root.attrs.update({
                'hpc_scheduler': 'LSF',
                'lsf_job_id': lsf.get('job_id', 'unknown'),
                'lsf_job_name': lsf.get('job_name', 'unknown'),
                'lsf_queue': lsf.get('queue', 'unknown'),
                'lsf_hosts': lsf.get('hosts', 'unknown'),
            })
        elif 'slurm' in env_info['platform']:
            slurm = env_info['platform']['slurm']
            root.attrs.update({
                'hpc_scheduler': 'SLURM',
                'slurm_job_id': slurm.get('job_id', 'unknown'),
                'slurm_job_name': slurm.get('job_name', 'unknown'),
                'slurm_node_list': slurm.get('node_list', 'unknown'),
            })

        if env_info.get('gpu', {}).get('available'):
            gpu_info = env_info['gpu']
            root.attrs.update({
                'gpu_available': True,
                'gpu_backend': gpu_info.get('backend', 'unknown'),
                'gpu_count': len(gpu_info.get('devices', [])),
            })
            if 'cuda_version' in gpu_info:
                root.attrs['cuda_version'] = gpu_info['cuda_version']
            if gpu_info.get('devices'):
                primary_gpu = gpu_info['devices'][0]
                root.attrs.update({
                    'gpu_name': primary_gpu.get('name', 'unknown'),
                    'gpu_compute_capability': primary_gpu.get('compute_capability', 'unknown'),
                    'gpu_memory_total_gb': primary_gpu.get('total_memory_gb', 0),
                })

        env_summary = env_info.get('environment', {})
        if env_summary:
            root.attrs.update({
                'environment_type': env_summary.get('environment_type', 'unknown'),
                'environment_name': env_summary.get('environment_name', 'unknown'),
                'total_packages': env_summary.get('total_packages', 0),
            })
            if 'deep_learning_framework' in env_summary:
                root.attrs['deep_learning_framework'] = env_summary['deep_learning_framework']
            if 'key_packages' in env_summary:
                import json
                root.attrs['key_packages_json'] = json.dumps(env_summary['key_packages'])

        import json
        root.attrs['_full_environment_info'] = json.dumps(env_info, default=str)

    if source_zarr_meta:
        palette_attr_map = {
            "fps": "palette_fps",
            "total_frames": "palette_total_frames",
            "duration_seconds": "palette_duration_seconds",
            "video_codec": "palette_video_codec",
            "video_pix_fmt": "palette_video_pix_fmt",
            "downsampled_resolution": "palette_downsampled_resolution",
            "downsample_method": "palette_downsample_method",
            "has_full_resolution": "palette_has_full_resolution",
            "has_downsampled": "palette_has_downsampled",
            "original_resolution": "palette_original_resolution",
        }
        for key, attr_name in palette_attr_map.items():
            value = source_zarr_meta.get(key)
            if value is None:
                continue
            if isinstance(value, tuple):
                value = list(value)
            root.attrs[attr_name] = value
        if source_zarr_meta.get("full_width") is not None:
            root.attrs["palette_video_width"] = int(source_zarr_meta["full_width"])
        if source_zarr_meta.get("full_height") is not None:
            root.attrs["palette_video_height"] = int(source_zarr_meta["full_height"])

    root.attrs['source_video_width'] = source_video_width
    root.attrs['source_video_height'] = source_video_height
    root.attrs['source_video_resolution'] = [source_video_width, source_video_height]
    root.attrs['source_full_width'] = source_full_width
    root.attrs['source_full_height'] = source_full_height
    root.attrs['source_full_resolution'] = [source_full_width, source_full_height]
    root.attrs['inference_resolution'] = [int(inference_width), int(inference_height)]
    root.attrs['inference_width'] = int(inference_width)
    root.attrs['inference_height'] = int(inference_height)
    root.attrs['resized_for_inference'] = effective_input_resize_dims is not None
    root.attrs['resize_dims_requested'] = (
        [int(requested_resize_dims[0]), int(requested_resize_dims[1])]
        if requested_resize_dims is not None
        else None
    )
    root.attrs['resize_dims_source'] = resize_dims_source
    root.attrs['imgsz_applied'] = imgsz_applied
    root.attrs['imgsz_legacy_input'] = cli_imgsz_legacy
    root.attrs['pre_resize_dims'] = (
        [int(pre_resize_dims[0]), int(pre_resize_dims[1])]
        if pre_resize_dims is not None
        else None
    )
    root.attrs['effective_input_resize_dims'] = (
        [int(effective_input_resize_dims[0]), int(effective_input_resize_dims[1])]
        if effective_input_resize_dims is not None
        else None
    )
    root.attrs['tensor_resize_dims'] = (
        [int(effective_input_resize_dims[0]), int(effective_input_resize_dims[1])]
        if tensor_input_path and effective_input_resize_dims is not None
        else None
    )
    root.attrs['decode_backend_requested'] = decode_backend_requested
    root.attrs['decode_backend_effective'] = decode_backend_effective
    root.attrs['video_reader_type'] = video_reader_type
    if source_zarr_path is not None:
        root.attrs['source_zarr_path'] = str(source_zarr_path)

    if write_raw_video_metadata:
        updates = write_video_metadata(
            root,
            vid_meta,
            overwrite=overwrite_raw_video_metadata,
            import_purpose="production",
        )
        raw_updates = updates.get("raw_video", {})
        root_updates = updates.get("root", {})
        if raw_updates or root_updates:
            console.print(
                "[green]✓[/green] Wrote metadata-only attrs "
                f"(root={len(root_updates)}, raw_video={len(raw_updates)})"
            )
    
    if run_name is not None:
        run_name = str(run_name).strip()
        if not run_name or "/" in run_name or run_name in {".", ".."}:
            raise ValueError(f"Invalid detect run name: {run_name!r}")
        parent_group = require_runs_parent(root, "detect_runs")
        if run_name in parent_group:
            raise ValueError(f"detect_runs/{run_name} already exists")
        detect_group = parent_group.create_group(run_name)
        mark_run_started(detect_group, run_name=run_name, stage="detect")
        note_pending_latest(parent_group, run_name)
        console.print(f"Created run group: [cyan]detect_runs/{run_name}[/cyan]")
    else:
        detect_group, run_name = get_run_group(root, 'detect', console, create_new=True)
        parent_group = root["detect_runs"]
        mark_run_started(detect_group, run_name=run_name, stage="detect")
        note_pending_latest(parent_group, run_name)
    console.print(f"[green]✓[/green] Writing detections to detect_runs/{run_name}")
    
    # Storage for detections
    frame_counts = np.zeros(n_frames, dtype=np.int32)
    batch_results = []

    def accumulate_results(result, global_frame_idx: int) -> None:
        """Vectorize detection accumulation for a single frame."""
        if result.boxes is None or len(result.boxes) == 0:
            return

        boxes_xyxy = result.boxes.xyxy.detach().cpu().numpy()
        scores_np = result.boxes.conf.detach().cpu().numpy()
        num_detections = boxes_xyxy.shape[0]
        if num_detections == 0:
            return

        cx = (boxes_xyxy[:, 0] + boxes_xyxy[:, 2]) * 0.5 / inference_width
        cy = (boxes_xyxy[:, 1] + boxes_xyxy[:, 3]) * 0.5 / inference_height
        w = (boxes_xyxy[:, 2] - boxes_xyxy[:, 0]) / inference_width
        h = (boxes_xyxy[:, 3] - boxes_xyxy[:, 1]) / inference_height

        bbox_norm = np.column_stack((cx, cy, w, h)).astype(np.float64, copy=False)
        indices = np.full(num_detections, global_frame_idx, dtype=np.int32)
        scores = scores_np.astype(np.float32, copy=False)

        cls_tensor = result.boxes.cls
        if cls_tensor is None:
            class_ids = np.zeros(num_detections, dtype=np.int32)
        else:
            class_ids = cls_tensor.detach().cpu().numpy().astype(np.int32, copy=False)

        batch_results.append((indices, bbox_norm, scores, class_ids))
        frame_counts[global_frame_idx] += num_detections
    
    # Process video in batches
    frame_idx = 0
    batch_frames = []
    batch_indices = []
    
    console.print("\n[bold]Running inference...[/bold]")
    console.print(f"[cyan]Decoder: {video_reader_type}[/cyan]")
    
    # Performance tracking
    inference_times = []
    read_times = []
    stage_timings = {
        'read_decode_seconds_total': 0.0,
        'preprocess_resize_seconds_total': 0.0,
        'predict_seconds_total': 0.0,
        'postprocess_seconds_total': 0.0,
        'array_assembly_seconds_total': 0.0,
        'zarr_write_seconds_total': 0.0,
    }
    processing_start = time.time()
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("•"),
        TextColumn("[cyan]{task.fields[fps]:.1f} fps"),
        TimeRemainingColumn(),
        console=console
    ) as progress:
        task = progress.add_task("Processing frames", total=n_frames, fps=0.0)
        
        batch_count = 0
        
        if use_pynvvc:
            if pynvvc_reader is None:
                raise RuntimeError("PyNvVideoCodec reader was not initialized.")
            if effective_input_resize_dims is None:
                raise RuntimeError(f"{decode_backend_effective} requires a resolved tensor resize.")
            device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
            dtype = torch.float16 if model_fp16 else torch.float32
            frame_iter = pynvvc_reader.iter_frames()

            while frame_idx < n_frames:
                max_batch_frames = min(int(batch_size), int(n_frames - frame_idx))
                processed, actual_count, read_elapsed, preprocess_elapsed = (
                    _read_and_preprocess_pynvvc_batch(
                        frame_iter=frame_iter,
                        max_batch_frames=max_batch_frames,
                        decode_backend_effective=decode_backend_effective,
                        source_height=pynvvc_reader.source_height,
                        device=device,
                        dtype=dtype,
                        resize_hw=effective_input_resize_dims,
                    )
                )
                stage_timings['read_decode_seconds_total'] += float(read_elapsed)
                stage_timings['preprocess_resize_seconds_total'] += float(preprocess_elapsed)
                read_times.append(read_elapsed)

                if actual_count <= 0 or processed is None:
                    break
                actual_indices = list(range(frame_idx, frame_idx + actual_count))

                inference_start = time.perf_counter()
                with torch.inference_mode():
                    results = model.predict(processed, **predict_kwargs)
                inference_elapsed = _record_timing(
                    stage_timings, 'predict_seconds_total', inference_start
                )
                inference_times.append(inference_elapsed)

                postprocess_start = time.perf_counter()
                for batch_i, result in enumerate(results):
                    accumulate_results(result, actual_indices[batch_i])
                _record_timing(
                    stage_timings, 'postprocess_seconds_total', postprocess_start
                )

                del results
                del processed

                frame_idx += actual_count
                batch_count += 1

                elapsed = time.time() - processing_start
                current_fps = frame_idx / elapsed if elapsed > 0 else 0
                progress.update(task, advance=actual_count, fps=current_fps)

                if batch_count % 100 == 0:
                    avg_inference = np.mean(inference_times[-100:])
                    avg_read = np.mean(read_times[-100:])
                    console.print(f"[dim]Batch {batch_count}: read={avg_read*1000:.1f}ms, "
                                f"inference={avg_inference*1000:.1f}ms, fps={current_fps:.1f}[/dim]")

                if actual_count < max_batch_frames:
                    break

        elif use_decord:
            # Double-buffered decode: prefetch next batch while current inference runs.
            batch_starts = list(range(0, n_frames, batch_size))
            prefetched = None
            
            for idx, batch_start in enumerate(batch_starts):
                batch_end = min(batch_start + batch_size, n_frames)
                indices = list(range(batch_start, batch_end))
                
                if prefetched is None:
                    read_start = time.perf_counter()
                    current_batch = vr.get_batch(indices)
                    read_elapsed = _record_timing(
                        stage_timings, 'read_decode_seconds_total', read_start
                    )
                    read_times.append(read_elapsed)
                else:
                    current_batch = prefetched
                
                next_indices = None
                prefetched = None
                if idx + 1 < len(batch_starts):
                    next_start = batch_starts[idx + 1]
                    next_end = min(next_start + batch_size, n_frames)
                    next_indices = list(range(next_start, next_end))
                    if not decord_on_gpu or torch.cuda.is_available():
                        prefetch_start = time.perf_counter()
                        prefetched = vr.get_batch(next_indices)
                        prefetch_elapsed = _record_timing(
                            stage_timings, 'read_decode_seconds_total', prefetch_start
                        )
                        read_times.append(prefetch_elapsed)
                
                if decord_on_gpu:
                    import torch.nn.functional as F
                    device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
                    dtype = torch.float16 if model_fp16 else torch.float32
                    frames_chw = current_batch.permute(0, 3, 1, 2).contiguous()  # [B, C, H, W] uint8
                    
                    total = frames_chw.shape[0]
                    chunk_size = total
                    results = []
                    start = 0
                    
                    while start < total:
                        end = min(start + chunk_size, total)
                        chunk = frames_chw[start:end]
                        try:
                            preprocess_start = time.perf_counter()
                            chunk = chunk.to(device=device, dtype=dtype, non_blocking=True).contiguous(memory_format=torch.channels_last)
                            if effective_input_resize_dims:
                                chunk = F.interpolate(
                                    chunk,
                                    size=effective_input_resize_dims,
                                    mode='bilinear',
                                    align_corners=False
                                )
                            chunk = chunk.mul_(1.0 / 255.0)
                            _record_timing(
                                stage_timings,
                                'preprocess_resize_seconds_total',
                                preprocess_start,
                            )
                            
                            inference_start = time.perf_counter()
                            preds = model.predict(chunk, **predict_kwargs)
                            inference_elapsed = _record_timing(
                                stage_timings, 'predict_seconds_total', inference_start
                            )
                            inference_times.append(inference_elapsed)
                            results.extend(preds)
                            start = end
                        except torch.cuda.OutOfMemoryError:
                            torch.cuda.empty_cache()
                            if chunk_size == 1:
                                raise
                            chunk_size = max(1, chunk_size // 2)
                            continue
                        finally:
                            del chunk
                    
                    del frames_chw
                else:
                    frames_nd = current_batch.asnumpy() if hasattr(current_batch, "asnumpy") else np.asarray(current_batch)
                    preprocess_start = time.perf_counter()
                    if pre_resize_dims:
                        batch_frames_np = [
                            cv2.resize(frame, (int(pre_resize_dims[1]), int(pre_resize_dims[0]))) for frame in frames_nd
                        ]
                    else:
                        batch_frames_np = [np.asarray(frame) for frame in frames_nd]
                    _record_timing(
                        stage_timings, 'preprocess_resize_seconds_total', preprocess_start
                    )
                    del frames_nd
                
                    inference_start = time.perf_counter()
                    results = model.predict(batch_frames_np, **predict_kwargs)
                    inference_elapsed = _record_timing(
                        stage_timings, 'predict_seconds_total', inference_start
                    )
                    inference_times.append(inference_elapsed)
                
                postprocess_start = time.perf_counter()
                for batch_i, result in enumerate(results):
                    accumulate_results(result, indices[batch_i])
                _record_timing(
                    stage_timings, 'postprocess_seconds_total', postprocess_start
                )
                
                del current_batch
                if not decord_on_gpu:
                    del batch_frames_np
                
                frame_idx += len(indices)
                batch_count += 1
                
                elapsed = time.time() - processing_start
                current_fps = frame_idx / elapsed if elapsed > 0 else 0
                progress.update(task, advance=len(indices), fps=current_fps)
                
                if batch_count % 100 == 0:
                    avg_inference = np.mean(inference_times[-100:])
                    avg_read = np.mean(read_times[-100:])
                    console.print(f"[dim]Batch {batch_count}: read={avg_read*1000:.1f}ms, "
                                f"inference={avg_inference*1000:.1f}ms, fps={current_fps:.1f}[/dim]")
            
            if prefetched is not None:
                del prefetched
        
        else:
            # OpenCV frame-by-frame processing
            while True:
                # Time frame reading
                read_start = time.perf_counter()
                ret, frame = cap.read()
                read_elapsed = _record_timing(
                    stage_timings, 'read_decode_seconds_total', read_start
                )
                read_times.append(read_elapsed)
                
                if not ret:
                    break
                
                preprocess_start = time.perf_counter()
                # Resize if specified
                if pre_resize_dims:
                    frame = cv2.resize(frame, (int(pre_resize_dims[1]), int(pre_resize_dims[0])))
                
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                _record_timing(
                    stage_timings, 'preprocess_resize_seconds_total', preprocess_start
                )
                batch_frames.append(frame_rgb)
                batch_indices.append(frame_idx)
                frame_idx += 1
                
                # Process batch when full or at end
                if len(batch_frames) == batch_size or frame_idx == n_frames:
                    # Time inference
                    inference_start = time.perf_counter()
                    
                    # Run inference
                    results = model.predict(batch_frames, **predict_kwargs)
                    
                    inference_time = _record_timing(
                        stage_timings, 'predict_seconds_total', inference_start
                    )
                    inference_times.append(inference_time)
                    
                    # Calculate FPS
                    elapsed = time.time() - processing_start
                    current_fps = frame_idx / elapsed if elapsed > 0 else 0
                    
                    # Extract detections
                    postprocess_start = time.perf_counter()
                    for batch_i, result in enumerate(results):
                        accumulate_results(result, batch_indices[batch_i])
                    _record_timing(
                        stage_timings, 'postprocess_seconds_total', postprocess_start
                    )
                    
                    batch_size_actual = len(batch_frames)
                    
                    # Clear batch
                    batch_frames = []
                    batch_indices = []
                    batch_count += 1
                    
                    # Update progress
                    progress.update(task, advance=batch_size_actual, fps=current_fps)
                    
                    # Print diagnostics every 100 batches
                    if batch_count % 100 == 0:
                        avg_inference = np.mean(inference_times[-100:]) if len(inference_times) > 0 else 0
                        avg_read = np.mean(read_times[-100:]) if len(read_times) > 0 else 0
                        console.print(f"[dim]Batch {batch_count}: inference={avg_inference*1000:.1f}ms, "
                                    f"read={avg_read*1000:.1f}ms, fps={current_fps:.1f}[/dim]")
            
            cap.release()

    if use_pynvvc and pynvvc_reader is not None:
        pynvvc_reader.close()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if use_decord and vr is not None:
        del vr
        if decord_on_gpu and torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    total_time = time.time() - processing_start
    console.print(f"[green]✓[/green] Inference complete")
    console.print(f"[cyan]  Total time: {total_time:.1f}s ({total_time/60:.1f} min)[/cyan]")
    console.print(f"[cyan]  Average FPS: {n_frames/total_time:.1f}[/cyan]")
    
    if len(inference_times) > 0:
        avg_inference = np.mean(inference_times)
        console.print(f"[cyan]  Avg inference time per batch: {avg_inference*1000:.1f}ms[/cyan]")
        console.print(f"[cyan]  Avg read time per batch: {np.mean(read_times)*1000:.1f}ms[/cyan]")
    else:
        avg_inference = 0.0
    
    # Convert to arrays
    console.print("\n[bold]Saving detections to zarr...[/bold]")
    assembly_start = time.perf_counter()
    if batch_results:
        frame_indices = np.concatenate([res[0] for res in batch_results])
        bbox_coords = np.concatenate([res[1] for res in batch_results])
        scores = np.concatenate([res[2] for res in batch_results])
        class_ids = np.concatenate([res[3] for res in batch_results]).astype(np.int32, copy=False)
    else:
        frame_indices = np.empty((0,), dtype=np.int32)
        bbox_coords = np.empty((0, 4), dtype=np.float64)
        scores = np.empty((0,), dtype=np.float32)
        class_ids = np.empty((0,), dtype=np.int32)
    
    total_detections = frame_indices.size
    
    if total_detections > 0:
        frame_counts = np.bincount(frame_indices, minlength=n_frames).astype(np.int32, copy=False)
    else:
        frame_counts = np.zeros(n_frames, dtype=np.int32)
    _record_timing(stage_timings, 'array_assembly_seconds_total', assembly_start)
    
    preferred_det_chunk = max(1024, max(1, batch_size) * 8)
    det_chunk = max(1, min(max(1, total_detections), preferred_det_chunk, 16384))
    
    # Save to zarr
    zarr_write_start = time.perf_counter()
    detect_group.create_array('frame_indices', data=frame_indices, chunks=(det_chunk,), overwrite=True)
    detect_group.create_array('bbox_norm_coords', data=bbox_coords, chunks=(det_chunk, 4), overwrite=True)
    detect_group.create_array('scores', data=scores, chunks=(det_chunk,), overwrite=True)
    detect_group.create_array('class_ids', data=class_ids, chunks=(det_chunk,), overwrite=True)
    create_geometry_preload_array(detect_group, 'n_detections', data=frame_counts, overwrite=True)
    create_geometry_preload_array(detect_group, 'frame_counts', data=frame_counts, overwrite=True)
    _record_timing(stage_timings, 'zarr_write_seconds_total', zarr_write_start)
    
    # Calculate statistics
    frames_with_detections = np.sum(frame_counts > 0)
    coverage_percent = (frames_with_detections / n_frames) * 100
    
    stats = {
        'total_detections': int(total_detections),
        'frames_with_detections': int(frames_with_detections),
        'percent_frames_with_detections': float(coverage_percent),
        'frames_with_zero_detections': int(np.sum(frame_counts == 0)),
        'frames_with_multiple_detections': int(np.sum(frame_counts > 1)),
        'mean_detections_per_frame': float(total_detections / n_frames),
        'mean_confidence': float(np.mean(scores)) if len(scores) > 0 else 0.0,
        'min_confidence': float(np.min(scores)) if len(scores) > 0 else 0.0,
        'max_confidence': float(np.max(scores)) if len(scores) > 0 else 0.0,
    }
    timing_summary = {
        'schema_version': 1,
        'timing_policy': 'wall_clock_no_per_batch_cuda_sync',
        'decode_backend_requested': decode_backend_requested,
        'decode_backend_effective': decode_backend_effective,
        'pynvvc_surface_materialization': (
            PYNVVC_SURFACE_MATERIALIZATION if use_pynvvc else 'not_applicable'
        ),
        'video_reader_type': video_reader_type,
        'frames_processed': int(frame_idx),
        'batches_processed': int(len(inference_times)),
        'processing_seconds_total': float(total_time),
        'processing_fps': float(frame_idx / total_time) if total_time > 0 else 0.0,
        'read_decode_seconds_total': float(stage_timings['read_decode_seconds_total']),
        'preprocess_resize_seconds_total': float(stage_timings['preprocess_resize_seconds_total']),
        'predict_seconds_total': float(stage_timings['predict_seconds_total']),
        'postprocess_seconds_total': float(stage_timings['postprocess_seconds_total']),
        'array_assembly_seconds_total': float(stage_timings['array_assembly_seconds_total']),
        'zarr_write_seconds_total': float(stage_timings['zarr_write_seconds_total']),
        'predict_avg_batch_ms': float(avg_inference * 1000.0) if inference_times else 0.0,
        'read_decode_avg_batch_ms': float(np.mean(read_times) * 1000.0) if read_times else 0.0,
    }
    
    # Store metadata
    detect_group.attrs.update({
        'detect_timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'detection_method': 'yolo',  # 'blob' for traditional, 'yolo' for neural net
        'detection_source': 'external_video',
        'model_type': 'yolo_object_detection',
        'model_path': str(model_path.absolute()),
        'model_name': model_path.name,
        'source_video_width': source_video_width,
        'source_video_height': source_video_height,
        'source_full_width': source_full_width,
        'source_full_height': source_full_height,
        'inference_width': int(inference_width),
        'inference_height': int(inference_height),
        'parameters': {
            'conf_threshold': conf_threshold,
            'iou_threshold': iou_threshold,
            'max_det': max_det,
            'batch_size': batch_size,
            'resize_dims': requested_resize_dims,
            'resize_dims_source': resize_dims_source,
            'imgsz': imgsz_applied,
            'imgsz_applied': imgsz_applied,
            'imgsz_legacy_input': cli_imgsz_legacy,
            'pre_resize_dims': pre_resize_dims,
            'effective_input_resize_dims': effective_input_resize_dims,
            'tensor_resize_dims': (
                effective_input_resize_dims
                if tensor_input_path and effective_input_resize_dims is not None
                else None
            ),
            'legacy_video_resize_dims': legacy_video_resize_dims,
            'decode_backend_requested': decode_backend_requested,
            'decode_backend_effective': decode_backend_effective,
            'pynvvc_surface_materialization': (
                PYNVVC_SURFACE_MATERIALIZATION if use_pynvvc else 'not_applicable'
            ),
        },
        'summary_statistics': stats,
        'timing_summary': timing_summary,
        'git_commit': git_info.get('commit_hash', 'unknown'),
        'git_branch': git_info.get('branch', 'unknown'),
        'hostname': env_info['platform']['hostname']
    })
    if source_zarr_meta:
        if source_zarr_meta.get("downsampled_resolution") is not None:
            detect_group.attrs['palette_downsampled_resolution'] = list(source_zarr_meta["downsampled_resolution"])
        if source_zarr_meta.get("has_downsampled") is not None:
            detect_group.attrs['palette_has_downsampled'] = bool(source_zarr_meta["has_downsampled"])
        if source_zarr_meta.get("has_full_resolution") is not None:
            detect_group.attrs['palette_has_full_resolution'] = bool(source_zarr_meta["has_full_resolution"])
        if source_zarr_meta.get("original_resolution") is not None:
            detect_group.attrs['palette_original_resolution'] = list(source_zarr_meta["original_resolution"])
    if source_zarr_path is not None:
        detect_group.attrs['source_zarr_path'] = str(source_zarr_path)
    detect_group.attrs['inference_duration_seconds'] = float(total_time)
    detect_group.attrs['inference_average_fps'] = float(n_frames / total_time) if total_time > 0 else 0.0
    detect_group.attrs['inference_avg_batch_ms'] = float(avg_inference * 1000.0) if inference_times else 0.0
    detect_group.attrs['inference_avg_read_ms'] = float(np.mean(read_times) * 1000.0) if read_times else 0.0
    detect_group.attrs['read_decode_seconds_total'] = timing_summary['read_decode_seconds_total']
    detect_group.attrs['preprocess_resize_seconds_total'] = timing_summary['preprocess_resize_seconds_total']
    detect_group.attrs['predict_seconds_total'] = timing_summary['predict_seconds_total']
    detect_group.attrs['postprocess_seconds_total'] = timing_summary['postprocess_seconds_total']
    detect_group.attrs['array_assembly_seconds_total'] = timing_summary['array_assembly_seconds_total']
    detect_group.attrs['zarr_write_seconds_total'] = timing_summary['zarr_write_seconds_total']
    detect_group.attrs['decode_backend_requested'] = decode_backend_requested
    detect_group.attrs['decode_backend_effective'] = decode_backend_effective
    detect_group.attrs['video_reader_type'] = video_reader_type

    platform_info = env_info.get("platform") or {}
    scheduler_info = platform_info.get("lsf") or platform_info.get("slurm")
    provenance_record = build_stage_provenance(
        stage="detect",
        command=" ".join(sys.argv),
        created_at_utc=str(detect_group.attrs.get("detect_timestamp_utc")),
        version=git_info.get("short_hash") or git_info.get("commit_hash"),
        git={
            "commit": git_info.get("commit_hash"),
            "short": git_info.get("short_hash"),
            "branch": git_info.get("branch"),
            "is_dirty": git_info.get("is_dirty"),
            "remote": git_info.get("remote_url"),
        },
        environment=env_info.get("environment"),
        platform={
            "hostname": platform_info.get("hostname"),
            "fqdn": platform_info.get("fqdn"),
            "system": platform_info.get("system"),
            "release": platform_info.get("release"),
            "version": platform_info.get("version"),
            "machine": platform_info.get("machine"),
            "processor": platform_info.get("processor"),
            "cpu_cores": platform_info.get("cpu_cores"),
            "cpu_details": platform_info.get("cpu_details"),
            "memory": platform_info.get("memory"),
            "disk": platform_info.get("disk"),
            "python_version": platform_info.get("python_version"),
            "python_implementation": platform_info.get("python_implementation"),
        },
        scheduler=scheduler_info,
        parameters=dict(detect_group.attrs.get("parameters") or {}),
        inputs={
            "frame_source": "external",
            "source_video_path": root.attrs.get("source_video_path"),
            "source_zarr_path": detect_group.attrs.get("source_zarr_path"),
            "decode_backend_requested": decode_backend_requested,
            "decode_backend_effective": decode_backend_effective,
        },
        artifacts={
            "model_path": str(model_path.absolute()),
            "model_name": model_path.name,
            "device": "cuda" if use_gpu else "cpu",
            "gpu": env_info.get("gpu"),
        },
    )
    provenance_record["timing"] = dict(timing_summary)
    if cli_provenance is not None:
        detect_group.attrs["cli_provenance"] = dict(cli_provenance)
    write_stage_provenance(detect_group, provenance_record)
    
    mark_run_complete(detect_group, parent_group=parent_group, run_name=run_name)
    
    console.print(f"[green]✓[/green] Detections saved")
    
    # Calculate storage savings
    zarr_size_mb = (total_detections * 32) / 1024 / 1024  # Rough estimate
    video_size_mb = (n_frames * width * height) / 1024 / 1024  # If we stored grayscale
    if zarr_size_mb > 0:
        storage_comparison_line = f"  Saved vs full import: ~{video_size_mb:.1f} MB ({video_size_mb/zarr_size_mb:.0f}× smaller)"
    else:
        storage_comparison_line = f"  Saved vs full import: ~{video_size_mb:.1f} MB"
    
    # Print summary
    summary_text = f"""[green]✓[/green] Inference complete!

[bold]Results:[/bold]
  Detections: {total_detections:,}
  Coverage: {coverage_percent:.1f}% ({frames_with_detections:,}/{n_frames:,} frames)
  Mean confidence: {stats['mean_confidence']:.3f}

[bold]Storage:[/bold]
  Zarr size: ~{zarr_size_mb:.1f} MB (detections only)
{storage_comparison_line}

[bold]Output:[/bold]
  {output_zarr}
  
[bold]Next steps:[/bold]
  # Refine detections
  python -m fisheye.refinement.refine_detect {output_zarr}
  
  # Run arena assignment
  python -m fisheye.tracking.arena_assignment {output_zarr}
  
  # Visualize
  python -m fisheye.visualization.detection_visualizer {output_zarr}
"""
    
    panel = Panel(
        summary_text,
        title="[bold green]Detection Summary[/bold green]",
        border_style="green"
    )
    console.print("\n")
    console.print(panel)
    
    return run_name


def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run YOLO inference on video without importing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Using config file (model path in config)
  python -m fisheye.detection.detect_yolo video.mp4
  python -m fisheye.detection.detect_yolo video.mp4 --output test.zarr
  
  # Explicit model and output
  python -m fisheye.detection.detect_yolo video.mp4 --model model.pt --output test.zarr
  
  # With custom thresholds (overrides config)
  python -m fisheye.detection.detect_yolo video.mp4 --conf 0.35 --batch-size 64
  
  # Force CPU
  python -m fisheye.detection.detect_yolo video.mp4 --cpu
  
  # Then run downstream analysis
  python -m fisheye.refinement.refine_detect output.zarr
  python -m fisheye.tracking.arena_assignment output.zarr
        """
    )
    
    parser.add_argument('video_path', help='Input video file')
    parser.add_argument('--model', '--model-path', dest='model_path', default=None,
                       help='Trained YOLO model (.pt) - optional if in config')
    parser.add_argument('--output', '--output-zarr', dest='output_zarr', default=None,
                       help='Output zarr path - optional, auto-generated if not provided')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to YAML config file')
    parser.add_argument('--conf', type=float, default=None, 
                       help='Confidence threshold (overrides config)')
    parser.add_argument('--iou', type=float, default=None, 
                       help='IoU threshold for NMS (overrides config)')
    parser.add_argument('--max-det', type=int, default=None, 
                       help='Max detections per frame (overrides config)')
    parser.add_argument('--batch-size', type=int, default=None, 
                       help='Inference batch size (overrides config)')
    parser.add_argument(
        '--resize-dims',
        nargs='+',
        type=int,
        default=None,
        help='Canonical inference size override [h w] (or one value for square); mapped to YOLO imgsz',
    )
    parser.add_argument(
        '--imgsz',
        nargs='+',
        type=int,
        default=None,
        help='Legacy alias for YOLO inference size; normalized into --resize-dims',
    )
    parser.add_argument(
        '--decode-backend',
        choices=DECODE_BACKEND_CHOICES,
        default=None,
        help=(
            "Video decode backend. Default auto prefers pynvvc_nv12_rgb "
            "when CUDA and resize are available, then falls back to Decord/OpenCV."
        ),
    )
    parser.add_argument('--cpu', action='store_true', 
                       help='Force CPU inference')
    parser.add_argument(
        '--write-raw-video-metadata',
        action='store_true',
        help='Write metadata-only raw_video attrs (no frames) for registry/provenance',
    )
    parser.add_argument(
        '--overwrite-raw-video-metadata',
        action='store_true',
        help='Overwrite existing raw_video attrs when writing metadata-only import',
    )
    parser.add_argument(
        '--run-name',
        default=None,
        help='Optional explicit detect run group name (default: timestamped detect_<utc>).',
    )
    
    args = parser.parse_args()
    
    try:
        detect_yolo(
            video_path=args.video_path,
            model_path=args.model_path,
            output_zarr=args.output_zarr,
            config_path=args.config,
            conf_threshold=args.conf,
            iou_threshold=args.iou,
            max_det=args.max_det,
            batch_size=args.batch_size,
            resize_dims=args.resize_dims,
            imgsz=args.imgsz,
            decode_backend=args.decode_backend,
            use_gpu=not args.cpu if args.cpu else None,
            write_raw_video_metadata=args.write_raw_video_metadata,
            overwrite_raw_video_metadata=args.overwrite_raw_video_metadata,
            run_name=args.run_name,
        )
    except Exception as e:
        console = Console()
        console.print(f"[bold red]Error:[/bold red] {e}")
        import traceback
        console.print(traceback.format_exc())
        sys.exit(1)


if __name__ == '__main__':
    main()
