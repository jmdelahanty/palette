"""YOLO-based keypoint detection for FishEye Zarr archives.

This mirrors :mod:`fisheye.detection.detect_keypoints_traditional` but uses a
trained Ultralytics YOLO pose model on existing ROI crops. It creates a new
``keypoints_runs`` group without overwriting prior runs and records metadata so
downstream tooling can distinguish between traditional and YOLO-derived
keypoints.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple, Sequence
from datetime import datetime, timezone
import time

import numpy as np
import torch
import zarr
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeRemainingColumn

from ..registry.db import RegistryPaths
from ..registry.inline_refresh import refresh_keypoint_performance_details
from ..shared.crop_image_source import CropImageSource
from ..shared.inference_timing import InferenceTimingProfiler
from ..shared.keypoint_summary import build_frame_keypoint_counts
from ..shared.model_input_transform import MODEL_INPUT_TRANSFORM_CHOICES, ModelInputTransform, resolve_model_input_transform
from ..shared.provenance_attrs import build_source_crop_snapshot_attrs, build_source_roi_pixel_attrs
from ..registry.stage_complete import emit_stage_completion
from ..shared.row_lineage import copy_row_lineage_arrays, write_direct_source_crop_row_ids
from ..shared.stage_provenance import build_stage_provenance, write_stage_provenance
from ..shared.type_conversions import normalize_attr
from ..shared.zarr.schema import get_run_group
from ..shared.zarr_run_completion import (
    mark_run_complete,
    mark_run_started,
    note_pending_latest,
    require_runs_parent,
)
from ..pose.heading import compute_heading_from_spec
from ..utils.system import get_environment_info, get_git_info
from ..pose.schema import PoseSchema, normalize_kpt_shape, schema_payload_from_package
from ultralytics import YOLO, __version__ as ultralytics_version

DEFAULT_POSE_SCHEMA_NAME = "traditional_v1"
TRADITIONAL_POSE_SCHEMA, TRADITIONAL_POSE_ATTR_PAYLOAD = schema_payload_from_package(
    DEFAULT_POSE_SCHEMA_NAME
)
_KEYPOINT_STEP_NAME = "keypoints"
_KEYPOINT_STATUS_SOURCE = "runtime_keypoints_detect"
_KEYPOINT_INPUT_MODES = ("numpy-list", "tensor", "auto")
_KEYPOINT_PROGRESS_SCHEMA_ID = "palette.keypoint_inference_progress.v1"


def _progress_json_ready(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _progress_json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_progress_json_ready(item) for item in value]
    return str(value)


def _write_keypoint_progress_jsonl(
    progress_jsonl: Optional[Path],
    event: str,
    **payload: Any,
) -> None:
    if progress_jsonl is None:
        return
    progress_path = Path(progress_jsonl).expanduser()
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    record: Dict[str, Any] = {
        "schema_id": _KEYPOINT_PROGRESS_SCHEMA_ID,
        "event": str(event),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        **payload,
    }
    with progress_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_progress_json_ready(record), sort_keys=True) + "\n")


def _infer_roi_cache_source_tier(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    resolved = str(Path(path).expanduser().resolve())
    if resolved.startswith("/scratch/") or resolved.startswith("/tmp/"):
        return "node_scratch"
    if resolved.startswith("/groups/") or resolved.startswith("/misc/public/"):
        return "prfs_workflow_scratch"
    return "unknown"


def _current_model_device(model: YOLO, *, fallback: Optional[str]) -> Optional[str]:
    try:
        return str(next(model.model.parameters()).device)
    except (AttributeError, StopIteration):
        return fallback


def _scheduler_payload(env_info: Dict[str, Any]) -> Dict[str, Any]:
    platform_info = env_info.get("platform", {})
    gpu_info = env_info.get("gpu", {})
    scheduler: Dict[str, Any] = {
        "execution_hostname": platform_info.get("hostname"),
        "execution_fqdn": platform_info.get("fqdn"),
    }
    lsf = platform_info.get("lsf")
    if isinstance(lsf, dict):
        scheduler.update(
            {
                "scheduler": "lsf",
                "job_id": lsf.get("job_id"),
                "job_name": lsf.get("job_name"),
                "job_index": lsf.get("job_index"),
                "queue": lsf.get("queue"),
                "num_processors": lsf.get("num_processors"),
                "hosts": lsf.get("hosts"),
                "mcpu_hosts": lsf.get("mcpu_hosts"),
                "djob_hostfile": lsf.get("djob_hostfile"),
                "gpu_request": lsf.get("gpu_request"),
                "cuda_visible_devices": lsf.get("cuda_visible_devices"),
                "cuda_visible_devices_orig": lsf.get("cuda_visible_devices_orig"),
            }
        )
    elif isinstance(platform_info.get("slurm"), dict):
        slurm = platform_info["slurm"]
        scheduler.update(
            {
                "scheduler": "slurm",
                "job_id": slurm.get("job_id"),
                "job_name": slurm.get("job_name"),
                "node_list": slurm.get("node_list"),
                "num_nodes": slurm.get("num_nodes"),
                "proc_id": slurm.get("proc_id"),
                "cpus_per_task": slurm.get("cpus_per_task"),
            }
        )
    else:
        scheduler["scheduler"] = None

    if isinstance(gpu_info, dict):
        scheduler["gpu_backend"] = gpu_info.get("backend")
        scheduler["gpu_available"] = gpu_info.get("available")
        scheduler["gpu_devices"] = gpu_info.get("devices")
        scheduler["lsf_allocated_gpus"] = gpu_info.get("lsf_allocated_gpus")
        scheduler["lsf_original_gpus"] = gpu_info.get("lsf_original_gpus")
    return {key: value for key, value in scheduler.items() if value is not None}


def _prepare_run_group(
    root: zarr.Group,
    run_name: Optional[str],
    console: Console,
) -> Tuple[zarr.Group, str]:
    parent = require_runs_parent(root, "keypoints_runs")
    if run_name:
        if run_name in parent:
            raise ValueError(f"keypoints_runs/{run_name} already exists")
        run_group = parent.create_group(run_name)
        mark_run_started(run_group, run_name=run_name, stage="keypoints")
        note_pending_latest(parent, run_name)
        console.print(f"Created run group: [cyan]keypoints_runs/{run_name}[/cyan]")
        return run_group, run_name
    run_group, resolved_name = get_run_group(root, "keypoints", console=console, create_new=True)
    mark_run_started(run_group, run_name=resolved_name, stage="keypoints")
    note_pending_latest(parent, resolved_name)
    return run_group, resolved_name


def _resolve_registry_path(registry: Optional[Path]) -> Optional[Path]:
    if registry is not None:
        return registry.expanduser().resolve()
    inferred = RegistryPaths.from_env(Path.cwd()).path.expanduser().resolve()
    if not inferred.exists():
        return None
    return inferred


def _emit_keypoint_step_status(
    *,
    root: zarr.Group,
    zarr_path: Path,
    run_name: str,
    method: Optional[str],
    coverage_pct: Optional[float],
    details: Dict[str, object],
    console: Optional[Console],
    registry: Optional[Path],
) -> None:
    registry_path = _resolve_registry_path(registry)
    if registry_path is None:
        return
    status_details = dict(details)
    status_details.update(
        refresh_keypoint_performance_details(
            root=root,
            zarr_path=zarr_path,
            run_name=run_name,
            registry_path=registry_path,
            console=console,
        )
    )
    emit_stage_completion(
        root,
        zarr_path,
        step_name=_KEYPOINT_STEP_NAME,
        status="ok",
        source=_KEYPOINT_STATUS_SOURCE,
        run_name=run_name,
        method=method,
        coverage_pct=coverage_pct,
        details_json=status_details,
        console=console,
        registry=registry_path,
        auto_registry_from_env=False,
        trigger_run_name=run_name,
    )


def _create_output_arrays(
    group: zarr.Group,
    total_rois: int,
    chunk_hint: int,
    *,
    n_keypoints: int,
) -> Dict[str, zarr.Array]:
    chunk_len = min(max(chunk_hint, 1), total_rois) if total_rois > 0 else 1
    data_chunk = (chunk_len, int(n_keypoints), 2)
    scalar_chunk = (chunk_len,)

    arrays = {
        "keypoints_roi": group.create_array(
            "keypoints_roi",
            shape=(total_rois, int(n_keypoints), 2),
            chunks=data_chunk,
            dtype="f8",
            fill_value=np.nan,
            overwrite=True,
        ),
        "keypoints_img": group.create_array(
            "keypoints_img",
            shape=(total_rois, int(n_keypoints), 2),
            chunks=data_chunk,
            dtype="f8",
            fill_value=np.nan,
            overwrite=True,
        ),
        "keypoints_norm": group.create_array(
            "keypoints_norm",
            shape=(total_rois, int(n_keypoints), 2),
            chunks=data_chunk,
            dtype="f8",
            fill_value=np.nan,
            overwrite=True,
        ),
        "heading": group.create_array(
            "heading",
            shape=(total_rois,),
            chunks=scalar_chunk,
            dtype="f8",
            fill_value=np.nan,
            overwrite=True,
        ),
        "confidence": group.create_array(
            "confidence",
            shape=(total_rois,),
            chunks=scalar_chunk,
            dtype="f8",
            fill_value=np.nan,
            overwrite=True,
        ),
        "keypoint_confidences": group.create_array(
            "keypoint_confidences",
            shape=(total_rois, int(n_keypoints)),
            chunks=(chunk_len, int(n_keypoints)),
            dtype="f8",
            fill_value=np.nan,
            overwrite=True,
        ),
        "detection_success": group.create_array(
            "detection_success",
            shape=(total_rois,),
            chunks=scalar_chunk,
            dtype="bool",
            fill_value=False,
            overwrite=True,
        ),
        "pose_bbox_xyxy_roi": group.create_array(
            "pose_bbox_xyxy_roi",
            shape=(total_rois, 4),
            chunks=(chunk_len, 4),
            dtype="f4",
            fill_value=np.nan,
            overwrite=True,
        ),
        "heading_finite": group.create_array(
            "heading_finite",
            shape=(total_rois,),
            chunks=scalar_chunk,
            dtype="bool",
            fill_value=False,
            overwrite=True,
        ),
        "heading_usable": group.create_array(
            "heading_usable",
            shape=(total_rois,),
            chunks=scalar_chunk,
            dtype="bool",
            fill_value=False,
            overwrite=True,
        ),
        "effective_threshold": group.create_array(
            "effective_threshold",
            shape=(total_rois,),
            chunks=scalar_chunk,
            dtype="f8",
            fill_value=np.nan,
            overwrite=True,
        ),
        "effective_se2_radius": group.create_array(
            "effective_se2_radius",
            shape=(total_rois,),
            chunks=scalar_chunk,
            dtype="f8",
            fill_value=np.nan,
            overwrite=True,
        ),
    }
    return arrays


def _prepare_refined_roi_overrides(
    root: zarr.Group,
    crop_group: zarr.Group,
    total_rois: int,
    roi_shape: Tuple[int, int],
    console: Console,
) -> Optional[Dict[str, Any]]:
    path = crop_group.attrs.get("refined_roi_path")
    if not path:
        return None
    if path not in root:
        console.print(
            f"[yellow]Refined ROI group '{path}' not found; continuing with original crops.[/yellow]"
        )
        return None
    group = root[path]
    required = {"detection_indices", "roi_images", "roi_coordinates_full"}
    if not required.issubset(set(group.keys())):
        console.print(
            f"[yellow]Refined ROI group '{path}' missing {required - set(group.keys())}; skipping overrides.[/yellow]"
        )
        return None

    detection_indices = group["detection_indices"][:].astype(np.int64, copy=False)
    if detection_indices.size == 0:
        return None
    if detection_indices.min(initial=0) < 0 or detection_indices.max(initial=0) >= total_rois:
        raise ValueError("Refined ROI detection indices out of range for current crop run.")

    refined_rois = group["roi_images"][:]
    if refined_rois.shape[1:3] != roi_shape:
        raise ValueError(
            f"Refined ROI shape {refined_rois.shape[1:3]} does not match crop ROI shape {roi_shape}."
        )
    refined_coords = group["roi_coordinates_full"][:]

    override_map = np.full(total_rois, -1, dtype=np.int64)
    override_map[detection_indices] = np.arange(detection_indices.size, dtype=np.int64)

    frame_indices_override = (
        group["frame_indices"][:].astype(np.int64, copy=False)
        if "frame_indices" in group
        else None
    )

    decoder = (
        group.attrs.get("video_device")
        or group.attrs.get("refined_roi_decoder")
        or crop_group.attrs.get("refined_roi_decoder")
    )
    duration = (
        group.attrs.get("duration_seconds")
        or crop_group.attrs.get("refined_roi_generation_duration_seconds")
    )

    return {
        "path": path,
        "count": detection_indices.size,
        "indices": detection_indices,
        "map": override_map,
        "rois": refined_rois,
        "coords": refined_coords,
        "frame_indices": frame_indices_override,
        "decoder": decoder,
        "duration": duration,
    }


def _compute_heading(points: np.ndarray, pose_schema: PoseSchema) -> float:
    return compute_heading_from_spec(
        pose_schema.metadata.get("heading_computation"),
        labels=pose_schema.node_names,
        points=np.asarray(points, dtype=np.float64),
    )


def _repeat_to_rgb(batch: np.ndarray) -> List[np.ndarray]:
    if batch.ndim != 3:
        raise ValueError("ROI images should have shape (N, H, W)")
    return [np.repeat(img[..., None], 3, axis=2) for img in batch]


def _normalize_input_mode(value: str) -> str:
    text = str(value or "").strip().lower().replace("_", "-")
    if text not in _KEYPOINT_INPUT_MODES:
        choices = ", ".join(_KEYPOINT_INPUT_MODES)
        raise ValueError(f"Invalid keypoint input mode '{value}'. Expected one of: {choices}")
    return text


def _first_attr(attrs: Any, names: Tuple[str, ...]) -> Any:
    for name in names:
        value = attrs.get(name)
        if value is not None:
            return value
    return None


def _resolve_full_image_shape(root: zarr.Group, crop_group: zarr.Group) -> Tuple[Tuple[int, int], Optional[int]]:
    """Resolve full-frame shape for normalized keypoint coordinates.

    Modern crop-first analysis zarrs may intentionally omit raw_video/images_full.
    In that case the geometry-only crop run is the authoritative source for the
    source-video dimensions used to compute roi_coordinates_full.
    """

    total_frames_attr = (
        root.attrs.get("total_frames")
        or root.attrs.get("n_frames")
        or crop_group.attrs.get("total_frames")
    )
    total_frames: Optional[int] = int(total_frames_attr) if total_frames_attr is not None else None

    try:
        images_full = root["raw_video/images_full"]
        frame_dim, img_h, img_w = images_full.shape
        if total_frames is None:
            total_frames = int(frame_dim)
        return (int(img_h), int(img_w)), total_frames
    except KeyError:
        pass

    width_names = (
        "video_width",
        "palette_video_width",
        "source_full_width",
        "source_video_width",
        "width",
    )
    height_names = (
        "video_height",
        "palette_video_height",
        "source_full_height",
        "source_video_height",
        "height",
    )
    img_w = _first_attr(root.attrs, width_names)
    img_h = _first_attr(root.attrs, height_names)
    if img_w is None:
        img_w = _first_attr(crop_group.attrs, width_names)
    if img_h is None:
        img_h = _first_attr(crop_group.attrs, height_names)

    if img_w is None or img_h is None:
        raise ValueError(
            "Unable to determine full-resolution image dimensions. "
            "Expected raw_video/images_full, root video_width/video_height attrs, "
            "or crop-run width/height attrs."
        )
    return (int(img_h), int(img_w)), total_frames


def _tensor_input_blocker(batch: np.ndarray, *, model_input_transform: ModelInputTransform) -> Optional[str]:
    if batch.ndim != 3:
        return f"expected ROI batch shape (N, H, W), got {batch.shape}"
    _, height, width = batch.shape
    if height != width:
        return f"tensor mode requires square ROIs, got {height}x{width}"
    if (height, width) != model_input_transform.native_shape:
        return (
            "tensor mode requires ROI dimensions to match model input transform native shape "
            f"{model_input_transform.native_shape}, got {height}x{width}"
        )
    model_height, model_width = model_input_transform.model_shape
    if model_height != model_width:
        return f"tensor mode requires square model inputs, got {model_height}x{model_width}"
    if model_height % 32 or model_width % 32:
        return f"tensor mode requires model input dimensions divisible by 32, got {model_height}x{model_width}"
    return None


def _prepare_model_inputs(
    batch: np.ndarray,
    *,
    input_mode: str,
    model_input_transform: ModelInputTransform,
    device: Optional[str],
) -> Tuple[object, str]:
    """Prepare one ROI batch for Ultralytics prediction.

    ``numpy-list`` preserves the legacy path. ``tensor`` avoids constructing a
    Python list of RGB numpy arrays and instead supplies normalized BCHW data.
    ``auto`` uses tensor mode only when the geometry is equivalent to the legacy
    path; otherwise it falls back to the list path.
    """
    mode = _normalize_input_mode(input_mode)
    blocker = _tensor_input_blocker(batch, model_input_transform=model_input_transform)
    if mode == "tensor" and blocker is not None:
        raise ValueError(f"Cannot use keypoint tensor input mode: {blocker}")
    batch = model_input_transform.apply_numpy_luma_batch(batch)
    if mode == "numpy-list" or blocker is not None:
        return _repeat_to_rgb(batch), "numpy-list"

    if not batch.flags.c_contiguous or not batch.flags.writeable:
        batch = np.array(batch, copy=True, order="C")
    scale_by_255 = not np.issubdtype(batch.dtype, np.floating)
    if not scale_by_255 and batch.size:
        scale_by_255 = bool(np.nanmax(batch) > 1.0 + np.finfo(np.float32).eps)

    tensor = torch.from_numpy(batch)
    if device:
        tensor = tensor.to(torch.device(device), non_blocking=True)
    tensor = tensor.float()
    if scale_by_255:
        tensor = tensor.div_(255.0)
    return tensor[:, None, :, :].expand(-1, 3, -1, -1).contiguous(), "tensor"


def _select_detection(result) -> Optional[int]:
    boxes = getattr(result, "boxes", None)
    if boxes is None or boxes is False:
        return None
    if boxes.conf is None or boxes.conf.numel() == 0:
        return 0 if boxes.xyxy.shape[0] > 0 else None
    conf = boxes.conf.detach().cpu().numpy()
    return int(conf.argmax())


def _extract_keypoint_confidences(keypoints, det_idx: int, *, n_keypoints: int = 3) -> np.ndarray:
    """Extract per-keypoint confidences for one detection.

    Returns a float array of length ``n_keypoints``. Missing/conf-unavailable
    values are left as NaN.
    """
    out = np.full(n_keypoints, np.nan, dtype=np.float64)
    kp_conf = getattr(keypoints, "conf", None)
    if kp_conf is None:
        return out
    try:
        conf_np = kp_conf[det_idx].detach().cpu().numpy()
    except Exception:
        return out
    conf_flat = np.asarray(conf_np, dtype=np.float64).reshape(-1)
    if conf_flat.size == 0:
        return out
    take = min(n_keypoints, conf_flat.size)
    out[:take] = conf_flat[:take]
    return out


def _resolve_model_kpt_shape(model: YOLO) -> Optional[tuple[int, int]]:
    """Best-effort extraction of the Ultralytics model keypoint shape."""
    model_obj = getattr(model, "model", None)
    for raw in (
        getattr(model_obj, "kpt_shape", None),
        getattr(model_obj, "args", {}).get("kpt_shape")
        if isinstance(getattr(model_obj, "args", None), dict)
        else None,
        getattr(model_obj, "yaml", {}).get("kpt_shape")
        if isinstance(getattr(model_obj, "yaml", None), dict)
        else None,
    ):
        shape = normalize_kpt_shape(raw)
        if shape is not None:
            return shape
    return None


def _normalize_torch_device(device: Optional[str]) -> Optional[str]:
    """Accept Ultralytics-style GPU ids while passing a valid torch device string."""
    if device is None:
        return None
    text = str(device).strip()
    if not text:
        return None
    if text.isdigit():
        return f"cuda:{text}"
    return text


def _extract_pose_bbox_xyxy_roi(
    boxes,
    det_idx: int,
    *,
    roi_height: int,
    roi_width: int,
) -> np.ndarray:
    out = np.full(4, np.nan, dtype=np.float32)
    xyxy = getattr(boxes, "xyxy", None)
    if xyxy is None:
        return out
    try:
        bbox = xyxy[det_idx].detach().cpu().numpy()
    except Exception:
        return out

    bbox = np.asarray(bbox, dtype=np.float32).reshape(-1)
    if bbox.size < 4:
        return out

    max_x = float(max(roi_width - 1, 0))
    max_y = float(max(roi_height - 1, 0))
    x0 = float(np.clip(bbox[0], 0.0, max_x))
    y0 = float(np.clip(bbox[1], 0.0, max_y))
    x1 = float(np.clip(bbox[2], 0.0, max_x))
    y1 = float(np.clip(bbox[3], 0.0, max_y))
    if roi_width > 1 and x1 <= x0:
        x1 = min(max_x, x0 + 1.0)
    if roi_height > 1 and y1 <= y0:
        y1 = min(max_y, y0 + 1.0)
    if x1 <= x0 or y1 <= y0:
        return out
    out[:] = (x0, y0, x1, y1)
    return out


def _clip_xyxy_to_roi(box_xyxy: np.ndarray, *, roi_height: int, roi_width: int) -> np.ndarray:
    out = np.asarray(box_xyxy, dtype=np.float32).reshape(-1).copy()
    if out.size < 4 or not np.all(np.isfinite(out[:4])):
        return np.full(4, np.nan, dtype=np.float32)
    max_x = float(max(roi_width - 1, 0))
    max_y = float(max(roi_height - 1, 0))
    out[0] = np.clip(out[0], 0.0, max_x)
    out[2] = np.clip(out[2], 0.0, max_x)
    out[1] = np.clip(out[1], 0.0, max_y)
    out[3] = np.clip(out[3], 0.0, max_y)
    if roi_width > 1 and out[2] <= out[0]:
        out[2] = min(max_x, out[0] + 1.0)
    if roi_height > 1 and out[3] <= out[1]:
        out[3] = min(max_y, out[1] + 1.0)
    if out[2] <= out[0] or out[3] <= out[1]:
        return np.full(4, np.nan, dtype=np.float32)
    return out[:4].astype(np.float32, copy=False)


def detect_keypoints_yolo(
    zarr_path: str,
    model_path: str,
    *,
    run_name: Optional[str] = None,
    crop_run: Optional[str] = None,
    pose_schema: str = DEFAULT_POSE_SCHEMA_NAME,
    batch_size: int = 256,
    device: Optional[str] = None,
    imgsz: Optional[int] = None,
    conf: float = 0.25,
    iou: float = 0.5,
    max_det: int = 1,
    verbose: bool = False,
    mask_threshold: float = 0.5,
    roi_cache_policy: str = "auto",
    roi_cache_dir: Optional[Path] = None,
    roi_cache_manifest: Optional[Path] = None,
    roi_cache_source_tier: Optional[str] = None,
    roi_cache_staged_to_node_scratch: bool = False,
    roi_cache_staging_details: Optional[Dict[str, Any]] = None,
    roi_live_acceleration: str = "auto",
    roi_live_gpu_chunk_frames: int = 32,
    input_mode: str = "numpy-list",
    model_input_transform_mode: str = "auto",
    profile_timings: bool = False,
    progress_jsonl: Optional[Path] = None,
    progress_every_batches: int = 1,
    registry: Optional[Path] = None,
    console: Optional[Console] = None,
    cli_provenance: Optional[Mapping[str, Any]] = None,
) -> str:
    """Run YOLO pose inference and record outputs in ``keypoints_runs``.

    Returns the name of the created run group.
    """

    console = console or Console()
    console.rule("[bold cyan]YOLO Pose Inference[/bold cyan]")

    zarr_path = Path(zarr_path)
    if not zarr_path.exists():
        raise FileNotFoundError(f"Zarr path not found: {zarr_path}")

    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model path not found: {model_path}")

    model = YOLO(str(model_path))
    torch_device = _normalize_torch_device(device)
    if torch_device:
        model.to(torch_device)
    try:
        model_device = str(next(model.model.parameters()).device)
    except (AttributeError, StopIteration):
        model_device = torch_device or ("cuda" if torch.cuda.is_available() else "cpu")
    model_path_resolved = model_path.resolve()
    pose_schema_obj, pose_schema_attrs = schema_payload_from_package(pose_schema)
    n_keypoints = int(pose_schema_obj.num_keypoints)
    model_kpt_shape = _resolve_model_kpt_shape(model)
    if model_kpt_shape is not None and int(model_kpt_shape[0]) != n_keypoints:
        raise ValueError(
            f"Model keypoint count {int(model_kpt_shape[0])} does not match "
            f"pose schema '{pose_schema_obj.name}' keypoint count {n_keypoints}."
        )

    root = zarr.open_group(str(zarr_path), mode="a", use_consolidated=False)
    crop_source = CropImageSource.open(
        root,
        crop_run=crop_run,
        zarr_path=zarr_path,
        roi_cache_policy=roi_cache_policy,
        roi_live_acceleration=roi_live_acceleration,
        roi_live_gpu_chunk_frames=roi_live_gpu_chunk_frames,
        roi_cache_dir=roi_cache_dir,
        roi_cache_manifest=roi_cache_manifest,
        console=console,
    )
    crop_group = crop_source.crop_group
    latest_crop = crop_source.crop_run_name

    roi_coords = crop_source.roi_coordinates_full.copy()
    frame_indices = crop_source.frame_indices.astype(np.int64, copy=True)
    total_rois = crop_source.total_rois
    if total_rois == 0:
        console.print("[yellow]No ROIs found in crop run; nothing to process[/yellow]")
        crop_source.close()
        return ""

    roi_h, roi_w = crop_source.roi_shape
    source_detect_run = crop_group.attrs.get("source_detect_run")
    source_refined_run = crop_group.attrs.get("source_refined_run")
    override_data = _prepare_refined_roi_overrides(
        root, crop_group, total_rois, (roi_h, roi_w), console
    )
    override_map: Optional[np.ndarray] = None
    override_rois: Optional[np.ndarray] = None
    if override_data is not None:
        indices = override_data["indices"]
        roi_coords[indices] = override_data["coords"]
        frame_override = override_data["frame_indices"]
        if frame_override is not None:
            frame_indices[indices] = frame_override.astype(frame_indices.dtype, copy=False)
        override_map = override_data["map"]
        override_rois = override_data["rois"]
        console.print(
            f"[cyan]Applying refined ROI overrides:[/cyan] {override_data['count']} detections"
        )

    imgsz = imgsz or max(roi_h, roi_w)
    model_input_transform = resolve_model_input_transform(
        (roi_h, roi_w),
        mode=model_input_transform_mode,
        model_hw=(int(imgsz), int(imgsz)),
    )
    resolved_input_mode = _normalize_input_mode(input_mode)

    run_group, resolved_run_name = _prepare_run_group(root, run_name, console)
    run_group.attrs["keypoint_labels"] = list(pose_schema_obj.node_names)
    run_group.attrs["keypoint_confidence_labels"] = list(pose_schema_obj.node_names)
    run_group.attrs["skeleton_id"] = str(pose_schema_attrs["skeleton_id"])
    run_group.attrs["kpt_shape"] = list(pose_schema_attrs["kpt_shape"])
    run_group.attrs["pose_schema"] = dict(pose_schema_attrs)
    if model_kpt_shape is not None:
        run_group.attrs["model_kpt_shape"] = list(model_kpt_shape)
    root.attrs["current_keypoint_group_path"] = run_group.path

    arrays = _create_output_arrays(
        run_group,
        total_rois,
        chunk_hint=batch_size * 4,
        n_keypoints=n_keypoints,
    )

    lineage_result = copy_row_lineage_arrays(
        run_group,
        crop_group,
        total_rois=total_rois,
        use_geometry_preload_profile=True,
    )
    if "source_crop_row_ids" not in lineage_result.copied:
        write_direct_source_crop_row_ids(run_group, total_rois=total_rois)
    if "detection_indices" not in lineage_result.copied:
        console.print("[yellow]Crop run missing 'detection_indices'; YOLO keypoint run will omit them.[/yellow]")

    full_img_shape, total_frames = _resolve_full_image_shape(root, crop_group)

    norm_factor = np.array([full_img_shape[1], full_img_shape[0]], dtype="f8")

    if total_frames is None:
        total_frames = int(frame_indices.max() + 1) if frame_indices.size > 0 else 0

    if "frame_counts" in lineage_result.copied:
        frame_counts_total = run_group["frame_counts"][:].astype("i4", copy=False)
    else:
        frame_counts_total = (
            np.bincount(frame_indices, minlength=total_frames).astype("i4", copy=False)
            if frame_indices.size > 0
            else np.zeros(total_frames, dtype="i4")
        )
    count_chunks = (min(len(frame_counts_total), batch_size * 4),) if frame_counts_total.size > 0 else None
    run_group.create_array(
        "n_rois",
        data=frame_counts_total,
        chunks=count_chunks,
        overwrite=True,
    )
    if "frame_counts" not in lineage_result.copied:
        run_group.create_array(
            "frame_counts",
            data=frame_counts_total,
            chunks=count_chunks,
            overwrite=True,
        )

    crop_detection_source = crop_group.get("detection_source")
    if crop_detection_source is not None and crop_detection_source.shape[0] != total_rois:
        raise ValueError(
            f"Crop run detection_source length {crop_detection_source.shape[0]} does not match ROI count {total_rois}"
        )
    scalar_chunk = arrays["heading"].chunks
    detection_source_dst = run_group.create_array(
        "detection_source",
        shape=(total_rois,),
        chunks=scalar_chunk,
        dtype="i1",
        fill_value=0,
        overwrite=True,
    )

    success_total = 0
    confidence_accum: List[float] = []
    timing_profiler = InferenceTimingProfiler(enabled=profile_timings)
    progress_jsonl_path = Path(progress_jsonl).expanduser() if progress_jsonl is not None else None
    progress_interval = max(1, int(progress_every_batches))

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeRemainingColumn(),
        console=console,
    )

    start_time = time.time()
    _write_keypoint_progress_jsonl(
        progress_jsonl_path,
        "start",
        zarr_path=str(zarr_path.resolve()),
        run_name=resolved_run_name,
        crop_run=latest_crop,
        total_rois=int(total_rois),
        batch_size=int(batch_size),
        pose_schema=pose_schema_obj.name,
        model_path=str(model_path_resolved),
        requested_device=device,
        normalized_torch_device=torch_device,
        initial_model_device=model_device,
        roi_read_mode=crop_source.roi_read_mode,
        roi_cache_policy=crop_source.roi_cache_policy,
        source_roi_cache_used=bool(crop_source.roi_cache_used),
        frame_source=crop_source.frame_source_kind,
        source_video_path=crop_source.frame_source_path or crop_group.attrs.get("video_source_path"),
    )

    with progress:
        task = progress.add_task("[cyan]Predicting keypoints...", total=total_rois)
        for batch_index, start in enumerate(range(0, total_rois, batch_size), start=1):
            batch_started = time.perf_counter()
            end = min(start + batch_size, total_rois)
            batch_coords = roi_coords[start:end]
            batch_count = end - start
            with timing_profiler.time("roi_read", items=batch_count):
                batch_roi_np = crop_source.read_slice(start, end)
            if override_map is not None and override_rois is not None:
                with timing_profiler.time("roi_override_apply", items=batch_count):
                    local_map = override_map[start:end]
                    valid = local_map >= 0
                    if np.any(valid):
                        batch_roi_np[valid] = override_rois[local_map[valid]]

            with timing_profiler.time("input_prepare", items=batch_count):
                model_inputs, effective_input_mode = _prepare_model_inputs(
                    batch_roi_np,
                    input_mode=resolved_input_mode,
                    model_input_transform=model_input_transform,
                    device=torch_device,
                )
            with timing_profiler.time("model_predict", items=batch_count):
                results = tuple(
                    model.predict(
                        model_inputs,
                        imgsz=imgsz,
                        conf=conf,
                        iou=iou,
                        max_det=max_det,
                        device=torch_device,
                        verbose=verbose,
                        stream=True,
                    )
                )

            batch_keypoints_roi = np.full((batch_count, n_keypoints, 2), np.nan, dtype=np.float64)
            batch_keypoints_img = np.full_like(batch_keypoints_roi, np.nan)
            batch_keypoints_norm = np.full_like(batch_keypoints_roi, np.nan)
            batch_heading = np.full(batch_count, np.nan, dtype=np.float64)
            batch_conf = np.full(batch_count, np.nan, dtype=np.float64)
            batch_keypoint_conf = np.full((batch_count, n_keypoints), np.nan, dtype=np.float64)
            batch_success = np.zeros(batch_count, dtype=bool)
            batch_pose_bbox_roi = np.full((batch_count, 4), np.nan, dtype=np.float32)

            with timing_profiler.time("result_decode", items=batch_count):
                for i, (res, top_left) in enumerate(zip(results, batch_coords)):
                    det_idx = _select_detection(res)
                    if det_idx is None:
                        continue
                    keypoints = getattr(res, "keypoints", None)
                    if keypoints is None or keypoints.xy is None:
                        continue
                    kp_xy = keypoints.xy
                    if kp_xy is None or kp_xy.ndim != 3 or kp_xy.shape[0] == 0:
                        continue

                    kp = kp_xy[det_idx].detach().cpu().numpy()
                    if kp.shape[0] < n_keypoints:
                        continue
                    if kp.shape[0] > n_keypoints:
                        kp = kp[:n_keypoints]
                    kp = model_input_transform.invert_points_xy(kp)

                    kp[:, 0] = np.clip(kp[:, 0], 0.0, roi_w - 1)
                    kp[:, 1] = np.clip(kp[:, 1], 0.0, roi_h - 1)

                    batch_keypoints_roi[i] = kp
                    top_left = np.asarray(top_left, dtype=np.float64)
                    kp_img = kp + np.array([top_left[0], top_left[1]])
                    batch_keypoints_img[i] = kp_img
                    batch_keypoints_norm[i] = kp_img / norm_factor
                    batch_heading[i] = _compute_heading(kp, pose_schema_obj)
                    batch_keypoint_conf[i] = _extract_keypoint_confidences(
                        keypoints,
                        det_idx,
                        n_keypoints=n_keypoints,
                    )

                    boxes = getattr(res, "boxes", None)
                    if boxes is not None and boxes.conf is not None and boxes.conf.numel() > 0:
                        det_conf = float(boxes.conf[det_idx].detach().cpu())
                    else:
                        kp_conf = getattr(keypoints, "conf", None)
                        det_conf = float(kp_conf[det_idx].detach().cpu().mean()) if kp_conf is not None else 0.0
                    if boxes is not None:
                        pose_bbox_model = _extract_pose_bbox_xyxy_roi(
                            boxes,
                            det_idx,
                            roi_height=model_input_transform.model_height,
                            roi_width=model_input_transform.model_width,
                        )
                        if np.all(np.isfinite(pose_bbox_model)):
                            batch_pose_bbox_roi[i] = _clip_xyxy_to_roi(
                                model_input_transform.invert_boxes_xyxy(pose_bbox_model),
                                roi_height=roi_h,
                                roi_width=roi_w,
                            )

                    batch_conf[i] = det_conf
                    batch_success[i] = True
                    success_total += 1
                    confidence_accum.append(det_conf)

            with timing_profiler.time("output_write", items=batch_count):
                arrays["keypoints_roi"][start:end] = batch_keypoints_roi
                arrays["keypoints_img"][start:end] = batch_keypoints_img
                arrays["keypoints_norm"][start:end] = batch_keypoints_norm
                arrays["heading"][start:end] = batch_heading
                arrays["confidence"][start:end] = batch_conf
                arrays["keypoint_confidences"][start:end] = batch_keypoint_conf
                arrays["detection_success"][start:end] = batch_success
                arrays["pose_bbox_xyxy_roi"][start:end] = batch_pose_bbox_roi
                arrays["effective_threshold"][start:end] = np.nan
                arrays["effective_se2_radius"][start:end] = np.nan

                if crop_detection_source is not None:
                    source_chunk = crop_detection_source[start:end].astype("i1", copy=False)
                else:
                    source_chunk = np.zeros(end - start, dtype="i1")
                detection_source_dst[start:end] = source_chunk
                heading_finite_chunk = np.isfinite(batch_heading)
                arrays["heading_finite"][start:end] = heading_finite_chunk
                arrays["heading_usable"][start:end] = np.logical_and(
                    np.logical_and(batch_success, source_chunk == 0),
                    heading_finite_chunk,
                )

            progress.update(task, advance=end - start)
            if batch_index % progress_interval == 0 or end >= total_rois:
                batch_seconds = time.perf_counter() - batch_started
                elapsed_seconds = time.time() - start_time
                _write_keypoint_progress_jsonl(
                    progress_jsonl_path,
                    "batch_complete",
                    zarr_path=str(zarr_path.resolve()),
                    run_name=resolved_run_name,
                    crop_run=latest_crop,
                    batch_index=int(batch_index),
                    row_start=int(start),
                    row_stop=int(end),
                    batch_rows=int(batch_count),
                    rows_written=int(end),
                    total_rois=int(total_rois),
                    percent_complete=float((end / total_rois) * 100.0) if total_rois else 100.0,
                    batch_successful=int(np.count_nonzero(batch_success)),
                    successful_so_far=int(success_total),
                    failed_so_far=int(end - success_total),
                    batch_seconds=float(batch_seconds),
                    elapsed_seconds=float(elapsed_seconds),
                    batch_rois_per_second=float(batch_count / batch_seconds) if batch_seconds > 0 else None,
                    cumulative_rois_per_second=float(end / elapsed_seconds) if elapsed_seconds > 0 else None,
                    input_mode_requested=resolved_input_mode,
                    input_mode_effective=effective_input_mode,
                )

    total_time = time.time() - start_time
    inference_rate = success_total / total_time if total_time > 0 else 0.0
    resolved_model_device = _current_model_device(model, fallback=model_device)

    success_rate = (success_total / total_rois * 100.0) if total_rois > 0 else 0.0
    failure_total = total_rois - success_total

    success_counts = build_frame_keypoint_counts(
        frame_indices,
        arrays["detection_success"][:],
        frame_axis_len=int(frame_counts_total.shape[0]),
        keypoint_count=n_keypoints,
    )
    success_chunks = (min(len(success_counts), batch_size * 4),) if success_counts.size > 0 else None
    run_group.create_array(
        "n_keypoints",
        data=success_counts,
        chunks=success_chunks,
        overwrite=True,
    )

    git_info = get_git_info()
    env_info = get_environment_info(
        include_all_packages=False,
        disk_path=str(zarr_path),
        collect_ip=False,
        capture_env_vars=False,
    )
    platform_info = env_info.get("platform", {})
    scheduler_info = _scheduler_payload(env_info)
    crop_snapshot_attrs = build_source_crop_snapshot_attrs(
        crop_group.attrs,
        source_crop_storage_mode=crop_source.storage_mode,
    )
    crop_pixel_attrs = build_source_roi_pixel_attrs(crop_source)
    effective_roi_cache_source_tier = roi_cache_source_tier or _infer_roi_cache_source_tier(crop_source.roi_cache_path)
    roi_cache_staging_payload = dict(roi_cache_staging_details) if isinstance(roi_cache_staging_details, dict) else None
    roi_cache_staging_policy = (
        roi_cache_staging_payload.get("policy")
        if isinstance(roi_cache_staging_payload, dict)
        else None
    )

    run_group.attrs.update({
        "method": "yolo_pose",
        "keypoints_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model_path": str(model_path_resolved),
        "model_name": model_path.name,
        "ultralytics_version": ultralytics_version,
        "device": resolved_model_device,
        "requested_device": device,
        "normalized_torch_device": torch_device,
        "initial_model_device": model_device,
        "resolved_model_device": resolved_model_device,
        "execution_hostname": scheduler_info.get("execution_hostname"),
        "scheduler": scheduler_info.get("scheduler"),
        "scheduler_job_id": scheduler_info.get("job_id"),
        "scheduler_job_name": scheduler_info.get("job_name"),
        "scheduler_job_index": scheduler_info.get("job_index"),
        "scheduler_queue": scheduler_info.get("queue"),
        "scheduler_hosts": scheduler_info.get("hosts") or scheduler_info.get("node_list"),
        "scheduler_mcpu_hosts": scheduler_info.get("mcpu_hosts"),
        "scheduler_cuda_visible_devices": scheduler_info.get("cuda_visible_devices"),
        "scheduler_gpu_request": scheduler_info.get("gpu_request"),
        "source_crop_run": latest_crop,
        **crop_snapshot_attrs,
        **crop_pixel_attrs,
        "source_roi_read_mode": crop_source.roi_read_mode,
        "roi_cache_policy": crop_source.roi_cache_policy,
        "source_roi_cache_used": bool(crop_source.roi_cache_used),
        "source_roi_cache_backend": getattr(crop_source, "roi_cache_backend", None),
        "source_roi_cache_source_tier": effective_roi_cache_source_tier,
        "source_roi_cache_staged_to_node_scratch": bool(roi_cache_staged_to_node_scratch),
        "source_roi_cache_staging_policy": roi_cache_staging_policy,
        "source_roi_live_acceleration_requested": crop_source.roi_live_acceleration_requested,
        "source_roi_live_acceleration_effective": crop_source.roi_live_acceleration_effective,
        "source_roi_live_acceleration_fallback_reason": crop_source.roi_live_acceleration_fallback_reason,
        "source_roi_live_gpu_chunk_frames": int(crop_source.roi_live_gpu_chunk_frames),
        "source_detect_run": source_detect_run or "unknown",
        "keypoints_processed": total_rois,
        "success_rate": round(success_rate, 2),
        "parameters": {
            "confidence_threshold": conf,
            "iou_threshold": iou,
            "max_det": max_det,
            "imgsz": imgsz,
            "batch_size": batch_size,
            "device": resolved_model_device,
            "requested_device": device,
            "normalized_torch_device": torch_device,
            "initial_model_device": model_device,
            "resolved_model_device": resolved_model_device,
            "profile_timings": bool(profile_timings),
            "progress_jsonl": str(progress_jsonl_path) if progress_jsonl_path is not None else None,
            "progress_every_batches": int(progress_interval),
            "pose_schema": pose_schema_obj.name,
            "n_keypoints": int(n_keypoints),
            "model_kpt_shape": list(model_kpt_shape) if model_kpt_shape is not None else None,
            "roi_live_acceleration": str(roi_live_acceleration),
            "roi_live_gpu_chunk_frames": int(roi_live_gpu_chunk_frames),
            "roi_cache_manifest": str(roi_cache_manifest) if roi_cache_manifest is not None else None,
            "roi_cache_source_tier": effective_roi_cache_source_tier,
            "roi_cache_staged_to_node_scratch": bool(roi_cache_staged_to_node_scratch),
            "roi_cache_staging_policy": roi_cache_staging_policy,
            "input_mode_requested": resolved_input_mode,
            "input_mode_effective": effective_input_mode,
            "model_input_transform": model_input_transform.to_attrs(),
            # Maintained for API compatibility with pipeline/batch configs.
            "mask_threshold": float(mask_threshold),
        },
        "model_names": getattr(model.model, "names", None),
        "summary_statistics": {
            "total_rois": int(total_rois),
            "successful_detections": int(success_total),
            "failed_detections": int(failure_total),
            "success_rate_percent": round(success_rate, 2),
            "mean_confidence": float(np.mean(confidence_accum)) if confidence_accum else 0.0,
        },
        "git_commit": git_info.get("commit_hash", "unknown"),
        "git_branch": git_info.get("branch", "unknown"),
        "hostname": env_info["platform"].get("hostname", "unknown"),
        "inference_duration_seconds": float(total_time),
        "inference_poses_per_second": float(inference_rate),
        "profile_timings_enabled": bool(profile_timings),
        "progress_jsonl": str(progress_jsonl_path) if progress_jsonl_path is not None else None,
        "input_mode_requested": resolved_input_mode,
        "input_mode_effective": effective_input_mode,
        "model_input_transform": model_input_transform.to_attrs(),
        "model_input_transform_name": model_input_transform.name,
        "model_input_shape_hw": list(model_input_transform.model_shape),
        "native_roi_shape_hw": list(model_input_transform.native_shape),
    })
    if timing_profiler.enabled:
        run_group.attrs["timing_profile"] = timing_profiler.summary(
            total_items=int(total_rois),
            wall_seconds=float(total_time),
            notes=[
                "roi_read measures ROI slice fetch from the active crop image source.",
                "model_predict wraps Ultralytics predict(), including model-side preprocessing and postprocessing.",
                "tensor input mode supplies normalized BCHW tensors and bypasses the legacy numpy-list RGB expansion.",
                "output_write measures Zarr writes for keypoint outputs and lineage flags.",
            ],
        )
    if crop_source.roi_cache_key:
        run_group.attrs["source_roi_cache_key"] = crop_source.roi_cache_key
    if crop_source.roi_cache_path:
        run_group.attrs["source_roi_cache_path"] = crop_source.roi_cache_path
    if roi_cache_staging_payload is not None:
        run_group.attrs["source_roi_cache_staging"] = roi_cache_staging_payload
    if source_refined_run:
        run_group.attrs["source_refined_run"] = source_refined_run
    if cli_provenance is not None:
        run_group.attrs["cli_provenance"] = dict(cli_provenance)
    provenance_record = build_stage_provenance(
        stage="keypoints_detect",
        command=" ".join(sys.argv),
        created_at_utc=str(run_group.attrs.get("keypoints_timestamp_utc")),
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
            "python_version": platform_info.get("python_version"),
            "machine": platform_info.get("machine"),
        },
        scheduler=scheduler_info,
        parameters=dict(run_group.attrs.get("parameters") or {}),
        inputs={
            "source_crop_run": latest_crop,
            **crop_snapshot_attrs,
            **crop_pixel_attrs,
            "source_roi_read_mode": crop_source.roi_read_mode,
            "roi_cache_policy": crop_source.roi_cache_policy,
            "roi_cache_used": bool(crop_source.roi_cache_used),
            "roi_cache_backend": getattr(crop_source, "roi_cache_backend", None),
            "roi_cache_key": crop_source.roi_cache_key,
            "roi_cache_path": crop_source.roi_cache_path,
            "roi_cache_source_tier": effective_roi_cache_source_tier,
            "roi_cache_staged_to_node_scratch": bool(roi_cache_staged_to_node_scratch),
            "roi_cache_staging_policy": roi_cache_staging_policy,
            "roi_cache_staging": roi_cache_staging_payload,
            "roi_live_acceleration_requested": crop_source.roi_live_acceleration_requested,
            "roi_live_acceleration_effective": crop_source.roi_live_acceleration_effective,
            "roi_live_acceleration_fallback_reason": crop_source.roi_live_acceleration_fallback_reason,
            "roi_live_gpu_chunk_frames": int(crop_source.roi_live_gpu_chunk_frames),
            "input_mode_requested": resolved_input_mode,
            "input_mode_effective": effective_input_mode,
            "source_detect_run": source_detect_run or "unknown",
            "source_refined_run": source_refined_run,
            "frame_source": crop_source.frame_source_kind,
            "source_video_path": crop_source.frame_source_path or crop_group.attrs.get("video_source_path"),
        },
        artifacts={
            "model_path": str(model_path_resolved),
            "model_name": model_path.name,
            "ultralytics_version": ultralytics_version,
            "device": resolved_model_device,
            "requested_device": device,
            "normalized_torch_device": torch_device,
            "initial_model_device": model_device,
            "resolved_model_device": resolved_model_device,
            "pose_schema": pose_schema_obj.name,
            "skeleton_id": str(pose_schema_attrs["skeleton_id"]),
            "kpt_shape": list(pose_schema_attrs["kpt_shape"]),
            "model_kpt_shape": list(model_kpt_shape) if model_kpt_shape is not None else None,
        },
    )
    write_stage_provenance(run_group, provenance_record)
    if override_data is not None:
        run_group.attrs["refined_roi_overrides"] = int(override_data["count"])
        run_group.attrs["refined_roi_source"] = override_data["path"]
        if override_data["decoder"]:
            run_group.attrs["refined_roi_decoder"] = override_data["decoder"]
        if override_data["duration"] is not None:
            run_group.attrs["refined_roi_generation_duration_seconds"] = float(override_data["duration"])

    status_details: Dict[str, object] = {
        "reason": "present",
        "run_group": "keypoints_runs",
        "source_crop_run": latest_crop,
        **crop_snapshot_attrs,
        **crop_pixel_attrs,
        "source_roi_read_mode": crop_source.roi_read_mode,
        "roi_cache_policy": crop_source.roi_cache_policy,
        "roi_cache_used": bool(crop_source.roi_cache_used),
        "roi_cache_backend": getattr(crop_source, "roi_cache_backend", None),
        "roi_cache_source_tier": effective_roi_cache_source_tier,
        "roi_cache_staged_to_node_scratch": bool(roi_cache_staged_to_node_scratch),
        "roi_cache_staging_policy": roi_cache_staging_policy,
        "roi_cache_staging": roi_cache_staging_payload,
        "roi_live_acceleration_requested": crop_source.roi_live_acceleration_requested,
        "roi_live_acceleration_effective": crop_source.roi_live_acceleration_effective,
        "roi_live_acceleration_fallback_reason": crop_source.roi_live_acceleration_fallback_reason,
        "roi_live_gpu_chunk_frames": int(crop_source.roi_live_gpu_chunk_frames),
        "source_detect_run": source_detect_run or "unknown",
        "total_rois": int(total_rois),
        "successful_detections": int(success_total),
        "failed_detections": int(failure_total),
        "success_rate_percent": round(float(success_rate), 2),
        "pose_schema": pose_schema_obj.name,
        "skeleton_id": str(pose_schema_attrs["skeleton_id"]),
        "kpt_shape": list(pose_schema_attrs["kpt_shape"]),
        "model_kpt_shape": list(model_kpt_shape) if model_kpt_shape is not None else None,
        "input_mode_requested": resolved_input_mode,
        "input_mode_effective": effective_input_mode,
        "requested_device": device,
        "normalized_torch_device": torch_device,
        "initial_model_device": model_device,
        "resolved_model_device": resolved_model_device,
        "execution_hostname": scheduler_info.get("execution_hostname"),
        "scheduler": scheduler_info.get("scheduler"),
        "scheduler_job_id": scheduler_info.get("job_id"),
        "scheduler_job_name": scheduler_info.get("job_name"),
        "scheduler_job_index": scheduler_info.get("job_index"),
        "scheduler_queue": scheduler_info.get("queue"),
        "scheduler_hosts": scheduler_info.get("hosts") or scheduler_info.get("node_list"),
        "scheduler_mcpu_hosts": scheduler_info.get("mcpu_hosts"),
        "scheduler_cuda_visible_devices": scheduler_info.get("cuda_visible_devices"),
        "scheduler_gpu_request": scheduler_info.get("gpu_request"),
    }
    if source_refined_run:
        status_details["source_refined_run"] = source_refined_run
    if override_data is not None:
        status_details["refined_roi_overrides"] = int(override_data["count"])
        status_details["refined_roi_source"] = str(override_data["path"])

    mark_run_complete(run_group, parent_group=root["keypoints_runs"], run_name=resolved_run_name)

    try:
        _emit_keypoint_step_status(
            root=root,
            zarr_path=zarr_path.resolve(),
            run_name=resolved_run_name,
            method=normalize_attr(run_group.attrs.get("method")) or "yolo_pose",
            coverage_pct=float(success_rate),
            details=status_details,
            console=console,
            registry=registry,
        )
    finally:
        crop_source.close()
    _write_keypoint_progress_jsonl(
        progress_jsonl_path,
        "complete",
        zarr_path=str(zarr_path.resolve()),
        run_name=resolved_run_name,
        crop_run=latest_crop,
        total_rois=int(total_rois),
        successful_detections=int(success_total),
        failed_detections=int(failure_total),
        success_rate_percent=round(float(success_rate), 2),
        elapsed_seconds=float(total_time),
        poses_per_second=float(inference_rate),
        resolved_model_device=resolved_model_device,
    )

    summary_lines = [
        "[green]✓[/green] Pose inference complete",
        "",
        f"[bold]Run:[/bold] keypoints_runs/{resolved_run_name}",
        f"[bold]Total ROIs:[/bold] {total_rois}",
        f"[bold]Successful:[/bold] {success_total} ({success_rate:.2f}%)",
        f"[bold]Failed:[/bold] {failure_total}",
        f"[bold]Model:[/bold] {model_path_resolved}",
        f"[bold]Duration:[/bold] {total_time:.1f}s ({inference_rate:.1f} poses/s)",
    ]
    if timing_profiler.enabled:
        summary_lines.append("[bold]Timing Profile:[/bold]")
        for line in timing_profiler.render_lines(total_items=total_rois, wall_seconds=total_time, limit=5):
            summary_lines.append(f"[dim]{line}[/dim]")
    if override_data is not None:
        summary_lines.append(
            f"[dim]Refined ROI overrides: {override_data['count']} from {override_data['path']}[/dim]"
        )
    completion = Panel(
        "\n".join(summary_lines),
        title="YOLO Pose Inference",
        border_style="green",
    )
    console.print("\n")
    console.print(completion)

    return resolved_run_name


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="YOLO-based keypoint inference on Palette Zarr crops")
    parser.add_argument("zarr_path", type=str, help="Path to the Palette Zarr archive")
    parser.add_argument("--model", required=True, help="Path to the trained YOLO pose weights (.pt)")
    parser.add_argument("--run-name", help="Optional custom run name for keypoints_runs")
    parser.add_argument("--crop-run", help="Optional crop run override (defaults to latest)")
    parser.add_argument(
        "--pose-schema",
        default=DEFAULT_POSE_SCHEMA_NAME,
        help=(
            "Pose schema from configs/fisheye/pose_schemas to stamp on the output run "
            f"(default: {DEFAULT_POSE_SCHEMA_NAME})."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size for inference")
    parser.add_argument("--imgsz", type=int, default=None, help="Image size for YOLO inference")
    parser.add_argument("--device", default=None, help="Torch device string (e.g. '0' or 'cuda:0')")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.5, help="IoU threshold for NMS")
    parser.add_argument("--max-det", type=int, default=1, help="Maximum detections per ROI")
    parser.add_argument(
        "--mask-threshold",
        type=float,
        default=0.5,
        help="Compatibility parameter recorded in run metadata (not used for pose decoding).",
    )
    parser.add_argument("--registry", type=Path, default=None, help="Optional registry SQLite path.")
    parser.add_argument(
        "--roi-cache-policy",
        choices=("never", "auto", "always"),
        default="auto",
        help="Temporary ROI cache policy for geometry-only crop runs (default: auto).",
    )
    parser.add_argument(
        "--roi-cache-dir",
        type=Path,
        default=None,
        help="Optional scratch directory for temporary ROI caches.",
    )
    parser.add_argument(
        "--roi-cache-manifest",
        type=Path,
        default=None,
        help="Optional flat_bin_v1 ROI cache manifest to read instead of materializing/re-decoding ROIs.",
    )
    parser.add_argument(
        "--roi-cache-source-tier",
        choices=("node_scratch", "prfs_workflow_scratch", "canonical_materialized", "unknown"),
        default=None,
        help="Optional provenance label for where --roi-cache-manifest was read from.",
    )
    parser.add_argument(
        "--roi-cache-staged-to-node-scratch",
        action="store_true",
        help="Stamp provenance that the effective ROI cache manifest was staged to node-local scratch before inference.",
    )
    parser.add_argument(
        "--profile-timings",
        action="store_true",
        help="Collect per-stage timing diagnostics and store them in the output run attrs.",
    )
    parser.add_argument(
        "--progress-jsonl",
        type=Path,
        default=None,
        help="Optional JSONL file for live progress events written after each durable output batch.",
    )
    parser.add_argument(
        "--progress-every-batches",
        type=int,
        default=1,
        help="Write one progress JSONL event every N completed batches (default: 1).",
    )
    parser.add_argument(
        "--input-mode",
        choices=_KEYPOINT_INPUT_MODES,
        default="numpy-list",
        help=(
            "Input preparation mode for Ultralytics prediction. 'numpy-list' preserves the legacy "
            "RGB numpy-list path; 'tensor' feeds normalized BCHW tensors directly; 'auto' uses "
            "tensor mode only when ROI geometry is equivalent to the legacy path."
        ),
    )
    parser.add_argument(
        "--model-input-transform",
        choices=MODEL_INPUT_TRANSFORM_CHOICES,
        default="auto",
        help=(
            "Reversible transform from native ROI crops to the requested model input size. "
            "'auto' is identity when sizes match and centered zero-padding when --imgsz is larger."
        ),
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose Ultralytics output")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    detect_keypoints_yolo(
        zarr_path=args.zarr_path,
        model_path=args.model,
        run_name=args.run_name,
        crop_run=args.crop_run,
        pose_schema=args.pose_schema,
        batch_size=args.batch_size,
        device=args.device,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        verbose=args.verbose,
        mask_threshold=args.mask_threshold,
        roi_cache_policy=args.roi_cache_policy,
        roi_cache_dir=args.roi_cache_dir,
        roi_cache_manifest=args.roi_cache_manifest,
        roi_cache_source_tier=args.roi_cache_source_tier,
        roi_cache_staged_to_node_scratch=bool(args.roi_cache_staged_to_node_scratch),
        input_mode=args.input_mode,
        model_input_transform_mode=args.model_input_transform,
        profile_timings=args.profile_timings,
        progress_jsonl=args.progress_jsonl,
        progress_every_batches=args.progress_every_batches,
        registry=args.registry,
    )


__all__ = ["detect_keypoints_yolo", "main"]


if __name__ == "__main__":
    main()
