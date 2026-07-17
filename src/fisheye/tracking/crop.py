"""
Crop ROIs from full-resolution frames based on detection results.
Part of the FishEye tracking pipeline.

This version supports multiple detection sources:
- 'detect': Original blob/YOLO detections
- 'filtered': Legacy refined sparse compatibility surface
- 'interpolated': Legacy refined sparse compatibility surface
- 'manual': Legacy refined sparse compatibility surface
- 'refined': Canonical refined detect surface; current runs read curated rows
  from refined_detect_runs/<run>/instances
- 'auto': Resolve to the canonical refined detect surface first, else fall
  back to the legacy sparse preference chain

Streams work with Dask and writes directly from workers to Zarr
to avoid accumulating large results in driver memory.
"""

import time
import json
import zarr
import os
import sys
import subprocess
from queue import Queue
import numpy as np
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List, Mapping
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn, MofNCompleteColumn
from rich.align import Align

# Metadata helpers
from ..registry.db import Registry, RegistryPaths
from ..registry.stage_complete import emit_stage_completion
from ..shared.metadata import has_raw_video, get_video_source_path, get_total_frames, get_detection_method
from ..shared.refined_detect_review import (
    DEFAULT_DETECT_GROUP_PREFERENCE,
    resolve_refined_detect_group,
)
from ..shared.refined_detect_curation import (
    build_curated_detection_source_array,
    extract_present_curated_rows,
    has_curated_refined_detect_surface,
    has_sparse_curated_refined_detect_instances_arrays,
)
from ..shared.stage_provenance import build_stage_provenance, write_stage_provenance
from ..shared.crop_signature import build_crop_signature
from ..shared.grayscale import rgb_to_gray_unweighted_mean_torch
from ..shared.crop_roi_layout import (
    DEFAULT_CANONICAL_CROP_ROI_CHUNK_LEN,
    DEFAULT_SCRATCH_ROI_CACHE_CHUNK_LEN,
    DEFAULT_SCRATCH_ROI_CACHE_GPU_CHUNK_FRAMES,
    SCRATCH_ROI_CACHE_LAYOUT_PROFILE,
    build_canonical_crop_roi_layout,
    build_crop_roi_create_kwargs,
    build_scratch_roi_cache_layout,
    crop_roi_layout_attrs,
    normalize_crop_roi_storage,
)
from ..shared.roi_pixel_contract import (
    CENTER_ROUNDING_NP_ROUND,
    crop_run_pixel_contract,
)
from ..shared.type_conversions import normalize_attr
from ..shared.run_resolution import RunResolution, resolve_run
from ..shared.run_provenance import (
    CLI_RUN_PROVENANCE_ATTR,
    RUN_PROVENANCE_ATTR,
    build_run_provenance,
)
from ..shared.zarr_run_completion import (
    COMPLETION_EPOCH_REQUIRE_PROVENANCE,
    mark_run_complete,
    mark_run_failed,
    mark_run_started,
    note_pending_latest,
)
from ..shared.zarr.chunk_profiles import create_geometry_preload_array

REFINED_DETECT_GROUP = "refined_detect_runs"
LEGACY_REFINED_DETECT_GROUP = "refined_runs"
_CROP_STATUS_SOURCE = "runtime_crop"
_DISABLE_REGISTRY_WRITES_ENV = "PALETTE_DISABLE_REGISTRY_WRITES"

# Dask imports
import dask
from dask import delayed
from dask.diagnostics import ProgressBar

# Optional distributed scheduler
try:
    from dask.distributed import Client, LocalCluster, as_completed
    HAVE_DISTRIBUTED = True
except Exception:
    HAVE_DISTRIBUTED = False

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

try:
    import cupy as cp
    _CUPY_AVAILABLE = True
except ImportError:
    _CUPY_AVAILABLE = False

try:
    import kvikio
    import kvikio.defaults as kvikio_defaults
    import kvikio.zarr
    _KVIKIO_AVAILABLE = True
except ImportError:
    _KVIKIO_AVAILABLE = False

try:
    import decord
    from decord import VideoReader, cpu, gpu
    _DECORD_AVAILABLE = True
except Exception:
    decord = None  # type: ignore
    VideoReader = None  # type: ignore
    cpu = None  # type: ignore
    gpu = None  # type: ignore
    _DECORD_AVAILABLE = False

import cv2  # For explicit CPU crop/inspection paths.

from ..shared.system_metadata import get_environment_info


def check_gpu_crop_available() -> Tuple[bool, str]:
    """
    Check if GPU-accelerated cropping is available.
    
    Returns:
        Tuple of (available, reason)
    """
    if not _TORCH_AVAILABLE:
        return False, "PyTorch not installed"
    
    if not _DECORD_AVAILABLE:
        return False, "Decord not installed"
    
    if not torch.cuda.is_available():
        return False, "CUDA not available"
    
    return True, "GPU cropping available"


def _gpu_decode_unavailable(reason: str) -> RuntimeError:
    return RuntimeError(
        "GPU decode unavailable; refusing CPU fallback - pixels would differ from the "
        f"production path ({reason})"
    )


_VALID_CROP_STORAGE_MODES = {"materialized", "geometry_only"}


def _normalize_crop_storage_mode(value: object) -> str:
    text = str(value or "materialized").strip().lower()
    if text not in _VALID_CROP_STORAGE_MODES:
        choices = ", ".join(sorted(_VALID_CROP_STORAGE_MODES))
        raise ValueError(f"Invalid crop_storage_mode '{text}'. Expected one of: {choices}")
    return text


def _infer_archive_use(root: zarr.Group, zarr_path: str | Path | None = None) -> Optional[str]:
    for attr_name in ("zarr_use", "zarr_purpose"):
        value = root.attrs.get(attr_name)
        if value is None:
            continue
        text = str(value).strip().lower()
        if text in {"analysis", "training"}:
            return text
    if zarr_path is not None:
        name = Path(zarr_path).name.lower()
        if name.endswith("_analysis.zarr"):
            return "analysis"
        if name.endswith("_training.zarr"):
            return "training"
    return None


def _enforce_training_materialized_crop_contract(
    root: zarr.Group,
    *,
    zarr_path: str | Path | None,
    crop_storage_mode: str,
) -> None:
    archive_use = _infer_archive_use(root, zarr_path)
    if archive_use == "training" and crop_storage_mode != "materialized":
        raise ValueError(
            "Training zarrs require materialized crop runs; "
            "crop_storage_mode=geometry_only is only supported for analysis archives."
        )


def _infer_crop_run_storage_mode(run_group: zarr.Group) -> str:
    explicit = run_group.attrs.get("crop_storage_mode")
    if explicit is not None:
        text = str(explicit).strip().lower()
        if text in _VALID_CROP_STORAGE_MODES:
            return text
    if "roi_images" in run_group:
        return "materialized"
    return "geometry_only"


def _set_crop_pixel_contract_attrs(
    crop_group: zarr.Group,
    *,
    crop_storage_mode: str,
    video_source_type: str | None,
    acceleration: str | None,
) -> dict[str, object]:
    contract = crop_run_pixel_contract(
        crop_storage_mode=crop_storage_mode,
        video_source_type=video_source_type,
        acceleration=acceleration,
    )
    crop_group.attrs["roi_image_representation"] = contract.get("image_representation")
    crop_group.attrs["roi_pixel_contract"] = contract
    crop_group.attrs["roi_pixel_contract_name"] = contract.get("name")
    for attr_name in (
        "source_pixels",
        "decode_backend",
        "applied_range_semantics",
        "container_color_range_handling",
        "center_rounding",
    ):
        value = contract.get(attr_name)
        if value is not None:
            crop_group.attrs[attr_name] = value
    crop_group.attrs.setdefault("center_rounding", CENTER_ROUNDING_NP_ROUND)
    return contract


def _round_crop_center_pixels(
    cx_norm: float,
    cy_norm: float,
    *,
    width: int,
    height: int,
) -> Tuple[int, int]:
    """Quantize normalized crop centers with the persisted round convention."""

    center = np.round(
        np.asarray([float(cx_norm) * int(width), float(cy_norm) * int(height)], dtype=np.float64)
    ).astype(np.int64, copy=False)
    return int(center[0]), int(center[1])


def _coerce_existing_crop_pointer(
    crop_parent: zarr.Group,
    value: object,
    *,
    exclude_run_name: Optional[str] = None,
) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text == exclude_run_name:
        return None
    if text not in crop_parent:
        return None
    return text


def _set_or_clear_crop_parent_attr(
    crop_parent: zarr.Group,
    name: str,
    value: Optional[str],
) -> None:
    if value is None:
        if name in crop_parent.attrs:
            del crop_parent.attrs[name]
        return
    crop_parent.attrs[name] = value


def _finalize_crop_parent_pointers(
    crop_parent: zarr.Group,
    *,
    run_name: str,
    crop_storage_mode: str,
    success: bool,
    previous_latest: object,
    previous_latest_materialized: object,
    previous_latest_any: object,
) -> None:
    if success:
        if crop_storage_mode == "materialized":
            latest = run_name
            latest_materialized = run_name
            latest_any = run_name
        else:
            latest_materialized = _coerce_existing_crop_pointer(
                crop_parent,
                previous_latest_materialized,
                exclude_run_name=run_name,
            )
            if latest_materialized is None:
                latest_materialized = _coerce_existing_crop_pointer(
                    crop_parent,
                    previous_latest,
                    exclude_run_name=run_name,
                )
                if latest_materialized is not None and _infer_crop_run_storage_mode(crop_parent[latest_materialized]) != "materialized":
                    latest_materialized = None
            latest = latest_materialized
            latest_any = run_name
    else:
        latest = _coerce_existing_crop_pointer(
            crop_parent,
            previous_latest,
            exclude_run_name=run_name,
        )
        latest_materialized = _coerce_existing_crop_pointer(
            crop_parent,
            previous_latest_materialized,
            exclude_run_name=run_name,
        )
        latest_any = _coerce_existing_crop_pointer(
            crop_parent,
            previous_latest_any,
            exclude_run_name=run_name,
        )

    _set_or_clear_crop_parent_attr(crop_parent, "latest", latest)
    _set_or_clear_crop_parent_attr(crop_parent, "latest_materialized", latest_materialized)
    _set_or_clear_crop_parent_attr(crop_parent, "latest_any", latest_any)


def _compute_roi_coordinates(
    bbox_coords: np.ndarray,
    roi_sz: Tuple[int, int],
    video_shape: Tuple[int, int],
    *,
    scale_factor: Optional[float] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Return top-left ROI coordinates in full-res and optional ds space."""
    roi_h, roi_w = int(roi_sz[0]), int(roi_sz[1])
    video_h, video_w = int(video_shape[0]), int(video_shape[1])
    bbox_coords_array = bbox_coords
    if _CUPY_AVAILABLE and isinstance(bbox_coords_array, cp.ndarray):
        bbox_coords_array = cp.asnumpy(bbox_coords_array)
    elif _TORCH_AVAILABLE and isinstance(bbox_coords_array, torch.Tensor):
        bbox_coords_array = bbox_coords_array.detach().cpu().numpy()
    elif hasattr(bbox_coords_array, "get") and callable(getattr(bbox_coords_array, "get")):
        bbox_coords_array = bbox_coords_array.get()
    centers = np.round(
        np.asarray(bbox_coords_array[:, :2], dtype=np.float32) * np.array([video_w, video_h], dtype=np.float32)
    ).astype(np.int32, copy=False)
    coords_full = np.empty((centers.shape[0], 2), dtype=np.int32)
    coords_full[:, 0] = centers[:, 0] - (roi_w // 2)
    coords_full[:, 1] = centers[:, 1] - (roi_h // 2)

    coords_ds: Optional[np.ndarray] = None
    if scale_factor is not None:
        coords_ds = np.empty_like(coords_full)
        coords_ds[:, 0] = (coords_full[:, 0].astype(np.float32) * float(scale_factor)).astype(np.int32, copy=False)
        coords_ds[:, 1] = (coords_full[:, 1].astype(np.float32) * float(scale_factor)).astype(np.int32, copy=False)

    return coords_full, coords_ds


def _build_crop_stage_provenance(
    *,
    created_at_utc: str,
    command: str,
    env_info: Mapping[str, Any],
    parameters: Mapping[str, Any],
    parameter_source: Optional[str],
    inputs: Mapping[str, Any],
    detection_source: Optional[Mapping[str, Any]] = None,
    scheduler: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    git_info = env_info.get("git") or {}
    platform_info = env_info.get("platform") or {}
    provenance = build_stage_provenance(
        stage="crop",
        command=command,
        created_at_utc=created_at_utc,
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
            "system": platform_info.get("system"),
            "release": platform_info.get("release"),
            "python_version": platform_info.get("python_version"),
            "machine": platform_info.get("machine"),
        },
        parameters=dict(parameters),
        inputs=dict(inputs),
        scheduler=dict(scheduler) if scheduler is not None else None,
    )
    if parameter_source is not None:
        provenance["parameter_source"] = parameter_source
    if detection_source is not None:
        provenance["detection_source"] = dict(detection_source)
    return {key: value for key, value in provenance.items() if value is not None}


def _refresh_crop_quality_inline(
    *,
    root: zarr.Group,
    zarr_file: Path,
    run_name: str,
    console: Optional[Console],
) -> Dict[str, Any]:
    registry_path = RegistryPaths.from_env(Path.cwd()).path.expanduser().resolve()
    registry: Optional[Registry] = None
    try:
        registry = Registry(registry_path)
        dataset_id, row_count = registry.refresh_crop_quality_from_root(root, zarr_file)
        refreshed_row = registry.conn.execute(
            """
            SELECT 1
            FROM crop_quality
            WHERE dataset_id = ? AND crop_run = ?
            LIMIT 1;
            """,
            (dataset_id, str(run_name)),
        ).fetchone()
        return {
            "crop_quality_refresh_status": "ok",
            "crop_quality_refresh_dataset_id": dataset_id,
            "crop_quality_refresh_rows": int(row_count),
            "crop_quality_refresh_run": str(run_name),
            "crop_quality_refresh_run_present": refreshed_row is not None,
        }
    except Exception as exc:
        if console is not None:
            console.print(
                "[yellow]Warning:[/yellow] failed to refresh crop_quality "
                f"for crop run {run_name!r}: {exc}"
            )
        return {
            "crop_quality_refresh_status": "error",
            "crop_quality_refresh_run": str(run_name),
            "crop_quality_refresh_reason": f"{type(exc).__name__}: {exc}",
        }
    finally:
        if registry is not None:
            registry.close()


def _emit_crop_step_status(
    *,
    root: zarr.Group,
    zarr_path: str,
    status: str,
    run_name: Optional[str],
    method: Optional[str],
    coverage_pct: Optional[float],
    review_status: Optional[Dict[str, Any]],
    details: Dict[str, Any],
    console: Optional[Console],
) -> None:
    if os.environ.get(_DISABLE_REGISTRY_WRITES_ENV, "").strip().lower() in {"1", "true", "yes", "on"}:
        if console is not None:
            console.print(
                "[cyan]Registry status write deferred[/cyan] "
                f"({_DISABLE_REGISTRY_WRITES_ENV}=1); batch finalizer must sync crop status."
            )
        return

    zarr_file = Path(zarr_path).expanduser().resolve()
    status_details = dict(details)
    if status == "ok" and run_name:
        status_details.update(
            _refresh_crop_quality_inline(
                root=root,
                zarr_file=zarr_file,
                run_name=run_name,
                console=console,
            )
        )
    emit_stage_completion(
        root,
        zarr_file,
        step_name="crop",
        status=status,
        source=_CROP_STATUS_SOURCE,
        run_name=run_name,
        method=method,
        coverage_pct=coverage_pct,
        review_status_json=review_status,
        details_json=status_details,
        console=console,
        warning_label="crop",
        auto_registry_from_env=True,
        require_env_registry_exists=False,
        invalidate_on_ok=True,
        trigger_run_name=run_name,
    )


def infer_detection_source_type(
    source_path: Optional[str],
    fallback: Optional[str] = None
) -> str:
    """
    Infer the detection source type ('detect', 'filtered', 'interpolated',
    'manual', 'refined', 'auto') from a path.
    
    Args:
        source_path: Path like 'detect_runs/<run>' or the preferred current
            refined override 'refined_detect_runs/<run>/instances'
        fallback: Optional fallback type if the path does not encode it
    
    Returns:
        Normalized detection source type
    """
    valid = {'detect', 'filtered', 'interpolated', 'manual', 'refined', 'auto'}
    fallback_type = fallback if fallback in valid else None
    
    if not source_path:
        return fallback_type or 'detect'
    
    canonical = str(source_path).strip().strip('/')
    if not canonical:
        return fallback_type or 'detect'
    
    last_token = canonical.split('/')[-1]
    if last_token in {'filtered', 'interpolated', 'manual'}:
        return last_token
    
    if canonical.startswith('detect_runs/'):
        return 'detect'
    
    if canonical.startswith(REFINED_DETECT_GROUP) or canonical.startswith(LEGACY_REFINED_DETECT_GROUP):
        return fallback_type or 'refined'
    
    return fallback_type or 'detect'


def _ensure_numpy_array(
    array_like: Any,
    *,
    dtype: Optional[np.dtype | str] = None,
    name: str = "array",
) -> np.ndarray:
    """
    Convert array-like values to a NumPy array, handling GPU-backed arrays that
    require explicit host transfer (e.g., CuPy `.get()`).
    """
    arr_obj: Any = array_like

    if not isinstance(arr_obj, np.ndarray):
        getter = getattr(arr_obj, "get", None)
        if callable(getter):
            try:
                arr_obj = getter()
            except Exception:
                # Fall through to generic conversion path.
                arr_obj = array_like

    if not isinstance(arr_obj, np.ndarray):
        try:
            arr_obj = np.asarray(arr_obj)
        except Exception as exc:  # pragma: no cover - defensive path
            raise TypeError(f"Failed to convert {name} to NumPy array: {exc}") from exc

    if dtype is not None:
        arr_obj = arr_obj.astype(dtype, copy=False)
    return arr_obj


def _extract_detection_rows(
    source_group: zarr.Group,
) -> Tuple[np.ndarray, np.ndarray]:
    payload = _extract_detection_row_payload(source_group)
    return payload["frame_indices"], payload["bbox_norm_coords"]


def _source_label(source_path: Optional[str] = None, source_group: Optional[zarr.Group] = None) -> str:
    if source_path:
        return str(source_path)
    if source_group is not None:
        for attr in ("path", "name"):
            value = getattr(source_group, attr, None)
            if value:
                return str(value)
    return "unknown source"


def _validate_frame_indices_in_bounds(
    frame_indices: np.ndarray,
    *,
    total_frames: int,
    source_label: str,
) -> None:
    frame_indices_np = _ensure_numpy_array(frame_indices, dtype=np.int64, name="frame_indices").reshape(-1)
    total_frames_int = int(total_frames)
    if total_frames_int < 0:
        raise ValueError(f"Cannot validate frame_indices for {source_label}: total_frames={total_frames_int}")
    invalid = np.flatnonzero((frame_indices_np < 0) | (frame_indices_np >= total_frames_int))
    if invalid.size:
        pos = int(invalid[0])
        frame_idx = int(frame_indices_np[pos])
        raise ValueError(
            f"Out-of-range frame index in {source_label}: frame_indices[{pos}]={frame_idx} "
            f"is outside valid range [0, {max(total_frames_int - 1, -1)}] for {total_frames_int} frames"
        )


def _validate_frame_indices_sorted(frame_indices: np.ndarray, *, source_label: str) -> None:
    frame_indices_np = _ensure_numpy_array(frame_indices, dtype=np.int64, name="frame_indices").reshape(-1)
    if frame_indices_np.size < 2:
        return
    out_of_order = np.flatnonzero(np.diff(frame_indices_np) < 0)
    if out_of_order.size:
        prev_pos = int(out_of_order[0])
        pos = prev_pos + 1
        raise ValueError(
            f"Detection rows from {source_label} must be sorted by ascending frame_index; "
            f"first out-of-order row at position {pos}: "
            f"frame_indices[{prev_pos}]={int(frame_indices_np[prev_pos])}, "
            f"frame_indices[{pos}]={int(frame_indices_np[pos])}"
        )


def _extract_optional_detection_row_array(
    source_group: zarr.Group,
    name: str,
    *,
    dtype: np.dtype | str,
    expected_len: int,
) -> Optional[np.ndarray]:
    if name not in source_group:
        return None
    values = _ensure_numpy_array(source_group[name][:], dtype=dtype, name=name).reshape(-1)
    if values.shape[0] != expected_len:
        raise ValueError(
            f"{name} length {values.shape[0]} does not match detection row count {expected_len}"
        )
    return values


def _extract_detection_row_payload(source_group: zarr.Group) -> Dict[str, np.ndarray]:
    """Return source detection rows plus optional stable row identity fields."""
    if has_curated_refined_detect_surface(source_group):
        curated_rows = extract_present_curated_rows(source_group)
        payload: Dict[str, np.ndarray] = {
            "frame_indices": _ensure_numpy_array(
                curated_rows["frame_indices"],
                dtype="i4",
                name="frame_indices",
            ),
            "bbox_norm_coords": _ensure_numpy_array(
                curated_rows["bbox_norm_coords"],
                dtype="f8",
                name="bbox_norm_coords",
            ),
        }
        for source_name, dtype in (
            ("refined_row_ids", "i8"),
            ("source_detect_row_index", "i4"),
            ("instance_key", "u8"),
        ):
            if source_name in curated_rows:
                payload[source_name] = _ensure_numpy_array(
                    curated_rows[source_name],
                    dtype=dtype,
                    name=source_name,
                ).reshape(-1)
        return payload

    frame_indices = _ensure_numpy_array(
        source_group["frame_indices"][:],
        dtype="i4",
        name="frame_indices",
    )
    bbox_norm_coords = _ensure_numpy_array(
        source_group["bbox_norm_coords"][:],
        dtype="f8",
        name="bbox_norm_coords",
    )
    row_count = int(frame_indices.shape[0])
    payload = {
        "frame_indices": frame_indices,
        "bbox_norm_coords": bbox_norm_coords,
    }
    for source_name, dtype in (
        ("refined_row_ids", "i8"),
        ("source_detect_row_index", "i4"),
        ("instance_key", "u8"),
    ):
        values = _extract_optional_detection_row_array(
            source_group,
            source_name,
            dtype=dtype,
            expected_len=row_count,
        )
        if values is not None:
            payload[source_name] = values
    return payload


def _write_optional_detection_row_lineage(
    crop_group: zarr.Group,
    payload: Mapping[str, np.ndarray],
    *,
    total_detections: int,
) -> None:
    source_refined_row_ids = payload.get("refined_row_ids")
    if source_refined_row_ids is not None:
        source_refined_row_ids = _ensure_numpy_array(
            source_refined_row_ids,
            dtype="i8",
            name="refined_row_ids",
        ).reshape(-1)
        if source_refined_row_ids.shape[0] != total_detections:
            raise ValueError(
                "refined_row_ids length "
                f"{source_refined_row_ids.shape[0]} does not match total detections {total_detections}"
            )
        create_geometry_preload_array(
            crop_group,
            "source_refined_row_ids",
            data=source_refined_row_ids,
            overwrite=True,
        )
        crop_group.attrs["source_refined_row_ids_available"] = True
        crop_group.attrs["source_refined_row_id_policy"] = "copied_from_detection_source"
    else:
        crop_group.attrs["source_refined_row_ids_available"] = False

    source_detect_row_index = payload.get("source_detect_row_index")
    if source_detect_row_index is not None:
        source_detect_row_index = _ensure_numpy_array(
            source_detect_row_index,
            dtype="i4",
            name="source_detect_row_index",
        ).reshape(-1)
        if source_detect_row_index.shape[0] != total_detections:
            raise ValueError(
                "source_detect_row_index length "
                f"{source_detect_row_index.shape[0]} does not match total detections {total_detections}"
            )
        create_geometry_preload_array(
            crop_group,
            "source_detect_row_index",
            data=source_detect_row_index,
            overwrite=True,
        )
        crop_group.attrs["source_detect_row_index_available"] = True
    else:
        crop_group.attrs["source_detect_row_index_available"] = False

    instance_key = payload.get("instance_key")
    if instance_key is not None:
        instance_key = _ensure_numpy_array(
            instance_key,
            dtype="u8",
            name="instance_key",
        ).reshape(-1)
        if instance_key.shape[0] != total_detections:
            raise ValueError(
                f"instance_key length {instance_key.shape[0]} does not match total detections {total_detections}"
            )
        create_geometry_preload_array(
            crop_group,
            "instance_key",
            data=instance_key,
            overwrite=True,
        )
        crop_group.attrs["instance_key_available"] = True
        crop_group.attrs["instance_key_policy"] = "copied_from_detection_source"
    else:
        crop_group.attrs["instance_key_available"] = False


def resolve_source_run_info(
    root: zarr.Group,
    source_path: Optional[str]
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Determine the originating detect run, background run, and refined run (if any).
    
    Returns:
        Tuple of (detect_run, background_run, refined_run)
    """
    detect_run = None
    background_run = None
    refined_run = None
    
    normalized = str(source_path).strip().strip('/') if source_path else None
    if not normalized:
        return detect_run, background_run, refined_run
    
    parts = normalized.split('/')
    if len(parts) >= 2 and parts[0] == 'detect_runs':
        detect_run = parts[1]
        detect_group = root.get(f"detect_runs/{detect_run}")
        if isinstance(detect_group, zarr.Group):
            background_run = detect_group.attrs.get('source_background_run')
    elif len(parts) >= 2 and parts[0] in {REFINED_DETECT_GROUP, LEGACY_REFINED_DETECT_GROUP}:
        refined_run = parts[1]
        refined_parent = root.get(parts[0])
        refined_group = refined_parent.get(refined_run) if isinstance(refined_parent, zarr.Group) else None
        if isinstance(refined_group, zarr.Group):
            detect_run = refined_group.attrs.get('source_detect_run')
        if detect_run:
            detect_group = root.get(f"detect_runs/{detect_run}")
            if isinstance(detect_group, zarr.Group):
                background_run = detect_group.attrs.get('source_background_run')
    return detect_run, background_run, refined_run


def _resolve_detect_review_status(
    root: zarr.Group,
    refined_run_name: Optional[str],
    source_path: Optional[str],
) -> Tuple[Optional[Dict[str, object]], Optional[str]]:
    if not refined_run_name:
        return None, None
    parent = None
    if source_path:
        if source_path.startswith(LEGACY_REFINED_DETECT_GROUP):
            parent = root.get(LEGACY_REFINED_DETECT_GROUP)
        elif source_path.startswith(REFINED_DETECT_GROUP):
            parent = root.get(REFINED_DETECT_GROUP)
    if parent is None:
        parent = root.get(REFINED_DETECT_GROUP) or root.get(LEGACY_REFINED_DETECT_GROUP)
    if parent is None or refined_run_name not in parent:
        return None, None
    refined_group = parent[refined_run_name]
    status = refined_group.attrs.get("detect_review_status")
    status_dict = status if isinstance(status, dict) else None
    ref = f"{parent.path}/{refined_run_name}"
    return status_dict, ref

def get_video_source(root: zarr.Group, console: Console) -> Tuple[str, Optional[str]]:
    """
    Determine video source - zarr or external file.
    
    Args:
        root: Zarr root group
        console: Rich console for output
    
    Returns:
        Tuple of (source_type, video_path)
        - source_type: 'zarr' or 'external'
        - video_path: Path to video file (None if zarr)
    """
    from ..shared.metadata import has_raw_video, get_video_source_path
    
    # Try zarr first because it is typically faster.
    if has_raw_video(root):
        console.print("[green]✓[/green] Video source: zarr (raw_video)")
        return 'zarr', None
    
    # Try external video
    video_path = get_video_source_path(root)
    if video_path:
        video_path = Path(video_path)
        if video_path.exists():
            console.print(f"[cyan]✓[/cyan] Video source: external file")
            console.print(f"[dim]  Path: {video_path}[/dim]")
            return 'external', str(video_path)
        else:
            console.print(f"[yellow]⚠[/yellow] Video path in metadata not found: {video_path}")
    
    # No video source found
    raise ValueError(
        "No video source found. Need either:\n"
        "  1. raw_video in zarr (run import stage), OR\n"
        "  2. source_video_path in metadata pointing to valid video file"
    )


def create_crop_batches(
    frame_indices: np.ndarray,
    max_frames_per_batch: int = 32
) -> List[np.ndarray]:
    """
    Group detections into batches where frames are close together.
    
    Args:
        frame_indices: Frame index for each detection
        max_frames_per_batch: Maximum unique frames to decode at once
    
    Returns:
        List of detection index arrays, one per batch
    """
    # Sort detections by frame
    sorted_idx = np.argsort(frame_indices)
    
    batches = []
    current_batch = []
    unique_frames_in_batch = set()
    
    for det_idx in sorted_idx:
        frame = frame_indices[det_idx]
        
        # Start new batch if too many unique frames
        if len(unique_frames_in_batch) >= max_frames_per_batch:
            batches.append(np.array(current_batch))
            current_batch = []
            unique_frames_in_batch = set()
        
        current_batch.append(det_idx)
        unique_frames_in_batch.add(frame)
    
    if current_batch:
        batches.append(np.array(current_batch))
    
    return batches


def crop_batch_gpu(
    video_reader: Any,
    frame_indices: np.ndarray,
    bbox_coords: np.ndarray,
    roi_sz: Tuple[int, int],
    video_shape: Tuple[int, int]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Crop a batch of detections using GPU acceleration.
    
    Args:
        video_reader: Decord VideoReader with GPU context
        frame_indices: Frame index for each detection in batch
        bbox_coords: Normalized bbox coordinates [N, 4]
        roi_sz: (height, width) of ROI
        video_shape: (height, width) of video
    
    Returns:
        Tuple of (crops, coordinates)
    """
    # Get unique frames and create mapping
    unique_frames = np.unique(frame_indices)
    frame_to_idx = {int(f): i for i, f in enumerate(unique_frames)}
    
    console.print(
        f"[debug] crop_batch_gpu: decoding {len(unique_frames)} frames starting at {int(unique_frames[0]) if len(unique_frames) else 'N/A'}"
    )
    decode_start = time.perf_counter()
    # Decode frames on GPU
    frames_gpu = video_reader.get_batch(unique_frames.tolist())  # [N, H, W, C]
    decode_time = time.perf_counter() - decode_start
    console.print(
        f"[debug] crop_batch_gpu: decode completed in {decode_time*1000:.1f} ms"
    )
    
    # Convert to grayscale on GPU
    frames_gray = rgb_to_gray_unweighted_mean_torch(
        frames_gpu,
        accumulator_dtype=torch.float32,
    )  # [N, H, W]
    
    # Prepare outputs
    num_crops = len(frame_indices)
    roi_h, roi_w = roi_sz
    H, W = video_shape
    
    crops_gpu = torch.zeros((num_crops, roi_h, roi_w), dtype=torch.uint8, device='cuda')
    coords_cpu = np.zeros((num_crops, 2), dtype=np.int32)
    
    # Extract each crop
    for i, (frame_idx, bbox) in enumerate(zip(frame_indices, bbox_coords)):
        frame_batch_idx = frame_to_idx[int(frame_idx)]
        frame_tensor = frames_gray[frame_batch_idx]
        
        # Calculate crop coordinates
        cx_norm, cy_norm = bbox[:2]
        cx_px, cy_px = _round_crop_center_pixels(cx_norm, cy_norm, width=W, height=H)
        
        x1 = cx_px - roi_w // 2
        y1 = cy_px - roi_h // 2
        x2 = x1 + roi_w
        y2 = y1 + roi_h
        
        coords_cpu[i] = (x1, y1)
        
        # Extract with bounds checking
        if 0 <= x1 and x2 <= W and 0 <= y1 and y2 <= H:
            crops_gpu[i] = frame_tensor[y1:y2, x1:x2]
        else:
            vy1 = max(0, y1); vy2 = min(H, y2)
            vx1 = max(0, x1); vx2 = min(W, x2)
            
            if vy2 > vy1 and vx2 > vx1:
                py1 = max(0, -y1)
                px1 = max(0, -x1)
                py2 = py1 + (vy2 - vy1)
                px2 = px1 + (vx2 - vx1)
                
                crops_gpu[i, py1:py2, px1:px2] = frame_tensor[vy1:vy2, vx1:vx2]
    
    # Copy to CPU
    crops_cpu = crops_gpu.cpu().numpy()
    
    # Cleanup
    del frames_gpu, frames_gray, crops_gpu
    torch.cuda.empty_cache()
    
    return crops_cpu, coords_cpu


def crop_batch_cpu(
    video_path: str,
    frame_indices: np.ndarray,
    bbox_coords: np.ndarray,
    roi_sz: Tuple[int, int],
    video_shape: Tuple[int, int],
    *,
    total_frames: Optional[int] = None,
    source_label: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """
    Crop a batch of detections using CPU (OpenCV).
    """
    source = source_label or str(video_path)
    if total_frames is not None:
        _validate_frame_indices_in_bounds(frame_indices, total_frames=int(total_frames), source_label=source)
    elif np.any(_ensure_numpy_array(frame_indices, dtype=np.int64, name="frame_indices") < 0):
        _validate_frame_indices_in_bounds(frame_indices, total_frames=0, source_label=source)

    cap = cv2.VideoCapture(str(video_path))
    
    # Get unique frames and decode
    decode_start = time.perf_counter()
    unique_frames = np.unique(frame_indices)
    frame_cache = {}
    
    for frame_idx in unique_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ret, frame = cap.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_cache[int(frame_idx)] = gray
    decode_seconds = time.perf_counter() - decode_start

    cap.release()

    missing_frames = [int(frame_idx) for frame_idx in unique_frames if int(frame_idx) not in frame_cache]
    if missing_frames:
        sample = ", ".join(str(frame_idx) for frame_idx in missing_frames[:8])
        raise RuntimeError(
            f"CPU crop decode failed for frame(s) {sample} from {source}; "
            "refusing to write zero-filled crops with plausible coordinates"
        )
    
    # Extract crops
    num_crops = len(frame_indices)
    roi_h, roi_w = roi_sz
    H, W = video_shape
    
    crops = np.zeros((num_crops, roi_h, roi_w), dtype=np.uint8)
    coords = np.zeros((num_crops, 2), dtype=np.int32)
    
    compute_start = time.perf_counter()
    for i, (frame_idx, bbox) in enumerate(zip(frame_indices, bbox_coords)):
        frame = frame_cache.get(int(frame_idx))
        if frame is None:
            continue
        
        cx_norm, cy_norm = bbox[:2]
        cx_px, cy_px = _round_crop_center_pixels(cx_norm, cy_norm, width=W, height=H)
        
        x1 = cx_px - roi_w // 2
        y1 = cy_px - roi_h // 2
        x2 = x1 + roi_w
        y2 = y1 + roi_h
        
        coords[i] = (x1, y1)
        
        if 0 <= x1 and x2 <= W and 0 <= y1 and y2 <= H:
            crops[i] = frame[y1:y2, x1:x2]
        else:
            vy1 = max(0, y1); vy2 = min(H, y2)
            vx1 = max(0, x1); vx2 = min(W, x2)
            
            if vy2 > vy1 and vx2 > vx1:
                py1 = max(0, -y1)
                px1 = max(0, -x1)
                py2 = py1 + (vy2 - vy1)
                px2 = px1 + (vx2 - vx1)
                
                crops[i, py1:py2, px1:px2] = frame[vy1:vy2, vx1:vx2]
    compute_seconds = time.perf_counter() - compute_start

    return crops, coords, {
        "decode_seconds": float(decode_seconds),
        "compute_seconds": float(compute_seconds),
    }


def crop_batch_cpu_from_top_left(
    video_path: str,
    frame_indices: np.ndarray,
    roi_coordinates_full: np.ndarray,
    roi_sz: Tuple[int, int],
    video_shape: Tuple[int, int],
    *,
    total_frames: Optional[int] = None,
    source_label: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """Crop a batch of detections using stored ROI top-left coordinates on CPU."""
    source = source_label or str(video_path)
    if total_frames is not None:
        _validate_frame_indices_in_bounds(frame_indices, total_frames=int(total_frames), source_label=source)
    elif np.any(_ensure_numpy_array(frame_indices, dtype=np.int64, name="frame_indices") < 0):
        _validate_frame_indices_in_bounds(frame_indices, total_frames=0, source_label=source)

    cap = cv2.VideoCapture(str(video_path))

    decode_start = time.perf_counter()
    unique_frames = np.unique(frame_indices)
    frame_cache = {}

    for frame_idx in unique_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ret, frame = cap.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_cache[int(frame_idx)] = gray
    decode_seconds = time.perf_counter() - decode_start

    cap.release()

    missing_frames = [int(frame_idx) for frame_idx in unique_frames if int(frame_idx) not in frame_cache]
    if missing_frames:
        sample = ", ".join(str(frame_idx) for frame_idx in missing_frames[:8])
        raise RuntimeError(
            f"CPU crop decode failed for frame(s) {sample} from {source}; "
            "refusing to write zero-filled crops with plausible coordinates"
        )

    num_crops = len(frame_indices)
    roi_h, roi_w = roi_sz
    H, W = video_shape

    crops = np.zeros((num_crops, roi_h, roi_w), dtype=np.uint8)
    coords = np.zeros((num_crops, 2), dtype=np.int32)

    compute_start = time.perf_counter()
    for i, (frame_idx, coord) in enumerate(zip(frame_indices, roi_coordinates_full)):
        frame = frame_cache.get(int(frame_idx))
        if frame is None:
            continue

        x1 = int(coord[0])
        y1 = int(coord[1])
        x2 = x1 + roi_w
        y2 = y1 + roi_h

        coords[i] = (x1, y1)

        if 0 <= x1 and x2 <= W and 0 <= y1 and y2 <= H:
            crops[i] = frame[y1:y2, x1:x2]
        else:
            vy1 = max(0, y1); vy2 = min(H, y2)
            vx1 = max(0, x1); vx2 = min(W, x2)

            if vy2 > vy1 and vx2 > vx1:
                py1 = max(0, -y1)
                px1 = max(0, -x1)
                py2 = py1 + (vy2 - vy1)
                px2 = px1 + (vx2 - vx1)

                crops[i, py1:py2, px1:px2] = frame[vy1:vy2, vx1:vx2]
    compute_seconds = time.perf_counter() - compute_start

    return crops, coords, {
        "decode_seconds": float(decode_seconds),
        "compute_seconds": float(compute_seconds),
    }


def _process_chunk_gpu(
    chunk_idx: int,
    chunk_frames: List[int],
    frames_gpu: "torch.Tensor",
    frame_to_det: Dict[int, List[int]],
    bbox_coords: np.ndarray,
    roi_sz: Tuple[int, int],
    video_shape: Tuple[int, int],
    *,
    return_device: bool = False,
) -> Tuple[int, np.ndarray, Any, np.ndarray, float]:
    """Process a contiguous chunk of frames on GPU and return crops/co-ordinates."""
    start = time.perf_counter()
    # Keep conversion in fp16 to reduce transient GPU memory on high-res frames.
    frames_gray = rgb_to_gray_unweighted_mean_torch(
        frames_gpu,
        accumulator_dtype=torch.float16,
    )
    H, W = video_shape

    chunk_det_indices: List[int] = []
    chunk_coords_full: List[Tuple[int, int]] = []
    chunk_rois: List[torch.Tensor] = []

    for local_idx, frame_idx in enumerate(chunk_frames):
        det_list = frame_to_det.get(int(frame_idx))
        if not det_list:
            continue

        frame_tensor = frames_gray[local_idx]
        for det_idx in det_list:
            bbox = bbox_coords[det_idx]
            cx_norm, cy_norm = bbox[:2]
            cx_px, cy_px = _round_crop_center_pixels(cx_norm, cy_norm, width=W, height=H)

            x1 = cx_px - roi_sz[1] // 2
            y1 = cy_px - roi_sz[0] // 2
            x2 = x1 + roi_sz[1]
            y2 = y1 + roi_sz[0]

            chunk_det_indices.append(det_idx)
            chunk_coords_full.append((x1, y1))

            if 0 <= x1 and x2 <= W and 0 <= y1 and y2 <= H:
                roi_tensor = frame_tensor[y1:y2, x1:x2].clone()
            else:
                roi_tensor = torch.zeros((roi_sz[0], roi_sz[1]), dtype=torch.uint8, device=frame_tensor.device)
                vy1 = max(0, y1); vy2 = min(H, y2)
                vx1 = max(0, x1); vx2 = min(W, x2)
                if vy2 > vy1 and vx2 > vx1:
                    py1 = max(0, -y1); py2 = py1 + (vy2 - vy1)
                    px1 = max(0, -x1); px2 = px1 + (vx2 - vx1)
                    roi_tensor[py1:py2, px1:px2] = frame_tensor[vy1:vy2, vx1:vx2]

            chunk_rois.append(roi_tensor)

    chunk_time = time.perf_counter() - start

    if not chunk_rois:
        if return_device:
            return (
                chunk_idx,
                np.array([], dtype=np.int64),
                torch.empty((0, roi_sz[0], roi_sz[1]), dtype=torch.uint8, device=frames_gray.device),
                np.empty((0, 2), dtype=np.int32),
                chunk_time
            )
        return (
            chunk_idx,
            np.array([], dtype=np.int64),
            np.empty((0, roi_sz[0], roi_sz[1]), dtype=np.uint8),
            np.empty((0, 2), dtype=np.int32),
            chunk_time
        )

    crops_gpu = torch.stack(chunk_rois, dim=0)
    coords_full_cpu = np.array(chunk_coords_full, dtype=np.int32)
    det_indices_np = np.array(chunk_det_indices, dtype=np.int64)
    if return_device:
        return chunk_idx, det_indices_np, crops_gpu, coords_full_cpu, chunk_time

    crops_cpu = crops_gpu.cpu().numpy()

    return chunk_idx, det_indices_np, crops_cpu, coords_full_cpu, chunk_time


def _process_chunk_gpu_from_top_left(
    chunk_idx: int,
    chunk_frames: List[int],
    frames_gpu: "torch.Tensor",
    frame_to_roi: Dict[int, List[int]],
    roi_coordinates_full: np.ndarray,
    roi_sz: Tuple[int, int],
    video_shape: Tuple[int, int],
    *,
    return_device: bool = False,
) -> Tuple[int, np.ndarray, Any, np.ndarray, float]:
    """Process a contiguous chunk of frames on GPU using stored ROI top-left coordinates."""
    start = time.perf_counter()
    frames_gray = rgb_to_gray_unweighted_mean_torch(
        frames_gpu,
        accumulator_dtype=torch.float16,
    )
    H, W = video_shape

    chunk_roi_indices: List[int] = []
    chunk_coords_full: List[Tuple[int, int]] = []
    chunk_rois: List[torch.Tensor] = []

    for local_idx, frame_idx in enumerate(chunk_frames):
        roi_list = frame_to_roi.get(int(frame_idx))
        if not roi_list:
            continue

        frame_tensor = frames_gray[local_idx]
        for roi_idx in roi_list:
            coord = roi_coordinates_full[roi_idx]
            x1 = int(coord[0])
            y1 = int(coord[1])
            x2 = x1 + roi_sz[1]
            y2 = y1 + roi_sz[0]

            chunk_roi_indices.append(roi_idx)
            chunk_coords_full.append((x1, y1))

            if 0 <= x1 and x2 <= W and 0 <= y1 and y2 <= H:
                roi_tensor = frame_tensor[y1:y2, x1:x2].clone()
            else:
                roi_tensor = torch.zeros((roi_sz[0], roi_sz[1]), dtype=torch.uint8, device=frame_tensor.device)
                vy1 = max(0, y1); vy2 = min(H, y2)
                vx1 = max(0, x1); vx2 = min(W, x2)
                if vy2 > vy1 and vx2 > vx1:
                    py1 = max(0, -y1); py2 = py1 + (vy2 - vy1)
                    px1 = max(0, -x1); px2 = px1 + (vx2 - vx1)
                    roi_tensor[py1:py2, px1:px2] = frame_tensor[vy1:vy2, vx1:vx2]

            chunk_rois.append(roi_tensor)

    chunk_time = time.perf_counter() - start

    if not chunk_rois:
        if return_device:
            return (
                chunk_idx,
                np.array([], dtype=np.int64),
                torch.empty((0, roi_sz[0], roi_sz[1]), dtype=torch.uint8, device=frames_gray.device),
                np.empty((0, 2), dtype=np.int32),
                chunk_time,
            )
        return (
            chunk_idx,
            np.array([], dtype=np.int64),
            np.empty((0, roi_sz[0], roi_sz[1]), dtype=np.uint8),
            np.empty((0, 2), dtype=np.int32),
            chunk_time,
        )

    crops_gpu = torch.stack(chunk_rois, dim=0)
    coords_full_cpu = np.array(chunk_coords_full, dtype=np.int32)
    roi_indices_np = np.array(chunk_roi_indices, dtype=np.int64)
    if return_device:
        return chunk_idx, roi_indices_np, crops_gpu, coords_full_cpu, chunk_time

    crops_cpu = crops_gpu.cpu().numpy()

    return chunk_idx, roi_indices_np, crops_cpu, coords_full_cpu, chunk_time


def _contiguous_detection_slice(det_ids: np.ndarray) -> Optional[slice]:
    """Return contiguous slice if det_ids form [start, start+1, ..., end-1]."""
    if det_ids.size == 0:
        return None
    start = int(det_ids[0])
    stop = int(det_ids[-1]) + 1
    if stop <= start:
        return None
    if (stop - start) != int(det_ids.size):
        return None
    if not np.all(det_ids == np.arange(start, stop, dtype=det_ids.dtype)):
        return None
    return slice(start, stop)


def materialize_external_roi_cache(
    *,
    cache_path: str | Path,
    video_path: str | Path,
    frame_indices: np.ndarray,
    roi_coordinates_full: np.ndarray,
    roi_sz: Tuple[int, int],
    video_shape: Tuple[int, int],
    console: Optional[Console] = None,
    write_backend: str = "kvikio",
    roi_storage: str = "uncompressed",
    use_sharding: bool = False,
    roi_chunk_size: int = DEFAULT_SCRATCH_ROI_CACHE_CHUNK_LEN,
    roi_shard_size: Optional[int] = None,
    gpu_chunk_frames: int = DEFAULT_SCRATCH_ROI_CACHE_GPU_CHUNK_FRAMES,
    require_kvikio: bool = False,
    prefer_gpu: bool = True,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Materialize ROI pixels into a temporary cache using the external crop workflow."""
    cache_path = Path(cache_path).expanduser().resolve()
    video_path = Path(video_path).expanduser().resolve()
    frame_indices_np = _ensure_numpy_array(frame_indices, dtype=np.int64, name="frame_indices")
    roi_coordinates_np = _ensure_numpy_array(
        roi_coordinates_full,
        dtype=np.int32,
        name="roi_coordinates_full",
    )
    total_rois = int(frame_indices_np.shape[0])
    if roi_coordinates_np.shape[0] != total_rois:
        raise ValueError(
            f"roi_coordinates_full length {roi_coordinates_np.shape[0]} does not match frame_indices length {total_rois}"
        )

    backend_norm = str(write_backend or "standard").strip().lower()
    storage_norm = normalize_crop_roi_storage(roi_storage, default="compressed")
    if backend_norm not in {"standard", "kvikio"}:
        backend_norm = "standard"
    layout = build_scratch_roi_cache_layout(
        total_rois=total_rois,
        preferred_chunk_len=int(roi_chunk_size),
    )
    roi_chunk_len = layout.roi_chunk_len
    shard_len = layout.roi_shard_len if layout.roi_shard_len is not None else roi_chunk_len
    gpu_chunk_frames = max(1, int(gpu_chunk_frames))

    use_gpu = False
    gpu_reason = "GPU disabled"
    if prefer_gpu:
        use_gpu, gpu_reason = check_gpu_crop_available()
        if not use_gpu:
            raise _gpu_decode_unavailable(gpu_reason)

    use_kvikio_writes = False
    fallback_reason: Optional[str] = None
    store = None
    if require_kvikio and backend_norm != "kvikio":
        raise ValueError("require_kvikio requires write_backend='kvikio'")

    if backend_norm == "kvikio":
        if not use_gpu:
            fallback_reason = "gpu_unavailable"
        elif not _KVIKIO_AVAILABLE:
            fallback_reason = "kvikio_unavailable"
        elif not _CUPY_AVAILABLE:
            fallback_reason = "cupy_unavailable"
        else:
            kvikio_defaults.set(
                {
                    "num_threads": 8,
                    "task_size": 32 * 1024 * 1024,
                    "bounce_buffer_size": 64 * 1024 * 1024,
                    "gds_threshold": 1024,
                }
            )
            zarr.config.enable_gpu()
            store = kvikio.zarr.GDSStore(str(cache_path))
            cache_root = zarr.open_group(store=store, mode="a")
            use_kvikio_writes = True
    if require_kvikio and not use_kvikio_writes:
        detail = fallback_reason or "kvikio_write_path_not_available"
        raise RuntimeError(f"kvikio required for ROI cache writes but unavailable ({detail})")
    if not use_kvikio_writes:
        cache_root = zarr.open_group(str(cache_path), mode="a")

    effective_backend = "kvikio_gds" if use_kvikio_writes else "standard_zarr"
    if console is not None:
        accel_label = "gpu" if use_gpu else "cpu"
        detail = f"backend={effective_backend}, acceleration={accel_label}, source={video_path.name}"
        if fallback_reason:
            detail += f", fallback={fallback_reason}"
        console.print(f"[dim]Temporary ROI cache materialization: {detail}[/dim]")

    roi_create_kwargs = build_crop_roi_create_kwargs(
        total_rois=total_rois,
        roi_sz=roi_sz,
        layout=layout,
        overwrite=True,
    )
    roi_images = cache_root.create_array("roi_images", **roi_create_kwargs)
    cache_root.attrs.update(crop_roi_layout_attrs(layout))
    cache_root.attrs["cache_layout_profile"] = SCRATCH_ROI_CACHE_LAYOUT_PROFILE
    pixel_contract = crop_run_pixel_contract(
        crop_storage_mode="materialized",
        video_source_type="external",
        acceleration="gpu" if use_gpu else "cpu",
    )
    cache_root.attrs["roi_image_representation"] = pixel_contract.get("image_representation")
    cache_root.attrs["roi_pixel_contract"] = pixel_contract
    cache_root.attrs["roi_pixel_contract_name"] = pixel_contract.get("name")

    decode_seconds = 0.0
    compute_seconds = 0.0
    write_seconds = 0.0
    start_time = time.perf_counter()
    video_reader = None
    try:
        if total_rois == 0:
            return {
                "total_rois": 0,
                "write_backend_requested": backend_norm,
                "write_backend_effective": "kvikio_gds" if use_kvikio_writes else "standard_zarr",
                "acceleration": "gpu" if use_gpu else "cpu",
                "fallback_reason": fallback_reason,
            }

        if use_gpu:
            decord.bridge.set_bridge('torch')
            video_reader = VideoReader(str(video_path), ctx=gpu(0))
            frame_to_roi: Dict[int, List[int]] = {}
            for roi_idx, frame_idx in enumerate(frame_indices_np):
                frame_to_roi.setdefault(int(frame_idx), []).append(int(roi_idx))

            frame_min = int(frame_indices_np.min())
            frame_max = int(frame_indices_np.max())
            chunk_starts = list(range(frame_min, frame_max + 1, gpu_chunk_frames))
            progress_every = max(1, len(chunk_starts) // 20) if chunk_starts else 1
            processed_total = 0
            for chunk_idx, chunk_start in enumerate(chunk_starts):
                chunk_end = min(chunk_start + gpu_chunk_frames, frame_max + 1)
                chunk_frames = list(range(chunk_start, chunk_end))
                if not any(frame in frame_to_roi for frame in chunk_frames):
                    continue

                decode_start = time.perf_counter()
                frames_gpu = video_reader.get_batch(chunk_frames)
                decode_seconds += time.perf_counter() - decode_start
                compute_start = time.perf_counter()
                _, roi_ids, crops_buf, _coords_cpu, _chunk_time = _process_chunk_gpu_from_top_left(
                    chunk_idx,
                    chunk_frames,
                    frames_gpu,
                    frame_to_roi,
                    roi_coordinates_np,
                    roi_sz,
                    video_shape,
                    return_device=use_kvikio_writes,
                )
                compute_seconds += time.perf_counter() - compute_start
                del frames_gpu

                processed = int(roi_ids.size)
                if not processed:
                    continue

                write_start = time.perf_counter()
                if use_kvikio_writes:
                    roi_slice = _contiguous_detection_slice(roi_ids)
                    if roi_slice is not None:
                        cupy_crops = cp.from_dlpack(crops_buf.contiguous())
                        roi_images[roi_slice] = cupy_crops
                        cp.cuda.Stream.null.synchronize()
                        del cupy_crops
                    else:
                        crops_cpu_fallback = crops_buf.cpu().numpy()
                        roi_images[roi_ids] = crops_cpu_fallback
                        del crops_cpu_fallback
                else:
                    roi_images[roi_ids] = crops_buf
                write_seconds += time.perf_counter() - write_start
                processed_total += processed
                if console is not None and (
                    processed_total == total_rois
                    or (chunk_idx + 1) % progress_every == 0
                ):
                    pct = (processed_total / total_rois) * 100 if total_rois > 0 else 100.0
                    console.print(
                        f"[dim]  Cache progress: {processed_total:,}/{total_rois:,} ROIs ({pct:.1f}%)[/dim]"
                    )
        else:
            batches = create_crop_batches(frame_indices_np, max_frames_per_batch=32)
            progress_every = max(1, len(batches) // 20) if batches else 1
            processed_total = 0
            for batch_idx, roi_ids in enumerate(batches):
                crops, _coords, batch_profile = crop_batch_cpu_from_top_left(
                    str(video_path),
                    frame_indices_np[roi_ids],
                    roi_coordinates_np[roi_ids],
                    roi_sz,
                    video_shape,
                    source_label=str(video_path),
                )
                decode_seconds += float(batch_profile.get("decode_seconds", 0.0))
                compute_seconds += float(batch_profile.get("compute_seconds", 0.0))
                write_start = time.perf_counter()
                roi_images[roi_ids] = crops
                write_seconds += time.perf_counter() - write_start
                processed_total += int(len(roi_ids))
                if console is not None and (
                    processed_total == total_rois
                    or (batch_idx + 1) % progress_every == 0
                ):
                    pct = (processed_total / total_rois) * 100 if total_rois > 0 else 100.0
                    console.print(
                        f"[dim]  Cache progress: {processed_total:,}/{total_rois:,} ROIs ({pct:.1f}%)[/dim]"
                    )
    finally:
        if video_reader is not None:
            video_reader = None
        if _TORCH_AVAILABLE and use_gpu and torch.cuda.is_available():
            torch.cuda.empty_cache()
        if store is not None:
            try:
                store.close()
            except Exception:
                pass

    duration = time.perf_counter() - start_time
    return {
        "total_rois": total_rois,
        "duration_seconds": duration,
        "decode_seconds": float(decode_seconds),
        "compute_seconds": float(compute_seconds),
        "write_seconds": float(write_seconds),
        "write_backend_requested": backend_norm,
        "write_backend_effective": effective_backend,
        "acceleration": "gpu" if use_gpu else "cpu",
        "fallback_reason": fallback_reason,
        "roi_chunk_len": int(roi_chunk_len),
        "roi_shard_len": int(shard_len),
        "roi_storage": layout.roi_storage,
        "roi_use_sharding": bool(layout.roi_use_sharding),
        "roi_layout_profile": SCRATCH_ROI_CACHE_LAYOUT_PROFILE,
        "gpu_chunk_frames": int(gpu_chunk_frames),
        "roi_image_representation": pixel_contract.get("image_representation"),
        "roi_pixel_contract": pixel_contract,
        "roi_pixel_contract_name": pixel_contract.get("name"),
        "video_path": str(video_path),
        "cache_path": str(cache_path),
        "verbose": bool(verbose),
    }


def _load_external_roi_cache_inputs(
    *,
    source_zarr_path: str | Path,
    crop_run_name: str,
) -> Dict[str, Any]:
    source_zarr_path = Path(source_zarr_path).expanduser().resolve()
    root = zarr.open_group(str(source_zarr_path), mode="r")
    crop_parent = root.get("crop_runs")
    if crop_parent is None:
        raise KeyError(f"Missing crop_runs in {source_zarr_path}")
    if crop_run_name not in crop_parent:
        raise KeyError(f"Crop run '{crop_run_name}' not found in {source_zarr_path}")
    crop_group = crop_parent[crop_run_name]

    if "frame_indices" not in crop_group:
        raise KeyError(f"Crop run '{crop_run_name}' is missing frame_indices")
    if "roi_coordinates_full" not in crop_group:
        raise KeyError(f"Crop run '{crop_run_name}' is missing roi_coordinates_full")

    roi_size_attr = crop_group.attrs.get("roi_size")
    if isinstance(roi_size_attr, (list, tuple)) and len(roi_size_attr) == 2:
        roi_sz = (int(roi_size_attr[0]), int(roi_size_attr[1]))
    elif "roi_images" in crop_group and len(crop_group["roi_images"].shape) >= 3:
        roi_images_shape = crop_group["roi_images"].shape
        roi_sz = (int(roi_images_shape[1]), int(roi_images_shape[2]))
    else:
        raise ValueError(f"Crop run '{crop_run_name}' is missing roi_size metadata")

    height = crop_group.attrs.get("height", root.attrs.get("height"))
    width = crop_group.attrs.get("width", root.attrs.get("width"))
    if height is None or width is None:
        raise ValueError(
            f"Unable to determine video shape for crop run '{crop_run_name}' in {source_zarr_path}"
        )

    video_path = (
        crop_group.attrs.get("source_video_path")
        or crop_group.attrs.get("video_source_path")
    )
    if not video_path:
        video_path = get_video_source_path(root, zarr_path=source_zarr_path)
    if not video_path:
        raise ValueError(
            f"Unable to determine source video path for crop run '{crop_run_name}' in {source_zarr_path}"
        )

    return {
        "video_path": str(video_path),
        "frame_indices": np.asarray(crop_group["frame_indices"][:], dtype=np.int64),
        "roi_coordinates_full": np.asarray(crop_group["roi_coordinates_full"][:], dtype=np.int32),
        "roi_sz": roi_sz,
        "video_shape": (int(height), int(width)),
    }


def materialize_external_roi_cache_for_crop_run(
    *,
    cache_path: str | Path,
    source_zarr_path: str | Path,
    crop_run_name: str,
    console: Optional[Console] = None,
    write_backend: str = "kvikio",
    roi_storage: str = "uncompressed",
    use_sharding: bool = False,
    roi_chunk_size: int = DEFAULT_SCRATCH_ROI_CACHE_CHUNK_LEN,
    roi_shard_size: Optional[int] = None,
    gpu_chunk_frames: int = DEFAULT_SCRATCH_ROI_CACHE_GPU_CHUNK_FRAMES,
    require_kvikio: bool = False,
    prefer_gpu: bool = True,
    verbose: bool = False,
    isolate_process: bool = True,
) -> Dict[str, Any]:
    cache_path = Path(cache_path).expanduser().resolve()
    source_zarr_path = Path(source_zarr_path).expanduser().resolve()

    if not isolate_process:
        inputs = _load_external_roi_cache_inputs(
            source_zarr_path=source_zarr_path,
            crop_run_name=crop_run_name,
        )
        return materialize_external_roi_cache(
            cache_path=cache_path,
            console=console,
            write_backend=write_backend,
            roi_storage=roi_storage,
            use_sharding=use_sharding,
            roi_chunk_size=roi_chunk_size,
            roi_shard_size=roi_shard_size,
            gpu_chunk_frames=gpu_chunk_frames,
            require_kvikio=require_kvikio,
            prefer_gpu=prefer_gpu,
            verbose=verbose,
            **inputs,
        )

    cmd = [
        sys.executable,
        "-m",
        "fisheye.tracking.crop",
        "--roi-cache-worker",
        "--source-zarr",
        str(source_zarr_path),
        "--source-crop-run",
        str(crop_run_name),
        "--cache-path",
        str(cache_path),
        "--write-backend",
        str(write_backend),
        "--roi-storage",
        str(roi_storage),
        "--roi-chunk-size",
        str(int(roi_chunk_size)),
        "--gpu-chunk-frames",
        str(int(gpu_chunk_frames)),
    ]
    if roi_shard_size is not None:
        cmd.extend(["--roi-shard-size", str(int(roi_shard_size))])
    if use_sharding:
        cmd.append("--use-sharding")
    if require_kvikio:
        cmd.append("--require-kvikio")
    if prefer_gpu:
        cmd.append("--prefer-gpu")
    else:
        cmd.append("--no-prefer-gpu")
    if verbose:
        cmd.append("--verbose")

    if console is not None:
        console.print("[dim]Spawning isolated ROI cache worker process[/dim]")
    completed = subprocess.run(cmd, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"ROI cache worker failed with exit code {completed.returncode}")

    cache_root = zarr.open_group(str(cache_path), mode="r")
    worker_result = cache_root.attrs.get("roi_cache_worker_result")
    if isinstance(worker_result, Mapping):
        return dict(worker_result)

    return {
        "cache_path": str(cache_path),
        "write_backend_requested": normalize_attr(cache_root.attrs.get("cache_write_backend_requested")),
        "write_backend_effective": normalize_attr(cache_root.attrs.get("cache_write_backend_effective")),
        "acceleration": normalize_attr(cache_root.attrs.get("cache_acceleration")),
        "fallback_reason": normalize_attr(cache_root.attrs.get("cache_fallback_reason")),
    }


def crop_from_external_video(
    zarr_path: str,
    video_path: str,
    source_path: str,
    source_group: zarr.Group,
    detection_source: Optional[np.ndarray],
    source_type: str,
    roi_sz: Tuple[int, int],
    use_gpu: bool,
    console: Console,
    selection_policy: Optional[str] = None,
    external_write_backend: str = "standard",
    external_roi_storage: str = "compressed",
    external_use_sharding: bool = False,
    external_roi_chunk_size: int = 1024,
    external_roi_shard_size: Optional[int] = None,
    external_gpu_chunk_frames: int = 96,
    external_require_kvikio: bool = False,
    crop_storage_mode: str = "materialized",
    verbose: bool = False,
    cli_provenance: Optional[Mapping[str, Any]] = None,
    run_provenance: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Crop detections from external video file (GPU or CPU).
    """
    start_time = time.perf_counter()
    
    use_kvikio_writes = False
    backend_norm = str(external_write_backend or "standard").strip().lower()
    fallback_reason: Optional[str] = None
    roi_storage_norm = str(external_roi_storage or "compressed").strip().lower()
    if roi_storage_norm not in {"compressed", "uncompressed"}:
        roi_storage_norm = "compressed"

    if external_require_kvikio and backend_norm != "kvikio":
        raise ValueError("external_require_kvikio requires external_write_backend='kvikio'")

    if backend_norm == "kvikio":
        if not use_gpu:
            fallback_reason = "gpu_mode_disabled"
            console.print("[yellow]kvikio writes requested but GPU mode disabled; falling back to standard writes[/yellow]")
        elif not _KVIKIO_AVAILABLE:
            fallback_reason = "kvikio_unavailable"
            console.print("[yellow]kvikio writes requested but kvikio is unavailable; falling back to standard writes[/yellow]")
        elif not _CUPY_AVAILABLE:
            fallback_reason = "cupy_unavailable"
            console.print("[yellow]kvikio writes requested but cupy is unavailable; falling back to standard writes[/yellow]")
        else:
            kvikio_defaults.set(
                {
                    "num_threads": 8,
                    "task_size": 32 * 1024 * 1024,
                    "bounce_buffer_size": 64 * 1024 * 1024,
                    "gds_threshold": 1024,
                }
            )
            zarr.config.enable_gpu()
            store = kvikio.zarr.GDSStore(str(zarr_path))
            root = zarr.open_group(store=store, mode='a')
            use_kvikio_writes = True
            console.print("[green]✓ External crop writes using kvikIO GDSStore[/green]")
    if external_require_kvikio and not use_kvikio_writes:
        detail = fallback_reason or "kvikio_write_path_not_available"
        raise RuntimeError(f"kvikio required for crop writes but unavailable ({detail})")
    if not use_kvikio_writes:
        root = zarr.open(zarr_path, mode='a')
    crop_storage_mode = _normalize_crop_storage_mode(crop_storage_mode)
    crop_parent_before = root.get("crop_runs")
    previous_latest = crop_parent_before.attrs.get("latest") if crop_parent_before is not None else None
    previous_latest_materialized = (
        crop_parent_before.attrs.get("latest_materialized") if crop_parent_before is not None else None
    )
    previous_latest_any = crop_parent_before.attrs.get("latest_any") if crop_parent_before is not None else None
    
    video_reader = None
    crop_group: Optional[zarr.Group] = None
    run_name: Optional[str] = None
    success = False
    error_message: Optional[str] = None
    started_at = datetime.now(timezone.utc).isoformat()
    decode_seconds = 0.0
    compute_seconds = 0.0
    write_seconds = 0.0
    
    actual_use_gpu = use_gpu
    try:
        # Load detection data
        frame_indices, bbox_coords = _extract_detection_rows(source_group)
        total_detections = frame_indices.shape[0]
        
        if total_detections == 0:
            console.print("[yellow]No detections to crop[/yellow]")
            return {'total_crops': 0}
        
        # Get video dimensions from metadata
        video_height = root.attrs['height']
        video_width = root.attrs['width']
        video_shape = (video_height, video_width)
        num_frames = root.attrs['total_frames']
        _validate_frame_indices_in_bounds(
            frame_indices,
            total_frames=int(num_frames),
            source_label=_source_label(source_path, source_group),
        )
        
        console.print(f"Total detections: {total_detections:,}")
        console.print(f"Video dimensions: {video_width}x{video_height}")
        console.print(f"ROI size: {roi_sz[1]}x{roi_sz[0]}")
        
        # Create crop group
        from ..shared.zarr.schema import get_run_group
        crop_group, run_name = get_run_group(
            root,
            'crop',
            console,
            completion_epoch=COMPLETION_EPOCH_REQUIRE_PROVENANCE,
        )
        crop_parent = root.get("crop_runs")
        if crop_parent is not None and run_name is not None:
            mark_run_started(crop_group, run_name=run_name, stage="crop")
            note_pending_latest(crop_parent, run_name)
            _finalize_crop_parent_pointers(
                crop_parent,
                run_name=run_name,
                crop_storage_mode=crop_storage_mode,
                success=False,
                previous_latest=previous_latest,
                previous_latest_materialized=previous_latest_materialized,
                previous_latest_any=previous_latest_any,
            )
        
        # Create output arrays
        roi_layout = build_canonical_crop_roi_layout(
            total_rois=total_detections,
            preferred_chunk_len=int(external_roi_chunk_size),
            roi_storage=roi_storage_norm,
            use_sharding=bool(external_use_sharding),
            roi_shard_len=external_roi_shard_size,
        )
        roi_chunk_len = roi_layout.roi_chunk_len
        roi_shard_len = roi_layout.roi_shard_len if roi_layout.roi_shard_len is not None else roi_chunk_len
        roi_create_kwargs = build_crop_roi_create_kwargs(
            total_rois=total_detections,
            roi_sz=roi_sz,
            layout=roi_layout,
            overwrite=True,
        )
        if crop_storage_mode == "materialized":
            crop_group.create_array(
                'roi_images',
                **roi_create_kwargs,
            )
        
        crop_group.create_array(
            'roi_coordinates_full',
            shape=(total_detections, 2),
            chunks=(min(2048, total_detections), 2),
            dtype='i4',
            overwrite=True
        )
        
        # Get environment info for complete metadata
        env_info = get_environment_info(
            include_all_packages=False,
            disk_path=str(zarr_path),
            collect_ip=False
        )

        detect_run_name, background_run_name, refined_run_name = resolve_source_run_info(root, source_path)
        review_status, review_ref = _resolve_detect_review_status(
            root, refined_run_name, source_path
        )

        # Store comprehensive metadata following unified spec

        crop_group.attrs.update({
            # === Core Identifiers ===
            'created_at_utc': started_at,
            'started_at_utc': started_at,
            'stage': 'crop',
            'pipeline_type': 'fisheye_tracking',
            'status': 'running',
            'crop_storage_mode': crop_storage_mode,
            'crop_revision': 0,
            
            # === Video Source ===
            'video_source_type': 'external',
            'video_source_path': str(video_path),
            'source_video_path': str(video_path),
            'total_frames': num_frames,
            'width': video_width,
            'height': video_height,
            
            # === Detection Source ===
            'detection_source_type': source_type,
            'detection_source_path': source_path,
            'detection_method': get_detection_method(source_group),
            'total_detections': total_detections,
            
            # === Crop Configuration ===
            'roi_size': list(roi_sz),
            'parameter_source': 'config',  # External video uses config params
            
            # === Acceleration ===
            'acceleration': 'gpu' if actual_use_gpu else 'cpu',
            'device': 'cuda:0' if actual_use_gpu else 'cpu',
            
            # === Git Provenance ===
            'git_commit': env_info['git'].get('commit_hash', 'unknown'),
            'git_commit_hash': env_info['git'].get('commit_hash', 'unknown'),
            'git_short_hash': env_info['git'].get('short_hash', 'unknown'),
            'git_branch': env_info['git'].get('branch', 'unknown'),
            'git_is_dirty': env_info['git'].get('is_dirty', False),
            'git_remote_url': env_info['git'].get('remote_url', 'unknown'),
            
            # === Platform Info ===
            'hostname': env_info['platform']['hostname'],
            'system_os': env_info['platform']['system'],
            'system_machine': env_info['platform']['machine'],
            'python_version': env_info['platform']['python_version'],
            'cpu_cores': env_info['platform']['cpu_cores'],
            
            # === GPU Info ===
            'gpu_available': env_info['gpu']['available'],
            
            # === Environment ===
            'environment_type': env_info['environment']['environment_type'],
            'environment_name': env_info['environment']['environment_name'],
            
            # === Execution ===
            'command': ' '.join(sys.argv),
        })

        if detect_run_name:
            crop_group.attrs['source_detect_run'] = detect_run_name
        if background_run_name:
            crop_group.attrs['source_background_run'] = background_run_name
        if refined_run_name:
            crop_group.attrs['source_refined_run'] = refined_run_name
        if review_ref:
            crop_group.attrs['detect_review_status_ref'] = review_ref
        if review_status:
            crop_group.attrs['detect_review_status'] = review_status
        if selection_policy:
            crop_group.attrs['detection_selection_policy'] = selection_policy
        _set_crop_pixel_contract_attrs(
            crop_group,
            crop_storage_mode=crop_storage_mode,
            video_source_type="external",
            acceleration=str(crop_group.attrs.get("acceleration") or ""),
        )
        effective_run_provenance = run_provenance if run_provenance is not None else cli_provenance
        if effective_run_provenance is not None:
            crop_group.attrs[RUN_PROVENANCE_ATTR] = dict(effective_run_provenance)
            crop_group.attrs[CLI_RUN_PROVENANCE_ATTR] = dict(effective_run_provenance)
        crop_group.attrs['crop_signature'] = build_crop_signature(crop_group.attrs)
        effective_backend = 'kvikio_gds' if use_kvikio_writes else 'standard_zarr'
        crop_group.attrs['write_backend'] = effective_backend
        crop_group.attrs['write_backend_requested'] = backend_norm
        crop_group.attrs['write_backend_effective'] = effective_backend
        crop_group.attrs['kvikio_required'] = bool(external_require_kvikio)
        if fallback_reason:
            crop_group.attrs['write_backend_fallback_reason'] = fallback_reason
        else:
            crop_group.attrs.pop('write_backend_fallback_reason', None)
        crop_group.attrs.update(crop_roi_layout_attrs(roi_layout))
        if not roi_layout.roi_use_sharding and 'roi_shard_len' in crop_group.attrs:
            del crop_group.attrs['roi_shard_len']

        # Add detailed GPU info if using GPU
        if actual_use_gpu and env_info['gpu']['available'] and env_info['gpu'].get('devices'):
            primary_gpu = env_info['gpu']['devices'][0]
            crop_group.attrs.update({
                'gpu_name': primary_gpu.get('name', 'unknown'),
                'gpu_memory_total_mb': primary_gpu.get('memory_total_mb', 0),
                'gpu_compute_capability': primary_gpu.get('compute_capability', 'unknown'),
                'gpu_driver_version': env_info['gpu'].get('driver_version', 'unknown'),
                'cuda_version': env_info['gpu'].get('cuda_version', 'unknown'),
            })
            
        # Copy source metadata
        save_crop_metadata(
            crop_group=crop_group,
            source_group=source_group,
            source_path=source_path,
            source_type=source_type,
            detection_source=detection_source,
            total_detections=total_detections,
            num_frames=num_frames
        )
        
        # Geometry-only runs do not decode frames, so avoid touching the video reader.
        if crop_storage_mode != "geometry_only" and actual_use_gpu:
            console.print("[green]Initializing GPU video decoder...[/green]")
            try:
                decord.bridge.set_bridge('torch')
                video_reader = VideoReader(str(video_path), ctx=gpu(0))
                console.print(f"[green]✓[/green] GPU decoder ready")
            except Exception as gpu_exc:
                if _DECORD_AVAILABLE:
                    decord.bridge.set_bridge('native')
                raise _gpu_decode_unavailable(f"Decord GPU crop decoder failed: {gpu_exc}") from gpu_exc
        if crop_storage_mode != "geometry_only" and not actual_use_gpu:
            console.print("[cyan]Using CPU video decoder...[/cyan]")

        _set_crop_pixel_contract_attrs(
            crop_group,
            crop_storage_mode=crop_storage_mode,
            video_source_type="external",
            acceleration=str(crop_group.attrs.get("acceleration") or ""),
        )
        crop_group.attrs['crop_signature'] = build_crop_signature(crop_group.attrs)
        provenance_record = _build_crop_stage_provenance(
            created_at_utc=str(crop_group.attrs.get("created_at_utc")),
            command=" ".join(sys.argv),
            env_info=env_info,
            parameters={
                "roi_size": list(roi_sz),
                "crop_storage_mode": crop_storage_mode,
                "acceleration": crop_group.attrs.get("acceleration"),
                "write_backend_requested": crop_group.attrs.get("write_backend_requested"),
                "write_backend_effective": crop_group.attrs.get("write_backend_effective"),
                "roi_storage": crop_group.attrs.get("roi_storage"),
                "roi_chunk_len": crop_group.attrs.get("roi_chunk_len"),
                "roi_use_sharding": crop_group.attrs.get("roi_use_sharding"),
                "roi_shard_len": crop_group.attrs.get("roi_shard_len"),
                "roi_image_representation": crop_group.attrs.get("roi_image_representation"),
                "roi_pixel_contract": crop_group.attrs.get("roi_pixel_contract"),
                "roi_pixel_contract_name": crop_group.attrs.get("roi_pixel_contract_name"),
            },
            parameter_source=str(crop_group.attrs.get("parameter_source") or "config"),
            inputs={
                "source_detect_run": detect_run_name,
                "source_refined_run": refined_run_name,
                "source_background_run": background_run_name,
                "frame_source": crop_group.attrs.get("video_source_type", "external"),
                "source_video_path": crop_group.attrs.get("video_source_path"),
            },
            detection_source={
                "type": source_type,
                "path": source_path,
                "method": get_detection_method(source_group),
            },
        )
        write_stage_provenance(crop_group, provenance_record)

        if crop_storage_mode == "geometry_only":
            coords_full, _ = _compute_roi_coordinates(
                bbox_coords,
                roi_sz,
                video_shape,
                scale_factor=None,
            )
            crop_group['roi_coordinates_full'][:] = coords_full

            duration = time.perf_counter() - start_time
            frames_with_crops = int(np.sum(crop_group['frame_counts'][:] > 0))
            percent_cropped = (frames_with_crops / num_frames) * 100 if num_frames > 0 else 0

            crop_group.attrs['summary_statistics'] = {
                'total_frames': num_frames,
                'frames_with_crops': frames_with_crops,
                'total_rois_cropped': total_detections,
                'percent_frames_with_crops': round(percent_cropped, 2),
                'roi_size': list(roi_sz),
                'roi_pixels_materialized': False,
            }
            crop_group.attrs['duration_seconds'] = duration
            crop_group.attrs['avg_batch_ms'] = 0.0
            crop_group.attrs['timing_profile'] = {
                'decode_seconds': 0.0,
                'compute_seconds': 0.0,
                'zarr_write_seconds': 0.0,
                'other_seconds': float(duration),
                'decode_percent': 0.0,
                'compute_percent': 0.0,
                'zarr_write_percent': 0.0,
                'other_percent': 100.0,
                'complete': True,
            }

            console.print("[green]✓[/green] Geometry-only crop run created")
            console.print(f"[cyan]  Time: {duration:.1f}s[/cyan]")
            console.print(f"[cyan]  Total ROIs: {total_detections:,}[/cyan]")
            console.print(f"[cyan]  Crop storage mode: {crop_storage_mode}[/cyan]")

            success = True
            return {
                'total_crops': total_detections,
                'frames_with_crops': frames_with_crops,
                'percent_cropped': percent_cropped,
                'duration_seconds': duration,
                'run_name': run_name,
                'detection_source_type': source_type,
                'detection_source_path': source_path,
                'crop_storage_mode': crop_storage_mode,
            }
        
        total_processed = 0
        batch_times: List[float] = []
        if actual_use_gpu:
            chunk_size = max(1, int(external_gpu_chunk_frames))
            console.print(f"[dim]  GPU frame chunk size: {chunk_size}[/dim]")
            frame_to_det: Dict[int, List[int]] = {}
            for det_idx, frame_idx in enumerate(frame_indices):
                frame_to_det.setdefault(int(frame_idx), []).append(int(det_idx))

            frame_min = int(frame_indices.min())
            frame_max = int(frame_indices.max())
            chunk_starts = list(range(frame_min, frame_max + 1, chunk_size))
            total_chunks = len(chunk_starts)

            if verbose:
                console.print(f"Processing {total_chunks} contiguous chunks... (use_gpu=True)")
                iterator = enumerate(chunk_starts)
            else:
                from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn

                progress = Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                    TextColumn("•"),
                    TextColumn("[cyan]{task.fields[rate]:.1f} crops/s"),
                    TimeRemainingColumn(),
                    console=console
                )
                progress_task = progress.add_task("Cropping ROIs (GPU)", total=total_detections, rate=0.0)
                progress.__enter__()
                iterator = enumerate(chunk_starts)

            try:
                for chunk_idx, chunk_start in iterator:
                    chunk_end = min(chunk_start + chunk_size, frame_max + 1)
                    chunk_frames = list(range(chunk_start, chunk_end))
                    if not any(frame in frame_to_det for frame in chunk_frames):
                        continue

                    if verbose:
                        console.print(
                            f"[debug] Chunk {chunk_idx + 1}/{total_chunks}: frames {chunk_frames[0]}-{chunk_frames[-1]} ({len(chunk_frames)} frames)"
                        )

                    decode_start = time.perf_counter()
                    frames_gpu = video_reader.get_batch(chunk_frames)
                    decode_seconds += time.perf_counter() - decode_start
                    compute_start = time.perf_counter()
                    _, det_ids, crops_buf, coords_full_cpu, chunk_time = _process_chunk_gpu(
                        chunk_idx,
                        chunk_frames,
                        frames_gpu,
                        frame_to_det,
                        bbox_coords,
                        roi_sz,
                        (video_height, video_width),
                        return_device=use_kvikio_writes,
                    )
                    compute_seconds += time.perf_counter() - compute_start
                    del frames_gpu
                    batch_times.append(chunk_time)

                    processed = int(det_ids.size)
                    if processed:
                        write_start = time.perf_counter()
                        if use_kvikio_writes:
                            det_slice = _contiguous_detection_slice(det_ids)
                            if det_slice is not None:
                                cupy_crops = cp.from_dlpack(crops_buf.contiguous())
                                crop_group['roi_images'][det_slice] = cupy_crops
                                crop_group['roi_coordinates_full'][det_slice] = coords_full_cpu
                                cp.cuda.Stream.null.synchronize()
                                del cupy_crops
                            else:
                                # Fallback for sparse/non-contiguous indices: CPU path is safer.
                                crops_cpu_fallback = crops_buf.cpu().numpy()
                                crop_group['roi_images'][det_ids] = crops_cpu_fallback
                                crop_group['roi_coordinates_full'][det_ids] = coords_full_cpu
                                del crops_cpu_fallback
                        else:
                            crop_group['roi_images'][det_ids] = crops_buf
                            crop_group['roi_coordinates_full'][det_ids] = coords_full_cpu
                        write_seconds += time.perf_counter() - write_start
                        total_processed += processed

                    if verbose:
                        console.print(
                            f"[debug] <- Chunk {chunk_idx + 1}/{total_chunks}: {processed} detections in {chunk_time*1000:.1f} ms"
                        )
                    else:
                        elapsed_total = time.perf_counter() - start_time
                        rate = total_processed / elapsed_total if elapsed_total > 0 else 0
                        progress.update(progress_task, advance=processed, rate=rate)
            finally:
                if not verbose:
                    progress.__exit__(None, None, None)

        else:
            max_frames = 32
            batches = create_crop_batches(frame_indices, max_frames_per_batch=max_frames)
            console.print(f"Processing {len(batches)} batches... (use_gpu=False)")
            total_batches = len(batches)
            progress_every = max(1, total_batches // 20)

            for batch_idx, det_indices in enumerate(batches):
                batch_start = time.perf_counter()

                batch_frames = frame_indices[det_indices]
                batch_bboxes = bbox_coords[det_indices]

                crops, coords, batch_profile = crop_batch_cpu(
                    video_path, batch_frames, batch_bboxes,
                    roi_sz, video_shape,
                    total_frames=int(num_frames),
                    source_label=_source_label(source_path, source_group),
                )
                decode_seconds += float(batch_profile.get("decode_seconds", 0.0))
                compute_seconds += float(batch_profile.get("compute_seconds", 0.0))

                write_start = time.perf_counter()
                crop_group['roi_images'][det_indices] = crops
                crop_group['roi_coordinates_full'][det_indices] = coords
                write_seconds += time.perf_counter() - write_start

                batch_time = time.perf_counter() - batch_start
                batch_times.append(batch_time)
                total_processed += len(det_indices)

                if verbose:
                    console.print(
                        f"[debug] Batch {batch_idx + 1}/{total_batches} completed in {batch_time*1000:.1f} ms"
                    )
                else:
                    if (batch_idx + 1) % progress_every == 0 or batch_idx == total_batches - 1:
                        console.print(
                            f"[cyan]Batch {batch_idx + 1}/{total_batches}: {total_processed} detections processed[/cyan]"
                        )

        duration = time.perf_counter() - start_time

        # Add summary statistics
        console.print("[dim]Calculating summary statistics...[/dim]")
        if verbose:
            console.print("[debug] Reading frame_counts for summary computation")
        frames_with_crops = int(np.sum(crop_group['frame_counts'][:] > 0))
        percent_cropped = (frames_with_crops / num_frames) * 100 if num_frames > 0 else 0

        crop_group.attrs['summary_statistics'] = {
            'total_frames': num_frames,
            'frames_with_crops': frames_with_crops,
            'total_rois_cropped': total_detections,
            'percent_frames_with_crops': round(percent_cropped, 2),
            'roi_size': list(roi_sz)
        }
        crop_group.attrs['duration_seconds'] = duration
        crop_group.attrs['avg_batch_ms'] = float(np.mean(batch_times)) if batch_times else 0.0
        profile_total = duration if duration > 0 else 1.0
        profile_other = max(0.0, duration - (decode_seconds + compute_seconds + write_seconds))
        crop_group.attrs['timing_profile'] = {
            'decode_seconds': float(decode_seconds),
            'compute_seconds': float(compute_seconds),
            'zarr_write_seconds': float(write_seconds),
            'other_seconds': float(profile_other),
            'decode_percent': float((decode_seconds / profile_total) * 100.0),
            'compute_percent': float((compute_seconds / profile_total) * 100.0),
            'zarr_write_percent': float((write_seconds / profile_total) * 100.0),
            'other_percent': float((profile_other / profile_total) * 100.0),
            'complete': True,
        }

        if verbose:
            avg_batch = (np.mean(batch_times) * 1000) if batch_times else 0.0
            console.print(
                f"[debug] Completed {total_processed} crops in {duration:.2f}s "
                f"({total_processed/duration if duration > 0 else 0:.1f} crops/s). "
                f"Average batch time: {avg_batch:.1f} ms"
            )
            if actual_use_gpu and _TORCH_AVAILABLE and torch.cuda.is_available():
                try:
                    mem_alloc = torch.cuda.memory_allocated() / (1024 * 1024)
                    mem_reserved = torch.cuda.memory_reserved() / (1024 * 1024)
                    console.print(
                        f"[debug] GPU memory - allocated: {mem_alloc:.1f} MB, reserved: {mem_reserved:.1f} MB"
                    )
                except Exception:
                    pass

        console.print(f"[green]✓[/green] Cropping complete")
        console.print(f"[cyan]  Time: {duration:.1f}s[/cyan]")
        console.print(f"[cyan]  Rate: {total_detections/duration:.1f} crops/s[/cyan]")
        console.print(
            "[cyan]  Profile:[/cyan] "
            f"decode={decode_seconds:.1f}s ({(decode_seconds/profile_total)*100.0:.1f}%), "
            f"compute={compute_seconds:.1f}s ({(compute_seconds/profile_total)*100.0:.1f}%), "
            f"write={write_seconds:.1f}s ({(write_seconds/profile_total)*100.0:.1f}%), "
            f"other={profile_other:.1f}s ({(profile_other/profile_total)*100.0:.1f}%)"
        )
        console.print("[dim]Finalizing crop metadata...[/dim]")

        success = True
        return {
            'total_crops': total_detections,
            'frames_with_crops': frames_with_crops,
            'percent_cropped': percent_cropped,
            'duration_seconds': duration,
            'run_name': run_name,
            'detection_source_type': source_type,
            'detection_source_path': source_path,
            'crop_storage_mode': crop_storage_mode,
        }
    except KeyboardInterrupt:
        error_message = "Interrupted by user"
        console.print("[red]Cropping interrupted by user[/red]")
        raise
    except Exception as exc:
        error_message = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if actual_use_gpu:
            # Dereference the video_reader to allow the garbage collector
            # to release the underlying CUDA resources.
            video_reader = None
            
            # Clear PyTorch's cache of unused memory
            if _TORCH_AVAILABLE:
                if verbose:
                    console.print("[debug] Clearing CUDA cache...")
                torch.cuda.empty_cache()
                if verbose:
                    console.print("[debug] CUDA cache cleared")

        if use_kvikio_writes:
            try:
                root.store.close()
            except Exception:
                pass

        if crop_group is not None:
            elapsed = time.perf_counter() - start_time
            profile_total = elapsed if elapsed > 0 else 1.0
            profile_other = max(0.0, elapsed - (decode_seconds + compute_seconds + write_seconds))
            crop_group.attrs['timing_profile'] = {
                'decode_seconds': float(decode_seconds),
                'compute_seconds': float(compute_seconds),
                'zarr_write_seconds': float(write_seconds),
                'other_seconds': float(profile_other),
                'decode_percent': float((decode_seconds / profile_total) * 100.0),
                'compute_percent': float((compute_seconds / profile_total) * 100.0),
                'zarr_write_percent': float((write_seconds / profile_total) * 100.0),
                'other_percent': float((profile_other / profile_total) * 100.0),
                'complete': bool(success),
            }
            timestamp = datetime.now(timezone.utc).isoformat()
            crop_parent = root.get("crop_runs")
            if success:
                crop_group.attrs['status'] = 'completed'
                crop_group.attrs['completed_at_utc'] = timestamp
                crop_group.attrs.pop('error_message', None)
                crop_group.attrs.pop('failed_at_utc', None)
                if crop_parent is not None and run_name is not None:
                    mark_run_complete(
                        crop_group,
                        parent_group=crop_parent,
                        run_name=run_name,
                        run_provenance=effective_run_provenance,
                    )
                if verbose:
                    console.print("[debug] Crop run marked as completed")
            else:
                crop_group.attrs['status'] = 'failed'
                crop_group.attrs['failed_at_utc'] = timestamp
                if error_message:
                    crop_group.attrs['error_message'] = error_message
                mark_run_failed(crop_group, error=error_message)
                if verbose:
                    console.print("[debug] Crop run marked as failed")
            if crop_parent is not None and run_name is not None:
                _finalize_crop_parent_pointers(
                    crop_parent,
                    run_name=run_name,
                    crop_storage_mode=crop_storage_mode,
                    success=bool(success),
                    previous_latest=previous_latest,
                    previous_latest_materialized=previous_latest_materialized,
                    previous_latest_any=previous_latest_any,
                )

def get_detection_source_info(
    root: zarr.Group,
    source_type: str = 'detect',
    source_path_override: Optional[str] = None,
    console: Optional[Console] = None,
    selection_policy: Optional[str] = None,
) -> Tuple[str, zarr.Group, Optional[np.ndarray], str]:
    """
    Get information about the detection source to use for cropping.
    
    Args:
        root: Zarr root group
        source_type: 'detect', 'filtered', 'interpolated', 'manual',
            'refined', or 'auto' (hint)
        source_path_override: Explicit path like 'detect_runs/<run>' or the
            canonical refined path
            'refined_detect_runs/<run>/instances'
        console: Optional Rich console for output
        selection_policy: Optional policy label for auto source selection (e.g., training/full_recording)
        
    Returns:
        Tuple of (source_path, source_group, detection_source_array, resolved_source_type)
        - source_path: Path string like 'detect_runs/latest',
          or 'refined_detect_runs/latest/instances'
        - source_group: Zarr group containing the detection data and metadata
        - detection_source_array: Array indicating real (0) vs interpolated (1), or None
        - resolved_source_type: Normalized detection source label
    """
    def _maybe_resolve_curated_instances_override(
        normalized_path: str,
    ) -> Optional[Tuple[str, zarr.Group, Optional[np.ndarray], str]]:
        parts = normalized_path.split("/")
        if len(parts) != 3:
            return None
        parent_name, refined_label, tail = parts
        if tail != "instances":
            return None
        if parent_name not in {REFINED_DETECT_GROUP, LEGACY_REFINED_DETECT_GROUP}:
            return None
        refined_parent = root.get(parent_name)
        if refined_parent is None or refined_label not in refined_parent:
            return None
        refined_group = refined_parent[refined_label]
        if not has_curated_refined_detect_surface(refined_group):
            raise ValueError(
                f"Refined detect run '{refined_label}' does not have a canonical curated detect surface."
            )
        detection_source = build_curated_detection_source_array(refined_group, present_only=True)
        return normalized_path, refined_group, detection_source, "refined"

    def _validate_detection_group(path_label: str, group: zarr.Group) -> None:
        if has_curated_refined_detect_surface(group):
            return
        required = ("frame_indices", "bbox_norm_coords")
        missing = [name for name in required if name not in group]
        if missing:
            missing_text = ", ".join(missing)
            raise ValueError(
                f"Detection source '{path_label}' missing required arrays: {missing_text}"
            )

    if source_path_override:
        normalized_path = str(source_path_override).strip().strip('/')
        if not normalized_path:
            raise ValueError("Empty detection source path provided for cropping.")
        if normalized_path not in root:
            raise ValueError(f"Detection source '{normalized_path}' not found in zarr file.")
        curated_override = _maybe_resolve_curated_instances_override(normalized_path)
        if curated_override is not None:
            if console:
                console.print(f"[cyan]Using detections:[/cyan] {normalized_path}")
            return curated_override
        source_group = root[normalized_path]
        resolved_type = infer_detection_source_type(normalized_path, None)
        _validate_detection_group(normalized_path, source_group)
        is_refined = normalized_path.startswith(REFINED_DETECT_GROUP) or normalized_path.startswith(LEGACY_REFINED_DETECT_GROUP)
        detection_source = None
        if resolved_type == 'interpolated' and 'detection_source' in source_group:
            detection_source = _ensure_numpy_array(
                source_group['detection_source'][:],
                dtype='i1',
                name=f"{normalized_path}/detection_source",
            )
        elif resolved_type == 'filtered' and is_refined:
            total = int(source_group['frame_indices'].shape[0])
            detection_source = np.zeros(total, dtype='i1')
        elif resolved_type == 'manual' and is_refined:
            if 'detection_source' in source_group:
                detection_source = _ensure_numpy_array(
                    source_group['detection_source'][:],
                    dtype='i1',
                    name=f"{normalized_path}/detection_source",
                )
            else:
                total = int(source_group['frame_indices'].shape[0])
                detection_source = np.zeros(total, dtype='i1')
        elif resolved_type == 'refined' and is_refined:
            if has_curated_refined_detect_surface(source_group):
                detection_source = build_curated_detection_source_array(source_group, present_only=True)
            else:
                total = int(source_group['frame_indices'].shape[0])
                detection_source = np.zeros(total, dtype='i1')
        if console:
            console.print(f"[cyan]Using detections:[/cyan] {normalized_path}")
        return normalized_path, source_group, detection_source, resolved_type

    def _select_detect_run() -> Tuple[str, zarr.Group, Optional[np.ndarray], str]:
        if 'detect_runs' not in root:
            raise ValueError("No detect_runs found in zarr file")

        resolved = resolve_run(
            root['detect_runs'],
            RunResolution.AUTHORITATIVE,
            parent_path='detect_runs',
            run_label='Detect run',
        )
        if resolved.run_name is None or resolved.run_group is None:
            raise ValueError("No authoritative or latest-complete detect run found")

        source_path = f'detect_runs/{resolved.run_name}'
        source_group = resolved.run_group
        detection_source = None

        if console:
            console.print(
                f"[cyan]Using original detections:[/cyan] {resolved.run_name} "
                f"({resolved.resolution_source})"
            )
        return source_path, source_group, detection_source, 'detect'

    def _load_refined_root() -> Tuple[zarr.Group, str]:
        if REFINED_DETECT_GROUP in root:
            return root[REFINED_DETECT_GROUP], REFINED_DETECT_GROUP
        if LEGACY_REFINED_DETECT_GROUP in root:
            return root[LEGACY_REFINED_DETECT_GROUP], LEGACY_REFINED_DETECT_GROUP
        raise ValueError("No refined detection runs found. Run refinement pipeline first.")

    def _build_refined_source(
        refined_root: zarr.Group,
        refined_group: zarr.Group,
        refined_label: str,
        source_key: str,
    ) -> Tuple[str, zarr.Group, Optional[np.ndarray], str]:
        source_group = refined_group[source_key]
        source_path = f"{refined_root.path}/{refined_label}/{source_key}"
        _validate_detection_group(source_path, source_group)
        detection_source = None
        if source_key == 'filtered':
            total = int(source_group['frame_indices'].shape[0])
            detection_source = np.zeros(total, dtype='i1')
        elif source_key in ('interpolated', 'manual'):
            if 'detection_source' in source_group:
                detection_source = _ensure_numpy_array(
                    source_group['detection_source'][:],
                    dtype='i1',
                    name=f"{source_path}/detection_source",
                )
            else:
                total = int(source_group['frame_indices'].shape[0])
                detection_source = np.zeros(total, dtype='i1')
        return source_path, source_group, detection_source, source_key

    def _build_curated_refined_source(
        refined_root: zarr.Group,
        refined_group: zarr.Group,
        refined_label: str,
    ) -> Tuple[str, zarr.Group, Optional[np.ndarray], str]:
        if not has_curated_refined_detect_surface(refined_group):
            raise ValueError(
                f"Refined run '{refined_label}' is missing a canonical curated detect surface."
            )
        if has_sparse_curated_refined_detect_instances_arrays(refined_group):
            source_path = f"{refined_root.path}/{refined_label}/instances"
        else:
            source_path = f"{refined_root.path}/{refined_label}"
        _validate_detection_group(source_path, refined_group)
        detection_source = build_curated_detection_source_array(refined_group, present_only=True)
        return source_path, refined_group, detection_source, 'refined'

    if source_type == 'detect':
        return _select_detect_run()
        
    elif source_type in ['filtered', 'interpolated', 'manual', 'refined', 'auto']:
        if source_type == 'auto':
            try:
                refined_root, refined_parent_path = _load_refined_root()
            except ValueError:
                if console:
                    console.print("[yellow]No refined detections found; falling back to original detections.[/yellow]")
                return _select_detect_run()
        else:
            refined_root, refined_parent_path = _load_refined_root()

        resolved_refined = resolve_run(
            refined_root,
            RunResolution.AUTHORITATIVE,
            parent_path=refined_parent_path,
            run_label="Refined detect run",
        )
        latest_refined = resolved_refined.run_name
        if latest_refined is None or resolved_refined.run_group is None:
            raise ValueError("No authoritative or latest-complete refined detection run found")

        refined_group = resolved_refined.run_group

        if source_type in ('refined', 'auto') and has_curated_refined_detect_surface(refined_group):
            source_path, source_group, detection_source, resolved_type = _build_curated_refined_source(
                refined_root,
                refined_group,
                latest_refined,
            )
            if console:
                console.print(
                    f"[cyan]Using canonical refined detections:[/cyan] {latest_refined} "
                    f"({resolved_refined.resolution_source})"
                )
            return source_path, source_group, detection_source, resolved_type
        if source_type == 'refined':
            raise ValueError(
                f"Refined detect run '{latest_refined}' does not have a canonical curated detect surface."
            )

        manual_label = refined_group.attrs.get('manual_review_latest')
        if not manual_label and 'manual' in refined_group:
            manual_label = 'manual'

        resolved_source_type = source_type
        resolved_key = None

        if source_type == 'auto':
            review_status = refined_group.attrs.get('detect_review_status')
            hint = None
            if isinstance(review_status, dict):
                hint = review_status.get('target_group') or review_status.get('resolved_group')
            if hint:
                hint = str(hint)
                if hint in ('raw', 'detect'):
                    if console:
                        console.print(f"[cyan]Using original detections (review_status={hint}).[/cyan]")
                    return _select_detect_run()
                if hint in refined_group:
                    resolved_key = hint
                elif hint == 'manual' and manual_label:
                    resolved_key = manual_label
                elif hint in ('interpolated', 'filtered') and hint in refined_group:
                    resolved_key = hint

            if resolved_key is None:
                policy = (selection_policy or "training").strip().lower()
                preference = DEFAULT_DETECT_GROUP_PREFERENCE
                if policy != "training":
                    preference = DEFAULT_DETECT_GROUP_PREFERENCE
                resolution = resolve_refined_detect_group(refined_group, preference=preference)
                if resolution.label == "raw":
                    if console:
                        console.print("[cyan]Using original detections (auto fallback).[/cyan]")
                    return _select_detect_run()
                resolved_key = resolution.group
                resolved_source_type = resolution.label or resolution.group or source_type
        else:
            resolved_key = source_type

        if resolved_key == 'manual':
            if manual_label:
                resolved_key = manual_label
            else:
                raise ValueError(f"Stage '{source_type}' not found in refined detection run {latest_refined}")

        if resolved_key not in refined_group:
            if source_type == 'auto':
                if console:
                    console.print(
                        f"[yellow]Auto-selected refined stage '{resolved_key}' not found in "
                        f"{latest_refined}; falling back to original detections.[/yellow]"
                    )
                return _select_detect_run()
            raise ValueError(f"Stage '{resolved_key}' not found in refined detection run {latest_refined}")

        try:
            source_path, source_group, detection_source, resolved_type = _build_refined_source(
                refined_root, refined_group, latest_refined, resolved_key
            )
        except ValueError as exc:
            if source_type == 'auto':
                if console:
                    console.print(
                        f"[yellow]Auto-selected refined stage '{resolved_key}' is unusable "
                        f"({exc}); falling back to original detections.[/yellow]"
                    )
                return _select_detect_run()
            raise

        if manual_label and resolved_key == manual_label:
            resolved_source_type = 'manual'
        elif resolved_source_type in ('manual', 'filtered', 'interpolated'):
            resolved_source_type = resolved_source_type
        else:
            resolved_source_type = resolved_type

        if console:
            console.print(
                f"[cyan]Using refined detections ({resolved_source_type}):[/cyan] {latest_refined} "
                f"({resolved_refined.resolution_source})"
            )
            if detection_source is not None:
                detection_source = _ensure_numpy_array(
                    detection_source,
                    dtype='i1',
                    name=f"{source_path}/detection_source",
                )
                n_real = np.sum(detection_source == 0)
                n_interp = np.sum(detection_source == 1)
                console.print(f"  Real detections: {n_real}, Interpolated: {n_interp}")

        return source_path, source_group, detection_source, resolved_source_type
    
    else:
        raise ValueError(
            f"Invalid source_type: {source_type}. Must be 'detect', 'filtered', "
            "'interpolated', 'manual', 'refined', or 'auto'"
        )


def get_crop_parameters(
    root: zarr.Group,
    config: Dict[str, Any],
    console: Optional[Console] = None
) -> Tuple[Dict[str, Any], str]:
    """
    Get crop parameters with zarr-first resolution.
    
    Priority order:
    1. Zarr analysis_metadata (if crop tuning exists)
    2. Config file defaults
    """
    # Start with config defaults
    crop_params = config.get('crop', {}).copy()
    crop_params.setdefault('roi_sz', [512, 512])
    crop_params.setdefault('acceleration', 'auto')
    crop_params.setdefault('gpu_min_detections', 200)
    crop_params.setdefault('crop_storage_mode', 'materialized')
    
    param_source = 'config_default'
    
    # Check for tuned parameters in zarr (future: crop tuning)
    if 'analysis_metadata' in root:
        analysis_meta = root['analysis_metadata']
        
        # Future: if we add crop parameter tuning
        if 'crop_tuning' in analysis_meta.attrs:
            tuning_data = analysis_meta.attrs['crop_tuning']
            tuned_params = tuning_data.get('tuned_parameters', {})
            if tuned_params:
                crop_params.update(tuned_params)
                param_source = 'zarr_tuned'
                if console:
                    console.print(f"[green]✓ Using tuned crop parameters from zarr[/green]")
        
        # Check for mask tuning (this is the main one)
        if 'dish_mask' in analysis_meta.attrs:
            mask_attr = analysis_meta.attrs['dish_mask']
            mask_data = dict(mask_attr)
            shape = mask_data.get('shape')
            if not shape:
                if 'detected_circle' in mask_data:
                    shape = 'circle'
                elif 'rectangle' in mask_data:
                    shape = 'rectangle'
            if 'dish_mask' not in crop_params:
                crop_params['dish_mask'] = {}
            if shape == 'rectangle' and 'rectangle' in mask_data:
                roi = mask_data['rectangle'].get('roi')
                if roi:
                    crop_params['dish_mask'].update({
                        'shape': 'rectangle',
                        'roi': [int(v) for v in roi],
                    })
                    if console:
                        console.print(f"[green]✓ Using rectangular dish mask from zarr[/green]")
            elif shape == 'circle' and 'detected_circle' in mask_data:
                circle = mask_data['detected_circle']
                crop_params['dish_mask'].update({
                    'shape': 'circle',
                    'center': circle['center'],
                    'radius': circle['radius']
                })
                if console:
                    console.print(f"[green]✓ Using circular dish mask from zarr[/green]")
    
    return crop_params, param_source


# -------- Worker task: compute + WRITE directly into Zarr -------- #

def save_crop_metadata(
    crop_group: zarr.Group,
    source_group: zarr.Group,
    source_path: str,
    source_type: str,
    detection_source: Optional[np.ndarray],
    total_detections: int,
    num_frames: int
) -> None:
    """
    Copy coordinates and metadata from source to make crops self-contained.
    
    Args:
        crop_group: Target crop group
        source_group: Source detection group
        source_path: Path to source (for provenance)
        source_type: 'detect', legacy sparse refined subgroup label, or
            'refined'
        detection_source: Array indicating real (0) vs interpolated (1), or None
        total_detections: Total number of detections
        num_frames: Total number of frames in video  # ADD THIS
    """
    # Copy bbox coordinates and row identity from the exact source rowset used.
    row_payload = _extract_detection_row_payload(source_group)
    frame_indices = row_payload["frame_indices"]
    bbox_coords = row_payload["bbox_norm_coords"]
    if frame_indices.shape[0] != total_detections:
        raise ValueError(
            f"frame_indices length {frame_indices.shape[0]} does not match total detections {total_detections}"
        )
    _validate_frame_indices_in_bounds(
        frame_indices,
        total_frames=int(num_frames),
        source_label=_source_label(source_path, source_group),
    )
    if bbox_coords.shape[0] != total_detections:
        raise ValueError(
            f"bbox_norm_coords length {bbox_coords.shape[0]} does not match total detections {total_detections}"
        )
    crop_group.create_array(
        'bbox_norm_coords',
        chunks=(min(1000, len(bbox_coords)), 4),
        data=bbox_coords,
        overwrite=True
    )
    
    # Copy frame_indices directly
    create_geometry_preload_array(
        crop_group,
        'frame_indices',
        data=frame_indices,
        overwrite=True
    )
    
    # Compute and store frame_counts for visualization
    frame_counts = np.bincount(frame_indices, minlength=num_frames)
    crop_group.create_array(
        'frame_counts',
        chunks=(min(10000, num_frames),),
        data=frame_counts,
        overwrite=True
    )

    # Create detection_indices mapping (identity w.r.t source detections)
    detection_indices = np.arange(total_detections, dtype='i4')
    create_geometry_preload_array(
        crop_group,
        'detection_indices',
        data=detection_indices,
        overwrite=True
    )
    _write_optional_detection_row_lineage(
        crop_group,
        row_payload,
        total_detections=total_detections,
    )
    
    # Copy detection_source if provided (refined metadata)
    if detection_source is not None:
        detection_source = _ensure_numpy_array(detection_source, dtype='i1', name='detection_source')
        if detection_source.shape[0] != total_detections:
            raise ValueError(
                f"detection_source length {detection_source.shape[0]} does not match total detections {total_detections}"
            )
        crop_group.create_array(
            'detection_source',
            chunks=(min(1000, len(detection_source)),),
            data=detection_source,
            overwrite=True
        )
        
        n_real = int(np.sum(detection_source == 0))
        n_interp = int(np.sum(detection_source == 1))
        
        crop_group.attrs['includes_interpolated'] = n_interp > 0
        crop_group.attrs['n_real_detections'] = n_real
        crop_group.attrs['n_interpolated_detections'] = n_interp
    else:
        crop_group.attrs['includes_interpolated'] = False
        crop_group.attrs['n_real_detections'] = total_detections
        crop_group.attrs['n_interpolated_detections'] = 0
    
    # Store provenance
    crop_group.attrs['source_coords_path'] = source_path
    crop_group.attrs['detection_source_type'] = source_type

@delayed
def crop_and_store_chunk_delayed(
    zarr_path: str,
    chunk_slice: slice,
    out_slice: Tuple[int, int],
    roi_sz: Tuple[int, int],
    scale_factor: float,
    source_path: str
) -> Dict[str, int]:
    """
    Crops ROIs for a chunk and writes them directly into the precreated Zarr arrays.

    Args:
        zarr_path: path to zarr archive
        chunk_slice: frames [start:stop] to process
        out_slice: (start_det, end_det) in the flattened detection space for this chunk
        roi_sz: (H, W) of the crop
        scale_factor: ds/full scale for coordinates_ds
        source_path: Path to detection source (e.g., 'detect_runs/latest',
            the canonical refined path
            'refined_detect_runs/latest/instances')

    Returns:
        Tiny dict with counts/indices for bookkeeping.
    """
    root = zarr.open(zarr_path, mode='a')

    # Find the target crop group via root attrs (set by driver before dispatch)
    crop_group_path = root.attrs.get('current_crop_group_path')
    if crop_group_path is None:
        raise RuntimeError("Root attrs missing 'current_crop_group_path' for worker writes.")
    crop_group = root[crop_group_path]

    # Load full-resolution images
    images_full_chunk = root['raw_video/images_full'][chunk_slice]
    full_img_shape = images_full_chunk.shape[1:]  # (H, W)

    # Load detection data from specified source
    source_group = root[source_path]
    
    # Load frame indices and bbox coordinates
    # Load only the slice of frame indices/bboxes we need
    frame_indices, bbox_coords = _extract_detection_rows(source_group)
    source_label = _source_label(source_path, source_group)
    _validate_frame_indices_sorted(frame_indices, source_label=source_label)
    _validate_frame_indices_in_bounds(
        frame_indices,
        total_frames=int(root['raw_video/images_full'].shape[0]),
        source_label=source_label,
    )
    chunk_frames = np.arange(chunk_slice.start, chunk_slice.stop)
    
    # Determine which detections fall into this chunk
    mask = np.isin(frame_indices, chunk_frames)
    detection_indices = np.where(mask)[0]
    
    bbox_coords_chunk = bbox_coords[detection_indices]
    frames_for_detections = frame_indices[detection_indices]
    
    # Build n_per_frame for this chunk
    n_per_frame = np.zeros(len(chunk_frames), dtype=int)
    for i, frame in enumerate(chunk_frames):
        n_per_frame[i] = np.sum(frames_for_detections == frame)

    start_det_out, end_det_out = out_slice
    count = end_det_out - start_det_out
    if count == 0:
        return {"frames": int(np.sum(n_per_frame)), "start": start_det_out, "end": end_det_out}

    # Allocate local buffers (live only within the worker)
    rois_buf = np.zeros((count, *roi_sz), dtype='uint8')
    coords_full_buf = np.zeros((count, 2), dtype='i4')
    coords_ds_buf = np.zeros((count, 2), dtype='i4')

    cursor_in = 0
    cursor_out = 0
    H, W = full_img_shape

    for i in range(len(images_full_chunk)):
        nd = int(n_per_frame[i])
        if nd == 0:
            continue

        img = images_full_chunk[i]
        
        for _ in range(nd):
            center_norm = bbox_coords_chunk[cursor_in][:2]  # (cx_norm, cy_norm)
            # Note: bbox coords are normalized w.r.t (W, H), so multiply in (W, H) order
            full_centroid_px = np.round(center_norm * np.array([W, H])).astype(int)

            x1 = int(full_centroid_px[0] - roi_sz[1] // 2)
            y1 = int(full_centroid_px[1] - roi_sz[0] // 2)

            # Extract ROI with padding if needed
            y2 = y1 + roi_sz[0]
            x2 = x1 + roi_sz[1]

            # Compute valid region within img
            vy1 = max(0, y1); vy2 = min(H, y2)
            vx1 = max(0, x1); vx2 = min(W, x2)

            if (vy2 - vy1) == roi_sz[0] and (vx2 - vx1) == roi_sz[1] and 0 <= y1 < H and 0 <= x1 < W:
                roi = img[vy1:vy2, vx1:vx2]
            else:
                # Pad when ROI extends outside edges
                roi = np.zeros(roi_sz, dtype='uint8')
                if vy2 > vy1 and vx2 > vx1:
                    py1 = max(0, -y1)
                    px1 = max(0, -x1)
                    py2 = py1 + (vy2 - vy1)
                    px2 = px1 + (vx2 - vx1)
                    roi[py1:py2, px1:px2] = img[vy1:vy2, vx1:vx2]

            rois_buf[cursor_out] = roi
            coords_full_buf[cursor_out] = (x1, y1)

            # Downsampled coords (integer)
            dx = int(x1 * scale_factor)
            dy = int(y1 * scale_factor)
            coords_ds_buf[cursor_out] = (dx, dy)

            cursor_in += 1
            cursor_out += 1

    # Single write per array per worker (targeting non-overlapping slices)
    crop_group['roi_images'][start_det_out:end_det_out] = rois_buf
    crop_group['roi_coordinates_full'][start_det_out:end_det_out] = coords_full_buf
    crop_group['roi_coordinates_ds'][start_det_out:end_det_out] = coords_ds_buf

    return {"frames": int(np.sum(n_per_frame)), "start": start_det_out, "end": end_det_out}


def crop_detections(
    zarr_path: str,
    config: Dict[str, Any],
    source_type: str = 'detect',
    source_path: Optional[str] = None,
    selection_policy: Optional[str] = None,
    scheduler: str = None,
    num_workers: Optional[int] = None,
    console: Optional[Console] = None,
    acceleration: Optional[str] = None,
    external_write_backend: Optional[str] = None,
    external_roi_storage: Optional[str] = None,
    external_use_sharding: Optional[bool] = None,
    external_roi_chunk_size: Optional[int] = None,
    external_roi_shard_size: Optional[int] = None,
    external_gpu_chunk_frames: Optional[int] = None,
    external_require_kvikio: Optional[bool] = None,
    crop_storage_mode: Optional[str] = None,
    use_gpu_allowed: bool = True,
    force_cpu: bool = False,
    verbose: bool = False,
    cli_provenance: Optional[Mapping[str, Any]] = None,
    run_provenance: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Main function to crop ROIs from full-resolution frames based on detections.
    
    Supports both:
    - Zarr video (traditional workflow)
    - External video files (YOLO workflow with GPU/CPU acceleration)
    
    Args:
        zarr_path: Path to zarr file
        config: Configuration dictionary
        source_type: Detection source - 'detect', 'filtered', 'interpolated', 'manual', 'refined', or 'auto'
        source_path: Explicit detection source path override (optional)
        selection_policy: Optional policy label for auto source selection
        scheduler: Dask scheduler ('processes', 'threads', or 'distributed')
        num_workers: Number of workers (None = auto)
        console: Optional Rich console for output
        acceleration: 'auto', 'gpu', or 'cpu' to override external video mode selection
        external_write_backend: External video write backend ('standard' or 'kvikio')
        external_roi_storage: ROI array storage mode ('compressed' or 'uncompressed')
        external_use_sharding: Whether to shard ROI writes
        external_roi_chunk_size: Detection-axis chunk length for ROI arrays
        external_roi_shard_size: Detection-axis shard length for ROI arrays
        external_gpu_chunk_frames: Number of decoded frames per GPU crop chunk
        external_require_kvikio: Require kvikIO backend for external writes
        crop_storage_mode: Crop persistence mode ('materialized' or 'geometry_only')
        use_gpu_allowed: Whether GPU usage is permitted globally
        force_cpu: Force CPU processing regardless of availability
        verbose: Enable additional logging and disable progress bars
    
    Returns:
        Dictionary with cropping statistics
    """
    if console is None:
        console = Console()

    console.rule("[bold]Stage: Cropping ROIs from Detections[/bold]")
    start_time = time.perf_counter()

    root = zarr.open_group(zarr_path, mode='a', use_consolidated=False)

    crop_params_cfg = config.get('crop', {}) or {}
    if selection_policy is None:
        selection_policy = crop_params_cfg.get('selection_policy')

    # Get detection source information
    source_path, source_group, detection_source, resolved_source_type = get_detection_source_info(
        root=root,
        source_type=source_type,
        source_path_override=source_path,
        console=console,
        selection_policy=selection_policy,
    )
    source_type = resolved_source_type

    # Get crop parameters
    crop_params, param_source = get_crop_parameters(root, config, console)
    crop_storage_mode_resolved = _normalize_crop_storage_mode(
        crop_storage_mode if crop_storage_mode is not None else crop_params.get('crop_storage_mode', 'materialized')
    )
    _enforce_training_materialized_crop_contract(
        root,
        zarr_path=zarr_path,
        crop_storage_mode=crop_storage_mode_resolved,
    )
    crop_params['crop_storage_mode'] = crop_storage_mode_resolved
    roi_sz = tuple(crop_params.get('roi_sz', [512, 512]))
    detect_run_name, background_run_name, refined_run_name = resolve_source_run_info(root, source_path)
    effective_run_provenance = run_provenance if run_provenance is not None else cli_provenance
    if effective_run_provenance is None:
        effective_run_provenance = build_run_provenance(
            command="fisheye.tracking.crop.crop_detections",
            params={
                "zarr_path": zarr_path,
                "source_type": source_type,
                "source_path": source_path,
                "selection_policy": selection_policy,
                "scheduler": scheduler,
                "num_workers": num_workers,
                "acceleration": acceleration,
                "crop_storage_mode": crop_storage_mode_resolved,
                "roi_size": list(roi_sz),
                "external_write_backend": external_write_backend,
                "external_roi_storage": external_roi_storage,
                "external_use_sharding": external_use_sharding,
                "external_roi_chunk_size": external_roi_chunk_size,
                "external_roi_shard_size": external_roi_shard_size,
                "external_gpu_chunk_frames": external_gpu_chunk_frames,
                "external_require_kvikio": external_require_kvikio,
                "use_gpu_allowed": use_gpu_allowed,
                "force_cpu": force_cpu,
            },
            input_run_ids={
                "detect": detect_run_name,
                "background": background_run_name,
                "refined_detect": refined_run_name,
            },
            cwd=Path.cwd(),
        )
    
    # Determine video source
    video_source_type, video_path = get_video_source(root, console)
    
    # Route to appropriate implementation
    if video_source_type == 'external':
        total_detections = int(_extract_detection_rows(source_group)[0].shape[0])
        accel_choice = (acceleration or crop_params.get('acceleration', 'auto') or 'auto').lower()
        gpu_threshold = int(crop_params.get('gpu_min_detections', 200))
        valid_accels = {'auto', 'gpu', 'cpu'}
        if accel_choice not in valid_accels:
            console.print(f"[yellow]Unknown crop acceleration '{accel_choice}', defaulting to auto[/yellow]")
            accel_choice = 'auto'

        use_gpu = False
        decision_note = ""
        gpu_available = False
        gpu_reason = "GPU disabled"

        if force_cpu:
            decision_note = "forced CPU via --force-cpu"
        elif not use_gpu_allowed and accel_choice != 'gpu':
            decision_note = "GPU disabled via --no-gpu flag"
        elif accel_choice == 'cpu':
            decision_note = "crop acceleration configured as CPU"
        else:
            gpu_available, gpu_reason = check_gpu_crop_available()
            if not gpu_available:
                decision_note = f"GPU unavailable ({gpu_reason})"
            elif not use_gpu_allowed:
                decision_note = "GPU globally disabled"
            elif accel_choice == 'gpu':
                use_gpu = True
                decision_note = "crop acceleration configured as GPU"
            else:  # auto
                use_gpu = total_detections >= gpu_threshold
                if use_gpu:
                    decision_note = f"auto mode threshold met ({total_detections:,} >= {gpu_threshold})"
                else:
                    decision_note = f"auto mode threshold not met ({total_detections:,} < {gpu_threshold})"

        if crop_storage_mode_resolved == "geometry_only":
            console.print("[green]✓ Using geometry-only crop writing[/green]")
            console.print("[dim]  Skipping frame decode and ROI pixel extraction[/dim]")
            if use_gpu:
                console.print(f"[dim]  GPU-capable path available ({decision_note})[/dim]")
                console.print(f"[dim]  Device: {torch.cuda.get_device_name(0)}[/dim]")
            else:
                console.print(f"[dim]  {decision_note}[/dim]")
        elif use_gpu:
            console.print(f"[green]✓ Using GPU-accelerated cropping[/green]")
            console.print(f"[dim]  {decision_note}[/dim]")
            console.print(f"[dim]  Device: {torch.cuda.get_device_name(0)}[/dim]")
        else:
            console.print(f"[cyan]Using CPU ({decision_note})[/cyan]")
            if accel_choice == 'gpu' and not force_cpu and use_gpu_allowed and not gpu_available:
                console.print(f"[yellow]Warning: GPU requested but unavailable ({gpu_reason})[/yellow]")

        cfg_backend = str(crop_params_cfg.get('external_write_backend', 'standard')).strip().lower()
        cfg_storage = normalize_crop_roi_storage(
            crop_params_cfg.get('external_roi_storage', 'compressed'),
            default='compressed',
        )
        cfg_use_sharding = bool(crop_params_cfg.get('external_use_sharding', False))
        cfg_chunk = int(
            crop_params_cfg.get(
                'external_roi_chunk_size',
                config.get('import', {}).get('chunk_size', DEFAULT_CANONICAL_CROP_ROI_CHUNK_LEN),
            )
        )
        cfg_shard = crop_params_cfg.get('external_roi_shard_size')
        cfg_gpu_chunk_frames = int(crop_params_cfg.get('external_gpu_chunk_frames', 24))
        cfg_require_kvikio = bool(crop_params_cfg.get('external_require_kvikio', False))
        cfg_shard = int(cfg_shard) if cfg_shard is not None else None

        backend = (external_write_backend or cfg_backend).strip().lower()
        storage = normalize_crop_roi_storage(external_roi_storage or cfg_storage, default="compressed")
        use_sharding_value = cfg_use_sharding if external_use_sharding is None else bool(external_use_sharding)
        chunk_len = cfg_chunk if external_roi_chunk_size is None else int(external_roi_chunk_size)
        shard_len = cfg_shard if external_roi_shard_size is None else int(external_roi_shard_size)
        gpu_chunk_frames = cfg_gpu_chunk_frames if external_gpu_chunk_frames is None else int(external_gpu_chunk_frames)
        require_kvikio = cfg_require_kvikio if external_require_kvikio is None else bool(external_require_kvikio)
        if backend not in {"standard", "kvikio"}:
            backend = "standard"
        chunk_len = max(1, chunk_len)
        gpu_chunk_frames = max(1, gpu_chunk_frames)
        if shard_len is not None:
            shard_len = max(chunk_len, shard_len)

        if crop_storage_mode_resolved == "geometry_only":
            console.print(
                "[dim]  Metadata-only write path; ROI image backend settings are inactive "
                f"(configured backend={backend}, storage={storage})[/dim]"
            )
        else:
            console.print(
                f"[dim]  Write backend={backend}, roi_storage={storage}, "
                f"sharding={use_sharding_value}, chunk={chunk_len}, shard={shard_len or '-'}, "
                f"gpu_frames={gpu_chunk_frames}, require_kvikio={require_kvikio}[/dim]"
            )

        # Use external video cropping
        external_result = crop_from_external_video(
            zarr_path=zarr_path,
            video_path=video_path,
            source_path=source_path,
            source_group=source_group,
            detection_source=detection_source,
            source_type=source_type,
            roi_sz=roi_sz,
            use_gpu=use_gpu,
            console=console,
            selection_policy=selection_policy,
            external_write_backend=backend,
            external_roi_storage=storage,
            external_use_sharding=use_sharding_value,
            external_roi_chunk_size=chunk_len,
            external_roi_shard_size=shard_len,
            external_gpu_chunk_frames=gpu_chunk_frames,
            external_require_kvikio=require_kvikio,
            crop_storage_mode=crop_storage_mode_resolved,
            verbose=verbose,
            cli_provenance=effective_run_provenance,
            run_provenance=effective_run_provenance,
        )
        external_run_name = external_result.get('run_name') if isinstance(external_result, dict) else None
        external_run = (
            root['crop_runs'][external_run_name]
            if external_run_name and 'crop_runs' in root and external_run_name in root['crop_runs']
            else None
        )
        external_run_status = (
            str(external_run.attrs.get('status')).strip().lower()
            if external_run is not None and external_run.attrs.get('status') is not None
            else None
        )
        external_review = (
            external_run.attrs.get('crop_review_status')
            if external_run is not None
            else None
        )
        external_step_status = 'ok'
        if isinstance(external_result, dict) and int(external_result.get('total_crops', 0) or 0) <= 0:
            external_step_status = 'missing'
        if external_run_status in {'failed', 'error'}:
            external_step_status = 'error'
        _emit_crop_step_status(
            root=root,
            zarr_path=zarr_path,
            status=external_step_status,
            run_name=external_run_name,
            method=source_type,
            coverage_pct=(external_result.get('percent_cropped') if isinstance(external_result, dict) else None),
            review_status=external_review if isinstance(external_review, dict) else None,
            details={
                'reason': (
                    'present'
                    if external_step_status == 'ok'
                    else ('run_failed' if external_step_status == 'error' else 'no_detections')
                ),
                'run_state': external_run_status,
                'detection_source_type': source_type,
                'detection_source_path': source_path,
                'video_source_type': 'external',
                'crop_storage_mode': crop_storage_mode_resolved,
            },
            console=console,
        )
        return external_result
    
    # Otherwise, use zarr-based cropping
    
    # Use config values if not explicitly provided
    if scheduler is None:
        scheduler = crop_params.get('scheduler', 'processes')
    if num_workers is None:
        num_workers = crop_params.get('num_workers', None)
    
    # Determine if we'll use distributed BEFORE building metadata
    use_distributed = (scheduler == "distributed") and HAVE_DISTRIBUTED
    
    roi_sz = tuple(crop_params.get('roi_sz', [512, 512]))
    chunk_size = config.get('import', {}).get('chunk_size', 32)

    console.print(f"ROI size: {roi_sz[0]}×{roi_sz[1]} pixels")
    console.print(f"Chunk size: {chunk_size} frames")
    console.print(f"Scheduler: {scheduler}, Workers: {num_workers or 'default'}")

    # Create run group
    from ..shared.zarr.schema import get_run_group
    crop_parent_before = root.get("crop_runs")
    previous_latest = crop_parent_before.attrs.get("latest") if crop_parent_before is not None else None
    previous_latest_materialized = (
        crop_parent_before.attrs.get("latest_materialized") if crop_parent_before is not None else None
    )
    previous_latest_any = crop_parent_before.attrs.get("latest_any") if crop_parent_before is not None else None
    crop_group, run_group_name = get_run_group(
        root,
        'crop',
        console,
        completion_epoch=COMPLETION_EPOCH_REQUIRE_PROVENANCE,
    )
    success = False
    error_message: Optional[str] = None
    crop_parent = root.get("crop_runs")
    if crop_parent is not None:
        mark_run_started(crop_group, run_name=run_group_name, stage="crop")
        note_pending_latest(crop_parent, run_group_name)
        _finalize_crop_parent_pointers(
            crop_parent,
            run_name=run_group_name,
            crop_storage_mode=crop_storage_mode_resolved,
            success=False,
            previous_latest=previous_latest,
            previous_latest_materialized=previous_latest_materialized,
            previous_latest_any=previous_latest_any,
        )

    num_images = get_total_frames(root, source_group)
    if num_images is None:
        # Fallback to video shape if helper can't determine
        if 'raw_video/images_ds' in root:
            num_images = root['raw_video/images_ds'].shape[0]
        else:
            raise ValueError("Cannot determine total frames")

    # Get detection info using frame_indices (BEFORE metadata collection)
    frame_indices, bbox_coords = _extract_detection_rows(source_group)
    source_label = _source_label(source_path, source_group)
    if crop_storage_mode_resolved == "materialized":
        _validate_frame_indices_sorted(frame_indices, source_label=source_label)
    _validate_frame_indices_in_bounds(
        frame_indices,
        total_frames=int(num_images),
        source_label=source_label,
    )
    total_detections = len(frame_indices)
    
    if total_detections == 0:
        console.print("[yellow]Warning: No detections found. Nothing to crop.[/yellow]")
        _emit_crop_step_status(
            root=root,
            zarr_path=zarr_path,
            status='missing',
            run_name=None,
            method=source_type,
            coverage_pct=0.0,
            review_status=None,
            details={
                'reason': 'no_detections',
                'detection_source_type': source_type,
                'detection_source_path': source_path,
                'video_source_type': 'zarr',
            },
            console=console,
        )
        return {'total_crops': 0}
    
    console.print(f"Total detections to crop: {total_detections:,}")
    console.print(f"Total frames: {num_images:,}")

    # Get environment info
    env_info = get_environment_info(
        include_all_packages=False,
        disk_path=str(zarr_path),
        collect_ip=False
    )

    # Determine if GPU will be used
    use_distributed = (scheduler == "distributed") and HAVE_DISTRIBUTED

    review_status, review_ref = _resolve_detect_review_status(
        root, refined_run_name, source_path
    )

    # Build comprehensive metadata following unified spec
    crop_group.attrs.update({
        # === Core Identifiers ===
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'started_at_utc': datetime.now(timezone.utc).isoformat(),
        'stage': 'crop',
        'pipeline_type': 'fisheye_tracking',
        'status': 'running',
        'crop_storage_mode': crop_storage_mode_resolved,
        'crop_revision': 0,
        
        # === Video Source ===
        'video_source_type': 'zarr',
        'video_source_path': None,
        'source_video_path': root.attrs.get('source_video_path', 'unknown'),
        'total_frames': num_images,
        'width': root.attrs.get('width', 0),
        'height': root.attrs.get('height', 0),
        
        # === Detection Source ===
        'detection_source_type': source_type,
        'detection_source_path': source_path,
        'detection_method': get_detection_method(source_group),
        'total_detections': total_detections,
        
        # === Crop Configuration ===
        'roi_size': list(roi_sz),
        'parameter_source': param_source,
        'parameters': crop_params,
        
        # === Dask Configuration ===
        'scheduler': scheduler,
        'num_workers': num_workers or os.cpu_count(),
        'use_distributed': use_distributed,
        
        # === Git Provenance ===
        'git_commit': env_info['git'].get('commit_hash', 'unknown'),
        'git_commit_hash': env_info['git'].get('commit_hash', 'unknown'),
        'git_short_hash': env_info['git'].get('short_hash', 'unknown'),
        'git_branch': env_info['git'].get('branch', 'unknown'),
        'git_is_dirty': env_info['git'].get('is_dirty', False),
        'git_remote_url': env_info['git'].get('remote_url', 'unknown'),
        
        # === Platform Info ===
        'hostname': env_info['platform']['hostname'],
        'system_os': env_info['platform']['system'],
        'system_machine': env_info['platform']['machine'],
        'python_version': env_info['platform']['python_version'],
        'cpu_cores': env_info['platform']['cpu_cores'],
        
        # === GPU Info (CPU workflow but still track GPU availability) ===
        'gpu_available': env_info['gpu']['available'],
        
        # === Environment ===
        'environment_type': env_info['environment']['environment_type'],
        'environment_name': env_info['environment']['environment_name'],
        
        # === Execution ===
        'command': ' '.join(sys.argv),
    })

    if detect_run_name:
        crop_group.attrs['source_detect_run'] = detect_run_name
    if background_run_name:
        crop_group.attrs['source_background_run'] = background_run_name
    if refined_run_name:
        crop_group.attrs['source_refined_run'] = refined_run_name
    if review_ref:
        crop_group.attrs['detect_review_status_ref'] = review_ref
    if review_status:
        crop_group.attrs['detect_review_status'] = review_status
    if selection_policy:
        crop_group.attrs['detection_selection_policy'] = selection_policy
    _set_crop_pixel_contract_attrs(
        crop_group,
        crop_storage_mode=crop_storage_mode_resolved,
        video_source_type="zarr",
        acceleration="cpu",
    )
    if effective_run_provenance is not None:
        crop_group.attrs[RUN_PROVENANCE_ATTR] = dict(effective_run_provenance)
        crop_group.attrs[CLI_RUN_PROVENANCE_ATTR] = dict(effective_run_provenance)
    provenance_record = _build_crop_stage_provenance(
        created_at_utc=str(crop_group.attrs.get("created_at_utc")),
        command=" ".join(sys.argv),
        env_info=env_info,
        parameters={
            **dict(crop_params),
            "roi_image_representation": crop_group.attrs.get("roi_image_representation"),
            "roi_pixel_contract": crop_group.attrs.get("roi_pixel_contract"),
            "roi_pixel_contract_name": crop_group.attrs.get("roi_pixel_contract_name"),
        },
        parameter_source=param_source,
        inputs={
            "source_detect_run": detect_run_name,
            "source_refined_run": refined_run_name,
            "source_background_run": background_run_name,
            "frame_source": crop_group.attrs.get("video_source_type", "zarr"),
            "source_video_path": crop_group.attrs.get("video_source_path"),
        },
        detection_source={
            "type": source_type,
            "path": source_path,
            "method": get_detection_method(source_group),
        },
        scheduler={
            "dask_scheduler": scheduler,
            "dask_num_workers": num_workers or os.cpu_count(),
            "distributed": use_distributed,
        },
    )
    write_stage_provenance(crop_group, provenance_record)
    crop_group.attrs['crop_signature'] = build_crop_signature(crop_group.attrs)

    # Add GPU details if available (even though zarr workflow uses CPU)
    if env_info['gpu']['available'] and env_info['gpu'].get('devices'):
        primary_gpu = env_info['gpu']['devices'][0]
        crop_group.attrs.update({
            'gpu_name': primary_gpu.get('name', 'unknown'),
            'gpu_memory_total_mb': primary_gpu.get('memory_total_mb', 0),
            'gpu_compute_capability': primary_gpu.get('compute_capability', 'unknown'),
            'gpu_driver_version': env_info['gpu'].get('driver_version', 'unknown'),
            'cuda_version': env_info['gpu'].get('cuda_version', 'unknown'),
        })

    # Get video dimensions and scale factor
    ds_img_shape = root['raw_video/images_ds'].shape[1:]
    full_img_shape = root['raw_video/images_full'].shape[1:]
    scale_factor = ds_img_shape[0] / full_img_shape[0]

    roi_layout = build_canonical_crop_roi_layout(
        total_rois=total_detections,
        preferred_chunk_len=int(chunk_size),
        roi_storage="compressed",
        use_sharding=False,
        roi_shard_len=None,
    )

    # Create output arrays in crop group
    roi_images = None
    if crop_storage_mode_resolved == "materialized":
        roi_images = crop_group.create_array(
            'roi_images',
            **build_crop_roi_create_kwargs(
                total_rois=total_detections,
                roi_sz=roi_sz,
                layout=roi_layout,
                overwrite=True,
            )
        )
    crop_group.attrs.update(crop_roi_layout_attrs(roi_layout))
    if 'roi_shard_len' in crop_group.attrs and not roi_layout.roi_use_sharding:
        del crop_group.attrs['roi_shard_len']
    
    roi_coordinates_full = crop_group.create_array(
        'roi_coordinates_full',
        shape=(total_detections, 2),
        chunks=(min(chunk_size, total_detections), 2),
        dtype='i4',
        overwrite=True
    )
    
    roi_coordinates_ds = crop_group.create_array(
        'roi_coordinates_ds',
        shape=(total_detections, 2),
        chunks=(min(chunk_size, total_detections), 2),
        dtype='i4',
        overwrite=True
    )
    
    # Copy source coordinates to make crops self-contained
    console.print("[cyan]Copying source coordinates to crop group...[/cyan]")
    save_crop_metadata(
        crop_group=crop_group,
        source_group=source_group,
        source_path=source_path,
        source_type=source_type,
        detection_source=detection_source,
        total_detections=total_detections,
        num_frames=num_images
    )

    if crop_storage_mode_resolved == "geometry_only":
        coords_full, coords_ds = _compute_roi_coordinates(
            bbox_coords,
            roi_sz,
            full_img_shape,
            scale_factor=scale_factor,
        )
        crop_group['roi_coordinates_full'][:] = coords_full
        if coords_ds is not None:
            crop_group['roi_coordinates_ds'][:] = coords_ds

        frames_with_crops = int(np.sum(crop_group['frame_counts'][:] > 0))
        percent_cropped = (frames_with_crops / num_images) * 100 if num_images > 0 else 0
        duration = time.perf_counter() - start_time
        summary_stats = {
            'total_frames': num_images,
            'frames_with_crops': frames_with_crops,
            'total_rois_cropped': total_detections,
            'percent_frames_with_crops': round(percent_cropped, 2),
            'roi_size': list(roi_sz),
            'scale_factor': float(scale_factor),
            'roi_pixels_materialized': False,
        }
        crop_group.attrs['summary_statistics'] = summary_stats
        crop_group.attrs['duration_seconds'] = duration

        completion_text = f"""[green]✓[/green] Geometry-only crop run created

[bold]Performance:[/bold]
  Time: {duration:.1f}s ({duration/60:.1f} min)

[bold]Output:[/bold]
  Path: {zarr_path}
  Detection source: {source_type}
  Storage mode: {crop_storage_mode_resolved}

[bold]Arrays created:[/bold]
  - crop_runs/{run_group_name}/roi_coordinates_full: ({total_detections}, 2)
  - crop_runs/{run_group_name}/roi_coordinates_ds: ({total_detections}, 2)
  - crop_runs/{run_group_name}/bbox_norm_coords: ({total_detections}, 4)
  - crop_runs/{run_group_name}/frame_indices: ({total_detections},)
  - crop_runs/{run_group_name}/frame_counts: ({num_images},)
  - crop_runs/{run_group_name}/detection_indices: ({total_detections},)"""
        if 'detection_source' in crop_group:
            completion_text += f"\n  - crop_runs/{run_group_name}/detection_source: ({total_detections},)"

        console.print(Panel(
            Align.center(completion_text),
            title="[bold green]Cropping Complete[/bold green]",
            border_style="green"
        ))

        success = True
        crop_group.attrs['status'] = 'completed'
        crop_group.attrs['completed_at_utc'] = datetime.now(timezone.utc).isoformat()
        crop_group.attrs.pop('error_message', None)
        crop_group.attrs.pop('failed_at_utc', None)
        if crop_parent is not None:
            _finalize_crop_parent_pointers(
                crop_parent,
                run_name=run_group_name,
                crop_storage_mode=crop_storage_mode_resolved,
                success=True,
                previous_latest=previous_latest,
                previous_latest_materialized=previous_latest_materialized,
                previous_latest_any=previous_latest_any,
            )
        _emit_crop_step_status(
            root=root,
            zarr_path=zarr_path,
            status='ok',
            run_name=run_group_name,
            method=source_type,
            coverage_pct=percent_cropped,
            review_status=(crop_group.attrs.get('crop_review_status') if isinstance(crop_group.attrs.get('crop_review_status'), dict) else None),
            details={
                'reason': 'present',
                'run_state': str(crop_group.attrs.get('status')).strip().lower() if crop_group.attrs.get('status') is not None else None,
                'detection_source_type': source_type,
                'detection_source_path': source_path,
                'video_source_type': 'zarr',
                'crop_storage_mode': crop_storage_mode_resolved,
            },
            console=console,
        )
        return {
            'total_crops': total_detections,
            'frames_with_crops': frames_with_crops,
            'percent_cropped': percent_cropped,
            'duration_seconds': duration,
            'detection_source_type': source_type,
            'detection_source_path': source_path,
            'run_name': run_group_name,
            'crop_storage_mode': crop_storage_mode_resolved,
        }

    # Store path to this crop group in root for workers to find
    root.attrs['current_crop_group_path'] = crop_group.path

    # Build chunk frame slices
    chunk_slices = [slice(i, min(i + chunk_size, num_images))
                    for i in range(0, num_images, chunk_size)]
    console.print(f"Creating [yellow]{len(chunk_slices)}[/yellow] Dask tasks for cropping...")

    # Build detection counts per frame from frame_indices
    n_detections_per_frame = np.zeros(num_images, dtype='i4')
    unique_frames, counts = np.unique(frame_indices, return_counts=True)
    n_detections_per_frame[unique_frames] = counts
    
    # Precompute cumulative detection offsets ONCE on driver
    cumulative_detections = np.cumsum(np.insert(n_detections_per_frame, 0, 0))
    
    # Build chunks with output slices
    chunks = []
    for chunk_slice in chunk_slices:
        start_det = int(cumulative_detections[chunk_slice.start])
        end_det = int(cumulative_detections[chunk_slice.stop])
        
        if end_det > start_det:  # Only add if chunk has detections
            chunks.append((chunk_slice, (start_det, end_det)))
    
    frames_with_crops = int(np.sum(n_detections_per_frame > 0))

    if use_distributed:
        det_chunk_len = None
        if getattr(roi_images, "chunks", None):
            det_chunk_len = int(roi_images.chunks[0])
        if det_chunk_len:
            boundaries = set()
            for _, (start_det, end_det) in chunks:
                boundaries.add(start_det)
                boundaries.add(end_det)
            unsafe = [
                b for b in boundaries
                if b not in (0, total_detections) and (b % det_chunk_len) != 0
            ]
            if unsafe:
                sample = ", ".join(str(b) for b in sorted(unsafe)[:8])
                raise ValueError(
                    "Distributed cropping would write to overlapping Zarr chunks. "
                    f"Detection boundaries must align to chunk size {det_chunk_len}, "
                    f"but found misaligned boundaries at: {sample}. "
                    "Use --scheduler single-threaded (or processes) to avoid parallel writes, "
                    "or adjust crop chunking so detection boundaries align."
                )

    # Create delayed tasks
    delayed_tasks = [
        crop_and_store_chunk_delayed(
            zarr_path, frame_slice, out_slice, roi_sz, scale_factor, source_path
        )
        for frame_slice, out_slice in chunks
    ]
    
    # Dask config and scheduler
    dask.config.set({
        "distributed.worker.memory.target": 0.65,
        "distributed.worker.memory.spill": 0.75,
        "distributed.worker.memory.pause": 0.90,
        "distributed.worker.memory.terminate": 0.98,
    })
    
    client = None

    # Execute based on scheduler
    if use_distributed:
        # Distributed execution with Rich progress bar
        client = Client()
        console.print(f"[green]Dask distributed dashboard:[/green] {client.dashboard_link}")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeRemainingColumn(),
            console=console
        ) as progress:
            task = progress.add_task("[cyan]Cropping chunks (distributed)...", total=len(delayed_tasks))
            
            futures = client.compute(delayed_tasks)
            for future in as_completed(futures):
                _ = future.result()
                progress.update(task, advance=1)
        
        client.close()
    
    else:
        # Local execution with Rich progress bar
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeRemainingColumn(),
            console=console
        ) as progress:
            task = progress.add_task(f"[cyan]Cropping chunks ({scheduler})...", total=len(delayed_tasks))
            
            for d in delayed_tasks:
                _ = d.compute()
                progress.update(task, advance=1)

    # Summary stats and attrs
    percent_cropped = (frames_with_crops / num_images) * 100 if num_images > 0 else 0
    summary_stats = {
        'total_frames': num_images,
        'frames_with_crops': frames_with_crops,
        'total_rois_cropped': total_detections,
        'percent_frames_with_crops': round(percent_cropped, 2),
        'roi_size': list(roi_sz),
        'scale_factor': float(scale_factor)
    }
    crop_group.attrs['summary_statistics'] = summary_stats
    duration = time.perf_counter() - start_time
    crop_group.attrs['duration_seconds'] = duration

    # Clean up root attrs
    if 'current_crop_group_path' in root.attrs:
        del root.attrs['current_crop_group_path']

    # Create completion panel
    source_info = f"{source_type}"
    source_path_display = crop_group.attrs.get('detection_source_path')
    if source_path_display:
        source_info += f" [{source_path_display}]"
    if 'includes_interpolated' in crop_group.attrs and crop_group.attrs['includes_interpolated']:
        n_real = crop_group.attrs['n_real_detections']
        n_interp = crop_group.attrs['n_interpolated_detections']
        source_info += f" ({n_real} real + {n_interp} interpolated)"
    
    completion_text = f"""[green]✓[/green] Cropping completed successfully

[bold]Performance:[/bold]
  Time: {duration:.1f}s ({duration/60:.1f} min)
  ROIs/sec: {total_detections/duration:.1f}
  Throughput: {(total_detections * roi_sz[0] * roi_sz[1]) / (1024*1024*duration):.2f} MP/s

[bold]Output:[/bold]
  Path: {zarr_path}
  Detection source: {source_info}

[bold]Arrays created:[/bold]
  - crop_runs/{run_group_name}/roi_images: ({total_detections}, {roi_sz[0]}, {roi_sz[1]})
  - crop_runs/{run_group_name}/roi_coordinates_full: ({total_detections}, 2)
  - crop_runs/{run_group_name}/roi_coordinates_ds: ({total_detections}, 2)
  - crop_runs/{run_group_name}/bbox_norm_coords: ({total_detections}, 4)
  - crop_runs/{run_group_name}/frame_indices: ({total_detections},)
  - crop_runs/{run_group_name}/frame_counts: ({num_images},)"""
    
    if 'detection_source' in crop_group:
        completion_text += f"\n  - crop_runs/{run_group_name}/detection_source: ({total_detections},)"

    console.print(Panel(
        Align.center(completion_text),
        title="[bold green]Cropping Complete[/bold green]",
        border_style="green"
    ))

    success = True
    crop_group.attrs['status'] = 'completed'
    crop_group.attrs['completed_at_utc'] = datetime.now(timezone.utc).isoformat()
    crop_group.attrs.pop('error_message', None)
    crop_group.attrs.pop('failed_at_utc', None)
    if crop_parent is not None:
        mark_run_complete(
            crop_group,
            parent_group=crop_parent,
            run_name=run_group_name,
            run_provenance=effective_run_provenance,
        )
        _finalize_crop_parent_pointers(
            crop_parent,
            run_name=run_group_name,
            crop_storage_mode=crop_storage_mode_resolved,
            success=True,
            previous_latest=previous_latest,
            previous_latest_materialized=previous_latest_materialized,
            previous_latest_any=previous_latest_any,
        )
    _emit_crop_step_status(
        root=root,
        zarr_path=zarr_path,
        status='ok',
        run_name=run_group_name,
        method=source_type,
        coverage_pct=percent_cropped,
        review_status=(crop_group.attrs.get('crop_review_status') if isinstance(crop_group.attrs.get('crop_review_status'), dict) else None),
        details={
            'reason': 'present',
            'run_state': str(crop_group.attrs.get('status')).strip().lower() if crop_group.attrs.get('status') is not None else None,
            'detection_source_type': source_type,
            'detection_source_path': source_path,
            'video_source_type': 'zarr',
            'crop_storage_mode': crop_storage_mode_resolved,
        },
        console=console,
    )
    return {
        'total_crops': total_detections,
        'frames_with_crops': frames_with_crops,
        'percent_cropped': percent_cropped,
        'duration_seconds': duration,
        'detection_source_type': source_type,
        'detection_source_path': source_path,
        'run_name': run_group_name,
        'crop_storage_mode': crop_storage_mode_resolved,
    }


def _roi_cache_worker_main(argv: Optional[List[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Materialize temporary ROI cache from an external video source")
    parser.add_argument("--roi-cache-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--source-zarr", required=True, type=str)
    parser.add_argument("--source-crop-run", required=True, type=str)
    parser.add_argument("--cache-path", required=True, type=str)
    parser.add_argument("--write-backend", default="kvikio", choices=["standard", "kvikio"])
    parser.add_argument("--roi-storage", default="uncompressed", choices=["compressed", "uncompressed"])
    parser.add_argument("--use-sharding", action="store_true")
    parser.add_argument("--roi-chunk-size", type=int, default=1024)
    parser.add_argument("--roi-shard-size", type=int, default=None)
    parser.add_argument(
        "--gpu-chunk-frames",
        type=int,
        default=DEFAULT_SCRATCH_ROI_CACHE_GPU_CHUNK_FRAMES,
    )
    parser.add_argument("--require-kvikio", action="store_true")
    parser.add_argument("--prefer-gpu", action="store_true")
    parser.add_argument("--no-prefer-gpu", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    if args.prefer_gpu and args.no_prefer_gpu:
        raise SystemExit("Choose either --prefer-gpu or --no-prefer-gpu, not both.")
    prefer_gpu = True
    if args.no_prefer_gpu:
        prefer_gpu = False
    elif args.prefer_gpu:
        prefer_gpu = True

    console = Console()
    try:
        inputs = _load_external_roi_cache_inputs(
            source_zarr_path=args.source_zarr,
            crop_run_name=args.source_crop_run,
        )
        result = materialize_external_roi_cache(
            cache_path=args.cache_path,
            console=console,
            write_backend=args.write_backend,
            roi_storage=args.roi_storage,
            use_sharding=args.use_sharding,
            roi_chunk_size=args.roi_chunk_size,
            roi_shard_size=args.roi_shard_size,
            gpu_chunk_frames=args.gpu_chunk_frames,
            require_kvikio=args.require_kvikio,
            prefer_gpu=prefer_gpu,
            verbose=args.verbose,
            **inputs,
        )
        cache_root = zarr.open_group(str(Path(args.cache_path).expanduser().resolve()), mode="a")
        cache_root.attrs.update(
            {
                "cache_write_backend_requested": result.get("write_backend_requested"),
                "cache_write_backend_effective": result.get("write_backend_effective"),
                "cache_acceleration": result.get("acceleration"),
                "cache_fallback_reason": result.get("fallback_reason"),
                "cache_decode_seconds": result.get("decode_seconds"),
                "cache_compute_seconds": result.get("compute_seconds"),
                "cache_write_seconds": result.get("write_seconds"),
                "cache_duration_seconds": result.get("duration_seconds"),
                "cache_roi_chunk_len": result.get("roi_chunk_len"),
                "cache_roi_shard_len": result.get("roi_shard_len"),
                "cache_roi_storage": result.get("roi_storage"),
                "cache_roi_use_sharding": result.get("roi_use_sharding"),
                "cache_gpu_chunk_frames": result.get("gpu_chunk_frames"),
                "roi_image_representation": result.get("roi_image_representation"),
                "roi_pixel_contract": result.get("roi_pixel_contract"),
                "roi_pixel_contract_name": result.get("roi_pixel_contract_name"),
                "roi_cache_worker_result": json.dumps(result, sort_keys=True, default=str),
            }
        )
        cache_root.attrs["roi_cache_worker_mode"] = "subprocess"
        store = getattr(cache_root, "store", None)
        if store is not None:
            try:
                store.close()
            except Exception:
                pass
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
    except Exception as exc:
        console.print(f"[red]ROI cache worker failed:[/red] {exc}")
        console.print_exception()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)


def main():
    """CLI entry point."""
    if "--roi-cache-worker" in sys.argv:
        return _roi_cache_worker_main()

    import argparse
    import yaml
    
    parser = argparse.ArgumentParser(description="Crop ROIs from detections")
    parser.add_argument("zarr_path", type=str, help="Path to zarr file")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument(
        "--source-type",
        type=str,
        default=None,
        choices=['auto', 'refined', 'detect', 'manual', 'filtered', 'interpolated'],
        help=(
            "Detection source to use. 'auto' prefers the canonical current "
            "refined surface and falls back to raw detect; 'refined' requires "
            "the canonical curated refined surface. "
            "'manual'/'filtered'/'interpolated' are legacy sparse "
            "compatibility modes for older archives."
        ),
    )
    parser.add_argument("--source-path", type=str, default=None,
                       help=(
                           "Explicit detection source path (e.g. "
                           "detect_runs/<run> or the canonical refined path "
                           "refined_detect_runs/<run>/instances)"
                       ))
    parser.add_argument(
        "--selection-policy",
        type=str,
        default=None,
        choices=["training", "full_recording"],
        help="Policy for auto source selection.",
    )
    parser.add_argument("--scheduler", type=str, default=None,
                       choices=['processes', 'threads', 'distributed'],
                       help="Dask scheduler type")
    parser.add_argument("--num-workers", type=int, default=None,
                       help="Number of workers")
    parser.add_argument("--acceleration", type=str, default=None,
                       choices=['auto', 'gpu', 'cpu'],
                       help="Acceleration mode for external video cropping")
    parser.add_argument(
        "--crop-storage-mode",
        type=str,
        default=None,
        choices=["materialized", "geometry_only"],
        help="Crop persistence mode for the new run.",
    )
    parser.add_argument(
        "--external-write-backend",
        type=str,
        default=None,
        choices=["standard", "kvikio"],
        help="Write backend for external-video crop runs.",
    )
    parser.add_argument(
        "--external-roi-storage",
        type=str,
        default=None,
        choices=["compressed", "uncompressed"],
        help="ROI image storage mode for external-video crop runs.",
    )
    parser.add_argument(
        "--external-use-sharding",
        action="store_true",
        help="Enable sharding for external-video ROI image writes.",
    )
    parser.add_argument(
        "--no-external-use-sharding",
        action="store_true",
        help="Disable sharding for external-video ROI image writes.",
    )
    parser.add_argument(
        "--external-roi-chunk-size",
        type=int,
        default=None,
        help="Detection-axis chunk length for external-video ROI writes.",
    )
    parser.add_argument(
        "--external-roi-shard-size",
        type=int,
        default=None,
        help="Detection-axis shard length for external-video ROI writes.",
    )
    parser.add_argument(
        "--external-gpu-chunk-frames",
        type=int,
        default=None,
        help="Frame count decoded per GPU crop chunk (external-video mode).",
    )
    parser.add_argument(
        "--require-kvikio",
        action="store_true",
        help="Fail if kvikIO GDS writes cannot be enabled in external-video mode.",
    )
    parser.add_argument("--no-gpu", action="store_true",
                       help="Disable GPU acceleration")
    parser.add_argument("--force-cpu", action="store_true",
                       help="Force CPU processing even if GPU available")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    console = Console()
    
    # Warn if distributed requested but not available
    if args.scheduler == "distributed" and not HAVE_DISTRIBUTED:
        console.print("[yellow]Warning: distributed scheduler not available, falling back to processes[/yellow]")
        args.scheduler = "processes"

    crop_params = config.get('crop', {}) or {}
    config_source_type = crop_params.get('source_type')
    config_source_path = crop_params.get('source_path')

    cli_source_type = args.source_type
    cli_source_path = args.source_path

    source_path = cli_source_path or config_source_path
    if source_path:
        source_path = str(source_path).strip().strip('/')
        if not source_path:
            source_path = None

    raw_source_type = cli_source_type or config_source_type
    resolved_source_type = infer_detection_source_type(source_path, raw_source_type)

    selection_policy = args.selection_policy or crop_params.get('selection_policy')
    external_use_sharding = None
    if args.external_use_sharding and args.no_external_use_sharding:
        raise SystemExit("Choose either --external-use-sharding or --no-external-use-sharding, not both.")
    if args.external_use_sharding:
        external_use_sharding = True
    elif args.no_external_use_sharding:
        external_use_sharding = False

    try:
        results = crop_detections(
            zarr_path=args.zarr_path,
            config=config,
            source_type=resolved_source_type,
            source_path=source_path,
            selection_policy=selection_policy,
            scheduler=args.scheduler,
            num_workers=args.num_workers,
            console=console,
            acceleration=args.acceleration,
            external_write_backend=args.external_write_backend,
            external_roi_storage=args.external_roi_storage,
            external_use_sharding=external_use_sharding,
            external_roi_chunk_size=args.external_roi_chunk_size,
            external_roi_shard_size=args.external_roi_shard_size,
            external_gpu_chunk_frames=args.external_gpu_chunk_frames,
            external_require_kvikio=args.require_kvikio,
            crop_storage_mode=args.crop_storage_mode,
            use_gpu_allowed=not args.no_gpu,
            force_cpu=args.force_cpu,
            verbose=args.verbose
        )
        console.print(f"\n[green]Cropping complete![/green]")
        console.print(f"Total ROIs cropped: {results['total_crops']}")
        console.print(f"Detection source: {results['detection_source_type']}")
        if results.get('detection_source_path'):
            console.print(f"Source path: {results['detection_source_path']}")
        return 0
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
