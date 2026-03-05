# src/fisheye/refinement/refine_eye_masks.py
"""Refine eye-mask segmentation runs using keypoint geometry.

This post-processes an existing ``eye_masks_runs`` entry (typically produced by
YOLO segmentation) and rewrites left/right mask channels so they align with the
keypoint pipeline's anatomical labels. The refined result is stored under
``refined_eye_masks_runs`` – mirroring the traditional schema but avoiding any
mutation of the original segmentation output.

Usage::

    python -m fisheye.refinement.refine_eye_masks /path/to/archive.zarr \
        --source-run yolo_2025_01_01 \
        --run-name refined_from_yolo
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional, Tuple

import dask
import dask.array as da
import numpy as np
import zarr
from zarr.core.dtype import VariableLengthUTF8
from dask import delayed
from dask.diagnostics import ProgressBar
from rich.console import Console
from scipy.ndimage import distance_transform_edt, gaussian_filter, median_filter
from skimage import measure, morphology
from skimage.measure import EllipseModel
from collections import Counter

from ..registry.db import Registry, RegistryPaths, resolve_dataset_id
from ..registry.status_ledger import upsert_recording_step_status
from ..shared.mask_source import load_mask_bundle
from ..shared.provenance_attrs import build_source_keypoints_attrs, resolve_source_keypoints_run
from ..shared.row_alignment import assert_row_alignment
from ..shared.stage_provenance import build_stage_provenance, write_stage_provenance
from ..shared.zarr.schema import get_run_group
from ..utils.system import get_environment_info, get_git_info


REFINED_STAGE_NAME = "refined_eye_masks"

# Refinement knobs: keep public API stable while letting us tune connectivity.
_SMOOTHING_MODE = "median"  # {"off","median","sdf","morph"}
_MEDIAN_K = 3
_SDF_SIGMA = 1.0
_MORPH_CLOSING_RADIUS = 1
_MORPH_OPENING_RADIUS = 0
_MIN_OBJECT_AREA = 12
_PROBABILITY_THRESHOLD = 0.45
_AREA_FILTER_Z_DEFAULT = 2.0
_AREA_FILTER_MODE_DEFAULT = "either"
_SUCCESS_MIN_EYE_AREA_PX_DEFAULT = 50.0
_REFINED_EYE_MASKS_STATUS_SOURCE = "runtime_refine_eye_masks"

_ZARR_GROUP_CACHE: Dict[str, zarr.Group] = {}
_ZARR_ARRAY_CACHE: Dict[Tuple[str, str], zarr.Array] = {}


def _status_text(value: object) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, bytes):
        text = value.decode("utf-8", "ignore").strip()
    else:
        text = str(value).strip()
    return text or None


def _status_float(value: object) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_positive_float(value: object) -> Optional[float]:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if numeric <= 0:
        return None
    return float(numeric)


def _safe_zarr_mtime_ns(path: Path) -> Optional[int]:
    try:
        return int(path.stat().st_mtime_ns)
    except OSError:
        return None


def _emit_refined_eye_masks_status(
    *,
    root: zarr.Group,
    zarr_path: Path,
    status: str,
    run_name: Optional[str],
    method: Optional[str],
    coverage_pct: Optional[float],
    review_status: Optional[Dict[str, object]],
    details: Dict[str, object],
    console: Optional[Console],
) -> None:
    try:
        dataset_id, session_uuid = resolve_dataset_id(root, zarr_path)
        recording_id = _status_text(root.attrs.get("recording_id")) or _status_text(session_uuid)
        zarr_use = _status_text(root.attrs.get("zarr_use"))
        zarr_purpose = _status_text(root.attrs.get("zarr_purpose"))

        registry_path = RegistryPaths.from_env(Path.cwd()).path
        registry = Registry(registry_path)
        try:
            registry.upsert_dataset(
                dataset_id,
                session_uuid=session_uuid,
                zarr_path=zarr_path,
                recording_id=recording_id,
                zarr_use=zarr_use,
                zarr_purpose=zarr_purpose,
            )
            upsert_recording_step_status(
                registry,
                dataset_id=dataset_id,
                recording_id=recording_id,
                step_name="refined_eye_masks",
                status=status,
                run_name=run_name,
                method=method,
                coverage_pct=coverage_pct,
                review_status_json=review_status,
                details_json=details,
                source=_REFINED_EYE_MASKS_STATUS_SOURCE,
                zarr_mtime_ns=_safe_zarr_mtime_ns(zarr_path),
            )
        finally:
            registry.close()
    except Exception as exc:
        if console is not None:
            console.print(
                f"[yellow]Warning:[/yellow] failed to write recording step status "
                f"for refined_eye_masks: {exc}"
            )


def _refresh_refined_eye_masks_registry_rows(
    *,
    root: zarr.Group,
    zarr_path: Path,
    console: Optional[Console],
) -> None:
    """Refresh eye-mask performance/quality rows for one dataset (non-fatal)."""
    try:
        dataset_id, session_uuid = resolve_dataset_id(root, zarr_path)
        recording_id = _status_text(root.attrs.get("recording_id")) or _status_text(session_uuid)
        zarr_use = _status_text(root.attrs.get("zarr_use")) or _status_text(root.attrs.get("zarr_purpose"))
        zarr_purpose = _status_text(root.attrs.get("zarr_purpose"))

        registry_path = RegistryPaths.from_env(Path.cwd()).path
        registry = Registry(registry_path)
        try:
            registry.upsert_dataset(
                dataset_id,
                session_uuid=session_uuid,
                zarr_path=zarr_path,
                recording_id=recording_id,
                zarr_use=zarr_use,
                zarr_purpose=zarr_purpose,
            )
            perf_rows = int(
                registry.refresh_eye_mask_performance_for_dataset(
                    dataset_id,
                    zarr_path=zarr_path,
                    recording_id=recording_id,
                    zarr_use=zarr_use,
                )
            )
            quality_rows = int(
                registry.refresh_eye_mask_quality_for_dataset(
                    dataset_id,
                    zarr_path=zarr_path,
                    recording_id=recording_id,
                    zarr_use=zarr_use,
                )
            )
        finally:
            registry.close()
        if console is not None:
            console.print(
                "[dim]Refreshed registry eye-mask rows "
                f"(performance={perf_rows}, quality={quality_rows}) for dataset {dataset_id}[/dim]"
            )
    except Exception as exc:
        if console is not None:
            console.print(
                "[yellow]Warning:[/yellow] failed to refresh eye-mask registry rows "
                f"for refined_eye_masks: {exc}"
            )


def _get_zarr_array(zarr_path: str, array_path: str) -> zarr.Array:
    """Fetch (and memoize per process) a Zarr array for repeated chunk reads."""
    key = (zarr_path, array_path)
    cached = _ZARR_ARRAY_CACHE.get(key)
    if cached is not None:
        return cached

    group = _ZARR_GROUP_CACHE.get(zarr_path)
    if group is None:
        group = zarr.open(zarr_path, mode="r", use_consolidated=False)
        _ZARR_GROUP_CACHE[zarr_path] = group

    arr = group[array_path]
    _ZARR_ARRAY_CACHE[key] = arr
    return arr


def _resolve_keypoint_group(
    root: zarr.Group,
    keypoint_run: Optional[str],
    source_keypoint_group: Optional[str],
    source_keypoint_run: Optional[str],
    *,
    allow_latest_fallback: bool = False,
) -> Tuple[zarr.Group, str, str]:
    """Resolve keypoint lineage with fail-closed defaults.

    Resolution order:
    1. Explicit `keypoint_run` argument.
    2. Source attrs (`source_keypoint_group` + `source_keypoint_run`).
    3. Source run only (resolved across refined/raw groups; ambiguous matches fail).
    4. Optional latest fallback only when `allow_latest_fallback=True`.
    """
    refined = root.get("refined_keypoints_runs")
    raw = root.get("keypoints_runs")

    if keypoint_run:
        if refined is not None and keypoint_run in refined:
            return refined[keypoint_run], keypoint_run, "refined_keypoints_runs"
        if raw is not None and keypoint_run in raw:
            return raw[keypoint_run], keypoint_run, "keypoints_runs"
        raise ValueError(f"Keypoint run '{keypoint_run}' not found in refined or raw runs.")

    if source_keypoint_group and source_keypoint_run:
        group = root.get(source_keypoint_group)
        if group is None:
            raise ValueError(
                f"Source keypoint group '{source_keypoint_group}' is missing. "
                "Pass --keypoint-run explicitly or backfill source lineage attrs."
            )
        if source_keypoint_run not in group:
            raise ValueError(
                f"Source keypoint run '{source_keypoint_run}' not found in group '{source_keypoint_group}'. "
                "Pass --keypoint-run explicitly or backfill source lineage attrs."
            )
        return group[source_keypoint_run], source_keypoint_run, source_keypoint_group

    if source_keypoint_group and not source_keypoint_run:
        raise ValueError(
            f"Source keypoint group '{source_keypoint_group}' is present but source keypoint run is missing. "
            "Pass --keypoint-run explicitly or backfill source lineage attrs."
        )

    if source_keypoint_run:
        matches: List[Tuple[zarr.Group, str, str]] = []
        if refined is not None and source_keypoint_run in refined:
            matches.append((refined[source_keypoint_run], source_keypoint_run, "refined_keypoints_runs"))
        if raw is not None and source_keypoint_run in raw:
            matches.append((raw[source_keypoint_run], source_keypoint_run, "keypoints_runs"))
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise ValueError(
                f"Source keypoint run '{source_keypoint_run}' exists in both refined and raw groups. "
                "Pass --keypoint-run explicitly or set source_keypoint_group."
            )
        raise ValueError(
            f"Source keypoint run '{source_keypoint_run}' not found in refined or raw groups. "
            "Pass --keypoint-run explicitly or backfill source lineage attrs."
        )

    if not allow_latest_fallback:
        raise ValueError(
            "Missing source keypoint lineage attrs on source eye-mask run "
            "(expected source_keypoints_run/source_keypoint_run and source_keypoint_group). "
            "Pass --keypoint-run explicitly or backfill lineage attrs. "
            "Use --allow-latest-keypoint-fallback only for legacy archives."
        )

    refined_latest = refined.attrs.get("latest") if refined is not None else None
    raw_latest = raw.attrs.get("latest") if raw is not None else None

    if refined is not None and refined_latest in refined:
        return refined[refined_latest], refined_latest, "refined_keypoints_runs"
    if raw is not None and raw_latest in raw:
        return raw[raw_latest], raw_latest, "keypoints_runs"
    raise ValueError("No keypoint runs found; run keypoints stage first.")


def _validate_input_row_alignment(
    *,
    total_rois: int,
    src_run: zarr.Group,
    src_run_name: str,
    kp_group: zarr.Group,
    keypoint_group_name: str,
    keypoint_run_name: str,
    success_dataset_name: str,
    binary_data: object,
    probs_data: object,
) -> None:
    keypoints_roi = kp_group.get("keypoints_roi")
    if keypoints_roi is None:
        raise ValueError(f"Keypoint run '{keypoint_run_name}' missing 'keypoints_roi'.")
    heading = kp_group.get("heading")
    if heading is None:
        raise ValueError(f"Keypoint run '{keypoint_run_name}' missing 'heading'.")
    success_arr = kp_group.get(success_dataset_name)
    if success_arr is None:
        raise ValueError(f"Keypoint run '{keypoint_run_name}' missing '{success_dataset_name}'.")

    assert_row_alignment(
        total_rois,
        (
            ("loaded_mask_bundle/binary", binary_data),
            ("loaded_mask_bundle/probs", probs_data),
            (f"eye_masks_runs/{src_run_name}/masks_roi", src_run.get("masks_roi")),
            (f"eye_masks_runs/{src_run_name}/mask_probs_roi", src_run.get("mask_probs_roi")),
            (
                f"eye_masks_runs/{src_run_name}/mask_probs_roi_refined",
                src_run.get("mask_probs_roi_refined"),
            ),
            (f"{keypoint_group_name}/{keypoint_run_name}/keypoints_roi", keypoints_roi),
            (f"{keypoint_group_name}/{keypoint_run_name}/heading", heading),
            (
                f"{keypoint_group_name}/{keypoint_run_name}/{success_dataset_name}",
                success_arr,
            ),
            (f"eye_masks_runs/{src_run_name}/frame_indices", src_run.get("frame_indices")),
            (f"eye_masks_runs/{src_run_name}/detection_indices", src_run.get("detection_indices")),
            (f"eye_masks_runs/{src_run_name}/detection_source", src_run.get("detection_source")),
        ),
        stage="refine_eye_masks input",
    )


def _compression_kwargs(array: zarr.Array) -> Dict[str, object]:
    """Return compression-related kwargs compatible with Zarr v2/v3 arrays."""
    kwargs: Dict[str, object] = {}
    compressors = getattr(array, "compressors", None)
    if compressors:
        kwargs["compressors"] = compressors
    else:
        try:
            compressor = array.compressors
        except (TypeError, AttributeError):
            compressor = None
        if compressor is not None:
            kwargs["compressor"] = compressor

    chunk_codecs = getattr(array, "chunk_codecs", None)
    if chunk_codecs:
        kwargs.setdefault("chunk_codecs", chunk_codecs)

    filters = getattr(array, "filters", None)
    if filters:
        kwargs.setdefault("filters", filters)

    return kwargs


def _append_reason_tag(reason: Optional[str], tag: str) -> str:
    """Append a tag to a pipe-delimited reason string without duplicates."""
    if not reason:
        return tag
    parts = set(reason.split("|"))
    if tag in parts:
        return reason
    return f"{reason}|{tag}"


def _extract_contour(mask: np.ndarray, min_points: int) -> Optional[np.ndarray]:
    contours = measure.find_contours(mask.astype(float), 0.5)
    if not contours:
        return None
    contour = max(contours, key=lambda c: c.shape[0])
    if contour.shape[0] < min_points:
        return None
    return contour[:, ::-1]  # Convert to (x, y)

def _largest_component(mask: np.ndarray) -> np.ndarray:
    """Return the largest 8-connected component of a boolean mask."""
    labeled = measure.label(mask.astype(bool), connectivity=2)
    num_components = int(labeled.max())
    if num_components <= 1:
        return mask.astype(bool, copy=False)

    counts = np.bincount(labeled.ravel())
    if counts.size <= 1:
        return np.zeros_like(mask, dtype=bool)

    counts[0] = 0
    label = int(np.argmax(counts))
    return labeled == label


def _remove_small(mask: np.ndarray) -> np.ndarray:
    """Remove crumbs below the configured minimum area."""
    cleaned = morphology.remove_small_objects(
        mask.astype(bool),
        min_size=_MIN_OBJECT_AREA,
        connectivity=2,
    )
    return cleaned.astype(bool, copy=False)


def _mask_centroid(mask: np.ndarray) -> Optional[np.ndarray]:
    """Return the (x, y) centroid of a binary mask, or ``None`` if empty."""
    mask_bool = mask.astype(bool, copy=False)
    if not mask_bool.any():
        return None
    ys, xs = np.nonzero(mask_bool)
    if ys.size == 0:
        return None
    return np.array([float(xs.mean()), float(ys.mean())], dtype=np.float32)


def _smooth_binary_mask(mask: np.ndarray) -> Tuple[np.ndarray, bool]:
    """Apply topology-friendly smoothing while preserving mask connectivity."""
    mask_bool = mask.astype(bool, copy=False)
    if mask_bool.sum() == 0:
        return mask_bool, False

    mode = (_SMOOTHING_MODE or "").lower()
    if mode == "off":
        return mask_bool, False

    smoothed = mask_bool
    if mode == "median":
        k = max(1, int(_MEDIAN_K))
        if k % 2 == 0:
            k += 1  # median filter requires odd kernel; bump minimally to avoid bias
        smoothed = median_filter(smoothed.astype(np.uint8), size=k) > 0
    elif mode == "sdf":
        inside = distance_transform_edt(smoothed)
        outside = distance_transform_edt(~smoothed)
        sdf = inside - outside
        sigma = float(_SDF_SIGMA)
        if sigma > 0.0:
            sdf = gaussian_filter(sdf, sigma=sigma)
        smoothed = sdf >= 0.0
    elif mode == "morph":
        if _MORPH_CLOSING_RADIUS > 0:
            struct = morphology.disk(int(_MORPH_CLOSING_RADIUS))
            smoothed = morphology.binary_closing(smoothed, struct)
        if _MORPH_OPENING_RADIUS > 0:
            struct = morphology.disk(int(_MORPH_OPENING_RADIUS))
            smoothed = morphology.binary_opening(smoothed, struct)
    else:
        return mask_bool, False

    smoothed = smoothed.astype(bool, copy=False)
    smoothed = _largest_component(smoothed)
    smoothed = _remove_small(smoothed)
    changed = not np.array_equal(smoothed, mask_bool)
    return smoothed, changed


def _select_component_near_keypoint(mask: np.ndarray, keypoint: np.ndarray) -> Tuple[np.ndarray, np.ndarray, bool]:
    """Keep the connected component closest to the given keypoint."""
    labeled = measure.label(mask.astype(bool), connectivity=2)
    num_components = int(labeled.max())
    if num_components <= 1:
        return mask.astype(bool, copy=False), np.zeros_like(mask, dtype=bool), False

    keypoint = np.asarray(keypoint, dtype=np.float32)
    best_label = None
    best_distance = float("inf")

    props = measure.regionprops(labeled)
    for prop in props:
        cy, cx = prop.centroid  # row, col
        dist = float(math.hypot(cx - keypoint[0], cy - keypoint[1]))
        if dist < best_distance:
            best_distance = dist
            best_label = prop.label

    selected = labeled == best_label if best_label is not None else np.zeros_like(mask, dtype=bool)
    removed = mask & ~selected
    return selected, removed, True


def _enforce_single_component(
    left_mask: np.ndarray,
    right_mask: np.ndarray,
    eye_left: np.ndarray,
    eye_right: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Ensure each eye mask has a single contiguous component."""

    left_in = left_mask.astype(bool, copy=False)
    right_in = right_mask.astype(bool, copy=False)

    left_pre = _remove_small(left_in)
    right_pre = _remove_small(right_in)
    left_pre_changed = not np.array_equal(left_pre, left_in)
    right_pre_changed = not np.array_equal(right_pre, right_in)

    left_selected, left_removed, left_changed = _select_component_near_keypoint(left_pre, eye_left)
    right_selected, right_removed, right_changed = _select_component_near_keypoint(right_pre, eye_right)

    # Reassign removed pixels to the opposite mask to preserve union coverage.
    if left_removed.any():
        right_selected = right_selected | left_removed
    if right_removed.any():
        left_selected = left_selected | right_removed

    # Final pass to guarantee single component after reassignment.
    left_final, left_removed_final, left_changed_final = _select_component_near_keypoint(left_selected, eye_left)
    right_final, right_removed_final, right_changed_final = _select_component_near_keypoint(right_selected, eye_right)

    left_out = _remove_small(left_final)
    right_out = _remove_small(right_final)
    left_post_changed = not np.array_equal(left_out, left_final)
    right_post_changed = not np.array_equal(right_out, right_final)

    reassigned_pixels = int(left_removed.sum() + right_removed.sum())
    reassigned_pixels += int(left_removed_final.sum() + right_removed_final.sum())

    changed_flags = np.array(
        [
            left_pre_changed
            or left_changed
            or left_changed_final
            or left_post_changed
            or bool(left_removed.sum())
            or bool(left_removed_final.sum()),
            right_pre_changed
            or right_changed
            or right_changed_final
            or right_post_changed
            or bool(right_removed.sum())
            or bool(right_removed_final.sum()),
        ],
        dtype=bool,
    )

    return left_out, right_out, changed_flags, reassigned_pixels


@dataclass
class ROIOutput:
    """Container for refined measurements of a single ROI."""

    masks: np.ndarray  # (2, H, W) uint8
    ellipse_params: np.ndarray  # (2, 5) float32
    ellipse_success: np.ndarray  # (2,) bool
    centroids: np.ndarray  # (2, 2) float32 (x, y) or nan
    contours: Tuple[Optional[np.ndarray], Optional[np.ndarray]]
    eye_separation: float
    reason: Optional[str]
    smoothing_changed: np.ndarray  # (2,) bool
    reassigned_pixels: int
    used_probabilities: bool = False
    probabilities: Optional[np.ndarray] = None  # (2, H, W) float32
    refined_areas: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    source_areas: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    source_union_area: float = float("nan")
    refined_union_area: float = float("nan")
    centroid_errors: np.ndarray = field(default_factory=lambda: np.full(2, np.nan, dtype=np.float32))
    symmetry_offsets: np.ndarray = field(default_factory=lambda: np.full(2, np.nan, dtype=np.float32))
    keypoint_separation: float = float("nan")
    separation_delta: float = float("nan")
    axis_ratio: np.ndarray = field(default_factory=lambda: np.full(2, np.nan, dtype=np.float32))
    circularity: np.ndarray = field(default_factory=lambda: np.full(2, np.nan, dtype=np.float32))
    probability_mean: Optional[np.ndarray] = None  # (2,) float32
    probability_max: Optional[np.ndarray] = None  # (2,) float32
    probability_var: Optional[np.ndarray] = None  # (2,) float32
    probability_high_fraction: Optional[np.ndarray] = None  # (2,) float32


def _prepare_run_group(root: zarr.Group, run_name: Optional[str], console: Console) -> Tuple[zarr.Group, str]:
    parent_name = f"{REFINED_STAGE_NAME}_runs"
    parent = root.require_group(parent_name)
    if run_name:
        if run_name in parent:
            raise ValueError(f"{parent_name}/{run_name} already exists")
        run_group = parent.create_group(run_name)
        parent.attrs["latest"] = run_name
        console.print(f"Created run group: [cyan]{parent_name}/{run_name}[/cyan]")
        return run_group, run_name
    return get_run_group(root, REFINED_STAGE_NAME, console=console, create_new=True)


def _split_mask_by_keypoints(
    union_mask: np.ndarray,
    eye_left: np.ndarray,
    eye_right: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Split a union mask into left/right halves using closest-keypoint distance."""

    left_mask = np.zeros_like(union_mask, dtype=bool)
    right_mask = np.zeros_like(union_mask, dtype=bool)

    ys, xs = np.nonzero(union_mask)
    if ys.size == 0:
        return left_mask, right_mask

    if not (np.all(np.isfinite(eye_left)) and np.all(np.isfinite(eye_right))):
        return left_mask, right_mask

    if np.allclose(eye_left, eye_right):
        return left_mask, right_mask

    x_coords = xs.astype(np.float32)
    y_coords = ys.astype(np.float32)

    dist_left = (x_coords - float(eye_left[0])) ** 2 + (y_coords - float(eye_left[1])) ** 2
    dist_right = (x_coords - float(eye_right[0])) ** 2 + (y_coords - float(eye_right[1])) ** 2

    assign_left = dist_left <= dist_right
    left_mask[ys[assign_left], xs[assign_left]] = True
    right_mask[ys[~assign_left], xs[~assign_left]] = True
    return left_mask, right_mask


def _rotation_matrix(angle_degrees: float) -> np.ndarray:
    """Create a 2×2 rotation matrix matching the keypoint pipeline."""
    theta = math.radians(angle_degrees)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    return np.array([[cos_t, -sin_t], [sin_t, cos_t]], dtype=np.float32)


def _split_mask_by_heading(
    union_mask: np.ndarray,
    heading_deg: float,
    center: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fallback: split mask using a heading-aligned midpoint plane."""

    left_mask = np.zeros_like(union_mask, dtype=bool)
    right_mask = np.zeros_like(union_mask, dtype=bool)

    ys, xs = np.nonzero(union_mask)
    if ys.size == 0 or not np.isfinite(heading_deg):
        return left_mask, right_mask

    center = center.astype(np.float32)
    coords = np.stack([xs.astype(np.float32), ys.astype(np.float32)], axis=1) - center
    rot = _rotation_matrix(float(heading_deg))
    rotated = coords @ rot.T

    split_mask = rotated[:, 1] <= 0.0
    left_mask[ys[split_mask], xs[split_mask]] = True
    right_mask[ys[~split_mask], xs[~split_mask]] = True
    return left_mask, right_mask


def _measure_mask(mask: np.ndarray, min_contour_points: int = 5) -> Tuple[bool, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Extract metrics from a binary mask using least-squares ellipse fitting."""

    if mask.sum() == 0:
        ellipse = np.full(5, np.nan, dtype=np.float32)
        centroid = np.full(2, np.nan, dtype=np.float32)
        return False, ellipse, centroid, None

    # Extract contour first (already in x, y format)
    contour = _extract_contour(mask.astype(float), min_contour_points)
    if contour is None:
        ellipse = np.full(5, np.nan, dtype=np.float32)
        centroid = np.full(2, np.nan, dtype=np.float32)
        return False, ellipse, centroid, None

    contour = contour.astype(np.float32)

    # Fit ellipse to contour using least-squares (more stable than moments)
    ellipse_model = EllipseModel()
    success = ellipse_model.estimate(contour)

    if not success or ellipse_model.params is None:
        # Fallback: use centroid from mask, return NaN for ellipse params
        region_mask = mask.astype(np.uint8)
        props = measure.regionprops(region_mask)
        if props:
            centroid = np.array([float(props[0].centroid[1]), float(props[0].centroid[0])], dtype=np.float32)
        else:
            centroid = np.full(2, np.nan, dtype=np.float32)
        ellipse = np.full(5, np.nan, dtype=np.float32)
        return False, ellipse, centroid, contour

    # EllipseModel.params = (xc, yc, a, b, theta)
    # a, b are semi-axes; theta is rotation angle in radians
    xc, yc, a, b, theta = ellipse_model.params

    # Canonicalize ellipse parameters so major >= minor across refined outputs.
    if a < b:
        a, b = b, a
        theta = theta + (math.pi / 2.0)

    centroid = np.array([float(xc), float(yc)], dtype=np.float32)

    # Convert semi-axes to full axis lengths (to match existing schema)
    # and angle to degrees
    ellipse = np.array(
        [
            float(xc),
            float(yc),
            float(2 * a),  # major_axis_length (full length)
            float(2 * b),  # minor_axis_length (full length)
            float(np.rad2deg(theta)),  # angle in degrees
        ],
        dtype=np.float32,
    )

    return True, ellipse, centroid, contour


def _compute_roi_metrics(
    left_mask: np.ndarray,
    right_mask: np.ndarray,
    source_left: np.ndarray,
    source_right: np.ndarray,
    source_union: np.ndarray,
    centroids: np.ndarray,
    eye_left: np.ndarray,
    eye_right: np.ndarray,
    keypoints_valid: bool,
    refined_separation: float,
    ellipse_params: np.ndarray,
    contours: Tuple[Optional[np.ndarray], Optional[np.ndarray]],
    probs_out: Optional[np.ndarray],
) -> Dict[str, object]:
    refined_left = left_mask.astype(bool, copy=False)
    refined_right = right_mask.astype(bool, copy=False)
    source_left = source_left.astype(bool, copy=False)
    source_right = source_right.astype(bool, copy=False)
    source_union = source_union.astype(bool, copy=False)

    refined_areas = np.array(
        [refined_left.sum(), refined_right.sum()],
        dtype=np.float32,
    )
    source_areas = np.array(
        [source_left.sum(), source_right.sum()],
        dtype=np.float32,
    )
    union_refined_area = float(refined_areas.sum())
    union_source_area = float(source_union.sum())

    centroid_errors = np.full(2, np.nan, dtype=np.float32)
    keypoint_coords = (np.asarray(eye_left, dtype=np.float32), np.asarray(eye_right, dtype=np.float32))
    for idx in range(2):
        centroid = centroids[idx]
        kp = keypoint_coords[idx]
        if np.all(np.isfinite(centroid)) and np.all(np.isfinite(kp)):
            centroid_errors[idx] = float(np.linalg.norm(centroid - kp))

    keypoint_separation = float(np.linalg.norm(keypoint_coords[0] - keypoint_coords[1])) if np.all(np.isfinite(keypoint_coords[0])) and np.all(np.isfinite(keypoint_coords[1])) else float("nan")
    if not np.isfinite(refined_separation) or not np.isfinite(keypoint_separation):
        separation_delta = float("nan")
    else:
        separation_delta = float(refined_separation - keypoint_separation)

    axis_ratio = np.full(2, np.nan, dtype=np.float32)
    for idx in range(2):
        major = float(ellipse_params[idx, 2]) if np.isfinite(ellipse_params[idx, 2]) else float("nan")
        minor = float(ellipse_params[idx, 3]) if np.isfinite(ellipse_params[idx, 3]) else float("nan")
        if major > 0 and minor > 0:
            axis_ratio[idx] = float(minor / major)

    circularity = np.full(2, np.nan, dtype=np.float32)
    for idx, mask in enumerate((refined_left, refined_right)):
        area = refined_areas[idx]
        if area <= 0:
            continue
        if contours[idx] is not None and len(contours[idx]) >= 2:
            contour = contours[idx]
            diffs = np.diff(contour, axis=0, append=contour[0:1])
            perimeter = float(np.linalg.norm(diffs, axis=1).sum())
        else:
            try:
                perimeter = float(measure.perimeter(mask.astype(np.uint8), neighborhood=8))
            except TypeError:
                perimeter = float(measure.perimeter(mask.astype(np.uint8), neighbourhood=8))
        if perimeter > 0:
            circularity[idx] = float((4.0 * math.pi * area) / (perimeter ** 2 + 1e-12))

    symmetry_offsets = np.full(2, np.nan, dtype=np.float32)
    if keypoints_valid:
        direction = keypoint_coords[1] - keypoint_coords[0]
        norm_dir = float(np.linalg.norm(direction))
        if norm_dir > 0:
            axis = direction / norm_dir
            perp = np.array([-axis[1], axis[0]], dtype=np.float32)
            midpoint = (keypoint_coords[0] + keypoint_coords[1]) / 2.0
            for idx in range(2):
                centroid = centroids[idx]
                if np.all(np.isfinite(centroid)):
                    symmetry_offsets[idx] = float(np.dot(centroid - midpoint, perp))

    prob_mean = prob_max = prob_var = prob_high = None
    if probs_out is not None:
        prob_mean = np.full(2, np.nan, dtype=np.float32)
        prob_max = np.full(2, np.nan, dtype=np.float32)
        prob_var = np.full(2, np.nan, dtype=np.float32)
        prob_high = np.full(2, np.nan, dtype=np.float32)
        for idx, mask in enumerate((refined_left, refined_right)):
            mask_vals = probs_out[idx][mask]
            if mask_vals.size == 0:
                continue
            prob_mean[idx] = float(mask_vals.mean())
            prob_max[idx] = float(mask_vals.max())
            prob_var[idx] = float(mask_vals.var()) if mask_vals.size > 1 else 0.0
            prob_high[idx] = float(np.count_nonzero(mask_vals >= _PROBABILITY_THRESHOLD) / mask_vals.size)

    return {
        "refined_areas": refined_areas.astype(np.float32),
        "source_areas": source_areas.astype(np.float32),
        "source_union_area": float(union_source_area),
        "refined_union_area": float(union_refined_area),
        "centroid_errors": centroid_errors,
        "symmetry_offsets": symmetry_offsets,
        "keypoint_separation": float(keypoint_separation),
        "separation_delta": float(separation_delta),
        "axis_ratio": axis_ratio,
        "circularity": circularity,
        "probability_mean": prob_mean,
        "probability_max": prob_max,
        "probability_var": prob_var,
        "probability_high_fraction": prob_high,
    }


def _build_passthrough_output(
    source_masks: np.ndarray,
    source_ellipse_params: np.ndarray,
    source_ellipse_success: np.ndarray,
    source_eye_separation: Optional[float],
    keypoints_roi: np.ndarray,
    reason: Optional[str],
) -> ROIOutput:
    """Build a ROIOutput that preserves the source masks/ellipses unchanged."""

    masks = np.asarray(source_masks)
    if masks.ndim == 2:
        masks = masks[None, ...]
    if masks.shape[0] == 1:
        masks = np.repeat(masks, 2, axis=0)

    masks_out = masks.astype(np.uint8, copy=False)
    ellipse_params = np.asarray(source_ellipse_params, dtype=np.float32)
    ellipse_success = np.asarray(source_ellipse_success, dtype=bool)

    if ellipse_params.shape != (2, 5):
        ellipse_params = np.full((2, 5), np.nan, dtype=np.float32)
    if ellipse_success.shape != (2,):
        ellipse_success = np.zeros(2, dtype=bool)

    centroids = np.full((2, 2), np.nan, dtype=np.float32)
    for idx in range(2):
        if ellipse_success[idx] and np.all(np.isfinite(ellipse_params[idx, :2])):
            centroids[idx] = ellipse_params[idx, :2]
        else:
            centroid = _mask_centroid(masks_out[idx])
            if centroid is not None:
                centroids[idx] = centroid

    if source_eye_separation is not None and np.isfinite(source_eye_separation):
        separation = float(source_eye_separation)
    elif np.all(np.isfinite(centroids)):
        separation = float(
            math.hypot(
                float(centroids[0, 0] - centroids[1, 0]),
                float(centroids[0, 1] - centroids[1, 1]),
            )
        )
    else:
        separation = float("nan")

    eye_left = keypoints_roi[1]
    eye_right = keypoints_roi[2]
    keypoints_valid = (
        eye_left.size >= 2
        and eye_right.size >= 2
        and np.all(np.isfinite(eye_left))
        and np.all(np.isfinite(eye_right))
        and not np.allclose(eye_left, eye_right, atol=1e-3)
    )

    reason_val = str(reason) if reason not in (None, "") else ""
    reason_val = _append_reason_tag(reason_val, "copied_original")

    source_union = (masks_out[0] | masks_out[1]).astype(bool, copy=False)
    metrics = _compute_roi_metrics(
        masks_out[0],
        masks_out[1],
        masks_out[0],
        masks_out[1],
        source_union,
        centroids,
        eye_left,
        eye_right,
        keypoints_valid,
        separation,
        ellipse_params,
        (None, None),
        None,
    )

    return ROIOutput(
        masks_out,
        ellipse_params,
        ellipse_success,
        centroids,
        (None, None),
        separation,
        reason_val,
        np.zeros(2, dtype=bool),
        0,
        False,
        None,
        **metrics,
    )

def _refine_roi(
    source_masks: np.ndarray,
    keypoints_roi: np.ndarray,
    heading_deg: float,
    success_flag: bool,
    mask_probs: Optional[np.ndarray] = None
) -> ROIOutput:
    """Refine a single ROI's mask assignment."""

    source_masks = np.asarray(source_masks)
    if source_masks.ndim == 2:
        source_masks = source_masks[None, ...]
    elif source_masks.ndim != 3:
        raise ValueError(f"Unexpected source mask shape {source_masks.shape}")

    channel_count = int(source_masks.shape[0])
    if channel_count <= 0:
        raise ValueError("Source masks contain no channels.")
    if channel_count > 2:
        raise ValueError(
            f"Source masks provide {channel_count} channels; expected 1 (union) or 2 (left/right)."
        )

    union_source = channel_count == 1
    roi_h = int(source_masks.shape[-2])
    roi_w = int(source_masks.shape[-1])
    base_mask = source_masks[0] > 0
    initial_masks = [
        base_mask.copy(),
        (source_masks[1] > 0) if channel_count > 1 else base_mask.copy(),
    ]
    source_union_mask = (initial_masks[0] | initial_masks[1]).astype(bool, copy=False)

    masks_out = np.zeros((2, roi_h, roi_w), dtype=np.uint8)
    ellipse_params = np.full((2, 5), np.nan, dtype=np.float32)
    ellipse_success = np.zeros(2, dtype=bool)
    centroids = np.full((2, 2), np.nan, dtype=np.float32)

    def _append_reason(reason_val: Optional[str], tag: str) -> str:
        return tag if not reason_val else f"{reason_val}|{tag}"

    def _copy_original(reason: str) -> ROIOutput:
        contours: Tuple[Optional[np.ndarray], Optional[np.ndarray]] = (None, None)
        smoothing_flags = np.zeros(2, dtype=bool)
        for eye_idx, mask_bool in enumerate(initial_masks):
            mask_bool = np.array(mask_bool, dtype=bool, copy=False)
            mask_bool, changed = _smooth_binary_mask(mask_bool)
            smoothing_flags[eye_idx] = changed
            masks_out[eye_idx] = mask_bool.astype(np.uint8)
            (
                success,
                ellipse,
                centroid,
                contour,
            ) = _measure_mask(mask_bool.astype(np.uint8))
            ellipse_success[eye_idx] = success
            ellipse_params[eye_idx] = ellipse
            centroids[eye_idx] = centroid
            if contour is not None:
                if eye_idx == 0:
                    contours = (contour, contours[1])
                else:
                    contours = (contours[0], contour)

        if np.all(ellipse_success):
            separation = float(
                math.hypot(
                    float(centroids[0, 0] - centroids[1, 0]),
                    float(centroids[0, 1] - centroids[1, 1]),
                )
            )
        else:
            separation = float("nan")

        eye_left_local = keypoints_roi[1]
        eye_right_local = keypoints_roi[2]
        keypoints_valid_local = (
            eye_left_local.size >= 2
            and eye_right_local.size >= 2
            and np.all(np.isfinite(eye_left_local))
            and np.all(np.isfinite(eye_right_local))
            and not np.allclose(eye_left_local, eye_right_local, atol=1e-3)
        )

        metrics = _compute_roi_metrics(
            masks_out[0].astype(bool, copy=False),
            masks_out[1].astype(bool, copy=False),
            initial_masks[0].astype(bool, copy=False),
            initial_masks[1].astype(bool, copy=False),
            source_union_mask,
            centroids,
            eye_left_local,
            eye_right_local,
            keypoints_valid_local,
            separation,
            ellipse_params,
            contours,
            None,
        )

        final_reason = reason
        if union_source:
            existing = set((final_reason or "").split("|")) if final_reason else set()
            if "union_source" not in existing:
                final_reason = _append_reason(final_reason, "union_source")

        return ROIOutput(
            masks_out,
            ellipse_params,
            ellipse_success,
            centroids,
            contours,
            separation,
            final_reason,
            smoothing_flags,
            0,
            False,
            None,
            **metrics,
        )

    if not success_flag:
        base_reason = "keypoint_fail"
        if union_source:
            base_reason = _append_reason(base_reason, "union_source")
        return _copy_original(base_reason)

    reason: Optional[str] = None
    if union_source:
        reason = _append_reason(reason, "union_source")
    eye_left = keypoints_roi[1]
    eye_right = keypoints_roi[2]

    def _valid_keypoint(point: np.ndarray) -> bool:
        arr = np.asarray(point)
        return arr.size >= 2 and np.all(np.isfinite(arr))

    left_instance_idx = 0
    right_instance_idx = 1
    keypoints_valid = _valid_keypoint(eye_left) and _valid_keypoint(eye_right) and not np.allclose(
        eye_left,
        eye_right,
        atol=1e-3,
    )

    if union_source:
        union_mask = base_mask.astype(bool, copy=False)
        fallback_left_source = union_mask
        fallback_right_source = union_mask

        if keypoints_valid and union_mask.any():
            split_left, split_right = _split_mask_by_keypoints(union_mask, eye_left, eye_right)
            if split_left.sum() > 0 and split_right.sum() > 0:
                left_mask = split_left
                right_mask = split_right
                reason = _append_reason(reason, "split_by_keypoint")
            else:
                left_mask = union_mask.copy()
                right_mask = union_mask.copy()
        else:
            left_mask = union_mask.copy()
            right_mask = union_mask.copy()

        if (left_mask.sum() == 0 or right_mask.sum() == 0) and np.isfinite(heading_deg):
            midpoint = np.array(
                [
                    float(eye_left[0] + eye_right[0]) / 2.0,
                    float(eye_left[1] + eye_right[1]) / 2.0,
                ],
                dtype=np.float32,
            )
            split_left, split_right = _split_mask_by_heading(union_mask, heading_deg, midpoint)
            if split_left.sum() > 0 and split_right.sum() > 0:
                left_mask = split_left
                right_mask = split_right
                reason = _append_reason(reason, "heading_split")
    else:
        instance_masks = [
            source_masks[0] > 0,
            source_masks[1] > 0,
        ]
        if keypoints_valid and instance_masks[0].any() and instance_masks[1].any():
            centroid_0 = _mask_centroid(instance_masks[0])
            centroid_1 = _mask_centroid(instance_masks[1])
            if centroid_0 is not None and centroid_1 is not None:
                cost_direct = math.hypot(
                    float(centroid_0[0] - eye_left[0]),
                    float(centroid_0[1] - eye_left[1]),
                ) + math.hypot(
                    float(centroid_1[0] - eye_right[0]),
                    float(centroid_1[1] - eye_right[1]),
                )
                cost_reversed = math.hypot(
                    float(centroid_0[0] - eye_right[0]),
                    float(centroid_0[1] - eye_right[1]),
                ) + math.hypot(
                    float(centroid_1[0] - eye_left[0]),
                    float(centroid_1[1] - eye_left[1]),
                )
                if cost_reversed + 1e-3 < cost_direct:
                    left_instance_idx = 1
                    right_instance_idx = 0
                    reason = _append_reason(reason, "assigned_by_keypoint")
        fallback_left_source = instance_masks[left_instance_idx]
        fallback_right_source = instance_masks[right_instance_idx]
        left_mask = fallback_left_source.copy()
        right_mask = fallback_right_source.copy()

    if union_source:
        fallback_left_source = base_mask
        fallback_right_source = base_mask

    used_probabilities = mask_probs is not None
    probs_arr: Optional[np.ndarray] = None
    if mask_probs is not None:
        probs_arr = np.asarray(mask_probs, dtype=np.float32)
        probs_arr = np.nan_to_num(probs_arr, nan=0.0, posinf=1.0, neginf=0.0)
        probs_arr = np.clip(probs_arr, 0.0, 1.0)
        if probs_arr.ndim == 2:
            probs_arr = probs_arr[None, ...]
        if probs_arr.shape[0] >= 2:
            left_prob = probs_arr[left_instance_idx]
            right_prob = probs_arr[right_instance_idx]
            left_mask = left_mask | (left_prob >= _PROBABILITY_THRESHOLD)
            right_mask = right_mask | (right_prob >= _PROBABILITY_THRESHOLD)
            used_probabilities = True

    if left_mask.sum() == 0 and right_mask.sum() == 0:
        return _copy_original("empty_union")
    if left_mask.sum() == 0:
        left_mask = fallback_left_source.copy()
        reason = _append_reason(reason, "left_empty")
    if right_mask.sum() == 0:
        right_mask = fallback_right_source.copy()
        reason = _append_reason(reason, "right_empty")

    left_mask_sm, left_changed = _smooth_binary_mask(left_mask)
    right_mask_sm, right_changed = _smooth_binary_mask(right_mask)

    left_mask_connected, right_mask_connected, component_flags, reassigned_pixels = _enforce_single_component(
        left_mask_sm,
        right_mask_sm,
        eye_left,
        eye_right,
    )

    def _ncc(mask: np.ndarray) -> int:
        return int(measure.label(mask.astype(bool), connectivity=2).max())

    smoothing_flags = np.array([left_changed, right_changed], dtype=bool) | component_flags

    masks_out[0] = left_mask_connected.astype(np.uint8)
    masks_out[1] = right_mask_connected.astype(np.uint8)

    contours: List[Optional[np.ndarray]] = [None, None]
    for eye_idx, mask in enumerate((left_mask_connected, right_mask_connected)):
        (
            success,
            ellipse,
            centroid,
            contour,
        ) = _measure_mask(mask.astype(np.uint8))

        ellipse_success[eye_idx] = success
        ellipse_params[eye_idx] = ellipse
        centroids[eye_idx] = centroid
        contours[eye_idx] = contour

    if np.all(ellipse_success):
        eye_separation = float(
            math.hypot(
                float(centroids[0, 0] - centroids[1, 0]),
                float(centroids[0, 1] - centroids[1, 1]),
            )
        )
    else:
        eye_separation = float("nan")

    probs_out: Optional[np.ndarray] = None
    if probs_arr is not None:
        if probs_arr.shape[0] >= 2:
            left_prob = probs_arr[left_instance_idx]
            right_prob = probs_arr[right_instance_idx]
        else:
            left_prob = right_prob = probs_arr[0]

        union_prob = np.maximum(left_prob, right_prob)
        probs_out = np.zeros((2, left_mask_connected.shape[0], left_mask_connected.shape[1]), dtype=np.float32)
        probs_out[0] = np.where(left_mask_connected, union_prob, 0.0)
        probs_out[1] = np.where(right_mask_connected, union_prob, 0.0)

    metrics = _compute_roi_metrics(
        left_mask_connected,
        right_mask_connected,
        fallback_left_source.astype(bool, copy=False),
        fallback_right_source.astype(bool, copy=False),
        source_union_mask,
        centroids,
        eye_left,
        eye_right,
        keypoints_valid,
        eye_separation,
        ellipse_params,
        (contours[0], contours[1]),
        probs_out,
    )

    return ROIOutput(
        masks_out,
        ellipse_params,
        ellipse_success,
        centroids,
        (contours[0], contours[1]),
        eye_separation,
        reason,
        smoothing_flags,
        int(reassigned_pixels),
        used_probabilities,
        probs_out,
        **metrics,
    )


def _process_refine_chunk(
    binary_chunk: np.ndarray,
    probs_chunk: Optional[np.ndarray],
    zarr_path: str,
    keypoints_path: str,
    heading_path: str,
    success_path: str,
    start: int,
    stop: int,
    *,
    fast_path: bool = False,
    source_ellipse_params_path: Optional[str] = None,
    source_ellipse_success_path: Optional[str] = None,
    source_eye_sep_path: Optional[str] = None,
    source_reason_path: Optional[str] = None,
) -> List[Tuple[int, ROIOutput]]:
    """Process a batch of ROI indices and return their refinement outputs."""
    slice_obj = slice(start, stop)

    masks_np = np.array(binary_chunk, dtype=np.uint8, copy=False)
    mask_probs_np = np.array(probs_chunk, dtype=np.float32, copy=False) if probs_chunk is not None else None

    keypoints_np = np.asarray(_get_zarr_array(zarr_path, keypoints_path)[slice_obj])
    heading_np = np.asarray(_get_zarr_array(zarr_path, heading_path)[slice_obj])
    success_np = np.asarray(_get_zarr_array(zarr_path, success_path)[slice_obj])

    if fast_path:
        if not source_ellipse_params_path or not source_ellipse_success_path:
            raise ValueError("Fast path requires source ellipse parameters.")
        ellipse_params_np = np.asarray(_get_zarr_array(zarr_path, source_ellipse_params_path)[slice_obj])
        ellipse_success_np = np.asarray(_get_zarr_array(zarr_path, source_ellipse_success_path)[slice_obj])
        eye_sep_np = (
            np.asarray(_get_zarr_array(zarr_path, source_eye_sep_path)[slice_obj])
            if source_eye_sep_path
            else None
        )
        reason_np = (
            np.asarray(_get_zarr_array(zarr_path, source_reason_path)[slice_obj])
            if source_reason_path
            else None
        )

    results: List[Tuple[int, ROIOutput]] = []
    for local_idx in range(masks_np.shape[0]):
        global_idx = start + local_idx
        source_masks = masks_np[local_idx]
        keypoints_roi = keypoints_np[local_idx]
        if fast_path:
            reason_val = reason_np[local_idx] if reason_np is not None else None
            eye_sep_val = float(eye_sep_np[local_idx]) if eye_sep_np is not None else None
            results.append(
                (
                    global_idx,
                    _build_passthrough_output(
                        source_masks,
                        ellipse_params_np[local_idx],
                        ellipse_success_np[local_idx],
                        eye_sep_val,
                        keypoints_roi,
                        reason_val,
                    ),
                )
            )
        else:
            heading = float(heading_np[local_idx])
            success_flag = bool(success_np[local_idx])
            mask_probs_roi = mask_probs_np[local_idx] if mask_probs_np is not None else None
            results.append(
                (
                    global_idx,
                    _refine_roi(
                        source_masks,
                        keypoints_roi,
                        heading,
                        success_flag,
                        mask_probs=mask_probs_roi,
                    ),
                )
            )
    return results


def _process_and_write_chunk(
    binary_chunk: np.ndarray,
    probs_chunk: Optional[np.ndarray],
    zarr_path: str,
    run_group_path: str,
    keypoints_path: str,
    heading_path: str,
    success_path: str,
    start: int,
    stop: int,
    *,
    write_probabilities: bool,
    fast_path: bool = False,
    source_ellipse_params_path: Optional[str] = None,
    source_ellipse_success_path: Optional[str] = None,
    source_eye_sep_path: Optional[str] = None,
    source_reason_path: Optional[str] = None,
) -> List[Tuple[int, ROIOutput]]:
    """Process a chunk, write outputs into Zarr arrays, and return ROI results."""
    results = _process_refine_chunk(
        binary_chunk,
        probs_chunk,
        zarr_path,
        keypoints_path,
        heading_path,
        success_path,
        start,
        stop,
        fast_path=fast_path,
        source_ellipse_params_path=source_ellipse_params_path,
        source_ellipse_success_path=source_ellipse_success_path,
        source_eye_sep_path=source_eye_sep_path,
        source_reason_path=source_reason_path,
    )
    if not results:
        return results

    _, first_output = results[0]
    _, roi_h, roi_w = first_output.masks.shape
    chunk_len = stop - start

    masks_chunk = np.stack([roi_output.masks for _, roi_output in results], axis=0).astype(np.uint8, copy=False)
    ellipse_params_chunk = np.stack([roi_output.ellipse_params for _, roi_output in results], axis=0)
    ellipse_success_chunk = np.stack([roi_output.ellipse_success for _, roi_output in results], axis=0)
    eye_separation_chunk = np.array([roi_output.eye_separation for _, roi_output in results], dtype=np.float32)

    if write_probabilities:
        probs_chunk_out = np.zeros((chunk_len, 2, roi_h, roi_w), dtype=np.float32)
        for idx, (_, roi_output) in enumerate(results):
            if roi_output.probabilities is not None:
                probs_chunk_out[idx] = roi_output.probabilities
    else:
        probs_chunk_out = None

    root = zarr.open(zarr_path, mode="a", use_consolidated=False)
    run_group = root[run_group_path]

    run_group["masks_roi"][start:stop] = masks_chunk
    run_group["ellipse_params"][start:stop] = ellipse_params_chunk.astype(np.float32, copy=False)
    run_group["ellipse_success"][start:stop] = ellipse_success_chunk.astype(bool, copy=False)
    run_group["eye_separation"][start:stop] = eye_separation_chunk

    if write_probabilities and probs_chunk_out is not None:
        run_group["mask_probs_roi_refined"][start:stop] = probs_chunk_out.astype(np.float16)

    return results


def refine_eye_masks(
    zarr_path: str,
    source_run: Optional[str] = None,
    run_name: Optional[str] = None,
    *,
    keypoint_run: Optional[str] = None,
    chunk_size: int = 512,
    scheduler: str = "processes",
    num_workers: Optional[int] = None,
    console: Optional[Console] = None,
    command: Optional[str] = None,
    created_at_utc: Optional[str] = None,
    area_filter_z: Optional[float] = _AREA_FILTER_Z_DEFAULT,
    area_filter_mode: str = _AREA_FILTER_MODE_DEFAULT,
    success_min_eye_area_px: Optional[float] = _SUCCESS_MIN_EYE_AREA_PX_DEFAULT,
    force_refine_traditional: bool = False,
    allow_latest_keypoint_fallback: bool = False,
) -> str:
    """Refine an eye-mask run and return the name of the new run."""

    console = console or Console()
    stage_start = time.perf_counter()

    area_filter_mode = (area_filter_mode or _AREA_FILTER_MODE_DEFAULT).lower()
    if area_filter_mode not in {"either", "both"}:
        area_filter_mode = _AREA_FILTER_MODE_DEFAULT
    area_filter_z = _coerce_positive_float(area_filter_z)
    success_min_eye_area_px = _coerce_positive_float(success_min_eye_area_px)

    chunk_size = max(1, int(chunk_size))

    zarr_path = str(Path(zarr_path))
    console.print(f"Opening Zarr archive: [cyan]{zarr_path}[/cyan]")

    root = zarr.open(zarr_path, mode="a", use_consolidated=False)

    if "eye_masks_runs" not in root:
        raise ValueError("Zarr archive missing eye_masks_runs; run segmentation first.")
    eye_parent = root["eye_masks_runs"]
    src_run_name = source_run or eye_parent.attrs.get("latest")
    if src_run_name is None or src_run_name not in eye_parent:
        raise ValueError("Source eye mask run not found.")
    src_run = eye_parent[src_run_name]
    source_method = str(src_run.attrs.get("method", "unknown"))
    console.print(
        f"Refining eye masks from [cyan]eye_masks_runs/{src_run_name}[/cyan] "
        f"(method={source_method})"
    )

    crop_run_name = src_run.attrs.get("source_crop_run") or root.get("crop_runs", {}).attrs.get("latest")
    if crop_run_name is None:
        raise ValueError("Unable to determine crop run (missing attribute 'source_crop_run').")

    kp_group, keypoint_run_name, keypoint_group_name = _resolve_keypoint_group(
        root,
        keypoint_run,
        src_run.attrs.get("source_keypoint_group"),
        resolve_source_keypoints_run(src_run.attrs),
        allow_latest_fallback=allow_latest_keypoint_fallback,
    )

    success_dataset_name = None
    for candidate in ("detection_success", "refined_success", "source_success"):
        if candidate in kp_group:
            success_dataset_name = candidate
            break
    if success_dataset_name is None:
        raise ValueError(f"Keypoint run '{keypoint_run_name}' missing success flags.")

    required_kp = ["keypoints_roi", "heading", success_dataset_name]
    for arr in required_kp:
        if arr not in kp_group:
            raise ValueError(f"Keypoint run '{keypoint_run_name}' missing '{arr}'.")
    kp_paths = ", ".join(f"{keypoint_group_name}/{keypoint_run_name}/{name}" for name in required_kp)
    console.print(f"Using keypoint datasets: [cyan]{kp_paths}[/cyan]")

    console.print(
        "Loading mask bundle (prefer_probs=True, threshold="
        f"{_PROBABILITY_THRESHOLD})..."
    )
    load_start = time.perf_counter()
    bundle = load_mask_bundle(
        src_run,
        threshold=_PROBABILITY_THRESHOLD,
        prefer_probs=True,
        materialize=False,
        lazy=True,
    )
    load_duration = time.perf_counter() - load_start
    binary_data = bundle.binary
    probs_data = bundle.probs

    provenance_sources = bundle.provenance.get("source", [])
    dataset_paths: List[str] = []
    for name in provenance_sources:
        if name in {"masks_roi", "mask_probs_roi", "mask_probs_roi_refined"}:
            dataset_paths.append(f"eye_masks_runs/{src_run_name}/{name}")
        else:
            dataset_paths.append(str(name))
    if not dataset_paths:
        dataset_paths.append("<unknown>")
    console.print(
        "Loaded mask bundle "
        f"(binary_identity={bundle.binary_identity}, probs_identity={bundle.probs_identity or 'none'}) "
        f"from: [cyan]{', '.join(dataset_paths)}[/cyan] "
        f"[dim](took {load_duration:.2f}s)[/dim]"
    )

    total_rois, source_channels, roi_h, roi_w = binary_data.shape
    total_rois = int(total_rois)
    source_channels = int(source_channels)
    roi_h = int(roi_h)
    roi_w = int(roi_w)

    source_method_lower = source_method.lower()
    source_has_ellipses = all(
        name in src_run for name in ("ellipse_params", "ellipse_success", "eye_separation")
    )
    use_fast_path = (
        "traditional" in source_method_lower
        and not force_refine_traditional
        and source_channels == 2
        and source_has_ellipses
    )
    if use_fast_path:
        console.print(
            "[yellow]Traditional fast path enabled:[/yellow] preserving source masks/ellipses and skipping "
            "smoothing/component enforcement. Use --force-refine-traditional to opt in to refinement."
        )

    mask_bundle_provenance = dict(bundle.provenance)
    mask_binary_source = None
    for name in provenance_sources:
        if name in {"masks_roi", "synth_binary_from_probs"}:
            mask_binary_source = name
            break
    if mask_binary_source is None and provenance_sources:
        mask_binary_source = provenance_sources[0]

    if isinstance(binary_data, np.ndarray):
        binary_data = da.from_array(
            binary_data,
            chunks=(chunk_size, source_channels, roi_h, roi_w),
        )
    else:
        binary_data = binary_data.rechunk({0: chunk_size})

    if probs_data is not None:
        if isinstance(probs_data, np.ndarray):
            probs_data = da.from_array(
                probs_data,
                chunks=(chunk_size, probs_data.shape[1], roi_h, roi_w),
            )
        else:
            probs_data = probs_data.rechunk({0: chunk_size})

    provenance_sources = bundle.provenance.get("source", [])
    if "mask_probs_roi_refined" in provenance_sources:
        mask_prob_dataset = "mask_probs_roi_refined"
    elif "mask_probs_roi" in provenance_sources:
        mask_prob_dataset = "mask_probs_roi"
    else:
        mask_prob_dataset = None

    if "masks_roi" in provenance_sources:
        binary_reference = src_run["masks_roi"]
    elif mask_prob_dataset:
        binary_reference = src_run[mask_prob_dataset]
    else:
        binary_reference = src_run["masks_roi"] if "masks_roi" in src_run else None

    if mask_prob_dataset:
        console.print(
            f"Probability source: [cyan]eye_masks_runs/{src_run_name}/{mask_prob_dataset}[/cyan] "
            f"(threshold={_PROBABILITY_THRESHOLD})"
        )
    else:
        console.print("Probability source: [yellow]none[/yellow]; using binary masks only.")

    keypoints_path = f"{keypoint_group_name}/{keypoint_run_name}/keypoints_roi"
    heading_path = f"{keypoint_group_name}/{keypoint_run_name}/heading"
    success_path = f"{keypoint_group_name}/{keypoint_run_name}/{success_dataset_name}"

    source_ellipse_params_path = (
        f"eye_masks_runs/{src_run_name}/ellipse_params" if "ellipse_params" in src_run else None
    )
    source_ellipse_success_path = (
        f"eye_masks_runs/{src_run_name}/ellipse_success" if "ellipse_success" in src_run else None
    )
    source_eye_sep_path = (
        f"eye_masks_runs/{src_run_name}/eye_separation" if "eye_separation" in src_run else None
    )
    source_reason_path = f"eye_masks_runs/{src_run_name}/reason" if "reason" in src_run else None

    has_mask_probs = probs_data is not None and not use_fast_path

    _validate_input_row_alignment(
        total_rois=total_rois,
        src_run=src_run,
        src_run_name=src_run_name,
        kp_group=kp_group,
        keypoint_group_name=keypoint_group_name,
        keypoint_run_name=keypoint_run_name,
        success_dataset_name=success_dataset_name,
        binary_data=binary_data,
        probs_data=probs_data,
    )

    run_group, resolved_run_name = _prepare_run_group(root, run_name, console)

    if "detection_source" in src_run:
        src_detection_source = src_run["detection_source"][:].astype(np.int8, copy=False)
        console.print("[dim]Copied detection_source from source run[/dim]")
    else:
        src_detection_source = np.zeros(total_rois, dtype=np.int8)
        console.print("[yellow]Source run missing detection_source; defaulting to zeros.[/yellow]")
    run_group.create_array(
        "detection_source",
        data=src_detection_source,
        chunks=(min(1024, total_rois),),
        overwrite=True,
    )

    left_ptr = np.full((total_rois,), -1, dtype=np.int64)
    left_len = np.zeros((total_rois,), dtype=np.int32)
    right_ptr = np.full((total_rois,), -1, dtype=np.int64)
    right_len = np.zeros((total_rois,), dtype=np.int32)

    left_points: List[np.ndarray] = []
    right_points: List[np.ndarray] = []
    left_total = 0
    right_total = 0

    stats = {
        "total": total_rois,
        "refined": 0,
        "fallback_heading": 0,
        "copied_original": 0,
        "keypoint_fail": 0,
        "empty_union": 0,
        "smoothed_rois": 0,
        "smoothed_channels": 0,
        "components_reassigned": 0,
        "chunk_size": chunk_size,
        "probability_split": 0,
        "assigned_by_keypoint": 0,
        "mask_sources": list(provenance_sources),
        "probability_source": mask_prob_dataset or "none",
        "filtered_rois": 0,
        "filtered_left": 0,
        "filtered_right": 0,
        "small_area_rois": 0,
        "small_area_left": 0,
        "small_area_right": 0,
    }

    chunk_rois = min(512, total_rois) if total_rois > 0 else 1

    def _prepare_dataset(
        name: str,
        *,
        shape: Tuple[int, ...],
        chunks: Tuple[int, ...],
        dtype: np.dtype,
        fill_value: float | int | bool = 0,
        **kwargs: object,
    ) -> zarr.Array:
        if name in run_group:
            del run_group[name]
        return run_group.create_array(
            name,
            shape=shape,
            chunks=chunks,
            dtype=dtype,
            fill_value=fill_value,
            overwrite=True,
            **kwargs,
        )

    _prepare_dataset(
        "masks_roi",
        shape=(total_rois, 2, roi_h, roi_w),
        chunks=(chunk_rois, 2, roi_h, roi_w),
        dtype=np.dtype(np.uint8),
    )
    _prepare_dataset(
        "ellipse_params",
        shape=(total_rois, 2, 5),
        chunks=(chunk_rois, 2, 5),
        dtype=np.dtype(np.float32),
        fill_value=np.float32(np.nan),
    )
    _prepare_dataset(
        "ellipse_success",
        shape=(total_rois, 2),
        chunks=(chunk_rois, 2),
        dtype=np.dtype(bool),
        fill_value=False,
    )
    _prepare_dataset(
        "eye_separation",
        shape=(total_rois,),
        chunks=(chunk_rois,),
        dtype=np.dtype(np.float32),
        fill_value=np.float32(np.nan),
    )

    def _copy_from_source(
        array_name: str,
        chunk_fallback: Tuple[int, ...],
    ) -> None:
        """Copy bookkeeping arrays from the source run if they exist."""
        if array_name not in src_run:
            console.print(
                f"[yellow]Source eye-mask run missing '{array_name}'; refined run will too.[/yellow]"
            )
            return
        src_arr = src_run[array_name]
        src_data = src_arr[:]
        chunks = getattr(src_arr, "chunks", None)
        if not chunks:
            chunks = chunk_fallback
        elif isinstance(chunks, int):
            chunks = (chunks,)
        chunk_dims: List[int] = []
        for axis, dim in enumerate(src_data.shape):
            source_chunk = chunks[axis] if axis < len(chunks) else chunks[-1]
            chunk_dims.append(int(max(1, min(dim, int(source_chunk)))))
        chunks = tuple(chunk_dims)
        target = _prepare_dataset(
            array_name,
            shape=src_data.shape,
            chunks=chunks,
            dtype=src_data.dtype,
        )
        target[:] = src_data
        console.print(f"[dim]Copied {array_name} from source run[/dim]")

    # Preserve frame/detection mappings so downstream tooling can align ROIs.
    _copy_from_source("frame_indices", (chunk_rois,))
    _copy_from_source("detection_indices", (chunk_rois,))
    _copy_from_source("frame_counts", (max(1, chunk_rois),))

    run_group_path = f"{REFINED_STAGE_NAME}_runs/{resolved_run_name}"

    if has_mask_probs:
        compression_kwargs = (
            _compression_kwargs(binary_reference) if binary_reference is not None else {}
        )
        chunk_spec = getattr(binary_reference, "chunks", None) if binary_reference is not None else None
        if not chunk_spec:
            chunk_spec = (chunk_rois, 2, roi_h, roi_w)
        compression_source = None
        if binary_reference is not None:
            compression_source = getattr(binary_reference, "path", None)
            if not compression_source:
                compression_source = f"eye_masks_runs/{src_run_name}/{mask_prob_dataset or 'masks_roi'}"
        console.print(
            "Prepared refined probability dataset "
            f"(compression source: [cyan]{compression_source or 'default'}[/cyan])"
        )
        _prepare_dataset(
            "mask_probs_roi_refined",
            shape=(total_rois, 2, roi_h, roi_w),
            chunks=chunk_spec,
            dtype=np.dtype(np.float16),
            fill_value=np.float16(0.0),
            **compression_kwargs,
        )

    scheduler_key = (scheduler or "processes").lower()
    if scheduler_key not in {"threads", "processes"}:
        console.print(f"[yellow]Unknown scheduler '{scheduler_key}', defaulting to 'threads'.[/yellow]")
        scheduler_key = "threads"

    compute_kwargs: Dict[str, object] = {"scheduler": scheduler_key}
    if num_workers is not None:
        compute_kwargs["num_workers"] = int(num_workers)

    if isinstance(binary_data, da.Array):
        binary_data = binary_data.rechunk({0: chunk_size, 1: -1, 2: -1, 3: -1})
    if isinstance(probs_data, da.Array):
        probs_data = probs_data.rechunk({0: chunk_size, 1: -1, 2: -1, 3: -1})

    tasks: List[object] = []
    if total_rois > 0:
        binary_chunks = binary_data.to_delayed().reshape((-1,)).tolist()
        if has_mask_probs and probs_data is not None:
            probs_chunks = probs_data.to_delayed().reshape((-1,)).tolist()
        else:
            probs_chunks = [None] * len(binary_chunks)

        chunk_offsets = list(binary_data.chunks[0])
        if len(chunk_offsets) != len(binary_chunks):
            raise RuntimeError(
                f"Unexpected chunk alignment (offsets={len(chunk_offsets)}, chunks={len(binary_chunks)})"
            )
        cumulative = 0
        for idx, (binary_chunk, probs_chunk) in enumerate(zip(binary_chunks, probs_chunks)):
            chunk_len = int(chunk_offsets[idx])
            start = cumulative
            stop = start + chunk_len
            tasks.append(
                delayed(_process_and_write_chunk)(
                    binary_chunk,
                    probs_chunk,
                    zarr_path,
                    run_group_path,
                    keypoints_path,
                    heading_path,
                    success_path,
                    start,
                    stop,
                    write_probabilities=has_mask_probs,
                    fast_path=use_fast_path,
                    source_ellipse_params_path=source_ellipse_params_path,
                    source_ellipse_success_path=source_ellipse_success_path,
                    source_eye_sep_path=source_eye_sep_path,
                    source_reason_path=source_reason_path,
                )
            )
            cumulative = stop

    chunk_results: List[List[Tuple[int, ROIOutput]]] = []
    if tasks:
        console.print(f"[cyan]Submitting {len(tasks)} chunk(s) to Dask ({scheduler_key})[/cyan]")
        with ProgressBar():
            chunk_results = list(dask.compute(*tasks, **compute_kwargs))

    del binary_data
    if probs_data is not None:
        del probs_data

    gathered_results: List[Tuple[int, ROIOutput]] = []
    for chunk in chunk_results:
        gathered_results.extend(chunk)

    gathered_results.sort(key=lambda item: item[0])

    wrote_any_probs = False

    refined_area_metrics = np.zeros((total_rois, 2), dtype=np.float32)
    source_area_metrics = np.zeros((total_rois, 2), dtype=np.float32)
    ellipse_success_metrics = np.zeros((total_rois, 2), dtype=bool)
    union_refined_metrics = np.zeros(total_rois, dtype=np.float32)
    union_source_metrics = np.zeros(total_rois, dtype=np.float32)
    centroid_error_metrics = np.full((total_rois, 2), np.nan, dtype=np.float32)
    symmetry_offset_metrics = np.full((total_rois, 2), np.nan, dtype=np.float32)
    separation_refined_metrics = np.full(total_rois, np.nan, dtype=np.float32)
    separation_keypoint_metrics = np.full(total_rois, np.nan, dtype=np.float32)
    separation_delta_metrics = np.full(total_rois, np.nan, dtype=np.float32)
    axis_ratio_metrics = np.full((total_rois, 2), np.nan, dtype=np.float32)
    circularity_metrics = np.full((total_rois, 2), np.nan, dtype=np.float32)
    prob_mean_metrics = np.full((total_rois, 2), np.nan, dtype=np.float32)
    prob_max_metrics = np.full((total_rois, 2), np.nan, dtype=np.float32)
    prob_var_metrics = np.full((total_rois, 2), np.nan, dtype=np.float32)
    prob_high_metrics = np.full((total_rois, 2), np.nan, dtype=np.float32)
    connectivity_flags_metrics = np.zeros(total_rois, dtype=np.uint8)
    smoothing_flags_metrics = np.zeros((total_rois, 2), dtype=np.uint8)
    pixels_reassigned_metrics = np.zeros(total_rois, dtype=np.int32)
    probabilities_used_metrics = np.zeros(total_rois, dtype=bool)
    reason_strings: List[str] = [""] * total_rois

    for global_idx, result in gathered_results:
        if result.probabilities is not None:
            wrote_any_probs = True
        if result.smoothing_changed.any():
            stats["smoothed_rois"] += 1
            stats["smoothed_channels"] += int(result.smoothing_changed.sum())

        if result.reassigned_pixels:
            stats["components_reassigned"] += int(result.reassigned_pixels)
        if result.used_probabilities:
            stats["probability_split"] += 1

        refined_area_metrics[global_idx] = result.refined_areas
        source_area_metrics[global_idx] = result.source_areas
        ellipse_success_metrics[global_idx] = result.ellipse_success.astype(bool, copy=False)
        union_refined_metrics[global_idx] = result.refined_union_area
        union_source_metrics[global_idx] = result.source_union_area
        centroid_error_metrics[global_idx] = result.centroid_errors
        symmetry_offset_metrics[global_idx] = result.symmetry_offsets
        separation_refined_metrics[global_idx] = result.eye_separation
        separation_keypoint_metrics[global_idx] = result.keypoint_separation
        separation_delta_metrics[global_idx] = result.separation_delta
        axis_ratio_metrics[global_idx] = result.axis_ratio
        circularity_metrics[global_idx] = result.circularity
        if result.probability_mean is not None:
            prob_mean_metrics[global_idx] = result.probability_mean
            prob_max_metrics[global_idx] = result.probability_max
            prob_var_metrics[global_idx] = result.probability_var
            prob_high_metrics[global_idx] = result.probability_high_fraction
        connectivity_flag = 0
        if result.smoothing_changed[0]:
            connectivity_flag |= 1
        if result.smoothing_changed[1]:
            connectivity_flag |= 2
        if result.reassigned_pixels:
            connectivity_flag |= 4
        if result.used_probabilities:
            connectivity_flag |= 8
        connectivity_flags_metrics[global_idx] = connectivity_flag
        smoothing_flags_metrics[global_idx] = result.smoothing_changed.astype(np.uint8)
        pixels_reassigned_metrics[global_idx] = int(result.reassigned_pixels)
        probabilities_used_metrics[global_idx] = bool(result.used_probabilities)

        if result.contours[0] is not None:
            contour = result.contours[0]
            contour_len = contour.shape[0]
            left_ptr[global_idx] = left_total
            left_len[global_idx] = contour_len
            left_points.append(contour)
            left_total += contour_len
        if result.contours[1] is not None:
            contour = result.contours[1]
            contour_len = contour.shape[0]
            right_ptr[global_idx] = right_total
            right_len[global_idx] = contour_len
            right_points.append(contour)
            right_total += contour_len

        reason = result.reason or "refined"
        reason_strings[global_idx] = reason
        reason_tags = set(reason.split("|")) if reason else set()
        if "assigned_by_keypoint" in reason_tags:
            stats["assigned_by_keypoint"] += 1
        if "heading_split" in reason_tags:
            stats["fallback_heading"] += 1
            stats["refined"] += 1
        elif "keypoint_fail" in reason_tags:
            stats["keypoint_fail"] += 1
            stats["copied_original"] += 1
        elif "empty_union" in reason_tags:
            stats["empty_union"] += 1
            stats["copied_original"] += 1
        elif "copied_original" in reason_tags:
            stats["copied_original"] += 1
        else:
            stats["refined"] += 1

    area_ratio_lr = np.divide(
        refined_area_metrics[:, 0],
        np.maximum(refined_area_metrics[:, 1], 1e-6),
        dtype=np.float32,
    )
    area_diff_lr = refined_area_metrics[:, 0] - refined_area_metrics[:, 1]
    area_delta_vs_source = refined_area_metrics - source_area_metrics
    area_ratio_vs_source = refined_area_metrics / np.maximum(source_area_metrics, 1e-6)
    union_delta = union_refined_metrics - union_source_metrics
    union_ratio = union_refined_metrics / np.maximum(union_source_metrics, 1e-6)

    area_mean = np.full(2, np.nan, dtype=np.float32)
    area_std = np.full(2, np.nan, dtype=np.float32)
    area_zscores = np.full_like(refined_area_metrics, np.nan)
    for ch in range(2):
        vals = refined_area_metrics[:, ch]
        valid = vals > 0
        if np.any(valid):
            mean_val = float(vals[valid].mean())
            std_val = float(vals[valid].std())
            area_mean[ch] = mean_val
            area_std[ch] = std_val
            if std_val > 1e-6:
                area_zscores[:, ch] = (vals - mean_val) / std_val

    filter_flags = np.zeros((total_rois, 2), dtype=bool)
    pair_filter_flags = np.zeros(total_rois, dtype=bool)
    if area_filter_z is not None and total_rois > 0:
        threshold = area_filter_z
        filter_flags = np.abs(area_zscores) > threshold
        if area_filter_mode == "both":
            pair_filter_flags = np.all(filter_flags, axis=1)
        else:
            pair_filter_flags = np.any(filter_flags, axis=1)

        filtered_left_indices = np.nonzero(filter_flags[:, 0])[0]
        filtered_right_indices = np.nonzero(filter_flags[:, 1])[0]
        for idx in filtered_left_indices:
            reason_strings[idx] = _append_reason_tag(reason_strings[idx], "filtered_left")
        for idx in filtered_right_indices:
            reason_strings[idx] = _append_reason_tag(reason_strings[idx], "filtered_right")
        for idx in np.nonzero(pair_filter_flags)[0]:
            reason_strings[idx] = _append_reason_tag(reason_strings[idx], "filtered_pair")

        stats["filtered_left"] = int(filter_flags[:, 0].sum())
        stats["filtered_right"] = int(filter_flags[:, 1].sum())
        stats["filtered_rois"] = int(pair_filter_flags.sum())
    else:
        stats["filtered_left"] = 0
        stats["filtered_right"] = 0
        stats["filtered_rois"] = 0

    small_area_flags = np.zeros((total_rois, 2), dtype=bool)
    small_area_pair_flags = np.zeros(total_rois, dtype=bool)
    if success_min_eye_area_px is not None and total_rois > 0:
        small_area_flags = refined_area_metrics < float(success_min_eye_area_px)
        small_area_pair_flags = np.any(small_area_flags, axis=1)

        for idx in np.nonzero(small_area_flags[:, 0])[0]:
            reason_strings[idx] = _append_reason_tag(reason_strings[idx], "small_area_left")
        for idx in np.nonzero(small_area_flags[:, 1])[0]:
            reason_strings[idx] = _append_reason_tag(reason_strings[idx], "small_area_right")
        for idx in np.nonzero(small_area_pair_flags)[0]:
            reason_strings[idx] = _append_reason_tag(reason_strings[idx], "small_area_pair")

        stats["small_area_left"] = int(small_area_flags[:, 0].sum())
        stats["small_area_right"] = int(small_area_flags[:, 1].sum())
        stats["small_area_rois"] = int(small_area_pair_flags.sum())
    else:
        stats["small_area_left"] = 0
        stats["small_area_right"] = 0
        stats["small_area_rois"] = 0

    effective_success = ellipse_success_metrics & ~small_area_flags
    if total_rois > 0:
        run_group["ellipse_success"][:] = effective_success.astype(bool, copy=False)

    reason_array = np.array(reason_strings, dtype=object)
    reason_counts = dict(Counter(reason_strings))

    symmetry_sum_metrics = np.full(total_rois, np.nan, dtype=np.float32)
    symmetry_abs_diff_metrics = np.full(total_rois, np.nan, dtype=np.float32)
    symmetry_mask = np.all(np.isfinite(symmetry_offset_metrics), axis=1)
    symmetry_sum_metrics[symmetry_mask] = symmetry_offset_metrics[symmetry_mask].sum(axis=1)
    symmetry_abs_diff_metrics[symmetry_mask] = np.abs(
        np.abs(symmetry_offset_metrics[symmetry_mask, 0]) - np.abs(symmetry_offset_metrics[symmetry_mask, 1])
    )

    if "metrics" in run_group:
        del run_group["metrics"]
    metrics_group = run_group.create_group("metrics")

    metrics_group.create_array(
        "area_refined",
        data=refined_area_metrics.astype(np.float32),
        chunks=(chunk_rois, 2),
    )
    metrics_group.create_array(
        "area_source",
        data=source_area_metrics.astype(np.float32),
        chunks=(chunk_rois, 2),
    )
    metrics_group.create_array(
        "area_union_refined",
        data=union_refined_metrics.astype(np.float32),
        chunks=(chunk_rois,),
    )
    metrics_group.create_array(
        "area_union_source",
        data=union_source_metrics.astype(np.float32),
        chunks=(chunk_rois,),
    )
    metrics_group.create_array(
        "area_zscore",
        data=area_zscores.astype(np.float32),
        chunks=(chunk_rois, 2),
    )
    metrics_group.create_array(
        "area_ratio_left_right",
        data=area_ratio_lr.astype(np.float32),
        chunks=(chunk_rois,),
    )
    metrics_group.create_array(
        "area_diff_left_right",
        data=area_diff_lr.astype(np.float32),
        chunks=(chunk_rois,),
    )
    metrics_group.create_array(
        "area_delta_vs_source",
        data=area_delta_vs_source.astype(np.float32),
        chunks=(chunk_rois, 2),
    )
    metrics_group.create_array(
        "area_ratio_vs_source",
        data=area_ratio_vs_source.astype(np.float32),
        chunks=(chunk_rois, 2),
    )
    metrics_group.create_array(
        "area_union_delta",
        data=union_delta.astype(np.float32),
        chunks=(chunk_rois,),
    )
    metrics_group.create_array(
        "area_union_ratio",
        data=union_ratio.astype(np.float32),
        chunks=(chunk_rois,),
    )
    metrics_group.create_array(
        "centroid_error",
        data=centroid_error_metrics.astype(np.float32),
        chunks=(chunk_rois, 2),
    )
    metrics_group.create_array(
        "symmetry_offsets",
        data=symmetry_offset_metrics.astype(np.float32),
        chunks=(chunk_rois, 2),
    )
    metrics_group.create_array(
        "symmetry_sum",
        data=symmetry_sum_metrics.astype(np.float32),
        chunks=(chunk_rois,),
    )
    metrics_group.create_array(
        "symmetry_abs_diff",
        data=symmetry_abs_diff_metrics.astype(np.float32),
        chunks=(chunk_rois,),
    )
    metrics_group.create_array(
        "separation_refined",
        data=separation_refined_metrics.astype(np.float32),
        chunks=(chunk_rois,),
    )
    metrics_group.create_array(
        "separation_keypoint",
        data=separation_keypoint_metrics.astype(np.float32),
        chunks=(chunk_rois,),
    )
    metrics_group.create_array(
        "separation_delta",
        data=separation_delta_metrics.astype(np.float32),
        chunks=(chunk_rois,),
    )
    metrics_group.create_array(
        "axis_ratio",
        data=axis_ratio_metrics.astype(np.float32),
        chunks=(chunk_rois, 2),
    )
    metrics_group.create_array(
        "circularity",
        data=circularity_metrics.astype(np.float32),
        chunks=(chunk_rois, 2),
    )
    metrics_group.create_array(
        "connectivity_flags",
        data=connectivity_flags_metrics.astype(np.uint8, copy=False),
        chunks=(chunk_rois,),
    )
    metrics_group.create_array(
        "smoothing_flags",
        data=smoothing_flags_metrics.astype(np.uint8, copy=False),
        chunks=(chunk_rois, 2),
    )
    metrics_group.create_array(
        "pixels_reassigned",
        data=pixels_reassigned_metrics.astype(np.int32),
        chunks=(chunk_rois,),
    )
    metrics_group.create_array(
        "probabilities_used",
        data=probabilities_used_metrics.astype(bool, copy=False),
        chunks=(chunk_rois,),
    )
    metrics_group.create_array(
        "filter_flags",
        data=filter_flags.astype(bool, copy=False),
        chunks=(chunk_rois, 2),
    )
    metrics_group.create_array(
        "probability_mean",
        data=prob_mean_metrics.astype(np.float32),
        chunks=(chunk_rois, 2),
    )
    metrics_group.create_array(
        "probability_max",
        data=prob_max_metrics.astype(np.float32),
        chunks=(chunk_rois, 2),
    )
    metrics_group.create_array(
        "probability_var",
        data=prob_var_metrics.astype(np.float32),
        chunks=(chunk_rois, 2),
    )
    metrics_group.create_array(
        "probability_high_fraction",
        data=prob_high_metrics.astype(np.float32),
        chunks=(chunk_rois, 2),
    )
    reason_ds = metrics_group.create_array(
        "reason",
        shape=(total_rois,),
        chunks=(chunk_rois,),
        dtype=VariableLengthUTF8(),
        fill_value="",
    )
    reason_ds[:] = reason_array

    centroid_mean = np.nanmean(centroid_error_metrics, axis=0) if np.isfinite(centroid_error_metrics).any() else np.full(2, np.nan, dtype=np.float32)
    symmetry_sum_mean = float(np.nanmean(symmetry_sum_metrics)) if np.isfinite(symmetry_sum_metrics).any() else float("nan")
    separation_delta_mean = float(np.nanmean(separation_delta_metrics)) if np.isfinite(separation_delta_metrics).any() else float("nan")
    prob_mean_mean = (
        np.nanmean(prob_mean_metrics, axis=0)
        if np.isfinite(prob_mean_metrics).any()
        else np.full(2, np.nan, dtype=np.float32)
    )

    metrics_summary = {
        "num_rois": total_rois,
        "area_refined_mean": area_mean.tolist(),
        "area_refined_std": area_std.tolist(),
        "area_union_mean": float(np.nanmean(union_refined_metrics)) if total_rois else float("nan"),
        "area_union_std": float(np.nanstd(union_refined_metrics)) if total_rois else float("nan"),
        "centroid_error_mean": centroid_mean.tolist(),
        "symmetry_sum_mean": symmetry_sum_mean,
        "separation_delta_mean": separation_delta_mean,
        "probability_mean_mean": prob_mean_mean.tolist(),
        "probability_threshold": float(_PROBABILITY_THRESHOLD),
        "reason_counts": reason_counts,
        "probability_usage_rate": float(np.count_nonzero(probabilities_used_metrics) / total_rois) if total_rois else 0.0,
        "filtered_left": stats["filtered_left"],
        "filtered_right": stats["filtered_right"],
        "filtered_roi_pairs": stats["filtered_rois"],
        "small_area_left": stats["small_area_left"],
        "small_area_right": stats["small_area_right"],
        "small_area_roi_pairs": stats["small_area_rois"],
        "area_filter_threshold": area_filter_z,
        "area_filter_mode": area_filter_mode,
        "success_min_eye_area_px": success_min_eye_area_px,
    }

    left_concat = np.concatenate(left_points, axis=0).astype(np.float32) if left_points else np.zeros((0, 2), dtype=np.float32)
    right_concat = np.concatenate(right_points, axis=0).astype(np.float32) if right_points else np.zeros((0, 2), dtype=np.float32)

    left_store = left_concat if left_concat.size > 0 else np.zeros((1, 2), dtype=np.float32)
    right_store = right_concat if right_concat.size > 0 else np.zeros((1, 2), dtype=np.float32)

    run_group.create_array(
        "contour_left_ptr",
        data=left_ptr,
        chunks=(chunk_rois,),
        overwrite=True,
    )
    run_group.create_array(
        "contour_left_len",
        data=left_len,
        chunks=(chunk_rois,),
        overwrite=True,
    )
    run_group.create_array(
        "contour_right_ptr",
        data=right_ptr,
        chunks=(chunk_rois,),
        overwrite=True,
    )
    run_group.create_array(
        "contour_right_len",
        data=right_len,
        chunks=(chunk_rois,),
        overwrite=True,
    )
    run_group.create_array(
        "contours_left",
        data=left_store,
        chunks=(max(1, min(4096, left_store.shape[0])), 2),
        overwrite=True,
    )
    run_group.create_array(
        "contours_right",
        data=right_store,
        chunks=(max(1, min(4096, right_store.shape[0])), 2),
        overwrite=True,
    )

    source_method = src_run.attrs.get("method", "unknown")
    source_eye_labels = list(src_run.attrs.get("eye_labels", ["eye_0", "eye_1"]))
    eye_labels = ["eye_left", "eye_right"]

    total_eyes = int(effective_success.sum()) if total_rois else 0
    successful_pairs = int(np.all(effective_success, axis=1).sum()) if total_rois else 0
    pair_rate = float(successful_pairs / total_rois) if total_rois else float("nan")

    git_info = get_git_info()
    env_info = get_environment_info(
        include_all_packages=False,
        disk_path=str(zarr_path),
        collect_ip=False,
        capture_env_vars=False,
    )
    environment_summary = env_info.get("environment")
    duration = time.perf_counter() - stage_start

    probabilities_available = bool(has_mask_probs) or wrote_any_probs

    smoothing_mode = "off" if use_fast_path else _SMOOTHING_MODE
    smoothing_enabled = (smoothing_mode or "").lower() != "off"
    smoothing_info = {
        "mode": smoothing_mode,
        "enabled": smoothing_enabled,
        "median_k": int(_MEDIAN_K),
        "sdf_sigma": float(_SDF_SIGMA),
        "morph_closing_radius": int(_MORPH_CLOSING_RADIUS),
        "morph_opening_radius": int(_MORPH_OPENING_RADIUS),
        "closing_radius": int(_MORPH_CLOSING_RADIUS),
        "opening_radius": int(_MORPH_OPENING_RADIUS),
        "min_object_area": int(_MIN_OBJECT_AREA),
        "rois_modified": int(stats["smoothed_rois"]),
        "channels_modified": int(stats["smoothed_channels"]),
        "components_reassigned": int(stats["components_reassigned"]),
        "probabilities_available": probabilities_available,
        "probability_threshold": float(_PROBABILITY_THRESHOLD) if probabilities_available else None,
        "probability_splits": int(stats["probability_split"]),
        "probability_source": mask_prob_dataset or "none",
    }

    scheduler_info: Optional[Dict[str, object]] = None
    if scheduler_key:
        scheduler_info = {"type": scheduler_key}
        if num_workers is not None:
            scheduler_info["num_workers"] = int(num_workers)

    platform_info = {
        "hostname": env_info["platform"].get("hostname", "unknown"),
        "python_version": env_info["platform"].get("python_version", "unknown"),
        "system": env_info["platform"].get("system", "unknown"),
        "release": env_info["platform"].get("release", "unknown"),
        "machine": env_info["platform"].get("machine", "unknown"),
    }

    area_filter_info = {
        "enabled": area_filter_z is not None,
        "z_threshold": area_filter_z,
        "mode": area_filter_mode,
        "filtered_left": stats["filtered_left"],
        "filtered_right": stats["filtered_right"],
        "filtered_roi_pairs": stats["filtered_rois"],
    }

    small_area_filter_info = {
        "enabled": success_min_eye_area_px is not None,
        "min_eye_area_px": success_min_eye_area_px,
        "small_area_left": stats["small_area_left"],
        "small_area_right": stats["small_area_right"],
        "small_area_roi_pairs": stats["small_area_rois"],
    }

    mask_parameters = {
        "chunk_size": chunk_size,
        "probability_threshold": float(_PROBABILITY_THRESHOLD),
        "probabilities_available": probabilities_available,
        "mask_probability_source": mask_prob_dataset or ("refined" if wrote_any_probs else "none"),
        "mask_binary_source": mask_binary_source or "unknown",
        "mask_bundle": mask_bundle_provenance,
        "smoothing": smoothing_info,
        "traditional_fast_path": bool(use_fast_path),
        "force_refine_traditional": bool(force_refine_traditional),
        "components_reassigned": int(stats["components_reassigned"]),
        "probability_split": int(stats["probability_split"]),
        "metrics_version": 1,
        "area_filter": area_filter_info,
        "success_area_filter": small_area_filter_info,
    }

    artifact_keys = [
        "segmenter",
        "segmenter_method",
        "segmenter_label_mode",
        "source_checkpoint",
        "source_checkpoint_best_val_dice",
        "dataset_meta",
        "inference_device",
        "inference_batch_size",
        "inference_duration_seconds",
        "model_path",
        "model_name",
    ]
    artifact_info = {key: src_run.attrs[key] for key in artifact_keys if key in src_run.attrs}

    created_timestamp = created_at_utc or datetime.now(timezone.utc).isoformat()
    provenance_record = build_stage_provenance(
        stage="refine_eye_masks",
        command=command or "unknown",
        created_at_utc=created_timestamp,
        version=git_info.get("short_hash") or git_info.get("commit_hash"),
        git={
            "commit": git_info.get("commit_hash"),
            "short": git_info.get("short_hash"),
            "branch": git_info.get("branch"),
            "is_dirty": git_info.get("is_dirty"),
            "remote": git_info.get("remote_url"),
        },
        environment=environment_summary,
        platform=platform_info,
        scheduler=scheduler_info,
        parameters=mask_parameters,
        inputs={
            "eye_masks_run": src_run_name,
            "keypoints_run": keypoint_run_name,
            "crop_run": crop_run_name,
        },
        artifacts=artifact_info,
    )

    attrs_payload: Dict[str, object] = {
        "method": "refine_eye_masks",
        "source_eye_masks_run": src_run_name,
        "source_eye_masks_method": source_method,
        **build_source_keypoints_attrs(keypoint_run_name, include_legacy_alias=True),
        "source_keypoint_group": keypoint_group_name,
        "source_crop_run": crop_run_name,
        "total_rois": total_rois,
        "successful_eyes": total_eyes,
        "successful_roi_pairs": successful_pairs,
        "successful_roi_pair_rate": pair_rate,
        "refine_stats": stats,
        "duration_seconds": duration,
        "source_eye_labels": source_eye_labels,
        "eye_labels": eye_labels,
        "smoothing": smoothing_info,
        "mask_probabilities_available": probabilities_available,
        "mask_probability_threshold": float(_PROBABILITY_THRESHOLD) if probabilities_available else None,
        "mask_probability_source": mask_prob_dataset or ("refined" if wrote_any_probs else "none"),
        "mask_probability_policy": "union_prob",
        "mask_binary_source": mask_binary_source or "unknown",
        "mask_binary_identity": bundle.binary_identity,
        "mask_probability_identity": bundle.probs_identity or "none",
        "mask_bundle_provenance": mask_bundle_provenance,
        "success_min_eye_area_px": success_min_eye_area_px,
        "min_eye_area_success_px": success_min_eye_area_px,
        "traditional_fast_path": bool(use_fast_path),
        "force_refine_traditional": bool(force_refine_traditional),
        "dask_scheduler": scheduler_key,
        "dask_num_workers": int(num_workers) if num_workers is not None else None,
        "dask_chunk_size": chunk_size,
        "dask_version": getattr(dask, "__version__", "unknown"),
        "hostname": env_info["platform"].get("hostname", "unknown"),
        "metrics_summary": metrics_summary,
        "summary_statistics": {"refine": stats},
    }

    source_provenance = src_run.attrs.get("provenance")
    if source_provenance is not None:
        attrs_payload["source_eye_masks_provenance"] = source_provenance

    run_group.attrs.update(attrs_payload)
    write_stage_provenance(run_group, provenance_record)

    console.print(
        f"[green]✓[/green] Refined eye masks saved to [cyan]{REFINED_STAGE_NAME}_runs/{resolved_run_name}[/cyan] "
        f"({successful_pairs}/{total_rois} ROI pairs refined)"
    )

    review_status = run_group.attrs.get("eye_mask_review_status")
    coverage_pct = _status_float(run_group.attrs.get("successful_roi_pair_rate"))
    if coverage_pct is not None and coverage_pct <= 1.0:
        coverage_pct *= 100.0
    resolved_zarr_path = Path(zarr_path).expanduser().resolve()
    _emit_refined_eye_masks_status(
        root=root,
        zarr_path=resolved_zarr_path,
        status="ok",
        run_name=resolved_run_name,
        method=_status_text(run_group.attrs.get("method")) or "refine_eye_masks",
        coverage_pct=coverage_pct,
        review_status=review_status if isinstance(review_status, dict) else None,
        details={
            "reason": "present",
            "source_eye_masks_run": src_run_name,
            "source_crop_run": crop_run_name,
            "source_keypoints_run": keypoint_run_name,
            "source_keypoint_group": keypoint_group_name,
            "total_rois": total_rois,
            "successful_roi_pairs": successful_pairs,
            "successful_roi_pair_rate": pair_rate,
            "filtered_roi_pairs": stats.get("filtered_rois"),
            "small_area_roi_pairs": stats.get("small_area_rois"),
            "probability_source": mask_prob_dataset or "none",
        },
        console=console,
    )
    _refresh_refined_eye_masks_registry_rows(
        root=root,
        zarr_path=resolved_zarr_path,
        console=console,
    )
    return resolved_run_name


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Refine eye-mask segmentation outputs.")
    parser.add_argument("zarr_path", help="Path to the Palette Zarr archive.")
    parser.add_argument(
        "--source-run",
        help="Eye mask run name to refine (default: latest in eye_masks_runs).",
    )
    parser.add_argument(
        "--keypoint-run",
        help=(
            "Keypoint run providing headings. "
            "Defaults to source lineage attrs on the eye-mask run (fail-closed if missing)."
        ),
    )
    parser.add_argument(
        "--allow-latest-keypoint-fallback",
        action="store_true",
        help=(
            "Allow fallback to latest refined/raw keypoint run when source lineage attrs are missing. "
            "Intended only for legacy archives."
        ),
    )
    parser.add_argument(
        "--run-name",
        help="Name for the new refined run (default: auto-generated).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=512,
        help="Number of ROIs to refine per chunk (default: 512).",
    )
    parser.add_argument(
        "--scheduler",
        choices=["threads", "processes"],
        default="processes",
        help="Dask scheduler to use for refinement (default: processes).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Number of worker threads/processes for the Dask scheduler.",
    )
    parser.add_argument(
        "--area-filter-z",
        type=float,
        default=_AREA_FILTER_Z_DEFAULT,
        help="Z-score threshold for per-eye area filtering (set to 2.0 by default; set <=0 to disable).",
    )
    parser.add_argument(
        "--area-filter-mode",
        choices=["either", "both"],
        default=_AREA_FILTER_MODE_DEFAULT,
        help="Require either eye or both eyes to exceed the threshold before flagging a pair.",
    )
    parser.add_argument(
        "--success-min-eye-area-px",
        type=float,
        default=_SUCCESS_MIN_EYE_AREA_PX_DEFAULT,
        help="Require each eye area to be at least this many pixels to count ROI-pair success (default: 50).",
    )
    parser.add_argument(
        "--force-refine-traditional",
        action="store_true",
        help="Run full refinement even for traditional eye-mask runs (enables smoothing/component enforcement).",
    )
    return parser


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    arg_list = list(argv) if argv is not None else sys.argv[1:]
    command_str = "python -m fisheye.refinement.refine_eye_masks"
    if arg_list:
        command_str = f"{command_str} " + " ".join(str(val) for val in arg_list)
    created_at = datetime.now(timezone.utc).isoformat()

    console = Console()
    try:
        refine_eye_masks(
            args.zarr_path,
            source_run=args.source_run,
            run_name=args.run_name,
            keypoint_run=args.keypoint_run,
            chunk_size=args.chunk_size,
            scheduler=args.scheduler,
            num_workers=args.num_workers,
            console=console,
            command=command_str,
            created_at_utc=created_at,
            area_filter_z=args.area_filter_z,
            area_filter_mode=args.area_filter_mode,
            success_min_eye_area_px=args.success_min_eye_area_px,
            force_refine_traditional=args.force_refine_traditional,
            allow_latest_keypoint_fallback=args.allow_latest_keypoint_fallback,
        )
    except Exception as exc:
        resolved_zarr_path = Path(args.zarr_path).expanduser().resolve()
        error_root: Optional[zarr.Group] = None
        try:
            error_root = zarr.open(str(resolved_zarr_path), mode="r", use_consolidated=False)
        except Exception:
            error_root = None
        if error_root is not None:
            _emit_refined_eye_masks_status(
                root=error_root,
                zarr_path=resolved_zarr_path,
                status="error",
                run_name=args.run_name,
                method="refine_eye_masks",
                coverage_pct=None,
                review_status=None,
                details={
                    "reason": "runtime_error",
                    "error": str(exc),
                    "source_eye_masks_run": args.source_run,
                    "source_keypoints_run": args.keypoint_run,
                },
                console=console,
            )
        console.print(f"[red]✗[/red] Failed to refine eye masks: {exc}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
