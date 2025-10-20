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
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import dask
import numpy as np
import zarr
from dask import delayed
from dask.diagnostics import ProgressBar
from rich.console import Console
from scipy.ndimage import distance_transform_edt, gaussian_filter, median_filter
from skimage import measure, morphology

from ..segmentation.eye_segmentation import (
    _extract_contour,
    _feret_mask_from_region,
)
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

_DEBUG_ONCE = True


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
    feret_major: np.ndarray  # (2, 4) float32
    feret_minor: np.ndarray  # (2, 4) float32
    feret_roundness: np.ndarray  # (2,) float32
    centroids: np.ndarray  # (2, 2) float32 (x, y) or nan
    contours: Tuple[Optional[np.ndarray], Optional[np.ndarray]]
    eye_separation: float
    used_original_order: bool
    reason: Optional[str]
    smoothing_changed: np.ndarray  # (2,) bool
    reassigned_pixels: int


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


def _measure_mask(mask: np.ndarray, min_contour_points: int = 5) -> Tuple[bool, np.ndarray, np.ndarray, np.ndarray, float, np.ndarray, Optional[np.ndarray]]:
    """Extract metrics from a binary mask."""

    if mask.sum() == 0:
        ellipse = np.full(5, np.nan, dtype=np.float32)
        feret_major = np.full(4, np.nan, dtype=np.float32)
        feret_minor = np.full(4, np.nan, dtype=np.float32)
        centroid = np.full(2, np.nan, dtype=np.float32)
        return False, ellipse, feret_major, feret_minor, float("nan"), centroid, None

    region_mask = mask.astype(np.uint8)
    props = measure.regionprops(region_mask)
    if not props:
        ellipse = np.full(5, np.nan, dtype=np.float32)
        feret_major = np.full(4, np.nan, dtype=np.float32)
        feret_minor = np.full(4, np.nan, dtype=np.float32)
        centroid = np.full(2, np.nan, dtype=np.float32)
        return False, ellipse, feret_major, feret_minor, float("nan"), centroid, None

    region = props[0]
    centroid = np.array([float(region.centroid[1]), float(region.centroid[0])], dtype=np.float32)
    ellipse = np.array(
        [
            centroid[0],
            centroid[1],
            float(region.major_axis_length),
            float(region.minor_axis_length),
            float(np.rad2deg(region.orientation)),
        ],
        dtype=np.float32,
    )

    feret_major = np.full(4, np.nan, dtype=np.float32)
    feret_minor = np.full(4, np.nan, dtype=np.float32)
    feret_roundness = float("nan")

    feret_mask, info = _feret_mask_from_region(region_mask, 0.0)
    if feret_mask is not None and info:
        feret_roundness = float(info.get("roundness", float("nan")))
        major_pts = info.get("major_pts")
        minor_pts = info.get("minor_pts")
        if isinstance(major_pts, np.ndarray) and major_pts.shape == (2, 2):
            feret_major = np.array(
                [major_pts[0, 0], major_pts[0, 1], major_pts[1, 0], major_pts[1, 1]],
                dtype=np.float32,
            )
        if isinstance(minor_pts, np.ndarray) and minor_pts.shape == (2, 2):
            feret_minor = np.array(
                [minor_pts[0, 0], minor_pts[0, 1], minor_pts[1, 0], minor_pts[1, 1]],
                dtype=np.float32,
            )

    contour = _extract_contour(mask.astype(float), min_contour_points)
    if contour is not None:
        contour = contour.astype(np.float32)

    return True, ellipse, feret_major, feret_minor, feret_roundness, centroid, contour


def _refine_roi(
    source_masks: np.ndarray,
    keypoints_roi: np.ndarray,
    heading_deg: float,
    success_flag: bool,
) -> ROIOutput:
    """Refine a single ROI's mask assignment."""

    roi_h, roi_w = source_masks.shape[1:]
    masks_out = np.zeros((2, roi_h, roi_w), dtype=np.uint8)
    ellipse_params = np.full((2, 5), np.nan, dtype=np.float32)
    ellipse_success = np.zeros(2, dtype=bool)
    feret_major = np.full((2, 4), np.nan, dtype=np.float32)
    feret_minor = np.full((2, 4), np.nan, dtype=np.float32)
    feret_roundness = np.full(2, np.nan, dtype=np.float32)
    centroids = np.full((2, 2), np.nan, dtype=np.float32)

    def _copy_original(reason: str) -> ROIOutput:
        contours: Tuple[Optional[np.ndarray], Optional[np.ndarray]] = (None, None)
        smoothing_flags = np.zeros(2, dtype=bool)
        for eye_idx, mask_raw in enumerate(source_masks):
            mask_bool = mask_raw > 0
            mask_bool, changed = _smooth_binary_mask(mask_bool)
            smoothing_flags[eye_idx] = changed
            masks_out[eye_idx] = mask_bool.astype(np.uint8)
            (
                success,
                ellipse,
                major,
                minor,
                roundness,
                centroid,
                contour,
            ) = _measure_mask(mask_bool.astype(np.uint8))
            ellipse_success[eye_idx] = success
            ellipse_params[eye_idx] = ellipse
            feret_major[eye_idx] = major
            feret_minor[eye_idx] = minor
            feret_roundness[eye_idx] = roundness
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

        return ROIOutput(
            masks_out,
            ellipse_params,
            ellipse_success,
            feret_major,
            feret_minor,
            feret_roundness,
            centroids,
            contours,
            separation,
            True,
            reason,
            smoothing_flags,
            0,
        )

    if not success_flag:
        return _copy_original("keypoint_fail")

    union_mask = (source_masks[0] > 0) | (source_masks[1] > 0)
    if union_mask.sum() == 0:
        return _copy_original("empty_union")

    eye_left = keypoints_roi[1]
    eye_right = keypoints_roi[2]

    # Dilate once before splitting so thin seams stay connected, then clamp back.
    union_pad = morphology.binary_dilation(union_mask, morphology.square(3))
    left_mask, right_mask = _split_mask_by_keypoints(union_pad, eye_left, eye_right)
    left_mask &= union_mask
    right_mask &= union_mask
    used_original_order = False
    reason = None

    if left_mask.sum() == 0 or right_mask.sum() == 0:
        # Try heading-based split around midpoint of keypoints.
        midpoint = np.array([(eye_left[0] + eye_right[0]) / 2.0, (eye_left[1] + eye_right[1]) / 2.0], dtype=np.float32)
        fallback_left, fallback_right = _split_mask_by_heading(union_mask, heading_deg, midpoint)
        if fallback_left.sum() > 0 and fallback_right.sum() > 0:
            left_mask, right_mask = fallback_left, fallback_right
            reason = "heading_split"
        else:
            # Give up and retain original ordering.
            left_mask = source_masks[0] > 0
            right_mask = source_masks[1] > 0
            used_original_order = True
            reason = "original_order"

    left_mask_sm, left_changed = _smooth_binary_mask(left_mask)
    right_mask_sm, right_changed = _smooth_binary_mask(right_mask)

    left_mask_connected, right_mask_connected, component_flags, reassigned_pixels = _enforce_single_component(
        left_mask_sm,
        right_mask_sm,
        eye_left,
        eye_right,
    )

    global _DEBUG_ONCE
    if _DEBUG_ONCE:
        def _ncc(mask: np.ndarray) -> int:
            return int(measure.label(mask.astype(bool), connectivity=2).max())

        dbg = {
            "left_cc_split": _ncc(left_mask),
            "right_cc_split": _ncc(right_mask),
            "left_cc_sm": _ncc(left_mask_sm),
            "right_cc_sm": _ncc(right_mask_sm),
            "left_cc_final": _ncc(left_mask_connected),
            "right_cc_final": _ncc(right_mask_connected),
        }
        # Use regular print instead of Console to avoid pickling issues with multiprocessing
        print(f"refine dbg: {dbg}")
        _DEBUG_ONCE = False

    smoothing_flags = np.array([left_changed, right_changed], dtype=bool) | component_flags

    masks_out[0] = left_mask_connected.astype(np.uint8)
    masks_out[1] = right_mask_connected.astype(np.uint8)

    contours: List[Optional[np.ndarray]] = [None, None]
    for eye_idx, mask in enumerate((left_mask_connected, right_mask_connected)):
        (
            success,
            ellipse,
            major,
            minor,
            roundness,
            centroid,
            contour,
        ) = _measure_mask(mask.astype(np.uint8))

        ellipse_success[eye_idx] = success
        ellipse_params[eye_idx] = ellipse
        feret_major[eye_idx] = major
        feret_minor[eye_idx] = minor
        feret_roundness[eye_idx] = roundness
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

    return ROIOutput(
        masks_out,
        ellipse_params,
        ellipse_success,
        feret_major,
        feret_minor,
        feret_roundness,
        centroids,
        (contours[0], contours[1]),
        eye_separation,
        used_original_order,
        reason,
        smoothing_flags,
        int(reassigned_pixels),
    )


def _process_refine_chunk(
    offset: int,
    masks_chunk: np.ndarray,
    keypoints_chunk: np.ndarray,
    heading_chunk: np.ndarray,
    success_chunk: np.ndarray,
) -> List[Tuple[int, ROIOutput]]:
    """Process a batch of ROI indices and return their refinement outputs."""
    masks_np = np.asarray(masks_chunk)
    keypoints_np = np.asarray(keypoints_chunk)
    heading_np = np.asarray(heading_chunk)
    success_np = np.asarray(success_chunk)

    results: List[Tuple[int, ROIOutput]] = []
    for local_idx in range(masks_np.shape[0]):
        global_idx = offset + local_idx
        source_masks = masks_np[local_idx]
        keypoints_roi = keypoints_np[local_idx]
        heading = float(heading_np[local_idx])
        success_flag = bool(success_np[local_idx])
        results.append((global_idx, _refine_roi(source_masks, keypoints_roi, heading, success_flag)))
    return results


def refine_eye_masks(
    zarr_path: str,
    source_run: Optional[str] = None,
    run_name: Optional[str] = None,
    *,
    keypoint_run: Optional[str] = None,
    chunk_size: int = 1024,
    scheduler: str = "threads",
    num_workers: Optional[int] = None,
    console: Optional[Console] = None,
) -> str:
    """Refine an eye-mask run and return the name of the new run."""

    console = console or Console()
    global _DEBUG_ONCE
    _DEBUG_ONCE = True
    stage_start = time.perf_counter()

    zarr_path = str(Path(zarr_path))

    root = zarr.open(zarr_path, mode="a")

    if "eye_masks_runs" not in root:
        raise ValueError("Zarr archive missing eye_masks_runs; run segmentation first.")
    eye_parent = root["eye_masks_runs"]
    src_run_name = source_run or eye_parent.attrs.get("latest")
    if src_run_name is None or src_run_name not in eye_parent:
        raise ValueError("Source eye mask run not found.")
    src_run = eye_parent[src_run_name]

    if "masks_roi" not in src_run:
        raise ValueError(f"eye_masks_runs/{src_run_name} lacks 'masks_roi'.")

    crop_run_name = src_run.attrs.get("source_crop_run") or root.get("crop_runs", {}).attrs.get("latest")
    if crop_run_name is None:
        raise ValueError("Unable to determine crop run (missing attribute 'source_crop_run').")

    kp_parent = root.require_group("keypoints_runs")
    keypoint_run_name = (
        keypoint_run
        or src_run.attrs.get("source_keypoints_run")
        or kp_parent.attrs.get("latest")
    )
    if keypoint_run_name is None or keypoint_run_name not in kp_parent:
        raise ValueError("Keypoint run required for refinement (set --keypoint-run).")
    kp_group = kp_parent[keypoint_run_name]

    required_kp = ["keypoints_roi", "heading", "detection_success"]
    for arr in required_kp:
        if arr not in kp_group:
            raise ValueError(f"Keypoint run '{keypoint_run_name}' missing '{arr}'.")

    masks_ds = src_run["masks_roi"]
    total_rois, _, roi_h, roi_w = masks_ds.shape

    run_group, resolved_run_name = _prepare_run_group(root, run_name, console)

    masks_out = np.zeros((total_rois, 2, roi_h, roi_w), dtype=np.uint8)
    ellipse_params = np.full((total_rois, 2, 5), np.nan, dtype=np.float32)
    ellipse_success = np.zeros((total_rois, 2), dtype=bool)
    feret_major = np.full((total_rois, 2, 4), np.nan, dtype=np.float32)
    feret_minor = np.full((total_rois, 2, 4), np.nan, dtype=np.float32)
    feret_roundness = np.full((total_rois, 2), np.nan, dtype=np.float32)
    eye_separation = np.full((total_rois,), np.nan, dtype=np.float32)

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
        "original_order": 0,
        "smoothed_rois": 0,
        "smoothed_channels": 0,
        "components_reassigned": 0,
    }

    scheduler_key = (scheduler or "processes").lower()
    if scheduler_key not in {"threads", "processes"}:
        console.print(f"[yellow]Unknown scheduler '{scheduler_key}', defaulting to 'threads'.[/yellow]")
        scheduler_key = "threads"

    compute_kwargs: Dict[str, object] = {"scheduler": scheduler_key}
    if num_workers is not None:
        compute_kwargs["num_workers"] = int(num_workers)

    chunk_size = max(1, chunk_size)
    tasks: List[object] = []
    if total_rois > 0:
        keypoints_ds = kp_group["keypoints_roi"]
        heading_ds = kp_group["heading"]
        success_ds = kp_group["detection_success"]

        for start in range(0, total_rois, chunk_size):
            stop = min(start + chunk_size, total_rois)
            sl = slice(start, stop)

            masks_block = np.asarray(masks_ds[sl])
            keypoints_block = np.asarray(keypoints_ds[sl])
            heading_block = np.asarray(heading_ds[sl])
            success_block = np.asarray(success_ds[sl])

            tasks.append(
                delayed(_process_refine_chunk)(
                    start,
                    masks_block,
                    keypoints_block,
                    heading_block,
                    success_block,
                )
            )

    chunk_results: List[List[Tuple[int, ROIOutput]]] = []
    if tasks:
        console.print(f"[cyan]Submitting {len(tasks)} chunk(s) to Dask ({scheduler_key})[/cyan]")
        with ProgressBar():
            chunk_results = list(dask.compute(*tasks, **compute_kwargs))

    gathered_results: List[Tuple[int, ROIOutput]] = []
    for chunk in chunk_results:
        gathered_results.extend(chunk)

    gathered_results.sort(key=lambda item: item[0])

    for global_idx, result in gathered_results:
        masks_out[global_idx] = result.masks
        ellipse_params[global_idx] = result.ellipse_params
        ellipse_success[global_idx] = result.ellipse_success
        feret_major[global_idx] = result.feret_major
        feret_minor[global_idx] = result.feret_minor
        feret_roundness[global_idx] = result.feret_roundness
        eye_separation[global_idx] = result.eye_separation
        if result.smoothing_changed.any():
            stats["smoothed_rois"] += 1
            stats["smoothed_channels"] += int(result.smoothing_changed.sum())

        if result.reassigned_pixels:
            stats["components_reassigned"] += int(result.reassigned_pixels)

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
        if reason == "heading_split":
            stats["fallback_heading"] += 1
            stats["refined"] += 1
        elif reason == "original_order":
            stats["original_order"] += 1
            stats["copied_original"] += 1
        elif reason == "keypoint_fail":
            stats["keypoint_fail"] += 1
            stats["copied_original"] += 1
        elif reason == "empty_union":
            stats["empty_union"] += 1
            stats["copied_original"] += 1
        else:
            stats["refined"] += 1

    left_concat = np.concatenate(left_points, axis=0).astype(np.float32) if left_points else np.zeros((0, 2), dtype=np.float32)
    right_concat = np.concatenate(right_points, axis=0).astype(np.float32) if right_points else np.zeros((0, 2), dtype=np.float32)

    left_store = left_concat if left_concat.size > 0 else np.zeros((1, 2), dtype=np.float32)
    right_store = right_concat if right_concat.size > 0 else np.zeros((1, 2), dtype=np.float32)

    chunk_rois = min(512, total_rois) if total_rois > 0 else 1

    run_group.create_array(
        "masks_roi",
        data=masks_out,
        chunks=(chunk_rois, 2, roi_h, roi_w),
        overwrite=True,
    )
    run_group.create_array(
        "ellipse_params",
        data=ellipse_params,
        chunks=(chunk_rois, 2, 5),
        overwrite=True,
    )
    run_group.create_array(
        "ellipse_success",
        data=ellipse_success,
        chunks=(chunk_rois, 2),
        overwrite=True,
    )
    run_group.create_array(
        "feret_axes_major",
        data=feret_major,
        chunks=(chunk_rois, 2, 4),
        overwrite=True,
    )
    run_group.create_array(
        "feret_axes_minor",
        data=feret_minor,
        chunks=(chunk_rois, 2, 4),
        overwrite=True,
    )
    run_group.create_array(
        "feret_roundness",
        data=feret_roundness,
        chunks=(chunk_rois, 2),
        overwrite=True,
    )
    run_group.create_array(
        "eye_separation",
        data=eye_separation,
        chunks=(chunk_rois,),
        overwrite=True,
    )
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

    total_eyes = int(ellipse_success.sum())
    successful_pairs = int(np.sum(ellipse_success.all(axis=1)))
    pair_rate = float(successful_pairs / total_rois) if total_rois else float("nan")

    git_info = get_git_info()
    env_info = get_environment_info()
    duration = time.perf_counter() - stage_start

    smoothing_info = {
        "mode": _SMOOTHING_MODE,
        "enabled": (_SMOOTHING_MODE or "").lower() != "off",
        "median_k": int(_MEDIAN_K),
        "sdf_sigma": float(_SDF_SIGMA),
        "morph_closing_radius": int(_MORPH_CLOSING_RADIUS),
        "morph_opening_radius": int(_MORPH_OPENING_RADIUS),
        "min_object_area": int(_MIN_OBJECT_AREA),
        "rois_modified": int(stats["smoothed_rois"]),
        "channels_modified": int(stats["smoothed_channels"]),
        "components_reassigned": int(stats["components_reassigned"]),
    }

    run_group.attrs.update(
        {
            "method": "refine_eye_masks",
            "source_eye_masks_run": src_run_name,
            "source_eye_masks_method": source_method,
            "source_keypoints_run": keypoint_run_name,
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
            "git_commit": git_info.get("commit_hash", "unknown"),
            "git_branch": git_info.get("branch", "unknown"),
            "hostname": env_info["platform"].get("hostname", "unknown"),
        }
    )

    console.print(
        f"[green]✓[/green] Refined eye masks saved to [cyan]{REFINED_STAGE_NAME}_runs/{resolved_run_name}[/cyan] "
        f"({successful_pairs}/{total_rois} ROI pairs refined)"
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
        help="Keypoint run providing headings (default: infer from source or latest).",
    )
    parser.add_argument(
        "--run-name",
        help="Name for the new refined run (default: auto-generated).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1024,
        help="Number of ROIs to refine per chunk (default: 1024).",
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
    return parser


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

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
        )
    except Exception as exc:
        console.print(f"[red]✗[/red] Failed to refine eye masks: {exc}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
