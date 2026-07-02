"""Shared metrics and contour packing for refined eye-mask-compatible runs."""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import zarr
from skimage import measure

from .mask_geometry import extract_mask_contour


def compute_refined_eye_roi_metrics(
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
    probability_threshold: float = 0.45,
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

    if np.all(np.isfinite(keypoint_coords[0])) and np.all(np.isfinite(keypoint_coords[1])):
        keypoint_separation = float(np.linalg.norm(keypoint_coords[0] - keypoint_coords[1]))
    else:
        keypoint_separation = float("nan")
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
            circularity[idx] = float((4.0 * math.pi * area) / (perimeter**2 + 1e-12))

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
            prob_high[idx] = float(np.count_nonzero(mask_vals >= probability_threshold) / mask_vals.size)

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


def write_refined_eye_contours_from_masks(
    run_group: zarr.Group,
    *,
    total_rois: int,
    chunk_rois: int,
) -> None:
    """Build packed left/right contour stores from final masks."""
    left_ptr = np.full((total_rois,), -1, dtype=np.int64)
    left_len = np.zeros((total_rois,), dtype=np.int32)
    right_ptr = np.full((total_rois,), -1, dtype=np.int64)
    right_len = np.zeros((total_rois,), dtype=np.int32)

    left_points: List[np.ndarray] = []
    right_points: List[np.ndarray] = []
    left_total = 0
    right_total = 0

    masks_arr = run_group["masks_roi"]
    for start in range(0, total_rois, chunk_rois):
        stop = min(total_rois, start + chunk_rois)
        mask_chunk = np.asarray(masks_arr[start:stop], dtype=np.uint8)
        for local_idx, masks_row in enumerate(mask_chunk):
            global_idx = start + local_idx
            left_contour = extract_mask_contour(masks_row[0], min_points=5)
            if left_contour is not None:
                contour_len = int(left_contour.shape[0])
                left_ptr[global_idx] = left_total
                left_len[global_idx] = contour_len
                left_points.append(left_contour.astype(np.float32, copy=False))
                left_total += contour_len

            right_contour = extract_mask_contour(masks_row[1], min_points=5)
            if right_contour is not None:
                contour_len = int(right_contour.shape[0])
                right_ptr[global_idx] = right_total
                right_len[global_idx] = contour_len
                right_points.append(right_contour.astype(np.float32, copy=False))
                right_total += contour_len

    left_concat = np.concatenate(left_points, axis=0).astype(np.float32) if left_points else np.zeros((0, 2), dtype=np.float32)
    right_concat = (
        np.concatenate(right_points, axis=0).astype(np.float32) if right_points else np.zeros((0, 2), dtype=np.float32)
    )
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
        chunks=(max(1, min(4096, int(left_store.shape[0]))), 2),
        overwrite=True,
    )
    run_group.create_array(
        "contours_right",
        data=right_store,
        chunks=(max(1, min(4096, int(right_store.shape[0]))), 2),
        overwrite=True,
    )


_compute_roi_metrics = compute_refined_eye_roi_metrics
_write_contours_from_masks = write_refined_eye_contours_from_masks
