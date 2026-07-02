#!/usr/bin/env python3
"""Interactive tuner for traditional eye segmentation parameters with threshold visualization."""

from __future__ import annotations

# Limit threading BEFORE importing numpy/cv2/skimage to prevent using all CPU cores
import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')

import argparse
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, Sequence, List
from concurrent.futures import ThreadPoolExecutor

import cv2

# Limit OpenCV threads
cv2.setNumThreads(2)
import numpy as np
import zarr
from skimage import measure
from datetime import datetime, timezone

from ..pose.schema import resolve_required_keypoint_indices_from_attrs
from ..segmentation.eye_segmentation import EyeSegmentationConfig, _process_roi_data
from ..shared.mask_tuning_helpers import (
    DEBUG_PANEL_MARGIN,
    DEBUG_PANEL_SCALE,
    DEBUG_PANEL_SPACING,
    SLIDER_MAX_AREA,
    SLIDER_MAX_CIRCULARITY,
    SLIDER_MAX_EYE_GAP,
    SLIDER_MAX_PADDING,
    SLIDER_MAX_PRETHRESH,
    SLIDER_MAX_RADIUS,
    SLIDER_MAX_SOBEL,
    _resolve_keypoints_group,
    apply_circular_window,
    apply_sobel_filter,
    compute_heading_deg,
    create_debug_panel,
    draw_overlay,
    global_mask,
    rotate_image_and_points,
    select_region,
)


_DEFAULT_SUCCESS_MIN_EYE_AREA_PX = 50.0
_HEAD_KEYPOINT_LABELS = ("swim_bladder", "eye_left", "eye_right")
_EYE_KEYPOINT_LABELS = ("eye_left", "eye_right")


def save_eye_mask_params(zarr_path: Path, params: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Persist eye mask tuning parameters into the Zarr's analysis_metadata group.
    """
    try:
        root = zarr.open(str(zarr_path), mode='a', use_consolidated=False)
        if 'analysis_metadata' not in root:
            root.create_group('analysis_metadata')
        analysis_meta = root['analysis_metadata']

        metadata = dict(analysis_meta.attrs) if analysis_meta.attrs else {}
        metadata['eye_mask_tuning'] = {
            'method': 'global_threshold_otsu',
            'version': '1.0',
            'tuned_timestamp': datetime.now(timezone.utc).isoformat(),
            'tuned_parameters': params['tuned_parameters'],
            'context': params['context'],
        }
        analysis_meta.attrs.update(metadata)

        print(f"\n✓ Eye mask parameters saved to zarr analysis_metadata")
        for key, value in params['tuned_parameters'].items():
            print(f"   {key}: {value}")
        print("   Context:")
        for key, value in params['context'].items():
            print(f"     {key}: {value}")
        return True, "Parameters saved"
    except Exception as exc:
        return False, f"Error saving parameters: {exc}"

def _resolve_required_keypoints(
    kp_group: zarr.Group,
    required_labels: Sequence[str],
) -> Dict[str, int]:
    keypoint_count = int(kp_group["keypoints_roi"].shape[1])
    try:
        return resolve_required_keypoint_indices_from_attrs(
            kp_group.attrs,
            required_labels,
            keypoint_count=keypoint_count,
        )
    except ValueError as exc:
        raise RuntimeError(
            f"Keypoint run is missing required labels {tuple(required_labels)}: {exc}"
        ) from exc


def _get_sep_limits(root: zarr.Group, refined: zarr.Group) -> tuple[Optional[float], Optional[float]]:
    analysis = root.get("analysis_metadata")
    if analysis is not None:
        tuning = analysis.attrs.get("eye_mask_tuning")
        if isinstance(tuning, dict):
            params = tuning.get("tuned_parameters", {})
            min_sep = params.get("min_eye_separation")
            max_sep = params.get("max_eye_separation")
            return (
                float(min_sep) if min_sep is not None else None,
                float(max_sep) if max_sep is not None else None,
            )

    source_run = refined.attrs.get("source_eye_masks_run")
    if source_run and "eye_masks_runs" in root and source_run in root["eye_masks_runs"]:
        src = root["eye_masks_runs"][source_run]
        min_sep = src.attrs.get("min_eye_separation")
        max_sep = src.attrs.get("max_eye_separation")
        return (
            float(min_sep) if min_sep is not None else None,
            float(max_sep) if max_sep is not None else None,
        )
    return None, None


def _positive_float(value: object) -> Optional[float]:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if numeric <= 0:
        return None
    return numeric


def _resolve_success_min_eye_area_px(refined: zarr.Group) -> Optional[float]:
    raw = refined.attrs.get("success_min_eye_area_px")
    if raw is None:
        raw = refined.attrs.get("min_eye_area_success_px")
    threshold = _positive_float(raw)
    if threshold is not None:
        return threshold
    return float(_DEFAULT_SUCCESS_MIN_EYE_AREA_PX)


def _load_area_refined(refined: zarr.Group) -> Optional[np.ndarray]:
    masks_arr = refined.get("masks_roi")
    shape = getattr(masks_arr, "shape", None) if masks_arr is not None else None
    if isinstance(shape, tuple) and len(shape) == 4 and int(shape[1]) >= 2:
        total = int(shape[0])
        if total <= 0:
            return np.zeros((0, 2), dtype=np.float32)
        chunks = getattr(masks_arr, "chunks", None)
        if isinstance(chunks, tuple) and chunks:
            step = max(1, int(chunks[0]))
        elif isinstance(chunks, int):
            step = max(1, int(chunks))
        else:
            step = 256
        out = np.zeros((total, 2), dtype=np.float32)
        for start in range(0, total, step):
            stop = min(total, start + step)
            block = np.asarray(masks_arr[start:stop], dtype=np.uint8)
            if block.ndim != 4 or int(block.shape[1]) < 2:
                return None
            area = np.sum((block[:, :2] > 0).reshape(block.shape[0], 2, -1), axis=2, dtype=np.int64)
            out[start:stop, :] = np.asarray(area, dtype=np.float32)
        return out

    metrics = refined.get("metrics")
    if not isinstance(metrics, zarr.Group):
        return None
    area_arr = metrics.get("area_refined")
    if area_arr is None:
        return None
    area_refined = np.asarray(area_arr[:], dtype=np.float32)
    if area_refined.ndim != 2 or area_refined.shape[1] < 2:
        return None
    return area_refined[:, :2]


def _compute_success_mask(
    ellipse_success: np.ndarray,
    eye_separation: np.ndarray,
    min_sep: Optional[float],
    max_sep: Optional[float],
    area_refined: Optional[np.ndarray],
    min_eye_area_px: Optional[float],
) -> np.ndarray:
    pair_success = np.all(ellipse_success, axis=1)
    sep_ok = np.ones_like(pair_success, dtype=bool)
    if eye_separation.size:
        sep_ok = np.isfinite(eye_separation)
        if min_sep is not None:
            sep_ok &= eye_separation >= float(min_sep)
        if max_sep is not None:
            sep_ok &= eye_separation <= float(max_sep)

    area_ok = np.ones_like(pair_success, dtype=bool)
    threshold = _positive_float(min_eye_area_px) if min_eye_area_px is not None else None
    if threshold is not None and area_refined is not None and area_refined.shape[0] == pair_success.shape[0]:
        finite = np.all(np.isfinite(area_refined[:, :2]), axis=1)
        area_ok = finite & np.all(area_refined[:, :2] >= float(threshold), axis=1)
    return pair_success & sep_ok & area_ok


def _load_failure_indices(
    refined: zarr.Group,
    min_sep: Optional[float],
    max_sep: Optional[float],
) -> np.ndarray:
    ellipse_success = np.asarray(refined["ellipse_success"][:], dtype=bool)
    eye_separation = np.asarray(refined["eye_separation"][:], dtype=np.float32)
    area_refined = _load_area_refined(refined)
    min_eye_area_px = _resolve_success_min_eye_area_px(refined)
    success_mask = _compute_success_mask(
        ellipse_success,
        eye_separation,
        min_sep,
        max_sep,
        area_refined,
        min_eye_area_px,
    )
    return np.where(~success_mask)[0].astype("i4", copy=False)


def _get_reason_array(refined: zarr.Group) -> Optional[zarr.Array]:
    metrics = refined.get("metrics")
    if isinstance(metrics, zarr.Group) and "reason" in metrics:
        return metrics["reason"]
    return None


def _ensure_retune_id_array(refined: zarr.Group, chunks: Sequence[int]) -> zarr.Array:
    if "retune_id" in refined:
        return refined["retune_id"]
    total_rois = refined["masks_roi"].shape[0]
    return refined.create_array(
        "retune_id",
        shape=(total_rois,),
        chunks=chunks,
        dtype="i4",
        fill_value=-1,
        overwrite=True,
    )


def _get_or_create_retune_id(refined: zarr.Group, params: Dict[str, Any]) -> int:
    existing = refined.attrs.get("retune_params")
    retune_params = existing if isinstance(existing, dict) else {}

    def signature(values: Dict[str, Any]) -> tuple:
        return tuple(sorted(values.items()))

    target = signature(params)
    for key, value in retune_params.items():
        if isinstance(value, dict) and signature(value) == target:
            try:
                return int(key)
            except ValueError:
                continue

    existing_ids = [int(k) for k in retune_params.keys() if str(k).isdigit()]
    next_id = max(existing_ids, default=0) + 1
    retune_params[str(next_id)] = params
    refined.attrs["retune_params"] = retune_params
    return next_id


def _merge_reason(existing: str, tags: Sequence[str]) -> str:
    existing_tags = [tag for tag in existing.split("|") if tag]
    merged = sorted(set(existing_tags + list(tags)))
    return "|".join(merged) if merged else "clean"


def _sanitize_reason_array(reason_arr: zarr.Array) -> None:
    try:
        raw = reason_arr[:]
    except Exception:
        return
    if raw.size == 0:
        return

    def coerce(val: Any) -> str:
        if val is None:
            return ""
        if isinstance(val, np.ndarray):
            if val.size == 0:
                return ""
            if val.size == 1:
                return str(val.item())
            return "|".join(str(item) for item in val.tolist())
        return str(val)

    cleaned = np.array([coerce(v) for v in raw], dtype=object)
    reason_arr[:] = cleaned


def _update_contour_arrays(
    refined: zarr.Group,
    roi_idx: int,
    contour: Optional[np.ndarray],
    *,
    side: str,
) -> None:
    ptr_name = f"contour_{side}_ptr"
    len_name = f"contour_{side}_len"
    cont_name = f"contours_{side}"

    if ptr_name not in refined or len_name not in refined or cont_name not in refined:
        return

    ptr_arr = refined[ptr_name]
    len_arr = refined[len_name]
    cont_arr = refined[cont_name]

    if contour is None or contour.size == 0:
        ptr_arr[roi_idx] = -1
        len_arr[roi_idx] = 0
        return

    contour = np.asarray(contour, dtype=np.float32)
    n_points = contour.shape[0]
    existing_ptr = int(ptr_arr[roi_idx])
    existing_len = int(len_arr[roi_idx])

    if existing_ptr >= 0 and existing_len >= n_points:
        cont_arr[existing_ptr:existing_ptr + n_points] = contour
        len_arr[roi_idx] = n_points
        return

    current_len = cont_arr.shape[0]
    new_len = current_len + n_points
    try:
        cont_arr.resize((new_len, 2))
    except Exception:
        old = cont_arr[:]
        del refined[cont_name]
        cont_arr = refined.create_array(
            cont_name,
            shape=(new_len, 2),
            chunks=(max(1, min(4096, new_len)), 2),
            dtype=old.dtype,
            overwrite=True,
        )
        cont_arr[:old.shape[0]] = old
    cont_arr[current_len:new_len] = contour
    ptr_arr[roi_idx] = current_len
    len_arr[roi_idx] = n_points


def _build_config_from_params(
    *,
    roi_padding: int,
    pre_threshold: Optional[int],
    sobel_strength: float,
    min_area: int,
    max_area: Optional[int],
    min_circularity: Optional[float],
    closing_radius: int,
    opening_radius: int,
    min_eye_separation: Optional[float],
    max_eye_separation: Optional[float],
) -> EyeSegmentationConfig:
    return EyeSegmentationConfig(
        roi_padding=int(roi_padding),
        pre_threshold=int(pre_threshold) if pre_threshold is not None else None,
        sobel_strength=float(sobel_strength),
        min_area=int(min_area),
        max_area=int(max_area) if max_area is not None else None,
        min_circularity=float(min_circularity) if min_circularity is not None else None,
        closing_radius=int(closing_radius),
        opening_radius=int(opening_radius),
        min_eye_separation=float(min_eye_separation) if min_eye_separation is not None else None,
        max_eye_separation=float(max_eye_separation) if max_eye_separation is not None else None,
    )


def run_tuner(args: argparse.Namespace) -> None:
    zarr_path = Path(args.zarr_path)
    if not zarr_path.exists():
        raise FileNotFoundError(zarr_path)

    root = zarr.open(str(zarr_path), mode="r", use_consolidated=False)

    crop_runs = root.get("crop_runs", None)
    if crop_runs is None or "latest" not in crop_runs.attrs:
        raise RuntimeError("No crop_runs found; run crop stage first")
    crop_run = args.crop_run or crop_runs.attrs["latest"]
    crop_group = crop_runs[crop_run]

    refined_keypoints = root.get("refined_keypoints_runs", None)
    keypoint_runs = root.get("keypoints_runs", None)

    keypoint_run = args.keypoint_run
    kp_group = None
    kp_source = None

    if keypoint_run:
        if refined_keypoints is not None and keypoint_run in refined_keypoints:
            kp_group = refined_keypoints[keypoint_run]
            kp_source = "refined_keypoints_runs"
        elif keypoint_runs is not None and keypoint_run in keypoint_runs:
            kp_group = keypoint_runs[keypoint_run]
            kp_source = "keypoints_runs"
        else:
            raise RuntimeError(f"Keypoint run '{keypoint_run}' not found in refined or raw runs.")
    else:
        refined_latest = refined_keypoints.attrs.get("latest") if refined_keypoints is not None else None
        raw_latest = keypoint_runs.attrs.get("latest") if keypoint_runs is not None else None

        if refined_keypoints is not None and refined_latest in refined_keypoints:
            keypoint_run = refined_latest
            kp_group = refined_keypoints[keypoint_run]
            kp_source = "refined_keypoints_runs"
        elif keypoint_runs is not None and raw_latest in keypoint_runs:
            keypoint_run = raw_latest
            kp_group = keypoint_runs[keypoint_run]
            kp_source = "keypoints_runs"
        else:
            raise RuntimeError("No keypoint runs found; run keypoints stage first")

    roi_images = crop_group["roi_images"]
    keypoints = kp_group["keypoints_roi"][:]
    head_keypoint_indices = _resolve_required_keypoints(kp_group, _HEAD_KEYPOINT_LABELS)
    eye_keypoint_indices = _resolve_required_keypoints(kp_group, _EYE_KEYPOINT_LABELS)
    if "refined_success" in kp_group:
        success = kp_group["refined_success"][:]
    elif "detection_success" in kp_group:
        success = kp_group["detection_success"][:]
    elif "source_success" in kp_group:
        success = kp_group["source_success"][:]
    else:
        success = np.ones(keypoints.shape[0], dtype=bool)
    heading_vals = kp_group["heading"][:] if "heading" in kp_group else None

    total_rois = roi_images.shape[0]
    if total_rois == 0:
        raise RuntimeError("No ROIs available in crop run")

    # Create two windows: one for main display, one for debug panels
    main_window = "Eye Mask Tuner - Main"
    debug_window = "Eye Mask Tuner - Threshold Debug"
    cv2.namedWindow(main_window, cv2.WINDOW_NORMAL)
    cv2.namedWindow(debug_window, cv2.WINDOW_NORMAL)

    def nothing(val: int) -> None:
        pass

    cv2.createTrackbar("ROI Index", main_window, 0, total_rois - 1, nothing)
    cv2.createTrackbar("ROI Padding", main_window, min(12, SLIDER_MAX_PADDING), SLIDER_MAX_PADDING, nothing)
    cv2.createTrackbar("PreThresh", main_window, 0, SLIDER_MAX_PRETHRESH, nothing)
    cv2.createTrackbar("Sobel %", main_window, 0, SLIDER_MAX_SOBEL, nothing)
    cv2.createTrackbar("Min Area", main_window, 15, SLIDER_MAX_AREA, nothing)
    cv2.createTrackbar("Max Area", main_window, 0, SLIDER_MAX_AREA, nothing)  # 0 => None
    cv2.createTrackbar("Min Circ %", main_window, 0, SLIDER_MAX_CIRCULARITY, nothing)  # 0 => None
    cv2.createTrackbar("Closing r", main_window, 3, SLIDER_MAX_RADIUS, nothing)
    cv2.createTrackbar("Opening r", main_window, 1, SLIDER_MAX_RADIUS, nothing)
    cv2.createTrackbar("Min Gap", main_window, 4, SLIDER_MAX_EYE_GAP, nothing)
    cv2.createTrackbar("Max Gap", main_window, 0, SLIDER_MAX_EYE_GAP, nothing)  # 0 => unlimited
    cv2.createTrackbar("Rotate ROI", main_window, 1, 1, nothing)

    roi_idx = max(0, min(total_rois - 1, args.roi_index))
    cv2.setTrackbarPos("ROI Index", main_window, roi_idx)

    print("\n=== Eye Mask Tuner ===")
    print(f"Keypoints source: {kp_source}/{keypoint_run}")
    print("Main window: Shows final segmentation result")
    print("Debug window: Shows threshold processing steps")
    print("Controls:")
    print("  n/p: Next/Previous ROI")
    print("  s: Save current parameters to Zarr metadata")
    print("  q/ESC: Quit")
    print("  Rotate ROI: Align heading to 0° for tuner-only preview")
    print("  Min Gap / Max Gap: enforce eye-center separation bounds (Max Gap=0 disables upper limit)")
    print("  Min Circ %: reject non-circular connected components (0 disables)")
    print("  Sobel %: Blend Sobel edge subtraction into global thresholding (0=off)")
    print("  Axis colors: Major (cyan), Minor (red)")
    print("  Adjust other sliders to tune parameters\n")

    while True:
        roi_idx = cv2.getTrackbarPos("ROI Index", main_window)
        padding = cv2.getTrackbarPos("ROI Padding", main_window)
        pre_thresh_val = cv2.getTrackbarPos("PreThresh", main_window)
        pre_thresh = pre_thresh_val if pre_thresh_val > 0 else None
        sobel_slider = cv2.getTrackbarPos("Sobel %", main_window)
        sobel_strength = sobel_slider / float(SLIDER_MAX_SOBEL) if SLIDER_MAX_SOBEL > 0 else 0.0
        min_area = cv2.getTrackbarPos("Min Area", main_window)
        max_area_slider = cv2.getTrackbarPos("Max Area", main_window)
        max_area = max_area_slider if max_area_slider > 0 else None
        min_circularity_slider = cv2.getTrackbarPos("Min Circ %", main_window)
        min_circularity = (
            float(min_circularity_slider) / float(SLIDER_MAX_CIRCULARITY)
            if min_circularity_slider > 0 and SLIDER_MAX_CIRCULARITY > 0
            else None
        )
        closing = cv2.getTrackbarPos("Closing r", main_window)
        opening = cv2.getTrackbarPos("Opening r", main_window)
        min_gap_slider = cv2.getTrackbarPos("Min Gap", main_window)
        max_gap_slider = cv2.getTrackbarPos("Max Gap", main_window)
        min_gap = float(min_gap_slider)
        max_gap = float(max_gap_slider) if max_gap_slider > 0 else None

        roi_img = np.asarray(roi_images[roi_idx])
        kp = np.asarray(keypoints[roi_idx])
        success_flag = success[roi_idx]
        rotate_roi = cv2.getTrackbarPos("Rotate ROI", main_window) > 0
        heading_deg = None
        if heading_vals is not None:
            heading_val = float(heading_vals[roi_idx])
            if np.isfinite(heading_val):
                heading_deg = heading_val
        if heading_deg is None:
            heading_deg = compute_heading_deg(kp, head_keypoint_indices)
        if rotate_roi and heading_deg is not None:
            roi_img, kp = rotate_image_and_points(roi_img, kp, -heading_deg)

        # Main display
        display = cv2.cvtColor(roi_img.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        info_lines = [f"ROI {roi_idx+1}/{total_rois}"]
        info_lines.append(
            f"Threshold: Otsu, PreThresh: {pre_thresh or 'None'} (keep DARKER), Sobel: {sobel_strength:.2f}"
        )
        info_lines.append("Mask: Original Seg | Axes: Moment")
        if rotate_roi and heading_deg is not None:
            info_lines.append(f"Rotation: {heading_deg:.1f}° -> 0°")
        elif rotate_roi:
            info_lines.append("Rotation: on (heading missing)")
        if max_gap is None:
            info_lines.append(f"Gap limits: min≥{min_gap:.1f}px | max=None")
        else:
            info_lines.append(f"Gap limits: min≥{min_gap:.1f}px | max≤{max_gap:.1f}px")
        info_lines.append(
            f"Min circularity: {min_circularity:.2f}" if min_circularity is not None else "Min circularity: None"
        )

        # Debug panels list
        debug_panels = []

        if success_flag:
            masks = [None, None]
            contours = [None, None]
            regions_info = [None, None]
            eye_labels = ["Left", "Right"]
            eye_centers_roi = [None, None]

            for eye_idx, label in enumerate(_EYE_KEYPOINT_LABELS):
                center = kp[int(eye_keypoint_indices[label])]
                cx, cy = float(center[0]), float(center[1])
                if not np.isfinite(cx) or not np.isfinite(cy):
                    info_lines.append(f"{eye_labels[eye_idx]}: keypoint missing")
                    continue

                roi_h, roi_w = roi_img.shape
                x0 = max(0, int(round(cx)) - padding)
                x1 = min(roi_w, int(round(cx)) + padding + 1)
                y0 = max(0, int(round(cy)) - padding)
                y1 = min(roi_h, int(round(cy)) + padding + 1)

                patch = roi_img[y0:y1, x0:x1]
                if patch.size == 0:
                    info_lines.append(f"{eye_labels[eye_idx]}: patch empty")
                    continue

                filtered_patch, sobel_panel = apply_sobel_filter(patch, sobel_strength)

                binary, threshold_value = global_mask(filtered_patch)
                if pre_thresh is not None:
                    # PreThresh: keep pixels DARKER than this value (eyes are dark)
                    binary = np.logical_and(binary, filtered_patch < pre_thresh)
                
                # Get the mask to actually use
                region_mask = select_region(
                    binary,
                    (cx - x0, cy - y0),
                    min_area,
                    max_area,
                    min_circularity,
                    closing,
                    opening,
                )
                
                # Create debug panel for this eye
                debug_panel = create_debug_panel(
                    patch,
                    filtered_patch,
                    binary,
                    region_mask,
                    (cx - x0, cy - y0),
                    eye_labels[eye_idx],
                    pre_thresh,
                    sobel_panel=sobel_panel
                )
                debug_panels.append(debug_panel)
                
                if region_mask is None:
                    info_lines.append(f"{eye_labels[eye_idx]}: no region")
                    continue

                full_mask = np.zeros_like(roi_img, dtype=np.uint8)
                full_mask[y0:y1, x0:x1][region_mask] = 1
                masks[eye_idx] = full_mask

                region = measure.regionprops(region_mask.astype(int))[0]

                # Store region info for axis drawing (in full ROI coordinates)
                regions_info[eye_idx] = {
                    'centroid': (region.centroid[0] + y0, region.centroid[1] + x0),
                    'orientation': region.orientation,
                    'major_axis_length': region.major_axis_length,
                    'minor_axis_length': region.minor_axis_length,
                }
                eye_centers_roi[eye_idx] = (
                    float(region.centroid[1] + x0),
                    float(region.centroid[0] + y0),
                )
                
                info_line = (
                    f"{eye_labels[eye_idx]}: area={region.area:.0f} "
                    f"major={region.major_axis_length:.1f} minor={region.minor_axis_length:.1f} "
                    f"thr={threshold_value:.1f}"
                )
                perimeter = float(region.perimeter)
                circularity = (
                    float((4.0 * np.pi * float(region.area)) / (perimeter * perimeter))
                    if perimeter > 0.0
                    else float("nan")
                )
                if np.isfinite(circularity):
                    info_line += f" circ={circularity:.2f}"
                info_lines.append(info_line)

                contour = measure.find_contours(region_mask.astype(float), 0.5)
                if contour:
                    best = max(contour, key=lambda c: c.shape[0])
                    if best.shape[0] >= 5:
                        best = best[:, ::-1]
                        best[:, 0] += x0
                        best[:, 1] += y0
                        contours[eye_idx] = best

            if all(c is not None for c in eye_centers_roi):
                separation = float(np.hypot(
                    eye_centers_roi[0][0] - eye_centers_roi[1][0],
                    eye_centers_roi[0][1] - eye_centers_roi[1][1]
                ))
                info_lines.append(f"Eye separation: {separation:.1f}px")
            display = draw_overlay(roi_img, tuple(masks), tuple(contours), tuple(regions_info))
        else:
            info_lines.append("Keypoints failed for this ROI")
            display = draw_overlay(roi_img, (None, None), (None, None), (None, None))

        if rotate_roi:
            display = apply_circular_window(display)

        # Build info panel
        info_width = max(700, int(display.shape[1] * 0.5))
        info_panel = np.full((display.shape[0], info_width, 3), 240, dtype=np.uint8)
        for idx, line in enumerate(info_lines):
            cv2.putText(info_panel, line, (18, 30 + idx * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

        # Add spacer between info panel and display
        spacer_width = 30  # pixels of space between panels
        spacer = np.full((display.shape[0], spacer_width, 3), 200, dtype=np.uint8)
        combined_display = np.hstack([info_panel, spacer, display])

        cv2.imshow(main_window, combined_display)
        
        # Show debug panels
        if debug_panels:
            if len(debug_panels) == 2:
                # Stack both eye panels vertically
                debug_display = np.vstack(debug_panels)
            else:
                debug_display = debug_panels[0]
            cv2.imshow(debug_window, debug_display)
        else:
            # Show empty debug window
            empty = np.zeros((100, 400, 3), dtype=np.uint8)
            cv2.putText(empty, "No valid keypoints for this ROI", (10, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.imshow(debug_window, empty)

        key = cv2.waitKey(30) & 0xFF

        if key in (ord("q"), 27):
            break
        elif key == ord("s"):
            tuned_parameters = {
                'roi_padding': int(padding),
                'pre_threshold': int(pre_thresh) if pre_thresh is not None else None,
                'sobel_strength': float(sobel_strength),
                'min_area': int(min_area),
                'max_area': int(max_area) if max_area is not None else None,
                'min_circularity': float(min_circularity) if min_circularity is not None else None,
                'closing_radius': int(closing),
                'opening_radius': int(opening),
                'min_eye_separation': float(min_gap) if min_gap > 0 else None,
                'max_eye_separation': float(max_gap) if max_gap is not None and max_gap > 0 else None,
            }
            context = {
                'roi_index': int(roi_idx),
                'roi_index_one_based': int(roi_idx + 1),
                'total_rois': int(total_rois),
                'crop_run': crop_run,
                'keypoint_run': keypoint_run,
                'keypoint_source': kp_source,
                'roi_success': bool(success_flag),
            }
            success_save, message = save_eye_mask_params(
                zarr_path,
                {
                    'tuned_parameters': tuned_parameters,
                    'context': context,
                },
            )
            if not success_save:
                print(f"✗ {message}")
        elif key == ord("n"):
            cv2.setTrackbarPos("ROI Index", main_window, min(total_rois - 1, roi_idx + 1))
        elif key == ord("p"):
            cv2.setTrackbarPos("ROI Index", main_window, max(0, roi_idx - 1))

    cv2.destroyAllWindows()


def run_failure_tuner(
    zarr_path: str,
    refined_run: str,
    start_failure: int = 1,
    *,
    apply_batch_size: int = 128,
    apply_workers: int = 4,
) -> None:
    root = zarr.open(str(zarr_path), mode="a", use_consolidated=False)
    refined_parent = root.get("refined_eye_masks_runs")
    if refined_parent is None or refined_run not in refined_parent:
        raise RuntimeError(f"Refined eye mask run '{refined_run}' not found.")
    refined = refined_parent[refined_run]

    crop_run = refined.attrs.get("source_crop_run") or root["crop_runs"].attrs.get("latest")
    if not crop_run or "crop_runs" not in root or crop_run not in root["crop_runs"]:
        raise RuntimeError("Crop run not found for eye mask retune.")
    crop_group = root["crop_runs"][crop_run]

    keypoint_run_name = refined.attrs.get("source_keypoints_run")
    keypoint_group_name = refined.attrs.get("source_keypoint_group")
    if (
        keypoint_group_name
        and keypoint_run_name
        and keypoint_group_name in root
        and keypoint_run_name in root[keypoint_group_name]
    ):
        kp_group = root[keypoint_group_name][keypoint_run_name]
        kp_run = keypoint_run_name
        kp_group_name = keypoint_group_name
    else:
        kp_group, kp_run = _resolve_keypoints_group(root, keypoint_run_name)
        kp_group_name = "refined_keypoints_runs" if "refined_keypoints_runs" in root and kp_run in root["refined_keypoints_runs"] else "keypoints_runs"

    roi_images = crop_group["roi_images"]
    keypoints = kp_group["keypoints_roi"][:]
    eye_keypoint_indices = _resolve_required_keypoints(kp_group, _EYE_KEYPOINT_LABELS)
    if "refined_success" in kp_group:
        success_flags = kp_group["refined_success"][:]
    elif "detection_success" in kp_group:
        success_flags = kp_group["detection_success"][:]
    elif "source_success" in kp_group:
        success_flags = kp_group["source_success"][:]
    else:
        success_flags = np.ones(keypoints.shape[0], dtype=bool)

    masks_arr = refined["masks_roi"]
    ellipse_params_arr = refined["ellipse_params"]
    ellipse_success_arr = refined["ellipse_success"]
    eye_separation_arr = refined["eye_separation"]

    min_sep, max_sep = _get_sep_limits(root, refined)
    failures = _load_failure_indices(refined, min_sep, max_sep)
    if failures.size == 0:
        print("No failed eye masks to retune.")
        return

    reason_arr = _get_reason_array(refined)
    if reason_arr is not None:
        _sanitize_reason_array(reason_arr)

    retune_id_arr = _ensure_retune_id_array(refined, (min(1024, roi_images.shape[0]),))

    tuning = None
    analysis = root.get("analysis_metadata")
    if analysis is not None:
        tuning = analysis.attrs.get("eye_mask_tuning")
    tuned_params = tuning.get("tuned_parameters", {}) if isinstance(tuning, dict) else {}

    cfg_defaults = EyeSegmentationConfig()
    roi_padding_default = int(tuned_params.get("roi_padding", cfg_defaults.roi_padding))
    pre_thresh_default = tuned_params.get("pre_threshold", cfg_defaults.pre_threshold)
    sobel_default = float(tuned_params.get("sobel_strength", cfg_defaults.sobel_strength))
    min_area_default = int(tuned_params.get("min_area", cfg_defaults.min_area))
    max_area_default = tuned_params.get("max_area", cfg_defaults.max_area)
    min_circularity_default = tuned_params.get("min_circularity", cfg_defaults.min_circularity)
    closing_default = int(tuned_params.get("closing_radius", cfg_defaults.closing_radius))
    opening_default = int(tuned_params.get("opening_radius", cfg_defaults.opening_radius))
    min_gap_default = tuned_params.get("min_eye_separation", cfg_defaults.min_eye_separation)
    max_gap_default = tuned_params.get("max_eye_separation", cfg_defaults.max_eye_separation)

    window_name = "Eye Mask Failure Retune"
    debug_window = "Eye Mask Failure Retune - Debug"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.namedWindow(debug_window, cv2.WINDOW_NORMAL)

    current_failure = max(1, min(start_failure, len(failures)))

    def update_failure(val: int) -> None:
        nonlocal current_failure
        if len(failures) == 0:
            current_failure = 1
            return
        current_failure = max(1, min(val, len(failures)))

    cv2.createTrackbar("Failure", window_name, current_failure, max(1, len(failures)), update_failure)
    cv2.createTrackbar("ROI Padding", window_name, min(roi_padding_default, SLIDER_MAX_PADDING), SLIDER_MAX_PADDING, nothing)
    cv2.createTrackbar("PreThresh", window_name, int(pre_thresh_default) if pre_thresh_default is not None else 0, SLIDER_MAX_PRETHRESH, nothing)
    cv2.createTrackbar("Sobel %", window_name, int(sobel_default * SLIDER_MAX_SOBEL), SLIDER_MAX_SOBEL, nothing)
    cv2.createTrackbar("Min Area", window_name, min(min_area_default, SLIDER_MAX_AREA), SLIDER_MAX_AREA, nothing)
    cv2.createTrackbar("Max Area", window_name, int(max_area_default) if max_area_default is not None else 0, SLIDER_MAX_AREA, nothing)
    cv2.createTrackbar(
        "Min Circ %",
        window_name,
        int(round(float(min_circularity_default) * SLIDER_MAX_CIRCULARITY)) if min_circularity_default is not None else 0,
        SLIDER_MAX_CIRCULARITY,
        nothing,
    )
    cv2.createTrackbar("Closing r", window_name, min(closing_default, SLIDER_MAX_RADIUS), SLIDER_MAX_RADIUS, nothing)
    cv2.createTrackbar("Opening r", window_name, min(opening_default, SLIDER_MAX_RADIUS), SLIDER_MAX_RADIUS, nothing)
    cv2.createTrackbar("Min Gap", window_name, int(min_gap_default) if min_gap_default is not None else 0, SLIDER_MAX_EYE_GAP, nothing)
    cv2.createTrackbar("Max Gap", window_name, int(max_gap_default) if max_gap_default is not None else 0, SLIDER_MAX_EYE_GAP, nothing)

    print("\nKeypoint Failure Retune (Eye Masks)")
    print(f"  Zarr: {zarr_path}")
    print(f"  Refined run: {refined_run}")
    print(f"  Crop run: {crop_run}")
    print(f"  Keypoint run: {kp_group_name}/{kp_run}")
    print(f"  Failures to retune: {len(failures)}")
    print(f"  Apply batch: {apply_batch_size} | Workers: {apply_workers}")
    print("Controls:")
    print("  Arrow keys: Navigate failures")
    print("  e: Quick eval on a sample of remaining failures")
    print("  E: Eval all remaining failures (slow)")
    print("  a: Apply params to remaining failures")
    print("  Min Circ %: reject non-circular components (0 disables)")
    print("  q/ESC: Quit")

    def _current_params() -> Dict[str, Any]:
        padding = cv2.getTrackbarPos("ROI Padding", window_name)
        pre_thresh_val = cv2.getTrackbarPos("PreThresh", window_name)
        pre_thresh = pre_thresh_val if pre_thresh_val > 0 else None
        sobel_slider = cv2.getTrackbarPos("Sobel %", window_name)
        sobel_strength = sobel_slider / float(SLIDER_MAX_SOBEL) if SLIDER_MAX_SOBEL > 0 else 0.0
        min_area = cv2.getTrackbarPos("Min Area", window_name)
        max_area_slider = cv2.getTrackbarPos("Max Area", window_name)
        max_area = max_area_slider if max_area_slider > 0 else None
        min_circularity_slider = cv2.getTrackbarPos("Min Circ %", window_name)
        min_circularity = (
            float(min_circularity_slider) / float(SLIDER_MAX_CIRCULARITY)
            if min_circularity_slider > 0 and SLIDER_MAX_CIRCULARITY > 0
            else None
        )
        closing = cv2.getTrackbarPos("Closing r", window_name)
        opening = cv2.getTrackbarPos("Opening r", window_name)
        min_gap_slider = cv2.getTrackbarPos("Min Gap", window_name)
        max_gap_slider = cv2.getTrackbarPos("Max Gap", window_name)
        min_gap = float(min_gap_slider) if min_gap_slider > 0 else None
        max_gap = float(max_gap_slider) if max_gap_slider > 0 else None
        return {
            "roi_padding": padding,
            "pre_threshold": pre_thresh,
            "sobel_strength": sobel_strength,
            "min_area": min_area,
            "max_area": max_area,
            "min_circularity": min_circularity,
            "closing_radius": closing,
            "opening_radius": opening,
            "min_eye_separation": min_gap,
            "max_eye_separation": max_gap,
        }

    def _eval_failures(sample_limit: Optional[int]) -> None:
        if len(failures) == 0:
            print("No failures remaining.")
            return
        params = _current_params()
        cfg = _build_config_from_params(**params)
        if sample_limit is not None and len(failures) > sample_limit:
            rng = np.random.default_rng(0)
            sample = rng.choice(failures, size=sample_limit, replace=False)
        else:
            sample = failures
        success = 0
        for idx in sample:
            roi_img = np.asarray(roi_images[idx])
            kp = np.asarray(keypoints[idx])
            success_flag = bool(success_flags[idx])
            result = _process_roi_data(int(idx), roi_img, kp, success_flag, cfg)
            if result.get("reject_reason") is None:
                success += 1
        total = len(sample)
        rate = (success / total * 100.0) if total else 0.0
        label = f"sample {total}/{len(failures)}" if sample_limit is not None else f"all {total}"
        print(f"Eval ({label}): {success}/{total} would pass ({rate:.1f}%)")

    def apply_params() -> None:
        nonlocal failures
        if len(failures) == 0:
            print("No failures remaining.")
            return
        params = _current_params()
        cfg = _build_config_from_params(**params)
        retune_id = _get_or_create_retune_id(refined, params)

        updated = 0
        total = len(failures)

        def process_one(idx: int) -> tuple[int, Dict[str, Any]]:
            roi_img = np.asarray(roi_images[idx])
            kp = np.asarray(keypoints[idx])
            success_flag = bool(success_flags[idx])
            return idx, _process_roi_data(int(idx), roi_img, kp, success_flag, cfg)

        batch_size = max(1, int(apply_batch_size))
        batches = [
            failures[i:i + batch_size]
            for i in range(0, len(failures), batch_size)
        ]

        for batch_idx, batch in enumerate(batches, start=1):
            if apply_workers and apply_workers > 1:
                with ThreadPoolExecutor(max_workers=apply_workers) as executor:
                    results = list(executor.map(process_one, batch))
            else:
                results = [process_one(idx) for idx in batch]

            for roi_idx, result in results:
                if result.get("reject_reason") is not None:
                    continue
                masks = result["masks"]
                ellipse_params = result["ellipse_params"]
                ellipse_success = result["ellipse_success"]
                contours = result["contours"]
                separation = result.get("eye_separation", np.nan)

                masks_arr[roi_idx, 0] = masks[0]
                masks_arr[roi_idx, 1] = masks[1]
                ellipse_params_arr[roi_idx, 0] = ellipse_params[0]
                ellipse_params_arr[roi_idx, 1] = ellipse_params[1]
                ellipse_success_arr[roi_idx, 0] = ellipse_success[0]
                ellipse_success_arr[roi_idx, 1] = ellipse_success[1]
                eye_separation_arr[roi_idx] = separation

                _update_contour_arrays(refined, roi_idx, contours[0], side="left")
                _update_contour_arrays(refined, roi_idx, contours[1], side="right")

                retune_id_arr[roi_idx] = retune_id
                if reason_arr is not None:
                    existing = "" if reason_arr[roi_idx] is None else str(reason_arr[roi_idx])
                    reason_value = _merge_reason(existing, ["retuned"])
                    reason_arr[roi_idx:roi_idx + 1] = np.array([reason_value], dtype=object)

                updated += 1

            if len(batches) > 1:
                print(f"  Batch {batch_idx}/{len(batches)} processed ({updated}/{total} updated)")

        min_sep_local = params["min_eye_separation"]
        max_sep_local = params["max_eye_separation"]
        failures = _load_failure_indices(refined, min_sep_local, max_sep_local)
        remaining = len(failures)
        rate = (updated / total * 100.0) if total else 0.0
        print(f"Applied retune {retune_id}: {updated}/{total} updated ({rate:.1f}%)")
        print(f"Remaining failures: {remaining}")
        new_max = max(1, remaining)
        cv2.setTrackbarMax("Failure", window_name, new_max)
        current_failure = min(current_failure, remaining if remaining > 0 else 1)
        cv2.setTrackbarPos("Failure", window_name, current_failure)

    while True:
        if len(failures) == 0:
            print("No failures remaining.")
            break

        failure_pos = max(1, min(current_failure, len(failures))) - 1
        roi_idx = int(failures[failure_pos])

        params = _current_params()
        cfg = _build_config_from_params(**params)

        roi_img = np.asarray(roi_images[roi_idx])
        kp = np.asarray(keypoints[roi_idx])
        success_flag = bool(success_flags[roi_idx])

        masks = [None, None]
        contours = [None, None]
        regions_info = [None, None]
        eye_labels = ["Left", "Right"]
        debug_panels: List[np.ndarray] = []

        info_lines = [f"Failure {failure_pos + 1}/{len(failures)} | ROI {roi_idx}"]
        info_lines.append(
            f"PreThresh: {params['pre_threshold'] or 'None'} | Sobel: {params['sobel_strength']:.2f}"
        )
        info_lines.append(
            (
                f"Min circularity: {params['min_circularity']:.2f}"
                if params["min_circularity"] is not None
                else "Min circularity: None"
            )
        )

        if success_flag:
            for eye_idx, label in enumerate(_EYE_KEYPOINT_LABELS):
                center = kp[int(eye_keypoint_indices[label])]
                cx, cy = float(center[0]), float(center[1])
                if not np.isfinite(cx) or not np.isfinite(cy):
                    continue

                roi_h, roi_w = roi_img.shape
                x0 = max(0, int(round(cx)) - cfg.roi_padding)
                x1 = min(roi_w, int(round(cx)) + cfg.roi_padding + 1)
                y0 = max(0, int(round(cy)) - cfg.roi_padding)
                y1 = min(roi_h, int(round(cy)) + cfg.roi_padding + 1)

                patch = roi_img[y0:y1, x0:x1]
                if patch.size == 0:
                    continue

                filtered_patch, sobel_panel = apply_sobel_filter(patch, cfg.sobel_strength)
                binary, threshold_value = global_mask(filtered_patch)
                if cfg.pre_threshold is not None:
                    binary = np.logical_and(binary, filtered_patch < cfg.pre_threshold)

                region_mask = select_region(
                    binary,
                    (cx - x0, cy - y0),
                    cfg.min_area,
                    cfg.max_area,
                    cfg.min_circularity,
                    cfg.closing_radius,
                    cfg.opening_radius,
                )

                debug_panel = create_debug_panel(
                    patch,
                    filtered_patch,
                    binary,
                    region_mask,
                    (cx - x0, cy - y0),
                    eye_labels[eye_idx],
                    cfg.pre_threshold,
                    sobel_panel=sobel_panel
                )
                debug_panels.append(debug_panel)

                if region_mask is None:
                    continue

                full_mask = np.zeros_like(roi_img, dtype=np.uint8)
                full_mask[y0:y1, x0:x1][region_mask] = 1
                masks[eye_idx] = full_mask

                region = measure.regionprops(region_mask.astype(int))[0]
                regions_info[eye_idx] = {
                    'centroid': (region.centroid[0] + y0, region.centroid[1] + x0),
                    'orientation': region.orientation,
                    'major_axis_length': region.major_axis_length,
                    'minor_axis_length': region.minor_axis_length,
                }

                contour = measure.find_contours(region_mask.astype(float), 0.5)
                if contour:
                    best = max(contour, key=lambda c: c.shape[0])
                    if best.shape[0] >= 5:
                        best = best[:, ::-1]
                        best[:, 0] += x0
                        best[:, 1] += y0
                        contours[eye_idx] = best

            display = draw_overlay(roi_img, tuple(masks), tuple(contours), tuple(regions_info))
        else:
            info_lines.append("Keypoints failed for this ROI")
            display = draw_overlay(roi_img, (None, None), (None, None), (None, None))
            debug_panels = []

        info_width = max(500, int(display.shape[1] * 0.5))
        info_panel = np.full((display.shape[0], info_width, 3), 240, dtype=np.uint8)
        for idx, line in enumerate(info_lines):
            cv2.putText(info_panel, line, (18, 30 + idx * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
        spacer = np.full((display.shape[0], 30, 3), 200, dtype=np.uint8)
        combined_display = np.hstack([info_panel, spacer, display])
        cv2.imshow(window_name, combined_display)

        if debug_panels:
            if len(debug_panels) == 2:
                debug_display = np.vstack(debug_panels)
            else:
                debug_display = debug_panels[0]
            cv2.imshow(debug_window, debug_display)
        else:
            empty = np.zeros((100, 400, 3), dtype=np.uint8)
            cv2.putText(
                empty,
                "No valid keypoints for this ROI",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
            )
            cv2.imshow(debug_window, empty)

        key = cv2.waitKey(30) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key == ord("e"):
            _eval_failures(min(300, len(failures)))
        elif key == ord("E"):
            _eval_failures(None)
        elif key == ord("a"):
            apply_params()
        elif key == ord("n"):
            current_failure = min(len(failures), current_failure + 1)
            cv2.setTrackbarPos("Failure", window_name, current_failure)
        elif key == ord("p"):
            current_failure = max(1, current_failure - 1)
            cv2.setTrackbarPos("Failure", window_name, current_failure)

    cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive tuner for eye segmentation parameters")
    parser.add_argument("zarr_path", help="Path to Palette Zarr store")
    parser.add_argument("--roi-index", type=int, default=0, help="Initial ROI index")
    parser.add_argument("--crop-run", help="Specific crop run name")
    parser.add_argument(
        "--keypoint-run",
        help="Specific keypoint run name (checks refined_keypoints_runs first, then keypoints_runs)",
    )
    args = parser.parse_args()
    run_tuner(args)


if __name__ == "__main__":
    main()
