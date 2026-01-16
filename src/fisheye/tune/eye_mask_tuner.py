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
from skimage import filters, measure, morphology
from skimage.measure import EllipseModel
from datetime import datetime, timezone

from ..segmentation.eye_segmentation import EyeSegmentationConfig, _process_roi_data


SLIDER_MAX_PADDING = 40
SLIDER_MAX_AREA = 500
SLIDER_MAX_RADIUS = 10
SLIDER_MAX_PRETHRESH = 255
SLIDER_MAX_EYE_GAP = 200
SLIDER_MAX_SOBEL = 100  # maps to strength in range [0, 1]
DEBUG_PANEL_SCALE = 5
DEBUG_PANEL_MARGIN = 10
DEBUG_PANEL_SPACING = 20


def save_eye_mask_params(zarr_path: Path, params: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Persist eye mask tuning parameters into the Zarr's analysis_metadata group.
    """
    try:
        root = zarr.open(str(zarr_path), mode='a')
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

def global_mask(patch: np.ndarray) -> Tuple[np.ndarray, float]:
    try:
        threshold_value = float(filters.threshold_otsu(patch))
    except ValueError:
        threshold_value = float(np.mean(patch))
    return patch < threshold_value, threshold_value  # Eyes are DARKER than background


def apply_sobel_filter(patch: np.ndarray, strength: float) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Return Sobel-enhanced patch and visualization of edge magnitude."""
    if strength <= 0.0:
        return patch, None

    patch_float = patch.astype(np.float32) / 255.0
    sobel_response = filters.sobel(patch_float)
    max_resp = float(np.max(sobel_response))
    if max_resp > 0.0:
        sobel_norm = sobel_response / max_resp
    else:
        sobel_norm = np.zeros_like(sobel_response)

    # Subtract scaled edges to further darken the eye interior.
    filtered = np.clip(patch_float - strength * sobel_norm, 0.0, 1.0)
    filtered_uint8 = (filtered * 255.0).astype(patch.dtype, copy=False)

    sobel_visual = (sobel_norm * 255.0).astype(np.uint8, copy=False)
    return filtered_uint8, sobel_visual


def compute_heading_deg(kp: np.ndarray) -> Optional[float]:
    if kp.shape[0] < 3:
        return None
    bladder = kp[0]
    eye_left = kp[1]
    eye_right = kp[2]
    if not (np.all(np.isfinite(bladder)) and np.all(np.isfinite(eye_left)) and np.all(np.isfinite(eye_right))):
        return None
    eye_mean = (eye_left + eye_right) / 2.0
    head_vec = eye_mean - bladder
    if not np.any(np.isfinite(head_vec)):
        return None
    dx = float(head_vec[0])
    dy = float(head_vec[1])
    if dx == 0.0 and dy == 0.0:
        return None
    return float(np.degrees(np.arctan2(-dy, dx)))


def rotate_image_and_points(
    image: np.ndarray,
    points: np.ndarray,
    angle_deg: float,
) -> Tuple[np.ndarray, np.ndarray]:
    h, w = image.shape[:2]
    center = (w / 2.0, h / 2.0)
    mat = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    rotated = cv2.warpAffine(
        image,
        mat,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    pts = points.astype(np.float64, copy=True)
    mask = np.all(np.isfinite(pts), axis=1)
    if np.any(mask):
        pts_xy = pts[mask]
        rotated_xy = (pts_xy @ mat[:, :2].T) + mat[:, 2]
        pts[mask] = rotated_xy
    return rotated, pts


def apply_circular_window(image: np.ndarray) -> np.ndarray:
    """Mask image to a centered circular window to hide rotation edges."""
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    radius = min(center)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, center, radius, 255, -1)
    if image.ndim == 2:
        return np.where(mask > 0, image, 0)
    masked = image.copy()
    masked[mask == 0] = 0
    return masked


def select_region(
    mask: np.ndarray,
    center: Tuple[float, float],
    min_area: int,
    max_area: Optional[int],
    closing: int,
    opening: int,
) -> Optional[np.ndarray]:
    if closing > 0:
        mask = morphology.binary_closing(mask, morphology.disk(closing))
    if opening > 0:
        mask = morphology.binary_opening(mask, morphology.disk(opening))

    labeled = measure.label(mask)
    if labeled.max() == 0:
        return None

    regions = measure.regionprops(labeled)
    if not regions:
        return None

    cx, cy = center
    best = None
    best_dist = None
    for region in regions:
        area = region.area
        if area < max(min_area, 1):
            continue
        if max_area is not None and area > max_area:
            continue
        rcx, rcy = region.centroid
        dist = (rcx - cy) ** 2 + (rcy - cx) ** 2
        if best is None or dist < best_dist:
            best = region
            best_dist = dist

    if best is None:
        return None

    region_mask = (labeled == best.label)

    return region_mask


def draw_overlay(roi_img: np.ndarray, masks: Tuple[np.ndarray, np.ndarray], contours: Tuple[Optional[np.ndarray], Optional[np.ndarray]], regions_info: Tuple[Optional[dict], Optional[dict]]) -> np.ndarray:
    base = cv2.cvtColor(roi_img.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    overlay = base.copy()

    colors = [(255, 0, 0), (0, 0, 255)]  # left blue, right red
    for idx, mask in enumerate(masks):
        if mask is None:
            continue
        color = colors[idx]
        overlay[mask > 0] = color
    output = cv2.addWeighted(overlay, 0.4, base, 0.6, 0)

    for idx, contour in enumerate(contours):
        if contour is None:
            continue
        pts = contour.astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(output, [pts], isClosed=True, color=colors[idx], thickness=1)
    
    # Draw moment-based major/minor axes
    for idx, region_info in enumerate(regions_info):
        if region_info is None:
            continue

        color = colors[idx]

        cy, cx = region_info['centroid']
        orientation = region_info['orientation']
        major_len = region_info['major_axis_length']
        minor_len = region_info['minor_axis_length']

        cos_angle = np.cos(orientation)
        sin_angle = np.sin(orientation)

        major_dx = cos_angle * major_len / 2
        major_dy = sin_angle * major_len / 2
        major_p1 = (int(cx - major_dx), int(cy - major_dy))
        major_p2 = (int(cx + major_dx), int(cy + major_dy))

        minor_dx = -sin_angle * minor_len / 2
        minor_dy = cos_angle * minor_len / 2
        minor_p1 = (int(cx - minor_dx), int(cy - minor_dy))
        minor_p2 = (int(cx + minor_dx), int(cy + minor_dy))

        cv2.line(output, major_p1, major_p2, color, 2)
        cv2.line(output, minor_p1, minor_p2, color, 1)
        cv2.circle(output, (int(cx), int(cy)), 3, color, -1)

    return output


def create_debug_panel(
    original_patch: np.ndarray,
    filtered_patch: np.ndarray,
    binary: np.ndarray,
    region_mask: Optional[np.ndarray],
    center: Tuple[float, float],
    label: str,
    pre_thresh: Optional[int],
    sobel_panel: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Create a debug panel showing all processing steps for one eye."""
    h, w = original_patch.shape

    text_height = 16  # reserve space for labels (scaled later)
    panel_total_w = w + 2 * DEBUG_PANEL_MARGIN
    panel_total_h = h + text_height + 2 * DEBUG_PANEL_MARGIN

    def to_bgr(img: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    panels: list[Tuple[str, np.ndarray]] = []

    original_panel = to_bgr(original_patch)
    cx, cy = center
    if 0 <= cx < w and 0 <= cy < h:
        cv2.drawMarker(original_panel, (int(cx), int(cy)), (0, 255, 0), cv2.MARKER_CROSS, 5, 1)
    panels.append(("Original", original_panel))

    if sobel_panel is not None:
        panels.append(("Sobel (edges)", to_bgr(sobel_panel)))
        panels.append(("Edge-adjusted", to_bgr(filtered_patch)))

    if pre_thresh is None:
        pre_thresh_label = "PreThresh (off)"
        pre_thresh_display = np.zeros_like(filtered_patch, dtype=np.uint8)
    else:
        pre_thresh_label = f"PreThresh<{int(pre_thresh)}"
        pre_mask = filtered_patch < pre_thresh
        pre_thresh_display = (pre_mask.astype(np.uint8) * 255)
    panels.append((pre_thresh_label, to_bgr(pre_thresh_display)))

    binary_display = (binary.astype(np.uint8) * 255)
    if pre_thresh is not None:
        pre_mask = original_patch >= pre_thresh  # Too bright, rejected
        panel_binary = to_bgr(binary_display)
        panel_binary[pre_mask] = [0, 0, 255]
    else:
        panel_binary = to_bgr(binary_display)
    panels.append(("Binary", panel_binary))

    if region_mask is not None:
        region_display = cv2.cvtColor((region_mask.astype(np.uint8) * 255), cv2.COLOR_GRAY2BGR)
    else:
        region_display = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.putText(region_display, "NO REGION", (5, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
    panels.append(("Segmented", region_display))

    overlay_panel = to_bgr(original_patch).copy()
    major_color = (255, 255, 0)  # cyan in BGR
    minor_color = (0, 0, 255)
    if region_mask is not None:
        overlay_color = np.zeros_like(overlay_panel, dtype=np.uint8)
        overlay_color[region_mask] = (0, 255, 0)
        overlay_panel = cv2.addWeighted(overlay_panel, 1.0, overlay_color, 0.4, 0)
    else:
        cv2.putText(overlay_panel, "NO REGION", (5, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
    panels.append(("Overlay", overlay_panel))

    # Ellipse fit panel - TRUE least-squares fit to contour
    ellipse_fit_panel = to_bgr(original_patch).copy()
    ellipse_fit_label = "Ellipse Fit"

    if region_mask is not None:
        # Find contour points
        contours = measure.find_contours(region_mask.astype(float), 0.5)
        if contours:
            contour = max(contours, key=lambda c: c.shape[0])
            # contour is (row, col) = (y, x), need (x, y) for EllipseModel
            contour_xy = contour[:, ::-1]  # Convert to (x, y)

            if contour_xy.shape[0] >= 5:  # Need at least 5 points for ellipse
                ellipse_model = EllipseModel()
                success = ellipse_model.estimate(contour_xy)

                if success and ellipse_model.params is not None:
                    xc, yc, a, b, theta = ellipse_model.params
                    # Ensure major >= minor (swap if needed, rotate theta by 90°)
                    if a < b:
                        a, b = b, a
                        theta = theta + np.pi / 2

                    ellipse_fit_label = f"Ellipse θ={np.degrees(theta):.1f}°"

                    # Draw fitted ellipse outline (cv2.ellipse expects (major, minor) order)
                    axes = (max(1, int(round(a))), max(1, int(round(b))))
                    center_pt = (int(round(xc)), int(round(yc)))
                    angle_deg = float(np.degrees(theta))
                    cv2.ellipse(ellipse_fit_panel, center_pt, axes, angle_deg, 0, 360, (0, 255, 0), 1)

                    # Draw major axis (along theta direction)
                    cos_t = np.cos(theta)
                    sin_t = np.sin(theta)
                    major_dx = cos_t * a
                    major_dy = sin_t * a
                    major_p1 = (int(round(xc - major_dx)), int(round(yc - major_dy)))
                    major_p2 = (int(round(xc + major_dx)), int(round(yc + major_dy)))
                    cv2.line(ellipse_fit_panel, major_p1, major_p2, major_color, 1)

                    # Draw minor axis (perpendicular)
                    minor_dx = -sin_t * b
                    minor_dy = cos_t * b
                    minor_p1 = (int(round(xc - minor_dx)), int(round(yc - minor_dy)))
                    minor_p2 = (int(round(xc + minor_dx)), int(round(yc + minor_dy)))
                    cv2.line(ellipse_fit_panel, minor_p1, minor_p2, minor_color, 1)

                    # Center point
                    cv2.circle(ellipse_fit_panel, center_pt, 2, major_color, -1)
                else:
                    ellipse_fit_label = "Ellipse (fit failed)"
                    cv2.putText(ellipse_fit_panel, "FIT FAILED", (5, h // 2),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
            else:
                ellipse_fit_label = "Ellipse (<5 pts)"
                cv2.putText(ellipse_fit_panel, "TOO FEW PTS", (5, h // 2),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
        else:
            cv2.putText(ellipse_fit_panel, "NO CONTOUR", (5, h // 2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
    else:
        cv2.putText(ellipse_fit_panel, "NO REGION", (5, h // 2),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
    panels.append((ellipse_fit_label, ellipse_fit_panel))

    num_panels = len(panels)
    total_width = num_panels * panel_total_w + (num_panels - 1) * DEBUG_PANEL_SPACING
    debug = np.zeros((panel_total_h, total_width, 3), dtype=np.uint8)

    x_positions: list[int] = []

    def place_panel(idx: int, panel_img: np.ndarray) -> int:
        x = idx * (panel_total_w + DEBUG_PANEL_SPACING) + DEBUG_PANEL_MARGIN
        y = DEBUG_PANEL_MARGIN
        debug[y:y + h, x:x + w] = panel_img
        return x

    for idx, (_, panel_img) in enumerate(panels):
        x_positions.append(place_panel(idx, panel_img))

    text_y = max(8, DEBUG_PANEL_MARGIN - 2)
    for idx, (label_text, _) in enumerate(panels):
        cv2.putText(debug, label_text, (x_positions[idx] + 2, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)

    label_y = DEBUG_PANEL_MARGIN + h + text_height - 5
    cv2.putText(debug, f"{label} Eye", (x_positions[0] + 2, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

    if DEBUG_PANEL_SCALE != 1:
        debug = cv2.resize(
            debug,
            (0, 0),
            fx=DEBUG_PANEL_SCALE,
            fy=DEBUG_PANEL_SCALE,
            interpolation=cv2.INTER_NEAREST
        )
    return debug


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


def _compute_success_mask(
    ellipse_success: np.ndarray,
    eye_separation: np.ndarray,
    min_sep: Optional[float],
    max_sep: Optional[float],
) -> np.ndarray:
    pair_success = np.all(ellipse_success, axis=1)
    sep_ok = np.ones_like(pair_success, dtype=bool)
    if eye_separation.size:
        sep_ok = np.isfinite(eye_separation)
        if min_sep is not None:
            sep_ok &= eye_separation >= float(min_sep)
        if max_sep is not None:
            sep_ok &= eye_separation <= float(max_sep)
    return pair_success & sep_ok


def _load_failure_indices(
    refined: zarr.Group,
    min_sep: Optional[float],
    max_sep: Optional[float],
) -> np.ndarray:
    ellipse_success = np.asarray(refined["ellipse_success"][:], dtype=bool)
    eye_separation = np.asarray(refined["eye_separation"][:], dtype=np.float32)
    success_mask = _compute_success_mask(ellipse_success, eye_separation, min_sep, max_sep)
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


def _resolve_keypoints_group(
    root: zarr.Group,
    keypoint_run: Optional[str],
) -> tuple[zarr.Group, str]:
    refined_keypoints = root.get("refined_keypoints_runs")
    keypoint_runs = root.get("keypoints_runs")

    if keypoint_run:
        if refined_keypoints is not None and keypoint_run in refined_keypoints:
            return refined_keypoints[keypoint_run], keypoint_run
        if keypoint_runs is not None and keypoint_run in keypoint_runs:
            return keypoint_runs[keypoint_run], keypoint_run
        raise RuntimeError(f"Keypoint run '{keypoint_run}' not found.")

    refined_latest = refined_keypoints.attrs.get("latest") if refined_keypoints is not None else None
    raw_latest = keypoint_runs.attrs.get("latest") if keypoint_runs is not None else None

    if refined_keypoints is not None and refined_latest in refined_keypoints:
        return refined_keypoints[refined_latest], refined_latest
    if keypoint_runs is not None and raw_latest in keypoint_runs:
        return keypoint_runs[raw_latest], raw_latest
    raise RuntimeError("No keypoint runs found; run keypoints stage first.")


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
        closing_radius=int(closing_radius),
        opening_radius=int(opening_radius),
        min_eye_separation=float(min_eye_separation) if min_eye_separation is not None else None,
        max_eye_separation=float(max_eye_separation) if max_eye_separation is not None else None,
    )


def run_tuner(args: argparse.Namespace) -> None:
    zarr_path = Path(args.zarr_path)
    if not zarr_path.exists():
        raise FileNotFoundError(zarr_path)

    root = zarr.open(str(zarr_path), mode="r")

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
            heading_deg = compute_heading_deg(kp)
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

        # Debug panels list
        debug_panels = []

        if success_flag:
            masks = [None, None]
            contours = [None, None]
            regions_info = [None, None]
            eye_labels = ["Left", "Right"]
            eye_centers_roi = [None, None]

            for eye_idx in (0, 1):
                center = kp[1 + eye_idx]
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
    root = zarr.open(str(zarr_path), mode="a")
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

        if success_flag:
            for eye_idx in (0, 1):
                center = kp[1 + eye_idx]
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
