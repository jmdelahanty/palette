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
from typing import Optional, Tuple, Dict, Any

import cv2

# Limit OpenCV threads
cv2.setNumThreads(2)
import numpy as np
import zarr
from skimage import filters, measure, morphology
from datetime import datetime, timezone


SLIDER_MAX_PADDING = 40
SLIDER_MAX_AREA = 500
SLIDER_MAX_RADIUS = 10
SLIDER_MAX_PRETHRESH = 255
SLIDER_MAX_EYE_GAP = 200
SLIDER_MAX_SOBEL = 100  # maps to strength in range [0, 1]
DEBUG_PANEL_SCALE = 5
DEBUG_PANEL_MARGIN = 10
DEBUG_PANEL_SPACING = 20


def axis_roundness(major: float, minor: float) -> float:
    if major <= 0:
        return 0.0
    return float(np.clip(minor / major, 0.0, 1.0))

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


def max_distance_and_perp_chord(
    mask: np.ndarray,
    max_points: int = 500,
) -> Optional[Tuple[np.ndarray, np.ndarray, float, Optional[np.ndarray], Optional[np.ndarray]]]:
    contours = measure.find_contours(mask.astype(float), 0.5)
    if not contours:
        return None
    contour = max(contours, key=lambda c: c.shape[0])
    if contour.shape[0] < 2:
        return None
    points_full = contour[:, ::-1]  # (x, y)
    points = points_full
    if points.shape[0] > max_points:
        idx = np.linspace(0, points.shape[0] - 1, max_points).astype(int)
        points = points[idx]

    diffs = points[:, None, :] - points[None, :, :]
    dist2 = np.sum(diffs * diffs, axis=2)
    max_idx = np.unravel_index(int(np.argmax(dist2)), dist2.shape)
    p1 = points[max_idx[0]]
    p2 = points[max_idx[1]]
    max_dist = float(np.sqrt(dist2[max_idx]))

    d = p2 - p1
    norm = float(np.linalg.norm(d))
    if norm == 0.0:
        return p1, p2, max_dist, None, None
    n = d / norm  # line normal for perpendicular chord
    u = np.array([-n[1], n[0]], dtype=np.float64)  # line direction for perpendicular chord
    mid = (p1 + p2) / 2.0

    intersections: list[np.ndarray] = []
    for i in range(points_full.shape[0]):
        a = points_full[i]
        b = points_full[(i + 1) % points_full.shape[0]]
        da = float(np.dot(a - mid, n))
        db = float(np.dot(b - mid, n))
        if da == 0.0 and db == 0.0:
            continue
        if da * db > 0.0:
            continue
        denom = da - db
        if denom == 0.0:
            continue
        t = da / denom
        if 0.0 <= t <= 1.0:
            intersections.append(a + t * (b - a))

    if len(intersections) < 2:
        return p1, p2, max_dist, None, None

    proj = [float(np.dot(p - mid, u)) for p in intersections]
    min_idx = int(np.argmin(proj))
    max_idx = int(np.argmax(proj))
    perp_p1 = intersections[min_idx]
    perp_p2 = intersections[max_idx]
    return p1, p2, max_dist, perp_p1, perp_p2


def select_region(
    mask: np.ndarray,
    center: Tuple[float, float],
    min_area: int,
    max_area: Optional[int],
    closing: int,
    opening: int,
) -> Tuple[Optional[np.ndarray], dict]:
    if closing > 0:
        mask = morphology.binary_closing(mask, morphology.disk(closing))
    if opening > 0:
        mask = morphology.binary_opening(mask, morphology.disk(opening))

    labeled = measure.label(mask)
    if labeled.max() == 0:
        return None, {}

    regions = measure.regionprops(labeled)
    if not regions:
        return None, {}

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
        return None, {}

    region_mask = (labeled == best.label)

    major = best.major_axis_length
    minor = best.minor_axis_length
    selection_info: dict = {'roundness': axis_roundness(major, minor)}
    return region_mask, selection_info


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
    major_color = (0, 200, 255)
    minor_color = (0, 0, 255)
    if region_mask is not None:
        overlay_color = np.zeros_like(overlay_panel, dtype=np.uint8)
        overlay_color[region_mask] = (0, 255, 0)
        overlay_panel = cv2.addWeighted(overlay_panel, 1.0, overlay_color, 0.4, 0)
    else:
        cv2.putText(overlay_panel, "NO REGION", (5, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
    panels.append(("Overlay", overlay_panel))

    overlay_axes_panel = overlay_panel.copy()
    if region_mask is not None:
        region_props = measure.regionprops(region_mask.astype(np.uint8))
        if region_props:
            region = region_props[0]
            cy, cx = region.centroid
            orientation = region.orientation
            major_len = region.major_axis_length
            minor_len = region.minor_axis_length
            center_pt = (int(round(cx)), int(round(cy)))
            cos_angle = np.cos(orientation)
            sin_angle = np.sin(orientation)
            major_dx = cos_angle * major_len / 2
            major_dy = sin_angle * major_len / 2
            minor_dx = -sin_angle * minor_len / 2
            minor_dy = cos_angle * minor_len / 2
            major_p1 = (int(round(cx - major_dx)), int(round(cy - major_dy)))
            major_p2 = (int(round(cx + major_dx)), int(round(cy + major_dy)))
            minor_p1 = (int(round(cx - minor_dx)), int(round(cy - minor_dy)))
            minor_p2 = (int(round(cx + minor_dx)), int(round(cy + minor_dy)))
            cv2.line(overlay_axes_panel, major_p1, major_p2, major_color, 1)
            cv2.line(overlay_axes_panel, minor_p1, minor_p2, minor_color, 1)
            cv2.circle(overlay_axes_panel, center_pt, 2, major_color, -1)
    panels.append(("Overlay+Axes", overlay_axes_panel))

    max_dist_panel = to_bgr(original_patch).copy()
    max_dist_label = "Max Dist"
    if region_mask is not None:
        segment = max_distance_and_perp_chord(region_mask)
        if segment is not None:
            p1, p2, dist, perp_p1, perp_p2 = segment
            max_dist_label = f"Max Dist {dist:.1f}px"
            p1_i = (int(round(p1[0])), int(round(p1[1])))
            p2_i = (int(round(p2[0])), int(round(p2[1])))
            cv2.line(max_dist_panel, p1_i, p2_i, (220, 220, 220), 1)
            cv2.circle(max_dist_panel, p1_i, 2, (220, 220, 220), -1)
            cv2.circle(max_dist_panel, p2_i, 2, (220, 220, 220), -1)
            if perp_p1 is not None and perp_p2 is not None:
                q1_i = (int(round(perp_p1[0])), int(round(perp_p1[1])))
                q2_i = (int(round(perp_p2[0])), int(round(perp_p2[1])))
                cv2.line(max_dist_panel, q1_i, q2_i, (255, 0, 0), 1)
                cv2.circle(max_dist_panel, q1_i, 2, (255, 0, 0), -1)
                cv2.circle(max_dist_panel, q2_i, 2, (255, 0, 0), -1)
    panels.append((max_dist_label, max_dist_panel))

    ellipse_panel = to_bgr(original_patch).copy()
    ellipse_label = "Ellipse Fit"
    ellipse_mask = region_mask

    if ellipse_mask is not None:
        region_props = measure.regionprops(ellipse_mask.astype(np.uint8))
        if region_props:
            region = region_props[0]
            cy, cx = region.centroid
            orientation = region.orientation
            major_len = region.major_axis_length
            minor_len = region.minor_axis_length
            center_pt = (int(round(cx)), int(round(cy)))
            axes = (
                max(1, int(round(major_len / 2))),
                max(1, int(round(minor_len / 2))),
            )
            angle_deg = float(np.degrees(orientation))
            cos_angle = np.cos(orientation)
            sin_angle = np.sin(orientation)
            major_dx = cos_angle * major_len / 2
            major_dy = sin_angle * major_len / 2
            minor_dx = -sin_angle * minor_len / 2
            minor_dy = cos_angle * minor_len / 2
            major_p1 = (int(round(cx - major_dx)), int(round(cy - major_dy)))
            major_p2 = (int(round(cx + major_dx)), int(round(cy + major_dy)))
            minor_p1 = (int(round(cx - minor_dx)), int(round(cy - minor_dy)))
            minor_p2 = (int(round(cx + minor_dx)), int(round(cy + minor_dy)))
            cv2.line(ellipse_panel, major_p1, major_p2, major_color, 1)
            cv2.line(ellipse_panel, minor_p1, minor_p2, minor_color, 1)
            cv2.circle(ellipse_panel, center_pt, 2, major_color, -1)
    else:
        cv2.putText(ellipse_panel, "NO REGION", (5, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
    panels.append((ellipse_label, ellipse_panel))

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

    keypoint_runs = root.get("keypoints_runs", None)
    if keypoint_runs is None or "latest" not in keypoint_runs.attrs:
        raise RuntimeError("No keypoints_runs found; run keypoints stage first")
    keypoint_run = args.keypoint_run or keypoint_runs.attrs["latest"]
    kp_group = keypoint_runs[keypoint_run]

    roi_images = crop_group["roi_images"]
    keypoints = kp_group["keypoints_roi"][:]
    success = kp_group["detection_success"][:]
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
    print("Main window: Shows final segmentation result")
    print("Debug window: Shows threshold processing steps")
    print("Controls:")
    print("  n/p: Next/Previous ROI")
    print("  s: Save current parameters to Zarr metadata")
    print("  q/ESC: Quit")
    print("  Rotate ROI: Align heading to 0° for tuner-only preview")
    print("  Min Gap / Max Gap: enforce eye-center separation bounds (Max Gap=0 disables upper limit)")
    print("  Sobel %: Blend Sobel edge subtraction into global thresholding (0=off)")
    print("  Axis colors: Major (blue), Minor (red)")
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
                region_mask, selection_info = select_region(
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
                roundness_val = selection_info.get('roundness')
                regions_info[eye_idx] = {
                    'centroid': (region.centroid[0] + y0, region.centroid[1] + x0),
                    'orientation': region.orientation,
                    'major_axis_length': region.major_axis_length,
                    'minor_axis_length': region.minor_axis_length,
                    'roundness': roundness_val,
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
                if roundness_val is not None:
                    info_line += f" round={roundness_val:.2f}"
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive tuner for eye segmentation parameters")
    parser.add_argument("zarr_path", help="Path to Palette Zarr store")
    parser.add_argument("--roi-index", type=int, default=0, help="Initial ROI index")
    parser.add_argument("--crop-run", help="Specific crop run name")
    parser.add_argument("--keypoint-run", help="Specific keypoint run name")
    args = parser.parse_args()
    run_tuner(args)


if __name__ == "__main__":
    main()
