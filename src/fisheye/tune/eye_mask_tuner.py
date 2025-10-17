#!/usr/bin/env python3
"""Interactive tuner for traditional eye segmentation parameters with threshold visualization."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import cv2
import numpy as np
import zarr
from skimage import filters, measure, morphology
from datetime import datetime, timezone


SLIDER_MAX_BLOCK = 20  # maps to block size = 3 + 2 * value
SLIDER_MAX_OFFSET = 200  # maps to offset in range [-20, 20]
SLIDER_MAX_PADDING = 40
SLIDER_MAX_AREA = 500
SLIDER_MAX_RADIUS = 10
SLIDER_MAX_PRETHRESH = 255
SLIDER_MAX_ROUNDNESS = 100
SLIDER_MAX_EYE_GAP = 200
SLIDER_MAX_SCALE = 10  # corresponds to 0.5x .. 5x
DEBUG_PANEL_SCALE = 5
DEBUG_PANEL_MARGIN = 10
DEBUG_PANEL_SPACING = 20


def block_from_slider(val: int) -> int:
    return max(3, 3 + 2 * val)


def offset_from_slider(val: int) -> float:
    return (val - 100) / 5.0


def slider_from_offset(offset: float) -> int:
    return int(np.clip(round(offset * 5 + 100), 0, SLIDER_MAX_OFFSET))


def roundness_from_slider(val: int) -> float:
    return np.clip(val / SLIDER_MAX_ROUNDNESS, 0.0, 1.0)


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
            'method': 'adaptive_threshold_with_feret',
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

def adaptive_mask(patch: np.ndarray, block_size: int, offset: float) -> np.ndarray:
    thresh = filters.threshold_local(patch, block_size=block_size, offset=offset)
    return patch < thresh  # Eyes are DARKER than background


def create_feret_ellipse_mask(contour: np.ndarray, shape: Tuple[int, int]) -> Tuple[np.ndarray, float, float, float]:
    """Create an ellipse mask based on Feret diameter.

    Returns:
        mask: Boolean mask of the Feret ellipse.
        roundness: minor-to-major axis ratio (0-1).
    """
    from skimage.draw import ellipse
    
    # Get Feret diameter
    p1, p2, max_dist = calculate_max_feret(contour)
    if p1 is None or p2 is None or max_dist <= 0:
        return np.zeros(shape, dtype=bool), 0.0, 0.0, 0.0
    
    # Midpoint is the center
    center = (p1 + p2) / 2
    cy, cx = center[1], center[0]  # Note: y, x order for ellipse function
    
    # Major axis length is the Feret diameter
    major_len = max_dist
    
    # Calculate orientation from Feret line
    feret_vec = p2 - p1
    orientation = np.arctan2(feret_vec[1], feret_vec[0])
    
    # Calculate perpendicular extent for minor axis
    feret_vec_norm = feret_vec / np.linalg.norm(feret_vec)
    perp_vec = np.array([-feret_vec_norm[1], feret_vec_norm[0]])
    projections = np.dot(contour - center, perp_vec)
    minor_len = np.max(projections) - np.min(projections)
    
    # Create mask
    mask = np.zeros(shape, dtype=bool)
    
    roundness = axis_roundness(major_len, minor_len)
    
    try:
        rr, cc = ellipse(
            cy, cx,
            minor_len / 2, major_len / 2,
            shape=shape,
            rotation=-orientation
        )
        mask[rr, cc] = True
    except Exception:
        pass
    
    return mask, roundness, float(major_len), float(minor_len)


def select_region(mask: np.ndarray, center: Tuple[float, float], min_area: int, max_area: Optional[int], closing: int, opening: int, use_feret_ellipse: bool = False, min_roundness: float = 0.0) -> Tuple[Optional[np.ndarray], dict]:
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

    selection_info: dict = {'used_feret': False}

    if use_feret_ellipse:
        region_contours = measure.find_contours(region_mask.astype(float), 0.5)
        if region_contours:
            best_contour = max(region_contours, key=lambda c: c.shape[0])
            best_contour = best_contour[:, ::-1]  # convert to (x, y)
            feret_mask, feret_roundness, major_len, minor_len = create_feret_ellipse_mask(best_contour, mask.shape)
            selection_info['feret_roundness'] = feret_roundness
            selection_info['feret_major_len'] = major_len
            selection_info['feret_minor_len'] = minor_len
            if feret_roundness >= min_roundness:
                selection_info['used_feret'] = True
                selection_info['roundness'] = feret_roundness
                return feret_mask, selection_info

    # Fallback to original segmentation
    major = best.major_axis_length
    minor = best.minor_axis_length
    selection_info['roundness'] = axis_roundness(major, minor)
    selection_info['feret_major_len'] = float(major)
    selection_info['feret_minor_len'] = float(minor)
    return region_mask, selection_info


def calculate_max_feret(contour: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Calculate the maximum Feret diameter (longest distance between any two points).
    Returns: (point1, point2, distance)
    """
    max_dist = 0
    p1, p2 = None, None
    
    # Check all pairs of contour points
    for i in range(len(contour)):
        for j in range(i + 1, len(contour)):
            dist = np.linalg.norm(contour[i] - contour[j])
            if dist > max_dist:
                max_dist = dist
                p1, p2 = contour[i], contour[j]
    
    return p1, p2, max_dist


def draw_overlay(roi_img: np.ndarray, masks: Tuple[np.ndarray, np.ndarray], contours: Tuple[Optional[np.ndarray], Optional[np.ndarray]], regions_info: Tuple[Optional[dict], Optional[dict]], feret_axes: Tuple[bool, bool]) -> np.ndarray:
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
    
    # Draw major/minor axes or Feret diameter
    for idx, region_info in enumerate(regions_info):
        if region_info is None:
            continue
        
        color = colors[idx]
        
        if feret_axes[idx] and contours[idx] is not None:
            # Use maximum Feret diameter
            p1, p2, max_dist = calculate_max_feret(contours[idx])
            
            # Draw the longest line
            cv2.line(output, tuple(p1.astype(int)), tuple(p2.astype(int)), color, 2)
            
            # Calculate perpendicular line at midpoint
            midpoint = (p1 + p2) / 2
            
            # Vector along Feret diameter
            feret_vec = p2 - p1
            feret_len = np.linalg.norm(feret_vec)
            feret_vec_norm = feret_vec / feret_len
            
            # Perpendicular vector
            perp_vec = np.array([-feret_vec_norm[1], feret_vec_norm[0]])
            
            # Find extent in perpendicular direction by projecting contour points
            projections = np.dot(contours[idx] - midpoint, perp_vec)
            perp_len = np.max(projections) - np.min(projections)
            
            # Draw perpendicular line
            perp_p1 = midpoint + perp_vec * perp_len / 2
            perp_p2 = midpoint - perp_vec * perp_len / 2
            cv2.line(output, tuple(perp_p1.astype(int)), tuple(perp_p2.astype(int)), color, 1)
            
            # Draw center point at midpoint
            cv2.circle(output, tuple(midpoint.astype(int)), 3, color, -1)
        else:
            # Use moment-based major/minor axes
            cy, cx = region_info['centroid']
            orientation = region_info['orientation']
            major_len = region_info['major_axis_length']
            minor_len = region_info['minor_axis_length']
            
            # Calculate endpoints of major axis
            cos_angle = np.cos(orientation)
            sin_angle = np.sin(orientation)
            
            # Major axis endpoints
            major_dx = cos_angle * major_len / 2
            major_dy = sin_angle * major_len / 2
            major_p1 = (int(cx - major_dx), int(cy - major_dy))
            major_p2 = (int(cx + major_dx), int(cy + major_dy))
            
            # Minor axis endpoints (perpendicular to major)
            minor_dx = -sin_angle * minor_len / 2
            minor_dy = cos_angle * minor_len / 2
            minor_p1 = (int(cx - minor_dx), int(cy - minor_dy))
            minor_p2 = (int(cx + minor_dx), int(cy + minor_dy))
            
            # Draw major axis (solid line)
            cv2.line(output, major_p1, major_p2, color, 2)
            # Draw minor axis (dashed-like, shorter segments)
            cv2.line(output, minor_p1, minor_p2, color, 1)
            # Draw center point
            cv2.circle(output, (int(cx), int(cy)), 3, color, -1)

    return output


def create_debug_panel(patch: np.ndarray, binary: np.ndarray, region_mask: Optional[np.ndarray], 
                       center: Tuple[float, float], label: str, block_size: int, offset: float,
                       pre_thresh: Optional[int], feret_mask: Optional[np.ndarray] = None,
                       show_feret_panel: bool = False) -> np.ndarray:
    """Create a debug panel showing all processing steps for one eye."""
    h, w = patch.shape
    
    # Calculate the threshold image for visualization
    thresh_values = filters.threshold_local(patch, block_size=block_size, offset=offset)
    
    text_height = 16  # reserve space for labels (scaled later)
    num_panels = 5 if show_feret_panel else 4
    panel_total_w = w + 2 * DEBUG_PANEL_MARGIN
    panel_total_h = h + text_height + 2 * DEBUG_PANEL_MARGIN
    total_width = num_panels * panel_total_w + (num_panels - 1) * DEBUG_PANEL_SPACING
    debug = np.zeros((panel_total_h, total_width, 3), dtype=np.uint8)

    def place_panel(idx: int, panel_img: np.ndarray) -> int:
        x = idx * (panel_total_w + DEBUG_PANEL_SPACING) + DEBUG_PANEL_MARGIN
        y = DEBUG_PANEL_MARGIN
        debug[y:y + h, x:x + w] = panel_img
        return x
    
    # Panel 1: Original patch with center marker
    panel1 = cv2.cvtColor(patch, cv2.COLOR_GRAY2BGR)
    cx, cy = center
    if 0 <= cx < w and 0 <= cy < h:
        cv2.drawMarker(panel1, (int(cx), int(cy)), (0, 255, 0), cv2.MARKER_CROSS, 5, 1)
    x_positions = [place_panel(0, panel1)]
    
    # Panel 2: Threshold values (normalized for display)
    thresh_display = ((thresh_values - thresh_values.min()) / 
                     (thresh_values.max() - thresh_values.min() + 1e-8) * 255).astype(np.uint8)
    panel2 = cv2.cvtColor(thresh_display, cv2.COLOR_GRAY2BGR)
    x_positions.append(place_panel(1, panel2))
    
    # Panel 3: Binary threshold result
    binary_display = (binary.astype(np.uint8) * 255)
    if pre_thresh is not None:
        # Show pre-threshold effect in red (rejected = too bright)
        pre_mask = patch >= pre_thresh  # Too bright, rejected
        panel3 = cv2.cvtColor(binary_display, cv2.COLOR_GRAY2BGR)
        panel3[pre_mask] = [0, 0, 255]  # Red for rejected by pre-threshold
    else:
        panel3 = cv2.cvtColor(binary_display, cv2.COLOR_GRAY2BGR)
    x_positions.append(place_panel(2, panel3))
    
    # Panel 4: Final selected region (original segmentation)
    if region_mask is not None:
        region_display = (region_mask.astype(np.uint8) * 255)
        panel4 = cv2.cvtColor(region_display, cv2.COLOR_GRAY2BGR)
    else:
        panel4 = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.putText(panel4, "NO REGION", (5, h//2), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
    x_positions.append(place_panel(3, panel4))
    
    # Panel 5: Feret-derived mask (if enabled)
    if show_feret_panel:
        if feret_mask is not None:
            feret_display = (feret_mask.astype(np.uint8) * 255)
            panel5 = cv2.cvtColor(feret_display, cv2.COLOR_GRAY2BGR)
        else:
            panel5 = np.zeros((h, w, 3), dtype=np.uint8)
            cv2.putText(panel5, "NO FERET", (5, h//2), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
        x_positions.append(place_panel(4, panel5))
    
    # Add labels
    text_y = max(8, DEBUG_PANEL_MARGIN - 2)
    cv2.putText(debug, "Original", (x_positions[0] + 2, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
    cv2.putText(debug, "Threshold", (x_positions[1] + 2, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
    cv2.putText(debug, "Binary", (x_positions[2] + 2, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
    cv2.putText(debug, "Segmented", (x_positions[3] + 2, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
    if show_feret_panel:
        cv2.putText(debug, "Feret", (x_positions[4] + 2, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
    
    # Add eye label
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
    cv2.createTrackbar("Block Size", main_window, 4, SLIDER_MAX_BLOCK, nothing)
    cv2.createTrackbar("Offset", main_window, slider_from_offset(-10.0), SLIDER_MAX_OFFSET, nothing)
    cv2.createTrackbar("Min Area", main_window, 15, SLIDER_MAX_AREA, nothing)
    cv2.createTrackbar("Max Area", main_window, 0, SLIDER_MAX_AREA, nothing)  # 0 => None
    cv2.createTrackbar("Closing r", main_window, 3, SLIDER_MAX_RADIUS, nothing)
    cv2.createTrackbar("Opening r", main_window, 1, SLIDER_MAX_RADIUS, nothing)
    cv2.createTrackbar("Min Round", main_window, 40, SLIDER_MAX_ROUNDNESS, nothing)
    cv2.createTrackbar("Min Gap", main_window, 4, SLIDER_MAX_EYE_GAP, nothing)
    cv2.createTrackbar("Max Gap", main_window, 0, SLIDER_MAX_EYE_GAP, nothing)  # 0 => unlimited
    cv2.createTrackbar("Feret Ellipse", main_window, 0, 1, nothing)  # Toggle: 0=off, 1=Feret-based ellipse
    cv2.createTrackbar("Scale", main_window, 2, SLIDER_MAX_SCALE, nothing)  # 2 -> 1.0x

    roi_idx = max(0, min(total_rois - 1, args.roi_index))
    cv2.setTrackbarPos("ROI Index", main_window, roi_idx)

    print("\n=== Eye Mask Tuner ===")
    print("Main window: Shows final segmentation result")
    print("Debug window: Shows threshold processing steps")
    print("Controls:")
    print("  n/p: Next/Previous ROI")
    print("  s: Save current parameters to Zarr metadata")
    print("  q/ESC: Quit")
    print("  Feret Ellipse: Create ellipse using Feret dimensions and axes (0=off, 1=on)")
    print("  Min Round: Require Feret ellipse minor/major ratio >= slider value (0-1)")
    print("  Min Gap / Max Gap: enforce eye-center separation bounds (Max Gap=0 disables upper limit)")
    print("  Scale: Adjust display magnification (0.5x to 5x)")
    print("  Adjust other sliders to tune parameters\n")

    while True:
        roi_idx = cv2.getTrackbarPos("ROI Index", main_window)
        padding = cv2.getTrackbarPos("ROI Padding", main_window)
        block_size = block_from_slider(cv2.getTrackbarPos("Block Size", main_window))
        pre_thresh_val = cv2.getTrackbarPos("PreThresh", main_window)
        pre_thresh = pre_thresh_val if pre_thresh_val > 0 else None
        offset = offset_from_slider(cv2.getTrackbarPos("Offset", main_window))
        min_area = cv2.getTrackbarPos("Min Area", main_window)
        max_area_slider = cv2.getTrackbarPos("Max Area", main_window)
        max_area = max_area_slider if max_area_slider > 0 else None
        closing = cv2.getTrackbarPos("Closing r", main_window)
        opening = cv2.getTrackbarPos("Opening r", main_window)
        min_roundness = roundness_from_slider(cv2.getTrackbarPos("Min Round", main_window))
        min_gap_slider = cv2.getTrackbarPos("Min Gap", main_window)
        max_gap_slider = cv2.getTrackbarPos("Max Gap", main_window)
        min_gap = float(min_gap_slider)
        max_gap = float(max_gap_slider) if max_gap_slider > 0 else None
        use_feret_ellipse = cv2.getTrackbarPos("Feret Ellipse", main_window) > 0
        scale_slider = cv2.getTrackbarPos("Scale", main_window)
        display_scale = max(1, scale_slider) / 2.0  # 0.5x .. 5x

        roi_img = np.asarray(roi_images[roi_idx])
        kp = keypoints[roi_idx]
        success_flag = success[roi_idx]

        # Main display
        display = cv2.cvtColor(roi_img.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        info_lines = [f"ROI {roi_idx+1}/{total_rois}"]
        info_lines.append(f"Block: {block_size}, Offset: {offset:.1f}, PreThresh: {pre_thresh or 'None'} (keep DARKER)")
        mask_mode = "Feret Ellipse" if use_feret_ellipse else "Original Seg"
        axes_mode = "Feret" if use_feret_ellipse else "Moment"
        roundness_text = f" | Round≥{min_roundness:.2f}" if use_feret_ellipse else ""
        info_lines.append(f"Mask: {mask_mode} | Axes: {axes_mode}{roundness_text}")
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
            feret_axes_flags = [False, False]
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

                binary = adaptive_mask(patch, block_size, offset)
                if pre_thresh is not None:
                    # PreThresh: keep pixels DARKER than this value (eyes are dark)
                    binary = np.logical_and(binary, patch < pre_thresh)
                
                # Get original segmented region for comparison
                region_mask_original, _ = select_region(
                    binary, (cx - x0, cy - y0), min_area, max_area, closing, opening
                )
                
                # Get the mask to actually use
                region_mask, selection_info = select_region(
                    binary, (cx - x0, cy - y0), min_area, max_area, closing, opening, 
                    use_feret_ellipse=use_feret_ellipse, min_roundness=min_roundness
                )
                
                # For debug panel, show Feret-based mask when enabled
                feret_used = selection_info.get('used_feret', False)
                feret_axes_flags[eye_idx] = feret_used
                feret_mask_for_debug = region_mask if feret_used else None
                
                # Create debug panel for this eye
                debug_panel = create_debug_panel(
                    patch, binary, region_mask_original, (cx - x0, cy - y0), 
                    eye_labels[eye_idx], block_size, offset, pre_thresh,
                    feret_mask=feret_mask_for_debug,
                    show_feret_panel=use_feret_ellipse
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
                feret_roundness = selection_info.get('feret_roundness')
                feret_major_len = selection_info.get('feret_major_len')
                feret_minor_len = selection_info.get('feret_minor_len')
                regions_info[eye_idx] = {
                    'centroid': (region.centroid[0] + y0, region.centroid[1] + x0),
                    'orientation': region.orientation,
                    'major_axis_length': region.major_axis_length,
                    'minor_axis_length': region.minor_axis_length,
                    'roundness': roundness_val,
                    'used_feret': feret_used,
                    'feret_roundness': feret_roundness,
                    'feret_major_len': feret_major_len,
                    'feret_minor_len': feret_minor_len,
                }
                eye_centers_roi[eye_idx] = (
                    float(region.centroid[1] + x0),
                    float(region.centroid[0] + y0),
                )
                
                info_line = (
                    f"{eye_labels[eye_idx]}: area={region.area:.0f} "
                    f"major={region.major_axis_length:.1f} minor={region.minor_axis_length:.1f}"
                )
                if roundness_val is not None:
                    info_line += f" round={roundness_val:.2f}"
                if feret_roundness is not None and not feret_used:
                    info_line += f" feret={feret_roundness:.2f}"
                if feret_major_len is not None and feret_minor_len is not None:
                    info_line += (
                        f" | maxR={feret_major_len / 2.0:.1f}px minR={max(feret_minor_len, 0) / 2.0:.1f}px"
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
            display = draw_overlay(roi_img, tuple(masks), tuple(contours), tuple(regions_info), tuple(feret_axes_flags))
        else:
            info_lines.append("Keypoints failed for this ROI")
            display = draw_overlay(roi_img, (None, None), (None, None), (None, None), (False, False))

        # Build info panel
        info_width = max(700, int(display.shape[1] * 0.5))
        info_panel = np.full((display.shape[0], info_width, 3), 240, dtype=np.uint8)
        for idx, line in enumerate(info_lines):
            cv2.putText(info_panel, line, (18, 30 + idx * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

        # Add spacer between info panel and display
        spacer_width = 30  # pixels of space between panels
        spacer = np.full((display.shape[0], spacer_width, 3), 200, dtype=np.uint8)
        combined_display = np.hstack([info_panel, spacer, display])

        if display_scale != 1.0:
            interpolation = cv2.INTER_NEAREST if display_scale >= 1.0 else cv2.INTER_AREA
            combined_display = cv2.resize(combined_display, None, fx=display_scale, fy=display_scale, interpolation=interpolation)

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
                'threshold_block_size': int(block_size),
                'threshold_offset': float(offset),
                'pre_threshold': int(pre_thresh) if pre_thresh is not None else None,
                'min_area': int(min_area),
                'max_area': int(max_area) if max_area is not None else None,
                'closing_radius': int(closing),
                'opening_radius': int(opening),
                'use_feret_ellipse': bool(use_feret_ellipse),
                'min_roundness': float(min_roundness) if use_feret_ellipse else None,
                'min_eye_separation': float(min_gap) if min_gap > 0 else None,
                'max_eye_separation': float(max_gap) if max_gap is not None and max_gap > 0 else None,
            }
            context = {
                'roi_index': int(roi_idx),
                'roi_index_one_based': int(roi_idx + 1),
                'total_rois': int(total_rois),
                'crop_run': crop_run,
                'keypoint_run': keypoint_run,
                'feret_enabled': bool(use_feret_ellipse),
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
