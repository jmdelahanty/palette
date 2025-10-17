#!/usr/bin/env python3
"""Interactive tuner for traditional eye segmentation parameters with threshold visualization."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import zarr
from skimage import filters, measure, morphology


SLIDER_MAX_BLOCK = 20  # maps to block size = 3 + 2 * value
SLIDER_MAX_OFFSET = 200  # maps to offset in range [-20, 20]
SLIDER_MAX_PADDING = 40
SLIDER_MAX_AREA = 500
SLIDER_MAX_RADIUS = 10
SLIDER_MAX_PRETHRESH = 255
USE_ELLIPSE_MASK = True  # Toggle for ellipse vs original mask


def block_from_slider(val: int) -> int:
    return max(3, 3 + 2 * val)


def offset_from_slider(val: int) -> float:
    return (val - 100) / 5.0


def slider_from_offset(offset: float) -> int:
    return int(np.clip(round(offset * 5 + 100), 0, SLIDER_MAX_OFFSET))


def adaptive_mask(patch: np.ndarray, block_size: int, offset: float) -> np.ndarray:
    thresh = filters.threshold_local(patch, block_size=block_size, offset=offset)
    return patch < thresh  # Eyes are DARKER than background


def create_ellipse_mask(region, shape: Tuple[int, int]) -> np.ndarray:
    """Create a filled ellipse mask from region properties."""
    from skimage.draw import ellipse
    
    # Get ellipse parameters
    cy, cx = region.centroid
    orientation = region.orientation
    major_axis = region.major_axis_length
    minor_axis = region.minor_axis_length
    
    # Create empty mask
    mask = np.zeros(shape, dtype=bool)
    
    # Draw filled ellipse
    try:
        rr, cc = ellipse(
            cy, cx,
            minor_axis / 2, major_axis / 2,  # Note: order is (r_radius, c_radius)
            shape=shape,
            rotation=-orientation  # Negative because of coordinate system
        )
        mask[rr, cc] = True
    except Exception:
        # If ellipse drawing fails, return empty mask
        pass
    
    return mask


def create_feret_ellipse_mask(contour: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    """Create an ellipse mask based on Feret diameter."""
    from skimage.draw import ellipse
    
    # Get Feret diameter
    p1, p2, max_dist = calculate_max_feret(contour)
    
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
    
    return mask


def select_region(mask: np.ndarray, center: Tuple[float, float], min_area: int, max_area: Optional[int], closing: int, opening: int, use_ellipse: bool = False, use_feret_ellipse: bool = False, contour: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
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

    if use_feret_ellipse and contour is not None:
        # Extract contour from the best region
        region_mask = (labeled == best.label)
        region_contours = measure.find_contours(region_mask.astype(float), 0.5)
        if region_contours:
            best_contour = max(region_contours, key=lambda c: c.shape[0])
            # Convert to (x, y) coordinates
            best_contour = best_contour[:, ::-1]
            # Create Feret-based ellipse
            return create_feret_ellipse_mask(best_contour, mask.shape)
    elif use_ellipse:
        # Return moment-based ellipse mask instead of original region
        return create_ellipse_mask(best, mask.shape)
    else:
        return labeled == best.label


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


def draw_overlay(roi_img: np.ndarray, masks: Tuple[np.ndarray, np.ndarray], contours: Tuple[Optional[np.ndarray], Optional[np.ndarray]], regions_info: Tuple[Optional[dict], Optional[dict]], use_feret: bool = False) -> np.ndarray:
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
        
        if use_feret and contours[idx] is not None:
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
                       pre_thresh: Optional[int], ellipse_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Create a debug panel showing all processing steps for one eye."""
    h, w = patch.shape
    
    # Calculate the threshold image for visualization
    thresh_values = filters.threshold_local(patch, block_size=block_size, offset=offset)
    
    # Create visualization panels (now 5 panels if we have ellipse)
    num_panels = 5 if ellipse_mask is not None else 4
    panel_w = w * num_panels + (num_panels - 1) * 10  # panels with spacing
    debug = np.zeros((h, panel_w, 3), dtype=np.uint8)
    
    # Panel 1: Original patch with center marker
    panel1 = cv2.cvtColor(patch, cv2.COLOR_GRAY2BGR)
    cx, cy = center
    if 0 <= cx < w and 0 <= cy < h:
        cv2.drawMarker(panel1, (int(cx), int(cy)), (0, 255, 0), cv2.MARKER_CROSS, 5, 1)
    debug[:h, :w] = panel1
    
    # Panel 2: Threshold values (normalized for display)
    thresh_display = ((thresh_values - thresh_values.min()) / 
                     (thresh_values.max() - thresh_values.min() + 1e-8) * 255).astype(np.uint8)
    panel2 = cv2.cvtColor(thresh_display, cv2.COLOR_GRAY2BGR)
    debug[:h, w+10:2*w+10] = panel2
    
    # Panel 3: Binary threshold result
    binary_display = (binary.astype(np.uint8) * 255)
    if pre_thresh is not None:
        # Show pre-threshold effect in red (rejected = too bright)
        pre_mask = patch >= pre_thresh  # Too bright, rejected
        panel3 = cv2.cvtColor(binary_display, cv2.COLOR_GRAY2BGR)
        panel3[pre_mask] = [0, 0, 255]  # Red for rejected by pre-threshold
    else:
        panel3 = cv2.cvtColor(binary_display, cv2.COLOR_GRAY2BGR)
    debug[:h, 2*w+20:3*w+20] = panel3
    
    # Panel 4: Final selected region (original segmentation)
    if region_mask is not None:
        region_display = (region_mask.astype(np.uint8) * 255)
        panel4 = cv2.cvtColor(region_display, cv2.COLOR_GRAY2BGR)
    else:
        panel4 = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.putText(panel4, "NO REGION", (5, h//2), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
    debug[:h, 3*w+30:4*w+30] = panel4
    
    # Panel 5: Ellipse mask (if available)
    if ellipse_mask is not None:
        ellipse_display = (ellipse_mask.astype(np.uint8) * 255)
        panel5 = cv2.cvtColor(ellipse_display, cv2.COLOR_GRAY2BGR)
        debug[:h, 4*w+40:5*w+40] = panel5
    
    # Add labels
    cv2.putText(debug, "Original", (5, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
    cv2.putText(debug, "Threshold", (w+15, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
    cv2.putText(debug, "Binary", (2*w+25, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
    cv2.putText(debug, "Segmented", (3*w+35, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
    if ellipse_mask is not None:
        cv2.putText(debug, "Ellipse", (4*w+45, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
    
    # Add eye label
    cv2.putText(debug, f"{label} Eye", (5, h - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
    
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
    cv2.createTrackbar("Use Ellipse", main_window, 0, 1, nothing)  # Toggle: 0=original, 1=moment ellipse
    cv2.createTrackbar("Use Feret", main_window, 0, 1, nothing)  # Toggle: 0=moments, 1=Feret diameter
    cv2.createTrackbar("Feret Ellipse", main_window, 0, 1, nothing)  # Toggle: 0=off, 1=Feret-based ellipse

    roi_idx = max(0, min(total_rois - 1, args.roi_index))
    cv2.setTrackbarPos("ROI Index", main_window, roi_idx)

    print("\n=== Eye Mask Tuner ===")
    print("Main window: Shows final segmentation result")
    print("Debug window: Shows threshold processing steps")
    print("Controls:")
    print("  n/p: Next/Previous ROI")
    print("  q/ESC: Quit")
    print("  Use Ellipse: Moment-based ellipse fit (0=off, 1=on)")
    print("  Use Feret: Show Feret diameter axes (0=moments, 1=Feret)")
    print("  Feret Ellipse: Create ellipse using Feret dimensions (0=off, 1=on)")
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
        use_ellipse = cv2.getTrackbarPos("Use Ellipse", main_window) > 0
        use_feret = cv2.getTrackbarPos("Use Feret", main_window) > 0
        use_feret_ellipse = cv2.getTrackbarPos("Feret Ellipse", main_window) > 0

        roi_img = np.asarray(roi_images[roi_idx])
        kp = keypoints[roi_idx]
        success_flag = success[roi_idx]

        # Main display
        display = cv2.cvtColor(roi_img.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        info_lines = [f"ROI {roi_idx+1}/{total_rois}"]
        info_lines.append(f"Block: {block_size}, Offset: {offset:.1f}, PreThresh: {pre_thresh or 'None'} (keep DARKER)")
        mask_mode = "Feret Ellipse" if use_feret_ellipse else ("Moment Ellipse" if use_ellipse else "Original Seg")
        info_lines.append(f"Mask: {mask_mode} | Axes: {'Feret' if use_feret else 'Moment'}")

        # Debug panels list
        debug_panels = []

        if success_flag:
            masks = [None, None]
            contours = [None, None]
            regions_info = [None, None]
            eye_labels = ["Left", "Right"]

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
                
                # Get original segmented region (for comparison and ellipse fitting)
                region_mask_original = select_region(binary, (cx - x0, cy - y0), min_area, max_area, closing, opening, use_ellipse=False, use_feret_ellipse=False)
                
                # Get the mask to actually use
                region_mask = select_region(binary, (cx - x0, cy - y0), min_area, max_area, closing, opening, 
                                           use_ellipse=use_ellipse, use_feret_ellipse=use_feret_ellipse)
                
                # For debug panel, show ellipse if using either ellipse mode
                ellipse_mask_for_debug = None
                if (use_ellipse or use_feret_ellipse) and region_mask is not None:
                    ellipse_mask_for_debug = region_mask
                
                # Create debug panel for this eye
                debug_panel = create_debug_panel(
                    patch, binary, region_mask_original, (cx - x0, cy - y0), 
                    eye_labels[eye_idx], block_size, offset, pre_thresh,
                    ellipse_mask=ellipse_mask_for_debug
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
                    'minor_axis_length': region.minor_axis_length
                }
                
                info_lines.append(
                    f"{eye_labels[eye_idx]}: area={region.area:.0f} major={region.major_axis_length:.1f} minor={region.minor_axis_length:.1f}"
                )

                contour = measure.find_contours(region_mask.astype(float), 0.5)
                if contour:
                    best = max(contour, key=lambda c: c.shape[0])
                    if best.shape[0] >= 5:
                        best = best[:, ::-1]
                        best[:, 0] += x0
                        best[:, 1] += y0
                        contours[eye_idx] = best

            display = draw_overlay(roi_img, tuple(masks), tuple(contours), tuple(regions_info), use_feret=use_feret)
        else:
            info_lines.append("Keypoints failed for this ROI")
            display = draw_overlay(roi_img, (None, None), (None, None), (None, None), use_feret=use_feret)

        # Add info text to main display
        for idx, line in enumerate(info_lines):
            cv2.putText(display, line, (10, 20 + idx * 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        cv2.imshow(main_window, display)
        
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