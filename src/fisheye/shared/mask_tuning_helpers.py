"""Shared interactive mask-tuning image helpers."""

from __future__ import annotations

from typing import Dict, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np
import zarr
from skimage import filters, measure, morphology
from skimage.measure import EllipseModel


SLIDER_MAX_PADDING = 40
SLIDER_MAX_AREA = 500
SLIDER_MAX_RADIUS = 10
SLIDER_MAX_PRETHRESH = 255
SLIDER_MAX_EYE_GAP = 200
SLIDER_MAX_SOBEL = 100
SLIDER_MAX_CIRCULARITY = 100
DEBUG_PANEL_SCALE = 5
DEBUG_PANEL_MARGIN = 10
DEBUG_PANEL_SPACING = 20


def global_mask(patch: np.ndarray) -> Tuple[np.ndarray, float]:
    try:
        threshold_value = float(filters.threshold_otsu(patch))
    except ValueError:
        threshold_value = float(np.mean(patch))
    return patch < threshold_value, threshold_value


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

    filtered = np.clip(patch_float - strength * sobel_norm, 0.0, 1.0)
    filtered_uint8 = (filtered * 255.0).astype(patch.dtype, copy=False)

    sobel_visual = (sobel_norm * 255.0).astype(np.uint8, copy=False)
    return filtered_uint8, sobel_visual


def compute_heading_deg(
    kp: np.ndarray,
    keypoint_indices: Mapping[str, int] | None = None,
) -> Optional[float]:
    if kp.ndim < 2 or kp.shape[1] < 2:
        return None
    indices = keypoint_indices or {"swim_bladder": 0, "eye_left": 1, "eye_right": 2}
    try:
        bladder = kp[int(indices["swim_bladder"])]
        eye_left = kp[int(indices["eye_left"])]
        eye_right = kp[int(indices["eye_right"])]
    except Exception:
        return None
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
    min_circularity: Optional[float],
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
        if min_circularity is not None:
            perimeter = float(region.perimeter)
            if perimeter <= 0.0:
                continue
            circularity = float((4.0 * np.pi * float(area)) / (perimeter * perimeter))
            if circularity < min_circularity:
                continue
        rcx, rcy = region.centroid
        dist = (rcx - cy) ** 2 + (rcy - cx) ** 2
        if best is None or dist < best_dist:
            best = region
            best_dist = dist

    if best is None:
        return None

    return labeled == best.label


def draw_overlay(
    roi_img: np.ndarray,
    masks: Tuple[np.ndarray, np.ndarray],
    contours: Tuple[Optional[np.ndarray], Optional[np.ndarray]],
    regions_info: Tuple[Optional[dict], Optional[dict]],
) -> np.ndarray:
    base = cv2.cvtColor(roi_img.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    overlay = base.copy()

    colors = [(255, 0, 0), (0, 0, 255)]
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

    for idx, region_info in enumerate(regions_info):
        if region_info is None:
            continue

        color = colors[idx]

        cy, cx = region_info["centroid"]
        orientation = region_info["orientation"]
        major_len = region_info["major_axis_length"]
        minor_len = region_info["minor_axis_length"]

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
    """Create a debug panel showing all processing steps for one component."""
    h, w = original_patch.shape

    text_height = 16
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
        pre_thresh_display = pre_mask.astype(np.uint8) * 255
    panels.append((pre_thresh_label, to_bgr(pre_thresh_display)))

    binary_display = binary.astype(np.uint8) * 255
    if pre_thresh is not None:
        pre_mask = original_patch >= pre_thresh
        panel_binary = to_bgr(binary_display)
        panel_binary[pre_mask] = [0, 0, 255]
    else:
        panel_binary = to_bgr(binary_display)
    panels.append(("Binary", panel_binary))

    if region_mask is not None:
        region_display = cv2.cvtColor(region_mask.astype(np.uint8) * 255, cv2.COLOR_GRAY2BGR)
    else:
        region_display = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.putText(region_display, "NO REGION", (5, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
    panels.append(("Segmented", region_display))

    overlay_panel = to_bgr(original_patch).copy()
    major_color = (255, 255, 0)
    minor_color = (0, 0, 255)
    if region_mask is not None:
        overlay_color = np.zeros_like(overlay_panel, dtype=np.uint8)
        overlay_color[region_mask] = (0, 255, 0)
        overlay_panel = cv2.addWeighted(overlay_panel, 1.0, overlay_color, 0.4, 0)
    else:
        cv2.putText(overlay_panel, "NO REGION", (5, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
    panels.append(("Overlay", overlay_panel))

    ellipse_fit_panel = to_bgr(original_patch).copy()
    ellipse_fit_label = "Ellipse Fit"

    if region_mask is not None:
        contours = measure.find_contours(region_mask.astype(float), 0.5)
        if contours:
            contour = max(contours, key=lambda c: c.shape[0])
            contour_xy = contour[:, ::-1]

            if contour_xy.shape[0] >= 5:
                ellipse_model = EllipseModel()
                success = ellipse_model.estimate(contour_xy)

                if success and ellipse_model.params is not None:
                    xc, yc, a, b, theta = ellipse_model.params
                    if a < b:
                        a, b = b, a
                        theta = theta + np.pi / 2

                    ellipse_fit_label = f"Ellipse theta={np.degrees(theta):.1f} deg"

                    axes = (max(1, int(round(a))), max(1, int(round(b))))
                    center_pt = (int(round(xc)), int(round(yc)))
                    angle_deg = float(np.degrees(theta))
                    cv2.ellipse(ellipse_fit_panel, center_pt, axes, angle_deg, 0, 360, (0, 255, 0), 1)

                    cos_t = np.cos(theta)
                    sin_t = np.sin(theta)
                    major_dx = cos_t * a
                    major_dy = sin_t * a
                    major_p1 = (int(round(xc - major_dx)), int(round(yc - major_dy)))
                    major_p2 = (int(round(xc + major_dx)), int(round(yc + major_dy)))
                    cv2.line(ellipse_fit_panel, major_p1, major_p2, major_color, 1)

                    minor_dx = -sin_t * b
                    minor_dy = cos_t * b
                    minor_p1 = (int(round(xc - minor_dx)), int(round(yc - minor_dy)))
                    minor_p2 = (int(round(xc + minor_dx)), int(round(yc + minor_dy)))
                    cv2.line(ellipse_fit_panel, minor_p1, minor_p2, minor_color, 1)
                    cv2.circle(ellipse_fit_panel, center_pt, 2, major_color, -1)
                else:
                    ellipse_fit_label = "Ellipse (fit failed)"
                    cv2.putText(ellipse_fit_panel, "FIT FAILED", (5, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
            else:
                ellipse_fit_label = "Ellipse (<5 pts)"
                cv2.putText(ellipse_fit_panel, "TOO FEW PTS", (5, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
        else:
            cv2.putText(ellipse_fit_panel, "NO CONTOUR", (5, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
    else:
        cv2.putText(ellipse_fit_panel, "NO REGION", (5, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
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
            interpolation=cv2.INTER_NEAREST,
        )
    return debug


def resolve_keypoints_group(
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


_resolve_keypoints_group = resolve_keypoints_group
