#!/usr/bin/env python3
"""Interactive tuner for traditional eye segmentation parameters."""

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


def block_from_slider(val: int) -> int:
    return max(3, 3 + 2 * val)


def offset_from_slider(val: int) -> float:
    return (val - 100) / 5.0


def slider_from_offset(offset: float) -> int:
    return int(np.clip(round(offset * 5 + 100), 0, SLIDER_MAX_OFFSET))


def adaptive_mask(patch: np.ndarray, block_size: int, offset: float) -> np.ndarray:
    thresh = filters.threshold_local(patch, block_size=block_size, offset=offset)
    return patch > thresh


def select_region(mask: np.ndarray, center: Tuple[float, float], min_area: int, max_area: Optional[int], closing: int, opening: int) -> Optional[np.ndarray]:
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

    return labeled == best.label


def draw_overlay(roi_img: np.ndarray, masks: Tuple[np.ndarray, np.ndarray], contours: Tuple[Optional[np.ndarray], Optional[np.ndarray]]) -> np.ndarray:
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

    return output


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

    window = "Eye Mask Tuner"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    def nothing(val: int) -> None:
        pass

    cv2.createTrackbar("ROI Index", window, 0, total_rois - 1, nothing)
    cv2.createTrackbar("ROI Padding", window, min(12, SLIDER_MAX_PADDING), SLIDER_MAX_PADDING, nothing)
    cv2.createTrackbar("PreThresh", window, 0, SLIDER_MAX_PRETHRESH, nothing)
    cv2.createTrackbar("Block Size", window, 4, SLIDER_MAX_BLOCK, nothing)
    cv2.createTrackbar("Offset", window, slider_from_offset(-10.0), SLIDER_MAX_OFFSET, nothing)
    cv2.createTrackbar("Min Area", window, 15, SLIDER_MAX_AREA, nothing)
    cv2.createTrackbar("Max Area", window, 0, SLIDER_MAX_AREA, nothing)  # 0 => None
    cv2.createTrackbar("Closing r", window, 3, SLIDER_MAX_RADIUS, nothing)
    cv2.createTrackbar("Opening r", window, 1, SLIDER_MAX_RADIUS, nothing)

    roi_idx = max(0, min(total_rois - 1, args.roi_index))
    cv2.setTrackbarPos("ROI Index", window, roi_idx)

    while True:
        roi_idx = cv2.getTrackbarPos("ROI Index", window)
        padding = cv2.getTrackbarPos("ROI Padding", window)
        block_size = block_from_slider(cv2.getTrackbarPos("Block Size", window))
        pre_thresh_val = cv2.getTrackbarPos("PreThresh", window)
        pre_thresh = pre_thresh_val if pre_thresh_val > 0 else None
        offset = offset_from_slider(cv2.getTrackbarPos("Offset", window))
        min_area = cv2.getTrackbarPos("Min Area", window)
        max_area_slider = cv2.getTrackbarPos("Max Area", window)
        max_area = max_area_slider if max_area_slider > 0 else None
        closing = cv2.getTrackbarPos("Closing r", window)
        opening = cv2.getTrackbarPos("Opening r", window)

        roi_img = np.asarray(roi_images[roi_idx])
        kp = keypoints[roi_idx]
        success_flag = success[roi_idx]

        display = cv2.cvtColor(roi_img.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        info_lines = [f"ROI {roi_idx+1}/{total_rois}"]

        if success_flag:
            masks = [None, None]
            contours = [None, None]
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
                    binary = np.logical_and(binary, patch > pre_thresh)
                region_mask = select_region(binary, (cx - x0, cy - y0), min_area, max_area, closing, opening)
                if region_mask is None:
                    info_lines.append(f"{eye_labels[eye_idx]}: no region")
                    continue

                full_mask = np.zeros_like(roi_img, dtype=np.uint8)
                full_mask[y0:y1, x0:x1][region_mask] = 1
                masks[eye_idx] = full_mask

                region = measure.regionprops(region_mask.astype(int))[0]
                centroid_local = region.centroid
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

            display = draw_overlay(roi_img, tuple(masks), tuple(contours))
        else:
            info_lines.append("Keypoints failed for this ROI")

        for idx, line in enumerate(info_lines):
            cv2.putText(display, line, (10, 20 + idx * 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        cv2.imshow(window, display)
        key = cv2.waitKey(30) & 0xFF

        if key in (ord("q"), 27):
            break
        elif key == ord("n"):
            cv2.setTrackbarPos("ROI Index", window, min(total_rois - 1, roi_idx + 1))
        elif key == ord("p"):
            cv2.setTrackbarPos("ROI Index", window, max(0, roi_idx - 1))

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
