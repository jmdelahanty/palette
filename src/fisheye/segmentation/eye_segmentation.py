"""Traditional eye segmentation on ROI crops using existing keypoints."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import zarr
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
from skimage import filters, measure, morphology

from ..shared.zarr.schema import get_run_group
from ..utils.system import get_environment_info, get_git_info


@dataclass
class EyeSegmentationConfig:
    roi_padding: int = 12
    threshold_block_size: int = 21
    threshold_offset: float = -10.0
    pre_threshold: Optional[int] = None
    min_area: int = 15
    max_area: Optional[int] = None
    closing_radius: int = 3
    opening_radius: int = 1
    contour_min_points: int = 5
    keypoint_run: Optional[str] = None
    crop_run: Optional[str] = None


def _prepare_run_group(root: zarr.Group, console: Console) -> Tuple[zarr.Group, str]:
    group, name = get_run_group(root, "eye_masks", console=console, create_new=True)
    return group, name


def _local_threshold(patch: np.ndarray, config: EyeSegmentationConfig) -> np.ndarray:
    block = max(3, config.threshold_block_size | 1)
    thresh = filters.threshold_local(patch, block_size=block, offset=config.threshold_offset)
    binary = patch > thresh
    return binary


def _select_region(binary: np.ndarray, center: Tuple[float, float], config: EyeSegmentationConfig) -> Optional[np.ndarray]:
    if config.closing_radius > 0:
        binary = morphology.binary_closing(binary, morphology.disk(config.closing_radius))
    if config.opening_radius > 0:
        binary = morphology.binary_opening(binary, morphology.disk(config.opening_radius))

    labeled = measure.label(binary)
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
        if area < config.min_area:
            continue
        if config.max_area is not None and area > config.max_area:
            continue
        rcx, rcy = region.centroid
        dist = (rcx - cy) ** 2 + (rcy - cx) ** 2
        if best is None or dist < best_dist:
            best = region
            best_dist = dist

    if best is None:
        return None

    mask = labeled == best.label
    return mask


def _extract_contour(mask: np.ndarray, min_points: int) -> Optional[np.ndarray]:
    contours = measure.find_contours(mask.astype(float), 0.5)
    if not contours:
        return None
    contour = max(contours, key=lambda c: c.shape[0])
    if contour.shape[0] < min_points:
        return None
    return contour[:, ::-1]  # Convert to (x, y)


def segment_eye_masks(
    zarr_path: str,
    config_dict: Optional[Dict] = None,
    console: Optional[Console] = None,
) -> str:
    console = console or Console()
    cfg = EyeSegmentationConfig(**(config_dict or {}))

    root = zarr.open(zarr_path, mode="a")

    if "crop_runs" not in root:
        raise ValueError("crop_runs missing from Zarr; run crop stage first")
    crop_run = cfg.crop_run or root["crop_runs"].attrs.get("latest")
    if crop_run is None:
        raise ValueError("No crop run available")
    crop_group = root[f"crop_runs/{crop_run}"]

    if "keypoints_runs" not in root:
        raise ValueError("keypoints_runs missing from Zarr; run keypoints stage first")
    keypoint_run = cfg.keypoint_run or root["keypoints_runs"].attrs.get("latest")
    if keypoint_run is None:
        raise ValueError("No keypoint run available")
    kp_group = root[f"keypoints_runs/{keypoint_run}"]

    roi_images = crop_group["roi_images"]
    keypoints_roi = kp_group["keypoints_roi"][:]
    success_flags = kp_group["detection_success"][:]

    total_rois = roi_images.shape[0]
    roi_h, roi_w = roi_images.shape[1:3]

    masks = np.zeros((total_rois, 2, roi_h, roi_w), dtype=np.uint8)
    ellipse_params = np.full((total_rois, 2, 5), np.nan, dtype=np.float32)
    ellipse_success = np.zeros((total_rois, 2), dtype=bool)

    left_ptr = np.full((total_rois,), -1, dtype=np.int64)
    left_len = np.zeros((total_rois,), dtype=np.int32)
    right_ptr = np.full((total_rois,), -1, dtype=np.int64)
    right_len = np.zeros((total_rois,), dtype=np.int32)
    left_points: List[np.ndarray] = []
    right_points: List[np.ndarray] = []
    left_total = 0
    right_total = 0

    run_group, run_name = _prepare_run_group(root, console)

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeRemainingColumn(),
        console=console,
    )

    with progress:
        task = progress.add_task("[cyan]Segmenting eyes...", total=total_rois)
        for idx in range(total_rois):
            if not success_flags[idx]:
                progress.update(task, advance=1)
                continue

            roi_img = np.asarray(roi_images[idx])
            kp = keypoints_roi[idx]
            for eye_idx, label in enumerate(("eye_left", "eye_right")):
                center = kp[1 + eye_idx]
                cx, cy = float(center[0]), float(center[1])
                if not np.isfinite(cx) or not np.isfinite(cy):
                    continue

                x0 = max(0, int(round(cx)) - cfg.roi_padding)
                x1 = min(roi_w, int(round(cx)) + cfg.roi_padding + 1)
                y0 = max(0, int(round(cy)) - cfg.roi_padding)
                y1 = min(roi_h, int(round(cy)) + cfg.roi_padding + 1)
                if x1 - x0 <= 2 or y1 - y0 <= 2:
                    continue

                patch = roi_img[y0:y1, x0:x1]
                binary = _local_threshold(patch, cfg)
                if cfg.pre_threshold is not None:
                    base_mask = patch > cfg.pre_threshold
                    binary = np.logical_and(binary, base_mask)
                region_mask = _select_region(binary, (cx - x0, cy - y0), cfg)
                if region_mask is None:
                    continue

                mask_full = np.zeros_like(patch, dtype=np.uint8)
                mask_full[region_mask] = 1
                masks[idx, eye_idx, y0:y1, x0:x1] = np.maximum(
                    masks[idx, eye_idx, y0:y1, x0:x1], mask_full
                )

                region = measure.regionprops(region_mask.astype(int))[0]
                centroid_local = region.centroid
                ellipse_params[idx, eye_idx] = [
                    float(x0 + centroid_local[1]),
                    float(y0 + centroid_local[0]),
                    float(region.major_axis_length),
                    float(region.minor_axis_length),
                    float(np.rad2deg(region.orientation)),
                ]
                ellipse_success[idx, eye_idx] = True

                contour = _extract_contour(region_mask, cfg.contour_min_points)
                if contour is not None:
                    contour[:, 0] += x0
                    contour[:, 1] += y0
                    if eye_idx == 0:
                        left_ptr[idx] = left_total
                        left_len[idx] = contour.shape[0]
                        left_total += contour.shape[0]
                        left_points.append(contour.astype(np.float32))
                    else:
                        right_ptr[idx] = right_total
                        right_len[idx] = contour.shape[0]
                        right_total += contour.shape[0]
                        right_points.append(contour.astype(np.float32))

            progress.update(task, advance=1)

    left_concat = np.concatenate(left_points, axis=0) if left_points else np.zeros((0, 2), dtype=np.float32)
    right_concat = np.concatenate(right_points, axis=0) if right_points else np.zeros((0, 2), dtype=np.float32)
    left_store = left_concat if left_concat.size > 0 else np.zeros((1, 2), dtype=np.float32)
    right_store = right_concat if right_concat.size > 0 else np.zeros((1, 2), dtype=np.float32)

    run_group.create_array(
        "masks_roi",
        data=masks,
        chunks=(min(512, total_rois), 2, roi_h, roi_w),
        dtype="uint8",
        overwrite=True,
    )
    run_group.create_array(
        "ellipse_params",
        data=ellipse_params,
        chunks=(min(1024, total_rois), 2, 5),
        overwrite=True,
    )
    run_group.create_array(
        "ellipse_success",
        data=ellipse_success,
        chunks=(min(1024, total_rois), 2),
        overwrite=True,
    )
    run_group.create_array("contour_left_ptr", data=left_ptr, overwrite=True)
    run_group.create_array("contour_left_len", data=left_len, overwrite=True)
    run_group.create_array("contour_right_ptr", data=right_ptr, overwrite=True)
    run_group.create_array("contour_right_len", data=right_len, overwrite=True)
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

    git_info = get_git_info()
    env_info = get_environment_info()

    run_group.attrs.update(
        {
            "method": "traditional_eye_segmentation",
            "config": cfg.__dict__,
            "source_crop_run": crop_run,
            "source_keypoint_run": keypoint_run,
            "total_rois": total_rois,
            "successful_eyes": int(ellipse_success.sum()),
            "contours_left_count": int(left_concat.shape[0]),
            "contours_right_count": int(right_concat.shape[0]),
            "eye_labels": ["eye_left", "eye_right"],
            "git_commit": git_info.get("commit_hash", "unknown"),
            "git_branch": git_info.get("branch", "unknown"),
            "hostname": env_info["platform"].get("hostname", "unknown"),
        }
    )

    console.print(
        f"[green]✓[/green] Eye masks saved as [cyan]eye_masks_runs/{run_name}[/cyan] "
        f"({ellipse_success.sum()} successful eyes)"
    )

    return run_name


__all__ = ["EyeSegmentationConfig", "segment_eye_masks"]
