"""Traditional eye segmentation on ROI crops using existing keypoints."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import zarr
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
import dask
from dask import delayed
from dask.diagnostics import ProgressBar
from skimage import filters, measure, morphology
from skimage.draw import ellipse

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
    use_feret_ellipse: bool = False
    feret_min_roundness: float = 0.0
    min_eye_separation: float = 4.0
    max_eye_separation: float = 80.0
    min_eye_separation: float = 4.0
    keypoint_run: Optional[str] = None
    crop_run: Optional[str] = None


def _set_config_value(cfg: EyeSegmentationConfig, key: str, value: Any) -> None:
    if value is None:
        if key in {"max_area", "pre_threshold"}:
            setattr(cfg, key, None)
        return
    if key in {"roi_padding", "threshold_block_size", "min_area", "closing_radius", "opening_radius", "contour_min_points"}:
        setattr(cfg, key, int(value))
    elif key in {"max_area", "pre_threshold"}:
        setattr(cfg, key, int(value))
    elif key in {"threshold_offset"}:
        setattr(cfg, key, float(value))
    elif key in {"feret_min_roundness"}:
        setattr(cfg, key, float(np.clip(value, 0.0, 1.0)))
    elif key in {"min_eye_separation", "max_eye_separation"}:
        if value is None:
            setattr(cfg, key, None)
        else:
            setattr(cfg, key, float(max(0.0, value)))
    elif key == "min_eye_separation":
        setattr(cfg, key, float(max(0.0, value)))
    elif key == "use_feret_ellipse":
        setattr(cfg, key, bool(value))
    else:
        setattr(cfg, key, value)


def _apply_tuned_parameters(
    root: zarr.Group,
    cfg: EyeSegmentationConfig,
    console: Optional[Console] = None,
) -> EyeSegmentationConfig:
    if "analysis_metadata" not in root:
        return cfg
    analysis_meta = root["analysis_metadata"]
    attrs = analysis_meta.attrs
    if "eye_mask_tuning" not in attrs:
        return cfg

    tuning = attrs["eye_mask_tuning"]
    tuned_params = tuning.get("tuned_parameters", {}) if isinstance(tuning, dict) else {}
    timestamp = tuning.get("tuned_timestamp") if isinstance(tuning, dict) else None

    for key, value in tuned_params.items():
        target_key = key
        if key == "min_roundness":
            target_key = "feret_min_roundness"
        if hasattr(cfg, target_key):
            _set_config_value(cfg, target_key, value)

    if cfg.use_feret_ellipse and cfg.feret_min_roundness is None:
        cfg.feret_min_roundness = 0.0

    if console is not None:
        ts_msg = f" (saved {timestamp})" if timestamp else ""
        console.print(f"[cyan]Using eye mask tuning from analysis_metadata{ts_msg}[/cyan]")

    return cfg


def _apply_overrides(cfg: EyeSegmentationConfig, overrides: Optional[Dict[str, Any]]) -> EyeSegmentationConfig:
    if not overrides:
        return cfg
    for key, value in overrides.items():
        target_key = "feret_min_roundness" if key == "min_roundness" else key
        if hasattr(cfg, target_key):
            _set_config_value(cfg, target_key, value)
    return cfg


def _prepare_run_group(root: zarr.Group, console: Console) -> Tuple[zarr.Group, str]:
    group, name = get_run_group(root, "eye_masks", console=console, create_new=True)
    return group, name


def _local_threshold(patch: np.ndarray, config: EyeSegmentationConfig) -> np.ndarray:
    block_size = int(config.threshold_block_size)
    block = max(3, block_size | 1)
    thresh = filters.threshold_local(patch, block_size=block, offset=config.threshold_offset)
    binary = patch < thresh
    return binary


def _select_region(
    binary: np.ndarray, center: Tuple[float, float], config: EyeSegmentationConfig
) -> Tuple[Optional[np.ndarray], Optional[Dict[str, Any]]]:
    if config.closing_radius > 0:
        binary = morphology.binary_closing(binary, morphology.disk(config.closing_radius))
    if config.opening_radius > 0:
        binary = morphology.binary_opening(binary, morphology.disk(config.opening_radius))

    labeled = measure.label(binary)
    if labeled.max() == 0:
        return None, None

    regions = measure.regionprops(labeled)
    if not regions:
        return None, None

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
        return None, None

    region_mask = labeled == best.label

    if config.use_feret_ellipse:
        feret_mask, info = _feret_mask_from_region(region_mask, config.feret_min_roundness)
        if feret_mask is not None:
            return feret_mask, info

    return region_mask, None


def _extract_contour(mask: np.ndarray, min_points: int) -> Optional[np.ndarray]:
    contours = measure.find_contours(mask.astype(float), 0.5)
    if not contours:
        return None
    contour = max(contours, key=lambda c: c.shape[0])
    if contour.shape[0] < min_points:
        return None
    return contour[:, ::-1]  # Convert to (x, y)


def _calculate_max_feret(contour: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], float]:
    max_dist = 0.0
    p1 = p2 = None
    num_points = contour.shape[0]
    for i in range(num_points):
        for j in range(i + 1, num_points):
            dist = np.linalg.norm(contour[i] - contour[j])
            if dist > max_dist:
                max_dist = dist
                p1, p2 = contour[i], contour[j]
    return p1, p2, max_dist


def _create_feret_ellipse_mask(
    contour: np.ndarray, shape: Tuple[int, int]
) -> Tuple[Optional[np.ndarray], float, Optional[np.ndarray], Optional[np.ndarray]]:
    p1, p2, max_dist = _calculate_max_feret(contour)
    if p1 is None or p2 is None or max_dist <= 0:
        return None, 0.0, None, None

    center = (p1 + p2) / 2.0
    cx, cy = center  # contour already in (x, y)

    feret_vec = p2 - p1
    feret_len = np.linalg.norm(feret_vec)
    if feret_len == 0:
        return None, 0.0, None, None
    orientation = np.arctan2(feret_vec[1], feret_vec[0])
    major_len = feret_len

    feret_vec_norm = feret_vec / feret_len
    perp_vec = np.array([-feret_vec_norm[1], feret_vec_norm[0]])
    projections = np.dot(contour - center, perp_vec)
    minor_len = np.max(projections) - np.min(projections)
    roundness = float(np.clip(minor_len / major_len if major_len > 0 else 0.0, 0.0, 1.0))

    major_pts = np.array([p1, p2], dtype=np.float32)
    midpoint = (p1 + p2) / 2.0
    if minor_len > 0:
        perp_unit = perp_vec / np.linalg.norm(perp_vec)
        half_minor = minor_len / 2.0
        minor_pts = np.array(
            [
                midpoint + perp_unit * half_minor,
                midpoint - perp_unit * half_minor,
            ],
            dtype=np.float32,
        )
    else:
        minor_pts = np.array(
            [
                midpoint,
                midpoint,
            ],
            dtype=np.float32,
        )

    mask = np.zeros(shape, dtype=bool)
    try:
        rr, cc = ellipse(
            cy,
            cx,
            minor_len / 2.0,
            major_len / 2.0,
            shape=shape,
            rotation=-orientation,
        )
        mask[rr, cc] = True
    except Exception:
        return None, roundness, None, None

    return mask, roundness, major_pts, minor_pts


def _feret_mask_from_region(
    region_mask: np.ndarray, min_roundness: float
) -> Tuple[Optional[np.ndarray], Optional[Dict[str, Any]]]:
    contours = measure.find_contours(region_mask.astype(float), 0.5)
    if not contours:
        return None, None
    best_contour = max(contours, key=lambda c: c.shape[0])
    if best_contour.shape[0] < 5:
        return None, None
    contour_xy = best_contour[:, ::-1]
    feret_mask, roundness, major_pts, minor_pts = _create_feret_ellipse_mask(contour_xy, region_mask.shape)
    if feret_mask is None:
        return None, None
    if roundness < min_roundness:
        return None, None
    info = {
        "roundness": roundness,
        "major_pts": major_pts,
        "minor_pts": minor_pts,
    }
    return feret_mask, info


def _process_roi_data(
    idx: int,
    roi_img: np.ndarray,
    kp: np.ndarray,
    success_flag: bool,
    cfg: EyeSegmentationConfig,
) -> Dict[str, Any]:
    roi_h, roi_w = roi_img.shape
    masks = [np.zeros((roi_h, roi_w), dtype=np.uint8) for _ in range(2)]
    ellipse_rows = [np.full(5, np.nan, dtype=np.float32) for _ in range(2)]
    ellipse_success = [False, False]
    contours = [None, None]
    feret_major = [np.full(4, np.nan, dtype=np.float32) for _ in range(2)]
    feret_minor = [np.full(4, np.nan, dtype=np.float32) for _ in range(2)]
    feret_roundness = [np.nan, np.nan]
    centroids_roi = [None, None]
    reject_reason: Optional[str] = None
    separation_value = np.nan

    if not success_flag:
        return {
            "index": idx,
            "masks": masks,
            "ellipse_params": ellipse_rows,
            "ellipse_success": ellipse_success,
            "contours": contours,
            "feret_major": feret_major,
            "feret_minor": feret_minor,
            "feret_roundness": feret_roundness,
            "reject_reason": "keypoint_fail",
        }

    for eye_idx in (0, 1):
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
        if patch.size == 0:
            continue

        binary = _local_threshold(patch, cfg)
        if cfg.pre_threshold is not None:
            base_mask = patch < cfg.pre_threshold
            binary = np.logical_and(binary, base_mask)

        region_mask, feret_info = _select_region(binary, (cx - x0, cy - y0), cfg)
        if region_mask is None:
            reject_reason = reject_reason or "no_region"
            continue

        masks[eye_idx][y0:y1, x0:x1][region_mask] = 1

        region = measure.regionprops(region_mask.astype(int))[0]
        centroid_local = region.centroid
        centroids_roi[eye_idx] = (
            float(x0 + centroid_local[1]),
            float(y0 + centroid_local[0]),
        )
        ellipse_rows[eye_idx] = [
            float(x0 + centroid_local[1]),
            float(y0 + centroid_local[0]),
            float(region.major_axis_length),
            float(region.minor_axis_length),
            float(np.rad2deg(region.orientation)),
        ]
        ellipse_success[eye_idx] = True

        contour = measure.find_contours(region_mask.astype(float), 0.5)
        if contour:
            best = max(contour, key=lambda c: c.shape[0])
            if best.shape[0] >= 5:
                best = best[:, ::-1]
                best[:, 0] += x0
                best[:, 1] += y0
                contours[eye_idx] = best.astype(np.float32)

        if feret_info is not None and feret_info.get("major_pts") is not None:
            roundness_val = feret_info.get("roundness", np.nan)
            major_pts_local = feret_info["major_pts"]
            minor_pts_local = feret_info["minor_pts"]
            if major_pts_local is not None:
                major_pts = major_pts_local + np.array([x0, y0], dtype=np.float32)
                feret_major[eye_idx] = np.array(
                    [major_pts[0, 0], major_pts[0, 1], major_pts[1, 0], major_pts[1, 1]],
                    dtype=np.float32,
                )
            if minor_pts_local is not None:
                minor_pts = minor_pts_local + np.array([x0, y0], dtype=np.float32)
                feret_minor[eye_idx] = np.array(
                    [minor_pts[0, 0], minor_pts[0, 1], minor_pts[1, 0], minor_pts[1, 1]],
                    dtype=np.float32,
                )
            feret_roundness[eye_idx] = float(roundness_val)

    valid = all(ellipse_success)
    if valid:
        overlap = np.logical_and(masks[0], masks[1]).any()
        if overlap:
            reject_reason = "overlap"
            valid = False
        else:
            if centroids_roi[0] is not None and centroids_roi[1] is not None:
                separation = float(
                    np.hypot(
                        centroids_roi[0][0] - centroids_roi[1][0],
                        centroids_roi[0][1] - centroids_roi[1][1],
                    )
                )
                separation_value = separation
                if cfg.min_eye_separation is not None and separation < cfg.min_eye_separation:
                    reject_reason = "too_close"
                    valid = False
                elif cfg.max_eye_separation is not None and separation > cfg.max_eye_separation:
                    reject_reason = "too_far"
                    valid = False
            else:
                separation_value = np.nan

    if not valid:
        ellipse_success = [False, False]
        if reject_reason is None:
            reject_reason = "incomplete"

    if not all(ellipse_success):
        zero_template = np.zeros_like(masks[0], dtype=np.uint8)
        masks = [zero_template.copy(), zero_template.copy()]
        ellipse_rows = [np.full(5, np.nan, dtype=np.float32) for _ in range(2)]
        ellipse_success = [False, False]
        contours = [None, None]
        feret_major = [np.full(4, np.nan, dtype=np.float32) for _ in range(2)]
        feret_minor = [np.full(4, np.nan, dtype=np.float32) for _ in range(2)]
        feret_roundness = [np.nan, np.nan]

    return {
        "index": idx,
        "masks": masks,
        "ellipse_params": ellipse_rows,
        "ellipse_success": ellipse_success,
        "contours": contours,
        "feret_major": feret_major,
        "feret_minor": feret_minor,
        "feret_roundness": feret_roundness,
        "eye_separation": separation_value,
        "reject_reason": reject_reason,
    }


def _process_roi_chunk(
    indices: List[int],
    zarr_path: str,
    roi_path: str,
    kp_path: str,
    success_path: str,
    cfg_dict: Dict[str, Any],
) -> List[Dict[str, Any]]:
    cfg_local = EyeSegmentationConfig(**cfg_dict)
    root = zarr.open(zarr_path, mode="r")
    roi_ds = root[roi_path]
    kp_ds = root[kp_path]
    success_ds = root[success_path]

    results: List[Dict[str, Any]] = []
    for idx in indices:
        roi_img = np.asarray(roi_ds[idx])
        kp = np.asarray(kp_ds[idx])
        success_flag = bool(success_ds[idx])
        results.append(_process_roi_data(idx, roi_img, kp, success_flag, cfg_local))
    return results


def segment_eye_masks(
    zarr_path: str,
    config_dict: Optional[Dict] = None,
    console: Optional[Console] = None,
    scheduler: str = "threads",
    num_workers: Optional[int] = None,
) -> str:
    console = console or Console()
    stage_start = time.perf_counter()

    root = zarr.open(zarr_path, mode="a")

    cfg = EyeSegmentationConfig()
    cfg = _apply_tuned_parameters(root, cfg, console)
    cfg = _apply_overrides(cfg, config_dict)
    if cfg.threshold_block_size % 2 == 0:
        cfg.threshold_block_size += 1
    feret_round = cfg.feret_min_roundness if cfg.feret_min_roundness is not None else 0.0
    cfg.feret_min_roundness = float(np.clip(feret_round, 0.0, 1.0))

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

    total_rois = roi_images.shape[0]
    roi_h, roi_w = roi_images.shape[1:3]

    masks = np.zeros((total_rois, 2, roi_h, roi_w), dtype=np.uint8)
    ellipse_params = np.full((total_rois, 2, 5), np.nan, dtype=np.float32)
    ellipse_success = np.zeros((total_rois, 2), dtype=bool)
    feret_axes_major = np.full((total_rois, 2, 4), np.nan, dtype=np.float32)
    feret_axes_minor = np.full((total_rois, 2, 4), np.nan, dtype=np.float32)
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

    run_group, run_name = _prepare_run_group(root, console)

    roi_results: List[Dict[str, Any]] = []
    cfg_dict = asdict(cfg)
    roi_dataset_path = f"crop_runs/{crop_run}/roi_images"
    kp_dataset_path = f"keypoints_runs/{keypoint_run}/keypoints_roi"
    success_dataset_path = f"keypoints_runs/{keypoint_run}/detection_success"
    if total_rois > 0:
        default_workers = os.cpu_count() or 4
        worker_count = num_workers or min(default_workers, 16)
        indices = list(range(total_rois))
        chunk_size = max(64, min(1024, (total_rois + worker_count - 1) // worker_count))
        chunks = [indices[i:i + chunk_size] for i in range(0, total_rois, chunk_size)]

        tasks = [
            delayed(_process_roi_chunk)(
                chunk,
                zarr_path,
                roi_dataset_path,
                kp_dataset_path,
                success_dataset_path,
                cfg_dict,
            )
            for chunk in chunks
        ]

        scheduler_key = (scheduler or "threads").lower()
        if scheduler_key in {"single-thread", "single_thread"}:
            scheduler_key = "single-threaded"
        if scheduler_key not in {"threads", "processes", "distributed", "single-threaded"}:
            console.print(f"[yellow]Unknown scheduler '{scheduler_key}', defaulting to 'threads'[/yellow]")
            scheduler_key = "threads"

        client = None
        cluster = None
        try:
            if scheduler_key == "distributed":
                try:
                    from dask.distributed import Client, LocalCluster
                except ImportError:
                    console.print("[yellow]dask[distributed] not installed; falling back to 'threads'.[/yellow]")
                    scheduler_key = "threads"
                else:
                    dashboard_addr = os.environ.get("PALETTE_DASK_DASHBOARD", ":0")
                    cluster = LocalCluster(
                        n_workers=worker_count,
                        threads_per_worker=1,
                        processes=True,
                        memory_limit="auto",
                        dashboard_address=dashboard_addr,
                    )
                    client = Client(cluster)
                    dash_link = getattr(client, "dashboard_link", None)
                    if dash_link:
                        console.print(f"[cyan]Dask dashboard:[/cyan] [link={dash_link}]{dash_link}[/link]")
                    else:
                        console.print(
                            "[yellow]Dask dashboard unavailable. Install bokeh>=3 or set "
                            "PALETTE_DASK_DASHBOARD=\":8787\" to expose it.[/yellow]"
                        )
                    console.print(
                        f"[cyan]eye_masks using distributed scheduler with {len(client.scheduler_info()['workers'])} workers[/cyan]"
                    )
                    futures = client.compute(tasks, sync=False)
                    gathered = client.gather(futures)
                    roi_results = [item for chunk_result in gathered for item in chunk_result]

            if not roi_results:
                if scheduler_key == "single-threaded" or total_rois == 1:
                    for chunk in chunks:
                        roi_results.extend(
                            _process_roi_chunk(
                                chunk,
                                zarr_path,
                                roi_dataset_path,
                                kp_dataset_path,
                                success_dataset_path,
                                cfg_dict,
                            )
                        )
                else:
                    compute_kwargs: Dict[str, Any] = {"scheduler": scheduler_key}
                    if num_workers:
                        compute_kwargs["num_workers"] = num_workers
                    with ProgressBar():
                        computed = dask.compute(*tasks, **compute_kwargs)
                    for chunk_result in computed:
                        roi_results.extend(chunk_result)
        except Exception:
            if client is not None:
                try:
                    client.close(timeout=5)
                except Exception:
                    pass
            if cluster is not None:
                try:
                    cluster.close(timeout=5)
                except Exception:
                    pass
            raise

    roi_results.sort(key=lambda r: r["index"])
    overlap_rejects = 0
    proximity_rejects = 0
    distance_rejects = 0

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeRemainingColumn(),
        console=console,
    )

    with progress:
        aggregate_task = progress.add_task(
            "[cyan]Aggregating eye masks...[/cyan]", total=len(roi_results)
        )
        for result in roi_results:
            idx = result["index"]
            reason = result.get("reject_reason")
            if reason == "overlap":
                overlap_rejects += 1
            elif reason == "too_close":
                proximity_rejects += 1
            elif reason == "too_far":
                distance_rejects += 1
            for eye_idx in (0, 1):
                masks[idx, eye_idx] = result["masks"][eye_idx]
                if result["ellipse_success"][eye_idx]:
                    ellipse_success[idx, eye_idx] = True
                    ellipse_params[idx, eye_idx] = result["ellipse_params"][eye_idx]
                feret_axes_major[idx, eye_idx] = np.asarray(
                    result["feret_major"][eye_idx], dtype=np.float32
                )
                feret_axes_minor[idx, eye_idx] = np.asarray(
                    result["feret_minor"][eye_idx], dtype=np.float32
                )
                feret_roundness[idx, eye_idx] = result["feret_roundness"][eye_idx]
                contour = result["contours"][eye_idx]
                if contour is not None:
                    contour = np.asarray(contour, dtype=np.float32)
                    if eye_idx == 0:
                        left_ptr[idx] = left_total
                        left_len[idx] = contour.shape[0]
                        left_total += contour.shape[0]
                        left_points.append(contour)
                    else:
                        right_ptr[idx] = right_total
                        right_len[idx] = contour.shape[0]
                        right_total += contour.shape[0]
                        right_points.append(contour)

            eye_separation[idx] = result.get("eye_separation", np.nan)

            progress.update(aggregate_task, advance=1)

    left_concat = np.concatenate(left_points, axis=0) if left_points else np.zeros((0, 2), dtype=np.float32)
    right_concat = np.concatenate(right_points, axis=0) if right_points else np.zeros((0, 2), dtype=np.float32)
    left_store = left_concat if left_concat.size > 0 else np.zeros((1, 2), dtype=np.float32)
    right_store = right_concat if right_concat.size > 0 else np.zeros((1, 2), dtype=np.float32)

    if client is not None:
        try:
            client.close(timeout=5)
        except Exception:
            pass
    if cluster is not None:
        try:
            cluster.close(timeout=5)
        except Exception:
            pass

    run_group.create_array(
        "masks_roi",
        data=masks,
        chunks=(min(512, total_rois), 2, roi_h, roi_w),
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
    run_group.create_array(
        "feret_axes_major",
        data=feret_axes_major,
        chunks=(min(1024, total_rois), 2, 4),
        overwrite=True,
    )
    run_group.create_array(
        "feret_axes_minor",
        data=feret_axes_minor,
        chunks=(min(1024, total_rois), 2, 4),
        overwrite=True,
    )
    run_group.create_array(
        "feret_roundness",
        data=feret_roundness,
        chunks=(min(1024, total_rois), 2),
        overwrite=True,
    )
    run_group.create_array(
        "eye_separation",
        data=eye_separation,
        chunks=(min(1024, total_rois),),
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

    both_eyes_success_count = int(np.all(ellipse_success, axis=1).sum())
    dual_eye_success_rate = (
        float(both_eyes_success_count / total_rois) if total_rois > 0 else float("nan")
    )

    if overlap_rejects or proximity_rejects or distance_rejects:
        min_sep = cfg.min_eye_separation if cfg.min_eye_separation is not None else 0.0
        max_sep_str = (
            f"{cfg.max_eye_separation:.1f}"
            if cfg.max_eye_separation is not None
            else "∞"
        )
        console.print(
            f"[yellow]Rejected {overlap_rejects} ROI(s) due to overlap, {proximity_rejects} ROI(s) < {min_sep:.1f}px, "
            f"and {distance_rejects} ROI(s) > {max_sep_str}px.[/yellow]"
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
    run_group.attrs["rejected_overlap"] = int(overlap_rejects)
    run_group.attrs["rejected_too_close"] = int(proximity_rejects)
    run_group.attrs["rejected_too_far"] = int(distance_rejects)
    run_group.attrs["min_eye_separation"] = float(cfg.min_eye_separation if cfg.min_eye_separation is not None else 0.0)
    run_group.attrs["max_eye_separation"] = float(cfg.max_eye_separation if cfg.max_eye_separation is not None else np.inf)
    run_group.attrs["successful_roi_pairs"] = int(both_eyes_success_count)
    run_group.attrs["successful_roi_pair_rate"] = float(dual_eye_success_rate)

    duration = time.perf_counter() - stage_start
    run_group.attrs['duration_seconds'] = float(duration)

    console.print(
        f"[green]✓[/green] Eye masks saved as [cyan]eye_masks_runs/{run_name}[/cyan] "
        f"({ellipse_success.sum()} successful eyes, {both_eyes_success_count}/{total_rois} ROI pairs) in {duration:.1f}s"
    )

    return run_name


__all__ = ["EyeSegmentationConfig", "segment_eye_masks"]
