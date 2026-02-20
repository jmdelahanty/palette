"""Traditional eye segmentation on ROI crops using existing keypoints."""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from collections import Counter
import numpy as np
import yaml
import zarr
from zarr.core.dtype import VariableLengthUTF8
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
import dask
from dask import delayed
from dask.diagnostics import ProgressBar
from skimage import filters, measure, morphology
from skimage.measure import EllipseModel
from ..shared.provenance_attrs import build_source_keypoints_attrs
from ..shared.row_alignment import assert_row_alignment
from ..shared.zarr.schema import get_run_group
from ..utils.system import get_environment_info, get_git_info


@dataclass
class EyeSegmentationConfig:
    roi_padding: int = 12
    sobel_strength: float = 0.0
    pre_threshold: Optional[int] = None
    min_area: int = 15
    max_area: Optional[int] = None
    min_circularity: Optional[float] = None
    closing_radius: int = 3
    opening_radius: int = 1
    contour_min_points: int = 5
    min_eye_separation: float = 4.0
    max_eye_separation: float = 80.0
    keypoint_run: Optional[str] = None
    crop_run: Optional[str] = None


def _resolve_keypoint_group(
    root: zarr.Group,
    keypoint_run: Optional[str],
) -> Tuple[zarr.Group, str, str]:
    refined = root.get("refined_keypoints_runs")
    raw = root.get("keypoints_runs")

    if keypoint_run:
        if refined is not None and keypoint_run in refined:
            return refined[keypoint_run], keypoint_run, "refined_keypoints_runs"
        if raw is not None and keypoint_run in raw:
            return raw[keypoint_run], keypoint_run, "keypoints_runs"
        raise ValueError(f"Keypoint run '{keypoint_run}' not found in refined or raw runs.")

    refined_latest = refined.attrs.get("latest") if refined is not None else None
    raw_latest = raw.attrs.get("latest") if raw is not None else None

    if refined is not None and refined_latest in refined:
        return refined[refined_latest], refined_latest, "refined_keypoints_runs"
    if raw is not None and raw_latest in raw:
        return raw[raw_latest], raw_latest, "keypoints_runs"
    raise ValueError("No keypoint runs found; run keypoints stage first.")


def _set_config_value(cfg: EyeSegmentationConfig, key: str, value: Any) -> None:
    if value is None:
        if key in {"max_area", "pre_threshold", "min_circularity"}:
            setattr(cfg, key, None)
        return
    if key in {"roi_padding", "min_area", "closing_radius", "opening_radius", "contour_min_points"}:
        setattr(cfg, key, int(value))
    elif key in {"max_area", "pre_threshold"}:
        setattr(cfg, key, int(value))
    elif key == "min_circularity":
        setattr(cfg, key, float(np.clip(value, 0.0, 1.0)))
    elif key == "sobel_strength":
        setattr(cfg, key, float(np.clip(value, 0.0, 1.0)))
    elif key in {"min_eye_separation", "max_eye_separation"}:
        if value is None:
            setattr(cfg, key, None)
        else:
            setattr(cfg, key, float(max(0.0, value)))
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
        if hasattr(cfg, key):
            _set_config_value(cfg, key, value)

    if console is not None:
        ts_msg = f" (saved {timestamp})" if timestamp else ""
        console.print(f"[cyan]Using eye mask tuning from analysis_metadata{ts_msg}[/cyan]")

    return cfg


def _apply_overrides(cfg: EyeSegmentationConfig, overrides: Optional[Dict[str, Any]]) -> EyeSegmentationConfig:
    if not overrides:
        return cfg
    for key, value in overrides.items():
        if hasattr(cfg, key):
            _set_config_value(cfg, key, value)
    return cfg


def _prepare_run_group(root: zarr.Group, console: Console) -> Tuple[zarr.Group, str]:
    group, name = get_run_group(root, "eye_masks", console=console, create_new=True)
    return group, name


def _validate_input_row_alignment(
    *,
    crop_group: zarr.Group,
    crop_run: str,
    kp_group: zarr.Group,
    keypoint_group_name: str,
    keypoint_run: str,
    success_dataset_name: str,
    total_rois: int,
) -> None:
    keypoints_roi = kp_group.get("keypoints_roi")
    if keypoints_roi is None:
        raise ValueError(f"Keypoint run '{keypoint_run}' missing 'keypoints_roi'.")
    success_arr = kp_group.get(success_dataset_name)
    if success_arr is None:
        raise ValueError(f"Keypoint run '{keypoint_run}' missing '{success_dataset_name}'.")

    assert_row_alignment(
        total_rois,
        (
            (f"crop_runs/{crop_run}/roi_images", crop_group["roi_images"]),
            (f"{keypoint_group_name}/{keypoint_run}/keypoints_roi", keypoints_roi),
            (f"{keypoint_group_name}/{keypoint_run}/{success_dataset_name}", success_arr),
            (f"crop_runs/{crop_run}/frame_indices", crop_group.get("frame_indices")),
            (f"crop_runs/{crop_run}/detection_indices", crop_group.get("detection_indices")),
            (f"crop_runs/{crop_run}/detection_source", crop_group.get("detection_source")),
        ),
        stage="eye_segmentation input",
    )


def _normalize_chunks_for_data(
    source_chunks: Optional[Tuple[int, ...] | int],
    data_shape: Tuple[int, ...],
) -> Tuple[int, ...]:
    if len(data_shape) == 0:
        return (1,)
    if isinstance(source_chunks, int):
        source_chunks = (source_chunks,)
    if not source_chunks:
        return tuple(max(1, min(dim, 1024)) for dim in data_shape)
    chunk_dims: List[int] = []
    for axis, dim in enumerate(data_shape):
        source_chunk = source_chunks[axis] if axis < len(source_chunks) else source_chunks[-1]
        chunk_dims.append(int(max(1, min(dim, int(source_chunk)))))
    return tuple(chunk_dims)


def _copy_lineage_arrays_from_crop_with_keypoint_fallback(
    *,
    run_group: zarr.Group,
    crop_group: zarr.Group,
    crop_run: str,
    kp_group: zarr.Group,
    keypoint_group_name: str,
    keypoint_run: str,
    total_rois: int,
    console: Console,
) -> None:
    for array_name in ("frame_indices", "detection_indices", "frame_counts"):
        crop_arr = crop_group.get(array_name)
        kp_arr = kp_group.get(array_name)

        crop_data = np.asarray(crop_arr[:]) if crop_arr is not None else None
        kp_data = np.asarray(kp_arr[:]) if kp_arr is not None else None

        if crop_data is not None and kp_data is not None:
            if crop_data.shape != kp_data.shape or not np.array_equal(crop_data, kp_data):
                raise ValueError(
                    "eye_masks lineage mismatch for "
                    f"'{array_name}': crop_runs/{crop_run}/{array_name} and "
                    f"{keypoint_group_name}/{keypoint_run}/{array_name} differ"
                )

        chosen_data: Optional[np.ndarray]
        chosen_chunks: Optional[Tuple[int, ...]]
        if crop_data is not None:
            chosen_data = crop_data
            chosen_chunks = getattr(crop_arr, "chunks", None)
        elif kp_data is not None:
            chosen_data = kp_data
            chosen_chunks = getattr(kp_arr, "chunks", None)
            console.print(
                "[yellow]Crop run missing "
                f"'{array_name}'; falling back to "
                f"{keypoint_group_name}/{keypoint_run}/{array_name}.[/yellow]"
            )
        else:
            console.print(
                "[yellow]Missing lineage array "
                f"'{array_name}' in both crop_runs/{crop_run} and "
                f"{keypoint_group_name}/{keypoint_run}; skipping.[/yellow]"
            )
            continue

        if chosen_data.ndim == 0:
            raise ValueError(
                f"eye_masks lineage array '{array_name}' must be 1D, got scalar value"
            )
        if array_name in {"frame_indices", "detection_indices"} and int(chosen_data.shape[0]) != int(total_rois):
            raise ValueError(
                f"eye_masks lineage array '{array_name}' has {chosen_data.shape[0]} rows, expected {total_rois}"
            )
        if array_name == "frame_counts" and int(np.sum(chosen_data, dtype=np.int64)) != int(total_rois):
            raise ValueError(
                f"eye_masks lineage array 'frame_counts' sums to "
                f"{int(np.sum(chosen_data, dtype=np.int64))}, expected {total_rois}"
            )

        if array_name in run_group:
            del run_group[array_name]
        run_group.create_array(
            array_name,
            data=chosen_data,
            chunks=_normalize_chunks_for_data(chosen_chunks, chosen_data.shape),
            overwrite=True,
        )
        if crop_data is not None:
            console.print(f"[dim]Copied {array_name} from crop run[/dim]")


def _apply_sobel_filter(patch: np.ndarray, strength: float) -> np.ndarray:
    if strength <= 0.0:
        return patch

    patch_float = patch.astype(np.float32) / 255.0
    sobel_response = filters.sobel(patch_float)
    max_resp = float(np.max(sobel_response))
    if max_resp > 0.0:
        sobel_norm = sobel_response / max_resp
    else:
        sobel_norm = np.zeros_like(sobel_response)

    filtered = np.clip(patch_float - strength * sobel_norm, 0.0, 1.0)
    return (filtered * 255.0).astype(patch.dtype, copy=False)


def _global_threshold(patch: np.ndarray, config: EyeSegmentationConfig) -> Tuple[np.ndarray, np.ndarray]:
    filtered_patch = _apply_sobel_filter(patch, config.sobel_strength)
    try:
        threshold_value = float(filters.threshold_otsu(filtered_patch))
    except ValueError:
        threshold_value = float(np.mean(filtered_patch))
    binary = filtered_patch < threshold_value
    return binary, filtered_patch


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
    area_pass_count = 0
    circularity_fail_count = 0
    min_circularity = config.min_circularity
    for region in regions:
        area = region.area
        if area < config.min_area:
            continue
        if config.max_area is not None and area > config.max_area:
            continue
        area_pass_count += 1
        if min_circularity is not None:
            perimeter = float(region.perimeter)
            if perimeter <= 0.0:
                circularity_fail_count += 1
                continue
            circularity = float((4.0 * np.pi * float(area)) / (perimeter * perimeter))
            if circularity < min_circularity:
                circularity_fail_count += 1
                continue
        rcx, rcy = region.centroid
        dist = (rcx - cy) ** 2 + (rcy - cx) ** 2
        if best is None or dist < best_dist:
            best = region
            best_dist = dist

    if best is None:
        filtered_non_circular = (
            min_circularity is not None
            and area_pass_count > 0
            and circularity_fail_count == area_pass_count
        )
        return None, {"filtered_non_circular": filtered_non_circular}

    region_mask = labeled == best.label

    return region_mask, {"filtered_non_circular": False}


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

        binary, filtered_patch = _global_threshold(patch, cfg)
        if cfg.pre_threshold is not None:
            base_mask = filtered_patch < cfg.pre_threshold
            binary = np.logical_and(binary, base_mask)

        region_mask, region_meta = _select_region(binary, (cx - x0, cy - y0), cfg)
        if region_mask is None:
            if region_meta is not None and bool(region_meta.get("filtered_non_circular")):
                reject_reason = reject_reason or "non_circular"
            else:
                reject_reason = reject_reason or "no_region"
            continue

        masks[eye_idx][y0:y1, x0:x1][region_mask] = 1

        # Get centroid from regionprops (used for separation check)
        region = measure.regionprops(region_mask.astype(int))[0]
        centroid_local = region.centroid
        centroids_roi[eye_idx] = (
            float(x0 + centroid_local[1]),
            float(y0 + centroid_local[0]),
        )

        # Fit ellipse using EllipseModel (true least-squares contour fit)
        contour = measure.find_contours(region_mask.astype(float), 0.5)
        if contour:
            best = max(contour, key=lambda c: c.shape[0])
            if best.shape[0] >= cfg.contour_min_points:
                # contour is (row, col) = (y, x), need (x, y) for EllipseModel
                contour_xy = best[:, ::-1]
                ellipse_model = EllipseModel()
                if ellipse_model.estimate(contour_xy) and ellipse_model.params is not None:
                    xc, yc, a, b, theta = ellipse_model.params
                    # Ensure major >= minor (swap if needed, rotate theta by 90°)
                    if a < b:
                        a, b = b, a
                        theta = theta + np.pi / 2
                    # a, b are semi-axes; multiply by 2 for full axis lengths
                    ellipse_rows[eye_idx] = [
                        float(x0 + xc),
                        float(y0 + yc),
                        float(a * 2),
                        float(b * 2),
                        float(np.degrees(theta)),
                    ]
                    ellipse_success[eye_idx] = True

                # Store contour in ROI coords
                best_roi = best[:, ::-1].copy()
                best_roi[:, 0] += x0
                best_roi[:, 1] += y0
                contours[eye_idx] = best_roi.astype(np.float32)

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

    return {
        "index": idx,
        "masks": masks,
        "ellipse_params": ellipse_rows,
        "ellipse_success": ellipse_success,
        "contours": contours,
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
    root = zarr.open(zarr_path, mode="r", use_consolidated=False)
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

    root = zarr.open(zarr_path, mode="a", use_consolidated=False)

    cfg = EyeSegmentationConfig()
    cfg = _apply_tuned_parameters(root, cfg, console)
    cfg = _apply_overrides(cfg, config_dict)
    if "crop_runs" not in root:
        raise ValueError("crop_runs missing from Zarr; run crop stage first")
    crop_run = cfg.crop_run or root["crop_runs"].attrs.get("latest")
    if crop_run is None:
        raise ValueError("No crop run available")
    crop_group = root[f"crop_runs/{crop_run}"]

    if "keypoints_runs" not in root and "refined_keypoints_runs" not in root:
        raise ValueError("Keypoints runs missing from Zarr; run keypoints stage first")
    kp_group, keypoint_run, keypoint_group_name = _resolve_keypoint_group(root, cfg.keypoint_run)
    console.print(f"[dim]Using keypoints from {keypoint_group_name}/{keypoint_run}[/dim]")

    roi_images = crop_group["roi_images"]

    total_rois = roi_images.shape[0]
    roi_h, roi_w = roi_images.shape[1:3]

    masks = np.zeros((total_rois, 2, roi_h, roi_w), dtype=np.uint8)
    ellipse_params = np.full((total_rois, 2, 5), np.nan, dtype=np.float32)
    ellipse_success = np.zeros((total_rois, 2), dtype=bool)
    eye_separation = np.full((total_rois,), np.nan, dtype=np.float32)

    left_ptr = np.full((total_rois,), -1, dtype=np.int64)
    left_len = np.zeros((total_rois,), dtype=np.int32)
    right_ptr = np.full((total_rois,), -1, dtype=np.int64)
    right_len = np.zeros((total_rois,), dtype=np.int32)
    left_points: List[np.ndarray] = []
    right_points: List[np.ndarray] = []
    left_total = 0
    right_total = 0

    roi_results: List[Dict[str, Any]] = []
    cfg_dict = asdict(cfg)
    roi_dataset_path = f"crop_runs/{crop_run}/roi_images"
    kp_dataset_path = f"{keypoint_group_name}/{keypoint_run}/keypoints_roi"
    success_dataset_name = None
    for candidate in ("detection_success", "refined_success", "source_success"):
        if candidate in kp_group:
            success_dataset_name = candidate
            break
    if success_dataset_name is None:
        raise ValueError(
            f"Keypoint run '{keypoint_run}' missing detection/refinement success flags."
        )
    success_dataset_path = f"{keypoint_group_name}/{keypoint_run}/{success_dataset_name}"

    _validate_input_row_alignment(
        crop_group=crop_group,
        crop_run=str(crop_run),
        kp_group=kp_group,
        keypoint_group_name=keypoint_group_name,
        keypoint_run=keypoint_run,
        success_dataset_name=success_dataset_name,
        total_rois=int(total_rois),
    )

    crop_detection_source = crop_group.get("detection_source")
    detection_source = (
        crop_detection_source[:].astype(np.int8, copy=False)
        if crop_detection_source is not None
        else np.zeros(total_rois, dtype=np.int8)
    )
    run_group, run_name = _prepare_run_group(root, console)
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
    reason_strings: List[str] = ["clean"] * total_rois

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
            if reason:
                reason_strings[idx] = str(reason)
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
        "eye_separation",
        data=eye_separation,
        chunks=(min(1024, total_rois),),
        overwrite=True,
    )
    run_group.create_array(
        "detection_source",
        data=detection_source,
        chunks=(min(1024, total_rois),),
        overwrite=True,
    )
    if crop_detection_source is not None:
        console.print("[dim]Copied detection_source from crop run[/dim]")
    else:
        console.print("[yellow]Crop run missing detection_source; defaulting to zeros.[/yellow]")

    _copy_lineage_arrays_from_crop_with_keypoint_fallback(
        run_group=run_group,
        crop_group=crop_group,
        crop_run=str(crop_run),
        kp_group=kp_group,
        keypoint_group_name=keypoint_group_name,
        keypoint_run=str(keypoint_run),
        total_rois=int(total_rois),
        console=console,
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
    reason_ds = run_group.create_array(
        "reason",
        shape=(total_rois,),
        dtype=VariableLengthUTF8(),
        fill_value="clean",
        chunks=(min(1024, total_rois),),
        overwrite=True,
    )
    reason_ds[:] = np.array(reason_strings, dtype=object)

    both_eyes_success_count = int(np.all(ellipse_success, axis=1).sum())
    dual_eye_success_rate = (
        float(both_eyes_success_count / total_rois) if total_rois > 0 else float("nan")
    )
    reason_counts = dict(Counter(reason_strings))

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
    env_info = get_environment_info(
        include_all_packages=False,
        disk_path=str(zarr_path),
        collect_ip=False,
        capture_env_vars=False,
    )
    platform_info = env_info.get("platform", {})

    run_group.attrs.update(
        {
            "method": "traditional_eye_segmentation",
            "config": cfg.__dict__,
            "source_crop_run": crop_run,
            **build_source_keypoints_attrs(keypoint_run, include_legacy_alias=True),
            "source_keypoint_group": keypoint_group_name,
            "total_rois": total_rois,
            "successful_eyes": int(ellipse_success.sum()),
            "contours_left_count": int(left_concat.shape[0]),
            "contours_right_count": int(right_concat.shape[0]),
            "eye_labels": ["eye_left", "eye_right"],
            "git_commit": git_info.get("commit_hash", "unknown"),
            "git_branch": git_info.get("branch", "unknown"),
            "hostname": env_info["platform"].get("hostname", "unknown"),
            "ellipse_angle_units": "degrees",
            "ellipse_angle_ref": "EllipseModel theta, deg CCW from +x (true contour fit)",
            "ellipse_fit_method": "EllipseModel_least_squares",
            "reason_counts": reason_counts,
        }
    )
    provenance = {
        "stage": "eye_masks",
        "method": "traditional_eye_segmentation",
        "command": " ".join(sys.argv),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git": git_info,
        "environment": env_info.get("environment"),
        "platform": {
            "hostname": platform_info.get("hostname"),
            "system": platform_info.get("system"),
            "release": platform_info.get("release"),
            "python_version": platform_info.get("python_version"),
            "machine": platform_info.get("machine"),
        },
        "parameters": cfg.__dict__,
        "inputs": {
            "source_crop_run": crop_run,
            "source_keypoints_run": keypoint_run,
            "source_keypoint_group": keypoint_group_name,
            "frame_source": crop_group.attrs.get("video_source_type", "zarr"),
            "source_video_path": crop_group.attrs.get("video_source_path"),
        },
    }
    provenance = {k: v for k, v in provenance.items() if v is not None}
    run_group.attrs["provenance"] = provenance
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


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Command-line entry point for running traditional eye segmentation."""
    parser = argparse.ArgumentParser(
        description="Run traditional eye segmentation on ROI crops stored in a Palette Zarr archive.",
    )
    parser.add_argument(
        "zarr_path",
        type=Path,
        help="Path to the Palette Zarr archive.",
    )
    parser.add_argument(
        "--config",
        "-c",
        type=Path,
        help="Optional YAML file with eye mask parameters. Uses the 'eye_masks' section if present.",
    )
    parser.add_argument(
        "--scheduler",
        choices=["threads", "processes", "distributed", "single-threaded"],
        default="threads",
        help="Dask scheduler to use for ROI processing (default: threads).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Override the number of workers for parallel processing.",
    )
    parser.add_argument(
        "--crop-run",
        help="Specify a crop run name instead of using the latest available.",
    )
    parser.add_argument(
        "--keypoint-run",
        help="Specify a keypoint run name instead of using the latest available (checks refined first).",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override configuration values (may be repeated). Values use YAML parsing.",
    )

    args = parser.parse_args(argv)
    console = Console()

    config_dict: Dict[str, Any] = {}
    if args.config:
        if not args.config.exists():
            parser.error(f"Config file not found: {args.config}")
        try:
            with args.config.open("r", encoding="utf-8") as handle:
                loaded = yaml.safe_load(handle) or {}
        except yaml.YAMLError as exc:
            parser.error(f"Failed to parse config file {args.config}: {exc}")
        if not isinstance(loaded, dict):
            parser.error("Config file must contain a mapping/object at the top level.")
        config_section = loaded.get("eye_masks") if "eye_masks" in loaded else loaded
        if config_section is None:
            config_dict = {}
        elif not isinstance(config_section, dict):
            parser.error("eye_masks section in config must be a mapping/object.")
        else:
            config_dict = dict(config_section)

    if args.crop_run:
        config_dict["crop_run"] = args.crop_run
    if args.keypoint_run:
        config_dict["keypoint_run"] = args.keypoint_run

    for item in args.overrides:
        if "=" not in item:
            parser.error(f"Invalid override '{item}'. Expected format KEY=VALUE.")
        key, value_text = item.split("=", 1)
        key = key.strip()
        if not key:
            parser.error(f"Invalid override '{item}'. Key cannot be empty.")
        try:
            value = yaml.safe_load(value_text)
        except yaml.YAMLError as exc:
            parser.error(f"Failed to parse override '{item}': {exc}")
        config_dict[key] = value

    run_name = segment_eye_masks(
        zarr_path=str(args.zarr_path),
        config_dict=config_dict or None,
        console=console,
        scheduler=args.scheduler,
        num_workers=args.num_workers,
    )

    console.print(f"[green]Eye mask run:[/green] [cyan]eye_masks_runs/{run_name}[/cyan]")
    return 0


__all__ = ["EyeSegmentationConfig", "segment_eye_masks", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
