#!/usr/bin/env python3
"""Traditional swim-bladder segmentation on Palette crop runs."""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import dask
import numpy as np
import zarr
from dask import delayed
from dask.diagnostics import ProgressBar

from ..shared.crop_image_source import CropImageSource
from ..shared.provenance_attrs import build_source_crop_snapshot_attrs
from ..shared.row_lineage import copy_row_lineage_arrays
from ..shared.subject_mask_registry_status import emit_subject_mask_stage_completion
from ..shared.stage_provenance import build_stage_provenance, write_stage_provenance
from ..shared.subject_mask_chunks import subject_mask_metric_row_chunk, subject_mask_storage_chunks
from ..shared.subject_mask_component_provenance import write_subject_mask_component_provenance
from ..tune import subject_mask_tuner as subject_tuning
from ..tune import swim_bladder_mask_tuner as swim_tuning
from ..utils.system import get_environment_info, get_git_info
from ..utils.zarr_io import open_zarr_root
from ..visualization.visualize_swim_bladder_mask_patches import (
    _extract_patch_bounds,
    _load_keypoint_success_flags,
    _resolve_keypoint_group,
    _resolve_swim_bladder_keypoint_center,
    _validate_keypoint_group_alignment,
)
from .subject_segmentation import (
    SUBJECT_MASK_LABELS,
    SUBJECT_MASK_LABEL_SCHEMA,
    _coerce_roi_to_gray,
    _compute_channel_metrics,
    _snapshot_tuning_entry,
    _prepare_run_group,
)

try:
    from rich.console import Console
except Exception:  # pragma: no cover - rich is optional at runtime
    Console = None  # type: ignore


SUBJECT_MASK_AVAILABLE_CHANNELS = (False, False, True)
TUNING_OVERRIDE_KEYS = (
    "subject_method_family",
    "roi_padding",
    "pre_threshold",
    "sobel_strength",
    "min_area",
    "max_area",
    "min_circularity",
    "closing_radius",
    "opening_radius",
    "angle_step_degrees",
    "min_radius_px",
    "max_radius_px",
    "smoothing_sigma",
    "response_threshold",
    "max_missing_gap_degrees",
    "min_valid_ray_fraction",
    "gradient_mode",
    "prefilter_sigma",
)
VALID_SCHEDULERS = ("threads", "processes", "distributed", "single-threaded")
SWIM_ROW_STATUS_OK = np.int8(0)
SWIM_ROW_STATUS_MISSING_KEYPOINT = np.int8(1)
SWIM_ROW_STATUS_UNSUCCESSFUL_KEYPOINT = np.int8(2)
_SUBJECT_MASKS_STATUS_SOURCE = "runtime_swim_bladder_segmentation"


@dataclass
class SwimBladderSegmentationConfig:
    subject_method_family: str = swim_tuning.DEFAULT_METHOD_FAMILY
    roi_padding: int = swim_tuning.DEFAULT_ROI_PADDING
    pre_threshold: Optional[int] = swim_tuning.DEFAULT_PRE_THRESHOLD
    sobel_strength: float = swim_tuning.DEFAULT_SOBEL_STRENGTH
    min_area: int = swim_tuning.DEFAULT_MIN_AREA
    max_area: Optional[int] = swim_tuning.DEFAULT_MAX_AREA
    min_circularity: Optional[float] = swim_tuning.DEFAULT_MIN_CIRCULARITY
    closing_radius: int = swim_tuning.DEFAULT_CLOSING_RADIUS
    opening_radius: int = swim_tuning.DEFAULT_OPENING_RADIUS
    angle_step_degrees: int = 8
    min_radius_px: int = 3
    max_radius_px: int = swim_tuning.DEFAULT_ROI_PADDING
    smoothing_sigma: float = 1.5
    response_threshold: float = 0.12
    max_missing_gap_degrees: int = 40
    min_valid_ray_fraction: float = 0.55
    gradient_mode: str = "sobel_magnitude"
    prefilter_sigma: float = 1.0
    crop_run: Optional[str] = None
    keypoint_run: Optional[str] = None
    roi_cache_policy: str = "auto"
    roi_cache_dir: Optional[str] = None
    roi_cache_manifest: Optional[str] = None
    roi_live_acceleration: str = "auto"
    roi_live_gpu_chunk_frames: int = 32


def _print(console: Optional[Console], message: str) -> None:
    if console is not None:
        console.print(message)
    else:  # pragma: no cover
        print(message)


def _resolve_effective_method_name(method_family: str) -> str:
    normalized_family = swim_tuning._normalize_method_family(method_family)
    if normalized_family == swim_tuning.POLAR_BOUNDARY_METHOD_FAMILY:
        return "polar_boundary_center_seed"
    return "global_threshold_otsu"


def _collect_tuning_override_keys(overrides: Optional[Mapping[str, Any]]) -> list[str]:
    if not overrides:
        return []
    return sorted(str(key) for key in TUNING_OVERRIDE_KEYS if key in overrides)


def _apply_tuned_parameters(
    root: zarr.Group,
    cfg: SwimBladderSegmentationConfig,
    console: Optional[Console] = None,
) -> tuple[SwimBladderSegmentationConfig, Optional[dict[str, Any]]]:
    entry = subject_tuning._load_subject_tuning_entry_from_root(root, "swim_bladder")
    tuned = entry.get("tuned_parameters") if isinstance(entry, dict) else None
    if not isinstance(tuned, Mapping):
        return cfg, None

    cfg.subject_method_family = swim_tuning._normalize_method_family(
        entry.get("subject_method_family") if isinstance(entry, dict) else None
    )
    normalized = swim_tuning._normalize_swim_bladder_params(
        tuned,
        method_family=cfg.subject_method_family,
    )
    cfg.roi_padding = int(normalized["roi_padding"])
    if cfg.subject_method_family == swim_tuning.POLAR_BOUNDARY_METHOD_FAMILY:
        cfg.angle_step_degrees = int(normalized["angle_step_degrees"])
        cfg.min_radius_px = int(normalized["min_radius_px"])
        cfg.max_radius_px = int(normalized["max_radius_px"])
        cfg.smoothing_sigma = float(normalized["smoothing_sigma"])
        cfg.response_threshold = float(normalized["response_threshold"])
        cfg.max_missing_gap_degrees = int(normalized["max_missing_gap_degrees"])
        cfg.min_valid_ray_fraction = float(normalized["min_valid_ray_fraction"])
        cfg.gradient_mode = str(normalized["gradient_mode"])
        cfg.prefilter_sigma = float(normalized["prefilter_sigma"])
    else:
        cfg.pre_threshold = (
            int(normalized["pre_threshold"]) if normalized.get("pre_threshold") is not None else None
        )
        cfg.sobel_strength = float(normalized["sobel_strength"])
        cfg.min_area = int(normalized["min_area"])
        cfg.max_area = int(normalized["max_area"]) if normalized.get("max_area") is not None else None
        cfg.min_circularity = (
            float(normalized["min_circularity"])
            if normalized.get("min_circularity") is not None
            else None
        )
        cfg.closing_radius = int(normalized["closing_radius"])
        cfg.opening_radius = int(normalized["opening_radius"])

    if console is not None:
        timestamp = entry.get("tuned_timestamp") if isinstance(entry, dict) else None
        suffix = f" (saved {timestamp})" if timestamp else ""
        console.print(f"[cyan]Using swim-bladder tuning from analysis_metadata{suffix}[/cyan]")
    return cfg, dict(entry) if isinstance(entry, dict) else None


def _apply_overrides(
    cfg: SwimBladderSegmentationConfig,
    overrides: Optional[Mapping[str, Any]],
) -> SwimBladderSegmentationConfig:
    if not overrides:
        return cfg
    if "subject_method_family" in overrides:
        cfg.subject_method_family = swim_tuning._normalize_method_family(overrides["subject_method_family"])
    param_keys = set(TUNING_OVERRIDE_KEYS) - {"subject_method_family"}
    param_overrides = {key: overrides[key] for key in param_keys if key in overrides}
    if param_overrides:
        normalized = swim_tuning._normalize_swim_bladder_params(
            param_overrides,
            method_family=cfg.subject_method_family,
        )
        for key in param_keys:
            if key in param_overrides:
                setattr(cfg, key, normalized[key])
    if "crop_run" in overrides:
        cfg.crop_run = str(overrides["crop_run"]) if overrides["crop_run"] is not None else None
    if "keypoint_run" in overrides:
        cfg.keypoint_run = str(overrides["keypoint_run"]) if overrides["keypoint_run"] is not None else None
    return cfg


def _resolve_swim_bladder_point(
    keypoints_row: np.ndarray,
    keypoint_labels: Optional[Sequence[str]],
) -> Optional[tuple[float, float]]:
    center_xy, center_source = _resolve_swim_bladder_keypoint_center(
        keypoints_row,
        keypoint_labels,
        success_flag=True,
    )
    return tuple(center_xy) if center_source == "keypoint" and center_xy is not None else None


def _normalize_scheduler(scheduler: Optional[str]) -> str:
    scheduler_key = str(scheduler or "single-threaded").strip().lower()
    if scheduler_key in {"single-thread", "single_thread"}:
        scheduler_key = "single-threaded"
    if scheduler_key not in VALID_SCHEDULERS:
        scheduler_key = "single-threaded"
    return scheduler_key


def _parse_scheduler_arg(text: str) -> str:
    normalized = str(text or "").strip().lower()
    if normalized not in {"threads", "processes", "distributed", "single-threaded", "single-thread", "single_thread"}:
        raise argparse.ArgumentTypeError(
            "Invalid scheduler. Expected one of: threads, processes, distributed, "
            "single-threaded, single-thread, single_thread."
        )
    return _normalize_scheduler(text)


def _parse_roi_indices_arg(text: str) -> list[int]:
    raw = str(text or "").replace(" ", "")
    if not raw:
        raise argparse.ArgumentTypeError("ROI indices must not be empty.")
    try:
        return [int(token) for token in raw.split(",") if token]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid ROI indices '{text}'. Expected comma-separated integers."
        ) from exc


def _normalize_roi_indices(roi_indices: Optional[Sequence[int]], total_rois: int) -> list[int]:
    if roi_indices is None:
        return list(range(int(total_rois)))
    normalized: list[int] = []
    seen: set[int] = set()
    for raw_idx in roi_indices:
        idx = int(raw_idx)
        if idx < 0 or idx >= int(total_rois):
            raise ValueError(f"roi index {idx} is out of bounds for run with {int(total_rois)} rows.")
        if idx in seen:
            continue
        seen.add(idx)
        normalized.append(idx)
    if not normalized:
        raise ValueError("At least one ROI index is required.")
    return normalized


def _bootstrap_swim_row_status(
    *,
    keypoints_roi: np.ndarray,
    success_flags: np.ndarray,
    keypoint_labels: Optional[Sequence[str]],
) -> np.ndarray:
    total_rois = int(keypoints_roi.shape[0])
    status = np.zeros((total_rois,), dtype=np.int8)
    for row_idx in range(total_rois):
        if row_idx >= int(success_flags.shape[0]) or not bool(success_flags[row_idx]):
            status[row_idx] = SWIM_ROW_STATUS_UNSUCCESSFUL_KEYPOINT
            continue
        if _resolve_swim_bladder_point(np.asarray(keypoints_roi[row_idx], dtype=np.float32), keypoint_labels) is None:
            status[row_idx] = SWIM_ROW_STATUS_MISSING_KEYPOINT
    return status


def _compute_swim_summary_statistics(
    *,
    roi_count: int,
    run_name: str,
    crop_run: str,
    kp_group_name: str,
    kp_run_name: str,
    duration_seconds: float,
    created_at_utc: str,
    updated_at_utc: Optional[str],
    mask_present_swim: np.ndarray,
    area_px_swim: np.ndarray,
    prob_max_swim: np.ndarray,
    swim_row_status: np.ndarray,
    previous_summary: Optional[Mapping[str, Any]] = None,
    swim_row_diagnostics_complete: bool,
    otsu_thresholds: Optional[np.ndarray] = None,
    valid_ray_fraction_values: Optional[np.ndarray] = None,
    max_missing_gap_values: Optional[np.ndarray] = None,
) -> dict[str, Any]:
    roi_total = int(roi_count)
    mask_present_arr = np.asarray(mask_present_swim, dtype=bool)
    area_arr = np.asarray(area_px_swim, dtype=np.float32)
    prob_arr = np.asarray(prob_max_swim, dtype=np.float32)
    status_arr = np.asarray(swim_row_status, dtype=np.int8)
    nonempty_count = int(np.count_nonzero(mask_present_arr))
    summary_statistics = {
        "rows_total": roi_total,
        "rows_with_nonempty_masks": nonempty_count,
        "rows_empty_masks": int(roi_total - nonempty_count),
        "rows_skipped_missing_keypoint": int(np.count_nonzero(status_arr == SWIM_ROW_STATUS_MISSING_KEYPOINT)),
        "rows_skipped_unsuccessful_keypoint": int(np.count_nonzero(status_arr == SWIM_ROW_STATUS_UNSUCCESSFUL_KEYPOINT)),
        "area_px_min": float(area_arr.min()) if roi_total else None,
        "area_px_mean": float(area_arr.mean()) if roi_total else None,
        "area_px_max": float(area_arr.max()) if roi_total else None,
        "prob_max_min": float(prob_arr.min()) if roi_total else None,
        "prob_max_mean": float(prob_arr.mean()) if roi_total else None,
        "prob_max_max": float(prob_arr.max()) if roi_total else None,
        "output_run": run_name,
        "crop_run": str(crop_run),
        "keypoint_group": str(kp_group_name),
        "keypoint_run": str(kp_run_name),
        "duration_seconds": duration_seconds,
        "created_at_utc": created_at_utc,
    }
    if updated_at_utc:
        summary_statistics["updated_at_utc"] = str(updated_at_utc)

    if swim_row_diagnostics_complete and otsu_thresholds is not None and valid_ray_fraction_values is not None and max_missing_gap_values is not None:
        finite_otsu = np.isfinite(otsu_thresholds)
        finite_valid_fraction = np.isfinite(valid_ray_fraction_values)
        finite_max_gap = np.isfinite(max_missing_gap_values)
        summary_statistics["otsu_threshold_min"] = float(np.nanmin(otsu_thresholds)) if np.any(finite_otsu) else None
        summary_statistics["otsu_threshold_mean"] = float(np.nanmean(otsu_thresholds)) if np.any(finite_otsu) else None
        summary_statistics["otsu_threshold_max"] = float(np.nanmax(otsu_thresholds)) if np.any(finite_otsu) else None
        summary_statistics["valid_ray_fraction_mean"] = (
            float(np.nanmean(valid_ray_fraction_values)) if np.any(finite_valid_fraction) else None
        )
        summary_statistics["max_missing_gap_degrees_mean"] = (
            float(np.nanmean(max_missing_gap_values)) if np.any(finite_max_gap) else None
        )
    else:
        previous = dict(previous_summary) if isinstance(previous_summary, Mapping) else {}
        for key in (
            "otsu_threshold_min",
            "otsu_threshold_mean",
            "otsu_threshold_max",
            "valid_ray_fraction_mean",
            "max_missing_gap_degrees_mean",
        ):
            summary_statistics[key] = previous.get(key)
    return summary_statistics


def _process_swim_bladder_roi(
    row_idx: int,
    roi_image: np.ndarray,
    keypoints_row: np.ndarray,
    *,
    success_flag: bool,
    keypoint_labels: Optional[Sequence[str]],
    cfg: SwimBladderSegmentationConfig,
) -> Dict[str, Any]:
    if not success_flag:
        return {"index": int(row_idx), "status": "unsuccessful_keypoint"}

    gray_roi = _coerce_roi_to_gray(np.asarray(roi_image))
    center_xy = _resolve_swim_bladder_point(keypoints_row, keypoint_labels)
    if center_xy is None:
        return {"index": int(row_idx), "status": "missing_keypoint"}

    if cfg.subject_method_family == swim_tuning.POLAR_BOUNDARY_METHOD_FAMILY:
        params = {
            "roi_padding": int(cfg.roi_padding),
            "angle_step_degrees": int(cfg.angle_step_degrees),
            "min_radius_px": int(cfg.min_radius_px),
            "max_radius_px": int(cfg.max_radius_px),
            "smoothing_sigma": float(cfg.smoothing_sigma),
            "response_threshold": float(cfg.response_threshold),
            "max_missing_gap_degrees": int(cfg.max_missing_gap_degrees),
            "min_valid_ray_fraction": float(cfg.min_valid_ray_fraction),
            "gradient_mode": str(cfg.gradient_mode),
            "prefilter_sigma": float(cfg.prefilter_sigma),
        }
    else:
        params = {
            "roi_padding": int(cfg.roi_padding),
            "pre_threshold": cfg.pre_threshold,
            "sobel_strength": float(cfg.sobel_strength),
            "min_area": int(cfg.min_area),
            "max_area": cfg.max_area,
            "min_circularity": cfg.min_circularity,
            "closing_radius": int(cfg.closing_radius),
            "opening_radius": int(cfg.opening_radius),
        }

    x0, x1, y0, y1 = _extract_patch_bounds(tuple(gray_roi.shape), center_xy, int(cfg.roi_padding))
    patch = np.asarray(gray_roi[y0:y1, x0:x1], dtype=np.uint8)
    patch_center_xy = (float(center_xy[0]) - float(x0), float(center_xy[1]) - float(y0))
    preview = swim_tuning._compute_swim_bladder_patch_preview(
        patch,
        center_xy=patch_center_xy,
        params=params,
        method_family=cfg.subject_method_family,
    )
    probability_patch = np.asarray(preview.get("probability_patch"), dtype=np.float32)
    if probability_patch.shape != patch.shape:
        probability_patch = np.zeros_like(patch, dtype=np.float32)

    return {
        "index": int(row_idx),
        "status": "ok",
        "bounds": (int(x0), int(x1), int(y0), int(y1)),
        "proposal_patch": np.asarray(preview["proposal_mask"], dtype=np.uint8),
        "probability_patch": np.clip(probability_patch, 0.0, 1.0),
        "threshold_value": preview["stats"].get("threshold_value"),
        "valid_ray_fraction": preview["stats"].get("valid_ray_fraction"),
        "max_missing_gap_degrees": preview["stats"].get("max_missing_gap_degrees"),
    }


def _process_swim_bladder_chunk(
    chunk_indices: Sequence[int],
    zarr_path: str,
    cfg_dict: Dict[str, Any],
) -> list[Dict[str, Any]]:
    if not chunk_indices:
        return []

    root = open_zarr_root(zarr_path, mode="r")
    cfg = SwimBladderSegmentationConfig(**cfg_dict)
    crop_source = CropImageSource.open(
        root,
        crop_run=cfg.crop_run,
        zarr_path=zarr_path,
        roi_cache_policy=cfg.roi_cache_policy,
        roi_live_acceleration=cfg.roi_live_acceleration,
        roi_live_gpu_chunk_frames=int(cfg.roi_live_gpu_chunk_frames),
        roi_cache_dir=cfg.roi_cache_dir,
        roi_cache_manifest=cfg.roi_cache_manifest,
        console=None,
    )
    try:
        kp_group, kp_group_name, kp_run_name = _resolve_keypoint_group(
            root,
            subject_group=None,
            refined_group=None,
            explicit_run=cfg.keypoint_run,
            explicit_group=None,
            expected_crop_run=cfg.crop_run,
        )
        if kp_group is None or kp_group_name is None or kp_run_name is None:
            fallback = subject_tuning._resolve_eye_keypoint_source(root, cfg.keypoint_run)
            kp_group = fallback.group
            kp_group_name = fallback.group_name
            kp_run_name = fallback.run_name
        _validate_keypoint_group_alignment(
            crop_source.crop_group,
            str(cfg.crop_run),
            kp_group,
            total_rois=int(crop_source.shape[0]),
        )
        keypoint_labels_raw = kp_group.attrs.get("keypoint_labels")
        keypoint_labels = (
            [str(item) for item in keypoint_labels_raw]
            if isinstance(keypoint_labels_raw, (list, tuple))
            else None
        )
        start = int(chunk_indices[0])
        stop = int(chunk_indices[-1]) + 1
        keypoints_chunk = np.asarray(kp_group["keypoints_roi"][start:stop], dtype=np.float32)
        success_all = _load_keypoint_success_flags(kp_group, total_rois=int(crop_source.shape[0]))
        success_chunk = np.asarray(success_all[start:stop], dtype=bool)
        results: list[Dict[str, Any]] = []
        for row_idx in chunk_indices:
            local_idx = int(row_idx) - start
            results.append(
                _process_swim_bladder_roi(
                    int(row_idx),
                    np.asarray(crop_source[int(row_idx)]),
                    np.asarray(keypoints_chunk[local_idx], dtype=np.float32),
                    success_flag=bool(success_chunk[local_idx]),
                    keypoint_labels=keypoint_labels,
                    cfg=cfg,
                )
            )
        return results
    finally:
        crop_source.close()


def segment_swim_bladder_masks_from_root(
    root: zarr.Group,
    *,
    zarr_path: str | Path | None = None,
    config_dict: Optional[Mapping[str, Any]] = None,
    console: Optional[Console] = None,
    output_run: Optional[str] = None,
    overwrite: bool = False,
    scheduler: str = "single-threaded",
    num_workers: Optional[int] = None,
    roi_indices: Optional[Sequence[int]] = None,
) -> str:
    stage_start = time.perf_counter()
    cfg = SwimBladderSegmentationConfig()
    cfg, tuning_entry = _apply_tuned_parameters(root, cfg, console)
    tuning_override_keys = _collect_tuning_override_keys(config_dict)
    cfg = _apply_overrides(cfg, config_dict)
    effective_method = _resolve_effective_method_name(cfg.subject_method_family)

    crop_source = CropImageSource.open(
        root,
        crop_run=cfg.crop_run,
        zarr_path=zarr_path,
        roi_cache_policy=cfg.roi_cache_policy,
        roi_live_acceleration=cfg.roi_live_acceleration,
        roi_live_gpu_chunk_frames=int(cfg.roi_live_gpu_chunk_frames),
        roi_cache_dir=cfg.roi_cache_dir,
        roi_cache_manifest=cfg.roi_cache_manifest,
        console=console,
    )
    crop_group = crop_source.crop_group
    crop_run = crop_source.crop_run_name
    roi_images = crop_source

    try:
        kp_group, kp_group_name, kp_run_name = _resolve_keypoint_group(
            root,
            subject_group=None,
            refined_group=None,
            explicit_run=cfg.keypoint_run,
            explicit_group=None,
            expected_crop_run=str(crop_run),
        )
        if kp_group is None or kp_group_name is None or kp_run_name is None:
            fallback = subject_tuning._resolve_eye_keypoint_source(root, cfg.keypoint_run)
            kp_group = fallback.group
            kp_group_name = fallback.group_name
            kp_run_name = fallback.run_name
        _validate_keypoint_group_alignment(
            crop_group,
            str(crop_run),
            kp_group,
            total_rois=int(roi_images.shape[0]),
        )
        keypoints_roi = np.asarray(kp_group["keypoints_roi"][:], dtype=np.float32)
        success_flags = _load_keypoint_success_flags(kp_group, total_rois=int(roi_images.shape[0]))
        keypoint_labels_raw = kp_group.attrs.get("keypoint_labels")
        keypoint_labels = (
            [str(item) for item in keypoint_labels_raw]
            if isinstance(keypoint_labels_raw, (list, tuple))
            else None
        )
        cfg.crop_run = str(crop_run)
        cfg.keypoint_run = str(kp_run_name)

        roi_count = int(roi_images.shape[0])
        roi_h = int(roi_images.shape[1])
        roi_w = int(roi_images.shape[2])
        n_channels = len(SUBJECT_MASK_LABELS)
        selected_rows = _normalize_roi_indices(roi_indices, roi_count)
        partial_refresh = roi_indices is not None

        selected_swim_masks = np.zeros((len(selected_rows), roi_h, roi_w), dtype=np.uint8)
        selected_swim_probs = np.zeros((len(selected_rows), roi_h, roi_w), dtype=np.float32)
        selected_status = np.full((len(selected_rows),), SWIM_ROW_STATUS_OK, dtype=np.int8)
        selected_otsu_thresholds = np.full((len(selected_rows),), np.nan, dtype=np.float32)
        selected_valid_ray_fraction_values = np.full((len(selected_rows),), np.nan, dtype=np.float32)
        selected_max_missing_gap_values = np.full((len(selected_rows),), np.nan, dtype=np.float32)
        scheduler_key = _normalize_scheduler(scheduler)
        requested_scheduler = str(scheduler or "").strip().lower()
        if requested_scheduler and scheduler_key != requested_scheduler and requested_scheduler not in {"single-thread", "single_thread"}:
            _print(console, f"[yellow]Unknown scheduler '{scheduler}', defaulting to 'single-threaded'[/yellow]")

        roi_results: list[Dict[str, Any]] = []
        if total_selected := int(len(selected_rows)):
            can_parallelize = zarr_path is not None
            if scheduler_key != "single-threaded" and not can_parallelize:
                _print(console, "[yellow]Parallel scheduler requested without zarr_path; falling back to single-threaded.[/yellow]")
                scheduler_key = "single-threaded"

            if scheduler_key == "single-threaded" or total_selected == 1:
                for row_idx in selected_rows:
                    roi_results.append(
                        _process_swim_bladder_roi(
                            row_idx,
                            np.asarray(roi_images[row_idx]),
                            np.asarray(keypoints_roi[row_idx], dtype=np.float32),
                            success_flag=bool(success_flags[row_idx]) if row_idx < success_flags.shape[0] else False,
                            keypoint_labels=keypoint_labels,
                            cfg=cfg,
                        )
                    )
            else:
                default_workers = os.cpu_count() or 4
                worker_count = int(num_workers) if num_workers is not None else min(default_workers, 16)
                indices = list(selected_rows)
                chunk_size = max(64, min(1024, (total_selected + worker_count - 1) // worker_count))
                chunks = [indices[i:i + chunk_size] for i in range(0, total_selected, chunk_size)]
                cfg_dict = asdict(cfg)
                tasks = [
                    delayed(_process_swim_bladder_chunk)(
                        chunk,
                        str(zarr_path),
                        cfg_dict,
                    )
                    for chunk in chunks
                ]

                client = None
                cluster = None
                try:
                    if scheduler_key == "distributed":
                        try:
                            from dask.distributed import Client, LocalCluster
                        except ImportError:
                            _print(console, "[yellow]dask[distributed] not installed; falling back to 'threads'.[/yellow]")
                            scheduler_key = "threads"
                        else:
                            dashboard_addr = os.environ.get("PALETTE_DASK_DASHBOARD", ":0")
                            cluster_kwargs: Dict[str, Any] = {
                                "threads_per_worker": 1,
                                "processes": True,
                                "memory_limit": "auto",
                                "dashboard_address": dashboard_addr,
                            }
                            if num_workers is not None:
                                cluster_kwargs["n_workers"] = int(num_workers)
                            cluster = LocalCluster(**cluster_kwargs)
                            client = Client(cluster)
                            dash_link = getattr(client, "dashboard_link", None)
                            if dash_link:
                                _print(console, f"[cyan]Dask dashboard:[/cyan] [link={dash_link}]{dash_link}[/link]")
                            futures = client.compute(tasks, sync=False)
                            gathered = client.gather(futures)
                            roi_results = [item for chunk_result in gathered for item in chunk_result]

                    if not roi_results:
                        compute_kwargs: Dict[str, Any] = {"scheduler": scheduler_key}
                        if num_workers is not None:
                            compute_kwargs["num_workers"] = int(num_workers)
                        with ProgressBar():
                            computed = dask.compute(*tasks, **compute_kwargs)
                        for chunk_result in computed:
                            roi_results.extend(chunk_result)
                finally:
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

        roi_results.sort(key=lambda result: int(result["index"]))
        row_to_offset = {int(row_idx): idx for idx, row_idx in enumerate(selected_rows)}
        for result in roi_results:
            row_idx = int(result["index"])
            row_offset = int(row_to_offset[row_idx])
            status = str(result["status"])
            if status == "unsuccessful_keypoint":
                selected_status[row_offset] = SWIM_ROW_STATUS_UNSUCCESSFUL_KEYPOINT
                continue
            if status == "missing_keypoint":
                selected_status[row_offset] = SWIM_ROW_STATUS_MISSING_KEYPOINT
                continue

            selected_status[row_offset] = SWIM_ROW_STATUS_OK
            x0, x1, y0, y1 = result["bounds"]
            selected_swim_masks[row_offset, y0:y1, x0:x1] = np.asarray(result["proposal_patch"], dtype=np.uint8)
            selected_swim_probs[row_offset, y0:y1, x0:x1] = np.asarray(result["probability_patch"], dtype=np.float32)
            threshold_value = result.get("threshold_value")
            if threshold_value is not None:
                selected_otsu_thresholds[row_offset] = float(threshold_value)
            valid_fraction = result.get("valid_ray_fraction")
            if valid_fraction is not None:
                selected_valid_ray_fraction_values[row_offset] = float(valid_fraction)
            max_gap = result.get("max_missing_gap_degrees")
            if max_gap is not None:
                selected_max_missing_gap_values[row_offset] = float(max_gap)

        channel_metrics = _compute_channel_metrics(selected_swim_masks, selected_swim_probs)
        metric_row_chunk = subject_mask_metric_row_chunk(roi_count)
        op_timestamp = datetime.now(timezone.utc).isoformat()
        tuning_entry_snapshot = _snapshot_tuning_entry(tuning_entry)

        if partial_refresh:
            if output_run is None:
                raise ValueError("Partial swim-bladder refresh requires --run-name / output_run.")
            subject_parent = root.get("subject_mask_runs")
            if subject_parent is None or not hasattr(subject_parent, "attrs") or str(output_run) not in subject_parent:
                raise ValueError(
                    f"subject_mask_runs/{output_run} not found. Partial swim-bladder refresh requires an existing run."
                )
            run_group = subject_parent[str(output_run)]
            run_name = str(output_run)
            created_at = str(run_group.attrs.get("created_at_utc") or op_timestamp)
            updated_at = op_timestamp

            masks_arr = run_group.get("masks_roi")
            probs_arr = run_group.get("mask_probs_roi")
            if masks_arr is None or probs_arr is None:
                raise RuntimeError(f"subject_mask_runs/{run_name} is missing masks_roi or mask_probs_roi.")
            if tuple(int(dim) for dim in masks_arr.shape) != (roi_count, n_channels, roi_h, roi_w):
                raise RuntimeError(
                    f"subject_mask_runs/{run_name} masks_roi shape mismatch: expected {(roi_count, n_channels, roi_h, roi_w)}, "
                    f"got {tuple(int(dim) for dim in masks_arr.shape)}."
                )
            if tuple(int(dim) for dim in probs_arr.shape) != (roi_count, n_channels, roi_h, roi_w):
                raise RuntimeError(
                    f"subject_mask_runs/{run_name} mask_probs_roi shape mismatch: expected {(roi_count, n_channels, roi_h, roi_w)}, "
                    f"got {tuple(int(dim) for dim in probs_arr.shape)}."
                )

            metrics_group = run_group.require_group("metrics")
            prob_max_arr = metrics_group["prob_max"]
            mask_present_arr = metrics_group["mask_present"]
            area_px_arr = metrics_group["area_px"]
            centroid_xy_arr = metrics_group["centroid_xy"]
            centroid_valid_arr = metrics_group["centroid_valid"]
            bbox_xyxy_arr = metrics_group["bbox_xyxy"]
            bbox_valid_arr = metrics_group["bbox_valid"]

            swim_row_status_arr = metrics_group.get("swim_row_status")
            if swim_row_status_arr is None or int(swim_row_status_arr.shape[0]) != roi_count:
                metrics_group.create_array(
                    "swim_row_status",
                    data=_bootstrap_swim_row_status(
                        keypoints_roi=keypoints_roi,
                        success_flags=success_flags,
                        keypoint_labels=keypoint_labels,
                    ),
                    chunks=(metric_row_chunk,),
                    overwrite=True,
                )
                swim_row_status_arr = metrics_group["swim_row_status"]
            run_group.attrs["swim_row_status_labels"] = {
                "0": "ok",
                "1": "missing_keypoint",
                "2": "unsuccessful_keypoint",
            }

            diagnostics_complete = bool(run_group.attrs.get("swim_row_diagnostics_complete", False))
            for metric_name in (
                "swim_otsu_threshold",
                "swim_valid_ray_fraction",
                "swim_max_missing_gap_degrees",
            ):
                metric_arr = metrics_group.get(metric_name)
                if metric_arr is None or int(metric_arr.shape[0]) != roi_count:
                    metrics_group.create_array(
                        metric_name,
                        data=np.full((roi_count,), np.nan, dtype=np.float32),
                        chunks=(metric_row_chunk,),
                        overwrite=True,
                    )
                    diagnostics_complete = False
            otsu_threshold_arr = metrics_group["swim_otsu_threshold"]
            valid_ray_fraction_arr = metrics_group["swim_valid_ray_fraction"]
            max_missing_gap_arr = metrics_group["swim_max_missing_gap_degrees"]

            for row_offset, roi_idx in enumerate(selected_rows):
                masks_arr[int(roi_idx), 2] = np.asarray(selected_swim_masks[row_offset], dtype=np.uint8)
                probs_arr[int(roi_idx), 2] = np.asarray(selected_swim_probs[row_offset], dtype=np.float16)
                prob_max_arr[int(roi_idx), 2] = np.float32(channel_metrics["prob_max"][row_offset])
                mask_present_arr[int(roi_idx), 2] = bool(channel_metrics["mask_present"][row_offset])
                area_px_arr[int(roi_idx), 2] = np.float32(channel_metrics["area_px"][row_offset])
                centroid_xy_arr[int(roi_idx), 2] = np.asarray(channel_metrics["centroid_xy"][row_offset], dtype=np.float32)
                centroid_valid_arr[int(roi_idx), 2] = bool(channel_metrics["centroid_valid"][row_offset])
                bbox_xyxy_arr[int(roi_idx), 2] = np.asarray(channel_metrics["bbox_xyxy"][row_offset], dtype=np.float32)
                bbox_valid_arr[int(roi_idx), 2] = bool(channel_metrics["bbox_valid"][row_offset])
                swim_row_status_arr[int(roi_idx)] = np.int8(selected_status[row_offset])
                otsu_threshold_arr[int(roi_idx)] = np.float32(selected_otsu_thresholds[row_offset])
                valid_ray_fraction_arr[int(roi_idx)] = np.float32(selected_valid_ray_fraction_values[row_offset])
                max_missing_gap_arr[int(roi_idx)] = np.float32(selected_max_missing_gap_values[row_offset])

            run_group.attrs["swim_row_diagnostics_complete"] = bool(diagnostics_complete)
            prob_max_swim = np.asarray(prob_max_arr[:, 2], dtype=np.float32)
            mask_present_swim = np.asarray(mask_present_arr[:, 2], dtype=bool)
            area_px_swim = np.asarray(area_px_arr[:, 2], dtype=np.float32)
            swim_row_status = np.asarray(swim_row_status_arr[:], dtype=np.int8)
            previous_summary = run_group.attrs.get("summary_statistics")
            diagnostic_thresholds = np.asarray(otsu_threshold_arr[:], dtype=np.float32)
            diagnostic_valid_fraction = np.asarray(valid_ray_fraction_arr[:], dtype=np.float32)
            diagnostic_max_gap = np.asarray(max_missing_gap_arr[:], dtype=np.float32)
        else:
            run_group, run_name = _prepare_run_group(root, output_run=output_run, overwrite=overwrite)
            created_at = op_timestamp
            updated_at = None

            masks_full = np.zeros((roi_count, n_channels, roi_h, roi_w), dtype=np.uint8)
            probs_full = np.zeros((roi_count, n_channels, roi_h, roi_w), dtype=np.float16)
            for row_offset, roi_idx in enumerate(selected_rows):
                masks_full[int(roi_idx), 2] = np.asarray(selected_swim_masks[row_offset], dtype=np.uint8)
                probs_full[int(roi_idx), 2] = np.asarray(selected_swim_probs[row_offset], dtype=np.float16)

            detection_source_arr = crop_group.get("detection_source")
            detection_source = (
                np.asarray(detection_source_arr[:], dtype=np.int8)
                if detection_source_arr is not None
                else np.zeros((roi_count,), dtype=np.int8)
            )

            copy_row_lineage_arrays(run_group, crop_group, total_rois=roi_count)
            storage_chunks = subject_mask_storage_chunks(roi_count, roi_h, roi_w)
            run_group.create_array("detection_source", data=detection_source, overwrite=True)
            run_group.create_array("masks_roi", data=masks_full, chunks=storage_chunks, overwrite=True)
            run_group.create_array("mask_probs_roi", data=probs_full, chunks=storage_chunks, overwrite=True)
            run_group.create_array(
                "available_channels",
                data=np.asarray(SUBJECT_MASK_AVAILABLE_CHANNELS, dtype=bool),
                overwrite=True,
            )

            metrics_group = run_group.require_group("metrics")
            prob_max = np.zeros((roi_count, n_channels), dtype=np.float32)
            mask_present = np.zeros((roi_count, n_channels), dtype=bool)
            area_px = np.zeros((roi_count, n_channels), dtype=np.float32)
            centroid_xy = np.zeros((roi_count, n_channels, 2), dtype=np.float32)
            centroid_valid = np.zeros((roi_count, n_channels), dtype=bool)
            bbox_xyxy = np.zeros((roi_count, n_channels, 4), dtype=np.float32)
            bbox_valid = np.zeros((roi_count, n_channels), dtype=bool)
            for row_offset, roi_idx in enumerate(selected_rows):
                prob_max[int(roi_idx), 2] = np.float32(channel_metrics["prob_max"][row_offset])
                mask_present[int(roi_idx), 2] = bool(channel_metrics["mask_present"][row_offset])
                area_px[int(roi_idx), 2] = np.float32(channel_metrics["area_px"][row_offset])
                centroid_xy[int(roi_idx), 2, :] = np.asarray(channel_metrics["centroid_xy"][row_offset], dtype=np.float32)
                centroid_valid[int(roi_idx), 2] = bool(channel_metrics["centroid_valid"][row_offset])
                bbox_xyxy[int(roi_idx), 2, :] = np.asarray(channel_metrics["bbox_xyxy"][row_offset], dtype=np.float32)
                bbox_valid[int(roi_idx), 2] = bool(channel_metrics["bbox_valid"][row_offset])

            metrics_group.create_array("prob_max", data=prob_max, chunks=(metric_row_chunk, 1), overwrite=True)
            metrics_group.create_array("mask_present", data=mask_present, chunks=(metric_row_chunk, 1), overwrite=True)
            metrics_group.create_array("area_px", data=area_px, chunks=(metric_row_chunk, 1), overwrite=True)
            metrics_group.create_array(
                "centroid_xy",
                data=centroid_xy,
                chunks=(metric_row_chunk, 1, 2),
                overwrite=True,
            )
            metrics_group.create_array(
                "centroid_valid",
                data=centroid_valid,
                chunks=(metric_row_chunk, 1),
                overwrite=True,
            )
            metrics_group.create_array("bbox_xyxy", data=bbox_xyxy, chunks=(metric_row_chunk, 1, 4), overwrite=True)
            metrics_group.create_array("bbox_valid", data=bbox_valid, chunks=(metric_row_chunk, 1), overwrite=True)
            metrics_group.create_array(
                "swim_row_status",
                data=np.asarray(selected_status, dtype=np.int8),
                chunks=(metric_row_chunk,),
                overwrite=True,
            )
            metrics_group.create_array(
                "swim_otsu_threshold",
                data=np.asarray(selected_otsu_thresholds, dtype=np.float32),
                chunks=(metric_row_chunk,),
                overwrite=True,
            )
            metrics_group.create_array(
                "swim_valid_ray_fraction",
                data=np.asarray(selected_valid_ray_fraction_values, dtype=np.float32),
                chunks=(metric_row_chunk,),
                overwrite=True,
            )
            metrics_group.create_array(
                "swim_max_missing_gap_degrees",
                data=np.asarray(selected_max_missing_gap_values, dtype=np.float32),
                chunks=(metric_row_chunk,),
                overwrite=True,
            )
            run_group.attrs["swim_row_status_labels"] = {
                "0": "ok",
                "1": "missing_keypoint",
                "2": "unsuccessful_keypoint",
            }
            run_group.attrs["swim_row_diagnostics_complete"] = True

            prob_max_swim = np.asarray(prob_max[:, 2], dtype=np.float32)
            mask_present_swim = np.asarray(mask_present[:, 2], dtype=bool)
            area_px_swim = np.asarray(area_px[:, 2], dtype=np.float32)
            swim_row_status = np.asarray(selected_status, dtype=np.int8)
            previous_summary = None
            diagnostic_thresholds = np.asarray(selected_otsu_thresholds, dtype=np.float32)
            diagnostic_valid_fraction = np.asarray(selected_valid_ray_fraction_values, dtype=np.float32)
            diagnostic_max_gap = np.asarray(selected_max_missing_gap_values, dtype=np.float32)

        duration_seconds = float(time.perf_counter() - stage_start)
        crop_snapshot_attrs = build_source_crop_snapshot_attrs(
            crop_group.attrs,
            source_crop_storage_mode=crop_source.storage_mode,
        )

        run_group.attrs.update(
            {
                "method": effective_method,
                "config": asdict(cfg),
                "source_crop_run": str(crop_run),
                **crop_snapshot_attrs,
                "source_roi_read_mode": crop_source.roi_read_mode,
                "roi_cache_policy": crop_source.roi_cache_policy,
                "source_roi_cache_used": bool(crop_source.roi_cache_used),
                "source_roi_cache_backend": crop_source.roi_cache_backend,
                "source_roi_live_acceleration_requested": crop_source.roi_live_acceleration_requested,
                "source_roi_live_acceleration_effective": crop_source.roi_live_acceleration_effective,
                "source_roi_live_acceleration_fallback_reason": crop_source.roi_live_acceleration_fallback_reason,
                "source_roi_live_gpu_chunk_frames": int(crop_source.roi_live_gpu_chunk_frames),
                "source_keypoints_run": str(kp_run_name),
                "source_keypoint_run": str(kp_run_name),
                "source_keypoint_group": str(kp_group_name),
                "label_schema_id": SUBJECT_MASK_LABEL_SCHEMA,
                "mask_labels": list(SUBJECT_MASK_LABELS),
                "output_semantics": "multilabel",
                "overlap_policy": "independent_sigmoid",
                "run_semantics": "traditional_swim_bladder_inference",
                "input_format": "gray",
                "probabilities_dtype": "float16",
                "probabilities_encoding": "unit_float",
                "probability_semantics": (
                    "normalized_boundary_response"
                    if cfg.subject_method_family == swim_tuning.POLAR_BOUNDARY_METHOD_FAMILY
                    else "normalized_patch_darkness"
                ),
                "tuning_source": (
                    "analysis_metadata.subject_mask_tuning.components.swim_bladder"
                    if tuning_entry is not None
                    else "defaults_or_overrides"
                ),
                "dask_scheduler": scheduler_key,
                "dask_num_workers": int(num_workers) if num_workers is not None else None,
                "tuning_timestamp": tuning_entry.get("tuned_timestamp") if isinstance(tuning_entry, dict) else None,
                "tuning_override_keys": list(tuning_override_keys),
                "created_at_utc": created_at,
                "updated_at_utc": updated_at,
                "duration_seconds": duration_seconds,
            }
        )
        if crop_source.roi_cache_key is not None:
            run_group.attrs["source_roi_cache_key"] = crop_source.roi_cache_key
        if crop_source.roi_cache_path is not None:
            run_group.attrs["source_roi_cache_path"] = crop_source.roi_cache_path
        if tuning_entry_snapshot is not None:
            run_group.attrs["tuning_entry_snapshot"] = tuning_entry_snapshot

        write_subject_mask_component_provenance(
            run_group,
            component_name="swim_bladder",
            source_stage="subject_mask_runs",
            source_run=run_name,
            source_method=str(run_group.attrs["method"]),
            source_channels=["swim_bladder"],
            source_label_schema_id=SUBJECT_MASK_LABEL_SCHEMA,
            source_created_at_utc=created_at,
        )

        summary_statistics = _compute_swim_summary_statistics(
            roi_count=roi_count,
            run_name=run_name,
            crop_run=str(crop_run),
            kp_group_name=str(kp_group_name),
            kp_run_name=str(kp_run_name),
            duration_seconds=duration_seconds,
            created_at_utc=created_at,
            updated_at_utc=updated_at,
            mask_present_swim=mask_present_swim,
            area_px_swim=area_px_swim,
            prob_max_swim=prob_max_swim,
            swim_row_status=swim_row_status,
            previous_summary=previous_summary if isinstance(previous_summary, Mapping) else None,
            swim_row_diagnostics_complete=bool(run_group.attrs.get("swim_row_diagnostics_complete", False)),
            otsu_thresholds=diagnostic_thresholds,
            valid_ray_fraction_values=diagnostic_valid_fraction,
            max_missing_gap_values=diagnostic_max_gap,
        )
        run_group.attrs["summary_statistics"] = summary_statistics

        git_info = get_git_info()
        env_info = get_environment_info(
            include_all_packages=False,
            disk_path=str(zarr_path) if zarr_path is not None else None,
            collect_ip=False,
            capture_env_vars=False,
        )
        platform_info = env_info.get("platform", {})
        provenance_inputs = {
            "source_crop_run": str(crop_run),
            **crop_snapshot_attrs,
            "source_roi_read_mode": crop_source.roi_read_mode,
            "roi_cache_policy": crop_source.roi_cache_policy,
            "roi_cache_used": bool(crop_source.roi_cache_used),
            "roi_cache_key": crop_source.roi_cache_key,
            "roi_live_acceleration_requested": crop_source.roi_live_acceleration_requested,
            "roi_live_acceleration_effective": crop_source.roi_live_acceleration_effective,
            "roi_live_acceleration_fallback_reason": crop_source.roi_live_acceleration_fallback_reason,
            "roi_live_gpu_chunk_frames": int(crop_source.roi_live_gpu_chunk_frames),
            "source_keypoint_run": str(kp_run_name),
            "source_keypoint_group": str(kp_group_name),
            "frame_source": crop_source.frame_source_kind,
            "source_video_path": crop_source.frame_source_path or crop_group.attrs.get("video_source_path"),
        }
        if crop_source.roi_cache_path is not None:
            provenance_inputs["roi_cache_path"] = crop_source.roi_cache_path
        scheduler_info: Optional[Dict[str, Any]] = None
        if scheduler_key:
            scheduler_info = {"type": scheduler_key}
            if num_workers is not None:
                scheduler_info["num_workers"] = int(num_workers)
        provenance = build_stage_provenance(
            stage="subject_masks",
            command=" ".join(sys.argv),
            created_at_utc=updated_at or created_at,
            version=git_info.get("short_hash") or git_info.get("commit_hash"),
            git={
                "commit": git_info.get("commit_hash"),
                "short": git_info.get("short_hash"),
                "branch": git_info.get("branch"),
                "is_dirty": git_info.get("is_dirty"),
                "remote": git_info.get("remote_url"),
            },
            environment=env_info.get("environment"),
            platform={
                "hostname": platform_info.get("hostname"),
                "system": platform_info.get("system"),
                "release": platform_info.get("release"),
                "python_version": platform_info.get("python_version"),
                "machine": platform_info.get("machine"),
            },
            scheduler=scheduler_info,
            parameters={
                **asdict(cfg),
                "method": run_group.attrs.get("method"),
                "run_semantics": run_group.attrs.get("run_semantics"),
                "probability_semantics": run_group.attrs.get("probability_semantics"),
                "refresh_mode": "partial" if partial_refresh else "full",
                "roi_indices": list(selected_rows) if partial_refresh else None,
                "tuning_source": run_group.attrs.get("tuning_source"),
                "dask_scheduler": run_group.attrs.get("dask_scheduler"),
                "dask_num_workers": run_group.attrs.get("dask_num_workers"),
                "tuning_timestamp": run_group.attrs.get("tuning_timestamp"),
                "tuning_override_keys": list(tuning_override_keys),
                "tuning_entry_snapshot": tuning_entry_snapshot,
            },
            inputs=provenance_inputs,
        )
        write_stage_provenance(run_group, provenance)
        if zarr_path is not None:
            emit_subject_mask_stage_completion(
                root,
                zarr_path,
                run_group=run_group,
                run_name=run_name,
                source=_SUBJECT_MASKS_STATUS_SOURCE,
                console=console,
                invalidate_on_ok=True,
            )

        _print(
            console,
            (
                f"[green]✓[/green] Swim-bladder masks {'updated' if partial_refresh else 'saved'} as "
                f"[cyan]subject_mask_runs/{run_name}[/cyan] "
                f"({int(np.count_nonzero(mask_present_swim))}/{roi_count} nonempty swim-bladder masks"
                f"{'; touched ' + str(len(selected_rows)) + ' rows' if partial_refresh else ''}) in {duration_seconds:.1f}s"
            ),
        )
        return run_name
    finally:
        crop_source.close()


def segment_swim_bladder_masks(
    zarr_path: str | Path,
    *,
    config_dict: Optional[Mapping[str, Any]] = None,
    console: Optional[Console] = None,
    output_run: Optional[str] = None,
    overwrite: bool = False,
    scheduler: str = "single-threaded",
    num_workers: Optional[int] = None,
    roi_indices: Optional[Sequence[int]] = None,
) -> str:
    root = open_zarr_root(zarr_path, mode="a")
    return segment_swim_bladder_masks_from_root(
        root,
        zarr_path=zarr_path,
        config_dict=config_dict,
        console=console,
        output_run=output_run,
        overwrite=overwrite,
        scheduler=scheduler,
        num_workers=num_workers,
        roi_indices=roi_indices,
    )


def _build_cli_overrides(args: argparse.Namespace) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {}
    for key in (
        "crop_run",
        "keypoint_run",
        "roi_cache_policy",
        "roi_cache_dir",
        "roi_cache_manifest",
        "roi_live_acceleration",
        "roi_live_gpu_chunk_frames",
        *TUNING_OVERRIDE_KEYS,
    ):
        value = getattr(args, key)
        if value is not None:
            overrides[key] = value
    return overrides


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run traditional swim-bladder segmentation on ROI crops stored in a Palette Zarr archive.",
    )
    parser.add_argument("zarr_path", type=Path, help="Path to the Palette Zarr archive.")
    parser.add_argument("--crop-run", type=str, help="Optional crop_runs/<run> to segment.")
    parser.add_argument("--keypoint-run", type=str, help="Optional keypoint run providing swim-bladder anchors.")
    parser.add_argument(
        "--scheduler",
        type=_parse_scheduler_arg,
        default="single-threaded",
        help=(
            "Scheduler to use for ROI processing. Accepted values: threads, processes, distributed, "
            "single-threaded, single-thread, single_thread. Default: single-threaded."
        ),
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Optional number of worker threads/processes for scheduler-backed ROI processing.",
    )
    parser.add_argument(
        "--roi-cache-policy",
        choices=("never", "auto", "always"),
        default="auto",
        help="Temporary ROI cache policy for geometry-only crop runs (default: auto).",
    )
    parser.add_argument("--roi-cache-dir", type=str, help="Optional scratch directory for temporary ROI caches.")
    parser.add_argument(
        "--roi-cache-manifest",
        type=str,
        help="Optional flat_bin_v1 ROI cache manifest to read instead of materializing/re-decoding ROIs.",
    )
    parser.add_argument(
        "--roi-live-acceleration",
        choices=("auto", "cpu", "gpu"),
        default="auto",
        help="Live ROI read acceleration for geometry-only crop runs (default: auto).",
    )
    parser.add_argument(
        "--roi-live-gpu-chunk-frames",
        type=int,
        default=32,
        help="Frame batch size for GPU-accelerated live ROI reads (default: 32).",
    )
    parser.add_argument(
        "--subject-method-family",
        type=str,
        help="Optional swim-bladder method family override (e.g. threshold_blob or polar_boundary).",
    )
    parser.add_argument("--run-name", type=str, help="Optional subject_mask_runs/<run> output name.")
    parser.add_argument(
        "--roi-indices",
        type=_parse_roi_indices_arg,
        help=(
            "Optional comma-separated ROI row indices to refresh in-place inside an existing --run-name. "
            "When omitted, the segmenter writes a full run."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing subject_mask_runs/<run> when --run-name already exists.",
    )
    parser.add_argument("--roi-padding", type=int, help="Override swim-bladder patch padding.")
    parser.add_argument("--pre-threshold", type=int, help="Override patch pre-threshold.")
    parser.add_argument("--sobel-strength", type=float, help="Override Sobel strength.")
    parser.add_argument("--min-area", type=int, help="Override minimum connected-component area.")
    parser.add_argument("--max-area", type=int, help="Override maximum connected-component area.")
    parser.add_argument("--min-circularity", type=float, help="Override minimum circularity.")
    parser.add_argument("--closing-radius", type=int, help="Override morphological closing radius.")
    parser.add_argument("--opening-radius", type=int, help="Override morphological opening radius.")
    parser.add_argument("--angle-step-degrees", type=int, help="Override polar boundary angle step.")
    parser.add_argument("--min-radius-px", type=int, help="Override polar boundary minimum radius.")
    parser.add_argument("--max-radius-px", type=int, help="Override polar boundary maximum radius.")
    parser.add_argument("--smoothing-sigma", type=float, help="Override polar boundary smoothing sigma.")
    parser.add_argument("--response-threshold", type=float, help="Override polar boundary response threshold.")
    parser.add_argument(
        "--gradient-mode",
        type=str,
        help="Override polar boundary gradient response (e.g. sobel_magnitude, scharr_magnitude, laplacian_abs).",
    )
    parser.add_argument(
        "--max-missing-gap-degrees",
        type=int,
        help="Override maximum allowed missing-angle gap before rejecting a polar proposal.",
    )
    parser.add_argument(
        "--min-valid-ray-fraction",
        type=float,
        help="Override minimum valid-ray fraction required for a polar proposal.",
    )
    parser.add_argument("--prefilter-sigma", type=float, help="Override polar boundary prefilter sigma.")

    args = parser.parse_args(argv)
    console = Console() if Console is not None else None
    overrides = _build_cli_overrides(args)
    segment_swim_bladder_masks(
        args.zarr_path,
        config_dict=overrides,
        console=console,
        output_run=args.run_name,
        overwrite=bool(args.overwrite),
        scheduler=args.scheduler,
        num_workers=args.num_workers,
        roi_indices=list(args.roi_indices) if args.roi_indices is not None else None,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
