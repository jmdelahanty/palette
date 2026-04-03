#!/usr/bin/env python3
"""Traditional swim-bladder segmentation on materialized ROI crops."""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import zarr

from ..shared.crop_image_source import resolve_materialized_crop_run
from ..shared.stage_provenance import build_stage_provenance, write_stage_provenance
from ..shared.subject_mask_chunks import subject_mask_metric_row_chunk, subject_mask_storage_chunks
from ..shared.subject_mask_component_provenance import write_subject_mask_component_provenance
from ..tune import subject_mask_tuner as subject_tuning
from ..tune import swim_bladder_mask_tuner as swim_tuning
from ..utils.system import get_environment_info, get_git_info
from ..utils.zarr_io import open_zarr_root
from ..visualization.visualize_swim_bladder_mask_patches import (
    _extract_patch_bounds,
    _resolve_swim_bladder_center_with_source,
)
from .subject_segmentation import (
    SUBJECT_MASK_LABELS,
    SUBJECT_MASK_LABEL_SCHEMA,
    _coerce_roi_to_gray,
    _compute_channel_metrics,
    _copy_lineage_array,
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
    roi_shape: tuple[int, int],
) -> Optional[tuple[float, float]]:
    center_xy, center_source = _resolve_swim_bladder_center_with_source(
        keypoints_row,
        keypoint_labels,
        np.zeros(roi_shape, dtype=np.uint8),
        roi_shape,
    )
    return tuple(center_xy) if center_source == "keypoint" else None


def segment_swim_bladder_masks_from_root(
    root: zarr.Group,
    *,
    zarr_path: str | Path | None = None,
    config_dict: Optional[Mapping[str, Any]] = None,
    console: Optional[Console] = None,
    output_run: Optional[str] = None,
    overwrite: bool = False,
) -> str:
    stage_start = time.perf_counter()
    cfg = SwimBladderSegmentationConfig()
    cfg, tuning_entry = _apply_tuned_parameters(root, cfg, console)
    tuning_override_keys = _collect_tuning_override_keys(config_dict)
    cfg = _apply_overrides(cfg, config_dict)
    effective_method = _resolve_effective_method_name(cfg.subject_method_family)

    _, crop_group, crop_run = resolve_materialized_crop_run(root, crop_run=cfg.crop_run, zarr_path=zarr_path)
    roi_images = crop_group.get("roi_images")
    if roi_images is None:
        raise ValueError(f"crop_runs/{crop_run} must provide roi_images for traditional swim-bladder segmentation.")

    keypoint_source = subject_tuning._resolve_eye_keypoint_source(root, cfg.keypoint_run)
    keypoints_roi = np.asarray(keypoint_source.keypoints_roi[:], dtype=np.float32)
    success_flags = np.asarray(keypoint_source.success_flags, dtype=bool)
    if int(keypoints_roi.shape[0]) != int(roi_images.shape[0]):
        raise ValueError(
            f"Keypoint rows {int(keypoints_roi.shape[0])} do not match crop ROI rows {int(roi_images.shape[0])}."
        )
    keypoint_labels_raw = keypoint_source.group.attrs.get("keypoint_labels")
    keypoint_labels = (
        [str(item) for item in keypoint_labels_raw]
        if isinstance(keypoint_labels_raw, (list, tuple))
        else None
    )

    roi_count = int(roi_images.shape[0])
    roi_h = int(roi_images.shape[1])
    roi_w = int(roi_images.shape[2])
    n_channels = len(SUBJECT_MASK_LABELS)

    swim_masks = np.zeros((roi_count, roi_h, roi_w), dtype=np.uint8)
    swim_probs = np.zeros((roi_count, roi_h, roi_w), dtype=np.float32)
    rows_skipped_missing_keypoint = 0
    rows_skipped_unsuccessful_keypoint = 0
    otsu_thresholds = np.full((roi_count,), np.nan, dtype=np.float32)
    valid_ray_fraction_values = np.full((roi_count,), np.nan, dtype=np.float32)
    max_missing_gap_values = np.full((roi_count,), np.nan, dtype=np.float32)

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

    for row_idx in range(roi_count):
        if row_idx < success_flags.shape[0] and not bool(success_flags[row_idx]):
            rows_skipped_unsuccessful_keypoint += 1
            continue

        roi_image = _coerce_roi_to_gray(np.asarray(roi_images[row_idx]))
        keypoints_row = np.asarray(keypoints_roi[row_idx], dtype=np.float32)
        center_xy = _resolve_swim_bladder_point(keypoints_row, keypoint_labels, tuple(roi_image.shape))
        if center_xy is None:
            rows_skipped_missing_keypoint += 1
            continue

        x0, x1, y0, y1 = _extract_patch_bounds(tuple(roi_image.shape), center_xy, int(cfg.roi_padding))
        patch = np.asarray(roi_image[y0:y1, x0:x1], dtype=np.uint8)
        patch_center_xy = (float(center_xy[0]) - float(x0), float(center_xy[1]) - float(y0))
        preview = swim_tuning._compute_swim_bladder_patch_preview(
            patch,
            center_xy=patch_center_xy,
            params=params,
            method_family=cfg.subject_method_family,
        )

        proposal_patch = np.asarray(preview["proposal_mask"], dtype=np.uint8)
        swim_masks[row_idx, y0:y1, x0:x1] = proposal_patch
        probability_patch = np.asarray(preview.get("probability_patch"), dtype=np.float32)
        if probability_patch.shape != patch.shape:
            probability_patch = np.zeros_like(patch, dtype=np.float32)
        swim_probs[row_idx, y0:y1, x0:x1] = np.clip(probability_patch, 0.0, 1.0)
        threshold_value = preview["stats"].get("threshold_value")
        if threshold_value is not None:
            otsu_thresholds[row_idx] = float(threshold_value)
        valid_fraction = preview["stats"].get("valid_ray_fraction")
        if valid_fraction is not None:
            valid_ray_fraction_values[row_idx] = float(valid_fraction)
        max_gap = preview["stats"].get("max_missing_gap_degrees")
        if max_gap is not None:
            max_missing_gap_values[row_idx] = float(max_gap)

    run_group, run_name = _prepare_run_group(root, output_run=output_run, overwrite=overwrite)
    created_at = datetime.now(timezone.utc).isoformat()

    masks_full = np.zeros((roi_count, n_channels, roi_h, roi_w), dtype=np.uint8)
    probs_full = np.zeros((roi_count, n_channels, roi_h, roi_w), dtype=np.float16)
    masks_full[:, 2] = swim_masks
    probs_full[:, 2] = swim_probs.astype(np.float16, copy=False)

    detection_source_arr = crop_group.get("detection_source")
    detection_source = (
        np.asarray(detection_source_arr[:], dtype=np.int8)
        if detection_source_arr is not None
        else np.zeros((roi_count,), dtype=np.int8)
    )

    run_group.attrs.update(
        {
            "method": effective_method,
            "config": asdict(cfg),
            "source_crop_run": str(crop_run),
            "source_keypoints_run": str(keypoint_source.run_name),
            "source_keypoint_run": str(keypoint_source.run_name),
            "source_keypoint_group": str(keypoint_source.group_name),
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
            "tuning_timestamp": tuning_entry.get("tuned_timestamp") if isinstance(tuning_entry, dict) else None,
            "tuning_override_keys": list(tuning_override_keys),
            "created_at_utc": created_at,
        }
    )
    tuning_entry_snapshot = _snapshot_tuning_entry(tuning_entry)
    if tuning_entry_snapshot is not None:
        run_group.attrs["tuning_entry_snapshot"] = tuning_entry_snapshot

    _copy_lineage_array(run_group, crop_group, "frame_indices")
    _copy_lineage_array(run_group, crop_group, "frame_counts")
    _copy_lineage_array(run_group, crop_group, "detection_indices")
    storage_chunks = subject_mask_storage_chunks(roi_count, roi_h, roi_w)
    metric_row_chunk = subject_mask_metric_row_chunk(roi_count)
    run_group.create_array("detection_source", data=detection_source, overwrite=True)
    run_group.create_array("masks_roi", data=masks_full, chunks=storage_chunks, overwrite=True)
    run_group.create_array("mask_probs_roi", data=probs_full, chunks=storage_chunks, overwrite=True)
    run_group.create_array(
        "available_channels",
        data=np.asarray(SUBJECT_MASK_AVAILABLE_CHANNELS, dtype=bool),
        overwrite=True,
    )
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

    channel_metrics = _compute_channel_metrics(swim_masks, swim_probs)
    metrics_group = run_group.require_group("metrics")
    prob_max = np.zeros((roi_count, n_channels), dtype=np.float32)
    mask_present = np.zeros((roi_count, n_channels), dtype=bool)
    area_px = np.zeros((roi_count, n_channels), dtype=np.float32)
    centroid_xy = np.zeros((roi_count, n_channels, 2), dtype=np.float32)
    centroid_valid = np.zeros((roi_count, n_channels), dtype=bool)
    bbox_xyxy = np.zeros((roi_count, n_channels, 4), dtype=np.float32)
    bbox_valid = np.zeros((roi_count, n_channels), dtype=bool)

    prob_max[:, 2] = channel_metrics["prob_max"]
    mask_present[:, 2] = channel_metrics["mask_present"]
    area_px[:, 2] = channel_metrics["area_px"]
    centroid_xy[:, 2, :] = channel_metrics["centroid_xy"]
    centroid_valid[:, 2] = channel_metrics["centroid_valid"]
    bbox_xyxy[:, 2, :] = channel_metrics["bbox_xyxy"]
    bbox_valid[:, 2] = channel_metrics["bbox_valid"]

    metrics_group.create_array("prob_max", data=prob_max, chunks=(metric_row_chunk, 1), overwrite=True)
    metrics_group.create_array("mask_present", data=mask_present, chunks=(metric_row_chunk, 1), overwrite=True)
    metrics_group.create_array("area_px", data=area_px, chunks=(metric_row_chunk, 1), overwrite=True)
    metrics_group.create_array("centroid_xy", data=centroid_xy, chunks=(metric_row_chunk, 1, 2), overwrite=True)
    metrics_group.create_array("centroid_valid", data=centroid_valid, chunks=(metric_row_chunk, 1), overwrite=True)
    metrics_group.create_array("bbox_xyxy", data=bbox_xyxy, chunks=(metric_row_chunk, 1, 4), overwrite=True)
    metrics_group.create_array("bbox_valid", data=bbox_valid, chunks=(metric_row_chunk, 1), overwrite=True)

    duration_seconds = float(time.perf_counter() - stage_start)
    nonempty_rows = np.any(swim_masks > 0, axis=(1, 2))
    summary_statistics = {
        "rows_total": int(roi_count),
        "rows_with_nonempty_masks": int(np.sum(nonempty_rows)),
        "rows_empty_masks": int(roi_count - np.sum(nonempty_rows)),
        "rows_skipped_missing_keypoint": int(rows_skipped_missing_keypoint),
        "rows_skipped_unsuccessful_keypoint": int(rows_skipped_unsuccessful_keypoint),
        "area_px_min": float(area_px[:, 2].min()) if roi_count else None,
        "area_px_mean": float(area_px[:, 2].mean()) if roi_count else None,
        "area_px_max": float(area_px[:, 2].max()) if roi_count else None,
        "prob_max_min": float(prob_max[:, 2].min()) if roi_count else None,
        "prob_max_mean": float(prob_max[:, 2].mean()) if roi_count else None,
        "prob_max_max": float(prob_max[:, 2].max()) if roi_count else None,
        "otsu_threshold_min": float(np.nanmin(otsu_thresholds)) if np.any(np.isfinite(otsu_thresholds)) else None,
        "otsu_threshold_mean": float(np.nanmean(otsu_thresholds)) if np.any(np.isfinite(otsu_thresholds)) else None,
        "otsu_threshold_max": float(np.nanmax(otsu_thresholds)) if np.any(np.isfinite(otsu_thresholds)) else None,
        "valid_ray_fraction_mean": (
            float(np.nanmean(valid_ray_fraction_values)) if np.any(np.isfinite(valid_ray_fraction_values)) else None
        ),
        "max_missing_gap_degrees_mean": (
            float(np.nanmean(max_missing_gap_values)) if np.any(np.isfinite(max_missing_gap_values)) else None
        ),
        "output_run": run_name,
        "crop_run": str(crop_run),
        "keypoint_group": str(keypoint_source.group_name),
        "keypoint_run": str(keypoint_source.run_name),
        "duration_seconds": duration_seconds,
        "created_at_utc": created_at,
    }
    run_group.attrs["duration_seconds"] = duration_seconds
    run_group.attrs["summary_statistics"] = summary_statistics

    git_info = get_git_info()
    env_info = get_environment_info(
        include_all_packages=False,
        disk_path=str(zarr_path) if zarr_path is not None else None,
        collect_ip=False,
        capture_env_vars=False,
    )
    platform_info = env_info.get("platform", {})
    provenance = build_stage_provenance(
        stage="subject_masks",
        command=" ".join(sys.argv),
        created_at_utc=created_at,
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
        parameters={
            **asdict(cfg),
            "method": run_group.attrs.get("method"),
            "run_semantics": run_group.attrs.get("run_semantics"),
            "probability_semantics": run_group.attrs.get("probability_semantics"),
            "tuning_source": run_group.attrs.get("tuning_source"),
            "tuning_timestamp": run_group.attrs.get("tuning_timestamp"),
            "tuning_override_keys": list(tuning_override_keys),
            "tuning_entry_snapshot": tuning_entry_snapshot,
        },
        inputs={
            "source_crop_run": str(crop_run),
            "source_keypoint_run": str(keypoint_source.run_name),
            "source_keypoint_group": str(keypoint_source.group_name),
        },
    )
    write_stage_provenance(run_group, provenance)

    _print(
        console,
        f"[green]✓[/green] Swim-bladder masks saved as [cyan]subject_mask_runs/{run_name}[/cyan] "
        f"({int(np.sum(nonempty_rows))}/{roi_count} nonempty swim-bladder masks) in {duration_seconds:.1f}s",
    )
    return run_name


def segment_swim_bladder_masks(
    zarr_path: str | Path,
    *,
    config_dict: Optional[Mapping[str, Any]] = None,
    console: Optional[Console] = None,
    output_run: Optional[str] = None,
    overwrite: bool = False,
) -> str:
    root = open_zarr_root(zarr_path, mode="a")
    return segment_swim_bladder_masks_from_root(
        root,
        zarr_path=zarr_path,
        config_dict=config_dict,
        console=console,
        output_run=output_run,
        overwrite=overwrite,
    )


def _build_cli_overrides(args: argparse.Namespace) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {}
    for key in ("crop_run", "keypoint_run", *TUNING_OVERRIDE_KEYS):
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
        "--subject-method-family",
        type=str,
        help="Optional swim-bladder method family override (e.g. threshold_blob or polar_boundary).",
    )
    parser.add_argument("--run-name", type=str, help="Optional subject_mask_runs/<run> output name.")
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
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
