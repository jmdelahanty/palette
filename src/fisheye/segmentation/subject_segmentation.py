#!/usr/bin/env python3
"""Traditional subject-body segmentation on Palette crop runs."""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import cv2
import numpy as np
import zarr

from ..shared.roi_background import (
    _extract_background_roi_ds,
    _extract_background_roi_full,
    _resolve_background_source,
    _resolve_run_name,
)
from ..shared.crop_image_source import CropImageSource
from ..shared.provenance_attrs import build_source_crop_snapshot_attrs, build_source_roi_pixel_attrs
from ..shared.row_lineage import copy_row_lineage_arrays, write_direct_source_crop_row_ids
from ..shared.subject_mask_registry_status import emit_subject_mask_stage_completion
from ..shared.stage_provenance import build_stage_provenance, write_stage_provenance
from ..shared.subject_mask_chunks import subject_mask_metric_row_chunk, subject_mask_storage_chunks
from ..shared.subject_mask_component_provenance import write_subject_mask_component_provenance
from ..shared.zarr_run_completion import (
    mark_run_complete,
    mark_run_started,
    note_pending_latest,
    require_runs_parent,
)
from ..tune import subject_mask_tuner as tuning
from ..shared.system_metadata import get_environment_info, get_git_info
from ..shared.zarr_io import open_zarr_root

try:
    from rich.console import Console
except Exception:  # pragma: no cover - rich is optional at runtime
    Console = None  # type: ignore


SUBJECT_MASK_LABEL_SCHEMA = "subject_v1_union"
SUBJECT_MASK_LABELS: tuple[str, ...] = ("subject_body", "eyes_union", "swim_bladder")
SUBJECT_MASK_AVAILABLE_CHANNELS = (True, False, False)
_SUBJECT_MASKS_STATUS_SOURCE = "runtime_subject_segmentation"


@dataclass
class SubjectSegmentationConfig:
    diff_threshold: int = tuning.DEFAULT_THRESHOLD
    gaussian_blur_kernel: int = tuning.DEFAULT_BLUR_KERNEL
    closing_radius: int = tuning.DEFAULT_CLOSING_RADIUS
    opening_radius: int = tuning.DEFAULT_OPENING_RADIUS
    min_area: int = tuning.DEFAULT_MIN_AREA
    keep_largest_component: bool = bool(tuning.DEFAULT_KEEP_LARGEST_COMPONENT)
    crop_run: Optional[str] = None
    background_run: Optional[str] = None
    roi_cache_policy: str = "auto"
    roi_cache_dir: Optional[str] = None
    roi_cache_manifest: Optional[str] = None
    roi_live_acceleration: str = "auto"
    roi_live_gpu_chunk_frames: int = 32


def _print(console: Optional[Console], message: str) -> None:
    if console is not None:
        console.print(message)
    else:  # pragma: no cover - exercised only when rich is unavailable
        print(message)


def _normalize_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, np.integer)):
        return bool(int(value))
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _apply_tuned_parameters(
    root: zarr.Group,
    cfg: SubjectSegmentationConfig,
    console: Optional[Console] = None,
) -> tuple[SubjectSegmentationConfig, Optional[dict[str, Any]]]:
    entry = tuning._load_subject_tuning_entry_from_root(root, "subject_body")
    tuned = entry.get("tuned_parameters") if isinstance(entry, dict) else None
    if not isinstance(tuned, Mapping):
        return cfg, None

    normalized = tuning._normalize_subject_component_params("subject_body", tuned)
    cfg.diff_threshold = int(normalized["diff_threshold"])
    cfg.gaussian_blur_kernel = int(normalized["gaussian_blur_kernel"])
    cfg.closing_radius = int(normalized["closing_radius"])
    cfg.opening_radius = int(normalized["opening_radius"])
    cfg.min_area = int(normalized["min_area"])
    cfg.keep_largest_component = bool(normalized["keep_largest_component"])

    if console is not None:
        timestamp = entry.get("tuned_timestamp") if isinstance(entry, dict) else None
        suffix = f" (saved {timestamp})" if timestamp else ""
        console.print(f"[cyan]Using subject-body tuning from analysis_metadata{suffix}[/cyan]")
    return cfg, dict(entry) if isinstance(entry, dict) else None


def _freeze_attr_payload(value: Any) -> Any:
    """Convert nested payloads into plain Python attr-safe values."""
    if isinstance(value, Mapping):
        return {str(key): _freeze_attr_payload(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_freeze_attr_payload(item) for item in value]
    if isinstance(value, np.ndarray):
        return _freeze_attr_payload(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


def _snapshot_tuning_entry(tuning_entry: Optional[Mapping[str, Any]]) -> Optional[dict[str, Any]]:
    if not isinstance(tuning_entry, Mapping):
        return None
    frozen = _freeze_attr_payload(dict(tuning_entry))
    return frozen if isinstance(frozen, dict) else None


def _apply_overrides(
    cfg: SubjectSegmentationConfig,
    overrides: Optional[Mapping[str, Any]],
) -> SubjectSegmentationConfig:
    if not overrides:
        return cfg
    param_keys = {
        "diff_threshold",
        "gaussian_blur_kernel",
        "closing_radius",
        "opening_radius",
        "min_area",
        "keep_largest_component",
    }
    param_overrides = {key: overrides[key] for key in param_keys if key in overrides}
    if param_overrides:
        normalized = tuning._normalize_subject_component_params("subject_body", param_overrides)
        if "diff_threshold" in param_overrides:
            cfg.diff_threshold = int(normalized["diff_threshold"])
        if "gaussian_blur_kernel" in param_overrides:
            cfg.gaussian_blur_kernel = int(normalized["gaussian_blur_kernel"])
        if "closing_radius" in param_overrides:
            cfg.closing_radius = int(normalized["closing_radius"])
        if "opening_radius" in param_overrides:
            cfg.opening_radius = int(normalized["opening_radius"])
        if "min_area" in param_overrides:
            cfg.min_area = int(normalized["min_area"])
        if "keep_largest_component" in param_overrides:
            cfg.keep_largest_component = bool(normalized["keep_largest_component"])
    if "crop_run" in overrides:
        cfg.crop_run = str(overrides["crop_run"]) if overrides["crop_run"] is not None else None
    if "background_run" in overrides:
        cfg.background_run = (
            str(overrides["background_run"]) if overrides["background_run"] is not None else None
        )
    return cfg


def _coerce_roi_to_gray(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 2:
        return arr.astype(np.uint8, copy=False)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        return arr[..., 0].astype(np.uint8, copy=False)
    if arr.ndim == 3 and arr.shape[-1] == 3:
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8, copy=False)
        return cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    raise ValueError(f"Unsupported roi image shape {arr.shape}; expected (H,W), (H,W,1), or (H,W,3).")


def _normalize_background_frame(background: np.ndarray) -> np.ndarray:
    arr = np.asarray(background)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3 and arr.shape[0] == 1:
        return arr[0]
    if arr.ndim == 3 and arr.shape[-1] == 1:
        return arr[..., 0]
    raise ValueError(f"Unsupported background array shape {arr.shape}; expected 2D background image.")


def _create_dish_mask_image(mask_params: Mapping[str, Any], img_shape: tuple[int, int]) -> Optional[np.ndarray]:
    if not mask_params:
        return None
    dish_shape = mask_params.get("shape", "circle")
    mask = None
    if dish_shape == "rectangle" and "roi" in mask_params:
        x, y, w, h = [int(v) for v in mask_params["roi"]]
        mask = np.zeros(img_shape, dtype=np.uint8)
        cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
    elif dish_shape == "circle":
        center = mask_params.get("center")
        radius = mask_params.get("radius")
        if center and radius:
            mask = np.zeros(img_shape, dtype=np.uint8)
            cv2.circle(mask, tuple(int(v) for v in center), int(radius), 255, -1)
    return mask


def _resolve_full_frame_shape(root: zarr.Group) -> Optional[tuple[int, int]]:
    raw_video = root.get("raw_video")
    images_full = raw_video.get("images_full") if raw_video is not None else None
    if images_full is not None and len(images_full.shape) >= 3:
        return int(images_full.shape[1]), int(images_full.shape[2])
    height = root.attrs.get("height") or root.attrs.get("video_height")
    width = root.attrs.get("width") or root.attrs.get("video_width")
    if height is not None and width is not None:
        return int(height), int(width)
    return None


def _resolve_dish_mask_projection(
    root: zarr.Group,
) -> tuple[Optional[np.ndarray], Optional[str], Optional[tuple[int, int]]]:
    analysis = root.get("analysis_metadata")
    if analysis is None:
        return None, None, None
    raw_mask = analysis.attrs.get("dish_mask")
    if not isinstance(raw_mask, Mapping):
        return None, None, None
    mask_data = dict(raw_mask)
    shape = mask_data.get("shape")
    if not shape:
        if "detected_circle" in mask_data:
            shape = "circle"
        elif "rectangle" in mask_data:
            shape = "rectangle"
    tuned_on_array = (
        mask_data.get("tuned_on_array")
        or mask_data.get("source", {}).get("array")
        or "images_full"
    )
    metrics = mask_data.get("metrics", {}) if isinstance(mask_data.get("metrics"), Mapping) else {}
    image_shape_raw = metrics.get("image_shape")
    image_shape = None
    if isinstance(image_shape_raw, (list, tuple)) and len(image_shape_raw) == 2:
        image_shape = (int(image_shape_raw[0]), int(image_shape_raw[1]))

    if shape == "rectangle" and isinstance(mask_data.get("rectangle"), Mapping):
        params = {
            "shape": "rectangle",
            "roi": [int(v) for v in mask_data["rectangle"]["roi"]],
        }
    elif shape == "circle" and isinstance(mask_data.get("detected_circle"), Mapping):
        circle = mask_data["detected_circle"]
        params = {
            "shape": "circle",
            "center": [int(v) for v in circle["center"]],
            "radius": int(circle["radius"]),
        }
    else:
        return None, None, None

    if image_shape is None:
        if tuned_on_array == "images_ds":
            raw_video = root.get("raw_video")
            images_ds = raw_video.get("images_ds") if raw_video is not None else None
            if images_ds is not None and len(images_ds.shape) >= 3:
                image_shape = (int(images_ds.shape[1]), int(images_ds.shape[2]))
        if image_shape is None and tuned_on_array == "images_full":
            image_shape = _resolve_full_frame_shape(root)
    if image_shape is None:
        return None, None, None
    return _create_dish_mask_image(params, image_shape), str(tuned_on_array), image_shape


def _extract_dish_mask_roi(
    dish_mask_image: np.ndarray,
    *,
    tuned_on_array: str,
    image_shape: tuple[int, int],
    top_left_xy_full: np.ndarray,
    roi_shape: tuple[int, int],
    full_shape: Optional[tuple[int, int]],
) -> np.ndarray:
    if tuned_on_array == "images_ds":
        if full_shape is None:
            raise ValueError("Cannot project a dish mask tuned on images_ds without a full-frame shape.")
        return _extract_background_roi_ds(
            dish_mask_image,
            top_left_xy_full,
            roi_shape,
            full_shape,
        )
    return _extract_background_roi_full(dish_mask_image, top_left_xy_full, roi_shape)


def _compute_channel_metrics(
    binary_masks: np.ndarray,
    probs: np.ndarray,
) -> dict[str, np.ndarray]:
    binary = np.asarray(binary_masks, dtype=np.uint8)
    prob_arr = np.asarray(probs, dtype=np.float32)
    if binary.ndim != 3 or prob_arr.shape != binary.shape:
        raise ValueError("binary_masks and probs must both have shape (N,H,W).")

    row_count = int(binary.shape[0])
    area_px = binary.sum(axis=(1, 2), dtype=np.int64).astype(np.float32)
    mask_present = area_px > 0.0
    prob_max = (
        prob_arr.max(axis=(1, 2)).astype(np.float32, copy=False)
        if row_count
        else np.zeros((0,), dtype=np.float32)
    )

    centroid_xy = np.zeros((row_count, 2), dtype=np.float32)
    centroid_valid = np.zeros((row_count,), dtype=bool)
    bbox_xyxy = np.zeros((row_count, 4), dtype=np.float32)
    bbox_valid = np.zeros((row_count,), dtype=bool)

    for row_idx in range(row_count):
        ys, xs = np.nonzero(binary[row_idx] > 0)
        if ys.size == 0:
            continue
        centroid_xy[row_idx, 0] = float(xs.mean())
        centroid_xy[row_idx, 1] = float(ys.mean())
        centroid_valid[row_idx] = True
        bbox_xyxy[row_idx] = np.asarray(
            [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())],
            dtype=np.float32,
        )
        bbox_valid[row_idx] = True

    return {
        "prob_max": prob_max,
        "mask_present": mask_present.astype(bool, copy=False),
        "area_px": area_px,
        "centroid_xy": centroid_xy,
        "centroid_valid": centroid_valid,
        "bbox_xyxy": bbox_xyxy,
        "bbox_valid": bbox_valid,
    }


def _prepare_run_group(
    root: zarr.Group,
    *,
    output_run: Optional[str],
    overwrite: bool,
) -> tuple[zarr.Group, str]:
    parent = require_runs_parent(root, "subject_mask_runs")
    run_name = output_run
    if run_name is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
        base_name = f"subject_masks_{timestamp}"
        run_name = base_name
        suffix = 1
        while run_name in parent:
            run_name = f"{base_name}_{suffix:03d}"
            suffix += 1
    if run_name in parent:
        if not overwrite:
            raise ValueError(
                f"subject_mask_runs/{run_name} already exists. Pass --overwrite to replace it."
            )
        del parent[run_name]
    run_group = parent.create_group(run_name)
    mark_run_started(run_group, run_name=str(run_name), stage="subject_masks")
    note_pending_latest(parent, str(run_name))
    return run_group, str(run_name)


def segment_subject_masks_from_root(
    root: zarr.Group,
    *,
    zarr_path: str | Path | None = None,
    config_dict: Optional[Mapping[str, Any]] = None,
    console: Optional[Console] = None,
    output_run: Optional[str] = None,
    overwrite: bool = False,
) -> str:
    stage_start = time.perf_counter()
    cfg = SubjectSegmentationConfig()
    cfg, tuning_entry = _apply_tuned_parameters(root, cfg, console)
    cfg = _apply_overrides(cfg, config_dict)

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
    roi_coordinates_full = crop_source.roi_coordinates_full

    try:
        background_run = _resolve_run_name(root, "background_runs", cfg.background_run)
        background_source = _resolve_background_source(root, background_run)
        background_group = root["background_runs"][background_run]
        background_frame = _normalize_background_frame(np.asarray(background_group[background_source.array_name][:]))
        dish_mask_image, dish_mask_array_name, dish_mask_image_shape = _resolve_dish_mask_projection(root)
        full_frame_shape = _resolve_full_frame_shape(root)

        roi_count = int(roi_images.shape[0])
        roi_h = int(roi_images.shape[1])
        roi_w = int(roi_images.shape[2])
        n_channels = len(SUBJECT_MASK_LABELS)

        body_masks = np.zeros((roi_count, roi_h, roi_w), dtype=np.uint8)
        body_probs = np.zeros((roi_count, roi_h, roi_w), dtype=np.float32)
        otsu_thresholds = np.zeros((roi_count,), dtype=np.float32)
        candidate_components = np.zeros((roi_count,), dtype=np.int32)
        kept_components = np.zeros((roi_count,), dtype=np.int32)
        largest_area = np.zeros((roi_count,), dtype=np.float32)
        kept_area = np.zeros((roi_count,), dtype=np.float32)

        params = {
            "diff_threshold": int(cfg.diff_threshold),
            "gaussian_blur_kernel": int(cfg.gaussian_blur_kernel),
            "closing_radius": int(cfg.closing_radius),
            "opening_radius": int(cfg.opening_radius),
            "min_area": int(cfg.min_area),
            "keep_largest_component": bool(cfg.keep_largest_component),
        }

        for row_idx in range(roi_count):
            roi_image = _coerce_roi_to_gray(np.asarray(roi_images[row_idx]))
            top_left_xy = np.asarray(roi_coordinates_full[row_idx], dtype=np.float32)
            roi_shape = (roi_image.shape[0], roi_image.shape[1])
            if background_source.array_name == "background_full":
                background_roi = _extract_background_roi_full(background_frame, top_left_xy, roi_shape)
            else:
                assert background_source.full_shape is not None
                background_roi = _extract_background_roi_ds(
                    background_frame,
                    top_left_xy,
                    roi_shape,
                    background_source.full_shape,
                )
            dish_mask_roi = None
            if dish_mask_image is not None and dish_mask_array_name is not None and dish_mask_image_shape is not None:
                dish_mask_roi = _extract_dish_mask_roi(
                    dish_mask_image,
                    tuned_on_array=dish_mask_array_name,
                    image_shape=dish_mask_image_shape,
                    top_left_xy_full=top_left_xy,
                    roi_shape=roi_shape,
                    full_shape=full_frame_shape,
                )

            seed = tuning._compute_subject_mask_seed(
                roi_image,
                background_roi,
                params,
                allowed_mask=dish_mask_roi,
            )
            body_masks[row_idx] = np.asarray(seed["processed_mask"], dtype=np.uint8)
            body_probs[row_idx] = (
                np.asarray(seed["diff_filtered"], dtype=np.float32) / 255.0
            )
            otsu_thresholds[row_idx] = float(seed["otsu_threshold"])
            stats = seed["stats"]
            candidate_components[row_idx] = int(stats["candidate_components"])
            kept_components[row_idx] = int(stats["kept_components"])
            largest_area[row_idx] = float(stats["largest_area"])
            kept_area[row_idx] = float(stats["kept_area"])
        run_group, run_name = _prepare_run_group(root, output_run=output_run, overwrite=overwrite)
        created_at = datetime.now(timezone.utc).isoformat()

        masks_full = np.zeros((roi_count, n_channels, roi_h, roi_w), dtype=np.uint8)
        probs_full = np.zeros((roi_count, n_channels, roi_h, roi_w), dtype=np.float16)
        masks_full[:, 0] = body_masks
        probs_full[:, 0] = body_probs.astype(np.float16, copy=False)

        detection_source_arr = crop_group.get("detection_source")
        detection_source = (
            np.asarray(detection_source_arr[:], dtype=np.int8)
            if detection_source_arr is not None
            else np.zeros((roi_count,), dtype=np.int8)
        )
        tuning_entry_snapshot = _snapshot_tuning_entry(tuning_entry)
        crop_snapshot_attrs = build_source_crop_snapshot_attrs(
            crop_group.attrs,
            source_crop_storage_mode=crop_source.storage_mode,
        )
        crop_pixel_attrs = build_source_roi_pixel_attrs(crop_source)

        run_group.attrs.update(
            {
                "method": str(
                    tuning_entry.get("method")
                    if isinstance(tuning_entry, dict) and tuning_entry.get("method")
                    else "traditional_subject_mask_seed"
                ),
                "config": asdict(cfg),
                "source_crop_run": str(crop_run),
                **crop_snapshot_attrs,
                **crop_pixel_attrs,
                "source_roi_read_mode": crop_source.roi_read_mode,
                "roi_cache_policy": crop_source.roi_cache_policy,
                "source_roi_cache_used": bool(crop_source.roi_cache_used),
                "source_roi_cache_backend": getattr(crop_source, "roi_cache_backend", None),
                "source_roi_live_acceleration_requested": crop_source.roi_live_acceleration_requested,
                "source_roi_live_acceleration_effective": crop_source.roi_live_acceleration_effective,
                "source_roi_live_acceleration_fallback_reason": crop_source.roi_live_acceleration_fallback_reason,
                "source_roi_live_gpu_chunk_frames": int(crop_source.roi_live_gpu_chunk_frames),
                "source_background_run": str(background_run),
                "source_background_array": str(background_source.array_name),
                "source_dish_mask_array": dish_mask_array_name,
                "label_schema_id": SUBJECT_MASK_LABEL_SCHEMA,
                "mask_labels": list(SUBJECT_MASK_LABELS),
                "output_semantics": "multilabel",
                "overlap_policy": "independent_sigmoid",
                "run_semantics": "traditional_subject_body_inference",
                "input_format": "gray",
                "probabilities_dtype": "float16",
                "probabilities_encoding": "unit_float",
                "probability_semantics": "normalized_background_diff",
                "tuning_source": (
                    "analysis_metadata.subject_mask_tuning.components.subject_body"
                    if tuning_entry is not None
                    else "defaults_or_overrides"
                ),
                "tuning_timestamp": tuning_entry.get("tuned_timestamp") if isinstance(tuning_entry, dict) else None,
                "created_at_utc": created_at,
            }
        )
        if crop_source.roi_cache_key is not None:
            run_group.attrs["source_roi_cache_key"] = crop_source.roi_cache_key
        if crop_source.roi_cache_path is not None:
            run_group.attrs["source_roi_cache_path"] = crop_source.roi_cache_path
        if tuning_entry_snapshot is not None:
            run_group.attrs["tuning_entry_snapshot"] = tuning_entry_snapshot

        copy_row_lineage_arrays(run_group, crop_group, total_rois=roi_count)
        write_direct_source_crop_row_ids(run_group, total_rois=roi_count)
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
            component_name="subject_body",
            source_stage="subject_mask_runs",
            source_run=run_name,
            source_method=str(run_group.attrs["method"]),
            source_channels=["subject_body"],
            source_label_schema_id=SUBJECT_MASK_LABEL_SCHEMA,
            source_created_at_utc=created_at,
        )

        channel_metrics = _compute_channel_metrics(body_masks, body_probs)
        metrics_group = run_group.require_group("metrics")
        prob_max = np.zeros((roi_count, n_channels), dtype=np.float32)
        mask_present = np.zeros((roi_count, n_channels), dtype=bool)
        area_px = np.zeros((roi_count, n_channels), dtype=np.float32)
        centroid_xy = np.zeros((roi_count, n_channels, 2), dtype=np.float32)
        centroid_valid = np.zeros((roi_count, n_channels), dtype=bool)
        bbox_xyxy = np.zeros((roi_count, n_channels, 4), dtype=np.float32)
        bbox_valid = np.zeros((roi_count, n_channels), dtype=bool)

        prob_max[:, 0] = channel_metrics["prob_max"]
        mask_present[:, 0] = channel_metrics["mask_present"]
        area_px[:, 0] = channel_metrics["area_px"]
        centroid_xy[:, 0, :] = channel_metrics["centroid_xy"]
        centroid_valid[:, 0] = channel_metrics["centroid_valid"]
        bbox_xyxy[:, 0, :] = channel_metrics["bbox_xyxy"]
        bbox_valid[:, 0] = channel_metrics["bbox_valid"]

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

        duration_seconds = float(time.perf_counter() - stage_start)
        nonempty_rows = np.any(body_masks > 0, axis=(1, 2))
        nonempty_count = int(np.sum(nonempty_rows))
        summary_statistics = {
            "rows_total": int(roi_count),
            "rows_with_nonempty_masks": nonempty_count,
            "rows_empty_masks": int(roi_count - nonempty_count),
            "area_px_min": float(area_px[:, 0].min()) if roi_count else None,
            "area_px_mean": float(area_px[:, 0].mean()) if roi_count else None,
            "area_px_max": float(area_px[:, 0].max()) if roi_count else None,
            "prob_max_min": float(prob_max[:, 0].min()) if roi_count else None,
            "prob_max_mean": float(prob_max[:, 0].mean()) if roi_count else None,
            "prob_max_max": float(prob_max[:, 0].max()) if roi_count else None,
            "otsu_threshold_min": float(otsu_thresholds.min()) if roi_count else None,
            "otsu_threshold_mean": float(otsu_thresholds.mean()) if roi_count else None,
            "otsu_threshold_max": float(otsu_thresholds.max()) if roi_count else None,
            "candidate_components_mean": float(candidate_components.mean()) if roi_count else None,
            "kept_components_mean": float(kept_components.mean()) if roi_count else None,
            "largest_area_mean": float(largest_area.mean()) if roi_count else None,
            "kept_area_mean": float(kept_area.mean()) if roi_count else None,
            "output_run": run_name,
            "crop_run": str(crop_run),
            "background_run": str(background_run),
            "background_array": str(background_source.array_name),
            "dish_mask_array": dish_mask_array_name,
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
        provenance_inputs = {
            "source_crop_run": str(crop_run),
            **crop_snapshot_attrs,
            **crop_pixel_attrs,
            "source_roi_read_mode": crop_source.roi_read_mode,
            "roi_cache_policy": crop_source.roi_cache_policy,
            "roi_cache_used": bool(crop_source.roi_cache_used),
            "roi_cache_key": crop_source.roi_cache_key,
            "roi_live_acceleration_requested": crop_source.roi_live_acceleration_requested,
            "roi_live_acceleration_effective": crop_source.roi_live_acceleration_effective,
            "roi_live_acceleration_fallback_reason": crop_source.roi_live_acceleration_fallback_reason,
            "roi_live_gpu_chunk_frames": int(crop_source.roi_live_gpu_chunk_frames),
            "source_background_run": str(background_run),
            "source_background_array": str(background_source.array_name),
            "source_dish_mask_array": dish_mask_array_name,
            "frame_source": crop_source.frame_source_kind,
            "source_video_path": crop_source.frame_source_path or crop_group.attrs.get("video_source_path"),
        }
        if crop_source.roi_cache_path is not None:
            provenance_inputs["roi_cache_path"] = crop_source.roi_cache_path
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
                "probability_semantics": "normalized_background_diff",
                "tuning_source": run_group.attrs.get("tuning_source"),
                "tuning_timestamp": run_group.attrs.get("tuning_timestamp"),
                "tuning_entry_snapshot": tuning_entry_snapshot,
            },
            inputs=provenance_inputs,
        )
        write_stage_provenance(run_group, provenance)
        mark_run_complete(run_group, parent_group=root["subject_mask_runs"], run_name=run_name)
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
            f"[green]✓[/green] Subject masks saved as [cyan]subject_mask_runs/{run_name}[/cyan] "
            f"({nonempty_count}/{roi_count} nonempty body masks) in {duration_seconds:.1f}s",
        )
        return run_name
    finally:
        crop_source.close()


def segment_subject_masks(
    zarr_path: str | Path,
    *,
    config_dict: Optional[Mapping[str, Any]] = None,
    console: Optional[Console] = None,
    output_run: Optional[str] = None,
    overwrite: bool = False,
) -> str:
    root = open_zarr_root(zarr_path, mode="a")
    return segment_subject_masks_from_root(
        root,
        zarr_path=zarr_path,
        config_dict=config_dict,
        console=console,
        output_run=output_run,
        overwrite=overwrite,
    )


def _build_cli_overrides(args: argparse.Namespace) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {}
    for key in (
        "crop_run",
        "background_run",
        "roi_cache_policy",
        "roi_cache_dir",
        "roi_cache_manifest",
        "roi_live_acceleration",
        "roi_live_gpu_chunk_frames",
        "diff_threshold",
        "gaussian_blur_kernel",
        "closing_radius",
        "opening_radius",
        "min_area",
    ):
        value = getattr(args, key)
        if value is not None:
            overrides[key] = value
    if args.keep_largest_component is not None:
        overrides["keep_largest_component"] = bool(args.keep_largest_component)
    return overrides


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run traditional subject-body segmentation on a Palette crop run.",
    )
    parser.add_argument("zarr_path", type=Path, help="Path to the Palette Zarr archive.")
    parser.add_argument("--crop-run", type=str, help="Optional crop_runs/<run> to segment.")
    parser.add_argument("--background-run", type=str, help="Optional background_runs/<run> to use.")
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
    parser.add_argument("--run-name", type=str, help="Optional subject_mask_runs/<run> output name.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing subject_mask_runs/<run> when --run-name already exists.",
    )
    parser.add_argument("--diff-threshold", type=int, help="Override diff threshold.")
    parser.add_argument("--gaussian-blur-kernel", type=int, help="Override blur kernel (odd values preferred).")
    parser.add_argument("--closing-radius", type=int, help="Override morphological closing radius.")
    parser.add_argument("--opening-radius", type=int, help="Override morphological opening radius.")
    parser.add_argument("--min-area", type=int, help="Override minimum component area.")
    keep_group = parser.add_mutually_exclusive_group()
    keep_group.add_argument(
        "--keep-largest-component",
        dest="keep_largest_component",
        action="store_true",
        help="Keep only the largest surviving component after filtering.",
    )
    keep_group.add_argument(
        "--keep-all-components",
        dest="keep_largest_component",
        action="store_false",
        help="Keep all surviving components after filtering.",
    )
    parser.set_defaults(keep_largest_component=None)

    args = parser.parse_args(argv)
    console = Console() if Console is not None else None
    overrides = _build_cli_overrides(args)
    segment_subject_masks(
        args.zarr_path,
        config_dict=overrides,
        console=console,
        output_run=args.run_name,
        overwrite=bool(args.overwrite),
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
