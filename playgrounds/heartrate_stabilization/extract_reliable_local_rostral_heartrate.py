from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, replace
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from fisheye.analysis.dynamic_heart_support import (
    DynamicHeartSupportResult,
    analyze_dynamic_heart_support,
    reconstruct_crossfit_heart_phase,
)
from fisheye.analysis.local_rostral_heartrate import (
    HeartrateConfig,
    InjectionSpec,
    LocalCoordinateDataset,
    analyze_heartrate,
    bilinear_sample,
    build_risk_surfaces,
    injection_operating_characteristics,
    run_injection_recovery,
)
from fisheye.analysis.regional_phase_delay import analyze_regional_phase_delay

from _common import (
    cfg_path,
    cfg_value,
    compute_body_transform,
    crop_row_frame_id,
    ensure_output_dir,
    get_video_info,
    keypoints_to_crop_pixels,
    load_config,
    load_keypoint_data,
    read_crop_meta,
    row_float,
    selected_crop_rows,
    transform_points,
)
from map_pixel_band_contributions import _load_mask_component, _split_components
from measure_live_mask_stability import _dilate, _load_fixed_masks
from measure_local_roi_signal_compare import (
    _invert_affine,
    _project_component_crop,
    _warp_crop_mask_to_stable,
)
from render_local_rostral_alignment_comparison import (
    _component_edge_midpoint,
    _reference_component_mask,
    _rigid_from_anchor_pair,
    _swim_anchor,
)
from render_roi_mask_overlay_video import _read_status_csv
from render_dynamic_heart_phase import write_dynamic_phase_outputs
from render_regional_phase_delay import write_regional_phase_delay_outputs


NUISANCE_NAMES = (
    "global_mean",
    "body_control_mean",
    "external_control_mean",
    "body_origin_x",
    "body_origin_y",
    "body_forward_angle_deg",
    "body_scale",
    "local_rotation_deg",
    "local_translation_px",
    "source_axis_length_px",
    "target_axis_length_px",
    "sampled_pixel_count",
    "body_mask_area",
    "eye_mask_area",
    "swim_mask_area",
    "crop_x",
    "crop_y",
    "detection_x",
    "detection_y",
    "detection_confidence",
)


class _RowBlockCachedMaskStore:
    """Reuse an aligned dense Zarr row block for sequential frame reads."""

    def __init__(self, store: Any, requested_rows: int):
        self._store = store
        dense = getattr(store, "dense_array", None)
        chunks = getattr(dense, "chunks", None)
        physical_rows = int(chunks[0]) if chunks and int(chunks[0]) > 0 else 1
        requested = max(1, int(requested_rows))
        self.block_rows = int(math.ceil(requested / physical_rows) * physical_rows)
        self._key: tuple[str, int, int] | None = None
        self._block: np.ndarray | None = None
        self.hit_count = 0
        self.miss_count = 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self._store, name)

    def read_dense(self, rows: Any = None, channels: Any = None) -> np.ndarray:
        if rows is None or not np.isscalar(rows):
            return self._store.read_dense(rows=rows, channels=channels)
        row = int(rows)
        start = (row // self.block_rows) * self.block_rows
        stop = min(start + self.block_rows, int(self._store.shape[0]))
        key = (str(channels), start, stop)
        if self._key != key or self._block is None:
            self._block = np.asarray(
                self._store.read_dense(rows=slice(start, stop), channels=channels),
                dtype=np.uint8,
            )
            self._key = key
            self.miss_count += 1
        else:
            self.hit_count += 1
        local_row = row - start
        return self._block[local_row : local_row + 1]

    def cache_summary(self) -> dict[str, int]:
        return {
            "block_rows": int(self.block_rows),
            "hits": int(self.hit_count),
            "misses": int(self.miss_count),
        }


def _with_mask_row_cache(data: Any, requested_rows: int) -> Any:
    if int(requested_rows) <= 0:
        return data
    return replace(
        data,
        mask_store=_RowBlockCachedMaskStore(data.mask_store, int(requested_rows)),
    )


def _split_floats(raw: str) -> tuple[float, ...]:
    values = tuple(float(part.strip()) for part in str(raw).split(",") if part.strip())
    if not values:
        raise ValueError("expected at least one comma-separated number")
    return values


def _split_xy(raw: str, *, name: str) -> np.ndarray:
    values = _split_floats(raw)
    if len(values) != 2 or not np.isfinite(values).all():
        raise ValueError(f"{name} must contain two finite comma-separated coordinates")
    return np.asarray(values, dtype=np.float64)


def _timestamp_scale(unit: str, values: np.ndarray) -> tuple[float, str]:
    normalized = str(unit).strip().lower()
    scales = {"s": 1.0, "ms": 1e-3, "us": 1e-6, "ns": 1e-9}
    if normalized in scales:
        return scales[normalized], normalized
    if normalized != "auto":
        raise ValueError(f"unsupported timestamp unit {unit!r}")
    magnitude = float(np.nanmedian(np.abs(values)))
    if magnitude >= 1e16:
        return 1e-9, "ns"
    if magnitude >= 1e13:
        return 1e-6, "us"
    if magnitude >= 1e10:
        return 1e-3, "ms"
    return 1.0, "s"


def _timestamps_for_rows(
    selected: Sequence[tuple[int, Mapping[str, str]]],
    *,
    column: str,
    unit: str,
    nominal_fps: float,
    allow_nominal_fps: bool,
) -> tuple[np.ndarray, dict[str, Any]]:
    raw = np.asarray([row_float(row, column) for _index, row in selected], dtype=np.float64)
    if not np.isfinite(raw).all():
        if not allow_nominal_fps:
            raise ValueError(
                f"crop metadata column {column!r} is missing/nonfinite; "
                "use --allow-nominal-fps only for explicit fallback"
            )
        frame = np.asarray([index for index, _row in selected], dtype=np.float64)
        seconds = (frame - frame[0]) / float(nominal_fps)
        source = "nominal_fps_fallback"
        resolved_unit = "frame_index"
    else:
        scale, resolved_unit = _timestamp_scale(unit, raw)
        seconds = (raw - raw[0]) * scale
        source = str(column)
    diffs = np.diff(seconds)
    if diffs.size == 0 or np.any(~np.isfinite(diffs)) or np.any(diffs <= 0.0):
        raise ValueError("selected timestamps are not finite and strictly increasing")
    median_dt = float(np.median(diffs))
    effective_fps = 1.0 / median_dt
    jitter = np.abs(diffs - median_dt)
    frame_indices = np.asarray([index for index, _row in selected], dtype=np.int64)
    frame_steps = np.diff(frame_indices)
    return seconds, {
        "source": source,
        "resolved_unit": resolved_unit,
        "sample_count": int(seconds.size),
        "elapsed_seconds": float(seconds[-1] - seconds[0]),
        "median_dt_seconds": median_dt,
        "effective_fps": effective_fps,
        "dt_min_seconds": float(np.min(diffs)),
        "dt_p95_seconds": float(np.quantile(diffs, 0.95)),
        "dt_max_seconds": float(np.max(diffs)),
        "jitter_p99_seconds": float(np.quantile(jitter, 0.99)),
        "non_unit_frame_steps": int(np.count_nonzero(frame_steps != frame_steps[0]))
        if frame_steps.size
        else 0,
        "frame_step": int(frame_steps[0]) if frame_steps.size else 1,
    }


def _sample_mask_occupancy(mask: np.ndarray, points_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values, valid, _weights = bilinear_sample(np.asarray(mask, dtype=np.float64), points_xy)
    values[~valid] = np.nan
    return values, valid


def _mask_mean(gray: np.ndarray, mask: np.ndarray) -> float:
    values = np.asarray(gray, dtype=np.float64)[np.asarray(mask, dtype=bool)]
    return float(np.mean(values)) if values.size else math.nan


def _reference_anchors(
    *,
    args: argparse.Namespace,
    config: Mapping[str, Any],
    crop_rows: list[dict[str, str]],
    keypoints: Any,
    video: Any,
    mask_data: Mapping[str, Any],
    fixed: Mapping[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    stable_width = int(cfg_value(config, "alignment", "stable_width", 256))
    stable_height = int(cfg_value(config, "alignment", "stable_height", 256))
    common = {
        "crop_rows": crop_rows,
        "keypoints": keypoints,
        "video": video,
        "frame_id_column": str(cfg_value(config, "alignment", "frame_id_column", "camera_frame_id")),
        "keypoint_coordinate_array": str(
            cfg_value(config, "alignment", "keypoint_coordinate_array", "keypoints_img")
        ),
        "stable_width": stable_width,
        "stable_height": stable_height,
        "stable_center_x": float(cfg_value(config, "alignment", "stable_center_x", stable_width / 2.0)),
        "stable_center_y": float(cfg_value(config, "alignment", "stable_center_y", stable_height / 2.0)),
        "origin": str(cfg_value(config, "alignment", "origin", "eye_swim_midpoint")),
        "target_forward": str(cfg_value(config, "alignment", "target_forward", "up")),
        "scale": float(cfg_value(config, "alignment", "scale", 1.0)),
        "min_forward": float(cfg_value(config, "alignment", "min_forward_length_px", 8.0)),
        "min_eye_span": float(cfg_value(config, "alignment", "min_eye_span_px", 4.0)),
        "frame_start": int(args.reference_frame_start),
        "frame_count": int(args.reference_frame_count),
        "stride": int(args.reference_stride),
    }
    fixed_swim, swim_summary = _reference_component_mask(
        mask_data=mask_data["swim"],
        occupancy_threshold=float(args.swim_occupancy_threshold),
        **common,
    )
    fixed_eye_components: list[np.ndarray] = []
    eye_summaries: dict[str, Any] = {}
    for name, data in mask_data["eyes"].items():
        component, summary = _reference_component_mask(
            mask_data=data,
            occupancy_threshold=float(args.eye_occupancy_threshold),
            **common,
        )
        fixed_eye_components.append(component)
        eye_summaries[name] = summary
    anterior = _component_edge_midpoint(
        fixed_eye_components,
        fallback_mask=fixed["eye_mask"],
        quantile=0.98,
        side="lower",
        band_px=int(args.anchor_band_px),
    )
    posterior = _swim_anchor(
        fixed_swim,
        mode=str(args.swim_anchor_mode),
        band_px=int(args.anchor_band_px),
    )
    if anterior is None or posterior is None:
        raise ValueError("could not compute fixed local anchors")
    return anterior, posterior, {"swim": swim_summary, "eyes": eye_summaries}


def _frame_nuisance_row(
    *,
    gray: np.ndarray,
    crop_row: Mapping[str, str],
    transform: Any,
    local_details: Mapping[str, Any],
    pixel_count: int,
    body_crop: np.ndarray,
    eye_crop: np.ndarray,
    swim_crop: np.ndarray,
    local_roi_crop: np.ndarray,
) -> np.ndarray:
    body_control = np.asarray(body_crop, dtype=bool) & ~np.asarray(eye_crop, dtype=bool) & ~local_roi_crop
    external = ~np.asarray(body_crop, dtype=bool)
    values = {
        "global_mean": float(np.mean(gray)),
        "body_control_mean": _mask_mean(gray, body_control),
        "external_control_mean": _mask_mean(gray, external),
        "body_origin_x": float(transform.origin_crop_xy[0]),
        "body_origin_y": float(transform.origin_crop_xy[1]),
        "body_forward_angle_deg": float(transform.forward_angle_deg),
        "body_scale": float(transform.scale),
        "local_rotation_deg": float(local_details.get("local_rotation_deg", math.nan)),
        "local_translation_px": float(local_details.get("local_translation_px", math.nan)),
        "source_axis_length_px": float(local_details.get("source_axis_length_px", math.nan)),
        "target_axis_length_px": float(local_details.get("target_axis_length_px", math.nan)),
        "sampled_pixel_count": float(pixel_count),
        "body_mask_area": float(np.count_nonzero(body_crop)),
        "eye_mask_area": float(np.count_nonzero(eye_crop)),
        "swim_mask_area": float(np.count_nonzero(swim_crop)),
        "crop_x": row_float(crop_row, "crop_x"),
        "crop_y": row_float(crop_row, "crop_y"),
        "detection_x": row_float(crop_row, "detection_x"),
        "detection_y": row_float(crop_row, "detection_y"),
        "detection_confidence": row_float(crop_row, "detection_confidence"),
    }
    return np.asarray([values[name] for name in NUISANCE_NAMES], dtype=np.float64)


def build_dataset(
    args: argparse.Namespace,
) -> tuple[LocalCoordinateDataset, list[dict[str, Any]], dict[str, Any]]:
    import cv2
    from scipy import ndimage

    config = load_config(args.config)
    crop_video = cfg_path(config, "inputs", "crop_video")
    crop_meta_csv = cfg_path(config, "inputs", "crop_meta_csv")
    zarr_path = cfg_path(config, "inputs", "zarr_path")
    keypoint_group = str(cfg_value(config, "inputs", "keypoint_group"))
    frame_id_column = str(cfg_value(config, "alignment", "frame_id_column", "camera_frame_id"))
    frame_array = str(cfg_value(config, "alignment", "keypoint_frame_array", "frame_indices"))
    keypoint_array = str(cfg_value(config, "alignment", "keypoint_coordinate_array", "keypoints_img"))
    valid_array = str(cfg_value(config, "alignment", "keypoint_valid_array", "usable_keypoints"))
    stable_width = int(cfg_value(config, "alignment", "stable_width", 256))
    stable_height = int(cfg_value(config, "alignment", "stable_height", 256))
    stable_center_x = float(cfg_value(config, "alignment", "stable_center_x", stable_width / 2.0))
    stable_center_y = float(cfg_value(config, "alignment", "stable_center_y", stable_height / 2.0))
    origin = str(cfg_value(config, "alignment", "origin", "eye_swim_midpoint"))
    target_forward = str(cfg_value(config, "alignment", "target_forward", "up"))
    scale = float(cfg_value(config, "alignment", "scale", 1.0))
    min_forward = float(cfg_value(config, "alignment", "min_forward_length_px", 8.0))
    min_eye_span = float(cfg_value(config, "alignment", "min_eye_span_px", 4.0))
    mask_parent = str(args.mask_parent or cfg_value(config, "mask", "parent", "auto"))
    mask_run = str(args.mask_run or cfg_value(config, "mask", "run", "latest"))

    crop_rows = read_crop_meta(crop_meta_csv)
    selected = selected_crop_rows(
        crop_rows,
        frame_id_column=frame_id_column,
        frame_start=int(args.frame_start),
        frame_count=int(args.frame_count),
        stride=max(1, int(args.stride)),
    )
    if not selected:
        raise ValueError("no crop rows selected")
    timestamps, timebase = _timestamps_for_rows(
        selected,
        column=str(args.timestamp_column),
        unit=str(args.timestamp_unit),
        nominal_fps=float(args.fps),
        allow_nominal_fps=bool(args.allow_nominal_fps),
    )
    if float(args.band_max_hz) >= 0.5 * float(timebase["effective_fps"]):
        raise ValueError("requested band is not supported by the measured timestamp Nyquist limit")
    video = get_video_info(crop_video)
    keypoints = load_keypoint_data(
        zarr_path,
        keypoint_group,
        frame_array=frame_array,
        keypoint_array=keypoint_array,
        valid_array=valid_array,
    )
    fixed = _load_fixed_masks(args.mask_npz, shape_hw=(stable_height, stable_width))
    status = _read_status_csv(args.status_csv)
    body_data = _with_mask_row_cache(
        _load_mask_component(
            zarr_path,
            parent=mask_parent,
            run_name=mask_run,
            component_name=str(args.body_component),
        ),
        int(args.mask_read_cache_rows),
    )
    eye_data = {
        component: _with_mask_row_cache(
            _load_mask_component(
                zarr_path,
                parent=mask_parent,
                run_name=mask_run,
                component_name=component,
            ),
            int(args.mask_read_cache_rows),
        )
        for component in _split_components(args.eye_components)
    }
    swim_data = _with_mask_row_cache(
        _load_mask_component(
            zarr_path,
            parent=mask_parent,
            run_name=mask_run,
            component_name=str(args.swim_component),
        ),
        int(args.mask_read_cache_rows),
    )
    explicit_anchors = (
        args.reference_anterior_xy is not None
        or args.reference_posterior_xy is not None
    )
    if explicit_anchors:
        if args.reference_anterior_xy is None or args.reference_posterior_xy is None:
            raise ValueError(
                "--reference-anterior-xy and --reference-posterior-xy must be supplied together"
            )
        fixed_anterior = _split_xy(
            args.reference_anterior_xy,
            name="reference anterior",
        )
        fixed_posterior = _split_xy(
            args.reference_posterior_xy,
            name="reference posterior",
        )
        reference_summary = {
            "source": "explicit_fixed_coordinates",
            "anterior_xy": fixed_anterior.tolist(),
            "posterior_xy": fixed_posterior.tolist(),
        }
    else:
        fixed_anterior, fixed_posterior, reference_summary = _reference_anchors(
            args=args,
            config=config,
            crop_rows=crop_rows,
            keypoints=keypoints,
            video=video,
            mask_data={"body": body_data, "eyes": eye_data, "swim": swim_data},
            fixed=fixed,
        )

    roi_mask = np.asarray(fixed["roi_mask"], dtype=bool)
    pixel_y, pixel_x = np.nonzero(roi_mask)
    pixel_xy = np.column_stack([pixel_x, pixel_y]).astype(np.float64)
    pixel_count = int(pixel_xy.shape[0])
    if pixel_count < int(args.min_sample_pixels):
        raise ValueError(f"ROI has only {pixel_count} canonical pixels")
    physical_mask = np.asarray(fixed["body_mask"], dtype=bool) & ~np.asarray(
        fixed["eye_exclusion"], dtype=bool
    )
    physical_distance_image = ndimage.distance_transform_edt(physical_mask)
    administrative = np.zeros_like(roi_mask, dtype=bool)
    yy, xx = np.nonzero(roi_mask)
    administrative[int(np.min(yy)) : int(np.max(yy)) + 1, int(np.min(xx)) : int(np.max(xx)) + 1] = True
    administrative_distance_image = ndimage.distance_transform_edt(administrative)
    physical_distance = physical_distance_image[pixel_y, pixel_x]
    administrative_distance = administrative_distance_image[pixel_y, pixel_x]

    frame_count = len(selected)
    traces = np.full((frame_count, pixel_count), np.nan, dtype=np.float32)
    pixel_valid = np.zeros((frame_count, pixel_count), dtype=bool)
    frame_valid = np.zeros(frame_count, dtype=bool)
    source_xy = np.full((frame_count, pixel_count, 2), np.nan, dtype=np.float32)
    weights = np.full((frame_count, pixel_count, 4), np.nan, dtype=np.float32)
    body_occupancy = np.full((frame_count, pixel_count), np.nan, dtype=np.float32)
    eye_occupancy = np.full((frame_count, pixel_count), np.nan, dtype=np.float32)
    gradient_magnitude = np.full((frame_count, pixel_count), np.nan, dtype=np.float32)
    gradient_x_sample = np.full((frame_count, pixel_count), np.nan, dtype=np.float32)
    gradient_y_sample = np.full((frame_count, pixel_count), np.nan, dtype=np.float32)
    motion_prediction = np.full((frame_count, pixel_count), np.nan, dtype=np.float32)
    nuisance = np.full((frame_count, len(NUISANCE_NAMES)), np.nan, dtype=np.float64)
    uncertainty = np.full(frame_count, np.nan, dtype=np.float64)
    frame_indices = np.asarray([index for index, _row in selected], dtype=np.int64)
    rows_out: list[dict[str, Any]] = []

    capture = cv2.VideoCapture(str(crop_video))
    if not capture.isOpened():
        raise ValueError(f"could not open crop video {crop_video}")
    next_expected: int | None = None
    previous_gray: np.ndarray | None = None
    try:
        for output_row, (crop_video_index, crop_row) in enumerate(selected):
            if output_row and output_row % 250 == 0:
                print(
                    f"extract_progress: {output_row}/{frame_count} "
                    f"valid={int(np.count_nonzero(frame_valid[:output_row]))}",
                    flush=True,
                )
            record: dict[str, Any] = {
                "row": int(output_row),
                "crop_video_frame_index": int(crop_video_index),
                "timestamp_s": float(timestamps[output_row]),
                "valid": 0,
                "reason": "not_evaluated",
                "valid_pixel_count": 0,
                "local_rotation_deg": math.nan,
                "local_translation_px": math.nan,
            }
            if next_expected is None or int(crop_video_index) != int(next_expected):
                capture.set(cv2.CAP_PROP_POS_FRAMES, int(crop_video_index))
            read_ok, frame = capture.read()
            next_expected = int(crop_video_index) + 1
            if not read_ok:
                record["reason"] = "video_read_failed"
                rows_out.append(record)
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float64)
            if previous_gray is not None and np.array_equal(gray, previous_gray):
                record["reason"] = "duplicate_decoded_frame"
                rows_out.append(record)
                previous_gray = gray
                continue
            previous_gray = gray
            if not bool(status.get(int(crop_video_index), True)):
                record["reason"] = "status_invalid"
                rows_out.append(record)
                continue
            if str(crop_row.get("blank_frame", "0")).strip() not in {"", "0", "false", "False"}:
                record["reason"] = "blank_frame"
                rows_out.append(record)
                continue
            frame_id = crop_row_frame_id(int(crop_video_index), crop_row, frame_id_column)
            keypoint_row = keypoints.frame_to_row.get(frame_id)
            if keypoint_row is None or not bool(keypoints.valid[keypoint_row]):
                record["reason"] = "missing_or_invalid_keypoints"
                rows_out.append(record)
                continue
            keypoints_crop = keypoints_to_crop_pixels(
                keypoints.keypoints_img[keypoint_row],
                crop_row,
                video_width=video.width,
                video_height=video.height,
            )
            transform = compute_body_transform(
                keypoints_crop,
                stable_width=stable_width,
                stable_height=stable_height,
                stable_center_x=stable_center_x,
                stable_center_y=stable_center_y,
                origin=origin,
                target_forward=target_forward,
                scale=scale,
                min_forward_length_px=min_forward,
                min_eye_span_px=min_eye_span,
            )
            if not transform.valid:
                record["reason"] = transform.reason
                rows_out.append(record)
                continue
            body_crop, body_details = _project_component_crop(
                mask_data=body_data,
                frame_id=frame_id,
                crop_row=crop_row,
                video=video,
            )
            swim_crop, swim_details = _project_component_crop(
                mask_data=swim_data,
                frame_id=frame_id,
                crop_row=crop_row,
                video=video,
            )
            eye_components: list[np.ndarray] = []
            eye_crop = np.zeros(gray.shape, dtype=bool)
            for data in eye_data.values():
                component, _details = _project_component_crop(
                    mask_data=data,
                    frame_id=frame_id,
                    crop_row=crop_row,
                    video=video,
                )
                if component is not None:
                    eye_components.append(component)
                    eye_crop |= component
            if body_crop is None or swim_crop is None or not eye_components:
                record["reason"] = (
                    f"mask_unavailable:body={body_details['reason']}:"
                    f"swim={swim_details['reason']}"
                )
                rows_out.append(record)
                continue
            eye_stable_components = [
                _warp_crop_mask_to_stable(
                    component,
                    crop_to_stable=transform.crop_to_stable,
                    stable_width=stable_width,
                    stable_height=stable_height,
                )
                for component in eye_components
            ]
            eye_stable = np.logical_or.reduce(eye_stable_components)
            swim_stable = _warp_crop_mask_to_stable(
                swim_crop,
                crop_to_stable=transform.crop_to_stable,
                stable_width=stable_width,
                stable_height=stable_height,
            )
            live_anterior = _component_edge_midpoint(
                eye_stable_components,
                fallback_mask=eye_stable,
                quantile=0.98,
                side="lower",
                band_px=int(args.anchor_band_px),
            )
            live_posterior = _swim_anchor(
                swim_stable,
                mode=str(args.swim_anchor_mode),
                band_px=int(args.anchor_band_px),
            )
            if live_anterior is None or live_posterior is None:
                record["reason"] = "missing_live_anchor"
                rows_out.append(record)
                continue
            local_matrix, local_details = _rigid_from_anchor_pair(
                source_posterior=live_posterior,
                source_anterior=live_anterior,
                target_posterior=fixed_posterior,
                target_anterior=fixed_anterior,
                allow_scale=bool(args.allow_local_scale),
                min_axis_length_px=float(args.min_axis_length_px),
            )
            rotation = abs(float(local_details.get("local_rotation_deg", math.inf)))
            translation = float(local_details.get("local_translation_px", math.inf))
            record["local_rotation_deg"] = float(local_details.get("local_rotation_deg", math.nan))
            record["local_translation_px"] = translation
            if local_matrix is None:
                record["reason"] = str(local_details.get("reason", "local_transform_failed"))
                rows_out.append(record)
                continue
            if rotation > float(args.max_local_rotation_deg):
                record["reason"] = "rejected_rotation_limit"
                rows_out.append(record)
                continue
            if translation > float(args.max_local_translation_px):
                record["reason"] = "rejected_translation_limit"
                rows_out.append(record)
                continue
            fixed_to_current_stable = _invert_affine(local_matrix)
            current_stable_xy = transform_points(fixed_to_current_stable, pixel_xy)
            source_points = transform_points(transform.stable_to_crop, current_stable_xy)
            sampled, interpolation_valid, interpolation_weights = bilinear_sample(gray, source_points)
            body_values, body_valid = _sample_mask_occupancy(body_crop, source_points)
            eye_exclusion = _dilate(eye_crop, int(args.eye_dilate_px))
            eye_values, eye_valid = _sample_mask_occupancy(eye_exclusion, source_points)
            grad_y, grad_x = np.gradient(gray)
            gx, gx_valid, _gx_weights = bilinear_sample(grad_x, source_points)
            gy, gy_valid, _gy_weights = bilinear_sample(grad_y, source_points)
            valid_pixels = (
                interpolation_valid
                & body_valid
                & eye_valid
                & gx_valid
                & gy_valid
                & (body_values >= float(args.frame_body_occupancy_threshold))
                & (eye_values <= float(args.frame_eye_occupancy_threshold))
            )
            if int(np.count_nonzero(valid_pixels)) < int(args.min_sample_pixels):
                record["reason"] = "too_few_valid_local_pixels"
                rows_out.append(record)
                continue
            sampled[~valid_pixels] = np.nan
            traces[output_row] = sampled.astype(np.float32)
            pixel_valid[output_row] = valid_pixels
            source_xy[output_row] = source_points.astype(np.float32)
            weights[output_row] = interpolation_weights.astype(np.float32)
            body_occupancy[output_row] = body_values.astype(np.float32)
            eye_occupancy[output_row] = eye_values.astype(np.float32)
            gradient_x_sample[output_row] = gx.astype(np.float32)
            gradient_y_sample[output_row] = gy.astype(np.float32)
            gradient_magnitude[output_row] = np.hypot(gx, gy).astype(np.float32)
            local_roi_crop = np.zeros(gray.shape, dtype=bool)
            nearest_x = np.rint(source_points[valid_pixels, 0]).astype(np.int64)
            nearest_y = np.rint(source_points[valid_pixels, 1]).astype(np.int64)
            inside = (
                (nearest_x >= 0)
                & (nearest_x < gray.shape[1])
                & (nearest_y >= 0)
                & (nearest_y < gray.shape[0])
            )
            local_roi_crop[nearest_y[inside], nearest_x[inside]] = True
            nuisance[output_row] = _frame_nuisance_row(
                gray=gray,
                crop_row=crop_row,
                transform=transform,
                local_details=local_details,
                pixel_count=int(np.count_nonzero(valid_pixels)),
                body_crop=body_crop,
                eye_crop=eye_exclusion,
                swim_crop=swim_crop,
                local_roi_crop=local_roi_crop,
            )
            axis_difference = abs(
                float(local_details.get("source_axis_length_px", math.nan))
                - float(local_details.get("target_axis_length_px", math.nan))
            ) / max(float(local_details.get("target_axis_length_px", 1.0)), 1.0)
            uncertainty[output_row] = (
                rotation / max(float(args.max_local_rotation_deg), 1.0)
                + translation / max(float(args.max_local_translation_px), 1.0)
                + axis_difference
            )
            frame_valid[output_row] = True
            record["valid"] = 1
            record["reason"] = "ok"
            record["valid_pixel_count"] = int(np.count_nonzero(valid_pixels))
            rows_out.append(record)
    finally:
        capture.release()

    for row in range(1, frame_count):
        if not frame_valid[row] or not frame_valid[row - 1]:
            continue
        displacement = source_xy[row].astype(np.float64) - source_xy[row - 1].astype(np.float64)
        motion_prediction[row] = (
            gradient_x_sample[row].astype(np.float64) * displacement[:, 0]
            + gradient_y_sample[row].astype(np.float64) * displacement[:, 1]
        ).astype(np.float32)
    motion_prediction[0] = 0.0
    for row in range(frame_count):
        if frame_valid[row] and not np.isfinite(motion_prediction[row]).any():
            motion_prediction[row] = 0.0

    dataset = LocalCoordinateDataset(
        frame_indices=frame_indices,
        timestamps_s=timestamps,
        traces=traces,
        pixel_xy=pixel_xy,
        pixel_valid=pixel_valid,
        frame_valid=frame_valid,
        source_xy=source_xy,
        bilinear_weights=weights,
        body_occupancy=body_occupancy,
        eye_occupancy=eye_occupancy,
        gradient_magnitude=gradient_magnitude,
        motion_prediction=motion_prediction,
        nuisance_values=nuisance,
        nuisance_names=NUISANCE_NAMES,
        image_shape_hw=(stable_height, stable_width),
        administrative_boundary_distance_px=administrative_distance,
        physical_boundary_distance_px=physical_distance,
        transform_uncertainty=uncertainty,
        metadata={
            "pixel_contract": "bilinear samples from original acquisition crop-video frames",
            "source_image_shape_hw": [int(video.height), int(video.width)],
            "crop_video": str(crop_video),
            "crop_meta_csv": str(crop_meta_csv),
            "zarr_path": str(zarr_path),
            "keypoint_group": keypoint_group,
            "frame_id_column": frame_id_column,
            "mask_parent": mask_parent,
            "mask_run": mask_run,
            "roi_json": str(args.roi_json) if args.roi_json is not None else None,
            "roi_mask_npz": str(args.mask_npz),
            "timebase": timebase,
            "reference_anchors": {
                "anterior_xy": fixed_anterior.tolist(),
                "posterior_xy": fixed_posterior.tolist(),
                "sources": reference_summary,
            },
            "local_correction_limits": {
                "max_rotation_deg": float(args.max_local_rotation_deg),
                "max_translation_px": float(args.max_local_translation_px),
                "allow_scale": bool(args.allow_local_scale),
            },
            "frame_occupancy_thresholds": {
                "body_min": float(args.frame_body_occupancy_threshold),
                "eye_max": float(args.frame_eye_occupancy_threshold),
            },
            "mask_read_cache": {
                "requested_rows": int(args.mask_read_cache_rows),
                "body": body_data.mask_store.cache_summary()
                if isinstance(body_data.mask_store, _RowBlockCachedMaskStore)
                else None,
                "eyes": {
                    name: data.mask_store.cache_summary()
                    if isinstance(data.mask_store, _RowBlockCachedMaskStore)
                    else None
                    for name, data in eye_data.items()
                },
                "swim": swim_data.mask_store.cache_summary()
                if isinstance(swim_data.mask_store, _RowBlockCachedMaskStore)
                else None,
            },
        },
    ).validated()
    summary = {
        "frame_count": int(frame_count),
        "valid_frame_count": int(np.count_nonzero(frame_valid)),
        "valid_frame_fraction": float(np.mean(frame_valid)),
        "pixel_count": int(pixel_count),
        "timebase": timebase,
        "reason_counts": {
            reason: int(sum(row["reason"] == reason for row in rows_out))
            for reason in sorted({str(row["reason"]) for row in rows_out})
        },
        "metadata": dataset.metadata,
    }
    return dataset, rows_out, summary


def save_dataset(path: Path, dataset: LocalCoordinateDataset) -> None:
    ensure_output_dir(path.parent)
    np.savez_compressed(
        path,
        frame_indices=np.asarray(dataset.frame_indices, dtype=np.int64),
        timestamps_s=np.asarray(dataset.timestamps_s, dtype=np.float64),
        traces=np.asarray(dataset.traces, dtype=np.float32),
        pixel_xy=np.asarray(dataset.pixel_xy, dtype=np.float32),
        pixel_valid=np.asarray(dataset.pixel_valid, dtype=np.uint8),
        frame_valid=np.asarray(dataset.frame_valid, dtype=np.uint8),
        source_xy=np.asarray(dataset.source_xy, dtype=np.float32),
        bilinear_weights=np.asarray(dataset.bilinear_weights, dtype=np.float32),
        body_occupancy=np.asarray(dataset.body_occupancy, dtype=np.float32),
        eye_occupancy=np.asarray(dataset.eye_occupancy, dtype=np.float32),
        gradient_magnitude=np.asarray(dataset.gradient_magnitude, dtype=np.float32),
        motion_prediction=np.asarray(dataset.motion_prediction, dtype=np.float32),
        nuisance_values=np.asarray(dataset.nuisance_values, dtype=np.float64),
        nuisance_names=np.asarray(dataset.nuisance_names, dtype=np.str_),
        image_shape_hw=np.asarray(dataset.image_shape_hw, dtype=np.int32),
        administrative_boundary_distance_px=np.asarray(
            dataset.administrative_boundary_distance_px, dtype=np.float32
        ),
        physical_boundary_distance_px=np.asarray(dataset.physical_boundary_distance_px, dtype=np.float32),
        transform_uncertainty=np.asarray(dataset.transform_uncertainty, dtype=np.float32),
        metadata_json=np.asarray(json.dumps(dict(dataset.metadata), sort_keys=True)),
    )


def load_dataset(path: Path) -> LocalCoordinateDataset:
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"].item())) if "metadata_json" in data else {}
        dataset = LocalCoordinateDataset(
            frame_indices=np.asarray(data["frame_indices"], dtype=np.int64),
            timestamps_s=np.asarray(data["timestamps_s"], dtype=np.float64),
            traces=np.asarray(data["traces"], dtype=np.float32),
            pixel_xy=np.asarray(data["pixel_xy"], dtype=np.float64),
            pixel_valid=np.asarray(data["pixel_valid"], dtype=bool),
            frame_valid=np.asarray(data["frame_valid"], dtype=bool),
            source_xy=np.asarray(data["source_xy"], dtype=np.float32),
            bilinear_weights=np.asarray(data["bilinear_weights"], dtype=np.float32),
            body_occupancy=np.asarray(data["body_occupancy"], dtype=np.float32),
            eye_occupancy=np.asarray(data["eye_occupancy"], dtype=np.float32),
            gradient_magnitude=np.asarray(data["gradient_magnitude"], dtype=np.float32),
            motion_prediction=np.asarray(data["motion_prediction"], dtype=np.float32),
            nuisance_values=np.asarray(data["nuisance_values"], dtype=np.float64),
            nuisance_names=tuple(str(value) for value in data["nuisance_names"].tolist()),
            image_shape_hw=tuple(int(value) for value in data["image_shape_hw"].tolist()),
            administrative_boundary_distance_px=np.asarray(
                data["administrative_boundary_distance_px"], dtype=np.float64
            ),
            physical_boundary_distance_px=np.asarray(data["physical_boundary_distance_px"], dtype=np.float64),
            transform_uncertainty=np.asarray(data["transform_uncertainty"], dtype=np.float64),
            metadata=metadata,
        )
    return dataset.validated()


def _write_dict_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    ensure_output_dir(path.parent)
    if not rows:
        path.write_text("")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(str(key))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _result_summary(result: Any, config: HeartrateConfig, dataset: LocalCoordinateDataset) -> dict[str, Any]:
    fold_rows: list[dict[str, Any]] = []
    for fold in result.folds:
        candidate = fold.discovery.candidate
        fold_rows.append(
            {
                "fold_index": int(fold.fold_index),
                "discovery_detected": bool(fold.discovery.detected),
                "discovery_p_value": float(fold.discovery.p_value),
                "discovery_cluster_mass": float(candidate.cluster_mass),
                "discovery_null_threshold": float(fold.discovery.threshold),
                "frequency_hz": float(candidate.frequency_hz),
                "selected_pixel_count": int(candidate.pixel_indices.size),
                "chunk_count": int(candidate.chunk_count),
                "spatial_phase_coherence": float(candidate.spatial_phase_coherence),
                "chunk_phase_coherence": float(candidate.chunk_phase_coherence),
                "confirmation_p_value": float(fold.confirmation_p_value),
                "confirmation_score": float(fold.confirmation_score),
                "confirmation_chunk_scores": fold.confirmation_chunk_scores.tolist(),
                "confirmation_chunk_p_values": fold.confirmation_chunk_p_values.tolist(),
                "confirmed_chunk_count": int(fold.confirmed_chunk_count),
                "confirmation_chunk_count": int(fold.confirmation_chunk_count),
                "control_scores": {str(key): float(value) for key, value in fold.control_scores.items()},
                "control_ratio": float(fold.control_ratio),
                "confirmed": bool(fold.confirmed),
                "polarity": str(fold.polarity),
                "event_count": int(fold.events.timestamps_s.size) if fold.events is not None else 0,
                "rejected_filter_edge_events": int(fold.events.rejected_edge_events)
                if fold.events is not None
                else 0,
            }
        )
    finite_bpm = np.asarray(result.instantaneous_bpm, dtype=np.float64)
    finite_bpm = finite_bpm[np.isfinite(finite_bpm)]
    return {
        "detected": bool(result.detected),
        "reason": str(result.reason),
        "crossfit_dilated_overlap": float(result.crossfit_dilated_overlap),
        "crossfit_frequency_difference_hz": float(result.crossfit_frequency_difference_hz),
        "event_count": int(result.event_timestamps_s.size),
        "bpm_median": float(np.median(finite_bpm)) if finite_bpm.size else None,
        "bpm_iqr": float(np.quantile(finite_bpm, 0.75) - np.quantile(finite_bpm, 0.25))
        if finite_bpm.size >= 2
        else None,
        "coverage_fraction": float(result.coverage_fraction),
        "no_estimate_intervals_s": [list(interval) for interval in result.no_estimate_intervals_s],
        "folds": fold_rows,
        "config": asdict(config),
        "dataset_metadata": dict(dataset.metadata),
    }


def _write_analysis_outputs(
    output_prefix: Path,
    dataset: LocalCoordinateDataset,
    config: HeartrateConfig,
    result: Any,
) -> dict[str, str]:
    summary_path = output_prefix.with_suffix(".analysis.summary.json")
    events_path = output_prefix.with_suffix(".analysis.events.csv")
    arrays_path = output_prefix.with_suffix(".analysis.arrays.npz")
    figure_path = output_prefix.with_suffix(".analysis.diagnostic.png")
    summary = _result_summary(result, config, dataset)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    event_rows: list[dict[str, Any]] = []
    for index, (frame, timestamp) in enumerate(
        zip(result.event_frame_indices.tolist(), result.event_timestamps_s.tolist())
    ):
        event_rows.append(
            {
                "event_index": int(index),
                "frame_index": int(frame),
                "timestamp_s": float(timestamp),
            }
        )
    _write_dict_rows(events_path, event_rows)
    risks = build_risk_surfaces(dataset, config)
    arrays: dict[str, Any] = {
        "eligible_pixels": risks.eligible.astype(np.uint8),
        "combined_risk_penalty": risks.combined_penalty.astype(np.float32),
        "physical_boundary_distance_px": risks.physical_boundary_distance_px.astype(np.float32),
        "administrative_boundary_distance_px": risks.administrative_boundary_distance_px.astype(np.float32),
        "pixel_xy": np.asarray(dataset.pixel_xy, dtype=np.float32),
    }
    for fold in result.folds:
        arrays[f"fold_{fold.fold_index}_cluster_mask"] = (
            fold.discovery.candidate.cluster_mask.astype(np.uint8)
        )
        arrays[f"fold_{fold.fold_index}_pixel_scores"] = (
            fold.discovery.candidate.pixel_scores.astype(np.float32)
        )
        arrays[f"fold_{fold.fold_index}_confirmation_trace"] = fold.confirmation_trace.astype(np.float32)
        arrays[f"fold_{fold.fold_index}_confirmation_valid"] = fold.confirmation_valid.astype(np.uint8)
        arrays[f"fold_{fold.fold_index}_event_valid"] = fold.event_valid.astype(np.uint8)
        arrays[f"fold_{fold.fold_index}_discovery_null"] = (
            fold.discovery.null_max_cluster_mass.astype(np.float32)
        )
        arrays[f"fold_{fold.fold_index}_confirmation_null"] = fold.confirmation_null_scores.astype(np.float32)
    np.savez_compressed(arrays_path, **arrays)
    _write_diagnostic_figure(figure_path, dataset, config, result, risks)
    return {
        "summary_json": str(summary_path),
        "events_csv": str(events_path),
        "arrays_npz": str(arrays_path),
        "diagnostic_png": str(figure_path),
    }


def _scatter(values: np.ndarray, dataset: LocalCoordinateDataset) -> np.ndarray:
    image = np.full(dataset.image_shape_hw, np.nan, dtype=np.float64)
    xy = np.rint(dataset.pixel_xy).astype(np.int64)
    image[xy[:, 1], xy[:, 0]] = np.asarray(values, dtype=np.float64)
    return image


def _scatter_mask(values: np.ndarray, dataset: LocalCoordinateDataset) -> np.ndarray:
    image = np.zeros(dataset.image_shape_hw, dtype=bool)
    xy = np.rint(dataset.pixel_xy).astype(np.int64)
    selected = np.asarray(values, dtype=bool)
    image[xy[selected, 1], xy[selected, 0]] = True
    return image


def _load_canonical_mask(
    path: Path | None,
    *,
    key: str,
    shape_hw: tuple[int, int],
) -> np.ndarray | None:
    if path is None:
        return None
    with np.load(path, allow_pickle=False) as data:
        if key not in data:
            raise ValueError(f"{path} does not contain mask key {key!r}")
        mask = np.asarray(data[key], dtype=bool)
    if mask.shape != shape_hw:
        raise ValueError(f"{path}:{key} shape {mask.shape} does not match {shape_hw}")
    return mask


def _canonical_mask_at_pixels(
    mask: np.ndarray | None,
    dataset: LocalCoordinateDataset,
) -> np.ndarray | None:
    if mask is None:
        return None
    xy = np.rint(np.asarray(dataset.pixel_xy, dtype=np.float64)).astype(np.int64)
    selected = np.zeros(dataset.pixel_count, dtype=bool)
    inside = (
        (xy[:, 0] >= 0)
        & (xy[:, 0] < mask.shape[1])
        & (xy[:, 1] >= 0)
        & (xy[:, 1] < mask.shape[0])
    )
    selected[inside] = np.asarray(mask, dtype=bool)[xy[inside, 1], xy[inside, 0]]
    return selected


def _dynamic_support_summary(result: DynamicHeartSupportResult) -> dict[str, Any]:
    return {
        "support_source": str(result.support_source),
        "frequency_search_source": str(result.frequency_search_source),
        "confirmatory_eligible": bool(result.confirmatory_eligible),
        "interpretation": str(result.interpretation),
        "phase_pattern": str(result.phase_pattern),
        "frequency_hz": float(result.frequency_hz),
        "frequency_search_min_hz": float(np.min(result.frequency_grid_hz)),
        "frequency_search_max_hz": float(np.max(result.frequency_grid_hz)),
        "support_score": float(result.support_score),
        "support_p_value": float(result.support_p_value),
        "support_exceeds_null": bool(result.support_exceeds_null),
        "shared_phase_score": float(result.shared_phase_score),
        "shared_phase_p_value": float(result.shared_phase_p_value),
        "shared_phase_exceeds_null": bool(result.shared_phase_exceeds_null),
        "joint_p_value": float(result.joint_p_value),
        "joint_exceeds_null": bool(result.joint_exceeds_null),
        "latent_score": float(result.latent_score),
        "latent_p_value": float(result.latent_p_value),
        "latent_exceeds_null": bool(result.latent_exceeds_null),
        "union_to_core_score_ratio": float(result.union_to_core_score_ratio),
        "strongest_control": result.strongest_control,
        "control_ratio": float(result.control_ratio),
        "control_scores": {
            str(name): float(value) for name, value in result.control_scores.items()
        },
        "pixel_counts": {
            str(name): int(np.count_nonzero(selected))
            for name, selected in result.pixel_groups.items()
        },
        "group_summary": {
            str(name): dict(values) for name, values in result.group_summary.items()
        },
        "block_count": int(len(result.block_summary)),
        "surrogate_count": int(result.null_max_shared_phase_scores.size),
    }


def _write_dynamic_support_outputs(
    output_prefix: Path,
    dataset: LocalCoordinateDataset,
    heartrate_result: Any,
    result: DynamicHeartSupportResult,
) -> dict[str, str]:
    summary_path = output_prefix.with_suffix(".dynamic_support.summary.json")
    blocks_path = output_prefix.with_suffix(".dynamic_support.blocks.csv")
    arrays_path = output_prefix.with_suffix(".dynamic_support.arrays.npz")
    figure_path = output_prefix.with_suffix(".dynamic_support.diagnostic.png")
    summary_path.write_text(
        json.dumps(_dynamic_support_summary(result), indent=2, sort_keys=True) + "\n"
    )
    _write_dict_rows(blocks_path, result.block_summary)
    arrays: dict[str, Any] = {
        "pixel_xy": np.asarray(dataset.pixel_xy, dtype=np.float32),
        "frequency_grid_hz": np.asarray(result.frequency_grid_hz, dtype=np.float32),
        "frequency_support_scores": np.asarray(
            result.frequency_support_scores, dtype=np.float32
        ),
        "frequency_shared_phase_scores": np.asarray(
            result.frequency_shared_phase_scores, dtype=np.float32
        ),
        "frequency_latent_scores": np.asarray(
            result.frequency_latent_scores, dtype=np.float32
        ),
        "null_max_support_scores": np.asarray(
            result.null_max_support_scores, dtype=np.float32
        ),
        "null_max_shared_phase_scores": np.asarray(
            result.null_max_shared_phase_scores, dtype=np.float32
        ),
        "null_max_latent_scores": np.asarray(
            result.null_max_latent_scores, dtype=np.float32
        ),
        "block_coefficients_real": np.asarray(
            result.block_coefficients.real, dtype=np.float32
        ),
        "block_coefficients_imag": np.asarray(
            result.block_coefficients.imag, dtype=np.float32
        ),
        "block_model_fold_indices": np.asarray(
            result.block_model_fold_indices, dtype=np.int16
        ),
        "latent_block_coefficients_real": np.asarray(
            result.latent_block_coefficients.real, dtype=np.float32
        ),
        "latent_block_coefficients_imag": np.asarray(
            result.latent_block_coefficients.imag, dtype=np.float32
        ),
        "latent_block_alignment_coherence": np.asarray(
            result.latent_block_alignment_coherence, dtype=np.float32
        ),
    }
    for name, selected in result.pixel_groups.items():
        arrays[f"{name}_mask"] = _scatter_mask(selected, dataset).astype(np.uint8)
    np.savez_compressed(arrays_path, **arrays)
    _write_dynamic_support_figure(
        figure_path,
        dataset,
        heartrate_result,
        result,
    )
    return {
        "dynamic_support_summary_json": str(summary_path),
        "dynamic_support_blocks_csv": str(blocks_path),
        "dynamic_support_arrays_npz": str(arrays_path),
        "dynamic_support_diagnostic_png": str(figure_path),
    }


def _write_dynamic_support_figure(
    path: Path,
    dataset: LocalCoordinateDataset,
    heartrate_result: Any,
    result: DynamicHeartSupportResult,
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/palette-matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    xy = np.asarray(dataset.pixel_xy, dtype=np.float64)
    x0 = max(0, int(np.floor(np.nanmin(xy[:, 0]))) - 2)
    x1 = min(dataset.image_shape_hw[1], int(np.ceil(np.nanmax(xy[:, 0]))) + 3)
    y0 = max(0, int(np.floor(np.nanmin(xy[:, 1]))) - 2)
    y1 = min(dataset.image_shape_hw[0], int(np.ceil(np.nanmax(xy[:, 1]))) + 3)
    image_slice = np.s_[y0:y1, x0:x1]
    mean_image = _scatter(np.nanmedian(dataset.traces, axis=0), dataset)
    score_images = [
        _scatter(fold.discovery.candidate.pixel_scores, dataset)
        for fold in heartrate_result.folds
    ]
    finite_scores = np.concatenate(
        [image[np.isfinite(image)] for image in score_images if np.isfinite(image).any()]
    )
    score_limit = max(1.0, float(np.quantile(np.abs(finite_scores), 0.98)))
    mask_images = {
        name: _scatter_mask(selected, dataset)
        for name, selected in result.pixel_groups.items()
    }
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    fig.patch.set_facecolor("white")
    for axis in axes.flat:
        axis.set_facecolor("white")
    axes[0, 0].imshow(mean_image[image_slice], cmap="gray", interpolation="nearest")
    contour_styles = {
        "heart_support": ("black", "frozen heart support"),
        "core": ("magenta", "core"),
        "fold0_only": ("red", "fold 0 only"),
        "fold1_only": ("blue", "fold 1 only"),
        "anatomical_only": ("gold", "anatomical only"),
        "esophagus_control": ("cyan", "esophagus control"),
    }
    legend: list[Line2D] = []
    for name, (color, label) in contour_styles.items():
        cropped = mask_images[name][image_slice]
        if not np.any(cropped):
            continue
        axes[0, 0].contour(
            cropped.astype(np.float64),
            levels=[0.5],
            colors=[color],
            linewidths=1.5,
        )
        legend.append(Line2D([0], [0], color=color, lw=2, label=label))
    axes[0, 0].set_title("Median source intensity with support contours")
    if legend:
        axes[0, 0].legend(handles=legend, fontsize=8, loc="upper right")
    score_plot = None
    for fold_index, axis in enumerate(axes[0, 1:]):
        score_plot = axis.imshow(
            score_images[fold_index][image_slice],
            cmap="coolwarm",
            vmin=-score_limit,
            vmax=score_limit,
            interpolation="nearest",
        )
        axis.set_title(f"Fold {fold_index} continuous pixel score")
    if score_plot is not None:
        fig.colorbar(score_plot, ax=axes[0, 1:].tolist(), fraction=0.03)
    block_indices = np.arange(len(result.block_summary), dtype=np.int64)
    plotted_groups = [
        name
        for name in ("heart_support", "core", "fold0_only", "fold1_only", "anatomical_only")
        if int(result.group_summary[name]["pixel_count"]) > 0
    ]
    colors = {
        "heart_support": "black",
        "core": "magenta",
        "fold0_only": "red",
        "fold1_only": "blue",
        "anatomical_only": "goldenrod",
    }
    for name in plotted_groups:
        ratios = [
            float(row[f"{name}_target_to_sideband_ratio"])
            for row in result.block_summary
        ]
        axes[1, 0].plot(
            block_indices,
            ratios,
            marker="o",
            color=colors[name],
            label=name.replace("_", " "),
        )
    axes[1, 0].axhline(1.0, color="0.5", lw=1, ls="--")
    axes[1, 0].set_title("Target/sideband amplitude by held-out block")
    axes[1, 0].set_xlabel("block")
    axes[1, 0].legend(fontsize=8)
    for name in ("fold0_only", "fold1_only", "anatomical_only"):
        if int(result.group_summary[name]["pixel_count"]) == 0:
            continue
        offsets = [
            float(row[f"{name}_phase_offset_to_reference_deg"])
            for row in result.block_summary
        ]
        axes[1, 1].plot(
            block_indices,
            offsets,
            marker="o",
            color=colors[name],
            label=name.replace("_", " "),
        )
    axes[1, 1].axhline(0.0, color="0.5", lw=1, ls="--")
    axes[1, 1].set_ylim(-190, 190)
    phase_reference = (
        "reproducible core"
        if int(result.group_summary["core"]["pixel_count"]) >= 3
        else "frozen heart support"
    )
    axes[1, 1].set_title(f"Phase offset to {phase_reference}")
    axes[1, 1].set_xlabel("block")
    if axes[1, 1].lines:
        axes[1, 1].legend(fontsize=8)
    axes[1, 2].hist(
        result.null_max_latent_scores,
        bins=20,
        color="0.75",
        edgecolor="0.4",
    )
    axes[1, 2].axvline(result.latent_score, color="black", lw=1.5)
    axes[1, 2].set_title("Cross-fit whole-mask latent-pattern null")
    source_label = (
        "external anatomical mask"
        if result.support_source == "external_anatomical_mask"
        else "post-hoc cluster union"
    )
    search_label = (
        "explicit bounds"
        if result.frequency_search_source == "explicit_prespecified_bounds"
        else result.frequency_search_source
    )
    union_core = (
        f"{result.union_to_core_score_ratio:.2f}"
        if np.isfinite(result.union_to_core_score_ratio)
        else "n/a"
    )
    lines = [
        f"source: {source_label}",
        f"frequency search: {search_label}",
        f"confirmatory eligible: {result.confirmatory_eligible}",
        f"frequency: {result.frequency_hz:.3f} Hz",
        f"support p: {result.support_p_value:.3g}",
        f"shared-phase p: {result.shared_phase_p_value:.3g}",
        f"joint p: {result.joint_p_value:.3g}",
        f"latent p: {result.latent_p_value:.3g}",
        f"union/core score: {union_core}",
    ]
    axes[1, 2].text(
        0.02,
        0.98,
        "\n".join(lines),
        transform=axes[1, 2].transAxes,
        va="top",
        family="monospace",
        fontsize=8,
    )
    for axis in axes[0]:
        axis.set_axis_off()
    fig.savefig(path, dpi=160, facecolor="white")
    plt.close(fig)


def _write_diagnostic_figure(
    path: Path,
    dataset: LocalCoordinateDataset,
    config: HeartrateConfig,
    result: Any,
    risks: Any,
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/palette-matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mean_image = _scatter(np.nanmedian(dataset.traces, axis=0), dataset)
    risk_image = _scatter(risks.combined_penalty, dataset)
    xy = np.asarray(dataset.pixel_xy, dtype=np.float64)
    x0 = max(0, int(np.floor(np.nanmin(xy[:, 0]))) - 2)
    x1 = min(dataset.image_shape_hw[1], int(np.ceil(np.nanmax(xy[:, 0]))) + 3)
    y0 = max(0, int(np.floor(np.nanmin(xy[:, 1]))) - 2)
    y1 = min(dataset.image_shape_hw[0], int(np.ceil(np.nanmax(xy[:, 1]))) + 3)
    image_slice = np.s_[y0:y1, x0:x1]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    axes[0, 0].imshow(mean_image[image_slice], cmap="gray", interpolation="nearest")
    axes[0, 0].set_title("Median original-pixel intensity (source crop)")
    image = axes[0, 1].imshow(
        risk_image[image_slice], cmap="magma", interpolation="nearest"
    )
    axes[0, 1].set_title("Physical/measurement risk penalty")
    fig.colorbar(image, ax=axes[0, 1], fraction=0.046)
    overlay = np.zeros((*dataset.image_shape_hw, 3), dtype=np.float32)
    if result.folds:
        overlay[result.folds[0].discovery.candidate.cluster_mask, 0] = 1.0
        overlay[result.folds[1].discovery.candidate.cluster_mask, 2] = 1.0
    axes[0, 2].imshow(overlay[image_slice], interpolation="nearest")
    axes[0, 2].set_title(f"Cross-fit clusters, overlap={result.crossfit_dilated_overlap:.2f}")
    time = np.asarray(dataset.timestamps_s) - float(dataset.timestamps_s[0])
    for fold in result.folds:
        trace = np.asarray(fold.confirmation_trace, dtype=np.float64)
        axes[1, 0].plot(time, trace, lw=0.6, label=f"fold {fold.fold_index}")
    axes[1, 0].set_title("Held-out residual traces")
    axes[1, 0].set_xlabel("time (s)")
    if result.folds:
        axes[1, 0].legend(fontsize=8)
    for fold in result.folds:
        axes[1, 1].hist(
            fold.discovery.null_max_cluster_mass,
            bins=20,
            alpha=0.45,
            label=f"fold {fold.fold_index}",
        )
        axes[1, 1].axvline(fold.discovery.candidate.cluster_mass, lw=1.0)
    axes[1, 1].set_title("Full-pipeline discovery null")
    if result.folds:
        axes[1, 1].legend(fontsize=8)
    axes[1, 2].axis("off")
    lines = [
        f"detected: {result.detected}",
        f"reason: {result.reason}",
        f"events: {result.event_timestamps_s.size}",
        f"coverage: {result.coverage_fraction:.3f}",
        f"cross-fit df: {result.crossfit_frequency_difference_hz:.3f} Hz",
        f"band: {config.band_min_hz:g}-{config.band_max_hz:g} Hz",
    ]
    for fold in result.folds:
        lines.append(
            f"fold {fold.fold_index}: f={fold.discovery.candidate.frequency_hz:.3f} Hz "
            f"pD={fold.discovery.p_value:.3g} pC={fold.confirmation_p_value:.3g} "
            f"control={fold.control_ratio:.2f}"
        )
    axes[1, 2].text(0.0, 1.0, "\n".join(lines), va="top", family="monospace", fontsize=9)
    for axis in axes.flat[:3]:
        axis.set_axis_off()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _injection_specs(dataset: LocalCoordinateDataset, args: argparse.Namespace) -> list[InjectionSpec]:
    amplitudes = _split_floats(args.injection_amplitudes)
    frequencies = _split_floats(args.injection_frequencies_hz)
    radii = _split_floats(args.injection_radii_px)
    drifts = _split_floats(args.injection_phase_drifts_hz_per_s)
    active_fractions = _split_floats(args.injection_active_fractions)
    center = np.median(np.asarray(dataset.pixel_xy, dtype=np.float64), axis=0)
    risks = np.asarray(dataset.physical_boundary_distance_px, dtype=np.float64)
    edge_index = int(np.nanargmin(risks))
    locations = [tuple(center.tolist()), tuple(np.asarray(dataset.pixel_xy)[edge_index].tolist())]
    if str(args.injection_locations) == "center":
        locations = locations[:1]
    elif str(args.injection_locations) == "edge":
        locations = locations[1:]
    full = [
        InjectionSpec(
            amplitude_sigma=amplitude,
            frequency_hz=frequency,
            center_xy=(float(location[0]), float(location[1])),
            radius_px=radius,
            phase_drift_hz_per_s=drift,
            active_fraction=active,
        )
        for amplitude in amplitudes
        for frequency in frequencies
        for radius in radii
        for drift in drifts
        for active in active_fractions
        for location in locations
    ]
    if str(args.injection_design) == "full-factorial":
        return full

    positive_amplitudes = [value for value in amplitudes if value > 0.0]
    maximum_amplitude = max(positive_amplitudes) if positive_amplitudes else 0.0
    baseline_frequency = frequencies[len(frequencies) // 2]
    baseline_radius = radii[0]
    baseline_drift = drifts[0]
    baseline_active = max(active_fractions)
    baseline_location = locations[0]
    coverage: list[InjectionSpec] = [
        InjectionSpec(0.0, baseline_frequency, baseline_location, baseline_radius),
    ]
    coverage.extend(
        InjectionSpec(amplitude, baseline_frequency, baseline_location, baseline_radius)
        for amplitude in positive_amplitudes
    )
    coverage.extend(
        InjectionSpec(maximum_amplitude, frequency, baseline_location, baseline_radius)
        for frequency in frequencies
    )
    coverage.extend(
        InjectionSpec(maximum_amplitude, baseline_frequency, baseline_location, radius)
        for radius in radii
    )
    coverage.extend(
        InjectionSpec(
            maximum_amplitude,
            baseline_frequency,
            baseline_location,
            baseline_radius,
            phase_drift_hz_per_s=drift,
        )
        for drift in drifts
    )
    coverage.extend(
        InjectionSpec(
            maximum_amplitude,
            baseline_frequency,
            baseline_location,
            baseline_radius,
            active_fraction=active,
        )
        for active in active_fractions
    )
    coverage.extend(
        InjectionSpec(maximum_amplitude, baseline_frequency, location, baseline_radius)
        for location in locations
    )
    return list(dict.fromkeys(coverage))


def _config_from_args(args: argparse.Namespace) -> HeartrateConfig:
    return HeartrateConfig(
        band_min_hz=float(args.band_min_hz),
        band_max_hz=float(args.band_max_hz),
        frequency_step_hz=float(args.frequency_step_hz),
        partition_block_seconds=float(args.partition_block_seconds),
        partition_guard_seconds=float(args.partition_guard_seconds),
        min_partition_blocks_per_fold=int(args.min_partition_blocks_per_fold),
        discovery_chunk_seconds=float(args.discovery_chunk_seconds),
        min_chunk_seconds=float(args.min_chunk_seconds),
        max_interpolated_gap_seconds=float(args.max_interpolated_gap_seconds),
        min_pixel_valid_fraction=float(args.min_pixel_valid_fraction),
        min_body_occupancy=float(args.min_body_occupancy),
        max_eye_occupancy=float(args.max_eye_occupancy),
        min_physical_boundary_distance_px=float(args.min_physical_boundary_distance_px),
        max_warp_invalid_fraction=float(args.max_warp_invalid_fraction),
        gradient_risk_weight=float(args.gradient_risk_weight),
        boundary_risk_weight=float(args.boundary_risk_weight),
        warp_risk_weight=float(args.warp_risk_weight),
        transform_risk_weight=float(args.transform_risk_weight),
        pixel_score_threshold_z=float(args.pixel_score_threshold_z),
        min_cluster_pixels=int(args.min_cluster_pixels),
        surrogate_count=int(args.surrogate_count),
        surrogate_spatial_block_px=int(args.surrogate_spatial_block_px),
        surrogate_min_shift_seconds=float(args.surrogate_min_shift_seconds),
        alpha=float(args.alpha),
        min_control_ratio=float(args.min_control_ratio),
        min_crossfit_dilated_overlap=float(args.min_crossfit_dilated_overlap),
        max_crossfit_frequency_difference_hz=float(args.max_crossfit_frequency_difference_hz),
        event_polarity=str(args.event_polarity),
        event_prominence_mad=float(args.event_prominence_mad),
        event_filter_edge_seconds=float(args.event_filter_edge_seconds),
        random_seed=int(args.seed),
    ).validated()


def _add_analysis_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--band-min-hz", type=float, default=1.5)
    parser.add_argument("--band-max-hz", type=float, default=3.5)
    parser.add_argument("--frequency-step-hz", type=float, default=0.05)
    parser.add_argument("--partition-block-seconds", type=float, default=4.0)
    parser.add_argument("--partition-guard-seconds", type=float, default=0.25)
    parser.add_argument("--min-partition-blocks-per-fold", type=int, default=2)
    parser.add_argument("--discovery-chunk-seconds", type=float, default=4.0)
    parser.add_argument("--min-chunk-seconds", type=float, default=2.0)
    parser.add_argument("--max-interpolated-gap-seconds", type=float, default=0.02)
    parser.add_argument("--min-pixel-valid-fraction", type=float, default=0.75)
    parser.add_argument("--min-body-occupancy", type=float, default=0.75)
    parser.add_argument("--max-eye-occupancy", type=float, default=0.05)
    parser.add_argument("--min-physical-boundary-distance-px", type=float, default=1.0)
    parser.add_argument("--max-warp-invalid-fraction", type=float, default=0.25)
    parser.add_argument("--gradient-risk-weight", type=float, default=0.25)
    parser.add_argument("--boundary-risk-weight", type=float, default=0.5)
    parser.add_argument("--warp-risk-weight", type=float, default=1.0)
    parser.add_argument("--transform-risk-weight", type=float, default=0.25)
    parser.add_argument("--pixel-score-threshold-z", type=float, default=1.5)
    parser.add_argument("--min-cluster-pixels", type=int, default=3)
    parser.add_argument("--surrogate-count", type=int, default=199)
    parser.add_argument("--surrogate-spatial-block-px", type=int, default=2)
    parser.add_argument("--surrogate-min-shift-seconds", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--min-control-ratio", type=float, default=1.1)
    parser.add_argument("--min-crossfit-dilated-overlap", type=float, default=0.5)
    parser.add_argument("--max-crossfit-frequency-difference-hz", type=float, default=0.1)
    parser.add_argument("--event-polarity", choices=("darkening", "brightening", "auto"), default="darkening")
    parser.add_argument("--event-prominence-mad", type=float, default=1.0)
    parser.add_argument("--event-filter-edge-seconds", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=0)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract and confirm a local-rostral heart-rate signal from original crop-video pixels."
    )
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.example.toml"))
    parser.add_argument(
        "--dataset-npz",
        type=Path,
        default=None,
        help="Reuse an already extracted local-coordinate matrix.",
    )
    parser.add_argument(
        "--roi-json",
        type=Path,
        default=None,
        help="Recorded in provenance; geometry comes from --mask-npz.",
    )
    parser.add_argument("--mask-npz", type=Path, default=None)
    parser.add_argument("--status-csv", type=Path, default=None)
    parser.add_argument("--frame-start", type=int, default=30000)
    parser.add_argument("--frame-count", type=int, default=3000)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--fps", type=float, default=100.0)
    parser.add_argument("--timestamp-column", type=str, default="timestamp")
    parser.add_argument("--timestamp-unit", choices=("auto", "s", "ms", "us", "ns"), default="auto")
    parser.add_argument("--allow-nominal-fps", action="store_true")
    parser.add_argument("--mask-parent", type=str, default=None)
    parser.add_argument("--mask-run", type=str, default=None)
    parser.add_argument(
        "--mask-read-cache-rows",
        type=int,
        default=0,
        help="Cache aligned dense-mask row blocks; use 256 for this recording's Zarr layout.",
    )
    parser.add_argument("--body-component", type=str, default="subject_body")
    parser.add_argument("--eye-components", type=str, default="eye_left,eye_right")
    parser.add_argument("--swim-component", type=str, default="swim_bladder")
    parser.add_argument("--eye-dilate-px", type=int, default=2)
    parser.add_argument("--frame-body-occupancy-threshold", type=float, default=0.5)
    parser.add_argument("--frame-eye-occupancy-threshold", type=float, default=0.5)
    parser.add_argument("--anchor-band-px", type=int, default=2)
    parser.add_argument("--eye-occupancy-threshold", type=float, default=0.50)
    parser.add_argument("--swim-occupancy-threshold", type=float, default=0.25)
    parser.add_argument("--swim-anchor-mode", choices=("upper_edge", "centroid"), default="upper_edge")
    parser.add_argument("--reference-frame-start", type=int, default=30000)
    parser.add_argument("--reference-frame-count", type=int, default=3000)
    parser.add_argument("--reference-stride", type=int, default=10)
    parser.add_argument("--reference-anterior-xy", type=str, default=None)
    parser.add_argument("--reference-posterior-xy", type=str, default=None)
    parser.add_argument("--min-axis-length-px", type=float, default=8.0)
    parser.add_argument("--max-local-rotation-deg", type=float, default=25.0)
    parser.add_argument("--max-local-translation-px", type=float, default=40.0)
    parser.add_argument("--allow-local-scale", action="store_true")
    parser.add_argument("--min-sample-pixels", type=int, default=20)
    _add_analysis_arguments(parser)
    parser.add_argument("--extract-only", action="store_true")
    parser.add_argument("--analyze-dynamic-support", action="store_true")
    parser.add_argument("--dynamic-heart-mask-npz", type=Path, default=None)
    parser.add_argument("--dynamic-heart-mask-key", type=str, default="heart_mask")
    parser.add_argument("--dynamic-esophagus-mask-npz", type=Path, default=None)
    parser.add_argument(
        "--dynamic-esophagus-mask-key",
        type=str,
        default="esophagus_mask",
    )
    parser.add_argument(
        "--dynamic-support-mask-independent",
        action="store_true",
        help="Declare that the supplied anatomical mask was fixed independently of this interval.",
    )
    parser.add_argument("--dynamic-support-surrogate-count", type=int, default=199)
    parser.add_argument("--dynamic-support-frequency-margin-hz", type=float, default=0.25)
    parser.add_argument("--dynamic-support-frequency-min-hz", type=float, default=None)
    parser.add_argument("--dynamic-support-frequency-max-hz", type=float, default=None)
    parser.add_argument("--render-dynamic-phase", action="store_true")
    parser.add_argument("--dynamic-phase-frame-stride", type=int, default=3)
    parser.add_argument("--dynamic-phase-playback-fps", type=float, default=30.0)
    parser.add_argument("--dynamic-phase-panel-size", type=int, default=360)
    parser.add_argument("--analyze-regional-phase-delay", action="store_true")
    parser.add_argument("--regional-phase-split-y", type=float, default=None)
    parser.add_argument("--regional-phase-split-gap-px", type=float, default=0.0)
    parser.add_argument("--regional-phase-surrogate-count", type=int, default=199)
    parser.add_argument("--regional-phase-regions-npz", type=Path, default=None)
    parser.add_argument("--regional-phase-upper-key", type=str, default="upper_mask")
    parser.add_argument("--regional-phase-lower-key", type=str, default="lower_mask")
    parser.add_argument("--regional-phase-regions-independent", action="store_true")
    parser.add_argument("--run-injection-study", action="store_true")
    parser.add_argument("--injection-design", choices=("coverage", "full-factorial"), default="coverage")
    parser.add_argument("--injection-amplitudes", type=str, default="0,0.5,1,2")
    parser.add_argument("--injection-frequencies-hz", type=str, default="2.0,2.5,3.0")
    parser.add_argument("--injection-radii-px", type=str, default="1.5,3")
    parser.add_argument("--injection-phase-drifts-hz-per-s", type=str, default="0,0.01")
    parser.add_argument("--injection-active-fractions", type=str, default="1,0.5")
    parser.add_argument("--injection-locations", choices=("center", "edge", "both"), default="both")
    parser.add_argument("--injection-replicates", type=int, default=2)
    parser.add_argument("--injection-surrogate-count", type=int, default=39)
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("playgrounds/heartrate_stabilization/outputs/reliable_local_rostral_heartrate"),
    )
    args = parser.parse_args()
    if args.render_dynamic_phase and not args.analyze_dynamic_support:
        raise ValueError("--render-dynamic-phase requires --analyze-dynamic-support")
    if args.analyze_regional_phase_delay and not args.analyze_dynamic_support:
        raise ValueError(
            "--analyze-regional-phase-delay requires --analyze-dynamic-support"
        )
    output_prefix = Path(args.output_prefix)
    ensure_output_dir(output_prefix.parent)
    dataset_path = output_prefix.with_suffix(".local_pixel_matrix.npz")
    frame_status_path = output_prefix.with_suffix(".local_pixel_matrix.frames.csv")
    extraction_summary_path = output_prefix.with_suffix(".local_pixel_matrix.summary.json")
    if args.dataset_npz is not None:
        dataset = load_dataset(args.dataset_npz)
        extraction_summary = {
            "reused_dataset_npz": str(args.dataset_npz),
            "frame_count": dataset.frame_count,
            "valid_frame_count": int(np.count_nonzero(dataset.frame_valid)),
            "pixel_count": dataset.pixel_count,
            "metadata": dict(dataset.metadata),
        }
    else:
        if args.mask_npz is None:
            raise ValueError("--mask-npz is required when extracting a new dataset")
        dataset, frame_rows, extraction_summary = build_dataset(args)
        save_dataset(dataset_path, dataset)
        _write_dict_rows(frame_status_path, frame_rows)
        extraction_summary_path.write_text(json.dumps(extraction_summary, indent=2, sort_keys=True) + "\n")
    print(f"dataset_frames: {dataset.frame_count}")
    print(f"dataset_valid_frames: {int(np.count_nonzero(dataset.frame_valid))}")
    print(f"dataset_pixels: {dataset.pixel_count}")
    if args.extract_only:
        print(f"dataset_npz: {args.dataset_npz if args.dataset_npz is not None else dataset_path}")
        return
    analysis_config = _config_from_args(args)
    result = analyze_heartrate(dataset, analysis_config)
    outputs = _write_analysis_outputs(output_prefix, dataset, analysis_config, result)
    print(f"detected: {result.detected}")
    print(f"reason: {result.reason}")
    print(f"event_count: {result.event_timestamps_s.size}")
    print(f"coverage_fraction: {result.coverage_fraction:.6f}")
    for name, path in outputs.items():
        print(f"{name}: {path}")
    if args.analyze_dynamic_support:
        heart_mask = _load_canonical_mask(
            args.dynamic_heart_mask_npz,
            key=str(args.dynamic_heart_mask_key),
            shape_hw=dataset.image_shape_hw,
        )
        esophagus_mask = _load_canonical_mask(
            args.dynamic_esophagus_mask_npz,
            key=str(args.dynamic_esophagus_mask_key),
            shape_hw=dataset.image_shape_hw,
        )
        dynamic_result = analyze_dynamic_heart_support(
            dataset,
            analysis_config,
            result,
            heart_mask=heart_mask,
            esophagus_mask=esophagus_mask,
            mask_is_independent=bool(args.dynamic_support_mask_independent),
            frequency_margin_hz=float(args.dynamic_support_frequency_margin_hz),
            frequency_min_hz=args.dynamic_support_frequency_min_hz,
            frequency_max_hz=args.dynamic_support_frequency_max_hz,
            surrogate_count=int(args.dynamic_support_surrogate_count),
            seed=int(args.seed) + 30011,
        )
        dynamic_outputs = _write_dynamic_support_outputs(
            output_prefix,
            dataset,
            result,
            dynamic_result,
        )
        print(f"dynamic_support_source: {dynamic_result.support_source}")
        print(f"dynamic_support_frequency_hz: {dynamic_result.frequency_hz:.6f}")
        print(f"dynamic_support_p_value: {dynamic_result.support_p_value:.6f}")
        print(
            "dynamic_shared_phase_p_value: "
            f"{dynamic_result.shared_phase_p_value:.6f}"
        )
        print(f"dynamic_latent_p_value: {dynamic_result.latent_p_value:.6f}")
        print(f"dynamic_phase_pattern: {dynamic_result.phase_pattern}")
        print(f"dynamic_interpretation: {dynamic_result.interpretation}")
        for name, path in dynamic_outputs.items():
            print(f"{name}: {path}")
        phase = None
        if args.render_dynamic_phase or args.analyze_regional_phase_delay:
            phase = reconstruct_crossfit_heart_phase(
                dataset,
                analysis_config,
                result,
                dynamic_result,
            )
        if args.render_dynamic_phase:
            assert phase is not None
            phase_outputs = write_dynamic_phase_outputs(
                output_prefix,
                dataset,
                dynamic_result,
                phase,
                frame_stride=int(args.dynamic_phase_frame_stride),
                playback_fps=float(args.dynamic_phase_playback_fps),
                panel_size=int(args.dynamic_phase_panel_size),
            )
            print(
                "dynamic_phase_valid_fraction: "
                f"{float(np.mean(phase.frame_valid)):.6f}"
            )
            print(
                "dynamic_phase_median_alignment: "
                f"{float(np.nanmedian(phase.spatial_alignment)):.6f}"
            )
            for name, path in phase_outputs.items():
                print(f"{name}: {path}")
        if args.analyze_regional_phase_delay:
            assert phase is not None
            upper_image = _load_canonical_mask(
                args.regional_phase_regions_npz,
                key=str(args.regional_phase_upper_key),
                shape_hw=dataset.image_shape_hw,
            )
            lower_image = _load_canonical_mask(
                args.regional_phase_regions_npz,
                key=str(args.regional_phase_lower_key),
                shape_hw=dataset.image_shape_hw,
            )
            regional_result = analyze_regional_phase_delay(
                dataset,
                phase,
                upper_pixels=_canonical_mask_at_pixels(upper_image, dataset),
                lower_pixels=_canonical_mask_at_pixels(lower_image, dataset),
                split_y=args.regional_phase_split_y,
                split_gap_px=float(args.regional_phase_split_gap_px),
                regions_independent=bool(args.regional_phase_regions_independent),
                surrogate_count=int(args.regional_phase_surrogate_count),
                alpha=float(analysis_config.alpha),
                max_gap_factor=float(analysis_config.max_timestamp_gap_factor),
                seed=int(args.seed) + 40009,
            )
            regional_outputs = write_regional_phase_delay_outputs(
                output_prefix,
                dataset,
                regional_result,
            )
            print(f"regional_phase_split_y: {regional_result.split_y:.6f}")
            print(
                "regional_phase_lower_lag_ms: "
                f"{regional_result.across_block_lower_lag_ms:.6f}"
            )
            print(
                "regional_phase_across_block_plv: "
                f"{regional_result.across_block_phase_locking_value:.6f}"
            )
            print(
                "regional_phase_stable_delay_p_value: "
                f"{regional_result.stable_delay_p_value:.6f}"
            )
            print(f"regional_phase_interpretation: {regional_result.interpretation}")
            for name, path in regional_outputs.items():
                print(f"{name}: {path}")
    if args.run_injection_study:
        injection_config = replace(
            analysis_config,
            surrogate_count=int(args.injection_surrogate_count),
        )
        specs = _injection_specs(dataset, args)
        injection_rows = run_injection_recovery(
            dataset,
            injection_config,
            specs,
            replicates=int(args.injection_replicates),
            seed=int(args.seed) + 50021,
        )
        injection_csv = output_prefix.with_suffix(".injection_recovery.csv")
        injection_json = output_prefix.with_suffix(".injection_recovery.summary.json")
        _write_dict_rows(injection_csv, injection_rows)
        operating = injection_operating_characteristics(injection_rows)
        injection_json.write_text(
            json.dumps(
                {
                    "background": "spatial-block circular-shift null of the real source-pixel matrix",
                    "design": str(args.injection_design),
                    "spec_count": len(specs),
                    "replicates": int(args.injection_replicates),
                    "analysis_config": asdict(injection_config),
                    "operating_characteristics": operating,
                    "rows_csv": str(injection_csv),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        print(f"injection_recovery_csv: {injection_csv}")
        print(f"injection_recovery_summary_json: {injection_json}")


if __name__ == "__main__":
    main()
