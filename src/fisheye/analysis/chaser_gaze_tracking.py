"""Body-frame eye tracking of chaser objects with rotated wall controls.

The primary question is whether an eye moves in the fish body frame as an
object's bearing changes in that same body frame.  World-frame gaze alignment
is intentionally not used: it would confound eye rotation with the fish simply
turning its body toward the object.

This component is written below an immutable chaser-distance run because it
depends on that run's exact object positions, roles, epochs, and egocentric
bearing component.  It also records an exact eye-angle source.  Frame rows are
descriptive; inference across a cohort must use one recording-level summary per
fish (with session handling where warranted), never pooled frame p-values.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from io import BytesIO
import json
import math
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import zarr

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from fisheye.analysis.chaser_bout_response import (
    DEFAULT_MIN_VIRTUAL_SEPARATION_MM,
    DEFAULT_VIRTUAL_ROTATIONS_DEG,
)
from fisheye.analysis.chaser_distance_runs import _bytes_array, _write_array
from fisheye.analysis.chaser_component_writer import (
    require_chaser_component_staging_capability,
    sealed_chaser_component_writer,
)
from fisheye.analysis.chaser_component_publication import (
    load_chaser_component_handle_json,
    open_explicit_chaser_component_group,
)
from fisheye.analysis.chaser_egocentric_bearing import (
    ANGLE_CONVENTION,
    COMPONENT_PARENT_NAME as EGOCENTRIC_COMPONENT_PARENT,
    SCHEMA_ID as EGOCENTRIC_SCHEMA_ID,
    SCHEMA_VERSION as EGOCENTRIC_SCHEMA_VERSION,
    _load_configured_chaser_behavior_labels,
    compute_egocentric_chaser_bearing,
    wrap_degrees_signed,
)
from fisheye.analysis.chaser_radial_occupancy import (
    _decode_text_column,
    _resolve_arena_geometry,
    _resolve_chaser_distance_run,
    _safe_float,
)
from fisheye.analysis.gaze_convention_validation import (
    EXPECTED_BODY_FRAME_CONVENTION,
    EXPECTED_GAZE_SIGN_CONVENTION,
)
from fisheye.shared.json_safety import decode_null_terminated_text, json_attr_safe
from fisheye.shared.plot_artifacts import write_png_visualization_artifact
from fisheye.shared.run_lineage_fingerprint import build_run_lineage_payload, write_run_lineage_attrs
from fisheye.shared.system_metadata import get_git_info
from fisheye.shared.zarr_io import open_zarr_root
from fisheye.shared.zarr_run_completion import resolve_authoritative_run_name


SCHEMA_ID = "palette.chaser_gaze_tracking.v1"
SCHEMA_VERSION = 1
METHOD = "body_frame_eye_gaze_vs_body_frame_object_bearing_with_rotated_wall_controls"
METHOD_VERSION = "1"
COMPONENT_PARENT_NAME = "gaze_tracking"
DEFAULT_COMPONENT_NAME = "chaser_gaze_tracking_v1"
EYE_LABELS = ("left", "right")
DEFAULT_EYE_RANGE_QUANTILES = (0.01, 0.99)
DEFAULT_LOCK_THRESHOLD_DEG = 10.0
DEFAULT_LOCK_MIN_DURATION_S = 0.10
DEFAULT_MAX_TRACKING_DISTANCE_MM = 50.0
DEFAULT_MAX_LAG_S = 0.50
DEFAULT_MAX_VIRTUAL_COLLISION_FRACTION = 0.05
DEFAULT_DISTANCE_BIN_EDGES_MM = (0.0, 8.0, 16.0, 30.0, 50.0, float("inf"))
DEFAULT_BEARING_BIN_EDGES_DEG = tuple(float(v) for v in range(-180, 181, 30))
SUMMARY_PNG_ARTIFACT_NAME = "chaser_gaze_tracking_summary_png"
SUMMARY_VISUALIZATION_CONTRACT_ID = "palette.chaser_gaze_tracking.summary.v1"
SUMMARY_RENDERER = "fisheye.analysis.chaser_gaze_tracking"
SUMMARY_RENDERER_VERSION = "1"
MIN_REGRESSION_SAMPLES = 30
MIN_REGRESSION_SPAN_DEG = 5.0


@dataclass(frozen=True)
class LinearTrackingFit:
    sample_count: int
    gain: float
    intercept_deg: float
    correlation: float
    bearing_span_deg: float


@dataclass(frozen=True)
class DynamicTrackingFit:
    sample_count: int
    gain: float
    correlation: float
    lag_frames: int
    lag_seconds: float


@dataclass(frozen=True)
class VirtualReference:
    label: str
    parent_chaser_index: int
    rotation_deg: float


@dataclass(frozen=True)
class LockEvent:
    start_frame: int
    end_frame: int
    duration_s: float
    epoch_window_id: int
    eye_index: int
    chaser_index: int
    behavior_class: str
    median_distance_mm: float
    median_bearing_deg: float
    median_gaze_error_deg: float
    mean_vergence_eye_angle_deg: float


@dataclass(frozen=True)
class ChaserGazeTrackingResult:
    recording_id: str
    chaser_distance_run_name: str
    chaser_distance_run_path: str
    egocentric_component_name: str
    egocentric_component_path: str
    egocentric_component_manifest_sha256: str | None
    eye_angle_run_name: str
    eye_angle_run_path: str
    component_name: str
    fps: float
    camera_frame_id: np.ndarray
    stimulus_epoch_window_id: np.ndarray
    epoch_window_id: np.ndarray
    epoch_label: tuple[str, ...]
    epoch_start_frame: np.ndarray
    epoch_end_frame: np.ndarray
    chaser_index: np.ndarray
    chaser_behavior_class: tuple[str, ...]
    eye_range_deg: np.ndarray
    eye_valid: np.ndarray
    major_axis_marginal: np.ndarray
    gaze_signed_deg: np.ndarray
    vergence_eye_angle_deg: np.ndarray
    distance_mm: np.ndarray
    bearing_deg: np.ndarray
    gaze_error_deg: np.ndarray
    accessible: np.ndarray
    lock_on: np.ndarray
    summary: Mapping[str, np.ndarray]
    virtual_references: tuple[VirtualReference, ...]
    virtual_summary: Mapping[str, np.ndarray]
    object_vs_virtual: Mapping[str, np.ndarray]
    binned_summary: Mapping[str, np.ndarray]
    lock_events: tuple[LockEvent, ...]
    diagnostics: Mapping[str, Any]


def fit_linear_tracking_gain(
    object_bearing_deg: np.ndarray,
    gaze_signed_deg: np.ndarray,
    valid: np.ndarray,
    *,
    min_samples: int = MIN_REGRESSION_SAMPLES,
    min_span_deg: float = MIN_REGRESSION_SPAN_DEG,
) -> LinearTrackingFit:
    """Fit body-frame gaze on body-frame object bearing inside an accessible window."""

    x = np.asarray(object_bearing_deg, dtype=np.float64).reshape(-1)
    y = np.asarray(gaze_signed_deg, dtype=np.float64).reshape(-1)
    mask = np.asarray(valid, dtype=bool).reshape(-1) & np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    count = int(x.size)
    if count < int(min_samples):
        return LinearTrackingFit(count, math.nan, math.nan, math.nan, math.nan)
    span = float(np.quantile(x, 0.95) - np.quantile(x, 0.05))
    if not np.isfinite(span) or span < float(min_span_deg):
        return LinearTrackingFit(count, math.nan, math.nan, math.nan, span)
    centered_x = x - float(np.mean(x))
    centered_y = y - float(np.mean(y))
    denominator = float(np.dot(centered_x, centered_x))
    if denominator <= 0.0:
        return LinearTrackingFit(count, math.nan, math.nan, math.nan, span)
    gain = float(np.dot(centered_x, centered_y) / denominator)
    intercept = float(np.mean(y) - gain * np.mean(x))
    x_std = float(np.sqrt(np.dot(centered_x, centered_x)))
    y_std = float(np.sqrt(np.dot(centered_y, centered_y)))
    correlation = float(np.dot(centered_x, centered_y) / (x_std * y_std)) if x_std > 0 and y_std > 0 else math.nan
    return LinearTrackingFit(count, gain, intercept, correlation, span)


def _lagged_pair(x: np.ndarray, y: np.ndarray, valid: np.ndarray, lag: int) -> tuple[np.ndarray, np.ndarray]:
    if lag > 0:
        return x[:-lag][valid[:-lag] & valid[lag:]], y[lag:][valid[:-lag] & valid[lag:]]
    if lag < 0:
        offset = -lag
        return x[offset:][valid[offset:] & valid[:-offset]], y[:-offset][valid[offset:] & valid[:-offset]]
    return x[valid], y[valid]


def fit_dynamic_tracking_gain(
    object_bearing_deg: np.ndarray,
    gaze_signed_deg: np.ndarray,
    valid: np.ndarray,
    *,
    fps: float,
    max_lag_s: float = DEFAULT_MAX_LAG_S,
    min_samples: int = MIN_REGRESSION_SAMPLES,
) -> DynamicTrackingFit:
    """Fit wrapped frame-to-frame gaze changes to bearing changes over a lag grid.

    A positive lag means the eye change follows the object-bearing change.  This
    is a descriptive within-recording estimate; it is not a frame-level
    inferential p-value.
    """

    bearing = np.asarray(object_bearing_deg, dtype=np.float64).reshape(-1)
    gaze = np.asarray(gaze_signed_deg, dtype=np.float64).reshape(-1)
    usable = np.asarray(valid, dtype=bool).reshape(-1)
    if bearing.shape != gaze.shape or usable.shape != bearing.shape:
        raise ValueError("bearing, gaze, and valid must have matching one-dimensional shapes.")
    if bearing.size < 2 or not np.isfinite(fps) or fps <= 0:
        return DynamicTrackingFit(0, math.nan, math.nan, 0, math.nan)
    delta_bearing = wrap_degrees_signed(np.diff(bearing))
    delta_gaze = wrap_degrees_signed(np.diff(gaze))
    delta_valid = usable[:-1] & usable[1:] & np.isfinite(delta_bearing) & np.isfinite(delta_gaze)
    max_lag_frames = max(0, int(round(float(max_lag_s) * float(fps))))
    best: Optional[DynamicTrackingFit] = None
    # A gaze response cannot precede its putative object-bearing driver.  Search
    # zero and positive lags only; negative-lag maxima are useful diagnostics of
    # shared motion/autocorrelation, not evidence that the eye tracked the object.
    for lag in range(0, max_lag_frames + 1):
        x, y = _lagged_pair(delta_bearing, delta_gaze, delta_valid, lag)
        count = int(x.size)
        if count < int(min_samples):
            continue
        centered_x = x - float(np.mean(x))
        centered_y = y - float(np.mean(y))
        denominator = float(np.dot(centered_x, centered_x))
        x_std = float(np.sqrt(denominator))
        y_std = float(np.sqrt(np.dot(centered_y, centered_y)))
        if denominator <= 0 or x_std <= 0 or y_std <= 0:
            continue
        gain = float(np.dot(centered_x, centered_y) / denominator)
        correlation = float(np.dot(centered_x, centered_y) / (x_std * y_std))
        candidate = DynamicTrackingFit(count, gain, correlation, lag, float(lag / fps))
        if best is None or candidate.correlation > best.correlation:
            best = candidate
    return best or DynamicTrackingFit(0, math.nan, math.nan, 0, math.nan)


def sustained_true_runs(mask: np.ndarray, *, min_frames: int) -> tuple[tuple[int, int], ...]:
    """Return inclusive index intervals for sustained true runs."""

    values = np.asarray(mask, dtype=bool).reshape(-1)
    if values.size == 0:
        return ()
    padded = np.pad(values.astype(np.int8), (1, 1))
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    stops = np.flatnonzero(changes == -1) - 1
    return tuple(
        (int(start), int(stop))
        for start, stop in zip(starts.tolist(), stops.tolist())
        if int(stop - start + 1) >= int(min_frames)
    )


def _resolve_eye_run(root: zarr.Group, requested: str) -> tuple[zarr.Group, str, str]:
    parent = root.get("analysis/eye_angle_runs")
    if parent is None:
        raise ValueError("Recording has no analysis/eye_angle_runs.")
    run_name = requested
    if requested == "latest":
        run_name = str(resolve_authoritative_run_name(parent) or "")
    if not run_name or run_name not in parent:
        raise ValueError(f"Eye-angle run {requested!r} is not available.")
    path = f"analysis/eye_angle_runs/{run_name}"
    return parent[run_name], str(run_name), path


def _decode_channel_names(index_group: zarr.Group, count: int) -> list[str]:
    if "name" not in index_group:
        raise ValueError(f"Channel index {index_group.path!r} is missing name.")
    values = np.asarray(index_group["name"][:])
    names = [str(decode_null_terminated_text(value)) for value in values]
    if len(names) != int(count):
        raise ValueError(f"Channel index has {len(names)} names for {count} channels.")
    return names


def _packed_columns(run_group: zarr.Group, data_name: str, index_name: str, requested: Sequence[str]) -> dict[str, np.ndarray]:
    if data_name not in run_group or index_name not in run_group:
        raise ValueError(f"Eye-angle run lacks compact {data_name}/{index_name}.")
    data = run_group[data_name]
    names = _decode_channel_names(run_group[index_name], int(data.shape[1]))
    missing = [name for name in requested if name not in names]
    if missing:
        raise ValueError(f"{data_name} is missing required channels: {missing}.")
    indexes = [names.index(name) for name in requested]
    try:
        packed = np.asarray(
            data.get_orthogonal_selection((slice(None), indexes))
        )
    except (AttributeError, TypeError, IndexError):
        packed = np.column_stack(
            [np.asarray(data[:, index]) for index in indexes]
        )
    if packed.ndim == 1:
        packed = packed.reshape(-1, 1)
    return {
        name: np.asarray(packed[:, output_index])
        for output_index, name in enumerate(requested)
    }


def _dense_frame_row_lookup(
    source_row_count: int,
    target_frame_id: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Map camera frames to the bounded dense ``frame_angles`` row axis.

    ``roi_angles`` uses sparse detection rows and ``support/frame_indices``.
    ``frame_angles`` is the dense camera-frame projection, but its length ends
    at the last frame represented by a detection row. Recording-tail frames can
    therefore be outside this array and must remain explicitly unavailable.
    """

    target = np.asarray(target_frame_id, dtype=np.int64).reshape(-1)
    count = int(source_row_count)
    if count < 0:
        raise ValueError("source_row_count must be non-negative.")
    matched = (target >= 0) & (target < count)
    row_index = np.full(target.shape, -1, dtype=np.int64)
    row_index[matched] = target[matched]
    return row_index, matched


def _metadata_eye_convention(run_group: zarr.Group) -> None:
    attrs = dict(run_group.attrs)
    if str(attrs.get("layout") or attrs.get("storage_layout") or "") != "compact_dense_v2":
        raise ValueError("chaser_gaze_tracking requires a modern compact_dense_v2 eye-angle run.")
    if str(attrs.get("body_frame_angle_convention") or "") != EXPECTED_BODY_FRAME_CONVENTION:
        raise ValueError("Eye-angle run does not declare the canonical body-frame angle convention.")
    variant = attrs.get("eye_angle_variant_schema")
    gaze: Mapping[str, Any] = {}
    if isinstance(variant, Mapping):
        representations = variant.get("representations")
        if isinstance(representations, Mapping) and isinstance(representations.get("gaze"), Mapping):
            gaze = representations["gaze"]
    if str(gaze.get("coordinate_frame") or "") != "fish_body_frame" or str(gaze.get("sign_convention") or "") != EXPECTED_GAZE_SIGN_CONVENTION:
        raise ValueError("Eye-angle gaze fields are not declared fish-body-frame/anatomical-left-positive.")


def _resolve_egocentric_component(
    root: zarr.Group,
    *,
    snapshot: Any,
    run_group: zarr.Group,
    requested: str,
    dependency_handle: Mapping[str, Any] | None,
    legacy_compatibility: bool,
) -> tuple[zarr.Group, str, str, str | None]:
    if dependency_handle is not None:
        verified = open_explicit_chaser_component_group(
            root,
            snapshot=snapshot,
            handle=dependency_handle,
            expected_semantic_schema_id=EGOCENTRIC_SCHEMA_ID,
            expected_semantic_schema_version=EGOCENTRIC_SCHEMA_VERSION,
        )
        if verified.component_family != EGOCENTRIC_COMPONENT_PARENT:
            raise ValueError(
                "Egocentric dependency handle belongs to a different component family."
            )
        component = verified.group
        name = verified.component_name
        path = verified.component_path
        manifest_sha256: str | None = verified.manifest_sha256
    else:
        if not legacy_compatibility:
            raise ValueError(
                "chaser_gaze_tracking requires an explicit self-digested "
                "egocentric dependency handle; historical name/latest discovery "
                "requires legacy_egocentric_component_compatibility=True."
            )
        parent = run_group.get(EGOCENTRIC_COMPONENT_PARENT)
        if parent is None:
            raise ValueError("Chaser-distance run has no egocentric_bearing component.")
        name = requested
        if requested == "latest":
            name = str(
                parent.attrs.get("latest_complete")
                or parent.attrs.get("latest")
                or ""
            )
        if not name or name not in parent:
            raise ValueError(f"Egocentric-bearing component {requested!r} is not available.")
        component = parent[name]
        path = f"{snapshot.run_path}/{EGOCENTRIC_COMPONENT_PARENT}/{name}"
        manifest_sha256 = None
    per_chaser = component.get("per_chaser")
    if per_chaser is None or "bearing_deg" not in per_chaser:
        raise ValueError(f"Egocentric component {name!r} lacks per_chaser/bearing_deg.")
    angle_convention = str(per_chaser.attrs.get("angle_convention") or "")
    if "positive=anatomical_left" not in angle_convention:
        raise ValueError("Egocentric bearing does not declare anatomical-left-positive angles.")
    return component, str(name), path, manifest_sha256


def _epoch_table(run_group: zarr.Group) -> tuple[np.ndarray, tuple[str, ...], np.ndarray, np.ndarray]:
    group = run_group["epoch_summary"]
    window_id = np.asarray(group["window_id"][:], dtype=np.int32)
    labels = tuple(_decode_text_column(np.asarray(group["label_bytes"][:])))
    start = np.asarray(group["start_frame"][:], dtype=np.int64)
    end = np.asarray(group["end_frame"][:], dtype=np.int64)
    return window_id, labels, start, end


def _empirical_eye_ranges(gaze: np.ndarray, valid: np.ndarray, quantiles: tuple[float, float]) -> np.ndarray:
    low_q, high_q = quantiles
    if not 0.0 <= low_q < high_q <= 1.0:
        raise ValueError("eye_range_quantiles must satisfy 0 <= low < high <= 1.")
    ranges = np.full((2, 2), np.nan, dtype=np.float32)
    for eye in range(2):
        values = np.asarray(gaze[:, eye], dtype=np.float64)
        keep = np.asarray(valid[:, eye], dtype=bool) & np.isfinite(values)
        if int(np.sum(keep)) < MIN_REGRESSION_SAMPLES:
            continue
        low, high = np.quantile(values[keep], [low_q, high_q])
        if float(high - low) >= 180.0:
            raise ValueError(
                f"Empirical {EYE_LABELS[eye]} gaze range spans >=180 degrees; a linear accessibility window is ambiguous."
            )
        ranges[eye] = (float(low), float(high))
    return ranges


def _summary_shape(epoch_count: int, chaser_count: int) -> tuple[int, int, int]:
    return epoch_count, chaser_count, 2


def _empty_summary(epoch_count: int, chaser_count: int) -> dict[str, np.ndarray]:
    shape = _summary_shape(epoch_count, chaser_count)
    return {
        "valid_frame_count": np.zeros(shape, dtype=np.int64),
        "accessible_frame_count": np.zeros(shape, dtype=np.int64),
        "lock_on_frame_count": np.zeros(shape, dtype=np.int64),
        "lock_on_fraction": np.full(shape, np.nan, dtype=np.float32),
        "median_abs_gaze_error_deg": np.full(shape, np.nan, dtype=np.float32),
        "tracking_gain": np.full(shape, np.nan, dtype=np.float32),
        "tracking_intercept_deg": np.full(shape, np.nan, dtype=np.float32),
        "tracking_correlation": np.full(shape, np.nan, dtype=np.float32),
        "bearing_span_deg": np.full(shape, np.nan, dtype=np.float32),
        "dynamic_zero_lag_gain": np.full(shape, np.nan, dtype=np.float32),
        "dynamic_zero_lag_correlation": np.full(shape, np.nan, dtype=np.float32),
        "dynamic_best_lag_gain": np.full(shape, np.nan, dtype=np.float32),
        "dynamic_best_lag_correlation": np.full(shape, np.nan, dtype=np.float32),
        "dynamic_best_lag_frames": np.zeros(shape, dtype=np.int32),
        "dynamic_best_lag_seconds": np.full(shape, np.nan, dtype=np.float32),
        "mean_vergence_eye_angle_deg": np.full(shape, np.nan, dtype=np.float32),
    }


def _fill_summary(
    *,
    epoch_ids_by_frame: np.ndarray,
    epoch_window_ids: np.ndarray,
    gaze: np.ndarray,
    vergence: np.ndarray,
    bearing: np.ndarray,
    distance: np.ndarray,
    valid: np.ndarray,
    accessible: np.ndarray,
    lock_on: np.ndarray,
    fps: float,
    max_lag_s: float,
) -> dict[str, np.ndarray]:
    epoch_count = int(epoch_window_ids.size)
    chaser_count = int(bearing.shape[1])
    summary = _empty_summary(epoch_count, chaser_count)
    for epoch_pos, window_id in enumerate(epoch_window_ids.tolist()):
        epoch_mask = np.asarray(epoch_ids_by_frame == int(window_id), dtype=bool)
        for chaser in range(chaser_count):
            for eye in range(2):
                base = epoch_mask & valid[:, eye, chaser]
                usable = base & accessible[:, eye, chaser]
                summary["valid_frame_count"][epoch_pos, chaser, eye] = int(np.sum(base))
                summary["accessible_frame_count"][epoch_pos, chaser, eye] = int(np.sum(usable))
                locked = usable & lock_on[:, eye, chaser]
                summary["lock_on_frame_count"][epoch_pos, chaser, eye] = int(np.sum(locked))
                if np.any(usable):
                    summary["lock_on_fraction"][epoch_pos, chaser, eye] = float(np.mean(lock_on[usable, eye, chaser]))
                    error = wrap_degrees_signed(gaze[:, eye] - bearing[:, chaser])
                    summary["median_abs_gaze_error_deg"][epoch_pos, chaser, eye] = float(np.median(np.abs(error[usable])))
                    vergence_values = vergence[usable]
                    finite_vergence = vergence_values[np.isfinite(vergence_values)]
                    if finite_vergence.size:
                        summary["mean_vergence_eye_angle_deg"][epoch_pos, chaser, eye] = float(np.mean(finite_vergence))
                fit = fit_linear_tracking_gain(bearing[:, chaser], gaze[:, eye], usable)
                summary["tracking_gain"][epoch_pos, chaser, eye] = fit.gain
                summary["tracking_intercept_deg"][epoch_pos, chaser, eye] = fit.intercept_deg
                summary["tracking_correlation"][epoch_pos, chaser, eye] = fit.correlation
                summary["bearing_span_deg"][epoch_pos, chaser, eye] = fit.bearing_span_deg
                zero_lag = fit_dynamic_tracking_gain(
                    bearing[:, chaser],
                    gaze[:, eye],
                    usable,
                    fps=fps,
                    max_lag_s=0.0,
                )
                summary["dynamic_zero_lag_gain"][epoch_pos, chaser, eye] = zero_lag.gain
                summary["dynamic_zero_lag_correlation"][epoch_pos, chaser, eye] = zero_lag.correlation
                dynamic = fit_dynamic_tracking_gain(
                    bearing[:, chaser],
                    gaze[:, eye],
                    usable,
                    fps=fps,
                    max_lag_s=max_lag_s,
                )
                summary["dynamic_best_lag_gain"][epoch_pos, chaser, eye] = dynamic.gain
                summary["dynamic_best_lag_correlation"][epoch_pos, chaser, eye] = dynamic.correlation
                summary["dynamic_best_lag_frames"][epoch_pos, chaser, eye] = dynamic.lag_frames
                summary["dynamic_best_lag_seconds"][epoch_pos, chaser, eye] = dynamic.lag_seconds
    return summary


def _virtual_positions(
    *,
    chaser_xy: np.ndarray,
    chaser_indices: np.ndarray,
    center_xy: tuple[float, float],
    rotations_deg: Sequence[float],
    min_separation_mm: float,
    pixels_per_mm: float,
    max_collision_fraction: float = DEFAULT_MAX_VIRTUAL_COLLISION_FRACTION,
) -> tuple[tuple[VirtualReference, ...], np.ndarray]:
    center = np.asarray(center_xy, dtype=np.float64)
    real = np.asarray(chaser_xy, dtype=np.float64)
    references: list[VirtualReference] = []
    positions: list[np.ndarray] = []
    threshold_px = float(min_separation_mm) * float(pixels_per_mm)
    for chaser_pos, chaser_index in enumerate(chaser_indices.tolist()):
        rel = real[:, chaser_pos, :] - center
        for rotation in rotations_deg:
            theta = np.deg2rad(float(rotation))
            cos_t, sin_t = float(np.cos(theta)), float(np.sin(theta))
            rotated = np.column_stack(
                (rel[:, 0] * cos_t - rel[:, 1] * sin_t, rel[:, 0] * sin_t + rel[:, 1] * cos_t)
            ) + center
            collides = False
            for other in range(real.shape[1]):
                gap = np.linalg.norm(rotated - real[:, other, :], axis=1)
                finite = gap[np.isfinite(gap)]
                collision_fraction = (
                    float(np.mean(finite < threshold_px)) if finite.size else 0.0
                )
                if collision_fraction > float(max_collision_fraction):
                    collides = True
                    break
            if collides:
                continue
            references.append(
                VirtualReference(
                    label=f"virtual_chaser{int(chaser_index)}_{float(rotation):g}",
                    parent_chaser_index=int(chaser_index),
                    rotation_deg=float(rotation),
                )
            )
            positions.append(rotated)
    if not positions:
        return (), np.empty((real.shape[0], 0, 2), dtype=np.float64)
    return tuple(references), np.stack(positions, axis=1)


def _heading_alignment_gate(
    *,
    eye_group: zarr.Group,
    camera_frame_id: np.ndarray,
    track_heading_deg: np.ndarray,
    track_heading_valid: np.ndarray,
    median_tolerance_deg: float = 20.0,
    minimum_resultant_length: float = 0.80,
) -> dict[str, float | int | bool]:
    """Empirically gate eye-body and track-heading sign/lineage alignment."""

    required = (
        "support/frame_indices",
        "support/body_frame/heading_deg",
        "support/body_frame/valid",
    )
    missing = [path for path in required if eye_group.get(path) is None]
    if missing:
        raise ValueError(f"Eye-angle run lacks body-heading alignment support: {missing}.")
    eye_frames = np.asarray(eye_group["support/frame_indices"][:], dtype=np.int64)
    eye_heading = np.asarray(eye_group["support/body_frame/heading_deg"][:], dtype=np.float64)
    eye_valid = np.asarray(eye_group["support/body_frame/valid"][:], dtype=bool)
    if eye_frames.shape != eye_heading.shape or eye_valid.shape != eye_frames.shape:
        raise ValueError("Eye body-heading support arrays have inconsistent row counts.")
    frame_axis = np.asarray(camera_frame_id, dtype=np.int64)
    track_heading = np.asarray(track_heading_deg, dtype=np.float64)
    track_valid = np.asarray(track_heading_valid, dtype=bool)
    if frame_axis.shape != track_heading.shape or track_valid.shape != frame_axis.shape:
        raise ValueError("Egocentric track-heading arrays have inconsistent frame axes.")
    if frame_axis.size == 0 or np.any(np.diff(frame_axis) <= 0):
        raise ValueError("Chaser camera_frame_id must be a strictly increasing axis.")
    positions = np.searchsorted(frame_axis, eye_frames)
    in_range = (positions >= 0) & (positions < frame_axis.size)
    matched = np.zeros(eye_frames.shape, dtype=bool)
    matched[in_range] = frame_axis[positions[in_range]] == eye_frames[in_range]
    base_indices = np.flatnonzero(matched & eye_valid & np.isfinite(eye_heading))
    if base_indices.size:
        base_positions = positions[base_indices]
        keep = track_valid[base_positions] & np.isfinite(track_heading[base_positions])
        base_indices = base_indices[keep]
        base_positions = base_positions[keep]
    else:
        base_positions = np.asarray([], dtype=np.int64)
    if base_indices.size == 0:
        raise ValueError("Eye and egocentric heading sources have no jointly valid frame rows.")
    delta = wrap_degrees_signed(eye_heading[base_indices] - track_heading[base_positions])
    absolute = np.abs(delta)
    median = float(np.median(absolute))
    p95 = float(np.quantile(absolute, 0.95))
    resultant = float(np.abs(np.mean(np.exp(1j * np.deg2rad(delta)))))
    passed = median <= float(median_tolerance_deg) and resultant >= float(minimum_resultant_length)
    report: dict[str, float | int | bool] = {
        "passed": bool(passed),
        "sample_count": int(delta.size),
        "median_abs_difference_deg": median,
        "p95_abs_difference_deg": p95,
        "circular_resultant_length": resultant,
        "median_tolerance_deg": float(median_tolerance_deg),
        "minimum_resultant_length": float(minimum_resultant_length),
    }
    if not passed:
        raise ValueError(
            "Eye body heading and egocentric track heading failed empirical alignment: "
            f"median |delta|={median:.2f} deg, resultant={resultant:.3f}. "
            "Refuse gaze error until sign and lineage are reconciled."
        )
    return report


def _binned_summary(
    *,
    epoch_ids_by_frame: np.ndarray,
    epoch_window_ids: np.ndarray,
    gaze_error: np.ndarray,
    vergence: np.ndarray,
    bearing: np.ndarray,
    distance: np.ndarray,
    valid: np.ndarray,
    accessible: np.ndarray,
    lock_on: np.ndarray,
    distance_edges: np.ndarray,
    bearing_edges: np.ndarray,
) -> dict[str, np.ndarray]:
    shape = (
        int(epoch_window_ids.size),
        int(bearing.shape[1]),
        2,
        int(distance_edges.size - 1),
        int(bearing_edges.size - 1),
    )
    count = np.zeros(shape, dtype=np.int64)
    mean_error = np.full(shape, np.nan, dtype=np.float32)
    lock_fraction = np.full(shape, np.nan, dtype=np.float32)
    mean_vergence = np.full(shape, np.nan, dtype=np.float32)
    for e_pos, window_id in enumerate(epoch_window_ids.tolist()):
        epoch = epoch_ids_by_frame == int(window_id)
        for chaser in range(bearing.shape[1]):
            d_bin = np.digitize(distance[:, chaser], distance_edges[1:-1], right=False)
            b_bin = np.digitize(bearing[:, chaser], bearing_edges[1:-1], right=False)
            for eye in range(2):
                usable = epoch & valid[:, eye, chaser] & accessible[:, eye, chaser]
                for d_pos in range(distance_edges.size - 1):
                    for b_pos in range(bearing_edges.size - 1):
                        mask = usable & (d_bin == d_pos) & (b_bin == b_pos)
                        n = int(np.sum(mask))
                        count[e_pos, chaser, eye, d_pos, b_pos] = n
                        if n:
                            mean_error[e_pos, chaser, eye, d_pos, b_pos] = float(np.mean(np.abs(gaze_error[mask, eye, chaser])))
                            lock_fraction[e_pos, chaser, eye, d_pos, b_pos] = float(np.mean(lock_on[mask, eye, chaser]))
                            vergence_values = vergence[mask]
                            if np.any(np.isfinite(vergence_values)):
                                mean_vergence[e_pos, chaser, eye, d_pos, b_pos] = float(np.nanmean(vergence_values))
    return {
        "distance_bin_edges_mm": distance_edges.astype(np.float32),
        "bearing_bin_edges_deg": bearing_edges.astype(np.float32),
        "frame_count": count,
        "mean_abs_gaze_error_deg": mean_error,
        "lock_on_fraction": lock_fraction,
        "mean_vergence_eye_angle_deg": mean_vergence,
    }


def _lock_events(
    *,
    camera_frame_id: np.ndarray,
    epoch_ids_by_frame: np.ndarray,
    epoch_window_ids: np.ndarray,
    chaser_indices: np.ndarray,
    behavior_labels: Sequence[str],
    distance: np.ndarray,
    bearing: np.ndarray,
    gaze_error: np.ndarray,
    vergence: np.ndarray,
    lock_on: np.ndarray,
    fps: float,
    min_duration_s: float,
) -> tuple[LockEvent, ...]:
    events: list[LockEvent] = []
    min_frames = max(1, int(math.ceil(float(min_duration_s) * float(fps))))
    for window_id in epoch_window_ids.tolist():
        epoch = epoch_ids_by_frame == int(window_id)
        for chaser_pos, chaser_index in enumerate(chaser_indices.tolist()):
            behavior = behavior_labels[chaser_pos] if chaser_pos < len(behavior_labels) else "unknown"
            for eye in range(2):
                mask = epoch & lock_on[:, eye, chaser_pos]
                for start, end in sustained_true_runs(mask, min_frames=min_frames):
                    row_slice = slice(start, end + 1)
                    events.append(
                        LockEvent(
                            start_frame=int(camera_frame_id[start]),
                            end_frame=int(camera_frame_id[end]),
                            duration_s=float((end - start + 1) / fps),
                            epoch_window_id=int(window_id),
                            eye_index=eye,
                            chaser_index=int(chaser_index),
                            behavior_class=str(behavior),
                            median_distance_mm=float(np.nanmedian(distance[row_slice, chaser_pos])),
                            median_bearing_deg=float(np.nanmedian(bearing[row_slice, chaser_pos])),
                            median_gaze_error_deg=float(np.nanmedian(gaze_error[row_slice, eye, chaser_pos])),
                            mean_vergence_eye_angle_deg=float(np.nanmean(vergence[row_slice])),
                        )
                    )
    return tuple(events)


def build_chaser_gaze_tracking_result(
    zarr_path: Path,
    *,
    chaser_distance_run: str = "latest",
    egocentric_component: str = "latest",
    egocentric_dependency_handle: Mapping[str, Any] | None = None,
    legacy_egocentric_component_compatibility: bool = False,
    eye_angle_run: str = "latest",
    component_name: str = DEFAULT_COMPONENT_NAME,
    eye_range_quantiles: tuple[float, float] = DEFAULT_EYE_RANGE_QUANTILES,
    lock_threshold_deg: float = DEFAULT_LOCK_THRESHOLD_DEG,
    lock_min_duration_s: float = DEFAULT_LOCK_MIN_DURATION_S,
    max_tracking_distance_mm: float = DEFAULT_MAX_TRACKING_DISTANCE_MM,
    max_lag_s: float = DEFAULT_MAX_LAG_S,
    virtual_rotations_deg: Sequence[float] = DEFAULT_VIRTUAL_ROTATIONS_DEG,
    min_virtual_separation_mm: float = DEFAULT_MIN_VIRTUAL_SEPARATION_MM,
    distance_bin_edges_mm: Sequence[float] = DEFAULT_DISTANCE_BIN_EDGES_MM,
    bearing_bin_edges_deg: Sequence[float] = DEFAULT_BEARING_BIN_EDGES_DEG,
) -> ChaserGazeTrackingResult:
    root = open_zarr_root(zarr_path, mode="r")
    distance_snapshot, distance_run_name, distance_run_path = (
        _resolve_chaser_distance_run(root, chaser_distance_run)
    )
    # Gaze-lock outputs and event tables assign behavior roles to observations.
    # Those protocol-derived role surfaces are not yet sealed by the canonical
    # chaser-distance publication, so this role-dependent consumer must stop at
    # the typed boundary instead of falling back to protocol_json or raw attrs.
    distance_snapshot.require_behavior_authority()
    distance_run_group = root[distance_run_path]
    ego_group, ego_name, ego_path, ego_manifest_sha256 = (
        _resolve_egocentric_component(
            root,
            snapshot=distance_snapshot,
            run_group=distance_run_group,
            requested=egocentric_component,
            dependency_handle=egocentric_dependency_handle,
            legacy_compatibility=legacy_egocentric_component_compatibility,
        )
    )
    eye_group, eye_name, eye_path = _resolve_eye_run(root, eye_angle_run)
    _metadata_eye_convention(eye_group)

    frames = distance_run_group["frames"]
    camera_frame_id = np.asarray(frames["camera_frame_id"][:], dtype=np.int64)
    epoch_ids = np.asarray(frames["stimulus_epoch_window_id"][:], dtype=np.int32)
    total_frames = int(camera_frame_id.size)
    if camera_frame_id.size and (int(np.min(camera_frame_id)) < 0):
        raise ValueError("Chaser-distance camera_frame_id contains negative values.")

    eye_fields = _packed_columns(
        eye_group,
        "frame_angles",
        "angle_channel_index",
        (
            "left_gaze_signed_deg_smoothed",
            "right_gaze_signed_deg_smoothed",
            "vergence_eye_angle_deg_smoothed",
        ),
    )
    eye_qa = _packed_columns(
        eye_group,
        "frame_qa",
        "qa_channel_index",
        ("valid_frame", "major_axis_marginal"),
    )
    eye_frame_count = int(next(iter(eye_fields.values())).shape[0])
    eye_row_index, eye_row_present = _dense_frame_row_lookup(
        eye_frame_count,
        camera_frame_id,
    )
    gaze = np.full((total_frames, 2), np.nan, dtype=np.float32)
    vergence = np.full(total_frames, np.nan, dtype=np.float32)
    frame_eye_valid = np.zeros(total_frames, dtype=bool)
    marginal = np.zeros(total_frames, dtype=bool)
    if np.any(eye_row_present):
        source_rows = eye_row_index[eye_row_present]
        gaze[eye_row_present, 0] = eye_fields[
            "left_gaze_signed_deg_smoothed"
        ][source_rows]
        gaze[eye_row_present, 1] = eye_fields[
            "right_gaze_signed_deg_smoothed"
        ][source_rows]
        vergence[eye_row_present] = eye_fields[
            "vergence_eye_angle_deg_smoothed"
        ][source_rows]
        frame_eye_valid[eye_row_present] = np.asarray(
            eye_qa["valid_frame"][source_rows], dtype=bool
        )
        marginal[eye_row_present] = np.asarray(
            eye_qa["major_axis_marginal"][source_rows], dtype=bool
        )
    eye_valid = np.isfinite(gaze) & frame_eye_valid[:, None] & ~marginal[:, None]

    ego_frames = ego_group["frames"]
    ego_frame_id = np.asarray(ego_frames["camera_frame_id"][:], dtype=np.int64)
    if not np.array_equal(ego_frame_id, camera_frame_id):
        raise ValueError("Egocentric-bearing and chaser-distance camera frame axes differ.")
    fish_heading = np.asarray(ego_frames["fish_heading_deg"][:], dtype=np.float64)
    fish_heading_valid = np.asarray(ego_frames["fish_heading_valid"][:], dtype=bool)
    heading_alignment = _heading_alignment_gate(
        eye_group=eye_group,
        camera_frame_id=camera_frame_id,
        track_heading_deg=fish_heading,
        track_heading_valid=fish_heading_valid,
    )
    ego_per_chaser = ego_group["per_chaser"]
    bearing = np.asarray(ego_per_chaser["bearing_deg"][:], dtype=np.float32)
    distance = np.asarray(ego_per_chaser["distance_mm"][:], dtype=np.float32)
    object_valid = np.asarray(ego_per_chaser["valid"][:], dtype=bool)

    chasers = distance_run_group["chasers"]
    chaser_indices = np.asarray(chasers["chaser_index"][:], dtype=np.int64)
    if "behavior_class_label_bytes" in chasers:
        behavior_labels = tuple(_decode_text_column(np.asarray(chasers["behavior_class_label_bytes"][:])))
        behavior_label_source = f"{distance_run_path}/chasers/behavior_class_label_bytes"
    elif "behavior_class_label_bytes" in ego_per_chaser:
        # Role labels were added to the egocentric component before every
        # historical distance run was regenerated.  They remain authoritative
        # protocol-derived labels, not identity guesses.
        behavior_labels = tuple(
            _decode_text_column(np.asarray(ego_per_chaser["behavior_class_label_bytes"][:]))
        )
        behavior_label_source = (
            f"{distance_run_path}/egocentric_bearing/{ego_name}/"
            "per_chaser/behavior_class_label_bytes"
        )
    else:
        behavior_labels = _load_configured_chaser_behavior_labels(
            root,
            distance_run_group,
            chaser_indices,
        )
        behavior_label_source = "source_stimulus_protocol_json_fallback"
    if len(behavior_labels) != int(chaser_indices.size):
        raise ValueError("Chaser behavior-role labels do not match the variable-length chaser axis.")
    if any(str(label).strip().lower() in {"", "unknown"} for label in behavior_labels):
        raise ValueError(
            "Chaser roles are absent from persisted components and cannot be resolved "
            "from the source stimulus protocol; refusing identity-based role guesses."
        )
    if bearing.shape != (total_frames, int(chaser_indices.size)):
        raise ValueError("Egocentric bearing shape does not match frame/chaser axes.")

    fps = _safe_float(distance_run_group.attrs.get("fps"), math.nan)
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("Chaser-distance run lacks a positive fps.")
    ranges = _empirical_eye_ranges(gaze, eye_valid, eye_range_quantiles)
    valid = eye_valid[:, :, None] & object_valid[:, None, :] & fish_heading_valid[:, None, None]
    valid &= np.isfinite(distance[:, None, :]) & (distance[:, None, :] <= float(max_tracking_distance_mm))
    accessible = np.zeros(valid.shape, dtype=bool)
    for eye in range(2):
        low, high = ranges[eye]
        if np.isfinite(low) and np.isfinite(high):
            accessible[:, eye, :] = (bearing >= low) & (bearing <= high)
    gaze_error = wrap_degrees_signed(gaze[:, :, None] - bearing[:, None, :]).astype(np.float32)
    lock_on = valid & accessible & (np.abs(gaze_error) <= float(lock_threshold_deg))

    epoch_window_ids, epoch_labels, epoch_start, epoch_end = _epoch_table(distance_run_group)
    summary = _fill_summary(
        epoch_ids_by_frame=epoch_ids,
        epoch_window_ids=epoch_window_ids,
        gaze=gaze,
        vergence=vergence,
        bearing=bearing,
        distance=distance,
        valid=valid,
        accessible=accessible,
        lock_on=lock_on,
        fps=float(fps),
        max_lag_s=float(max_lag_s),
    )

    positions = distance_run_group["positions"]
    fish_xy = np.asarray(positions["fish_centroid_arena_xy"][:], dtype=np.float64)
    fish_valid = np.asarray(positions["fish_valid"][:], dtype=bool)
    chaser_xy = np.asarray(positions["chaser_arena_xy"][:], dtype=np.float64)
    pixels_per_mm = _safe_float(distance_run_group.attrs.get("pixels_per_mm_projector"), math.nan)
    geometry = _resolve_arena_geometry(root, distance_run_group, pixels_per_mm=pixels_per_mm)
    if geometry.shape != "circle" or geometry.center_x_px is None or geometry.center_y_px is None:
        raise ValueError("Rotated gaze controls require a resolved circular arena centre.")
    virtual_refs, virtual_xy = _virtual_positions(
        chaser_xy=chaser_xy,
        chaser_indices=chaser_indices,
        center_xy=(float(geometry.center_x_px), float(geometry.center_y_px)),
        rotations_deg=virtual_rotations_deg,
        min_separation_mm=float(min_virtual_separation_mm),
        pixels_per_mm=float(pixels_per_mm),
    )
    if virtual_refs:
        virtual_vector, virtual_bearing, _alignment, _lateral, virtual_valid_base = compute_egocentric_chaser_bearing(
            fish_arena_xy=fish_xy,
            chaser_arena_xy=virtual_xy,
            fish_heading_deg=fish_heading,
            fish_valid=fish_valid,
            fish_heading_valid=fish_heading_valid,
        )
        virtual_distance = np.linalg.norm(virtual_vector, axis=2) / float(pixels_per_mm)
        virtual_valid = eye_valid[:, :, None] & virtual_valid_base[:, None, :]
        virtual_valid &= np.isfinite(virtual_distance[:, None, :]) & (virtual_distance[:, None, :] <= float(max_tracking_distance_mm))
        virtual_accessible = np.zeros(virtual_valid.shape, dtype=bool)
        for eye in range(2):
            low, high = ranges[eye]
            if np.isfinite(low) and np.isfinite(high):
                virtual_accessible[:, eye, :] = (virtual_bearing >= low) & (virtual_bearing <= high)
        virtual_error = wrap_degrees_signed(gaze[:, :, None] - virtual_bearing[:, None, :])
        virtual_lock = virtual_valid & virtual_accessible & (np.abs(virtual_error) <= float(lock_threshold_deg))
        virtual_summary = _fill_summary(
            epoch_ids_by_frame=epoch_ids,
            epoch_window_ids=epoch_window_ids,
            gaze=gaze,
            vergence=vergence,
            bearing=virtual_bearing,
            distance=virtual_distance,
            valid=virtual_valid,
            accessible=virtual_accessible,
            lock_on=virtual_lock,
            fps=float(fps),
            max_lag_s=float(max_lag_s),
        )
    else:
        virtual_summary = _empty_summary(int(epoch_window_ids.size), 0)

    shape = _summary_shape(int(epoch_window_ids.size), int(chaser_indices.size))
    excess = {
        "tracking_gain_excess_vs_virtual": np.full(shape, np.nan, dtype=np.float32),
        "dynamic_zero_lag_gain_excess_vs_virtual": np.full(shape, np.nan, dtype=np.float32),
        "dynamic_best_lag_gain_excess_vs_virtual": np.full(shape, np.nan, dtype=np.float32),
        "lock_on_fraction_excess_vs_virtual": np.full(shape, np.nan, dtype=np.float32),
        "median_abs_error_improvement_vs_virtual_deg": np.full(shape, np.nan, dtype=np.float32),
        "virtual_reference_count": np.zeros((int(chaser_indices.size),), dtype=np.int32),
    }
    for chaser_pos, chaser_index in enumerate(chaser_indices.tolist()):
        ref_positions = [idx for idx, ref in enumerate(virtual_refs) if ref.parent_chaser_index == int(chaser_index)]
        excess["virtual_reference_count"][chaser_pos] = len(ref_positions)
        if not ref_positions:
            continue
        for key, out_key in (
            ("tracking_gain", "tracking_gain_excess_vs_virtual"),
            ("dynamic_zero_lag_gain", "dynamic_zero_lag_gain_excess_vs_virtual"),
            ("dynamic_best_lag_gain", "dynamic_best_lag_gain_excess_vs_virtual"),
            ("lock_on_fraction", "lock_on_fraction_excess_vs_virtual"),
        ):
            virtual_mean = np.nanmean(virtual_summary[key][:, ref_positions, :], axis=1)
            excess[out_key][:, chaser_pos, :] = summary[key][:, chaser_pos, :] - virtual_mean
        virtual_error_mean = np.nanmean(virtual_summary["median_abs_gaze_error_deg"][:, ref_positions, :], axis=1)
        excess["median_abs_error_improvement_vs_virtual_deg"][:, chaser_pos, :] = (
            virtual_error_mean - summary["median_abs_gaze_error_deg"][:, chaser_pos, :]
        )

    distance_edges = np.asarray(distance_bin_edges_mm, dtype=np.float64)
    bearing_edges = np.asarray(bearing_bin_edges_deg, dtype=np.float64)
    if distance_edges.ndim != 1 or distance_edges.size < 2 or np.any(np.diff(distance_edges) <= 0):
        raise ValueError("distance_bin_edges_mm must be strictly increasing.")
    if bearing_edges.ndim != 1 or bearing_edges.size < 2 or np.any(np.diff(bearing_edges) <= 0):
        raise ValueError("bearing_bin_edges_deg must be strictly increasing.")
    binned = _binned_summary(
        epoch_ids_by_frame=epoch_ids,
        epoch_window_ids=epoch_window_ids,
        gaze_error=gaze_error,
        vergence=vergence,
        bearing=bearing,
        distance=distance,
        valid=valid,
        accessible=accessible,
        lock_on=lock_on,
        distance_edges=distance_edges,
        bearing_edges=bearing_edges,
    )
    events = _lock_events(
        camera_frame_id=camera_frame_id,
        epoch_ids_by_frame=epoch_ids,
        epoch_window_ids=epoch_window_ids,
        chaser_indices=chaser_indices,
        behavior_labels=behavior_labels,
        distance=distance,
        bearing=bearing,
        gaze_error=gaze_error,
        vergence=vergence,
        lock_on=lock_on,
        fps=float(fps),
        min_duration_s=float(lock_min_duration_s),
    )
    recording_id = str(distance_run_group.attrs.get("recording_id") or root.attrs.get("recording_id") or zarr_path.stem)
    diagnostics = {
        "eye_angle_source_field": "left/right_gaze_signed_deg_smoothed",
        "object_bearing_source_field": f"{ego_path}/per_chaser/bearing_deg",
        "source_egocentric_bearing_manifest_sha256": ego_manifest_sha256,
        "chaser_behavior_label_source": behavior_label_source,
        "coordinate_frame": "fish_body_frame",
        "zero_definition": "fish_forward",
        "positive_definition": "anatomical_left",
        "world_frame_gaze_prohibited": True,
        "nasal_positive_eye_angles_prohibited_for_object_bearing": True,
        "ellipse_direction_assumption": str(eye_group.attrs.get("gaze_angle_source") or ""),
        "eye_body_vs_track_heading_alignment": heading_alignment,
        "frame_rows_are_descriptive_not_independent": True,
        "cohort_inference_unit": "recording_fish",
        "virtual_control_definition": (
            "each real chaser position rotated about the circular arena centre; preserves "
            "distance from centre and wall proximity while placing the reference at an empty location"
        ),
        "angle_convention": ANGLE_CONVENTION,
        "eye_range_quantiles": [float(value) for value in eye_range_quantiles],
        "eye_dense_frame_alignment": {
            "source_frame_row_count": int(eye_frame_count),
            "target_frame_count": int(total_frames),
            "matched_frame_count": int(np.sum(eye_row_present)),
            "unmatched_frame_count": int(np.sum(~eye_row_present)),
            "join_key": "frame_angles row index == chaser camera_frame_id",
            "note": (
                "roi_angles uses sparse detection rows keyed by support/frame_indices; "
                "frame_angles is the dense frame projection used here"
            ),
        },
        "lock_threshold_deg": float(lock_threshold_deg),
        "lock_min_duration_s": float(lock_min_duration_s),
        "max_tracking_distance_mm": float(max_tracking_distance_mm),
        "max_lag_s": float(max_lag_s),
    }
    return ChaserGazeTrackingResult(
        recording_id=recording_id,
        chaser_distance_run_name=distance_run_name,
        chaser_distance_run_path=distance_run_path,
        egocentric_component_name=ego_name,
        egocentric_component_path=ego_path,
        egocentric_component_manifest_sha256=ego_manifest_sha256,
        eye_angle_run_name=eye_name,
        eye_angle_run_path=eye_path,
        component_name=component_name,
        fps=float(fps),
        camera_frame_id=camera_frame_id,
        stimulus_epoch_window_id=epoch_ids,
        epoch_window_id=epoch_window_ids,
        epoch_label=epoch_labels,
        epoch_start_frame=epoch_start,
        epoch_end_frame=epoch_end,
        chaser_index=chaser_indices,
        chaser_behavior_class=behavior_labels,
        eye_range_deg=ranges,
        eye_valid=eye_valid,
        major_axis_marginal=marginal,
        gaze_signed_deg=gaze,
        vergence_eye_angle_deg=vergence,
        distance_mm=distance,
        bearing_deg=bearing,
        gaze_error_deg=gaze_error,
        accessible=accessible,
        lock_on=lock_on,
        summary=summary,
        virtual_references=virtual_refs,
        virtual_summary=virtual_summary,
        object_vs_virtual=excess,
        binned_summary=binned,
        lock_events=events,
        diagnostics=diagnostics,
    )


def _event_arrays(events: Sequence[LockEvent]) -> dict[str, np.ndarray]:
    return {
        "start_frame": np.asarray([event.start_frame for event in events], dtype=np.int64),
        "end_frame": np.asarray([event.end_frame for event in events], dtype=np.int64),
        "duration_s": np.asarray([event.duration_s for event in events], dtype=np.float32),
        "epoch_window_id": np.asarray([event.epoch_window_id for event in events], dtype=np.int32),
        "eye_index": np.asarray([event.eye_index for event in events], dtype=np.int8),
        "chaser_index": np.asarray([event.chaser_index for event in events], dtype=np.int32),
        "behavior_class_label_bytes": _bytes_array([event.behavior_class for event in events], width=32),
        "median_distance_mm": np.asarray([event.median_distance_mm for event in events], dtype=np.float32),
        "median_bearing_deg": np.asarray([event.median_bearing_deg for event in events], dtype=np.float32),
        "median_gaze_error_deg": np.asarray([event.median_gaze_error_deg for event in events], dtype=np.float32),
        "mean_vergence_eye_angle_deg": np.asarray([event.mean_vergence_eye_angle_deg for event in events], dtype=np.float32),
    }


def _source_refs(result: ChaserGazeTrackingResult) -> dict[str, Any]:
    return {
        "chaser_distance_run": result.chaser_distance_run_name,
        "chaser_distance_path": result.chaser_distance_run_path,
        "egocentric_component": result.egocentric_component_name,
        "egocentric_component_path": result.egocentric_component_path,
        "egocentric_component_manifest_sha256": (
            result.egocentric_component_manifest_sha256
        ),
        "eye_angle_run": result.eye_angle_run_name,
        "eye_angle_path": result.eye_angle_run_path,
    }


def render_chaser_gaze_tracking_summary_png(
    result: ChaserGazeTrackingResult,
    *,
    dpi: int = 150,
) -> bytes:
    """Render recording-level real-vs-virtual metrics and a lock-event raster."""

    eye_colors = ("#dc2626", "#2563eb")
    epoch_count = int(result.epoch_window_id.size)
    chaser_count = int(result.chaser_index.size)
    labels = [
        f"{result.epoch_label[e]}\n{result.chaser_behavior_class[c]} (ch{int(result.chaser_index[c])})"
        for e in range(epoch_count)
        for c in range(chaser_count)
    ]
    x = np.arange(len(labels), dtype=np.float64)
    width = 0.36
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True)
    panels = (
        (
            "tracking_gain_excess_vs_virtual",
            "Static tracking gain: real − rotated virtual",
            "gain excess",
        ),
        (
            "dynamic_zero_lag_gain_excess_vs_virtual",
            "Dynamic compensation at zero lag: real − virtual",
            "Δgaze/Δbearing gain excess",
        ),
        (
            "lock_on_fraction_excess_vs_virtual",
            "Sustained lock-on occupancy: real − virtual",
            "fraction excess",
        ),
    )
    for ax, (field, title, ylabel) in zip(axes.flat[:3], panels):
        values = np.asarray(result.object_vs_virtual[field], dtype=np.float64)
        for eye in range(2):
            flattened = values[:, :, eye].reshape(-1)
            ax.bar(
                x + (eye - 0.5) * width,
                flattened,
                width=width,
                color=eye_colors[eye],
                alpha=0.85,
                label=f"{EYE_LABELS[eye]} eye",
            )
        ax.axhline(0.0, color="#111827", linewidth=1.0)
        ax.set_xticks(x, labels, rotation=35, ha="right", fontsize=8)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.2)
        ax.legend(frameon=False, fontsize=8)

    raster = axes.flat[3]
    raster_labels: list[str] = []
    raster_y: dict[tuple[int, int], int] = {}
    for chaser_pos, chaser_index in enumerate(result.chaser_index.tolist()):
        role = result.chaser_behavior_class[chaser_pos]
        for eye in range(2):
            raster_y[(int(chaser_index), eye)] = len(raster_labels)
            raster_labels.append(f"{role} ch{int(chaser_index)} · {EYE_LABELS[eye]}")
    if result.camera_frame_id.size:
        first_frame = int(result.camera_frame_id[0])
        for event in result.lock_events:
            y = raster_y.get((event.chaser_index, event.eye_index))
            if y is None:
                continue
            start_min = (event.start_frame - first_frame) / result.fps / 60.0
            end_min = (event.end_frame - first_frame + 1) / result.fps / 60.0
            raster.hlines(
                y,
                start_min,
                end_min,
                color=eye_colors[event.eye_index],
                linewidth=2.2,
                alpha=0.8,
            )
        for start, end, label in zip(
            result.epoch_start_frame.tolist(),
            result.epoch_end_frame.tolist(),
            result.epoch_label,
        ):
            start_min = (int(start) - first_frame) / result.fps / 60.0
            end_min = (int(end) - first_frame + 1) / result.fps / 60.0
            raster.axvspan(start_min, end_min, color="#94a3b8", alpha=0.06)
            raster.text(
                0.5 * (start_min + end_min),
                len(raster_labels) - 0.25,
                str(label),
                ha="center",
                va="bottom",
                fontsize=8,
                color="#475569",
            )
    raster.set_yticks(np.arange(len(raster_labels)), raster_labels, fontsize=8)
    raster.set_xlabel("recording time (min)")
    lock_threshold = float(result.diagnostics.get("lock_threshold_deg", math.nan))
    minimum_duration = float(result.diagnostics.get("lock_min_duration_s", math.nan))
    raster.set_title(
        f"Sustained |gaze error| ≤ {lock_threshold:g}° for ≥ {minimum_duration:g} s "
        f"(n={len(result.lock_events)})"
    )
    raster.grid(axis="x", alpha=0.2)

    fig.suptitle(
        f"Chaser gaze tracking · {result.recording_id}\n"
        "Body-frame gaze vs body-frame bearing; recording-level descriptive summary; rotated wall controls",
        fontsize=14,
    )
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)
    return buffer.getvalue()


@sealed_chaser_component_writer(
    component_family=COMPONENT_PARENT_NAME,
    semantic_schema_id=SCHEMA_ID,
    semantic_schema_version=SCHEMA_VERSION,
    method_id=METHOD,
    method_version=METHOD_VERSION,
)
def write_chaser_gaze_tracking_component(
    zarr_path: Path,
    result: ChaserGazeTrackingResult,
    *,
    overwrite: bool = False,
    write_png: bool = True,
    _chaser_component_staging_capability: object | None = None,
) -> str:
    require_chaser_component_staging_capability(
        _chaser_component_staging_capability
    )
    root = open_zarr_root(zarr_path, mode="a")
    distance_run = root[result.chaser_distance_run_path]
    parent = distance_run.require_group(COMPONENT_PARENT_NAME)
    if result.component_name in parent:
        raise RuntimeError(
            "Private chaser component staging archive contains a same-name child."
        )
    component = parent.create_group(result.component_name)

    frames = component.require_group("frames")
    for name, values in (
        ("camera_frame_id", result.camera_frame_id),
        ("stimulus_epoch_window_id", result.stimulus_epoch_window_id),
        ("eye_valid", result.eye_valid),
        ("major_axis_marginal", result.major_axis_marginal),
        ("gaze_signed_deg", result.gaze_signed_deg),
        ("vergence_eye_angle_deg", result.vergence_eye_angle_deg),
    ):
        _write_array(frames, name, values)
    frames.attrs.update(
        {
            "row_axis": "camera_frames",
            "eye_axis": list(EYE_LABELS),
            "gaze_angle_convention": "fish_body_frame; 0=forward; positive=anatomical_left",
            "eye_angle_source": "smoothed framewise gaze; convergence remains nasal-positive",
        }
    )

    objects = component.require_group("objects")
    _write_array(objects, "chaser_index", result.chaser_index)
    _write_array(objects, "behavior_class_label_bytes", _bytes_array(result.chaser_behavior_class, width=32))
    _write_array(objects, "eye_range_deg", result.eye_range_deg)
    _write_array(objects, "distance_mm", result.distance_mm)
    _write_array(objects, "bearing_deg", result.bearing_deg)
    _write_array(objects, "gaze_error_deg", result.gaze_error_deg)
    _write_array(objects, "accessible", result.accessible)
    _write_array(objects, "lock_on", result.lock_on)
    objects.attrs.update(
        {
            "eye_axis": list(EYE_LABELS),
            "chaser_axis": "variable_length_chaser_rows",
            "gaze_error_definition": "wrap(gaze_signed_deg - chaser_bearing_deg)",
            "accessibility_definition": "chaser bearing inside the recording/eye empirical gaze quantile range",
        }
    )

    epochs = component.require_group("epochs")
    _write_array(epochs, "window_id", result.epoch_window_id)
    _write_array(epochs, "label_bytes", _bytes_array(result.epoch_label, width=64))
    _write_array(epochs, "start_frame", result.epoch_start_frame)
    _write_array(epochs, "end_frame", result.epoch_end_frame)

    summary = component.require_group("recording_summary")
    for name, values in result.summary.items():
        _write_array(summary, name, values)
    summary.attrs.update(
        {
            "axis_order": ["epoch", "chaser", "eye"],
            "eye_axis": list(EYE_LABELS),
            "tracking_gain_definition": "OLS gaze_signed_deg ~ object_bearing_deg within the empirical eye-accessibility range",
            "dynamic_zero_lag_definition": "OLS wrapped delta_gaze(t) ~ wrapped delta_bearing(t)",
            "dynamic_best_lag_definition": (
                "maximum positive correlation over causal lags only; lag >= 0 means eye follows object bearing; "
                "descriptive and interpreted against rotated virtual controls"
            ),
            "inference_unit": "recording_fish",
            "frame_rows_are_not_independent_replicates": True,
        }
    )

    virtual = component.require_group("virtual_controls")
    _write_array(virtual, "reference_label_bytes", _bytes_array([ref.label for ref in result.virtual_references], width=64))
    _write_array(virtual, "parent_chaser_index", np.asarray([ref.parent_chaser_index for ref in result.virtual_references], dtype=np.int32))
    _write_array(virtual, "rotation_deg", np.asarray([ref.rotation_deg for ref in result.virtual_references], dtype=np.float32))
    for name, values in result.virtual_summary.items():
        _write_array(virtual, name, values)
    virtual.attrs.update(
        {
            "summary_axis_order": ["epoch", "virtual_reference", "eye"],
            "eye_axis": list(EYE_LABELS),
            "definition": result.diagnostics["virtual_control_definition"],
        }
    )

    excess = component.require_group("object_vs_virtual")
    for name, values in result.object_vs_virtual.items():
        _write_array(excess, name, values)
    excess.attrs.update(
        {
            "summary_axis_order": ["epoch", "chaser", "eye"],
            "gain_and_lock_excess": "real minus mean rotated virtual controls",
            "error_improvement": "mean virtual median absolute error minus real median absolute error; positive favors real object",
        }
    )

    binned = component.require_group("distance_bearing_summary")
    for name, values in result.binned_summary.items():
        _write_array(binned, name, values)
    binned.attrs.update(
        {
            "summary_axis_order": ["epoch", "chaser", "eye", "distance_bin", "bearing_bin"],
            "eye_axis": list(EYE_LABELS),
        }
    )

    events = component.require_group("lock_on_events")
    for name, values in _event_arrays(result.lock_events).items():
        _write_array(events, name, values)
    events.attrs.update(
        {
            "row_axis": "sustained_lock_on_events",
            "eye_axis_vocabulary": {"0": "left", "1": "right"},
        }
    )

    git = get_git_info(Path(__file__).resolve().parents[3])
    source_refs = _source_refs(result)
    attrs = {
        "schema_id": SCHEMA_ID,
        "schema_version": SCHEMA_VERSION,
        "method": METHOD,
        "method_version": METHOD_VERSION,
        "status": "writing",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "recording_id": result.recording_id,
        "component_name": result.component_name,
        "source_refs": source_refs,
        "diagnostics": dict(result.diagnostics),
        "summary": {
            "frame_count": int(result.camera_frame_id.size),
            "chaser_count": int(result.chaser_index.size),
            "virtual_reference_count": len(result.virtual_references),
            "lock_on_event_count": len(result.lock_events),
            "eye_range_deg": result.eye_range_deg.tolist(),
        },
        "git_commit": git.get("commit_hash"),
        "git_branch": git.get("branch"),
        "git_dirty": git.get("is_dirty"),
    }
    component.attrs.update(json_attr_safe(attrs))
    lineage = build_run_lineage_payload(
        run_family=f"{result.chaser_distance_run_path}/{COMPONENT_PARENT_NAME}",
        analysis_schema={
            "schema_id": SCHEMA_ID,
            "schema_version": SCHEMA_VERSION,
            "row_axis": "fish_recording",
        },
        method=METHOD,
        method_version=METHOD_VERSION,
        parameters={
            "fps": result.fps,
            "diagnostics": dict(result.diagnostics),
        },
        source_refs=source_refs,
        code={"git_commit": git.get("commit_hash"), "git_dirty": git.get("is_dirty")},
    )
    write_run_lineage_attrs(
        component,
        lineage,
        fingerprint_status="best_effort",
        overwrite=True,
    )
    if write_png:
        component_path = (
            f"{result.chaser_distance_run_path}/{COMPONENT_PARENT_NAME}/{result.component_name}"
        )
        write_png_visualization_artifact(
            component,
            SUMMARY_PNG_ARTIFACT_NAME,
            render_chaser_gaze_tracking_summary_png(result),
            description=(
                "Recording-level body-frame gaze tracking, real-minus-rotated-virtual controls, "
                "and sustained lock-on event raster."
            ),
            created_by=SUMMARY_RENDERER,
            visualization_contract_id=SUMMARY_VISUALIZATION_CONTRACT_ID,
            renderer=SUMMARY_RENDERER,
            renderer_version=SUMMARY_RENDERER_VERSION,
            role="analysis_summary",
            source_paths={
                "recording_summary": f"{component_path}/recording_summary",
                "object_vs_virtual": f"{component_path}/object_vs_virtual",
                "lock_on_events": f"{component_path}/lock_on_events",
            },
            source_runs=source_refs,
            parameters={"method": METHOD, "method_version": METHOD_VERSION},
            overwrite=True,
        )
    component.attrs["status"] = "complete"
    component.attrs["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    return f"{result.chaser_distance_run_path}/{COMPONENT_PARENT_NAME}/{result.component_name}"


def _parse_pair(value: str, *, name: str) -> tuple[float, float]:
    fields = [field.strip() for field in str(value).split(",") if field.strip()]
    if len(fields) != 2:
        raise argparse.ArgumentTypeError(f"{name} must contain exactly two comma-separated values.")
    return float(fields[0]), float(fields[1])


def _parse_float_list(value: str) -> tuple[float, ...]:
    return tuple(float(field.strip()) for field in str(value).split(",") if field.strip())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute body-frame chaser gaze tracking with rotated virtual-object controls."
    )
    parser.add_argument("zarr_path", type=Path)
    parser.add_argument("--chaser-distance-run", default="latest")
    parser.add_argument("--egocentric-component", default="latest")
    parser.add_argument(
        "--egocentric-dependency-handle-json",
        type=Path,
        help=(
            "Strict JSON dependency handle for one immutable selector-ineligible "
            "egocentric-bearing candidate."
        ),
    )
    parser.add_argument(
        "--legacy-egocentric-component-compatibility",
        action="store_true",
        help=(
            "Explicitly permit historical egocentric component name/latest discovery; "
            "exact handles remain the maintained default."
        ),
    )
    parser.add_argument("--eye-angle-run", default="latest")
    parser.add_argument("--component-name", default=DEFAULT_COMPONENT_NAME)
    parser.add_argument("--eye-range-quantiles", default="0.01,0.99")
    parser.add_argument("--lock-threshold-deg", type=float, default=DEFAULT_LOCK_THRESHOLD_DEG)
    parser.add_argument("--lock-min-duration-s", type=float, default=DEFAULT_LOCK_MIN_DURATION_S)
    parser.add_argument("--max-tracking-distance-mm", type=float, default=DEFAULT_MAX_TRACKING_DISTANCE_MM)
    parser.add_argument("--max-lag-s", type=float, default=DEFAULT_MAX_LAG_S)
    parser.add_argument("--virtual-rotations-deg", default=",".join(f"{value:g}" for value in DEFAULT_VIRTUAL_ROTATIONS_DEG))
    parser.add_argument("--min-virtual-separation-mm", type=float, default=DEFAULT_MIN_VIRTUAL_SEPARATION_MM)
    parser.add_argument("--distance-bin-edges-mm", default=",".join(str(value) for value in DEFAULT_DISTANCE_BIN_EDGES_MM))
    parser.add_argument("--bearing-bin-edges-deg", default=",".join(str(value) for value in DEFAULT_BEARING_BIN_EDGES_DEG))
    parser.add_argument("--apply", action="store_true", help="Write the immutable component; default is read-only build/report.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-png", action="store_true", help="Skip the bounded recording-level summary PNG.")
    parser.add_argument(
        "--preview-png",
        type=Path,
        help="Optional read-only-mode summary PNG written outside the source Zarr.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    egocentric_dependency_handle = (
        load_chaser_component_handle_json(args.egocentric_dependency_handle_json)
        if args.egocentric_dependency_handle_json is not None
        else None
    )
    result = build_chaser_gaze_tracking_result(
        args.zarr_path,
        chaser_distance_run=args.chaser_distance_run,
        egocentric_component=args.egocentric_component,
        egocentric_dependency_handle=egocentric_dependency_handle,
        legacy_egocentric_component_compatibility=bool(
            args.legacy_egocentric_component_compatibility
        ),
        eye_angle_run=args.eye_angle_run,
        component_name=args.component_name,
        eye_range_quantiles=_parse_pair(args.eye_range_quantiles, name="eye-range-quantiles"),
        lock_threshold_deg=args.lock_threshold_deg,
        lock_min_duration_s=args.lock_min_duration_s,
        max_tracking_distance_mm=args.max_tracking_distance_mm,
        max_lag_s=args.max_lag_s,
        virtual_rotations_deg=_parse_float_list(args.virtual_rotations_deg),
        min_virtual_separation_mm=args.min_virtual_separation_mm,
        distance_bin_edges_mm=_parse_float_list(args.distance_bin_edges_mm),
        bearing_bin_edges_deg=_parse_float_list(args.bearing_bin_edges_deg),
    )
    output_path = None
    if args.preview_png is not None:
        args.preview_png.parent.mkdir(parents=True, exist_ok=True)
        args.preview_png.write_bytes(render_chaser_gaze_tracking_summary_png(result))
    if args.apply:
        output_path = write_chaser_gaze_tracking_component(
            args.zarr_path,
            result,
            overwrite=args.overwrite,
            write_png=not args.no_png,
        )
    print(
        json.dumps(
            {
                "recording_id": result.recording_id,
                "component_name": result.component_name,
                "source_eye_angle_run": result.eye_angle_run_name,
                "source_chaser_distance_run": result.chaser_distance_run_name,
                "source_egocentric_component": result.egocentric_component_name,
                "frame_count": int(result.camera_frame_id.size),
                "chaser_count": int(result.chaser_index.size),
                "chaser_roles": list(result.chaser_behavior_class),
                "eye_range_deg": result.eye_range_deg.tolist(),
                "virtual_reference_count": len(result.virtual_references),
                "lock_on_event_count": len(result.lock_events),
                "recording_summary": {
                    "tracking_gain": result.summary["tracking_gain"].tolist(),
                    "bearing_span_deg": result.summary["bearing_span_deg"].tolist(),
                    "accessible_frame_count": result.summary["accessible_frame_count"].tolist(),
                    "dynamic_zero_lag_gain": result.summary["dynamic_zero_lag_gain"].tolist(),
                    "dynamic_best_lag_gain": result.summary["dynamic_best_lag_gain"].tolist(),
                    "dynamic_best_lag_seconds": result.summary["dynamic_best_lag_seconds"].tolist(),
                    "lock_on_fraction": result.summary["lock_on_fraction"].tolist(),
                    "median_abs_gaze_error_deg": result.summary["median_abs_gaze_error_deg"].tolist(),
                },
                "object_vs_virtual": {
                    name: values.tolist()
                    for name, values in result.object_vs_virtual.items()
                },
                "heading_alignment": result.diagnostics[
                    "eye_body_vs_track_heading_alignment"
                ],
                "written_path": output_path,
                "preview_png": str(args.preview_png) if args.preview_png is not None else None,
                "mode": "apply" if args.apply else "read_only",
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
