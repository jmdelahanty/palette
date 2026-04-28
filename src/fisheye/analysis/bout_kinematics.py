"""Compute per-bout heading metrics from track kinematics and swim-bout candidates."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import zarr

from fisheye.analysis.chaser_state_interpolator import (
    load_structured_dataset,
    write_columnar_dataset,
)
from fisheye.analysis.detect_bouts_multi_level import normalize_speed_level
from fisheye.shared.plot_artifacts import (
    write_interactive_plot_spec_artifact,
    write_png_visualization_artifact,
)
from fisheye.shared.stage_provenance import build_stage_provenance, write_stage_provenance
from fisheye.utils.system import get_environment_info, get_git_info
from fisheye.utils.zarr_io import open_zarr_root


SCHEMA_ID = "analysis.bout_kinematics_runs"
SCHEMA_VERSION = 5
METHOD = "heading_window_and_within_bout_metrics"
METHOD_VERSION = "bout_kinematics.v5"
BOUT_KINEMATICS_PLOT_SPEC_SCHEMA_ID = "palette.plot_spec.bout_kinematics_summary.v1"
BOUT_KINEMATICS_PLOT_RENDERER = "matplotlib_static_plotly_spec.v1"
BOUT_KINEMATICS_PNG_PREFIX = "bout_kinematics_summary"

HEADING_LEVEL_TO_ARRAY = {
    "heading_smoothed": "smoothed_heading_degrees",
    "heading_raw": "heading_degrees",
}
HEADING_LEVEL_ALIASES = {
    "smoothed": "heading_smoothed",
    "raw": "heading_raw",
    **{level: level for level in HEADING_LEVEL_TO_ARRAY},
}
WITHIN_WINDOWS = ("bout_start_end", "core_start_end")
PRE_POST_MODES = ("fixed_window", "interbout_epoch")


def normalize_heading_level(value: str) -> str:
    """Normalize user-facing heading-level names to stored subgroup names."""

    normalized = HEADING_LEVEL_ALIASES.get(str(value).strip())
    if normalized not in HEADING_LEVEL_TO_ARRAY:
        expected = ", ".join(sorted(HEADING_LEVEL_ALIASES))
        raise ValueError(f"Unsupported heading level {value!r}; expected one of: {expected}")
    return normalized


def _metrics_dtype() -> np.dtype:
    return np.dtype(
        [
            ("bout_id", "i4"),
            ("source_start_frame", "i8"),
            ("source_end_frame", "i8"),
            ("source_core_start_frame", "i8"),
            ("source_core_end_frame", "i8"),
            ("source_core_start_time_s_interpolated", "f8"),
            ("source_core_end_time_s_interpolated", "f8"),
            ("source_core_duration_s_interpolated", "f8"),
            ("source_core_start_time_interpolated_valid", "?"),
            ("source_core_end_time_interpolated_valid", "?"),
            ("source_peak_frame", "i8"),
            ("source_peak_time_s", "f8"),
            ("source_peak_signal_value_mm_s", "f8"),
            ("source_peak_prominence_mm_s", "f8"),
            ("source_peak_width_s", "f8"),
            ("source_peak_width_height_mm_s", "f8"),
            ("source_peak_left_width_frame_interpolated", "f8"),
            ("source_peak_right_width_frame_interpolated", "f8"),
            ("source_peak_left_width_time_s", "f8"),
            ("source_peak_right_width_time_s", "f8"),
            ("source_peak_boundary_mode_bytes", "S64"),
            ("source_peak_shape_split_policy_bytes", "S64"),
            ("pre_epoch_start_frame", "i8"),
            ("pre_epoch_end_frame", "i8"),
            ("post_epoch_start_frame", "i8"),
            ("post_epoch_end_frame", "i8"),
            ("pre_heading_mean_deg", "f8"),
            ("post_heading_mean_deg", "f8"),
            ("net_delta_heading_deg", "f8"),
            ("abs_net_delta_heading_deg", "f8"),
            ("pre_position_mean_x_mm", "f8"),
            ("pre_position_mean_y_mm", "f8"),
            ("post_position_mean_x_mm", "f8"),
            ("post_position_mean_y_mm", "f8"),
            ("interbout_epoch_displacement_mm", "f8"),
            ("pre_position_mean_x_px", "f8"),
            ("pre_position_mean_y_px", "f8"),
            ("post_position_mean_x_px", "f8"),
            ("post_position_mean_y_px", "f8"),
            ("interbout_epoch_displacement_px", "f8"),
            ("within_heading_range_deg", "f8"),
            ("within_heading_peak_to_peak_deg", "f8"),
            ("within_heading_path_deg", "f8"),
            ("within_heading_std_deg", "f8"),
            ("within_heading_zero_crossings", "i4"),
            ("within_heading_dominant_frequency_hz", "f8"),
            ("within_angular_velocity_mean_deg_s", "f8"),
            ("within_angular_speed_mean_deg_s", "f8"),
            ("within_angular_speed_max_deg_s", "f8"),
            ("within_angular_velocity_std_deg_s", "f8"),
            ("pre_window_valid", "?"),
            ("post_window_valid", "?"),
            ("pre_position_valid", "?"),
            ("post_position_valid", "?"),
            ("within_window_valid", "?"),
            ("within_angular_velocity_valid", "?"),
            ("dominant_frequency_valid", "?"),
            ("pre_window_sample_count", "i4"),
            ("post_window_sample_count", "i4"),
            ("pre_position_sample_count", "i4"),
            ("post_position_sample_count", "i4"),
            ("within_window_sample_count", "i4"),
            ("within_angular_velocity_transition_count", "i4"),
            ("failure_reason_bytes", "S256"),
        ]
    )


def _wrap_degrees(delta: float) -> float:
    if not np.isfinite(delta):
        return float("nan")
    return float((delta + 180.0) % 360.0 - 180.0)


def _circular_mean_deg(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float("nan")
    radians = np.deg2rad(finite)
    return float(np.rad2deg(np.arctan2(np.mean(np.sin(radians)), np.mean(np.cos(radians)))))


def _unwrap_degrees(values: np.ndarray) -> np.ndarray:
    return np.rad2deg(np.unwrap(np.deg2rad(np.asarray(values, dtype=np.float64))))


def _std_unwrapped_deg(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float("nan")
    return float(np.std(_unwrap_degrees(finite)))


def _zero_crossings(
    headings: np.ndarray,
    times: np.ndarray,
    *,
    derivative_threshold_deg_s: float,
) -> int:
    finite_mask = np.isfinite(headings) & np.isfinite(times)
    if np.count_nonzero(finite_mask) < 3:
        return 0
    values = _unwrap_degrees(np.asarray(headings, dtype=np.float64)[finite_mask])
    t = np.asarray(times, dtype=np.float64)[finite_mask]
    dt = np.diff(t)
    valid_dt = dt > 0
    if np.count_nonzero(valid_dt) < 2:
        return 0
    velocity = np.diff(values)[valid_dt] / dt[valid_dt]
    threshold = abs(float(derivative_threshold_deg_s))
    signs = np.sign(velocity)
    if threshold > 0:
        signs[np.abs(velocity) < threshold] = 0
    signs = signs[signs != 0]
    if signs.size < 2:
        return 0
    return int(np.count_nonzero(signs[1:] != signs[:-1]))


def _angular_velocity_steps(
    headings: np.ndarray,
    times: np.ndarray,
    *,
    transition_valid: Optional[np.ndarray] = None,
    sample_valid: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, bool, int, Optional[str]]:
    """Return valid per-transition angular velocity values for a heading window."""

    heading_values = np.asarray(headings, dtype=np.float64)
    time_values = np.asarray(times, dtype=np.float64)
    if heading_values.size < 2 or time_values.size != heading_values.size:
        return np.asarray([], dtype=np.float64), False, 0, "insufficient_angular_velocity_samples"

    dt = np.diff(time_values)
    unwrapped = np.rad2deg(np.unwrap(np.deg2rad(heading_values)))
    delta = np.diff(unwrapped)
    valid = (
        np.isfinite(heading_values[1:])
        & np.isfinite(heading_values[:-1])
        & np.isfinite(dt)
        & (dt > 0)
    )
    if transition_valid is not None:
        transition = np.asarray(transition_valid, dtype=bool)
        if transition.shape[0] == heading_values.shape[0]:
            valid &= transition[1:]
    if sample_valid is not None:
        samples = np.asarray(sample_valid, dtype=bool)
        if samples.shape[0] == heading_values.shape[0]:
            valid &= samples[1:] & samples[:-1]

    transition_count = int(valid.size)
    if transition_count == 0:
        return np.asarray([], dtype=np.float64), False, 0, "insufficient_angular_velocity_samples"
    if not bool(np.all(valid)):
        return np.asarray([], dtype=np.float64), False, transition_count, "heading_transition_contains_gap"

    return delta[valid] / dt[valid], True, transition_count, None


def _dominant_frequency_hz(
    headings: np.ndarray,
    times: np.ndarray,
    *,
    enabled: bool,
    min_samples: int,
    detrend: bool,
) -> tuple[float, bool, Optional[str]]:
    if not enabled:
        return float("nan"), False, "dominant_frequency_disabled"

    finite_mask = np.isfinite(headings) & np.isfinite(times)
    if np.count_nonzero(finite_mask) < int(min_samples):
        return float("nan"), False, "dominant_frequency_insufficient_samples"

    values = _unwrap_degrees(np.asarray(headings, dtype=np.float64)[finite_mask])
    t = np.asarray(times, dtype=np.float64)[finite_mask]
    dt = np.diff(t)
    finite_dt = dt[np.isfinite(dt) & (dt > 0)]
    if finite_dt.size == 0:
        return float("nan"), False, "dominant_frequency_insufficient_samples"

    sample_spacing = float(np.median(finite_dt))
    if sample_spacing <= 0:
        return float("nan"), False, "dominant_frequency_insufficient_samples"

    if detrend:
        values = values - np.linspace(values[0], values[-1], values.size)
    values = values - np.mean(values)
    spectrum = np.abs(np.fft.rfft(values))
    freqs = np.fft.rfftfreq(values.size, d=sample_spacing)
    if spectrum.size <= 1:
        return float("nan"), False, "dominant_frequency_insufficient_samples"
    peak_idx = int(np.argmax(spectrum[1:]) + 1)
    return float(freqs[peak_idx]), True, None


def _artifact_signature(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _png_bytes_from_figure(fig: plt.Figure, *, dpi: int) -> bytes:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)
    return buffer.getvalue()


def _field_or_default(bouts: np.ndarray, field: str, default: int) -> np.ndarray:
    if field in (bouts.dtype.names or ()):
        return np.asarray(bouts[field])
    return np.full(len(bouts), default)


def _float_field_or_nan(bouts: np.ndarray, field: str) -> np.ndarray:
    if field in (bouts.dtype.names or ()):
        return np.asarray(bouts[field], dtype=np.float64)
    return np.full(len(bouts), float("nan"), dtype=np.float64)


def _bool_field_or_false(bouts: np.ndarray, field: str) -> np.ndarray:
    if field in (bouts.dtype.names or ()):
        return np.asarray(bouts[field], dtype=bool)
    return np.zeros(len(bouts), dtype=bool)


def _bytes_field_or_empty(records: Optional[np.ndarray], field: str, count: int) -> np.ndarray:
    output = np.full(count, b"", dtype="S64")
    if records is None or field not in (records.dtype.names or ()):
        return output
    for idx, value in enumerate(records[field]):
        if idx >= count:
            break
        if isinstance(value, bytes):
            output[idx] = value[:64]
        else:
            output[idx] = str(value).encode("utf-8")[:64]
    return output


def _records_align_by_bout_id(bouts: np.ndarray, records: np.ndarray) -> bool:
    bout_names = bouts.dtype.names or ()
    record_names = records.dtype.names or ()
    if "bout_id" not in bout_names or "bout_id" not in record_names:
        return True
    try:
        return bool(np.array_equal(bouts["bout_id"], records["bout_id"]))
    except Exception:
        return False


def _epoch_bounds(frames: np.ndarray, epoch_slice: slice) -> tuple[int, int]:
    if epoch_slice.stop <= epoch_slice.start:
        return -1, -1
    return int(frames[epoch_slice.start]), int(frames[epoch_slice.stop - 1])


def _position_epoch_stats(
    positions: Optional[np.ndarray],
    epoch_slice: slice,
) -> tuple[float, float, int, bool]:
    if positions is None:
        return float("nan"), float("nan"), 0, False
    epoch = np.asarray(positions[epoch_slice], dtype=np.float64)
    if epoch.ndim != 2 or epoch.shape[1] != 2 or epoch.shape[0] == 0:
        return float("nan"), float("nan"), 0, False
    finite_rows = np.isfinite(epoch).all(axis=1)
    finite_count = int(np.count_nonzero(finite_rows))
    valid = finite_count == int(epoch.shape[0])
    if not valid:
        return float("nan"), float("nan"), finite_count, False
    mean = np.mean(epoch, axis=0)
    return float(mean[0]), float(mean[1]), finite_count, True


def _distance_2d(x0: float, y0: float, x1: float, y1: float) -> float:
    values = np.asarray([x0, y0, x1, y1], dtype=np.float64)
    if not np.isfinite(values).all():
        return float("nan")
    return float(np.hypot(float(x1) - float(x0), float(y1) - float(y0)))


def _resolve_track_run(
    root: zarr.Group,
    track_kinematics_run: str,
    *,
    track_scope: str,
) -> tuple[zarr.Group, str, str, str]:
    parent = root.get("analysis/track_kinematics_runs")
    if parent is None:
        raise ValueError("No analysis/track_kinematics_runs group found.")

    spec = str(track_kinematics_run).strip().strip("/")
    parts = spec.split("/")
    if spec.startswith("analysis/track_kinematics_runs/") and len(parts) >= 4:
        scope, run_name = parts[2], parts[3]
    elif len(parts) == 2 and parts[0] in parent:
        scope, run_name = parts
    else:
        scope = track_scope
        if scope not in parent:
            raise ValueError(f"Track kinematics scope {scope!r} not found.")
        run_name = parent[scope].attrs.get("latest") if spec == "latest" else spec

    if not run_name:
        raise ValueError(f"No track kinematics run resolved for {track_kinematics_run!r}.")
    if scope not in parent or run_name not in parent[scope]:
        raise ValueError(f"Track kinematics run {scope}/{run_name} not found.")

    run_path = f"analysis/track_kinematics_runs/{scope}/{run_name}"
    return parent[scope][run_name], str(run_name), run_path, str(scope)


def _resolve_swim_bout_run(
    root: zarr.Group,
    swim_bout_run: str,
    speed_level: str,
) -> tuple[zarr.Group, zarr.Group, str, str, str]:
    parent = root.get("analysis/swim_bout_runs")
    if parent is None:
        raise ValueError("No analysis/swim_bout_runs group found.")

    run_name = str(swim_bout_run).strip()
    if run_name == "latest":
        run_name = parent.attrs.get("latest")
    if not run_name or run_name not in parent:
        raise ValueError(f"Swim-bout run {swim_bout_run!r} not found.")

    level = normalize_speed_level(speed_level)
    run_group = parent[run_name]
    if level not in run_group:
        raise ValueError(f"Speed level {level!r} not found in swim-bout run {run_name!r}.")
    level_path = f"analysis/swim_bout_runs/{run_name}/{level}"
    return parent, run_group, str(run_name), level, level_path


def _build_metrics_for_heading(
    *,
    bouts: np.ndarray,
    peak_events: Optional[np.ndarray],
    frames: np.ndarray,
    times: np.ndarray,
    headings: np.ndarray,
    transition_valid: Optional[np.ndarray],
    sample_valid: Optional[np.ndarray],
    positions_mm: Optional[np.ndarray],
    positions_px: Optional[np.ndarray],
    fps: float,
    pre_post_mode: str,
    pre_window_frames: int,
    post_window_frames: int,
    within_window: str,
    derivative_threshold_deg_s: float,
    dominant_frequency_enabled: bool,
    dominant_frequency_min_samples: int,
    dominant_frequency_detrend: bool,
) -> np.ndarray:
    metrics = np.zeros(len(bouts), dtype=_metrics_dtype())
    if len(metrics) == 0:
        return metrics

    frame_to_index = {int(frame): idx for idx, frame in enumerate(np.asarray(frames, dtype=np.int64))}
    start_frames = np.asarray(bouts["start_frame"], dtype=np.int64)
    end_frames = np.asarray(bouts["end_frame"], dtype=np.int64)
    bout_ids = (
        np.asarray(bouts["bout_id"], dtype=np.int32)
        if "bout_id" in (bouts.dtype.names or ())
        else np.arange(1, len(bouts) + 1, dtype=np.int32)
    )
    core_starts = _field_or_default(bouts, "core_start_frame", -1).astype(np.int64)
    core_ends = _field_or_default(bouts, "core_end_frame", -1).astype(np.int64)
    source_core_start_time_s_interpolated = _float_field_or_nan(
        bouts,
        "core_start_time_s_interpolated",
    )
    source_core_end_time_s_interpolated = _float_field_or_nan(
        bouts,
        "core_end_time_s_interpolated",
    )
    source_core_duration_s_interpolated = _float_field_or_nan(
        bouts,
        "core_duration_s_interpolated",
    )
    source_core_start_time_interpolated_valid = _bool_field_or_false(
        bouts,
        "core_start_time_interpolated_valid",
    )
    source_core_end_time_interpolated_valid = _bool_field_or_false(
        bouts,
        "core_end_time_interpolated_valid",
    )
    aligned_peak_events = (
        peak_events
        if peak_events is not None
        and len(peak_events) == len(bouts)
        and peak_events.dtype.names is not None
        and _records_align_by_bout_id(bouts, peak_events)
        else None
    )
    if aligned_peak_events is None:
        source_peak_frame = np.full(len(bouts), -1, dtype=np.int64)
        source_peak_time_s = np.full(len(bouts), float("nan"), dtype=np.float64)
        source_peak_signal_value_mm_s = np.full(len(bouts), float("nan"), dtype=np.float64)
        source_peak_prominence_mm_s = np.full(len(bouts), float("nan"), dtype=np.float64)
        source_peak_width_s = np.full(len(bouts), float("nan"), dtype=np.float64)
        source_peak_width_height_mm_s = np.full(len(bouts), float("nan"), dtype=np.float64)
        source_peak_left_width_frame_interpolated = np.full(len(bouts), float("nan"), dtype=np.float64)
        source_peak_right_width_frame_interpolated = np.full(len(bouts), float("nan"), dtype=np.float64)
    else:
        source_peak_frame = _field_or_default(aligned_peak_events, "peak_frame", -1).astype(np.int64)
        source_peak_time_s = _float_field_or_nan(aligned_peak_events, "peak_time_s")
        source_peak_signal_value_mm_s = _float_field_or_nan(aligned_peak_events, "peak_signal_value_mm_s")
        source_peak_prominence_mm_s = _float_field_or_nan(aligned_peak_events, "peak_prominence_mm_s")
        source_peak_width_s = _float_field_or_nan(aligned_peak_events, "peak_width_s")
        source_peak_width_height_mm_s = _float_field_or_nan(aligned_peak_events, "peak_width_height_mm_s")
        source_peak_left_width_frame_interpolated = _float_field_or_nan(
            aligned_peak_events,
            "left_width_frame_interpolated",
        )
        source_peak_right_width_frame_interpolated = _float_field_or_nan(
            aligned_peak_events,
            "right_width_frame_interpolated",
        )
    source_peak_left_width_time_s = (
        source_peak_left_width_frame_interpolated / float(fps)
        if fps > 0
        else np.full(len(bouts), float("nan"), dtype=np.float64)
    )
    source_peak_right_width_time_s = (
        source_peak_right_width_frame_interpolated / float(fps)
        if fps > 0
        else np.full(len(bouts), float("nan"), dtype=np.float64)
    )
    source_peak_boundary_mode_bytes = _bytes_field_or_empty(aligned_peak_events, "boundary_mode", len(bouts))
    source_peak_shape_split_policy_bytes = _bytes_field_or_empty(
        aligned_peak_events,
        "shape_split_policy",
        len(bouts),
    )
    sorted_rows = np.argsort(start_frames)
    previous_end_indices = np.full(len(bouts), -1, dtype=np.int64)
    next_start_indices = np.full(len(bouts), -1, dtype=np.int64)
    for order_idx, row_idx in enumerate(sorted_rows):
        if order_idx > 0:
            previous_row = int(sorted_rows[order_idx - 1])
            previous_end_indices[int(row_idx)] = int(frame_to_index.get(int(end_frames[previous_row]), -1))
        if order_idx + 1 < len(sorted_rows):
            next_row = int(sorted_rows[order_idx + 1])
            next_start_indices[int(row_idx)] = int(frame_to_index.get(int(start_frames[next_row]), -1))

    for row_idx, (bout_id, start_frame, end_frame, core_start, core_end) in enumerate(
        zip(bout_ids, start_frames, end_frames, core_starts, core_ends)
    ):
        reasons: list[str] = []
        metrics[row_idx]["bout_id"] = int(bout_id)
        metrics[row_idx]["source_start_frame"] = int(start_frame)
        metrics[row_idx]["source_end_frame"] = int(end_frame)
        metrics[row_idx]["source_core_start_frame"] = int(core_start)
        metrics[row_idx]["source_core_end_frame"] = int(core_end)
        metrics[row_idx]["source_core_start_time_s_interpolated"] = float(
            source_core_start_time_s_interpolated[row_idx]
        )
        metrics[row_idx]["source_core_end_time_s_interpolated"] = float(
            source_core_end_time_s_interpolated[row_idx]
        )
        metrics[row_idx]["source_core_duration_s_interpolated"] = float(
            source_core_duration_s_interpolated[row_idx]
        )
        metrics[row_idx]["source_core_start_time_interpolated_valid"] = bool(
            source_core_start_time_interpolated_valid[row_idx]
        )
        metrics[row_idx]["source_core_end_time_interpolated_valid"] = bool(
            source_core_end_time_interpolated_valid[row_idx]
        )
        metrics[row_idx]["source_peak_frame"] = int(source_peak_frame[row_idx])
        metrics[row_idx]["source_peak_time_s"] = float(source_peak_time_s[row_idx])
        metrics[row_idx]["source_peak_signal_value_mm_s"] = float(source_peak_signal_value_mm_s[row_idx])
        metrics[row_idx]["source_peak_prominence_mm_s"] = float(source_peak_prominence_mm_s[row_idx])
        metrics[row_idx]["source_peak_width_s"] = float(source_peak_width_s[row_idx])
        metrics[row_idx]["source_peak_width_height_mm_s"] = float(source_peak_width_height_mm_s[row_idx])
        metrics[row_idx]["source_peak_left_width_frame_interpolated"] = float(
            source_peak_left_width_frame_interpolated[row_idx]
        )
        metrics[row_idx]["source_peak_right_width_frame_interpolated"] = float(
            source_peak_right_width_frame_interpolated[row_idx]
        )
        metrics[row_idx]["source_peak_left_width_time_s"] = float(source_peak_left_width_time_s[row_idx])
        metrics[row_idx]["source_peak_right_width_time_s"] = float(source_peak_right_width_time_s[row_idx])
        metrics[row_idx]["source_peak_boundary_mode_bytes"] = source_peak_boundary_mode_bytes[row_idx]
        metrics[row_idx]["source_peak_shape_split_policy_bytes"] = source_peak_shape_split_policy_bytes[row_idx]
        metrics[row_idx]["pre_epoch_start_frame"] = -1
        metrics[row_idx]["pre_epoch_end_frame"] = -1
        metrics[row_idx]["post_epoch_start_frame"] = -1
        metrics[row_idx]["post_epoch_end_frame"] = -1
        for field in (
            "pre_heading_mean_deg",
            "post_heading_mean_deg",
            "net_delta_heading_deg",
            "abs_net_delta_heading_deg",
            "pre_position_mean_x_mm",
            "pre_position_mean_y_mm",
            "post_position_mean_x_mm",
            "post_position_mean_y_mm",
            "interbout_epoch_displacement_mm",
            "pre_position_mean_x_px",
            "pre_position_mean_y_px",
            "post_position_mean_x_px",
            "post_position_mean_y_px",
            "interbout_epoch_displacement_px",
            "within_heading_range_deg",
            "within_heading_peak_to_peak_deg",
            "within_heading_path_deg",
            "within_heading_std_deg",
            "within_heading_dominant_frequency_hz",
            "within_angular_velocity_mean_deg_s",
            "within_angular_speed_mean_deg_s",
            "within_angular_speed_max_deg_s",
            "within_angular_velocity_std_deg_s",
        ):
            metrics[row_idx][field] = float("nan")
        metrics[row_idx]["within_heading_zero_crossings"] = 0

        start_idx = frame_to_index.get(int(start_frame))
        end_idx = frame_to_index.get(int(end_frame))
        if start_idx is None or end_idx is None or end_idx < start_idx:
            reasons.append("source_bout_missing")
            metrics[row_idx]["failure_reason_bytes"] = ";".join(reasons).encode("utf-8")
            continue

        if pre_post_mode == "fixed_window":
            pre_slice = slice(max(0, start_idx - pre_window_frames), start_idx)
            post_slice = slice(end_idx + 1, min(len(headings), end_idx + 1 + post_window_frames))
        else:
            previous_end_idx = int(previous_end_indices[row_idx])
            next_start_idx = int(next_start_indices[row_idx])
            if previous_end_idx >= 0 and previous_end_idx < start_idx:
                pre_slice = slice(previous_end_idx + 1, start_idx)
            else:
                pre_slice = slice(start_idx, start_idx)
            if next_start_idx >= 0 and end_idx < next_start_idx:
                post_slice = slice(end_idx + 1, next_start_idx)
            else:
                post_slice = slice(end_idx + 1, end_idx + 1)
        (
            metrics[row_idx]["pre_epoch_start_frame"],
            metrics[row_idx]["pre_epoch_end_frame"],
        ) = _epoch_bounds(frames, pre_slice)
        (
            metrics[row_idx]["post_epoch_start_frame"],
            metrics[row_idx]["post_epoch_end_frame"],
        ) = _epoch_bounds(frames, post_slice)

        pre = np.asarray(headings[pre_slice], dtype=np.float64)
        post = np.asarray(headings[post_slice], dtype=np.float64)

        pre_valid_count = int(np.count_nonzero(np.isfinite(pre)))
        post_valid_count = int(np.count_nonzero(np.isfinite(post)))
        metrics[row_idx]["pre_window_sample_count"] = pre_valid_count
        metrics[row_idx]["post_window_sample_count"] = post_valid_count
        if pre_post_mode == "fixed_window":
            pre_valid = (
                pre.size == pre_window_frames
                and pre_window_frames > 0
                and pre_valid_count == pre_window_frames
            )
            post_valid = (
                post.size == post_window_frames
                and post_window_frames > 0
                and post_valid_count == post_window_frames
            )
        else:
            pre_valid = pre.size > 0 and pre_valid_count == pre.size
            post_valid = post.size > 0 and post_valid_count == post.size
        metrics[row_idx]["pre_window_valid"] = pre_valid
        metrics[row_idx]["post_window_valid"] = post_valid
        if not pre_valid:
            reasons.append(
                "insufficient_pre_window"
                if pre.size == 0 or (pre_post_mode == "fixed_window" and pre.size < pre_window_frames)
                else "heading_contains_gap"
            )
        if not post_valid:
            reasons.append(
                "insufficient_post_window"
                if post.size == 0 or (pre_post_mode == "fixed_window" and post.size < post_window_frames)
                else "heading_contains_gap"
            )

        if pre_valid:
            metrics[row_idx]["pre_heading_mean_deg"] = _circular_mean_deg(pre)
        if post_valid:
            metrics[row_idx]["post_heading_mean_deg"] = _circular_mean_deg(post)
        if pre_valid and post_valid:
            delta = _wrap_degrees(
                float(metrics[row_idx]["post_heading_mean_deg"])
                - float(metrics[row_idx]["pre_heading_mean_deg"])
            )
            metrics[row_idx]["net_delta_heading_deg"] = delta
            metrics[row_idx]["abs_net_delta_heading_deg"] = abs(delta)

        pre_x_mm, pre_y_mm, pre_count_mm, pre_valid_mm = _position_epoch_stats(positions_mm, pre_slice)
        post_x_mm, post_y_mm, post_count_mm, post_valid_mm = _position_epoch_stats(positions_mm, post_slice)
        pre_x_px, pre_y_px, pre_count_px, pre_valid_px = _position_epoch_stats(positions_px, pre_slice)
        post_x_px, post_y_px, post_count_px, post_valid_px = _position_epoch_stats(positions_px, post_slice)
        metrics[row_idx]["pre_position_mean_x_mm"] = pre_x_mm
        metrics[row_idx]["pre_position_mean_y_mm"] = pre_y_mm
        metrics[row_idx]["post_position_mean_x_mm"] = post_x_mm
        metrics[row_idx]["post_position_mean_y_mm"] = post_y_mm
        metrics[row_idx]["pre_position_mean_x_px"] = pre_x_px
        metrics[row_idx]["pre_position_mean_y_px"] = pre_y_px
        metrics[row_idx]["post_position_mean_x_px"] = post_x_px
        metrics[row_idx]["post_position_mean_y_px"] = post_y_px
        metrics[row_idx]["interbout_epoch_displacement_mm"] = _distance_2d(
            pre_x_mm,
            pre_y_mm,
            post_x_mm,
            post_y_mm,
        )
        metrics[row_idx]["interbout_epoch_displacement_px"] = _distance_2d(
            pre_x_px,
            pre_y_px,
            post_x_px,
            post_y_px,
        )
        pre_position_valid = pre_valid_mm or pre_valid_px
        post_position_valid = post_valid_mm or post_valid_px
        metrics[row_idx]["pre_position_valid"] = pre_position_valid
        metrics[row_idx]["post_position_valid"] = post_position_valid
        metrics[row_idx]["pre_position_sample_count"] = max(pre_count_mm, pre_count_px)
        metrics[row_idx]["post_position_sample_count"] = max(post_count_mm, post_count_px)
        has_position_source = positions_mm is not None or positions_px is not None
        if not pre_position_valid:
            reasons.append("missing_position_source" if not has_position_source else "insufficient_pre_position")
        if not post_position_valid:
            reasons.append("missing_position_source" if not has_position_source else "insufficient_post_position")

        within_start_frame = core_start if within_window == "core_start_end" and core_start >= 0 else start_frame
        within_end_frame = core_end if within_window == "core_start_end" and core_end >= 0 else end_frame
        within_start_idx = frame_to_index.get(int(within_start_frame))
        within_end_idx = frame_to_index.get(int(within_end_frame))
        if within_start_idx is None or within_end_idx is None or within_end_idx < within_start_idx:
            reasons.append("source_bout_missing")
            within = np.asarray([], dtype=np.float64)
            within_times = np.asarray([], dtype=np.float64)
            within_transition_valid = None
            within_sample_valid = None
        else:
            within = np.asarray(headings[within_start_idx : within_end_idx + 1], dtype=np.float64)
            within_times = np.asarray(times[within_start_idx : within_end_idx + 1], dtype=np.float64)
            within_transition_valid = (
                np.asarray(transition_valid[within_start_idx : within_end_idx + 1], dtype=bool)
                if transition_valid is not None
                else None
            )
            within_sample_valid = (
                np.asarray(sample_valid[within_start_idx : within_end_idx + 1], dtype=bool)
                if sample_valid is not None
                else None
            )

        within_valid_count = int(np.count_nonzero(np.isfinite(within)))
        metrics[row_idx]["within_window_sample_count"] = within_valid_count
        within_valid = within.size >= 2 and within_valid_count == within.size
        metrics[row_idx]["within_window_valid"] = within_valid
        if not within_valid:
            reasons.append(
                "insufficient_within_bout_samples" if within_valid_count < 2 else "heading_contains_gap"
            )
        if within_valid:
            unwrapped = _unwrap_degrees(within)
            diffs = np.diff(unwrapped)
            heading_range = float(np.max(unwrapped) - np.min(unwrapped))
            metrics[row_idx]["within_heading_range_deg"] = heading_range
            metrics[row_idx]["within_heading_peak_to_peak_deg"] = heading_range
            metrics[row_idx]["within_heading_path_deg"] = float(np.sum(np.abs(diffs)))
            metrics[row_idx]["within_heading_std_deg"] = _std_unwrapped_deg(within)
            metrics[row_idx]["within_heading_zero_crossings"] = _zero_crossings(
                within,
                within_times,
                derivative_threshold_deg_s=derivative_threshold_deg_s,
            )
            frequency, frequency_valid, frequency_reason = _dominant_frequency_hz(
                within,
                within_times,
                enabled=dominant_frequency_enabled,
                min_samples=dominant_frequency_min_samples,
                detrend=dominant_frequency_detrend,
            )
            metrics[row_idx]["within_heading_dominant_frequency_hz"] = frequency
            metrics[row_idx]["dominant_frequency_valid"] = frequency_valid
            if frequency_reason is not None:
                reasons.append(frequency_reason)
            angular_velocity, angular_valid, angular_count, angular_reason = _angular_velocity_steps(
                within,
                within_times,
                transition_valid=within_transition_valid,
                sample_valid=within_sample_valid,
            )
            metrics[row_idx]["within_angular_velocity_transition_count"] = angular_count
            metrics[row_idx]["within_angular_velocity_valid"] = angular_valid
            if angular_valid:
                angular_speed = np.abs(angular_velocity)
                metrics[row_idx]["within_angular_velocity_mean_deg_s"] = float(np.mean(angular_velocity))
                metrics[row_idx]["within_angular_speed_mean_deg_s"] = float(np.mean(angular_speed))
                metrics[row_idx]["within_angular_speed_max_deg_s"] = float(np.max(angular_speed))
                metrics[row_idx]["within_angular_velocity_std_deg_s"] = float(np.std(angular_velocity))
            elif angular_reason is not None:
                reasons.append(angular_reason)

        unique_reasons = list(dict.fromkeys(reasons))
        metrics[row_idx]["failure_reason_bytes"] = (
            ";".join(unique_reasons).encode("utf-8") if unique_reasons else b"ok"
        )

    return metrics


def _safe_metric_values(metrics: np.ndarray, field: str) -> np.ndarray:
    if field not in (metrics.dtype.names or ()):
        return np.asarray([], dtype=np.float64)
    values = np.asarray(metrics[field], dtype=np.float64)
    return values[np.isfinite(values)]


def _plot_bout_kinematics_summary(
    *,
    metrics_by_level: Mapping[str, np.ndarray],
    default_heading_level: str,
    source_speed_level: str,
    bins: int,
) -> bytes:
    default_metrics = metrics_by_level.get(default_heading_level)
    if default_metrics is None:
        default_metrics = next(iter(metrics_by_level.values()))

    fig, axes = plt.subplots(2, 3, figsize=(17, 8))
    axes_flat = axes.ravel()
    metric_specs = [
        ("net_delta_heading_deg", "Net heading change (deg)", (-180.0, 180.0)),
        ("abs_net_delta_heading_deg", "Absolute net heading change (deg)", (0.0, 180.0)),
        ("within_heading_range_deg", "Within-bout heading range (deg)", None),
        ("within_heading_path_deg", "Within-bout heading path (deg)", None),
        ("within_angular_speed_mean_deg_s", "Mean angular speed (deg/s)", None),
        ("within_angular_speed_max_deg_s", "Peak angular speed (deg/s)", None),
    ]
    for ax, (field, label, xlim) in zip(axes_flat, metric_specs):
        for level, metrics in metrics_by_level.items():
            values = _safe_metric_values(metrics, field)
            if values.size == 0:
                continue
            ax.hist(
                values,
                bins=int(bins),
                alpha=0.55,
                label=level.replace("heading_", ""),
            )
        ax.set_title(label)
        ax.set_xlabel(label)
        ax.set_ylabel("Bout count")
        ax.grid(alpha=0.25)
        if xlim is not None:
            ax.set_xlim(*xlim)
        if ax.has_data():
            ax.legend()

    n_bouts = int(len(default_metrics))
    fig.suptitle(
        f"Bout heading kinematics ({source_speed_level}, {n_bouts} bouts)",
        fontsize=14,
    )
    fig.tight_layout()
    return _png_bytes_from_figure(fig, dpi=150)


def _build_bout_kinematics_interactive_spec(
    *,
    run_name: str,
    source_refs: Mapping[str, Any],
    parameters: Mapping[str, Any],
    heading_levels: Sequence[str],
    default_heading_level: str,
    bins: int,
) -> dict[str, Any]:
    source_paths: dict[str, str] = {"run": f"analysis/bout_kinematics_runs/{run_name}"}
    for level in heading_levels:
        base = f"analysis/bout_kinematics_runs/{run_name}/{level}/per_bout_metrics"
        source_paths[f"{level}.per_bout_metrics"] = base
        for field in (
            "bout_id",
            "source_start_frame",
            "source_end_frame",
            "source_core_start_frame",
            "source_core_end_frame",
            "source_core_start_time_s_interpolated",
            "source_core_end_time_s_interpolated",
            "source_core_duration_s_interpolated",
            "source_core_start_time_interpolated_valid",
            "source_core_end_time_interpolated_valid",
            "source_peak_frame",
            "source_peak_time_s",
            "source_peak_signal_value_mm_s",
            "source_peak_prominence_mm_s",
            "source_peak_width_s",
            "source_peak_width_height_mm_s",
            "source_peak_left_width_frame_interpolated",
            "source_peak_right_width_frame_interpolated",
            "source_peak_left_width_time_s",
            "source_peak_right_width_time_s",
            "source_peak_boundary_mode_bytes",
            "source_peak_shape_split_policy_bytes",
            "pre_epoch_start_frame",
            "pre_epoch_end_frame",
            "post_epoch_start_frame",
            "post_epoch_end_frame",
            "net_delta_heading_deg",
            "abs_net_delta_heading_deg",
            "pre_position_mean_x_mm",
            "pre_position_mean_y_mm",
            "post_position_mean_x_mm",
            "post_position_mean_y_mm",
            "interbout_epoch_displacement_mm",
            "pre_position_mean_x_px",
            "pre_position_mean_y_px",
            "post_position_mean_x_px",
            "post_position_mean_y_px",
            "interbout_epoch_displacement_px",
            "within_heading_range_deg",
            "within_heading_peak_to_peak_deg",
            "within_heading_path_deg",
            "within_heading_std_deg",
            "within_heading_zero_crossings",
            "within_angular_velocity_mean_deg_s",
            "within_angular_speed_mean_deg_s",
            "within_angular_speed_max_deg_s",
            "within_angular_velocity_std_deg_s",
            "within_angular_velocity_valid",
            "within_angular_velocity_transition_count",
        ):
            source_paths[f"{level}.{field}"] = f"{base}/{field}"

    return {
        "schema_id": BOUT_KINEMATICS_PLOT_SPEC_SCHEMA_ID,
        "title": "Bout heading kinematics",
        "run_name": run_name,
        "renderer": BOUT_KINEMATICS_PLOT_RENDERER,
        "source_refs": dict(source_refs),
        "source_paths": source_paths,
        "parameters": dict(parameters),
        "default_heading_level": default_heading_level,
        "heading_levels": list(heading_levels),
        "panels": [
            {
                "id": "net_heading_change_histograms",
                "kind": "facet_histogram",
                "heading_levels": list(heading_levels),
                "metrics": [
                    "net_delta_heading_deg",
                    "abs_net_delta_heading_deg",
                ],
                "x_ranges_deg": {
                    "net_delta_heading_deg": [-180.0, 180.0],
                    "abs_net_delta_heading_deg": [0.0, 180.0],
                },
                "bins": int(bins),
            },
            {
                "id": "within_bout_heading_histograms",
                "kind": "facet_histogram",
                "heading_levels": list(heading_levels),
                "metrics": [
                    "within_heading_range_deg",
                    "within_heading_peak_to_peak_deg",
                    "within_heading_path_deg",
                    "within_heading_std_deg",
                ],
                "x_axis_policy": "independent_positive_degrees",
                "bins": int(bins),
            },
            {
                "id": "per_bout_heading_change",
                "kind": "scatter",
                "x": "bout_id",
                "y": "net_delta_heading_deg",
                "heading_levels": list(heading_levels),
            },
            {
                "id": "within_bout_angular_velocity_histograms",
                "kind": "facet_histogram",
                "heading_levels": list(heading_levels),
                "metrics": [
                    "within_angular_velocity_mean_deg_s",
                    "within_angular_speed_mean_deg_s",
                    "within_angular_speed_max_deg_s",
                    "within_angular_velocity_std_deg_s",
                ],
                "x_axis_policy": "independent_degrees_per_second",
                "bins": int(bins),
            },
        ],
    }


def write_bout_kinematics_visualization_artifacts(
    *,
    zarr_path: Path,
    run_group: zarr.Group,
    run_name: str,
    metrics_by_level: Mapping[str, np.ndarray],
    source_refs: Mapping[str, Any],
    parameters: Mapping[str, Any],
    heading_levels: Sequence[str],
    default_heading_level: str,
    source_speed_level: str,
    bins: int,
    artifact_dpi: int,
    command: Optional[str],
) -> None:
    png_artifact_name = f"{BOUT_KINEMATICS_PNG_PREFIX}_track_{int(source_refs['source_track_id'])}_png"
    spec_artifact_name = f"{BOUT_KINEMATICS_PNG_PREFIX}_track_{int(source_refs['source_track_id'])}_interactive"
    source_paths = {
        "run": f"analysis/bout_kinematics_runs/{run_name}",
        **{
            f"{level}.per_bout_metrics": (
                f"analysis/bout_kinematics_runs/{run_name}/{level}/per_bout_metrics"
            )
            for level in heading_levels
        },
    }
    source_runs = {
        "bout_kinematics": run_name,
        "track_kinematics": source_refs.get("source_track_kinematics_run"),
        "swim_bout": source_refs.get("source_swim_bout_run"),
        "swim_bout_speed_level": source_refs.get("source_swim_bout_speed_level"),
    }
    plot_parameters = {
        "bins": int(bins),
        "artifact_dpi": int(artifact_dpi),
        "heading_levels": list(heading_levels),
        "default_heading_level": default_heading_level,
        "pre_post_mode": parameters.get("pre_post_mode"),
    }
    signature = _artifact_signature(
        {
            "schema_id": BOUT_KINEMATICS_PLOT_SPEC_SCHEMA_ID,
            "run_name": run_name,
            "source_refs": source_refs,
            "parameters": plot_parameters,
        }
    )
    created_at_utc = datetime.now(timezone.utc).isoformat()
    env_info = get_environment_info(disk_path=str(zarr_path), capture_env_vars=False)
    provenance = build_stage_provenance(
        stage="bout_kinematics_visualization",
        created_at_utc=created_at_utc,
        parameters={
            **plot_parameters,
            "plot_schema_id": BOUT_KINEMATICS_PLOT_SPEC_SCHEMA_ID,
            "renderer": BOUT_KINEMATICS_PLOT_RENDERER,
        },
        inputs={
            "zarr_path": str(zarr_path),
            "source_refs": dict(source_refs),
            "source_paths": source_paths,
            "source_runs": source_runs,
        },
        command=command,
        version=BOUT_KINEMATICS_PLOT_RENDERER,
        git=get_git_info(),
        environment=env_info.get("environment"),
        platform=env_info.get("platform"),
        artifacts={
            "png_artifact": f"visualizations/{png_artifact_name}",
            "interactive_artifact": f"visualizations/{spec_artifact_name}",
            "artifact_signature": signature,
        },
    )
    png_bytes = _plot_bout_kinematics_summary(
        metrics_by_level=metrics_by_level,
        default_heading_level=default_heading_level,
        source_speed_level=source_speed_level,
        bins=bins,
    )
    write_png_visualization_artifact(
        run_group,
        png_artifact_name,
        png_bytes,
        description="Bout heading kinematics summary PNG",
        created_by="fisheye.analysis.bout_kinematics",
        artifact_signature=signature,
        created_at_utc=created_at_utc,
        source_paths=source_paths,
        source_runs=source_runs,
        parameters=plot_parameters,
        extra_attrs={
            "plot_schema_id": BOUT_KINEMATICS_PLOT_SPEC_SCHEMA_ID,
            "provenance": provenance,
        },
    )
    spec = _build_bout_kinematics_interactive_spec(
        run_name=run_name,
        source_refs=source_refs,
        parameters=parameters,
        heading_levels=heading_levels,
        default_heading_level=default_heading_level,
        bins=bins,
    )
    write_interactive_plot_spec_artifact(
        run_group,
        spec_artifact_name,
        spec,
        description="Bout heading kinematics interactive plot spec",
        created_by="fisheye.analysis.bout_kinematics",
        renderer=BOUT_KINEMATICS_PLOT_RENDERER,
        artifact_signature=signature,
        created_at_utc=created_at_utc,
        snapshot_artifact=png_artifact_name,
        source_paths=source_paths,
        source_runs=source_runs,
        parameters=plot_parameters,
        extra_attrs={
            "plot_schema_id": BOUT_KINEMATICS_PLOT_SPEC_SCHEMA_ID,
            "provenance": provenance,
        },
    )


def compute_and_save_bout_kinematics(
    zarr_path: Path | str,
    *,
    run_name: Optional[str] = None,
    track_kinematics_run: str = "latest",
    track_scope: str = "offline",
    track_id: int = 0,
    swim_bout_run: str = "latest",
    speed_level: str = "filtered",
    heading_levels: Sequence[str] = ("heading_smoothed", "heading_raw"),
    default_heading_level: str = "heading_smoothed",
    pre_post_mode: str = "fixed_window",
    pre_window_s: float = 0.05,
    post_window_s: float = 0.05,
    within_window: str = "bout_start_end",
    zero_crossing_derivative_threshold_deg_s: float = 0.0,
    dominant_frequency: bool = False,
    dominant_frequency_min_samples: int = 8,
    dominant_frequency_detrend: bool = True,
    write_visualizations: bool = False,
    visualization_bins: int = 40,
    visualization_dpi: int = 150,
    overwrite: bool = False,
    command: Optional[str] = None,
) -> str:
    """Compute and persist linked per-bout heading metrics."""

    if within_window not in WITHIN_WINDOWS:
        expected = ", ".join(WITHIN_WINDOWS)
        raise ValueError(f"Unsupported within_window {within_window!r}; expected one of: {expected}")
    if pre_post_mode not in PRE_POST_MODES:
        expected = ", ".join(PRE_POST_MODES)
        raise ValueError(f"Unsupported pre_post_mode {pre_post_mode!r}; expected one of: {expected}")

    default_heading_level = normalize_heading_level(default_heading_level)
    normalized_heading_levels = tuple(dict.fromkeys(normalize_heading_level(level) for level in heading_levels))
    if default_heading_level not in normalized_heading_levels:
        normalized_heading_levels = (default_heading_level, *normalized_heading_levels)

    zarr_path = Path(zarr_path)
    root = open_zarr_root(zarr_path, mode="r+")
    track_run_group, track_run_name, track_run_path, resolved_scope = _resolve_track_run(
        root,
        track_kinematics_run,
        track_scope=track_scope,
    )
    tracks = track_run_group.get("tracks")
    if tracks is None or f"id_{int(track_id)}" not in tracks:
        raise ValueError(f"Track id_{track_id} not found in {track_run_path}.")
    track_group = tracks[f"id_{int(track_id)}"]

    frames = np.asarray(track_group["frame_indices"][:], dtype=np.int64)
    if "time_seconds" in track_group:
        times = np.asarray(track_group["time_seconds"][:], dtype=np.float64)
    else:
        fps_for_time = float(track_run_group.attrs.get("fps", 0.0))
        times = frames.astype(np.float64) / fps_for_time if fps_for_time > 0 else np.arange(frames.size)
    fps = float(track_run_group.attrs.get("fps", 0.0))
    if fps <= 0:
        raise ValueError(f"Track kinematics run {track_run_path} has invalid fps={fps!r}.")
    positions_mm = None
    positions_px = None
    source_position_arrays: dict[str, str] = {}
    source_validity_arrays: dict[str, str] = {}
    transition_valid = None
    sample_valid = None
    if "positions_mm" in track_group:
        positions_mm = np.asarray(track_group["positions_mm"][:], dtype=np.float64)
        if positions_mm.shape != (frames.shape[0], 2):
            raise ValueError(
                f"positions_mm shape {positions_mm.shape} does not match expected {(frames.shape[0], 2)}."
            )
        source_position_arrays["positions_mm"] = f"{track_run_path}/tracks/id_{int(track_id)}/positions_mm"
    if "positions_px" in track_group:
        positions_px = np.asarray(track_group["positions_px"][:], dtype=np.float64)
        if positions_px.shape != (frames.shape[0], 2):
            raise ValueError(
                f"positions_px shape {positions_px.shape} does not match expected {(frames.shape[0], 2)}."
            )
        source_position_arrays["positions_px"] = f"{track_run_path}/tracks/id_{int(track_id)}/positions_px"
    if "transition_valid" in track_group:
        transition_valid = np.asarray(track_group["transition_valid"][:], dtype=bool)
        if transition_valid.shape[0] != frames.shape[0]:
            raise ValueError(
                f"transition_valid length {transition_valid.shape[0]} does not match frames length {frames.shape[0]}."
            )
        source_validity_arrays["transition_valid"] = f"{track_run_path}/tracks/id_{int(track_id)}/transition_valid"
    if "sample_valid" in track_group:
        sample_valid = np.asarray(track_group["sample_valid"][:], dtype=bool)
        if sample_valid.shape[0] != frames.shape[0]:
            raise ValueError(
                f"sample_valid length {sample_valid.shape[0]} does not match frames length {frames.shape[0]}."
            )
        source_validity_arrays["sample_valid"] = f"{track_run_path}/tracks/id_{int(track_id)}/sample_valid"

    pre_window_frames = max(1, int(round(float(pre_window_s) * fps)))
    post_window_frames = max(1, int(round(float(post_window_s) * fps)))
    source_heading_arrays = {
        heading_level: f"{track_run_path}/tracks/id_{int(track_id)}/{HEADING_LEVEL_TO_ARRAY[heading_level]}"
        for heading_level in normalized_heading_levels
    }

    _, swim_run_group, swim_run_name, source_speed_level, swim_level_path = _resolve_swim_bout_run(
        root,
        swim_bout_run,
        speed_level,
    )
    source_track_id = swim_run_group.attrs.get("track_id")
    if source_track_id is not None and int(source_track_id) != int(track_id):
        raise ValueError(
            f"Swim-bout run {swim_run_name!r} was derived from track_id={source_track_id}, "
            f"not requested track_id={track_id}."
        )

    source_track_run = swim_run_group.attrs.get("source_track_kinematics_run")
    if source_track_run is not None and str(source_track_run).strip("/") not in {
        track_run_name,
        f"{resolved_scope}/{track_run_name}",
        track_run_path,
    }:
        raise ValueError(
            f"Swim-bout run {swim_run_name!r} source_track_kinematics_run={source_track_run!r} "
            f"does not match selected {track_run_path!r}."
        )

    swim_level_group = swim_run_group[source_speed_level]
    bouts, bout_attrs = load_structured_dataset(swim_level_group, "bouts")
    peak_events: Optional[np.ndarray] = None
    peak_event_attrs: Mapping[str, Any] = {}
    if "peak_events" in swim_level_group:
        loaded_peak_events, loaded_peak_event_attrs = load_structured_dataset(swim_level_group, "peak_events")
        if len(loaded_peak_events) == len(bouts) and _records_align_by_bout_id(bouts, loaded_peak_events):
            peak_events = loaded_peak_events
            peak_event_attrs = loaded_peak_event_attrs

    if "analysis" not in root:
        analysis = root.create_group("analysis")
    else:
        analysis = root["analysis"]
    if "bout_kinematics_runs" not in analysis:
        parent = analysis.create_group("bout_kinematics_runs")
    else:
        parent = analysis["bout_kinematics_runs"]

    if run_name is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        run_name = f"bout_kinematics_{timestamp}"
    elif run_name in parent:
        if not overwrite:
            raise ValueError(
                f"Bout kinematics run {run_name!r} already exists. Use --overwrite or a different name."
            )
        del parent[run_name]

    run_group = parent.create_group(run_name)
    created_at_utc = datetime.now(timezone.utc).isoformat()
    source_refs = {
        "zarr_path": str(zarr_path),
        "source_track_kinematics_run": track_run_name,
        "source_track_kinematics_scope": resolved_scope,
        "source_track_kinematics_path": track_run_path,
        "source_track_kinematics_track_path": f"{track_run_path}/tracks/id_{int(track_id)}",
        "source_swim_bout_run": swim_run_name,
        "source_swim_bout_speed_level": source_speed_level,
        "source_swim_bout_path": swim_level_path,
        "source_track_id": int(track_id),
        "source_heading_arrays": source_heading_arrays,
        "source_position_arrays": source_position_arrays,
        "source_validity_arrays": source_validity_arrays,
    }
    if peak_events is not None:
        source_refs["source_peak_events_path"] = f"{swim_level_path}/peak_events"
    source_bout_field_names = list(bout_attrs.get("field_names", bouts.dtype.names or []))
    source_interpolated_threshold_fields = [
        field
        for field in (
            "core_start_time_s_interpolated",
            "core_end_time_s_interpolated",
            "core_duration_s_interpolated",
            "core_start_time_interpolated_valid",
            "core_end_time_interpolated_valid",
        )
        if field in source_bout_field_names
    ]
    source_peak_event_fields = list(peak_event_attrs.get("field_names", peak_events.dtype.names if peak_events is not None else []))
    parameters = {
        "default_heading_level": default_heading_level,
        "heading_levels": list(normalized_heading_levels),
        "pre_post_mode": pre_post_mode,
        "pre_window_s": float(pre_window_s),
        "post_window_s": float(post_window_s),
        "resolved_pre_window_frames": int(pre_window_frames),
        "resolved_post_window_frames": int(post_window_frames),
        "within_window": within_window,
        "heading_units": "degrees",
        "heading_unwrap_policy": "numpy.unwrap_contiguous_window",
        "source_interpolated_threshold_fields": source_interpolated_threshold_fields,
        "source_peak_event_fields": source_peak_event_fields,
        "zero_crossing_derivative_threshold_deg_s": float(zero_crossing_derivative_threshold_deg_s),
        "dominant_frequency": {
            "enabled": bool(dominant_frequency),
            "min_samples": int(dominant_frequency_min_samples),
            "method": "rfft_peak",
            "detrend": bool(dominant_frequency_detrend),
        },
    }
    run_group.attrs["schema_id"] = SCHEMA_ID
    run_group.attrs["schema_version"] = SCHEMA_VERSION
    run_group.attrs["method"] = METHOD
    run_group.attrs["method_version"] = METHOD_VERSION
    run_group.attrs["row_axis"] = "swim_bout_rows"
    run_group.attrs["source_refs"] = source_refs
    run_group.attrs["parameters"] = parameters
    run_group.attrs["source_track_id"] = int(track_id)
    run_group.attrs["source_swim_bout_run"] = swim_run_name
    run_group.attrs["source_swim_bout_speed_level"] = source_speed_level
    run_group.attrs["source_track_kinematics_run"] = track_run_name
    run_group.attrs["default_heading_level"] = default_heading_level

    written_levels: list[str] = []
    metrics_by_level: dict[str, np.ndarray] = {}
    for heading_level in normalized_heading_levels:
        array_name = HEADING_LEVEL_TO_ARRAY[heading_level]
        if array_name not in track_group:
            raise ValueError(f"Heading source array {array_name!r} not found in {track_run_path}/tracks/id_{track_id}.")
        headings = np.asarray(track_group[array_name][:], dtype=np.float64)
        if headings.shape[0] != frames.shape[0]:
            raise ValueError(
                f"Heading source {array_name!r} length {headings.shape[0]} does not match frames length {frames.shape[0]}."
            )
        level_group = run_group.create_group(heading_level)
        level_group.attrs["heading_source_array"] = array_name
        level_group.attrs["is_default_heading_level"] = heading_level == default_heading_level
        level_group.attrs["source_swim_bout_path"] = swim_level_path
        metrics = _build_metrics_for_heading(
            bouts=bouts,
            peak_events=peak_events,
            frames=frames,
            times=times,
            headings=headings,
            transition_valid=transition_valid,
            sample_valid=sample_valid,
            positions_mm=positions_mm,
            positions_px=positions_px,
            fps=fps,
            pre_post_mode=pre_post_mode,
            pre_window_frames=pre_window_frames,
            post_window_frames=post_window_frames,
            within_window=within_window,
            derivative_threshold_deg_s=zero_crossing_derivative_threshold_deg_s,
            dominant_frequency_enabled=dominant_frequency,
            dominant_frequency_min_samples=dominant_frequency_min_samples,
            dominant_frequency_detrend=dominant_frequency_detrend,
        )
        write_columnar_dataset(
            level_group,
            "per_bout_metrics",
            metrics,
            attrs={
                "schema_id": f"{SCHEMA_ID}.per_bout_metrics",
                "schema_version": SCHEMA_VERSION,
                "heading_level": heading_level,
                "heading_source_array": array_name,
                "source_bout_count": int(len(bouts)),
                "source_bout_field_names": source_bout_field_names,
                "source_interpolated_threshold_fields": source_interpolated_threshold_fields,
                "source_peak_event_fields": source_peak_event_fields,
            },
        )
        written_levels.append(heading_level)
        metrics_by_level[heading_level] = metrics

    run_group.attrs["heading_levels"] = written_levels
    parent.attrs["latest"] = run_name

    git_info = get_git_info()
    env_info = get_environment_info(disk_path=str(zarr_path), capture_env_vars=False)
    provenance = build_stage_provenance(
        stage="bout_kinematics",
        created_at_utc=created_at_utc,
        parameters=parameters,
        inputs=source_refs,
        command=command,
        version=METHOD_VERSION,
        git=git_info,
        environment=env_info.get("environment"),
        platform=env_info.get("platform"),
        artifacts={
            "run_path": f"analysis/bout_kinematics_runs/{run_name}",
            "heading_levels": written_levels,
        },
    )
    write_stage_provenance(run_group, provenance)

    if write_visualizations:
        write_bout_kinematics_visualization_artifacts(
            zarr_path=zarr_path,
            run_group=run_group,
            run_name=str(run_name),
            metrics_by_level=metrics_by_level,
            source_refs=source_refs,
            parameters=parameters,
            heading_levels=written_levels,
            default_heading_level=default_heading_level,
            source_speed_level=source_speed_level,
            bins=int(visualization_bins),
            artifact_dpi=int(visualization_dpi),
            command=command,
        )

    return str(run_name)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Compute per-bout heading kinematics.")
    parser.add_argument("zarr_path", type=Path, help="Path to the Palette Zarr archive.")
    parser.add_argument("--run-name", type=str, default=None, help="Output bout-kinematics run name.")
    parser.add_argument("--overwrite", action="store_true", help="Replace --run-name if it exists.")
    parser.add_argument("--track-kinematics-run", type=str, default="latest", help="Track kinematics run name/path.")
    parser.add_argument("--track-scope", type=str, default="offline", help="Track kinematics scope for bare run names.")
    parser.add_argument("--track-id", type=int, default=0, help="Track ID to analyze.")
    parser.add_argument("--swim-bout-run", type=str, default="latest", help="Source swim-bout run name.")
    parser.add_argument("--speed-level", type=str, default="filtered", help="Source swim-bout speed level.")
    parser.add_argument(
        "--heading-level",
        action="append",
        dest="heading_levels",
        default=None,
        help="Heading level to compute. Repeatable. Defaults to smoothed and raw.",
    )
    parser.add_argument("--default-heading-level", type=str, default="heading_smoothed")
    parser.add_argument(
        "--pre-post-mode",
        choices=PRE_POST_MODES,
        default="fixed_window",
        help="How to resolve pre/post measurement epochs.",
    )
    parser.add_argument("--pre-window-s", type=float, default=0.05)
    parser.add_argument("--post-window-s", type=float, default=0.05)
    parser.add_argument("--within-window", choices=WITHIN_WINDOWS, default="bout_start_end")
    parser.add_argument("--zero-crossing-derivative-threshold-deg-s", type=float, default=0.0)
    parser.add_argument("--dominant-frequency", action="store_true", help="Compute optional dominant frequency.")
    parser.add_argument("--dominant-frequency-min-samples", type=int, default=8)
    parser.add_argument(
        "--dominant-frequency-no-detrend",
        action="store_true",
        help="Disable linear detrending before frequency estimation.",
    )
    parser.add_argument(
        "--write-zarr-artifacts",
        action="store_true",
        help="Write PNG and interactive visualization artifacts under the bout-kinematics run.",
    )
    parser.add_argument("--visualization-bins", type=int, default=40)
    parser.add_argument("--visualization-dpi", type=int, default=150)
    args = parser.parse_args(argv)

    compute_and_save_bout_kinematics(
        zarr_path=args.zarr_path,
        run_name=args.run_name,
        track_kinematics_run=args.track_kinematics_run,
        track_scope=args.track_scope,
        track_id=args.track_id,
        swim_bout_run=args.swim_bout_run,
        speed_level=args.speed_level,
        heading_levels=tuple(args.heading_levels) if args.heading_levels else ("heading_smoothed", "heading_raw"),
        default_heading_level=args.default_heading_level,
        pre_post_mode=args.pre_post_mode,
        pre_window_s=args.pre_window_s,
        post_window_s=args.post_window_s,
        within_window=args.within_window,
        zero_crossing_derivative_threshold_deg_s=args.zero_crossing_derivative_threshold_deg_s,
        dominant_frequency=args.dominant_frequency,
        dominant_frequency_min_samples=args.dominant_frequency_min_samples,
        dominant_frequency_detrend=not args.dominant_frequency_no_detrend,
        write_visualizations=args.write_zarr_artifacts,
        visualization_bins=args.visualization_bins,
        visualization_dpi=args.visualization_dpi,
        overwrite=args.overwrite,
        command=" ".join(sys.argv if argv is None else [sys.argv[0], *argv]),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
