#!/usr/bin/env python3
"""
Detect swim bouts from speed processing levels in track kinematics tracks.

This script reads pre-computed speed data from track kinematics tracks plus
optional transformed detector signals and detects swim bouts across candidate
levels. Results are stored hierarchically under a single run name with one
subgroup per detector-signal candidate.

Default storage structure (`--layout compact_v2`):
    analysis/swim_bout_runs/<run_name>/
    ├── indexes/
    ├── tables/
    ├── series/
    └── attrs: layout = "compact_tabular_v2"

Compatibility storage structure (`--layout hierarchical_v1`):
    analysis/swim_bout_runs/<run_name>/
    ├── speed_raw/
    │   ├── bouts (structured array)
    │   └── metadata (attrs)
    ├── speed_filtered/
    │   ├── bouts
    │   └── metadata
    ├── speed_smoothed/
    │   ├── bouts
    │   └── metadata
    ├── speed_averaged/
    │   ├── bouts
    │   └── metadata
    ├── speed_exponential/
    │   ├── bouts
    │   ├── detection_signal_mm_s
    │   └── metadata
    ├── default_level = "speed_exponential" or another stored speed level (attr)
    └── run_metadata (attrs: threshold, source_track_kinematics_run, etc.)

Usage (basic):
    scripts/py -m fisheye.analysis.detect_bouts_multi_level /path/to/archive.zarr

    Auto-generates run name like: swim_bout_detect_20250905_143022

Usage (with options):
    scripts/py -m fisheye.analysis.detect_bouts_multi_level /path/to/archive.zarr \\
        --run-name custom_run \\
        --track-kinematics-run latest \\
        --method peak_event \\
        --default-level exponential \\
        --exponential-tau-s 0.025 \\
        --min-peak-prominence-mm-s 4.0 \\
        --min-peak-distance-s 0.10 \\
        --peak-width-rel-height 0.98 \\
        --overwrite
"""

from __future__ import annotations

import argparse
import hashlib
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
from scipy import signal

from fisheye.shared.zarr.columnar import store_array, write_columnar_dataset
from fisheye.analysis.track_kinematics_io import load_track_kinematics_track
from fisheye.shared.json_safety import json_attr_safe, json_attr_safe_mapping, strict_json_dumps
from fisheye.shared.run_lineage_fingerprint import write_best_effort_run_lineage_attrs
from fisheye.shared.run_provenance import build_run_provenance_from_stage_record
from fisheye.shared.stage_provenance import build_stage_provenance, write_stage_provenance
from fisheye.shared.zarr_run_completion import mark_run_complete, mark_run_started, require_runs_parent
from fisheye.shared.system_metadata import get_environment_info, get_git_info
from fisheye.shared.zarr_io import open_zarr_root


SPEED_LEVELS = (
    "speed_raw",
    "speed_filtered",
    "speed_smoothed",
    "speed_averaged",
    "speed_exponential",
)
BASE_SPEED_LEVELS = ("speed_raw", "speed_filtered", "speed_smoothed", "speed_averaged")
PATH_DISTANCE_LEVEL_SOURCE = {
    "speed_raw": "raw",
    "speed_filtered": "filtered",
    "speed_smoothed": "smoothed",
    # speed_averaged is a moving average of the smoothed speed trace; it does
    # not have an independent frame-path-distance array.
    "speed_averaged": "smoothed",
    # speed_exponential is a derived response trace; use the same movement
    # metric source as the configured source speed level.
    "speed_exponential": "filtered",
}
SPEED_LEVEL_ALIASES = {
    "raw": "speed_raw",
    "filtered": "speed_filtered",
    "smoothed": "speed_smoothed",
    "averaged": "speed_averaged",
    "exp": "speed_exponential",
    "exponential": "speed_exponential",
}
SPEED_LEVEL_CHOICES = tuple(SPEED_LEVEL_ALIASES) + SPEED_LEVELS
DEFAULT_DETECTION_METHOD = "peak_event"
DEFAULT_SWIM_BOUT_LEVEL = "speed_exponential"
DEFAULT_EXPONENTIAL_TAU_S = 0.025
DEFAULT_EXPONENTIAL_SOURCE_LEVEL = "filtered"
DEFAULT_MIN_PEAK_PROMINENCE_MM_S = 4.0
DEFAULT_MIN_PEAK_DISTANCE_S = 0.10
DEFAULT_PEAK_WIDTH_REL_HEIGHT = 0.98
BOUNDARY_MODES = ("threshold", "local_minimum")
GAP_MERGE_POLICIES = ("sampled_frame_gap", "interpolated_core_gap")
PEAK_EVENT_BOUNDARY_MODES = ("relative_prominence_width",)
SHAPE_SPLIT_POLICIES = ("none",)
DURATION_FRAME_ROUNDING_POLICY = "ceil_seconds_times_fps"
BOUNDARY_FRAME_ROUNDING_POLICY = "round_seconds_times_fps"
SWIM_BOUT_RUN_SCHEMA_ID = "palette.swim_bout_runs"
SWIM_BOUT_RUN_SCHEMA_VERSION = 6
SWIM_BOUT_RUN_SCHEMA_VERSION_COMPACT_V2 = 7
SWIM_BOUT_LAYOUT_HIERARCHICAL_V1 = "hierarchical_v1"
SWIM_BOUT_LAYOUT_COMPACT_V2 = "compact_v2"
SWIM_BOUT_STORED_LAYOUT_COMPACT_V2 = "compact_tabular_v2"
SWIM_BOUT_LAYOUT_CHOICES = (SWIM_BOUT_LAYOUT_HIERARCHICAL_V1, SWIM_BOUT_LAYOUT_COMPACT_V2)
SWIM_BOUT_LAYOUT_DEFAULT = SWIM_BOUT_LAYOUT_COMPACT_V2
BOUT_METRIC_SCHEMA_ID = "palette.swim_bout_metrics.v3"
DETECTION_SIGNAL_SCHEMA_ID = "palette.swim_bout_detection_signal.v1"
DETECTION_SIGNAL_SCHEMA_VERSION = 1
PEAK_EVENT_SCHEMA_ID = "palette.swim_bout_peak_events.v1"
PEAK_EVENT_SCHEMA_VERSION = 1
METHOD_VERSION = "detect_bouts_multi_level.v7"
THRESHOLD_CROSSING_INTERPOLATION = "linear_between_samples"
PHASE_TIMING_SCHEMA_ID = "palette.swim_bout_phase_timing"
PHASE_TIMING_SCHEMA_VERSION = 1


_json_safe_attr_value = json_attr_safe
_json_safe_attrs = json_attr_safe_mapping


def _finish_timed_phase(
    phase_durations_s: Dict[str, float],
    phase_name: str,
    started_at: float,
) -> float:
    """Record and immediately report a monotonic-clock phase duration."""

    elapsed_s = max(0.0, float(perf_counter() - started_at))
    phase_durations_s[phase_name] = elapsed_s
    print(
        f"phase_timing phase={phase_name} elapsed_s={elapsed_s:.6f}",
        flush=True,
    )
    return elapsed_s


def _build_phase_timing_payload(
    *,
    phase_durations_s: Mapping[str, float],
    detection_levels: Mapping[str, Mapping[str, Any]],
    timed_pipeline_elapsed_s: float,
) -> Dict[str, Any]:
    """Build the additive, non-scientific performance telemetry contract."""

    normalized_phases = {
        str(name): max(0.0, float(elapsed_s))
        for name, elapsed_s in phase_durations_s.items()
    }
    phase_sum_s = float(sum(normalized_phases.values()))
    total_s = max(0.0, float(timed_pipeline_elapsed_s))
    return {
        "schema_id": PHASE_TIMING_SCHEMA_ID,
        "schema_version": PHASE_TIMING_SCHEMA_VERSION,
        "clock": "time.perf_counter",
        "scope": "load_track_kinematics_through_payload_write",
        "phase_durations_s": normalized_phases,
        "detection_levels": {
            str(level): dict(values) for level, values in detection_levels.items()
        },
        "timed_pipeline_elapsed_s": total_s,
        "phase_sum_s": phase_sum_s,
        "unattributed_elapsed_s": max(0.0, total_s - phase_sum_s),
    }


def _strict_json_dumps(value: Any) -> str:
    """Serialize strict JSON metadata strings for compact table rows."""

    return strict_json_dumps(value)


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_strict_json_dumps(value).encode("utf-8")).hexdigest()


def _duration_seconds_to_frames(duration_s: float, fps: float) -> int:
    """Resolve a duration in seconds to an integer frame count.

    Ceiling preserves the user-facing lower bound: a positive duration that
    falls between frame intervals still requires at least the next frame.
    """

    duration_s = float(duration_s)
    fps = float(fps)
    if duration_s < 0:
        raise ValueError(f"duration_s must be >= 0, got {duration_s!r}.")
    if fps <= 0:
        raise ValueError(f"fps must be > 0, got {fps!r}.")
    if duration_s == 0:
        return 0
    # Subtract a tiny epsilon so exact products like 0.1 * 60 do not become
    # 7 frames because of floating point representation.
    return max(0, int(math.ceil(duration_s * fps - 1e-9)))


def _resolve_min_gap_frames(
    *,
    min_gap_duration_s: float,
    fps: float,
    min_gap_frames: Optional[int],
) -> tuple[int, str]:
    if min_gap_frames is not None:
        resolved = int(min_gap_frames)
        if resolved < 0:
            raise ValueError(f"min_gap_frames must be >= 0, got {min_gap_frames!r}.")
        return resolved, "explicit_frames"
    return _duration_seconds_to_frames(min_gap_duration_s, fps), DURATION_FRAME_ROUNDING_POLICY


def normalize_speed_level(value: str) -> str:
    """Normalize user-facing speed-level names to stored subgroup names."""

    level = str(value).strip()
    normalized = SPEED_LEVEL_ALIASES.get(level, level)
    if normalized not in SPEED_LEVELS:
        expected = ", ".join(SPEED_LEVEL_CHOICES)
        raise ValueError(f"Unsupported speed level {value!r}; expected one of: {expected}")
    return normalized


def _bout_dtype() -> np.dtype:
    return np.dtype([
        ('bout_id', 'i4'),
        ('start_frame', 'i8'),
        ('end_frame', 'i8'),
        ('core_start_frame', 'i8'),
        ('core_end_frame', 'i8'),
        ('duration_frames', 'i8'),
        ('duration_s', 'f8'),
        ('elapsed_duration_s', 'f8'),
        ('observed_duration_s', 'f8'),
        ('core_duration_frames', 'i8'),
        ('core_duration_s', 'f8'),
        ('path_length_mm', 'f8'),
        ('path_length_px', 'f8'),
        ('net_displacement_mm', 'f8'),
        ('net_displacement_px', 'f8'),
        ('mean_speed_mm_s', 'f8'),
        ('peak_detection_signal_mm_s', 'f8'),
        ('peak_physical_speed_mm_s', 'f8'),
        ('n_valid_transitions', 'i8'),
        ('n_invalid_transitions', 'i8'),
        ('valid_transition_fraction', 'f8'),
        ('gap_censored', '?'),
        ('start_time_s', 'f8'),
        ('end_time_s', 'f8'),
        ('core_start_time_s', 'f8'),
        ('core_end_time_s', 'f8'),
        ('core_start_time_s_interpolated', 'f8'),
        ('core_end_time_s_interpolated', 'f8'),
        ('core_duration_s_interpolated', 'f8'),
        ('core_start_time_interpolated_valid', '?'),
        ('core_end_time_interpolated_valid', '?'),
    ])


def _empty_bouts() -> np.ndarray:
    return np.zeros(0, dtype=_bout_dtype())


def _peak_event_dtype() -> np.dtype:
    return np.dtype([
        ('bout_id', 'i4'),
        ('peak_index', 'i8'),
        ('peak_frame', 'i8'),
        ('peak_time_s', 'f8'),
        ('peak_signal_value_mm_s', 'f8'),
        ('peak_prominence_mm_s', 'f8'),
        ('peak_width_samples', 'f8'),
        ('peak_width_s', 'f8'),
        ('peak_width_height_mm_s', 'f8'),
        ('left_ips', 'f8'),
        ('right_ips', 'f8'),
        ('left_width_frame_interpolated', 'f8'),
        ('right_width_frame_interpolated', 'f8'),
        ('left_base_index', 'i8'),
        ('right_base_index', 'i8'),
        ('left_base_frame', 'i8'),
        ('right_base_frame', 'i8'),
        ('left_base_signal_value_mm_s', 'f8'),
        ('right_base_signal_value_mm_s', 'f8'),
        ('boundary_mode', 'S32'),
        ('shape_split_policy', 'S32'),
    ])


def _empty_peak_events() -> np.ndarray:
    return np.zeros(0, dtype=_peak_event_dtype())


def _finite_sum_or_nan(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return float(np.sum(finite)) if finite.size else float("nan")


def _finite_mean_or_nan(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return float(np.mean(finite)) if finite.size else float("nan")


def _transition_delta_seconds(frames: np.ndarray, fps: float) -> np.ndarray:
    delta_seconds = np.zeros(frames.shape[0], dtype=np.float64)
    if frames.size >= 2 and fps > 0:
        delta_seconds[1:] = np.diff(frames.astype(np.float64, copy=False)) / float(fps)
    return delta_seconds


def _fallback_transition_valid(
    speed: np.ndarray,
    frames: np.ndarray,
    fps: float,
) -> np.ndarray:
    valid = np.zeros(frames.shape[0], dtype=bool)
    if frames.size >= 2 and fps > 0:
        delta_frames = np.diff(frames)
        valid[1:] = (delta_frames == 1) & np.isfinite(speed[1:])
    return valid


def _effective_transition_valid(
    *,
    speed: np.ndarray,
    frames: np.ndarray,
    fps: float,
    transition_valid: Optional[np.ndarray],
    sample_valid: Optional[np.ndarray],
) -> np.ndarray:
    if transition_valid is None:
        effective = _fallback_transition_valid(speed, frames, fps)
    else:
        effective = np.asarray(transition_valid, dtype=bool).copy()
        if effective.shape[0] != frames.shape[0]:
            raise ValueError(
                f"transition_valid length {effective.shape[0]} does not match frames length {frames.shape[0]}."
            )

    if sample_valid is not None:
        sample_valid = np.asarray(sample_valid, dtype=bool)
        if sample_valid.shape[0] != frames.shape[0]:
            raise ValueError(
                f"sample_valid length {sample_valid.shape[0]} does not match frames length {frames.shape[0]}."
            )
        endpoint_valid = np.zeros_like(effective)
        if endpoint_valid.size >= 2:
            endpoint_valid[1:] = sample_valid[1:] & sample_valid[:-1]
        effective &= endpoint_valid

    return effective


def _causal_exponential_speed_response(
    speed: np.ndarray,
    frames: np.ndarray,
    fps: float,
    *,
    tau_s: float,
    transition_valid: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Return a causal normalized exponential response to a speed trace.

    The recurrence is equivalent to convolution with a one-sided exponential
    kernel. Invalid samples reset the response so motion does not smear across
    track gaps.
    """

    if tau_s <= 0:
        raise ValueError(f"exponential_tau_s must be > 0, got {tau_s!r}.")
    speed = np.asarray(speed, dtype=np.float64)
    frames = np.asarray(frames, dtype=np.int64)
    if speed.shape[0] != frames.shape[0]:
        raise ValueError(f"speed length {speed.shape[0]} does not match frames length {frames.shape[0]}.")
    if fps <= 0:
        raise ValueError(f"fps must be > 0, got {fps!r}.")

    if transition_valid is None:
        valid_transition = _fallback_transition_valid(speed, frames, fps)
    else:
        valid_transition = np.asarray(transition_valid, dtype=bool)
        if valid_transition.shape[0] != speed.shape[0]:
            raise ValueError(
                f"transition_valid length {valid_transition.shape[0]} does not match speed length {speed.shape[0]}."
            )

    response = np.full(speed.shape, np.nan, dtype=np.float64)
    state: Optional[float] = None
    for idx, value in enumerate(speed):
        if not np.isfinite(value):
            state = None
            continue
        value = max(float(value), 0.0)
        if state is None or idx == 0 or not bool(valid_transition[idx]):
            state = value
        else:
            dt_s = max(float(frames[idx] - frames[idx - 1]) / float(fps), 0.0)
            alpha = 1.0 - float(np.exp(-dt_s / float(tau_s)))
            state = alpha * value + (1.0 - alpha) * state
        response[idx] = state
    return response


def _detection_signal_attrs(
    *,
    level: str,
    source_track_path: str,
    path_distance_source_level: str,
    exponential_source_key: str,
    exponential_tau_s: float,
) -> Dict[str, Any]:
    """Describe the signal used to define bout boundaries for one subgroup."""

    is_exponential = level == "speed_exponential"
    source_level = exponential_source_key if is_exponential else level
    source_array = f"{source_level}_mm"
    attrs: Dict[str, Any] = {
        "signal_level": level,
        "detection_signal_schema_id": DETECTION_SIGNAL_SCHEMA_ID,
        "detection_signal_schema_version": DETECTION_SIGNAL_SCHEMA_VERSION,
        "detection_signal_role": "bout_detection",
        "detection_signal_units": "mm/s",
        "detection_signal_source_level": source_level,
        "detection_signal_source_array": source_array,
        "detection_signal_source_path": f"{source_track_path}/{source_array}",
        "detection_signal_array": "detection_signal_mm_s" if is_exponential else source_array,
        "detection_signal_transform_type": "convolution" if is_exponential else "identity",
        "detection_signal_transform_family": "causal_exponential" if is_exponential else "identity",
        "detection_signal_is_primary_physical_speed": not is_exponential,
        "movement_metric_source_level": path_distance_source_level,
        "movement_metric_path_distance_array": f"frame_path_distance_{path_distance_source_level}_mm",
        "peak_detection_signal_field": "peak_detection_signal_mm_s",
        "peak_physical_speed_field": "peak_physical_speed_mm_s",
    }
    if is_exponential:
        attrs.update(
            {
                "detection_signal_kernel": "exp(-t/tau)",
                "detection_signal_kernel_family": "causal_exponential",
                "detection_signal_tau_s": float(exponential_tau_s),
                "detection_signal_causal": True,
                "detection_signal_normalized": True,
                # Legacy attrs retained for old consumers while the generic
                # detection-signal contract becomes the canonical wording.
                "speed_transform": "causal_exponential_response",
                "speed_transform_kernel": "exp(-t/tau)",
                "exponential_tau_s": float(exponential_tau_s),
                "exponential_source_level": exponential_source_key,
            }
        )
    return attrs


def _sum_valid_path_distance(
    path_distance: Optional[np.ndarray],
    valid_mask: np.ndarray,
) -> float:
    if path_distance is None:
        return float("nan")
    values = np.asarray(path_distance, dtype=np.float64)
    if values.shape[0] != valid_mask.shape[0]:
        raise ValueError(
            f"path-distance length {values.shape[0]} does not match validity length {valid_mask.shape[0]}."
        )
    return _finite_sum_or_nan(values[valid_mask])


def _net_displacement(
    positions: Optional[np.ndarray],
    start_idx: int,
    end_idx: int,
) -> float:
    if positions is None:
        return float("nan")
    values = np.asarray(positions, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < 2:
        return float("nan")
    if start_idx < 0 or end_idx < 0 or start_idx >= values.shape[0] or end_idx >= values.shape[0]:
        return float("nan")
    start = values[start_idx, :2]
    end = values[end_idx, :2]
    if not (np.all(np.isfinite(start)) and np.all(np.isfinite(end))):
        return float("nan")
    return float(np.linalg.norm(end - start))


def _threshold_crossing_time_s(
    *,
    speed: np.ndarray,
    frames: np.ndarray,
    fps: float,
    threshold: float,
    below_idx: int,
    above_idx: int,
) -> tuple[float, bool]:
    """Linearly interpolate a threshold crossing between adjacent samples."""

    if fps <= 0:
        return float("nan"), False
    if below_idx < 0 or above_idx < 0 or below_idx >= speed.size or above_idx >= speed.size:
        return float("nan"), False
    if abs(int(above_idx) - int(below_idx)) != 1:
        return float("nan"), False
    below_speed = float(speed[below_idx])
    above_speed = float(speed[above_idx])
    if not (np.isfinite(below_speed) and np.isfinite(above_speed)):
        return float("nan"), False
    if below_speed == above_speed:
        return float("nan"), False
    threshold = float(threshold)
    lower = min(below_speed, above_speed)
    upper = max(below_speed, above_speed)
    if threshold < lower or threshold > upper:
        return float("nan"), False

    fraction = (threshold - below_speed) / (above_speed - below_speed)
    if not np.isfinite(fraction):
        return float("nan"), False
    fraction = float(np.clip(fraction, 0.0, 1.0))
    frame = float(frames[below_idx]) + fraction * float(frames[above_idx] - frames[below_idx])
    return frame / float(fps), True


def _core_threshold_crossing_times_s(
    *,
    speed: np.ndarray,
    frames: np.ndarray,
    fps: float,
    threshold: Optional[float],
    core_start_idx: int,
    core_end_exclusive: int,
) -> tuple[float, bool, float, bool, float]:
    """Return interpolated start/end crossing times for a threshold core."""

    if threshold is None:
        return float("nan"), False, float("nan"), False, float("nan")

    core_end_idx = int(core_end_exclusive) - 1
    start_time, start_valid = _threshold_crossing_time_s(
        speed=speed,
        frames=frames,
        fps=fps,
        threshold=float(threshold),
        below_idx=int(core_start_idx) - 1,
        above_idx=int(core_start_idx),
    )
    end_time, end_valid = _threshold_crossing_time_s(
        speed=speed,
        frames=frames,
        fps=fps,
        threshold=float(threshold),
        below_idx=core_end_idx + 1,
        above_idx=core_end_idx,
    )
    duration = (
        float(end_time - start_time)
        if start_valid and end_valid and end_time >= start_time
        else float("nan")
    )
    return start_time, start_valid, end_time, end_valid, duration


def _interpolated_frame_at_sample_index(frames: np.ndarray, sample_index: float) -> float:
    """Map a fractional signal sample index to a source frame coordinate."""

    if frames.size == 0 or not np.isfinite(sample_index):
        return float("nan")
    if sample_index <= 0:
        return float(frames[0])
    last_index = frames.size - 1
    if sample_index >= last_index:
        return float(frames[last_index])
    left = int(np.floor(sample_index))
    right = int(np.ceil(sample_index))
    if left == right:
        return float(frames[left])
    fraction = float(sample_index - left)
    return float(frames[left]) + fraction * float(frames[right] - frames[left])


def _gap_merge_min_gap_duration_s(
    *,
    min_gap_duration_s: float,
    fps: float,
    min_gap_frames: Optional[int],
) -> tuple[float, str]:
    if min_gap_frames is not None:
        return float(int(min_gap_frames) / float(fps)), "explicit_frames"
    return float(min_gap_duration_s), "seconds"


def _merge_threshold_segments(
    *,
    speed: np.ndarray,
    frames: np.ndarray,
    fps: float,
    threshold: float,
    starts: np.ndarray,
    ends_exclusive: np.ndarray,
    min_gap_frames: int,
    min_gap_duration_s: float,
    gap_merge_policy: str,
) -> tuple[np.ndarray, np.ndarray]:
    if len(starts) <= 1:
        return starts, ends_exclusive

    if gap_merge_policy not in GAP_MERGE_POLICIES:
        expected = ", ".join(GAP_MERGE_POLICIES)
        raise ValueError(f"Unsupported gap_merge_policy {gap_merge_policy!r}; expected one of: {expected}")

    merged_starts = [int(starts[0])]
    merged_ends: list[int] = []
    current_end = int(ends_exclusive[0])
    for i in range(1, len(starts)):
        next_start = int(starts[i])
        next_end = int(ends_exclusive[i])
        if gap_merge_policy == "sampled_frame_gap":
            should_merge = (next_start - current_end) < int(min_gap_frames)
        else:
            prev_end_idx = current_end - 1
            prev_end_time_s, prev_end_valid = _threshold_crossing_time_s(
                speed=speed,
                frames=frames,
                fps=fps,
                threshold=float(threshold),
                below_idx=current_end,
                above_idx=prev_end_idx,
            )
            next_start_time_s, next_start_valid = _threshold_crossing_time_s(
                speed=speed,
                frames=frames,
                fps=fps,
                threshold=float(threshold),
                below_idx=next_start - 1,
                above_idx=next_start,
            )
            if prev_end_valid and next_start_valid:
                gap_s = float(next_start_time_s - prev_end_time_s)
                should_merge = gap_s < float(min_gap_duration_s)
            else:
                should_merge = (next_start - current_end) < int(min_gap_frames)

        if should_merge:
            current_end = next_end
        else:
            merged_ends.append(current_end)
            merged_starts.append(next_start)
            current_end = next_end

    merged_ends.append(current_end)
    return np.asarray(merged_starts, dtype=np.int64), np.asarray(merged_ends, dtype=np.int64)


def _nearest_global_minimum_index(values: np.ndarray, *, prefer_last: bool) -> int:
    """Return index of a finite global minimum, tie-broken toward the core."""

    finite = np.isfinite(values)
    if not finite.any():
        return values.size - 1 if prefer_last else 0
    minimum = np.nanmin(values[finite])
    candidate_indices = np.flatnonzero(finite & (values == minimum))
    if candidate_indices.size == 0:
        return values.size - 1 if prefer_last else 0
    return int(candidate_indices[-1] if prefer_last else candidate_indices[0])


def _expand_core_boundaries_to_local_minima(
    speed: np.ndarray,
    core_starts: np.ndarray,
    core_ends_exclusive: np.ndarray,
    *,
    window_frames: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Expand threshold/peak cores to nearby low-speed local minima.

    Cores remain the authoritative threshold-crossing segments. Expanded
    boundaries are bounded by adjacent cores and previously expanded bouts so
    they cannot overlap.
    """

    if core_starts.size == 0:
        return core_starts.copy(), core_ends_exclusive.copy()
    if window_frames <= 0:
        return core_starts.copy(), core_ends_exclusive.copy()

    speed_for_min = np.asarray(speed, dtype=np.float64)
    n_samples = int(speed_for_min.size)
    expanded_starts = np.zeros_like(core_starts)
    expanded_ends_exclusive = np.zeros_like(core_ends_exclusive)
    previous_expanded_end = 0

    for idx, (core_start, core_end_exclusive) in enumerate(zip(core_starts, core_ends_exclusive)):
        core_start = int(core_start)
        core_end_exclusive = int(core_end_exclusive)
        next_core_start = int(core_starts[idx + 1]) if idx + 1 < core_starts.size else n_samples

        left_start = max(0, core_start - window_frames, previous_expanded_end)
        left_stop = min(core_start + 1, n_samples)
        if left_start < left_stop:
            local_offset = _nearest_global_minimum_index(
                speed_for_min[left_start:left_stop],
                prefer_last=True,
            )
            expanded_start = left_start + local_offset
        else:
            expanded_start = core_start

        right_start = max(0, min(core_end_exclusive - 1, n_samples - 1))
        right_stop = min(n_samples, core_end_exclusive + window_frames, next_core_start)
        if right_start < right_stop:
            local_offset = _nearest_global_minimum_index(
                speed_for_min[right_start:right_stop],
                prefer_last=False,
            )
            expanded_end_exclusive = right_start + local_offset + 1
        else:
            expanded_end_exclusive = core_end_exclusive

        expanded_end_exclusive = max(expanded_end_exclusive, core_end_exclusive)
        expanded_start = min(expanded_start, core_start)

        expanded_starts[idx] = expanded_start
        expanded_ends_exclusive[idx] = expanded_end_exclusive
        previous_expanded_end = int(expanded_end_exclusive)

    return expanded_starts, expanded_ends_exclusive


def _build_bout_array(
    speed: np.ndarray,
    frames: np.ndarray,
    fps: float,
    core_starts: np.ndarray,
    core_ends_exclusive: np.ndarray,
    starts: np.ndarray,
    ends_exclusive: np.ndarray,
    *,
    path_distance_mm: Optional[np.ndarray] = None,
    path_distance_px: Optional[np.ndarray] = None,
    positions_mm: Optional[np.ndarray] = None,
    positions_px: Optional[np.ndarray] = None,
    physical_speed_mm: Optional[np.ndarray] = None,
    delta_seconds: Optional[np.ndarray] = None,
    transition_valid: Optional[np.ndarray] = None,
    sample_valid: Optional[np.ndarray] = None,
    threshold: Optional[float] = None,
) -> np.ndarray:
    """Build the structured bout table from core and expanded boundaries."""

    n_bouts = len(starts)
    bouts = np.zeros(n_bouts, dtype=_bout_dtype())
    if delta_seconds is None:
        delta_seconds_arr = _transition_delta_seconds(frames, fps)
    else:
        delta_seconds_arr = np.asarray(delta_seconds, dtype=np.float64)
        if delta_seconds_arr.shape[0] != frames.shape[0]:
            raise ValueError(
                f"delta_seconds length {delta_seconds_arr.shape[0]} does not match frames length {frames.shape[0]}."
            )
    effective_transition_valid = _effective_transition_valid(
        speed=speed,
        frames=frames,
        fps=fps,
        transition_valid=transition_valid,
        sample_valid=sample_valid,
    )

    for i, (start_idx, end_exclusive, core_start_idx, core_end_exclusive) in enumerate(
        zip(starts, ends_exclusive, core_starts, core_ends_exclusive)
    ):
        start_idx = int(start_idx)
        end_exclusive = int(end_exclusive)
        core_start_idx = int(core_start_idx)
        core_end_exclusive = int(core_end_exclusive)

        bout_detection_signal = speed[start_idx:end_exclusive]
        valid_detection_signal = bout_detection_signal[np.isfinite(bout_detection_signal)]
        if physical_speed_mm is not None:
            physical_speed_slice = np.asarray(physical_speed_mm, dtype=np.float64)[start_idx:end_exclusive]
        else:
            physical_speed_slice = bout_detection_signal
        valid_physical_speeds = physical_speed_slice[np.isfinite(physical_speed_slice)]
        transition_slice = slice(start_idx, end_exclusive)
        valid_transition_mask = effective_transition_valid[transition_slice]

        duration_frames = end_exclusive - start_idx
        duration_s = duration_frames / fps if fps > 0 else 0.0
        elapsed_duration_s = duration_s
        observed_duration_s = float(
            np.sum(delta_seconds_arr[transition_slice][valid_transition_mask])
        )
        core_duration_frames = core_end_exclusive - core_start_idx
        core_duration_s = core_duration_frames / fps if fps > 0 else 0.0
        n_possible_transitions = int(max(0, end_exclusive - start_idx))
        n_valid_transitions = int(np.sum(valid_transition_mask))
        n_invalid_transitions = int(max(0, n_possible_transitions - n_valid_transitions))
        valid_transition_fraction = (
            float(n_valid_transitions / n_possible_transitions)
            if n_possible_transitions > 0
            else float("nan")
        )
        gap_censored = bool(n_invalid_transitions > 0)

        path_length_mm = _sum_valid_path_distance(
            None if path_distance_mm is None else path_distance_mm[transition_slice],
            valid_transition_mask,
        )
        path_length_px = _sum_valid_path_distance(
            None if path_distance_px is None else path_distance_px[transition_slice],
            valid_transition_mask,
        )
        mean_speed_mm_s = (
            path_length_mm / observed_duration_s
            if observed_duration_s > 0.0 and np.isfinite(path_length_mm)
            else float("nan")
        )
        peak_detection_signal_mm_s = (
            float(np.max(valid_detection_signal)) if len(valid_detection_signal) > 0 else float("nan")
        )
        peak_physical_speed_mm_s = (
            float(np.max(valid_physical_speeds)) if len(valid_physical_speeds) > 0 else float("nan")
        )

        end_idx = end_exclusive - 1
        core_end_idx = core_end_exclusive - 1
        (
            core_start_time_s_interpolated,
            core_start_time_interpolated_valid,
            core_end_time_s_interpolated,
            core_end_time_interpolated_valid,
            core_duration_s_interpolated,
        ) = _core_threshold_crossing_times_s(
            speed=speed,
            frames=frames,
            fps=fps,
            threshold=threshold,
            core_start_idx=core_start_idx,
            core_end_exclusive=core_end_exclusive,
        )
        bouts[i] = (
            i + 1,
            int(frames[start_idx]),
            int(frames[end_idx]),
            int(frames[core_start_idx]),
            int(frames[core_end_idx]),
            duration_frames,
            duration_s,
            elapsed_duration_s,
            observed_duration_s,
            core_duration_frames,
            core_duration_s,
            path_length_mm,
            path_length_px,
            _net_displacement(positions_mm, start_idx, end_idx),
            _net_displacement(positions_px, start_idx, end_idx),
            mean_speed_mm_s,
            peak_detection_signal_mm_s,
            peak_physical_speed_mm_s,
            n_valid_transitions,
            n_invalid_transitions,
            valid_transition_fraction,
            gap_censored,
            frames[start_idx] / fps if fps > 0 else float('nan'),
            frames[end_idx] / fps if fps > 0 else float('nan'),
            frames[core_start_idx] / fps if fps > 0 else float('nan'),
            frames[core_end_idx] / fps if fps > 0 else float('nan'),
            core_start_time_s_interpolated,
            core_end_time_s_interpolated,
            core_duration_s_interpolated,
            core_start_time_interpolated_valid,
            core_end_time_interpolated_valid,
        )

    return bouts


def _compute_inter_bout_intervals(bouts: np.ndarray, fps: float) -> Tuple[np.ndarray, Dict[str, float], np.ndarray]:
    """
    Compute inter-bout intervals (gaps between consecutive bouts).

    Returns:
        Tuple of (intervals array, summary metrics dict, histogram array)
    """
    interval_dtype = np.dtype([
        ('prev_bout_id', 'i4'),
        ('next_bout_id', 'i4'),
        ('prev_end_frame', 'i8'),
        ('next_start_frame', 'i8'),
        ('interval_frames', 'i8'),
        ('prev_end_time_s', 'f8'),
        ('next_start_time_s', 'f8'),
        ('interval_s', 'f8'),
    ])

    histogram_dtype = np.dtype([
        ('bin_left_edge_s', 'f8'),
        ('bin_right_edge_s', 'f8'),
        ('count', 'i8'),
    ])

    if len(bouts) < 2:
        empty_intervals = np.zeros(0, dtype=interval_dtype)
        empty_histogram = np.zeros(0, dtype=histogram_dtype)
        metrics = {
            'inter_bout_interval_count': 0,
            'inter_bout_interval_mean_s': float('nan'),
            'inter_bout_interval_std_s': float('nan'),
            'inter_bout_interval_median_s': float('nan'),
            'inter_bout_interval_min_s': float('nan'),
            'inter_bout_interval_max_s': float('nan'),
        }
        return empty_intervals, metrics, empty_histogram

    # Sort by start time
    sorted_indices = np.argsort(bouts['start_time_s'])
    sorted_bouts = bouts[sorted_indices]

    n_intervals = len(sorted_bouts) - 1
    intervals = np.zeros(n_intervals, dtype=interval_dtype)

    for idx in range(n_intervals):
        prev_bout = sorted_bouts[idx]
        next_bout = sorted_bouts[idx + 1]

        raw_gap_frames = int(next_bout['start_frame']) - int(prev_bout['end_frame'])
        gap_frames = max(0, raw_gap_frames)

        raw_gap_seconds = float(next_bout['start_time_s']) - float(prev_bout['end_time_s'])
        gap_seconds = max(0.0, raw_gap_seconds)

        if not np.isfinite(gap_seconds) and fps > 0:
            gap_seconds = gap_frames / fps
        elif fps > 0 and gap_frames and not np.isclose(gap_seconds, gap_frames / fps):
            # Prefer frame-derived duration when timestamps/frames disagree
            gap_seconds = gap_frames / fps

        intervals[idx] = (
            int(prev_bout['bout_id']),
            int(next_bout['bout_id']),
            int(prev_bout['end_frame']),
            int(next_bout['start_frame']),
            gap_frames,
            float(prev_bout['end_time_s']),
            float(next_bout['start_time_s']),
            gap_seconds,
        )

    interval_values = intervals['interval_s']
    metrics = {
        'inter_bout_interval_count': int(n_intervals),
        'inter_bout_interval_mean_s': float(interval_values.mean()),
        'inter_bout_interval_std_s': float(interval_values.std()),
        'inter_bout_interval_median_s': float(np.median(interval_values)),
        'inter_bout_interval_min_s': float(interval_values.min()),
        'inter_bout_interval_max_s': float(interval_values.max()),
    }

    # Create histogram
    hist_counts, hist_edges = np.histogram(interval_values, bins='auto')
    histogram = np.zeros(hist_counts.size, dtype=histogram_dtype)
    if hist_counts.size:
        histogram['bin_left_edge_s'] = hist_edges[:-1]
        histogram['bin_right_edge_s'] = hist_edges[1:]
        histogram['count'] = hist_counts.astype('i8')

    return intervals, metrics, histogram


def _create_bout_points(
    bouts: np.ndarray,
    positions_mm: Optional[np.ndarray],
    positions_px: Optional[np.ndarray],
    frames: np.ndarray,
    fps: float
) -> np.ndarray:
    """
    Create bout_points dataset with start/end positions for each bout.

    Args:
        bouts: Bout array with start/end frames
        positions_mm: Position array (N, 2) with x,y positions in mm (or None)
        positions_px: Position array (N, 2) with x,y positions in pixels (or None)
        frames: Frame indices corresponding to positions
        fps: Frames per second

    Returns:
        Structured array with bout start/end points (includes both px and mm)
    """
    point_dtype = np.dtype([
        ('bout_id', 'i4'),
        ('point_type', 'S5'),  # b'start' or b'end'
        ('frame', 'i8'),
        ('time_s', 'f8'),
        ('x_px', 'f8'),
        ('y_px', 'f8'),
        ('x_mm', 'f8'),
        ('y_mm', 'f8'),
    ])

    if len(bouts) == 0 or (positions_mm is None and positions_px is None):
        return np.zeros(0, dtype=point_dtype)

    point_array = np.zeros(len(bouts) * 2, dtype=point_dtype)

    for idx, bout in enumerate(bouts):
        start_frame = bout['start_frame']
        end_frame = bout['end_frame']

        # Find positions at start and end frames
        start_idx = np.where(frames == start_frame)[0]
        end_idx = np.where(frames == end_frame)[0]

        # Extract pixel positions
        if positions_px is not None:
            start_x_px = positions_px[start_idx[0], 0] if len(start_idx) > 0 else float('nan')
            start_y_px = positions_px[start_idx[0], 1] if len(start_idx) > 0 else float('nan')
            end_x_px = positions_px[end_idx[0], 0] if len(end_idx) > 0 else float('nan')
            end_y_px = positions_px[end_idx[0], 1] if len(end_idx) > 0 else float('nan')
        else:
            start_x_px = end_x_px = start_y_px = end_y_px = float('nan')

        # Extract mm positions
        if positions_mm is not None:
            start_x_mm = positions_mm[start_idx[0], 0] if len(start_idx) > 0 else float('nan')
            start_y_mm = positions_mm[start_idx[0], 1] if len(start_idx) > 0 else float('nan')
            end_x_mm = positions_mm[end_idx[0], 0] if len(end_idx) > 0 else float('nan')
            end_y_mm = positions_mm[end_idx[0], 1] if len(end_idx) > 0 else float('nan')
        else:
            start_x_mm = end_x_mm = start_y_mm = end_y_mm = float('nan')

        start_point_idx = idx * 2
        end_point_idx = start_point_idx + 1

        point_array[start_point_idx] = (
            int(bout['bout_id']),
            b'start',
            int(start_frame),
            float(bout['start_time_s']),
            start_x_px,
            start_y_px,
            start_x_mm,
            start_y_mm,
        )

        point_array[end_point_idx] = (
            int(bout['bout_id']),
            b'end',
            int(end_frame),
            float(bout['end_time_s']),
            end_x_px,
            end_y_px,
            end_x_mm,
            end_y_mm,
        )

    return point_array


def _compute_global_metrics(bouts: np.ndarray, fps: float, total_frames: int) -> np.ndarray:
    """
    Compute global bout metrics.

    Returns:
        Structured array with a single row containing global metrics
    """
    global_dtype = np.dtype([
        ('n_bouts', 'i4'),
        ('bout_rate_per_min', 'f8'),
        ('total_active_time_s', 'f8'),
        ('total_observed_active_time_s', 'f8'),
        ('percent_active', 'f8'),
        ('mean_bout_duration_s', 'f8'),
        ('mean_bout_observed_duration_s', 'f8'),
        ('mean_bout_path_length_mm', 'f8'),
        ('total_path_length_mm', 'f8'),
        ('mean_bout_speed_mm_s', 'f8'),
        ('mean_bout_peak_detection_signal_mm_s', 'f8'),
        ('mean_bout_peak_physical_speed_mm_s', 'f8'),
        ('mean_valid_transition_fraction', 'f8'),
        ('n_gap_censored_bouts', 'i4'),
        ('inter_bout_interval_count', 'i4'),
        ('inter_bout_interval_mean_s', 'f8'),
        ('inter_bout_interval_std_s', 'f8'),
        ('inter_bout_interval_median_s', 'f8'),
        ('inter_bout_interval_min_s', 'f8'),
        ('inter_bout_interval_max_s', 'f8'),
    ])

    global_metrics = np.zeros(1, dtype=global_dtype)

    n_bouts = len(bouts)
    total_duration_s = total_frames / fps if fps > 0 else 0.0

    if n_bouts > 0:
        total_active_time = float(np.sum(bouts['duration_s']))
        total_observed_active_time = float(np.sum(bouts['observed_duration_s']))
        percent_active = (total_active_time / total_duration_s * 100.0) if total_duration_s > 0 else 0.0
        bout_rate_per_min = (n_bouts / total_duration_s * 60.0) if total_duration_s > 0 else 0.0

        global_metrics[0] = (
            n_bouts,
            bout_rate_per_min,
            total_active_time,
            total_observed_active_time,
            percent_active,
            float(np.mean(bouts['duration_s'])),
            float(np.mean(bouts['observed_duration_s'])),
            _finite_mean_or_nan(bouts['path_length_mm']),
            _finite_sum_or_nan(bouts['path_length_mm']),
            _finite_mean_or_nan(bouts['mean_speed_mm_s']),
            _finite_mean_or_nan(bouts['peak_detection_signal_mm_s']),
            _finite_mean_or_nan(bouts['peak_physical_speed_mm_s']),
            _finite_mean_or_nan(bouts['valid_transition_fraction']),
            int(np.sum(bouts['gap_censored'])),
            0,  # Will be updated with interval metrics
            float('nan'),
            float('nan'),
            float('nan'),
            float('nan'),
            float('nan'),
        )
    else:
        global_metrics[0] = (
            0, 0.0, 0.0, 0.0, 0.0, float('nan'), float('nan'), float('nan'),
            float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), 0,
            0, float('nan'), float('nan'), float('nan'), float('nan'), float('nan'),
        )

    return global_metrics


def _bytes_dtype(values: List[object], *, minimum: int = 16) -> str:
    max_len = minimum
    for value in values:
        max_len = max(max_len, len(str(value).encode("utf-8")))
    return f"S{max_len}"


def _add_fields(records: np.ndarray, fields: Mapping[str, tuple[Any, Any]]) -> np.ndarray:
    """Return a structured array with extra fields prepended."""

    existing_names = records.dtype.names or ()
    dtype = [(name, np.dtype(dtype)) for name, (dtype, _value) in fields.items()]
    dtype.extend((name, records.dtype.fields[name][0]) for name in existing_names if name not in fields)
    output = np.zeros(records.shape[0], dtype=dtype)
    for name, (_dtype, value) in fields.items():
        arr = np.asarray(value)
        if arr.shape == ():
            output[name] = value
        else:
            output[name] = arr
    for name in existing_names:
        if name not in fields:
            output[name] = records[name]
    return output


def _concatenate_structured(records: List[np.ndarray], dtype: Optional[np.dtype] = None) -> np.ndarray:
    non_empty = [record for record in records if record.size > 0]
    if non_empty:
        return np.concatenate(non_empty)
    if dtype is not None:
        return np.zeros(0, dtype=dtype)
    if records:
        return np.zeros(0, dtype=records[0].dtype)
    return np.zeros(0, dtype=[])


def _metric_units(metric_name: str) -> str:
    if metric_name.endswith("_s") or "_time_s" in metric_name or "_duration_s" in metric_name:
        return "s"
    if metric_name.endswith("_mm") or "path_length_mm" in metric_name:
        return "mm"
    if metric_name.endswith("_mm_s") or "speed_mm_s" in metric_name:
        return "mm/s"
    if metric_name.endswith("_percent") or metric_name == "percent_active":
        return "percent"
    if metric_name.startswith("n_") or metric_name.endswith("_count"):
        return "count"
    return ""


def _summary_metrics_rows(
    global_metrics_by_level: Mapping[str, np.ndarray],
    *,
    signal_id_by_level: Mapping[str, int],
) -> np.ndarray:
    rows: list[tuple[int, int, bytes, float, bytes, bytes]] = []
    metric_names: list[str] = []
    units: list[str] = []
    for level, metrics in global_metrics_by_level.items():
        if metrics.size == 0 or metrics.dtype.names is None:
            continue
        signal_id = int(signal_id_by_level[level])
        record = metrics[0]
        for name in metrics.dtype.names:
            value = record[name]
            try:
                numeric_value = float(value)
            except (TypeError, ValueError):
                continue
            metric_names.append(name)
            unit = _metric_units(name)
            units.append(unit)
            rows.append((0, signal_id, name.encode("utf-8"), numeric_value, unit.encode("utf-8"), b"global_metrics"))
    dtype = np.dtype([
        ("candidate_id", "i4"),
        ("signal_id", "i4"),
        ("metric_name", _bytes_dtype(metric_names, minimum=32)),
        ("value", "f8"),
        ("units", _bytes_dtype(units, minimum=8)),
        ("source_table", "S32"),
    ])
    result = np.zeros(len(rows), dtype=dtype)
    for idx, row in enumerate(rows):
        result[idx] = row
    return result


def _compact_bout_rows(
    bouts: np.ndarray,
    peak_events: np.ndarray,
    *,
    candidate_id: int,
    signal_id: int,
    estimator_signal_id: int,
    track_id: int,
    pixel_to_mm: Optional[float],
) -> np.ndarray:
    peak_by_bout: dict[int, np.void] = {}
    if peak_events.dtype.names is not None and "bout_id" in peak_events.dtype.names:
        for event in peak_events:
            peak_by_bout.setdefault(int(event["bout_id"]), event)

    n_rows = bouts.shape[0]
    peak_frame = np.full(n_rows, -1, dtype=np.int64)
    peak_time_s = np.full(n_rows, np.nan, dtype=np.float64)
    peak_detection_signal_px_s = np.full(n_rows, np.nan, dtype=np.float64)
    threshold_crossing_valid = np.zeros(n_rows, dtype=bool)
    mean_speed_px_s = np.full(n_rows, np.nan, dtype=np.float64)
    if n_rows and bouts.dtype.names is not None:
        if {"path_length_px", "observed_duration_s"}.issubset(set(bouts.dtype.names)):
            valid = np.asarray(bouts["observed_duration_s"], dtype=np.float64) > 0
            mean_speed_px_s[valid] = (
                np.asarray(bouts["path_length_px"], dtype=np.float64)[valid]
                / np.asarray(bouts["observed_duration_s"], dtype=np.float64)[valid]
            )
        if {"core_start_time_interpolated_valid", "core_end_time_interpolated_valid"}.issubset(set(bouts.dtype.names)):
            threshold_crossing_valid = (
                np.asarray(bouts["core_start_time_interpolated_valid"], dtype=bool)
                & np.asarray(bouts["core_end_time_interpolated_valid"], dtype=bool)
            )
        for idx, bout in enumerate(bouts):
            event = peak_by_bout.get(int(bout["bout_id"]))
            if event is not None:
                if "peak_frame" in event.dtype.names:
                    peak_frame[idx] = int(event["peak_frame"])
                if "peak_time_s" in event.dtype.names:
                    peak_time_s[idx] = float(event["peak_time_s"])
            if pixel_to_mm and np.isfinite(pixel_to_mm) and pixel_to_mm > 0 and "peak_detection_signal_mm_s" in bouts.dtype.names:
                peak_detection_signal_px_s[idx] = float(bouts["peak_detection_signal_mm_s"][idx]) / float(pixel_to_mm)

    return _add_fields(
        bouts,
        {
            "candidate_id": ("i4", int(candidate_id)),
            "signal_id": ("i4", int(signal_id)),
            "estimator_signal_id": ("i4", int(estimator_signal_id)),
            "track_id": ("i4", int(track_id)),
            "mean_speed_px_s": ("f8", mean_speed_px_s),
            "peak_detection_signal_px_s": ("f8", peak_detection_signal_px_s),
            "peak_frame": ("i8", peak_frame),
            "peak_time_s": ("f8", peak_time_s),
            "threshold_crossing_valid": ("?", threshold_crossing_valid),
        },
    )


def _compact_peak_event_rows(
    peak_events: np.ndarray,
    *,
    candidate_id: int,
    signal_id: int,
) -> np.ndarray:
    return _add_fields(
        peak_events,
        {
            "peak_event_id": ("i8", np.arange(peak_events.shape[0], dtype=np.int64)),
            "candidate_id": ("i4", int(candidate_id)),
            "signal_id": ("i4", int(signal_id)),
            "accepted": ("?", True),
            "rejection_reason": ("S16", b""),
        },
    )


def _compact_interval_rows(
    intervals: np.ndarray,
    *,
    candidate_id: int,
    signal_id: int,
) -> np.ndarray:
    return _add_fields(
        intervals,
        {
            "interval_id": ("i8", np.arange(intervals.shape[0], dtype=np.int64)),
            "candidate_id": ("i4", int(candidate_id)),
            "signal_id": ("i4", int(signal_id)),
            "valid": ("?", True),
        },
    )


def _compact_histogram_rows(
    histogram: np.ndarray,
    *,
    candidate_id: int,
    signal_id: int,
) -> np.ndarray:
    density = np.full(histogram.shape[0], np.nan, dtype=np.float64)
    return _add_fields(
        histogram,
        {
            "candidate_id": ("i4", int(candidate_id)),
            "signal_id": ("i4", int(signal_id)),
            "metric_name": ("S32", b"inter_bout_interval_s"),
            "bin_left": ("f8", histogram["bin_left_edge_s"] if "bin_left_edge_s" in histogram.dtype.names else np.nan),
            "bin_right": ("f8", histogram["bin_right_edge_s"] if "bin_right_edge_s" in histogram.dtype.names else np.nan),
            "density": ("f8", density),
            "units": ("S8", b"s"),
        },
    )


def _compact_bout_point_rows(
    bout_points: np.ndarray,
    *,
    candidate_id: int,
    signal_id: int,
) -> np.ndarray:
    point_role = bout_points["point_type"] if bout_points.size and "point_type" in bout_points.dtype.names else b""
    return _add_fields(
        bout_points,
        {
            "candidate_id": ("i4", int(candidate_id)),
            "signal_id": ("i4", int(signal_id)),
            "point_role": ("S8", point_role),
        },
    )


def _write_compact_v2_swim_bout_payloads(
    run_group: zarr.Group,
    *,
    run_name: str,
    speed_levels: List[str],
    level_payloads: Mapping[str, Mapping[str, Any]],
    signal_id_by_level: Mapping[str, int],
    estimator_signal_id_by_level: Mapping[str, int],
    default_level_key: str,
    method: str,
    parameters: Mapping[str, Any],
    provenance: Mapping[str, Any],
    track_id: int,
    pixel_to_mm: Optional[float],
    path_distance_level_source: Mapping[str, str],
    source_track_path: str,
    exponential_source_key: str,
    exponential_tau_s: float,
    frames: np.ndarray,
    speeds: Mapping[str, Optional[np.ndarray]],
) -> None:
    """Write the compact tabular v2 swim-bout representation."""

    indexes = run_group.create_group("indexes")
    tables = run_group.create_group("tables")
    signals_group = run_group.create_group("signals")

    parameters_json = _strict_json_dumps(parameters)
    candidate_dtype = np.dtype([
        ("candidate_id", "i4"),
        ("candidate_name", _bytes_dtype([run_name], minimum=32)),
        ("is_default", "?"),
        ("detection_method", "S32"),
        ("boundary_mode", "S32"),
        ("boundary_window_s", "f8"),
        ("boundary_constraint", "S32"),
        ("gap_merge_policy", "S32"),
        ("min_bout_duration_s", "f8"),
        ("min_gap_duration_s", "f8"),
        ("min_gap_frames", "i4"),
        ("parameter_hash", "S64"),
        ("parameters_json", _bytes_dtype([parameters_json], minimum=256)),
        ("provenance_json", "S1"),
    ])
    candidates = np.zeros(1, dtype=candidate_dtype)
    boundary_window_value = parameters.get("boundary_window_s")
    min_bout_duration_value = parameters.get("min_bout_duration_s")
    min_gap_duration_value = parameters.get("min_gap_duration_s")
    candidates[0] = (
        0,
        run_name.encode("utf-8"),
        True,
        str(method).encode("utf-8"),
        str(parameters.get("boundary_mode", "")).encode("utf-8"),
        float(boundary_window_value) if boundary_window_value is not None else float("nan"),
        b"",
        str(parameters.get("gap_merge_policy", "")).encode("utf-8"),
        float(min_bout_duration_value) if min_bout_duration_value is not None else float("nan"),
        float(min_gap_duration_value) if min_gap_duration_value is not None else float("nan"),
        int(parameters.get("min_gap_frames")) if parameters.get("min_gap_frames") is not None else -1,
        _sha256_json(parameters).encode("utf-8"),
        parameters_json.encode("utf-8"),
        b"",
    )

    signal_rows = []
    signal_parameters_json: list[str] = []
    for level in speed_levels:
        signal_id = int(signal_id_by_level[level])
        attrs = _detection_signal_attrs(
            level=level,
            source_track_path=source_track_path,
            path_distance_source_level=path_distance_level_source[level],
            exponential_source_key=exponential_source_key,
            exponential_tau_s=float(exponential_tau_s),
        )
        signal_params = {
            "speed_level": level,
            "signal_id": signal_id,
            "path_distance_source_level": path_distance_level_source[level],
            "detection_signal_attrs": attrs,
        }
        signal_json = _strict_json_dumps(signal_params)
        signal_parameters_json.append(signal_json)
        is_exponential = level == "speed_exponential"
        source_level = exponential_source_key if is_exponential else level
        transform_source_signal_id = (
            int(signal_id_by_level.get(exponential_source_key, -1)) if is_exponential else -1
        )
        signal_rows.append(
            (
                signal_id,
                level.encode("utf-8"),
                level.replace("speed_", "", 1).encode("utf-8"),
                (b"detector_response" if is_exponential else b"physical_estimator"),
                source_level.encode("utf-8"),
                (b"exponential" if is_exponential else b"identity"),
                transform_source_signal_id,
                float(exponential_tau_s) if is_exponential else float("nan"),
                b"mm/s",
                path_distance_level_source[level].encode("utf-8"),
                signal_json.encode("utf-8"),
            )
        )
    signal_dtype = np.dtype([
        ("signal_id", "i4"),
        ("speed_level", "S32"),
        ("signal_name", "S32"),
        ("role", "S32"),
        ("source_level", "S32"),
        ("transform_type", "S32"),
        ("transform_source_signal_id", "i4"),
        ("tau_s", "f8"),
        ("units", "S16"),
        ("path_distance_source_level", "S32"),
        ("parameters_json", _bytes_dtype(signal_parameters_json, minimum=256)),
    ])
    signal_variants = np.zeros(len(signal_rows), dtype=signal_dtype)
    for idx, row in enumerate(signal_rows):
        signal_variants[idx] = row

    compact_bouts = []
    compact_peak_events = []
    compact_intervals = []
    compact_histograms = []
    compact_bout_points = []
    global_metrics_by_level: dict[str, np.ndarray] = {}
    for level in speed_levels:
        payload = level_payloads[level]
        signal_id = int(signal_id_by_level[level])
        estimator_signal_id = int(estimator_signal_id_by_level[level])
        compact_bouts.append(
            _compact_bout_rows(
                payload["bouts"],
                payload["peak_events"],
                candidate_id=0,
                signal_id=signal_id,
                estimator_signal_id=estimator_signal_id,
                track_id=track_id,
                pixel_to_mm=pixel_to_mm,
            )
        )
        compact_peak_events.append(
            _compact_peak_event_rows(payload["peak_events"], candidate_id=0, signal_id=signal_id)
        )
        compact_intervals.append(
            _compact_interval_rows(payload["intervals"], candidate_id=0, signal_id=signal_id)
        )
        compact_histograms.append(
            _compact_histogram_rows(payload["interval_histogram"], candidate_id=0, signal_id=signal_id)
        )
        compact_bout_points.append(
            _compact_bout_point_rows(payload["bout_points"], candidate_id=0, signal_id=signal_id)
        )
        global_metrics_by_level[level] = payload["global_metrics"]

    write_columnar_dataset(indexes, "candidates", candidates)
    write_columnar_dataset(indexes, "signal_variants", signal_variants)
    write_columnar_dataset(tables, "bouts", _concatenate_structured(compact_bouts))
    write_columnar_dataset(tables, "peak_events", _concatenate_structured(compact_peak_events))
    write_columnar_dataset(tables, "inter_bout_intervals", _concatenate_structured(compact_intervals))
    write_columnar_dataset(
        tables,
        "summary_metrics",
        _summary_metrics_rows(global_metrics_by_level, signal_id_by_level=signal_id_by_level),
    )
    write_columnar_dataset(tables, "histograms", _concatenate_structured(compact_histograms))
    write_columnar_dataset(tables, "bout_points", _concatenate_structured(compact_bout_points))

    detector_levels = [level for level in speed_levels if level == "speed_exponential"]
    detector_rows = []
    detector_signal_ids = []
    for level in detector_levels:
        arr = speeds.get(f"{level}_mm")
        if arr is None:
            continue
        detector_rows.append(np.asarray(arr, dtype=np.float32))
        detector_signal_ids.append(int(signal_id_by_level[level]))
    if detector_rows:
        store_array(
            signals_group,
            "detector_signal_mm_s",
            np.stack(detector_rows, axis=0),
            attrs={
                "units": "mm/s",
                "axis_0": "detector_signal_id",
                "axis_1": "frame",
            },
        )
        store_array(
            signals_group,
            "detector_signal_signal_ids",
            np.asarray(detector_signal_ids, dtype=np.int32),
        )
        store_array(
            signals_group,
            "frame_indices",
            np.asarray(frames, dtype=np.int64),
            attrs={"source": "track_kinematics.frame_indices"},
        )

    run_group.attrs.update(
        _json_safe_attrs(
            {
                "layout": SWIM_BOUT_STORED_LAYOUT_COMPACT_V2,
                "default_candidate_id": 0,
                "default_signal_id": int(signal_id_by_level[default_level_key]),
                "compact_writer": "detect_bouts_multi_level",
                "compact_writer_opt_in": True,
                "compact_tables": [
                    "bouts",
                    "peak_events",
                    "inter_bout_intervals",
                    "summary_metrics",
                    "histograms",
                    "bout_points",
                ],
            }
        )
    )


def _detect_bouts_from_speed(
    speed: np.ndarray,
    frames: np.ndarray,
    fps: float,
    threshold: float,
    min_bout_duration_s: float = 0.05,
    min_gap_duration_s: float = 0.1,
    min_gap_frames: Optional[int] = None,
    gap_merge_policy: str = "sampled_frame_gap",
    boundary_mode: str = "threshold",
    boundary_window_s: float = 0.25,
    path_distance_mm: Optional[np.ndarray] = None,
    path_distance_px: Optional[np.ndarray] = None,
    positions_mm: Optional[np.ndarray] = None,
    positions_px: Optional[np.ndarray] = None,
    physical_speed_mm: Optional[np.ndarray] = None,
    delta_seconds: Optional[np.ndarray] = None,
    transition_valid: Optional[np.ndarray] = None,
    sample_valid: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Detect swim bouts from speed trace using threshold-based detection.

    Args:
        speed: Speed array (in same units as threshold)
        frames: Frame indices
        fps: Frames per second
        threshold: Speed threshold for bout detection
        min_bout_duration_s: Minimum duration to count as bout. Resolved to
            frames with ceiling.
        min_gap_duration_s: Minimum gap between bouts. Resolved to frames with
            ceiling when min_gap_frames is not provided.
        min_gap_frames: Explicit minimum gap in frames, overriding
            min_gap_duration_s when provided.
        gap_merge_policy: "sampled_frame_gap" merges by below-threshold frame
            count; "interpolated_core_gap" merges threshold-separated segments
            by interpolated core crossing times.
        boundary_mode: "threshold" keeps threshold-crossing boundaries;
            "local_minimum" expands start/end to nearby local minima while
            preserving core_* threshold boundaries.
        boundary_window_s: Maximum search window on each side for local-minimum
            boundary expansion.

    Returns:
        Structured array with bout data
    """
    if gap_merge_policy not in GAP_MERGE_POLICIES:
        expected = ", ".join(GAP_MERGE_POLICIES)
        raise ValueError(f"Unsupported gap_merge_policy {gap_merge_policy!r}; expected one of: {expected}")

    # Convert duration thresholds to frames
    min_bout_frames = _duration_seconds_to_frames(min_bout_duration_s, fps)
    resolved_min_gap_frames, _min_gap_frame_source = _resolve_min_gap_frames(
        min_gap_duration_s=min_gap_duration_s,
        fps=fps,
        min_gap_frames=min_gap_frames,
    )
    boundary_window_frames = max(0, int(round(boundary_window_s * fps)))

    # Find frames above threshold
    above_threshold = speed > threshold

    # Handle NaN values - treat as below threshold
    above_threshold[np.isnan(speed)] = False

    # Find transitions
    padded = np.concatenate([[False], above_threshold, [False]])
    diff = np.diff(padded.astype(int))

    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]

    merge_min_gap_duration_s, _merge_min_gap_source = _gap_merge_min_gap_duration_s(
        min_gap_duration_s=min_gap_duration_s,
        fps=fps,
        min_gap_frames=min_gap_frames,
    )
    starts, ends = _merge_threshold_segments(
        speed=speed,
        frames=frames,
        fps=fps,
        threshold=threshold,
        starts=starts,
        ends_exclusive=ends,
        min_gap_frames=resolved_min_gap_frames,
        min_gap_duration_s=merge_min_gap_duration_s,
        gap_merge_policy=gap_merge_policy,
    )

    core_starts = starts
    core_ends_exclusive = ends

    # Filter by minimum core duration
    durations = core_ends_exclusive - core_starts
    valid_bouts = durations >= min_bout_frames
    core_starts = core_starts[valid_bouts]
    core_ends_exclusive = core_ends_exclusive[valid_bouts]

    if boundary_mode == "local_minimum":
        starts, ends_exclusive = _expand_core_boundaries_to_local_minima(
            speed,
            core_starts,
            core_ends_exclusive,
            window_frames=boundary_window_frames,
        )
    elif boundary_mode == "threshold":
        starts = core_starts.copy()
        ends_exclusive = core_ends_exclusive.copy()
    else:
        raise ValueError(f"Unsupported boundary_mode: {boundary_mode!r}")

    return _build_bout_array(
        speed,
        frames,
        fps,
        core_starts,
        core_ends_exclusive,
        starts,
        ends_exclusive,
        path_distance_mm=path_distance_mm,
        path_distance_px=path_distance_px,
        positions_mm=positions_mm,
        positions_px=positions_px,
        physical_speed_mm=physical_speed_mm,
        delta_seconds=delta_seconds,
        transition_valid=transition_valid,
        sample_valid=sample_valid,
        threshold=threshold,
    )


def _detect_bouts_from_peaks(
    speed: np.ndarray,
    frames: np.ndarray,
    fps: float,
    prominence: float,
    min_peak_height: Optional[float] = None,
    rel_height: float = 0.9,
    min_bout_duration_s: float = 0.05,
    min_gap_duration_s: float = 0.1,
    min_gap_frames: Optional[int] = None,
    gap_merge_policy: str = "sampled_frame_gap",
    boundary_mode: str = "threshold",
    boundary_window_s: float = 0.25,
    path_distance_mm: Optional[np.ndarray] = None,
    path_distance_px: Optional[np.ndarray] = None,
    positions_mm: Optional[np.ndarray] = None,
    positions_px: Optional[np.ndarray] = None,
    physical_speed_mm: Optional[np.ndarray] = None,
    delta_seconds: Optional[np.ndarray] = None,
    transition_valid: Optional[np.ndarray] = None,
    sample_valid: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Detect swim bouts from speed trace using peak-based detection.

    Args:
        speed: Speed array (in same units as prominence/height)
        frames: Frame indices
        fps: Frames per second
        prominence: Minimum prominence for peak detection (relative height above baseline)
        min_peak_height: Minimum absolute peak height (optional)
        rel_height: Relative height for bout boundaries. Higher values (0.7-1.0) capture more
            of the bout tail; lower values (0.3-0.5) set tighter boundaries. Default 0.9 works
            well for asymmetric speed peaks with gradual tails.
        min_bout_duration_s: Minimum duration to count as bout. Resolved to
            frames with ceiling.
        min_gap_duration_s: Minimum gap between bouts. Resolved to frames with
            ceiling when min_gap_frames is not provided.
        min_gap_frames: Explicit minimum gap in frames, overriding
            min_gap_duration_s when provided.
        gap_merge_policy: Currently only "sampled_frame_gap" is supported for
            peak detection.
        boundary_mode: "threshold" keeps peak-width boundaries; "local_minimum"
            expands start/end to nearby local minima.
        boundary_window_s: Maximum search window on each side for local-minimum
            boundary expansion.

    Returns:
        Structured array with bout data
    """
    # Convert duration thresholds to frames
    min_bout_frames = _duration_seconds_to_frames(min_bout_duration_s, fps)
    resolved_min_gap_frames, _min_gap_frame_source = _resolve_min_gap_frames(
        min_gap_duration_s=min_gap_duration_s,
        fps=fps,
        min_gap_frames=min_gap_frames,
    )
    boundary_window_frames = max(0, int(round(boundary_window_s * fps)))
    if gap_merge_policy != "sampled_frame_gap":
        raise ValueError("Peak detection currently supports gap_merge_policy='sampled_frame_gap' only.")

    # Handle NaN values - replace with 0 for peak detection
    speed_clean = np.where(np.isnan(speed), 0.0, speed)

    # Find peaks with prominence criterion
    peak_kwargs = {'prominence': prominence}
    if min_peak_height is not None:
        peak_kwargs['height'] = min_peak_height

    peaks, properties = signal.find_peaks(speed_clean, **peak_kwargs)

    # If no peaks found, return empty array
    if len(peaks) == 0:
        return _empty_bouts()

    # Get bout boundaries using peak widths at relative height
    widths, width_heights, left_ips, right_ips = signal.peak_widths(
        speed_clean,
        peaks,
        rel_height=rel_height
    )

    # Convert interpolated positions to integer indices
    # Round down for start, round up for end to be inclusive
    starts = np.floor(left_ips).astype(int)
    ends_exclusive = np.ceil(right_ips).astype(int) + 1

    # Ensure indices are within bounds
    starts = np.clip(starts, 0, len(speed) - 1)
    ends_exclusive = np.clip(ends_exclusive, 1, len(speed))

    # Merge bouts separated by small gaps
    if len(starts) > 1:
        merged_starts = [starts[0]]
        merged_ends = []

        for i in range(1, len(starts)):
            gap = starts[i] - ends_exclusive[i-1]
            if gap < resolved_min_gap_frames:
                # Merge with previous bout (extend end)
                continue
            else:
                merged_ends.append(ends_exclusive[i-1])
                merged_starts.append(starts[i])

        merged_ends.append(ends_exclusive[-1])
        starts = np.array(merged_starts)
        ends_exclusive = np.array(merged_ends)

    core_starts = starts
    core_ends_exclusive = ends_exclusive

    # Filter by minimum core duration
    durations = core_ends_exclusive - core_starts
    valid_bouts = durations >= min_bout_frames
    core_starts = core_starts[valid_bouts]
    core_ends_exclusive = core_ends_exclusive[valid_bouts]

    if boundary_mode == "local_minimum":
        starts, ends_exclusive = _expand_core_boundaries_to_local_minima(
            speed,
            core_starts,
            core_ends_exclusive,
            window_frames=boundary_window_frames,
        )
    elif boundary_mode == "threshold":
        starts = core_starts.copy()
        ends_exclusive = core_ends_exclusive.copy()
    else:
        raise ValueError(f"Unsupported boundary_mode: {boundary_mode!r}")

    return _build_bout_array(
        speed,
        frames,
        fps,
        core_starts,
        core_ends_exclusive,
        starts,
        ends_exclusive,
        path_distance_mm=path_distance_mm,
        path_distance_px=path_distance_px,
        positions_mm=positions_mm,
        positions_px=positions_px,
        physical_speed_mm=physical_speed_mm,
        delta_seconds=delta_seconds,
        transition_valid=transition_valid,
        sample_valid=sample_valid,
    )


def _build_peak_event_array(
    *,
    peaks: np.ndarray,
    properties: Mapping[str, np.ndarray],
    widths: np.ndarray,
    width_heights: np.ndarray,
    left_ips: np.ndarray,
    right_ips: np.ndarray,
    frames: np.ndarray,
    speed: np.ndarray,
    fps: float,
    boundary_mode: str,
    shape_split_policy: str,
) -> np.ndarray:
    peak_events = np.zeros(peaks.size, dtype=_peak_event_dtype())
    if peaks.size == 0:
        return peak_events

    prominences = np.asarray(properties.get("prominences"), dtype=np.float64)
    left_bases = np.asarray(properties.get("left_bases"), dtype=np.int64)
    right_bases = np.asarray(properties.get("right_bases"), dtype=np.int64)
    if prominences.shape[0] != peaks.size:
        prominences = np.full(peaks.size, np.nan, dtype=np.float64)
    if left_bases.shape[0] != peaks.size:
        left_bases = np.full(peaks.size, -1, dtype=np.int64)
    if right_bases.shape[0] != peaks.size:
        right_bases = np.full(peaks.size, -1, dtype=np.int64)

    for idx, peak_index in enumerate(peaks):
        peak_index = int(peak_index)
        left_base = int(left_bases[idx])
        right_base = int(right_bases[idx])
        left_width_frame = _interpolated_frame_at_sample_index(frames, float(left_ips[idx]))
        right_width_frame = _interpolated_frame_at_sample_index(frames, float(right_ips[idx]))
        peak_width_s = (
            float((right_width_frame - left_width_frame) / fps)
            if fps > 0 and np.isfinite(left_width_frame) and np.isfinite(right_width_frame)
            else float("nan")
        )
        peak_events[idx] = (
            idx + 1,
            peak_index,
            int(frames[peak_index]),
            float(frames[peak_index] / fps) if fps > 0 else float("nan"),
            float(speed[peak_index]) if np.isfinite(speed[peak_index]) else float("nan"),
            float(prominences[idx]),
            float(widths[idx]) if idx < widths.size else float("nan"),
            peak_width_s,
            float(width_heights[idx]) if idx < width_heights.size else float("nan"),
            float(left_ips[idx]) if idx < left_ips.size else float("nan"),
            float(right_ips[idx]) if idx < right_ips.size else float("nan"),
            left_width_frame,
            right_width_frame,
            left_base,
            right_base,
            int(frames[left_base]) if 0 <= left_base < frames.size else -1,
            int(frames[right_base]) if 0 <= right_base < frames.size else -1,
            float(speed[left_base]) if 0 <= left_base < speed.size and np.isfinite(speed[left_base]) else float("nan"),
            float(speed[right_base]) if 0 <= right_base < speed.size and np.isfinite(speed[right_base]) else float("nan"),
            boundary_mode.encode("utf-8"),
            shape_split_policy.encode("utf-8"),
        )
    return peak_events


def _resolve_peak_event_boundaries(
    *,
    speed: np.ndarray,
    peaks: np.ndarray,
    left_ips: np.ndarray,
    right_ips: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    starts = np.floor(left_ips).astype(np.int64)
    ends_exclusive = np.ceil(right_ips).astype(np.int64) + 1
    starts = np.clip(starts, 0, max(0, speed.size - 1))
    ends_exclusive = np.clip(ends_exclusive, 1, speed.size)

    # Width estimates for adjacent peaks can overlap. Keep one event per peak
    # but split overlapping envelopes at the local valley between peak centers.
    for idx in range(1, peaks.size):
        previous_end = int(ends_exclusive[idx - 1])
        current_start = int(starts[idx])
        if current_start >= previous_end:
            continue
        left_peak = int(peaks[idx - 1])
        right_peak = int(peaks[idx])
        valley_start = min(left_peak, right_peak)
        valley_stop = max(left_peak, right_peak) + 1
        valley_values = np.asarray(speed[valley_start:valley_stop], dtype=np.float64)
        finite = np.isfinite(valley_values)
        if finite.any():
            local_indices = np.flatnonzero(finite)
            local_min_idx = int(local_indices[np.argmin(valley_values[finite])])
            split = valley_start + local_min_idx + 1
        else:
            split = int(round((left_peak + right_peak) / 2.0))
        split = max(int(peaks[idx - 1]) + 1, min(split, int(peaks[idx])))
        ends_exclusive[idx - 1] = split
        starts[idx] = split

    return starts, ends_exclusive


def _detect_bouts_from_peak_events(
    speed: np.ndarray,
    frames: np.ndarray,
    fps: float,
    *,
    min_peak_height_mm_s: Optional[float] = None,
    min_peak_prominence_mm_s: Optional[float] = 1.0,
    min_peak_distance_s: float = 0.05,
    peak_width_rel_height: float = 0.9,
    peak_event_boundary_mode: str = "relative_prominence_width",
    shape_split_policy: str = "none",
    min_bout_duration_s: float = 0.05,
    path_distance_mm: Optional[np.ndarray] = None,
    path_distance_px: Optional[np.ndarray] = None,
    positions_mm: Optional[np.ndarray] = None,
    positions_px: Optional[np.ndarray] = None,
    physical_speed_mm: Optional[np.ndarray] = None,
    delta_seconds: Optional[np.ndarray] = None,
    transition_valid: Optional[np.ndarray] = None,
    sample_valid: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Detect one event per accepted speed peak."""

    if peak_event_boundary_mode not in PEAK_EVENT_BOUNDARY_MODES:
        expected = ", ".join(PEAK_EVENT_BOUNDARY_MODES)
        raise ValueError(
            f"Unsupported peak_event_boundary_mode {peak_event_boundary_mode!r}; expected one of: {expected}"
        )
    if shape_split_policy not in SHAPE_SPLIT_POLICIES:
        expected = ", ".join(SHAPE_SPLIT_POLICIES)
        raise ValueError(f"Unsupported shape_split_policy {shape_split_policy!r}; expected one of: {expected}")
    if min_peak_distance_s < 0:
        raise ValueError(f"min_peak_distance_s must be >= 0, got {min_peak_distance_s!r}.")
    if peak_width_rel_height <= 0:
        raise ValueError(f"peak_width_rel_height must be > 0, got {peak_width_rel_height!r}.")

    speed_clean = np.asarray(speed, dtype=np.float64)
    speed_clean = np.where(np.isfinite(speed_clean), speed_clean, 0.0)
    min_bout_frames = _duration_seconds_to_frames(min_bout_duration_s, fps)
    min_peak_distance_frames = _duration_seconds_to_frames(min_peak_distance_s, fps)
    peak_kwargs: dict[str, Any] = {}
    if min_peak_height_mm_s is not None:
        peak_kwargs["height"] = float(min_peak_height_mm_s)
    if min_peak_prominence_mm_s is not None:
        peak_kwargs["prominence"] = float(min_peak_prominence_mm_s)
    else:
        peak_kwargs["prominence"] = 0.0
    if min_peak_distance_frames > 0:
        peak_kwargs["distance"] = int(min_peak_distance_frames)

    peaks, properties = signal.find_peaks(speed_clean, **peak_kwargs)
    if peaks.size == 0:
        return _empty_bouts(), _empty_peak_events()

    widths, width_heights, left_ips, right_ips = signal.peak_widths(
        speed_clean,
        peaks,
        rel_height=float(peak_width_rel_height),
    )
    starts, ends_exclusive = _resolve_peak_event_boundaries(
        speed=speed_clean,
        peaks=peaks,
        left_ips=left_ips,
        right_ips=right_ips,
    )

    durations = ends_exclusive - starts
    valid = (durations >= min_bout_frames) & (starts <= peaks) & (peaks < ends_exclusive)
    peaks = peaks[valid]
    starts = starts[valid]
    ends_exclusive = ends_exclusive[valid]
    widths = widths[valid]
    width_heights = width_heights[valid]
    left_ips = left_ips[valid]
    right_ips = right_ips[valid]
    properties = {
        key: np.asarray(value)[valid]
        for key, value in properties.items()
        if np.asarray(value).shape[0] == valid.shape[0]
    }
    if peaks.size == 0:
        return _empty_bouts(), _empty_peak_events()

    bouts = _build_bout_array(
        speed,
        frames,
        fps,
        starts,
        ends_exclusive,
        starts.copy(),
        ends_exclusive.copy(),
        path_distance_mm=path_distance_mm,
        path_distance_px=path_distance_px,
        positions_mm=positions_mm,
        positions_px=positions_px,
        physical_speed_mm=physical_speed_mm,
        delta_seconds=delta_seconds,
        transition_valid=transition_valid,
        sample_valid=sample_valid,
        threshold=None,
    )
    peak_events = _build_peak_event_array(
        peaks=peaks,
        properties=properties,
        widths=widths,
        width_heights=width_heights,
        left_ips=left_ips,
        right_ips=right_ips,
        frames=frames,
        speed=speed,
        fps=fps,
        boundary_mode=peak_event_boundary_mode,
        shape_split_policy=shape_split_policy,
    )
    return bouts, peak_events


def _load_track_kinematics_track_speeds(
    zarr_path: Path,
    track_kinematics_run: str,
    track_id: int = 0,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """
    Load speed levels from a track kinematics track.

    Args:
        zarr_path: Path to capture zarr
        track_kinematics_run: Track kinematics run name (or "latest")
        track_id: Track ID to load

    Returns:
        Tuple of (speed_dict, metadata_dict)
        speed_dict keys include speed_raw_mm, speed_filtered_mm,
        speed_smoothed_mm, speed_averaged_mm, and frames.
        metadata_dict keys: fps, pixel_to_mm, n_frames, etc.
    """
    root = open_zarr_root(zarr_path, mode='r')
    track = load_track_kinematics_track(
        root,
        run_name=track_kinematics_run,
        scope="offline",
        track_id=track_id,
    )
    speeds = track.speed_level_dict()

    source_provenance = track.run_attrs.get('provenance')
    if not isinstance(source_provenance, dict):
        source_provenance = {}
    source_git = source_provenance.get('git')
    if not isinstance(source_git, dict):
        source_git = {}

    # Load metadata
    metadata = {
        'fps': track.run_attrs.get('fps', 60.0),
        'pixel_to_mm': track.run_attrs.get('pixel_to_mm'),
        'n_frames': len(speeds['frames']),
        'track_kinematics_run': track.run_name,
        'track_kinematics_scope': track.scope,
        'track_kinematics_created_at_utc': track.run_attrs.get('created_at_utc'),
        'track_kinematics_stage': source_provenance.get('stage'),
        'track_kinematics_version': source_provenance.get('version'),
        'track_kinematics_git_commit': track.run_attrs.get('git_commit') or source_git.get('commit'),
        'track_kinematics_git_dirty': track.run_attrs.get('git_dirty', source_git.get('is_dirty')),
        'track_id': track_id,
        'positions_mm': track.positions_mm,
        'positions_px': track.positions_px,
    }

    return speeds, metadata


def _metric_inputs_for_level(
    speeds: Dict[str, Optional[np.ndarray]],
    metadata: Dict[str, Any],
    level: str,
    *,
    path_distance_level_source: Optional[Mapping[str, str]] = None,
) -> Dict[str, Optional[np.ndarray]]:
    path_sources = path_distance_level_source or PATH_DISTANCE_LEVEL_SOURCE
    path_source = path_sources[level]
    return {
        "path_distance_mm": speeds.get(f"frame_path_distance_{path_source}_mm"),
        "path_distance_px": speeds.get(f"frame_path_distance_{path_source}_px"),
        "positions_mm": metadata.get("positions_mm"),
        "positions_px": metadata.get("positions_px"),
        "physical_speed_mm": speeds.get(f"speed_{path_source}_mm"),
        "delta_seconds": speeds.get("delta_seconds"),
        "transition_valid": speeds.get("transition_valid"),
        "sample_valid": speeds.get("sample_valid"),
    }


def detect_and_save_bouts(
    zarr_path: Path,
    run_name: Optional[str],
    track_kinematics_run: str = "latest",
    track_id: int = 0,
    method: str = DEFAULT_DETECTION_METHOD,
    threshold_mm: float = 0.01,
    prominence: float = 1.0,
    min_peak_height: Optional[float] = None,
    rel_height: float = 0.9,
    min_bout_duration_s: float = 0.05,
    min_gap_duration_s: float = 0.1,
    min_gap_frames: Optional[int] = None,
    gap_merge_policy: str = "sampled_frame_gap",
    min_peak_height_mm_s: Optional[float] = None,
    min_peak_prominence_mm_s: Optional[float] = DEFAULT_MIN_PEAK_PROMINENCE_MM_S,
    min_peak_distance_s: float = DEFAULT_MIN_PEAK_DISTANCE_S,
    peak_width_rel_height: float = DEFAULT_PEAK_WIDTH_REL_HEIGHT,
    peak_event_boundary_mode: str = "relative_prominence_width",
    shape_split_policy: str = "none",
    default_level: str = DEFAULT_SWIM_BOUT_LEVEL,
    overwrite: bool = False,
    boundary_mode: str = "threshold",
    boundary_window_s: float = 0.25,
    exponential_tau_s: float = DEFAULT_EXPONENTIAL_TAU_S,
    exponential_source_level: str = DEFAULT_EXPONENTIAL_SOURCE_LEVEL,
    layout: str = SWIM_BOUT_LAYOUT_DEFAULT,
    command: Optional[str] = None,
) -> str:
    """
    Detect bouts from source and derived speed levels and save hierarchically.

    Args:
        zarr_path: Path to capture zarr
        run_name: Name for this bout detection run (auto-generated if None)
        track_kinematics_run: Track kinematics run name (or "latest")
        track_id: Track ID to analyze
        method: Detection method ("threshold", "peak", or "peak_event")
        threshold_mm: Speed threshold in mm/s (for threshold method)
        prominence: Minimum peak prominence in mm/s (for peak method)
        min_peak_height: Minimum absolute peak height in mm/s (for peak method, optional)
        rel_height: Relative height for bout boundaries (for peak method). Higher values
            (0.7-1.0) capture more of the bout tail (default: 0.9)
        min_bout_duration_s: Minimum bout duration
        min_gap_duration_s: Minimum gap between bouts, resolved to frames with
            ceiling when min_gap_frames is not provided.
        min_gap_frames: Explicit minimum gap in frames. Overrides
            min_gap_duration_s when provided.
        gap_merge_policy: Rule for merging threshold-separated segments.
            "sampled_frame_gap" uses sampled below-threshold frame counts.
            "interpolated_core_gap" uses interpolated core threshold crossing
            times when available and falls back to sampled frames otherwise.
        min_peak_height_mm_s: Minimum absolute peak height in mm/s for
            peak_event mode.
        min_peak_prominence_mm_s: Minimum peak prominence in mm/s for
            peak_event mode.
        min_peak_distance_s: Minimum spacing between accepted peak-event peaks.
        peak_width_rel_height: Relative height passed to scipy.signal.peak_widths
            for peak_event boundaries.
        peak_event_boundary_mode: Boundary assignment mode for peak_event runs.
            The first implementation supports relative_prominence_width.
        shape_split_policy: Optional waveform-shape split policy. The first
            implementation supports none only.
        default_level: Speed subgroup that downstream consumers should use by
            default. Accepts raw/filtered/smoothed/averaged/exponential aliases.
        overwrite: Delete and recreate an existing run with the same name.
        boundary_mode: Boundary mode for stored start/end. "threshold" stores
            threshold/peak-width boundaries; "local_minimum" expands to nearby
            low-speed minima and preserves threshold/peak boundaries in core_*.
        boundary_window_s: Local-minimum search window on each side of each
            threshold/peak core.
        exponential_tau_s: Time constant for the causal exponential response
            speed candidate.
        exponential_source_level: Source speed level for the exponential
            response candidate. Accepts raw/filtered/smoothed/averaged aliases.
        layout: Storage layout. hierarchical_v1 preserves the existing physical
            tree shape. compact_v2 writes the opt-in tabular v2 layout.
        command: Optional command string to record in stage provenance.

    Returns:
        The run name used (either provided or auto-generated)
    """
    default_level_key = normalize_speed_level(default_level)
    exponential_source_key = normalize_speed_level(exponential_source_level)
    if exponential_source_key == "speed_exponential":
        raise ValueError("exponential_source_level cannot be speed_exponential.")
    if exponential_tau_s <= 0:
        raise ValueError(f"exponential_tau_s must be > 0, got {exponential_tau_s!r}.")
    if layout not in SWIM_BOUT_LAYOUT_CHOICES:
        expected = ", ".join(SWIM_BOUT_LAYOUT_CHOICES)
        raise ValueError(f"Unsupported layout {layout!r}; expected one of: {expected}")
    if boundary_mode not in BOUNDARY_MODES:
        expected = ", ".join(BOUNDARY_MODES)
        raise ValueError(f"Unsupported boundary_mode {boundary_mode!r}; expected one of: {expected}")
    if gap_merge_policy not in GAP_MERGE_POLICIES:
        expected = ", ".join(GAP_MERGE_POLICIES)
        raise ValueError(f"Unsupported gap_merge_policy {gap_merge_policy!r}; expected one of: {expected}")
    if method in {"peak", "peak_event"} and gap_merge_policy != "sampled_frame_gap":
        raise ValueError(f"{method} detection currently supports gap_merge_policy='sampled_frame_gap' only.")
    if peak_event_boundary_mode not in PEAK_EVENT_BOUNDARY_MODES:
        expected = ", ".join(PEAK_EVENT_BOUNDARY_MODES)
        raise ValueError(
            f"Unsupported peak_event_boundary_mode {peak_event_boundary_mode!r}; expected one of: {expected}"
        )
    if shape_split_policy not in SHAPE_SPLIT_POLICIES:
        expected = ", ".join(SHAPE_SPLIT_POLICIES)
        raise ValueError(f"Unsupported shape_split_policy {shape_split_policy!r}; expected one of: {expected}")
    gap_merge_policy_active = method != "peak_event"

    print(f"\n{'='*60}")
    print(f"MULTI-LEVEL SWIM BOUT DETECTION")
    print(f"{'='*60}")
    print(f"Capture: {zarr_path.name}")
    print(f"Track kinematics run: {track_kinematics_run}")
    print(f"Track ID: {track_id}")
    print(f"Method: {method}")
    print(f"Boundary mode: {boundary_mode}")
    print(f"Gap merge policy: {gap_merge_policy}")
    if boundary_mode == "local_minimum":
        print(f"Boundary window: {boundary_window_s} s")
    print(f"Exponential response: source={exponential_source_key}, tau={exponential_tau_s} s")
    print(f"Storage layout: {layout}")
    if method == "threshold":
        print(f"Threshold: {threshold_mm} mm/s")
    elif method == "peak":
        print(f"Prominence: {prominence} mm/s")
        if min_peak_height is not None:
            print(f"Min peak height: {min_peak_height} mm/s")
        print(f"Relative height: {rel_height}")
    elif method == "peak_event":
        print(f"Peak-event min prominence: {min_peak_prominence_mm_s} mm/s")
        if min_peak_height_mm_s is not None:
            print(f"Peak-event min height: {min_peak_height_mm_s} mm/s")
        print(f"Peak-event min distance: {min_peak_distance_s} s")
        print(f"Peak-event boundary mode: {peak_event_boundary_mode}")
        print(f"Peak-event width relative height: {peak_width_rel_height}")
        print(f"Shape split policy: {shape_split_policy}")
    print(f"Default level: {default_level_key}")
    print()

    timed_pipeline_started_at = perf_counter()
    phase_durations_s: Dict[str, float] = {}
    detection_level_timings: Dict[str, Dict[str, Any]] = {}

    # Load speed data
    print("Loading track kinematics track data...")
    phase_started_at = perf_counter()
    speeds, metadata = _load_track_kinematics_track_speeds(
        zarr_path, track_kinematics_run, track_id
    )
    _finish_timed_phase(
        phase_durations_s,
        "load_track_kinematics",
        phase_started_at,
    )

    fps = metadata['fps']
    frames = speeds['frames']
    resolved_min_bout_frames = _duration_seconds_to_frames(min_bout_duration_s, fps)
    resolved_min_gap_frames, min_gap_frame_source = _resolve_min_gap_frames(
        min_gap_duration_s=min_gap_duration_s,
        fps=fps,
        min_gap_frames=min_gap_frames,
    )
    gap_merge_min_gap_duration_s, gap_merge_min_gap_source = _gap_merge_min_gap_duration_s(
        min_gap_duration_s=min_gap_duration_s,
        fps=fps,
        min_gap_frames=min_gap_frames,
    )
    resolved_boundary_window_frames = max(0, int(round(boundary_window_s * fps)))
    phase_started_at = perf_counter()
    speeds["speed_exponential_mm"] = _causal_exponential_speed_response(
        speeds[f"{exponential_source_key}_mm"],
        frames,
        fps,
        tau_s=float(exponential_tau_s),
        transition_valid=speeds.get("transition_valid"),
    )
    _finish_timed_phase(
        phase_durations_s,
        "build_exponential_response",
        phase_started_at,
    )
    path_distance_level_source = {
        **PATH_DISTANCE_LEVEL_SOURCE,
        "speed_exponential": PATH_DISTANCE_LEVEL_SOURCE[exponential_source_key],
    }

    print(f"  FPS: {fps}")
    print(f"  Frames: {metadata['n_frames']}")
    print(f"  Track kinematics run: {metadata['track_kinematics_run']}")
    if gap_merge_policy_active:
        print(
            f"  Min gap: {resolved_min_gap_frames} frames "
            f"({resolved_min_gap_frames / fps:.4f} s effective, source={min_gap_frame_source})"
        )
        print(
            f"  Gap merge: {gap_merge_policy} "
            f"({gap_merge_min_gap_duration_s:.4f} s threshold, source={gap_merge_min_gap_source})"
        )
    else:
        print("  Gap merge: not used by peak_event")
    print()

    # Detect bouts for each speed level
    speed_levels = list(SPEED_LEVELS)
    bout_results = {}
    peak_event_results = {}

    print("Detecting bouts for each speed level:")
    detection_started_at = perf_counter()
    for level in speed_levels:
        level_started_at = perf_counter()
        speed_key = f"{level}_mm"
        speed = speeds[speed_key]
        metric_inputs = _metric_inputs_for_level(
            speeds,
            metadata,
            level,
            path_distance_level_source=path_distance_level_source,
        )

        # Skip if all NaN
        if np.all(np.isnan(speed)):
            print(f"  {level}: SKIPPED (all NaN)")
            bout_results[level] = _empty_bouts()
            peak_event_results[level] = _empty_peak_events()
            level_elapsed_s = max(0.0, float(perf_counter() - level_started_at))
            detection_level_timings[level] = {
                "elapsed_s": level_elapsed_s,
                "status": "skipped_all_nan",
                "n_bouts": 0,
                "n_peak_events": 0,
            }
            print(
                f"phase_timing phase=detect_level level={level} "
                f"elapsed_s={level_elapsed_s:.6f} status=skipped_all_nan",
                flush=True,
            )
            continue

        if method == "threshold":
            bouts = _detect_bouts_from_speed(
                speed=speed,
                frames=frames,
                fps=fps,
                threshold=threshold_mm,
                min_bout_duration_s=min_bout_duration_s,
                min_gap_duration_s=min_gap_duration_s,
                min_gap_frames=min_gap_frames,
                gap_merge_policy=gap_merge_policy,
                boundary_mode=boundary_mode,
                boundary_window_s=boundary_window_s,
                **metric_inputs,
            )
        elif method == "peak":
            bouts = _detect_bouts_from_peaks(
                speed=speed,
                frames=frames,
                fps=fps,
                prominence=prominence,
                min_peak_height=min_peak_height,
                rel_height=rel_height,
                min_bout_duration_s=min_bout_duration_s,
                min_gap_duration_s=min_gap_duration_s,
                min_gap_frames=min_gap_frames,
                gap_merge_policy=gap_merge_policy,
                boundary_mode=boundary_mode,
                boundary_window_s=boundary_window_s,
                **metric_inputs,
            )
            peak_events = _empty_peak_events()
        elif method == "peak_event":
            bouts, peak_events = _detect_bouts_from_peak_events(
                speed=speed,
                frames=frames,
                fps=fps,
                min_peak_height_mm_s=min_peak_height_mm_s,
                min_peak_prominence_mm_s=min_peak_prominence_mm_s,
                min_peak_distance_s=min_peak_distance_s,
                peak_width_rel_height=peak_width_rel_height,
                peak_event_boundary_mode=peak_event_boundary_mode,
                shape_split_policy=shape_split_policy,
                min_bout_duration_s=min_bout_duration_s,
                **metric_inputs,
            )
        else:
            raise ValueError(
                f"Unknown detection method: {method}. Must be 'threshold', 'peak', or 'peak_event'."
            )

        bout_results[level] = bouts
        if method == "threshold":
            peak_events = _empty_peak_events()
        peak_event_results[level] = peak_events
        print(f"  {level}: {len(bouts)} bouts detected")
        level_elapsed_s = max(0.0, float(perf_counter() - level_started_at))
        detection_level_timings[level] = {
            "elapsed_s": level_elapsed_s,
            "status": "complete",
            "n_bouts": int(len(bouts)),
            "n_peak_events": int(len(peak_events)),
        }
        print(
            f"phase_timing phase=detect_level level={level} "
            f"elapsed_s={level_elapsed_s:.6f} status=complete "
            f"n_bouts={len(bouts)} n_peak_events={len(peak_events)}",
            flush=True,
        )

    _finish_timed_phase(
        phase_durations_s,
        "detect_levels",
        detection_started_at,
    )

    print()

    # Save to zarr
    print("Saving to zarr...")
    phase_started_at = perf_counter()
    root = open_zarr_root(zarr_path, mode='r+')

    # Create analysis/swim_bout_runs if needed
    if 'analysis' not in root:
        analysis_group = root.create_group('analysis')
    else:
        analysis_group = root['analysis']

    swim_bout_runs = require_runs_parent(analysis_group, 'swim_bout_runs')

    # Auto-generate run name if not provided
    if run_name is None:
        timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
        run_name = f"swim_bout_detect_{timestamp}"
        print(f"  Auto-generated run name: {run_name}")
    elif run_name in swim_bout_runs:
        if not overwrite:
            raise ValueError(
                f"Swim bout run '{run_name}' already exists. Use --overwrite, "
                "a different name, or delete the existing run."
            )
        print(f"  Overwriting existing run: {run_name}")
        del swim_bout_runs[run_name]

    # Create run group
    run_group = swim_bout_runs.create_group(run_name)
    mark_run_started(run_group, run_name=run_name, stage="swim_bout")

    # Save metadata at run level
    git_info = get_git_info()
    env_info = get_environment_info(
        disk_path=str(zarr_path),
        capture_env_vars=False,
    )
    created_at_utc = datetime.now(timezone.utc).isoformat()
    schema_version = (
        SWIM_BOUT_RUN_SCHEMA_VERSION_COMPACT_V2
        if layout == SWIM_BOUT_LAYOUT_COMPACT_V2
        else SWIM_BOUT_RUN_SCHEMA_VERSION
    )
    run_group.attrs['schema_id'] = SWIM_BOUT_RUN_SCHEMA_ID
    run_group.attrs['schema_version'] = schema_version
    run_group.attrs['layout'] = layout
    run_group.attrs['created_at_utc'] = created_at_utc
    run_group.attrs['detection_method'] = method
    run_group.attrs['min_bout_duration_s'] = min_bout_duration_s
    run_group.attrs['min_gap_duration_s'] = min_gap_duration_s
    run_group.attrs['min_gap_frames'] = (
        int(min_gap_frames) if min_gap_frames is not None else None
    )
    run_group.attrs['resolved_min_bout_frames'] = int(resolved_min_bout_frames)
    run_group.attrs['resolved_min_gap_frames'] = int(resolved_min_gap_frames)
    run_group.attrs['effective_min_bout_duration_s'] = float(resolved_min_bout_frames / fps)
    run_group.attrs['effective_min_gap_duration_s'] = float(resolved_min_gap_frames / fps)
    run_group.attrs['min_gap_frame_source'] = min_gap_frame_source
    run_group.attrs['gap_merge_policy'] = gap_merge_policy
    run_group.attrs['gap_merge_policy_active'] = bool(gap_merge_policy_active)
    run_group.attrs['gap_merge_min_gap_duration_s'] = float(gap_merge_min_gap_duration_s)
    run_group.attrs['gap_merge_min_gap_source'] = gap_merge_min_gap_source
    run_group.attrs['duration_frame_rounding_policy'] = DURATION_FRAME_ROUNDING_POLICY
    run_group.attrs['boundary_mode'] = boundary_mode
    run_group.attrs['boundary_window_s'] = float(boundary_window_s)
    run_group.attrs['resolved_boundary_window_frames'] = int(resolved_boundary_window_frames)
    run_group.attrs['boundary_frame_rounding_policy'] = BOUNDARY_FRAME_ROUNDING_POLICY
    run_group.attrs['threshold_crossing_interpolation'] = THRESHOLD_CROSSING_INTERPOLATION
    run_group.attrs['interpolated_threshold_fields'] = [
        'core_start_time_s_interpolated',
        'core_end_time_s_interpolated',
        'core_duration_s_interpolated',
    ]
    run_group.attrs['exponential_tau_s'] = float(exponential_tau_s)
    run_group.attrs['exponential_source_level'] = exponential_source_key
    run_group.attrs['detection_signal_schema_id'] = DETECTION_SIGNAL_SCHEMA_ID
    run_group.attrs['detection_signal_schema_version'] = DETECTION_SIGNAL_SCHEMA_VERSION
    run_group.attrs['peak_event_schema_id'] = PEAK_EVENT_SCHEMA_ID
    run_group.attrs['peak_event_schema_version'] = PEAK_EVENT_SCHEMA_VERSION

    # Method-specific parameters
    if method == "threshold":
        run_group.attrs['threshold_mm'] = threshold_mm
    elif method == "peak":
        run_group.attrs['prominence'] = prominence
        run_group.attrs['min_peak_height'] = _json_safe_attr_value(min_peak_height)
        run_group.attrs['rel_height'] = rel_height
    elif method == "peak_event":
        run_group.attrs['min_peak_height_mm_s'] = _json_safe_attr_value(min_peak_height_mm_s)
        run_group.attrs['min_peak_prominence_mm_s'] = _json_safe_attr_value(min_peak_prominence_mm_s)
        run_group.attrs['min_peak_distance_s'] = float(min_peak_distance_s)
        run_group.attrs['peak_width_rel_height'] = float(peak_width_rel_height)
        run_group.attrs['peak_event_boundary_mode'] = peak_event_boundary_mode
        run_group.attrs['shape_split_policy'] = shape_split_policy

    run_group.attrs['source_track_kinematics_run'] = metadata['track_kinematics_run']
    run_group.attrs['track_id'] = track_id
    run_group.attrs['fps'] = fps
    run_group.attrs['pixel_to_mm'] = _json_safe_attr_value(metadata.get('pixel_to_mm'))
    run_group.attrs['default_level'] = default_level_key
    run_group.attrs['git_commit'] = git_info['commit_hash']
    run_group.attrs['git_branch'] = git_info['branch']
    run_group.attrs['git_dirty'] = git_info['is_dirty']
    source_track_path = (
        f"analysis/track_kinematics_runs/offline/"
        f"{metadata['track_kinematics_run']}/tracks/id_{int(track_id)}"
    )

    parameters = _json_safe_attr_value({
        'method': method,
        'threshold_mm': float(threshold_mm) if method == "threshold" else None,
        'prominence': float(prominence) if method == "peak" else None,
        'min_peak_height': float(min_peak_height) if min_peak_height is not None else None,
        'rel_height': float(rel_height) if method == "peak" else None,
        'min_peak_height_mm_s': (
            float(min_peak_height_mm_s) if min_peak_height_mm_s is not None else None
        ),
        'min_peak_prominence_mm_s': (
            float(min_peak_prominence_mm_s) if min_peak_prominence_mm_s is not None else None
        ),
        'min_peak_distance_s': float(min_peak_distance_s),
        'peak_width_rel_height': float(peak_width_rel_height),
        'peak_event_boundary_mode': peak_event_boundary_mode,
        'shape_split_policy': shape_split_policy,
        'min_bout_duration_s': float(min_bout_duration_s),
        'min_gap_duration_s': float(min_gap_duration_s),
        'min_gap_frames': int(min_gap_frames) if min_gap_frames is not None else None,
        'resolved_min_bout_frames': int(resolved_min_bout_frames),
        'resolved_min_gap_frames': int(resolved_min_gap_frames),
        'effective_min_bout_duration_s': float(resolved_min_bout_frames / fps),
        'effective_min_gap_duration_s': float(resolved_min_gap_frames / fps),
        'min_gap_frame_source': min_gap_frame_source,
        'gap_merge_policy': gap_merge_policy,
        'gap_merge_policy_active': bool(gap_merge_policy_active),
        'gap_merge_min_gap_duration_s': float(gap_merge_min_gap_duration_s),
        'gap_merge_min_gap_source': gap_merge_min_gap_source,
        'duration_frame_rounding_policy': DURATION_FRAME_ROUNDING_POLICY,
        'default_level': default_level_key,
        'boundary_mode': boundary_mode,
        'boundary_window_s': float(boundary_window_s),
        'resolved_boundary_window_frames': int(resolved_boundary_window_frames),
        'boundary_frame_rounding_policy': BOUNDARY_FRAME_ROUNDING_POLICY,
        'threshold_crossing_interpolation': THRESHOLD_CROSSING_INTERPOLATION,
        'interpolated_threshold_fields': [
            'core_start_time_s_interpolated',
            'core_end_time_s_interpolated',
            'core_duration_s_interpolated',
        ],
        'exponential_tau_s': float(exponential_tau_s),
        'exponential_source_level': exponential_source_key,
        'speed_levels': list(speed_levels),
        'layout': layout,
        'swim_bout_run_schema_id': SWIM_BOUT_RUN_SCHEMA_ID,
        'swim_bout_run_schema_version': schema_version,
        'bout_metric_schema_id': BOUT_METRIC_SCHEMA_ID,
        'peak_event_schema_id': PEAK_EVENT_SCHEMA_ID,
        'peak_event_schema_version': PEAK_EVENT_SCHEMA_VERSION,
        'distance_policy': 'path_length_from_track_frame_path_distance_only',
        'overwrite': bool(overwrite),
    })
    inputs = _json_safe_attr_value({
        'zarr_path': str(zarr_path),
        'source_track_kinematics_run': metadata['track_kinematics_run'],
        'source_track_kinematics_stage': metadata.get('track_kinematics_stage'),
        'source_track_kinematics_version': metadata.get('track_kinematics_version'),
        'source_track_path': source_track_path,
        'source_track_kinematics_created_at_utc': metadata.get(
            'track_kinematics_created_at_utc'
        ),
        'source_track_kinematics_git_commit': metadata.get('track_kinematics_git_commit'),
        'source_track_kinematics_git_dirty': metadata.get('track_kinematics_git_dirty'),
        'track_id': int(track_id),
        'fps': float(fps),
        'pixel_to_mm': metadata.get('pixel_to_mm'),
        'n_frames': int(metadata['n_frames']),
    })
    provenance = _json_safe_attr_value(build_stage_provenance(
        stage="detect_bouts_multi_level",
        created_at_utc=created_at_utc,
        parameters=parameters,
        inputs=inputs,
        command=command,
        version=METHOD_VERSION,
        git=git_info,
        environment=env_info.get("environment"),
        platform=env_info.get("platform"),
        artifacts={
            'run_path': f"analysis/swim_bout_runs/{run_name}",
            'default_level': default_level_key,
            'layout': layout,
        },
    ))
    write_stage_provenance(run_group, provenance)
    _finish_timed_phase(
        phase_durations_s,
        "initialize_output_and_metadata",
        phase_started_at,
    )

    signal_id_by_level = {level: idx for idx, level in enumerate(speed_levels)}
    estimator_signal_id_by_level = {
        level: int(signal_id_by_level[f"speed_{path_distance_level_source[level]}"])
        for level in speed_levels
    }
    level_payloads: dict[str, dict[str, Any]] = {}
    phase_started_at = perf_counter()
    for level in speed_levels:
        bouts = bout_results[level]
        peak_events = peak_event_results[level]

        # Compute inter-bout intervals and global metrics
        intervals, interval_metrics, interval_histogram = _compute_inter_bout_intervals(bouts, fps)
        global_metrics = _compute_global_metrics(bouts, fps, metadata['n_frames'])
        bout_points = _create_bout_points(bouts, metadata['positions_mm'], metadata['positions_px'], frames, fps)

        # Update global metrics with interval statistics
        if len(bouts) >= 2:
            global_metrics['inter_bout_interval_count'][0] = interval_metrics['inter_bout_interval_count']
            global_metrics['inter_bout_interval_mean_s'][0] = interval_metrics['inter_bout_interval_mean_s']
            global_metrics['inter_bout_interval_std_s'][0] = interval_metrics['inter_bout_interval_std_s']
            global_metrics['inter_bout_interval_median_s'][0] = interval_metrics['inter_bout_interval_median_s']
            global_metrics['inter_bout_interval_min_s'][0] = interval_metrics['inter_bout_interval_min_s']
            global_metrics['inter_bout_interval_max_s'][0] = interval_metrics['inter_bout_interval_max_s']

        level_specific_attrs = {
            'n_bouts': len(bouts),
            'speed_level': level,
            'bout_metric_schema_id': BOUT_METRIC_SCHEMA_ID,
            'peak_event_schema_id': PEAK_EVENT_SCHEMA_ID,
            'peak_event_schema_version': PEAK_EVENT_SCHEMA_VERSION,
            'n_peak_events': int(len(peak_events)),
            'path_distance_source_level': path_distance_level_source[level],
            'is_default_level': level == default_level_key,
            'min_bout_duration_s': float(min_bout_duration_s),
            'min_gap_duration_s': float(min_gap_duration_s),
            'min_gap_frames': int(min_gap_frames) if min_gap_frames is not None else None,
            'resolved_min_bout_frames': int(resolved_min_bout_frames),
            'resolved_min_gap_frames': int(resolved_min_gap_frames),
            'effective_min_bout_duration_s': float(resolved_min_bout_frames / fps),
            'effective_min_gap_duration_s': float(resolved_min_gap_frames / fps),
            'min_gap_frame_source': min_gap_frame_source,
            'gap_merge_policy': gap_merge_policy,
            'gap_merge_policy_active': bool(gap_merge_policy_active),
            'gap_merge_min_gap_duration_s': float(gap_merge_min_gap_duration_s),
            'gap_merge_min_gap_source': gap_merge_min_gap_source,
            'duration_frame_rounding_policy': DURATION_FRAME_ROUNDING_POLICY,
            'threshold_crossing_interpolation': THRESHOLD_CROSSING_INTERPOLATION,
            'interpolated_threshold_fields': [
                'core_start_time_s_interpolated',
                'core_end_time_s_interpolated',
                'core_duration_s_interpolated',
            ],
            'peak_event_boundary_mode': peak_event_boundary_mode,
            'shape_split_policy': shape_split_policy,
            'min_peak_height_mm_s': _json_safe_attr_value(min_peak_height_mm_s),
            'min_peak_prominence_mm_s': _json_safe_attr_value(min_peak_prominence_mm_s),
            'min_peak_distance_s': float(min_peak_distance_s),
            'peak_width_rel_height': float(peak_width_rel_height),
        }
        level_specific_attrs.update(
            _detection_signal_attrs(
                level=level,
                source_track_path=source_track_path,
                path_distance_source_level=path_distance_level_source[level],
                exponential_source_key=exponential_source_key,
                exponential_tau_s=float(exponential_tau_s),
            )
        )
        if len(bouts) > 0:
            level_specific_attrs['total_bout_time_s'] = float(np.sum(bouts['duration_s']))
            level_specific_attrs['total_observed_bout_time_s'] = float(np.sum(bouts['observed_duration_s']))
            level_specific_attrs['mean_bout_duration_s'] = float(np.mean(bouts['duration_s']))
            level_specific_attrs['mean_bout_observed_duration_s'] = float(np.mean(bouts['observed_duration_s']))
            level_specific_attrs['mean_bout_speed_mm_s'] = _finite_mean_or_nan(bouts['mean_speed_mm_s'])
            level_specific_attrs['mean_bout_peak_detection_signal_mm_s'] = _finite_mean_or_nan(
                bouts['peak_detection_signal_mm_s']
            )
            level_specific_attrs['mean_bout_peak_physical_speed_mm_s'] = _finite_mean_or_nan(
                bouts['peak_physical_speed_mm_s']
            )
            level_specific_attrs['total_path_length_mm'] = _finite_sum_or_nan(bouts['path_length_mm'])
            level_specific_attrs['mean_valid_transition_fraction'] = _finite_mean_or_nan(
                bouts['valid_transition_fraction']
            )
            level_specific_attrs['n_gap_censored_bouts'] = int(np.sum(bouts['gap_censored']))

        level_specific_attrs = _json_safe_attrs(level_specific_attrs)
        level_payloads[level] = {
            "bouts": bouts,
            "peak_events": peak_events,
            "intervals": intervals,
            "interval_metrics": interval_metrics,
            "interval_histogram": interval_histogram,
            "global_metrics": global_metrics,
            "bout_points": bout_points,
            "attrs": level_specific_attrs,
        }

    _finish_timed_phase(
        phase_durations_s,
        "prepare_level_payloads",
        phase_started_at,
    )

    phase_started_at = perf_counter()
    if layout == SWIM_BOUT_LAYOUT_COMPACT_V2:
        _write_compact_v2_swim_bout_payloads(
            run_group,
            run_name=run_name,
            speed_levels=speed_levels,
            level_payloads=level_payloads,
            signal_id_by_level=signal_id_by_level,
            estimator_signal_id_by_level=estimator_signal_id_by_level,
            default_level_key=default_level_key,
            method=method,
            parameters=parameters,
            provenance=provenance,
            track_id=track_id,
            pixel_to_mm=metadata.get("pixel_to_mm"),
            path_distance_level_source=path_distance_level_source,
            source_track_path=source_track_path,
            exponential_source_key=exponential_source_key,
            exponential_tau_s=float(exponential_tau_s),
            frames=frames,
            speeds=speeds,
        )
        for level in speed_levels:
            payload = level_payloads[level]
            print(f"  Saved {level}: {len(payload['bouts'])} bouts, {len(payload['intervals'])} intervals")
    else:
        # Save each speed level's bouts and statistics in v1 subgroups.
        for level in speed_levels:
            level_group = run_group.create_group(level)
            payload = level_payloads[level]
            level_specific_attrs = payload["attrs"]
            level_group.attrs.update(level_specific_attrs)
            if level == "speed_exponential":
                store_array(
                    level_group,
                    "detection_signal_mm_s",
                    np.asarray(speeds["speed_exponential_mm"], dtype=np.float32),
                    attrs={
                        "units": "mm/s",
                        **_detection_signal_attrs(
                            level=level,
                            source_track_path=source_track_path,
                            path_distance_source_level=path_distance_level_source[level],
                            exponential_source_key=exponential_source_key,
                            exponential_tau_s=float(exponential_tau_s),
                        ),
                    },
                )
                store_array(
                    level_group,
                    "frame_indices",
                    np.asarray(frames, dtype=np.int64),
                    attrs={"source": "track_kinematics.frame_indices"},
                )
            write_columnar_dataset(level_group, 'bouts', payload["bouts"], attrs=level_specific_attrs)
            write_columnar_dataset(level_group, 'peak_events', payload["peak_events"], attrs=level_specific_attrs)
            write_columnar_dataset(level_group, 'inter_bout_intervals', payload["intervals"], attrs=None)
            write_columnar_dataset(level_group, 'inter_bout_interval_histogram', payload["interval_histogram"], attrs=None)
            write_columnar_dataset(level_group, 'global_metrics', payload["global_metrics"], attrs=None)
            write_columnar_dataset(level_group, 'bout_points', payload["bout_points"], attrs=None)
            print(f"  Saved {level}: {len(payload['bouts'])} bouts, {len(payload['intervals'])} intervals")

    _finish_timed_phase(
        phase_durations_s,
        "write_payloads",
        phase_started_at,
    )
    phase_timing = _json_safe_attr_value(
        _build_phase_timing_payload(
            phase_durations_s=phase_durations_s,
            detection_levels=detection_level_timings,
            timed_pipeline_elapsed_s=perf_counter() - timed_pipeline_started_at,
        )
    )
    run_group.attrs["phase_timing"] = phase_timing
    provenance = dict(provenance)
    provenance["performance"] = phase_timing
    write_stage_provenance(run_group, provenance)
    write_best_effort_run_lineage_attrs(run_group, run_family="swim_bout_run")
    print(
        "phase_timing "
        f"scope={phase_timing['scope']} "
        f"elapsed_s={phase_timing['timed_pipeline_elapsed_s']:.6f} "
        f"unattributed_s={phase_timing['unattributed_elapsed_s']:.6f}",
        flush=True,
    )

    mark_run_complete(
        run_group,
        parent_group=swim_bout_runs,
        run_name=run_name,
        run_provenance=build_run_provenance_from_stage_record(provenance),
    )

    print()
    print(f"{'='*60}")
    print(f"DETECTION COMPLETE")
    print(f"{'='*60}")
    print(f"Run saved: analysis/swim_bout_runs/{run_name}")
    print(f"  speed_raw: {len(bout_results['speed_raw'])} bouts")
    print(f"  speed_filtered: {len(bout_results['speed_filtered'])} bouts")
    print(f"  speed_smoothed: {len(bout_results['speed_smoothed'])} bouts")
    print(f"  speed_averaged: {len(bout_results['speed_averaged'])} bouts")
    print(f"  speed_exponential: {len(bout_results['speed_exponential'])} bouts")
    print(f"Default level: {default_level_key}")
    print()

    return run_name


def main():
    parser = argparse.ArgumentParser(
        description="Detect swim bouts from speed processing levels",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        'zarr_path',
        type=Path,
        help='Path to the Palette Zarr archive',
    )

    parser.add_argument(
        '--run-name',
        type=str,
        default=None,
        help='Name for this bout detection run (auto-generated if not provided)',
    )

    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Delete and recreate --run-name if it already exists. Ignored for auto-generated run names.',
    )

    parser.add_argument(
        '--layout',
        type=str,
        choices=SWIM_BOUT_LAYOUT_CHOICES,
        default=SWIM_BOUT_LAYOUT_DEFAULT,
        help=(
            'Storage layout for the output run. hierarchical_v1 preserves the existing '
            'tree-shaped layout. compact_v2 writes the tabular schema version 7. '
            f'Default: {SWIM_BOUT_LAYOUT_DEFAULT}.'
        ),
    )

    parser.add_argument(
        '--track-kinematics-run',
        dest='track_kinematics_run',
        type=str,
        default='latest',
        help='Track kinematics run name (default: latest)',
    )

    parser.add_argument(
        '--track-id',
        type=int,
        default=0,
        help='Track ID to analyze (default: 0)',
    )

    parser.add_argument(
        '--method',
        type=str,
        choices=['threshold', 'peak', 'peak_event'],
        default=DEFAULT_DETECTION_METHOD,
        help=(
            'Detection method: "threshold", "peak", or "peak_event" '
            f'(default: {DEFAULT_DETECTION_METHOD})'
        ),
    )

    # Threshold method parameters
    parser.add_argument(
        '--threshold-mm',
        type=float,
        default=0.01,
        help='Speed threshold in mm/s for threshold method (default: 0.01)',
    )

    parser.add_argument(
        '--default-level',
        type=str,
        choices=SPEED_LEVEL_CHOICES,
        default=DEFAULT_SWIM_BOUT_LEVEL,
        help=(
            'Speed level downstream consumers should use by default. '
            'Accepts raw/filtered/smoothed/averaged/exponential aliases or stored subgroup '
            f'names. Default: {DEFAULT_SWIM_BOUT_LEVEL}.'
        ),
    )

    parser.add_argument(
        '--boundary-mode',
        type=str,
        choices=BOUNDARY_MODES,
        default='threshold',
        help=(
            'How to write bout start/end boundaries. "threshold" keeps the '
            'threshold/peak-width core. "local_minimum" expands start/end to '
            'nearby low-speed minima and stores threshold/peak boundaries in '
            'core_* fields. Default: threshold.'
        ),
    )

    parser.add_argument(
        '--boundary-window-s',
        type=float,
        default=0.25,
        help='Search window in seconds on each side for --boundary-mode local_minimum (default: 0.25).',
    )

    parser.add_argument(
        '--exponential-tau-s',
        type=float,
        default=DEFAULT_EXPONENTIAL_TAU_S,
        help=(
            'Time constant in seconds for the causal exponential speed candidate '
            f'(default: {DEFAULT_EXPONENTIAL_TAU_S}).'
        ),
    )

    parser.add_argument(
        '--exponential-source-level',
        type=str,
        choices=tuple(choice for choice in SPEED_LEVEL_CHOICES if normalize_speed_level(choice) != "speed_exponential"),
        default=DEFAULT_EXPONENTIAL_SOURCE_LEVEL,
        help=(
            'Source speed level for the exponential response candidate '
            f'(default: {DEFAULT_EXPONENTIAL_SOURCE_LEVEL}).'
        ),
    )

    # Peak method parameters
    parser.add_argument(
        '--prominence',
        type=float,
        default=1.0,
        help='Minimum peak prominence in mm/s for peak method (default: 1.0)',
    )

    parser.add_argument(
        '--min-peak-height',
        type=float,
        default=None,
        help='Minimum absolute peak height in mm/s for peak method (optional)',
    )

    parser.add_argument(
        '--rel-height',
        type=float,
        default=0.9,
        help='Relative height for bout boundaries in peak method. Higher values (0.7-1.0) capture more of the bout tail; lower values (0.3-0.5) set tighter boundaries (default: 0.9)',
    )

    # Peak-event method parameters
    parser.add_argument(
        '--min-peak-height-mm-s',
        type=float,
        default=None,
        help='Minimum absolute peak height in mm/s for peak_event method.',
    )

    parser.add_argument(
        '--min-peak-prominence-mm-s',
        type=float,
        default=DEFAULT_MIN_PEAK_PROMINENCE_MM_S,
        help=(
            'Minimum peak prominence in mm/s for peak_event method '
            f'(default: {DEFAULT_MIN_PEAK_PROMINENCE_MM_S}).'
        ),
    )

    parser.add_argument(
        '--min-peak-distance-s',
        type=float,
        default=DEFAULT_MIN_PEAK_DISTANCE_S,
        help=(
            'Minimum time between accepted peak_event peaks in seconds '
            f'(default: {DEFAULT_MIN_PEAK_DISTANCE_S}).'
        ),
    )

    parser.add_argument(
        '--peak-width-rel-height',
        type=float,
        default=DEFAULT_PEAK_WIDTH_REL_HEIGHT,
        help=(
            'Relative height passed to scipy.signal.peak_widths for peak_event boundaries '
            f'(default: {DEFAULT_PEAK_WIDTH_REL_HEIGHT}).'
        ),
    )

    parser.add_argument(
        '--peak-event-boundary-mode',
        type=str,
        choices=PEAK_EVENT_BOUNDARY_MODES,
        default='relative_prominence_width',
        help='Boundary assignment mode for peak_event method (default: relative_prominence_width).',
    )

    parser.add_argument(
        '--shape-split-policy',
        type=str,
        choices=SHAPE_SPLIT_POLICIES,
        default='none',
        help='Optional waveform-shape split policy for peak_event method (default: none).',
    )

    # Common parameters
    parser.add_argument(
        '--min-bout-duration',
        type=float,
        default=0.05,
        help='Minimum bout duration in seconds, resolved with ceil(seconds * fps) (default: 0.05)',
    )

    parser.add_argument(
        '--min-gap-duration',
        type=float,
        default=0.1,
        help=(
            'Minimum gap between bouts in seconds. Resolved to frames with '
            'ceil(seconds * fps) unless --min-gap-frames is provided '
            '(default: 0.1).'
        ),
    )

    parser.add_argument(
        '--min-gap-frames',
        type=int,
        default=None,
        help='Explicit minimum gap between bouts in frames. Overrides --min-gap-duration.',
    )

    parser.add_argument(
        '--gap-merge-policy',
        type=str,
        choices=GAP_MERGE_POLICIES,
        default='sampled_frame_gap',
        help=(
            'How threshold-separated segments are merged. sampled_frame_gap '
            'uses resolved below-threshold frame counts. interpolated_core_gap '
            'uses interpolated core threshold-crossing times when available '
            'and falls back to sampled frame gaps otherwise. Default: sampled_frame_gap.'
        ),
    )

    args = parser.parse_args()

    zarr_path = args.zarr_path
    if not zarr_path.exists():
        print(f"ERROR: Zarr archive not found: {zarr_path}")
        return 1

    run_name = detect_and_save_bouts(
        zarr_path=zarr_path,
        run_name=args.run_name,
        track_kinematics_run=args.track_kinematics_run,
        track_id=args.track_id,
        method=args.method,
        threshold_mm=args.threshold_mm,
        prominence=args.prominence,
        min_peak_height=args.min_peak_height,
        rel_height=args.rel_height,
        min_bout_duration_s=args.min_bout_duration,
        min_gap_duration_s=args.min_gap_duration,
        min_gap_frames=args.min_gap_frames,
        gap_merge_policy=args.gap_merge_policy,
        min_peak_height_mm_s=args.min_peak_height_mm_s,
        min_peak_prominence_mm_s=args.min_peak_prominence_mm_s,
        min_peak_distance_s=args.min_peak_distance_s,
        peak_width_rel_height=args.peak_width_rel_height,
        peak_event_boundary_mode=args.peak_event_boundary_mode,
        shape_split_policy=args.shape_split_policy,
        default_level=args.default_level,
        overwrite=args.overwrite,
        boundary_mode=args.boundary_mode,
        boundary_window_s=args.boundary_window_s,
        exponential_tau_s=args.exponential_tau_s,
        exponential_source_level=args.exponential_source_level,
        layout=args.layout,
        command=" ".join(sys.argv),
    )

    return 0


if __name__ == '__main__':
    exit(main())
