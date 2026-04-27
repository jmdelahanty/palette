#!/usr/bin/env python3
"""
Detect swim bouts from all 4 speed processing levels in track kinematics tracks.

This script reads pre-computed speed data from track kinematics tracks (raw,
filtered, smoothed, averaged) and detects swim bouts using the same threshold
across all levels. Results are stored hierarchically under a single run name
with 4 subgroups.

Storage structure:
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
    ├── default_level = "speed_smoothed" or "speed_filtered" (attr)
    └── run_metadata (attrs: threshold, source_track_kinematics_run, etc.)

Usage (basic):
    scripts/py -m fisheye.analysis.detect_bouts_multi_level /path/to/archive.zarr

    Auto-generates run name like: swim_bout_detect_20250905_143022

Usage (with options):
    scripts/py -m fisheye.analysis.detect_bouts_multi_level /path/to/archive.zarr \\
        --run-name custom_run \\
        --track-kinematics-run latest \\
        --threshold-mm 5.0 \\
        --default-level filtered \\
        --boundary-mode local_minimum \\
        --overwrite
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import signal

from fisheye.analysis.chaser_state_interpolator import write_columnar_dataset
from fisheye.shared.stage_provenance import build_stage_provenance, write_stage_provenance
from fisheye.utils.system import get_environment_info, get_git_info
from fisheye.utils.zarr_io import open_zarr_root


SPEED_LEVELS = ("speed_raw", "speed_filtered", "speed_smoothed", "speed_averaged")
PATH_DISTANCE_LEVEL_SOURCE = {
    "speed_raw": "raw",
    "speed_filtered": "filtered",
    "speed_smoothed": "smoothed",
    # speed_averaged is a moving average of the smoothed speed trace; it does
    # not have an independent frame-path-distance array.
    "speed_averaged": "smoothed",
}
SPEED_LEVEL_ALIASES = {
    "raw": "speed_raw",
    "filtered": "speed_filtered",
    "smoothed": "speed_smoothed",
    "averaged": "speed_averaged",
}
SPEED_LEVEL_CHOICES = tuple(SPEED_LEVEL_ALIASES) + SPEED_LEVELS
BOUNDARY_MODES = ("threshold", "local_minimum")


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
        ('peak_speed_mm_s', 'f8'),
        ('n_valid_transitions', 'i8'),
        ('n_invalid_transitions', 'i8'),
        ('valid_transition_fraction', 'f8'),
        ('gap_censored', '?'),
        ('start_time_s', 'f8'),
        ('end_time_s', 'f8'),
        ('core_start_time_s', 'f8'),
        ('core_end_time_s', 'f8'),
    ])


def _empty_bouts() -> np.ndarray:
    return np.zeros(0, dtype=_bout_dtype())


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
    delta_seconds: Optional[np.ndarray] = None,
    transition_valid: Optional[np.ndarray] = None,
    sample_valid: Optional[np.ndarray] = None,
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

        bout_speeds = speed[start_idx:end_exclusive]
        valid_speeds = bout_speeds[np.isfinite(bout_speeds)]
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
        peak_speed_mm_s = float(np.max(valid_speeds)) if len(valid_speeds) > 0 else float("nan")

        end_idx = end_exclusive - 1
        core_end_idx = core_end_exclusive - 1
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
            peak_speed_mm_s,
            n_valid_transitions,
            n_invalid_transitions,
            valid_transition_fraction,
            gap_censored,
            frames[start_idx] / fps if fps > 0 else float('nan'),
            frames[end_idx] / fps if fps > 0 else float('nan'),
            frames[core_start_idx] / fps if fps > 0 else float('nan'),
            frames[core_end_idx] / fps if fps > 0 else float('nan'),
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
            float('nan'), float('nan'), float('nan'), 0,
            0, float('nan'), float('nan'), float('nan'), float('nan'), float('nan'),
        )

    return global_metrics


def _detect_bouts_from_speed(
    speed: np.ndarray,
    frames: np.ndarray,
    fps: float,
    threshold: float,
    min_bout_duration_s: float = 0.05,
    min_gap_duration_s: float = 0.1,
    boundary_mode: str = "threshold",
    boundary_window_s: float = 0.25,
    path_distance_mm: Optional[np.ndarray] = None,
    path_distance_px: Optional[np.ndarray] = None,
    positions_mm: Optional[np.ndarray] = None,
    positions_px: Optional[np.ndarray] = None,
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
        min_bout_duration_s: Minimum duration to count as bout
        min_gap_duration_s: Minimum gap between bouts
        boundary_mode: "threshold" keeps threshold-crossing boundaries;
            "local_minimum" expands start/end to nearby local minima while
            preserving core_* threshold boundaries.
        boundary_window_s: Maximum search window on each side for local-minimum
            boundary expansion.

    Returns:
        Structured array with bout data
    """
    # Convert duration thresholds to frames
    min_bout_frames = int(min_bout_duration_s * fps)
    min_gap_frames = int(min_gap_duration_s * fps)
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

    # Merge bouts separated by small gaps
    if len(starts) > 1:
        merged_starts = [starts[0]]
        merged_ends = []

        for i in range(1, len(starts)):
            gap = starts[i] - ends[i-1]
            if gap < min_gap_frames:
                # Merge with previous bout (extend end)
                continue
            else:
                merged_ends.append(ends[i-1])
                merged_starts.append(starts[i])

        merged_ends.append(ends[-1])
        starts = np.array(merged_starts)
        ends = np.array(merged_ends)

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
        delta_seconds=delta_seconds,
        transition_valid=transition_valid,
        sample_valid=sample_valid,
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
    boundary_mode: str = "threshold",
    boundary_window_s: float = 0.25,
    path_distance_mm: Optional[np.ndarray] = None,
    path_distance_px: Optional[np.ndarray] = None,
    positions_mm: Optional[np.ndarray] = None,
    positions_px: Optional[np.ndarray] = None,
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
        min_bout_duration_s: Minimum duration to count as bout
        min_gap_duration_s: Minimum gap between bouts
        boundary_mode: "threshold" keeps peak-width boundaries; "local_minimum"
            expands start/end to nearby local minima.
        boundary_window_s: Maximum search window on each side for local-minimum
            boundary expansion.

    Returns:
        Structured array with bout data
    """
    # Convert duration thresholds to frames
    min_bout_frames = int(min_bout_duration_s * fps)
    min_gap_frames = int(min_gap_duration_s * fps)
    boundary_window_frames = max(0, int(round(boundary_window_s * fps)))

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
            if gap < min_gap_frames:
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
        delta_seconds=delta_seconds,
        transition_valid=transition_valid,
        sample_valid=sample_valid,
    )


def _load_track_kinematics_track_speeds(
    zarr_path: Path,
    track_kinematics_run: str,
    track_id: int = 0,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """
    Load all 4 speed levels from a track kinematics track.

    Args:
        zarr_path: Path to capture zarr
        track_kinematics_run: Track kinematics run name (or "latest")
        track_id: Track ID to load

    Returns:
        Tuple of (speed_dict, metadata_dict)
        speed_dict keys: speed_raw_mm, speed_filtered_mm, speed_smoothed_mm, speed_averaged_mm, frames
        metadata_dict keys: fps, pixel_to_mm, n_frames, etc.
    """
    root = open_zarr_root(zarr_path, mode='r')

    # Navigate to track_kinematics_runs
    if 'analysis' not in root or 'track_kinematics_runs' not in root['analysis']:
        raise ValueError(f"No track_kinematics_runs found in {zarr_path}")

    track_kinematics_runs = root['analysis']['track_kinematics_runs']

    if 'offline' not in track_kinematics_runs:
        raise ValueError("No offline track_kinematics_runs found")

    offline_group = track_kinematics_runs['offline']

    # Resolve "latest" if needed
    if track_kinematics_run == "latest":
        track_kinematics_run = offline_group.attrs.get('latest')
        if track_kinematics_run is None:
            raise ValueError("No 'latest' offline track kinematics run found")

    if track_kinematics_run not in offline_group:
        raise ValueError(
            f"Track kinematics run '{track_kinematics_run}' not found in offline runs"
        )

    run_group = offline_group[track_kinematics_run]

    # Load track data
    tracks_group = run_group['tracks']
    track_key = f"id_{track_id}"

    if track_key not in tracks_group:
        raise ValueError(
            f"Track {track_key} not found in track kinematics run {track_kinematics_run}"
        )

    track_group = tracks_group[track_key]

    def _optional_track_array(name: str) -> Optional[np.ndarray]:
        if name not in track_group:
            return None
        return track_group[name][:]

    # Load speed arrays
    speeds = {
        'speed_raw_mm': track_group['speed_raw_mm'][:],
        'speed_filtered_mm': track_group['speed_filtered_mm'][:],
        'speed_smoothed_mm': track_group['speed_smoothed_mm'][:],
        'speed_averaged_mm': track_group['speed_averaged_mm'][:],
        'frames': track_group['frame_indices'][:],
        'frame_path_distance_raw_mm': _optional_track_array('frame_path_distance_raw_mm'),
        'frame_path_distance_raw_px': _optional_track_array('frame_path_distance_raw_px'),
        'frame_path_distance_filtered_mm': _optional_track_array('frame_path_distance_filtered_mm'),
        'frame_path_distance_filtered_px': _optional_track_array('frame_path_distance_filtered_px'),
        'frame_path_distance_smoothed_mm': _optional_track_array('frame_path_distance_smoothed_mm'),
        'frame_path_distance_smoothed_px': _optional_track_array('frame_path_distance_smoothed_px'),
        'delta_seconds': _optional_track_array('delta_seconds'),
        'transition_valid': _optional_track_array('transition_valid'),
        'sample_valid': _optional_track_array('sample_valid'),
    }

    # Load position data for bout_points (both px and mm)
    positions_mm = None
    if 'positions_mm' in track_group:
        positions_mm = track_group['positions_mm'][:]

    positions_px = None
    if 'positions_px' in track_group:
        positions_px = track_group['positions_px'][:]

    source_provenance = run_group.attrs.get('provenance')
    if not isinstance(source_provenance, dict):
        source_provenance = {}
    source_git = source_provenance.get('git')
    if not isinstance(source_git, dict):
        source_git = {}

    # Load metadata
    metadata = {
        'fps': run_group.attrs.get('fps', 60.0),
        'pixel_to_mm': run_group.attrs.get('pixel_to_mm'),
        'n_frames': len(speeds['frames']),
        'track_kinematics_run': track_kinematics_run,
        'track_kinematics_created_at_utc': run_group.attrs.get('created_at_utc'),
        'track_kinematics_stage': source_provenance.get('stage'),
        'track_kinematics_version': source_provenance.get('version'),
        'track_kinematics_git_commit': run_group.attrs.get('git_commit') or source_git.get('commit'),
        'track_kinematics_git_dirty': run_group.attrs.get('git_dirty', source_git.get('is_dirty')),
        'track_id': track_id,
        'positions_mm': positions_mm,
        'positions_px': positions_px,
    }

    return speeds, metadata


def _metric_inputs_for_level(
    speeds: Dict[str, Optional[np.ndarray]],
    metadata: Dict[str, Any],
    level: str,
) -> Dict[str, Optional[np.ndarray]]:
    path_source = PATH_DISTANCE_LEVEL_SOURCE[level]
    return {
        "path_distance_mm": speeds.get(f"frame_path_distance_{path_source}_mm"),
        "path_distance_px": speeds.get(f"frame_path_distance_{path_source}_px"),
        "positions_mm": metadata.get("positions_mm"),
        "positions_px": metadata.get("positions_px"),
        "delta_seconds": speeds.get("delta_seconds"),
        "transition_valid": speeds.get("transition_valid"),
        "sample_valid": speeds.get("sample_valid"),
    }


def detect_and_save_bouts(
    zarr_path: Path,
    run_name: Optional[str],
    track_kinematics_run: str = "latest",
    track_id: int = 0,
    method: str = "threshold",
    threshold_mm: float = 2.0,
    prominence: float = 1.0,
    min_peak_height: Optional[float] = None,
    rel_height: float = 0.9,
    min_bout_duration_s: float = 0.05,
    min_gap_duration_s: float = 0.1,
    default_level: str = "speed_smoothed",
    overwrite: bool = False,
    boundary_mode: str = "threshold",
    boundary_window_s: float = 0.25,
    command: Optional[str] = None,
) -> str:
    """
    Detect bouts from all 4 speed levels and save hierarchically.

    Args:
        zarr_path: Path to capture zarr
        run_name: Name for this bout detection run (auto-generated if None)
        track_kinematics_run: Track kinematics run name (or "latest")
        track_id: Track ID to analyze
        method: Detection method ("threshold" or "peak")
        threshold_mm: Speed threshold in mm/s (for threshold method)
        prominence: Minimum peak prominence in mm/s (for peak method)
        min_peak_height: Minimum absolute peak height in mm/s (for peak method, optional)
        rel_height: Relative height for bout boundaries (for peak method). Higher values
            (0.7-1.0) capture more of the bout tail (default: 0.9)
        min_bout_duration_s: Minimum bout duration
        min_gap_duration_s: Minimum gap between bouts
        default_level: Speed subgroup that downstream consumers should use by
            default. Accepts raw/filtered/smoothed/averaged aliases.
        overwrite: Delete and recreate an existing run with the same name.
        boundary_mode: Boundary mode for stored start/end. "threshold" stores
            threshold/peak-width boundaries; "local_minimum" expands to nearby
            low-speed minima and preserves threshold/peak boundaries in core_*.
        boundary_window_s: Local-minimum search window on each side of each
            threshold/peak core.
        command: Optional command string to record in stage provenance.

    Returns:
        The run name used (either provided or auto-generated)
    """
    default_level_key = normalize_speed_level(default_level)
    if boundary_mode not in BOUNDARY_MODES:
        expected = ", ".join(BOUNDARY_MODES)
        raise ValueError(f"Unsupported boundary_mode {boundary_mode!r}; expected one of: {expected}")

    print(f"\n{'='*60}")
    print(f"MULTI-LEVEL SWIM BOUT DETECTION")
    print(f"{'='*60}")
    print(f"Capture: {zarr_path.name}")
    print(f"Track kinematics run: {track_kinematics_run}")
    print(f"Track ID: {track_id}")
    print(f"Method: {method}")
    print(f"Boundary mode: {boundary_mode}")
    if boundary_mode == "local_minimum":
        print(f"Boundary window: {boundary_window_s} s")
    if method == "threshold":
        print(f"Threshold: {threshold_mm} mm/s")
    elif method == "peak":
        print(f"Prominence: {prominence} mm/s")
        if min_peak_height is not None:
            print(f"Min peak height: {min_peak_height} mm/s")
        print(f"Relative height: {rel_height}")
    print(f"Default level: {default_level_key}")
    print()

    # Load speed data
    print("Loading track kinematics track data...")
    speeds, metadata = _load_track_kinematics_track_speeds(
        zarr_path, track_kinematics_run, track_id
    )

    fps = metadata['fps']
    frames = speeds['frames']

    print(f"  FPS: {fps}")
    print(f"  Frames: {metadata['n_frames']}")
    print(f"  Track kinematics run: {metadata['track_kinematics_run']}")
    print()

    # Detect bouts for each speed level
    speed_levels = list(SPEED_LEVELS)
    bout_results = {}

    print("Detecting bouts for each speed level:")
    for level in speed_levels:
        speed_key = f"{level}_mm"
        speed = speeds[speed_key]
        metric_inputs = _metric_inputs_for_level(speeds, metadata, level)

        # Skip if all NaN
        if np.all(np.isnan(speed)):
            print(f"  {level}: SKIPPED (all NaN)")
            bout_results[level] = _empty_bouts()
            continue

        if method == "threshold":
            bouts = _detect_bouts_from_speed(
                speed,
                frames,
                fps,
                threshold_mm,
                min_bout_duration_s,
                min_gap_duration_s,
                boundary_mode,
                boundary_window_s,
                **metric_inputs,
            )
        elif method == "peak":
            bouts = _detect_bouts_from_peaks(
                speed,
                frames,
                fps,
                prominence,
                min_peak_height,
                rel_height,
                min_bout_duration_s,
                min_gap_duration_s,
                boundary_mode,
                boundary_window_s,
                **metric_inputs,
            )
        else:
            raise ValueError(f"Unknown detection method: {method}. Must be 'threshold' or 'peak'.")

        bout_results[level] = bouts
        print(f"  {level}: {len(bouts)} bouts detected")

    print()

    # Save to zarr
    print("Saving to zarr...")
    root = open_zarr_root(zarr_path, mode='r+')

    # Create analysis/swim_bout_runs if needed
    if 'analysis' not in root:
        analysis_group = root.create_group('analysis')
    else:
        analysis_group = root['analysis']

    if 'swim_bout_runs' not in analysis_group:
        swim_bout_runs = analysis_group.create_group('swim_bout_runs')
    else:
        swim_bout_runs = analysis_group['swim_bout_runs']

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

    # Save metadata at run level
    git_info = get_git_info()
    env_info = get_environment_info(
        disk_path=str(zarr_path),
        capture_env_vars=False,
    )
    created_at_utc = datetime.now(timezone.utc).isoformat()
    run_group.attrs['created_at_utc'] = created_at_utc
    run_group.attrs['detection_method'] = method
    run_group.attrs['min_bout_duration_s'] = min_bout_duration_s
    run_group.attrs['min_gap_duration_s'] = min_gap_duration_s
    run_group.attrs['boundary_mode'] = boundary_mode
    run_group.attrs['boundary_window_s'] = float(boundary_window_s)

    # Method-specific parameters
    if method == "threshold":
        run_group.attrs['threshold_mm'] = threshold_mm
    elif method == "peak":
        run_group.attrs['prominence'] = prominence
        run_group.attrs['min_peak_height'] = min_peak_height if min_peak_height is not None else float('nan')
        run_group.attrs['rel_height'] = rel_height

    run_group.attrs['source_track_kinematics_run'] = metadata['track_kinematics_run']
    run_group.attrs['track_id'] = track_id
    run_group.attrs['fps'] = fps
    run_group.attrs['pixel_to_mm'] = metadata.get('pixel_to_mm', float('nan'))
    run_group.attrs['default_level'] = default_level_key
    run_group.attrs['git_commit'] = git_info['commit_hash']
    run_group.attrs['git_branch'] = git_info['branch']
    run_group.attrs['git_dirty'] = git_info['is_dirty']

    parameters = {
        'method': method,
        'threshold_mm': float(threshold_mm) if method == "threshold" else None,
        'prominence': float(prominence) if method == "peak" else None,
        'min_peak_height': float(min_peak_height) if min_peak_height is not None else None,
        'rel_height': float(rel_height) if method == "peak" else None,
        'min_bout_duration_s': float(min_bout_duration_s),
        'min_gap_duration_s': float(min_gap_duration_s),
        'default_level': default_level_key,
        'boundary_mode': boundary_mode,
        'boundary_window_s': float(boundary_window_s),
        'speed_levels': list(speed_levels),
        'bout_metric_schema_id': 'palette.swim_bout_metrics.v2',
        'distance_policy': 'path_length_from_track_frame_path_distance_only',
        'overwrite': bool(overwrite),
    }
    inputs = {
        'zarr_path': str(zarr_path),
        'source_track_kinematics_run': metadata['track_kinematics_run'],
        'source_track_kinematics_stage': metadata.get('track_kinematics_stage'),
        'source_track_kinematics_version': metadata.get('track_kinematics_version'),
        'source_track_path': (
            f"analysis/track_kinematics_runs/offline/"
            f"{metadata['track_kinematics_run']}/tracks/id_{int(track_id)}"
        ),
        'source_track_kinematics_created_at_utc': metadata.get(
            'track_kinematics_created_at_utc'
        ),
        'source_track_kinematics_git_commit': metadata.get('track_kinematics_git_commit'),
        'source_track_kinematics_git_dirty': metadata.get('track_kinematics_git_dirty'),
        'track_id': int(track_id),
        'fps': float(fps),
        'pixel_to_mm': metadata.get('pixel_to_mm'),
        'n_frames': int(metadata['n_frames']),
    }
    provenance = build_stage_provenance(
        stage="detect_bouts_multi_level",
        created_at_utc=created_at_utc,
        parameters=parameters,
        inputs=inputs,
        command=command,
        version="detect_bouts_multi_level.v1",
        git=git_info,
        environment=env_info.get("environment"),
        platform=env_info.get("platform"),
        artifacts={
            'run_path': f"analysis/swim_bout_runs/{run_name}",
            'default_level': default_level_key,
        },
    )
    write_stage_provenance(run_group, provenance)

    # Save each speed level's bouts and statistics in subgroups
    for level in speed_levels:
        level_group = run_group.create_group(level)
        bouts = bout_results[level]

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

        # Save datasets using columnar format (Zarr v3-stable)
        level_specific_attrs = {
            'n_bouts': len(bouts),
            'speed_level': level,
            'bout_metric_schema_id': 'palette.swim_bout_metrics.v2',
            'path_distance_source_level': PATH_DISTANCE_LEVEL_SOURCE[level],
            'is_default_level': level == default_level_key,
        }

        if len(bouts) > 0:
            level_specific_attrs['total_bout_time_s'] = float(np.sum(bouts['duration_s']))
            level_specific_attrs['total_observed_bout_time_s'] = float(np.sum(bouts['observed_duration_s']))
            level_specific_attrs['mean_bout_duration_s'] = float(np.mean(bouts['duration_s']))
            level_specific_attrs['mean_bout_observed_duration_s'] = float(np.mean(bouts['observed_duration_s']))
            level_specific_attrs['mean_bout_speed_mm_s'] = _finite_mean_or_nan(bouts['mean_speed_mm_s'])
            level_specific_attrs['total_path_length_mm'] = _finite_sum_or_nan(bouts['path_length_mm'])
            level_specific_attrs['mean_valid_transition_fraction'] = _finite_mean_or_nan(
                bouts['valid_transition_fraction']
            )
            level_specific_attrs['n_gap_censored_bouts'] = int(np.sum(bouts['gap_censored']))

        write_columnar_dataset(level_group, 'bouts', bouts, attrs=level_specific_attrs)
        write_columnar_dataset(level_group, 'inter_bout_intervals', intervals, attrs=None)
        write_columnar_dataset(level_group, 'inter_bout_interval_histogram', interval_histogram, attrs=None)
        write_columnar_dataset(level_group, 'global_metrics', global_metrics, attrs=None)
        write_columnar_dataset(level_group, 'bout_points', bout_points, attrs=None)

        print(f"  Saved {level}: {len(bouts)} bouts, {len(intervals)} intervals")

    # Update latest pointer
    swim_bout_runs.attrs['latest'] = run_name

    print()
    print(f"{'='*60}")
    print(f"DETECTION COMPLETE")
    print(f"{'='*60}")
    print(f"Run saved: analysis/swim_bout_runs/{run_name}")
    print(f"  speed_raw: {len(bout_results['speed_raw'])} bouts")
    print(f"  speed_filtered: {len(bout_results['speed_filtered'])} bouts")
    print(f"  speed_smoothed: {len(bout_results['speed_smoothed'])} bouts")
    print(f"  speed_averaged: {len(bout_results['speed_averaged'])} bouts")
    print(f"Default level: {default_level_key}")
    print()

    return run_name


def main():
    parser = argparse.ArgumentParser(
        description="Detect swim bouts from all 4 speed processing levels",
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
        choices=['threshold', 'peak'],
        default='threshold',
        help='Detection method: "threshold" or "peak" (default: threshold)',
    )

    # Threshold method parameters
    parser.add_argument(
        '--threshold-mm',
        type=float,
        default=2.0,
        help='Speed threshold in mm/s for threshold method (default: 2.0)',
    )

    parser.add_argument(
        '--default-level',
        type=str,
        choices=SPEED_LEVEL_CHOICES,
        default='speed_smoothed',
        help=(
            'Speed level downstream consumers should use by default. '
            'Accepts raw/filtered/smoothed/averaged aliases or stored subgroup '
            'names. Default: speed_smoothed.'
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

    # Common parameters
    parser.add_argument(
        '--min-bout-duration',
        type=float,
        default=0.05,
        help='Minimum bout duration in seconds (default: 0.05)',
    )

    parser.add_argument(
        '--min-gap-duration',
        type=float,
        default=0.1,
        help='Minimum gap between bouts in seconds (default: 0.1)',
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
        default_level=args.default_level,
        overwrite=args.overwrite,
        boundary_mode=args.boundary_mode,
        boundary_window_s=args.boundary_window_s,
        command=" ".join(sys.argv),
    )

    return 0


if __name__ == '__main__':
    exit(main())
