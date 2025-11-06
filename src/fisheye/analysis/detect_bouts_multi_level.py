#!/usr/bin/env python3
"""
Detect swim bouts from all 4 speed processing levels in movement tracks.

This script reads pre-computed speed data from movement tracks (raw, filtered,
smoothed, averaged) and detects swim bouts using the same threshold across all
levels. Results are stored hierarchically under a single run name with 4 subgroups.

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
    ├── default_level = "speed_smoothed" (attr)
    └── run_metadata (attrs: threshold, source_movement_run, etc.)

Usage (basic):
    python -m fisheye.analysis.detect_bouts_multi_level /path/to/archive.zarr

    Auto-generates run name like: swim_bout_detect_20250905_143022

Usage (with options):
    python -m fisheye.analysis.detect_bouts_multi_level /path/to/archive.zarr \\
        --run-name custom_run \\
        --movement-run latest \\
        --threshold-mm 5.0
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import zarr
from scipy import signal

from fisheye.analysis.chaser_state_interpolator import write_columnar_dataset
from fisheye.utils.system import get_git_info


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
        ('percent_active', 'f8'),
        ('mean_bout_duration_s', 'f8'),
        ('mean_bout_distance', 'f8'),
        ('mean_bout_speed', 'f8'),
        ('total_distance', 'f8'),
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
        percent_active = (total_active_time / total_duration_s * 100.0) if total_duration_s > 0 else 0.0
        bout_rate_per_min = (n_bouts / total_duration_s * 60.0) if total_duration_s > 0 else 0.0

        global_metrics[0] = (
            n_bouts,
            bout_rate_per_min,
            total_active_time,
            percent_active,
            float(np.mean(bouts['duration_s'])),
            float(np.mean(bouts['distance'])),
            float(np.mean(bouts['mean_speed'])),
            float(np.sum(bouts['distance'])),
            0,  # Will be updated with interval metrics
            float('nan'),
            float('nan'),
            float('nan'),
            float('nan'),
            float('nan'),
        )
    else:
        global_metrics[0] = (
            0, 0.0, 0.0, 0.0, float('nan'), float('nan'), float('nan'), 0.0,
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

    Returns:
        Structured array with bout data
    """
    # Convert duration thresholds to frames
    min_bout_frames = int(min_bout_duration_s * fps)
    min_gap_frames = int(min_gap_duration_s * fps)

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

    # Filter by minimum duration
    durations = ends - starts
    valid_bouts = durations >= min_bout_frames
    starts = starts[valid_bouts]
    ends = ends[valid_bouts]

    # Build structured array
    n_bouts = len(starts)

    bout_dtype = np.dtype([
        ('bout_id', 'i4'),
        ('start_frame', 'i8'),
        ('end_frame', 'i8'),
        ('duration_frames', 'i8'),
        ('duration_s', 'f8'),
        ('distance', 'f8'),
        ('mean_speed', 'f8'),
        ('peak_speed', 'f8'),
        ('start_time_s', 'f8'),
        ('end_time_s', 'f8'),
    ])

    bouts = np.zeros(n_bouts, dtype=bout_dtype)

    for i, (start_idx, end_idx) in enumerate(zip(starts, ends)):
        # Get bout speed slice (inclusive of start, exclusive of end)
        bout_speeds = speed[start_idx:end_idx]
        valid_speeds = bout_speeds[np.isfinite(bout_speeds)]

        # Calculate statistics
        duration_frames = end_idx - start_idx
        duration_s = duration_frames / fps

        if len(valid_speeds) > 0:
            mean_speed = np.mean(valid_speeds)
            peak_speed = np.max(valid_speeds)
            # Estimate distance as mean speed * duration
            distance = mean_speed * duration_s
        else:
            mean_speed = 0.0
            peak_speed = 0.0
            distance = 0.0

        bouts[i] = (
            i + 1,  # bout_id
            int(frames[start_idx]),  # start_frame
            int(frames[end_idx - 1]),  # end_frame (inclusive)
            duration_frames,
            duration_s,
            distance,
            mean_speed,
            peak_speed,
            frames[start_idx] / fps,  # start_time_s
            frames[end_idx - 1] / fps,  # end_time_s
        )

    return bouts


def _detect_bouts_from_peaks(
    speed: np.ndarray,
    frames: np.ndarray,
    fps: float,
    prominence: float,
    min_peak_height: Optional[float] = None,
    rel_height: float = 0.9,
    min_bout_duration_s: float = 0.05,
    min_gap_duration_s: float = 0.1,
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

    Returns:
        Structured array with bout data
    """
    # Convert duration thresholds to frames
    min_bout_frames = int(min_bout_duration_s * fps)
    min_gap_frames = int(min_gap_duration_s * fps)

    # Handle NaN values - replace with 0 for peak detection
    speed_clean = np.where(np.isnan(speed), 0.0, speed)

    # Find peaks with prominence criterion
    peak_kwargs = {'prominence': prominence}
    if min_peak_height is not None:
        peak_kwargs['height'] = min_peak_height

    peaks, properties = signal.find_peaks(speed_clean, **peak_kwargs)

    # If no peaks found, return empty array
    if len(peaks) == 0:
        bout_dtype = np.dtype([
            ('bout_id', 'i4'),
            ('start_frame', 'i8'),
            ('end_frame', 'i8'),
            ('duration_frames', 'i8'),
            ('duration_s', 'f8'),
            ('distance', 'f8'),
            ('mean_speed', 'f8'),
            ('peak_speed', 'f8'),
            ('start_time_s', 'f8'),
            ('end_time_s', 'f8'),
        ])
        return np.zeros(0, dtype=bout_dtype)

    # Get bout boundaries using peak widths at relative height
    widths, width_heights, left_ips, right_ips = signal.peak_widths(
        speed_clean,
        peaks,
        rel_height=rel_height
    )

    # Convert interpolated positions to integer indices
    # Round down for start, round up for end to be inclusive
    starts = np.floor(left_ips).astype(int)
    ends = np.ceil(right_ips).astype(int)

    # Ensure indices are within bounds
    starts = np.clip(starts, 0, len(speed) - 1)
    ends = np.clip(ends, 0, len(speed) - 1)

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

    # Filter by minimum duration
    durations = ends - starts
    valid_bouts = durations >= min_bout_frames
    starts = starts[valid_bouts]
    ends = ends[valid_bouts]

    # Build structured array
    n_bouts = len(starts)

    bout_dtype = np.dtype([
        ('bout_id', 'i4'),
        ('start_frame', 'i8'),
        ('end_frame', 'i8'),
        ('duration_frames', 'i8'),
        ('duration_s', 'f8'),
        ('distance', 'f8'),
        ('mean_speed', 'f8'),
        ('peak_speed', 'f8'),
        ('start_time_s', 'f8'),
        ('end_time_s', 'f8'),
    ])

    bouts = np.zeros(n_bouts, dtype=bout_dtype)

    for i, (start_idx, end_idx) in enumerate(zip(starts, ends)):
        # Get bout speed slice (inclusive of start, exclusive of end+1)
        bout_speeds = speed[start_idx:end_idx+1]
        valid_speeds = bout_speeds[np.isfinite(bout_speeds)]

        # Calculate statistics
        duration_frames = end_idx - start_idx + 1
        duration_s = duration_frames / fps

        if len(valid_speeds) > 0:
            mean_speed = np.mean(valid_speeds)
            peak_speed = np.max(valid_speeds)
            # Estimate distance as mean speed * duration
            distance = mean_speed * duration_s
        else:
            mean_speed = 0.0
            peak_speed = 0.0
            distance = 0.0

        bouts[i] = (
            i + 1,  # bout_id
            int(frames[start_idx]),  # start_frame
            int(frames[end_idx]),  # end_frame
            duration_frames,
            duration_s,
            distance,
            mean_speed,
            peak_speed,
            frames[start_idx] / fps,  # start_time_s
            frames[end_idx] / fps,  # end_time_s
        )

    return bouts


def _load_movement_track_speeds(
    zarr_path: Path,
    movement_run: str,
    track_id: int = 0,
) -> Tuple[Dict[str, np.ndarray], Dict[str, any]]:
    """
    Load all 4 speed levels from a movement track.

    Args:
        zarr_path: Path to capture zarr
        movement_run: Movement run name (or "latest")
        track_id: Track ID to load

    Returns:
        Tuple of (speed_dict, metadata_dict)
        speed_dict keys: speed_raw_mm, speed_filtered_mm, speed_smoothed_mm, speed_averaged_mm, frames
        metadata_dict keys: fps, pixel_to_mm, n_frames, etc.
    """
    root = zarr.open(str(zarr_path), mode='r')

    # Navigate to movement_runs
    if 'analysis' not in root or 'movement_runs' not in root['analysis']:
        raise ValueError(f"No movement_runs found in {zarr_path}")

    movement_runs = root['analysis']['movement_runs']

    if 'offline' not in movement_runs:
        raise ValueError("No offline movement_runs found")

    offline_group = movement_runs['offline']

    # Resolve "latest" if needed
    if movement_run == "latest":
        movement_run = offline_group.attrs.get('latest')
        if movement_run is None:
            raise ValueError("No 'latest' offline movement run found")

    if movement_run not in offline_group:
        raise ValueError(f"Movement run '{movement_run}' not found in offline runs")

    run_group = offline_group[movement_run]

    # Load track data
    tracks_group = run_group['tracks']
    track_key = f"id_{track_id}"

    if track_key not in tracks_group:
        raise ValueError(f"Track {track_key} not found in movement run {movement_run}")

    track_group = tracks_group[track_key]

    # Load speed arrays
    speeds = {
        'speed_raw_mm': track_group['speed_raw_mm'][:],
        'speed_filtered_mm': track_group['speed_filtered_mm'][:],
        'speed_smoothed_mm': track_group['speed_smoothed_mm'][:],
        'speed_averaged_mm': track_group['speed_averaged_mm'][:],
        'frames': track_group['frame_indices'][:],
    }

    # Load position data for bout_points (both px and mm)
    positions_mm = None
    if 'positions_mm' in track_group:
        positions_mm = track_group['positions_mm'][:]

    positions_px = None
    if 'positions_px' in track_group:
        positions_px = track_group['positions_px'][:]

    # Load metadata
    metadata = {
        'fps': run_group.attrs.get('fps', 60.0),
        'pixel_to_mm': run_group.attrs.get('pixel_to_mm'),
        'n_frames': len(speeds['frames']),
        'movement_run': movement_run,
        'track_id': track_id,
        'positions_mm': positions_mm,
        'positions_px': positions_px,
    }

    return speeds, metadata


def detect_and_save_bouts(
    zarr_path: Path,
    run_name: Optional[str],
    movement_run: str = "latest",
    track_id: int = 0,
    method: str = "threshold",
    threshold_mm: float = 2.0,
    prominence: float = 1.0,
    min_peak_height: Optional[float] = None,
    rel_height: float = 0.9,
    min_bout_duration_s: float = 0.05,
    min_gap_duration_s: float = 0.1,
) -> str:
    """
    Detect bouts from all 4 speed levels and save hierarchically.

    Args:
        zarr_path: Path to capture zarr
        run_name: Name for this bout detection run (auto-generated if None)
        movement_run: Movement run name (or "latest")
        track_id: Track ID to analyze
        method: Detection method ("threshold" or "peak")
        threshold_mm: Speed threshold in mm/s (for threshold method)
        prominence: Minimum peak prominence in mm/s (for peak method)
        min_peak_height: Minimum absolute peak height in mm/s (for peak method, optional)
        rel_height: Relative height for bout boundaries (for peak method). Higher values
            (0.7-1.0) capture more of the bout tail (default: 0.9)
        min_bout_duration_s: Minimum bout duration
        min_gap_duration_s: Minimum gap between bouts

    Returns:
        The run name used (either provided or auto-generated)
    """
    print(f"\n{'='*60}")
    print(f"MULTI-LEVEL SWIM BOUT DETECTION")
    print(f"{'='*60}")
    print(f"Capture: {zarr_path.name}")
    print(f"Movement run: {movement_run}")
    print(f"Track ID: {track_id}")
    print(f"Method: {method}")
    if method == "threshold":
        print(f"Threshold: {threshold_mm} mm/s")
    elif method == "peak":
        print(f"Prominence: {prominence} mm/s")
        if min_peak_height is not None:
            print(f"Min peak height: {min_peak_height} mm/s")
        print(f"Relative height: {rel_height}")
    print()

    # Load speed data
    print("Loading movement track data...")
    speeds, metadata = _load_movement_track_speeds(zarr_path, movement_run, track_id)

    fps = metadata['fps']
    frames = speeds['frames']

    print(f"  FPS: {fps}")
    print(f"  Frames: {metadata['n_frames']}")
    print(f"  Movement run: {metadata['movement_run']}")
    print()

    # Detect bouts for each speed level
    speed_levels = ['speed_raw', 'speed_filtered', 'speed_smoothed', 'speed_averaged']
    bout_results = {}

    print("Detecting bouts for each speed level:")
    for level in speed_levels:
        speed_key = f"{level}_mm"
        speed = speeds[speed_key]

        # Skip if all NaN
        if np.all(np.isnan(speed)):
            print(f"  {level}: SKIPPED (all NaN)")
            bout_results[level] = np.array([], dtype=np.dtype([
                ('bout_id', 'i4'),
                ('start_frame', 'i8'),
                ('end_frame', 'i8'),
                ('duration_frames', 'i8'),
                ('duration_s', 'f8'),
                ('distance', 'f8'),
                ('mean_speed', 'f8'),
                ('peak_speed', 'f8'),
                ('start_time_s', 'f8'),
                ('end_time_s', 'f8'),
            ]))
            continue

        if method == "threshold":
            bouts = _detect_bouts_from_speed(
                speed,
                frames,
                fps,
                threshold_mm,
                min_bout_duration_s,
                min_gap_duration_s,
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
            )
        else:
            raise ValueError(f"Unknown detection method: {method}. Must be 'threshold' or 'peak'.")

        bout_results[level] = bouts
        print(f"  {level}: {len(bouts)} bouts detected")

    print()

    # Save to zarr
    print("Saving to zarr...")
    root = zarr.open(str(zarr_path), mode='r+')

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
        raise ValueError(f"Swim bout run '{run_name}' already exists. Use a different name or delete the existing run.")

    # Create run group
    run_group = swim_bout_runs.create_group(run_name)

    # Save metadata at run level
    git_info = get_git_info()
    run_group.attrs['created_at_utc'] = datetime.now(timezone.utc).isoformat()
    run_group.attrs['detection_method'] = method
    run_group.attrs['min_bout_duration_s'] = min_bout_duration_s
    run_group.attrs['min_gap_duration_s'] = min_gap_duration_s

    # Method-specific parameters
    if method == "threshold":
        run_group.attrs['threshold_mm'] = threshold_mm
    elif method == "peak":
        run_group.attrs['prominence'] = prominence
        run_group.attrs['min_peak_height'] = min_peak_height if min_peak_height is not None else float('nan')
        run_group.attrs['rel_height'] = rel_height

    run_group.attrs['source_movement_run'] = metadata['movement_run']
    run_group.attrs['track_id'] = track_id
    run_group.attrs['fps'] = fps
    run_group.attrs['pixel_to_mm'] = metadata.get('pixel_to_mm', float('nan'))
    run_group.attrs['default_level'] = 'speed_smoothed'
    run_group.attrs['git_commit'] = git_info['commit_hash']
    run_group.attrs['git_branch'] = git_info['branch']
    run_group.attrs['git_dirty'] = git_info['is_dirty']

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
        }

        if len(bouts) > 0:
            level_specific_attrs['total_bout_time_s'] = float(np.sum(bouts['duration_s']))
            level_specific_attrs['mean_bout_duration_s'] = float(np.mean(bouts['duration_s']))
            level_specific_attrs['mean_bout_speed'] = float(np.mean(bouts['mean_speed']))
            level_specific_attrs['total_distance'] = float(np.sum(bouts['distance']))

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
    print(f"Default level: speed_smoothed")
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
        '--movement-run',
        type=str,
        default='latest',
        help='Movement run name (default: latest)',
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
        movement_run=args.movement_run,
        track_id=args.track_id,
        method=args.method,
        threshold_mm=args.threshold_mm,
        prominence=args.prominence,
        min_peak_height=args.min_peak_height,
        rel_height=args.rel_height,
        min_bout_duration_s=args.min_bout_duration,
        min_gap_duration_s=args.min_gap_duration,
    )

    return 0


if __name__ == '__main__':
    exit(main())
