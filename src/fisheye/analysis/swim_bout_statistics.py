#!/usr/bin/env python3
"""
Generate swimming-bout statistics for Palette archives with provenance.

This tool wraps ``EnhancedBoutAnalyzer`` from ``chaser_analysis.swimming_bout_analysis``
to compute bout metrics for the entire session and for per-trial segments defined
by stimulus events imported via ``analysis/stimulus_runs``.

Outputs a JSON report containing:
  • Global bout metrics across the full session
  • Per-trial bout metrics (based on configurable start/end event types)
  • Swim bout coverage and rates
  • Full provenance (git hash, CLI arguments, Zarr run lineage, calibration info)

Note:
    We persist the results as numpy structured arrays in Zarr. The Zarr v3
    specification currently marks these dtypes (and the fixed-length byte field
    used in ``bout_points``) as unstable, so zarr-python emits
    ``UnstableSpecificationWarning`` when writing. The data is still written, and
    we accept the warning for now; we will revisit when an official dtype spec is
    published upstream.
"""

from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import zarr

from chaser_analysis.swimming_bout_analysis import (
    BoutWithUnits,
    EnhancedBoutAnalyzer,
    CalibrationData,
)
from fisheye.utils.system import get_git_info

try:
    from zarr.errors import ZarrFutureWarning, UnstableSpecificationWarning  # type: ignore
except Exception:  # pragma: no cover
    class ZarrFutureWarning(Warning):
        pass

    class UnstableSpecificationWarning(ZarrFutureWarning):
        pass

warnings.filterwarnings("ignore", category=UnstableSpecificationWarning)
warnings.filterwarnings("ignore", category=ZarrFutureWarning)

EXPERIMENT_EVENT_TYPE = {
    0: "PROTOCOL_START",
    1: "PROTOCOL_STOP",
    2: "PROTOCOL_PAUSE",
    3: "PROTOCOL_RESUME",
    4: "PROTOCOL_FINISH",
    5: "PROTOCOL_CLEAR",
    6: "PROTOCOL_LOAD",
    7: "STEP_ADD",
    8: "STEP_REMOVE",
    9: "STEP_MOVE_UP",
    10: "STEP_MOVE_DOWN",
    11: "STEP_START",
    12: "STEP_END",
    13: "ITI_START",
    14: "ITI_END",
    15: "PARAMS_APPLIED",
    16: "MANAGER_REINIT",
    17: "MANAGER_REINIT_FAIL",
    18: "LOOM_AUTO_REPEAT_TRIGGER",
    19: "LOOM_MANUAL_START",
    20: "USER_INTERVENTION",
    21: "ERROR_RUNTIME",
    22: "LOG_MESSAGE",
    23: "IPC_BOUNDING_BOX_RECEIVED",
    24: "CHASER_PRE_PERIOD_START",
    25: "CHASER_TRAINING_START",
    26: "CHASER_POST_PERIOD_START",
    27: "CHASER_CHASE_SEQUENCE_START",
    28: "CHASER_CHASE_SEQUENCE_END",
    29: "CHASER_RANDOM_TARGET_SET",
}
EVENT_NAME_TO_ID = {v: k for k, v in EXPERIMENT_EVENT_TYPE.items()}


def _normalize_column(values: np.ndarray) -> np.ndarray:
    """Decode byte arrays to unicode for DataFrame reconstruction."""
    if values.ndim > 1:
        if values.dtype == np.uint8:
            decoded: List[str] = []
            for row in values:
                if row.ndim != 1:
                    decoded.append("")
                    continue
                decoded.append(bytes(row).rstrip(b"\x00").decode("utf-8", errors="ignore"))
            return np.array(decoded, dtype=object)
        # For other dtypes, flatten the trailing dimensions so each entry becomes a tuple/list
        flattened = values.reshape(values.shape[0], -1)
        flattened_objects = [tuple(row.tolist()) for row in flattened]
        return np.array(flattened_objects, dtype=object)
    if values.dtype.kind == "S":
        return np.char.decode(values, "utf-8")
    if values.dtype.kind == "U":
        return np.asarray(values, dtype=object)
    if values.dtype.kind == "O":
        decoded = [
            item.decode("utf-8", errors="ignore") if isinstance(item, (bytes, bytearray)) else item
            for item in values
        ]
        return np.array(decoded, dtype=object)
    return values


def _to_serializable(value: Any) -> Any:
    """Recursively convert values to JSON-serializable primitives."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [_to_serializable(v) for v in value.tolist()]
    if isinstance(value, (list, tuple, set)):
        return [_to_serializable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _to_serializable(v) for k, v in value.items()}
    return value


def _structured_to_dicts(array: np.ndarray) -> List[Dict[str, Any]]:
    """Convert a structured numpy array into list-of-dicts."""
    if array.size == 0:
        return []
    records: List[Dict[str, Any]] = []
    for row in array:
        record: Dict[str, Any] = {}
        for name in array.dtype.names:
            value = row[name]
            if isinstance(value, bytes):
                record[name] = value.decode("utf-8")
            else:
                record[name] = _to_serializable(value)
        records.append(record)
    return records


def _first_valid(values: np.ndarray) -> float:
    """Return first non-NaN value from array, else NaN."""
    for val in values:
        if not np.isnan(val):
            return float(val)
    return float("nan")


def _last_valid(values: np.ndarray) -> float:
    """Return last non-NaN value from array, else NaN."""
    for val in values[::-1]:
        if not np.isnan(val):
            return float(val)
    return float("nan")


def _make_global_array(global_metrics: Dict[str, Any]) -> np.ndarray:
    """Convert global metrics dict to structured array."""
    keys = sorted(global_metrics.keys())
    dtype = [(key, "f8") for key in keys]
    data = np.zeros(1, dtype=dtype)
    for key in keys:
        value = global_metrics.get(key)
        data[key][0] = float(value) if value is not None else np.nan
    return data


def _make_trial_array(trials: List[Dict[str, Any]]) -> np.ndarray:
    """Convert trial metrics list into structured array."""
    base_dtype: List[Tuple[str, Any]] = [
        ("trial_id", "i4"),
        ("start_frame", "i8"),
        ("end_frame", "i8"),
        ("start_time_s", "f8"),
        ("end_time_s", "f8"),
    ]
    metrics_keys = sorted({key for trial in trials for key in trial.get("metrics", {}).keys()})
    dtype = base_dtype + [(key, "f8") for key in metrics_keys]
    data = np.zeros(len(trials), dtype=dtype)
    for idx, trial in enumerate(trials):
        data["trial_id"][idx] = int(trial["trial_id"])
        data["start_frame"][idx] = int(trial["start_frame"])
        data["end_frame"][idx] = int(trial["end_frame"])
        data["start_time_s"][idx] = float(trial["start_time_s"])
        data["end_time_s"][idx] = float(trial["end_time_s"])
        metrics = trial.get("metrics", {})
        for key in metrics_keys:
            value = metrics.get(key)
            data[key][idx] = float(value) if value is not None else np.nan
    if len(trials) == 0:
        # ensure dtype defined even for empty array
        data = np.zeros(0, dtype=dtype)
    return data


def _make_bout_arrays(
    analyzer: EnhancedBoutAnalyzer,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build structured arrays for bouts and bout start/end points."""
    bouts = analyzer.bouts
    calibration = analyzer.calibration
    has_calibration = calibration is not None

    base_dtype: List[Tuple[str, Any]] = [
        ("bout_id", "i4"),
        ("start_frame", "i8"),
        ("end_frame", "i8"),
        ("duration_frames", "i8"),
        ("duration_s", "f8"),
        ("start_time_s", "f8"),
        ("end_time_s", "f8"),
        ("distance_px", "f8"),
        ("mean_speed_px_s", "f8"),
        ("peak_speed_px_s", "f8"),
        ("start_x_px", "f8"),
        ("start_y_px", "f8"),
        ("end_x_px", "f8"),
        ("end_y_px", "f8"),
    ]
    if has_calibration:
        base_dtype += [
            ("distance_mm", "f8"),
            ("distance_bl", "f8"),
            ("mean_speed_mm_s", "f8"),
            ("mean_speed_bl_s", "f8"),
            ("peak_speed_mm_s", "f8"),
            ("peak_speed_bl_s", "f8"),
            ("start_x_mm", "f8"),
            ("start_y_mm", "f8"),
            ("end_x_mm", "f8"),
            ("end_y_mm", "f8"),
        ]

    bout_array = np.zeros(len(bouts), dtype=base_dtype)
    point_dtype: List[Tuple[str, Any]] = [
        ("bout_id", "i4"),
        ("point_type", "S5"),
        ("frame", "i8"),
        ("time_s", "f8"),
        ("x_px", "f8"),
        ("y_px", "f8"),
    ]
    if has_calibration:
        point_dtype += [
            ("x_mm", "f8"),
            ("y_mm", "f8"),
        ]
    point_array = np.zeros(len(bouts) * 2, dtype=point_dtype)

    for idx, bout in enumerate(bouts):
        start_x = _first_valid(bout.positions_x)
        start_y = _first_valid(bout.positions_y)
        end_x = _last_valid(bout.positions_x)
        end_y = _last_valid(bout.positions_y)

        bout_array["bout_id"][idx] = int(bout.bout_id)
        bout_array["start_frame"][idx] = int(bout.start_frame)
        bout_array["end_frame"][idx] = int(bout.end_frame)
        bout_array["duration_frames"][idx] = int(bout.duration_frames)
        bout_array["duration_s"][idx] = float(bout.duration_s)
        bout_array["start_time_s"][idx] = float(bout.start_time_s)
        bout_array["end_time_s"][idx] = float(bout.end_time_s)
        bout_array["distance_px"][idx] = float(bout.distance_px)
        bout_array["mean_speed_px_s"][idx] = float(bout.mean_speed_px_s)
        bout_array["peak_speed_px_s"][idx] = float(bout.peak_speed_px_s)
        bout_array["start_x_px"][idx] = start_x
        bout_array["start_y_px"][idx] = start_y
        bout_array["end_x_px"][idx] = end_x
        bout_array["end_y_px"][idx] = end_y

        if has_calibration and calibration:
            bout_array["distance_mm"][idx] = float(bout.distance_mm or np.nan)
            bout_array["distance_bl"][idx] = float(bout.distance_bl or np.nan)
            bout_array["mean_speed_mm_s"][idx] = float(bout.mean_speed_mm_s or np.nan)
            bout_array["mean_speed_bl_s"][idx] = float(bout.mean_speed_bl_s or np.nan)
            bout_array["peak_speed_mm_s"][idx] = float(bout.peak_speed_mm_s or np.nan)
            bout_array["peak_speed_bl_s"][idx] = float(bout.peak_speed_bl_s or np.nan)
            px_to_mm = calibration.pixel_to_mm
            bout_array["start_x_mm"][idx] = start_x * px_to_mm if not np.isnan(start_x) else np.nan
            bout_array["start_y_mm"][idx] = start_y * px_to_mm if not np.isnan(start_y) else np.nan
            bout_array["end_x_mm"][idx] = end_x * px_to_mm if not np.isnan(end_x) else np.nan
            bout_array["end_y_mm"][idx] = end_y * px_to_mm if not np.isnan(end_y) else np.nan

        start_idx = idx * 2
        end_idx = start_idx + 1
        point_array["bout_id"][start_idx] = int(bout.bout_id)
        point_array["point_type"][start_idx] = b"start"
        point_array["frame"][start_idx] = int(bout.start_frame)
        point_array["time_s"][start_idx] = float(bout.start_time_s)
        point_array["x_px"][start_idx] = start_x
        point_array["y_px"][start_idx] = start_y

        point_array["bout_id"][end_idx] = int(bout.bout_id)
        point_array["point_type"][end_idx] = b"end"
        point_array["frame"][end_idx] = int(bout.end_frame)
        point_array["time_s"][end_idx] = float(bout.end_time_s)
        point_array["x_px"][end_idx] = end_x
        point_array["y_px"][end_idx] = end_y

        if has_calibration and calibration:
            px_to_mm = calibration.pixel_to_mm
            point_array["x_mm"][start_idx] = start_x * px_to_mm if not np.isnan(start_x) else np.nan
            point_array["y_mm"][start_idx] = start_y * px_to_mm if not np.isnan(start_y) else np.nan
            point_array["x_mm"][end_idx] = end_x * px_to_mm if not np.isnan(end_x) else np.nan
            point_array["y_mm"][end_idx] = end_y * px_to_mm if not np.isnan(end_y) else np.nan

    return bout_array, point_array


def _compute_inter_bout_intervals(
    bouts: Iterable[BoutWithUnits],
    fps: float,
) -> Tuple[np.ndarray, Dict[str, float], np.ndarray]:
    """Return per-gap intervals, summary stats, and histogram."""
    sorted_bouts = sorted(bouts, key=lambda b: b.start_time_s)
    interval_dtype: List[Tuple[str, Any]] = [
        ("prev_bout_id", "i4"),
        ("next_bout_id", "i4"),
        ("prev_end_frame", "i8"),
        ("next_start_frame", "i8"),
        ("interval_frames", "i8"),
        ("prev_end_time_s", "f8"),
        ("next_start_time_s", "f8"),
        ("interval_s", "f8"),
    ]
    histogram_dtype: List[Tuple[str, Any]] = [
        ("bin_left_edge_s", "f8"),
        ("bin_right_edge_s", "f8"),
        ("count", "i8"),
    ]

    if len(sorted_bouts) < 2:
        empty_intervals = np.zeros(0, dtype=interval_dtype)
        empty_histogram = np.zeros(0, dtype=histogram_dtype)
        metrics = {
            "inter_bout_interval_count": 0.0,
            "inter_bout_interval_mean_s": float("nan"),
            "inter_bout_interval_std_s": float("nan"),
            "inter_bout_interval_median_s": float("nan"),
            "inter_bout_interval_min_s": float("nan"),
            "inter_bout_interval_max_s": float("nan"),
        }
        return empty_intervals, metrics, empty_histogram

    n_intervals = len(sorted_bouts) - 1
    intervals = np.zeros(n_intervals, dtype=interval_dtype)
    for idx, (prev_bout, next_bout) in enumerate(zip(sorted_bouts[:-1], sorted_bouts[1:])):
        raw_gap_frames = int(next_bout.start_frame) - int(prev_bout.end_frame)
        gap_frames = max(0, raw_gap_frames)
        raw_gap_seconds = float(next_bout.start_time_s) - float(prev_bout.end_time_s)
        gap_seconds = max(0.0, raw_gap_seconds)
        if not np.isfinite(gap_seconds) and fps > 0:
            gap_seconds = gap_frames / fps if fps > 0 else float("nan")
        elif fps > 0 and gap_frames and not np.isclose(gap_seconds, gap_frames / fps):
            # Prefer frame-derived duration when timestamps/frames disagree.
            gap_seconds = gap_frames / fps

        intervals["prev_bout_id"][idx] = int(prev_bout.bout_id)
        intervals["next_bout_id"][idx] = int(next_bout.bout_id)
        intervals["prev_end_frame"][idx] = int(prev_bout.end_frame)
        intervals["next_start_frame"][idx] = int(next_bout.start_frame)
        intervals["interval_frames"][idx] = int(gap_frames)
        intervals["prev_end_time_s"][idx] = float(prev_bout.end_time_s)
        intervals["next_start_time_s"][idx] = float(next_bout.start_time_s)
        intervals["interval_s"][idx] = float(gap_seconds)

    interval_values = intervals["interval_s"]
    metrics = {
        "inter_bout_interval_count": float(n_intervals),
        "inter_bout_interval_mean_s": float(interval_values.mean()),
        "inter_bout_interval_std_s": float(interval_values.std()),
        "inter_bout_interval_median_s": float(np.median(interval_values)),
        "inter_bout_interval_min_s": float(interval_values.min()),
        "inter_bout_interval_max_s": float(interval_values.max()),
    }

    hist_counts, hist_edges = np.histogram(interval_values, bins="auto")
    histogram = np.zeros(hist_counts.size, dtype=histogram_dtype)
    if hist_counts.size:
        histogram["bin_left_edge_s"] = hist_edges[:-1]
        histogram["bin_right_edge_s"] = hist_edges[1:]
        histogram["count"] = hist_counts.astype("i8")

    return intervals, metrics, histogram


def _print_summary(global_metrics: Dict[str, Any], calibration: Optional[CalibrationData]) -> None:
    """Print an analysis summary including both pixel and real-world units."""
    def _valid(value: Any) -> bool:
        return value is not None and not (isinstance(value, float) and np.isnan(value))

    print("\n" + "=" * 60)
    print("BOUT ANALYSIS SUMMARY")
    print("=" * 60)

    n_bouts = int(global_metrics.get("n_bouts", 0) or 0)
    print(f"Total bouts detected: {n_bouts}")
    if n_bouts == 0:
        return

    rate = global_metrics.get("bout_rate_per_min")
    if _valid(rate):
        print(f"Bout rate: {rate:.1f} bouts/min")

    total_active = global_metrics.get("total_active_time_s")
    percent_active = global_metrics.get("percent_active")
    if _valid(total_active) and _valid(percent_active):
        print(f"Active time: {total_active:.1f}s ({percent_active:.1f}%)")

    duration_mean = global_metrics.get("duration_mean_s")
    duration_std = global_metrics.get("duration_std_s")
    if _valid(duration_mean) and _valid(duration_std):
        print(f"\nBout duration: {duration_mean:.3f} ± {duration_std:.3f} s")

    dist_mean_px = global_metrics.get("distance_mean_px")
    dist_std_px = global_metrics.get("distance_std_px")
    dist_mean_mm = global_metrics.get("distance_mean_mm")
    dist_std_mm = global_metrics.get("distance_std_mm")
    if _valid(dist_mean_px) and _valid(dist_std_px):
        line = f"Distance per bout: {dist_mean_px:.1f} ± {dist_std_px:.1f} px"
        if calibration and _valid(dist_mean_mm) and _valid(dist_std_mm):
            line += f" ({dist_mean_mm:.2f} ± {dist_std_mm:.2f} mm)"
        print(line)

    speed_mean_px = global_metrics.get("mean_speed_mean_px_s")
    speed_std_px = global_metrics.get("mean_speed_std_px_s")
    speed_mean_mm = global_metrics.get("mean_speed_mean_mm_s")
    speed_std_mm = global_metrics.get("mean_speed_std_mm_s")
    if _valid(speed_mean_px) and _valid(speed_std_px):
        line = f"Mean speed: {speed_mean_px:.1f} ± {speed_std_px:.1f} px/s"
        if calibration and _valid(speed_mean_mm) and _valid(speed_std_mm):
            line += f" ({speed_mean_mm:.2f} ± {speed_std_mm:.2f} mm/s)"
        print(line)

    peak_speed_px = global_metrics.get("peak_speed_max_px_s")
    peak_speed_mm = global_metrics.get("peak_speed_max_mm_s")
    if _valid(peak_speed_px):
        line = f"Peak speed: {peak_speed_px:.1f} px/s"
        if calibration and _valid(peak_speed_mm):
            line += f" ({peak_speed_mm:.2f} mm/s)"
        print(line)

    total_distance_px = global_metrics.get("total_distance_px")
    total_distance_mm = global_metrics.get("total_distance_mm")
    if _valid(total_distance_px):
        line = f"Total distance: {total_distance_px:.1f} px"
        if calibration and _valid(total_distance_mm):
            line += f" ({total_distance_mm:.1f} mm)"
        print(line)

    coverage = global_metrics.get("coverage_pct")
    if _valid(coverage):
        print(f"Coverage: {coverage:.1f}%")

    ibi_mean = global_metrics.get("inter_bout_interval_mean_s")
    ibi_std = global_metrics.get("inter_bout_interval_std_s")
    if _valid(ibi_mean) and _valid(ibi_std):
        print(f"Inter-bout interval: {ibi_mean:.3f} ± {ibi_std:.3f} s")


def _load_table_from_group(obj: Any) -> pd.DataFrame:
    """Reconstruct a structured dataset stored column-wise in Zarr."""
    if hasattr(obj, "shape") and hasattr(obj, "dtype") and not hasattr(obj, "array_keys"):
        return pd.DataFrame(obj[:])

    field_names = obj.attrs.get("field_names")
    if not field_names:
        data: Dict[str, np.ndarray] = {}
        for name in obj.array_keys():
            data[name] = obj[name][:]
        return pd.DataFrame(data)

    table: Dict[str, np.ndarray] = {}
    for field in field_names:
        if field not in obj:
            continue
        table[field] = _normalize_column(obj[field][:])
    return pd.DataFrame(table)


def _resolve_event_type(value: str) -> int:
    """Convert event name or integer string to numeric ID."""
    if value.isdigit():
        return int(value)
    upper = value.upper()
    if upper not in EVENT_NAME_TO_ID:
        raise ValueError(f"Unknown event name '{value}'. Available: {sorted(EVENT_NAME_TO_ID)}")
    return EVENT_NAME_TO_ID[upper]


def _build_stimulus_lookup(stimulus_group: zarr.Group) -> Tuple[pd.DataFrame, Dict[int, int]]:
    """Return events DataFrame and mapping stimulus frame -> camera frame."""
    video_meta = stimulus_group.get("video_metadata")
    if video_meta is None or "frame_metadata" not in video_meta:
        raise ValueError("Stimulus run lacks video_metadata/frame_metadata.")

    frame_meta = _load_table_from_group(video_meta["frame_metadata"])
    for col in ("stimulus_frame_num", "triggering_camera_frame_id"):
        if col in frame_meta.columns:
            frame_meta[col] = pd.to_numeric(frame_meta[col], errors="coerce")
    frame_meta = frame_meta.dropna(subset=["stimulus_frame_num", "triggering_camera_frame_id"])
    frame_meta["stimulus_frame_num"] = frame_meta["stimulus_frame_num"].astype(int)
    frame_meta["triggering_camera_frame_id"] = frame_meta["triggering_camera_frame_id"].astype(int)
    stim_to_cam = dict(
        zip(frame_meta["stimulus_frame_num"].to_numpy(), frame_meta["triggering_camera_frame_id"].to_numpy())
    )

    if "events" not in stimulus_group:
        raise ValueError("Stimulus run lacks events dataset.")
    events_df = _load_table_from_group(stimulus_group["events"])
    for col in ("event_type_id", "stimulus_frame_num", "camera_frame_id", "camera_frame_num"):
        if col in events_df.columns:
            events_df[col] = pd.to_numeric(events_df[col], errors="coerce")
    return events_df, stim_to_cam


def _resolve_event_camera_frame(event: pd.Series, stim_to_cam: Dict[int, int]) -> Optional[int]:
    """Determine camera frame for an event record."""
    for col in ("camera_frame_id", "camera_frame_num"):
        if col in event and not pd.isna(event[col]):
            return int(event[col])
    stim_frame = event.get("stimulus_frame_num")
    if pd.isna(stim_frame):
        return None
    return stim_to_cam.get(int(stim_frame))


def _generate_trial_segments(
    events: pd.DataFrame,
    stim_to_cam: Dict[int, int],
    start_event_id: int,
    end_event_id: int,
) -> List[Dict[str, Any]]:
    """Pair start/end events into trial segments."""
    events_sorted = events.sort_values(by=list(events.columns), ignore_index=True)
    segments: List[Dict[str, Any]] = []
    current: Optional[Dict[str, Any]] = None
    trial_idx = 0

    for _, row in events_sorted.iterrows():
        evt_type = row.get("event_type_id")
        if pd.isna(evt_type):
            continue
        evt_type = int(evt_type)

        if evt_type == start_event_id:
            cam_frame = _resolve_event_camera_frame(row, stim_to_cam)
            if cam_frame is None:
                continue
            current = {
                "trial_id": trial_idx,
                "start_event": row.to_dict(),
                "start_frame": cam_frame,
            }
            trial_idx += 1
        elif evt_type == end_event_id and current is not None:
            cam_frame = _resolve_event_camera_frame(row, stim_to_cam)
            if cam_frame is None:
                continue
            if cam_frame <= current["start_frame"]:
                continue
            current["end_event"] = row.to_dict()
            current["end_frame"] = cam_frame
            segments.append(current)
            current = None

    return segments


def _summarize_bouts(
    bouts: Iterable[BoutWithUnits],
    fps: float,
    interval_frames: int,
    calibration: Optional[Any],
) -> Dict[str, Any]:
    """Compute bout summary statistics for a subset of bouts."""
    bouts = list(bouts)
    if interval_frames <= 0:
        interval_frames = 1
    total_time = interval_frames / fps
    if not bouts:
        return {
            "n_bouts": 0,
            "total_interval_s": total_time,
            "bout_rate_per_min": 0.0,
            "total_active_time_s": 0.0,
            "percent_active": 0.0,
        }

    durations = np.array([b.duration_s for b in bouts])
    distances_px = np.array([b.distance_px for b in bouts])
    mean_speeds_px = np.array([b.mean_speed_px_s for b in bouts])
    peak_speeds_px = np.array([b.peak_speed_px_s for b in bouts])

    summary: Dict[str, Any] = {
        "n_bouts": len(bouts),
        "total_interval_s": total_time,
        "bout_rate_per_min": float(len(bouts) / (total_time / 60.0)),
        "total_active_time_s": float(durations.sum()),
        "percent_active": float((durations.sum() / total_time) * 100.0),
        "total_distance_px": float(distances_px.sum()),
        "duration_mean_s": float(durations.mean()),
        "duration_std_s": float(durations.std()),
        "duration_median_s": float(np.median(durations)),
        "duration_min_s": float(durations.min()),
        "duration_max_s": float(durations.max()),
        "distance_mean_px": float(distances_px.mean()),
        "distance_std_px": float(distances_px.std()),
        "distance_median_px": float(np.median(distances_px)),
        "mean_speed_mean_px_s": float(mean_speeds_px.mean()),
        "mean_speed_std_px_s": float(mean_speeds_px.std()),
        "peak_speed_max_px_s": float(peak_speeds_px.max()),
    }

    if calibration:
        distances_mm = np.array([b.distance_mm for b in bouts if b.distance_mm is not None], dtype=float)
        distances_bl = np.array([b.distance_bl for b in bouts if b.distance_bl is not None], dtype=float)
        mean_speeds_mm = np.array([b.mean_speed_mm_s for b in bouts if b.mean_speed_mm_s is not None], dtype=float)
        mean_speeds_bl = np.array([b.mean_speed_bl_s for b in bouts if b.mean_speed_bl_s is not None], dtype=float)
        peak_speeds_mm = np.array([b.peak_speed_mm_s for b in bouts if b.peak_speed_mm_s is not None], dtype=float)
        peak_speeds_bl = np.array([b.peak_speed_bl_s for b in bouts if b.peak_speed_bl_s is not None], dtype=float)

        if distances_mm.size:
            summary.update(
                {
                    "total_distance_mm": float(distances_mm.sum()),
                    "distance_mean_mm": float(distances_mm.mean()),
                    "distance_std_mm": float(distances_mm.std()),
                    "distance_mean_bl": float(distances_bl.mean()) if distances_bl.size else None,
                    "distance_std_bl": float(distances_bl.std()) if distances_bl.size else None,
                }
            )
        if mean_speeds_mm.size:
            summary.update(
                {
                    "mean_speed_mean_mm_s": float(mean_speeds_mm.mean()),
                    "mean_speed_std_mm_s": float(mean_speeds_mm.std()),
                    "mean_speed_mean_bl_s": float(mean_speeds_bl.mean()) if mean_speeds_bl.size else None,
                    "mean_speed_std_bl_s": float(mean_speeds_bl.std()) if mean_speeds_bl.size else None,
                }
            )
        if peak_speeds_mm.size:
            summary.update(
                {
                    "peak_speed_max_mm_s": float(peak_speeds_mm.max()),
                    "peak_speed_max_bl_s": float(peak_speeds_bl.max()) if peak_speeds_bl.size else None,
                }
            )

    if len(bouts) > 1:
        ibis = []
        sorted_bouts = sorted(bouts, key=lambda b: b.start_time_s)
        for prev, curr in zip(sorted_bouts[:-1], sorted_bouts[1:]):
            ibis.append(curr.start_time_s - prev.end_time_s)
        ibis_array = np.array(ibis, dtype=float)
        if ibis_array.size:
            summary.update(
                {
                    "ibi_mean_s": float(ibis_array.mean()),
                    "ibi_std_s": float(ibis_array.std()),
                    "ibi_median_s": float(np.median(ibis_array)),
                    "ibi_min_s": float(ibis_array.min()),
                    "ibi_max_s": float(ibis_array.max()),
                }
            )

    return summary


def _filter_bouts_for_interval(
    bouts: List[BoutWithUnits],
    start_frame: int,
    end_frame: int,
) -> List[BoutWithUnits]:
    """Return bouts that overlap the [start_frame, end_frame) interval."""
    selected: List[BoutWithUnits] = []
    for bout in bouts:
        if bout.end_frame <= start_frame:
            continue
        if bout.start_frame >= end_frame:
            continue
        selected.append(bout)
    return selected


def _extract_calibration_from_stimulus(
    stimulus_group: zarr.Group,
    fallback_fish_length: float = 4.0,
) -> Optional[CalibrationData]:
    """Build CalibrationData from stimulus run attributes if available."""
    arena_json = stimulus_group.attrs.get("arena_config_json")
    if not arena_json:
        return None
    if isinstance(arena_json, bytes):
        arena_json = arena_json.decode("utf-8")
    try:
        arena_cfg = json.loads(arena_json)
    except (TypeError, json.JSONDecodeError):
        return None

    camera_calibrations = arena_cfg.get("camera_calibrations") or []
    if not camera_calibrations:
        return None
    camera = camera_calibrations[0]

    pixels_per_mm = None
    if camera.get("pixels_per_mm_camera"):
        pixels_per_mm = float(camera["pixels_per_mm_camera"])
    elif camera.get("pixels_per_mm_projector"):
        pixels_per_mm = float(camera["pixels_per_mm_projector"])

    if not pixels_per_mm or pixels_per_mm <= 0:
        return None

    pixel_to_mm = 1.0 / pixels_per_mm
    fish_length_mm = arena_cfg.get("fish_length_mm", fallback_fish_length)
    arena_diameter_mm = arena_cfg.get("experimental_area_width_mm")

    calibration = CalibrationData(
        pixel_to_mm=pixel_to_mm,
        pixels_per_mm=pixels_per_mm,
        arena_diameter_mm=arena_diameter_mm,
        fish_length_mm=fish_length_mm,
        camera_info=camera,
    )
    return calibration


def _apply_calibration_override(analyzer: EnhancedBoutAnalyzer, calibration: CalibrationData, verbose: bool = False) -> None:
    """Override analyzer calibration and update derived bout fields."""
    analyzer.calibration = calibration

    params = analyzer.init_params
    prev_verbose = getattr(analyzer, "verbose", False)
    analyzer.verbose = verbose
    analyzer._set_speed_threshold(
        params.get("speed_threshold_px_s"),
        params.get("speed_threshold_mm_s"),
        params.get("speed_threshold_bl_s"),
    )
    analyzer.verbose = prev_verbose

    px_to_mm = calibration.pixel_to_mm
    fish_length_mm = calibration.fish_length_mm
    for bout in analyzer.bouts:
        bout.distance_mm = (
            bout.distance_px * px_to_mm if bout.distance_px is not None else None
        )
        bout.distance_bl = (
            bout.distance_mm / fish_length_mm if bout.distance_mm is not None else None
        )
        bout.mean_speed_mm_s = (
            bout.mean_speed_px_s * px_to_mm if bout.mean_speed_px_s is not None else None
        )
        bout.mean_speed_bl_s = (
            bout.mean_speed_mm_s / fish_length_mm if bout.mean_speed_mm_s is not None else None
        )
        bout.peak_speed_mm_s = (
            bout.peak_speed_px_s * px_to_mm if bout.peak_speed_px_s is not None else None
        )
        bout.peak_speed_bl_s = (
            bout.peak_speed_mm_s / fish_length_mm if bout.peak_speed_mm_s is not None else None
        )

    if verbose:
        print(
            f"Calibration override applied: 1 pixel = {calibration.pixel_to_mm:.4f} mm "
            f"({calibration.pixels_per_mm:.2f} px/mm)"
        )


def run_statistics(args: argparse.Namespace) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    """Main execution routine returning report and structured arrays."""
    analyzer = EnhancedBoutAnalyzer(
        args.zarr_path,
        source=args.source,
        speed_threshold_px_s=args.threshold_px,
        speed_threshold_mm_s=args.threshold_mm,
        speed_threshold_bl_s=args.threshold_bl,
        min_bout_duration_s=args.min_bout_duration,
        min_gap_duration_s=args.min_gap_duration,
        verbose=False,
    )

    root = analyzer.root
    fps = analyzer.fps
    stimulus_run = args.stimulus_run
    trials: List[Dict[str, Any]] = []
    trial_params: Dict[str, Any] = {}

    if stimulus_run or "analysis" in root:
        analysis_group = root.get("analysis")
        if analysis_group is None or "stimulus_runs" not in analysis_group:
            if args.verbose:
                print("No stimulus_runs found; skipping per-trial statistics.")
        else:
            stim_parent = analysis_group["stimulus_runs"]
            if stimulus_run is None:
                stimulus_run = stim_parent.attrs.get("latest")
            if stimulus_run and stimulus_run in stim_parent:
                stimulus_group = stim_parent[stimulus_run]
                fallback_fish_length = analyzer.calibration.fish_length_mm if analyzer.calibration else 4.0
                calibration_override = _extract_calibration_from_stimulus(
                    stimulus_group,
                    fallback_fish_length=fallback_fish_length,
                )
                if calibration_override:
                    _apply_calibration_override(analyzer, calibration_override, verbose=not args.quiet)

                calibration = analyzer.calibration
                events_df, stim_to_cam = _build_stimulus_lookup(stimulus_group)
                start_id = _resolve_event_type(args.trial_start_event)
                end_id = _resolve_event_type(args.trial_end_event)
                segments = _generate_trial_segments(events_df, stim_to_cam, start_id, end_id)

                for segment in segments:
                    start_frame = int(segment["start_frame"])
                    end_frame = int(segment["end_frame"])
                    interval_frames = max(end_frame - start_frame, 1)
                    segment_bouts = _filter_bouts_for_interval(analyzer.bouts, start_frame, end_frame)
                    metrics = _summarize_bouts(segment_bouts, fps, interval_frames, calibration)
                    trials.append(
                        {
                            "trial_id": segment["trial_id"],
                            "start_frame": start_frame,
                            "end_frame": end_frame,
                            "start_time_s": start_frame / fps,
                            "end_time_s": end_frame / fps,
                            "metrics": metrics,
                            "start_event": segment.get("start_event"),
                            "end_event": segment.get("end_event"),
                        }
                    )

                trial_params = {
                    "stimulus_run": stimulus_run,
                    "trial_count": len(trials),
                    "trial_start_event": start_id,
                    "trial_end_event": end_id,
                }
            elif args.verbose:
                print("Stimulus run not found; skipping per-trial statistics.")

    calibration = analyzer.calibration
    global_metrics = analyzer.calculate_summary_statistics()
    global_metrics["coverage_pct"] = float(
        np.sum(~np.isnan(analyzer.positions_x)) / analyzer.total_frames * 100.0
    )
    inter_bout_intervals, inter_bout_summary, inter_bout_histogram = _compute_inter_bout_intervals(
        analyzer.bouts,
        fps,
    )
    global_metrics.update(inter_bout_summary)
    global_array = _make_global_array(global_metrics)
    bout_array, point_array = _make_bout_arrays(analyzer)
    trial_array = _make_trial_array(trials)

    provenance = {
        "script": "fisheye.analysis.swim_bout_statistics",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git": get_git_info(),
        "arguments": {k: _to_serializable(v) for k, v in vars(args).items() if k not in {"output"}},
        "zarr_path": str(args.zarr_path),
        "detect_source": analyzer.source_info,
        "calibration": asdict(calibration) if calibration else None,
    }
    provenance.update(trial_params)

    report = {
        "global_metrics": global_metrics,
        "trials": trials,
        "provenance": provenance,
    }
    report["inter_bout_interval_histogram"] = _structured_to_dicts(inter_bout_histogram)
    if args.include_bouts:
        report["bouts"] = _structured_to_dicts(bout_array)
        report["bout_points"] = _structured_to_dicts(point_array)
        report["inter_bout_intervals"] = _structured_to_dicts(inter_bout_intervals)

    storage = {
        "global_metrics": global_array,
        "trials": trial_array,
        "bouts": bout_array,
        "bout_points": point_array,
        "inter_bout_intervals": inter_bout_intervals,
        "inter_bout_interval_histogram": inter_bout_histogram,
    }

    if not args.quiet:
        _print_summary(global_metrics, calibration)

    return report, storage


def _save_report_to_zarr(
    zarr_path: Path,
    storage: Dict[str, np.ndarray],
    report: Dict[str, Any],
    run_name: Optional[str],
    verbose: bool,
) -> str:
    """Persist swim bout analysis results under analysis/swim_bout_runs."""
    root = zarr.open(str(zarr_path), mode="a")
    analysis = root.require_group("analysis")
    runs_parent = analysis.require_group("swim_bout_runs")

    if run_name:
        target_name = run_name
    else:
        target_name = datetime.now(timezone.utc).strftime("swim_bout_%Y%m%d_%H%M%S")

    if target_name in runs_parent:
        raise ValueError(f"Run '{target_name}' already exists in analysis/swim_bout_runs")

    run_group = runs_parent.create_group(target_name)
    runs_parent.attrs["latest"] = target_name

    for dataset_name, array in storage.items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UnstableSpecificationWarning)
            run_group.create_array(name=dataset_name, data=array, overwrite=True)

    run_group.attrs["created_at_utc"] = report["provenance"].get("timestamp_utc")
    run_group.attrs["provenance"] = json.loads(json.dumps(_to_serializable(report["provenance"])))
    run_group.attrs["dataset_fields"] = {
        name: list(array.dtype.names) if array.dtype.names else []
        for name, array in storage.items()
    }
    run_group.attrs["report_version"] = "1.0"

    if verbose:
        print(f"Saved swim bout analysis to analysis/swim_bout_runs/{target_name}")

    return target_name


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute swimming bout statistics for Palette archives."
    )
    parser.add_argument("zarr_path", type=Path, help="Path to Palette Zarr archive.")
    parser.add_argument(
        "--source",
        type=str,
        default="latest",
        help="Analyzer source selector (default: latest).",
    )
    parser.add_argument(
        "--stimulus-run",
        type=str,
        help="Stimulus import run under analysis/stimulus_runs (default: latest).",
    )
    parser.add_argument(
        "--trial-start-event",
        type=str,
        default="STEP_START",
        help="Event type (name or ID) that marks the start of a trial (default: STEP_START).",
    )
    parser.add_argument(
        "--trial-end-event",
        type=str,
        default="STEP_END",
        help="Event type (name or ID) that marks the end of a trial (default: STEP_END).",
    )
    parser.add_argument(
        "--threshold-px",
        type=float,
        help="Speed threshold for bout detection in px/s.",
    )
    parser.add_argument(
        "--threshold-mm",
        type=float,
        help="Speed threshold for bout detection in mm/s.",
    )
    parser.add_argument(
        "--threshold-bl",
        type=float,
        help="Speed threshold for bout detection in body lengths/s.",
    )
    parser.add_argument(
        "--min-bout-duration",
        type=float,
        default=0.1,
        help="Minimum bout duration in seconds (default: 0.1).",
    )
    parser.add_argument(
        "--min-gap-duration",
        type=float,
        default=0.1,
        help="Minimum gap between bouts in seconds (default: 0.1).",
    )
    parser.add_argument(
        "--include-bouts",
        action="store_true",
        help="Include per-bout records and bout points in the JSON output.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Skip writing results to analysis/swim_bout_runs in the Zarr archive.",
    )
    parser.add_argument(
        "--zarr-run-name",
        type=str,
        help="Optional run name when saving to Zarr (default: timestamp-based).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON path to write results. If omitted, prints to stdout.",
    )
    parser.add_argument(
        "--no-stdout",
        action="store_true",
        help="Do not print the JSON report to stdout (use with --output to write to file only).",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON output with indentation.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose analyzer output.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print additional diagnostic information.",
    )
    return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(None if argv is None else list(argv))

    report, storage = run_statistics(args)

    run_id: Optional[str] = None
    if not args.no_save:
        run_id = _save_report_to_zarr(
            args.zarr_path,
            storage,
            report,
            run_name=args.zarr_run_name,
            verbose=args.verbose,
        )
        report["provenance"]["zarr_run"] = run_id
    else:
        report["provenance"]["zarr_run"] = None

    json_kwargs = {"indent": 2, "sort_keys": True} if args.pretty else {"separators": (",", ":"), "sort_keys": True}
    output_text = json.dumps(_to_serializable(report), **json_kwargs)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output_text)
        if args.verbose:
            print(f"Wrote bout statistics to {args.output}")

    if not args.no_stdout:
        print(output_text)

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
