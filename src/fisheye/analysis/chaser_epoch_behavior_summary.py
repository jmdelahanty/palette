"""Persist protocol-neutral per-epoch behavior and per-chaser summaries."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import math
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import zarr

from fisheye.analysis.chaser_distance_runs import (
    ChaserDistanceWindow,
)
from fisheye.analysis.chaser_distance_io import (
    ChaserDistanceReadSnapshot,
    load_chaser_distance_run,
    reject_unsealed_chaser_derived_publication,
)
from fisheye.shared.zarr.columnar import write_columnar_dataset
from fisheye.analysis.epoch_segments import (
    HistogramMetricSpec,
    histogram_table,
    segments_from_window_objects,
)
from fisheye.analysis.swim_bout_io import load_default_swim_bout_tables
from fisheye.analysis.track_kinematics_io import (
    TRACK_KINEMATICS_SPEED_LEVELS,
    load_track_kinematics_track,
)
from fisheye.shared.arena_geometry import resolve_arena_geometry as _resolve_shared_arena_geometry
from fisheye.shared.json_safety import decode_null_terminated_text, json_attr_safe
from fisheye.shared.system_metadata import get_git_info

SCHEMA_ID = "palette.chaser.epoch_behavior_summary.v1"
SCHEMA_VERSION = 1
METHOD = "chaser_epoch_behavior_summary"
METHOD_VERSION = "1"
COMPONENT_PARENT_NAME = "epoch_behavior_summary"
DEFAULT_COMPONENT_NAME = "kinematics_bouts_v1"
REQUIRED_TRACK_SCOPE = "offline"
DEFAULT_CENTER_DISTANCE_BIN_WIDTH_MM = 2.5
DEFAULT_WALL_BAND_MM = 5.0
DEFAULT_BOUT_DURATION_BIN_WIDTH_S = 0.02
DEFAULT_BOUT_DISTANCE_BIN_WIDTH_MM = 0.25
DEFAULT_BOUT_HEADING_BIN_WIDTH_DEG = 10.0
DEFAULT_IBI_BIN_WIDTH_S = 0.1


@dataclass(frozen=True)
class ArenaGeometry:
    status: str
    source: Optional[str]
    shape: str
    width_px: float
    height_px: float
    center_x_px: Optional[float]
    center_y_px: Optional[float]
    radius_px: Optional[float]


@dataclass(frozen=True)
class ChaserEpochBehaviorSummaryResult:
    zarr_path: str
    recording_id: str
    component_name: str
    chaser_distance_run_name: str
    chaser_distance_run_path: str
    source_track_kinematics_run: Optional[str]
    source_track_kinematics_scope: Optional[str]
    source_track_kinematics_track_id: Optional[int]
    source_track_kinematics_track_path: Optional[str]
    source_speed_level: Optional[str]
    source_speed_level_selection: str
    source_swim_bout_run: Optional[str]
    source_swim_bout_path: Optional[str]
    source_swim_bout_level_path: Optional[str]
    source_swim_bout_signal_level: Optional[str]
    fps: float
    windows: tuple[ChaserDistanceWindow, ...]
    per_epoch_fish: np.ndarray
    per_epoch_chaser: np.ndarray
    per_epoch_bouts: np.ndarray
    per_epoch_bout_histograms: np.ndarray
    per_epoch_inter_bout_interval_histograms: np.ndarray
    center_distance_histogram: np.ndarray
    arena_geometry: ArenaGeometry
    center_distance_bin_width_mm: float
    wall_band_mm: float
    warnings: tuple[str, ...]


def _open_root(zarr_path: Path, *, mode: str) -> zarr.Group:
    return zarr.open_group(str(zarr_path), mode=mode, use_consolidated=False)


def _attrs_dict(group: zarr.Group) -> dict[str, Any]:
    try:
        return dict(group.attrs.asdict())
    except Exception:
        return dict(group.attrs)


def _resolve_chaser_distance_run(
    root: zarr.Group,
    run_name: str | None,
) -> tuple[ChaserDistanceReadSnapshot, str, str]:
    snapshot = load_chaser_distance_run(
        root,
        run_name=str(run_name or "latest").strip() or "latest",
    )
    return snapshot, snapshot.run_name, snapshot.run_path


def _decode_text_column(data: np.ndarray) -> list[str]:
    values = np.asarray(data)
    if values.ndim == 2 and values.dtype.kind in ("u", "i"):
        return [decode_null_terminated_text(row).strip() for row in values]
    return [decode_null_terminated_text(value).strip() for value in values.reshape(-1)]


def _load_windows(run_group: zarr.Group, *, fps: float) -> tuple[ChaserDistanceWindow, ...]:
    if "epoch_summary" not in run_group:
        return ()
    summary = run_group["epoch_summary"]
    if "start_frame" not in summary or "end_frame" not in summary:
        return ()
    starts = np.asarray(summary["start_frame"][:], dtype=np.int64).reshape(-1)
    ends = np.asarray(summary["end_frame"][:], dtype=np.int64).reshape(-1)
    n = min(starts.shape[0], ends.shape[0])
    if "window_id" in summary:
        ids = np.asarray(summary["window_id"][:], dtype=np.int32).reshape(-1)
    else:
        ids = np.arange(n, dtype=np.int32)
    if "label_bytes" in summary:
        labels = _decode_text_column(np.asarray(summary["label_bytes"][:]))
    else:
        labels = [f"window_{idx}" for idx in range(n)]
    out: list[ChaserDistanceWindow] = []
    for idx in range(n):
        start = int(starts[idx])
        end = int(ends[idx])
        start_s = start / float(fps) if fps > 0 else float(idx)
        end_s = (end + 1) / float(fps) if fps > 0 else float(idx + 1)
        out.append(
            ChaserDistanceWindow(
                window_id=int(ids[idx]) if idx < ids.shape[0] else int(idx),
                label=str(labels[idx]) if idx < len(labels) else f"window_{idx}",
                start_frame=start,
                end_frame=end,
                start_time_s=float(start_s),
                end_time_s=float(end_s),
                duration_s=max(0.0, float(end_s - start_s)),
            )
        )
    return tuple(out)


def _structured_field(records: np.ndarray, *names: str) -> Optional[np.ndarray]:
    if records.size == 0 or records.dtype.names is None:
        return None
    for name in names:
        if name in records.dtype.names:
            return np.asarray(records[name])
    return None


def _first_nonnegative_frame(records: np.ndarray, *field_names: str) -> Optional[np.ndarray]:
    if records.size == 0:
        return None
    out: Optional[np.ndarray] = None
    for name in field_names:
        values = _structured_field(records, name)
        if values is None:
            continue
        current = np.asarray(values, dtype=np.int64)
        if out is None:
            out = np.full(current.shape, -1, dtype=np.int64)
        mask = (out < 0) & (current >= 0)
        out[mask] = current[mask]
    if out is None or not np.any(out >= 0):
        return None
    return out


def _finite_summary(values: np.ndarray) -> tuple[int, float, float, float, float, float]:
    data = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return 0, np.nan, np.nan, np.nan, np.nan, np.nan
    return (
        int(finite.size),
        float(np.nanmean(finite)),
        float(np.nanmedian(finite)),
        float(np.nanpercentile(finite, 5.0)),
        float(np.nanpercentile(finite, 95.0)),
        float(np.nanmax(finite)),
    )


def _window_time_mask(values_s: Optional[np.ndarray], *, start_s: float, end_s: float) -> np.ndarray:
    if values_s is None:
        return np.zeros(0, dtype=bool)
    values = np.asarray(values_s, dtype=np.float64)
    return np.isfinite(values) & (values >= float(start_s)) & (values <= float(end_s))


def _speed_level_key(value: object | None) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    level = text.replace("speed_", "", 1)
    if level not in TRACK_KINEMATICS_SPEED_LEVELS:
        supported = ", ".join(TRACK_KINEMATICS_SPEED_LEVELS)
        raise ValueError(
            f"Unsupported physical track speed level {text!r}; expected one of: "
            f"{supported}. Detector-only signals such as 'exponential' are selected "
            "by the persisted swim-bout run, not by --speed-level."
        )
    return level


def _resolve_speed_sources(
    root: zarr.Group,
    *,
    swim_bout_run: str | None,
    track_kinematics_run: str | None,
    track_kinematics_scope: str,
    track_id: int | None,
    speed_level: str | None,
) -> tuple[
    Optional[Any],
    Optional[Any],
    Optional[str],
    Optional[str],
    Optional[int],
    Optional[str],
    str,
    list[str],
]:
    warnings: list[str] = []
    swim_tables = None
    try:
        swim_tables = load_default_swim_bout_tables(root, run_name=swim_bout_run or "latest")
    except Exception as exc:
        warnings.append(f"swim_bout_unavailable: {exc}")

    track_selector = str(track_kinematics_run or "").strip()
    source_track_run = (
        None if track_selector in {"", "latest"} else track_selector
    )
    source_track_id = track_id
    source_speed_level = _speed_level_key(speed_level)
    speed_level_selection = (
        "explicit_physical_track_speed_level"
        if source_speed_level is not None
        else "unresolved"
    )
    if swim_tables is not None:
        source_track_run = (
            source_track_run
            or swim_tables.candidate.source_track_kinematics_run
            or str(swim_tables.run_attrs.get("source_track_kinematics_run") or "").strip()
            or None
        )
        source_track_id = (
            source_track_id
            if source_track_id is not None
            else swim_tables.candidate.track_id
            if swim_tables.candidate.track_id is not None
            else _safe_int(swim_tables.run_attrs.get("track_id"))
        )
        if source_speed_level is None and swim_tables.signal.source_level:
            source_speed_level = _speed_level_key(swim_tables.signal.source_level)
            speed_level_selection = "persisted_swim_bout_signal_physical_source_level"
        if source_speed_level is None:
            source_speed_level = _speed_level_key(swim_tables.signal.speed_level)
            speed_level_selection = "persisted_swim_bout_signal_level"
    if source_track_run is None and track_selector == "latest":
        source_track_run = "latest"
    if source_track_id is None:
        source_track_id = 0
    if source_speed_level is None:
        source_speed_level = "filtered"
        speed_level_selection = "default_filtered_physical_track_speed_level"

    track = None
    if source_track_run:
        try:
            track = load_track_kinematics_track(
                root,
                run_name=source_track_run,
                scope=track_kinematics_scope,
                track_id=int(source_track_id),
                required_speed_levels=(source_speed_level,),
            )
            source_track_run = track.run_name
        except Exception as exc:
            warnings.append(f"track_kinematics_unavailable: {exc}")
    else:
        warnings.append("track_kinematics_unavailable: no source track run resolved")
    return (
        swim_tables,
        track,
        source_track_run,
        track_kinematics_scope if track is not None else None,
        int(source_track_id) if source_track_id is not None else None,
        source_speed_level,
        speed_level_selection,
        warnings,
    )


def _safe_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except Exception:
        return None


def _optional_float(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def _get_group_by_path(root: zarr.Group, path: str | None) -> Optional[zarr.Group]:
    normalized = "/".join(part for part in str(path or "").strip("/").split("/") if part)
    if not normalized:
        return root
    current: Any = root
    for part in normalized.split("/"):
        if part not in current:
            return None
        current = current[part]
    return current if isinstance(current, zarr.Group) else None


def _resolve_arena_geometry(root: zarr.Group, run_group: zarr.Group) -> ArenaGeometry:
    """Prefer the fitted dish mask over the projector's nominal experimental_area circle.

    See fisheye.shared.arena_geometry -- these are different circles, and the nominal one is
    ~3 mm off-centre and ~2.4 mm small, which places a wall-hugging fish "outside the arena".
    """

    pixels_per_mm = _optional_float(run_group.attrs.get("pixels_per_mm_projector")) or 1.0
    try:
        resolved, _notes = _resolve_shared_arena_geometry(root, run_group, pixels_per_mm=float(pixels_per_mm))
    except ValueError:
        return ArenaGeometry(
            status="missing",
            source=None,
            shape="unknown",
            width_px=float("nan"),
            height_px=float("nan"),
            center_x_px=None,
            center_y_px=None,
            radius_px=None,
        )
    return ArenaGeometry(
        status=resolved.status,
        source=resolved.source,
        shape=resolved.shape,
        width_px=float(resolved.width_px),
        height_px=float(resolved.height_px),
        center_x_px=resolved.center_x_px,
        center_y_px=resolved.center_y_px,
        radius_px=resolved.radius_px,
    )


def _wrap_heading_delta_degrees(values: np.ndarray | float) -> np.ndarray:
    data = np.asarray(values, dtype=np.float64)
    return ((data + 180.0) % 360.0) - 180.0


def _center_distance_mm(
    run_group: zarr.Group,
    geometry: ArenaGeometry,
) -> tuple[np.ndarray, np.ndarray, float]:
    pixels_per_mm = _optional_float(run_group.attrs.get("pixels_per_mm_projector")) or float("nan")
    if (
        "positions" not in run_group
        or "fish_centroid_arena_xy" not in run_group["positions"]
        or not math.isfinite(pixels_per_mm)
        or pixels_per_mm <= 0
        or geometry.center_x_px is None
        or geometry.center_y_px is None
    ):
        total_frames = int(run_group.attrs.get("total_frames") or 0)
        return np.full(total_frames, np.nan, dtype=np.float64), np.zeros(total_frames, dtype=bool), float("nan")
    xy = np.asarray(run_group["positions/fish_centroid_arena_xy"][:], dtype=np.float64)
    distance_px = np.sqrt((xy[:, 0] - float(geometry.center_x_px)) ** 2 + (xy[:, 1] - float(geometry.center_y_px)) ** 2)
    distance_mm = distance_px / pixels_per_mm
    if geometry.radius_px is not None and math.isfinite(float(geometry.radius_px)):
        radius_mm = float(geometry.radius_px) / pixels_per_mm
        in_bounds = distance_px <= float(geometry.radius_px)
    else:
        radius_mm = float("nan")
        in_bounds = np.isfinite(distance_mm)
    return distance_mm, in_bounds, radius_mm


def _bout_frame_bounds(
    bouts: np.ndarray,
    *,
    fps: float,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    start_frame = _first_nonnegative_frame(bouts, "start_frame", "core_start_frame", "onset_frame", "peak_frame")
    end_frame = _first_nonnegative_frame(bouts, "end_frame", "core_end_frame", "offset_frame", "peak_frame")
    if start_frame is not None and end_frame is not None:
        return start_frame, end_frame
    start_time = _structured_field(bouts, "start_time_s", "start_s", "peak_time_s")
    end_time = _structured_field(bouts, "end_time_s", "end_s", "peak_time_s")
    if start_time is None or end_time is None or not math.isfinite(float(fps)) or fps <= 0:
        return start_frame, end_frame
    return (
        np.rint(np.asarray(start_time, dtype=np.float64) * float(fps)).astype(np.int64),
        np.rint(np.asarray(end_time, dtype=np.float64) * float(fps)).astype(np.int64),
    )


def _bout_heading_values(
    *,
    bouts: np.ndarray,
    track: Optional[Any],
    fps: float,
) -> tuple[np.ndarray, np.ndarray]:
    n_bouts = int(bouts.shape[0])
    net_delta = np.full(n_bouts, np.nan, dtype=np.float64)
    heading_path = np.full(n_bouts, np.nan, dtype=np.float64)
    if n_bouts == 0 or track is None:
        return net_delta, heading_path
    heading = track.smoothed_heading_degrees if track.smoothed_heading_degrees is not None else track.heading_degrees
    if heading is None:
        return net_delta, heading_path
    frames = np.asarray(track.frame_indices, dtype=np.int64).reshape(-1)
    heading_values = np.asarray(heading, dtype=np.float64).reshape(-1)
    if frames.shape[0] != heading_values.shape[0]:
        return net_delta, heading_path
    sample_valid = (
        np.asarray(track.sample_valid, dtype=bool).reshape(-1)
        if track.sample_valid is not None and np.asarray(track.sample_valid).shape[0] == frames.shape[0]
        else np.ones(frames.shape[0], dtype=bool)
    )
    start_frame, end_frame = _bout_frame_bounds(bouts, fps=fps)
    if start_frame is None or end_frame is None:
        return net_delta, heading_path
    starts = np.asarray(start_frame, dtype=np.int64).reshape(-1)
    ends = np.asarray(end_frame, dtype=np.int64).reshape(-1)
    n = min(n_bouts, starts.shape[0], ends.shape[0])
    for idx in range(n):
        start = int(min(starts[idx], ends[idx]))
        end = int(max(starts[idx], ends[idx]))
        mask = (frames >= start) & (frames <= end) & sample_valid & np.isfinite(heading_values)
        values = heading_values[mask]
        if values.shape[0] < 2:
            continue
        net_delta[idx] = float(_wrap_heading_delta_degrees(values[-1] - values[0]))
        heading_path[idx] = float(np.nansum(np.abs(_wrap_heading_delta_degrees(np.diff(values)))))
    return net_delta, heading_path


def _fish_valid_by_frame(run_group: zarr.Group) -> np.ndarray:
    if "positions" not in run_group or "fish_valid" not in run_group["positions"]:
        total_frames = int(run_group.attrs.get("total_frames") or 0)
        return np.zeros(total_frames, dtype=bool)
    positions = run_group["positions"]
    valid = np.asarray(positions["fish_valid"][:], dtype=bool).reshape(-1)
    if "fish_centroid_arena_xy" in positions:
        xy = np.asarray(positions["fish_centroid_arena_xy"][:], dtype=np.float64)
        if xy.ndim == 2 and xy.shape[0] == valid.shape[0]:
            valid &= np.isfinite(xy).all(axis=1)
    return valid


def _make_per_epoch_fish(
    *,
    windows: Sequence[ChaserDistanceWindow],
    run_group: zarr.Group,
    swim_tables: Optional[Any],
    track: Optional[Any],
    source_speed_level: Optional[str],
    geometry: ArenaGeometry,
    wall_band_mm: float,
) -> np.ndarray:
    dtype = np.dtype(
        [
            ("window_id", np.int32),
            ("window_index", np.int32),
            ("window_label", "S96"),
            ("start_frame", np.int64),
            ("end_frame", np.int64),
            ("start_time_s", np.float64),
            ("end_time_s", np.float64),
            ("duration_s", np.float64),
            ("total_span_frames", np.int64),
            ("valid_frame_count", np.int64),
            ("missing_frame_count", np.int64),
            ("tracking_dropout_fraction", np.float64),
            ("center_distance_sample_count", np.int64),
            ("mean_distance_from_arena_center_mm", np.float64),
            ("median_distance_from_arena_center_mm", np.float64),
            ("p05_distance_from_arena_center_mm", np.float64),
            ("p95_distance_from_arena_center_mm", np.float64),
            ("max_distance_from_arena_center_mm", np.float64),
            ("arena_radius_mm", np.float64),
            ("wall_band_mm", np.float64),
            ("wall_frame_count", np.int64),
            ("wall_fraction", np.float64),
            ("wall_time_s", np.float64),
            ("speed_sample_count", np.int64),
            ("mean_speed_mm_s", np.float64),
            ("median_speed_mm_s", np.float64),
            ("p05_speed_mm_s", np.float64),
            ("p95_speed_mm_s", np.float64),
            ("max_speed_mm_s", np.float64),
            ("total_path_mm", np.float64),
            ("bout_count", np.int64),
            ("bout_rate_per_min", np.float64),
            ("median_bout_duration_s", np.float64),
            ("mean_bout_duration_s", np.float64),
            ("median_bout_path_length_mm", np.float64),
            ("mean_bout_path_length_mm", np.float64),
            ("bout_heading_sample_count", np.int64),
            ("mean_bout_net_heading_change_deg", np.float64),
            ("median_bout_net_heading_change_deg", np.float64),
            ("mean_abs_bout_net_heading_change_deg", np.float64),
            ("median_abs_bout_net_heading_change_deg", np.float64),
            ("mean_bout_heading_path_deg", np.float64),
            ("median_bout_heading_path_deg", np.float64),
            ("inter_bout_interval_count", np.int64),
            ("mean_inter_bout_interval_s", np.float64),
            ("median_inter_bout_interval_s", np.float64),
            ("p05_inter_bout_interval_s", np.float64),
            ("p95_inter_bout_interval_s", np.float64),
            ("inter_bout_interval_rate_per_min", np.float64),
        ]
    )
    out = np.zeros(len(windows), dtype=dtype)
    for name in out.dtype.names or ():
        if out.dtype[name].kind == "f":
            out[name] = np.nan

    fish_valid = _fish_valid_by_frame(run_group)
    center_distance_mm, in_arena_mask, arena_radius_mm = _center_distance_mm(run_group, geometry)
    if center_distance_mm.shape[0] != fish_valid.shape[0]:
        n = min(center_distance_mm.shape[0], fish_valid.shape[0])
        padded_center = np.full(fish_valid.shape[0], np.nan, dtype=np.float64)
        padded_in_arena = np.zeros(fish_valid.shape[0], dtype=bool)
        if n > 0:
            padded_center[:n] = center_distance_mm[:n]
            padded_in_arena[:n] = in_arena_mask[:n]
        center_distance_mm = padded_center
        in_arena_mask = padded_in_arena
    wall_threshold_mm = (
        float(arena_radius_mm) - max(0.0, float(wall_band_mm))
        if math.isfinite(float(arena_radius_mm))
        else float("nan")
    )
    speed_frames = np.zeros(0, dtype=np.int64)
    speed_values = np.asarray([], dtype=np.float64)
    path_values = np.asarray([], dtype=np.float64)
    if track is not None:
        speed_frames = np.asarray(track.frame_indices, dtype=np.int64)
        if source_speed_level and source_speed_level in track.speed_mm_by_level:
            speed_values = np.asarray(track.speed_mm_by_level[source_speed_level], dtype=np.float64)
        elif "filtered" in track.speed_mm_by_level:
            speed_values = np.asarray(track.speed_mm_by_level["filtered"], dtype=np.float64)
        if source_speed_level and source_speed_level in track.frame_path_distance_mm_by_level:
            path_values = np.asarray(track.frame_path_distance_mm_by_level[source_speed_level], dtype=np.float64)
        elif "filtered" in track.frame_path_distance_mm_by_level:
            path_values = np.asarray(track.frame_path_distance_mm_by_level["filtered"], dtype=np.float64)

    bouts = swim_tables.bouts if swim_tables is not None else np.zeros(0, dtype=[])
    intervals = swim_tables.inter_bout_intervals if swim_tables is not None else np.zeros(0, dtype=[])
    bout_event_frame = _first_nonnegative_frame(bouts, "peak_frame", "core_start_frame", "start_frame")
    bout_time_s = _structured_field(bouts, "peak_time_s", "start_time_s", "start_s")
    bout_duration_s = _structured_field(bouts, "duration_s", "observed_duration_s", "elapsed_duration_s")
    bout_path_mm = _structured_field(bouts, "path_length_mm")
    interval_s = _structured_field(intervals, "interval_s")
    interval_prev_end_frame = _structured_field(intervals, "prev_end_frame")
    interval_next_start_frame = _structured_field(intervals, "next_start_frame")
    interval_prev_end_s = _structured_field(intervals, "prev_end_time_s")
    interval_next_start_s = _structured_field(intervals, "next_start_time_s")
    interval_valid = _structured_field(intervals, "valid")
    bout_net_heading_change_deg, bout_heading_path_deg = _bout_heading_values(
        bouts=bouts,
        track=track,
        fps=float(run_group.attrs.get("fps") or 0.0),
    )

    for window_index, window in enumerate(windows):
        duration_s = max(0.0, float(window.duration_s))
        start = max(0, int(window.start_frame))
        end = min(int(window.end_frame), int(fish_valid.shape[0]) - 1) if fish_valid.size else -1
        total_span_frames = max(0, end - start + 1) if end >= start else 0
        valid_frame_count = int(np.count_nonzero(fish_valid[start : end + 1])) if total_span_frames else 0
        missing_frame_count = max(0, total_span_frames - valid_frame_count)
        center_values = (
            center_distance_mm[start : end + 1][
                fish_valid[start : end + 1]
                & in_arena_mask[start : end + 1]
                & np.isfinite(center_distance_mm[start : end + 1])
            ]
            if total_span_frames
            else np.asarray([], dtype=np.float64)
        )
        (
            center_count,
            center_mean,
            center_median,
            center_p05,
            center_p95,
            center_max,
        ) = _finite_summary(center_values)
        if total_span_frames and math.isfinite(wall_threshold_mm):
            wall_mask = (
                fish_valid[start : end + 1]
                & in_arena_mask[start : end + 1]
                & np.isfinite(center_distance_mm[start : end + 1])
                & (center_distance_mm[start : end + 1] >= wall_threshold_mm)
            )
            wall_frame_count = int(np.count_nonzero(wall_mask))
        else:
            wall_frame_count = 0
        wall_fraction = float(wall_frame_count) / float(center_count) if center_count > 0 else np.nan
        wall_time_s = float(wall_frame_count) / float(run_group.attrs.get("fps") or 1.0) if wall_frame_count else 0.0

        speed_mask = (
            (speed_frames >= int(window.start_frame))
            & (speed_frames <= int(window.end_frame))
            & (speed_frames.shape[0] == speed_values.shape[0])
        )
        speed_count, speed_mean, speed_median, speed_p05, speed_p95, speed_max = _finite_summary(
            speed_values[speed_mask] if speed_frames.shape[0] == speed_values.shape[0] else np.asarray([])
        )
        if speed_frames.shape[0] == path_values.shape[0]:
            path_mask = (speed_frames >= int(window.start_frame)) & (speed_frames <= int(window.end_frame))
            finite_path = path_values[path_mask]
            total_path_mm = float(np.nansum(finite_path[np.isfinite(finite_path)])) if finite_path.size else np.nan
        else:
            total_path_mm = np.nan

        if bout_event_frame is not None:
            event_frame = np.asarray(bout_event_frame, dtype=np.int64)
            bout_mask = (event_frame >= int(window.start_frame)) & (event_frame <= int(window.end_frame))
        else:
            bout_mask = _window_time_mask(bout_time_s, start_s=window.start_time_s, end_s=window.end_time_s)
        bout_count = int(np.sum(bout_mask))
        bout_rate = (float(bout_count) / (duration_s / 60.0)) if duration_s > 0 else np.nan
        _dur_count, bout_duration_mean, bout_duration_median, _dur_p05, _dur_p95, _dur_max = _finite_summary(
            np.asarray(bout_duration_s, dtype=np.float64)[bout_mask] if bout_duration_s is not None else np.asarray([])
        )
        _path_count, bout_path_mean, bout_path_median, _path_p05, _path_p95, _path_max = _finite_summary(
            np.asarray(bout_path_mm, dtype=np.float64)[bout_mask] if bout_path_mm is not None else np.asarray([])
        )
        heading_values = bout_net_heading_change_deg[bout_mask] if bout_mask.shape[0] == bout_net_heading_change_deg.shape[0] else np.asarray([])
        heading_abs_values = np.abs(heading_values)
        heading_path_values = bout_heading_path_deg[bout_mask] if bout_mask.shape[0] == bout_heading_path_deg.shape[0] else np.asarray([])
        (
            heading_count,
            heading_mean,
            heading_median,
            _heading_p05,
            _heading_p95,
            _heading_max,
        ) = _finite_summary(heading_values)
        (
            _heading_abs_count,
            heading_abs_mean,
            heading_abs_median,
            _heading_abs_p05,
            _heading_abs_p95,
            _heading_abs_max,
        ) = _finite_summary(heading_abs_values)
        (
            _heading_path_count,
            heading_path_mean,
            heading_path_median,
            _heading_path_p05,
            _heading_path_p95,
            _heading_path_max,
        ) = _finite_summary(heading_path_values)

        if interval_s is not None and interval_prev_end_frame is not None and interval_next_start_frame is not None:
            interval_values = np.asarray(interval_s, dtype=np.float64)
            interval_mask = (
                np.isfinite(interval_values)
                & (np.asarray(interval_prev_end_frame, dtype=np.int64) >= int(window.start_frame))
                & (np.asarray(interval_next_start_frame, dtype=np.int64) <= int(window.end_frame))
            )
            if interval_valid is not None:
                interval_mask &= np.asarray(interval_valid, dtype=bool)
        elif interval_s is not None and interval_prev_end_s is not None and interval_next_start_s is not None:
            interval_values = np.asarray(interval_s, dtype=np.float64)
            interval_mask = (
                np.isfinite(interval_values)
                & (np.asarray(interval_prev_end_s, dtype=np.float64) >= float(window.start_time_s))
                & (np.asarray(interval_next_start_s, dtype=np.float64) <= float(window.end_time_s))
            )
            if interval_valid is not None:
                interval_mask &= np.asarray(interval_valid, dtype=bool)
        else:
            interval_values = np.asarray([], dtype=np.float64)
            interval_mask = np.zeros(0, dtype=bool)
        interval_count, interval_mean, interval_median, interval_p05, interval_p95, _interval_max = _finite_summary(
            interval_values[interval_mask] if interval_values.size else np.asarray([])
        )
        interval_rate = (float(interval_count) / (duration_s / 60.0)) if duration_s > 0 else np.nan

        out[window_index] = (
            int(window.window_id),
            int(window_index),
            str(window.label).encode("utf-8", "ignore")[:95],
            int(window.start_frame),
            int(window.end_frame),
            float(window.start_time_s),
            float(window.end_time_s),
            duration_s,
            int(total_span_frames),
            int(valid_frame_count),
            int(missing_frame_count),
            float(missing_frame_count / total_span_frames) if total_span_frames > 0 else np.nan,
            int(center_count),
            center_mean,
            center_median,
            center_p05,
            center_p95,
            center_max,
            float(arena_radius_mm),
            float(wall_band_mm),
            int(wall_frame_count),
            wall_fraction,
            wall_time_s,
            int(speed_count),
            speed_mean,
            speed_median,
            speed_p05,
            speed_p95,
            speed_max,
            total_path_mm,
            int(bout_count),
            bout_rate,
            bout_duration_median,
            bout_duration_mean,
            bout_path_median,
            bout_path_mean,
            int(heading_count),
            heading_mean,
            heading_median,
            heading_abs_mean,
            heading_abs_median,
            heading_path_mean,
            heading_path_median,
            int(interval_count),
            interval_mean,
            interval_median,
            interval_p05,
            interval_p95,
            interval_rate,
        )
    return out


def _make_center_distance_histogram(
    *,
    windows: Sequence[ChaserDistanceWindow],
    run_group: zarr.Group,
    geometry: ArenaGeometry,
    bin_width_mm: float,
    wall_band_mm: float,
) -> np.ndarray:
    dtype = np.dtype(
        [
            ("window_id", np.int32),
            ("window_index", np.int32),
            ("window_label", "S96"),
            ("start_frame", np.int64),
            ("end_frame", np.int64),
            ("start_time_s", np.float64),
            ("end_time_s", np.float64),
            ("duration_s", np.float64),
            ("bin_index", np.int32),
            ("bin_left_mm", np.float64),
            ("bin_right_mm", np.float64),
            ("bin_center_mm", np.float64),
            ("bin_width_mm", np.float64),
            ("hist_count", np.int64),
            ("hist_fraction", np.float64),
            ("hist_density_per_mm", np.float64),
            ("valid_frame_count", np.int64),
            ("arena_radius_mm", np.float64),
            ("wall_band_mm", np.float64),
            ("geometry_status", "S48"),
        ]
    )
    center_distance_mm, in_arena_mask, arena_radius_mm = _center_distance_mm(run_group, geometry)
    fish_valid = _fish_valid_by_frame(run_group)
    if center_distance_mm.shape[0] != fish_valid.shape[0]:
        n = min(center_distance_mm.shape[0], fish_valid.shape[0])
        padded_center = np.full(fish_valid.shape[0], np.nan, dtype=np.float64)
        padded_in_arena = np.zeros(fish_valid.shape[0], dtype=bool)
        if n > 0:
            padded_center[:n] = center_distance_mm[:n]
            padded_in_arena[:n] = in_arena_mask[:n]
        center_distance_mm = padded_center
        in_arena_mask = padded_in_arena
    if not math.isfinite(float(arena_radius_mm)) or arena_radius_mm <= 0:
        return np.zeros(0, dtype=dtype)
    width = max(0.1, float(bin_width_mm))
    edges = np.arange(0.0, float(arena_radius_mm) + width, width, dtype=np.float64)
    if edges.shape[0] < 2 or edges[-1] < float(arena_radius_mm):
        edges = np.append(edges, float(arena_radius_mm))
    rows = np.zeros(len(windows) * (edges.shape[0] - 1), dtype=dtype)
    for name in rows.dtype.names or ():
        if rows.dtype[name].kind == "f":
            rows[name] = np.nan
    row_idx = 0
    for window_index, window in enumerate(windows):
        start = max(0, int(window.start_frame))
        end = min(int(window.end_frame), int(fish_valid.shape[0]) - 1) if fish_valid.size else -1
        if end >= start:
            values = center_distance_mm[start : end + 1]
            valid = fish_valid[start : end + 1] & in_arena_mask[start : end + 1] & np.isfinite(values)
            valid_values = values[valid]
        else:
            valid_values = np.asarray([], dtype=np.float64)
        counts, _ = np.histogram(valid_values, bins=edges)
        total = int(np.sum(counts))
        for bin_index in range(edges.shape[0] - 1):
            left = float(edges[bin_index])
            right = float(edges[bin_index + 1])
            bin_width = right - left
            count = int(counts[bin_index])
            fraction = float(count) / float(total) if total > 0 else np.nan
            rows[row_idx] = (
                int(window.window_id),
                int(window_index),
                str(window.label).encode("utf-8", "ignore")[:95],
                int(window.start_frame),
                int(window.end_frame),
                float(window.start_time_s),
                float(window.end_time_s),
                float(window.duration_s),
                int(bin_index),
                left,
                right,
                (left + right) / 2.0,
                bin_width,
                count,
                fraction,
                fraction / bin_width if total > 0 and bin_width > 0 else np.nan,
                total,
                float(arena_radius_mm),
                float(wall_band_mm),
                str(geometry.status).encode("utf-8", "ignore")[:47],
            )
            row_idx += 1
    return rows


def _make_per_epoch_bouts(
    *,
    windows: Sequence[ChaserDistanceWindow],
    run_group: zarr.Group,
    swim_tables: Optional[Any],
    track: Optional[Any],
) -> np.ndarray:
    dtype = np.dtype(
        [
            ("window_id", np.int32),
            ("window_index", np.int32),
            ("window_label", "S96"),
            ("start_frame", np.int64),
            ("end_frame", np.int64),
            ("start_time_s", np.float64),
            ("end_time_s", np.float64),
            ("duration_s", np.float64),
            ("bout_source_row", np.int64),
            ("bout_id", np.int64),
            ("bout_event_frame", np.int64),
            ("bout_event_time_s", np.float64),
            ("bout_start_frame", np.int64),
            ("bout_end_frame", np.int64),
            ("bout_start_time_s", np.float64),
            ("bout_end_time_s", np.float64),
            ("bout_duration_s", np.float64),
            ("bout_path_length_mm", np.float64),
            ("bout_net_heading_change_deg", np.float64),
            ("abs_bout_net_heading_change_deg", np.float64),
            ("bout_heading_path_deg", np.float64),
        ]
    )
    if swim_tables is None or swim_tables.bouts.size == 0:
        return np.zeros(0, dtype=dtype)

    bouts = swim_tables.bouts
    fps = float(run_group.attrs.get("fps") or 0.0)
    bout_event_frame = _first_nonnegative_frame(bouts, "peak_frame", "core_start_frame", "start_frame")
    bout_time_s = _structured_field(bouts, "peak_time_s", "start_time_s", "start_s")
    start_frame, end_frame = _bout_frame_bounds(bouts, fps=fps)
    start_time_s = _structured_field(bouts, "start_time_s", "start_s")
    end_time_s = _structured_field(bouts, "end_time_s", "end_s")
    bout_duration_s = _structured_field(bouts, "duration_s", "observed_duration_s", "elapsed_duration_s")
    bout_path_mm = _structured_field(bouts, "path_length_mm")
    bout_id = _structured_field(bouts, "bout_id")
    bout_net_heading_change_deg, bout_heading_path_deg = _bout_heading_values(
        bouts=bouts,
        track=track,
        fps=fps,
    )

    rows: list[tuple[Any, ...]] = []
    n_bouts = int(bouts.shape[0])
    for window_index, window in enumerate(windows):
        if bout_event_frame is not None:
            event_frame = np.asarray(bout_event_frame, dtype=np.int64)
            bout_mask = (event_frame >= int(window.start_frame)) & (event_frame <= int(window.end_frame))
        else:
            bout_mask = _window_time_mask(bout_time_s, start_s=window.start_time_s, end_s=window.end_time_s)
        if bout_mask.shape[0] != n_bouts:
            continue
        for bout_row in np.flatnonzero(bout_mask):
            event_frame_value = (
                int(np.asarray(bout_event_frame, dtype=np.int64)[bout_row])
                if bout_event_frame is not None
                else -1
            )
            event_time_value = (
                float(np.asarray(bout_time_s, dtype=np.float64)[bout_row])
                if bout_time_s is not None
                else np.nan
            )
            start_frame_value = (
                int(np.asarray(start_frame, dtype=np.int64)[bout_row])
                if start_frame is not None
                else -1
            )
            end_frame_value = (
                int(np.asarray(end_frame, dtype=np.int64)[bout_row])
                if end_frame is not None
                else -1
            )
            start_time_value = (
                float(np.asarray(start_time_s, dtype=np.float64)[bout_row])
                if start_time_s is not None
                else np.nan
            )
            end_time_value = (
                float(np.asarray(end_time_s, dtype=np.float64)[bout_row])
                if end_time_s is not None
                else np.nan
            )
            heading_value = (
                float(bout_net_heading_change_deg[bout_row])
                if bout_row < bout_net_heading_change_deg.shape[0]
                else np.nan
            )
            heading_path_value = (
                float(bout_heading_path_deg[bout_row])
                if bout_row < bout_heading_path_deg.shape[0]
                else np.nan
            )
            rows.append(
                (
                    int(window.window_id),
                    int(window_index),
                    str(window.label).encode("utf-8", "ignore")[:95],
                    int(window.start_frame),
                    int(window.end_frame),
                    float(window.start_time_s),
                    float(window.end_time_s),
                    float(window.duration_s),
                    int(bout_row),
                    int(np.asarray(bout_id, dtype=np.int64)[bout_row]) if bout_id is not None else int(bout_row),
                    event_frame_value,
                    event_time_value,
                    start_frame_value,
                    end_frame_value,
                    start_time_value,
                    end_time_value,
                    float(np.asarray(bout_duration_s, dtype=np.float64)[bout_row])
                    if bout_duration_s is not None
                    else np.nan,
                    float(np.asarray(bout_path_mm, dtype=np.float64)[bout_row])
                    if bout_path_mm is not None
                    else np.nan,
                    heading_value,
                    abs(heading_value) if math.isfinite(heading_value) else np.nan,
                    heading_path_value,
                )
            )
    if not rows:
        return np.zeros(0, dtype=dtype)
    return np.asarray(rows, dtype=dtype)


def _degree_edges(start: float, stop: float, width: float) -> tuple[float, ...]:
    return tuple(float(value) for value in np.arange(float(start), float(stop) + float(width) * 0.5, float(width)))


def _bout_histogram_specs() -> tuple[HistogramMetricSpec, ...]:
    heading_width = float(DEFAULT_BOUT_HEADING_BIN_WIDTH_DEG)
    return (
        HistogramMetricSpec(
            metric_name="bout_duration_s",
            units="s",
            bin_policy="fixed_width_from_zero_to_component_max",
            bin_width=float(DEFAULT_BOUT_DURATION_BIN_WIDTH_S),
            range_min=0.0,
        ),
        HistogramMetricSpec(
            metric_name="bout_path_length_mm",
            units="mm",
            bin_policy="fixed_width_from_zero_to_component_max",
            bin_width=float(DEFAULT_BOUT_DISTANCE_BIN_WIDTH_MM),
            range_min=0.0,
        ),
        HistogramMetricSpec(
            metric_name="bout_net_heading_change_deg",
            units="deg",
            bin_policy="fixed_edges_-180_to_180_deg",
            bin_edges=_degree_edges(-180.0, 180.0, heading_width),
        ),
        HistogramMetricSpec(
            metric_name="abs_bout_net_heading_change_deg",
            units="deg",
            bin_policy="fixed_edges_0_to_180_deg",
            bin_edges=_degree_edges(0.0, 180.0, heading_width),
        ),
        HistogramMetricSpec(
            metric_name="bout_heading_path_deg",
            units="deg",
            bin_policy="fixed_width_from_zero_to_component_max",
            bin_width=heading_width,
            range_min=0.0,
        ),
    )


def _make_per_epoch_bout_histograms(
    *,
    windows: Sequence[ChaserDistanceWindow],
    per_epoch_bouts: np.ndarray,
) -> np.ndarray:
    segments = segments_from_window_objects(windows)
    tables: list[np.ndarray] = []
    for spec in _bout_histogram_specs():
        if per_epoch_bouts.size == 0 or per_epoch_bouts.dtype.names is None or spec.metric_name not in per_epoch_bouts.dtype.names:
            values_by_window = [np.asarray([], dtype=np.float64) for _ in segments]
        else:
            values = np.asarray(per_epoch_bouts[spec.metric_name], dtype=np.float64)
            window_ids = np.asarray(per_epoch_bouts["window_id"], dtype=np.int32)
            values_by_window = [values[window_ids == int(segment.segment_id)] for segment in segments]
        tables.append(
            histogram_table(
                segments=segments,
                values_by_segment=values_by_window,
                metric_spec=spec,
            )
        )
    if not tables:
        return np.zeros(0, dtype=[])
    return np.concatenate(tables) if any(table.size for table in tables) else tables[0]


def _inter_bout_interval_values_by_window(
    *,
    windows: Sequence[ChaserDistanceWindow],
    swim_tables: Optional[Any],
) -> list[np.ndarray]:
    if swim_tables is None or swim_tables.inter_bout_intervals.size == 0:
        return [np.asarray([], dtype=np.float64) for _ in windows]
    intervals = swim_tables.inter_bout_intervals
    interval_s = _structured_field(intervals, "interval_s")
    interval_prev_end_frame = _structured_field(intervals, "prev_end_frame")
    interval_next_start_frame = _structured_field(intervals, "next_start_frame")
    interval_prev_end_s = _structured_field(intervals, "prev_end_time_s")
    interval_next_start_s = _structured_field(intervals, "next_start_time_s")
    interval_valid = _structured_field(intervals, "valid")
    if interval_s is None:
        return [np.asarray([], dtype=np.float64) for _ in windows]
    interval_values = np.asarray(interval_s, dtype=np.float64)
    values_by_window: list[np.ndarray] = []
    for window in windows:
        if interval_prev_end_frame is not None and interval_next_start_frame is not None:
            interval_mask = (
                np.isfinite(interval_values)
                & (np.asarray(interval_prev_end_frame, dtype=np.int64) >= int(window.start_frame))
                & (np.asarray(interval_next_start_frame, dtype=np.int64) <= int(window.end_frame))
            )
        elif interval_prev_end_s is not None and interval_next_start_s is not None:
            interval_mask = (
                np.isfinite(interval_values)
                & (np.asarray(interval_prev_end_s, dtype=np.float64) >= float(window.start_time_s))
                & (np.asarray(interval_next_start_s, dtype=np.float64) <= float(window.end_time_s))
            )
        else:
            interval_mask = np.zeros(interval_values.shape[0], dtype=bool)
        if interval_valid is not None:
            interval_mask &= np.asarray(interval_valid, dtype=bool)
        values_by_window.append(interval_values[interval_mask])
    return values_by_window


def _make_per_epoch_inter_bout_interval_histograms(
    *,
    windows: Sequence[ChaserDistanceWindow],
    swim_tables: Optional[Any],
) -> np.ndarray:
    segments = segments_from_window_objects(windows)
    spec = HistogramMetricSpec(
        metric_name="inter_bout_interval_s",
        units="s",
        bin_policy="fixed_width_from_zero_to_component_max",
        bin_width=float(DEFAULT_IBI_BIN_WIDTH_S),
        range_min=0.0,
    )
    return histogram_table(
        segments=segments,
        values_by_segment=_inter_bout_interval_values_by_window(windows=windows, swim_tables=swim_tables),
        metric_spec=spec,
    )


def _array_float(values: Optional[np.ndarray], *indices: int) -> float:
    if values is None:
        return np.nan
    try:
        return float(np.asarray(values)[indices])
    except Exception:
        return np.nan


def _array_int(values: Optional[np.ndarray], *indices: int) -> int:
    if values is None:
        return 0
    try:
        return int(np.asarray(values)[indices])
    except Exception:
        return 0


def _make_per_epoch_chaser(
    *,
    windows: Sequence[ChaserDistanceWindow],
    run_group: zarr.Group,
) -> np.ndarray:
    summary = run_group["epoch_summary"]
    valid_count = np.asarray(summary["valid_frame_count"][:]) if "valid_frame_count" in summary else None
    n_windows = len(windows)
    if valid_count is not None and np.asarray(valid_count).ndim == 2:
        n_chasers = int(np.asarray(valid_count).shape[1])
    elif "chasers" in run_group and "chaser_index" in run_group["chasers"]:
        n_chasers = int(np.asarray(run_group["chasers/chaser_index"][:]).reshape(-1).shape[0])
    else:
        n_chasers = 0
    if "chasers" in run_group and "chaser_index" in run_group["chasers"]:
        chaser_group = run_group["chasers"]
        chaser_indices = np.asarray(
            chaser_group["chaser_index"][:], dtype=np.int32
        ).reshape(-1)
    else:
        chaser_group = None
        chaser_indices = np.arange(n_chasers, dtype=np.int32)
    if chaser_group is not None and "behavior_class_id" in chaser_group:
        behavior_class_ids = np.asarray(
            chaser_group["behavior_class_id"][:], dtype=np.int8
        ).reshape(-1)
    else:
        behavior_class_ids = np.zeros(n_chasers, dtype=np.int8)
    if chaser_group is not None and "behavior_class_label_bytes" in chaser_group:
        behavior_class_labels = _decode_text_column(
            np.asarray(chaser_group["behavior_class_label_bytes"][:])
        )
    else:
        behavior_class_labels = ["unknown"] * n_chasers

    mean_distance = np.asarray(summary["mean_distance_mm"][:]) if "mean_distance_mm" in summary else None
    min_distance = np.asarray(summary["min_distance_mm"][:]) if "min_distance_mm" in summary else None
    p05_distance = np.asarray(summary["p05_distance_mm"][:]) if "p05_distance_mm" in summary else None
    p50_distance = np.asarray(summary["p50_distance_mm"][:]) if "p50_distance_mm" in summary else None
    p95_distance = np.asarray(summary["p95_distance_mm"][:]) if "p95_distance_mm" in summary else None
    fraction_within = (
        np.asarray(summary["fraction_within_threshold"][:])
        if "fraction_within_threshold" in summary
        else None
    )
    threshold_mm = float(summary.attrs.get("threshold_mm", np.nan))

    dtype = np.dtype(
        [
            ("window_id", np.int32),
            ("window_index", np.int32),
            ("window_label", "S96"),
            ("start_frame", np.int64),
            ("end_frame", np.int64),
            ("start_time_s", np.float64),
            ("end_time_s", np.float64),
            ("duration_s", np.float64),
            ("chaser_column_index", np.int32),
            ("chaser_index", np.int32),
            ("behavior_class_id", np.int8),
            ("behavior_class", "S32"),
            ("threshold_mm", np.float64),
            ("distance_sample_count", np.int64),
            ("mean_distance_mm", np.float64),
            ("median_distance_mm", np.float64),
            ("p05_distance_mm", np.float64),
            ("p95_distance_mm", np.float64),
            ("min_distance_mm", np.float64),
            ("fraction_within_threshold", np.float64),
        ]
    )
    out = np.zeros(n_windows * n_chasers, dtype=dtype)
    for name in out.dtype.names or ():
        if out.dtype[name].kind == "f":
            out[name] = np.nan

    row_idx = 0
    for window_index, window in enumerate(windows):
        for chaser_col in range(n_chasers):
            chaser_index = (
                int(chaser_indices[chaser_col])
                if chaser_col < chaser_indices.shape[0]
                else int(chaser_col)
            )
            behavior_class_id = (
                int(behavior_class_ids[chaser_col])
                if chaser_col < behavior_class_ids.shape[0]
                else 0
            )
            behavior_class = (
                str(behavior_class_labels[chaser_col])
                if chaser_col < len(behavior_class_labels)
                else "unknown"
            )
            out[row_idx] = (
                int(window.window_id),
                int(window_index),
                str(window.label).encode("utf-8", "ignore")[:95],
                int(window.start_frame),
                int(window.end_frame),
                float(window.start_time_s),
                float(window.end_time_s),
                float(window.duration_s),
                int(chaser_col),
                int(chaser_index),
                int(behavior_class_id),
                behavior_class.encode("utf-8", "ignore")[:31],
                threshold_mm,
                _array_int(valid_count, window_index, chaser_col),
                _array_float(mean_distance, window_index, chaser_col),
                _array_float(p50_distance, window_index, chaser_col),
                _array_float(p05_distance, window_index, chaser_col),
                _array_float(p95_distance, window_index, chaser_col),
                _array_float(min_distance, window_index, chaser_col),
                _array_float(fraction_within, window_index, chaser_col),
            )
            row_idx += 1
    return out


def build_chaser_epoch_behavior_summary_result(
    zarr_path: Path | str,
    *,
    chaser_distance_run: str | None = "latest",
    component_name: str = DEFAULT_COMPONENT_NAME,
    swim_bout_run: str | None = "latest",
    track_kinematics_run: str | None = None,
    track_kinematics_scope: str = REQUIRED_TRACK_SCOPE,
    track_id: int | None = None,
    speed_level: str | None = None,
    center_distance_bin_width_mm: float = DEFAULT_CENTER_DISTANCE_BIN_WIDTH_MM,
    wall_band_mm: float = DEFAULT_WALL_BAND_MM,
) -> ChaserEpochBehaviorSummaryResult:
    archive = Path(zarr_path)
    root = _open_root(archive, mode="r")
    distance, run_name, run_path = _resolve_chaser_distance_run(
        root,
        chaser_distance_run,
    )
    # per_epoch_chaser persists behavior class IDs/labels, so it cannot be
    # published from the current unsealed protocol-derived role surfaces.
    distance.require_behavior_authority()
    run_group = root[run_path]
    attrs = _attrs_dict(run_group)
    fps = float(attrs.get("fps") or 1.0)
    windows = _load_windows(run_group, fps=fps)
    if not windows:
        raise ValueError(f"{run_path} has no epoch_summary windows.")

    (
        swim_tables,
        track,
        source_track_run,
        source_track_scope,
        source_track_id,
        source_speed_level,
        source_speed_level_selection,
        warnings,
    ) = _resolve_speed_sources(
        root,
        swim_bout_run=swim_bout_run,
        track_kinematics_run=track_kinematics_run,
        track_kinematics_scope=track_kinematics_scope,
        track_id=track_id,
        speed_level=speed_level,
    )
    geometry = _resolve_arena_geometry(root, run_group)
    if geometry.shape != "circle":
        warnings.append(f"arena_geometry_unavailable: {geometry.status}")
    per_epoch_fish = _make_per_epoch_fish(
        windows=windows,
        run_group=run_group,
        swim_tables=swim_tables,
        track=track,
        source_speed_level=source_speed_level,
        geometry=geometry,
        wall_band_mm=float(wall_band_mm),
    )
    per_epoch_chaser = _make_per_epoch_chaser(windows=windows, run_group=run_group)
    per_epoch_bouts = _make_per_epoch_bouts(
        windows=windows,
        run_group=run_group,
        swim_tables=swim_tables,
        track=track,
    )
    per_epoch_bout_histograms = _make_per_epoch_bout_histograms(
        windows=windows,
        per_epoch_bouts=per_epoch_bouts,
    )
    per_epoch_inter_bout_interval_histograms = _make_per_epoch_inter_bout_interval_histograms(
        windows=windows,
        swim_tables=swim_tables,
    )
    center_distance_histogram = _make_center_distance_histogram(
        windows=windows,
        run_group=run_group,
        geometry=geometry,
        bin_width_mm=float(center_distance_bin_width_mm),
        wall_band_mm=float(wall_band_mm),
    )
    recording_id = str(
        attrs.get("recording_id") or root.attrs.get("recording_id") or archive.stem
    )
    return ChaserEpochBehaviorSummaryResult(
        zarr_path=str(archive),
        recording_id=recording_id,
        component_name=str(component_name),
        chaser_distance_run_name=run_name,
        chaser_distance_run_path=run_path,
        source_track_kinematics_run=source_track_run,
        source_track_kinematics_scope=source_track_scope,
        source_track_kinematics_track_id=source_track_id,
        source_track_kinematics_track_path=track.track_path if track is not None else None,
        source_speed_level=source_speed_level,
        source_speed_level_selection=source_speed_level_selection,
        source_swim_bout_run=swim_tables.run_name if swim_tables is not None else None,
        source_swim_bout_path=swim_tables.run_path if swim_tables is not None else None,
        source_swim_bout_level_path=swim_tables.level_path if swim_tables is not None else None,
        source_swim_bout_signal_level=swim_tables.signal.speed_level if swim_tables is not None else None,
        fps=fps,
        windows=windows,
        per_epoch_fish=per_epoch_fish,
        per_epoch_chaser=per_epoch_chaser,
        per_epoch_bouts=per_epoch_bouts,
        per_epoch_bout_histograms=per_epoch_bout_histograms,
        per_epoch_inter_bout_interval_histograms=per_epoch_inter_bout_interval_histograms,
        center_distance_histogram=center_distance_histogram,
        arena_geometry=geometry,
        center_distance_bin_width_mm=float(center_distance_bin_width_mm),
        wall_band_mm=float(wall_band_mm),
        warnings=tuple(warnings),
    )


def write_chaser_epoch_behavior_summary_component(
    zarr_path: Path | str,
    result: ChaserEpochBehaviorSummaryResult,
    *,
    overwrite: bool = False,
) -> str:
    root = _open_root(Path(zarr_path), mode="a")
    reject_unsealed_chaser_derived_publication(
        root,
        run_name=result.chaser_distance_run_name,
        run_path=result.chaser_distance_run_path,
        relative_path=f"{COMPONENT_PARENT_NAME}/{result.component_name}",
    )
    run_group = root[result.chaser_distance_run_path]
    parent = run_group.require_group(COMPONENT_PARENT_NAME)
    component_name = result.component_name
    if component_name in parent:
        if not overwrite:
            raise ValueError(
                "chaser epoch behavior summary component already exists: "
                f"{result.chaser_distance_run_path}/{COMPONENT_PARENT_NAME}/{component_name}"
            )
        del parent[component_name]
    component = parent.create_group(component_name)

    write_columnar_dataset(
        component,
        "per_epoch_fish",
        result.per_epoch_fish,
        {"row_axis": "stimulus_epoch_windows"},
    )
    write_columnar_dataset(
        component,
        "per_epoch_chaser",
        result.per_epoch_chaser,
        {"row_axis": "stimulus_epoch_windows_x_chasers"},
    )
    write_columnar_dataset(
        component,
        "per_epoch_bouts",
        result.per_epoch_bouts,
        {
            "row_axis": "stimulus_epoch_windows_x_swim_bouts",
            "unit_of_analysis": "swim_bout",
            "epoch_assignment_rule": "first nonnegative peak/core_start/start frame within inclusive epoch; time fallback",
        },
    )
    write_columnar_dataset(
        component,
        "per_epoch_bout_histograms",
        result.per_epoch_bout_histograms,
        {
            "row_axis": "stimulus_epoch_windows_x_bout_metrics_x_bins",
            "unit_of_analysis": "swim_bout",
            "source_table": "per_epoch_bouts",
            "bin_contract": "analysis_owned_shared_bins_per_metric_within_component",
        },
    )
    write_columnar_dataset(
        component,
        "per_epoch_inter_bout_interval_histograms",
        result.per_epoch_inter_bout_interval_histograms,
        {
            "row_axis": "stimulus_epoch_windows_x_inter_bout_interval_bins",
            "unit_of_analysis": "inter_bout_interval",
            "source_table": "source_swim_bout_run/inter_bout_intervals",
            "epoch_assignment_rule": "prev_end and next_start within inclusive epoch; time fallback",
            "bin_contract": "analysis_owned_shared_bins_within_component",
        },
    )
    write_columnar_dataset(
        component,
        "center_distance_histogram",
        result.center_distance_histogram,
        {"row_axis": "stimulus_epoch_windows_x_center_distance_bins"},
    )
    git = get_git_info(Path(__file__).resolve().parents[3])
    source_refs = {
        "source_chaser_distance_run": result.chaser_distance_run_name,
        "source_chaser_distance_path": result.chaser_distance_run_path,
        "source_stimulus_epoch_run": run_group.attrs.get("source_stimulus_epoch_run"),
        "source_stimulus_epoch_path": run_group.attrs.get("source_stimulus_epoch_path"),
        "source_track_kinematics_run": result.source_track_kinematics_run,
        "source_track_kinematics_scope": result.source_track_kinematics_scope,
        "source_track_kinematics_track_id": result.source_track_kinematics_track_id,
        "source_track_kinematics_track_path": result.source_track_kinematics_track_path,
        "source_swim_bout_run": result.source_swim_bout_run,
        "source_swim_bout_path": result.source_swim_bout_path,
        "source_swim_bout_level_path": result.source_swim_bout_level_path,
    }
    parameters = {
        "speed_level": result.source_speed_level,
        "speed_level_semantics": "physical_track_kinematics_speed_for_epoch_summaries",
        "speed_level_selection": result.source_speed_level_selection,
        "swim_bout_signal_level": result.source_swim_bout_signal_level,
        "swim_bout_signal_semantics": "persisted_detector_signal_for_bout_events",
        "bout_assignment_rule": "first nonnegative peak/core_start/start frame within inclusive epoch; time fallback",
        "inter_bout_interval_assignment_rule": "prev_end and next_start within inclusive epoch; time fallback",
        "bout_heading_change_assignment_rule": "heading samples within bout start/end frame or time-derived frame bounds",
        "center_distance_bin_width_mm": float(result.center_distance_bin_width_mm),
        "wall_band_mm": float(result.wall_band_mm),
        "window_boundary_rule": "inclusive start_frame/end_frame",
        "per_epoch_bout_histogram_metrics": [spec.metric_name for spec in _bout_histogram_specs()],
        "bout_duration_bin_width_s": float(DEFAULT_BOUT_DURATION_BIN_WIDTH_S),
        "bout_distance_bin_width_mm": float(DEFAULT_BOUT_DISTANCE_BIN_WIDTH_MM),
        "bout_heading_bin_width_deg": float(DEFAULT_BOUT_HEADING_BIN_WIDTH_DEG),
        "inter_bout_interval_bin_width_s": float(DEFAULT_IBI_BIN_WIDTH_S),
        "histogram_bin_contract": "analysis_owned_shared_bins_per_metric_within_component",
    }
    summary = {
        "epoch_labels": [
            decode_null_terminated_text(value, errors="ignore")
            for value in result.per_epoch_fish["window_label"]
        ],
        "bout_count": result.per_epoch_fish["bout_count"].astype(int).tolist(),
        "inter_bout_interval_count": result.per_epoch_fish["inter_bout_interval_count"].astype(int).tolist(),
        "mean_inter_bout_interval_s": result.per_epoch_fish["mean_inter_bout_interval_s"].astype(float).tolist(),
        "per_epoch_bout_histogram_rows": int(result.per_epoch_bout_histograms.shape[0]),
        "per_epoch_inter_bout_interval_histogram_rows": int(
            result.per_epoch_inter_bout_interval_histograms.shape[0]
        ),
    }
    component.attrs.update(
        json_attr_safe(
            {
                "schema_id": SCHEMA_ID,
                "schema_version": SCHEMA_VERSION,
                "method": METHOD,
                "method_version": METHOD_VERSION,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "component_name": component_name,
                "recording_id": result.recording_id,
                "zarr_path": str(zarr_path),
                "status": "complete",
                "source_refs": source_refs,
                "parameters": parameters,
                "arena_geometry": {
                    "status": result.arena_geometry.status,
                    "source": result.arena_geometry.source,
                    "shape": result.arena_geometry.shape,
                    "center_x_px": result.arena_geometry.center_x_px,
                    "center_y_px": result.arena_geometry.center_y_px,
                    "radius_px": result.arena_geometry.radius_px,
                },
                "summary": summary,
                "warnings": list(result.warnings),
                "git_commit": git.get("commit_hash"),
                "git_branch": git.get("branch"),
                "git_dirty": git.get("is_dirty"),
                "provenance": {
                    "stage": METHOD,
                    "created_by": "fisheye.analysis.chaser_epoch_behavior_summary",
                    "inputs": source_refs,
                    "parameters": parameters,
                },
            }
        )
    )
    parent.attrs["latest"] = component_name
    parent.attrs["latest_complete"] = component_name
    return f"{result.chaser_distance_run_path}/{COMPONENT_PARENT_NAME}/{component_name}"


def run_for_zarr(
    zarr_path: Path | str,
    *,
    chaser_distance_run: str | None = "latest",
    component_name: str = DEFAULT_COMPONENT_NAME,
    swim_bout_run: str | None = "latest",
    track_kinematics_run: str | None = None,
    track_kinematics_scope: str = REQUIRED_TRACK_SCOPE,
    track_id: int | None = None,
    speed_level: str | None = None,
    center_distance_bin_width_mm: float = DEFAULT_CENTER_DISTANCE_BIN_WIDTH_MM,
    wall_band_mm: float = DEFAULT_WALL_BAND_MM,
    overwrite: bool = False,
) -> str:
    result = build_chaser_epoch_behavior_summary_result(
        zarr_path,
        chaser_distance_run=chaser_distance_run,
        component_name=component_name,
        swim_bout_run=swim_bout_run,
        track_kinematics_run=track_kinematics_run,
        track_kinematics_scope=track_kinematics_scope,
        track_id=track_id,
        speed_level=speed_level,
        center_distance_bin_width_mm=float(center_distance_bin_width_mm),
        wall_band_mm=float(wall_band_mm),
    )
    return write_chaser_epoch_behavior_summary_component(
        zarr_path,
        result,
        overwrite=overwrite,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zarr_path", type=Path)
    parser.add_argument("--chaser-distance-run", default="latest")
    parser.add_argument("--component-name", default=DEFAULT_COMPONENT_NAME)
    parser.add_argument("--swim-bout-run", default="latest")
    parser.add_argument("--track-kinematics-run")
    parser.add_argument("--track-kinematics-scope", default=REQUIRED_TRACK_SCOPE)
    parser.add_argument("--track-id", type=int)
    parser.add_argument("--speed-level")
    parser.add_argument("--center-distance-bin-width-mm", type=float, default=DEFAULT_CENTER_DISTANCE_BIN_WIDTH_MM)
    parser.add_argument("--wall-band-mm", type=float, default=DEFAULT_WALL_BAND_MM)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    path = run_for_zarr(
        args.zarr_path,
        chaser_distance_run=args.chaser_distance_run,
        component_name=args.component_name,
        swim_bout_run=args.swim_bout_run,
        track_kinematics_run=args.track_kinematics_run,
        track_kinematics_scope=args.track_kinematics_scope,
        track_id=args.track_id,
        speed_level=args.speed_level,
        center_distance_bin_width_mm=args.center_distance_bin_width_mm,
        wall_band_mm=args.wall_band_mm,
        overwrite=bool(args.overwrite),
    )
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
