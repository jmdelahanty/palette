"""GoodCopBadCop chaser dashboard component for Palette marimo explorers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np
import pandas as pd
import polars as pl

from fisheye.analysis.swim_bout_io import load_default_swim_bout_tables
from fisheye.analysis.track_kinematics_io import load_track_kinematics_track
from fisheye.utils.view_zarr_visualization import load_png_artifact_bytes
from fisheye.utils.zarr_io import open_zarr_root
from fisheye.visualization.goodcopbadcop_interactive import (
    GOODCOPBADCOP_CHASER_DASHBOARD_RENDERER,
    GoodCopBadCopCRAEndpointData,
    GoodCopBadCopCRANearFieldData,
    GoodCopBadCopEscapeFreezeData,
    GoodCopBadCopEpochBehaviorData,
    GoodCopBadCopInteractiveData,
    load_goodcopbadcop_escape_freeze_data,
    load_goodcopbadcop_epoch_behavior_data,
    load_goodcopbadcop_cra_near_field_data,
    load_goodcopbadcop_cra_primary_endpoint_data,
    load_goodcopbadcop_interactive_data,
    to_chaser_position_dataframe,
    to_distance_timeseries_dataframe,
    to_egocentric_bearing_dataframe,
    to_egocentric_distance_alignment_dataframe,
    to_egocentric_heading_dataframe,
    to_position_dataframe,
    to_spatial_occupancy_dataframe,
    to_window_dataframe,
)

from .common import add_epoch_overlays, apply_full_width_timeseries_layout, join_path, png_bytes_to_markdown_image
from .registry import InteractiveSpecOption


EGOCENTRIC_PRE_POST_POLAR_PNG_ARTIFACT_NAME = "egocentric_bearing_pre_post_polar_png"
EGOCENTRIC_PRE_POST_POLAR_POINT_CLOUD_PNG_ARTIFACT_NAME = "egocentric_bearing_pre_post_polar_point_cloud_png"
EGOCENTRIC_POLAR_DISPLAY_MIN_DISTANCE_BIN_WIDTH_MM = 5.0
EGOCENTRIC_POLAR_DISPLAY_MIN_BEARING_BIN_WIDTH_DEG = 30.0
EGOCENTRIC_POLAR_DISPLAY_COLOR_CMAX_QUANTILE = 0.98
CHASER_MARKER_COLORS = (
    "#2563eb",
    "#dc2626",
    "#16a34a",
    "#9333ea",
    "#ea580c",
    "#0891b2",
)
CHASER_ZONE_PATTERN_SHAPES = ("/", "|", "-", "\\", ".", "+")
CHASER_ZONE_PATTERN_KEY = (
    ("chaser 0 zone", "/"),
    ("chaser 1 zone", "|"),
    ("multiple chasers", "x"),
)


@dataclass(frozen=True)
class GoodCopBadCopLoadedView:
    data: GoodCopBadCopInteractiveData
    cra_endpoint: Optional[GoodCopBadCopCRAEndpointData]
    cra_near_field: Optional[GoodCopBadCopCRANearFieldData]
    escape_freeze: Optional[GoodCopBadCopEscapeFreezeData]
    epoch_behavior: Optional[GoodCopBadCopEpochBehaviorData]
    distance_df: pd.DataFrame
    position_df: pd.DataFrame
    windows_df: pd.DataFrame
    chaser_position_df: pd.DataFrame
    spatial_occupancy_df: pd.DataFrame
    egocentric_bearing_df: pl.DataFrame
    egocentric_alignment_df: pl.DataFrame
    egocentric_heading_df: pl.DataFrame
    egocentric_pre_post_polar_png_path: Optional[str]
    egocentric_pre_post_polar_png_bytes: bytes
    egocentric_pre_post_polar_png_error: Optional[str]
    egocentric_pre_post_polar_point_cloud_png_path: Optional[str]
    egocentric_pre_post_polar_point_cloud_png_bytes: bytes
    egocentric_pre_post_polar_point_cloud_png_error: Optional[str]
    epoch_summary_df: pl.DataFrame
    epoch_summary_source: Optional[str]
    epoch_summary_error: Optional[str]
    epoch_summary_computed_in_viewer: bool
    load_duration_ms: float


@dataclass(frozen=True)
class GoodCopBadCopControls:
    distance_series_picker: Any
    chaser_picker: Any
    time_window: Any
    epoch_picker: Any
    epoch_options: Mapping[str, Optional[int]]
    heatmap_bins: Any
    chaser_overlay: Any
    spatial_zone_set_picker: Any
    view: Any


@dataclass(frozen=True)
class GoodCopBadCopTimeWindow:
    selected_epoch_id: Optional[int]
    selected_epoch_label: str
    start_s: float
    stop_s: float


def is_goodcopbadcop_option(option: InteractiveSpecOption) -> bool:
    return option.renderer == GOODCOPBADCOP_CHASER_DASHBOARD_RENDERER


def _load_egocentric_pre_post_polar_png(
    zarr_path: Path | str,
    data: GoodCopBadCopInteractiveData,
    *,
    artifact_name: str = EGOCENTRIC_PRE_POST_POLAR_PNG_ARTIFACT_NAME,
) -> tuple[Optional[str], bytes, Optional[str]]:
    if not data.egocentric_component_path:
        return None, b"", None
    artifact_path = join_path(
        data.egocentric_component_path,
        "visualizations",
        artifact_name,
    )
    try:
        root = open_zarr_root(Path(zarr_path), mode="r")
        resolved_path, png_bytes = load_png_artifact_bytes(root, artifact_path)
    except Exception as exc:
        return artifact_path, b"", str(exc)
    return resolved_path, png_bytes, None


def _empty_epoch_summary_frame() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "window_id": pl.Int64,
            "window_label": pl.Utf8,
            "window_start_s": pl.Float64,
            "window_end_s": pl.Float64,
            "window_duration_s": pl.Float64,
            "chaser_index": pl.Int64,
            "distance_sample_count": pl.Int64,
            "median_distance_mm": pl.Float64,
            "mean_distance_mm": pl.Float64,
            "p05_distance_mm": pl.Float64,
            "p95_distance_mm": pl.Float64,
            "speed_sample_count": pl.Int64,
            "mean_speed_mm_s": pl.Float64,
            "median_speed_mm_s": pl.Float64,
            "p95_speed_mm_s": pl.Float64,
            "bout_count": pl.Int64,
            "bout_rate_per_min": pl.Float64,
            "median_bout_duration_s": pl.Float64,
            "mean_bout_duration_s": pl.Float64,
            "median_bout_path_length_mm": pl.Float64,
            "mean_bout_path_length_mm": pl.Float64,
            "mean_inter_bout_interval_s": pl.Float64,
            "median_inter_bout_interval_s": pl.Float64,
            "inter_bout_interval_count": pl.Int64,
            "inter_bout_interval_rate_per_min": pl.Float64,
        }
    )


def _structured_field(records: np.ndarray, *names: str) -> Optional[np.ndarray]:
    if records.size == 0 or records.dtype.names is None:
        return None
    for name in names:
        if name in records.dtype.names:
            return np.asarray(records[name])
    return None


def _finite_summary(values: np.ndarray) -> tuple[int, float, float, float, float]:
    data = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return 0, np.nan, np.nan, np.nan, np.nan
    return (
        int(finite.size),
        float(np.nanmedian(finite)),
        float(np.nanmean(finite)),
        float(np.nanpercentile(finite, 5.0)),
        float(np.nanpercentile(finite, 95.0)),
    )


def _window_time_mask(values_s: Optional[np.ndarray], *, start_s: float, end_s: float) -> np.ndarray:
    if values_s is None:
        return np.zeros(0, dtype=bool)
    values = np.asarray(values_s, dtype=np.float64)
    return np.isfinite(values) & (values >= float(start_s)) & (values <= float(end_s))


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
    return out


def _load_epoch_summary_dataframe(
    zarr_path: Path | str,
    data: GoodCopBadCopInteractiveData,
    windows_df: pd.DataFrame,
) -> tuple[pl.DataFrame, Optional[str], Optional[str]]:
    if not len(windows_df):
        return _empty_epoch_summary_frame(), None, "No epoch windows are available."
    try:
        root = open_zarr_root(Path(zarr_path), mode="r")
        swim_tables = load_default_swim_bout_tables(root, run_name="latest")
    except Exception as exc:
        swim_tables = None
        swim_error = str(exc)
    else:
        swim_error = None

    speed_frames = np.zeros(0, dtype=np.int64)
    speed_mm_s = np.asarray([], dtype=np.float64)
    source_track_run: Optional[str] = None
    source_speed_level: Optional[str] = None
    swim_source: Optional[str] = None
    if swim_tables is not None:
        swim_source = swim_tables.run_path
        source_track_run = (
            swim_tables.candidate.source_track_kinematics_run
            or str(swim_tables.run_attrs.get("source_track_kinematics_run") or "").strip()
            or None
        )
        source_speed_level = (
            str(swim_tables.signal.source_level or swim_tables.signal.speed_level or "")
            .replace("speed_", "", 1)
            .strip()
            or None
        )
        try:
            track = load_track_kinematics_track(
                root,
                run_name=source_track_run or "latest",
                scope="offline",
                track_id=int(swim_tables.candidate.track_id or swim_tables.run_attrs.get("track_id") or 0),
                required_speed_levels=(source_speed_level,) if source_speed_level else (),
            )
            speed_frames = np.asarray(track.frame_indices, dtype=np.int64)
            if source_speed_level and source_speed_level in track.speed_mm_by_level:
                speed_mm_s = np.asarray(track.speed_mm_by_level[source_speed_level], dtype=np.float64)
            elif "filtered" in track.speed_mm_by_level:
                source_speed_level = "filtered"
                speed_mm_s = np.asarray(track.speed_mm_by_level["filtered"], dtype=np.float64)
        except Exception:
            speed_frames = np.zeros(0, dtype=np.int64)
            speed_mm_s = np.asarray([], dtype=np.float64)

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

    distance = np.asarray(data.distance_mm, dtype=np.float64)
    n_frames = min(distance.shape[0], data.camera_frame_id.shape[0], data.time_seconds.shape[0])
    distance = distance[:n_frames]
    camera_frames = np.asarray(data.camera_frame_id[:n_frames], dtype=np.int64)

    rows: list[dict[str, object]] = []
    for window_row in windows_df.to_dict("records"):
        window_id = int(window_row["window_id"])
        label = str(window_row["label"])
        start_frame = int(window_row["start_frame"])
        end_frame = int(window_row["end_frame"])
        start_s = float(window_row["start_time_s"])
        end_s = float(window_row["end_time_s"])
        duration_s = max(0.0, end_s - start_s)

        frame_mask = (camera_frames >= start_frame) & (camera_frames <= end_frame)
        if speed_frames.shape[0] == speed_mm_s.shape[0]:
            speed_mask = (speed_frames >= start_frame) & (speed_frames <= end_frame)
            speed_values = speed_mm_s[speed_mask]
        else:
            speed_values = np.asarray([], dtype=np.float64)
        speed_count, speed_median, speed_mean, _speed_p05, speed_p95 = _finite_summary(speed_values)

        if bout_event_frame is not None:
            event_frame = np.asarray(bout_event_frame, dtype=np.int64)
            bout_mask = (event_frame >= start_frame) & (event_frame <= end_frame)
        else:
            bout_mask = _window_time_mask(bout_time_s, start_s=start_s, end_s=end_s)
        bout_count = int(np.sum(bout_mask))
        bout_rate = (float(bout_count) / (duration_s / 60.0)) if duration_s > 0 else np.nan
        _dur_count, bout_duration_median, bout_duration_mean, _dur_p05, _dur_p95 = _finite_summary(
            np.asarray(bout_duration_s, dtype=np.float64)[bout_mask] if bout_duration_s is not None else np.asarray([])
        )
        _path_count, bout_path_median, bout_path_mean, _path_p05, _path_p95 = _finite_summary(
            np.asarray(bout_path_mm, dtype=np.float64)[bout_mask] if bout_path_mm is not None else np.asarray([])
        )

        if interval_s is not None and interval_prev_end_frame is not None and interval_next_start_frame is not None:
            interval_mask = (
                np.isfinite(np.asarray(interval_s, dtype=np.float64))
                & (np.asarray(interval_prev_end_frame, dtype=np.int64) >= start_frame)
                & (np.asarray(interval_next_start_frame, dtype=np.int64) <= end_frame)
            )
            if interval_valid is not None:
                interval_mask &= np.asarray(interval_valid, dtype=bool)
        elif interval_s is not None and interval_prev_end_s is not None and interval_next_start_s is not None:
            interval_mask = (
                np.isfinite(np.asarray(interval_s, dtype=np.float64))
                & (np.asarray(interval_prev_end_s, dtype=np.float64) >= start_s)
                & (np.asarray(interval_next_start_s, dtype=np.float64) <= end_s)
            )
            if interval_valid is not None:
                interval_mask &= np.asarray(interval_valid, dtype=bool)
        else:
            interval_mask = np.zeros(0, dtype=bool)
        interval_count, interval_median, interval_mean, _interval_p05, _interval_p95 = _finite_summary(
            np.asarray(interval_s, dtype=np.float64)[interval_mask] if interval_s is not None else np.asarray([])
        )
        interval_rate = (float(interval_count) / (duration_s / 60.0)) if duration_s > 0 else np.nan

        for chaser_col, chaser_index in enumerate(data.chaser_indices.tolist()):
            chaser_dist = distance[:, chaser_col] if chaser_col < distance.shape[1] else np.asarray([])
            dist_count, dist_median, dist_mean, dist_p05, dist_p95 = _finite_summary(chaser_dist[frame_mask])
            rows.append(
                {
                    "window_id": window_id,
                    "window_label": label,
                    "window_start_s": start_s,
                    "window_end_s": end_s,
                    "window_duration_s": duration_s,
                    "chaser_index": int(chaser_index),
                    "distance_sample_count": dist_count,
                    "median_distance_mm": dist_median,
                    "mean_distance_mm": dist_mean,
                    "p05_distance_mm": dist_p05,
                    "p95_distance_mm": dist_p95,
                    "speed_sample_count": speed_count,
                    "mean_speed_mm_s": speed_mean,
                    "median_speed_mm_s": speed_median,
                    "p95_speed_mm_s": speed_p95,
                    "bout_count": bout_count,
                    "bout_rate_per_min": bout_rate,
                    "median_bout_duration_s": bout_duration_median,
                    "mean_bout_duration_s": bout_duration_mean,
                    "median_bout_path_length_mm": bout_path_median,
                    "mean_bout_path_length_mm": bout_path_mean,
                    "mean_inter_bout_interval_s": interval_mean,
                    "median_inter_bout_interval_s": interval_median,
                    "inter_bout_interval_count": interval_count,
                    "inter_bout_interval_rate_per_min": interval_rate,
                    "source_swim_bout_run": swim_source,
                    "source_track_kinematics_run": source_track_run,
                    "source_speed_level": source_speed_level,
                }
            )
    frame = pl.DataFrame(rows) if rows else _empty_epoch_summary_frame()
    return frame, swim_source, swim_error


def _epoch_summary_from_persisted_behavior(
    epoch_behavior: GoodCopBadCopEpochBehaviorData,
) -> pl.DataFrame:
    chaser = epoch_behavior.per_epoch_chaser_df
    fish = epoch_behavior.per_epoch_fish_df
    if chaser.is_empty():
        return _empty_epoch_summary_frame()

    rename_map = {
        "start_time_s": "window_start_s",
        "end_time_s": "window_end_s",
        "duration_s": "window_duration_s",
    }
    chaser = chaser.rename(
        {
            old: new
            for old, new in rename_map.items()
            if old in chaser.columns and new not in chaser.columns
        }
    )
    if fish.is_empty() or "window_id" not in fish.columns:
        return chaser

    fish_metric_columns = [
        "window_id",
        "total_span_frames",
        "valid_frame_count",
        "missing_frame_count",
        "tracking_dropout_fraction",
        "center_distance_sample_count",
        "mean_distance_from_arena_center_mm",
        "median_distance_from_arena_center_mm",
        "p05_distance_from_arena_center_mm",
        "p95_distance_from_arena_center_mm",
        "max_distance_from_arena_center_mm",
        "arena_radius_mm",
        "wall_band_mm",
        "wall_frame_count",
        "wall_fraction",
        "wall_time_s",
        "speed_sample_count",
        "mean_speed_mm_s",
        "median_speed_mm_s",
        "p05_speed_mm_s",
        "p95_speed_mm_s",
        "max_speed_mm_s",
        "total_path_mm",
        "bout_count",
        "bout_rate_per_min",
        "median_bout_duration_s",
        "mean_bout_duration_s",
        "median_bout_path_length_mm",
        "mean_bout_path_length_mm",
        "bout_heading_sample_count",
        "mean_bout_net_heading_change_deg",
        "median_bout_net_heading_change_deg",
        "mean_abs_bout_net_heading_change_deg",
        "median_abs_bout_net_heading_change_deg",
        "mean_bout_heading_path_deg",
        "median_bout_heading_path_deg",
        "inter_bout_interval_count",
        "mean_inter_bout_interval_s",
        "median_inter_bout_interval_s",
        "p05_inter_bout_interval_s",
        "p95_inter_bout_interval_s",
        "inter_bout_interval_rate_per_min",
    ]
    fish_metrics = fish.select([column for column in fish_metric_columns if column in fish.columns])
    if fish_metrics.is_empty() or "window_id" not in fish_metrics.columns:
        return chaser
    return chaser.join(fish_metrics, on="window_id", how="left")


def load_goodcopbadcop_view(
    zarr_path: Path | str,
    option: InteractiveSpecOption,
    *,
    timer: Any,
) -> GoodCopBadCopLoadedView:
    load_t0 = timer.perf_counter()
    data = load_goodcopbadcop_interactive_data(
        zarr_path,
        run_path=option.run_path,
        artifact_name=option.artifact_name,
    )
    cra_endpoint = load_goodcopbadcop_cra_primary_endpoint_data(
        zarr_path,
        run_path=option.run_path,
    )
    cra_near_field = load_goodcopbadcop_cra_near_field_data(
        zarr_path,
        run_path=option.run_path,
    )
    escape_freeze = load_goodcopbadcop_escape_freeze_data(
        zarr_path,
        run_path=option.run_path,
    )
    epoch_behavior = load_goodcopbadcop_epoch_behavior_data(
        zarr_path,
        run_path=option.run_path,
    )
    distance_df = to_distance_timeseries_dataframe(data)
    position_df = to_position_dataframe(data)
    windows_df = to_window_dataframe(data)
    chaser_position_df = to_chaser_position_dataframe(
        data,
        sample_step=max(1, int(data.fps // 2) or 1),
    )
    spatial_occupancy_df = to_spatial_occupancy_dataframe(data)
    egocentric_bearing_df = to_egocentric_bearing_dataframe(data, valid_only=True)
    egocentric_alignment_df = to_egocentric_distance_alignment_dataframe(data)
    egocentric_heading_df = to_egocentric_heading_dataframe(data, valid_only=True)
    epoch_summary_computed_in_viewer = False
    if epoch_behavior is not None:
        epoch_summary_df = _epoch_summary_from_persisted_behavior(epoch_behavior)
        epoch_summary_source = epoch_behavior.component_path
        epoch_summary_error = None
    else:
        epoch_summary_df, epoch_summary_source, epoch_summary_error = _load_epoch_summary_dataframe(
            zarr_path,
            data,
            windows_df,
        )
        epoch_summary_computed_in_viewer = True
        fallback_detail = "epoch_behavior_summary component is not backfilled"
        epoch_summary_error = (
            f"{fallback_detail}; fallback warning: {epoch_summary_error}"
            if epoch_summary_error
            else fallback_detail
        )
    polar_png_path, polar_png_bytes, polar_png_error = _load_egocentric_pre_post_polar_png(zarr_path, data)
    (
        polar_point_cloud_png_path,
        polar_point_cloud_png_bytes,
        polar_point_cloud_png_error,
    ) = _load_egocentric_pre_post_polar_png(
        zarr_path,
        data,
        artifact_name=EGOCENTRIC_PRE_POST_POLAR_POINT_CLOUD_PNG_ARTIFACT_NAME,
    )
    return GoodCopBadCopLoadedView(
        data=data,
        cra_endpoint=cra_endpoint,
        cra_near_field=cra_near_field,
        escape_freeze=escape_freeze,
        epoch_behavior=epoch_behavior,
        distance_df=distance_df,
        position_df=position_df,
        windows_df=windows_df,
        chaser_position_df=chaser_position_df,
        spatial_occupancy_df=spatial_occupancy_df,
        egocentric_bearing_df=egocentric_bearing_df,
        egocentric_alignment_df=egocentric_alignment_df,
        egocentric_heading_df=egocentric_heading_df,
        egocentric_pre_post_polar_png_path=polar_png_path,
        egocentric_pre_post_polar_png_bytes=polar_png_bytes,
        egocentric_pre_post_polar_png_error=polar_png_error,
        egocentric_pre_post_polar_point_cloud_png_path=polar_point_cloud_png_path,
        egocentric_pre_post_polar_point_cloud_png_bytes=polar_point_cloud_png_bytes,
        egocentric_pre_post_polar_point_cloud_png_error=polar_point_cloud_png_error,
        epoch_summary_df=epoch_summary_df,
        epoch_summary_source=epoch_summary_source,
        epoch_summary_error=epoch_summary_error,
        epoch_summary_computed_in_viewer=epoch_summary_computed_in_viewer,
        load_duration_ms=(timer.perf_counter() - load_t0) * 1000.0,
    )


def build_summary(mo: Any, *, loaded: GoodCopBadCopLoadedView, option: InteractiveSpecOption) -> Any:
    data = loaded.data
    zone_set_count = (
        loaded.spatial_occupancy_df["zone_set_id"].nunique()
        if len(loaded.spatial_occupancy_df)
        else 0
    )
    return mo.vstack(
        [
            mo.md(f"## GoodCopBadCop Chaser Dashboard\n\n`{data.zarr_path}`"),
            mo.hstack(
                [
                    mo.stat(label="Run", value=option.run_name or data.run_name),
                    mo.stat(label="Frames", value=f"{data.camera_frame_id.shape[0]:,}"),
                    mo.stat(label="Chasers", value=f"{data.chaser_indices.shape[0]:,}"),
                    mo.stat(label="Windows", value=f"{len(loaded.windows_df):,}"),
                    mo.stat(label="Position rows", value=f"{len(loaded.position_df):,}"),
                    mo.stat(label="Distance rows", value=f"{len(loaded.distance_df):,}"),
                    mo.stat(label="Zone sets", value=f"{zone_set_count:,}"),
                    mo.stat(label="Egocentric rows", value=f"{loaded.egocentric_bearing_df.height:,}"),
                    mo.stat(label="Epoch summary rows", value=f"{loaded.epoch_summary_df.height:,}"),
                    mo.stat(
                        label="CRA endpoint",
                        value=loaded.cra_endpoint.component_name if loaded.cra_endpoint is not None else "none",
                    ),
                    mo.stat(
                        label="CRA near-field",
                        value=loaded.cra_near_field.component_name if loaded.cra_near_field is not None else "none",
                    ),
                    mo.stat(
                        label="Escape/freezing",
                        value=loaded.escape_freeze.component_name if loaded.escape_freeze is not None else "none",
                    ),
                    mo.stat(label="Load ms", value=f"{loaded.load_duration_ms:.1f}"),
                ]
            ),
        ]
    )


def build_controls(mo: Any, *, loaded: GoodCopBadCopLoadedView) -> GoodCopBadCopControls:
    data = loaded.data
    distance_columns = [
        column
        for column in loaded.distance_df.columns
        if column.startswith("distance_mm_chaser_") or column == "nearest_distance_mm"
    ]
    default_distance_columns = [
        column for column in distance_columns if column.startswith("distance_mm_chaser_")
    ][:2] or distance_columns[:1]
    distance_series_picker = mo.ui.multiselect(
        options=distance_columns,
        value=default_distance_columns,
        label="Distance traces",
    )
    chaser_options = ["All chasers"] + [f"chaser {int(value)}" for value in data.chaser_indices.tolist()]
    chaser_picker = mo.ui.dropdown(
        options=chaser_options,
        value=chaser_options[0],
        label="Chaser",
    )
    max_time = float(data.time_seconds[-1]) if data.time_seconds.size else 0.0
    time_window = mo.ui.range_slider(
        start=0.0,
        stop=max_time,
        value=[0.0, max_time],
        step=max(max_time / 1000.0, 0.001),
        label="Time window (s)",
    )
    epoch_options: dict[str, Optional[int]] = {"Custom time window": None}
    for row in loaded.windows_df.to_dict("records"):
        label = f'{row["label"]} ({row["start_time_s"]:.1f}-{row["end_time_s"]:.1f}s)'
        epoch_options[label] = int(row["window_id"])
    epoch_picker = mo.ui.dropdown(
        options=list(epoch_options),
        value="Custom time window",
        label="Epoch",
    )
    heatmap_bins = mo.ui.slider(start=20, stop=160, value=80, step=10, label="Heatmap bins")
    chaser_overlay = mo.ui.checkbox(value=True, label="Chaser overlay")
    zone_set_options = (
        sorted(str(value) for value in loaded.spatial_occupancy_df["zone_set_id"].dropna().unique())
        if len(loaded.spatial_occupancy_df)
        else []
    )
    spatial_zone_set_picker = (
        mo.ui.dropdown(
            options=zone_set_options,
            value=zone_set_options[0],
            label="Spatial zones",
        )
        if zone_set_options
        else None
    )
    return GoodCopBadCopControls(
        distance_series_picker=distance_series_picker,
        chaser_picker=chaser_picker,
        time_window=time_window,
        epoch_picker=epoch_picker,
        epoch_options=epoch_options,
        heatmap_bins=heatmap_bins,
        chaser_overlay=chaser_overlay,
        spatial_zone_set_picker=spatial_zone_set_picker,
        view=mo.vstack(
            [
                item
                for item in (
                    distance_series_picker,
                    chaser_picker,
                    epoch_picker,
                    spatial_zone_set_picker,
                    time_window,
                    heatmap_bins,
                    chaser_overlay,
                )
                if item is not None
            ]
        ),
    )


def build_controls_panel_from_widgets(
    mo: Any,
    *,
    distance_series_picker: Any,
    time_window: Any,
    epoch_picker: Any,
    epoch_options: Mapping[str, Optional[int]],
    heatmap_bins: Any,
    chaser_overlay: Any,
    chaser_picker: Any = None,
    spatial_zone_set_picker: Any = None,
) -> Any:
    items = [distance_series_picker]
    if chaser_picker is not None:
        items.append(chaser_picker)
    items.append(epoch_picker)
    if spatial_zone_set_picker is not None:
        items.append(spatial_zone_set_picker)
    if epoch_options.get(epoch_picker.value) is None:
        items.append(time_window)
    items.extend([heatmap_bins, chaser_overlay])
    return mo.vstack(items)


def build_controls_panel(mo: Any, *, controls: GoodCopBadCopControls) -> Any:
    return build_controls_panel_from_widgets(
        mo,
        distance_series_picker=controls.distance_series_picker,
        chaser_picker=controls.chaser_picker,
        time_window=controls.time_window,
        epoch_picker=controls.epoch_picker,
        epoch_options=controls.epoch_options,
        heatmap_bins=controls.heatmap_bins,
        chaser_overlay=controls.chaser_overlay,
        spatial_zone_set_picker=controls.spatial_zone_set_picker,
    )


def resolve_time_window(
    *,
    controls: GoodCopBadCopControls,
    windows_df: pd.DataFrame,
) -> GoodCopBadCopTimeWindow:
    return resolve_time_window_from_widgets(
        epoch_options=controls.epoch_options,
        epoch_picker=controls.epoch_picker,
        time_window=controls.time_window,
        windows_df=windows_df,
    )


def resolve_time_window_from_widgets(
    *,
    epoch_options: Mapping[str, Optional[int]],
    epoch_picker: Any,
    time_window: Any,
    windows_df: pd.DataFrame,
) -> GoodCopBadCopTimeWindow:
    selected_epoch_id = epoch_options[epoch_picker.value]
    if selected_epoch_id is None or not len(windows_df):
        start_s, stop_s = [float(value) for value in time_window.value]
        selected_epoch_label = "custom"
    else:
        row = windows_df[windows_df["window_id"].astype(int) == int(selected_epoch_id)].iloc[0]
        start_s = float(row["start_time_s"])
        stop_s = float(row["end_time_s"])
        selected_epoch_label = str(row["label"])
    return GoodCopBadCopTimeWindow(
        selected_epoch_id=selected_epoch_id,
        selected_epoch_label=selected_epoch_label,
        start_s=start_s,
        stop_s=stop_s,
    )


def build_distance_figure(
    go: Any,
    *,
    loaded: GoodCopBadCopLoadedView,
    distance_series_picker: Any,
    window: GoodCopBadCopTimeWindow,
) -> tuple[Any, pd.DataFrame]:
    visible = loaded.distance_df[
        (loaded.distance_df["time_s"] >= window.start_s)
        & (loaded.distance_df["time_s"] <= window.stop_s)
    ].copy()
    fig = go.Figure()
    add_epoch_overlays(fig, loaded.windows_df)
    for column in distance_series_picker.value:
        if column not in visible:
            continue
        fig.add_trace(
            go.Scattergl(
                x=visible["time_s"],
                y=visible[column],
                mode="lines",
                name=column,
            )
        )
    apply_full_width_timeseries_layout(
        fig,
        title=f"Fish-to-Chaser Distance ({window.selected_epoch_label})",
        yaxis_title="Distance (mm)",
    )
    return fig, visible


def _filter_egocentric_window(frame: pl.DataFrame, window: GoodCopBadCopTimeWindow) -> pl.DataFrame:
    if frame.is_empty():
        return frame
    return frame.filter(
        (pl.col("time_s") >= float(window.start_s))
        & (pl.col("time_s") <= float(window.stop_s))
    )


def _selected_chaser_index(chaser_picker: Any = None) -> Optional[int]:
    if chaser_picker is None:
        return None
    value = getattr(chaser_picker, "value", None)
    if value is None:
        return None
    text = str(value).strip().lower()
    if not text or text == "all chasers":
        return None
    if text.startswith("chaser "):
        text = text.split(" ", 1)[1].strip()
    try:
        return int(text)
    except ValueError:
        return None


def _filter_selected_chaser(frame: pl.DataFrame, chaser_picker: Any = None) -> pl.DataFrame:
    selected = _selected_chaser_index(chaser_picker)
    if selected is None or frame.is_empty() or "chaser_index" not in frame.columns:
        return frame
    return frame.filter(pl.col("chaser_index") == int(selected))


def _epoch_plot_values(frame: pl.DataFrame, *columns: str) -> list[Any]:
    for column in columns:
        if column in frame.columns:
            return frame[column].to_list()
    return [np.nan] * frame.height


def _has_finite_column(frame: pl.DataFrame, column: str) -> bool:
    if column not in frame.columns:
        return False
    values = np.asarray(frame[column].to_list(), dtype=np.float64)
    return bool(np.any(np.isfinite(values)))


def _build_epoch_bout_distribution_plots(go: Any, bout_rows: pl.DataFrame) -> list[Any]:
    if go is None or bout_rows.is_empty() or "window_label" not in bout_rows.columns:
        return []
    metrics = [
        ("bout_duration_s", "Swim Bout Duration Distribution", "Duration (s)"),
        ("bout_path_length_mm", "Swim Bout Distance Distribution", "Distance (mm)"),
        ("bout_net_heading_change_deg", "Swim Bout Net Heading Change Distribution", "Signed degrees"),
        ("abs_bout_net_heading_change_deg", "Swim Bout Absolute Net Heading Change Distribution", "Degrees"),
        ("bout_heading_path_deg", "Swim Bout Heading Path Distribution", "Degrees"),
    ]
    labels = bout_rows.select("window_label").unique(maintain_order=True)["window_label"].to_list()
    figures: list[Any] = []
    for metric, title, xaxis_title in metrics:
        if not _has_finite_column(bout_rows, metric):
            continue
        fig = go.Figure()
        for label in labels:
            rows = bout_rows.filter(pl.col("window_label") == label)
            if rows.is_empty():
                continue
            values = np.asarray(rows[metric].to_list(), dtype=np.float64)
            values = values[np.isfinite(values)]
            if values.size == 0:
                continue
            fig.add_trace(
                go.Histogram(
                    x=values,
                    name=str(label),
                    opacity=0.62,
                    nbinsx=40,
                    hovertemplate=(
                        "Epoch=%{fullData.name}<br>"
                        f"{xaxis_title}=%{{x:.3g}}<br>"
                        "Bouts=%{y}<extra></extra>"
                    ),
                )
            )
        if not fig.data:
            continue
        fig.update_layout(
            title=title,
            xaxis_title=xaxis_title,
            yaxis_title="Bout count",
            barmode="overlay",
            bargap=0.05,
            margin=dict(l=50, r=20, t=50, b=50),
        )
        figures.append(fig)
    return figures


_BOUT_HISTOGRAM_TITLES = {
    "bout_duration_s": ("Swim Bout Duration Distribution", "Duration (s)"),
    "bout_path_length_mm": ("Swim Bout Distance Distribution", "Distance (mm)"),
    "bout_net_heading_change_deg": ("Swim Bout Net Heading Change Distribution", "Signed degrees"),
    "abs_bout_net_heading_change_deg": ("Swim Bout Absolute Net Heading Change Distribution", "Degrees"),
    "bout_heading_path_deg": ("Swim Bout Heading Path Distribution", "Degrees"),
}


def _build_persisted_epoch_distribution_plots(
    go: Any,
    hist_rows: pl.DataFrame,
    *,
    metric_titles: Mapping[str, tuple[str, str]],
) -> list[Any]:
    required = {
        "metric_name",
        "window_label",
        "bin_center",
        "bin_width",
        "hist_count",
        "hist_fraction",
        "hist_density",
        "finite_sample_count",
    }
    if go is None or hist_rows.is_empty() or not required.issubset(set(hist_rows.columns)):
        return []
    figures: list[Any] = []
    metrics = hist_rows.select("metric_name").unique(maintain_order=True)["metric_name"].to_list()
    for metric in metrics:
        metric_key = str(metric)
        title, xaxis_title = metric_titles.get(metric_key, (metric_key, "Value"))
        rows_for_metric = hist_rows.filter(pl.col("metric_name") == metric_key)
        if rows_for_metric.is_empty():
            continue
        labels = rows_for_metric.select("window_label").unique(maintain_order=True)["window_label"].to_list()
        fig = go.Figure()
        for label in labels:
            rows = rows_for_metric.filter(pl.col("window_label") == label).sort("bin_center")
            if rows.is_empty():
                continue
            fig.add_trace(
                go.Bar(
                    x=rows["bin_center"].to_numpy(),
                    y=rows["hist_count"].to_numpy(),
                    width=(rows["bin_width"].to_numpy() * 0.92),
                    name=str(label),
                    opacity=0.62,
                    customdata=np.column_stack(
                        [
                            rows["bin_left"].to_numpy() if "bin_left" in rows.columns else rows["bin_center"].to_numpy(),
                            rows["bin_right"].to_numpy() if "bin_right" in rows.columns else rows["bin_center"].to_numpy(),
                            rows["hist_fraction"].to_numpy(),
                            rows["hist_density"].to_numpy(),
                            rows["finite_sample_count"].to_numpy(),
                        ]
                    ),
                    hovertemplate=(
                        "Epoch=%{fullData.name}<br>"
                        "Bin=%{customdata[0]:.3g} to %{customdata[1]:.3g}<br>"
                        "Count=%{y}<br>"
                        "Fraction=%{customdata[2]:.3g}<br>"
                        "Density=%{customdata[3]:.3g}<br>"
                        "Finite samples=%{customdata[4]}"
                        "<extra></extra>"
                    ),
                )
            )
        if not fig.data:
            continue
        fig.update_layout(
            title=title,
            xaxis_title=xaxis_title,
            yaxis_title="Count",
            barmode="overlay",
            bargap=0.05,
            margin=dict(l=50, r=20, t=50, b=50),
        )
        figures.append(fig)
    return figures


def build_epoch_summary_output(
    mo: Any,
    go: Any = None,
    *,
    loaded: GoodCopBadCopLoadedView,
    chaser_picker: Any = None,
) -> Any:
    frame = loaded.epoch_summary_df
    if frame.is_empty():
        detail = f" `{loaded.epoch_summary_error}`" if loaded.epoch_summary_error else ""
        return mo.md(f"No epoch kinematics summary is available for this recording.{detail}")

    visible = _filter_selected_chaser(frame, chaser_picker).sort(["window_id", "chaser_index"])
    if visible.is_empty():
        return mo.md("No epoch kinematics summary rows match the selected chaser.")

    if loaded.epoch_behavior is not None and not loaded.epoch_behavior.per_epoch_fish_df.is_empty():
        by_epoch = loaded.epoch_behavior.per_epoch_fish_df.sort("window_id")
    else:
        by_epoch = visible.unique(subset=["window_id"], keep="first").sort("window_id")
    total_bouts = by_epoch.select(pl.col("bout_count").sum()).item() if "bout_count" in by_epoch.columns else 0
    mean_speed = (
        by_epoch.select(pl.col("mean_speed_mm_s").mean()).item()
        if "mean_speed_mm_s" in by_epoch.columns
        else np.nan
    )
    mean_ibi = (
        by_epoch.select(pl.col("mean_inter_bout_interval_s").mean()).item()
        if "mean_inter_bout_interval_s" in by_epoch.columns
        else np.nan
    )
    source = loaded.epoch_summary_source or "none"
    source_label = "computed in viewer fallback" if loaded.epoch_summary_computed_in_viewer else "persisted zarr component"
    warning = (
        "\n\n**Warning:** this summary is being computed in the viewer because "
        "`epoch_behavior_summary` has not been backfilled for this recording."
        if loaded.epoch_summary_computed_in_viewer
        else ""
    )
    plots: list[Any] = []
    if go is not None and not by_epoch.is_empty() and "window_label" in by_epoch.columns:
        x_values = by_epoch["window_label"].to_list()
        duration_values = _epoch_plot_values(by_epoch, "window_duration_s", "duration_s")
        dropout_values = _epoch_plot_values(by_epoch, "tracking_dropout_fraction")
        if "bout_rate_per_min" in by_epoch.columns:
            bout_rate_fig = go.Figure(
                data=[
                    go.Bar(
                        x=x_values,
                        y=by_epoch["bout_rate_per_min"].to_list(),
                        customdata=np.column_stack(
                            [
                                _epoch_plot_values(by_epoch, "bout_count"),
                                duration_values,
                                dropout_values,
                            ]
                        ),
                        hovertemplate=(
                            "Epoch=%{x}<br>"
                            "Bout rate=%{y:.3g} / min<br>"
                            "Bout count=%{customdata[0]}<br>"
                            "Duration=%{customdata[1]:.3g} s<br>"
                            "Tracking dropout=%{customdata[2]:.3g}"
                            "<extra></extra>"
                        ),
                        marker_color="#7c3aed",
                        name="Bout rate",
                    )
                ]
            )
            bout_rate_fig.update_layout(
                title="Bout Rate by Epoch",
                xaxis_title="Epoch",
                yaxis_title="Bouts / min",
                margin=dict(l=50, r=20, t=50, b=50),
            )
            plots.append(bout_rate_fig)
        if "bout_count" in by_epoch.columns:
            bout_count_fig = go.Figure(
                data=[
                    go.Bar(
                        x=x_values,
                        y=by_epoch["bout_count"].to_list(),
                        customdata=np.column_stack(
                            [
                                _epoch_plot_values(by_epoch, "bout_rate_per_min"),
                                duration_values,
                                dropout_values,
                            ]
                        ),
                        hovertemplate=(
                            "Epoch=%{x}<br>"
                            "Bout count=%{y}<br>"
                            "Bout rate=%{customdata[0]:.3g} / min<br>"
                            "Duration=%{customdata[1]:.3g} s<br>"
                            "Tracking dropout=%{customdata[2]:.3g}"
                            "<extra></extra>"
                        ),
                        marker_color="#a78bfa",
                        name="Bout count",
                    )
                ]
            )
            bout_count_fig.update_layout(
                title="Bout Count by Epoch",
                xaxis_title="Epoch",
                yaxis_title="Bouts",
                margin=dict(l=50, r=20, t=50, b=50),
            )
            plots.append(bout_count_fig)
        if "inter_bout_interval_count" in by_epoch.columns:
            count_fig = go.Figure(
                data=[
                    go.Bar(
                        x=x_values,
                        y=by_epoch["inter_bout_interval_count"].to_list(),
                        customdata=np.column_stack(
                            [
                                _epoch_plot_values(by_epoch, "inter_bout_interval_rate_per_min"),
                                duration_values,
                                dropout_values,
                            ]
                        ),
                        hovertemplate=(
                            "Epoch=%{x}<br>"
                            "IBI count=%{y}<br>"
                            "IBI rate=%{customdata[0]:.3g} / min<br>"
                            "Duration=%{customdata[1]:.3g} s<br>"
                            "Tracking dropout=%{customdata[2]:.3g}"
                            "<extra></extra>"
                        ),
                        marker_color="#2563eb",
                        name="IBI count",
                    )
                ]
            )
            count_fig.update_layout(
                title="Inter-Bout Interval Count by Epoch",
                xaxis_title="Epoch",
                yaxis_title="Intervals",
                margin=dict(l=50, r=20, t=50, b=50),
            )
            plots.append(count_fig)
        ibi_metric = (
            "mean_inter_bout_interval_s"
            if "mean_inter_bout_interval_s" in by_epoch.columns
            else "median_inter_bout_interval_s"
        )
        if ibi_metric in by_epoch.columns:
            ibi_fig = go.Figure(
                data=[
                    go.Bar(
                        x=x_values,
                        y=by_epoch[ibi_metric].to_list(),
                        customdata=np.column_stack(
                            [
                                _epoch_plot_values(by_epoch, "inter_bout_interval_count"),
                                duration_values,
                                dropout_values,
                            ]
                        ),
                        hovertemplate=(
                            "Epoch=%{x}<br>"
                            "IBI=%{y:.3g} s<br>"
                            "IBI count=%{customdata[0]}<br>"
                            "Duration=%{customdata[1]:.3g} s<br>"
                            "Tracking dropout=%{customdata[2]:.3g}"
                            "<extra></extra>"
                        ),
                        marker_color="#16a34a",
                        name="Mean IBI" if ibi_metric.startswith("mean") else "Median IBI",
                    )
                ]
            )
            ibi_fig.update_layout(
                title=(
                    "Mean Inter-Bout Interval by Epoch"
                    if ibi_metric.startswith("mean")
                    else "Median Inter-Bout Interval by Epoch"
                ),
                xaxis_title="Epoch",
                yaxis_title="Interval (s)",
                margin=dict(l=50, r=20, t=50, b=50),
            )
            plots.append(ibi_fig)
        if _has_finite_column(by_epoch, "mean_bout_duration_s"):
            duration_fig = go.Figure(
                data=[
                    go.Bar(
                        x=x_values,
                        y=by_epoch["mean_bout_duration_s"].to_list(),
                        customdata=np.column_stack(
                            [
                                _epoch_plot_values(by_epoch, "bout_count"),
                                duration_values,
                                dropout_values,
                            ]
                        ),
                        hovertemplate=(
                            "Epoch=%{x}<br>"
                            "Mean bout duration=%{y:.3g} s<br>"
                            "Bout count=%{customdata[0]}<br>"
                            "Epoch duration=%{customdata[1]:.3g} s<br>"
                            "Tracking dropout=%{customdata[2]:.3g}"
                            "<extra></extra>"
                        ),
                        marker_color="#0f766e",
                        name="Mean bout duration",
                    )
                ]
            )
            duration_fig.update_layout(
                title="Mean Bout Duration by Epoch",
                xaxis_title="Epoch",
                yaxis_title="Duration (s)",
                margin=dict(l=50, r=20, t=50, b=50),
            )
            plots.append(duration_fig)
        if _has_finite_column(by_epoch, "mean_bout_path_length_mm"):
            distance_fig = go.Figure(
                data=[
                    go.Bar(
                        x=x_values,
                        y=by_epoch["mean_bout_path_length_mm"].to_list(),
                        customdata=np.column_stack(
                            [
                                _epoch_plot_values(by_epoch, "bout_count"),
                                duration_values,
                                dropout_values,
                            ]
                        ),
                        hovertemplate=(
                            "Epoch=%{x}<br>"
                            "Mean bout distance=%{y:.3g} mm<br>"
                            "Bout count=%{customdata[0]}<br>"
                            "Epoch duration=%{customdata[1]:.3g} s<br>"
                            "Tracking dropout=%{customdata[2]:.3g}"
                            "<extra></extra>"
                        ),
                        marker_color="#ea580c",
                        name="Mean bout distance",
                    )
                ]
            )
            distance_fig.update_layout(
                title="Mean Bout Distance by Epoch",
                xaxis_title="Epoch",
                yaxis_title="Distance (mm)",
                margin=dict(l=50, r=20, t=50, b=50),
            )
            plots.append(distance_fig)
        if _has_finite_column(by_epoch, "mean_bout_net_heading_change_deg"):
            signed_heading_fig = go.Figure(
                data=[
                    go.Bar(
                        x=x_values,
                        y=by_epoch["mean_bout_net_heading_change_deg"].to_list(),
                        customdata=np.column_stack(
                            [
                                _epoch_plot_values(by_epoch, "bout_heading_sample_count"),
                                _epoch_plot_values(by_epoch, "median_bout_net_heading_change_deg"),
                                duration_values,
                            ]
                        ),
                        hovertemplate=(
                            "Epoch=%{x}<br>"
                            "Mean net heading change=%{y:.3g} deg<br>"
                            "Heading bouts=%{customdata[0]}<br>"
                            "Median net heading change=%{customdata[1]:.3g} deg<br>"
                            "Epoch duration=%{customdata[2]:.3g} s"
                            "<extra></extra>"
                        ),
                        marker_color="#be123c",
                        name="Mean net heading change",
                    )
                ]
            )
            signed_heading_fig.update_layout(
                title="Mean Net Bout Heading Change by Epoch",
                xaxis_title="Epoch",
                yaxis_title="Signed degrees",
                yaxis=dict(zeroline=True, zerolinecolor="#64748b", zerolinewidth=1),
                margin=dict(l=50, r=20, t=50, b=50),
            )
            plots.append(signed_heading_fig)
        if _has_finite_column(by_epoch, "mean_abs_bout_net_heading_change_deg"):
            heading_change_fig = go.Figure(
                data=[
                    go.Bar(
                        x=x_values,
                        y=by_epoch["mean_abs_bout_net_heading_change_deg"].to_list(),
                        customdata=np.column_stack(
                            [
                                _epoch_plot_values(by_epoch, "bout_heading_sample_count"),
                                _epoch_plot_values(by_epoch, "mean_bout_heading_path_deg"),
                                duration_values,
                            ]
                        ),
                        hovertemplate=(
                            "Epoch=%{x}<br>"
                            "Mean abs net heading change=%{y:.3g} deg<br>"
                            "Heading bouts=%{customdata[0]}<br>"
                            "Mean heading path=%{customdata[1]:.3g} deg<br>"
                            "Epoch duration=%{customdata[2]:.3g} s"
                            "<extra></extra>"
                        ),
                        marker_color="#db2777",
                        name="Mean abs heading change",
                    )
                ]
            )
            heading_change_fig.update_layout(
                title="Mean Absolute Net Bout Heading Change by Epoch",
                xaxis_title="Epoch",
                yaxis_title="Degrees",
                margin=dict(l=50, r=20, t=50, b=50),
            )
            plots.append(heading_change_fig)
        if _has_finite_column(by_epoch, "wall_fraction"):
            wall_fig = go.Figure(
                data=[
                    go.Bar(
                        x=x_values,
                        y=by_epoch["wall_fraction"].to_list(),
                        customdata=np.column_stack(
                            [
                                _epoch_plot_values(by_epoch, "wall_frame_count"),
                                _epoch_plot_values(by_epoch, "wall_band_mm"),
                                _epoch_plot_values(by_epoch, "arena_radius_mm"),
                            ]
                        ),
                        hovertemplate=(
                            "Epoch=%{x}<br>"
                            "Wall fraction=%{y:.3g}<br>"
                            "Wall frames=%{customdata[0]}<br>"
                            "Wall band=%{customdata[1]:.3g} mm<br>"
                            "Arena radius=%{customdata[2]:.3g} mm"
                            "<extra></extra>"
                        ),
                        marker_color="#475569",
                        name="Wall fraction",
                    )
                ]
            )
            wall_fig.update_layout(
                title="Wall Fraction by Epoch",
                xaxis_title="Epoch",
                yaxis_title="Fraction",
                margin=dict(l=50, r=20, t=50, b=50),
            )
            plots.append(wall_fig)
        center_hist = (
            loaded.epoch_behavior.center_distance_histogram_df
            if loaded.epoch_behavior is not None
            else pl.DataFrame()
        )
        if not center_hist.is_empty() and {"window_label", "bin_center_mm", "hist_fraction"}.issubset(center_hist.columns):
            center_fig = go.Figure()
            for label in x_values:
                rows = center_hist.filter(pl.col("window_label") == label).sort("bin_center_mm")
                if rows.is_empty():
                    continue
                center_fig.add_scatter(
                    x=rows["bin_center_mm"].to_numpy(),
                    y=rows["hist_fraction"].to_numpy(),
                    mode="lines+markers",
                    name=str(label),
                    hovertemplate=(
                        "Epoch=%{fullData.name}<br>"
                        "Distance from center=%{x:.3g} mm<br>"
                        "Fraction=%{y:.3g}"
                        "<extra></extra>"
                    ),
                )
            if center_fig.data:
                center_fig.update_layout(
                    title="Fish Distance From Arena Center",
                    xaxis_title="Distance from center (mm)",
                    yaxis_title="Fraction of valid frames",
                    margin=dict(l=50, r=20, t=50, b=50),
                )
                plots.append(center_fig)

    display_columns = [
        "window_label",
        "chaser_index",
        "window_duration_s",
        "tracking_dropout_fraction",
        "center_distance_sample_count",
        "mean_distance_from_arena_center_mm",
        "median_distance_from_arena_center_mm",
        "p95_distance_from_arena_center_mm",
        "arena_radius_mm",
        "wall_band_mm",
        "wall_fraction",
        "wall_time_s",
        "median_distance_mm",
        "mean_distance_mm",
        "p05_distance_mm",
        "p95_distance_mm",
        "mean_speed_mm_s",
        "median_speed_mm_s",
        "p95_speed_mm_s",
        "bout_count",
        "bout_rate_per_min",
        "median_bout_duration_s",
        "mean_bout_duration_s",
        "median_bout_path_length_mm",
        "mean_bout_path_length_mm",
        "bout_heading_sample_count",
        "mean_bout_net_heading_change_deg",
        "median_bout_net_heading_change_deg",
        "mean_abs_bout_net_heading_change_deg",
        "median_abs_bout_net_heading_change_deg",
        "mean_bout_heading_path_deg",
        "median_bout_heading_path_deg",
        "mean_inter_bout_interval_s",
        "median_inter_bout_interval_s",
        "inter_bout_interval_count",
        "inter_bout_interval_rate_per_min",
    ]
    display = visible.select([column for column in display_columns if column in visible.columns])
    display_pd = pd.DataFrame(display.to_dicts())
    if len(display_pd):
        numeric_cols = display_pd.select_dtypes(include=[np.number]).columns
        display_pd[numeric_cols] = display_pd[numeric_cols].round(4)

    items: list[Any] = [
        mo.md(f"## Epoch Kinematics Summary\n\n`{source}`\n\nSource: **{source_label}**{warning}"),
        mo.hstack(
            [
                mo.stat(label="Epochs", value=f"{by_epoch.height:,}"),
                mo.stat(label="Rows", value=f"{visible.height:,}"),
                mo.stat(label="Bouts", value=f"{int(total_bouts):,}"),
                mo.stat(label="Mean speed", value=_metric_text(mean_speed, digits=2, suffix=" mm/s")),
                mo.stat(label="Mean IBI", value=_metric_text(mean_ibi, digits=3, suffix=" s")),
            ]
        ),
    ]
    if plots:
        items.append(mo.vstack(plots))
    if loaded.epoch_behavior is not None:
        distribution_plots = _build_persisted_epoch_distribution_plots(
            go,
            loaded.epoch_behavior.per_epoch_bout_histograms_df,
            metric_titles=_BOUT_HISTOGRAM_TITLES,
        )
        if not distribution_plots:
            distribution_plots = _build_epoch_bout_distribution_plots(
                go,
                loaded.epoch_behavior.per_epoch_bouts_df,
            )
        if distribution_plots:
            items.append(mo.md("## Swim Bout Distributions"))
            items.append(mo.vstack(distribution_plots))
        ibi_distribution_plots = _build_persisted_epoch_distribution_plots(
            go,
            loaded.epoch_behavior.per_epoch_inter_bout_interval_histograms_df,
            metric_titles={"inter_bout_interval_s": ("Inter-Bout Interval Distribution", "Interval (s)")},
        )
        if ibi_distribution_plots:
            items.append(mo.md("## Inter-Bout Interval Distributions"))
            items.append(mo.vstack(ibi_distribution_plots))
    items.append(mo.ui.table(display_pd, selection=None, page_size=12))
    return mo.vstack(items)


def _chaser_color(loaded: GoodCopBadCopLoadedView, chaser_index: int) -> str:
    return loaded.data.chaser_color_hex.get(
        int(chaser_index),
        CHASER_MARKER_COLORS[int(chaser_index) % len(CHASER_MARKER_COLORS)],
    )


def build_egocentric_bearing_output(
    mo: Any,
    go: Any,
    *,
    loaded: GoodCopBadCopLoadedView,
    window: GoodCopBadCopTimeWindow,
    chaser_picker: Any = None,
    max_points: int = 16000,
) -> Any:
    if loaded.data.egocentric_component_name is None or loaded.egocentric_bearing_df.is_empty():
        return mo.md("No egocentric chaser-bearing component is linked to this chaser-distance run.")

    visible = _filter_selected_chaser(
        _filter_egocentric_window(loaded.egocentric_bearing_df, window),
        chaser_picker,
    )
    if visible.is_empty():
        return mo.md("No valid egocentric chaser-bearing samples in the selected window.")
    if visible.height > int(max_points):
        step = max(1, int(np.ceil(visible.height / float(max_points))))
        visible = visible.with_row_index("_row").filter((pl.col("_row") % step) == 0).drop("_row")

    fig = go.Figure()
    for chaser_index in visible["chaser_index"].unique().sort().to_list():
        rows = visible.filter(pl.col("chaser_index") == int(chaser_index))
        if rows.is_empty():
            continue
        fig.add_trace(
            go.Scatterpolar(
                theta=rows["bearing_deg"].to_numpy(),
                r=rows["distance_mm"].to_numpy(),
                mode="markers",
                name=f"chaser {int(chaser_index)}",
                marker=dict(
                    size=4,
                    opacity=0.32,
                    color=_chaser_color(loaded, int(chaser_index)),
                ),
                customdata=np.column_stack(
                    [
                        rows["time_s"].to_numpy(),
                        rows["frame_index"].to_numpy(),
                        rows["alignment_cos"].to_numpy(),
                    ]
                ),
                hovertemplate=(
                    "bearing=%{theta:.1f} deg<br>"
                    "distance=%{r:.2f} mm<br>"
                    "time=%{customdata[0]:.2f}s<br>"
                    "frame=%{customdata[1]}<br>"
                    "alignment=%{customdata[2]:.3f}"
                    "<extra>%{fullData.name}</extra>"
                ),
            )
        )
    fig.update_layout(
        title=f"Egocentric Chaser Bearing ({window.selected_epoch_label})",
        height=560,
        margin=dict(l=40, r=40, t=58, b=48),
        polar=dict(
            radialaxis=dict(title="Distance (mm)", rangemode="tozero"),
            angularaxis=dict(
                rotation=90,
                direction="counterclockwise",
                tickmode="array",
                tickvals=[-180, -90, 0, 90, 180],
                ticktext=["behind", "right", "front", "left", "behind"],
            ),
        ),
        legend=dict(orientation="h", yanchor="top", y=-0.08, xanchor="left", x=0.0),
    )
    return fig


def _egocentric_distance_bin_width_mm(loaded: GoodCopBadCopLoadedView) -> float:
    edges = loaded.data.egocentric_distance_bin_edges_mm
    if edges is not None and edges.shape[0] > 1:
        width = float(edges[1] - edges[0])
        if np.isfinite(width) and width > 0:
            return width
    return 2.0


def _egocentric_bearing_bin_width_deg(loaded: GoodCopBadCopLoadedView) -> float:
    edges = loaded.data.egocentric_bearing_bin_edges_deg
    if edges is not None and edges.shape[0] > 1:
        width = float(edges[1] - edges[0])
        if np.isfinite(width) and width > 0:
            return width
    return 15.0


def _egocentric_display_distance_bin_width_mm(loaded: GoodCopBadCopLoadedView) -> float:
    return max(
        _egocentric_distance_bin_width_mm(loaded),
        float(EGOCENTRIC_POLAR_DISPLAY_MIN_DISTANCE_BIN_WIDTH_MM),
    )


def _egocentric_display_bearing_bin_width_deg(loaded: GoodCopBadCopLoadedView) -> float:
    return min(
        180.0,
        max(
            _egocentric_bearing_bin_width_deg(loaded),
            float(EGOCENTRIC_POLAR_DISPLAY_MIN_BEARING_BIN_WIDTH_DEG),
        ),
    )


def _positive_quantile(values: np.ndarray, quantile: float) -> float:
    finite_positive = np.asarray(values, dtype=np.float64)
    finite_positive = finite_positive[np.isfinite(finite_positive) & (finite_positive > 0.0)]
    if finite_positive.size == 0:
        return 1.0
    return max(float(np.quantile(finite_positive, float(quantile))), float(np.finfo(np.float64).eps))


def _egocentric_polar_density_frame(
    visible: pl.DataFrame,
    *,
    distance_bin_width_mm: float,
    bearing_bin_width_deg: float,
) -> pl.DataFrame:
    if visible.is_empty():
        return pl.DataFrame(
            schema={
                "chaser_index": pl.Int32,
                "distance_bin_start_mm": pl.Float64,
                "distance_bin_center_mm": pl.Float64,
                "bearing_bin_start_deg": pl.Float64,
                "bearing_bin_center_deg": pl.Float64,
                "n": pl.UInt32,
                "probability": pl.Float64,
            }
        )

    distance_width = float(distance_bin_width_mm)
    bearing_width = float(bearing_bin_width_deg)
    bearing_bin_count = max(1, int(np.ceil(360.0 / bearing_width)))
    return (
        visible.filter(
            (pl.col("distance_mm") >= 0.0)
            & pl.col("distance_mm").is_finite()
            & pl.col("bearing_deg").is_finite()
        )
        .with_columns(
            (pl.col("distance_mm") / distance_width).floor().cast(pl.Int64).alias("_distance_bin"),
            ((pl.col("bearing_deg") + 180.0) / bearing_width)
            .floor()
            .cast(pl.Int64)
            .clip(0, bearing_bin_count - 1)
            .alias("_bearing_bin"),
        )
        .with_columns(
            (pl.col("_distance_bin").cast(pl.Float64) * distance_width).alias("distance_bin_start_mm"),
            ((pl.col("_distance_bin").cast(pl.Float64) + 0.5) * distance_width).alias("distance_bin_center_mm"),
            (-180.0 + pl.col("_bearing_bin").cast(pl.Float64) * bearing_width).alias("bearing_bin_start_deg"),
            (-180.0 + (pl.col("_bearing_bin").cast(pl.Float64) + 0.5) * bearing_width).alias(
                "bearing_bin_center_deg"
            ),
        )
        .group_by(
            [
                "chaser_index",
                "distance_bin_start_mm",
                "distance_bin_center_mm",
                "bearing_bin_start_deg",
                "bearing_bin_center_deg",
            ]
        )
        .agg(pl.len().cast(pl.UInt32).alias("n"))
        .with_columns((pl.col("n") / pl.col("n").sum().over("chaser_index")).alias("probability"))
        .sort(["chaser_index", "distance_bin_start_mm", "bearing_bin_start_deg"])
    )


def build_egocentric_polar_heatmap_output(
    mo: Any,
    go: Any,
    *,
    loaded: GoodCopBadCopLoadedView,
    window: GoodCopBadCopTimeWindow,
    chaser_picker: Any = None,
) -> Any:
    if loaded.data.egocentric_component_name is None or loaded.egocentric_bearing_df.is_empty():
        return mo.md("No egocentric chaser-bearing component is linked to this chaser-distance run.")

    visible = _filter_selected_chaser(
        _filter_egocentric_window(loaded.egocentric_bearing_df, window),
        chaser_picker,
    )
    if visible.is_empty():
        return mo.md("No valid egocentric chaser-bearing samples in the selected window.")

    distance_width = _egocentric_display_distance_bin_width_mm(loaded)
    bearing_width = _egocentric_display_bearing_bin_width_deg(loaded)
    density = _egocentric_polar_density_frame(
        visible,
        distance_bin_width_mm=distance_width,
        bearing_bin_width_deg=bearing_width,
    )
    if density.is_empty():
        return mo.md("No egocentric samples could be binned for the selected window.")

    max_probability = _positive_quantile(
        density["probability"].to_numpy(),
        EGOCENTRIC_POLAR_DISPLAY_COLOR_CMAX_QUANTILE,
    )
    if max_probability is None or not np.isfinite(float(max_probability)) or float(max_probability) <= 0:
        max_probability = 1.0

    figures = []
    for chaser_index in density["chaser_index"].unique().sort().to_list():
        rows = density.filter(pl.col("chaser_index") == int(chaser_index))
        if rows.is_empty():
            continue
        fig = go.Figure()
        fig.add_trace(
            go.Barpolar(
                theta=rows["bearing_bin_center_deg"].to_numpy(),
                r=np.full(rows.height, distance_width, dtype=np.float64),
                base=rows["distance_bin_start_mm"].to_numpy(),
                width=np.full(rows.height, bearing_width, dtype=np.float64),
                marker=dict(
                    color=rows["probability"].to_numpy(),
                    colorscale="Viridis",
                    cmin=0.0,
                    cmax=float(max_probability),
                    colorbar=dict(title="fraction/bin"),
                    line=dict(width=0),
                ),
                opacity=0.95,
                name=f"chaser {int(chaser_index)}",
                customdata=np.column_stack(
                    [
                        rows["n"].to_numpy(),
                        rows["probability"].to_numpy(),
                        rows["distance_bin_start_mm"].to_numpy(),
                        (rows["distance_bin_start_mm"] + distance_width).to_numpy(),
                        rows["bearing_bin_start_deg"].to_numpy(),
                        (rows["bearing_bin_start_deg"] + bearing_width).to_numpy(),
                    ]
                ),
                hovertemplate=(
                    "bearing bin=%{customdata[4]:.1f} to %{customdata[5]:.1f} deg<br>"
                    "distance bin=%{customdata[2]:.2f} to %{customdata[3]:.2f} mm<br>"
                    "samples=%{customdata[0]:,}<br>"
                    "fraction=%{customdata[1]:.4f}"
                    "<extra>%{fullData.name}</extra>"
                ),
            )
        )
        fig.update_layout(
            title=(
                f"Egocentric Bearing Density ({window.selected_epoch_label}, chaser {int(chaser_index)}; "
                f"{distance_width:g} mm x {bearing_width:g} deg display bins)"
            ),
            height=520,
            margin=dict(l=40, r=72, t=58, b=48),
            showlegend=False,
            polar=dict(
                radialaxis=dict(title="Distance (mm)", rangemode="tozero"),
                angularaxis=dict(
                    rotation=90,
                    direction="counterclockwise",
                    tickmode="array",
                    tickvals=[-180, -90, 0, 90, 180],
                    ticktext=["behind", "right", "front", "left", "behind"],
                ),
            ),
        )
        figures.append(fig)

    if not figures:
        return mo.md("No egocentric samples could be binned for the selected window.")
    if len(figures) == 1:
        return figures[0]
    return mo.vstack(figures)


def build_egocentric_static_polar_output(
    mo: Any,
    *,
    loaded: GoodCopBadCopLoadedView,
) -> Any:
    if loaded.data.egocentric_component_name is None:
        return mo.md("No egocentric chaser-bearing component is linked to this chaser-distance run.")
    items = []
    if loaded.egocentric_pre_post_polar_point_cloud_png_bytes:
        path_text = (
            loaded.egocentric_pre_post_polar_point_cloud_png_path
            or EGOCENTRIC_PRE_POST_POLAR_POINT_CLOUD_PNG_ARTIFACT_NAME
        )
        items.extend(
            [
                mo.md(f"## Persisted Egocentric Bearing Point Clouds\n\n`{path_text}`"),
                png_bytes_to_markdown_image(
                    mo,
                    loaded.egocentric_pre_post_polar_point_cloud_png_bytes,
                    alt_text="Egocentric chaser bearing pre/post polar point-cloud snapshot",
                ),
            ]
        )
    if loaded.egocentric_pre_post_polar_png_bytes:
        path_text = loaded.egocentric_pre_post_polar_png_path or EGOCENTRIC_PRE_POST_POLAR_PNG_ARTIFACT_NAME
        items.extend(
            [
                mo.md(f"## Persisted Egocentric Bearing Circular Histograms\n\n`{path_text}`"),
                png_bytes_to_markdown_image(
                    mo,
                    loaded.egocentric_pre_post_polar_png_bytes,
                    alt_text="Egocentric chaser bearing pre/post polar histogram snapshot",
                ),
            ]
        )
    if items:
        return mo.vstack(items)
    errors = [
        error
        for error in (
            loaded.egocentric_pre_post_polar_point_cloud_png_error,
            loaded.egocentric_pre_post_polar_png_error,
        )
        if error
    ]
    if errors:
        return mo.vstack(
            [
                mo.md(
                    "Egocentric chaser-bearing component is present, but one or more persisted "
                    f"pre/post polar PNGs could not be loaded: `{'; '.join(errors)}`"
                )
            ]
        )
    return mo.md(
        "Egocentric chaser-bearing component is present, but no persisted pre/post polar PNG "
        f"was found at `{EGOCENTRIC_PRE_POST_POLAR_POINT_CLOUD_PNG_ARTIFACT_NAME}` or "
        f"`{EGOCENTRIC_PRE_POST_POLAR_PNG_ARTIFACT_NAME}`."
    )


def build_fish_heading_output(
    mo: Any,
    go: Any,
    *,
    loaded: GoodCopBadCopLoadedView,
    window: GoodCopBadCopTimeWindow,
    max_points: int = 20000,
) -> Any:
    if loaded.data.egocentric_component_name is None or loaded.egocentric_heading_df.is_empty():
        return mo.md("No fish-heading samples are linked to this chaser-distance run.")

    visible = _filter_egocentric_window(loaded.egocentric_heading_df, window)
    if visible.is_empty():
        return mo.md("No valid fish-heading samples in the selected window.")
    if visible.height > int(max_points):
        step = max(1, int(np.ceil(visible.height / float(max_points))))
        visible = visible.with_row_index("_row").filter((pl.col("_row") % step) == 0).drop("_row")

    fig = go.Figure()
    add_epoch_overlays(fig, loaded.windows_df)
    fig.add_trace(
        go.Scattergl(
            x=visible["time_s"].to_numpy(),
            y=visible["fish_heading_deg"].to_numpy(),
            mode="markers",
            name="fish heading",
            marker=dict(size=4, opacity=0.62, color="#0f766e"),
            customdata=np.column_stack(
                [
                    visible["frame_index"].to_numpy(),
                    visible["window_label"].to_numpy(),
                ]
            ),
            hovertemplate=(
                "time=%{x:.2f}s<br>"
                "heading=%{y:.1f} deg<br>"
                "frame=%{customdata[0]}<br>"
                "epoch=%{customdata[1]}"
                "<extra></extra>"
            ),
        )
    )
    apply_full_width_timeseries_layout(
        fig,
        title=f"Fish Heading ({window.selected_epoch_label})",
        yaxis_title="Fish heading (deg)",
        height=390,
    )
    fig.update_yaxes(range=[-185, 185], tickmode="array", tickvals=[-180, -90, 0, 90, 180])
    fig.update_xaxes(range=[float(window.start_s), float(window.stop_s)])
    return fig


def build_egocentric_alignment_output(
    mo: Any,
    go: Any,
    *,
    loaded: GoodCopBadCopLoadedView,
    window: GoodCopBadCopTimeWindow,
    chaser_picker: Any = None,
) -> Any:
    if loaded.data.egocentric_component_name is None or loaded.egocentric_bearing_df.is_empty():
        return mo.md("No egocentric chaser-bearing component is linked to this chaser-distance run.")

    visible = _filter_selected_chaser(
        _filter_egocentric_window(loaded.egocentric_bearing_df, window),
        chaser_picker,
    )
    if visible.is_empty():
        return mo.md("No valid egocentric alignment samples in the selected window.")

    if (
        loaded.data.egocentric_distance_bin_edges_mm is not None
        and loaded.data.egocentric_distance_bin_edges_mm.shape[0] > 1
    ):
        width = float(
            loaded.data.egocentric_distance_bin_edges_mm[1]
            - loaded.data.egocentric_distance_bin_edges_mm[0]
        )
    else:
        width = 2.0
    width = width if np.isfinite(width) and width > 0 else 2.0
    grouped = (
        visible.with_columns(
            ((pl.col("distance_mm") / width).floor() * width).alias("distance_bin_start_mm")
        )
        .group_by(["chaser_index", "distance_bin_start_mm"])
        .agg(
            pl.len().cast(pl.UInt32).alias("n"),
            pl.col("alignment_cos").mean().alias("mean_alignment_cos"),
            pl.col("bearing_deg").abs().mean().alias("mean_abs_bearing_deg"),
        )
        .with_columns((pl.col("distance_bin_start_mm") + (width / 2.0)).alias("distance_bin_center_mm"))
        .sort(["chaser_index", "distance_bin_start_mm"])
    )

    fig = go.Figure()
    for chaser_index in grouped["chaser_index"].unique().sort().to_list():
        rows = grouped.filter(pl.col("chaser_index") == int(chaser_index))
        if rows.is_empty():
            continue
        fig.add_trace(
            go.Scatter(
                x=rows["distance_bin_center_mm"].to_numpy(),
                y=rows["mean_alignment_cos"].to_numpy(),
                mode="lines+markers",
                name=f"chaser {int(chaser_index)}",
                marker=dict(color=_chaser_color(loaded, int(chaser_index))),
                line=dict(color=_chaser_color(loaded, int(chaser_index))),
                customdata=np.column_stack(
                    [
                        rows["n"].to_numpy(),
                        rows["mean_abs_bearing_deg"].to_numpy(),
                    ]
                ),
                hovertemplate=(
                    "distance bin=%{x:.2f} mm<br>"
                    "mean alignment=%{y:.3f}<br>"
                    "n=%{customdata[0]}<br>"
                    "mean |bearing|=%{customdata[1]:.1f} deg"
                    "<extra>%{fullData.name}</extra>"
                ),
            )
        )
    fig.add_hline(y=0.0, line_width=1, line_dash="dot", line_color="rgba(15,23,42,0.55)")
    fig.update_layout(
        title=f"Egocentric Alignment by Distance ({window.selected_epoch_label})",
        xaxis_title="Distance to chaser (mm)",
        yaxis_title="Mean cos(bearing)",
        yaxis=dict(range=[-1.05, 1.05]),
        height=430,
        margin=dict(l=56, r=20, t=58, b=70),
        legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="left", x=0.0),
    )
    return fig


def build_arena_heatmap(
    px: Any,
    *,
    loaded: GoodCopBadCopLoadedView,
    heatmap_bins: Any,
    chaser_overlay: Any,
    window: GoodCopBadCopTimeWindow,
) -> tuple[Any, pd.DataFrame]:
    visible_positions = loaded.position_df[
        (loaded.position_df["time_s"] >= window.start_s)
        & (loaded.position_df["time_s"] <= window.stop_s)
        & loaded.position_df["fish_valid"].astype(bool)
    ].copy()
    if not len(visible_positions):
        return "No valid fish positions in the selected window.", visible_positions
    arena_heatmap = px.density_heatmap(
        visible_positions,
        x="x",
        y="y",
        nbinsx=int(heatmap_bins.value),
        nbinsy=int(heatmap_bins.value),
        title=f"Arena Occupancy ({window.selected_epoch_label})",
        labels={"x": "Arena X (px)", "y": "Arena Y (px, down)"},
    )
    if chaser_overlay.value and len(loaded.chaser_position_df):
        visible_chasers = loaded.chaser_position_df[
            (loaded.chaser_position_df["time_s"] >= window.start_s)
            & (loaded.chaser_position_df["time_s"] <= window.stop_s)
            & loaded.chaser_position_df["chaser_valid"].astype(bool)
        ].copy()
        if len(visible_chasers):
            for chaser_index, rows in visible_chasers.groupby("chaser_index"):
                chaser_i = int(chaser_index)
                color = loaded.data.chaser_color_hex.get(
                    chaser_i,
                    CHASER_MARKER_COLORS[chaser_i % len(CHASER_MARKER_COLORS)],
                )
                arena_heatmap.add_scattergl(
                    x=rows["x"],
                    y=rows["y"],
                    mode="markers",
                    marker=dict(size=5, opacity=0.35, color=color),
                    name=f"chaser {chaser_i} trace",
                )
                finite = rows[np.isfinite(rows["x"]) & np.isfinite(rows["y"])]
                if len(finite):
                    marker_x = float(finite["x"].median())
                    marker_y = float(finite["y"].median())
                    arena_heatmap.add_scatter(
                        x=[marker_x],
                        y=[marker_y],
                        mode="markers+text",
                        marker=dict(
                            size=17,
                            symbol="diamond",
                            color=color,
                            line=dict(width=2, color="white"),
                        ),
                        text=[f"C{chaser_i}"],
                        textposition="top center",
                        textfont=dict(size=13, color=color),
                        name=f"chaser {chaser_i} position",
                        hovertemplate=(
                            f"chaser {chaser_i}<br>"
                            "x=%{x:.1f}<br>"
                            "y=%{y:.1f}<extra></extra>"
                        ),
                    )
    arena_heatmap.update_yaxes(scaleanchor="x", scaleratio=1, autorange="reversed")
    arena_heatmap.update_layout(height=590, margin=dict(l=52, r=20, t=58, b=64))
    return arena_heatmap, visible_positions


def build_detection_occupancy_output(
    mo: Any,
    go: Any,
    *,
    loaded: GoodCopBadCopLoadedView,
    window: GoodCopBadCopTimeWindow,
) -> Any:
    data = loaded.data
    if data.occupancy_normalized is None or data.occupancy_x_edges is None or data.occupancy_y_edges is None:
        return mo.md("No persisted detection-occupancy heatmap cube is linked to this spec.")
    if not len(loaded.windows_df):
        return mo.md("No epoch windows are available for persisted detection-occupancy heatmaps.")
    window_idx = 0
    if window.selected_epoch_id is not None:
        matches = np.flatnonzero(
            loaded.windows_df["window_id"].to_numpy(dtype=int) == int(window.selected_epoch_id)
        )
        window_idx = int(matches[0]) if matches.size else 0
    window_idx = max(0, min(window_idx, int(data.occupancy_normalized.shape[0]) - 1))
    label = (
        str(loaded.windows_df.iloc[window_idx]["label"])
        if window_idx < len(loaded.windows_df)
        else f"window {window_idx}"
    )
    x_edges = np.asarray(data.occupancy_x_edges, dtype=float)
    y_edges = np.asarray(data.occupancy_y_edges, dtype=float)
    x_centers = (
        (x_edges[:-1] + x_edges[1:]) / 2.0
        if x_edges.size > 1
        else np.arange(data.occupancy_normalized.shape[2])
    )
    y_centers = (
        (y_edges[:-1] + y_edges[1:]) / 2.0
        if y_edges.size > 1
        else np.arange(data.occupancy_normalized.shape[1])
    )
    occupancy_fig = go.Figure(
        data=[
            go.Heatmap(
                z=np.asarray(data.occupancy_normalized[window_idx], dtype=float),
                x=x_centers,
                y=y_centers,
                colorscale="Viridis",
                colorbar=dict(title="normalized"),
            )
        ]
    )
    occupancy_fig.update_yaxes(scaleanchor="x", scaleratio=1, autorange="reversed")
    occupancy_fig.update_layout(
        title=f"Persisted Detection Occupancy ({label})",
        xaxis_title="Source image X (px)",
        yaxis_title="Source image Y (px, down)",
        height=520,
        margin=dict(l=52, r=20, t=58, b=64),
    )
    return occupancy_fig


def _is_chaser_zone_marker_epoch(label: object) -> bool:
    text = str(label).lower()
    return ("pre" in text or "post" in text) and "training" not in text


def _chaser_pattern_shape(chaser_indices: tuple[int, ...]) -> str:
    if not chaser_indices:
        return ""
    if len(chaser_indices) > 1:
        return "x"
    return CHASER_ZONE_PATTERN_SHAPES[int(chaser_indices[0]) % len(CHASER_ZONE_PATTERN_SHAPES)]


def _format_chaser_zone_presence(chaser_indices: tuple[int, ...]) -> str:
    if not chaser_indices:
        return "none"
    return ", ".join(f"chaser {int(index)}" for index in chaser_indices)


def _point_in_zone(row: pd.Series, point_xy: np.ndarray) -> bool:
    x_min = float(row["x_min"])
    y_min = float(row["y_min"])
    x_max = float(row["x_max"])
    y_max = float(row["y_max"])
    if not all(np.isfinite(value) for value in (x_min, y_min, x_max, y_max)):
        return False
    x = float(point_xy[0])
    y = float(point_xy[1])
    return x_min <= x < x_max and y_min <= y < y_max


def _chaser_zone_presence_by_epoch(
    *,
    loaded: GoodCopBadCopLoadedView,
    zone_df: pd.DataFrame,
) -> dict[tuple[int, str], tuple[int, ...]]:
    data = loaded.data
    if data.chaser_source_img_xy is None or not len(zone_df):
        return {}
    if not (zone_df["coordinate_frame"].astype(str) == "source_image_px").all():
        return {}

    positions = np.asarray(data.chaser_source_img_xy, dtype=float)
    n = min(data.time_seconds.shape[0], positions.shape[0])
    if n == 0:
        return {}

    time_s = data.time_seconds[:n]
    zones_by_window = {
        int(window_id): rows.sort_values(["display_order", "zone_label"]).copy()
        for window_id, rows in zone_df.groupby("window_id", sort=False)
    }
    out: dict[tuple[int, str], set[int]] = {}
    for window_row in loaded.windows_df.to_dict("records"):
        window_id = int(window_row["window_id"])
        if window_id not in zones_by_window:
            continue
        if not _is_chaser_zone_marker_epoch(window_row["label"]):
            continue

        frame_mask = (time_s >= float(window_row["start_time_s"])) & (time_s <= float(window_row["end_time_s"]))
        if not np.any(frame_mask):
            continue

        for col_idx, chaser_index in enumerate(data.chaser_indices.tolist()):
            if col_idx >= positions.shape[1]:
                continue
            valid = frame_mask.copy()
            if data.chaser_valid is not None and data.chaser_valid.shape[0] >= n and col_idx < data.chaser_valid.shape[1]:
                valid &= data.chaser_valid[:n, col_idx].astype(bool)
            chaser_points = positions[:n, col_idx, :]
            valid &= np.isfinite(chaser_points).all(axis=1)
            if not np.any(valid):
                continue
            point_xy = np.nanmedian(chaser_points[valid], axis=0)
            for _, zone_row in zones_by_window[window_id].iterrows():
                if _point_in_zone(zone_row, point_xy):
                    key = (window_id, str(zone_row["zone_id"]))
                    out.setdefault(key, set()).add(int(chaser_index))
                    break

    return {key: tuple(sorted(values)) for key, values in out.items()}


def _add_chaser_pattern_key(go: Any, fig: Any) -> None:
    for name, pattern_shape in CHASER_ZONE_PATTERN_KEY:
        fig.add_trace(
            go.Bar(
                x=[None],
                y=[None],
                name=name,
                marker=dict(
                    color="rgba(148,163,184,0.45)",
                    pattern=dict(
                        shape=pattern_shape,
                        fillmode="overlay",
                        fgcolor="rgba(15,23,42,0.96)",
                        fgopacity=0.95,
                        size=8,
                        solidity=0.42,
                    ),
                ),
                hoverinfo="skip",
                showlegend=True,
            )
        )


def _build_spatial_occupancy_epoch_bar(
    go: Any,
    *,
    zone_set_id: str,
    window_label: str,
    rows: pd.DataFrame,
    chaser_presence: Mapping[tuple[int, str], tuple[int, ...]],
    y_axis_max: Optional[float],
) -> Any:
    row_records = rows.sort_values(["display_order", "zone_label"]).to_dict("records")
    pattern_shapes = [
        _chaser_pattern_shape(
            chaser_presence.get((int(row["window_id"]), str(row["zone_id"])), ())
        )
        for row in row_records
    ]
    customdata = [
        [
            int(row["frame_count"]),
            float(row["fraction_of_epoch"]) * 100.0,
            float(row["fraction_of_detected"]) * 100.0,
            float(row["coverage_pct"]),
            _format_chaser_zone_presence(
                chaser_presence.get((int(row["window_id"]), str(row["zone_id"])), ())
            ),
        ]
        for row in row_records
    ]
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=[str(row["zone_label"]) for row in row_records],
            y=[float(row["time_s"]) for row in row_records],
            name=str(window_label),
            marker=dict(
                pattern=dict(
                    shape=pattern_shapes,
                    fillmode="overlay",
                    fgcolor="rgba(15,23,42,0.96)",
                    fgopacity=0.95,
                    size=8,
                    solidity=0.42,
                )
            ),
            customdata=customdata,
            hovertemplate=(
                "<b>%{x}</b><br>"
                "time=%{y:.2f}s<br>"
                "frames=%{customdata[0]:,}<br>"
                "epoch fraction=%{customdata[1]:.1f}%<br>"
                "detected fraction=%{customdata[2]:.1f}%<br>"
                "coverage=%{customdata[3]:.1f}%<br>"
                "chaser zone=%{customdata[4]}"
                "<extra>%{fullData.name}</extra>"
            ),
        )
    )
    _add_chaser_pattern_key(go, fig)
    fig.update_layout(
        title=f"Spatial Occupancy Zones ({zone_set_id}, {window_label})",
        xaxis_title="Zone",
        yaxis_title="Time in zone (s)",
        showlegend=True,
        height=390,
        margin=dict(l=56, r=20, t=58, b=108),
        legend=dict(
            title=dict(text="Pattern key"),
            orientation="h",
            yanchor="top",
            y=-0.26,
            xanchor="left",
            x=0.0,
        ),
    )
    if y_axis_max is not None and np.isfinite(y_axis_max) and y_axis_max > 0:
        fig.update_yaxes(range=[0, float(y_axis_max)])
    return fig


def build_spatial_occupancy_output(
    mo: Any,
    go: Any,
    *,
    loaded: GoodCopBadCopLoadedView,
    window: GoodCopBadCopTimeWindow,
    spatial_zone_set_picker: Any = None,
) -> Any:
    if not len(loaded.spatial_occupancy_df):
        return mo.md("No persisted spatial occupancy zone summaries are linked to this spec.")

    zone_set_id = (
        str(spatial_zone_set_picker.value)
        if spatial_zone_set_picker is not None
        else str(loaded.spatial_occupancy_df["zone_set_id"].iloc[0])
    )
    zone_df = loaded.spatial_occupancy_df[
        loaded.spatial_occupancy_df["zone_set_id"].astype(str) == zone_set_id
    ].copy()
    if not len(zone_df):
        return mo.md(f"No spatial occupancy rows found for `{zone_set_id}`.")

    zone_df = zone_df[zone_df["window_label"].map(_is_chaser_zone_marker_epoch)].copy()
    if not len(zone_df):
        return mo.md(f"No pre/post spatial occupancy rows found for `{zone_set_id}`.")

    zone_df = zone_df.sort_values(["window_index", "display_order", "zone_label"]).copy()
    chaser_presence = _chaser_zone_presence_by_epoch(loaded=loaded, zone_df=zone_df)
    y_values = zone_df["time_s"].to_numpy(dtype=float)
    finite_y = y_values[np.isfinite(y_values)]
    y_axis_max = float(np.max(finite_y) * 1.08) if finite_y.size else None

    figures = []
    for window_label, rows in zone_df.groupby("window_label", sort=False):
        figures.append(
            _build_spatial_occupancy_epoch_bar(
                go,
                zone_set_id=zone_set_id,
                window_label=str(window_label),
                rows=rows,
                chaser_presence=chaser_presence,
                y_axis_max=y_axis_max,
            )
        )

    if len(figures) == 1:
        return figures[0]
    return mo.vstack(figures)


def _pl_to_pandas(frame: pl.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(frame.to_dicts())


def _metric_text(value: object, *, digits: int = 3, suffix: str = "") -> str:
    try:
        number = float(value)
    except Exception:
        return "n/a"
    if not np.isfinite(number):
        return "n/a"
    return f"{number:.{digits}f}{suffix}"


def _summary_metric(summary: Mapping[str, Any], key: str) -> object:
    return summary.get(key, None)


def _role_colors_from_objects(objects_df: pl.DataFrame) -> dict[str, str]:
    if objects_df.is_empty():
        return {}
    return {
        str(row.get("object_role") or ""): str(row.get("raw_color_hex") or "#64748b")
        for row in objects_df.to_dicts()
    }


def _role_colors(endpoint: GoodCopBadCopCRAEndpointData) -> dict[str, str]:
    return _role_colors_from_objects(endpoint.objects_df)


def _cra_metric_bars(
    go: Any,
    *,
    endpoint: GoodCopBadCopCRAEndpointData,
    metric: str,
    title: str,
    yaxis_title: str,
    yaxis_range: Optional[list[float]] = None,
) -> Any:
    frame = endpoint.per_object_phase_df
    if frame.is_empty() or metric not in frame.columns:
        return None
    colors = _role_colors(endpoint)
    fig = go.Figure()
    for role in ("aggressive", "benign"):
        rows = frame.filter(pl.col("object_role") == role).sort("phase_index")
        if rows.is_empty():
            continue
        fig.add_trace(
            go.Bar(
                x=rows["phase_label"].to_list(),
                y=rows[metric].to_numpy(),
                name=role,
                marker=dict(color=colors.get(role, "#64748b")),
                customdata=np.column_stack(
                    [
                        rows["object_quadrant"].to_numpy(),
                        rows["valid_frame_count"].to_numpy(),
                        rows["tracking_dropout_fraction"].to_numpy(),
                    ]
                ),
                hovertemplate=(
                    "phase=%{x}<br>"
                    f"{metric}=%{{y:.4f}}<br>"
                    "object quadrant=%{customdata[0]}<br>"
                    "valid frames=%{customdata[1]:,}<br>"
                    "dropout=%{customdata[2]:.3f}"
                    "<extra>%{fullData.name}</extra>"
                ),
            )
        )
    fig.update_layout(
        title=title,
        xaxis_title="Phase",
        yaxis_title=yaxis_title,
        barmode="group",
        height=390,
        margin=dict(l=56, r=20, t=58, b=72),
        legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="left", x=0.0),
    )
    if yaxis_range is not None:
        fig.update_yaxes(range=yaxis_range)
    return fig


def _quadrant_rect(code: int, *, width: float, height: float) -> Optional[tuple[float, float, float, float]]:
    if code < 0 or code > 3 or not (np.isfinite(width) and np.isfinite(height) and width > 0 and height > 0):
        return None
    half_w = float(width) / 2.0
    half_h = float(height) / 2.0
    x0 = half_w if code in {1, 3} else 0.0
    y0 = half_h if code in {2, 3} else 0.0
    return x0, y0, x0 + half_w, y0 + half_h


def _cra_phase_quadrant_figure(
    go: Any,
    *,
    endpoint: GoodCopBadCopCRAEndpointData,
    phase_label: str,
) -> Any:
    rows = endpoint.object_phase_df.filter(pl.col("phase_label") == str(phase_label))
    fig = go.Figure()
    width = float(endpoint.quadrant_width_px)
    height = float(endpoint.quadrant_height_px)
    if np.isfinite(width) and np.isfinite(height) and width > 0 and height > 0:
        fig.add_shape(type="rect", x0=0, y0=0, x1=width, y1=height, line=dict(color="#334155", width=1.2))
        fig.add_shape(type="line", x0=width / 2.0, y0=0, x1=width / 2.0, y1=height, line=dict(color="#94a3b8", width=1, dash="dot"))
        fig.add_shape(type="line", x0=0, y0=height / 2.0, x1=width, y1=height / 2.0, line=dict(color="#94a3b8", width=1, dash="dot"))

    for row in rows.to_dicts():
        color = str(row.get("raw_color_hex") or "#64748b")
        role = str(row.get("object_role") or "object")
        code = int(row.get("object_quadrant_code") or -1)
        rect = _quadrant_rect(code, width=width, height=height)
        if rect is not None:
            x0, y0, x1, y1 = rect
            fig.add_shape(
                type="rect",
                x0=x0,
                y0=y0,
                x1=x1,
                y1=y1,
                fillcolor=color,
                opacity=0.14,
                line=dict(color=color, width=1.5, dash="dash"),
            )
        fig.add_trace(
            go.Scatter(
                x=[float(row.get("object_x_px", np.nan))],
                y=[float(row.get("object_y_px", np.nan))],
                mode="markers+text",
                name=role,
                text=[role],
                textposition="top center",
                marker=dict(
                    size=18,
                    symbol="circle" if role == "aggressive" else "square",
                    color=color,
                    line=dict(width=2, color="white"),
                ),
                customdata=[[row.get("object_quadrant"), row.get("object_max_drift_mm")]],
                hovertemplate=(
                    f"{role}<br>"
                    "x=%{x:.1f}px<br>"
                    "y=%{y:.1f}px<br>"
                    "quadrant=%{customdata[0]}<br>"
                    "max drift=%{customdata[1]:.3f}mm"
                    "<extra></extra>"
                ),
            )
        )
    fig.update_yaxes(autorange="reversed", scaleanchor="x", scaleratio=1, title="Arena Y (px, down)")
    fig.update_xaxes(title="Arena X (px)")
    fig.update_layout(
        title=f"CRA Object-Relative Quadrants ({phase_label})",
        height=420,
        margin=dict(l=52, r=20, t=58, b=64),
        legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="left", x=0.0),
    )
    return fig


def build_cra_primary_endpoint_output(mo: Any, go: Any, *, loaded: GoodCopBadCopLoadedView) -> Any:
    endpoint = loaded.cra_endpoint
    if endpoint is None:
        return mo.md("No CRA primary endpoint component is linked to this chaser-distance run.")

    summary = endpoint.summary
    distance_fig = _cra_metric_bars(
        go,
        endpoint=endpoint,
        metric="median_distance_mm",
        title="CRA Primary Endpoint: Median Distance",
        yaxis_title="Median distance (mm)",
    )
    occupancy_fig = _cra_metric_bars(
        go,
        endpoint=endpoint,
        metric="occupancy_fraction",
        title="CRA Primary Endpoint: Object-Quadrant Occupancy",
        yaxis_title="Fraction of valid frames",
        yaxis_range=[0, 1],
    )
    phase_figures = [
        _cra_phase_quadrant_figure(go, endpoint=endpoint, phase_label=str(phase_label))
        for phase_label in endpoint.phases_df.sort("phase_index")["phase_label"].to_list()
    ]
    summary_rows = [
        {
            "metric": "delta_agg",
            "value": _summary_metric(summary, "delta_agg"),
            "meaning": "post - pre aggressive distance",
        },
        {
            "metric": "delta_benign",
            "value": _summary_metric(summary, "delta_benign"),
            "meaning": "post - pre benign distance",
        },
        {
            "metric": "specificity_distance",
            "value": _summary_metric(summary, "specificity_distance"),
            "meaning": "aggressive distance delta - benign distance delta",
        },
        {
            "metric": "delta_occ_agg",
            "value": _summary_metric(summary, "delta_occ_agg"),
            "meaning": "post - pre aggressive object-quadrant occupancy",
        },
        {
            "metric": "delta_occ_benign",
            "value": _summary_metric(summary, "delta_occ_benign"),
            "meaning": "post - pre benign object-quadrant occupancy",
        },
        {
            "metric": "specificity_occupancy",
            "value": _summary_metric(summary, "specificity_occupancy"),
            "meaning": "aggressive occupancy delta - benign occupancy delta",
        },
    ]
    warnings_text = ", ".join(endpoint.qc_warnings) if endpoint.qc_warnings else "none"
    items = [
        mo.md(f"## CRA Primary Endpoint\n\n`{endpoint.component_path}`"),
        mo.hstack(
            [
                mo.stat(label="Agg distance delta", value=_metric_text(_summary_metric(summary, "delta_agg"), suffix=" mm")),
                mo.stat(label="Distance specificity", value=_metric_text(_summary_metric(summary, "specificity_distance"), suffix=" mm")),
                mo.stat(label="Agg occupancy delta", value=_metric_text(_summary_metric(summary, "delta_occ_agg"))),
                mo.stat(label="Occupancy specificity", value=_metric_text(_summary_metric(summary, "specificity_occupancy"))),
                mo.stat(label="Post dropout", value=_metric_text(_summary_metric(summary, "frac_tracking_dropout_post"))),
            ]
        ),
        mo.md(f"QC warnings: `{warnings_text}`"),
        mo.ui.table(pd.DataFrame(summary_rows), selection=None, page_size=10),
    ]
    if distance_fig is not None:
        items.append(distance_fig)
    if occupancy_fig is not None:
        items.append(occupancy_fig)
    items.extend(phase_figures)
    items.append(
        mo.accordion(
            {
                "Phase windows": mo.ui.table(_pl_to_pandas(endpoint.phases_df), selection=None, page_size=10),
                "Object positions and drift": mo.ui.table(
                    _pl_to_pandas(endpoint.object_phase_df),
                    selection=None,
                    page_size=10,
                ),
                "Per-object phase metrics": mo.ui.table(
                    _pl_to_pandas(endpoint.per_object_phase_df),
                    selection=None,
                    page_size=10,
                ),
                "Object roles": mo.ui.table(_pl_to_pandas(endpoint.objects_df), selection=None, page_size=10),
            }
        )
    )
    return mo.vstack(items)


def _first_approach_metric_column(frame: pl.DataFrame) -> Optional[str]:
    if frame.is_empty():
        return None
    columns = [column for column in frame.columns if column.startswith("approach_p") and column.endswith("_mm")]
    if "approach_p05_mm" in columns:
        return "approach_p05_mm"
    return sorted(columns)[0] if columns else None


def _near_field_metric_bars(
    go: Any,
    *,
    near_field: GoodCopBadCopCRANearFieldData,
    metric: str,
    title: str,
    yaxis_title: str,
    yaxis_range: Optional[list[float]] = None,
) -> Any:
    frame = near_field.per_object_phase_df
    if frame.is_empty() or metric not in frame.columns:
        return None
    colors = _role_colors_from_objects(near_field.objects_df)
    fig = go.Figure()
    for role in ("aggressive", "benign"):
        rows = frame.filter(pl.col("object_role") == role).sort("phase_index")
        if rows.is_empty():
            continue
        fig.add_trace(
            go.Bar(
                x=rows["phase_label"].to_list(),
                y=rows[metric].to_numpy(),
                name=role,
                marker=dict(color=colors.get(role, "#64748b")),
                customdata=np.column_stack(
                    [
                        rows["valid_distance_count"].to_numpy(),
                        rows["tracking_dropout_fraction"].to_numpy(),
                        rows["near_zone_available_area_mm2"].to_numpy(),
                    ]
                ),
                hovertemplate=(
                    "phase=%{x}<br>"
                    f"{metric}=%{{y:.4f}}<br>"
                    "valid distances=%{customdata[0]:,}<br>"
                    "dropout=%{customdata[1]:.3f}<br>"
                    "near-zone area=%{customdata[2]:.3f} mm2"
                    "<extra>%{fullData.name}</extra>"
                ),
            )
        )
    fig.update_layout(
        title=title,
        xaxis_title="Phase",
        yaxis_title=yaxis_title,
        barmode="group",
        height=390,
        margin=dict(l=56, r=20, t=58, b=72),
        legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="left", x=0.0),
    )
    if yaxis_range is not None:
        fig.update_yaxes(range=yaxis_range)
    return fig


def _near_field_radial_density_figure(
    go: Any,
    *,
    near_field: GoodCopBadCopCRANearFieldData,
    density_column: str = "radial_density_per_mm2",
    fraction_column: str = "radial_fraction",
    area_column: str = "radial_available_area_mm2",
    title: str = "CRA Near-Field: Radial Occupancy Density",
) -> Any:
    frame = near_field.radial_density_df
    if frame.is_empty() or density_column not in frame.columns:
        return None
    colors = _role_colors_from_objects(near_field.objects_df)
    fig = go.Figure()
    phase_dash = {"pre_static": "solid", "post_static": "dash"}
    trace_count = 0
    for role in ("aggressive", "benign"):
        for phase_label in frame["phase_label"].unique().sort().to_list():
            rows = frame.filter((pl.col("object_role") == role) & (pl.col("phase_label") == phase_label)).sort(
                "radial_bin_index"
            )
            if rows.is_empty():
                continue
            if not np.isfinite(rows[density_column].to_numpy()).any():
                continue
            fig.add_trace(
                go.Scatter(
                    x=rows["radial_bin_center_mm"].to_numpy(),
                    y=rows[density_column].to_numpy(),
                    mode="lines+markers",
                    name=f"{phase_label} {role}",
                    marker=dict(color=colors.get(role, "#64748b"), size=7),
                    line=dict(color=colors.get(role, "#64748b"), dash=phase_dash.get(str(phase_label), "solid")),
                    customdata=np.column_stack(
                        [
                            rows["radial_bin_start_mm"].to_numpy(),
                            rows["radial_bin_end_mm"].to_numpy(),
                            rows[fraction_column].to_numpy() if fraction_column in rows.columns else np.full(rows.height, np.nan),
                            rows[area_column].to_numpy() if area_column in rows.columns else np.full(rows.height, np.nan),
                            rows["radial_wall_excluded_valid_count"].to_numpy()
                            if "radial_wall_excluded_valid_count" in rows.columns
                            else np.full(rows.height, np.nan),
                        ]
                    ),
                    hovertemplate=(
                        "bin=%{customdata[0]:.2f}-%{customdata[1]:.2f} mm<br>"
                        "density=%{y:.6f}<br>"
                        "fraction=%{customdata[2]:.4f}<br>"
                        "available area=%{customdata[3]:.3f} mm2<br>"
                        "wall-excluded valid=%{customdata[4]}"
                        "<extra>%{fullData.name}</extra>"
                    ),
                )
            )
            trace_count += 1
    if trace_count == 0:
        return None
    fig.update_layout(
        title=title,
        xaxis_title="Distance to object (mm)",
        yaxis_title="Fraction per mm2",
        height=430,
        margin=dict(l=62, r=20, t=58, b=78),
        legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="left", x=0.0),
    )
    return fig


def _near_field_cdf_figure(go: Any, *, near_field: GoodCopBadCopCRANearFieldData) -> Any:
    frame = near_field.cdf_df
    if frame.is_empty():
        return None
    colors = _role_colors_from_objects(near_field.objects_df)
    fig = go.Figure()
    phase_dash = {"pre_static": "solid", "post_static": "dash"}
    for role in ("aggressive", "benign"):
        for phase_label in frame["phase_label"].unique().sort().to_list():
            rows = frame.filter((pl.col("object_role") == role) & (pl.col("phase_label") == phase_label)).sort(
                "threshold_index"
            )
            if rows.is_empty():
                continue
            fig.add_trace(
                go.Scatter(
                    x=rows["threshold_mm"].to_numpy(),
                    y=rows["cdf_fraction"].to_numpy(),
                    mode="lines+markers",
                    name=f"{phase_label} {role}",
                    marker=dict(color=colors.get(role, "#64748b"), size=7),
                    line=dict(color=colors.get(role, "#64748b"), dash=phase_dash.get(str(phase_label), "solid")),
                    hovertemplate=(
                        "threshold=%{x:.2f} mm<br>"
                        "P(distance <= threshold)=%{y:.4f}"
                        "<extra>%{fullData.name}</extra>"
                    ),
                )
            )
    per_phase = near_field.per_object_phase_df
    if not per_phase.is_empty() and "approach_p05_mm" in per_phase.columns:
        marker_rows = per_phase.sort(["object_role", "phase_index"])
        for role in ("aggressive", "benign"):
            rows = marker_rows.filter(pl.col("object_role") == role)
            if rows.is_empty():
                continue
            cdf_y = (
                rows["approach_p05_cdf_fraction"].to_numpy()
                if "approach_p05_cdf_fraction" in rows.columns
                else np.full(rows.height, 0.05)
            )
            fig.add_trace(
                go.Scatter(
                    x=rows["approach_p05_mm"].to_numpy(),
                    y=cdf_y,
                    mode="markers",
                    name=f"p05 check {role}",
                    marker=dict(
                        color=colors.get(role, "#64748b"),
                        symbol="diamond-open",
                        size=10,
                        line=dict(width=1.6),
                    ),
                    customdata=np.column_stack(
                        [
                            rows["phase_label"].to_numpy(),
                            rows["valid_distance_count"].to_numpy(),
                            cdf_y,
                        ]
                    ),
                    hovertemplate=(
                        "phase=%{customdata[0]}<br>"
                        "p05 distance=%{x:.3f} mm<br>"
                        "CDF at p05=%{customdata[2]:.4f}<br>"
                        "valid distances=%{customdata[1]:,}"
                        "<extra>%{fullData.name}</extra>"
                    ),
                )
            )
    fig.update_layout(
        title="CRA Near-Field: Distance CDF",
        xaxis_title="Distance threshold (mm)",
        yaxis_title="Fraction of valid frames",
        height=430,
        margin=dict(l=62, r=20, t=58, b=78),
        legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="left", x=0.0),
    )
    fig.update_yaxes(range=[0, 1])
    return fig


def _near_field_control_reference_cdf_figure(go: Any, *, near_field: GoodCopBadCopCRANearFieldData) -> Any:
    frame = near_field.control_reference_cdf_df
    if frame.is_empty():
        return None
    fig = go.Figure()
    phase_dash = {"pre_static": "solid", "post_static": "dash"}
    palette = {"dish_center": "#0f766e"}
    for reference_label in frame["reference_label"].unique().sort().to_list():
        for phase_label in frame["phase_label"].unique().sort().to_list():
            rows = frame.filter(
                (pl.col("reference_label") == reference_label) & (pl.col("phase_label") == phase_label)
            ).sort("threshold_index")
            if rows.is_empty():
                continue
            fig.add_trace(
                go.Scatter(
                    x=rows["threshold_mm"].to_numpy(),
                    y=rows["cdf_fraction"].to_numpy(),
                    mode="lines+markers",
                    name=f"{phase_label} {reference_label}",
                    marker=dict(color=palette.get(str(reference_label), "#0f766e"), size=7),
                    line=dict(color=palette.get(str(reference_label), "#0f766e"), dash=phase_dash.get(str(phase_label), "solid")),
                    hovertemplate=(
                        "threshold=%{x:.2f} mm<br>"
                        "P(distance <= threshold)=%{y:.4f}"
                        "<extra>%{fullData.name}</extra>"
                    ),
                )
            )
    fig.update_layout(
        title="CRA Near-Field: Dish-Center Control CDF",
        xaxis_title="Distance threshold (mm)",
        yaxis_title="Fraction of valid frames",
        height=400,
        margin=dict(l=62, r=20, t=58, b=78),
        legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="left", x=0.0),
    )
    fig.update_yaxes(range=[0, 1])
    return fig


def _near_field_control_reference_radial_density_figure(go: Any, *, near_field: GoodCopBadCopCRANearFieldData) -> Any:
    frame = near_field.control_reference_radial_density_df
    if frame.is_empty():
        return None
    fig = go.Figure()
    phase_dash = {"pre_static": "solid", "post_static": "dash"}
    palette = {"dish_center": "#0f766e"}
    for reference_label in frame["reference_label"].unique().sort().to_list():
        for phase_label in frame["phase_label"].unique().sort().to_list():
            rows = frame.filter(
                (pl.col("reference_label") == reference_label) & (pl.col("phase_label") == phase_label)
            ).sort("radial_bin_index")
            if rows.is_empty():
                continue
            fig.add_trace(
                go.Scatter(
                    x=rows["radial_bin_center_mm"].to_numpy(),
                    y=rows["radial_density_per_mm2"].to_numpy(),
                    mode="lines+markers",
                    name=f"{phase_label} {reference_label}",
                    marker=dict(color=palette.get(str(reference_label), "#0f766e"), size=7),
                    line=dict(color=palette.get(str(reference_label), "#0f766e"), dash=phase_dash.get(str(phase_label), "solid")),
                    customdata=np.column_stack(
                        [
                            rows["radial_bin_start_mm"].to_numpy(),
                            rows["radial_bin_end_mm"].to_numpy(),
                            rows["radial_fraction"].to_numpy(),
                            rows["radial_available_area_mm2"].to_numpy(),
                        ]
                    ),
                    hovertemplate=(
                        "bin=%{customdata[0]:.2f}-%{customdata[1]:.2f} mm<br>"
                        "density=%{y:.6f}<br>"
                        "fraction=%{customdata[2]:.4f}<br>"
                        "available area=%{customdata[3]:.3f} mm2"
                        "<extra>%{fullData.name}</extra>"
                    ),
                )
            )
    fig.update_layout(
        title="CRA Near-Field: Dish-Center Control Radial Density",
        xaxis_title="Distance to reference (mm)",
        yaxis_title="Fraction per mm2",
        height=400,
        margin=dict(l=62, r=20, t=58, b=78),
        legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="left", x=0.0),
    )
    return fig


def _near_field_thigmotaxis_figure(go: Any, *, near_field: GoodCopBadCopCRANearFieldData) -> Any:
    frame = near_field.thigmotaxis_df
    if frame.is_empty() or "thigmotaxis_fraction" not in frame.columns:
        return None
    rows = frame.sort("phase_index")
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Wall/immobility fractions", "Mean speed"),
        specs=[[{"secondary_y": False}, {"secondary_y": False}]],
    )
    fig.add_trace(
        go.Bar(
            x=rows["phase_label"].to_list(),
            y=rows["thigmotaxis_fraction"].to_numpy(),
            name="wall band fraction",
            marker=dict(color="#475569"),
            customdata=np.column_stack(
                [
                    rows["thigmotaxis_dwell_s"].to_numpy(),
                    rows["geometry_status"].to_numpy(),
                    rows["speed_sample_count"].to_numpy() if "speed_sample_count" in rows.columns else np.zeros(rows.height),
                ]
            ),
            hovertemplate=(
                "phase=%{x}<br>"
                "fraction=%{y:.4f}<br>"
                "dwell=%{customdata[0]:.3f}s<br>"
                "geometry=%{customdata[1]}<br>"
                "speed samples=%{customdata[2]}"
                "<extra></extra>"
            ),
        ),
        row=1,
        col=1,
    )
    if "immobile_fraction" in rows.columns:
        fig.add_trace(
            go.Bar(
                x=rows["phase_label"].to_list(),
                y=rows["immobile_fraction"].to_numpy(),
                name="immobile fraction",
                marker=dict(color="#a16207"),
                hovertemplate="phase=%{x}<br>fraction=%{y:.4f}<extra></extra>",
            ),
            row=1,
            col=1,
        )
    if "mean_speed_mm_s" in rows.columns:
        fig.add_trace(
            go.Bar(
                x=rows["phase_label"].to_list(),
                y=rows["mean_speed_mm_s"].to_numpy(),
                name="mean speed",
                marker=dict(color="#2563eb"),
                customdata=np.column_stack(
                    [
                        rows["median_speed_mm_s"].to_numpy()
                        if "median_speed_mm_s" in rows.columns
                        else np.full(rows.height, np.nan),
                    ]
                ),
                hovertemplate=(
                    "phase=%{x}<br>"
                    "mean speed=%{y:.3f} mm/s<br>"
                    "median speed=%{customdata[0]:.3f} mm/s"
                    "<extra></extra>"
                ),
            ),
            row=1,
            col=2,
        )
    fig.update_layout(
        title="CRA Near-Field: Global-State QC",
        barmode="group",
        height=360,
        margin=dict(l=56, r=20, t=58, b=64),
        legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="left", x=0.0),
    )
    fig.update_yaxes(title_text="Fraction of valid frames", range=[0, 1], row=1, col=1)
    fig.update_yaxes(title_text="Speed (mm/s)", rangemode="tozero", row=1, col=2)
    return fig


def build_cra_near_field_output(mo: Any, go: Any, *, loaded: GoodCopBadCopLoadedView) -> Any:
    near_field = loaded.cra_near_field
    if near_field is None:
        return mo.md("No CRA near-field component is linked to this chaser-distance run.")

    summary = near_field.summary
    approach_metric = _first_approach_metric_column(near_field.per_object_phase_df)
    approach_key = approach_metric[:-3] if approach_metric and approach_metric.endswith("_mm") else None
    figures = []
    if approach_metric:
        figures.append(
            _near_field_metric_bars(
                go,
                near_field=near_field,
                metric=approach_metric,
                title="CRA Near-Field: Close-Approach Distance",
                yaxis_title="Distance percentile (mm)",
            )
        )
    figures.extend(
        [
            _near_field_metric_bars(
                go,
                near_field=near_field,
                metric="near_zone_occupancy_fraction",
                title="CRA Near-Field: Near-Zone Occupancy",
                yaxis_title="Fraction of valid distance frames",
                yaxis_range=[0, 1],
            ),
            _near_field_metric_bars(
                go,
                near_field=near_field,
                metric="near_zone_entry_rate_per_min",
                title="CRA Near-Field: Near-Zone Entry Rate",
                yaxis_title="Entries per minute",
            ),
            _near_field_radial_density_figure(go, near_field=near_field),
            _near_field_radial_density_figure(
                go,
                near_field=near_field,
                density_column="radial_density_wall_excluded_per_mm2",
                fraction_column="radial_fraction_wall_excluded",
                area_column="radial_available_area_wall_excluded_mm2",
                title="CRA Near-Field: Wall-Band-Excluded Radial Density",
            ),
            _near_field_cdf_figure(go, near_field=near_field),
            _near_field_control_reference_radial_density_figure(go, near_field=near_field),
            _near_field_control_reference_cdf_figure(go, near_field=near_field),
            _near_field_thigmotaxis_figure(go, near_field=near_field),
        ]
    )
    figures = [figure for figure in figures if figure is not None]

    summary_rows = [
        {"metric": key, "value": value}
        for key, value in sorted(summary.items())
        if key.startswith("approach_") or key.startswith("nearzone_") or key.startswith("thigmotaxis_")
    ]
    warnings_text = ", ".join(near_field.qc_warnings) if near_field.qc_warnings else "none"
    items = [
        mo.md(f"## CRA Near-Field Avoidance\n\n`{near_field.component_path}`"),
        mo.hstack(
            [
                mo.stat(
                    label="Approach specificity",
                    value=_metric_text(
                        _summary_metric(summary, f"{approach_key}_specificity") if approach_key else None,
                        suffix=" mm",
                    ),
                ),
                mo.stat(
                    label="Near-zone specificity",
                    value=_metric_text(_summary_metric(summary, "nearzone_occ_specificity")),
                ),
                mo.stat(
                    label="Entry-rate specificity",
                    value=_metric_text(_summary_metric(summary, "nearzone_entry_rate_specificity")),
                ),
                mo.stat(label="Geometry", value=near_field.geometry_status or "unknown"),
                mo.stat(label="Arena", value=near_field.arena_shape or "unknown"),
                mo.stat(
                    label="Pctl/CDF check",
                    value=_metric_text(_summary_metric(summary, "approach_percentile_cdf_max_abs_error")),
                ),
            ]
        ),
        mo.md(f"QC warnings: `{warnings_text}`"),
        mo.ui.table(pd.DataFrame(summary_rows), selection=None, page_size=12),
        *figures,
        mo.accordion(
            {
                "Near-field phase metrics": mo.ui.table(
                    _pl_to_pandas(near_field.per_object_phase_df),
                    selection=None,
                    page_size=10,
                ),
                "Radial occupancy density": mo.ui.table(
                    _pl_to_pandas(near_field.radial_density_df),
                    selection=None,
                    page_size=15,
                ),
                "Distance CDF": mo.ui.table(_pl_to_pandas(near_field.cdf_df), selection=None, page_size=15),
                "Global-state QC": mo.ui.table(
                    _pl_to_pandas(near_field.thigmotaxis_df),
                    selection=None,
                    page_size=10,
                ),
                "Control reference radial density": mo.ui.table(
                    _pl_to_pandas(near_field.control_reference_radial_density_df),
                    selection=None,
                    page_size=15,
                ),
                "Control reference CDF": mo.ui.table(
                    _pl_to_pandas(near_field.control_reference_cdf_df),
                    selection=None,
                    page_size=15,
                ),
                "Control reference phase metrics": mo.ui.table(
                    _pl_to_pandas(near_field.control_reference_phase_df),
                    selection=None,
                    page_size=10,
                ),
                "Object roles": mo.ui.table(_pl_to_pandas(near_field.objects_df), selection=None, page_size=10),
                "Phase windows": mo.ui.table(_pl_to_pandas(near_field.phases_df), selection=None, page_size=10),
            }
        ),
    ]
    return mo.vstack(items)


def build_escape_freeze_output(mo: Any, *, loaded: GoodCopBadCopLoadedView) -> Any:
    escape_freeze = loaded.escape_freeze
    if escape_freeze is None:
        return mo.md("No chaser escape/freeze canary component is linked to this chaser-distance run.")

    summary = escape_freeze.summary
    diagnostics = escape_freeze.diagnostics
    warning_text = ", ".join(escape_freeze.warnings) if escape_freeze.warnings else "none"
    image_items = []
    if escape_freeze.response_class_bar_png_bytes:
        image_items.append(
            mo.vstack(
                [
                    mo.md(f"### Candidate Response Summary\n\n`{escape_freeze.response_class_bar_png_path}`"),
                    png_bytes_to_markdown_image(
                        mo,
                        escape_freeze.response_class_bar_png_bytes,
                        alt_text="Candidate escape attempt versus not escape response-class summary",
                    ),
                ]
            )
        )
    elif escape_freeze.response_class_bar_png_error:
        image_items.append(
            mo.md(f"Response-class bar PNG unavailable: `{escape_freeze.response_class_bar_png_error}`")
        )

    if escape_freeze.trial_outcome_timeline_png_bytes:
        image_items.append(
            mo.vstack(
                [
                    mo.md(f"### Trial Outcome Timeline\n\n`{escape_freeze.trial_outcome_timeline_png_path}`"),
                    png_bytes_to_markdown_image(
                        mo,
                        escape_freeze.trial_outcome_timeline_png_bytes,
                        alt_text="Candidate escape attempt versus not escape trial-outcome timeline",
                    ),
                ]
            )
        )
    elif escape_freeze.trial_outcome_timeline_png_error:
        image_items.append(
            mo.md(
                "Trial-outcome timeline PNG unavailable: "
                f"`{escape_freeze.trial_outcome_timeline_png_error}`"
            )
        )

    if escape_freeze.fish_centered_polar_approach_png_bytes:
        image_items.append(
            mo.vstack(
                [
                    mo.md(
                        "### Fish-Centered Polar Chaser Approach\n\n"
                        f"`{escape_freeze.fish_centered_polar_approach_png_path}`"
                    ),
                    png_bytes_to_markdown_image(
                        mo,
                        escape_freeze.fish_centered_polar_approach_png_bytes,
                        alt_text="Fish-centered polar chaser approach scatter",
                    ),
                ]
            )
        )
    elif escape_freeze.fish_centered_polar_approach_png_error:
        image_items.append(
            mo.md(
                "Fish-centered polar approach PNG unavailable: "
                f"`{escape_freeze.fish_centered_polar_approach_png_error}`"
            )
        )

    if escape_freeze.fish_centered_polar_density_png_bytes:
        image_items.append(
            mo.vstack(
                [
                    mo.md(
                        "### Fish-Centered Polar Chaser Density\n\n"
                        f"`{escape_freeze.fish_centered_polar_density_png_path}`"
                    ),
                    png_bytes_to_markdown_image(
                        mo,
                        escape_freeze.fish_centered_polar_density_png_bytes,
                        alt_text="Fish-centered polar chaser approach density",
                    ),
                ]
            )
        )
    elif escape_freeze.fish_centered_polar_density_png_error:
        image_items.append(
            mo.md(
                "Fish-centered polar density PNG unavailable: "
                f"`{escape_freeze.fish_centered_polar_density_png_error}`"
            )
        )

    if escape_freeze.per_trial_png_bytes:
        image_items.append(
            mo.vstack(
                [
                    mo.md(f"### Per-Trial Diagnostic\n\n`{escape_freeze.per_trial_png_path}`"),
                    png_bytes_to_markdown_image(
                        mo,
                        escape_freeze.per_trial_png_bytes,
                        alt_text="Chaser-centric per-trial escape/freeze diagnostic",
                    ),
                ]
            )
        )
    elif escape_freeze.per_trial_png_error:
        image_items.append(mo.md(f"Per-trial PNG unavailable: `{escape_freeze.per_trial_png_error}`"))

    if escape_freeze.fish_centered_png_bytes:
        image_items.append(
            mo.vstack(
                [
                    mo.md(f"### Fish-Centered Chaser Diagnostic\n\n`{escape_freeze.fish_centered_png_path}`"),
                    png_bytes_to_markdown_image(
                        mo,
                        escape_freeze.fish_centered_png_bytes,
                        alt_text="Fish-centered per-trial chaser-position diagnostic",
                    ),
                ]
            )
        )
    elif escape_freeze.fish_centered_png_error:
        image_items.append(mo.md(f"Fish-centered PNG unavailable: `{escape_freeze.fish_centered_png_error}`"))

    if escape_freeze.scatter_png_bytes:
        image_items.append(
            mo.vstack(
                [
                    mo.md(f"### Speed vs Displacement\n\n`{escape_freeze.scatter_png_path}`"),
                    png_bytes_to_markdown_image(
                        mo,
                        escape_freeze.scatter_png_bytes,
                        alt_text="Speed versus displacement escape/freeze canary scatter",
                    ),
                ]
            )
        )
    elif escape_freeze.scatter_png_error:
        image_items.append(mo.md(f"Scatter PNG unavailable: `{escape_freeze.scatter_png_error}`"))

    summary_rows = [{"metric": key, "value": value} for key, value in sorted(summary.items())]
    diagnostic_rows = [{"metric": key, "value": value} for key, value in sorted(diagnostics.items())]
    return mo.vstack(
        [
            mo.md(f"## Chaser Escape/Freeze Canary\n\n`{escape_freeze.component_path}`"),
            mo.hstack(
                [
                    mo.stat(label="Trials", value=f"{int(summary.get('trial_count') or 0):,}"),
                    mo.stat(label="Chaser", value=str(summary.get("chaser_index", "unknown"))),
                    mo.stat(
                        label="Classifier",
                        value="locked" if bool(summary.get("classification_locked")) else "candidate",
                    ),
                    mo.stat(
                        label="Escape attempts",
                        value=(
                            f"{int(summary.get('escape_attempt_count') or 0):,}/"
                            f"{int(summary.get('trial_count') or 0):,}"
                        ),
                    ),
                    mo.stat(
                        label="Path threshold",
                        value=_metric_text(summary.get("escape_path_threshold_mm"), suffix=" mm"),
                    ),
                    mo.stat(
                        label="Mean resp speed",
                        value=_metric_text(summary.get("mean_response_speed_mm_s"), suffix=" mm/s"),
                    ),
                    mo.stat(
                        label="Mean low speed",
                        value=_metric_text(summary.get("mean_freeze_low_speed_fraction")),
                    ),
                ]
            ),
            mo.md(
                "This canary is diagnostic/US-validation only for the current cohort; active pursuit is "
                "available for chaser 0 but not the benign chaser. Escape labels are candidate labels from "
                "full-trial fish path length; classifier thresholds are not cohort-locked."
            ),
            mo.md(f"Warnings: `{warning_text}`"),
            *image_items,
            mo.accordion(
                {
                    "Summary": mo.ui.table(pd.DataFrame(summary_rows), selection=None, page_size=12),
                    "Diagnostics": mo.ui.table(pd.DataFrame(diagnostic_rows), selection=None, page_size=12),
                    "Trials": mo.ui.table(_pl_to_pandas(escape_freeze.trials_df), selection=None, page_size=15),
                    "Trial metrics": mo.ui.table(
                        _pl_to_pandas(escape_freeze.trial_metrics_df),
                        selection=None,
                        page_size=15,
                    ),
                    "Trajectory samples": mo.ui.table(
                        _pl_to_pandas(escape_freeze.trial_trajectories_df.head(1000)),
                        selection=None,
                        page_size=15,
                    ),
                }
            ),
        ]
    )


def build_debug_tables(
    mo: Any,
    *,
    loaded: GoodCopBadCopLoadedView,
    visible_distance_df: pd.DataFrame,
    visible_position_df: pd.DataFrame,
) -> Any:
    return mo.accordion(
        {
            "Windows": mo.ui.table(loaded.windows_df, selection=None, page_size=10),
            "Visible distance rows": mo.ui.table(visible_distance_df.head(1000), selection=None, page_size=15),
            "Visible position rows": mo.ui.table(visible_position_df.head(1000), selection=None, page_size=15),
            "Spatial occupancy": mo.ui.table(loaded.spatial_occupancy_df, selection=None, page_size=15),
            "Epoch kinematics summary": mo.ui.table(_pl_to_pandas(loaded.epoch_summary_df), selection=None, page_size=15),
        }
    )
