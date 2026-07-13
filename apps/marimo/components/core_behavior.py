"""Core-behavior component with deferred Zarr projections and Polars queries.

Polars cannot scan Zarr directly.  This component therefore keeps the Zarr
source unopened until an analysis is selected, reads only that analysis' array
projection (and selected row interval), then exposes the in-memory projection
as a ``polars.LazyFrame``.  Exported Parquet data uses ``scan_parquet`` and is
lazy from storage through collection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import polars as pl

from fisheye.analysis.swim_bout_io import (
    SwimBoutIOError,
    discover_swim_bout_candidates,
    load_swim_bout_events,
    structured_records_to_dicts,
)
from fisheye.analytics_exports.baseline import is_baseline_label
from fisheye.shared.json_safety import decode_null_terminated_text
from fisheye.shared.zarr_io import open_zarr_root
from fisheye.visualization.interactive_track_kinematics import (
    discover_eye_angle_run_options,
    load_eye_angle_timeseries_data,
)

from .registry import InteractiveSpecOption


TRACK_KINEMATICS_RENDERER = "palette-track-kinematics-summary-v1"

SPEED_SERIES_PREFIXES = ("speed_", "acceleration_", "smoothed_acceleration_")
HEADING_SERIES_TOKENS = ("heading", "angular_velocity", "angular_speed")
PATH_SERIES_TOKENS = ("cumulative_path_distance", "frame_path_distance")


@dataclass(frozen=True)
class BaselineInterval:
    label: str
    start_s: float
    stop_s: float


@dataclass(frozen=True)
class CoreBehaviorProjection:
    analysis_id: str
    frame: pl.LazyFrame
    columns: tuple[str, ...]
    source_paths: tuple[str, ...]
    start_s: float
    stop_s: float
    row_count: int
    load_duration_ms: float
    note: str
    related_frames: Mapping[str, pl.LazyFrame] = field(default_factory=dict)


def scan_export_parquet(
    paths: str | Path | Sequence[str | Path],
    *,
    columns: Sequence[str] | None = None,
) -> pl.LazyFrame:
    """Return a true lazy scan for an exported Parquet table or partition."""

    if isinstance(paths, (str, Path)):
        source: str | list[str] = str(paths)
    else:
        source = [str(path) for path in paths]
    lazy = pl.scan_parquet(source)
    return lazy.select(list(columns)) if columns is not None else lazy


def is_core_behavior_option(option: InteractiveSpecOption) -> bool:
    return option.renderer == TRACK_KINEMATICS_RENDERER


def _normal_path(value: object) -> str:
    return "/".join(part for part in str(value or "").strip("/").split("/") if part)


def _source_paths(option: InteractiveSpecOption) -> dict[str, str]:
    raw = option.spec.get("source_paths")
    if not isinstance(raw, Mapping):
        return {}
    return {str(key): _normal_path(value) for key, value in raw.items() if _normal_path(value)}


def _finite_bounds(values: np.ndarray) -> tuple[float, float]:
    finite = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0.0, 0.0
    return float(finite[0]), float(finite[-1])


def _structured_lazy(records: np.ndarray) -> pl.LazyFrame:
    rows = structured_records_to_dicts(np.asarray(records))
    return pl.from_dicts(rows).lazy() if rows else pl.DataFrame().lazy()


def _is_physical_speed_column(name: str) -> bool:
    return (
        name.startswith("speed_")
        and (name.endswith("_mm") or name.endswith("_px"))
        and "acceleration" not in name
        and "frame_path_distance" not in name
    )


class CoreBehaviorSource:
    """Metadata-first, read-only source for one track-kinematics spec."""

    def __init__(self, zarr_path: Path | str, option: InteractiveSpecOption):
        if not is_core_behavior_option(option):
            raise ValueError(f"Not a core-behavior renderer: {option.renderer!r}")
        self.zarr_path = Path(zarr_path)
        self.option = option
        self.source_paths = _source_paths(option)
        self.track_id = int(option.spec.get("track_id") or 0)
        self._time_seconds_cache: np.ndarray | None = None
        self._swim_bout_selection_cache: tuple[Any, Any] | None = None
        self._swim_bout_events_cache: Any = None
        self._available_analysis_ids_cache: tuple[str, ...] | None = None

    @property
    def available_series(self) -> tuple[str, ...]:
        excluded = {
            "run",
            "track",
            "time_seconds",
            "frame_indices",
            "positions_mm",
            "positions_px",
            "sample_valid",
            "sample_reason_code",
            "transition_valid",
            "transition_reason_code",
        }
        return tuple(sorted(key for key in self.source_paths if key not in excluded))

    def series_for(self, analysis_id: str) -> tuple[str, ...]:
        if analysis_id == "speed":
            return tuple(
                key
                for key in self.available_series
                if key.startswith(SPEED_SERIES_PREFIXES) or any(token in key for token in PATH_SERIES_TOKENS)
            )
        if analysis_id == "heading":
            return tuple(
                key for key in self.available_series if any(token in key for token in HEADING_SERIES_TOKENS)
            )
        return ()

    def default_series_for(self, analysis_id: str) -> tuple[str, ...]:
        available = set(self.series_for(analysis_id))
        if analysis_id == "speed":
            preferred = (
                "speed_smoothed_mm",
                "speed_filtered_mm",
                "speed_raw_mm",
                "speed_averaged_mm",
                "speed_smoothed_px",
                "speed_filtered_px",
                "speed_raw_px",
                "speed_averaged_px",
            )
            return tuple(name for name in preferred if name in available)[:1]
        if analysis_id == "heading":
            preferred = (
                "smoothed_heading_degrees",
                "angular_speed_smoothed_deg_s",
                "angular_velocity_smoothed_deg_s",
                "delta_heading_smoothed_degrees",
            )
            return tuple(name for name in preferred if name in available)[:2]
        return ()

    def available_analysis_ids(self) -> tuple[str, ...]:
        """Detect analysis capabilities from array and lineage metadata."""

        if self._available_analysis_ids_cache is not None:
            return self._available_analysis_ids_cache

        available: list[str] = []
        if self.series_for("speed"):
            available.append("speed")
        if self.series_for("heading"):
            available.append("heading")
        if "positions_mm" in self.source_paths or "positions_px" in self.source_paths:
            available.append("position")
        try:
            if self._swim_bout_selection() is not None:
                available.append("swim_bouts")
        except Exception:
            pass
        try:
            if discover_eye_angle_run_options(self.zarr_path):
                available.append("eye_angles")
        except Exception:
            pass
        try:
            if self.baseline_interval() is not None and (
                "positions_mm" in self.source_paths or "positions_px" in self.source_paths
            ):
                available.append("baseline")
        except Exception:
            pass
        self._available_analysis_ids_cache = tuple(available)
        return self._available_analysis_ids_cache

    def _swim_bout_selection(self) -> tuple[Any, Any] | None:
        if self._swim_bout_selection_cache is not None:
            return self._swim_bout_selection_cache
        root = self._root()
        track_run_name = _normal_path(self.option.run_path).split("/")[-1]
        candidates = discover_swim_bout_candidates(
            root,
            track_run_name=track_run_name,
            track_id=self.track_id,
            include_bout_counts=False,
        )
        if not candidates:
            return None
        candidate = candidates[0]
        signal = next((item for item in candidate.signals if item.is_default), candidate.signals[0])
        self._swim_bout_selection_cache = (candidate, signal)
        return self._swim_bout_selection_cache

    def _swim_bout_events(self) -> Any:
        if self._swim_bout_events_cache is not None:
            return self._swim_bout_events_cache
        selection = self._swim_bout_selection()
        if selection is None:
            raise SwimBoutIOError("No lineage-compatible swim-bout run is available")
        candidate, signal = selection
        self._swim_bout_events_cache = load_swim_bout_events(
            self._root(),
            candidate=candidate,
            signal=signal,
        )
        return self._swim_bout_events_cache

    def _root(self):
        return open_zarr_root(self.zarr_path, mode="r")

    def _array(self, root: Any, key: str):
        path = self.source_paths.get(key)
        if not path:
            return None
        try:
            return root[path]
        except Exception:
            return None

    def time_bounds(self) -> tuple[float, float]:
        if self._time_seconds_cache is not None:
            return _finite_bounds(self._time_seconds_cache)
        root = self._root()
        array = self._array(root, "time_seconds")
        if array is None or int(array.shape[0]) == 0:
            return 0.0, 0.0
        first = float(np.asarray(array[0]).reshape(-1)[0])
        last = float(np.asarray(array[int(array.shape[0]) - 1]).reshape(-1)[0])
        if not np.isfinite(first) or not np.isfinite(last):
            return _finite_bounds(np.asarray(array[:], dtype=np.float64))
        return (min(first, last), max(first, last))

    def _row_projection(
        self,
        root: Any,
        *,
        start_s: float | None,
        stop_s: float | None,
    ) -> tuple[np.ndarray, slice]:
        time_array = self._array(root, "time_seconds")
        if time_array is None:
            raise ValueError("Track-kinematics spec does not resolve time_seconds")
        # Only the coordinate is materialized to resolve a chunk-friendly
        # contiguous slice. Payload arrays are read with that slice below.
        if self._time_seconds_cache is None:
            self._time_seconds_cache = np.asarray(time_array[:], dtype=np.float64).reshape(-1)
        times = self._time_seconds_cache
        if times.size == 0:
            return times, slice(0, 0)
        lo = float(start_s) if start_s is not None else float(times[0])
        hi = float(stop_s) if stop_s is not None else float(times[-1])
        if hi < lo:
            lo, hi = hi, lo
        finite_monotonic = np.isfinite(times).all() and bool(np.all(np.diff(times) >= 0))
        if finite_monotonic:
            start_index = int(np.searchsorted(times, lo, side="left"))
            stop_index = int(np.searchsorted(times, hi, side="right"))
        else:
            selected = np.flatnonzero(np.isfinite(times) & (times >= lo) & (times <= hi))
            if selected.size == 0:
                return times[:0], slice(0, 0)
            start_index = int(selected[0])
            stop_index = int(selected[-1]) + 1
        selected_slice = slice(start_index, stop_index)
        return times[selected_slice], selected_slice

    def project_timeseries(
        self,
        analysis_id: str,
        *,
        start_s: float | None = None,
        stop_s: float | None = None,
        series_keys: Sequence[str] | None = None,
    ) -> CoreBehaviorProjection:
        started = time.perf_counter()
        root = self._root()
        times, row_slice = self._row_projection(root, start_s=start_s, stop_s=stop_s)
        columns: dict[str, Any] = {"time_s": times}
        loaded_paths: list[str] = [self.source_paths["time_seconds"]]
        frame_array = self._array(root, "frame_indices")
        if frame_array is not None:
            frame_values = np.asarray(frame_array[row_slice], dtype=np.int64).reshape(-1)
            if frame_values.shape[0] == times.shape[0]:
                columns["frame_index"] = frame_values
                loaded_paths.append(self.source_paths["frame_indices"])
        for key in tuple(series_keys) if series_keys is not None else self.series_for(analysis_id):
            array = self._array(root, key)
            if array is None:
                continue
            values = np.asarray(array[row_slice])
            if values.ndim != 1 or values.shape[0] != times.shape[0]:
                continue
            columns[key] = values.astype(np.float64, copy=False)
            loaded_paths.append(self.source_paths[key])
        frame = pl.DataFrame(columns)
        bounds = _finite_bounds(times)
        return CoreBehaviorProjection(
            analysis_id=analysis_id,
            frame=frame.lazy(),
            columns=tuple(frame.columns),
            source_paths=tuple(loaded_paths),
            start_s=bounds[0],
            stop_s=bounds[1],
            row_count=frame.height,
            load_duration_ms=(time.perf_counter() - started) * 1000.0,
            note="Deferred Zarr projection; Polars transformations are lazy after array read.",
        )

    def project_positions(
        self,
        *,
        start_s: float | None = None,
        stop_s: float | None = None,
        analysis_id: str = "position",
    ) -> CoreBehaviorProjection:
        started = time.perf_counter()
        root = self._root()
        times, row_slice = self._row_projection(root, start_s=start_s, stop_s=stop_s)
        position_key = "positions_mm" if "positions_mm" in self.source_paths else "positions_px"
        position_array = self._array(root, position_key)
        if position_array is None:
            raise ValueError("Track-kinematics spec has no projected position array")
        positions = np.asarray(position_array[row_slice], dtype=np.float64)
        if positions.ndim != 2 or positions.shape[1] < 2 or positions.shape[0] != times.shape[0]:
            raise ValueError("Projected position array must have shape (time, >=2)")
        unit = "mm" if position_key.endswith("_mm") else "px"
        columns: dict[str, Any] = {
            "time_s": times,
            "x": positions[:, 0],
            "y": positions[:, 1],
            "unit": np.full(times.shape[0], unit, dtype=object),
        }
        loaded_paths = [self.source_paths["time_seconds"], self.source_paths[position_key]]
        frame_array = self._array(root, "frame_indices")
        if frame_array is not None:
            frame_values = np.asarray(frame_array[row_slice], dtype=np.int64).reshape(-1)
            if frame_values.shape[0] == times.shape[0]:
                columns["frame_index"] = frame_values
                loaded_paths.append(self.source_paths["frame_indices"])
        frame = pl.DataFrame(columns)
        bounds = _finite_bounds(times)
        return CoreBehaviorProjection(
            analysis_id=analysis_id,
            frame=frame.lazy(),
            columns=tuple(frame.columns),
            source_paths=tuple(loaded_paths),
            start_s=bounds[0],
            stop_s=bounds[1],
            row_count=frame.height,
            load_duration_ms=(time.perf_counter() - started) * 1000.0,
            note="Deferred position projection from Zarr; Polars transformations are lazy after array read.",
        )

    def project_swim_bouts(
        self,
        *,
        start_s: float | None = None,
        stop_s: float | None = None,
    ) -> CoreBehaviorProjection:
        started = time.perf_counter()
        events = self._swim_bout_events()
        lazy = _structured_lazy(events.bouts)
        schema = lazy.collect_schema()
        aliases: list[pl.Expr] = []
        if "start_time_s" in schema and "start_s" not in schema:
            aliases.append(pl.col("start_time_s").alias("start_s"))
        if "end_time_s" in schema and "end_s" not in schema:
            aliases.append(pl.col("end_time_s").alias("end_s"))
        if aliases:
            lazy = lazy.with_columns(aliases)
            schema = lazy.collect_schema()
        if {"start_s", "end_s"}.issubset(schema):
            if start_s is not None:
                lazy = lazy.filter(pl.col("end_s") >= float(start_s))
            if stop_s is not None:
                lazy = lazy.filter(pl.col("start_s") <= float(stop_s))
        bounds_columns = [name for name in ("start_s", "end_s", "peak_time_s") if name in schema]
        if bounds_columns:
            bounds_row = lazy.select(
                pl.min_horizontal(*[pl.col(name) for name in bounds_columns]).min().alias("start"),
                pl.max_horizontal(*[pl.col(name) for name in bounds_columns]).max().alias("stop"),
            ).collect().row(0)
            bounds = (float(bounds_row[0] or 0.0), float(bounds_row[1] or 0.0))
        else:
            bounds = (0.0, 0.0)
        row_count = int(lazy.select(pl.len()).collect().item())
        signal_prefix = str(events.signal.speed_level or "").strip()
        speed_candidates = [
            name
            for name in (
                f"{signal_prefix}_mm",
                f"{signal_prefix}_px",
                "speed_smoothed_mm",
                "speed_filtered_mm",
                "speed_raw_mm",
                "speed_averaged_mm",
                "speed_smoothed_px",
                "speed_filtered_px",
                "speed_raw_px",
                "speed_averaged_px",
            )
            if name in self.source_paths and _is_physical_speed_column(name)
        ]
        selected_speed = speed_candidates[0] if speed_candidates else None
        speed = self.project_timeseries(
            "speed",
            start_s=start_s,
            stop_s=stop_s,
            series_keys=(selected_speed,) if selected_speed is not None else (),
        )
        related_frames = (
            {"speed_trace": speed.frame.select(["time_s", selected_speed])}
            if selected_speed is not None
            else {}
        )
        return CoreBehaviorProjection(
            analysis_id="swim_bouts",
            frame=lazy,
            columns=tuple(schema.names()),
            source_paths=tuple(dict.fromkeys((events.level_path, *speed.source_paths))),
            start_s=bounds[0],
            stop_s=bounds[1],
            row_count=row_count,
            load_duration_ms=(time.perf_counter() - started) * 1000.0,
            note=(
                f"Persisted `{events.signal.speed_level}` bout segmentation from `{events.run_name}`; "
                "the lineage-compatible speed trace and downstream Polars queries are read-only."
            ),
            related_frames=related_frames,
        )

    def project_eye_angles(
        self,
        *,
        start_s: float | None = None,
        stop_s: float | None = None,
    ) -> CoreBehaviorProjection:
        started = time.perf_counter()
        options = discover_eye_angle_run_options(self.zarr_path)
        if not options:
            raise ValueError("No eye-angle run is available")
        selected = options[0]
        payload = load_eye_angle_timeseries_data(
            self.zarr_path,
            run_name=selected.run_name,
            prefer_frame=True,
        )
        frame = pl.from_pandas(payload.dataframe)
        if "time_s" in frame.columns:
            if start_s is not None:
                frame = frame.filter(pl.col("time_s") >= float(start_s))
            if stop_s is not None:
                frame = frame.filter(pl.col("time_s") <= float(stop_s))
            bounds = _finite_bounds(frame["time_s"].to_numpy())
        else:
            bounds = (0.0, 0.0)
        return CoreBehaviorProjection(
            analysis_id="eye_angles",
            frame=frame.lazy(),
            columns=tuple(frame.columns),
            source_paths=(payload.run_path,),
            start_s=bounds[0],
            stop_s=bounds[1],
            row_count=frame.height,
            load_duration_ms=(time.perf_counter() - started) * 1000.0,
            note="Eye arrays loaded only after selection; downstream Polars transformations are lazy.",
        )

    def baseline_interval(self) -> BaselineInterval | None:
        """Resolve a canonical pre period from chaser window metadata only."""

        root = self._root()
        parent = root.get("analysis/chaser_distance_runs")
        if parent is None:
            return None
        latest = str(parent.attrs.get("latest_complete") or parent.attrs.get("latest") or "")
        run_names = list(parent.group_keys())
        if latest in run_names:
            run_names = [latest, *[name for name in run_names if name != latest]]
        for run_name in run_names:
            visualizations = parent[run_name].get("visualizations")
            if visualizations is None:
                continue
            for artifact_name in visualizations.group_keys():
                artifact = visualizations[artifact_name]
                if "spec_json" not in artifact:
                    continue
                try:
                    import json

                    spec = json.loads(np.asarray(artifact["spec_json"][:], dtype=np.uint8).tobytes())
                except Exception:
                    continue
                paths = spec.get("source_paths") if isinstance(spec, Mapping) else None
                if not isinstance(paths, Mapping):
                    continue
                label_path = paths.get("epoch_label_bytes") or paths.get("detection_occupancy_windows_label_bytes")
                start_path = paths.get("epoch_start_frame") or paths.get("detection_occupancy_windows_start_frame")
                end_path = paths.get("epoch_end_frame") or paths.get("detection_occupancy_windows_end_frame")
                if not label_path or not start_path or not end_path:
                    continue
                try:
                    labels_raw = np.asarray(root[_normal_path(label_path)][:])
                    starts = np.asarray(root[_normal_path(start_path)][:], dtype=np.int64).reshape(-1)
                    ends = np.asarray(root[_normal_path(end_path)][:], dtype=np.int64).reshape(-1)
                except Exception:
                    continue
                labels = [decode_null_terminated_text(value) for value in labels_raw]
                fps = float(spec.get("fps") or 1.0)
                for index, label in enumerate(labels[: min(len(starts), len(ends))]):
                    if is_baseline_label(label):
                        return BaselineInterval(
                            label=label,
                            start_s=float(starts[index]) / fps,
                            stop_s=float(ends[index] + 1) / fps,
                        )
        return None


def load_core_behavior_projection(
    source: CoreBehaviorSource,
    analysis_id: str,
    *,
    start_s: float | None = None,
    stop_s: float | None = None,
    series_keys: Sequence[str] | None = None,
) -> CoreBehaviorProjection:
    if analysis_id in {"speed", "heading"}:
        return source.project_timeseries(
            analysis_id,
            start_s=start_s,
            stop_s=stop_s,
            series_keys=series_keys,
        )
    if analysis_id == "position":
        return source.project_positions(start_s=start_s, stop_s=stop_s)
    if analysis_id == "swim_bouts":
        return source.project_swim_bouts(start_s=start_s, stop_s=stop_s)
    if analysis_id == "eye_angles":
        return source.project_eye_angles(start_s=start_s, stop_s=stop_s)
    if analysis_id == "baseline":
        interval = source.baseline_interval()
        if interval is None:
            raise ValueError("No canonical pre-period window is available in this recording")
        position = source.project_positions(
            start_s=interval.start_s,
            stop_s=interval.stop_s,
            analysis_id="baseline",
        )
        speed = source.project_timeseries(
            "speed",
            start_s=interval.start_s,
            stop_s=interval.stop_s,
        )
        speed_columns_available = [
            name
            for name in speed.columns
            if _is_physical_speed_column(name)
        ]
        preferred_speed_columns = (
            "speed_smoothed_mm",
            "speed_filtered_mm",
            "speed_raw_mm",
            "speed_averaged_mm",
            "speed_smoothed_px",
            "speed_filtered_px",
            "speed_raw_px",
            "speed_averaged_px",
        )
        speed_columns = [
            name for name in preferred_speed_columns if name in speed_columns_available
        ][:4]
        joined = position.frame
        if speed_columns:
            joined = joined.join(
                speed.frame.select(["time_s", *speed_columns]),
                on="time_s",
                how="left",
            )
        schema = joined.collect_schema()
        return CoreBehaviorProjection(
            analysis_id="baseline",
            frame=joined,
            columns=tuple(schema.names()),
            source_paths=tuple(dict.fromkeys((*position.source_paths, *speed.source_paths))),
            start_s=position.start_s,
            stop_s=position.stop_s,
            row_count=position.row_count,
            load_duration_ms=position.load_duration_ms + speed.load_duration_ms,
            note=(
                f"Canonical pre-period `{interval.label}` resolved from persisted epoch metadata; "
                "trajectory and activity are descriptive viewer projections."
            ),
        )
    raise ValueError(f"Unsupported core-behavior analysis: {analysis_id!r}")


def collect_projection(
    projection: CoreBehaviorProjection,
    *,
    columns: Iterable[str] | None = None,
    start_s: float | None = None,
    stop_s: float | None = None,
) -> pl.DataFrame:
    """Collect a bounded view from a projected LazyFrame."""

    query = projection.frame
    schema = query.collect_schema()
    if "time_s" in schema and (start_s is not None or stop_s is not None):
        if start_s is not None:
            query = query.filter(pl.col("time_s") >= float(start_s))
        if stop_s is not None:
            query = query.filter(pl.col("time_s") <= float(stop_s))
    if columns is not None:
        selected = [name for name in columns if name in query.collect_schema()]
        query = query.select(selected)
    return query.collect()


def _decimate_for_display(
    frame: pl.DataFrame,
    *,
    trace_count: int,
    max_total_values: int = 60000,
) -> pl.DataFrame:
    """Bound serialized plotting payload without changing persisted data."""

    if frame.is_empty():
        return frame
    row_budget = max(1000, int(max_total_values) // max(1, int(trace_count)))
    if frame.height <= row_budget:
        return frame
    stride = max(1, int(np.ceil(frame.height / float(row_budget))))
    return (
        frame.lazy()
        .with_row_index("_display_row")
        .filter((pl.col("_display_row") % stride) == 0)
        .drop("_display_row")
        .collect()
    )


def build_core_behavior_output(
    mo: Any,
    go: Any,
    px: Any,
    *,
    projection: CoreBehaviorProjection,
) -> Any:
    """Render one selected core analysis without activating sibling analyses."""

    frame = collect_projection(projection)
    header = mo.hstack(
        [
            mo.stat(label="Rows projected", value=f"{projection.row_count:,}"),
            mo.stat(label="Columns", value=f"{len(projection.columns):,}"),
            mo.stat(label="Zarr read ms", value=f"{projection.load_duration_ms:.1f}"),
        ]
    )
    source_note = mo.md(
        f"{projection.note} Source arrays: `{', '.join(projection.source_paths) or 'none'}`"
    )
    if projection.analysis_id in {"speed", "heading", "eye_angles"}:
        value_columns = [
            name
            for name in frame.columns
            if name not in {"time_s", "frame_index", "row_index"}
            and frame.schema[name].is_numeric()
            and frame.schema[name] != pl.Boolean
        ]
        display_frame = _decimate_for_display(frame, trace_count=len(value_columns))
        figure = go.Figure()
        for column in value_columns:
            figure.add_trace(
                go.Scattergl(
                    x=display_frame["time_s"],
                    y=display_frame[column],
                    mode="lines",
                    name=column,
                )
            )
        figure.update_layout(
            title=(
                "Speed and path traces"
                if projection.analysis_id == "speed"
                else "Eye angles and convergence"
                if projection.analysis_id == "eye_angles"
                else "Heading and turning traces"
            ),
            xaxis_title="Time (s)",
            yaxis_title="Value",
            height=500,
            margin=dict(l=55, r=25, t=60, b=50),
        )
        body = figure if value_columns else mo.md("No compatible series are present in this run.")
    elif projection.analysis_id in {"position", "baseline"}:
        if frame.height:
            display_frame = _decimate_for_display(
                frame,
                trace_count=3,
                max_total_values=75000,
            )
            figure = px.scatter(
                display_frame.to_pandas(),
                x="x",
                y="y",
                color="time_s",
                render_mode="webgl",
                title=(
                    "Pre-period trajectory (descriptive viewer projection)"
                    if projection.analysis_id == "baseline"
                    else "Trajectory"
                ),
                labels={"time_s": "Time (s)"},
            )
            figure.update_yaxes(scaleanchor="x", scaleratio=1)
            figure.update_layout(height=600)
            if projection.analysis_id == "baseline":
                speed_column = next(
                    (
                        name
                        for name in frame.columns
                        if _is_physical_speed_column(name)
                    ),
                    None,
                )
                pieces: list[Any] = [figure]
                if speed_column is not None:
                    speed_fig = go.Figure(
                        go.Scattergl(
                            x=display_frame["time_s"],
                            y=display_frame[speed_column],
                            mode="lines",
                            name=speed_column,
                        )
                    )
                    speed_fig.update_layout(
                        title="Pre-period activity trace",
                        xaxis_title="Time (s)",
                        yaxis_title=speed_column,
                        height=360,
                    )
                    summary = projection.frame.select(
                        pl.col(speed_column).drop_nans().drop_nulls().count().alias("finite_speed_samples"),
                        pl.col(speed_column).drop_nans().drop_nulls().mean().alias("mean_speed"),
                        pl.col(speed_column).drop_nans().drop_nulls().median().alias("median_speed"),
                        pl.col(speed_column).drop_nans().drop_nulls().max().alias("max_speed"),
                    ).collect()
                    pieces.extend(
                        [speed_fig, mo.md("### Descriptive pre-period activity"), mo.ui.table(summary)]
                    )
                body = mo.vstack(pieces)
            else:
                body = figure
        else:
            body = mo.md("No position rows fall inside the selected interval.")
    elif projection.analysis_id == "swim_bouts":
        duration_column = next(
            (name for name in ("duration_s", "bout_duration_s") if name in frame.columns),
            None,
        )
        pieces: list[Any] = []
        speed_lazy = projection.related_frames.get("speed_trace")
        speed_frame = speed_lazy.collect() if speed_lazy is not None else pl.DataFrame()
        if speed_frame.height > 50000:
            stride = max(1, int(np.ceil(speed_frame.height / 50000.0)))
            speed_frame = (
                speed_frame.lazy()
                .with_row_index("_row")
                .filter((pl.col("_row") % stride) == 0)
                .drop("_row")
                .collect()
            )
        speed_column = next((name for name in speed_frame.columns if name != "time_s"), None)
        segmentation_figure = go.Figure()
        if speed_column is not None and speed_frame.height:
            segmentation_figure.add_trace(
                go.Scattergl(
                    x=speed_frame["time_s"],
                    y=speed_frame[speed_column],
                    mode="lines",
                    name=speed_column,
                    line=dict(color="#334155", width=1.25),
                )
            )
        if frame.height and {"start_s", "end_s"}.issubset(frame.columns):
            starts = frame["start_s"].to_numpy().astype(np.float64, copy=False)
            stops = frame["end_s"].to_numpy().astype(np.float64, copy=False)
            widths = stops - starts
            valid = np.isfinite(starts) & np.isfinite(stops) & (widths > 0)
            bout_column = next((name for name in ("bout_id", "source_bout_id") if name in frame.columns), None)
            bout_ids = (
                frame[bout_column].to_numpy()
                if bout_column is not None
                else np.arange(frame.height, dtype=np.int64)
            )
            if valid.any():
                segmentation_figure.add_trace(
                    go.Bar(
                        x=(starts[valid] + stops[valid]) / 2.0,
                        y=np.ones(int(np.count_nonzero(valid)), dtype=np.float64),
                        width=widths[valid],
                        base=np.zeros(int(np.count_nonzero(valid)), dtype=np.float64),
                        yaxis="y2",
                        name="Persisted swim bouts",
                        marker=dict(color="#f59e0b", line=dict(width=0)),
                        opacity=0.24,
                        customdata=np.column_stack([bout_ids[valid], starts[valid], stops[valid]]),
                        hovertemplate=(
                            "bout=%{customdata[0]}<br>"
                            "start=%{customdata[1]:.3f}s<br>"
                            "end=%{customdata[2]:.3f}s<extra></extra>"
                        ),
                    )
                )
        if segmentation_figure.data:
            segmentation_figure.update_layout(
                title="Speed trace with persisted swim-bout segmentation",
                xaxis_title="Time (s)",
                yaxis_title=speed_column or "Speed",
                yaxis2=dict(
                    overlaying="y",
                    side="right",
                    range=[0, 1],
                    showgrid=False,
                    showticklabels=False,
                    title="Bout intervals",
                ),
                barmode="overlay",
                height=480,
                margin=dict(l=55, r=55, t=60, b=50),
                legend=dict(orientation="h", yanchor="top", y=-0.14, xanchor="left", x=0.0),
            )
            pieces.append(segmentation_figure)
        if duration_column and frame.height:
            pieces.append(
                px.histogram(
                    frame.to_pandas(),
                    x=duration_column,
                    nbins=40,
                    title="Swim-bout duration distribution",
                )
            )
        pieces.extend([mo.md("### Persisted bout rows"), mo.ui.table(frame, selection=None, page_size=12)])
        body = mo.vstack(pieces)
    else:
        body = mo.md(f"Unsupported analysis `{projection.analysis_id}`")
    return mo.vstack([header, source_note, body])
