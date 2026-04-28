#!/usr/bin/env python3
"""Marimo app for exploring Palette track-kinematics visualization specs.

Run after installing optional UI dependencies:

    scripts/py -m marimo run apps/marimo/track_kinematics_explorer.py -- \
      --zarr-path /path/to/archive.zarr

Optional selectors:

    --run-path analysis/track_kinematics_runs/offline/<run>
    --swim-bout-run <run-name>
    --speed-level filtered
    --performance-log /tmp/palette_track_kinematics_explorer_perf.jsonl
"""

import marimo

__generated_with = "0.23.3"
app = marimo.App(width="full")


@app.cell
def _():
    import json
    import marimo as mo
    import numpy as np
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
    import time

    from fisheye.visualization.interactive_track_kinematics import (
        DEFAULT_INTERACTIVE_ARTIFACT,
        bout_kinematics_records_to_dataframe,
        discover_bout_kinematics_run_options,
        discover_swim_bout_run_options,
        discover_track_kinematics_run_options,
        load_bout_kinematics_records,
        load_track_kinematics_interactive_data,
        to_inter_bout_interval_dataframe,
        to_position_dataframe,
        to_swim_bout_dataframe,
        to_timeseries_dataframe,
        to_validity_span_dataframe,
    )

    def add_interval_bar_overlay(
        fig,
        *,
        starts,
        ends,
        name,
        color,
        opacity,
        labels=None,
        hovertemplate=None,
        hoverinfo=None,
    ):
        starts = np.asarray(starts, dtype=float)
        ends = np.asarray(ends, dtype=float)
        widths = ends - starts
        valid = widths > 0
        if not valid.any():
            return "none"
        starts = starts[valid]
        widths = widths[valid]
        centers = starts + (widths / 2.0)
        customdata = None
        if labels is not None:
            customdata = np.asarray(labels, dtype=object)[valid]
        fig.add_trace(
            go.Bar(
                x=centers,
                y=np.ones_like(centers),
                width=widths,
                base=np.zeros_like(centers),
                yaxis="y2",
                marker=dict(color=color, line=dict(width=0)),
                opacity=opacity,
                customdata=customdata,
                hovertemplate=hovertemplate,
                hoverinfo=hoverinfo,
                name=name,
            )
        )
        return "bar_trace"

    def apply_full_width_timeseries_layout(
        fig,
        *,
        title,
        yaxis_title,
        height=420,
    ):
        fig.update_layout(
            title=title,
            xaxis_title="Time (s)",
            yaxis_title=yaxis_title,
            hovermode="x unified",
            height=int(height),
            margin=dict(l=56, r=20, t=56, b=110),
            barmode="overlay",
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.22,
                xanchor="left",
                x=0.0,
            ),
            yaxis2=dict(
                overlaying="y",
                range=[0, 1],
                visible=False,
                fixedrange=True,
            ),
        )

    return (
        DEFAULT_INTERACTIVE_ARTIFACT,
        add_interval_bar_overlay,
        apply_full_width_timeseries_layout,
        bout_kinematics_records_to_dataframe,
        discover_bout_kinematics_run_options,
        discover_swim_bout_run_options,
        discover_track_kinematics_run_options,
        go,
        json,
        load_bout_kinematics_records,
        load_track_kinematics_interactive_data,
        mo,
        np,
        pd,
        px,
        time,
        to_inter_bout_interval_dataframe,
        to_position_dataframe,
        to_swim_bout_dataframe,
        to_timeseries_dataframe,
        to_validity_span_dataframe,
    )


@app.cell
def _(
    DEFAULT_INTERACTIVE_ARTIFACT,
    discover_track_kinematics_run_options,
    json,
    mo,
    time,
):
    from datetime import datetime, timezone
    from pathlib import Path
    import uuid

    cli_args = mo.cli_args()
    zarr_path_raw = cli_args.get("zarr-path")
    artifact = str(cli_args.get("artifact", DEFAULT_INTERACTIVE_ARTIFACT))
    initial_run_path = cli_args.get("run-path")
    initial_swim_bout_run = cli_args.get("swim-bout-run")
    initial_speed_level = cli_args.get("speed-level")
    performance_log_raw = cli_args.get(
        "performance-log",
        "/tmp/palette_track_kinematics_explorer_perf.jsonl",
    )
    performance_session_id = uuid.uuid4().hex
    performance_log_path = (
        None
        if str(performance_log_raw).strip().lower() in {"", "0", "false", "none", "off"}
        else Path(str(performance_log_raw)).expanduser()
    )

    def _jsonable(value):
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {str(key): _jsonable(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [_jsonable(item) for item in value]
        return str(value)

    def write_perf_event(phase, duration_s=None, **fields):
        if performance_log_path is None:
            return
        payload = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "session_id": performance_session_id,
            "app": "track_kinematics_explorer",
            "phase": str(phase),
        }
        if duration_s is not None:
            payload["duration_ms"] = round(float(duration_s) * 1000.0, 3)
        payload.update({str(key): _jsonable(value) for key, value in fields.items()})
        try:
            performance_log_path.parent.mkdir(parents=True, exist_ok=True)
            with performance_log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, sort_keys=True) + "\n")
        except OSError:
            pass

    if not zarr_path_raw:
        raise ValueError(
            "Required CLI arg is missing. Run with: "
            "scripts/py -m marimo run apps/marimo/track_kinematics_explorer.py -- "
            "--zarr-path <archive.zarr>"
        )
    zarr_path = Path(str(zarr_path_raw))
    _discovery_t0 = time.perf_counter()
    track_options = discover_track_kinematics_run_options(
        zarr_path,
        artifact_name=artifact,
    )
    write_perf_event(
        "discover_track_options",
        time.perf_counter() - _discovery_t0,
        zarr_path=zarr_path,
        artifact=artifact,
        n_track_options=len(track_options),
    )
    if not track_options:
        raise ValueError(
            "No track-kinematics interactive artifacts were found. "
            "Run plot_track_kinematics with --write-zarr-artifacts first."
        )
    return (
        artifact,
        initial_run_path,
        initial_speed_level,
        initial_swim_bout_run,
        performance_log_path,
        performance_session_id,
        track_options,
        write_perf_event,
        zarr_path,
    )


@app.cell
def _(initial_run_path, mo, track_options):
    preferred_default_run_name = "tk_hyst4_low2_s005"

    def _matches_initial_run(option):
        if initial_run_path is None:
            return False
        wanted = str(initial_run_path).strip("/")
        return option.run_path.strip("/") == wanted or option.run_name == wanted.split("/")[-1]

    track_label_to_option = {option.label: option for option in track_options}
    _selected_default = next(
        (option.label for option in track_options if _matches_initial_run(option)),
        next(
            (
                option.label
                for option in track_options
                if option.run_name == preferred_default_run_name
            ),
            track_options[0].label,
        ),
    )
    track_picker = mo.ui.dropdown(
        options=list(track_label_to_option),
        value=_selected_default,
        label="Track kinematics run",
    )
    track_picker
    return track_label_to_option, track_picker


@app.cell
def _(
    discover_swim_bout_run_options,
    time,
    track_label_to_option,
    track_picker,
    write_perf_event,
    zarr_path,
):
    selected_track = track_label_to_option[track_picker.value]
    _discovery_t0 = time.perf_counter()
    swim_bout_options = discover_swim_bout_run_options(
        zarr_path,
        track_run_path=selected_track.run_path,
        track_id=selected_track.track_id,
    )
    write_perf_event(
        "discover_swim_bout_options",
        time.perf_counter() - _discovery_t0,
        zarr_path=zarr_path,
        selected_track_run=selected_track.run_path,
        selected_track_id=selected_track.track_id,
        n_swim_bout_options=len(swim_bout_options),
    )
    return selected_track, swim_bout_options


@app.cell
def _(initial_speed_level, initial_swim_bout_run, mo, swim_bout_options):
    _none_label = "No swim-bout overlay"
    swim_label_to_option = {_none_label: None}
    swim_label_to_option.update({_option.label: _option for _option in swim_bout_options})
    _initial_level = str(initial_speed_level).removeprefix("speed_") if initial_speed_level is not None else None

    def _matches_initial_bout(_option):
        if _option is None or initial_swim_bout_run is None:
            return False
        if _option.run_name != str(initial_swim_bout_run):
            return False
        return _initial_level is None or _option.speed_level == _initial_level

    _selected_default = next(
        (
            _label
            for _label, _option in swim_label_to_option.items()
            if _matches_initial_bout(_option)
        ),
        swim_bout_options[0].label if swim_bout_options else _none_label,
    )
    swim_bout_picker = mo.ui.dropdown(
        options=list(swim_label_to_option),
        value=_selected_default,
        label="Derived swim-bout candidate",
    )
    swim_bout_picker
    return swim_bout_picker, swim_label_to_option


@app.cell
def _(initial_speed_level, swim_bout_picker, swim_label_to_option):
    selected_swim_bout = swim_label_to_option[swim_bout_picker.value]
    selected_speed_level = (
        str(initial_speed_level)
        if selected_swim_bout is None and initial_speed_level is not None
        else selected_swim_bout.speed_level
        if selected_swim_bout is not None
        else None
    )
    return selected_speed_level, selected_swim_bout


@app.cell
def _(
    discover_bout_kinematics_run_options,
    selected_speed_level,
    selected_swim_bout,
    selected_track,
    time,
    write_perf_event,
    zarr_path,
):
    _discovery_t0 = time.perf_counter()
    bout_kinematics_options = discover_bout_kinematics_run_options(
        zarr_path,
        track_run_path=selected_track.run_path,
        track_id=selected_track.track_id,
        swim_bout_run=selected_swim_bout.run_name if selected_swim_bout is not None else None,
        speed_level=selected_speed_level,
    )
    write_perf_event(
        "discover_bout_kinematics_options",
        time.perf_counter() - _discovery_t0,
        zarr_path=zarr_path,
        selected_track_run=selected_track.run_path,
        selected_track_id=selected_track.track_id,
        selected_swim_bout_run=selected_swim_bout.run_name if selected_swim_bout is not None else None,
        selected_speed_level=selected_speed_level,
        n_bout_kinematics_options=len(bout_kinematics_options),
    )
    return (bout_kinematics_options,)


@app.cell
def _(bout_kinematics_options, mo):
    _none_label = "No bout-kinematics overlay"
    bout_kinematics_label_to_option = {_none_label: None}
    bout_kinematics_label_to_option.update({_option.label: _option for _option in bout_kinematics_options})
    _selected_default = bout_kinematics_options[0].label if bout_kinematics_options else _none_label
    bout_kinematics_picker = mo.ui.dropdown(
        options=list(bout_kinematics_label_to_option),
        value=_selected_default,
        label="Bout-kinematics candidate",
    )
    bout_kinematics_picker
    return bout_kinematics_label_to_option, bout_kinematics_picker


@app.cell
def _(
    bout_kinematics_label_to_option,
    bout_kinematics_picker,
    bout_kinematics_records_to_dataframe,
    load_bout_kinematics_records,
    time,
    write_perf_event,
    zarr_path,
):
    selected_bout_kinematics = bout_kinematics_label_to_option[bout_kinematics_picker.value]
    _load_t0 = time.perf_counter()
    if selected_bout_kinematics is None:
        bout_kinematics_df = bout_kinematics_records_to_dataframe({})
        bout_kinematics_attrs = {}
    else:
        _records_by_level, bout_kinematics_attrs = load_bout_kinematics_records(
            zarr_path,
            run_name=selected_bout_kinematics.run_name,
        )
        bout_kinematics_df = bout_kinematics_records_to_dataframe(_records_by_level)
    write_perf_event(
        "load_bout_kinematics",
        time.perf_counter() - _load_t0,
        selected_bout_kinematics_run=(
            selected_bout_kinematics.run_name if selected_bout_kinematics is not None else None
        ),
        n_bout_kinematics_rows=len(bout_kinematics_df),
    )
    return bout_kinematics_attrs, bout_kinematics_df, selected_bout_kinematics


@app.cell
def _(
    artifact,
    load_track_kinematics_interactive_data,
    selected_speed_level,
    selected_swim_bout,
    selected_track,
    time,
    write_perf_event,
    zarr_path,
):
    _load_t0 = time.perf_counter()
    data = load_track_kinematics_interactive_data(
        zarr_path,
        run_path=selected_track.run_path,
        artifact_name=artifact,
        swim_bout_run=selected_swim_bout.run_name if selected_swim_bout is not None else "none",
        speed_level=selected_speed_level,
    )
    write_perf_event(
        "load_interactive_data",
        time.perf_counter() - _load_t0,
        zarr_path=zarr_path,
        run_path=selected_track.run_path,
        track_id=selected_track.track_id,
        swim_bout_run=selected_swim_bout.run_name if selected_swim_bout is not None else None,
        speed_level=selected_speed_level,
        n_time_rows=int(data.time_seconds.shape[0]),
        n_series=len(data.series),
        n_position_rows=int(data.positions.shape[0]) if data.positions is not None else 0,
        n_swim_bouts=int(data.swim_bouts.shape[0]),
    )
    return (data,)


@app.cell
def _(
    bout_kinematics_attrs,
    bout_kinematics_df,
    data,
    selected_bout_kinematics,
    mo,
    pd,
    performance_log_path,
    performance_session_id,
    selected_speed_level,
    selected_swim_bout,
    selected_track,
):
    mo.md(
        f"""
        # Track Kinematics Explorer

        **Archive:** `{data.zarr_path}`

        **Run:** `{data.run_path}`

        **Track Selection:** `{selected_track.label}`

        **Artifact:** `{data.artifact_name}`

        **Renderer:** `{data.attrs.get("renderer", "unknown")}`

        **Swim Bout Overlay:** `{data.swim_bout_label or "none"}`

        **Selected Bout Candidate:** `{selected_swim_bout.label if selected_swim_bout else "none"}`

        **Bout Kinematics:** `{selected_bout_kinematics.label if selected_bout_kinematics else "none"}`

        **Speed Level:** `{selected_speed_level or "artifact/default"}`

        **Performance Log:** `{performance_log_path or "disabled"}`

        **Performance Session:** `{performance_session_id}`
        """
    )
    _selected_summary_rows = [
        {
            "surface": "track_kinematics",
            "field": "run",
            "value": selected_track.run_name,
        },
        {
            "surface": "track_kinematics",
            "field": "scope",
            "value": selected_track.run_scope,
        },
        {
            "surface": "track_kinematics",
            "field": "track_id",
            "value": selected_track.track_id,
        },
        {
            "surface": "track_kinematics",
            "field": "artifact",
            "value": data.artifact_name,
        },
        {
            "surface": "swim_bout",
            "field": "run",
            "value": selected_swim_bout.run_name if selected_swim_bout else "none",
        },
        {
            "surface": "swim_bout",
            "field": "speed_level",
            "value": selected_speed_level or "artifact/default",
        },
        {
            "surface": "swim_bout",
            "field": "n_bouts",
            "value": (
                selected_swim_bout.n_bouts_by_level.get(f"speed_{selected_swim_bout.speed_level}", 0)
                if selected_swim_bout
                else 0
            ),
        },
        {
            "surface": "swim_bout",
            "field": "threshold_mm_s",
            "value": selected_swim_bout.threshold_mm if selected_swim_bout else None,
        },
        {
            "surface": "swim_bout",
            "field": "exponential_tau_s",
            "value": selected_swim_bout.exponential_tau_s if selected_swim_bout else None,
        },
        {
            "surface": "swim_bout",
            "field": "exponential_source_level",
            "value": selected_swim_bout.exponential_source_level if selected_swim_bout else None,
        },
        {
            "surface": "swim_bout",
            "field": "method",
            "value": selected_swim_bout.detection_method if selected_swim_bout else "none",
        },
        {
            "surface": "bout_kinematics",
            "field": "run",
            "value": selected_bout_kinematics.run_name if selected_bout_kinematics else "none",
        },
        {
            "surface": "bout_kinematics",
            "field": "pre_post_mode",
            "value": selected_bout_kinematics.pre_post_mode if selected_bout_kinematics else "none",
        },
        {
            "surface": "bout_kinematics",
            "field": "default_heading_level",
            "value": selected_bout_kinematics.default_heading_level if selected_bout_kinematics else "none",
        },
        {
            "surface": "bout_kinematics",
            "field": "heading_levels",
            "value": ", ".join(selected_bout_kinematics.heading_levels) if selected_bout_kinematics else "none",
        },
        {
            "surface": "bout_kinematics",
            "field": "rows_loaded",
            "value": len(bout_kinematics_df),
        },
        {
            "surface": "bout_kinematics",
            "field": "schema_version",
            "value": bout_kinematics_attrs.get("schema_version", "none"),
        },
    ]
    selected_summary_df = pd.DataFrame(_selected_summary_rows)
    mo.vstack(
        [
            mo.md("### Selected Candidate Details"),
            mo.ui.table(selected_summary_df, selection=None),
        ]
    )
    return (selected_summary_df,)


@app.cell
def _(
    data,
    time,
    to_inter_bout_interval_dataframe,
    to_position_dataframe,
    to_swim_bout_dataframe,
    to_timeseries_dataframe,
    to_validity_span_dataframe,
    write_perf_event,
):
    _dataframe_t0 = time.perf_counter()
    timeseries_df = to_timeseries_dataframe(data)
    position_df = to_position_dataframe(data)
    swim_bout_df = to_swim_bout_dataframe(data)
    inter_bout_interval_df = to_inter_bout_interval_dataframe(data)
    validity_df = to_validity_span_dataframe(data)
    write_perf_event(
        "build_dataframes",
        time.perf_counter() - _dataframe_t0,
        run_path=data.run_path,
        n_timeseries_rows=len(timeseries_df),
        n_timeseries_columns=len(timeseries_df.columns),
        n_position_rows=len(position_df),
        n_swim_bout_rows=len(swim_bout_df),
        n_inter_bout_interval_rows=len(inter_bout_interval_df),
        n_validity_rows=len(validity_df),
    )
    return inter_bout_interval_df, position_df, swim_bout_df, timeseries_df, validity_df


@app.cell
def _(data, mo, np, selected_speed_level, selected_swim_bout, swim_bout_df, timeseries_df):
    _speed_columns = [
        _column
        for _column in timeseries_df.columns
        if ("speed" in _column.lower() or "detection_signal" in _column.lower())
        and timeseries_df[_column].notna().any()
    ]
    _level = str(selected_speed_level or "").strip()
    if _level.startswith("speed_"):
        _level = _level.removeprefix("speed_")
    _preferred_candidates = (
        ("detection_signal_mm_s", f"speed_{_level}_mm", f"speed_{_level}_px")
        if _level == "exponential"
        else (f"speed_{_level}_mm", f"speed_{_level}_px")
    )
    _preferred_columns = [
        _column
        for _column in _preferred_candidates
        if _level and _column in _speed_columns
    ]
    _fallback_columns = [
        _column
        for _column in ("speed_smoothed_mm", "speed_filtered_mm", "speed_raw_mm")
        if _column in _speed_columns
    ]
    _default_speed_columns = _preferred_columns[:1] or _fallback_columns[:1] or _speed_columns[:1]
    speed_series_picker = mo.ui.multiselect(
        options=_speed_columns,
        value=_default_speed_columns,
        label="Speed / detection traces",
    )
    _acceleration_columns = [
        _column
        for _column in timeseries_df.columns
        if "acceleration" in _column.lower()
        and timeseries_df[_column].notna().any()
    ]
    _default_acceleration_columns = [
        _column
        for _column in ("smoothed_acceleration_mm", "smoothed_acceleration_px", "acceleration_mm", "acceleration_px")
        if _column in _acceleration_columns
    ][:1]
    acceleration_series_picker = mo.ui.multiselect(
        options=_acceleration_columns,
        value=_default_acceleration_columns,
        label="Acceleration traces",
    )
    _turning_columns = [
        _column
        for _column in timeseries_df.columns
        if (
            "angular_velocity" in _column.lower()
            or "angular_speed" in _column.lower()
            or "delta_heading" in _column.lower()
        )
        and timeseries_df[_column].notna().any()
    ]
    _default_turning_columns = [
        _column
        for _column in (
            "angular_speed_smoothed_deg_s",
            "angular_velocity_smoothed_deg_s",
            "angular_speed_raw_deg_s",
            "angular_velocity_raw_deg_s",
            "angular_velocity_deg_s",
        )
        if _column in _turning_columns
    ][:2]
    turning_series_picker = mo.ui.multiselect(
        options=_turning_columns,
        value=_default_turning_columns,
        label="Turning / angular traces",
    )
    _max_time = float(timeseries_df["time_s"].max()) if len(timeseries_df) else 0.0
    time_window = mo.ui.range_slider(
        start=0.0,
        stop=_max_time,
        value=[0.0, _max_time],
        step=max(_max_time / 1000.0, 0.001),
        label="Time window (s)",
    )
    show_swim_bouts = mo.ui.checkbox(
        value=bool(data.swim_bout_label),
        label="Overlay swim bouts",
    )
    _interpolated_columns = {
        "core_start_time_s_interpolated",
        "core_end_time_s_interpolated",
        "core_start_time_interpolated_valid",
        "core_end_time_interpolated_valid",
    }
    _has_interpolated_boundaries = (
        len(swim_bout_df)
        and _interpolated_columns.issubset(swim_bout_df.columns)
        and (
            swim_bout_df["core_start_time_interpolated_valid"].astype(bool)
            & swim_bout_df["core_end_time_interpolated_valid"].astype(bool)
        ).any()
    )
    _peak_width_columns = {
        "peak_event_left_width_time_s",
        "peak_event_right_width_time_s",
    }
    _has_peak_width_boundaries = (
        len(swim_bout_df)
        and _peak_width_columns.issubset(swim_bout_df.columns)
        and (
            np.isfinite(swim_bout_df["peak_event_left_width_time_s"].to_numpy(dtype=float))
            & np.isfinite(swim_bout_df["peak_event_right_width_time_s"].to_numpy(dtype=float))
        ).any()
    )
    _boundary_options = ["sampled_frame_boundaries"]
    if _has_interpolated_boundaries:
        _boundary_options.append("interpolated_core_threshold")
    if _has_peak_width_boundaries:
        _boundary_options.append("interpolated_peak_width")
    _is_peak_event = (
        selected_swim_bout is not None
        and str(selected_swim_bout.detection_method).lower() == "peak_event"
    )
    _default_boundary = (
        "interpolated_peak_width"
        if _is_peak_event and _has_peak_width_boundaries
        else _boundary_options[-1]
        if _has_interpolated_boundaries
        else _boundary_options[0]
    )
    swim_bout_boundary_picker = mo.ui.dropdown(
        options=_boundary_options,
        value=_default_boundary,
        label="Bout overlay boundaries",
    )
    show_invalid_intervals = mo.ui.checkbox(
        value=bool(data.validity_source),
        label="Overlay invalid/gap intervals",
    )
    histogram_bins = mo.ui.slider(
        start=5,
        stop=100,
        value=40,
        step=5,
        label="Histogram bins",
    )
    mo.vstack(
        [
            speed_series_picker,
            acceleration_series_picker,
            turning_series_picker,
            time_window,
            show_swim_bouts,
            swim_bout_boundary_picker,
            show_invalid_intervals,
            histogram_bins,
        ]
    )
    return (
        acceleration_series_picker,
        histogram_bins,
        show_invalid_intervals,
        show_swim_bouts,
        speed_series_picker,
        swim_bout_boundary_picker,
        time_window,
        turning_series_picker,
    )


@app.cell
def _(
    inter_bout_interval_df,
    np,
    swim_bout_df,
    swim_bout_boundary_picker,
    time,
    time_window,
    timeseries_df,
    validity_df,
    write_perf_event,
):
    _filter_t0 = time.perf_counter()
    start_s, stop_s = time_window.value
    time_mask = (timeseries_df["time_s"] >= start_s) & (timeseries_df["time_s"] <= stop_s)
    filtered_timeseries_df = timeseries_df.loc[time_mask].copy()
    swim_bout_boundary_mode = str(swim_bout_boundary_picker.value)
    _swim_bout_df_for_filter = swim_bout_df
    swim_bout_start_col = "start_s"
    swim_bout_end_col = "end_s"
    if (
        swim_bout_boundary_mode == "interpolated_core_threshold"
        and {
            "core_start_time_s_interpolated",
            "core_end_time_s_interpolated",
            "core_start_time_interpolated_valid",
            "core_end_time_interpolated_valid",
        }.issubset(swim_bout_df.columns)
    ):
        _swim_bout_df_for_filter = swim_bout_df.copy()
        _interp_valid = (
            _swim_bout_df_for_filter["core_start_time_interpolated_valid"].astype(bool)
            & _swim_bout_df_for_filter["core_end_time_interpolated_valid"].astype(bool)
            & np.isfinite(_swim_bout_df_for_filter["core_start_time_s_interpolated"].to_numpy(dtype=float))
            & np.isfinite(_swim_bout_df_for_filter["core_end_time_s_interpolated"].to_numpy(dtype=float))
        )
        _swim_bout_df_for_filter["overlay_start_s"] = np.where(
            _interp_valid,
            _swim_bout_df_for_filter["core_start_time_s_interpolated"].to_numpy(dtype=float),
            _swim_bout_df_for_filter["start_s"].to_numpy(dtype=float),
        )
        _swim_bout_df_for_filter["overlay_end_s"] = np.where(
            _interp_valid,
            _swim_bout_df_for_filter["core_end_time_s_interpolated"].to_numpy(dtype=float),
            _swim_bout_df_for_filter["end_s"].to_numpy(dtype=float),
        )
        swim_bout_start_col = "overlay_start_s"
        swim_bout_end_col = "overlay_end_s"
    elif (
        swim_bout_boundary_mode == "interpolated_peak_width"
        and {
            "peak_event_left_width_time_s",
            "peak_event_right_width_time_s",
        }.issubset(swim_bout_df.columns)
    ):
        _swim_bout_df_for_filter = swim_bout_df.copy()
        _peak_width_valid = (
            np.isfinite(_swim_bout_df_for_filter["peak_event_left_width_time_s"].to_numpy(dtype=float))
            & np.isfinite(_swim_bout_df_for_filter["peak_event_right_width_time_s"].to_numpy(dtype=float))
        )
        _swim_bout_df_for_filter["overlay_start_s"] = np.where(
            _peak_width_valid,
            _swim_bout_df_for_filter["peak_event_left_width_time_s"].to_numpy(dtype=float),
            _swim_bout_df_for_filter["start_s"].to_numpy(dtype=float),
        )
        _swim_bout_df_for_filter["overlay_end_s"] = np.where(
            _peak_width_valid,
            _swim_bout_df_for_filter["peak_event_right_width_time_s"].to_numpy(dtype=float),
            _swim_bout_df_for_filter["end_s"].to_numpy(dtype=float),
        )
        swim_bout_start_col = "overlay_start_s"
        swim_bout_end_col = "overlay_end_s"
    bout_mask = (
        (
            (_swim_bout_df_for_filter[swim_bout_end_col] >= start_s)
            & (_swim_bout_df_for_filter[swim_bout_start_col] <= stop_s)
        )
        if len(_swim_bout_df_for_filter)
        else []
    )
    filtered_swim_bout_df = (
        _swim_bout_df_for_filter.loc[bout_mask].copy()
        if len(_swim_bout_df_for_filter)
        else _swim_bout_df_for_filter
    )
    if len(inter_bout_interval_df) and {"prev_end_time_s", "next_start_time_s"}.issubset(
        inter_bout_interval_df.columns
    ):
        interval_mask = (
            (inter_bout_interval_df["next_start_time_s"] >= start_s)
            & (inter_bout_interval_df["prev_end_time_s"] <= stop_s)
        )
        filtered_inter_bout_interval_df = inter_bout_interval_df.loc[interval_mask].copy()
    else:
        filtered_inter_bout_interval_df = inter_bout_interval_df
    validity_mask = (
        (validity_df["end_s"] >= start_s) & (validity_df["start_s"] <= stop_s)
        if len(validity_df)
        else []
    )
    filtered_validity_df = validity_df.loc[validity_mask].copy() if len(validity_df) else validity_df
    write_perf_event(
        "filter_time_window",
        time.perf_counter() - _filter_t0,
        start_s=float(start_s),
        stop_s=float(stop_s),
        n_timeseries_rows_in=len(timeseries_df),
        n_timeseries_rows_out=len(filtered_timeseries_df),
        n_swim_bout_rows_in=len(swim_bout_df),
        n_swim_bout_rows_out=len(filtered_swim_bout_df),
        n_inter_bout_interval_rows_in=len(inter_bout_interval_df),
        n_inter_bout_interval_rows_out=len(filtered_inter_bout_interval_df),
        n_validity_rows_in=len(validity_df),
        n_validity_rows_out=len(filtered_validity_df),
        swim_bout_boundary_mode=swim_bout_boundary_mode,
    )
    return (
        filtered_inter_bout_interval_df,
        filtered_swim_bout_df,
        filtered_timeseries_df,
        filtered_validity_df,
        swim_bout_boundary_mode,
        swim_bout_end_col,
        swim_bout_start_col,
        start_s,
        stop_s,
    )


@app.cell
def _(
    add_interval_bar_overlay,
    apply_full_width_timeseries_layout,
    data,
    filtered_swim_bout_df,
    filtered_timeseries_df,
    filtered_validity_df,
    go,
    np,
    show_invalid_intervals,
    show_swim_bouts,
    speed_series_picker,
    swim_bout_boundary_mode,
    swim_bout_end_col,
    swim_bout_start_col,
    time,
    write_perf_event,
):
    _figure_t0 = time.perf_counter()
    fig = go.Figure()
    swim_bout_overlay_renderer = "none"
    validity_overlay_renderer = "none"
    if show_invalid_intervals.value and len(filtered_validity_df):
        validity_overlay_renderer = add_interval_bar_overlay(
            fig,
            starts=filtered_validity_df["start_s"].to_numpy(dtype=float),
            ends=filtered_validity_df["end_s"].to_numpy(dtype=float),
            name="Invalid/gap intervals",
            color="crimson",
            opacity=0.16,
            labels=filtered_validity_df["reason"].to_numpy(dtype=object),
            hovertemplate="Invalid interval<br>%{x:.3f}s<br>%{customdata}<extra></extra>",
        )
    if show_swim_bouts.value and len(filtered_swim_bout_df):
        swim_bout_overlay_renderer = add_interval_bar_overlay(
            fig,
            starts=filtered_swim_bout_df[swim_bout_start_col].to_numpy(dtype=float),
            ends=filtered_swim_bout_df[swim_bout_end_col].to_numpy(dtype=float),
            name=f"Swim bouts: {data.swim_bout_label} ({swim_bout_boundary_mode})",
            color="orange",
            opacity=0.18,
            hoverinfo="skip",
        )
    for column in speed_series_picker.value:
        if column not in filtered_timeseries_df:
            continue
        fig.add_trace(
            go.Scattergl(
                x=filtered_timeseries_df["time_s"],
                y=filtered_timeseries_df[column],
                mode="lines",
                name=column,
            )
        )
    apply_full_width_timeseries_layout(
        fig,
        title="Speed Metrics with Swim Bout and Validity Overlays",
        yaxis_title="Speed / detection signal",
        height=420,
    )
    write_perf_event(
        "build_timeseries_figure",
        time.perf_counter() - _figure_t0,
        run_path=data.run_path,
        n_rows=len(filtered_timeseries_df),
        n_traces=len(speed_series_picker.value),
        n_rendered_traces=len(fig.data),
        n_layout_shapes=len(fig.layout.shapes or []),
        n_visible_bouts=len(filtered_swim_bout_df),
        n_visible_validity_intervals=len(filtered_validity_df),
        swim_bout_overlay_renderer=swim_bout_overlay_renderer,
        swim_bout_boundary_mode=swim_bout_boundary_mode,
        validity_overlay_renderer=validity_overlay_renderer,
        show_swim_bouts=bool(show_swim_bouts.value),
        show_invalid_intervals=bool(show_invalid_intervals.value),
    )
    fig
    return


@app.cell
def _(
    acceleration_series_picker,
    add_interval_bar_overlay,
    apply_full_width_timeseries_layout,
    data,
    filtered_swim_bout_df,
    filtered_timeseries_df,
    filtered_validity_df,
    go,
    show_invalid_intervals,
    show_swim_bouts,
    swim_bout_boundary_mode,
    swim_bout_end_col,
    swim_bout_start_col,
    time,
    write_perf_event,
):
    _figure_t0 = time.perf_counter()
    acceleration_fig = go.Figure()
    acceleration_swim_overlay_renderer = "none"
    acceleration_validity_overlay_renderer = "none"
    if show_invalid_intervals.value and len(filtered_validity_df):
        acceleration_validity_overlay_renderer = add_interval_bar_overlay(
            acceleration_fig,
            starts=filtered_validity_df["start_s"].to_numpy(dtype=float),
            ends=filtered_validity_df["end_s"].to_numpy(dtype=float),
            name="Invalid/gap intervals",
            color="crimson",
            opacity=0.16,
            labels=filtered_validity_df["reason"].to_numpy(dtype=object),
            hovertemplate="Invalid interval<br>%{x:.3f}s<br>%{customdata}<extra></extra>",
        )
    if show_swim_bouts.value and len(filtered_swim_bout_df):
        acceleration_swim_overlay_renderer = add_interval_bar_overlay(
            acceleration_fig,
            starts=filtered_swim_bout_df[swim_bout_start_col].to_numpy(dtype=float),
            ends=filtered_swim_bout_df[swim_bout_end_col].to_numpy(dtype=float),
            name=f"Swim bouts: {data.swim_bout_label} ({swim_bout_boundary_mode})",
            color="orange",
            opacity=0.18,
            hoverinfo="skip",
        )
    for _acceleration_column in acceleration_series_picker.value:
        if _acceleration_column not in filtered_timeseries_df:
            continue
        acceleration_fig.add_trace(
            go.Scattergl(
                x=filtered_timeseries_df["time_s"],
                y=filtered_timeseries_df[_acceleration_column],
                mode="lines",
                name=_acceleration_column,
            )
        )
    apply_full_width_timeseries_layout(
        acceleration_fig,
        title="Acceleration Metrics with Swim Bout and Validity Overlays",
        yaxis_title="Acceleration",
        height=360,
    )
    write_perf_event(
        "build_acceleration_figure",
        time.perf_counter() - _figure_t0,
        run_path=data.run_path,
        n_rows=len(filtered_timeseries_df),
        n_traces=len(acceleration_series_picker.value),
        n_rendered_traces=len(acceleration_fig.data),
        n_visible_bouts=len(filtered_swim_bout_df),
        n_visible_validity_intervals=len(filtered_validity_df),
        swim_bout_overlay_renderer=acceleration_swim_overlay_renderer,
        swim_bout_boundary_mode=swim_bout_boundary_mode,
        validity_overlay_renderer=acceleration_validity_overlay_renderer,
    )
    acceleration_fig
    return


@app.cell
def _(
    add_interval_bar_overlay,
    apply_full_width_timeseries_layout,
    data,
    filtered_swim_bout_df,
    filtered_timeseries_df,
    filtered_validity_df,
    go,
    show_invalid_intervals,
    show_swim_bouts,
    swim_bout_boundary_mode,
    swim_bout_end_col,
    swim_bout_start_col,
    time,
    turning_series_picker,
    write_perf_event,
):
    _figure_t0 = time.perf_counter()
    turning_fig = go.Figure()
    turning_swim_overlay_renderer = "none"
    turning_validity_overlay_renderer = "none"
    if show_invalid_intervals.value and len(filtered_validity_df):
        turning_validity_overlay_renderer = add_interval_bar_overlay(
            turning_fig,
            starts=filtered_validity_df["start_s"].to_numpy(dtype=float),
            ends=filtered_validity_df["end_s"].to_numpy(dtype=float),
            name="Invalid/gap intervals",
            color="crimson",
            opacity=0.16,
            labels=filtered_validity_df["reason"].to_numpy(dtype=object),
            hovertemplate="Invalid interval<br>%{x:.3f}s<br>%{customdata}<extra></extra>",
        )
    if show_swim_bouts.value and len(filtered_swim_bout_df):
        turning_swim_overlay_renderer = add_interval_bar_overlay(
            turning_fig,
            starts=filtered_swim_bout_df[swim_bout_start_col].to_numpy(dtype=float),
            ends=filtered_swim_bout_df[swim_bout_end_col].to_numpy(dtype=float),
            name=f"Swim bouts: {data.swim_bout_label} ({swim_bout_boundary_mode})",
            color="orange",
            opacity=0.18,
            hoverinfo="skip",
        )
    for _turning_column in turning_series_picker.value:
        if _turning_column not in filtered_timeseries_df:
            continue
        turning_fig.add_trace(
            go.Scattergl(
                x=filtered_timeseries_df["time_s"],
                y=filtered_timeseries_df[_turning_column],
                mode="lines",
                name=_turning_column,
            )
        )
    apply_full_width_timeseries_layout(
        turning_fig,
        title="Turning and Angular Velocity Metrics with Swim Bout and Validity Overlays",
        yaxis_title="Degrees or degrees/s",
        height=360,
    )
    write_perf_event(
        "build_turning_figure",
        time.perf_counter() - _figure_t0,
        run_path=data.run_path,
        n_rows=len(filtered_timeseries_df),
        n_traces=len(turning_series_picker.value),
        n_rendered_traces=len(turning_fig.data),
        n_visible_bouts=len(filtered_swim_bout_df),
        n_visible_validity_intervals=len(filtered_validity_df),
        swim_bout_overlay_renderer=turning_swim_overlay_renderer,
        swim_bout_boundary_mode=swim_bout_boundary_mode,
        validity_overlay_renderer=turning_validity_overlay_renderer,
    )
    turning_fig
    return


@app.cell
def _(
    bout_kinematics_df,
    filtered_inter_bout_interval_df,
    filtered_swim_bout_df,
    histogram_bins,
    np,
    pd,
    px,
    time,
    write_perf_event,
):
    _figure_t0 = time.perf_counter()

    _metric_specs = [
        (filtered_swim_bout_df, "duration_s", "Bout duration (s)"),
        (filtered_swim_bout_df, "observed_duration_s", "Observed bout duration (s)"),
        (filtered_swim_bout_df, "path_length_mm", "Bout path length (mm)"),
        (filtered_swim_bout_df, "net_displacement_mm", "Bout net displacement (mm)"),
        (filtered_inter_bout_interval_df, "interval_s", "Inter-bout interval (s)"),
    ]
    histogram_frames = []
    for _frame, _column, _label in _metric_specs:
        if _column not in _frame:
            continue
        _values = _frame[_column].to_numpy(dtype=float, copy=False)
        _values = _values[np.isfinite(_values)]
        if _values.size:
            histogram_frames.append(pd.DataFrame({"metric": _label, "value": _values}))

    if histogram_frames:
        histogram_df = pd.concat(histogram_frames, ignore_index=True)
        histogram_plot = px.histogram(
            histogram_df,
            x="value",
            facet_col="metric",
            facet_col_wrap=2,
            nbins=int(histogram_bins.value),
            title="Bout Metric Histograms",
            labels={"value": "Value", "count": "Count"},
            opacity=0.82,
        )
        histogram_plot.update_xaxes(matches=None)
        histogram_plot.update_yaxes(matches=None)
        histogram_plot.update_layout(height=620, margin=dict(l=40, r=20, t=60, b=40), showlegend=False)
    else:
        histogram_df = pd.DataFrame(columns=["metric", "value"])
        histogram_plot = "No bout metric values available for the selected run/window."

    write_perf_event(
        "build_bout_histograms",
        time.perf_counter() - _figure_t0,
        n_histogram_rows=len(histogram_df),
        n_histogram_metrics=int(histogram_df["metric"].nunique()) if len(histogram_df) else 0,
        bins=int(histogram_bins.value),
    )
    histogram_plot
    return (histogram_df,)


@app.cell
def _(bout_kinematics_df, histogram_bins, mo, np, pd, px, time, write_perf_event):
    _figure_t0 = time.perf_counter()
    _net_metric_specs = [
        ("net_delta_heading_deg", "Net heading change (deg)"),
        ("abs_net_delta_heading_deg", "Absolute net heading change (deg)"),
    ]
    _within_metric_specs = [
        ("within_heading_range_deg", "Within-bout heading range (deg)"),
        ("within_heading_peak_to_peak_deg", "Within-bout heading peak-to-peak (deg)"),
        ("within_heading_path_deg", "Within-bout heading path (deg)"),
        ("within_heading_std_deg", "Within-bout heading std (deg)"),
    ]

    def _metric_frames(_metric_specs):
        _frames = []
        for _column, _label in _metric_specs:
            if _column not in bout_kinematics_df:
                continue
            _values = bout_kinematics_df[["heading_level", _column]].copy()
            _values = _values.rename(columns={_column: "value"})
            _values["metric"] = _label
            _values = _values[np.isfinite(_values["value"].to_numpy(dtype=float, copy=False))]
            if len(_values):
                _frames.append(_values[["heading_level", "metric", "value"]])
        return _frames

    _net_frames = _metric_frames(_net_metric_specs)
    _within_frames = _metric_frames(_within_metric_specs)
    _all_frames = [*_net_frames, *_within_frames]

    if _net_frames:
        _net_heading_histogram_df = pd.concat(_net_frames, ignore_index=True)
        _net_heading_histogram_plot = px.histogram(
            _net_heading_histogram_df,
            x="value",
            color="heading_level",
            facet_col="metric",
            facet_col_wrap=2,
            nbins=int(histogram_bins.value),
            barmode="overlay",
            opacity=0.68,
            title="Net Heading-Change Histograms",
            labels={"value": "Degrees", "count": "Bout count", "heading_level": "Heading level"},
        )
        _net_heading_histogram_plot.update_xaxes(matches=None, range=[-180, 180])
        _net_heading_histogram_plot.update_yaxes(matches=None)
        _net_heading_histogram_plot.update_layout(
            height=420,
            margin=dict(l=40, r=20, t=70, b=40),
        )
    else:
        _net_heading_histogram_df = pd.DataFrame(columns=["heading_level", "metric", "value"])
        _net_heading_histogram_plot = "No net heading-change metrics available for the selected candidate."

    if _within_frames:
        _within_heading_histogram_df = pd.concat(_within_frames, ignore_index=True)
        _within_heading_histogram_plot = px.histogram(
            _within_heading_histogram_df,
            x="value",
            color="heading_level",
            facet_col="metric",
            facet_col_wrap=2,
            nbins=int(histogram_bins.value),
            barmode="overlay",
            opacity=0.68,
            title="Within-Bout Heading Metrics",
            labels={"value": "Degrees", "count": "Bout count", "heading_level": "Heading level"},
        )
        _within_heading_histogram_plot.update_xaxes(matches=None)
        _within_heading_histogram_plot.update_yaxes(matches=None)
        _within_heading_histogram_plot.update_layout(
            height=640,
            margin=dict(l=40, r=20, t=70, b=40),
        )
    else:
        _within_heading_histogram_df = pd.DataFrame(columns=["heading_level", "metric", "value"])
        _within_heading_histogram_plot = "No within-bout heading metrics available for the selected candidate."

    if _all_frames:
        bout_heading_histogram_df = pd.concat(_all_frames, ignore_index=True)
    else:
        bout_heading_histogram_df = pd.DataFrame(columns=["heading_level", "metric", "value"])

    write_perf_event(
        "build_bout_heading_histograms",
        time.perf_counter() - _figure_t0,
        n_histogram_rows=len(bout_heading_histogram_df),
        n_net_histogram_rows=len(_net_heading_histogram_df),
        n_within_histogram_rows=len(_within_heading_histogram_df),
        n_heading_levels=(
            int(bout_heading_histogram_df["heading_level"].nunique())
            if len(bout_heading_histogram_df)
            else 0
        ),
        bins=int(histogram_bins.value),
    )
    mo.vstack([_net_heading_histogram_plot, _within_heading_histogram_plot])
    return (bout_heading_histogram_df,)


@app.cell
def _(position_df, px, start_s, stop_s, time, write_perf_event):
    _figure_t0 = time.perf_counter()
    filtered_position_df = position_df[
        (position_df["time_s"] >= start_s) & (position_df["time_s"] <= stop_s)
    ].copy()
    if len(filtered_position_df):
        pos_plot = px.density_heatmap(
            filtered_position_df,
            x="x",
            y="y",
            nbinsx=120,
            nbinsy=120,
            title="Position Density in Selected Time Window",
            labels={
                "x": f"X ({filtered_position_df['unit'].iloc[0]})",
                "y": f"Y ({filtered_position_df['unit'].iloc[0]})",
            },
        )
        pos_plot.update_yaxes(scaleanchor="x", scaleratio=1)
        pos_plot.update_layout(height=560, margin=dict(l=40, r=20, t=50, b=40))
    else:
        pos_plot = "No positions available for the selected time window."
    write_perf_event(
        "build_position_density_figure",
        time.perf_counter() - _figure_t0,
        n_position_rows_in=len(position_df),
        n_position_rows_out=len(filtered_position_df),
        start_s=float(start_s),
        stop_s=float(stop_s),
    )
    pos_plot
    return (filtered_position_df,)


@app.cell
def _(
    bout_kinematics_df,
    data,
    filtered_inter_bout_interval_df,
    filtered_position_df,
    filtered_swim_bout_df,
    filtered_timeseries_df,
    filtered_validity_df,
    mo,
    inter_bout_interval_df,
    swim_bout_df,
    validity_df,
):
    mo.hstack(
        [
            mo.stat(label="Rows", value=f"{len(filtered_timeseries_df):,}"),
            mo.stat(label="Position rows", value=f"{len(filtered_position_df):,}"),
            mo.stat(label="Swim bouts", value=f"{len(swim_bout_df):,}"),
            mo.stat(label="Visible bouts", value=f"{len(filtered_swim_bout_df):,}"),
            mo.stat(label="Inter-bout intervals", value=f"{len(inter_bout_interval_df):,}"),
            mo.stat(label="Visible intervals", value=f"{len(filtered_inter_bout_interval_df):,}"),
            mo.stat(label="Bout kinematics rows", value=f"{len(bout_kinematics_df):,}"),
            mo.stat(label="Invalid intervals", value=f"{len(validity_df):,}"),
            mo.stat(label="Visible invalid", value=f"{len(filtered_validity_df):,}"),
            mo.stat(label="Track ID", value=str(data.spec.get("track_id", "unknown"))),
            mo.stat(label="Position unit", value=data.position_unit),
        ]
    )
    return


@app.cell
def _(bout_kinematics_attrs, bout_kinematics_df, data, inter_bout_interval_df, mo, swim_bout_df):
    mo.accordion(
        {
            "Interactive spec": mo.tree(dict(data.spec)),
            "Artifact attrs": mo.tree(dict(data.attrs)),
            "Source paths": mo.tree(dict(data.source_paths)),
            "Swim bout overlay": mo.tree(
                {
                    "label": data.swim_bout_label,
                    "source": data.swim_bout_source,
                    "count": int(data.swim_bouts.shape[0]),
                }
            ),
            "Bout metric tables": mo.tree(
                {
                    "bout_rows": int(len(swim_bout_df)),
                    "bout_columns": list(swim_bout_df.columns),
                    "inter_bout_interval_rows": int(len(inter_bout_interval_df)),
                    "inter_bout_interval_columns": list(inter_bout_interval_df.columns),
                }
            ),
            "Bout kinematics": mo.tree(
                {
                    "rows": int(len(bout_kinematics_df)),
                    "columns": list(bout_kinematics_df.columns),
                    "attrs": dict(bout_kinematics_attrs),
                }
            ),
            "Validity overlay": mo.tree(
                {
                    "source": data.validity_source,
                    "count": int(data.validity_spans.shape[0]),
                    "labels": sorted(set(str(label) for label in data.validity_labels)),
                }
            ),
        }
    )
    return


if __name__ == "__main__":
    app.run()
