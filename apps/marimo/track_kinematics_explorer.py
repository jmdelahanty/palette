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
    import plotly.express as px
    import plotly.graph_objects as go
    import time

    from fisheye.visualization.interactive_track_kinematics import (
        DEFAULT_INTERACTIVE_ARTIFACT,
        discover_swim_bout_run_options,
        discover_track_kinematics_run_options,
        load_track_kinematics_interactive_data,
        to_position_dataframe,
        to_swim_bout_dataframe,
        to_timeseries_dataframe,
    )

    return (
        DEFAULT_INTERACTIVE_ARTIFACT,
        discover_swim_bout_run_options,
        discover_track_kinematics_run_options,
        go,
        json,
        load_track_kinematics_interactive_data,
        mo,
        px,
        time,
        to_position_dataframe,
        to_swim_bout_dataframe,
        to_timeseries_dataframe,
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
    def _matches_initial_run(option):
        if initial_run_path is None:
            return False
        wanted = str(initial_run_path).strip("/")
        return option.run_path.strip("/") == wanted or option.run_name == wanted.split("/")[-1]

    track_label_to_option = {option.label: option for option in track_options}
    _selected_default = next(
        (option.label for option in track_options if _matches_initial_run(option)),
        track_options[0].label,
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
def _(initial_swim_bout_run, mo, swim_bout_options):
    none_label = "No swim-bout overlay"
    swim_label_to_option = {none_label: None}
    swim_label_to_option.update({option.label: option for option in swim_bout_options})

    def _matches_initial_bout(option):
        if option is None or initial_swim_bout_run is None:
            return False
        return option.run_name == str(initial_swim_bout_run)

    _selected_default = next(
        (
            label
            for label, option in swim_label_to_option.items()
            if _matches_initial_bout(option)
        ),
        swim_bout_options[0].label if swim_bout_options else none_label,
    )
    swim_bout_picker = mo.ui.dropdown(
        options=list(swim_label_to_option),
        value=_selected_default,
        label="Derived swim-bout run",
    )
    swim_bout_picker
    return swim_bout_picker, swim_label_to_option


@app.cell
def _(
    artifact,
    initial_speed_level,
    load_track_kinematics_interactive_data,
    selected_track,
    swim_bout_picker,
    swim_label_to_option,
    time,
    write_perf_event,
    zarr_path,
):
    selected_swim_bout = swim_label_to_option[swim_bout_picker.value]
    selected_speed_level = (
        selected_swim_bout.speed_level
        if selected_swim_bout is not None
        else (str(initial_speed_level) if initial_speed_level is not None else None)
    )
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
    return data, selected_speed_level, selected_swim_bout


@app.cell
def _(
    data,
    mo,
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

        **Speed Level:** `{selected_speed_level or "artifact/default"}`

        **Performance Log:** `{performance_log_path or "disabled"}`

        **Performance Session:** `{performance_session_id}`
        """
    )
    return


@app.cell
def _(
    data,
    time,
    to_position_dataframe,
    to_swim_bout_dataframe,
    to_timeseries_dataframe,
    write_perf_event,
):
    _dataframe_t0 = time.perf_counter()
    timeseries_df = to_timeseries_dataframe(data)
    position_df = to_position_dataframe(data)
    swim_bout_df = to_swim_bout_dataframe(data)
    write_perf_event(
        "build_dataframes",
        time.perf_counter() - _dataframe_t0,
        run_path=data.run_path,
        n_timeseries_rows=len(timeseries_df),
        n_timeseries_columns=len(timeseries_df.columns),
        n_position_rows=len(position_df),
        n_swim_bout_rows=len(swim_bout_df),
    )
    return position_df, swim_bout_df, timeseries_df


@app.cell
def _(data, mo, timeseries_df):
    numeric_columns = [
        column
        for column in timeseries_df.columns
        if column not in {"time_s", "frame_index"} and timeseries_df[column].notna().any()
    ]
    default_columns = [
        column
        for column in ("speed_smoothed_mm", "speed_smoothed_px", "smoothed_heading_degrees")
        if column in numeric_columns
    ]
    series_picker = mo.ui.multiselect(
        options=numeric_columns,
        value=default_columns[:2] if default_columns else numeric_columns[:1],
        label="Time-series traces",
    )
    max_time = float(timeseries_df["time_s"].max()) if len(timeseries_df) else 0.0
    time_window = mo.ui.range_slider(
        start=0.0,
        stop=max_time,
        value=[0.0, max_time],
        step=max(max_time / 1000.0, 0.001),
        label="Time window (s)",
    )
    show_swim_bouts = mo.ui.checkbox(
        value=bool(data.swim_bout_label),
        label="Overlay swim bouts",
    )
    mo.vstack([series_picker, time_window, show_swim_bouts])
    return series_picker, show_swim_bouts, time_window


@app.cell
def _(swim_bout_df, time, time_window, timeseries_df, write_perf_event):
    _filter_t0 = time.perf_counter()
    start_s, stop_s = time_window.value
    time_mask = (timeseries_df["time_s"] >= start_s) & (timeseries_df["time_s"] <= stop_s)
    filtered_timeseries_df = timeseries_df.loc[time_mask].copy()
    bout_mask = (
        (swim_bout_df["end_s"] >= start_s) & (swim_bout_df["start_s"] <= stop_s)
        if len(swim_bout_df)
        else []
    )
    filtered_swim_bout_df = swim_bout_df.loc[bout_mask].copy() if len(swim_bout_df) else swim_bout_df
    write_perf_event(
        "filter_time_window",
        time.perf_counter() - _filter_t0,
        start_s=float(start_s),
        stop_s=float(stop_s),
        n_timeseries_rows_in=len(timeseries_df),
        n_timeseries_rows_out=len(filtered_timeseries_df),
        n_swim_bout_rows_in=len(swim_bout_df),
        n_swim_bout_rows_out=len(filtered_swim_bout_df),
    )
    return filtered_swim_bout_df, filtered_timeseries_df, start_s, stop_s


@app.cell
def _(
    data,
    filtered_swim_bout_df,
    filtered_timeseries_df,
    go,
    series_picker,
    show_swim_bouts,
    time,
    write_perf_event,
):
    _figure_t0 = time.perf_counter()
    fig = go.Figure()
    for column in series_picker.value:
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
    if show_swim_bouts.value and len(filtered_swim_bout_df):
        for bout in filtered_swim_bout_df.itertuples(index=False):
            fig.add_vrect(
                x0=float(bout.start_s),
                x1=float(bout.end_s),
                fillcolor="orange",
                opacity=0.18,
                layer="below",
                line_width=0,
            )
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(color="orange", opacity=0.35),
                name=f"Swim bouts: {data.swim_bout_label}",
            )
        )
    fig.update_layout(
        title="Selected Time-Series",
        xaxis_title="Time (s)",
        yaxis_title="Value",
        hovermode="x unified",
        height=420,
        margin=dict(l=40, r=20, t=50, b=40),
    )
    write_perf_event(
        "build_timeseries_figure",
        time.perf_counter() - _figure_t0,
        run_path=data.run_path,
        n_rows=len(filtered_timeseries_df),
        n_traces=len(series_picker.value),
        n_visible_bouts=len(filtered_swim_bout_df),
        show_swim_bouts=bool(show_swim_bouts.value),
    )
    fig
    return


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
    data,
    filtered_position_df,
    filtered_swim_bout_df,
    filtered_timeseries_df,
    mo,
    swim_bout_df,
):
    mo.hstack(
        [
            mo.stat(label="Rows", value=f"{len(filtered_timeseries_df):,}"),
            mo.stat(label="Position rows", value=f"{len(filtered_position_df):,}"),
            mo.stat(label="Swim bouts", value=f"{len(swim_bout_df):,}"),
            mo.stat(label="Visible bouts", value=f"{len(filtered_swim_bout_df):,}"),
            mo.stat(label="Track ID", value=str(data.spec.get("track_id", "unknown"))),
            mo.stat(label="Position unit", value=data.position_unit),
        ]
    )
    return


@app.cell
def _(data, mo):
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
        }
    )
    return


if __name__ == "__main__":
    app.run()
