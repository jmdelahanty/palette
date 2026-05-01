#!/usr/bin/env python3
"""Marimo app for exploring Palette track-kinematics visualization specs.

Run after installing optional UI dependencies:

    scripts/py -m marimo run apps/marimo/track_kinematics_explorer.py -- \
      --zarr-path /path/to/archive.zarr

Optional selectors:

    --run-path analysis/track_kinematics_runs/offline/<run>
    --swim-bout-run <run-name>
    --speed-level filtered
    --eye-angle-representation eye_frame
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
        discover_eye_angle_run_options,
        discover_swim_bout_run_options,
        discover_track_kinematics_run_options,
        load_eye_angle_timeseries_data,
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
        discover_eye_angle_run_options,
        discover_swim_bout_run_options,
        discover_track_kinematics_run_options,
        go,
        json,
        load_eye_angle_timeseries_data,
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
    discover_eye_angle_run_options,
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
    initial_eye_angle_run = cli_args.get("eye-angle-run")
    initial_eye_angle_representation = cli_args.get("eye-angle-representation")
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
    _eye_discovery_t0 = time.perf_counter()
    eye_angle_options = discover_eye_angle_run_options(zarr_path)
    write_perf_event(
        "discover_eye_angle_options",
        time.perf_counter() - _eye_discovery_t0,
        zarr_path=zarr_path,
        n_eye_angle_options=len(eye_angle_options),
    )
    return (
        artifact,
        eye_angle_options,
        initial_eye_angle_representation,
        initial_eye_angle_run,
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
    mo.vstack(
        [
            mo.md("### Analysis Selection"),
            track_picker,
        ]
    )
    return track_label_to_option, track_picker


@app.cell
def _(eye_angle_options, initial_eye_angle_run, mo):
    _none_label = "No eye-angle run"

    def _matches_initial_eye(_option):
        if initial_eye_angle_run is None:
            return False
        _wanted = str(initial_eye_angle_run).strip("/")
        return _option.run_path.strip("/") == _wanted or _option.run_name == _wanted.split("/")[-1]

    eye_angle_label_to_option = {_none_label: None}
    eye_angle_label_to_option.update({_option.label: _option for _option in eye_angle_options})
    _selected_default = next(
        (
            _option.label
            for _option in eye_angle_options
            if _matches_initial_eye(_option)
        ),
        next(
            (
                _option.label
                for _option in eye_angle_options
                if _option.is_latest
            ),
            eye_angle_options[0].label if eye_angle_options else _none_label,
        ),
    )
    eye_angle_picker = mo.ui.dropdown(
        options=list(eye_angle_label_to_option),
        value=_selected_default,
        label="Eye-angle run",
    )
    eye_angle_picker
    return eye_angle_label_to_option, eye_angle_picker


@app.cell
def _(
    eye_angle_label_to_option,
    eye_angle_picker,
    load_eye_angle_timeseries_data,
    pd,
    time,
    write_perf_event,
    zarr_path,
):
    selected_eye_angle = eye_angle_label_to_option[eye_angle_picker.value]
    _load_eye_t0 = time.perf_counter()
    if selected_eye_angle is None:
        eye_angle_attrs = {}
        eye_angle_data = None
        eye_angle_df = pd.DataFrame(columns=["time_s"])
        eye_angle_row_axis = "none"
    else:
        eye_angle_data = load_eye_angle_timeseries_data(
            zarr_path,
            run_name=selected_eye_angle.run_name,
        )
        eye_angle_attrs = dict(eye_angle_data.attrs)
        eye_angle_df = eye_angle_data.dataframe
        eye_angle_row_axis = eye_angle_data.row_axis
    write_perf_event(
        "load_eye_angle_timeseries",
        time.perf_counter() - _load_eye_t0,
        selected_eye_angle_run=selected_eye_angle.run_name if selected_eye_angle is not None else None,
        row_axis=eye_angle_row_axis,
        n_eye_angle_rows=len(eye_angle_df),
        n_eye_angle_columns=len(eye_angle_df.columns),
    )
    return (
        eye_angle_attrs,
        eye_angle_data,
        eye_angle_df,
        eye_angle_row_axis,
        selected_eye_angle,
    )


@app.cell
def _(eye_angle_attrs, eye_angle_df, initial_eye_angle_representation, mo):
    _REPRESENTATION_DEFAULT_COLUMNS = {
        "eye_frame": [
            "vergence_eye_angle_deg_smoothed",
            "left_eye_angle_deg_smoothed",
            "right_eye_angle_deg_smoothed",
            "vergence_eye_angle_deg",
            "left_eye_angle_deg",
            "right_eye_angle_deg",
        ],
        "gaze": [
            "left_gaze_signed_deg_smoothed",
            "right_gaze_signed_deg_smoothed",
            "vergence_gaze_signed_deg_smoothed",
            "left_gaze_signed_deg",
            "right_gaze_signed_deg",
            "vergence_gaze_signed_deg",
        ],
        "nasal_gaze": [
            "mean_eye_vergence_gaze_deg_smoothed",
            "left_nasal_gaze_deg_smoothed",
            "right_nasal_gaze_deg_smoothed",
            "mean_eye_vergence_gaze_deg",
            "left_nasal_gaze_deg",
            "right_nasal_gaze_deg",
        ],
        "major": [
            "left_major_signed_deg_smoothed",
            "right_major_signed_deg_smoothed",
            "vergence_major_signed_deg_smoothed",
            "version_major_deg_smoothed",
            "left_major_signed_deg",
            "right_major_signed_deg",
            "vergence_major_signed_deg",
            "version_major_deg",
        ],
        "centroid": [
            "left_centroid_deg_smoothed",
            "right_centroid_deg_smoothed",
            "vergence_centroid_deg_smoothed",
            "left_centroid_deg",
            "right_centroid_deg",
            "vergence_centroid_deg",
        ],
        "legacy": [
            "left_minor_signed_deg_smoothed",
            "right_minor_signed_deg_smoothed",
            "vergence_minor_signed_deg_smoothed",
            "left_minor_signed_deg",
            "right_minor_signed_deg",
            "vergence_minor_signed_deg",
        ],
    }
    _REPRESENTATION_LABELS = {
        "eye_frame": "Eye frame (Bianco/Engert nasal-positive)",
        "gaze": "Gaze axis (body-frame rays)",
        "nasal_gaze": "Nasal gaze (BEAST/Johnson comparable)",
        "major": "Major axis (canonical body frame)",
        "centroid": "Centroid position",
        "legacy": "Legacy minor aliases",
    }

    def _variant_schema():
        _schema = eye_angle_attrs.get("eye_angle_variant_schema", {})
        if not isinstance(_schema, dict):
            _output_schema = eye_angle_attrs.get("eye_angle_output_schema", {})
            _schema = _output_schema.get("variant_schema", {}) if isinstance(_output_schema, dict) else {}
        return _schema if isinstance(_schema, dict) else {}

    def _has_representation(_representation):
        return any(
            _column in eye_angle_df.columns and eye_angle_df[_column].notna().any()
            for _column in _REPRESENTATION_DEFAULT_COLUMNS.get(_representation, [])
        )

    _schema = _variant_schema()
    _schema_order = _schema.get("representation_order", [])
    _order = [
        _representation
        for _representation in _schema_order
        if _representation in _REPRESENTATION_DEFAULT_COLUMNS
    ]
    for _fallback in ("eye_frame", "gaze", "nasal_gaze", "major", "centroid", "legacy"):
        if _fallback not in _order:
            _order.append(_fallback)
    _available_representations = [
        _representation for _representation in _order if _has_representation(_representation)
    ]
    if not _available_representations:
        _available_representations = ["eye_frame"]
    eye_angle_representation_label_to_value = {
        _REPRESENTATION_LABELS.get(_representation, _representation): _representation
        for _representation in _available_representations
    }
    _default_representation = str(
        initial_eye_angle_representation
        or _schema.get("default_representation")
        or _available_representations[0]
    )
    if _default_representation not in _available_representations:
        _default_representation = _available_representations[0]
    _default_label = next(
        _label
        for _label, _representation in eye_angle_representation_label_to_value.items()
        if _representation == _default_representation
    )
    eye_angle_representation_picker = mo.ui.dropdown(
        options=list(eye_angle_representation_label_to_value),
        value=_default_label,
        label="Eye-angle representation",
    )
    eye_angle_representation_picker
    return (
        eye_angle_representation_label_to_value,
        eye_angle_representation_picker,
    )


@app.cell
def _(
    discover_swim_bout_run_options,
    track_label_to_option,
    track_picker,
    time,
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
    eye_angle_attrs,
    eye_angle_df,
    eye_angle_row_axis,
    selected_bout_kinematics,
    selected_eye_angle,
    mo,
    pd,
    performance_log_path,
    performance_session_id,
    selected_eye_angle_representation,
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

        **Eye Angles:** `{selected_eye_angle.label if selected_eye_angle else "none"}`

        **Eye-Angle Representation:** `{selected_eye_angle_representation}`

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
        {
            "surface": "eye_angle",
            "field": "run",
            "value": selected_eye_angle.run_name if selected_eye_angle else "none",
        },
        {
            "surface": "eye_angle",
            "field": "schema_version",
            "value": eye_angle_attrs.get("schema_version", "none"),
        },
        {
            "surface": "eye_angle",
            "field": "preferred_angle_family",
            "value": eye_angle_attrs.get("preferred_angle_family", "none"),
        },
        {
            "surface": "eye_angle",
            "field": "preferred_eye_axis",
            "value": eye_angle_attrs.get("preferred_eye_axis", "none"),
        },
        {
            "surface": "eye_angle",
            "field": "row_axis_loaded",
            "value": eye_angle_row_axis,
        },
        {
            "surface": "eye_angle",
            "field": "representation",
            "value": selected_eye_angle_representation,
        },
        {
            "surface": "eye_angle",
            "field": "rows_loaded",
            "value": len(eye_angle_df),
        },
    ]
    selected_summary_df = pd.DataFrame(_selected_summary_rows)
    selected_analysis_tree = {
        "archive": str(data.zarr_path),
        "track_kinematics": {
            "run": selected_track.run_name,
            "run_path": selected_track.run_path,
            "scope": selected_track.run_scope,
            "track_id": int(selected_track.track_id),
            "latest": bool(selected_track.is_latest),
            "artifact": {
                "name": data.artifact_name,
                "renderer": data.attrs.get("renderer", "unknown"),
                "schema_id": data.spec.get("schema_id", "unknown"),
            },
            "speed_selection": {
                "selected_level": selected_speed_level or "artifact/default",
                "position_unit": data.position_unit,
                "available_speed_series": [
                    _column
                    for _column in sorted(data.series)
                    if _column.startswith("speed_")
                    and (_column.endswith("_mm") or _column.endswith("_px"))
                ],
                "available_derivative_series": [
                    _column
                    for _column in sorted(data.series)
                    if "acceleration" in _column
                ],
            },
        },
        "swim_bout_candidate": (
            {
                "run": selected_swim_bout.run_name,
                "label": selected_swim_bout.label,
                "latest": bool(selected_swim_bout.is_latest),
                "selected_speed_level": selected_swim_bout.speed_level,
                "default_level": selected_swim_bout.default_level,
                "source_track_kinematics_run": selected_swim_bout.source_track_kinematics_run,
                "track_id": selected_swim_bout.track_id,
                "detection_method": selected_swim_bout.detection_method,
                "threshold_mm_s": selected_swim_bout.threshold_mm,
                "exponential_tau_s": selected_swim_bout.exponential_tau_s,
                "exponential_source_level": selected_swim_bout.exponential_source_level,
                "n_bouts_by_level": dict(selected_swim_bout.n_bouts_by_level),
                "loaded_overlay": {
                    "label": data.swim_bout_label,
                    "source": data.swim_bout_source,
                    "rows": int(data.swim_bouts.shape[0]),
                },
            }
            if selected_swim_bout
            else None
        ),
        "bout_kinematics_candidate": (
            {
                "run": selected_bout_kinematics.run_name,
                "label": selected_bout_kinematics.label,
                "latest": bool(selected_bout_kinematics.is_latest),
                "source_track_kinematics_run": selected_bout_kinematics.source_track_kinematics_run,
                "source_track_id": selected_bout_kinematics.source_track_id,
                "source_swim_bout_run": selected_bout_kinematics.source_swim_bout_run,
                "source_swim_bout_speed_level": selected_bout_kinematics.source_swim_bout_speed_level,
                "pre_post_mode": selected_bout_kinematics.pre_post_mode,
                "default_heading_level": selected_bout_kinematics.default_heading_level,
                "heading_levels": list(selected_bout_kinematics.heading_levels),
                "n_rows_by_level": dict(selected_bout_kinematics.n_rows_by_level),
                "loaded_rows": int(len(bout_kinematics_df)),
                "attrs": dict(bout_kinematics_attrs),
            }
            if selected_bout_kinematics
            else None
        ),
        "eye_angle_run": (
            {
                "run": selected_eye_angle.run_name,
                "run_path": selected_eye_angle.run_path,
                "label": selected_eye_angle.label,
                "latest": bool(selected_eye_angle.is_latest),
                "schema_version": selected_eye_angle.schema_version,
                "preferred_angle_family": selected_eye_angle.preferred_angle_family,
                "preferred_eye_axis": selected_eye_angle.preferred_eye_axis,
                "declared_row_axis": selected_eye_angle.row_axis,
                "loaded_row_axis": eye_angle_row_axis,
                "selected_representation": selected_eye_angle_representation,
                "declared_rows": int(selected_eye_angle.n_rows),
                "loaded_rows": int(len(eye_angle_df)),
                "attrs": dict(eye_angle_attrs),
            }
            if selected_eye_angle
            else None
        ),
    }
    mo.vstack(
        [
            mo.md("### Selected Candidate Details"),
            mo.ui.table(selected_summary_df, selection=None),
            mo.md("### Selected Analysis Tree"),
            mo.tree(selected_analysis_tree),
        ]
    )
    return selected_analysis_tree, selected_summary_df


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
def _(
    data,
    eye_angle_df,
    eye_angle_representation_label_to_value,
    eye_angle_representation_picker,
    mo,
    np,
    selected_speed_level,
    selected_swim_bout,
    swim_bout_df,
    timeseries_df,
):
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
    _accel_level = _level
    if _accel_level == "exponential" and selected_swim_bout is not None:
        _accel_level = str(selected_swim_bout.exponential_source_level or "filtered").removeprefix("speed_")
    _source_acceleration_candidates = (
        f"speed_{_accel_level}_smoothed_acceleration_mm",
        f"speed_{_accel_level}_acceleration_mm",
        f"speed_{_accel_level}_smoothed_acceleration_px",
        f"speed_{_accel_level}_acceleration_px",
    ) if _accel_level else ()
    _default_acceleration_columns = [
        _column
        for _column in (
            *_source_acceleration_candidates,
            "smoothed_acceleration_mm",
            "smoothed_acceleration_px",
            "acceleration_mm",
            "acceleration_px",
        )
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
    _eye_angle_columns = [
        _column
        for _column in eye_angle_df.columns
        if _column not in {"time_s", "frame_index", "row_index", "valid_frame", "valid_left", "valid_right"}
        and eye_angle_df[_column].notna().any()
    ]
    selected_eye_angle_representation = eye_angle_representation_label_to_value.get(
        eye_angle_representation_picker.value,
        "eye_frame",
    )
    _eye_angle_representation_defaults = {
        "eye_frame": [
            "vergence_eye_angle_deg_smoothed",
            "left_eye_angle_deg_smoothed",
            "right_eye_angle_deg_smoothed",
            "vergence_eye_angle_deg",
            "left_eye_angle_deg",
            "right_eye_angle_deg",
        ],
        "gaze": [
            "left_gaze_signed_deg_smoothed",
            "right_gaze_signed_deg_smoothed",
            "vergence_gaze_signed_deg_smoothed",
            "left_gaze_signed_deg",
            "right_gaze_signed_deg",
            "vergence_gaze_signed_deg",
        ],
        "nasal_gaze": [
            "mean_eye_vergence_gaze_deg_smoothed",
            "left_nasal_gaze_deg_smoothed",
            "right_nasal_gaze_deg_smoothed",
            "mean_eye_vergence_gaze_deg",
            "left_nasal_gaze_deg",
            "right_nasal_gaze_deg",
        ],
        "major": [
            "left_major_signed_deg_smoothed",
            "right_major_signed_deg_smoothed",
            "vergence_major_signed_deg_smoothed",
            "version_major_deg_smoothed",
            "left_major_signed_deg",
            "right_major_signed_deg",
            "vergence_major_signed_deg",
            "version_major_deg",
        ],
        "centroid": [
            "left_centroid_deg_smoothed",
            "right_centroid_deg_smoothed",
            "vergence_centroid_deg_smoothed",
            "left_centroid_deg",
            "right_centroid_deg",
            "vergence_centroid_deg",
        ],
        "legacy": [
            "left_minor_signed_deg_smoothed",
            "right_minor_signed_deg_smoothed",
            "vergence_minor_signed_deg_smoothed",
            "left_minor_signed_deg",
            "right_minor_signed_deg",
            "vergence_minor_signed_deg",
        ],
    }
    _preferred_eye_angle_columns = [
        _column
        for _column in _eye_angle_representation_defaults.get(selected_eye_angle_representation, [])
        if _column in _eye_angle_columns
    ]
    _default_eye_angle_columns = _preferred_eye_angle_columns[:3]
    if not _default_eye_angle_columns:
        _default_eye_angle_columns = _eye_angle_columns[:3]
    eye_angle_series_picker = mo.ui.multiselect(
        options=_eye_angle_columns,
        value=_default_eye_angle_columns,
        label="Eye-angle traces",
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
            eye_angle_series_picker,
            time_window,
            show_swim_bouts,
            swim_bout_boundary_picker,
            show_invalid_intervals,
            histogram_bins,
        ]
    )
    return (
        acceleration_series_picker,
        eye_angle_series_picker,
        histogram_bins,
        selected_eye_angle_representation,
        show_invalid_intervals,
        show_swim_bouts,
        speed_series_picker,
        swim_bout_boundary_picker,
        time_window,
        turning_series_picker,
    )


@app.cell
def _(
    eye_angle_df,
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
    eye_time_mask = (
        (eye_angle_df["time_s"] >= start_s) & (eye_angle_df["time_s"] <= stop_s)
        if len(eye_angle_df) and "time_s" in eye_angle_df
        else []
    )
    filtered_eye_angle_df = eye_angle_df.loc[eye_time_mask].copy() if len(eye_angle_df) else eye_angle_df
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
        n_eye_angle_rows_in=len(eye_angle_df),
        n_eye_angle_rows_out=len(filtered_eye_angle_df),
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
        filtered_eye_angle_df,
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
    add_interval_bar_overlay,
    apply_full_width_timeseries_layout,
    eye_angle_data,
    eye_angle_row_axis,
    eye_angle_series_picker,
    filtered_eye_angle_df,
    filtered_swim_bout_df,
    filtered_validity_df,
    go,
    mo,
    selected_eye_angle,
    selected_eye_angle_representation,
    show_invalid_intervals,
    show_swim_bouts,
    swim_bout_boundary_mode,
    swim_bout_end_col,
    swim_bout_start_col,
    time,
    write_perf_event,
):
    _figure_t0 = time.perf_counter()
    if selected_eye_angle is None:
        eye_angle_plot_output = mo.md("No eye-angle run selected.")
        _rendered_eye_traces = 0
        _eye_bout_overlay_renderer = "none"
        _eye_validity_overlay_renderer = "none"
    elif not len(filtered_eye_angle_df) or not eye_angle_series_picker.value:
        eye_angle_plot_output = mo.md("No eye-angle traces available for the selected run/window.")
        _rendered_eye_traces = 0
        _eye_bout_overlay_renderer = "none"
        _eye_validity_overlay_renderer = "none"
    else:
        eye_angle_fig = go.Figure()
        _eye_bout_overlay_renderer = "none"
        _eye_validity_overlay_renderer = "none"
        if show_invalid_intervals.value and len(filtered_validity_df):
            _eye_validity_overlay_renderer = add_interval_bar_overlay(
                eye_angle_fig,
                starts=filtered_validity_df["start_s"].to_numpy(dtype=float),
                ends=filtered_validity_df["end_s"].to_numpy(dtype=float),
                name="Invalid/gap intervals",
                color="crimson",
                opacity=0.16,
                labels=filtered_validity_df["reason"].to_numpy(dtype=object),
                hovertemplate="Invalid interval<br>%{x:.3f}s<br>%{customdata}<extra></extra>",
            )
        if show_swim_bouts.value and len(filtered_swim_bout_df):
            _eye_bout_overlay_renderer = add_interval_bar_overlay(
                eye_angle_fig,
                starts=filtered_swim_bout_df[swim_bout_start_col].to_numpy(dtype=float),
                ends=filtered_swim_bout_df[swim_bout_end_col].to_numpy(dtype=float),
                name=f"Swim bouts ({swim_bout_boundary_mode})",
                color="orange",
                opacity=0.18,
                hoverinfo="skip",
            )
        _rendered_eye_traces = 0
        for _eye_column in eye_angle_series_picker.value:
            if _eye_column not in filtered_eye_angle_df:
                continue
            eye_angle_fig.add_trace(
                go.Scattergl(
                    x=filtered_eye_angle_df["time_s"],
                    y=filtered_eye_angle_df[_eye_column],
                    mode="lines",
                    name=_eye_column,
                )
            )
            _rendered_eye_traces += 1
        _eye_title = (
            f"Eye-Angle Traces ({eye_angle_data.run_name}; {eye_angle_row_axis} rows; {selected_eye_angle_representation})"
            if eye_angle_data is not None
            else "Eye-Angle Traces"
        )
        apply_full_width_timeseries_layout(
            eye_angle_fig,
            title=_eye_title,
            yaxis_title="Eye angle (deg)",
            height=420,
        )
        eye_angle_plot_output = eye_angle_fig
    write_perf_event(
        "build_eye_angle_figure",
        time.perf_counter() - _figure_t0,
        selected_eye_angle_run=selected_eye_angle.run_name if selected_eye_angle is not None else None,
        row_axis=eye_angle_row_axis,
        n_rows=len(filtered_eye_angle_df),
        n_traces=len(eye_angle_series_picker.value),
        n_rendered_traces=_rendered_eye_traces,
        n_visible_bouts=len(filtered_swim_bout_df),
        n_visible_validity_intervals=len(filtered_validity_df),
        swim_bout_overlay_renderer=_eye_bout_overlay_renderer,
        validity_overlay_renderer=_eye_validity_overlay_renderer,
        representation=selected_eye_angle_representation,
    )
    eye_angle_plot_output
    return


@app.cell
def _(
    eye_angle_series_picker,
    filtered_eye_angle_df,
    histogram_bins,
    np,
    pd,
    px,
    selected_eye_angle,
    selected_eye_angle_representation,
    time,
    write_perf_event,
):
    _figure_t0 = time.perf_counter()
    _frames = []
    if selected_eye_angle is not None and len(filtered_eye_angle_df):
        for _eye_column in eye_angle_series_picker.value:
            if _eye_column not in filtered_eye_angle_df:
                continue
            _values = filtered_eye_angle_df[_eye_column].to_numpy(dtype=float, copy=False)
            _values = _values[np.isfinite(_values)]
            if _values.size:
                _frames.append(pd.DataFrame({"metric": _eye_column, "angle_deg": _values}))
    if _frames:
        eye_angle_histogram_df = pd.concat(_frames, ignore_index=True)
        eye_angle_histogram_plot = px.histogram(
            eye_angle_histogram_df,
            x="angle_deg",
            facet_col="metric",
            facet_col_wrap=2,
            nbins=int(histogram_bins.value),
            title=f"Eye-Angle Distributions ({selected_eye_angle_representation})",
            labels={"angle_deg": "Angle (deg)", "count": "Count"},
            opacity=0.82,
        )
        eye_angle_histogram_plot.update_xaxes(matches=None)
        eye_angle_histogram_plot.update_yaxes(matches=None)
        eye_angle_histogram_plot.update_layout(
            height=620,
            margin=dict(l=40, r=20, t=70, b=40),
            showlegend=False,
        )
    else:
        eye_angle_histogram_df = pd.DataFrame(columns=["metric", "angle_deg"])
        eye_angle_histogram_plot = "No eye-angle histogram values available for the selected run/window."
    write_perf_event(
        "build_eye_angle_histograms",
        time.perf_counter() - _figure_t0,
        selected_eye_angle_run=selected_eye_angle.run_name if selected_eye_angle is not None else None,
        representation=selected_eye_angle_representation,
        n_histogram_rows=len(eye_angle_histogram_df),
        n_histogram_metrics=(
            int(eye_angle_histogram_df["metric"].nunique())
            if len(eye_angle_histogram_df)
            else 0
        ),
        bins=int(histogram_bins.value),
    )
    eye_angle_histogram_plot
    return (eye_angle_histogram_df,)


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
        (filtered_swim_bout_df, "duration_s", "Detector-window duration (s)"),
        (filtered_swim_bout_df, "observed_duration_s", "Detector observed duration (s)"),
        (filtered_swim_bout_df, "path_length_mm", "Segmentation-window path length (mm)"),
        (filtered_swim_bout_df, "net_displacement_mm", "Segmentation-window net displacement (mm)"),
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
            title="Swim-Bout Segmentation Histograms",
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
def _(bout_kinematics_df, histogram_bins, np, pd, px, time, write_perf_event):
    _figure_t0 = time.perf_counter()
    _movement_df = (
        bout_kinematics_df[bout_kinematics_df["analysis_level"] == "movement"].copy()
        if "analysis_level" in bout_kinematics_df
        else pd.DataFrame()
    )
    _metric_specs = [
        ("physical_active_duration_s", "Physical active duration (s)"),
        ("physical_active_duration_s_interpolated", "Physical active duration interpolated (s)"),
        ("physical_active_path_length_mm", "Physical active path length (mm)"),
        ("physical_active_mean_speed_mm_s", "Physical active mean speed (mm/s)"),
        ("physical_active_peak_speed_mm_s", "Physical active peak speed (mm/s)"),
        ("detector_duration_s", "Copied detector duration (s)"),
    ]
    _frames = []
    for _column, _label in _metric_specs:
        if _column not in _movement_df:
            continue
        _values = _movement_df[_column].to_numpy(dtype=float, copy=False)
        _values = _values[np.isfinite(_values)]
        if _values.size:
            _frames.append(pd.DataFrame({"metric": _label, "value": _values}))

    if _frames:
        bout_movement_histogram_df = pd.concat(_frames, ignore_index=True)
        bout_movement_histogram_plot = px.histogram(
            bout_movement_histogram_df,
            x="value",
            facet_col="metric",
            facet_col_wrap=2,
            nbins=int(histogram_bins.value),
            title="Bout Physical Movement Histograms",
            labels={"value": "Value", "count": "Bout count"},
            opacity=0.82,
        )
        bout_movement_histogram_plot.update_xaxes(matches=None)
        bout_movement_histogram_plot.update_yaxes(matches=None)
        bout_movement_histogram_plot.update_layout(
            height=860,
            margin=dict(l=40, r=20, t=70, b=40),
            showlegend=False,
        )
    else:
        bout_movement_histogram_df = pd.DataFrame(columns=["metric", "value"])
        bout_movement_histogram_plot = "No physical movement metrics available for the selected bout-kinematics candidate."

    write_perf_event(
        "build_bout_movement_histograms",
        time.perf_counter() - _figure_t0,
        n_histogram_rows=len(bout_movement_histogram_df),
        n_histogram_metrics=(
            int(bout_movement_histogram_df["metric"].nunique())
            if len(bout_movement_histogram_df)
            else 0
        ),
        bins=int(histogram_bins.value),
    )
    bout_movement_histogram_plot
    return (bout_movement_histogram_df,)


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
    _eye_metric_specs = [
        ("pre_vergence_gaze_mean_deg", "Pre-bout vergence gaze (deg)"),
        ("post_vergence_gaze_mean_deg", "Post-bout vergence gaze (deg)"),
        ("within_bout_vergence_gaze_mean_deg", "Within-bout mean vergence gaze (deg)"),
        ("within_bout_vergence_gaze_max_deg", "Within-bout max vergence gaze (deg)"),
        ("within_bout_vergence_gaze_range_deg", "Within-bout vergence range (deg)"),
        ("within_bout_converged_fraction", "Within-bout converged fraction"),
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
    _eye_frames = _metric_frames(_eye_metric_specs)
    _all_frames = [*_net_frames, *_within_frames, *_eye_frames]

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

    if _eye_frames:
        _eye_gaze_histogram_df = pd.concat(_eye_frames, ignore_index=True)
        _eye_gaze_histogram_plot = px.histogram(
            _eye_gaze_histogram_df,
            x="value",
            color="heading_level",
            facet_col="metric",
            facet_col_wrap=2,
            nbins=int(histogram_bins.value),
            barmode="overlay",
            opacity=0.68,
            title="Bout Eye-Gaze Metrics",
            labels={"value": "Value", "count": "Bout count", "heading_level": "Analysis level"},
        )
        _eye_gaze_histogram_plot.update_xaxes(matches=None)
        _eye_gaze_histogram_plot.update_yaxes(matches=None)
        _eye_gaze_histogram_plot.update_layout(
            height=760,
            margin=dict(l=40, r=20, t=70, b=40),
        )
    else:
        _eye_gaze_histogram_df = pd.DataFrame(columns=["heading_level", "metric", "value"])
        _eye_gaze_histogram_plot = "No bout eye-gaze metrics available for the selected candidate."

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
        n_eye_gaze_histogram_rows=len(_eye_gaze_histogram_df),
        n_heading_levels=(
            int(bout_heading_histogram_df["heading_level"].nunique())
            if len(bout_heading_histogram_df)
            else 0
        ),
        bins=int(histogram_bins.value),
    )
    mo.vstack([_net_heading_histogram_plot, _within_heading_histogram_plot, _eye_gaze_histogram_plot])
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
