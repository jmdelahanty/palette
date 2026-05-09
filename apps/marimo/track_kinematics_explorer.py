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
    import base64
    import json
    import marimo as mo
    import numpy as np
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
    import time
    import zarr

    from fisheye.analysis.plot_stimulus_response_omr import (
        OMR_BOUT_TRAJECTORY_PNG_ARTIFACT_NAME,
        OMR_SUMMARY_PNG_ARTIFACT_NAME,
        load_omr_step_summaries,
    )
    from fisheye.visualization.interactive_track_kinematics import (
        DEFAULT_INTERACTIVE_ARTIFACT,
        bout_classification_records_to_dataframe,
        bout_kinematics_records_to_dataframe,
        discover_bout_classification_run_options,
        discover_bout_kinematics_run_options,
        discover_eye_angle_run_options,
        discover_swim_bout_run_options,
        discover_track_kinematics_run_options,
        load_bout_classification_records,
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

    def png_bytes_to_markdown_image(png_bytes, *, alt_text):
        if not png_bytes:
            return mo.md(f"*No PNG bytes available for `{alt_text}`.*")
        encoded = base64.b64encode(bytes(png_bytes)).decode("ascii")
        return mo.md(
            f'<img alt="{alt_text}" src="data:image/png;base64,{encoded}" '
            'style="max-width:100%; height:auto; border: 1px solid #ddd;" />'
        )

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
        OMR_BOUT_TRAJECTORY_PNG_ARTIFACT_NAME,
        OMR_SUMMARY_PNG_ARTIFACT_NAME,
        add_interval_bar_overlay,
        apply_full_width_timeseries_layout,
        bout_classification_records_to_dataframe,
        bout_kinematics_records_to_dataframe,
        discover_bout_classification_run_options,
        discover_bout_kinematics_run_options,
        discover_eye_angle_run_options,
        discover_swim_bout_run_options,
        discover_track_kinematics_run_options,
        go,
        json,
        load_omr_step_summaries,
        load_bout_classification_records,
        load_eye_angle_timeseries_data,
        load_bout_kinematics_records,
        load_track_kinematics_interactive_data,
        mo,
        np,
        pd,
        png_bytes_to_markdown_image,
        px,
        time,
        to_inter_bout_interval_dataframe,
        to_position_dataframe,
        to_swim_bout_dataframe,
        to_timeseries_dataframe,
        to_validity_span_dataframe,
        zarr,
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
                if option.is_latest
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
    discover_bout_classification_run_options,
    selected_speed_level,
    selected_swim_bout,
    time,
    write_perf_event,
    zarr_path,
):
    _classification_discovery_t0 = time.perf_counter()
    bout_classification_options = discover_bout_classification_run_options(
        zarr_path,
        swim_bout_run=selected_swim_bout.run_name if selected_swim_bout is not None else None,
        speed_level=selected_speed_level,
    )
    write_perf_event(
        "discover_bout_classification_options",
        time.perf_counter() - _classification_discovery_t0,
        zarr_path=zarr_path,
        selected_swim_bout_run=selected_swim_bout.run_name if selected_swim_bout is not None else None,
        selected_speed_level=selected_speed_level,
        n_bout_classification_options=len(bout_classification_options),
    )
    return (bout_classification_options,)


@app.cell
def _(bout_classification_options, mo):
    _classification_none_label = "No bout-classification overlay"
    bout_classification_label_to_option = {_classification_none_label: None}
    bout_classification_label_to_option.update(
        {_classification_option.label: _classification_option for _classification_option in bout_classification_options}
    )
    _classification_default = (
        bout_classification_options[0].label
        if bout_classification_options
        else _classification_none_label
    )
    bout_classification_picker = mo.ui.dropdown(
        options=list(bout_classification_label_to_option),
        value=_classification_default,
        label="Bout-classification candidate",
    )
    bout_classification_picker
    return bout_classification_label_to_option, bout_classification_picker


@app.cell
def _(
    bout_classification_label_to_option,
    bout_classification_picker,
    bout_classification_records_to_dataframe,
    load_bout_classification_records,
    np,
    pd,
    time,
    write_perf_event,
    zarr_path,
):
    selected_bout_classification = bout_classification_label_to_option[bout_classification_picker.value]
    _classification_load_t0 = time.perf_counter()
    if selected_bout_classification is None:
        bout_classification_attrs = {}
        bout_classification_df = pd.DataFrame()
    else:
        _classification_records, bout_classification_attrs = load_bout_classification_records(
            zarr_path,
            run_name=selected_bout_classification.run_name,
        )
        bout_classification_df = bout_classification_records_to_dataframe(_classification_records)
        if len(bout_classification_df):
            bout_classification_df.insert(
                0,
                "source_bout_row",
                np.arange(len(bout_classification_df), dtype=np.int64),
            )
    write_perf_event(
        "load_bout_classification",
        time.perf_counter() - _classification_load_t0,
        selected_bout_classification_run=(
            selected_bout_classification.run_name if selected_bout_classification is not None else None
        ),
        n_bout_classification_rows=len(bout_classification_df),
    )
    return bout_classification_attrs, bout_classification_df, selected_bout_classification


@app.cell
def _(bout_classification_df, mo):
    _classification_categories = (
        sorted(str(_classification_value) for _classification_value in bout_classification_df["category_label"].dropna().unique())
        if "category_label" in bout_classification_df and len(bout_classification_df)
        else []
    )
    bout_classification_category_picker = mo.ui.multiselect(
        options=_classification_categories,
        value=_classification_categories,
        label="Classification categories",
    )
    bout_classification_category_picker
    return (bout_classification_category_picker,)


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
    bout_classification_attrs,
    bout_classification_df,
    bout_kinematics_attrs,
    bout_kinematics_df,
    data,
    eye_angle_attrs,
    eye_angle_df,
    eye_angle_row_axis,
    selected_bout_classification,
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

        **Bout Classification:** `{selected_bout_classification.label if selected_bout_classification else "none"}`

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
            "surface": "bout_classification",
            "field": "run",
            "value": selected_bout_classification.run_name if selected_bout_classification else "none",
        },
        {
            "surface": "bout_classification",
            "field": "classifier",
            "value": (
                f"{selected_bout_classification.classifier_family}/"
                f"{selected_bout_classification.classifier_name}"
                if selected_bout_classification
                else "none"
            ),
        },
        {
            "surface": "bout_classification",
            "field": "rows_loaded",
            "value": len(bout_classification_df),
        },
        {
            "surface": "bout_classification",
            "field": "classified_rows",
            "value": (
                int(bout_classification_df["classified"].sum())
                if "classified" in bout_classification_df
                else 0
            ),
        },
        {
            "surface": "bout_classification",
            "field": "schema_version",
            "value": bout_classification_attrs.get("schema_version", "none"),
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
                "layout": selected_swim_bout.layout,
                "candidate_id": int(selected_swim_bout.candidate_id),
                "signal_id": int(selected_swim_bout.signal_id),
                "signal_role": selected_swim_bout.signal_role,
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
        "bout_classification_candidate": (
            {
                "run": selected_bout_classification.run_name,
                "label": selected_bout_classification.label,
                "latest": bool(selected_bout_classification.is_latest),
                "classifier_family": selected_bout_classification.classifier_family,
                "classifier_name": selected_bout_classification.classifier_name,
                "classifier_version": selected_bout_classification.classifier_version,
                "source_swim_bout_run": selected_bout_classification.source_swim_bout_run,
                "source_swim_bout_speed_level": selected_bout_classification.source_swim_bout_speed_level,
                "source_bout_count": int(selected_bout_classification.source_bout_count),
                "classified_bout_count": int(selected_bout_classification.classified_bout_count),
                "skipped_bout_count": int(selected_bout_classification.skipped_bout_count),
                "loaded_rows": int(len(bout_classification_df)),
                "attrs": dict(bout_classification_attrs),
            }
            if selected_bout_classification
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
def _(json, mo, pd, time, write_perf_event, zarr, zarr_path):
    _stimulus_step_discovery_t0 = time.perf_counter()

    def _group_keys(_group):
        _keys_fn = getattr(_group, "group_keys", None)
        if callable(_keys_fn):
            try:
                return sorted(str(_key) for _key in _keys_fn())
            except Exception:
                pass
        try:
            return sorted(str(_key) for _key in _group.keys())
        except Exception:
            return []

    def _local_child_group_names(_parent_path):
        if not _parent_path.exists():
            return []
        return sorted(
            _child.name
            for _child in _parent_path.iterdir()
            if _child.is_dir() and (_child / "zarr.json").exists()
        )

    def _open_child_group(_parent, _parent_path, _name):
        try:
            return _parent[_name]
        except Exception:
            _child_path = _parent_path / str(_name)
            if (_child_path / "zarr.json").exists():
                return zarr.open_group(str(_child_path), mode="r", use_consolidated=False)
            raise

    def _mode_attrs(_step_group, _step_path, _mode_group_name):
        try:
            if _mode_group_name in _step_group:
                return dict(_step_group[_mode_group_name].attrs)
        except Exception:
            pass
        _mode_path = _step_path / _mode_group_name
        if (_mode_path / "zarr.json").exists():
            try:
                return dict(zarr.open_group(str(_mode_path), mode="r", use_consolidated=False).attrs)
            except Exception:
                return {}
        return {}

    def _parse_raw_protocol(_value):
        if not _value:
            return {}
        try:
            _payload = json.loads(str(_value))
        except Exception:
            return {}
        return _payload if isinstance(_payload, dict) else {}

    _step_rows = []
    try:
        _root = zarr.open_group(str(zarr_path), mode="r", use_consolidated=False)
        _parent = _root["analysis/stimulus_runs"]
        _parent_path = zarr_path / "analysis" / "stimulus_runs"
        _latest = str(_parent.attrs.get("latest", ""))
        _run_names = sorted(set(_group_keys(_parent)) | set(_local_child_group_names(_parent_path)))
        for _run_name in _run_names:
            _run_path = _parent_path / _run_name
            try:
                _run_group = _open_child_group(_parent, _parent_path, _run_name)
            except Exception:
                continue
            _steps_path = _run_path / "steps"
            try:
                _steps_group = _run_group["steps"]
            except Exception:
                if (_steps_path / "zarr.json").exists():
                    _steps_group = zarr.open_group(str(_steps_path), mode="r", use_consolidated=False)
                else:
                    continue
            _step_names = sorted(set(_group_keys(_steps_group)) | set(_local_child_group_names(_steps_path)))
            for _step_name in _step_names:
                _step_path = _steps_path / _step_name
                try:
                    _step_group = _open_child_group(_steps_group, _steps_path, _step_name)
                except Exception:
                    continue
                _attrs = dict(_step_group.attrs)
                _mode = str(_attrs.get("stimulus_mode", ""))
                _moving = _mode_attrs(_step_group, _step_path, "moving_grating")
                _concentric = _mode_attrs(_step_group, _step_path, "concentric_grating")
                _raw_protocol = _parse_raw_protocol(_attrs.get("raw_protocol_params_json"))
                _step_rows.append(
                    {
                        "stimulus_run": _run_name,
                        "latest": "yes" if _run_name == _latest else "",
                        "step_index": _attrs.get("step_index"),
                        "step_name": _attrs.get("step_name", _step_name),
                        "stimulus_mode": _mode,
                        "start_camera_frame": _attrs.get("start_camera_frame"),
                        "end_camera_frame": _attrs.get("end_camera_frame"),
                        "duration_s": _attrs.get("duration_s"),
                        "moving_direction_camera_deg": _moving.get("grating_direction_camera_deg"),
                        "moving_orientation_authored_deg": _moving.get("orientation_degrees_authored"),
                        "camera_to_projector_offset_deg": _moving.get("camera_to_projector_offset_deg"),
                        "concentric_role": _concentric.get("stimulus_role"),
                        "concentric_polarity": _concentric.get("radial_polarity_authored"),
                        "concentric_center_x_px": _concentric.get("center_x_px"),
                        "concentric_center_y_px": _concentric.get("center_y_px"),
                        "protocol_param_type": _raw_protocol.get("parameters", {}).get("type")
                        if isinstance(_raw_protocol.get("parameters"), dict)
                        else None,
                    }
                )
    except Exception:
        _step_rows = []

    stimulus_step_df = pd.DataFrame(_step_rows)
    write_perf_event(
        "discover_stimulus_steps",
        time.perf_counter() - _stimulus_step_discovery_t0,
        zarr_path=zarr_path,
        n_stimulus_step_rows=len(stimulus_step_df),
    )
    if stimulus_step_df.empty:
        stimulus_step_view = mo.md("### Stimulus Step Metadata\n\nNo canonical stimulus step metadata found.")
    else:
        stimulus_step_view = mo.vstack(
            [
                mo.md("### Stimulus Step Metadata"),
                mo.ui.table(stimulus_step_df, selection=None, page_size=10),
            ]
        )
    stimulus_step_view
    return stimulus_step_df


@app.cell
def _(mo, time, write_perf_event, zarr, zarr_path):
    _stimulus_response_discovery_t0 = time.perf_counter()
    _stimulus_response_options = []

    def _group_keys(_group):
        _keys_fn = getattr(_group, "group_keys", None)
        if callable(_keys_fn):
            try:
                return sorted(str(_key) for _key in _keys_fn())
            except Exception:
                pass
        try:
            return sorted(str(_key) for _key in _group.keys())
        except Exception:
            return []

    def _local_child_group_names(_parent_path):
        if not _parent_path.exists():
            return []
        return sorted(
            _child.name
            for _child in _parent_path.iterdir()
            if _child.is_dir() and (_child / "zarr.json").exists()
        )

    def _open_child_group(_parent, _parent_path, _name):
        try:
            return _parent[_name]
        except Exception:
            _child_path = _parent_path / str(_name)
            if (_child_path / "zarr.json").exists():
                return zarr.open_group(str(_child_path), mode="r", use_consolidated=False)
            raise

    try:
        _root = zarr.open_group(str(zarr_path), mode="r", use_consolidated=False)
        _parent = _root["analysis/stimulus_response_runs"]
        _parent_path = zarr_path / "analysis" / "stimulus_response_runs"
        _latest = str(_parent.attrs.get("latest", ""))
        _run_names = sorted(set(_group_keys(_parent)) | set(_local_child_group_names(_parent_path)))
        for _run_name in _run_names:
            try:
                _run_group = _open_child_group(_parent, _parent_path, _run_name)
            except Exception:
                continue
            _moving_omr_step_count = 0
            _concentric_radial_step_count = 0
            _steps_path = _parent_path / _run_name / "steps"
            try:
                _steps_group = _run_group["steps"]
            except Exception:
                if (_steps_path / "zarr.json").exists():
                    _steps_group = zarr.open_group(str(_steps_path), mode="r", use_consolidated=False)
                else:
                    _steps_group = None
            if _steps_group is not None:
                _step_names = sorted(
                    set(_group_keys(_steps_group)) | set(_local_child_group_names(_steps_path))
                )
                for _step_key in _step_names:
                    try:
                        _step_group = _open_child_group(_steps_group, _steps_path, _step_key)
                    except Exception:
                        continue
                    if "grating" in _step_group and "omr" in _step_group["grating"]:
                        _moving_omr_step_count += 1
                    if (
                        "concentric_grating" in _step_group
                        and "radial_omr" in _step_group["concentric_grating"]
                    ):
                        _concentric_radial_step_count += 1
            _response_step_count = _moving_omr_step_count + _concentric_radial_step_count
            if _response_step_count == 0:
                continue
            _provenance = _run_group.attrs.get("provenance", {})
            _parameters = _provenance.get("parameters", {}) if isinstance(_provenance, dict) else {}
            _offset = _parameters.get("camera_to_projector_offset_deg")
            _source_track = str(_run_group.attrs.get("source_track_kinematics_run", "unknown"))
            _source_bouts = str(_run_group.attrs.get("source_bout_run", "none"))
            _label_parts = [
                _run_name,
                f"{_moving_omr_step_count} moving OMR steps",
                f"{_concentric_radial_step_count} radial OMR steps",
                f"offset {_offset} deg" if _offset is not None else "offset unknown",
                f"track {_source_track}",
            ]
            if _run_name == _latest:
                _label_parts.append("latest")
            _stimulus_response_options.append(
                {
                    "run_name": _run_name,
                    "run_path": f"analysis/stimulus_response_runs/{_run_name}",
                    "label": " | ".join(_label_parts),
                    "is_latest": _run_name == _latest,
                    "source_track_kinematics_run": _source_track,
                    "source_bout_run": _source_bouts,
                    "camera_to_projector_offset_deg": _offset,
                    "n_omr_steps": _response_step_count,
                    "n_moving_omr_steps": _moving_omr_step_count,
                    "n_concentric_radial_omr_steps": _concentric_radial_step_count,
                }
            )
    except Exception:
        _stimulus_response_options = []
    write_perf_event(
        "discover_stimulus_response_options",
        time.perf_counter() - _stimulus_response_discovery_t0,
        zarr_path=zarr_path,
        n_stimulus_response_options=len(_stimulus_response_options),
    )

    _none_label = "No stimulus-response / OMR run"
    stimulus_response_label_to_option = {_none_label: None}
    stimulus_response_label_to_option.update(
        {_option["label"]: _option for _option in _stimulus_response_options}
    )
    _default_label = next(
        (
            _option["label"]
            for _option in _stimulus_response_options
            if _option["is_latest"]
        ),
        _stimulus_response_options[0]["label"] if _stimulus_response_options else _none_label,
    )
    stimulus_response_picker = mo.ui.dropdown(
        options=list(stimulus_response_label_to_option),
        value=_default_label,
        label="Stimulus-response / OMR run",
    )
    mo.vstack(
        [
            mo.md("### Stimulus Response / OMR Selection"),
            stimulus_response_picker,
        ]
    )
    return stimulus_response_label_to_option, stimulus_response_picker


@app.cell
def _(
    OMR_BOUT_TRAJECTORY_PNG_ARTIFACT_NAME,
    OMR_SUMMARY_PNG_ARTIFACT_NAME,
    load_omr_step_summaries,
    np,
    pd,
    stimulus_response_label_to_option,
    stimulus_response_picker,
    time,
    write_perf_event,
    zarr,
    zarr_path,
):
    selected_stimulus_response = stimulus_response_label_to_option[stimulus_response_picker.value]
    _load_omr_t0 = time.perf_counter()

    def _first_value(_mapping, _name):
        _values = _mapping.get(_name)
        if _values is None:
            return np.nan
        _array = np.asarray(_values)
        if _array.size == 0:
            return np.nan
        return _array.ravel()[0].item() if hasattr(_array.ravel()[0], "item") else _array.ravel()[0]

    def _read_png_artifact(_run_group, _artifact_name):
        if "visualizations" not in _run_group:
            return b""
        _visualizations = _run_group["visualizations"]
        if _artifact_name not in _visualizations:
            return b""
        _artifact = _visualizations[_artifact_name]
        if not hasattr(_artifact, "shape"):
            return b""
        return np.asarray(_artifact[:], dtype=np.uint8).tobytes()

    def _group_keys(_group):
        _keys_fn = getattr(_group, "group_keys", None)
        if callable(_keys_fn):
            try:
                return sorted(str(_key) for _key in _keys_fn())
            except Exception:
                pass
        try:
            return sorted(str(_key) for _key in _group.keys())
        except Exception:
            return []

    def _step_sort_key(_name):
        try:
            return int(str(_name).rsplit("_", 1)[-1])
        except Exception:
            return str(_name)

    def _read_array_mapping(_group):
        _mapping = {}
        if _group is None:
            return _mapping
        for _name in _group.keys():
            try:
                _value = _group[_name]
            except Exception:
                continue
            if hasattr(_value, "shape"):
                _mapping[str(_name)] = np.asarray(_value[:])
        return _mapping

    def _read_child_array_mapping(_group, _child_name):
        if _child_name not in _group:
            return {}
        return _read_array_mapping(_group[_child_name])

    def _load_concentric_radial_omr_frames(_run_group):
        _step_rows = []
        _bout_frames = []
        _window_frames = []
        _early_window_frames = []
        if "steps" not in _run_group:
            return (
                pd.DataFrame(),
                pd.DataFrame(),
                pd.DataFrame(),
                pd.DataFrame(),
            )
        for _step_key in sorted(_group_keys(_run_group["steps"]), key=_step_sort_key):
            _step_group = _run_group["steps"][_step_key]
            if "concentric_grating" not in _step_group:
                continue
            _concentric_group = _step_group["concentric_grating"]
            if "radial_omr" not in _concentric_group:
                continue
            _radial_group = _concentric_group["radial_omr"]
            _step_attrs = dict(_step_group.attrs)
            _radial_attrs = dict(_radial_group.attrs)
            _per_fish = _read_child_array_mapping(_radial_group, "per_fish")
            _per_bout = _read_child_array_mapping(_radial_group, "per_bout")
            _windows = _read_child_array_mapping(_radial_group, "windows")
            _early_windows = _read_child_array_mapping(_radial_group, "early_windows")
            _center = _radial_attrs.get("stimulus_center_mm") or [np.nan, np.nan]
            _step_rows.append(
                {
                    "step_index": _step_attrs.get("step_index", _step_sort_key(_step_key)),
                    "step_name": _step_attrs.get("step_name", _step_key),
                    "start_frame": _step_attrs.get("start_frame", _step_attrs.get("start_camera_frame")),
                    "end_frame": _step_attrs.get("end_frame", _step_attrs.get("end_camera_frame")),
                    "duration_s": _step_attrs.get("duration_s"),
                    "method_version": _radial_attrs.get("method_version"),
                    "stimulus_radial_polarity": _radial_attrs.get("stimulus_radial_polarity"),
                    "stimulus_radial_sign": _radial_attrs.get("stimulus_radial_sign"),
                    "stimulus_radial_polarity_source": _radial_attrs.get("stimulus_radial_polarity_source"),
                    "stimulus_radial_polarity_validated": _radial_attrs.get(
                        "stimulus_radial_polarity_validated"
                    ),
                    "concentric_grating_role": _radial_attrs.get("concentric_grating_role"),
                    "stimulus_center_x_mm": _center[0] if len(_center) > 0 else np.nan,
                    "stimulus_center_y_mm": _center[1] if len(_center) > 1 else np.nan,
                    "arena_radius_mm": _radial_attrs.get("arena_radius_mm"),
                    "omr_path_index": _first_value(_per_fish, "omr_path_index"),
                    "radial_path_index": _first_value(_per_fish, "radial_path_index"),
                    "omr_net_direction_index": _first_value(_per_fish, "omr_net_direction_index"),
                    "tangential_bias_index": _first_value(_per_fish, "tangential_bias_index"),
                    "stimulus_aligned_radial_displacement_mm": _first_value(
                        _per_fish, "stimulus_aligned_radial_displacement_mm"
                    ),
                    "radial_displacement_integrated_mm": _first_value(
                        _per_fish, "radial_displacement_integrated_mm"
                    ),
                    "tangential_displacement_mm": _first_value(
                        _per_fish, "tangential_displacement_mm"
                    ),
                    "path_length_mm": _first_value(_per_fish, "path_length_mm"),
                    "start_radius_mm": _first_value(_per_fish, "start_radius_mm"),
                    "mean_radius_mm": _first_value(_per_fish, "mean_radius_mm"),
                    "end_radius_mm": _first_value(_per_fish, "end_radius_mm"),
                    "start_radius_norm": _first_value(_per_fish, "start_radius_norm"),
                    "mean_radius_norm": _first_value(_per_fish, "mean_radius_norm"),
                    "end_radius_norm": _first_value(_per_fish, "end_radius_norm"),
                    "bout_fraction_correct_classified": _first_value(
                        _per_fish, "bout_fraction_correct_classified"
                    ),
                    "bout_choice_index": _first_value(_per_fish, "bout_choice_index"),
                    "time_choice_index": _first_value(_per_fish, "time_choice_index"),
                    "bout_count_correct": _first_value(_per_fish, "bout_count_correct"),
                    "bout_count_opposing": _first_value(_per_fish, "bout_count_opposing"),
                    "bout_count_ambiguous": _first_value(_per_fish, "bout_count_ambiguous"),
                    "bout_count_total": _first_value(_per_fish, "bout_count_total"),
                    "first_aligned_bout_latency_s": _first_value(
                        _per_fish, "first_aligned_bout_latency_s"
                    ),
                    "first_opposing_bout_latency_s": _first_value(
                        _per_fish, "first_opposing_bout_latency_s"
                    ),
                    "coverage_fraction": _first_value(_per_fish, "coverage_fraction"),
                    "quality_flag": _first_value(_per_fish, "quality_flag"),
                }
            )
            for _mapping, _frames in (
                (_per_bout, _bout_frames),
                (_windows, _window_frames),
                (_early_windows, _early_window_frames),
            ):
                if not _mapping:
                    continue
                _df = pd.DataFrame({str(_key): np.asarray(_value) for _key, _value in _mapping.items()})
                _df.insert(0, "step_index", _step_attrs.get("step_index", _step_sort_key(_step_key)))
                _df.insert(1, "step_name", _step_attrs.get("step_name", _step_key))
                _df.insert(2, "stimulus_radial_polarity", _radial_attrs.get("stimulus_radial_polarity"))
                _df.insert(3, "stimulus_radial_sign", _radial_attrs.get("stimulus_radial_sign"))
                _frames.append(_df)
        return (
            pd.DataFrame(_step_rows),
            pd.concat(_bout_frames, ignore_index=True) if _bout_frames else pd.DataFrame(),
            pd.concat(_window_frames, ignore_index=True) if _window_frames else pd.DataFrame(),
            pd.concat(_early_window_frames, ignore_index=True) if _early_window_frames else pd.DataFrame(),
        )

    if selected_stimulus_response is None:
        stimulus_response_attrs = {}
        omr_step_df = pd.DataFrame()
        omr_bout_df = pd.DataFrame()
        omr_window_df = pd.DataFrame()
        omr_early_window_df = pd.DataFrame()
        concentric_omr_step_df = pd.DataFrame()
        concentric_omr_bout_df = pd.DataFrame()
        concentric_omr_window_df = pd.DataFrame()
        concentric_omr_early_window_df = pd.DataFrame()
        omr_summary_png_bytes = b""
        omr_trajectory_png_bytes = b""
    else:
        _root = zarr.open_group(str(zarr_path), mode="r", use_consolidated=False)
        _run_group = _root[selected_stimulus_response["run_path"]]
        stimulus_response_attrs = dict(_run_group.attrs)
        try:
            _steps = load_omr_step_summaries(_run_group)
        except Exception:
            _steps = []
        _step_rows = []
        _bout_frames = []
        _window_frames = []
        _early_window_frames = []
        for _step in _steps:
            _step_rows.append(
                {
                    "step_index": _step.step_index,
                    "step_name": _step.step_name,
                    "start_frame": _step.start_frame,
                    "end_frame": _step.end_frame,
                    "duration_s": _step.duration_s,
                    "stimulus_direction_deg": _step.stimulus_direction_deg,
                    "omr_path_index": _first_value(_step.per_fish, "omr_path_index"),
                    "omr_net_direction_index": _first_value(_step.per_fish, "omr_net_direction_index"),
                    "bout_fraction_correct_classified": _first_value(
                        _step.per_fish, "bout_fraction_correct_classified"
                    ),
                    "bout_choice_index": _first_value(_step.per_fish, "bout_choice_index"),
                    "bout_path_index": _first_value(_step.per_fish, "bout_path_index"),
                    "bout_fraction_correct_weighted_by_path": _first_value(
                        _step.per_fish, "bout_fraction_correct_weighted_by_path"
                    ),
                    "bout_fraction_correct_weighted_by_displacement": _first_value(
                        _step.per_fish, "bout_fraction_correct_weighted_by_displacement"
                    ),
                    "time_choice_index": _first_value(_step.per_fish, "time_choice_index"),
                    "start_position_axis_norm": _first_value(_step.per_fish, "start_position_axis_norm"),
                    "mean_position_axis_norm": _first_value(_step.per_fish, "mean_position_axis_norm"),
                    "end_position_axis_norm": _first_value(_step.per_fish, "end_position_axis_norm"),
                    "fraction_time_correct_side": _first_value(_step.per_fish, "fraction_time_correct_side"),
                    "first_aligned_bout_latency_s": _first_value(
                        _step.per_fish, "first_aligned_bout_latency_s"
                    ),
                    "first_opposing_bout_latency_s": _first_value(
                        _step.per_fish, "first_opposing_bout_latency_s"
                    ),
                    "bout_count_correct": _first_value(_step.per_fish, "bout_count_correct"),
                    "bout_count_opposing": _first_value(_step.per_fish, "bout_count_opposing"),
                    "bout_count_ambiguous": _first_value(_step.per_fish, "bout_count_ambiguous"),
                    "bout_count_total": _first_value(_step.per_fish, "bout_count_total"),
                }
            )
            if _step.per_bout:
                _bout_df = pd.DataFrame({str(_key): np.asarray(_value) for _key, _value in _step.per_bout.items()})
                _bout_df.insert(0, "step_index", _step.step_index)
                _bout_df.insert(1, "step_name", _step.step_name)
                _bout_frames.append(_bout_df)
            if _step.windows:
                _window_df = pd.DataFrame({str(_key): np.asarray(_value) for _key, _value in _step.windows.items()})
                _window_df.insert(0, "step_index", _step.step_index)
                _window_df.insert(1, "step_name", _step.step_name)
                _window_frames.append(_window_df)
            if _step.early_windows:
                _early_window_df = pd.DataFrame(
                    {str(_key): np.asarray(_value) for _key, _value in _step.early_windows.items()}
                )
                _early_window_df.insert(0, "step_index", _step.step_index)
                _early_window_df.insert(1, "step_name", _step.step_name)
                _early_window_frames.append(_early_window_df)
        omr_step_df = pd.DataFrame(_step_rows)
        omr_bout_df = pd.concat(_bout_frames, ignore_index=True) if _bout_frames else pd.DataFrame()
        omr_window_df = pd.concat(_window_frames, ignore_index=True) if _window_frames else pd.DataFrame()
        omr_early_window_df = (
            pd.concat(_early_window_frames, ignore_index=True)
            if _early_window_frames else pd.DataFrame()
        )
        omr_summary_png_bytes = _read_png_artifact(_run_group, OMR_SUMMARY_PNG_ARTIFACT_NAME)
        omr_trajectory_png_bytes = _read_png_artifact(_run_group, OMR_BOUT_TRAJECTORY_PNG_ARTIFACT_NAME)
        (
            concentric_omr_step_df,
            concentric_omr_bout_df,
            concentric_omr_window_df,
            concentric_omr_early_window_df,
        ) = _load_concentric_radial_omr_frames(_run_group)

    write_perf_event(
        "load_stimulus_response_omr",
        time.perf_counter() - _load_omr_t0,
        selected_stimulus_response_run=(
            selected_stimulus_response["run_name"] if selected_stimulus_response is not None else None
        ),
        n_omr_steps=len(omr_step_df),
        n_omr_bout_rows=len(omr_bout_df),
        n_omr_window_rows=len(omr_window_df),
        n_omr_early_window_rows=len(omr_early_window_df),
        n_concentric_radial_omr_steps=len(concentric_omr_step_df),
        n_concentric_radial_omr_bout_rows=len(concentric_omr_bout_df),
        n_concentric_radial_omr_window_rows=len(concentric_omr_window_df),
        n_concentric_radial_omr_early_window_rows=len(concentric_omr_early_window_df),
        has_summary_png=bool(omr_summary_png_bytes),
        has_trajectory_png=bool(omr_trajectory_png_bytes),
    )
    return (
        concentric_omr_bout_df,
        concentric_omr_early_window_df,
        concentric_omr_step_df,
        concentric_omr_window_df,
        omr_bout_df,
        omr_early_window_df,
        omr_step_df,
        omr_summary_png_bytes,
        omr_trajectory_png_bytes,
        omr_window_df,
        selected_stimulus_response,
        stimulus_response_attrs,
    )


@app.cell
def _(
    concentric_omr_bout_df,
    concentric_omr_early_window_df,
    concentric_omr_step_df,
    concentric_omr_window_df,
    go,
    mo,
    omr_bout_df,
    omr_early_window_df,
    omr_step_df,
    omr_summary_png_bytes,
    omr_trajectory_png_bytes,
    omr_window_df,
    pd,
    png_bytes_to_markdown_image,
    selected_stimulus_response,
    stimulus_response_attrs,
    time,
    write_perf_event,
):
    _omr_view_t0 = time.perf_counter()
    if selected_stimulus_response is None:
        stimulus_response_omr_view = mo.md("No stimulus-response / OMR run selected.")
    elif not len(omr_step_df) and not len(concentric_omr_step_df):
        stimulus_response_omr_view = mo.md("Selected stimulus-response run has no OMR step metrics.")
    else:
        _provenance = stimulus_response_attrs.get("provenance", {})
        _parameters = _provenance.get("parameters", {}) if isinstance(_provenance, dict) else {}

        _sections = [
            mo.md(
                f"""
                ## Stimulus Response / OMR Explorer

                **Run:** `{selected_stimulus_response["run_name"]}`

                **Source track:** `{selected_stimulus_response["source_track_kinematics_run"]}`

                **Source bouts:** `{selected_stimulus_response["source_bout_run"]}`

                **Moving-grating direction correction:** `{_parameters.get("camera_to_projector_offset_deg", "unknown")}` deg
                """
            ),
            mo.hstack(
                [
                    mo.stat(label="Moving OMR steps", value=f"{len(omr_step_df):,}"),
                    mo.stat(label="Moving OMR bouts", value=f"{len(omr_bout_df):,}"),
                    mo.stat(label="Radial OMR steps", value=f"{len(concentric_omr_step_df):,}"),
                    mo.stat(label="Radial OMR bouts", value=f"{len(concentric_omr_bout_df):,}"),
                ]
            ),
        ]

        if len(omr_step_df):
            _direction_fig = go.Figure()
            _x = omr_step_df["step_index"].astype(str)
            for _metric_name, _color in (
                ("omr_path_index", "#2a9d8f"),
                ("bout_path_index", "#43aa8b"),
                ("bout_choice_index", "#e76f51"),
                ("time_choice_index", "#457b9d"),
            ):
                if _metric_name not in omr_step_df:
                    continue
                _direction_fig.add_trace(
                    go.Bar(
                        x=_x,
                        y=omr_step_df[_metric_name],
                        name=_metric_name,
                        marker_color=_color,
                        customdata=omr_step_df[
                            [
                                "stimulus_direction_deg",
                                "bout_count_correct",
                                "bout_count_opposing",
                                "bout_count_total",
                            ]
                        ],
                        hovertemplate=(
                            "step %{x}<br>"
                            f"{_metric_name}: %{{y:.3f}}<br>"
                            "direction: %{customdata[0]:.1f} deg<br>"
                            "correct/opposing/total: %{customdata[1]} / %{customdata[2]} / %{customdata[3]}"
                            "<extra></extra>"
                        ),
                    )
                )
            _direction_fig.add_hline(y=0.0, line_width=1, line_color="rgba(0,0,0,0.45)")
            _direction_fig.update_layout(
                title="Moving-Grating OMR Direction Metrics",
                xaxis_title="Step index",
                yaxis_title="Signed OMR metric",
                yaxis=dict(range=[-1.05, 1.05]),
                barmode="group",
                height=420,
                margin=dict(l=52, r=20, t=58, b=80),
                legend=dict(orientation="h", yanchor="top", y=-0.22, xanchor="left", x=0.0),
            )

            _position_fig = go.Figure()
            for _metric_name, _mode in (
                ("start_position_axis_norm", "markers+lines"),
                ("mean_position_axis_norm", "markers+lines"),
                ("end_position_axis_norm", "markers+lines"),
                ("fraction_time_correct_side", "markers+lines"),
            ):
                if _metric_name not in omr_step_df:
                    continue
                _position_fig.add_trace(
                    go.Scatter(
                        x=_x,
                        y=omr_step_df[_metric_name],
                        mode=_mode,
                        name=_metric_name,
                    )
                )
            _position_fig.add_hline(y=0.0, line_width=1, line_color="rgba(0,0,0,0.45)")
            _position_fig.update_layout(
                title="Moving-Grating Axis Position and Correct-Side Occupancy",
                xaxis_title="Step index",
                yaxis_title="Normalized axis / fraction",
                height=380,
                margin=dict(l=52, r=20, t=58, b=80),
                legend=dict(orientation="h", yanchor="top", y=-0.24, xanchor="left", x=0.0),
            )

            _table_columns = [
                _column
                for _column in (
                    "step_index",
                    "step_name",
                    "stimulus_direction_deg",
                    "omr_path_index",
                    "bout_path_index",
                    "bout_choice_index",
                    "time_choice_index",
                    "bout_fraction_correct_weighted_by_path",
                    "bout_fraction_correct_weighted_by_displacement",
                    "bout_fraction_correct_classified",
                    "bout_count_correct",
                    "bout_count_opposing",
                    "bout_count_ambiguous",
                    "bout_count_total",
                    "start_position_axis_norm",
                    "mean_position_axis_norm",
                    "end_position_axis_norm",
                    "fraction_time_correct_side",
                    "first_aligned_bout_latency_s",
                    "first_opposing_bout_latency_s",
                )
                if _column in omr_step_df
            ]
            _bout_columns = [
                _column
                for _column in (
                    "step_index",
                    "fish_id",
                    "bout_id",
                    "start_frame",
                    "end_frame",
                    "per_bout_omr_score",
                    "correct_label",
                    "bout_path_length_mm",
                    "bout_displacement_mm",
                    "parallel_displacement_mm",
                )
                if _column in omr_bout_df
            ]
            _early_window_columns = [
                _column
                for _column in (
                    "step_index",
                    "fish_id",
                    "window_length_s",
                    "actual_window_length_s",
                    "omr_path_index",
                    "bout_path_index",
                    "bout_fraction_correct_weighted_by_path",
                    "time_choice_index",
                    "parallel_displacement_mm",
                    "path_length_mm",
                    "n_aligned_bouts",
                    "n_opposing_bouts",
                    "n_ambiguous_bouts",
                    "coverage_fraction",
                )
                if _column in omr_early_window_df
            ]
            _sections.extend(
                [
                    mo.md("### Moving-Grating OMR"),
                    _direction_fig,
                    _position_fig,
                    mo.accordion(
                        {
                            "Step OMR metrics": mo.ui.table(
                                omr_step_df[_table_columns],
                                selection=None,
                                page_size=10,
                            ),
                            "Per-bout OMR rows": (
                                mo.ui.table(
                                    omr_bout_df[_bout_columns],
                                    selection=None,
                                    page_size=15,
                                )
                                if len(omr_bout_df)
                                else mo.md("No per-bout OMR rows.")
                            ),
                            "Early-response windows": (
                                mo.ui.table(
                                    omr_early_window_df[_early_window_columns],
                                    selection=None,
                                    page_size=15,
                                )
                                if len(omr_early_window_df)
                                else mo.md("No early-response OMR windows.")
                            ),
                            "Persisted summary PNG": png_bytes_to_markdown_image(
                                omr_summary_png_bytes,
                                alt_text="OMR summary PNG",
                            ),
                            "Persisted trajectory PNG": png_bytes_to_markdown_image(
                                omr_trajectory_png_bytes,
                                alt_text="OMR bout trajectory PNG",
                            ),
                        }
                    ),
                ]
            )
        else:
            _sections.append(mo.md("### Moving-Grating OMR\n\nNo moving-grating OMR metrics in this run."))

        if len(concentric_omr_step_df):
            _radial_x = concentric_omr_step_df["step_index"].astype(str)
            _radial_direction_fig = go.Figure()
            for _metric_name, _color in (
                ("omr_path_index", "#2a9d8f"),
                ("radial_path_index", "#00a896"),
                ("omr_net_direction_index", "#277da1"),
                ("bout_choice_index", "#e76f51"),
                ("time_choice_index", "#457b9d"),
                ("tangential_bias_index", "#f4a261"),
            ):
                if _metric_name not in concentric_omr_step_df:
                    continue
                _radial_direction_fig.add_trace(
                    go.Bar(
                        x=_radial_x,
                        y=concentric_omr_step_df[_metric_name],
                        name=_metric_name,
                        marker_color=_color,
                        customdata=concentric_omr_step_df[
                            [
                                "stimulus_radial_polarity",
                                "stimulus_radial_polarity_validated",
                                "bout_count_correct",
                                "bout_count_opposing",
                                "bout_count_total",
                            ]
                        ],
                        hovertemplate=(
                            "step %{x}<br>"
                            f"{_metric_name}: %{{y:.3f}}<br>"
                            "polarity: %{customdata[0]}<br>"
                            "validated: %{customdata[1]}<br>"
                            "aligned/opposing/total: %{customdata[2]} / %{customdata[3]} / %{customdata[4]}"
                            "<extra></extra>"
                        ),
                    )
                )
            _radial_direction_fig.add_hline(y=0.0, line_width=1, line_color="rgba(0,0,0,0.45)")
            _radial_direction_fig.update_layout(
                title="Concentric Radial OMR Metrics",
                xaxis_title="Step index",
                yaxis_title="Signed radial metric",
                yaxis=dict(range=[-1.05, 1.05]),
                barmode="group",
                height=440,
                margin=dict(l=52, r=20, t=58, b=95),
                legend=dict(orientation="h", yanchor="top", y=-0.24, xanchor="left", x=0.0),
            )

            _radius_fig = go.Figure()
            for _metric_name, _label in (
                ("start_radius_norm", "start radius / arena"),
                ("mean_radius_norm", "mean radius / arena"),
                ("end_radius_norm", "end radius / arena"),
            ):
                if _metric_name not in concentric_omr_step_df:
                    continue
                _radius_fig.add_trace(
                    go.Scatter(
                        x=_radial_x,
                        y=concentric_omr_step_df[_metric_name],
                        mode="markers+lines",
                        name=_label,
                    )
                )
            _radius_fig.update_layout(
                title="Concentric Step Radius Summary",
                xaxis_title="Step index",
                yaxis_title="Normalized radius",
                yaxis=dict(range=[0.0, 1.05]),
                height=360,
                margin=dict(l=52, r=20, t=58, b=80),
                legend=dict(orientation="h", yanchor="top", y=-0.24, xanchor="left", x=0.0),
            )

            _radial_step_columns = [
                _column
                for _column in (
                    "step_index",
                    "step_name",
                    "stimulus_radial_polarity",
                    "stimulus_radial_sign",
                    "stimulus_radial_polarity_source",
                    "stimulus_radial_polarity_validated",
                    "omr_path_index",
                    "radial_path_index",
                    "omr_net_direction_index",
                    "tangential_bias_index",
                    "bout_choice_index",
                    "time_choice_index",
                    "bout_fraction_correct_classified",
                    "bout_count_correct",
                    "bout_count_opposing",
                    "bout_count_ambiguous",
                    "bout_count_total",
                    "start_radius_mm",
                    "mean_radius_mm",
                    "end_radius_mm",
                    "start_radius_norm",
                    "mean_radius_norm",
                    "end_radius_norm",
                    "first_aligned_bout_latency_s",
                    "first_opposing_bout_latency_s",
                    "coverage_fraction",
                    "quality_flag",
                )
                if _column in concentric_omr_step_df
            ]
            _radial_bout_columns = [
                _column
                for _column in (
                    "step_index",
                    "fish_id",
                    "bout_id",
                    "start_frame",
                    "end_frame",
                    "stimulus_radial_polarity",
                    "radial_omr_score",
                    "radial_net_direction_score",
                    "tangential_bias_score",
                    "omr_label",
                    "start_radius_mm",
                    "end_radius_mm",
                    "mean_radius_mm",
                    "stimulus_aligned_radial_displacement_mm",
                    "radial_displacement_integrated_mm",
                    "tangential_displacement_mm",
                    "path_length_mm",
                    "valid_radial_basis",
                    "quality_flag",
                )
                if _column in concentric_omr_bout_df
            ]
            _radial_window_columns = [
                _column
                for _column in (
                    "step_index",
                    "fish_id",
                    "window_id",
                    "start_time_s",
                    "end_time_s",
                    "window_length_s",
                    "stimulus_radial_polarity",
                    "omr_path_index",
                    "time_choice_index",
                    "mean_radius_norm",
                    "n_bouts",
                    "coverage_fraction",
                    "quality_flag",
                )
                if _column in concentric_omr_window_df
            ]
            _radial_early_columns = [
                _column
                for _column in _radial_window_columns
                if _column in concentric_omr_early_window_df
            ]
            _sections.extend(
                [
                    mo.md(
                        """
                        ### Concentric Radial OMR

                        Positive `omr_path_index` means motion aligned to the persisted radial polarity
                        (`expanding` = outward, `contracting` = inward). The `radial_path_index`
                        column remains outward-positive independent of stimulus polarity.
                        """
                    ),
                    _radial_direction_fig,
                    _radius_fig,
                    mo.accordion(
                        {
                            "Step radial OMR metrics": mo.ui.table(
                                concentric_omr_step_df[_radial_step_columns],
                                selection=None,
                                page_size=10,
                            ),
                            "Per-bout radial OMR rows": (
                                mo.ui.table(
                                    concentric_omr_bout_df[_radial_bout_columns],
                                    selection=None,
                                    page_size=15,
                                )
                                if len(concentric_omr_bout_df)
                                else mo.md("No per-bout radial OMR rows.")
                            ),
                            "Radial OMR windows": (
                                mo.ui.table(
                                    concentric_omr_window_df[_radial_window_columns],
                                    selection=None,
                                    page_size=15,
                                )
                                if len(concentric_omr_window_df)
                                else mo.md("No radial OMR window rows.")
                            ),
                            "Early radial OMR windows": (
                                mo.ui.table(
                                    concentric_omr_early_window_df[_radial_early_columns],
                                    selection=None,
                                    page_size=15,
                                )
                                if len(concentric_omr_early_window_df)
                                else mo.md("No early radial OMR windows.")
                            ),
                        }
                    ),
                ]
            )
        else:
            _sections.append(mo.md("### Concentric Radial OMR\n\nNo concentric radial OMR metrics in this run."))

        _sections.append(
            mo.accordion({"Stimulus-response attrs": mo.tree(dict(stimulus_response_attrs))})
        )
        stimulus_response_omr_view = mo.vstack(_sections)
    write_perf_event(
        "build_stimulus_response_omr_view",
        time.perf_counter() - _omr_view_t0,
        selected_stimulus_response_run=(
            selected_stimulus_response["run_name"] if selected_stimulus_response is not None else None
        ),
        n_omr_steps=len(omr_step_df),
        n_omr_bout_rows=len(omr_bout_df),
        n_omr_window_rows=len(omr_window_df),
        n_concentric_radial_omr_steps=len(concentric_omr_step_df),
        n_concentric_radial_omr_bout_rows=len(concentric_omr_bout_df),
        n_concentric_radial_omr_window_rows=len(concentric_omr_window_df),
    )
    stimulus_response_omr_view
    return


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
    bout_classification_category_picker,
    bout_classification_df,
    filtered_swim_bout_df,
    pd,
    time,
    write_perf_event,
):
    _classification_filter_t0 = time.perf_counter()
    if len(bout_classification_df):
        _selected_classification_categories = set(str(_classification_category) for _classification_category in bout_classification_category_picker.value)
        _classification_df_for_filter = bout_classification_df.copy()
        if _selected_classification_categories and "category_label" in _classification_df_for_filter:
            _classification_df_for_filter = _classification_df_for_filter[
                _classification_df_for_filter["category_label"].astype(str).isin(_selected_classification_categories)
            ].copy()
        filtered_bout_classification_df = _classification_df_for_filter
    else:
        filtered_bout_classification_df = bout_classification_df

    if len(filtered_swim_bout_df) and len(filtered_bout_classification_df):
        _visible_bouts = filtered_swim_bout_df.copy()
        _visible_bouts["source_bout_row"] = _visible_bouts.index.astype("int64")
        _classification_columns = [
            _classification_column
            for _classification_column in (
                "source_bout_row",
                "source_bout_id",
                "category_label",
                "probability",
                "classified",
                "valid",
                "failure_reason",
                "tail_valid_fraction",
                "traj_valid_fraction",
            )
            if _classification_column in filtered_bout_classification_df.columns
        ]
        classified_filtered_swim_bout_df = _visible_bouts.merge(
            filtered_bout_classification_df[_classification_columns],
            on="source_bout_row",
            how="inner",
        )
    else:
        classified_filtered_swim_bout_df = pd.DataFrame()

    write_perf_event(
        "filter_bout_classification",
        time.perf_counter() - _classification_filter_t0,
        n_classification_rows_in=len(bout_classification_df),
        n_classification_rows_out=len(filtered_bout_classification_df),
        n_visible_classified_bouts=len(classified_filtered_swim_bout_df),
    )
    return classified_filtered_swim_bout_df, filtered_bout_classification_df


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
    apply_full_width_timeseries_layout,
    classified_filtered_swim_bout_df,
    filtered_bout_classification_df,
    filtered_timeseries_df,
    go,
    mo,
    np,
    pd,
    px,
    selected_bout_classification,
    speed_series_picker,
    swim_bout_boundary_mode,
    swim_bout_end_col,
    swim_bout_start_col,
    time,
    write_perf_event,
):
    _classification_figure_t0 = time.perf_counter()
    if selected_bout_classification is None:
        bout_classification_output = mo.md("No bout-classification run selected.")
        bout_classification_category_counts_df = pd.DataFrame(columns=["category_label", "count"])
    elif not len(filtered_bout_classification_df):
        bout_classification_output = mo.md("No bout-classification rows match the selected filters.")
        bout_classification_category_counts_df = pd.DataFrame(columns=["category_label", "count"])
    else:
        bout_classification_category_counts_df = (
            filtered_bout_classification_df["category_label"]
            .astype(str)
            .value_counts()
            .rename_axis("category_label")
            .reset_index(name="count")
        )
        _classification_count_fig = px.bar(
            bout_classification_category_counts_df,
            x="category_label",
            y="count",
            color="category_label",
            title="Bout Classification Counts",
            labels={"category_label": "Category", "count": "Bout count"},
        )
        _classification_count_fig.update_layout(
            height=420,
            margin=dict(l=40, r=20, t=60, b=120),
            showlegend=False,
        )
        _classification_count_fig.update_xaxes(tickangle=35)

        _classification_timeline_fig = go.Figure()
        _speed_trace_columns = [
            _classification_speed_column
            for _classification_speed_column in speed_series_picker.value[:1]
            if _classification_speed_column in filtered_timeseries_df
        ]
        for _classification_speed_column in _speed_trace_columns:
            _classification_timeline_fig.add_trace(
                go.Scattergl(
                    x=filtered_timeseries_df["time_s"],
                    y=filtered_timeseries_df[_classification_speed_column],
                    mode="lines",
                    name=_classification_speed_column,
                    line=dict(color="rgba(40, 58, 70, 0.85)", width=1.4),
                )
            )
        if len(classified_filtered_swim_bout_df):
            _palette = px.colors.qualitative.Dark24 + px.colors.qualitative.Set3
            _categories = sorted(classified_filtered_swim_bout_df["category_label"].astype(str).unique())
            _color_by_category = {
                _classification_category: _palette[_classification_idx % len(_palette)]
                for _classification_idx, _classification_category in enumerate(_categories)
            }
            for _classification_category in _categories:
                _category_rows = classified_filtered_swim_bout_df[
                    classified_filtered_swim_bout_df["category_label"].astype(str) == _classification_category
                ]
                _starts = _category_rows[swim_bout_start_col].to_numpy(dtype=float)
                _ends = _category_rows[swim_bout_end_col].to_numpy(dtype=float)
                _widths = _ends - _starts
                _valid_widths = np.isfinite(_starts) & np.isfinite(_ends) & (_widths > 0)
                if not _valid_widths.any():
                    continue
                _custom = np.stack(
                    [
                        _category_rows["source_bout_row"].to_numpy(dtype=object),
                        _category_rows["probability"].to_numpy(dtype=object)
                        if "probability" in _category_rows
                        else np.full(len(_category_rows), np.nan, dtype=object),
                        _category_rows["failure_reason"].to_numpy(dtype=object)
                        if "failure_reason" in _category_rows
                        else np.full(len(_category_rows), "", dtype=object),
                    ],
                    axis=1,
                )
                _classification_timeline_fig.add_trace(
                    go.Bar(
                        x=_starts[_valid_widths] + (_widths[_valid_widths] / 2.0),
                        y=np.ones(int(np.count_nonzero(_valid_widths))),
                        width=_widths[_valid_widths],
                        base=np.zeros(int(np.count_nonzero(_valid_widths))),
                        yaxis="y2",
                        marker=dict(color=_color_by_category[_classification_category], line=dict(width=0)),
                        opacity=0.45,
                        customdata=_custom[_valid_widths],
                        hovertemplate=(
                            f"{_classification_category}<br>"
                            "row %{customdata[0]}<br>"
                            "probability %{customdata[1]:.3f}<br>"
                            "%{customdata[2]}<extra></extra>"
                        ),
                        name=_classification_category,
                    )
                )
        apply_full_width_timeseries_layout(
            _classification_timeline_fig,
            title=f"Bout Classification Timeline ({swim_bout_boundary_mode})",
            yaxis_title="Speed / detection signal",
            height=460,
        )
        _classification_timeline_fig.update_layout(bargap=0)

        _classification_table_columns = [
            _classification_column
            for _classification_column in (
                "source_bout_row",
                "source_bout_id",
                "category_label",
                "probability",
                "classified",
                "valid",
                "failure_reason",
                "tail_valid_fraction",
                "traj_valid_fraction",
            )
            if _classification_column in filtered_bout_classification_df.columns
        ]
        bout_classification_output = mo.vstack(
            [
                _classification_count_fig,
                _classification_timeline_fig,
                mo.md("### Visible Classification Rows"),
                mo.ui.table(
                    filtered_bout_classification_df[_classification_table_columns],
                    selection=None,
                    page_size=10,
                ),
            ]
        )
    write_perf_event(
        "build_bout_classification_view",
        time.perf_counter() - _classification_figure_t0,
        selected_bout_classification_run=(
            selected_bout_classification.run_name if selected_bout_classification is not None else None
        ),
        n_classification_rows=len(filtered_bout_classification_df),
        n_visible_classified_bouts=len(classified_filtered_swim_bout_df),
        n_categories=(
            int(bout_classification_category_counts_df["category_label"].nunique())
            if len(bout_classification_category_counts_df)
            else 0
        ),
    )
    bout_classification_output
    return (bout_classification_category_counts_df,)


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
    bout_classification_df,
    bout_kinematics_df,
    classified_filtered_swim_bout_df,
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
            mo.stat(label="Classified bouts", value=f"{len(bout_classification_df):,}"),
            mo.stat(label="Visible classified", value=f"{len(classified_filtered_swim_bout_df):,}"),
            mo.stat(label="Invalid intervals", value=f"{len(validity_df):,}"),
            mo.stat(label="Visible invalid", value=f"{len(filtered_validity_df):,}"),
            mo.stat(label="Track ID", value=str(data.spec.get("track_id", "unknown"))),
            mo.stat(label="Position unit", value=data.position_unit),
        ]
    )
    return


@app.cell
def _(
    bout_classification_attrs,
    bout_classification_df,
    bout_kinematics_attrs,
    bout_kinematics_df,
    data,
    inter_bout_interval_df,
    mo,
    swim_bout_df,
):
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
            "Bout classification": mo.tree(
                {
                    "rows": int(len(bout_classification_df)),
                    "columns": list(bout_classification_df.columns),
                    "attrs": dict(bout_classification_attrs),
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
