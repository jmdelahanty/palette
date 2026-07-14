#!/usr/bin/env python3
"""Marimo app for chaser-protocol dashboard specs.

Run with:

    scripts/py -m marimo run apps/marimo/goodcopbadcop_explorer.py -- \
      --zarr-path /path/to/archive.zarr
"""

import marimo

__generated_with = "0.23.3"
app = marimo.App(width="full")


@app.cell
def _():
    import base64
    from pathlib import Path
    import time

    import marimo as mo
    import numpy as np
    import polars as pl
    import plotly.express as px
    import plotly.graph_objects as go

    from fisheye.utils.view_zarr_visualization import load_png_artifact_bytes
    from fisheye.shared.zarr_io import open_zarr_root
    from fisheye.visualization.goodcopbadcop_interactive import (
        DEFAULT_CHASER_DASHBOARD_INTERACTIVE_ARTIFACT,
        discover_chaser_dashboard_options,
        load_chaser_dashboard_data,
        to_chaser_position_dataframe,
        to_distance_timeseries_dataframe,
        to_position_dataframe,
        to_window_dataframe,
    )

    def png_bytes_to_markdown_image(png_bytes, *, alt_text):
        if not png_bytes:
            return mo.md(f"*No PNG bytes available for `{alt_text}`.*")
        encoded = base64.b64encode(bytes(png_bytes)).decode("ascii")
        return mo.md(
            f'<img alt="{alt_text}" src="data:image/png;base64,{encoded}" '
            'style="max-width:100%; height:auto; border: 1px solid #ddd;" />'
        )

    def add_epoch_overlays(fig, windows_df):
        if not len(windows_df):
            return
        for row in windows_df.iter_rows(named=True):
            fig.add_vrect(
                x0=float(row["start_time_s"]),
                x1=float(row["end_time_s"]),
                fillcolor="rgba(80, 120, 180, 0.08)",
                line_width=0,
                layer="below",
            )

    return (
        DEFAULT_CHASER_DASHBOARD_INTERACTIVE_ARTIFACT,
        Path,
        add_epoch_overlays,
        discover_chaser_dashboard_options,
        go,
        load_chaser_dashboard_data,
        load_png_artifact_bytes,
        mo,
        np,
        open_zarr_root,
        pl,
        png_bytes_to_markdown_image,
        px,
        time,
        to_chaser_position_dataframe,
        to_distance_timeseries_dataframe,
        to_position_dataframe,
        to_window_dataframe,
    )


@app.cell
def _(DEFAULT_CHASER_DASHBOARD_INTERACTIVE_ARTIFACT, Path, discover_chaser_dashboard_options, mo):
    cli_args = mo.cli_args()
    zarr_path_raw = cli_args.get("zarr-path")
    artifact = str(cli_args.get("artifact", DEFAULT_CHASER_DASHBOARD_INTERACTIVE_ARTIFACT))
    initial_run_path = cli_args.get("run-path")
    if not zarr_path_raw:
        raise ValueError(
            "Required CLI arg is missing. Run with: "
            "scripts/py -m marimo run apps/marimo/goodcopbadcop_explorer.py -- "
            "--zarr-path <archive.zarr>"
        )
    zarr_path = Path(str(zarr_path_raw))
    options = discover_chaser_dashboard_options(zarr_path, artifact_name=artifact)
    if not options:
        raise ValueError(
            "No chaser-protocol dashboard interactive artifacts were found. "
            "Refresh chaser-distance runs so they write chaser_protocol_dashboard_interactive."
        )
    return artifact, initial_run_path, options, zarr_path


@app.cell
def _(initial_run_path, mo, options):
    def _matches_initial_run(option):
        if initial_run_path is None:
            return False
        wanted = str(initial_run_path).strip("/")
        return option.run_path.strip("/") == wanted or option.run_name == wanted.split("/")[-1]

    label_to_option = {option.label: option for option in options}
    default_label = next(
        (option.label for option in options if _matches_initial_run(option)),
        next((option.label for option in options if option.is_latest), options[0].label),
    )
    run_picker = mo.ui.dropdown(
        options=list(label_to_option),
        value=default_label,
        label="Chaser distance run",
    )
    run_picker
    return label_to_option, run_picker


@app.cell
def _(
    artifact,
    label_to_option,
    load_chaser_dashboard_data,
    run_picker,
    time,
    to_chaser_position_dataframe,
    to_distance_timeseries_dataframe,
    to_position_dataframe,
    to_window_dataframe,
    zarr_path,
):
    selected_run = label_to_option[run_picker.value]
    _load_t0 = time.perf_counter()
    data = load_chaser_dashboard_data(
        zarr_path,
        run_path=selected_run.run_path,
        artifact_name=artifact,
    )
    distance_df = to_distance_timeseries_dataframe(data)
    position_df = to_position_dataframe(data)
    windows_df = to_window_dataframe(data)
    chaser_position_df = to_chaser_position_dataframe(data, sample_step=max(1, int(data.fps // 2) or 1))
    load_duration_ms = (time.perf_counter() - _load_t0) * 1000.0
    return chaser_position_df, data, distance_df, load_duration_ms, position_df, selected_run, windows_df


@app.cell
def _(data, distance_df, load_duration_ms, mo, position_df, selected_run, windows_df):
    mo.vstack(
        [
            mo.md(f"# Chaser Protocol Dashboard\n\n`{data.zarr_path}`"),
            mo.hstack(
                [
                    mo.stat(label="Run", value=selected_run.run_name),
                    mo.stat(label="Frames", value=f"{data.camera_frame_id.shape[0]:,}"),
                    mo.stat(label="Chasers", value=f"{data.chaser_indices.shape[0]:,}"),
                    mo.stat(label="Windows", value=f"{len(windows_df):,}"),
                    mo.stat(label="Position rows", value=f"{len(position_df):,}"),
                    mo.stat(label="Distance rows", value=f"{len(distance_df):,}"),
                    mo.stat(label="Load ms", value=f"{load_duration_ms:.1f}"),
                ]
            ),
        ]
    )
    return


@app.cell
def _(data, distance_df, mo, np, windows_df):
    distance_columns = [
        column
        for column in distance_df.columns
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
    max_time = float(data.time_seconds[-1]) if data.time_seconds.size else 0.0
    time_window = mo.ui.range_slider(
        start=0.0,
        stop=max_time,
        value=[0.0, max_time],
        step=max(max_time / 1000.0, 0.001),
        label="Time window (s)",
    )
    epoch_options = {"Custom time window": None}
    for _row in windows_df.iter_rows(named=True):
        epoch_options[f'{_row["label"]} ({_row["start_time_s"]:.1f}-{_row["end_time_s"]:.1f}s)'] = int(_row["window_id"])
    epoch_picker = mo.ui.dropdown(
        options=list(epoch_options),
        value="Custom time window",
        label="Epoch",
    )
    heatmap_bins = mo.ui.slider(start=20, stop=160, value=80, step=10, label="Heatmap bins")
    chaser_overlay = mo.ui.checkbox(value=True, label="Chaser overlay")
    return chaser_overlay, distance_series_picker, epoch_options, epoch_picker, heatmap_bins, time_window


@app.cell
def _(chaser_overlay, distance_series_picker, epoch_options, epoch_picker, heatmap_bins, mo, time_window):
    controls = [distance_series_picker, epoch_picker]
    if epoch_options.get(epoch_picker.value) is None:
        controls.append(time_window)
    controls.extend([heatmap_bins, chaser_overlay])
    mo.vstack(controls)
    return


@app.cell
def _(epoch_options, epoch_picker, pl, time_window, windows_df):
    selected_epoch_id = epoch_options[epoch_picker.value]
    if selected_epoch_id is None or not len(windows_df):
        start_s, stop_s = [float(value) for value in time_window.value]
        selected_epoch_label = "custom"
    else:
        _row = windows_df.filter(
            pl.col("window_id").cast(pl.Int64) == int(selected_epoch_id)
        ).row(0, named=True)
        start_s = float(_row["start_time_s"])
        stop_s = float(_row["end_time_s"])
        selected_epoch_label = str(_row["label"])
    return selected_epoch_id, selected_epoch_label, start_s, stop_s


@app.cell
def _(
    add_epoch_overlays,
    distance_df,
    distance_series_picker,
    go,
    pl,
    selected_epoch_label,
    start_s,
    stop_s,
    windows_df,
):
    visible = distance_df.filter(
        pl.col("time_s").is_between(start_s, stop_s, closed="both")
    )
    distance_fig = go.Figure()
    add_epoch_overlays(distance_fig, windows_df)
    for column in distance_series_picker.value:
        if column not in visible.columns:
            continue
        distance_fig.add_trace(
            go.Scattergl(
                x=visible.get_column("time_s").to_numpy(),
                y=visible.get_column(column).to_numpy(),
                mode="lines",
                name=column,
            )
        )
    distance_fig.update_layout(
        title=f"Fish-to-Chaser Distance ({selected_epoch_label})",
        xaxis_title="Time (s)",
        yaxis_title="Distance (mm)",
        hovermode="x unified",
        height=430,
        margin=dict(l=56, r=20, t=56, b=70),
        legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="left", x=0.0),
    )
    distance_fig
    return (visible,)


@app.cell
def _(
    chaser_overlay,
    chaser_position_df,
    heatmap_bins,
    position_df,
    pl,
    px,
    selected_epoch_label,
    start_s,
    stop_s,
):
    visible_positions = position_df.filter(
        pl.col("time_s").is_between(start_s, stop_s, closed="both")
        & pl.col("fish_valid").cast(pl.Boolean)
    )
    if len(visible_positions):
        arena_heatmap = px.density_heatmap(
            {
                "x": visible_positions.get_column("x").to_numpy(),
                "y": visible_positions.get_column("y").to_numpy(),
            },
            x="x",
            y="y",
            nbinsx=int(heatmap_bins.value),
            nbinsy=int(heatmap_bins.value),
            title=f"Arena Occupancy ({selected_epoch_label})",
            labels={"x": "Arena X (px)", "y": "Arena Y (px, down)"},
        )
        if chaser_overlay.value and len(chaser_position_df):
            visible_chasers = chaser_position_df.filter(
                pl.col("time_s").is_between(start_s, stop_s, closed="both")
                & pl.col("chaser_valid").cast(pl.Boolean)
            )
            if len(visible_chasers):
                for rows in visible_chasers.partition_by(
                    "chaser_index", maintain_order=True
                ):
                    chaser_index = int(rows.get_column("chaser_index")[0])
                    arena_heatmap.add_scattergl(
                        x=rows.get_column("x").to_numpy(),
                        y=rows.get_column("y").to_numpy(),
                        mode="markers",
                        marker=dict(size=5, opacity=0.5),
                        name=f"chaser {int(chaser_index)}",
                    )
        arena_heatmap.update_yaxes(scaleanchor="x", scaleratio=1, autorange="reversed")
        arena_heatmap.update_layout(height=590, margin=dict(l=52, r=20, t=58, b=64))
    else:
        arena_heatmap = "No valid fish positions in the selected window."
    arena_heatmap
    return (visible_positions,)


@app.cell
def _(data, go, mo, np, pl, selected_epoch_id, windows_df):
    if data.occupancy_normalized is None or data.occupancy_x_edges is None or data.occupancy_y_edges is None:
        occupancy_output = mo.md("No persisted detection-occupancy heatmap cube is linked to this spec.")
    elif not len(windows_df):
        occupancy_output = mo.md("No epoch windows are available for persisted detection-occupancy heatmaps.")
    else:
        window_idx = 0
        if selected_epoch_id is not None:
            matches = np.flatnonzero(
                windows_df.get_column("window_id").cast(pl.Int64).to_numpy()
                == int(selected_epoch_id)
            )
            window_idx = int(matches[0]) if matches.size else 0
        window_idx = max(0, min(window_idx, int(data.occupancy_normalized.shape[0]) - 1))
        label = str(windows_df.get_column("label")[window_idx]) if window_idx < len(windows_df) else f"window {window_idx}"
        x_edges = np.asarray(data.occupancy_x_edges, dtype=float)
        y_edges = np.asarray(data.occupancy_y_edges, dtype=float)
        x_centers = (x_edges[:-1] + x_edges[1:]) / 2.0 if x_edges.size > 1 else np.arange(data.occupancy_normalized.shape[2])
        y_centers = (y_edges[:-1] + y_edges[1:]) / 2.0 if y_edges.size > 1 else np.arange(data.occupancy_normalized.shape[1])
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
        occupancy_output = occupancy_fig
    occupancy_output
    return


@app.cell
def _(data, load_png_artifact_bytes, mo, open_zarr_root, png_bytes_to_markdown_image):
    root = open_zarr_root(data.zarr_path, mode="r")

    def _resolve_artifact_path(path):
        if not path:
            return None
        text = str(path).strip("/")
        if text.startswith("visualizations/"):
            return f"{data.run_path}/{text}"
        return text

    snapshot_views = {}
    for key, raw_path in dict(data.spec.get("static_artifacts", {})).items():
        artifact_path = _resolve_artifact_path(raw_path)
        if not artifact_path:
            continue
        try:
            _resolved, png_bytes = load_png_artifact_bytes(root, artifact_path)
        except Exception:
            continue
        snapshot_views[str(key)] = png_bytes_to_markdown_image(png_bytes, alt_text=str(key))
    if snapshot_views:
        static_output = mo.accordion(snapshot_views)
    else:
        static_output = mo.md("No static snapshots resolved for this spec.")
    static_output
    return


@app.cell
def _(data, mo, visible, visible_positions, windows_df):
    mo.accordion(
        {
            "Windows": mo.ui.table(windows_df, selection=None, page_size=10),
            "Visible distance rows": mo.ui.table(visible.head(1000), selection=None, page_size=15),
            "Visible position rows": mo.ui.table(visible_positions.head(1000), selection=None, page_size=15),
            "Interactive spec": mo.tree(dict(data.spec)),
            "Artifact attrs": mo.tree(dict(data.attrs)),
        }
    )
    return


if __name__ == "__main__":
    app.run()
