"""GoodCopBadCop chaser dashboard component for Palette marimo explorers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np
import pandas as pd

from fisheye.visualization.goodcopbadcop_interactive import (
    GOODCOPBADCOP_CHASER_DASHBOARD_RENDERER,
    GoodCopBadCopInteractiveData,
    load_goodcopbadcop_interactive_data,
    to_chaser_position_dataframe,
    to_distance_timeseries_dataframe,
    to_position_dataframe,
    to_spatial_occupancy_dataframe,
    to_window_dataframe,
)

from .common import add_epoch_overlays, apply_full_width_timeseries_layout
from .registry import InteractiveSpecOption


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
    distance_df: pd.DataFrame
    position_df: pd.DataFrame
    windows_df: pd.DataFrame
    chaser_position_df: pd.DataFrame
    spatial_occupancy_df: pd.DataFrame
    load_duration_ms: float


@dataclass(frozen=True)
class GoodCopBadCopControls:
    distance_series_picker: Any
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
    distance_df = to_distance_timeseries_dataframe(data)
    position_df = to_position_dataframe(data)
    windows_df = to_window_dataframe(data)
    chaser_position_df = to_chaser_position_dataframe(
        data,
        sample_step=max(1, int(data.fps // 2) or 1),
    )
    spatial_occupancy_df = to_spatial_occupancy_dataframe(data)
    return GoodCopBadCopLoadedView(
        data=data,
        distance_df=distance_df,
        position_df=position_df,
        windows_df=windows_df,
        chaser_position_df=chaser_position_df,
        spatial_occupancy_df=spatial_occupancy_df,
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
    spatial_zone_set_picker: Any = None,
) -> Any:
    items = [distance_series_picker, epoch_picker]
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
        }
    )
