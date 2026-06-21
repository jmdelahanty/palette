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
    to_window_dataframe,
)

from .common import add_epoch_overlays, apply_full_width_timeseries_layout
from .registry import InteractiveSpecOption


@dataclass(frozen=True)
class GoodCopBadCopLoadedView:
    data: GoodCopBadCopInteractiveData
    distance_df: pd.DataFrame
    position_df: pd.DataFrame
    windows_df: pd.DataFrame
    chaser_position_df: pd.DataFrame
    load_duration_ms: float


@dataclass(frozen=True)
class GoodCopBadCopControls:
    distance_series_picker: Any
    time_window: Any
    epoch_picker: Any
    epoch_options: Mapping[str, Optional[int]]
    heatmap_bins: Any
    chaser_overlay: Any
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
    return GoodCopBadCopLoadedView(
        data=data,
        distance_df=distance_df,
        position_df=position_df,
        windows_df=windows_df,
        chaser_position_df=chaser_position_df,
        load_duration_ms=(timer.perf_counter() - load_t0) * 1000.0,
    )


def build_summary(mo: Any, *, loaded: GoodCopBadCopLoadedView, option: InteractiveSpecOption) -> Any:
    data = loaded.data
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
    return GoodCopBadCopControls(
        distance_series_picker=distance_series_picker,
        time_window=time_window,
        epoch_picker=epoch_picker,
        epoch_options=epoch_options,
        heatmap_bins=heatmap_bins,
        chaser_overlay=chaser_overlay,
        view=build_controls_panel_from_widgets(
            mo,
            distance_series_picker=distance_series_picker,
            time_window=time_window,
            epoch_picker=epoch_picker,
            epoch_options=epoch_options,
            heatmap_bins=heatmap_bins,
            chaser_overlay=chaser_overlay,
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
) -> Any:
    items = [distance_series_picker, epoch_picker]
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
                arena_heatmap.add_scattergl(
                    x=rows["x"],
                    y=rows["y"],
                    mode="markers",
                    marker=dict(size=5, opacity=0.5),
                    name=f"chaser {int(chaser_index)}",
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
        }
    )
