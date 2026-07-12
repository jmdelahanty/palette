"""Pure panel routing and Plotly helpers for the group analytics Marimo app."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence
import zlib

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


@dataclass(frozen=True)
class GroupPanelDefinition:
    panel_id: str
    label: str
    description: str
    all_capabilities: tuple[str, ...] = ()
    any_capabilities: tuple[str, ...] = ()
    requires_statistics: bool = False

    def available(self, capabilities: set[str], *, statistics_available: bool) -> bool:
        if self.requires_statistics and not statistics_available:
            return False
        if self.all_capabilities and not set(self.all_capabilities).issubset(capabilities):
            return False
        if self.any_capabilities and not set(self.any_capabilities).intersection(capabilities):
            return False
        return True


@dataclass(frozen=True)
class PanelControlSpec:
    panel_id: str
    analysis_options_key: str | None = None
    analysis_label: str | None = None
    preferred_analysis: str | None = None
    show_window: bool = False
    show_chaser: bool = False
    show_statistic: bool = False
    show_egocentric_bins: bool = False
    show_position_bins: bool = False


GROUP_PANEL_DEFINITIONS = (
    GroupPanelDefinition(
        "behavior",
        "Core behavior",
        "Recording-weighted speed, path, bout, heading, and arena-distance summaries by epoch.",
        all_capabilities=("chaser.epoch.behavior_summary",),
    ),
    GroupPanelDefinition(
        "bout_distributions",
        "Bout distributions",
        "Persisted count-first bout and inter-bout-interval distributions.",
        any_capabilities=(
            "chaser.epoch.bout_histogram",
            "chaser.epoch.inter_bout_interval_histogram",
        ),
    ),
    GroupPanelDefinition(
        "spatial",
        "Spatial occupancy",
        "Position-density heatmaps and recording-weighted occupancy summaries across spatial zones.",
        any_capabilities=(
            "position.epoch.occupancy_histogram_2d",
            "chaser.epoch.spatial_occupancy",
        ),
    ),
    GroupPanelDefinition(
        "chaser_distance",
        "Chaser distance",
        "Distance summaries, persisted distance histograms, and speed-distance relationships.",
        all_capabilities=("chaser.distance.summary",),
    ),
    GroupPanelDefinition(
        "cra",
        "CRA primary endpoints",
        "Pre/post aggressive and inert object-distance and quadrant-occupancy endpoints.",
        all_capabilities=("chaser.cra.primary",),
    ),
    GroupPanelDefinition(
        "near_field",
        "CRA near field",
        "Close-approach, near-zone, radial-density, and distance-CDF results.",
        all_capabilities=("chaser.cra.near_field",),
    ),
    GroupPanelDefinition(
        "egocentric",
        "Egocentric bearing",
        "Fish-centered bearing summaries and pooled distance-by-bearing histograms.",
        all_capabilities=("chaser.egocentric",),
    ),
    GroupPanelDefinition(
        "statistics",
        "Linked statistics",
        "Persisted statistical results linked to the selected immutable export.",
        all_capabilities=("group.statistics",),
        requires_statistics=True,
    ),
    GroupPanelDefinition(
        "inventory",
        "Recordings and provenance",
        "Recording-level drilldown, table inventory, health diagnostics, and physical schemas.",
    ),
)


SAMPLE_GRAIN_SURFACES = (
    {
        "surface": "Frame kinematic traces",
        "capability_id": "kinematics.frame_trace",
        "description": "Framewise speed, heading, position, velocity, and distance traces.",
    },
    {
        "surface": "Trajectory reconstruction",
        "capability_id": "kinematics.frame_trace",
        "description": "Framewise calibrated positions sufficient to reconstruct trajectories.",
    },
    {
        "surface": "Eye angles and convergence",
        "capability_id": "eye.frame_trace",
        "description": "Framewise left/right eye angles and convergence or vergence traces.",
    },
    {
        "surface": "Tail motion",
        "capability_id": "tail.sampled_spline",
        "description": "Bout-aligned tail angles or sampled spline positions over time.",
    },
)


PANEL_CONTROL_SPECS = {
    "behavior": PanelControlSpec(
        "behavior",
        analysis_options_key="epoch_speed_metrics",
        analysis_label="Behavior analysis",
        preferred_analysis="bout_rate_per_min",
        show_window=True,
        show_statistic=True,
    ),
    "bout_distributions": PanelControlSpec(
        "bout_distributions",
        analysis_options_key="epoch_bout_histogram_metrics",
        analysis_label="Bout analysis",
        preferred_analysis="bout_path_length_mm",
        show_window=True,
    ),
    "spatial": PanelControlSpec(
        "spatial",
        analysis_options_key="spatial_metrics",
        analysis_label="Spatial analysis",
        preferred_analysis="fraction_of_epoch",
        show_window=True,
        show_position_bins=True,
    ),
    "chaser_distance": PanelControlSpec(
        "chaser_distance",
        analysis_options_key="chaser_metrics",
        analysis_label="Chaser-distance analysis",
        preferred_analysis="p50_distance_mm",
        show_window=True,
        show_chaser=True,
        show_statistic=True,
    ),
    "cra": PanelControlSpec(
        "cra",
        analysis_options_key="cra_object_phase_metrics",
        analysis_label="CRA analysis",
        preferred_analysis="median_distance_mm",
        show_statistic=True,
    ),
    "near_field": PanelControlSpec(
        "near_field",
        analysis_options_key="cra_near_field_object_phase_metrics",
        analysis_label="Near-field analysis",
        preferred_analysis="near_zone_occupancy_fraction",
        show_statistic=True,
    ),
    "egocentric": PanelControlSpec(
        "egocentric",
        analysis_options_key="egocentric_metrics",
        analysis_label="Egocentric analysis",
        preferred_analysis="mean_alignment_cos",
        show_window=True,
        show_chaser=True,
        show_statistic=True,
        show_egocentric_bins=True,
    ),
    "statistics": PanelControlSpec("statistics"),
    "inventory": PanelControlSpec("inventory"),
}


def available_group_panels(
    capabilities: Iterable[str],
    *,
    statistics_available: bool,
) -> tuple[GroupPanelDefinition, ...]:
    available = set(str(item) for item in capabilities)
    return tuple(
        definition
        for definition in GROUP_PANEL_DEFINITIONS
        if definition.available(available, statistics_available=statistics_available)
    )


def panel_control_spec(panel_id: str) -> PanelControlSpec:
    try:
        return PANEL_CONTROL_SPECS[str(panel_id)]
    except KeyError as exc:
        raise ValueError(f"Unknown group analytics panel: {panel_id}") from exc


def capability_inventory_rows(
    available_capabilities: Iterable[str],
    capability_statuses: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    available = set(str(item) for item in available_capabilities)
    by_id = {
        str(row.get("capability_id")): row
        for row in capability_statuses
        if row.get("capability_id")
    }
    capability_ids = sorted(available | set(by_id))
    return [
        {
            "capability_id": capability_id,
            "available": capability_id in available,
            "missing_tables": ", ".join(
                str(item) for item in by_id.get(capability_id, {}).get("missing_tables", [])
            ),
            "missing_columns": "; ".join(
                f"{table}: {', '.join(str(column) for column in columns)}"
                for table, columns in (
                    by_id.get(capability_id, {}).get("missing_columns_by_table", {}) or {}
                ).items()
            ),
        }
        for capability_id in capability_ids
    ]


def sample_grain_status_rows(
    available_capabilities: Iterable[str],
) -> list[dict[str, str | bool]]:
    """Describe dense surfaces without claiming anything about source Zarrs."""

    available = set(str(item) for item in available_capabilities)
    rows: list[dict[str, str | bool]] = []
    for definition in SAMPLE_GRAIN_SURFACES:
        capability_id = str(definition["capability_id"])
        included = capability_id in available
        rows.append(
            {
                **definition,
                "included_in_export": included,
                "status": (
                    "available"
                    if included
                    else "not included in this exported dataset"
                ),
            }
        )
    return rows


def filter_rows_by_chasers(
    rows: Sequence[Mapping[str, Any]],
    selected_chasers: Iterable[int],
) -> list[dict[str, Any]]:
    """Keep rows for the explicitly selected chasers, preserving row order."""

    selected = {int(value) for value in selected_chasers}
    return [
        dict(row)
        for row in rows
        if row.get("chaser_index") is not None
        and int(row["chaser_index"]) in selected
    ]


def chaser_selection_options(
    chaser_indices: Iterable[int],
) -> tuple[dict[str, int], list[str]]:
    """Return explicit chaser labels with every available chaser selected by default."""

    options = {
        f"Chaser {int(chaser_index)}": int(chaser_index)
        for chaser_index in chaser_indices
    }
    return options, list(options)


def epoch_selection_options(
    window_labels: Iterable[str],
) -> tuple[list[str], list[str]]:
    """Return unique epoch labels with every available epoch selected by default."""

    options = list(dict.fromkeys(str(label) for label in window_labels))
    return options, list(options)


def filter_rows_by_windows(
    rows: Sequence[Mapping[str, Any]],
    selected_windows: Iterable[str],
) -> list[dict[str, Any]]:
    """Keep rows for the explicitly selected epochs, preserving row order."""

    selected = {str(value) for value in selected_windows}
    return [
        dict(row)
        for row in rows
        if row.get("window_label") is not None
        and str(row["window_label"]) in selected
    ]


def group_egocentric_histogram_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    window_labels: Iterable[str],
    chaser_indices: Iterable[int],
) -> dict[tuple[str, int], list[dict[str, Any]]]:
    """Group exported polar bins in explicit epoch-by-chaser display order."""

    windows = tuple(str(value) for value in window_labels)
    chasers = tuple(int(value) for value in chaser_indices)
    grouped = {(window, chaser): [] for window in windows for chaser in chasers}
    for row in rows:
        window_label = row.get("window_label")
        chaser_index = row.get("chaser_index")
        if window_label is None or chaser_index is None:
            continue
        key = (str(window_label), int(chaser_index))
        if key in grouped:
            grouped[key].append(dict(row))
    return grouped


def egocentric_probability_color_max(
    rows: Sequence[Mapping[str, Any]],
    *,
    quantile: float = 0.98,
) -> float:
    """Return one robust positive color maximum shared by polar panels."""

    frame = pd.DataFrame(rows)
    if frame.empty or "pooled_probability" not in frame.columns:
        return 1.0
    values = pd.to_numeric(frame["pooled_probability"], errors="coerce")
    positive = values[values.notna() & (values > 0.0)]
    if positive.empty:
        return 1.0
    return max(float(positive.quantile(float(quantile))), float.fromhex("0x1p-52"))


def grouped_bar_figure(
    rows: Sequence[Mapping[str, Any]],
    *,
    title: str,
    x_key: str,
    y_key: str,
    series_key: str,
    yaxis_title: str,
    color_key: str | None = None,
) -> go.Figure | None:
    frame = pd.DataFrame(rows)
    if frame.empty or x_key not in frame or y_key not in frame:
        return None
    fig = go.Figure()
    if series_key not in frame:
        frame[series_key] = "all"
    grouped = list(frame.groupby(series_key, sort=False, dropna=False))
    colors = []
    for _series, group in grouped:
        values = (
            sorted(
                {
                    str(value).strip().lower()
                    for value in group[color_key].dropna().tolist()
                    if str(value).strip()
                }
            )
            if color_key and color_key in group
            else []
        )
        colors.append(values[0] if len(values) == 1 else None)
    color_counts = Counter(color for color in colors if color)
    pattern_shapes = ("", "/", "\\", "x", ".", "+")
    for series_index, ((series, group), series_color) in enumerate(zip(grouped, colors)):
        custom_columns = [
            column
            for column in ("recording_count", "mean", "median", "sem", "n")
            if column in group
        ]
        marker: dict[str, Any] | None = None
        if series_color:
            marker = {"color": series_color}
            if color_counts[series_color] > 1:
                marker["pattern"] = {
                    "shape": pattern_shapes[series_index % len(pattern_shapes)]
                }
        fig.add_trace(
            go.Bar(
                x=group[x_key],
                y=group[y_key],
                name=str(series),
                customdata=group[custom_columns] if custom_columns else None,
                marker=marker,
            )
        )
    fig.update_layout(
        title=title,
        xaxis_title=x_key.replace("_", " ").title(),
        yaxis_title=yaxis_title,
        barmode="group",
        margin=dict(l=55, r=25, t=55, b=60),
        legend=dict(orientation="h", yanchor="top", y=-0.2),
    )
    return fig


def line_figure(
    rows: Sequence[Mapping[str, Any]],
    *,
    title: str,
    x_key: str,
    y_key: str,
    series_keys: Sequence[str],
    xaxis_title: str,
    yaxis_title: str,
    color_key: str | None = None,
) -> go.Figure | None:
    frame = pd.DataFrame(rows)
    if frame.empty or x_key not in frame or y_key not in frame:
        return None
    fig = go.Figure()
    keys = [key for key in series_keys if key in frame]
    if keys:
        grouped = list(
            frame.groupby(
                keys[0] if len(keys) == 1 else keys,
                sort=False,
                dropna=False,
            )
        )
    else:
        grouped = [("all", frame)]
    colors = []
    for _raw_series, group in grouped:
        values = (
            sorted(
                {
                    str(value).strip().lower()
                    for value in group[color_key].dropna().tolist()
                    if str(value).strip()
                }
            )
            if color_key and color_key in group
            else []
        )
        colors.append(values[0] if len(values) == 1 else None)
    color_counts = Counter(color for color in colors if color)
    dash_styles = ("solid", "dash", "dot", "dashdot")
    for series_index, ((raw_series, group), series_color) in enumerate(zip(grouped, colors)):
        values = raw_series if isinstance(raw_series, tuple) else (raw_series,)
        label = " · ".join(str(value) for value in values)
        group = group.sort_values(x_key)
        line: dict[str, Any] | None = None
        if series_color:
            line = {"color": series_color}
            if color_counts[series_color] > 1:
                line["dash"] = dash_styles[series_index % len(dash_styles)]
        fig.add_trace(
            go.Scatter(
                x=group[x_key],
                y=group[y_key],
                mode="lines",
                name=label,
                line=line,
            )
        )
    fig.update_layout(
        title=title,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        margin=dict(l=55, r=25, t=55, b=60),
        legend=dict(orientation="h", yanchor="top", y=-0.2),
    )
    return fig


def egocentric_heatmap_figure(
    rows: Sequence[Mapping[str, Any]],
    *,
    title: str,
) -> go.Figure | None:
    frame = pd.DataFrame(rows)
    required = {
        "distance_bin_center_mm",
        "bearing_bin_center_deg",
        "pooled_probability",
    }
    if frame.empty or not required.issubset(frame.columns):
        return None
    pivot = frame.pivot_table(
        index="bearing_bin_center_deg",
        columns="distance_bin_center_mm",
        values="pooled_probability",
        aggfunc="sum",
        fill_value=0.0,
    ).sort_index()
    fig = go.Figure(
        data=go.Heatmap(
            x=list(pivot.columns),
            y=list(pivot.index),
            z=pivot.to_numpy(),
            colorscale="Viridis",
            colorbar=dict(title="Probability"),
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Distance to chaser (mm)",
        yaxis_title="Egocentric bearing (deg)",
        margin=dict(l=60, r=30, t=55, b=55),
    )
    return fig


def position_occupancy_heatmap_figure(
    rows: Sequence[Mapping[str, Any]],
    *,
    title: str,
    color_max: float | None = None,
    show_colorbar: bool = True,
) -> go.Figure | None:
    """Render pooled normalized position-occupancy bins in source-image orientation."""

    frame = pd.DataFrame(rows)
    required = {
        "x_bin_center_fraction",
        "y_bin_center_fraction",
        "pooled_probability",
    }
    if frame.empty or not required.issubset(frame.columns):
        return None
    for column in required:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=sorted(required))
    if frame.empty:
        return None
    pivot = frame.pivot_table(
        index="y_bin_center_fraction",
        columns="x_bin_center_fraction",
        values="pooled_probability",
        aggfunc="sum",
        fill_value=0.0,
    ).sort_index()
    effective_color_max = (
        float(color_max)
        if color_max is not None and float(color_max) > 0.0
        else egocentric_probability_color_max(frame.to_dict("records"))
    )
    fig = go.Figure(
        data=go.Heatmap(
            x=list(pivot.columns),
            y=list(pivot.index),
            z=pivot.to_numpy(),
            colorscale="Inferno",
            zmin=0.0,
            zmax=effective_color_max,
            showscale=bool(show_colorbar),
            colorbar=dict(title="Probability"),
            hovertemplate=(
                "x=%{x:.3f}<br>y=%{y:.3f}<br>probability=%{z:.4f}<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        title=title,
        height=480,
        margin=dict(l=55, r=70 if show_colorbar else 30, t=55, b=55),
        xaxis_title="Normalized arena X (right)",
        yaxis_title="Normalized arena Y (down)",
    )
    fig.update_yaxes(autorange="reversed", scaleanchor="x", scaleratio=1)
    return fig


def cra_quadrant_occupancy_figure(
    data: Mapping[str, Any],
    *,
    title: str = "Descriptive gross quadrant relocation",
) -> go.Figure | None:
    """Recreate the CRA chaser-vs-other-quadrants strip and density display."""

    rows = [dict(row) for row in data.get("rows", [])]
    phases = sorted(
        [dict(row) for row in data.get("phases", [])],
        key=lambda row: int(row.get("phase_axis_index") or 0),
    )
    if not rows or not phases:
        return None
    quadrant_rows = [dict(row) for row in data.get("quadrant_rows", [])]
    density_rows = [dict(row) for row in data.get("density_rows", [])]
    subplot_titles = []
    row_heights = []
    for phase in phases:
        subplot_titles.extend([str(phase.get("phase_label") or "phase"), ""])
        row_heights.extend([0.68, 0.32])
    fig = make_subplots(
        rows=2 * len(phases),
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.045,
        row_heights=row_heights,
        subplot_titles=subplot_titles,
    )

    for phase_index, phase in enumerate(phases):
        phase_label = str(phase.get("phase_label") or "")
        strip_row = phase_index * 2 + 1
        density_row = strip_row + 1
        phase_rows = [row for row in rows if str(row.get("phase_label") or "") == phase_label]
        phase_quadrants = sorted(
            [
                row
                for row in quadrant_rows
                if str(row.get("phase_label") or "") == phase_label
            ],
            key=lambda row: (
                int(row.get("display_order") or 0),
                str(row.get("quadrant_id") or ""),
            ),
        )
        display_order_by_id = {
            str(row.get("quadrant_id") or ""): int(row.get("display_order") or 0)
            for row in phase_quadrants
        }
        tickvals = [int(row.get("display_order") or 0) for row in phase_quadrants]
        ticktext = [
            str(row.get("quadrant_label") or row.get("quadrant_id") or "quadrant")
            for row in phase_quadrants
        ]

        for is_chaser, name, color, opacity, size in (
            (False, "non-chaser quadrants", "#334155", 0.42, 7),
            (True, "chaser quadrant", "#b54848", 0.88, 9),
        ):
            selected = [
                row for row in phase_rows if bool(row.get("is_chaser_quadrant")) is is_chaser
            ]
            x_values = []
            y_values = []
            customdata = []
            for row in selected:
                value = pd.to_numeric(row.get("occupancy_fraction"), errors="coerce")
                quadrant_id = str(row.get("quadrant_id") or "")
                if pd.isna(value) or quadrant_id not in display_order_by_id:
                    continue
                recording_id = str(row.get("recording_id") or "")
                jitter_key = f"{recording_id}|{phase_label}|{quadrant_id}".encode()
                jitter = ((zlib.crc32(jitter_key) % 1000) / 999.0 - 0.5) * 0.36
                x_values.append(float(value))
                y_values.append(float(display_order_by_id[quadrant_id]) + jitter)
                customdata.append([recording_id, quadrant_id])
            fig.add_trace(
                go.Scatter(
                    x=x_values,
                    y=y_values,
                    mode="markers",
                    name=name,
                    legendgroup=name,
                    showlegend=phase_index == 0,
                    marker=dict(color=color, opacity=opacity, size=size),
                    customdata=customdata,
                    hovertemplate=(
                        "recording=%{customdata[0]}<br>"
                        "quadrant=%{customdata[1]}<br>"
                        "occupancy=%{x:.4f}<extra>%{fullData.name}</extra>"
                    ),
                ),
                row=strip_row,
                col=1,
            )

        means = [
            float(row["mean"])
            for row in phase_quadrants
            if row.get("mean") is not None
        ]
        mean_rows = [row for row in phase_quadrants if row.get("mean") is not None]
        fig.add_trace(
            go.Scatter(
                x=means,
                y=[float(row.get("display_order") or 0) - 0.28 for row in mean_rows],
                mode="markers",
                name="mean ± SEM",
                legendgroup="mean",
                showlegend=phase_index == 0,
                marker=dict(
                    symbol="diamond",
                    size=9,
                    color=[
                        "#b54848"
                        if int(row.get("chaser_recording_count") or 0) > 0
                        else "#182026"
                        for row in mean_rows
                    ],
                    line=dict(color="white", width=1),
                ),
                error_x=dict(
                    type="data",
                    array=[float(row.get("sem") or 0.0) for row in mean_rows],
                    visible=True,
                    thickness=1.6,
                ),
                customdata=[
                    [row.get("quadrant_label"), row.get("recording_count")]
                    for row in mean_rows
                ],
                hovertemplate=(
                    "quadrant=%{customdata[0]}<br>mean=%{x:.4f}<br>"
                    "recordings=%{customdata[1]}<extra></extra>"
                ),
            ),
            row=strip_row,
            col=1,
        )
        fig.update_yaxes(
            tickmode="array",
            tickvals=tickvals,
            ticktext=ticktext,
            range=[max(tickvals, default=3) + 0.55, -0.55],
            row=strip_row,
            col=1,
        )

        for series_role, name, color, opacity in (
            ("non_chaser", "non-chaser density", "#334155", 0.16),
            ("chaser", "chaser density", "#b54848", 0.24),
        ):
            selected_density = sorted(
                [
                    row
                    for row in density_rows
                    if str(row.get("phase_label") or "") == phase_label
                    and str(row.get("series_role") or "") == series_role
                ],
                key=lambda row: float(row.get("x") or 0.0),
            )
            fig.add_trace(
                go.Scatter(
                    x=[row.get("x") for row in selected_density],
                    y=[row.get("density") for row in selected_density],
                    mode="lines",
                    fill="tozeroy",
                    fillcolor=color,
                    opacity=opacity,
                    line=dict(color=color, width=2.3 if series_role == "chaser" else 1.8),
                    name=name,
                    legendgroup=name,
                    showlegend=phase_index == 0,
                    hovertemplate="occupancy=%{x:.3f}<br>density=%{y:.3f}<extra>%{fullData.name}</extra>",
                ),
                row=density_row,
                col=1,
            )
        fig.update_yaxes(title_text="KDE", rangemode="tozero", row=density_row, col=1)
        fig.add_vline(x=float(data.get("chance") or 0.25), line_dash="dash", line_color="#637080", row=strip_row, col=1)
        fig.add_vline(x=float(data.get("chance") or 0.25), line_dash="dash", line_color="#637080", row=density_row, col=1)

    statistics = data.get("statistics") or {}
    subtitle = (
        f"chaser-only paired pre/post: n={statistics.get('n', 0)}, "
        f"p={statistics.get('p_value')}, median Δ={statistics.get('median_difference')}"
    )
    fig.update_xaxes(range=[0.0, 1.0])
    fig.update_xaxes(
        title_text="Time spent in quadrant (normalized occupancy)",
        row=2 * len(phases),
        col=1,
    )
    fig.update_layout(
        title=f"{title}<br><sup>{subtitle}</sup>",
        height=max(640, 430 * len(phases)),
        margin=dict(l=115, r=35, t=90, b=65),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0.0),
        hovermode="closest",
    )
    return fig


def egocentric_polar_figure(
    rows: Sequence[Mapping[str, Any]],
    *,
    title: str,
    color_max: float | None = None,
    show_colorbar: bool = True,
) -> go.Figure | None:
    """Render pooled exported distance-by-bearing bins on a polar axis."""

    frame = pd.DataFrame(rows)
    required = {
        "distance_bin_left_mm",
        "distance_bin_width_mm",
        "bearing_bin_center_deg",
        "bearing_bin_width_deg",
        "pooled_probability",
    }
    if frame.empty or not required.issubset(frame.columns):
        return None
    numeric_columns = sorted(required)
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=numeric_columns)
    frame = frame[
        (frame["distance_bin_width_mm"] > 0.0)
        & (frame["bearing_bin_width_deg"] > 0.0)
        & (frame["pooled_probability"] >= 0.0)
    ]
    if frame.empty:
        return None
    effective_color_max = (
        float(color_max)
        if color_max is not None and float(color_max) > 0.0
        else egocentric_probability_color_max(frame.to_dict("records"))
    )
    pooled_counts = (
        pd.to_numeric(frame["pooled_count"], errors="coerce").fillna(0).to_numpy()
        if "pooled_count" in frame.columns
        else [0] * len(frame)
    )
    fig = go.Figure(
        data=go.Barpolar(
            theta=frame["bearing_bin_center_deg"].to_numpy(),
            r=frame["distance_bin_width_mm"].to_numpy(),
            base=frame["distance_bin_left_mm"].to_numpy(),
            width=frame["bearing_bin_width_deg"].to_numpy(),
            marker=dict(
                color=frame["pooled_probability"].to_numpy(),
                colorscale="Viridis",
                cmin=0.0,
                cmax=effective_color_max,
                showscale=bool(show_colorbar),
                colorbar=dict(title="Probability"),
                line=dict(width=0),
            ),
            customdata=list(
                zip(
                    pooled_counts,
                    frame["pooled_probability"].to_numpy(),
                    frame["distance_bin_left_mm"].to_numpy(),
                    (
                        frame["distance_bin_left_mm"]
                        + frame["distance_bin_width_mm"]
                    ).to_numpy(),
                    (
                        frame["bearing_bin_center_deg"]
                        - frame["bearing_bin_width_deg"] / 2.0
                    ).to_numpy(),
                    (
                        frame["bearing_bin_center_deg"]
                        + frame["bearing_bin_width_deg"] / 2.0
                    ).to_numpy(),
                )
            ),
            hovertemplate=(
                "bearing=%{customdata[4]:.1f} to %{customdata[5]:.1f} deg<br>"
                "distance=%{customdata[2]:.2f} to %{customdata[3]:.2f} mm<br>"
                "samples=%{customdata[0]:,}<br>"
                "probability=%{customdata[1]:.4f}<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        title=title,
        height=500,
        margin=dict(l=35, r=70 if show_colorbar else 35, t=60, b=40),
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
    return fig


__all__ = [
    "GROUP_PANEL_DEFINITIONS",
    "PANEL_CONTROL_SPECS",
    "SAMPLE_GRAIN_SURFACES",
    "GroupPanelDefinition",
    "PanelControlSpec",
    "available_group_panels",
    "capability_inventory_rows",
    "chaser_selection_options",
    "cra_quadrant_occupancy_figure",
    "epoch_selection_options",
    "egocentric_heatmap_figure",
    "egocentric_polar_figure",
    "egocentric_probability_color_max",
    "filter_rows_by_chasers",
    "filter_rows_by_windows",
    "group_egocentric_histogram_rows",
    "grouped_bar_figure",
    "line_figure",
    "panel_control_spec",
    "position_occupancy_heatmap_figure",
    "sample_grain_status_rows",
]
