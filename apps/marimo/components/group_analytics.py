"""Pure panel routing and Plotly helpers for the group analytics Marimo app."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd
import plotly.graph_objects as go


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
        "Recording-weighted or count-first occupancy summaries across persisted spatial zones.",
        all_capabilities=("chaser.epoch.spatial_occupancy",),
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


def grouped_bar_figure(
    rows: Sequence[Mapping[str, Any]],
    *,
    title: str,
    x_key: str,
    y_key: str,
    series_key: str,
    yaxis_title: str,
) -> go.Figure | None:
    frame = pd.DataFrame(rows)
    if frame.empty or x_key not in frame or y_key not in frame:
        return None
    fig = go.Figure()
    if series_key not in frame:
        frame[series_key] = "all"
    for series, group in frame.groupby(series_key, sort=False, dropna=False):
        custom_columns = [
            column
            for column in ("recording_count", "mean", "median", "sem", "n")
            if column in group
        ]
        fig.add_trace(
            go.Bar(
                x=group[x_key],
                y=group[y_key],
                name=str(series),
                customdata=group[custom_columns] if custom_columns else None,
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
) -> go.Figure | None:
    frame = pd.DataFrame(rows)
    if frame.empty or x_key not in frame or y_key not in frame:
        return None
    fig = go.Figure()
    keys = [key for key in series_keys if key in frame]
    if keys:
        grouped = frame.groupby(keys[0] if len(keys) == 1 else keys, sort=False, dropna=False)
    else:
        grouped = [("all", frame)]
    for raw_series, group in grouped:
        values = raw_series if isinstance(raw_series, tuple) else (raw_series,)
        label = " · ".join(str(value) for value in values)
        group = group.sort_values(x_key)
        fig.add_trace(
            go.Scatter(
                x=group[x_key],
                y=group[y_key],
                mode="lines",
                name=label,
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


__all__ = [
    "GROUP_PANEL_DEFINITIONS",
    "SAMPLE_GRAIN_SURFACES",
    "GroupPanelDefinition",
    "available_group_panels",
    "capability_inventory_rows",
    "egocentric_heatmap_figure",
    "grouped_bar_figure",
    "line_figure",
    "sample_grain_status_rows",
]
