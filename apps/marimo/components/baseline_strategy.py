"""Pure filtering and Plotly helpers for baseline-strategy QC."""

from __future__ import annotations

from collections import Counter, defaultdict
import math
from typing import Any, Iterable, Mapping, Sequence

import plotly.graph_objects as go


STRATEGY_FEATURE_METRICS = {
    "wall_fraction": "Wall fraction",
    "active_wall_fraction": "Active wall fraction",
    "occupancy_coverage_fraction": "Accessible-area coverage",
    "occupancy_entropy_accessible_normalized": "Occupancy entropy",
    "active_sample_fraction": "Active sample fraction",
    "path_per_min_mm": "Path per minute (mm)",
    "tracking_dropout_fraction": "Tracking dropout fraction",
}

STRATEGY_CATEGORY_FIELDS = {
    "primary_strategy": "Primary strategy",
    "activity_state": "Activity state",
    "boundary_strategy": "Boundary strategy",
    "spatial_organization": "Spatial organization",
    "temporal_pattern": "Temporal pattern",
    "cluster_id": "Unsupervised cluster",
}


def _finite(value: object) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def filter_qc_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    protocols: Iterable[str],
    statuses: Iterable[str],
) -> list[dict[str, Any]]:
    selected_protocols = {str(value) for value in protocols}
    selected_statuses = {str(value) for value in statuses}
    return [
        dict(row)
        for row in rows
        if str(row.get("protocol_name") or "unknown") in selected_protocols
        and str(row.get("classification_status") or "unknown") in selected_statuses
    ]


def category_count_figure(
    rows: Sequence[Mapping[str, Any]],
    *,
    category_key: str,
    title: str,
) -> go.Figure | None:
    if not rows or not any(category_key in row for row in rows):
        return None
    counts = Counter(
        (
            str(row.get("protocol_name") or "unknown"),
            str(row.get(category_key) if row.get(category_key) is not None else "unavailable"),
        )
        for row in rows
    )
    categories = sorted({category for _, category in counts})
    protocols = sorted({protocol for protocol, _ in counts})
    figure = go.Figure()
    for protocol in protocols:
        figure.add_bar(
            name=protocol,
            x=categories,
            y=[counts[(protocol, category)] for category in categories],
            hovertemplate="%{x}<br>%{y} recording(s)<extra>%{fullData.name}</extra>",
        )
    figure.update_layout(
        title=title,
        barmode="group",
        xaxis_title="Category",
        yaxis_title="Recordings",
        legend_title="Protocol",
        margin=dict(l=55, r=20, t=55, b=100),
    )
    return figure


def feature_distribution_figure(
    rows: Sequence[Mapping[str, Any]],
    *,
    metric: str,
    label: str,
) -> go.Figure | None:
    grouped: dict[str, list[tuple[float, str]]] = defaultdict(list)
    for row in rows:
        value = _finite(row.get(metric))
        if value is None:
            continue
        grouped[str(row.get("protocol_name") or "unknown")].append(
            (value, str(row.get("recording_id") or ""))
        )
    if not grouped:
        return None
    figure = go.Figure()
    for protocol in sorted(grouped):
        values = grouped[protocol]
        figure.add_box(
            name=protocol,
            y=[value for value, _ in values],
            text=[recording for _, recording in values],
            boxpoints="all",
            jitter=0.25,
            pointpos=0,
            hovertemplate="%{text}<br>%{y:.4g}<extra>%{fullData.name}</extra>",
        )
    figure.update_layout(
        title=f"{label} by protocol",
        yaxis_title=label,
        xaxis_title="Protocol",
        margin=dict(l=65, r=20, t=55, b=55),
    )
    return figure


def feature_scatter_figure(
    rows: Sequence[Mapping[str, Any]],
    *,
    x_key: str = "active_wall_fraction",
    y_key: str = "occupancy_coverage_fraction",
) -> go.Figure | None:
    grouped: dict[str, list[tuple[float, float, str, str]]] = defaultdict(list)
    for row in rows:
        x = _finite(row.get(x_key))
        y = _finite(row.get(y_key))
        if x is None or y is None:
            continue
        grouped[str(row.get("primary_strategy") or "unavailable")].append(
            (
                x,
                y,
                str(row.get("recording_id") or ""),
                str(row.get("protocol_name") or "unknown"),
            )
        )
    if not grouped:
        return None
    figure = go.Figure()
    for strategy in sorted(grouped):
        values = grouped[strategy]
        figure.add_trace(
            go.Scattergl(
                name=strategy,
                mode="markers",
                x=[value[0] for value in values],
                y=[value[1] for value in values],
                text=[f"{value[2]}<br>{value[3]}" for value in values],
                marker={"size": 9, "opacity": 0.8},
                hovertemplate=(
                    "%{text}<br>active wall=%{x:.3f}<br>coverage=%{y:.3f}"
                    "<extra>%{fullData.name}</extra>"
                ),
            )
        )
    figure.update_layout(
        title="Boundary affinity versus spatial coverage",
        xaxis_title="Active wall fraction",
        yaxis_title="Accessible-area coverage",
        legend_title="Descriptive strategy",
        margin=dict(l=65, r=20, t=55, b=55),
    )
    return figure


def trajectory_figure(
    rows: Sequence[Mapping[str, Any]], *, recording_id: str
) -> go.Figure | None:
    points = [
        (_finite(row.get("x_arena_mm")), _finite(row.get("y_arena_mm")))
        for row in rows
        if bool(row.get("position_valid", True)) and bool(row.get("sample_valid", True))
    ]
    points = [(x, y) for x, y in points if x is not None and y is not None]
    if not points:
        return None
    figure = go.Figure(
        go.Scattergl(
            x=[point[0] for point in points],
            y=[point[1] for point in points],
            mode="lines",
            line={"width": 1.2, "color": "#355c9a"},
            hoverinfo="skip",
            name="trajectory",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=[points[0][0], points[-1][0]],
            y=[points[0][1], points[-1][1]],
            mode="markers",
            marker={"size": 8, "color": ["#2ca02c", "#d62728"]},
            text=["start", "end"],
            hovertemplate="%{text}<extra></extra>",
            name="endpoints",
        )
    )
    figure.update_layout(
        title=f"Baseline trajectory · {recording_id}",
        xaxis={"title": "Arena x (mm)", "scaleanchor": "y", "scaleratio": 1},
        yaxis={"title": "Arena y (mm)"},
        showlegend=False,
        margin=dict(l=60, r=20, t=55, b=55),
    )
    return figure


def speed_trace_figure(
    rows: Sequence[Mapping[str, Any]], *, recording_id: str
) -> go.Figure | None:
    points = [
        (_finite(row.get("relative_time_s")), _finite(row.get("speed_mm_s")))
        for row in rows
        if bool(row.get("sample_valid", True))
    ]
    points = [(time, speed) for time, speed in points if time is not None and speed is not None]
    if not points:
        return None
    figure = go.Figure(
        go.Scattergl(
            x=[point[0] for point in points],
            y=[point[1] for point in points],
            mode="lines",
            line={"width": 1, "color": "#6f4aa8"},
            hovertemplate="%{x:.2f} s<br>%{y:.3g} mm/s<extra></extra>",
        )
    )
    figure.update_layout(
        title=f"Portable 10 Hz speed trace · {recording_id}",
        xaxis_title="Baseline time (s)",
        yaxis_title="Speed (mm/s)",
        margin=dict(l=65, r=20, t=55, b=55),
    )
    return figure


__all__ = [
    "STRATEGY_CATEGORY_FIELDS",
    "STRATEGY_FEATURE_METRICS",
    "category_count_figure",
    "feature_distribution_figure",
    "feature_scatter_figure",
    "filter_qc_rows",
    "speed_trace_figure",
    "trajectory_figure",
]
