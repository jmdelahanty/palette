"""Pure Plotly helpers for exact validated-behavior distributions."""

from __future__ import annotations

from collections import defaultdict
import json
import math
from typing import Any, Mapping

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from fisheye.group_statistics.validated_behavior_distribution_views import (
    COHORT_STATISTIC_LABELS,
    DEFAULT_COHORT_STATISTIC,
    DEFAULT_DISPLAY_RANGE,
    FULL_EVIDENCE_RANGE,
    resolve_distribution_display_range,
    validate_distribution_view_payload,
    validate_motion_trace_payload,
)


def _rows(payload: Mapping[str, object], key: str) -> list[Mapping[str, Any]]:
    raw = payload.get(key)
    if not isinstance(raw, list):
        raise ValueError(f"Distribution payload lacks {key}")
    return [row for row in raw if isinstance(row, Mapping)]


def distribution_metric_options(
    metric_specs: tuple[Mapping[str, object], ...],
) -> dict[str, str]:
    """Return unique human-readable metric label -> exact metric ID options."""

    result: dict[str, str] = {}
    for metric in metric_specs:
        metric_id = str(metric["metric_id"])
        label = str(metric.get("interpretation") or metric_id)
        if label in result:
            label = f"{label} [{metric_id}]"
        result[label] = metric_id
    return result


def distribution_dimension_options(
    payload: Mapping[str, object], dimension: str
) -> tuple[str, ...]:
    validate_distribution_view_payload(payload)
    values = {
        str(row[dimension])
        for row in _rows(payload, "cohort_rows")
        if row.get(dimension) is not None
    }
    return tuple(sorted(values))


def distribution_provenance_rows(
    payload: Mapping[str, object],
) -> list[dict[str, object]]:
    source = payload.get("source_distribution")
    source_record = source if isinstance(source, Mapping) else {}
    recipe = payload.get("histogram_recipe")
    recipe_record = recipe if isinstance(recipe, Mapping) else {}
    metric = payload.get("metric")
    metric_record = metric if isinstance(metric, Mapping) else {}
    return [
        {"field": "view_payload_sha256", "value": payload.get("payload_sha256")},
        {
            "field": "distribution_run_id",
            "value": source_record.get("distribution_run_id"),
        },
        {
            "field": "distribution_manifest_sha256",
            "value": source_record.get("distribution_manifest_sha256"),
        },
        {
            "field": "source_export_run_id",
            "value": source_record.get("source_export_run_id"),
        },
        {
            "field": "source_export_manifest_sha256",
            "value": source_record.get("source_export_manifest_sha256"),
        },
        {
            "field": "metric_id",
            "value": metric_record.get("metric_id"),
        },
        {
            "field": "metric_spec_sha256",
            "value": recipe_record.get("metric_spec_sha256"),
        },
        {
            "field": "histogram_recipe_sha256",
            "value": recipe_record.get("histogram_recipe_sha256"),
        },
        {"field": "bin_count", "value": recipe_record.get("bin_count")},
        {"field": "bin_width", "value": recipe_record.get("bin_width")},
        {
            "field": "bin_width_domain",
            "value": recipe_record.get("bin_width_domain"),
        },
        {"field": "axis_scale", "value": recipe_record.get("axis_scale")},
        {"field": "weighting_id", "value": payload.get("weighting_id")},
        {
            "field": "experimental_unit",
            "value": source_record.get("experimental_unit"),
        },
        {
            "field": "cohort_weighting",
            "value": source_record.get("cohort_weighting"),
        },
    ]


def _series_style(
    payload: Mapping[str, object], group: Mapping[str, Any], scope_id: str
) -> tuple[str, str, str, str | None, str]:
    provider = str(group.get("provider_role", ""))
    role = str(group.get("behavior_role", ""))
    if role:
        styles = payload.get("behavior_role_styles", {})
        style = (
            styles.get(role, {})
            if isinstance(styles, Mapping) and isinstance(styles.get(role), Mapping)
            else {}
        )
        color = str(style.get("aggregate_color_css", "#555555"))
        symbol: str | None = str(style.get("plotly_role_symbol") or "circle")
    elif provider:
        colors = payload.get("provider_colors", {})
        color = (
            str(colors.get(provider, "#555555"))
            if isinstance(colors, Mapping)
            else "#555555"
        )
        symbol = None
    else:
        colors = payload.get("scope_colors", {})
        color = (
            str(colors.get(scope_id, "#4C78A8"))
            if isinstance(colors, Mapping)
            else "#4C78A8"
        )
        symbol = None
    styles = payload.get("provider_line_styles", {})
    style = (
        str(styles.get(provider, "solid"))
        if provider and isinstance(styles, Mapping)
        else "solid"
    )
    dash = {"solid": "solid", "dashed": "dash", "dotted": "dot"}.get(
        style, "solid"
    )
    pattern = {"solid": "", "dashed": "/", "dotted": "."}.get(style, "")
    label = " · ".join(
        value
        for value in (
            provider.title() if provider else "",
            role.title() if role else "",
        )
        if value
    )
    return color, dash, pattern, symbol, label or "All observations"


def validated_behavior_distribution_figure(
    payload: Mapping[str, object],
    *,
    cohort_statistic: str = DEFAULT_COHORT_STATISTIC,
    provider_role: str | None = None,
    behavior_role: str | None = None,
    show_recording_iqr: bool = True,
    display_range_id: str = DEFAULT_DISPLAY_RANGE,
) -> go.Figure:
    """Render the same four-scope histogram payload used by static figures."""

    validate_distribution_view_payload(payload)
    if cohort_statistic not in COHORT_STATISTIC_LABELS:
        raise ValueError(f"Unknown cohort statistic: {cohort_statistic}")
    metric = payload["metric"]
    recipe = payload["histogram_recipe"]
    assert isinstance(metric, Mapping) and isinstance(recipe, Mapping)
    rows = [
        row
        for row in _rows(payload, "cohort_rows")
        if (provider_role is None or row.get("provider_role") == provider_role)
        and (behavior_role is None or row.get("behavior_role") == behavior_role)
    ]
    if not rows:
        raise ValueError("No distribution series matches the selected dimensions")
    display_range = resolve_distribution_display_range(
        payload,
        display_range_id=display_range_id,
        provider_role=provider_role,
        behavior_role=behavior_role,
    )
    scopes = tuple(str(value) for value in payload["scope_order"])
    labels = payload.get("scope_labels", {})
    figure = make_subplots(
        rows=1,
        cols=len(scopes),
        shared_yaxes=True,
        subplot_titles=[
            str(labels.get(scope, scope)) if isinstance(labels, Mapping) else scope
            for scope in scopes
        ],
        horizontal_spacing=0.035,
    )
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["scope_id"]), str(row["group_key_sha256"]))].append(row)
    group_count = len({str(row["group_key_sha256"]) for row in rows})
    bar_opacity = 0.78 if group_count == 1 else 0.42
    seen_labels: set[str] = set()
    for column, scope_id in enumerate(scopes, start=1):
        for (row_scope, _digest), series in sorted(grouped.items()):
            if row_scope != scope_id:
                continue
            ordered = sorted(series, key=lambda row: int(row["bin_index"]))
            group = json.loads(str(ordered[0]["group_key_json"]))
            color, dash, pattern, symbol, label = _series_style(
                payload, group, scope_id
            )
            left = [float(row["bin_left"]) for row in ordered]
            right = [float(row["bin_right"]) for row in ordered]
            centers = [float(row["bin_center"]) for row in ordered]
            widths = [end - start for start, end in zip(left, right, strict=True)]
            values = [
                (
                    None
                    if row.get(cohort_statistic) is None
                    else 100.0 * float(row[cohort_statistic])
                )
                for row in ordered
            ]
            support = int(ordered[0]["finite_recording_count"])
            custom = [
                [float(row["bin_left"]), float(row["bin_right"]), support]
                for row in ordered
            ]
            showlegend = label not in seen_labels
            seen_labels.add(label)
            figure.add_trace(
                go.Bar(
                    x=centers,
                    y=values,
                    width=widths,
                    opacity=bar_opacity,
                    marker={
                        "color": color,
                        "line": {"color": color, "width": 1.0},
                        "pattern": {"shape": pattern, "solidity": 0.25},
                    },
                    name=label,
                    legendgroup=label,
                    showlegend=showlegend and symbol is None,
                    customdata=custom,
                    hovertemplate=(
                        "bin left=%{customdata[0]:.5g}"
                        "<br>bin right=%{customdata[1]:.5g}"
                        "<br>fraction=%{y:.4g}%"
                        "<br>finite recordings=%{customdata[2]}<extra>%{fullData.name}</extra>"
                    ),
                ),
                row=1,
                col=column,
            )
            if symbol is not None:
                finite = [
                    index
                    for index, value in enumerate(values)
                    if value is not None and math.isfinite(value)
                ]
                marker_count = min(12, len(finite))
                marker_indices = (
                    []
                    if marker_count == 0
                    else [
                        finite[index]
                        for index in np.unique(
                            np.linspace(
                                0,
                                len(finite) - 1,
                                marker_count,
                                dtype=np.int64,
                            )
                        )
                    ]
                )
                figure.add_trace(
                    go.Scatter(
                        x=[centers[index] for index in marker_indices],
                        y=[values[index] for index in marker_indices],
                        mode="markers",
                        marker={
                            "color": color,
                            "symbol": symbol,
                            "size": 6,
                            "line": {"color": "#ffffff", "width": 0.7},
                        },
                        line={"color": color, "dash": dash},
                        name=label,
                        legendgroup=label,
                        showlegend=showlegend,
                        hoverinfo="skip",
                    ),
                    row=1,
                    col=column,
                )
            if show_recording_iqr and cohort_statistic != "pooled_fraction":
                lower = [
                    (
                        None
                        if row.get("p25_recording_fraction") is None
                        else 100.0 * float(row["p25_recording_fraction"])
                    )
                    for row in ordered
                ]
                upper = [
                    (
                        None
                        if row.get("p75_recording_fraction") is None
                        else 100.0 * float(row["p75_recording_fraction"])
                    )
                    for row in ordered
                ]
                figure.add_trace(
                    go.Scatter(
                        x=centers + centers[::-1],
                        y=upper + lower[::-1],
                        fill="toself",
                        fillcolor=color,
                        opacity=0.08,
                        line={"width": 0},
                        hoverinfo="skip",
                        legendgroup=label,
                        showlegend=False,
                    ),
                    row=1,
                    col=column,
                )
        figure.update_xaxes(
            title_text=f"{metric['unit']}",
            type="log" if recipe.get("axis_scale") == "log10" else "linear",
            range=(
                [
                    math.log10(float(display_range["display_lower_bound"])),
                    math.log10(float(display_range["display_upper_bound"])),
                ]
                if recipe.get("axis_scale") == "log10"
                else [
                    float(display_range["display_lower_bound"]),
                    float(display_range["display_upper_bound"]),
                ]
            ),
            row=1,
            col=column,
        )
    figure.update_yaxes(
        title_text=f"{COHORT_STATISTIC_LABELS[cohort_statistic]} (%)", row=1, col=1
    )
    warning = " · pooled diagnostic" if cohort_statistic == "pooled_fraction" else ""
    range_note = (
        ""
        if display_range["effective_display_range_id"] == FULL_EVIDENCE_RANGE
        else (
            "<br><sup>Central x-view retains ≥"
            f"{100.0 * float(display_range['minimum_series_fraction_retained']):.2f}% "
            "of every displayed series; full tails remain in the sealed payload</sup>"
        )
    )
    figure.update_layout(
        title=(
            f"{metric['interpretation']} · {str(payload['weighting_id']).title()} "
            f"weighted{warning}{range_note}"
        ),
        template="plotly_white",
        height=520,
        barmode="overlay",
        bargap=0,
        bargroupgap=0,
        hovermode="closest",
        margin={"l": 60, "r": 25, "t": 90, "b": 60},
        legend={"title": {"text": "Provider · role"}, "orientation": "h"},
        meta={
            "display_range": dict(display_range),
            "histogram_rendering": "exact_bin_width_bars_v1",
        },
    )
    return figure


def validated_behavior_motion_trace_figure(
    payload: Mapping[str, object],
) -> go.Figure:
    validate_motion_trace_payload(payload)
    metric = payload["metric"]
    assert isinstance(metric, Mapping)
    points = _rows(payload, "points")
    figure = go.Figure(
        go.Scattergl(
            x=[point["coordinate"] for point in points],
            y=[point["value"] for point in points],
            customdata=[
                [point["acquisition_frame_id"], point["time_s"], point["valid"]]
                for point in points
            ],
            mode="lines",
            line={"color": "#4C78A8", "width": 1},
            connectgaps=False,
            hovertemplate=(
                "frame=%{customdata[0]}<br>time=%{customdata[1]:.4f} s"
                "<br>valid=%{customdata[2]}<br>value=%{y:.5g}<extra></extra>"
            ),
        )
    )
    figure.update_layout(
        title=(
            f"{metric['interpretation']} · {payload['recording_id']} · "
            f"{payload['provider_role']}"
        ),
        template="plotly_white",
        height=420,
        margin={"l": 65, "r": 25, "t": 80, "b": 60},
        xaxis_title=(
            "Acquisition frame ID"
            if payload["coordinate_id"] == "frame"
            else "Time (s)"
        ),
        yaxis_title=f"{metric['unit']}",
        hovermode="closest",
    )
    return figure


__all__ = [
    "distribution_dimension_options",
    "distribution_metric_options",
    "distribution_provenance_rows",
    "validated_behavior_distribution_figure",
    "validated_behavior_motion_trace_figure",
]
