"""Pure Plotly views for receipt-bound validated-behavior cohort statistics.

Every renderer consumes the same normalized payload used by the static report.
It never opens Parquet, discovers ``latest``, or recomputes cohort statistics.
"""

from __future__ import annotations

from collections import defaultdict
import math
from typing import Any, Mapping

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from fisheye.group_statistics.validated_behavior_views import (
    validate_statistics_view_payload,
)

ROBUST_OCCUPANCY_QUANTILE = 0.99
ROBUST_HISTOGRAM_QUANTILE = 0.98


def _rows(payload: Mapping[str, object], key: str) -> list[Mapping[str, Any]]:
    value = payload.get(key)
    if not isinstance(value, list):
        raise ValueError(f"Statistics view payload lacks {key}")
    return [row for row in value if isinstance(row, Mapping)]


def _catalog(payload: Mapping[str, object]) -> dict[str, Mapping[str, Any]]:
    return {str(row["metric_id"]): row for row in _rows(payload, "metric_catalog")}


def _finite(value: object) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _display_scale(unit: str) -> tuple[float, str]:
    return (100.0, "%") if unit == "fraction" else (1.0, unit)


def _condition_order(payload: Mapping[str, object]) -> tuple[str, ...]:
    return tuple(str(value) for value in payload.get("condition_order", ()))


def _conditions_for_metric(
    payload: Mapping[str, object], metric_id: str
) -> tuple[str, ...]:
    observed = {
        str(row["condition"])
        for key in ("recording_rows", "descriptive_rows")
        for row in _rows(payload, key)
        if row.get("metric_id") == metric_id and row.get("condition") is not None
    }
    preferred = _condition_order(payload)
    return tuple(value for value in preferred if value in observed) + tuple(
        sorted(observed - set(preferred))
    )


def _condition_label(payload: Mapping[str, object], condition: str) -> str:
    labels = payload.get("condition_labels")
    return (
        str(labels.get(condition, condition))
        if isinstance(labels, Mapping)
        else condition
    )


def _condition_color(payload: Mapping[str, object], condition: str) -> str:
    colors = payload.get("condition_colors")
    return (
        str(colors.get(condition, "#666666"))
        if isinstance(colors, Mapping)
        else "#666666"
    )


def _role_color(payload: Mapping[str, object], role: str) -> str:
    styles = payload.get("behavior_role_styles")
    if isinstance(styles, Mapping):
        style = styles.get(role)
        if isinstance(style, Mapping):
            color = style.get("aggregate_color_css")
            if isinstance(color, str):
                return color
            color = style.get("aggregate_color_hex")
            if isinstance(color, str):
                return color
    colors = payload.get("behavior_role_colors")
    return (
        str(colors.get(role, "#555555")) if isinstance(colors, Mapping) else "#555555"
    )


def _role_symbol(payload: Mapping[str, object], role: str) -> str:
    styles = payload.get("behavior_role_styles")
    if isinstance(styles, Mapping):
        style = styles.get(role)
        if isinstance(style, Mapping):
            symbol = style.get("plotly_role_symbol")
            if isinstance(symbol, str) and symbol:
                return symbol
    return "circle"


def _provider_dash(payload: Mapping[str, object], provider: str) -> str:
    styles = payload.get("provider_line_styles")
    style = (
        str(styles.get(provider, "solid")) if isinstance(styles, Mapping) else "solid"
    )
    return {"solid": "solid", "dashed": "dash", "dotted": "dot"}.get(style, "solid")


def _rgba(color: str, alpha: float) -> str:
    stripped = color.removeprefix("#")
    if len(stripped) != 6:
        return f"rgba(90,90,90,{alpha})"
    red, green, blue = (int(stripped[index : index + 2], 16) for index in (0, 2, 4))
    return f"rgba({red},{green},{blue},{alpha})"


def _metric_label(metric: Mapping[str, Any]) -> str:
    return str(metric.get("interpretation") or metric["metric_id"])


def statistics_metric_options(payload: Mapping[str, object]) -> dict[str, str]:
    """Return unique user-facing label -> exact metric-ID options."""

    result: dict[str, str] = {}
    for metric in _rows(payload, "metric_catalog"):
        metric_id = str(metric["metric_id"])
        label = _metric_label(metric)
        if label in result:
            label = f"{label} [{metric_id}]"
        result[label] = metric_id
    return result


def statistics_dimension_options(
    payload: Mapping[str, object],
    dimension: str,
) -> tuple[str, ...]:
    """Return sorted exact dimension values represented by descriptive rows."""

    values = {
        str(row[dimension])
        for row in _rows(payload, "descriptive_rows")
        if row.get(dimension) is not None
    }
    if dimension == "condition":
        ordered = [value for value in _condition_order(payload) if value in values]
        return tuple(ordered + sorted(values - set(ordered)))
    return tuple(sorted(values))


def statistics_provenance_rows(
    payload: Mapping[str, object],
) -> list[dict[str, object]]:
    source = payload.get("source_statistics")
    source_mapping = source if isinstance(source, Mapping) else {}
    rows = [
        {"field": "view_payload_sha256", "value": payload.get("payload_sha256")},
        {
            "field": "statistics_run_id",
            "value": source_mapping.get("statistics_run_id"),
        },
        {
            "field": "statistics_manifest_sha256",
            "value": source_mapping.get("statistics_manifest_sha256"),
        },
        {
            "field": "source_export_manifest_sha256",
            "value": source_mapping.get("source_export_manifest_sha256"),
        },
        {
            "field": "experimental_unit",
            "value": source_mapping.get("experimental_unit"),
        },
        {"field": "cohort_weighting", "value": source_mapping.get("cohort_weighting")},
        {
            "field": "acquisition_batch_adjustment",
            "value": source_mapping.get("acquisition_batch_adjustment"),
        },
    ]
    for recipe in _rows(payload, "histogram_recipes"):
        rows.append(
            {
                "field": f"histogram_recipe_sha256:{recipe.get('metric_id')}",
                "value": recipe.get("histogram_recipe_sha256"),
            }
        )
    return rows


def statistics_contrast_rows(
    payload: Mapping[str, object], metric_id: str
) -> list[dict[str, object]]:
    return [
        dict(row)
        for row in _rows(payload, "contrast_rows")
        if str(row.get("metric_id")) == metric_id
    ]


def _figure_layout(
    figure: go.Figure,
    *,
    title: str,
    height: int = 520,
    legend_title: str | None = None,
) -> go.Figure:
    figure.update_layout(
        title=title,
        template="plotly_white",
        height=height,
        hovermode="closest",
        margin={"l": 65, "r": 30, "t": 85, "b": 65},
        legend={"title": {"text": legend_title or "Series"}},
    )
    return figure


def core_behavior_metric_figure(
    payload: Mapping[str, object], metric_id: str
) -> go.Figure:
    catalog = _catalog(payload)
    metric = catalog[metric_id]
    scale, display_unit = _display_scale(str(metric["unit"]))
    conditions = _conditions_for_metric(payload, metric_id)
    rows = [
        row for row in _rows(payload, "recording_rows") if row["metric_id"] == metric_id
    ]
    by_recording: dict[str, dict[str, float]] = defaultdict(dict)
    for row in rows:
        value = _finite(row.get("value"))
        if value is not None:
            by_recording[str(row["recording_id"])][str(row["condition"])] = (
                value * scale
            )

    figure = go.Figure()
    for recording_id, values in sorted(by_recording.items()):
        selected = [condition for condition in conditions if condition in values]
        if len(selected) < 2:
            continue
        figure.add_trace(
            go.Scatter(
                x=[_condition_label(payload, value) for value in selected],
                y=[values[value] for value in selected],
                mode="lines",
                line={"color": "rgba(110,110,110,0.16)", "width": 0.8},
                hovertemplate=f"recording={recording_id}<br>%{{x}}: %{{y:.4g}} {display_unit}<extra></extra>",
                showlegend=False,
            )
        )

    summaries = {
        str(row["condition"]): row
        for row in _rows(payload, "descriptive_rows")
        if row["metric_id"] == metric_id
    }
    for condition in conditions:
        row = summaries.get(condition)
        if row is None:
            continue
        median = _finite(row.get("median"))
        p25 = _finite(row.get("p25"))
        p75 = _finite(row.get("p75"))
        mean = _finite(row.get("mean"))
        color = _condition_color(payload, condition)
        label = _condition_label(payload, condition)
        if median is not None and p25 is not None and p75 is not None:
            figure.add_trace(
                go.Scatter(
                    x=[label],
                    y=[median * scale],
                    mode="markers",
                    marker={"color": color, "size": 11, "symbol": "circle"},
                    error_y={
                        "type": "data",
                        "symmetric": False,
                        "array": [(p75 - median) * scale],
                        "arrayminus": [(median - p25) * scale],
                        "thickness": 2,
                        "width": 5,
                    },
                    name=f"{label} median/IQR",
                    customdata=[[int(row["finite_recording_count"])]],
                    hovertemplate=(
                        "%{x}<br>median=%{y:.4g} "
                        f"{display_unit}<br>finite recordings=%{{customdata[0]}}<extra></extra>"
                    ),
                )
            )
        if mean is not None:
            figure.add_trace(
                go.Scatter(
                    x=[label],
                    y=[mean * scale],
                    mode="markers",
                    marker={
                        "color": "white",
                        "line": {"color": color, "width": 2},
                        "size": 9,
                        "symbol": "diamond",
                    },
                    name=f"{label} mean",
                    hovertemplate=f"%{{x}}<br>mean=%{{y:.4g}} {display_unit}<extra></extra>",
                )
            )
    figure.update_yaxes(title_text=display_unit)
    return _figure_layout(
        figure,
        title=f"{_metric_label(metric)} across exact chaser epochs",
        legend_title="Summary",
    )


def distance_traveled_metric_figure(
    payload: Mapping[str, object], metric_id: str
) -> go.Figure:
    """Render one receipt-bound session or epoch distance metric."""

    figure = core_behavior_metric_figure(payload, metric_id)
    metric = _catalog(payload)[metric_id]
    figure.update_layout(title=f"{_metric_label(metric)} · equal recording weight")
    return figure


def grouped_epoch_metric_figure(
    payload: Mapping[str, object],
    metric_id: str,
    *,
    provider_role: str | None = None,
    behavior_role: str | None = None,
) -> go.Figure:
    metric = _catalog(payload)[metric_id]
    scale, display_unit = _display_scale(str(metric["unit"]))
    conditions = _condition_order(payload)
    rows = [
        row
        for row in _rows(payload, "descriptive_rows")
        if row["metric_id"] == metric_id
        and (
            provider_role is None
            or str(row.get("provider_role", "all")) == provider_role
        )
        and (
            behavior_role is None
            or str(row.get("behavior_role", "all")) == behavior_role
        )
    ]
    series = sorted(
        {
            (str(row.get("provider_role", "all")), str(row.get("behavior_role", "all")))
            for row in rows
        }
    )
    figure = go.Figure()
    for provider, role in series:
        keyed = {
            str(row["condition"]): row
            for row in rows
            if str(row.get("provider_role", "all")) == provider
            and str(row.get("behavior_role", "all")) == role
        }
        points = []
        for condition in conditions:
            row = keyed.get(condition)
            if row is None:
                continue
            values = tuple(
                _finite(row.get(field)) for field in ("median", "p25", "p75")
            )
            if any(value is None for value in values):
                continue
            median, p25, p75 = values
            assert median is not None and p25 is not None and p75 is not None
            points.append(
                (condition, median, p25, p75, int(row["finite_recording_count"]))
            )
        if not points:
            continue
        label_parts = [part.title() for part in (role, provider) if part != "all"]
        figure.add_trace(
            go.Scatter(
                x=[_condition_label(payload, point[0]) for point in points],
                y=[point[1] * scale for point in points],
                mode="lines+markers",
                line={
                    "color": _role_color(payload, role),
                    "dash": _provider_dash(payload, provider),
                    "width": 2.4,
                },
                marker={"size": 8, "symbol": _role_symbol(payload, role)},
                error_y={
                    "type": "data",
                    "symmetric": False,
                    "array": [(point[3] - point[1]) * scale for point in points],
                    "arrayminus": [(point[1] - point[2]) * scale for point in points],
                    "thickness": 1.3,
                    "width": 3,
                },
                customdata=[[point[4]] for point in points],
                name=" · ".join(label_parts) or "Cohort",
                hovertemplate=(
                    "%{x}<br>median=%{y:.4g} "
                    f"{display_unit}<br>finite recordings=%{{customdata[0]}}<extra>%{{fullData.name}}</extra>"
                ),
            )
        )
    figure.update_yaxes(title_text=display_unit)
    return _figure_layout(figure, title=_metric_label(metric))


def _distance_band_label(row: Mapping[str, Any]) -> str | None:
    if row.get("distance_bin_index") is None:
        return None
    start = float(row["distance_bin_start_mm"])
    end = _finite(row.get("distance_bin_end_mm"))
    return f"{start:g}–∞" if end is None else f"{start:g}–{end:g}"


def _curve_x(row: Mapping[str, Any]) -> float:
    if row.get("distance_bin_index") is not None:
        return float(row["distance_bin_index"])
    if row.get("distance_bin_center_mm") is not None:
        return float(row["distance_bin_center_mm"])
    if (
        row.get("radial_bin_start_mm") is not None
        and row.get("radial_bin_end_mm") is not None
    ):
        return (
            float(row["radial_bin_start_mm"]) + float(row["radial_bin_end_mm"])
        ) / 2.0
    if row.get("threshold_mm") is not None:
        return float(row["threshold_mm"])
    raise ValueError("Distance-curve payload lacks an exact persisted x coordinate")


def distance_curve_metric_figure(
    payload: Mapping[str, object],
    metric_id: str,
    *,
    provider_role: str | None = None,
    behavior_role: str | None = None,
) -> go.Figure:
    metric = _catalog(payload)[metric_id]
    scale, display_unit = _display_scale(str(metric["unit"]))
    rows = [
        row
        for row in _rows(payload, "descriptive_rows")
        if row["metric_id"] == metric_id
        and (
            provider_role is None
            or str(row.get("provider_role", "all")) == provider_role
        )
        and (
            behavior_role is None
            or str(row.get("behavior_role", "all")) == behavior_role
        )
    ]
    roles = sorted({str(row.get("behavior_role", "all")) for row in rows})
    providers = sorted({str(row.get("provider_role", "all")) for row in rows})
    facets = [(provider, role) for provider in providers for role in roles]
    if not facets:
        return _figure_layout(go.Figure(), title=f"{_metric_label(metric)} — no rows")
    titles = [
        " · ".join(part.title() for part in (role, provider) if part != "all")
        or "Cohort"
        for provider, role in facets
    ]
    figure = make_subplots(rows=1, cols=len(facets), subplot_titles=titles)
    for column, (provider, role) in enumerate(facets, start=1):
        facet_rows = [
            row
            for row in rows
            if str(row.get("provider_role", "all")) == provider
            and str(row.get("behavior_role", "all")) == role
        ]
        for condition in _condition_order(payload):
            points = []
            for row in facet_rows:
                if str(row["condition"]) != condition:
                    continue
                values = tuple(
                    _finite(row.get(field)) for field in ("median", "p25", "p75")
                )
                if any(value is None for value in values):
                    continue
                median, p25, p75 = values
                assert median is not None and p25 is not None and p75 is not None
                points.append(
                    (
                        _curve_x(row),
                        median * scale,
                        p25 * scale,
                        p75 * scale,
                        _distance_band_label(row),
                    )
                )
            points.sort(key=lambda point: point[0])
            if not points:
                continue
            x = [point[0] for point in points]
            color = _condition_color(payload, condition)
            show_legend = column == 1
            figure.add_trace(
                go.Scatter(
                    x=x,
                    y=[point[3] for point in points],
                    mode="lines",
                    line={"width": 0, "color": color},
                    hoverinfo="skip",
                    showlegend=False,
                ),
                row=1,
                col=column,
            )
            figure.add_trace(
                go.Scatter(
                    x=x,
                    y=[point[2] for point in points],
                    mode="lines",
                    line={"width": 0, "color": color},
                    fill="tonexty",
                    fillcolor=_rgba(color, 0.13),
                    hoverinfo="skip",
                    showlegend=False,
                ),
                row=1,
                col=column,
            )
            figure.add_trace(
                go.Scatter(
                    x=x,
                    y=[point[1] for point in points],
                    mode="lines",
                    line={"color": color, "width": 2.2},
                    name=_condition_label(payload, condition),
                    legendgroup=condition,
                    showlegend=show_legend,
                    customdata=[[point[4] or f"{point[0]:g} mm"] for point in points],
                    hovertemplate=(
                        "distance=%{customdata[0]}<br>median=%{y:.4g} "
                        f"{display_unit}<extra>%{{fullData.name}}</extra>"
                    ),
                ),
                row=1,
                col=column,
            )
        band_labels = {
            int(row["distance_bin_index"]): _distance_band_label(row)
            for row in facet_rows
            if row.get("distance_bin_index") is not None
        }
        if band_labels:
            ticks = sorted(band_labels)
            figure.update_xaxes(
                title_text="Fish–chaser distance band (mm)",
                tickmode="array",
                tickvals=ticks,
                ticktext=[band_labels[tick] for tick in ticks],
                row=1,
                col=column,
            )
        else:
            figure.update_xaxes(
                title_text="Fish–chaser distance (mm)", row=1, col=column
            )
        figure.update_yaxes(
            title_text=display_unit if column == 1 else None, row=1, col=column
        )
    return _figure_layout(
        figure,
        title=_metric_label(metric),
        height=560,
        legend_title="Epoch",
    ).update_layout(width=max(760, 420 * len(facets)))


def trial_response_metric_figure(
    payload: Mapping[str, object], metric_id: str, *, behavior_role: str | None = None
) -> go.Figure:
    metric = _catalog(payload)[metric_id]
    scale, display_unit = _display_scale(str(metric["unit"]))
    rows = [
        row
        for row in _rows(payload, "descriptive_rows")
        if row["metric_id"] == metric_id
        and (
            behavior_role is None
            or str(row.get("behavior_role", "all")) == behavior_role
        )
    ]
    roles = sorted({str(row.get("behavior_role", "all")) for row in rows})
    figure = go.Figure()
    for role in roles:
        points = []
        for row in rows:
            if str(row.get("behavior_role", "all")) != role:
                continue
            values = tuple(
                _finite(row.get(field)) for field in ("median", "p25", "p75")
            )
            if any(value is None for value in values):
                continue
            median, p25, p75 = values
            assert median is not None and p25 is not None and p75 is not None
            points.append(
                (
                    int(row["trial_ordinal"]),
                    median * scale,
                    p25 * scale,
                    p75 * scale,
                    int(row["finite_recording_count"]),
                )
            )
        points.sort(key=lambda point: point[0])
        if not points:
            continue
        figure.add_trace(
            go.Scatter(
                x=[point[0] for point in points],
                y=[point[1] for point in points],
                mode="lines+markers",
                line={"color": _role_color(payload, role), "width": 2.4},
                marker={"symbol": _role_symbol(payload, role), "size": 8},
                error_y={
                    "type": "data",
                    "symmetric": False,
                    "array": [point[3] - point[1] for point in points],
                    "arrayminus": [point[1] - point[2] for point in points],
                    "thickness": 1.3,
                    "width": 3,
                },
                customdata=[[point[4]] for point in points],
                name=role.title(),
                hovertemplate=(
                    "trial=%{x}<br>median=%{y:.4g} "
                    f"{display_unit}<br>finite recordings=%{{customdata[0]}}<extra>%{{fullData.name}}</extra>"
                ),
            )
        )
    figure.update_xaxes(title_text="Trial ordinal", dtick=1)
    figure.update_yaxes(title_text=display_unit)
    return _figure_layout(
        figure, title=_metric_label(metric), legend_title="Behavior role"
    )


def spatial_occupancy_metric_figure(
    payload: Mapping[str, object],
    metric_id: str,
    *,
    provider_role: str | None = None,
    condition: str | None = None,
    statistic: str = "mean",
    color_scale_quantile: float = ROBUST_OCCUPANCY_QUANTILE,
) -> go.Figure:
    if statistic not in {"mean", "median"}:
        raise ValueError("Spatial occupancy statistic must be mean or median")
    if not 0.5 <= color_scale_quantile <= 1.0:
        raise ValueError("Spatial occupancy color-scale quantile must be in [0.5, 1.0]")
    metric = _catalog(payload)[metric_id]
    scale, display_unit = _display_scale(str(metric["unit"]))
    all_rows = [
        row
        for row in _rows(payload, "descriptive_rows")
        if row["metric_id"] == metric_id
    ]
    providers = sorted({str(row["provider_role"]) for row in all_rows})
    conditions = statistics_dimension_options(payload, "condition")
    selected_provider = provider_role or (providers[0] if providers else None)
    selected_condition = condition or (conditions[0] if conditions else None)
    selected = [
        row
        for row in all_rows
        if str(row["provider_role"]) == selected_provider
        and str(row["condition"]) == selected_condition
    ]
    x_indices = sorted({int(row["x_bin_index"]) for row in selected})
    y_indices = sorted({int(row["y_bin_index"]) for row in selected})
    if not x_indices or not y_indices:
        return _figure_layout(go.Figure(), title="Spatial occupancy — no rows")
    if x_indices != list(range(len(x_indices))) or y_indices != list(
        range(len(y_indices))
    ):
        raise ValueError("Spatial occupancy bin indices are not contiguous")
    grid = np.full((len(y_indices), len(x_indices)), np.nan, dtype=np.float64)
    counts = np.full_like(grid, np.nan)
    x_centers = np.full(len(x_indices), np.nan, dtype=np.float64)
    y_centers = np.full(len(y_indices), np.nan, dtype=np.float64)
    seen_coordinates: set[tuple[int, int]] = set()
    seen_member_coordinates: set[tuple[int, int]] = set()
    for row in selected:
        x_index = int(row["x_bin_index"])
        y_index = int(row["y_bin_index"])
        coordinate = (x_index, y_index)
        seen_coordinates.add(coordinate)
        x_center = (float(row["x_bin_start_mm"]) + float(row["x_bin_end_mm"])) / 2.0
        y_center = (float(row["y_bin_start_mm"]) + float(row["y_bin_end_mm"])) / 2.0
        if np.isfinite(x_centers[x_index]) and x_centers[x_index] != x_center:
            raise ValueError("Spatial occupancy view x-bin centers are inconsistent")
        if np.isfinite(y_centers[y_index]) and y_centers[y_index] != y_center:
            raise ValueError("Spatial occupancy view y-bin centers are inconsistent")
        x_centers[x_index] = x_center
        y_centers[y_index] = y_center
        value = _finite(row.get(statistic))
        if bool(row["arena_bin_center_member"]):
            if coordinate in seen_member_coordinates:
                raise ValueError(
                    "Spatial occupancy view has a duplicate arena-member bin"
                )
            seen_member_coordinates.add(coordinate)
        if bool(row["arena_bin_center_member"]) and value is not None:
            grid[y_index, x_index] = value * scale
            counts[y_index, x_index] = int(row["finite_recording_count"])
    if len(seen_coordinates) != len(x_indices) * len(y_indices):
        raise ValueError("Spatial occupancy view grid is incomplete")

    all_values = np.asarray(
        [
            value * scale
            for row in all_rows
            if bool(row["arena_bin_center_member"])
            and (value := _finite(row.get(statistic))) is not None
            and value > 0
        ],
        dtype=np.float64,
    )
    zmax = (
        float(np.quantile(all_values, color_scale_quantile)) if all_values.size else 1.0
    )
    zmax = max(zmax, np.finfo(np.float64).eps)
    figure = go.Figure(
        go.Heatmap(
            x=x_centers,
            y=y_centers,
            z=grid,
            zmin=0.0,
            zmax=zmax,
            colorscale="Viridis",
            colorbar={"title": display_unit},
            customdata=counts,
            hovertemplate=(
                "x=%{x:.3g} mm<br>y=%{y:.3g} mm<br>"
                f"{statistic}=%{{z:.4g}} {display_unit}<br>finite recordings=%{{customdata:.0f}}<extra></extra>"
            ),
        )
    )
    figure.update_xaxes(title_text="x (mm)", scaleanchor="y", scaleratio=1)
    figure.update_yaxes(title_text="y (mm; +down)", autorange="reversed")
    return _figure_layout(
        figure,
        title=(
            f"Spatial occupancy · {str(selected_provider).title()} · "
            f"{_condition_label(payload, str(selected_condition))} · "
            f"shared q{color_scale_quantile:.2f} scale"
        ),
        height=680,
    )


def _bearing_histogram_panels(
    payload: Mapping[str, object],
    metric_id: str,
    *,
    behavior_role: str | None,
    condition: str | None,
) -> tuple[
    tuple[str, ...],
    tuple[str, ...],
    dict[tuple[str, str], list[Mapping[str, Any]]],
]:
    rows = [
        row
        for row in _rows(payload, "descriptive_rows")
        if str(row["metric_id"]) == metric_id
        and (behavior_role is None or str(row.get("behavior_role")) == behavior_role)
        and (condition is None or str(row["condition"]) == condition)
    ]
    represented = {str(row["condition"]) for row in rows}
    conditions = tuple(
        value for value in _condition_order(payload) if value in represented
    )
    roles = tuple(sorted({str(row["behavior_role"]) for row in rows}))
    if not conditions or not roles:
        raise ValueError("Body-bearing histogram selection has no exact panels")
    panels = {
        (epoch, role): [
            row
            for row in rows
            if str(row["condition"]) == epoch and str(row["behavior_role"]) == role
        ]
        for epoch in conditions
        for role in roles
    }
    if any(not panel for panel in panels.values()):
        raise ValueError("Body-bearing histogram panel registry is incomplete")
    return conditions, roles, panels


def body_bearing_polar_metric_figure(
    payload: Mapping[str, object],
    metric_id: str,
    *,
    behavior_role: str | None = None,
    condition: str | None = None,
) -> go.Figure:
    """Render persisted equal-recording signed-bearing fractions."""

    conditions, roles, panels = _bearing_histogram_panels(
        payload,
        metric_id,
        behavior_role=behavior_role,
        condition=condition,
    )
    titles = [
        f"{_condition_label(payload, epoch)} · {role.title()}"
        for epoch in conditions
        for role in roles
    ]
    figure = make_subplots(
        rows=len(conditions),
        cols=len(roles),
        specs=[[{"type": "polar"} for _ in roles] for _ in conditions],
        subplot_titles=titles,
    )
    maximum = 0.0
    prepared: dict[
        tuple[str, str], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ] = {}
    for key, rows in panels.items():
        if any(row.get("axis_0_id") != "bearing" for row in rows) or any(
            row.get("axis_1_id") is not None for row in rows
        ):
            raise ValueError("Signed-bearing polar payload has incompatible axes")
        ordered = sorted(rows, key=lambda row: int(row["axis_0_bin_index"]))
        indices = [int(row["axis_0_bin_index"]) for row in ordered]
        if indices != list(range(len(indices))):
            raise ValueError("Signed-bearing polar bins are not contiguous")
        starts = np.asarray(
            [float(row["axis_0_bin_start"]) for row in ordered], dtype=np.float64
        )
        ends = np.asarray(
            [float(row["axis_0_bin_end"]) for row in ordered], dtype=np.float64
        )
        fractions = np.asarray(
            [float(row["mean_fraction"]) for row in ordered], dtype=np.float64
        )
        support = np.asarray(
            [int(row["finite_recording_count"]) for row in ordered],
            dtype=np.int64,
        )
        if (
            not np.isclose(starts[0], -180.0)
            or not np.isclose(ends[-1], 180.0)
            or np.any(~np.isclose(ends[:-1], starts[1:]))
            or np.any(~np.isfinite(fractions) | (fractions < 0.0))
            or not np.isclose(np.sum(fractions), 1.0, atol=1e-8)
            or np.unique(support).size != 1
        ):
            raise ValueError("Signed-bearing polar panel is incomplete or unnormalized")
        percentages = fractions * 100.0
        maximum = max(maximum, float(np.max(percentages)))
        prepared[key] = (starts, ends, percentages, support)

    for row_index, epoch in enumerate(conditions, start=1):
        for column_index, role in enumerate(roles, start=1):
            starts, ends, percentages, support = prepared[(epoch, role)]
            centers = (starts + ends) / 2.0
            custom = np.column_stack((starts, ends, support))
            figure.add_trace(
                go.Barpolar(
                    theta=centers,
                    r=percentages,
                    width=ends - starts,
                    marker={
                        "color": _role_color(payload, role),
                        "line": {"color": "white", "width": 0.3},
                    },
                    opacity=0.85,
                    name=role.title(),
                    showlegend=False,
                    customdata=custom,
                    hovertemplate=(
                        "bearing %{customdata[0]:.0f}° to "
                        "%{customdata[1]:.0f}°<br>mean recording "
                        "fraction=%{r:.4g}%<br>finite recordings="
                        "%{customdata[2]:.0f}<extra></extra>"
                    ),
                ),
                row=row_index,
                col=column_index,
            )
    figure.update_polars(
        radialaxis={
            "title": "mean recording fraction/bin (%)",
            "range": [0.0, max(maximum * 1.08, 1.0)],
        },
        angularaxis={
            "direction": "counterclockwise",
            "rotation": 90,
            "tickmode": "array",
            "tickvals": [-180, -90, 0, 90, 180],
            "ticktext": ["behind", "right", "front", "left", "behind"],
        },
    )
    return _figure_layout(
        figure,
        title="Signed anatomical bearing to the chaser",
        height=max(560, 330 * len(conditions)),
    ).update_layout(width=max(760, 430 * len(roles)))


def body_bearing_distance_metric_figure(
    payload: Mapping[str, object],
    metric_id: str,
    *,
    behavior_role: str | None = None,
    condition: str | None = None,
    color_scale_quantile: float = ROBUST_HISTOGRAM_QUANTILE,
) -> go.Figure:
    """Render persisted joint bearing-by-distance cohort densities."""

    if not 0.5 <= float(color_scale_quantile) <= 1.0:
        raise ValueError("Histogram color-scale quantile must be in [0.5, 1.0]")
    conditions, roles, panels = _bearing_histogram_panels(
        payload,
        metric_id,
        behavior_role=behavior_role,
        condition=condition,
    )
    titles = [
        f"{_condition_label(payload, epoch)} · {role.title()}"
        for epoch in conditions
        for role in roles
    ]
    figure = make_subplots(
        rows=len(conditions),
        cols=len(roles),
        specs=[[{"type": "polar"} for _ in roles] for _ in conditions],
        subplot_titles=titles,
    )
    prepared: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    positive: list[float] = []
    radial_maximum = 0.0
    for key, rows in panels.items():
        if any(
            row.get("axis_0_id") != "bearing" or row.get("axis_1_id") != "distance"
            for row in rows
        ):
            raise ValueError("Bearing-by-distance payload has incompatible axes")
        bearing_indices = sorted({int(row["axis_0_bin_index"]) for row in rows})
        distance_indices = sorted({int(row["axis_1_bin_index"]) for row in rows})
        if bearing_indices != list(
            range(len(bearing_indices))
        ) or distance_indices != list(range(len(distance_indices))):
            raise ValueError("Bearing-by-distance bin indices are not contiguous")
        expected_bins = len(bearing_indices) * len(distance_indices)
        if len(rows) != expected_bins:
            raise ValueError("Bearing-by-distance panel grid is incomplete")
        values = np.asarray(
            [float(row["mean_fraction"]) for row in rows], dtype=np.float64
        )
        supports = {int(row["finite_recording_count"]) for row in rows}
        if (
            np.any(~np.isfinite(values) | (values < 0.0))
            or not np.isclose(np.sum(values), 1.0, atol=1e-8)
            or len(supports) != 1
        ):
            raise ValueError("Bearing-by-distance panel is incomplete or unnormalized")
        positive.extend((values[values > 0.0] * 100.0).tolist())
        radial_maximum = max(
            radial_maximum,
            max(float(row["axis_1_bin_end"]) for row in rows),
        )
        prepared[key] = rows
    color_maximum = (
        float(np.quantile(np.asarray(positive), color_scale_quantile))
        if positive
        else 1.0
    )
    color_maximum = max(color_maximum, float(np.finfo(np.float64).eps))

    show_colorbar = True
    for row_index, epoch in enumerate(conditions, start=1):
        for column_index, role in enumerate(roles, start=1):
            rows = sorted(
                prepared[(epoch, role)],
                key=lambda row: (
                    int(row["axis_1_bin_index"]),
                    int(row["axis_0_bin_index"]),
                ),
            )
            theta = np.asarray(
                [
                    (float(row["axis_0_bin_start"]) + float(row["axis_0_bin_end"]))
                    / 2.0
                    for row in rows
                ]
            )
            width = np.asarray(
                [
                    float(row["axis_0_bin_end"]) - float(row["axis_0_bin_start"])
                    for row in rows
                ]
            )
            base = np.asarray([float(row["axis_1_bin_start"]) for row in rows])
            radial_width = np.asarray(
                [
                    float(row["axis_1_bin_end"]) - float(row["axis_1_bin_start"])
                    for row in rows
                ]
            )
            percentages = np.asarray(
                [float(row["mean_fraction"]) * 100.0 for row in rows]
            )
            custom = np.column_stack(
                (
                    [float(row["axis_0_bin_start"]) for row in rows],
                    [float(row["axis_0_bin_end"]) for row in rows],
                    base,
                    base + radial_width,
                    [int(row["finite_recording_count"]) for row in rows],
                    [int(row["source_bin_count_sum"]) for row in rows],
                    [int(row["source_denominator_count_sum"]) for row in rows],
                )
            )
            figure.add_trace(
                go.Barpolar(
                    theta=theta,
                    r=radial_width,
                    base=base,
                    width=width,
                    marker={
                        "color": percentages,
                        "colorscale": "Viridis",
                        "cmin": 0.0,
                        "cmax": color_maximum,
                        "showscale": show_colorbar,
                        "colorbar": {"title": "mean %/bin"},
                        "line": {"width": 0},
                    },
                    opacity=0.96,
                    showlegend=False,
                    customdata=custom,
                    hovertemplate=(
                        "bearing %{customdata[0]:.0f}° to "
                        "%{customdata[1]:.0f}°<br>distance "
                        "%{customdata[2]:.0f} to %{customdata[3]:.0f} mm"
                        "<br>mean recording fraction=%{marker.color:.5f}%"
                        "<br>finite recordings=%{customdata[4]:.0f}"
                        "<br>source rows=%{customdata[5]:.0f} / "
                        "%{customdata[6]:.0f}<extra></extra>"
                    ),
                ),
                row=row_index,
                col=column_index,
            )
            show_colorbar = False
    figure.update_polars(
        radialaxis={"title": "distance (mm)", "range": [0.0, radial_maximum]},
        angularaxis={
            "direction": "counterclockwise",
            "rotation": 90,
            "tickmode": "array",
            "tickvals": [-180, -90, 0, 90, 180],
            "ticktext": ["behind", "right", "front", "left", "behind"],
        },
    )
    return _figure_layout(
        figure,
        title=(
            "Signed anatomical bearing × fish–chaser distance · "
            f"shared q{color_scale_quantile:.2f} scale"
        ),
        height=max(580, 350 * len(conditions)),
    ).update_layout(width=max(780, 450 * len(roles)))


def validated_behavior_statistics_figure(
    payload: Mapping[str, object],
    *,
    metric_id: str | None = None,
    provider_role: str | None = None,
    behavior_role: str | None = None,
    condition: str | None = None,
    occupancy_statistic: str = "mean",
) -> go.Figure:
    """Dispatch one interactive view without performing statistical computation."""

    validate_statistics_view_payload(payload)
    selected_metric = metric_id or str(payload["default_metric_id"])
    if selected_metric not in _catalog(payload):
        raise KeyError(f"Metric is unavailable in selected payload: {selected_metric}")
    view_id = str(payload["view_id"])
    if view_id == "core_behavior":
        figure = core_behavior_metric_figure(payload, selected_metric)
    elif view_id == "distance_traveled":
        figure = distance_traveled_metric_figure(payload, selected_metric)
    elif view_id in {"near_field", "same_quadrant", "occupancy_support"}:
        figure = grouped_epoch_metric_figure(
            payload,
            selected_metric,
            provider_role=provider_role,
            behavior_role=behavior_role,
        )
    elif view_id in {
        "bout_response_by_distance",
        "body_alignment_by_distance",
        "radial_distribution",
        "distance_cdf",
    }:
        figure = distance_curve_metric_figure(
            payload,
            selected_metric,
            provider_role=provider_role,
            behavior_role=behavior_role,
        )
    elif view_id == "trial_response":
        figure = trial_response_metric_figure(
            payload,
            selected_metric,
            behavior_role=behavior_role,
        )
    elif view_id == "spatial_occupancy":
        figure = spatial_occupancy_metric_figure(
            payload,
            selected_metric,
            provider_role=provider_role,
            condition=condition,
            statistic=occupancy_statistic,
        )
    elif view_id == "body_bearing_polar":
        figure = body_bearing_polar_metric_figure(
            payload,
            selected_metric,
            behavior_role=behavior_role,
            condition=condition,
        )
    elif view_id == "body_bearing_distance":
        figure = body_bearing_distance_metric_figure(
            payload,
            selected_metric,
            behavior_role=behavior_role,
            condition=condition,
        )
    else:
        raise KeyError(f"No interactive renderer is registered for view {view_id!r}")
    source = payload["source_statistics"]
    assert isinstance(source, Mapping)
    figure.update_layout(
        meta={
            "view_payload_sha256": payload["payload_sha256"],
            "statistics_run_id": source["statistics_run_id"],
            "statistics_manifest_sha256": source["statistics_manifest_sha256"],
            "source_export_manifest_sha256": source["source_export_manifest_sha256"],
            "analysis_status": source["analysis_status"],
            "experimental_unit": source["experimental_unit"],
            "cohort_weighting": source["cohort_weighting"],
            "acquisition_batch_adjustment": source["acquisition_batch_adjustment"],
            "histogram_color_scale_quantile": (
                ROBUST_HISTOGRAM_QUANTILE
                if view_id == "body_bearing_distance"
                else None
            ),
            "histogram_recipe_sha256": [
                record.get("histogram_recipe_sha256")
                for record in payload.get("histogram_recipes", [])
                if isinstance(record, Mapping)
            ],
        }
    )
    return figure


__all__ = [
    "ROBUST_HISTOGRAM_QUANTILE",
    "ROBUST_OCCUPANCY_QUANTILE",
    "body_bearing_distance_metric_figure",
    "body_bearing_polar_metric_figure",
    "core_behavior_metric_figure",
    "distance_traveled_metric_figure",
    "distance_curve_metric_figure",
    "grouped_epoch_metric_figure",
    "spatial_occupancy_metric_figure",
    "statistics_contrast_rows",
    "statistics_dimension_options",
    "statistics_metric_options",
    "statistics_provenance_rows",
    "trial_response_metric_figure",
    "validated_behavior_statistics_figure",
]
