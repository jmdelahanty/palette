"""Static Matplotlib figures from shared grouped-statistics view payloads."""

from __future__ import annotations

from collections import defaultdict
import math
from textwrap import fill
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from fisheye.group_statistics.validated_behavior_views import (
    validate_statistics_view_payload,
)

ROBUST_OCCUPANCY_QUANTILE = 0.99
ROBUST_HISTOGRAM_QUANTILE = 0.98
PLOT_DPI = 170


def _rows(payload: Mapping[str, object], key: str) -> list[Mapping[str, Any]]:
    value = payload.get(key)
    if not isinstance(value, list):
        raise ValueError(f"Statistics view payload lacks {key}")
    return [row for row in value if isinstance(row, Mapping)]


def _catalog(payload: Mapping[str, object]) -> dict[str, Mapping[str, Any]]:
    rows = _rows(payload, "metric_catalog")
    return {str(row["metric_id"]): row for row in rows}


def _finite(value: Any) -> float | None:
    if value is None:
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _display_scale(unit: str) -> tuple[float, str]:
    return (100.0, "%") if unit == "fraction" else (1.0, unit)


def _metric_label(record: Mapping[str, Any]) -> str:
    return str(record.get("interpretation") or record["metric_id"])


def _panel_metric_label(record: Mapping[str, Any], *, width: int = 46) -> str:
    return fill(_metric_label(record), width=width)


def _axis_metric_label(record: Mapping[str, Any]) -> str:
    label = str(record.get("value_column") or record["metric_id"]).replace("_", " ")
    suffix = {
        "1/min": " per min",
        "s": " s",
        "mm": " mm",
        "mm/s": " mm s",
        "deg": " deg",
    }.get(str(record.get("unit")))
    if suffix and label.endswith(suffix):
        label = label[: -len(suffix)]
    return label.capitalize()


def _condition_order(payload: Mapping[str, object]) -> tuple[str, ...]:
    return tuple(str(value) for value in payload.get("condition_order", ()))


def _condition_label(payload: Mapping[str, object], condition: str) -> str:
    labels = payload.get("condition_labels")
    if isinstance(labels, Mapping):
        return str(labels.get(condition, condition))
    return condition


def _condition_color(payload: Mapping[str, object], condition: str) -> str:
    colors = payload.get("condition_colors")
    if isinstance(colors, Mapping):
        return str(colors.get(condition, "#666666"))
    return "#666666"


def _role_color(payload: Mapping[str, object], role: str) -> str:
    styles = payload.get("behavior_role_styles")
    if isinstance(styles, Mapping):
        style = styles.get(role)
        if isinstance(style, Mapping):
            color = style.get("aggregate_color_hex")
            if isinstance(color, str):
                return color
    colors = payload.get("behavior_role_colors")
    if isinstance(colors, Mapping):
        return str(colors.get(role, "#666666"))
    return "#666666"


def _role_marker(payload: Mapping[str, object], role: str) -> str:
    styles = payload.get("behavior_role_styles")
    if isinstance(styles, Mapping):
        style = styles.get(role)
        if isinstance(style, Mapping):
            marker = style.get("matplotlib_role_marker")
            if isinstance(marker, str) and marker:
                return marker
    return "o"


def _provider_linestyle(payload: Mapping[str, object], provider: str) -> str:
    styles = payload.get("provider_line_styles")
    style = (
        str(styles.get(provider, "solid")) if isinstance(styles, Mapping) else "solid"
    )
    return {"solid": "-", "dashed": "--", "dotted": ":"}.get(style, "-")


def _footer(figure: Any, payload: Mapping[str, object]) -> None:
    source = payload.get("source_statistics")
    digest = (
        str(source.get("statistics_manifest_sha256", "unknown"))[:12]
        if isinstance(source, Mapping)
        else "unknown"
    )
    layout_engine = figure.get_layout_engine()
    if layout_engine is not None:
        # Reserve a stable band for provenance.  Otherwise the footer can
        # overlap the second row of condition labels when tight-bounding a
        # multi-panel figure.
        layout_engine.set(rect=(0.0, 0.065, 1.0, 0.82 if figure.legends else 0.86))
    figure.text(
        0.5,
        0.012,
        (
            "Exploratory · equal recording weight · recording_id unit · "
            f"no batch adjustment · statistics {digest}"
        ),
        ha="center",
        va="bottom",
        fontsize=8,
        color="#555555",
    )


def _style_axis(axis: Any) -> None:
    axis.grid(axis="y", alpha=0.2, linewidth=0.7)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def _subplot_grid(
    count: int, *, columns: int = 4, width: float = 4.2, height: float = 3.5
):
    n_columns = min(columns, max(1, count))
    n_rows = int(math.ceil(count / n_columns))
    figure, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(width * n_columns, height * n_rows),
        squeeze=False,
        constrained_layout=True,
    )
    return figure, axes.reshape(-1)


def core_behavior_figure(payload: Mapping[str, object]) -> Any:
    catalog = _catalog(payload)
    metric_ids = tuple(catalog)
    summaries = _rows(payload, "descriptive_rows")
    recording = _rows(payload, "recording_rows")
    conditions = _condition_order(payload)
    figure, axes = _subplot_grid(len(metric_ids), columns=4)
    for axis, metric_id in zip(axes, metric_ids):
        metric = catalog[metric_id]
        unit = str(metric["unit"])
        scale, display_unit = _display_scale(unit)
        metric_recording = [row for row in recording if row["metric_id"] == metric_id]
        by_recording: dict[str, dict[str, float]] = defaultdict(dict)
        for row in metric_recording:
            value = _finite(row.get("value"))
            if value is not None:
                by_recording[str(row["recording_id"])][str(row["condition"])] = value
        for values in by_recording.values():
            x_values = [
                index
                for index, condition in enumerate(conditions)
                if condition in values
            ]
            y_values = [values[conditions[index]] * scale for index in x_values]
            if len(x_values) >= 2:
                axis.plot(
                    x_values, y_values, color="#777777", alpha=0.10, linewidth=0.6
                )

        metric_summary = {
            str(row["condition"]): row
            for row in summaries
            if row["metric_id"] == metric_id
        }
        labels = []
        for index, condition in enumerate(conditions):
            row = metric_summary.get(condition)
            if row is None:
                labels.append(_condition_label(payload, condition))
                continue
            median = _finite(row.get("median"))
            p25 = _finite(row.get("p25"))
            p75 = _finite(row.get("p75"))
            mean = _finite(row.get("mean"))
            color = _condition_color(payload, condition)
            if median is not None and p25 is not None and p75 is not None:
                axis.errorbar(
                    [index],
                    [median * scale],
                    yerr=[
                        [max(0.0, (median - p25) * scale)],
                        [max(0.0, (p75 - median) * scale)],
                    ],
                    color=color,
                    marker="o",
                    markersize=7,
                    linewidth=2.4,
                    capsize=4,
                    zorder=4,
                )
            if mean is not None:
                axis.scatter(
                    [index],
                    [mean * scale],
                    marker="D",
                    s=26,
                    facecolors="white",
                    edgecolors=color,
                    linewidths=1.3,
                    zorder=5,
                )
            labels.append(
                f"{_condition_label(payload, condition)}\n"
                f"n={int(row['finite_recording_count'])}"
            )
        axis.set_xticks(np.arange(len(conditions)), labels)
        axis.set_ylabel(display_unit)
        axis.set_title(_panel_metric_label(metric), fontsize=10)
        _style_axis(axis)
    for axis in axes[len(metric_ids) :]:
        axis.set_visible(False)
    figure.suptitle("Core behavior across exact chaser epochs", fontsize=16, y=0.99)
    _footer(figure, payload)
    return figure


def distance_traveled_figure(payload: Mapping[str, object]) -> Any:
    """Render session and exact-epoch path summaries without pooling frames."""

    catalog = _catalog(payload)
    metric_ids = tuple(catalog)
    summaries = _rows(payload, "descriptive_rows")
    recording = _rows(payload, "recording_rows")
    preferred = _condition_order(payload)
    figure, axes = _subplot_grid(len(metric_ids), columns=2, width=5.2, height=4.0)
    for axis, metric_id in zip(axes, metric_ids):
        metric = catalog[metric_id]
        scale, display_unit = _display_scale(str(metric["unit"]))
        metric_recording = [row for row in recording if row["metric_id"] == metric_id]
        metric_summaries = [row for row in summaries if row["metric_id"] == metric_id]
        observed = {
            str(row["condition"]) for row in (*metric_recording, *metric_summaries)
        }
        conditions = tuple(value for value in preferred if value in observed) + tuple(
            sorted(observed - set(preferred))
        )
        by_recording: dict[str, dict[str, float]] = defaultdict(dict)
        for row in metric_recording:
            value = _finite(row.get("value"))
            if value is not None:
                by_recording[str(row["recording_id"])][str(row["condition"])] = value
        for values in by_recording.values():
            x_values = [
                index for index, condition in enumerate(conditions) if condition in values
            ]
            y_values = [values[conditions[index]] * scale for index in x_values]
            if len(x_values) >= 2:
                axis.plot(x_values, y_values, color="#777777", alpha=0.10, linewidth=0.6)
            elif x_values:
                axis.scatter(
                    x_values,
                    y_values,
                    color="#777777",
                    alpha=0.16,
                    s=12,
                    zorder=2,
                )

        summary_by_condition = {
            str(row["condition"]): row for row in metric_summaries
        }
        labels: list[str] = []
        for index, condition in enumerate(conditions):
            row = summary_by_condition.get(condition)
            label = _condition_label(payload, condition)
            if row is None:
                labels.append(label)
                continue
            median = _finite(row.get("median"))
            p25 = _finite(row.get("p25"))
            p75 = _finite(row.get("p75"))
            mean = _finite(row.get("mean"))
            color = _condition_color(payload, condition)
            if median is not None and p25 is not None and p75 is not None:
                axis.errorbar(
                    [index],
                    [median * scale],
                    yerr=[
                        [max(0.0, (median - p25) * scale)],
                        [max(0.0, (p75 - median) * scale)],
                    ],
                    color=color,
                    marker="o",
                    markersize=7,
                    linewidth=2.4,
                    capsize=4,
                    zorder=4,
                )
            if mean is not None:
                axis.scatter(
                    [index],
                    [mean * scale],
                    marker="D",
                    s=26,
                    facecolors="white",
                    edgecolors=color,
                    linewidths=1.3,
                    zorder=5,
                )
            labels.append(f"{label}\nn={int(row['finite_recording_count'])}")
        axis.set_xticks(np.arange(len(conditions)), labels)
        axis.set_ylabel(display_unit)
        axis.set_title(_panel_metric_label(metric), fontsize=10)
        _style_axis(axis)
    for axis in axes[len(metric_ids) :]:
        axis.set_visible(False)
    figure.suptitle(
        "Observed distance traveled · whole session and exact chaser epochs",
        fontsize=16,
        y=0.995,
    )
    _footer(figure, payload)
    return figure


def grouped_epoch_figure(payload: Mapping[str, object], *, columns: int = 4) -> Any:
    """Render provider/behavior-role series across the three exact epochs."""

    catalog = _catalog(payload)
    metric_ids = tuple(catalog)
    summaries = _rows(payload, "descriptive_rows")
    conditions = _condition_order(payload)
    figure, axes = _subplot_grid(len(metric_ids), columns=columns)
    for axis, metric_id in zip(axes, metric_ids):
        metric = catalog[metric_id]
        scale, display_unit = _display_scale(str(metric["unit"]))
        rows = [row for row in summaries if row["metric_id"] == metric_id]
        series_keys = sorted(
            {
                (
                    str(row.get("provider_role", "all")),
                    str(row.get("behavior_role", "all")),
                )
                for row in rows
            }
        )
        for provider, role in series_keys:
            selected = {
                str(row["condition"]): row
                for row in rows
                if str(row.get("provider_role", "all")) == provider
                and str(row.get("behavior_role", "all")) == role
            }
            x_values: list[int] = []
            medians: list[float] = []
            lows: list[float] = []
            highs: list[float] = []
            for index, condition in enumerate(conditions):
                row = selected.get(condition)
                if row is None:
                    continue
                median = _finite(row.get("median"))
                p25 = _finite(row.get("p25"))
                p75 = _finite(row.get("p75"))
                if median is None or p25 is None or p75 is None:
                    continue
                x_values.append(index)
                medians.append(median * scale)
                lows.append(p25 * scale)
                highs.append(p75 * scale)
            if not x_values:
                continue
            color = _role_color(payload, role) if role != "all" else "#333333"
            linestyle = _provider_linestyle(payload, provider)
            label_parts = [part.title() for part in (role, provider) if part != "all"]
            label = " · ".join(label_parts) or "Cohort"
            axis.plot(
                x_values,
                medians,
                color=color,
                linestyle=linestyle,
                marker=_role_marker(payload, role),
                linewidth=2.0,
                label=label,
            )
            axis.fill_between(x_values, lows, highs, color=color, alpha=0.08)
        axis.set_xticks(
            np.arange(len(conditions)),
            [_condition_label(payload, value) for value in conditions],
        )
        axis.set_ylabel(display_unit)
        axis.set_title(_panel_metric_label(metric), fontsize=10)
        _style_axis(axis)
    for axis in axes[len(metric_ids) :]:
        axis.set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        figure.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.952),
            ncol=min(4, len(handles)),
        )
    figure.suptitle(
        str(payload.get("label", "Grouped epoch statistics")),
        fontsize=16,
        y=0.99,
    )
    _footer(figure, payload)
    return figure


def _distance_band_label(row: Mapping[str, Any]) -> str | None:
    if row.get("distance_bin_index") is None:
        return None
    start = float(row["distance_bin_start_mm"])
    end = _finite(row.get("distance_bin_end_mm"))
    return f"{start:g}–∞" if end is None else f"{start:g}–{end:g}"


def _curve_x(row: Mapping[str, Any]) -> float:
    # Distance bands are categorical because their terminal interval may be
    # open-ended.  The persisted bin index is the only honest common x-axis.
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


def _display_band_ticks(
    band_labels: Mapping[int, str | None], *, maximum: int = 6
) -> tuple[list[int], list[str]]:
    ticks = sorted(band_labels)
    if len(ticks) > maximum:
        indices = np.linspace(0, len(ticks) - 1, num=maximum, dtype=np.int64)
        ticks = [ticks[int(index)] for index in np.unique(indices)]
    return ticks, [str(band_labels[tick]) for tick in ticks]


def distance_curve_figure(payload: Mapping[str, object]) -> Any:
    catalog = _catalog(payload)
    summaries = _rows(payload, "descriptive_rows")
    conditions = _condition_order(payload)
    roles = sorted({str(row.get("behavior_role", "all")) for row in summaries})
    providers = sorted({str(row.get("provider_role", "all")) for row in summaries})
    facets = [(provider, role) for provider in providers for role in roles]
    metric_ids = tuple(catalog)
    n_columns = max(1, len(facets))
    figure, axes = plt.subplots(
        len(metric_ids),
        n_columns,
        figsize=(4.2 * n_columns, 3.3 * len(metric_ids)),
        squeeze=False,
        constrained_layout=True,
    )
    for row_index, metric_id in enumerate(metric_ids):
        metric = catalog[metric_id]
        scale, display_unit = _display_scale(str(metric["unit"]))
        metric_rows = [row for row in summaries if row["metric_id"] == metric_id]
        for column_index, (provider, role) in enumerate(facets):
            axis = axes[row_index, column_index]
            selected = [
                row
                for row in metric_rows
                if str(row.get("provider_role", "all")) == provider
                and str(row.get("behavior_role", "all")) == role
            ]
            for condition in conditions:
                points = []
                for row in selected:
                    if row["condition"] != condition:
                        continue
                    median = _finite(row.get("median"))
                    p25 = _finite(row.get("p25"))
                    p75 = _finite(row.get("p75"))
                    if median is not None and p25 is not None and p75 is not None:
                        points.append(
                            (_curve_x(row), median * scale, p25 * scale, p75 * scale)
                        )
                points.sort(key=lambda item: item[0])
                if not points:
                    continue
                values = np.asarray(points, dtype=np.float64)
                color = _condition_color(payload, condition)
                axis.plot(
                    values[:, 0],
                    values[:, 1],
                    color=color,
                    linewidth=1.8,
                    label=_condition_label(payload, condition),
                )
                axis.fill_between(
                    values[:, 0], values[:, 2], values[:, 3], color=color, alpha=0.10
                )
            band_labels = {
                int(row["distance_bin_index"]): _distance_band_label(row)
                for row in selected
                if row.get("distance_bin_index") is not None
            }
            if band_labels:
                ticks, tick_labels = _display_band_ticks(band_labels)
                axis.set_xticks(ticks, tick_labels)
            if row_index == 0:
                title = " · ".join(
                    part.title() for part in (role, provider) if part != "all"
                )
                axis.set_title(title or "Cohort", fontsize=10)
            if column_index == 0:
                axis.set_ylabel(f"{_axis_metric_label(metric)}\n({display_unit})")
            if row_index == len(metric_ids) - 1:
                axis.set_xlabel(
                    "Fish–chaser distance band (mm)"
                    if band_labels
                    else "Fish–chaser distance (mm)"
                )
            _style_axis(axis)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        figure.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.952),
            ncol=len(handles),
        )
    figure.suptitle(
        str(payload.get("label", "Distance-conditioned statistics")),
        fontsize=16,
        y=0.99,
    )
    _footer(figure, payload)
    return figure


def trial_response_figure(payload: Mapping[str, object]) -> Any:
    catalog = _catalog(payload)
    summaries = _rows(payload, "descriptive_rows")
    metric_ids = tuple(catalog)
    figure, axes = _subplot_grid(len(metric_ids), columns=3)
    roles = sorted({str(row.get("behavior_role", "all")) for row in summaries})
    for axis, metric_id in zip(axes, metric_ids):
        metric = catalog[metric_id]
        scale, display_unit = _display_scale(str(metric["unit"]))
        metric_rows = [row for row in summaries if row["metric_id"] == metric_id]
        for role in roles:
            points = []
            for row in metric_rows:
                if str(row.get("behavior_role", "all")) != role:
                    continue
                median = _finite(row.get("median"))
                p25 = _finite(row.get("p25"))
                p75 = _finite(row.get("p75"))
                if median is not None and p25 is not None and p75 is not None:
                    points.append(
                        (
                            int(row["trial_ordinal"]),
                            median * scale,
                            p25 * scale,
                            p75 * scale,
                        )
                    )
            points.sort(key=lambda item: item[0])
            if not points:
                continue
            values = np.asarray(points, dtype=np.float64)
            color = _role_color(payload, role)
            axis.plot(
                values[:, 0],
                values[:, 1],
                color=color,
                marker=_role_marker(payload, role),
                linewidth=2.0,
                label=role.title(),
            )
            axis.fill_between(
                values[:, 0], values[:, 2], values[:, 3], color=color, alpha=0.10
            )
        axis.set_xlabel("Trial ordinal")
        axis.set_ylabel(display_unit)
        axis.set_title(_panel_metric_label(metric, width=42), fontsize=10)
        axis.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        _style_axis(axis)
    for axis in axes[len(metric_ids) :]:
        axis.set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        figure.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.952),
            ncol=len(handles),
        )
    figure.suptitle("Trial-locked escape and freeze responses", fontsize=16, y=0.99)
    _footer(figure, payload)
    return figure


def spatial_occupancy_figure(payload: Mapping[str, object]) -> Any:
    catalog = _catalog(payload)
    metric_id = str(payload.get("default_metric_id"))
    if metric_id not in catalog:
        raise ValueError("Spatial occupancy payload lacks its default metric")
    rows = [
        row
        for row in _rows(payload, "descriptive_rows")
        if row["metric_id"] == metric_id
    ]
    conditions = _condition_order(payload)
    providers = sorted({str(row["provider_role"]) for row in rows})
    panels: dict[tuple[str, str], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    finite_values: list[float] = []
    for provider in providers:
        for condition in conditions:
            selected = [
                row
                for row in rows
                if row["provider_role"] == provider and row["condition"] == condition
            ]
            x_indices = sorted({int(row["x_bin_index"]) for row in selected})
            y_indices = sorted({int(row["y_bin_index"]) for row in selected})
            if not x_indices or not y_indices:
                raise ValueError("Spatial occupancy panel has no persisted bins")
            if x_indices != list(range(len(x_indices))) or y_indices != list(
                range(len(y_indices))
            ):
                raise ValueError("Spatial occupancy bin indices are not contiguous")
            grid = np.full((len(y_indices), len(x_indices)), np.nan, dtype=np.float64)
            x_edges = np.full(len(x_indices) + 1, np.nan, dtype=np.float64)
            y_edges = np.full(len(y_indices) + 1, np.nan, dtype=np.float64)
            seen_coordinates: set[tuple[int, int]] = set()
            seen_member_coordinates: set[tuple[int, int]] = set()
            for row in selected:
                x_index = int(row["x_bin_index"])
                y_index = int(row["y_bin_index"])
                coordinate = (x_index, y_index)
                seen_coordinates.add(coordinate)
                edge_updates = (
                    (x_edges, x_index, float(row["x_bin_start_mm"])),
                    (x_edges, x_index + 1, float(row["x_bin_end_mm"])),
                    (y_edges, y_index, float(row["y_bin_start_mm"])),
                    (y_edges, y_index + 1, float(row["y_bin_end_mm"])),
                )
                for edges, edge_index, edge_value in edge_updates:
                    previous = edges[edge_index]
                    if np.isfinite(previous) and previous != edge_value:
                        raise ValueError(
                            "Spatial occupancy view bin edges are inconsistent"
                        )
                    edges[edge_index] = edge_value
                value = _finite(row.get("mean"))
                if bool(row["arena_bin_center_member"]):
                    if coordinate in seen_member_coordinates:
                        raise ValueError(
                            "Spatial occupancy view has a duplicate arena-member bin"
                        )
                    seen_member_coordinates.add(coordinate)
                if bool(row["arena_bin_center_member"]) and value is not None:
                    grid[y_index, x_index] = value * 100.0
                    finite_values.append(value * 100.0)
            if len(seen_coordinates) != len(x_indices) * len(y_indices):
                raise ValueError("Spatial occupancy view grid is incomplete")
            if np.any(~np.isfinite(x_edges)) or np.any(~np.isfinite(y_edges)):
                raise ValueError("Spatial occupancy view edges are incomplete")
            panels[(provider, condition)] = (x_edges, y_edges, grid)

    positive = np.asarray(
        [value for value in finite_values if value > 0], dtype=np.float64
    )
    vmax = (
        float(np.quantile(positive, ROBUST_OCCUPANCY_QUANTILE))
        if positive.size
        else 1.0
    )
    vmax = max(vmax, np.finfo(np.float64).eps)
    figure, axes = plt.subplots(
        len(providers),
        len(conditions),
        figsize=(4.4 * len(conditions), 4.0 * len(providers)),
        squeeze=False,
        constrained_layout=True,
    )
    image = None
    for row_index, provider in enumerate(providers):
        for column_index, condition in enumerate(conditions):
            axis = axes[row_index, column_index]
            x_edges, y_edges, grid = panels[(provider, condition)]
            image = axis.pcolormesh(
                x_edges,
                y_edges,
                grid,
                cmap="viridis",
                vmin=0.0,
                vmax=vmax,
                shading="flat",
            )
            axis.set_aspect("equal")
            axis.invert_yaxis()
            axis.set_title(
                f"{provider.title()} · {_condition_label(payload, condition)}",
                fontsize=11,
            )
            axis.set_xlabel("x (mm)")
            axis.set_ylabel("y (mm; +down)")
    if image is not None:
        figure.colorbar(
            image,
            ax=axes.ravel().tolist(),
            label=(
                "Mean occupancy (% valid in-arena rows/bin; " "arena-member recordings)"
            ),
            shrink=0.88,
        )
    figure.suptitle(
        f"Cohort spatial occupancy (shared q{ROBUST_OCCUPANCY_QUANTILE:.2f} scale)",
        fontsize=16,
        y=0.99,
    )
    _footer(figure, payload)
    return figure


def _body_bearing_polar_axis(axis: Any) -> None:
    axis.set_theta_zero_location("N")
    axis.set_theta_direction(1)
    axis.set_thetagrids(
        (-180.0, -90.0, 0.0, 90.0, 180.0),
        ("Behind", "Right", "Front", "Left", "Behind"),
        fontsize=8,
    )
    # Matplotlib's default polar domain is [0, 2*pi].  Adding signed negative
    # bearings can otherwise autoscale it to [-pi, 2*pi], a 540-degree domain
    # that renders as a misleading half-circle.  The persisted contract is one
    # signed circle in [-180, 180], so make that display boundary explicit.
    axis.set_thetalim(-np.pi, np.pi)
    axis.grid(alpha=0.22, linewidth=0.7)


def _bearing_histogram_panels(
    payload: Mapping[str, object],
) -> tuple[
    tuple[str, ...],
    tuple[str, ...],
    dict[tuple[str, str], list[Mapping[str, Any]]],
]:
    rows = _rows(payload, "descriptive_rows")
    conditions = tuple(
        condition
        for condition in _condition_order(payload)
        if any(str(row["condition"]) == condition for row in rows)
    )
    roles = tuple(sorted({str(row["behavior_role"]) for row in rows}))
    if not conditions or not roles:
        raise ValueError("Body-bearing histogram payload has no exact panels")
    panels = {
        (condition, role): [
            row
            for row in rows
            if str(row["condition"]) == condition and str(row["behavior_role"]) == role
        ]
        for condition in conditions
        for role in roles
    }
    if any(not values for values in panels.values()):
        raise ValueError("Body-bearing histogram panel registry is incomplete")
    return conditions, roles, panels


def body_bearing_polar_figure(payload: Mapping[str, object]) -> Any:
    """Render equal-recording signed anatomical-bearing distributions."""

    conditions, roles, panels = _bearing_histogram_panels(payload)
    figure, axes = plt.subplots(
        len(conditions),
        len(roles),
        figsize=(5.0 * len(roles), 4.3 * len(conditions)),
        subplot_kw={"projection": "polar"},
        squeeze=False,
        constrained_layout=True,
    )
    global_max = 0.0
    prepared: dict[tuple[str, str], tuple[np.ndarray, np.ndarray, np.ndarray, int]] = {}
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
            [float(row["axis_0_bin_start"]) for row in ordered],
            dtype=np.float64,
        )
        ends = np.asarray(
            [float(row["axis_0_bin_end"]) for row in ordered],
            dtype=np.float64,
        )
        if (
            not np.isclose(starts[0], -180.0)
            or not np.isclose(ends[-1], 180.0)
            or np.any(~np.isclose(ends[:-1], starts[1:]))
        ):
            raise ValueError("Signed-bearing polar edges do not cover one circle")
        fractions = np.asarray(
            [float(row["mean_fraction"]) for row in ordered],
            dtype=np.float64,
        )
        if np.any(~np.isfinite(fractions) | (fractions < 0.0)) or not np.isclose(
            np.sum(fractions), 1.0, atol=1e-8
        ):
            raise ValueError(
                "Equal-recording signed-bearing fractions do not sum to one"
            )
        support = {int(row["finite_recording_count"]) for row in ordered}
        if len(support) != 1:
            raise ValueError("Signed-bearing panel support changes across bins")
        percentages = fractions * 100.0
        global_max = max(global_max, float(np.max(percentages)))
        prepared[key] = (starts, ends, percentages, support.pop())

    for row_index, condition in enumerate(conditions):
        for column_index, role in enumerate(roles):
            axis = axes[row_index, column_index]
            starts, ends, percentages, support = prepared[(condition, role)]
            centers = np.deg2rad((starts + ends) / 2.0)
            widths = np.deg2rad(ends - starts)
            axis.bar(
                centers,
                percentages,
                width=widths,
                bottom=0.0,
                color=_role_color(payload, role),
                alpha=0.82,
                linewidth=0.25,
                edgecolor="white",
            )
            axis.set_ylim(0.0, max(global_max * 1.08, 1.0))
            axis.set_title(
                f"{_condition_label(payload, condition)} · {role.title()} · n={support}",
                fontsize=10,
                pad=18,
            )
            _body_bearing_polar_axis(axis)
    figure.suptitle(
        (
            "Signed anatomical bearing to the chaser\n"
            "Radius: mean recording fraction per 10° bin (%)"
        ),
        fontsize=16,
        y=0.995,
    )
    _footer(figure, payload)
    return figure


def body_bearing_distance_figure(payload: Mapping[str, object]) -> Any:
    """Render equal-recording joint bearing-by-distance polar densities."""

    conditions, roles, panels = _bearing_histogram_panels(payload)
    prepared: dict[tuple[str, str], tuple[np.ndarray, np.ndarray, np.ndarray, int]] = {}
    positive: list[float] = []
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
        grid = np.full(
            (len(distance_indices), len(bearing_indices)),
            np.nan,
            dtype=np.float64,
        )
        bearing_edges = np.full(len(bearing_indices) + 1, np.nan, dtype=np.float64)
        distance_edges = np.full(len(distance_indices) + 1, np.nan, dtype=np.float64)
        supports: set[int] = set()
        for row in rows:
            bearing_index = int(row["axis_0_bin_index"])
            distance_index = int(row["axis_1_bin_index"])
            value = _finite(row.get("mean_fraction"))
            if value is None or value < 0.0:
                raise ValueError("Bearing-by-distance cohort fraction is invalid")
            grid[distance_index, bearing_index] = value * 100.0
            supports.add(int(row["finite_recording_count"]))
            for edges, index, start, end in (
                (
                    bearing_edges,
                    bearing_index,
                    float(row["axis_0_bin_start"]),
                    float(row["axis_0_bin_end"]),
                ),
                (
                    distance_edges,
                    distance_index,
                    float(row["axis_1_bin_start"]),
                    float(row["axis_1_bin_end"]),
                ),
            ):
                if np.isfinite(edges[index]) and not np.isclose(edges[index], start):
                    raise ValueError("Bearing-by-distance bin starts disagree")
                if np.isfinite(edges[index + 1]) and not np.isclose(
                    edges[index + 1], end
                ):
                    raise ValueError("Bearing-by-distance bin ends disagree")
                edges[index] = start
                edges[index + 1] = end
        if np.any(~np.isfinite(grid)) or not np.isclose(np.sum(grid), 100.0, atol=1e-6):
            raise ValueError("Bearing-by-distance panel is incomplete or unnormalized")
        if len(supports) != 1:
            raise ValueError("Bearing-by-distance support changes across bins")
        positive.extend(grid[grid > 0.0].tolist())
        prepared[key] = (bearing_edges, distance_edges, grid, supports.pop())

    vmax = (
        float(np.quantile(np.asarray(positive), ROBUST_HISTOGRAM_QUANTILE))
        if positive
        else 1.0
    )
    vmax = max(vmax, float(np.finfo(np.float64).eps))
    figure, axes = plt.subplots(
        len(conditions),
        len(roles),
        figsize=(5.0 * len(roles), 4.5 * len(conditions)),
        subplot_kw={"projection": "polar"},
        squeeze=False,
        constrained_layout=True,
    )
    image = None
    for row_index, condition in enumerate(conditions):
        for column_index, role in enumerate(roles):
            axis = axes[row_index, column_index]
            bearing_edges, distance_edges, grid, support = prepared[(condition, role)]
            image = axis.pcolormesh(
                np.deg2rad(bearing_edges),
                distance_edges,
                grid,
                cmap="viridis",
                vmin=0.0,
                vmax=vmax,
                shading="flat",
            )
            axis.set_ylim(0.0, float(distance_edges[-1]))
            axis.set_title(
                f"{_condition_label(payload, condition)} · {role.title()} · n={support}",
                fontsize=10,
                pad=18,
            )
            _body_bearing_polar_axis(axis)
    if image is not None:
        figure.colorbar(
            image,
            ax=axes.ravel().tolist(),
            label="Mean recording fraction per bearing × distance bin (%)",
            shrink=0.86,
        )
    figure.suptitle(
        (
            "Signed anatomical bearing × fish–chaser distance\n"
            "Radius: distance (mm) · "
            f"shared q{ROBUST_HISTOGRAM_QUANTILE:.2f} color scale"
        ),
        fontsize=16,
        y=0.995,
    )
    _footer(figure, payload)
    return figure


def render_statistics_view(payload: Mapping[str, object]) -> Any:
    validate_statistics_view_payload(payload)
    view_id = str(payload.get("view_id"))
    if view_id == "core_behavior":
        return core_behavior_figure(payload)
    if view_id == "distance_traveled":
        return distance_traveled_figure(payload)
    if view_id in {"near_field", "same_quadrant", "occupancy_support"}:
        return grouped_epoch_figure(payload)
    if view_id in {
        "bout_response_by_distance",
        "body_alignment_by_distance",
        "radial_distribution",
        "distance_cdf",
    }:
        return distance_curve_figure(payload)
    if view_id == "trial_response":
        return trial_response_figure(payload)
    if view_id == "spatial_occupancy":
        return spatial_occupancy_figure(payload)
    if view_id == "body_bearing_polar":
        return body_bearing_polar_figure(payload)
    if view_id == "body_bearing_distance":
        return body_bearing_distance_figure(payload)
    raise KeyError(f"No static renderer is registered for statistics view {view_id!r}")


__all__ = [
    "PLOT_DPI",
    "ROBUST_HISTOGRAM_QUANTILE",
    "ROBUST_OCCUPANCY_QUANTILE",
    "body_bearing_distance_figure",
    "body_bearing_polar_figure",
    "core_behavior_figure",
    "distance_traveled_figure",
    "distance_curve_figure",
    "grouped_epoch_figure",
    "render_statistics_view",
    "spatial_occupancy_figure",
    "trial_response_figure",
]
