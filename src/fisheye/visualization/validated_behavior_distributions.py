"""Static figures for shared validated-behavior distribution payloads."""

from __future__ import annotations

from collections import defaultdict
import json
import math
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from fisheye.group_statistics.validated_behavior_distribution_views import (
    COHORT_STATISTIC_LABELS,
    DEFAULT_COHORT_STATISTIC,
    DEFAULT_DISPLAY_RANGE,
    FULL_EVIDENCE_RANGE,
    resolve_distribution_display_range,
    validate_distribution_view_payload,
    validate_motion_trace_payload,
)

PLOT_DPI = 170


def _mapping_rows(payload: Mapping[str, object], key: str) -> list[Mapping[str, Any]]:
    raw = payload.get(key)
    if not isinstance(raw, list):
        raise ValueError(f"Distribution payload lacks {key}")
    return [row for row in raw if isinstance(row, Mapping)]


def _role_style(payload: Mapping[str, object], role: str) -> Mapping[str, Any]:
    styles = payload.get("behavior_role_styles")
    if isinstance(styles, Mapping) and isinstance(styles.get(role), Mapping):
        return styles[role]  # type: ignore[index,return-value]
    return {}


def _series_style(
    payload: Mapping[str, object], group: Mapping[str, Any], scope_id: str
) -> tuple[str, str, str, str | None, str]:
    provider = str(group.get("provider_role", ""))
    role = str(group.get("behavior_role", ""))
    if role:
        role_style = _role_style(payload, role)
        color = str(role_style.get("aggregate_color_hex", "#555555"))
        marker: str | None = str(
            role_style.get("matplotlib_role_marker") or "o"
        )
    elif provider:
        colors = payload.get("provider_colors", {})
        color = (
            str(colors.get(provider, "#555555"))
            if isinstance(colors, Mapping)
            else "#555555"
        )
        marker = None
    else:
        colors = payload.get("scope_colors", {})
        color = (
            str(colors.get(scope_id, "#4C78A8"))
            if isinstance(colors, Mapping)
            else "#4C78A8"
        )
        marker = None
    styles = payload.get("provider_line_styles", {})
    named = (
        str(styles.get(provider, "solid"))
        if provider and isinstance(styles, Mapping)
        else "solid"
    )
    linestyle = {"solid": "-", "dashed": "--", "dotted": ":"}.get(named, "-")
    hatch = {"solid": "", "dashed": "//", "dotted": ".."}.get(named, "")
    parts = []
    if provider:
        parts.append(provider.title())
    if role:
        parts.append(role.title())
    return color, linestyle, hatch, marker, " · ".join(parts) or "All observations"


def _selected_rows(
    payload: Mapping[str, object],
    *,
    provider_role: str | None,
    behavior_role: str | None,
) -> list[Mapping[str, Any]]:
    rows = _mapping_rows(payload, "cohort_rows")
    return [
        row
        for row in rows
        if (provider_role is None or row.get("provider_role") == provider_role)
        and (behavior_role is None or row.get("behavior_role") == behavior_role)
    ]


def _footer(
    figure: Any,
    payload: Mapping[str, object],
    statistic: str,
    display_range: Mapping[str, object],
) -> None:
    source = payload.get("source_distribution")
    digest = (
        str(source.get("distribution_manifest_sha256", "unknown"))[:12]
        if isinstance(source, Mapping)
        else "unknown"
    )
    figure.text(
        0.5,
        0.022,
        (
            "Exploratory · recording_id unit · "
            f"{COHORT_STATISTIC_LABELS[statistic]} · "
            + (
                "full evidence x-range"
                if display_range["effective_display_range_id"] == FULL_EVIDENCE_RANGE
                else (
                    "central x ≥"
                    f"{100.0 * float(display_range['minimum_series_fraction_retained']):.2f}% "
                    "per series; tails sealed"
                )
            )
            + f" · distribution {digest}"
        ),
        ha="center",
        va="bottom",
        fontsize=8,
        color="#555555",
    )


def render_distribution_figure(
    payload: Mapping[str, object],
    *,
    cohort_statistic: str = DEFAULT_COHORT_STATISTIC,
    provider_role: str | None = None,
    behavior_role: str | None = None,
    show_recording_iqr: bool = True,
    display_range_id: str = DEFAULT_DISPLAY_RANGE,
) -> Any:
    """Render aligned whole/pre/training/post histograms from one shared payload."""

    validate_distribution_view_payload(payload)
    if cohort_statistic not in COHORT_STATISTIC_LABELS:
        raise ValueError(f"Unknown cohort statistic: {cohort_statistic}")
    metric = payload["metric"]
    recipe = payload["histogram_recipe"]
    assert isinstance(metric, Mapping) and isinstance(recipe, Mapping)
    rows = _selected_rows(
        payload,
        provider_role=provider_role,
        behavior_role=behavior_role,
    )
    if not rows:
        raise ValueError("No distribution series matches the selected dimensions")
    display_range = resolve_distribution_display_range(
        payload,
        display_range_id=display_range_id,
        provider_role=provider_role,
        behavior_role=behavior_role,
    )
    by_scope_group: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_scope_group[(str(row["scope_id"]), str(row["group_key_sha256"]))].append(row)
    group_count = len({str(row["group_key_sha256"]) for row in rows})
    bar_alpha = 0.78 if group_count == 1 else 0.42

    scopes = tuple(str(value) for value in payload["scope_order"])
    figure, axes = plt.subplots(
        1,
        len(scopes),
        figsize=(5.0 * len(scopes), 4.8),
        sharex=True,
        sharey=True,
        constrained_layout=False,
    )
    axes_array = np.asarray(axes, dtype=object).reshape(-1)
    for axis, scope_id in zip(axes_array, scopes, strict=True):
        support_values: list[int] = []
        for (row_scope, _group_digest), series in sorted(by_scope_group.items()):
            if row_scope != scope_id:
                continue
            ordered = sorted(series, key=lambda row: int(row["bin_index"]))
            group = json.loads(str(ordered[0]["group_key_json"]))
            color, linestyle, hatch, marker, label = _series_style(
                payload, group, scope_id
            )
            left = np.asarray(
                [float(row["bin_left"]) for row in ordered], dtype=np.float64
            )
            right = np.asarray(
                [float(row["bin_right"]) for row in ordered], dtype=np.float64
            )
            values = np.asarray(
                [
                    (
                        np.nan
                        if row.get(cohort_statistic) is None
                        else 100.0 * float(row[cohort_statistic])
                    )
                    for row in ordered
                ],
                dtype=np.float64,
            )
            axis.bar(
                left,
                values,
                width=right - left,
                align="edge",
                color=color,
                edgecolor=color,
                linewidth=0.9,
                linestyle=linestyle,
                alpha=bar_alpha,
                hatch=hatch,
                label="_nolegend_" if marker is not None else label,
            )
            centers = np.asarray([float(row["bin_center"]) for row in ordered])
            finite = np.flatnonzero(np.isfinite(values))
            if marker is not None and finite.size:
                marker_count = min(12, finite.size)
                marker_indices = finite[
                    np.unique(
                        np.linspace(0, finite.size - 1, marker_count, dtype=np.int64)
                    )
                ]
                axis.plot(
                    centers[marker_indices],
                    values[marker_indices],
                    linestyle="none",
                    marker=marker,
                    markersize=4.5,
                    markerfacecolor=color,
                    markeredgecolor="#ffffff",
                    markeredgewidth=0.6,
                    color=color,
                )
                axis.plot(
                    [],
                    [],
                    color=color,
                    linestyle=linestyle,
                    marker=marker,
                    markerfacecolor=color,
                    markeredgecolor="#ffffff",
                    markeredgewidth=0.6,
                    markersize=5,
                    linewidth=1.8,
                    label=label,
                )
            if show_recording_iqr and cohort_statistic != "pooled_fraction":
                lower = np.asarray(
                    [
                        (
                            np.nan
                            if row.get("p25_recording_fraction") is None
                            else 100.0 * float(row["p25_recording_fraction"])
                        )
                        for row in ordered
                    ]
                )
                upper = np.asarray(
                    [
                        (
                            np.nan
                            if row.get("p75_recording_fraction") is None
                            else 100.0 * float(row["p75_recording_fraction"])
                        )
                        for row in ordered
                    ]
                )
                axis.fill_between(
                    centers,
                    lower,
                    upper,
                    step="mid",
                    color=color,
                    alpha=0.10,
                    linewidth=0,
                )
            support_values.append(int(ordered[0]["finite_recording_count"]))
        labels = payload.get("scope_labels", {})
        label = (
            str(labels.get(scope_id, scope_id))
            if isinstance(labels, Mapping)
            else scope_id
        )
        n_text = (
            f"n={min(support_values)}–{max(support_values)} recordings"
            if support_values and min(support_values) != max(support_values)
            else f"n={support_values[0]} recordings" if support_values else "no support"
        )
        axis.set_title(f"{label}\n{n_text}", fontsize=11)
        axis.grid(axis="y", alpha=0.2, linewidth=0.7)
        axis.set_axisbelow(True)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        if recipe.get("axis_scale") == "log10":
            axis.set_xscale("log")
        axis.set_xlim(
            float(display_range["display_lower_bound"]),
            float(display_range["display_upper_bound"]),
        )
    axes_array[0].set_ylabel(f"{COHORT_STATISTIC_LABELS[cohort_statistic]} (%)")
    handles, legend_labels = axes_array[-1].get_legend_handles_labels()
    unique_handles: list[Any] = []
    unique_labels: list[str] = []
    for handle, label in zip(handles, legend_labels, strict=True):
        if label not in unique_labels:
            unique_handles.append(handle)
            unique_labels.append(label)
    show_legend = len(unique_labels) > 1
    if show_legend:
        figure.legend(
            unique_handles,
            unique_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.885),
            ncols=min(4, len(unique_handles)),
            frameon=False,
        )
    warning = " · pooled diagnostic" if cohort_statistic == "pooled_fraction" else ""
    range_note = (
        ""
        if display_range["effective_display_range_id"] == FULL_EVIDENCE_RANGE
        else " · Central ≥99% x-view"
    )
    figure.suptitle(
        f"{metric['interpretation']} · {str(payload['weighting_id']).title()} "
        f"weighted{warning}{range_note}",
        fontsize=15,
        y=0.98,
    )
    figure.supxlabel(
        f"{metric['interpretation']} ({metric['unit']})", y=0.085, fontsize=10
    )
    figure.subplots_adjust(
        left=0.055,
        right=0.99,
        bottom=0.19,
        top=0.76 if show_legend else 0.84,
        wspace=0.08,
    )
    _footer(figure, payload, cohort_statistic, display_range)
    return figure


def render_motion_trace_figure(payload: Mapping[str, object]) -> Any:
    """Render the optional exact-row motion trace using frame or time x."""

    validate_motion_trace_payload(payload)
    metric = payload["metric"]
    assert isinstance(metric, Mapping)
    points = _mapping_rows(payload, "points")
    x = [point["coordinate"] for point in points]
    y = [point["value"] for point in points]
    figure, axis = plt.subplots(figsize=(13, 4.2), constrained_layout=True)
    axis.plot(x, y, color="#4C78A8", linewidth=0.8)
    axis.set_xlabel(
        "Acquisition frame ID" if payload["coordinate_id"] == "frame" else "Time (s)"
    )
    axis.set_ylabel(f"{metric['interpretation']} ({metric['unit']})")
    axis.set_title(
        f"{payload['recording_id']} · {payload['provider_role']} · "
        f"{payload['display_point_count']:,}/{payload['source_row_count']:,} display rows"
    )
    axis.grid(alpha=0.2, linewidth=0.7)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    return figure


__all__ = [
    "PLOT_DPI",
    "render_distribution_figure",
    "render_motion_trace_figure",
]
