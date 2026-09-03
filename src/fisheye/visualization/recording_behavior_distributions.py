"""Static rendering for sealed recording-level behavior distributions."""

from __future__ import annotations

from collections import defaultdict
import math
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from fisheye.group_statistics.recording_behavior_distribution_views import (
    RecordingBehaviorDistributionView,
    RecordingDistributionSeries,
)


PLOT_DPI = 170
_COLORS = (
    "#4C78A8",
    "#F58518",
    "#54A24B",
    "#E45756",
    "#72B7B2",
    "#B279A2",
    "#FF9DA6",
    "#9D755D",
    "#BAB0AC",
)
_LINE_STYLES = ("-", "--", ":", "-.")


def _style_roster(
    view: RecordingBehaviorDistributionView,
) -> dict[tuple[str, str], tuple[str, str]]:
    identities = sorted(
        {
            (series.group_key_sha256, series.source_identity_key_sha256)
            for series in view.series
        }
    )
    return {
        identity: (
            _COLORS[index % len(_COLORS)],
            _LINE_STYLES[(index // len(_COLORS)) % len(_LINE_STYLES)],
        )
        for index, identity in enumerate(identities)
    }


def _edges(series: RecordingDistributionSeries) -> np.ndarray:
    if not series.bin_left.size:
        return np.asarray([], dtype=np.float64)
    return np.concatenate((series.bin_left[:1], series.bin_right))


def render_recording_behavior_distribution_figure(
    view: RecordingBehaviorDistributionView,
    *,
    probability_percent: bool = True,
    maximum_columns: int = 4,
) -> Any:
    """Render exact persisted bins without viewer-side rebinning or clipping."""

    if type(view) is not RecordingBehaviorDistributionView:
        raise TypeError("view must be one RecordingBehaviorDistributionView")
    if maximum_columns < 1:
        raise ValueError("maximum_columns must be positive")
    scopes = tuple(sorted(view.scopes, key=lambda row: int(row["order"])))
    if not scopes:
        raise ValueError("view has no selected scopes")
    columns = min(maximum_columns, len(scopes))
    rows = math.ceil(len(scopes) / columns)
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(4.6 * columns, 3.9 * rows),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    axes_flat = axes.reshape(-1)
    styles = _style_roster(view)
    by_scope: dict[str, list[RecordingDistributionSeries]] = defaultdict(list)
    for series in view.series:
        by_scope[series.scope_id].append(series)

    for axis, scope in zip(axes_flat, scopes, strict=False):
        scope_series = by_scope.get(str(scope["scope_id"]), [])
        finite_evidence = False
        for series in scope_series:
            color, linestyle = styles[
                (series.group_key_sha256, series.source_identity_key_sha256)
            ]
            values = series.fraction * (100.0 if probability_percent else 1.0)
            edges = _edges(series)
            if edges.size and np.any(np.isfinite(values)):
                finite_evidence = True
                axis.stairs(
                    values,
                    edges,
                    color=color,
                    linestyle=linestyle,
                    linewidth=1.8,
                    label=(
                        f"{series.label} "
                        f"(n={int(series.support['valid_count']):,})"
                    ),
                )
        if not finite_evidence:
            axis.text(
                0.5,
                0.5,
                "No valid evidence",
                transform=axis.transAxes,
                ha="center",
                va="center",
                color="#666666",
            )
        axis.set_title(str(scope["scope_label"]))
        axis.set_xlabel(f"{view.metric['interpretation']} ({view.metric['unit']})")
        axis.set_ylabel("Probability per bin (%)" if probability_percent else "Fraction")
        axis.grid(alpha=0.2)
        if view.metric.get("axis_scale") == "log10":
            axis.set_xscale("log")
        handles, labels = axis.get_legend_handles_labels()
        if handles:
            axis.legend(handles, labels, fontsize=8)
    for axis in axes_flat[len(scopes) :]:
        axis.set_visible(False)

    figure.suptitle(
        f"{view.metric['interpretation']} · {view.weighting_id} weighted",
        fontsize=14,
    )
    figure.text(
        0.5,
        0.012,
        (
            f"Recording {view.recording_id} · persisted bins · no rebinning/clipping · "
            f"run {view.distribution_run_id} · view {view.view_sha256[:12]}"
        ),
        ha="center",
        va="bottom",
        fontsize=8,
        color="#555555",
    )
    figure.tight_layout(rect=(0.0, 0.045, 1.0, 0.94))
    return figure


__all__ = ["PLOT_DPI", "render_recording_behavior_distribution_figure"]
