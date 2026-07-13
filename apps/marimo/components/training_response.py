"""Pure controls metadata and Plotly helpers for training-response QC."""

from __future__ import annotations

from collections import defaultdict
import math
from typing import Any, Iterable, Mapping, Sequence

import plotly.graph_objects as go


TRAINING_RESPONSE_METRICS = {
    "locomotor_response_score": "Locomotor response score",
    "boundary_response_score": "Boundary response score",
    "aggressive_proximity_score": "Aggressive proximity score",
    "role_distance_selectivity_score": "Role-distance selectivity score",
    "close_contact_vigor_score": "Close-contact vigor score",
    "mean_speed_mm_s_log2_ratio": "Training/pre mean-speed log2 ratio",
    "wall_fraction_delta": "Training minus pre wall fraction",
    "aggressive_training_p50_distance_mm": "Aggressive training median distance (mm)",
    "aggressive_training_fraction_within_threshold": (
        "Aggressive training fraction within threshold"
    ),
    "training_role_p50_distance_contrast_mm": (
        "Aggressive minus inert median distance (mm)"
    ),
    "aggressive_near_minus_far_speed_mm_s": (
        "Near minus far speed for aggressive chaser (mm/s)"
    ),
    "training_tracking_dropout_fraction": "Training tracking dropout fraction",
}

TRAINING_RESPONSE_CATEGORY_FIELDS = {
    "primary_training_profile": "Primary training profile",
    "locomotor_response": "Locomotor response",
    "boundary_response": "Boundary response",
    "aggressive_proximity_state": "Aggressive proximity state",
    "role_distance_selectivity": "Role-distance selectivity",
    "close_contact_vigor": "Close-contact vigor",
    "cluster_status": "Unsupervised cluster model status",
    "cluster_id": "Unsupervised cluster ID (inspect model status)",
}


def _finite(value: object) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def filter_training_response_rows(
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


def training_response_scatter_figure(
    rows: Sequence[Mapping[str, Any]],
) -> go.Figure | None:
    """Plot locomotor change against aggressive proximity without causal labels."""

    grouped: dict[str, list[tuple[float, float, str, str]]] = defaultdict(list)
    for row in rows:
        x = _finite(row.get("aggressive_proximity_score"))
        y = _finite(row.get("locomotor_response_score"))
        if x is None or y is None:
            continue
        grouped[str(row.get("primary_training_profile") or "unavailable")].append(
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
    for profile in sorted(grouped):
        values = grouped[profile]
        figure.add_trace(
            go.Scattergl(
                name=profile,
                mode="markers",
                x=[value[0] for value in values],
                y=[value[1] for value in values],
                text=[f"{value[2]}<br>{value[3]}" for value in values],
                marker={"size": 9, "opacity": 0.8},
                hovertemplate=(
                    "%{text}<br>proximity score=%{x:.3f}"
                    "<br>locomotor score=%{y:.3f}"
                    "<extra>%{fullData.name}</extra>"
                ),
            )
        )
    figure.add_vline(x=0, line_width=1, line_dash="dot", line_color="#888")
    figure.add_hline(y=0, line_width=1, line_dash="dot", line_color="#888")
    figure.update_layout(
        title="Whole-training locomotor response versus aggressive proximity",
        xaxis_title="Aggressive proximity score (farther →)",
        yaxis_title="Locomotor response score (activated →)",
        legend_title="Descriptive profile",
        margin=dict(l=65, r=20, t=55, b=55),
    )
    return figure


__all__ = [
    "TRAINING_RESPONSE_CATEGORY_FIELDS",
    "TRAINING_RESPONSE_METRICS",
    "filter_training_response_rows",
    "training_response_scatter_figure",
]
