"""Pure controls metadata and Plotly helpers for training-response QC."""

from __future__ import annotations

from collections import Counter, defaultdict
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


def strategy_transition_sankey_figure(
    baseline_rows: Sequence[Mapping[str, Any]],
    training_rows: Sequence[Mapping[str, Any]],
    *,
    baseline_category_key: str = "primary_strategy",
    training_category_key: str = "primary_training_profile",
) -> go.Figure | None:
    """Link complete baseline strategies to complete whole-training profiles.

    Each link counts recording-level focal-fish sessions. The diagram is a
    descriptive correspondence, not evidence that one state caused another.
    """

    baseline_by_recording: dict[str, Mapping[str, Any]] = {}
    for row in baseline_rows:
        recording_id = str(row.get("recording_id") or "")
        if not recording_id or row.get("classification_status") != "complete":
            continue
        if recording_id in baseline_by_recording:
            raise ValueError(f"duplicate complete baseline row for {recording_id!r}")
        baseline_by_recording[recording_id] = row
    pair_recordings: dict[tuple[str, str], list[str]] = defaultdict(list)
    pair_protocols: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    seen_training_recordings: set[str] = set()
    for training in training_rows:
        recording_id = str(training.get("recording_id") or "")
        if not recording_id or training.get("classification_status") != "complete":
            continue
        if recording_id in seen_training_recordings:
            raise ValueError(f"duplicate complete training row for {recording_id!r}")
        seen_training_recordings.add(recording_id)
        baseline = baseline_by_recording.get(recording_id)
        if baseline is None:
            continue
        baseline_category = str(baseline.get(baseline_category_key) or "").strip()
        training_category = str(training.get(training_category_key) or "").strip()
        if not baseline_category or not training_category:
            continue
        pair = (baseline_category, training_category)
        pair_recordings[pair].append(recording_id)
        pair_protocols[pair][str(training.get("protocol_name") or "unknown")] += 1
    if not pair_recordings:
        return None

    baseline_categories = sorted({pair[0] for pair in pair_recordings})
    training_categories = sorted({pair[1] for pair in pair_recordings})
    node_labels = [
        *(f"Baseline · {category}" for category in baseline_categories),
        *(f"Training · {category}" for category in training_categories),
    ]
    baseline_indexes = {
        category: index for index, category in enumerate(baseline_categories)
    }
    training_indexes = {
        category: len(baseline_categories) + index
        for index, category in enumerate(training_categories)
    }
    colors = (
        "#4E79A7",
        "#F28E2B",
        "#E15759",
        "#76B7B2",
        "#59A14F",
        "#EDC948",
        "#B07AA1",
        "#FF9DA7",
        "#9C755F",
        "#BAB0AC",
    )
    node_colors = [colors[index % len(colors)] for index in range(len(node_labels))]
    links = sorted(pair_recordings)
    link_details = []
    for pair in links:
        protocol_counts = pair_protocols[pair]
        protocol_text = ", ".join(
            f"{protocol}: {count}"
            for protocol, count in sorted(protocol_counts.items())
        )
        link_details.append(protocol_text)
    figure = go.Figure(
        go.Sankey(
            arrangement="snap",
            node={
                "label": node_labels,
                "color": node_colors,
                "pad": 18,
                "thickness": 20,
                "line": {"color": "rgba(45,45,45,0.35)", "width": 0.5},
            },
            link={
                "source": [baseline_indexes[pair[0]] for pair in links],
                "target": [training_indexes[pair[1]] for pair in links],
                "value": [len(pair_recordings[pair]) for pair in links],
                "customdata": link_details,
                "color": "rgba(100, 116, 139, 0.30)",
                "hovertemplate": (
                    "%{source.label} → %{target.label}<br>"
                    "%{value} focal-fish session(s)<br>%{customdata}<extra></extra>"
                ),
            },
        )
    )
    included_count = sum(len(recordings) for recordings in pair_recordings.values())
    figure.update_layout(
        title=(
            "Baseline strategy → whole-training response profile "
            f"({included_count} matched focal-fish sessions)"
        ),
        font={"size": 12},
        height=max(520, 38 * max(len(baseline_categories), len(training_categories))),
        margin=dict(l=20, r=20, t=65, b=25),
    )
    return figure


__all__ = [
    "TRAINING_RESPONSE_CATEGORY_FIELDS",
    "TRAINING_RESPONSE_METRICS",
    "filter_training_response_rows",
    "strategy_transition_sankey_figure",
    "training_response_scatter_figure",
]
