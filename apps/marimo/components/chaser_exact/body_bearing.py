"""Accepted body-axis chaser-bearing polar view for exact successors."""

from __future__ import annotations

from typing import Any

import numpy as np

from .projection import ExactChaserSuccessorProjection, identity_registry
from .provenance import plain

BEARING_BIN_WIDTH_DEG = 10.0


def _bearing_histogram(
    bearing_deg: np.ndarray,
    valid: np.ndarray,
    *,
    bin_width_deg: float = BEARING_BIN_WIDTH_DEG,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return fixed whole-circle bin centers, probability, and counts."""

    if 360.0 % float(bin_width_deg) != 0.0:
        raise ValueError("Bearing bin width must divide 360 degrees exactly.")
    values = np.asarray(bearing_deg, dtype=np.float64).reshape(-1)
    observed = np.asarray(valid, dtype=bool).reshape(-1)
    if values.size != observed.size:
        raise ValueError("Bearing values and validity have inconsistent lengths.")
    if np.any(observed & (~np.isfinite(values) | (values < -180) | (values > 180))):
        raise ValueError("Valid body-bearing values must be finite in [-180, 180].")
    edges = np.arange(-180.0, 180.0 + bin_width_deg, bin_width_deg)
    counts = np.histogram(values[observed], bins=edges)[0].astype(np.int64)
    total = int(np.sum(counts))
    probability = (
        counts.astype(np.float64) / total
        if total
        else np.zeros(counts.size, dtype=np.float64)
    )
    centers = (edges[:-1] + edges[1:]) / 2.0
    return centers, probability, counts


def build_exact_body_bearing_output(
    mo: Any, go: Any, projection: ExactChaserSuccessorProjection
) -> Any:
    """Render keypoint-authority anatomical bearing without position substitution."""

    if projection.relatives is None:
        raise ValueError(
            "Body-bearing polar views require the keypoint relative frame."
        )
    keypoint = projection.relatives[0]
    if set(keypoint.body_arrays) != {"body_bearing_deg", "body_bearing_valid"}:
        raise ValueError(
            "Keypoint relative frame lacks the sealed body-bearing arrays."
        )
    bearing = keypoint.body_frame_chaser("body_bearing_deg")
    body_valid = keypoint.body_frame_chaser("body_bearing_valid").astype(bool)
    occurrence = keypoint.frame_chaser("chaser_occurrence_member").astype(bool)
    identity = keypoint.frame_chaser("chaser_identity_code").astype(np.int64)
    role = keypoint.frame_chaser("chaser_behavior_role_code").astype(np.int64)
    selected = keypoint.collapsed_frame("selection_member").astype(bool)
    frame_id = keypoint.collapsed_frame("acquisition_frame_id").astype(np.int64)
    if not np.all(identity == identity[:1]) or not np.all(role == role[:1]):
        raise ValueError("Exact chaser identity or behavior role changes by frame.")

    panels: list[tuple[str, np.ndarray]] = [
        ("full recording", np.ones(keypoint.n_frames, dtype=bool))
    ]
    for record in projection.epoch_records:
        panels.append(
            (
                str(record["analysis_role"]),
                selected
                & (frame_id >= int(record["start_frame"]))
                & (frame_id < int(record["end_frame_exclusive"])),
            )
        )
    from plotly.subplots import make_subplots

    registry = identity_registry(
        projection.radials[0].scientific_manifest, "behavior_role"
    )
    titles = [
        (
            f"{label} · "
            f"{registry.get(str(int(role[0, column])), f'role {int(role[0, column])}')} "
            f"· chaser {int(identity[0, column])}"
        )
        for label, _ in panels
        for column in range(keypoint.n_chasers)
    ]
    figure = make_subplots(
        rows=len(panels),
        cols=keypoint.n_chasers,
        specs=[[{"type": "polar"} for _ in range(keypoint.n_chasers)] for _ in panels],
        subplot_titles=titles,
    )
    colors = ("#3b5b92", "#d95f02")
    for row_index, (_, frame_mask) in enumerate(panels, start=1):
        for column in range(keypoint.n_chasers):
            valid = frame_mask & occurrence[:, column] & body_valid[:, column]
            centers, probability, counts = _bearing_histogram(bearing[:, column], valid)
            figure.add_trace(
                go.Barpolar(
                    theta=centers,
                    r=probability,
                    width=np.full(centers.size, BEARING_BIN_WIDTH_DEG),
                    customdata=counts,
                    marker_color=colors[column % len(colors)],
                    name=f"chaser {int(identity[0, column])}",
                    showlegend=False,
                    hovertemplate=(
                        "bearing %{theta:.0f}°<br>probability %{r:.4f}"
                        "<br>valid rows %{customdata}<extra></extra>"
                    ),
                ),
                row=row_index,
                col=column + 1,
            )
    figure.update_polars(
        angularaxis={"direction": "counterclockwise", "rotation": 90},
        radialaxis={"title": "probability"},
    )
    figure.update_layout(
        title=(
            "Anatomical body-frame chaser bearing · keypoint body-axis authority · "
            f"{projection.recording_id}"
        ),
        height=330 * len(panels),
        meta={
            **plain(projection.provenance),
            "display_recipe": {
                "recipe_id": "accepted_body_axis_bearing_polar_histogram_v1",
                "bin_width_deg": BEARING_BIN_WIDTH_DEG,
                "normalization": "probability_within_panel_chaser",
                "body_axis_fallback": "prohibited",
                "detection_position_substitution": "prohibited",
            },
        },
    )
    return mo.vstack(
        [
            mo.callout(
                "Bearing is the chaser direction in the accepted keypoint body frame. "
                "Detection centroids are not used to infer anatomy; invalid rows stay excluded.",
                kind="info",
            ),
            figure,
        ]
    )


__all__ = [
    "BEARING_BIN_WIDTH_DEG",
    "_bearing_histogram",
    "build_exact_body_bearing_output",
]
