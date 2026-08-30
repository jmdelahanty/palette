"""Reviewed-arena exact-epoch trajectory overlays for chaser successors."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .projection import ExactChaserSuccessorProjection, identity_registry
from .provenance import TRAJECTORY_MAX_POINTS, plain


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be one object.")
    return value


def _trajectory_display_indices(
    xy: np.ndarray,
    valid: np.ndarray,
    *,
    max_points: int = TRAJECTORY_MAX_POINTS,
) -> np.ndarray:
    points = np.asarray(xy, dtype=np.float64)
    observed = np.asarray(valid, dtype=bool).reshape(-1)
    if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] != observed.size:
        raise ValueError("Trajectory display vectors have inconsistent shapes.")
    candidates = np.flatnonzero(observed & np.all(np.isfinite(points), axis=1))
    if candidates.size <= max_points:
        return candidates
    chosen = set(
        int(value)
        for value in candidates[
            np.linspace(0, candidates.size - 1, max_points, dtype=np.int64)
        ].tolist()
    )
    for column in range(2):
        chosen.add(int(candidates[int(np.argmin(points[candidates, column]))]))
        chosen.add(int(candidates[int(np.argmax(points[candidates, column]))]))
    return np.asarray(sorted(chosen), dtype=np.int64)


def build_exact_trajectory_overlays_output(
    mo: Any, go: Any, projection: ExactChaserSuccessorProjection
) -> Any:
    """Render display-only fish and logged-chaser positions by exact epoch."""

    if projection.relatives is None:
        raise ValueError("Trajectory overlays require exact relative-frame sources.")
    from plotly.subplots import make_subplots

    keypoint = projection.relatives[0]
    frame_id = keypoint.collapsed_frame("acquisition_frame_id").astype(np.int64)
    selected = keypoint.collapsed_frame("selection_member").astype(bool)
    chaser_xy = keypoint.frame_chaser("chaser_position_xy_px")
    chaser_valid = keypoint.frame_chaser(
        "chaser_position_valid"
    ) & keypoint.frame_chaser("chaser_occurrence_member")
    identities = keypoint.frame_chaser("chaser_identity_code")
    roles = keypoint.frame_chaser("chaser_behavior_role_code")
    registry = identity_registry(
        projection.radials[0].scientific_manifest, "behavior_role"
    )
    arena = _mapping(
        projection.radials[0].scientific_manifest.get("arena"),
        label="reviewed arena",
    )
    center_x = float(arena["center_x_px"])
    center_y = float(arena["center_y_px"])
    radius = float(arena["radius_px"])
    titles = [
        f"{record['analysis_role']} · {provider_id}"
        for provider_id in projection.provider_ids
        for record in projection.epoch_records
    ]
    figure = make_subplots(
        rows=2, cols=len(projection.epoch_records), subplot_titles=titles
    )
    chaser_colors = ("#2ca02c", "#9467bd", "#8c564b", "#e377c2")
    for provider_index, (provider_id, relative) in enumerate(
        zip(projection.provider_ids, projection.relatives, strict=True), start=1
    ):
        fish_xy = relative.collapsed_frame("fish_position_xy_px")
        fish_valid = relative.collapsed_frame("fish_position_valid").astype(bool)
        for epoch_index, record in enumerate(projection.epoch_records, start=1):
            mask = (
                selected
                & (frame_id >= int(record["start_frame"]))
                & (frame_id < int(record["end_frame_exclusive"]))
            )
            rows = np.flatnonzero(mask)
            if not rows.size:
                raise ValueError("An exact trajectory panel has no source rows.")
            fish_display = _trajectory_display_indices(fish_xy[rows], fish_valid[rows])
            source_rows = rows[fish_display]
            figure.add_trace(
                go.Scattergl(
                    x=fish_xy[source_rows, 0],
                    y=fish_xy[source_rows, 1],
                    mode="markers",
                    name=f"fish · {provider_id}",
                    legendgroup=f"fish-{provider_id}",
                    showlegend=epoch_index == 1,
                    marker={"color": "#222222", "size": 2, "opacity": 0.25},
                ),
                row=provider_index,
                col=epoch_index,
            )
            for column in range(keypoint.n_chasers):
                local_valid = chaser_valid[rows, column]
                local_xy = chaser_xy[rows, column]
                display = _trajectory_display_indices(local_xy, local_valid)
                chaser_rows = rows[display]
                role = registry.get(
                    str(int(roles[0, column])), f"role {int(roles[0, column])}"
                )
                label = f"{role} · chaser {int(identities[0, column])}"
                figure.add_trace(
                    go.Scattergl(
                        x=chaser_xy[chaser_rows, column, 0],
                        y=chaser_xy[chaser_rows, column, 1],
                        mode="markers",
                        name=label,
                        legendgroup=label,
                        showlegend=provider_index == 1 and epoch_index == 1,
                        marker={
                            "color": chaser_colors[column % len(chaser_colors)],
                            "size": 3,
                            "opacity": 0.55,
                        },
                    ),
                    row=provider_index,
                    col=epoch_index,
                )
            figure.add_shape(
                type="circle",
                x0=center_x - radius,
                x1=center_x + radius,
                y0=center_y - radius,
                y1=center_y + radius,
                line={"color": "#666666", "width": 1},
                row=provider_index,
                col=epoch_index,
            )
            figure.update_xaxes(
                range=[center_x - radius * 1.03, center_x + radius * 1.03],
                row=provider_index,
                col=epoch_index,
            )
            figure.update_yaxes(
                range=[center_y + radius * 1.03, center_y - radius * 1.03],
                scaleanchor=(
                    f"x{(provider_index - 1) * len(projection.epoch_records) + epoch_index}"
                ),
                scaleratio=1,
                row=provider_index,
                col=epoch_index,
            )
    figure.update_layout(
        title=(
            "Exact-epoch fish positions with logged chaser overlays · "
            f"{projection.recording_id}"
        ),
        height=820,
        meta=plain(projection.provenance),
    )
    return mo.vstack(
        [
            mo.callout(
                f"Display-only deterministic source-order projection, at most {TRAJECTORY_MAX_POINTS:,} valid points per series and panel; scientific occupancy remains in the persisted successor.",
                kind="info",
            ),
            figure,
        ]
    )


__all__ = ["build_exact_trajectory_overlays_output"]
