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

    if projection.relatives is None or projection.chaser_appearance is None:
        raise ValueError(
            "Trajectory overlays require exact relative-frame and appearance sources."
        )
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
    if not np.all(identities == identities[:1]) or not np.all(roles == roles[:1]):
        raise ValueError("Exact chaser identity or behavior role changes by frame.")
    appearance_by_code = projection.chaser_appearance.by_identity_code()
    if set(appearance_by_code) != set(int(value) for value in identities[0]):
        raise ValueError("Chaser appearance identities differ from trajectory columns.")
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
                appearance = appearance_by_code[int(identities[0, column])]
                if (
                    appearance.behavior_role_code != int(roles[0, column])
                    or appearance.behavior_role != role
                ):
                    raise ValueError(
                        "Chaser appearance role differs from trajectory evidence."
                    )
                label = f"{role} · protocol chaser {appearance.chaser_index}"
                figure.add_trace(
                    go.Scattergl(
                        x=chaser_xy[chaser_rows, column, 0],
                        y=chaser_xy[chaser_rows, column, 1],
                        mode="markers",
                        name=label,
                        legendgroup=label,
                        showlegend=provider_index == 1 and epoch_index == 1,
                        marker={
                            "color": appearance.experimental_color_css,
                            "symbol": appearance.plotly_role_symbol,
                            "size": 4,
                            "opacity": 0.62,
                            "line": {
                                "color": appearance.contrast_outline_hex,
                                "width": 0.7,
                            },
                        },
                        hovertemplate=(
                            f"{role} · protocol chaser {appearance.chaser_index}<br>"
                            f"identity={appearance.identity}<br>"
                            f"experimental color={appearance.experimental_color_hex}<br>"
                            "x=%{x:.2f} px<br>y=%{y:.2f} px<extra></extra>"
                        ),
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
        meta={
            **plain(projection.provenance),
            "trajectory_chaser_appearance": {
                "color_source": "sealed_protocol_rgba",
                "role_encoding": "independent_marker_symbol_and_legend_text",
                "identity_encoding": "protocol_chaser_index_and_exact_identity_hover",
                "index_palette_fallback": "prohibited",
                "display_opacity": 0.62,
            },
        },
    )
    return mo.vstack(
        [
            mo.callout(
                f"Display-only deterministic source-order projection, at most {TRAJECTORY_MAX_POINTS:,} valid points per series and panel. Marker fill is the sealed experimental protocol color; shape and text independently encode behavior role. Scientific occupancy remains in the persisted successor.",
                kind="info",
            ),
            figure,
        ]
    )


__all__ = ["build_exact_trajectory_overlays_output"]
