"""Exact-session distance traces for paired chaser position providers."""

from __future__ import annotations

from typing import Any

import numpy as np

from .projection import ExactChaserSuccessorProjection, identity_registry
from .provenance import TRACE_MAX_POINTS, plain


def _trace_display_projection(
    x: np.ndarray,
    y: np.ndarray,
    valid: np.ndarray,
    *,
    max_points: int = TRACE_MAX_POINTS,
) -> tuple[np.ndarray, np.ndarray]:
    """Bound display size while preserving extrema and every observed gap."""

    x_values = np.asarray(x, dtype=np.float64).reshape(-1)
    y_values = np.asarray(y, dtype=np.float64).reshape(-1)
    observed = (
        np.asarray(valid, dtype=bool).reshape(-1)
        & np.isfinite(x_values)
        & np.isfinite(y_values)
    )
    if not (x_values.size == y_values.size == observed.size):
        raise ValueError("Trace display vectors have inconsistent lengths.")
    if x_values.size <= max_points:
        output = y_values.copy()
        output[~observed] = np.nan
        return x_values.copy(), output
    bucket_count = max(1, max_points // 4)
    edges = np.linspace(0, x_values.size, bucket_count + 1, dtype=np.int64)
    selected: set[int] = set()
    for start, end in zip(edges[:-1], edges[1:], strict=True):
        candidates = np.flatnonzero(observed[start:end]) + start
        if not candidates.size:
            continue
        selected.update((int(candidates[0]), int(candidates[-1])))
        local = y_values[candidates]
        selected.add(int(candidates[int(np.argmin(local))]))
        selected.add(int(candidates[int(np.argmax(local))]))
    indices = np.asarray(sorted(selected), dtype=np.int64)
    projected_x: list[float] = []
    projected_y: list[float] = []
    previous: int | None = None
    for index in indices.tolist():
        if previous is not None and np.any(~observed[previous + 1 : index]):
            projected_x.append(float(x_values[previous]))
            projected_y.append(float("nan"))
        projected_x.append(float(x_values[index]))
        projected_y.append(float(y_values[index]))
        previous = index
    return np.asarray(projected_x), np.asarray(projected_y)


def build_exact_distance_traces_output(
    mo: Any, go: Any, projection: ExactChaserSuccessorProjection
) -> Any:
    """Render full-recording and exact-epoch distance without interpolation."""

    if projection.relatives is None:
        raise ValueError("Distance traces require exact relative-frame sources.")
    from plotly.subplots import make_subplots

    keypoint = projection.relatives[0]
    frame_id = keypoint.collapsed_frame("acquisition_frame_id").astype(np.int64)
    timestamp = keypoint.collapsed_frame("timestamp_ns").astype(np.int64)
    timestamp_valid = keypoint.collapsed_frame("timestamp_valid").astype(bool)
    selected = keypoint.collapsed_frame("selection_member").astype(bool)
    valid_timestamp_rows = np.flatnonzero(timestamp_valid)
    if not valid_timestamp_rows.size:
        raise ValueError("Exact relative-frame source has no valid session time.")
    time_s = (
        timestamp.astype(np.float64) - float(timestamp[valid_timestamp_rows[0]])
    ) / 1e9
    identities = keypoint.frame_chaser("chaser_identity_code")
    roles = keypoint.frame_chaser("chaser_behavior_role_code")
    if not np.all(identities == identities[:1]) or not np.all(roles == roles[:1]):
        raise ValueError("Exact chaser identity or behavior roles change by frame.")
    registry = identity_registry(
        projection.radials[0].scientific_manifest, "behavior_role"
    )
    panels: list[tuple[str, np.ndarray]] = [
        ("full recording", np.ones(frame_id.size, dtype=bool))
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
    titles = [
        f"{panel} · {registry.get(str(int(roles[0, column])), f'role {int(roles[0, column])}')} · chaser {int(identities[0, column])}"
        for panel, _ in panels
        for column in range(keypoint.n_chasers)
    ]
    figure = make_subplots(
        rows=len(panels), cols=keypoint.n_chasers, subplot_titles=titles
    )
    colors = ("#1f77b4", "#d95f02")
    for row_index, (_, panel_mask) in enumerate(panels, start=1):
        indices = np.flatnonzero(panel_mask)
        if not indices.size:
            raise ValueError("An exact distance panel has no source rows.")
        for column in range(keypoint.n_chasers):
            for provider_index, (provider_id, relative) in enumerate(
                zip(projection.provider_ids, projection.relatives, strict=True)
            ):
                values = relative.frame_chaser("relative_distance_physical")[:, column]
                valid = (
                    relative.frame_chaser("relative_physical_valid")[:, column]
                    & relative.frame_chaser("chaser_occurrence_member")[:, column]
                    & timestamp_valid
                )
                display_x, display_y = _trace_display_projection(
                    time_s[indices], values[indices], valid[indices]
                )
                figure.add_trace(
                    go.Scattergl(
                        x=display_x,
                        y=display_y,
                        mode="lines",
                        name=provider_id,
                        legendgroup=provider_id,
                        showlegend=row_index == 1 and column == 0,
                        connectgaps=False,
                        line={"color": colors[provider_index], "width": 1},
                    ),
                    row=row_index,
                    col=column + 1,
                )
    figure.update_xaxes(title_text="session time from first valid timestamp (s)")
    figure.update_yaxes(title_text="distance (mm)")
    figure.update_layout(
        title=(
            "Full-recording and exact-epoch fish–chaser distance · "
            f"{projection.recording_id}"
        ),
        height=280 * len(panels),
        meta=plain(projection.provenance),
    )
    return mo.vstack(
        [
            mo.callout(
                f"Display-only extrema-preserving projection, at most {TRACE_MAX_POINTS:,} source points per trace; missing rows always break lines.",
                kind="info",
            ),
            figure,
        ]
    )


__all__ = ["build_exact_distance_traces_output"]
