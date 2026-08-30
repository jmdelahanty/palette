"""Exact anatomical fish-heading view from the keypoint body extension."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from ..chaser_exact_body_heading_contract import (
    BODY_HEADING_ARRAY_PATHS,
    BODY_HEADING_FRAME_COLLAPSE_POLICY,
)
from .projection import ExactChaserSuccessorProjection
from .provenance import plain

BODY_HEADING_BIN_WIDTH_DEG = 10.0
BODY_HEADING_DISPLAY_RECIPE = "accepted_body_axis_fish_heading_polar_histogram_v1"
_BODY_HEADING_ARRAY_NAMES = tuple(
    path.removeprefix("body/") for path in BODY_HEADING_ARRAY_PATHS
)


def _collapsed_body_heading(keypoint: Any) -> Mapping[str, np.ndarray]:
    """Collapse repeated frame evidence only after exact cross-chaser equality."""

    if set(keypoint.body_arrays) != set(_BODY_HEADING_ARRAY_NAMES):
        raise ValueError("Keypoint relative frame lacks the closed heading roster.")
    reshaped = {
        name: keypoint.body_frame_chaser(name) for name in _BODY_HEADING_ARRAY_NAMES
    }
    for name, values in reshaped.items():
        reference = values[:, :1, ...]
        if values.dtype.kind == "f":
            repeated = np.array_equal(
                values, np.broadcast_to(reference, values.shape), equal_nan=True
            )
        else:
            repeated = np.array_equal(values, np.broadcast_to(reference, values.shape))
        if not repeated:
            raise ValueError(
                f"Frame-level body-heading evidence {name!r} differs by chaser row."
            )
    collapsed = {name: values[:, 0, ...] for name, values in reshaped.items()}
    heading = np.asarray(collapsed["body_heading_deg"], dtype=np.float64)
    valid = np.asarray(collapsed["body_heading_valid"], dtype=bool)
    source_row = np.asarray(collapsed["body_source_row_id"], dtype=np.int64)
    source_valid = np.asarray(collapsed["body_source_row_valid"], dtype=bool)
    reason = np.asarray(collapsed["body_heading_reason_code"], dtype=np.int64)
    if any(
        values.ndim != 1
        for values in (heading, valid, source_row, source_valid, reason)
    ):
        raise ValueError("Collapsed body-heading arrays must be one-dimensional.")
    if (
        not np.array_equal(source_valid, source_row >= 0)
        or np.any(valid & ~source_valid)
        or np.any(
            valid & (~np.isfinite(heading) | (heading < -180.0) | (heading > 180.0))
        )
        or np.any(~valid & np.isfinite(heading))
        or np.any(valid & (reason != 0))
        or np.any(~valid & (reason == 0))
        or np.any(reason < 0)
    ):
        raise ValueError(
            "Persisted body-heading validity or source identity is invalid."
        )
    return {
        "heading_deg": heading,
        "valid": valid,
        "source_row_id": source_row,
        "source_row_valid": source_valid,
        "reason_code": reason,
    }


def _heading_histogram(
    heading: np.ndarray, valid: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if 360.0 % BODY_HEADING_BIN_WIDTH_DEG != 0.0:
        raise ValueError("Heading bin width must divide the whole circle.")
    edges = np.arange(
        -180.0,
        180.0 + BODY_HEADING_BIN_WIDTH_DEG,
        BODY_HEADING_BIN_WIDTH_DEG,
    )
    counts = np.histogram(heading[valid], bins=edges)[0].astype(np.int64)
    total = int(np.sum(counts))
    probability = (
        counts.astype(np.float64) / total
        if total
        else np.zeros(counts.size, dtype=np.float64)
    )
    return (edges[:-1] + edges[1:]) / 2.0, probability, counts


def build_exact_body_heading_output(
    mo: Any, go: Any, projection: ExactChaserSuccessorProjection
) -> Any:
    """Render one fish-heading sample per acquisition frame and semantic panel."""

    if projection.relatives is None:
        raise ValueError("Fish heading requires the keypoint relative frame.")
    keypoint = projection.relatives[0]
    body_authority = keypoint.source_authorities.get("body_frame")
    if not isinstance(body_authority, Mapping):
        raise ValueError("Fish heading lacks its sealed body-frame authority.")
    body = _collapsed_body_heading(keypoint)
    frame_id = keypoint.collapsed_frame("acquisition_frame_id").astype(np.int64)
    selected = keypoint.collapsed_frame("selection_member").astype(bool)
    if frame_id.shape != body["heading_deg"].shape or selected.shape != frame_id.shape:
        raise ValueError("Fish-heading frame axes are inconsistent.")

    panels: list[tuple[str, np.ndarray]] = [
        ("full recording", np.ones(keypoint.n_frames, dtype=bool))
    ]
    for record in projection.epoch_records:
        role = record.get("analysis_role")
        window_id = record.get("window_id")
        start = record.get("start_frame")
        end = record.get("end_frame_exclusive")
        if (
            type(role) is not str
            or type(window_id) is not int
            or type(start) is not int
            or type(end) is not int
            or end <= start
        ):
            raise ValueError("Fish heading received an invalid exact epoch record.")
        panels.append(
            (
                f"{role} · window {window_id}",
                selected & (frame_id >= start) & (frame_id < end),
            )
        )
    if not panels:
        raise ValueError("Fish-heading view has no panels.")

    from plotly.subplots import make_subplots

    figure = make_subplots(
        rows=len(panels),
        cols=1,
        specs=[[{"type": "polar"}] for _ in panels],
        subplot_titles=[label for label, _ in panels],
        vertical_spacing=min(0.08, 0.25 / max(len(panels), 1)),
    )
    panel_records = []
    for panel_index, (label, membership) in enumerate(panels):
        valid = membership & np.asarray(body["valid"], dtype=bool)
        centers, probability, counts = _heading_histogram(
            np.asarray(body["heading_deg"], dtype=np.float64),
            valid,
        )
        source_present = membership & np.asarray(body["source_row_valid"], dtype=bool)
        panel_records.append(
            {
                "label": label,
                "candidate_frame_count": int(np.count_nonzero(membership)),
                "body_source_present_count": int(np.count_nonzero(source_present)),
                "valid_heading_count": int(np.count_nonzero(valid)),
                "missing_body_source_count": int(
                    np.count_nonzero(membership & ~body["source_row_valid"])
                ),
                "present_invalid_heading_count": int(
                    np.count_nonzero(source_present & ~body["valid"])
                ),
            }
        )
        figure.add_trace(
            go.Barpolar(
                theta=centers,
                r=probability,
                width=np.full(centers.shape, BODY_HEADING_BIN_WIDTH_DEG),
                customdata=counts,
                name=label,
                hovertemplate=(
                    "heading=%{theta:.1f}°<br>"
                    "probability=%{r:.5f}<br>"
                    "exact frame count=%{customdata}<extra>%{fullData.name}</extra>"
                ),
            ),
            row=panel_index + 1,
            col=1,
        )
    display = {
        "recipe_id": BODY_HEADING_DISPLAY_RECIPE,
        "source_relative_frame": keypoint.run_path,
        "source_relative_manifest_sha256": keypoint.manifest_sha256,
        "body_frame_authority": plain(body_authority),
        "source_arrays": list(BODY_HEADING_ARRAY_PATHS),
        "frame_collapse_policy": BODY_HEADING_FRAME_COLLAPSE_POLICY,
        "bin_width_deg": BODY_HEADING_BIN_WIDTH_DEG,
        "bin_edges_deg": np.arange(
            -180.0,
            180.0 + BODY_HEADING_BIN_WIDTH_DEG,
            BODY_HEADING_BIN_WIDTH_DEG,
        ).tolist(),
        "normalization": "probability_within_semantic_panel_one_row_per_frame",
        "angle_convention": (
            "atan2(-forward_axis_y_camera_down,forward_axis_x)_degrees_"
            "counterclockwise_in_y_up"
        ),
        "body_axis_fallback": "prohibited",
        "motion_heading_fallback": "prohibited",
        "detection_position_substitution": "prohibited",
        "panels": panel_records,
        "scientific_recomputation": False,
    }
    meta = plain(projection.provenance)
    meta["body_heading_display"] = display
    figure.update_layout(
        title="Exact anatomical fish heading by protocol-semantic epoch",
        height=330 * len(panels),
        meta=meta,
    )
    figure.update_polars(
        angularaxis={"direction": "counterclockwise", "rotation": 0},
        radialaxis={"title": "probability"},
    )
    return mo.vstack(
        [
            mo.callout(
                "Heading comes only from the accepted keypoint body-axis supplier. "
                "Flattened chaser rows are required to agree exactly and are collapsed "
                "to one fish-heading sample per acquisition frame. Missing body rows "
                "remain missing; motion and detection fallbacks are prohibited.",
                kind="info",
            ),
            figure,
        ]
    )


__all__ = [
    "BODY_HEADING_BIN_WIDTH_DEG",
    "BODY_HEADING_DISPLAY_RECIPE",
    "_collapsed_body_heading",
    "_heading_histogram",
    "build_exact_body_heading_output",
]
