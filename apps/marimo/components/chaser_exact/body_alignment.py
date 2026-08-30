"""Render persisted anatomical alignment summaries without scientific rebinning."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from fisheye.analysis_workflows.chaser_body_alignment_by_distance_successor import (
    SUMMARY_VIEW_ARRAY_NAMES,
)
from fisheye.visualization.chaser_body_alignment_by_distance import (
    validate_persisted_body_alignment_summary,
)

from .projection import ExactChaserSuccessorProjection, identity_registry
from .provenance import plain


BODY_ALIGNMENT_DISPLAY_RECIPE = "persisted_anatomical_alignment_distance_bins_v1"
_EPOCH_COLORS = {
    1: "#4c78a8",
    2: "#e45756",
    3: "#54a24b",
}
_DASHES = ("solid", "dash", "dot", "dashdot")


def _body_alignment_values(
    projection: ExactChaserSuccessorProjection,
) -> Mapping[str, Any]:
    """Validate the selected persisted summary through the shared parser."""

    handle = projection.body_alignment_by_distance
    if handle is None:
        raise ValueError(
            "Alignment by distance requires one exact persisted body-alignment child."
        )
    return validate_persisted_body_alignment_summary(handle)


def _label(
    *,
    epoch: int,
    chaser: int,
    behavior_role: int,
    epoch_registry: Mapping[str, str],
    role_registry: Mapping[str, str],
) -> str:
    return (
        f"{epoch_registry.get(str(epoch), f'epoch {epoch}')} · "
        f"{role_registry.get(str(behavior_role), f'role {behavior_role}')} · "
        f"chaser {chaser}"
    )


def build_exact_body_alignment_output(
    mo: Any,
    go: Any,
    projection: ExactChaserSuccessorProjection,
) -> Any:
    """Render persisted alignment, bearing, circular, and support summaries."""

    from plotly.subplots import make_subplots

    values = _body_alignment_values(projection)
    handle = projection.body_alignment_by_distance
    epoch_registry = identity_registry(handle.scientific_manifest, "epoch_role")
    if not epoch_registry:
        epoch_registry = {
            "1": "chaser_pre",
            "2": "chaser_training",
            "3": "chaser_post",
        }
    role_registry = identity_registry(handle.scientific_manifest, "behavior_role")
    alignment = go.Figure()
    bearing = go.Figure()
    circular = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        subplot_titles=("Circular mean bearing", "Circular resultant length"),
    )
    support = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        subplot_titles=("Jointly valid fraction", "Persisted support counts"),
    )

    role_code = np.asarray(values["summary_epoch_role_code"], dtype=np.int64)
    identity = np.asarray(values["summary_chaser_identity_code"], dtype=np.int64)
    bin_index = np.asarray(values["summary_distance_bin_index"], dtype=np.int64)
    centers = np.asarray(values["summary_distance_bin_center_mm"], dtype=np.float64)
    candidate = np.asarray(values["summary_candidate_row_count"], dtype=np.int64)
    joint = np.asarray(values["summary_joint_valid_row_count"], dtype=np.int64)
    for epoch in (1, 2, 3):
        for chaser_position, chaser in enumerate(values["identities"]):
            member = (role_code == epoch) & (identity == chaser)
            indices = np.flatnonzero(member)[
                np.argsort(bin_index[member], kind="stable")
            ]
            behavior_role = int(
                np.asarray(values["summary_chaser_behavior_role_code"])[indices[0]]
            )
            label = _label(
                epoch=epoch,
                chaser=int(chaser),
                behavior_role=behavior_role,
                epoch_registry=epoch_registry,
                role_registry=role_registry,
            )
            line = {
                "color": _EPOCH_COLORS[epoch],
                "dash": _DASHES[chaser_position % len(_DASHES)],
            }
            custom_alignment = np.column_stack(
                (
                    candidate[indices],
                    joint[indices],
                    np.asarray(values["summary_alignment_cos_p25"])[indices],
                    np.asarray(values["summary_alignment_cos_p75"])[indices],
                )
            )
            alignment.add_trace(
                go.Scatter(
                    x=centers[indices],
                    y=np.asarray(values["summary_mean_alignment_cos"])[indices],
                    mode="lines+markers",
                    name=label,
                    line=line,
                    customdata=custom_alignment,
                    hovertemplate=(
                        "distance %{x:.1f} mm<br>mean alignment %{y:.3f}"
                        "<br>distance-valid %{customdata[0]:.0f}"
                        "<br>joint-valid %{customdata[1]:.0f}"
                        "<br>P25–P75 %{customdata[2]:.3f}–%{customdata[3]:.3f}"
                        "<extra>%{fullData.name}</extra>"
                    ),
                )
            )
            custom_bearing = np.column_stack(
                (
                    candidate[indices],
                    joint[indices],
                    np.asarray(values["summary_abs_bearing_p25_deg"])[indices],
                    np.asarray(values["summary_abs_bearing_p75_deg"])[indices],
                )
            )
            bearing.add_trace(
                go.Scatter(
                    x=centers[indices],
                    y=np.asarray(values["summary_mean_abs_bearing_deg"])[indices],
                    mode="lines+markers",
                    name=label,
                    line=line,
                    customdata=custom_bearing,
                    hovertemplate=(
                        "distance %{x:.1f} mm<br>mean |bearing| %{y:.2f}°"
                        "<br>distance-valid %{customdata[0]:.0f}"
                        "<br>joint-valid %{customdata[1]:.0f}"
                        "<br>P25–P75 %{customdata[2]:.2f}°–%{customdata[3]:.2f}°"
                        "<extra>%{fullData.name}</extra>"
                    ),
                )
            )
            circular.add_trace(
                go.Scatter(
                    x=centers[indices],
                    y=np.asarray(values["summary_circular_mean_bearing_deg"])[indices],
                    mode="lines+markers",
                    name=label,
                    legendgroup=label,
                    line=line,
                    showlegend=True,
                ),
                row=1,
                col=1,
            )
            circular.add_trace(
                go.Scatter(
                    x=centers[indices],
                    y=np.asarray(values["summary_circular_resultant_length"])[indices],
                    mode="lines+markers",
                    name=label,
                    legendgroup=label,
                    line=line,
                    showlegend=False,
                ),
                row=2,
                col=1,
            )
            fraction = np.divide(
                joint[indices].astype(np.float64),
                candidate[indices].astype(np.float64),
                out=np.full(indices.size, np.nan),
                where=candidate[indices] > 0,
            )
            support.add_trace(
                go.Scatter(
                    x=centers[indices],
                    y=fraction * 100.0,
                    mode="lines+markers",
                    name=label,
                    legendgroup=label,
                    line=line,
                    showlegend=True,
                ),
                row=1,
                col=1,
            )
            support.add_trace(
                go.Scatter(
                    x=centers[indices],
                    y=joint[indices],
                    mode="lines+markers",
                    name=f"joint valid · {label}",
                    legendgroup=label,
                    line=line,
                    showlegend=False,
                ),
                row=2,
                col=1,
            )

    alignment.add_hline(y=0.0, line_dash="dot", line_color="#777")
    alignment.update_layout(
        title="Anatomical fish alignment by exact chaser distance",
        xaxis_title="fish–chaser distance (mm; persisted bins)",
        yaxis_title="mean cos(body bearing): +1 front, −1 behind",
    )
    alignment.update_yaxes(range=[-1.05, 1.05])
    bearing.update_layout(
        title="Absolute anatomical bearing by exact chaser distance",
        xaxis_title="fish–chaser distance (mm; persisted bins)",
        yaxis_title="mean absolute bearing (degrees)",
    )
    bearing.update_yaxes(range=[0.0, 180.0])
    circular.update_layout(title="Circular anatomical bearing summaries")
    circular.update_yaxes(title_text="degrees", range=[-180.0, 180.0], row=1, col=1)
    circular.update_yaxes(title_text="resultant", range=[0.0, 1.0], row=2, col=1)
    circular.update_xaxes(
        title_text="fish–chaser distance (mm; persisted bins)", row=2, col=1
    )
    support.update_layout(title="Alignment validity and support")
    support.update_yaxes(title_text="joint valid (%)", range=[0.0, 100.0], row=1, col=1)
    support.update_yaxes(title_text="rows", row=2, col=1)
    support.update_xaxes(
        title_text="fish–chaser distance (mm; persisted bins)", row=2, col=1
    )

    display = {
        "recipe_id": BODY_ALIGNMENT_DISPLAY_RECIPE,
        "source_arrays": list(SUMMARY_VIEW_ARRAY_NAMES),
        "distance_bin_edges_mm": np.asarray(values["distance_bin_edges_mm"]).tolist(),
        "distance_bin_recipe": plain(handle.scientific_manifest["distance_bin_recipe"]),
        "alignment_definition": "cos(body_bearing_deg)",
        "lateral_definition": "sin(body_bearing_deg)",
        "distance_surface": "base/relative_distance_physical",
        "body_origin_distance_substitution": "prohibited",
        "motion_heading_fallback": "prohibited",
        "viewer_rebinning": "prohibited",
        "viewer_scientific_groupby": "prohibited",
        "scientific_recomputation": False,
    }
    meta = plain(projection.provenance)
    meta["body_alignment_by_distance_display"] = display
    for figure in (alignment, bearing, circular, support):
        figure.update_layout(meta=meta)
    return mo.vstack(
        [
            mo.callout(
                "These curves are read from persisted semantic-epoch × chaser × "
                "distance-bin summaries. Distance comes from the selected keypoint "
                "position provider, while orientation comes only from the accepted "
                "anatomical body frame. Missing body rows and invalid distance rows "
                "remain explicit evidence; no motion-heading fallback or viewer "
                "rebinning is used.",
                kind="info",
            ),
            alignment,
            bearing,
            circular,
            support,
        ]
    )


__all__ = [
    "BODY_ALIGNMENT_DISPLAY_RECIPE",
    "_body_alignment_values",
    "build_exact_body_alignment_output",
]
