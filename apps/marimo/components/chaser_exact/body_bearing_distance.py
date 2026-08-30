"""Receipt-bound anatomical bearing by fish--chaser distance views."""

from __future__ import annotations

from typing import Any

import numpy as np

from fisheye.visualization.chaser_body_bearing_distance import (
    BEARING_BIN_WIDTH_DEG,
    DENSITY_COLOR_CMAX_QUANTILE,
    DISPLAY_RECIPE_ID,
    DISTANCE_BIN_WIDTH_MM,
    INTERACTIVE_POINT_CLOUD_MAX_ROWS_PER_PANEL_CHASER,
    bearing_bin_edges_deg,
    body_bearing_distance_histogram,
    body_bearing_distance_valid_mask,
    distance_bin_edges_mm,
    positive_probability_color_max,
    uniformly_sample_indices,
)

from .projection import ExactChaserSuccessorProjection, identity_registry
from .provenance import plain


def _panel_specs(
    projection: ExactChaserSuccessorProjection,
    *,
    selected: np.ndarray,
    frame_id: np.ndarray,
) -> tuple[tuple[str, np.ndarray], ...]:
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
    return tuple(panels)


def _subplot_titles(
    panels: tuple[tuple[str, np.ndarray], ...],
    identity: np.ndarray,
    role: np.ndarray,
    role_registry: dict[str, str],
) -> list[str]:
    return [
        (
            f"{label} · "
            f"{role_registry.get(str(int(role[0, column])), f'role {int(role[0, column])}')} "
            f"· chaser {int(identity[0, column])}"
        )
        for label, _ in panels
        for column in range(identity.shape[1])
    ]


def _polar_layout(figure: Any, *, radial_max_mm: float) -> None:
    figure.update_polars(
        radialaxis={
            "title": "distance (mm)",
            "range": [0.0, float(radial_max_mm)],
        },
        angularaxis={
            "direction": "counterclockwise",
            "rotation": 90,
            "tickmode": "array",
            "tickvals": [-180, -90, 0, 90, 180],
            "ticktext": ["behind", "right", "front", "left", "behind"],
        },
    )


def build_exact_body_bearing_distance_output(
    mo: Any,
    go: Any,
    projection: ExactChaserSuccessorProjection,
) -> Any:
    """Render exact point clouds and joint polar density without interpolation."""

    if projection.relatives is None:
        raise ValueError(
            "Body-bearing distance views require the keypoint relative frame."
        )
    keypoint = projection.relatives[0]
    required_body_arrays = {"body_bearing_deg", "body_bearing_valid"}
    if not required_body_arrays.issubset(keypoint.body_arrays):
        raise ValueError(
            "Keypoint relative frame lacks the sealed body-bearing arrays."
        )
    distance = np.asarray(
        keypoint.frame_chaser("relative_distance_physical"),
        dtype=np.float64,
    )
    distance_valid = np.asarray(
        keypoint.frame_chaser("relative_physical_valid"), dtype=bool
    )
    bearing = np.asarray(
        keypoint.body_frame_chaser("body_bearing_deg"), dtype=np.float64
    )
    bearing_valid = np.asarray(
        keypoint.body_frame_chaser("body_bearing_valid"), dtype=bool
    )
    occurrence = np.asarray(
        keypoint.frame_chaser("chaser_occurrence_member"), dtype=bool
    )
    identity = np.asarray(keypoint.frame_chaser("chaser_identity_code"), dtype=np.int64)
    role = np.asarray(
        keypoint.frame_chaser("chaser_behavior_role_code"), dtype=np.int64
    )
    selected = keypoint.collapsed_frame("selection_member").astype(bool)
    frame_id = keypoint.collapsed_frame("acquisition_frame_id").astype(np.int64)
    expected_shape = (keypoint.n_frames, keypoint.n_chasers)
    if any(
        values.shape != expected_shape
        for values in (
            distance,
            distance_valid,
            bearing,
            bearing_valid,
            occurrence,
            identity,
            role,
        )
    ):
        raise ValueError(
            "Body-bearing distance arrays do not preserve the frame/chaser axes."
        )
    if not np.all(identity == identity[:1]) or not np.all(role == role[:1]):
        raise ValueError("Exact chaser identity or behavior role changes by frame.")

    full_member = np.ones(expected_shape, dtype=bool)
    full_valid = body_bearing_distance_valid_mask(
        distance,
        bearing,
        distance_valid,
        bearing_valid,
        occurrence,
        full_member,
    )
    if not np.any(full_valid):
        return mo.callout(
            "The exact body-bearing binding is present, but it contains no jointly "
            "valid anatomical-bearing and physical-distance rows.",
            kind="warn",
        )
    distance_edges = distance_bin_edges_mm(distance, full_valid)
    bearing_edges = bearing_bin_edges_deg()
    panels = _panel_specs(projection, selected=selected, frame_id=frame_id)
    role_registry = dict(
        identity_registry(projection.radials[0].scientific_manifest, "behavior_role")
    )
    titles = _subplot_titles(panels, identity, role, role_registry)

    from plotly.subplots import make_subplots

    subplot_specs = [
        [{"type": "polar"} for _ in range(keypoint.n_chasers)] for _ in panels
    ]
    point_cloud = make_subplots(
        rows=len(panels),
        cols=keypoint.n_chasers,
        specs=subplot_specs,
        subplot_titles=titles,
    )
    density = make_subplots(
        rows=len(panels),
        cols=keypoint.n_chasers,
        specs=subplot_specs,
        subplot_titles=titles,
    )
    colors = ("#3b5b92", "#d95f02")
    histograms = []
    panel_records: list[dict[str, Any]] = []
    for row_index, (label, frame_member) in enumerate(panels, start=1):
        panel_member = np.broadcast_to(frame_member[:, None], expected_shape)
        panel_valid = body_bearing_distance_valid_mask(
            distance,
            bearing,
            distance_valid,
            bearing_valid,
            occurrence,
            panel_member,
        )
        for column in range(keypoint.n_chasers):
            valid_indices = np.flatnonzero(panel_valid[:, column])
            display_indices = uniformly_sample_indices(
                valid_indices,
                maximum=INTERACTIVE_POINT_CLOUD_MAX_ROWS_PER_PANEL_CHASER,
            )
            point_cloud.add_trace(
                go.Scatterpolar(
                    theta=bearing[display_indices, column],
                    r=distance[display_indices, column],
                    mode="markers",
                    marker={
                        "size": 4,
                        "opacity": 0.3,
                        "color": colors[column % len(colors)],
                    },
                    customdata=frame_id[display_indices],
                    name=f"chaser {int(identity[0, column])}",
                    showlegend=False,
                    hovertemplate=(
                        "bearing %{theta:.1f}°<br>distance %{r:.2f} mm"
                        "<br>acquisition frame %{customdata}<extra></extra>"
                    ),
                ),
                row=row_index,
                col=column + 1,
            )
            histogram = body_bearing_distance_histogram(
                distance[:, column],
                bearing[:, column],
                panel_valid[:, column],
                distance_edges_mm=distance_edges,
                bearing_edges_deg=bearing_edges,
            )
            histograms.append(histogram)
            panel_records.append(
                {
                    "panel": label,
                    "chaser_identity_code": int(identity[0, column]),
                    "valid_row_count": histogram.denominator,
                    "point_cloud_display_row_count": int(display_indices.size),
                }
            )

    color_max = positive_probability_color_max(histograms)
    bearing_centers = (bearing_edges[:-1] + bearing_edges[1:]) / 2.0
    distance_widths = np.diff(distance_edges)
    show_colorbar = True
    histogram_index = 0
    for row_index, _panel in enumerate(panels, start=1):
        for column in range(keypoint.n_chasers):
            histogram = histograms[histogram_index]
            histogram_index += 1
            distance_left = np.repeat(distance_edges[:-1], bearing_centers.size)
            radial_width = np.repeat(distance_widths, bearing_centers.size)
            theta = np.tile(bearing_centers, distance_widths.size)
            bearing_width = np.tile(np.diff(bearing_edges), distance_widths.size)
            counts = histogram.counts.reshape(-1)
            probability = histogram.probability.reshape(-1)
            observed_bins = counts > 0
            distance_left = distance_left[observed_bins]
            radial_width = radial_width[observed_bins]
            theta = theta[observed_bins]
            bearing_width = bearing_width[observed_bins]
            counts = counts[observed_bins]
            probability = probability[observed_bins]
            customdata = np.column_stack(
                (
                    counts,
                    probability,
                    distance_left,
                    distance_left + radial_width,
                    theta - bearing_width / 2.0,
                    theta + bearing_width / 2.0,
                    np.full(counts.size, histogram.denominator, dtype=np.int64),
                )
            )
            density.add_trace(
                go.Barpolar(
                    theta=theta,
                    r=radial_width,
                    base=distance_left,
                    width=bearing_width,
                    marker={
                        "color": probability,
                        "colorscale": "Viridis",
                        "cmin": 0.0,
                        "cmax": float(color_max),
                        "showscale": show_colorbar,
                        "colorbar": {"title": "fraction/bin"},
                        "line": {"width": 0},
                    },
                    opacity=0.95,
                    name=f"chaser {int(identity[0, column])}",
                    showlegend=False,
                    customdata=customdata,
                    hovertemplate=(
                        "bearing bin %{customdata[4]:.1f}° to "
                        "%{customdata[5]:.1f}°<br>distance bin "
                        "%{customdata[2]:.1f} to %{customdata[3]:.1f} mm"
                        "<br>rows %{customdata[0]:,} / %{customdata[6]:,}"
                        "<br>fraction %{customdata[1]:.5f}<extra></extra>"
                    ),
                ),
                row=row_index,
                col=column + 1,
            )
            show_colorbar = False

    recipe = {
        "recipe_id": DISPLAY_RECIPE_ID,
        "source_arrays": [
            "base/relative_distance_physical",
            "base/relative_physical_valid",
            "body/body_bearing_deg",
            "body/body_bearing_valid",
            "base/chaser_occurrence_member",
        ],
        "joint_validity": (
            "panel_member_and_chaser_occurrence_and_relative_physical_valid_"
            "and_body_bearing_valid"
        ),
        "distance_bin_width_mm": DISTANCE_BIN_WIDTH_MM,
        "bearing_bin_width_deg": BEARING_BIN_WIDTH_DEG,
        "distance_bin_edges_mm": distance_edges.tolist(),
        "bearing_bin_edges_deg": bearing_edges.tolist(),
        "density_normalization": "probability_within_panel_chaser",
        "density_color_cmax_quantile": DENSITY_COLOR_CMAX_QUANTILE,
        "density_color_cmax": float(color_max),
        "point_cloud_sampling": "source_order_uniform_including_endpoints",
        "point_cloud_max_rows_per_panel_chaser": (
            INTERACTIVE_POINT_CLOUD_MAX_ROWS_PER_PANEL_CHASER
        ),
        "interpolation": "prohibited",
        "body_axis_fallback": "prohibited",
        "detection_position_substitution": "prohibited",
        "panel_records": panel_records,
    }
    point_cloud.update_layout(
        title=(
            "Anatomical body-frame chaser bearing × distance · exact point rows · "
            f"{projection.recording_id}"
        ),
        height=330 * len(panels),
        meta={**plain(projection.provenance), "display_recipe": recipe},
    )
    density.update_layout(
        title=(
            "Anatomical body-frame chaser bearing × distance density · "
            f"{projection.recording_id}"
        ),
        height=330 * len(panels),
        meta={**plain(projection.provenance), "display_recipe": recipe},
    )
    _polar_layout(point_cloud, radial_max_mm=float(distance_edges[-1]))
    _polar_layout(density, radial_max_mm=float(distance_edges[-1]))
    return mo.vstack(
        [
            mo.callout(
                "Angle is accepted anatomical body-frame bearing; radius is "
                "fish–chaser distance. Density uses every jointly valid exact row. "
                "The point cloud is display-bounded without interpolation.",
                kind="info",
            ),
            point_cloud,
            density,
        ]
    )


__all__ = ["build_exact_body_bearing_distance_output"]
