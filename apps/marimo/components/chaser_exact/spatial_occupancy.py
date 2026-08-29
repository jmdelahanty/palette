"""Persisted exact-epoch spatial occupancy heatmaps for paired providers."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .projection import ExactChaserSuccessorProjection, identity_registry
from .provenance import freeze, plain

SPATIAL_OCCUPANCY_DISPLAY_RECIPE = (
    "paired_provider_exact_epoch_spatial_occupancy_heatmap_v2"
)
SPATIAL_OCCUPANCY_DENSITY_MULTIPLIER = 100.0
SPATIAL_OCCUPANCY_SOURCE_ARRAY = "occupancy_density_valid_in_arena"
SPATIAL_OCCUPANCY_COVERAGE_ARRAY = "in_arena_coverage_fraction_candidate"


def _array(handle: Any, name: str, *, dtype: Any) -> np.ndarray:
    try:
        return np.asarray(handle.array(name), dtype=dtype)
    except KeyError as exc:
        raise ValueError(
            f"Spatial occupancy lacks required persisted array {name!r}."
        ) from exc


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"Spatial occupancy lacks {label}.")
    return value


def _spatial_values(
    projection: ExactChaserSuccessorProjection,
) -> Mapping[str, Any]:
    """Validate sealed display arrays without deriving scientific occupancy."""

    handle = projection.spatial
    if handle.successor_kind != "chaser_spatial_occupancy":
        raise ValueError("Projection anchor is not spatial occupancy.")
    if handle.deep_audited is not True:
        raise ValueError("Spatial occupancy display requires a deep content audit.")
    scientific = handle.scientific_manifest
    dimensions = _mapping(scientific.get("dimensions"), label="dimensions")
    expected = (
        int(dimensions.get("n_providers", 0)),
        int(dimensions.get("n_epochs", 0)),
        int(dimensions.get("grid_rows", 0)),
        int(dimensions.get("grid_columns", 0)),
    )
    if expected[:2] != (2, 3) or expected[2] <= 0 or expected[3] <= 0:
        raise ValueError(
            "Spatial occupancy display requires two providers and three exact epochs."
        )

    counts = _array(handle, "occupancy_count", dtype=np.int64)
    density = _array(handle, "occupancy_density_valid_in_arena", dtype=np.float64)
    candidate_fraction = _array(
        handle, "occupancy_fraction_candidate_epoch", dtype=np.float64
    )
    for name, values in (
        ("occupancy_count", counts),
        ("occupancy_density_valid_in_arena", density),
        ("occupancy_fraction_candidate_epoch", candidate_fraction),
    ):
        if values.shape != expected:
            raise ValueError(f"Persisted {name} has inconsistent dimensions.")
    if (
        np.any(counts < 0)
        or not np.all(np.isfinite(density))
        or not np.all(np.isfinite(candidate_fraction))
        or np.any(density < 0)
        or np.any(candidate_fraction < 0)
    ):
        raise ValueError("Spatial occupancy grids contain invalid values.")

    x_edges = _array(handle, "x_bin_edges_mm", dtype=np.float64)
    y_edges = _array(handle, "y_bin_edges_mm", dtype=np.float64)
    arena_mask = _array(handle, "arena_bin_center_mask", dtype=bool)
    if (
        x_edges.shape != (expected[3] + 1,)
        or y_edges.shape != (expected[2] + 1,)
        or np.any(~np.isfinite(x_edges))
        or np.any(~np.isfinite(y_edges))
        or np.any(np.diff(x_edges) <= 0)
        or np.any(np.diff(y_edges) <= 0)
        or arena_mask.shape != expected[2:]
    ):
        raise ValueError("Spatial occupancy bin coordinates or arena mask are invalid.")

    denominator_names = (
        "candidate_frame_count",
        "declared_valid_position_frame_count",
        "finite_valid_position_frame_count",
        "in_arena_position_frame_count",
        "invalid_position_frame_count",
        "out_of_arena_position_frame_count",
    )
    denominators = {
        name: _array(handle, name, dtype=np.int64) for name in denominator_names
    }
    coverage = _array(handle, "in_arena_coverage_fraction_candidate", dtype=np.float64)
    in_arena_fraction_valid = _array(
        handle, "in_arena_fraction_finite_valid", dtype=np.float64
    )
    denominator_shape = expected[:2]
    if any(values.shape != denominator_shape for values in denominators.values()):
        raise ValueError("Spatial occupancy denominator dimensions are invalid.")
    if (
        coverage.shape != denominator_shape
        or in_arena_fraction_valid.shape != denominator_shape
    ):
        raise ValueError("Spatial occupancy coverage dimensions are invalid.")
    if any(np.any(values < 0) for values in denominators.values()):
        raise ValueError("Spatial occupancy denominators contain negative counts.")
    candidate = denominators["candidate_frame_count"]
    declared = denominators["declared_valid_position_frame_count"]
    finite = denominators["finite_valid_position_frame_count"]
    in_arena = denominators["in_arena_position_frame_count"]
    invalid = denominators["invalid_position_frame_count"]
    out_of_arena = denominators["out_of_arena_position_frame_count"]
    if (
        np.any(candidate <= 0)
        or np.any(declared > candidate)
        or np.any(finite > declared)
        or not np.array_equal(finite + invalid, candidate)
        or not np.array_equal(in_arena + out_of_arena, finite)
        or not np.array_equal(counts.sum(axis=(2, 3)), in_arena)
    ):
        raise ValueError("Spatial occupancy denominator conservation failed.")
    expected_density_sum = (in_arena > 0).astype(np.float64)
    expected_coverage = in_arena / candidate
    expected_valid_fraction = np.divide(
        in_arena,
        finite,
        out=np.zeros_like(coverage),
        where=finite > 0,
    )
    if (
        not np.allclose(
            density.sum(axis=(2, 3)),
            expected_density_sum,
            rtol=1e-10,
            atol=1e-12,
        )
        or not np.allclose(
            candidate_fraction.sum(axis=(2, 3)),
            expected_coverage,
            rtol=1e-10,
            atol=1e-12,
        )
        or not np.allclose(coverage, expected_coverage, rtol=1e-10, atol=1e-12)
        or not np.allclose(
            in_arena_fraction_valid,
            expected_valid_fraction,
            rtol=1e-10,
            atol=1e-12,
        )
    ):
        raise ValueError("Spatial occupancy persisted normalization is inconsistent.")

    provider_registry = identity_registry(scientific, "provider_role")
    epoch_registry = identity_registry(scientific, "epoch_role")
    if provider_registry != {"0": "keypoint", "1": "detection"}:
        raise ValueError("Spatial occupancy provider order is not keypoint/detection.")
    if epoch_registry != {
        "0": "chaser_pre",
        "1": "chaser_training",
        "2": "chaser_post",
    }:
        raise ValueError("Spatial occupancy epoch order is not pre/training/post.")
    arena = _mapping(scientific.get("arena"), label="reviewed arena")
    grid = _mapping(scientific.get("grid"), label="grid policy")
    if (
        grid.get("coordinate_orientation") != "+x_right_+y_down"
        or grid.get("normalization_policy_id")
        != "valid_in_arena_and_candidate_epoch_denominators_v1"
        or float(arena.get("radius_mm", 0.0)) <= 0
    ):
        raise ValueError("Spatial occupancy grid or arena policy is unsupported.")
    return freeze(
        {
            "counts": counts,
            "density": density,
            "candidate_fraction": candidate_fraction,
            "x_edges": x_edges,
            "y_edges": y_edges,
            "arena_mask": arena_mask,
            "candidate": candidate,
            "in_arena": in_arena,
            "coverage": coverage,
            "provider_registry": provider_registry,
            "epoch_registry": epoch_registry,
            "arena": arena,
            "grid": grid,
        }
    )


def build_exact_spatial_occupancy_output(
    mo: Any,
    go: Any,
    projection: ExactChaserSuccessorProjection,
) -> Any:
    """Render sealed provider/epoch occupancy and detection-minus-keypoint."""

    from plotly.subplots import make_subplots

    values = _spatial_values(projection)
    density_percent = (
        np.asarray(values["density"]) * SPATIAL_OCCUPANCY_DENSITY_MULTIPLIER
    )
    difference = density_percent[1] - density_percent[0]
    density_max = float(np.max(density_percent))
    if not np.isfinite(density_max) or density_max <= 0:
        raise ValueError("Spatial occupancy has no positive persisted density.")
    difference_limit = float(np.max(np.abs(difference)))
    if difference_limit <= 0:
        difference_limit = max(
            density_max * 1e-6,
            float(np.finfo(np.float64).eps),
        )
    x_edges = np.asarray(values["x_edges"])
    y_edges = np.asarray(values["y_edges"])
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2.0
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2.0
    arena_mask = np.asarray(values["arena_mask"], dtype=bool)
    radius_mm = float(values["arena"]["radius_mm"])
    epoch_labels = ("pre", "training", "post")
    titles = []
    for row in range(3):
        for epoch_index, epoch_label in enumerate(epoch_labels):
            if row < 2:
                coverage = 100.0 * float(values["coverage"][row, epoch_index])
                titles.append(
                    f"{epoch_label} · coverage {coverage:.2f}% · "
                    f"n={int(values['in_arena'][row, epoch_index]):,}/"
                    f"{int(values['candidate'][row, epoch_index]):,}"
                )
            else:
                titles.append(f"{epoch_label} · detection − keypoint")
    figure = make_subplots(rows=3, cols=3, subplot_titles=titles)
    for epoch_index in range(3):
        for provider_index in range(2):
            figure.add_trace(
                go.Heatmap(
                    x=x_centers,
                    y=y_centers,
                    z=density_percent[provider_index, epoch_index],
                    customdata=arena_mask,
                    coloraxis="coloraxis",
                    hovertemplate=(
                        "x=%{x:.2f} mm<br>y=%{y:.2f} mm<br>"
                        "occupancy=%{z:.4f}%/bin<br>"
                        "bin center inside reviewed circle=%{customdata}<extra></extra>"
                    ),
                ),
                row=provider_index + 1,
                col=epoch_index + 1,
            )
        figure.add_trace(
            go.Heatmap(
                x=x_centers,
                y=y_centers,
                z=difference[epoch_index],
                customdata=arena_mask,
                coloraxis="coloraxis2",
                hovertemplate=(
                    "x=%{x:.2f} mm<br>y=%{y:.2f} mm<br>"
                    "detection − keypoint=%{z:.4f} pp/bin<br>"
                    "bin center inside reviewed circle=%{customdata}<extra></extra>"
                ),
            ),
            row=3,
            col=epoch_index + 1,
        )
    for row in range(1, 4):
        for column in range(1, 4):
            axis_index = (row - 1) * 3 + column
            axis_suffix = "" if axis_index == 1 else str(axis_index)
            figure.add_shape(
                type="circle",
                x0=-radius_mm,
                x1=radius_mm,
                y0=-radius_mm,
                y1=radius_mm,
                line={"color": "white" if row < 3 else "black", "width": 1},
                row=row,
                col=column,
            )
            figure.update_xaxes(
                range=[float(x_edges[0]), float(x_edges[-1])],
                title_text="x from arena center (mm)",
                row=row,
                col=column,
            )
            figure.update_yaxes(
                range=[float(y_edges[-1]), float(y_edges[0])],
                title_text="y from arena center (mm; +down)" if column == 1 else None,
                scaleanchor=f"x{axis_suffix}",
                scaleratio=1,
                row=row,
                col=column,
            )
    display_parameters = {
        "recipe_id": SPATIAL_OCCUPANCY_DISPLAY_RECIPE,
        "source_array": SPATIAL_OCCUPANCY_SOURCE_ARRAY,
        "density_multiplier_to_percent": SPATIAL_OCCUPANCY_DENSITY_MULTIPLIER,
        "density_color_limits_percent_per_bin": [0.0, density_max],
        "provider_difference": ("detection_minus_keypoint_percentage_points_per_bin"),
        "difference_color_limits_percentage_points_per_bin": [
            -difference_limit,
            difference_limit,
        ],
        "coverage_annotation_array": SPATIAL_OCCUPANCY_COVERAGE_ARRAY,
        "x_bin_edges_mm": x_edges.tolist(),
        "y_bin_edges_mm": y_edges.tolist(),
        "arena_bin_center_mask_role": (
            "hover_evidence_only_bins_not_discarded_boundary_bins_may_straddle_circle"
        ),
        "coordinate_orientation": "+x_right_+y_down",
        "interpolation": "prohibited",
        "scientific_recomputation": False,
    }
    figure_meta = plain(projection.provenance)
    figure_meta["spatial_occupancy_display"] = display_parameters
    figure.update_layout(
        title=(
            "Exact protocol-semantic spatial occupancy · " f"{projection.recording_id}"
        ),
        height=1_150,
        coloraxis={
            "colorscale": "Viridis",
            "cmin": 0.0,
            "cmax": density_max,
            "colorbar": {"title": "occupancy (% valid in-arena/bin)", "x": 1.01},
        },
        coloraxis2={
            "colorscale": "RdBu",
            "reversescale": True,
            "cmin": -difference_limit,
            "cmax": difference_limit,
            "colorbar": {"title": "detection − keypoint (pp/bin)", "x": 1.08},
        },
        meta=figure_meta,
    )
    return mo.vstack(
        [
            mo.callout(
                "Persisted conditional valid-in-arena occupancy on the sealed shared physical grid; coverage retains missing and out-of-arena evidence. No bins are recomputed or interpolated.",
                kind="info",
            ),
            figure,
        ]
    )


__all__ = [
    "SPATIAL_OCCUPANCY_COVERAGE_ARRAY",
    "SPATIAL_OCCUPANCY_DENSITY_MULTIPLIER",
    "SPATIAL_OCCUPANCY_DISPLAY_RECIPE",
    "SPATIAL_OCCUPANCY_SOURCE_ARRAY",
    "build_exact_spatial_occupancy_output",
]
