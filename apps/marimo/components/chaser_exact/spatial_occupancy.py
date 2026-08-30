"""Persisted exact-epoch spatial occupancy heatmaps for paired providers."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from fisheye.visualization.chaser_spatial_occupancy_display import (
    ROBUST_COLOR_QUANTILE,
    SpatialOccupancyDisplaySurface,
    aligned_coarsened_spatial_occupancy_display_surface,
    exact_spatial_occupancy_display_surface,
    shared_robust_color_scale,
)

from .array_requirements import SPATIAL_OCCUPANCY_ARRAYS
from .chaser_locations import (
    CHASER_LOCATION_DISPLAY_RECIPE,
    exact_static_chaser_epoch_locations,
)
from .projection import ExactChaserSuccessorProjection, identity_registry
from .provenance import freeze, plain

SPATIAL_OCCUPANCY_DISPLAY_RECIPE = (
    "paired_provider_exact_epoch_spatial_occupancy_heatmap_v3"
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
    handle.require_verified_arrays(SPATIAL_OCCUPANCY_ARRAYS)
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


def _surface_plot_payload(
    surface: SpatialOccupancyDisplaySurface,
) -> Mapping[str, Any]:
    """Build deterministic Plotly coordinates while retaining exact hover values."""

    density_percent = (
        np.asarray(surface.density_valid_in_arena)
        * SPATIAL_OCCUPANCY_DENSITY_MULTIPLIER
    )
    difference = density_percent[1] - density_percent[0]
    density_scale = shared_robust_color_scale(
        density_percent, quantile=ROBUST_COLOR_QUANTILE
    )
    difference_scale = shared_robust_color_scale(
        np.abs(difference),
        quantile=ROBUST_COLOR_QUANTILE,
        empty_fallback_limit=max(
            density_scale.full_limit * 1e-6,
            float(np.finfo(np.float64).eps),
        ),
    )
    x_edges = np.asarray(surface.x_edges_mm)
    y_edges = np.asarray(surface.y_edges_mm)
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2.0
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2.0
    arena_mask = np.asarray(surface.arena_bin_center_mask, dtype=bool)
    counts = np.asarray(surface.counts, dtype=np.int64)
    bin_width = np.full(
        arena_mask.shape, surface.display_bin_width_mm, dtype=np.float64
    )
    provider_customdata = np.empty((2, 3, *arena_mask.shape, 3), dtype=np.float64)
    difference_customdata = np.empty((3, *arena_mask.shape, 4), dtype=np.float64)
    for provider_index in range(2):
        for epoch_index in range(3):
            provider_customdata[provider_index, epoch_index] = np.stack(
                (
                    counts[provider_index, epoch_index],
                    arena_mask,
                    bin_width,
                ),
                axis=-1,
            )
    for epoch_index in range(3):
        difference_customdata[epoch_index] = np.stack(
            (
                counts[0, epoch_index],
                counts[1, epoch_index],
                arena_mask,
                bin_width,
            ),
            axis=-1,
        )
    trace_z: list[np.ndarray] = []
    trace_customdata: list[np.ndarray] = []
    for epoch_index in range(3):
        for provider_index in range(2):
            trace_z.append(density_percent[provider_index, epoch_index])
            trace_customdata.append(provider_customdata[provider_index, epoch_index])
        trace_z.append(difference[epoch_index])
        trace_customdata.append(difference_customdata[epoch_index])
    return {
        "surface": surface,
        "density_percent": density_percent,
        "difference": difference,
        "density_scale": density_scale,
        "difference_scale": difference_scale,
        "x_centers": x_centers,
        "y_centers": y_centers,
        "trace_x": [x_centers for _ in trace_z],
        "trace_y": [y_centers for _ in trace_z],
        "trace_z": trace_z,
        "trace_customdata": trace_customdata,
    }


def _color_axis(*, robust: bool, payload: Mapping[str, Any]) -> dict[str, Any]:
    scale = payload["density_scale"]
    limit = scale.robust_limit if robust else scale.full_limit
    return {
        "colorscale": "Viridis",
        "cmin": 0.0,
        "cmax": float(limit),
        "colorbar": {"title": "occupancy (% valid in-arena/bin)", "x": 1.01},
    }


def _difference_color_axis(
    *, robust: bool, payload: Mapping[str, Any]
) -> dict[str, Any]:
    scale = payload["difference_scale"]
    limit = scale.robust_limit if robust else scale.full_limit
    return {
        "colorscale": "RdBu",
        "reversescale": True,
        "cmin": -float(limit),
        "cmax": float(limit),
        "colorbar": {"title": "detection − keypoint (pp/bin)", "x": 1.08},
    }


def build_exact_spatial_occupancy_output(
    mo: Any,
    go: Any,
    projection: ExactChaserSuccessorProjection,
) -> Any:
    """Render sealed provider/epoch occupancy and detection-minus-keypoint."""

    from plotly.subplots import make_subplots

    values = _spatial_values(projection)
    chaser_locations = exact_static_chaser_epoch_locations(projection)
    canonical_surface = exact_spatial_occupancy_display_surface(
        counts=np.asarray(values["counts"]),
        density_valid_in_arena=np.asarray(values["density"]),
        fraction_candidate_epoch=np.asarray(values["candidate_fraction"]),
        x_edges_mm=np.asarray(values["x_edges"]),
        y_edges_mm=np.asarray(values["y_edges"]),
        arena_bin_center_mask=np.asarray(values["arena_mask"]),
    )
    declared_bin_width = float(values["grid"].get("bin_width_mm", np.nan))
    if not np.isclose(
        canonical_surface.source_bin_width_mm,
        declared_bin_width,
        rtol=0.0,
        atol=1e-10,
    ) or not np.isclose(declared_bin_width, 2.0, rtol=0.0, atol=1e-10):
        raise ValueError(
            "The aligned 4 mm display requires one sealed uniform 2 mm source grid."
        )
    radius_mm = float(values["arena"]["radius_mm"])
    coarse_surface = aligned_coarsened_spatial_occupancy_display_surface(
        canonical_surface,
        factor=2,
        in_arena_denominator=np.asarray(values["in_arena"]),
        candidate_denominator=np.asarray(values["candidate"]),
        arena_radius_mm=radius_mm,
    )
    if not np.isclose(coarse_surface.display_bin_width_mm, 4.0, rtol=0.0, atol=1e-10):
        raise ValueError("Aligned spatial display did not produce exact 4 mm bins.")
    payloads = {
        "2mm": _surface_plot_payload(canonical_surface),
        "4mm": _surface_plot_payload(coarse_surface),
    }
    default_payload = payloads["2mm"]
    x_edges = np.asarray(canonical_surface.x_edges_mm)
    y_edges = np.asarray(canonical_surface.y_edges_mm)
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
    density_percent = np.asarray(default_payload["density_percent"])
    difference = np.asarray(default_payload["difference"])
    x_centers = np.asarray(default_payload["x_centers"])
    y_centers = np.asarray(default_payload["y_centers"])
    for epoch_index in range(3):
        for provider_index in range(2):
            trace_index = epoch_index * 3 + provider_index
            figure.add_trace(
                go.Heatmap(
                    x=x_centers,
                    y=y_centers,
                    z=density_percent[provider_index, epoch_index],
                    customdata=default_payload["trace_customdata"][trace_index],
                    coloraxis="coloraxis",
                    hovertemplate=(
                        "x=%{x:.2f} mm<br>y=%{y:.2f} mm<br>"
                        "occupancy=%{z:.4f}%/bin<br>"
                        "exact count=%{customdata[0]:,.0f}<br>"
                        "bin center inside reviewed circle=%{customdata[1]:.0f}<br>"
                        "display bin width=%{customdata[2]:g} mm<extra></extra>"
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
                customdata=default_payload["trace_customdata"][epoch_index * 3 + 2],
                coloraxis="coloraxis2",
                hovertemplate=(
                    "x=%{x:.2f} mm<br>y=%{y:.2f} mm<br>"
                    "detection − keypoint=%{z:.4f} pp/bin<br>"
                    "keypoint exact count=%{customdata[0]:,.0f}<br>"
                    "detection exact count=%{customdata[1]:,.0f}<br>"
                    "bin center inside reviewed circle=%{customdata[2]:.0f}<br>"
                    "display bin width=%{customdata[3]:g} mm<extra></extra>"
                ),
            ),
            row=3,
            col=epoch_index + 1,
        )
    for location in chaser_locations:
        appearance = location.appearance
        label = (
            f"{appearance.behavior_role} · protocol chaser "
            f"{appearance.chaser_index}"
        )
        for row in range(1, 4):
            figure.add_trace(
                go.Scatter(
                    x=[location.x_mm],
                    y=[location.y_mm],
                    mode="markers",
                    name=label,
                    legendgroup=f"chaser-{appearance.identity_code}",
                    showlegend=(location.analysis_role == "chaser_pre" and row == 1),
                    marker={
                        "color": appearance.experimental_color_css,
                        "symbol": appearance.plotly_role_symbol,
                        "size": 14,
                        "line": {
                            "color": appearance.contrast_outline_hex,
                            "width": 1.5,
                        },
                    },
                    hovertemplate=(
                        f"{location.analysis_role} · {appearance.behavior_role}<br>"
                        f"protocol chaser {appearance.chaser_index}<br>"
                        f"identity={appearance.identity}<br>"
                        f"experimental color={appearance.experimental_color_hex}<br>"
                        "median x=%{x:.2f} mm<br>median y=%{y:.2f} mm<br>"
                        f"valid logged samples={location.sample_count:,}<br>"
                        f"p95 drift={location.p95_drift_mm:.3g} mm<br>"
                        f"maximum drift={location.maximum_drift_mm:.3g} mm"
                        "<extra></extra>"
                    ),
                ),
                row=row,
                col=location.epoch_index + 1,
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
    heatmap_trace_indices = list(range(9))
    display_buttons = []
    for resolution, payload in payloads.items():
        surface = payload["surface"]
        for robust in (True, False):
            scale_label = (
                f"robust p{int(round(ROBUST_COLOR_QUANTILE * 100))}"
                if robust
                else "full range"
            )
            display_buttons.append(
                {
                    "label": f"{surface.display_bin_width_mm:g} mm · {scale_label}",
                    "method": "update",
                    "args": [
                        {
                            "x": payload["trace_x"],
                            "y": payload["trace_y"],
                            "z": payload["trace_z"],
                            "customdata": payload["trace_customdata"],
                        },
                        {
                            "coloraxis": _color_axis(robust=robust, payload=payload),
                            "coloraxis2": _difference_color_axis(
                                robust=robust, payload=payload
                            ),
                            "title.text": (
                                "Exact protocol-semantic spatial occupancy · "
                                f"{projection.recording_id}<br>"
                                f"{surface.display_bin_width_mm:g} mm display · "
                                f"{scale_label} shared color scale"
                            ),
                        },
                        heatmap_trace_indices,
                    ],
                }
            )
    display_parameters = {
        "recipe_id": SPATIAL_OCCUPANCY_DISPLAY_RECIPE,
        "source_array": SPATIAL_OCCUPANCY_SOURCE_ARRAY,
        "source_count_array": "occupancy_count",
        "density_multiplier_to_percent": SPATIAL_OCCUPANCY_DENSITY_MULTIPLIER,
        "provider_difference": ("detection_minus_keypoint_percentage_points_per_bin"),
        "default_display_mode": "2_mm_robust_p98",
        "available_display_modes": [
            "2_mm_robust_p98",
            "2_mm_full_range",
            "4_mm_robust_p98",
            "4_mm_full_range",
        ],
        "display_surfaces": {
            resolution: {
                **payload["surface"].provenance_record(),
                "density_color_scale_percent_per_bin": payload[
                    "density_scale"
                ].provenance_record(),
                "difference_color_scale_absolute_percentage_points_per_bin": payload[
                    "difference_scale"
                ].provenance_record(),
            }
            for resolution, payload in payloads.items()
        },
        "coverage_annotation_array": SPATIAL_OCCUPANCY_COVERAGE_ARRAY,
        "arena_bin_center_mask_role": (
            "hover_evidence_only_bins_not_discarded_boundary_bins_may_straddle_circle"
        ),
        "coordinate_orientation": "+x_right_+y_down",
        "interpolation": "prohibited",
        "scientific_recomputation": False,
        "display_only_derivation": (
            "optional_aligned_2x2_exact_count_sum_then_persisted_denominator_normalization"
        ),
        "color_saturation_semantics": (
            "display_only_exact_hover_values_and_full_range_reference_retained"
        ),
        "chaser_location_overlay": {
            "recipe_id": CHASER_LOCATION_DISPLAY_RECIPE,
            "epochs": ["chaser_pre", "chaser_post"],
            "position_summary": "median_of_exact_valid_logged_rows",
            "training_location_summary": "omitted_dynamic_trajectory_available_separately",
            "color_source": "sealed_protocol_rgba",
            "role_encoding": "independent_marker_symbol_and_legend_text",
            "identity_encoding": "protocol_chaser_index_and_exact_identity_hover",
            "index_or_role_color_fallback": "prohibited",
            "locations": [
                location.provenance_record() for location in chaser_locations
            ],
        },
    }
    figure_meta = plain(projection.provenance)
    figure_meta["spatial_occupancy_display"] = display_parameters
    figure.update_layout(
        title=(
            f"Exact protocol-semantic spatial occupancy · {projection.recording_id}"
            "<br>2 mm display · robust p98 shared color scale"
        ),
        height=1_150,
        coloraxis=_color_axis(robust=True, payload=default_payload),
        coloraxis2=_difference_color_axis(robust=True, payload=default_payload),
        updatemenus=[
            {
                "type": "dropdown",
                "active": 0,
                "buttons": display_buttons,
                "x": 0.0,
                "xanchor": "left",
                "y": 1.12,
                "yanchor": "top",
            }
        ],
        meta=figure_meta,
    )
    return mo.vstack(
        [
            mo.callout(
                "The default robust p98 color scale reveals broad occupancy while saturating only the recorded high-end bins; exact values remain in hover and the full-range reference is available in the figure menu. The canonical 2 mm surface is unchanged. The optional 4 mm view is an aligned exact 2×2 count sum with conserved counts and the same persisted denominators—never interpolation or a replacement scientific authority. Pre/post chaser marker fill is sealed protocol color; shape and text independently encode role. The moving training trajectory remains in the trajectory view.",
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
