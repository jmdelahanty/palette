from __future__ import annotations

import numpy as np
import pytest

from fisheye.visualization.chaser_spatial_occupancy_display import (
    DEFAULT_DISPLAY_MODE_ID,
    DISPLAY_RECIPE_ID,
    STATIC_EXPORT_MODE_IDS,
    ChaserSpatialOccupancyDisplayError,
    aligned_coarsened_spatial_occupancy_display_surface,
    build_spatial_occupancy_display_plan,
    exact_spatial_occupancy_display_surface,
    shared_robust_color_scale,
    spatial_occupancy_display_contract_record,
)


def _surface():
    counts = np.arange(2 * 3 * 4 * 4, dtype=np.int64).reshape(2, 3, 4, 4)
    in_arena = counts.sum(axis=(-2, -1), dtype=np.int64)
    candidate = in_arena + 100
    density = np.divide(
        counts,
        in_arena[..., None, None],
        out=np.zeros_like(counts, dtype=np.float64),
        where=in_arena[..., None, None] > 0,
    )
    fraction = counts / candidate[..., None, None]
    edges = np.asarray([-4.0, -2.0, 0.0, 2.0, 4.0])
    centers = (edges[:-1] + edges[1:]) / 2.0
    x, y = np.meshgrid(centers, centers)
    surface = exact_spatial_occupancy_display_surface(
        counts=counts,
        density_valid_in_arena=density,
        fraction_candidate_epoch=fraction,
        x_edges_mm=edges,
        y_edges_mm=edges,
        arena_bin_center_mask=np.hypot(x, y) <= 4.0,
    )
    return surface, counts, in_arena, candidate


def test_aligned_four_mm_coarsening_conserves_exact_counts() -> None:
    surface, counts, in_arena, candidate = _surface()

    coarse = aligned_coarsened_spatial_occupancy_display_surface(
        surface,
        factor=2,
        in_arena_denominator=in_arena,
        candidate_denominator=candidate,
        arena_radius_mm=4.0,
    )

    assert coarse.display_bin_width_mm == 4.0
    assert coarse.counts.shape == (2, 3, 2, 2)
    assert np.array_equal(coarse.counts.sum(axis=(-2, -1)), counts.sum(axis=(-2, -1)))
    assert np.allclose(
        coarse.density_valid_in_arena.sum(axis=(-2, -1)),
        (in_arena > 0).astype(float),
    )
    assert np.allclose(
        coarse.fraction_candidate_epoch.sum(axis=(-2, -1)),
        in_arena / candidate,
    )
    assert not coarse.counts.flags.writeable
    provenance = coarse.provenance_record()
    assert provenance["count_aggregation"] == "exact_2x2_sum"
    assert provenance["scientific_authority"] is False
    assert provenance["interpolation"] == "prohibited"


def test_coarsening_rejects_nonconserving_denominator() -> None:
    surface, _counts, in_arena, candidate = _surface()
    changed = in_arena.copy()
    changed[0, 0] += 1

    with pytest.raises(ChaserSpatialOccupancyDisplayError, match="denominators"):
        aligned_coarsened_spatial_occupancy_display_surface(
            surface,
            factor=2,
            in_arena_denominator=changed,
            candidate_denominator=candidate,
            arena_radius_mm=4.0,
        )


def test_robust_scale_retains_full_reference_and_clipping_count() -> None:
    scale = shared_robust_color_scale(np.asarray([0.0, 1.0, 2.0, 100.0]), quantile=0.5)

    assert scale.robust_limit == 2.0
    assert scale.full_limit == 100.0
    assert scale.positive_bin_count == 3
    assert scale.bins_above_robust_limit == 1
    assert scale.provenance_record()["full_range_reference_available"] is True


def test_zero_difference_scale_requires_explicit_display_fallback() -> None:
    with pytest.raises(ChaserSpatialOccupancyDisplayError, match="no positive"):
        shared_robust_color_scale(np.zeros(4))

    scale = shared_robust_color_scale(
        np.zeros(4), empty_fallback_limit=np.finfo(np.float64).eps
    )
    assert scale.positive_bin_count == 0
    assert scale.robust_limit == np.finfo(np.float64).eps
    assert scale.empty_fallback_limit == np.finfo(np.float64).eps


def test_shared_plan_is_the_complete_renderer_neutral_display_contract() -> None:
    surface, counts, in_arena, candidate = _surface()

    plan = build_spatial_occupancy_display_plan(
        counts=counts,
        density_valid_in_arena=surface.density_valid_in_arena,
        fraction_candidate_epoch=surface.fraction_candidate_epoch,
        x_edges_mm=surface.x_edges_mm,
        y_edges_mm=surface.y_edges_mm,
        arena_bin_center_mask=surface.arena_bin_center_mask,
        in_arena_denominator=in_arena,
        candidate_denominator=candidate,
        arena_radius_mm=4.0,
        declared_bin_width_mm=2.0,
    )

    assert plan.default_mode.mode_id == DEFAULT_DISPLAY_MODE_ID
    assert plan.default_mode.payload.surface.display_bin_width_mm == 4.0
    assert plan.default_mode.robust is True
    assert tuple(mode.mode_id for mode in plan.modes) == tuple(
        spatial_occupancy_display_contract_record()["available_display_modes"]
    )
    assert tuple(STATIC_EXPORT_MODE_IDS) == (
        "4_mm_valid_in_arena_robust_p98",
        "4_mm_valid_in_arena_full_range",
    )
    provenance = plan.provenance_record()
    assert provenance["recipe_id"] == DISPLAY_RECIPE_ID
    assert provenance["default_display_mode"] == DEFAULT_DISPLAY_MODE_ID
    assert (
        provenance["display_surfaces"]["4mm_valid_in_arena"]["count_aggregation"]
        == "exact_2x2_sum"
    )
    assert (
        provenance["display_surfaces"]["2mm_valid_in_arena"]["count_aggregation"]
        == "none"
    )
    assert plan.default_mode.value_limit <= (
        plan.default_mode.payload.value_scale.full_limit
    )


def test_shared_plan_rejects_persisted_density_that_differs_from_counts() -> None:
    surface, counts, in_arena, candidate = _surface()
    changed = np.asarray(surface.density_valid_in_arena).copy()
    changed[0, 0, 0, 0] += 0.01

    with pytest.raises(
        ChaserSpatialOccupancyDisplayError,
        match="normalizations differ",
    ):
        build_spatial_occupancy_display_plan(
            counts=counts,
            density_valid_in_arena=changed,
            fraction_candidate_epoch=surface.fraction_candidate_epoch,
            x_edges_mm=surface.x_edges_mm,
            y_edges_mm=surface.y_edges_mm,
            arena_bin_center_mask=surface.arena_bin_center_mask,
            in_arena_denominator=in_arena,
            candidate_denominator=candidate,
            arena_radius_mm=4.0,
            declared_bin_width_mm=2.0,
        )
