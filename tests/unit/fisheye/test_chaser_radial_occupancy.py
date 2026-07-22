from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np
import pytest
import zarr
from fisheye.analysis.chaser_distance_runs import (
    ChaserDistanceResult,
    ChaserDistanceWindow,
    write_chaser_distance_run,
)
from fisheye.analysis.chaser_radial_occupancy import (
    COMPONENT_PARENT_NAME,
    DEFAULT_COMPONENT_NAME,
    INTERACTIVE_ARTIFACT_NAME,
    RADIAL_DENSITY_PNG_ARTIFACT_NAME,
    SCHEMA_ID,
    SELECTION_INDEX_PNG_ARTIFACT_NAME,
    _disc_overlap_area_mm2,
    build_chaser_radial_occupancy_result,
    write_chaser_radial_occupancy_component,
)
from fisheye.shared.json_safety import decode_null_terminated_text
from fisheye.shared.plot_artifacts import INTERACTIVE_SPEC_SCHEMA_ID, PNG_ARTIFACT_SCHEMA_ID


pytestmark = pytest.mark.usefixtures("logical_chaser_distance_reader")


PPM = 2.0  # pixels per mm
ARENA_CENTER_PX = 100.0
ARENA_RADIUS_PX = 80.0
ARENA_RADIUS_MM = ARENA_RADIUS_PX / PPM  # 40 mm
ARENA_CENTER_MM = ARENA_CENTER_PX / PPM  # 50 mm


def _decode_first(array: zarr.Array) -> str:
    return decode_null_terminated_text(np.asarray(array[0], dtype=np.uint8)).strip()


def _sample_uniform_in_disc(rng: np.random.Generator, n: int, *, radius_mm: float) -> np.ndarray:
    """Uniform over disc area (not over radius): r = R*sqrt(u)."""

    radius = float(radius_mm) * np.sqrt(rng.random(n))
    theta = rng.random(n) * 2.0 * math.pi
    return np.stack(
        [ARENA_CENTER_MM + radius * np.cos(theta), ARENA_CENTER_MM + radius * np.sin(theta)],
        axis=1,
    )


def _make_archive(
    tmp_path: Path,
    *,
    fish_xy_mm: np.ndarray,
    chaser_xy_mm: np.ndarray,
    windows: Sequence[ChaserDistanceWindow] | None = None,
    fps: float = 10.0,
    fish_valid: np.ndarray | None = None,
    name: str = "radial_analysis.zarr",
) -> Path:
    """Build a minimal analysis archive with a circular arena and one chaser-distance run.

    fish_xy_mm: (frames, 2); chaser_xy_mm: (frames, n_chasers, 2) -- both in mm.
    """

    zarr_path = tmp_path / name
    root = zarr.open_group(str(zarr_path), mode="w")
    root.attrs["recording_id"] = "test_radial"
    analysis = root.create_group("analysis")
    stimulus_parent = analysis.create_group("stimulus_runs")
    stimulus_parent.attrs["latest"] = "stimulus_1"
    stimulus_parent.attrs["latest_complete"] = "stimulus_1"
    stimulus = stimulus_parent.create_group("stimulus_1")
    geometry = stimulus.require_group("calibration").require_group("arena_geometry")
    geometry.attrs.update(
        {
            "coordinate_frame": "arena_relative_canvas_px",
            "coordinate_origin": "top_left_of_active_arena",
            "experimental_area_shape": "CIRCLE",
            "experimental_area_center_x_px": ARENA_CENTER_PX,
            "experimental_area_center_y_px": ARENA_CENTER_PX,
            "experimental_area_radius_px": ARENA_RADIUS_PX,
            "experimental_area_radius_mm": ARENA_RADIUS_MM,
            "arena_region_width_px": 2.0 * ARENA_CENTER_PX,
            "arena_region_height_px": 2.0 * ARENA_CENTER_PX,
        }
    )

    fish_px = np.asarray(fish_xy_mm, dtype=np.float32) * PPM
    chaser_px = np.asarray(chaser_xy_mm, dtype=np.float32) * PPM
    n = int(fish_px.shape[0])
    n_chasers = int(chaser_px.shape[1])
    distance_px = np.linalg.norm(fish_px[:, None, :] - chaser_px, axis=2).astype(np.float32)
    distance_mm = (distance_px / PPM).astype(np.float32)
    if windows is None:
        windows = (ChaserDistanceWindow(0, "all_frames", 0, n - 1, 0.0, n / fps, n / fps),)
    valid = np.ones(n, dtype=bool) if fish_valid is None else np.asarray(fish_valid, dtype=bool)
    n_w = len(windows)
    zeros_wc = np.zeros((n_w, n_chasers), dtype=np.float32)

    result = ChaserDistanceResult(
        zarr_path=str(zarr_path),
        recording_id="test_radial",
        run_name="chaser_distance_1",
        source_detection_path="refined_detect_runs/refined_1/instances",
        source_detection_kind="refined",
        source_stimulus_run="stimulus_1",
        source_stimulus_path="analysis/stimulus_runs/stimulus_1",
        source_stimulus_epoch_run="epochs_1",
        source_stimulus_epoch_path="analysis/stimulus_epoch_runs/epochs_1",
        fps=float(fps),
        total_frames=n,
        pixels_per_mm_projector=PPM,
        coordinate_frame="arena_relative_canvas_px",
        coordinate_origin="top_left_of_active_arena",
        arena_origin_in_canvas_xy=(0.0, 0.0),
        chaser_indices=np.arange(n_chasers, dtype=np.uint8),
        chaser_behavior_class_id=np.asarray([1] + [3] * (n_chasers - 1), dtype=np.int8),
        chaser_behavior_labels=tuple(["aggressive"] + ["inert"] * (n_chasers - 1)),
        camera_frame_id=np.arange(n, dtype=np.int64),
        stimulus_frame_num=np.arange(n, dtype=np.int64),
        timestamp_ns=np.arange(n, dtype=np.int64),
        stimulus_epoch_window_id=np.zeros(n, dtype=np.int32),
        fish_centroid_img_xy=fish_px,
        fish_centroid_arena_xy=fish_px,
        chaser_arena_xy=chaser_px,
        fish_valid=valid,
        chaser_valid=np.ones((n, n_chasers), dtype=bool),
        distance_px=distance_px,
        distance_mm=distance_mm,
        nearest_chaser_index=np.argmin(distance_mm, axis=1).astype(np.int16),
        nearest_distance_mm=np.min(distance_mm, axis=1).astype(np.float32),
        windows=tuple(windows),
        epoch_valid_frame_count=np.full((n_w, n_chasers), n, dtype=np.int64),
        epoch_mean_distance_mm=zeros_wc,
        epoch_min_distance_mm=zeros_wc,
        epoch_p05_distance_mm=zeros_wc,
        epoch_p50_distance_mm=zeros_wc,
        epoch_p95_distance_mm=zeros_wc,
        epoch_fraction_within_threshold=zeros_wc,
        threshold_mm=20.0,
        distribution_bin_width_mm=2.0,
        histogram_bin_edges_mm=np.asarray([0.0, 2.0, 4.0], dtype=np.float32),
        histogram_bin_centers_mm=np.asarray([1.0, 3.0], dtype=np.float32),
        histogram_counts=np.zeros((n_w, n_chasers, 2), dtype=np.uint32),
        histogram_density=np.zeros((n_w, n_chasers, 2), dtype=np.float32),
    )
    write_chaser_distance_run(
        zarr_path,
        result,
        overwrite=True,
        legacy_compatibility=True,
    )
    return zarr_path


# --------------------------------------------------------------------------------------
# Closed-form annulus area
# --------------------------------------------------------------------------------------


def test_disc_overlap_area_matches_analytic_cases() -> None:
    arena_radius = 40.0

    # Ring disc entirely inside the arena -> full disc area.
    contained = _disc_overlap_area_mm2(5.0, np.asarray([0.0, 10.0, 34.9]), arena_radius)
    assert np.allclose(contained, math.pi * 25.0)

    # Ring disc entirely outside the arena -> zero.
    disjoint = _disc_overlap_area_mm2(5.0, np.asarray([45.1, 100.0]), arena_radius)
    assert np.allclose(disjoint, 0.0)

    # Ring disc engulfs the arena -> arena area.
    engulfing = _disc_overlap_area_mm2(100.0, np.asarray([0.0, 5.0]), arena_radius)
    assert np.allclose(engulfing, math.pi * arena_radius**2)

    # Two equal circles whose centres coincide with each other's rim (d == r == R):
    # the classic lens area is 2*R^2*(pi/3) - (sqrt(3)/2)*R^2.
    lens = _disc_overlap_area_mm2(arena_radius, np.asarray([arena_radius]), arena_radius)
    expected = 2.0 * arena_radius**2 * (math.pi / 3.0) - (math.sqrt(3.0) / 2.0) * arena_radius**2
    assert lens[0] == pytest.approx(expected, rel=1e-9)

    # Chaser on the arena centre with r == R: the ring disc is exactly the arena.
    assert _disc_overlap_area_mm2(arena_radius, np.asarray([0.0]), arena_radius)[0] == pytest.approx(
        math.pi * arena_radius**2
    )


# --------------------------------------------------------------------------------------
# The core property: a uniformly-distributed fish must score 1.0 everywhere
# --------------------------------------------------------------------------------------


def test_uniform_fish_yields_flat_selection_index_static_chaser(tmp_path: Path) -> None:
    """A fish placed uniformly at random has no preference, so every ring must come out at
    chance. The raw observed fraction, by contrast, must rise steeply with radius -- that
    rise is the geometric bias this component exists to remove."""

    rng = np.random.default_rng(20260713)
    n = 30000
    fish = _sample_uniform_in_disc(rng, n, radius_mm=ARENA_RADIUS_MM)
    chaser = np.full((n, 1, 2), ARENA_CENTER_MM + 20.0, dtype=np.float64)
    chaser[:, 0, 1] = ARENA_CENTER_MM  # 20 mm off-centre along +x

    zarr_path = _make_archive(tmp_path, fish_xy_mm=fish, chaser_xy_mm=chaser)
    result = build_chaser_radial_occupancy_result(zarr_path, chaser_distance_run="chaser_distance_1")

    selection = result.radial_selection_index[0, 0, :]
    counts = result.radial_count[0, 0, :]
    well_sampled = counts >= 200
    assert well_sampled.sum() >= 5
    assert np.allclose(selection[well_sampled], 1.0, atol=0.12)

    # The uncorrected histogram is emphatically not flat: outer rings hold far more area.
    observed = result.radial_observed_fraction[0, 0, :]
    assert observed[well_sampled].max() > 4.0 * observed[well_sampled].min()

    # The chaser never moves, so it must not be flagged as pursuing.
    assert not bool(result.chaser_is_moving[0, 0])
    assert not any(w.startswith("closed_loop_null") for w in result.qc_warnings)


def test_uniform_fish_yields_flat_selection_index_moving_chaser(tmp_path: Path) -> None:
    """The same must hold when the chaser roams: the available area of each ring changes
    every frame, and the accumulation has to track it frame by frame."""

    rng = np.random.default_rng(11)
    n = 30000
    fish = _sample_uniform_in_disc(rng, n, radius_mm=ARENA_RADIUS_MM)
    chaser = _sample_uniform_in_disc(rng, n, radius_mm=ARENA_RADIUS_MM * 0.9).reshape(n, 1, 2)

    zarr_path = _make_archive(tmp_path, fish_xy_mm=fish, chaser_xy_mm=chaser)
    result = build_chaser_radial_occupancy_result(zarr_path, chaser_distance_run="chaser_distance_1")

    selection = result.radial_selection_index[0, 0, :]
    counts = result.radial_count[0, 0, :]
    well_sampled = counts >= 200
    assert well_sampled.sum() >= 5
    assert np.allclose(selection[well_sampled], 1.0, atol=0.12)

    assert bool(result.chaser_is_moving[0, 0])
    assert any(w.startswith("closed_loop_null") for w in result.qc_warnings)


def test_near_zone_expected_fraction_matches_closed_form(tmp_path: Path) -> None:
    """With the chaser well inside the arena the near zone is an unclipped disc, so its
    expected fraction is exactly (r_zone / arena_radius)^2."""

    rng = np.random.default_rng(3)
    n = 4000
    fish = _sample_uniform_in_disc(rng, n, radius_mm=ARENA_RADIUS_MM)
    chaser = np.full((n, 1, 2), ARENA_CENTER_MM, dtype=np.float64)

    zarr_path = _make_archive(tmp_path, fish_xy_mm=fish, chaser_xy_mm=chaser)
    result = build_chaser_radial_occupancy_result(
        zarr_path, chaser_distance_run="chaser_distance_1", r_zone_mm=5.0
    )
    expected = (5.0 / ARENA_RADIUS_MM) ** 2
    assert float(result.near_zone_expected_fraction[0, 0]) == pytest.approx(expected, rel=1e-3)


def test_fish_clustered_on_chaser_is_enriched(tmp_path: Path) -> None:
    rng = np.random.default_rng(7)
    n = 2000
    chaser_pos = np.asarray([ARENA_CENTER_MM, ARENA_CENTER_MM])
    fish = chaser_pos + rng.normal(scale=1.0, size=(n, 2))
    chaser = np.tile(chaser_pos, (n, 1)).reshape(n, 1, 2)

    zarr_path = _make_archive(tmp_path, fish_xy_mm=fish, chaser_xy_mm=chaser)
    result = build_chaser_radial_occupancy_result(zarr_path, chaser_distance_run="chaser_distance_1")

    assert float(result.near_zone_observed_fraction[0, 0]) > 0.95
    assert float(result.near_zone_enrichment[0, 0]) > 20.0
    assert float(result.radial_selection_index[0, 0, 0]) > 10.0


def test_fish_avoiding_chaser_is_depleted(tmp_path: Path) -> None:
    """Fish held on the far side of the arena from a static chaser: near rings must fall
    below chance."""

    rng = np.random.default_rng(9)
    n = 2000
    chaser_pos = np.asarray([ARENA_CENTER_MM + 25.0, ARENA_CENTER_MM])
    fish = np.asarray([ARENA_CENTER_MM - 25.0, ARENA_CENTER_MM]) + rng.normal(scale=2.0, size=(n, 2))
    chaser = np.tile(chaser_pos, (n, 1)).reshape(n, 1, 2)

    zarr_path = _make_archive(tmp_path, fish_xy_mm=fish, chaser_xy_mm=chaser)
    result = build_chaser_radial_occupancy_result(zarr_path, chaser_distance_run="chaser_distance_1")

    assert float(result.near_zone_observed_fraction[0, 0]) == pytest.approx(0.0, abs=1e-6)
    assert float(result.near_zone_enrichment[0, 0]) == pytest.approx(0.0, abs=1e-6)


# --------------------------------------------------------------------------------------
# Epochs, controls, wall exclusion
# --------------------------------------------------------------------------------------


def test_epochs_are_scored_independently(tmp_path: Path) -> None:
    rng = np.random.default_rng(5)
    n_half = 1500
    center = np.asarray([ARENA_CENTER_MM, ARENA_CENTER_MM])
    near = center + rng.normal(scale=1.0, size=(n_half, 2))
    far = _sample_uniform_in_disc(rng, n_half, radius_mm=ARENA_RADIUS_MM)
    fish = np.concatenate([near, far], axis=0)
    n = fish.shape[0]
    chaser = np.tile(center, (n, 1)).reshape(n, 1, 2)
    windows = (
        ChaserDistanceWindow(0, "training_event", 0, n_half - 1, 0.0, 1.0, 1.0),
        ChaserDistanceWindow(1, "post_event", n_half, n - 1, 1.0, 2.0, 1.0),
    )

    zarr_path = _make_archive(tmp_path, fish_xy_mm=fish, chaser_xy_mm=chaser, windows=windows)
    result = build_chaser_radial_occupancy_result(zarr_path, chaser_distance_run="chaser_distance_1")

    assert [e.label for e in result.epochs] == ["training_event", "post_event"]
    assert float(result.near_zone_enrichment[0, 0]) > 20.0
    assert float(result.near_zone_enrichment[1, 0]) == pytest.approx(1.0, abs=0.4)


def test_dish_center_control_flags_thigmotaxis(tmp_path: Path) -> None:
    """A wall-hugging fish must show up as an outer-ring excess against the fixed dish-centre
    reference, independent of anything the chaser does."""

    rng = np.random.default_rng(13)
    n = 4000
    radius = ARENA_RADIUS_MM - np.abs(rng.normal(scale=1.0, size=n))
    theta = rng.random(n) * 2.0 * math.pi
    fish = np.stack(
        [ARENA_CENTER_MM + radius * np.cos(theta), ARENA_CENTER_MM + radius * np.sin(theta)], axis=1
    )
    chaser = np.full((n, 1, 2), ARENA_CENTER_MM, dtype=np.float64)

    zarr_path = _make_archive(tmp_path, fish_xy_mm=fish, chaser_xy_mm=chaser)
    result = build_chaser_radial_occupancy_result(zarr_path, chaser_distance_run="chaser_distance_1")

    control = result.control_radial_selection_index[0, :]
    centers = result.radial_bin_centers_mm
    outer = centers >= ARENA_RADIUS_MM - 4.0
    assert np.nanmax(control[outer]) > 3.0

    # The wall-excluded pass must drop those frames entirely.
    assert int(result.wall_excluded_frame_count[0, 0]) == 0
    assert int(result.valid_frame_count[0, 0]) == n


def test_wall_exclusion_keeps_interior_frames(tmp_path: Path) -> None:
    rng = np.random.default_rng(17)
    n = 6000
    fish = _sample_uniform_in_disc(rng, n, radius_mm=ARENA_RADIUS_MM)
    chaser = np.full((n, 1, 2), ARENA_CENTER_MM, dtype=np.float64)

    zarr_path = _make_archive(tmp_path, fish_xy_mm=fish, chaser_xy_mm=chaser)
    result = build_chaser_radial_occupancy_result(
        zarr_path, chaser_distance_run="chaser_distance_1", perimeter_band_mm=5.0
    )

    kept = int(result.wall_excluded_frame_count[0, 0])
    assert 0 < kept < int(result.valid_frame_count[0, 0])
    # Core disc is radius 35 of 40 -> (35/40)^2 ~= 0.766 of the area, hence of a uniform fish.
    assert kept / float(result.valid_frame_count[0, 0]) == pytest.approx((35.0 / 40.0) ** 2, rel=0.05)

    # A uniform fish is still at chance once the wall band is removed from both sides.
    selection = result.radial_selection_index_wall_excluded[0, 0, :]
    counts = result.radial_count_wall_excluded[0, 0, :]
    well_sampled = counts >= 150
    assert np.allclose(selection[well_sampled], 1.0, atol=0.15)


def test_invalid_fish_frames_are_excluded(tmp_path: Path) -> None:
    rng = np.random.default_rng(23)
    n = 1000
    center = np.asarray([ARENA_CENTER_MM, ARENA_CENTER_MM])
    fish = np.tile(center + np.asarray([30.0, 0.0]), (n, 1))
    fish[:500] = center  # would be a near-zone hit, but marked invalid below
    chaser = np.tile(center, (n, 1)).reshape(n, 1, 2)
    valid = np.ones(n, dtype=bool)
    valid[:500] = False

    zarr_path = _make_archive(tmp_path, fish_xy_mm=fish, chaser_xy_mm=chaser, fish_valid=valid)
    result = build_chaser_radial_occupancy_result(zarr_path, chaser_distance_run="chaser_distance_1")

    assert int(result.valid_frame_count[0, 0]) == 500
    assert float(result.near_zone_observed_fraction[0, 0]) == pytest.approx(0.0, abs=1e-6)


def test_cdf_enrichment_matches_ring_sums(tmp_path: Path) -> None:
    rng = np.random.default_rng(29)
    n = 5000
    fish = _sample_uniform_in_disc(rng, n, radius_mm=ARENA_RADIUS_MM)
    chaser = np.full((n, 1, 2), ARENA_CENTER_MM, dtype=np.float64)

    zarr_path = _make_archive(tmp_path, fish_xy_mm=fish, chaser_xy_mm=chaser)
    result = build_chaser_radial_occupancy_result(
        zarr_path, chaser_distance_run="chaser_distance_1", cdf_thresholds_mm=(10.0, 20.0)
    )

    thresholds = np.asarray(result.cdf_thresholds_mm, dtype=np.float64)
    for t_idx, threshold in enumerate(thresholds):
        # Uniform fish -> observed CDF ~= (threshold / arena_radius)^2 and enrichment ~= 1.
        assert float(result.cdf_observed_fraction[0, 0, t_idx]) == pytest.approx(
            (threshold / ARENA_RADIUS_MM) ** 2, rel=0.1
        )
        assert float(result.cdf_enrichment[0, 0, t_idx]) == pytest.approx(1.0, abs=0.12)


def test_low_expected_count_rings_are_suppressed(tmp_path: Path) -> None:
    """Rings near the maximum attainable distance have almost no available area, so their
    observed/expected ratio is unstable. They must come back NaN rather than as a spike."""

    rng = np.random.default_rng(47)
    n = 300
    # Chaser pinned near the wall, so the far rings (approaching the arena diameter) are
    # slivers of area diametrically opposite it.
    chaser_pos = np.asarray([ARENA_CENTER_MM + ARENA_RADIUS_MM - 2.0, ARENA_CENTER_MM])
    fish = _sample_uniform_in_disc(rng, n, radius_mm=ARENA_RADIUS_MM)
    chaser = np.tile(chaser_pos, (n, 1)).reshape(n, 1, 2)

    zarr_path = _make_archive(tmp_path, fish_xy_mm=fish, chaser_xy_mm=chaser)
    result = build_chaser_radial_occupancy_result(
        zarr_path, chaser_distance_run="chaser_distance_1", min_expected_count=5.0
    )

    expected = result.radial_expected_fraction[0, 0, :]
    selection = result.radial_selection_index[0, 0, :]
    n_valid = int(result.valid_frame_count[0, 0])
    expected_count = expected * n_valid
    # Rings the null can still reach, but too thinly to support a stable ratio.
    underpowered = np.isfinite(expected) & (expected_count > 0) & (expected_count < 5.0)
    assert underpowered.any(), "fixture should produce at least one underpowered outer ring"
    assert np.all(np.isnan(selection[underpowered]))

    # Well-powered rings survive, and the raw counts/areas are still persisted for them all.
    powered = np.isfinite(expected) & (expected_count >= 5.0)
    assert np.isfinite(selection[powered]).any()
    assert np.isfinite(result.radial_expected_area_mm2[0, 0, :]).all()
    assert any(w.startswith("low_expected_count_rings") for w in result.qc_warnings)

    # Turning the guard off restores the raw (unstable) ratio.
    unguarded = build_chaser_radial_occupancy_result(
        zarr_path, chaser_distance_run="chaser_distance_1", min_expected_count=0.0
    )
    raw = unguarded.radial_selection_index[0, 0, :]
    assert np.isfinite(raw[underpowered]).any()


# --------------------------------------------------------------------------------------
# Settle trim: objects repositioning at a phase boundary
# --------------------------------------------------------------------------------------


def _make_settle_archive(tmp_path: Path, *, fps: float = 10.0, name: str = "settle.zarr") -> Path:
    """Two epochs at 10 fps. The object repositions over the first 2 s of `post_event`
    (a scripted transition) and is static afterwards. `training_event` has the object
    moving throughout -- that is the stimulus, not a settle."""

    rng = np.random.default_rng(101)
    center = np.asarray([ARENA_CENTER_MM, ARENA_CENTER_MM])
    n_train, n_post = 100, 100
    settle = int(2.0 * fps)  # 20 frames

    # training: chaser sweeps a wide arc for the whole epoch.
    t = np.linspace(0.0, 2.0 * math.pi, n_train)
    train_chaser = np.stack([center[0] + 20.0 * np.cos(t), center[1] + 20.0 * np.sin(t)], axis=1)

    # post: chaser travels from (-20, 0) to (+20, 0) over the settle window, then parks.
    post_chaser = np.tile(center + np.asarray([20.0, 0.0]), (n_post, 1))
    ramp = np.linspace(-20.0, 20.0, settle)
    post_chaser[:settle, 0] = center[0] + ramp
    post_chaser[:settle, 1] = center[1]

    chaser = np.concatenate([train_chaser, post_chaser], axis=0).reshape(-1, 1, 2)
    n = chaser.shape[0]
    fish = _sample_uniform_in_disc(rng, n, radius_mm=ARENA_RADIUS_MM)
    windows = (
        ChaserDistanceWindow(0, "training_event", 0, n_train - 1, 0.0, 10.0, 10.0),
        ChaserDistanceWindow(1, "post_event", n_train, n - 1, 10.0, 20.0, 10.0),
    )
    return _make_archive(
        tmp_path, fish_xy_mm=fish, chaser_xy_mm=chaser, windows=windows, fps=fps, name=name
    )


def test_settle_trim_excludes_repositioning_frames(tmp_path: Path) -> None:
    zarr_path = _make_settle_archive(tmp_path)
    result = build_chaser_radial_occupancy_result(
        zarr_path, chaser_distance_run="chaser_distance_1", settle_trim_s=2.0
    )
    by_label = {e.label: e for e in result.epochs}

    post = by_label["post_event"]
    assert post.static_configuration is True
    assert post.settle_excluded_frame_count == 20
    assert post.start_frame == post.source_start_frame + 20

    # The object is stationary across the whole effective window, so it must not be
    # flagged as pursuing -- that flag was previously manufactured by the transit frames.
    post_idx = result.epochs.index(post)
    assert not bool(result.chaser_is_moving[post_idx, 0])
    assert float(result.chaser_position_spread_mm[post_idx, 0]) == pytest.approx(0.0, abs=1e-3)
    assert not any(w.startswith("closed_loop_null") and "post_event" in w for w in result.qc_warnings)


def test_settle_trim_leaves_dynamic_epochs_alone(tmp_path: Path) -> None:
    """The chase epoch has no settle -- trimming it would discard real stimulus frames."""

    zarr_path = _make_settle_archive(tmp_path)
    result = build_chaser_radial_occupancy_result(
        zarr_path, chaser_distance_run="chaser_distance_1", settle_trim_s=2.0
    )
    training = {e.label: e for e in result.epochs}["training_event"]

    assert training.static_configuration is False
    assert training.settle_excluded_frame_count == 0
    assert training.start_frame == training.source_start_frame

    train_idx = result.epochs.index(training)
    assert bool(result.chaser_is_moving[train_idx, 0])


def test_settle_trim_defaults_to_protocol_transition_duration(tmp_path: Path) -> None:
    zarr_path = _make_settle_archive(tmp_path, name="settle_protocol.zarr")
    root = zarr.open_group(str(zarr_path), mode="a", use_consolidated=False)
    root["analysis/stimulus_runs/stimulus_1"].attrs["protocol_json"] = json.dumps(
        {"steps": [{"parameters": {"position_transition_duration_s": 2.0}}]}
    )

    result = build_chaser_radial_occupancy_result(zarr_path, chaser_distance_run="chaser_distance_1")
    assert result.settle_trim_s == pytest.approx(2.0)
    assert {e.label: e for e in result.epochs}["post_event"].settle_excluded_frame_count == 20

    # No protocol -> no trim, and the transit frames leak back in (the original defect).
    del root["analysis/stimulus_runs/stimulus_1"].attrs["protocol_json"]
    untrimmed = build_chaser_radial_occupancy_result(zarr_path, chaser_distance_run="chaser_distance_1")
    assert untrimmed.settle_trim_s == pytest.approx(0.0)
    post = {e.label: e for e in untrimmed.epochs}["post_event"]
    assert post.settle_excluded_frame_count == 0
    post_idx = untrimmed.epochs.index(post)
    assert bool(untrimmed.chaser_is_moving[post_idx, 0])


# --------------------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------------------


def test_write_component_round_trips(tmp_path: Path) -> None:
    rng = np.random.default_rng(31)
    n = 2000
    fish = _sample_uniform_in_disc(rng, n, radius_mm=ARENA_RADIUS_MM)
    chaser = _sample_uniform_in_disc(rng, n, radius_mm=ARENA_RADIUS_MM * 0.8).reshape(n, 1, 2)

    zarr_path = _make_archive(tmp_path, fish_xy_mm=fish, chaser_xy_mm=chaser)
    result = build_chaser_radial_occupancy_result(zarr_path, chaser_distance_run="chaser_distance_1")
    component_path = write_chaser_radial_occupancy_component(zarr_path, result, overwrite=True)

    assert component_path == (
        f"analysis/chaser_distance_runs/chaser_distance_1/{COMPONENT_PARENT_NAME}/{DEFAULT_COMPONENT_NAME}"
    )
    root = zarr.open_group(str(zarr_path), mode="r", use_consolidated=False)
    component = root[component_path]
    assert component.attrs["schema_id"] == SCHEMA_ID
    assert component.attrs["status"] == "computed"
    assert component.attrs["geometry_status"] == "circle"

    parent = root[f"analysis/chaser_distance_runs/chaser_distance_1/{COMPONENT_PARENT_NAME}"]
    assert parent.attrs["latest"] == DEFAULT_COMPONENT_NAME
    assert parent.attrs["latest_complete"] == DEFAULT_COMPONENT_NAME

    radial = component["radial_occupancy"]
    n_bins = int(result.radial_bin_centers_mm.shape[0])
    for name in (
        "radial_count",
        "radial_observed_fraction",
        "radial_expected_area_mm2",
        "radial_expected_fraction",
        "radial_occupancy_density_per_mm2",
        "radial_selection_index",
        "radial_selection_index_wall_excluded",
    ):
        assert radial[name].shape == (1, 1, n_bins), name
    assert radial.attrs["axis_order"] == ["epoch", "chaser", "radial_bin"]

    per_epoch = component["per_epoch_chaser"]
    assert per_epoch["near_zone_enrichment"].shape == (1, 1)
    assert bool(np.asarray(per_epoch["chaser_is_moving"][:])[0, 0])

    assert _decode_first(component["epochs"]["label_bytes"]) == "all_frames"
    assert _decode_first(component["control_reference"]["reference_label_bytes"]) == "dish_center"
    assert _decode_first(component["summary"]["status_bytes"]) == "computed"

    visualizations = component["visualizations"]
    for artifact in (RADIAL_DENSITY_PNG_ARTIFACT_NAME, SELECTION_INDEX_PNG_ARTIFACT_NAME):
        png = visualizations[artifact]
        assert png.attrs["artifact_schema_id"] == PNG_ARTIFACT_SCHEMA_ID
        assert np.asarray(png[:], dtype=np.uint8).tobytes().startswith(b"\x89PNG")

    spec = visualizations[INTERACTIVE_ARTIFACT_NAME]
    assert spec.attrs["artifact_schema_id"] == INTERACTIVE_SPEC_SCHEMA_ID
    assert spec.attrs["renderer"] == "palette-chaser-radial-occupancy-v1"


def test_write_component_requires_overwrite(tmp_path: Path) -> None:
    rng = np.random.default_rng(37)
    n = 500
    fish = _sample_uniform_in_disc(rng, n, radius_mm=ARENA_RADIUS_MM)
    chaser = np.full((n, 1, 2), ARENA_CENTER_MM, dtype=np.float64)

    zarr_path = _make_archive(tmp_path, fish_xy_mm=fish, chaser_xy_mm=chaser)
    result = build_chaser_radial_occupancy_result(zarr_path, chaser_distance_run="chaser_distance_1")
    write_chaser_radial_occupancy_component(zarr_path, result, overwrite=True, write_png=False)

    with pytest.raises(ValueError, match="already exists"):
        write_chaser_radial_occupancy_component(zarr_path, result, overwrite=False, write_png=False)


def test_ignores_stale_legacy_coordinate_frame_attr(tmp_path: Path) -> None:
    rng = np.random.default_rng(41)
    n = 200
    fish = _sample_uniform_in_disc(rng, n, radius_mm=ARENA_RADIUS_MM)
    chaser = np.full((n, 1, 2), ARENA_CENTER_MM, dtype=np.float64)
    zarr_path = _make_archive(tmp_path, fish_xy_mm=fish, chaser_xy_mm=chaser)

    root = zarr.open_group(str(zarr_path), mode="a", use_consolidated=False)
    root["analysis/chaser_distance_runs/chaser_distance_1"].attrs["coordinate_frame"] = "camera_px"

    result = build_chaser_radial_occupancy_result(
        zarr_path,
        chaser_distance_run="chaser_distance_1",
    )
    assert result.coordinate_frame == "arena_relative_canvas_px"


def test_missing_arena_geometry_is_fatal(tmp_path: Path) -> None:
    rng = np.random.default_rng(43)
    n = 200
    fish = _sample_uniform_in_disc(rng, n, radius_mm=ARENA_RADIUS_MM)
    chaser = np.full((n, 1, 2), ARENA_CENTER_MM, dtype=np.float64)
    zarr_path = _make_archive(tmp_path, fish_xy_mm=fish, chaser_xy_mm=chaser)

    root = zarr.open_group(str(zarr_path), mode="a", use_consolidated=False)
    del root["analysis/stimulus_runs/stimulus_1/calibration"]["arena_geometry"]

    with pytest.raises(ValueError, match="arena geometry"):
        build_chaser_radial_occupancy_result(zarr_path, chaser_distance_run="chaser_distance_1")
