"""Tests for analysis_stage_arrays — input read contracts and output specs."""

from __future__ import annotations

import numpy as np
import zarr

from fisheye.shared.zarr.analysis_stage_arrays import (
    STIMULUS_RESPONSE_GLOBAL_ARRAYS,
    STIMULUS_RESPONSE_PER_FISH_ARRAYS,
    STIMULUS_RESPONSE_TRACK_INPUTS,
    validate_global_output,
    validate_per_fish_output,
    validate_track_inputs,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_track_group(n_samples: int = 10) -> zarr.Group:
    """Create a synthetic track subgroup matching the input contract."""
    g = zarr.group()
    g.create_array("frame_indices", data=np.arange(n_samples, dtype=np.int64))
    g.create_array("time_seconds", data=np.linspace(0, 1, n_samples, dtype=np.float32))
    g.create_array("positions_mm", data=np.zeros((n_samples, 2), dtype=np.float32))
    g.create_array("heading_degrees", data=np.zeros(n_samples, dtype=np.float32))
    g.create_array("speed_smoothed_mm", data=np.ones(n_samples, dtype=np.float32))
    g.create_array("angular_velocity_deg_s", data=np.zeros(n_samples, dtype=np.float32))
    g.create_array("detection_source", data=np.zeros(n_samples, dtype=np.int8))
    return g


def _make_global_group(n_fish: int = 2) -> zarr.Group:
    g = zarr.group()
    g.create_array("fish_id", data=np.arange(n_fish, dtype=np.int32))
    g.create_array("total_distance_mm", data=np.zeros(n_fish, dtype=np.float32))
    g.create_array("mean_speed_mm_s", data=np.zeros(n_fish, dtype=np.float32))
    g.create_array("total_active_s", data=np.zeros(n_fish, dtype=np.float32))
    g.create_array("fraction_moving", data=np.zeros(n_fish, dtype=np.float32))
    return g


def _make_per_fish_group(n_fish: int = 2) -> zarr.Group:
    g = zarr.group()
    g.create_array("fish_id", data=np.arange(n_fish, dtype=np.int32))
    g.create_array("total_distance_mm", data=np.zeros(n_fish, dtype=np.float32))
    g.create_array("mean_speed_mm_s", data=np.zeros(n_fish, dtype=np.float32))
    g.create_array("median_speed_mm_s", data=np.zeros(n_fish, dtype=np.float32))
    g.create_array("max_speed_mm_s", data=np.zeros(n_fish, dtype=np.float32))
    g.create_array("fraction_moving", data=np.zeros(n_fish, dtype=np.float32))
    g.create_array("coverage", data=np.ones(n_fish, dtype=np.float32))
    return g


# ---------------------------------------------------------------------------
# Input contract tests
# ---------------------------------------------------------------------------


def test_validate_track_inputs_happy_path() -> None:
    g = _make_track_group()
    result = validate_track_inputs(g)
    assert result.valid, f"Unexpected errors: {result.errors}"


def test_validate_track_inputs_missing_array() -> None:
    g = _make_track_group()
    del g["speed_smoothed_mm"]
    result = validate_track_inputs(g)
    assert not result.valid
    assert any("speed_smoothed_mm" in e for e in result.errors)


def test_validate_track_inputs_wrong_dtype() -> None:
    g = _make_track_group()
    # Replace float32 with int32.
    g.create_array("speed_smoothed_mm", data=np.zeros(10, dtype=np.int32), overwrite=True)
    result = validate_track_inputs(g)
    assert not result.valid
    assert any("dtype kind mismatch" in e for e in result.errors)


def test_validate_track_inputs_wrong_rank() -> None:
    g = _make_track_group()
    # positions_mm should be (n, 2) not (n,).
    g.create_array("positions_mm", data=np.zeros(10, dtype=np.float32), overwrite=True)
    result = validate_track_inputs(g)
    assert not result.valid
    assert any("rank mismatch" in e for e in result.errors)


# ---------------------------------------------------------------------------
# Output spec tests
# ---------------------------------------------------------------------------


def test_validate_global_output_happy_path() -> None:
    g = _make_global_group()
    result = validate_global_output(g)
    assert result.valid, f"Unexpected errors: {result.errors}"


def test_validate_global_output_missing_array() -> None:
    g = _make_global_group()
    del g["fraction_moving"]
    result = validate_global_output(g)
    assert not result.valid
    assert any("fraction_moving" in e for e in result.errors)


def test_validate_per_fish_output_happy_path() -> None:
    g = _make_per_fish_group()
    result = validate_per_fish_output(g)
    assert result.valid, f"Unexpected errors: {result.errors}"


def test_validate_per_fish_output_missing_coverage() -> None:
    g = _make_per_fish_group()
    del g["coverage"]
    result = validate_per_fish_output(g)
    assert not result.valid
    assert any("coverage" in e for e in result.errors)


def test_leading_dimension_consistency() -> None:
    """All arrays in per_fish should share the n_fish leading dim."""
    g = zarr.group()
    g.create_array("fish_id", data=np.arange(3, dtype=np.int32))
    g.create_array("total_distance_mm", data=np.zeros(3, dtype=np.float32))
    g.create_array("mean_speed_mm_s", data=np.zeros(3, dtype=np.float32))
    g.create_array("median_speed_mm_s", data=np.zeros(3, dtype=np.float32))
    g.create_array("max_speed_mm_s", data=np.zeros(3, dtype=np.float32))
    g.create_array("fraction_moving", data=np.zeros(3, dtype=np.float32))
    # Mismatched leading dim.
    g.create_array("coverage", data=np.zeros(5, dtype=np.float32))
    result = validate_per_fish_output(g)
    assert not result.valid
    assert any("leading dimension mismatch" in e for e in result.errors)


def test_spec_tuples_are_nonempty() -> None:
    assert len(STIMULUS_RESPONSE_TRACK_INPUTS) > 0
    assert len(STIMULUS_RESPONSE_GLOBAL_ARRAYS) > 0
    assert len(STIMULUS_RESPONSE_PER_FISH_ARRAYS) > 0
