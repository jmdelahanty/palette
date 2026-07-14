from __future__ import annotations

import numpy as np
import pytest

from fisheye.analysis.chaser_gaze_tracking import (
    _dense_frame_row_lookup,
    _virtual_positions,
    fit_dynamic_tracking_gain,
    fit_linear_tracking_gain,
    sustained_true_runs,
)


def test_dense_frame_row_lookup_marks_trailing_frames_unavailable() -> None:
    row_index, present = _dense_frame_row_lookup(
        5,
        np.asarray([0, 2, 4, 7, 8, 9], dtype=np.int64),
    )

    np.testing.assert_array_equal(present, [True, True, True, False, False, False])
    np.testing.assert_array_equal(row_index, [0, 2, 4, -1, -1, -1])


def test_dense_frame_row_lookup_refuses_negative_row_count() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        _dense_frame_row_lookup(-1, np.asarray([0, 1, 2], dtype=np.int64))


def test_static_tracking_gain_distinguishes_eye_tracking_from_head_fixed_eye() -> None:
    bearing = np.linspace(50.0, 100.0, 500)
    valid = np.ones(bearing.shape, dtype=bool)
    tracked = 0.8 * bearing + 5.0
    fixed = np.full(bearing.shape, 72.0)
    tracked_fit = fit_linear_tracking_gain(bearing, tracked, valid)
    fixed_fit = fit_linear_tracking_gain(bearing, fixed, valid)
    assert np.isclose(tracked_fit.gain, 0.8)
    assert np.isclose(tracked_fit.intercept_deg, 5.0)
    assert np.isclose(fixed_fit.gain, 0.0)


def test_dynamic_tracking_gain_recovers_positive_eye_lag() -> None:
    rng = np.random.default_rng(3)
    increments = rng.normal(0.0, 1.0, 600)
    bearing = np.cumsum(increments)
    gaze = np.zeros_like(bearing)
    gaze[3:] = bearing[:-3]
    valid = np.ones(bearing.shape, dtype=bool)
    fit = fit_dynamic_tracking_gain(bearing, gaze, valid, fps=100.0, max_lag_s=0.1)
    assert fit.lag_frames == 3
    assert fit.lag_seconds == 0.03
    assert fit.correlation > 0.99
    assert np.isclose(fit.gain, 1.0, atol=0.02)


def test_sustained_true_runs_uses_inclusive_intervals_and_minimum_length() -> None:
    mask = np.asarray([False, True, True, False, True, True, True, False, True])
    assert sustained_true_runs(mask, min_frames=3) == ((4, 6),)


def test_virtual_reference_is_dropped_when_it_overlaps_a_real_object() -> None:
    chaser_xy = np.asarray(
        [
            [[1.0, 1.0], [9.0, 9.0]],
            [[1.0, 1.0], [9.0, 9.0]],
            [[1.0, 1.0], [9.0, 9.0]],
        ]
    )
    refs, _positions = _virtual_positions(
        chaser_xy=chaser_xy,
        chaser_indices=np.asarray([0, 1]),
        center_xy=(5.0, 5.0),
        rotations_deg=(60.0, 180.0),
        min_separation_mm=1.0,
        pixels_per_mm=1.0,
    )
    assert {ref.rotation_deg for ref in refs} == {60.0}
    assert len(refs) == 2
