from __future__ import annotations

import numpy as np
import pytest

from fisheye.analysis.subject_shape_spline import (
    DEFAULT_BSPLINE_DEGREE,
    fit_subject_body_spline_batch,
)


pytest.importorskip("scipy")


def test_fit_subject_body_spline_batch_interpolates_straight_tail() -> None:
    sample_count = 16
    tail_count = 5
    x = np.linspace(0.0, 10.0, sample_count, dtype=np.float32)
    y = np.full_like(x, 5.0)
    centerline = np.stack([x, y], axis=1)[None, :, :]

    batch = fit_subject_body_spline_batch(
        centerline,
        np.asarray([True], dtype=bool),
        np.asarray([True], dtype=bool),
        np.asarray([4.0], dtype=np.float32),
        centerline_failure_reasons=["ok"],
        tail_base_failure_reasons=["ok"],
        centerline_sample_count=sample_count,
        tail_sample_count=tail_count,
        degree=DEFAULT_BSPLINE_DEGREE,
        smoothing=0.0,
    )

    assert batch.bspline_valid.tolist() == [True]
    assert batch.tail_sample_valid.tolist() == [True]
    assert batch.bspline_failure_reasons.tolist() == ["ok"]
    assert batch.tail_sample_failure_reasons.tolist() == ["ok"]
    assert batch.bspline_control_points_xy.shape == (1, sample_count, 2)
    assert batch.bspline_knots.shape == (1, sample_count + DEFAULT_BSPLINE_DEGREE + 1)
    assert batch.bspline_degree_used.tolist() == [DEFAULT_BSPLINE_DEGREE]
    np.testing.assert_allclose(batch.bspline_arc_length_px, [10.0], atol=0.1)
    np.testing.assert_allclose(batch.tail_sample_s, np.linspace(0.0, 1.0, tail_count), atol=1e-6)
    np.testing.assert_allclose(batch.tail_sample_xy[0, 0], [4.0, 5.0], atol=0.15)
    np.testing.assert_allclose(batch.tail_sample_xy[0, -1], [10.0, 5.0], atol=0.15)
    np.testing.assert_allclose(batch.tail_tangent_xy[0], np.tile([[1.0, 0.0]], (tail_count, 1)), atol=1e-5)
    np.testing.assert_allclose(batch.tail_normal_xy[0], np.tile([[0.0, 1.0]], (tail_count, 1)), atol=1e-5)
    np.testing.assert_allclose(batch.tail_curvature_px_inv[0], np.zeros((tail_count,)), atol=1e-5)


def test_fit_subject_body_spline_batch_keeps_body_spline_when_tail_base_missing() -> None:
    sample_count = 16
    centerline = np.stack(
        [
            np.linspace(0.0, 10.0, sample_count, dtype=np.float32),
            np.zeros((sample_count,), dtype=np.float32),
        ],
        axis=1,
    )[None, :, :]

    batch = fit_subject_body_spline_batch(
        centerline,
        np.asarray([True], dtype=bool),
        np.asarray([False], dtype=bool),
        np.asarray([np.nan], dtype=np.float32),
        centerline_failure_reasons=["ok"],
        tail_base_failure_reasons=["missing_tail_anchor"],
        centerline_sample_count=sample_count,
        tail_sample_count=4,
    )

    assert batch.bspline_valid.tolist() == [True]
    assert batch.tail_sample_valid.tolist() == [False]
    assert batch.bspline_failure_reasons.tolist() == ["ok"]
    assert batch.tail_sample_failure_reasons.tolist() == ["missing_tail_anchor"]
    assert np.all(np.isfinite(batch.bspline_sample_xy[0]))
    assert np.all(np.isnan(batch.tail_sample_xy[0]))


def test_fit_subject_body_spline_batch_propagates_invalid_centerline_reason() -> None:
    centerline = np.full((1, 8, 2), np.nan, dtype=np.float32)

    batch = fit_subject_body_spline_batch(
        centerline,
        np.asarray([False], dtype=bool),
        np.asarray([False], dtype=bool),
        np.asarray([np.nan], dtype=np.float32),
        centerline_failure_reasons=["source_body_mask_qc_failed"],
        tail_base_failure_reasons=["source_body_mask_qc_failed"],
        centerline_sample_count=8,
        tail_sample_count=4,
    )

    assert batch.bspline_valid.tolist() == [False]
    assert batch.tail_sample_valid.tolist() == [False]
    assert batch.bspline_failure_reasons.tolist() == ["source_body_mask_qc_failed"]
    assert batch.tail_sample_failure_reasons.tolist() == ["source_body_mask_qc_failed"]
    assert np.all(np.isnan(batch.bspline_sample_xy[0]))


def _batch(centerline, *, curvature_smoothing_px, sample_count=64, tail_count=32):
    n = 1
    return fit_subject_body_spline_batch(
        centerline[None, :, :].astype(np.float32),
        np.asarray([True] * n, dtype=bool),
        np.asarray([True] * n, dtype=bool),
        np.asarray([float(sample_count) * 0.5], dtype=np.float32),  # tail base at mid-body
        centerline_failure_reasons=["ok"],
        tail_base_failure_reasons=["ok"],
        centerline_sample_count=sample_count,
        tail_sample_count=tail_count,
        degree=DEFAULT_BSPLINE_DEGREE,
        smoothing=0.0,
        curvature_smoothing_px=curvature_smoothing_px,
    )


def test_tail_curvature_uses_a_smoothing_spline_not_the_interpolating_one() -> None:
    """The reported bug: an interpolating spline (s=0) turns pixel-quantization jitter on the
    skeleton into meaningless sub-pixel curvature radii. Curvature must come from a separate
    SMOOTHING spline, so a near-straight but jittery centerline reads as nearly straight."""

    rng = np.random.default_rng(0)
    n = 64
    x = np.linspace(0.0, 75.0, n)                 # ~75 px straight body
    y = np.zeros(n) + rng.normal(0.0, 0.6, n)     # sub-pixel skeleton jitter
    line = np.stack([x, y], axis=1)

    interp = _batch(line, curvature_smoothing_px=0.0)      # old behaviour (differentiate s=0)
    smooth = _batch(line, curvature_smoothing_px=0.75)     # the fix

    max_interp = np.nanmax(np.abs(interp.tail_curvature_px_inv[0]))
    max_smooth = np.nanmax(np.abs(smooth.tail_curvature_px_inv[0]))
    # the interpolating spline invents high curvature from the jitter (small radius)...
    assert max_interp > 0.1                        # radius < 10 px, pure noise on a straight line
    # ...the smoothing spline reports a nearly straight tail (large radius)
    assert max_smooth < 0.02                       # radius > 50 px
    assert max_interp > 8 * max_smooth
    # positions and arc length are unchanged (still the interpolating spline)
    np.testing.assert_allclose(smooth.tail_sample_xy[0], interp.tail_sample_xy[0], atol=1e-4)
    np.testing.assert_allclose(smooth.bspline_arc_length_px, interp.bspline_arc_length_px, atol=1e-4)


def test_smoothing_preserves_a_real_body_bend() -> None:
    """Smoothing must remove noise, not signal: a genuine coherent arc of known radius must
    still read back at ~1/radius, so a real C-bend would not be flattened away."""

    n = 64
    radius = 20.0                                   # a real, tight bend (20 px radius)
    theta = np.linspace(0.0, np.pi / 2.0, n)        # quarter circle
    arc = np.stack([radius * np.cos(theta), radius * np.sin(theta)], axis=1)
    rng = np.random.default_rng(1)
    arc = arc + rng.normal(0.0, 0.6, arc.shape)     # same jitter on top of the real bend

    smooth = _batch(arc, curvature_smoothing_px=0.75)
    k = np.abs(smooth.tail_curvature_px_inv[0])
    k = k[np.isfinite(k)]
    assert k.size
    # the recovered curvature should sit near 1/radius, not be smoothed to zero
    assert 0.6 / radius < np.median(k) < 1.6 / radius
