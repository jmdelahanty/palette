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
