from __future__ import annotations

import numpy as np
import pytest

from fisheye.analysis.eye_angle_analysis import (
    MAJOR_AXIS_MARGINAL_DOT_THRESHOLD,
    _process_chunk,
    _signed_angle_from_body_axes,
)


def _alpha_from_body_major_deg(major_deg: float) -> float:
    """Convert a body-frame major-axis angle to OpenCV ellipse angle for the test body frame."""
    radians = np.deg2rad(float(major_deg))
    vector_xy = np.asarray([np.cos(radians), -np.sin(radians)], dtype=np.float64)
    return float(np.rad2deg(np.arctan2(vector_xy[1], vector_xy[0])) % 180.0)


def _run_eye_chunk(*, left_major_deg: float, right_major_deg: float):
    ellipse_params = np.asarray(
        [
            [
                [2.0, -1.0, 4.0, 1.5, _alpha_from_body_major_deg(left_major_deg)],
                [2.0, 1.0, 4.0, 1.5, _alpha_from_body_major_deg(right_major_deg)],
            ]
        ],
        dtype=np.float32,
    )
    keypoints_roi = np.asarray(
        [
            [
                [0.0, 0.0],
                [2.0, -1.0],
                [2.0, 1.0],
            ]
        ],
        dtype=np.float32,
    )
    return _process_chunk(
        ellipse_params=ellipse_params,
        ellipse_success=np.asarray([[True, True]], dtype=bool),
        keypoints_roi=keypoints_roi,
        heading_deg=np.asarray([0.0], dtype=np.float32),
        detection_success=np.asarray([True], dtype=bool),
        keypoint_indices={"swim_bladder": 0, "eye_left": 1, "eye_right": 2},
    )


def _old_minor_half_plane_signed_deg(major_deg: float) -> float:
    """Reproduce the previous independent minor-axis half-plane resolution."""
    alpha = np.deg2rad(_alpha_from_body_major_deg(major_deg))
    forward_axis = np.asarray([[1.0, 0.0]], dtype=np.float64)
    left_axis = np.asarray([[0.0, -1.0]], dtype=np.float64)
    axis_minor = np.asarray([[-np.sin(alpha), np.cos(alpha)]], dtype=np.float64)
    sign_minor = 1.0 if float(axis_minor[0] @ forward_axis[0]) >= 0.0 else -1.0
    signed = _signed_angle_from_body_axes(axis_minor * sign_minor, forward_axis, left_axis)
    return float(np.clip(signed[0], -90.0, 90.0))


def test_gaze_is_derived_from_resolved_major_axis_without_independent_flip() -> None:
    result = _run_eye_chunk(left_major_deg=-20.0, right_major_deg=20.0)

    assert result.left_major_signed_deg[0] == pytest.approx(-20.0, abs=1e-4)
    assert result.right_major_signed_deg[0] == pytest.approx(20.0, abs=1e-4)
    assert result.left_gaze_signed_deg[0] == pytest.approx(70.0, abs=1e-4)
    assert result.right_gaze_signed_deg[0] == pytest.approx(-70.0, abs=1e-4)
    assert result.left_minor_signed_deg[0] == pytest.approx(result.left_gaze_signed_deg[0], abs=1e-4)
    assert result.right_minor_signed_deg[0] == pytest.approx(result.right_gaze_signed_deg[0], abs=1e-4)


def test_gaze_vector_and_signed_angle_are_consistent() -> None:
    result = _run_eye_chunk(left_major_deg=-35.0, right_major_deg=25.0)
    forward = result.body_frame_forward_axis_xy
    left = result.body_frame_left_axis_xy

    left_from_vector = _signed_angle_from_body_axes(result.left_gaze_xy, forward, left)
    right_from_vector = _signed_angle_from_body_axes(result.right_gaze_xy, forward, left)
    np.testing.assert_allclose(left_from_vector, result.left_gaze_signed_deg, atol=1e-5)
    np.testing.assert_allclose(right_from_vector, result.right_gaze_signed_deg, atol=1e-5)
    np.testing.assert_allclose(np.linalg.norm(result.left_gaze_xy, axis=1), [1.0], atol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(result.right_gaze_xy, axis=1), [1.0], atol=1e-6)


def test_axis_resolution_is_smooth_at_lateral_gaze_boundary() -> None:
    left_gaze_values = []
    for major_deg in np.linspace(-2.0, 2.0, 9):
        result = _run_eye_chunk(left_major_deg=float(major_deg), right_major_deg=20.0)
        left_gaze_values.append(float(result.left_gaze_signed_deg[0]))

    assert np.nanmax(np.abs(np.diff(left_gaze_values))) < 1.0
    np.testing.assert_allclose(left_gaze_values, np.linspace(88.0, 92.0, 9), atol=1e-4)


def test_typical_larval_gaze_matches_previous_minor_resolution() -> None:
    for left_major_deg in np.linspace(-60.0, -10.0, 6):
        result = _run_eye_chunk(left_major_deg=float(left_major_deg), right_major_deg=20.0)
        assert result.left_gaze_signed_deg[0] == pytest.approx(
            _old_minor_half_plane_signed_deg(float(left_major_deg)),
            abs=1e-4,
        )


def test_major_axis_marginal_is_nonfatal_qa_flag() -> None:
    result = _run_eye_chunk(left_major_deg=88.0, right_major_deg=20.0)

    assert bool(result.valid_left[0])
    assert bool(result.valid_frame[0])
    assert bool(result.left_major_axis_marginal[0])
    assert not bool(result.right_major_axis_marginal[0])
    assert bool(result.major_axis_marginal[0])
    assert np.cos(np.deg2rad(88.0)) < MAJOR_AXIS_MARGINAL_DOT_THRESHOLD
