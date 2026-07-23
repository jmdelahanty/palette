from __future__ import annotations

import numpy as np
import pytest

from fisheye.analysis.eye_angle_analysis import _process_chunk


def _alpha_from_body_major_deg(major_deg: float) -> float:
    """Convert a body-frame major-axis angle to OpenCV ellipse angle for this test frame."""
    radians = np.deg2rad(float(major_deg))
    vector_xy = np.asarray([np.cos(radians), -np.sin(radians)], dtype=np.float64)
    return float(np.rad2deg(np.arctan2(vector_xy[1], vector_xy[0])) % 180.0)


def _run_eye_chunk(
    *,
    left_major_deg: float,
    right_major_deg: float,
    ellipse_success: np.ndarray | None = None,
):
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
    if ellipse_success is None:
        ellipse_success = np.asarray([[True, True]], dtype=bool)
    return _process_chunk(
        ellipse_params=ellipse_params,
        ellipse_success=ellipse_success,
        keypoints_roi=keypoints_roi,
        detection_success=np.asarray([True], dtype=bool),
        keypoint_indices={"swim_bladder": 0, "eye_left": 1, "eye_right": 2},
    )


def test_eye_frame_angles_are_positive_for_convergence() -> None:
    result = _run_eye_chunk(left_major_deg=-20.0, right_major_deg=20.0)

    assert result.left_eye_angle_deg[0] == pytest.approx(20.0, abs=1e-4)
    assert result.right_eye_angle_deg[0] == pytest.approx(20.0, abs=1e-4)
    assert result.vergence_eye_angle_deg[0] == pytest.approx(40.0, abs=1e-4)


def test_eye_frame_angles_are_negative_for_divergence() -> None:
    result = _run_eye_chunk(left_major_deg=20.0, right_major_deg=-20.0)

    assert result.left_eye_angle_deg[0] == pytest.approx(-20.0, abs=1e-4)
    assert result.right_eye_angle_deg[0] == pytest.approx(-20.0, abs=1e-4)
    assert result.vergence_eye_angle_deg[0] == pytest.approx(-40.0, abs=1e-4)
    assert result.vergence_major_signed_deg[0] == pytest.approx(40.0, abs=1e-4)


def test_eye_frame_angles_do_not_report_yoked_rotation_as_vergence() -> None:
    result = _run_eye_chunk(left_major_deg=20.0, right_major_deg=20.0)

    assert result.left_eye_angle_deg[0] == pytest.approx(-20.0, abs=1e-4)
    assert result.right_eye_angle_deg[0] == pytest.approx(20.0, abs=1e-4)
    assert result.vergence_eye_angle_deg[0] == pytest.approx(0.0, abs=1e-4)


def test_eye_frame_angles_are_zero_at_rest() -> None:
    result = _run_eye_chunk(left_major_deg=0.0, right_major_deg=0.0)

    assert result.left_eye_angle_deg[0] == pytest.approx(0.0, abs=1e-4)
    assert result.right_eye_angle_deg[0] == pytest.approx(0.0, abs=1e-4)
    assert result.vergence_eye_angle_deg[0] == pytest.approx(0.0, abs=1e-4)


def test_eye_frame_angles_match_nasal_gaze_in_typical_range() -> None:
    for major_deg in np.linspace(-60.0, 60.0, 13):
        result = _run_eye_chunk(left_major_deg=float(major_deg), right_major_deg=float(major_deg))

        assert result.left_eye_angle_deg[0] == pytest.approx(result.left_nasal_gaze_deg[0], abs=1e-4)
        assert result.right_eye_angle_deg[0] == pytest.approx(result.right_nasal_gaze_deg[0], abs=1e-4)


def test_eye_frame_angles_preserve_invalid_eye_nans() -> None:
    result = _run_eye_chunk(
        left_major_deg=-20.0,
        right_major_deg=20.0,
        ellipse_success=np.asarray([[False, True]], dtype=bool),
    )

    assert np.isnan(result.left_eye_angle_deg[0])
    assert result.right_eye_angle_deg[0] == pytest.approx(20.0, abs=1e-4)
    assert np.isnan(result.vergence_eye_angle_deg[0])
