from __future__ import annotations

import numpy as np

from fisheye.analysis.gaze_convention_validation import (
    body_frame_angles_from_vectors,
    validate_gaze_geometry_arrays,
    wrap_degrees_signed,
)


def _synthetic_geometry() -> dict[str, np.ndarray]:
    heading = np.asarray([0.0, 90.0, -90.0, 180.0], dtype=np.float64)
    radians = np.deg2rad(heading)
    forward = np.column_stack((np.cos(radians), -np.sin(radians)))
    left = np.column_stack((forward[:, 1], -forward[:, 0]))
    left_major = np.asarray([-20.0, -10.0, 5.0, 15.0])
    right_major = np.asarray([20.0, 12.0, -4.0, -14.0])
    left_gaze = wrap_degrees_signed(left_major + 90.0)
    right_gaze = wrap_degrees_signed(right_major - 90.0)

    def vectors(angles: np.ndarray) -> np.ndarray:
        rad = np.deg2rad(angles)
        return np.cos(rad)[:, None] * forward + np.sin(rad)[:, None] * left

    return {
        "left_major_signed_deg": left_major,
        "right_major_signed_deg": right_major,
        "left_eye_angle_deg": -left_major,
        "right_eye_angle_deg": right_major,
        "vergence_eye_angle_deg": -left_major + right_major,
        "left_gaze_signed_deg": left_gaze,
        "right_gaze_signed_deg": right_gaze,
        "left_gaze_xy": vectors(left_gaze),
        "right_gaze_xy": vectors(right_gaze),
        "forward_axis_xy": forward,
        "left_axis_xy": left,
        "heading_deg": heading,
        "valid": np.ones(heading.shape, dtype=bool),
    }


def test_gaze_geometry_gate_accepts_canonical_conventions() -> None:
    checks = validate_gaze_geometry_arrays(**_synthetic_geometry())
    assert checks
    assert all(check.passed for check in checks)


def test_gaze_geometry_gate_rejects_left_eye_sign_inversion() -> None:
    geometry = _synthetic_geometry()
    geometry["left_eye_angle_deg"] = geometry["left_major_signed_deg"].copy()
    checks = {check.name: check for check in validate_gaze_geometry_arrays(**geometry)}
    assert not checks["left_eye_angle_nasal_sign"].passed
    assert checks["right_eye_angle_nasal_sign"].passed


def test_gaze_geometry_gate_rejects_left_right_gaze_swap() -> None:
    geometry = _synthetic_geometry()
    geometry["left_gaze_xy"], geometry["right_gaze_xy"] = (
        geometry["right_gaze_xy"].copy(),
        geometry["left_gaze_xy"].copy(),
    )
    checks = {check.name: check for check in validate_gaze_geometry_arrays(**geometry)}
    assert not checks["left_gaze_vector_body_angle"].passed
    assert not checks["right_gaze_vector_body_angle"].passed


def test_body_frame_vector_angles_use_anatomical_left_positive() -> None:
    forward = np.asarray([[1.0, 0.0]] * 3)
    left = np.asarray([[0.0, -1.0]] * 3)
    vectors = np.asarray([[1.0, 0.0], [0.0, -1.0], [0.0, 1.0]])
    np.testing.assert_allclose(
        body_frame_angles_from_vectors(vectors, forward, left),
        np.asarray([0.0, 90.0, -90.0]),
    )
