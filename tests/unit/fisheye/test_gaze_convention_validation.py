from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import fisheye.analysis.gaze_convention_validation as validation_module
from fisheye.analysis.gaze_convention_validation import (
    _resolve_eye_run,
    _resolve_review_masks,
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


def test_review_masks_follow_subject_shape_refined_subject_lineage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    masks = np.zeros((5, 2, 8, 8), dtype=np.uint8)
    subject_shape = SimpleNamespace(
        masks_roi=None,
        ellipse_params=np.zeros((5, 2, 5), dtype=np.float32),
        group_path="analysis/subject_shape_runs/shape_a",
        source_refined_subject_run="refined_subject_a",
    )
    refined_subject = SimpleNamespace(
        masks_roi=masks,
        group_path="refined_subject_masks_runs/refined_subject_a",
    )
    calls: list[str] = []

    def _fake_resolve(_root: object, *, refined_subject_run: str):
        calls.append(refined_subject_run)
        return refined_subject

    monkeypatch.setattr(
        validation_module,
        "resolve_eye_geometry_source",
        _fake_resolve,
    )

    resolved, source_path = _resolve_review_masks(object(), subject_shape)

    assert resolved is masks
    assert source_path == "refined_subject_masks_runs/refined_subject_a"
    assert calls == ["refined_subject_a"]


def test_review_masks_reject_row_mismatch() -> None:
    geometry = SimpleNamespace(
        masks_roi=np.zeros((4, 2, 8, 8), dtype=np.uint8),
        ellipse_params=np.zeros((5, 2, 5), dtype=np.float32),
        group_path="analysis/subject_shape_runs/shape_a",
        source_refined_subject_run="refined_subject_a",
    )

    with pytest.raises(ValueError, match="not row-aligned"):
        _resolve_review_masks(object(), geometry)


def test_eye_run_resolution_uses_canonical_eye_angle_reader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_group = object()
    calls: list[tuple[object, object, bool]] = []

    def _fake_resolve(root, run_name, *, legacy_compatibility):
        calls.append((root, run_name, legacy_compatibility))
        return run_group, "eye_current", "analysis/eye_angle_runs/eye_current"

    monkeypatch.setattr(validation_module, "resolve_eye_angle_run", _fake_resolve)
    root = object()

    resolved, group = _resolve_eye_run(
        root,
        "analysis/eye_angle_runs/eye_current",
    )

    assert resolved == "eye_current"
    assert group is run_group
    assert calls == [
        (root, "analysis/eye_angle_runs/eye_current", False)
    ]
