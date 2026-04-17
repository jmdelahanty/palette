from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from fisheye.detection import detect_keypoints_traditional as detect_mod
from fisheye.pose.heuristics import heuristic_profile_from_package
from fisheye.tune import keypoint_tuner as tuner_mod


@dataclass(frozen=True)
class _FakeRegion:
    centroid: tuple[float, float]
    area: float = 10.0


def test_traditional_pose_heuristic_profiles_load_packaged_defaults() -> None:
    profile_v1 = heuristic_profile_from_package("traditional_pose", "traditional_v1")
    profile_v2 = heuristic_profile_from_package("traditional_pose", "traditional_v2")

    assert profile_v1.method == "traditional_pose"
    assert profile_v1.blob_assignment is not None
    assert profile_v1.blob_assignment.family == "triangle_3blob"
    assert profile_v1.geometry_qc is not None
    assert profile_v1.geometry_qc.min_triangle_angle_deg == 10.0
    assert profile_v1.geometry_qc.max_triangle_angle_deg == 90.0
    assert profile_v1.geometry_qc.min_triangle_area_px == 100.0
    assert profile_v1.label_requirements is not None
    assert profile_v1.label_requirements.required_labels == (
        "swim_bladder",
        "eye_left",
        "eye_right",
    )

    assert profile_v2.skeleton_id == "pose_skel_traditional_v2"
    assert profile_v2.label_requirements is not None
    assert profile_v2.label_requirements.core_assignment_labels == (
        "swim_bladder",
        "eye_left",
        "eye_right",
    )


def test_traditional_detector_and_tuner_defaults_follow_packaged_profile() -> None:
    profile = heuristic_profile_from_package("traditional_pose", "traditional_v1")
    assert profile.geometry_qc is not None

    assert detect_mod.DEFAULT_MIN_VALID_ANGLE == profile.geometry_qc.min_triangle_angle_deg
    assert detect_mod.DEFAULT_MAX_VALID_ANGLE == profile.geometry_qc.max_triangle_angle_deg
    assert detect_mod.DEFAULT_MIN_TRIANGLE_AREA == profile.geometry_qc.min_triangle_area_px
    assert detect_mod.DEFAULT_MAX_TRIANGLE_AREA == profile.geometry_qc.max_triangle_area_px

    assert tuner_mod.min_valid_angle == int(profile.geometry_qc.min_triangle_angle_deg)
    assert tuner_mod.max_valid_angle == int(profile.geometry_qc.max_triangle_angle_deg)
    assert tuner_mod.min_triangle_area == int(profile.geometry_qc.min_triangle_area_px)
    assert tuner_mod.max_triangle_area == 0


def test_traditional_detector_assignment_uses_packaged_profile_rules() -> None:
    stats = [
        _FakeRegion((10.0, 10.0)),
        _FakeRegion((15.0, 20.0)),
        _FakeRegion((5.0, 20.0)),
    ]

    result = detect_mod.identify_keypoints_by_geometry(stats)

    assert result is not None
    np.testing.assert_allclose(result["bladder"], np.array([10.0, 10.0], dtype=np.float64))
    np.testing.assert_allclose(result["eye_left"], np.array([20.0, 5.0], dtype=np.float64))
    np.testing.assert_allclose(result["eye_right"], np.array([20.0, 15.0], dtype=np.float64))
    assert math.isclose(float(result["heading"]), 0.0, abs_tol=1e-6)


def test_traditional_tuner_assignment_uses_packaged_profile_rules() -> None:
    stats = [
        _FakeRegion((10.0, 10.0)),
        _FakeRegion((15.0, 20.0)),
        _FakeRegion((5.0, 20.0)),
    ]

    result = tuner_mod.identify_keypoints_by_geometry(stats)

    assert result is not None
    assert result["bladder_idx"] == 0
    assert result["left_eye_idx"] == 2
    assert result["right_eye_idx"] == 1
    assert math.isclose(float(result["heading"]), 0.0, abs_tol=1e-6)
