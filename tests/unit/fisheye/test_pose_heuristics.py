from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from fisheye.detection import detect_keypoints_traditional as detect_mod
from fisheye.pose.heuristics import (
    FlipDetectionHeuristic,
    GeometryQcHeuristic,
    heuristic_profile_from_package,
    maybe_flip_detection_from_attrs,
    maybe_geometry_qc_from_attrs,
    maybe_heuristic_profile_from_attrs,
)
from fisheye.refinement import refine_keypoints as refine_mod
from fisheye.tune import keypoint_tuner as tuner_mod
from fisheye.tune import keypoint_failure_review as review_mod
from fisheye.utils import patch_keypoints_from_crops as patch_mod


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


def test_pose_heuristic_profile_helpers_resolve_from_attrs() -> None:
    attrs = {"pose_schema": {"name": "traditional_v1"}}

    profile = maybe_heuristic_profile_from_attrs("traditional_pose", attrs)
    geometry = maybe_geometry_qc_from_attrs("traditional_pose", attrs)
    flip = maybe_flip_detection_from_attrs("traditional_pose", attrs)

    assert profile is not None
    assert profile.profile_name == "traditional_pose_traditional_v1"
    assert geometry is not None
    assert geometry.min_triangle_angle_deg == 10.0
    assert geometry.max_triangle_angle_deg == 90.0
    assert flip is not None
    assert flip.family == "traditional_eye_flip_v1"


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
    assert refine_mod.DEFAULT_MIN_TRIANGLE_ANGLE == profile.geometry_qc.min_triangle_angle_deg
    assert refine_mod.DEFAULT_MIN_TRIANGLE_AREA == profile.geometry_qc.min_triangle_area_px
    assert refine_mod.DEFAULT_MAX_TRIANGLE_AREA == profile.geometry_qc.max_triangle_area_px
    assert review_mod.DEFAULT_MIN_TRIANGLE_ANGLE == profile.geometry_qc.min_triangle_angle_deg
    assert review_mod.DEFAULT_MIN_TRIANGLE_AREA == profile.geometry_qc.min_triangle_area_px
    assert review_mod.DEFAULT_MAX_TRIANGLE_AREA == profile.geometry_qc.max_triangle_area_px


def test_phase1_geometry_resolution_uses_packaged_profile_helpers(monkeypatch) -> None:
    geometry = GeometryQcHeuristic(
        min_triangle_angle_deg=12.5,
        max_triangle_angle_deg=82.0,
        min_triangle_area_px=222.0,
        max_triangle_area_px=444.0,
    )
    flip = FlipDetectionHeuristic(family="traditional_eye_flip_v1")

    monkeypatch.setattr(refine_mod, "maybe_geometry_qc_from_attrs", lambda *args, **kwargs: geometry)
    monkeypatch.setattr(refine_mod, "maybe_flip_detection_from_attrs", lambda *args, **kwargs: flip)
    monkeypatch.setattr(review_mod, "maybe_geometry_qc_from_attrs", lambda *args, **kwargs: geometry)
    monkeypatch.setattr(patch_mod, "maybe_geometry_qc_from_attrs", lambda *args, **kwargs: geometry)

    assert refine_mod._resolve_refinement_geometry_defaults({"pose_schema": {"name": "custom"}}) == (
        12.5,
        222.0,
        444.0,
    )
    assert review_mod._resolve_review_geometry_defaults({"pose_schema": {"name": "custom"}}) == (
        12.5,
        222.0,
        444.0,
    )
    assert patch_mod._resolve_detect_geometry_defaults({"pose_schema": {"name": "custom"}}) == (
        12.5,
        82.0,
        222.0,
        444.0,
    )


def test_patch_keypoints_maps_traditional_output_into_runtime_labels(monkeypatch) -> None:
    roi_images = np.zeros((1, 4, 4), dtype=np.uint8)
    roi_coords = np.array([[10, 20]], dtype=np.int32)
    background = np.zeros((64, 64), dtype=np.uint8)

    def _fake_detect(*args, **kwargs):
        return {
            "bladder": np.array([1.0, 1.0], dtype=np.float64),
            "eye_left": np.array([2.0, 1.0], dtype=np.float64),
            "eye_right": np.array([2.0, 3.0], dtype=np.float64),
            "heading": 0.0,
            "confidence": 0.9,
            "keypoint_confidences": [0.7, 0.8, 0.9],
            "triangle_angles": [30.0, 60.0, 90.0],
            "triangle_angles_raw": [90.0, 60.0, 30.0],
            "triangle_area": 2.0,
        }

    monkeypatch.setattr(patch_mod, "detect_keypoints_traditional", _fake_detect)

    outputs = patch_mod._compute_keypoints_for_indices(
        roi_images,
        roi_coords,
        background,
        np.array([0], dtype=np.int64),
        {},
        source_attrs={
            "keypoint_labels": ["tail_tip", "eye_left", "swim_bladder", "eye_right", "pelvis"],
            "heading_computation_override": {
                "enabled": True,
                "direction_from": {"op": "keypoint", "label": "swim_bladder"},
                "direction_to": {"op": "midpoint", "labels": ["eye_left", "eye_right"]},
            },
        },
        keypoint_count=5,
        keypoint_labels=("tail_tip", "eye_left", "swim_bladder", "eye_right", "pelvis"),
    )

    assert outputs["keypoints_roi"].shape == (1, 5, 2)
    assert np.isnan(outputs["keypoints_roi"][0, 0]).all()
    np.testing.assert_allclose(outputs["keypoints_roi"][0, 1], [2.0, 1.0])
    np.testing.assert_allclose(outputs["keypoints_roi"][0, 2], [1.0, 1.0])
    np.testing.assert_allclose(outputs["keypoints_roi"][0, 3], [2.0, 3.0])
    assert np.isnan(outputs["keypoints_roi"][0, 4]).all()
    np.testing.assert_allclose(outputs["keypoint_confidences"][0, 1:4], [0.8, 0.7, 0.9])
    assert np.isfinite(outputs["heading"][0])


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
