from __future__ import annotations

from fisheye.analysis.eye_angle_analysis import (
    _resolve_keypoint_run_name as resolve_eye_angle_keypoint_run,
)
from fisheye.visualization.visualize_eye_angle_overlays import (
    _resolve_keypoint_run_name as resolve_overlay_keypoint_run,
)


def test_eye_angle_keypoint_resolution_prefers_explicit() -> None:
    resolved = resolve_eye_angle_keypoint_run(
        explicit_keypoint_run="kp_explicit",
        refined_attrs={
            "source_keypoints_run": "kp_canonical",
            "source_keypoint_run": "kp_legacy",
        },
        parent_latest="kp_latest",
    )
    assert resolved == "kp_explicit"


def test_eye_angle_keypoint_resolution_prefers_canonical_over_legacy() -> None:
    resolved = resolve_eye_angle_keypoint_run(
        explicit_keypoint_run=None,
        refined_attrs={
            "source_keypoints_run": "kp_canonical",
            "source_keypoint_run": "kp_legacy",
        },
        parent_latest="kp_latest",
    )
    assert resolved == "kp_canonical"


def test_eye_angle_keypoint_resolution_falls_back_to_legacy_then_latest() -> None:
    resolved_legacy = resolve_eye_angle_keypoint_run(
        explicit_keypoint_run=None,
        refined_attrs={"source_keypoint_run": "kp_legacy"},
        parent_latest="kp_latest",
    )
    resolved_latest = resolve_eye_angle_keypoint_run(
        explicit_keypoint_run=None,
        refined_attrs={},
        parent_latest="kp_latest",
    )
    assert resolved_legacy == "kp_legacy"
    assert resolved_latest == "kp_latest"


def test_overlay_keypoint_resolution_prefers_explicit() -> None:
    resolved = resolve_overlay_keypoint_run(
        explicit_keypoint_run="kp_explicit",
        run_attrs={
            "source_keypoints_run": "kp_canonical",
            "source_keypoint_run": "kp_legacy",
        },
    )
    assert resolved == "kp_explicit"


def test_overlay_keypoint_resolution_prefers_canonical_over_legacy() -> None:
    resolved = resolve_overlay_keypoint_run(
        explicit_keypoint_run=None,
        run_attrs={
            "source_keypoints_run": "kp_canonical",
            "source_keypoint_run": "kp_legacy",
        },
    )
    assert resolved == "kp_canonical"


def test_overlay_keypoint_resolution_falls_back_to_legacy() -> None:
    resolved = resolve_overlay_keypoint_run(
        explicit_keypoint_run=None,
        run_attrs={"source_keypoint_run": "kp_legacy"},
    )
    assert resolved == "kp_legacy"
