from __future__ import annotations

import numpy as np

import fisheye.analysis.eye_angle_analysis as eye_angle_analysis
from fisheye.analysis.eye_angle_analysis import _eye_angle_definition_attrs, _process_chunk
from fisheye.analysis.eye_angle_analysis import (
    _resolve_keypoint_run_name as resolve_eye_angle_keypoint_run,
)
from fisheye.visualization.visualize_eye_angle_overlays import (
    _resolve_keypoint_run_name as resolve_overlay_keypoint_run,
)


def test_eye_angle_archive_opener_uses_palette_zarr_policy(monkeypatch, tmp_path) -> None:
    calls = []
    sentinel = object()

    def fake_open_zarr_root(path, *, mode):
        calls.append((path, mode))
        return sentinel

    zarr_path = tmp_path / "archive.zarr"
    monkeypatch.setattr(eye_angle_analysis, "open_zarr_root", fake_open_zarr_root)

    assert eye_angle_analysis._open_archive_for_eye_angle(zarr_path) is sentinel
    assert calls == [(zarr_path, "a")]


def test_eye_angle_definition_attrs_match_nasal_positive_binocular_math() -> None:
    attrs = _eye_angle_definition_attrs()

    assert attrs["signed_angles"] is True
    assert attrs["signed_angle_convention"] == "per-eye signed angles are temporal-positive"
    assert attrs["vergence_definition"] == "abs(vergence_signed_deg)"
    assert attrs["vergence_signed_definition"] == "-(left_signed_deg + right_signed_deg)"
    assert attrs["version_definition"] == "0.5*(-left_signed_deg + right_signed_deg)"
    assert attrs["minor_signed_angles"] is True
    assert attrs["minor_signed_angle_convention"] == "per-eye minor signed angles are temporal-positive"
    assert attrs["minor_vergence_definition"] == "abs(vergence_minor_signed_deg)"
    assert attrs["minor_vergence_signed_definition"] == "-(left_minor_signed_deg + right_minor_signed_deg)"
    assert attrs["minor_version_definition"] == "0.5*(-left_minor_signed_deg + right_minor_signed_deg)"


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


def test_eye_angle_chunk_uses_label_resolved_indices() -> None:
    ellipse_params = np.asarray(
        [[[3.0, 0.0, 4.0, 1.5, 0.0], [3.0, 2.0, 4.0, 1.5, 0.0]]],
        dtype=np.float32,
    )
    ellipse_success = np.asarray([[True, True]], dtype=bool)
    keypoints_roi = np.asarray(
        [
            [
                [3.0, 0.0],   # eye_left
                [9.0, 9.0],   # extra label
                [1.0, 1.0],   # swim_bladder
                [3.0, 2.0],   # eye_right
                [0.0, 0.0],   # extra label
            ]
        ],
        dtype=np.float32,
    )
    heading_deg = np.asarray([0.0], dtype=np.float32)
    detection_success = np.asarray([True], dtype=bool)

    result = _process_chunk(
        ellipse_params=ellipse_params,
        ellipse_success=ellipse_success,
        keypoints_roi=keypoints_roi,
        heading_deg=heading_deg,
        detection_success=detection_success,
        keypoint_indices={
            "swim_bladder": 2,
            "eye_left": 0,
            "eye_right": 3,
        },
    )

    assert bool(result.valid_left[0])
    assert bool(result.valid_right[0])
    assert bool(result.valid_frame[0])
    assert np.isfinite(result.left_deg[0])
    assert np.isfinite(result.right_deg[0])
