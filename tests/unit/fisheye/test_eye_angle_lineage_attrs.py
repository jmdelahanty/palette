from __future__ import annotations

import numpy as np
import pytest

import fisheye.analysis.eye_angle_analysis as eye_angle_analysis
from fisheye.analysis.eye_angle_analysis import _eye_angle_definition_attrs, _process_chunk
from fisheye.analysis.eye_angle_analysis import (
    _resolve_keypoint_run_name as resolve_eye_angle_keypoint_run,
)
from fisheye.shared.eye_geometry_source import (
    EYE_GEOMETRY_STAGE_REFINED_SUBJECT,
    EYE_GEOMETRY_STAGE_SUBJECT_SHAPE,
    resolve_eye_geometry_source,
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


def test_eye_angle_output_schema_describes_run_layout_and_conventions() -> None:
    schema = eye_angle_analysis._eye_angle_output_schema()

    assert schema["schema_id"] == "analysis.eye_angle_output_schema"
    assert schema["schema_version"] == 1
    assert schema["row_axes"]["roi"] == "keypoint_detection_rows"
    assert schema["row_axes"]["frame"] == "video_frame_rows"
    assert schema["groups"]["angles/roi"]["units"] == "deg"
    assert "left_signed_deg" in schema["groups"]["angles/roi"]["base_outputs"]
    assert "heading_deg" in schema["groups"]["angles/roi"]["base_outputs"]
    assert "left_speed_deg_s" in schema["groups"]["angles/roi"]["derivative_outputs"]
    assert schema["groups"]["support"]["row_axis"] == "mixed"
    support_outputs = schema["groups"]["support"]["outputs"]
    assert {"name": "frame_time_seconds", "row_axis": "frame", "units": "s", "optional": True} in support_outputs
    assert schema["groups"]["qa/roi"]["outputs"] == [
        "valid_left",
        "valid_right",
        "valid_frame",
        "reason_codes",
    ]
    assert schema["signed_angle_convention"] == "per-eye signed angles are temporal-positive"
    assert schema["vergence_signed_definition"] == "-(left_signed_deg + right_signed_deg)"
    assert schema["qa_reason_codes_attr"] == "reason_code_map"


def test_eye_angle_source_geometry_kind_maps_known_sources() -> None:
    assert eye_angle_analysis._source_geometry_kind("analysis/subject_shape_runs") == "subject_shape_eye_geometry"
    assert eye_angle_analysis._source_geometry_kind("refined_subject_masks_runs") == "refined_subject_eye_geometry"
    assert eye_angle_analysis._source_geometry_kind("refined_eye_masks_runs") == "legacy_refined_eye_geometry"
    assert eye_angle_analysis._source_geometry_kind("unknown") == "unknown_eye_geometry"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("serial", "serial_driver"),
        ("driver", "serial_driver"),
        ("dask", "dask_worker_chunks"),
        ("dask-chunks", "dask_worker_chunks"),
    ],
)
def test_eye_angle_execution_backend_aliases(raw: str, expected: str) -> None:
    assert eye_angle_analysis._normalize_execution_backend(raw) == expected


def test_eye_angle_frame_projection_flags_missing_and_multi_detection_frames() -> None:
    frame_arrays, frame_valid, frame_reason = eye_angle_analysis._project_detection_arrays_to_frames(
        np.asarray([0, 2, 2, 4], dtype=np.int64),
        num_frames=5,
        valid_frame=np.asarray([True, True, True, False], dtype=bool),
        reason_codes=np.asarray([0, 4, 8, 2], dtype=np.uint16),
        arrays={"left": np.asarray([10.0, 20.0, 30.0, 40.0], dtype=np.float32)},
    )

    assert frame_arrays["left"][0] == 10.0
    assert np.isnan(frame_arrays["left"][1])
    assert np.isnan(frame_arrays["left"][2])
    assert np.isnan(frame_arrays["left"][3])
    assert frame_arrays["left"][4] == 40.0
    assert frame_valid.tolist() == [True, False, False, False, False]
    assert int(frame_reason[1]) & int(eye_angle_analysis.REASON_NO_DETECTION)
    assert int(frame_reason[2]) & int(eye_angle_analysis.REASON_MULTI_DETECTION)
    assert int(frame_reason[3]) & int(eye_angle_analysis.REASON_NO_DETECTION)
    assert int(frame_reason[4]) & int(eye_angle_analysis.REASON_HEADING_INVALID)


def _add_refined_subject_eye_geometry(root):
    parent = root.create_group("refined_subject_masks_runs")
    parent.attrs["latest"] = "refined_001"
    run = parent.create_group("refined_001")
    run.attrs.update(
        {
            "mask_labels": ["subject_body", "eye_left", "eye_right"],
            "source_keypoints_run": "kp_from_refined",
        }
    )
    run.create_array("masks_roi", data=np.zeros((2, 3, 4, 4), dtype=np.uint8), overwrite=True)
    for component in ("eye_left", "eye_right"):
        geometry = run.create_group(f"components/{component}/geometry")
        geometry.create_array("ellipse_params", data=np.ones((2, 5), dtype=np.float32), overwrite=True)
        geometry.create_array("ellipse_success", data=np.ones((2,), dtype=bool), overwrite=True)
    metrics = run.create_group("relations/eye_pair/metrics")
    metrics.create_array("separation_px", data=np.asarray([4.0, 4.5], dtype=np.float32), overwrite=True)
    return run


def _add_subject_shape_eye_geometry(root):
    analysis = root.create_group("analysis")
    parent = analysis.create_group("subject_shape_runs")
    parent.attrs["latest"] = "shape_001"
    run = parent.create_group("shape_001")
    run.attrs.update(
        {
            "source_refined_subject_masks_run": "refined_001",
            "source_keypoints_run": "kp_from_shape",
        }
    )
    for component, value in (("eye_left", 1.0), ("eye_right", 2.0)):
        group = run.create_group(f"components/{component}")
        group.create_array("ellipse_params", data=np.full((2, 5), value, dtype=np.float32), overwrite=True)
        group.create_array("ellipse_success", data=np.ones((2,), dtype=bool), overwrite=True)
    pair = run.create_group("relations/eye_pair")
    pair.create_array("separation_px", data=np.asarray([5.0, 5.5], dtype=np.float32), overwrite=True)
    return run


def test_eye_geometry_resolution_prefers_latest_subject_shape_when_enabled() -> None:
    import zarr

    root = zarr.group()
    _add_refined_subject_eye_geometry(root)
    _add_subject_shape_eye_geometry(root)

    source = resolve_eye_geometry_source(root, prefer_subject_shape=True)

    assert source.stage_group == EYE_GEOMETRY_STAGE_SUBJECT_SHAPE
    assert source.run_name == "shape_001"
    assert source.source_subject_shape_run == "shape_001"
    assert source.source_refined_subject_run == "refined_001"
    assert source.ellipse_params.shape == (2, 2, 5)
    assert source.ellipse_success.shape == (2, 2)
    np.testing.assert_allclose(source.ellipse_params[:, 0, 0], [1.0, 1.0])
    np.testing.assert_allclose(source.ellipse_params[:, 1, 0], [2.0, 2.0])


def test_eye_geometry_resolution_default_keeps_mask_capable_source() -> None:
    import zarr

    root = zarr.group()
    _add_refined_subject_eye_geometry(root)
    _add_subject_shape_eye_geometry(root)

    source = resolve_eye_geometry_source(root)

    assert source.stage_group == EYE_GEOMETRY_STAGE_REFINED_SUBJECT
    assert source.run_name == "refined_001"
    assert source.masks_roi is not None


def test_eye_geometry_resolution_honors_explicit_refined_subject_run() -> None:
    import zarr

    root = zarr.group()
    _add_refined_subject_eye_geometry(root)
    _add_subject_shape_eye_geometry(root)

    source = resolve_eye_geometry_source(root, refined_subject_run="refined_001")

    assert source.stage_group == EYE_GEOMETRY_STAGE_REFINED_SUBJECT
    assert source.run_name == "refined_001"
    assert source.source_subject_shape_run is None


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
