from __future__ import annotations

import math

import numpy as np
import zarr

from fisheye.analysis import tail_posture_view_runs as mod
from fisheye.shared.detect_reason_codec import decode_reason_bytes


def _patch_provenance(monkeypatch) -> None:
    monkeypatch.setattr(
        mod,
        "get_git_info",
        lambda repo_path=None: {  # noqa: ARG005
            "commit_hash": "d" * 40,
            "short_hash": "dddddddd",
            "branch": "main",
            "is_dirty": False,
            "remote_url": "git@example.com:palette.git",
        },
    )
    monkeypatch.setattr(
        mod,
        "get_environment_info",
        lambda **kwargs: {  # noqa: ARG005
            "environment": {"python": "3.11"},
            "platform": {
                "hostname": "posture-host",
                "system": "Linux",
                "release": "6.8",
                "python_version": "3.11.0",
                "machine": "x86_64",
            },
        },
    )


def _source_arrays() -> dict[str, np.ndarray]:
    source_s = np.linspace(0.0, 1.0, 4, dtype=np.float32)
    tail_xy = np.zeros((2, 4, 2), dtype=np.float32)
    tail_xy[:, :, 0] = -np.linspace(0.0, 10.0, 4, dtype=np.float32)[None, :]
    return {
        "source_tail_sample_s": source_s,
        "tail_sample_xy": tail_xy,
        "head_xy": np.asarray([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32),
        "tail_sample_valid": np.ones((2,), dtype=bool),
        "bspline_valid": np.ones((2,), dtype=bool),
    }


def test_cumulative_segment_angles_straight_tail_are_zero() -> None:
    sources = _source_arrays()

    batch = mod.compute_tail_posture_view_from_subject_shape_arrays(**sources, keypoint_count=11)

    assert batch.valid.tolist() == [True, True]
    assert batch.tail_keypoints_xy.shape == (2, 11, 2)
    assert batch.tail_angle_rad.shape == (2, 10)
    np.testing.assert_allclose(batch.tail_angle_rad, 0.0, atol=1e-7)
    np.testing.assert_allclose(batch.head_yaw_rad, 0.0, atol=1e-7)


def test_cumulative_segment_angles_follow_keypoint_turns() -> None:
    head_xy = np.asarray([[1.0, 0.0]], dtype=np.float32)
    tail_keypoints = np.asarray(
        [[[0.0, 0.0], [-1.0, 0.0], [-2.0, 1.0]]],
        dtype=np.float32,
    )

    angle, head_yaw, valid = mod.compute_cumulative_segment_angles_from_keypoints(
        head_xy=head_xy,
        tail_keypoints_xy=tail_keypoints,
    )

    assert valid.tolist() == [True]
    np.testing.assert_allclose(head_yaw, 0.0, atol=1e-7)
    np.testing.assert_allclose(angle[0, 0], 0.0, atol=1e-7)
    np.testing.assert_allclose(angle[0, 1], -math.pi / 4.0, atol=1e-7)


def test_invalid_rows_are_nan_and_preserve_failure_reason() -> None:
    sources = _source_arrays()
    sources["tail_sample_valid"] = np.asarray([True, False], dtype=bool)

    batch = mod.compute_tail_posture_view_from_subject_shape_arrays(
        **sources,
        tail_sample_failure_reason=np.asarray(["ok", "scratch_artifact"], dtype=object),
        keypoint_count=11,
    )

    assert batch.valid.tolist() == [True, False]
    assert str(batch.failure_reason[1]) == "scratch_artifact"
    assert np.all(np.isnan(batch.tail_keypoints_xy[1]))
    assert np.all(np.isnan(batch.tail_angle_rad[1]))
    assert np.isnan(batch.head_yaw_rad[1])
    decoded = decode_reason_bytes(batch.failure_reason_bytes)
    assert decoded.tolist() == ["ok", "scratch_artifact"]


def _build_shape_root() -> zarr.Group:
    root = zarr.group()
    analysis = root.create_group("analysis")
    parent = analysis.create_group("subject_shape_runs")
    parent.attrs["latest"] = "shape_001"
    shape = parent.create_group("shape_001")
    shape.attrs["source_refined_subject_masks_run"] = "refined_001"

    row_index = shape.create_group("row_index")
    row_index.create_array("frame_indices", data=np.asarray([10, 11], dtype=np.int32), overwrite=True)
    row_index.create_array("detection_indices", data=np.asarray([0, 1], dtype=np.int32), overwrite=True)
    row_index.create_array("source_refined_row_ids", data=np.asarray([100, 101], dtype=np.int64), overwrite=True)

    components = shape.create_group("components")
    body = components.create_group("subject_body")
    sources = _source_arrays()
    body.create_array("tail_sample_s", data=sources["source_tail_sample_s"], overwrite=True)
    body.create_array("tail_sample_xy", data=sources["tail_sample_xy"], overwrite=True)
    body.create_array("head_endpoint_xy", data=sources["head_xy"], overwrite=True)
    body.create_array("tail_sample_valid", data=sources["tail_sample_valid"], overwrite=True)
    body.create_array("bspline_valid", data=sources["bspline_valid"], overwrite=True)
    body.create_array("tail_sample_failure_reason_bytes", data=mod._encode_reasons(["ok", "ok"]), overwrite=True)
    body.create_array("bspline_failure_reason_bytes", data=mod._encode_reasons(["ok", "ok"]), overwrite=True)
    return root


def test_write_tail_posture_view_run_group_writes_schema_and_arrays(monkeypatch) -> None:
    _patch_provenance(monkeypatch)
    root = _build_shape_root()

    summary = mod.write_tail_posture_view_run_group(
        root,
        subject_shape_run="shape_001",
        run_name="megabouts_view_001",
        source_tail_kinematics_run="tail_001",
    )

    assert summary["status"] == "updated"
    assert summary["valid_row_count"] == 2
    parent = root["analysis"]["tail_posture_view_runs"]
    assert parent.attrs["latest"] == "megabouts_view_001"
    assert parent.attrs["latest_megabouts_compatible"] == "megabouts_view_001"
    run = parent["megabouts_view_001"]
    assert run.attrs["schema_id"] == "analysis.tail_posture_view_runs"
    assert run.attrs["view_family"] == "megabouts_compatible"
    assert run.attrs["dependency_policy"] == "no_megabouts_dependency_required"
    assert run.attrs["angle_convention"] == "megabouts_cumulative_segment_angle"
    assert run.attrs["keypoint_count"] == 11
    assert run.attrs["angle_count"] == 10
    assert run.attrs["source_subject_shape_run"] == "shape_001"
    assert run.attrs["source_tail_kinematics_run"] == "tail_001"
    assert run["frame_index"][:].tolist() == [10, 11]
    assert run["row_index"]["frame_indices"][:].tolist() == [10, 11]
    assert np.asarray(run["tail_keypoints_xy"][:], dtype=np.float32).shape == (2, 11, 2)
    assert np.asarray(run["tail_angle_rad"][:], dtype=np.float32).shape == (2, 10)
    np.testing.assert_allclose(np.asarray(run["tail_angle_rad"][:], dtype=np.float32), 0.0, atol=1e-7)
    assert run.attrs["provenance"]["stage"] == "analysis.tail_posture_view_runs"
