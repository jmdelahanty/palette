from __future__ import annotations

import math

import numpy as np
import pytest
import zarr

from fisheye.analysis import tail_kinematics_runs as mod
from fisheye.shared.detect_reason_codec import decode_reason_bytes


def _patch_provenance(monkeypatch) -> None:
    monkeypatch.setattr(
        mod,
        "get_git_info",
        lambda repo_path=None: {  # noqa: ARG005
            "commit_hash": "c" * 40,
            "short_hash": "cccccccc",
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
                "hostname": "tail-host",
                "system": "Linux",
                "release": "6.8",
                "python_version": "3.11.0",
                "machine": "x86_64",
            },
        },
    )


def _source_arrays(
    tangent_rows: np.ndarray | None = None,
    *,
    row_count: int | None = None,
) -> dict[str, np.ndarray]:
    source_s = np.linspace(0.0, 1.0, 4, dtype=np.float32)
    resolved_row_count = int(
        row_count if row_count is not None else (2 if tangent_rows is None else tangent_rows.shape[0])
    )
    tail_xy = np.zeros((resolved_row_count, 4, 2), dtype=np.float32)
    tail_xy[:, :, 0] = -source_s[None, :] * 10.0
    if tangent_rows is None:
        tangent_rows = np.repeat(
            np.asarray([[[-1.0, 0.0]]], dtype=np.float32), resolved_row_count * 4, axis=0
        ).reshape(
            resolved_row_count, 4, 2
        )
    return {
        "source_tail_sample_s": source_s,
        "tail_sample_xy": tail_xy,
        "tail_tangent_xy": tangent_rows.astype(np.float32),
        "tail_curvature_px_inv": np.zeros((resolved_row_count, 4), dtype=np.float32),
        "tail_sample_valid": np.ones((resolved_row_count,), dtype=bool),
        "bspline_valid": np.ones((resolved_row_count,), dtype=bool),
        "tail_base_xy": np.zeros((resolved_row_count, 2), dtype=np.float32),
        "body_forward_axis_xy": np.repeat(
            np.asarray([[1.0, 0.0]], dtype=np.float32), resolved_row_count, axis=0
        ),
        "body_left_axis_xy": np.repeat(
            np.asarray([[0.0, 1.0]], dtype=np.float32), resolved_row_count, axis=0
        ),
        "body_frame_valid": np.ones((resolved_row_count,), dtype=bool),
    }


def test_tail_kinematics_straight_tail_is_zero_angle() -> None:
    batch = mod.compute_tail_kinematics_from_subject_shape_arrays(**_source_arrays(), tail_angle_sample_count=10)

    assert batch.valid.tolist() == [True, True]
    np.testing.assert_allclose(batch.tail_angle_rad, np.zeros((2, 10), dtype=np.float32), atol=1e-6)
    np.testing.assert_allclose(batch.tail_tip_angle_deg, np.zeros((2,), dtype=np.float32), atol=1e-5)
    np.testing.assert_allclose(batch.tail_lateral_deflection_px, np.zeros((2, 10), dtype=np.float32), atol=1e-5)


def test_tail_kinematics_left_positive_right_negative() -> None:
    theta = math.radians(30.0)
    left_tangent = np.asarray([-math.cos(theta), math.sin(theta)], dtype=np.float32)
    right_tangent = np.asarray([-math.cos(theta), -math.sin(theta)], dtype=np.float32)
    tangent_rows = np.stack(
        [
            np.repeat(left_tangent[None, :], 4, axis=0),
            np.repeat(right_tangent[None, :], 4, axis=0),
        ],
        axis=0,
    )

    batch = mod.compute_tail_kinematics_from_subject_shape_arrays(
        **_source_arrays(tangent_rows),
        tail_angle_sample_count=10,
    )

    np.testing.assert_allclose(batch.tail_angle_deg[0], np.full((10,), 30.0, dtype=np.float32), atol=1e-4)
    np.testing.assert_allclose(batch.tail_angle_deg[1], np.full((10,), -30.0, dtype=np.float32), atol=1e-4)
    np.testing.assert_allclose(batch.max_abs_tail_angle_deg, np.full((2,), 30.0, dtype=np.float32), atol=1e-4)


def test_tail_kinematics_invalid_rows_preserve_failure_reason() -> None:
    sources = _source_arrays()
    sources["tail_sample_valid"] = np.asarray([True, False], dtype=bool)

    batch = mod.compute_tail_kinematics_from_subject_shape_arrays(
        **sources,
        tail_sample_failure_reason=np.asarray(["ok", "tail_segment_too_short"], dtype=object),
        tail_angle_sample_count=10,
    )

    assert batch.valid.tolist() == [True, False]
    assert str(batch.failure_reason[1]) == "tail_segment_too_short"
    assert np.all(np.isnan(batch.tail_angle_rad[1]))
    decoded = decode_reason_bytes(batch.failure_reason_bytes)
    assert decoded.tolist() == ["ok", "tail_segment_too_short"]


def test_vectorized_interpolation_matches_scalar_numpy_reference() -> None:
    source_s = np.asarray([0.0, 0.2, 0.65, 1.0], dtype=np.float64)
    target_s = np.asarray([-0.1, 0.0, 0.1, 0.2, 0.5, 0.65, 0.9, 1.0, 1.1], dtype=np.float64)
    values_2d = np.asarray(
        [
            [[0.0, 4.0], [2.0, 3.0], [6.5, 2.0], [10.0, 1.0]],
            [[1.0, 0.0], [3.0, 1.0], [7.5, 2.0], [11.0, 3.0]],
            [[2.0, 3.0], [4.0, 2.0], [8.5, 1.0], [12.0, 0.0]],
        ],
        dtype=np.float64,
    )
    values_2d[2, 1, 0] = np.nan
    row_valid = np.asarray([True, False, True], dtype=bool)

    expected_2d = np.full((3, target_s.size, 2), np.nan, dtype=np.float32)
    for row_idx in range(3):
        if not row_valid[row_idx] or not np.all(np.isfinite(values_2d[row_idx])):
            continue
        for dim_idx in range(2):
            expected_2d[row_idx, :, dim_idx] = np.interp(
                target_s,
                source_s,
                values_2d[row_idx, :, dim_idx],
            ).astype(np.float32)

    actual_2d = mod._interp_rows_2d(source_s, values_2d, target_s, row_valid)
    np.testing.assert_allclose(actual_2d, expected_2d, rtol=0.0, atol=1e-7, equal_nan=True)

    values_1d = values_2d[:, :, 1]
    expected_1d = np.full((3, target_s.size), np.nan, dtype=np.float32)
    for row_idx in range(3):
        if row_valid[row_idx] and np.all(np.isfinite(values_1d[row_idx])):
            expected_1d[row_idx] = np.interp(target_s, source_s, values_1d[row_idx]).astype(np.float32)
    actual_1d = mod._interp_rows_1d(source_s, values_1d, target_s, row_valid)
    np.testing.assert_allclose(actual_1d, expected_1d, rtol=0.0, atol=1e-7, equal_nan=True)


def test_vectorized_validity_preserves_failure_precedence() -> None:
    sources = _source_arrays(row_count=6)
    sources["body_frame_valid"][1] = False
    sources["bspline_valid"][[1, 2]] = False
    sources["tail_sample_valid"][[1, 2, 3]] = False
    sources["tail_sample_xy"][4, 1, 0] = np.nan
    sources["body_forward_axis_xy"][5] = 0.0

    batch = mod.compute_tail_kinematics_from_subject_shape_arrays(
        **sources,
        body_frame_failure_reason=np.asarray(["ok", "body_missing", "ok", "ok", "ok", "ok"], dtype=object),
        bspline_failure_reason=np.asarray(
            ["ok", "spline_should_lose", "ok", "ok", "ok", "ok"], dtype=object
        ),
        tail_sample_failure_reason=np.asarray(
            ["ok", "tail_should_lose", "tail_should_lose", "tail_too_short", "ok", "ok"],
            dtype=object,
        ),
        tail_angle_sample_count=10,
    )

    assert batch.valid.tolist() == [True, False, False, False, False, False]
    assert batch.failure_reason.tolist() == [
        "ok",
        "body_missing",
        "bspline_invalid",
        "tail_too_short",
        "tail_geometry_nonfinite",
        "body_frame_invalid",
    ]


def _build_shape_root(*, row_count: int = 2) -> zarr.Group:
    root = zarr.group()
    analysis = root.create_group("analysis")
    parent = analysis.create_group("subject_shape_runs")
    parent.attrs["latest"] = "shape_001"
    shape = parent.create_group("shape_001")
    shape.attrs["source_refined_subject_masks_run"] = "refined_001"
    shape.attrs["body_frame_schema_id"] = "fish_anatomical_body_frame"

    source_revisions = shape.create_group("source_refined_subject_masks")
    source_revisions.attrs["source_run"] = "refined_001"
    source_revisions.attrs["component_names"] = ["subject_body"]
    source_revisions.create_array(
        "row_revision",
        data=np.arange(3, 3 + int(row_count), dtype=np.int64)[:, None],
        overwrite=True,
    )
    source_revisions.create_array("row_revision_available", data=np.asarray([True], dtype=bool), overwrite=True)

    row_index = shape.create_group("row_index")
    row_index.create_array(
        "frame_indices", data=np.arange(10, 10 + int(row_count), dtype=np.int32), overwrite=True
    )
    row_index.create_array(
        "detection_indices", data=np.arange(int(row_count), dtype=np.int32), overwrite=True
    )
    row_index.create_array(
        "source_refined_row_ids", data=np.arange(100, 100 + int(row_count), dtype=np.int64), overwrite=True
    )

    components = shape.create_group("components")
    body = components.create_group("subject_body")
    sources = _source_arrays(row_count=int(row_count))
    body.attrs["tail_sample_count"] = int(sources["source_tail_sample_s"].shape[0])
    body.create_array("tail_sample_s", data=sources["source_tail_sample_s"], overwrite=True)
    body.create_array("tail_sample_xy", data=sources["tail_sample_xy"], overwrite=True)
    body.create_array("tail_tangent_xy", data=sources["tail_tangent_xy"], overwrite=True)
    body.create_array("tail_curvature_px_inv", data=sources["tail_curvature_px_inv"], overwrite=True)
    body.create_array("tail_sample_valid", data=sources["tail_sample_valid"], overwrite=True)
    body.create_array("bspline_valid", data=sources["bspline_valid"], overwrite=True)
    body.create_array("tail_base_xy", data=sources["tail_base_xy"], overwrite=True)
    body.create_array(
        "tail_sample_failure_reason_bytes",
        data=mod._encode_reasons(["ok"] * int(row_count)),
        overwrite=True,
    )
    body.create_array(
        "bspline_failure_reason_bytes",
        data=mod._encode_reasons(["ok"] * int(row_count)),
        overwrite=True,
    )

    body_frame = shape.create_group("body_frame")
    body_frame.create_array("forward_axis_xy", data=sources["body_forward_axis_xy"], overwrite=True)
    body_frame.create_array("left_axis_xy", data=sources["body_left_axis_xy"], overwrite=True)
    body_frame.create_array("valid", data=sources["body_frame_valid"], overwrite=True)
    body_frame.create_array(
        "failure_reason_bytes",
        data=mod._encode_reasons(["ok"] * int(row_count)),
        overwrite=True,
    )
    return root


def test_write_tail_kinematics_run_group_writes_schema_and_row_lineage(monkeypatch) -> None:
    _patch_provenance(monkeypatch)
    root = _build_shape_root()

    summary = mod.write_tail_kinematics_run_group(
        root,
        shape_run="shape_001",
        run_name="tail_001",
        tail_angle_sample_count=10,
    )

    assert summary["status"] == "updated"
    assert summary["valid_row_count"] == 2
    parent = root["analysis"]["tail_kinematics_runs"]
    assert parent.attrs["latest"] == "tail_001"
    run = parent["tail_001"]
    assert run.attrs["schema_id"] == "analysis.tail_kinematics_runs"
    assert run.attrs["schema_version"] == 1
    assert run.attrs["source_subject_shape_run"] == "shape_001"
    assert run.attrs["source_refined_subject_masks_run"] == "refined_001"
    assert run.attrs["source_refined_subject_masks_revision_snapshot"] is True
    assert run.attrs["tail_angle_sample_count"] == 10
    assert run["source_refined_subject_masks"].attrs["copied_from_subject_shape_run"] == "shape_001"
    np.testing.assert_array_equal(
        np.asarray(run["source_refined_subject_masks"]["row_revision"][:], dtype=np.int64),
        np.asarray([[3], [4]], dtype=np.int64),
    )
    assert run["frame_index"][:].tolist() == [10, 11]
    assert run["row_index"]["frame_indices"][:].tolist() == [10, 11]
    assert np.asarray(run["tail_angle_rad"][:], dtype=np.float32).shape == (2, 10)
    np.testing.assert_allclose(np.asarray(run["tail_angle_deg"][:], dtype=np.float32), 0.0, atol=1e-5)
    assert run.attrs["provenance"]["stage"] == "analysis.tail_kinematics_runs"
    assert run.attrs["materialization_mode"] == "bounded_streaming_single_writer"
    assert run.attrs["compute_kernel"] == "vectorized_shared_grid_v1"
    assert run.attrs["provenance"]["parameters"]["compute_kernel"] == "vectorized_shared_grid_v1"
    assert run.attrs["completed_block_count"] == 1
    assert tuple(run["tail_angle_rad"].chunks) == (2, 10)
    assert tuple(run["tail_angle_rad"].shards) == (2, 10)
    assert tuple(run["row_index"]["frame_indices"].shards) == (2,)


def test_write_tail_kinematics_run_group_streams_aligned_blocks_with_whole_batch_parity(monkeypatch) -> None:
    _patch_provenance(monkeypatch)
    monkeypatch.setattr(mod, "refined_subject_mask_metric_row_chunk", lambda _total_rows: 2)
    root = _build_shape_root(row_count=9)
    expected = mod.compute_tail_kinematics_from_subject_shape_arrays(
        **_source_arrays(row_count=9),
        tail_angle_sample_count=10,
    )
    source_slices: list[tuple[int, int]] = []
    original_read = mod._read_tail_kinematics_source_block

    def _recording_read(sources, row_slice):
        source_slices.append((int(row_slice.start), int(row_slice.stop)))
        return original_read(sources, row_slice)

    monkeypatch.setattr(mod, "_read_tail_kinematics_source_block", _recording_read)

    summary = mod.write_tail_kinematics_run_group(
        root,
        shape_run="shape_001",
        run_name="tail_streamed",
        tail_angle_sample_count=10,
        block_rows=3,
    )

    assert source_slices == [(0, 4), (4, 8), (8, 9)]
    assert summary["requested_block_rows"] == 3
    assert summary["effective_block_rows"] == 4
    assert summary["block_count"] == 3
    assert summary["completed_block_count"] == 3
    run = root["analysis"]["tail_kinematics_runs"]["tail_streamed"]
    assert run.attrs["output_row_chunk"] == 2
    assert run.attrs["output_shard_rows"] == 4
    assert tuple(run["tail_angle_rad"].shards) == (4, 10)
    assert run.attrs["palette_run_completion_status"] == "complete"
    np.testing.assert_allclose(
        np.asarray(run["tail_angle_rad"][:], dtype=np.float32),
        expected.tail_angle_rad,
        atol=0.0,
    )
    np.testing.assert_allclose(
        np.asarray(run["tail_lateral_deflection_px"][:], dtype=np.float32),
        expected.tail_lateral_deflection_px,
        atol=0.0,
    )
    np.testing.assert_array_equal(
        np.asarray(run["valid"][:], dtype=bool),
        expected.valid,
    )
    assert run["frame_index"][:].tolist() == list(range(10, 19))
    assert np.asarray(run["source_refined_subject_masks"]["row_revision"][:]).shape == (9, 1)


def test_write_tail_kinematics_run_group_dry_run_does_not_read_frame_blocks(monkeypatch) -> None:
    root = _build_shape_root(row_count=9)

    def _refuse_block_read(*_args, **_kwargs):
        raise AssertionError("dry-run must not read framewise source blocks")

    monkeypatch.setattr(mod, "_read_tail_kinematics_source_block", _refuse_block_read)
    summary = mod.write_tail_kinematics_run_group(
        root,
        shape_run="shape_001",
        run_name="tail_dry_run",
        block_rows=3,
        dry_run=True,
    )

    assert summary["status"] == "planned"
    assert "tail_kinematics_runs" not in root["analysis"]


def test_write_tail_kinematics_run_group_marks_partial_stream_failed(monkeypatch) -> None:
    _patch_provenance(monkeypatch)
    monkeypatch.setattr(mod, "refined_subject_mask_metric_row_chunk", lambda _total_rows: 2)
    root = _build_shape_root(row_count=9)
    original_compute = mod.compute_tail_kinematics_from_subject_shape_arrays
    call_count = 0

    def _fail_second_block(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise RuntimeError("intentional block failure")
        return original_compute(**kwargs)

    monkeypatch.setattr(mod, "compute_tail_kinematics_from_subject_shape_arrays", _fail_second_block)

    with pytest.raises(RuntimeError, match="intentional block failure"):
        mod.write_tail_kinematics_run_group(
            root,
            shape_run="shape_001",
            run_name="tail_failed",
            block_rows=3,
        )

    parent = root["analysis"]["tail_kinematics_runs"]
    run = parent["tail_failed"]
    assert run.attrs["palette_run_completion_status"] == "failed"
    assert run.attrs["completed_block_count"] == 1
    assert parent.attrs.get("latest") != "tail_failed"


def test_write_tail_kinematics_run_group_copies_instance_key_lineage(monkeypatch) -> None:
    _patch_provenance(monkeypatch)
    root = _build_shape_root()
    shape_row_index = root["analysis"]["subject_shape_runs"]["shape_001"]["row_index"]
    shape_row_index.create_array("instance_key", data=np.asarray([11, 22], dtype=np.uint64), overwrite=True)
    shape_row_index.create_array("source_crop_row_ids", data=np.asarray([5, 6], dtype=np.int64), overwrite=True)

    mod.write_tail_kinematics_run_group(
        root,
        shape_run="shape_001",
        run_name="tail_001",
        tail_angle_sample_count=10,
    )

    run = root["analysis"]["tail_kinematics_runs"]["tail_001"]
    assert run["row_index"]["instance_key"][:].tolist() == [11, 22]
    assert run["row_index"]["source_crop_row_ids"][:].tolist() == [5, 6]
    assert "instance_key" in run.attrs["row_lineage_copied"]
    assert "source_crop_row_ids" in run.attrs["row_lineage_copied"]


def test_write_tail_kinematics_run_group_reports_instance_key_missing_for_legacy_sources(monkeypatch) -> None:
    _patch_provenance(monkeypatch)
    root = _build_shape_root()

    mod.write_tail_kinematics_run_group(
        root,
        shape_run="shape_001",
        run_name="tail_001",
        tail_angle_sample_count=10,
    )

    run = root["analysis"]["tail_kinematics_runs"]["tail_001"]
    assert "instance_key" not in run["row_index"]
    assert "instance_key" in run.attrs["row_lineage_missing"]
    assert "source_crop_row_ids" in run.attrs["row_lineage_missing"]
