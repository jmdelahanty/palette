from __future__ import annotations

from pathlib import Path

import numpy as np
import zarr

from fisheye.analysis import subject_shape_runs as mod
from fisheye.analysis_workflows.materializers import subject_shape as materializer
from fisheye.refinement.subject_body_mask_qc import write_subject_body_mask_qc_group
from fisheye.shared.mask_store import write_component_rle_mask_store_from_dense


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
                "hostname": "shape-host",
                "system": "Linux",
                "release": "6.8",
                "python_version": "3.11.0",
                "machine": "x86_64",
            },
        },
    )


def _disk_mask(height: int, width: int, center_y: float, center_x: float, radius: float) -> np.ndarray:
    yy, xx = np.ogrid[:height, :width]
    return ((yy - center_y) ** 2 + (xx - center_x) ** 2 <= radius**2).astype(np.uint8)


def _decode_reason_row(row: np.ndarray) -> str:
    return bytes(np.asarray(row, dtype=np.uint8)).split(b"\0", 1)[0].decode("utf-8")


def _build_refined_root(store_path: Path | None = None) -> zarr.Group:
    root = zarr.open_group(str(store_path), mode="w") if store_path is not None else zarr.group()
    parent = root.create_group("refined_subject_masks_runs")
    parent.attrs["latest"] = "refined_001"
    run = parent.create_group("refined_001")
    labels = ["subject_body", "eye_left", "eye_right", "swim_bladder"]
    run.attrs.update(
        {
            "mask_labels": labels,
            "label_schema_id": "subject_v1_lr",
            "component_metrics_schema_id": "refined_subject_component_mask_metrics_v1",
            "source_subject_mask_run": "subject_001",
            "component_review_statuses": {
                component: {"state": "approved", "method": "unit_test"}
                for component in labels
            },
        }
    )
    run.create_array("available_channels", data=np.asarray([True, True, True, True], dtype=bool), overwrite=True)
    run.create_array("detection_source", data=np.asarray([0, 0, 0], dtype=np.int8), overwrite=True)
    run.create_array("frame_indices", data=np.asarray([10, 11, 12], dtype=np.int32), overwrite=True)
    run.create_array("detection_indices", data=np.asarray([0, 1, 2], dtype=np.int32), overwrite=True)
    run.create_array("source_refined_row_ids", data=np.asarray([100, 101, 102], dtype=np.int64), overwrite=True)

    masks = np.zeros((3, 4, 16, 16), dtype=np.uint8)
    for row_idx in range(3):
        masks[row_idx, 0, 2:14, 4:12] = 1
        masks[row_idx, 1] = _disk_mask(16, 16, 5 + row_idx, 5, 2.5)
        masks[row_idx, 2] = _disk_mask(16, 16, 5 + row_idx, 10, 2.5)
        masks[row_idx, 3] = _disk_mask(16, 16, 10, 8, 2.0)
    run.create_array("masks_roi", data=masks, chunks=(2, 1, 16, 16), overwrite=True)
    return root


def test_snout_bridge_path_can_follow_curved_mask_corridor() -> None:
    mask = np.zeros((32, 32), dtype=bool)
    mask[6:10, 5:22] = True
    mask[6:23, 18:22] = True

    bridge, reason = mod._snout_bridge_path_xy(mask, np.asarray([5.0, 7.0]), np.asarray([20.0, 21.0]))

    assert reason == "ok"
    assert bridge is not None
    np.testing.assert_allclose(bridge[0], np.asarray([5.0, 7.0]), atol=1e-5)
    np.testing.assert_allclose(bridge[-1], np.asarray([20.0, 21.0]), atol=1e-5)
    assert bridge.shape[0] > 2


def test_snout_join_prefers_medial_head_region_over_branch_endpoint() -> None:
    path = np.asarray(
        [
            [269.0, 225.0],
            [269.0, 229.0],
            [269.0, 233.0],
            [269.0, 237.0],
            [270.0, 246.0],
            [268.0, 250.0],
            [260.0, 260.0],
        ],
        dtype=np.float64,
    )
    origin = np.asarray([271.0, 247.0], dtype=np.float64)
    left = np.asarray([-0.624, -0.782], dtype=np.float64)

    selected = mod._snout_join_index(path, origin, left)

    assert selected > 0
    assert abs(float(np.dot(path[selected] - origin, left))) < abs(float(np.dot(path[0] - origin, left)))


def _add_component_row_revisions(root: zarr.Group, revisions: dict[str, list[int]]) -> None:
    run = root["refined_subject_masks_runs"]["refined_001"]
    components = run.require_group("components")
    for component_name, values in revisions.items():
        component = components.require_group(component_name)
        component.create_array("row_revision", data=np.asarray(values, dtype=np.int64), overwrite=True)


def test_write_subject_shape_run_group_creates_coherent_components_and_relations(monkeypatch) -> None:
    _patch_provenance(monkeypatch)
    root = _build_refined_root()

    summary = mod.write_subject_shape_run_group(
        root,
        refined_run="refined_001",
        run_name="shape_001",
        chunk_size=2,
        include_chunk_timings=True,
    )

    assert summary["status"] == "updated"
    run = root["analysis"]["subject_shape_runs"]["shape_001"]
    assert root["analysis"]["subject_shape_runs"].attrs["latest"] == "shape_001"
    assert run.attrs["schema_id"] == "analysis.subject_shape_runs"
    assert run.attrs["schema_version"] == 3
    assert run.attrs["method"] == "subject_shape_from_refined_masks_v10"
    assert run.attrs["method_version"] == 10
    assert run.attrs["snout_tip_estimator"] == "subject_body_contour_max_forward_projection_v1"
    assert run.attrs["centerline_method"] == "snout_anchored_skeleton_longest_endpoint_path_v1"
    assert run.attrs["centerline_skeleton_method"] == "skeleton_longest_endpoint_path_v1"
    assert run.attrs["centerline_snout_extension_method"] == "prepend_mask_path_to_body_frame_guided_join_v1"
    assert run.attrs["centerline_snout_join_method"] == "body_frame_lateral_min_head_region_v1"
    assert run.attrs["head_endpoint_semantics"] == "validated_snout_tip"
    assert run.attrs["centerline_snout_check_method"] == "head_endpoint_to_snout_distance_v1"
    assert run.attrs["source_refined_subject_masks_run"] == "refined_001"
    assert run.attrs["component_names"] == ["subject_body", "swim_bladder", "eye_left", "eye_right"]
    assert run["row_index"]["frame_indices"][:].tolist() == [10, 11, 12]
    assert run["row_index"]["source_refined_row_ids"][:].tolist() == [100, 101, 102]
    source_revisions = run["source_refined_subject_masks"]
    assert source_revisions.attrs["schema_id"] == "analysis.subject_shape.source_refined_subject_masks_v1"
    assert source_revisions.attrs["source_run"] == "refined_001"
    assert source_revisions.attrs["component_names"] == ["subject_body", "swim_bladder", "eye_left", "eye_right"]
    np.testing.assert_array_equal(
        np.asarray(source_revisions["row_revision"][:], dtype=np.int64),
        np.zeros((3, 4), dtype=np.int64),
    )
    np.testing.assert_array_equal(
        np.asarray(source_revisions["row_revision_available"][:], dtype=bool),
        np.asarray([False, False, False, False], dtype=bool),
    )

    body = run["components"]["subject_body"]
    assert body["mask_present"][:].tolist() == [True, True, True]
    assert np.all(np.asarray(body["principal_axis_valid"][:], dtype=bool))
    assert np.all(np.asarray(body["principal_axis_length_px"][:], dtype=np.float32) > 0)

    eye_pair = run["relations"]["eye_pair"]
    assert eye_pair["separation_valid"][:].tolist() == [True, True, True]
    assert np.all(np.asarray(eye_pair["separation_px"][:], dtype=np.float32) > 0)
    swim_to_body = run["relations"]["swim_bladder_to_body"]
    assert swim_to_body["relation_valid"][:].tolist() == [True, True, True]
    eyes_to_body = run["relations"]["eyes_to_body"]
    assert eyes_to_body["left_eye_relation_valid"][:].tolist() == [True, True, True]

    body_frame = run["body_frame"]
    assert body_frame.attrs["body_frame_estimator"] == "mask_component_axis"
    assert body_frame["valid"][:].tolist() == [True, True, True]
    forward = np.asarray(body_frame["forward_axis_xy"][:], dtype=np.float32)
    assert np.all(np.isfinite(forward))
    assert np.all(forward[:, 1] < 0.0)

    swim = run["components"]["swim_bladder"]
    assert swim["caudal_contour_valid"][:].tolist() == [True, True, True]
    caudal = np.asarray(swim["caudal_contour_point_xy"][:], dtype=np.float32)
    assert np.all(np.isfinite(caudal))
    assert np.all(caudal[:, 1] > 10.0)

    assert body.attrs["centerline_method"] == "snout_anchored_skeleton_longest_endpoint_path_v1"
    assert body.attrs["centerline_skeleton_method"] == "skeleton_longest_endpoint_path_v1"
    assert body.attrs["centerline_snout_extension_method"] == "prepend_mask_path_to_body_frame_guided_join_v1"
    assert body.attrs["centerline_snout_join_method"] == "body_frame_lateral_min_head_region_v1"
    assert body.attrs["head_endpoint_semantics"] == "validated_snout_tip"
    assert body.attrs["snout_tip_estimator"] == "subject_body_contour_max_forward_projection_v1"
    assert body.attrs["centerline_snout_check_method"] == "head_endpoint_to_snout_distance_v1"
    assert body.attrs["bspline_method"] == "centerline_scipy_splprep_v1"
    assert body.attrs["bspline_fit_mode"] == "interpolating"
    assert body["snout_tip_valid"][:].tolist() == [True, True, True]
    snout = np.asarray(body["snout_tip_xy"][:], dtype=np.float32)
    assert np.all(np.isfinite(snout))
    assert body["centerline_valid"][:].tolist() == [True, True, True]
    centerline = np.asarray(body["centerline_xy"][:], dtype=np.float32)
    assert centerline.shape == (3, mod.CENTERLINE_SAMPLE_COUNT, 2)
    head = np.asarray(body["head_endpoint_xy"][:], dtype=np.float32)
    tail = np.asarray(body["tail_tip_xy"][:], dtype=np.float32)
    np.testing.assert_allclose(head, snout, atol=1e-5)
    np.testing.assert_allclose(
        np.asarray(body["head_endpoint_to_snout_distance_px"][:], dtype=np.float32),
        np.zeros((3,), dtype=np.float32),
        atol=1e-5,
    )
    assert body["centerline_reaches_snout"][:].tolist() == [True, True, True]
    assert _decode_reason_row(body["centerline_snout_check_reason_bytes"][0]) == "ok"
    assert np.all(head[:, 1] < tail[:, 1])
    assert body["tail_base_valid"][:].tolist() == [True, True, True]
    assert np.all(np.asarray(body["body_arclength_px"][:], dtype=np.float32) > 0)
    assert np.all(np.asarray(body["tail_segment_arclength_px"][:], dtype=np.float32) >= 0)
    assert body["bspline_valid"][:].tolist() == [True, True, True]
    assert body["tail_sample_valid"][:].tolist() == [False, False, False]
    assert _decode_reason_row(body["tail_sample_failure_reason_bytes"][0]) == "tail_segment_too_short"
    assert np.asarray(body["bspline_sample_xy"][:], dtype=np.float32).shape == (3, mod.CENTERLINE_SAMPLE_COUNT, 2)
    assert np.asarray(body["tail_sample_xy"][:], dtype=np.float32).shape == (3, mod.TAIL_SAMPLE_COUNT, 2)
    np.testing.assert_allclose(
        np.asarray(body["tail_sample_s"][:], dtype=np.float32),
        np.linspace(0.0, 1.0, mod.TAIL_SAMPLE_COUNT, dtype=np.float32),
    )
    assert np.all(np.asarray(body["bspline_arc_length_px"][:], dtype=np.float32) > 0)
    assert run.attrs["provenance"]["stage"] == "analysis.subject_shape_runs"
    assert run.attrs["subject_shape_chunk_timing_storage"] == "embedded_full_records"
    assert len(run.attrs["subject_shape_chunk_timings"]) == 2


def test_write_subject_shape_run_group_can_read_source_and_write_separate_root(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _patch_provenance(monkeypatch)
    source_path = tmp_path / "source.zarr"
    output_path = tmp_path / "output.zarr"
    source = _build_refined_root(source_path)
    output = zarr.open_group(str(output_path), mode="w", zarr_format=3)

    summary = mod.write_subject_shape_run_group(
        source,
        zarr_path=source_path,
        output_root=output,
        output_zarr_path=output_path,
        refined_run="refined_001",
        run_name="shape_separate",
        chunk_size=2,
        execution_backend="dask_worker_chunks",
        scheduler="single-threaded",
        centerline_crop_to_foreground=True,
        native_threads=1,
    )

    assert summary["status"] == "updated"
    assert "analysis" not in source
    run = output["analysis"]["subject_shape_runs"]["shape_separate"]
    assert run.attrs["palette_run_completion_status"] == "complete"
    assert run.attrs["centerline_crop_to_foreground"] is True
    assert run.attrs["native_threads_per_worker"] == 1
    assert run["row_index"]["frame_indices"][:].tolist() == [10, 11, 12]


def test_subject_shape_materializer_computes_shards_and_publishes(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _patch_provenance(monkeypatch)
    monkeypatch.setattr(
        materializer,
        "write_best_effort_run_lineage_attrs",
        lambda *args, **kwargs: None,
    )
    source_path = tmp_path / "source.zarr"
    source = _build_refined_root(source_path)
    masks_before = np.asarray(
        source["refined_subject_masks_runs"]["refined_001"]["masks_roi"][:]
    ).copy()
    scratch = tmp_path / "scratch"

    planned = materializer.materialize_subject_shape(
        source_path,
        scratch_root=scratch,
        refined_run="refined_001",
        run_name="shape_materialized",
        block_rows=2,
        output_shard_rows=4,
        execution_backend="serial_driver",
        scheduler="single-threaded",
        num_workers=1,
        shard_copy_workers=1,
        native_threads=1,
        copy_backend="python",
        apply=False,
    )
    assert planned["status"] == "planned"
    assert planned["plan"]["source_access_policy"] == "authoritative_shared_read_only"
    assert not scratch.exists()

    result = materializer.materialize_subject_shape(
        source_path,
        scratch_root=scratch,
        refined_run="refined_001",
        run_name="shape_materialized",
        block_rows=2,
        output_shard_rows=4,
        execution_backend="serial_driver",
        scheduler="single-threaded",
        num_workers=1,
        shard_copy_workers=1,
        native_threads=1,
        copy_backend="python",
        apply=True,
        keep_scratch=True,
        check_capacity=False,
        stage_command="unit-test-subject-shape-materializer",
    )

    assert result["status"] == "complete"
    assert result["local_materialization"]["node_local_sharding"]["exact_decoded_validation"] is True
    root = zarr.open_group(str(source_path), mode="r", use_consolidated=False)
    np.testing.assert_array_equal(
        np.asarray(root["refined_subject_masks_runs"]["refined_001"]["masks_roi"][:]),
        masks_before,
    )
    parent = root["analysis"]["subject_shape_runs"]
    assert parent.attrs["latest"] == "shape_materialized"
    run = parent["shape_materialized"]
    assert run.attrs["palette_run_completion_status"] == "complete"
    assert run.attrs["centerline_crop_to_foreground"] is True
    assert run.attrs["physical_storage_layout"]["layout"] == "zarr_v3_indexed_sharding"
    assert tuple(run["components"]["subject_body"]["centerline_xy"].shards) == (6, 64, 2)
    assert run.attrs["cluster_output_staging"]["schema_id"] == materializer.PUBLISH_SCHEMA_ID
    assert scratch.is_dir()


def test_write_subject_shape_run_group_reads_compact_mask_store_without_dense_masks(monkeypatch) -> None:
    _patch_provenance(monkeypatch)
    root = _build_refined_root()
    refined = root["refined_subject_masks_runs"]["refined_001"]
    dense = np.asarray(refined["masks_roi"][:], dtype=np.uint8)
    write_component_rle_mask_store_from_dense(
        refined,
        refined["masks_roi"],
        component_names=tuple(str(value) for value in refined.attrs["mask_labels"]),
        encode_row_chunk_size=2,
    )
    del refined["masks_roi"]

    summary = mod.write_subject_shape_run_group(
        root,
        refined_run="refined_001",
        run_name="shape_rle_only",
        chunk_size=2,
    )

    assert summary["status"] == "updated"
    run = root["analysis"]["subject_shape_runs"]["shape_rle_only"]
    assert run.attrs["source_mask_store_encoding"] == "component_rle_v1"
    assert run.attrs["source_mask_storage_surface"] == "mask_rle"
    assert run.attrs["source_mask_store_path"] == "refined_subject_masks_runs/refined_001/mask_rle"
    assert run.attrs["source_refs"]["refined_subject_masks_mask_store"] == (
        "refined_subject_masks_runs/refined_001/mask_rle"
    )
    assert "refined_subject_masks_masks_roi" not in run.attrs["source_refs"]
    assert run.attrs["source_refs"]["refined_subject_masks_mask_rle"] == (
        "refined_subject_masks_runs/refined_001/mask_rle"
    )
    assert run["components"]["subject_body"]["mask_present"][:].tolist() == [True, True, True]
    assert run["components"]["eye_left"]["area_px"][:].tolist() == dense[:, 1].sum(axis=(1, 2)).tolist()
    assert run["body_frame"]["valid"][:].tolist() == [True, True, True]


def test_write_subject_shape_run_uses_dask_worker_chunks(tmp_path: Path, monkeypatch) -> None:
    _patch_provenance(monkeypatch)
    zarr_path = tmp_path / "shape.zarr"
    _build_refined_root(zarr_path)

    summary = mod.write_subject_shape_run(
        zarr_path,
        refined_run="refined_001",
        run_name="shape_dask",
        chunk_size=1,
        execution_backend="dask_worker_chunks",
        scheduler="single-threaded",
    )

    assert summary["status"] == "updated"
    assert summary["execution_backend"] == "dask_worker_chunks"
    assert summary["chunk_size"] == 1
    assert summary["worker_chunk_size"] == 3
    assert summary["dask_requested_chunk_size"] == 1
    assert summary["dask_chunk_size"] == 3
    assert summary["dask_chunk_alignment"] == "refined_subject_mask_metric_row_chunk"
    assert summary["chunk_count"] == 1
    root = zarr.open_group(str(zarr_path), mode="r")
    run = root["analysis"]["subject_shape_runs"]["shape_dask"]
    assert run.attrs["dask_execution_enabled"] is True
    assert run.attrs["dask_scheduler"] == "single-threaded"
    assert run.attrs["chunk_size"] == 1
    assert run.attrs["worker_chunk_size"] == 3
    assert run.attrs["dask_requested_chunk_size"] == 1
    assert run.attrs["dask_chunk_size"] == 3
    assert run.attrs["dask_chunk_alignment"] == "refined_subject_mask_metric_row_chunk"
    assert run.attrs["provenance"]["parameters"]["worker_chunk_size"] == 3
    assert run.attrs["provenance"]["scheduler"]["dask_requested_chunk_size"] == 1
    assert run.attrs["provenance"]["scheduler"]["dask_chunk_size"] == 3
    assert run.attrs["subject_shape_chunk_timing_count"] == 1
    assert run.attrs["subject_shape_chunk_timing_storage"] == "summary_only"
    assert "subject_shape_chunk_timings" not in run.attrs
    assert run.attrs["subject_shape_timing_summary"]["chunk_timings"]["chunk_count"] == 1
    assert run["components"]["eye_left"]["ellipse_success"][:].tolist() == [True, True, True]
    assert run["relations"]["eye_pair"]["separation_valid"][:].tolist() == [True, True, True]
    assert run["body_frame"]["valid"][:].tolist() == [True, True, True]
    assert run["components"]["subject_body"]["centerline_valid"][:].tolist() == [True, True, True]


def test_subject_shape_dask_worker_chunk_size_rounds_to_metric_grid() -> None:
    assert mod._worker_chunk_size_for_backend(1000, 1, "dask_worker_chunks") == 256
    assert mod._worker_chunk_size_for_backend(1000, 256, "dask_worker_chunks") == 256
    assert mod._worker_chunk_size_for_backend(1000, 257, "dask_worker_chunks") == 512
    assert mod._worker_chunk_size_for_backend(1000, 500, "dask_worker_chunks") == 512
    assert mod._worker_chunk_size_for_backend(1000, 500, "serial_driver") == 500


def test_write_subject_shape_run_dry_run_does_not_create_analysis_group() -> None:
    root = _build_refined_root()

    summary = mod.write_subject_shape_run_group(
        root,
        refined_run="refined_001",
        run_name="shape_dry",
        dry_run=True,
    )

    assert summary["status"] == "planned"
    assert "analysis" not in root


def test_subject_shape_tail_geometry_fails_closed_for_fragmented_body(monkeypatch) -> None:
    _patch_provenance(monkeypatch)
    root = _build_refined_root()
    refined = root["refined_subject_masks_runs"]["refined_001"]
    masks = np.asarray(refined["masks_roi"][:], dtype=np.uint8)
    masks[1, 0] = 0
    masks[1, 0, 2:5, 4:7] = 1
    masks[1, 0, 10:13, 9:12] = 1
    refined["masks_roi"][:] = masks

    mod.write_subject_shape_run_group(
        root,
        refined_run="refined_001",
        run_name="shape_fragmented",
        chunk_size=2,
    )

    body = root["analysis"]["subject_shape_runs"]["shape_fragmented"]["components"]["subject_body"]
    assert body["snout_tip_valid"][:].tolist() == [True, False, True]
    assert _decode_reason_row(body["snout_tip_failure_reason_bytes"][1]) == "fragmented_subject_body_mask"
    assert body["centerline_reaches_snout"][:].tolist() == [True, False, True]
    assert _decode_reason_row(body["centerline_snout_check_reason_bytes"][1]) == "missing_snout_tip"
    assert body["centerline_valid"][:].tolist() == [True, False, True]
    assert _decode_reason_row(body["centerline_failure_reason_bytes"][1]) == "fragmented_subject_body_mask"
    assert body["bspline_valid"][:].tolist() == [True, False, True]
    assert _decode_reason_row(body["bspline_failure_reason_bytes"][1]) == "fragmented_subject_body_mask"


def test_subject_shape_tail_geometry_fails_closed_for_source_body_qc(monkeypatch) -> None:
    _patch_provenance(monkeypatch)
    root = _build_refined_root()
    refined = root["refined_subject_masks_runs"]["refined_001"]
    masks = np.asarray(refined["masks_roi"][:], dtype=np.uint8)
    masks[1, 0, 7:9, 0:16] = 1
    masks[1, 0, 10:12, 0:16] = 1
    refined["masks_roi"][:] = masks
    write_subject_body_mask_qc_group(root, refined_run="refined_001", chunk_size=2)
    assert bool(np.asarray(refined["components/subject_body/qc/severe_qc_failure"][1], dtype=bool)) is True

    mod.write_subject_shape_run_group(
        root,
        refined_run="refined_001",
        run_name="shape_source_body_qc",
        chunk_size=2,
    )

    body = root["analysis"]["subject_shape_runs"]["shape_source_body_qc"]["components"]["subject_body"]
    assert body["source_mask_qc_available"][:].tolist() == [True, True, True]
    assert bool(np.asarray(body["source_mask_qc_severe_failure"][1], dtype=bool)) is True
    assert body["snout_tip_valid"][:].tolist() == [True, False, True]
    assert _decode_reason_row(body["snout_tip_failure_reason_bytes"][1]) == "source_body_mask_qc_failed"
    assert body["centerline_reaches_snout"][:].tolist() == [True, False, True]
    assert _decode_reason_row(body["centerline_snout_check_reason_bytes"][1]) == "missing_snout_tip"
    assert body["centerline_valid"][:].tolist() == [True, False, True]
    assert _decode_reason_row(body["centerline_failure_reason_bytes"][1]) == "source_body_mask_qc_failed"
    assert _decode_reason_row(body["tail_base_failure_reason_bytes"][1]) == "source_body_mask_qc_failed"
    assert body["bspline_valid"][:].tolist() == [True, False, True]
    assert _decode_reason_row(body["bspline_failure_reason_bytes"][1]) == "source_body_mask_qc_failed"
    assert _decode_reason_row(body["tail_sample_failure_reason_bytes"][1]) == "source_body_mask_qc_failed"


def test_subject_shape_run_records_and_audits_source_row_revisions(monkeypatch) -> None:
    _patch_provenance(monkeypatch)
    root = _build_refined_root()
    _add_component_row_revisions(
        root,
        {
            "subject_body": [0, 2, 0],
            "swim_bladder": [0, 0, 1],
        },
    )

    mod.write_subject_shape_run_group(
        root,
        refined_run="refined_001",
        run_name="shape_revisions",
        chunk_size=2,
    )

    run = root["analysis"]["subject_shape_runs"]["shape_revisions"]
    source_revisions = run["source_refined_subject_masks"]
    np.testing.assert_array_equal(
        np.asarray(source_revisions["row_revision"][:], dtype=np.int64),
        np.asarray(
            [
                [0, 0, 0, 0],
                [2, 0, 0, 0],
                [0, 1, 0, 0],
            ],
            dtype=np.int64,
        ),
    )
    np.testing.assert_array_equal(
        np.asarray(source_revisions["row_revision_available"][:], dtype=bool),
        np.asarray([True, True, False, False], dtype=bool),
    )
    assert mod.audit_subject_shape_source_revisions_group(root, shape_run="shape_revisions")["status"] == "current"

    refined = root["refined_subject_masks_runs"]["refined_001"]
    refined["components/subject_body/row_revision"][1] = np.int64(3)
    audit = mod.audit_subject_shape_source_revisions_group(root, shape_run="shape_revisions")

    assert audit["status"] == "stale"
    assert audit["stale_row_count"] == 1
    assert audit["stale_rows"] == [1]
    assert audit["stale_rows_by_component"] == {"subject_body": [1]}


def test_write_subject_shape_run_group_copies_instance_key_lineage(monkeypatch) -> None:
    _patch_provenance(monkeypatch)
    root = _build_refined_root()
    refined = root["refined_subject_masks_runs"]["refined_001"]
    refined.create_array("instance_key", data=np.asarray([11, 22, 33], dtype=np.uint64), overwrite=True)
    refined.create_array("source_crop_row_ids", data=np.asarray([5, 6, 7], dtype=np.int64), overwrite=True)

    mod.write_subject_shape_run_group(
        root,
        refined_run="refined_001",
        run_name="shape_001",
        chunk_size=2,
    )

    run = root["analysis"]["subject_shape_runs"]["shape_001"]
    assert run["row_index"]["instance_key"][:].tolist() == [11, 22, 33]
    assert run["row_index"]["source_crop_row_ids"][:].tolist() == [5, 6, 7]
    assert "instance_key" in run.attrs["row_lineage_copied"]
    assert "source_crop_row_ids" in run.attrs["row_lineage_copied"]


def test_write_subject_shape_run_group_reports_instance_key_missing_for_legacy_sources(monkeypatch) -> None:
    _patch_provenance(monkeypatch)
    root = _build_refined_root()

    mod.write_subject_shape_run_group(
        root,
        refined_run="refined_001",
        run_name="shape_001",
        chunk_size=2,
    )

    run = root["analysis"]["subject_shape_runs"]["shape_001"]
    assert "instance_key" not in run["row_index"]
    assert "instance_key" in run.attrs["row_lineage_missing"]
    assert "source_crop_row_ids" in run.attrs["row_lineage_missing"]
