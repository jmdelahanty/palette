from __future__ import annotations

from pathlib import Path
import shutil
from types import SimpleNamespace

import numpy as np
import pytest
import zarr

from fisheye.analysis import subject_shape_runs as mod
from fisheye.analysis.subject_shape_storage import (
    SUBJECT_SHAPE_ACCESS_AWARE_CANDIDATE_PROFILE_ID,
    materialize_subject_shape_storage_candidate,
    validate_subject_shape_candidate_storage,
    validate_subject_shape_direct_consolidated_storage,
)
from fisheye.analysis_workflows.materializers import subject_shape as materializer
from fisheye.analysis_workflows.materializers.atomic_run_publisher import (
    ATOMIC_PUBLICATION_TOMBSTONE_ATTR,
)
from fisheye.refinement.subject_body_mask_qc import write_subject_body_mask_qc_group
from fisheye.shared.mask_store import write_component_rle_mask_store_from_dense
from fisheye.shared.proof_verification import proof_verification_scope
from fisheye.shared.refined_subject_mask_mutation import (
    RefinedSubjectMaskMutationError,
)
from fisheye.shared.subject_shape_coordinate_publication import (
    SUBJECT_SHAPE_BOUND_CANONICAL_STATUS,
    SUBJECT_SHAPE_CONSUMED_UNBOUND_STAGE_ATTR,
    SUBJECT_SHAPE_DERIVATION_ATTR,
    SUBJECT_SHAPE_PARENT_PUBLICATION_LEASE_ATTR,
    SUBJECT_SHAPE_PUBLICATION_GENERATION_ATTR,
    SUBJECT_SHAPE_PUBLICATION_OWNER_ATTR,
    SUBJECT_SHAPE_PUBLICATION_POLICY,
    SUBJECT_SHAPE_PUBLICATION_POLICY_ATTR,
    SUBJECT_SHAPE_UNBOUND_MANIFEST_ATTR,
    SubjectShapeCoordinatePublicationError,
    load_persisted_subject_shape_coordinate_publication,
)
from tests.unit.fisheye.subject_shape_test_fixtures import (
    resolve_canonical_refined_archive_template,
)


@pytest.fixture(scope="session")
def canonical_refined_template() -> Path:
    """Reuse a validated immutable source graph; tests clone before mutation."""

    return resolve_canonical_refined_archive_template()


@pytest.fixture(scope="session")
def canonical_subject_shape_template(
    tmp_path_factory: pytest.TempPathFactory,
    canonical_refined_template: Path,
) -> tuple[Path, dict[str, object]]:
    """Build one fresh canonical publication for isolated downstream clones."""

    target = tmp_path_factory.mktemp("canonical-subject-shape") / "canonical.zarr"
    shutil.copytree(canonical_refined_template, target)
    root = zarr.open_group(str(target), mode="a", use_consolidated=False)
    with pytest.MonkeyPatch.context() as monkeypatch:
        _patch_provenance(monkeypatch)
        summary = mod.write_subject_shape_run_group(
            root,
            refined_run="r1",
            run_name="shape_001",
            chunk_size=2,
            include_chunk_timings=True,
        )
    return target, summary


@pytest.fixture
def canonical_subject_shape_root(
    tmp_path: Path,
    canonical_subject_shape_template: tuple[Path, dict[str, object]],
) -> tuple[zarr.Group, dict[str, object]]:
    """Clone the immutable publication so every test owns its mutations."""

    template, summary = canonical_subject_shape_template
    target = tmp_path / "canonical.zarr"
    shutil.copytree(template, target)
    root = zarr.open_group(str(target), mode="a", use_consolidated=False)
    return root, summary


@pytest.fixture
def canonical_refined_root(
    tmp_path: Path,
    canonical_refined_template: Path,
) -> zarr.Group:
    target = tmp_path / "canonical.zarr"
    shutil.copytree(canonical_refined_template, target)
    return zarr.open_group(str(target), mode="a", use_consolidated=False)


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


def test_write_subject_shape_run_group_creates_coherent_components_and_relations(
    canonical_subject_shape_root: tuple[zarr.Group, dict[str, object]],
) -> None:
    root, summary = canonical_subject_shape_root

    assert summary["status"] == "updated"
    run = root["analysis"]["subject_shape_runs"]["shape_001"]
    assert root["analysis"]["subject_shape_runs"].attrs["latest"] == "shape_001"
    assert run.attrs["schema_id"] == "analysis.subject_shape_runs"
    assert run.attrs["schema_version"] == 4
    assert run.attrs["method"] == "subject_shape_from_refined_masks_v11"
    assert run.attrs["method_version"] == 11
    assert run.attrs["snout_tip_estimator"] == "subject_body_contour_max_forward_projection_v1"
    assert run.attrs["centerline_method"] == "snout_anchored_skeleton_longest_endpoint_path_v1"
    assert run.attrs["centerline_skeleton_method"] == "skeleton_longest_endpoint_path_v1"
    assert run.attrs["centerline_snout_extension_method"] == "prepend_mask_path_to_body_frame_guided_join_v1"
    assert run.attrs["centerline_snout_join_method"] == "body_frame_lateral_min_head_region_v1"
    assert run.attrs["head_endpoint_semantics"] == "validated_snout_tip"
    assert run.attrs["centerline_snout_check_method"] == "head_endpoint_to_snout_distance_v1"
    assert run.attrs["source_refined_subject_masks_run"] == "r1"
    assert run.attrs["component_names"] == ["subject_body", "swim_bladder", "eye_left", "eye_right"]
    assert run.attrs["relation_names"] == [
        "eye_pair",
        "swim_bladder_to_body",
        "eyes_to_body",
    ]
    assert set(run["row_index"].array_keys()) == {
        "source_crop_row_ids",
        "instance_key",
    }
    assert run.attrs["row_lineage_copied"] == [
        "source_crop_row_ids",
        "instance_key",
    ]
    assert run.attrs["row_lineage_missing"] == [
        "frame_indices",
        "detection_indices",
        "source_refined_row_ids",
        "source_detect_row_index",
    ]
    np.testing.assert_array_equal(run["instance_key"][:], root["refined_subject_masks_runs/r1/instance_key"][:])
    np.testing.assert_array_equal(
        run["source_acquisition_frame_index"][:],
        root["refined_subject_masks_runs/r1/source_acquisition_frame_index"][:],
    )
    source_revisions = run["source_refined_subject_masks"]
    assert source_revisions.attrs["schema_id"] == "analysis.subject_shape.source_refined_subject_masks_v1"
    assert source_revisions.attrs["source_run"] == "r1"
    assert source_revisions.attrs["component_names"] == ["subject_body", "swim_bladder", "eye_left", "eye_right"]
    np.testing.assert_array_equal(
        np.asarray(source_revisions["row_revision"][:], dtype=np.int64),
        np.zeros((2, 4), dtype=np.int64),
    )
    np.testing.assert_array_equal(
        np.asarray(source_revisions["row_revision_available"][:], dtype=bool),
        np.asarray([False, False, False, False], dtype=bool),
    )

    body = run["components"]["subject_body"]
    assert body["mask_present"][:].tolist() == [True, True]
    assert np.all(np.asarray(body["principal_axis_valid"][:], dtype=bool))
    assert np.all(np.asarray(body["principal_axis_length_px"][:], dtype=np.float32) > 0)

    eye_pair = run["relations"]["eye_pair"]
    assert eye_pair["separation_valid"][:].tolist() == [True, False]
    assert float(eye_pair["separation_px"][0]) > 0
    assert np.isnan(float(eye_pair["separation_px"][1]))
    swim_to_body = run["relations"]["swim_bladder_to_body"]
    assert swim_to_body["relation_valid"][:].tolist() == [True, True]
    eyes_to_body = run["relations"]["eyes_to_body"]
    assert eyes_to_body["left_eye_relation_valid"][:].tolist() == [True, True]

    body_frame = run["body_frame"]
    assert body_frame.attrs["body_frame_estimator"] == "mask_component_axis"
    assert body_frame["valid"][:].tolist() == [True, False]
    forward = np.asarray(body_frame["forward_axis_xy"][:], dtype=np.float32)
    assert np.all(np.isfinite(forward[0]))
    assert np.all(np.isnan(forward[1]))
    assert forward[0, 1] < 0.0

    swim = run["components"]["swim_bladder"]
    assert swim["caudal_contour_valid"][:].tolist() == [True, False]
    caudal = np.asarray(swim["caudal_contour_point_xy"][:], dtype=np.float32)
    assert np.all(np.isfinite(caudal[0]))
    assert np.all(np.isnan(caudal[1]))
    assert float(caudal[0, 1]) > 10.0

    assert body.attrs["centerline_method"] == "snout_anchored_skeleton_longest_endpoint_path_v1"
    assert body.attrs["centerline_skeleton_method"] == "skeleton_longest_endpoint_path_v1"
    assert body.attrs["centerline_snout_extension_method"] == "prepend_mask_path_to_body_frame_guided_join_v1"
    assert body.attrs["centerline_snout_join_method"] == "body_frame_lateral_min_head_region_v1"
    assert body.attrs["head_endpoint_semantics"] == "validated_snout_tip"
    assert body.attrs["snout_tip_estimator"] == "subject_body_contour_max_forward_projection_v1"
    assert body.attrs["centerline_snout_check_method"] == "head_endpoint_to_snout_distance_v1"
    assert body.attrs["bspline_method"] == "centerline_scipy_splprep_v1"
    assert body.attrs["bspline_fit_mode"] == "interpolating"
    assert body["snout_tip_valid"][:].tolist() == [True, False]
    snout = np.asarray(body["snout_tip_xy"][:], dtype=np.float32)
    assert np.all(np.isfinite(snout[0]))
    assert np.all(np.isnan(snout[1]))
    assert body["centerline_valid"][:].tolist() == [True, False]
    centerline = np.asarray(body["centerline_xy"][:], dtype=np.float32)
    assert centerline.shape == (2, mod.CENTERLINE_SAMPLE_COUNT, 2)
    head = np.asarray(body["head_endpoint_xy"][:], dtype=np.float32)
    tail = np.asarray(body["tail_tip_xy"][:], dtype=np.float32)
    np.testing.assert_allclose(head, snout, atol=1e-5, equal_nan=True)
    np.testing.assert_allclose(
        np.asarray(body["head_endpoint_to_snout_distance_px"][:], dtype=np.float32),
        np.asarray([0.0, np.nan], dtype=np.float32),
        atol=1e-5,
        equal_nan=True,
    )
    assert body["centerline_reaches_snout"][:].tolist() == [True, False]
    assert _decode_reason_row(body["centerline_snout_check_reason_bytes"][0]) == "ok"
    assert _decode_reason_row(body["centerline_snout_check_reason_bytes"][1]) == (
        "missing_snout_tip"
    )
    assert float(head[0, 1]) < float(tail[0, 1])
    assert np.all(np.isnan(head[1]))
    assert np.all(np.isnan(tail[1]))
    assert body["tail_base_valid"][:].tolist() == [True, False]
    body_arclength = np.asarray(body["body_arclength_px"][:], dtype=np.float32)
    tail_arclength = np.asarray(
        body["tail_segment_arclength_px"][:], dtype=np.float32
    )
    assert float(body_arclength[0]) > 0
    assert np.isnan(float(body_arclength[1]))
    assert float(tail_arclength[0]) >= 0
    assert np.isnan(float(tail_arclength[1]))
    assert body["bspline_valid"][:].tolist() == [True, False]
    assert np.asarray(body["bspline_sample_xy"][:], dtype=np.float32).shape == (2, mod.CENTERLINE_SAMPLE_COUNT, 2)
    assert np.asarray(body["tail_sample_xy"][:], dtype=np.float32).shape == (2, mod.TAIL_SAMPLE_COUNT, 2)
    np.testing.assert_allclose(
        np.asarray(body["tail_sample_s"][:], dtype=np.float32),
        np.linspace(0.0, 1.0, mod.TAIL_SAMPLE_COUNT, dtype=np.float32),
    )
    bspline_arclength = np.asarray(
        body["bspline_arc_length_px"][:], dtype=np.float32
    )
    assert float(bspline_arclength[0]) > 0
    assert np.isnan(float(bspline_arclength[1]))
    assert run.attrs["provenance"]["stage"] == "analysis.subject_shape_runs"
    assert run.attrs["subject_shape_chunk_timing_storage"] == "embedded_full_records"
    assert len(run.attrs["subject_shape_chunk_timings"]) == 1


def test_direct_canonical_writer_rejects_separate_root(
    monkeypatch,
    tmp_path: Path,
    canonical_refined_root: zarr.Group,
) -> None:
    _patch_provenance(monkeypatch)
    source_path = tmp_path / "canonical.zarr"
    output_path = tmp_path / "output.zarr"
    source = canonical_refined_root
    output = zarr.open_group(str(output_path), mode="w", zarr_format=3)

    with pytest.raises(ValueError, match="same archive"):
        mod.write_subject_shape_run_group(
            source,
            zarr_path=source_path,
            output_root=output,
            output_zarr_path=output_path,
            refined_run="r1",
            run_name="shape_separate",
            chunk_size=2,
            execution_backend="dask_worker_chunks",
            scheduler="single-threaded",
            centerline_crop_to_foreground=True,
            native_threads=1,
        )


def test_canonical_subject_shape_writer_and_materializer_reject_reordered_components(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    canonical_refined_root: zarr.Group,
) -> None:
    _patch_provenance(monkeypatch)
    reordered = ("swim_bladder", "subject_body", "eye_left", "eye_right")
    with pytest.raises(ValueError, match="exact component order"):
        mod.write_subject_shape_run_group(
            canonical_refined_root,
            refined_run="r1",
            run_name="shape_reordered",
            components=reordered,
            chunk_size=2,
        )
    assert canonical_refined_root.get("analysis/subject_shape_runs") is None

    with pytest.raises(ValueError, match="exact component order"):
        materializer.build_subject_shape_materialization_plan(
            tmp_path / "canonical.zarr",
            scratch_root=tmp_path / "scratch-reordered",
            refined_run="r1",
            run_name="shape_reordered",
            components=reordered,
            block_rows=2,
            output_shard_rows=4,
            execution_backend="serial_driver",
            scheduler="single-threaded",
            num_workers=1,
            shard_copy_workers=1,
            native_threads=1,
        )


def test_subject_shape_candidate_plan_rejects_resolved_containment_and_existing_name(
    tmp_path: Path,
    canonical_refined_root: zarr.Group,
) -> None:
    source = tmp_path / "canonical.zarr"
    source_alias = tmp_path / "canonical-alias.zarr"
    source_alias.symlink_to(source, target_is_directory=True)
    with pytest.raises(ValueError, match="either containment direction"):
        materializer.build_subject_shape_materialization_plan(
            source_alias,
            scratch_root=source / "unsafe-scratch",
            refined_run="r1",
            run_name="shape_candidate",
            storage_profile=SUBJECT_SHAPE_ACCESS_AWARE_CANDIDATE_PROFILE_ID,
        )
    with pytest.raises(ValueError, match="either containment direction"):
        materializer.build_subject_shape_materialization_plan(
            source,
            scratch_root=source,
            refined_run="r1",
            run_name="shape_candidate_equal",
            storage_profile=SUBJECT_SHAPE_ACCESS_AWARE_CANDIDATE_PROFILE_ID,
        )
    with pytest.raises(ValueError, match="either containment direction"):
        materializer.build_subject_shape_materialization_plan(
            source,
            scratch_root=tmp_path,
            refined_run="r1",
            run_name="shape_candidate_reverse",
            storage_profile=SUBJECT_SHAPE_ACCESS_AWARE_CANDIDATE_PROFILE_ID,
        )

    parent = canonical_refined_root.require_group("analysis").require_group(
        "subject_shape_runs"
    )
    parent.create_group("occupied_candidate")
    with pytest.raises(FileExistsError, match="replace existing authoritative run"):
        materializer.build_subject_shape_materialization_plan(
            source,
            scratch_root=tmp_path / "safe-scratch",
            refined_run="r1",
            run_name="occupied_candidate",
            storage_profile=SUBJECT_SHAPE_ACCESS_AWARE_CANDIDATE_PROFILE_ID,
        )


def test_subject_shape_materializer_computes_shards_and_publishes(
    monkeypatch,
    tmp_path: Path,
    canonical_refined_root: zarr.Group,
) -> None:
    _patch_provenance(monkeypatch)
    monkeypatch.setattr(
        materializer,
        "write_best_effort_run_lineage_attrs",
        lambda *args, **kwargs: None,
    )
    source_path = tmp_path / "canonical.zarr"
    source = canonical_refined_root
    masks_before = np.asarray(
        source["refined_subject_masks_runs"]["r1"]["masks_roi"][:]
    ).copy()
    scratch = tmp_path / "scratch"

    planned = materializer.materialize_subject_shape(
        source_path,
        scratch_root=scratch,
        refined_run="r1",
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
        refined_run="r1",
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
        np.asarray(root["refined_subject_masks_runs"]["r1"]["masks_roi"][:]),
        masks_before,
    )
    parent = root["analysis"]["subject_shape_runs"]
    assert parent.attrs["latest"] == "shape_materialized"
    run = parent["shape_materialized"]
    assert run.attrs["palette_run_completion_status"] == "complete"
    assert run.attrs["coordinate_binding_status"] == "bound_canonical_v2"
    assert run.attrs["stage_selector_eligible"] is True
    assert run.attrs["centerline_crop_to_foreground"] is True
    assert run.attrs["physical_storage_layout"]["layout"] == "zarr_v3_indexed_sharding"
    assert tuple(run["components"]["subject_body"]["centerline_xy"].shards[1:]) == (
        64,
        2,
    )
    cluster_output_staging = run.attrs["cluster_output_staging"]
    assert cluster_output_staging["schema_id"] == materializer.PUBLISH_SCHEMA_ID
    assert cluster_output_staging["final_validation"]["valid"] is True
    assert scratch.is_dir()
    scratch_compute = zarr.open_group(
        str(scratch / "compute.zarr"),
        mode="r",
        use_consolidated=False,
    )["analysis/subject_shape_runs/shape_materialized"]
    assert scratch_compute.attrs["coordinate_binding_status"] == (
        "unbound_numeric_stage_complete_v1"
    )
    assert "coordinate_records" not in scratch_compute
    assert "instance_key" not in scratch_compute
    consumed = run["coordinate_records/consumed_unbound_stage"]
    scratch_receipt = scratch_compute.attrs[SUBJECT_SHAPE_UNBOUND_MANIFEST_ATTR]
    assert consumed.attrs[SUBJECT_SHAPE_CONSUMED_UNBOUND_STAGE_ATTR] == scratch_receipt
    assert run.attrs["unbound_numeric_stage_manifest_sha256_consumed"] == (
        consumed.attrs[f"{SUBJECT_SHAPE_CONSUMED_UNBOUND_STAGE_ATTR}_sha256"]
    )
    derivation = run.attrs[SUBJECT_SHAPE_DERIVATION_ATTR]
    assert derivation["unbound_numeric_stage"]["record_ref"].endswith(
        "coordinate_records/consumed_unbound_stage@"
        f"{SUBJECT_SHAPE_CONSUMED_UNBOUND_STAGE_ATTR}"
    )


def test_subject_shape_byte_planned_candidate_is_complete_ineligible_and_pointer_free(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    canonical_refined_root: zarr.Group,
) -> None:
    _patch_provenance(monkeypatch)
    monkeypatch.setattr(
        materializer,
        "write_best_effort_run_lineage_attrs",
        lambda *args, **kwargs: None,
    )
    source_path = tmp_path / "canonical.zarr"
    original_pointers = {
        "latest": "preexisting_subject_shape",
        "latest_complete": "preexisting_subject_shape",
    }
    canonical_refined_root.require_group("analysis").require_group(
        "subject_shape_runs"
    ).attrs.update(original_pointers)
    result = materializer.materialize_subject_shape(
        source_path,
        scratch_root=tmp_path / "candidate-scratch",
        refined_run="r1",
        run_name="shape_byte_candidate",
        storage_profile=SUBJECT_SHAPE_ACCESS_AWARE_CANDIDATE_PROFILE_ID,
        block_rows=2,
        execution_backend="serial_driver",
        scheduler="single-threaded",
        num_workers=1,
        shard_copy_workers=1,
        native_threads=1,
        copy_backend="python",
        apply=True,
        keep_scratch=True,
        check_capacity=False,
        stage_command="unit-test-subject-shape-byte-candidate",
    )

    assert result["status"] == "complete"
    direct_root = zarr.open_group(
        str(source_path), mode="r", use_consolidated=False
    )
    consolidated_root = zarr.open_group(
        str(source_path), mode="r", use_consolidated=True
    )
    direct_parent = direct_root["analysis/subject_shape_runs"]
    consolidated_parent = consolidated_root["analysis/subject_shape_runs"]
    assert {
        name: direct_parent.attrs[name] for name in original_pointers
    } == original_pointers
    assert dict(direct_parent.attrs) == dict(consolidated_parent.attrs)
    direct = direct_parent["shape_byte_candidate"]
    consolidated = consolidated_parent["shape_byte_candidate"]
    assert direct.attrs["palette_run_completion_status"] == "complete"
    assert direct.attrs["stage_selector_eligible"] is False
    assert direct.attrs["subject_shape_storage_profile_id"] == (
        SUBJECT_SHAPE_ACCESS_AWARE_CANDIDATE_PROFILE_ID
    )
    assert not validate_subject_shape_candidate_storage(direct, phase="bound")
    assert not validate_subject_shape_direct_consolidated_storage(
        direct,
        consolidated,
    )
    compute = zarr.open_group(
        str(tmp_path / "candidate-scratch/compute.zarr"),
        mode="a",
        use_consolidated=False,
    )["analysis/subject_shape_runs/shape_byte_candidate"]
    local_candidate = zarr.open_group(
        str(tmp_path / "candidate-scratch/subject-shape-sharded-run"),
        mode="r",
        use_consolidated=False,
    )

    def arrays(group: zarr.Group, prefix: str = "") -> dict[str, zarr.Array]:
        result: dict[str, zarr.Array] = {}
        for name in group.array_keys():
            path = f"{prefix}/{name}" if prefix else str(name)
            result[path] = group[name]
        for name in group.group_keys():
            path = f"{prefix}/{name}" if prefix else str(name)
            result.update(arrays(group[name], path))
        return result

    compute_arrays = arrays(compute)
    candidate_arrays = arrays(local_candidate)
    assert set(candidate_arrays) == set(compute_arrays)
    for path in sorted(compute_arrays):
        source_array = compute_arrays[path]
        candidate_array = candidate_arrays[path]
        assert np.dtype(candidate_array.dtype) == np.dtype(source_array.dtype)
        source_fill = source_array.fill_value
        candidate_fill = candidate_array.fill_value
        if isinstance(source_fill, (float, np.floating)) and np.isnan(source_fill):
            assert isinstance(candidate_fill, (float, np.floating))
            assert np.isnan(candidate_fill)
        else:
            assert candidate_fill == source_fill
    assert candidate_arrays[
        "components/subject_body/bspline_degree_used"
    ].fill_value == -1
    receipt = direct.attrs["subject_shape_storage_plan"]
    degree_receipt = next(
        entry
        for entry in receipt["payload"]["arrays"]
        if entry["path"] == "components/subject_body/bspline_degree_used"
    )
    assert degree_receipt["declaration"]["fill_semantics"] == (
        "minus_one_means_invalid"
    )
    assert receipt["payload"]["object_estimate"]["array_metadata_objects"] == len(
        tuple(direct.array_keys())
    ) + sum(
        len(tuple(group.array_keys()))
        for group in (
            direct["row_index"],
            direct["body_frame"],
            direct["source_refined_subject_masks"],
            direct["components/subject_body"],
            direct["components/swim_bladder"],
            direct["components/eye_left"],
            direct["components/eye_right"],
            direct["relations/eye_pair"],
            direct["relations/swim_bladder_to_body"],
            direct["relations/eyes_to_body"],
        )
    )
    centerline_metadata = direct[
        "components/subject_body/centerline_xy"
    ].metadata.to_dict()
    codecs = centerline_metadata["codecs"]
    data_codecs = (
        codecs[0]["configuration"]["codecs"]
        if codecs[0]["name"] == "sharding_indexed"
        else codecs
    )
    assert data_codecs[1] == {
        "name": "zstd",
        "configuration": {"level": 0, "checksum": False},
    }
    writable = zarr.open_group(
        str(source_path), mode="a", use_consolidated=False
    )["analysis/subject_shape_runs/shape_byte_candidate"]
    digest = writable.attrs["subject_shape_storage_plan_payload_sha256"]
    writable.attrs["subject_shape_storage_plan_payload_sha256"] = "0" * 64
    assert any(
        "digest mismatch" in error
        for error in validate_subject_shape_candidate_storage(
            writable,
            phase="bound",
        )
    )
    writable.attrs["subject_shape_storage_plan_payload_sha256"] = digest
    centerline = writable["components/subject_body/centerline_xy"]
    storage_profile_id = centerline.attrs["storage_profile_id"]
    centerline.attrs["storage_profile_id"] = "hostile_profile"
    assert any(
        "physical metadata differs" in error
        for error in validate_subject_shape_candidate_storage(
            writable,
            phase="bound",
        )
    )
    centerline.attrs["storage_profile_id"] = storage_profile_id
    writable["components/subject_body"].attrs["unconsolidated_tamper"] = True
    assert any(
        "group declarations differ" in error
        for error in validate_subject_shape_direct_consolidated_storage(
            writable,
            consolidated,
        )
    )
    writable.create_group("unexpected_unconsolidated_group")
    assert any(
        "group paths differ" in error
        for error in validate_subject_shape_direct_consolidated_storage(
            writable,
            consolidated,
        )
    )
    degree = compute["components/subject_body/bspline_degree_used"]
    original_degree = int(degree[0])
    degree[0] = original_degree + 1
    with pytest.raises(ValueError, match="manifest differs from live arrays"):
        materialize_subject_shape_storage_candidate(
            compute,
            tmp_path / "candidate-from-tampered-source",
            copy_block_rows=2,
        )
    degree[0] = original_degree
    revision_group = compute["source_refined_subject_masks"]
    revision_available = np.asarray(
        revision_group["row_revision_available"][:],
        dtype=np.uint8,
    )
    del revision_group["row_revision_available"]
    revision_group.create_array(
        "row_revision_available",
        data=revision_available,
        chunks=revision_available.shape,
    )
    with pytest.raises(
        (ValueError, SubjectShapeCoordinatePublicationError),
        match="manifest|source-revision arrays",
    ):
        materialize_subject_shape_storage_candidate(
            compute,
            tmp_path / "candidate-from-dtype-tampered-source",
            copy_block_rows=2,
        )


def test_subject_shape_candidate_repairs_failed_visibility_after_consolidation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.zarr"
    root = zarr.open_group(str(source), mode="w", zarr_format=3)
    root.require_group("analysis").require_group("subject_shape_runs")
    local = tmp_path / "local-candidate"
    local_run = zarr.open_group(str(local), mode="w", zarr_format=3)
    owner = "a" * 32
    local_run.attrs.update(
        {
            SUBJECT_SHAPE_PUBLICATION_OWNER_ATTR: owner,
            "palette_run_completion_status": "running",
            "stage_selector_eligible": False,
        }
    )
    plan = SimpleNamespace(
        source_zarr=source,
        sharded_run=local,
        publication_run_path=local,
        target_run_path=(
            source / "analysis" / "subject_shape_runs" / "shape_failed_candidate"
        ),
        run_name="shape_failed_candidate",
        row_count=1,
        refined_run="refined_1",
        storage_profile_id=SUBJECT_SHAPE_ACCESS_AWARE_CANDIDATE_PROFILE_ID,
    )

    monkeypatch.setattr(
        materializer,
        "_validate_subject_shape_run",
        lambda *args, **kwargs: {"valid": True, "errors": []},
    )
    monkeypatch.setattr(
        materializer,
        "write_best_effort_run_lineage_attrs",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        materializer,
        "refresh_unbound_subject_shape_manifest_after_storage_materialization",
        lambda *args, **kwargs: {"valid": True},
    )
    monkeypatch.setattr(
        materializer,
        "validate_subject_shape_storage_source_manifest_link",
        lambda *args, **kwargs: (),
    )
    monkeypatch.setattr(
        materializer,
        "bind_staged_subject_shape_run",
        lambda *args, **kwargs: {"valid": True},
    )
    monkeypatch.setattr(
        materializer,
        "finalize_bound_subject_shape_storage_receipt",
        lambda *args, **kwargs: {
            "payload": {"object_estimate": {"array_metadata_objects": 0}}
        },
    )
    monkeypatch.setattr(
        materializer,
        "set_subject_shape_metadata_visibility_policy",
        lambda *args, **kwargs: None,
    )

    def complete_candidate(_root, run_group, **_kwargs):
        run_group.attrs["palette_run_completion_status"] = "complete"
        run_group.attrs["coordinate_binding_status"] = SUBJECT_SHAPE_BOUND_CANONICAL_STATUS
        return {"valid": True}

    monkeypatch.setattr(
        materializer,
        "complete_bound_subject_shape_candidate_run",
        complete_candidate,
    )
    monkeypatch.setattr(
        materializer,
        "validate_subject_shape_direct_consolidated_storage",
        lambda *args, **kwargs: ("injected post-consolidation failure",),
    )
    monkeypatch.setattr(
        materializer,
        "load_completed_ineligible_subject_shape_coordinate_publication",
        lambda *args, **kwargs: SimpleNamespace(
            row_identity=SimpleNamespace(leading_dimension=1),
            manifest=SimpleNamespace(record_sha256="b" * 64),
        ),
    )

    with pytest.raises(
        RuntimeError,
        match="direct/consolidated candidate declarations differ",
    ):
        materializer.publish_subject_shape_run(
            plan,
            materialization_payload={},
            copy_backend="python",
        )

    direct = zarr.open_group(
        str(source), mode="r", use_consolidated=False
    )["analysis/subject_shape_runs/shape_failed_candidate"]
    consolidated = zarr.open_group(
        str(source), mode="r", use_consolidated=True
    )["analysis/subject_shape_runs/shape_failed_candidate"]
    assert dict(direct.attrs) == dict(consolidated.attrs)
    assert direct.attrs["palette_run_completion_status"] == "failed"
    assert direct.attrs["stage_selector_eligible"] is False
    assert ATOMIC_PUBLICATION_TOMBSTONE_ATTR in direct.attrs


def test_subject_shape_failed_exact_receipt_rollback_never_restores_precopy_snapshot(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.zarr"
    root = zarr.open_group(str(source), mode="w", use_consolidated=False)
    parent = root.require_group("analysis").require_group("subject_shape_runs")
    parent.attrs.update({"latest": "shape_a", "latest_complete": "shape_a"})

    local = tmp_path / "local-shape.zarr"
    local_run = zarr.open_group(str(local), mode="w", use_consolidated=False)
    candidate_owner = "a" * 32
    local_run.attrs.update(
        {
            SUBJECT_SHAPE_PUBLICATION_OWNER_ATTR: candidate_owner,
            "palette_run_completion_status": "running",
            "stage_selector_eligible": False,
        }
    )
    target = source / "analysis" / "subject_shape_runs" / "shape_c"
    plan = SimpleNamespace(
        source_zarr=source,
        sharded_run=local,
        target_run_path=target,
        run_name="shape_c",
        row_count=1,
        refined_run="refined_1",
    )
    receipt = object()
    intervening_seen: list[bool] = []
    rollback_attempted: list[bool] = []
    intervening_owner = "b" * 32

    def validate(path: Path, **_kwargs):
        group = zarr.open_group(str(path), mode="r", use_consolidated=False)
        complete = group.attrs.get("palette_run_completion_status") == "complete"
        return {
            "valid": not complete,
            "errors": ["injected failure after deferred receipt"] if complete else [],
            "row_count": 1,
        }

    def complete_with_intervening_publication(
        open_root,
        run_group,
        *,
        expected_run_name,
        publication_owner,
    ):
        assert expected_run_name == "shape_c"
        assert publication_owner == candidate_owner
        run_group.attrs.update(
            {
                "palette_run_completion_status": "complete",
                "stage_selector_eligible": False,
                "coordinate_binding_status": SUBJECT_SHAPE_BOUND_CANONICAL_STATUS,
            }
        )
        live_parent = open_root["analysis/subject_shape_runs"]
        live_parent.attrs.update(
            {
                "latest": "shape_b",
                "latest_complete": "shape_b",
                SUBJECT_SHAPE_PUBLICATION_POLICY_ATTR: (
                    SUBJECT_SHAPE_PUBLICATION_POLICY
                ),
                SUBJECT_SHAPE_PUBLICATION_GENERATION_ATTR: 1,
                SUBJECT_SHAPE_PARENT_PUBLICATION_LEASE_ATTR: {
                    "owner_uuid": intervening_owner,
                    "base_generation": 0,
                    "next_generation": 1,
                },
            }
        )
        intervening_seen.append(True)
        live_parent.attrs.update(
            {
                "latest": "shape_c",
                "latest_complete": "shape_c",
                SUBJECT_SHAPE_PUBLICATION_POLICY_ATTR: (
                    SUBJECT_SHAPE_PUBLICATION_POLICY
                ),
                SUBJECT_SHAPE_PUBLICATION_GENERATION_ATTR: 2,
                SUBJECT_SHAPE_PARENT_PUBLICATION_LEASE_ATTR: {
                    "schema_id": "palette.subject_shape_publication_lease",
                    "schema_version": 1,
                    "policy": SUBJECT_SHAPE_PUBLICATION_POLICY,
                    "run_path": "analysis/subject_shape_runs/shape_c",
                    "publication_owner": candidate_owner,
                    "owner_uuid": candidate_owner,
                    "base_generation": 1,
                    "next_generation": 2,
                },
            }
        )
        return {"valid": True}, receipt

    def fail_exact_rollback(value) -> None:
        assert value is receipt
        rollback_attempted.append(True)
        raise RuntimeError("injected exact subject-shape receipt rollback failure")

    monkeypatch.setattr(materializer, "_validate_subject_shape_run", validate)
    monkeypatch.setattr(
        materializer,
        "write_best_effort_run_lineage_attrs",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        materializer,
        "bind_staged_subject_shape_run",
        lambda *_args, **_kwargs: {"valid": True},
    )
    monkeypatch.setattr(
        materializer,
        "complete_bound_subject_shape_run_for_deferred_activation",
        complete_with_intervening_publication,
    )
    monkeypatch.setattr(
        materializer,
        "rollback_deferred_subject_shape_coordinate_activation",
        fail_exact_rollback,
    )

    with pytest.raises(
        RuntimeError,
        match="rollback was incomplete.*exact subject-shape receipt rollback failure",
    ):
        materializer.publish_subject_shape_run(
            plan,
            materialization_payload={},
            copy_backend="python",
        )

    assert intervening_seen == [True]
    assert rollback_attempted == [True]
    root = zarr.open_group(str(source), mode="r", use_consolidated=False)
    parent = root["analysis/subject_shape_runs"]
    failed = parent["shape_c"]
    assert failed.attrs["stage_selector_eligible"] is False
    assert failed.attrs["palette_run_completion_status"] == "failed"
    assert parent.attrs["latest"] == "shape_c"
    assert parent.attrs["latest_complete"] == "shape_c"
    assert parent.attrs[SUBJECT_SHAPE_PUBLICATION_GENERATION_ATTR] == 2
    assert parent.attrs[SUBJECT_SHAPE_PARENT_PUBLICATION_LEASE_ATTR][
        "owner_uuid"
    ] == candidate_owner


def test_unbound_subject_shape_receipt_rejects_scientific_and_schema_drift_before_binding(
    monkeypatch,
    tmp_path: Path,
    canonical_refined_root: zarr.Group,
) -> None:
    _patch_provenance(monkeypatch)
    root = canonical_refined_root
    scratch_path = tmp_path / "hostile-subject-shape-compute.zarr"
    scratch_root = zarr.open_group(str(scratch_path), mode="w", zarr_format=3)
    mod.write_subject_shape_run_group(
        root,
        zarr_path=tmp_path / "canonical.zarr",
        output_root=scratch_root,
        output_zarr_path=scratch_path,
        refined_run="r1",
        run_name="shape_unbound_hostile",
        chunk_size=2,
        _unbound_coordinate_stage=True,
    )
    run = scratch_root["analysis/subject_shape_runs/shape_unbound_hostile"]
    receipt = dict(run.attrs[SUBJECT_SHAPE_UNBOUND_MANIFEST_ATTR])
    assert receipt["scientific_configuration"]["run_attrs"]["bspline_smoothing"] == (
        mod.DEFAULT_BSPLINE_SMOOTHING
    )
    assert receipt["schema_inventory"]["closed_group_inventory"] is True
    assert receipt["schema_inventory"]["closed_array_inventory"] is True
    assert receipt["schema_inventory"]["closed_attr_inventory"] is True

    def validate() -> dict[str, object]:
        return mod.validate_unbound_subject_shape_run(
            root,
            run,
            expected_refined_run="r1",
            expected_run_name="shape_unbound_hostile",
        )

    # The authoritative source is immutable throughout this tamper family, so
    # its exact proofs may be shared. Every staged-run inventory/attr mutation
    # is still read and rejected independently, and the source is rechecked
    # when the outer proof scope closes.
    with proof_verification_scope():
        original_smoothing = run.attrs["bspline_smoothing"]
        run.attrs["bspline_smoothing"] = float(original_smoothing) + 1.0
        with pytest.raises(ValueError, match="unbound|scientific|manifest"):
            validate()
        run.attrs["bspline_smoothing"] = original_smoothing

        body = run["components/subject_body"]
        body.create_array(
            "debug_point_xy",
            data=np.zeros((2, 2), dtype=np.float32),
        )
        with pytest.raises(ValueError, match="array inventory"):
            validate()
        del body["debug_point_xy"]

        body.create_group("debug_empty")
        with pytest.raises(ValueError, match="group inventory"):
            validate()
        del body["debug_empty"]

        body.attrs["coordinate_space"] = "roi_local_px"
        with pytest.raises(ValueError, match="attrs differ"):
            validate()
        del body.attrs["coordinate_space"]

        body["centroid_xy"].attrs["coordinate_space"] = "roi_local_px"
        with pytest.raises(ValueError, match="attrs differ"):
            validate()
        del body["centroid_xy"].attrs["coordinate_space"]

        result = validate()
        assert result["valid"] is True


def test_future_subject_shape_writer_fails_closed_for_compact_only_refined_source(
    monkeypatch,
    canonical_refined_root: zarr.Group,
) -> None:
    _patch_provenance(monkeypatch)
    root = canonical_refined_root
    refined = root["refined_subject_masks_runs"]["r1"]
    write_component_rle_mask_store_from_dense(
        refined,
        refined["masks_roi"],
        component_names=tuple(str(value) for value in refined.attrs["mask_labels"]),
        encode_row_chunk_size=2,
    )
    del refined["masks_roi"]

    with pytest.raises(ValueError):
        mod.write_subject_shape_run_group(
            root,
            refined_run="r1",
            run_name="shape_rle_only",
            chunk_size=2,
        )


def test_write_subject_shape_run_uses_dask_worker_chunks(
    tmp_path: Path,
    monkeypatch,
    canonical_refined_root: zarr.Group,
) -> None:
    _patch_provenance(monkeypatch)
    zarr_path = tmp_path / "canonical.zarr"
    assert canonical_refined_root.path == ""

    summary = mod.write_subject_shape_run(
        zarr_path,
        refined_run="r1",
        run_name="shape_dask",
        chunk_size=1,
        execution_backend="dask_worker_chunks",
        scheduler="single-threaded",
    )

    assert summary["status"] == "updated"
    assert summary["execution_backend"] == "dask_worker_chunks"
    assert summary["chunk_size"] == 1
    assert summary["worker_chunk_size"] == 2
    assert summary["dask_requested_chunk_size"] == 1
    assert summary["dask_chunk_size"] == 2
    assert summary["dask_chunk_alignment"] == "refined_subject_mask_metric_row_chunk"
    assert summary["chunk_count"] == 1
    root = zarr.open_group(str(zarr_path), mode="r")
    run = root["analysis"]["subject_shape_runs"]["shape_dask"]
    assert run.attrs["dask_execution_enabled"] is True
    assert run.attrs["dask_scheduler"] == "single-threaded"
    assert run.attrs["chunk_size"] == 1
    assert run.attrs["worker_chunk_size"] == 2
    assert run.attrs["dask_requested_chunk_size"] == 1
    assert run.attrs["dask_chunk_size"] == 2
    assert run.attrs["dask_chunk_alignment"] == "refined_subject_mask_metric_row_chunk"
    assert run.attrs["provenance"]["parameters"]["worker_chunk_size"] == 2
    assert run.attrs["provenance"]["scheduler"]["dask_requested_chunk_size"] == 1
    assert run.attrs["provenance"]["scheduler"]["dask_chunk_size"] == 2
    assert run.attrs["subject_shape_chunk_timing_count"] == 1
    assert run.attrs["subject_shape_chunk_timing_storage"] == "summary_only"
    assert "subject_shape_chunk_timings" not in run.attrs
    assert run.attrs["subject_shape_timing_summary"]["chunk_timings"]["chunk_count"] == 1
    assert run["components"]["eye_left"]["ellipse_success"][:].tolist() == [True, True]
    assert run["relations"]["eye_pair"]["separation_valid"][:].tolist() == [True, False]
    assert run["body_frame"]["valid"][:].tolist() == [True, False]
    assert run["components"]["subject_body"]["centerline_valid"][:].tolist() == [True, False]


def test_subject_shape_dask_worker_chunk_size_rounds_to_metric_grid() -> None:
    assert mod._worker_chunk_size_for_backend(1000, 1, "dask_worker_chunks") == 256
    assert mod._worker_chunk_size_for_backend(1000, 256, "dask_worker_chunks") == 256
    assert mod._worker_chunk_size_for_backend(1000, 257, "dask_worker_chunks") == 512
    assert mod._worker_chunk_size_for_backend(1000, 500, "dask_worker_chunks") == 512
    assert mod._worker_chunk_size_for_backend(1000, 500, "serial_driver") == 500


def test_write_subject_shape_run_dry_run_does_not_create_subject_shape_group(
    canonical_refined_root: zarr.Group,
) -> None:
    root = canonical_refined_root

    summary = mod.write_subject_shape_run_group(
        root,
        refined_run="r1",
        run_name="shape_dry",
        dry_run=True,
    )

    assert summary["status"] == "planned"
    assert root.get("analysis/subject_shape_runs") is None


def test_subject_shape_writer_rejects_post_publication_mask_payload_mutation(
    monkeypatch,
    canonical_refined_root: zarr.Group,
) -> None:
    _patch_provenance(monkeypatch)
    root = canonical_refined_root
    refined = root["refined_subject_masks_runs"]["r1"]
    masks = np.asarray(refined["masks_roi"][:], dtype=np.uint8)
    masks[1, 0] = 0
    masks[1, 0, 2:5, 4:7] = 1
    masks[1, 0, 10:13, 9:12] = 1
    refined["masks_roi"][:] = masks

    with pytest.raises(ValueError):
        mod.write_subject_shape_run_group(
            root,
            refined_run="r1",
            run_name="shape_fragmented",
            chunk_size=2,
        )


def test_subject_shape_writer_rejects_unsealed_source_body_qc_mutation(
    monkeypatch,
    canonical_refined_root: zarr.Group,
) -> None:
    _patch_provenance(monkeypatch)
    root = canonical_refined_root
    refined = root["refined_subject_masks_runs"]["r1"]
    # The supported writer must honor the immutable canonical refined-run
    # guard; the test does not disable or monkeypatch that protection.
    with pytest.raises(RefinedSubjectMaskMutationError, match="immutable canonical publication"):
        write_subject_body_mask_qc_group(root, refined_run="r1", chunk_size=2)

    # Model an adversarial/out-of-band archive mutation after publication.  A
    # newly injected, unsealed QC group must invalidate the exact component-QC
    # inventory during subject-shape source preflight.
    body = refined.require_group("components").require_group("subject_body")
    qc = body.create_group("qc")
    qc.attrs.update(
        {
            "schema_id": "refined_subject_body_mask_qc",
            "schema_version": 1,
            "status": "adversarial_unsealed_test_fixture",
        }
    )
    assert "qc" in refined["components/subject_body"]

    with pytest.raises(ValueError):
        mod.write_subject_shape_run_group(
            root,
            refined_run="r1",
            run_name="shape_source_body_qc",
            chunk_size=2,
        )


def test_subject_shape_publication_rejects_post_publication_source_revision_injection(
    canonical_subject_shape_root: tuple[zarr.Group, dict[str, object]],
) -> None:
    root, _summary = canonical_subject_shape_root

    refined = root["refined_subject_masks_runs/r1"]
    body = refined.require_group("components").require_group("subject_body")
    body.create_array(
        "row_revision",
        data=np.asarray([0, 1], dtype=np.int64),
        overwrite=True,
    )

    with pytest.raises(SubjectShapeCoordinatePublicationError):
        load_persisted_subject_shape_coordinate_publication(
            root,
            "analysis/subject_shape_runs/shape_001",
        )
    with pytest.raises(ValueError):
        mod.write_subject_shape_run_group(
            root,
            refined_run="r1",
            run_name="shape_after_revision_injection",
            chunk_size=2,
        )


def test_write_subject_shape_run_group_copies_instance_key_lineage(
    canonical_subject_shape_root: tuple[zarr.Group, dict[str, object]],
) -> None:
    root, _summary = canonical_subject_shape_root
    refined = root["refined_subject_masks_runs"]["r1"]

    run = root["analysis"]["subject_shape_runs"]["shape_001"]
    np.testing.assert_array_equal(run["instance_key"][:], refined["instance_key"][:])
    np.testing.assert_array_equal(
        run["source_crop_row_ids"][:],
        refined["source_crop_row_ids"][:],
    )
    assert "instance_key" in run.attrs["row_lineage_copied"]
    assert "source_crop_row_ids" in run.attrs["row_lineage_copied"]


def test_future_subject_shape_writer_fails_closed_without_instance_key(monkeypatch) -> None:
    _patch_provenance(monkeypatch)
    root = _build_refined_root()

    with pytest.raises(ValueError):
        mod.write_subject_shape_run_group(
            root,
            refined_run="refined_001",
            run_name="shape_001",
            chunk_size=2,
        )


def test_canonical_subject_shape_template_clones_isolate_mutations(
    tmp_path: Path,
    canonical_subject_shape_template: tuple[Path, dict[str, object]],
) -> None:
    template, _summary = canonical_subject_shape_template
    first_path = tmp_path / "first.zarr"
    second_path = tmp_path / "second.zarr"
    shutil.copytree(template, first_path)
    shutil.copytree(template, second_path)
    first = zarr.open_group(str(first_path), mode="a", use_consolidated=False)
    second = zarr.open_group(str(second_path), mode="r", use_consolidated=False)
    first_values = np.asarray(
        first["analysis/subject_shape_runs/shape_001/instance_key"][:]
    ).copy()
    first["analysis/subject_shape_runs/shape_001/instance_key"][:] = first_values[::-1]

    np.testing.assert_array_equal(
        second["analysis/subject_shape_runs/shape_001/instance_key"][:],
        first_values,
    )
    pristine = zarr.open_group(str(template), mode="r", use_consolidated=False)
    np.testing.assert_array_equal(
        pristine["analysis/subject_shape_runs/shape_001/instance_key"][:],
        first_values,
    )
