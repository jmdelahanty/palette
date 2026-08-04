from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import numpy as np
import pytest
import zarr

from fisheye.analysis import tail_kinematics_runs as tail_mod
from fisheye.analysis import subject_shape_io
from fisheye.analysis_workflows.materializers import tail_kinematics as mod
from fisheye.analysis_workflows.materializers import atomic_run_publisher as atomic_mod
from fisheye.analysis_workflows.tail_kinematics_candidate_execution import (
    compute_tail_kinematics_logical_hashes,
)
from fisheye.shared.coordinate_frame_record import array_payload_sha256
from fisheye.shared.zarr.storage_profiles import PUBLISHED_HTTP_V1

_REAL_SUBJECT_SHAPE_PUBLICATION_LOADER = (
    subject_shape_io.load_persisted_subject_shape_coordinate_publication
)


def _fake_coordinate_publication(shape: zarr.Group, run_path: str):
    arrays: dict[str, dict[str, object]] = {}
    for relative_ref in (
        *tail_mod._REQUIRED_SOURCE_ARRAY_PATHS,
        *tail_mod._OPTIONAL_SOURCE_ARRAY_PATHS,
    ):
        node = shape.get(relative_ref)
        if node is None:
            continue
        arrays[relative_ref] = {
            "array_ref": f"/{run_path}/{relative_ref}",
            "relative_ref": relative_ref,
            "dtype": np.dtype(node.dtype).str,
            "shape": [int(value) for value in node.shape],
            "content_sha256": array_payload_sha256(node),
            "canonicalization": "numpy_dtype_shape_c_order_bytes_v1",
        }
    semantics = SimpleNamespace(
        record_ref=f"/{run_path}/components/subject_body/tail_curvature_px_inv@subject_shape_scalar_surface",
        record_sha256="4" * 64,
    )

    def require_scalar_surface(relative_ref, *, units=None, surface_kind=None):
        assert relative_ref == "components/subject_body/tail_curvature_px_inv"
        assert units == "px^-1"
        assert surface_kind == "row_profile"
        return SimpleNamespace(semantics=semantics)

    return SimpleNamespace(
        manifest=SimpleNamespace(
            record={"arrays": arrays},
            record_ref=f"/{run_path}@subject_shape_publication_manifest",
            record_sha256="1" * 64,
        ),
        row_identity=SimpleNamespace(
            record_ref=f"/{run_path}@row_identity",
            record_sha256="2" * 64,
        ),
        tail_sample_axis=SimpleNamespace(
            record_ref=f"/{run_path}/coordinate_records/tail_sample_axis@subject_shape_tail_sample_axis",
            record_sha256="3" * 64,
        ),
        body_frame=SimpleNamespace(
            record_ref=f"/{run_path}/body_frame@fish_anatomical_body_frame",
            record_sha256="5" * 64,
        ),
        require_scalar_surface=require_scalar_surface,
    )


@pytest.fixture(autouse=True)
def _canonical_subject_shape_publication(monkeypatch):
    monkeypatch.setattr(
        subject_shape_io,
        "load_persisted_subject_shape_coordinate_publication",
        lambda root, run_path: _fake_coordinate_publication(root[run_path], run_path),
    )


def _patch_provenance(monkeypatch) -> None:
    monkeypatch.setattr(
        tail_mod,
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
        tail_mod,
        "get_environment_info",
        lambda **kwargs: {  # noqa: ARG005
            "environment": {"python": "3.11"},
            "platform": {
                "hostname": "materializer-host",
                "system": "Linux",
                "release": "6.8",
                "python_version": "3.11.0",
                "machine": "x86_64",
            },
        },
    )

    def _fake_publish(_root, run_group):
        run_group.attrs["tail_coordinate_publication_manifest_sha256"] = "9" * 64

    monkeypatch.setattr(
        tail_mod,
        "publish_tail_kinematics_coordinate_surfaces",
        _fake_publish,
    )

    activation_receipt = object()
    staged: dict[str, object] = {}

    def _defer(_root, parent, run_group, *, run_name, **_kwargs):
        assert run_group.attrs["palette_run_completion_status"] == "complete"
        assert run_group.attrs["stage_selector_eligible"] is False
        parent.attrs["latest_complete"] = run_name
        parent.attrs["latest"] = run_name
        staged.update(parent=parent, run_group=run_group)
        return activation_receipt

    monkeypatch.setattr(
        mod.tail_publication_mod,
        "defer_tail_coordinate_publication_activation",
        _defer,
    )

    def _commit(receipt, *, root, parent, run):
        assert receipt is activation_receipt
        assert root["analysis/tail_kinematics_runs"] is not None
        assert parent.path == "analysis/tail_kinematics_runs"
        assert (
            run.path == f"analysis/tail_kinematics_runs/{run.attrs['palette_run_name']}"
        )
        assert run.attrs["stage_selector_eligible"] is False
        run.attrs["stage_selector_eligible"] = True

    monkeypatch.setattr(
        mod.tail_publication_mod,
        "commit_deferred_tail_coordinate_publication_activation",
        _commit,
    )
    monkeypatch.setattr(
        mod.tail_publication_mod,
        "rollback_deferred_tail_coordinate_publication_activation",
        lambda receipt: None,
    )


def _build_source_zarr(path: Path, *, row_count: int = 9) -> None:
    root = zarr.open_group(str(path), mode="w")
    root.attrs["recording_id"] = "materializer_fixture"
    analysis = root.create_group("analysis")
    shape_parent = analysis.create_group("subject_shape_runs")
    shape_parent.attrs["latest"] = "shape_001"
    shape = shape_parent.create_group("shape_001")
    shape.attrs["schema_id"] = "analysis.subject_shape_runs"
    shape.attrs["schema_version"] = 3
    shape.attrs["method"] = "fixture_subject_shape"
    shape.attrs["method_version"] = 1
    shape.attrs["palette_run_completion_status"] = "complete"
    shape.attrs["source_refined_subject_masks_run"] = "refined_001"
    shape.attrs["body_frame_schema_id"] = "fish_anatomical_body_frame"

    source_s = np.linspace(0.0, 1.0, 4, dtype=np.float32)
    tail_xy = np.zeros((row_count, 4, 2), dtype=np.float32)
    tail_xy[:, :, 0] = -10.0 * source_s[None, :]
    tangent = np.zeros_like(tail_xy)
    tangent[:, :, 0] = -1.0

    components = shape.create_group("components")
    body = components.create_group("subject_body")
    body.attrs["tail_sample_count"] = 4
    body.create_array("tail_sample_s", data=source_s, chunks=(4,), overwrite=True)
    body.create_array("tail_sample_xy", data=tail_xy, chunks=(2, 4, 2), overwrite=True)
    body.create_array("tail_tangent_xy", data=tangent, chunks=(2, 4, 2), overwrite=True)
    body.create_array(
        "tail_curvature_px_inv",
        data=np.zeros((row_count, 4), dtype=np.float32),
        chunks=(2, 4),
        overwrite=True,
    )
    body.create_array(
        "tail_sample_valid",
        data=np.ones((row_count,), dtype=bool),
        chunks=(2,),
        overwrite=True,
    )
    body.create_array(
        "bspline_valid",
        data=np.ones((row_count,), dtype=bool),
        chunks=(2,),
        overwrite=True,
    )
    body.create_array(
        "tail_base_xy",
        data=np.zeros((row_count, 2), dtype=np.float32),
        chunks=(2, 2),
        overwrite=True,
    )
    body.create_array(
        "tail_sample_failure_reason_bytes",
        data=tail_mod._encode_reasons(["ok"] * row_count),
        chunks=(2, 64),
        overwrite=True,
    )
    body.create_array(
        "bspline_failure_reason_bytes",
        data=tail_mod._encode_reasons(["ok"] * row_count),
        chunks=(2, 64),
        overwrite=True,
    )
    body.create_array(
        "unrelated_debug_surface",
        data=np.arange(row_count * 32, dtype=np.float32).reshape(row_count, 32),
        chunks=(2, 32),
        overwrite=True,
    )

    body_frame = shape.create_group("body_frame")
    body_frame.create_array(
        "forward_axis_xy",
        data=np.repeat(np.asarray([[1.0, 0.0]], dtype=np.float32), row_count, axis=0),
        chunks=(2, 2),
        overwrite=True,
    )
    body_frame.create_array(
        "left_axis_xy",
        data=np.repeat(np.asarray([[0.0, 1.0]], dtype=np.float32), row_count, axis=0),
        chunks=(2, 2),
        overwrite=True,
    )
    body_frame.create_array(
        "axis_valid",
        data=np.ones((row_count,), dtype=bool),
        chunks=(2,),
        overwrite=True,
    )
    body_frame.create_array(
        "failure_reason_bytes",
        data=tail_mod._encode_reasons(["ok"] * row_count),
        chunks=(2, 64),
        overwrite=True,
    )

    shape.create_array(
        "source_acquisition_frame_index",
        data=np.arange(100, 100 + row_count, dtype=np.int32),
        chunks=(2,),
        overwrite=True,
    )
    shape.create_array(
        "source_crop_row_ids",
        data=np.arange(200, 200 + row_count, dtype=np.int64),
        chunks=(2,),
        overwrite=True,
    )
    shape.create_array(
        "instance_key",
        data=np.arange(300, 300 + row_count, dtype=np.uint64),
        chunks=(2,),
        overwrite=True,
    )

    revisions = shape.create_group("source_refined_subject_masks")
    revisions.attrs["source_run"] = "refined_001"
    revisions.create_array(
        "row_revision",
        data=np.arange(row_count, dtype=np.int64)[:, None],
        chunks=(2, 1),
        overwrite=True,
    )
    revisions.create_array(
        "row_revision_available",
        data=np.asarray([True], dtype=bool),
        chunks=(1,),
        overwrite=True,
    )


def test_materialization_plan_selects_only_required_subject_shape_surface(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.zarr"
    scratch = tmp_path / "scratch"
    _build_source_zarr(source)

    plan = mod.build_tail_kinematics_materialization_plan(
        source,
        scratch_root=scratch,
        shape_run="shape_001",
        run_name="tail_001",
        block_rows=3,
        output_shard_rows=7,
    )

    selected = {item.relative_path for item in plan.physical_files}
    assert any("tail_sample_xy" in path for path in selected)
    assert not any("unrelated_debug_surface" in path for path in selected)
    assert plan.row_count == 9
    assert plan.requested_block_rows == 3
    assert plan.requested_output_shard_rows == 7
    assert plan.source_bytes > 0
    assert plan.estimated_output_bytes > 0
    assert len(plan.source_metadata_sha256) == 64
    assert plan.source_contract["schema_id"] == "analysis.subject_shape_runs"
    assert plan.source_contract["schema_version"] == 3
    assert plan.source_contract["canonical_publication_manifest_sha256"] == "1" * 64
    assert len(plan.staged_source_authority["record_sha256"]) == 64
    assert plan.staged_source_authority["closed_array_inventory"] is True
    assert plan.staged_source_authority["normal_reader_authority"] is False
    assert not scratch.exists()


def test_staged_authority_rejects_swap_tamper_and_normal_reader_bypass(
    monkeypatch,
    tmp_path: Path,
) -> None:
    source_a = tmp_path / "source-a.zarr"
    source_b = tmp_path / "source-b.zarr"
    _build_source_zarr(source_a)
    _build_source_zarr(source_b)
    root_b = zarr.open_group(str(source_b), mode="a", use_consolidated=False)
    tail_b = root_b[
        "analysis/subject_shape_runs/shape_001/components/subject_body/tail_sample_xy"
    ]
    changed = np.asarray(tail_b[:], dtype=np.float32)
    changed[:, :, 1] = 7.0
    tail_b[:] = changed

    plan_a = mod.build_tail_kinematics_materialization_plan(
        source_a,
        scratch_root=tmp_path / "scratch-a",
        shape_run="shape_001",
        run_name="tail_a",
    )
    plan_b = mod.build_tail_kinematics_materialization_plan(
        source_b,
        scratch_root=tmp_path / "scratch-b",
        shape_run="shape_001",
        run_name="tail_b",
    )
    mod.stage_tail_kinematics_sources(
        plan_a,
        copy_backend="python",
        check_capacity=False,
    )
    staged_root = zarr.open_group(
        str(plan_a.staged_zarr),
        mode="a",
        use_consolidated=False,
    )

    with pytest.raises(subject_shape_io.SubjectShapeIOError, match="canonical payload"):
        tail_mod._resolve_tail_kinematics_sources(
            staged_root,
            "shape_001",
            _staged_source_authority=plan_b.staged_source_authority,
        )

    staged_tail = staged_root[
        "analysis/subject_shape_runs/shape_001/components/subject_body/tail_sample_xy"
    ]
    tampered = np.asarray(staged_tail[:], dtype=np.float32)
    tampered[0, 0, 0] = 12345.0
    staged_tail[:] = tampered
    with pytest.raises(subject_shape_io.SubjectShapeIOError, match="canonical payload"):
        tail_mod._resolve_tail_kinematics_sources(
            staged_root,
            "shape_001",
            _staged_source_authority=plan_a.staged_source_authority,
        )

    monkeypatch.setattr(
        subject_shape_io,
        "load_persisted_subject_shape_coordinate_publication",
        _REAL_SUBJECT_SHAPE_PUBLICATION_LOADER,
    )
    with pytest.raises(
        subject_shape_io.SubjectShapeIOError,
        match="not a valid canonical coordinate publication",
    ):
        subject_shape_io.resolve_subject_shape_run(staged_root, "shape_001")


def test_materialize_tail_kinematics_stages_computes_and_atomically_publishes(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _patch_provenance(monkeypatch)
    monkeypatch.setattr(
        tail_mod, "refined_subject_mask_metric_row_chunk", lambda _total_rows: 2
    )
    source = tmp_path / "source.zarr"
    scratch = tmp_path / "scratch"
    _build_source_zarr(source)

    result = mod.materialize_tail_kinematics(
        source,
        scratch_root=scratch,
        shape_run="shape_001",
        run_name="tail_001",
        block_rows=3,
        copy_backend="python",
        apply=True,
        keep_scratch=True,
        check_capacity=False,
        stage_command="unit-test-materializer",
    )

    assert result["status"] == "complete"
    assert result["staging"]["status"] == "complete"
    assert result["publish"]["final_validation"]["valid"] is True
    assert scratch.is_dir()
    staged_body = (
        scratch
        / "source-subset.zarr/analysis/subject_shape_runs/shape_001/components/subject_body"
    )
    assert (staged_body / "tail_sample_xy").is_dir()
    assert not (staged_body / "unrelated_debug_surface").exists()
    manifest = json.loads(
        (scratch / "staging-manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["inventory"]["valid"] is True
    assert manifest["source_contract"]["method"] == "fixture_subject_shape"

    root = zarr.open_group(str(source), mode="r", use_consolidated=False)
    parent = root["analysis"]["tail_kinematics_runs"]
    assert parent.attrs["latest"] == "tail_001"
    run = parent["tail_001"]
    assert run.attrs["palette_run_completion_status"] == "complete"
    assert run.attrs["compute_kernel"] == "vectorized_shared_grid_v1"
    assert run.attrs["cluster_output_staging"]["schema_id"] == mod.PUBLISH_SCHEMA_ID
    assert (
        run.attrs["cluster_output_staging"]["pre_pointer_validation"]["valid"] is True
    )
    assert run.attrs["cluster_output_staging"]["final_validation"]["valid"] is True
    assert run.attrs["cluster_output_staging"]["rollback_policy"] == (
        "retain_owner_bound_failed_public_tombstone_and_"
        "stage_specific_receipt_rollback_only"
    )
    assert (
        run.attrs["cluster_output_staging"]["serialization_policy"]
        == "per_recording_advisory_file_lock"
    )
    assert (
        tmp_path / f".source.zarr.{atomic_mod.ARCHIVE_PUBLICATION_LOCK_SUFFIX}.lock"
    ).is_file()
    assert tuple(run["tail_angle_rad"].chunks) == (2, 10)
    assert run.attrs["effective_block_rows"] == 4
    assert run.attrs["effective_output_shard_rows"] == 10
    assert tuple(run["tail_angle_rad"].shards) == (10, 10)
    assert run["source_acquisition_frame_index"][:].tolist() == list(range(100, 109))


def test_materialize_byte_planned_tail_candidate_never_updates_selectors(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _patch_provenance(monkeypatch)
    source = tmp_path / "source.zarr"
    scratch = tmp_path / "scratch"
    _build_source_zarr(source, row_count=20_000)
    source_root = zarr.open_group(str(source), mode="a", use_consolidated=False)
    shape = source_root["analysis/subject_shape_runs/shape_001"]
    frames = np.asarray(shape["source_acquisition_frame_index"][:], dtype=np.int64)
    del shape["source_acquisition_frame_index"]
    shape.create_array(
        "source_acquisition_frame_index",
        data=frames,
        chunks=(2_048,),
        overwrite=True,
    )
    tail_parent = source_root["analysis"].require_group("tail_kinematics_runs")
    tail_parent.attrs.update(
        {"latest": "tail_existing", "latest_complete": "tail_existing"}
    )

    result = mod.materialize_tail_kinematics(
        source,
        scratch_root=scratch,
        shape_run="shape_001",
        run_name="tail_candidate",
        block_rows=3_000,
        execution_backend="serial",
        num_workers=1,
        copy_backend="python",
        apply=True,
        keep_scratch=True,
        check_capacity=False,
        storage_profile=PUBLISHED_HTTP_V1,
    )

    assert result["status"] == "complete"
    assert result["plan"]["byte_planner_candidate"] is True
    root = zarr.open_group(str(source), mode="r", use_consolidated=False)
    parent = root["analysis/tail_kinematics_runs"]
    assert parent.attrs["latest"] == "tail_existing"
    assert parent.attrs["latest_complete"] == "tail_existing"
    run = parent["tail_candidate"]
    assert run.attrs["palette_run_completion_status"] == "complete"
    assert run.attrs["stage_selector_eligible"] is False
    assert run.attrs["analysis_storage_profile_id"] == "published_http_v1"
    assert tail_mod.validate_tail_kinematics_storage_receipt(run) == ()
    consolidated = zarr.open_group(str(source), mode="r", use_consolidated=True)
    consolidated_run = consolidated["analysis/tail_kinematics_runs/tail_candidate"]
    assert consolidated_run.attrs["stage_selector_eligible"] is False
    assert np.asarray(consolidated_run["tail_angle_rad"][:]).shape == (20_000, 10)


def test_typed_candidate_executes_all_phases_and_supports_owned_tombstone(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _patch_provenance(monkeypatch)
    source = tmp_path / "source.zarr"
    _build_source_zarr(source, row_count=9)
    root = zarr.open_group(str(source), mode="a", use_consolidated=False)
    shape = root["analysis/subject_shape_runs/shape_001"]
    frames = np.asarray(shape["source_acquisition_frame_index"][:], dtype=np.int64)
    del shape["source_acquisition_frame_index"]
    shape.create_array(
        "source_acquisition_frame_index",
        data=frames,
        chunks=(9,),
        overwrite=True,
    )

    source_result = mod.materialize_tail_kinematics(
        source,
        scratch_root=tmp_path / "source-scratch",
        shape_run="shape_001",
        run_name="tail_source",
        copy_backend="python",
        apply=True,
        keep_scratch=False,
        check_capacity=False,
    )
    assert source_result["status"] == "complete"
    source_root = zarr.open_group(str(source), mode="r", use_consolidated=False)
    source_hashes = compute_tail_kinematics_logical_hashes(
        source_root["analysis/tail_kinematics_runs/tail_source"]
    )
    binding = {
        "schema_id": "palette.analysis_candidate_execution_binding",
        "schema_version": 1,
        "execution_id": "tail-unit",
        "request_payload_digest": "a" * 64,
        "candidate_run_path": "analysis/tail_kinematics_runs/tail_candidate",
    }

    def accept(_root, parent, candidate):
        assert parent.attrs["latest"] == "tail_source"
        assert candidate.attrs[mod.EXECUTION_BINDING_ATTR] == binding
        assert candidate.attrs["storage_candidate_profile_promoted"] is False
        return {"execution_binding": binding, "selector": "tail_source"}

    result = mod.materialize_tail_kinematics(
        source,
        scratch_root=tmp_path / "candidate-scratch",
        shape_run="shape_001",
        run_name="tail_candidate",
        block_rows=9,
        execution_backend="serial",
        num_workers=1,
        copy_backend="python",
        apply=True,
        keep_scratch=False,
        check_capacity=False,
        storage_profile=PUBLISHED_HTTP_V1,
        execution_binding=binding,
        expected_source_logical_hashes=source_hashes,
        publication_acceptance_validator=accept,
    )

    assert result["status"] == "complete"
    assert [phase["name"] for phase in result["runtime_telemetry"]["phases"]] == list(
        mod.TAIL_KINEMATICS_EXECUTION_PHASE_ORDER
    )
    assert result["local_direct_consolidated_array_count"] == 23
    assert result["published_direct_consolidated_array_count"] == 23
    assert (
        result["source_logical_manifest_sha256"]
        == result["published_logical_manifest_sha256"]
    )
    assert result["caller_acceptance"] == {
        "execution_binding": binding,
        "selector": "tail_source",
    }
    published = zarr.open_group(str(source), mode="r", use_consolidated=False)
    parent = published["analysis/tail_kinematics_runs"]
    assert parent.attrs["latest"] == "tail_source"
    assert parent.attrs["latest_complete"] == "tail_source"
    assert parent["tail_candidate"].attrs["stage_selector_eligible"] is False

    tombstone = mod.tombstone_tail_kinematics_execution_candidate(
        source,
        run_name="tail_candidate",
        expected_execution_binding=binding,
        failure_phase="driver_receipt_publication",
        error_type="RuntimeError",
        error_message="receipt write failed",
    )
    assert tombstone["tombstoned"] is True
    direct = zarr.open_group(str(source), mode="r", use_consolidated=False)
    consolidated = zarr.open_group(str(source), mode="r", use_consolidated=True)
    for view in (direct, consolidated):
        run = view["analysis/tail_kinematics_runs/tail_candidate"]
        assert run.attrs["palette_run_completion_status"] == "failed"
        assert run.attrs["stage_selector_eligible"] is False
        assert (
            run.attrs[mod.EXECUTION_FAILURE_TOMBSTONE_ATTR]["execution_binding"]
            == binding
        )


def test_materialize_tail_kinematics_process_workers_own_complete_shards(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _patch_provenance(monkeypatch)
    monkeypatch.setattr(
        tail_mod, "refined_subject_mask_metric_row_chunk", lambda _total_rows: 2
    )
    source = tmp_path / "source.zarr"
    scratch = tmp_path / "scratch"
    _build_source_zarr(source)

    result = mod.materialize_tail_kinematics(
        source,
        scratch_root=scratch,
        shape_run="shape_001",
        run_name="tail_process_shards",
        block_rows=3,
        output_shard_rows=7,
        execution_backend="process_shards",
        num_workers=2,
        copy_backend="python",
        apply=True,
        keep_scratch=True,
        check_capacity=False,
        stage_command="unit-test-process-shards",
    )

    summary = result["local_materialization"]
    assert summary["execution_backend"] == "process_shards"
    assert summary["worker_count_requested"] == 2
    assert summary["worker_count_effective"] == 2
    assert summary["effective_block_rows"] == 4
    assert summary["effective_output_shard_rows"] == 8
    assert summary["output_shard_count"] == 2
    assert summary["worker_task_count"] == 2
    assert summary["completed_block_count"] == 3
    assert summary["completed_worker_task_count"] == 2
    root = zarr.open_group(str(source), mode="r", use_consolidated=False)
    run = root["analysis/tail_kinematics_runs/tail_process_shards"]
    assert run.attrs["materialization_mode"] == "bounded_process_shards"
    assert (
        run.attrs["worker_chunk_alignment"]
        == "compute_blocks_align_to_output_row_chunks"
    )
    assert (
        run.attrs["worker_write_ownership"]
        == "one_complete_nonoverlapping_output_shard_per_task"
    )
    assert tuple(run["tail_angle_rad"].chunks) == (2, 10)
    assert (
        run.attrs["worker_compute_blocking"]
        == "bounded_subblocks_within_owned_output_shard"
    )
    assert run.attrs["requested_output_shard_rows"] == 7
    assert run.attrs["effective_output_shard_rows"] == 8
    assert tuple(run["tail_angle_rad"].shards) == (8, 10)
    assert tuple(run["source_acquisition_frame_index"].shards) == (8,)
    np.testing.assert_allclose(np.asarray(run["tail_angle_deg"][:]), 0.0, atol=1e-5)


def test_process_shard_slice_validation_rejects_partial_shared_shards() -> None:
    with pytest.raises(ValueError, match="output shard"):
        tail_mod._validate_process_shard_slices(
            (slice(0, 2), slice(2, 6), slice(6, 9)),
            row_count=9,
            output_row_chunk=2,
            output_shard_rows=4,
        )


def test_tail_publish_retains_owned_tombstone_and_parent_fails_closed_after_rename_failure(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _patch_provenance(monkeypatch)
    monkeypatch.setattr(
        tail_mod, "refined_subject_mask_metric_row_chunk", lambda _rows: 2
    )
    source = tmp_path / "source.zarr"
    scratch = tmp_path / "scratch"
    _build_source_zarr(source)
    root = zarr.open_group(str(source), mode="a", use_consolidated=False)
    parent = root["analysis"].create_group("tail_kinematics_runs")
    parent.attrs.update(
        {
            "latest": "tail_existing",
            "latest_complete": "tail_existing",
            "custom_parent_attr": "preserve-me",
        }
    )
    existing = parent.create_group("tail_existing")
    existing.attrs["palette_run_completion_status"] = "complete"
    parent_attrs_before = dict(parent.attrs)
    target = source / "analysis" / "tail_kinematics_runs" / "tail_rollback"

    real_validate = mod._validate_tail_run

    def fail_target_validation(path: Path, *, row_count: int, sample_count: int):
        if Path(path).resolve() == target.resolve():
            return {
                "valid": False,
                "errors": ["injected post-rename validation failure"],
                "row_count": row_count,
                "sample_count": sample_count,
            }
        return real_validate(path, row_count=row_count, sample_count=sample_count)

    monkeypatch.setattr(mod, "_validate_tail_run", fail_target_validation)

    with pytest.raises(RuntimeError, match="pre-pointer validation"):
        mod.materialize_tail_kinematics(
            source,
            scratch_root=scratch,
            shape_run="shape_001",
            run_name="tail_rollback",
            block_rows=3,
            output_shard_rows=7,
            execution_backend="process_shards",
            num_workers=2,
            copy_backend="python",
            apply=True,
            keep_scratch=True,
            check_capacity=False,
            stage_command="unit-test-publish-rollback",
        )

    assert target.exists()
    root = zarr.open_group(str(source), mode="r", use_consolidated=False)
    parent = root["analysis"]["tail_kinematics_runs"]
    assert dict(parent.attrs) == parent_attrs_before
    assert "tail_existing" in parent
    failed = parent["tail_rollback"]
    owner = failed.attrs[tail_mod.TAIL_PUBLICATION_OWNER_ATTR]
    tombstone = failed.attrs[atomic_mod.ATOMIC_PUBLICATION_TOMBSTONE_ATTR]
    assert failed.attrs["stage_selector_eligible"] is False
    assert failed.attrs["palette_run_completion_status"] == "failed"
    assert "palette_run_completed_at_utc" not in failed.attrs
    assert tombstone["schema_id"] == "palette.atomic_publication_tombstone"
    assert tombstone["publication_owner_attr"] == (tail_mod.TAIL_PUBLICATION_OWNER_ATTR)
    assert tombstone["publication_owner_uuid"] == owner
    assert tombstone["public_path_retained"] is True
    assert tombstone["selector_eligible"] is False
    assert tombstone["retry_policy"] == "new_immutable_run_name_required"
    assert parent.attrs["latest"] == "tail_existing"
    assert parent.attrs["latest_complete"] == "tail_existing"
    assert scratch.is_dir()


def test_tail_failed_exact_receipt_rollback_never_restores_precopy_snapshot(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.zarr"
    root = zarr.open_group(str(source), mode="w", use_consolidated=False)
    parent = root.require_group("analysis").require_group("tail_kinematics_runs")
    parent.attrs.update({"latest": "tail_a", "latest_complete": "tail_a"})

    local = tmp_path / "local-tail.zarr"
    local_run = zarr.open_group(str(local), mode="w", use_consolidated=False)
    candidate_owner = str(uuid4())
    local_run.attrs.update(
        {
            mod.tail_publication_mod.TAIL_PUBLICATION_OWNER_ATTR: candidate_owner,
            "palette_run_completion_status": "running",
            "stage_selector_eligible": False,
        }
    )
    target = source / "analysis" / "tail_kinematics_runs" / "tail_c"
    plan = SimpleNamespace(
        source_zarr=source,
        staged_zarr=local,
        local_run_path=local,
        target_run_path=target,
        run_name="tail_c",
        row_count=1,
        tail_angle_sample_count=1,
        storage_profile_id=None,
    )
    receipt = object()
    intervening_seen: list[bool] = []
    rollback_attempted: list[bool] = []
    intervening_owner = str(uuid4())

    def validate(path: Path, **_kwargs):
        group = zarr.open_group(str(path), mode="r", use_consolidated=False)
        complete = group.attrs.get("palette_run_completion_status") == "complete"
        return {
            "valid": not complete,
            "errors": ["injected failure after deferred receipt"] if complete else [],
            "row_count": 1,
            "sample_count": 1,
        }

    def mark_complete(run_group, **_kwargs) -> None:
        run_group.attrs["palette_run_completion_status"] = "complete"

    def defer_with_intervening_publication(
        _root,
        live_parent,
        _run_group,
        *,
        run_name,
        expected_publication_owner_uuid,
    ):
        assert run_name == "tail_c"
        assert expected_publication_owner_uuid == candidate_owner
        live_parent.attrs.update(
            {
                "latest": "tail_b",
                "latest_complete": "tail_b",
                mod.tail_publication_mod.TAIL_PUBLICATION_POLICY_ATTR: (
                    mod.tail_publication_mod.TAIL_PUBLICATION_POLICY
                ),
                mod.tail_publication_mod.TAIL_PUBLICATION_GENERATION_ATTR: 1,
                mod.tail_publication_mod.TAIL_PARENT_PUBLICATION_LEASE_ATTR: {
                    "owner_uuid": intervening_owner,
                    "base_generation": 0,
                    "next_generation": 1,
                },
            }
        )
        intervening_seen.append(True)
        live_parent.attrs.update(
            {
                "latest": "tail_c",
                "latest_complete": "tail_c",
                mod.tail_publication_mod.TAIL_PUBLICATION_POLICY_ATTR: (
                    mod.tail_publication_mod.TAIL_PUBLICATION_POLICY
                ),
                mod.tail_publication_mod.TAIL_PUBLICATION_GENERATION_ATTR: 2,
                mod.tail_publication_mod.TAIL_PARENT_PUBLICATION_LEASE_ATTR: {
                    "schema_id": "palette.tail_publication_lease",
                    "schema_version": 1,
                    "policy": mod.tail_publication_mod.TAIL_PUBLICATION_POLICY,
                    "run_path": "analysis/tail_kinematics_runs/tail_c",
                    "publication_owner": candidate_owner,
                    "owner_uuid": candidate_owner,
                    "base_generation": 1,
                    "next_generation": 2,
                },
            }
        )
        return receipt

    def fail_exact_rollback(value) -> None:
        assert value is receipt
        rollback_attempted.append(True)
        raise RuntimeError("injected exact tail receipt rollback failure")

    monkeypatch.setattr(mod, "_validate_tail_run", validate)
    monkeypatch.setattr(
        mod.tail_mod,
        "publish_tail_kinematics_coordinate_surfaces",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(mod, "mark_run_complete", mark_complete)
    monkeypatch.setattr(
        mod.tail_publication_mod,
        "defer_tail_coordinate_publication_activation",
        defer_with_intervening_publication,
    )
    monkeypatch.setattr(
        mod.tail_publication_mod,
        "rollback_deferred_tail_coordinate_publication_activation",
        fail_exact_rollback,
    )

    with pytest.raises(
        RuntimeError,
        match="rollback was incomplete.*exact tail receipt rollback failure",
    ):
        mod.publish_tail_kinematics_run(
            plan,
            staging_payload={},
            copy_backend="python",
        )

    assert intervening_seen == [True]
    assert rollback_attempted == [True]
    root = zarr.open_group(str(source), mode="r", use_consolidated=False)
    parent = root["analysis/tail_kinematics_runs"]
    failed = parent["tail_c"]
    assert failed.attrs["stage_selector_eligible"] is False
    assert failed.attrs["palette_run_completion_status"] == "failed"
    assert parent.attrs["latest"] == "tail_c"
    assert parent.attrs["latest_complete"] == "tail_c"
    assert parent.attrs[mod.tail_publication_mod.TAIL_PUBLICATION_GENERATION_ATTR] == 2
    assert (
        parent.attrs[mod.tail_publication_mod.TAIL_PARENT_PUBLICATION_LEASE_ATTR][
            "owner_uuid"
        ]
        == candidate_owner
    )


def test_materializer_refuses_to_replace_existing_authoritative_run(
    monkeypatch, tmp_path: Path
) -> None:
    _patch_provenance(monkeypatch)
    source = tmp_path / "source.zarr"
    _build_source_zarr(source)

    first = mod.materialize_tail_kinematics(
        source,
        scratch_root=tmp_path / "scratch-first",
        shape_run="shape_001",
        run_name="tail_001",
        copy_backend="python",
        apply=True,
        keep_scratch=False,
        check_capacity=False,
    )
    assert first["status"] == "complete"

    second_scratch = tmp_path / "scratch-second"
    with pytest.raises(FileExistsError, match="Refusing to replace"):
        mod.materialize_tail_kinematics(
            source,
            scratch_root=second_scratch,
            shape_run="shape_001",
            run_name="tail_001",
            copy_backend="python",
            apply=True,
            keep_scratch=False,
            check_capacity=False,
        )

    assert not second_scratch.exists()
    root = zarr.open_group(str(source), mode="r", use_consolidated=False)
    assert root["analysis"]["tail_kinematics_runs"].attrs["latest"] == "tail_001"
