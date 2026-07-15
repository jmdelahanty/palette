from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import zarr

from fisheye.analysis import tail_kinematics_runs as tail_mod
from fisheye.analysis_workflows.materializers import tail_kinematics as mod


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
        "valid",
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

    row_index = shape.create_group("row_index")
    row_index.create_array(
        "frame_indices",
        data=np.arange(100, 100 + row_count, dtype=np.int32),
        chunks=(2,),
        overwrite=True,
    )
    row_index.create_array(
        "detection_indices",
        data=np.arange(row_count, dtype=np.int32),
        chunks=(2,),
        overwrite=True,
    )
    row_index.create_array(
        "source_refined_row_ids",
        data=np.arange(200, 200 + row_count, dtype=np.int64),
        chunks=(2,),
        overwrite=True,
    )
    row_index.create_array(
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


def test_materialization_plan_selects_only_required_subject_shape_surface(tmp_path: Path) -> None:
    source = tmp_path / "source.zarr"
    scratch = tmp_path / "scratch"
    _build_source_zarr(source)

    plan = mod.build_tail_kinematics_materialization_plan(
        source,
        scratch_root=scratch,
        shape_run="shape_001",
        run_name="tail_001",
        block_rows=3,
    )

    selected = {item.relative_path for item in plan.physical_files}
    assert any("tail_sample_xy" in path for path in selected)
    assert not any("unrelated_debug_surface" in path for path in selected)
    assert plan.row_count == 9
    assert plan.source_bytes > 0
    assert plan.estimated_output_bytes > 0
    assert len(plan.source_metadata_sha256) == 64
    assert plan.source_contract["schema_id"] == "analysis.subject_shape_runs"
    assert plan.source_contract["schema_version"] == 3
    assert not scratch.exists()


def test_materialize_tail_kinematics_stages_computes_and_atomically_publishes(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _patch_provenance(monkeypatch)
    monkeypatch.setattr(tail_mod, "refined_subject_mask_metric_row_chunk", lambda _total_rows: 2)
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
    staged_body = scratch / "source-subset.zarr/analysis/subject_shape_runs/shape_001/components/subject_body"
    assert (staged_body / "tail_sample_xy").is_dir()
    assert not (staged_body / "unrelated_debug_surface").exists()
    manifest = json.loads((scratch / "staging-manifest.json").read_text(encoding="utf-8"))
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
        run.attrs["cluster_output_staging"]["serialization_policy"]
        == "per_recording_advisory_file_lock"
    )
    assert (tmp_path / ".source.zarr.tail-kinematics-publish.lock").is_file()
    assert tuple(run["tail_angle_rad"].chunks) == (2, 10)
    assert tuple(run["tail_angle_rad"].shards) == (4, 10)
    assert run["frame_index"][:].tolist() == list(range(100, 109))


def test_materialize_tail_kinematics_process_workers_own_complete_shards(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _patch_provenance(monkeypatch)
    monkeypatch.setattr(tail_mod, "refined_subject_mask_metric_row_chunk", lambda _total_rows: 2)
    source = tmp_path / "source.zarr"
    scratch = tmp_path / "scratch"
    _build_source_zarr(source)

    result = mod.materialize_tail_kinematics(
        source,
        scratch_root=scratch,
        shape_run="shape_001",
        run_name="tail_process_shards",
        block_rows=3,
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
    assert summary["completed_block_count"] == 3
    root = zarr.open_group(str(source), mode="r", use_consolidated=False)
    run = root["analysis/tail_kinematics_runs/tail_process_shards"]
    assert run.attrs["materialization_mode"] == "bounded_process_shards"
    assert run.attrs["worker_chunk_alignment"] == "align_to_output_row_chunks_and_shards"
    assert (
        run.attrs["worker_write_ownership"]
        == "one_complete_nonoverlapping_output_shard_per_task"
    )
    assert tuple(run["tail_angle_rad"].chunks) == (2, 10)
    assert tuple(run["tail_angle_rad"].shards) == (4, 10)
    np.testing.assert_allclose(np.asarray(run["tail_angle_deg"][:]), 0.0, atol=1e-5)


def test_process_shard_slice_validation_rejects_partial_shared_shards() -> None:
    with pytest.raises(ValueError, match="output shard"):
        tail_mod._validate_process_shard_slices(
            (slice(0, 2), slice(2, 6), slice(6, 9)),
            row_count=9,
            output_row_chunk=2,
            output_shard_rows=4,
        )


def test_materializer_refuses_to_replace_existing_authoritative_run(monkeypatch, tmp_path: Path) -> None:
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
