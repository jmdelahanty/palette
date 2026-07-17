from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest
import zarr

from fisheye.analysis_workflows.materializers import eye_angles as mod


def _build_source(path: Path, *, rows: int = 4) -> None:
    root = zarr.open_group(str(path), mode="w", zarr_format=3)
    root.attrs["recording_id"] = "eye-angle-materializer-fixture"
    analysis = root.create_group("analysis")
    shape_parent = analysis.create_group("subject_shape_runs")
    shape_parent.attrs["latest"] = "shape_1"
    shape = shape_parent.create_group("shape_1")
    shape.attrs.update(
        {
            "schema_id": "analysis.subject_shape_runs",
            "schema_version": 3,
            "method": "subject_shape",
            "method_version": "subject_shape.v3",
            "palette_run_completion_status": "complete",
            "source_refined_subject_masks_run": "refined_masks_1",
            "source_keypoints_run": "kp_refined_1",
            "source_fingerprint": "shape-fixture-fingerprint",
        }
    )
    for component, center_y, angle in (
        ("eye_left", 1.0, 90.0),
        ("eye_right", -1.0, 90.0),
    ):
        group = shape.create_group(f"components/{component}")
        group.attrs.update(
            {
                "ellipse_method": "cv2.fitEllipse_component_contour_v1",
                "geometry_schema_id": "subject_shape.eye_ellipse.v1",
            }
        )
        ellipse = np.tile(
            np.asarray([1.0, center_y, 4.0, 2.0, angle], dtype=np.float32),
            (rows, 1),
        )
        group.create_array("ellipse_params", data=ellipse, chunks=(2, 5))
        group.create_array(
            "ellipse_success",
            data=np.ones(rows, dtype=bool),
            chunks=(2,),
        )
    pair = shape.create_group("relations/eye_pair")
    pair.create_array(
        "separation_px",
        data=np.full(rows, 2.0, dtype=np.float32),
        chunks=(2,),
    )

    refined_parent = root.create_group("refined_keypoints_runs")
    refined_parent.attrs["latest"] = "kp_refined_1"
    refined = refined_parent.create_group("kp_refined_1")
    refined.attrs.update(
        {
            "schema_id": "refined_keypoints",
            "schema_version": 4,
            "method": "manual_plus_model_refinement",
            "method_version": "refined_keypoints.v4",
            "palette_run_completion_status": "complete",
            "source_keypoints_run": "kp_raw_1",
            "source_lineage_hash": "refined-keypoint-lineage",
            "keypoint_labels": ["swim_bladder", "eye_left", "eye_right"],
        }
    )
    keypoints = np.tile(
        np.asarray(
            [[0.0, 0.0], [1.0, 1.0], [1.0, -1.0]],
            dtype=np.float32,
        ),
        (rows, 1, 1),
    )
    refined.create_array("keypoints_roi", data=keypoints, chunks=(2, 3, 2))
    refined.create_array(
        "heading",
        data=np.zeros(rows, dtype=np.float32),
        chunks=(2,),
    )
    refined.create_array(
        "refined_success",
        data=np.ones(rows, dtype=bool),
        chunks=(2,),
    )

    raw_parent = root.create_group("keypoints_runs")
    raw = raw_parent.create_group("kp_raw_1")
    raw.attrs.update(
        {
            "schema_id": "keypoints",
            "schema_version": 2,
            "method": "yolo_pose",
            "method_version": "detector.v2",
            "palette_run_completion_status": "complete",
            "lineage_hash": "raw-keypoint-lineage",
        }
    )
    raw.create_array(
        "frame_indices",
        data=np.arange(rows, dtype=np.int64),
        chunks=(2,),
    )


def test_plan_is_read_only_and_selects_only_resolved_geometry_and_keypoints(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.zarr"
    scratch = tmp_path / "scratch"
    _build_source(source)

    result = mod.materialize_eye_angles(
        source,
        scratch_root=scratch,
        subject_shape_run="shape_1",
        keypoint_run="kp_refined_1",
        run_name="eye_1",
        fps=100.0,
        apply=False,
    )

    assert result["status"] == "planned"
    assert result["mutates_archive"] is False
    assert not scratch.exists()
    plan = result["plan"]
    assert plan["row_count"] == 4
    assert plan["frame_count"] == 4
    assert plan["angle_chunk_rows"] == 4096
    assert plan["angle_chunk_columns"] == 16
    assert plan["output_shard_rows"] == 131072
    assert plan["angle_shard_columns"] == 32
    assert plan["fps_source"] == "cli_override"
    assert all("masks_roi" not in path for path in plan["selected_arrays"])
    assert plan["selected_arrays"] == sorted(
        [
            "analysis/subject_shape_runs/shape_1/components/eye_left/ellipse_params",
            "analysis/subject_shape_runs/shape_1/components/eye_left/ellipse_success",
            "analysis/subject_shape_runs/shape_1/components/eye_right/ellipse_params",
            "analysis/subject_shape_runs/shape_1/components/eye_right/ellipse_success",
            "analysis/subject_shape_runs/shape_1/relations/eye_pair/separation_px",
            "keypoints_runs/kp_raw_1/frame_indices",
            "refined_keypoints_runs/kp_refined_1/heading",
            "refined_keypoints_runs/kp_refined_1/keypoints_roi",
            "refined_keypoints_runs/kp_refined_1/refined_success",
        ]
    )
    root = zarr.open_group(str(source), mode="r", use_consolidated=False)
    assert "eye_angle_runs" not in root["analysis"]


@pytest.mark.skipif(shutil.which("rsync") is None, reason="rsync is unavailable")
def test_rsync_staging_preserves_the_planned_physical_revision(tmp_path: Path) -> None:
    source = tmp_path / "source.zarr"
    scratch = tmp_path / "scratch"
    _build_source(source)
    plan = mod.build_eye_angle_materialization_plan(
        source,
        scratch_root=scratch,
        subject_shape_run="shape_1",
        keypoint_run="kp_refined_1",
        run_name="eye_1",
        fps=100.0,
    )

    staging = mod.stage_eye_angle_sources(
        plan,
        copy_backend="rsync",
        check_capacity=False,
    )

    assert staging["status"] == "complete"
    assert staging["inventory"]["valid"] is True
    assert staging["inventory"]["mtime_mismatches"] == []
    assert staging["source_revision_audit"]["status"] == "current"


def test_materializer_stages_computes_shards_and_publishes_with_provenance(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.zarr"
    scratch = tmp_path / "scratch"
    _build_source(source)

    result = mod.materialize_eye_angles(
        source,
        scratch_root=scratch,
        subject_shape_run="shape_1",
        keypoint_run="kp_refined_1",
        run_name="eye_1",
        chunk_rows=2,
        angle_chunk_rows=2,
        angle_chunk_columns=4,
        output_shard_rows=3,
        angle_shard_columns=4,
        execution_backend="serial_driver",
        scheduler="single-threaded",
        num_workers=1,
        shard_workers=1,
        fps=100.0,
        smoothing_window=3,
        copy_backend="python",
        apply=True,
        keep_scratch=True,
        check_capacity=False,
        stage_command="unit-test-eye-materializer",
    )

    assert result["status"] == "complete"
    assert result["publish"]["pre_pointer_validation"]["valid"] is True
    assert result["publish"]["final_validation"]["valid"] is True
    assert result["publish"]["source_revision_audit"]["status"] == "current"

    root = zarr.open_group(str(source), mode="r", use_consolidated=False)
    parent = root["analysis/eye_angle_runs"]
    run = parent["eye_1"]
    assert parent.attrs["latest"] == "eye_1"
    assert parent.attrs["latest_complete"] == "eye_1"
    assert run.attrs["schema_version"] == 5
    assert run.attrs["eye_angle_output_schema"]["schema_version"] == 8
    assert run.attrs["eye_angle_algorithm_contract"]["schema_version"] == 1
    assert tuple(run["frame_angles"].chunks) == (2, 4)
    assert tuple(run["roi_angles"].chunks) == (2, 4)
    assert tuple(run["frame_angles"].shards) == (4, 4)
    assert tuple(run["roi_angles"].shards) == (4, 4)

    local = run.attrs["node_local_materialization"]
    assert local["authoritative_source_zarr"] == str(source.resolve())
    assert local["source_staging"]["node_local_staged_zarr"] == str(
        (scratch / "eye-inputs-and-output.zarr").resolve()
    )
    assert local["source_staging"]["source_revision_audit"]["status"] == "current"
    assert local["compute"]["writer"] == "fisheye.analysis.eye_angle_analysis"
    assert local["compute"]["angle_chunk_rows"] == 2
    assert local["compute"]["angle_chunk_columns"] == 4
    assert local["compute"]["angle_column_order_profile"] == (
        "semantic_bundles_v1"
    )
    assert local["compute"]["stage_command"] == "unit-test-eye-materializer"
    assert local["algorithm_contract"]["sha256"]
    assert local["output_contract"]["sha256"]
    assert local["sharding"]["exact_decoded_validation"] is True
    layouts = {
        item["path"]: item for item in local["sharding"]["angle_array_layouts"]
    }
    assert tuple(layouts["frame_angles"]["inner_chunks"]) == (2, 4)
    assert tuple(layouts["frame_angles"]["outer_shards"]) == (4, 4)

    provenance = run.attrs["provenance"]
    assert provenance["materialization"]["authoritative_source_zarr"] == str(
        source.resolve()
    )
    assert provenance["materialization"]["selected_arrays"]
    publication = run.attrs["cluster_output_staging"]
    assert publication["publisher_contract"] == {
        "schema_id": "palette.atomic_run_group_publisher",
        "schema_version": 1,
    }
    assert publication["physical_copy"]["verification"] == (
        "sha256_all_physical_files"
    )


def test_publication_rolls_back_when_source_revision_changes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.zarr"
    scratch = tmp_path / "scratch"
    _build_source(source)
    real_audit = mod.audit_eye_angle_source_revision
    call_count = 0

    def changing_audit(plan):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return real_audit(plan)
        return {
            "schema_id": mod.SOURCE_REVISION_AUDIT_SCHEMA_ID,
            "status": "changed",
            "errors": ["injected source revision change"],
        }

    monkeypatch.setattr(mod, "audit_eye_angle_source_revision", changing_audit)

    with pytest.raises(RuntimeError, match="inputs changed during materialization"):
        mod.materialize_eye_angles(
            source,
            scratch_root=scratch,
            subject_shape_run="shape_1",
            keypoint_run="kp_refined_1",
            run_name="eye_1",
            chunk_rows=2,
            output_shard_rows=3,
            execution_backend="serial_driver",
            scheduler="single-threaded",
            num_workers=1,
            shard_workers=1,
            fps=100.0,
            copy_backend="python",
            apply=True,
            check_capacity=False,
        )

    root = zarr.open_group(str(source), mode="r", use_consolidated=False)
    assert "eye_1" not in root["analysis/eye_angle_runs"]
