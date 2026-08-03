from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pytest

from fisheye.analysis_workflows import (
    StageAvailability,
    WorkflowExecutionError,
    build_workflow_execution_plan,
    default_core_behavior_profile_path,
    load_analysis_workflow,
    plan_analysis_workflow,
)
from fisheye.utils.execute_analysis_workflow import execute_workflow_plan, main
from fisheye.analysis_workflows.dag import NodePlan
from fisheye.analysis_workflows.execution import (
    STAGE_COMMAND_BUILDERS,
    StageCommandContext,
)


def _status(
    stage_id: str, *, available: bool, run_name: str | None = None
) -> StageAvailability:
    return StageAvailability(
        stage_id=stage_id,
        available=available,
        run_name=run_name,
        reason="complete" if available else "missing",
    )


def _write_group(path: Path, attributes: dict[str, object] | None = None) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "zarr.json").write_text(
        json.dumps(
            {
                "zarr_format": 3,
                "node_type": "group",
                "attributes": dict(attributes or {}),
            }
        ),
        encoding="utf-8",
    )


def _command_context(
    tmp_path: Path,
    *,
    stage_id: str,
    dependencies: dict[str, str],
) -> StageCommandContext:
    return StageCommandContext(
        zarr_path=tmp_path / "recording_analysis.zarr",
        node=NodePlan(
            node_id=stage_id,
            kind="analysis",
            stage_id=stage_id,
            action="run",
            depends_on=tuple(dependencies),
            output_run_from=None,
            artifact_path=None,
            selected_run=None,
            reason="unit test",
            execution_policy="unit_test",
            temporal_policy={},
        ),
        output_run=f"{stage_id}_candidate",
        dependency_runs=dependencies,
        python_executable="/palette/python",
        num_workers=8,
    )


def test_extended_analysis_command_adapters_are_exact_and_dependency_bound(
    tmp_path: Path,
) -> None:
    tail = STAGE_COMMAND_BUILDERS["tail_posture_view"](
        _command_context(
            tmp_path,
            stage_id="tail_posture_view",
            dependencies={
                "subject_shape": "shape_v4",
                "tail_kinematics": "tail_v2",
            },
        )
    )
    assert tail[:4] == (
        "/palette/python",
        "-m",
        "fisheye.analysis.tail_posture_view_runs",
        str(tmp_path / "recording_analysis.zarr"),
    )
    assert tail[tail.index("--subject-shape-run") + 1] == "shape_v4"
    assert tail[tail.index("--source-tail-kinematics-run") + 1] == "tail_v2"
    assert tail[tail.index("--run-name") + 1] == "tail_posture_view_candidate"

    classification = STAGE_COMMAND_BUILDERS["bout_classification"](
        _command_context(
            tmp_path,
            stage_id="bout_classification",
            dependencies={
                "tail_posture_view": "posture_v3",
                "track_kinematics": "track_v1",
                "swim_bouts": "bouts_v8",
            },
        )
    )
    assert classification[:3] == (
        "/palette/python",
        "-m",
        "fisheye.analysis.megabouts_classifier",
    )
    assert classification[classification.index("--tail-posture-view-run") + 1] == (
        "posture_v3"
    )
    assert classification[classification.index("--track-kinematics-run") + 1] == (
        "track_v1"
    )
    assert classification[classification.index("--swim-bout-run") + 1] == "bouts_v8"

    stimulus = STAGE_COMMAND_BUILDERS["stimulus_response"](
        _command_context(
            tmp_path,
            stage_id="stimulus_response",
            dependencies={
                "stimulus": "stimulus_v2",
                "track_kinematics": "track_v1",
                "swim_bouts": "bouts_v8",
            },
        )
    )
    assert stimulus[:3] == (
        "/palette/python",
        "-m",
        "fisheye.analysis_workflows.materializers.stimulus_response",
    )
    assert stimulus[stimulus.index("--layout") + 1] == "compact_tabular_v3"
    assert stimulus[stimulus.index("--stimulus-run") + 1] == "stimulus_v2"
    assert stimulus[stimulus.index("--track-kinematics-run") + 1] == "track_v1"
    assert stimulus[stimulus.index("--bout-run") + 1] == "bouts_v8"


def _analysis_execution_plan(tmp_path: Path):
    workflow = load_analysis_workflow(default_core_behavior_profile_path())
    availability = {
        "refined_keypoints": _status(
            "refined_keypoints",
            available=True,
            run_name="refined_kp_a",
        ),
        "refined_subject_masks": _status(
            "refined_subject_masks",
            available=True,
            run_name="refined_masks_a",
        ),
        "tracks": _status(
            "tracks",
            available=True,
            run_name="tracking_a",
        ),
        "track_kinematics": _status(
            "track_kinematics",
            available=True,
            run_name="track_a",
        ),
        "swim_bouts": _status("swim_bouts", available=False),
        "track_kinematics_visualization": _status(
            "track_kinematics_visualization", available=False
        ),
        "bout_kinematics": _status("bout_kinematics", available=False),
        "eye_angles": _status("eye_angles", available=False),
        "subject_shape": _status("subject_shape", available=False),
    }
    plan = plan_analysis_workflow(
        workflow,
        availability,
        targets=("bout_kinematics", "eye_angles", "subject_shape"),
    )
    execution = build_workflow_execution_plan(
        workflow,
        plan,
        zarr_path=tmp_path / "recording_analysis.zarr",
        execution_id="canary_20260713_01",
        num_workers=8,
        python_executable="/palette/python",
    )
    return workflow, execution


def test_execution_plan_renders_exact_dependency_runs_and_parallel_backends(
    tmp_path: Path,
) -> None:
    _workflow, execution = _analysis_execution_plan(tmp_path)

    assert [command.node_id for command in execution.commands] == [
        "swim_bouts",
        "track_kinematics_visualization",
        "subject_shape",
        "eye_angles",
        "bout_kinematics",
    ]
    commands = {command.node_id: command for command in execution.commands}
    swim = commands["swim_bouts"].argv
    assert swim[:4] == (
        "/palette/python",
        "-m",
        "fisheye.analysis_workflows.materializers.swim_bouts",
        str(tmp_path / "recording_analysis.zarr"),
    )
    assert swim[swim.index("--track-kinematics-run") + 1] == "track_a"
    assert swim[swim.index("--run-name") + 1] == ("swim_bouts_canary_20260713_01")

    visualization = commands["track_kinematics_visualization"]
    view = visualization.argv
    assert view[:4] == (
        "/palette/python",
        "-m",
        "fisheye.analysis.plot_track_kinematics",
        str(tmp_path / "recording_analysis.zarr"),
    )
    assert view[view.index("--track-kinematics-run") + 1] == "track_a"
    assert view[view.index("--swim-bout-run") + 1] == ("swim_bouts_canary_20260713_01")
    assert view[view.index("--speed-level") + 1] == "exponential"
    assert visualization.output_run == "track_a"
    assert execution.output_runs["track_kinematics_visualization"] == "track_a"

    bout = commands["bout_kinematics"].argv
    assert bout[:4] == (
        "/palette/python",
        "-m",
        "fisheye.analysis_workflows.materializers.bout_kinematics",
        str(tmp_path / "recording_analysis.zarr"),
    )
    assert "--compute" in bout
    assert "--apply" in bout
    assert bout[bout.index("--output-shard-rows") + 1] == "262144"
    assert "--" in bout
    assert bout[bout.index("--swim-bout-run") + 1] == ("swim_bouts_canary_20260713_01")
    assert bout[bout.index("--track-kinematics-run") + 1] == "track_a"
    assert "--include-eye-gaze" in bout
    assert bout[bout.index("--eye-angle-run") + 1] == ("eye_angles_canary_20260713_01")
    assert bout[bout.index("--eye-angle-family") + 1] == "gaze"

    eyes = commands["eye_angles"].argv
    assert eyes[:4] == (
        "/palette/python",
        "-m",
        "fisheye.analysis_workflows.materializers.eye_angles",
        str(tmp_path / "recording_analysis.zarr"),
    )
    assert eyes[eyes.index("--subject-shape-run") + 1] == (
        "subject_shape_canary_20260713_01"
    )
    assert "--keypoint-run" not in eyes
    assert commands["eye_angles"].dependency_runs == {
        "subject_shape": "subject_shape_canary_20260713_01"
    }
    assert eyes[eyes.index("--execution-backend") + 1] == "dask_worker_chunks"
    assert eyes[eyes.index("--num-workers") + 1] == "8"
    assert eyes[eyes.index("--angle-chunk-rows") + 1] == "4096"
    assert eyes[eyes.index("--angle-chunk-columns") + 1] == "16"
    assert eyes[eyes.index("--output-shard-rows") + 1] == "131072"
    assert eyes[eyes.index("--angle-shard-columns") + 1] == "32"
    assert eyes[eyes.index("--shard-workers") + 1] == "8"
    assert eyes[eyes.index("--native-threads") + 1] == "1"
    assert "--apply" in eyes

    shape = commands["subject_shape"].argv
    assert shape[:4] == (
        "/palette/python",
        "-m",
        "fisheye.analysis_workflows.materializers.subject_shape",
        str(tmp_path / "recording_analysis.zarr"),
    )
    assert shape[shape.index("--refined-run") + 1] == "refined_masks_a"
    assert shape[shape.index("--execution-backend") + 1] == "dask_worker_chunks"
    assert shape[shape.index("--num-workers") + 1] == "8"
    assert shape[shape.index("--block-rows") + 1] == "1024"
    assert shape[shape.index("--output-shard-rows") + 1] == "131072"
    assert shape[shape.index("--native-threads") + 1] == "1"
    assert "--apply" in shape


def test_output_run_override_is_used_by_downstream_commands(tmp_path: Path) -> None:
    workflow = load_analysis_workflow(default_core_behavior_profile_path())
    availability = {
        "refined_keypoints": _status(
            "refined_keypoints", available=True, run_name="refined_kp_a"
        ),
        "track_kinematics": _status(
            "track_kinematics", available=True, run_name="track_a"
        ),
        "swim_bouts": _status("swim_bouts", available=False),
        "eye_angles": _status("eye_angles", available=True, run_name="eye_angles_a"),
        "bout_kinematics": _status("bout_kinematics", available=False),
    }
    plan = plan_analysis_workflow(
        workflow,
        availability,
        targets=("bout_kinematics",),
    )

    execution = build_workflow_execution_plan(
        workflow,
        plan,
        zarr_path=tmp_path,
        execution_id="run_a",
        num_workers=1,
        output_run_overrides={"swim_bouts": "custom_bouts"},
        python_executable="python",
    )

    bout_command = execution.commands[-1].argv
    assert bout_command[bout_command.index("--swim-bout-run") + 1] == "custom_bouts"
    assert bout_command[bout_command.index("--eye-angle-run") + 1] == "eye_angles_a"


def test_visualization_refuses_independent_output_run_override(tmp_path: Path) -> None:
    workflow = load_analysis_workflow(default_core_behavior_profile_path())
    availability = {
        "refined_keypoints": _status(
            "refined_keypoints", available=True, run_name="refined_kp_a"
        ),
        "track_kinematics": _status(
            "track_kinematics", available=True, run_name="track_a"
        ),
        "swim_bouts": _status("swim_bouts", available=True, run_name="swim_a"),
        "track_kinematics_visualization": _status(
            "track_kinematics_visualization", available=False
        ),
    }
    plan = plan_analysis_workflow(
        workflow,
        availability,
        targets=("track_kinematics_visualization",),
    )

    with pytest.raises(WorkflowExecutionError, match="inherits its output run"):
        build_workflow_execution_plan(
            workflow,
            plan,
            zarr_path=tmp_path,
            execution_id="run_a",
            num_workers=1,
            output_run_overrides={
                "track_kinematics_visualization": "independent_view_run"
            },
            python_executable="python",
        )


def test_execution_refuses_export_target_without_materializer(tmp_path: Path) -> None:
    workflow = load_analysis_workflow(default_core_behavior_profile_path())
    availability = {
        "refined_keypoints": _status(
            "refined_keypoints", available=True, run_name="refined_kp_a"
        ),
        "track_kinematics": _status(
            "track_kinematics", available=True, run_name="track_a"
        ),
    }
    plan = plan_analysis_workflow(
        workflow,
        availability,
        targets=("kinematics_samples",),
    )

    with pytest.raises(WorkflowExecutionError, match="adapter that is not implemented"):
        build_workflow_execution_plan(
            workflow,
            plan,
            zarr_path=tmp_path,
            execution_id="run_a",
            num_workers=1,
            python_executable="python",
        )


def test_execution_renders_staged_tail_materializer(tmp_path: Path) -> None:
    workflow = load_analysis_workflow(default_core_behavior_profile_path())
    availability = {
        "refined_subject_masks": _status(
            "refined_subject_masks", available=True, run_name="refined_masks_a"
        ),
        "subject_shape": _status(
            "subject_shape", available=True, run_name="subject_shape_a"
        ),
        "tail_kinematics": _status("tail_kinematics", available=False),
    }
    plan = plan_analysis_workflow(workflow, availability, targets=("tail_kinematics",))

    execution = build_workflow_execution_plan(
        workflow,
        plan,
        zarr_path=tmp_path,
        execution_id="run_a",
        num_workers=6,
        python_executable="python",
    )

    command = execution.commands[0]
    assert command.stage_id == "tail_kinematics"
    assert command.dependency_runs["subject_shape"] == "subject_shape_a"
    assert "fisheye.analysis_workflows.materializers.tail_kinematics" in command.argv
    assert "--execution-backend" in command.argv
    assert "process_shards" in command.argv
    assert command.argv[command.argv.index("--block-rows") + 1] == "16384"
    assert command.argv[command.argv.index("--output-shard-rows") + 1] == "262144"
    assert command.argv[command.argv.index("--num-workers") + 1] == "6"
    assert command.argv[command.argv.index("--shape-run") + 1] == "subject_shape_a"


def test_execution_renders_staged_track_kinematics_materializer(tmp_path: Path) -> None:
    workflow = load_analysis_workflow(default_core_behavior_profile_path())
    availability = {
        "tracks": _status("tracks", available=True, run_name="tracking_a"),
        "refined_keypoints": _status(
            "refined_keypoints", available=True, run_name="refined_kp_a"
        ),
        "track_kinematics": _status("track_kinematics", available=False),
    }
    plan = plan_analysis_workflow(workflow, availability, targets=("track_kinematics",))

    execution = build_workflow_execution_plan(
        workflow,
        plan,
        zarr_path=tmp_path,
        execution_id="run_a",
        num_workers=5,
        python_executable="python",
    )

    command = execution.commands[0]
    assert "fisheye.analysis_workflows.materializers.track_kinematics" in command.argv
    assert (
        command.argv[command.argv.index("--keypoint-run") + 1] == "refined/refined_kp_a"
    )
    assert command.argv[command.argv.index("--output-shard-rows") + 1] == "262144"
    assert command.argv[command.argv.index("--shard-workers") + 1] == "5"
    assert "--apply" in command.argv
    assert "--" in command.argv


def test_dry_run_writes_report_without_creating_stage_outputs(tmp_path: Path) -> None:
    workflow, execution = _analysis_execution_plan(tmp_path)
    zarr_path = tmp_path / "recording_analysis.zarr"
    _write_group(zarr_path)
    report_path = tmp_path / "reports" / "dry_run.json"

    payload = execute_workflow_plan(
        zarr_path,
        execution,
        workflow_payload=workflow.to_dict(),
        apply=False,
        report_path=report_path,
    )

    assert payload["status"] == "planned"
    assert report_path.is_file()
    assert not (zarr_path / "analysis").exists()
    assert {result["status"] for result in payload["node_results"]} == {
        "reused",
        "planned",
    }


def test_apply_is_rejected_outside_lsf(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workflow, execution = _analysis_execution_plan(tmp_path)
    zarr_path = tmp_path / "recording_analysis.zarr"
    _write_group(zarr_path)
    monkeypatch.delenv("LSB_JOBID", raising=False)

    with pytest.raises(WorkflowExecutionError, match="only inside an LSF allocation"):
        execute_workflow_plan(
            zarr_path,
            execution,
            workflow_payload=workflow.to_dict(),
            apply=True,
            report_path=tmp_path / "apply.json",
        )


def test_apply_verifies_completed_run_before_reporting_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workflow = load_analysis_workflow(default_core_behavior_profile_path())
    availability = {
        "refined_keypoints": _status(
            "refined_keypoints", available=True, run_name="refined_kp_a"
        ),
        "track_kinematics": _status(
            "track_kinematics", available=True, run_name="track_a"
        ),
        "swim_bouts": _status("swim_bouts", available=False),
    }
    plan = plan_analysis_workflow(workflow, availability, targets=("swim_bouts",))
    zarr_path = tmp_path / "recording_analysis.zarr"
    _write_group(zarr_path)
    execution = build_workflow_execution_plan(
        workflow,
        plan,
        zarr_path=zarr_path,
        execution_id="apply_canary",
        num_workers=2,
        python_executable="python",
    )

    def fake_run(argv, *, check, env):
        assert check is False
        assert env["MPLBACKEND"] == "Agg"
        assert env["PALETTE_DISABLE_REGISTRY_WRITES"] == "1"
        parent = zarr_path / "analysis" / "swim_bout_runs"
        _write_group(parent)
        _write_group(
            parent / "swim_bouts_apply_canary",
            {"palette_run_completion_status": "complete"},
        )
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setenv("LSB_JOBID", "12345")
    monkeypatch.setattr(
        "fisheye.utils.execute_analysis_workflow.subprocess.run",
        fake_run,
    )
    report_path = tmp_path / "execution.json"

    payload = execute_workflow_plan(
        zarr_path,
        execution,
        workflow_payload=workflow.to_dict(),
        apply=True,
        report_path=report_path,
    )

    assert payload["status"] == "complete"
    assert payload["registry_write_mode"] == "deferred_to_serial_finalizer"
    swim_result = next(
        result
        for result in payload["node_results"]
        if result["node_id"] == "swim_bouts"
    )
    assert swim_result["status"] == "complete"
    assert swim_result["verification"]["available"] is True
    assert json.loads(report_path.read_text(encoding="utf-8"))["status"] == "complete"


def test_cli_renders_sleepyfish_style_swim_bout_command(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    zarr_path = tmp_path / "sleepyfish_analysis.zarr"
    _write_group(zarr_path)
    refined_parent = zarr_path / "refined_keypoints_runs"
    _write_group(
        refined_parent,
        {"latest": "refined_kp_a", "latest_complete": "refined_kp_a"},
    )
    _write_group(
        refined_parent / "refined_kp_a",
        {
            "palette_run_completion_status": "complete",
            "stage_selector_eligible": True,
        },
    )
    tracks_parent = zarr_path / "tracking_runs"
    _write_group(
        tracks_parent,
        {"latest": "tracking_a", "latest_complete": "tracking_a"},
    )
    _write_group(
        tracks_parent / "tracking_a",
        {
            "palette_run_completion_status": "complete",
            "stage_selector_eligible": True,
        },
    )
    track_parent = zarr_path / "analysis" / "track_kinematics_runs" / "offline"
    _write_group(
        track_parent,
        {"latest": "track_a", "latest_complete": "track_a"},
    )
    _write_group(
        track_parent / "track_a",
        {
            "palette_run_completion_status": "complete",
            "stage_selector_eligible": True,
        },
    )

    exit_code = main(
        [
            str(zarr_path),
            "--target",
            "swim_bouts",
            "--execution-id",
            "sleepyfish_canary_01",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0, (captured.out, captured.err)
    assert "status=planned" in captured.out
    assert "fisheye.analysis_workflows.materializers.swim_bouts" in captured.out
    assert "--track-kinematics-run track_a" in captured.out
    assert captured.err == ""
