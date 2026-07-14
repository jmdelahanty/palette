from __future__ import annotations

import json
from pathlib import Path

import pytest

from fisheye.analysis_workflows import (
    ANALYSIS_WORKFLOW_SCHEMA_ID,
    ANALYSIS_WORKFLOW_SCHEMA_VERSION,
    AnalysisWorkflow,
    StageAvailability,
    TemporalPolicy,
    WorkflowNode,
    default_core_behavior_profile_path,
    discover_stage_availability,
    load_analysis_workflow,
    plan_analysis_workflow,
)


def _write_zarr_metadata(path: Path, attributes: dict[str, object] | None = None) -> None:
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


def test_core_behavior_profile_declares_portable_and_framewise_resolutions() -> None:
    workflow = load_analysis_workflow(default_core_behavior_profile_path())

    assert workflow.workflow_id == "core_behavior_v1"
    assert workflow.temporal_policy.kinematics_sample_rate_hz == 10.0
    assert workflow.temporal_policy.activity_spatial_bin_size_s == 5.0
    assert workflow.temporal_policy.eye_trace_resolution == "framewise"
    assert workflow.temporal_policy.tail_trace_resolution == "framewise"
    assert workflow.node_by_id["tail_kinematics"].runnable is False
    assert workflow.node_by_id["tail_traces"].depends_on == (
        "subject_shape",
        "tail_kinematics",
    )
    assert workflow.node_by_id["track_kinematics"].depends_on == (
        "refined_keypoints",
        "tracks",
    )
    visualization = workflow.node_by_id["track_kinematics_visualization"]
    assert visualization.kind == "visualization"
    assert visualization.depends_on == ("track_kinematics", "swim_bouts")
    assert visualization.output_run_from == "track_kinematics"
    assert "track_kinematics_visualization" in workflow.node_by_id[
        "bout_kinematics"
    ].depends_on


def test_temporal_policy_allows_numeric_overrides_but_not_trace_downsampling() -> None:
    policy = TemporalPolicy().with_overrides(
        kinematics_sample_rate_hz=20,
        activity_spatial_bin_size_s=2.5,
    )

    assert policy.product_policy("kinematics")["sample_rate_hz"] == 20.0
    assert policy.product_policy("activity_spatial")["bin_size_s"] == 2.5
    with pytest.raises(ValueError, match="eye_traces.resolution must remain"):
        TemporalPolicy(eye_trace_resolution="10_hz")
    with pytest.raises(ValueError, match="tail_traces.resolution must remain"):
        TemporalPolicy(tail_trace_resolution="boutwise")


def test_temporal_policy_rejects_unknown_configuration_fields() -> None:
    with pytest.raises(ValueError, match="unknown temporal_policy section"):
        TemporalPolicy.from_mapping({"kinematic": {"sample_rate_hz": 10}})
    with pytest.raises(ValueError, match="unknown temporal_policy.kinematics field"):
        TemporalPolicy.from_mapping({"kinematics": {"sample_hz": 10}})


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("kinematics_sample_rate_hz", 0),
        ("kinematics_sample_rate_hz", float("nan")),
        ("activity_spatial_bin_size_s", -1),
        ("activity_spatial_bin_size_s", float("inf")),
    ),
)
def test_temporal_policy_rejects_non_positive_or_non_finite_values(
    field: str,
    value: float,
) -> None:
    with pytest.raises(ValueError, match="positive finite"):
        TemporalPolicy(**{field: value})


def test_targeted_plan_reuses_authority_and_schedules_only_dependency_closure() -> None:
    workflow = load_analysis_workflow(default_core_behavior_profile_path())
    availability = {
        "refined_keypoints": StageAvailability(
            stage_id="refined_keypoints",
            available=True,
            artifact_path="refined_keypoints_runs/rkp_a",
            run_name="rkp_a",
            reason="complete",
        ),
        "tracks": StageAvailability(
            stage_id="tracks",
            available=True,
            artifact_path="tracking_runs/tracks_a",
            run_name="tracks_a",
            reason="complete",
        ),
        "track_kinematics": StageAvailability(
            stage_id="track_kinematics",
            available=False,
            reason="missing",
        ),
    }

    plan = plan_analysis_workflow(
        workflow,
        availability,
        targets=("kinematics_samples",),
    )

    assert plan.ready is True
    assert plan.topological_order == (
        "refined_keypoints",
        "tracks",
        "track_kinematics",
        "kinematics_samples",
    )
    assert plan.execution_order == ("track_kinematics", "kinematics_samples")
    assert plan.node_by_id["refined_keypoints"].action == "reuse"
    assert plan.node_by_id["tracks"].action == "reuse"
    assert plan.node_by_id["kinematics_samples"].temporal_policy == {
        "resolution": "sampled",
        "sample_rate_hz": 10.0,
        "source_authority": "framewise_zarr",
    }


def test_tail_plan_blocks_when_safe_large_recording_backend_is_unavailable() -> None:
    workflow = load_analysis_workflow(default_core_behavior_profile_path())
    availability = {
        "refined_subject_masks": StageAvailability(
            stage_id="refined_subject_masks",
            available=True,
            run_name="rsm_a",
            reason="complete",
        ),
        "subject_shape": StageAvailability(
            stage_id="subject_shape",
            available=False,
            reason="missing",
        ),
        "tail_kinematics": StageAvailability(
            stage_id="tail_kinematics",
            available=False,
            reason="missing",
        ),
    }

    plan = plan_analysis_workflow(workflow, availability, targets=("tail_traces",))

    assert plan.ready is False
    assert plan.node_by_id["subject_shape"].action == "run"
    assert plan.node_by_id["tail_kinematics"].action == "blocked"
    assert plan.node_by_id["tail_kinematics"].execution_policy == (
        "chunk_aligned_backend_required"
    )
    assert plan.node_by_id["tail_traces"].action == "blocked"


def test_track_kinematics_plan_blocks_without_tracking_authority() -> None:
    workflow = load_analysis_workflow(default_core_behavior_profile_path())
    availability = {
        "refined_keypoints": StageAvailability(
            stage_id="refined_keypoints",
            available=True,
            run_name="rkp_a",
            reason="complete",
        ),
        "tracks": StageAvailability(
            stage_id="tracks",
            available=False,
            reason="persisted run parent is missing",
        ),
        "track_kinematics": StageAvailability(
            stage_id="track_kinematics",
            available=False,
            reason="persisted run parent is missing",
        ),
    }

    plan = plan_analysis_workflow(
        workflow,
        availability,
        targets=("track_kinematics",),
    )

    assert plan.ready is False
    assert plan.node_by_id["tracks"].action == "blocked"
    assert plan.node_by_id["track_kinematics"].action == "blocked"
    assert plan.node_by_id["track_kinematics"].reason == "blocked by tracks"


def test_workflow_rejects_dependency_cycles() -> None:
    with pytest.raises(ValueError, match="dependency cycle"):
        AnalysisWorkflow(
            schema_id=ANALYSIS_WORKFLOW_SCHEMA_ID,
            schema_version=ANALYSIS_WORKFLOW_SCHEMA_VERSION,
            workflow_id="cycle",
            description="cycle fixture",
            nodes=(
                WorkflowNode(node_id="first", kind="export", depends_on=("second",)),
                WorkflowNode(node_id="second", kind="export", depends_on=("first",)),
            ),
            targets=("first",),
        )


def test_availability_resolver_uses_latest_complete_metadata_pointer(tmp_path: Path) -> None:
    parent = tmp_path / "analysis" / "track_kinematics_runs" / "offline"
    _write_zarr_metadata(parent, {"latest_complete": "track_a"})
    _write_zarr_metadata(
        parent / "track_a",
        {"palette_run_completion_status": "complete"},
    )

    status = discover_stage_availability(tmp_path, "track_kinematics")

    assert status.available is True
    assert status.run_name == "track_a"
    assert status.artifact_path == "analysis/track_kinematics_runs/offline/track_a"
    assert status.completion_status == "complete"


def test_availability_resolver_discovers_tracking_authority(tmp_path: Path) -> None:
    parent = tmp_path / "tracking_runs"
    _write_zarr_metadata(parent, {"latest_complete": "tracking_a"})
    _write_zarr_metadata(
        parent / "tracking_a",
        {"palette_run_completion_status": "complete"},
    )

    status = discover_stage_availability(tmp_path, "tracks")

    assert status.available is True
    assert status.run_name == "tracking_a"
    assert status.artifact_path == "tracking_runs/tracking_a"


def test_visualization_availability_is_tied_to_selected_track_run(
    tmp_path: Path,
) -> None:
    parent = tmp_path / "analysis" / "track_kinematics_runs" / "offline"
    _write_zarr_metadata(parent, {"latest_complete": "track_a"})
    _write_zarr_metadata(
        parent / "track_a",
        {"palette_run_completion_status": "complete"},
    )

    missing = discover_stage_availability(
        tmp_path,
        "track_kinematics_visualization",
        requested_run="track_a",
        dependency_runs={
            "track_kinematics": "track_a",
            "swim_bouts": "swim_a",
        },
    )

    assert missing.available is False
    assert missing.run_name == "track_a"
    assert "interactive track-kinematics contract is missing" in missing.reason

    artifact = (
        parent
        / "track_a"
        / "visualizations"
        / "track_kinematics_summary_track_0_interactive"
    )
    _write_zarr_metadata(
        artifact,
        {
            "renderer": "palette-track-kinematics-summary-v1",
            "source_runs": {"track_kinematics": "offline/track_a"},
            "parameters": {"swim_bout_run": "swim_a"},
        },
    )
    _write_zarr_metadata(artifact / "spec_json")

    available = discover_stage_availability(
        tmp_path,
        "track_kinematics_visualization",
        requested_run="track_a",
        dependency_runs={
            "track_kinematics": "track_a",
            "swim_bouts": "swim_a",
        },
    )

    assert available.available is True
    assert available.run_name == "track_a"
    assert available.artifact_path == (
        "analysis/track_kinematics_runs/offline/track_a/visualizations/"
        "track_kinematics_summary_track_0_interactive"
    )

    stale = discover_stage_availability(
        tmp_path,
        "track_kinematics_visualization",
        requested_run="track_a",
        dependency_runs={
            "track_kinematics": "track_a",
            "swim_bouts": "swim_b",
        },
    )

    assert stale.available is False
    assert "swim-bout lineage does not match" in stale.reason


def test_availability_resolver_requires_pointer_or_explicit_run(tmp_path: Path) -> None:
    parent = tmp_path / "analysis" / "swim_bout_runs"
    _write_zarr_metadata(parent)
    _write_zarr_metadata(parent / "bout_a", {"status": "complete"})

    unresolved = discover_stage_availability(tmp_path, "swim_bouts")
    explicit = discover_stage_availability(
        tmp_path,
        "swim_bouts",
        requested_run="bout_a",
    )

    assert unresolved.available is False
    assert "select an explicit run" in unresolved.reason
    assert explicit.available is True
    assert explicit.run_name == "bout_a"


def test_availability_resolver_does_not_reuse_incomplete_run(tmp_path: Path) -> None:
    parent = tmp_path / "analysis" / "eye_angle_runs"
    _write_zarr_metadata(parent, {"latest": "eye_a"})
    _write_zarr_metadata(parent / "eye_a", {"run_status": "running"})

    status = discover_stage_availability(tmp_path, "eye_angles")

    assert status.available is False
    assert status.completion_status == "running"
    assert "not complete" in status.reason


def test_availability_resolver_fails_closed_for_unmarked_strict_parent(
    tmp_path: Path,
) -> None:
    parent = tmp_path / "analysis" / "subject_shape_runs"
    _write_zarr_metadata(
        parent,
        {"palette_completion_epoch": 2, "latest_complete": "shape_a"},
    )
    _write_zarr_metadata(parent / "shape_a")

    status = discover_stage_availability(tmp_path, "subject_shape")

    assert status.available is False
    assert status.run_name == "shape_a"
    assert "required complete marker" in status.reason
