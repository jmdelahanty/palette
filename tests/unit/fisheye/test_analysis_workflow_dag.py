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
from fisheye.utils.plan_analysis_workflow import build_availability


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
    assert workflow.node_by_id["tail_kinematics"].runnable is True
    assert workflow.node_by_id["tracks"].runnable is True
    assert workflow.node_by_id["tracks"].depends_on == ("refined_keypoints",)
    assert workflow.node_by_id["tail_traces"].depends_on == (
        "subject_shape",
        "tail_kinematics",
        "track_kinematics",
    )
    assert workflow.node_by_id["track_kinematics"].depends_on == (
        "refined_keypoints",
        "tracks",
    )
    assert workflow.node_by_id["eye_angles"].depends_on == ("subject_shape",)
    assert workflow.node_by_id["eye_angles"].execution_policy == (
        "exact_source_subset_node_local_compute_shard_publish"
    )
    visualization = workflow.node_by_id["track_kinematics_visualization"]
    assert visualization.kind == "visualization"
    assert visualization.depends_on == ("track_kinematics", "swim_bouts")
    assert visualization.output_run_from == "track_kinematics"
    assert workflow.node_by_id["bout_kinematics"].depends_on == (
        "track_kinematics",
        "swim_bouts",
        "track_kinematics_visualization",
        "eye_angles",
    )


def test_eye_plan_derives_keypoint_authority_only_through_subject_shape() -> None:
    workflow = load_analysis_workflow(default_core_behavior_profile_path())
    availability = {
        "refined_keypoints": StageAvailability(
            stage_id="refined_keypoints",
            available=True,
            run_name="refined_keypoints_a",
            reason="complete",
        ),
        "tracks": StageAvailability(
            stage_id="tracks",
            available=True,
            run_name="tracks_a",
            reason="complete",
        ),
        "track_kinematics": StageAvailability(
            stage_id="track_kinematics",
            available=True,
            run_name="track_a",
            reason="complete",
        ),
        "refined_subject_masks": StageAvailability(
            stage_id="refined_subject_masks",
            available=True,
            artifact_path="refined_subject_masks_runs/masks_a",
            run_name="masks_a",
            reason="complete canonical publication",
        ),
        "subject_shape": StageAvailability(
            stage_id="subject_shape",
            available=True,
            artifact_path="analysis/subject_shape_runs/shape_a",
            run_name="shape_a",
            reason="complete canonical publication",
        ),
        "eye_angles": StageAvailability(
            stage_id="eye_angles",
            available=False,
            reason="missing",
        ),
    }

    plan = plan_analysis_workflow(workflow, availability, targets=("eye_angles",))

    assert plan.ready is True
    assert plan.topological_order == (
        "refined_subject_masks",
        "subject_shape",
        "eye_angles",
    )
    assert "refined_keypoints" not in plan.node_by_id
    assert plan.node_by_id["eye_angles"].depends_on == ("subject_shape",)
    assert plan.execution_order == ("eye_angles",)


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


def test_tail_plan_uses_staged_process_shard_backend() -> None:
    workflow = load_analysis_workflow(default_core_behavior_profile_path())
    availability = {
        "refined_keypoints": StageAvailability(
            stage_id="refined_keypoints",
            available=True,
            run_name="refined_keypoints_a",
            reason="complete",
        ),
        "tracks": StageAvailability(
            stage_id="tracks",
            available=True,
            run_name="tracks_a",
            reason="complete",
        ),
        "track_kinematics": StageAvailability(
            stage_id="track_kinematics",
            available=False,
            reason="missing",
        ),
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

    assert plan.ready is True
    assert plan.node_by_id["subject_shape"].action == "run"
    assert plan.node_by_id["tail_kinematics"].action == "run"
    assert plan.node_by_id["tail_kinematics"].execution_policy == (
        "node_local_staged_process_shards"
    )
    assert plan.node_by_id["track_kinematics"].action == "run"
    assert plan.node_by_id["tail_traces"].action == "run"


def test_track_kinematics_plan_materializes_missing_tracking_authority() -> None:
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

    assert plan.ready is True
    assert plan.node_by_id["tracks"].action == "run"
    assert plan.node_by_id["track_kinematics"].action == "run"
    assert plan.execution_order == ("tracks", "track_kinematics")


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
    _write_zarr_metadata(
        parent,
        {"latest": "track_a", "latest_complete": "track_a"},
    )
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
    _write_zarr_metadata(
        parent,
        {"latest": "tracking_a", "latest_complete": "tracking_a"},
    )
    _write_zarr_metadata(
        parent / "tracking_a",
        {"palette_run_completion_status": "complete"},
    )

    status = discover_stage_availability(tmp_path, "tracks")

    assert status.available is True
    assert status.run_name == "tracking_a"
    assert status.artifact_path == "tracking_runs/tracking_a"


def _write_keypoint_crop_tracking_lineage(
    root: Path,
    *,
    tracking_crop: str,
) -> None:
    _write_zarr_metadata(root)
    _write_zarr_metadata(
        root / "keypoints_runs",
        {"latest": "canonical_a", "latest_complete": "canonical_a"},
    )
    _write_zarr_metadata(
        root / "keypoints_runs" / "canonical_a",
        {
            "palette_run_completion_status": "complete",
            "stage_selector_eligible": True,
            "source_crop_run": "crop_geometry_a",
            "keypoints_processed": 4,
        },
    )
    _write_zarr_metadata(
        root / "crop_runs" / "crop_geometry_a",
        {
            "status": "complete",
            "artifact_class": "geometry_only_analysis",
            "stage_selector_eligible": False,
            "run_manifest": {
                "payload": {
                    "source_refined_snapshot": {"run_id": "refined_a"}
                }
            },
        },
    )
    _write_zarr_metadata(
        root / "tracking_runs",
        {"latest": "tracking_a", "latest_complete": "tracking_a"},
    )
    _write_zarr_metadata(
        root / "tracking_runs" / "tracking_a",
        {
            "palette_run_completion_status": "complete",
            "stage_selector_eligible": True,
            "source_rowset_path": f"crop_runs/{tracking_crop}",
            "source_rowset_row_count": 4,
            "source_refined_run": "refined_a",
        },
    )


def test_tracking_availability_rejects_selected_keypoint_crop_mismatch(
    tmp_path: Path,
) -> None:
    _write_keypoint_crop_tracking_lineage(
        tmp_path,
        tracking_crop="crop_hybrid_old",
    )

    status = discover_stage_availability(
        tmp_path,
        "tracks",
        dependency_runs={"refined_keypoints": "canonical_a"},
    )

    assert status.available is False
    assert status.run_name == "tracking_a"
    assert "does not match the selected keypoint crop lineage" in status.reason
    assert "crop_geometry_a" in status.reason


def test_workflow_availability_passes_keypoint_lineage_to_tracking_gate(
    tmp_path: Path,
) -> None:
    _write_keypoint_crop_tracking_lineage(
        tmp_path,
        tracking_crop="crop_geometry_a",
    )
    workflow = load_analysis_workflow(default_core_behavior_profile_path())

    statuses = build_availability(workflow, tmp_path)

    assert statuses["refined_keypoints"].available is True
    assert statuses["tracks"].available is True
    assert statuses["tracks"].run_name == "tracking_a"
    assert "matches the selected keypoint crop lineage" in statuses["tracks"].reason


def test_keypoint_authority_resolver_accepts_clipped_canonical_passthrough(
    tmp_path: Path,
) -> None:
    _write_zarr_metadata(tmp_path)
    parent = tmp_path / "keypoints_runs"
    _write_zarr_metadata(
        parent,
        {"latest": "canonical_a", "latest_complete": "canonical_a"},
    )
    _write_zarr_metadata(
        parent / "canonical_a",
        {
            "palette_run_completion_status": "complete",
            "stage_selector_eligible": True,
        },
    )

    status = discover_stage_availability(tmp_path, "refined_keypoints")

    assert status.available is True
    assert status.run_name == "canonical_a"
    assert status.artifact_path == "keypoints_runs/canonical_a"
    assert "canonical keypoint passthrough" in status.reason


def test_keypoint_authority_resolver_prefers_active_refined_bundle_member(
    tmp_path: Path,
) -> None:
    _write_zarr_metadata(
        tmp_path,
        {
            "keypoint_bundle_authority_generation": 1,
            "keypoint_bundle_authority": {
                "schema_id": "palette.keypoint.bundle_authority",
                "schema_version": 1,
                "generation": 1,
                "members": {
                    "refined_keypoints": {
                        "run_path": "refined_keypoints_runs/refined_a"
                    }
                },
            },
        },
    )
    parent = tmp_path / "refined_keypoints_runs"
    _write_zarr_metadata(parent)
    _write_zarr_metadata(
        parent / "refined_a",
        {
            "palette_run_completion_status": "complete",
            "stage_selector_eligible": False,
        },
    )

    status = discover_stage_availability(tmp_path, "refined_keypoints")

    assert status.available is True
    assert status.run_name == "refined/refined_a"
    assert status.artifact_path == "refined_keypoints_runs/refined_a"
    assert "active keypoint-bundle" in status.reason


def test_subject_mask_resolver_uses_active_root_bundle_authority(
    tmp_path: Path,
) -> None:
    bundle_id = "bundle_a"
    _write_zarr_metadata(
        tmp_path,
        {
            "subject_mask_authority_generation": 1,
            "subject_mask_authority": {
                "schema_id": "palette.subject_mask.bundle_authority",
                "schema_version": 1,
                "generation": 1,
                "bundle_id": bundle_id,
                "bundle_path": f"subject_mask_bundle_runs/{bundle_id}",
                "members": {
                    "refined": {
                        "run_path": "refined_subject_masks_runs/refined_a"
                    }
                },
            },
        },
    )
    bundle_parent = tmp_path / "subject_mask_bundle_runs"
    _write_zarr_metadata(bundle_parent)
    _write_zarr_metadata(
        bundle_parent / bundle_id,
        {
            "palette_run_completion_status": "complete",
            "subject_mask_bundle_selector_eligible": True,
        },
    )
    refined_parent = tmp_path / "refined_subject_masks_runs"
    _write_zarr_metadata(refined_parent)
    _write_zarr_metadata(
        refined_parent / "refined_a",
        {
            "palette_run_completion_status": "complete",
            "stage_selector_eligible": False,
            "subject_mask_bundle_selector_eligible": True,
        },
    )

    status = discover_stage_availability(tmp_path, "refined_subject_masks")

    assert status.available is True
    assert status.run_name == "bundle/bundle_a"
    assert status.artifact_path == "subject_mask_bundle_runs/bundle_a"
    assert "active subject-mask bundle" in status.reason


@pytest.mark.parametrize(
    ("stage_id", "attributes", "reason"),
    (
        (
            "refined_keypoints",
            {"keypoint_bundle_authority_lease": {"owner": "test"}},
            "activation lease",
        ),
        (
            "refined_subject_masks",
            {
                "subject_mask_authority_generation": 1,
                "subject_mask_authority": {"schema_id": "wrong"},
            },
            "malformed or incomplete",
        ),
    ),
)
def test_authority_resolvers_fail_closed_on_in_progress_or_malformed_root_state(
    tmp_path: Path,
    stage_id: str,
    attributes: dict[str, object],
    reason: str,
) -> None:
    _write_zarr_metadata(tmp_path, attributes)

    status = discover_stage_availability(tmp_path, stage_id)

    assert status.available is False
    assert reason in status.reason


def test_visualization_availability_is_tied_to_selected_track_run(
    tmp_path: Path,
) -> None:
    parent = tmp_path / "analysis" / "track_kinematics_runs" / "offline"
    _write_zarr_metadata(
        parent,
        {"latest": "track_a", "latest_complete": "track_a"},
    )
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
    assert "sibling track-kinematics visualization parent is missing" in missing.reason

    visualization_parent = (
        tmp_path
        / "analysis"
        / "track_kinematics_visualization_runs"
        / "offline"
        / "track_a"
        / "tracks"
        / "id_0"
    )
    _write_zarr_metadata(
        visualization_parent,
        {"latest": "render_a", "latest_complete": "render_a"},
    )
    motion_authority = {
        "run_ref": "/analysis/track_kinematics_runs/offline/track_a",
        "track_ref": (
            "/analysis/track_kinematics_runs/offline/track_a/tracks/id_0"
        ),
        "track_id": 0,
        "motion_manifest_sha256": "a" * 64,
        "positions_px_coordinate_descriptor_sha256": "b" * 64,
    }
    render = visualization_parent / "render_a"
    _write_zarr_metadata(
        render,
        {
            "palette_run_completion_status": "complete",
            "stage_selector_eligible": True,
            "source_track_motion_authority": motion_authority,
            "track_id": 0,
        },
    )
    artifact = (
        render
        / "visualizations"
        / "track_kinematics_summary_track_0_interactive"
    )
    _write_zarr_metadata(
        artifact,
        {
            "renderer": "palette-track-kinematics-summary-v1",
            "source_runs": {"track_kinematics": "offline/track_a"},
            "parameters": {"swim_bout_run": "swim_a"},
            "track_motion_authority": motion_authority,
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
        "analysis/track_kinematics_visualization_runs/offline/track_a/"
        "tracks/id_0/render_a/visualizations/"
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
    _write_zarr_metadata(
        parent / "bout_a",
        {"palette_run_completion_status": "complete"},
    )

    unresolved = discover_stage_availability(tmp_path, "swim_bouts")
    explicit = discover_stage_availability(
        tmp_path,
        "swim_bouts",
        requested_run="bout_a",
    )

    assert unresolved.available is False
    assert "no stable complete selector-eligible run" in unresolved.reason
    assert explicit.available is True
    assert explicit.run_name == "bout_a"


def test_availability_resolver_does_not_reuse_incomplete_run(tmp_path: Path) -> None:
    parent = tmp_path / "analysis" / "eye_angle_runs"
    _write_zarr_metadata(
        parent,
        {"latest": "eye_a", "latest_complete": "eye_a"},
    )
    _write_zarr_metadata(
        parent / "eye_a",
        {"palette_run_completion_status": "running"},
    )

    status = discover_stage_availability(tmp_path, "eye_angles")

    assert status.available is False
    assert "no stable complete selector-eligible run" in status.reason


def test_availability_resolver_fails_closed_for_unmarked_strict_parent(
    tmp_path: Path,
) -> None:
    parent = tmp_path / "analysis" / "subject_shape_runs"
    _write_zarr_metadata(
        parent,
        {
            "palette_completion_epoch": 2,
            "latest": "shape_a",
            "latest_complete": "shape_a",
        },
    )
    _write_zarr_metadata(parent / "shape_a")

    status = discover_stage_availability(tmp_path, "subject_shape")

    assert status.available is False
    assert status.run_name is None
    assert "no stable complete selector-eligible run" in status.reason


def test_availability_resolver_fails_closed_during_selector_activation(
    tmp_path: Path,
) -> None:
    parent = tmp_path / "analysis" / "eye_angle_runs"
    _write_zarr_metadata(
        parent,
        {"latest": "candidate", "latest_complete": "previous"},
    )
    _write_zarr_metadata(
        parent / "previous",
        {
            "palette_run_completion_status": "complete",
            "stage_selector_eligible": True,
        },
    )
    _write_zarr_metadata(
        parent / "candidate",
        {
            "palette_run_completion_status": "complete",
            "stage_selector_eligible": False,
        },
    )

    status = discover_stage_availability(tmp_path, "eye_angles")

    assert status.available is False
    assert status.run_name is None
    assert "selector activation may be in progress" in status.reason


def test_availability_resolver_rejects_explicit_ineligible_run(
    tmp_path: Path,
) -> None:
    parent = tmp_path / "analysis" / "swim_bout_runs"
    _write_zarr_metadata(parent)
    _write_zarr_metadata(
        parent / "candidate",
        {
            "palette_run_completion_status": "complete",
            "stage_selector_eligible": False,
        },
    )

    status = discover_stage_availability(
        tmp_path,
        "swim_bouts",
        requested_run="candidate",
    )

    assert status.available is False
    assert status.run_name == "candidate"
    assert "not selector-eligible" in status.reason


@pytest.mark.parametrize(
    "requested_run",
    (
        "  bout_a  ",
        "/analysis/swim_bout_runs/bout_a/",
    ),
)
def test_metadata_availability_normalizes_explicit_run_like_open_zarr(
    tmp_path: Path,
    requested_run: str,
) -> None:
    parent = tmp_path / "analysis" / "swim_bout_runs"
    _write_zarr_metadata(parent)
    _write_zarr_metadata(
        parent / "bout_a",
        {
            "palette_run_completion_status": "complete",
            "stage_selector_eligible": True,
        },
    )

    status = discover_stage_availability(
        tmp_path,
        "swim_bouts",
        requested_run=requested_run,
    )

    assert status.available is True
    assert status.run_name == "bout_a"
