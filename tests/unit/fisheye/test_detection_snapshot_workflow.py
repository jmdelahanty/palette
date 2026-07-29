from __future__ import annotations

from pathlib import Path

from fisheye.cluster.detection_snapshots import (
    DetectionSnapshotFragmentInputs,
    build_detection_snapshot_fragment,
    compose_detection_snapshot_workflow,
)
from fisheye.cluster.lsf import (
    LsfJob,
    LsfResources,
    LsfWorkflowFragment,
    compose_lsf_workflow,
)


def test_snapshot_fragment_is_typed_selector_ineligible_and_composable(
    tmp_path: Path,
) -> None:
    upstream = LsfWorkflowFragment(
        fragment_id="detect_refine",
        jobs=(
            LsfJob(
                job_key="refine_detect:recording_a",
                job_name="refine_detect_recording_a",
                command=("true",),
                resources=LsfResources(queue="short", ncores=1, mem_gb=1),
                stdout_path=tmp_path / "upstream.out",
                stderr_path=tmp_path / "upstream.err",
            ),
        ),
        provides=("refined_detection:recording_a",),
    )
    module = build_detection_snapshot_fragment(
        DetectionSnapshotFragmentInputs(
            workflow_id="campaign_1",
            family="analysis.full_recording",
            target_id="recording_a",
            analysis_zarr=tmp_path / "recording_a_analysis.zarr",
            recording_identity="recording_a",
            source_detect_group_path="detect_runs/detect_source",
            source_refined_group_path="refined_detect_runs/refined_source",
            canonical_run_id="detect_snapshot_v1",
            refined_run_id="refined_snapshot_v1",
            repo=Path("/groups/palette"),
            run_root=tmp_path / "workflow",
            upstream_job_keys=("refine_detect:recording_a",),
            required_artifacts=("refined_detection:recording_a",),
        )
    )

    outputs = module.outputs
    assert outputs.canonical_group_path == "detect_runs/detect_snapshot_v1"
    assert outputs.refined_group_path == ("refined_detect_runs/refined_snapshot_v1")
    assert outputs.to_json()["selector_eligible"] is False
    assert module.fragment.requires == ("refined_detection:recording_a",)
    assert module.fragment.provides == ("detection_snapshot_pair:recording_a",)
    assert module.fragment.metadata["lineage_profile"] == "full_acquisition"
    assert module.fragment.metadata["selector_activation"] == "deferred"

    job = module.fragment.jobs[0]
    assert job.dependency.upstream_job_keys == ("refine_detect:recording_a",)
    command = " ".join(job.command)
    assert "fisheye.utils.publish_detection_snapshots" in command
    assert "--source-detect-group detect_runs/detect_source" in command
    assert "--source-refined-group refined_detect_runs/refined_source" in command
    assert "--canonical-run detect_snapshot_v1" in command
    assert "--refined-run refined_snapshot_v1" in command
    assert "__PALETTE_LSF_JOBID__" in command

    workflow = compose_lsf_workflow(
        workflow_id="campaign_1",
        family="analysis.full_recording",
        fragments=(upstream, module.fragment),
    )
    assert workflow.metadata["fragments"][1]["provides"] == [
        "detection_snapshot_pair:recording_a"
    ]


def test_snapshot_only_workflow_accepts_preexisting_source_artifacts(
    tmp_path: Path,
) -> None:
    module = build_detection_snapshot_fragment(
        DetectionSnapshotFragmentInputs(
            workflow_id="snapshot_only",
            family="analysis.full_recording",
            target_id="recording_a",
            analysis_zarr=tmp_path / "recording_a_analysis.zarr",
            recording_identity="recording_a",
            source_detect_group_path="detect_runs/detect_source",
            source_refined_group_path="refined_detect_runs/refined_source",
            canonical_run_id="detect_snapshot_v1",
            refined_run_id="refined_snapshot_v1",
            repo=Path("/groups/palette"),
            run_root=tmp_path / "workflow",
            required_artifacts=("refined_detection:recording_a",),
        )
    )

    workflow = compose_detection_snapshot_workflow(
        workflow_id="snapshot_only",
        family="analysis.full_recording",
        modules=(module,),
        external_inputs=("refined_detection:recording_a",),
    )

    assert workflow.metadata["target_count"] == 1
    assert workflow.metadata["selector_activation"] == "deferred"
    assert workflow.metadata["outputs"][0]["selector_eligible"] is False


def test_snapshot_fragment_exposes_explicit_historical_migration_flags(
    tmp_path: Path,
) -> None:
    module = build_detection_snapshot_fragment(
        DetectionSnapshotFragmentInputs(
            workflow_id="historical",
            family="analysis.full_recording",
            target_id="recording_b",
            analysis_zarr=tmp_path / "recording_b_analysis.zarr",
            recording_identity="recording_b",
            source_detect_group_path="detect_runs/detect_source",
            source_refined_group_path="refined_detect_runs/refined_source",
            canonical_run_id="detect_snapshot_v1",
            refined_run_id="refined_snapshot_v1",
            repo=Path("/groups/palette"),
            run_root=tmp_path / "workflow",
            allow_initialize_missing_source_keys=True,
            allow_manual_score_reset=True,
        )
    )

    command = " ".join(module.fragment.jobs[0].command)
    assert "--allow-initialize-missing-source-keys" in command
    assert "--allow-manual-score-reset" in command
