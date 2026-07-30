from __future__ import annotations

from pathlib import Path

import pytest

from fisheye.cluster.clipped_detection_evidence import (
    ClipDetectionEvidenceInput,
    ClippedDetectionEvidenceInputs,
)
from fisheye.cluster.clipped_storage_finalization import (
    ClippedStorageFinalizationInputs,
    StrictClipRefinedDetectionInput,
)
from fisheye.cluster.crimson_storage_candidate import (
    CrimsonCandidateScale,
    CrimsonStorageCandidateInputs,
    build_crimson_storage_candidate_workflow,
)
from fisheye.cluster.keypoints.v2_finalization import (
    ClipKeypointV2FinalizationInput,
    ClippedKeypointV2FinalizationInputs,
    RecordingAggregateKeypointV2AdapterInputs,
)


def _inputs(
    tmp_path: Path,
    *,
    scale: CrimsonCandidateScale = CrimsonCandidateScale.FULL_DURATION,
) -> CrimsonStorageCandidateInputs:
    repo = tmp_path / "repo"
    run_root = tmp_path / "run"
    bundle = tmp_path / ".palette_benchmarks" / "candidate"
    clips = tuple(
        ClipDetectionEvidenceInput(
            clip_index=index,
            clip_id=f"clip_{index:06d}",
            source_detect_group_path=f"clips/{index}/detect",
            source_refined_group_path=f"clips/{index}/refined",
            canonical_run_id=f"canonical_{index}",
            refined_run_id=f"refined_{index}",
        )
        for index in range(2)
    )
    evidence = ClippedDetectionEvidenceInputs(
        workflow_id="wf",
        family="analysis.crimson_candidate",
        target_id="recording",
        analysis_zarr=tmp_path / "analysis.zarr",
        recording_canonical_archive=tmp_path / "canonical.zarr",
        recording_canonical_run_id="canonical_recording",
        recording_identity="recording",
        detection_plan_path=tmp_path / "detection_plan.json",
        collection_id="collection",
        recording_dir=tmp_path / "recording",
        bundle_root=bundle / "clip_evidence",
        clips=clips,
        repo=repo,
        run_root=run_root,
    )
    storage = ClippedStorageFinalizationInputs(
        workflow_id="wf",
        family="analysis.crimson_candidate",
        target_id="recording",
        analysis_zarr=tmp_path / "analysis.zarr",
        canonical_archive=tmp_path / "canonical.zarr",
        canonical_run_id="canonical_recording",
        clips=tuple(
            StrictClipRefinedDetectionInput(
                clip_index=index,
                clip_id=f"clip_{index:06d}",
                archive=tmp_path / f"old_{index}.zarr",
                run_id=f"old_{index}",
            )
            for index in range(2)
        ),
        clipped_binding_path=tmp_path / "binding.json",
        bundle_root=bundle,
        refined_run_id="refined_recording",
        refined_lineage_id="11111111-1111-4111-8111-111111111111",
        refined_snapshot_id="22222222-2222-4222-8222-222222222222",
        crop_run_id="crop_recording",
        recording_identity="recording",
        crop_purpose="pose",
        roi_width=512,
        roi_height=512,
        camera_id="2010095",
        repo=repo,
        run_root=run_root,
    )
    keypoints = ClippedKeypointV2FinalizationInputs(
        workflow_id="wf",
        family="analysis.crimson_candidate",
        target_id="recording",
        analysis_zarr=tmp_path / "analysis.zarr",
        crop_run_id="replaced_crop",
        clips=tuple(
            ClipKeypointV2FinalizationInput(
                clip_id=f"clip_{index:06d}",
                clip_index=index,
                source_group_path=f"keypoint_shard_runs/clip_{index}",
                input_package_manifest_path=tmp_path / f"package_{index}.json",
            )
            for index in range(2)
        ),
        pose_binding_path=tmp_path / "pose_binding.json",
        preprocessing_path=tmp_path / "preprocessing.json",
        bundle_root=bundle / "keypoints",
        raw_run_id="raw_keypoints",
        quality_run_id="keypoint_quality",
        refined_run_id="refined_keypoints",
        body_frame_run_id="body_frame",
        recording_identity="recording",
        refined_lineage_id="33333333-3333-4333-8333-333333333333",
        refined_snapshot_id="44444444-4444-4444-8444-444444444444",
        repo=repo,
        run_root=run_root,
    )
    return CrimsonStorageCandidateInputs(
        candidate_id="sleepyfish_full_v1",
        scale=scale,
        expected_n_frames=(1_188_000 if scale is CrimsonCandidateScale.FULL_DURATION else 23_287),
        expected_n_instances=(1_169_010 if scale is CrimsonCandidateScale.FULL_DURATION else 22_926),
        evidence=evidence,
        storage=storage,
        keypoints=keypoints,
        handoff_path=bundle / "handoff_manifest.json",
        palette_commit="e" * 40,
        crimson_contract_commit="a" * 40,
        crimson_contract_sha256="b" * 64,
    )


def test_full_candidate_composes_all_recording_level_gates(tmp_path: Path) -> None:
    plan = build_crimson_storage_candidate_workflow(_inputs(tmp_path))

    assert [
        fragment["metadata"]["module"]
        for fragment in plan.workflow.metadata["fragments"]
    ] == [
        "strict_clipped_detection_evidence",
        "clipped_storage_finalization",
        "clipped_keypoint_v2_finalization",
        "crimson_storage_candidate_handoff",
    ]
    assert plan.workflow.metadata["classification"] == "full_duration_fixture"
    assert plan.workflow.metadata["pixel_payload_in_analysis_archive"] is False
    assert plan.workflow.metadata["selector_eligible"] is False
    ordered = [job.job_key for job in plan.workflow.topological_jobs()]
    assert ordered[-1] == "crimson_storage_handoff:sleepyfish_full_v1"
    assert plan.handoff_fragment.requires == (
        "selector_ineligible_refined_detection:recording",
        "selector_ineligible_crop_v2:recording",
        "selector_ineligible_keypoint_v2_chain:recording",
    )
    rendered = " ".join(plan.handoff_fragment.jobs[0].command)
    assert "--classification full_duration_fixture" in rendered
    assert "--expected-n-frames 1188000" in rendered
    assert "--expected-n-instances 1169010" in rendered


def test_integration_candidate_remains_explicitly_non_scalability_evidence(
    tmp_path: Path,
) -> None:
    plan = build_crimson_storage_candidate_workflow(
        _inputs(tmp_path, scale=CrimsonCandidateScale.INTEGRATION)
    )

    assert plan.workflow.metadata["classification"] == "integration_fixture"
    rendered = " ".join(plan.handoff_fragment.jobs[0].command)
    assert "--classification integration_fixture" in rendered
    assert "--expected-n-frames 23287" in rendered


def test_candidate_rejects_keypoints_outside_its_complete_bundle(
    tmp_path: Path,
) -> None:
    inputs = _inputs(tmp_path)
    with pytest.raises(ValueError, match="keypoints.bundle_root"):
        CrimsonStorageCandidateInputs(
            **{
                **inputs.__dict__,
                "keypoints": ClippedKeypointV2FinalizationInputs(
                    **{
                        **inputs.keypoints.__dict__,
                        "bundle_root": tmp_path / "other",
                    }
                ),
            }
        )


def test_full_candidate_can_republish_pinned_recording_aggregate(
    tmp_path: Path,
) -> None:
    inputs = _inputs(tmp_path)
    aggregate = RecordingAggregateKeypointV2AdapterInputs(
        workflow_id=inputs.keypoints.workflow_id,
        family=inputs.keypoints.family,
        target_id=inputs.keypoints.target_id,
        analysis_zarr=inputs.keypoints.analysis_zarr,
        source_group_path="keypoints_runs/historical_recording",
        source_group_metadata_sha256="c" * 64,
        expected_model_sha256="d" * 64,
        expected_n_frames=inputs.expected_n_frames,
        expected_n_instances=inputs.expected_n_instances,
        crop_run_id=inputs.keypoints.crop_run_id,
        bundle_root=inputs.keypoints.bundle_root,
        raw_run_id=inputs.keypoints.raw_run_id,
        quality_run_id=inputs.keypoints.quality_run_id,
        refined_run_id=inputs.keypoints.refined_run_id,
        body_frame_run_id=inputs.keypoints.body_frame_run_id,
        recording_identity=inputs.keypoints.recording_identity,
        refined_lineage_id=inputs.keypoints.refined_lineage_id,
        refined_snapshot_id=inputs.keypoints.refined_snapshot_id,
        repo=inputs.keypoints.repo,
        run_root=inputs.keypoints.run_root,
    )

    plan = build_crimson_storage_candidate_workflow(
        CrimsonStorageCandidateInputs(
            **{**inputs.__dict__, "keypoints": aggregate}
        )
    )

    modules = [
        fragment["metadata"]["module"]
        for fragment in plan.workflow.metadata["fragments"]
    ]
    assert modules[2] == "recording_keypoint_v2_benchmark_adapter"
    adapter = plan.keypoints.fragment.jobs[0]
    assert adapter.metadata["stage"] == (
        "keypoint_v2_recording_aggregate_benchmark_adapter"
    )
    rendered = " ".join(adapter.command)
    assert "--source-group keypoints_runs/historical_recording" in rendered
    assert "--expected-n-frames 1188000" in rendered
    assert "--expected-n-instances 1169010" in rendered
    assert str(plan.detection_storage.storage.outputs.refined_archive) in rendered
    assert str(plan.detection_storage.storage.outputs.crop_archive) in rendered
