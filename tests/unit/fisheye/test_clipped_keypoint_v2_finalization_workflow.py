from __future__ import annotations

from pathlib import Path

from fisheye.cluster.keypoints.v2_finalization import (
    ClipKeypointV2FinalizationInput,
    ClippedKeypointV2FinalizationInputs,
    build_clipped_keypoint_v2_finalization_fragment,
)


def _inputs(tmp_path: Path) -> ClippedKeypointV2FinalizationInputs:
    clips = tuple(
        ClipKeypointV2FinalizationInput(
            clip_id=f"clip_{index}",
            clip_index=index,
            source_group_path=f"keypoint_shard_runs/shard_{index}",
            input_package_manifest_path=tmp_path / f"cache_{index}.json",
        )
        for index in range(2)
    )
    return ClippedKeypointV2FinalizationInputs(
        workflow_id="wf",
        family="analysis.clipped",
        target_id="sleepyfish",
        analysis_zarr=tmp_path / "recording_analysis.zarr",
        crop_run_id="crop_v2",
        clips=clips,
        pose_binding_path=tmp_path / "pose_binding.json",
        preprocessing_path=tmp_path / "preprocessing.json",
        bundle_root=tmp_path / ".palette_benchmarks" / "keypoint_v2_bundle",
        raw_run_id="raw_v2",
        quality_run_id="quality_v1",
        refined_run_id="refined_v2",
        body_frame_run_id="body_v1",
        recording_identity="sleepyfish",
        refined_lineage_id="33333333-3333-4333-8333-333333333333",
        refined_snapshot_id="44444444-4444-4444-8444-444444444444",
        repo=tmp_path / "repo",
        run_root=tmp_path / "run",
        upstream_job_keys=("keypoints_array:sleepyfish", "crop_snapshot:sleepyfish"),
        required_artifacts=("crop_snapshot:sleepyfish",),
    )


def test_fragment_writes_terminal_sidecars_before_one_recording_finalizer(
    tmp_path: Path,
) -> None:
    module = build_clipped_keypoint_v2_finalization_fragment(_inputs(tmp_path))

    assert len(module.fragment.jobs) == 2
    terminal, finalizer = module.fragment.jobs
    assert terminal.execution_group.mode.value == "array"
    assert terminal.execution_group.max_concurrent == 2
    assert terminal.dependency.upstream_job_keys == (
        "keypoints_array:sleepyfish",
        "crop_snapshot:sleepyfish",
    )
    assert finalizer.dependency.upstream_job_keys == (terminal.job_key,)
    rendered = " ".join(finalizer.command)
    assert "fisheye.utils.finalize_clipped_keypoint_v2_bundle" in rendered
    assert rendered.count("--clip-receipt") == 2
    assert str(module.outputs.finalization_receipt_path) in finalizer.command
    assert module.fragment.metadata["compute_partition"] == "clip_local"
    assert module.fragment.metadata["publication_partition"] == (
        "complete_recording_snapshot"
    )
    assert module.fragment.metadata["selector_activation"] == (
        "none_direct_path_only"
    )
    assert module.outputs.to_json()["selector_eligible"] is False


def test_fragment_rejects_noncontiguous_clip_indices(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    bad_clips = (
        inputs.clips[0],
        ClipKeypointV2FinalizationInput(
            clip_id="clip_2",
            clip_index=2,
            source_group_path="keypoint_shard_runs/shard_2",
            input_package_manifest_path=tmp_path / "cache_2.json",
        ),
    )
    try:
        ClippedKeypointV2FinalizationInputs(
            **{**inputs.__dict__, "clips": bad_clips}
        )
    except ValueError as exc:
        assert "Clip indices" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("noncontiguous clip indices unexpectedly passed")
