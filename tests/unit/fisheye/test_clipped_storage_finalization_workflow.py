from __future__ import annotations

from pathlib import Path

from fisheye.cluster.clipped_storage_finalization import (
    ClippedStorageFinalizationInputs,
    StrictClipRefinedDetectionInput,
    build_clipped_storage_finalization_fragment,
    build_clipped_storage_keypoint_chain_fragments,
)
from tests.unit.fisheye.test_clipped_keypoint_v2_finalization_workflow import (
    _inputs as keypoint_inputs,
)


def _inputs(tmp_path: Path) -> ClippedStorageFinalizationInputs:
    return ClippedStorageFinalizationInputs(
        workflow_id="wf",
        family="analysis.clipped",
        target_id="sleepyfish",
        analysis_zarr=tmp_path / "recording_analysis.zarr",
        canonical_archive=tmp_path / "canonical.zarr",
        canonical_run_id="detect_recording",
        clips=tuple(
            StrictClipRefinedDetectionInput(
                clip_index=index,
                clip_id=f"clip_{index}",
                archive=tmp_path / f"refined_clip_{index}.zarr",
                run_id=f"refined_clip_{index}",
            )
            for index in range(2)
        ),
        clipped_binding_path=tmp_path / "clipped_binding.json",
        bundle_root=tmp_path / ".palette_benchmarks" / "storage_bundle",
        refined_run_id="refined_recording",
        refined_lineage_id="11111111-1111-4111-8111-111111111111",
        refined_snapshot_id="22222222-2222-4222-8222-222222222222",
        crop_run_id="crop_recording",
        recording_identity="sleepyfish",
        crop_purpose="keypoints",
        roi_width=512,
        roi_height=384,
        camera_id="2010095",
        repo=tmp_path / "repo",
        run_root=tmp_path / "run",
        upstream_job_keys=("canonical_publish:sleepyfish",),
        required_artifacts=("canonical_detection:sleepyfish",),
    )


def test_fragment_finalizes_refined_then_crop_without_selectors(tmp_path: Path) -> None:
    module = build_clipped_storage_finalization_fragment(_inputs(tmp_path))

    assert len(module.fragment.jobs) == 2
    refined, crop = module.fragment.jobs
    assert refined.dependency.upstream_job_keys == ("canonical_publish:sleepyfish",)
    assert crop.dependency.upstream_job_keys == (refined.job_key,)
    refined_command = " ".join(refined.command)
    crop_command = " ".join(crop.command)
    assert "fisheye.utils.finalize_clipped_refined_detection_v1" in refined_command
    assert refined_command.count("--clip-archive") == 2
    assert refined_command.count("--clip-run") == 2
    assert "fisheye.utils.publish_clipped_crop_geometry_v2" in crop_command
    assert crop_command.count("--clip-archive") == 2
    assert module.fragment.requires == ("canonical_detection:sleepyfish",)
    assert module.fragment.metadata["physical_layout_source"] == (
        "shared_byte_planners"
    )
    assert module.outputs.to_json()["selector_eligible"] is False


def test_chain_binds_standalone_crop_to_keypoint_finalizer(tmp_path: Path) -> None:
    storage_inputs = _inputs(tmp_path)
    keypoints = keypoint_inputs(tmp_path)
    modules = build_clipped_storage_keypoint_chain_fragments(
        storage_inputs,
        keypoints,
    )

    assert (
        modules.keypoints.outputs.crop_archive == modules.storage.outputs.crop_archive
    )
    assert modules.keypoints.outputs.crop_run_id == modules.storage.outputs.crop_run_id
    assert modules.storage.outputs.crop_artifact_key in (
        modules.keypoints.fragment.requires
    )
    terminal = modules.keypoints.fragment.jobs[0]
    assert modules.storage.outputs.terminal_job_key in (
        terminal.dependency.upstream_job_keys
    )
    assert "--crop-archive" in " ".join(terminal.execution_group.tasks[0].command)


def test_rejects_noncontiguous_refined_clip_inputs(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    bad = (
        inputs.clips[0],
        StrictClipRefinedDetectionInput(
            clip_index=2,
            clip_id="clip_2",
            archive=tmp_path / "clip_2.zarr",
            run_id="refined_clip_2",
        ),
    )
    try:
        ClippedStorageFinalizationInputs(**{**inputs.__dict__, "clips": bad})
    except ValueError as exc:
        assert "Clip indices" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("noncontiguous clip inputs unexpectedly passed")
