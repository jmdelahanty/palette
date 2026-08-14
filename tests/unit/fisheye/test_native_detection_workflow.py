from __future__ import annotations

from pathlib import Path

import pytest

from fisheye.cluster.native_detection import (
    NativeDetectionAuthoritySpec,
    NativeDetectionClipSpec,
    NativeDetectionFragmentInputs,
    NativeDetectionModelSpec,
    build_native_detection_fragment,
    compose_native_detection_workflow,
)


def _clips(tmp_path: Path) -> tuple[NativeDetectionClipSpec, ...]:
    return tuple(
        NativeDetectionClipSpec(
            work_unit_id=f"recording_clip_{index}",
            clip_id=f"clip_{index:06d}",
            clip_index=index,
            camera_serial="2010093",
            video_path=tmp_path / "clips" / f"clip_{index:06d}.mp4",
            artifact_run_id=f"detect_artifact_{index}",
            artifact_group_path=(
                f"clips/clip_{index:06d}/cameras/2010093/"
                f"detection_artifact_runs/detect_artifact_{index}"
            ),
            report_path=tmp_path / "run" / "reports" / f"clip_{index:06d}.json",
        )
        for index in range(2)
    )


def _inputs(tmp_path: Path) -> NativeDetectionFragmentInputs:
    return NativeDetectionFragmentInputs(
        workflow_id="native_detection_fixture",
        family="analysis.detection",
        target_id="recording_a",
        recording_identity="recording:a",
        recording_dir=tmp_path / "recording",
        analysis_zarr=tmp_path / "recording_analysis.zarr",
        repo=tmp_path / "repo",
        run_root=tmp_path / "run",
        canonical_run_id="detect_native_recording_a",
        n_frames=6,
        source_width=640,
        source_height=480,
        source_frame_authority=NativeDetectionAuthoritySpec(
            record_ref="/analysis/acquisition_camera_frames/frame_axis@record",
            record_sha256="1" * 64,
        ),
        source_pixel_authority=NativeDetectionAuthoritySpec(
            record_ref="/raw_video@source_pixel_authority",
            record_sha256="2" * 64,
        ),
        producer_version="e3936b9a",
        clips=_clips(tmp_path),
        model=NativeDetectionModelSpec(
            set_id="detect_set",
            run_id="detect_model_run",
            path=tmp_path / "model.pt",
            sha256="3" * 64,
        ),
        detect_array_concurrency=2,
    )


def test_native_detection_fragment_separates_artifacts_from_canonical_run(
    tmp_path: Path,
) -> None:
    module = build_native_detection_fragment(_inputs(tmp_path))
    array_job, publish_job = module.fragment.jobs

    assert array_job.job_key == "detect_artifact_array:recording_a"
    assert array_job.metadata["execution_mode"] == "array"
    assert array_job.metadata["max_concurrent"] == 2
    assert len(array_job.execution_group.tasks) == 2
    first = array_job.execution_group.tasks[0]
    command = " ".join(first.command)
    assert "--target-group-path" in command
    assert "detection_artifact_runs/detect_artifact_0" in command
    assert "/detect_runs/" not in command
    assert "--frame-mapping-mode recording_frame_index" in command

    assert publish_job.job_key == "detect_native_publish:recording_a"
    assert publish_job.dependency.upstream_job_keys == (array_job.job_key,)
    publish_command = " ".join(publish_job.command)
    assert "fisheye.utils.assemble_clipped_native_detection" in publish_command
    assert publish_command.count("--work-unit-report") == 2
    assert "--run-id detect_native_recording_a" in publish_command
    assert "--producer-version e3936b9a" in publish_command

    outputs = module.outputs.to_json()
    assert outputs["canonical_group_path"] == (
        "detect_runs/detect_native_recording_a"
    )
    assert all(
        "/detection_artifact_runs/" in path
        for path in outputs["artifact_group_paths"]
    )
    assert outputs["native_run_manifest_schema_version"] == 3
    assert outputs["logical_schema_version"] == 1
    assert outputs["selector_eligible"] is True
    assert module.fragment.metadata["registry_update"] is False


def test_native_detection_module_is_detection_only_and_composable(
    tmp_path: Path,
) -> None:
    module = build_native_detection_fragment(_inputs(tmp_path))
    workflow = compose_native_detection_workflow(
        workflow_id="native_detection_fixture",
        family="analysis.detection",
        modules=(module,),
    )

    assert [job.job_key for job in workflow.topological_jobs()] == [
        "detect_artifact_array:recording_a",
        "detect_native_publish:recording_a",
    ]
    assert workflow.metadata["workflow_scope"] == "native_canonical_detection"
    assert workflow.metadata["selector_activation"] == (
        "atomic_after_canonical_v3_validation"
    )
    assert module.fragment.provides == ("canonical_detection:recording_a",)


def test_native_detection_rejects_canonical_path_as_artifact_target(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="detection_artifact_runs"):
        NativeDetectionClipSpec(
            work_unit_id="work",
            clip_id="clip_000000",
            clip_index=0,
            camera_serial="2010093",
            video_path=tmp_path / "clip.mp4",
            artifact_run_id="detect_0",
            artifact_group_path="clips/clip_000000/cameras/2010093/detect_runs/detect_0",
            report_path=tmp_path / "report.json",
        )


def test_native_clip_spec_consumes_explicit_artifact_path_from_plan(
    tmp_path: Path,
) -> None:
    spec = NativeDetectionClipSpec.from_plan_work_unit(
        {
            "work_unit_id": "recording_clip_0",
            "clip_id": "clip_000000",
            "clip_index": 0,
            "camera_serial": "2010093",
            "source": {"video_path": str(tmp_path / "clip.mp4")},
            "run_names": {"detect": "detect_artifact_0"},
            "zarr_paths": {
                "detection_artifact_target_group_path": (
                    "clips/clip_000000/cameras/2010093/"
                    "detection_artifact_runs/detect_artifact_0"
                )
            },
        },
        report_path=tmp_path / "report.json",
    )

    assert spec.artifact_group_path.endswith(
        "detection_artifact_runs/detect_artifact_0"
    )
