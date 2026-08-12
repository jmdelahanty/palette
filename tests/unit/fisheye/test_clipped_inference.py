from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from fisheye.cluster import clipped_inference as workflow
from fisheye.cluster import (
    clipped_inference_detect_quality_recovery as quality_recovery,
)
from fisheye.cluster import clipped_inference_import_recovery as import_recovery
from fisheye.cluster import clipped_inference_keypoint_recovery as recovery
from fisheye.cluster import refined_subject_mask_encoded_chunk_canary as encoded_canary
from fisheye.cluster.clipped_detection import (
    DetectionFragmentInputs,
    DetectionModelSpec,
    DetectionWorkUnitSpec,
    RawDetectionFragmentInputs,
    RawDetectionWorkUnitSpec,
    build_detection_fragment,
    build_raw_detection_fragment,
    compose_detection_workflow,
    compose_raw_detection_workflow,
)
from fisheye.cluster.clipped_inference_cleanup import cleanup
from fisheye.cluster.clipped_inference_validate import (
    _cache_manifest_report,
    _cache_validation_mode,
    _instance_keys,
    _refined_instance_keys,
    _require_exact_instance_key_order,
    _require_exact_vector,
    _validate_run_frame_counts,
)
from fisheye.cluster.recording_layout import (
    clipped_recording_target,
    whole_video_recording_target,
)
from fisheye.shared.flat_roi_cache import FLAT_ROI_CACHE_LAYOUT, FLAT_ROI_CACHE_SCHEMA


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_refined_detection_identity_uses_instances_subgroup() -> None:
    run = {
        "instance_key": np.asarray([999], dtype=np.uint64),
        "instances": {"instance_key": np.asarray([10, 11], dtype=np.uint64)},
    }

    report = _refined_instance_keys(run, label="refined_detect_runs/example")

    assert report == {"row_count": 2, "unique_count": 2, "dtype": "uint64"}


def test_clipped_validator_requires_exact_identity_order() -> None:
    with pytest.raises(RuntimeError, match="canonical lineage at row 0"):
        _require_exact_vector(
            np.asarray([11, 10], dtype=np.uint64),
            np.asarray([10, 11], dtype=np.uint64),
            label="refined_subject_masks/instance_key",
            dtype=np.uint64,
        )


def test_clipped_validator_rejects_equal_sized_unrelated_identity_rowsets() -> None:
    with pytest.raises(RuntimeError, match="first mismatch at row 1"):
        _require_exact_instance_key_order(
            np.asarray([10, 99, 30], dtype=np.uint64),
            np.asarray([10, 20, 30], dtype=np.uint64),
            label="raw subject-mask shard",
        )


def test_cache_manifest_report_reads_canonical_instance_keys(tmp_path: Path) -> None:
    archive = tmp_path / "recording_analysis.zarr"
    archive.mkdir()
    manifest_path = tmp_path / "cache.json"
    payload_path = tmp_path / "cache.bin"
    row_path = tmp_path / "cache.rows.parquet"
    payload_path.write_bytes(bytes(12))
    pq.write_table(
        pa.table({"instance_key": pa.array([10, 20, 30], type=pa.uint64())}),
        row_path,
    )
    _write_json(
        manifest_path,
        {
            "schema": FLAT_ROI_CACHE_SCHEMA,
            "layout": FLAT_ROI_CACHE_LAYOUT,
            "cache_complete": True,
            "source": {
                "archive_path": str(archive),
                "collection_id": "collection_001",
                "selection": {"clip_ids": ["clip_000001"]},
            },
            "array": {
                "shape": [3, 2, 2],
                "bin_path": payload_path.name,
            },
            "row_index": {
                "path": row_path.name,
                "row_count": 3,
                "columns": ["instance_key"],
            },
        },
    )

    report, keys = _cache_manifest_report(
        manifest_path,
        zarr_path=archive.resolve(),
        collection_id="collection_001",
        clip_id="clip_000001",
    )

    assert report["instance_key"]["unique_count"] == 3
    np.testing.assert_array_equal(keys, np.asarray([10, 20, 30], dtype=np.uint64))


def test_clipped_validator_requires_recording_wide_exact_frame_counts() -> None:
    run = {
        "frame_indices": np.asarray([0, 2, 2, 4], dtype=np.int64),
        "frame_counts": np.asarray([1, 0, 2, 0, 1, 0], dtype=np.int32),
    }
    expected = np.asarray([1, 0, 2, 0, 1, 0], dtype=np.int64)

    report, observed = _validate_run_frame_counts(
        run,
        label="refined_subject_masks/example",
        expected_row_count=4,
        expected_counts=expected,
    )

    assert report["frame_count"] == 6
    assert report["sum"] == 4
    assert report["exact_bincount_match"] is True
    np.testing.assert_array_equal(observed, expected)

    run["frame_counts"] = np.asarray([1, 0, 2, 0, 0, 1], dtype=np.int32)
    with pytest.raises(RuntimeError, match="first mismatch at frame 4"):
        _validate_run_frame_counts(
            run,
            label="refined_subject_masks/example",
            expected_row_count=4,
            expected_counts=expected,
        )


def test_clipped_validator_post_cleanup_mode_requires_all_caches_absent() -> None:
    assert (
        _cache_validation_mode(cleaned_cache_count=0, clip_count=22) == "live_payloads"
    )
    assert _cache_validation_mode(cleaned_cache_count=22, clip_count=22) == (
        "post_cleanup_all_absent"
    )
    with pytest.raises(RuntimeError, match="partial cache set"):
        _cache_validation_mode(cleaned_cache_count=3, clip_count=22)


def _target(
    tmp_path: Path, name: str = "sleepyfish_cam2010093"
) -> workflow.CampaignTarget:
    recording = tmp_path / "recordings" / name
    zarr = recording / "zarr" / f"{name}_analysis.zarr"
    _write_json(recording / "recording_clip_index.json", {"clips": []})
    _write_json(
        zarr / "zarr.json", {"zarr_format": 3, "node_type": "group", "attributes": {}}
    )
    return workflow.CampaignTarget(
        target_id=name,
        recording_id=f"{name}:zfixture",
        recording_dir=recording,
        analysis_zarr=zarr,
    )


def _detection_plan(
    target: workflow.CampaignTarget, workflow_id: str
) -> dict[str, object]:
    work_units = []
    for index in range(22):
        clip_id = f"clip_{index:06d}"
        camera = target.target_id.removeprefix("sleepyfish_cam")
        detect = f"detect_{workflow_id}_{clip_id}"
        refined = f"refined_detect_{workflow_id}_{clip_id}"
        work_units.append(
            {
                "clip_id": clip_id,
                "clip_index": index,
                "camera_serial": camera,
                "work_unit_id": f"{target.target_id}_{clip_id}",
                "source": {
                    "video_path": str(target.recording_dir / "clips" / f"{clip_id}.mp4")
                },
                "run_names": {
                    "detect": detect,
                    "detect_quality": f"quality_{workflow_id}_{clip_id}",
                    "refined_detect": refined,
                },
                "zarr_paths": {
                    "detection_artifact_family_path": f"clips/{clip_id}/cameras/{camera}/detection_artifact_runs",
                    "detection_artifact_target_group_path": f"clips/{clip_id}/cameras/{camera}/detection_artifact_runs/{detect}",
                    "detect_family_path": f"clips/{clip_id}/cameras/{camera}/detect_runs",
                    "detect_target_group_path": f"clips/{clip_id}/cameras/{camera}/detect_runs/{detect}",
                    "refined_group_path": f"clips/{clip_id}/cameras/{camera}/refined_detect_runs/{refined}",
                },
                "commands": {},
            }
        )
    return {"work_unit_count": 22, "work_units": work_units}


def _build_fixture_plan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    resume_existing_detections: bool = False,
    encoded_mask_packages: bool = False,
    target_count: int = 1,
    max_active_targets: int = 3,
    subject_mask_publication_profile: str = (
        workflow.SUBJECT_MASK_PUBLICATION_RECEIPT_COMPOSED
    ),
) -> workflow.ClippedInferencePlan:
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    (repo / "scripts" / "py").write_text("#!/bin/sh\n", encoding="utf-8")
    (repo / "configs" / "fisheye").mkdir(parents=True)
    (repo / "configs" / "fisheye" / "yolo_detect_config.yaml").write_text(
        "{}\n", encoding="utf-8"
    )
    (repo / "configs" / "fisheye" / "default.yaml").write_text("{}\n", encoding="utf-8")
    targets = tuple(
        _target(tmp_path, f"sleepyfish_cam{2010093 + index}")
        for index in range(target_count)
    )
    targets_by_recording = {
        target.recording_dir.resolve(): target for target in targets
    }
    model = tmp_path / "models" / "model.pt"
    model.parent.mkdir()
    model.write_bytes(b"model")
    detect_binding = workflow.ModelBinding(
        "detect", "detect_set", "detect_run", model, "d" * 64
    )
    subject_binding = workflow.ModelBinding(
        "subject_masks", "mask_set", "mask_run", model, "m" * 64
    )

    monkeypatch.setattr(
        workflow, "validate_registered_analysis_zarr", lambda **_kwargs: None
    )
    monkeypatch.setattr(
        workflow, "_resolve_ranked_binding", lambda **_kwargs: detect_binding
    )
    monkeypatch.setattr(
        workflow, "_resolve_subject_binding", lambda **_kwargs: subject_binding
    )
    monkeypatch.setattr(workflow, "_verify_binding", lambda _binding: None)
    monkeypatch.setattr(workflow, "_repo_commit", lambda _repo: "c" * 40)
    monkeypatch.setattr(
        workflow,
        "load_native_archive_authority",
        lambda target: SimpleNamespace(
            recording_identity=target.recording_id,
            camera_serial=target.target_id.removeprefix("sleepyfish_cam"),
            n_frames=2200,
            source_width=4512,
            source_height=4512,
            frame=SimpleNamespace(record_ref="/frame", record_sha256="f" * 64),
            pixel=SimpleNamespace(record_ref="/pixel", record_sha256="p" * 64),
            to_json=lambda: {
                "recording_identity": target.recording_id,
                "camera_serial": target.target_id.removeprefix("sleepyfish_cam"),
                "n_frames": 2200,
                "source_width": 4512,
                "source_height": 4512,
            },
        ),
    )
    monkeypatch.setattr(
        workflow,
        "validate_recording_frame_index",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        workflow,
        "recording_frame_work_unit_intervals",
        lambda *_args, **_kwargs: {
            (index, f"clip_{index:06d}"): (index * 100, (index + 1) * 100)
            for index in range(22)
        },
    )
    monkeypatch.setattr(
        workflow,
        "resolve_pose_model_binding",
        lambda **_kwargs: SimpleNamespace(
            set_id="pose_set",
            run_id="pose_run",
            model_path=model,
            model_sha256="p" * 64,
        ),
    )
    monkeypatch.setattr(
        workflow,
        "build_detection_plan",
        lambda recording_dir, **kwargs: _detection_plan(
            targets_by_recording[Path(recording_dir).resolve()],
            str(kwargs["workflow_id"]),
        ),
    )
    if resume_existing_detections:
        monkeypatch.setattr(
            workflow,
            "_validate_existing_detection_for_resume",
            lambda **kwargs: {
                "status": "ok",
                "clip_id": str(kwargs["clip"]["clip_id"]),
                "target_group_path": str(kwargs["clip"]["detect_group_path"]),
            },
        )
    return workflow.build_plan(
        targets=targets,
        run_label="sleepyfish_full_20260714",
        repo=repo,
        registry_path=tmp_path / "registry.sqlite",
        run_root=tmp_path / "run",
        detection_set_id="detect_set",
        detection_run_id="detect_run",
        pose_set_id="pose_set",
        pose_run_id="pose_run",
        subject_mask_set_id="mask_set",
        subject_mask_run_id="mask_run",
        cache_root=tmp_path / "cache_root",
        package_root=tmp_path / "package_root",
        resume_existing_detections=resume_existing_detections,
        encoded_mask_packages=encoded_mask_packages,
        max_active_targets=max_active_targets,
        subject_mask_publication_profile=subject_mask_publication_profile,
    )


def _execution_tasks(job: object) -> dict[str, object]:
    group = getattr(job, "execution_group", None)
    assert group is not None
    return {task.task_key: task for task in group.tasks}


def test_build_plan_has_parallel_keypoint_mask_branch_and_join(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = _build_fixture_plan(tmp_path, monkeypatch)
    jobs = {job.job_key: job for job in plan.lsf_workflow.jobs}
    target = plan.target_plans[0]
    target_safe = workflow.safe_component(
        str(target["target_id"]), default="target", max_length=56
    )
    clip_id = "clip_000000"

    assert len(plan.lsf_workflow.jobs) == 21
    keypoint_array = jobs[f"keypoints_array:{target_safe}"]
    subject_mask_array = jobs[f"subject_masks_array:{target_safe}"]
    package_array = jobs[f"mask_package_array:{target_safe}"]
    assert keypoint_array.dependency.upstream_job_keys == (f"proxy:{target_safe}",)
    assert subject_mask_array.dependency.upstream_job_keys == (f"proxy:{target_safe}",)
    assert package_array.dependency.upstream_job_keys == (
        f"subject_masks_array:{target_safe}",
        f"keypoint_refine:{target_safe}",
    )
    assert f"keypoints:{target_safe}:{clip_id}" in _execution_tasks(keypoint_array)
    assert f"subject_masks:{target_safe}:{clip_id}" in _execution_tasks(
        subject_mask_array
    )
    subject_mask_task = _execution_tasks(subject_mask_array)[
        f"subject_masks:{target_safe}:{clip_id}"
    ]
    subject_mask_command = list(subject_mask_task.command)
    assert "fisheye.cluster.subject_masks.staged_inference" in subject_mask_command
    assert "--roi-cache-staging-dir" in subject_mask_command
    assert "--worker-receipt-json" in subject_mask_command
    assert any(
        str(path).endswith(f"subject_masks_{target_safe}_{clip_id}.json")
        for path in subject_mask_task.expected_outputs
    )
    assert f"mask_package:{target_safe}:{clip_id}" in _execution_tasks(package_array)
    cache_job = jobs[f"cache_array:{target_safe}"]
    cache_tasks = _execution_tasks(cache_job)
    publish_job = jobs[f"mask_publish:{target_safe}"]
    validation_job = jobs[f"validate:{target_safe}"]
    assert "--run-direct" in cache_tasks[f"cache:{target_safe}:00"].command
    assert "bsub" not in cache_job.command
    assert f"mask_import:{target_safe}" not in jobs
    assert publish_job.dependency.upstream_job_keys == (package_array.job_key,)
    assert publish_job.command.count("--raw-draft-run") == len(target["clips"])
    assert "fisheye.cluster.subject_masks.publish_receipt_composed_bundle" in (
        publish_job.command
    )
    assert publish_job.command.count("--refined-package") == len(target["clips"])
    assert "--producer-commit" in publish_job.command
    assert "--cache-run" in publish_job.command
    package_task = _execution_tasks(package_array)[
        f"mask_package:{target_safe}:{clip_id}"
    ]
    assert "--publication-evidence-producer-commit" in package_task.command
    assert "--global-frame-start" in package_task.command
    assert "--global-frame-stop" in package_task.command
    assert validation_job.dependency.upstream_job_keys == (publish_job.job_key,)
    assert all(
        job.command[:2]
        == (
            "env",
            f"PYTHONPATH={plan.repo / 'src'}",
        )
        for job in jobs.values()
    )
    assert all(
        f"PYTHONPATH={plan.repo / 'src'}" in job.command
        and str(plan.repo / "scripts" / "py") in job.command
        for job in jobs.values()
    )
    quality_source = jobs[f"detect_quality_source:{target_safe}"]
    quality = jobs[f"detect_quality:{target_safe}"]
    refine_bundle = jobs[f"detect_refine_bundle:{target_safe}"]
    refine = _execution_tasks(refine_bundle)[f"detect_refine:{target_safe}:{clip_id}"]
    assert quality_source.dependency.upstream_job_keys == (
        f"detect_native_publish:{target_safe}",
    )
    assert quality.dependency.upstream_job_keys == (quality_source.job_key,)
    assert refine_bundle.dependency.upstream_job_keys == (quality.job_key,)
    assert "--quality-group-path" in " ".join(refine.command)
    native_array = jobs[f"detect_artifact_array:{target_safe}"]
    assert native_array.metadata["max_concurrent"] == 8
    assert keypoint_array.metadata["max_concurrent"] == 4
    assert subject_mask_array.metadata["max_concurrent"] == 4
    assert len(_execution_tasks(native_array)) == 22
    assert len(_execution_tasks(refine_bundle)) == 22
    assert len(cache_tasks) == 3
    first_cache_task = cache_tasks[f"cache:{target_safe}:00"]
    max_workers_index = first_cache_task.command.index("--max-workers")
    assert first_cache_task.command[max_workers_index + 1] == "8"
    assert refine_bundle.resources.ncores == 16
    assert refine_bundle.resources.mem_gb == 32
    assert refine_bundle.resources.queue == "short"
    assert refine_bundle.resources.walltime == "1:00"
    assert refine_bundle.resources.span_hosts == 1
    assert "OMP_NUM_THREADS=4" in refine_bundle.command

    fragments = plan.lsf_workflow.to_json()["metadata"]["fragments"]
    assert [fragment["fragment_id"] for fragment in fragments] == [
        f"native_detection:{target_safe}",
        f"detection_postprocess:{target_safe}",
        f"strict_clipped_detection_evidence:{target_safe}",
        f"clipped_storage_finalization:{target_safe}",
        f"crop_roi_cache:{target_safe}",
        f"keypoints:{target_safe}",
        f"subject_mask_inference:{target_safe}",
        f"subject_mask_refinement:{target_safe}",
        f"analysis_validation:{target_safe}",
        "campaign_finalize",
    ]
    detection_output = target["detection_module"]
    native_output = target["native_detection_module"]
    strict_output = target["strict_detection_storage"]
    assert detection_output["terminal_job_key"] == f"detect_collection:{target_safe}"
    assert fragments[0]["provides"] == [f"canonical_detection:{target_safe}"]
    assert fragments[1]["requires"] == [native_output["artifact_key"]]
    assert fragments[1]["provides"] == [detection_output["artifact_key"]]
    assert fragments[2]["requires"] == [detection_output["artifact_key"]]
    assert fragments[3]["requires"] == [strict_output["evidence"]["artifact_key"]]
    assert fragments[4]["requires"] == [strict_output["storage"]["crop_artifact_key"]]
    assert fragments[5]["requires"] == [f"crop_roi_cache:{target_safe}"]
    assert fragments[6]["requires"] == [f"crop_roi_cache:{target_safe}"]
    assert fragments[7]["requires"] == [
        f"raw_subject_masks:{target_safe}",
        f"refined_keypoints:{target_safe}",
    ]
    assert fragments[8]["requires"] == [
        f"refined_keypoints:{target_safe}",
        f"refined_subject_masks:{target_safe}",
    ]
    assert fragments[9]["provides"] == [
        "registry_reconciled",
        "nrs_cache_cleaned",
    ]


def test_build_plan_retains_explicit_subject_mask_streaming_rollback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = _build_fixture_plan(
        tmp_path,
        monkeypatch,
        subject_mask_publication_profile=(
            workflow.SUBJECT_MASK_PUBLICATION_STREAMING_ROLLBACK
        ),
    )
    jobs = {job.job_key: job for job in plan.lsf_workflow.jobs}
    target = plan.target_plans[0]
    target_safe = workflow.safe_component(
        str(target["target_id"]), default="target", max_length=56
    )
    import_job = jobs[f"mask_import:{target_safe}"]
    publish_job = jobs[f"mask_publish:{target_safe}"]

    assert plan.subject_mask_publication_profile == (
        workflow.SUBJECT_MASK_PUBLICATION_STREAMING_ROLLBACK
    )
    assert "fisheye.utils.import_refined_subject_mask_clip_packages" in (
        import_job.command
    )
    assert publish_job.dependency.upstream_job_keys == (import_job.job_key,)
    assert "fisheye.cluster.subject_masks.publish_recording_bundle" in (
        publish_job.command
    )
    assert "--refined-draft-run" in publish_job.command
    assert "--refined-package" not in publish_job.command


def test_detection_module_composes_as_a_first_class_detection_only_workflow(
    tmp_path: Path,
) -> None:
    rows = tuple(
        {
            "clip_id": f"clip_{index:06d}",
            "clip_index": index,
            "camera_serial": "2010093",
            "work_unit_id": f"work_unit_{index}",
            "source": {"video_path": tmp_path / "clips" / f"clip_{index:06d}.mp4"},
        }
        for index in range(2)
    )
    target = clipped_recording_target(
        target_id="sleepyfish_cam2010093",
        recording_id="sleepyfish_cam2010093:zfixture",
        recording_dir=tmp_path / "recording",
        analysis_zarr=tmp_path / "analysis.zarr",
        work_units=rows,
    )
    units = tuple(
        DetectionWorkUnitSpec(
            work_unit=target.work_units[index],
            detect_run=f"detect_{index}",
            detect_group_path=f"clips/{index}/detect_runs/detect_{index}",
            refined_detect_run=f"refined_{index}",
            refined_detect_group_path=(
                f"clips/{index}/refined_detect_runs/refined_{index}"
            ),
        )
        for index in range(2)
    )
    module = build_detection_fragment(
        DetectionFragmentInputs(
            workflow_id="detection_only_fixture",
            family="clipped_inference",
            target_label="fixture_target",
            target=target,
            repo=tmp_path / "repo",
            run_root=tmp_path / "run",
            detection_plan_path=tmp_path / "run" / "detection_plan.json",
            collection_id="refined_detect_collection_fixture",
            quality_source_run="detect_quality_source_fixture",
            quality_run="detect_quality_fixture",
            work_units=units,
            model=DetectionModelSpec(
                set_id="detect_set",
                run_id="detect_run",
                path=tmp_path / "model.pt",
                sha256="a" * 64,
            ),
            detect_array_concurrency=2,
            refine_bundle_concurrency=2,
        )
    )
    detection_only = compose_detection_workflow(
        workflow_id="detection_only_fixture",
        family="clipped_inference",
        modules=(module,),
    )

    assert [job.job_key for job in detection_only.topological_jobs()] == [
        "detect_array:sleepyfish_cam2010093",
        "detect_quality_source:sleepyfish_cam2010093",
        "detect_quality:sleepyfish_cam2010093",
        "detect_refine_bundle:sleepyfish_cam2010093",
        "detect_collection:sleepyfish_cam2010093",
    ]
    assert module.outputs.collection_id == "refined_detect_collection_fixture"
    assert module.outputs.terminal_job_key == "detect_collection:sleepyfish_cam2010093"
    assert [fragment.fragment_id for fragment in module.fragments] == [
        "raw_detection:sleepyfish_cam2010093",
        "detection_postprocess:sleepyfish_cam2010093",
    ]
    assert module.fragments[1].requires == (module.raw_outputs.artifact_key,)
    assert detection_only.metadata["workflow_scope"] == "detection_only"
    assert len(_execution_tasks(detection_only.jobs[0])) == 2
    assert len(_execution_tasks(detection_only.jobs[3])) == 2


def test_whole_video_raw_detection_uses_atomic_local_publisher(
    tmp_path: Path,
) -> None:
    target = whole_video_recording_target(
        target_id="batman_arena_1",
        recording_id="batman_arena_1:zfixture",
        recording_dir=tmp_path / "recording",
        analysis_zarr=tmp_path / "recording" / "zarr" / "analysis.zarr",
        video_path=tmp_path / "recording" / "cams" / "Cam2010093.mp4",
        camera_serial="2010093",
    )
    run_name = "detect_batman_fixture"
    module = build_raw_detection_fragment(
        RawDetectionFragmentInputs(
            workflow_id="batman_raw_detect",
            family="production_inference",
            target_label="batman_arena_1",
            target=target,
            repo=tmp_path / "repo",
            run_root=tmp_path / "run",
            work_units=(
                RawDetectionWorkUnitSpec(
                    work_unit=target.work_units[0],
                    detect_run=run_name,
                    detect_group_path=f"detect_runs/{run_name}",
                ),
            ),
            model=DetectionModelSpec(
                set_id="detect_set",
                run_id="detect_run",
                path=tmp_path / "model.pt",
                sha256="a" * 64,
            ),
            registry_path=tmp_path / "palette_registry.sqlite",
        )
    )

    job = module.fragment.jobs[0]
    command = list(job.command)
    assert job.execution_group is None
    assert "fisheye.utils.run_detection_local_publish" in command
    assert command[command.index("--zarr") + 1] == str(target.analysis_zarr)
    assert command[command.index("--video") + 1] == str(target.work_units[0].video_path)
    assert command[command.index("--run-name") + 1] == run_name
    assert command[command.index("--decode-backend") + 1] == "pynvvc_nv12_rgb"
    resize_index = command.index("--resize-dims")
    assert command[resize_index + 1 : resize_index + 3] == ["640", "640"]
    assert module.outputs.raw_detection_group_paths == (f"detect_runs/{run_name}",)
    assert module.fragment.metadata is not None
    assert module.fragment.metadata["recording_layout"] == "whole_video"
    assert module.fragment.metadata["publication_policy"] == (
        "node_local_complete_run_then_atomic_prfs_publication_v1"
    )


def test_whole_video_raw_detection_fails_closed_on_noncanonical_binding(
    tmp_path: Path,
) -> None:
    target = whole_video_recording_target(
        target_id="whole",
        recording_id="whole:zfixture",
        recording_dir=tmp_path / "recording",
        analysis_zarr=tmp_path / "analysis.zarr",
        video_path=tmp_path / "video.mp4",
        camera_serial="2010093",
    )
    unit = RawDetectionWorkUnitSpec(
        work_unit=target.work_units[0],
        detect_run="detect_fixture",
        detect_group_path="clips/0/detect_runs/detect_fixture",
    )
    kwargs = {
        "workflow_id": "fixture",
        "family": "production_inference",
        "target_label": "whole",
        "target": target,
        "repo": tmp_path / "repo",
        "run_root": tmp_path / "run",
        "work_units": (unit,),
        "model": DetectionModelSpec(
            set_id="set",
            run_id="run",
            path=tmp_path / "model.pt",
            sha256="a" * 64,
        ),
        "registry_path": tmp_path / "registry.sqlite",
    }

    with pytest.raises(ValueError, match="must publish to"):
        RawDetectionFragmentInputs(**kwargs)

    canonical_unit = RawDetectionWorkUnitSpec(
        work_unit=target.work_units[0],
        detect_run="detect_fixture",
        detect_group_path="detect_runs/detect_fixture",
    )
    kwargs["work_units"] = (canonical_unit,)
    kwargs["registry_path"] = None
    with pytest.raises(ValueError, match="explicit registry"):
        RawDetectionFragmentInputs(**kwargs)

    kwargs["registry_path"] = tmp_path / "registry.sqlite"
    kwargs["resume_existing_detections"] = True
    with pytest.raises(ValueError, match="reuse needs a separate validation"):
        RawDetectionFragmentInputs(**kwargs)


def test_raw_detection_workflow_composes_clipped_and_whole_targets(
    tmp_path: Path,
) -> None:
    clipped_target = clipped_recording_target(
        target_id="clipped",
        recording_id="clipped:zfixture",
        recording_dir=tmp_path / "clipped",
        analysis_zarr=tmp_path / "clipped.zarr",
        work_units=(
            {
                "work_unit_id": "clip_0",
                "clip_id": "clip_000000",
                "clip_index": 0,
                "camera_serial": "2010093",
                "source": {"video_path": tmp_path / "clip.mp4"},
            },
        ),
    )
    whole_target = whole_video_recording_target(
        target_id="whole",
        recording_id="whole:zfixture",
        recording_dir=tmp_path / "whole",
        analysis_zarr=tmp_path / "whole.zarr",
        video_path=tmp_path / "whole.mp4",
        camera_serial="2010094",
    )
    model = DetectionModelSpec(
        set_id="set",
        run_id="run",
        path=tmp_path / "model.pt",
        sha256="a" * 64,
    )
    clipped_module = build_raw_detection_fragment(
        RawDetectionFragmentInputs(
            workflow_id="mixed",
            family="production_inference",
            target_label="clipped",
            target=clipped_target,
            repo=tmp_path / "repo",
            run_root=tmp_path / "run",
            work_units=(
                RawDetectionWorkUnitSpec(
                    work_unit=clipped_target.work_units[0],
                    detect_run="detect_clip",
                    detect_group_path="clips/0/detect_runs/detect_clip",
                ),
            ),
            model=model,
        )
    )
    whole_module = build_raw_detection_fragment(
        RawDetectionFragmentInputs(
            workflow_id="mixed",
            family="production_inference",
            target_label="whole",
            target=whole_target,
            repo=tmp_path / "repo",
            run_root=tmp_path / "run",
            work_units=(
                RawDetectionWorkUnitSpec(
                    work_unit=whole_target.work_units[0],
                    detect_run="detect_whole",
                    detect_group_path="detect_runs/detect_whole",
                ),
            ),
            model=model,
            registry_path=tmp_path / "registry.sqlite",
        )
    )

    composed = compose_raw_detection_workflow(
        workflow_id="mixed",
        family="production_inference",
        modules=(clipped_module, whole_module),
    )

    assert [job.job_key for job in composed.topological_jobs()] == [
        "detect_array:clipped",
        "detect:whole",
    ]
    assert composed.metadata is not None
    assert composed.metadata["target_count"] == 2
    assert [
        fragment["metadata"]["recording_layout"]
        for fragment in composed.metadata["fragments"]
    ] == ["clipped_collection", "whole_video"]


def test_detection_fragment_split_preserves_pre_split_job_contract() -> None:
    """Freeze commands/resources/dependencies/outputs from the pre-split builder."""

    rows = tuple(
        {
            "work_unit_id": f"work_{index}",
            "clip_id": f"clip_{index:06d}",
            "clip_index": index,
            "camera_serial": "2010093",
            "source": {"video_path": f"/fixture/clips/clip_{index:06d}.mp4"},
        }
        for index in range(2)
    )
    target = clipped_recording_target(
        target_id="target",
        recording_id="recording:z1",
        recording_dir=Path("/fixture/recording"),
        analysis_zarr=Path("/fixture/analysis.zarr"),
        work_units=rows,
    )
    units = tuple(
        DetectionWorkUnitSpec(
            work_unit=target.work_units[index],
            detect_run=f"detect_{index}",
            detect_group_path=f"clips/{index}/detect_runs/detect_{index}",
            refined_detect_run=f"refined_{index}",
            refined_detect_group_path=(
                f"clips/{index}/refined_detect_runs/refined_{index}"
            ),
        )
        for index in range(2)
    )
    module = build_detection_fragment(
        DetectionFragmentInputs(
            workflow_id="detection_fixture",
            family="clipped_inference",
            target_label="fixture_target",
            target=target,
            repo=Path("/fixture/repo"),
            run_root=Path("/fixture/run"),
            detection_plan_path=Path("/fixture/run/detection_plan.json"),
            collection_id="collection_fixture",
            quality_source_run="quality_source_fixture",
            quality_run="quality_fixture",
            work_units=units,
            model=DetectionModelSpec(
                set_id="set",
                run_id="run",
                path=Path("/fixture/model.pt"),
                sha256="a" * 64,
            ),
            detect_array_concurrency=2,
            refine_bundle_concurrency=2,
        )
    )
    payload = {
        "jobs": [job.to_json() for job in module.jobs],
        "outputs": module.outputs.to_json(),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))

    # Captured from the monolithic builder immediately before the fragment
    # split. Fragment metadata is intentionally excluded; all executable job
    # commands, resources, dependencies, task envelopes, and outputs are in it.
    assert hashlib.sha256(encoded.encode("utf-8")).hexdigest() == (
        "183693e320e885eda3cf56293383797fce7b7e1411054c9b006275abbb1c9b85"
    )


def test_materialized_dry_run_is_immutable_and_has_no_submission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = _build_fixture_plan(tmp_path, monkeypatch)
    first = workflow.materialize_plan_bundle(plan)
    second = workflow.materialize_plan_bundle(plan)

    assert first == second
    assert (plan.run_root / "plan.json").is_file()
    assert (plan.run_root / "lsf_plan.json").is_file()
    assert not (plan.run_root / "lsf_submission.json").exists()
    assert first["models"]["detection"]["run_id"] == "detect_run"
    assert first["models"]["pose"]["run_id"] == "pose_run"
    assert first["models"]["subject_masks"]["run_id"] == "mask_run"


def test_encoded_mask_packages_add_global_grid_and_join(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = _build_fixture_plan(tmp_path, monkeypatch, encoded_mask_packages=True)
    jobs = {job.job_key: job for job in plan.lsf_workflow.jobs}
    target = plan.target_plans[0]
    target_safe = workflow.safe_component(
        str(target["target_id"]), default="target", max_length=56
    )
    grid_key = f"mask_grid:{target_safe}"
    package_key = f"mask_package_array:{target_safe}"
    publish_key = f"mask_publish:{target_safe}"

    assert plan.encoded_mask_packages is True
    assert len(plan.lsf_workflow.jobs) == 22
    assert jobs[grid_key].dependency.upstream_job_keys == (
        f"keypoint_finalize:{target_safe}",
    )
    assert jobs[package_key].dependency.upstream_job_keys == (
        f"subject_masks_array:{target_safe}",
        f"keypoint_refine:{target_safe}",
        grid_key,
    )
    package_task = _execution_tasks(jobs[package_key])[
        f"mask_package:{target_safe}:clip_000000"
    ]
    assert "--global-mask-grid-manifest" in package_task.command
    assert "--require-production-proof" in package_task.command
    assert f"mask_import:{target_safe}" not in jobs
    assert "--refined-package" in jobs[publish_key].command
    assert jobs[publish_key].dependency.upstream_job_keys == (package_key,)


def test_encoded_chunk_canary_serializes_prfs_ab_imports(tmp_path: Path) -> None:
    packages = [tmp_path / "clip_000000.tar.gz", tmp_path / "clip_000001.tar.gz"]
    for package in packages:
        package.write_bytes(b"package")
    plan = encoded_canary.build_plan(
        package_paths=packages,
        run_root=tmp_path / "canary_run",
        encoded_package_dir=tmp_path / "encoded_packages",
        canary_label="encoded_canary",
        repo=tmp_path / "repo",
    )
    jobs = {job.job_key: job for job in plan.workflow.jobs}

    assert len(plan.workflow.jobs) == 6
    assert jobs["baseline_import"].dependency.upstream_job_keys == ("prepare",)
    assert jobs["encoded_import"].dependency.upstream_job_keys == (
        "baseline_import",
        "convert:00",
        "convert:01",
    )
    assert jobs["validate"].dependency.upstream_job_keys == ("encoded_import",)
    assert jobs["baseline_import"].resources.queue == "local"
    assert jobs["encoded_import"].resources.queue == "local"
    assert "--encoded-copy-workers" in jobs["encoded_import"].command
    assert plan.workflow.family == encoded_canary.CANARY_FAMILY
    assert all(
        job.command[job.command.index("--family") + 1] == encoded_canary.CANARY_FAMILY
        for job in plan.workflow.jobs
    )


def test_resume_plan_revalidates_detections_on_cpu_and_preserves_dependencies(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _build_fixture_plan(tmp_path, monkeypatch, resume_existing_detections=True)
    jobs = {job.job_key: job for job in plan.lsf_workflow.jobs}
    target = plan.target_plans[0]
    target_safe = workflow.safe_component(
        str(target["target_id"]), default="target", max_length=56
    )
    clip_id = "clip_000000"
    detect_array = jobs[f"detect_artifact_array:{target_safe}"]
    detect = _execution_tasks(detect_array)[f"detect_artifact:{target_safe}:{clip_id}"]
    refine_bundle = jobs[f"detect_refine_bundle:{target_safe}"]

    assert plan.resume_existing_detections is True
    assert len(target["detection_resume_preflight"]) == 22
    assert detect.metadata["stage"] == "detect_artifact_reuse"
    assert detect_array.resources.queue == "short"
    assert detect_array.resources.gpus == 0
    assert "--reuse-existing" in detect.command
    assert refine_bundle.dependency.upstream_job_keys == (
        f"detect_quality:{target_safe}",
    )


def test_same_dag_plans_multiple_recordings_with_bounded_target_concurrency(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _build_fixture_plan(
        tmp_path,
        monkeypatch,
        target_count=2,
        max_active_targets=1,
    )
    jobs = {job.job_key: job for job in plan.lsf_workflow.jobs}
    first, second = plan.target_plans
    first_safe = workflow.safe_component(
        str(first["target_id"]), default="target", max_length=56
    )
    second_safe = workflow.safe_component(
        str(second["target_id"]), default="target", max_length=56
    )

    assert len(plan.targets) == 2
    assert len(plan.lsf_workflow.jobs) == 40
    assert jobs[f"detect_artifact_array:{first_safe}"].dependency is None
    assert jobs[
        f"detect_artifact_array:{second_safe}"
    ].dependency.upstream_job_keys == (f"validate:{first_safe}",)
    assert jobs["registry_finalize"].dependency.upstream_job_keys == (
        f"validate:{first_safe}",
        f"validate:{second_safe}",
    )
    fragments = {
        fragment["fragment_id"]: fragment
        for fragment in plan.lsf_workflow.to_json()["metadata"]["fragments"]
    }
    assert fragments[f"native_detection:{first_safe}"]["requires"] == []
    assert fragments[f"native_detection:{second_safe}"]["requires"] == [
        f"validated_analysis:{first_safe}"
    ]
    assert fragments[f"detection_postprocess:{first_safe}"]["requires"] == [
        f"canonical_detection:{first_safe}"
    ]
    assert fragments[f"detection_postprocess:{second_safe}"]["requires"] == [
        f"canonical_detection:{second_safe}"
    ]
    assert fragments["campaign_finalize"]["requires"] == [
        f"validated_analysis:{first_safe}",
        f"validated_analysis:{second_safe}",
    ]


def test_keypoint_recovery_reuses_cache_and_raw_masks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _build_fixture_plan(
        tmp_path,
        monkeypatch,
        resume_existing_detections=True,
        subject_mask_publication_profile=(
            workflow.SUBJECT_MASK_PUBLICATION_STREAMING_ROLLBACK
        ),
    )
    source_plan = tmp_path / "source_plan.json"
    _write_json(source_plan, source.to_json())
    monkeypatch.setattr(
        recovery,
        "prepare_keypoint_recovery",
        lambda *_args, **_kwargs: {"status": "ok", "clip_count": 22},
    )

    recovery_root = tmp_path / "recovery"
    plan = recovery.build_plan(
        source_plan_path=source_plan,
        run_root=recovery_root,
        recovery_label="sleepyfish_keypoint_recovery",
    )
    jobs = {job.job_key: job for job in plan.workflow.jobs}
    target = source.target_plans[0]
    target_safe = workflow.safe_component(
        str(target["target_id"]), default="target", max_length=56
    )
    keypoint_key = f"keypoints_array:{target_safe}"
    refine_key = f"keypoint_refine:{target_safe}"
    package_key = f"mask_package_array:{target_safe}"

    assert len(plan.workflow.jobs) == 9
    assert jobs[keypoint_key].dependency.upstream_job_keys == (
        "prepare_keypoint_recovery",
    )
    assert jobs[package_key].dependency.upstream_job_keys == (refine_key,)
    assert jobs["nrs_cleanup"].dependency.upstream_job_keys == ("registry_finalize",)
    assert jobs[f"mask_import:{target_safe}"].resources.queue == "local"
    assert {
        str(job.metadata["stage"])
        for job in plan.workflow.jobs
        if job.metadata is not None
    }.isdisjoint(
        {"detect", "detect_reuse", "detect_refine", "roi_cache", "subject_masks"}
    )
    keypoint_task = _execution_tasks(jobs[keypoint_key])[
        f"keypoints:{target_safe}:clip_000000"
    ]
    assert any(str(recovery_root) in value for value in keypoint_task.command)
    assert all(str(source.run_root) not in value for value in keypoint_task.command)
    assert plan.payload["targets"] == source.to_json()["targets"]


def test_keypoint_recovery_republishes_receipt_composed_mask_packages(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _build_fixture_plan(tmp_path, monkeypatch)
    source_plan = tmp_path / "source_receipt_plan.json"
    _write_json(source_plan, source.to_json())
    monkeypatch.setattr(
        recovery,
        "prepare_keypoint_recovery",
        lambda *_args, **_kwargs: {"status": "ok", "clip_count": 22},
    )

    plan = recovery.build_plan(
        source_plan_path=source_plan,
        run_root=tmp_path / "receipt_recovery",
        recovery_label="sleepyfish_receipt_keypoint_recovery",
    )
    jobs = {job.job_key: job for job in plan.workflow.jobs}
    target = source.target_plans[0]
    target_safe = workflow.safe_component(
        str(target["target_id"]), default="target", max_length=56
    )
    package_key = f"mask_package_array:{target_safe}"
    publish_key = f"mask_publish:{target_safe}"

    assert f"mask_import:{target_safe}" not in jobs
    assert publish_key in jobs
    assert jobs[publish_key].dependency.upstream_job_keys == (package_key,)
    assert "fisheye.cluster.subject_masks.publish_receipt_composed_bundle" in (
        jobs[publish_key].command
    )
    assert jobs[f"validate:{target_safe}"].dependency.upstream_job_keys == (
        publish_key,
    )


def test_detect_quality_recovery_reuses_source_and_clones_complete_dag_tail(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _build_fixture_plan(
        tmp_path,
        monkeypatch,
        subject_mask_publication_profile=(
            workflow.SUBJECT_MASK_PUBLICATION_STREAMING_ROLLBACK
        ),
    )
    workflow.materialize_plan_bundle(source)
    source_plan = source.run_root / "plan.json"
    source_payload = json.loads(source_plan.read_text(encoding="utf-8"))
    for prior_job in source_payload["lsf_workflow"]["jobs"]:
        if str(prior_job["job_key"]).startswith("detect_refine_bundle:"):
            prior_job["resources"]["walltime"] = "2:00"
    _write_json(source_plan, source_payload)
    target = source.target_plans[0]
    source_metadata = (
        Path(str(target["analysis_zarr"]))
        / "detect_collection_sources"
        / str(target["detect_quality_source_run"])
        / "zarr.json"
    )
    _write_json(
        source_metadata,
        {
            "zarr_format": 3,
            "node_type": "group",
            "attributes": {
                "schema_id": "palette.clipped_detect_quality_source.v1",
                "palette_run_completion_status": "complete",
            },
        },
    )

    recovery_root = tmp_path / "quality_recovery"
    plan = quality_recovery.build_plan(
        source_plan_path=source_plan,
        run_root=recovery_root,
        recovery_label="sleepyfish_quality_recovery",
        repo=source.repo,
    )
    jobs = {job.job_key: job for job in plan.workflow.jobs}
    target_safe = workflow.safe_component(
        str(target["target_id"]), default="target", max_length=56
    )
    repair_key = f"detect_quality_source_geometry_repair:{target_safe}"
    quality_key = f"detect_quality:{target_safe}"

    assert len(plan.workflow.jobs) == len(source.lsf_workflow.jobs) - 2
    assert f"detect_array:{target_safe}" not in jobs
    assert f"detect_quality_source:{target_safe}" not in jobs
    assert jobs[quality_key].dependency.upstream_job_keys == (repair_key,)
    assert jobs[f"detect_refine_bundle:{target_safe}"].dependency.upstream_job_keys == (
        quality_key,
    )
    assert jobs[f"detect_refine_bundle:{target_safe}"].resources.walltime == "1:00"
    assert jobs["nrs_cleanup"].dependency.upstream_job_keys == ("registry_finalize",)
    assert (
        plan.payload["detect_quality_recovery"]["source_array_payload_rewritten"]
        is False
    )
    assert (
        plan.payload["targets"][0]["detect_quality_source_run"]
        == target["detect_quality_source_run"]
    )
    quality_inner = quality_recovery._inner_command(jobs[quality_key].to_json())
    assert all(str(source.run_root) not in value for value in quality_inner)
    assert all(
        str(source.repo) not in value or str(plan.repo) in value
        for value in quality_inner
    )

    quality_recovery.materialize_plan(plan)
    assert plan.recovery_detection_plan.is_file()
    assert json.loads(
        plan.recovery_detection_plan.read_text(encoding="utf-8")
    ) == json.loads(plan.source_detection_plan.read_text(encoding="utf-8"))

    _write_json(
        source_metadata,
        {
            "zarr_format": 3,
            "node_type": "group",
            "attributes": {
                "schema_id": "palette.clipped_detect_quality_source.v1",
                "palette_run_completion_status": "complete",
                "source_row_count": 4,
                "recording_frame_count": 6,
                "source_video_width": 4512,
                "source_video_height": 4512,
                "full_frame_geometry_repair": {"status": "complete"},
            },
        },
    )
    quality_path = Path(str(target["analysis_zarr"])) / str(
        target["detect_quality_group_path"]
    )
    _write_json(
        quality_path / "zarr.json",
        {
            "zarr_format": 3,
            "node_type": "group",
            "attributes": {
                "palette_run_completion_status": "complete",
                "schema_id": "palette.detect_quality_collection.v2",
                "source_detection_group_path": target[
                    "detect_quality_source_group_path"
                ],
                "source_row_count": 4,
                "recording_frame_count": 6,
                "source_video_width": 4512,
                "source_video_height": 4512,
                "collection_quality_validation": {
                    "status": "complete",
                    "instance_key_exact": True,
                    "instance_key_unique": True,
                    "arrays_indexed_sharded": True,
                    "trace_ranges_complete_nonoverlapping": True,
                    "source_rows_canonical_frame_order": True,
                    "row_count": 4,
                    "recording_frame_count": 6,
                },
            },
        },
    )
    for name, shape, dtype in (
        ("instance_key", [4], "uint64"),
        ("detection_quality_labels", [4], "int8"),
        ("quality_flags", [6], "int8"),
    ):
        _write_json(
            quality_path / name / "zarr.json",
            {
                "zarr_format": 3,
                "node_type": "array",
                "shape": shape,
                "data_type": dtype,
            },
        )

    continuation = quality_recovery.build_plan(
        source_plan_path=source_plan,
        run_root=tmp_path / "quality_continuation",
        recovery_label="sleepyfish_quality_continuation",
        repo=source.repo,
        reuse_complete_quality=True,
    )
    continuation_jobs = {job.job_key: job for job in continuation.workflow.jobs}
    refine_key = f"detect_refine_bundle:{target_safe}"
    assert len(continuation.workflow.jobs) == len(source.lsf_workflow.jobs) - 4
    assert repair_key not in continuation_jobs
    assert quality_key not in continuation_jobs
    assert continuation_jobs[refine_key].dependency is None
    assert continuation_jobs[refine_key].resources.walltime == "1:00"
    assert (
        continuation.payload["detect_quality_recovery"]["reused_complete_quality"]
        is True
    )
    assert (
        continuation.payload["detect_quality_recovery"]["preflight"][
            "quality_validation"
        ]["instance_key_exact"]
        is True
    )


def test_import_recovery_reuses_packages_on_long_queue(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _build_fixture_plan(
        tmp_path,
        monkeypatch,
        subject_mask_publication_profile=(
            workflow.SUBJECT_MASK_PUBLICATION_STREAMING_ROLLBACK
        ),
    )
    source_plan = tmp_path / "source_import_plan.json"
    _write_json(source_plan, source.to_json())
    monkeypatch.setattr(
        import_recovery,
        "_preflight",
        lambda _source: {"status": "ok", "target_count": 1},
    )

    recovery_root = tmp_path / "import_recovery"
    plan = import_recovery.build_plan(
        source_plan_path=source_plan,
        run_root=recovery_root,
        recovery_label="sleepyfish_import_recovery",
    )
    jobs = {job.job_key: job for job in plan.workflow.jobs}
    target = source.target_plans[0]
    target_safe = workflow.safe_component(
        str(target["target_id"]), default="target", max_length=56
    )
    import_key = f"mask_import:{target_safe}"
    validation_key = f"validate:{target_safe}"

    assert len(plan.workflow.jobs) == 4
    assert jobs[import_key].dependency is None
    assert jobs[import_key].resources.queue == "local"
    assert jobs[import_key].resources.walltime == "3:00"
    assert "--overwrite" in jobs[import_key].command
    assert jobs[validation_key].dependency.upstream_job_keys == (import_key,)
    assert jobs["registry_finalize"].dependency.upstream_job_keys == (validation_key,)
    assert jobs["nrs_cleanup"].dependency.upstream_job_keys == ("registry_finalize",)


def test_import_recovery_can_resume_after_complete_import_validation_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _build_fixture_plan(
        tmp_path,
        monkeypatch,
        subject_mask_publication_profile=(
            workflow.SUBJECT_MASK_PUBLICATION_STREAMING_ROLLBACK
        ),
    )
    source_plan = tmp_path / "source_complete_import_plan.json"
    _write_json(source_plan, source.to_json())
    monkeypatch.setattr(
        import_recovery,
        "_preflight",
        lambda _source, **_kwargs: {
            "status": "ok",
            "target_count": 1,
            "import_action": "reuse_complete_output",
        },
    )

    plan = import_recovery.build_plan(
        source_plan_path=source_plan,
        run_root=tmp_path / "validation_only_recovery",
        recovery_label="sleepyfish_validation_only_recovery",
        validate_existing_complete_import=True,
    )
    jobs = {job.job_key: job for job in plan.workflow.jobs}
    target_safe = workflow.safe_component(
        str(source.target_plans[0]["target_id"]), default="target", max_length=56
    )
    validation_key = f"validate:{target_safe}"

    assert len(plan.workflow.jobs) == 3
    assert f"mask_import:{target_safe}" not in jobs
    assert jobs[validation_key].dependency is None
    assert jobs["registry_finalize"].dependency.upstream_job_keys == (validation_key,)
    assert jobs["nrs_cleanup"].dependency.upstream_job_keys == ("registry_finalize",)
    assert plan.payload["import_recovery"]["validation_only_complete_import"] is True


def test_import_recovery_can_convert_v1_packages_to_encoded_v2(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _build_fixture_plan(
        tmp_path,
        monkeypatch,
        subject_mask_publication_profile=(
            workflow.SUBJECT_MASK_PUBLICATION_STREAMING_ROLLBACK
        ),
    )
    source_plan = tmp_path / "source_import_plan.json"
    _write_json(source_plan, source.to_json())
    monkeypatch.setattr(
        import_recovery,
        "_preflight",
        lambda _source: {"status": "ok", "target_count": 1},
    )

    plan = import_recovery.build_plan(
        source_plan_path=source_plan,
        run_root=tmp_path / "encoded_recovery",
        recovery_label="sleepyfish_encoded_recovery",
        convert_packages_v2=True,
    )
    jobs = {job.job_key: job for job in plan.workflow.jobs}
    target = source.target_plans[0]
    target_safe = workflow.safe_component(
        str(target["target_id"]), default="target", max_length=56
    )
    grid_key = f"mask_grid:{target_safe}"
    conversion_key = f"mask_package_v2:{target_safe}:clip_000000"
    import_key = f"mask_import:{target_safe}"

    assert len(plan.workflow.jobs) == 27
    assert jobs[conversion_key].dependency.upstream_job_keys == (grid_key,)
    assert (
        jobs[import_key]
        .dependency.upstream_job_keys[0]
        .startswith(f"mask_package_v2:{target_safe}:")
    )
    assert len(jobs[import_key].dependency.upstream_job_keys) == 22
    assert "--encoded-copy-workers" in jobs[import_key].command
    assert all(
        "/encoded_v2/" in value
        for index, value in enumerate(jobs[import_key].command)
        if jobs[import_key].command[index - 1] == "--package"
    )


def test_existing_detection_resume_preflight_requires_exact_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = _target(tmp_path)
    binding = workflow.ModelBinding(
        "detect",
        "detect_set",
        "detect_run",
        (tmp_path / "model.pt").resolve(),
        "a" * 64,
    )
    clip = _detection_plan(target, "campaign")["work_units"][0]
    planned_clip = {
        "clip_id": clip["clip_id"],
        "clip_index": clip["clip_index"],
        "camera_serial": clip["camera_serial"],
        "video_path": clip["source"]["video_path"],
        "detect_run": clip["run_names"]["detect"],
        "detect_group_path": clip["zarr_paths"]["detect_target_group_path"],
    }
    group_metadata = (
        target.analysis_zarr / planned_clip["detect_group_path"] / "zarr.json"
    )
    provenance = {
        "command": "fisheye.utils.run_detection_artifact",
        "params": {
            "run_name": planned_clip["detect_run"],
            "video_path": planned_clip["video_path"],
            "target_zarr": str(target.analysis_zarr),
            "model_path": str(binding.path),
            "model_sha256": binding.sha256,
            "model_registry_set_id": binding.set_id,
            "model_registry_run_id": binding.run_id,
            "clip_context": {
                "workflow_id": "campaign",
                "recording_id": target.recording_id,
                "clip_id": planned_clip["clip_id"],
                "clip_index": planned_clip["clip_index"],
                "camera_serial": planned_clip["camera_serial"],
            },
        },
        "input_run_ids": {
            "model_registry_set_id": binding.set_id,
            "model_registry_run_id": binding.run_id,
        },
        "input_artifacts": [
            {
                "role": "detect_model",
                "path": str(binding.path),
                "sha256": binding.sha256,
            }
        ],
    }
    _write_json(
        group_metadata,
        {
            "zarr_format": 3,
            "node_type": "group",
            "attributes": {
                "palette_run_completion_status": "complete",
                "run_provenance": provenance,
            },
        },
    )
    monkeypatch.setattr(
        workflow,
        "validate_imported_run_group",
        lambda **_kwargs: {
            "status": "ok",
            "receipt_path": "/archive/.imports/detect_run.json",
        },
    )

    report = workflow._validate_existing_detection_for_resume(
        target=target,
        workflow_id="campaign",
        clip=planned_clip,
        binding=binding,
    )
    assert report["status"] == "ok"
    assert report["model_sha256"] == "a" * 64

    provenance["params"]["clip_context"]["workflow_id"] = "wrong_campaign"
    _write_json(
        group_metadata,
        {
            "zarr_format": 3,
            "node_type": "group",
            "attributes": {
                "palette_run_completion_status": "complete",
                "run_provenance": provenance,
            },
        },
    )
    with pytest.raises(ValueError, match="provenance mismatch"):
        workflow._validate_existing_detection_for_resume(
            target=target,
            workflow_id="campaign",
            clip=planned_clip,
            binding=binding,
        )


def test_instance_key_validation_rejects_duplicates() -> None:
    run = {"instance_key": np.asarray([11, 12, 12], dtype=np.uint64)}
    with pytest.raises(RuntimeError, match="duplicate instance_key"):
        _instance_keys(run, label="fixture")


@pytest.mark.parametrize(
    "plan_schema",
    (workflow.LEGACY_PLAN_SCHEMA, workflow.PLAN_SCHEMA),
)
def test_cleanup_is_confined_and_requires_registry_success(
    tmp_path: Path,
    plan_schema: str,
) -> None:
    cache_root = tmp_path / "cache"
    package_root = tmp_path / "packages"
    cache_dir = cache_root / "campaign" / "target"
    package_dir = package_root / "campaign" / "target"
    cache_dir.mkdir(parents=True)
    package_dir.mkdir(parents=True)
    run_root = tmp_path / "run"
    plan_path = run_root / "plan.json"
    _write_json(
        plan_path,
        {
            "schema": plan_schema,
            "run_root": str(run_root),
            "targets": [
                {
                    "target_id": "target",
                    "cache_dir": str(cache_dir),
                    "package_dir": str(package_dir),
                }
            ],
        },
    )
    _write_json(
        run_root / "registry" / "reconcile.json",
        {"status": "ok", "registry_integrity": "ok", "target_count": 1},
    )

    report = cleanup(
        plan_path,
        apply=True,
        cache_root=cache_root,
        package_root=package_root,
    )
    assert report["removed_count"] == 2
    assert not cache_dir.exists()
    assert not package_dir.exists()


def test_cleanup_refuses_allowed_root_itself(tmp_path: Path) -> None:
    cache_root = tmp_path / "cache"
    package_root = tmp_path / "packages"
    cache_root.mkdir()
    (package_root / "campaign").mkdir(parents=True)
    run_root = tmp_path / "run"
    plan_path = run_root / "plan.json"
    _write_json(
        plan_path,
        {
            "schema": workflow.PLAN_SCHEMA,
            "run_root": str(run_root),
            "targets": [
                {
                    "target_id": "target",
                    "cache_dir": str(cache_root),
                    "package_dir": str(package_root / "campaign"),
                }
            ],
        },
    )
    _write_json(
        run_root / "registry" / "reconcile.json",
        {"status": "ok", "registry_integrity": "ok", "target_count": 1},
    )
    with pytest.raises(ValueError, match="Refusing cache_dir cleanup"):
        cleanup(
            plan_path,
            apply=False,
            cache_root=cache_root,
            package_root=package_root,
        )


def test_ssh_runner_uses_poller_only_for_bsub(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return subprocess.CompletedProcess(command, 0, "Job <123> is submitted", "")

    monkeypatch.setattr(workflow.subprocess, "run", fake_run)
    runner = workflow.build_ssh_bsub_runner("login1-citrus-poller")
    result = runner(
        ["bsub", "-J", "fixture", "scripts/py", "-m", "worker"],
        cwd="/groups/repo",
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0
    assert captured["command"][:4] == [
        "ssh",
        "-o",
        "BatchMode=yes",
        "login1-citrus-poller",
    ]
    assert str(captured["command"][4]).startswith("cd /groups/repo && bsub ")
    with pytest.raises(ValueError, match="only bsub"):
        runner(["scripts/py", "-m", "worker"], cwd="/groups/repo")
