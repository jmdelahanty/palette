from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import zarr

from fisheye.cluster.subject_masks import full_duration_canary as canary
from fisheye.cluster.lsf import LsfExecutionMode


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _reference_archives(tmp_path: Path) -> tuple[Path, Path, str, str]:
    crop_path = tmp_path / "crop.zarr"
    crop_root = zarr.open_group(str(crop_path), mode="w", zarr_format=3)
    crop = crop_root.create_group("crop_runs").create_group("crop_v2")
    crop.attrs["run_manifest"] = {
        "schema_id": "crop",
        "schema_version": 1,
        "payload_digest": "1" * 64,
    }
    frames = np.asarray([0, 0, 2, 3], dtype=np.int64)
    offsets = np.asarray([0, 2, 2, 3, 4], dtype=np.int64)
    crop.create_array("frame_row_offsets", data=offsets)
    crop.create_array("source_acquisition_frame_index", data=frames)
    crop.create_array("frame_indices", data=frames)
    crop.create_array("instance_key", data=np.arange(10, 14, dtype=np.uint64))
    crop.create_array(
        "source_crop_xywh",
        data=np.tile(np.asarray([[1, 2, 8, 8]], dtype=np.int32), (4, 1)),
    )
    crop.create_array(
        "roi_coordinates_full",
        data=np.tile(np.asarray([[1, 2]], dtype=np.int32), (4, 1)),
    )
    crop.create_array("source_row_signature", data=np.zeros((4, 32), dtype=np.uint8))

    keypoint_path = tmp_path / "keypoints.zarr"
    keypoint_root = zarr.open_group(str(keypoint_path), mode="w", zarr_format=3)
    keypoints = keypoint_root.create_group("refined_keypoints_runs").create_group(
        "refined_v2"
    )
    keypoints.attrs["run_manifest"] = {
        "schema_id": "refined_keypoints",
        "schema_version": 1,
        "payload_digest": "2" * 64,
    }
    keypoints.create_array("source_crop_row_ids", data=np.arange(4, dtype=np.int64))
    keypoints.create_array("instance_key", data=np.arange(10, 14, dtype=np.uint64))
    return crop_path, keypoint_path, "crop_v2", "refined_v2"


def _prepare(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    crop_path, keypoint_path, crop_run, keypoint_run = _reference_archives(tmp_path)
    recording = tmp_path / "recording"
    first = recording / "clips" / "clip_000000" / "first.mp4"
    second = recording / "clips" / "clip_000001" / "second.mp4"
    first.parent.mkdir(parents=True)
    second.parent.mkdir(parents=True)
    first.write_bytes(b"first-video")
    second.write_bytes(b"second-video")
    first_manifest = first.parent / "clip_manifest.json"
    second_manifest = second.parent / "clip_manifest.json"
    _write_json(first_manifest, {"clip_id": "clip_000000"})
    _write_json(second_manifest, {"clip_id": "clip_000001"})
    clip_index = recording / "recording_clip_index.json"
    _write_json(
        clip_index,
        {
            "status": "ok",
            "mode": "materialized_stream_copy",
            "recording_id": "recording-1",
            "clip_count": 2,
            "checks": [{"code": "fixture", "status": "ok"}],
            "clips": [
                {
                    "clip_id": "clip_000000",
                    "clip_index": 0,
                    "status": "materialized",
                    "camera_serial": "camera-1",
                    "actual_start_frame": 0,
                    "end_frame_exclusive": 2,
                    "frame_count": 2,
                    "video_path": str(first.relative_to(recording)),
                    "clip_manifest_path": str(first_manifest.relative_to(recording)),
                },
                {
                    "clip_id": "clip_000001",
                    "clip_index": 1,
                    "status": "materialized",
                    "camera_serial": "camera-1",
                    "actual_start_frame": 2,
                    "end_frame_exclusive": 4,
                    "frame_count": 2,
                    "video_path": str(second.relative_to(recording)),
                    "clip_manifest_path": str(second_manifest.relative_to(recording)),
                },
            ],
        },
    )
    model = tmp_path / "model.pt"
    model.write_bytes(b"model")
    monkeypatch.setattr(canary, "validate_crop_run_manifest", lambda _value: ())
    monkeypatch.setattr(
        canary, "validate_refined_keypoint_run_manifest", lambda _value: ()
    )
    monkeypatch.setattr(
        canary,
        "_repo_identity",
        lambda _repo, require_clean=True: {
            "path": str(tmp_path / "repo"),
            "commit": "a" * 40,
            "branch": "test",
            "dirty": False,
        },
    )
    return canary.prepare_canary(
        run_root=tmp_path / ".palette_benchmarks" / "canary",
        repo=tmp_path / "repo",
        source_crop_zarr=crop_path,
        crop_run=crop_run,
        source_refined_keypoint_zarr=keypoint_path,
        refined_keypoint_run=keypoint_run,
        model_path=model,
        model_sha256=hashlib.sha256(model.read_bytes()).hexdigest(),
        recording_id="recording-1",
        recording_dir=recording,
        clip_index=clip_index,
        run_label="fixture_v1",
    )


def test_prepare_freezes_exact_clip_row_coverage_and_reference_copies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = _prepare(tmp_path, monkeypatch)

    assert plan["recording"]["n_frames"] == 4
    assert plan["recording"]["n_rows"] == 4
    assert plan["schema_version"] == canary.PLAN_SCHEMA_VERSION
    assert plan["execution"]["inference"]["synchronized_stage_profiling"] is False
    assert plan["execution"]["inference"]["gpu_runtime_telemetry"] == {
        "enabled": True,
        "sample_interval_seconds": 1,
        "schema_id": canary.GPU_RUNTIME_TELEMETRY_SCHEMA_ID,
        "schema_version": canary.GPU_RUNTIME_TELEMETRY_SCHEMA_VERSION,
        "identity_policy": canary.GPU_RUNTIME_TELEMETRY_IDENTITY_POLICY,
    }
    assert [window["row_count"] for window in plan["windows"]] == [2, 2]
    assert [window["row_start"] for window in plan["windows"]] == [0, 2]
    assert [window["row_stop"] for window in plan["windows"]] == [2, 4]
    assert plan["safety"] == {
        "production_registry_used": False,
        "production_selector_mutation_allowed": False,
        "bundle_activation_allowed": False,
        "all_outputs_below_run_root": True,
        "worker_writes_are_node_local_until_atomic_bundle_publish": True,
        "window_rows_are_exact_nonoverlapping_complete": True,
    }
    target = Path(plan["references"]["analysis_zarr"])
    assert (target / "crop_runs" / "crop_v2" / "zarr.json").is_file()
    assert (target / "refined_keypoints_runs" / "refined_v2" / "zarr.json").is_file()
    loaded = canary.load_plan(Path(plan["run_root"]) / "plan.json")
    assert loaded["plan_digest"] == plan["plan_digest"]


def test_prepare_rejects_a_gap_in_real_video_windows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    crop_path, keypoint_path, crop_run, keypoint_run = _reference_archives(tmp_path)
    del keypoint_path, crop_run, keypoint_run
    recording = tmp_path / "recording"
    video = recording / "clip.mp4"
    recording.mkdir()
    video.write_bytes(b"video")
    clip_manifest = recording / "clip_manifest.json"
    _write_json(clip_manifest, {"clip_id": "bad"})
    clip_index = recording / "recording_clip_index.json"
    _write_json(
        clip_index,
        {
            "status": "ok",
            "mode": "materialized_stream_copy",
            "recording_id": "recording-1",
            "clip_count": 1,
            "checks": [{"code": "fixture", "status": "ok"}],
            "clips": [
                {
                    "clip_id": "bad",
                    "clip_index": 0,
                    "status": "materialized",
                    "camera_serial": "camera-1",
                    "actual_start_frame": 1,
                    "end_frame_exclusive": 4,
                    "frame_count": 3,
                    "video_path": "clip.mp4",
                    "clip_manifest_path": "clip_manifest.json",
                }
            ],
        },
    )
    crop = zarr.open_group(str(crop_path), mode="r")["crop_runs/crop_v2"]
    with pytest.raises(ValueError, match="contiguously cover"):
        canary._resolve_windows(
            n_frames=4,
            frame_offsets=np.asarray(crop["frame_row_offsets"][:], dtype=np.int64),
            frame_indices=np.asarray(
                crop["source_acquisition_frame_index"][:], dtype=np.int64
            ),
            recording_dir=recording,
            clip_index=clip_index,
            whole_video=None,
            recording_id="recording-1",
            camera_identity=None,
            workflow_id="fixture_v1",
        )


def test_whole_video_is_one_real_window_with_the_same_row_contract(
    tmp_path: Path,
) -> None:
    video = tmp_path / "whole.mp4"
    video.write_bytes(b"whole")
    windows, source = canary._resolve_windows(
        n_frames=4,
        frame_offsets=np.asarray([0, 2, 2, 3, 4], dtype=np.int64),
        frame_indices=np.asarray([0, 0, 2, 3], dtype=np.int64),
        recording_dir=None,
        clip_index=None,
        whole_video=video,
        recording_id="recording-1",
        camera_identity="camera-1",
        workflow_id="fixture_v1",
    )

    assert source["mode"] == "whole_recording"
    assert len(source["window_index_sha256"]) == 64
    assert [(value["row_start"], value["row_stop"]) for value in windows] == [(0, 4)]


def test_cluster_file_identity_ignores_mount_local_device_and_inode() -> None:
    planned = {
        "path": "/groups/recording/clip.mp4",
        "size_bytes": 100,
        "mtime_ns": 123,
        "device": 84,
        "inode": 1000,
    }
    compute_node = {
        **planned,
        "device": 101,
        "inode": 2000,
    }

    assert canary._same_cluster_file_identity(compute_node, planned)
    assert not canary._same_cluster_file_identity(
        {**compute_node, "size_bytes": 101}, planned
    )
    assert not canary._same_cluster_file_identity(
        {**compute_node, "mtime_ns": 124}, planned
    )
    assert not canary._same_cluster_file_identity(
        {**compute_node, "path": "/groups/recording/other.mp4"}, planned
    )


def test_scratch_window_binding_supplies_geometry_video_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = _prepare(tmp_path, monkeypatch)
    root = zarr.open_group(str(plan["references"]["analysis_zarr"]), mode="r+")
    window = plan["windows"][0]

    binding = canary._bind_scratch_window_video(
        root,
        crop_run=str(plan["references"]["crop"]["run"]),
        window=window,
    )

    crop = root[f"crop_runs/{plan['references']['crop']['run']}"]
    assert crop.attrs["source_video_path"] == window["source_video_path"]
    assert binding["scope"] == "node_local_inference_reference"
    assert binding["declared_source_video_path"] == window["source_video_path"]
    assert binding["previous_declared_source_video_path"] is None


def test_lsf_workflow_keeps_inference_refinement_and_publication_separate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = _prepare(tmp_path, monkeypatch)
    workflow = canary.build_lsf_workflow(
        plan_path=Path(plan["run_root"]) / "plan.json",
        gpu_concurrency=2,
        cpu_concurrency=1,
    )

    assert [job.job_key for job in workflow.topological_jobs()] == [
        "subject_mask_inference_array",
        "subject_mask_refinement_array",
        "subject_mask_recording_publication",
    ]
    inference, refinement, publication = workflow.jobs
    assert inference.execution_group is not None
    assert inference.execution_group.mode is LsfExecutionMode.ARRAY
    assert len(inference.execution_group.tasks) == 2
    assert inference.resources.gpus == 1
    assert refinement.dependency is not None
    assert refinement.dependency.upstream_job_keys == ("subject_mask_inference_array",)
    assert refinement.resources.queue == "short"
    assert refinement.resources.walltime == "1:00"
    assert publication.dependency is not None
    assert publication.dependency.upstream_job_keys == (
        "subject_mask_refinement_array",
    )
    assert "--activate" not in publication.command


def test_plan_digest_and_benchmark_namespace_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = _prepare(tmp_path, monkeypatch)
    plan_path = Path(plan["run_root"]) / "plan.json"
    tampered = json.loads(plan_path.read_text(encoding="utf-8"))
    tampered["execution"]["inference"]["batch_size"] = 1
    plan_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="plan digest differs"):
        canary.load_plan(plan_path)

    with pytest.raises(ValueError, match=".palette_benchmarks"):
        canary._require_benchmark_root(tmp_path / "production" / "run")


def test_failed_worker_bundle_copy_removes_hidden_partial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    local_archive = tmp_path / "local.zarr"
    local_archive.mkdir()
    destination = tmp_path / "bundles" / "window_0"

    def fail_copy(*_args: object, **_kwargs: object) -> None:
        raise OSError("synthetic copy failure")

    monkeypatch.setattr(canary.shutil, "copytree", fail_copy)
    with pytest.raises(OSError, match="synthetic copy failure"):
        canary._publish_worker_bundle(
            local_archive=local_archive,
            parent="subject_mask_shard_runs",
            run_name="raw_window_0",
            bundle=destination,
            result={},
        )

    assert not destination.exists()
    assert list(destination.parent.iterdir()) == []
