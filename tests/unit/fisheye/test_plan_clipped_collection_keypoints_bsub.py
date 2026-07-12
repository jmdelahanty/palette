from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import zarr

from fisheye.utils.plan_clipped_collection_keypoints_bsub import (
    _parse_bsub_job_id,
    _replace_job_placeholders,
    apply_plan,
    build_plan,
)


def _write_cache_manifest(path: Path, *, collection_id: str, clip_id: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "palette_flat_roi_cache_manifest_v1",
        "cache_complete": True,
        "array": {
            "bin_path": path.with_suffix(".bin").name,
            "shape": [3, 512, 512],
            "dtype": "uint8",
        },
        "row_index": {
            "path": path.with_suffix(".rows.parquet").name,
            "row_count": 3,
            "schema": "palette_clipped_collection_flat_roi_cache_rows_v1",
        },
        "source": {
            "source_kind": "finalized_clipped_refined_detect_collection",
            "collection_id": collection_id,
            "bundle_child_clip_id": clip_id,
            "selection": {"clip_ids": [clip_id], "work_unit_ids": []},
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_collection(zarr_path: Path, *, collection_id: str) -> None:
    root = zarr.open_group(store=zarr_path, mode="w")
    collection = root.require_group("experiment_index").require_group("finalized_runs").create_group(collection_id)
    collection.attrs["selected_runs"] = [
        {"clip_id": "clip_000001", "work_unit_id": "wu1"},
        {"clip_id": "clip_000002", "work_unit_id": "wu2"},
    ]


def test_build_plan_resolves_manifests_and_dependency_commands(tmp_path: Path) -> None:
    zarr_path = tmp_path / "recording" / "zarr" / "sample_analysis.zarr"
    collection_id = "collection_test"
    _write_collection(zarr_path, collection_id=collection_id)
    cache_root = tmp_path / "cache"
    _write_cache_manifest(cache_root / "bundle_a" / "cache__clip_000001.flat_roi_cache.json", collection_id=collection_id, clip_id="clip_000001")
    _write_cache_manifest(cache_root / "bundle_a" / "cache__clip_000002.flat_roi_cache.json", collection_id=collection_id, clip_id="clip_000002")

    plan = build_plan(
        zarr_path=zarr_path,
        collection_id=collection_id,
        cache_dir_root=cache_root,
        clip_ids=[],
        all_clips=True,
        work_unit_ids=[],
        run_id="20260707T000000Z",
        run_label="test_run",
        repo=Path("/groups/repo"),
        registry=Path("/groups/registry.sqlite"),
        log_dir=tmp_path / "logs",
        pose_schema="traditional_v2",
        batch_size=256,
        device="0",
        queue="gpu_l4",
        ncores=4,
        mem_gb=32,
        gpus=1,
        finalizer_queue="normal",
        finalizer_ncores=4,
        finalizer_mem_gb=16,
        refine_queue="normal",
        refine_ncores=4,
        refine_mem_gb=16,
        refine_num_workers=4,
        refine_scheduler="threads",
        refine_chunk_size=2048,
        stage_roi_cache_to_scratch=True,
        allow_multiple_cache_manifests=False,
        overwrite_proxies=False,
        overwrite_final_outputs=False,
    )

    assert [clip.clip_id for clip in plan.clips] == ["clip_000001", "clip_000002"]
    assert [clip.cache_status for clip in plan.clips] == ["found", "found"]
    assert plan.merged_proxy_crop_run == "crop_proxy_test_run_collection"
    assert plan.keypoint_collection_run == "keypoints_test_run"
    assert plan.refined_keypoints_run == "refined_keypoints_test_run"
    assert plan.keypoint_storage["effective"] == {
        "keypoint_storage_layout": "indexed_sharding_v1",
        "keypoint_storage_policy": "default_indexed_sharding_v1",
        "keypoint_roi_shard_rows": 65536,
        "keypoint_frame_shard_rows": 262144,
    }
    assert "lsf_workflow" not in plan.to_json()
    assert [job.job_key for job in plan.lsf_workflow.topological_jobs()] == [
        "kp_test_run_clip_000001",
        "kp_test_run_clip_000002",
        "kp_finalize_test_run",
        "kp_refine_test_run",
    ]

    first = plan.clips[0]
    assert first.proxy_crop_run == "crop_proxy_test_run_clip_000001"
    assert first.keypoint_shard_run == "keypoint_shard_test_run_clip_000001"
    assert first.proxy_command is not None
    assert "--alias-manifest" in first.proxy_command
    assert first.keypoint_command is not None
    assert "--stage-roi-cache-to-scratch" in first.keypoint_command
    assert "--run-name" in first.keypoint_command
    assert "keypoint_shard_test_run_clip_000001" in first.keypoint_command
    assert first.keypoint_command[first.keypoint_command.index("--keypoint-roi-shard-rows") + 1] == "65536"
    assert first.keypoint_command[first.keypoint_command.index("--keypoint-frame-shard-rows") + 1] == "262144"
    assert first.keypoint_bsub_command is not None
    assert first.keypoint_bsub_command[:-3] == [
        "bsub",
        "-J",
        "kp_test_run_clip_000001",
        "-n",
        "4",
        "-R",
        "rusage[mem=32G]",
        "-oo",
        str(tmp_path / "logs" / "kp_test_run_clip_000001.%J.out"),
        "-eo",
        str(tmp_path / "logs" / "kp_test_run_clip_000001.%J.err"),
        "-q",
        "gpu_l4",
        "-gpu",
        "num=1",
    ]
    assert first.keypoint_bsub_command[-3:-1] == ["bash", "-lc"]
    assert first.keypoint_bsub_command[-1].startswith(
        "cd /groups/repo && scripts/py -m fisheye.utils.run_keypoints_with_registry_model"
    )

    finalizer_dependency_index = plan.finalizer_bsub_command.index("-w") + 1
    finalizer_dependency = plan.finalizer_bsub_command[finalizer_dependency_index]
    assert finalizer_dependency == "done(<jobid:kp_test_run_clip_000001>) && done(<jobid:kp_test_run_clip_000002>)"
    assert "keypoint_shard_test_run_clip_000001" in plan.finalize_command
    assert "keypoint_shard_test_run_clip_000002" in plan.finalize_command
    assert plan.refine_bsub_command[plan.refine_bsub_command.index("-w") + 1] == "done(<jobid:kp_finalize_test_run>)"


def test_build_plan_marks_missing_cache_without_keypoint_command(tmp_path: Path) -> None:
    zarr_path = tmp_path / "recording" / "zarr" / "sample_analysis.zarr"
    collection_id = "collection_test"
    _write_collection(zarr_path, collection_id=collection_id)

    plan = build_plan(
        zarr_path=zarr_path,
        collection_id=collection_id,
        cache_dir_root=tmp_path / "cache",
        clip_ids=["clip_000001"],
        all_clips=False,
        work_unit_ids=[],
        run_id="20260707T000000Z",
        run_label="test_run",
        repo=Path("/groups/repo"),
        registry=Path("/groups/registry.sqlite"),
        log_dir=tmp_path / "logs",
        pose_schema="traditional_v2",
        batch_size=256,
        keypoint_roi_shard_rows=None,
        keypoint_frame_shard_rows=262144,
        device="0",
        queue="gpu_l4",
        ncores=4,
        mem_gb=32,
        gpus=1,
        finalizer_queue="normal",
        finalizer_ncores=4,
        finalizer_mem_gb=16,
        refine_queue="normal",
        refine_ncores=4,
        refine_mem_gb=16,
        refine_num_workers=4,
        refine_scheduler="threads",
        refine_chunk_size=2048,
        stage_roi_cache_to_scratch=True,
        allow_multiple_cache_manifests=False,
        overwrite_proxies=False,
        overwrite_final_outputs=False,
    )
    assert plan.keypoint_storage["effective"]["keypoint_storage_layout"] == "regular_chunks_v1"
    assert len(plan.clips) == 1
    assert plan.clips[0].cache_status == "missing"
    assert plan.clips[0].proxy_command is None
    assert plan.clips[0].keypoint_bsub_command is None
    assert plan.to_json()["missing_cache_clip_ids"] == ["clip_000001"]


def test_parse_and_replace_bsub_job_placeholders() -> None:
    assert _parse_bsub_job_id("Job <12345> is submitted to queue <gpu_l4>.") == "12345"
    assert _parse_bsub_job_id("", "Job <67890> is submitted to queue <normal>.") == "67890"
    command = ["bsub", "-w", "done(<jobid:job_a>) && done(<jobid:job_b>)"]
    assert _replace_job_placeholders(command, {"job_a": "111", "job_b": "222"}) == [
        "bsub",
        "-w",
        "done(111) && done(222)",
    ]


def test_apply_plan_creates_proxies_and_submits_dependency_dag(tmp_path: Path) -> None:
    zarr_path = tmp_path / "recording" / "zarr" / "sample_analysis.zarr"
    collection_id = "collection_test"
    _write_collection(zarr_path, collection_id=collection_id)
    cache_root = tmp_path / "cache"
    _write_cache_manifest(cache_root / "bundle_a" / "cache__clip_000001.flat_roi_cache.json", collection_id=collection_id, clip_id="clip_000001")
    _write_cache_manifest(cache_root / "bundle_a" / "cache__clip_000002.flat_roi_cache.json", collection_id=collection_id, clip_id="clip_000002")
    plan = build_plan(
        zarr_path=zarr_path,
        collection_id=collection_id,
        cache_dir_root=cache_root,
        clip_ids=[],
        all_clips=True,
        work_unit_ids=[],
        run_id="20260707T000000Z",
        run_label="test_run",
        repo=Path("/groups/repo"),
        registry=Path("/groups/registry.sqlite"),
        log_dir=tmp_path / "logs",
        pose_schema="traditional_v2",
        batch_size=256,
        keypoint_roi_shard_rows=None,
        keypoint_frame_shard_rows=262144,
        device="0",
        queue="gpu_l4",
        ncores=4,
        mem_gb=32,
        gpus=1,
        finalizer_queue="normal",
        finalizer_ncores=4,
        finalizer_mem_gb=16,
        refine_queue="normal",
        refine_ncores=4,
        refine_mem_gb=16,
        refine_num_workers=4,
        refine_scheduler="threads",
        refine_chunk_size=2048,
        stage_roi_cache_to_scratch=True,
        allow_multiple_cache_manifests=False,
        overwrite_proxies=False,
        overwrite_final_outputs=False,
    )
    assert all(
        clip.keypoint_command is not None
        and "--no-keypoint-sharding" in clip.keypoint_command
        and "--keypoint-roi-shard-rows" not in clip.keypoint_command
        for clip in plan.clips
    )
    calls: list[list[str]] = []
    bsub_outputs = iter(
        [
            "Job <101> is submitted to queue <gpu_l4>.",
            "Job <102> is submitted to queue <gpu_l4>.",
            "Job <201> is submitted to queue <normal>.",
            "Job <301> is submitted to queue <normal>.",
        ]
    )

    def fake_runner(argv, **kwargs):
        del kwargs
        command = [str(item) for item in argv]
        calls.append(command)
        if command[0] == "bsub":
            return SimpleNamespace(returncode=0, stdout="", stderr=next(bsub_outputs))
        return SimpleNamespace(returncode=0, stdout='{"ok": true}', stderr="")

    submission = apply_plan(plan, runner=fake_runner)

    assert [call[0] for call in calls] == ["scripts/py", "scripts/py", "bsub", "bsub", "bsub", "bsub"]
    assert submission["job_ids_by_name"] == {
        "kp_test_run_clip_000001": "101",
        "kp_test_run_clip_000002": "102",
        "kp_finalize_test_run": "201",
        "kp_refine_test_run": "301",
    }
    finalizer_call = calls[4]
    assert finalizer_call[finalizer_call.index("-w") + 1] == "done(101) && done(102)"
    refine_call = calls[5]
    assert refine_call[refine_call.index("-w") + 1] == "done(201)"
    assert (tmp_path / "logs" / "submission_plan.json").exists()
    submission_path = tmp_path / "logs" / "submission.json"
    assert submission_path.exists()
    saved = json.loads(submission_path.read_text(encoding="utf-8"))
    assert saved["schema"] == "palette.clipped_collection_keypoint_bsub_submission.v1"
    assert saved["status"] == "submitted"
    assert saved["refine"]["job_id"] == "301"
    lsf_plan = json.loads(
        (tmp_path / "logs" / "lsf_plan.json").read_text(encoding="utf-8")
    )
    assert lsf_plan["family"] == "keypoints.clipped_collection"
    lsf_submission = json.loads(
        (tmp_path / "logs" / "lsf_submission.json").read_text(encoding="utf-8")
    )
    assert lsf_submission["status"] == "submitted"
    assert lsf_submission["job_ids_by_key"] == submission["job_ids_by_name"]
