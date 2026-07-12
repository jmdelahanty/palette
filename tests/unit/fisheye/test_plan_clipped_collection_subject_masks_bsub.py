from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import zarr

from fisheye.shared.subject_mask_chunks import DEFAULT_MASK_PROBS_SHARD_ROIS
from fisheye.utils.plan_clipped_collection_subject_masks_bsub import (
    SubjectMaskWorkflowPlan,
    apply_plan,
    build_arg_parser,
    build_plan,
)


def _write_cache_manifest(path: Path, *, collection_id: str, clip_id: str, work_unit_id: str = "") -> None:
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
            "collection_path": f"experiment_index/finalized_runs/{collection_id}",
            "bundle_child_clip_id": clip_id,
            "selection": {"clip_ids": [clip_id], "work_unit_ids": [work_unit_id] if work_unit_id else []},
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


def _build_subject_mask_plan(
    tmp_path: Path,
    *,
    components: list[str] | None = None,
    assignment_keypoints_run: str | None = "refined_keypoints_collection",
    finalization_mode: str = "collection_direct",
    mask_probs_shard_rois: int | None = DEFAULT_MASK_PROBS_SHARD_ROIS,
) -> SubjectMaskWorkflowPlan:
    zarr_path = tmp_path / "recording" / "zarr" / "sample_analysis.zarr"
    collection_id = "collection_test"
    _write_collection(zarr_path, collection_id=collection_id)
    cache_root = tmp_path / "cache"
    _write_cache_manifest(
        cache_root / "bundle_a" / "cache__clip_000001.flat_roi_cache.json",
        collection_id=collection_id,
        clip_id="clip_000001",
        work_unit_id="wu1",
    )
    _write_cache_manifest(
        cache_root / "bundle_a" / "cache__clip_000002.flat_roi_cache.json",
        collection_id=collection_id,
        clip_id="clip_000002",
        work_unit_id="wu2",
    )
    return build_plan(
        zarr_path=zarr_path,
        collection_id=collection_id,
        cache_dir_root=cache_root,
        clip_ids=[],
        all_clips=True,
        work_unit_ids=[],
        run_id="20260708T000000Z",
        run_label="test_masks",
        repo=Path("/groups/repo"),
        registry=Path("/groups/registry.sqlite"),
        log_dir=tmp_path / "logs",
        batch_size=128,
        device="0",
        queue="gpu_l4",
        ncores=8,
        mem_gb=32,
        gpus=1,
        finalizer_queue="normal",
        finalizer_ncores=8,
        finalizer_mem_gb=32,
        finalizer_num_workers=8,
        finalizer_chunk_size=256,
        finalizer_dense_mask_row_chunk=256,
        finalizer_execution_backend="process_shards",
        finalizer_postcompute_backend="process_shards",
        finalizer_postcompute_num_workers=8,
        finalizer_postcompute_chunk_size=256,
        metric_level="cheap",
        mask_storage="dense_and_bitpacked",
        mask_rle_validation_mode="invariants",
        components=components,
        assignment_keypoint_group="refined_keypoints_runs",
        assignment_keypoints_run=assignment_keypoints_run,
        write_eye_geometry=True,
        write_component_contours=True,
        retain_source_seeds=False,
        model_coverage_class="dense_all_components",
        model_component_coverage_key="body+eyes+swim_bladder",
        model_label_schema_id="subject_v1_union",
        model_top_k=5,
        model_require_unique=False,
        model_include_non_success=False,
        mask_probs_dtype="uint8",
        mask_probs_chunk_rois=32,
        mask_probs_shard_rois=mask_probs_shard_rois,
        output_queue_size=2,
        profile_timings=True,
        allow_multiple_cache_manifests=False,
        overwrite_proxies=False,
        overwrite_shards=False,
        overwrite_final_outputs=False,
        defer_registry_status=False,
        finalization_mode=finalization_mode,
        clip_finalizer_package_dir=tmp_path / "nrs_packages",
    )


def test_parser_defaults_to_probability_shards_and_accepts_regular_override() -> None:
    required = [
        "--zarr",
        "/recording.zarr",
        "--collection-id",
        "collection",
        "--cache-dir-root",
        "/cache",
        "--dry-run",
    ]

    default_args = build_arg_parser().parse_args(required)
    regular_args = build_arg_parser().parse_args([*required, "--no-mask-probs-sharding"])
    full_contour_args = build_arg_parser().parse_args(
        [*required, "--write-component-contours"]
    )

    assert default_args.mask_probs_shard_rois == DEFAULT_MASK_PROBS_SHARD_ROIS
    assert default_args.write_component_contours is False
    assert default_args.write_sampled_component_contours is True
    assert regular_args.mask_probs_shard_rois is None
    assert full_contour_args.write_component_contours is True


def test_build_plan_resolves_subject_mask_shard_commands(tmp_path: Path) -> None:
    plan = _build_subject_mask_plan(tmp_path)

    assert [clip.clip_id for clip in plan.clips] == ["clip_000001", "clip_000002"]
    assert [clip.cache_status for clip in plan.clips] == ["found", "found"]
    assert plan.merged_proxy_crop_run == "crop_proxy_test_masks_collection"
    assert plan.refined_subject_masks_run == "refined_subject_masks_test_masks"
    assert plan.components == ("subject_body", "eyes_union", "swim_bladder")

    first = plan.clips[0]
    assert first.proxy_crop_run == "crop_proxy_test_masks_clip_000001"
    assert first.subject_mask_shard_run == "subject_masks_test_masks_clip_000001"
    assert first.row_index_path == first.cache_manifest.with_suffix(".rows.parquet")
    assert first.subject_mask_command is not None
    assert "fisheye.segmentation.infer_unet_subject_masks" in first.subject_mask_command
    assert "--output-parent" in first.subject_mask_command
    assert "subject_mask_shard_runs" in first.subject_mask_command
    assert "--source-roi-cache-row-index-path" in first.subject_mask_command
    assert "--source-clip-index" in first.subject_mask_command
    assert "1" in first.subject_mask_command
    assert "--profile-timings" in first.subject_mask_command
    assert first.subject_mask_command[first.subject_mask_command.index("--mask-probs-shard-rois") + 1] == "2048"
    assert "--assignment-keypoint-group" not in first.subject_mask_command
    assert "--assignment-keypoint-run" not in first.subject_mask_command
    assert "refined_keypoints_collection" not in first.subject_mask_command

    finalizer_dependency_index = plan.finalizer_bsub_command.index("-w") + 1
    finalizer_dependency = plan.finalizer_bsub_command[finalizer_dependency_index]
    assert finalizer_dependency == "done(<jobid:sm_test_masks_clip_000001>) && done(<jobid:sm_test_masks_clip_000002>)"
    assert "subject_masks_test_masks_clip_000001" in plan.finalize_command
    assert "subject_masks_test_masks_clip_000002" in plan.finalize_command
    assert "--assignment-keypoints-run" in plan.finalize_command
    assert "refined_keypoints_collection" in plan.finalize_command
    assert "--registry" in plan.finalize_command


def test_build_plan_forwards_regular_probability_chunk_override(tmp_path: Path) -> None:
    plan = _build_subject_mask_plan(tmp_path, mask_probs_shard_rois=None)

    command = plan.clips[0].subject_mask_command
    assert command is not None
    assert "--no-mask-probs-sharding" in command
    assert "--mask-probs-shard-rois" not in command


def test_build_plan_can_emit_per_clip_finalizer_package_jobs(tmp_path: Path) -> None:
    plan = _build_subject_mask_plan(tmp_path, finalization_mode="per_clip_packages")

    assert plan.finalization_mode == "per_clip_packages"
    assert plan.finalizer_bsub_command == []
    assert plan.clip_finalizer_package_dir == (tmp_path / "nrs_packages" / "test_masks").resolve()
    first = plan.clips[0]
    assert first.refined_subject_mask_clip_run == "refined_subject_masks_test_masks_clip_000001"
    assert first.refined_subject_mask_package_path == (
        plan.clip_finalizer_package_dir / "refined_subject_masks_test_masks_clip_000001.tar.gz"
    )
    assert first.finalizer_command is not None
    assert "fisheye.utils.finalize_subject_mask_clip_package" in first.finalizer_command
    assert "--source-zarr" in first.finalizer_command
    assert "--subject-shard-run" in first.finalizer_command
    assert "subject_masks_test_masks_clip_000001" in first.finalizer_command
    assert "--target-crop-run" in first.finalizer_command
    assert "crop_proxy_test_masks_collection" in first.finalizer_command
    assert "--assignment-keypoints-run" in first.finalizer_command
    assert "refined_keypoints_collection" in first.finalizer_command
    assert first.finalizer_bsub_command is not None
    dependency = first.finalizer_bsub_command[first.finalizer_bsub_command.index("-w") + 1]
    assert dependency == "done(<jobid:sm_test_masks_clip_000001>)"
    assert plan.collection_import_job_name == "sm_import_test_masks"
    assert "fisheye.utils.import_refined_subject_mask_clip_packages" in plan.collection_import_command
    assert "--expected-target-crop-run" in plan.collection_import_command
    assert "crop_proxy_test_masks_collection" in plan.collection_import_command
    assert "--array-copy-workers" in plan.collection_import_command
    assert "1" in plan.collection_import_command
    assert "--output-run" in plan.collection_import_command
    assert "refined_subject_masks_test_masks" in plan.collection_import_command
    assert plan.collection_import_command.count("--package") == 2
    assert plan.collection_import_bsub_command
    import_dependency = plan.collection_import_bsub_command[plan.collection_import_bsub_command.index("-w") + 1]
    assert import_dependency == (
        "done(<jobid:sm_finalize_test_masks_clip_000001>) "
        "&& done(<jobid:sm_finalize_test_masks_clip_000002>)"
    )


def test_build_plan_requires_assignment_keypoints_for_eyes_union(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="eyes_union requires --assignment-keypoints-run"):
        _build_subject_mask_plan(tmp_path, assignment_keypoints_run=None)


def test_build_plan_allows_body_swim_without_assignment_keypoints(tmp_path: Path) -> None:
    plan = _build_subject_mask_plan(
        tmp_path,
        components=["subject_body", "swim_bladder"],
        assignment_keypoints_run=None,
    )

    assert plan.components == ("subject_body", "swim_bladder")
    assert "--assignment-keypoints-run" not in plan.finalize_command


def test_build_plan_marks_missing_cache_without_subject_mask_command(tmp_path: Path) -> None:
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
        run_id="20260708T000000Z",
        run_label="test_masks",
        repo=Path("/groups/repo"),
        registry=Path("/groups/registry.sqlite"),
        log_dir=tmp_path / "logs",
        batch_size=128,
        device="0",
        queue="gpu_l4",
        ncores=8,
        mem_gb=32,
        gpus=1,
        finalizer_queue="normal",
        finalizer_ncores=8,
        finalizer_mem_gb=32,
        finalizer_num_workers=8,
        finalizer_chunk_size=256,
        finalizer_dense_mask_row_chunk=256,
        finalizer_execution_backend="process_shards",
        finalizer_postcompute_backend="process_shards",
        finalizer_postcompute_num_workers=8,
        finalizer_postcompute_chunk_size=256,
        metric_level="cheap",
        mask_storage="dense_and_bitpacked",
        mask_rle_validation_mode="invariants",
        components=["subject_body", "swim_bladder"],
        assignment_keypoint_group="refined_keypoints_runs",
        assignment_keypoints_run=None,
        write_eye_geometry=False,
        write_component_contours=True,
        retain_source_seeds=False,
        model_coverage_class="dense_all_components",
        model_component_coverage_key="body+eyes+swim_bladder",
        model_label_schema_id="subject_v1_union",
        model_top_k=5,
        model_require_unique=False,
        model_include_non_success=False,
        mask_probs_dtype="uint8",
        mask_probs_chunk_rois=32,
        mask_probs_shard_rois=None,
        output_queue_size=2,
        profile_timings=False,
        allow_multiple_cache_manifests=False,
        overwrite_proxies=False,
        overwrite_shards=False,
        overwrite_final_outputs=False,
        defer_registry_status=True,
    )

    assert len(plan.clips) == 1
    assert plan.clips[0].cache_status == "missing"
    assert plan.clips[0].proxy_command is None
    assert plan.clips[0].subject_mask_bsub_command is None
    assert plan.to_json()["missing_cache_clip_ids"] == ["clip_000001"]


def test_apply_plan_creates_proxies_and_submits_dependency_dag(tmp_path: Path) -> None:
    plan = _build_subject_mask_plan(tmp_path, components=["subject_body", "swim_bladder"], assignment_keypoints_run=None)
    calls: list[list[str]] = []
    bsub_outputs = iter(
        [
            "Job <101> is submitted to queue <gpu_l4>.",
            "Job <102> is submitted to queue <gpu_l4>.",
            "Job <201> is submitted to queue <normal>.",
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

    assert [call[0] for call in calls] == ["scripts/py", "scripts/py", "bsub", "bsub", "bsub"]
    assert submission["job_ids_by_name"] == {
        "sm_test_masks_clip_000001": "101",
        "sm_test_masks_clip_000002": "102",
        "sm_finalize_test_masks": "201",
    }
    finalizer_call = calls[4]
    assert finalizer_call[finalizer_call.index("-w") + 1] == "done(101) && done(102)"
    assert (tmp_path / "logs" / "submission_plan.json").exists()
    submission_path = tmp_path / "logs" / "submission.json"
    assert submission_path.exists()
    saved = json.loads(submission_path.read_text(encoding="utf-8"))
    assert saved["schema"] == "palette.clipped_collection_subject_mask_bsub_submission.v1"
    assert saved["status"] == "submitted"
    assert saved["finalizer"]["job_id"] == "201"


def test_apply_plan_submits_per_clip_finalizer_package_jobs(tmp_path: Path) -> None:
    plan = _build_subject_mask_plan(
        tmp_path,
        components=["subject_body", "swim_bladder"],
        assignment_keypoints_run=None,
        finalization_mode="per_clip_packages",
    )
    calls: list[list[str]] = []
    bsub_outputs = iter(
        [
            "Job <101> is submitted to queue <gpu_l4>.",
            "Job <102> is submitted to queue <gpu_l4>.",
            "Job <301> is submitted to queue <normal>.",
            "Job <302> is submitted to queue <normal>.",
            "Job <401> is submitted to queue <normal>.",
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

    assert [call[0] for call in calls] == [
        "scripts/py",
        "scripts/py",
        "scripts/py",
        "bsub",
        "bsub",
        "bsub",
        "bsub",
        "bsub",
    ]
    assert "fisheye.utils.merge_clipped_proxy_crop_runs" in calls[2]
    assert submission["finalization_mode"] == "per_clip_packages"
    assert submission["job_ids_by_name"] == {
        "sm_test_masks_clip_000001": "101",
        "sm_test_masks_clip_000002": "102",
        "sm_finalize_test_masks_clip_000001": "301",
        "sm_finalize_test_masks_clip_000002": "302",
        "sm_import_test_masks": "401",
    }
    first_finalizer_call = calls[5]
    second_finalizer_call = calls[6]
    import_call = calls[7]
    assert first_finalizer_call[first_finalizer_call.index("-w") + 1] == "done(101)"
    assert second_finalizer_call[second_finalizer_call.index("-w") + 1] == "done(102)"
    assert import_call[import_call.index("-w") + 1] == "done(301) && done(302)"
    assert "fisheye.utils.import_refined_subject_mask_clip_packages" in import_call[-1]
    assert "--array-copy-workers 1" in import_call[-1]
    assert len(submission["clip_finalizers"]) == 2
    assert submission["finalizer"]["job_id"] == "401"
    assert submission["finalizer"]["source_package_count"] == 2
    assert submission["clip_finalizers"][0]["package_path"].endswith(
        "refined_subject_masks_test_masks_clip_000001.tar.gz"
    )

    saved = json.loads((tmp_path / "logs" / "submission.json").read_text(encoding="utf-8"))
    assert saved["merge_proxy"]["merged_proxy_crop_run"] == "crop_proxy_test_masks_collection"
    assert len(saved["clip_finalizers"]) == 2
