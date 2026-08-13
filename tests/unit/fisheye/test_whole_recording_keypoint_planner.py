from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from fisheye.cluster.keypoints import common as common_mod
from fisheye.cluster.keypoints.common import KeypointInputCapability, PoseModelBinding
from fisheye.cluster.keypoints import registry_finalize as registry_finalize_mod
from fisheye.cluster.keypoints import whole_recording as planner
from fisheye.cluster import whole_recording_analysis as analysis_planner
from fisheye.cluster.lsf import LsfJob, LsfResources
from fisheye.shared.roi_pixel_contract import (
    orange_mono_pynvvc_luma_pixel_contract,
)
from fisheye.shared.model_input_transform import resolve_model_input_transform
from fisheye.shared.pose_model_input_contract import PoseModelInputRuntimePlan


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _make_target(tmp_path: Path, name: str) -> dict[str, str]:
    recording_dir = tmp_path / "recordings" / name
    zarr_path = recording_dir / "zarr" / f"{name}_analysis.zarr"
    zarr_path.mkdir(parents=True, exist_ok=True)
    payload_path = tmp_path / "caches" / f"{name}.bin"
    payload_path.parent.mkdir(parents=True, exist_ok=True)
    total_bytes = 2 * 348 * 348
    payload_path.write_bytes(b"\x00" * total_bytes)
    manifest_path = tmp_path / "caches" / f"{name}.flat_roi_cache.json"
    _write_json(
        manifest_path,
        {
            "schema": "palette_roi_cache_flat_bin_v1",
            "layout": "flat_bin_v1",
            "cache_complete": True,
            "cache_key": f"cache-{name}",
            "source": {
                "archive_path": str(zarr_path.resolve()),
                "crop_run_name": "crop_001",
                "crop_signature": f"signature-{name}",
                "crop_revision": "revision-001",
            },
            "array": {
                "bin_path": payload_path.name,
                "dtype": "uint8",
                "shape": [2, 348, 348],
                "order": "C",
                "total_bytes": total_bytes,
                "sha256": f"sha-{name}",
            },
        },
    )
    return {
        "target_id": name,
        "recording_id": f"recording-{name}",
        "recording_dir": str(recording_dir),
        "analysis_zarr": str(zarr_path),
        "roi_cache_manifest": str(manifest_path),
        "crop_run": "crop_001",
    }


def _make_inputs(tmp_path: Path, *, target_count: int = 2) -> tuple[Path, Path, Path, Path]:
    repo = tmp_path / "repo"
    scripts_py = repo / "scripts" / "py"
    scripts_py.parent.mkdir(parents=True, exist_ok=True)
    scripts_py.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    registry = tmp_path / "registry.sqlite"
    registry.touch()
    targets = [_make_target(tmp_path, f"target_{index}") for index in range(target_count)]
    manifest = tmp_path / "targets.json"
    _write_json(
        manifest,
        {
            "schema": planner.TARGET_MANIFEST_SCHEMA,
            "expected_target_count": target_count,
            "targets": targets,
        },
    )
    model_path = tmp_path / "model.pt"
    model_path.write_bytes(b"model")
    return manifest, repo, registry, model_path


def _build_plan(
    tmp_path: Path,
    monkeypatch,
    *,
    target_count: int = 2,
    keypoint_roi_shard_rows: int | None = common_mod.DEFAULT_KEYPOINT_ROI_SHARD_ROWS,
    keypoint_frame_shard_rows: int = common_mod.DEFAULT_KEYPOINT_FRAME_SHARD_ROWS,
    planned_caches: bool = False,
    shared_cache_bundle: bool = False,
    palette_commit: str = "a" * 40,
):
    manifest, repo, registry, model_path = _make_inputs(
        tmp_path,
        target_count=target_count,
    )
    monkeypatch.setattr(
        planner,
        "validate_registered_analysis_zarr",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(planner, "_repo_head_commit", lambda _repo: "a" * 40)

    def resolve_model(**kwargs):
        return PoseModelBinding(
            recording_id=kwargs["recording_id"],
            set_id="pose_set",
            run_id="pose_run",
            model_path=model_path,
            model_sha256="abc123",
            score=10.0,
            created_utc="2026-07-10T00:00:00Z",
        )

    monkeypatch.setattr(planner, "resolve_pose_model_binding", resolve_model)
    model_input_contract_path = tmp_path / "pose_model_input_contract.json"
    model_input_contract_path.write_text("{}\n", encoding="utf-8")
    contract = SimpleNamespace(
        path=model_input_contract_path,
        sha256="c" * 64,
        payload_digest="d" * 64,
        training_source_shape_hw=(512, 512),
        network_shape_hw=(256, 256),
        model_stride=32,
        input_mode="numpy-list",
        to_json=lambda: {
            "path": str(model_input_contract_path),
            "sha256": "c" * 64,
            "payload_digest": "d" * 64,
            "training_source_shape_hw": [512, 512],
            "network_shape_hw": [256, 256],
            "model_stride": 32,
            "input_mode": "numpy-list",
        },
    )

    def plan_for_native_shape(native_shape):
        transform = resolve_model_input_transform(
            native_shape,
            mode="pad_to_size",
            model_hw=(512, 512),
        )
        return PoseModelInputRuntimePlan(
            transform=transform,
            network_shape_hw=(256, 256),
            model_stride=32,
            input_mode="numpy-list",
            profile_id="scale_matched_center_pad_ultralytics_v1",
            classification="scale_matched_diagnostic_not_training_context",
            contract_path=model_input_contract_path,
            contract_sha256="c" * 64,
            contract_payload_digest="d" * 64,
        )

    contract.plan_for_native_shape = plan_for_native_shape
    monkeypatch.setattr(
        planner,
        "load_pose_model_input_contract",
        lambda *_args, **_kwargs: contract,
    )
    monkeypatch.setattr(
        planner,
        "validate_keypoint_input_dag",
        lambda *, cache, min_roi_size, **_kwargs: KeypointInputCapability(
            selected_source="flat_roi_cache",
            min_roi_size=min_roi_size,
            crop_run=cache.crop_run,
            crop_storage_mode="geometry_only",
            crop_shape=cache.shape,
            persisted_roi_images_available=False,
            persisted_roi_images_eligible=False,
            acquisition_crop_video_available=True,
            acquisition_crop_video_eligible=True,
            acquisition_crop_video_path=tmp_path / "crop.mp4",
            acquisition_crop_video_shape=(348, 348),
            acquisition_crop_video_probe_status="ok",
            flat_roi_cache_eligible=True,
            rejected_sources={"persisted_roi_images": "missing"},
        ),
    )
    cache_bindings = None
    upstream_jobs: tuple[LsfJob, ...] = ()
    if planned_caches:
        cache_bindings = {}
        cache_jobs: list[LsfJob] = []
        for target in planner.load_target_manifest(manifest):
            existing_payload_name = json.loads(
                target.roi_cache_manifest.read_text(encoding="utf-8")
            )["array"]["bin_path"]
            (target.roi_cache_manifest.parent / existing_payload_name).unlink()
            target.roi_cache_manifest.unlink()
            producer_job_key = (
                "cache_bundle:000"
                if shared_cache_bundle
                else f"cache:{target.target_id}"
            )
            cache_bindings[target.target_id] = common_mod.FlatRoiCacheBinding(
                manifest_path=target.roi_cache_manifest,
                manifest_sha256=None,
                payload_path=target.roi_cache_manifest.with_suffix(".bin"),
                crop_run="crop_001",
                cache_key=None,
                crop_signature=f"signature-{target.target_id}",
                crop_revision="revision-001",
                shape=(2, 348, 348),
                total_bytes=2 * 348 * 348,
                payload_sha256=None,
                availability="planned",
                producer_job_key=producer_job_key,
            )
            if not any(job.job_key == producer_job_key for job in cache_jobs):
                cache_jobs.append(
                    LsfJob(
                        job_key=producer_job_key,
                        job_name=f"cache_{target.target_id}",
                        command=("true",),
                        resources=LsfResources(
                            queue="gpu_l4", ncores=4, mem_gb=64, gpus=1
                        ),
                        stdout_path=tmp_path / f"{target.target_id}.cache.out",
                        stderr_path=tmp_path / f"{target.target_id}.cache.err",
                    )
                )
        upstream_jobs = tuple(cache_jobs)
    plan = planner.build_plan(
        manifest_path=manifest,
        run_label="goodcopbadcop_20260710",
        repo=repo,
        palette_commit=palette_commit,
        registry=registry,
        run_root=tmp_path / "run",
        model_set_id="pose_set",
        model_run_id="pose_run",
        model_input_contract_path=model_input_contract_path,
        pose_schema="traditional_v2",
        min_roi_size=348,
        batch_size=256,
        device="0",
        input_mode="model-contract",
        model_input_stride=None,
        progress_every_batches=1,
        keypoint_roi_shard_rows=keypoint_roi_shard_rows,
        keypoint_frame_shard_rows=keypoint_frame_shard_rows,
        prediction_resources=LsfResources(
            queue="gpu_l4", ncores=4, mem_gb=32, gpus=1
        ),
        refinement_resources=LsfResources(
            queue="short", ncores=4, mem_gb=16, walltime="1:00"
        ),
        refine_chunk_size=2048,
        refine_scheduler="threads",
        refine_num_workers=4,
        refine_memory_limit=None,
        finalizer_resources=LsfResources(
            queue="short", ncores=1, mem_gb=8, walltime="1:00"
        ),
        cache_bindings=cache_bindings,
        upstream_jobs=upstream_jobs,
    )
    return plan


def test_whole_recording_plan_rejects_nonmatching_palette_commit(
    tmp_path: Path,
    monkeypatch,
) -> None:
    with pytest.raises(
        ValueError,
        match="does not match the exact HEAD",
    ):
        _build_plan(tmp_path, monkeypatch, palette_commit="b" * 40)


def test_manifest_loader_requires_explicit_count_and_unique_targets(tmp_path: Path) -> None:
    target = _make_target(tmp_path, "target_a")
    manifest = tmp_path / "targets.json"
    _write_json(
        manifest,
        {
            "schema": planner.TARGET_MANIFEST_SCHEMA,
            "expected_target_count": 2,
            "targets": [target],
        },
    )
    with pytest.raises(ValueError, match="expected_target_count"):
        planner.load_target_manifest(manifest)

    _write_json(
        manifest,
        {
            "schema": planner.TARGET_MANIFEST_SCHEMA,
            "targets": [target, target],
        },
    )
    with pytest.raises(ValueError, match="target_id"):
        planner.load_target_manifest(manifest)


def test_flat_cache_binding_rejects_small_zebrafish_surface(tmp_path: Path) -> None:
    target = _make_target(tmp_path, "small_cache")
    manifest_path = Path(target["roi_cache_manifest"])
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    small_bytes = 2 * 320 * 320
    payload["array"]["shape"] = [2, 320, 320]
    payload["array"]["total_bytes"] = small_bytes
    payload_path = manifest_path.parent / payload["array"]["bin_path"]
    payload_path.write_bytes(b"\x00" * small_bytes)
    _write_json(manifest_path, payload)

    with pytest.raises(ValueError, match="requires at least 348x348"):
        common_mod.validate_flat_roi_cache_binding(
            manifest_path=manifest_path,
            analysis_zarr=Path(target["analysis_zarr"]),
            crop_run="crop_001",
            min_roi_size=348,
        )


def test_flat_cache_binding_validates_metadata_without_memmap(
    tmp_path: Path,
    monkeypatch,
) -> None:
    target = _make_target(tmp_path, "metadata_only_cache")

    def reject_memmap(*args, **kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("planner preflight must not memory-map the cache payload")

    import fisheye.shared.flat_roi_cache as flat_roi_cache_mod

    monkeypatch.setattr(flat_roi_cache_mod.np, "memmap", reject_memmap)
    binding = common_mod.validate_flat_roi_cache_binding(
        manifest_path=Path(target["roi_cache_manifest"]),
        analysis_zarr=Path(target["analysis_zarr"]),
        crop_run="crop_001",
        min_roi_size=348,
    )

    assert binding.shape == (2, 348, 348)
    assert binding.total_bytes == 2 * 348 * 348


def test_whole_recording_plan_builds_independent_chains_and_serial_fanin(
    tmp_path: Path,
    monkeypatch,
) -> None:
    plan = _build_plan(tmp_path, monkeypatch)

    assert plan.to_json()["schema"] == planner.PLAN_SCHEMA
    assert plan.to_json()["palette_commit"] == "a" * 40
    assert len(plan.targets) == 2
    assert len(plan.targets[0].cache.manifest_sha256) == 64
    assert plan.targets[0].input_capability.selected_source == "flat_roi_cache"
    assert plan.targets[0].model_input_transform.to_attrs() == {
        "name": "pad_to_size",
        "native_shape_hw": [348, 348],
        "model_shape_hw": [512, 512],
        "pad_top": 82,
        "pad_bottom": 82,
        "pad_left": 82,
        "pad_right": 82,
        "coordinate_mapping": "native_xy = model_xy - [pad_left, pad_top]",
    }
    assert plan.targets[0].model_input_stride == 32
    assert plan.model_input_stride == 32
    assert plan.min_roi_size == 348
    assert plan.keypoint_storage["effective"]["keypoint_storage_layout"] == (
        "shared_byte_planned_indexed_sharding_v1"
    )
    assert plan.keypoint_storage["effective"]["storage_profile"]["profile_id"] == (
        "published_http_v1"
    )
    assert plan.keypoint_storage["effective"]["chunk_derivation"] == (
        "dtype_itemsize_times_per_row_shape_to_byte_budget"
    )
    assert plan.finalization_execution["effective"] == {
        "algorithm": "strict_keypoint_v2_chain_finalizer_v1",
        "write_ownership": "serial_whole_physical_units",
        "storage_planning": "shared_byte_planner_per_array",
        "publication": "atomic_per_run_then_single_root_consolidation",
        "selector_activation": False,
    }
    assert (
        plan.finalization_execution["requested_legacy_controls"]
        ["effect_on_v2_finalization"]
        == "none"
    )
    assert len(plan.lsf_workflow.jobs) == 5
    assert [job.job_key for job in plan.lsf_workflow.topological_jobs()] == [
        "predict:target_0",
        "predict:target_1",
        "refine:target_0",
        "refine:target_1",
        "registry_finalize",
    ]
    jobs = {job.job_key: job for job in plan.lsf_workflow.jobs}
    assert jobs["refine:target_0"].dependency.upstream_job_keys == (
        "predict:target_0",
    )
    assert jobs["registry_finalize"].dependency.upstream_job_keys == (
        "refine:target_0",
        "refine:target_1",
    )

    prediction_command = jobs["predict:target_0"].command
    assert prediction_command[:3] == (
        str(plan.repo / "scripts" / "py"),
        "-m",
        "fisheye.cluster.lsf.runtime",
    )
    assert "PALETTE_DISABLE_REGISTRY_WRITES=1" in prediction_command
    assert "PALETTE_COMMIT=" + ("a" * 40) in prediction_command
    assert jobs["predict:target_0"].metadata["palette_commit"] == "a" * 40
    assert "fisheye.utils.run_whole_recording_keypoint_terminal" in prediction_command
    assert "--model-run-id" in prediction_command
    assert prediction_command[prediction_command.index("--model-run-id") + 1] == (
        "pose_run"
    )
    assert "--cache-manifest" in prediction_command
    assert prediction_command[prediction_command.index("--model-input-size") + 1] == (
        "512"
    )
    assert prediction_command[
        prediction_command.index("--network-input-size") + 1
    ] == "256"
    assert prediction_command[prediction_command.index("--input-mode") + 1] == (
        "numpy-list"
    )
    assert prediction_command[
        prediction_command.index("--model-input-transform-mode") + 1
    ] == "pad_to_size"
    assert prediction_command[
        prediction_command.index("--model-input-stride") + 1
    ] == "32"
    assert jobs["predict:target_0"].metadata["model_input_stride"] == 32
    assert "--keypoint-roi-shard-rows" not in prediction_command
    assert "--keypoint-frame-shard-rows" not in prediction_command
    assert "--expected-output" in prediction_command
    assert (
        "/scratch/__PALETTE_LSF_USER__/__PALETTE_LSF_JOBID__/"
        "palette_keypoint_terminal"
    ) in (
        prediction_command
    )
    assert all("<jobid>" not in arg and "<user>" not in arg for arg in prediction_command)
    refinement_command = jobs["refine:target_0"].command
    assert "fisheye.utils.finalize_whole_recording_keypoint_v2" in refinement_command
    assert "--raw-run" in refinement_command
    assert "--quality-run" in refinement_command
    assert "--body-frame-run" in refinement_command
    assert "keypoints_goodcopbadcop_20260710" in refinement_command
    assert "refined_keypoints_goodcopbadcop_20260710" in refinement_command
    assert "--apply" not in jobs["registry_finalize"].command
    assert jobs["registry_finalize"].metadata["registry_mutation"] is False


def test_whole_recording_analysis_plan_forks_inference_and_joins_finalization(
    tmp_path: Path,
    monkeypatch,
) -> None:
    keypoint_plan = _build_plan(tmp_path, monkeypatch, target_count=2)
    plan = analysis_planner.build_plan(
        keypoint_plan=keypoint_plan,
        run_root=tmp_path / "combined",
        mask_run_label="goodcopbadcop_masks_20260712",
        mask_inference_resources=LsfResources(
            queue="gpu_l4", ncores=8, mem_gb=48, gpus=1
        ),
        mask_finalization_resources=LsfResources(
            queue="short", ncores=16, mem_gb=32
        ),
        handoff_package_dir=tmp_path / "handoff",
        cleanup_roi_caches=True,
        roi_cache_cleanup_allowed_root=tmp_path / "caches",
    )

    jobs = {job.job_key: job for job in plan.lsf_workflow.jobs}
    assert plan.to_json()["schema"] == analysis_planner.PLAN_SCHEMA
    assert jobs["predict:target_0"].dependency is None
    assert jobs["mask_infer:target_0"].dependency is None
    assert jobs["refine:target_0"].dependency.upstream_job_keys == (
        "predict:target_0",
    )
    assert jobs["mask_finalize:target_0"].dependency.upstream_job_keys == (
        "mask_infer:target_0",
        "refine:target_0",
    )
    assert jobs["mask_publish:target_0"].dependency.upstream_job_keys == (
        "mask_finalize:target_0",
    )
    assert jobs["analysis_validate"].dependency.upstream_job_keys == (
        "mask_publish:target_0",
        "mask_publish:target_1",
    )
    assert jobs["registry_finalize"].dependency.upstream_job_keys == (
        "analysis_validate",
    )
    assert jobs["roi_cache_cleanup"].dependency.upstream_job_keys == (
        "registry_finalize",
    )
    assert plan.roi_cache_cleanup_job_key == "roi_cache_cleanup"
    assert "fisheye.cluster.whole_recording_analysis_cache_cleanup" in (
        jobs["roi_cache_cleanup"].command
    )
    assert jobs["roi_cache_cleanup"].metadata["destructive_cleanup"] is True
    assert [job.job_key for job in plan.lsf_workflow.topological_jobs()][-3:] == [
        "analysis_validate",
        "registry_finalize",
        "roi_cache_cleanup",
    ]
    assert plan.analysis_validation_job_key == "analysis_validate"
    assert "fisheye.cluster.whole_recording_analysis_validate" in (
        jobs["analysis_validate"].command
    )
    assert "fisheye.cluster.whole_recording_analysis_registry_finalize" in (
        jobs["registry_finalize"].command
    )

    inference_command = jobs["mask_infer:target_0"].command
    assert "inference" in inference_command
    assert "--refined-keypoint-run" not in inference_command
    assert "--roi-cache-manifest" in inference_command
    assert "--roi-cache-staging-dir" in inference_command
    assert "--raw-worker-run" in inference_command

    finalization_command = jobs["mask_finalize:target_0"].command
    assert "finalization" in finalization_command
    exact_index = finalization_command.index("--refined-keypoint-run") + 1
    assert (
        finalization_command[exact_index]
        == "refined_keypoints_goodcopbadcop_20260710"
    )
    assert all("latest" not in argument for argument in finalization_command)
    assert jobs["mask_finalize:target_0"].metadata[
        "sampled_component_contours_requested"
    ] is True
    assert jobs["mask_finalize:target_0"].metadata[
        "component_contours_requested"
    ] is False
    publication_command = jobs["mask_publish:target_0"].command
    assert "fisheye.cluster.subject_masks.publish_recording_bundle" in (
        publication_command
    )
    assert "--activate" not in publication_command
    assert jobs["mask_publish:target_0"].metadata["selector_activation"] is False


def test_whole_recording_analysis_composes_cache_builds_into_both_inference_branches(
    tmp_path: Path,
    monkeypatch,
) -> None:
    keypoint_plan = _build_plan(
        tmp_path,
        monkeypatch,
        target_count=2,
        planned_caches=True,
    )
    plan = analysis_planner.build_plan(
        keypoint_plan=keypoint_plan,
        run_root=tmp_path / "combined",
        mask_run_label="masks_with_cache",
        mask_inference_resources=LsfResources(
            queue="gpu_l4", ncores=8, mem_gb=48, gpus=1
        ),
        mask_finalization_resources=LsfResources(
            queue="short", ncores=16, mem_gb=32
        ),
        cleanup_roi_caches=True,
        roi_cache_cleanup_allowed_root=tmp_path / "caches",
    )

    jobs = {job.job_key: job for job in plan.lsf_workflow.jobs}
    assert jobs["predict:target_0"].dependency.upstream_job_keys == (
        "cache:target_0",
    )
    assert jobs["mask_infer:target_0"].dependency.upstream_job_keys == (
        "cache:target_0",
    )
    assert jobs["refine:target_0"].dependency.upstream_job_keys == (
        "predict:target_0",
    )
    assert jobs["mask_finalize:target_0"].dependency.upstream_job_keys == (
        "mask_infer:target_0",
        "refine:target_0",
    )
    assert jobs["mask_publish:target_0"].dependency.upstream_job_keys == (
        "mask_finalize:target_0",
    )
    assert [
        fragment["fragment_id"]
        for fragment in plan.lsf_workflow.to_json()["metadata"]["fragments"]
    ] == [
        "roi_cache",
        "keypoints",
        "subject_masks",
        "analysis_validation",
        "registry",
        "cache_cleanup",
    ]
    assert plan.targets[0].roi_cache_manifest_sha256 is None
    assert plan.targets[0].roi_cache_availability == "planned"
    assert plan.targets[0].roi_cache_producer_job_key == "cache:target_0"

    analysis_planner.materialize_plan_bundle(plan)
    contract_path = plan.run_root / "cache_contracts" / "target_0.json"
    assert json.loads(contract_path.read_text(encoding="utf-8")) == dict(
        plan.targets[0].roi_cache_contract
    )


def test_whole_recording_analysis_limits_active_targets_with_rolling_gate(
    tmp_path: Path,
    monkeypatch,
) -> None:
    keypoint_plan = _build_plan(
        tmp_path,
        monkeypatch,
        target_count=10,
        planned_caches=True,
    )
    plan = analysis_planner.build_plan(
        keypoint_plan=keypoint_plan,
        run_root=tmp_path / "combined",
        mask_run_label="masks_with_concurrency_gate",
        mask_inference_resources=LsfResources(
            queue="gpu_l4", ncores=8, mem_gb=48, gpus=1
        ),
        mask_finalization_resources=LsfResources(
            queue="short", ncores=16, mem_gb=32
        ),
        max_active_targets=8,
    )

    jobs = {job.job_key: job for job in plan.lsf_workflow.jobs}
    assert plan.max_active_targets == 8
    assert plan.targets[7].target_concurrency_gate_job_key is None
    assert plan.targets[8].target_concurrency_gate_job_key == "mask_publish:target_0"
    assert plan.targets[9].target_concurrency_gate_job_key == "mask_publish:target_1"
    assert jobs["cache:target_8"].dependency.upstream_job_keys == (
        "mask_publish:target_0",
    )
    assert jobs["cache:target_9"].dependency.upstream_job_keys == (
        "mask_publish:target_1",
    )
    assert jobs["predict:target_8"].dependency.upstream_job_keys == (
        "cache:target_8",
    )
    assert jobs["mask_infer:target_8"].dependency.upstream_job_keys == (
        "cache:target_8",
    )
    assert jobs["cache:target_8"].metadata["target_concurrency_gate_job_key"] == (
        "mask_publish:target_0"
    )
    assert plan.lsf_workflow.metadata["target_concurrency_contract"] == (
        "rolling_finalization_gate_v1"
    )


def test_rolling_gate_applies_after_shared_cache_bundle_without_cycle(
    tmp_path: Path,
    monkeypatch,
) -> None:
    keypoint_plan = _build_plan(
        tmp_path,
        monkeypatch,
        target_count=2,
        planned_caches=True,
        shared_cache_bundle=True,
    )
    plan = analysis_planner.build_plan(
        keypoint_plan=keypoint_plan,
        run_root=tmp_path / "combined",
        mask_run_label="masks_shared_cache_bundle",
        mask_inference_resources=LsfResources(
            queue="gpu_l4", ncores=8, mem_gb=48, gpus=1
        ),
        mask_finalization_resources=LsfResources(
            queue="short", ncores=16, mem_gb=32
        ),
        max_active_targets=1,
    )

    jobs = {job.job_key: job for job in plan.lsf_workflow.jobs}
    assert jobs["cache_bundle:000"].dependency is None
    assert jobs["predict:target_1"].dependency.upstream_job_keys == (
        "cache_bundle:000",
        "mask_publish:target_0",
    )
    assert jobs["mask_infer:target_1"].dependency.upstream_job_keys == (
        "cache_bundle:000",
        "mask_publish:target_0",
    )
    assert [job.job_key for job in plan.lsf_workflow.topological_jobs()].index(
        "cache_bundle:000"
    ) < [job.job_key for job in plan.lsf_workflow.topological_jobs()].index(
        "mask_finalize:target_0"
    )


def test_whole_recording_v2_ignores_legacy_row_shard_override(
    tmp_path: Path,
    monkeypatch,
) -> None:
    plan = _build_plan(
        tmp_path,
        monkeypatch,
        target_count=1,
        keypoint_roi_shard_rows=None,
    )
    prediction = next(
        job for job in plan.lsf_workflow.jobs if job.job_key == "predict:target_0"
    )

    assert plan.keypoint_storage["requested"]["legacy_keypoint_roi_shard_rows"] is None
    assert plan.keypoint_storage["requested"]["effect_on_v2_publication"] == "none"
    assert plan.keypoint_storage["effective"]["keypoint_storage_layout"] == (
        "shared_byte_planned_indexed_sharding_v1"
    )
    assert "--no-keypoint-sharding" not in prediction.command
    assert "--keypoint-roi-shard-rows" not in prediction.command


def test_plan_bundle_is_durable_and_apply_submits_only_through_shared_backend(
    tmp_path: Path,
    monkeypatch,
) -> None:
    plan = _build_plan(tmp_path, monkeypatch, target_count=1)
    payload = planner.materialize_plan_bundle(plan)

    assert payload["target_count"] == 1
    assert json.loads((plan.run_root / "plan.json").read_text(encoding="utf-8"))[
        "schema"
    ] == planner.PLAN_SCHEMA
    assert (plan.run_root / "zarr_paths.txt").read_text(encoding="utf-8").strip() == str(
        plan.targets[0].target.analysis_zarr
    )
    assert (plan.run_root / "status").is_dir()

    job_ids = iter(("101", "102", "103"))
    calls: list[list[str]] = []

    def fake_runner(argv, **_kwargs):
        calls.append(list(argv))
        return SimpleNamespace(
            returncode=0,
            stdout=f"Job <{next(job_ids)}> is submitted to queue <test>.",
            stderr="",
        )

    result = planner.apply_plan(plan, runner=fake_runner)

    assert result["status"] == "submitted"
    assert [call[call.index("-J") + 1] for call in calls] == [
        plan.lsf_workflow.topological_jobs()[0].job_name,
        plan.lsf_workflow.topological_jobs()[1].job_name,
        plan.lsf_workflow.topological_jobs()[2].job_name,
    ]
    assert calls[1][calls[1].index("-w") + 1] == "done(101)"
    assert calls[2][calls[2].index("-w") + 1] == "done(102)"


def test_plan_refuses_existing_deterministic_output(tmp_path: Path, monkeypatch) -> None:
    manifest, _repo, _registry, _model = _make_inputs(tmp_path, target_count=1)
    target = planner.load_target_manifest(manifest)[0]
    collision = (
        target.analysis_zarr
        / "keypoints_runs"
        / "keypoints_goodcopbadcop_20260710"
    )
    collision.mkdir(parents=True)

    with pytest.raises(FileExistsError, match="Planned output already exists"):
        _build_plan(tmp_path, monkeypatch, target_count=1)


class _FakeGroup(dict):
    def __init__(self, **attrs: object) -> None:
        super().__init__()
        self.attrs = dict(attrs)


def test_keypoint_input_dag_accepts_large_cache_when_other_sources_are_unavailable(
    tmp_path: Path,
    monkeypatch,
) -> None:
    crop = _FakeGroup(
        crop_storage_mode="geometry_only",
        roi_size=[348, 348],
        crop_signature="signature-a",
        crop_revision=0,
        source_pixels="acquisition_crop_video",
        source_crop_video_path=str(tmp_path / "missing_crop.mp4"),
    )
    crop["roi_coordinates_full"] = SimpleNamespace(shape=(2, 2))
    crop["frame_indices"] = SimpleNamespace(shape=(2,))
    crop_parent = _FakeGroup()
    crop_parent["crop_001"] = crop
    root = _FakeGroup()
    root["crop_runs"] = crop_parent
    monkeypatch.setattr(
        common_mod,
        "open_zarr_group_direct",
        lambda *_args, **_kwargs: root,
    )
    monkeypatch.setattr(
        common_mod,
        "is_run_complete_in_parent",
        lambda *_args, **_kwargs: True,
    )
    cache = common_mod.FlatRoiCacheBinding(
        manifest_path=tmp_path / "cache.json",
        manifest_sha256="a" * 64,
        payload_path=tmp_path / "cache.bin",
        crop_run="crop_001",
        cache_key="cache-a",
        crop_signature="signature-a",
        crop_revision=0,
        shape=(2, 348, 348),
        total_bytes=2 * 348 * 348,
        payload_sha256=None,
        pixel_contract=orange_mono_pynvvc_luma_pixel_contract(),
    )

    capability = common_mod.validate_keypoint_input_dag(
        analysis_zarr=tmp_path / "analysis.zarr",
        cache=cache,
        min_roi_size=348,
    )

    assert capability.selected_source == "flat_roi_cache"
    assert capability.flat_roi_cache_eligible is True
    assert capability.persisted_roi_images_eligible is False
    assert capability.acquisition_crop_video_eligible is False
    assert capability.acquisition_crop_video_probe_status == "not_available"
    assert capability.rejected_sources == {
        "persisted_roi_images": "missing",
        "acquisition_crop_video": "missing_video_file",
    }


def test_keypoint_input_dag_rejects_crop_below_zebrafish_minimum(
    tmp_path: Path,
    monkeypatch,
) -> None:
    crop = _FakeGroup(
        crop_storage_mode="geometry_only",
        roi_size=[320, 320],
        crop_signature="signature-a",
        crop_revision=0,
    )
    crop["roi_coordinates_full"] = SimpleNamespace(shape=(2, 2))
    crop["frame_indices"] = SimpleNamespace(shape=(2,))
    crop_parent = _FakeGroup()
    crop_parent["crop_001"] = crop
    root = _FakeGroup()
    root["crop_runs"] = crop_parent
    monkeypatch.setattr(
        common_mod,
        "open_zarr_group_direct",
        lambda *_args, **_kwargs: root,
    )
    monkeypatch.setattr(
        common_mod,
        "is_run_complete_in_parent",
        lambda *_args, **_kwargs: True,
    )
    cache = common_mod.FlatRoiCacheBinding(
        manifest_path=tmp_path / "cache.json",
        manifest_sha256="a" * 64,
        payload_path=tmp_path / "cache.bin",
        crop_run="crop_001",
        cache_key="cache-a",
        crop_signature="signature-a",
        crop_revision=0,
        shape=(2, 320, 320),
        total_bytes=2 * 320 * 320,
        payload_sha256=None,
    )

    with pytest.raises(ValueError, match="requires at least 348x348"):
        common_mod.validate_keypoint_input_dag(
            analysis_zarr=tmp_path / "analysis.zarr",
            cache=cache,
            min_roi_size=348,
        )


def test_registered_geometry_keypoint_guard_requires_exact_finalized_crop_lineage(
    tmp_path: Path,
    monkeypatch,
) -> None:
    gate = {
        "requirement": "required",
        "status": "applied",
        "applied": True,
        "gate_run": "gate_001",
        "selection_digest": "a" * 64,
    }
    crop = _FakeGroup(
        source_refined_run_id="refined_final",
        source_refined_manifest_digest="b" * 64,
        source_registered_detection_gate_requirement="required",
        source_registered_detection_gate=gate,
    )
    crop_parent = _FakeGroup()
    crop_parent["crop_001"] = crop
    root = _FakeGroup()
    root["crop_runs"] = crop_parent
    source_run = SimpleNamespace(
        attrs={
            "finalized_recording_authority": True,
            "immutable_snapshot": True,
            "registered_detection_gate_requirement": "required",
            "registered_detection_gate": gate,
        }
    )
    monkeypatch.setattr(common_mod, "open_zarr_group_direct", lambda *_a, **_k: root)
    monkeypatch.setattr(
        common_mod,
        "is_run_complete_in_parent",
        lambda *_args, **_kwargs: True,
    )
    monkeypatch.setattr(
        common_mod,
        "bind_refined_detection_crop_source",
        lambda *_args, **_kwargs: SimpleNamespace(
            manifest={"payload_digest": "b" * 64},
            run_group=source_run,
        ),
    )

    result = common_mod.validate_registered_geometry_crop_authority(
        analysis_zarr=tmp_path / "analysis.zarr",
        crop_run="crop_001",
        registered_gate_requirement="required",
        registered_gate_run="gate_001",
    )
    assert result["source_refined_run"] == "refined_final"
    assert result["gate_applied"] is True

    with pytest.raises(ValueError, match="different registered gate"):
        common_mod.validate_registered_geometry_crop_authority(
            analysis_zarr=tmp_path / "analysis.zarr",
            crop_run="crop_001",
            registered_gate_requirement="required",
            registered_gate_run="gate_other",
        )


def test_candidate_validator_requires_all_four_v2_runs_in_plan(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    _write_json(
        run_root / "plan.json",
        {
            "schema": planner.PLAN_SCHEMA,
            "targets": [
                {
                    "target": {
                        "target_id": "target_a",
                        "recording_id": "recording_a",
                        "analysis_zarr": str(tmp_path / "recording_analysis.zarr"),
                    },
                    "run_names": {
                        "keypoint_run": "raw",
                        "refined_keypoint_run": "refined",
                    },
                    "model": {
                        "set_id": "pose_set",
                        "run_id": "pose_run",
                        "model_sha256": "a" * 64,
                    },
                    "cache": {"crop_run": "crop_001"},
                    "finalization_result": str(run_root / "finalization" / "a.json"),
                }
            ],
        },
    )
    with pytest.raises(ValueError, match="incomplete"):
        registry_finalize_mod.finalize_registry(
            run_root,
            registry_path=tmp_path / "registry.sqlite",
            apply=False,
        )


def test_candidate_validator_forbids_registry_or_selector_activation(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="separate reviewed selector/registry"):
        registry_finalize_mod.finalize_registry(
            tmp_path / "run",
            registry_path=tmp_path / "registry.sqlite",
            apply=True,
        )
