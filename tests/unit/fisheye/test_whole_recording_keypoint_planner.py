from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from fisheye.cluster.keypoints import common as common_mod
from fisheye.cluster.keypoints.common import KeypointInputCapability, PoseModelBinding
from fisheye.cluster.keypoints import registry_finalize as registry_finalize_mod
from fisheye.cluster.keypoints import whole_recording as planner
from fisheye.cluster.lsf import LsfResources


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
    plan = planner.build_plan(
        manifest_path=manifest,
        run_label="goodcopbadcop_20260710",
        repo=repo,
        registry=registry,
        run_root=tmp_path / "run",
        model_set_id="pose_set",
        model_run_id="pose_run",
        pose_schema="traditional_v2",
        min_roi_size=348,
        batch_size=256,
        device="0",
        input_mode="tensor",
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
    )
    return plan


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
    assert len(plan.targets) == 2
    assert len(plan.targets[0].cache.manifest_sha256) == 64
    assert plan.targets[0].input_capability.selected_source == "flat_roi_cache"
    assert plan.min_roi_size == 348
    assert plan.keypoint_storage["effective"] == {
        "keypoint_storage_layout": "indexed_sharding_v1",
        "keypoint_storage_policy": "default_indexed_sharding_v1",
        "keypoint_roi_shard_rows": 65536,
        "keypoint_frame_shard_rows": 262144,
    }
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
    assert "--model-run-id" in prediction_command
    assert prediction_command[prediction_command.index("--model-run-id") + 1] == (
        "pose_run"
    )
    assert "--stage-roi-cache-to-scratch" in prediction_command
    assert prediction_command[prediction_command.index("--keypoint-roi-shard-rows") + 1] == "65536"
    assert prediction_command[prediction_command.index("--keypoint-frame-shard-rows") + 1] == "262144"
    assert "--expected-output" in prediction_command
    assert "/scratch/<user>/<jobid>/palette_keypoint_roi_cache_stage" in (
        prediction_command
    )
    refinement_command = jobs["refine:target_0"].command
    assert "--keypoint-run" in refinement_command
    assert "keypoints_goodcopbadcop_20260710" in refinement_command
    assert "refined_keypoints_goodcopbadcop_20260710" in refinement_command


def test_whole_recording_plan_supports_regular_chunk_opt_out(
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

    assert plan.keypoint_storage["effective"] == {
        "keypoint_storage_layout": "regular_chunks_v1",
        "keypoint_storage_policy": "explicit_regular_chunks_override",
        "keypoint_roi_shard_rows": None,
        "keypoint_frame_shard_rows": None,
    }
    assert "--no-keypoint-sharding" in prediction.command
    assert "--keypoint-roi-shard-rows" not in prediction.command
    assert prediction.metadata["keypoint_storage"] == plan.keypoint_storage


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


def test_registry_finalizer_dry_run_requires_exact_source_lineage(
    tmp_path: Path,
    monkeypatch,
) -> None:
    run_root = tmp_path / "run"
    zarr_path = tmp_path / "recording_analysis.zarr"
    zarr_path.mkdir()
    _write_json(
        run_root / "plan.json",
        {
            "schema": planner.PLAN_SCHEMA,
            "targets": [
                {
                    "target": {
                        "target_id": "target_a",
                        "recording_id": "recording_a",
                        "analysis_zarr": str(zarr_path),
                    },
                    "run_names": {
                        "keypoint_run": "keypoints_test",
                        "refined_keypoint_run": "refined_keypoints_test",
                    },
                    "model": {
                        "set_id": "pose_set",
                        "run_id": "pose_run",
                        "model_path": str(tmp_path / "model.pt"),
                    },
                    "cache": {"crop_run": "crop_001"},
                }
            ],
        },
    )
    keypoint_run = _FakeGroup(
        model_resolution_selected_set_id="pose_set",
        model_resolution_selected_run_id="pose_run",
        model_resolution_selected_model_path=str(tmp_path / "model.pt"),
        source_crop_run="crop_001",
        summary_statistics={
            "total_rois": 2,
            "successful_detections": 2,
            "failed_detections": 0,
            "success_rate_percent": 100.0,
        }
    )
    refined_run = _FakeGroup(
        source_keypoints_run="keypoints_test",
        summary_statistics={
            "total_rois": 2,
            "refined_success": 2,
            "usable_keypoints": 2,
            "pass_rate_percent": 100.0,
        },
    )
    root = _FakeGroup()
    root["keypoints_runs"] = _FakeGroup()
    root["keypoints_runs"]["keypoints_test"] = keypoint_run
    root["refined_keypoints_runs"] = _FakeGroup()
    root["refined_keypoints_runs"]["refined_keypoints_test"] = refined_run
    monkeypatch.setattr(
        registry_finalize_mod,
        "open_zarr_group_direct",
        lambda *_args, **_kwargs: root,
    )
    monkeypatch.setattr(
        registry_finalize_mod,
        "is_run_complete_in_parent",
        lambda *_args, **_kwargs: True,
    )

    report = registry_finalize_mod.finalize_registry(
        run_root,
        registry_path=tmp_path / "missing-registry.sqlite",
        apply=False,
    )

    assert report["status"] == "ok"
    assert report["finalized_count"] == 1
    assert report["finalized"][0]["registry_status"] == "validated"

    refined_run.attrs["source_keypoints_run"] = "wrong_source"
    failed = registry_finalize_mod.finalize_registry(
        run_root,
        registry_path=tmp_path / "missing-registry.sqlite",
        apply=False,
    )
    assert failed["status"] == "error"
    assert "expected 'keypoints_test'" in failed["errors"][0]["error"]


def test_registry_finalizer_applies_prediction_then_refinement_serially(
    tmp_path: Path,
    monkeypatch,
) -> None:
    run_root = tmp_path / "run"
    zarr_path = tmp_path / "recording_analysis.zarr"
    zarr_path.mkdir()
    model_path = tmp_path / "model.pt"
    _write_json(
        run_root / "plan.json",
        {
            "schema": planner.PLAN_SCHEMA,
            "targets": [
                {
                    "target": {
                        "target_id": "target_a",
                        "recording_id": "recording_a",
                        "analysis_zarr": str(zarr_path),
                    },
                    "run_names": {
                        "keypoint_run": "keypoints_test",
                        "refined_keypoint_run": "refined_keypoints_test",
                    },
                    "model": {
                        "set_id": "pose_set",
                        "run_id": "pose_run",
                        "model_path": str(model_path),
                    },
                    "cache": {"crop_run": "crop_001"},
                }
            ],
        },
    )
    keypoint_run = _FakeGroup(
        method="yolo_pose",
        model_resolution_selected_set_id="pose_set",
        model_resolution_selected_run_id="pose_run",
        model_resolution_selected_model_path=str(model_path),
        source_crop_run="crop_001",
        summary_statistics={
            "total_rois": 2,
            "successful_detections": 2,
            "failed_detections": 0,
            "success_rate_percent": 100.0,
        },
    )
    refined_run = _FakeGroup(
        method="refine_keypoints",
        source_keypoints_run="keypoints_test",
        source_crop_run="crop_001",
        summary_statistics={
            "total_rois": 2,
            "refined_success": 2,
            "usable_keypoints": 2,
            "pass_rate_percent": 100.0,
        },
    )
    root = _FakeGroup()
    root["keypoints_runs"] = _FakeGroup()
    root["keypoints_runs"]["keypoints_test"] = keypoint_run
    root["refined_keypoints_runs"] = _FakeGroup()
    root["refined_keypoints_runs"]["refined_keypoints_test"] = refined_run
    monkeypatch.setattr(
        registry_finalize_mod,
        "open_zarr_group_direct",
        lambda *_args, **_kwargs: root,
    )
    monkeypatch.setattr(
        registry_finalize_mod,
        "is_run_complete_in_parent",
        lambda *_args, **_kwargs: True,
    )
    monkeypatch.setattr(
        registry_finalize_mod,
        "_check_registry_integrity",
        lambda _path: "ok",
    )
    calls: list[str] = []

    def refresh(**_kwargs):
        calls.append("performance")
        return {"keypoint_performance_refresh_status": "ok"}

    def emit_keypoints(*_args, **kwargs):
        calls.append(f"step:{kwargs['step_name']}")
        return True

    context = SimpleNamespace(dataset_id="dataset_a", recording_id="recording_a")

    def emit_refined(**_kwargs):
        calls.append("step:refined_keypoints")
        return True

    monkeypatch.setattr(
        registry_finalize_mod,
        "refresh_keypoint_performance_details",
        refresh,
    )
    monkeypatch.setattr(registry_finalize_mod, "emit_stage_completion", emit_keypoints)
    monkeypatch.setattr(
        registry_finalize_mod.refine_mod,
        "_resolve_status_context_from_root",
        lambda *_args, **_kwargs: context,
    )
    monkeypatch.setattr(
        registry_finalize_mod.refine_mod,
        "_emit_refined_keypoint_status",
        emit_refined,
    )
    registry = tmp_path / "registry.sqlite"
    registry.touch()

    report = registry_finalize_mod.finalize_registry(
        run_root,
        registry_path=registry,
        apply=True,
    )

    assert report["status"] == "ok"
    assert report["finalized_count"] == 1
    assert calls == ["performance", "step:keypoints", "step:refined_keypoints"]
    assert report["finalized"][0]["dataset_id"] == "dataset_a"
