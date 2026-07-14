from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from fisheye.cluster import clipped_inference as workflow
from fisheye.cluster.clipped_inference_cleanup import cleanup
from fisheye.cluster.clipped_inference_validate import _instance_keys


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _target(tmp_path: Path, name: str = "sleepyfish_cam2010093") -> workflow.CampaignTarget:
    recording = tmp_path / "recordings" / name
    zarr = recording / "zarr" / f"{name}_analysis.zarr"
    _write_json(recording / "recording_clip_index.json", {"clips": []})
    _write_json(zarr / "zarr.json", {"zarr_format": 3, "node_type": "group", "attributes": {}})
    return workflow.CampaignTarget(
        target_id=name,
        recording_id=f"{name}:zfixture",
        recording_dir=recording,
        analysis_zarr=zarr,
    )


def _detection_plan(target: workflow.CampaignTarget, workflow_id: str) -> dict[str, object]:
    work_units = []
    for index in range(22):
        clip_id = f"clip_{index:06d}"
        camera = "2010093"
        detect = f"detect_{workflow_id}_{clip_id}"
        refined = f"refined_detect_{workflow_id}_{clip_id}"
        work_units.append(
            {
                "clip_id": clip_id,
                "clip_index": index,
                "camera_serial": camera,
                "work_unit_id": f"{target.target_id}_{clip_id}",
                "source": {"video_path": str(target.recording_dir / "clips" / f"{clip_id}.mp4")},
                "run_names": {
                    "detect": detect,
                    "detect_quality": f"quality_{workflow_id}_{clip_id}",
                    "refined_detect": refined,
                },
                "zarr_paths": {
                    "detect_target_group_path": f"clips/{clip_id}/cameras/{camera}/detect_runs/{detect}",
                    "refined_group_path": f"clips/{clip_id}/cameras/{camera}/refined_detect_runs/{refined}",
                },
            }
        )
    return {"work_unit_count": 22, "work_units": work_units}


def _build_fixture_plan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    resume_existing_detections: bool = False,
) -> workflow.ClippedInferencePlan:
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    (repo / "scripts" / "py").write_text("#!/bin/sh\n", encoding="utf-8")
    (repo / "configs" / "fisheye").mkdir(parents=True)
    (repo / "configs" / "fisheye" / "yolo_detect_config.yaml").write_text("{}\n", encoding="utf-8")
    (repo / "configs" / "fisheye" / "default.yaml").write_text("{}\n", encoding="utf-8")
    target = _target(tmp_path)
    model = tmp_path / "models" / "model.pt"
    model.parent.mkdir()
    model.write_bytes(b"model")
    detect_binding = workflow.ModelBinding("detect", "detect_set", "detect_run", model, "d" * 64)
    subject_binding = workflow.ModelBinding("subject_masks", "mask_set", "mask_run", model, "m" * 64)

    monkeypatch.setattr(workflow, "validate_registered_analysis_zarr", lambda **_kwargs: None)
    monkeypatch.setattr(workflow, "_resolve_ranked_binding", lambda **_kwargs: detect_binding)
    monkeypatch.setattr(workflow, "_resolve_subject_binding", lambda **_kwargs: subject_binding)
    monkeypatch.setattr(workflow, "_verify_binding", lambda _binding: None)
    monkeypatch.setattr(
        workflow,
        "resolve_pose_model_binding",
        lambda **_kwargs: SimpleNamespace(
            set_id="pose_set", run_id="pose_run", model_path=model, model_sha256="p" * 64
        ),
    )
    monkeypatch.setattr(
        workflow,
        "build_detection_plan",
        lambda recording_dir, **kwargs: _detection_plan(target, str(kwargs["workflow_id"])),
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
        targets=(target,),
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
    )


def test_build_plan_has_parallel_keypoint_mask_branch_and_join(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    plan = _build_fixture_plan(tmp_path, monkeypatch)
    jobs = {job.job_key: job for job in plan.lsf_workflow.jobs}
    target = plan.target_plans[0]
    target_safe = workflow.safe_component(str(target["target_id"]), default="target", max_length=56)
    clip_id = "clip_000000"

    assert len(plan.lsf_workflow.jobs) == 124
    assert jobs[f"keypoints:{target_safe}:{clip_id}"].dependency.upstream_job_keys == (
        f"proxy:{target_safe}",
    )
    assert jobs[f"subject_masks:{target_safe}:{clip_id}"].dependency.upstream_job_keys == (
        f"proxy:{target_safe}",
    )
    assert jobs[f"mask_package:{target_safe}:{clip_id}"].dependency.upstream_job_keys == (
        f"subject_masks:{target_safe}:{clip_id}",
        f"keypoint_refine:{target_safe}",
    )
    cache_job = jobs[f"cache:{target_safe}:00"]
    assert "--run-direct" in cache_job.command
    assert "bsub" not in cache_job.command
    assert all(job.command[:3] == ("scripts/py", "-m", "fisheye.cluster.lsf.runtime") for job in jobs.values())


def test_materialized_dry_run_is_immutable_and_has_no_submission(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
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


def test_resume_plan_revalidates_detections_on_cpu_and_preserves_dependencies(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _build_fixture_plan(tmp_path, monkeypatch, resume_existing_detections=True)
    jobs = {job.job_key: job for job in plan.lsf_workflow.jobs}
    target = plan.target_plans[0]
    target_safe = workflow.safe_component(str(target["target_id"]), default="target", max_length=56)
    clip_id = "clip_000000"
    detect = jobs[f"detect:{target_safe}:{clip_id}"]
    refine = jobs[f"detect_refine:{target_safe}:{clip_id}"]

    assert plan.resume_existing_detections is True
    assert len(target["detection_resume_preflight"]) == 22
    assert detect.metadata["stage"] == "detect_reuse"
    assert detect.resources.queue == "short"
    assert detect.resources.gpus == 0
    assert "--reuse-existing" in detect.command
    assert refine.dependency.upstream_job_keys == (detect.job_key,)


def test_existing_detection_resume_preflight_requires_exact_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = _target(tmp_path)
    binding = workflow.ModelBinding(
        "detect", "detect_set", "detect_run", (tmp_path / "model.pt").resolve(), "a" * 64
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
    group_metadata = target.analysis_zarr / planned_clip["detect_group_path"] / "zarr.json"
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
        target_label="campaign",
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
            target_label="campaign",
            clip=planned_clip,
            binding=binding,
        )


def test_instance_key_validation_rejects_duplicates() -> None:
    run = {"instance_key": np.asarray([11, 12, 12], dtype=np.uint64)}
    with pytest.raises(RuntimeError, match="duplicate instance_key"):
        _instance_keys(run, label="fixture")


def test_cleanup_is_confined_and_requires_registry_success(tmp_path: Path) -> None:
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
            "schema": workflow.PLAN_SCHEMA,
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
