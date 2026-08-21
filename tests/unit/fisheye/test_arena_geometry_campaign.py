from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import fisheye.cluster.arena_geometry_campaign as arena_geometry_campaign
import fisheye.cluster.arena_geometry_review as arena_geometry_review
from fisheye.cluster.arena_geometry_campaign import (
    TARGET_MANIFEST_SCHEMA,
    build_plan,
    load_target_manifest,
)


def _target_files(tmp_path: Path) -> dict[str, Path]:
    recording = tmp_path / "recording"
    analysis = recording / "zarr" / "recording_analysis.zarr"
    analysis.mkdir(parents=True)
    (analysis / "zarr.json").write_text(
        json.dumps({"zarr_format": 3, "node_type": "group", "attributes": {}})
    )
    files = {
        "video": recording / "cams" / "Cam1.mp4",
        "summary": recording / "cams" / "Cam1_external_summary.json",
        "keyframes": recording / "cams" / "Cam1_keyframe.json",
        "receipt": recording / "raw" / "recording_geometry_recovery.json",
        "observation": recording / "raw" / "observation.json",
    }
    for path in files.values():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n")
    files["summary"].write_text(
        json.dumps(
            {
                "frames_received": 100,
                "fps": 100.0,
                "video_metadata": {
                    "camera_serial": "1",
                    "geometry": {"source_width": 640, "source_height": 480},
                },
            }
        )
    )
    return {"recording": recording, "analysis": analysis, **files}


def _manifest_row(paths: dict[str, Path]) -> dict[str, str]:
    return {
        "target_id": "recording_a",
        "recording_id": "recording-a",
        "recording_dir": str(paths["recording"]),
        "analysis_zarr": str(paths["analysis"]),
        "video": str(paths["video"]),
        "summary": str(paths["summary"]),
        "keyframes": str(paths["keyframes"]),
        "recovery_receipt": str(paths["receipt"]),
        "acquisition_observation": str(paths["observation"]),
    }


def test_target_manifest_preserves_exact_recording_bound_sources(
    tmp_path: Path,
) -> None:
    paths = _target_files(tmp_path)
    manifest = tmp_path / "targets.json"
    manifest.write_text(
        json.dumps(
            {"schema": TARGET_MANIFEST_SCHEMA, "targets": [_manifest_row(paths)]}
        )
    )

    (target,) = load_target_manifest(manifest)

    assert target.recording_id == "recording-a"
    assert target.video_path == paths["video"].resolve()
    assert target.keyframe_path == paths["keyframes"].resolve()
    assert target.acquisition_observation_path == paths["observation"].resolve()


def test_target_manifest_accepts_clipped_recording_metadata_source(
    tmp_path: Path,
) -> None:
    paths = _target_files(tmp_path)
    (paths["recording"] / "recording_clip_index.json").write_text("{}\n")
    row = _manifest_row(paths)
    for field in ("video", "summary", "keyframes"):
        row.pop(field)
    manifest = tmp_path / "targets.json"
    manifest.write_text(
        json.dumps({"schema": TARGET_MANIFEST_SCHEMA, "targets": [row]})
    )

    (target,) = load_target_manifest(manifest)

    assert target.video_path is None
    assert target.summary_path is None
    assert target.keyframe_path is None
    assert target.probe_source().to_json()["source_kind"] == (
        "recording_level_clipped_collection"
    )


def test_target_manifest_accepts_producer_native_folder_without_recovery_receipt(
    tmp_path: Path,
) -> None:
    paths = _target_files(tmp_path)
    row = _manifest_row(paths)
    row.pop("recovery_receipt")
    row.update(
        {
            "geometry_source": "producer-folder",
            "geometry_camera_serial": "2010093",
            "geometry_arena_id": "arena_1",
        }
    )
    manifest = tmp_path / "targets.json"
    manifest.write_text(
        json.dumps({"schema": TARGET_MANIFEST_SCHEMA, "targets": [row]})
    )

    (target,) = load_target_manifest(manifest)

    assert target.geometry_source == "producer-folder"
    assert target.geometry_camera_serial == "2010093"
    assert target.geometry_arena_id == "arena_1"
    assert target.recovery_receipt_path is None


def test_target_manifest_rejects_source_outside_recording(tmp_path: Path) -> None:
    paths = _target_files(tmp_path)
    outside = tmp_path / "other.mp4"
    outside.write_bytes(b"video")
    row = _manifest_row(paths)
    row["video"] = str(outside)
    manifest = tmp_path / "targets.json"
    manifest.write_text(
        json.dumps({"schema": TARGET_MANIFEST_SCHEMA, "targets": [row]})
    )

    with pytest.raises(ValueError, match="must belong to the recording"):
        load_target_manifest(manifest)


def test_target_manifest_rejects_partial_whole_video_source(tmp_path: Path) -> None:
    paths = _target_files(tmp_path)
    row = _manifest_row(paths)
    row.pop("summary")
    manifest = tmp_path / "targets.json"
    manifest.write_text(
        json.dumps({"schema": TARGET_MANIFEST_SCHEMA, "targets": [row]})
    )

    with pytest.raises(ValueError, match="requires video, summary, and keyframe"):
        load_target_manifest(manifest)


def test_target_manifest_rejects_non_v3_analysis_target(tmp_path: Path) -> None:
    paths = _target_files(tmp_path)
    (paths["analysis"] / "zarr.json").unlink()
    manifest = tmp_path / "targets.json"
    manifest.write_text(
        json.dumps(
            {"schema": TARGET_MANIFEST_SCHEMA, "targets": [_manifest_row(paths)]}
        )
    )

    with pytest.raises(FileNotFoundError, match="not Zarr v3"):
        load_target_manifest(manifest)


def test_campaign_plan_freezes_exact_sources_and_stops_before_review(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _target_files(tmp_path)
    manifest = tmp_path / "targets.json"
    manifest.write_text(
        json.dumps(
            {"schema": TARGET_MANIFEST_SCHEMA, "targets": [_manifest_row(paths)]}
        )
    )
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    (repo / "scripts" / "py").write_text("#!/bin/sh\n")
    monkeypatch.setattr(arena_geometry_campaign, "_repo_commit", lambda _repo: "a" * 40)
    monkeypatch.setattr(
        arena_geometry_campaign,
        "validate_registered_analysis_zarr",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        arena_geometry_review,
        "plan_recovered_acquisition_geometry_candidate",
        lambda **_kwargs: SimpleNamespace(
            candidate_id="arena-geometry-acquisition-test",
            target_run_path=(
                paths["analysis"]
                / "analysis"
                / "arena_geometry_runs"
                / "arena-geometry-acquisition-test"
            ),
        ),
    )
    registry = tmp_path / "registry.sqlite"
    registry.write_bytes(b"sqlite-fixture")

    plan = build_plan(
        targets=load_target_manifest(manifest),
        run_label="canary",
        repo=repo,
        registry_path=registry,
        run_root=tmp_path / "run",
        probe_array_concurrency=1,
        probe_queue="gpu_t4",
    )

    assert plan.repo_commit == "a" * 40
    assert len(plan.workflow.jobs) == 3
    assert plan.to_json()["post_review_publication"] == "not_scheduled"
    assert plan.to_json()["operational_selection"] == "not_scheduled"
    assert plan.to_json()["detection_gating"] == "not_scheduled"
    assert plan.to_json()["probe_queue"] == "gpu_t4"
    assert plan.workflow.jobs[1].resources.queue == "gpu_t4"
    assert plan.workflow.jobs[1].execution_group.max_concurrent == 1
    acquisition_task = plan.workflow.jobs[0].execution_group.tasks[0]
    acquisition_command = " ".join(acquisition_task.command)
    assert f"--analysis-zarr {paths['analysis']}" in acquisition_command
    assert "publish_arena_geometry_selection" not in acquisition_command
    registry_job = plan.workflow.jobs[2]
    registry_command = " ".join(registry_job.command)
    assert registry_job.dependency is not None
    assert registry_job.dependency.upstream_job_keys == (
        "arena_geometry_acquisition_array",
        "arena_geometry_probe_array",
    )
    assert "fisheye.utils.registry_rescan" in registry_command
    assert "--fail-on-error" in registry_command
    assert "--reconcile-step-status" in registry_command
    assert str(paths["analysis"]) in registry_command
    assert plan.to_json()["registry_update"] is True
