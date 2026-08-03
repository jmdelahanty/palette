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

    plan = build_plan(
        targets=load_target_manifest(manifest),
        run_label="canary",
        repo=repo,
        registry_path=tmp_path / "registry.sqlite",
        run_root=tmp_path / "run",
    )

    assert plan.repo_commit == "a" * 40
    assert len(plan.workflow.jobs) == 2
    assert plan.to_json()["post_review_publication"] == "not_scheduled"
    assert plan.to_json()["operational_selection"] == "not_scheduled"
    assert plan.to_json()["detection_gating"] == "not_scheduled"
    acquisition_task = plan.workflow.jobs[0].execution_group.tasks[0]
    acquisition_command = " ".join(acquisition_task.command)
    assert f"--analysis-zarr {paths['analysis']}" in acquisition_command
    assert "publish_arena_geometry_selection" not in acquisition_command
