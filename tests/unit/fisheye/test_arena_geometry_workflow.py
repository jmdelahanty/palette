from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import fisheye.cluster.arena_geometry_review as arena_geometry_review
from fisheye.cluster.arena_geometry_review import (
    ArenaGeometryProbeSource,
    ArenaGeometryReviewFragmentInputs,
    ReviewedArenaGeometryCandidateFragmentInputs,
    build_arena_geometry_review_array_fragment,
    build_arena_geometry_review_fragment,
    build_reviewed_arena_geometry_candidate_fragment,
    compose_arena_geometry_workflow,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture_paths(tmp_path: Path) -> dict[str, Path]:
    recording = tmp_path / "recording"
    analysis = recording / "zarr" / "recording_analysis.zarr"
    repo = tmp_path / "repo"
    analysis.mkdir(parents=True)
    (analysis / "zarr.json").write_text(
        json.dumps({"zarr_format": 3, "node_type": "group", "attributes": {}})
    )
    (repo / "scripts").mkdir(parents=True)
    (repo / "scripts" / "py").write_text("#!/bin/sh\n")
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
    return {"recording": recording, "analysis": analysis, "repo": repo, **files}


def test_pre_review_fragment_is_recording_level_and_stops_before_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture_paths(tmp_path)
    candidate_path = paths["analysis"] / "analysis" / "arena_geometry_runs" / "a"
    monkeypatch.setattr(
        arena_geometry_review,
        "plan_recovered_acquisition_geometry_candidate",
        lambda **_: SimpleNamespace(
            candidate_id="arena-geometry-acquisition-a",
            target_run_path=candidate_path,
        ),
    )

    module = build_arena_geometry_review_fragment(
        ArenaGeometryReviewFragmentInputs(
            workflow_id="geometry_canary",
            target_id="recording_a",
            recording_dir=paths["recording"],
            analysis_zarr=paths["analysis"],
            recovery_receipt_path=paths["receipt"],
            source=ArenaGeometryProbeSource(
                video_path=paths["video"],
                summary_path=paths["summary"],
                keyframe_path=paths["keyframes"],
                acquisition_observation_path=paths["observation"],
            ),
            repo=paths["repo"],
            run_root=tmp_path / "run",
        )
    )

    acquisition, probe = module.fragment.jobs
    assert acquisition.dependency is None
    assert probe.dependency is None
    acquisition_command = " ".join(acquisition.command)
    assert f"--analysis-zarr {paths['analysis']}" in acquisition_command
    probe_command = " ".join(probe.command)
    assert "probe_recording_dish_rim_fit" in probe_command
    assert "publish_arena_geometry_fit_review" in probe_command
    assert f"--keyframes {paths['keyframes']}" in probe_command
    assert "--acquisition-observation" in probe_command
    assert "publish_reviewed_palette_geometry_candidate" not in probe_command
    assert module.fragment.metadata["geometry_scope"] == "recording_level"
    assert module.fragment.metadata["downstream_layouts"] == [
        "clipped",
        "whole_recording",
    ]
    assert module.fragment.metadata["human_review_barrier"] is True
    assert module.fragment.metadata["selection_activation"] == "deferred"
    assert module.fragment.metadata["registry_update"] is False
    assert module.outputs.review_receipt_path.name == "review_package.json"
    assert module.outputs.fit_review_import_receipt_path.name == (
        "fit_review_import.json"
    )
    assert module.outputs.to_json()["review_evidence_storage"] == (
        "analysis_zarr_embedded"
    )

    workflow = compose_arena_geometry_workflow(
        workflow_id="geometry_canary", modules=(module,)
    )
    assert len(workflow.jobs) == 2
    assert workflow.metadata["human_review_barrier"] is True


def test_pre_review_fragment_uses_producer_native_folder_without_recovery_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture_paths(tmp_path)
    candidate_path = paths["analysis"] / "analysis" / "arena_geometry_runs" / "native"
    monkeypatch.setattr(
        arena_geometry_review,
        "plan_producer_native_acquisition_geometry_candidate",
        lambda **_: SimpleNamespace(
            candidate_id="arena-geometry-acquisition-native",
            target_run_path=candidate_path,
        ),
    )

    module = build_arena_geometry_review_fragment(
        ArenaGeometryReviewFragmentInputs(
            workflow_id="geometry_native",
            target_id="recording_a",
            recording_dir=paths["recording"],
            analysis_zarr=paths["analysis"],
            recovery_receipt_path=None,
            geometry_source="producer-folder",
            geometry_camera_serial="2010093",
            geometry_arena_id="arena_1",
            source=ArenaGeometryProbeSource(
                video_path=paths["video"],
                summary_path=paths["summary"],
                keyframe_path=paths["keyframes"],
                acquisition_observation_path=paths["observation"],
            ),
            repo=paths["repo"],
            run_root=tmp_path / "run",
        )
    )

    command = " ".join(module.fragment.jobs[0].command)
    assert "--geometry-source producer-folder" in command
    assert "--camera-serial 2010093" in command
    assert "--arena-id arena_1" in command
    assert module.fragment.metadata["geometry_source"] == "producer-folder"


def test_campaign_fragment_uses_independent_lsf_arrays_and_indexed_scratch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture_paths(tmp_path)
    candidate_path = paths["analysis"] / "analysis" / "arena_geometry_runs" / "a"
    monkeypatch.setattr(
        arena_geometry_review,
        "plan_recovered_acquisition_geometry_candidate",
        lambda **_: SimpleNamespace(
            candidate_id="arena-geometry-acquisition-a",
            target_run_path=candidate_path,
        ),
    )
    inputs = ArenaGeometryReviewFragmentInputs(
        workflow_id="geometry_campaign",
        target_id="recording_a",
        recording_dir=paths["recording"],
        analysis_zarr=paths["analysis"],
        recovery_receipt_path=paths["receipt"],
        source=ArenaGeometryProbeSource(
            video_path=paths["video"],
            summary_path=paths["summary"],
            keyframe_path=paths["keyframes"],
            acquisition_observation_path=paths["observation"],
        ),
        repo=paths["repo"],
        run_root=tmp_path / "run",
    )

    module = build_arena_geometry_review_array_fragment((inputs,))
    acquisition_array, probe_array = module.fragment.jobs

    assert acquisition_array.metadata["execution_mode"] == "array"
    assert probe_array.metadata["execution_mode"] == "array"
    assert acquisition_array.dependency is None
    assert probe_array.dependency is None
    assert len(acquisition_array.execution_group.tasks) == 1
    assert len(probe_array.execution_group.tasks) == 1
    acquisition_command = " ".join(acquisition_array.execution_group.tasks[0].command)
    assert "__PALETTE_LSF_JOBID_____PALETTE_LSF_JOBINDEX__" in acquisition_command
    assert module.fragment.metadata["arrays_independent"] is True
    assert module.fragment.metadata["human_review_barrier"] is True


def test_direct_review_fragment_rejects_source_outside_recording(
    tmp_path: Path,
) -> None:
    paths = _fixture_paths(tmp_path)
    outside_video = tmp_path / "outside.mp4"
    outside_video.write_bytes(b"video")

    with pytest.raises(ValueError, match="source video must belong"):
        ArenaGeometryReviewFragmentInputs(
            workflow_id="geometry_campaign",
            target_id="recording_a",
            recording_dir=paths["recording"],
            analysis_zarr=paths["analysis"],
            recovery_receipt_path=paths["receipt"],
            source=ArenaGeometryProbeSource(
                video_path=outside_video,
                summary_path=paths["summary"],
                keyframe_path=paths["keyframes"],
            ),
            repo=paths["repo"],
            run_root=tmp_path / "run",
        )


def test_direct_review_fragment_rejects_malformed_summary_before_submit(
    tmp_path: Path,
) -> None:
    paths = _fixture_paths(tmp_path)
    paths["summary"].write_text("frame,timestamp\n0,0.0\n")

    with pytest.raises(ValueError, match="summary is not a JSON object"):
        ArenaGeometryProbeSource(
            video_path=paths["video"],
            summary_path=paths["summary"],
            keyframe_path=paths["keyframes"],
        )


def test_direct_review_fragment_rejects_clip_as_recording_geometry_source(
    tmp_path: Path,
) -> None:
    paths = _fixture_paths(tmp_path)
    clip_dir = paths["recording"] / "clips" / "clip_000000"
    clip_video = clip_dir / paths["video"].name
    clip_summary = clip_dir / paths["summary"].name
    clip_keyframes = clip_dir / paths["keyframes"].name
    clip_video.parent.mkdir(parents=True, exist_ok=True)
    clip_video.write_bytes(b"source")
    clip_summary.write_bytes(paths["summary"].read_bytes())
    clip_keyframes.write_bytes(paths["keyframes"].read_bytes())

    with pytest.raises(ValueError, match="recording-level video"):
        ArenaGeometryReviewFragmentInputs(
            workflow_id="geometry_campaign",
            target_id="recording_a",
            recording_dir=paths["recording"],
            analysis_zarr=paths["analysis"],
            recovery_receipt_path=paths["receipt"],
            source=ArenaGeometryProbeSource(
                video_path=clip_video,
                summary_path=clip_summary,
                keyframe_path=clip_keyframes,
            ),
            repo=paths["repo"],
            run_root=tmp_path / "run",
        )


def test_post_review_publication_rejects_changed_bound_artifact(
    tmp_path: Path,
) -> None:
    paths = _fixture_paths(tmp_path)
    report = tmp_path / "fit_report.json"
    montage = tmp_path / "review.png"
    receipt = tmp_path / "review_package.json"
    report.write_text('{"fit":1}\n')
    montage.write_bytes(b"image")
    receipt.write_text(
        json.dumps(
            {
                "schema_id": arena_geometry_review.REVIEW_PACKAGE_SCHEMA_ID,
                "schema_version": 1,
                "status": "awaiting_explicit_human_review",
                "fit_report": {"sha256": _sha256(report)},
                "montage": {"sha256": _sha256(montage)},
            }
        )
    )
    report.write_text('{"fit":2}\n')

    with pytest.raises(ValueError, match="changed after review packaging"):
        build_reviewed_arena_geometry_candidate_fragment(
            ReviewedArenaGeometryCandidateFragmentInputs(
                workflow_id="geometry_reviewed",
                target_id="recording_a",
                analysis_zarr=paths["analysis"],
                fit_report_path=report,
                review_montage_path=montage,
                review_receipt_path=receipt,
                reviewer="reviewer",
                reviewed_at_utc="2026-07-29T12:00:00+00:00",
                repo=paths["repo"],
                run_root=tmp_path / "run",
            )
        )


def test_post_review_publication_remains_pointerless_and_separate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture_paths(tmp_path)
    report = tmp_path / "fit_report.json"
    montage = tmp_path / "review.png"
    receipt = tmp_path / "review_package.json"
    report.write_text('{"fit":1}\n')
    montage.write_bytes(b"image")
    receipt.write_text(
        json.dumps(
            {
                "schema_id": arena_geometry_review.REVIEW_PACKAGE_SCHEMA_ID,
                "schema_version": 1,
                "status": "awaiting_explicit_human_review",
                "fit_report": {"sha256": _sha256(report)},
                "montage": {"sha256": _sha256(montage)},
            }
        )
    )
    candidate_path = paths["analysis"] / "analysis" / "arena_geometry_runs" / "p"
    monkeypatch.setattr(
        arena_geometry_review,
        "plan_reviewed_palette_geometry_candidate",
        lambda **_: SimpleNamespace(
            candidate_id="arena-geometry-palette-p",
            target_run_path=candidate_path,
        ),
    )

    module = build_reviewed_arena_geometry_candidate_fragment(
        ReviewedArenaGeometryCandidateFragmentInputs(
            workflow_id="geometry_reviewed",
            target_id="recording_a",
            analysis_zarr=paths["analysis"],
            fit_report_path=report,
            review_montage_path=montage,
            review_receipt_path=receipt,
            reviewer="reviewer",
            reviewed_at_utc="2026-07-29T12:00:00+00:00",
            repo=paths["repo"],
            run_root=tmp_path / "run",
        )
    )

    command = " ".join(module.fragment.jobs[0].command)
    assert "publish_reviewed_palette_geometry_candidate" in command
    assert module.fragment.metadata["selection_activation"] == "deferred"
    assert module.fragment.metadata["detection_gate_applied"] is False
    assert module.fragment.metadata["registry_update"] is False


def test_post_review_publication_prefers_embedded_fit_review_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture_paths(tmp_path)
    candidate_path = paths["analysis"] / "analysis" / "arena_geometry_runs" / "p"
    monkeypatch.setattr(
        arena_geometry_review,
        "plan_reviewed_palette_geometry_candidate",
        lambda **kwargs: SimpleNamespace(
            candidate_id=(
                "arena-geometry-palette-p"
                if kwargs["fit_review_run"] == "arena-geometry-fit-review-a"
                else "wrong"
            ),
            target_run_path=candidate_path,
        ),
    )

    module = build_reviewed_arena_geometry_candidate_fragment(
        ReviewedArenaGeometryCandidateFragmentInputs(
            workflow_id="geometry_reviewed",
            target_id="recording_a",
            analysis_zarr=paths["analysis"],
            fit_report_path=None,
            review_montage_path=None,
            review_receipt_path=None,
            fit_review_run="arena-geometry-fit-review-a",
            reviewer="reviewer",
            reviewed_at_utc="2026-08-13T12:00:00+00:00",
            repo=paths["repo"],
            run_root=tmp_path / "run",
        )
    )

    command = " ".join(module.fragment.jobs[0].command)
    assert "--fit-review-run arena-geometry-fit-review-a" in command
    assert "--fit-report" not in command
    assert module.fragment.metadata["review_source_ref"] == (
        "analysis/arena_geometry_fit_runs/arena-geometry-fit-review-a"
    )
