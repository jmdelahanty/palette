from __future__ import annotations

from pathlib import Path

from fisheye.cluster import crimson_full_storage_candidate as mod


def _detection_plan(
    analysis: Path,
    recording_id: str,
    *,
    model: Path | None = None,
) -> dict[str, object]:
    work_units = []
    for index in range(2):
        clip = f"clip_{index:06d}"
        work_units.append(
            {
                "clip_index": index,
                "clip_id": clip,
                "frame_count": 100,
                "zarr_paths": {
                    "detect_target_group_path": (
                        f"clips/{clip}/cameras/1/detect_runs/detect_{clip}"
                    ),
                    "refined_group_path": (
                        f"clips/{clip}/cameras/1/refined_detect_runs/refined_{clip}"
                    ),
                },
            }
        )
    return {
        "recording_id": recording_id,
        "analysis_zarr": str(analysis),
        "model": str(model or analysis.parent / "detect.pt"),
        "work_unit_count": len(work_units),
        "work_units": work_units,
    }


def test_full_plan_pins_all_inputs_and_uses_recording_adapter(
    tmp_path: Path,
    monkeypatch,
) -> None:
    recording = tmp_path / "recording"
    analysis = recording / "analysis.zarr"
    detection_plan = recording / "plan.json"
    palette_repo = tmp_path / "palette"
    commit = "a" * 40
    source_sha = "b" * 64
    plan_sha = "c" * 64
    detection_model_sha = "1" * 64
    detection_model = recording / "detect.pt"
    detection_model.parent.mkdir(parents=True)
    detection_model.write_bytes(b"model")
    monkeypatch.setattr(
        mod,
        "_read_json",
        lambda _: _detection_plan(
            analysis,
            recording.name,
            model=detection_model,
        ),
    )
    monkeypatch.setattr(
        mod,
        "_sha256_file",
        lambda path: (
            source_sha
            if path.name == "zarr.json"
            else detection_model_sha
            if path == detection_model
            else plan_sha
        ),
    )
    monkeypatch.setattr(
        mod,
        "_git",
        lambda _repo, *args: commit if args == ("rev-parse", "HEAD") else "",
    )
    request = mod.CrimsonFullStorageCandidateRequest(
        candidate_id="sleepyfish_full_v1",
        recording_dir=recording,
        analysis_zarr=analysis,
        detection_plan_path=detection_plan,
        collection_id="collection",
        source_keypoint_group_path="keypoints_runs/full",
        source_keypoint_metadata_sha256=source_sha,
        expected_detection_model_sha256=detection_model_sha,
        expected_model_sha256="d" * 64,
        expected_n_frames=200,
        expected_canonical_n_instances=192,
        expected_n_instances=190,
        output_root=tmp_path / ".palette_benchmarks" / "candidate",
        palette_repo=palette_repo,
        palette_commit=commit,
        crimson_contract_commit="e" * 40,
        crimson_contract_sha256="f" * 64,
        camera_id="1",
    )

    plan = mod.build_full_storage_candidate_plan(request)

    payload = plan.plan_manifest["payload"]
    assert payload["classification"] == "full_duration_fixture"
    assert payload["inputs"]["source_keypoint_metadata_sha256"] == source_sha
    assert payload["inputs"]["detection_plan_sha256"] == plan_sha
    assert payload["inputs"]["expected_detection_model_sha256"] == detection_model_sha
    assert payload["inputs"]["canonical_source_role"] == (
        "native_v003_clipped_collection"
    )
    assert payload["dimensions"] == {
        "n_frames": 200,
        "canonical_n_instances": 192,
        "refined_n_instances": 190,
    }
    canonical_job = plan.candidate.workflow.jobs[0]
    assert "--canonical-anchor-archive" not in canonical_job.command
    assert (
        canonical_job.command[canonical_job.command.index("--expected-n-instances") + 1]
        == "192"
    )
    assert payload["publication"]["node_local_keypoint_materialization"] is True
    assert payload["publication"]["video_copy_included"] is False
    assert plan.candidate.workflow.metadata["selector_eligible"] is False
    modules = [
        fragment["metadata"]["module"]
        for fragment in plan.candidate.workflow.metadata["fragments"]
    ]
    assert modules == [
        "recording_canonical_detection_benchmark_adapter",
        "strict_clipped_detection_evidence",
        "clipped_storage_finalization",
        "recording_keypoint_v2_benchmark_adapter",
        "crimson_storage_candidate_handoff",
    ]
    evidence_job = plan.candidate.detection_storage.evidence.fragment.jobs[0]
    assert str(plan.lsf_plan_path) in evidence_job.command
    assert evidence_job.dependency is not None
    assert evidence_job.dependency.upstream_job_keys == (
        "canonical_detection_adapter:sleepyfish_full_v1",
    )
    assert plan.candidate.handoff_path == request.output_root / "handoff_manifest.json"
    for job in plan.candidate.workflow.jobs:
        if job.resources.walltime in {"2:00", "4:00"}:
            assert job.resources.queue == "local"


def test_detection_plan_requires_exact_contiguous_clips(tmp_path: Path) -> None:
    document = _detection_plan(tmp_path / "analysis.zarr", "recording")
    document["work_units"][1]["clip_index"] = 3

    try:
        mod._clips_from_detection_plan(document, candidate_id="candidate")
    except ValueError as exc:
        assert "ordered" in str(exc)
    else:
        raise AssertionError("non-contiguous detection plan was accepted")
