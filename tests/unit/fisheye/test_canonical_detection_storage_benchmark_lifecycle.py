from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
import zarr

from fisheye.diagnostics import run_canonical_detection_storage_benchmark_block as block
from fisheye.diagnostics.finalize_canonical_detection_storage_benchmark import (
    finalize_benchmark,
)
from fisheye.shared.zarr.benchmark_matrix import (
    BenchmarkLayout,
    BenchmarkScale,
    StorageCandidateRequest,
)
from fisheye.shared.zarr.benchmark_fixture import publish_benchmark_fixture
from fisheye.shared.zarr.benchmark_publication import publish_benchmark_candidate
from fisheye.shared.zarr.canonical_detection_benchmark import (
    build_canonical_detection_benchmark_input,
    load_canonical_detection_benchmark_input,
    write_detection_benchmark_candidate,
)
from fisheye.shared.zarr.detection_benchmark_matrix import (
    plan_canonical_detection_benchmark_matrix,
)
from fisheye.shared.zarr.detection_benchmark_reads import (
    benchmark_detection_candidate_reads,
)
from fisheye.shared.zarr.detection_benchmark_staging import (
    prepare_canonical_detection_benchmark_staging,
)
from fisheye.shared.zarr.detection_storage import plan_canonical_detection_storage
from fisheye.shared.zarr.storage_profiles import MIB, make_benchmark_storage_profile


class _Array:
    def __init__(self, values: np.ndarray) -> None:
        self.values = values

    def __getitem__(self, selection):
        return self.values[selection]


def _benchmark_input():
    return build_canonical_detection_benchmark_input(
        {
            "frame_indices": _Array(np.asarray([0, 0, 2, 4], dtype=np.int32)),
            "bbox_norm_coords": _Array(
                np.asarray(
                    [
                        [0.5, 0.5, 0.2, 0.2],
                        [0.4, 0.4, 0.1, 0.1],
                        [0.6, 0.6, 0.2, 0.2],
                        [0.3, 0.3, 0.1, 0.1],
                    ],
                    dtype=np.float64,
                )
            ),
            "scores": _Array(np.asarray([0.9, 0.8, 0.7, 0.6], dtype=np.float32)),
            "class_ids": _Array(np.asarray([0, 1, 0, 1], dtype=np.int32)),
        },
        recording_identity="recording-a",
        frame_count=5,
        source_width=640,
        source_height=480,
    )


def _thaw(root: Path) -> None:
    os.chmod(root, 0o755)
    for path in root.rglob("*"):
        os.chmod(path, 0o755 if path.is_dir() else 0o644)


def test_real_zarr_lifecycle_stages_writes_publishes_and_reads_exactly(
    tmp_path: Path,
) -> None:
    source = _benchmark_input()
    scratch = tmp_path / "scratch"
    staging = scratch / "canonical.zarr"

    staging_report = prepare_canonical_detection_benchmark_staging(
        source,
        destination=staging,
        scratch_root=scratch,
    )
    staged = load_canonical_detection_benchmark_input(staging)
    assert staging_report["status"] == "complete"
    assert staged.dimensions == source.dimensions
    assert all(
        np.array_equal(staged.arrays[path], source.arrays[path])
        for path in source.arrays
    )

    profile = make_benchmark_storage_profile(
        target_chunk_bytes=MIB,
        target_shard_bytes=8 * MIB,
        shard_immutable=True,
    )
    plans = plan_canonical_detection_storage(staged.dimensions, profile=profile)
    local_candidate = scratch / "candidate.zarr"
    local_report = write_detection_benchmark_candidate(
        staged,
        destination=local_candidate,
        plans=plans,
        benchmark_root=scratch,
    )
    assert all(
        bool(item["exact"])
        for item in local_report["digest_validation"].values()
    )

    workflow = tmp_path / "workflow"
    published = workflow / "candidates" / "matrix" / "candidate.zarr"
    publication = publish_benchmark_candidate(
        source=local_candidate,
        destination=published,
        workflow_root=workflow,
    )
    try:
        reads = benchmark_detection_candidate_reads(
            staged,
            candidate=published,
            plans=plans,
            storage_tier="test_shared_filesystem",
        )
        assert publication["exact_relative_path_size_content_match"] is True
        assert all(bool(item["exact"]) for item in reads["arrays"])
        assert (published.stat().st_mode & 0o777) == 0o555
        assert all(
            (path.stat().st_mode & 0o777) == (0o555 if path.is_dir() else 0o444)
            for path in published.rglob("*")
        )
    finally:
        _thaw(published)


def test_finalizer_requires_exact_planned_order_and_evidence(tmp_path: Path) -> None:
    workflow = tmp_path / "workflow"
    matrix = plan_canonical_detection_benchmark_matrix(
        matrix_id="tiny",
        scales=(
            BenchmarkScale.from_mapping(
                "frames_5",
                {
                    "n_frames": 5,
                    "n_instances": 4,
                    "source_width": 640,
                    "source_height": 480,
                },
            ),
        ),
        destination_root=workflow / "candidates",
        repetitions=1,
        candidate_requests=(
            StorageCandidateRequest(
                layout=BenchmarkLayout.REGULAR,
                target_chunk_bytes=MIB,
            ),
        ),
    ).as_manifest()
    matrix_path = workflow / "matrix.json"
    matrix_path.parent.mkdir(parents=True)
    matrix_path.write_text(json.dumps(matrix), encoding="utf-8")
    repetition = matrix["repetitions"][0]
    trial = repetition["trials"][0]
    candidate_id = str(trial["candidate_id"])
    candidate = matrix["candidates"][0]
    published = Path(str(trial["destination"]))
    published.mkdir(parents=True)
    (published / "zarr.json").write_text("{}", encoding="utf-8")
    evidence_paths = {
        field: workflow / "evidence" / f"{field}.json"
        for field in (
            "local_write_report",
            "publication_report",
            "prfs_read_report",
        )
    }
    for path in evidence_paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")
    block_path = workflow / "reports" / "blocks" / "frames_5_repetition_000.json"
    block_path.parent.mkdir(parents=True)
    block_path.write_text(
        json.dumps(
            {
                "status": "complete",
                "fixture_unchanged": True,
                "total_seconds": 1.0,
                "candidates": [
                    {
                        "candidate_id": candidate_id,
                        "physical_fingerprint": candidate["physical_fingerprint"],
                        "published_candidate": str(published),
                        **{key: str(value) for key, value in evidence_paths.items()},
                        "prfs_reads": {"all_exact": True},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    aggregate = finalize_benchmark(
        matrix_path=matrix_path,
        workflow_root=workflow,
        output=workflow / "aggregate.json",
    )

    assert aggregate["status"] == "complete"
    assert aggregate["summary"] == {
        "block_count": 1,
        "published_candidate_count": 1,
        "registry_updates": 0,
        "selector_updates": 0,
        "training_artifacts": 0,
        "profile_promoted": False,
    }


def test_tiny_block_runs_one_fixed_staging_candidate_and_copy_back(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "legacy-source.zarr"
    group = zarr.open_group(str(source), mode="w-", zarr_format=3)
    group.attrs.update({"source_video_width": 640, "source_video_height": 480})
    frames = np.asarray([0, 0, 2, 4], dtype=np.int32)
    group.create_array("frame_indices", data=frames, chunks=(4,))
    group.create_array(
        "bbox_norm_coords",
        data=np.asarray(
            [
                [0.5, 0.5, 0.2, 0.2],
                [0.4, 0.4, 0.1, 0.1],
                [0.6, 0.6, 0.2, 0.2],
                [0.3, 0.3, 0.1, 0.1],
            ],
            dtype=np.float64,
        ),
        chunks=(4, 4),
    )
    group.create_array(
        "scores",
        data=np.asarray([0.9, 0.8, 0.7, 0.6], dtype=np.float32),
        chunks=(4,),
    )
    group.create_array(
        "class_ids",
        data=np.asarray([0, 1, 0, 1], dtype=np.int32),
        chunks=(4,),
    )
    group.create_array(
        "frame_counts",
        data=np.bincount(frames, minlength=5).astype(np.int32),
        chunks=(5,),
    )
    source_manifest = tmp_path / "source-manifest.json"
    source_manifest.write_text(
        json.dumps(
            {
                "purpose": "disposable_storage_benchmark",
                "destination": str(source),
                "canonical": False,
                "registry_registered": False,
                "selector_eligible": False,
            }
        ),
        encoding="utf-8",
    )
    benchmark_root = tmp_path / "benchmarks"
    fixture = (
        benchmark_root
        / "canonical_detection_storage"
        / "fixtures"
        / "tiny-fixture"
    )
    publish_benchmark_fixture(
        fixture_id="tiny-fixture",
        source=source,
        source_manifest_path=source_manifest,
        destination=fixture,
        benchmark_root=benchmark_root,
    )
    workflow = (
        benchmark_root / "canonical_detection_storage" / "workflows" / "tiny"
    )
    matrix = plan_canonical_detection_benchmark_matrix(
        matrix_id="tiny",
        scales=(
            BenchmarkScale.from_mapping(
                "frames_5",
                {
                    "n_frames": 5,
                    "n_instances": 4,
                    "source_width": 640,
                    "source_height": 480,
                },
            ),
        ),
        destination_root=workflow / "candidates",
        repetitions=1,
        candidate_requests=(
            StorageCandidateRequest(
                layout=BenchmarkLayout.REGULAR,
                target_chunk_bytes=MIB,
            ),
        ),
    ).as_manifest()
    matrix_path = workflow / "matrix.json"
    matrix_path.parent.mkdir(parents=True)
    matrix_path.write_text(json.dumps(matrix), encoding="utf-8")
    palette_repo = Path(__file__).resolve().parents[3]
    monkeypatch.setattr(
        block,
        "_verify_repo",
        lambda repo, expected_commit: {
            "repo": str(repo),
            "commit": expected_commit,
            "clean": True,
            "fisheye_import": "test-worktree",
        },
    )
    report_path = workflow / "reports" / "blocks" / "frames_5_repetition_000.json"

    try:
        report = block.run_benchmark_block(
            matrix_path=matrix_path,
            fixture_root=fixture,
            workflow_root=workflow,
            block_report=report_path,
            scale_id="frames_5",
            repetition_index=0,
            recording_identity="recording-a",
            palette_repo=palette_repo,
            expected_commit="test-commit",
            scratch_base=tmp_path / "scratch",
            allow_local=True,
        )
        aggregate = finalize_benchmark(
            matrix_path=matrix_path,
            workflow_root=workflow,
            output=workflow / "aggregate.json",
        )

        assert report["status"] == "complete"
        assert report["fixture_unchanged"] is True
        assert len(report["candidates"]) == 1
        assert report["candidates"][0]["prfs_reads"]["all_exact"] is True
        assert not (
            tmp_path
            / "scratch"
            / "canonical_detection_storage"
            / "tiny"
            / "frames_5_repetition_000"
        ).exists()
        assert aggregate["summary"]["published_candidate_count"] == 1
    finally:
        for record in locals().get("report", {}).get("candidates", []):
            published = Path(str(record["published_candidate"]))
            if published.exists():
                _thaw(published)
        if fixture.exists():
            _thaw(fixture)
