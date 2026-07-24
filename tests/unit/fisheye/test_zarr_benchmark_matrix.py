from __future__ import annotations

import json
from pathlib import Path

import pytest

from fisheye.shared.zarr.benchmark_matrix import (
    BenchmarkLayout,
    BenchmarkScale,
    StorageCandidateRequest,
    require_storage_benchmark_matrix_manifest,
)
from fisheye.shared.zarr.detection_benchmark_matrix import (
    initial_detection_candidate_requests,
    plan_canonical_detection_benchmark_matrix,
)


def _scales() -> tuple[BenchmarkScale, ...]:
    return (
        BenchmarkScale.from_mapping(
            "frames_200k",
            {
                "n_frames": 200_000,
                "n_instances": 199_734,
                "source_width": 4512,
                "source_height": 4512,
            },
        ),
        BenchmarkScale.from_mapping(
            "frames_full",
            {
                "n_frames": 1_188_000,
                "n_instances": 1_187_087,
                "source_width": 4512,
                "source_height": 4512,
            },
        ),
    )


def _matrix(destination_root: Path, *, repetitions: int = 5, occupied=()):
    return plan_canonical_detection_benchmark_matrix(
        matrix_id="sleepyfish_detection_v1",
        scales=_scales(),
        destination_root=destination_root,
        repetitions=repetitions,
        seed=20_260_724,
        occupied_destinations=occupied,
    )


def test_detection_candidate_sweep_uses_only_byte_budgets() -> None:
    requests = initial_detection_candidate_requests()

    assert len(requests) == 20
    assert sum(request.layout is BenchmarkLayout.REGULAR for request in requests) == 4
    assert sum(request.layout is BenchmarkLayout.SHARDED for request in requests) == 16
    assert all(
        request.as_manifest()["row_overrides_supported"] is False
        for request in requests
    )
    with pytest.raises(ValueError, match="Regular candidates cannot declare"):
        StorageCandidateRequest(
            layout=BenchmarkLayout.REGULAR,
            target_chunk_bytes=1024,
            target_shard_bytes=8192,
        )


def test_detection_matrix_deduplicates_effective_physical_stage_plans(
    tmp_path: Path,
) -> None:
    matrix = _matrix(tmp_path)
    manifest = matrix.as_manifest()

    assert manifest["summary"] == {
        "requested_candidate_labels": 40,
        "unique_physical_candidates": 20,
        "removed_duplicate_labels": 20,
        "planned_trials": 100,
        "destination_collisions": 0,
        "payload_io_performed": False,
    }
    by_scale = {
        scale.scale_id: [
            candidate
            for candidate in matrix.candidates
            if candidate.scale_id == scale.scale_id
        ]
        for scale in _scales()
    }
    assert len(by_scale["frames_200k"]) == 8
    assert len(by_scale["frames_full"]) == 12
    assert all(
        duplicate.retained_candidate_id
        in {candidate.candidate_id for candidate in matrix.candidates}
        for duplicate in matrix.duplicates
    )
    assert all(
        duplicate.as_manifest()["reason"]
        == "identical_effective_physical_stage_plan"
        for duplicate in matrix.duplicates
    )
    assert json.loads(json.dumps(manifest, allow_nan=False)) == manifest


def test_detection_matrix_is_deterministic_and_balances_layout_positions(
    tmp_path: Path,
) -> None:
    first = _matrix(tmp_path)
    second = _matrix(tmp_path)

    assert first.as_manifest() == second.as_manifest()
    for scale in _scales():
        repetitions = [
            repetition
            for repetition in first.repetitions
            if repetition.scale_id == scale.scale_id
        ]
        first_layouts = {repetition.trials[0].layout for repetition in repetitions}
        last_layouts = {repetition.trials[-1].layout for repetition in repetitions}
        assert first_layouts == set(BenchmarkLayout)
        assert last_layouts == set(BenchmarkLayout)
        expected_ids = {trial.candidate_id for trial in repetitions[0].trials}
        assert all(
            {trial.candidate_id for trial in repetition.trials} == expected_ids
            for repetition in repetitions
        )


def test_detection_matrix_records_exact_destination_collisions(tmp_path: Path) -> None:
    initial = _matrix(tmp_path, repetitions=1)
    occupied = Path(initial.repetitions[0].trials[0].destination)

    matrix = _matrix(tmp_path, repetitions=1, occupied=(occupied,))

    assert matrix.collision_count == 1
    collisions = [
        trial
        for repetition in matrix.repetitions
        for trial in repetition.trials
        if trial.destination_collision
    ]
    assert [Path(trial.destination) for trial in collisions] == [occupied]


def test_matrix_manifest_fingerprint_rejects_serialized_drift(tmp_path: Path) -> None:
    manifest = _matrix(tmp_path, repetitions=1).as_manifest()

    require_storage_benchmark_matrix_manifest(manifest)
    manifest["seed"] = int(manifest["seed"]) + 1

    with pytest.raises(ValueError, match="fingerprint"):
        require_storage_benchmark_matrix_manifest(manifest)
