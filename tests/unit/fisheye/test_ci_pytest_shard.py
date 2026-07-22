from __future__ import annotations

from pathlib import Path

import pytest

from scripts.ci_pytest_shard import (
    PROOF_HEAVY_TEST_COST_MULTIPLIER,
    assign_test_file_shards,
    discover_test_files,
    estimated_test_file_cost,
)


def _test_file(path: Path, *, size: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * size)
    return path


def test_assign_test_file_shards_is_complete_unique_and_deterministic(
    tmp_path: Path,
) -> None:
    files = [
        _test_file(tmp_path / "tests" / f"test_{index}.py", size=size)
        for index, size in enumerate((100, 80, 60, 40, 20, 10))
    ]

    first = assign_test_file_shards(files, shard_count=3)
    second = assign_test_file_shards(list(reversed(files)), shard_count=3)

    assert first == second
    assigned = [path for shard in first for path in shard]
    assert sorted(assigned) == sorted(files)
    assert len(assigned) == len(set(assigned))


def test_discover_test_files_excludes_non_test_modules(tmp_path: Path) -> None:
    expected = _test_file(tmp_path / "tests" / "unit" / "test_kept.py", size=1)
    _test_file(tmp_path / "tests" / "unit" / "helper.py", size=1)

    assert discover_test_files(tmp_path / "tests") == (expected,)


def test_estimated_cost_weights_proof_heavy_publication_suites(tmp_path: Path) -> None:
    ordinary = _test_file(tmp_path / "test_ordinary.py", size=100)
    proof_heavy = _test_file(
        tmp_path / "test_subject_shape_coordinate_publication.py",
        size=100,
    )

    assert estimated_test_file_cost(ordinary) == 100
    assert estimated_test_file_cost(proof_heavy) == (
        100 * PROOF_HEAVY_TEST_COST_MULTIPLIER
    )


def test_proof_heavy_files_are_distributed_before_ordinary_fill(tmp_path: Path) -> None:
    heavy = [
        _test_file(tmp_path / name, size=100)
        for name in (
            "test_keypoint_coordinate_publication.py",
            "test_subject_mask_coordinate_publication.py",
            "test_track_motion_publication.py",
        )
    ]
    ordinary = [
        _test_file(tmp_path / f"test_ordinary_{index}.py", size=100)
        for index in range(3)
    ]

    shards = assign_test_file_shards([*heavy, *ordinary], shard_count=3)

    assert all(sum(path in heavy for path in shard) == 1 for shard in shards)


def test_assign_test_file_shards_rejects_nonpositive_count() -> None:
    with pytest.raises(ValueError, match="shard_count must be positive"):
        assign_test_file_shards([], shard_count=0)
