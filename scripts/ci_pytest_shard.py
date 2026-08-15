#!/usr/bin/env python3
"""Run one deterministic, file-isolated pytest shard.

Files are assigned by largest-estimated-cost-first. Source bytes are the base
proxy, with stable multipliers for proof-heavy publication suites whose runtime
is dominated by repeated Zarr integrity validation rather than source length.
A test file is never split between processes, which keeps module-scoped
immutable fixtures reusable and prevents concurrent tests from mutating the
same temporary Zarr or SQLite fixture.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import pytest

if __package__:
    from scripts.ci_pytest_timings import (
        DEFAULT_BASELINE_PATH,
        load_duration_baseline,
        measured_duration_seconds,
    )
else:
    from ci_pytest_timings import (
        DEFAULT_BASELINE_PATH,
        load_duration_baseline,
        measured_duration_seconds,
    )


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DURATION_BASELINE = (
    load_duration_baseline(DEFAULT_BASELINE_PATH)
    if DEFAULT_BASELINE_PATH.is_file()
    else {"files": {}}
)
COST_UNITS_PER_SECOND = 1_000

# A four-shard workstation run on 2026-07-22 finished in 9, 38, 61, and 51
# minutes despite nearly equal source-byte loads. The first eight-shard run
# improved that to 1, 2, 31, 9, 18, 34, 45, and 31 minutes, exposing four more
# proof-heavy files that source bytes had underweighted. These suites contain
# repeated fail-closed publication proofs and therefore need a larger stable
# scheduling weight than ordinary numerical/unit tests. Keep this list narrow
# and use CI's reported slow-test durations when revising it.
PROOF_HEAVY_TEST_FILE_NAMES = frozenset(
    {
        "test_canonical_coordinate_publication.py",
        "test_chaser_distance_coordinate_publication.py",
        "test_finalize_subject_masks.py",
        "test_keypoint_coordinate_publication.py",
        "test_observation_coordinate_publication.py",
        "test_refined_subject_mask_coordinate_publication.py",
        "test_stimulus_response.py",
        "test_subject_mask_coordinate_publication.py",
        "test_subject_shape_coordinate_publication.py",
        "test_subject_shape_runs.py",
        "test_tail_coordinate_publication.py",
        "test_track_kinematics_coordinate_contract.py",
        "test_track_motion_publication.py",
    }
)
PROOF_HEAVY_TEST_COST_MULTIPLIER = 6

# Hosted CI on 2026-07-22 measured these suites far above what their source
# sizes predict: refine-online took about 20m42s and subject-shape publication
# about 11m.  Keep the overrides narrow and evidence-based so the greedy
# assignment cannot place both long suites on one worker again.
PROOF_HEAVY_TEST_COST_MULTIPLIER_OVERRIDES = {
    "test_subject_shape_coordinate_publication.py": 12,
}

# GitHub Actions run 31840775086 measured the formerly monolithic
# refine-online contract suite at 20m42s. Its unchanged cases are now exposed
# through five thin collection modules so whole-file sharding can schedule
# them independently. Fixed relative-path costs preserve the measured balance;
# wrapper source bytes are intentionally not used as a runtime proxy.
HISTORICAL_TEST_FILE_COST_OVERRIDES = {
    "tests/unit/fisheye/test_refine_online_coordinate_completion_validation.py": 120_000,
    "tests/unit/fisheye/test_refine_online_coordinate_lifecycle_guards.py": 50_814,
    "tests/unit/fisheye/test_refine_online_coordinate_lifecycle_rollback.py": 200_118,
    "tests/unit/fisheye/test_refine_online_coordinate_loading.py": 29_187,
    "tests/unit/fisheye/test_refine_online_coordinate_publication.py": 160_000,
}


def estimated_test_file_cost(path: Path) -> int:
    """Return one deterministic relative runtime estimate for ``path``."""

    try:
        relative_path = path.resolve().relative_to(REPOSITORY_ROOT.resolve()).as_posix()
    except ValueError:
        relative_path = None
    if relative_path in HISTORICAL_TEST_FILE_COST_OVERRIDES:
        return HISTORICAL_TEST_FILE_COST_OVERRIDES[relative_path]
    measured_seconds = measured_duration_seconds(
        path,
        repository_root=REPOSITORY_ROOT,
        baseline=DURATION_BASELINE,
    )
    if measured_seconds is not None:
        return max(1, round(measured_seconds * COST_UNITS_PER_SECOND))
    size = path.stat().st_size
    multiplier = 1
    if path.name in PROOF_HEAVY_TEST_FILE_NAMES:
        multiplier = PROOF_HEAVY_TEST_COST_MULTIPLIER_OVERRIDES.get(
            path.name,
            PROOF_HEAVY_TEST_COST_MULTIPLIER,
        )
    return max(1, size) * multiplier


def discover_test_files(test_root: Path) -> tuple[Path, ...]:
    """Return every pytest file below ``test_root`` in stable path order."""

    return tuple(sorted(test_root.rglob("test_*.py"), key=lambda path: path.as_posix()))


def assign_test_file_shards(
    test_files: Sequence[Path],
    *,
    shard_count: int,
) -> tuple[tuple[Path, ...], ...]:
    """Balance complete files deterministically across ``shard_count`` shards."""

    if shard_count <= 0:
        raise ValueError("shard_count must be positive")
    buckets: list[list[Path]] = [[] for _ in range(shard_count)]
    estimated_loads = [0] * shard_count
    ordered = sorted(
        test_files,
        key=lambda path: (-estimated_test_file_cost(path), path.as_posix()),
    )
    for path in ordered:
        shard_index = min(
            range(shard_count),
            key=lambda index: (estimated_loads[index], index),
        )
        buckets[shard_index].append(path)
        estimated_loads[shard_index] += estimated_test_file_cost(path)
    return tuple(
        tuple(sorted(bucket, key=lambda path: path.as_posix()))
        for bucket in buckets
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="Print selected paths without invoking pytest.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args, pytest_args = _parser().parse_known_args(argv)
    if pytest_args[:1] == ["--"]:
        pytest_args = pytest_args[1:]
    if args.shard_index < 0 or args.shard_index >= args.shard_count:
        raise SystemExit(
            "shard-index must be in the half-open range [0, shard-count)"
        )

    test_files = discover_test_files(REPOSITORY_ROOT / "tests")
    shards = assign_test_file_shards(test_files, shard_count=args.shard_count)
    selected = shards[args.shard_index]
    relative_paths = [path.relative_to(REPOSITORY_ROOT) for path in selected]
    if args.list_only:
        for path in relative_paths:
            print(path.as_posix())
        return 0
    if not relative_paths:
        raise SystemExit(f"pytest shard {args.shard_index} contains no test files")
    return pytest.main([*pytest_args, *(str(path) for path in relative_paths)])


if __name__ == "__main__":
    raise SystemExit(main())
