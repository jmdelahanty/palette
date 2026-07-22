#!/usr/bin/env python3
"""Run one deterministic, file-isolated pytest shard.

Files are assigned by largest-processing-time-first using source bytes as a
stable cost proxy.  A test file is never split between processes, which keeps
module-scoped immutable fixtures reusable and prevents concurrent tests from
mutating the same temporary Zarr or SQLite fixture.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


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
    byte_loads = [0] * shard_count
    ordered = sorted(
        test_files,
        key=lambda path: (-path.stat().st_size, path.as_posix()),
    )
    for path in ordered:
        shard_index = min(
            range(shard_count),
            key=lambda index: (byte_loads[index], index),
        )
        buckets[shard_index].append(path)
        byte_loads[shard_index] += path.stat().st_size
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
