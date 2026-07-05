#!/usr/bin/env python3
"""Fail when guarded large files grow beyond their ratcheted line-count budget."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


DEFAULT_THRESHOLD = 200
DEFAULT_BASELINE = Path(__file__).with_name("file_size_ratchet_baseline.json")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _count_lines(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for _ in handle)


def _load_baseline(path: Path) -> dict[str, int]:
    with path.open("rb") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Baseline must be a JSON object: {path}")
    baseline: dict[str, int] = {}
    for key, value in data.items():
        if not isinstance(key, str) or not isinstance(value, int):
            raise ValueError(f"Baseline entries must be path -> integer line count: {path}")
        baseline[key] = value
    return baseline


def _write_baseline(path: Path, baseline: dict[str, int]) -> None:
    payload = json.dumps(dict(sorted(baseline.items())), indent=2)
    path.write_text(f"{payload}\n", encoding="utf-8")


def check_file_size_ratchet(
    *,
    baseline_path: Path = DEFAULT_BASELINE,
    threshold: int = DEFAULT_THRESHOLD,
    update_on_shrink: bool = True,
) -> int:
    repo_root = _repo_root()
    baseline_path = baseline_path if baseline_path.is_absolute() else repo_root / baseline_path
    baseline = _load_baseline(baseline_path)
    failures: list[str] = []
    tightened: list[tuple[str, int, int]] = []

    for relative_path, baseline_lines in sorted(baseline.items()):
        target = repo_root / relative_path
        if not target.exists():
            failures.append(f"{relative_path}: file is missing")
            continue

        current_lines = _count_lines(target)
        allowed_lines = baseline_lines + int(threshold)
        if current_lines > allowed_lines:
            failures.append(
                f"{relative_path}: {current_lines} lines exceeds ratchet budget "
                f"{allowed_lines} (baseline {baseline_lines} + {threshold})"
            )
        elif current_lines < baseline_lines:
            tightened.append((relative_path, baseline_lines, current_lines))
            if update_on_shrink:
                baseline[relative_path] = current_lines

    if tightened and update_on_shrink:
        _write_baseline(baseline_path, baseline)
        print("Updated file-size ratchet baseline for shrinkage:")
        for relative_path, old_lines, new_lines in tightened:
            print(f"  {relative_path}: {old_lines} -> {new_lines}")
    elif tightened:
        print("Files are below their ratcheted baselines:")
        for relative_path, old_lines, new_lines in tightened:
            print(f"  {relative_path}: {old_lines} -> {new_lines}")

    if failures:
        print("File-size ratchet failed:", file=sys.stderr)
        for failure in failures:
            print(f"  {failure}", file=sys.stderr)
        return 1

    print(f"File-size ratchet passed for {len(baseline)} files.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline",
        type=Path,
        default=DEFAULT_BASELINE,
        help="JSON baseline path, relative to repo root unless absolute.",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=DEFAULT_THRESHOLD,
        help="Allowed line growth above baseline before failing.",
    )
    parser.add_argument(
        "--no-update-on-shrink",
        action="store_true",
        help="Report shrinkage without tightening the baseline file.",
    )
    args = parser.parse_args(argv)
    return check_file_size_ratchet(
        baseline_path=args.baseline,
        threshold=args.threshold,
        update_on_shrink=not args.no_update_on_shrink,
    )


if __name__ == "__main__":
    raise SystemExit(main())
