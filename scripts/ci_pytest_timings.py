#!/usr/bin/env python3
"""Build and consume hash-bound pytest per-file duration baselines."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
from typing import Mapping, Sequence


SCHEMA_ID = "palette.ci_pytest_duration_baseline"
SCHEMA_VERSION = 1
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASELINE_PATH = Path(__file__).with_name(
    "ci_pytest_duration_baseline.json"
)

# Thin collector modules schedule unchanged cases from one non-collected helper.
# Include that helper in their effective source identity so timing evidence goes
# stale when the expensive case implementation changes.
RUNTIME_DEPENDENCIES_BY_TEST = {
    path: ("tests/unit/fisheye/refine_online_coordinate_contract_cases.py",)
    for path in (
        "tests/unit/fisheye/test_refine_online_coordinate_completion_validation.py",
        "tests/unit/fisheye/test_refine_online_coordinate_lifecycle_guards.py",
        "tests/unit/fisheye/test_refine_online_coordinate_lifecycle_rollback.py",
        "tests/unit/fisheye/test_refine_online_coordinate_loading.py",
        "tests/unit/fisheye/test_refine_online_coordinate_publication.py",
    )
}


def _sha256_file(path: Path) -> bytes:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.digest()


def runtime_source_sha256(path: Path, *, repository_root: Path) -> str:
    """Hash one collector and any declared runtime-bearing helper modules."""

    repository_root = repository_root.resolve()
    relative = path.resolve().relative_to(repository_root).as_posix()
    sources = (relative, *RUNTIME_DEPENDENCIES_BY_TEST.get(relative, ()))
    digest = hashlib.sha256()
    for source in sources:
        source_path = repository_root / source
        encoded = source.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
        digest.update(_sha256_file(source_path))
    return digest.hexdigest()


def load_duration_baseline(path: Path = DEFAULT_BASELINE_PATH) -> dict[str, object]:
    """Load and minimally validate one checked-in duration baseline."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("schema_id") != SCHEMA_ID
        or payload.get("schema_version") != SCHEMA_VERSION
        or not isinstance(payload.get("files"), dict)
    ):
        raise ValueError(f"Invalid pytest duration baseline schema: {path}")
    for relative, record in payload["files"].items():
        normalized = PurePosixPath(relative)
        if (
            not isinstance(relative, str)
            or normalized.is_absolute()
            or ".." in normalized.parts
            or not isinstance(record, dict)
        ):
            raise ValueError(f"Invalid pytest duration baseline path: {relative!r}")
        duration = record.get("duration_seconds_p75")
        if (
            not isinstance(duration, (int, float))
            or not math.isfinite(float(duration))
            or duration < 0
            or not isinstance(record.get("source_sha256"), str)
        ):
            raise ValueError(f"Invalid pytest duration record: {relative!r}")
    return payload


def measured_duration_seconds(
    path: Path,
    *,
    repository_root: Path,
    baseline: Mapping[str, object],
) -> float | None:
    """Return hash-current p75 runtime, or ``None`` for fallback scheduling."""

    try:
        relative = path.resolve().relative_to(repository_root.resolve()).as_posix()
    except ValueError:
        return None
    files = baseline.get("files")
    if not isinstance(files, Mapping):
        return None
    record = files.get(relative)
    if not isinstance(record, Mapping):
        return None
    if record.get("source_sha256") != runtime_source_sha256(
        path,
        repository_root=repository_root,
    ):
        return None
    duration = float(record["duration_seconds_p75"])
    return max(0.05, duration)


def _percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("Cannot compute a percentile of no values")
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def build_duration_baseline(
    summary_paths: Sequence[Path],
    *,
    repository_root: Path,
    source_ref: str,
    recorded_at_utc: str,
    excluded_paths: Sequence[str] = (),
) -> dict[str, object]:
    """Build a deterministic p50/p75 baseline from shard summary artifacts."""

    if not summary_paths:
        raise ValueError("At least one duration summary is required")
    durations: defaultdict[str, list[float]] = defaultdict(list)
    case_counts: defaultdict[str, list[int]] = defaultdict(list)
    for summary_path in summary_paths:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if (
            summary.get("schema_id") != "palette.ci_pytest_file_durations"
            or summary.get("schema_version") != 1
            or not isinstance(summary.get("files"), dict)
        ):
            raise ValueError(f"Invalid pytest duration summary: {summary_path}")
        for relative, record in summary["files"].items():
            durations[relative].append(float(record["duration_seconds"]))
            case_counts[relative].append(int(record["testcase_count"]))

    repository_root = repository_root.resolve()
    files: dict[str, object] = {}
    excluded = set(excluded_paths)
    for relative in sorted(durations):
        if relative in excluded:
            continue
        path = repository_root / relative
        if not path.is_file():
            raise FileNotFoundError(f"Timed pytest collector is missing: {relative}")
        files[relative] = {
            "source_sha256": runtime_source_sha256(
                path,
                repository_root=repository_root,
            ),
            "source_bytes": path.stat().st_size,
            "testcase_count": max(case_counts[relative]),
            "sample_count": len(durations[relative]),
            "duration_seconds_p50": round(_percentile(durations[relative], 0.5), 6),
            "duration_seconds_p75": round(_percentile(durations[relative], 0.75), 6),
        }
    return {
        "schema_id": SCHEMA_ID,
        "schema_version": SCHEMA_VERSION,
        "selection": {
            "marker_expression": "not gpu",
            "python_major_minor": "3.11",
            "junit_duration_report": "total",
        },
        "generated_from": {
            "recorded_at_utc": recorded_at_utc,
            "source_ref": source_ref,
            "summary_count": len(summary_paths),
            "minimum_file_sample_count": min(len(values) for values in durations.values()),
        },
        "files": files,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_BASELINE_PATH)
    parser.add_argument("--source-ref", required=True)
    parser.add_argument("--recorded-at-utc", required=True)
    parser.add_argument("--exclude", action="append", default=[])
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    payload = build_duration_baseline(
        args.summary,
        repository_root=REPOSITORY_ROOT,
        source_ref=args.source_ref,
        recorded_at_utc=args.recorded_at_utc,
        excluded_paths=args.exclude,
    )
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
