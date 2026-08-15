#!/usr/bin/env python3
"""Summarize pytest JUnit reports into deterministic per-file runtimes."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path, PurePosixPath
from typing import Sequence
import xml.etree.ElementTree as ElementTree


SCHEMA_ID = "palette.ci_pytest_file_durations"
SCHEMA_VERSION = 1


def _classname_path(testcase: ElementTree.Element) -> str | None:
    classname = testcase.get("classname", "")
    if classname:
        components = classname.split(".")
        test_module_indices = [
            index
            for index, component in enumerate(components)
            if component.startswith("test_")
        ]
        if test_module_indices:
            return (
                "/".join(components[: test_module_indices[-1] + 1])
                + ".py"
            )
    return None


def _testcase_path(testcase: ElementTree.Element) -> str:
    # Pytest reports an imported function's defining helper in ``file`` but
    # the collector module in ``classname``. Sharding schedules the collector,
    # so prefer its normalized test-module path whenever one is available.
    normalized = _classname_path(testcase)
    if normalized is None:
        raw_file = testcase.get("file")
        if not raw_file:
            raise ValueError("JUnit testcase lacks a schedulable file identity")
        normalized = raw_file.replace("\\", "/")
    path = PurePosixPath(normalized)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"JUnit testcase has unsafe file path: {normalized!r}")
    return path.as_posix()


def summarize_junit_reports(
    junit_paths: Sequence[Path],
    *,
    shard_index: int,
) -> dict[str, object]:
    """Return deterministic aggregate duration evidence for one CI shard."""

    if shard_index < 0:
        raise ValueError("shard_index must be nonnegative")
    if not junit_paths:
        raise ValueError("at least one JUnit report is required")

    durations: defaultdict[str, float] = defaultdict(float)
    case_counts: defaultdict[str, int] = defaultdict(int)
    report_names: list[str] = []
    for junit_path in junit_paths:
        report_names.append(junit_path.name)
        root = ElementTree.parse(junit_path).getroot()
        for testcase in root.iter("testcase"):
            path = _testcase_path(testcase)
            raw_seconds = testcase.get("time", "0")
            try:
                seconds = float(raw_seconds)
            except ValueError as exc:
                raise ValueError(
                    f"JUnit testcase has invalid duration {raw_seconds!r}: {path}"
                ) from exc
            if not math.isfinite(seconds) or seconds < 0:
                raise ValueError(
                    f"JUnit testcase has invalid duration {raw_seconds!r}: {path}"
                )
            durations[path] += seconds
            case_counts[path] += 1

    files = {
        path: {
            "testcase_count": case_counts[path],
            "duration_seconds": round(durations[path], 6),
        }
        for path in sorted(durations)
    }
    return {
        "schema_id": SCHEMA_ID,
        "schema_version": SCHEMA_VERSION,
        "shard_index": shard_index,
        "source_reports": sorted(report_names),
        "testcase_count": sum(case_counts.values()),
        "duration_seconds": round(sum(durations.values()), 6),
        "files": files,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--junitxml", type=Path, action="append", required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    payload = summarize_junit_reports(
        args.junitxml,
        shard_index=args.shard_index,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
