#!/usr/bin/env python3
"""Run Palette's file-isolated CI pytest shards concurrently on one machine."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Sequence

if __package__:
    from scripts.ci_pytest_junit_summary import summarize_junit_reports
else:
    from ci_pytest_junit_summary import summarize_junit_reports


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ShardResult:
    shard_index: int
    returncode: int
    duration_seconds: float
    log_path: str
    junit_path: str
    duration_summary_path: str | None


def build_shard_command(
    *,
    shard_index: int,
    shard_count: int,
    shard_root: Path,
) -> list[str]:
    """Return the local command corresponding to one hosted CI shard."""

    return [
        sys.executable,
        str(REPOSITORY_ROOT / "scripts/ci_pytest_shard.py"),
        "--shard-index",
        str(shard_index),
        "--shard-count",
        str(shard_count),
        "--",
        "-q",
        "-m",
        "not gpu",
        "--durations=25",
        "--durations-min=1.0",
        "-o",
        "junit_family=legacy",
        "-o",
        "junit_duration_report=total",
        f"--junitxml={shard_root / 'pytest.xml'}",
        f"--basetemp={shard_root / 'pytest-tmp'}",
        "-o",
        f"cache_dir={shard_root / 'pytest-cache'}",
    ]


def _run_shard(
    *,
    shard_index: int,
    shard_count: int,
    output_root: Path,
    fixture_cache_root: Path,
) -> ShardResult:
    shard_root = output_root / f"shard-{shard_index:02d}"
    shard_root.mkdir(parents=True, exist_ok=True)
    log_path = shard_root / "pytest.log"
    junit_path = shard_root / "pytest.xml"
    summary_path = shard_root / "durations.json"
    command = build_shard_command(
        shard_index=shard_index,
        shard_count=shard_count,
        shard_root=shard_root,
    )
    environment = os.environ.copy()
    environment["MPLBACKEND"] = "Agg"
    environment["PALETTE_PYTHON"] = sys.executable
    environment["PYTHONUNBUFFERED"] = "1"
    environment["PALETTE_TEST_FIXTURE_CACHE_DIR"] = str(fixture_cache_root)
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            command,
            cwd=REPOSITORY_ROOT,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
            text=True,
        )
    elapsed = time.monotonic() - started
    published_summary: str | None = None
    if junit_path.is_file():
        payload = summarize_junit_reports([junit_path], shard_index=shard_index)
        summary_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        published_summary = str(summary_path)
    return ShardResult(
        shard_index=shard_index,
        returncode=completed.returncode,
        duration_seconds=round(elapsed, 3),
        log_path=str(log_path),
        junit_path=str(junit_path),
        duration_summary_path=published_summary,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-count", type=int, default=16)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--fixture-cache-root",
        type=Path,
        help="Shared immutable fixture cache (defaults inside output-root).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.shard_count <= 0:
        raise SystemExit("--shard-count must be positive")
    if args.jobs <= 0:
        raise SystemExit("--jobs must be positive")
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=False)
    fixture_cache_root = (
        args.fixture_cache_root.expanduser().resolve()
        if args.fixture_cache_root is not None
        else output_root / "shared-fixture-cache"
    )
    fixture_cache_root.mkdir(parents=True, exist_ok=True)

    print(
        f"Running {args.shard_count} CI shards with {min(args.jobs, args.shard_count)} "
        f"concurrent workers; results: {output_root}",
        flush=True,
    )
    results: list[ShardResult] = []
    with ThreadPoolExecutor(max_workers=min(args.jobs, args.shard_count)) as executor:
        futures = {
            executor.submit(
                _run_shard,
                shard_index=index,
                shard_count=args.shard_count,
                output_root=output_root,
                fixture_cache_root=fixture_cache_root,
            ): index
            for index in range(args.shard_count)
        }
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(
                f"shard {result.shard_index}: returncode={result.returncode} "
                f"duration={result.duration_seconds:.1f}s log={result.log_path}",
                flush=True,
            )

    ordered = sorted(results, key=lambda result: result.shard_index)
    campaign = {
        "schema_id": "palette.local_ci_pytest_shards",
        "schema_version": 1,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository_root": str(REPOSITORY_ROOT),
        "shard_count": args.shard_count,
        "concurrency": min(args.jobs, args.shard_count),
        "fixture_cache_root": str(fixture_cache_root),
        "succeeded": all(result.returncode == 0 for result in ordered),
        "results": [asdict(result) for result in ordered],
    }
    (output_root / "campaign.json").write_text(
        json.dumps(campaign, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0 if campaign["succeeded"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
