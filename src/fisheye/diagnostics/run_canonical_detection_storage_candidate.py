#!/usr/bin/env python3
"""Plan or execute one candidate from a fixed local canonical detection store."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from fisheye.shared.zarr.canonical_detection_benchmark import (
    load_canonical_detection_benchmark_input,
    write_detection_benchmark_candidate,
)
from fisheye.shared.zarr.detection_benchmark_planning import (
    plan_detection_benchmark_candidate,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("canonical_staging", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--benchmark-root", required=True, type=Path)
    parser.add_argument("--chunk-bytes", required=True, type=int)
    parser.add_argument("--shard-bytes", type=int)
    parser.add_argument("--layout", choices=("regular", "sharded"), required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args(argv)

    benchmark_input = load_canonical_detection_benchmark_input(
        args.canonical_staging
    )
    if args.layout == "sharded" and args.shard_bytes is None:
        parser.error("--shard-bytes is required for sharded candidates")
    plans = plan_detection_benchmark_candidate(
        benchmark_input.dimensions,
        target_chunk_bytes=int(args.chunk_bytes),
        target_shard_bytes=args.shard_bytes,
        layout=args.layout,
    )
    if not args.apply:
        print(
            json.dumps(
                {
                    "status": "planned",
                    "canonical_source": benchmark_input.as_manifest(),
                    "destination": str(args.destination.expanduser().resolve()),
                    "report": str(args.report.expanduser().resolve()),
                    "storage_plan": plans.as_manifest(),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    report_path = args.report.expanduser().resolve()
    if report_path.exists():
        raise FileExistsError(f"Candidate report already exists: {report_path}")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = write_detection_benchmark_candidate(
        benchmark_input,
        destination=args.destination,
        plans=plans,
        benchmark_root=args.benchmark_root,
    )
    with report_path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "status": report["status"],
                "destination": report["destination"],
                "report": str(report_path),
                "all_digests_exact": all(
                    bool(item["exact"])
                    for item in report["digest_validation"].values()
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
