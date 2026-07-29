#!/usr/bin/env python3
"""Run one fresh-process read suite against a detection benchmark candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from fisheye.cluster.lsf.bundle import write_json_snapshot
from fisheye.shared.zarr.canonical_detection_benchmark import (
    load_canonical_detection_benchmark_input,
)
from fisheye.shared.zarr.detection_benchmark_access import (
    DetectionReadWorkloadConfig,
    require_detection_consumer_workloads,
)
from fisheye.shared.zarr.detection_benchmark_planning import (
    collect_access_chunk_bytes_options,
    parse_access_chunk_bytes_option,
    plan_detection_benchmark_candidate,
)
from fisheye.shared.zarr.detection_benchmark_reads import (
    benchmark_detection_candidate_reads,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("canonical_staging", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--storage-tier", required=True)
    parser.add_argument("--chunk-bytes", required=True, type=int)
    parser.add_argument("--shard-bytes", type=int)
    parser.add_argument(
        "--access-chunk-bytes",
        action="append",
        default=[],
        type=parse_access_chunk_bytes_option,
        metavar="ACCESS:BYTES",
    )
    parser.add_argument("--layout", choices=("regular", "sharded"), required=True)
    parser.add_argument("--read-seed", type=int, default=20_260_724)
    args = parser.parse_args(argv)

    report_path = args.report.expanduser().resolve()
    if report_path.exists():
        raise FileExistsError(f"Read benchmark report already exists: {report_path}")
    benchmark_input = load_canonical_detection_benchmark_input(
        args.canonical_staging
    )
    try:
        access_chunk_bytes = collect_access_chunk_bytes_options(
            args.access_chunk_bytes
        )
    except ValueError as exc:
        parser.error(str(exc))
    plans = plan_detection_benchmark_candidate(
        benchmark_input.dimensions,
        target_chunk_bytes=int(args.chunk_bytes),
        target_shard_bytes=args.shard_bytes,
        layout=args.layout,
        target_chunk_bytes_by_access=access_chunk_bytes,
    )
    report = benchmark_detection_candidate_reads(
        benchmark_input,
        candidate=args.candidate,
        plans=plans,
        storage_tier=args.storage_tier,
        workload_config=DetectionReadWorkloadConfig(seed=int(args.read_seed)),
    )
    require_detection_consumer_workloads(report["consumer_workloads"])
    write_json_snapshot(report_path, report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "candidate": report["candidate"],
                "storage_tier": report["storage_tier"],
                "report": str(report_path),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
