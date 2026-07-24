#!/usr/bin/env python3
"""Plan or run one disposable canonical detection Zarr storage candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from fisheye.shared.zarr import canonical_detection_benchmark as benchmark_kernel
from fisheye.shared.zarr.canonical_detection_benchmark import (
    CanonicalDetectionBenchmarkInput,
    build_canonical_detection_benchmark_input,
    load_detection_benchmark_input,
    write_detection_benchmark_candidate,
)
from fisheye.shared.zarr.detection_storage import plan_canonical_detection_storage
from fisheye.shared.zarr.storage_profiles import (
    KIB,
    MIB,
    make_benchmark_storage_profile,
)


BENCHMARK_OUTPUT_ROOT = benchmark_kernel.BENCHMARK_OUTPUT_ROOT


def _require_safe_destination(destination: Path) -> Path:
    """Compatibility wrapper for callers of the original diagnostic."""

    return benchmark_kernel.require_safe_benchmark_destination(
        destination,
        benchmark_root=BENCHMARK_OUTPUT_ROOT,
    )


def _candidate_profile(args: argparse.Namespace):
    return make_benchmark_storage_profile(
        target_chunk_bytes=int(args.chunk_kib) * KIB,
        target_shard_bytes=int(args.shard_mib) * MIB,
        shard_immutable=args.layout == "sharded",
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_group", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--recording-identity", required=True)
    parser.add_argument("--frame-limit", type=int)
    parser.add_argument("--chunk-kib", type=int, default=1024)
    parser.add_argument("--shard-mib", type=int, default=32)
    parser.add_argument("--layout", choices=("regular", "sharded"), default="sharded")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args(argv)

    benchmark_input = load_detection_benchmark_input(
        args.source_group,
        recording_identity=args.recording_identity,
        frame_limit=args.frame_limit,
    )
    profile = _candidate_profile(args)
    plans = plan_canonical_detection_storage(
        benchmark_input.dimensions,
        profile=profile,
    )
    if not args.apply:
        print(
            json.dumps(
                {
                    "status": "planned",
                    "source": benchmark_input.as_manifest(),
                    "storage_plan": plans.as_manifest(),
                    "destination": str(args.destination.expanduser().resolve()),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    report = write_detection_benchmark_candidate(
        benchmark_input,
        destination=args.destination,
        plans=plans,
    )
    report_path = args.destination.expanduser().resolve().with_suffix(
        ".benchmark.json"
    )
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "destination": report["destination"],
                "report": str(report_path),
                "profile_id": plans.profile.profile_id,
                "dimensions": benchmark_input.dimensions.as_manifest(),
                "timing": report["timing"],
                "physical": report["physical"],
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


__all__ = [
    "BENCHMARK_OUTPUT_ROOT",
    "CanonicalDetectionBenchmarkInput",
    "build_canonical_detection_benchmark_input",
    "load_detection_benchmark_input",
    "main",
    "write_detection_benchmark_candidate",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
