#!/usr/bin/env python3
"""Benchmark indexed physical shards for an immutable subject-shape run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from fisheye.diagnostics.benchmark_tail_kinematics_sharding import run_benchmark


DEFAULT_SHARD_ROWS = (
    16_384,
    32_768,
    65_536,
    131_072,
    262_144,
    524_288,
    1_048_576,
)
DEFAULT_READ_ARRAYS = (
    "components/subject_body/centerline_xy",
    "components/subject_body/bspline_sample_xy",
    "components/subject_body/tail_sample_xy",
    "components/subject_body/tail_curvature_px_inv",
    "body_frame/heading_deg",
    "row_index/frame_indices",
)
REPORT_SCHEMA = "palette.subject_shape_sharding_benchmark.v1"
ROW_COUNT_ARRAY = "row_index/frame_indices"


def _summary(report: dict[str, object]) -> str:
    if report["status"] != "complete":
        return (
            f"{report['status']} candidates={report['candidate_shard_rows']} "
            f"output={report['output_root']}"
        )
    rows: list[str] = []
    for variant in report["variants"]:  # type: ignore[union-attr]
        rows.append(
            "shard_rows={requested_shard_rows} files={files} payload={payload} "
            "write_s={seconds:.3f} full_scan_s={scan:.3f}".format(
                requested_shard_rows=variant["requested_shard_rows"],
                files=variant["storage"]["file_count"],
                payload=variant["storage"]["payload_file_count"],
                seconds=variant["write_seconds"],
                scan=variant["read_benchmark"]["patterns"]["full_scan"]["median_seconds"],
            )
        )
    return "complete exact={}\n{}".format(report["all_variants_exact"], "\n".join(rows))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_group", type=Path, help="Completed subject-shape run-group path.")
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--shard-rows", action="append", type=int, dest="shard_rows")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--read-repeats", type=int, default=3)
    parser.add_argument("--read-array", action="append", dest="read_arrays")
    parser.add_argument("--random-rows", type=int, default=32)
    parser.add_argument("--window-rows", type=int, default=1024)
    parser.add_argument("--window-count", type=int, default=8)
    parser.add_argument("--scan-rows", type=int, default=16_384)
    parser.add_argument("--digest-rows", type=int, default=16_384)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--transfer-root", type=Path)
    parser.add_argument("--keep-transfer-copies", action="store_true")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    report = run_benchmark(
        args.source_group,
        output_root=args.output_root,
        shard_rows=args.shard_rows or DEFAULT_SHARD_ROWS,
        workers=int(args.workers),
        read_repeats=int(args.read_repeats),
        read_arrays=args.read_arrays or DEFAULT_READ_ARRAYS,
        random_rows=int(args.random_rows),
        window_rows=int(args.window_rows),
        window_count=int(args.window_count),
        scan_rows=int(args.scan_rows),
        digest_rows=int(args.digest_rows),
        report_path=args.report,
        transfer_root=args.transfer_root,
        remove_transfer_copies=not bool(args.keep_transfer_copies),
        row_count_array=ROW_COUNT_ARRAY,
        source_label="Subject-shape",
        report_schema=REPORT_SCHEMA,
        apply=bool(args.apply),
        overwrite=bool(args.overwrite),
    )
    print(json.dumps(report, indent=2, sort_keys=True) if args.json else _summary(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
