"""CLI for the non-authoritative exact long-form tail-trace projection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from fisheye.analytics_exports.tail_trace_samples import export_tail_trace_samples


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Stream explicit tail-kinematics, subject-shape, and track-motion "
            "publications into an immutable manifest-selected long-form "
            "Parquet tail query product."
        )
    )
    parser.add_argument("zarr", type=Path, help="Recording analysis Zarr.")
    parser.add_argument("--tail-kinematics-run", required=True)
    parser.add_argument("--subject-shape-run", required=True)
    parser.add_argument("--track-kinematics-run", required=True)
    parser.add_argument(
        "--track-scope",
        choices=("online", "offline"),
        default="offline",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--export-run-id", required=True)
    parser.add_argument(
        "--scratch-root",
        type=Path,
        required=True,
        help="Node-local scratch root used for bounded multipart construction.",
    )
    parser.add_argument("--source-window-rows", type=int, default=16_384)
    parser.add_argument("--source-rows-per-part", type=int, default=131_072)
    parser.add_argument("--row-group-rows", type=int, default=65_536)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    result = export_tail_trace_samples(
        args.zarr,
        tail_kinematics_run=args.tail_kinematics_run,
        subject_shape_run=args.subject_shape_run,
        track_kinematics_run=args.track_kinematics_run,
        track_scope=args.track_scope,
        output_root=args.output_root,
        export_run_id=args.export_run_id,
        scratch_root=args.scratch_root,
        source_window_rows=args.source_window_rows,
        source_rows_per_part=args.source_rows_per_part,
        row_group_rows=args.row_group_rows,
        overwrite=bool(args.overwrite),
    )
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(f"export_run_id\t{result['export_run_id']}")
        print(f"manifest\t{result['manifest_path']}")
        print(
            "rows\ttail_trace_samples\t"
            f"{result['row_counts_by_table']['tail_trace_samples']}"
        )


if __name__ == "__main__":
    main()
