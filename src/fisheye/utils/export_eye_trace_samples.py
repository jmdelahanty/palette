"""CLI for the non-authoritative exact eye-trace Parquet projection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from fisheye.analytics_exports.eye_trace_samples import export_eye_trace_samples


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Stream one explicit compact-v7 eye-angle frame axis into an "
            "immutable manifest-selected Parquet query product."
        )
    )
    parser.add_argument("zarr", type=Path, help="Recording analysis Zarr.")
    parser.add_argument("--eye-angle-run", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--export-run-id", required=True)
    parser.add_argument(
        "--scratch-root",
        type=Path,
        required=True,
        help="Node-local scratch root used for streaming Parquet construction.",
    )
    parser.add_argument("--row-group-rows", type=int, default=65_536)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    result = export_eye_trace_samples(
        args.zarr,
        eye_angle_run=args.eye_angle_run,
        output_root=args.output_root,
        export_run_id=args.export_run_id,
        scratch_root=args.scratch_root,
        row_group_rows=args.row_group_rows,
        overwrite=bool(args.overwrite),
    )
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(f"export_run_id\t{result['export_run_id']}")
        print(f"manifest\t{result['manifest_path']}")
        print(
            "rows\teye_trace_samples\t"
            f"{result['row_counts_by_table']['eye_trace_samples']}"
        )


if __name__ == "__main__":
    main()
