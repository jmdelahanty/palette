"""Inspect or publish a selector-ineligible keypoint coordinate successor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from fisheye.shared.json_safety import write_json_atomic
from fisheye.shared.zarr.keypoint_coordinate_successor import (
    KEYPOINT_COORDINATE_SUCCESSOR_PUBLICATION_SCHEMA_ID,
    inspect_keypoint_coordinate_successor_source,
    publish_keypoint_coordinate_successor,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-zarr", type=Path, required=True)
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--successor-run", required=True)
    parser.add_argument("--keypoint-model-path", type=Path, required=True)
    parser.add_argument("--result-json", type=Path, required=True)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Publish after validation; omission performs a read-only inspection.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.apply:
            result = publish_keypoint_coordinate_successor(
                analysis_zarr=args.analysis_zarr,
                source_run_id=args.source_run,
                successor_run_id=args.successor_run,
                keypoint_model_path=args.keypoint_model_path,
            )
        else:
            result = inspect_keypoint_coordinate_successor_source(
                analysis_zarr=args.analysis_zarr,
                source_run_id=args.source_run,
                successor_run_id=args.successor_run,
                keypoint_model_path=args.keypoint_model_path,
            )
            result = {
                **result,
                "mode": "dry_run",
                "zarr_writes": False,
                "next_action": "rerun_with_apply_after_review",
            }
        write_json_atomic(args.result_json.expanduser().resolve(), result)
    except Exception as exc:
        result = {
            "schema_id": KEYPOINT_COORDINATE_SUCCESSOR_PUBLICATION_SCHEMA_ID,
            "schema_version": 1,
            "status": "failed",
            "mode": "apply" if args.apply else "dry_run",
            "analysis_zarr": str(args.analysis_zarr),
            "source_run_id": args.source_run,
            "successor_run_id": args.successor_run,
            "error": f"{type(exc).__name__}: {exc}",
        }
        write_json_atomic(args.result_json.expanduser().resolve(), result)
        print(json.dumps(result, sort_keys=True))
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
