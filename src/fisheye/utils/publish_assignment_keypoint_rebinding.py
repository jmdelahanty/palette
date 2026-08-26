"""Inspect or publish one immutable assignment-keypoint rebinding."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from fisheye.shared.json_safety import write_json_atomic
from fisheye.shared.zarr.assignment_keypoint_rebinding import (
    ASSIGNMENT_KEYPOINT_REBINDING_SCHEMA_ID,
    inspect_assignment_keypoint_rebinding,
    publish_assignment_keypoint_rebinding,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-zarr", type=Path, required=True)
    parser.add_argument("--subject-mask-bundle", required=True)
    parser.add_argument("--keypoint-run", required=True)
    parser.add_argument("--rebinding-run", required=True)
    parser.add_argument("--block-rows", type=int, default=131_072)
    parser.add_argument("--result-json", type=Path, required=True)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Publish after validation; omission performs a read-only inspection.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    kwargs = {
        "analysis_zarr": args.analysis_zarr,
        "subject_mask_bundle_id": args.subject_mask_bundle,
        "keypoint_run_id": args.keypoint_run,
        "rebinding_run_id": args.rebinding_run,
        "block_rows": args.block_rows,
    }
    try:
        if args.apply:
            result = publish_assignment_keypoint_rebinding(**kwargs)
        else:
            manifest = inspect_assignment_keypoint_rebinding(**kwargs)
            result = {
                "status": "ready",
                "mode": "dry_run",
                "zarr_writes": False,
                "manifest": manifest,
                "next_action": "rerun_with_apply_after_review",
            }
        write_json_atomic(args.result_json.expanduser().resolve(), result)
    except Exception as exc:
        result = {
            "schema_id": ASSIGNMENT_KEYPOINT_REBINDING_SCHEMA_ID,
            "status": "failed",
            "mode": "apply" if args.apply else "dry_run",
            "analysis_zarr": str(args.analysis_zarr),
            "subject_mask_bundle_id": args.subject_mask_bundle,
            "keypoint_run_id": args.keypoint_run,
            "rebinding_run_id": args.rebinding_run,
            "error": f"{type(exc).__name__}: {exc}",
        }
        write_json_atomic(args.result_json.expanduser().resolve(), result)
        print(json.dumps(result, sort_keys=True))
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
