"""Inspect or publish raw/refined subject-mask coordinate successors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from fisheye.shared.json_safety import write_json_atomic
from fisheye.shared.zarr.subject_mask_coordinate_successor import (
    SUBJECT_MASK_COORDINATE_SUCCESSOR_PUBLICATION_SCHEMA_ID,
    inspect_subject_mask_coordinate_successor_source,
    publish_subject_mask_coordinate_successors,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-zarr", type=Path, required=True)
    parser.add_argument("--source-raw-run", required=True)
    parser.add_argument("--source-refined-run", required=True)
    parser.add_argument("--source-bundle-run", required=True)
    parser.add_argument("--refined-evidence-run-path", required=True)
    parser.add_argument("--raw-successor-run", required=True)
    parser.add_argument("--refined-successor-run", required=True)
    parser.add_argument("--subject-mask-model-path", type=Path, required=True)
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
        "source_raw_run_id": args.source_raw_run,
        "source_refined_run_id": args.source_refined_run,
        "source_bundle_run_id": args.source_bundle_run,
        "refined_evidence_run_path": args.refined_evidence_run_path,
        "raw_successor_run_id": args.raw_successor_run,
        "refined_successor_run_id": args.refined_successor_run,
        "subject_mask_model_path": args.subject_mask_model_path,
    }
    try:
        if args.apply:
            result = publish_subject_mask_coordinate_successors(**kwargs)
        else:
            result = inspect_subject_mask_coordinate_successor_source(**kwargs)
            result = {
                **result,
                "mode": "dry_run",
                "zarr_writes": False,
                "next_action": "rerun_with_apply_after_review",
            }
        write_json_atomic(args.result_json.expanduser().resolve(), result)
    except Exception as exc:
        result = {
            "schema_id": SUBJECT_MASK_COORDINATE_SUCCESSOR_PUBLICATION_SCHEMA_ID,
            "schema_version": 1,
            "status": "failed",
            "mode": "apply" if args.apply else "dry_run",
            "analysis_zarr": str(args.analysis_zarr),
            "source_raw_run_id": args.source_raw_run,
            "source_refined_run_id": args.source_refined_run,
            "error": f"{type(exc).__name__}: {exc}",
        }
        write_json_atomic(args.result_json.expanduser().resolve(), result)
        print(json.dumps(result, sort_keys=True))
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
