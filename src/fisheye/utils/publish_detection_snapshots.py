"""Publish selector-ineligible canonical/refined detection v1 snapshots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from fisheye.shared.json_safety import write_json_atomic
from fisheye.shared.zarr.detection_snapshot_publication import (
    DETECTION_SNAPSHOT_PUBLICATION_SCHEMA_ID,
    publish_detection_snapshot_pair,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-zarr", type=Path, required=True)
    parser.add_argument("--source-detect-group", required=True)
    parser.add_argument("--source-refined-group", required=True)
    parser.add_argument("--recording-identity", required=True)
    parser.add_argument("--canonical-run", required=True)
    parser.add_argument("--refined-run", required=True)
    parser.add_argument("--scratch-root", type=Path, required=True)
    parser.add_argument("--result-json", type=Path, required=True)
    parser.add_argument(
        "--allow-initialize-missing-source-keys",
        action="store_true",
        help=(
            "Explicit historical compatibility migration; modern sources must "
            "already contain stable instance_key values."
        ),
    )
    parser.add_argument(
        "--allow-manual-score-reset",
        action="store_true",
        help="Explicitly reset legacy manual scores that have no v1 model meaning.",
    )
    parser.add_argument(
        "--copy-backend",
        choices=("python", "rsync"),
        default="python",
    )
    parser.add_argument("--keep-scratch", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = publish_detection_snapshot_pair(
            analysis_zarr=args.analysis_zarr,
            source_detect_group_path=args.source_detect_group,
            source_refined_group_path=args.source_refined_group,
            recording_identity=args.recording_identity,
            canonical_run_id=args.canonical_run,
            refined_run_id=args.refined_run,
            scratch_root=args.scratch_root,
            allow_initialize_missing_source_keys=(
                args.allow_initialize_missing_source_keys
            ),
            allow_manual_score_reset=args.allow_manual_score_reset,
            copy_backend=args.copy_backend,
            keep_scratch=args.keep_scratch,
        )
    except Exception as exc:
        result = {
            "schema_id": DETECTION_SNAPSHOT_PUBLICATION_SCHEMA_ID,
            "status": "failed",
            "analysis_zarr": str(args.analysis_zarr),
            "source_detect_group": args.source_detect_group,
            "source_refined_group": args.source_refined_group,
            "canonical_run": args.canonical_run,
            "refined_run": args.refined_run,
            "error": f"{type(exc).__name__}: {exc}",
        }
        write_json_atomic(args.result_json, result)
        print(json.dumps(result, sort_keys=True))
        return 1
    write_json_atomic(args.result_json, result)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
