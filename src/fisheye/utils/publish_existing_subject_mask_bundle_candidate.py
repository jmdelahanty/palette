"""Import an existing qualified subject-mask bundle without activation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from fisheye.shared.json_safety import write_json_atomic
from fisheye.shared.zarr.subject_mask_bundle_publication import (
    SUBJECT_MASK_BUNDLE_PUBLICATION_SCHEMA_ID,
    publish_subject_mask_bundle_candidate,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-zarr", required=True, type=Path)
    parser.add_argument("--recording-identity", required=True)
    parser.add_argument("--source-analysis-zarr", required=True, type=Path)
    parser.add_argument("--raw-run", required=True)
    parser.add_argument("--refined-run", required=True)
    parser.add_argument("--quality-run", required=True)
    parser.add_argument("--cache-run", required=True)
    parser.add_argument("--bundle-id", required=True)
    parser.add_argument("--copy-backend", choices=("python", "rsync"), default="rsync")
    parser.add_argument("--result-json", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = publish_subject_mask_bundle_candidate(
            analysis_zarr=args.analysis_zarr,
            recording_identity=args.recording_identity,
            raw_snapshot_root=args.source_analysis_zarr,
            raw_run_id=args.raw_run,
            refined_snapshot_root=args.source_analysis_zarr,
            refined_run_id=args.refined_run,
            quality_snapshot_root=args.source_analysis_zarr,
            quality_run_id=args.quality_run,
            cache_snapshot_root=args.source_analysis_zarr,
            cache_run_id=args.cache_run,
            bundle_id=args.bundle_id,
            copy_backend=args.copy_backend,
        )
    except Exception as exc:
        result = {
            "schema_id": SUBJECT_MASK_BUNDLE_PUBLICATION_SCHEMA_ID,
            "status": "failed",
            "analysis_zarr": str(args.analysis_zarr),
            "source_analysis_zarr": str(args.source_analysis_zarr),
            "bundle_id": args.bundle_id,
            "error": f"{type(exc).__name__}: {exc}",
        }
        write_json_atomic(args.result_json.expanduser().resolve(), result)
        print(json.dumps(result, sort_keys=True))
        return 1
    write_json_atomic(args.result_json.expanduser().resolve(), result)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
