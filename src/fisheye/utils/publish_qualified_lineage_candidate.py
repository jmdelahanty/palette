"""Import an exact refined-detection/crop production-path candidate pair."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from fisheye.shared.json_safety import write_json_atomic
from fisheye.shared.zarr.qualified_lineage_candidate_import import (
    QUALIFIED_LINEAGE_IMPORT_SCHEMA_ID,
    publish_qualified_lineage_candidate,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-zarr", required=True, type=Path)
    parser.add_argument("--refined-archive", required=True, type=Path)
    parser.add_argument("--refined-run", required=True)
    parser.add_argument("--refined-clip-evidence-root", required=True, type=Path)
    parser.add_argument("--crop-archive", required=True, type=Path)
    parser.add_argument("--crop-run", required=True)
    parser.add_argument("--scratch-root", required=True, type=Path)
    parser.add_argument("--copy-backend", choices=("python", "rsync"), default="rsync")
    parser.add_argument("--keep-scratch", action="store_true")
    parser.add_argument("--result-json", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = publish_qualified_lineage_candidate(
            analysis_zarr=args.analysis_zarr,
            refined_archive=args.refined_archive,
            refined_run_id=args.refined_run,
            refined_clip_evidence_root=args.refined_clip_evidence_root,
            crop_archive=args.crop_archive,
            crop_run_id=args.crop_run,
            scratch_root=args.scratch_root,
            copy_backend=args.copy_backend,
            keep_scratch=bool(args.keep_scratch),
        )
    except Exception as exc:
        result = {
            "schema_id": QUALIFIED_LINEAGE_IMPORT_SCHEMA_ID,
            "status": "failed",
            "analysis_zarr": str(args.analysis_zarr),
            "refined_run_id": args.refined_run,
            "crop_run_id": args.crop_run,
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
