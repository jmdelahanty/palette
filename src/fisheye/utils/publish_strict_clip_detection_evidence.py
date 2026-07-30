#!/usr/bin/env python3
"""Publish one selector-ineligible strict clip detection evidence pair."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from fisheye.shared.zarr.clipped_detection_evidence import (
    publish_strict_clip_detection_evidence,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-zarr", required=True, type=Path)
    parser.add_argument("--source-detect-group", required=True)
    parser.add_argument("--source-refined-group", required=True)
    parser.add_argument("--recording-canonical-archive", required=True, type=Path)
    parser.add_argument("--recording-canonical-run", required=True)
    parser.add_argument("--recording-identity", required=True)
    parser.add_argument("--clip-id", required=True)
    parser.add_argument("--clip-index", required=True, type=int)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--canonical-run", required=True)
    parser.add_argument("--refined-run", required=True)
    parser.add_argument(
        "--without-coordinate-catalog",
        action="store_true",
        help="Compatibility-only escape hatch; new evidence writes the catalog.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    publication = publish_strict_clip_detection_evidence(
        analysis_zarr=args.analysis_zarr,
        source_detect_group_path=args.source_detect_group,
        source_refined_group_path=args.source_refined_group,
        recording_canonical_archive=args.recording_canonical_archive,
        recording_canonical_run_id=args.recording_canonical_run,
        recording_identity=args.recording_identity,
        clip_id=args.clip_id,
        clip_index=args.clip_index,
        output_root=args.output_root,
        canonical_run_id=args.canonical_run,
        refined_run_id=args.refined_run,
        coordinate_catalog=not args.without_coordinate_catalog,
    )
    print(json.dumps(dict(publication.receipt), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
