#!/usr/bin/env python3
"""Build a strict clipped refined-detection binding from persisted evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from fisheye.shared.zarr.clipped_binding_builder import (
    build_clipped_refined_detection_binding,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-zarr", required=True, type=Path)
    parser.add_argument("--detection-plan", required=True, type=Path)
    parser.add_argument("--collection-id", required=True)
    parser.add_argument("--recording-frame-index", required=True, type=Path)
    parser.add_argument("--recording-clip-index", required=True, type=Path)
    parser.add_argument(
        "--strict-evidence-receipt",
        required=True,
        action="append",
        type=Path,
    )
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    binding, receipt = build_clipped_refined_detection_binding(
        analysis_zarr=args.analysis_zarr,
        detection_plan_path=args.detection_plan,
        collection_id=args.collection_id,
        recording_frame_index=args.recording_frame_index,
        recording_clip_index=args.recording_clip_index,
        strict_evidence_receipts=args.strict_evidence_receipt,
        output_path=args.output,
    )
    print(
        json.dumps(
            {"binding": binding.as_manifest(), "receipt": dict(receipt)},
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
