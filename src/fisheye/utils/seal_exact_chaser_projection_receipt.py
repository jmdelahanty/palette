"""Seal one closed receipt composition for exact-chaser interactive readers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from fisheye.analysis_workflows.exact_chaser_projection_receipt import (
    ensure_exact_chaser_projection_receipt,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-zarr", type=Path, required=True)
    parser.add_argument("--palette-commit", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--expected-recording-id")
    for key in (
        "semantic-selection",
        "keypoint-radial",
        "detection-radial",
        "controller",
        "bout",
        "escape",
        "spatial-occupancy",
    ):
        parser.add_argument(f"--{key}-receipt", type=Path, required=True)
    parser.add_argument(
        "--gaze-receipt",
        type=Path,
        help=(
            "Optional exact gaze successor receipt. Supplying it seals projection "
            "receipt schema v2; omitting it preserves the closed v1 roster."
        ),
    )
    parser.add_argument("--keypoint-relative-frame-receipt", type=Path, required=True)
    parser.add_argument("--detection-relative-frame-receipt", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    exact = {
        key: getattr(args, f"{key}_receipt")
        for key in (
            "semantic_selection",
            "keypoint_radial",
            "detection_radial",
            "controller",
            "bout",
            "escape",
            "spatial_occupancy",
        )
    }
    if args.gaze_receipt is not None:
        exact["gaze"] = args.gaze_receipt
    relative = {
        "keypoint": args.keypoint_relative_frame_receipt,
        "detection": args.detection_relative_frame_receipt,
    }
    receipt = ensure_exact_chaser_projection_receipt(
        args.analysis_zarr,
        exact_child_receipts=exact,
        relative_frame_receipts=relative,
        palette_commit=args.palette_commit,
        output_json=args.output_json,
        expected_recording_id=args.expected_recording_id,
    )
    print(json.dumps(receipt, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
