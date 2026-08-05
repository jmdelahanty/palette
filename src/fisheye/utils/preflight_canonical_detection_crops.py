"""Run a read-only crop-v2 geometry preflight for canonical-v3 detections."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from fisheye.shared.json_safety import write_json_atomic
from fisheye.shared.zarr.canonical_detection_crop_preflight import (
    CANONICAL_DETECTION_CROP_PREFLIGHT_SCHEMA_ID,
    inspect_canonical_detection_crop_preflight,
)
from fisheye.shared.zarr.crop_schema import (
    CropGeometryPolicy,
    CropPaddingMode,
    CropSizeMode,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-zarr", type=Path, required=True)
    parser.add_argument("--detection-run", required=True)
    parser.add_argument("--roi-width", type=int, default=348)
    parser.add_argument("--roi-height", type=int, default=348)
    parser.add_argument(
        "--padding-mode",
        choices=tuple(item.value for item in CropPaddingMode),
        default=CropPaddingMode.ZERO_OUTSIDE_SOURCE_FRAME.value,
    )
    parser.add_argument(
        "--allow-selector-ineligible-candidate",
        action="store_true",
        help="Allow only this read-only preflight to inspect an unselected candidate.",
    )
    parser.add_argument("--result-json", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    policy = CropGeometryPolicy(
        purpose="ordinary_zebrafish_analysis",
        size_mode=CropSizeMode.FIXED_PER_RUN,
        fixed_size_wh=(args.roi_width, args.roi_height),
        padding_mode=CropPaddingMode(args.padding_mode),
    )
    try:
        result = inspect_canonical_detection_crop_preflight(
            analysis_zarr=args.analysis_zarr,
            detection_run_id=args.detection_run,
            policy=policy,
            allow_selector_ineligible_candidate=(
                args.allow_selector_ineligible_candidate
            ),
        )
        write_json_atomic(args.result_json.expanduser().resolve(), result)
    except Exception as exc:
        result = {
            "schema_id": CANONICAL_DETECTION_CROP_PREFLIGHT_SCHEMA_ID,
            "schema_version": 1,
            "status": "failed",
            "mode": "read_only",
            "analysis_zarr": str(args.analysis_zarr),
            "detection_run_id": args.detection_run,
            "error": f"{type(exc).__name__}: {exc}",
            "crop_zarr_writes": False,
        }
        write_json_atomic(args.result_json.expanduser().resolve(), result)
        print(json.dumps(result, sort_keys=True))
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
