"""Publish one selector-ineligible immutable crop-geometry candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from fisheye.analysis_workflows.materializers.registered_detection_gate import (
    validate_registered_detection_gate_consumption,
)
from fisheye.shared.crop_defaults import DEFAULT_ZEBRAFISH_CROP_SIZE_PX
from fisheye.shared.json_safety import write_json_atomic
from fisheye.shared.zarr.crop_schema import (
    CropGeometryPolicy,
    CropPaddingMode,
    CropSizeMode,
)
from fisheye.shared.zarr.crop_snapshot_publication import (
    CROP_SNAPSHOT_PUBLICATION_SCHEMA_ID,
    publish_crop_geometry_production_candidate,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-zarr", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--purpose", required=True)
    parser.add_argument(
        "--roi-width",
        type=int,
        default=DEFAULT_ZEBRAFISH_CROP_SIZE_PX,
    )
    parser.add_argument(
        "--roi-height",
        type=int,
        default=DEFAULT_ZEBRAFISH_CROP_SIZE_PX,
    )
    parser.add_argument(
        "--padding-mode",
        choices=tuple(mode.value for mode in CropPaddingMode),
        default=CropPaddingMode.ZERO_OUTSIDE_SOURCE_FRAME.value,
    )
    parser.add_argument("--camera-id", required=True)
    parser.add_argument("--source-refined-run")
    parser.add_argument(
        "--registered-gate-requirement",
        choices=("off", "if_available", "required"),
        default="off",
    )
    parser.add_argument("--registered-gate-run")
    parser.add_argument(
        "--geometry-origin-provider-run",
        help=(
            "Exact signed hybrid crop run whose verified per-row integer origins "
            "become the strict crop geometry."
        ),
    )
    parser.add_argument("--scratch-root", type=Path, required=True)
    parser.add_argument(
        "--copy-backend",
        choices=("python", "rsync"),
        default="python",
    )
    parser.add_argument("--keep-scratch", action="store_true")
    parser.add_argument("--result-json", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        policy = CropGeometryPolicy(
            purpose=args.purpose,
            size_mode=CropSizeMode.FIXED_PER_RUN,
            fixed_size_wh=(int(args.roi_width), int(args.roi_height)),
            padding_mode=CropPaddingMode(args.padding_mode),
        )
        result = publish_crop_geometry_production_candidate(
            analysis_zarr=args.analysis_zarr,
            run_id=args.run_id,
            policy=policy,
            expected_camera_identity=args.camera_id,
            scratch_root=args.scratch_root,
            copy_backend=args.copy_backend,
            keep_scratch=bool(args.keep_scratch),
            source_refined_run_id=args.source_refined_run,
            registered_gate_requirement=args.registered_gate_requirement,
            registered_gate_run=args.registered_gate_run,
            registered_gate_validator=validate_registered_detection_gate_consumption,
            geometry_origin_provider_run_id=args.geometry_origin_provider_run,
        )
    except Exception as exc:
        result = {
            "schema_id": CROP_SNAPSHOT_PUBLICATION_SCHEMA_ID,
            "status": "failed",
            "analysis_zarr": str(args.analysis_zarr),
            "run_id": args.run_id,
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
