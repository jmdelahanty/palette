"""Run a read-only crop-v2 geometry preflight for canonical-v3 detections."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

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
from fisheye.utils.publish_canonical_detection_successor_batch import (
    validate_plan,
)


COHORT_PREFLIGHT_SCHEMA_ID = (
    "palette.canonical_detection.crop_geometry_cohort_preflight"
)
COHORT_PREFLIGHT_SCHEMA_VERSION = 1


def _strict_json_load(path: Path) -> dict[str, Any]:
    def reject_nonfinite(token: str) -> None:
        raise ValueError(f"Non-finite JSON token is forbidden: {token}")

    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle, parse_constant=reject_nonfinite)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return value


def inspect_cohort_plan(
    plan: Mapping[str, Any],
    *,
    policy: CropGeometryPolicy,
    allow_selector_ineligible_candidate: bool,
) -> dict[str, object]:
    errors = validate_plan(plan)
    if errors:
        raise ValueError("Invalid canonical successor plan: " + "; ".join(errors))
    reports: list[dict[str, object]] = []
    for candidate in plan["candidates"]:
        inspection = candidate["inspection"]
        reports.append(
            inspect_canonical_detection_crop_preflight(
                analysis_zarr=Path(str(candidate["analysis_zarr"])),
                detection_run_id=str(inspection["successor_run_id"]),
                policy=policy,
                allow_selector_ineligible_candidate=(
                    allow_selector_ineligible_candidate
                ),
            )
        )
    padded = [int(item["padding"]["padded_row_count"]) for item in reports]
    maxima = np.asarray(
        [item["padding"]["max_padding_ltrb"] for item in reports],
        dtype=np.int64,
    )
    return {
        "schema_id": COHORT_PREFLIGHT_SCHEMA_ID,
        "schema_version": COHORT_PREFLIGHT_SCHEMA_VERSION,
        "status": "ready",
        "mode": "read_only",
        "plan_digest": plan["plan_digest"],
        "crop_policy": policy.as_manifest(),
        "archive_count": len(reports),
        "affected_archive_count": sum(value > 0 for value in padded),
        "total_instance_count": sum(
            int(item["dimensions"]["n_instances"]) for item in reports
        ),
        "total_padded_row_count": sum(padded),
        "max_padding_ltrb": (
            [int(value) for value in maxima.max(axis=0)]
            if maxima.size
            else [0, 0, 0, 0]
        ),
        "reports": reports,
        "crop_zarr_writes": False,
        "registry_updated": False,
        "selector_updated": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-zarr", type=Path)
    parser.add_argument("--detection-run")
    parser.add_argument(
        "--plan-json",
        type=Path,
        help="Frozen canonical-successor cohort plan; mutually exclusive with one archive.",
    )
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
        if args.plan_json is not None:
            if args.analysis_zarr is not None or args.detection_run is not None:
                raise ValueError(
                    "--plan-json is mutually exclusive with --analysis-zarr and "
                    "--detection-run."
                )
            result = inspect_cohort_plan(
                _strict_json_load(args.plan_json.expanduser().resolve()),
                policy=policy,
                allow_selector_ineligible_candidate=(
                    args.allow_selector_ineligible_candidate
                ),
            )
        else:
            if args.analysis_zarr is None or args.detection_run is None:
                raise ValueError(
                    "Single preflight requires --analysis-zarr and --detection-run."
                )
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
            "analysis_zarr": (
                None if args.analysis_zarr is None else str(args.analysis_zarr)
            ),
            "detection_run_id": args.detection_run,
            "plan_json": None if args.plan_json is None else str(args.plan_json),
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
