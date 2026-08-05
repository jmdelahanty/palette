"""Read-only crop-v2 geometry preflight for approved or explicit refined data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from fisheye.shared.crop_defaults import (
    DEFAULT_ZEBRAFISH_CROP_PURPOSE,
    DEFAULT_ZEBRAFISH_CROP_SIZE_PX,
)
from fisheye.shared.json_safety import json_attr_safe, write_json_atomic
from fisheye.shared.zarr.benchmark_runtime import sha256_array
from fisheye.shared.zarr.crop_pixel_authority import (
    bind_refined_crop_source_pixel_authority,
)
from fisheye.shared.zarr.crop_schema import (
    CropGeometryPolicy,
    CropPaddingMode,
    CropSizeMode,
)
from fisheye.shared.zarr.crop_shadow import (
    prepare_crop_geometry_from_refined_source,
)
from fisheye.shared.zarr.crop_storage import plan_crop_geometry_storage
from fisheye.shared.zarr.refined_detection_crop_source import (
    bind_refined_detection_crop_source,
)


PREFLIGHT_SCHEMA_ID = "palette.refined_detection.crop_v2_preflight"
PREFLIGHT_SCHEMA_VERSION = 1
COHORT_PREFLIGHT_SCHEMA_ID = "palette.refined_detection.crop_v2_cohort_preflight"
COHORT_PREFLIGHT_SCHEMA_VERSION = 1


def inspect_refined_detection_crop_preflight(
    *,
    analysis_zarr: Path,
    refined_run_id: str | None = None,
    policy: CropGeometryPolicy,
    expected_camera_identity: str | None = None,
    max_examples: int = 32,
) -> dict[str, object]:
    """Prepare and validate all crop arrays in memory without publishing them.

    Omitting ``refined_run_id`` uses the approved production authority.  An
    explicit id remains the selector-ineligible candidate/benchmark boundary.
    """

    if type(max_examples) is not int or max_examples < 0:
        raise ValueError("max_examples must be a nonnegative exact integer.")
    archive = analysis_zarr.expanduser().resolve()
    if refined_run_id is None:
        source = bind_refined_detection_crop_source(archive)
    else:
        source = bind_refined_detection_crop_source(
            archive,
            run_id=str(refined_run_id),
            allow_selector_ineligible_benchmark=True,
        )
    pixels = bind_refined_crop_source_pixel_authority(
        source,
        expected_camera_identity=expected_camera_identity,
    )
    pixels.assert_verified()
    prepared = prepare_crop_geometry_from_refined_source(
        source,
        policy=policy,
        pixel_authority=pixels.pixel_authority,
    )
    plans = plan_crop_geometry_storage(prepared.dimensions)

    source_crop = np.asarray(prepared.arrays["source_crop_xywh"], dtype=np.int64)
    left = np.maximum(0, -source_crop[:, 0])
    top = np.maximum(0, -source_crop[:, 1])
    right = np.maximum(
        0,
        source_crop[:, 0]
        + source_crop[:, 2]
        - prepared.dimensions.source_width,
    )
    bottom = np.maximum(
        0,
        source_crop[:, 1]
        + source_crop[:, 3]
        - prepared.dimensions.source_height,
    )
    padding = np.column_stack((left, top, right, bottom)).astype(
        np.int32,
        copy=False,
    )
    padded = np.flatnonzero(np.any(padding != 0, axis=1))
    examples = [
        {
            "row_index": int(index),
            "instance_key": int(prepared.arrays["instance_key"][index]),
            "frame_index": int(prepared.arrays["frame_indices"][index]),
            "source_crop_xywh": source_crop[index].tolist(),
            "padding_ltrb": padding[index].tolist(),
        }
        for index in padded[:max_examples]
    ]
    return json_attr_safe(
        {
            "schema_id": PREFLIGHT_SCHEMA_ID,
            "schema_version": PREFLIGHT_SCHEMA_VERSION,
            "status": "ready",
            "analysis_zarr": str(archive),
            "refined_run_id": source.run_id,
            "refined_manifest_digest": source.manifest["payload_digest"],
            "refined_logical_content_digest": source.logical_content_digest,
            "selection_mode": source.selection_mode,
            "dimensions": prepared.dimensions.as_manifest(),
            "policy": policy.as_manifest(),
            "pixel_authority": {
                "binding_document_digest": pixels.binding_document_digest,
                "source_video_path": str(pixels.source_video_path),
                "authority": pixels.pixel_authority.as_manifest(),
            },
            "padding": {
                "semantics": "explicit_zero_outside_source_frame_ltrb",
                "padded_row_count": int(padded.size),
                "fully_contained_row_count": int(
                    prepared.dimensions.n_instances - padded.size
                ),
                "max_padding_ltrb": (
                    [0, 0, 0, 0]
                    if padding.size == 0
                    else padding.max(axis=0).tolist()
                ),
                "examples": examples,
            },
            "array_content_sha256": {
                path: sha256_array(values)
                for path, values in sorted(prepared.arrays.items())
            },
            "storage_plan": plans.as_manifest(),
            "crop_zarr_writes": False,
            "selector_activation": "none",
            "registry_updated": False,
        }
    )


def inspect_refined_detection_crop_cohort(
    plan: Mapping[str, Any],
    *,
    policy: CropGeometryPolicy,
) -> dict[str, object]:
    """Run the exact no-write crop-v2 preflight for one frozen refined plan."""

    from fisheye.utils.publish_accept_all_refined_detection_batch import (
        validate_plan,
    )

    errors = validate_plan(plan)
    if errors:
        raise ValueError("Refusing invalid refined cohort plan: " + "; ".join(errors))
    reports: list[dict[str, object]] = []
    for candidate in plan["candidates"]:
        inspection = candidate["inspection"]
        report = inspect_refined_detection_crop_preflight(
            analysis_zarr=Path(str(candidate["analysis_zarr"])),
            refined_run_id=str(plan["refined_run_id"]),
            policy=policy,
            max_examples=0,
        )
        reports.append(
            {
                "analysis_zarr": report["analysis_zarr"],
                "recording_identity": inspection["recording_identity"],
                "refined_run_id": report["refined_run_id"],
                "refined_manifest_digest": report["refined_manifest_digest"],
                "refined_logical_content_digest": report[
                    "refined_logical_content_digest"
                ],
                "dimensions": report["dimensions"],
                "pixel_authority_digest": report["pixel_authority"][
                    "binding_document_digest"
                ],
                "padding": {
                    key: value
                    for key, value in report["padding"].items()
                    if key != "examples"
                },
                "array_content_sha256": report["array_content_sha256"],
            }
        )
    max_padding = [0, 0, 0, 0]
    for report in reports:
        observed = report["padding"]["max_padding_ltrb"]
        max_padding = [
            max(int(current), int(candidate))
            for current, candidate in zip(max_padding, observed, strict=True)
        ]
    return json_attr_safe(
        {
            "schema_id": COHORT_PREFLIGHT_SCHEMA_ID,
            "schema_version": COHORT_PREFLIGHT_SCHEMA_VERSION,
            "status": "ready",
            "plan_digest": plan["plan_digest"],
            "canonical_successor_plan_digest": plan[
                "canonical_successor_plan_digest"
            ],
            "refined_run_id": plan["refined_run_id"],
            "policy": policy.as_manifest(),
            "archive_count": len(reports),
            "total_instance_count": sum(
                int(report["dimensions"]["n_instances"]) for report in reports
            ),
            "total_padded_row_count": sum(
                int(report["padding"]["padded_row_count"]) for report in reports
            ),
            "affected_archive_count": sum(
                int(report["padding"]["padded_row_count"]) > 0 for report in reports
            ),
            "max_padding_ltrb": max_padding,
            "archives": reports,
            "crop_zarr_writes": False,
            "selector_activation": "none",
            "registry_updated": False,
        }
    )


def _strict_json_load(path: Path) -> dict[str, Any]:
    def reject_nonfinite(token: str) -> None:
        raise ValueError(f"Non-finite JSON token is forbidden: {token}")

    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle, parse_constant=reject_nonfinite)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--analysis-zarr", type=Path)
    source.add_argument("--plan-json", type=Path)
    parser.add_argument("--refined-run")
    parser.add_argument("--camera-id")
    parser.add_argument("--purpose", default=DEFAULT_ZEBRAFISH_CROP_PURPOSE)
    parser.add_argument("--roi-width", type=int, default=DEFAULT_ZEBRAFISH_CROP_SIZE_PX)
    parser.add_argument("--roi-height", type=int, default=DEFAULT_ZEBRAFISH_CROP_SIZE_PX)
    parser.add_argument("--max-examples", type=int, default=32)
    parser.add_argument("--result-json", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    policy = CropGeometryPolicy(
        purpose=args.purpose,
        size_mode=CropSizeMode.FIXED_PER_RUN,
        fixed_size_wh=(args.roi_width, args.roi_height),
        padding_mode=CropPaddingMode.ZERO_OUTSIDE_SOURCE_FRAME,
    )
    try:
        if args.plan_json is not None:
            if args.camera_id is not None:
                raise ValueError("Cohort preflight resolves each persisted camera id.")
            plan = _strict_json_load(args.plan_json.expanduser().resolve())
            result = inspect_refined_detection_crop_cohort(plan, policy=policy)
        else:
            result = inspect_refined_detection_crop_preflight(
                analysis_zarr=args.analysis_zarr,
                refined_run_id=args.refined_run,
                policy=policy,
                expected_camera_identity=args.camera_id,
                max_examples=args.max_examples,
            )
    except Exception as exc:
        result = {
            "schema_id": PREFLIGHT_SCHEMA_ID,
            "schema_version": PREFLIGHT_SCHEMA_VERSION,
            "status": "failed",
            "analysis_zarr": (
                None if args.analysis_zarr is None else str(args.analysis_zarr)
            ),
            "plan_json": None if args.plan_json is None else str(args.plan_json),
            "refined_run_id": args.refined_run,
            "error": f"{type(exc).__name__}: {exc}",
            "crop_zarr_writes": False,
        }
        write_json_atomic(args.result_json, result)
        print(json.dumps(result, sort_keys=True))
        return 1
    write_json_atomic(args.result_json, result)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
