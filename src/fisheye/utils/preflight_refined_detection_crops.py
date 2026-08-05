"""Read-only crop-v2 geometry preflight for one explicit refined snapshot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

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


def inspect_refined_detection_crop_preflight(
    *,
    analysis_zarr: Path,
    refined_run_id: str,
    policy: CropGeometryPolicy,
    expected_camera_identity: str | None = None,
    max_examples: int = 32,
) -> dict[str, object]:
    """Prepare and validate all crop arrays in memory without publishing them."""

    if type(max_examples) is not int or max_examples < 0:
        raise ValueError("max_examples must be a nonnegative exact integer.")
    archive = analysis_zarr.expanduser().resolve()
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-zarr", type=Path, required=True)
    parser.add_argument("--refined-run", required=True)
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
            "analysis_zarr": str(args.analysis_zarr),
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
