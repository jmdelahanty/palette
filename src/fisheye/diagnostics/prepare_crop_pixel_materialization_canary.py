"""Prepare the node-local crop snapshot and keyed pixel package for a canary."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time
from typing import Any, Sequence

import numpy as np
import zarr

from fisheye.shared.crop_image_source import CropImageSource
from fisheye.shared.crop_pixel_work_package import (
    build_crop_pixel_work_package_from_source,
)
from fisheye.shared.instance_keys import resolve_recording_identity
from fisheye.shared.zarr.crop_consumer import (
    CROP_RUN_REFERENCE_SIGNED_PROFILE,
    build_crop_run_reference,
)
from fisheye.utils.build_analysis_acquisition_crop_run import (
    build_analysis_acquisition_crop_run,
)
from fisheye.utils.import_acquisition_detections_to_detect_run import (
    resolve_source_dimensions,
)


def _copy_minimal_root_identity(
    source: Any,
    target: Any,
    *,
    recording_identity: str,
    width: int,
    height: int,
) -> None:
    target.attrs.update(
        {
            "recording_id": recording_identity,
            "source_video_width": int(width),
            "source_video_height": int(height),
        }
    )
    for name in ("total_frames", "n_frames", "source_video_total_frames"):
        value = source.attrs.get(name)
        if type(value) is int and int(value) > 0:
            target.attrs[name] = int(value)
            break


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    source_archive = args.source_analysis_zarr.expanduser().resolve()
    scratch = args.scratch_root.expanduser().resolve()
    if not scratch.is_dir():
        raise ValueError(f"Prepared scratch directory does not exist: {scratch}")
    local_archive = scratch / "analysis.zarr"
    package_manifest = scratch / "crop_pixels" / "package.json"
    package_manifest.parent.mkdir(parents=True, exist_ok=True)

    source_root = zarr.open_group(
        str(source_archive), mode="r", use_consolidated=False
    )
    recording_identity = resolve_recording_identity(
        source_root.attrs,
        fallback_path=source_archive,
    )
    width, height = resolve_source_dimensions(
        source_root,
        recording_dir=args.recording_dir,
        source_width=args.source_width,
        source_height=args.source_height,
    )
    local_root = zarr.open_group(str(local_archive), mode="w", zarr_format=3)
    _copy_minimal_root_identity(
        source_root,
        local_root,
        recording_identity=recording_identity,
        width=width,
        height=height,
    )

    phases: dict[str, float] = {}
    phase = time.perf_counter()
    crop_result = build_analysis_acquisition_crop_run(
        local_archive,
        recording_dir=args.recording_dir,
        crop_meta_path=args.crop_meta,
        crop_video_path=args.crop_video,
        run_name=args.crop_run,
        source_width=width,
        source_height=height,
        apply=True,
    )
    phases["publish_local_modern_crop"] = float(time.perf_counter() - phase)

    local_root = zarr.open_group(str(local_archive), mode="r", use_consolidated=False)
    crop = local_root[f"crop_runs/{args.crop_run}"]
    reference = build_crop_run_reference(crop, run_id=args.crop_run)
    if reference["profile"] != CROP_RUN_REFERENCE_SIGNED_PROFILE:
        raise ValueError("Canary crop did not publish the signed current-source profile.")
    total_rows = int(crop["instance_key"].shape[0])
    selected_count = min(int(args.row_count), total_rows)
    if selected_count <= 0:
        raise ValueError("Canary crop has no rows.")
    selected_rows = np.arange(selected_count, dtype=np.int64)

    phase = time.perf_counter()
    direct_source = CropImageSource.open(
        local_root,
        crop_run=args.crop_run,
        zarr_path=local_archive,
        roi_cache_policy="never",
    )
    try:
        package = build_crop_pixel_work_package_from_source(
            direct_source,
            target_crop_rows=selected_rows,
            manifest_path=package_manifest,
            archive_path=local_archive,
            batch_rows=int(args.batch_rows),
        )
    finally:
        direct_source.close()
    phases["materialize_pixel_work_package"] = float(time.perf_counter() - phase)

    return {
        "status": "complete",
        "local_archive": str(local_archive),
        "package_manifest": str(package_manifest),
        "recording_identity": recording_identity,
        "source_width": int(width),
        "source_height": int(height),
        "crop_result": asdict(crop_result),
        "crop_run_reference": reference,
        "source_crop_total_rows": total_rows,
        "selected_rows": selected_count,
        "first_crop_row": int(selected_rows[0]),
        "last_crop_row": int(selected_rows[-1]),
        "package": package,
        "timing_seconds": phases,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-analysis-zarr", type=Path, required=True)
    parser.add_argument("--recording-dir", type=Path, required=True)
    parser.add_argument("--crop-meta", type=Path, required=True)
    parser.add_argument("--crop-video", type=Path, required=True)
    parser.add_argument("--scratch-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--crop-run", required=True)
    parser.add_argument("--row-count", type=int, required=True)
    parser.add_argument("--batch-rows", type=int, required=True)
    parser.add_argument("--source-width", type=int)
    parser.add_argument("--source-height", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output = args.output_json.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = prepare(args)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
