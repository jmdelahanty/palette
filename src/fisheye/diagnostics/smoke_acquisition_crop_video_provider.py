"""Smoke-test acquisition crop-video ROI reads through CropImageSource."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import zarr

from fisheye.shared.crop_image_source import CropImageSource


SCHEMA_ID = "palette.acquisition_crop_video_provider_smoke.v1"


def run_smoke(
    zarr_path: Path,
    *,
    crop_run: str,
    start: int,
    rows: int,
) -> dict[str, Any]:
    root = zarr.open_group(str(zarr_path), mode="r", use_consolidated=False)
    source = CropImageSource.open(
        root,
        crop_run=crop_run,
        zarr_path=zarr_path,
        roi_cache_policy="never",
    )
    try:
        stop = min(int(start) + int(rows), source.total_rois)
        if start < 0 or start >= source.total_rois or stop <= start:
            raise ValueError(
                f"Invalid smoke slice start={start} rows={rows} total_rois={source.total_rois}"
            )
        batch = source.read_slice(int(start), int(stop))
        crop_video_frame_indices = (
            source.crop_video_frame_indices[start:stop]
            if source.crop_video_frame_indices is not None
            else np.array([], dtype=np.int64)
        )
        return {
            "schema_id": SCHEMA_ID,
            "zarr_path": str(zarr_path),
            "crop_run": source.crop_run_name,
            "row_start": int(start),
            "row_stop": int(stop),
            "row_count": int(stop - start),
            "total_rois": int(source.total_rois),
            "roi_read_mode": source.roi_read_mode,
            "frame_source_kind": source.frame_source_kind,
            "frame_source_path": source.frame_source_path,
            "roi_shape": [int(source.roi_shape[0]), int(source.roi_shape[1])],
            "batch_shape": [int(v) for v in batch.shape],
            "batch_dtype": str(batch.dtype),
            "batch_min": int(batch.min()) if batch.size else None,
            "batch_max": int(batch.max()) if batch.size else None,
            "batch_mean": float(batch.mean()) if batch.size else None,
            "frame_indices": [int(v) for v in source.frame_indices[start:stop]],
            "source_crop_video_frame_indices": [int(v) for v in crop_video_frame_indices],
            "roi_pixel_contract_name": (
                source.roi_pixel_contract.get("name") if source.roi_pixel_contract else None
            ),
        }
    finally:
        source.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zarr_path", type=Path)
    parser.add_argument("--crop-run", required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--rows", type=int, default=4)
    parser.add_argument(
        "--require-acquisition-crop-video",
        action="store_true",
        help="Fail if the resolved provider is not acquisition_crop_video.",
    )
    args = parser.parse_args(argv)

    result = run_smoke(
        args.zarr_path.expanduser().resolve(),
        crop_run=str(args.crop_run),
        start=int(args.start),
        rows=int(args.rows),
    )
    if args.require_acquisition_crop_video and result["roi_read_mode"] != "acquisition_crop_video":
        raise SystemExit(
            "Expected roi_read_mode=acquisition_crop_video, got "
            f"{result['roi_read_mode']!r}"
        )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
