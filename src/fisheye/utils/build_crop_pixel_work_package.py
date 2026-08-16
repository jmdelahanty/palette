"""Plan or build one keyed subset ROI pixel work package."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import zarr

from fisheye.shared.crop_image_source import CropImageSource
from fisheye.shared.crop_pixel_work_package import (
    build_crop_pixel_work_package_from_source,
)


def _load_rows(args: argparse.Namespace) -> np.ndarray:
    values = [int(value) for value in (args.crop_row or [])]
    if args.crop_rows_json is not None:
        payload = json.loads(args.crop_rows_json.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError("--crop-rows-json must contain one JSON integer list.")
        values.extend(int(value) for value in payload)
    if args.crop_rows_npy is not None:
        values.extend(
            np.asarray(
                np.load(args.crop_rows_npy, allow_pickle=False), dtype=np.int64
            )
            .reshape(-1)
            .tolist()
        )
    rows = np.asarray(values, dtype=np.int64)
    if rows.size == 0:
        raise ValueError(
            "Provide at least one --crop-row, --crop-rows-json, or --crop-rows-npy."
        )
    return rows


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plan or persist only selected logical crop rows for shared keypoint/"
            "subject-mask delta inference. Dry-run is the default."
        )
    )
    parser.add_argument("zarr_path", type=Path)
    parser.add_argument("--crop-run", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--crop-row", type=int, action="append")
    parser.add_argument(
        "--crop-rows-json",
        type=Path,
        help="JSON file containing source crop row integers.",
    )
    parser.add_argument(
        "--crop-rows-npy",
        type=Path,
        help="NumPy file containing source crop row integers.",
    )
    parser.add_argument("--batch-rows", type=int, default=256)
    parser.add_argument(
        "--roi-cache-manifest",
        type=Path,
        help=(
            "Optional authenticated flat ROI-cache manifest used as the pixel "
            "provider while the package remains bound to --crop-run geometry."
        ),
    )
    parser.add_argument(
        "--roi-cache-expected-archive-path",
        type=Path,
        help=(
            "Optional canonical archive identity expected by a staged/cache "
            "manifest. Defaults to zarr_path when a cache is supplied."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write and validate the package. Without this flag, only print the plan.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    zarr_path = args.zarr_path.expanduser().resolve()
    rows = _load_rows(args)
    root = zarr.open_group(str(zarr_path), mode="r", use_consolidated=False)
    source = CropImageSource.open(
        root,
        crop_run=args.crop_run,
        zarr_path=zarr_path,
        roi_cache_policy="never",
        roi_cache_manifest=args.roi_cache_manifest,
        roi_cache_expected_archive_path=(
            args.roi_cache_expected_archive_path or zarr_path
            if args.roi_cache_manifest is not None
            else None
        ),
    )
    try:
        unique = np.unique(rows)
        in_bounds = bool(
            rows.size
            and int(rows.min()) >= 0
            and int(rows.max()) < int(source.total_rois)
        )
        canonical_order = bool(np.array_equal(rows, np.sort(rows, kind="stable")))
        plan = {
            "action": "apply" if args.apply else "dry_run",
            "zarr_path": str(zarr_path),
            "crop_run": str(source.crop_run_name),
            "manifest_path": str(args.manifest.expanduser().resolve()),
            "source_crop_rows": int(source.total_rois),
            "selected_rows": int(rows.shape[0]),
            "selected_rows_unique": int(unique.shape[0]),
            "selection_in_bounds": in_bounds,
            "selection_canonical_ascending": canonical_order,
            "roi_shape": [int(value) for value in source.roi_shape],
            "source_roi_read_mode": str(source.roi_read_mode),
            "source_roi_cache_used": bool(source.roi_cache_used),
            "estimated_pixel_payload_bytes": int(
                rows.shape[0] * source.roi_shape[0] * source.roi_shape[1]
            ),
            "downstream_contract": {
                "keypoints_output_parent": "keypoint_shard_runs",
                "subject_masks_output_parent": "subject_mask_shard_runs",
                "canonical_publication": "finalizer_only",
            },
        }
        if not in_bounds or int(unique.shape[0]) != int(rows.shape[0]) or not canonical_order:
            raise ValueError(
                "Selected crop rows must be unique, ascending, and within the crop run."
            )
        if args.apply:
            plan["package"] = build_crop_pixel_work_package_from_source(
                source,
                target_crop_rows=rows,
                manifest_path=args.manifest,
                archive_path=zarr_path,
                batch_rows=int(args.batch_rows),
                overwrite=bool(args.overwrite),
            )
        print(json.dumps(plan, indent=2, sort_keys=True, allow_nan=False))
    finally:
        source.close()


if __name__ == "__main__":
    main()
