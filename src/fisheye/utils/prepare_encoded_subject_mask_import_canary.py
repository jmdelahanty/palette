#!/usr/bin/env python3
"""Create a scratch Zarr and global grid from selected refined-mask packages."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import tempfile
from typing import Sequence

import numpy as np
import zarr

from fisheye.shared.refined_subject_mask_encoded_chunks import prepare_global_mask_chunk_grid
from fisheye.utils.import_refined_subject_mask_clip_packages import (
    _load_package,
    _validate_package_schema,
)


def prepare_canary(
    *,
    package_paths: Sequence[Path],
    output_zarr: Path,
    grid_manifest: Path,
    overwrite: bool = False,
) -> dict[str, object]:
    if len(package_paths) != 2:
        raise ValueError("Encoded import canary preparation requires exactly two packages.")
    output_zarr = output_zarr.expanduser().resolve()
    grid_manifest = grid_manifest.expanduser().resolve()
    if output_zarr.exists():
        if not overwrite:
            raise ValueError(f"Canary Zarr already exists: {output_zarr}")
        shutil.rmtree(output_zarr)
    with tempfile.TemporaryDirectory(prefix="palette_encoded_mask_canary_prepare_") as tmp:
        packages = [_load_package(path, Path(tmp)) for path in package_paths]
        _validate_package_schema(packages)
        all_ids = np.concatenate([package.source_crop_row_ids for package in packages])
        unique_ids = np.unique(all_ids)
        if int(unique_ids.shape[0]) != int(all_ids.shape[0]):
            raise ValueError("Canary packages contain duplicate source_crop_row_ids.")
        sorted_ids = np.sort(all_ids, kind="stable").astype(np.int64, copy=False)
        reference = packages[0].group
        labels = [str(value) for value in reference.attrs.get("mask_labels") or []]
        masks = reference["masks_roi"]
        row_shape = tuple(int(value) for value in masks.shape[1:])
        chunks = tuple(int(value) for value in masks.chunks)
        package_id_ranges = [
            (
                int(package.source_crop_row_ids.min()),
                int(package.source_crop_row_ids.max()),
                int(package.row_count),
            )
            for package in packages
        ]
        first_min, first_max, first_rows = package_id_ranges[0]
        second_min, _second_max, _second_rows = package_id_ranges[1]
        if first_max >= second_min or first_max + 1 != second_min:
            raise ValueError(
                "Canary packages must be ordered adjacent crop-identity intervals; "
                f"observed {package_id_ranges!r}."
            )
        boundary_offset = first_rows % int(chunks[0])
        if boundary_offset == 0:
            raise ValueError(
                f"Canary package boundary after {first_rows} rows aligns with row chunk {chunks[0]}; "
                "choose a deliberately misaligned adjacent pair."
            )
        crop_runs = sorted({str(package.group.attrs.get("source_crop_run") or "") for package in packages})
        if len(crop_runs) != 1 or not crop_runs[0]:
            raise ValueError(f"Canary packages do not share one source_crop_run: {crop_runs!r}")

    root = zarr.open_group(str(output_zarr), mode="w")
    crop = root.require_group("crop_runs").create_group(crop_runs[0])
    crop.create_array(
        "source_crop_row_ids",
        data=sorted_ids,
        chunks=(max(1, min(16384, int(sorted_ids.shape[0]))),),
        overwrite=True,
    )
    grid = prepare_global_mask_chunk_grid(
        zarr_path=output_zarr,
        crop_run=crop_runs[0],
        output_manifest=grid_manifest,
        mask_labels=labels,
        mask_height=row_shape[1],
        mask_width=row_shape[2],
        dense_mask_row_chunk=chunks[0],
    )
    return {
        "status": "ok",
        "output_zarr": str(output_zarr),
        "grid_manifest": str(grid_manifest),
        "package_count": int(len(package_paths)),
        "row_count": int(sorted_ids.shape[0]),
        "source_crop_run": crop_runs[0],
        "mask_labels": labels,
        "mask_shape": list(grid["mask_shape"]),
        "dense_mask_chunks": list(grid["dense_mask_chunks"]),
        "package_id_ranges": [list(values) for values in package_id_ranges],
        "boundary_offset_within_row_chunk": int(boundary_offset),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", dest="packages", action="append", required=True, type=Path)
    parser.add_argument("--output-zarr", required=True, type=Path)
    parser.add_argument("--grid-manifest", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    result = prepare_canary(
        package_paths=args.packages,
        output_zarr=args.output_zarr,
        grid_manifest=args.grid_manifest,
        overwrite=bool(args.overwrite),
    )
    print(json.dumps(result, indent=2, sort_keys=True) if args.json else result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
