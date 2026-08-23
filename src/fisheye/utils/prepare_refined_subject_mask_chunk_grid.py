#!/usr/bin/env python3
"""Prepare one immutable global row/chunk grid for refined mask packages."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import zarr

from fisheye.shared.refined_subject_mask_encoded_chunks import (
    prepare_global_mask_chunk_grid,
)


def _fixed_crop_mask_shape(zarr_path: Path, *, crop_run: str) -> tuple[int, int]:
    root = zarr.open_group(
        str(zarr_path.expanduser().resolve()), mode="r", use_consolidated=False
    )
    crop = root[f"crop_runs/{crop_run}"]
    if "roi_sizes_full" not in crop:
        raise ValueError(
            f"crop_runs/{crop_run} is missing authoritative roi_sizes_full."
        )
    sizes = np.asarray(crop["roi_sizes_full"][:])
    if (
        sizes.dtype != np.dtype(np.int32)
        or sizes.ndim != 2
        or sizes.shape[0] <= 0
        or sizes.shape[1] != 2
        or np.any(sizes <= 0)
        or not np.all(sizes == sizes[0])
    ):
        raise ValueError(
            f"crop_runs/{crop_run} must have one fixed positive int32 ROI size."
        )
    width, height = (int(value) for value in sizes[0])
    return height, width


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zarr", required=True, type=Path)
    parser.add_argument("--crop-run", required=True)
    parser.add_argument("--output-manifest", required=True, type=Path)
    parser.add_argument(
        "--mask-label", dest="mask_labels", action="append", required=True
    )
    parser.add_argument(
        "--mask-height",
        type=int,
        help="Explicit mask height; defaults to the exact crop-run ROI height.",
    )
    parser.add_argument(
        "--mask-width",
        type=int,
        help="Explicit mask width; defaults to the exact crop-run ROI width.",
    )
    parser.add_argument("--dense-mask-row-chunk", type=int, default=128)
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if (args.mask_height is None) != (args.mask_width is None):
        raise ValueError("--mask-height and --mask-width must be provided together.")
    if args.mask_height is None:
        mask_height, mask_width = _fixed_crop_mask_shape(
            args.zarr,
            crop_run=str(args.crop_run),
        )
    else:
        mask_height = int(args.mask_height)
        mask_width = int(args.mask_width)
        if mask_height <= 0 or mask_width <= 0:
            raise ValueError("Explicit mask dimensions must be positive.")
    result = prepare_global_mask_chunk_grid(
        zarr_path=args.zarr,
        crop_run=args.crop_run,
        output_manifest=args.output_manifest,
        mask_labels=args.mask_labels,
        mask_height=mask_height,
        mask_width=mask_width,
        dense_mask_row_chunk=int(args.dense_mask_row_chunk),
    )
    print(json.dumps(result, indent=2, sort_keys=True) if args.json else result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
