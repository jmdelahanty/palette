#!/usr/bin/env python3
"""Prepare one immutable global row/chunk grid for refined mask packages."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from fisheye.shared.refined_subject_mask_encoded_chunks import prepare_global_mask_chunk_grid


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zarr", required=True, type=Path)
    parser.add_argument("--crop-run", required=True)
    parser.add_argument("--output-manifest", required=True, type=Path)
    parser.add_argument("--mask-label", dest="mask_labels", action="append", required=True)
    parser.add_argument("--mask-height", type=int, default=512)
    parser.add_argument("--mask-width", type=int, default=512)
    parser.add_argument("--dense-mask-row-chunk", type=int, default=128)
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    result = prepare_global_mask_chunk_grid(
        zarr_path=args.zarr,
        crop_run=args.crop_run,
        output_manifest=args.output_manifest,
        mask_labels=args.mask_labels,
        mask_height=int(args.mask_height),
        mask_width=int(args.mask_width),
        dense_mask_row_chunk=int(args.dense_mask_row_chunk),
    )
    print(json.dumps(result, indent=2, sort_keys=True) if args.json else result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
