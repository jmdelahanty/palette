#!/usr/bin/env python3
"""Minimal Zarr reader for subject masks in the transferred example recording.

This example intentionally reads one ROI row at a time. Dense mask arrays can be
large, so avoid slicing ``[:]`` unless you really want to load the full run.

Example:

    python docs/examples/read_subject_masks_from_example_recording.py --row 0

With the Palette repo environment:

    scripts/py docs/examples/read_subject_masks_from_example_recording.py --row 0
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np
import zarr


DEFAULT_ANALYSIS_ZARR = Path(
    "/groups/anibody/anibody/fish/example_recording/zarr/"
    "2026-01-28T21-47-47Z_arena_1_DefaultScreen_analysis.zarr"
)
DEFAULT_RAW_RUN = "subject_masks_unet_registry_batch_20260504"
DEFAULT_REFINED_RUN = "refined_subject_masks_smart_finalizer_batch_20260504"


def _labels(group: zarr.Group) -> list[str]:
    labels = group.attrs.get("mask_labels", [])
    return [str(label) for label in labels]


def _print_array_summary(name: str, array: zarr.Array) -> None:
    print(f"{name}: shape={array.shape} dtype={array.dtype} chunks={array.chunks}")


def _component_area(mask_stack: zarr.Array, row: int, channel: int) -> int:
    # This reads only one 512x512 component mask, not the full mask run.
    mask = np.asarray(mask_stack[row, channel], dtype=np.uint8)
    return int(mask.astype(bool).sum())


def read_subject_mask_example(
    analysis_zarr: Path,
    *,
    raw_run: str,
    refined_run: str,
    row: int,
) -> None:
    root = zarr.open_group(str(analysis_zarr), mode="r")

    raw = root[f"subject_mask_runs/{raw_run}"]
    refined = root[f"refined_subject_masks_runs/{refined_run}"]

    raw_labels = _labels(raw)
    refined_labels = _labels(refined)

    print(f"analysis_zarr: {analysis_zarr}")
    print(f"raw_run: {raw_run}")
    print(f"raw_labels: {raw_labels}")
    _print_array_summary("raw/mask_probs_roi", raw["mask_probs_roi"])
    _print_array_summary("raw/metrics/prob_max", raw["metrics/prob_max"])
    print()

    print(f"refined_run: {refined_run}")
    print(f"refined_labels: {refined_labels}")
    _print_array_summary("refined/masks_roi", refined["masks_roi"])
    _print_array_summary("refined/metrics/area_px", refined["metrics/area_px"])
    print()

    n_rows = int(refined["masks_roi"].shape[0])
    if row < 0 or row >= n_rows:
        raise IndexError(f"--row must be in [0, {n_rows - 1}], got {row}")

    frame_index = int(refined["frame_indices"][row])
    detection_index = int(refined["detection_indices"][row])
    print(f"row: {row}")
    print(f"frame_index: {frame_index}")
    print(f"detection_index: {detection_index}")

    print("refined component areas from masks_roi[row, channel]:")
    for channel, label in enumerate(refined_labels):
        area_px = _component_area(refined["masks_roi"], row, channel)
        cached_area_px = float(refined["metrics/area_px"][row, channel])
        print(f"  {label}: mask_area_px={area_px} cached_area_px={cached_area_px:.1f}")

    body_channel = refined_labels.index("subject_body")
    body_mask = np.asarray(refined["masks_roi"][row, body_channel], dtype=np.uint8).astype(bool)
    print()
    print("single loaded mask:")
    print(f"  label: subject_body")
    print(f"  shape: {body_mask.shape}")
    print(f"  dtype_after_bool_conversion: {body_mask.dtype}")
    print(f"  foreground_pixels: {int(body_mask.sum())}")

    if "subject_body" in raw_labels:
        raw_body_channel = raw_labels.index("subject_body")
        body_prob_u8 = np.asarray(raw["mask_probs_roi"][row, raw_body_channel], dtype=np.uint8)
        print()
        print("single loaded raw probability image:")
        print(f"  label: subject_body")
        print(f"  shape: {body_prob_u8.shape}")
        print(f"  dtype: {body_prob_u8.dtype}")
        print(f"  min_max_uint8: {int(body_prob_u8.min())}, {int(body_prob_u8.max())}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--analysis-zarr",
        type=Path,
        default=DEFAULT_ANALYSIS_ZARR,
        help=f"Path to the analysis Zarr. Default: {DEFAULT_ANALYSIS_ZARR}",
    )
    parser.add_argument("--raw-run", default=DEFAULT_RAW_RUN)
    parser.add_argument("--refined-run", default=DEFAULT_REFINED_RUN)
    parser.add_argument("--row", type=int, default=0, help="ROI row to read. Default: 0.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    read_subject_mask_example(
        args.analysis_zarr,
        raw_run=str(args.raw_run),
        refined_run=str(args.refined_run),
        row=int(args.row),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
