#!/usr/bin/env python3
"""Write one refined subject-mask edit and refresh Palette-owned metadata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from fisheye.tune.refined_subject_mask_review import (
    DEFAULT_REFINED_SUBJECT_WRITEBACK_REASON,
    write_refined_subject_mask_edit,
)


def _parse_mask_shape(text: str) -> tuple[int, int]:
    raw = str(text or "").lower().replace("x", ",").replace(" ", "")
    parts = [part for part in raw.split(",") if part]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("Mask shape must be formatted as H,W or HxW.")
    try:
        height, width = (int(parts[0]), int(parts[1]))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid mask shape '{text}'.") from exc
    if height <= 0 or width <= 0:
        raise argparse.ArgumentTypeError("Mask shape dimensions must be positive.")
    return height, width


def _load_binary_mask(path: Path, *, raw_shape: tuple[int, int] | None = None) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        mask = np.load(path)
    elif suffix in {".png", ".tif", ".tiff", ".jpg", ".jpeg", ".bmp"}:
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"Unable to read image mask: {path}")
    elif suffix in {".raw", ".bin"}:
        if raw_shape is None:
            raise ValueError("--mask-shape is required for raw/bin mask payloads.")
        values = np.fromfile(path, dtype=np.uint8)
        expected = int(raw_shape[0]) * int(raw_shape[1])
        if int(values.size) != expected:
            raise ValueError(f"Raw mask size mismatch: expected {expected} bytes, got {int(values.size)}.")
        mask = values.reshape(raw_shape)
    else:
        raise ValueError(f"Unsupported mask payload extension '{suffix}'. Use .npy, image, .raw, or .bin.")

    arr = np.asarray(mask)
    if arr.ndim != 2:
        raise ValueError(f"Mask payload must be 2D, got shape {tuple(arr.shape)}.")
    return (arr > 0).astype(np.uint8, copy=False)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zarr-path", required=True, type=Path, help="Path to the Palette analysis zarr archive.")
    parser.add_argument("--refined-run", required=True, help="Target refined_subject_masks_runs/<run>.")
    parser.add_argument(
        "--component-name",
        required=True,
        help="Target refined subject-mask component. Resolved against mask_labels and available_channels.",
    )
    parser.add_argument("--roi-index", required=True, type=int, help="ROI row index to update.")
    parser.add_argument("--mask-path", required=True, type=Path, help="2D binary mask payload path.")
    parser.add_argument(
        "--mask-shape",
        type=_parse_mask_shape,
        default=None,
        help="Required for raw/bin payloads. Format as H,W or HxW.",
    )
    parser.add_argument(
        "--reason",
        default=DEFAULT_REFINED_SUBJECT_WRITEBACK_REASON,
        help="Row-local update reason recorded with the component revision.",
    )
    parser.add_argument(
        "--source-subject-mask-run",
        default=None,
        help="Optional subject_mask_runs/<run> override. Defaults to the refined run lineage attr.",
    )
    parser.add_argument("--validate", action="store_true", help="Validate row-local metrics after the write.")
    args = parser.parse_args(argv)

    mask = _load_binary_mask(args.mask_path, raw_shape=args.mask_shape)
    summary = write_refined_subject_mask_edit(
        args.zarr_path,
        refined_run=str(args.refined_run),
        component_name=str(args.component_name),
        roi_index=int(args.roi_index),
        mask=mask,
        reason=str(args.reason),
        source_subject_mask_run=str(args.source_subject_mask_run) if args.source_subject_mask_run else None,
        validate=bool(args.validate),
    )
    summary["mask_path"] = str(args.mask_path)
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
