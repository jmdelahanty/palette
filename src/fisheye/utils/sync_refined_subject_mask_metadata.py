#!/usr/bin/env python3
"""Sync refined-subject mask metadata for touched ROIs after external pixel edits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

from fisheye.tune.refined_subject_mask_review import sync_refined_subject_mask_metadata


def _parse_roi_indices(text: str) -> list[int]:
    raw = str(text or "").replace(" ", "")
    if not raw:
        raise argparse.ArgumentTypeError("ROI indices must not be empty.")
    try:
        return [int(token) for token in raw.split(",") if token]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid ROI indices '{text}'. Expected comma-separated integers.") from exc


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zarr-path", required=True, type=Path, help="Path to the Palette zarr archive.")
    parser.add_argument("--refined-run", required=True, help="Target refined_subject_masks_runs/<run>.")
    parser.add_argument("--component-name", required=True, help="Touched refined subject-mask component name.")
    parser.add_argument(
        "--dataset",
        default=None,
        help="Optional Paintera synthetic dataset path for logging/compatibility. Ignored by the Palette helper.",
    )
    parser.add_argument(
        "--roi-indices",
        required=True,
        type=_parse_roi_indices,
        help="Comma-separated ROI row indices touched by the external editor.",
    )
    parser.add_argument(
        "--source-subject-mask-run",
        default=None,
        help="Optional raw subject_mask_runs/<run> override. Defaults to the refined run lineage attr.",
    )
    args = parser.parse_args(argv)

    summary = sync_refined_subject_mask_metadata(
        args.zarr_path,
        refined_run=str(args.refined_run),
        component_name=str(args.component_name),
        roi_indices=list(args.roi_indices),
        source_subject_mask_run=str(args.source_subject_mask_run) if args.source_subject_mask_run else None,
    )
    if args.dataset:
        summary["dataset"] = str(args.dataset)
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
