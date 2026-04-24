#!/usr/bin/env python3
"""Validate merged subject-mask-training Zarr layout and split/index integrity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional

from fisheye.utils.export_subject_mask_training_zarr import validate_merged_subject_mask_training_zarr


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zarr_path", type=Path, help="Merged subject-mask training .zarr path.")
    parser.add_argument(
        "--expected-input-format",
        choices=["gray", "rgb"],
        default=None,
        help="Optional expected training input format for roi_images checks.",
    )
    parser.add_argument(
        "--expected-label-schema-id",
        choices=["subject_v1_union", "subject_v1_lr"],
        default=None,
        help="Optional expected subject-mask label schema.",
    )
    parser.add_argument(
        "--expected-total-samples",
        type=int,
        default=None,
        help="Optional expected sample count (N).",
    )
    args = parser.parse_args(argv)

    summary = validate_merged_subject_mask_training_zarr(
        args.zarr_path,
        expected_input_format=args.expected_input_format,
        expected_label_schema_id=args.expected_label_schema_id,
        expected_total_samples=args.expected_total_samples,
    )
    print("Validation passed.")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
