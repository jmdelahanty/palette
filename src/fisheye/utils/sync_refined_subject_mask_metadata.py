#!/usr/bin/env python3
"""Sync refined-subject rows after external edits or source subject-mask updates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

from fisheye.tune.refined_subject_mask_review import (
    check_refined_subject_source_updates,
    sync_refined_subject_mask_metadata,
)


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
    parser.add_argument(
        "--check-source-updates",
        action="store_true",
        help=(
            "Inspect the selected refined rows against the current source subject-mask run, "
            "auto-sync untouched rows, and mark manually overridden rows stale."
        ),
    )
    parser.add_argument(
        "--assume-source-changed-untracked",
        action="store_true",
        help=(
            "Legacy bootstrap for refined runs created before source-sync tracking existed. "
            "Treat the selected rows as source-changed when no historical source fingerprints are present."
        ),
    )
    parser.add_argument(
        "--force-source-changed",
        action="store_true",
        help=(
            "Force the selected refined rows to be treated as source-changed even if source-sync "
            "fingerprints already exist. Use this to recover after an earlier bootstrap seeded the baseline."
        ),
    )
    args = parser.parse_args(argv)

    helper = check_refined_subject_source_updates if args.check_source_updates else sync_refined_subject_mask_metadata
    kwargs = {
        "refined_run": str(args.refined_run),
        "component_name": str(args.component_name),
        "roi_indices": list(args.roi_indices),
        "source_subject_mask_run": str(args.source_subject_mask_run) if args.source_subject_mask_run else None,
    }
    if args.check_source_updates:
        kwargs["assume_source_changed_untracked"] = bool(args.assume_source_changed_untracked)
        kwargs["force_source_changed"] = bool(args.force_source_changed)
    summary = helper(
        args.zarr_path,
        **kwargs,
    )
    if args.dataset:
        summary["dataset"] = str(args.dataset)
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
