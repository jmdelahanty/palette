#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import zarr

from fisheye.shared.refined_detect_review import (
    DEFAULT_DETECT_GROUP_PREFERENCE,
    resolve_refined_detect_group,
)


def _pick_refined_parent(root: zarr.Group) -> Optional[zarr.Group]:
    if "refined_detect_runs" in root:
        return root["refined_detect_runs"]
    if "refined_runs" in root:
        return root["refined_runs"]
    return None


def _select_refined_run(parent: zarr.Group, requested: Optional[str]) -> str:
    if requested:
        if requested in parent:
            return requested
        raise RuntimeError(f"Refined run '{requested}' not found.")
    latest = parent.attrs.get("latest")
    if latest and latest in parent:
        return str(latest)
    try:
        names = list(parent.group_keys())
    except Exception:
        names = list(parent.keys())
    if not names:
        raise RuntimeError("No refined runs available.")
    return sorted(names)[-1]


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Set detect_review_status on refined detect runs."
    )
    parser.add_argument("zarr_path", type=Path, help="Path to the recording .zarr directory.")
    parser.add_argument("--refined-run", help="Refined run name (default: latest).")
    parser.add_argument(
        "--state",
        default="approved",
        choices=["approved", "pending", "rejected", "needs_review"],
        help="Review state (default: approved).",
    )
    parser.add_argument(
        "--method",
        default="manual",
        choices=["manual", "algorithmic", "hybrid", "spotcheck"],
        help="Review method (default: manual).",
    )
    parser.add_argument(
        "--intended-use",
        default="training",
        choices=["training", "full_recording"],
        help="Intended use (default: training).",
    )
    parser.add_argument("--reviewer", help="Reviewer name or identifier.")
    parser.add_argument("--notes", help="Optional notes.")
    parser.add_argument(
        "--target-group",
        help="Explicit refined group to approve (manual/interpolated/filtered/raw or custom).",
    )
    parser.add_argument(
        "--no-latest",
        action="store_true",
        help="Do not update refined_detect_runs.attrs['detect_review_status_latest'].",
    )
    args = parser.parse_args(argv)

    root = zarr.open_group(str(args.zarr_path), mode="a")
    refined_parent = _pick_refined_parent(root)
    if refined_parent is None:
        raise RuntimeError("No refined_detect_runs found in archive.")

    refined_run_name = _select_refined_run(refined_parent, args.refined_run)
    refined_run = refined_parent[refined_run_name]

    resolution = resolve_refined_detect_group(
        refined_run,
        preference=DEFAULT_DETECT_GROUP_PREFERENCE,
        override_group=args.target_group,
    )

    payload: dict[str, object] = {
        "state": args.state,
        "method": args.method,
        "intended_use": args.intended_use,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "resolved_group": resolution.label or resolution.group,
        "target_group": args.target_group or None,
        "preference_chain": list(DEFAULT_DETECT_GROUP_PREFERENCE),
    }
    if args.reviewer:
        payload["reviewer"] = args.reviewer
    if args.notes:
        payload["notes"] = args.notes

    # Drop None values to keep attrs clean.
    payload = {key: value for key, value in payload.items() if value is not None}
    refined_run.attrs["detect_review_status"] = payload

    if not args.no_latest:
        refined_parent.attrs["detect_review_status_latest"] = refined_run_name

    print(f"Set detect_review_status on refined_detect_runs/{refined_run_name}")
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
