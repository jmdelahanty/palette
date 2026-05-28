#!/usr/bin/env python3
"""
Batch review crop runs by launching the crop visualizer per Zarr.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import zarr

from fisheye.shared.zarr_discovery import iter_filesystem_zarrs, load_path_list


@dataclass
class CropReviewPlan:
    zarr_path: Path
    crop_run: Optional[str]
    status: str
    reason: Optional[str] = None
    total_crops: Optional[int] = None


def _latest_run(parent: zarr.Group) -> Optional[str]:
    latest = parent.attrs.get("latest")
    if latest and latest in parent:
        return str(latest)
    if hasattr(parent, "group_keys"):
        names = list(parent.group_keys())
    else:
        names = list(parent.keys())
    if not names:
        return None
    return sorted(names)[-1]


def _build_plans(roots: List[Path], recursive: bool, crop_run: Optional[str]) -> List[CropReviewPlan]:
    plans: List[CropReviewPlan] = []
    for zarr_path in iter_filesystem_zarrs(roots, recursive):
        try:
            root = zarr.open(str(zarr_path), mode="r")
        except Exception as exc:
            plans.append(
                CropReviewPlan(
                    zarr_path=zarr_path,
                    crop_run=None,
                    status="error",
                    reason=str(exc),
                )
            )
            continue

        if "crop_runs" not in root:
            plans.append(
                CropReviewPlan(
                    zarr_path=zarr_path,
                    crop_run=None,
                    status="missing",
                    reason="no crop_runs",
                )
            )
            continue

        crop_parent = root["crop_runs"]
        resolved_run = crop_run or _latest_run(crop_parent)
        if not resolved_run:
            plans.append(
                CropReviewPlan(
                    zarr_path=zarr_path,
                    crop_run=None,
                    status="missing",
                    reason="no crop runs",
                )
            )
            continue

        if resolved_run not in crop_parent:
            plans.append(
                CropReviewPlan(
                    zarr_path=zarr_path,
                    crop_run=resolved_run,
                    status="missing",
                    reason="crop run not found",
                )
            )
            continue

        crop_group = crop_parent[resolved_run]
        total_crops = None
        try:
            total_crops = int(crop_group["roi_images"].shape[0])
        except Exception:
            total_crops = None

        plans.append(
            CropReviewPlan(
                zarr_path=zarr_path,
                crop_run=resolved_run,
                status="ok",
                total_crops=total_crops,
            )
        )
    return sorted(plans, key=lambda p: str(p.zarr_path))


def _prompt_continue() -> bool:
    resp = input("Press Enter for next, or type 'q' to quit: ").strip().lower()
    return resp != "q"


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Batch review crop runs by opening the crop visualizer for each Zarr."
    )
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="Recording roots or Zarr paths to scan.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively scan for Zarrs under each root.",
    )
    parser.add_argument(
        "--file-list",
        type=Path,
        action="append",
        help="Text file with one zarr path per line (comments with # allowed).",
    )
    parser.add_argument(
        "--crop-run",
        type=str,
        help="Specific crop run name to use (default: latest per Zarr).",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Start index within the review list (default: 0).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Maximum number of recordings to review.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List candidates and exit without opening the visualizer.",
    )
    parser.add_argument(
        "--no-prompt",
        action="store_true",
        help="Advance without prompting between recordings.",
    )
    parser.add_argument(
        "--review-state",
        default="approved",
        choices=["approved", "pending", "rejected", "needs_review"],
        help="Review state to set when pressing 'a' in the crop viewer (default: approved).",
    )
    parser.add_argument(
        "--review-method",
        default="manual",
        choices=["manual", "algorithmic", "hybrid", "spotcheck"],
        help="Review method label passed to the crop viewer (default: manual).",
    )
    parser.add_argument(
        "--review-intended-use",
        default="training",
        choices=["training", "full_recording"],
        help="Intended use label passed to the crop viewer (default: training).",
    )
    parser.add_argument(
        "--reviewer",
        type=str,
        help="Optional reviewer name forwarded to the crop viewer.",
    )
    parser.add_argument(
        "--review-notes",
        type=str,
        help="Optional review notes forwarded to the crop viewer.",
    )

    args = parser.parse_args(argv)

    file_list_paths: List[Path] = []
    if args.file_list:
        for path in args.file_list:
            file_list_paths.extend(load_path_list(path, wrap_errors=True))

    if args.paths:
        roots = list(args.paths) + file_list_paths
    elif file_list_paths:
        roots = file_list_paths
    else:
        env_root = os.environ.get("PALETTE_RECORDINGS_ROOT")
        roots = [Path(env_root)] if env_root else [Path("/nvme1/recordings")]

    plans = _build_plans(roots, args.recursive, args.crop_run)

    if not plans:
        print("No recordings found to review.")
        return 0

    start = max(args.start, 0)
    if start >= len(plans):
        print(f"--start {start} >= total {len(plans)}")
        return 1

    end = len(plans)
    if args.limit is not None:
        end = min(end, start + max(args.limit, 0))

    print(f"Review list: {len(plans)} total; showing {start}–{end - 1}")
    for idx, plan in enumerate(plans):
        marker = "→" if start <= idx < end else " "
        status = plan.status
        reason = f" ({plan.reason})" if plan.reason else ""
        run = plan.crop_run or "none"
        total = f", crops={plan.total_crops}" if plan.total_crops is not None else ""
        print(f"{marker} [{idx:03d}] {plan.zarr_path} | crop={run}{total} | {status}{reason}")

    if args.list:
        return 0

    for idx in range(start, end):
        plan = plans[idx]
        if plan.status != "ok":
            print(f"\n[{idx + 1}/{end}] Skipping {plan.zarr_path} ({plan.status})")
            continue
        print(f"\n[{idx + 1}/{end}] Opening crop viewer for {plan.zarr_path}")
        cmd = [
            sys.executable,
            "-m",
            "fisheye.visualization.visualize_crops",
            str(plan.zarr_path),
            "--review-state",
            args.review_state,
            "--review-method",
            args.review_method,
            "--review-intended-use",
            args.review_intended_use,
        ]
        if plan.crop_run:
            cmd.extend(["--crop-run", plan.crop_run])
        if args.reviewer:
            cmd.extend(["--reviewer", args.reviewer])
        if args.review_notes:
            cmd.extend(["--review-notes", args.review_notes])
        subprocess.run(cmd, check=False)
        if idx < end - 1 and not args.no_prompt:
            if not _prompt_continue():
                break

    print("Review complete.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
