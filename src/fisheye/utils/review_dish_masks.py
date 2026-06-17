#!/usr/bin/env python3
"""
Batch review dish masks by launching the interactive mask tuner per Zarr.

Use this to step through recordings and visually confirm or adjust dish masks.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import h5py
import zarr

from fisheye.shared.type_conversions import normalize_attr as _normalize_attr


@dataclass
class ReviewPlan:
    zarr_path: Path
    chamber: Optional[str]
    camera_id: Optional[str]
    has_mask: bool
    status: str
    reason: Optional[str] = None


def _iter_zarr(paths: List[Path], recursive: bool) -> Iterable[Path]:
    for path in paths:
        path = path.expanduser()
        if path.is_dir() and path.suffix == ".zarr":
            yield path
            continue
        if not path.exists():
            continue
        if recursive:
            yield from path.rglob("*.zarr")
        else:
            yield from path.glob("*/zarr/*.zarr")
            yield from path.glob("*.zarr")


def _derive_camera_id(ipc_source_name: object) -> Optional[str]:
    if ipc_source_name is None:
        return None
    text = _normalize_attr(ipc_source_name)
    if text is None:
        return None
    digits = "".join([c for c in text if c.isdigit()])
    return digits or None


def _read_camera_id(h5_path: Path) -> Optional[str]:
    try:
        with h5py.File(h5_path, "r") as h5:
            root = h5.attrs
            if "camera_id" in root:
                cam = _normalize_attr(root.get("camera_id"))
                if cam:
                    return cam
            ipc = _normalize_attr(root.get("ipc_source_name"))
            return _derive_camera_id(ipc)
    except Exception:
        return None


def _find_h5_for_zarr(zarr_path: Path) -> Optional[Path]:
    recording_dir = zarr_path.parent.parent
    raw_dir = recording_dir / "raw"
    if not raw_dir.exists():
        return None
    h5s = list(raw_dir.glob("*.h5"))
    if len(h5s) == 1:
        return h5s[0]
    return None


def _build_plans(
    roots: List[Path],
    recursive: bool,
    chamber_filter: Optional[str],
    camera_filter: Optional[str],
    only_missing: bool,
    only_present: bool,
) -> List[ReviewPlan]:
    plans: List[ReviewPlan] = []
    for zarr_path in _iter_zarr(roots, recursive):
        try:
            root = zarr.open(str(zarr_path), mode="r")
        except Exception as exc:
            plans.append(
                ReviewPlan(
                    zarr_path=zarr_path,
                    chamber=None,
                    camera_id=None,
                    has_mask=False,
                    status="error",
                    reason=str(exc),
                )
            )
            continue

        chamber = _normalize_attr(root.attrs.get("experimental_chamber"))
        if chamber_filter and chamber != chamber_filter:
            continue

        analysis = root.get("analysis_metadata")
        has_mask = bool(analysis is not None and "dish_mask" in analysis.attrs)
        if only_missing and has_mask:
            continue
        if only_present and not has_mask:
            continue

        camera_id = None
        if camera_filter:
            h5_path = _find_h5_for_zarr(zarr_path)
            if h5_path is None:
                continue
            camera_id = _read_camera_id(h5_path)
            if camera_id != camera_filter:
                continue
        else:
            h5_path = _find_h5_for_zarr(zarr_path)
            if h5_path is not None:
                camera_id = _read_camera_id(h5_path)

        plans.append(
            ReviewPlan(
                zarr_path=zarr_path,
                chamber=chamber,
                camera_id=camera_id,
                has_mask=has_mask,
                status="ok",
            )
        )
    return sorted(plans, key=lambda p: str(p.zarr_path))


def _prompt_continue() -> bool:
    resp = input("Press Enter for next, or type 'q' to quit: ").strip().lower()
    return resp != "q"


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Batch review dish masks by opening the mask tuner for each Zarr."
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
        "--chamber",
        type=str,
        help="Only review recordings with this experimental_chamber.",
    )
    parser.add_argument(
        "--camera-id",
        type=str,
        help="Only review recordings with this camera_id (read from H5).",
    )
    parser.add_argument(
        "--only-missing",
        action="store_true",
        help="Only include Zarrs missing dish_mask.",
    )
    parser.add_argument(
        "--only-present",
        action="store_true",
        help="Only include Zarrs that already have dish_mask.",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Use full-resolution frames in the tuner.",
    )
    parser.add_argument(
        "--frame",
        type=int,
        help="Frame index to use in the tuner (default: tuner chooses mid-frame).",
    )
    parser.add_argument(
        "--registry",
        type=Path,
        help="Optional registry SQLite path passed to mask_tuner for dish_mask status sync on save.",
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
        help="List candidates and exit without opening the tuner.",
    )
    args = parser.parse_args(argv)

    if args.paths:
        roots = args.paths
    else:
        env_root = os.environ.get("PALETTE_RECORDINGS_ROOT")
        roots = [Path(env_root)] if env_root else [Path("/nvme1/recordings")]
    plans = _build_plans(
        roots,
        args.recursive,
        args.chamber,
        args.camera_id,
        args.only_missing,
        args.only_present,
    )

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
        mask_status = "mask" if plan.has_mask else "no-mask"
        chamber = plan.chamber or "unknown"
        cam = plan.camera_id or "unknown"
        print(f"{marker} [{idx:03d}] {plan.zarr_path} | {chamber} | cam={cam} | {mask_status}")

    if args.list:
        return 0

    for idx in range(start, end):
        plan = plans[idx]
        print(f"\n[{idx + 1}/{end}] Opening tuner for {plan.zarr_path}")
        cmd = [sys.executable, "-m", "fisheye.tune.mask_tuner", str(plan.zarr_path)]
        if args.full:
            cmd.append("--full")
        if args.frame is not None:
            cmd.extend(["--frame", str(args.frame)])
        if args.registry is not None:
            cmd.extend(["--registry", str(args.registry)])
        subprocess.run(cmd, check=False)
        if idx < end - 1:
            if not _prompt_continue():
                break

    print("Review complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
