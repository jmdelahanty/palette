#!/usr/bin/env python3
"""
Apply dish mask parameters across recordings that share experimental_chamber.

For circle masks, this re-detects the circle per target Zarr using the saved
Hough parameters from the source dish_mask.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import cv2
import numpy as np
import zarr

from fisheye.shared.batch_logging import JsonLogger as SharedJsonLogger
from fisheye.shared.batch_logging import make_run_id
from fisheye.shared.batch_logging import utc_now
from fisheye.tune.mask_tuner import save_mask_to_zarr


_utc_now = utc_now
_run_id = make_run_id
JsonLogger = SharedJsonLogger


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


def _load_dish_mask(source_zarr: Path) -> Optional[dict]:
    root = zarr.open(str(source_zarr), mode="r")
    analysis = root.get("analysis_metadata")
    if analysis is None:
        return None
    mask = analysis.attrs.get("dish_mask")
    return mask if isinstance(mask, dict) else None


def _resolve_roots(paths: List[Path]) -> List[Path]:
    if paths:
        return paths
    env_root = os.environ.get("PALETTE_RECORDINGS_ROOT")
    if env_root:
        return [Path(env_root)]
    return [Path("/nvme1/recordings")]


def _resolve_log_dir(arg_log_dir: Optional[Path], roots: List[Path]) -> Path:
    if arg_log_dir is not None:
        return arg_log_dir
    env_root = os.environ.get("PALETTE_LOG_ROOT")
    if env_root:
        return Path(env_root) / "apply_dish_mask_by_chamber"
    if roots:
        return roots[0] / "logs" / "apply_dish_mask_by_chamber"
    return Path.cwd() / "logs" / "apply_dish_mask_by_chamber"


def _detect_circle(frame: np.ndarray, param1: int, param2: int, radius_adjustment: int) -> Optional[dict]:
    if frame.ndim == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        1,
        100,
        param1=param1,
        param2=param2,
        minRadius=0,
        maxRadius=0,
    )
    if circles is None:
        return None
    circles = np.uint16(np.around(circles))
    x, y, r = circles[0, 0]
    r = int(r) + int(radius_adjustment)
    return {"center": [int(x), int(y)], "radius": int(r)}


@dataclass
class Plan:
    zarr_path: Path
    status: str
    reason: Optional[str] = None


def _build_plans(
    roots: List[Path], recursive: bool, chamber: str, source: Path
) -> List[Plan]:
    plans: List[Plan] = []
    for zarr_path in _iter_zarr(roots, recursive):
        if zarr_path == source:
            continue
        try:
            root = zarr.open(str(zarr_path), mode="r")
        except Exception as exc:
            plans.append(Plan(zarr_path=zarr_path, status="error", reason=str(exc)))
            continue
        if root.attrs.get("experimental_chamber") != chamber:
            continue
        plans.append(Plan(zarr_path=zarr_path, status="ok"))
    return plans


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Apply dish mask parameters across recordings sharing experimental_chamber.",
    )
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="Recording roots or Zarr paths to scan.",
    )
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Source Zarr with tuned dish_mask.",
    )
    parser.add_argument(
        "--chamber",
        type=str,
        help="Experimental chamber name to match (default: from source zarr root attrs).",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively scan for Zarrs under each root.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply updates (default: dry-run).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing dish_mask in targets.",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Use full-resolution frames (default: downsampled).",
    )
    parser.add_argument(
        "--frame",
        type=int,
        help="Frame index to use for detection (default: mid-frame).",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        help="Directory for JSONL logs (default: $PALETTE_LOG_ROOT/apply_dish_mask_by_chamber or <recordings_root>/logs/apply_dish_mask_by_chamber).",
    )
    parser.add_argument(
        "--no-log",
        action="store_true",
        help="Disable JSONL logging.",
    )
    args = parser.parse_args(argv)

    source = args.source.expanduser()
    if not source.exists():
        print(f"Source Zarr not found: {source}")
        return 1

    source_root = zarr.open(str(source), mode="r")
    chamber = args.chamber or source_root.attrs.get("experimental_chamber")
    if not chamber:
        print("experimental_chamber missing from source; pass --chamber explicitly.")
        return 1

    dish_mask = _load_dish_mask(source)
    if not dish_mask:
        print("dish_mask not found in source analysis_metadata.")
        return 1

    shape = dish_mask.get("shape")
    if shape not in {"circle", "rectangle"}:
        print(f"Unsupported dish_mask shape: {shape}")
        return 1

    roots = _resolve_roots(args.paths)
    logger: Optional[JsonLogger] = None
    log_path: Optional[Path] = None
    run_id = _run_id()
    if not args.no_log:
        log_dir = _resolve_log_dir(args.log_dir, roots)
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"apply_dish_mask_by_chamber_{run_id}.jsonl"
        logger = JsonLogger(log_path, run_id)
        print(f"Log file: {log_path}")
        logger.log(
            "run_start",
            roots=[str(root) for root in roots],
            recursive=bool(args.recursive),
            apply=bool(args.apply),
            dry_run=not bool(args.apply),
            overwrite=bool(args.overwrite),
            full=bool(args.full),
            frame=args.frame,
            source=str(source),
            chamber=chamber,
        )

    plans = _build_plans(roots, args.recursive, chamber, source)
    if not plans:
        print(f"No targets found for experimental_chamber={chamber}")
        if logger is not None:
            logger.log(
                "run_end",
                chamber=chamber,
                updated=0,
                skipped=0,
                failed=0,
                targets=0,
            )
            logger.close()
        return 0

    if not args.apply:
        for plan in plans:
            print(f"Would apply to {plan.zarr_path}")
            if logger is not None:
                logger.log(
                    "dish_mask_plan",
                    zarr=str(plan.zarr_path),
                    status=plan.status,
                    reason=plan.reason,
                )
        print(f"\nDry-run complete. Targets: {len(plans)}")
        if logger is not None:
            logger.log(
                "run_end",
                chamber=chamber,
                updated=0,
                skipped=0,
                failed=0,
                targets=len(plans),
            )
            logger.close()
        return 0

    updated = 0
    skipped = 0
    failed = 0
    for plan in plans:
        try:
            root = zarr.open(str(plan.zarr_path), mode="r")
            analysis = root.get("analysis_metadata")
            if analysis is None:
                print(f"Skipping (analysis_metadata missing): {plan.zarr_path}")
                failed += 1
                if logger is not None:
                    logger.log(
                        "dish_mask_apply",
                        zarr=str(plan.zarr_path),
                        status="failed",
                        reason="analysis_metadata missing",
                    )
                continue
            if not args.overwrite and "dish_mask" in analysis.attrs:
                print(f"Skipping (dish_mask exists): {plan.zarr_path}")
                skipped += 1
                if logger is not None:
                    logger.log(
                        "dish_mask_apply",
                        zarr=str(plan.zarr_path),
                        status="skipped",
                        reason="dish_mask exists",
                    )
                continue

            raw_video = root.get("raw_video")
            if raw_video is None:
                print(f"Skipping (raw_video missing): {plan.zarr_path}")
                failed += 1
                if logger is not None:
                    logger.log(
                        "dish_mask_apply",
                        zarr=str(plan.zarr_path),
                        status="failed",
                        reason="raw_video missing",
                    )
                continue

            array_name = "images_full" if args.full else "images_ds"
            if array_name not in raw_video:
                if args.full and "images_ds" in raw_video:
                    array_name = "images_ds"
                elif not args.full and "images_full" in raw_video:
                    array_name = "images_full"
                else:
                    print(f"Skipping (no video arrays): {plan.zarr_path}")
                    failed += 1
                    if logger is not None:
                        logger.log(
                            "dish_mask_apply",
                            zarr=str(plan.zarr_path),
                            status="failed",
                            reason="no video arrays",
                        )
                    continue

            video_array = raw_video[array_name]
            max_frames = video_array.shape[0]
            frame_idx = args.frame if args.frame is not None else max_frames // 2
            frame_idx = int(np.clip(frame_idx, 0, max_frames - 1))
            frame = video_array[frame_idx]

            if shape == "rectangle":
                # No parameters to re-detect; copy ROI directly.
                save_mask_to_zarr(
                    str(plan.zarr_path),
                    dish_mask,
                    array_name,
                    frame_idx,
                    params=None,
                    image_shape=frame.shape[:2],
                )
                updated += 1
                print(f"Applied rectangle dish_mask to {plan.zarr_path}")
                if logger is not None:
                    logger.log(
                        "dish_mask_apply",
                        zarr=str(plan.zarr_path),
                        status="updated",
                        shape="rectangle",
                        array_name=array_name,
                        frame_idx=frame_idx,
                    )
                continue

            hough_params = dish_mask.get("hough_params", {})
            param1 = int(hough_params.get("param1", 50))
            param2 = int(hough_params.get("param2", 30))
            radius_adjustment = int(hough_params.get("radius_adjustment", 0))
            circle = _detect_circle(frame, param1, param2, radius_adjustment)
            if circle is None:
                print(f"Failed to detect circle for {plan.zarr_path}")
                failed += 1
                if logger is not None:
                    logger.log(
                        "dish_mask_apply",
                        zarr=str(plan.zarr_path),
                        status="failed",
                        reason="circle detection failed",
                        array_name=array_name,
                        frame_idx=frame_idx,
                    )
                continue

            payload = {
                "shape": "circle",
                "detected_circle": circle,
                "method": "hough_circle_batch",
            }
            params = {
                "param1": param1,
                "param2": param2,
                "radius_adjustment": radius_adjustment,
            }
            save_mask_to_zarr(
                str(plan.zarr_path),
                payload,
                array_name,
                frame_idx,
                params=params,
                image_shape=frame.shape[:2],
            )
            updated += 1
            print(f"Applied dish_mask to {plan.zarr_path}")
            if logger is not None:
                logger.log(
                    "dish_mask_apply",
                    zarr=str(plan.zarr_path),
                    status="updated",
                    shape="circle",
                    array_name=array_name,
                    frame_idx=frame_idx,
                    detected_circle=circle,
                    params=params,
                )
        except Exception as exc:
            failed += 1
            print(f"Failed for {plan.zarr_path}: {exc}")
            if logger is not None:
                logger.log(
                    "dish_mask_apply",
                    zarr=str(plan.zarr_path),
                    status="failed",
                    reason=str(exc),
                )

    print("\nSummary")
    print(f"  chamber: {chamber}")
    print(f"  updated: {updated}")
    print(f"  skipped: {skipped}")
    print(f"  failed: {failed}")
    if logger is not None:
        logger.log(
            "run_end",
            chamber=chamber,
            updated=updated,
            skipped=skipped,
            failed=failed,
            targets=len(plans),
        )
        logger.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
