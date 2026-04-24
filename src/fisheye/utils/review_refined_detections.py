#!/usr/bin/env python3
"""Batch review refined detections by launching the detection visualizer per Zarr."""

from __future__ import annotations

import argparse
import os
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import zarr

from fisheye.shared.refined_detect_review import (
    DEFAULT_DETECT_GROUP_PREFERENCE,
    resolve_refined_detect_group,
)


@dataclass
class ReviewPlan:
    zarr_path: Path
    refined_run: Optional[str]
    source_label: Optional[str]
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


def _load_paths_file(path: Path) -> List[Path]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        raise
    except Exception as exc:
        raise RuntimeError(f"Failed to read {path}: {exc}") from exc

    items: List[Path] = []
    for line in lines:
        value = line.strip()
        if not value or value.startswith("#"):
            continue
        items.append(Path(value))
    return items


def _normalize_path(path: Path) -> str:
    return str(path.expanduser().resolve(strict=False))


def _load_frame_list(path: Path) -> dict[str, List[int]]:
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise
    except Exception as exc:
        raise RuntimeError(f"Failed to read {path}: {exc}") from exc
    try:
        data = json.loads(raw)
    except Exception as exc:
        raise RuntimeError(f"Frame list must be JSON: {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise RuntimeError(f"Frame list must be a JSON object: {path}")
    parsed: dict[str, List[int]] = {}
    for key, value in data.items():
        if not isinstance(value, list):
            continue
        frames: List[int] = []
        for item in value:
            try:
                frames.append(int(item))
            except (TypeError, ValueError):
                continue
        if not frames:
            continue
        norm_key = _normalize_path(Path(str(key)))
        parsed[norm_key] = sorted(set(frames))
    return parsed


def _get_refined_parent(root: zarr.Group) -> Optional[zarr.Group]:
    if "refined_detect_runs" in root:
        return root["refined_detect_runs"]
    if "refined_runs" in root:
        return root["refined_runs"]
    return None


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


def _legacy_manual_label(run_group: zarr.Group) -> Optional[str]:
    label = run_group.attrs.get("manual_review_latest")
    if label and label in run_group:
        return str(label)
    if "manual" in run_group:
        return "manual"
    return None


def _source_label(run_group: zarr.Group) -> Optional[str]:
    resolution = resolve_refined_detect_group(
        run_group,
        preference=DEFAULT_DETECT_GROUP_PREFERENCE,
    )
    return resolution.label or resolution.group


def _build_plans(roots: List[Path], recursive: bool, only_manual: bool) -> List[ReviewPlan]:
    plans: List[ReviewPlan] = []
    for zarr_path in _iter_zarr(roots, recursive):
        try:
            root = zarr.open(str(zarr_path), mode="r")
        except Exception as exc:
            plans.append(
                ReviewPlan(
                    zarr_path=zarr_path,
                    refined_run=None,
                    source_label=None,
                    status="error",
                    reason=str(exc),
                )
            )
            continue

        refined_parent = _get_refined_parent(root)
        if refined_parent is None:
            plans.append(
                ReviewPlan(
                    zarr_path=zarr_path,
                    refined_run=None,
                    source_label=None,
                    status="missing",
                    reason="no refined_detect_runs",
                )
            )
            continue

        refined_run = _latest_run(refined_parent)
        if not refined_run:
            plans.append(
                ReviewPlan(
                    zarr_path=zarr_path,
                    refined_run=None,
                    source_label=None,
                    status="missing",
                    reason="no refined runs",
                )
            )
            continue

        run_group = refined_parent[refined_run]
        legacy_manual_label = _legacy_manual_label(run_group)
        if only_manual and not legacy_manual_label:
            continue

        plans.append(
            ReviewPlan(
                zarr_path=zarr_path,
                refined_run=refined_run,
                source_label=_source_label(run_group),
                status="ok",
            )
        )
    return sorted(plans, key=lambda p: str(p.zarr_path))


def _prompt_continue() -> bool:
    resp = input("Press Enter for next, or type 'q' to quit: ").strip().lower()
    return resp != "q"


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Batch review refined detections by opening the detection visualizer "
            "for each Zarr."
        )
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
        "--only-manual",
        action="store_true",
        help="Only include recordings that have manual refined detections.",
    )
    parser.add_argument(
        "--manual-review",
        action="store_true",
        help="Open manual bounding box editor (detect_review) instead of visualizer.",
    )
    parser.add_argument(
        "--output-group",
        default="manual",
        help="Output group for manual review (default: manual).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output group when running manual review.",
    )
    parser.add_argument(
        "--review-state",
        default="approved",
        choices=["approved", "pending", "rejected", "needs_review"],
        help="Review state to set when approving in manual review.",
    )
    parser.add_argument(
        "--review-method",
        default="manual",
        choices=["manual", "algorithmic", "hybrid", "spotcheck"],
        help="Review method label (default: manual).",
    )
    parser.add_argument(
        "--review-intended-use",
        default="training",
        choices=["training", "full_recording"],
        help="Intended use label for review status (default: training).",
    )
    parser.add_argument("--reviewer", help="Reviewer name (defaults to $USER).")
    parser.add_argument("--review-notes", help="Optional review notes.")
    parser.add_argument(
        "--frame-list",
        type=Path,
        action="append",
        help="JSON file mapping zarr paths to frame lists (used with --manual-review).",
    )
    parser.add_argument(
        "--include-original",
        action="store_true",
        help="Show original detections alongside refined (default: refined only).",
    )
    parser.add_argument(
        "--refined-run",
        type=str,
        help="Specific refined run name to use (default: latest per Zarr).",
    )
    parser.add_argument(
        "--refined-variant",
        choices=["auto", "interpolated", "filtered", "manual"],
        default="auto",
        help="Refined group to visualize (default: auto prefers manual if available).",
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        default=0,
        help="Frame index to start on in the visualizer.",
    )
    parser.add_argument(
        "--video",
        type=str,
        help="Optional path to source video (when raw frames are missing).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Directory to save snapshots (passed through to visualizer).",
    )
    parser.add_argument(
        "--flag-file",
        type=str,
        default="retune_flags.txt",
        help="File to append flagged zarr paths (default: retune_flags.txt in cwd).",
    )
    parser.add_argument(
        "--frame-flag-file",
        type=str,
        default="retune_frame_flags.json",
        help="JSON file to append flagged frames (default: retune_frame_flags.json in cwd).",
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
    args = parser.parse_args(argv)

    file_list_paths: List[Path] = []
    if args.file_list:
        for path in args.file_list:
            file_list_paths.extend(_load_paths_file(path))

    frame_map: dict[str, List[int]] = {}
    if args.frame_list:
        for path in args.frame_list:
            data = _load_frame_list(path)
            for key, frames in data.items():
                if key in frame_map:
                    frame_map[key] = sorted(set(frame_map[key]).union(frames))
                else:
                    frame_map[key] = frames

    frame_list_roots = [Path(key) for key in frame_map.keys()]

    if args.paths:
        roots = list(args.paths) + file_list_paths + frame_list_roots
    elif file_list_paths or frame_list_roots:
        roots = file_list_paths + frame_list_roots
    else:
        env_root = os.environ.get("PALETTE_RECORDINGS_ROOT")
        roots = [Path(env_root)] if env_root else [Path("/nvme1/recordings")]

    if args.manual_review and not frame_map:
        print("--manual-review requires --frame-list (JSON mapping of zarr paths to frames).")
        return 1

    manual_overwrite = bool(args.overwrite or args.manual_review)
    plans = _build_plans(roots, args.recursive, args.only_manual)

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
    if args.manual_review:
        print("Mode: manual review (detect_review)")
        if manual_overwrite:
            print(f"Manual review will overwrite output group '{args.output_group}'.")
    if args.flag_file:
        print(f"Flag file: {Path(args.flag_file).expanduser().resolve(strict=False)}")
    if args.frame_flag_file:
        print(f"Frame flag file: {Path(args.frame_flag_file).expanduser().resolve(strict=False)}")
    for idx, plan in enumerate(plans):
        marker = "→" if start <= idx < end else " "
        source = plan.source_label or "none"
        refined = plan.refined_run or "none"
        status = plan.status
        reason = f" ({plan.reason})" if plan.reason else ""
        print(
            f"{marker} [{idx:03d}] {plan.zarr_path} | refined={refined} | "
            f"source={source} | {status}{reason}"
        )

    if args.list:
        return 0

    for idx in range(start, end):
        plan = plans[idx]
        if plan.status != "ok":
            print(f"\n[{idx + 1}/{end}] Skipping {plan.zarr_path} ({plan.status})")
            continue
        if args.manual_review:
            normalized = _normalize_path(plan.zarr_path)
            frames = frame_map.get(normalized, [])
            if not frames:
                print(f"\n[{idx + 1}/{end}] Skipping {plan.zarr_path} (no flagged frames)")
                continue
            print(f"\n[{idx + 1}/{end}] Opening manual review for {plan.zarr_path} ({len(frames)} frames)")
            cmd = [
                sys.executable,
                "-m",
                "fisheye.tune.detect_review",
                str(plan.zarr_path),
                "--frames",
                ",".join(str(frame) for frame in frames),
                "--output-group",
                args.output_group,
                "--review-state",
                args.review_state,
                "--review-method",
                args.review_method,
                "--review-intended-use",
                args.review_intended_use,
            ]
            if args.reviewer:
                cmd.extend(["--reviewer", args.reviewer])
            if args.review_notes:
                cmd.extend(["--review-notes", args.review_notes])
            if manual_overwrite:
                cmd.append("--overwrite")
            if plan.refined_run:
                cmd.extend(["--refined-run", plan.refined_run])
            if args.refined_variant in ("filtered", "interpolated"):
                cmd.extend(["--variant", args.refined_variant])
            subprocess.run(cmd, check=False)
        else:
            print(f"\n[{idx + 1}/{end}] Opening visualizer for {plan.zarr_path}")
            cmd = [
                sys.executable,
                "-m",
                "fisheye.visualization.detection_visualizer",
                str(plan.zarr_path),
                "--show-refined",
                "--refined-variant",
                args.refined_variant,
                "--start-frame",
                str(args.start_frame),
            ]
            if not args.include_original:
                cmd.append("--refined-only")
            if args.flag_file:
                cmd.extend(["--flag-file", args.flag_file])
            if args.frame_flag_file:
                cmd.extend(["--frame-flag-file", args.frame_flag_file])
            if args.refined_run:
                cmd.extend(["--refined-run", args.refined_run])
            if args.video:
                cmd.extend(["--video", args.video])
            if args.output_dir:
                cmd.extend(["--output-dir", args.output_dir])
            subprocess.run(cmd, check=False)
        if idx < end - 1:
            if not _prompt_continue():
                break

    print("Review complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
