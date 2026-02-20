#!/usr/bin/env python3
"""
Batch wrapper for eye-mask patch review across many recordings.

Opens visualize_eye_mask_patches per selected Zarr/refined run.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import zarr


REVIEW_STATES = {"approved", "pending", "rejected", "needs_review"}


@dataclass
class ReviewPlan:
    zarr_path: Path
    refined_run: Optional[str]
    review_state: Optional[str]
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
    return sorted(str(name) for name in names)[-1]


def _normalize_review_state(raw: object) -> Optional[str]:
    if not isinstance(raw, dict):
        return None
    state = raw.get("state")
    if state is None:
        return None
    text = str(state).strip().lower()
    if not text:
        return None
    return text


def _infer_zarr_use(root: zarr.Group, zarr_path: Path) -> Optional[str]:
    for key in ("zarr_use", "zarr_purpose"):
        raw = root.attrs.get(key)
        if raw is None:
            continue
        value = str(raw).strip().lower()
        if value in {"analysis", "training"}:
            return value
    name = zarr_path.name.lower()
    if name.endswith("_analysis.zarr"):
        return "analysis"
    if name.endswith("_training.zarr"):
        return "training"
    return None


def _status_matches_filter(review_state: Optional[str], status_filter: str) -> bool:
    if status_filter == "any":
        return True
    if status_filter == "missing":
        return review_state is None
    return review_state == status_filter


def _build_plans(
    roots: List[Path],
    recursive: bool,
    refined_run: Optional[str],
    status_filter: str,
    zarr_use: str,
) -> List[ReviewPlan]:
    plans: List[ReviewPlan] = []
    for zarr_path in _iter_zarr(roots, recursive):
        try:
            root = zarr.open_group(str(zarr_path), mode="r", use_consolidated=False)
        except Exception as exc:
            plans.append(
                ReviewPlan(
                    zarr_path=zarr_path,
                    refined_run=None,
                    review_state=None,
                    status="error",
                    reason=str(exc),
                )
            )
            continue

        observed_use = _infer_zarr_use(root, zarr_path)
        if zarr_use != "any" and observed_use != zarr_use:
            plans.append(
                ReviewPlan(
                    zarr_path=zarr_path,
                    refined_run=None,
                    review_state=None,
                    status="filtered",
                    reason=f"zarr_use={observed_use or 'unknown'}",
                )
            )
            continue

        refined_parent = root.get("refined_eye_masks_runs")
        if refined_parent is None:
            plans.append(
                ReviewPlan(
                    zarr_path=zarr_path,
                    refined_run=None,
                    review_state=None,
                    status="missing",
                    reason="no refined_eye_masks_runs",
                )
            )
            continue

        run_name = refined_run or _latest_run(refined_parent)
        if not run_name:
            plans.append(
                ReviewPlan(
                    zarr_path=zarr_path,
                    refined_run=None,
                    review_state=None,
                    status="missing",
                    reason="no refined eye-mask runs",
                )
            )
            continue
        if run_name not in refined_parent:
            plans.append(
                ReviewPlan(
                    zarr_path=zarr_path,
                    refined_run=run_name,
                    review_state=None,
                    status="missing",
                    reason="refined eye-mask run not found",
                )
            )
            continue

        run_group = refined_parent[run_name]
        review_state = _normalize_review_state(run_group.attrs.get("eye_mask_review_status"))
        if not _status_matches_filter(review_state, status_filter):
            plans.append(
                ReviewPlan(
                    zarr_path=zarr_path,
                    refined_run=run_name,
                    review_state=review_state,
                    status="filtered",
                    reason=f"review_state={review_state or 'missing'}",
                )
            )
            continue

        plans.append(
            ReviewPlan(
                zarr_path=zarr_path,
                refined_run=run_name,
                review_state=review_state,
                status="ok",
            )
        )
    return sorted(plans, key=lambda p: str(p.zarr_path))


def _prompt_continue() -> bool:
    resp = input("Press Enter for next, or type 'q' to quit: ").strip().lower()
    return resp != "q"


def _plan_counts(plans: List[ReviewPlan]) -> dict[str, int]:
    counts = {"ok": 0, "missing": 0, "error": 0, "filtered": 0}
    for plan in plans:
        counts[plan.status] = counts.get(plan.status, 0) + 1
    return counts


def _viewer_cmd(args: argparse.Namespace, plan: ReviewPlan) -> List[str]:
    cmd: List[str] = [
        sys.executable,
        "-m",
        "fisheye.visualization.visualize_eye_mask_patches",
        str(plan.zarr_path),
        "--refined-run",
        str(plan.refined_run),
        "--padding",
        str(args.padding),
        "--scale-percent",
        str(args.scale_percent),
        "--edit-zoom",
        str(args.edit_zoom),
        "--frame-flag-file",
        str(args.frame_flag_file),
        "--review-state",
        str(args.review_state),
        "--review-method",
        str(args.review_method),
        "--review-intended-use",
        str(args.review_intended_use),
    ]
    if args.crop_run:
        cmd.extend(["--crop-run", str(args.crop_run)])
    if args.keypoint_run:
        cmd.extend(["--keypoint-run", str(args.keypoint_run)])
    if args.keypoint_group:
        cmd.extend(["--keypoint-group", str(args.keypoint_group)])
    if args.reviewer:
        cmd.extend(["--reviewer", str(args.reviewer)])
    if args.review_notes:
        cmd.extend(["--review-notes", str(args.review_notes)])
    return cmd


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Batch review eye-mask patches by launching visualize_eye_mask_patches per recording."
    )
    parser.add_argument("paths", nargs="*", type=Path, help="Recording roots or zarr paths to scan.")
    parser.add_argument("--recursive", action="store_true", help="Recursively scan for zarrs.")
    parser.add_argument(
        "--file-list",
        type=Path,
        action="append",
        help="Text file with one zarr path per line (comments with # allowed).",
    )
    parser.add_argument(
        "--zarr-use",
        choices=["analysis", "training", "any"],
        default="any",
        help="Filter zarr archives by use (default: any).",
    )
    parser.add_argument(
        "--status",
        choices=["missing", "approved", "pending", "rejected", "needs_review", "any"],
        default="missing",
        help="Select runs by eye_mask_review_status state (default: missing).",
    )
    parser.add_argument("--refined-run", type=str, help="Specific refined eye-mask run (default: latest per zarr).")
    parser.add_argument("--crop-run", type=str, help="Crop run override for viewer.")
    parser.add_argument("--keypoint-run", type=str, help="Keypoint run override for viewer.")
    parser.add_argument("--keypoint-group", type=str, help="Keypoint group override for viewer.")
    parser.add_argument("--padding", type=int, default=16, help="Patch half-width in pixels (viewer arg).")
    parser.add_argument("--scale-percent", type=int, default=220, help="Viewer scale percent.")
    parser.add_argument("--edit-zoom", type=int, default=4, help="Patch edit zoom for viewer.")
    parser.add_argument("--frame-flag-file", default="eye_mask_frame_flags.json", help="Flag JSON path.")
    parser.add_argument(
        "--review-state",
        default="approved",
        choices=sorted(REVIEW_STATES),
        help="Review state written by viewer when pressing 'a'.",
    )
    parser.add_argument(
        "--review-method",
        default="manual",
        choices=["manual", "algorithmic", "hybrid", "spotcheck"],
        help="Review method written by viewer when pressing 'a'.",
    )
    parser.add_argument(
        "--review-intended-use",
        default="training",
        choices=["training", "full_recording"],
        help="Review intended_use written by viewer when pressing 'a'.",
    )
    parser.add_argument("--reviewer", help="Reviewer name (defaults to $USER in viewer).")
    parser.add_argument("--review-notes", help="Optional review notes for viewer.")
    parser.add_argument("--start", type=int, default=0, help="Start index within selected plans.")
    parser.add_argument("--limit", type=int, help="Maximum number of selected plans to run.")
    parser.add_argument("--list", action="store_true", help="List candidates and exit.")
    parser.add_argument("--no-prompt", action="store_true", help="Do not prompt between recordings.")
    args = parser.parse_args(argv)

    file_list_paths: List[Path] = []
    if args.file_list:
        for path in args.file_list:
            file_list_paths.extend(_load_paths_file(path))

    explicit_paths: List[Path] = []
    if args.paths:
        explicit_paths.extend(args.paths)
    if file_list_paths:
        explicit_paths.extend(file_list_paths)
    if not explicit_paths:
        explicit_paths = [Path("/nvme1/recordings")]

    plans = _build_plans(
        explicit_paths,
        recursive=bool(args.recursive),
        refined_run=args.refined_run,
        status_filter=str(args.status),
        zarr_use=str(args.zarr_use),
    )
    counts = _plan_counts(plans)
    selected = [plan for plan in plans if plan.status == "ok"]

    start = max(0, int(args.start))
    if start:
        selected = selected[start:]
    if args.limit is not None:
        selected = selected[: max(0, int(args.limit))]

    print(
        "Eye-mask batch review plans: "
        f"ok={counts['ok']} missing={counts['missing']} filtered={counts['filtered']} errors={counts['error']}"
    )

    if args.list:
        for plan in selected:
            print(
                f"{plan.zarr_path}\n"
                f"  refined_run={plan.refined_run}\n"
                f"  review_state={plan.review_state or 'missing'}"
            )
        if not selected:
            print("No matching review candidates.")
        return 0

    if not selected:
        print("No matching review candidates.")
        return 0

    failures = 0
    for idx, plan in enumerate(selected, start=1):
        print(f"\n[{idx}/{len(selected)}] Reviewing: {plan.zarr_path}")
        cmd = _viewer_cmd(args, plan)
        rc = subprocess.call(cmd)
        if rc != 0:
            failures += 1
            print(f"Review tool failed (exit={rc}) for {plan.zarr_path}")
        if idx < len(selected) and not args.no_prompt:
            if not _prompt_continue():
                break

    if failures:
        print(f"\nCompleted with {failures} viewer failure(s).")
        return 1
    print("\nCompleted batch eye-mask review.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
