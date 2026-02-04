#!/usr/bin/env python3
"""
Generate file lists for review workflows using simple metadata filters.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Dict

import zarr


@dataclass
class ReviewCandidate:
    zarr_path: Path
    status: str
    reason: Optional[str] = None
    review_state: Optional[str] = None
    review_method: Optional[str] = None
    review_intended_use: Optional[str] = None
    run_name: Optional[str] = None


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
    return sorted(names)[-1]


def _normalize_state(value: object) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, bytes):
        text = value.decode("utf-8", "ignore").strip()
    else:
        text = str(value).strip()
    return text.lower() if text else None


def _extract_review_fields(status: Optional[Dict[str, object]]) -> tuple[Optional[str], Optional[str], Optional[str]]:
    if not status:
        return None, None, None
    state = _normalize_state(status.get("state"))
    method = _normalize_state(status.get("method"))
    intended = _normalize_state(status.get("intended_use"))
    return state, method, intended


def _select_stage_parent(root: zarr.Group, stage: str) -> Optional[zarr.Group]:
    if stage == "crop":
        return root.get("crop_runs")
    if stage == "detect":
        return root.get("refined_detect_runs") or root.get("refined_runs")
    if stage == "keypoints":
        return root.get("refined_keypoints_runs") or root.get("keypoints_refined_runs")
    raise ValueError(f"Unknown stage: {stage}")


def _build_candidates(
    roots: List[Path],
    recursive: bool,
    stage: str,
    run_name: Optional[str],
    include_missing_runs: bool,
) -> List[ReviewCandidate]:
    candidates: List[ReviewCandidate] = []
    for zarr_path in _iter_zarr(roots, recursive):
        try:
            root = zarr.open(str(zarr_path), mode="r")
        except Exception as exc:
            candidates.append(
                ReviewCandidate(
                    zarr_path=zarr_path,
                    status="error",
                    reason=str(exc),
                )
            )
            continue

        parent = _select_stage_parent(root, stage)
        if parent is None:
            if include_missing_runs:
                candidates.append(
                    ReviewCandidate(
                        zarr_path=zarr_path,
                        status="missing",
                        reason=f"no {stage}_runs",
                    )
                )
            continue

        resolved_run = run_name or _latest_run(parent)
        if not resolved_run or resolved_run not in parent:
            if include_missing_runs:
                candidates.append(
                    ReviewCandidate(
                        zarr_path=zarr_path,
                        status="missing",
                        reason=f"{stage} run not found",
                        run_name=resolved_run,
                    )
                )
            continue

        run_group = parent[resolved_run]
        if stage == "crop":
            status_attr = run_group.attrs.get("crop_review_status")
        elif stage == "detect":
            status_attr = run_group.attrs.get("detect_review_status")
        else:
            status_attr = run_group.attrs.get("keypoint_review_status")
        status = status_attr if isinstance(status_attr, dict) else None
        state, method, intended = _extract_review_fields(status)

        candidates.append(
            ReviewCandidate(
                zarr_path=zarr_path,
                status="ok",
                review_state=state,
                review_method=method,
                review_intended_use=intended,
                run_name=resolved_run,
            )
        )
    return sorted(candidates, key=lambda item: str(item.zarr_path))


def _match_filters(
    candidate: ReviewCandidate,
    state_filters: List[str],
    method_filter: Optional[str],
    intended_filter: Optional[str],
) -> bool:
    state = candidate.review_state or "missing"
    if "any" not in state_filters and state not in state_filters:
        return False
    if method_filter and (candidate.review_method or "missing") != method_filter:
        return False
    if intended_filter and (candidate.review_intended_use or "missing") != intended_filter:
        return False
    return True


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate review file lists based on review metadata filters."
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
        "--stage",
        choices=["crop", "detect", "keypoints"],
        default="crop",
        help="Stage to filter review status for (default: crop).",
    )
    parser.add_argument(
        "--run-name",
        help="Specific run name to inspect (default: latest for each Zarr).",
    )
    parser.add_argument(
        "--include-missing-runs",
        action="store_true",
        help="Include recordings missing the selected run/stage.",
    )
    parser.add_argument(
        "--review-state",
        action="append",
        default=["missing"],
        choices=["missing", "approved", "pending", "rejected", "needs_review", "any"],
        help="Filter by review state (repeatable). Default: missing.",
    )
    parser.add_argument(
        "--review-method",
        choices=["manual", "algorithmic", "hybrid", "spotcheck", "missing", "any"],
        default=None,
        help="Filter by review method (default: no filter).",
    )
    parser.add_argument(
        "--review-intended-use",
        choices=["training", "full_recording", "missing", "any"],
        default=None,
        help="Filter by intended use (default: no filter).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output file to write (default: <stage>_review_list.txt in cwd).",
    )
    parser.add_argument(
        "--print",
        action="store_true",
        help="Print matched paths to stdout.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List candidates with status and exit without writing output.",
    )
    args = parser.parse_args(argv)

    file_list_paths: List[Path] = []
    if args.file_list:
        for path in args.file_list:
            file_list_paths.extend(_load_paths_file(path))

    if args.paths:
        roots = list(args.paths) + file_list_paths
    elif file_list_paths:
        roots = file_list_paths
    else:
        env_root = os.environ.get("PALETTE_RECORDINGS_ROOT")
        roots = [Path(env_root)] if env_root else [Path("/nvme1/recordings")]

    candidates = _build_candidates(
        roots=roots,
        recursive=args.recursive,
        stage=args.stage,
        run_name=args.run_name,
        include_missing_runs=args.include_missing_runs,
    )

    if args.list:
        print(f"Candidates: {len(candidates)}")
        for item in candidates:
            state = item.review_state or "missing"
            method = item.review_method or "missing"
            intended = item.review_intended_use or "missing"
            run = item.run_name or "none"
            reason = f" ({item.reason})" if item.reason else ""
            print(f"{item.zarr_path} | {args.stage}={run} | {state}/{method}/{intended}{reason}")
        return 0

    state_filters = [state.lower() for state in args.review_state]
    method_filter = args.review_method
    intended_filter = args.review_intended_use
    if method_filter == "any":
        method_filter = None
    if intended_filter == "any":
        intended_filter = None

    matched = [
        item for item in candidates if _match_filters(item, state_filters, method_filter, intended_filter)
    ]

    output_path = args.output or Path.cwd() / f"{args.stage}_review_list.txt"
    output_path.write_text(
        "\n".join(str(item.zarr_path.resolve()) for item in matched) + ("\n" if matched else ""),
        encoding="utf-8",
    )

    print(f"Wrote {len(matched)} paths to {output_path}")
    if args.print:
        for item in matched:
            print(item.zarr_path.resolve())

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
