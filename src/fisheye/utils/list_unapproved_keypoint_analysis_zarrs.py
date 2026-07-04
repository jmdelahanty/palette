#!/usr/bin/env python3
"""List/write analysis zarr archives whose keypoint review is not approved."""

from __future__ import annotations

from fisheye.shared.zarr_discovery import iter_filesystem_zarrs as _iter_zarr
import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional


@dataclass
class UnapprovedKeypointRow:
    zarr_path: str
    zarr_use: str
    latest_refined_run: Optional[str]
    review_state: Optional[str]
    review_intended_use: Optional[str]
    reason: str


def _resolve_roots(paths: Optional[List[Path]]) -> List[Path]:
    if paths:
        return paths
    env_root = os.environ.get("PALETTE_RECORDINGS_ROOT")
    if env_root:
        return [Path(env_root)]
    return [Path("/nvme1/recordings")]


def _read_zarr_attrs(zarr_json_path: Path) -> Dict[str, object]:
    if not zarr_json_path.exists():
        return {}
    try:
        payload = json.loads(zarr_json_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    attrs = payload.get("attributes")
    return attrs if isinstance(attrs, dict) else {}


def _infer_zarr_use(zarr_path: Path, root_attrs: Dict[str, object]) -> str:
    purpose = root_attrs.get("zarr_purpose")
    if purpose is not None:
        value = str(purpose).strip().lower()
        if value in {"analysis", "training"}:
            return value
    name = zarr_path.name.lower()
    if name.endswith("_analysis.zarr"):
        return "analysis"
    if name.endswith("_training.zarr"):
        return "training"
    return "unknown"


def _resolve_refined_parent(zarr_path: Path) -> tuple[Optional[Path], Optional[str]]:
    for name in ("refined_keypoints_runs", "keypoints_refined_runs"):
        parent = zarr_path / name
        if parent.exists():
            return parent, name
    return None, None


def _collect_unapproved_rows(
    roots: List[Path],
    *,
    recursive: bool,
    zarr_use_filter: str,
    approved_state: str,
    required_intended_use: Optional[str],
) -> List[UnapprovedKeypointRow]:
    rows: List[UnapprovedKeypointRow] = []
    approved_norm = approved_state.strip().lower()
    required_use_norm = required_intended_use.strip().lower() if required_intended_use else None

    for zarr_path in _iter_zarr(roots, recursive):
        root_attrs = _read_zarr_attrs(zarr_path / "zarr.json")
        zarr_use = _infer_zarr_use(zarr_path, root_attrs)
        if zarr_use_filter != "any" and zarr_use != zarr_use_filter:
            continue

        parent_path, _ = _resolve_refined_parent(zarr_path)
        latest_run: Optional[str] = None
        review_state: Optional[str] = None
        review_intended_use: Optional[str] = None
        reason = "approved"

        if parent_path is None:
            reason = "no_refined_keypoints_runs"
        else:
            parent_attrs = _read_zarr_attrs(parent_path / "zarr.json")
            latest = parent_attrs.get("latest")
            latest_run = str(latest).strip() if latest is not None else None
            latest_run = latest_run or None
            if latest_run is None:
                reason = "no_latest_refined_run"
            else:
                run_attrs = _read_zarr_attrs(parent_path / latest_run / "zarr.json")
                status = run_attrs.get("keypoint_review_status")
                if not isinstance(status, dict):
                    reason = "no_keypoint_review_status"
                else:
                    raw_state = status.get("state")
                    review_state = str(raw_state).strip() if raw_state is not None else None
                    review_state = review_state or None

                    raw_use = status.get("intended_use")
                    review_intended_use = str(raw_use).strip() if raw_use is not None else None
                    review_intended_use = review_intended_use or None

                    if review_state is None:
                        reason = "review_state_missing"
                    elif review_state.strip().lower() != approved_norm:
                        reason = "review_state_not_approved"
                    elif required_use_norm is not None:
                        if review_intended_use is None:
                            reason = "review_intended_use_missing"
                        elif review_intended_use.strip().lower() != required_use_norm:
                            reason = "review_intended_use_mismatch"

        if reason == "approved":
            continue

        rows.append(
            UnapprovedKeypointRow(
                zarr_path=str(zarr_path),
                zarr_use=zarr_use,
                latest_refined_run=latest_run,
                review_state=review_state,
                review_intended_use=review_intended_use,
                reason=reason,
            )
        )

    rows.sort(key=lambda row: row.zarr_path)
    return rows


def _write_text(paths: List[str], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = "\n".join(paths)
    if payload:
        payload += "\n"
    output_path.write_text(payload, encoding="utf-8")


def _write_tsv(rows: List[UnapprovedKeypointRow], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["zarr_path\tzarr_use\tlatest_refined_run\treview_state\treview_intended_use\treason"]
    for row in rows:
        lines.append(
            "\t".join(
                [
                    row.zarr_path,
                    row.zarr_use,
                    row.latest_refined_run or "<none>",
                    row.review_state or "<none>",
                    row.review_intended_use or "<none>",
                    row.reason,
                ]
            )
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Write a list of analysis zarr archives that do not have approved keypoint review status."
    )
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="Recording roots or zarr paths (default: $PALETTE_RECORDINGS_ROOT or /nvme1/recordings).",
    )
    parser.add_argument("--recursive", action="store_true", help="Recursively scan for zarr archives.")
    parser.add_argument(
        "--zarr-use",
        choices=["analysis", "training", "any"],
        default="analysis",
        help="Filter by zarr use (default: analysis).",
    )
    parser.add_argument(
        "--approved-state",
        default="approved",
        help="Review state to consider approved (default: approved).",
    )
    parser.add_argument(
        "--required-intended-use",
        choices=["training", "full_recording"],
        help="Optional intended_use requirement for approved rows.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/tmp/analysis_without_keypoint_approval.txt"),
        help="Output text file with one zarr path per line.",
    )
    parser.add_argument(
        "--details",
        type=Path,
        help="Optional TSV output with detailed status columns.",
    )
    parser.add_argument("--json", action="store_true", help="Print JSON rows to stdout.")
    args = parser.parse_args(argv)

    roots = _resolve_roots(args.paths)
    rows = _collect_unapproved_rows(
        roots,
        recursive=bool(args.recursive),
        zarr_use_filter=str(args.zarr_use),
        approved_state=str(args.approved_state),
        required_intended_use=args.required_intended_use,
    )

    _write_text([row.zarr_path for row in rows], args.output)
    if args.details is not None:
        _write_tsv(rows, args.details)

    if args.json:
        print(json.dumps([asdict(row) for row in rows], indent=2))
    else:
        print(f"wrote: {args.output}")
        print(f"count: {len(rows)}")
        if args.details is not None:
            print(f"details: {args.details}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
