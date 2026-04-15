#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import zarr

from fisheye.shared.refined_detect_resolution import resolve_detect_review_target


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


def _emit(result: dict[str, object], *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(result, indent=2))
        return
    print(f"zarr_path: {result['zarr_path']}")
    print(f"refined_run: {result['refined_run']}")
    print(f"resolved_group: {result['resolved_group']}")
    print(f"state: {result['state']}")
    print(f"method: {result['method']}")
    print(f"intended_use: {result['intended_use']}")
    print(f"reviewer: {result.get('reviewer') or '—'}")
    print(f"latest_updated: {result['latest_updated']}")
    print(f"dry_run: {result['dry_run']}")


def _resolve_review_target(
    root: zarr.Group,
    refined_run_name: str,
    refined_run: zarr.Group,
    *,
    override_group: Optional[str],
) -> tuple[Optional[str], list[str]]:
    resolution = resolve_detect_review_target(
        root,
        refined_run_name=refined_run_name,
        refined_run=refined_run,
        override_group=override_group,
    )
    return resolution.resolved_group, list(resolution.preference_chain)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Accept/update detect review status on refined detect runs with "
            "optional strict guardrails."
        )
    )
    parser.add_argument("zarr_path", type=Path, help="Path to the recording .zarr directory.")
    parser.add_argument("--refined-run", help="Refined run name (default: latest).")
    parser.add_argument(
        "--target-group",
        help="Explicit detect source to approve (refined/manual/interpolated/filtered/raw or custom sparse group).",
    )
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
        choices=["training", "full_recording"],
        help="Intended use (required in strict mode).",
    )
    parser.add_argument("--reviewer", help="Reviewer name or identifier.")
    parser.add_argument("--notes", help="Optional notes.")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Enable fail-closed review guardrails (reviewer/intended_use requirements).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve target and payload without writing zarr attrs.",
    )
    parser.add_argument(
        "--no-latest",
        action="store_true",
        help="Do not update refined_detect_runs.attrs['detect_review_status_latest'].",
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON output.")
    args = parser.parse_args(argv)

    try:
        mode = "r" if args.dry_run else "a"
        root = zarr.open_group(str(args.zarr_path), mode=mode)
        refined_parent = _pick_refined_parent(root)
        if refined_parent is None:
            raise RuntimeError("No refined_detect_runs found in archive.")

        refined_run_name = _select_refined_run(refined_parent, args.refined_run)
        refined_run = refined_parent[refined_run_name]

        resolved_group, preference_chain = _resolve_review_target(
            root,
            refined_run_name,
            refined_run,
            override_group=args.target_group,
        )

        if args.target_group and resolved_group is None:
            raise RuntimeError(f"Target group '{args.target_group}' could not be resolved.")
        if args.state == "approved" and resolved_group is None:
            raise RuntimeError("Cannot set state=approved when no resolved detect group is available.")

        if args.strict:
            if not args.intended_use:
                raise RuntimeError("--intended-use is required in strict mode.")
            if args.state == "approved" and not args.reviewer:
                raise RuntimeError("--reviewer is required for approved state in strict mode.")
            if args.state == "rejected" and not args.notes:
                raise RuntimeError("--notes is required for rejected state in strict mode.")

        intended_use = args.intended_use or "training"
        timestamp_utc = datetime.now(timezone.utc).isoformat()
        payload: dict[str, object] = {
            "state": args.state,
            "method": args.method,
            "intended_use": intended_use,
            "timestamp_utc": timestamp_utc,
            "timestamp": timestamp_utc,
            "resolved_group": resolved_group,
            "target_group": args.target_group or None,
            "preference_chain": preference_chain,
        }
        if args.reviewer:
            payload["reviewer"] = args.reviewer
        if args.notes:
            payload["notes"] = args.notes
        payload = {k: v for k, v in payload.items() if v is not None}

        latest_updated = False
        if not args.dry_run:
            refined_run.attrs["detect_review_status"] = payload
            if not args.no_latest:
                refined_parent.attrs["detect_review_status_latest"] = refined_run_name
                latest_updated = True

        result: dict[str, object] = {
            "zarr_path": str(args.zarr_path),
            "refined_run": refined_run_name,
            "resolved_group": resolved_group,
            "state": args.state,
            "method": args.method,
            "intended_use": intended_use,
            "reviewer": args.reviewer,
            "latest_updated": latest_updated,
            "dry_run": bool(args.dry_run),
            "payload": payload,
        }
        _emit(result, as_json=bool(args.json))
        return 0
    except Exception as exc:
        if args.json:
            print(json.dumps({"error": str(exc)}, indent=2))
        else:
            print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
