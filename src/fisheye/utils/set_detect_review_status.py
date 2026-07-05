#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import zarr

from fisheye.shared.refined_detect_resolution import resolve_detect_review_target


def _approve_refined_detect_authority(
    *,
    zarr_path: Path,
    refined_run_name: str,
    reviewer: Optional[str],
    notes: Optional[str],
) -> dict[str, object]:
    """Route an approved review through the authoritative approval path (fail-closed)."""

    from fisheye.cli.palette import ApproveRequest, approve

    envelope = approve(
        ApproveRequest(
            recording=zarr_path,
            stage="refined_detect",
            run=refined_run_name,
            approved_by=reviewer,
            note=notes or "detect review sign-off",
            apply=True,
        )
    )
    if str(envelope.get("status") or "").strip().lower() != "ok":
        reason = envelope.get("reason_code") or "UNKNOWN"
        hints = envelope.get("next_hints") or []
        raise RuntimeError(
            "could not set authoritative refined detect run "
            f"{refined_run_name!r} for {zarr_path}: {reason}; hints={hints}"
        )
    return envelope


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
        help="Explicit detect source to approve (refined/manual/interpolated/filtered/raw or custom sparse group).",
    )
    args = parser.parse_args(argv)

    root = zarr.open_group(str(args.zarr_path), mode="a")
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
        raise RuntimeError("Cannot set state=approved when no resolved detect source is available.")

    payload: dict[str, object] = {
        "state": args.state,
        "method": args.method,
        "intended_use": args.intended_use,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "resolved_group": resolved_group,
        "target_group": args.target_group or None,
        "preference_chain": preference_chain,
    }
    if args.reviewer:
        payload["reviewer"] = args.reviewer
    if args.notes:
        payload["notes"] = args.notes

    if args.state == "approved":
        envelope = _approve_refined_detect_authority(
            zarr_path=args.zarr_path.expanduser().resolve(),
            refined_run_name=refined_run_name,
            reviewer=args.reviewer,
            notes=args.notes,
        )
        payload["authoritative_approval"] = {
            "status": envelope.get("status"),
            "reason_code": envelope.get("reason_code"),
            "run": envelope.get("run"),
        }

    # Drop None values to keep attrs clean.
    payload = {key: value for key, value in payload.items() if value is not None}
    refined_run.attrs["detect_review_status"] = payload

    print(f"Set detect_review_status on refined_detect_runs/{refined_run_name}")
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
