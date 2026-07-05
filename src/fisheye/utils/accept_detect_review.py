#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import zarr

from fisheye.registry.db import Registry, RegistryPaths
from fisheye.shared.refined_detect_resolution import resolve_detect_review_target
from fisheye.shared.zarr_helpers import open_zarr_group_direct
from fisheye.utils.detection_profile import write_detection_profile
from fisheye.registry.inline_refresh import sync_latest_detection_profile_for_zarr


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


def _pick_refined_parent_name(root: zarr.Group) -> Optional[str]:
    if "refined_detect_runs" in root:
        return "refined_detect_runs"
    if "refined_runs" in root:
        return "refined_runs"
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
    profile_result = result.get("detection_profile")
    if isinstance(profile_result, dict):
        print(f"detection_profile: {profile_result.get('status')}")
        print(f"detection_profile_run: {profile_result.get('profile_run') or '—'}")
        registry_sync = profile_result.get("registry_sync")
        if isinstance(registry_sync, dict):
            print(f"detection_profile_registry_sync: {registry_sync.get('status')}")
    approval = result.get("authoritative_approval")
    if isinstance(approval, dict):
        print(f"authoritative_approval: {approval.get('status')} ({approval.get('run')})")
    else:
        print("authoritative_approval: —")
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


def _source_detection_path_for_profile(
    refined_parent_name: str,
    refined_run_name: str,
    refined_run: zarr.Group,
    *,
    target_group: Optional[str],
    resolved_group: Optional[str],
) -> Optional[str]:
    requested = (target_group or "").strip()
    resolved = (resolved_group or "").strip()
    if resolved == "refined" and "instances" in refined_run:
        return f"{refined_parent_name}/{refined_run_name}/instances"
    if resolved == "refined" and "bbox_norm_coords" in refined_run and "frame_indices" in refined_run:
        return f"{refined_parent_name}/{refined_run_name}"
    if requested and requested in refined_run:
        return f"{refined_parent_name}/{refined_run_name}/{requested}"
    if resolved == "manual":
        manual_latest = refined_run.attrs.get("manual_review_latest")
        if manual_latest and str(manual_latest) in refined_run:
            return f"{refined_parent_name}/{refined_run_name}/{manual_latest}"
    if resolved and resolved in refined_run:
        return f"{refined_parent_name}/{refined_run_name}/{resolved}"
    return None


def _resolve_registry_path(requested: Optional[Path]) -> Optional[Path]:
    if requested is not None:
        return requested.expanduser().resolve()
    inferred = RegistryPaths.from_env(Path.cwd()).path.expanduser().resolve()
    return inferred if inferred.exists() else None


def _write_profile_and_sync_registry(
    *,
    root: zarr.Group,
    zarr_path: Path,
    refined_parent_name: str,
    refined_run_name: str,
    refined_run: zarr.Group,
    target_group: Optional[str],
    resolved_group: Optional[str],
    profile_run: Optional[str],
    overwrite_profile: bool,
    registry_path: Optional[Path],
    sync_registry: bool,
    dry_run: bool,
) -> dict[str, object]:
    source_detection_path = _source_detection_path_for_profile(
        refined_parent_name,
        refined_run_name,
        refined_run,
        target_group=target_group,
        resolved_group=resolved_group,
    )
    result: dict[str, object] = {
        "enabled": True,
        "dry_run": bool(dry_run),
        "source_detection_path": source_detection_path,
        "profile_run": profile_run,
        "registry_sync": {"status": "skipped_dry_run" if dry_run else "skipped"},
    }
    if dry_run:
        result["status"] = "would_write"
        return result

    profile_result = write_detection_profile(
        root,
        zarr_path=zarr_path,
        run_name=profile_run,
        overwrite=overwrite_profile,
        source_detection_path=source_detection_path,
        refined_run=refined_run_name if source_detection_path is None else None,
    )
    result.update(
        {
            "status": "updated",
            "profile_run": profile_result.run_name,
            "source_detection_path": profile_result.source_detection_path,
            "source_detection_type": profile_result.source_detection_type,
            "source_detection_content_hash": profile_result.source_detection_content_hash,
        }
    )

    if not sync_registry:
        result["registry_sync"] = {"status": "skipped_disabled"}
        return result

    resolved_registry_path = _resolve_registry_path(registry_path)
    if resolved_registry_path is None:
        result["registry_sync"] = {"status": "skipped_no_registry"}
        return result

    registry = Registry(resolved_registry_path)
    try:
        sync_result = sync_latest_detection_profile_for_zarr(
            registry,
            zarr_path,
            root=root,
            profile_run=profile_result.run_name,
            apply=True,
        )
    finally:
        registry.close()
    sync_result.pop("payload", None)
    sync_result["registry_path"] = str(resolved_registry_path)
    result["registry_sync"] = sync_result
    return result


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
        "--profile-run",
        help="Optional detection-profile run name to write when approving training labels.",
    )
    parser.add_argument(
        "--overwrite-profile",
        action="store_true",
        help="Allow replacing an existing detection-profile run named by --profile-run.",
    )
    parser.add_argument(
        "--skip-detection-profile",
        action="store_true",
        help="Do not materialize analysis/detection_profile_runs when approving for training.",
    )
    parser.add_argument(
        "--registry",
        type=Path,
        help=(
            "Optional registry SQLite path for automatic detection-profile sync. "
            "Defaults to PALETTE_REGISTRY_PATH/config only when that path already exists."
        ),
    )
    parser.add_argument(
        "--no-registry-sync",
        action="store_true",
        help="Write the Zarr detection profile but do not sync registry projection rows.",
    )
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
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON output.")
    args = parser.parse_args(argv)

    try:
        mode = "r" if args.dry_run else "a"
        root = open_zarr_group_direct(args.zarr_path, mode=mode)
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

        authoritative_approval: Optional[dict[str, object]] = None
        if not args.dry_run:
            if args.state == "approved":
                envelope = _approve_refined_detect_authority(
                    zarr_path=args.zarr_path.expanduser().resolve(),
                    refined_run_name=refined_run_name,
                    reviewer=args.reviewer,
                    notes=args.notes,
                )
                authoritative_approval = {
                    "status": envelope.get("status"),
                    "reason_code": envelope.get("reason_code"),
                    "run": envelope.get("run"),
                }
                payload["authoritative_approval"] = authoritative_approval
            refined_run.attrs["detect_review_status"] = payload

        detection_profile_result: dict[str, object] = {"enabled": False, "status": "skipped"}
        should_write_profile = (
            args.state == "approved"
            and intended_use == "training"
            and not bool(args.skip_detection_profile)
        )
        if should_write_profile:
            refined_parent_name = _pick_refined_parent_name(root)
            if refined_parent_name is None:
                raise RuntimeError("No refined detect parent found while writing detection profile.")
            detection_profile_result = _write_profile_and_sync_registry(
                root=root,
                zarr_path=args.zarr_path.expanduser().resolve(),
                refined_parent_name=refined_parent_name,
                refined_run_name=refined_run_name,
                refined_run=refined_run,
                target_group=args.target_group,
                resolved_group=resolved_group,
                profile_run=args.profile_run,
                overwrite_profile=bool(args.overwrite_profile),
                registry_path=args.registry,
                sync_registry=not bool(args.no_registry_sync),
                dry_run=bool(args.dry_run),
            )

        result: dict[str, object] = {
            "zarr_path": str(args.zarr_path),
            "refined_run": refined_run_name,
            "resolved_group": resolved_group,
            "state": args.state,
            "method": args.method,
            "intended_use": intended_use,
            "reviewer": args.reviewer,
            "authoritative_approval": authoritative_approval,
            "dry_run": bool(args.dry_run),
            "payload": payload,
            "detection_profile": detection_profile_result,
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
