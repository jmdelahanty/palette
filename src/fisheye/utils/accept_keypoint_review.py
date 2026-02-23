#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import zarr


def _pick_refined_parent(root: zarr.Group) -> Optional[zarr.Group]:
    if "refined_keypoints_runs" in root:
        return root["refined_keypoints_runs"]
    if "keypoints_refined_runs" in root:
        return root["keypoints_refined_runs"]
    return None


def _select_refined_run(parent: zarr.Group, requested: Optional[str]) -> str:
    if requested:
        if requested in parent:
            return requested
        raise RuntimeError(f"Refined keypoint run '{requested}' not found.")
    latest = parent.attrs.get("latest")
    if latest and latest in parent:
        return str(latest)
    try:
        names = list(parent.group_keys())
    except Exception:
        names = list(parent.keys())
    if not names:
        raise RuntimeError("No refined keypoint runs available.")
    return sorted(names)[-1]


def _hash_parameters(params: object) -> Optional[str]:
    if params is None:
        return None
    try:
        payload = json.dumps(params, sort_keys=True, default=str).encode("utf-8")
    except (TypeError, ValueError):
        payload = str(params).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _build_keypoint_signature(attrs: Dict[str, Any]) -> Dict[str, object]:
    params = attrs.get("parameters")
    if not isinstance(params, dict):
        provenance = attrs.get("provenance")
        if isinstance(provenance, dict):
            params = provenance.get("parameters")
    if not isinstance(params, dict):
        params = None

    parameter_source = attrs.get("parameter_source")
    if parameter_source is None and isinstance(params, dict):
        parameter_source = params.get("parameter_source")

    return {
        "signature_version": 1,
        "source_keypoints_run": attrs.get("source_keypoints_run"),
        "source_crop_run": attrs.get("source_crop_run"),
        "source_detect_run": attrs.get("source_detect_run"),
        "source_refined_run": attrs.get("source_refined_run"),
        "parameter_source": parameter_source,
        "parameters_hash": _hash_parameters(params),
    }


def _emit(result: dict[str, object], *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(result, indent=2))
        return
    print(f"zarr_path: {result['zarr_path']}")
    print(f"refined_run: {result['refined_run']}")
    print(f"state: {result['state']}")
    print(f"method: {result['method']}")
    print(f"intended_use: {result['intended_use']}")
    print(f"reviewer: {result.get('reviewer') or '—'}")
    print(f"latest_updated: {result['latest_updated']}")
    print(f"dry_run: {result['dry_run']}")


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Accept/update keypoint review status on refined keypoint runs with "
            "optional strict guardrails."
        )
    )
    parser.add_argument("zarr_path", type=Path, help="Path to the recording .zarr directory.")
    parser.add_argument("--refined-run", help="Refined keypoint run name (default: latest).")
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
        help="Do not update refined_keypoints_runs.attrs['keypoint_review_status_latest'].",
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON output.")
    args = parser.parse_args(argv)

    try:
        mode = "r" if args.dry_run else "a"
        root = zarr.open_group(str(args.zarr_path), mode=mode)
        refined_parent = _pick_refined_parent(root)
        if refined_parent is None:
            raise RuntimeError("No refined_keypoints_runs found in archive.")

        refined_run_name = _select_refined_run(refined_parent, args.refined_run)
        refined_run = refined_parent[refined_run_name]

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
        }
        if args.reviewer:
            payload["reviewer"] = args.reviewer
        if args.notes:
            payload["notes"] = args.notes
        payload = {k: v for k, v in payload.items() if v is not None}

        run_attrs = dict(refined_run.attrs)
        signature = run_attrs.get("keypoint_signature")
        if not isinstance(signature, dict):
            signature = _build_keypoint_signature(run_attrs)

        latest_updated = False
        if not args.dry_run:
            run_attrs["keypoint_review_status"] = payload
            if not isinstance(run_attrs.get("keypoint_signature"), dict):
                run_attrs["keypoint_signature"] = signature
            run_attrs["keypoint_review_signature"] = signature
            refined_run.attrs.put(run_attrs)
            if not args.no_latest:
                parent_attrs = dict(refined_parent.attrs)
                parent_attrs["keypoint_review_status_latest"] = refined_run_name
                refined_parent.attrs.put(parent_attrs)
                latest_updated = True

        result: dict[str, object] = {
            "zarr_path": str(args.zarr_path),
            "refined_run": refined_run_name,
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
