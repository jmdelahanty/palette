#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any

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


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Set keypoint_review_status on refined keypoint runs."
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
        default="training",
        choices=["training", "full_recording"],
        help="Intended use (default: training).",
    )
    parser.add_argument("--reviewer", help="Reviewer name or identifier.")
    parser.add_argument("--notes", help="Optional notes.")
    parser.add_argument(
        "--no-latest",
        action="store_true",
        help="Do not update refined_keypoints_runs.attrs['keypoint_review_status_latest'].",
    )
    args = parser.parse_args(argv)

    root = zarr.open_group(str(args.zarr_path), mode="a")
    refined_parent = _pick_refined_parent(root)
    if refined_parent is None:
        raise RuntimeError("No refined_keypoints_runs found in archive.")

    refined_run_name = _select_refined_run(refined_parent, args.refined_run)
    refined_run = refined_parent[refined_run_name]

    payload: dict[str, object] = {
        "state": args.state,
        "method": args.method,
        "intended_use": args.intended_use,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if args.reviewer:
        payload["reviewer"] = args.reviewer
    if args.notes:
        payload["notes"] = args.notes

    payload = {key: value for key, value in payload.items() if value is not None}

    refined_attrs = dict(refined_run.attrs)
    refined_attrs["keypoint_review_status"] = payload

    signature = refined_attrs.get("keypoint_signature")
    if not isinstance(signature, dict):
        signature = _build_keypoint_signature(refined_attrs)
        refined_attrs["keypoint_signature"] = signature
    refined_attrs["keypoint_review_signature"] = signature
    refined_run.attrs.put(refined_attrs)

    if not args.no_latest:
        parent_attrs = dict(refined_parent.attrs)
        parent_attrs["keypoint_review_status_latest"] = refined_run_name
        refined_parent.attrs.put(parent_attrs)

    print(f"Set keypoint_review_status on refined_keypoints_runs/{refined_run_name}")
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
