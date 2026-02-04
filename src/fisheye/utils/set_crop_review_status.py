#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any

import zarr


def _select_crop_run(parent: zarr.Group, requested: Optional[str]) -> str:
    if requested:
        if requested in parent:
            return requested
        raise RuntimeError(f"Crop run '{requested}' not found.")
    latest = parent.attrs.get("latest")
    if latest and latest in parent:
        return str(latest)
    try:
        names = list(parent.group_keys())
    except Exception:
        names = list(parent.keys())
    if not names:
        raise RuntimeError("No crop runs available.")
    return sorted(names)[-1]


def _hash_parameters(params: object) -> Optional[str]:
    if params is None:
        return None
    try:
        payload = json.dumps(params, sort_keys=True, default=str).encode("utf-8")
    except (TypeError, ValueError):
        payload = str(params).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _build_crop_signature(attrs: Dict[str, Any]) -> Dict[str, object]:
    return {
        "signature_version": 1,
        "detection_source_path": attrs.get("detection_source_path"),
        "detection_source_type": attrs.get("detection_source_type"),
        "detection_preferred_policy": attrs.get("detection_preferred_policy"),
        "source_detect_run": attrs.get("source_detect_run"),
        "source_refined_run": attrs.get("source_refined_run"),
        "roi_size": attrs.get("roi_size"),
        "parameter_source": attrs.get("parameter_source"),
        "parameters_hash": _hash_parameters(attrs.get("parameters")),
    }


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Set crop_review_status on crop runs.")
    parser.add_argument("zarr_path", type=Path, help="Path to the recording .zarr directory.")
    parser.add_argument("--crop-run", help="Crop run name (default: latest).")
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
        help="Do not update crop_runs.attrs['crop_review_status_latest'].",
    )
    args = parser.parse_args(argv)

    root = zarr.open_group(str(args.zarr_path), mode="a")
    if "crop_runs" not in root:
        raise RuntimeError("No crop_runs found in archive.")

    crop_parent = root["crop_runs"]
    crop_run_name = _select_crop_run(crop_parent, args.crop_run)
    crop_run = crop_parent[crop_run_name]

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

    crop_run.attrs["crop_review_status"] = payload

    signature = crop_run.attrs.get("crop_signature")
    if not isinstance(signature, dict):
        signature = _build_crop_signature(crop_run.attrs)
        crop_run.attrs["crop_signature"] = signature
    crop_run.attrs["crop_review_signature"] = signature

    if not args.no_latest:
        crop_parent.attrs["crop_review_status_latest"] = crop_run_name

    print(f"Set crop_review_status on crop_runs/{crop_run_name}")
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
