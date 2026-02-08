#!/usr/bin/env python3
"""Temporary helper to inspect refined keypoint review attrs on a Zarr archive.

Prints:
1) Live `group.attrs` value
2) `refined_keypoints_runs/<run>/zarr.json` attributes value
3) `refined_keypoints_runs/zarr.json` consolidated metadata value
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import zarr


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zarr", required=True, type=Path, help="Path to recording .zarr")
    parser.add_argument(
        "--run",
        required=True,
        help="Refined keypoint run name (inside refined_keypoints_runs/<run>)",
    )
    args = parser.parse_args()

    root = zarr.open_group(str(args.zarr), mode="r")
    group = root["refined_keypoints_runs"][args.run]
    value = group.attrs.get("keypoint_review_status")

    run_json_path = args.zarr / "refined_keypoints_runs" / args.run / "zarr.json"
    parent_json_path = args.zarr / "refined_keypoints_runs" / "zarr.json"
    run_payload = _load_json(run_json_path)
    parent_payload = _load_json(parent_json_path)

    run_json_value = None
    if isinstance(run_payload.get("attributes"), dict):
        run_json_value = run_payload["attributes"].get("keypoint_review_status")

    parent_json_value = None
    consolidated = parent_payload.get("consolidated_metadata")
    if isinstance(consolidated, dict):
        metadata = consolidated.get("metadata")
        if isinstance(metadata, dict):
            entry = metadata.get(args.run)
            if isinstance(entry, dict):
                attrs = entry.get("attributes")
                if isinstance(attrs, dict):
                    parent_json_value = attrs.get("keypoint_review_status")

    print("== live attrs ==")
    print("type:", type(value))
    print("value:", repr(value))
    print("has_key:", "keypoint_review_status" in list(group.attrs.keys()))

    print("\n== run zarr.json ==")
    print("path:", run_json_path)
    print("value:", repr(run_json_value))

    print("\n== parent consolidated metadata ==")
    print("path:", parent_json_path)
    print("value:", repr(parent_json_value))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
