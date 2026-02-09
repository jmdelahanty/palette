"""Tests for keypoint review status repair utility."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import zarr

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from fisheye.utils.repair_keypoint_review_status import _repair_one


def _seed_archive_with_missing_review(path: Path) -> tuple[str, str]:
    root = zarr.open_group(str(path), mode="w")
    parent = root.create_group("refined_keypoints_runs")
    parent.attrs["latest"] = "refined_pose_001"
    run_name = "refined_pose_001"
    run = parent.create_group(run_name)
    run.attrs["source_keypoints_run"] = "kp_pose_001"
    run.attrs["created_utc"] = "2026-02-08T00:00:00+00:00"
    run.create_array("usable_keypoints", data=np.array([True, False], dtype=np.bool_), chunks=(2,))

    group_name = "refined_keypoints_runs"
    parent_zarr = path / group_name / "zarr.json"
    parent_payload = json.loads(parent_zarr.read_text(encoding="utf-8"))
    consolidated = parent_payload.get("consolidated_metadata")
    if not isinstance(consolidated, dict):
        consolidated = {"kind": "inline", "must_understand": False, "metadata": {}}
        parent_payload["consolidated_metadata"] = consolidated
    consolidated.setdefault("kind", "inline")
    consolidated.setdefault("must_understand", False)
    metadata = consolidated.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
        consolidated["metadata"] = metadata
    parent_payload["consolidated_metadata"]["metadata"][run_name] = {
        "zarr_format": 3,
        "node_type": "group",
        "attributes": {
            "keypoint_review_status": {
                "state": "approved",
                "intended_use": "training",
                "method": "manual",
                "reviewer": "pytest",
            }
        }
    }
    parent_zarr.write_text(json.dumps(parent_payload), encoding="utf-8")
    return group_name, run_name


def test_repair_one_backfills_review_status_from_parent_consolidated_metadata(tmp_path: Path) -> None:
    zarr_path = tmp_path / "sample.zarr"
    _seed_archive_with_missing_review(zarr_path)

    dry_counts = _repair_one(zarr_path, refined_run=None, dry_run=True, no_latest=False)
    assert dry_counts["checked"] == 1
    assert dry_counts["repaired"] + dry_counts["already_set"] == 1
    assert dry_counts["missing"] == 0

    root = zarr.open_group(str(zarr_path), mode="r")
    existing_or_missing = root["refined_keypoints_runs"]["refined_pose_001"].attrs.get("keypoint_review_status")
    assert existing_or_missing is None or isinstance(existing_or_missing, dict)

    apply_counts = _repair_one(zarr_path, refined_run=None, dry_run=False, no_latest=False)
    assert apply_counts["checked"] == 1
    assert apply_counts["repaired"] + apply_counts["already_set"] == 1
    assert apply_counts["missing"] == 0

    root_after = zarr.open_group(str(zarr_path), mode="r")
    review_status = root_after["refined_keypoints_runs"]["refined_pose_001"].attrs.get("keypoint_review_status")
    assert isinstance(review_status, dict)
    assert review_status["state"] == "approved"
    assert review_status["intended_use"] == "training"
