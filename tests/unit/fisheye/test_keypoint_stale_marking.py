from __future__ import annotations

from pathlib import Path

import zarr

from fisheye.shared.keypoint_stale import (
    mark_downstream_eye_mask_runs_stale,
    resolve_downstream_eye_mask_runs_stale,
)


def test_mark_downstream_eye_mask_runs_stale_marks_matching_runs(tmp_path: Path) -> None:
    zarr_path = tmp_path / "rec.zarr"
    root = zarr.open_group(str(zarr_path), mode="w")

    eye_parent = root.create_group("eye_masks_runs")
    eye_match = eye_parent.create_group("eye_masks_001")
    eye_match.attrs.update(
        {
            "source_keypoints_run": "refined_keypoints_001",
            "source_keypoint_group": "refined_keypoints_runs",
        }
    )
    eye_other = eye_parent.create_group("eye_masks_002")
    eye_other.attrs.update(
        {
            "source_keypoints_run": "refined_keypoints_999",
            "source_keypoint_group": "refined_keypoints_runs",
        }
    )

    refined_parent = root.create_group("refined_eye_masks_runs")
    refined_match = refined_parent.create_group("refined_eye_masks_001")
    refined_match.attrs.update(
        {
            "source_keypoint_run": "refined_keypoints_001",
            # Legacy runs may omit source_keypoint_group; this should still match.
        }
    )

    touched = mark_downstream_eye_mask_runs_stale(
        root,
        source_keypoint_group="refined_keypoints_runs",
        source_keypoints_run="refined_keypoints_001",
        roi_indices=[56],
        frame_indices=[1234],
        reason="keypoint_manual_correction",
    )
    assert touched == 2

    root_live = zarr.open_group(str(zarr_path), mode="r")
    eye_match = root_live["eye_masks_runs"]["eye_masks_001"]
    refined_match = root_live["refined_eye_masks_runs"]["refined_eye_masks_001"]
    eye_other = root_live["eye_masks_runs"]["eye_masks_002"]

    payload_eye = dict(eye_match.attrs["source_keypoint_stale"])
    payload_refined = dict(refined_match.attrs["source_keypoint_stale"])
    assert payload_eye["state"] == "stale"
    assert payload_eye["reason"] == "keypoint_manual_correction"
    assert payload_eye["source_keypoint_group"] == "refined_keypoints_runs"
    assert payload_eye["source_keypoints_run"] == "refined_keypoints_001"
    assert payload_eye["roi_indices"] == [56]
    assert payload_eye["frame_indices"] == [1234]
    assert payload_refined["roi_indices"] == [56]
    assert "source_keypoint_stale" not in eye_other.attrs


def test_mark_downstream_eye_mask_runs_stale_merges_indices_without_duplicates(tmp_path: Path) -> None:
    zarr_path = tmp_path / "rec.zarr"
    root = zarr.open_group(str(zarr_path), mode="w")
    parent = root.create_group("eye_masks_runs")
    run = parent.create_group("eye_masks_001")
    run.attrs.update(
        {
            "source_keypoints_run": "refined_keypoints_001",
            "source_keypoint_group": "refined_keypoints_runs",
        }
    )

    first = mark_downstream_eye_mask_runs_stale(
        root,
        source_keypoint_group="refined_keypoints_runs",
        source_keypoints_run="refined_keypoints_001",
        roi_indices=[10, 11],
        frame_indices=[20],
        reason="keypoint_manual_correction",
    )
    second = mark_downstream_eye_mask_runs_stale(
        root,
        source_keypoint_group="refined_keypoints_runs",
        source_keypoints_run="refined_keypoints_001",
        roi_indices=[11, 12],
        frame_indices=[20, 21],
        reason="keypoint_mark_no_keypoints",
    )

    assert first == 1
    assert second == 1
    root_live = zarr.open_group(str(zarr_path), mode="r")
    run = root_live["eye_masks_runs"]["eye_masks_001"]
    payload = dict(run.attrs["source_keypoint_stale"])
    assert payload["roi_indices"] == [10, 11, 12]
    assert payload["frame_indices"] == [20, 21]
    assert payload["reason"] == "keypoint_mark_no_keypoints"


def test_resolve_downstream_eye_mask_runs_stale_updates_matching_stale_runs(tmp_path: Path) -> None:
    zarr_path = tmp_path / "rec.zarr"
    root = zarr.open_group(str(zarr_path), mode="w")
    parent = root.create_group("refined_eye_masks_runs")
    run = parent.create_group("refined_eye_masks_001")
    run.attrs.update(
        {
            "source_keypoints_run": "refined_keypoints_001",
            "source_keypoint_group": "refined_keypoints_runs",
            "source_keypoint_stale": {
                "state": "stale",
                "timestamp": "2026-02-12T00:00:00+00:00",
                "reason": "keypoint_manual_correction",
                "roi_indices": [5, 6],
                "frame_indices": [100, 101],
            },
        }
    )

    touched = resolve_downstream_eye_mask_runs_stale(
        root,
        source_keypoint_group="refined_keypoints_runs",
        source_keypoints_run="refined_keypoints_001",
        resolution="manual_accept_after_keypoint_nudge_preserve_masks",
        reviewer="tester",
        notes="Masks were curated and preserved.",
    )
    assert touched == 1

    root_live = zarr.open_group(str(zarr_path), mode="r")
    run = root_live["refined_eye_masks_runs"]["refined_eye_masks_001"]
    payload = dict(run.attrs["source_keypoint_stale"])
    assert payload["state"] == "resolved"
    assert payload["reason"] == "keypoint_manual_correction"
    assert payload["stale_timestamp_utc"] == "2026-02-12T00:00:00+00:00"
    assert payload["resolution"] == "manual_accept_after_keypoint_nudge_preserve_masks"
    assert payload["resolved_by"] == "tester"
    assert payload["resolved_notes"] == "Masks were curated and preserved."
    assert payload["roi_indices"] == [5, 6]
    assert payload["frame_indices"] == [100, 101]
    assert "resolved_at_utc" in payload


def test_resolve_downstream_eye_mask_runs_stale_dry_run_does_not_write(tmp_path: Path) -> None:
    zarr_path = tmp_path / "rec.zarr"
    root = zarr.open_group(str(zarr_path), mode="w")
    parent = root.create_group("eye_masks_runs")
    run = parent.create_group("eye_masks_001")
    run.attrs.update(
        {
            "source_keypoints_run": "refined_keypoints_001",
            "source_keypoint_group": "refined_keypoints_runs",
            "source_keypoint_stale": {"state": "stale", "reason": "keypoint_manual_correction"},
        }
    )

    touched = resolve_downstream_eye_mask_runs_stale(
        root,
        source_keypoint_group="refined_keypoints_runs",
        source_keypoints_run="refined_keypoints_001",
        resolution="manual_accept_after_keypoint_nudge_preserve_masks",
        dry_run=True,
    )
    assert touched == 1
    root_live = zarr.open_group(str(zarr_path), mode="r")
    run = root_live["eye_masks_runs"]["eye_masks_001"]
    payload = dict(run.attrs["source_keypoint_stale"])
    assert payload["state"] == "stale"
    assert "resolved_at_utc" not in payload
