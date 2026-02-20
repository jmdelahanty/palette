from __future__ import annotations

from pathlib import Path

import zarr

from fisheye.shared.keypoint_stale import mark_downstream_eye_mask_runs_stale


def test_mark_downstream_eye_mask_runs_stale_marks_matching_runs(tmp_path: Path) -> None:
    root = zarr.open_group(str(tmp_path / "rec.zarr"), mode="w")

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
    root = zarr.open_group(str(tmp_path / "rec.zarr"), mode="w")
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
    payload = dict(run.attrs["source_keypoint_stale"])
    assert payload["roi_indices"] == [10, 11, 12]
    assert payload["frame_indices"] == [20, 21]
    assert payload["reason"] == "keypoint_mark_no_keypoints"

