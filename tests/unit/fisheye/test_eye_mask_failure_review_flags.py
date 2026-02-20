from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from fisheye.tune import eye_mask_failure_review as mod


def test_collect_flagged_roi_indices_merges_roi_and_frame_entries(tmp_path: Path) -> None:
    flag_path = tmp_path / "eye_mask_frame_flags.json"
    zarr_path = "/tmp/recording_training.zarr"
    flag_path.write_text(
        json.dumps(
            {
                zarr_path: [
                    {"frame_idx": 10, "roi_idx": 1},
                    {"frame_idx": 11, "roi_idx": None},
                    10,
                ]
            }
        ),
        encoding="utf-8",
    )

    frame_indices = np.array([9, 10, 10, 11], dtype=np.int64)
    out = mod._collect_flagged_roi_indices(
        flag_path=flag_path,
        zarr_path=zarr_path,
        total_rois=4,
        frame_indices=frame_indices,
    )

    np.testing.assert_array_equal(out, np.array([1, 2, 3], dtype=np.int32))


def test_collect_flagged_roi_indices_matches_resolved_zarr_key(tmp_path: Path) -> None:
    zarr_path = tmp_path / "recording_training.zarr"
    resolved_key = str(zarr_path.resolve())
    flag_path = tmp_path / "eye_mask_frame_flags.json"
    flag_path.write_text(
        json.dumps(
            {
                resolved_key: [
                    {"frame_idx": 5, "roi_idx": 0},
                ]
            }
        ),
        encoding="utf-8",
    )

    out = mod._collect_flagged_roi_indices(
        flag_path=flag_path,
        zarr_path=str(zarr_path),
        total_rois=3,
        frame_indices=np.array([5, 6, 7], dtype=np.int64),
    )

    np.testing.assert_array_equal(out, np.array([0], dtype=np.int32))


def test_next_review_position_wraps_when_current_removed_from_end() -> None:
    failures = np.array([100, 200], dtype=np.int32)
    out = mod._next_review_position(
        failures,
        previous_roi_idx=300,
        previous_pos=2,
    )
    assert out == 0


def test_next_review_position_keeps_forward_order_when_current_removed() -> None:
    failures = np.array([10, 30, 40], dtype=np.int32)
    out = mod._next_review_position(
        failures,
        previous_roi_idx=20,
        previous_pos=1,
    )
    assert out == 1
