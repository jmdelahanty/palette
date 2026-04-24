from __future__ import annotations

import numpy as np
import zarr

from fisheye.utils.patch_keypoints_from_crops import _resolve_target_keypoint_indices


def test_resolve_target_keypoint_indices_prefers_refined_row_identity(tmp_path) -> None:
    root = zarr.open_group(str(tmp_path / "targets.zarr"), mode="w")
    keypoints = root.create_group("keypoints_runs/keypoints_001")
    keypoints.create_array("frame_indices", data=np.array([12, 12, 13], dtype=np.int32))
    keypoints.create_array(
        "source_refined_row_ids",
        data=np.array([100, 101, 102], dtype=np.int64),
    )

    target_indices, target_frames, _frame_indices = _resolve_target_keypoint_indices(
        keypoints,
        [],
        flag_entries=[
            {
                "frame_idx": 12,
                "roi_idx": 0,
                "source_refined_row_id": 102,
            }
        ],
    )

    np.testing.assert_array_equal(target_indices, np.array([2], dtype=np.int64))
    np.testing.assert_array_equal(target_frames, np.array([13], dtype=np.int64))
