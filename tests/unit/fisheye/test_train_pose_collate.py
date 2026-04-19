from __future__ import annotations

import numpy as np

from fisheye.training.train_pose import pose_collate_fn


def _sample(*, cls: np.ndarray, keypoints: np.ndarray, num_keypoints: int) -> dict:
    return {
        "img": np.zeros((3, 16, 16), dtype=np.uint8),
        "cls": cls,
        "bboxes": np.zeros((int(cls.size), 4), dtype=np.float32),
        "keypoints": keypoints,
        "num_keypoints": int(num_keypoints),
        "im_file": "sample",
        "ori_shape": (16, 16),
    }


def test_pose_collate_empty_batch_preserves_runtime_keypoint_count() -> None:
    batch = [
        _sample(
            cls=np.zeros((0,), dtype=np.float32),
            keypoints=np.zeros((0, 15), dtype=np.float32),
            num_keypoints=5,
        )
    ]

    out = pose_collate_fn(batch)

    assert tuple(out["keypoints"].shape) == (0, 5, 3)


def test_pose_collate_missing_keypoints_uses_runtime_keypoint_count() -> None:
    batch = [
        _sample(
            cls=np.array([0], dtype=np.float32),
            keypoints=np.zeros((0, 15), dtype=np.float32),
            num_keypoints=5,
        )
    ]

    out = pose_collate_fn(batch)

    assert tuple(out["keypoints"].shape) == (1, 5, 3)
