from __future__ import annotations

import numpy as np
import pytest

from fisheye.shared.instance_keys import (
    INSTANCE_KEY_ALGORITHM,
    instance_key_attrs,
    mint_detection_instance_keys,
    resolve_recording_identity,
)


def test_mint_detection_instance_keys_is_deterministic_and_reorder_stable() -> None:
    frames = np.asarray([10, 10, 11], dtype=np.int32)
    bboxes = np.asarray(
        [
            [0.5, 0.5, 0.1, 0.1],
            [0.6, 0.5, 0.1, 0.1],
            [0.5, 0.6, 0.1, 0.1],
        ],
        dtype=np.float64,
    )
    classes = np.asarray([0, 0, 1], dtype=np.int32)

    keys = mint_detection_instance_keys(
        recording_identity="rec_a",
        frame_indices=frames,
        bbox_norm_coords=bboxes,
        class_ids=classes,
    )
    order = np.asarray([2, 0, 1], dtype=np.int64)
    reordered_keys = mint_detection_instance_keys(
        recording_identity="rec_a",
        frame_indices=frames[order],
        bbox_norm_coords=bboxes[order],
        class_ids=classes[order],
    )

    assert keys.dtype == np.uint64
    assert keys.shape == (3,)
    np.testing.assert_array_equal(reordered_keys, keys[order])


def test_mint_detection_instance_keys_breaks_exact_duplicate_content_ties() -> None:
    keys = mint_detection_instance_keys(
        recording_identity="rec_a",
        frame_indices=np.asarray([10, 10], dtype=np.int32),
        bbox_norm_coords=np.asarray([[0.5, 0.5, 0.1, 0.1], [0.5, 0.5, 0.1, 0.1]], dtype=np.float64),
        class_ids=np.asarray([0, 0], dtype=np.int32),
    )

    assert keys.shape == (2,)
    assert keys[0] != keys[1]


def test_mint_detection_instance_keys_rejects_misaligned_arrays() -> None:
    with pytest.raises(ValueError, match="bbox_norm_coords row count"):
        mint_detection_instance_keys(
            recording_identity="rec_a",
            frame_indices=np.asarray([1, 2], dtype=np.int32),
            bbox_norm_coords=np.asarray([[0.5, 0.5, 0.1, 0.1]], dtype=np.float64),
        )


def test_detect_producers_share_instance_key_policy_helpers() -> None:
    attrs = {"recording_id": "rec_a", "zarr_use": "analysis"}

    assert resolve_recording_identity(attrs, fallback_path="/tmp/other.zarr") == "rec_a"
    assert instance_key_attrs("rec_a")["instance_key_algorithm"] == INSTANCE_KEY_ALGORITHM
