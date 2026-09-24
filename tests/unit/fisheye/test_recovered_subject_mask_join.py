import numpy as np
import pytest

from fisheye.training.recovered_subject_mask_source import (
    join_mask_rows,
    verify_crop_pair,
)


def test_join_uses_identity_and_preserves_source_order_with_duplicate_pixels():
    mask, pose = join_mask_rows(
        "a",
        ["b", "a:zmask"],
        np.array([1, 0, 1]),
        np.array([5, 5, 2]),
        np.array([2, 5, 8]),
    )
    assert mask.tolist() == [0, 2]
    assert pose.tolist() == [1, 0]
    image = np.zeros((8, 8), np.uint8)
    verify_crop_pair(
        image, image.copy(), np.array([0.1] * 4, np.float32), np.array([0.1] * 4)
    )


@pytest.mark.parametrize(
    "frames,pose", [([5, 5], [5, 8]), ([5, 2], [5, 5]), ([5, 2], [5, 8])]
)
def test_join_rejects_ambiguous_and_missing_rows(frames, pose):
    with pytest.raises(ValueError):
        join_mask_rows("a", ["a"], np.array([0, 0]), np.array(frames), np.array(pose))


def test_pixel_and_box_mismatch_refuse_even_if_frame_matches():
    image = np.zeros((8, 8), np.uint8)
    with pytest.raises(ValueError, match="pixels"):
        verify_crop_pair(image, image + 1, np.zeros(4), np.zeros(4))
    with pytest.raises(ValueError, match="box"):
        verify_crop_pair(image, image, np.zeros(4), np.ones(4) * 1e-6)
