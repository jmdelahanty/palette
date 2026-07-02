from __future__ import annotations

import numpy as np

from fisheye.shared import mask_tuning_helpers as mod


def test_compute_heading_deg_uses_canonical_keypoint_order_by_default() -> None:
    keypoints = np.array(
        [
            [0.0, 0.0],
            [1.0, -1.0],
            [1.0, 1.0],
        ],
        dtype=np.float32,
    )

    assert mod.compute_heading_deg(keypoints) == 0.0


def test_apply_sobel_filter_preserves_patch_when_strength_zero() -> None:
    patch = np.arange(16, dtype=np.uint8).reshape(4, 4)

    filtered, sobel = mod.apply_sobel_filter(patch, 0.0)

    assert filtered is patch
    assert sobel is None


def test_select_region_chooses_component_nearest_center() -> None:
    mask = np.zeros((20, 20), dtype=bool)
    mask[2:6, 2:6] = True
    mask[12:18, 12:18] = True

    selected = mod.select_region(
        mask,
        center=(15.0, 15.0),
        min_area=4,
        max_area=None,
        min_circularity=None,
        closing=0,
        opening=0,
    )

    assert selected is not None
    assert int(np.count_nonzero(selected[12:18, 12:18])) == 36
    assert int(np.count_nonzero(selected[2:6, 2:6])) == 0
