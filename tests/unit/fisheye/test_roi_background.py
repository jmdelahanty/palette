from __future__ import annotations

import numpy as np

from fisheye.shared import roi_background as mod


def test_extract_background_roi_full_pads_out_of_bounds() -> None:
    bg = np.arange(25, dtype=np.uint8).reshape(5, 5)
    roi = mod._extract_background_roi_full(bg, top_left_xy=(-2, -1), roi_shape=(4, 4))

    assert roi.shape == (4, 4)
    assert np.all(roi[0, :] == 0)
    assert np.all(roi[:, 0:2] == 0)
    assert np.array_equal(roi[1:4, 2:4], bg[0:3, 0:2])


def test_extract_background_roi_ds_resizes_matching_full_space_window() -> None:
    bg_ds = np.arange(16, dtype=np.uint8).reshape(4, 4)

    roi = mod._extract_background_roi_ds(
        bg_ds,
        top_left_xy=(2, 2),
        roi_shape=(4, 4),
        full_shape=(8, 8),
    )

    assert roi.shape == (4, 4)
    assert int(roi[0, 0]) == int(bg_ds[1, 1])
    assert int(roi[-1, -1]) == int(bg_ds[2, 2])


def test_prepare_panel_returns_bgr_fixed_size_panel() -> None:
    image = np.full((4, 6), 17, dtype=np.uint8)

    panel = mod._prepare_panel(image, "demo", 32)

    assert panel.shape == (32, 32, 3)
    assert panel.dtype == np.uint8
