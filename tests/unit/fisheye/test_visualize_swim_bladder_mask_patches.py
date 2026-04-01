from __future__ import annotations

import numpy as np
import pytest

from fisheye.visualization.visualize_swim_bladder_mask_patches import (
    _build_view,
    _extract_patch_bounds,
    _mouse_modifier_state,
    _resolve_erase_mode,
    _resolve_swim_bladder_center_with_source,
    parse_args,
)


def test_extract_patch_bounds_clamps_at_edges() -> None:
    x0, x1, y0, y1 = _extract_patch_bounds((20, 20), (1.2, 2.8), padding=5)
    assert (x0, x1, y0, y1) == (0, 7, 0, 9)


def test_resolve_swim_bladder_center_prefers_label_then_mask_centroid_then_roi_center() -> None:
    keypoints = np.array(
        [
            [10.0, 11.0],  # eye_left
            [20.0, 21.0],  # eye_right
            [3.0, 4.0],    # swim_bladder
        ],
        dtype=np.float32,
    )
    labels = ["eye_left", "eye_right", "swim_bladder"]
    mask = np.zeros((12, 12), dtype=np.uint8)
    mask[6, 7] = 1

    center_keypoint, source_keypoint = _resolve_swim_bladder_center_with_source(keypoints, labels, mask, (12, 12))
    assert center_keypoint == (3.0, 4.0)
    assert source_keypoint == "keypoint"

    keypoints_missing = np.full((3, 2), np.nan, dtype=np.float32)
    mask_centroid = np.zeros((12, 12), dtype=np.uint8)
    mask_centroid[2, 4] = 1
    mask_centroid[4, 6] = 1
    center_mask, source_mask = _resolve_swim_bladder_center_with_source(
        keypoints_missing,
        labels,
        mask_centroid,
        (12, 12),
    )
    assert center_mask == (5.0, 3.0)
    assert source_mask == "mask_centroid"

    empty_mask = np.zeros((10, 14), dtype=np.uint8)
    center_roi, source_roi = _resolve_swim_bladder_center_with_source(
        keypoints_missing,
        labels,
        empty_mask,
        (10, 14),
    )
    assert center_roi == (7.0, 5.0)
    assert source_roi == "roi_center"


def test_parse_args_sets_defaults() -> None:
    args = parse_args(["/tmp/example.zarr"])
    assert args.padding == 18
    assert args.scale_percent == 220
    assert args.edit_zoom == 8
    assert args.review_state == "approved"
    assert args.review_method == "manual"
    assert args.review_intended_use == "training"


def test_mouse_modifier_state_decodes_ctrl_shift_lmb() -> None:
    cv2 = pytest.importorskip("cv2")
    flags = int(cv2.EVENT_FLAG_CTRLKEY | cv2.EVENT_FLAG_SHIFTKEY | cv2.EVENT_FLAG_LBUTTON)
    ctrl, shift, lmb = _mouse_modifier_state(flags)
    assert ctrl is True
    assert shift is True
    assert lmb is True


def test_resolve_erase_mode_allows_shift_temporary_inverse() -> None:
    assert _resolve_erase_mode(False, False) is False
    assert _resolve_erase_mode(True, False) is True
    assert _resolve_erase_mode(False, True) is True
    assert _resolve_erase_mode(True, True) is False


def test_build_view_returns_edit_meta() -> None:
    roi = np.zeros((16, 16), dtype=np.uint8)
    roi[4:12, 4:12] = 80
    source_mask = np.zeros((16, 16), dtype=np.uint8)
    source_mask[7:9, 7:9] = 1
    current_mask = np.zeros((16, 16), dtype=np.uint8)
    current_mask[6:10, 6:10] = 1

    canvas, edit_meta = _build_view(
        roi,
        source_mask,
        current_mask,
        center_xy=(8.0, 8.0),
        center_source="keypoint",
        padding=4,
        edit_zoom=4,
        brush_radius=3,
        cursor_patch_xy=(2, 2),
    )

    assert canvas.ndim == 3
    assert canvas.shape[2] == 3
    assert edit_meta["patch_x0"] == 4
    assert edit_meta["patch_y0"] == 4
    assert edit_meta["patch_w"] == 9
    assert edit_meta["patch_h"] == 9
    assert edit_meta["zoom"] == 4
