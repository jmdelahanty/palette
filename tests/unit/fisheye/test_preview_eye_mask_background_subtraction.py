from __future__ import annotations

from typing import Any

import numpy as np

from fisheye.diagnostics import preview_eye_mask_background_subtraction as mod


class _FakeGroup(dict):
    def __init__(self, *args: Any, attrs: dict[str, Any] | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.attrs = attrs or {}

    def get(self, key: str, default: Any = None) -> Any:
        return super().get(key, default)

    def group_keys(self) -> list[str]:
        keys: list[str] = []
        for key, value in self.items():
            if isinstance(value, _FakeGroup):
                keys.append(str(key))
        return keys


def test_resolve_keypoint_source_prefers_latest_refined() -> None:
    refined_run = _FakeGroup(
        {
            "keypoints_roi": np.zeros((3, 3, 2), dtype=np.float32),
            "refined_success": np.array([True, True, False], dtype=bool),
        }
    )
    raw_run = _FakeGroup(
        {
            "keypoints_roi": np.zeros((3, 3, 2), dtype=np.float32),
            "detection_success": np.array([True, True, True], dtype=bool),
        }
    )
    root = _FakeGroup(
        {
            "refined_keypoints_runs": _FakeGroup({"rk_001": refined_run}, attrs={"latest": "rk_001"}),
            "keypoints_runs": _FakeGroup({"kp_001": raw_run}, attrs={"latest": "kp_001"}),
        }
    )

    source = mod._resolve_keypoint_source(root, explicit=None)
    assert source is not None
    assert source.group_name == "refined_keypoints_runs"
    assert source.run_name == "rk_001"
    assert source.success_name == "refined_success"


def test_resolve_keypoint_source_supports_explicit_raw_run() -> None:
    raw_run = _FakeGroup(
        {
            "keypoints_roi": np.zeros((2, 3, 2), dtype=np.float32),
            "detection_success": np.array([True, False], dtype=bool),
        }
    )
    root = _FakeGroup(
        {
            "keypoints_runs": _FakeGroup({"kp_123": raw_run}, attrs={"latest": "kp_123"}),
        }
    )

    source = mod._resolve_keypoint_source(root, explicit="kp_123")
    assert source is not None
    assert source.group_name == "keypoints_runs"
    assert source.run_name == "kp_123"
    assert source.success_name == "detection_success"


def test_extract_background_roi_full_pads_out_of_bounds() -> None:
    bg = np.arange(25, dtype=np.uint8).reshape(5, 5)
    roi = mod._extract_background_roi_full(bg, top_left_xy=(-2, -1), roi_shape=(4, 4))

    assert roi.shape == (4, 4)
    assert np.all(roi[0, :] == 0)
    assert np.all(roi[:, 0:2] == 0)
    assert np.array_equal(roi[1:4, 2:4], bg[0:3, 0:2])


def test_extract_patch_zero_pads_edges() -> None:
    image = np.arange(25, dtype=np.uint8).reshape(5, 5)
    patch = mod._extract_patch(image, center_xy=(0, 0), half_width=1)

    expected = np.array(
        [
            [0, 0, 0],
            [0, 0, 1],
            [0, 5, 6],
        ],
        dtype=np.uint8,
    )
    assert np.array_equal(patch, expected)
