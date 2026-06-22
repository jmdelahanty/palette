from inspect import signature
from typing import Any

import pytest

from fisheye.detection import detect_keypoints_yolo as yolo_mod
from fisheye.detection.detect_keypoints_yolo import detect_keypoints_yolo


class _FakeArray:
    def __init__(self, shape: tuple[int, ...]) -> None:
        self.shape = shape


class _FakeGroup(dict):
    def __init__(self, children: dict[str, Any] | None = None, *, attrs: dict[str, Any] | None = None) -> None:
        super().__init__(children or {})
        self.attrs = attrs or {}

    def __getitem__(self, key: str) -> Any:
        if "/" not in key:
            return super().__getitem__(key)
        current: Any = self
        for token in key.split("/"):
            current = dict.__getitem__(current, token)
        return current


def test_detect_keypoints_yolo_accepts_mask_threshold() -> None:
    params = signature(detect_keypoints_yolo).parameters
    assert "mask_threshold" in params
    assert "registry" in params
    assert "roi_cache_policy" in params
    assert "roi_cache_dir" in params
    assert "roi_live_acceleration" in params
    assert "roi_live_gpu_chunk_frames" in params
    assert "input_mode" in params
    assert "profile_timings" in params


def test_resolve_full_image_shape_prefers_raw_video_shape() -> None:
    root = _FakeGroup(
        {
            "raw_video": _FakeGroup({"images_full": _FakeArray((7, 60, 50))}),
        },
        attrs={"width": 10, "height": 20},
    )
    crop_group = _FakeGroup(attrs={"width": 4512, "height": 4512})

    shape, total_frames = yolo_mod._resolve_full_image_shape(root, crop_group)

    assert shape == (60, 50)
    assert total_frames == 7


def test_resolve_full_image_shape_uses_crop_run_dimensions_without_raw_video() -> None:
    root = _FakeGroup(attrs={})
    crop_group = _FakeGroup(attrs={"width": 4512, "height": 4512, "total_frames": 143447})

    shape, total_frames = yolo_mod._resolve_full_image_shape(root, crop_group)

    assert shape == (4512, 4512)
    assert total_frames == 143447


def test_resolve_full_image_shape_uses_root_dimension_aliases_without_raw_video() -> None:
    root = _FakeGroup(attrs={"source_video_width": "4512", "source_video_height": 4512, "n_frames": 12})
    crop_group = _FakeGroup(attrs={})

    shape, total_frames = yolo_mod._resolve_full_image_shape(root, crop_group)

    assert shape == (4512, 4512)
    assert total_frames == 12


def test_resolve_full_image_shape_rejects_missing_dimensions_without_raw_video() -> None:
    root = _FakeGroup(attrs={})
    crop_group = _FakeGroup(attrs={})

    with pytest.raises(ValueError, match="crop-run width/height"):
        yolo_mod._resolve_full_image_shape(root, crop_group)
