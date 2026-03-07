from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from fisheye.detection import detect_keypoints_traditional as keypoint_mod
from fisheye.detection import detect_traditional as detect_mod
from fisheye.shared.crop_image_source import resolve_materialized_crop_run


class _FakeArray:
    def __init__(self, data: np.ndarray) -> None:
        self._data = np.asarray(data)
        self.shape = self._data.shape
        self.dtype = self._data.dtype

    def __getitem__(self, key):
        return self._data[key]


class _FakeGroup:
    def __init__(self, children: dict[str, Any] | None = None) -> None:
        self._children: dict[str, Any] = children or {}
        self.attrs: dict[str, Any] = {}

    def create_group(self, name: str) -> "_FakeGroup":
        child = _FakeGroup()
        self._children[name] = child
        return child

    def create_array(self, name: str, data, **_kwargs) -> _FakeArray:
        array = _FakeArray(np.asarray(data))
        self._children[name] = array
        return array

    def get(self, name: str) -> Any:
        return self._children.get(name)

    def group_keys(self):
        return [key for key, value in self._children.items() if isinstance(value, _FakeGroup)]

    def keys(self):
        return self._children.keys()

    def __contains__(self, key: str) -> bool:
        return key in self._children

    def __getitem__(self, key: str) -> Any:
        if "/" not in key:
            return self._children[key]
        current: Any = self
        for token in key.split("/"):
            if not isinstance(current, _FakeGroup):
                raise KeyError(key)
            current = current._children[token]
        return current


def test_resolve_materialized_crop_run_prefers_latest_materialized() -> None:
    root = _FakeGroup()
    crop_parent = root.create_group("crop_runs")
    crop_parent.attrs["latest"] = "crop_geometry"
    crop_parent.attrs["latest_any"] = "crop_geometry"
    crop_parent.attrs["latest_materialized"] = "crop_materialized"

    crop_geometry = crop_parent.create_group("crop_geometry")
    crop_geometry.attrs["crop_storage_mode"] = "geometry_only"
    crop_geometry.attrs["roi_size"] = [4, 4]
    crop_geometry.create_array("roi_coordinates_full", data=np.array([[0, 0]], dtype=np.int32))
    crop_geometry.create_array("frame_indices", data=np.array([0], dtype=np.int32))

    crop_materialized = crop_parent.create_group("crop_materialized")
    crop_materialized.attrs["crop_storage_mode"] = "materialized"
    crop_materialized.create_array("roi_images", data=np.zeros((1, 4, 4), dtype=np.uint8))
    crop_materialized.create_array("roi_coordinates_full", data=np.array([[0, 0]], dtype=np.int32))
    crop_materialized.create_array("frame_indices", data=np.array([0], dtype=np.int32))

    _parent, _group, run_name = resolve_materialized_crop_run(root)

    assert run_name == "crop_materialized"


def test_resolve_materialized_crop_run_rejects_geometry_only_latest_any() -> None:
    root = _FakeGroup()
    crop_parent = root.create_group("crop_runs")
    crop_parent.attrs["latest"] = "crop_geometry"
    crop_parent.attrs["latest_any"] = "crop_geometry"

    crop_geometry = crop_parent.create_group("crop_geometry")
    crop_geometry.attrs["crop_storage_mode"] = "geometry_only"
    crop_geometry.attrs["roi_size"] = [4, 4]
    crop_geometry.create_array("roi_coordinates_full", data=np.array([[0, 0]], dtype=np.int32))
    crop_geometry.create_array("frame_indices", data=np.array([0], dtype=np.int32))

    with pytest.raises(ValueError, match="geometry-only|materialized crop run"):
        resolve_materialized_crop_run(root)


def test_require_imported_detection_inputs_rejects_missing_images_ds() -> None:
    root = _FakeGroup()
    background_parent = root.create_group("background_runs")
    background_run = background_parent.create_group("background_001")
    background_run.create_array("background_ds", data=np.zeros((4, 4), dtype=np.uint8))

    with pytest.raises(ValueError, match="raw_video/images_ds"):
        detect_mod._require_imported_detection_inputs(root, "background_001")


def test_resolve_traditional_crop_background_inputs_rejects_missing_images_full() -> None:
    root = _FakeGroup()
    crop_parent = root.create_group("crop_runs")
    crop_parent.attrs["latest_materialized"] = "crop_001"
    crop_run = crop_parent.create_group("crop_001")
    crop_run.attrs["crop_storage_mode"] = "materialized"
    crop_run.create_array("roi_images", data=np.zeros((1, 4, 4), dtype=np.uint8))
    crop_run.create_array("roi_coordinates_full", data=np.array([[0, 0]], dtype=np.int32))
    crop_run.create_array("frame_indices", data=np.array([0], dtype=np.int32))

    background_parent = root.create_group("background_runs")
    background_parent.attrs["latest"] = "background_001"
    background_run = background_parent.create_group("background_001")
    background_run.create_array("background_full", data=np.zeros((6, 6), dtype=np.uint8))

    with pytest.raises(ValueError, match="raw_video/images_full"):
        keypoint_mod._resolve_traditional_crop_background_inputs(root)
