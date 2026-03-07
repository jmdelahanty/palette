from __future__ import annotations

import sys
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from rich.console import Console

from fisheye.segmentation import eye_segmentation_yolo as mod


class _FakeArray:
    def __init__(self, data: np.ndarray) -> None:
        self._data = np.asarray(data)
        self.shape = self._data.shape


class _FakeCropGroup:
    def __init__(self) -> None:
        self.attrs: dict[str, object] = {}
        self._children: dict[str, object] = {}

    def get(self, name: str):
        return self._children.get(name)

    def __contains__(self, key: str) -> bool:
        return key in self._children

    def __getitem__(self, key: str):
        return self._children[key]


def test_validate_input_row_alignment_allows_geometry_only_crop_group() -> None:
    crop_group = _FakeCropGroup()
    crop_group._children["frame_indices"] = _FakeArray(np.array([0, 1], dtype=np.int32))
    crop_group._children["detection_indices"] = _FakeArray(np.array([0, 1], dtype=np.int32))
    crop_group._children["detection_source"] = _FakeArray(np.array([0, 0], dtype=np.int8))

    mod._validate_input_row_alignment(
        crop_group=crop_group,
        crop_run_name="crop_geometry",
        total_rois=2,
    )


def test_segment_eye_masks_yolo_closes_geometry_only_crop_source_on_no_rois(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    zarr_path = tmp_path / "recording_analysis.zarr"
    model_path = tmp_path / "eye_model.pt"
    zarr_path.mkdir()
    model_path.write_text("", encoding="utf-8")

    class _FakeInnerModel:
        def parameters(self):
            return iter(())

    class _FakeYOLO:
        def __init__(self, _path: str) -> None:
            self.model = _FakeInnerModel()

        def to(self, _device: str) -> "_FakeYOLO":
            return self

    monkeypatch.setitem(
        sys.modules,
        "ultralytics",
        SimpleNamespace(YOLO=_FakeYOLO, __version__="test-ultralytics"),
    )

    fake_root = object()
    monkeypatch.setattr(mod.zarr, "open", lambda *_args, **_kwargs: fake_root)

    fake_crop_group = _FakeCropGroup()

    class _FakeCropSource:
        def __init__(self) -> None:
            self.crop_run_name = "crop_geometry"
            self.crop_group = fake_crop_group
            self.total_rois = 0
            self.roi_shape = (64, 64)
            self.storage_mode = "geometry_only"
            self.frame_source_kind = "raw_video/images_full"
            self.frame_source_path = None
            self.closed = False

        def close(self) -> None:
            self.closed = True

    fake_source = _FakeCropSource()
    open_kwargs: dict[str, object] = {}

    def _fake_open(*_args, **kwargs):
        open_kwargs.update(kwargs)
        return fake_source

    monkeypatch.setattr(mod.CropImageSource, "open", _fake_open)

    console = Console(file=StringIO(), force_terminal=False, color_system=None)

    result = mod.segment_eye_masks_yolo(
        str(zarr_path),
        str(model_path),
        roi_cache_policy="always",
        roi_cache_dir=tmp_path / "roi-cache",
        console=console,
        registry=None,
    )

    assert result == ""
    assert fake_source.closed is True
    assert open_kwargs["roi_cache_policy"] == "always"
    assert open_kwargs["roi_cache_dir"] == tmp_path / "roi-cache"
