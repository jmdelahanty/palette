from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import zarr
from rich.console import Console

from fisheye.utils import keypoint_retry as mod


class _ShapeOnlyArray:
    def __init__(self, shape: tuple[int, ...]) -> None:
        self.shape = shape


class _ShapeOnlyGroup(dict):
    def __init__(self, *, attrs: dict[str, Any] | None = None) -> None:
        super().__init__()
        self.attrs = attrs or {}

    def get(self, key: str, default: Any = None) -> Any:
        return super().get(key, default)


class _FakeBoxes:
    def __init__(self) -> None:
        self.conf = torch.tensor([0.9], dtype=torch.float32)
        self.xyxy = torch.tensor([[0.0, 0.0, 4.0, 4.0]], dtype=torch.float32)


class _FakeKeypoints:
    def __init__(self) -> None:
        self.xy = torch.tensor(
            [[[1.0, 1.0], [2.0, 1.0], [2.5, 2.0]]],
            dtype=torch.float32,
        )
        self.conf = torch.tensor([[0.7, 0.8, 0.9]], dtype=torch.float32)


class _FakeResult:
    def __init__(self) -> None:
        self.boxes = _FakeBoxes()
        self.keypoints = _FakeKeypoints()


class _FakeModelInner:
    def parameters(self):
        return iter(())


class _FakeYOLO:
    def __init__(self, _path: str) -> None:
        self.model = _FakeModelInner()

    def to(self, _device: str) -> "_FakeYOLO":
        return self

    def predict(self, inputs, **_kwargs: Any):
        return [_FakeResult() for _ in inputs]


class _ConfigurableFakeKeypoints:
    def __init__(self, xy: list[list[list[float]]], conf: list[list[float]]) -> None:
        self.xy = torch.tensor(xy, dtype=torch.float32)
        self.conf = torch.tensor(conf, dtype=torch.float32)


class _ConfigurableFakeResult:
    def __init__(self, *, xy: list[list[list[float]]], conf: list[list[float]]) -> None:
        self.boxes = _FakeBoxes()
        self.keypoints = _ConfigurableFakeKeypoints(xy, conf)


class _ConfigurableFakeYOLO:
    def __init__(self, *, xy: list[list[list[float]]], conf: list[list[float]]) -> None:
        self.model = _FakeModelInner()
        self._xy = xy
        self._conf = conf

    def to(self, _device: str) -> "_ConfigurableFakeYOLO":
        return self

    def predict(self, inputs, **_kwargs: Any):
        return [_ConfigurableFakeResult(xy=self._xy, conf=self._conf) for _ in inputs]


def test_resolve_retry_keypoint_contract_uses_source_run_labels() -> None:
    source_run = _ShapeOnlyGroup(
        attrs={
            "keypoint_labels": ["eye_left", "tail_tip", "bladder", "eye_right", "pelvis"],
        }
    )
    retry_group = _ShapeOnlyGroup()
    retry_group["keypoints_roi"] = _ShapeOnlyArray((2, 5, 2))
    retry_group["keypoints_img"] = _ShapeOnlyArray((2, 5, 2))
    retry_group["keypoints_norm"] = _ShapeOnlyArray((2, 5, 2))
    retry_group["keypoint_confidences"] = _ShapeOnlyArray((2, 5))

    labels, count = mod._resolve_retry_keypoint_contract(source_run, retry_group)

    assert count == 5
    assert labels == ("eye_left", "tail_tip", "swim_bladder", "eye_right", "pelvis")


def test_resolve_retry_keypoint_contract_allows_legacy_three_point_retry() -> None:
    source_run = _ShapeOnlyGroup(attrs={})
    retry_group = _ShapeOnlyGroup()
    retry_group["keypoints_roi"] = _ShapeOnlyArray((2, 3, 2))
    retry_group["keypoints_img"] = _ShapeOnlyArray((2, 3, 2))
    retry_group["keypoints_norm"] = _ShapeOnlyArray((2, 3, 2))
    retry_group["keypoint_confidences"] = _ShapeOnlyArray((2, 3))

    labels, count = mod._resolve_retry_keypoint_contract(source_run, retry_group)

    assert count == 3
    assert labels == ("swim_bladder", "eye_left", "eye_right")


def test_retry_failed_keypoints_yolo_uses_geometry_only_latest_any_crop(tmp_path: Path, monkeypatch) -> None:
    zarr_path = tmp_path / "recording_analysis.zarr"
    root = zarr.open_group(str(zarr_path), mode="w")
    root.attrs["width"] = 6
    root.attrs["height"] = 6

    raw_video = root.create_group("raw_video")
    raw_video.create_array(
        "images_full",
        data=np.arange(36, dtype=np.uint8).reshape(1, 6, 6),
        overwrite=True,
    )

    crop_parent = root.create_group("crop_runs")
    crop_parent.attrs["latest"] = "crop_materialized"
    crop_parent.attrs["latest_any"] = "crop_geometry"
    crop_parent.attrs["latest_materialized"] = "crop_materialized"
    crop_parent.create_group("crop_materialized")
    crop_geometry = crop_parent.create_group("crop_geometry")
    crop_geometry.attrs["crop_storage_mode"] = "geometry_only"
    crop_geometry.attrs["roi_size"] = [4, 4]
    crop_geometry.attrs["crop_signature"] = "sig-geometry-001"
    crop_geometry.attrs["crop_revision"] = 5
    crop_geometry.attrs["detect_review_status_ref"] = "refined_detect_runs/refined_detect_001"
    crop_geometry.create_array(
        "roi_coordinates_full",
        data=np.array([[1, 1]], dtype=np.int32),
        overwrite=True,
    )
    crop_geometry.create_array(
        "frame_indices",
        data=np.array([0], dtype=np.int32),
        overwrite=True,
    )

    keypoints_parent = root.create_group("keypoints_runs")
    keypoints_parent.attrs["latest"] = "keypoints_source"
    source_run = keypoints_parent.create_group("keypoints_source")
    source_run.attrs["method"] = "yolo_pose"
    source_run.create_array("keypoints_roi", data=np.full((1, 3, 2), np.nan, dtype=np.float64), overwrite=True)
    source_run.create_array("keypoints_img", data=np.full((1, 3, 2), np.nan, dtype=np.float64), overwrite=True)
    source_run.create_array("keypoints_norm", data=np.full((1, 3, 2), np.nan, dtype=np.float64), overwrite=True)
    source_run.create_array("heading", data=np.full((1,), np.nan, dtype=np.float64), overwrite=True)
    source_run.create_array("confidence", data=np.full((1,), np.nan, dtype=np.float64), overwrite=True)
    source_run.create_array("keypoint_confidences", data=np.full((1, 3), np.nan, dtype=np.float64), overwrite=True)
    source_run.create_array("detection_success", data=np.zeros((1,), dtype=bool), overwrite=True)
    source_run.create_array("heading_finite", data=np.zeros((1,), dtype=bool), overwrite=True)
    source_run.create_array("heading_usable", data=np.zeros((1,), dtype=bool), overwrite=True)
    source_run.create_array("frame_indices", data=np.array([0], dtype=np.int32), overwrite=True)
    source_run.create_array("frame_counts", data=np.array([1, 0, 0, 0, 0], dtype=np.int32), overwrite=True)
    source_run.create_array("n_rois", data=np.array([1], dtype=np.int32), overwrite=True)

    refined_parent = root.create_group("refined_keypoints_runs")
    refined_parent.attrs["latest"] = "refined_001"
    refined_group = refined_parent.create_group("refined_001")
    refined_group.attrs["source_keypoints_run"] = "keypoints_source"
    refined_group.create_array("failure_indices", data=np.array([0], dtype=np.int32), overwrite=True)

    model_path = tmp_path / "pose_model.pt"
    model_path.write_text("", encoding="utf-8")

    monkeypatch.setattr(mod, "YOLO", _FakeYOLO)
    monkeypatch.setattr(mod, "read_reason_labels", lambda _group: None)
    monkeypatch.setattr(
        mod,
        "get_run_group",
        lambda root_arg, *_args, **_kwargs: (
            root_arg["keypoints_runs"].create_group("keypoints_retry_001"),
            "keypoints_retry_001",
        ),
    )

    result = mod.retry_failed_keypoints_yolo(
        zarr_path=str(zarr_path),
        model_path=str(model_path),
        source_keypoints_run="keypoints_source",
        refined_run="refined_001",
        batch_size=1,
        roi_cache_policy="always",
        roi_cache_dir=tmp_path / "roi-cache",
        console=Console(file=None, force_terminal=False, color_system=None),
        registry=None,
    )

    retry_group = zarr.open_group(str(zarr_path), mode="r")["keypoints_runs"]["keypoints_retry_001"]

    assert result["run_name"] == "keypoints_retry_001"
    assert result["updated"] is True
    assert retry_group.attrs["source_crop_run"] == "crop_geometry"
    assert retry_group.attrs["source_crop_storage_mode"] == "geometry_only"
    assert retry_group.attrs["source_crop_signature"] == "sig-geometry-001"
    assert retry_group.attrs["source_crop_revision"] == 5
    assert retry_group.attrs["source_detect_review_status_ref"] == "refined_detect_runs/refined_detect_001"
    assert retry_group.attrs["source_roi_read_mode"] == "temporary_cache"
    assert retry_group.attrs["roi_cache_policy"] == "always"
    assert bool(retry_group.attrs["source_roi_cache_used"]) is True
    assert bool(retry_group["detection_success"][0]) is True
    assert bool(retry_group["re_predicted_used"][0]) is True
    np.testing.assert_array_equal(retry_group["n_keypoints"][:], np.array([3, 0, 0, 0, 0], dtype=np.int32))


def test_retry_failed_keypoints_yolo_uses_source_run_labels_and_heading_contract(
    tmp_path: Path,
    monkeypatch,
) -> None:
    zarr_path = tmp_path / "recording_analysis.zarr"
    root = zarr.open_group(str(zarr_path), mode="w")
    root.attrs["width"] = 8
    root.attrs["height"] = 8

    raw_video = root.create_group("raw_video")
    raw_video.create_array(
        "images_full",
        data=np.arange(64, dtype=np.uint8).reshape(1, 8, 8),
        overwrite=True,
    )

    crop_parent = root.create_group("crop_runs")
    crop_parent.attrs["latest"] = "crop_001"
    crop_group = crop_parent.create_group("crop_001")
    crop_group.create_array(
        "roi_images",
        data=np.zeros((1, 4, 4), dtype=np.uint8),
        overwrite=True,
    )
    crop_group.create_array(
        "roi_coordinates_full",
        data=np.array([[1, 1]], dtype=np.int32),
        overwrite=True,
    )

    keypoints_parent = root.create_group("keypoints_runs")
    keypoints_parent.attrs["latest"] = "keypoints_source"
    source_run = keypoints_parent.create_group("keypoints_source")
    source_run.attrs.update(
        {
            "method": "yolo_pose",
            "source_crop_run": "crop_001",
            "keypoint_labels": ["eye_left", "tail_tip", "swim_bladder", "eye_right", "pelvis"],
            "heading_computation_override": {
                "enabled": True,
                "direction_from": {"op": "keypoint", "label": "swim_bladder"},
                "direction_to": {"op": "midpoint", "labels": ["eye_left", "eye_right"]},
            },
        }
    )
    for name in ("keypoints_roi", "keypoints_img", "keypoints_norm"):
        source_run.create_array(
            name,
            data=np.full((1, 5, 2), np.nan, dtype=np.float64),
            overwrite=True,
        )
    source_run.create_array("heading", data=np.full((1,), np.nan, dtype=np.float64), overwrite=True)
    source_run.create_array("confidence", data=np.full((1,), np.nan, dtype=np.float64), overwrite=True)
    source_run.create_array(
        "keypoint_confidences",
        data=np.full((1, 5), np.nan, dtype=np.float64),
        overwrite=True,
    )
    source_run.create_array("detection_success", data=np.zeros((1,), dtype=bool), overwrite=True)
    source_run.create_array("heading_finite", data=np.zeros((1,), dtype=bool), overwrite=True)
    source_run.create_array("heading_usable", data=np.zeros((1,), dtype=bool), overwrite=True)
    source_run.create_array("frame_indices", data=np.array([0], dtype=np.int32), overwrite=True)
    source_run.create_array("n_rois", data=np.array([1], dtype=np.int32), overwrite=True)

    refined_parent = root.create_group("refined_keypoints_runs")
    refined_parent.attrs["latest"] = "refined_001"
    refined_group = refined_parent.create_group("refined_001")
    refined_group.attrs["source_keypoints_run"] = "keypoints_source"
    refined_group.create_array("failure_indices", data=np.array([0], dtype=np.int32), overwrite=True)

    model_path = tmp_path / "pose_model.pt"
    model_path.write_text("", encoding="utf-8")

    monkeypatch.setattr(
        mod,
        "YOLO",
        lambda _path: _ConfigurableFakeYOLO(
            xy=[[[2.0, 1.0], [0.0, 0.0], [1.0, 1.0], [3.0, 1.0], [4.0, 4.0]]],
            conf=[[0.61, 0.62, 0.63, 0.64, 0.65]],
        ),
    )
    monkeypatch.setattr(mod, "read_reason_labels", lambda _group: None)
    monkeypatch.setattr(
        mod,
        "get_run_group",
        lambda root_arg, *_args, **_kwargs: (
            root_arg["keypoints_runs"].create_group("keypoints_retry_005"),
            "keypoints_retry_005",
        ),
    )

    result = mod.retry_failed_keypoints_yolo(
        zarr_path=str(zarr_path),
        model_path=str(model_path),
        source_keypoints_run="keypoints_source",
        refined_run="refined_001",
        batch_size=1,
        console=Console(file=None, force_terminal=False, color_system=None),
        registry=None,
    )

    retry_group = zarr.open_group(str(zarr_path), mode="r")["keypoints_runs"]["keypoints_retry_005"]
    np.testing.assert_allclose(
        retry_group["keypoint_confidences"][0],
        np.array([0.61, 0.62, 0.63, 0.64, 0.65], dtype=np.float64),
    )
    assert bool(retry_group["heading_finite"][0]) is True
    assert bool(retry_group["heading_usable"][0]) is True
    assert np.isclose(float(retry_group["heading"][0]), 0.0)
    assert result["retry_replaced_count"] == 1
