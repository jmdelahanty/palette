from __future__ import annotations

import numpy as np
import pytest

from fisheye.training import zarr_yolo_dataset_loader as mod


class _FakeArray:
    def __init__(self, data: np.ndarray) -> None:
        self._data = np.asarray(data)
        self.shape = self._data.shape
        self.chunks = None

    def __getitem__(self, item):
        return self._data[item]


class _FakeGroup:
    def __init__(self, *, attrs: dict[str, object] | None = None, members: dict[str, object] | None = None) -> None:
        self.attrs = attrs or {}
        self._members = members or {}

    def __getitem__(self, key: str):
        current: object = self
        for part in str(key).split("/"):
            if not isinstance(current, _FakeGroup) or part not in current._members:
                raise KeyError(key)
            current = current._members[part]
        return current

    def __contains__(self, key: str) -> bool:
        return key in self._members

    def get(self, key: str, default=None):
        try:
            return self[key]
        except KeyError:
            return default


class _FakeIndexManager:
    metadata_rows: list[mod.DatasetMetadata] = []
    split_rows: list[tuple[str, int]] = []

    def __init__(self, config: mod.ZarrDatasetConfig) -> None:
        self.config = config
        self.metadata_list = list(self.metadata_rows)

    def get_split_indices(self):
        return list(self.split_rows), []


def _fake_pose_root() -> _FakeGroup:
    crop = _FakeGroup(
        attrs={},
        members={
            "roi_images": _FakeArray(np.zeros((1, 32, 32), dtype=np.uint8)),
            "bbox_norm_coords": _FakeArray(np.array([[0.5, 0.5, 0.2, 0.2]], dtype=np.float32)),
        },
    )
    kp = _FakeGroup(
        attrs={"keypoint_labels": ["eye_left", "tail_tip", "swim_bladder"]},
        members={
            "keypoints_roi": _FakeArray(
                np.array([[[10.0, 10.0], [12.0, 12.0], [14.0, 14.0]]], dtype=np.float32)
            ),
            "detection_success": _FakeArray(np.array([True], dtype=np.bool_)),
        },
    )
    return _FakeGroup(
        attrs={},
        members={
            "crop_runs": _FakeGroup(
                attrs={"latest": "crop_001"},
                members={"crop_001": crop},
            ),
            "keypoints_runs": _FakeGroup(
                attrs={"latest": "kp_001"},
                members={"kp_001": kp},
            ),
        },
    )


def test_pose_loader_uses_metadata_label_signature(monkeypatch) -> None:
    path = "/tmp/pose_a.zarr"
    _FakeIndexManager.metadata_rows = [
        mod.DatasetMetadata(
            path=path,
            name="pose_a",
            total_frames=1,
            valid_frames=1,
            column_names=["eye_left", "tail_tip", "swim_bladder"],
            keypoint_run="kp_001",
            bbox_array_path="crop_runs/crop_001/bbox_norm_coords",
        )
    ]
    _FakeIndexManager.split_rows = [(path, 0)]

    monkeypatch.setattr(mod, "GlobalIndexManager", _FakeIndexManager)
    monkeypatch.setattr(mod.zarr, "open", lambda *args, **kwargs: _fake_pose_root())

    cfg = mod.ZarrDatasetConfig(
        datasets={
            "pose": {
                "zarr_path": path,
                "source_type": "filtered",
                "input_format": "gray",
                "split": {"train": 1.0, "val": 0.0},
            }
        },
        task="pose",
        random_seed=11,
        sampling_strategy="proportional",
    )

    ds = mod.ZarrYOLODataset(cfg, mode="train")

    assert ds.keypoint_labels == ["eye_left", "tail_tip", "swim_bladder"]


def test_pose_loader_rejects_mixed_metadata_label_signatures(monkeypatch) -> None:
    _FakeIndexManager.metadata_rows = [
        mod.DatasetMetadata(
            path="/tmp/pose_a.zarr",
            name="pose_a",
            total_frames=1,
            valid_frames=1,
            column_names=["swim_bladder", "eye_left", "eye_right"],
        ),
        mod.DatasetMetadata(
            path="/tmp/pose_b.zarr",
            name="pose_b",
            total_frames=1,
            valid_frames=1,
            column_names=["tail_tip", "eye_left", "eye_right"],
        ),
    ]
    _FakeIndexManager.split_rows = [("/tmp/pose_a.zarr", 0)]

    monkeypatch.setattr(mod, "GlobalIndexManager", _FakeIndexManager)

    cfg = mod.ZarrDatasetConfig(
        datasets={
            "pose_a": {
                "zarr_path": "/tmp/pose_a.zarr",
                "source_type": "filtered",
                "input_format": "gray",
                "split": {"train": 1.0, "val": 0.0},
            },
            "pose_b": {
                "zarr_path": "/tmp/pose_b.zarr",
                "source_type": "filtered",
                "input_format": "gray",
                "split": {"train": 1.0, "val": 0.0},
            },
        },
        task="pose",
        random_seed=11,
        sampling_strategy="proportional",
    )

    with pytest.raises(ValueError, match="Mixed keypoint_labels across configured pose datasets"):
        mod.ZarrYOLODataset(cfg, mode="train")
