from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from fisheye.training.config import SubjectMaskTrainingConfig
from fisheye.training import zarr_subject_mask_dataset as mod


class _FakeArray:
    def __init__(self, data: np.ndarray, *, chunks: tuple[int, ...] | None = None) -> None:
        self._data = np.asarray(data)
        self.shape = self._data.shape
        self.dtype = self._data.dtype
        self.chunks = chunks or (self.shape[0],)

    def __getitem__(self, item):
        return self._data[item]


class _FakeGroup:
    def __init__(self) -> None:
        self.attrs: dict[str, object] = {}
        self._children: dict[str, object] = {}

    def create_group(self, name: str) -> "_FakeGroup":
        group = _FakeGroup()
        self._children[name] = group
        return group

    def create_array(self, name: str, *, data: np.ndarray, chunks: tuple[int, ...] | None = None, overwrite: bool = False):
        if not overwrite and name in self._children:
            raise ValueError(f"{name} already exists")
        arr = _FakeArray(np.asarray(data), chunks=chunks)
        self._children[name] = arr
        return arr

    def get(self, name: str, default=None):
        return self._children.get(name, default)

    def __getitem__(self, name: str):
        return self._children[name]

    def __contains__(self, name: str) -> bool:
        return name in self._children

    def group_keys(self):
        return [name for name, value in self._children.items() if isinstance(value, _FakeGroup)]


def _build_training_root() -> _FakeGroup:
    root = _FakeGroup()
    root.attrs["zarr_purpose"] = "training"
    root.attrs["training_task"] = "subject_masks"
    root.attrs["training_export"] = {
        "input_format": "gray",
        "source_subject_mask_run": "subject_masks_source_001",
        "source_subject_mask_runs": ["subject_masks_source_001"],
        "source_crop_run": "crop_source_001",
        "source_crop_runs": ["crop_source_001"],
        "channel_supervision_summary": {
            "coverage_class": "partial_subject_masks",
        },
    }

    crop_parent = root.create_group("crop_runs")
    crop_parent.attrs["latest"] = "training_export_001"
    crop = crop_parent.create_group("training_export_001")
    crop.create_array("roi_images", data=np.zeros((4, 8, 8), dtype=np.uint8), chunks=(2, 8, 8))

    mask_parent = root.create_group("subject_mask_runs")
    mask_parent.attrs["latest"] = "training_export_001"
    mask = mask_parent.create_group("training_export_001")
    mask.attrs["label_schema_id"] = "subject_v1_union"
    mask.attrs["mask_labels"] = ["subject_body", "eyes_union", "swim_bladder"]
    masks = np.zeros((4, 3, 8, 8), dtype=np.uint8)
    masks[:, 0, 1:4, 1:4] = 1
    masks[:, 1, 2:5, 3:6] = 1
    valid = np.array(
        [
            [True, True, False],
            [True, True, False],
            [True, False, True],
            [True, False, True],
        ],
        dtype=np.bool_,
    )
    mask.create_array("masks_roi", data=masks, chunks=(2, 3, 8, 8))
    mask.create_array("target_valid_channels", data=valid, chunks=(4, 3))

    splits = root.create_group("splits")
    splits.create_array("train_indices", data=np.array([0, 1, 2], dtype=np.int64), chunks=(3,))
    splits.create_array("val_indices", data=np.array([3], dtype=np.int64), chunks=(1,))
    return root


def test_load_subject_mask_training_artifact_reads_schema_and_splits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = tmp_path / "subject_training.zarr"
    artifact.mkdir()
    (artifact / "zarr.json").write_text("{}", encoding="utf-8")
    fake_root = _build_training_root()
    monkeypatch.setattr(mod.zarr, "open_group", lambda *_args, **_kwargs: fake_root)
    monkeypatch.setattr(mod.zarr, "Group", _FakeGroup)

    store = mod.load_subject_mask_training_artifact(
        artifact,
        subject_mask_run=None,
        crop_run=None,
        expected_label_schema_id="subject_v1_union",
    )

    assert tuple(store.meta["mask_labels"]) == ("subject_body", "eyes_union", "swim_bladder")
    assert store.train_indices.tolist() == [0, 1, 2]
    assert store.val_indices.tolist() == [3]
    assert store.meta["source_subject_mask_run"] == "subject_masks_source_001"
    assert store.meta["source_crop_run"] == "crop_source_001"


def test_load_subject_mask_training_artifact_falls_back_to_source_index_run_names(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = tmp_path / "subject_training.zarr"
    artifact.mkdir()
    (artifact / "zarr.json").write_text("{}", encoding="utf-8")
    fake_root = _build_training_root()
    fake_root.attrs["training_export"].pop("source_subject_mask_run", None)
    fake_root.attrs["training_export"].pop("source_subject_mask_runs", None)
    source_index = fake_root.create_group("source_index")
    source_index.create_array(
        "source_run_name",
        data=np.asarray(["subject_masks_source_legacy"], dtype=object),
        chunks=(1,),
    )
    monkeypatch.setattr(mod.zarr, "open_group", lambda *_args, **_kwargs: fake_root)
    monkeypatch.setattr(mod.zarr, "Group", _FakeGroup)

    store = mod.load_subject_mask_training_artifact(
        artifact,
        subject_mask_run=None,
        crop_run=None,
        expected_label_schema_id="subject_v1_union",
    )

    assert store.meta["source_subject_mask_run"] == "subject_masks_source_legacy"
    assert store.meta["source_crop_run"] == "crop_source_001"


def test_build_subject_mask_training_datasets_returns_valid_channel_batches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = tmp_path / "subject_training.zarr"
    artifact.mkdir()
    (artifact / "zarr.json").write_text("{}", encoding="utf-8")
    fake_root = _build_training_root()
    monkeypatch.setattr(mod.zarr, "open_group", lambda *_args, **_kwargs: fake_root)
    monkeypatch.setattr(mod.zarr, "Group", _FakeGroup)

    cfg_path = tmp_path / "subject_mask_training.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "datasets:",
                "  ds1:",
                f"    zarr_path: {artifact}",
                "training_params:",
                "  model: unet_small",
                "  epochs: 1",
                "  batch: 2",
                "  batch_size: 2",
                "  imgsz: 8",
                "  lr0: 0.001",
                "  momentum: 0.9",
                "  weight_decay: 0.0",
                "  patience: 3",
                "  device: cpu",
                "  label_schema_id: subject_v1_union",
                "random_seed: 0",
                "num_workers: 0",
            ]
        ),
        encoding="utf-8",
    )

    config = SubjectMaskTrainingConfig.from_yaml(cfg_path)
    bundle = mod.build_subject_mask_training_datasets(config)
    sample = bundle.train_dataset[0]

    assert bundle.label_schema_id == "subject_v1_union"
    assert bundle.mask_labels == ("subject_body", "eyes_union", "swim_bladder")
    assert sample["img"].shape == (1, 8, 8)
    assert sample["masks"].shape == (3, 8, 8)
    assert sample["valid_channels"].shape == (3,)
