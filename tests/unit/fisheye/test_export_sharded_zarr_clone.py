from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from fisheye.utils import export_sharded_zarr_clone as mod


class _FakeMetadata:
    def __init__(self, data_type: object) -> None:
        self.data_type = data_type


class _FakeArray:
    def __init__(
        self,
        data: np.ndarray,
        *,
        chunks: tuple[int, ...] | None = None,
        shards: tuple[int, ...] | None = None,
        fill_value: object | None = None,
        dtype: object | None = None,
    ) -> None:
        self._data = np.asarray(data)
        self.shape = self._data.shape
        self.ndim = self._data.ndim
        self.dtype = self._data.dtype if dtype is None else dtype
        self.chunks = tuple(int(v) for v in chunks) if chunks is not None else None
        self.shards = tuple(int(v) for v in shards) if shards is not None else None
        self.fill_value = fill_value
        self.compressors = None
        self.filters = None
        self.serializer = None
        self.attrs: dict[str, object] = {}
        self.metadata = _FakeMetadata(self.dtype)

    def __getitem__(self, key):
        return self._data[key]

    def __setitem__(self, key, value) -> None:
        self._data[key] = value


class _FakeGroup(dict):
    def __init__(self) -> None:
        super().__init__()
        self.attrs: dict[str, object] = {}

    def get(self, key: str, default=None):
        return super().get(key, default)

    def items(self):
        return super().items()

    def create_group(self, name: str):
        group = _FakeGroup()
        self[name] = group
        return group

    def require_group(self, name: str):
        value = self.get(name)
        if isinstance(value, _FakeGroup):
            return value
        value = _FakeGroup()
        self[name] = value
        return value

    def create_array(self, name: str, **kwargs):
        data = kwargs.get("data")
        dtype = kwargs.get("dtype")
        if data is None:
            shape = tuple(int(v) for v in kwargs["shape"])
            fill_value = kwargs.get("fill_value", 0)
            data = np.full(shape, fill_value, dtype=np.dtype(dtype))
        array = _FakeArray(
            np.asarray(data),
            chunks=kwargs.get("chunks"),
            shards=kwargs.get("shards"),
            fill_value=kwargs.get("fill_value"),
            dtype=dtype,
        )
        self[name] = array
        return array


def _build_source_root(*, raw_already_sharded: bool = False) -> _FakeGroup:
    root = _FakeGroup()
    root.attrs["root_attr"] = "source"

    raw_video = root.create_group("raw_video")
    raw_video.attrs["raw_attr"] = "copied"
    raw_video.create_array(
        "images_full",
        data=np.arange(4 * 4 * 4, dtype=np.uint8).reshape(4, 4, 4),
        chunks=(2, 4, 4),
        shards=(4, 4, 4) if raw_already_sharded else None,
        overwrite=True,
    )

    crop_parent = root.create_group("crop_runs")
    crop = crop_parent.create_group("crop_001")
    crop.attrs["crop_attr"] = "copied"
    crop.create_array(
        "roi_images",
        data=np.arange(5 * 4 * 4, dtype=np.uint8).reshape(5, 4, 4),
        chunks=(2, 4, 4),
        overwrite=True,
    )
    crop.create_array(
        "roi_coordinates_full",
        data=np.arange(10, dtype=np.int32).reshape(5, 2),
        chunks=(5, 2),
        overwrite=True,
    )

    subject_parent = root.create_group("subject_mask_runs")
    subject_run = subject_parent.create_group("run_a")
    subject_run.create_array(
        "masks_roi",
        data=np.arange(4 * 3 * 4 * 4, dtype=np.uint8).reshape(4, 3, 4, 4),
        chunks=(2, 1, 4, 4),
        overwrite=True,
    )
    subject_run.create_array(
        "mask_probs_roi",
        data=np.linspace(0.0, 1.0, 4 * 3 * 4 * 4, dtype=np.float16).reshape(4, 3, 4, 4),
        chunks=(2, 1, 4, 4),
        overwrite=True,
    )
    subject_run.create_array(
        "detection_source",
        data=np.asarray([0, 1, 0, 1], dtype=np.int8),
        chunks=(4,),
        overwrite=True,
    )

    refined_parent = root.create_group("refined_subject_masks_runs")
    refined_run = refined_parent.create_group("run_refined")
    refined_run.create_array(
        "masks_roi",
        data=np.ones((4, 4, 4, 4), dtype=np.uint8),
        chunks=(2, 1, 4, 4),
        overwrite=True,
    )
    refined_run.create_array(
        "mask_probs_roi",
        data=np.ones((4, 4, 4, 4), dtype=np.float16),
        chunks=(2, 1, 4, 4),
        overwrite=True,
    )
    return root


def _install_open_group(monkeypatch, stores: dict[str, _FakeGroup]) -> None:
    def _fake_open_group(path: str, *, mode: str, zarr_format=None):
        key = str(Path(path))
        if mode == "w":
            stores[key] = _FakeGroup()
            return stores[key]
        return stores[key]

    monkeypatch.setattr(mod.zarr, "open_group", _fake_open_group)
    monkeypatch.setattr(mod.zarr, "Group", _FakeGroup)
    monkeypatch.setattr(mod.zarr, "Array", _FakeArray)


def test_build_export_plan_selects_expected_paths(monkeypatch, tmp_path: Path) -> None:
    source = tmp_path / "source.zarr"
    dest = tmp_path / "dest.zarr"
    stores = {str(source): _build_source_root()}
    _install_open_group(monkeypatch, stores)

    plan = mod.build_export_plan(source, dest, policy="raw_and_crops", target_mb=1)
    rows = {row.path: row for row in plan.array_plans}

    assert rows["raw_video/images_full"].action == "add_shards"
    assert rows["raw_video/images_full"].dest_shards == (4, 4, 4)
    assert rows["raw_video/images_full"].dest_chunks == (2, 4, 4)
    assert rows["crop_runs/crop_001/roi_images"].action == "add_shards"
    assert rows["crop_runs/crop_001/roi_images"].dest_shards == (4, 4, 4)
    assert rows["crop_runs/crop_001/roi_images"].dest_chunks == (2, 4, 4)
    assert rows["subject_mask_runs/run_a/masks_roi"].action == "keep_chunked"
    assert rows["subject_mask_runs/run_a/masks_roi"].dest_shards is None
    assert rows["subject_mask_runs/run_a/masks_roi"].dest_chunks == (2, 1, 4, 4)


def test_export_sharded_zarr_clone_applies_policy_and_copies_data(monkeypatch, tmp_path: Path) -> None:
    source = tmp_path / "source.zarr"
    dest = tmp_path / "dest.zarr"
    stores = {str(source): _build_source_root()}
    _install_open_group(monkeypatch, stores)

    summary = mod.export_sharded_zarr_clone(
        source,
        dest_zarr=dest,
        policy="dense_readmostly_v1",
        target_mb=1,
        apply=True,
    )

    assert summary["status"] == "updated"
    manifest_path = Path(summary["manifest_path"])
    assert manifest_path.exists()

    dest_root = stores[str(dest)]
    assert dest_root.attrs["root_attr"] == "source"
    assert dest_root["raw_video"].attrs["raw_attr"] == "copied"
    assert dest_root["crop_runs"]["crop_001"].attrs["crop_attr"] == "copied"
    assert dest_root["raw_video"]["images_full"].chunks == (2, 4, 4)
    assert dest_root["raw_video"]["images_full"].shards == (4, 4, 4)
    assert dest_root["crop_runs"]["crop_001"]["roi_images"].shards == (4, 4, 4)
    assert dest_root["subject_mask_runs"]["run_a"]["masks_roi"].chunks == (2, 1, 4, 4)
    assert dest_root["subject_mask_runs"]["run_a"]["masks_roi"].shards == (4, 1, 4, 4)
    assert dest_root["refined_subject_masks_runs"]["run_refined"]["mask_probs_roi"].shards == (4, 1, 4, 4)
    assert dest_root["subject_mask_runs"]["run_a"]["detection_source"].shards is None

    np.testing.assert_array_equal(
        np.asarray(dest_root["subject_mask_runs"]["run_a"]["masks_roi"][:], dtype=np.uint8),
        np.asarray(stores[str(source)]["subject_mask_runs"]["run_a"]["masks_roi"][:], dtype=np.uint8),
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["policy"] == "dense_readmostly_v1"
    assert manifest["arrays_added_shards"] >= 4
    assert manifest["arrays_rechunked"] == 0


def test_build_export_plan_preserves_existing_shards(monkeypatch, tmp_path: Path) -> None:
    source = tmp_path / "source.zarr"
    dest = tmp_path / "dest.zarr"
    stores = {str(source): _build_source_root(raw_already_sharded=True)}
    _install_open_group(monkeypatch, stores)

    plan = mod.build_export_plan(source, dest, policy="raw_only", target_mb=1)
    rows = {row.path: row for row in plan.array_plans}

    assert rows["raw_video/images_full"].action == "preserve_existing_shards"
    assert rows["raw_video/images_full"].source_shards == (4, 4, 4)
    assert rows["raw_video/images_full"].dest_shards == (4, 4, 4)


def test_compute_shards_rounds_down_to_chunk_multiple_for_nondivisible_shape(monkeypatch, tmp_path: Path) -> None:
    source = tmp_path / "source.zarr"
    dest = tmp_path / "dest.zarr"
    stores = {str(source): _build_source_root()}
    _install_open_group(monkeypatch, stores)

    plan = mod.build_export_plan(source, dest, policy="raw_and_crops", target_mb=1)
    rows = {row.path: row for row in plan.array_plans}

    assert rows["crop_runs/crop_001/roi_images"].shape == (5, 4, 4)
    assert rows["crop_runs/crop_001/roi_images"].chunks == (2, 4, 4)
    assert rows["crop_runs/crop_001/roi_images"].dest_shards == (4, 4, 4)


def test_build_export_plan_rechunks_dense_subject_masks(monkeypatch, tmp_path: Path) -> None:
    source = tmp_path / "source.zarr"
    dest = tmp_path / "dest.zarr"
    stores = {str(source): _build_source_root()}
    _install_open_group(monkeypatch, stores)

    plan = mod.build_export_plan(source, dest, policy="dense_readmostly_rechunk_v1", target_mb=1)
    rows = {row.path: row for row in plan.array_plans}

    assert rows["subject_mask_runs/run_a/masks_roi"].chunks == (2, 1, 4, 4)
    assert rows["subject_mask_runs/run_a/masks_roi"].dest_chunks == (4, 1, 4, 4)
    assert rows["subject_mask_runs/run_a/masks_roi"].action == "rechunk_and_add_shards"
    assert rows["subject_mask_runs/run_a/masks_roi"].dest_shards == (4, 1, 4, 4)
    assert rows["refined_subject_masks_runs/run_refined/mask_probs_roi"].dest_chunks == (4, 1, 4, 4)
    assert rows["refined_subject_masks_runs/run_refined/mask_probs_roi"].action == "rechunk_and_add_shards"
    assert rows["crop_runs/crop_001/roi_images"].dest_chunks == (2, 4, 4)


def test_export_sharded_zarr_clone_rechunks_dense_subject_masks(monkeypatch, tmp_path: Path) -> None:
    source = tmp_path / "source.zarr"
    dest = tmp_path / "dest.zarr"
    stores = {str(source): _build_source_root()}
    _install_open_group(monkeypatch, stores)

    summary = mod.export_sharded_zarr_clone(
        source,
        dest_zarr=dest,
        policy="dense_readmostly_rechunk_v1",
        target_mb=1,
        apply=True,
    )

    assert summary["status"] == "updated"
    assert summary["arrays_rechunked"] >= 4
    assert summary["arrays_rechunked_and_sharded"] >= 4

    dest_root = stores[str(dest)]
    assert dest_root["subject_mask_runs"]["run_a"]["masks_roi"].chunks == (4, 1, 4, 4)
    assert dest_root["subject_mask_runs"]["run_a"]["masks_roi"].shards == (4, 1, 4, 4)
    assert dest_root["subject_mask_runs"]["run_a"]["mask_probs_roi"].chunks == (4, 1, 4, 4)
    assert dest_root["refined_subject_masks_runs"]["run_refined"]["masks_roi"].chunks == (4, 1, 4, 4)

    np.testing.assert_array_equal(
        np.asarray(dest_root["subject_mask_runs"]["run_a"]["mask_probs_roi"][:], dtype=np.float16),
        np.asarray(stores[str(source)]["subject_mask_runs"]["run_a"]["mask_probs_roi"][:], dtype=np.float16),
    )
