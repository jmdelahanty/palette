from __future__ import annotations

import numpy as np

from fisheye.utils import backfill_keypoint_derived_metrics as mod


class _FakeArray:
    def __init__(self, data: np.ndarray) -> None:
        self._data = np.asarray(data)
        self.shape = self._data.shape
        self.dtype = self._data.dtype
        self.ndim = self._data.ndim
        self.chunks = (self._data.shape[0],) if self._data.ndim >= 1 else None

    def __getitem__(self, key):
        return self._data[key]

    def __setitem__(self, key, value) -> None:
        self._data[key] = value


class _FakeGroup:
    def __init__(self, children: dict[str, object] | None = None) -> None:
        self._children = children or {}
        self.attrs: dict[str, object] = {}

    def create_group(self, name: str) -> "_FakeGroup":
        child = _FakeGroup()
        self._children[name] = child
        return child

    def create_array(self, name: str, data=None, shape=None, dtype=None, fill_value=None, overwrite=False, **_kwargs):
        if data is None:
            assert shape is not None
            assert dtype is not None
            if dtype == "bool":
                arr = np.full(shape, bool(fill_value) if fill_value is not None else False, dtype=bool)
            else:
                np_dtype = np.dtype(dtype)
                if fill_value is None:
                    fill = np.nan if np.issubdtype(np_dtype, np.floating) else 0
                else:
                    fill = fill_value
                arr = np.full(shape, fill, dtype=np_dtype)
        else:
            arr = np.asarray(data)
        wrapped = _FakeArray(arr)
        if overwrite or name not in self._children:
            self._children[name] = wrapped
        else:
            raise ValueError(f"{name} already exists")
        return wrapped

    def get(self, name: str):
        return self._children.get(name)

    def __contains__(self, key: str) -> bool:
        return key in self._children

    def __getitem__(self, key: str):
        if "/" in key:
            current = self
            for part in key.split("/"):
                current = current._children[part]
            return current
        return self._children[key]

    def group_keys(self):
        return [k for k, v in self._children.items() if isinstance(v, _FakeGroup)]

    def keys(self):
        return self._children.keys()


def test_iter_zarr_accepts_existing_zarr_directory(tmp_path) -> None:
    zarr_dir = tmp_path / "recording_training.zarr"
    zarr_dir.mkdir()
    paths = list(mod._iter_zarr([zarr_dir], recursive=False))
    assert paths == [zarr_dir]


def test_backfill_run_group_writes_metrics_and_finalizes_migration(monkeypatch) -> None:
    root = _FakeGroup()
    crop = root.create_group("crop_runs").create_group("crop_1")
    crop.create_array("roi_images", data=np.zeros((2, 3, 4), dtype=np.uint8))

    refined = root.create_group("refined_keypoints_runs").create_group("refined_v2")
    refined.attrs["pose_schema"] = {
        "name": "traditional_v2",
        "skeleton_id": "pose_skel_traditional_v2",
    }
    refined.attrs["keypoint_labels"] = ["swim_bladder", "eye_left", "eye_right", "snout_tip", "tail_tip"]
    refined.attrs["source_crop_run"] = "crop_1"
    refined.attrs["migration_status"] = "needs_keypoint_completion"
    refined.attrs["migration_completion_required_keypoints"] = ["snout_tip", "tail_tip"]
    refined.create_array(
        "keypoints_roi",
        data=np.array(
            [
                [[1.0, 1.0], [2.0, 0.0], [2.0, 2.0], [3.0, 1.0], [0.0, 1.0]],
                [[1.0, 1.0], [2.0, 0.0], [2.0, 2.0], [3.0, 1.0], [0.0, 1.0]],
            ],
            dtype=np.float64,
        ),
    )
    refined.create_array("refined_success", data=np.array([True, True], dtype=bool))

    monkeypatch.setattr(mod, "_utc_now", lambda: "2026-03-11T12:00:00+00:00")

    result = mod._backfill_run_group(
        root,
        refined,
        overwrite_existing=False,
        apply=True,
        finalize_migration=True,
    )

    assert result.status == "ok"
    assert result.finalized_migration is True
    assert refined.attrs["derived_metric_schema_id"] == "traditional_v2_derived_metrics"
    assert refined.attrs["migration_status"] == "completed"
    assert refined.attrs["migration_completion_required_keypoints"] == []
    assert refined.attrs["migration_completed_at_utc"] == "2026-03-11T12:00:00+00:00"
    np.testing.assert_array_equal(
        np.asarray(refined["derived_metric_valid"][:], dtype=bool),
        np.ones((2, 4), dtype=bool),
    )


def test_backfill_run_group_skips_without_schema() -> None:
    root = _FakeGroup()
    refined = root.create_group("refined_keypoints_runs").create_group("refined_unknown")
    refined.attrs["keypoint_labels"] = ["a", "b", "c"]
    refined.create_array("keypoints_roi", data=np.zeros((1, 3, 2), dtype=np.float64))

    result = mod._backfill_run_group(
        root,
        refined,
        overwrite_existing=False,
        apply=False,
        finalize_migration=False,
    )

    assert result.status == "no_schema"
