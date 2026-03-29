from __future__ import annotations

import numpy as np
import pytest

from fisheye.utils import extend_keypoint_skeleton as mod


class _FakeArray:
    def __init__(self, data: np.ndarray, *, dtype=None) -> None:
        self._data = np.asarray(data)
        self.shape = self._data.shape
        self.dtype = self._data.dtype
        self.ndim = self._data.ndim
        self.zarr_dtype = dtype

    def __getitem__(self, key):
        return self._data[key]

    def __setitem__(self, key, value) -> None:
        self._data[key] = value


class _FakeGroup(dict):
    def __init__(self, *args, attrs: dict | None = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.attrs = attrs or {}

    def get(self, key: str, default=None):
        return super().get(key, default)

    def group_keys(self) -> list[str]:
        return [str(key) for key, value in self.items() if isinstance(value, _FakeGroup)]

    def array_keys(self) -> list[str]:
        return [str(key) for key, value in self.items() if isinstance(value, _FakeArray)]

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

    def create_array(
        self,
        name: str,
        data=None,
        *,
        shape: tuple[int, ...] | None = None,
        dtype=None,
        fill_value=0,
        overwrite: bool = False,
        **_kwargs,
    ):
        if data is None:
            assert shape is not None
            if dtype is not None and not isinstance(dtype, np.dtype):
                arr = np.full(shape, fill_value, dtype=object)
            else:
                arr = np.full(shape, fill_value, dtype=np.dtype(dtype) if dtype is not None else np.float32)
        else:
            arr = np.asarray(data)
        fake = _FakeArray(arr, dtype=dtype)
        if overwrite or name not in self:
            self[name] = fake
        else:
            raise ValueError(f"{name} already exists")
        return fake


def _install_fake_reason_codec(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_read_reason_labels(group: _FakeGroup):
        reason = group.get("reason")
        if isinstance(reason, _FakeArray):
            return np.asarray(reason[:], dtype=object)
        detection_source = group.get("detection_source")
        if isinstance(detection_source, _FakeArray):
            source = np.asarray(detection_source[:], dtype=np.int8)
            return np.where(source == 1, "interpolated", "clean").astype(object)
        return None

    def _fake_write_reason_columns(group: _FakeGroup, reason, _chunk_size: int, *, overwrite: bool = False, **_kwargs):
        labels = np.asarray(reason, dtype=object).reshape(-1)
        if overwrite or "reason" not in group:
            group["reason"] = _FakeArray(labels)
        else:
            raise ValueError("reason already exists")
        group.attrs["reason_fallback_order"] = ["reason", "detection_source"]
        return ["reason"]

    monkeypatch.setattr(mod, "read_reason_labels", _fake_read_reason_labels)
    monkeypatch.setattr(mod, "write_reason_columns", _fake_write_reason_columns)


def _build_keypoint_root(*, refined: bool = False) -> _FakeGroup:
    root = _FakeGroup()
    parent_name = "refined_keypoints_runs" if refined else "keypoints_runs"
    parent = root.create_group(parent_name)
    parent.attrs["latest"] = "kp_001"

    run = parent.create_group("kp_001")
    run.attrs["kpt_shape"] = [3, 2]
    run.attrs["skeleton_id"] = "pose_schema:traditional_v1"
    run.attrs["keypoint_labels"] = ["swim_bladder", "eye_left", "eye_right"]
    run.attrs["pose_schema"] = {
        "name": "traditional_v1",
        "skeleton_id": "pose_schema:traditional_v1",
        "kpt_shape": [3, 2],
        "keypoint_labels": ["swim_bladder", "eye_left", "eye_right"],
        "nodes": ["swim_bladder", "eye_left", "eye_right"],
        "edges": [[0, 1], [0, 2], [1, 2]],
        "source": "configs/fisheye/pose_schemas/traditional_v1.json",
    }
    run.attrs["custom_attr"] = "source_value"
    run.attrs["keypoint_confidence_labels"] = ["swim_bladder", "eye_left", "eye_right"]

    coords = np.asarray(
        [
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]],
        ],
        dtype=np.float32,
    )
    run.create_array("frame_indices", data=np.asarray([10, 11], dtype=np.int32))
    run.create_array("frame_counts", data=np.asarray([1, 1], dtype=np.int32))
    run.create_array("detection_indices", data=np.asarray([0, 1], dtype=np.int32))
    run.create_array("detection_source", data=np.asarray([0, 1], dtype=np.int8))
    run.create_array("keypoints_roi", data=coords)
    run.create_array("keypoints_img", data=coords + 100.0)
    run.create_array("keypoints_norm", data=coords / 100.0)
    run.create_array("keypoint_confidences", data=np.asarray([[0.9, 0.8, 0.7], [0.6, 0.5, 0.4]], dtype=np.float32))

    metrics = run.create_group("metrics")
    metrics.attrs["units"] = "px"
    metrics.create_array("example_metric", data=np.asarray([1.0, 2.0], dtype=np.float32))
    metrics.create_array("notes", data=np.asarray(["alpha", "beta"], dtype=object))

    if refined:
        run.create_array("refined_success", data=np.asarray([True, True], dtype=np.bool_))
        run.create_array("confidence_valid", data=np.asarray([True, False], dtype=np.bool_))
        run.create_array("geometry_valid", data=np.asarray([True, True], dtype=np.bool_))
        run.create_array("usable_keypoints", data=np.asarray([True, True], dtype=np.bool_))
        run.create_array("heading_usable", data=np.asarray([False, True], dtype=np.bool_))
        run.create_array("failure_indices", data=np.asarray([1], dtype=np.int32))
        run.create_array("reason", data=np.asarray(["clean", "interpolated"], dtype=object))

    return root


def test_extend_keypoint_skeleton_run_dry_run_reports_new_schema() -> None:
    root = _build_keypoint_root()

    summary = mod.extend_keypoint_skeleton_run(
        root,
        source_run="kp_001",
        apply=False,
    )

    assert summary["status"] == "planned"
    assert summary["source_parent"] == "keypoints_runs"
    assert summary["source_run"] == "kp_001"
    assert summary["target_run"] == "kp_001_traditional_v2_seed"
    assert summary["source_kpt_shape"] == [3, 2]
    assert summary["target_schema"] == "traditional_v2"
    assert summary["target_kpt_shape"] == [5, 2]
    assert summary["completion_required_keypoints"] == ["snout_tip", "tail_tip"]
    assert root["keypoints_runs"].attrs["latest"] == "kp_001"
    assert "kp_001_traditional_v2_seed" not in root["keypoints_runs"]


def test_extend_keypoint_skeleton_run_apply_expands_arrays_and_preserves_latest() -> None:
    root = _build_keypoint_root()

    summary = mod.extend_keypoint_skeleton_run(
        root,
        source_run="kp_001",
        target_run="kp_001_v2_seed",
        apply=True,
    )

    assert summary["status"] == "updated"
    parent = root["keypoints_runs"]
    assert parent.attrs["latest"] == "kp_001"
    assert "kp_001_v2_seed" in parent

    run = parent["kp_001_v2_seed"]
    source_run = root["keypoints_runs"]["kp_001"]
    assert run.attrs["custom_attr"] == "source_value"
    assert run.attrs["migration_status"] == "needs_keypoint_completion"
    assert run.attrs["migration_source_run"] == "kp_001"
    assert run.attrs["migration_source_group"] == "keypoints_runs"
    assert run.attrs["migration_target_schema"] == "traditional_v2"
    assert run.attrs["migration_completion_required_keypoints"] == ["snout_tip", "tail_tip"]
    assert run.attrs["source_skeleton_id"] == "pose_schema:traditional_v1"
    assert run.attrs["source_kpt_shape"] == [3, 2]
    assert run.attrs["skeleton_id"] == "pose_skel_traditional_v2"
    assert run.attrs["keypoint_labels"] == [
        "swim_bladder",
        "eye_left",
        "eye_right",
        "snout_tip",
        "tail_tip",
    ]
    assert run.attrs["keypoint_confidence_labels"] == [
        "swim_bladder",
        "eye_left",
        "eye_right",
        "snout_tip",
        "tail_tip",
    ]
    assert run["metrics"].attrs["units"] == "px"
    assert run["metrics"]["notes"].zarr_dtype is not None

    keypoints_roi = np.asarray(run["keypoints_roi"][:], dtype=np.float32)
    keypoints_img = np.asarray(run["keypoints_img"][:], dtype=np.float32)
    keypoints_norm = np.asarray(run["keypoints_norm"][:], dtype=np.float32)
    confidences = np.asarray(run["keypoint_confidences"][:], dtype=np.float32)

    assert keypoints_roi.shape == (2, 5, 2)
    assert keypoints_img.shape == (2, 5, 2)
    assert keypoints_norm.shape == (2, 5, 2)
    assert confidences.shape == (2, 5)
    np.testing.assert_allclose(keypoints_roi[:, :3, :], np.asarray(source_run["keypoints_roi"][:]))
    np.testing.assert_allclose(keypoints_img[:, :3, :], np.asarray(source_run["keypoints_img"][:]))
    np.testing.assert_allclose(keypoints_norm[:, :3, :], np.asarray(source_run["keypoints_norm"][:]))
    np.testing.assert_allclose(confidences[:, :3], np.asarray(source_run["keypoint_confidences"][:]))
    assert np.isnan(keypoints_roi[:, 3:, :]).all()
    assert np.isnan(keypoints_img[:, 3:, :]).all()
    assert np.isnan(keypoints_norm[:, 3:, :]).all()
    assert np.isnan(confidences[:, 3:]).all()


def test_extend_keypoint_skeleton_run_apply_resets_refined_seed_state_and_can_set_latest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_reason_codec(monkeypatch)
    root = _build_keypoint_root(refined=True)

    summary = mod.extend_keypoint_skeleton_run(
        root,
        source_run="kp_001",
        source_parent="refined_keypoints_runs",
        target_run="kp_001_v2_seed",
        set_latest=True,
        apply=True,
    )

    assert summary["target_parent"] == "refined_keypoints_runs"
    parent = root["refined_keypoints_runs"]
    assert parent.attrs["latest"] == "kp_001_v2_seed"

    run = parent["kp_001_v2_seed"]
    np.testing.assert_array_equal(np.asarray(run["refined_success"][:], dtype=bool), np.asarray([False, False]))
    np.testing.assert_array_equal(np.asarray(run["confidence_valid"][:], dtype=bool), np.asarray([False, False]))
    np.testing.assert_array_equal(np.asarray(run["geometry_valid"][:], dtype=bool), np.asarray([False, False]))
    np.testing.assert_array_equal(np.asarray(run["usable_keypoints"][:], dtype=bool), np.asarray([False, False]))
    np.testing.assert_array_equal(np.asarray(run["heading_usable"][:], dtype=bool), np.asarray([False, False]))
    np.testing.assert_array_equal(
        np.asarray(run["failure_indices"][:], dtype=np.int32),
        np.asarray([0, 1], dtype=np.int32),
    )

    reason_labels = mod.read_reason_labels(run)
    assert reason_labels is not None
    assert reason_labels.tolist() == [
        "clean|needs_skeleton_extension",
        "interpolated|needs_skeleton_extension",
    ]
