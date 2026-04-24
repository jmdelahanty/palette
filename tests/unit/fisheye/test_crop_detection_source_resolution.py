from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from fisheye.tracking.crop import (
    _ensure_numpy_array,
    _extract_detection_rows,
    get_detection_source_info,
    save_crop_metadata,
)


class _FakeArray:
    def __init__(self, data: Any) -> None:
        self._data = np.asarray(data)
        self.shape = self._data.shape

    def __getitem__(self, item):
        return self._data[item]


class _FakeGroup:
    def __init__(self, *, path: str = "") -> None:
        self._children: dict[str, _FakeGroup | _FakeArray] = {}
        self.attrs: dict[str, Any] = {}
        self.path = path

    def create_group(self, name: str) -> "_FakeGroup":
        if "/" in name:
            head, tail = name.split("/", 1)
            return self.require_group(head).create_group(tail)
        child_path = f"{self.path}/{name}" if self.path else name
        child = _FakeGroup(path=child_path)
        self._children[name] = child
        return child

    def require_group(self, name: str) -> "_FakeGroup":
        if "/" in name:
            head, tail = name.split("/", 1)
            return self.require_group(head).require_group(tail)
        existing = self._children.get(name)
        if isinstance(existing, _FakeGroup):
            return existing
        child_path = f"{self.path}/{name}" if self.path else name
        child = _FakeGroup(path=child_path)
        self._children[name] = child
        return child

    def create_array(self, name: str, *, data: Any = None, overwrite: bool = False, **kwargs: Any) -> _FakeArray:
        if not overwrite and name in self._children:
            raise ValueError(f"{name} already exists")
        if data is None:
            data = np.zeros(kwargs["shape"], dtype=kwargs.get("dtype", np.float32))
        arr = _FakeArray(data)
        self._children[name] = arr
        return arr

    def get(self, name: str):
        try:
            return self[name]
        except Exception:
            return None

    def group_keys(self):
        return [key for key, value in self._children.items() if isinstance(value, _FakeGroup)]

    def __contains__(self, key: str) -> bool:
        try:
            _ = self[key]
            return True
        except Exception:
            return False

    def __getitem__(self, key: str):
        if "/" in key:
            current: _FakeGroup | _FakeArray = self
            for token in key.split("/"):
                if not isinstance(current, _FakeGroup):
                    raise KeyError(key)
                current = current._children[token]
            return current
        return self._children[key]


def _build_root() -> _FakeGroup:
    root = _FakeGroup()

    detect_runs = root.create_group("detect_runs")
    detect = detect_runs.create_group("detect_a")
    detect_runs.attrs["latest"] = "detect_a"
    detect.create_array(
        "frame_indices",
        data=np.array([0, 1], dtype=np.int32),
        overwrite=True,
    )
    detect.create_array(
        "bbox_norm_coords",
        data=np.array([[0.5, 0.5, 0.1, 0.1], [0.5, 0.5, 0.1, 0.1]], dtype=np.float32),
        overwrite=True,
    )
    return root


def test_auto_falls_back_to_detect_when_refined_stage_is_incomplete() -> None:
    root = _build_root()
    refined_runs = root.create_group("refined_detect_runs")
    refined = refined_runs.create_group("refined_a")
    refined_runs.attrs["latest"] = "refined_a"
    interpolated = refined.create_group("interpolated")
    interpolated.create_array(
        "bbox_norm_coords",
        data=np.array([[0.5, 0.5, 0.1, 0.1]], dtype=np.float32),
        overwrite=True,
    )

    source_path, _group, _detection_source, source_type = get_detection_source_info(
        root=root,  # type: ignore[arg-type]
        source_type="auto",
    )

    assert source_path == "detect_runs/detect_a"
    assert source_type == "detect"


def test_explicit_refined_source_raises_clear_error_when_incomplete() -> None:
    root = _build_root()
    refined_runs = root.create_group("refined_detect_runs")
    refined = refined_runs.create_group("refined_a")
    refined_runs.attrs["latest"] = "refined_a"
    interpolated = refined.create_group("interpolated")
    interpolated.create_array(
        "bbox_norm_coords",
        data=np.array([[0.5, 0.5, 0.1, 0.1]], dtype=np.float32),
        overwrite=True,
    )

    with pytest.raises(ValueError, match="missing required arrays: frame_indices"):
        get_detection_source_info(root=root, source_type="interpolated")  # type: ignore[arg-type]


def test_auto_uses_curated_refined_root_when_present() -> None:
    root = _build_root()
    refined_runs = root.create_group("refined_detect_runs")
    refined_runs.attrs["latest"] = "refined_a"
    refined = refined_runs.create_group("refined_a")
    refined.create_array("refined_row_ids", data=np.array([0, 1], dtype=np.int64), overwrite=True)
    refined.create_array("frame_indices", data=np.array([0, 1], dtype=np.int32), overwrite=True)
    refined.create_array("entity_ids", data=np.array([0, 0], dtype=np.int32), overwrite=True)
    refined.create_array(
        "bbox_img_xyxy",
        data=np.array([[10, 10, 20, 20], [30, 30, 40, 40]], dtype=np.float64),
        overwrite=True,
    )
    refined.create_array(
        "bbox_norm_coords",
        data=np.array([[0.5, 0.5, 0.1, 0.1], [0.5, 0.5, 0.1, 0.1]], dtype=np.float64),
        overwrite=True,
    )
    for name in (
        "status_codes",
        "source_kind_codes",
        "source_sparse_row_index",
        "review_state_codes",
        "keypoints_state_codes",
        "subject_mask_state_codes",
        "eye_mask_state_codes",
        "swim_bladder_state_codes",
    ):
        refined.create_array(name, data=np.array([0, 0], dtype=np.int8), overwrite=True)
    refined.create_array(
        "source_sparse_group_codes",
        data=np.array([1, 2], dtype=np.int8),
        overwrite=True,
    )

    source_path, _group, detection_source, source_type = get_detection_source_info(
        root=root,  # type: ignore[arg-type]
        source_type="auto",
    )

    assert source_path == "refined_detect_runs/refined_a"
    assert source_type == "refined"
    assert detection_source is not None
    assert detection_source.tolist() == [0, 1]


def test_auto_uses_curated_refined_instances_when_present() -> None:
    root = _build_root()
    refined_runs = root.create_group("refined_detect_runs")
    refined_runs.attrs["latest"] = "refined_a"
    refined = refined_runs.create_group("refined_a")
    instances = refined.create_group("instances")
    instances.create_array("refined_row_ids", data=np.array([0, 1], dtype=np.int64), overwrite=True)
    instances.create_array("frame_indices", data=np.array([0, 1], dtype=np.int32), overwrite=True)
    instances.create_array("frame_offsets", data=np.array([0, 1, 2], dtype=np.int64), overwrite=True)
    instances.create_array(
        "bbox_img_xyxy",
        data=np.array([[10, 10, 20, 20], [30, 30, 40, 40]], dtype=np.float64),
        overwrite=True,
    )
    instances.create_array(
        "bbox_norm_coords",
        data=np.array([[0.5, 0.5, 0.1, 0.1], [0.5, 0.5, 0.1, 0.1]], dtype=np.float64),
        overwrite=True,
    )
    instances.create_array("source_kind_codes", data=np.array([1, 2], dtype=np.int8), overwrite=True)
    instances.create_array("manual_edit_flags", data=np.array([0, 1], dtype=np.int8), overwrite=True)
    instances.create_array("source_detect_row_index", data=np.array([0, 1], dtype=np.int32), overwrite=True)
    instances.create_array("frame_counts", data=np.array([1, 1], dtype=np.int32), overwrite=True)

    source_path, _group, detection_source, source_type = get_detection_source_info(
        root=root,  # type: ignore[arg-type]
        source_type="auto",
    )

    assert source_path == "refined_detect_runs/refined_a/instances"
    assert source_type == "refined"
    assert detection_source is not None
    assert detection_source.tolist() == [0, 1]


def test_explicit_instances_override_uses_curated_detection_source_array() -> None:
    root = _build_root()
    refined_runs = root.create_group("refined_detect_runs")
    refined_runs.attrs["latest"] = "refined_a"
    refined = refined_runs.create_group("refined_a")
    instances = refined.create_group("instances")
    instances.create_array("refined_row_ids", data=np.array([0, 1], dtype=np.int64), overwrite=True)
    instances.create_array("frame_indices", data=np.array([0, 1], dtype=np.int32), overwrite=True)
    instances.create_array("frame_offsets", data=np.array([0, 1, 2], dtype=np.int64), overwrite=True)
    instances.create_array(
        "bbox_img_xyxy",
        data=np.array([[10, 10, 20, 20], [30, 30, 40, 40]], dtype=np.float64),
        overwrite=True,
    )
    instances.create_array(
        "bbox_norm_coords",
        data=np.array([[0.5, 0.5, 0.1, 0.1], [0.5, 0.5, 0.1, 0.1]], dtype=np.float64),
        overwrite=True,
    )
    instances.create_array("source_kind_codes", data=np.array([1, 2], dtype=np.int8), overwrite=True)
    instances.create_array("manual_edit_flags", data=np.array([0, 1], dtype=np.int8), overwrite=True)
    instances.create_array("source_detect_row_index", data=np.array([0, 1], dtype=np.int32), overwrite=True)
    instances.create_array("frame_counts", data=np.array([1, 1], dtype=np.int32), overwrite=True)

    source_path, source_group, detection_source, source_type = get_detection_source_info(
        root=root,  # type: ignore[arg-type]
        source_path_override="refined_detect_runs/refined_a/instances",
    )

    assert source_path == "refined_detect_runs/refined_a/instances"
    assert source_group is refined
    assert source_type == "refined"
    assert detection_source is not None
    assert detection_source.tolist() == [0, 1]


def _seed_curated_instances_source(parent: _FakeGroup) -> _FakeGroup:
    instances = parent.create_group("instances")
    instances.create_array(
        "refined_row_ids",
        data=np.array([10, 12], dtype=np.int64),
        overwrite=True,
    )
    instances.create_array(
        "frame_indices",
        data=np.array([2, 3], dtype=np.int32),
        overwrite=True,
    )
    instances.create_array(
        "frame_offsets",
        data=np.array([0, 0, 1, 2, 2], dtype=np.int64),
        overwrite=True,
    )
    instances.create_array(
        "bbox_img_xyxy",
        data=np.array([[10, 10, 20, 20], [30, 30, 44, 44]], dtype=np.float64),
        overwrite=True,
    )
    instances.create_array(
        "bbox_norm_coords",
        data=np.array([[0.5, 0.5, 0.1, 0.1], [0.4, 0.4, 0.2, 0.2]], dtype=np.float64),
        overwrite=True,
    )
    instances.create_array(
        "source_kind_codes",
        data=np.array([1, 3], dtype=np.int8),
        overwrite=True,
    )
    instances.create_array(
        "manual_edit_flags",
        data=np.array([0, 1], dtype=np.int8),
        overwrite=True,
    )
    instances.create_array(
        "source_detect_row_index",
        data=np.array([4, -1], dtype=np.int32),
        overwrite=True,
    )
    instances.create_array(
        "frame_counts",
        data=np.array([0, 0, 1, 1], dtype=np.int32),
        overwrite=True,
    )
    return instances


def test_save_crop_metadata_copies_refined_row_identity_from_curated_run_root() -> None:
    source = _FakeGroup(path="refined_detect_runs/refined_a")
    _seed_curated_instances_source(source)
    crop = _FakeGroup(path="crop_runs/crop_a")

    save_crop_metadata(
        crop_group=crop,  # type: ignore[arg-type]
        source_group=source,  # type: ignore[arg-type]
        source_path="refined_detect_runs/refined_a/instances",
        source_type="refined",
        detection_source=np.array([0, 0], dtype=np.int8),
        total_detections=2,
        num_frames=5,
    )

    assert crop["detection_indices"][:].tolist() == [0, 1]
    assert crop["source_refined_row_ids"][:].tolist() == [10, 12]
    assert crop["source_detect_row_index"][:].tolist() == [4, -1]
    assert crop.attrs["source_refined_row_ids_available"] is True
    assert crop.attrs["source_refined_row_id_policy"] == "copied_from_detection_source"
    assert crop.attrs["source_detect_row_index_available"] is True


def test_extract_detection_rows_supports_direct_curated_instances_group() -> None:
    source = _FakeGroup(path="refined_detect_runs/refined_a")
    instances = _seed_curated_instances_source(source)

    frame_indices, bbox_norm = _extract_detection_rows(instances)  # type: ignore[arg-type]

    assert frame_indices.tolist() == [2, 3]
    assert bbox_norm.tolist() == [[0.5, 0.5, 0.1, 0.1], [0.4, 0.4, 0.2, 0.2]]


def test_save_crop_metadata_rejects_misaligned_refined_row_identity() -> None:
    source = _FakeGroup(path="refined_detect_runs/refined_a/instances")
    source.create_array("refined_row_ids", data=np.array([10], dtype=np.int64), overwrite=True)
    source.create_array("frame_indices", data=np.array([2, 3], dtype=np.int32), overwrite=True)
    source.create_array(
        "bbox_norm_coords",
        data=np.array([[0.5, 0.5, 0.1, 0.1], [0.4, 0.4, 0.2, 0.2]], dtype=np.float64),
        overwrite=True,
    )
    crop = _FakeGroup(path="crop_runs/crop_a")

    with pytest.raises(
        ValueError,
        match="refined_row_ids length 1 does not match detection row count 2",
    ):
        save_crop_metadata(
            crop_group=crop,  # type: ignore[arg-type]
            source_group=source,  # type: ignore[arg-type]
            source_path="refined_detect_runs/refined_a/instances",
            source_type="refined",
            detection_source=None,
            total_detections=2,
            num_frames=5,
        )


class _FakeGpuArray:
    """Mimic CuPy-style explicit host transfer with implicit conversion disabled."""

    def __init__(self, values: np.ndarray) -> None:
        self._values = values

    def get(self) -> np.ndarray:
        return self._values

    def __array__(self, *args, **kwargs):  # pragma: no cover - should never be used
        raise TypeError(
            "Implicit conversion to a NumPy array is not allowed. "
            "Please use `.get()` to construct a NumPy array explicitly."
        )


def test_ensure_numpy_array_uses_get_for_gpu_like_arrays() -> None:
    gpu_like = _FakeGpuArray(np.array([0, 1, 0], dtype=np.int8))
    out = _ensure_numpy_array(gpu_like, dtype="i1", name="detection_source")
    assert isinstance(out, np.ndarray)
    assert out.dtype == np.int8
    assert out.tolist() == [0, 1, 0]
