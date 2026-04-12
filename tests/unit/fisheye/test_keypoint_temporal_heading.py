import numpy as np

from fisheye.shared.keypoint_temporal_heading import (
    compute_temporal_heading_arrays,
    refresh_refined_keypoint_heading_fields,
)


class _FakeArray:
    def __init__(self, data: np.ndarray, *, chunks: tuple[int, ...] | None = None) -> None:
        self._data = np.array(data, copy=True)
        self.chunks = chunks or ((max(1, int(self._data.shape[0])),) if self._data.ndim else (1,))

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(int(dim) for dim in self._data.shape)

    def __getitem__(self, key):
        return self._data[key]

    def __setitem__(self, key, value) -> None:
        self._data[key] = value


class _FakeGroup:
    def __init__(self) -> None:
        self.attrs: dict[str, object] = {}
        self._children: dict[str, object] = {}

    def create_group(self, name: str) -> "_FakeGroup":
        group = _FakeGroup()
        self._children[name] = group
        return group

    def create_array(
        self,
        name: str,
        *,
        data=None,
        shape=None,
        chunks=None,
        dtype=None,
        fill_value=None,
        overwrite: bool = False,
    ) -> _FakeArray:
        if not overwrite and name in self._children:
            raise ValueError(f"{name} already exists")
        if data is None:
            dtype_obj = np.dtype(dtype) if dtype is not None else np.float64
            data = np.full(shape, fill_value, dtype=dtype_obj)
        array = _FakeArray(np.asarray(data), chunks=chunks)
        self._children[name] = array
        return array

    def get(self, name: str, default=None):
        return self._children.get(name, default)

    def __getitem__(self, name: str):
        return self._children[name]

    def __contains__(self, name: str) -> bool:
        return name in self._children

    def __delitem__(self, name: str) -> None:
        del self._children[name]


def test_compute_temporal_heading_arrays_flags_isolated_flip() -> None:
    result = compute_temporal_heading_arrays(
        frame_indices=np.array([0, 1, 2], dtype=np.int32),
        heading_values=np.array([0.0, 179.0, 2.0], dtype=np.float64),
        heading_usable=np.array([True, True, True], dtype=bool),
    )

    np.testing.assert_allclose(
        result["heading_delta_prev_deg"],
        np.array([np.nan, 179.0, 177.0], dtype=np.float32),
        equal_nan=True,
    )
    np.testing.assert_allclose(
        result["heading_delta_next_deg"],
        np.array([179.0, 177.0, np.nan], dtype=np.float32),
        equal_nan=True,
    )
    np.testing.assert_array_equal(
        result["heading_temporal_outlier"],
        np.array([False, True, False], dtype=bool),
    )
    assert result["heading_temporal_evaluable"] == 1
    assert result["heading_temporal_outlier_count"] == 1


def test_compute_temporal_heading_arrays_respects_detection_indices() -> None:
    result = compute_temporal_heading_arrays(
        frame_indices=np.array([0, 0, 1, 1, 2, 2], dtype=np.int32),
        heading_values=np.array([0.0, 150.0, 10.0, 150.0, 20.0, 150.0], dtype=np.float64),
        heading_usable=np.array([True, True, True, True, True, True], dtype=bool),
        detection_indices=np.array([0, 1, 0, 1, 0, 1], dtype=np.int32),
    )

    np.testing.assert_allclose(
        result["heading_delta_prev_deg"],
        np.array([np.nan, np.nan, 10.0, 0.0, 10.0, 0.0], dtype=np.float32),
        equal_nan=True,
    )
    np.testing.assert_allclose(
        result["heading_delta_next_deg"],
        np.array([10.0, 0.0, 10.0, 0.0, np.nan, np.nan], dtype=np.float32),
        equal_nan=True,
    )
    np.testing.assert_array_equal(
        result["heading_temporal_outlier"],
        np.zeros(6, dtype=bool),
    )


def test_refresh_refined_keypoint_heading_fields_writes_arrays() -> None:
    root = _FakeGroup()
    refined = root.create_group("refined")
    refined.create_array("frame_indices", data=np.array([0, 1, 2], dtype=np.int32))
    refined.create_array("detection_indices", data=np.array([0, 0, 0], dtype=np.int32))
    refined.create_array("heading", data=np.array([0.0, 179.0, 2.0], dtype=np.float64))
    refined.create_array("refined_success", data=np.array([True, True, True], dtype=bool))
    refined.create_array("detection_source", data=np.array([0, 0, 0], dtype=np.int8))

    summary = refresh_refined_keypoint_heading_fields(refined, root=root)

    assert summary["available"] is True
    assert summary["heading_finite"] == 3
    assert summary["heading_usable"] == 3
    assert summary["heading_temporal_outlier_count"] == 1
    assert summary["temporal_heading_status"] == "enabled"
    np.testing.assert_array_equal(
        refined["heading_temporal_outlier"][:],
        np.array([False, True, False], dtype=bool),
    )


def test_refresh_refined_keypoint_heading_fields_disables_sampled_imports() -> None:
    root = _FakeGroup()
    raw = root.create_group("raw_video")
    raw.attrs["import_mode"] = "sampled"
    raw.attrs["frame_step"] = 100
    raw.create_array("original_frame_indices", data=np.array([0, 100, 200], dtype=np.int32))

    refined = root.create_group("refined")
    refined.create_array("frame_indices", data=np.array([0, 1, 2], dtype=np.int32))
    refined.create_array("detection_indices", data=np.array([0, 0, 0], dtype=np.int32))
    refined.create_array("heading", data=np.array([0.0, 179.0, 2.0], dtype=np.float64))
    refined.create_array("refined_success", data=np.array([True, True, True], dtype=bool))
    refined.create_array("detection_source", data=np.array([0, 0, 0], dtype=np.int8))

    summary = refresh_refined_keypoint_heading_fields(refined, root=root)

    assert summary["available"] is True
    assert summary["heading_finite"] == 3
    assert summary["heading_usable"] == 3
    assert summary["heading_temporal_evaluable"] == 0
    assert summary["heading_temporal_outlier_count"] == 0
    assert summary["temporal_heading_status"] == "disabled_sampled_import"
    assert summary["temporal_heading_disabled_reason"] == "sampled_import"
    assert "heading_delta_prev_deg" not in refined
    assert "heading_delta_next_deg" not in refined
    assert "heading_temporal_outlier" not in refined
