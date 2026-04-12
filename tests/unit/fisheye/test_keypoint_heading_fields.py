import numpy as np

from fisheye.tune.keypoint_review import _update_postprocess_summary
from fisheye.utils.patch_keypoints_from_crops import _update_keypoints_summary


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


def test_update_keypoints_summary_writes_heading_fields_and_drops_heading_valid() -> None:
    root = _FakeGroup()
    keypoints_runs = root.create_group("keypoints_runs")
    keypoints = keypoints_runs.create_group("keypoints_001")

    keypoints.create_array("keypoints_roi", data=np.zeros((4, 3, 2), dtype=np.float64))
    keypoints.create_array("detection_success", data=np.array([True, True, False, True], dtype=bool))
    keypoints.create_array("frame_indices", data=np.array([0, 0, 1, 2], dtype=np.int32))
    keypoints.create_array("frame_counts", data=np.array([2, 1, 1], dtype=np.int32))
    keypoints.create_array("heading", data=np.array([10.0, np.nan, 30.0, 40.0], dtype=np.float64))
    keypoints.create_array("detection_source", data=np.array([0, 0, 0, 1], dtype=np.int8))
    keypoints.create_array("heading_valid", data=np.array([True, False, False, False], dtype=bool))

    _update_keypoints_summary(root, keypoints)

    assert "heading_valid" not in keypoints
    np.testing.assert_array_equal(
        keypoints["heading_finite"][:],
        np.array([True, False, True, True], dtype=bool),
    )
    np.testing.assert_array_equal(
        keypoints["heading_usable"][:],
        np.array([True, False, False, False], dtype=bool),
    )


def test_update_postprocess_summary_reports_heading_finite_and_usable() -> None:
    root = _FakeGroup()
    refined = root.create_group("refined")

    refined.create_array("keypoints_roi", data=np.zeros((3, 3, 2), dtype=np.float64))
    refined.create_array("refined_success", data=np.array([True, True, False], dtype=bool))
    refined.create_array("usable_keypoints", data=np.array([True, False, False], dtype=bool))
    refined.create_array("confidence_valid", data=np.array([True, False, False], dtype=bool))
    refined.create_array("geometry_valid", data=np.array([True, True, False], dtype=bool))
    refined.create_array("flip_corrected", data=np.array([False, True, False], dtype=bool))
    refined.create_array("heading_finite", data=np.array([True, True, False], dtype=bool))
    refined.create_array("heading_usable", data=np.array([True, False, False], dtype=bool))
    refined.create_array("source_success", data=np.array([True, True, False], dtype=bool))

    stats = _update_postprocess_summary(refined, root=root, print_summary=False)

    assert stats["heading_finite"] == 2
    assert stats["heading_usable"] == 1
    assert "heading_valid" not in stats


def test_update_postprocess_summary_backfills_temporal_heading_fields() -> None:
    root = _FakeGroup()
    refined = root.create_group("refined")

    refined.create_array("keypoints_roi", data=np.zeros((3, 3, 2), dtype=np.float64))
    refined.create_array("frame_indices", data=np.array([0, 1, 2], dtype=np.int32))
    refined.create_array("detection_indices", data=np.array([0, 0, 0], dtype=np.int32))
    refined.create_array("heading", data=np.array([0.0, 179.0, 2.0], dtype=np.float64))
    refined.create_array("refined_success", data=np.array([True, True, True], dtype=bool))
    refined.create_array("usable_keypoints", data=np.array([True, True, True], dtype=bool))
    refined.create_array("confidence_valid", data=np.array([True, True, True], dtype=bool))
    refined.create_array("geometry_valid", data=np.array([True, True, True], dtype=bool))
    refined.create_array("flip_corrected", data=np.array([False, False, False], dtype=bool))
    refined.create_array("source_success", data=np.array([True, True, True], dtype=bool))
    refined.create_array("detection_source", data=np.array([0, 0, 0], dtype=np.int8))

    stats = _update_postprocess_summary(refined, root=root, print_summary=False)

    assert stats["heading_finite"] == 3
    assert stats["heading_usable"] == 3
    assert stats["heading_temporal_evaluable"] == 1
    assert stats["heading_temporal_outlier"] == 1
    assert stats["heading_temporal_outlier_rate_percent"] == 100.0
    assert stats["temporal_heading_status"] == "enabled"
    np.testing.assert_allclose(
        refined["heading_delta_prev_deg"][:],
        np.array([np.nan, 179.0, 177.0], dtype=np.float32),
        equal_nan=True,
    )
    np.testing.assert_allclose(
        refined["heading_delta_next_deg"][:],
        np.array([179.0, 177.0, np.nan], dtype=np.float32),
        equal_nan=True,
    )
    np.testing.assert_array_equal(
        refined["heading_temporal_outlier"][:],
        np.array([False, True, False], dtype=bool),
    )


def test_update_postprocess_summary_disables_temporal_heading_for_sampled_import() -> None:
    root = _FakeGroup()
    raw = root.create_group("raw_video")
    raw.attrs["import_mode"] = "sampled"
    raw.attrs["frame_step"] = 100
    raw.create_array("original_frame_indices", data=np.array([0, 100, 200], dtype=np.int32))

    refined = root.create_group("refined")
    refined.create_array("keypoints_roi", data=np.zeros((3, 3, 2), dtype=np.float64))
    refined.create_array("frame_indices", data=np.array([0, 1, 2], dtype=np.int32))
    refined.create_array("detection_indices", data=np.array([0, 0, 0], dtype=np.int32))
    refined.create_array("heading", data=np.array([0.0, 179.0, 2.0], dtype=np.float64))
    refined.create_array("refined_success", data=np.array([True, True, True], dtype=bool))
    refined.create_array("usable_keypoints", data=np.array([True, True, True], dtype=bool))
    refined.create_array("confidence_valid", data=np.array([True, True, True], dtype=bool))
    refined.create_array("geometry_valid", data=np.array([True, True, True], dtype=bool))
    refined.create_array("flip_corrected", data=np.array([False, False, False], dtype=bool))
    refined.create_array("source_success", data=np.array([True, True, True], dtype=bool))
    refined.create_array("detection_source", data=np.array([0, 0, 0], dtype=np.int8))

    stats = _update_postprocess_summary(refined, root=root, print_summary=False)

    assert stats["heading_finite"] == 3
    assert stats["heading_usable"] == 3
    assert stats["heading_temporal_evaluable"] == 0
    assert stats["heading_temporal_outlier"] == 0
    assert stats["temporal_heading_status"] == "disabled_sampled_import"
    assert stats["temporal_heading_disabled_reason"] == "sampled_import"
    assert "heading_delta_prev_deg" not in refined
    assert "heading_delta_next_deg" not in refined
    assert "heading_temporal_outlier" not in refined
