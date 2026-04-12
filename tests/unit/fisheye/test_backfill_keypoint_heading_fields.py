import numpy as np

from fisheye.utils.backfill_keypoint_heading_fields import _backfill_heading_columns


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


def test_backfill_heading_columns_raw_keypoints_writes_and_drops_legacy() -> None:
    root = _FakeGroup()
    run = root.create_group("keypoints_runs").create_group("keypoints_001")
    run.create_array("heading", data=np.array([10.0, np.nan, 30.0], dtype=np.float64))
    run.create_array("detection_success", data=np.array([True, True, False], dtype=bool))
    run.create_array("detection_source", data=np.array([0, 0, 1], dtype=np.int8))
    run.create_array("heading_valid", data=np.array([True, False, False], dtype=bool))

    result = _backfill_heading_columns(
        run,
        root=root,
        success_array_name="detection_success",
        overwrite_existing=False,
        apply=True,
    )

    assert result.status == "ok"
    assert "heading_valid" not in run
    np.testing.assert_array_equal(run["heading_finite"][:], np.array([True, False, True], dtype=bool))
    np.testing.assert_array_equal(run["heading_usable"][:], np.array([True, False, False], dtype=bool))


def test_backfill_heading_columns_refined_defaults_detection_source_to_real() -> None:
    root = _FakeGroup()
    run = root.create_group("refined_keypoints_runs").create_group("refined_keypoints_001")
    run.create_array("keypoints_roi", data=np.zeros((3, 3, 2), dtype=np.float64))
    run.create_array("frame_indices", data=np.array([0, 1, 2], dtype=np.int32))
    run.create_array("detection_indices", data=np.array([0, 0, 0], dtype=np.int32))
    run.create_array("heading", data=np.array([1.0, np.nan, 2.0], dtype=np.float64))
    run.create_array("refined_success", data=np.array([True, False, True], dtype=bool))

    result = _backfill_heading_columns(
        run,
        root=root,
        success_array_name="refined_success",
        overwrite_existing=False,
        apply=True,
    )

    assert result.status == "ok"
    np.testing.assert_array_equal(run["heading_finite"][:], np.array([True, False, True], dtype=bool))
    np.testing.assert_array_equal(run["heading_usable"][:], np.array([True, False, True], dtype=bool))
    assert "heading_delta_prev_deg" in run
    assert "heading_delta_next_deg" in run
    assert "heading_temporal_outlier" in run
    summary = run.attrs["summary_statistics"]
    assert "postprocess" in summary
    assert "heading_temporal_outlier" in summary["postprocess"]
    assert summary["postprocess"]["temporal_heading_status"] == "enabled"


def test_backfill_heading_columns_skips_when_fields_present_and_temporal_summary_ready() -> None:
    root = _FakeGroup()
    run = root.create_group("refined_keypoints_runs").create_group("refined_keypoints_001")
    run.create_array("keypoints_roi", data=np.zeros((2, 3, 2), dtype=np.float64))
    run.create_array("frame_indices", data=np.array([0, 1], dtype=np.int32))
    run.create_array("detection_indices", data=np.array([0, 0], dtype=np.int32))
    run.create_array("heading", data=np.array([1.0, 2.0], dtype=np.float64))
    run.create_array("refined_success", data=np.array([True, True], dtype=bool))
    run.create_array("heading_finite", data=np.array([False, False], dtype=bool))
    run.create_array("heading_usable", data=np.array([False, False], dtype=bool))
    run.create_array("heading_delta_prev_deg", data=np.array([np.nan, 1.0], dtype=np.float32))
    run.create_array("heading_delta_next_deg", data=np.array([1.0, np.nan], dtype=np.float32))
    run.create_array("heading_temporal_outlier", data=np.array([False, False], dtype=bool))
    run.attrs["summary_statistics"] = {
        "postprocess": {
            "heading_temporal_evaluable": 0,
            "heading_temporal_outlier": 0,
            "heading_temporal_outlier_rate_percent": 0.0,
            "temporal_heading_status": "enabled",
        }
    }

    result = _backfill_heading_columns(
        run,
        root=root,
        success_array_name="refined_success",
        overwrite_existing=False,
        apply=True,
    )

    assert result.status == "skipped_existing"
    np.testing.assert_array_equal(run["heading_finite"][:], np.array([False, False], dtype=bool))
    np.testing.assert_array_equal(run["heading_usable"][:], np.array([False, False], dtype=bool))


def test_backfill_heading_columns_refined_disables_sampled_import_temporal_fields() -> None:
    root = _FakeGroup()
    raw = root.create_group("raw_video")
    raw.attrs["import_mode"] = "sampled"
    raw.attrs["frame_step"] = 100
    raw.create_array("original_frame_indices", data=np.array([0, 100, 200], dtype=np.int32))

    run = root.create_group("refined_keypoints_runs").create_group("refined_keypoints_001")
    run.create_array("keypoints_roi", data=np.zeros((3, 3, 2), dtype=np.float64))
    run.create_array("frame_indices", data=np.array([0, 1, 2], dtype=np.int32))
    run.create_array("detection_indices", data=np.array([0, 0, 0], dtype=np.int32))
    run.create_array("heading", data=np.array([0.0, 179.0, 2.0], dtype=np.float64))
    run.create_array("refined_success", data=np.array([True, True, True], dtype=bool))
    run.create_array("usable_keypoints", data=np.array([True, True, True], dtype=bool))
    run.create_array("confidence_valid", data=np.array([True, True, True], dtype=bool))
    run.create_array("geometry_valid", data=np.array([True, True, True], dtype=bool))
    run.create_array("flip_corrected", data=np.array([False, False, False], dtype=bool))
    run.create_array("source_success", data=np.array([True, True, True], dtype=bool))
    run.create_array("detection_source", data=np.array([0, 0, 0], dtype=np.int8))
    run.create_array("heading_delta_prev_deg", data=np.array([np.nan, 1.0, np.nan], dtype=np.float32))
    run.create_array("heading_delta_next_deg", data=np.array([1.0, np.nan, np.nan], dtype=np.float32))
    run.create_array("heading_temporal_outlier", data=np.array([False, False, False], dtype=bool))

    result = _backfill_heading_columns(
        run,
        root=root,
        success_array_name="refined_success",
        overwrite_existing=False,
        apply=True,
    )

    assert result.status == "ok"
    summary = run.attrs["summary_statistics"]["postprocess"]
    assert summary["temporal_heading_status"] == "disabled_sampled_import"
    assert summary["temporal_heading_disabled_reason"] == "sampled_import"
    assert "heading_delta_prev_deg" not in run
    assert "heading_delta_next_deg" not in run
    assert "heading_temporal_outlier" not in run


def test_backfill_heading_columns_detects_shape_mismatch() -> None:
    root = _FakeGroup()
    run = root.create_group("keypoints_runs").create_group("keypoints_001")
    run.create_array("heading", data=np.array([1.0, 2.0], dtype=np.float64))
    run.create_array("detection_success", data=np.array([True], dtype=bool))

    result = _backfill_heading_columns(
        run,
        root=root,
        success_array_name="detection_success",
        overwrite_existing=False,
        apply=True,
    )

    assert result.status == "shape_mismatch"
