import numpy as np
import zarr

from fisheye.analysis.chaser_state_interpolator import analyze_frame_gaps, interpolate_metadata
from fisheye.shared.zarr.columnar import (
    load_structured_dataset,
    pick_chunks,
    write_columnar_dataset,
)


def test_chaser_interpolator_reexports_historical_storage_names():
    from fisheye.analysis import chaser_state_interpolator
    from fisheye.shared.zarr import columnar

    assert chaser_state_interpolator.load_structured_dataset is columnar.load_structured_dataset
    assert chaser_state_interpolator.pick_chunks is columnar.pick_chunks
    assert chaser_state_interpolator.read_columnar_dataset is columnar.read_columnar_dataset
    assert chaser_state_interpolator.store_array is columnar.store_array
    assert chaser_state_interpolator.write_columnar_dataset is columnar.write_columnar_dataset


def test_write_columnar_dataset_roundtrip(tmp_path):
    zarr_path = tmp_path / "roundtrip.zarr"
    root = zarr.open(str(zarr_path), mode="w")

    video_group = root.create_group("video_metadata")
    dtype = np.dtype(
        [
            ("stimulus_frame_num", np.uint64),
            ("triggering_camera_frame_id", np.uint64),
            ("timestamp_ns", np.int64),
        ]
    )
    data = np.zeros(4, dtype=dtype)
    data["stimulus_frame_num"] = np.arange(100, 104, dtype=np.uint64)
    data["triggering_camera_frame_id"] = np.arange(200, 204, dtype=np.uint64)
    data["timestamp_ns"] = np.array([1_000, 2_000, 3_000, 4_000], dtype=np.int64)

    attrs = {"original_records": 4, "total_records": 4}
    write_columnar_dataset(video_group, "frame_metadata", data, attrs)

    loaded, loaded_attrs = load_structured_dataset(video_group, "frame_metadata")

    np.testing.assert_array_equal(loaded["stimulus_frame_num"], data["stimulus_frame_num"])
    np.testing.assert_array_equal(
        loaded["triggering_camera_frame_id"], data["triggering_camera_frame_id"]
    )
    np.testing.assert_array_equal(loaded["timestamp_ns"], data["timestamp_ns"])
    assert loaded_attrs["original_records"] == 4
    assert loaded_attrs["total_records"] == 4


def test_pick_chunks_returns_positive_chunks_for_empty_arrays():
    assert pick_chunks(()) is None
    assert pick_chunks((0,)) == (1,)
    assert pick_chunks((0, 2)) == (1, 2)


def test_write_columnar_dataset_empty_roundtrip(tmp_path):
    zarr_path = tmp_path / "empty_roundtrip.zarr"
    root = zarr.open(str(zarr_path), mode="w")
    group = root.create_group("analysis")
    dtype = np.dtype(
        [
            ("bout_id", np.int32),
            ("start_time_s", np.float64),
            ("point_type", "S5"),
        ]
    )
    data = np.zeros(0, dtype=dtype)

    write_columnar_dataset(group, "empty_bouts", data, {"n_bouts": 0})
    loaded, loaded_attrs = load_structured_dataset(group, "empty_bouts")

    assert loaded.shape == (0,)
    assert loaded.dtype == dtype
    assert loaded_attrs["n_bouts"] == 0


def test_interpolate_metadata_accepts_epoch_timestamp_field():
    dtype = np.dtype(
        [
            ("stimulus_frame_num", np.uint64),
            ("triggering_camera_frame_id", np.uint64),
            ("timestamp_ns_epoch", np.int64),
        ]
    )
    data = np.zeros(2, dtype=dtype)
    data["stimulus_frame_num"] = np.array([100, 102], dtype=np.uint64)
    data["triggering_camera_frame_id"] = np.array([10, 12], dtype=np.uint64)
    data["timestamp_ns_epoch"] = np.array([1_000, 3_000], dtype=np.int64)

    stats = analyze_frame_gaps(data, console=None)
    combined, mask = interpolate_metadata(data, stats, console=None)

    assert combined.shape == (3,)
    assert mask.tolist() == [True, False, True]
    assert combined["stimulus_frame_num"].tolist() == [100, 101, 102]
    assert combined["triggering_camera_frame_id"].tolist() == [10, 11, 12]
    assert combined["timestamp_ns_epoch"].tolist() == [1_000, 2_000, 3_000]
