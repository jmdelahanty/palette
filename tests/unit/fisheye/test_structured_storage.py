import numpy as np
import zarr

from fisheye.analysis.chaser_state_interpolator import (
    load_structured_dataset,
    write_columnar_dataset,
)


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
