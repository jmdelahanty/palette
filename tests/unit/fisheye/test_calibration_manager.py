import numpy as np
import pytest
import zarr

from fisheye.analysis.calibration_manager import CalibrationManager
from fisheye.analysis.track_kinematics import resolve_calibration


def test_save_and_load_calibration_with_stimulus_offsets(tmp_path):
    zarr_path = tmp_path / "calibration.zarr"
    # Initialize empty zarr archive
    zarr.open(str(zarr_path), mode="w")

    manager = CalibrationManager(str(zarr_path), verbose=False)

    calibration_payload = {
        "pixel_to_mm": 0.02,
        "primary_camera_id": "0",
        "stimulus_offset_x": 12.5,
        "stimulus_offset_y": -4.0,
        "cameras": {
            "0": {
                "stimulus_offset_x": 12.5,
                "stimulus_offset_y": -4.0,
                "homography_matrix": np.eye(3).tolist(),
                "homography_metadata": {
                    "rows": 3,
                    "cols": 3,
                    "dt": "f",
                    "calibration_timestamp_utc": "2024-01-01T00:00:00Z",
                },
            },
            "1": {
                "stimulus_offset_x": 3.2,
                "stimulus_offset_y": 1.1,
            },
        },
    }

    manager.save_calibration(calibration_payload, overwrite=True)

    root = zarr.open(str(zarr_path), mode="r")
    calib_group = root["calibration"]

    assert calib_group.attrs["stimulus_offset_x"] == pytest.approx(12.5)
    assert calib_group.attrs["stimulus_offset_y"] == pytest.approx(-4.0)
    assert calib_group.attrs["primary_camera_id"] == "0"

    cameras_group = calib_group["cameras"]
    cam0 = cameras_group["0"]
    assert cam0.attrs["stimulus_offset_x"] == pytest.approx(12.5)
    assert cam0.attrs["stimulus_offset_y"] == pytest.approx(-4.0)
    np.testing.assert_array_almost_equal(
        cam0["homography_matrix"][:], np.eye(3, dtype=np.float64)
    )

    cam1 = cameras_group["1"]
    assert cam1.attrs["stimulus_offset_x"] == pytest.approx(3.2)
    assert cam1.attrs["stimulus_offset_y"] == pytest.approx(1.1)

    # Ensure CalibrationManager.get_calibration surfaces the per-camera data
    loaded = manager.get_calibration()
    assert loaded["stimulus_offset_x"] == pytest.approx(12.5)
    assert loaded["stimulus_offset_y"] == pytest.approx(-4.0)
    assert loaded["primary_camera_id"] == "0"
    assert "cameras" in loaded
    assert loaded["cameras"]["0"]["stimulus_offset_x"] == pytest.approx(12.5)
    assert loaded["cameras"]["1"]["stimulus_offset_y"] == pytest.approx(1.1)

    # Verify resolve_calibration exposes offsets for downstream consumers
    pixel_to_mm, info = resolve_calibration(root)
    assert pixel_to_mm == pytest.approx(0.02)
    assert info["stimulus_offset_x"] == pytest.approx(12.5)
    assert info["stimulus_offset_y"] == pytest.approx(-4.0)
    assert info["primary_camera_id"] == "0"
    assert info["camera_offsets"]["0"]["stimulus_offset_x"] == pytest.approx(12.5)
    assert info["camera_offsets"]["1"]["stimulus_offset_y"] == pytest.approx(1.1)


def test_track_kinematics_resolves_analysis_calibration(tmp_path):
    zarr_path = tmp_path / "analysis_calibration.zarr"
    root = zarr.open(str(zarr_path), mode="w")
    calibration = root.create_group("analysis").create_group("calibration")
    calibration.attrs["pixel_to_mm"] = 0.0188
    calibration.attrs["pixels_per_mm_camera"] = 53.2
    calibration.attrs["measured_fps"] = 100.0

    pixel_to_mm, info = resolve_calibration(root)

    assert pixel_to_mm == pytest.approx(0.0188)
    assert info["has_calibration"] is True
    assert info["calibration_path"] == "analysis/calibration"
    assert info["measured_fps"] == pytest.approx(100.0)
