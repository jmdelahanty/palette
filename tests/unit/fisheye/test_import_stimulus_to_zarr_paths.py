from __future__ import annotations

from pathlib import Path
import json

import h5py
import numpy as np
import zarr

from fisheye.analysis import import_stimulus_to_zarr as mod


def _write_minimal_stimulus_h5(path: Path) -> None:
    dtype = np.dtype(
        [
            ("stimulus_frame_num", np.uint64),
            ("triggering_camera_frame_id", np.uint64),
            ("timestamp_ns", np.int64),
        ]
    )
    frame_metadata = np.array(
        [
            (1000, 2000, 1_000_000_000),
            (1001, 2001, 1_008_333_333),
        ],
        dtype=dtype,
    )
    with h5py.File(path, "w") as h5:
        video_metadata = h5.create_group("video_metadata")
        video_metadata.create_dataset("frame_metadata", data=frame_metadata)


def _write_stimulus_h5_with_calibration(path: Path) -> None:
    _write_minimal_stimulus_h5(path)
    arena_config = {
        "active_camera_id": "2010093",
        "calculated_z_eff_mm": 0.0,
        "experimental_area_center_x_px": 172.0,
        "experimental_area_center_y_px": 172.0,
        "experimental_area_radius_px": 166.0,
        "experimental_area_shape": "CIRCLE",
        "sub_arena_x_px": 270,
        "sub_arena_y_px": 520,
        "sub_arena_width_px": 344,
        "sub_arena_height_px": 344,
        "camera_calibrations": [
            {
                "camera_id": "2010093",
                "native_width_px": 4512,
                "native_height_px": 4512,
                "pixels_per_mm_camera": 50.0,
                "pixels_per_mm_projector": 5.0,
                "real_world_ref_mm": 10.0,
            }
        ],
    }
    homography_yml = """%YAML:1.0
---
homography_matrix:
  rows: 3
  cols: 3
  dt: d
  data: [1, 0, 10, 0, 1, 20, 0, 0, 1]
"""
    with h5py.File(path, "a") as h5:
        calib = h5.create_group("calibration_snapshot")
        calib.create_dataset("arena_config_json", data=json.dumps(arena_config).encode("utf-8"))
        cam = calib.create_group("2010093")
        cam.attrs["pixels_per_mm_camera"] = 50.0
        cam.attrs["pixels_per_mm_projector"] = 5.0
        cam.create_dataset("homography_matrix_yml", data=homography_yml.encode("utf-8"))


def test_import_sets_source_stimulus_video_path_when_rendered_mp4_exists(tmp_path: Path) -> None:
    h5_path = tmp_path / "session.h5"
    rendered_mp4 = tmp_path / "session.mp4"
    zarr_path = tmp_path / "sample_analysis.zarr"

    _write_minimal_stimulus_h5(h5_path)
    rendered_mp4.touch()
    zarr.open_group(str(zarr_path), mode="w")

    run_name = mod.import_stimulus_to_zarr(
        stimulus_h5=h5_path,
        zarr_path=zarr_path,
        run_name="stimulus_test",
        overwrite=False,
        verbose=False,
        repair_chaser_gaps=False,
    )

    root = zarr.open_group(str(zarr_path), mode="r")
    run_group = root["analysis"]["stimulus_runs"][run_name]
    assert run_group.attrs.get("source_h5") == str(h5_path.resolve())
    assert run_group.attrs.get("source_stimulus_video_path") == str(rendered_mp4.resolve())


def test_import_omits_source_stimulus_video_path_when_rendered_mp4_missing(tmp_path: Path) -> None:
    h5_path = tmp_path / "session.h5"
    zarr_path = tmp_path / "sample_analysis.zarr"

    _write_minimal_stimulus_h5(h5_path)
    zarr.open_group(str(zarr_path), mode="w")

    run_name = mod.import_stimulus_to_zarr(
        stimulus_h5=h5_path,
        zarr_path=zarr_path,
        run_name="stimulus_test_no_video",
        overwrite=False,
        verbose=False,
        repair_chaser_gaps=False,
    )

    root = zarr.open_group(str(zarr_path), mode="r")
    run_group = root["analysis"]["stimulus_runs"][run_name]
    assert run_group.attrs.get("source_h5") == str(h5_path.resolve())
    assert "source_stimulus_video_path" not in run_group.attrs


def test_import_materializes_h5_calibration_to_analysis_calibration(tmp_path: Path) -> None:
    h5_path = tmp_path / "session.h5"
    zarr_path = tmp_path / "sample_analysis.zarr"

    _write_stimulus_h5_with_calibration(h5_path)
    zarr.open_group(str(zarr_path), mode="w")

    run_name = mod.import_stimulus_to_zarr(
        stimulus_h5=h5_path,
        zarr_path=zarr_path,
        run_name="stimulus_with_calibration",
        overwrite=False,
        verbose=False,
        repair_chaser_gaps=False,
    )

    root = zarr.open_group(str(zarr_path), mode="r")
    calib = root["analysis"]["calibration"]
    assert calib.attrs["source_h5"] == str(h5_path.resolve())
    assert calib.attrs["source_stimulus_run"] == run_name
    assert calib.attrs["active_camera_id"] == "2010093"
    assert calib.attrs["pixel_to_mm"] == np.float64(1.0 / 50.0)
    assert calib.attrs["pixels_per_mm_camera"] == 50.0
    assert calib.attrs["pixels_per_mm_projector"] == 5.0
    assert calib.attrs["z_eff_status"] == "unusable_nonpositive"
    np.testing.assert_allclose(
        calib["homography_matrix"][:],
        np.array([[1.0, 0.0, 10.0], [0.0, 1.0, 20.0], [0.0, 0.0, 1.0]]),
    )

    run_calib = root["analysis"]["stimulus_runs"][run_name]["calibration"]["2010093"]
    np.testing.assert_allclose(run_calib["homography_matrix"][:], calib["homography_matrix"][:])
