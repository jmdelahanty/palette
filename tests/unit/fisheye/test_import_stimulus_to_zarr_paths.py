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
        projected_surface = cam.create_group("scale_models").create_group("projected_surface")
        projected_surface.attrs["model_name"] = "projected_surface"
        projected_surface.create_dataset("scale_image_png_buffer", data=np.array([1, 2, 3], dtype=np.uint8))


def _write_stimulus_h5_with_arena_relative_chaser_states(path: Path) -> None:
    _write_stimulus_h5_with_calibration(path)
    chaser_dtype = np.dtype(
        [
            ("stimulus_frame_num", np.uint64),
            ("chaser_pos_x", np.float32),
            ("chaser_pos_y", np.float32),
            ("target_pos_x", np.float32),
            ("target_pos_y", np.float32),
            ("target_clamped_pos_x", np.float32),
            ("target_clamped_pos_y", np.float32),
        ]
    )
    chaser_states = np.array(
        [
            (1000, 20.0, 30.0, 350.0, 358.5, 343.0, 344.0),
            (1001, 21.0, 31.0, 340.0, 330.0, 340.0, 330.0),
        ],
        dtype=chaser_dtype,
    )
    with h5py.File(path, "a") as h5:
        tracking = h5.create_group("tracking_data")
        ds = tracking.create_dataset("chaser_states", data=chaser_states)
        ds.attrs["coordinate_frame"] = "arena_relative_canvas_px"
        ds.attrs["coordinate_origin"] = "top_left_of_active_arena"
        ds.attrs[
            "position_fields"
        ] = "chaser_pos_x,chaser_pos_y,target_pos_x,target_pos_y,target_clamped_pos_x,target_clamped_pos_y"
        ds.attrs["x_axis_direction"] = "right"
        ds.attrs["y_axis_direction"] = "down"


def _write_stimulus_h5_with_protocol_steps(path: Path) -> None:
    _write_stimulus_h5_with_calibration(path)
    events_dtype = np.dtype(
        [
            ("event_name", "S32"),
            ("current_step_index", np.int32),
            ("stimulus_mode_id", np.int32),
            ("camera_frame_id", np.int64),
        ]
    )
    events = np.array(
        [
            (b"STEP_START", 0, 3, 10),
            (b"STEP_END", 0, 3, 70),
            (b"STEP_START", 1, 6, 80),
            (b"STEP_END", 1, 6, 140),
        ],
        dtype=events_dtype,
    )
    protocol = {
        "steps": [
            {
                "name": "left grating",
                "stimulus_mode_str": "MOVING_GRATING",
                "duration_seconds": 1.0,
                "parameters": {
                    "type": "ProtocolMovingGratingParams",
                    "orientation_degrees": 180.0,
                    "speed_mm_per_sec": 3.5,
                    "speed_pps": 17.5,
                    "spatial_freq_cycles_per_mm": 0.2,
                    "spatial_freq_rpp": 0.04,
                    "duty_cycle": 0.5,
                },
            },
            {
                "name": "concentric center",
                "stimulus_mode_str": "CONCENTRIC_GRATING",
                "duration_seconds": 1.0,
                "parameters": {
                    "type": "ProtocolConcentricGratingParams",
                    "is_expanding": False,
                    "speed_mm_per_sec": 4.0,
                    "speed_pps": 20.0,
                    "spatial_freq_cycles_per_mm": 0.25,
                    "spatial_freq_rpp": 0.05,
                    "stimulus_role": "centering_utility",
                    "target_radius_min_mm": 8.0,
                    "target_radius_max_mm": 14.0,
                    "centering_success_fraction_threshold": 0.8,
                },
            },
        ]
    }
    with h5py.File(path, "a") as h5:
        h5.create_dataset("events", data=events)
        protocol_group = h5.create_group("protocol_snapshot")
        protocol_group.create_dataset(
            "protocol_definition_json",
            data=json.dumps(protocol).encode("utf-8"),
        )
        coords = h5.create_group("stimulus_coordinates")
        arena = coords.create_group("arena_1")
        custom = arena.create_group("custom_coordinates")
        custom.attrs["texture_center_x"] = 172.0
        custom.attrs["texture_center_y"] = 173.0


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


def test_import_creates_missing_zarr_root_after_h5_resolves(tmp_path: Path) -> None:
    h5_path = tmp_path / "session.h5"
    zarr_path = tmp_path / "created_analysis.zarr"

    _write_minimal_stimulus_h5(h5_path)

    run_name = mod.import_stimulus_to_zarr(
        stimulus_h5=h5_path,
        zarr_path=zarr_path,
        run_name="stimulus_created_root",
        overwrite=False,
        verbose=False,
        repair_chaser_gaps=False,
    )

    root = zarr.open_group(str(zarr_path), mode="r")
    assert run_name in root["analysis"]["stimulus_runs"]


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
    projected_surface = run_calib["scale_models"]["projected_surface"]
    assert projected_surface.attrs["model_name"] == "projected_surface"
    np.testing.assert_array_equal(
        projected_surface["scale_image_png_buffer"][:],
        np.array([1, 2, 3], dtype=np.uint8),
    )


def test_import_keeps_legacy_run_coordinate_transform_without_group_local_positions(tmp_path: Path) -> None:
    h5_path = tmp_path / "session.h5"
    zarr_path = tmp_path / "sample_analysis.zarr"

    _write_stimulus_h5_with_calibration(h5_path)
    zarr.open_group(str(zarr_path), mode="w")

    run_name = mod.import_stimulus_to_zarr(
        stimulus_h5=h5_path,
        zarr_path=zarr_path,
        run_name="stimulus_legacy_coordinate_transform",
        overwrite=False,
        verbose=False,
        repair_chaser_gaps=False,
    )

    root = zarr.open_group(str(zarr_path), mode="r")
    run_group = root["analysis"]["stimulus_runs"][run_name]
    transform = json.loads(run_group.attrs["coordinate_transform"])
    assert run_group.attrs["coordinate_transform_status"] == "legacy_run_level_texture_to_camera"
    assert transform["scope"] == "run_level_legacy_texture_space"
    assert transform["texture_dimensions"] == [358, 358]
    assert transform["camera_dimensions"] == [4512, 4512]
    assert np.isclose(transform["texture_to_camera_scale"], 4512 / 358)
    assert "legacy_texture_to_camera_transform" not in run_group.attrs


def test_import_suppresses_run_coordinate_transform_for_group_local_positions(tmp_path: Path) -> None:
    h5_path = tmp_path / "session.h5"
    zarr_path = tmp_path / "sample_analysis.zarr"

    _write_stimulus_h5_with_arena_relative_chaser_states(h5_path)
    zarr.open_group(str(zarr_path), mode="w")

    run_name = mod.import_stimulus_to_zarr(
        stimulus_h5=h5_path,
        zarr_path=zarr_path,
        run_name="stimulus_group_local_coordinate_transform",
        overwrite=False,
        verbose=False,
        repair_chaser_gaps=False,
    )

    root = zarr.open_group(str(zarr_path), mode="r")
    run_group = root["analysis"]["stimulus_runs"][run_name]
    chaser_group = run_group["tracking_data"]["chaser_states"]

    assert chaser_group.attrs["coordinate_frame"] == "arena_relative_canvas_px"
    assert chaser_group.attrs["coordinate_origin"] == "top_left_of_active_arena"
    assert "coordinate_transform" not in run_group.attrs
    assert (
        run_group.attrs["coordinate_transform_status"]
        == "suppressed_child_group_coordinate_metadata_authoritative"
    )

    legacy = json.loads(run_group.attrs["legacy_texture_to_camera_transform"])
    assert legacy["scope"] == "legacy_texture_space_fallback"
    assert legacy["texture_dimensions"] == [358, 358]
    position_groups = json.loads(run_group.attrs["position_coordinate_groups"])
    assert position_groups == [
        {
            "coordinate_frame": "arena_relative_canvas_px",
            "coordinate_origin": "top_left_of_active_arena",
            "path": "tracking_data/chaser_states",
            "position_fields": (
                "chaser_pos_x,chaser_pos_y,target_pos_x,target_pos_y,"
                "target_clamped_pos_x,target_clamped_pos_y"
            ),
            "x_axis_direction": "right",
            "y_axis_direction": "down",
        }
    ]


def test_import_materializes_canonical_protocol_step_metadata(tmp_path: Path) -> None:
    h5_path = tmp_path / "session.h5"
    zarr_path = tmp_path / "sample_analysis.zarr"

    _write_stimulus_h5_with_protocol_steps(h5_path)
    zarr.open_group(str(zarr_path), mode="w")

    run_name = mod.import_stimulus_to_zarr(
        stimulus_h5=h5_path,
        zarr_path=zarr_path,
        run_name="stimulus_with_steps",
        overwrite=False,
        verbose=False,
        repair_chaser_gaps=False,
    )

    root = zarr.open_group(str(zarr_path), mode="r")
    run_group = root["analysis"]["stimulus_runs"][run_name]
    steps = run_group["steps"]
    assert steps.attrs["metadata_schema_version"] == 1

    step0 = steps["step_0"]
    assert step0.attrs["step_name"] == "left grating"
    assert step0.attrs["stimulus_mode"] == "MOVING_GRATING"
    assert step0.attrs["start_camera_frame"] == 10
    assert step0.attrs["end_camera_frame"] == 70
    assert json.loads(step0.attrs["raw_protocol_params_json"])["parameters"]["orientation_degrees"] == 180.0

    moving = step0["moving_grating"]
    assert moving.attrs["orientation_degrees_authored"] == 180.0
    assert moving.attrs["grating_direction_camera_deg"] == 180.0
    assert moving.attrs["speed_mm_s"] == 3.5
    assert np.isclose(moving.attrs["temporal_frequency_hz"], 0.7)
    assert moving.attrs["direction_mapping_status"] == "unvalidated_default_zero_offset"

    step1 = steps["step_1"]
    assert step1.attrs["step_name"] == "concentric center"
    assert step1.attrs["stimulus_mode"] == "CONCENTRIC_GRATING"
    concentric = step1["concentric_grating"]
    assert concentric.attrs["radial_polarity_authored"] == "contracting"
    assert concentric.attrs["radial_sign_authored"] == -1
    assert concentric.attrs["stimulus_role"] == "centering_utility"
    assert concentric.attrs["center_source"] == "stimulus_coordinates/arena_1/custom_coordinates.texture_center"
    assert concentric.attrs["center_x_px"] == 172.0
    assert concentric.attrs["center_y_px"] == 173.0
    assert np.isclose(concentric.attrs["center_x_mm"], 34.4)
    assert concentric.attrs["target_radius_min_mm"] == 8.0

    copied_center = run_group["stimulus_coordinates"]["arena_1"]["custom_coordinates"]
    assert copied_center.attrs["texture_center_x"] == 172.0
