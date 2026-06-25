from __future__ import annotations

from pathlib import Path

import numpy as np
import zarr

from fisheye.analysis.track_kinematics import save_track_kinematics_tracks
from fisheye.shared.row_lineage import copy_row_lineage_arrays
from fisheye.shared.zarr.chunk_profiles import (
    GEOMETRY_PRELOAD_ROW_CHUNK,
    GEOMETRY_PRELOAD_STORAGE_PROFILE_ID,
    create_geometry_preload_array,
    geometry_preload_chunks_for_shape,
)
from fisheye.tracking.crop import save_crop_metadata


def _root(path: Path) -> zarr.Group:
    return zarr.open_group(path, mode="w")


def test_geometry_preload_chunks_preserve_trailing_dimensions() -> None:
    assert geometry_preload_chunks_for_shape((20_000,)) == (GEOMETRY_PRELOAD_ROW_CHUNK,)
    assert geometry_preload_chunks_for_shape((20_000, 4)) == (GEOMETRY_PRELOAD_ROW_CHUNK, 4)
    assert geometry_preload_chunks_for_shape((7,)) == (7,)
    assert geometry_preload_chunks_for_shape((0,)) == (1,)
    assert geometry_preload_chunks_for_shape(()) is None


def test_create_geometry_preload_array_stamps_chunks_and_attrs(tmp_path: Path) -> None:
    root = _root(tmp_path / "profile.zarr")

    arr = create_geometry_preload_array(
        root,
        "frame_indices",
        data=np.arange(20_000, dtype=np.int32),
    )

    assert arr.chunks == (GEOMETRY_PRELOAD_ROW_CHUNK,)
    assert arr.attrs["storage_profile_id"] == GEOMETRY_PRELOAD_STORAGE_PROFILE_ID
    assert arr.attrs["chunk_policy_version"] == GEOMETRY_PRELOAD_STORAGE_PROFILE_ID
    assert arr.attrs["storage_profile_row_chunk"] == GEOMETRY_PRELOAD_ROW_CHUNK


def test_row_lineage_copy_uses_geometry_preload_only_when_requested(tmp_path: Path) -> None:
    root = _root(tmp_path / "lineage.zarr")
    source = root.create_group("source")
    source.create_array("frame_indices", data=np.arange(20_000, dtype=np.int32), chunks=(1000,))

    default_target = root.create_group("default_target")
    copy_row_lineage_arrays(
        default_target,
        source,
        names=("frame_indices",),
        total_rois=20_000,
    )
    assert default_target["frame_indices"].chunks == (1000,)
    assert "storage_profile_id" not in default_target["frame_indices"].attrs

    profiled_target = root.create_group("profiled_target")
    copy_row_lineage_arrays(
        profiled_target,
        source,
        names=("frame_indices",),
        total_rois=20_000,
        use_geometry_preload_profile=True,
    )
    assert profiled_target["frame_indices"].chunks == (GEOMETRY_PRELOAD_ROW_CHUNK,)
    assert profiled_target["frame_indices"].attrs["storage_profile_id"] == GEOMETRY_PRELOAD_STORAGE_PROFILE_ID


def test_crop_metadata_writes_validated_lineage_arrays_with_geometry_preload_profile(tmp_path: Path) -> None:
    root = _root(tmp_path / "crop.zarr")
    source = root.create_group("detect")
    total = 20_000
    frame_indices = np.arange(total, dtype=np.int32)
    source.create_array("frame_indices", data=frame_indices, chunks=(1000,))
    source.create_array("bbox_norm_coords", data=np.zeros((total, 4), dtype=np.float64), chunks=(1000, 4))
    source.create_array("refined_row_ids", data=np.arange(total, dtype=np.int64), chunks=(1000,))
    source.create_array("source_detect_row_index", data=np.arange(total, dtype=np.int32), chunks=(1000,))
    crop = root.create_group("crop")

    save_crop_metadata(
        crop_group=crop,
        source_group=source,
        source_path="detect_runs/detect_1",
        source_type="detect",
        detection_source=None,
        total_detections=total,
        num_frames=total + 10,
    )

    for name in (
        "frame_indices",
        "detection_indices",
        "source_refined_row_ids",
        "source_detect_row_index",
    ):
        assert crop[name].chunks == (GEOMETRY_PRELOAD_ROW_CHUNK,)
        assert crop[name].attrs["storage_profile_id"] == GEOMETRY_PRELOAD_STORAGE_PROFILE_ID

    assert crop["bbox_norm_coords"].chunks == (1000, 4)


def _track_data(row_count: int = 20_000) -> dict[str, np.ndarray | dict[str, dict[str, np.ndarray]]]:
    frames = np.arange(row_count, dtype=np.int32)
    floats = np.linspace(0.0, 1.0, row_count, dtype=np.float32)
    bools = np.ones(row_count, dtype=bool)
    positions = np.column_stack([floats, floats + 1.0]).astype(np.float32)
    seconds = np.arange(100, dtype=np.int32)
    second_floats = np.linspace(0.0, 1.0, seconds.size, dtype=np.float32)

    speed_derivatives = {
        level: {
            "acceleration_px": floats,
            "acceleration_mm": floats,
            "smoothed_acceleration_px": floats,
            "smoothed_acceleration_mm": floats,
        }
        for level in ("speed_raw", "speed_filtered", "speed_smoothed", "speed_averaged")
    }

    data: dict[str, np.ndarray | dict[str, dict[str, np.ndarray]]] = {
        "frame_indices": frames,
        "time_seconds": floats,
        "detection_indices": frames,
        "positions_px": positions,
        "positions_mm": positions,
        "heading_degrees": floats,
        "heading_radians": floats,
        "delta_heading_degrees": floats,
        "angular_velocity_deg_s": floats,
        "angular_velocity_raw_deg_s": floats,
        "angular_speed_raw_deg_s": floats,
        "delta_heading_smoothed_degrees": floats,
        "angular_velocity_smoothed_deg_s": floats,
        "angular_speed_smoothed_deg_s": floats,
        "smoothed_heading_degrees": floats,
        "smoothed_heading_radians": floats,
        "keypoint_success": bools,
        "detection_source": np.zeros(row_count, dtype=np.int8),
        "sample_observed": bools,
        "sample_valid": bools,
        "source_observed": bools,
        "keypoint_usable": bools,
        "position_finite": bools,
        "heading_usable": bools,
        "sample_reason_code": np.zeros(row_count, dtype=np.uint16),
        "delta_frames": np.ones(row_count, dtype=np.int32),
        "delta_seconds": floats,
        "transition_valid": bools,
        "transition_reason_code": np.zeros(row_count, dtype=np.uint16),
        "speed_raw_px": floats,
        "speed_raw_mm": floats,
        "speed_filtered_px": floats,
        "speed_filtered_mm": floats,
        "speed_smoothed_px": floats,
        "speed_smoothed_mm": floats,
        "speed_averaged_px": floats,
        "speed_averaged_mm": floats,
        "acceleration_px": floats,
        "acceleration_mm": floats,
        "smoothed_acceleration_px": floats,
        "smoothed_acceleration_mm": floats,
        "speed_derivatives": speed_derivatives,
        "frame_path_distance_raw_px": floats,
        "frame_path_distance_raw_mm": floats,
        "frame_path_distance_filtered_px": floats,
        "frame_path_distance_filtered_mm": floats,
        "frame_path_distance_smoothed_px": floats,
        "frame_path_distance_smoothed_mm": floats,
        "cumulative_path_distance_px": floats,
        "cumulative_path_distance_mm": floats,
        "second_indices": seconds,
        "speed_per_second_px": second_floats,
        "speed_per_second_mm": second_floats,
        "heading_per_second_degrees": second_floats,
        "heading_per_second_resultant": second_floats,
    }
    return data


def test_track_kinematics_writer_uses_geometry_preload_chunks_and_attrs(tmp_path: Path) -> None:
    root = _root(tmp_path / "track.zarr")
    run = root.create_group("track_kinematics")

    save_track_kinematics_tracks(
        run,
        {0: _track_data()},
        [{"track_id": 0, "total_distance_px": 1.0}],
    )

    track = run["tracks"]["id_0"]
    assert run["track_ids"].attrs["storage_profile_id"] == GEOMETRY_PRELOAD_STORAGE_PROFILE_ID
    assert track["frame_indices"].chunks == (GEOMETRY_PRELOAD_ROW_CHUNK,)
    assert track["positions_px"].chunks == (GEOMETRY_PRELOAD_ROW_CHUNK, 2)
    assert (
        track["speed_derivatives"]["speed_raw"]["acceleration_px"].chunks
        == (GEOMETRY_PRELOAD_ROW_CHUNK,)
    )
    assert track["movement"]["speed"]["raw"]["px"].chunks == (GEOMETRY_PRELOAD_ROW_CHUNK,)
    assert track["frame_indices"].attrs["storage_profile_id"] == GEOMETRY_PRELOAD_STORAGE_PROFILE_ID
