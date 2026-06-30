from __future__ import annotations

from pathlib import Path

import numpy as np
import zarr

from fisheye.utils.build_hybrid_acquisition_offline_crop_run import (
    build_hybrid_acquisition_offline_crop_run,
)


def _create_array(group, name: str, data) -> None:
    group.create_array(name, data=np.asarray(data), overwrite=True)


def _make_hybrid_source_archive(tmp_path: Path) -> tuple[Path, Path]:
    recording_dir = tmp_path / "recording"
    cams_dir = recording_dir / "cams"
    cams_dir.mkdir(parents=True)
    source_video = cams_dir / "Cam123_recording.mp4"
    source_video.write_bytes(b"fake")
    crop_video = recording_dir / "derived" / "external_crop_recorder" / "Cam123_crop.mp4"
    crop_video.parent.mkdir(parents=True)
    crop_video.write_bytes(b"fake-crop")

    zarr_path = recording_dir / "zarr" / "recording_analysis.zarr"
    root = zarr.open_group(str(zarr_path), mode="w")
    root.attrs["width"] = 10
    root.attrs["height"] = 10
    root.attrs["source_video_path"] = str(source_video)

    crop_parent = root.create_group("crop_runs")
    crop_parent.attrs["latest_any"] = "crop_acquisition"
    crop = crop_parent.create_group("crop_acquisition")
    crop.attrs["crop_storage_mode"] = "geometry_only"
    crop.attrs["source_pixels"] = "acquisition_crop_video"
    crop.attrs["roi_pixel_provider"] = "acquisition_crop_video"
    crop.attrs["source_crop_video_path"] = str(crop_video)
    crop.attrs["roi_size"] = [4, 4]
    _create_array(crop, "frame_indices", np.array([0, 2], dtype=np.int64))
    _create_array(crop, "source_crop_video_frame_indices", np.array([0, 1], dtype=np.int64))
    _create_array(crop, "source_crop_local_frame_ids", np.array([0, 2], dtype=np.int64))
    _create_array(crop, "source_crop_meta_row_indices", np.array([0, 2], dtype=np.int64))
    _create_array(crop, "roi_coordinates_full", np.array([[1, 1], [3, 3]], dtype=np.int32))
    _create_array(crop, "roi_sizes_full", np.array([[4, 4], [4, 4]], dtype=np.int32))
    _create_array(crop, "source_crop_xywh", np.array([[1, 1, 4, 4], [3, 3, 4, 4]], dtype=np.float32))
    _create_array(crop, "bbox_img_xyxy", np.array([[2, 2, 4, 4], [4, 4, 6, 6]], dtype=np.float64))
    _create_array(crop, "bbox_norm_coords", np.array([[0.3, 0.3, 0.2, 0.2], [0.5, 0.5, 0.2, 0.2]], dtype=np.float64))
    _create_array(crop, "bbox_roi_xyxy", np.array([[1, 1, 3, 3], [1, 1, 3, 3]], dtype=np.float64))
    _create_array(crop, "bbox_crop_norm_coords", np.array([[0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5]], dtype=np.float64))

    refined_parent = root.create_group("refined_detect_runs")
    refined = refined_parent.create_group("refined_detect_001")
    instances = refined.create_group("instances")
    _create_array(instances, "refined_row_ids", np.array([0, 1, 2], dtype=np.int64))
    _create_array(instances, "frame_indices", np.array([0, 1, 2], dtype=np.int32))
    _create_array(instances, "frame_offsets", np.array([0, 1, 2, 3], dtype=np.int64))
    _create_array(
        instances,
        "bbox_img_xyxy",
        np.array([[2, 2, 4, 4], [6, 6, 8, 8], [4, 4, 6, 6]], dtype=np.float64),
    )
    _create_array(
        instances,
        "bbox_norm_coords",
        np.array([[0.3, 0.3, 0.2, 0.2], [0.7, 0.7, 0.2, 0.2], [0.5, 0.5, 0.2, 0.2]], dtype=np.float64),
    )
    _create_array(instances, "source_kind_codes", np.array([0, 0, 0], dtype=np.int8))
    _create_array(instances, "manual_edit_flags", np.array([False, False, False], dtype=bool))
    _create_array(instances, "source_detect_row_index", np.array([0, 1, 2], dtype=np.int64))
    _create_array(instances, "frame_counts", np.array([1, 1, 1], dtype=np.int32))
    return zarr_path, source_video


def test_build_hybrid_acquisition_offline_crop_run_dry_run_selects_offline_recovered_rows(
    tmp_path: Path,
) -> None:
    zarr_path, source_video = _make_hybrid_source_archive(tmp_path)

    report = build_hybrid_acquisition_offline_crop_run(
        zarr_path,
        acquisition_crop_run="crop_acquisition",
        refined_detect_run="refined_detect_001",
        run_name="crop_hybrid_test",
        source_video_path=source_video,
        apply=False,
    )

    assert report["status"] == "dry_run"
    assert report["summary"]["online_rows"] == 2
    assert report["summary"]["offline_recovered_rows"] == 1
    root = zarr.open_group(str(zarr_path), mode="r")
    assert "crop_hybrid_test" not in root["crop_runs"]
