from __future__ import annotations

from pathlib import Path

import numpy as np
import zarr

from fisheye.shared.frame_domains import FRAME_DOMAIN_MAPS_GROUP, STORED_ZARR_TO_ACQUISITION_MAP


def _open_store(path: Path):
    return zarr.open_group(str(path), mode="w")


def _write_identity_mapping(raw_group, frame_count: int) -> None:
    maps = raw_group.create_group(FRAME_DOMAIN_MAPS_GROUP)
    maps.attrs["schema_id"] = "palette.frame_domain_maps.v1"
    mapping = maps.create_array(
        STORED_ZARR_TO_ACQUISITION_MAP,
        data=np.arange(int(frame_count), dtype=np.int64),
        chunks=(max(1, int(frame_count)),),
        overwrite=True,
    )
    mapping.attrs["source_domain"] = "stored_zarr_frame"
    mapping.attrs["target_domain"] = "acquisition_frame"
    mapping.attrs["semantics"] = "identity_map_zero_based_full_import"


def build_full_identity_store(tmp_path: Path) -> Path:
    zarr_path = tmp_path / "frame_domains_full_identity.zarr"
    root = _open_store(zarr_path)
    root.attrs["zarr_purpose"] = "analysis"
    root.attrs["total_frames"] = 5
    raw = root.create_group("raw_video")
    raw.attrs["import_mode"] = "full"
    raw.attrs["total_frames"] = 5
    raw.create_array("images_full", shape=(5, 2, 2), chunks=(5, 2, 2), dtype=np.uint8, fill_value=0)
    _write_identity_mapping(raw, 5)
    return zarr_path


def build_full_missing_mapping_store(tmp_path: Path) -> Path:
    zarr_path = tmp_path / "frame_domains_full_missing_mapping.zarr"
    root = _open_store(zarr_path)
    root.attrs["zarr_purpose"] = "analysis"
    root.attrs["total_frames"] = 5
    raw = root.create_group("raw_video")
    raw.attrs["import_mode"] = "full"
    raw.attrs["total_frames"] = 5
    raw.create_array("images_full", shape=(5, 2, 2), chunks=(5, 2, 2), dtype=np.uint8, fill_value=0)
    return zarr_path


def build_subsampled_store(tmp_path: Path) -> Path:
    zarr_path = tmp_path / "frame_domains_subsampled.zarr"
    root = _open_store(zarr_path)
    root.attrs["zarr_purpose"] = "training"
    raw = root.create_group("raw_video")
    raw.attrs["import_mode"] = "sampled"
    raw.attrs["frame_step"] = 2
    raw.attrs["original_video_length"] = 5
    raw.attrs["imported_frame_count"] = 3
    raw.attrs["total_frames"] = 3
    raw.create_array("images_full", shape=(3, 2, 2), chunks=(3, 2, 2), dtype=np.uint8, fill_value=0)
    raw.create_array(
        "original_frame_indices",
        data=np.asarray([0, 2, 4], dtype=np.int64),
        chunks=(3,),
        overwrite=True,
    )
    return zarr_path


def build_crop_video_drop_store(tmp_path: Path) -> Path:
    zarr_path = tmp_path / "frame_domains_crop_video_drop.zarr"
    root = _open_store(zarr_path)
    root.attrs["zarr_purpose"] = "training"
    raw = root.create_group("raw_video")
    raw.attrs["import_mode"] = "sampled"
    raw.attrs["frame_step"] = 2
    raw.attrs["original_video_length"] = 12
    raw.attrs["imported_frame_count"] = 6
    raw.attrs["total_frames"] = 6
    raw.create_array("images_full", shape=(6, 2, 2), chunks=(6, 2, 2), dtype=np.uint8, fill_value=0)
    raw.create_array(
        "original_frame_indices",
        data=np.asarray([0, 2, 4, 6, 8, 10], dtype=np.int64),
        chunks=(6,),
        overwrite=True,
    )

    crop_parent = root.create_group("crop_runs")
    crop = crop_parent.create_group("crop_001")
    crop.attrs["source_crop_video_frame_count"] = 4
    crop.attrs["source_crop_video_frame_indices_semantics"] = (
        "zero_based_frame_index_in_acquisition_crop_video_or_-1_for_supplemental_rows"
    )
    crop.attrs["source_pixel_kind_code_map"] = {
        "acquisition_crop_video": 0,
        "offline_full_frame_supplemental_flat_cache": 1,
    }
    crop.create_array(
        "source_crop_video_frame_indices",
        data=np.asarray([0, 1, 2, 3, -1], dtype=np.int64),
        chunks=(5,),
        overwrite=True,
    )
    crop.create_array(
        "source_frame_indices",
        data=np.asarray([0, 2, 6, 8, 10], dtype=np.int64),
        chunks=(5,),
        overwrite=True,
    )
    crop.create_array(
        "frame_indices",
        data=np.asarray([0, 2, 6, 8, 10], dtype=np.int64),
        chunks=(5,),
        overwrite=True,
    )
    crop.create_array(
        "source_pixel_kind_codes",
        data=np.asarray([0, 0, 0, 0, 1], dtype=np.int8),
        chunks=(5,),
        overwrite=True,
    )
    crop.create_array(
        "supplemental_cache_row_indices",
        data=np.asarray([-1, -1, -1, -1, 0], dtype=np.int64),
        chunks=(5,),
        overwrite=True,
    )
    crop.create_array(
        "frame_counts",
        data=np.asarray([1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0], dtype=np.int32),
        chunks=(12,),
        overwrite=True,
    )
    return zarr_path
