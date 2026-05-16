from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import zarr

from fisheye.shared.crop_image_source import CropImageSource
from fisheye.shared.flat_roi_cache import (
    FLAT_ROI_CACHE_LAYOUT,
    FLAT_ROI_CACHE_SCHEMA,
    build_flat_roi_cache,
    open_flat_roi_cache,
)


def _make_materialized_crop_archive(tmp_path: Path) -> tuple[Path, np.ndarray]:
    zarr_path = tmp_path / "recording_analysis.zarr"
    root = zarr.open_group(str(zarr_path), mode="w")
    crop_parent = root.create_group("crop_runs")
    crop_parent.attrs["latest"] = "crop_001"
    crop_parent.attrs["latest_any"] = "crop_001"
    crop_parent.attrs["latest_materialized"] = "crop_001"

    roi_images = np.arange(5 * 4 * 3, dtype=np.uint8).reshape(5, 4, 3)
    crop = crop_parent.create_group("crop_001")
    crop.attrs["crop_storage_mode"] = "materialized"
    crop.attrs["roi_size"] = [4, 3]
    crop.attrs["crop_signature"] = "sig-flat-cache-test"
    crop.attrs["crop_revision"] = "rev-001"
    crop.create_array("roi_images", data=roi_images, overwrite=True)
    crop.create_array(
        "roi_coordinates_full",
        data=np.array([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4]], dtype=np.int32),
        overwrite=True,
    )
    crop.create_array("frame_indices", data=np.arange(5, dtype=np.int64), overwrite=True)
    return zarr_path, roi_images


def test_build_flat_roi_cache_roundtrips_through_manifest(tmp_path: Path) -> None:
    zarr_path, roi_images = _make_materialized_crop_archive(tmp_path)
    cache_dir = tmp_path / "cache"

    manifest = build_flat_roi_cache(
        zarr_path=zarr_path,
        output_dir=cache_dir,
        batch_size=2,
        compute_sha256=True,
    )

    assert manifest["schema"] == FLAT_ROI_CACHE_SCHEMA
    assert manifest["layout"] == FLAT_ROI_CACHE_LAYOUT
    assert manifest["cache_complete"] is True
    assert manifest["source"]["crop_run_name"] == "crop_001"
    assert manifest["array"]["shape"] == [5, 4, 3]
    assert manifest["array"]["dtype"] == "uint8"
    assert manifest["array"]["sha256"]

    manifest_path = Path(str(manifest["manifest_path"]))
    cache = open_flat_roi_cache(
        manifest_path,
        expected_archive_path=zarr_path,
        expected_crop_run="crop_001",
        expected_shape=roi_images.shape,
    )
    try:
        np.testing.assert_array_equal(cache[1:4], roi_images[1:4])
    finally:
        cache.close()


def test_crop_image_source_reads_flat_roi_cache_manifest(tmp_path: Path) -> None:
    zarr_path, roi_images = _make_materialized_crop_archive(tmp_path)
    manifest = build_flat_roi_cache(
        zarr_path=zarr_path,
        output_dir=tmp_path / "cache",
        batch_size=3,
    )

    root = zarr.open_group(str(zarr_path), mode="r")
    source = CropImageSource.open(
        root,
        zarr_path=zarr_path,
        roi_cache_manifest=manifest["manifest_path"],
    )
    try:
        assert source.crop_run_name == "crop_001"
        assert source.roi_read_mode == "flat_bin_roi_cache"
        assert source.roi_cache_used is True
        assert source.roi_cache_backend == "flat_bin_v1"
        np.testing.assert_array_equal(source.read_indices([4, 0, 2]), roi_images[[4, 0, 2]])
    finally:
        source.close()


def test_flat_roi_cache_rejects_wrong_shape(tmp_path: Path) -> None:
    zarr_path, _roi_images = _make_materialized_crop_archive(tmp_path)
    manifest = build_flat_roi_cache(
        zarr_path=zarr_path,
        output_dir=tmp_path / "cache",
        batch_size=3,
    )

    with pytest.raises(ValueError, match="shape mismatch"):
        open_flat_roi_cache(manifest["manifest_path"], expected_shape=(5, 4, 4))
