from __future__ import annotations

import numpy as np
import zarr
from rich.console import Console

from fisheye.diagnostics.check_crop_runs import _check_crop_runs, _check_crop_runs_with_use


def _create_geometry_arrays(group: zarr.Group) -> None:
    group.create_array("frame_indices", data=np.array([0, 1], dtype=np.int32), overwrite=True)
    group.create_array("frame_counts", data=np.array([1, 1], dtype=np.int32), overwrite=True)
    group.create_array("detection_indices", data=np.array([0, 1], dtype=np.int32), overwrite=True)
    group.create_array(
        "roi_coordinates_full",
        data=np.array([[10, 20], [30, 40]], dtype=np.int32),
        overwrite=True,
    )
    group.create_array(
        "bbox_norm_coords",
        data=np.array([[0.5, 0.5, 0.1, 0.1], [0.6, 0.6, 0.1, 0.1]], dtype=np.float32),
        overwrite=True,
    )


def test_check_crop_runs_accepts_geometry_only_runs() -> None:
    root = zarr.group()
    parent = root.create_group("crop_runs")
    parent.attrs["latest"] = "crop_001"
    parent.attrs["latest_any"] = "crop_001"

    crop = parent.create_group("crop_001")
    crop.attrs["crop_storage_mode"] = "geometry_only"
    crop.attrs["source_detect_run"] = "detect_001"
    crop.attrs["source_background_run"] = "background_001"
    _create_geometry_arrays(crop)

    console = Console(record=True, width=200)
    _check_crop_runs(console, parent)
    text = console.export_text()

    assert "crop_storage_mode: geometry_only" in text
    assert "healthy" in text
    assert "missing 'roi_images'" not in text


def test_check_crop_runs_flags_materialized_run_missing_roi_images() -> None:
    root = zarr.group()
    parent = root.create_group("crop_runs")
    parent.attrs["latest"] = "crop_001"
    parent.attrs["latest_materialized"] = "crop_001"
    parent.attrs["latest_any"] = "crop_001"

    crop = parent.create_group("crop_001")
    crop.attrs["crop_storage_mode"] = "materialized"
    crop.attrs["source_detect_run"] = "detect_001"
    crop.attrs["source_background_run"] = "background_001"
    _create_geometry_arrays(crop)

    console = Console(record=True, width=200)
    _check_crop_runs(console, parent)
    text = console.export_text()

    assert "crop_storage_mode: materialized" in text
    assert "missing 'roi_images'" in text


def test_check_crop_runs_flags_geometry_only_training_run_as_contract_violation() -> None:
    root = zarr.group()
    root.attrs["zarr_purpose"] = "training"
    parent = root.create_group("crop_runs")
    parent.attrs["latest"] = "crop_001"
    parent.attrs["latest_any"] = "crop_001"

    crop = parent.create_group("crop_001")
    crop.attrs["crop_storage_mode"] = "geometry_only"
    crop.attrs["source_detect_run"] = "detect_001"
    _create_geometry_arrays(crop)

    console = Console(record=True, width=200)
    _check_crop_runs_with_use(console, parent, zarr_use="training")
    text = console.export_text()

    assert "contract violation" in text
    assert "training contract: crop runs must be materialized" in text


def test_check_crop_runs_does_not_flag_missing_optional_background_attr() -> None:
    root = zarr.group()
    parent = root.create_group("crop_runs")
    parent.attrs["latest"] = "crop_001"
    parent.attrs["latest_materialized"] = "crop_001"
    parent.attrs["latest_any"] = "crop_001"

    crop = parent.create_group("crop_001")
    crop.attrs["crop_storage_mode"] = "materialized"
    crop.attrs["source_detect_run"] = "detect_001"
    crop.create_array(
        "roi_images",
        data=np.zeros((2, 8, 8), dtype=np.uint8),
        overwrite=True,
    )
    _create_geometry_arrays(crop)

    console = Console(record=True, width=200)
    _check_crop_runs(console, parent)
    text = console.export_text()

    assert "healthy" in text
    assert "missing provenance" not in text
    assert "optional attrs missing: source_background_run" in text
    assert "optional attr missing 'source_background_run'" not in text


def test_check_crop_runs_verbose_expands_optional_details() -> None:
    root = zarr.group()
    parent = root.create_group("crop_runs")
    parent.attrs["latest"] = "crop_001"
    parent.attrs["latest_materialized"] = "crop_001"
    parent.attrs["latest_any"] = "crop_001"

    crop = parent.create_group("crop_001")
    crop.attrs["crop_storage_mode"] = "materialized"
    crop.attrs["source_detect_run"] = "detect_001"
    crop.create_array(
        "roi_images",
        data=np.zeros((2, 8, 8), dtype=np.uint8),
        overwrite=True,
    )
    _create_geometry_arrays(crop)

    console = Console(record=True, width=200)
    _check_crop_runs(console, parent, verbose=True)
    text = console.export_text()

    assert "optional attr missing 'source_background_run'" in text
    assert "optional attrs missing:" not in text


def test_check_crop_runs_reports_failed_runs_separately() -> None:
    root = zarr.group()
    parent = root.create_group("crop_runs")
    parent.attrs["latest"] = "crop_failed"
    parent.attrs["latest_any"] = "crop_failed"

    crop = parent.create_group("crop_failed")
    crop.attrs["crop_storage_mode"] = "geometry_only"
    crop.attrs["status"] = "failed"
    crop.attrs["error_message"] = "NameError: selection_policy"
    crop.attrs["detection_source_path"] = "detect_runs/detect_001"
    crop.create_array(
        "roi_coordinates_full",
        data=np.array([[10, 20], [30, 40]], dtype=np.int32),
        overwrite=True,
    )

    console = Console(record=True, width=200)
    _check_crop_runs(console, parent)
    text = console.export_text()

    assert "failed" in text
    assert "pipeline_status: failed" in text
    assert "error: NameError: selection_policy" in text
    assert "missing 'frame_indices'" not in text


def test_check_crop_runs_default_summarizes_long_failed_error_text() -> None:
    root = zarr.group()
    parent = root.create_group("crop_runs")
    parent.attrs["latest"] = "crop_failed"
    parent.attrs["latest_any"] = "crop_failed"

    crop = parent.create_group("crop_failed")
    crop.attrs["crop_storage_mode"] = "materialized"
    crop.attrs["status"] = "failed"
    crop.attrs["error_message"] = (
        "OutOfMemoryError: CUDA out of memory.\n"
        "Tried to allocate 7.28 GiB. GPU 0 has a total capacity of 47.40 GiB.\n"
        "See documentation for Memory Management."
    )
    crop.create_array(
        "roi_images",
        data=np.zeros((1, 8, 8), dtype=np.uint8),
        overwrite=True,
    )

    console = Console(record=True, width=200)
    _check_crop_runs(console, parent)
    text = console.export_text()

    assert "error: OutOfMemoryError: CUDA out of memory. …" in text
    assert "Tried to allocate 7.28 GiB" not in text


def test_check_crop_runs_verbose_keeps_full_failed_error_text() -> None:
    root = zarr.group()
    parent = root.create_group("crop_runs")
    parent.attrs["latest"] = "crop_failed"
    parent.attrs["latest_any"] = "crop_failed"

    crop = parent.create_group("crop_failed")
    crop.attrs["crop_storage_mode"] = "materialized"
    crop.attrs["status"] = "failed"
    crop.attrs["error_message"] = (
        "OutOfMemoryError: CUDA out of memory.\n"
        "Tried to allocate 7.28 GiB. GPU 0 has a total capacity of 47.40 GiB.\n"
        "See documentation for Memory Management."
    )
    crop.create_array(
        "roi_images",
        data=np.zeros((1, 8, 8), dtype=np.uint8),
        overwrite=True,
    )

    console = Console(record=True, width=200)
    _check_crop_runs(console, parent, verbose=True)
    text = console.export_text()

    assert "Tried to allocate 7.28 GiB" in text
    assert "See documentation for Memory Management." in text
