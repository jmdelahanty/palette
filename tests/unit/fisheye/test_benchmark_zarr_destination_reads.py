from __future__ import annotations

import zarr

from fisheye.utils.benchmark_zarr_destination_reads import (
    _benchmark_index,
    _resolve_latest_run_name,
    benchmark_open_group_reads,
)


def test_benchmark_index_defaults_to_midpoint_and_clamps() -> None:
    assert _benchmark_index(10, None) == 5
    assert _benchmark_index(10, -1) == 0
    assert _benchmark_index(10, 99) == 9
    assert _benchmark_index(0, None) == 0


def test_resolve_latest_run_name_uses_attr_then_sorted_fallback() -> None:
    root = zarr.group()
    parent = root.create_group("runs")
    parent.create_group("run_b")
    parent.create_group("run_a")
    assert _resolve_latest_run_name(parent) == "run_b"
    parent.attrs["latest"] = "run_a"
    assert _resolve_latest_run_name(parent) == "run_a"


def test_benchmark_open_group_reads_returns_expected_summary() -> None:
    root = zarr.group()
    root.create_array(
        "raw_video/images_full",
        shape=(4, 6, 6),
        dtype="uint8",
        chunks=(2, 6, 6),
        overwrite=True,
    )[:] = 1

    crop_parent = root.create_group("crop_runs")
    crop_parent.attrs["latest"] = "crop_test"
    root.create_array(
        "crop_runs/crop_test/roi_images",
        shape=(4, 8, 8),
        dtype="uint8",
        chunks=(2, 8, 8),
        overwrite=True,
    )[:] = 2

    subject_parent = root.create_group("subject_mask_runs")
    subject_parent.attrs["latest"] = "subject_test"
    root.create_array(
        "subject_mask_runs/subject_test/masks_roi",
        shape=(4, 3, 8, 8),
        dtype="uint8",
        chunks=(2, 1, 8, 8),
        overwrite=True,
    )[:] = 3

    summary = benchmark_open_group_reads(root, zarr_path="/tmp/example.zarr", variant="raw", row_index=1)
    assert summary["zarr_path"] == "/tmp/example.zarr"
    assert summary["variant"] == "raw"
    reads = summary["reads"]
    assert reads["raw_video/images_full"]["path"] == "raw_video/images_full"
    assert reads["raw_video/images_full"]["row_index"] == 1
    assert reads["raw_video/images_full"]["block_shape"] == [6, 6]
    assert reads["crop_runs/latest/roi_images"]["selected_run"] == "crop_test"
    assert reads["crop_runs/latest/roi_images"]["path"] == "crop_runs/crop_test/roi_images"
    assert reads["subject_mask_runs/latest/masks_roi"]["selected_run"] == "subject_test"
    assert reads["subject_mask_runs/latest/masks_roi"]["path"] == "subject_mask_runs/subject_test/masks_roi"


def test_benchmark_open_group_reads_reports_missing_arrays() -> None:
    root = zarr.group()
    summary = benchmark_open_group_reads(root, zarr_path="/tmp/example.zarr")
    reads = summary["reads"]
    assert reads["raw_video/images_full"]["status"] == "missing"
    assert reads["crop_runs/latest/roi_images"]["status"] == "missing"
    assert reads["subject_mask_runs/latest/masks_roi"]["status"] == "missing"
