from __future__ import annotations

from pathlib import Path

import numpy as np
import zarr

from fisheye.diagnostics.benchmark_subject_shape_compute import (
    ComputeVariant,
    parse_variant,
    run_benchmark,
)


def _disk(height: int, width: int, center_y: float, center_x: float, radius: float) -> np.ndarray:
    yy, xx = np.ogrid[:height, :width]
    return ((yy - center_y) ** 2 + (xx - center_x) ** 2 <= radius**2).astype(np.uint8)


def _write_source(path: Path) -> None:
    root = zarr.open_group(str(path), mode="w", zarr_format=3)
    parent = root.create_group("refined_subject_masks_runs")
    parent.attrs["latest"] = "refined"
    run = parent.create_group("refined")
    labels = ["subject_body", "eye_left", "eye_right", "swim_bladder"]
    run.attrs.update(
        {
            "mask_labels": labels,
            "label_schema_id": "subject_v1_lr",
            "component_metrics_schema_id": "refined_subject_component_mask_metrics_v1",
            "source_subject_mask_run": "subject",
        }
    )
    run.create_array("available_channels", data=np.ones((4,), dtype=bool), overwrite=True)
    run.create_array("frame_indices", data=np.arange(3, dtype=np.int64), overwrite=True)
    run.create_array("detection_indices", data=np.arange(3, dtype=np.int64), overwrite=True)
    run.create_array("source_refined_row_ids", data=np.arange(3, dtype=np.int64), overwrite=True)
    masks = np.zeros((3, 4, 32, 32), dtype=np.uint8)
    for row in range(3):
        masks[row, 0, 5:27, 10:22] = 1
        masks[row, 1] = _disk(32, 32, 9, 11, 3)
        masks[row, 2] = _disk(32, 32, 9, 21, 3)
        masks[row, 3] = _disk(32, 32, 21, 16, 3)
    run.create_array("masks_roi", data=masks, chunks=(3, 1, 32, 32), overwrite=True)


def test_parse_variant_supports_crop_and_per_task_open() -> None:
    variant = parse_variant("candidate:8:1024:1:crop,per-task-open")

    assert variant == ComputeVariant(
        name="candidate",
        workers=8,
        block_rows=1024,
        native_threads=1,
        persistent_worker_inputs=False,
        centerline_crop_to_foreground=True,
    )


def test_benchmark_runs_real_kernels_on_bounded_read_only_source(tmp_path: Path) -> None:
    source = tmp_path / "source.zarr"
    outputs = tmp_path / "outputs"
    report_path = tmp_path / "report.json"
    _write_source(source)
    source_metadata_before = (source / "zarr.json").read_bytes()

    report = run_benchmark(
        source,
        output_root=outputs,
        refined_run="refined",
        source_start_row=0,
        row_count=3,
        variants=(
            ComputeVariant("baseline", workers=1, block_rows=256, native_threads=1),
            ComputeVariant(
                "cropped",
                workers=1,
                block_rows=256,
                native_threads=1,
                centerline_crop_to_foreground=True,
            ),
        ),
        report_path=report_path,
        apply=True,
    )

    assert report["status"] == "complete"
    assert report["mutates_source"] is False
    assert report["all_variants_exact"] is True
    assert report_path.exists()
    assert (source / "zarr.json").read_bytes() == source_metadata_before
    assert len(report["results"]) == 2
    for result in report["results"]:
        assert result["task_count"] == 1
        assert result["rows_per_second"] > 0
        summed = result["timings"]["summed_timing_seconds"]
        assert summed["source_read_seconds"] >= 0
        assert summed["compute_seconds"] >= 0
        assert summed["persist_seconds"] >= 0
        assert result["native_thread_control"]["opencv_threads"] == 1
        assert result["exactness_vs_first_variant"]["all_arrays_exact"] is True

    baseline = zarr.open_group(str(outputs / "baseline.zarr"), mode="r")
    cropped = zarr.open_group(str(outputs / "cropped.zarr"), mode="r")
    np.testing.assert_array_equal(
        baseline["components/subject_body/centerline_xy"][:],
        cropped["components/subject_body/centerline_xy"][:],
    )
