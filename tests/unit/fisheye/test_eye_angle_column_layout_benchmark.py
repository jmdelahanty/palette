from __future__ import annotations

from pathlib import Path

import numpy as np
import zarr

from fisheye.diagnostics.benchmark_eye_angle_column_layout import run_benchmark


def _encoded_names(names: list[str], width: int = 64) -> np.ndarray:
    values = np.zeros((len(names), width), dtype=np.uint8)
    for index, name in enumerate(names):
        encoded = name.encode("utf-8")
        values[index, : len(encoded)] = np.frombuffer(encoded, dtype=np.uint8)
    return values


def test_eye_angle_column_layout_benchmark_validates_values_by_name(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "source.zarr"
    output_path = tmp_path / "benchmark"
    root = zarr.open_group(str(source_path), mode="w", zarr_format=3)
    parent = root.create_group("analysis/eye_angle_runs")
    parent.attrs["latest_complete"] = "eye_1"
    run = parent.create_group("eye_1")
    run.attrs["palette_run_completion_status"] = "complete"
    names = [
        "heading_deg",
        "left_eye_angle_deg",
        "left_gaze_signed_deg",
        "right_eye_angle_deg",
        "right_gaze_signed_deg",
        "vergence_eye_angle_deg",
        "vergence_gaze_deg",
        "version_deg",
        "left_centroid_deg",
    ]
    values = np.arange(108, dtype=np.float32).reshape(12, 9)
    run.create_array("frame_angles", data=values, chunks=(6, 9))
    index = run.create_group("angle_channel_index")
    index.create_array("name", data=_encoded_names(names), chunks=(9, 64))

    report = run_benchmark(
        source_path,
        output_root=output_path,
        max_rows=10,
        narrow_rows=4,
        repeats=1,
    )

    assert report["source_access"] == "read_only"
    assert report["benchmarked_rows"] == 10
    assert len(report["candidates"]) == 3
    assert all(item["exact_values_by_name"] for item in report["candidates"])
    recommended = next(
        item
        for item in report["candidates"]
        if item["candidate"]["name"] == "recommended_semantic_8"
    )
    assert recommended["layout"]["chunks"] == [10, 8]
    assert recommended["workloads"]["narrow_common_three"]["channel_count"] == 3
    assert recommended["workloads"]["full_duration_common_three"]["rows"] == 10
    assert (output_path / "report.json").is_file()
