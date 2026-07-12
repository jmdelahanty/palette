from __future__ import annotations

from pathlib import Path

import numpy as np
import zarr

from fisheye.diagnostics.benchmark_keypoint_sharding import build_plan, run_canary


def _write_source(path: Path) -> None:
    root = zarr.open_group(str(path), mode="w", zarr_format=3)
    root.attrs.update(
        {
            "palette_run_completion_status": "complete",
            "keypoint_labels": ["head", "tail"],
        }
    )
    keypoints = np.arange(10 * 2 * 2, dtype=np.float64).reshape(10, 2, 2)
    keypoints[3, 0, 0] = np.nan
    root.create_array("keypoints_roi", data=keypoints, chunks=(4, 2, 2), overwrite=True)
    root.create_array(
        "confidence",
        data=np.linspace(0.0, 1.0, 10, dtype=np.float64),
        chunks=(4,),
        overwrite=True,
    )
    root.create_array(
        "detection_success",
        data=np.asarray([True, False] * 5, dtype=bool),
        chunks=(4,),
        overwrite=True,
    )
    frame_counts = np.zeros((25,), dtype=np.int32)
    frame_counts[:10] = 1
    root.create_array("frame_counts", data=frame_counts, chunks=(5,), overwrite=True)


def test_build_plan_aligns_outer_shards_to_each_domain_chunk_grid(tmp_path: Path) -> None:
    source = tmp_path / "source.zarr"
    _write_source(source)

    plans = {row.name: row for row in build_plan(source, roi_shard_rows=7, frame_shard_rows=11)}

    assert plans["keypoints_roi"].domain == "roi"
    assert plans["keypoints_roi"].outer_shards == (8, 2, 2)
    assert plans["confidence"].outer_shards == (8,)
    assert plans["frame_counts"].domain == "frame"
    assert plans["frame_counts"].outer_shards == (15,)


def test_canary_clones_all_arrays_exactly_into_indexed_shards(tmp_path: Path) -> None:
    source = tmp_path / "source.zarr"
    destination = tmp_path / "sharded.zarr"
    _write_source(source)

    report = run_canary(
        source,
        destination=destination,
        roi_shard_rows=8,
        frame_shard_rows=10,
        read_repeats=2,
        apply=True,
    )

    assert report["status"] == "complete"
    assert report["all_arrays_exact"] is True
    assert report["destination_storage"]["payload_file_count"] < report["source_storage"]["payload_file_count"]
    assert all(row["exact_match"] for row in report["array_results"])
    assert all(
        comparison["checksums_match"]
        for comparison in report["read_benchmark"]["comparisons"].values()
    )

    source_root = zarr.open_group(str(source), mode="r", use_consolidated=False)
    destination_root = zarr.open_group(str(destination), mode="r", use_consolidated=False)
    assert destination_root.attrs["benchmark_only"] is True
    assert destination_root.attrs["palette_run_completion_status"] == "complete"
    assert destination_root["keypoints_roi"].chunks == source_root["keypoints_roi"].chunks
    assert destination_root["keypoints_roi"].shards == (8, 2, 2)
    np.testing.assert_array_equal(
        destination_root["keypoints_roi"][:],
        source_root["keypoints_roi"][:],
    )
