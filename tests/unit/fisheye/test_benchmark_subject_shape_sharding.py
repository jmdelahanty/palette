from __future__ import annotations

from pathlib import Path

import numpy as np
import zarr

from fisheye.diagnostics.benchmark_subject_shape_sharding import (
    REPORT_SCHEMA,
    ROW_COUNT_ARRAY,
)
from fisheye.diagnostics.benchmark_tail_kinematics_sharding import build_plan, run_benchmark


def _write_source(path: Path) -> None:
    rows = 24
    root = zarr.open_group(str(path), mode="w", zarr_format=3)
    root.attrs.update(
        {
            "palette_run_completion_status": "complete",
            "schema_id": "analysis.subject_shape_runs",
        }
    )
    row_index = root.create_group("row_index")
    row_index.create_array(
        "frame_indices",
        data=np.arange(rows, dtype=np.int64),
        chunks=(4,),
        overwrite=True,
    )
    body = root.create_group("components").create_group("subject_body")
    body.create_array(
        "centerline_xy",
        data=np.arange(rows * 3 * 2, dtype=np.float32).reshape(rows, 3, 2),
        chunks=(4, 3, 2),
        overwrite=True,
    )
    body.create_array(
        "tail_sample_s",
        data=np.linspace(0.0, 1.0, 3, dtype=np.float32),
        chunks=(3,),
        overwrite=True,
    )


def test_subject_shape_plan_uses_nested_row_count_array(tmp_path: Path) -> None:
    source = tmp_path / "shape.zarr"
    _write_source(source)

    plans = {
        plan.path: plan
        for plan in build_plan(
            source,
            shard_rows=7,
            row_count_array=ROW_COUNT_ARRAY,
            source_label="Subject-shape",
        )
    }

    assert plans["components/subject_body/centerline_xy"].outer_shards == (8, 3, 2)
    assert plans["row_index/frame_indices"].outer_shards == (8,)
    assert plans["components/subject_body/tail_sample_s"].outer_shards is None


def test_subject_shape_variants_preserve_all_decoded_arrays(tmp_path: Path) -> None:
    source = tmp_path / "shape.zarr"
    output = tmp_path / "variants"
    _write_source(source)

    report = run_benchmark(
        source,
        output_root=output,
        shard_rows=(8, 16),
        workers=2,
        read_repeats=1,
        read_arrays=("components/subject_body/centerline_xy",),
        random_rows=2,
        window_rows=4,
        window_count=1,
        scan_rows=8,
        digest_rows=8,
        row_count_array=ROW_COUNT_ARRAY,
        source_label="Subject-shape",
        report_schema=REPORT_SCHEMA,
        apply=True,
    )

    assert report["status"] == "complete"
    assert report["schema"] == REPORT_SCHEMA
    assert report["row_count_array"] == ROW_COUNT_ARRAY
    assert report["all_variants_exact"] is True
    assert all(item["all_arrays_exact"] for item in report["variants"])
    wider = zarr.open_group(str(output / "shard_rows_16.zarr"), mode="r")
    assert wider["components/subject_body/centerline_xy"].chunks == (4, 3, 2)
    assert wider["components/subject_body/centerline_xy"].shards == (16, 3, 2)
