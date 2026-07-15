from __future__ import annotations

from pathlib import Path

import numpy as np
import zarr

from fisheye.diagnostics.benchmark_tail_kinematics_sharding import build_plan, run_benchmark


def _write_source(path: Path) -> None:
    row_count = 24
    root = zarr.open_group(str(path), mode="w", zarr_format=3)
    root.attrs.update(
        {
            "palette_run_completion_status": "complete",
            "schema_id": "analysis.tail_kinematics_runs",
        }
    )
    root.create_array(
        "frame_index",
        data=np.arange(row_count, dtype=np.int64),
        chunks=(4,),
        shards=(8,),
        overwrite=True,
    )
    root.create_array(
        "valid",
        data=np.asarray([True, False] * (row_count // 2), dtype=bool),
        chunks=(4,),
        shards=(8,),
        overwrite=True,
    )
    root.create_array(
        "tail_tip_angle_deg",
        data=np.linspace(-20.0, 20.0, row_count, dtype=np.float32),
        chunks=(4,),
        shards=(8,),
        overwrite=True,
    )
    angles = np.arange(row_count * 3, dtype=np.float32).reshape(row_count, 3)
    angles[3, 1] = np.nan
    root.create_array(
        "tail_angle_deg",
        data=angles,
        chunks=(4, 3),
        shards=(8, 3),
        overwrite=True,
    )
    root.create_array(
        "tail_angle_sample_xy",
        data=np.arange(row_count * 3 * 2, dtype=np.float32).reshape(row_count, 3, 2),
        chunks=(4, 3, 2),
        shards=(8, 3, 2),
        overwrite=True,
    )
    root.create_array(
        "tail_angle_sample_s",
        data=np.linspace(0.0, 1.0, 3, dtype=np.float32),
        chunks=(3,),
        overwrite=True,
    )
    row_index = root.create_group("row_index")
    row_index.attrs["row_semantics"] = "test_rows"
    row_index.create_array(
        "frame_indices",
        data=np.arange(row_count, dtype=np.int64),
        chunks=(4,),
        shards=(8,),
        overwrite=True,
    )


def test_build_plan_only_changes_outer_shards_for_row_aligned_arrays(tmp_path: Path) -> None:
    source = tmp_path / "tail_run"
    _write_source(source)

    plans = {plan.path: plan for plan in build_plan(source, shard_rows=7)}

    assert plans["tail_angle_deg"].inner_chunks == (4, 3)
    assert plans["tail_angle_deg"].outer_shards == (8, 3)
    assert plans["row_index/frame_indices"].outer_shards == (8,)
    assert plans["tail_angle_sample_s"].row_aligned is False
    assert plans["tail_angle_sample_s"].outer_shards is None


def test_benchmark_clones_variants_exactly_with_parallel_shard_ownership(tmp_path: Path) -> None:
    source = tmp_path / "tail_run"
    output = tmp_path / "variants"
    report_path = tmp_path / "benchmark.json"
    transfer_root = tmp_path / "transfer"
    _write_source(source)
    source_metadata_before = (source / "zarr.json").read_bytes()

    report = run_benchmark(
        source,
        output_root=output,
        shard_rows=(8, 16),
        workers=2,
        read_repeats=1,
        random_rows=3,
        window_rows=4,
        window_count=2,
        scan_rows=8,
        digest_rows=8,
        report_path=report_path,
        transfer_root=transfer_root,
        apply=True,
    )

    assert report["status"] == "complete"
    assert report["all_variants_exact"] is True
    assert report_path.exists()
    assert (source / "zarr.json").read_bytes() == source_metadata_before
    assert len(report["variants"]) == 2
    assert all(variant["worker_count"] == 2 for variant in report["variants"])
    assert all(variant["worker_task_count"] > 0 for variant in report["variants"])
    assert all(variant["all_arrays_exact"] for variant in report["variants"])
    assert report["transfer_benchmark_status"] == "complete"
    assert not transfer_root.exists()
    assert all(
        variant["transfer_benchmark"]["physical_files_exact"]
        for variant in report["variants"]
    )
    assert all(
        variant["transfer_benchmark"]["removed_after_validation"]
        for variant in report["variants"]
    )

    source_root = zarr.open_group(str(source), mode="r", use_consolidated=False)
    wider = zarr.open_group(
        str(output / "shard_rows_16.zarr"),
        mode="r",
        use_consolidated=False,
    )
    assert wider.attrs["benchmark_only"] is True
    assert wider["tail_angle_deg"].chunks == source_root["tail_angle_deg"].chunks
    assert wider["tail_angle_deg"].shards == (16, 3)
    assert wider["row_index"].attrs["row_semantics"] == "test_rows"
    np.testing.assert_array_equal(
        wider["tail_angle_deg"][:],
        source_root["tail_angle_deg"][:],
    )
