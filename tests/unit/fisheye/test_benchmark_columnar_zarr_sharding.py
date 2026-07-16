from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import zarr

from fisheye.diagnostics.benchmark_columnar_zarr_sharding import run_benchmark


def _metadata_snapshot(path: Path) -> dict[str, bytes]:
    return {
        str(candidate.relative_to(path)): candidate.read_bytes()
        for candidate in sorted(path.rglob("zarr.json"))
    }


def test_benchmark_clones_completed_run_with_aligned_shards(tmp_path):
    source_path = tmp_path / "source.zarr"
    source = zarr.open_group(str(source_path), mode="w", zarr_format=3)
    source.attrs.update(
        {
            "palette_run_completion_status": "complete",
            "scientific_contract": "preserved",
        }
    )
    table = source.require_group("events")
    table.attrs["table_contract"] = "preserved"
    values = np.arange(10_000, dtype=np.int64)
    value_array = table.create_array("value", data=values, chunks=(4_096,))
    value_array.attrs["units"] = "frames"
    table.create_array("short", data=np.arange(3, dtype=np.int16), chunks=(3,))
    before = _metadata_snapshot(source_path)

    report = run_benchmark(
        source_path,
        output_root=tmp_path / "benchmark",
        shard_rows=(7_000,),
        scan_rows=2_048,
        window_rows=128,
    )

    regular = zarr.open_group(
        report["variants"][0]["path"], mode="r", use_consolidated=False
    )
    sharded = zarr.open_group(
        report["variants"][1]["path"], mode="r", use_consolidated=False
    )
    assert regular["events/value"].shards is None
    assert sharded["events/value"].chunks == (4_096,)
    assert sharded["events/value"].shards == (8_192,)
    assert sharded["events/short"].shards is None
    assert sharded.attrs["scientific_contract"] == "preserved"
    assert sharded["events"].attrs["table_contract"] == "preserved"
    assert sharded["events/value"].attrs["units"] == "frames"
    np.testing.assert_array_equal(sharded["events/value"][:], values)
    assert report["source_metadata_unchanged"] is True
    assert _metadata_snapshot(source_path) == before
    assert all(row["validation"]["passed"] for row in report["variants"])
    assert all(row["bounded_windows"]["read_operations"] == 3 for row in report["variants"])
    regular_files = report["variants"][0]["storage"]["payload_file_count"]
    sharded_files = report["variants"][1]["storage"]["payload_file_count"]
    assert sharded_files < regular_files
    persisted = json.loads(Path(report["report_path"]).read_text(encoding="utf-8"))
    assert persisted["schema"] == "palette.columnar_zarr_sharding_benchmark.v1"


def test_benchmark_refuses_output_inside_source(tmp_path):
    source_path = tmp_path / "source.zarr"
    source = zarr.open_group(str(source_path), mode="w", zarr_format=3)
    source.attrs["palette_run_completion_status"] = "complete"
    source.create_array("value", data=np.arange(3), chunks=(3,))

    with pytest.raises(ValueError, match="must not be inside"):
        run_benchmark(
            source_path,
            output_root=source_path / "benchmark",
            shard_rows=(8,),
        )
