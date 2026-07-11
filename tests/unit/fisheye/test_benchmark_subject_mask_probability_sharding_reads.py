from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import zarr

from fisheye.diagnostics.benchmark_subject_mask_probability_sharding_reads import (
    benchmark_probability_sharding_reads,
)


def _variant(root: Path, name: str, values: np.ndarray, *, shards=None) -> dict[str, object]:
    path = root / f"{name}.zarr"
    group = zarr.open_group(str(path), mode="w", zarr_format=3)
    group.create_array(
        "mask_probs_roi",
        data=values,
        chunks=(2, 1, 4, 4),
        shards=shards,
        overwrite=True,
    )
    return {
        "variant": name,
        "destination_zarr": str(path),
        "exact_match": True,
    }


def test_benchmark_probability_sharding_reads_records_repeated_cold_and_warm_runs(tmp_path: Path) -> None:
    values = np.arange(8 * 2 * 4 * 4, dtype=np.uint8).reshape(8, 2, 4, 4)
    variants = [
        _variant(tmp_path, "regular", values),
        _variant(tmp_path, "sharded", values, shards=(4, 1, 4, 4)),
    ]
    (tmp_path / "benchmark_set.json").write_text(
        json.dumps({"variants": variants}),
        encoding="utf-8",
    )

    result = benchmark_probability_sharding_reads(
        tmp_path,
        repeats=2,
        batch_rows=2,
        component=0,
        random_seed=7,
        evict_cache=False,
    )

    assert result["repeats"] == 2
    assert result["benchmark_filesystem"]["storage_tier"] == "local"
    assert len(result["rounds"]) == 2
    assert {row["variant"] for row in result["variant_summaries"]} == {"regular", "sharded"}
    for summary in result["variant_summaries"]:
        assert summary["cold_mib_per_second"]["median"] > 0
        assert summary["warm_mib_per_second"]["median"] > 0
        assert summary["metadata_open_seconds"]["median"] >= 0
        assert summary["cache_eviction_seconds"]["median"] >= 0
    checksums = {
        measurement[pass_name]["checksum"]
        for round_payload in result["rounds"]
        for measurement in round_payload["measurements"]
        for pass_name in ("cold", "warm")
    }
    assert len(checksums) == 1
    assert next(iter(checksums)) > 0
    persisted = json.loads((tmp_path / "read_benchmark.json").read_text(encoding="utf-8"))
    assert persisted["schema_id"] == result["schema_id"]


def test_benchmark_probability_sharding_reads_requires_declared_storage_tier(
    tmp_path: Path,
) -> None:
    values = np.arange(8 * 2 * 4 * 4, dtype=np.uint8).reshape(8, 2, 4, 4)
    variants = [_variant(tmp_path, "regular", values)]
    (tmp_path / "benchmark_set.json").write_text(
        json.dumps({"variants": variants}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="must be on storage tier 'prfs'"):
        benchmark_probability_sharding_reads(
            tmp_path,
            repeats=1,
            batch_rows=2,
            component=0,
            evict_cache=False,
            require_storage_tier_name="prfs",
        )


def test_benchmark_probability_sharding_reads_supports_separate_output_path(
    tmp_path: Path,
) -> None:
    values = np.arange(8 * 2 * 4 * 4, dtype=np.uint8).reshape(8, 2, 4, 4)
    variants = [_variant(tmp_path, "regular", values)]
    (tmp_path / "benchmark_set.json").write_text(
        json.dumps({"variants": variants}),
        encoding="utf-8",
    )
    output_path = tmp_path / "reports" / "compute_read.json"

    benchmark_probability_sharding_reads(
        tmp_path,
        repeats=1,
        batch_rows=2,
        component=0,
        evict_cache=False,
        output_json=output_path,
    )

    assert output_path.is_file()
    assert not (tmp_path / "read_benchmark.json").exists()
