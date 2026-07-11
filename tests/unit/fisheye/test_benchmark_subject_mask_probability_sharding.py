from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import zarr

from fisheye.diagnostics.benchmark_subject_mask_probability_sharding import (
    build_probability_sharding_variant,
)
from fisheye.diagnostics.benchmark_filesystem import describe_filesystem


def _source_run(tmp_path: Path) -> Path:
    path = tmp_path / "source.zarr"
    root = zarr.open_group(str(path), mode="w", zarr_format=3)
    run = root.create_group("subject_mask_shard_runs").create_group("run_001")
    values = np.arange(8 * 2 * 4 * 4, dtype=np.uint8).reshape(8, 2, 4, 4)
    run.create_array(
        "mask_probs_roi",
        data=values,
        chunks=(2, 1, 4, 4),
        overwrite=True,
    )
    run.attrs.update(
        {
            "mask_labels": ["subject_body", "eyes_union"],
            "probabilities_encoding": "linear_uint8_0_255",
        }
    )
    return path / "subject_mask_shard_runs" / "run_001"


@pytest.mark.parametrize(
    ("layout", "shard_rows", "expected_shards"),
    (("regular", None, None), ("sharded", 4, (4, 1, 4, 4))),
)
def test_build_probability_sharding_variant_exact_copy(
    tmp_path: Path,
    layout: str,
    shard_rows: int | None,
    expected_shards: tuple[int, ...] | None,
) -> None:
    source = _source_run(tmp_path)
    summary = build_probability_sharding_variant(
        source,
        output_root=tmp_path / "bench",
        layout=layout,
        shard_rows=shard_rows,
        sample_start=2,
        sample_rows=4,
        inner_chunk_rows=2,
        batch_rows=2,
        random_read_count=2,
    )

    assert summary["exact_match"] is True
    assert summary["shape"] == [4, 2, 4, 4]
    assert summary["shards"] == (list(expected_shards) if expected_shards else None)
    assert summary["destination_filesystem"]["storage_tier"] == "local"
    assert summary["source_filesystem"]["storage_tier"] == "local"
    assert summary["storage_inventory_seconds"] >= 0
    destination = zarr.open_group(summary["destination_zarr"], mode="r", use_consolidated=False)
    source_group = zarr.open_group(str(source), mode="r", use_consolidated=False)
    np.testing.assert_array_equal(
        destination["mask_probs_roi"][:],
        source_group["mask_probs_roi"][2:6],
    )
    manifest = json.loads(
        (Path(summary["destination_zarr"]).parent / f"{summary['variant']}.summary.json").read_text()
    )
    assert manifest["source_sha256"] == manifest["destination_sha256"]
    benchmark_set = json.loads(
        (Path(summary["destination_zarr"]).parent / "benchmark_set.json").read_text()
    )
    assert benchmark_set["variant_count"] == 1
    assert benchmark_set["all_exact_match"] is True
    assert benchmark_set["destination_storage_tiers"] == ["local"]


def test_build_probability_sharding_variant_rejects_misaligned_shard_rows(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="integer multiple"):
        build_probability_sharding_variant(
            _source_run(tmp_path),
            output_root=tmp_path / "bench",
            layout="sharded",
            shard_rows=3,
            sample_rows=4,
            inner_chunk_rows=2,
        )


def test_build_probability_sharding_variant_requires_declared_storage_tier(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="must be on storage tier 'prfs'"):
        build_probability_sharding_variant(
            _source_run(tmp_path),
            output_root=tmp_path / "bench",
            layout="regular",
            sample_rows=4,
            inner_chunk_rows=2,
            require_destination_storage_tier="prfs",
        )

    with pytest.raises(ValueError, match="Benchmark source must be on storage tier 'prfs'"):
        build_probability_sharding_variant(
            _source_run(tmp_path),
            output_root=tmp_path / "bench",
            layout="regular",
            sample_rows=4,
            inner_chunk_rows=2,
            require_source_storage_tier="prfs",
        )


@pytest.mark.parametrize(
    "mount_source",
    (
        "prfs.hhmi.org:/groups/johnson",
        "cluster.prfs.janelia.org:/groups3/johnson",
    ),
)
def test_describe_filesystem_identifies_prfs_mount(mount_source: str) -> None:
    mountinfo = "\n".join(
        (
            "25 1 8:2 / / rw,relatime - ext4 /dev/nvme0n1p2 rw",
            "36 25 0:32 / /groups/johnson rw,relatime - nfs4 "
            f"{mount_source} rw,vers=4.1",
        )
    )

    description = describe_filesystem(
        "/groups/johnson/johnsonlab/jeremy/benchmark",
        mountinfo_text=mountinfo,
    )

    assert description["storage_tier"] == "prfs"
    assert description["filesystem_type"] == "nfs4"
    assert description["mount_point"] == "/groups/johnson"
    assert description["mount_source"] == mount_source
