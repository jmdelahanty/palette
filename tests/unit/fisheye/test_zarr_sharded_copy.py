from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import zarr

from fisheye.shared.zarr_sharded_copy import (
    SHARD_POLICY_MULTI_CHUNK_CAPPED,
    ShardedArrayLayout,
    copy_completed_run_to_sharded,
)


@pytest.mark.parametrize("workers", [1, 2])
def test_copy_completed_run_to_sharded_preserves_values_and_owns_outer_shards(
    tmp_path: Path,
    workers: int,
) -> None:
    source_path = tmp_path / "source-run"
    destination_path = tmp_path / "destination-run"
    source = zarr.open_group(str(source_path), mode="w", zarr_format=3)
    source.attrs.update(
        {
            "schema_id": "fixture.run",
            "palette_run_completion_status": "complete",
            "custom": {"preserved": True},
        }
    )
    row_index = source.create_group("row_index")
    row_index.create_array(
        "frame_indices",
        data=np.arange(10, dtype=np.int32),
        chunks=(2,),
    )
    values = np.arange(30, dtype=np.float32).reshape(10, 3)
    source.create_array("values", data=values, chunks=(2, 3), fill_value=np.nan)
    source.create_array("sample_s", data=np.linspace(0.0, 1.0, 3), chunks=(3,))

    report = copy_completed_run_to_sharded(
        source_path,
        destination_path,
        row_count_array="row_index/frame_indices",
        shard_rows=5,
        workers=workers,
    )

    assert report["status"] == "complete"
    assert report["exact_decoded_validation"] is True
    assert report["duration_seconds"] >= 0.0
    assert report["decoded_mib_per_second"] > 0.0
    assert report["worker_ownership"] == (
        "one_complete_nonoverlapping_outer_row_shard_per_array_task"
    )
    destination = zarr.open_group(str(destination_path), mode="r", use_consolidated=False)
    np.testing.assert_array_equal(np.asarray(destination["values"][:]), values)
    assert tuple(destination["values"].chunks) == (2, 3)
    assert tuple(destination["values"].shards) == (6, 3)
    assert destination.attrs["custom"] == {"preserved": True}
    assert destination.attrs["physical_storage_layout"]["requested_outer_shard_rows"] == 5


def test_copy_completed_run_to_sharded_supports_different_track_lengths(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "multi-track-source"
    destination_path = tmp_path / "multi-track-destination"
    source = zarr.open_group(str(source_path), mode="w", zarr_format=3)
    source.attrs["palette_run_completion_status"] = "complete"
    tracks = source.create_group("tracks")
    first = tracks.create_group("id_0")
    second = tracks.create_group("id_1")
    first_values = np.arange(20, dtype=np.float32)
    second_values = np.arange(14, dtype=np.float32)
    first.create_array("speed", data=first_values, chunks=(4,))
    second.create_array("speed", data=second_values, chunks=(2,))

    report = copy_completed_run_to_sharded(
        source_path,
        destination_path,
        row_count_array=None,
        shard_rows=9,
        workers=2,
    )

    destination = zarr.open_group(str(destination_path), mode="r", use_consolidated=False)
    np.testing.assert_array_equal(destination["tracks/id_0/speed"][:], first_values)
    np.testing.assert_array_equal(destination["tracks/id_1/speed"][:], second_values)
    assert tuple(destination["tracks/id_0/speed"].shards) == (12,)
    assert tuple(destination["tracks/id_1/speed"].shards) == (10,)
    assert report["row_count_array"] is None
    assert report["row_aligned_array_count"] == 2
    assert destination.attrs["physical_storage_layout"]["eligibility"] == (
        "all_arrays_with_a_first_axis"
    )


@pytest.mark.parametrize("workers", [1, 2])
def test_copy_completed_run_keeps_structured_lineage_single_chunk(
    tmp_path: Path,
    workers: int,
) -> None:
    source_path = tmp_path / "structured-source"
    destination_path = tmp_path / "structured-destination"
    source = zarr.open_group(str(source_path), mode="w", zarr_format=3)
    source.attrs["palette_run_completion_status"] = "complete"
    dtype = np.dtype([("valid", "?"), ("instance_key", "<u8")])
    values = np.zeros(10, dtype=dtype)
    values[3] = (True, 101)
    source.create_array("source_instance_key", data=values, chunks=(2,))

    report = copy_completed_run_to_sharded(
        source_path,
        destination_path,
        row_count_array=None,
        shard_rows=5,
        workers=workers,
    )

    destination = zarr.open_group(
        str(destination_path),
        mode="r",
        use_consolidated=False,
    )
    copied = destination["source_instance_key"]
    np.testing.assert_array_equal(copied[:], values)
    assert tuple(copied.chunks) == (10,)
    assert copied.shards is None
    plan = next(item for item in report["arrays"] if item["path"] == "source_instance_key")
    assert plan["layout_profile"] == (
        "structured_dtype_single_chunk_zarr_v3_sharding_codec_workaround_v1"
    )


def test_copy_completed_run_to_sharded_applies_two_dimensional_layout_override(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "column-source"
    destination_path = tmp_path / "column-destination"
    source = zarr.open_group(str(source_path), mode="w", zarr_format=3)
    source.attrs["palette_run_completion_status"] = "complete"
    values = np.arange(70, dtype=np.float32).reshape(10, 7)
    source.create_array("frame_angles", data=values, chunks=(5, 7))

    report = copy_completed_run_to_sharded(
        source_path,
        destination_path,
        row_count_array=None,
        shard_rows=9,
        array_layouts={
            "frame_angles": ShardedArrayLayout(
                inner_chunks=(3, 2),
                outer_shards=(7, 5),
                layout_profile="fixture.semantic_columns.v1",
            )
        },
        workers=1,
    )

    destination = zarr.open_group(str(destination_path), mode="r", use_consolidated=False)
    np.testing.assert_array_equal(destination["frame_angles"][:], values)
    assert tuple(destination["frame_angles"].chunks) == (3, 2)
    assert tuple(destination["frame_angles"].shards) == (9, 6)
    plan = next(item for item in report["arrays"] if item["path"] == "frame_angles")
    assert tuple(plan["source_chunks"]) == (5, 7)
    assert tuple(plan["requested_inner_chunks"]) == (3, 2)
    assert tuple(plan["requested_outer_shards"]) == (7, 5)
    assert plan["layout_profile"] == "fixture.semantic_columns.v1"
    persisted = destination.attrs["physical_storage_layout"]
    assert persisted["array_layout_overrides"]["frame_angles"] == {
        "inner_chunks": [3, 2],
        "outer_shards": [7, 5],
        "layout_profile": "fixture.semantic_columns.v1",
    }


def test_copy_completed_run_to_sharded_supports_small_run_columnar_policy(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "columnar-source"
    destination_path = tmp_path / "columnar-destination"
    source = zarr.open_group(str(source_path), mode="w", zarr_format=3)
    source.attrs["palette_run_completion_status"] = "complete"
    long_values = np.arange(10, dtype=np.float32)
    short_values = np.arange(2, dtype=np.int16)
    source.create_array("long_values", data=long_values, chunks=(2,))
    source.create_array("short_values", data=short_values, chunks=(2,))

    report = copy_completed_run_to_sharded(
        source_path,
        destination_path,
        row_count_array=None,
        shard_rows=100,
        shard_policy=SHARD_POLICY_MULTI_CHUNK_CAPPED,
        workers=1,
    )

    destination = zarr.open_group(
        str(destination_path), mode="r", use_consolidated=False
    )
    assert tuple(destination["long_values"].chunks) == (2,)
    assert tuple(destination["long_values"].shards) == (10,)
    assert destination["short_values"].shards is None
    assert report["sharded_array_count"] == 1
    assert report["regular_array_count"] == 1
    assert report["shard_policy"] == SHARD_POLICY_MULTI_CHUNK_CAPPED
    assert destination.attrs["physical_storage_layout"]["eligibility"] == (
        "all_arrays_with_multiple_logical_row_chunks"
    )
