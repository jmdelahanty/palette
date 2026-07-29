from __future__ import annotations

import json

from fisheye.shared.zarr.crop_schema import CropDimensions
from fisheye.shared.zarr.crop_storage import (
    CROP_GEOMETRY_STORAGE_SCHEMA_ID,
    plan_crop_geometry_storage,
)


MIB = 1024 * 1024


def _representative_dimensions() -> CropDimensions:
    return CropDimensions(
        n_frames=1_188_000,
        n_instances=1_187_087,
        source_width=4512,
        source_height=4512,
    )


def test_all_crop_arrays_use_one_immutable_byte_planned_policy() -> None:
    plan_set = plan_crop_geometry_storage(_representative_dimensions())

    assert len(plan_set.entries) == 13
    for entry in plan_set.entries:
        assert entry.plan.write_mode == "immutable"
        assert entry.plan.profile_id == "published_http_v1"
        assert entry.plan.access_unit_shape == (
            1,
            *entry.plan.logical_shape[1:],
        )
    by_path = {entry.rule.path: entry for entry in plan_set.entries}
    assert by_path["frame_row_offsets"].plan.access_pattern == "eager"
    assert all(
        entry.plan.access_pattern == "windowed"
        for path, entry in by_path.items()
        if path != "frame_row_offsets"
    )


def test_chunk_rows_derive_from_actual_bytes_per_row() -> None:
    plan_set = plan_crop_geometry_storage(_representative_dimensions())
    by_path = {entry.rule.path: entry.plan for entry in plan_set.entries}

    assert by_path["instance_key"].chunk_shape == (131_072,)
    assert by_path["frame_indices"].chunk_shape == (131_072,)
    assert by_path["bbox_norm_coords"].chunk_shape == (65_536, 4)
    assert by_path["roi_coordinates_full"].chunk_shape == (131_072, 2)
    assert by_path["source_row_signature"].chunk_shape == (32_768, 32)
    assert by_path["frame_row_offsets"].chunk_shape == (131_072,)
    assert {entry.plan.chunk_nbytes for entry in plan_set.entries} == {MIB}


def test_shards_reduce_objects_and_preserve_complete_rows() -> None:
    plan_set = plan_crop_geometry_storage(_representative_dimensions())

    for entry in plan_set.entries:
        plan = entry.plan
        assert plan.shard_axes == (0,)
        assert plan.chunk_shape is not None
        assert plan.chunk_shape[1:] == plan.logical_shape[1:]
        assert plan.shard_shape is not None
        assert all(
            shard % chunk == 0
            for shard, chunk in zip(plan.shard_shape, plan.chunk_shape)
        )
        assert plan.write_ownership == "whole_shard_single_writer"
    assert plan_set.estimated_inner_chunk_count == 193
    assert plan_set.estimated_payload_objects == 14
    assert plan_set.estimated_array_metadata_objects == 13
    assert plan_set.estimated_stage_objects == 29


def test_small_crop_fixture_collapses_to_one_object_per_array() -> None:
    plan_set = plan_crop_geometry_storage(
        CropDimensions(
            n_frames=4,
            n_instances=6,
            source_width=640,
            source_height=480,
        )
    )

    assert all(not entry.plan.is_sharded for entry in plan_set.entries)
    assert all(entry.plan.estimated_payload_objects == 1 for entry in plan_set.entries)
    assert plan_set.estimated_payload_objects == 13
    assert plan_set.estimated_stage_objects == 28


def test_storage_manifest_is_json_safe_and_has_no_pixel_array() -> None:
    plan_set = plan_crop_geometry_storage(_representative_dimensions())
    manifest = plan_set.as_manifest()

    assert json.loads(json.dumps(manifest)) == manifest
    assert manifest["schema_id"] == CROP_GEOMETRY_STORAGE_SCHEMA_ID
    assert manifest["logical_stage_schema"] == {
        "id": "palette.stage.crop_geometry",
        "version": 1,
    }
    assert manifest["storage_profile"]["profile_id"] == "published_http_v1"
    assert manifest["storage_profile"]["target_chunk_bytes"] == MIB
    assert manifest["object_estimate"]["stage_objects"] == 29
    assert "roi_images" not in {item["path"] for item in manifest["arrays"]}
    assert manifest["write_partition_contract"]["partial_physical_unit_writes"] == (
        "forbidden"
    )
