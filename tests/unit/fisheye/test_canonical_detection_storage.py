from __future__ import annotations

import json

from fisheye.shared.zarr.detection_schema import CanonicalDetectionDimensions
from fisheye.shared.zarr.detection_storage import (
    CANONICAL_DETECTION_STORAGE_SCHEMA_ID,
    plan_canonical_detection_storage,
)


MIB = 1024 * 1024


def _representative_dimensions() -> CanonicalDetectionDimensions:
    return CanonicalDetectionDimensions(
        n_frames=1_188_000,
        n_instances=1_187_087,
        source_width=4512,
        source_height=4512,
    )


def test_every_canonical_array_has_one_immutable_access_rule() -> None:
    plan_set = plan_canonical_detection_storage(_representative_dimensions())

    assert tuple(entry.rule.path for entry in plan_set.entries) == (
        "instances/frame_indices",
        "instances/source_acquisition_frame_index",
        "instances/instance_key",
        "instances/bbox_norm_coords",
        "instances/bbox_img_xyxy",
        "instances/centers_img_xy",
        "instances/scores",
        "instances/class_ids",
        "instances/frame_row_offsets",
    )
    for entry in plan_set.entries[:-1]:
        assert entry.plan.access_pattern == "windowed"
        assert entry.plan.write_mode == "immutable"
        assert entry.rule.access_unit_semantics == (
            "one_complete_detection_instance_row"
        )
    offsets = plan_set.entries[-1]
    assert offsets.plan.access_pattern == "eager"
    assert offsets.rule.representative_request == (
        "whole_index_or_two_adjacent_frame_boundaries"
    )


def test_representative_chunks_derive_row_depth_from_bytes() -> None:
    plan_set = plan_canonical_detection_storage(_representative_dimensions())
    by_path = {entry.rule.path: entry.plan for entry in plan_set.entries}

    assert by_path["instances/frame_indices"].chunk_shape == (262_144,)
    assert by_path["instances/source_acquisition_frame_index"].chunk_shape == (131_072,)
    assert by_path["instances/bbox_norm_coords"].chunk_shape == (65_536, 4)
    assert by_path["instances/centers_img_xy"].chunk_shape == (131_072, 2)
    assert by_path["instances/frame_row_offsets"].chunk_shape == (131_072,)
    assert {entry.plan.chunk_nbytes for entry in plan_set.entries} == {MIB}


def test_representative_outer_shards_and_object_estimate() -> None:
    plan_set = plan_canonical_detection_storage(_representative_dimensions())
    by_path = {entry.rule.path: entry.plan for entry in plan_set.entries}

    assert by_path["instances/frame_indices"].shard_shape == (1_310_720,)
    assert by_path["instances/bbox_norm_coords"].shard_shape == (1_245_184, 4)
    assert by_path["instances/frame_row_offsets"].shard_shape == (1_310_720,)
    assert all(entry.plan.is_sharded for entry in plan_set.entries)
    assert all(entry.plan.estimated_shard_count == 1 for entry in plan_set.entries)
    assert plan_set.estimated_logical_nbytes == 90_225_924
    assert plan_set.estimated_inner_chunk_count == 93
    assert plan_set.estimated_payload_objects == 9
    assert plan_set.estimated_array_metadata_objects == 9
    assert plan_set.estimated_stage_objects == 20


def test_shards_preserve_whole_chunks_rows_and_single_writer_ownership() -> None:
    plan_set = plan_canonical_detection_storage(_representative_dimensions())

    for entry in plan_set.entries:
        plan = entry.plan
        assert plan.shard_axes == (0,)
        assert plan.access_unit_shape == (1, *plan.logical_shape[1:])
        assert plan.chunk_shape is not None
        assert plan.chunk_shape[1:] == plan.logical_shape[1:]
        assert plan.shard_shape is not None
        assert all(
            shard_axis % chunk_axis == 0
            for shard_axis, chunk_axis in zip(plan.shard_shape, plan.chunk_shape)
        )
        assert plan.write_ownership == "whole_shard_single_writer"


def test_small_arrays_remain_single_unsharded_objects() -> None:
    plan_set = plan_canonical_detection_storage(
        CanonicalDetectionDimensions(
            n_frames=4,
            n_instances=6,
            source_width=640,
            source_height=480,
        )
    )

    assert all(not entry.plan.is_sharded for entry in plan_set.entries)
    assert all(entry.plan.estimated_payload_objects == 1 for entry in plan_set.entries)
    assert all(entry.plan.estimated_shard_count == 0 for entry in plan_set.entries)
    assert all(
        entry.plan.estimated_regular_chunk_objects == 1 for entry in plan_set.entries
    )
    assert plan_set.estimated_payload_objects == 9
    assert plan_set.estimated_stage_objects == 20


def test_plan_set_manifest_is_json_safe_and_schema_linked() -> None:
    plan_set = plan_canonical_detection_storage(_representative_dimensions())

    manifest = plan_set.as_manifest()
    assert json.loads(json.dumps(manifest)) == manifest
    assert manifest["schema_id"] == CANONICAL_DETECTION_STORAGE_SCHEMA_ID
    assert manifest["logical_stage_schema"] == {
        "id": "palette.stage.canonical_detection",
        "version": 1,
    }
    assert manifest["storage_profile"]["profile_id"] == "published_http_v1"
    assert manifest["storage_profile"]["target_chunk_bytes"] == MIB
    assert manifest["object_estimate"]["stage_objects"] == 20
    assert manifest["write_partition_contract"]["partial_physical_unit_writes"] == (
        "forbidden"
    )
