from __future__ import annotations

import json

from fisheye.shared.zarr.refined_detection_schema import (
    RefinedDetectionDimensions,
    RefinedDetectionLineageProfile,
)
from fisheye.shared.zarr.refined_detection_storage import (
    REFINED_DETECTION_ACCESS_AWARE_CANDIDATE_V1,
    REFINED_DETECTION_REGULAR_CONTROL_V1,
    REFINED_DETECTION_STORAGE_SCHEMA_ID,
    plan_refined_detection_storage,
)
from fisheye.shared.zarr.storage_profiles import (
    DETECTION_PUBLISHED_ACCESS_AWARE_V1,
    DETECTION_REGULAR_ROLLBACK_V1,
)


MIB = 1024 * 1024


def _dimensions(
    *,
    clipped: bool = False,
) -> RefinedDetectionDimensions:
    return RefinedDetectionDimensions(
        n_frames=1_188_000,
        n_instances=1_187_087,
        n_source_detections=1_187_087,
        source_width=4512,
        source_height=4512,
        lineage_profile=(
            RefinedDetectionLineageProfile.CLIPPED_RECORDING_SNAPSHOT
            if clipped
            else RefinedDetectionLineageProfile.FULL_ACQUISITION
        ),
    )


def test_every_binding_has_one_immutable_access_rule() -> None:
    plans = plan_refined_detection_storage(_dimensions())

    assert len(plans.entries) == 28
    for entry in plans.entries:
        assert entry.plan.write_mode == "immutable"
        if entry.rule.path.endswith("/frame_row_offsets"):
            assert entry.plan.access_pattern == "eager"
        elif entry.rule.path.startswith("source_detections/"):
            assert entry.plan.access_pattern == "indexed"
        else:
            assert entry.plan.access_pattern == "windowed"


def test_chunk_rows_derive_from_uncompressed_bytes_and_complete_rows() -> None:
    plans = plan_refined_detection_storage(_dimensions())
    by_path = {entry.rule.path: entry.plan for entry in plans.entries}

    assert by_path["instances/frame_indices"].chunk_shape == (32_768,)
    assert by_path["instances/refined_row_ids"].chunk_shape == (16_384,)
    assert by_path["instances/bbox_norm_coords"].chunk_shape == (8_192, 4)
    assert by_path["instances/score_valid"].chunk_shape == (131_072,)
    assert by_path["instances/reason_codes"].chunk_shape == (65_536,)
    assert by_path["instances/frame_row_offsets"].chunk_shape == (131_072,)
    assert by_path["source_detections/bbox_norm_coords"].chunk_shape == (8_192, 4)
    assert {entry.plan.chunk_nbytes for entry in plans.entries} == {
        128 * 1024,
        MIB,
    }
    for entry in plans.entries:
        assert entry.plan.chunk_shape is not None
        assert entry.plan.chunk_shape[1:] == entry.plan.logical_shape[1:]


def test_large_snapshot_is_sharded_with_whole_shard_writer_ownership() -> None:
    plans = plan_refined_detection_storage(_dimensions())

    assert all(entry.plan.is_sharded for entry in plans.entries)
    assert all(entry.plan.shard_axes == (0,) for entry in plans.entries)
    assert all(
        entry.plan.write_ownership == "whole_shard_single_writer"
        for entry in plans.entries
    )
    assert plans.estimated_payload_objects == 48
    for entry in plans.entries:
        assert entry.plan.shard_shape is not None
        assert entry.plan.chunk_shape is not None
        assert all(
            shard % chunk == 0
            for shard, chunk in zip(entry.plan.shard_shape, entry.plan.chunk_shape)
        )


def test_small_snapshot_stays_one_unsharded_object_per_array() -> None:
    plans = plan_refined_detection_storage(
        RefinedDetectionDimensions(
            n_frames=4,
            n_instances=3,
            n_source_detections=3,
            source_width=640,
            source_height=480,
        )
    )

    assert all(not entry.plan.is_sharded for entry in plans.entries)
    assert all(entry.plan.estimated_payload_objects == 1 for entry in plans.entries)
    assert plans.estimated_payload_objects == 28
    assert plans.estimated_stage_objects == 59


def test_clipped_lineage_profile_extends_the_same_planner_contract() -> None:
    plans = plan_refined_detection_storage(_dimensions(clipped=True))

    assert len(plans.entries) == 38
    paths = {entry.rule.path for entry in plans.entries}
    assert "instances/source_clip_indices" in paths
    assert "source_detections/source_clip_detect_row_index" in paths


def test_storage_manifest_freezes_codec_metadata_and_consolidation_gate() -> None:
    plans = plan_refined_detection_storage(_dimensions())
    manifest = plans.as_manifest()

    assert json.loads(json.dumps(manifest)) == manifest
    assert manifest["schema_id"] == REFINED_DETECTION_STORAGE_SCHEMA_ID
    assert manifest["logical_stage_schema"] == {
        "id": "palette.stage.refined_detection",
        "version": 1,
    }
    assert manifest["storage_profile"]["profile_id"] == (
        "detection_published_access_aware_v1"
    )
    assert manifest["storage_profile_role"] == (
        "promoted_detection_snapshot_default"
    )
    assert manifest["codec_profile"]["zarr_format"] == 3
    assert manifest["codec_profile"]["codec_chain"] == [
        {"name": "bytes", "configuration": {"endian": "little"}},
        {
            "name": "zstd",
            "configuration": {"level": 0, "checksum": False},
        },
    ]
    assert manifest["codec_profile"]["sharding_index"]["location"] == "end"
    assert manifest["metadata_open_contract"][
        "direct_consolidated_equivalence"
    ] == "required_before_visibility"
    assert manifest["profile_status"] == "promoted_production_default"


def test_access_aware_candidate_is_exact_and_requires_explicit_selection() -> None:
    baseline = plan_refined_detection_storage(_dimensions())
    candidate = plan_refined_detection_storage(
        _dimensions(),
        profile=REFINED_DETECTION_ACCESS_AWARE_CANDIDATE_V1,
    )
    by_path = {entry.rule.path: entry.plan for entry in candidate.entries}

    assert baseline.profile == DETECTION_PUBLISHED_ACCESS_AWARE_V1
    assert candidate.profile.profile_id != baseline.profile.profile_id
    assert candidate.profile.target_chunk_bytes == 128 * 1024
    assert candidate.profile.target_shard_bytes == 8 * MIB
    assert dict(candidate.profile.target_chunk_bytes_by_access) == {
        "eager": MIB,
    }
    assert by_path["instances/bbox_norm_coords"].chunk_shape == (8_192, 4)
    assert by_path["instances/frame_indices"].chunk_shape == (32_768,)
    assert by_path["instances/frame_row_offsets"].chunk_shape == (131_072,)
    assert by_path["source_detections/bbox_norm_coords"].chunk_nbytes == 128 * 1024
    assert by_path["instances/frame_row_offsets"].chunk_nbytes == MIB
    assert all(
        entry.plan.shard_nbytes is None or entry.plan.shard_nbytes <= 8 * MIB
        for entry in candidate.entries
    )
    assert candidate.as_manifest()["profile_status"] == (
        "resolved_plan_evidence_not_a_production_default_promotion"
    )
    assert candidate.as_manifest()["storage_profile_role"] == (
        "unpromoted_access_aware_candidate"
    )


def test_promoted_profile_is_physically_unchanged_from_evidence_candidate() -> None:
    promoted = plan_refined_detection_storage(_dimensions())
    candidate = plan_refined_detection_storage(
        _dimensions(),
        profile=REFINED_DETECTION_ACCESS_AWARE_CANDIDATE_V1,
    )

    assert [
        (entry.plan.chunk_shape, entry.plan.shard_shape, entry.plan.codec_profile_id)
        for entry in promoted.entries
    ] == [
        (entry.plan.chunk_shape, entry.plan.shard_shape, entry.plan.codec_profile_id)
        for entry in candidate.entries
    ]
    assert promoted.profile.profile_id == "detection_published_access_aware_v1"
    assert candidate.profile.profile_id != promoted.profile.profile_id


def test_regular_rollback_is_explicit_and_matches_the_control_layout() -> None:
    rollback = plan_refined_detection_storage(
        _dimensions(),
        profile=DETECTION_REGULAR_ROLLBACK_V1,
    )
    control = plan_refined_detection_storage(
        _dimensions(),
        profile=REFINED_DETECTION_REGULAR_CONTROL_V1,
    )

    assert [
        (entry.plan.chunk_shape, entry.plan.shard_shape, entry.plan.codec_profile_id)
        for entry in rollback.entries
    ] == [
        (entry.plan.chunk_shape, entry.plan.shard_shape, entry.plan.codec_profile_id)
        for entry in control.entries
    ]
    assert rollback.as_manifest()["storage_profile_role"] == (
        "explicit_detection_snapshot_rollback"
    )
    assert rollback.as_manifest()["profile_status"] == (
        "available_only_by_explicit_rollback"
    )


def test_paired_gate_regular_control_is_genuinely_unsharded() -> None:
    control = plan_refined_detection_storage(
        _dimensions(),
        profile=REFINED_DETECTION_REGULAR_CONTROL_V1,
    )
    candidate = plan_refined_detection_storage(
        _dimensions(),
        profile=REFINED_DETECTION_ACCESS_AWARE_CANDIDATE_V1,
    )

    assert control.profile.target_chunk_bytes == MIB
    assert not control.profile.shard_immutable
    assert all(not entry.plan.is_sharded for entry in control.entries)
    assert all(entry.plan.chunk_nbytes == MIB for entry in control.entries)
    assert control.as_manifest()["storage_profile_role"] == (
        "paired_unsharded_control"
    )
    assert (
        control.estimated_payload_objects
        > 4 * candidate.estimated_payload_objects
    )
