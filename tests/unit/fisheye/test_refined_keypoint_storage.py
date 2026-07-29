from __future__ import annotations

from fisheye.shared.zarr.keypoint_schema import (
    REFINED_KEYPOINT_SCHEMA_V2,
    KeypointDimensions,
)
from fisheye.shared.zarr.refined_keypoint_storage import (
    REFINED_KEYPOINT_STORAGE_SCHEMA_ID,
    plan_refined_keypoint_storage,
)


def _dimensions(*, n_frames: int = 23_287, n_instances: int = 22_926):
    return KeypointDimensions(
        n_frames=n_frames,
        n_instances=n_instances,
        n_keypoints=3,
        source_width=4512,
        source_height=4512,
    )


def test_refined_storage_covers_exact_schema_and_preserves_complete_rows() -> None:
    plans = plan_refined_keypoint_storage(_dimensions())

    assert tuple(entry.rule.path for entry in plans.entries) == (
        REFINED_KEYPOINT_SCHEMA_V2.binding_paths
    )
    assert len(plans.entries) == 23
    for entry in plans.entries:
        plan = entry.plan
        assert plan.chunk_shape is not None
        assert plan.chunk_shape[1:] == tuple(
            max(1, value) for value in plan.logical_shape[1:]
        )
        assert plan.shard_axes == (0,)
        if plan.shard_shape is None:
            assert plan.write_ownership == "single_writer_immutable_materialization"
        else:
            assert plan.write_ownership == "whole_shard_single_writer"


def test_refined_storage_classifies_only_offsets_as_eager() -> None:
    plans = plan_refined_keypoint_storage(_dimensions())

    access = {entry.rule.path: entry.plan.access_pattern for entry in plans.entries}
    assert access["frame_row_offsets"] == "eager"
    assert set(access.values()) == {"eager", "windowed"}
    assert all(
        value == "windowed"
        for path, value in access.items()
        if path != "frame_row_offsets"
    )


def test_refined_storage_manifest_has_exact_object_accounting() -> None:
    plans = plan_refined_keypoint_storage(
        _dimensions(n_frames=1_188_000, n_instances=1_187_087)
    )
    manifest = plans.as_manifest()

    assert manifest["schema_id"] == REFINED_KEYPOINT_STORAGE_SCHEMA_ID
    assert manifest["logical_stage_schema"] == {
        "id": "palette.stage.refined_keypoints",
        "version": 2,
    }
    assert len(manifest["arrays"]) == 23
    estimate = manifest["object_estimate"]
    assert estimate["payload_objects"] == plans.estimated_payload_objects
    assert estimate["array_metadata_objects"] == 23
    assert estimate["group_metadata_objects"] == 2
    assert estimate["stage_objects"] == plans.estimated_stage_objects
