from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest

from fisheye.shared.zarr.storage_intent import (
    AccessPattern,
    ArrayIntent,
    WriteMode,
)
from fisheye.shared.zarr.storage_planner import plan_storage
from fisheye.shared.zarr.storage_profiles import (
    EDITABLE_LOCAL_V1,
    PUBLISHED_HTTP_V1,
    TRAINING_IMMUTABLE_V1,
)
from fisheye.shared.zarr.storage_report import (
    compare_array_storage,
    compare_storage_layout,
)


MIB = 1024 * 1024


def test_narrow_timelines_derive_different_row_counts_from_bytes() -> None:
    frame_counts = plan_storage(
        ArrayIntent(
            name="frame_counts",
            shape=(1_000_000,),
            dtype=np.int32,
            access=AccessPattern.WINDOWED,
            write_mode=WriteMode.IMMUTABLE,
        ),
        PUBLISHED_HTTP_V1,
    )
    frame_offsets = plan_storage(
        ArrayIntent(
            name="frame_offsets",
            shape=(1_000_001,),
            dtype=np.int64,
            access=AccessPattern.WINDOWED,
            write_mode=WriteMode.IMMUTABLE,
        ),
        PUBLISHED_HTTP_V1,
    )

    assert frame_counts.chunk_shape == (262_144,)
    assert frame_counts.chunk_nbytes == MIB
    assert frame_counts.shard_shape == (1_048_576,)
    assert frame_counts.estimated_payload_objects == 1

    assert frame_offsets.chunk_shape == (131_072,)
    assert frame_offsets.chunk_nbytes == MIB
    assert frame_offsets.shard_shape == (1_048_576,)
    assert frame_offsets.estimated_payload_objects == 1


def test_keypoint_records_preserve_trailing_axes_and_target_chunk_bytes() -> None:
    plan = plan_storage(
        ArrayIntent(
            name="keypoints_img",
            shape=(1_000_000, 5, 2),
            dtype=np.float64,
            access=AccessPattern.WINDOWED,
            write_mode=WriteMode.IMMUTABLE,
        ),
        TRAINING_IMMUTABLE_V1,
    )

    assert plan.access_unit_nbytes == 80
    assert plan.chunk_shape == (16_384, 5, 2)
    assert plan.chunk_nbytes == 1_310_720
    assert plan.shard_shape == (425_984, 5, 2)
    assert plan.shard_nbytes == 34_078_720
    assert plan.estimated_payload_objects == 3
    assert plan.is_sharded


def test_boolean_status_uses_near_one_mib_chunk_for_representative_run() -> None:
    plan = plan_storage(
        ArrayIntent(
            name="success",
            shape=(1_180_000,),
            dtype=np.bool_,
            access=AccessPattern.WINDOWED,
            write_mode=WriteMode.IMMUTABLE,
        ),
        PUBLISHED_HTTP_V1,
    )

    assert plan.chunk_shape == (1_048_576,)
    assert plan.chunk_nbytes == MIB
    assert plan.shard_shape == (2_097_152,)
    assert plan.estimated_payload_objects == 1


def test_editable_mask_and_published_mask_share_access_unit_not_sharding() -> None:
    intent_fields = {
        "name": "masks_roi",
        "shape": (1_000_000, 4, 512, 512),
        "dtype": np.uint8,
        "access": AccessPattern.PER_ROW,
        "access_unit_shape": (1, 1, 512, 512),
    }
    editable = plan_storage(
        ArrayIntent(**intent_fields, write_mode=WriteMode.RANDOM_UPDATE),
        EDITABLE_LOCAL_V1,
    )
    published = plan_storage(
        ArrayIntent(**intent_fields, write_mode=WriteMode.IMMUTABLE),
        PUBLISHED_HTTP_V1,
    )

    assert editable.access_unit_nbytes == 256 * 1024
    assert editable.chunk_shape == (4, 1, 512, 512)
    assert editable.chunk_nbytes == MIB
    assert editable.shard_shape is None
    assert editable.estimated_payload_objects == 1_000_000
    assert not editable.object_budget_satisfied

    assert published.chunk_shape == editable.chunk_shape
    assert published.shard_shape == (1_024, 1, 512, 512)
    assert published.shard_nbytes == 256 * MIB
    assert published.estimated_payload_objects == 3_908
    assert published.object_budget_satisfied


def test_indexed_flat_values_are_planned_from_encoded_point_width() -> None:
    plan = plan_storage(
        ArrayIntent(
            name="points_xy",
            shape=(110_685_000, 2),
            dtype=np.float32,
            access=AccessPattern.INDEXED,
            write_mode=WriteMode.IMMUTABLE,
        ),
        PUBLISHED_HTTP_V1,
    )

    assert plan.access_unit_nbytes == 8
    assert plan.chunk_shape == (131_072, 2)
    assert plan.chunk_nbytes == MIB
    assert plan.shard_shape == (4_194_304, 2)
    assert plan.shard_nbytes == 32 * MIB
    assert plan.estimated_payload_objects == 27


def test_total_dataset_size_can_expand_shards_to_meet_object_budget() -> None:
    two_object_profile = replace(PUBLISHED_HTTP_V1, max_payload_objects=2)
    plan = plan_storage(
        ArrayIntent(
            name="large_status",
            shape=(100_000_000,),
            dtype=np.uint8,
            access=AccessPattern.WINDOWED,
            write_mode=WriteMode.IMMUTABLE,
        ),
        two_object_profile,
    )

    assert plan.chunk_shape == (1_048_576,)
    assert plan.shard_shape == (50_331_648,)
    assert plan.shard_nbytes == 48 * MIB
    assert plan.estimated_payload_objects == 2
    assert plan.object_budget_satisfied


def test_small_eager_array_stays_one_regular_object() -> None:
    plan = plan_storage(
        ArrayIntent(
            name="field_names",
            shape=(100,),
            dtype="S32",
            access=AccessPattern.EAGER,
            write_mode=WriteMode.IMMUTABLE,
        ),
        PUBLISHED_HTTP_V1,
    )

    assert plan.chunk_shape == (100,)
    assert plan.chunk_nbytes == 3_200
    assert plan.shard_shape is None
    assert plan.estimated_payload_objects == 1


def test_variable_width_representation_requires_encoded_size_estimate() -> None:
    with pytest.raises(ValueError, match="logical_itemsize_bytes"):
        ArrayIntent(
            shape=(1_000,),
            dtype=object,
            access=AccessPattern.WINDOWED,
            write_mode=WriteMode.IMMUTABLE,
        )

    intent = ArrayIntent(
        shape=(1_000,),
        dtype=object,
        logical_itemsize_bytes=64,
        access=AccessPattern.WINDOWED,
        write_mode=WriteMode.IMMUTABLE,
    )
    plan = plan_storage(intent, PUBLISHED_HTTP_V1)

    assert plan.logical_nbytes == 64_000
    assert plan.chunk_shape == (1_000,)


def test_append_only_sharding_requires_explicit_whole_shard_ownership() -> None:
    common = {
        "shape": (10_000_000,),
        "dtype": np.int64,
        "access": AccessPattern.WINDOWED,
        "write_mode": WriteMode.APPEND_ONLY,
    }
    ordinary_append = plan_storage(
        ArrayIntent(**common),
        PUBLISHED_HTTP_V1,
    )
    owned_append = plan_storage(
        ArrayIntent(**common, whole_shard_writes=True),
        PUBLISHED_HTTP_V1,
    )

    assert ordinary_append.shard_shape is None
    assert owned_append.shard_shape is not None
    assert owned_append.write_ownership == "whole_shard_single_writer"


def test_scalar_and_empty_arrays_have_deterministic_safe_plans() -> None:
    scalar = plan_storage(
        ArrayIntent(
            shape=(),
            dtype=np.float32,
            access=AccessPattern.EAGER,
            write_mode=WriteMode.IMMUTABLE,
        ),
        PUBLISHED_HTTP_V1,
    )
    empty = plan_storage(
        ArrayIntent(
            shape=(0,),
            dtype=np.int32,
            access=AccessPattern.WINDOWED,
            write_mode=WriteMode.IMMUTABLE,
        ),
        PUBLISHED_HTTP_V1,
    )

    assert scalar.chunk_shape is None
    assert scalar.estimated_payload_objects == 1
    assert empty.chunk_shape == (1,)
    assert empty.chunk_grid_shape == (0,)
    assert empty.estimated_payload_objects == 0


def test_plan_is_deterministic_and_contract_is_json_safe() -> None:
    intent = ArrayIntent(
        shape=(1_000_000, 5, 2),
        dtype=np.float64,
        access=AccessPattern.WINDOWED,
        write_mode=WriteMode.IMMUTABLE,
    )

    first = plan_storage(intent, PUBLISHED_HTTP_V1)
    second = plan_storage(intent, PUBLISHED_HTTP_V1)

    assert first == second
    payload = first.as_dict()
    assert json.loads(json.dumps(payload)) == payload
    assert payload["policy_version"] == "palette.storage_planner.v1"
    assert payload["logical_shape"] == [1_000_000, 5, 2]
    assert payload["logical_dtype"] == "float64"
    assert payload["access_unit_shape"] == [1, 5, 2]
    assert payload["growth_axis"] == 0
    assert payload["shard_axes"] == [0, 1, 2]
    assert payload["access_pattern"] == "windowed"
    assert payload["write_mode"] == "immutable"
    assert payload["estimated_shard_count"] == 3
    assert payload["estimated_regular_chunk_objects"] == 0


def test_read_only_report_compares_observed_and_proposed_layouts() -> None:
    intent = ArrayIntent(
        name="frame_counts",
        shape=(1_000_000,),
        dtype=np.int32,
        access=AccessPattern.WINDOWED,
        write_mode=WriteMode.IMMUTABLE,
    )
    comparison = compare_storage_layout(
        intent=intent,
        profile=PUBLISHED_HTTP_V1,
        actual_chunk_shape=(16_384,),
        actual_shard_shape=None,
    )

    assert comparison.chunk_shape_changes
    assert comparison.shard_shape_changes
    assert comparison.proposed.chunk_shape == (262_144,)
    assert comparison.as_dict()["actual_chunk_shape"] == [16_384]


def test_array_report_reads_layout_attributes_without_writing() -> None:
    class ReadOnlyArray:
        shape = (100,)
        dtype = np.dtype("S32")
        chunks = (100,)
        shards = None

        def __setattr__(self, name, value):  # pragma: no cover - safety tripwire
            raise AssertionError(f"Unexpected mutation of {name}")

    intent = ArrayIntent(
        name="field_names",
        shape=(100,),
        dtype="S32",
        access=AccessPattern.EAGER,
        write_mode=WriteMode.IMMUTABLE,
    )

    comparison = compare_array_storage(
        ReadOnlyArray(),
        intent=intent,
        profile=PUBLISHED_HTTP_V1,
    )

    assert not comparison.chunk_shape_changes
    assert not comparison.shard_shape_changes


def test_array_report_rejects_dtype_mismatch() -> None:
    class ReadOnlyArray:
        shape = (100,)
        dtype = np.dtype(np.int64)
        chunks = (100,)
        shards = None

    intent = ArrayIntent(
        shape=(100,),
        dtype=np.int32,
        access=AccessPattern.EAGER,
        write_mode=WriteMode.IMMUTABLE,
    )

    with pytest.raises(ValueError, match="Observed dtype"):
        compare_array_storage(
            ReadOnlyArray(),
            intent=intent,
            profile=PUBLISHED_HTTP_V1,
        )


def test_invalid_access_shape_and_random_shard_ownership_fail_closed() -> None:
    with pytest.raises(ValueError, match="same rank"):
        ArrayIntent(
            shape=(100, 5, 2),
            dtype=np.float32,
            access=AccessPattern.WINDOWED,
            write_mode=WriteMode.IMMUTABLE,
            access_unit_shape=(1, 10),
        )

    with pytest.raises(ValueError, match="Random-update"):
        ArrayIntent(
            shape=(100,),
            dtype=np.float32,
            access=AccessPattern.WINDOWED,
            write_mode=WriteMode.RANDOM_UPDATE,
            whole_shard_writes=True,
        )
