from __future__ import annotations

import hashlib
from dataclasses import replace

import numpy as np
import pytest
import zarr

from fisheye.shared.keyed_delta import (
    ACTION_CODE_MAP,
    DEFAULT_DELTA_PLAN_SHARD_ROWS,
    OMIT_REASON_CODE_MAP,
    REASON_CODE_MAP,
    KeyedDeltaPlanError,
    build_keyed_delta_plan,
    validate_keyed_delta_plan,
    write_keyed_delta_plan,
)


SPEC_A = hashlib.sha256(b"spec-a").hexdigest()
SPEC_B = hashlib.sha256(b"spec-b").hexdigest()


def _signatures(*values: int) -> np.ndarray:
    rows = np.zeros((len(values), 32), dtype=np.uint8)
    rows[:, 0] = np.asarray(values, dtype=np.uint8)
    return rows


def test_plan_matches_by_key_across_reorder_and_tracks_deletion() -> None:
    plan = build_keyed_delta_plan(
        target_instance_keys=np.asarray([30, 10, 40], dtype=np.uint64),
        target_source_signatures=_signatures(3, 1, 4),
        target_signature_spec_digest=SPEC_A,
        source_instance_keys=np.asarray([10, 20, 30], dtype=np.uint64),
        source_row_signatures=_signatures(1, 2, 3),
        source_signature_spec_digest=SPEC_A,
    )

    np.testing.assert_array_equal(plan.target_row_indices, [0, 1, 2])
    np.testing.assert_array_equal(plan.source_row_indices, [2, 0, -1])
    np.testing.assert_array_equal(
        plan.action_codes,
        [ACTION_CODE_MAP["copy"], ACTION_CODE_MAP["copy"], ACTION_CODE_MAP["compute"]],
    )
    np.testing.assert_array_equal(
        plan.reason_codes,
        [REASON_CODE_MAP["unchanged"], REASON_CODE_MAP["unchanged"], REASON_CODE_MAP["added"]],
    )
    np.testing.assert_array_equal(plan.omitted_instance_keys, [20])
    np.testing.assert_array_equal(plan.omitted_source_row_indices, [1])
    np.testing.assert_array_equal(
        plan.omitted_reason_codes,
        [OMIT_REASON_CODE_MAP["deleted_from_target"]],
    )


def test_plan_recomputes_changed_source_and_preserves_compatible_manual_rows() -> None:
    plan = build_keyed_delta_plan(
        target_instance_keys=np.asarray([10, 20], dtype=np.uint64),
        target_source_signatures=_signatures(9, 2),
        target_signature_spec_digest=SPEC_A,
        source_instance_keys=np.asarray([10, 20], dtype=np.uint64),
        source_row_signatures=_signatures(1, 2),
        source_signature_spec_digest=SPEC_A,
        preserve_manual_target_mask=np.asarray([True, True]),
    )

    np.testing.assert_array_equal(
        plan.action_codes,
        [ACTION_CODE_MAP["compute"], ACTION_CODE_MAP["preserve_manual"]],
    )
    np.testing.assert_array_equal(
        plan.reason_codes,
        [REASON_CODE_MAP["source_changed"], REASON_CODE_MAP["preserved_manual"]],
    )


def test_plan_recomputes_matched_rows_when_signature_spec_changes() -> None:
    plan = build_keyed_delta_plan(
        target_instance_keys=np.asarray([10, 30], dtype=np.uint64),
        target_source_signatures=_signatures(1, 3),
        target_signature_spec_digest=SPEC_B,
        source_instance_keys=np.asarray([10], dtype=np.uint64),
        source_row_signatures=_signatures(1),
        source_signature_spec_digest=SPEC_A,
    )

    np.testing.assert_array_equal(
        plan.reason_codes,
        [REASON_CODE_MAP["signature_spec_changed"], REASON_CODE_MAP["added"]],
    )
    np.testing.assert_array_equal(
        plan.action_codes,
        [ACTION_CODE_MAP["compute"], ACTION_CODE_MAP["compute"]],
    )


def test_plan_without_reuse_source_computes_every_row() -> None:
    plan = build_keyed_delta_plan(
        target_instance_keys=np.asarray([10, 20], dtype=np.uint64),
        target_source_signatures=_signatures(1, 2),
        target_signature_spec_digest=SPEC_A,
    )

    np.testing.assert_array_equal(
        plan.reason_codes,
        [REASON_CODE_MAP["no_reuse_source"], REASON_CODE_MAP["no_reuse_source"]],
    )
    assert plan.summary()["source_row_count"] == 0


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        (
            {"target_instance_keys": np.asarray([1, 1], dtype=np.uint64)},
            "must be unique",
        ),
        (
            {"source_instance_keys": np.asarray([1], dtype=np.uint64)},
            "must be supplied together",
        ),
        (
            {"preserve_manual_target_mask": np.asarray([True])},
            "does not match target row count",
        ),
    ],
)
def test_plan_fails_closed_on_invalid_identity_inputs(
    kwargs: dict[str, np.ndarray],
    match: str,
) -> None:
    base: dict[str, object] = {
        "target_instance_keys": np.asarray([1, 2], dtype=np.uint64),
        "target_source_signatures": _signatures(1, 2),
        "target_signature_spec_digest": SPEC_A,
    }
    base.update(kwargs)
    with pytest.raises(KeyedDeltaPlanError, match=match):
        build_keyed_delta_plan(**base)


def test_plan_writer_persists_sharded_columns_and_code_contract() -> None:
    row_count = DEFAULT_DELTA_PLAN_SHARD_ROWS + 1
    keys = np.arange(row_count, dtype=np.uint64)
    plan = build_keyed_delta_plan(
        target_instance_keys=keys,
        target_source_signatures=np.zeros((row_count, 32), dtype=np.uint8),
        target_signature_spec_digest=SPEC_A,
    )
    root = zarr.group(store=zarr.storage.MemoryStore(), zarr_format=3)
    plan_group = root.create_group("materialization_plan")

    write_keyed_delta_plan(plan_group, plan)

    np.testing.assert_array_equal(plan_group["instance_key"][:], keys)
    np.testing.assert_array_equal(
        plan_group["action_codes"][:],
        np.full(row_count, ACTION_CODE_MAP["compute"], dtype=np.uint8),
    )
    assert plan_group["instance_key"].attrs["palette_physical_layout"] == (
        "indexed_sharding_v1"
    )
    assert plan_group.attrs["action_code_map"] == ACTION_CODE_MAP
    assert plan_group.attrs["reason_code_map"] == REASON_CODE_MAP
    assert plan_group.attrs["summary"]["target_row_count"] == row_count


def test_plan_validation_rejects_inconsistent_action_reason_pair() -> None:
    plan = build_keyed_delta_plan(
        target_instance_keys=np.asarray([10], dtype=np.uint64),
        target_source_signatures=_signatures(1),
        target_signature_spec_digest=SPEC_A,
    )
    invalid = replace(
        plan,
        action_codes=np.asarray([ACTION_CODE_MAP["copy"]], dtype=np.uint8),
        source_row_indices=np.asarray([0], dtype=np.int64),
    )

    with pytest.raises(KeyedDeltaPlanError, match="action/reason pair"):
        validate_keyed_delta_plan(invalid)
