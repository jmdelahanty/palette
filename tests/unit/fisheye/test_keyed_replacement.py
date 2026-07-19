from __future__ import annotations

import numpy as np
import pytest

from fisheye.shared.keyed_replacement import (
    REPLACEMENT_SOURCE_BASE,
    KeyedReplacementError,
    build_keyed_replacement_plan,
)


_DIGEST = "1" * 64


def _signatures(values: list[int]) -> np.ndarray:
    output = np.zeros((len(values), 32), dtype=np.uint8)
    output[:, 0] = values
    return output


def test_replacement_plan_handles_reorder_change_add_and_delete() -> None:
    plan = build_keyed_replacement_plan(
        target_instance_keys=np.asarray([30, 10, 40], dtype=np.uint64),
        target_source_signatures=_signatures([9, 1, 4]),
        target_signature_spec_digest=_DIGEST,
        base_instance_keys=np.asarray([10, 20, 30], dtype=np.uint64),
        base_source_signatures=_signatures([1, 2, 3]),
        base_signature_spec_digest=_DIGEST,
        replacement_instance_keys=[np.asarray([40, 30], dtype=np.uint64)],
    )

    np.testing.assert_array_equal(plan.source_run_indices, [0, REPLACEMENT_SOURCE_BASE, 0])
    np.testing.assert_array_equal(plan.source_row_indices, [1, 0, 0])
    np.testing.assert_array_equal(plan.replacement_target_rows, [0, 2])
    assert plan.summary()["delta_plan"]["omitted_row_count"] == 1


@pytest.mark.parametrize(
    "replacement_keys, message",
    [
        ([np.asarray([30], dtype=np.uint64)], "missing=1"),
        ([np.asarray([30, 40, 10], dtype=np.uint64)], "extra=1"),
        (
            [np.asarray([30], dtype=np.uint64), np.asarray([30, 40], dtype=np.uint64)],
            "more than one replacement",
        ),
    ],
)
def test_replacement_plan_fails_closed_on_nonexact_replacements(
    replacement_keys: list[np.ndarray], message: str
) -> None:
    with pytest.raises(KeyedReplacementError, match=message):
        build_keyed_replacement_plan(
            target_instance_keys=np.asarray([30, 10, 40], dtype=np.uint64),
            target_source_signatures=_signatures([9, 1, 4]),
            target_signature_spec_digest=_DIGEST,
            base_instance_keys=np.asarray([10, 20, 30], dtype=np.uint64),
            base_source_signatures=_signatures([1, 2, 3]),
            base_signature_spec_digest=_DIGEST,
            replacement_instance_keys=replacement_keys,
        )
