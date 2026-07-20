from __future__ import annotations

import numpy as np
import pytest

from fisheye.utils.migrate_merged_collection_proxy_coordinates import (
    _positions_for_unique_ids,
    _value_equal,
)


def test_positions_for_unique_ids_joins_stable_ids_not_array_offsets() -> None:
    available = np.asarray([5, 9, 20], dtype=np.int64)
    requested = np.asarray([20, 5], dtype=np.int64)

    assert np.array_equal(
        _positions_for_unique_ids(available, requested, label="refined rows"),
        np.asarray([2, 0], dtype=np.int64),
    )


def test_positions_for_unique_ids_rejects_missing_or_duplicate_identity() -> None:
    with pytest.raises(ValueError, match="does not contain"):
        _positions_for_unique_ids(
            np.asarray([5, 9], dtype=np.int64),
            np.asarray([10], dtype=np.int64),
            label="refined rows",
        )
    with pytest.raises(ValueError, match="duplicate"):
        _positions_for_unique_ids(
            np.asarray([5, 5], dtype=np.int64),
            np.asarray([5], dtype=np.int64),
            label="refined rows",
        )


def test_value_equal_allows_exact_persisted_storage_casts_only() -> None:
    assert _value_equal(
        np.asarray([0, 1], dtype=np.int64),
        np.asarray([0, 1], dtype=np.int32),
    )
    assert _value_equal(
        np.asarray([[0.25]], dtype=np.float32),
        np.asarray([[0.25]], dtype=np.float64),
    )
    assert not _value_equal(
        np.asarray([[0.25]], dtype=np.float32),
        np.asarray([[0.25001]], dtype=np.float64),
    )
