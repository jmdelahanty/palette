from __future__ import annotations

import numpy as np
import pytest

from fisheye.shared.historical_collection_proxy_v1 import (
    BoundHistoricalMergedCollectionProxyV1,
    HistoricalCollectionProxyV1Error,
    _positions_for_unique_ids,
)


def test_historical_binding_cannot_be_forged() -> None:
    with pytest.raises(
        HistoricalCollectionProxyV1Error,
        match="cannot be constructed directly",
    ):
        BoundHistoricalMergedCollectionProxyV1()


def test_positions_for_unique_ids_preserves_requested_order() -> None:
    available = np.asarray([40, 10, 30, 20], dtype=np.int64)
    requested = np.asarray([20, 40, 10], dtype=np.int64)

    positions = _positions_for_unique_ids(
        available,
        requested,
        label="test ids",
    )

    np.testing.assert_array_equal(available[positions], requested)


def test_positions_for_unique_ids_rejects_duplicate_or_missing_authority() -> None:
    with pytest.raises(HistoricalCollectionProxyV1Error, match="duplicate"):
        _positions_for_unique_ids(
            np.asarray([1, 1], dtype=np.int64),
            np.asarray([1], dtype=np.int64),
            label="test ids",
        )
    with pytest.raises(HistoricalCollectionProxyV1Error, match="every requested"):
        _positions_for_unique_ids(
            np.asarray([1, 2], dtype=np.int64),
            np.asarray([3], dtype=np.int64),
            label="test ids",
        )
