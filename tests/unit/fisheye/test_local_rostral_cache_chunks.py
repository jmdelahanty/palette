from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parents[3] / "playgrounds" / "heartrate_stabilization"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from extract_reliable_local_rostral_heartrate import (  # noqa: E402
    _RowBlockCachedMaskStore,
    _split_xy,
)


class _DenseArrayContract:
    chunks = (4, 1, 2, 2)


class _FakeMaskStore:
    def __init__(self) -> None:
        self.shape = (10, 1, 2, 2)
        self.dense_array = _DenseArrayContract()
        self.values = np.arange(40, dtype=np.uint8).reshape(10, 1, 2, 2)
        self.reads: list[tuple[int, int, str]] = []

    def read_dense(self, rows=None, channels=None):
        if isinstance(rows, slice):
            self.reads.append((int(rows.start), int(rows.stop), str(channels)))
        return self.values[rows]


def test_row_block_mask_cache_reuses_physical_chunks() -> None:
    source = _FakeMaskStore()
    cached = _RowBlockCachedMaskStore(source, requested_rows=3)

    np.testing.assert_array_equal(cached.read_dense(rows=0, channels="body"), source.values[0:1])
    np.testing.assert_array_equal(cached.read_dense(rows=1, channels="body"), source.values[1:2])
    np.testing.assert_array_equal(cached.read_dense(rows=3, channels="body"), source.values[3:4])
    np.testing.assert_array_equal(cached.read_dense(rows=4, channels="body"), source.values[4:5])

    assert cached.block_rows == 4
    assert source.reads == [(0, 4, "body"), (4, 8, "body")]
    assert cached.cache_summary() == {"block_rows": 4, "hits": 2, "misses": 2}


def test_explicit_reference_coordinates_require_two_finite_values() -> None:
    np.testing.assert_array_equal(
        _split_xy("128,113", name="anterior"),
        np.asarray([128.0, 113.0]),
    )
