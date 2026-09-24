from __future__ import annotations

import numpy as np
import zarr

from fisheye.shared.zarr_row_chunk_updates import update_rows_by_chunk


def test_chunk_row_io_preserves_order_duplicates_and_untouched_rows(monkeypatch) -> None:
    group = zarr.group()
    initial = np.arange(24, dtype=np.int32).reshape(6, 2, 2)
    array = group.create_array("rows", data=initial, chunks=(3, 2, 2))
    array_type = type(array)
    store_type = type(group.store)
    getitem = array_type.__getitem__
    setitem = array_type.__setitem__
    store_set = store_type.set
    counts = {"reads": 0, "writes": 0, "chunk_sets": 0}

    def counted_getitem(node, key):
        counts["reads"] += 1
        return getitem(node, key)

    def counted_setitem(node, key, value):
        counts["writes"] += 1
        return setitem(node, key, value)

    async def counted_store_set(store, key, value):
        if str(key).startswith("rows/c/"):
            counts["chunk_sets"] += 1
        return await store_set(store, key, value)

    monkeypatch.setattr(array_type, "__getitem__", counted_getitem)
    monkeypatch.setattr(array_type, "__setitem__", counted_setitem)
    monkeypatch.setattr(store_type, "set", counted_store_set)

    update_rows_by_chunk(
        array,
        [4, 1, 4],
        [np.full((2, 2), 10), np.full((2, 2), 20), np.full((2, 2), 30)],
    )
    assert counts == {"reads": 2, "writes": 2, "chunk_sets": 2}
    expected = initial.copy()
    expected[1] = 20
    expected[4] = 30
    np.testing.assert_array_equal(array[:], expected)
