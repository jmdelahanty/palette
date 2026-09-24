"""Serial, bounded updates to row-oriented Zarr arrays."""

from collections import defaultdict
from typing import Sequence

import numpy as np
import zarr


def update_rows_by_chunk(
    array: zarr.Array,
    rows: Sequence[int],
    values: Sequence[object] | np.ndarray,
    *,
    tail_index: tuple[int, ...] = (),
) -> None:
    """Read and replace each touched row chunk once, keeping other cells intact.

    Updates within a chunk retain input order, including repeated row indices.
    The caller owns the array for the duration of this serial read/modify/write.
    """
    if len(rows) != len(values):
        raise ValueError("rows and values must have the same length.")
    row_chunk = int(array.chunks[0])
    offsets: dict[int, list[int]] = defaultdict(list)
    for offset, raw_row in enumerate(rows):
        row = int(raw_row)
        if row < 0 or row >= int(array.shape[0]):
            raise ValueError(f"row {row} is outside the array.")
        offsets[row // row_chunk].append(offset)
    for chunk_index, selected in offsets.items():
        start = chunk_index * row_chunk
        stop = min(start + row_chunk, int(array.shape[0]))
        key = (slice(start, stop),) + tail_index
        block = np.asarray(array[key]).copy()
        for offset in selected:
            block[int(rows[offset]) - start] = values[offset]
        array[key] = block
