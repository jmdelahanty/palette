"""Batch row writes land exactly on sharded and chunk-only review arrays."""

from __future__ import annotations

import numpy as np
import pytest
import zarr

from fisheye.labeling.web_keypoint_checkpoints import _read_rows, _write_rows


@pytest.mark.parametrize("sharded", [None, "whole_run", "multi_shard"])
@pytest.mark.parametrize("tail", [(19, 2), (7,), ()])
def test_write_rows_changes_only_selected_rows(tmp_path, sharded, tail):
    n = 23
    shards = {None: None, "whole_run": (n, *tail), "multi_shard": (8, *tail)}[sharded]
    array = zarr.create_array(
        str(tmp_path / "a"), shape=(n, *tail), chunks=(1, *tail), shards=shards,
        dtype="f4", fill_value=np.nan,
    )
    base = np.random.default_rng(0).random((n, *tail)).astype("f4")
    array[:] = base
    rows = [2, 3, 9, 17, 22]  # Non-contiguous, spanning several shards.
    values = np.full((len(rows), *tail), 7.5, dtype="f4")
    _write_rows(array, rows, values)
    expected = base.copy()
    expected[rows] = values
    assert np.asarray(array[:]).tobytes() == expected.tobytes()
    np.testing.assert_array_equal(np.asarray(_read_rows(array, np.asarray(rows))), values)
