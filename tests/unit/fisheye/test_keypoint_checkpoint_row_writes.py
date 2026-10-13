"""Batch row writes land exactly on sharded and chunk-only review arrays."""

from __future__ import annotations

import numpy as np
import pytest
import zarr

from fisheye.labeling.web_keypoint_checkpoints import _read_rows, _write_rows
from fisheye.shared.zarr_helpers import archive_metadata_publication_lock


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
    with archive_metadata_publication_lock(tmp_path):
        _write_rows(array, rows, values, archive=tmp_path)
    expected = base.copy()
    expected[rows] = values
    assert np.asarray(array[:]).tobytes() == expected.tobytes()
    np.testing.assert_array_equal(np.asarray(_read_rows(array, np.asarray(rows))), values)


def test_sharded_row_write_refuses_without_the_archive_lock(tmp_path):
    array = zarr.create_array(
        str(tmp_path / "a"), shape=(6, 3), chunks=(1, 3), shards=(6, 3),
        dtype="f4", fill_value=0,
    )
    before = np.asarray(array[:]).copy()
    for archive in (None, tmp_path):
        with pytest.raises(RuntimeError, match="archive publication lock"):
            _write_rows(array, [1, 4], np.ones((2, 3), "f4"), archive=archive)
    assert np.asarray(array[:]).tobytes() == before.tobytes()
