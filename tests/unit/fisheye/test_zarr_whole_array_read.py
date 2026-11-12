"""read_whole_array must return exactly what array[:] returns, opening each file once."""

from __future__ import annotations

import builtins
from pathlib import Path

import numpy as np
import pytest
import zarr
from zarr.codecs import ZstdCodec

from fisheye.shared.zarr import whole_array_read
from fisheye.shared.zarr.whole_array_read import read_whole_array

LAYOUTS = {
    # The published frame-clock layout: 131072-row shards of 4096-row inner chunks.
    "sharded_small_inner": dict(shape=(362204,), chunks=(4096,), shards=(131072,), dtype="int64"),
    "sharded_bool": dict(shape=(362204,), chunks=(4096,), shards=(131072,), dtype="bool"),
    "chunked_partial_last": dict(shape=(100003,), chunks=(16384,), dtype="int64"),
    "single_chunk": dict(shape=(5000,), chunks=(5000,), dtype="float64"),
    "two_dimensional": dict(shape=(30001, 4), chunks=(16384, 4), dtype="int32"),
    "sharded_two_dimensional": dict(shape=(30001, 4), chunks=(1024, 4), shards=(8192, 4), dtype="uint16"),
}


def _array(tmp_path: Path, name: str, *, sparse: bool = False, fill_value=None):
    layout = dict(LAYOUTS[name])
    shape, dtype = layout.pop("shape"), layout.pop("dtype")
    root = zarr.open_group(str(tmp_path / "archive.zarr"), mode="a", zarr_format=3)
    group = root.require_group("analysis/run")
    array = group.create_array(
        name, shape=shape, dtype=dtype, compressors=ZstdCodec(level=0),
        fill_value=fill_value, **layout,
    )
    rng = np.random.default_rng(7)
    values = (rng.integers(0, 2, size=shape).astype(dtype) if dtype == "bool"
              else rng.integers(-(10**9), 10**9, size=shape).astype(dtype))
    if sparse:
        # Write only the first rows: unwritten chunks stay absent on disk and
        # must read back as the fill value through both paths.
        array[: shape[0] // 7] = values[: shape[0] // 7]
    else:
        array[...] = values
    return zarr.open_group(str(tmp_path / "archive.zarr"), mode="r", use_consolidated=False)[
        f"analysis/run/{name}"
    ]


@pytest.mark.parametrize("name", sorted(LAYOUTS))
@pytest.mark.parametrize("sparse", (False, True), ids=("dense", "sparse"))
def test_reads_exactly_what_array_slicing_reads(tmp_path, name, sparse):
    array = _array(tmp_path, name, sparse=sparse, fill_value=3 if not sparse else 5)
    expected = np.asarray(array[:])
    actual = read_whole_array(array)
    assert actual.dtype == expected.dtype and actual.shape == expected.shape
    assert np.array_equal(actual, expected)


def test_each_stored_file_is_opened_once(tmp_path, monkeypatch):
    array = _array(tmp_path, "sharded_small_inner")
    array_dir = tmp_path / "archive.zarr/analysis/run/sharded_small_inner"
    stored = sorted(p for p in array_dir.rglob("*") if p.is_file() and p.name != "zarr.json")
    opened: list[str] = []
    real_open = builtins.open

    def counting_open(path, *args, **kwargs):
        opened.append(str(path))
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", counting_open)
    read_whole_array(array)
    assert sorted(opened) == [str(p) for p in stored]  # one open per shard, no zarr.json


def _fallback_spy(monkeypatch) -> list[bool]:
    used = []
    real = whole_array_read._stored_files

    def spy(*args, **kwargs):
        result = real(*args, **kwargs)
        used.append(result is not None)
        return result

    monkeypatch.setattr(whole_array_read, "_stored_files", spy)
    return used


def test_more_stored_bytes_than_the_limit_falls_back(tmp_path, monkeypatch):
    array = _array(tmp_path, "sharded_small_inner")
    used = _fallback_spy(monkeypatch)
    assert np.array_equal(read_whole_array(array, max_stored_bytes=1024), array[:])
    assert used == [False]


def test_a_symlink_in_the_array_directory_falls_back(tmp_path, monkeypatch):
    array = _array(tmp_path, "chunked_partial_last")
    array_dir = tmp_path / "archive.zarr/analysis/run/chunked_partial_last"
    target = array_dir / "c" / "0"
    moved = tmp_path / "elsewhere"
    target.rename(moved)
    target.symlink_to(moved)
    used = _fallback_spy(monkeypatch)
    assert np.array_equal(read_whole_array(array), array[:])
    assert used == [False]


def test_a_non_local_store_falls_back(monkeypatch):
    group = zarr.open_group(store=zarr.storage.MemoryStore(), mode="w", zarr_format=3)
    array = group.create_array("values", data=np.arange(10, dtype="int64"), chunks=(4,))
    used = _fallback_spy(monkeypatch)
    assert np.array_equal(read_whole_array(array), np.arange(10))
    assert used == []


def test_a_plain_numpy_value_falls_back():
    assert np.array_equal(read_whole_array(np.arange(5)), np.arange(5))
