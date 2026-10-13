"""H5 → Zarr mirrors must not hold h5py's global lock while writing Zarr.

h5py's ``items()`` iterators hold ``phil`` across each loop body. A Zarr write
inside that body waits on Zarr's event-loop thread; a garbage collection there
that frees an h5py object needs ``phil`` and the two threads deadlock. CI shard
15 hung for an hour this way (runs 37563359720, 37584143156). The check below
is deterministic: every Zarr write asks another thread to take ``phil``.
"""

from __future__ import annotations

import threading

import h5py
import numpy as np
import pytest
import zarr
from h5py._objects import phil
from zarr.storage import MemoryStore

from fisheye.analysis import import_stimulus_to_zarr as mod


def _assert_phil_free() -> None:
    taken = threading.Event()

    def take() -> None:
        with phil:
            taken.set()

    threading.Thread(target=take, daemon=True).start()
    assert taken.wait(5.0), "h5py phil is held while writing Zarr"


class _Attrs:
    def __init__(self, attrs):
        self._attrs = attrs

    def __setitem__(self, key, value):
        _assert_phil_free()
        self._attrs[key] = value

    def __getattr__(self, name):
        return getattr(self._attrs, name)


class _CheckedGroup:
    """Forwards to a real Zarr group, checking ``phil`` on every write."""

    def __init__(self, group: zarr.Group):
        self._group = group
        self.writes = 0

    @property
    def attrs(self):
        return _Attrs(self._group.attrs)

    def require_group(self, name):
        _assert_phil_free()
        return _CheckedGroup(self._group.require_group(name))

    def create_array(self, *args, **kwargs):
        _assert_phil_free()
        return self._group.create_array(*args, **kwargs)

    def __contains__(self, name):
        return name in self._group

    def __getitem__(self, name):
        return self._group[name]

    def __delitem__(self, name):
        _assert_phil_free()
        del self._group[name]

    def __getattr__(self, name):
        return getattr(self._group, name)


@pytest.fixture()
def h5_tree(tmp_path):
    path = tmp_path / "calibration.h5"
    with h5py.File(path, "w") as h5:
        camera = h5.create_group("calibration_snapshot/CAM-1")
        camera.attrs["camera_id"] = "CAM-1"
        camera.create_dataset("homography", data=np.eye(3))
        camera.create_dataset("scale_px_per_mm", data=50.0)
        nested = camera.create_group("intrinsics")
        nested.attrs["model"] = "pinhole"
        nested.create_dataset("matrix", data=np.arange(9.0).reshape(3, 3))
    with h5py.File(path, "r") as h5:
        yield h5["calibration_snapshot/CAM-1"]


@pytest.mark.parametrize("copy", [mod._copy_h5_group_to_zarr_mirror, mod._copy_h5_tree])
def test_h5_mirror_releases_phil_before_zarr_writes(h5_tree, copy):
    root = zarr.open_group(store=MemoryStore(), mode="w")

    copy(h5_tree, _CheckedGroup(root))

    assert root.attrs["camera_id"] == "CAM-1"
    np.testing.assert_array_equal(root["homography"][:], np.eye(3))
    assert root.attrs["scale_px_per_mm"] == 50.0
    assert root["intrinsics"].attrs["model"] == "pinhole"
    np.testing.assert_array_equal(root["intrinsics/matrix"][:], np.arange(9.0).reshape(3, 3))
