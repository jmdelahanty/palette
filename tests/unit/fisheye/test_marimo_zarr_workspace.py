from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

import apps.marimo.components.zarr_workspace as workspace_module
from apps.marimo.components.zarr_workspace import ZarrExplorationWorkspace


class _FakeArray:
    def __init__(self, values, *, chunks=None, attrs=None) -> None:
        self.values = np.asarray(values)
        self.shape = self.values.shape
        self.dtype = self.values.dtype
        self.chunks = chunks or self.shape
        self.nbytes = self.values.nbytes
        self.attrs = attrs or {}
        self.reads: list[object] = []

    def __getitem__(self, selection):
        self.reads.append(selection)
        return self.values[selection]


class _FakeGroup:
    def __init__(self, members=None, *, attrs=None) -> None:
        self.members = dict(members or {})
        self.attrs = attrs or {}

    def keys(self):
        yield from self.members

    def __getitem__(self, path: str):
        node: object = self
        for part in str(path).split("/"):
            node = node.members[part]  # type: ignore[attr-defined]
        return node


def _workspace() -> tuple[ZarrExplorationWorkspace, _FakeArray, _FakeArray]:
    speed = _FakeArray(np.arange(20, dtype=np.float32), chunks=(5,))
    time_s = _FakeArray(np.arange(20, dtype=np.float64) / 10, chunks=(10,))
    images = _FakeArray(np.zeros((5, 100, 100), dtype=np.uint8), chunks=(1, 100, 100))
    tracks = _FakeGroup(
        {"speed": speed, "time_s": time_s, "images": images},
        attrs={"units": "metric"},
    )
    root = _FakeGroup({"tracks": tracks}, attrs={"recording": "canary"})
    return (
        ZarrExplorationWorkspace(
            zarr_path=Path("/data/source.zarr"),
            _root=root,
            max_read_elements=1_000,
        ),
        speed,
        images,
    )


def test_zarr_workspace_opens_source_read_only(monkeypatch, tmp_path: Path) -> None:
    root = _FakeGroup()
    calls: list[tuple[Path, str]] = []

    def _open(path: Path, *, mode: str):
        calls.append((path, mode))
        return root

    monkeypatch.setattr(workspace_module, "open_zarr_root", _open)

    workspace = ZarrExplorationWorkspace.open(tmp_path / "source.zarr")

    assert workspace.handle() is root
    assert calls == [(tmp_path / "source.zarr", "r")]
    assert workspace.summary()["read_only"] is True


def test_zarr_workspace_inventory_uses_metadata_without_reading_arrays() -> None:
    workspace, speed, images = _workspace()

    rows = workspace.walk(max_depth=2)

    assert [row["path"] for row in rows] == [
        "tracks",
        "tracks/images",
        "tracks/speed",
        "tracks/time_s",
    ]
    assert workspace.info("tracks/images") == {
        "path": "tracks/images",
        "kind": "array",
        "shape": (5, 100, 100),
        "dtype": "uint8",
        "chunks": (1, 100, 100),
        "ndim": 3,
        "elements": 50_000,
        "nbytes": 50_000,
    }
    assert speed.reads == []
    assert images.reads == []


def test_zarr_workspace_enforces_bounded_explicit_reads() -> None:
    workspace, speed, images = _workspace()

    np.testing.assert_array_equal(
        workspace.read("tracks/speed", slice(2, 7)),
        np.arange(2, 7, dtype=np.float32),
    )
    assert speed.reads == [(slice(2, 7, 1),)]

    with pytest.raises(ValueError, match="10,000 elements"):
        workspace.read("tracks/images", 0)
    assert images.reads == []

    image_crop = workspace.read(
        "tracks/images",
        (0, slice(0, 10), slice(0, 10)),
    )
    assert image_crop.shape == (10, 10)

    with pytest.raises(TypeError, match="fancy indexing"):
        workspace.read("tracks/speed", [1, 2, 3])


def test_zarr_workspace_builds_bounded_polars_table_without_pandas() -> None:
    workspace, _, _ = _workspace()

    frame = workspace.to_polars(
        "tracks",
        columns=["time_s", "speed"],
        start=3,
        stop=8,
    )

    assert isinstance(frame, pl.DataFrame)
    assert frame.columns == ["time_s", "speed"]
    assert frame.shape == (5, 2)
    assert frame["speed"].to_list() == [3.0, 4.0, 5.0, 6.0, 7.0]

    with pytest.raises(ValueError, match="3D"):
        workspace.to_polars("tracks", columns=["images"], stop=2)


def test_zarr_workspace_rejects_paths_outside_selected_root() -> None:
    workspace, _, _ = _workspace()

    with pytest.raises(ValueError, match="relative"):
        workspace.info("/etc")
    with pytest.raises(ValueError, match="Invalid relative"):
        workspace.info("../sibling.zarr")
