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


def _fixed_width_rows(values: list[str], width: int = 64) -> np.ndarray:
    result = np.zeros((len(values), width), dtype=np.uint8)
    for row, value in enumerate(values):
        encoded = value.encode("utf-8")[:width]
        result[row, : len(encoded)] = np.frombuffer(encoded, dtype=np.uint8)
    return result


def _eye_angle_workspace() -> ZarrExplorationWorkspace:
    frame_angles = _FakeArray(
        np.column_stack(
            [
                np.arange(20, dtype=np.float32),
                np.arange(20, dtype=np.float32) * 2,
                np.arange(20, dtype=np.float32) * 3,
            ]
        ),
        chunks=(8, 3),
    )
    channel_index = _FakeGroup(
        {
            "name": _FakeArray(
                _fixed_width_rows(
                    [
                        "left_eye_angle_deg_smoothed",
                        "right_eye_angle_deg_smoothed",
                        "vergence_eye_angle_deg_smoothed",
                    ]
                )
            ),
            "units": _FakeArray(_fixed_width_rows(["deg", "deg", "deg"])),
            "representation": _FakeArray(
                _fixed_width_rows(["eye_frame", "eye_frame", "eye_frame"])
            ),
            "eye": _FakeArray(_fixed_width_rows(["left", "right", "binocular"])),
            "value_kind": _FakeArray(
                _fixed_width_rows(["angle", "angle", "vergence"])
            ),
            "source_channel": _FakeArray(
                _fixed_width_rows(["left_raw", "right_raw", "derived"])
            ),
            "formula": _FakeArray(
                _fixed_width_rows(["smooth(left_raw)", "smooth(right_raw)", "left-right"])
            ),
            "compatibility_alias_of": _FakeArray(
                _fixed_width_rows(["", "", ""])
            ),
            "frame_available": _FakeArray(np.array([True, True, False])),
        }
    )
    support = _FakeGroup(
        {"frame_time_seconds": _FakeArray(np.arange(20, dtype=np.float32) / 10)}
    )
    run = _FakeGroup(
        {
            "frame_angles": frame_angles,
            "angle_channel_index": channel_index,
            "support": support,
        },
        attrs={
            "status": "complete",
            "layout": "compact_dense_v2",
            "method": "unified_subject_masks",
            "schema_version": "2.0",
        },
    )
    root = _FakeGroup(
        {
            "analysis": _FakeGroup(
                {"eye_angle_runs": _FakeGroup({"canary": run})}
            )
        }
    )
    return ZarrExplorationWorkspace(
        zarr_path=Path("/data/source.zarr"),
        _root=root,
        max_read_elements=1_000,
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


def test_zarr_workspace_has_empty_guided_discovery_for_arbitrary_zarr() -> None:
    workspace, _, _ = _workspace()

    assert workspace.eye_angle_runs() == []


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


def test_zarr_workspace_resolves_compact_dense_channel_names() -> None:
    workspace = _eye_angle_workspace()
    path = "analysis/eye_angle_runs/canary/frame_angles"

    assert workspace.channel_index(path) == [
        {
            "index": 0,
            "name": "left_eye_angle_deg_smoothed",
            "units": "deg",
            "representation": "eye_frame",
            "eye": "left",
            "value_kind": "angle",
            "source_channel": "left_raw",
            "formula": "smooth(left_raw)",
            "compatibility_alias_of": "",
            "available": True,
        },
        {
            "index": 1,
            "name": "right_eye_angle_deg_smoothed",
            "units": "deg",
            "representation": "eye_frame",
            "eye": "right",
            "value_kind": "angle",
            "source_channel": "right_raw",
            "formula": "smooth(right_raw)",
            "compatibility_alias_of": "",
            "available": True,
        },
    ]
    assert workspace.suggested_coordinate_path(path) == (
        "analysis/eye_angle_runs/canary/support/frame_time_seconds"
    )


def test_zarr_workspace_discovers_eye_angle_runs_without_frame_reads() -> None:
    workspace = _eye_angle_workspace()
    frame_angles = workspace.handle(
        "analysis/eye_angle_runs/canary/frame_angles"
    )

    assert workspace.eye_angle_runs() == [
        {
            "run_name": "canary",
            "run_path": "analysis/eye_angle_runs/canary",
            "status": "complete",
            "layout": "compact_dense_v2",
            "method": "unified_subject_masks",
            "schema_version": "2.0",
            "frame_count": 20,
            "frame_channel_count": 3,
            "frame_angles_path": "analysis/eye_angle_runs/canary/frame_angles",
        }
    ]
    assert frame_angles.reads == []


def test_zarr_workspace_summarizes_time_coordinate_with_scalar_reads() -> None:
    workspace = _eye_angle_workspace()
    path = "analysis/eye_angle_runs/canary/frame_angles"
    coordinate = workspace.handle(
        "analysis/eye_angle_runs/canary/support/frame_time_seconds"
    )

    summary = workspace.coordinate_summary(path)

    assert summary is not None
    assert summary["path"] == (
        "analysis/eye_angle_runs/canary/support/frame_time_seconds"
    )
    assert summary["row_count"] == 20
    assert summary["start_seconds"] == pytest.approx(0.0)
    assert summary["stop_seconds"] == pytest.approx(1.9)
    assert summary["sample_interval_seconds"] == pytest.approx(0.1)
    assert summary["sample_rate_hz"] == pytest.approx(10.0)
    assert coordinate.reads == [(0,), (19,), (1,)]


def test_zarr_workspace_builds_bounded_time_trace() -> None:
    workspace = _eye_angle_workspace()
    path = "analysis/eye_angle_runs/canary/frame_angles"

    frame = workspace.trace_frame(
        path,
        column=1,
        start=2,
        stop=18,
        max_points=4,
    )

    assert frame["row_index"].to_list() == [2, 6, 10, 14]
    assert frame["time_seconds"].to_list() == pytest.approx([0.2, 0.6, 1.0, 1.4])
    assert frame["value"].to_list() == [4.0, 12.0, 20.0, 28.0]

    with pytest.raises(ValueError, match="interactive limit"):
        workspace.trace_frame(
            path,
            column=0,
            start=0,
            stop=20,
            max_source_rows=10,
        )
