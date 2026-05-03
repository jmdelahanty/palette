from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import zarr

from fisheye.shared.zarr_helpers import resolve_zarr_run


class _FakeGroup:
    def __init__(self, *, path: str = "") -> None:
        self._children: dict[str, _FakeGroup] = {}
        self.attrs: dict[str, Any] = {}
        self.path = path

    def create_group(self, name: str) -> "_FakeGroup":
        if "/" in name:
            head, tail = name.split("/", 1)
            return self.require_group(head).create_group(tail)
        if name in self._children:
            raise ValueError(f"{name} already exists")
        child_path = f"{self.path}/{name}" if self.path else name
        child = _FakeGroup(path=child_path)
        self._children[name] = child
        return child

    def require_group(self, name: str) -> "_FakeGroup":
        if "/" in name:
            head, tail = name.split("/", 1)
            return self.require_group(head).require_group(tail)
        existing = self._children.get(name)
        if existing is not None:
            return existing
        child_path = f"{self.path}/{name}" if self.path else name
        child = _FakeGroup(path=child_path)
        self._children[name] = child
        return child

    def get(self, name: str):
        try:
            return self[name]
        except Exception:
            return None

    def group_keys(self):
        return list(self._children.keys())

    def keys(self):
        return list(self._children.keys())

    def __contains__(self, key: str) -> bool:
        try:
            _ = self[key]
            return True
        except Exception:
            return False

    def __getitem__(self, key: str) -> "_FakeGroup":
        if "/" in key:
            current: _FakeGroup = self
            for token in key.split("/"):
                current = current._children[token]
            return current
        return self._children[key]


def _build_root() -> _FakeGroup:
    root = _FakeGroup()
    runs = root.require_group("analysis/stimulus_runs")
    runs.create_group("stimulus_001")
    runs.create_group("stimulus_002")
    runs.create_group("stimulus_003")
    return root


def test_resolve_zarr_run_uses_explicit_run_name() -> None:
    root = _build_root()

    run_group, run_name = resolve_zarr_run(
        root,
        "analysis/stimulus_runs",
        "stimulus_002",
        run_label="Stimulus run",
    )

    assert run_name == "stimulus_002"
    assert run_group.path == "analysis/stimulus_runs/stimulus_002"


def test_resolve_zarr_run_uses_latest_attr_and_latest_alias() -> None:
    root = _build_root()
    root["analysis/stimulus_runs"].attrs["latest"] = b"stimulus_003"

    run_group, run_name = resolve_zarr_run(
        root,
        ("analysis", "stimulus_runs"),
        "latest",
        latest_aliases=("latest",),
        run_label="Stimulus run",
    )

    assert run_name == "stimulus_003"
    assert run_group.path == "analysis/stimulus_runs/stimulus_003"


def test_resolve_zarr_run_falls_back_to_sorted_last() -> None:
    root = _build_root()

    run_group, run_name = resolve_zarr_run(
        root,
        "analysis/stimulus_runs",
        None,
        fallback_to_sorted="last",
        run_label="Stimulus run",
    )

    assert run_name == "stimulus_003"
    assert run_group.path == "analysis/stimulus_runs/stimulus_003"


def test_resolve_zarr_run_falls_back_to_sorted_first() -> None:
    root = _build_root()

    run_group, run_name = resolve_zarr_run(
        root,
        "analysis/stimulus_runs",
        None,
        fallback_to_sorted="first",
        run_label="Stimulus run",
    )

    assert run_name == "stimulus_001"
    assert run_group.path == "analysis/stimulus_runs/stimulus_001"


def test_resolve_zarr_run_reports_missing_run_with_available_names() -> None:
    root = _build_root()

    with pytest.raises(ValueError, match="Stimulus run 'stimulus_999' not found under analysis/stimulus_runs"):
        resolve_zarr_run(
            root,
            "analysis/stimulus_runs",
            "stimulus_999",
            run_label="Stimulus run",
        )


def test_resolve_zarr_run_reports_missing_parent() -> None:
    root = _FakeGroup()

    with pytest.raises(ValueError, match="analysis/stimulus_runs not found in store"):
        resolve_zarr_run(
            root,
            "analysis/stimulus_runs",
            None,
            run_label="Stimulus run",
        )


def test_resolve_zarr_run_uses_direct_filesystem_group_for_explicit_run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = _FakeGroup()
    root.require_group("analysis/stimulus_runs")
    root._palette_fs_path = str(tmp_path / "archive.zarr")  # type: ignore[attr-defined]
    root._palette_open_mode = "r"  # type: ignore[attr-defined]
    direct_group_path = tmp_path / "archive.zarr" / "analysis" / "stimulus_runs" / "stimulus_777"
    direct_group_path.mkdir(parents=True)
    (direct_group_path / "zarr.json").write_text('{"zarr_format": 3, "node_type": "group", "attributes": {}}')
    sentinel = _FakeGroup(path="analysis/stimulus_runs/stimulus_777")

    def _fake_open_group(path: str, *, mode: str, use_consolidated: bool) -> _FakeGroup:
        assert Path(path) == direct_group_path
        assert mode == "r"
        assert use_consolidated is False
        return sentinel

    monkeypatch.setattr(zarr, "open_group", _fake_open_group)

    run_group, run_name = resolve_zarr_run(
        root,
        "analysis/stimulus_runs",
        "stimulus_777",
        run_label="Stimulus run",
    )

    assert run_name == "stimulus_777"
    assert run_group is sentinel


def test_resolve_zarr_run_uses_direct_filesystem_group_for_latest(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = _FakeGroup()
    parent = root.require_group("analysis/stimulus_runs")
    parent.attrs["latest"] = "stimulus_888"
    root._palette_fs_path = str(tmp_path / "archive.zarr")  # type: ignore[attr-defined]
    root._palette_open_mode = "r"  # type: ignore[attr-defined]
    direct_group_path = tmp_path / "archive.zarr" / "analysis" / "stimulus_runs" / "stimulus_888"
    direct_group_path.mkdir(parents=True)
    (direct_group_path / "zarr.json").write_text('{"zarr_format": 3, "node_type": "group", "attributes": {}}')
    sentinel = _FakeGroup(path="analysis/stimulus_runs/stimulus_888")

    def _fake_open_group(path: str, *, mode: str, use_consolidated: bool) -> _FakeGroup:
        assert Path(path) == direct_group_path
        assert mode == "r"
        assert use_consolidated is False
        return sentinel

    monkeypatch.setattr(zarr, "open_group", _fake_open_group)

    run_group, run_name = resolve_zarr_run(
        root,
        "analysis/stimulus_runs",
        None,
        run_label="Stimulus run",
    )

    assert run_name == "stimulus_888"
    assert run_group is sentinel


def test_resolve_zarr_run_uses_store_path_file_uri_when_palette_annotation_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = _FakeGroup()
    root.require_group("analysis/stimulus_runs")
    root.store_path = f"file://{tmp_path / 'archive.zarr'}"  # type: ignore[attr-defined]
    direct_group_path = tmp_path / "archive.zarr" / "analysis" / "stimulus_runs" / "stimulus_999"
    direct_group_path.mkdir(parents=True)
    (direct_group_path / "zarr.json").write_text('{"zarr_format": 3, "node_type": "group", "attributes": {}}')
    sentinel = _FakeGroup(path="analysis/stimulus_runs/stimulus_999")

    def _fake_open_group(path: str, *, mode: str, use_consolidated: bool) -> _FakeGroup:
        assert Path(path) == direct_group_path
        assert mode == "r"
        assert use_consolidated is False
        return sentinel

    monkeypatch.setattr(zarr, "open_group", _fake_open_group)

    run_group, run_name = resolve_zarr_run(
        root,
        "analysis/stimulus_runs",
        "stimulus_999",
        run_label="Stimulus run",
    )

    assert run_name == "stimulus_999"
    assert run_group is sentinel
