from __future__ import annotations

from pathlib import Path
import sqlite3
from types import SimpleNamespace

import pytest

from fisheye.diagnostics import audit_arena_geometry_sources as module


def _registry(path: Path, rows: list[tuple[str, str, str, str, str, str]]) -> None:
    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            CREATE TABLE dataset_context_current (
                dataset_id TEXT,
                recording_id TEXT,
                zarr_path TEXT,
                dataset_status TEXT,
                zarr_use TEXT,
                protocol_name TEXT
            )
            """
        )
        connection.executemany(
            "INSERT INTO dataset_context_current VALUES (?, ?, ?, ?, ?, ?)",
            rows,
        )


def test_active_analysis_archives_is_read_only_and_protocol_scoped(tmp_path: Path) -> None:
    registry = tmp_path / "registry.sqlite"
    _registry(
        registry,
        [
            ("d2", "r2", "/tmp/r2.zarr", "active", "analysis", "GoodCopBadCop"),
            ("d1", "r1", "/tmp/r1.zarr", "active", "analysis", "GoodCopBadCop"),
            ("d3", "r3", "/tmp/r3.zarr", "missing", "analysis", "GoodCopBadCop"),
            ("d4", "r4", "/tmp/r4.zarr", "active", "training", "GoodCopBadCop"),
            ("d5", "r5", "/tmp/r5.zarr", "active", "analysis", "Batman"),
        ],
    )

    records = module._active_analysis_archives(
        registry,
        protocol_name="GoodCopBadCop",
    )

    assert [record["recording_id"] for record in records] == ["r1", "r2"]


def test_active_analysis_archives_rejects_ambiguous_recording(tmp_path: Path) -> None:
    registry = tmp_path / "registry.sqlite"
    _registry(
        registry,
        [
            ("d1", "r1", "/tmp/a.zarr", "active", "analysis", "GoodCopBadCop"),
            ("d2", "r1", "/tmp/b.zarr", "active", "analysis", "GoodCopBadCop"),
        ],
    )

    with pytest.raises(ValueError, match="exactly one active analysis archive"):
        module._active_analysis_archives(
            registry,
            protocol_name="GoodCopBadCop",
        )


class _FakeGroup:
    def __init__(self, children: dict[str, object], attrs: dict[str, object] | None = None):
        self.children = children
        self.attrs = attrs or {}
        self.store = SimpleNamespace(close=lambda: None)

    def __getitem__(self, key: str) -> object:
        if key not in self.children:
            raise KeyError(key)
        return self.children[key]


def test_inspect_archive_records_exact_selector_and_geometry(monkeypatch: pytest.MonkeyPatch) -> None:
    run = _FakeGroup({}, {"pixels_per_mm_projector": 12.5})
    parent = _FakeGroup({}, {"latest_complete": "run-1"})
    root = _FakeGroup(
        {
            module._CHASER_PARENT_PATH: parent,
            f"{module._CHASER_PARENT_PATH}/run-1": run,
        }
    )
    monkeypatch.setattr(module, "open_zarr_root", lambda *_args, **_kwargs: root)
    monkeypatch.setattr(
        module,
        "resolve_chaser_distance_run_path",
        lambda *_args, **_kwargs: ("run-1", f"{module._CHASER_PARENT_PATH}/run-1"),
    )
    monkeypatch.setattr(
        module,
        "resolve_arena_geometry",
        lambda *_args, **_kwargs: (
            SimpleNamespace(
                status="dish_mask",
                source="analysis_metadata.dish_mask",
                shape="circle",
            ),
            [],
        ),
    )

    result = module.inspect_archive(
        {"dataset_id": "d1", "recording_id": "r1", "zarr_path": "/tmp/r1.zarr"}
    )

    assert result["status"] == "ok"
    assert result["selector"] == "latest_complete"
    assert result["run_name"] == "run-1"
    assert result["arena_geometry_status"] == "dish_mask"
    assert result["arena_geometry_source"] == "analysis_metadata.dish_mask"


def test_inspect_archive_marks_missing_parent_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        module,
        "open_zarr_root",
        lambda *_args, **_kwargs: _FakeGroup({}),
    )

    result = module.inspect_archive(
        {"dataset_id": "d1", "recording_id": "r1", "zarr_path": "/tmp/r1.zarr"}
    )

    assert result["status"] == "unavailable"
    assert result["unavailable_reason"] == "no_chaser_distance_runs_parent"

