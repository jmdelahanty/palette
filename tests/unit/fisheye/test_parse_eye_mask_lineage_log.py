from __future__ import annotations

import json
from pathlib import Path

import pytest

from fisheye.diagnostics import parse_eye_mask_lineage_log as mod


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


def test_find_latest_log_prefers_newer_file(tmp_path: Path) -> None:
    older = tmp_path / "check_eye_mask_lineage_20260211T000000Z_1.jsonl"
    newer = tmp_path / "check_eye_mask_lineage_20260212T000000Z_2.jsonl"
    _write_jsonl(older, [])
    _write_jsonl(newer, [])
    newer.touch()
    assert mod._find_latest_log(tmp_path) == newer


def test_parse_log_collects_failing_and_error_zarrs(tmp_path: Path) -> None:
    log_path = tmp_path / "lineage.jsonl"
    _write_jsonl(
        log_path,
        [
            {"event": "run_start", "run_id": "abc"},
            {"event": "zarr_checked", "zarr": "/a.zarr", "issues": False},
            {"event": "zarr_checked", "zarr": "/b.zarr", "issues": True},
            {"event": "zarr_error", "zarr": "/c.zarr"},
            {"event": "run_end", "zarr_scanned": 3, "issues": True},
        ],
    )

    parsed = mod._parse_log(log_path)
    assert parsed.run_id == "abc"
    assert parsed.failing_zarrs == ["/b.zarr"]
    assert parsed.error_zarrs == ["/c.zarr"]
    assert parsed.passing_zarrs == ["/a.zarr"]
    assert parsed.has_issues is True


def test_run_latest_and_failures_only(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys) -> None:
    log_path = tmp_path / "check_eye_mask_lineage_foo.jsonl"
    _write_jsonl(
        log_path,
        [
            {"event": "zarr_checked", "zarr": "/bad.zarr", "issues": True},
            {"event": "run_end", "issues": True},
        ],
    )
    monkeypatch.setattr(mod, "_default_log_dir", lambda: tmp_path)
    args = mod.build_parser().parse_args(["--failures-only", "--strict"])
    rc = mod.run(args)
    assert rc == 1
    out = capsys.readouterr().out.strip().splitlines()
    assert out == ["/bad.zarr"]


def test_run_rejects_latest_with_explicit_path(tmp_path: Path) -> None:
    log_path = tmp_path / "check_eye_mask_lineage_foo.jsonl"
    _write_jsonl(log_path, [])
    args = mod.build_parser().parse_args([str(log_path), "--latest"])
    with pytest.raises(ValueError, match="--latest cannot be combined"):
        mod.run(args)
