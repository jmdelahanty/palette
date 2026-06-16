from __future__ import annotations

import json
import sys
from pathlib import Path

from fisheye.utils.run_citrus_session_import import (
    build_import_command,
    build_organize_command,
    build_registry_command,
    _read_recording_dirs_from_organize_log,
    _read_zarr_paths_from_import_log,
)


def test_build_organize_command_is_single_session_import_only(tmp_path: Path) -> None:
    command = build_organize_command(
        session_dir=tmp_path / "session",
        dest_root=tmp_path / "recordings",
        log_dir=tmp_path / "logs",
        apply=True,
        rename_cams=True,
        run_video_diagnostics=False,
        run_h5_diagnostics=False,
    )

    assert command[:3] == [sys.executable, "-m", "fisheye.utils.organize_recordings"]
    assert str(tmp_path / "session") in command
    assert "--process-all" not in command
    assert "--cleanup-staging" not in command
    assert "--cleanup-empty" not in command
    assert "--apply" in command
    assert "--write-manifest" in command
    assert "--rename-cams" in command


def test_build_import_command_uses_organize_log_without_detect_or_refine(tmp_path: Path) -> None:
    command = build_import_command(
        organize_log=tmp_path / "organize.jsonl",
        log_dir=tmp_path / "import_logs",
        apply=False,
        recording_only=True,
        allow_preflight_failures=True,
    )

    assert command[:3] == [sys.executable, "-m", "fisheye.utils.import_organized_recordings_analysis"]
    assert "--organize-log" in command
    assert str(tmp_path / "organize.jsonl") in command
    assert "--dry-run" in command
    assert "--recording-only" in command
    assert "--allow-preflight-failures" in command
    assert "detect" not in " ".join(command)
    assert "refine" not in " ".join(command)


def test_build_registry_command_scans_file_list_only(tmp_path: Path) -> None:
    command = build_registry_command(
        registry=tmp_path / "registry.sqlite",
        file_list=tmp_path / "zarrs.txt",
    )

    assert command[:3] == [sys.executable, "-m", "fisheye.utils.registry_rescan"]
    assert "--file-list" in command
    assert str(tmp_path / "zarrs.txt") in command
    assert "--recursive" not in command


def test_read_zarr_paths_from_import_log_deduplicates_and_skips_missing(tmp_path: Path) -> None:
    log_path = tmp_path / "import.jsonl"
    zarr_a = tmp_path / "rec_a" / "zarr" / "rec_a_analysis.zarr"
    zarr_b = tmp_path / "rec_b" / "zarr" / "rec_b_analysis.zarr"
    rows = [
        {"event": "recording_plan", "zarr_path": str(zarr_a), "status": "ok"},
        {"event": "recording_ok", "zarr_path": str(zarr_a)},
        {"event": "recording_plan", "zarr_path": str(zarr_b), "status": "skipped"},
        {"event": "recording_plan", "zarr_path": str(tmp_path / "missing.zarr"), "status": "missing"},
        {"event": "run_end", "ok": 1},
    ]
    log_path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    assert _read_zarr_paths_from_import_log(log_path) == [zarr_a, zarr_b]


def test_read_recording_dirs_from_organize_log_requires_recording_applied(tmp_path: Path) -> None:
    log_path = tmp_path / "organize.jsonl"
    rec_a = tmp_path / "recordings" / "rec_a"
    rec_b = tmp_path / "recordings" / "rec_b"
    rows = [
        {"event": "run_start", "dest_root": str(tmp_path / "recordings")},
        {"event": "recording_plan", "dest_dir": str(rec_a)},
        {"event": "recording_applied", "dest_dir": str(rec_a)},
        {"event": "recording_applied", "dest_dir": str(rec_a)},
        {"event": "recording_applied", "dest_dir": str(rec_b)},
    ]
    log_path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    assert _read_recording_dirs_from_organize_log(log_path) == [rec_a, rec_b]


def test_read_recording_dirs_from_empty_organize_log_returns_empty(tmp_path: Path) -> None:
    log_path = tmp_path / "organize.jsonl"
    log_path.write_text(json.dumps({"event": "run_start"}) + "\n", encoding="utf-8")

    assert _read_recording_dirs_from_organize_log(log_path) == []
