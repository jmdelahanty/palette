from __future__ import annotations

import json
import sys
from pathlib import Path

from fisheye.utils import run_citrus_session_import as import_mod
from fisheye.utils.run_citrus_session_import import (
    CommandRecord,
    build_import_command,
    build_organize_command,
    _read_recording_dirs_from_organize_log,
    _read_zarr_paths_from_import_log,
    _organize_failure_reason,
)


def test_build_organize_command_is_single_session_import_only(tmp_path: Path) -> None:
    command = build_organize_command(
        session_dir=tmp_path / "session",
        dest_root=tmp_path / "recordings",
        log_dir=tmp_path / "logs",
        apply=True,
        rename_cams=True,
        recording_only=False,
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
    assert "--external-ipc-recording-only" not in command


def test_build_organize_command_recording_only_uses_external_ipc_no_h5_mode(tmp_path: Path) -> None:
    command = build_organize_command(
        session_dir=tmp_path / "session",
        dest_root=tmp_path / "recordings",
        log_dir=tmp_path / "logs",
        apply=True,
        rename_cams=True,
        recording_only=True,
        run_video_diagnostics=True,
        run_h5_diagnostics=True,
    )

    assert "--external-ipc-recording-only" in command
    assert "--run-video-diagnostics" in command
    assert "--run-h5-diagnostics" not in command


def test_build_organize_command_dry_run_suppresses_apply_only_diagnostics(tmp_path: Path) -> None:
    command = build_organize_command(
        session_dir=tmp_path / "session",
        dest_root=tmp_path / "recordings",
        log_dir=tmp_path / "logs",
        apply=False,
        rename_cams=True,
        recording_only=False,
        run_video_diagnostics=True,
        run_h5_diagnostics=True,
    )

    assert "--dry-run" in command
    assert "--run-video-diagnostics" not in command
    assert "--run-h5-diagnostics" not in command


def test_build_import_command_uses_organize_log_without_detect_or_refine(tmp_path: Path) -> None:
    command = build_import_command(
        organize_log=tmp_path / "organize.jsonl",
        log_dir=tmp_path / "import_logs",
        apply=False,
        recording_only=True,
        allow_preflight_failures=True,
        registry=tmp_path / "registry.sqlite",
    )

    assert command[:3] == [sys.executable, "-m", "fisheye.utils.import_organized_recordings_analysis"]
    assert "--organize-log" in command
    assert str(tmp_path / "organize.jsonl") in command
    assert "--dry-run" in command
    assert "--recording-only" in command
    assert "--allow-preflight-failures" in command
    assert "--registry" in command
    assert str(tmp_path / "registry.sqlite") in command
    assert "detect" not in " ".join(command)
    assert "refine" not in " ".join(command)


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


def test_applied_organizer_with_zero_recordings_fails_closed(tmp_path: Path) -> None:
    log_path = tmp_path / "organize.jsonl"
    log_path.write_text(json.dumps({"event": "batch_start"}) + "\n", encoding="utf-8")

    assert _organize_failure_reason(
        apply=True,
        returncode=0,
        organize_log=log_path,
        organized_recording_dirs=[],
    ) == "applied organizer produced zero recording_applied entries"


def test_dry_run_may_have_no_applied_recordings(tmp_path: Path) -> None:
    log_path = tmp_path / "organize.jsonl"
    log_path.write_text(json.dumps({"event": "batch_start"}) + "\n", encoding="utf-8")

    assert (
        _organize_failure_reason(
            apply=False,
            returncode=0,
            organize_log=log_path,
            organized_recording_dirs=[],
        )
        is None
    )


def test_applied_main_returns_failure_and_writes_disposition_for_zero_recordings(
    tmp_path: Path,
    monkeypatch,
) -> None:
    session = tmp_path / "staging" / "session"
    session.mkdir(parents=True)
    run_dir = tmp_path / "run"
    status_path = run_dir / "status.json"

    def fake_run_command(command, *, name: str, run_dir: Path) -> CommandRecord:
        assert name == "01_organize_recordings"
        log = run_dir / "organize_recordings" / "organize_recordings_test.jsonl"
        log.write_text(json.dumps({"event": "batch_start"}) + "\n", encoding="utf-8")
        return CommandRecord(
            name=name,
            command=list(command),
            returncode=0,
            stdout_path=str(run_dir / "stdout.txt"),
            stderr_path=str(run_dir / "stderr.txt"),
        )

    monkeypatch.setattr(import_mod, "_run_command", fake_run_command)

    result = import_mod.main(
        [
            str(session),
            "--dest-root",
            str(tmp_path / "recordings"),
            "--run-dir",
            str(run_dir),
            "--status-json",
            str(status_path),
            "--apply",
        ]
    )

    assert result == 1
    status = json.loads(status_path.read_text(encoding="utf-8"))
    assert status["status"] == "failed"
    assert status["organized_recording_dirs"] == []
    disposition = json.loads(
        (session / "_palette_batch_disposition.json").read_text(encoding="utf-8")
    )
    assert disposition["workflow"]["status"] == "failed"
    assert disposition["cleanup_assessment"]["safe_to_delete_batch"] is False
