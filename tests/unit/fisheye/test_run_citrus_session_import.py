from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from fisheye.utils import run_citrus_session_import as import_mod
from fisheye.utils.run_citrus_session_import import (
    build_import_command,
    _read_zarr_paths_from_import_log,
)


def test_build_import_command_uses_organize_log_without_detect_or_refine(tmp_path: Path) -> None:
    command = build_import_command(
        organize_log=tmp_path / "organize.jsonl",
        log_dir=tmp_path / "import_logs",
        apply=False,
        recording_only=True,
        registry=tmp_path / "registry.sqlite",
    )

    assert command[:3] == [sys.executable, "-m", "fisheye.utils.import_organized_recordings_analysis"]
    assert "--organize-log" in command
    assert str(tmp_path / "organize.jsonl") in command
    assert "--dry-run" in command
    assert "--recording-only" in command
    assert "--allow-preflight-failures" not in command
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

    assert _read_zarr_paths_from_import_log(log_path) == [zarr_a]


def test_new_log_selection_never_reuses_previous_invocation(tmp_path: Path) -> None:
    previous = tmp_path / "import_previous.jsonl"
    previous.write_text('{}\n')
    assert import_mod._newest_jsonl(tmp_path, "import_*.jsonl", before={previous}) is None


@pytest.mark.parametrize("payload", ['{', '[]', '{"event":"recording_failed"}'])
def test_import_log_errors_cannot_be_ignored(tmp_path: Path, payload: str) -> None:
    log = tmp_path / "import.jsonl"
    log.write_text(payload + '\n')
    with pytest.raises(ValueError):
        _read_zarr_paths_from_import_log(log)


@pytest.mark.parametrize(
    "legacy_flag",
    [
        "--transfer-v2", "--run-h5-diagnostics", "--run-video-diagnostics", "--no-rename-cams",
        # Recording context now comes only from the producer's transfer snapshot.
        "--recording-only",
    ],
)
def test_legacy_poller_options_are_gone(legacy_flag, capsys) -> None:
    with pytest.raises(SystemExit) as exc:
        import_mod.main(["/unused", legacy_flag])
    assert exc.value.code == 2
    assert "unrecognized arguments" in capsys.readouterr().err


def test_runner_dispatches_only_to_transfer_parent_workflow(monkeypatch) -> None:
    from fisheye.utils import citrus_transfer_parent_workflow as workflow

    seen = []
    monkeypatch.setattr(workflow, "run_transfer_parent_workflow", lambda args: seen.append(args) or 0)
    assert import_mod.main(["/session"]) == 0
    assert seen[0].dry_run and not seen[0].apply
