from __future__ import annotations

import json
import os
from pathlib import Path
import sqlite3
import sys

import pytest

from fisheye.registry.shadow_publish import RegistryShadowPublishError
import fisheye.utils.registry_backup as registry_backup


def _create_registry(path: Path, *, value: str = "present") -> None:
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE marker (value TEXT NOT NULL);")
        connection.execute("INSERT INTO marker VALUES (?);", (value,))
        connection.commit()


def _read_value(path: Path) -> str:
    with sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True) as connection:
        return str(connection.execute("SELECT value FROM marker;").fetchone()[0])


def test_backup_shell_cannot_fall_back_to_system_sqlite() -> None:
    repo = Path(__file__).resolve().parents[3]
    wrapper = (repo / "scripts" / "backup_palette_registry.sh").read_text(
        encoding="utf-8"
    )

    assert 'exec "$SOURCE_REPO/scripts/py" -m fisheye.utils.registry_backup "$@"' in (
        wrapper
    )
    assert "sqlite3" not in wrapper


def test_registry_backup_uses_one_runtime_for_source_and_backup(
    tmp_path: Path,
) -> None:
    registry = tmp_path / "registry.sqlite"
    backup = tmp_path / "backups" / "palette_registry_test.sqlite"
    _create_registry(registry)

    receipt = registry_backup.create_registry_backup(
        registry=registry,
        backup_path=backup,
    )

    assert _read_value(backup) == "present"
    assert receipt.status == "complete"
    assert receipt.backup_mode == "python_sqlite_backup_api"
    assert receipt.source_validation.sqlite_runtime_version == sqlite3.sqlite_version
    assert receipt.backup_validation.sqlite_runtime_version == sqlite3.sqlite_version
    assert receipt.source_validation.python_executable == sys.executable
    assert receipt.backup_validation.python_executable == sys.executable
    assert not list(backup.parent.glob(".*.tmp.*"))


def test_registry_backup_fails_closed_if_source_changes_during_copy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = tmp_path / "registry.sqlite"
    backup = tmp_path / "backups" / "palette_registry_test.sqlite"
    _create_registry(registry, value="before")
    original_backup = registry_backup._sqlite_backup

    def backup_then_mutate(source: Path, destination: Path) -> None:
        original_backup(source, destination)
        with sqlite3.connect(source) as connection:
            connection.execute("UPDATE marker SET value = 'concurrent';")
            connection.commit()

    monkeypatch.setattr(registry_backup, "_sqlite_backup", backup_then_mutate)

    with pytest.raises(RegistryShadowPublishError, match="changed during backup"):
        registry_backup.create_registry_backup(
            registry=registry,
            backup_path=backup,
        )

    assert _read_value(registry) == "concurrent"
    assert not backup.exists()
    assert not list(backup.parent.glob(".*.tmp.*"))


def test_registry_backup_prunes_only_old_matching_backups_after_success(
    tmp_path: Path,
) -> None:
    registry = tmp_path / "registry.sqlite"
    backup_dir = tmp_path / "backups"
    backup = backup_dir / "palette_registry_new.sqlite"
    old_backup = backup_dir / "palette_registry_old.sqlite"
    unrelated = backup_dir / "other.sqlite"
    _create_registry(registry)
    backup_dir.mkdir()
    old_backup.write_bytes(b"old")
    unrelated.write_bytes(b"unrelated")
    old_timestamp = 1_000_000.0
    os.utime(old_backup, (old_timestamp, old_timestamp))
    os.utime(unrelated, (old_timestamp, old_timestamp))

    receipt = registry_backup.create_registry_backup(
        registry=registry,
        backup_path=backup,
        days_to_keep=7,
    )

    assert receipt.deleted_old_backups == (str(old_backup.resolve()),)
    assert backup.exists()
    assert not old_backup.exists()
    assert unrelated.exists()


def test_registry_backup_cli_writes_runtime_bound_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = tmp_path / "registry.sqlite"
    backup_dir = tmp_path / "backups"
    result_json = tmp_path / "receipt.json"
    backup = backup_dir / "palette_registry_fixed.sqlite"
    _create_registry(registry)
    monkeypatch.setattr(registry_backup, "_default_backup_path", lambda _path: backup)

    status = registry_backup.main(
        [
            "--registry",
            str(registry),
            "--backup-dir",
            str(backup_dir),
            "--days-to-keep",
            "7",
            "--result-json",
            str(result_json),
        ]
    )

    assert status == 0
    payload = json.loads(result_json.read_text(encoding="utf-8"))
    assert payload["schema_id"] == "palette.registry_backup_receipt"
    assert payload["backup_path"] == str(backup.resolve())
    assert payload["source_validation"]["sqlite_runtime_version"] == (
        sqlite3.sqlite_version
    )
    assert payload["backup_validation"]["sqlite_runtime_version"] == (
        sqlite3.sqlite_version
    )
