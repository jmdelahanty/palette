"""Code older than the registry refuses to write unless newer migrations are additive."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from fisheye.registry.db import Registry
from fisheye.registry.migrations import (
    ADDITIVE_MIGRATIONS,
    LATEST_MIGRATION_VERSION,
    schema_compatibility_problem,
)
from fisheye.registry.shadow_publish import RegistryShadowPublishError, publish_registry_shadow


def _registry(tmp_path: Path) -> Path:
    path = tmp_path / "registry.sqlite"
    Registry(path).close()
    return path


def _add_future_migration(path: Path, *, additive) -> None:
    with sqlite3.connect(path) as connection:
        connection.execute(
            "INSERT INTO schema_version (version, name, applied_utc, additive) VALUES (?, ?, ?, ?);",
            (LATEST_MIGRATION_VERSION + 1, "future_migration", "2026-10-10T00:00:00+00:00", additive),
        )
        connection.commit()


def _publish(tmp_path: Path, path: Path):
    called = []

    def mutate(candidate: Path):
        called.append(candidate)
        return {}

    publication = publish_registry_shadow(
        canonical_registry=path, backup_path=tmp_path / "backups" / "before.sqlite",
        mutate=mutate, local_temp_root=tmp_path / "local",
    )
    return publication, called


def test_migrations_record_whether_they_are_additive(tmp_path: Path) -> None:
    path = _registry(tmp_path)
    with sqlite3.connect(path) as connection:
        rows = dict(connection.execute("SELECT version, additive FROM schema_version;").fetchall())
    assert rows[LATEST_MIGRATION_VERSION] == (1 if LATEST_MIGRATION_VERSION in ADDITIVE_MIGRATIONS else 0)
    assert rows[75] == 0
    assert 76 in ADDITIVE_MIGRATIONS and rows[76] == 1


def test_a_registry_at_this_code_s_version_is_writable(tmp_path: Path) -> None:
    path = _registry(tmp_path)
    with sqlite3.connect(path) as connection:
        assert schema_compatibility_problem(connection) is None
    _, called = _publish(tmp_path, path)
    assert len(called) == 1


def test_a_newer_additive_migration_is_writable(tmp_path: Path) -> None:
    path = _registry(tmp_path)
    _add_future_migration(path, additive=1)
    _, called = _publish(tmp_path, path)
    assert len(called) == 1


@pytest.mark.parametrize("additive", [0, None])
def test_a_newer_non_additive_or_undeclared_migration_refuses_before_any_write(tmp_path: Path, additive) -> None:
    path = _registry(tmp_path)
    _add_future_migration(path, additive=additive)
    before = path.read_bytes()
    with pytest.raises(RegistryShadowPublishError, match="not additive: .*future_migration"):
        _publish(tmp_path, path)
    assert path.read_bytes() == before
    assert not (tmp_path / "backups" / "before.sqlite").exists()


def test_a_newer_registry_without_additive_records_refuses(tmp_path: Path) -> None:
    path = tmp_path / "old_style.sqlite"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE schema_version (version INTEGER PRIMARY KEY, name TEXT NOT NULL, applied_utc TEXT NOT NULL);")
        connection.execute("INSERT INTO schema_version VALUES (?, 'x', 't');", (LATEST_MIGRATION_VERSION + 1,))
        connection.commit()
        assert "does not record whether" in schema_compatibility_problem(connection)
