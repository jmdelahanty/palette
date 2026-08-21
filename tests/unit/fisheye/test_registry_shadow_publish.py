from __future__ import annotations

import sqlite3
from pathlib import Path
import sys

import pytest

from fisheye.registry.shadow_publish import (
    RegistryShadowPublishError,
    publish_registry_shadow,
    validate_registry_sqlite,
)


def _create_registry(path: Path, *, value: str = "before") -> None:
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            CREATE TABLE records (
                id INTEGER PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE INDEX idx_records_value ON records(value);
            """
        )
        connection.execute(
            "INSERT INTO records (id, value) VALUES (1, ?);", (value,)
        )
        connection.commit()


def _read_value(path: Path) -> str:
    with sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True) as connection:
        return str(
            connection.execute(
                "SELECT value FROM records WHERE id = 1;"
            ).fetchone()[0]
        )


def test_shadow_publication_preserves_backup_and_atomically_publishes(
    tmp_path: Path,
) -> None:
    canonical = tmp_path / "registry.sqlite"
    backup = tmp_path / "backups" / "before.sqlite"
    _create_registry(canonical)

    def mutate(candidate: Path) -> dict[str, object]:
        with sqlite3.connect(candidate) as connection:
            connection.execute(
                "UPDATE records SET value = 'after' WHERE id = 1;"
            )
            connection.commit()
        return {"status": "complete", "updated_count": 1}

    publication = publish_registry_shadow(
        canonical_registry=canonical,
        backup_path=backup,
        mutate=mutate,
        local_temp_root=tmp_path / "local",
    )

    assert _read_value(canonical) == "after"
    assert _read_value(backup) == "before"
    assert publication.source_sha256 != publication.published_sha256
    assert publication.mutation_result["updated_count"] == 1
    assert publication.source_validation.integrity_check == "ok"
    assert publication.source_validation.sqlite_runtime_version == (
        sqlite3.sqlite_version
    )
    assert publication.source_validation.sqlite_python_module_version == (
        sqlite3.version
    )
    assert publication.source_validation.python_executable == sys.executable
    assert publication.source_validation.validation_backend == "python_stdlib_sqlite3"
    assert publication.published_validation.integrity_check == "ok"
    assert not list(tmp_path.glob(".registry.sqlite.publish_tmp.*"))


def test_shadow_publication_fails_closed_on_concurrent_source_change(
    tmp_path: Path,
) -> None:
    canonical = tmp_path / "registry.sqlite"
    backup = tmp_path / "backups" / "before.sqlite"
    _create_registry(canonical)

    def mutate(candidate: Path) -> dict[str, object]:
        with sqlite3.connect(candidate) as connection:
            connection.execute(
                "UPDATE records SET value = 'candidate' WHERE id = 1;"
            )
            connection.commit()
        with sqlite3.connect(canonical) as connection:
            connection.execute(
                "UPDATE records SET value = 'concurrent' WHERE id = 1;"
            )
            connection.commit()
        return {"status": "complete"}

    with pytest.raises(
        RegistryShadowPublishError,
        match="changed during shadow mutation",
    ):
        publish_registry_shadow(
            canonical_registry=canonical,
            backup_path=backup,
            mutate=mutate,
            local_temp_root=tmp_path / "local",
        )

    assert _read_value(canonical) == "concurrent"
    assert _read_value(backup) == "before"


def test_shadow_publication_rejects_invalid_candidate_without_touching_source(
    tmp_path: Path,
) -> None:
    canonical = tmp_path / "registry.sqlite"
    backup = tmp_path / "backups" / "before.sqlite"
    _create_registry(canonical)

    def mutate(candidate: Path) -> dict[str, object]:
        candidate.write_bytes(b"not a sqlite database")
        return {"status": "complete"}

    with pytest.raises(RegistryShadowPublishError, match="validation could not read"):
        publish_registry_shadow(
            canonical_registry=canonical,
            backup_path=backup,
            mutate=mutate,
            local_temp_root=tmp_path / "local",
        )

    assert _read_value(canonical) == "before"
    assert _read_value(backup) == "before"
    assert validate_registry_sqlite(canonical).integrity_check == "ok"


def test_shadow_publication_rejects_invalid_source_before_mutation(
    tmp_path: Path,
) -> None:
    canonical = tmp_path / "registry.sqlite"
    backup = tmp_path / "backups" / "before.sqlite"
    canonical.write_bytes(b"not a sqlite database")
    called = False

    def mutate(_candidate: Path) -> dict[str, object]:
        nonlocal called
        called = True
        return {"status": "complete"}

    with pytest.raises(RegistryShadowPublishError, match="validation could not read"):
        publish_registry_shadow(
            canonical_registry=canonical,
            backup_path=backup,
            mutate=mutate,
            local_temp_root=tmp_path / "local",
        )

    assert called is False
    assert not backup.exists()
