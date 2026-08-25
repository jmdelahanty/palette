from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path
import sys

import pytest

import fisheye.registry.db as registry_db
import fisheye.registry.shadow_publish as shadow_publish
from fisheye.registry.shadow_publish import (
    REGISTRY_SHADOW_BACKUP_DIR_ENV,
    REGISTRY_SHADOW_TEMP_ROOT_ENV,
    REGISTRY_WRITER_HOST_ENV,
    REGISTRY_WRITER_LOCK_PATH_ENV,
    RegistryShadowPublishError,
    publish_registry_shadow,
    shadow_synchronize_recording_import,
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


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def test_shadow_synchronize_temp_registry_uses_candidate_and_publishes_backup(
    monkeypatch,
    tmp_path: Path,
) -> None:
    canonical = tmp_path / "registry.sqlite"
    zarr_path = tmp_path / "recording.zarr"
    _create_registry(canonical)
    observed: dict[str, object] = {}

    class _FakeRegistry:
        def __init__(self, path: Path) -> None:
            observed["registry_path"] = Path(path)
            observed["exists_during_callback"] = Path(path).is_file()

        def synchronize_recording_import(self, **kwargs: object) -> str:
            observed.update(kwargs)
            candidate = observed["registry_path"]
            assert isinstance(candidate, Path)
            with sqlite3.connect(candidate) as connection:
                connection.execute("UPDATE records SET value = 'after' WHERE id = 1;")
                connection.commit()
            return "dataset-from-candidate"

        def close(self) -> None:
            observed["closed"] = True

    # Registry is imported inside the callback, so patching the source module
    # keeps this test independent of the full production registry schema.
    monkeypatch.setattr(registry_db, "Registry", _FakeRegistry)

    publication = shadow_synchronize_recording_import(
        canonical_registry=canonical,
        zarr_path=zarr_path,
        receipt={"receipt": "test"},
        decided_by="pytest",
    )

    callback_path = observed["registry_path"]
    assert isinstance(callback_path, Path)
    assert callback_path != canonical
    assert callback_path.name == "candidate.sqlite"
    assert observed["exists_during_callback"] is True
    assert observed["closed"] is True
    assert observed["zarr_path"] == zarr_path.resolve()
    assert observed["decided_by"] == "pytest"
    assert _read_value(canonical) == "after"
    backup = Path(publication.backup_path)
    assert backup.is_file()
    assert _read_value(backup) == "before"
    assert publication.mutation_result == {
        "operation": "synchronize_recording_import",
        "dataset_id": "dataset-from-candidate",
        "zarr_path": str(zarr_path.resolve()),
        "decided_by": "pytest",
    }


def test_shadow_synchronize_non_temp_registry_requires_configuration_before_registry(
    monkeypatch,
    tmp_path: Path,
) -> None:
    canonical = tmp_path / "shared" / "registry.sqlite"
    canonical.parent.mkdir()
    _create_registry(canonical)
    fake_temp_root = tmp_path / "mock-system-temp"
    monkeypatch.setattr(
        shadow_publish.tempfile, "gettempdir", lambda: str(fake_temp_root)
    )
    for name in (
        REGISTRY_WRITER_HOST_ENV,
        REGISTRY_WRITER_LOCK_PATH_ENV,
        REGISTRY_SHADOW_TEMP_ROOT_ENV,
        REGISTRY_SHADOW_BACKUP_DIR_ENV,
    ):
        monkeypatch.delenv(name, raising=False)

    class _UnexpectedRegistry:
        def __init__(self, _path: Path) -> None:
            raise AssertionError("Registry must not be opened before config checks")

    monkeypatch.setattr(registry_db, "Registry", _UnexpectedRegistry)

    with pytest.raises(
        RegistryShadowPublishError,
        match="shared registry publication requires explicit single-writer configuration",
    ):
        shadow_synchronize_recording_import(
            canonical_registry=canonical,
            zarr_path=tmp_path / "recording.zarr",
            receipt=None,
            decided_by="pytest",
        )


def test_shadow_synchronize_non_temp_registry_rejects_host_mismatch(
    monkeypatch,
    tmp_path: Path,
) -> None:
    canonical = tmp_path / "shared" / "registry.sqlite"
    canonical.parent.mkdir()
    _create_registry(canonical)
    fake_temp_root = tmp_path / "mock-system-temp"
    monkeypatch.setattr(
        shadow_publish.tempfile, "gettempdir", lambda: str(fake_temp_root)
    )
    monkeypatch.setenv(REGISTRY_WRITER_HOST_ENV, "designated-writer")
    monkeypatch.setenv(REGISTRY_WRITER_LOCK_PATH_ENV, str(tmp_path / "writer.lock"))
    monkeypatch.setenv(REGISTRY_SHADOW_TEMP_ROOT_ENV, str(tmp_path / "local"))
    monkeypatch.setenv(REGISTRY_SHADOW_BACKUP_DIR_ENV, str(tmp_path / "backups"))
    monkeypatch.setattr(shadow_publish.socket, "gethostname", lambda: "other-host")

    class _UnexpectedRegistry:
        def __init__(self, _path: Path) -> None:
            raise AssertionError("Registry must not be opened after host rejection")

    monkeypatch.setattr(registry_db, "Registry", _UnexpectedRegistry)

    with pytest.raises(
        RegistryShadowPublishError,
        match="restricted to the designated writer host",
    ):
        shadow_synchronize_recording_import(
            canonical_registry=canonical,
            zarr_path=tmp_path / "recording.zarr",
            receipt=None,
            decided_by="pytest",
        )


def test_shadow_synchronize_candidate_failure_preserves_canonical_hash(
    monkeypatch,
    tmp_path: Path,
) -> None:
    canonical = tmp_path / "registry.sqlite"
    _create_registry(canonical)
    source_hash = _sha256(canonical)
    observed: dict[str, Path] = {}

    class _FailingRegistry:
        def __init__(self, path: Path) -> None:
            observed["registry_path"] = Path(path)

        def synchronize_recording_import(self, **_kwargs: object) -> str:
            candidate = observed["registry_path"]
            with sqlite3.connect(candidate) as connection:
                connection.execute(
                    "UPDATE records SET value = 'candidate-only' WHERE id = 1;"
                )
                connection.commit()
            raise RegistryShadowPublishError("candidate synchronization failed")

        def close(self) -> None:
            pass

    monkeypatch.setattr(registry_db, "Registry", _FailingRegistry)

    with pytest.raises(
        RegistryShadowPublishError,
        match="candidate synchronization failed",
    ):
        shadow_synchronize_recording_import(
            canonical_registry=canonical,
            zarr_path=tmp_path / "recording.zarr",
            receipt=None,
            decided_by="pytest",
        )

    assert _sha256(canonical) == source_hash
    assert _read_value(canonical) == "before"
    assert observed["registry_path"] != canonical
