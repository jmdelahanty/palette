"""Safely mutate a shared SQLite registry through a local shadow copy.

Palette's canonical registry is stored on a multi-host network filesystem.  SQLite
rollback journals avoid WAL's cross-host shared-memory hazard, but they cannot make
network-filesystem locking reliable.  This module therefore keeps SQLite writes off
the shared file: validate and snapshot the source, mutate a local copy, validate the
candidate, then publish one fully formed database with an atomic rename.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import fcntl
import hashlib
import os
from pathlib import Path
import shutil
import sqlite3
import sys
import tempfile
from typing import Any, Callable, Iterator, Mapping
import uuid


class RegistryShadowPublishError(RuntimeError):
    """Raised when a registry shadow mutation cannot be published safely."""


@dataclass(frozen=True)
class RegistryValidation:
    path: str
    integrity_check: str
    foreign_key_issue_count: int
    sqlite_runtime_version: str
    sqlite_python_module_version: str
    python_executable: str
    validation_backend: str = "python_stdlib_sqlite3"

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RegistryShadowPublication:
    canonical_registry: str
    backup_path: str
    source_sha256: str
    published_sha256: str
    source_size_bytes: int
    published_size_bytes: int
    source_validation: RegistryValidation
    candidate_validation: RegistryValidation
    staged_validation: RegistryValidation
    published_validation: RegistryValidation
    mutation_result: Mapping[str, Any]
    publication_mode: str = "local_shadow_copy_atomic_replace"

    def to_json(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["mutation_result"] = dict(self.mutation_result)
        return payload


def _readonly_connection(path: Path) -> sqlite3.Connection:
    resolved = path.expanduser().resolve()
    return sqlite3.connect(f"{resolved.as_uri()}?mode=ro", uri=True)


def validate_registry_sqlite(path: str | Path) -> RegistryValidation:
    """Run the complete SQLite and foreign-key checks against one closed database."""

    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file() or resolved.stat().st_size <= 0:
        raise RegistryShadowPublishError(
            f"Registry is missing or empty: {resolved}"
        )
    try:
        with _readonly_connection(resolved) as connection:
            integrity_rows = [
                str(row[0]) for row in connection.execute("PRAGMA integrity_check;")
            ]
            foreign_key_rows = connection.execute(
                "PRAGMA foreign_key_check;"
            ).fetchall()
    except sqlite3.Error as exc:
        raise RegistryShadowPublishError(
            f"Registry validation could not read {resolved}: {exc}"
        ) from exc
    if integrity_rows != ["ok"]:
        preview = "; ".join(integrity_rows[:10])
        raise RegistryShadowPublishError(
            f"Registry integrity_check failed for {resolved}: {preview}"
        )
    if foreign_key_rows:
        preview = "; ".join(str(tuple(row)) for row in foreign_key_rows[:10])
        raise RegistryShadowPublishError(
            f"Registry foreign_key_check failed for {resolved}: {preview}"
        )
    return RegistryValidation(
        path=str(resolved),
        integrity_check="ok",
        foreign_key_issue_count=0,
        sqlite_runtime_version=sqlite3.sqlite_version,
        sqlite_python_module_version=sqlite3.version,
        python_executable=sys.executable,
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _sqlite_backup(source: Path, destination: Path) -> None:
    if destination.exists():
        raise RegistryShadowPublishError(
            f"SQLite backup destination already exists: {destination}"
        )
    with _readonly_connection(source) as source_connection:
        with sqlite3.connect(str(destination)) as destination_connection:
            source_connection.backup(destination_connection)
            destination_connection.commit()
    _fsync_file(destination)


def _copy_without_overwrite(source: Path, destination: Path, *, mode: int) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        validate_registry_sqlite(destination)
        if _sha256_file(source) != _sha256_file(destination):
            raise RegistryShadowPublishError(
                "Registry backup already exists with different content: "
                f"{destination}"
            )
        return
    temporary = destination.parent / f".{destination.name}.tmp.{uuid.uuid4().hex}"
    try:
        shutil.copyfile(source, temporary)
        os.chmod(temporary, mode)
        _fsync_file(temporary)
        try:
            os.link(temporary, destination)
        except FileExistsError as exc:
            raise RegistryShadowPublishError(
                f"Registry backup already exists: {destination}"
            ) from exc
        _fsync_directory(destination.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _require_no_sqlite_sidecars(path: Path) -> None:
    sidecars = [
        candidate
        for candidate in (
            Path(f"{path}-journal"),
            Path(f"{path}-wal"),
            Path(f"{path}-shm"),
        )
        if candidate.exists()
    ]
    if sidecars:
        raise RegistryShadowPublishError(
            "Canonical registry has active SQLite sidecars; refusing shadow "
            f"publication: {[str(item) for item in sidecars]}"
        )


@contextmanager
def _publication_lock(canonical: Path) -> Iterator[None]:
    lock_path = canonical.parent / f".{canonical.name}.writer.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def publish_registry_shadow(
    *,
    canonical_registry: str | Path,
    backup_path: str | Path,
    mutate: Callable[[Path], Mapping[str, Any]],
    local_temp_root: str | Path | None = None,
) -> RegistryShadowPublication:
    """Mutate a local registry snapshot and atomically publish the valid result.

    The mutation callback receives a node-local SQLite path.  It must close every
    connection before returning.  The canonical database is never opened writable.
    """

    canonical = Path(canonical_registry).expanduser().resolve()
    backup = Path(backup_path).expanduser().resolve()
    if canonical == backup:
        raise RegistryShadowPublishError(
            "Registry backup path must differ from the canonical registry."
        )
    temp_parent = (
        Path(local_temp_root).expanduser().resolve()
        if local_temp_root is not None
        else None
    )
    if temp_parent is not None:
        temp_parent.mkdir(parents=True, exist_ok=True)

    with _publication_lock(canonical):
        _require_no_sqlite_sidecars(canonical)
        source_validation = validate_registry_sqlite(canonical)
        source_stat = canonical.stat()
        source_mode = source_stat.st_mode & 0o777
        source_sha256 = _sha256_file(canonical)

        with tempfile.TemporaryDirectory(
            prefix="palette-registry-shadow-",
            dir=str(temp_parent) if temp_parent is not None else None,
        ) as temporary_directory:
            temporary_root = Path(temporary_directory)
            source_snapshot = temporary_root / "source.sqlite"
            candidate = temporary_root / "candidate.sqlite"
            _sqlite_backup(canonical, source_snapshot)
            validate_registry_sqlite(source_snapshot)
            _copy_without_overwrite(source_snapshot, backup, mode=source_mode)
            shutil.copyfile(source_snapshot, candidate)
            os.chmod(candidate, source_mode)

            mutation_result = dict(mutate(candidate))
            candidate_validation = validate_registry_sqlite(candidate)
            published_sha256 = _sha256_file(candidate)
            published_size = candidate.stat().st_size

            _require_no_sqlite_sidecars(canonical)
            if canonical.stat().st_size != source_stat.st_size:
                raise RegistryShadowPublishError(
                    "Canonical registry size changed during shadow mutation; "
                    "refusing to overwrite a concurrent update."
                )
            if _sha256_file(canonical) != source_sha256:
                raise RegistryShadowPublishError(
                    "Canonical registry content changed during shadow mutation; "
                    "refusing to overwrite a concurrent update."
                )

            staged = canonical.parent / (
                f".{canonical.name}.publish_tmp.{uuid.uuid4().hex}"
            )
            try:
                shutil.copyfile(candidate, staged)
                os.chmod(staged, source_mode)
                _fsync_file(staged)
                staged_validation = validate_registry_sqlite(staged)
                if _sha256_file(staged) != published_sha256:
                    raise RegistryShadowPublishError(
                        "Shared-filesystem registry staging copy changed bytes."
                    )
                _require_no_sqlite_sidecars(canonical)
                if _sha256_file(canonical) != source_sha256:
                    raise RegistryShadowPublishError(
                        "Canonical registry changed immediately before publication."
                    )
                os.replace(staged, canonical)
                _fsync_directory(canonical.parent)
            finally:
                staged.unlink(missing_ok=True)

        published_validation = validate_registry_sqlite(canonical)
        if _sha256_file(canonical) != published_sha256:
            raise RegistryShadowPublishError(
                "Published registry hash differs from the validated local candidate."
            )

    return RegistryShadowPublication(
        canonical_registry=str(canonical),
        backup_path=str(backup),
        source_sha256=source_sha256,
        published_sha256=published_sha256,
        source_size_bytes=int(source_stat.st_size),
        published_size_bytes=int(published_size),
        source_validation=source_validation,
        candidate_validation=candidate_validation,
        staged_validation=staged_validation,
        published_validation=published_validation,
        mutation_result=mutation_result,
    )


__all__ = [
    "RegistryShadowPublication",
    "RegistryShadowPublishError",
    "RegistryValidation",
    "publish_registry_shadow",
    "validate_registry_sqlite",
]
