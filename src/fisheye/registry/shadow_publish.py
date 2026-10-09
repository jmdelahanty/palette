"""Safely mutate a shared SQLite registry through a local shadow copy.

Palette's canonical registry is stored on a multi-host network filesystem.  SQLite
rollback journals avoid WAL's cross-host shared-memory hazard, but they cannot make
network-filesystem locking reliable.  This module therefore keeps SQLite writes off
the shared file: validate and snapshot the source, mutate a local copy, validate the
candidate, then publish one fully formed database with an atomic rename.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
import fcntl
import hashlib
import os
from pathlib import Path
import shutil
import socket
import sqlite3
import sys
import tempfile
from typing import Any, Callable, Iterator, Mapping, Sequence
import uuid


class RegistryShadowPublishError(RuntimeError):
    """Raised when a registry shadow mutation cannot be published safely."""


class RegistryProducerCommitMismatch(RegistryShadowPublishError):
    """A receipt was produced by another commit than this registering checkout.

    The identity authority binds a receipt only from its producing checkout,
    so this can never succeed from this deployment; it is detected before any
    backup or candidate copy is made.
    """

    def __init__(
        self,
        message: str,
        *,
        zarr_path: str,
        receipt_producer_git_sha: str,
        registrar_git_sha: str | None,
        registrar_git_dirty: bool | None,
    ):
        super().__init__(message)
        self.zarr_path = zarr_path
        self.receipt_producer_git_sha = receipt_producer_git_sha
        self.registrar_git_sha = registrar_git_sha
        self.registrar_git_dirty = registrar_git_dirty


REGISTRY_WRITER_HOST_ENV = "PALETTE_REGISTRY_WRITER_HOST"
REGISTRY_WRITER_LOCK_PATH_ENV = "PALETTE_REGISTRY_WRITER_LOCK_PATH"
REGISTRY_SHADOW_TEMP_ROOT_ENV = "PALETTE_REGISTRY_SHADOW_TEMP_ROOT"
REGISTRY_SHADOW_BACKUP_DIR_ENV = "PALETTE_REGISTRY_SHADOW_BACKUP_DIR"


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


def validate_registry_sqlite_copy(
    path: str | Path, *, temp_root: str | Path | None = None
) -> RegistryValidation:
    """Validate a registry by checking a byte-identical local copy of it.

    ``PRAGMA integrity_check`` and ``foreign_key_check`` are a function of the
    database file's bytes when no ``-wal``/``-journal`` sidecar exists. Run in
    place on NFS they read the file in many small round trips (about 200 s for
    the 71 MB canonical registry, 2026-10-09); on a local copy they take about
    0.1 s. The copy is proven identical: the source's sha256 before the copy,
    the copy's, and the source's after it must all agree.
    """

    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file() or resolved.stat().st_size <= 0:
        raise RegistryShadowPublishError(f"Registry is missing or empty: {resolved}")
    _require_no_sqlite_sidecars(resolved)
    if temp_root is not None:
        Path(temp_root).mkdir(parents=True, exist_ok=True)
    before = _sha256_file(resolved)
    with tempfile.TemporaryDirectory(
        prefix="palette-registry-validate-",
        dir=str(temp_root) if temp_root is not None else None,
        ignore_cleanup_errors=True,
    ) as temporary_directory:
        local = Path(temporary_directory) / "registry.sqlite"
        shutil.copyfile(resolved, local)
        copied = _sha256_file(local)
        after = _sha256_file(resolved)
        if not before == copied == after:
            raise RegistryShadowPublishError(
                f"Registry changed while being copied for validation: {resolved}"
            )
        validation = validate_registry_sqlite(local)
    return replace(
        validation,
        path=str(resolved),
        validation_backend="python_stdlib_sqlite3_local_byte_copy",
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
        validate_registry_sqlite_copy(destination)
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


@contextmanager
def _host_writer_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _configured_shadow_paths(
    canonical: Path,
) -> tuple[Path, Path, Path]:
    temporary_root = Path(tempfile.gettempdir()).resolve()
    if canonical.is_relative_to(temporary_root):
        token = hashlib.sha256(str(canonical).encode("utf-8")).hexdigest()[:16]
        return (
            temporary_root / f"palette-registry-writer-{token}.lock",
            canonical.parent / ".palette-registry-shadow-tmp",
            canonical.parent / ".palette-registry-backups",
        )

    configured = {
        REGISTRY_WRITER_HOST_ENV: os.environ.get(REGISTRY_WRITER_HOST_ENV),
        REGISTRY_WRITER_LOCK_PATH_ENV: os.environ.get(
            REGISTRY_WRITER_LOCK_PATH_ENV
        ),
        REGISTRY_SHADOW_TEMP_ROOT_ENV: os.environ.get(
            REGISTRY_SHADOW_TEMP_ROOT_ENV
        ),
        REGISTRY_SHADOW_BACKUP_DIR_ENV: os.environ.get(
            REGISTRY_SHADOW_BACKUP_DIR_ENV
        ),
    }
    missing = [name for name, value in configured.items() if not value]
    if missing:
        raise RegistryShadowPublishError(
            "shared registry publication requires explicit single-writer "
            "configuration: " + ", ".join(sorted(missing))
        )
    expected_host = str(configured[REGISTRY_WRITER_HOST_ENV])
    current_host = socket.gethostname()
    if current_host != expected_host:
        raise RegistryShadowPublishError(
            "registry publication is restricted to the designated writer host: "
            f"expected={expected_host!r}, current={current_host!r}"
        )
    return (
        Path(str(configured[REGISTRY_WRITER_LOCK_PATH_ENV])).expanduser().resolve(),
        Path(str(configured[REGISTRY_SHADOW_TEMP_ROOT_ENV])).expanduser().resolve(),
        Path(str(configured[REGISTRY_SHADOW_BACKUP_DIR_ENV])).expanduser().resolve(),
    )


def _publish_through_configured_writer(
    canonical: Path,
    *,
    backup_label: str,
    mutate: Callable[[Path], Mapping[str, Any]],
) -> RegistryShadowPublication:
    """One shadow publication under the designated writer's host-local mutex."""

    host_lock, local_temp_root, backup_dir = _configured_shadow_paths(canonical)
    local_temp_root.mkdir(parents=True, exist_ok=True)
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup = backup_dir / (f"{canonical.name}.before-{backup_label}-{uuid.uuid4().hex}.sqlite")
    with _host_writer_lock(host_lock):
        return publish_registry_shadow(
            canonical_registry=canonical,
            backup_path=backup,
            mutate=mutate,
            local_temp_root=local_temp_root,
        )


def preflight_recording_import_receipts(imports: Sequence[tuple[Path, object | None]]) -> None:
    """Refuse, before any backup, a receipt this checkout can never bind.

    Runs the identity authority's own live receipt verification with its
    producing-commit requirement (``_verify_live_import_receipt(...,
    require_current_producer_code=True)``), the same check the mutation would
    hit only after the full registry backup had been copied. Only actual
    :class:`RecordingImportReceipt` objects are checked; ``None`` (refresh of
    an already-bound import) needs no producing commit.
    """

    from fisheye.registry.recording_identity_authority import (
        RecordingIdentityProjectionConflict,
        _verify_live_import_receipt,
        collect_regular_source_recording_identity,
    )
    from fisheye.shared.recording_import_receipt import RecordingImportReceipt

    for zarr_path, receipt in imports:
        if not isinstance(receipt, RecordingImportReceipt):
            continue
        target = Path(zarr_path).expanduser().resolve()
        try:
            _verify_live_import_receipt(
                resolved_path=target,
                evidence=collect_regular_source_recording_identity(target),
                receipt=receipt,
                require_current_producer_code=True,
            )
        except RecordingIdentityProjectionConflict as exc:
            if "producer commit" not in str(exc):
                raise
            import fisheye.registry.recording_identity_authority as authority
            from fisheye.shared.run_provenance import git_identity

            # Reported for the operator only; the decision is the authority's.
            code = git_identity(cwd=Path(authority.__file__).resolve().parents[3])
            if code.get("git_sha") is None or code.get("git_dirty") is None:
                # git itself failed: not evidence of another commit. Retryable.
                raise RegistryShadowPublishError(
                    "the registering checkout's git identity is unavailable "
                    f"({code.get('git_unavailable_reason') or code.get('git_dirty_unavailable_reason')}); retry"
                ) from exc
            raise RegistryProducerCommitMismatch(
                f"receipt for {target} was produced by commit "
                f"{receipt.producer_git_sha}; this registering checkout is "
                f"{code.get('git_sha')} (dirty={code.get('git_dirty')}). Register it "
                "from a deployment at the receipt's producer commit.",
                zarr_path=str(target),
                receipt_producer_git_sha=receipt.producer_git_sha,
                registrar_git_sha=code.get("git_sha"),
                registrar_git_dirty=code.get("git_dirty"),
            ) from exc


def _require_decided_by(decided_by: object) -> None:
    if type(decided_by) is not str or not decided_by.strip():
        raise RegistryShadowPublishError("decided_by must be non-empty text")


def shadow_synchronize_recording_import(
    *,
    canonical_registry: str | Path,
    zarr_path: Path,
    receipt: object | None,
    decided_by: str,
) -> RegistryShadowPublication:
    """Synchronize one import without opening the shared registry writable.

    Non-temporary registries require a designated host, a host-local mutex, a
    node-local candidate directory, and a durable backup directory.  The
    existing NFS-side publication lock and source hash checks remain a second
    fence, but cooperating callers must all use this gateway.
    """

    canonical = Path(canonical_registry).expanduser().resolve()
    target = Path(zarr_path).expanduser().resolve()
    _require_decided_by(decided_by)
    preflight_recording_import_receipts([(target, receipt)])

    def mutate(candidate: Path) -> Mapping[str, Any]:
        from fisheye.registry.db import Registry

        registry = Registry(candidate)
        try:
            dataset_id = registry.synchronize_recording_import(
                zarr_path=target,
                receipt=receipt,
                decided_by=decided_by,
            )
        finally:
            registry.close()
        return {
            "operation": "synchronize_recording_import",
            "dataset_id": dataset_id,
            "zarr_path": str(target),
            "decided_by": decided_by,
        }

    return _publish_through_configured_writer(
        canonical, backup_label="recording-import", mutate=mutate
    )


def shadow_synchronize_recording_imports(
    *,
    canonical_registry: str | Path,
    imports: Sequence[tuple[Path, object | None]],
    decided_by: str,
) -> RegistryShadowPublication:
    """Synchronize every import of one delivery in ONE shadow publication.

    Each ``(zarr_path, receipt)`` pair goes through the same per-artifact
    owner as :func:`shadow_synchronize_recording_import`
    (``Registry.synchronize_recording_import``), in order, against one local
    candidate. Any failure discards the whole candidate, so the canonical
    registry receives either every artifact of the delivery or none of them:
    one backup, one validation, one atomic replace. Each per-artifact step is
    an idempotent upsert keyed by the artifact, so a retry is safe.
    """

    canonical = Path(canonical_registry).expanduser().resolve()
    _require_decided_by(decided_by)
    targets = [
        (Path(zarr_path).expanduser().resolve(), receipt) for zarr_path, receipt in imports
    ]
    if not targets:
        raise RegistryShadowPublishError("a batch synchronization needs at least one import")
    if len({target for target, _receipt in targets}) != len(targets):
        raise RegistryShadowPublishError("a batch synchronization names one artifact twice")
    preflight_recording_import_receipts(targets)

    def mutate(candidate: Path) -> Mapping[str, Any]:
        from fisheye.registry.db import Registry

        registry = Registry(candidate)
        try:
            datasets = [
                {
                    "zarr_path": str(target),
                    "dataset_id": registry.synchronize_recording_import(
                        zarr_path=target,
                        receipt=receipt,
                        decided_by=decided_by,
                    ),
                }
                for target, receipt in targets
            ]
        finally:
            registry.close()
        return {
            "operation": "synchronize_recording_imports",
            "datasets": datasets,
            "decided_by": decided_by,
        }

    return _publish_through_configured_writer(
        canonical, backup_label="recording-imports", mutate=mutate
    )


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
        source_validation = validate_registry_sqlite_copy(canonical, temp_root=temp_parent)
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
                staged_validation = validate_registry_sqlite_copy(staged, temp_root=temp_parent)
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

        published_validation = validate_registry_sqlite_copy(canonical, temp_root=temp_parent)
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
    "REGISTRY_SHADOW_BACKUP_DIR_ENV",
    "REGISTRY_SHADOW_TEMP_ROOT_ENV",
    "REGISTRY_WRITER_HOST_ENV",
    "REGISTRY_WRITER_LOCK_PATH_ENV",
    "RegistryProducerCommitMismatch",
    "RegistryShadowPublication",
    "RegistryShadowPublishError",
    "RegistryValidation",
    "preflight_recording_import_receipts",
    "publish_registry_shadow",
    "shadow_synchronize_recording_import",
    "shadow_synchronize_recording_imports",
    "validate_registry_sqlite",
    "validate_registry_sqlite_copy",
]
