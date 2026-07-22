"""Persistent, content-addressed directory fixtures for expensive tests.

The cache owns immutable templates only. Tests must copy a returned payload
before opening it for mutation. Entries are keyed by their declared schema,
exact source contents, and relevant runtime versions; every hit also checks
the payload tree digest before it is reused.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import time
from typing import Any
from uuid import uuid4


_CACHE_MANIFEST_SCHEMA = "palette.persistent_test_fixture.v1"
_DEFAULT_LOCK_TIMEOUT_SECONDS = 900.0


@dataclass(frozen=True)
class PersistentFixture:
    """One validated immutable fixture payload."""

    path: Path
    cache_key: str
    cache_hit: bool


def persistent_directory_fixture(
    *,
    namespace: str,
    schema_version: str,
    source_paths: Iterable[Path],
    dependency_versions: Mapping[str, str],
    build: Callable[[Path], None],
    validate: Callable[[Path], None],
    cache_root: Path | None = None,
    lock_timeout_seconds: float = _DEFAULT_LOCK_TIMEOUT_SECONDS,
) -> PersistentFixture:
    """Return a validated content-addressed directory built at most once.

    ``build`` receives a nonexistent destination and must create the complete
    directory there. ``validate`` must be read-only and is called both before
    publication and on cache hits. A tree SHA-256 protects cached bytes from
    accidental mutation independently of format-specific validation.
    """

    safe_namespace = _safe_component(namespace)
    if not safe_namespace:
        raise ValueError("Persistent fixture namespace must not be empty.")
    if not schema_version:
        raise ValueError("Persistent fixture schema_version must not be empty.")

    resolved_sources = tuple(sorted(_source_files(source_paths)))
    if not resolved_sources:
        raise ValueError("Persistent fixture cache requires at least one source file.")
    source_digest = _source_tree_sha256(resolved_sources)
    key_payload = {
        "manifest_schema": _CACHE_MANIFEST_SCHEMA,
        "namespace": safe_namespace,
        "schema_version": str(schema_version),
        "source_sha256": source_digest,
        "python": platform.python_version(),
        "dependencies": {
            str(name): str(version)
            for name, version in sorted(dependency_versions.items())
        },
    }
    cache_key = hashlib.sha256(_canonical_json(key_payload)).hexdigest()
    root = (
        Path(cache_root).expanduser().resolve()
        if cache_root is not None
        else _default_cache_root()
    )
    namespace_root = root / safe_namespace
    namespace_root.mkdir(parents=True, exist_ok=True)
    entry = namespace_root / cache_key

    if _valid_entry(entry, cache_key=cache_key, validate=validate):
        return PersistentFixture(entry / "payload", cache_key, True)

    lock = namespace_root / f".{cache_key}.lock"
    _acquire_lock(lock, timeout_seconds=lock_timeout_seconds)
    try:
        # Another process may have completed the entry while this process
        # waited for ownership.
        if _valid_entry(entry, cache_key=cache_key, validate=validate):
            return PersistentFixture(entry / "payload", cache_key, True)
        if entry.exists():
            shutil.rmtree(entry)

        staging = namespace_root / f".{cache_key}.{uuid4().hex}.tmp"
        payload = staging / "payload"
        try:
            staging.mkdir()
            build(payload)
            if not payload.is_dir():
                raise RuntimeError(
                    "Persistent fixture builder did not create its destination directory."
                )
            validate(payload)
            payload_sha256 = _directory_tree_sha256(payload)
            manifest = {
                **key_payload,
                "cache_key": cache_key,
                "payload_sha256": payload_sha256,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
            }
            (staging / "manifest.json").write_bytes(_canonical_json(manifest))
            os.replace(staging, entry)
        finally:
            if staging.exists():
                shutil.rmtree(staging)
    finally:
        shutil.rmtree(lock, ignore_errors=True)

    if not _valid_entry(entry, cache_key=cache_key, validate=validate):
        raise RuntimeError("New persistent fixture failed its post-publication validation.")
    return PersistentFixture(entry / "payload", cache_key, False)


def _default_cache_root() -> Path:
    configured = os.environ.get("PALETTE_TEST_FIXTURE_CACHE_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    return Path(__file__).resolve().parents[1] / ".pytest_cache" / "palette-fixtures"


def _safe_component(value: str) -> str:
    return "".join(
        character
        if character.isalnum() or character in {"-", "_", "."}
        else "_"
        for character in str(value).strip()
    )


def _source_files(paths: Iterable[Path]) -> Iterable[Path]:
    for raw_path in paths:
        path = Path(raw_path).resolve()
        if path.is_file():
            yield path
            continue
        if path.is_dir():
            yield from sorted(
                candidate
                for candidate in path.rglob("*.py")
                if candidate.is_file()
            )
            continue
        raise FileNotFoundError(f"Persistent fixture source does not exist: {path}")


def _source_tree_sha256(paths: Iterable[Path]) -> str:
    paths = tuple(paths)
    common_root = Path(
        os.path.commonpath(tuple(str(path.parent) for path in paths))
    )
    digest = hashlib.sha256()
    for path in paths:
        encoded = path.relative_to(common_root).as_posix().encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
        _update_file_digest(digest, path)
    return digest.hexdigest()


def _directory_tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "little"))
        digest.update(relative)
        _update_file_digest(digest, path)
    return digest.hexdigest()


def _update_file_digest(digest: Any, path: Path) -> None:
    size = path.stat().st_size
    digest.update(size.to_bytes(8, "little"))
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _valid_entry(
    entry: Path,
    *,
    cache_key: str,
    validate: Callable[[Path], None],
) -> bool:
    payload = entry / "payload"
    manifest_path = entry / "manifest.json"
    if not payload.is_dir() or not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("manifest_schema") != _CACHE_MANIFEST_SCHEMA
            or manifest.get("cache_key") != cache_key
            or manifest.get("payload_sha256") != _directory_tree_sha256(payload)
        ):
            return False
        validate(payload)
    except Exception:
        return False
    return True


def _acquire_lock(lock: Path, *, timeout_seconds: float) -> None:
    deadline = time.monotonic() + float(timeout_seconds)
    while True:
        try:
            lock.mkdir()
            (lock / "owner.json").write_bytes(
                _canonical_json(
                    {
                        "pid": os.getpid(),
                        "created_at_utc": datetime.now(timezone.utc).isoformat(),
                    }
                )
            )
            return
        except FileExistsError:
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Timed out waiting for fixture cache lock: {lock}")
            time.sleep(0.1)


__all__ = ["PersistentFixture", "persistent_directory_fixture"]
