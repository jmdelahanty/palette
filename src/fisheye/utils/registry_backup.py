#!/usr/bin/env python3
"""Create and validate a Palette registry backup with Palette's SQLite runtime."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
from datetime import datetime
import json
import os
from pathlib import Path
import time
from typing import Sequence
import uuid

from fisheye.registry.shadow_publish import (
    RegistryShadowPublishError,
    RegistryValidation,
    _fsync_directory,
    _fsync_file,
    _publication_lock,
    _require_no_sqlite_sidecars,
    _sha256_file,
    _sqlite_backup,
    validate_registry_sqlite,
)
from fisheye.shared.json_safety import write_json_atomic

DEFAULT_REGISTRY = Path(
    "/groups/johnson/johnsonlab/jeremy/registries/palette_registry.sqlite"
)
DEFAULT_BACKUP_DIR = Path("/groups/ahrens/ahrenslab/jeremy/zebrobot/backups")
SCHEMA_ID = "palette.registry_backup_receipt"
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class RegistryBackupReceipt:
    source_registry: str
    backup_path: str
    source_sha256: str
    backup_sha256: str
    source_size_bytes: int
    backup_size_bytes: int
    source_validation: RegistryValidation
    backup_validation: RegistryValidation
    deleted_old_backups: tuple[str, ...]
    schema_id: str = SCHEMA_ID
    schema_version: int = SCHEMA_VERSION
    status: str = "complete"
    backup_mode: str = "python_sqlite_backup_api"

    def to_json(self) -> dict[str, object]:
        return asdict(self)


def _default_backup_path(backup_dir: Path) -> Path:
    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    return backup_dir / f"palette_registry_{timestamp}.sqlite"


def _prune_old_backups(
    *, backup_dir: Path, days_to_keep: int, keep: Path, now: float | None = None
) -> tuple[str, ...]:
    # Match `find -mtime +N`: integer-day age must be strictly greater than N.
    cutoff = float(time.time() if now is None else now) - (days_to_keep + 1) * 86400
    deleted: list[str] = []
    for candidate in sorted(backup_dir.glob("palette_registry_*.sqlite")):
        resolved = candidate.resolve()
        if resolved == keep.resolve() or not candidate.is_file():
            continue
        if candidate.stat().st_mtime < cutoff:
            candidate.unlink()
            deleted.append(str(resolved))
    if deleted:
        _fsync_directory(backup_dir)
    return tuple(deleted)


def create_registry_backup(
    *,
    registry: str | Path,
    backup_path: str | Path,
    days_to_keep: int | None = None,
) -> RegistryBackupReceipt:
    source = Path(registry).expanduser().resolve()
    backup = Path(backup_path).expanduser().resolve()
    if source == backup:
        raise RegistryShadowPublishError(
            "Registry backup path must differ from the source registry."
        )
    if backup.exists():
        raise RegistryShadowPublishError(
            f"Registry backup destination already exists: {backup}"
        )
    if days_to_keep is not None and days_to_keep < 0:
        raise ValueError("days_to_keep must be non-negative.")

    backup.parent.mkdir(parents=True, exist_ok=True)
    temporary = backup.parent / f".{backup.name}.tmp.{uuid.uuid4().hex}"
    try:
        with _publication_lock(source):
            _require_no_sqlite_sidecars(source)
            source_validation = validate_registry_sqlite(source)
            source_stat = source.stat()
            source_sha256 = _sha256_file(source)

            _sqlite_backup(source, temporary)
            os.chmod(temporary, source_stat.st_mode & 0o777)
            _fsync_file(temporary)
            backup_validation = validate_registry_sqlite(temporary)
            backup_sha256 = _sha256_file(temporary)
            backup_size = temporary.stat().st_size

            _require_no_sqlite_sidecars(source)
            if source.stat().st_size != source_stat.st_size:
                raise RegistryShadowPublishError(
                    "Registry size changed during backup; refusing publication."
                )
            if _sha256_file(source) != source_sha256:
                raise RegistryShadowPublishError(
                    "Registry content changed during backup; refusing publication."
                )
            try:
                os.link(temporary, backup)
            except FileExistsError as exc:
                raise RegistryShadowPublishError(
                    f"Registry backup destination already exists: {backup}"
                ) from exc
            _fsync_directory(backup.parent)
    finally:
        temporary.unlink(missing_ok=True)

    published_validation = validate_registry_sqlite(backup)
    expected_published_validation = replace(backup_validation, path=str(backup))
    if published_validation != expected_published_validation:
        raise RegistryShadowPublishError(
            "Published backup validation differs from the validated temporary backup."
        )
    if _sha256_file(backup) != backup_sha256:
        raise RegistryShadowPublishError(
            "Published backup hash differs from the validated temporary backup."
        )

    deleted = (
        _prune_old_backups(
            backup_dir=backup.parent,
            days_to_keep=days_to_keep,
            keep=backup,
        )
        if days_to_keep is not None
        else ()
    )
    return RegistryBackupReceipt(
        source_registry=str(source),
        backup_path=str(backup),
        source_sha256=source_sha256,
        backup_sha256=backup_sha256,
        source_size_bytes=int(source_stat.st_size),
        backup_size_bytes=int(backup_size),
        source_validation=source_validation,
        backup_validation=published_validation,
        deleted_old_backups=deleted,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--registry",
        type=Path,
        default=Path(os.environ.get("PALETTE_REGISTRY_PATH", DEFAULT_REGISTRY)),
    )
    parser.add_argument(
        "--backup-dir",
        type=Path,
        default=Path(os.environ.get("PALETTE_REGISTRY_BACKUP_DIR", DEFAULT_BACKUP_DIR)),
    )
    parser.add_argument(
        "--days-to-keep",
        type=int,
        default=int(os.environ.get("PALETTE_REGISTRY_BACKUP_DAYS_TO_KEEP", "7")),
    )
    parser.add_argument(
        "--result-json",
        type=Path,
        help="Optional path for an atomic machine-readable backup receipt.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.days_to_keep < 0:
        raise SystemExit("--days-to-keep must be a non-negative integer.")
    backup_dir = args.backup_dir.expanduser().resolve()
    receipt = create_registry_backup(
        registry=args.registry,
        backup_path=_default_backup_path(backup_dir),
        days_to_keep=args.days_to_keep,
    )
    payload = receipt.to_json()
    if args.result_json is not None:
        write_json_atomic(args.result_json.expanduser().resolve(), payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
