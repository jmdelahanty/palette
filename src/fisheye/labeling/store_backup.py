"""Validated, rotating snapshots of labeling SQLite sidecar stores.

Live labeling stores stay on local disk, where SQLite file locking is reliable
for several server processes.  This module copies a consistent snapshot to
durable storage without migrating or otherwise writing the live store:

1. read the source through a read-only connection with SQLite's online
   backup API, which yields a transactionally consistent copy while servers
   keep writing;
2. validate the local copy with Palette's Python SQLite runtime
   (``PRAGMA integrity_check`` and ``PRAGMA foreign_key_check``);
3. skip publication when the snapshot is byte-identical to the latest one;
4. publish a zstd-compressed copy plus a receipt, verify the published bytes
   decompress to the validated snapshot, and prune by retention.

Restore by stopping every server that uses the store, then
``zstd -d <backup>.sqlite.zst -o <store>.sqlite`` and restarting them.
"""

from __future__ import annotations

import argparse
from contextlib import closing
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import sys
import tempfile
from typing import Sequence
import uuid

import zstandard

from fisheye.shared.json_safety import write_json_atomic

DEFAULT_BACKUP_DIR = Path(
    "/groups/johnson/johnsonlab/jeremy/palette_backups/labeling_stores"
)
SCHEMA_ID = "palette.labeling_store_backup_receipt"
SCHEMA_VERSION = 1
SNAPSHOT_SUFFIX = ".sqlite.zst"
RECEIPT_SUFFIX = ".receipt.json"
LATEST_NAME = "latest.json"
_CHUNK_BYTES = 1 << 20


class LabelingStoreBackupError(RuntimeError):
    """A snapshot could not be taken, validated, or published."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(_CHUNK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


def _fsync_path(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def snapshot_sqlite(source: Path, destination: Path) -> None:
    """Copy a consistent snapshot of ``source`` without writing to it."""

    uri = f"{Path(source).resolve().as_uri()}?mode=ro"
    with closing(sqlite3.connect(uri, uri=True, timeout=30.0)) as src:
        src.execute("PRAGMA busy_timeout = 30000;")
        with closing(sqlite3.connect(str(destination))) as dst:
            src.backup(dst)


def validate_labeling_sqlite(path: Path) -> dict[str, object]:
    """Fully validate a labeling store copy; raise if it is not sound."""

    with closing(sqlite3.connect(f"{Path(path).resolve().as_uri()}?mode=ro", uri=True)) as conn:
        integrity = [str(row[0]) for row in conn.execute("PRAGMA integrity_check;")]
        if integrity != ["ok"]:
            raise LabelingStoreBackupError(
                f"integrity_check failed for {path}: {integrity[:5]}"
            )
        foreign_key_violations = conn.execute("PRAGMA foreign_key_check;").fetchall()
        if foreign_key_violations:
            raise LabelingStoreBackupError(
                f"foreign_key_check found {len(foreign_key_violations)} violation(s) in {path}."
            )
        tables = [
            str(row[0])
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name;"
            )
        ]
        if "labeling_schema_meta" not in tables:
            raise LabelingStoreBackupError(f"{path} is not a labeling store.")
        schema_row = conn.execute(
            "SELECT value FROM labeling_schema_meta WHERE key = 'schema_version';"
        ).fetchone()
        row_counts = {
            table: int(conn.execute(f'SELECT COUNT(*) FROM "{table}";').fetchone()[0])
            for table in tables
            if not table.startswith("sqlite_")
        }
        state_violations = _state_violations(conn, tables)
    return {
        "integrity_check": "ok",
        "foreign_key_violations": 0,
        "schema_version": None if schema_row is None else str(schema_row[0]),
        "row_counts": row_counts,
        # Reported, not raised: an unknown state must never stop a backup.
        "state_violations": state_violations,
    }


def _state_violations(conn, tables) -> dict[str, dict[str, int]]:
    """Counts of state values outside each table's known domain."""

    from .assignment_store import TASK_STATES
    from .checkpoint_store import APPLY_RECEIPT_STATES, CHECKPOINT_STATES

    domains = {
        "labeling_tasks": TASK_STATES,
        "labeling_session_checkpoints": CHECKPOINT_STATES,
        "labeling_checkpoint_apply_receipts": APPLY_RECEIPT_STATES,
    }
    found: dict[str, dict[str, int]] = {}
    for table, allowed in domains.items():
        if table not in tables:
            continue
        rows = conn.execute(f'SELECT state, COUNT(*) FROM "{table}" GROUP BY state;').fetchall()
        bad = {str(state): int(count) for state, count in rows if state not in allowed}
        if bad:
            found[table] = bad
    return found


def write_validated_copy(
    source: str | Path, destination: str | Path, *, overwrite: bool = False
) -> dict[str, object]:
    """Write one uncompressed, validated snapshot of ``source`` to ``destination``."""

    source_path = Path(source).expanduser().resolve()
    if not source_path.is_file():
        raise LabelingStoreBackupError(f"Labeling store does not exist: {source_path}")
    output = Path(destination).expanduser().resolve()
    if output == source_path:
        raise LabelingStoreBackupError("Backup path must differ from the source store.")
    if output.exists() and not overwrite:
        raise FileExistsError(f"Backup destination already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.parent / f".{output.name}.tmp.{uuid.uuid4().hex}"
    try:
        snapshot_sqlite(source_path, temporary)
        validation = validate_labeling_sqlite(temporary)
        _fsync_path(temporary)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    _fsync_path(output.parent)
    return {
        "source_path": str(source_path),
        "backup_path": str(output),
        "overwrite": bool(overwrite),
        "snapshot_sha256": _sha256_file(output),
        "validation": validation,
        "runtime": _runtime(),
    }


def _snapshot_paths(target_dir: Path, label: str) -> list[Path]:
    return sorted(target_dir.glob(f"{label}_*{SNAPSHOT_SUFFIX}"), reverse=True)


def _snapshot_day(path: Path, label: str) -> str:
    # Names are ``<label>_YYYYmmdd_HHMMSS.sqlite.zst``.
    return path.name[len(label) + 1 :][:8]


def prune_snapshots(
    target_dir: Path, label: str, *, keep_recent: int, keep_daily: int
) -> list[str]:
    """Keep the newest ``keep_recent`` snapshots plus the newest per day for
    ``keep_daily`` distinct days; delete the rest with their receipts."""

    snapshots = _snapshot_paths(target_dir, label)
    keep = set(snapshots[: max(int(keep_recent), 1)])
    days: set[str] = set()
    for path in snapshots:
        day = _snapshot_day(path, label)
        if day not in days and len(days) < int(keep_daily):
            days.add(day)
            keep.add(path)
    deleted: list[str] = []
    for path in snapshots:
        if path in keep:
            continue
        path.unlink(missing_ok=True)
        Path(f"{path}{RECEIPT_SUFFIX}").unlink(missing_ok=True)
        deleted.append(str(path))
    if deleted:
        _fsync_path(target_dir)
    return deleted


def _compress(source: Path, destination: Path) -> None:
    compressor = zstandard.ZstdCompressor(level=3, threads=-1)
    with source.open("rb") as reader, destination.open("xb") as writer:
        compressor.copy_stream(reader, writer, read_size=_CHUNK_BYTES)
        writer.flush()
        os.fsync(writer.fileno())


def _decompressed_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle, zstandard.ZstdDecompressor().stream_reader(handle) as reader:
        for block in iter(lambda: reader.read(_CHUNK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


def _runtime() -> dict[str, str]:
    return {
        "python_executable": sys.executable,
        "sqlite_library_version": sqlite3.sqlite_version,
        "zstandard_version": zstandard.__version__,
    }


def backup_labeling_store(
    source: str | Path,
    backup_dir: str | Path,
    *,
    label: str | None = None,
    keep_recent: int = 48,
    keep_daily: int = 30,
) -> dict[str, object]:
    """Snapshot, validate, and publish one labeling store; return its receipt."""

    source_path = Path(source).expanduser().resolve()
    if not source_path.is_file():
        raise LabelingStoreBackupError(f"Labeling store does not exist: {source_path}")
    store_label = str(label or source_path.stem)
    target_dir = Path(backup_dir).expanduser().resolve() / store_label
    if source_path.is_relative_to(target_dir):
        raise LabelingStoreBackupError("Backups must not be written beside the source store.")
    started = datetime.now(timezone.utc)
    source_stat = source_path.stat()

    with tempfile.TemporaryDirectory(prefix="palette-labeling-backup-") as scratch:
        snapshot = Path(scratch) / "snapshot.sqlite"
        snapshot_sqlite(source_path, snapshot)
        validation = validate_labeling_sqlite(snapshot)
        snapshot_sha256 = _sha256_file(snapshot)
        snapshot_size = snapshot.stat().st_size

        receipt: dict[str, object] = {
            "schema_id": SCHEMA_ID,
            "schema_version": SCHEMA_VERSION,
            "label": store_label,
            "source_path": str(source_path),
            "source_size_bytes": int(source_stat.st_size),
            "source_mtime_utc": datetime.fromtimestamp(
                source_stat.st_mtime, timezone.utc
            ).isoformat(),
            "snapshot_sha256": snapshot_sha256,
            "snapshot_size_bytes": int(snapshot_size),
            "validation": validation,
            "backup_mode": "python_sqlite_online_backup_api_read_only",
            "runtime": _runtime(),
            "started_at_utc": started.isoformat(),
        }

        latest_path = target_dir / LATEST_NAME
        if latest_path.is_file():
            latest = json.loads(latest_path.read_text())
            published = Path(str(latest.get("backup_path") or ""))
            if latest.get("snapshot_sha256") == snapshot_sha256 and published.is_file():
                return {
                    **receipt,
                    "status": "unchanged",
                    "backup_path": str(published),
                    "deleted_old_backups": [],
                }

        target_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
        backup_path = target_dir / f"{store_label}_{stamp}{SNAPSHOT_SUFFIX}"
        if backup_path.exists():
            raise LabelingStoreBackupError(f"Backup already exists: {backup_path}")
        temporary = target_dir / f".{backup_path.name}.tmp.{uuid.uuid4().hex}"
        try:
            _compress(snapshot, temporary)
            os.link(temporary, backup_path)
        finally:
            temporary.unlink(missing_ok=True)
        _fsync_path(target_dir)

    if _decompressed_sha256(backup_path) != snapshot_sha256:
        backup_path.unlink(missing_ok=True)
        raise LabelingStoreBackupError(
            f"Published backup does not decompress to the validated snapshot: {backup_path}"
        )
    receipt.update(
        {
            "status": "complete",
            "backup_path": str(backup_path),
            "backup_sha256": _sha256_file(backup_path),
            "backup_size_bytes": int(backup_path.stat().st_size),
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        }
    )
    write_json_atomic(Path(f"{backup_path}{RECEIPT_SUFFIX}"), receipt)
    write_json_atomic(target_dir / LATEST_NAME, receipt)
    receipt["deleted_old_backups"] = prune_snapshots(
        target_dir, store_label, keep_recent=keep_recent, keep_daily=keep_daily
    )
    return receipt


def _store_argument(value: str) -> tuple[Path, str | None]:
    path, _, label = value.partition("=")
    return Path(path), (label or None)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--store",
        action="append",
        required=True,
        type=_store_argument,
        metavar="PATH[=LABEL]",
        help="Labeling store to back up; repeat for several stores.",
    )
    parser.add_argument(
        "--backup-dir",
        type=Path,
        default=Path(os.environ.get("PALETTE_LABELING_BACKUP_DIR", DEFAULT_BACKUP_DIR)),
    )
    parser.add_argument("--keep-recent", type=int, default=48)
    parser.add_argument("--keep-daily", type=int, default=30)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.keep_recent < 1 or args.keep_daily < 0:
        raise SystemExit("--keep-recent must be >= 1 and --keep-daily >= 0.")
    results: list[dict[str, object]] = []
    failed = False
    for path, label in args.store:
        try:
            results.append(
                backup_labeling_store(
                    path,
                    args.backup_dir,
                    label=label,
                    keep_recent=args.keep_recent,
                    keep_daily=args.keep_daily,
                )
            )
        except Exception as exc:  # noqa: BLE001 - report every store, then fail
            failed = True
            results.append(
                {"status": "failed", "source_path": str(path), "error": f"{type(exc).__name__}: {exc}"}
            )
    print(json.dumps(results, indent=2, sort_keys=True))
    return 1 if failed else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
