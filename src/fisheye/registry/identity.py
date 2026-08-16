"""Immutable identity for one logical Palette registry instance."""

from __future__ import annotations

from dataclasses import dataclass
import sqlite3
from typing import Callable
from uuid import UUID, uuid4


REGISTRY_IDENTITY_SINGLETON_ID = 1
REGISTRY_IDENTITY_SCHEMA_MANAGED = "schema_managed"
REGISTRY_IDENTITY_LEGACY_BOOTSTRAP_UNVERIFIED = "legacy_bootstrap_unverified"
REGISTRY_IDENTITY_PROVENANCES = frozenset(
    {
        REGISTRY_IDENTITY_SCHEMA_MANAGED,
        REGISTRY_IDENTITY_LEGACY_BOOTSTRAP_UNVERIFIED,
    }
)


class RegistryIdentityError(RuntimeError):
    """Raised when a registry does not have one valid immutable identity."""


@dataclass(frozen=True)
class RegistryIdentity:
    registry_uuid: str
    identity_provenance: str
    minted_at_utc: str


def _canonical_uuid(value: object) -> str:
    raw = str(value or "").strip()
    try:
        normalized = str(UUID(raw))
    except (TypeError, ValueError, AttributeError) as exc:
        raise RegistryIdentityError(
            "registry_identity.registry_uuid is not a UUID"
        ) from exc
    if raw != normalized:
        raise RegistryIdentityError(
            "registry_identity.registry_uuid must use canonical lowercase UUID text"
        )
    return normalized


def read_registry_identity(conn: sqlite3.Connection) -> RegistryIdentity:
    """Read and validate the singleton identity without mutating the database."""

    try:
        rows = conn.execute(
            """
            SELECT singleton_id, registry_uuid, identity_provenance, minted_at_utc
            FROM registry_identity
            ORDER BY singleton_id
            """
        ).fetchall()
    except sqlite3.DatabaseError as exc:
        raise RegistryIdentityError("registry_identity table is missing or invalid") from exc
    if len(rows) != 1:
        raise RegistryIdentityError(
            f"registry_identity must contain exactly one row; found {len(rows)}"
        )
    row = rows[0]
    try:
        singleton_id = int(row[0])
    except (TypeError, ValueError) as exc:
        raise RegistryIdentityError(
            "registry_identity singleton_id must be the integer 1"
        ) from exc
    if singleton_id != REGISTRY_IDENTITY_SINGLETON_ID:
        raise RegistryIdentityError("registry_identity singleton_id must be 1")
    registry_uuid = _canonical_uuid(row[1])
    identity_provenance = str(row[2] or "").strip()
    if identity_provenance not in REGISTRY_IDENTITY_PROVENANCES:
        raise RegistryIdentityError(
            "registry_identity.identity_provenance is unsupported: "
            f"{identity_provenance!r}"
        )
    minted_at_utc = str(row[3] or "").strip()
    if not minted_at_utc:
        raise RegistryIdentityError("registry_identity.minted_at_utc is required")
    return RegistryIdentity(
        registry_uuid=registry_uuid,
        identity_provenance=identity_provenance,
        minted_at_utc=minted_at_utc,
    )


def ensure_registry_identity(
    conn: sqlite3.Connection,
    *,
    identity_provenance: str,
    minted_at_utc: str,
) -> RegistryIdentity:
    """Create the immutable singleton when absent, then return its validated value."""

    if identity_provenance not in REGISTRY_IDENTITY_PROVENANCES:
        raise ValueError(
            f"unsupported registry identity provenance: {identity_provenance!r}"
        )
    minted = str(minted_at_utc).strip()
    if not minted:
        raise ValueError("minted_at_utc is required")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS registry_identity (
            singleton_id INTEGER PRIMARY KEY,
            registry_uuid TEXT NOT NULL UNIQUE,
            identity_provenance TEXT NOT NULL,
            minted_at_utc TEXT NOT NULL,
            CHECK(singleton_id = 1),
            CHECK(identity_provenance IN ('schema_managed', 'legacy_bootstrap_unverified'))
        )
        """
    )
    conn.execute(
        """
        CREATE TRIGGER IF NOT EXISTS registry_identity_reject_second_insert
        BEFORE INSERT ON registry_identity
        WHEN EXISTS (SELECT 1 FROM registry_identity)
        BEGIN
            SELECT RAISE(ABORT, 'registry_identity is immutable');
        END
        """
    )
    conn.execute(
        """
        CREATE TRIGGER IF NOT EXISTS registry_identity_reject_update
        BEFORE UPDATE ON registry_identity
        BEGIN
            SELECT RAISE(ABORT, 'registry_identity is immutable');
        END
        """
    )
    conn.execute(
        """
        CREATE TRIGGER IF NOT EXISTS registry_identity_reject_delete
        BEFORE DELETE ON registry_identity
        BEGIN
            SELECT RAISE(ABORT, 'registry_identity is immutable');
        END
        """
    )
    row = conn.execute("SELECT COUNT(*) FROM registry_identity").fetchone()
    row_count = int(row[0]) if row is not None else 0
    if row_count == 0:
        conn.execute(
            """
            INSERT INTO registry_identity (
                singleton_id, registry_uuid, identity_provenance, minted_at_utc
            ) VALUES (?, ?, ?, ?)
            """,
            (
                REGISTRY_IDENTITY_SINGLETON_ID,
                str(uuid4()),
                identity_provenance,
                minted,
            ),
        )
    return read_registry_identity(conn)


def bootstrap_legacy_registry_identity(
    conn: sqlite3.Connection,
    *,
    latest_schema_version: int,
    minted_at_utc: str,
) -> RegistryIdentity:
    """Atomically mark an unversioned legacy registry without claiming verification."""

    conn.execute("BEGIN IMMEDIATE")
    try:
        identity = ensure_registry_identity(
            conn,
            identity_provenance=REGISTRY_IDENTITY_LEGACY_BOOTSTRAP_UNVERIFIED,
            minted_at_utc=minted_at_utc,
        )
        conn.execute(
            """
            INSERT OR REPLACE INTO schema_version (version, name, applied_utc)
            VALUES (?, ?, ?)
            """,
            (
                int(latest_schema_version),
                REGISTRY_IDENTITY_LEGACY_BOOTSTRAP_UNVERIFIED,
                str(minted_at_utc),
            ),
        )
        conn.execute(f"PRAGMA user_version = {int(latest_schema_version)}")
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    return identity


def apply_migrations_and_validate_identity(
    conn: sqlite3.Connection, apply_migrations: Callable[[], None]
) -> RegistryIdentity:
    """Apply normal migrations, then require the immutable identity to exist."""

    apply_migrations()
    return read_registry_identity(conn)


__all__ = [
    "REGISTRY_IDENTITY_LEGACY_BOOTSTRAP_UNVERIFIED",
    "REGISTRY_IDENTITY_PROVENANCES",
    "REGISTRY_IDENTITY_SCHEMA_MANAGED",
    "RegistryIdentity",
    "RegistryIdentityError",
    "apply_migrations_and_validate_identity",
    "bootstrap_legacy_registry_identity",
    "ensure_registry_identity",
    "read_registry_identity",
]
