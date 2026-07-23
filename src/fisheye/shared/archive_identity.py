"""Exact archive/store identity for persisted contract bindings.

Archive-relative paths and content digests are necessary but insufficient:
two different Zarr archives can contain identical nodes at identical paths.
Canonical coordinate publication therefore binds every participating node to
one archive identity and rejects cross-archive composition.

Real local stores are identified by their existing resolved root and required
device/inode pair.  Other Zarr stores are identified by the exact
store object retained by the node.  Lightweight test doubles must expose one
shared opaque ``_coordinate_archive_token`` object; primitive/copyable tokens
are intentionally rejected so a path string cannot masquerade as authority.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Any


class ArchiveIdentityError(ValueError):
    """Raised when persisted nodes cannot be proven to share one archive."""


@dataclass(frozen=True)
class ArchiveIdentity:
    """Comparable identity plus a retained authority object."""

    kind: str
    key: tuple[Any, ...]
    _authority: Any = field(repr=False, compare=False)


def _explicit_identity(node: Any) -> ArchiveIdentity | None:
    token = getattr(node, "_coordinate_archive_token", None)
    if token is None:
        return None
    if isinstance(token, (str, bytes, bytearray, int, float, bool, tuple, frozenset)):
        raise ArchiveIdentityError(
            "Synthetic archive identity must be one shared opaque object, not a "
            "copyable primitive."
        )
    return ArchiveIdentity(
        kind="explicit_opaque_token",
        key=(id(token),),
        _authority=token,
    )


def _local_root_identity(store: Any) -> ArchiveIdentity | None:
    root = getattr(store, "root", None)
    if not isinstance(root, (str, os.PathLike)):
        return None
    root_text = os.fspath(root)
    if not isinstance(root_text, str):
        raise ArchiveIdentityError("Local Zarr store roots must use text paths.")
    if "://" in root_text:
        return None
    try:
        resolved = Path(root_text).expanduser().resolve(strict=True)
        stat = resolved.stat()
    except OSError as exc:
        raise ArchiveIdentityError(
            "Local Zarr store root must exist and remain stat-accessible."
        ) from exc
    key = (str(resolved), int(stat.st_dev), int(stat.st_ino))
    return ArchiveIdentity(
        kind="local_store_root",
        key=key,
        _authority=store,
    )


def archive_identity(node: Any) -> ArchiveIdentity:
    """Resolve one node to a fail-closed archive/store identity."""

    explicit = _explicit_identity(node)
    if explicit is not None:
        return explicit

    store_path = getattr(node, "store_path", None)
    store = getattr(store_path, "store", None)
    if store is None:
        candidate = getattr(node, "store", None)
        store = getattr(candidate, "store", candidate)
    if store is None:
        raise ArchiveIdentityError(
            "Persisted node does not expose a Zarr store or an opaque test archive token."
        )

    local = _local_root_identity(store)
    if local is not None:
        return local

    # Memory and non-local stores are safe within the live binding because all
    # evidence retains the exact store object.  Reopening a remote store through
    # a different object intentionally fails closed until that store type has a
    # typed stable identity resolver.
    return ArchiveIdentity(
        kind=f"store_object:{type(store).__module__}.{type(store).__qualname__}",
        key=(id(store),),
        _authority=store,
    )


def require_same_archive(*nodes: Any) -> ArchiveIdentity:
    """Return the common identity or reject an empty/mixed node set."""

    if not nodes:
        raise ArchiveIdentityError("At least one persisted node is required.")
    identities = tuple(archive_identity(node) for node in nodes)
    first = identities[0]
    if any(item.kind != first.kind or item.key != first.key for item in identities[1:]):
        raise ArchiveIdentityError(
            "Persisted coordinate evidence comes from different archives/stores."
        )
    return first


__all__ = [
    "ArchiveIdentity",
    "ArchiveIdentityError",
    "archive_identity",
    "require_same_archive",
]
