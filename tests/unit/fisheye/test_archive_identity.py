from __future__ import annotations

from pathlib import Path

import pytest

from fisheye.shared.archive_identity import (
    ArchiveIdentityError,
    archive_identity,
    require_same_archive,
)


class _Node:
    def __init__(self, token: object) -> None:
        self._coordinate_archive_token = token


class _LocalStore:
    def __init__(self, root: Path) -> None:
        self.root = root


class _StorePath:
    def __init__(self, store: object) -> None:
        self.store = store


class _StoredNode:
    def __init__(self, store: object) -> None:
        self.store_path = _StorePath(store)


def test_shared_opaque_token_proves_same_synthetic_archive() -> None:
    token = object()
    first = _Node(token)
    second = _Node(token)

    assert require_same_archive(first, second) == archive_identity(first)


def test_identical_paths_or_copyable_tokens_cannot_alias_archives() -> None:
    with pytest.raises(ArchiveIdentityError, match="different archives"):
        require_same_archive(_Node(object()), _Node(object()))
    with pytest.raises(ArchiveIdentityError, match="opaque object"):
        archive_identity(_Node("/same/archive/path"))


def test_local_store_reopens_resolve_to_same_root_identity(tmp_path: Path) -> None:
    root = tmp_path / "archive.zarr"
    root.mkdir()
    first = _StoredNode(_LocalStore(root))
    second = _StoredNode(_LocalStore(root))

    assert require_same_archive(first, second).kind == "local_store_root"


def test_missing_local_store_root_fails_closed(tmp_path: Path) -> None:
    missing = _StoredNode(_LocalStore(tmp_path / "deleted-archive.zarr"))

    with pytest.raises(ArchiveIdentityError, match="must exist"):
        archive_identity(missing)


def test_nonlocal_store_requires_exact_retained_store_object() -> None:
    class MemoryLikeStore:
        pass

    shared = MemoryLikeStore()
    assert require_same_archive(_StoredNode(shared), _StoredNode(shared))
    with pytest.raises(ArchiveIdentityError, match="different archives"):
        require_same_archive(
            _StoredNode(MemoryLikeStore()),
            _StoredNode(MemoryLikeStore()),
        )


def test_unidentifiable_node_fails_closed() -> None:
    with pytest.raises(ArchiveIdentityError, match="does not expose"):
        archive_identity(object())
