"""The pytest session cannot open the live registry."""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import pytest

from fisheye.registry.db import Registry, RegistryPaths
from tests import registry_isolation as guard

LIVE = guard.CANONICAL_REGISTRY_DIR / "palette_registry.sqlite"
REPO_ROOT = Path(__file__).resolve().parents[2]


def test_default_registry_path_is_a_session_temporary(tmp_path: Path) -> None:
    resolved = RegistryPaths.from_env(REPO_ROOT).path

    assert not guard.is_protected(resolved)
    assert Path(os.environ["PALETTE_REGISTRY_PATH"]) == resolved


@pytest.mark.parametrize(
    ("database", "uri"),
    [
        (str(LIVE), False),
        (LIVE, False),
        (os.fsencode(str(LIVE)), False),
        (f"file:{LIVE}?mode=ro", True),
        (str(guard.CANONICAL_REGISTRY_DIR / "backups" / "copy.sqlite"), False),
        (str(LIVE), True),
    ],
)
def test_protected_paths_are_refused(database, uri: bool) -> None:
    with pytest.raises(guard.LiveRegistryAccessError):
        sqlite3.connect(database, uri=uri)

    assert len(guard.take_violations()) == 1


def test_positional_arguments_cannot_unprotect_a_path() -> None:
    for args in (
        (5.0, 0, "DEFERRED", True, sqlite3.Connection),
        (5.0, 0, "DEFERRED", True, sqlite3.Connection, 128, True),
    ):
        with pytest.raises(guard.LiveRegistryAccessError):
            sqlite3.connect(str(LIVE), *args)

    assert len(guard.take_violations()) == 2


def test_registry_constructor_is_refused_on_the_live_path() -> None:
    with pytest.raises(guard.LiveRegistryAccessError):
        Registry(LIVE)

    assert guard.take_violations()


def test_swallowed_refusal_is_still_recorded() -> None:
    try:
        sqlite3.connect(str(LIVE))
    except Exception:
        pass

    assert len(guard.take_violations()) == 1


def test_temporary_and_memory_databases_open(tmp_path: Path) -> None:
    for database, uri in (
        (":memory:", False),
        (str(tmp_path / "a.sqlite"), False),
        (f"file:{tmp_path / 'b.sqlite'}?mode=rwc", True),
        ("file::memory:", True),
    ):
        sqlite3.connect(database, uri=uri).close()

    registry = Registry(tmp_path / "registry.sqlite")
    registry.close()
    assert guard.take_violations() == []
