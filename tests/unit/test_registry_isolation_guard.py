"""The pytest session cannot open the live registry.

Refusal tests point the guard at a writable protected directory under
``tmp_path``, so they hold on machines without ``/groups`` (CI) and show that
the guard, not the filesystem, refuses.
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import pytest

from fisheye.registry.db import Registry, RegistryPaths
from tests import registry_isolation as guard

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def live(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A stand-in live registry path inside a protected, writable directory."""

    protected = tmp_path / "registries"
    (protected / "backups").mkdir(parents=True)
    monkeypatch.setattr(guard, "_PROTECTED", (os.path.realpath(protected),))
    return protected / "palette_registry.sqlite"


def test_canonical_registry_directory_is_protected() -> None:
    assert os.path.realpath(guard.CANONICAL_REGISTRY_DIR) in guard.protected_registry_dirs()


def test_default_registry_path_is_a_session_temporary() -> None:
    resolved = RegistryPaths.from_env(REPO_ROOT).path

    assert not guard.is_protected(resolved)
    assert Path(os.environ["PALETTE_REGISTRY_PATH"]) == resolved


@pytest.mark.parametrize(
    ("form", "uri"),
    [
        ("str", False),
        ("path", False),
        ("bytes", False),
        ("ro_uri", True),
        ("backup", False),
        ("str", True),
    ],
)
def test_protected_paths_are_refused(live: Path, form: str, uri: bool) -> None:
    database = {
        "str": str(live),
        "path": live,
        "bytes": os.fsencode(str(live)),
        "ro_uri": f"file:{live}?mode=ro",
        "backup": str(live.parent / "backups" / "copy.sqlite"),
    }[form]

    with pytest.raises(guard.LiveRegistryAccessError):
        sqlite3.connect(database, uri=uri)

    assert len(guard.take_violations()) == 1
    assert not live.exists()


def test_positional_arguments_cannot_unprotect_a_path(live: Path) -> None:
    for args in (
        (5.0, 0, "DEFERRED", True, sqlite3.Connection),
        (5.0, 0, "DEFERRED", True, sqlite3.Connection, 128, True),
    ):
        with pytest.raises(guard.LiveRegistryAccessError):
            sqlite3.connect(str(live), *args)

    assert len(guard.take_violations()) == 2


def test_registry_constructor_is_refused_on_the_live_path(live: Path) -> None:
    with pytest.raises(guard.LiveRegistryAccessError):
        Registry(live)

    assert guard.take_violations()
    assert not live.exists()


def test_swallowed_refusal_is_still_recorded(live: Path) -> None:
    try:
        sqlite3.connect(str(live))
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
