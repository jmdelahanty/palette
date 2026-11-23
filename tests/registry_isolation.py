"""Keep pytest away from the live Palette registry.

On a workstation checkout, ``RegistryPaths.from_env`` falls back to
``configs/fisheye/registry.yaml``, which names the live registry, and several
modules default to that path outright. A test that reached one of them
without an explicit registry opened production: on 2026-11-22 a test run
applied an unmerged migration (079) to the live registry.

Two layers:

1. ``PALETTE_REGISTRY_PATH`` points at a per-process temporary registry for
   the whole session, so the ``from_env`` fallback never reaches the config.
2. ``sqlite3.connect`` refuses any database inside a protected registry
   directory (the configured registry's directory and the canonical
   ``/groups`` registries directory), read-only URIs included. The refusal is
   recorded, so a test fails by name even when the code under test swallows the
   exception.

Subprocesses inherit layer 1 only.
"""

from __future__ import annotations

import os
import sqlite3
import tempfile
from pathlib import Path
from typing import Any, List
from urllib.parse import unquote, urlsplit

import pytest
import yaml

CANONICAL_REGISTRY_DIR = Path("/groups/johnson/johnsonlab/jeremy/registries")
_REPO_ROOT = Path(__file__).resolve().parents[1]
_REAL_CONNECT = sqlite3.connect
_VIOLATIONS: List[str] = []


class LiveRegistryAccessError(RuntimeError):
    """A test tried to open a database in a protected registry directory."""


def _configured_registry_dir() -> Path | None:
    config = _REPO_ROOT / "configs" / "fisheye" / "registry.yaml"
    try:
        data = yaml.safe_load(config.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return None
    value = data.get("registry_path") if isinstance(data, dict) else None
    if not value and isinstance(data, dict) and isinstance(data.get("registry"), dict):
        value = data["registry"].get("path")
    if not value:
        return None
    path = Path(value)
    return (path if path.is_absolute() else config.parent / path).parent


def protected_registry_dirs() -> tuple[str, ...]:
    dirs = {os.path.realpath(CANONICAL_REGISTRY_DIR)}
    configured = _configured_registry_dir()
    if configured is not None:
        dirs.add(os.path.realpath(configured))
    return tuple(sorted(dirs))


_PROTECTED = protected_registry_dirs()


def _database_path(database: Any, uri: bool) -> str | None:
    if isinstance(database, bytes):
        database = os.fsdecode(database)
    text = os.fspath(database) if isinstance(database, (str, os.PathLike)) else None
    if not text or text == ":memory:":
        return None
    if uri or text.startswith("file:"):
        if not text.startswith("file:"):
            return None
        path = unquote(urlsplit(text).path)
        if not path or path == ":memory:" or "mode=memory" in text:
            return None
        text = path
    return os.path.realpath(text)


def is_protected(database: Any, *, uri: bool = False) -> bool:
    path = _database_path(database, uri)
    return path is not None and any(
        path == root or path.startswith(root + os.sep) for root in _PROTECTED
    )


def _guarded_connect(database: Any, *args: Any, **kwargs: Any) -> sqlite3.Connection:
    uri = bool(kwargs.get("uri", args[4] if len(args) > 4 else False))
    if is_protected(database, uri=uri):
        message = (
            f"test opened {database!r}, inside a protected registry directory "
            f"{_PROTECTED}; pass an explicit temporary registry"
        )
        _VIOLATIONS.append(message)
        raise LiveRegistryAccessError(message)
    return _REAL_CONNECT(database, *args, **kwargs)


_TEST_START = [0]


def take_violations() -> List[str]:
    """Return and clear the current test's refusals (for tests that trigger one on purpose).

    Refusals recorded outside any test (collection, imports) stay recorded and
    fail the session.
    """

    taken = _VIOLATIONS[_TEST_START[0]:]
    del _VIOLATIONS[_TEST_START[0]:]
    return taken


def pytest_configure(config: pytest.Config) -> None:
    current = os.environ.get("PALETTE_REGISTRY_PATH")
    if current and is_protected(current):
        raise pytest.UsageError(
            f"PALETTE_REGISTRY_PATH={current} is a protected registry; tests must not use it."
        )
    session_dir = Path(tempfile.mkdtemp(prefix="palette-pytest-registry-"))
    os.environ["PALETTE_REGISTRY_PATH"] = str(session_dir / "palette_registry.sqlite")
    sqlite3.connect = _guarded_connect  # type: ignore[assignment]


def pytest_unconfigure(config: pytest.Config) -> None:
    sqlite3.connect = _REAL_CONNECT  # type: ignore[assignment]


@pytest.fixture(autouse=True)
def _no_live_registry_access():
    _TEST_START[0] = len(_VIOLATIONS)
    yield
    violations = take_violations()
    if violations:
        pytest.fail("live registry access refused:\n" + "\n".join(violations), pytrace=False)


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    if _VIOLATIONS and session.exitstatus == 0:
        session.exitstatus = pytest.ExitCode.TESTS_FAILED


def pytest_terminal_summary(terminalreporter: Any) -> None:
    if _VIOLATIONS:
        terminalreporter.section("live registry access refused outside a test")
        for message in _VIOLATIONS:
            terminalreporter.line(message)
