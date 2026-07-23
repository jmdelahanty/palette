"""Fast, isolated SQLite registry fixtures.

Building Palette's complete registry schema is intentionally expensive because
it exercises every migration.  Most registry tests need the resulting empty
schema, not another execution of that migration history.  This module builds
one immutable empty template per pytest process and copies it for each test, so
tests retain filesystem and connection isolation without sharing mutable state.
"""

from __future__ import annotations

from pathlib import Path
import shutil
import tempfile
from threading import Lock

from fisheye.registry.db import Registry


_TEMPLATE_DIR = tempfile.TemporaryDirectory(prefix="palette_registry_template_")
_TEMPLATE_PATH = Path(_TEMPLATE_DIR.name) / "registry.sqlite"
_TEMPLATE_LOCK = Lock()


def registry_from_empty_template(path: Path) -> Registry:
    """Return an isolated current-schema registry cloned from one template."""

    destination = Path(path)
    if destination.exists():
        raise FileExistsError(
            f"Registry fixture destination already exists: {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with _TEMPLATE_LOCK:
        if not _TEMPLATE_PATH.exists():
            template = Registry(_TEMPLATE_PATH)
            template.close()
        shutil.copyfile(_TEMPLATE_PATH, destination)
    return Registry(destination)


__all__ = ["registry_from_empty_template"]
