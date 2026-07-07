"""Guards for refusing temp-store registrations in durable registries."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Iterable


ALLOW_TEMP_STORES_ENV = "PALETTE_REGISTRY_ALLOW_TEMP_STORES"


def _resolve_path(path: Path) -> Path:
    return Path(path).expanduser().resolve()


def _resolved_temp_roots() -> tuple[Path, ...]:
    raw_roots = (Path(tempfile.gettempdir()), Path("/tmp"), Path("/var/tmp"), Path("/dev/shm"))
    resolved_roots: list[Path] = []
    seen: set[Path] = set()
    for root in raw_roots:
        resolved = _resolve_path(root)
        if resolved not in seen:
            resolved_roots.append(resolved)
            seen.add(resolved)
    return tuple(resolved_roots)


def _is_relative_to_any(path: Path, roots: Iterable[Path]) -> bool:
    return any(path.is_relative_to(root) for root in roots)


def assert_temp_store_registration_allowed(*, registry_path: Path, store_path: Path) -> None:
    """Refuse temp Zarr stores in a registry that is not itself temporary."""

    temp_roots = _resolved_temp_roots()
    resolved_store_path = _resolve_path(store_path)
    if not _is_relative_to_any(resolved_store_path, temp_roots):
        return

    resolved_registry_path = _resolve_path(registry_path)
    if _is_relative_to_any(resolved_registry_path, temp_roots):
        return
    if os.environ.get(ALLOW_TEMP_STORES_ENV) == "1":
        return

    raise ValueError(
        "Refusing to register temporary Zarr store "
        f"{resolved_store_path} in non-temporary registry {resolved_registry_path}. "
        f"Set {ALLOW_TEMP_STORES_ENV}=1 to override."
    )
