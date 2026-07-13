"""Guards for refusing scratch dataset registrations in durable registries."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Iterable


ALLOW_TEMP_STORES_ENV = "PALETTE_REGISTRY_ALLOW_TEMP_STORES"
ALLOW_SCRATCH_STORES_ENV = "PALETTE_REGISTRY_ALLOW_SCRATCH_STORES"
ALLOW_UNOWNED_ANALYSIS_ENV = "PALETTE_REGISTRY_ALLOW_UNOWNED_ANALYSIS"

_IN_MEMORY_STORE_NAMES = {
    "in-memory.zarr",
    "in_memory.zarr",
    "inmemory.zarr",
}


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


def _scratch_store_reason(path: Path, *, temp_roots: Iterable[Path]) -> str | None:
    if _is_relative_to_any(path, temp_roots):
        return "temporary_root"
    if path.name.casefold() in _IN_MEMORY_STORE_NAMES:
        return "in_memory_store_name"
    lowered_parts = tuple(part.casefold() for part in path.parts)
    if ".claude" in lowered_parts and "worktrees" in lowered_parts:
        return "agent_worktree"
    return None


def assert_temp_store_registration_allowed(
    *,
    registry_path: Path,
    store_path: Path,
    recording_id: str | None = None,
    zarr_use: str | None = None,
) -> None:
    """Refuse scratch or unowned analysis stores in a durable registry.

    A registry beneath a temporary root remains an isolated test/development
    registry and may index arbitrary fixture paths. Durable registries require
    recording-level analysis datasets to carry a normalized recording ID.
    """

    temp_roots = _resolved_temp_roots()
    resolved_store_path = _resolve_path(store_path)
    resolved_registry_path = _resolve_path(registry_path)
    if _is_relative_to_any(resolved_registry_path, temp_roots):
        return

    scratch_reason = _scratch_store_reason(resolved_store_path, temp_roots=temp_roots)
    if scratch_reason is not None:
        override = (
            ALLOW_TEMP_STORES_ENV
            if scratch_reason == "temporary_root"
            else ALLOW_SCRATCH_STORES_ENV
        )
        if os.environ.get(override) != "1":
            raise ValueError(
                "Refusing to register scratch Zarr store "
                f"{resolved_store_path} in durable registry {resolved_registry_path} "
                f"(reason={scratch_reason}). Set {override}=1 to override."
            )

    normalized_use = str(zarr_use or "").strip().casefold()
    normalized_recording_id = str(recording_id or "").strip()
    if (
        normalized_use == "analysis"
        and not normalized_recording_id
        and os.environ.get(ALLOW_UNOWNED_ANALYSIS_ENV) != "1"
    ):
        raise ValueError(
            "Refusing to register analysis Zarr without a normalized recording_id: "
            f"{resolved_store_path} in durable registry {resolved_registry_path}. "
            f"Set {ALLOW_UNOWNED_ANALYSIS_ENV}=1 to override."
        )
