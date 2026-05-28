from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

from fisheye.shared.type_conversions import normalize_attr


@dataclass(frozen=True)
class RegistryZarrEntry:
    zarr_path: Path
    camera_id: Optional[str] = None


def load_path_list(path: Path, *, wrap_errors: bool = False) -> list[Path]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        raise
    except Exception as exc:
        if wrap_errors:
            raise RuntimeError(f"Failed to read {path}: {exc}") from exc
        raise

    items: list[Path] = []
    for line in lines:
        value = line.strip()
        if not value or value.startswith("#"):
            continue
        items.append(Path(value))
    return items


def iter_filesystem_zarrs(paths: Iterable[Path], recursive: bool) -> Iterable[Path]:
    for path in paths:
        path = path.expanduser()
        if path.is_dir() and path.suffix == ".zarr":
            yield path
            continue
        if not path.exists():
            continue
        if recursive:
            yield from path.rglob("*.zarr")
        else:
            yield from path.glob("*/zarr/*.zarr")
            yield from path.glob("*.zarr")


def discover_filesystem_zarrs(paths: Iterable[Path], *, recursive: bool) -> list[Path]:
    return list(iter_filesystem_zarrs(paths, recursive))


def _is_within_scope(path: Path, scope: Path) -> bool:
    try:
        path.relative_to(scope)
        return True
    except ValueError:
        return path == scope


def _apply_scope_filter(
    entries: Iterable[RegistryZarrEntry],
    scope_paths: Sequence[Path],
) -> list[RegistryZarrEntry]:
    if not scope_paths:
        return list(entries)
    resolved_scopes = [path.expanduser().resolve() for path in scope_paths]
    return [
        entry
        for entry in entries
        if any(_is_within_scope(entry.zarr_path, scope) for scope in resolved_scopes)
    ]


def _dedupe_entries(entries: Iterable[RegistryZarrEntry]) -> list[RegistryZarrEntry]:
    merged: dict[str, RegistryZarrEntry] = {}
    for entry in entries:
        key = str(entry.zarr_path)
        existing = merged.get(key)
        if existing is None:
            merged[key] = entry
            continue
        if existing.camera_id is None and entry.camera_id is not None:
            merged[key] = entry
    return sorted(merged.values(), key=lambda item: str(item.zarr_path))


def discover_registry_zarr_entries(
    *,
    registry_path: Path,
    scope_paths: Sequence[Path],
    zarr_use: str = "analysis",
    rig_id: Optional[str] = None,
    arena_id: Optional[str] = None,
    camera_id: Optional[str] = None,
    path_contains: Optional[str] = None,
    require_steps_ok: Optional[Sequence[str]] = None,
    exclude_step_ok: Optional[str] = None,
    zarr_suffix: Optional[str] = "_analysis.zarr",
    registry_cls: Optional[type[Any]] = None,
) -> list[RegistryZarrEntry]:
    if registry_cls is None:
        from fisheye.registry.db import Registry

        registry_cls = Registry

    registry = registry_cls(registry_path)
    try:
        query_kwargs: dict[str, Any] = dict(
            zarr_use=zarr_use,
            exclude_status="missing",
            require_recording=True,
            rig_id=rig_id,
            arena_id=arena_id,
            camera_id=camera_id,
            path_contains=path_contains,
        )
        if require_steps_ok:
            query_kwargs["require_steps_ok"] = list(require_steps_ok)
        if exclude_step_ok is not None:
            query_kwargs["exclude_step_ok"] = exclude_step_ok
        rows = registry.query_datasets(**query_kwargs)
    finally:
        registry.close()

    entries: list[RegistryZarrEntry] = []
    for row in rows:
        raw = row["zarr_path"]
        if raw is None:
            continue
        zarr_path = Path(str(raw))
        if zarr_suffix and not zarr_path.name.endswith(zarr_suffix):
            continue
        camera_raw: Any = None
        try:
            camera_raw = row["camera_id"]
        except Exception:
            camera_raw = None
        entries.append(
            RegistryZarrEntry(
                zarr_path=zarr_path.expanduser().resolve(),
                camera_id=normalize_attr(camera_raw),
            )
        )

    filtered = _apply_scope_filter(entries, scope_paths)
    return _dedupe_entries(filtered)


def discover_registry_zarrs(**kwargs: Any) -> list[Path]:
    return [entry.zarr_path for entry in discover_registry_zarr_entries(**kwargs)]


def discover_zarrs(
    *,
    source: str,
    registry_path: Path,
    scope_paths: Sequence[Path],
    **kwargs: Any,
) -> list[Path]:
    if source != "registry":
        raise ValueError(f"Unsupported discovery source: {source}")
    return discover_registry_zarrs(
        registry_path=registry_path,
        scope_paths=scope_paths,
        **kwargs,
    )
