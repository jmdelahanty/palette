from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal, Optional, Sequence

from fisheye.shared.type_conversions import normalize_attr

ZarrDiscoveryPolicy = Literal["recording", "under_zarr_dir", "top_level"]


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


def _candidate_key(path: Path) -> str:
    try:
        return str(path.resolve())
    except OSError:
        return str(path)


def _is_zarr_root(path: Path) -> bool:
    return (path / "zarr.json").is_file() or (path / ".zgroup").is_file()


def _explicit_zarr_candidate(
    path: Path,
    *,
    include_zarr_files: bool,
    require_zarr_root: bool,
) -> bool:
    if path.suffix != ".zarr":
        return False
    if path.is_dir():
        return not require_zarr_root or _is_zarr_root(path)
    return include_zarr_files and path.is_file()


def _iter_policy_candidates(
    path: Path,
    *,
    recursive: bool,
    pattern_policy: ZarrDiscoveryPolicy,
) -> Iterable[Path]:
    if recursive:
        if pattern_policy in {"recording", "top_level"}:
            yield from sorted(path.rglob("*.zarr"))
            return
        if pattern_policy == "under_zarr_dir":
            yield from sorted(path.rglob("zarr/*.zarr"))
            return
    if pattern_policy == "recording":
        yield from sorted(path.glob("*.zarr"))
        yield from sorted(path.glob("*/zarr/*.zarr"))
        return
    if pattern_policy == "under_zarr_dir":
        yield from sorted(path.glob("*/zarr/*.zarr"))
        return
    if pattern_policy == "top_level":
        yield from sorted(path.glob("*.zarr"))
        return
    raise ValueError(f"Unsupported Zarr discovery policy: {pattern_policy}")


def iter_filesystem_zarrs(
    paths: Iterable[Path],
    recursive: bool,
    *,
    pattern_policy: ZarrDiscoveryPolicy = "recording",
    dedupe: bool = True,
    include_zarr_files: bool = True,
    require_zarr_root: bool = False,
) -> Iterable[Path]:
    """Yield filesystem Zarr candidates from roots.

    ``pattern_policy="recording"`` is Palette's canonical recording discovery:
    non-recursive search finds loose ``*.zarr`` archives plus
    ``*/zarr/*.zarr`` recording-layout archives, while recursive search finds
    every nested ``*.zarr``. Narrower legacy policies remain explicit so
    migrations can preserve intent where needed.
    """

    seen: set[str] = set()
    for path in paths:
        path = path.expanduser()
        candidates: list[Path] = []
        if _explicit_zarr_candidate(
            path,
            include_zarr_files=include_zarr_files,
            require_zarr_root=require_zarr_root,
        ):
            candidates = [path]
        elif path.is_dir():
            candidates = list(
                _iter_policy_candidates(
                    path,
                    recursive=recursive,
                    pattern_policy=pattern_policy,
                )
            )
        for candidate in candidates:
            if require_zarr_root and (not candidate.is_dir() or not _is_zarr_root(candidate)):
                continue
            if not include_zarr_files and candidate.is_file():
                continue
            if dedupe:
                key = _candidate_key(candidate)
                if key in seen:
                    continue
                seen.add(key)
            yield candidate


def discover_filesystem_zarrs(
    paths: Iterable[Path],
    *,
    recursive: bool,
    pattern_policy: ZarrDiscoveryPolicy = "recording",
    dedupe: bool = True,
    include_zarr_files: bool = True,
    require_zarr_root: bool = False,
) -> list[Path]:
    return list(
        iter_filesystem_zarrs(
            paths,
            recursive,
            pattern_policy=pattern_policy,
            dedupe=dedupe,
            include_zarr_files=include_zarr_files,
            require_zarr_root=require_zarr_root,
        )
    )


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
