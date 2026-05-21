from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Mapping, TypeAlias
from urllib.parse import unquote, urlparse

import numpy as np
import zarr

from fisheye.shared.type_conversions import normalize_attr
from fisheye.shared.zarr_run_completion import resolve_latest_complete_run_name


ParentPath: TypeAlias = str | Sequence[str]


def _path_tokens(parent_path: ParentPath) -> tuple[str, ...]:
    if isinstance(parent_path, str):
        tokens = tuple(token for token in parent_path.split("/") if token)
    else:
        tokens = tuple(str(token).strip("/") for token in parent_path if str(token).strip("/"))
    if not tokens:
        raise ValueError("parent_path must not be empty")
    return tokens


def _group_names(parent: zarr.Group) -> list[str]:
    if hasattr(parent, "group_keys"):
        names = parent.group_keys()
    else:  # pragma: no cover - defensive fallback for fake group variants
        names = parent.keys()
    return sorted(str(name) for name in names)


def normalize_zarr_path(path: str) -> str:
    """Return a slash-normalized relative Zarr path."""

    return "/".join(part for part in str(path).strip("/").split("/") if part)


def zarr_attrs_dict(group: Any | None) -> dict[str, Any]:
    """Best-effort plain dict copy of Zarr attrs with string keys."""

    if group is None:
        return {}
    attrs = getattr(group, "attrs", {})
    try:
        return {str(key): value for key, value in attrs.items()}
    except Exception:
        return {}


def safe_int(value: Any) -> int | None:
    """Return ``int(value)`` when possible, otherwise ``None``."""

    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def zarr_group_keys(group: Any | None) -> list[str]:
    """Return sorted child group names from a Zarr-like group."""

    if group is None:
        return []
    keys_fn = getattr(group, "group_keys", None)
    if callable(keys_fn):
        try:
            return sorted(str(key) for key in keys_fn())
        except Exception:
            return []
    try:
        return sorted(str(key) for key, value in group.items() if hasattr(value, "keys"))
    except Exception:
        return []


def zarr_child_group(group: Any | None, path: str) -> Any | None:
    """Return a nested child group, or ``None`` when unavailable."""

    if group is None:
        return None
    current = group
    for part in normalize_zarr_path(path).split("/"):
        if not part:
            continue
        try:
            if part not in current:
                return None
            current = current[part]
        except Exception:
            return None
    return current if hasattr(current, "keys") or hasattr(current, "group_keys") else None


def zarr_array_names(group: Any | None) -> list[str]:
    """Return sorted child array names from a Zarr-like group."""

    if group is None:
        return []
    try:
        keys = list(group.keys())
    except Exception:
        return []
    names: list[str] = []
    for name in keys:
        try:
            value = group[name]
        except Exception:
            continue
        if hasattr(value, "shape"):
            names.append(str(name))
    return sorted(names)


def read_zarr_array_mapping(
    group: Any | None,
    *,
    physical_prefix: str,
    logical_prefix: str | None = None,
    source_paths: dict[str, str] | None = None,
    array_names: Sequence[str] | None = None,
) -> dict[str, np.ndarray]:
    """Materialize child arrays and optionally record logical-to-physical paths."""

    if group is None:
        return {}
    names = list(array_names) if array_names is not None else zarr_array_names(group)
    logical = physical_prefix if logical_prefix is None else logical_prefix
    arrays: dict[str, np.ndarray] = {}
    for name in names:
        try:
            if name not in group:
                continue
            value = group[name]
        except Exception:
            continue
        if not hasattr(value, "shape"):
            continue
        try:
            arrays[str(name)] = np.asarray(value[:])
        except Exception:
            continue
        if source_paths is not None:
            source_paths[f"{logical}/{name}"] = f"{physical_prefix}/{name}"
    return arrays


def first_array_length(
    arrays: Mapping[str, np.ndarray],
    names: Sequence[str],
) -> int:
    """Return the first non-scalar array length among named arrays."""

    for name in names:
        values = arrays.get(str(name))
        if values is not None and values.shape:
            return int(values.shape[0])
    return 0


def first_array_length_in_group(group: Any | None, names: Sequence[str]) -> int:
    """Return the first non-scalar child-array length among named arrays."""

    if group is None:
        return 0
    for name in names:
        try:
            if name not in group:
                continue
            node = group[name]
        except Exception:
            continue
        shape = getattr(node, "shape", None)
        if shape:
            return int(shape[0])
    return 0


def _normalize_run_name(value: object) -> str | None:
    normalized = normalize_attr(value)
    if normalized is None:
        return None
    return normalized or None


def _root_fs_path(root: zarr.Group) -> Path | None:
    raw = getattr(root, "_palette_fs_path", None)
    if raw is None:
        raw = getattr(root, "store_path", None)
    if raw is None:
        raw = getattr(root, "store", None)
    if raw is None:
        return None
    try:
        value = str(raw)
        parsed = urlparse(value)
        if parsed.scheme == "file":
            return Path(unquote(parsed.path))
        if "://" in value:
            return None
        return Path(value)
    except Exception:
        return None


def _open_mode(root: zarr.Group) -> str:
    raw = getattr(root, "_palette_open_mode", None)
    if isinstance(raw, str) and raw:
        return raw
    return "r"


def open_zarr_group_direct(path: str | Path, *, mode: str) -> zarr.Group:
    """Open a local Zarr group without using consolidated metadata.

    Palette writers often create new run groups and update ``latest`` attrs
    during review/finalization workflows. Direct metadata reads avoid stale
    consolidated-metadata views in local mutable stores.
    """

    resolved = Path(path)
    try:
        return zarr.open_group(str(resolved), mode=mode, use_consolidated=False)
    except TypeError:
        return zarr.open_group(str(resolved), mode=mode, consolidated=False)


def _open_group_direct(path: Path, *, mode: str) -> zarr.Group:
    return open_zarr_group_direct(path, mode=mode)


def _direct_group_names(path: Path | None) -> list[str]:
    if path is None or not path.is_dir():
        return []
    names: list[str] = []
    for candidate in sorted(path.iterdir()):
        if not candidate.is_dir():
            continue
        if (candidate / "zarr.json").is_file() or (candidate / ".zgroup").is_file():
            names.append(candidate.name)
    return names


def resolve_zarr_run(
    root: zarr.Group,
    parent_path: ParentPath,
    run_name: str | None,
    *,
    fallback_to_latest: bool = True,
    fallback_to_sorted: str | None = None,
    latest_aliases: Sequence[str] = (),
    run_label: str = "Run",
) -> tuple[zarr.Group, str]:
    """
    Resolve an existing run group under a parent path.

    ``fallback_to_sorted`` accepts ``"first"`` or ``"last"`` for the cases
    where callers historically selected a deterministic run even when the
    parent's ``latest`` attribute was missing or stale.
    """
    tokens = _path_tokens(parent_path)
    display_path = "/".join(tokens)
    parent: zarr.Group = root
    root_fs_path = _root_fs_path(root)
    current_fs_path = root_fs_path
    try:
        for token in tokens:
            try:
                parent = parent[token]
            except Exception as exc:
                if current_fs_path is None:
                    raise exc
                candidate_path = current_fs_path / token
                if not candidate_path.is_dir():
                    raise exc
                parent = _open_group_direct(candidate_path, mode=_open_mode(root))
            if current_fs_path is not None:
                current_fs_path = current_fs_path / token
    except Exception as exc:
        raise ValueError(f"{display_path} not found in store") from exc

    requested = _normalize_run_name(run_name)
    alias_set = {alias.strip() for alias in latest_aliases if alias and alias.strip()}
    if requested in alias_set:
        requested = None

    available = sorted(set(_group_names(parent)) | set(_direct_group_names(current_fs_path)))

    if requested is not None:
        if requested not in parent:
            if current_fs_path is not None and requested in available:
                return _open_group_direct(current_fs_path / requested, mode=_open_mode(root)), requested
            available_text = ", ".join(available) or "(none)"
            raise ValueError(
                f"{run_label} '{requested}' not found under {display_path}. "
                f"Available: {available_text}"
            )
        return parent[requested], requested

    latest = None
    if fallback_to_latest:
        latest = _normalize_run_name(resolve_latest_complete_run_name(parent, legacy_default=True))
        # Preserve the stale-consolidated-metadata fallback below: when the
        # parent cannot see the latest child but the filesystem can, the
        # completion resolver cannot inspect the run group.
        if latest is None:
            raw_latest = _normalize_run_name(parent.attrs.get("latest"))
            if raw_latest is not None and raw_latest not in parent:
                latest = raw_latest
    if latest is not None:
        if latest in parent:
            return parent[latest], latest
        if current_fs_path is not None and latest in available:
            return _open_group_direct(current_fs_path / latest, mode=_open_mode(root)), latest
        if fallback_to_sorted is None:
            available_text = ", ".join(available) or "(none)"
            raise ValueError(
                f"{run_label} latest '{latest}' not found under {display_path}. "
                f"Available: {available_text}"
            )

    if fallback_to_sorted is not None:
        if fallback_to_sorted not in {"first", "last"}:
            raise ValueError("fallback_to_sorted must be 'first', 'last', or None")
        if available:
            resolved = available[0] if fallback_to_sorted == "first" else available[-1]
            if resolved in parent:
                return parent[resolved], resolved
            if current_fs_path is not None:
                return _open_group_direct(current_fs_path / resolved, mode=_open_mode(root)), resolved
            return parent[resolved], resolved

    if fallback_to_latest:
        raise ValueError(
            f"No {run_label.lower()} specified and {display_path} has no 'latest' attribute"
        )
    raise ValueError(f"No {run_label.lower()} specified for {display_path}")
