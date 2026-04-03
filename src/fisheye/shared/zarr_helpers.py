from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TypeAlias

import zarr

from fisheye.shared.type_conversions import normalize_attr


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


def _normalize_run_name(value: object) -> str | None:
    normalized = normalize_attr(value)
    if normalized is None:
        return None
    return normalized or None


def _root_fs_path(root: zarr.Group) -> Path | None:
    raw = getattr(root, "_palette_fs_path", None)
    if raw is None:
        return None
    try:
        return Path(str(raw))
    except Exception:
        return None


def _open_mode(root: zarr.Group) -> str:
    raw = getattr(root, "_palette_open_mode", None)
    if isinstance(raw, str) and raw:
        return raw
    return "r"


def _open_group_direct(path: Path, *, mode: str) -> zarr.Group:
    try:
        return zarr.open_group(str(path), mode=mode, use_consolidated=False)
    except TypeError:
        return zarr.open_group(str(path), mode=mode, consolidated=False)


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

    latest = _normalize_run_name(parent.attrs.get("latest")) if fallback_to_latest else None
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
