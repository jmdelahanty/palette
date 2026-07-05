from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Collection, Mapping, TypeAlias
from urllib.parse import unquote, urlparse

import numpy as np
import zarr

from fisheye.shared.type_conversions import normalize_attr
from fisheye.shared.zarr_run_completion import resolve_authoritative_run_name


ParentPath: TypeAlias = str | Sequence[str]
DEFAULT_ZARR_USES: tuple[str, ...] = ("analysis", "training", "inference", "export")


def _path_tokens(parent_path: ParentPath) -> tuple[str, ...]:
    if isinstance(parent_path, str):
        tokens = tuple(token for token in parent_path.split("/") if token)
    else:
        tokens = tuple(str(token).strip("/") for token in parent_path if str(token).strip("/"))
    if not tokens:
        raise ValueError("parent_path must not be empty")
    return tokens


def zarr_group_names(parent: zarr.Group) -> list[str]:
    """Return sorted child group names from a Zarr-like parent."""

    if hasattr(parent, "group_keys"):
        names = parent.group_keys()
    else:  # pragma: no cover - defensive fallback for fake group variants
        names = parent.keys()
    return sorted(str(name) for name in names)


def _group_names(parent: zarr.Group) -> list[str]:
    return zarr_group_names(parent)


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


def _attrs_from_group_or_mapping(group_or_attrs: Any | None) -> Mapping[str, Any]:
    if group_or_attrs is None:
        return {}
    if isinstance(group_or_attrs, Mapping):
        return group_or_attrs
    attrs = getattr(group_or_attrs, "attrs", None)
    if attrs is None:
        return {}
    try:
        return {str(key): value for key, value in attrs.items()}
    except Exception:
        return {}


def infer_zarr_use(
    group_or_attrs: Any | None,
    zarr_path: str | Path | None = None,
    *,
    default: str | None = None,
    valid_uses: Collection[str] | None = DEFAULT_ZARR_USES,
) -> str | None:
    """Infer a Palette Zarr role from attrs, then filename suffix.

    ``zarr_purpose`` is the canonical store attr written by current Zarr
    producers. ``zarr_use`` is the registry/vocabulary alias and is checked as
    a fallback. Promoting ``zarr_use`` to canonical store metadata would be a
    separate explicit attr migration, not a reader precedence change.
    """

    allowed = (
        {str(value).strip().lower() for value in valid_uses}
        if valid_uses is not None
        else None
    )
    attrs = _attrs_from_group_or_mapping(group_or_attrs)
    for key in ("zarr_purpose", "zarr_use"):
        value = normalize_attr(attrs.get(key))
        if value is None:
            continue
        value = value.strip().lower()
        if allowed is None or value in allowed:
            return value
    if zarr_path is not None:
        name = Path(zarr_path).name.lower()
        if name.endswith("_analysis.zarr"):
            return "analysis"
        if name.endswith("_training.zarr"):
            return "training"
    return default


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


def zarr_root_fs_path(root: zarr.Group) -> Path | None:
    """Best-effort local filesystem path for a Zarr root group."""

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


def _root_fs_path(root: zarr.Group) -> Path | None:
    return zarr_root_fs_path(root)


def zarr_open_mode(root: zarr.Group) -> str:
    """Return the mode used to open a Zarr root, defaulting to read-only."""

    raw = getattr(root, "_palette_open_mode", None)
    if isinstance(raw, str) and raw:
        return raw
    return "r"


def _open_mode(root: zarr.Group) -> str:
    return zarr_open_mode(root)


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


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def reconsolidate_zarr_metadata(
    zarr_path: str | Path,
    *,
    group_path: str | None = None,
    policy: str = "manual",
    fail_on_error: bool = False,
) -> dict[str, Any]:
    """Refresh Zarr consolidated metadata for a root or subtree.

    Consolidation is a finalization/compatibility operation, not the source of
    correctness for mutable stores. Palette readers and registry validators must
    still tolerate stale or absent consolidated metadata.
    """

    root_path = Path(zarr_path).expanduser().resolve()
    normalized_group_path = normalize_zarr_path(group_path or "") if group_path else None
    target_path = root_path / normalized_group_path if normalized_group_path else root_path
    report: dict[str, Any] = {
        "status": "ok",
        "zarr_path": str(root_path),
        "group_path": normalized_group_path,
        "policy": str(policy),
        "consolidated_at_utc": _utc_now_iso(),
    }
    try:
        target_group = open_zarr_group_direct(target_path, mode="r+")
        target_group.attrs["metadata_consolidation_policy"] = str(policy)
        target_group.attrs["metadata_consolidation_status"] = "ok"
        target_group.attrs["metadata_consolidated_at_utc"] = report["consolidated_at_utc"]
        target_group.attrs["metadata_consolidation_group_path"] = normalized_group_path or ""
        zarr.consolidate_metadata(
            str(root_path),
            path=normalized_group_path,
        )
    except Exception as exc:
        report["status"] = "error"
        report["error"] = str(exc)
        try:
            target_group = open_zarr_group_direct(target_path, mode="r+")
            target_group.attrs["metadata_consolidation_policy"] = str(policy)
            target_group.attrs["metadata_consolidation_status"] = "error"
            target_group.attrs["metadata_consolidated_at_utc"] = report["consolidated_at_utc"]
            target_group.attrs["metadata_consolidation_group_path"] = normalized_group_path or ""
            target_group.attrs["metadata_consolidation_error"] = str(exc)
        except Exception:
            pass
        if fail_on_error:
            raise
    return report


def direct_zarr_group_names(path: Path | None) -> list[str]:
    """Return child group names by inspecting local Zarr metadata files directly."""

    if path is None or not path.is_dir():
        return []
    names: list[str] = []
    for candidate in sorted(path.iterdir()):
        if not candidate.is_dir():
            continue
        if (candidate / "zarr.json").is_file() or (candidate / ".zgroup").is_file():
            names.append(candidate.name)
    return names


def _open_group_direct(path: Path, *, mode: str) -> zarr.Group:
    return open_zarr_group_direct(path, mode=mode)


def _direct_group_names(path: Path | None) -> list[str]:
    return direct_zarr_group_names(path)


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
    root_fs_path = zarr_root_fs_path(root)
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
                parent = open_zarr_group_direct(candidate_path, mode=zarr_open_mode(root))
            if current_fs_path is not None:
                current_fs_path = current_fs_path / token
    except Exception as exc:
        raise ValueError(f"{display_path} not found in store") from exc

    requested = _normalize_run_name(run_name)
    alias_set = {alias.strip() for alias in latest_aliases if alias and alias.strip()}
    if requested in alias_set:
        requested = None

    available = sorted(set(zarr_group_names(parent)) | set(direct_zarr_group_names(current_fs_path)))

    if requested is not None:
        if requested not in parent:
            if current_fs_path is not None and requested in available:
                return open_zarr_group_direct(current_fs_path / requested, mode=zarr_open_mode(root)), requested
            available_text = ", ".join(available) or "(none)"
            raise ValueError(
                f"{run_label} '{requested}' not found under {display_path}. "
                f"Available: {available_text}"
            )
        return parent[requested], requested

    latest = None
    if fallback_to_latest:
        latest_attr = _normalize_run_name(parent.attrs.get("latest"))
        latest_complete_attr = _normalize_run_name(parent.attrs.get("latest_complete"))
        if latest_attr is not None or latest_complete_attr is not None or fallback_to_sorted is None:
            latest = _normalize_run_name(resolve_authoritative_run_name(parent))
        # Preserve the stale-consolidated-metadata fallback below: when the
        # parent cannot see the latest child but the filesystem can, the
        # completion resolver cannot inspect the run group.
        if latest is None:
            raw_latest = latest_attr
            if raw_latest is not None and raw_latest not in parent:
                latest = raw_latest
    if latest is not None:
        if latest in parent:
            return parent[latest], latest
        if current_fs_path is not None and latest in available:
            return open_zarr_group_direct(current_fs_path / latest, mode=zarr_open_mode(root)), latest
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
                return open_zarr_group_direct(current_fs_path / resolved, mode=zarr_open_mode(root)), resolved
            return parent[resolved], resolved

    if fallback_to_latest:
        raise ValueError(
            f"No {run_label.lower()} specified and {display_path} has no 'latest' attribute"
        )
    raise ValueError(f"No {run_label.lower()} specified for {display_path}")
