"""Completion markers for Palette Zarr run groups.

The contract is intentionally attr-based so it works for both Zarr v2/v3 and
existing fake-group unit tests. Parent groups stamped with
``palette_completion_epoch >= 1`` are strict: unmarked children are not trusted
as complete. Unstamped legacy parents still treat unmarked children as complete
for read compatibility until the backfill tool verifies and stamps them.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Iterator, Mapping, Optional
import warnings

from fisheye.shared.run_provenance import CLI_RUN_PROVENANCE_ATTR
from fisheye.shared.run_provenance import RUN_PROVENANCE_ATTR
from fisheye.shared.run_provenance import validate_run_provenance

RUN_COMPLETION_CONTRACT = "palette.zarr_run_completion.v1"
RUN_COMPLETION_CONTRACT_ATTR = "palette_run_completion_contract"
RUN_COMPLETION_STATUS_ATTR = "palette_run_completion_status"
RUN_COMPLETED_AT_ATTR = "palette_run_completed_at_utc"
RUN_STARTED_AT_ATTR = "palette_run_started_at_utc"
RUN_NAME_ATTR = "palette_run_name"
RUN_STAGE_ATTR = "palette_run_stage"
RUN_LATEST_COMPLETE_ATTR = "latest_complete"
RUN_LATEST_PENDING_ATTR = "latest_pending"
AUTHORITATIVE_RUN_ATTR = "authoritative_run"
AUTHORITATIVE_RUN_PROVENANCE_ATTR = "authoritative_run_provenance"
COMPLETION_EPOCH_ATTR = "palette_completion_epoch"
COMPLETION_EPOCH_STRICT = 1
COMPLETION_EPOCH_REQUIRE_PROVENANCE = 2
RUN_PROVENANCE_BYPASS_ATTR = "run_provenance_enforcement_bypass"

RUN_STATUS_RUNNING = "running"
RUN_STATUS_COMPLETE = "complete"
RUN_STATUS_FAILED = "failed"

_LEGACY_COMPLETION_WARNING_KEYS: set[str] = set()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _coerce_int(value: Any) -> Optional[int]:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def effective_legacy_default(parent_group: Any) -> bool:
    """Return whether unmarked child runs remain legacy-complete for a parent."""

    epoch = _coerce_int(getattr(parent_group, "attrs", {}).get(COMPLETION_EPOCH_ATTR))
    return not (epoch is not None and epoch >= COMPLETION_EPOCH_STRICT)


def requires_completion_provenance(parent_group: Any) -> bool:
    """Return whether a runs-parent requires run provenance to finalize."""

    epoch = _coerce_int(getattr(parent_group, "attrs", {}).get(COMPLETION_EPOCH_ATTR))
    return bool(epoch is not None and epoch >= COMPLETION_EPOCH_REQUIRE_PROVENANCE)


def _has_children(parent_group: Any) -> bool:
    """Conservatively report whether a runs-parent already has child groups."""

    try:
        return bool(_group_names(parent_group))
    except Exception:
        return True


def require_runs_parent(
    root: Any,
    name: str,
    *,
    completion_epoch: Optional[int] = None,
) -> Any:
    """Return a runs-parent group, stamping new empty parents as provenance-strict.

    Existing parents with children are not stamped here; they remain grandfathered
    until an explicit backfill/migration sets their epoch.
    """

    if hasattr(root, "require_group"):
        parent = root.require_group(name)
    elif name in root:
        parent = root[name]
    elif hasattr(root, "create_group"):
        parent = root.create_group(name)
    else:
        raise AttributeError(f"Root object does not support require_group/create_group for {name!r}.")
    attrs = parent.attrs
    if attrs.get(COMPLETION_EPOCH_ATTR) is None and not _has_children(parent):
        attrs[COMPLETION_EPOCH_ATTR] = (
            int(completion_epoch)
            if completion_epoch is not None
            else COMPLETION_EPOCH_REQUIRE_PROVENANCE
        )
    return parent


def _candidate_run_provenance(run_group: Any, explicit: Optional[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    if isinstance(explicit, Mapping):
        return explicit
    attrs = getattr(run_group, "attrs", {})
    for attr_name in (RUN_PROVENANCE_ATTR, CLI_RUN_PROVENANCE_ATTR):
        value = attrs.get(attr_name)
        if isinstance(value, Mapping):
            return value
    return None


def _record_run_provenance_bypass(
    run_group: Any,
    *,
    reason: str,
    errors: tuple[str, ...],
) -> None:
    run_group.attrs[RUN_PROVENANCE_BYPASS_ATTR] = {
        "bypassed_at_utc": utc_now_iso(),
        "reason": str(reason),
        "errors": list(errors),
    }


def _validate_or_record_run_provenance(
    run_group: Any,
    *,
    parent_group: Optional[Any],
    run_name: Optional[str],
    run_provenance: Optional[Mapping[str, Any]],
    allow_missing_run_provenance: bool,
    missing_run_provenance_reason: Optional[str],
) -> None:
    if parent_group is None or not requires_completion_provenance(parent_group):
        if isinstance(run_provenance, Mapping):
            run_group.attrs[RUN_PROVENANCE_ATTR] = validate_run_provenance(run_provenance).normalized or dict(run_provenance)
        return

    candidate = _candidate_run_provenance(run_group, run_provenance)
    result = validate_run_provenance(candidate)
    if result.valid:
        run_group.attrs[RUN_PROVENANCE_ATTR] = result.normalized or dict(candidate or {})
        return

    if allow_missing_run_provenance:
        reason = str(missing_run_provenance_reason or "").strip()
        if not reason:
            raise RuntimeError(
                "Refusing run-provenance enforcement bypass without "
                "missing_run_provenance_reason"
            )
        _record_run_provenance_bypass(run_group, reason=reason, errors=result.errors)
        return

    label = f" run {run_name!r}" if run_name is not None else ""
    raise RuntimeError(
        f"Refusing to mark{label} complete because parent epoch requires "
        f"valid run provenance: {'; '.join(result.errors)}"
    )


def mark_run_started(
    run_group: Any,
    *,
    run_name: Optional[str] = None,
    stage: Optional[str] = None,
    started_at_utc: Optional[str] = None,
) -> None:
    """Mark a run group as intentionally incomplete.

    Writers should call this immediately after creating a run group. Readers
    that understand this contract will ignore the run until
    :func:`mark_run_complete` is called.
    """

    attrs = run_group.attrs
    attrs[RUN_COMPLETION_CONTRACT_ATTR] = RUN_COMPLETION_CONTRACT
    attrs[RUN_COMPLETION_STATUS_ATTR] = RUN_STATUS_RUNNING
    attrs[RUN_STARTED_AT_ATTR] = started_at_utc or utc_now_iso()
    if run_name is not None:
        attrs[RUN_NAME_ATTR] = str(run_name)
    if stage is not None:
        attrs[RUN_STAGE_ATTR] = str(stage)


def mark_run_complete(
    run_group: Any,
    *,
    parent_group: Optional[Any] = None,
    run_name: Optional[str] = None,
    completed_at_utc: Optional[str] = None,
    run_provenance: Optional[Mapping[str, Any]] = None,
    allow_missing_run_provenance: bool = False,
    missing_run_provenance_reason: Optional[str] = None,
) -> None:
    """Mark a run as complete and then publish it as latest when requested."""

    _validate_or_record_run_provenance(
        run_group,
        parent_group=parent_group,
        run_name=run_name,
        run_provenance=run_provenance,
        allow_missing_run_provenance=allow_missing_run_provenance,
        missing_run_provenance_reason=missing_run_provenance_reason,
    )
    attrs = run_group.attrs
    attrs[RUN_COMPLETION_CONTRACT_ATTR] = RUN_COMPLETION_CONTRACT
    attrs[RUN_COMPLETION_STATUS_ATTR] = RUN_STATUS_COMPLETE
    attrs[RUN_COMPLETED_AT_ATTR] = completed_at_utc or utc_now_iso()
    if run_name is not None:
        attrs[RUN_NAME_ATTR] = str(run_name)
    if parent_group is not None and run_name is not None:
        name = str(run_name)
        parent_group.attrs[RUN_LATEST_COMPLETE_ATTR] = name
        parent_group.attrs["latest"] = name
        if parent_group.attrs.get(RUN_LATEST_PENDING_ATTR) == name:
            try:
                del parent_group.attrs[RUN_LATEST_PENDING_ATTR]
            except Exception:
                parent_group.attrs[RUN_LATEST_PENDING_ATTR] = None


def mark_run_failed(
    run_group: Any,
    *,
    failed_at_utc: Optional[str] = None,
    error: Optional[str] = None,
) -> None:
    attrs = run_group.attrs
    attrs[RUN_COMPLETION_CONTRACT_ATTR] = RUN_COMPLETION_CONTRACT
    attrs[RUN_COMPLETION_STATUS_ATTR] = RUN_STATUS_FAILED
    attrs["palette_run_failed_at_utc"] = failed_at_utc or utc_now_iso()
    if error:
        attrs["palette_run_error"] = str(error)


def has_run_completion_contract(run_group: Any) -> bool:
    return run_group.attrs.get(RUN_COMPLETION_CONTRACT_ATTR) == RUN_COMPLETION_CONTRACT


def _is_unmarked_legacy_run(run_group: Any) -> bool:
    attrs = run_group.attrs
    return attrs.get(RUN_COMPLETION_STATUS_ATTR) is None and not has_run_completion_contract(run_group)


def _parent_warning_key(parent_group: Any) -> str:
    for attr in ("store_path", "path", "name"):
        value = getattr(parent_group, attr, None)
        if value not in (None, ""):
            return f"{attr}:{value}"
    return f"id:{id(parent_group)}"


def _warn_legacy_completion_acceptance(parent_group: Any) -> None:
    key = _parent_warning_key(parent_group)
    if key in _LEGACY_COMPLETION_WARNING_KEYS:
        return
    _LEGACY_COMPLETION_WARNING_KEYS.add(key)
    warnings.warn(
        "Zarr run parent has no strict completion epoch; treating unmarked "
        "child runs as legacy-complete for compatibility. Backfill completion "
        "markers and stamp palette_completion_epoch to make this fail-closed.",
        RuntimeWarning,
        stacklevel=3,
    )


def is_run_complete(run_group: Any, *, legacy_default: bool = True) -> bool:
    """Return whether a run group is complete.

    The default legacy compatibility mode preserves existing archives that predate this
    contract. Once all active writers emit completion attrs, callers can switch
    to strict mode.
    """

    attrs = run_group.attrs
    status = attrs.get(RUN_COMPLETION_STATUS_ATTR)
    if status is None and not has_run_completion_contract(run_group):
        return bool(legacy_default)
    return str(status).lower() == RUN_STATUS_COMPLETE


def is_run_complete_in_parent(
    parent_group: Any,
    run_group: Any,
    *,
    legacy_default: bool | None = None,
) -> bool:
    """Return run completion using parent-scoped strictness metadata."""

    effective_default = (
        effective_legacy_default(parent_group)
        if legacy_default is None
        else bool(legacy_default)
    )
    if effective_default and _is_unmarked_legacy_run(run_group):
        _warn_legacy_completion_acceptance(parent_group)
    return is_run_complete(run_group, legacy_default=effective_default)


def _group_names(parent_group: Any) -> list[str]:
    if hasattr(parent_group, "group_keys"):
        names = parent_group.group_keys()
    else:
        names = parent_group.keys()
    return sorted(str(name) for name in names if isinstance(name, str))


def _get_child(parent_group: Any, name: str) -> Optional[Any]:
    parts = [part for part in str(name).split("/") if part]
    if len(parts) > 1:
        child = parent_group
        for part in parts:
            child = _get_child(child, part)
            if child is None:
                return None
        return child

    lookup_name = parts[0] if parts else str(name)
    try:
        if lookup_name in parent_group:
            return parent_group[lookup_name]
    except Exception:
        pass
    try:
        return parent_group.get(lookup_name)
    except Exception:
        return None


def _normalize_name(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def resolve_latest_complete_run_name(
    parent_group: Any,
    *,
    latest_attr: str = "latest",
    legacy_default: bool | None = None,
) -> Optional[str]:
    """Resolve the newest run that is complete under the run-completion contract."""

    for attr in (latest_attr, RUN_LATEST_COMPLETE_ATTR):
        candidate = parent_group.attrs.get(attr)
        if not candidate:
            continue
        name = str(candidate)
        child = _get_child(parent_group, name)
        if child is not None and is_run_complete_in_parent(
            parent_group,
            child,
            legacy_default=legacy_default,
        ):
            return name

    for name in reversed(_group_names(parent_group)):
        child = _get_child(parent_group, name)
        if child is not None and is_run_complete_in_parent(
            parent_group,
            child,
            legacy_default=legacy_default,
        ):
            return name
    return None


def resolve_latest_complete_run_group(
    parent_group: Any,
    *,
    latest_attr: str = "latest",
    legacy_default: bool | None = None,
) -> tuple[Optional[str], Optional[Any]]:
    name = resolve_latest_complete_run_name(
        parent_group,
        latest_attr=latest_attr,
        legacy_default=legacy_default,
    )
    if name is None:
        return None, None
    return name, _get_child(parent_group, name)


def _is_child_complete(
    parent_group: Any,
    run_name: str,
    *,
    legacy_default: bool | None = None,
) -> tuple[Optional[Any], bool]:
    child = _get_child(parent_group, run_name)
    if child is None:
        return None, False
    return child, is_run_complete_in_parent(
        parent_group,
        child,
        legacy_default=legacy_default,
    )


def resolve_authoritative_run_name(
    parent_group: Any,
    *,
    latest_attr: str = "latest",
    legacy_default: bool | None = None,
) -> Optional[str]:
    """Resolve the authoritative run, falling back to latest only when unset."""

    authoritative = _normalize_name(parent_group.attrs.get(AUTHORITATIVE_RUN_ATTR))
    if authoritative is not None:
        _child, complete = _is_child_complete(
            parent_group,
            authoritative,
            legacy_default=legacy_default,
        )
        return authoritative if complete else None
    return resolve_latest_complete_run_name(
        parent_group,
        latest_attr=latest_attr,
        legacy_default=legacy_default,
    )


def resolve_authoritative_run_group(
    parent_group: Any,
    *,
    latest_attr: str = "latest",
    legacy_default: bool | None = None,
) -> tuple[Optional[str], Optional[Any]]:
    name = resolve_authoritative_run_name(
        parent_group,
        latest_attr=latest_attr,
        legacy_default=legacy_default,
    )
    if name is None:
        return None, None
    return name, _get_child(parent_group, name)


def _restore_attr(attrs: Any, key: str, *, existed: bool, value: Any) -> None:
    if existed:
        if attrs.get(key) == value:
            return
        attrs[key] = value
        return
    if key not in attrs:
        return
    try:
        del attrs[key]
    except Exception:
        attrs[key] = None


def set_authoritative_run(
    parent_group: Any,
    run_name: str,
    *,
    approved_by: str | None = None,
    approved_at: str | None = None,
    git_sha: str | None = None,
    note: str | None = None,
    legacy_default: bool | None = None,
) -> dict[str, Any]:
    """Set a parent-scoped authoritative run pointer after validating completion."""

    name = _normalize_name(run_name)
    if name is None:
        raise ValueError("authoritative run name must be non-empty")
    child, complete = _is_child_complete(
        parent_group,
        name,
        legacy_default=legacy_default,
    )
    if child is None:
        raise ValueError(f"authoritative run {name!r} does not exist")
    if not complete:
        raise ValueError(f"authoritative run {name!r} is not complete")

    provenance = {
        "approved_by": _normalize_name(approved_by),
        "approved_at": approved_at or utc_now_iso(),
        "git_sha": _normalize_name(git_sha),
        "note": "" if note is None else str(note),
    }
    attrs = parent_group.attrs
    old_run_exists = AUTHORITATIVE_RUN_ATTR in attrs
    old_run = attrs.get(AUTHORITATIVE_RUN_ATTR)
    old_provenance_exists = AUTHORITATIVE_RUN_PROVENANCE_ATTR in attrs
    old_provenance = attrs.get(AUTHORITATIVE_RUN_PROVENANCE_ATTR)
    try:
        attrs[AUTHORITATIVE_RUN_PROVENANCE_ATTR] = provenance
        attrs[AUTHORITATIVE_RUN_ATTR] = name
    except Exception:
        _restore_attr(
            attrs,
            AUTHORITATIVE_RUN_ATTR,
            existed=old_run_exists,
            value=old_run,
        )
        _restore_attr(
            attrs,
            AUTHORITATIVE_RUN_PROVENANCE_ATTR,
            existed=old_provenance_exists,
            value=old_provenance,
        )
        raise
    return provenance


def clear_authoritative_run(parent_group: Any) -> None:
    """Clear the parent-scoped authoritative pointer and provenance attrs."""

    attrs = parent_group.attrs
    for key in (AUTHORITATIVE_RUN_ATTR, AUTHORITATIVE_RUN_PROVENANCE_ATTR):
        if key not in attrs:
            continue
        try:
            del attrs[key]
        except Exception:
            attrs[key] = None


def note_pending_latest(parent_group: Any, run_name: str) -> None:
    name = str(run_name)
    parent_group.attrs[RUN_LATEST_PENDING_ATTR] = name
    if parent_group.attrs.get("latest") != name:
        return
    previous_complete = parent_group.attrs.get(RUN_LATEST_COMPLETE_ATTR)
    if previous_complete and _get_child(parent_group, str(previous_complete)) is not None:
        parent_group.attrs["latest"] = str(previous_complete)
        return
    try:
        del parent_group.attrs["latest"]
    except Exception:
        parent_group.attrs["latest"] = None


def mark_run_pending(parent_group: Any, run_name: str) -> None:
    """Publish a newly-created run as pending without changing ``latest``."""

    parent_group.attrs[RUN_LATEST_PENDING_ATTR] = str(run_name)


def describe_run_parent(
    parent_group: Any,
    *,
    parent_path: str = "",
    legacy_default: bool | None = None,
) -> dict[str, Any]:
    """Return a JSON-serializable completion summary for a run parent group."""

    latest = _normalize_name(parent_group.attrs.get("latest"))
    latest_complete = _normalize_name(parent_group.attrs.get(RUN_LATEST_COMPLETE_ATTR))
    latest_pending = _normalize_name(parent_group.attrs.get(RUN_LATEST_PENDING_ATTR))
    authoritative_run = _normalize_name(parent_group.attrs.get(AUTHORITATIVE_RUN_ATTR))
    authoritative_provenance = parent_group.attrs.get(AUTHORITATIVE_RUN_PROVENANCE_ATTR)
    resolved_latest_complete = resolve_latest_complete_run_name(
        parent_group,
        legacy_default=legacy_default,
    )
    resolved_authoritative = resolve_authoritative_run_name(
        parent_group,
        legacy_default=legacy_default,
    )

    runs: list[dict[str, Any]] = []
    incomplete_runs: list[str] = []
    failed_runs: list[str] = []
    for name in _group_names(parent_group):
        child = _get_child(parent_group, name)
        if child is None:
            continue
        status = _normalize_name(child.attrs.get(RUN_COMPLETION_STATUS_ATTR))
        has_contract = has_run_completion_contract(child)
        complete = is_run_complete_in_parent(
            parent_group,
            child,
            legacy_default=legacy_default,
        )
        if not complete:
            incomplete_runs.append(name)
        if status == RUN_STATUS_FAILED:
            failed_runs.append(name)
        runs.append(
            {
                "name": name,
                "has_completion_contract": has_contract,
                "completion_status": status or ("legacy_complete" if complete else "legacy_incomplete"),
                "complete": complete,
            }
        )

    unsafe_reasons: list[str] = []
    latest_status: Optional[str] = None
    latest_complete_status: Optional[str] = None
    authoritative_status: Optional[str] = None
    authoritative_exists = False
    authoritative_complete = False
    latest_exists = latest is not None and _get_child(parent_group, latest) is not None
    if authoritative_run:
        authoritative_child, authoritative_complete = _is_child_complete(
            parent_group,
            authoritative_run,
            legacy_default=legacy_default,
        )
        authoritative_exists = authoritative_child is not None
        if authoritative_child is None:
            unsafe_reasons.append("authoritative_run_missing")
        else:
            authoritative_status = _normalize_name(
                authoritative_child.attrs.get(RUN_COMPLETION_STATUS_ATTR)
            )
            if not authoritative_complete:
                reason_status = authoritative_status or (
                    "legacy_missing_contract"
                    if not has_run_completion_contract(authoritative_child)
                    else "unknown"
                )
                unsafe_reasons.append(f"authoritative_run_incomplete:{reason_status}")
    if latest:
        latest_child = _get_child(parent_group, latest)
        if latest_child is None:
            unsafe_reasons.append("latest_missing")
        else:
            latest_status = _normalize_name(latest_child.attrs.get(RUN_COMPLETION_STATUS_ATTR))
            if not is_run_complete_in_parent(
                parent_group,
                latest_child,
                legacy_default=legacy_default,
            ):
                reason_status = latest_status or (
                    "legacy_missing_contract"
                    if not has_run_completion_contract(latest_child)
                    else "unknown"
                )
                unsafe_reasons.append(f"latest_incomplete:{reason_status}")

    if latest_complete:
        latest_complete_child = _get_child(parent_group, latest_complete)
        if latest_complete_child is None:
            unsafe_reasons.append("latest_complete_missing")
        else:
            latest_complete_status = _normalize_name(
                latest_complete_child.attrs.get(RUN_COMPLETION_STATUS_ATTR)
            )
            if not is_run_complete_in_parent(
                parent_group,
                latest_complete_child,
                legacy_default=legacy_default,
            ):
                reason_status = latest_complete_status or (
                    "legacy_missing_contract"
                    if not has_run_completion_contract(latest_complete_child)
                    else "unknown"
                )
                unsafe_reasons.append(
                    f"latest_complete_incomplete:{reason_status}"
                )

    return {
        "parent_path": parent_path,
        "completion_epoch": _coerce_int(parent_group.attrs.get(COMPLETION_EPOCH_ATTR)),
        "legacy_default": effective_legacy_default(parent_group) if legacy_default is None else bool(legacy_default),
        "latest": latest,
        "latest_exists": latest_exists,
        "latest_status": latest_status,
        "latest_complete": latest_complete,
        "latest_complete_status": latest_complete_status,
        "latest_pending": latest_pending,
        "authoritative_run": authoritative_run,
        "authoritative_run_exists": authoritative_exists,
        "authoritative_run_status": authoritative_status,
        "authoritative_run_complete": authoritative_complete,
        "authoritative_run_provenance": authoritative_provenance,
        "resolved_authoritative_run": resolved_authoritative,
        "resolved_latest_complete": resolved_latest_complete,
        "run_count": len(runs),
        "runs": runs,
        "incomplete_runs": incomplete_runs,
        "failed_runs": failed_runs,
        "unsafe": bool(unsafe_reasons),
        "unsafe_reasons": unsafe_reasons,
    }


def _looks_like_run_parent(group: Any, path: str, child_names: list[str]) -> bool:
    if not child_names:
        return False
    attrs = getattr(group, "attrs", {})
    if any(
        attr in attrs
        for attr in ("latest", RUN_LATEST_COMPLETE_ATTR, RUN_LATEST_PENDING_ATTR)
    ):
        return True
    base = path.rstrip("/").rsplit("/", 1)[-1]
    return base.endswith("_runs") or base == "quality_reports"


def iter_run_parent_summaries(
    root_group: Any,
    *,
    root_path: str = "",
    legacy_default: bool | None = None,
) -> Iterator[dict[str, Any]]:
    """Yield completion summaries for run-parent-like groups in a Zarr tree."""

    child_names = _group_names(root_group)
    if _looks_like_run_parent(root_group, root_path, child_names):
        yield describe_run_parent(
            root_group,
            parent_path=root_path,
            legacy_default=legacy_default,
        )

    for name in child_names:
        child = _get_child(root_group, name)
        if child is None:
            continue
        child_path = f"{root_path}/{name}" if root_path else name
        yield from iter_run_parent_summaries(
            child,
            root_path=child_path,
            legacy_default=legacy_default,
        )
