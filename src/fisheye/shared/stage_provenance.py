"""Shared helpers for stage-level provenance contract compatibility."""

from __future__ import annotations

import json
from copy import deepcopy
from typing import Any, Dict, Mapping, MutableMapping, Optional, Protocol

STAGE_PROVENANCE_CONTRACT_NAME = "palette_stage_provenance"
STAGE_PROVENANCE_CONTRACT_VERSION = 1

_MISSING_STRING_VALUES = {"", "unknown", "n/a", "na", "none", "null"}


class _AttrsContainer(Protocol):
    attrs: MutableMapping[str, Any]


def _normalize_scalar(value: Any) -> Any:
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8", "ignore")
    return value


def _is_missing(value: Any) -> bool:
    value = _normalize_scalar(value)
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in _MISSING_STRING_VALUES
    return False


def _first_present(*values: Any) -> Any:
    for value in values:
        normalized = _normalize_scalar(value)
        if not _is_missing(normalized):
            return normalized
    return None


def _as_mapping(value: Any) -> Dict[str, Any]:
    value = _normalize_scalar(value)
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, str):
        try:
            payload = json.loads(value)
        except json.JSONDecodeError:
            return {}
        if isinstance(payload, Mapping):
            return dict(payload)
    return {}


def _as_int(value: Any, default: int) -> int:
    if isinstance(value, bool):
        return int(value)
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _normalize_contract(raw_contract: Any) -> Dict[str, Any]:
    contract = _as_mapping(raw_contract)
    name = contract.get("name")
    if not isinstance(name, str) or not name.strip():
        name = STAGE_PROVENANCE_CONTRACT_NAME
    version = _as_int(contract.get("version"), STAGE_PROVENANCE_CONTRACT_VERSION)

    normalized: Dict[str, Any] = {"name": name, "version": version}
    for key, value in contract.items():
        if key in {"name", "version"}:
            continue
        normalized[key] = value
    return normalized


def _normalize_git_payload(raw_git: Any) -> Dict[str, Any]:
    git = _as_mapping(raw_git)
    commit = _first_present(git.get("commit"), git.get("commit_hash"))
    short = _first_present(git.get("short"))
    if short is None and isinstance(commit, str):
        short = commit[:8]
    return {
        "commit": commit,
        "short": short,
        "branch": _first_present(git.get("branch")),
        "is_dirty": _first_present(git.get("is_dirty")),
        "remote": _first_present(git.get("remote"), git.get("remote_url")),
    }


def get_stage_provenance(attrs: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """Return normalized stage provenance payload with contract defaults."""
    attrs_map = attrs or {}
    payload = deepcopy(_as_mapping(attrs_map.get("provenance")))
    payload["contract"] = _normalize_contract(payload.get("contract"))
    return payload


def get_stage_contract(attrs: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """Return normalized contract identifier payload."""
    contract = get_stage_provenance(attrs).get("contract")
    normalized = _normalize_contract(contract)
    return {
        "name": normalized["name"],
        "version": normalized["version"],
    }


def get_stage_git(attrs: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """Return normalized git payload with legacy fallbacks."""
    attrs_map = attrs or {}
    provenance = get_stage_provenance(attrs_map)
    git = _normalize_git_payload(provenance.get("git"))

    commit = _first_present(
        git.get("commit"),
        attrs_map.get("git_commit"),
        attrs_map.get("git_commit_hash"),
    )
    short = _first_present(git.get("short"), attrs_map.get("git_short"))
    if short is None and isinstance(commit, str):
        short = commit[:8]
    return {
        "commit": commit,
        "short": short,
        "branch": _first_present(git.get("branch"), attrs_map.get("git_branch")),
        "is_dirty": _first_present(git.get("is_dirty"), attrs_map.get("git_is_dirty")),
        "remote": _first_present(git.get("remote"), attrs_map.get("git_remote")),
    }


def build_stage_provenance(
    *,
    stage: str,
    created_at_utc: str,
    parameters: Mapping[str, Any],
    inputs: Mapping[str, Any],
    command: Optional[str] = None,
    version: Optional[str] = None,
    git: Optional[Mapping[str, Any]] = None,
    environment: Optional[Mapping[str, Any]] = None,
    platform: Optional[Mapping[str, Any]] = None,
    scheduler: Optional[Mapping[str, Any]] = None,
    artifacts: Optional[Mapping[str, Any]] = None,
    contract_name: str = STAGE_PROVENANCE_CONTRACT_NAME,
    contract_version: int = STAGE_PROVENANCE_CONTRACT_VERSION,
) -> Dict[str, Any]:
    """Build canonical stage provenance payload."""
    payload: Dict[str, Any] = {
        "contract": _normalize_contract(
            {"name": contract_name, "version": contract_version}
        ),
        "stage": stage,
        "created_at_utc": created_at_utc,
        "parameters": deepcopy(dict(parameters)),
        "inputs": deepcopy(dict(inputs)),
    }

    if command is not None:
        payload["command"] = command
    if version is not None:
        payload["version"] = version
    if git is not None:
        payload["git"] = _normalize_git_payload(git)
    if environment is not None:
        payload["environment"] = deepcopy(dict(environment))
    if platform is not None:
        payload["platform"] = deepcopy(dict(platform))
    if scheduler is not None:
        payload["scheduler"] = deepcopy(dict(scheduler))
    if artifacts is not None:
        payload["artifacts"] = deepcopy(dict(artifacts))

    return payload


def write_stage_provenance(
    run_group: _AttrsContainer,
    payload: Mapping[str, Any],
    *,
    include_top_level_git: bool = True,
) -> None:
    """Write canonical stage provenance and optional top-level git attrs."""
    normalized = get_stage_provenance({"provenance": dict(payload)})
    run_group.attrs["provenance"] = normalized
    created_at_utc = normalized.get("created_at_utc")
    if created_at_utc is not None:
        run_group.attrs["created_at_utc"] = created_at_utc

    if not include_top_level_git:
        return

    git = get_stage_git({"provenance": normalized})
    if git["commit"] is not None:
        run_group.attrs["git_commit"] = git["commit"]
    if git["branch"] is not None:
        run_group.attrs["git_branch"] = git["branch"]
