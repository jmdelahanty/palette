"""Shared helpers for model-resolution provenance payloads."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

from fisheye.shared.batch_logging import utc_now
from fisheye.shared.system_metadata import build_invocation_record

MODEL_RESOLUTION_CONTRACT_NAME = "palette_model_resolution_provenance"
MODEL_RESOLUTION_CONTRACT_VERSION = 1


_utc_now = utc_now


def _to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_jsonable(v) for v in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return str(value)


def _normalize_git_payload(raw_git: Any) -> Dict[str, Any]:
    git = raw_git if isinstance(raw_git, Mapping) else {}
    commit = git.get("commit") or git.get("commit_hash")
    short = git.get("short") or git.get("short_hash")
    if not short and isinstance(commit, str):
        short = commit[:8]
    return {
        "commit": _to_jsonable(commit),
        "short": _to_jsonable(short),
        "branch": _to_jsonable(git.get("branch")),
        "is_dirty": _to_jsonable(git.get("is_dirty")),
        "remote": _to_jsonable(git.get("remote") or git.get("remote_url")),
    }


def build_model_resolution_payload(
    *,
    tool: str,
    args: Any,
    argv: Optional[Sequence[str]],
    task: str,
    registry_path: Path,
    recording_id: str,
    target: Mapping[str, Any],
    selected: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    method: Optional[str] = None,
    parameters: Optional[Mapping[str, Any]] = None,
    inputs: Optional[Mapping[str, Any]] = None,
    artifacts: Optional[Mapping[str, Any]] = None,
    resolved_at_utc: Optional[str] = None,
) -> Dict[str, Any]:
    invocation = build_invocation_record(
        tool=tool,
        args=args,
        argv=list(argv) if argv is not None else None,
    )
    invocation_args = invocation.get("args")
    invocation_args_map = dict(invocation_args) if isinstance(invocation_args, Mapping) else {}

    merged_parameters: Dict[str, Any] = dict(invocation_args_map)
    if parameters:
        merged_parameters.update({str(k): v for k, v in parameters.items()})

    payload: Dict[str, Any] = {
        "contract": {
            "name": MODEL_RESOLUTION_CONTRACT_NAME,
            "version": MODEL_RESOLUTION_CONTRACT_VERSION,
        },
        "mode": "registry",
        "task": task,
        "registry_path": str(registry_path),
        "recording_id": recording_id,
        "resolved_at_utc": resolved_at_utc or _utc_now(),
        "target": _to_jsonable(deepcopy(dict(target))),
        "selected": _to_jsonable(deepcopy(dict(selected))),
        "candidates": _to_jsonable([deepcopy(dict(item)) for item in candidates]),
        "command": _to_jsonable(invocation.get("command")),
        "git": _normalize_git_payload(invocation.get("git")),
        "environment": _to_jsonable(invocation.get("environment")),
        "platform": _to_jsonable(invocation.get("platform")),
        "parameters": _to_jsonable(merged_parameters),
        "inputs": _to_jsonable(dict(inputs or {})),
        "artifacts": _to_jsonable(dict(artifacts or {})),
    }
    if method is not None:
        payload["method"] = _to_jsonable(method)
    return payload
