from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict, is_dataclass
from importlib import metadata
from pathlib import Path
from typing import Any, Mapping, Sequence


EXIT_OK = 0
EXIT_FAILED = 1
EXIT_BLOCKED = 2
EXIT_USAGE = 3

SCHEMA = "palette.cli.workflow_oracle.v1"


def json_ready(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return json_ready(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): json_ready(item) for key, item in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [json_ready(item) for item in value]
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return json_ready(item())
        except Exception:
            pass
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        try:
            return json_ready(tolist())
        except Exception:
            pass
    return str(value)


def stable_json(value: Any) -> str:
    return json.dumps(json_ready(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_payload(value: Any) -> str:
    return hashlib.sha256(stable_json(value).encode("utf-8")).hexdigest()


def _run_git(args: Sequence[str], *, cwd: Path | None) -> tuple[str | None, str | None]:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=str(cwd) if cwd is not None else None,
            check=False,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
    except Exception as exc:
        return None, str(exc)
    if completed.returncode != 0:
        message = (completed.stderr or completed.stdout or "").strip()
        return None, message or f"git exited {completed.returncode}"
    return completed.stdout.strip(), None


def git_identity(*, cwd: Path | None = None) -> dict[str, Any]:
    sha, sha_error = _run_git(["rev-parse", "HEAD"], cwd=cwd)
    short, _short_error = _run_git(["rev-parse", "--short", "HEAD"], cwd=cwd)
    status, dirty_error = _run_git(["status", "--porcelain"], cwd=cwd)
    dirty = None if status is None else bool(status.strip())
    out: dict[str, Any] = {
        "git_sha": sha,
        "git_short_sha": short,
        "git_dirty": dirty,
    }
    if sha_error:
        out["git_unavailable_reason"] = sha_error
    elif dirty_error:
        out["git_dirty_unavailable_reason"] = dirty_error
    return out


def fisheye_version() -> str | None:
    for package_name in ("palette", "fisheye"):
        try:
            return metadata.version(package_name)
        except metadata.PackageNotFoundError:
            continue
    return None


def build_run_provenance(
    *,
    command: str,
    params: Mapping[str, Any],
    input_run_ids: Mapping[str, Any] | None = None,
    cwd: Path | None = None,
) -> dict[str, Any]:
    normalized_params = json_ready(dict(params))
    normalized_input_run_ids = json_ready(dict(input_run_ids or {}))
    provenance = {
        **git_identity(cwd=cwd),
        "fisheye_version": fisheye_version(),
        "config_hash": sha256_payload(normalized_params),
        "params": normalized_params,
        "input_run_ids": normalized_input_run_ids,
        "command": str(command),
    }
    return provenance


def build_envelope(
    *,
    command: str,
    status: str,
    reason_code: str,
    recording: str | None = None,
    dataset_id: str | None = None,
    zarr_path: str | Path | None = None,
    run: str | None = None,
    artifacts: Sequence[Any] | None = None,
    metrics: Mapping[str, Any] | None = None,
    next_hints: Sequence[str] | None = None,
    provenance: Mapping[str, Any] | None = None,
    **extra: Any,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "command": str(command),
        "status": str(status),
        "reason_code": str(reason_code),
        "recording": recording,
        "dataset_id": dataset_id,
        "zarr_path": str(zarr_path) if zarr_path is not None else None,
        "run": run,
        "artifacts": json_ready(list(artifacts or ([] if zarr_path is None else [zarr_path]))),
        "metrics": json_ready(dict(metrics or {})),
        "next_hints": [str(item) for item in (next_hints or [])],
        "provenance": json_ready(dict(provenance or {})),
    }
    for key, value in extra.items():
        payload[str(key)] = json_ready(value)
    return payload


def exit_code_for_status(status: str) -> int:
    if status in {"ok", "dry_run"}:
        return EXIT_OK
    if status == "blocked":
        return EXIT_BLOCKED
    return EXIT_FAILED


def print_json(payload: Mapping[str, Any]) -> None:
    print(json.dumps(json_ready(dict(payload)), indent=2, sort_keys=True))
