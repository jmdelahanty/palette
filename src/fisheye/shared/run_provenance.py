"""Run-level provenance payloads required by completion finalization."""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
import hashlib
import json
from importlib import metadata
import os
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence

from fisheye.shared.system_metadata import get_environment_summary, get_gpu_info

RUN_PROVENANCE_SCHEMA = "palette.run_provenance.v1"
RUN_PROVENANCE_ATTR = "run_provenance"
CLI_RUN_PROVENANCE_ATTR = "cli_provenance"

VALUE_REQUIRED_FIELDS = ("git_sha", "config_hash")
STRUCTURALLY_REQUIRED_FIELDS = ("params", "input_run_ids", "command", "fisheye_version")


@dataclass(frozen=True)
class RunProvenanceValidation:
    """Validation result for the narrow finalization provenance gate."""

    valid: bool
    errors: tuple[str, ...] = ()
    missing_value_required: tuple[str, ...] = ()
    missing_structural: tuple[str, ...] = ()
    normalized: dict[str, Any] | None = None


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
    """Return the actual git state visible to this process."""

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


def scheduler_context() -> dict[str, Any]:
    """Return scheduler metadata when running under a batch system."""

    lsf_keys = ("LSB_JOBID", "LSB_JOBINDEX", "LSB_JOBNAME", "LSB_QUEUE", "LSB_HOSTS")
    lsf = {key.lower(): os.environ.get(key) for key in lsf_keys if os.environ.get(key) is not None}
    slurm_keys = ("SLURM_JOB_ID", "SLURM_JOB_NAME", "SLURM_NODELIST", "SLURM_PROCID")
    slurm = {key.lower(): os.environ.get(key) for key in slurm_keys if os.environ.get(key) is not None}
    out: dict[str, Any] = {}
    if lsf:
        out["lsf"] = lsf
    if slurm:
        out["slurm"] = slurm
    return out


def system_context() -> dict[str, Any]:
    """Return captured, non-blocking system metadata for provenance."""

    out: dict[str, Any] = {}
    try:
        out["environment"] = get_environment_summary()
    except Exception as exc:
        out["environment_unavailable_reason"] = str(exc)
    try:
        out["gpu"] = get_gpu_info()
    except Exception as exc:
        out["gpu_unavailable_reason"] = str(exc)
    return out


def build_run_provenance(
    *,
    command: str,
    params: Mapping[str, Any],
    input_run_ids: Mapping[str, Any] | None = None,
    cwd: Path | None = None,
    include_system_context: bool = True,
) -> dict[str, Any]:
    normalized_params = json_ready(dict(params))
    normalized_input_run_ids = json_ready(dict(input_run_ids or {}))
    provenance = {
        "schema": RUN_PROVENANCE_SCHEMA,
        **git_identity(cwd=cwd),
        "fisheye_version": fisheye_version(),
        "config_hash": sha256_payload(normalized_params),
        "params": normalized_params,
        "input_run_ids": normalized_input_run_ids,
        "command": str(command),
    }
    scheduler = scheduler_context()
    if scheduler:
        provenance["scheduler"] = scheduler
    if include_system_context:
        provenance["system"] = system_context()
    return provenance


def build_writer_run_provenance(
    *,
    command: str,
    params: Mapping[str, Any] | None = None,
    input_run_ids: Mapping[str, Any] | None = None,
    cwd: Path | None = None,
    include_system_context: bool = False,
) -> dict[str, Any]:
    """Build finalization provenance for direct writer paths.

    Lower-volume writers often already store richer stage provenance. This
    helper keeps the finalization gate narrow: command, normalized parameters,
    input identifiers, and code identity. System context remains opt-in because
    it is captured-never-blocks and can be comparatively expensive in tests.
    """

    return build_run_provenance(
        command=command,
        params=params or {},
        input_run_ids=input_run_ids or {},
        cwd=cwd,
        include_system_context=include_system_context,
    )


def build_run_provenance_from_stage_record(
    stage_record: Mapping[str, Any],
    *,
    fallback_command: str | None = None,
    cwd: Path | None = None,
    include_system_context: bool = False,
) -> dict[str, Any]:
    """Derive finalization provenance from a stage-provenance payload."""

    command = stage_record.get("command") or fallback_command or stage_record.get("stage") or "unknown"
    raw_params = stage_record.get("parameters")
    params = raw_params if isinstance(raw_params, Mapping) else {}
    raw_inputs = stage_record.get("inputs")
    input_run_ids = raw_inputs if isinstance(raw_inputs, Mapping) else {}
    return build_writer_run_provenance(
        command=str(command),
        params=params,
        input_run_ids=input_run_ids,
        cwd=cwd,
        include_system_context=include_system_context,
    )


def normalize_run_provenance(payload: Mapping[str, Any]) -> dict[str, Any]:
    normalized = json_ready(dict(payload))
    if not isinstance(normalized, dict):
        return {}
    normalized.setdefault("schema", RUN_PROVENANCE_SCHEMA)
    return normalized


def _is_empty_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    return False


def validate_run_provenance(payload: Mapping[str, Any] | None) -> RunProvenanceValidation:
    """Validate the approved finalization-gate fields.

    Only ``git_sha`` and ``config_hash`` are value-required. Structural fields
    must exist as keys but may carry empty values.
    """

    if not isinstance(payload, Mapping):
        return RunProvenanceValidation(
            valid=False,
            errors=("run provenance must be a mapping",),
            missing_value_required=VALUE_REQUIRED_FIELDS,
            missing_structural=STRUCTURALLY_REQUIRED_FIELDS,
            normalized={},
        )

    normalized = normalize_run_provenance(payload)
    missing_value = tuple(field for field in VALUE_REQUIRED_FIELDS if _is_empty_value(normalized.get(field)))
    missing_structural = tuple(field for field in STRUCTURALLY_REQUIRED_FIELDS if field not in normalized)
    errors: list[str] = []
    if missing_value:
        errors.append("missing value-required fields: " + ", ".join(missing_value))
    if missing_structural:
        errors.append("missing structurally-required fields: " + ", ".join(missing_structural))
    return RunProvenanceValidation(
        valid=not errors,
        errors=tuple(errors),
        missing_value_required=missing_value,
        missing_structural=missing_structural,
        normalized=normalized,
    )
