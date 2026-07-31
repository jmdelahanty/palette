"""Run-level provenance payloads required by completion finalization."""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
import hashlib
import json
from importlib import metadata
import os
import platform
from pathlib import Path
import socket
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


def _find_git_root(start: Path) -> Path | None:
    """Return the nearest Git worktree root at or above ``start``."""

    candidate = start.expanduser().resolve()
    if candidate.is_file():
        candidate = candidate.parent
    while True:
        if (candidate / ".git").exists():
            return candidate
        if candidate.parent == candidate:
            return None
        candidate = candidate.parent


def _git_command_cwd(cwd: Path | None) -> Path | None:
    """Resolve code identity independently of the process working directory."""

    if cwd is not None:
        return Path(cwd).expanduser().resolve()
    source_root = _find_git_root(Path(__file__).resolve())
    if source_root is not None:
        return source_root
    return _find_git_root(Path.cwd())


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

    git_cwd = _git_command_cwd(cwd)
    sha, sha_error = _run_git(["rev-parse", "HEAD"], cwd=git_cwd)
    short, _short_error = _run_git(["rev-parse", "--short", "HEAD"], cwd=git_cwd)
    status, dirty_error = _run_git(["status", "--porcelain"], cwd=git_cwd)
    dirty = None if status is None else bool(status.strip())
    out: dict[str, Any] = {
        "git_sha": sha,
        "git_short_sha": short,
        "git_dirty": dirty,
    }
    if git_cwd is not None:
        out["git_root"] = str(git_cwd)
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

    lsf_keys = (
        "LSB_JOBID",
        "LSB_JOBINDEX",
        "LSB_JOBNAME",
        "LSB_QUEUE",
        "LSB_HOSTS",
        "LSB_DJOB_NUMPROC",
        "LSB_MCPU_HOSTS",
    )
    lsf = {
        key.lower(): os.environ.get(key)
        for key in lsf_keys
        if os.environ.get(key) is not None
    }
    slurm_keys = (
        "SLURM_JOB_ID",
        "SLURM_JOB_NAME",
        "SLURM_NODELIST",
        "SLURM_PROCID",
        "SLURM_CPUS_PER_TASK",
        "SLURM_NTASKS",
        "SLURM_JOB_CPUS_PER_NODE",
    )
    slurm = {
        key.lower(): os.environ.get(key)
        for key in slurm_keys
        if os.environ.get(key) is not None
    }
    out: dict[str, Any] = {}
    if lsf:
        out["lsf"] = lsf
    if slurm:
        out["slurm"] = slurm
    return out


def _runtime_cpu_model() -> str:
    """Return a best-effort CPU model without launching another process."""

    cpuinfo_path = Path("/proc/cpuinfo")
    if cpuinfo_path.is_file():
        try:
            fields: dict[str, str] = {}
            cpuinfo = cpuinfo_path.read_text(encoding="utf-8", errors="replace")
            for line in cpuinfo.splitlines():
                key, separator, raw_value = line.partition(":")
                if not separator:
                    continue
                value = raw_value.strip()
                if value:
                    fields.setdefault(key.strip().lower(), value)
            for key in ("model name", "hardware", "processor"):
                if fields.get(key):
                    return fields[key]
        except OSError:
            pass
    return platform.processor().strip() or "unknown"


def runtime_context() -> dict[str, Any]:
    """Return lightweight host and CPU identity for every finalized run."""

    return {
        "host": {
            "hostname": socket.gethostname(),
        },
        "cpu": {
            "model": _runtime_cpu_model(),
            "architecture": platform.machine() or "unknown",
            "logical_count": os.cpu_count(),
        },
        "platform": {
            "system": platform.system() or "unknown",
            "kernel_release": platform.release() or "unknown",
        },
    }


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
    input_artifacts: Sequence[Mapping[str, Any]] | None = None,
    cwd: Path | None = None,
    include_system_context: bool = True,
) -> dict[str, Any]:
    normalized_params = json_ready(dict(params))
    normalized_input_run_ids = json_ready(dict(input_run_ids or {}))
    normalized_input_artifacts = json_ready(list(input_artifacts or []))
    provenance = {
        "schema": RUN_PROVENANCE_SCHEMA,
        **git_identity(cwd=cwd),
        "fisheye_version": fisheye_version(),
        "config_hash": sha256_payload(normalized_params),
        "params": normalized_params,
        "input_run_ids": normalized_input_run_ids,
        "input_artifacts": normalized_input_artifacts,
        "command": str(command),
    }
    scheduler = scheduler_context()
    if scheduler:
        provenance["scheduler"] = scheduler
    provenance["runtime"] = runtime_context()
    if include_system_context:
        provenance["system"] = system_context()
    return provenance


def build_writer_run_provenance(
    *,
    command: str,
    params: Mapping[str, Any] | None = None,
    input_run_ids: Mapping[str, Any] | None = None,
    input_artifacts: Sequence[Mapping[str, Any]] | None = None,
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
        input_artifacts=input_artifacts or [],
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
    provenance = build_writer_run_provenance(
        command=str(command),
        params=params,
        input_run_ids=input_run_ids,
        cwd=cwd,
        include_system_context=include_system_context,
    )
    # Reuse the accelerator snapshot already captured for stage provenance.
    # This avoids a second hardware probe and ensures short-lived inference
    # workers retain the same device identity in both provenance surfaces.
    raw_environment = stage_record.get("environment")
    if not include_system_context and isinstance(raw_environment, Mapping):
        accelerator = raw_environment.get("accelerator")
        if isinstance(accelerator, Mapping):
            environment = {
                str(key): value
                for key, value in raw_environment.items()
                if key != "accelerator"
            }
            provenance["system"] = {
                "environment": json_ready(environment),
                "gpu": json_ready(accelerator),
            }
    return provenance


def normalize_run_provenance(payload: Mapping[str, Any]) -> dict[str, Any]:
    normalized = json_ready(dict(payload))
    if not isinstance(normalized, dict):
        return {}
    normalized.setdefault("schema", RUN_PROVENANCE_SCHEMA)
    return normalized


def _artifact_dedup_key(artifact: Any) -> tuple[str, str] | None:
    if not isinstance(artifact, Mapping):
        return None
    role = artifact.get("role")
    path = artifact.get("path")
    if role is None or path is None:
        return None
    return str(role), str(path)


def append_input_artifacts(
    provenance: Mapping[str, Any],
    artifacts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Return provenance with ``artifacts`` merged into ``input_artifacts``.

    Existing provenance may be supplied by CLI/registry wrappers before a writer
    resolves its model files. Writers use this helper at model-load time so the
    model fingerprints are recorded regardless of who built the original
    provenance payload. Artifacts deduplicate on ``(role, path)``; later entries
    replace earlier entries for the same key.
    """

    normalized = normalize_run_provenance(provenance)
    existing_raw = normalized.get("input_artifacts")
    existing = existing_raw if isinstance(existing_raw, list) else []
    merged: list[Any] = []
    indices: dict[tuple[str, str], int] = {}

    for item in existing:
        normalized_item = json_ready(item)
        key = _artifact_dedup_key(normalized_item)
        if key is not None:
            indices[key] = len(merged)
        merged.append(normalized_item)

    for item in artifacts:
        normalized_item = json_ready(item)
        key = _artifact_dedup_key(normalized_item)
        if key is not None and key in indices:
            merged[indices[key]] = normalized_item
        else:
            if key is not None:
                indices[key] = len(merged)
            merged.append(normalized_item)

    normalized["input_artifacts"] = merged
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
