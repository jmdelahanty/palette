"""Runtime status envelope for commands executed inside LSF jobs."""

from __future__ import annotations

import argparse
import os
import signal
import shutil
import socket
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

from fisheye.cluster.lsf.bundle import write_json_snapshot
from fisheye.shared.batch_logging import utc_now


LSF_RUNTIME_STATUS_SCHEMA = "palette.lsf_job_runtime_status.v1"
RUNTIME_JOB_ID_TOKEN = "__PALETTE_LSF_JOBID__"
RUNTIME_JOB_INDEX_TOKEN = "__PALETTE_LSF_JOBINDEX__"
RUNTIME_USER_TOKEN = "__PALETTE_LSF_USER__"


def _runtime_tokens(environment: Mapping[str, str]) -> dict[str, str]:
    job_id = str(environment.get("LSB_JOBID") or "manual")
    job_index = str(environment.get("LSB_JOBINDEX") or "0")
    user = str(environment.get("USER") or "unknown")
    return {
        RUNTIME_JOB_ID_TOKEN: job_id,
        RUNTIME_JOB_INDEX_TOKEN: job_index,
        RUNTIME_USER_TOKEN: user,
        # Read historical plan/status templates without emitting shell-active
        # angle-bracket placeholders in new bsub command arguments.
        "<jobid>": job_id,
        "<jobindex>": job_index,
        "<user>": user,
    }


def expand_runtime_tokens(value: str | Path, environment: Mapping[str, str]) -> str:
    """Expand the small scheduler-token vocabulary used in planned paths."""

    text = str(value)
    for token, replacement in _runtime_tokens(environment).items():
        text = text.replace(token, replacement)
    return text


def _scheduler_payload(environment: Mapping[str, str]) -> dict[str, Any]:
    return {
        "job_id": environment.get("LSB_JOBID"),
        "job_index": environment.get("LSB_JOBINDEX"),
        "job_name": environment.get("LSB_JOBNAME"),
        "queue": environment.get("LSB_QUEUE"),
        "hosts": environment.get("LSB_HOSTS"),
        "mcpu_hosts": environment.get("LSB_MCPU_HOSTS"),
        "cuda_visible_devices": environment.get("CUDA_VISIBLE_DEVICES"),
        "hostname": socket.gethostname(),
    }


def _parse_environment_overrides(values: Sequence[str]) -> dict[str, str]:
    overrides: dict[str, str] = {}
    for value in values:
        key, separator, raw = str(value).partition("=")
        key = key.strip()
        if not separator or not key:
            raise ValueError(f"Runtime environment override must be KEY=VALUE: {value!r}")
        overrides[key] = raw
    return overrides


def _cleanup_job_scratch(
    path_template: str,
    *,
    environment: Mapping[str, str],
) -> dict[str, Any]:
    expanded = Path(expand_runtime_tokens(path_template, environment)).expanduser()
    user = str(environment.get("USER") or "")
    job_id = str(environment.get("LSB_JOBID") or "")
    job_index = str(environment.get("LSB_JOBINDEX") or "").strip()
    result: dict[str, Any] = {
        "requested_path": str(path_template),
        "expanded_path": str(expanded),
    }
    if not user or not job_id:
        return {
            **result,
            "status": "refused",
            "reason": "LSB_JOBID and USER are required for job-scratch cleanup",
        }
    work_unit = f"{job_id}_{job_index}" if job_index else job_id
    allowed_root = Path("/scratch") / user / work_unit
    resolved = expanded.resolve(strict=False)
    resolved_root = allowed_root.resolve(strict=False)
    if resolved == resolved_root or resolved_root not in resolved.parents:
        return {
            **result,
            "status": "refused",
            "allowed_root": str(resolved_root),
            "reason": "cleanup path is not a child of this LSF job scratch root",
        }
    if not resolved.exists():
        return {**result, "status": "not_found"}
    try:
        shutil.rmtree(resolved)
    except Exception as exc:
        return {
            **result,
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
        }
    return {**result, "status": "removed"}


def _status_payload(
    *,
    workflow_id: str,
    family: str,
    job_key: str,
    stage: str,
    status: str,
    command: Sequence[str],
    cwd: Path,
    environment: Mapping[str, str],
    environment_overrides: Mapping[str, str],
    started_at_utc: str,
    finished_at_utc: str | None = None,
    returncode: int | None = None,
    error: str | None = None,
    signal_number: int | None = None,
    cleanup: Sequence[Mapping[str, Any]] = (),
    expected_outputs: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": LSF_RUNTIME_STATUS_SCHEMA,
        "workflow_id": workflow_id,
        "family": family,
        "job_key": job_key,
        "stage": stage,
        "status": status,
        "command": list(command),
        "cwd": str(cwd),
        "environment_override_keys": sorted(environment_overrides),
        "scheduler": _scheduler_payload(environment),
        "started_at_utc": started_at_utc,
        "finished_at_utc": finished_at_utc,
        "returncode": returncode,
        "error": error,
        "signal_number": signal_number,
        "signal_name": (
            signal.Signals(signal_number).name if signal_number is not None else None
        ),
        "cleanup": [dict(item) for item in cleanup],
        "expected_outputs": [dict(item) for item in expected_outputs],
    }
    return payload


def _expected_output_results(
    path_templates: Sequence[str],
    *,
    environment: Mapping[str, str],
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for template in path_templates:
        expanded = Path(expand_runtime_tokens(template, environment)).expanduser()
        results.append(
            {
                "requested_path": str(template),
                "expanded_path": str(expanded),
                "exists": expanded.exists(),
            }
        )
    return results


def run_with_status(
    command: Sequence[str],
    *,
    status_path_template: str | Path,
    workflow_id: str,
    family: str,
    job_key: str,
    stage: str,
    cwd: Path,
    environment_overrides: Mapping[str, str] | None = None,
    cleanup_path_templates: Sequence[str] = (),
    expected_output_templates: Sequence[str] = (),
    base_environment: Mapping[str, str] | None = None,
) -> int:
    """Execute one command, publish runtime status, and return its exit code."""

    environment = dict(os.environ if base_environment is None else base_environment)
    overrides = {str(key): str(value) for key, value in (environment_overrides or {}).items()}
    environment.update(overrides)
    resolved_command = [expand_runtime_tokens(value, environment) for value in command]
    resolved_cwd = Path(expand_runtime_tokens(cwd, environment)).expanduser()
    status_path = Path(
        expand_runtime_tokens(status_path_template, environment)
    ).expanduser()
    started_at_utc = utc_now()
    running = _status_payload(
        workflow_id=workflow_id,
        family=family,
        job_key=job_key,
        stage=stage,
        status="running",
        command=resolved_command,
        cwd=resolved_cwd,
        environment=environment,
        environment_overrides=overrides,
        started_at_utc=started_at_utc,
        expected_outputs=_expected_output_results(
            expected_output_templates,
            environment=environment,
        ),
    )
    write_json_snapshot(status_path, running)

    returncode = 1
    error: str | None = None
    received_signal: int | None = None
    child: subprocess.Popen[Any] | None = None
    previous_handlers: dict[int, Any] = {}

    def _forward_signal(signum: int, _frame: Any) -> None:
        nonlocal received_signal
        received_signal = int(signum)
        if child is not None and child.poll() is None:
            try:
                child.send_signal(signum)
            except ProcessLookupError:
                pass

    try:
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous_handlers[int(signum)] = signal.getsignal(signum)
            signal.signal(signum, _forward_signal)
        child = subprocess.Popen(
            resolved_command,
            cwd=str(resolved_cwd),
            env=environment,
        )
        child_returncode = int(child.wait())
        if received_signal is None and child_returncode < 0:
            received_signal = -child_returncode
        returncode = (
            128 + int(received_signal)
            if received_signal is not None
            else child_returncode
        )
        if received_signal is not None:
            error = (
                f"Runtime received {signal.Signals(received_signal).name} "
                "and forwarded it to the worker."
            )
    except Exception as exc:
        returncode = 127
        error = f"{type(exc).__name__}: {exc}"
    finally:
        for signum, previous in previous_handlers.items():
            signal.signal(signum, previous)

    cleanup_results = [
        _cleanup_job_scratch(path, environment=environment)
        for path in cleanup_path_templates
    ]
    cleanup_errors = [
        item for item in cleanup_results if item.get("status") in {"refused", "error"}
    ]
    if returncode == 0 and cleanup_errors:
        returncode = 1
        error = "One or more requested job-scratch cleanup operations failed."
    expected_outputs = _expected_output_results(
        expected_output_templates,
        environment=environment,
    )
    missing_outputs = [item for item in expected_outputs if not item["exists"]]
    if returncode == 0 and missing_outputs:
        returncode = 1
        error = "Worker exited successfully but one or more expected outputs are missing."
    status = "succeeded" if returncode == 0 else "failed"
    finished = _status_payload(
        workflow_id=workflow_id,
        family=family,
        job_key=job_key,
        stage=stage,
        status=status,
        command=resolved_command,
        cwd=resolved_cwd,
        environment=environment,
        environment_overrides=overrides,
        started_at_utc=started_at_utc,
        finished_at_utc=utc_now(),
        returncode=returncode,
        error=error,
        signal_number=received_signal,
        cleanup=cleanup_results,
        expected_outputs=expected_outputs,
    )
    write_json_snapshot(status_path, finished)
    return returncode


def build_runtime_command(
    command: Sequence[str],
    *,
    status_path_template: str | Path,
    workflow_id: str,
    family: str,
    job_key: str,
    stage: str,
    cwd: Path,
    environment_overrides: Mapping[str, str] | None = None,
    cleanup_path_templates: Sequence[str] = (),
    expected_output_templates: Sequence[str] = (),
    python_launcher: Sequence[str] = ("scripts/py",),
) -> tuple[str, ...]:
    """Build the argv used by an LSF job to invoke :func:`run_with_status`."""

    argv = [
        *(str(value) for value in python_launcher),
        "-m",
        "fisheye.cluster.lsf.runtime",
        "--status-json",
        str(status_path_template),
        "--workflow-id",
        str(workflow_id),
        "--family",
        str(family),
        "--job-key",
        str(job_key),
        "--stage",
        str(stage),
        "--cwd",
        str(cwd),
    ]
    for key, value in (environment_overrides or {}).items():
        argv.extend(["--env", f"{key}={value}"])
    for path in cleanup_path_templates:
        argv.extend(["--cleanup-path", str(path)])
    for path in expected_output_templates:
        argv.extend(["--expected-output", str(path)])
    argv.extend(["--", *(str(value) for value in command)])
    return tuple(argv)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--status-json", required=True, type=Path)
    parser.add_argument("--workflow-id", required=True)
    parser.add_argument("--family", required=True)
    parser.add_argument("--job-key", required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--cwd", required=True, type=Path)
    parser.add_argument("--env", action="append", default=[])
    parser.add_argument("--cleanup-path", action="append", default=[])
    parser.add_argument("--expected-output", action="append", default=[])
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise SystemExit("A command is required after '--'.")
    overrides = _parse_environment_overrides(args.env)
    return run_with_status(
        command,
        status_path_template=args.status_json,
        workflow_id=args.workflow_id,
        family=args.family,
        job_key=args.job_key,
        stage=args.stage,
        cwd=args.cwd,
        environment_overrides=overrides,
        cleanup_path_templates=args.cleanup_path,
        expected_output_templates=args.expected_output,
    )


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "LSF_RUNTIME_STATUS_SCHEMA",
    "RUNTIME_JOB_ID_TOKEN",
    "RUNTIME_JOB_INDEX_TOKEN",
    "RUNTIME_USER_TOKEN",
    "build_runtime_command",
    "expand_runtime_tokens",
    "run_with_status",
]
