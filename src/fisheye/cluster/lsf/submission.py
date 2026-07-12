"""Topological LSF workflow submission with incremental durable state."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, Callable, Mapping

from fisheye.cluster.lsf.backend import (
    CommandExecutionError,
    CommandRunner,
    build_bsub_command,
    parse_bsub_job_id,
    render_dependency,
    run_command,
)
from fisheye.cluster.lsf.bundle import write_json_snapshot
from fisheye.cluster.lsf.models import LsfJob, LsfWorkflow
from fisheye.shared.batch_logging import utc_now


LSF_SUBMISSION_SCHEMA = "palette.lsf_workflow_submission.v1"
JobSubmittedCallback = Callable[[Mapping[str, Any]], None]


def _submission_payload(
    *,
    workflow: LsfWorkflow,
    plan_path: Path,
    submission_path: Path,
    cwd: Path,
    status: str,
    job_records: list[dict[str, Any]],
    job_ids_by_key: Mapping[str, str],
    error: Mapping[str, Any] | None = None,
    failed_job: Mapping[str, Any] | None = None,
    submitted_at_utc: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": LSF_SUBMISSION_SCHEMA,
        "status": status,
        "updated_at_utc": utc_now(),
        "workflow_id": workflow.workflow_id,
        "family": workflow.family,
        "plan_path": str(plan_path),
        "submission_path": str(submission_path),
        "cwd": str(cwd),
        "jobs": list(job_records),
        "job_ids_by_key": dict(job_ids_by_key),
    }
    if submitted_at_utc is not None:
        payload["submitted_at_utc"] = submitted_at_utc
    if failed_job is not None:
        payload["failed_job"] = dict(failed_job)
    if error is not None:
        payload["error"] = dict(error)
    return payload


def _job_record(
    *,
    job: LsfJob,
    job_id: str,
    bsub_command: list[str],
    command_result: Mapping[str, Any],
    job_ids_by_key: Mapping[str, str],
) -> dict[str, Any]:
    dependency_expression = (
        render_dependency(job.dependency, job_ids_by_key)
        if job.dependency is not None
        else None
    )
    return {
        "job_key": job.job_key,
        "job_name": job.job_name,
        "job_id": str(job_id),
        "dependency": dependency_expression,
        "bsub_command": list(bsub_command),
        "command_result": dict(command_result),
        "stdout_path": str(job.stdout_path).replace("%J", str(job_id)),
        "stderr_path": str(job.stderr_path).replace("%J", str(job_id)),
        "job": job.to_json(),
    }


def submit_lsf_workflow(
    workflow: LsfWorkflow,
    *,
    cwd: Path,
    plan_path: Path,
    submission_path: Path,
    runner: CommandRunner = subprocess.run,
    on_job_submitted: JobSubmittedCallback | None = None,
) -> dict[str, Any]:
    """Submit a validated workflow in dependency order and persist each step."""

    cwd = cwd.expanduser()
    plan_path = plan_path.expanduser()
    submission_path = submission_path.expanduser()
    write_json_snapshot(plan_path, workflow.to_json())

    job_records: list[dict[str, Any]] = []
    job_ids_by_key: dict[str, str] = {}
    initial = _submission_payload(
        workflow=workflow,
        plan_path=plan_path,
        submission_path=submission_path,
        cwd=cwd,
        status="submitting",
        job_records=job_records,
        job_ids_by_key=job_ids_by_key,
    )
    write_json_snapshot(submission_path, initial)

    current_job: LsfJob | None = None
    current_command: list[str] | None = None
    current_result: Mapping[str, Any] | None = None
    try:
        for job in workflow.topological_jobs():
            current_job = job
            current_command = build_bsub_command(job, job_ids_by_key)
            current_result = None
            current_result = run_command(current_command, cwd=cwd, runner=runner)
            job_id = parse_bsub_job_id(
                current_result["stdout"],
                current_result["stderr"],
            )
            job_ids_by_key[job.job_key] = job_id
            record = _job_record(
                job=job,
                job_id=job_id,
                bsub_command=current_command,
                command_result=current_result,
                job_ids_by_key=job_ids_by_key,
            )
            job_records.append(record)
            update = _submission_payload(
                workflow=workflow,
                plan_path=plan_path,
                submission_path=submission_path,
                cwd=cwd,
                status="submitting",
                job_records=job_records,
                job_ids_by_key=job_ids_by_key,
            )
            write_json_snapshot(submission_path, update)
            if on_job_submitted is not None:
                on_job_submitted(record)
    except Exception as exc:
        failed_job: dict[str, Any] = {}
        if current_job is not None:
            failed_job["job_key"] = current_job.job_key
            failed_job["job_name"] = current_job.job_name
        if current_command is not None:
            failed_job["bsub_command"] = list(current_command)
        if isinstance(exc, CommandExecutionError):
            failed_job["command_result"] = dict(exc.result)
        elif current_result is not None:
            failed_job["command_result"] = dict(current_result)
        failure = _submission_payload(
            workflow=workflow,
            plan_path=plan_path,
            submission_path=submission_path,
            cwd=cwd,
            status="submission_failed",
            job_records=job_records,
            job_ids_by_key=job_ids_by_key,
            failed_job=failed_job or None,
            error={"type": type(exc).__name__, "message": str(exc)},
        )
        write_json_snapshot(submission_path, failure)
        raise

    submitted_at_utc = utc_now()
    completed = _submission_payload(
        workflow=workflow,
        plan_path=plan_path,
        submission_path=submission_path,
        cwd=cwd,
        status="submitted",
        job_records=job_records,
        job_ids_by_key=job_ids_by_key,
        submitted_at_utc=submitted_at_utc,
    )
    return write_json_snapshot(submission_path, completed)


__all__ = [
    "JobSubmittedCallback",
    "LSF_SUBMISSION_SCHEMA",
    "submit_lsf_workflow",
]
