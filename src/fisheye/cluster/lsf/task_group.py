"""Dispatch typed LSF array tasks or bounded CPU bundles from an immutable plan."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from fisheye.cluster.lsf.bundle import write_json_snapshot
from fisheye.cluster.lsf.runtime import expand_runtime_tokens, run_with_status
from fisheye.shared.batch_logging import utc_now


TASK_GROUP_SUMMARY_SCHEMA = "palette.lsf_task_group_summary.v1"


def _read_json(path: Path) -> Any:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value!r}")

    return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)


def _load_job(plan_path: Path, job_key: str) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    plan = _read_json(plan_path)
    if not isinstance(plan, Mapping) or plan.get("schema") != "palette.lsf_workflow.v1":
        raise ValueError(f"Unsupported LSF workflow plan: {plan_path}")
    jobs = plan.get("jobs")
    if not isinstance(jobs, list):
        raise ValueError(f"LSF workflow plan has no jobs list: {plan_path}")
    matches = [
        job
        for job in jobs
        if isinstance(job, Mapping) and str(job.get("job_key") or "") == job_key
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one LSF job {job_key!r}; found {len(matches)}.")
    job = matches[0]
    group = job.get("execution_group")
    if not isinstance(group, Mapping):
        raise ValueError(f"LSF job {job_key!r} has no execution_group.")
    tasks = group.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError(f"LSF job {job_key!r} has no execution tasks.")
    if int(group.get("task_count") or -1) != len(tasks):
        raise ValueError(f"LSF job {job_key!r} execution task count is inconsistent.")
    return plan, job


def _task(job: Mapping[str, Any], task_index: int) -> Mapping[str, Any]:
    group = job["execution_group"]
    tasks = group["tasks"]
    if task_index < 1 or task_index > len(tasks):
        raise ValueError(
            f"Task index {task_index} is outside 1..{len(tasks)} for {job['job_key']!r}."
        )
    task = tasks[task_index - 1]
    if not isinstance(task, Mapping):
        raise ValueError(f"Execution task {task_index} is not an object.")
    command = task.get("command")
    if not isinstance(command, list) or not command:
        raise ValueError(f"Execution task {task_index} has no command.")
    return task


def _run_task(
    *,
    plan: Mapping[str, Any],
    job: Mapping[str, Any],
    task_index: int,
    cwd: Path,
    environment: Mapping[str, str],
) -> int:
    task = _task(job, task_index)
    return run_with_status(
        tuple(str(value) for value in task["command"]),
        status_path_template=Path(str(task["status_path"])),
        workflow_id=str(plan["workflow_id"]),
        family=str(plan["family"]),
        job_key=str(task["task_key"]),
        stage=str(task["stage"]),
        cwd=cwd,
        cleanup_path_templates=tuple(str(value) for value in task.get("cleanup_paths") or ()),
        expected_output_templates=tuple(
            str(value) for value in task.get("expected_outputs") or ()
        ),
        base_environment=environment,
    )


def _bundle_child_command(
    *,
    plan_path: Path,
    job_key: str,
    task_index: int,
    cwd: Path,
) -> tuple[str, ...]:
    return (
        sys.executable,
        "-m",
        "fisheye.cluster.lsf.task_group",
        "--plan",
        str(plan_path),
        "--job-key",
        job_key,
        "--cwd",
        str(cwd),
        "--task-index",
        str(task_index),
    )


def _run_bundle(
    *,
    plan_path: Path,
    plan: Mapping[str, Any],
    job: Mapping[str, Any],
    cwd: Path,
    environment: Mapping[str, str],
) -> int:
    group = job["execution_group"]
    tasks = group["tasks"]
    max_concurrent = int(group["max_concurrent"])
    if max_concurrent <= 0 or max_concurrent > len(tasks):
        raise ValueError("Bundle max_concurrent is outside the task-count range.")
    summary_template = str(group.get("summary_path") or "")
    if not summary_template:
        raise ValueError("Bundle execution requires a summary_path.")
    summary_path = Path(expand_runtime_tokens(summary_template, environment)).expanduser()
    started_at = utc_now()
    pending = list(range(1, len(tasks) + 1))
    running: dict[int, subprocess.Popen[Any]] = {}
    results: dict[int, int] = {}
    received_signal: int | None = None
    previous_handlers: dict[int, Any] = {}

    def forward_signal(signum: int, _frame: Any) -> None:
        nonlocal received_signal
        received_signal = int(signum)
        for child in tuple(running.values()):
            if child.poll() is None:
                try:
                    child.send_signal(signum)
                except ProcessLookupError:
                    pass

    try:
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous_handlers[int(signum)] = signal.getsignal(signum)
            signal.signal(signum, forward_signal)
        stop_launching = False
        while pending or running:
            while pending and len(running) < max_concurrent and not stop_launching:
                index = pending.pop(0)
                running[index] = subprocess.Popen(
                    _bundle_child_command(
                        plan_path=plan_path,
                        job_key=str(job["job_key"]),
                        task_index=index,
                        cwd=cwd,
                    ),
                    cwd=str(cwd),
                    env=dict(environment),
                )
            completed: list[int] = []
            for index, child in running.items():
                returncode = child.poll()
                if returncode is None:
                    continue
                results[index] = int(returncode)
                completed.append(index)
                if returncode != 0:
                    stop_launching = True
            for index in completed:
                del running[index]
            if received_signal is not None:
                stop_launching = True
            if running:
                time.sleep(0.1)
            elif stop_launching:
                break
    finally:
        for signum, previous in previous_handlers.items():
            signal.signal(signum, previous)

    skipped = pending
    task_results = []
    for index, task in enumerate(tasks, start=1):
        if index in results:
            status = "succeeded" if results[index] == 0 else "failed"
            returncode: int | None = results[index]
        else:
            status = "skipped_after_failure"
            returncode = None
        task_results.append(
            {
                "task_index": index,
                "task_key": str(task["task_key"]),
                "status": status,
                "returncode": returncode,
                "status_path": expand_runtime_tokens(
                    str(task["status_path"]), environment
                ),
            }
        )
    failed = [index for index, returncode in results.items() if returncode != 0]
    status = "succeeded" if not failed and not skipped and received_signal is None else "failed"
    payload = {
        "schema": TASK_GROUP_SUMMARY_SCHEMA,
        "status": status,
        "workflow_id": str(plan["workflow_id"]),
        "family": str(plan["family"]),
        "job_key": str(job["job_key"]),
        "mode": "bundle",
        "max_concurrent": max_concurrent,
        "task_count": len(tasks),
        "started_at_utc": started_at,
        "finished_at_utc": utc_now(),
        "scheduler": {
            "job_id": environment.get("LSB_JOBID"),
            "job_name": environment.get("LSB_JOBNAME"),
            "hosts": environment.get("LSB_HOSTS"),
        },
        "signal_number": received_signal,
        "tasks": task_results,
    }
    write_json_snapshot(summary_path, payload)
    if received_signal is not None:
        return 128 + received_signal
    return 0 if status == "succeeded" else 1


def run_task_group(
    *,
    plan_path: Path,
    job_key: str,
    cwd: Path,
    task_index: int | None = None,
    environment: Mapping[str, str] | None = None,
) -> int:
    environment = dict(os.environ if environment is None else environment)
    plan, job = _load_job(plan_path, job_key)
    group = job["execution_group"]
    mode = str(group.get("mode") or "")
    if task_index is not None:
        return _run_task(
            plan=plan,
            job=job,
            task_index=int(task_index),
            cwd=cwd,
            environment=environment,
        )
    if mode == "array":
        raw_index = str(environment.get("LSB_JOBINDEX") or "").strip()
        if not raw_index:
            raise ValueError("LSB_JOBINDEX is required for LSF array execution.")
        return _run_task(
            plan=plan,
            job=job,
            task_index=int(raw_index),
            cwd=cwd,
            environment=environment,
        )
    if mode == "bundle":
        return _run_bundle(
            plan_path=plan_path,
            plan=plan,
            job=job,
            cwd=cwd,
            environment=environment,
        )
    raise ValueError(f"Unsupported LSF execution-group mode: {mode!r}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--job-key", required=True)
    parser.add_argument("--cwd", required=True, type=Path)
    parser.add_argument("--task-index", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    return run_task_group(
        plan_path=args.plan.expanduser().resolve(),
        job_key=args.job_key,
        cwd=args.cwd.expanduser().resolve(),
        task_index=args.task_index,
    )


if __name__ == "__main__":
    raise SystemExit(main())
