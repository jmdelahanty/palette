"""Rendering and execution helpers for stage-agnostic LSF jobs."""

from __future__ import annotations

import re
import shlex
import subprocess
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

from fisheye.cluster.lsf.models import (
    LsfDependency,
    LsfExecutionMode,
    LsfJob,
    LsfResources,
)


class CompletedProcessLike(Protocol):
    returncode: int
    stdout: str
    stderr: str


CommandRunner = Callable[..., CompletedProcessLike]


class CommandExecutionError(RuntimeError):
    """A failed command with its structured diagnostic result attached."""

    def __init__(self, message: str, *, result: Mapping[str, Any]) -> None:
        super().__init__(message)
        self.result = dict(result)


def shell_join(argv: Sequence[object]) -> str:
    """Return a display-safe shell rendering without executing a shell."""

    return " ".join(shlex.quote(str(item)) for item in argv)


def parse_bsub_job_id(*streams: object) -> str:
    """Extract the numeric LSF job ID from one or more bsub output streams."""

    text = "\n".join(str(stream or "") for stream in streams)
    match = re.search(r"Job <([0-9]+)>", text)
    if not match:
        raise ValueError(f"Could not parse bsub job id from output: {text!r}")
    return match.group(1)


def render_dependency(
    dependency: LsfDependency,
    job_ids_by_key: Mapping[str, str] | None = None,
) -> str:
    """Render an LSF dependency with placeholders or resolved job IDs."""

    terms: list[str] = []
    function = dependency.condition.lsf_function
    for job_key in dependency.upstream_job_keys:
        if job_ids_by_key is None:
            job_ref = f"<jobid:{job_key}>"
        else:
            if job_key not in job_ids_by_key:
                raise ValueError(
                    f"No submitted LSF job id is available for job key: {job_key}"
                )
            job_ref = str(job_ids_by_key[job_key])
        terms.append(f"{function}({job_ref})")
    return " && ".join(terms)


def resolve_job_id_placeholders(
    argv: Sequence[str],
    job_ids_by_key: Mapping[str, str],
) -> list[str]:
    """Resolve ``<jobid:key>`` tokens in an already rendered command."""

    out: list[str] = []
    for arg in argv:
        text = str(arg)
        for job_key, job_id in job_ids_by_key.items():
            text = text.replace(f"<jobid:{job_key}>", str(job_id))
        if "<jobid:" in text:
            raise ValueError(f"Unresolved job id placeholder in command argument: {text}")
        out.append(text)
    return out


def build_bsub_prefix(
    *,
    job_name: str,
    resources: LsfResources,
    stdout_path: Path,
    stderr_path: Path,
    dependency_expression: str | None = None,
) -> list[str]:
    """Render the scheduler portion of a bsub command as an argv list."""

    args = [
        "bsub",
        "-J",
        str(job_name),
        "-n",
        str(resources.ncores),
    ]
    if resources.walltime is not None:
        args.extend(["-W", resources.walltime])
    resource_requirement = f"rusage[mem={resources.mem_gb}G]"
    if resources.span_hosts is not None:
        resource_requirement += f" span[hosts={resources.span_hosts}]"
    args.extend(
        [
            "-R",
            resource_requirement,
            "-oo",
            str(stdout_path),
            "-eo",
            str(stderr_path),
        ]
    )
    if resources.queue:
        args.extend(["-q", resources.queue])
    if resources.gpus > 0:
        args.extend(["-gpu", f"num={resources.gpus}"])
    args.extend(resources.extra_lsf_args)
    if dependency_expression:
        args.extend(["-w", dependency_expression])
    return args


def build_bsub_command(
    job: LsfJob,
    job_ids_by_key: Mapping[str, str] | None = None,
) -> list[str]:
    """Render one complete bsub argv list from a structured job model."""

    dependency_expression = (
        render_dependency(job.dependency, job_ids_by_key)
        if job.dependency is not None
        else None
    )
    job_name = job.job_name
    if (
        job.execution_group is not None
        and job.execution_group.mode is LsfExecutionMode.ARRAY
    ):
        task_count = len(job.execution_group.tasks)
        limit = (
            f"%{job.execution_group.max_concurrent}"
            if job.execution_group.max_concurrent < task_count
            else ""
        )
        job_name = f"{job_name}[1-{task_count}]{limit}"
    return [
        *build_bsub_prefix(
            job_name=job_name,
            resources=job.resources,
            stdout_path=job.stdout_path,
            stderr_path=job.stderr_path,
            dependency_expression=dependency_expression,
        ),
        *job.command,
    ]


def run_command(
    argv: Sequence[str],
    *,
    cwd: Path,
    runner: CommandRunner = subprocess.run,
) -> dict[str, Any]:
    """Run one argv command and retain the complete diagnostic result."""

    command = [str(item) for item in argv]
    result = runner(
        command,
        cwd=str(cwd),
        text=True,
        capture_output=True,
    )
    payload = {
        "command": command,
        "cwd": str(cwd),
        "returncode": int(result.returncode),
        "stdout": result.stdout,
        "stderr": result.stderr,
    }
    if result.returncode != 0:
        raise CommandExecutionError(
            (
                f"Command failed with exit code {result.returncode}: {shell_join(argv)}\n"
                f"stdout:\n{result.stdout}\n"
                f"stderr:\n{result.stderr}"
            ),
            result=payload,
        )
    return payload


__all__ = [
    "CommandRunner",
    "CommandExecutionError",
    "CompletedProcessLike",
    "build_bsub_command",
    "build_bsub_prefix",
    "parse_bsub_job_id",
    "render_dependency",
    "resolve_job_id_placeholders",
    "run_command",
    "shell_join",
]
