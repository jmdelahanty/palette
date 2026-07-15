"""Shared LSF envelope builders for composable clipped-recording workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from fisheye.cluster.keypoints.common import safe_component
from fisheye.cluster.lsf import (
    LsfDependency,
    LsfExecutionGroup,
    LsfExecutionMode,
    LsfExecutionTask,
    LsfJob,
    LsfResources,
    shell_join,
)
from fisheye.cluster.lsf.runtime import (
    RUNTIME_JOB_ID_TOKEN,
    RUNTIME_JOB_INDEX_TOKEN,
    build_runtime_command,
)


def chain_commands(commands: Sequence[Sequence[str]]) -> tuple[str, ...]:
    """Run commands serially inside one fail-fast shell envelope."""

    return ("bash", "-lc", " && ".join(shell_join(command) for command in commands))


def build_job(
    *,
    workflow_id: str,
    family: str = "clipped_inference",
    repo: Path,
    run_root: Path,
    job_key: str,
    stage: str,
    command: Sequence[str],
    resources: LsfResources,
    upstream: Sequence[str] = (),
    expected_outputs: Sequence[Path] = (),
    cleanup_paths: Sequence[str] = (),
) -> LsfJob:
    """Build one ordinary LSF job with the shared runtime contract."""

    safe = safe_component(job_key.replace(":", "_"), default="job", max_length=110)
    deployed_pythonpath = str(repo / "src")
    runtime_command = build_runtime_command(
        command,
        status_path_template=run_root / "status" / f"{safe}.{RUNTIME_JOB_ID_TOKEN}.json",
        workflow_id=workflow_id,
        family=family,
        job_key=job_key,
        stage=stage,
        cwd=repo,
        cleanup_path_templates=cleanup_paths,
        expected_output_templates=tuple(str(path) for path in expected_outputs),
        environment_overrides={"PYTHONPATH": deployed_pythonpath},
        python_launcher=(
            "env",
            f"PYTHONPATH={deployed_pythonpath}",
            str(repo / "scripts" / "py"),
        ),
    )
    return LsfJob(
        job_key=job_key,
        job_name=safe_component(f"ci_{workflow_id}_{safe}", default=safe, max_length=120),
        command=runtime_command,
        resources=resources,
        stdout_path=run_root / "logs" / f"{safe}.%J.out",
        stderr_path=run_root / "logs" / f"{safe}.%J.err",
        dependency=LsfDependency(tuple(upstream)) if upstream else None,
        metadata={"stage": stage},
    )


def build_execution_task(
    *,
    run_root: Path,
    task_key: str,
    stage: str,
    command: Sequence[str],
    expected_outputs: Sequence[Path] = (),
    cleanup_paths: Sequence[str] = (),
    array_indexed: bool,
) -> LsfExecutionTask:
    """Build one independently auditable array element or bundle child."""

    safe = safe_component(task_key.replace(":", "_"), default="task", max_length=110)
    suffix = (
        f".{RUNTIME_JOB_ID_TOKEN}.{RUNTIME_JOB_INDEX_TOKEN}.json"
        if array_indexed
        else f".{RUNTIME_JOB_ID_TOKEN}.json"
    )
    return LsfExecutionTask(
        task_key=task_key,
        stage=stage,
        command=tuple(str(value) for value in command),
        status_path=run_root / "status" / f"{safe}{suffix}",
        expected_outputs=tuple(str(path) for path in expected_outputs),
        cleanup_paths=tuple(str(path) for path in cleanup_paths),
        metadata={"stage": stage},
    )


def build_task_group_job(
    *,
    workflow_id: str,
    family: str = "clipped_inference",
    repo: Path,
    run_root: Path,
    job_key: str,
    stage: str,
    tasks: Sequence[LsfExecutionTask],
    mode: LsfExecutionMode,
    max_concurrent: int,
    resources: LsfResources,
    upstream: Sequence[str] = (),
) -> LsfJob:
    """Build one LSF array or bounded in-allocation task bundle."""

    safe = safe_component(job_key.replace(":", "_"), default="job", max_length=110)
    deployed_pythonpath = str(repo / "src")
    task_tuple = tuple(tasks)
    group = LsfExecutionGroup(
        mode=mode,
        tasks=task_tuple,
        max_concurrent=min(int(max_concurrent), len(task_tuple)),
        summary_path=(
            run_root / "status" / f"{safe}.{RUNTIME_JOB_ID_TOKEN}.bundle.json"
            if mode is LsfExecutionMode.BUNDLE
            else None
        ),
    )
    bundle_thread_env: tuple[str, ...] = ()
    if mode is LsfExecutionMode.BUNDLE:
        threads_per_task = max(1, int(resources.ncores) // group.max_concurrent)
        bundle_thread_env = (
            f"OMP_NUM_THREADS={threads_per_task}",
            f"MKL_NUM_THREADS={threads_per_task}",
            f"OPENBLAS_NUM_THREADS={threads_per_task}",
        )
    command = (
        "env",
        f"PYTHONPATH={deployed_pythonpath}",
        *bundle_thread_env,
        str(repo / "scripts" / "py"),
        "-m",
        "fisheye.cluster.lsf.task_group",
        "--plan",
        str(run_root / "lsf_plan.json"),
        "--job-key",
        job_key,
        "--cwd",
        str(repo),
    )
    log_suffix = ".%J.%I" if mode is LsfExecutionMode.ARRAY else ".%J"
    return LsfJob(
        job_key=job_key,
        job_name=safe_component(f"ci_{workflow_id}_{safe}", default=safe, max_length=120),
        command=command,
        resources=resources,
        stdout_path=run_root / "logs" / f"{safe}{log_suffix}.out",
        stderr_path=run_root / "logs" / f"{safe}{log_suffix}.err",
        dependency=LsfDependency(tuple(upstream)) if upstream else None,
        metadata={
            "stage": stage,
            "execution_mode": mode.value,
            "task_count": len(task_tuple),
            "max_concurrent": group.max_concurrent,
            "cores_per_task": (
                max(1, int(resources.ncores) // group.max_concurrent)
                if mode is LsfExecutionMode.BUNDLE
                else int(resources.ncores)
            ),
        },
        execution_group=group,
    )


__all__ = [
    "build_execution_task",
    "build_job",
    "build_task_group_job",
    "chain_commands",
]
