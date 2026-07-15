"""Immutable, stage-agnostic models for LSF submission planning."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping


class LsfDependencyCondition(str, Enum):
    """Supported LSF dependency conditions.

    Production dependencies should normally use :attr:`ALL_SUCCEEDED`.  The
    ``ended`` condition is retained for explicitly designed cleanup or recovery
    jobs that must run after an upstream failure.
    """

    ALL_SUCCEEDED = "all_succeeded"
    ALL_ENDED = "all_ended"

    @property
    def lsf_function(self) -> str:
        if self is LsfDependencyCondition.ALL_SUCCEEDED:
            return "done"
        return "ended"


class LsfExecutionMode(str, Enum):
    """How a scheduler allocation dispatches its declared execution tasks."""

    ARRAY = "array"
    BUNDLE = "bundle"


@dataclass(frozen=True)
class LsfResources:
    """Resources and scheduler options for one LSF job.

    ``mem_gb`` is rendered directly as LSF ``rusage[mem=<value>G]`` and
    therefore retains the cluster's per-slot accounting semantics.
    """

    queue: str
    ncores: int
    mem_gb: int
    gpus: int = 0
    walltime: str | None = None
    span_hosts: int | None = None
    extra_lsf_args: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "queue", str(self.queue or ""))
        object.__setattr__(self, "ncores", int(self.ncores))
        object.__setattr__(self, "mem_gb", int(self.mem_gb))
        object.__setattr__(self, "gpus", int(self.gpus))
        object.__setattr__(
            self,
            "walltime",
            str(self.walltime).strip() if self.walltime is not None else None,
        )
        object.__setattr__(
            self,
            "span_hosts",
            int(self.span_hosts) if self.span_hosts is not None else None,
        )
        object.__setattr__(
            self,
            "extra_lsf_args",
            tuple(str(value) for value in self.extra_lsf_args),
        )
        if self.ncores <= 0:
            raise ValueError("LSF ncores must be positive.")
        if self.mem_gb <= 0:
            raise ValueError("LSF mem_gb must be positive.")
        if self.gpus < 0:
            raise ValueError("LSF gpus cannot be negative.")
        if self.walltime == "":
            raise ValueError("LSF walltime cannot be empty when provided.")
        if self.span_hosts is not None and self.span_hosts <= 0:
            raise ValueError("LSF span_hosts must be positive when provided.")

    def to_json(self) -> dict[str, Any]:
        return {
            "queue": self.queue or None,
            "ncores": self.ncores,
            "mem_gb": self.mem_gb,
            "gpus": self.gpus,
            "walltime": self.walltime,
            "span_hosts": self.span_hosts,
            "extra_lsf_args": list(self.extra_lsf_args),
        }


@dataclass(frozen=True)
class LsfDependency:
    """A structured dependency on one or more upstream job keys."""

    upstream_job_keys: tuple[str, ...]
    condition: LsfDependencyCondition = LsfDependencyCondition.ALL_SUCCEEDED

    def __post_init__(self) -> None:
        keys = tuple(str(value).strip() for value in self.upstream_job_keys)
        object.__setattr__(self, "upstream_job_keys", keys)
        if not isinstance(self.condition, LsfDependencyCondition):
            object.__setattr__(
                self,
                "condition",
                LsfDependencyCondition(self.condition),
            )
        if not keys:
            raise ValueError("An LSF dependency requires at least one upstream job key.")
        if any(not key for key in keys):
            raise ValueError("LSF dependency job keys cannot be empty.")
        if len(set(keys)) != len(keys):
            raise ValueError("LSF dependency job keys must be unique.")
        if any("<" in key or ">" in key for key in keys):
            raise ValueError("LSF dependency job keys cannot contain '<' or '>'.")

    def to_json(self) -> dict[str, Any]:
        return {
            "upstream_job_keys": list(self.upstream_job_keys),
            "condition": self.condition.value,
        }


@dataclass(frozen=True)
class LsfExecutionTask:
    """One independently auditable task inside an LSF array or CPU bundle."""

    task_key: str
    stage: str
    command: tuple[str, ...]
    status_path: Path
    expected_outputs: tuple[str, ...] = ()
    cleanup_paths: tuple[str, ...] = ()
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        task_key = str(self.task_key).strip()
        stage = str(self.stage).strip()
        command = tuple(str(value) for value in self.command)
        object.__setattr__(self, "task_key", task_key)
        object.__setattr__(self, "stage", stage)
        object.__setattr__(self, "command", command)
        object.__setattr__(self, "status_path", Path(self.status_path))
        object.__setattr__(
            self,
            "expected_outputs",
            tuple(str(value) for value in self.expected_outputs),
        )
        object.__setattr__(
            self,
            "cleanup_paths",
            tuple(str(value) for value in self.cleanup_paths),
        )
        if not task_key:
            raise ValueError("LSF execution task_key cannot be empty.")
        if "<" in task_key or ">" in task_key:
            raise ValueError("LSF execution task_key cannot contain '<' or '>'.")
        if not stage:
            raise ValueError("LSF execution task stage cannot be empty.")
        if not command:
            raise ValueError("An LSF execution task command cannot be empty.")

    def to_json(self) -> dict[str, Any]:
        return {
            "task_key": self.task_key,
            "stage": self.stage,
            "command": list(self.command),
            "status_path": str(self.status_path),
            "expected_outputs": list(self.expected_outputs),
            "cleanup_paths": list(self.cleanup_paths),
            "metadata": dict(self.metadata) if self.metadata is not None else None,
        }


@dataclass(frozen=True)
class LsfExecutionGroup:
    """Tasks dispatched as one LSF array or one bounded in-allocation bundle."""

    mode: LsfExecutionMode
    tasks: tuple[LsfExecutionTask, ...]
    max_concurrent: int
    summary_path: Path | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.mode, LsfExecutionMode):
            object.__setattr__(self, "mode", LsfExecutionMode(self.mode))
        tasks = tuple(self.tasks)
        object.__setattr__(self, "tasks", tasks)
        object.__setattr__(self, "max_concurrent", int(self.max_concurrent))
        if self.summary_path is not None:
            object.__setattr__(self, "summary_path", Path(self.summary_path))
        if not tasks:
            raise ValueError("An LSF execution group requires at least one task.")
        if self.max_concurrent <= 0:
            raise ValueError("LSF execution-group max_concurrent must be positive.")
        if self.max_concurrent > len(tasks):
            raise ValueError(
                "LSF execution-group max_concurrent cannot exceed its task count."
            )
        task_keys = [task.task_key for task in tasks]
        if len(set(task_keys)) != len(task_keys):
            raise ValueError("LSF execution-group task keys must be unique.")
        if self.mode is LsfExecutionMode.BUNDLE and self.summary_path is None:
            raise ValueError("An LSF bundle execution group requires a summary_path.")

    def to_json(self) -> dict[str, Any]:
        return {
            "mode": self.mode.value,
            "task_count": len(self.tasks),
            "max_concurrent": self.max_concurrent,
            "summary_path": str(self.summary_path) if self.summary_path else None,
            "tasks": [task.to_json() for task in self.tasks],
        }


@dataclass(frozen=True)
class LsfJob:
    """One stage-agnostic LSF job ready for backend rendering."""

    job_key: str
    job_name: str
    command: tuple[str, ...]
    resources: LsfResources
    stdout_path: Path
    stderr_path: Path
    dependency: LsfDependency | None = None
    metadata: Mapping[str, Any] | None = None
    execution_group: LsfExecutionGroup | None = None

    def __post_init__(self) -> None:
        job_key = str(self.job_key).strip()
        job_name = str(self.job_name).strip()
        command = tuple(str(value) for value in self.command)
        object.__setattr__(self, "job_key", job_key)
        object.__setattr__(self, "job_name", job_name)
        object.__setattr__(self, "command", command)
        object.__setattr__(self, "stdout_path", Path(self.stdout_path))
        object.__setattr__(self, "stderr_path", Path(self.stderr_path))
        if not job_key:
            raise ValueError("LSF job_key cannot be empty.")
        if "<" in job_key or ">" in job_key:
            raise ValueError("LSF job_key cannot contain '<' or '>'.")
        if not job_name:
            raise ValueError("LSF job_name cannot be empty.")
        if not command:
            raise ValueError("An LSF job command cannot be empty.")
        if self.execution_group is not None:
            if not isinstance(self.execution_group, LsfExecutionGroup):
                raise TypeError("LSF execution_group must be an LsfExecutionGroup.")
            if self.execution_group.mode is LsfExecutionMode.ARRAY:
                if "%I" not in str(self.stdout_path) or "%I" not in str(self.stderr_path):
                    raise ValueError(
                        "LSF array stdout/stderr paths must include the %I array index token."
                    )

    def to_json(self) -> dict[str, Any]:
        return {
            "job_key": self.job_key,
            "job_name": self.job_name,
            "command": list(self.command),
            "resources": self.resources.to_json(),
            "stdout_path": str(self.stdout_path),
            "stderr_path": str(self.stderr_path),
            "dependency": self.dependency.to_json() if self.dependency else None,
            "metadata": dict(self.metadata) if self.metadata is not None else None,
            "execution_group": (
                self.execution_group.to_json() if self.execution_group else None
            ),
        }


@dataclass(frozen=True)
class LsfWorkflow:
    """A validated LSF job graph owned by one family-specific planner."""

    workflow_id: str
    family: str
    jobs: tuple[LsfJob, ...]
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        workflow_id = str(self.workflow_id).strip()
        family = str(self.family).strip()
        jobs = tuple(self.jobs)
        object.__setattr__(self, "workflow_id", workflow_id)
        object.__setattr__(self, "family", family)
        object.__setattr__(self, "jobs", jobs)
        if not workflow_id:
            raise ValueError("LSF workflow_id cannot be empty.")
        if not family:
            raise ValueError("LSF workflow family cannot be empty.")
        if not jobs:
            raise ValueError("An LSF workflow requires at least one job.")

        job_keys = [job.job_key for job in jobs]
        if len(set(job_keys)) != len(job_keys):
            raise ValueError("LSF workflow job keys must be unique.")
        job_names = [job.job_name for job in jobs]
        if len(set(job_names)) != len(job_names):
            raise ValueError("LSF workflow job names must be unique.")

        known_keys = set(job_keys)
        for job in jobs:
            if job.dependency is None:
                continue
            for upstream_key in job.dependency.upstream_job_keys:
                if upstream_key == job.job_key:
                    raise ValueError(
                        f"LSF job {job.job_key!r} cannot depend on itself."
                    )
                if upstream_key not in known_keys:
                    raise ValueError(
                        f"LSF job {job.job_key!r} depends on unknown job key "
                        f"{upstream_key!r}."
                    )
        self.topological_jobs()

    def topological_jobs(self) -> tuple[LsfJob, ...]:
        """Return a stable dependency order or raise when the graph is cyclic."""

        pending = list(self.jobs)
        ordered: list[LsfJob] = []
        submitted_keys: set[str] = set()
        while pending:
            ready = [
                job
                for job in pending
                if job.dependency is None
                or set(job.dependency.upstream_job_keys).issubset(submitted_keys)
            ]
            if not ready:
                blocked = ", ".join(job.job_key for job in pending)
                raise ValueError(
                    "LSF workflow dependency graph contains a cycle; "
                    f"blocked jobs: {blocked}"
                )
            for job in ready:
                ordered.append(job)
                submitted_keys.add(job.job_key)
                pending.remove(job)
        return tuple(ordered)

    def to_json(self) -> dict[str, Any]:
        ordered = self.topological_jobs()
        return {
            "schema": "palette.lsf_workflow.v1",
            "workflow_id": self.workflow_id,
            "family": self.family,
            "jobs": [job.to_json() for job in self.jobs],
            "submission_order": [job.job_key for job in ordered],
            "metadata": dict(self.metadata) if self.metadata is not None else None,
        }


@dataclass(frozen=True)
class LsfWorkflowFragment:
    """A reusable set of jobs that can be composed into an LSF workflow.

    Dependencies may point outside the fragment.  They are validated after all
    selected fragments are composed into the final :class:`LsfWorkflow`.
    """

    fragment_id: str
    jobs: tuple[LsfJob, ...]
    requires: tuple[str, ...] = ()
    provides: tuple[str, ...] = ()
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        fragment_id = str(self.fragment_id).strip()
        jobs = tuple(self.jobs)
        object.__setattr__(self, "fragment_id", fragment_id)
        object.__setattr__(self, "jobs", jobs)
        object.__setattr__(self, "requires", tuple(self.requires))
        object.__setattr__(self, "provides", tuple(self.provides))
        if not fragment_id:
            raise ValueError("LSF workflow fragment_id cannot be empty.")
        job_keys = [job.job_key for job in jobs]
        if len(set(job_keys)) != len(job_keys):
            raise ValueError(
                f"LSF workflow fragment {fragment_id!r} has duplicate job keys."
            )
        if len(set(self.requires)) != len(self.requires):
            raise ValueError(f"Fragment {fragment_id!r} requirements must be unique.")
        if len(set(self.provides)) != len(self.provides):
            raise ValueError(f"Fragment {fragment_id!r} outputs must be unique.")
        if any(not str(value).strip() for value in (*self.requires, *self.provides)):
            raise ValueError("Fragment requirements and outputs cannot be empty.")

    def to_json(self) -> dict[str, Any]:
        return {
            "fragment_id": self.fragment_id,
            "job_count": len(self.jobs),
            "job_keys": [job.job_key for job in self.jobs],
            "requires": list(self.requires),
            "provides": list(self.provides),
            "metadata": dict(self.metadata) if self.metadata is not None else None,
        }


def compose_lsf_workflow(
    *,
    workflow_id: str,
    family: str,
    fragments: tuple[LsfWorkflowFragment, ...],
    external_inputs: tuple[str, ...] = (),
    metadata: Mapping[str, Any] | None = None,
) -> LsfWorkflow:
    """Compose independently built fragments and validate the resulting DAG."""

    selected = tuple(fragments)
    fragment_ids = [fragment.fragment_id for fragment in selected]
    if len(set(fragment_ids)) != len(fragment_ids):
        raise ValueError("Composed LSF workflow fragment ids must be unique.")
    supplied = set(str(value) for value in external_inputs)
    for fragment in selected:
        missing = set(fragment.requires) - supplied
        if missing:
            raise ValueError(
                f"Composed LSF workflow fragment {fragment.fragment_id!r} is "
                "missing required fragment inputs: "
                + ", ".join(sorted(missing))
            )
        supplied.update(fragment.provides)
    jobs = tuple(job for fragment in selected for job in fragment.jobs)
    if not jobs:
        raise ValueError("A composed LSF workflow requires at least one job.")
    composed_metadata = dict(metadata or {})
    composed_metadata["fragments"] = [fragment.to_json() for fragment in selected]
    composed_metadata["external_inputs"] = list(external_inputs)
    return LsfWorkflow(
        workflow_id=workflow_id,
        family=family,
        jobs=jobs,
        metadata=composed_metadata,
    )


__all__ = [
    "LsfDependency",
    "LsfDependencyCondition",
    "LsfExecutionGroup",
    "LsfExecutionMode",
    "LsfExecutionTask",
    "LsfJob",
    "LsfResources",
    "LsfWorkflow",
    "LsfWorkflowFragment",
    "compose_lsf_workflow",
]
