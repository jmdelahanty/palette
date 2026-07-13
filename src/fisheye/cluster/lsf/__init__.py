"""Small shared kernel for planning and submitting LSF jobs.

Stage-specific planners own target discovery, commands, run names, artifacts,
and semantic validation.  This package owns only the reusable LSF mechanics.
"""

from fisheye.cluster.lsf.backend import (
    CommandExecutionError,
    CommandRunner,
    build_bsub_command,
    build_bsub_prefix,
    parse_bsub_job_id,
    render_dependency,
    resolve_job_id_placeholders,
    run_command,
    shell_join,
)
from fisheye.cluster.lsf.bundle import write_json_snapshot
from fisheye.cluster.lsf.models import (
    LsfDependency,
    LsfDependencyCondition,
    LsfJob,
    LsfResources,
    LsfWorkflow,
    LsfWorkflowFragment,
    compose_lsf_workflow,
)
from fisheye.cluster.lsf.submission import (
    JobSubmittedCallback,
    LSF_SUBMISSION_SCHEMA,
    submit_lsf_workflow,
)

__all__ = [
    "CommandRunner",
    "CommandExecutionError",
    "LsfDependency",
    "LsfDependencyCondition",
    "LsfJob",
    "LsfResources",
    "LsfWorkflow",
    "LsfWorkflowFragment",
    "JobSubmittedCallback",
    "LSF_SUBMISSION_SCHEMA",
    "build_bsub_command",
    "compose_lsf_workflow",
    "build_bsub_prefix",
    "parse_bsub_job_id",
    "render_dependency",
    "resolve_job_id_placeholders",
    "run_command",
    "shell_join",
    "submit_lsf_workflow",
    "write_json_snapshot",
]
