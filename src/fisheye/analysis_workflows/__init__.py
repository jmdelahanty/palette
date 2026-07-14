"""Declarative planning and fail-closed execution for analysis workflows."""

from .availability import (
    StageAvailability,
    discover_stage_availability,
    stage_run_relative_path,
)
from .contracts import (
    ANALYSIS_WORKFLOW_SCHEMA_ID,
    ANALYSIS_WORKFLOW_SCHEMA_VERSION,
    AnalysisWorkflow,
    TemporalPolicy,
    WorkflowNode,
    default_core_behavior_profile_path,
    load_analysis_workflow,
)
from .dag import NodePlan, WorkflowPlan, plan_analysis_workflow
from .execution import (
    EXECUTION_SCHEMA_ID,
    EXECUTION_SCHEMA_VERSION,
    StageCommand,
    WorkflowExecutionError,
    WorkflowExecutionPlan,
    build_workflow_execution_plan,
)

__all__ = [
    "ANALYSIS_WORKFLOW_SCHEMA_ID",
    "ANALYSIS_WORKFLOW_SCHEMA_VERSION",
    "EXECUTION_SCHEMA_ID",
    "EXECUTION_SCHEMA_VERSION",
    "AnalysisWorkflow",
    "NodePlan",
    "StageAvailability",
    "StageCommand",
    "TemporalPolicy",
    "WorkflowNode",
    "WorkflowExecutionError",
    "WorkflowExecutionPlan",
    "WorkflowPlan",
    "build_workflow_execution_plan",
    "default_core_behavior_profile_path",
    "discover_stage_availability",
    "load_analysis_workflow",
    "plan_analysis_workflow",
    "stage_run_relative_path",
]
