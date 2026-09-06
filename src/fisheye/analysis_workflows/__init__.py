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
    EXECUTION_LEGACY_SCHEMA_VERSION,
    EXECUTION_SCHEMA_ID,
    EXECUTION_SCHEMA_VERSION,
    StageAdmissionCommand,
    StageCommand,
    WorkflowExecutionError,
    WorkflowExecutionPlan,
    build_workflow_execution_plan,
)
from .execution_profiles import (
    PRODUCTION_EXECUTION_PROFILE_ID,
    SELECTOR_INELIGIBLE_CANARY_EXECUTION_PROFILE_ID,
    WorkflowExecutionProfile,
    resolve_workflow_execution_profile,
    workflow_execution_profile_ids,
)

__all__ = [
    "ANALYSIS_WORKFLOW_SCHEMA_ID",
    "ANALYSIS_WORKFLOW_SCHEMA_VERSION",
    "EXECUTION_SCHEMA_ID",
    "EXECUTION_LEGACY_SCHEMA_VERSION",
    "EXECUTION_SCHEMA_VERSION",
    "PRODUCTION_EXECUTION_PROFILE_ID",
    "SELECTOR_INELIGIBLE_CANARY_EXECUTION_PROFILE_ID",
    "WorkflowExecutionProfile",
    "resolve_workflow_execution_profile",
    "workflow_execution_profile_ids",
    "AnalysisWorkflow",
    "NodePlan",
    "StageAvailability",
    "StageAdmissionCommand",
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
