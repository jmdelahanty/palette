"""Composable, registry-aware dataset reporting."""

from .catalog import ANALYSIS_FAMILIES, PROVIDERS, VISUALIZATIONS
from .export import (
    MATERIALIZATION_POLICIES,
    REPORT_EXPORT_SCHEMA_ID,
    export_report_bundle,
    report_manifest_sha256,
    verify_report_manifest_sha256,
)
from .manifest import report_plan_json, report_plan_sha256, report_plan_to_dict
from .montage import SEMANTIC_MONTAGE_SCHEMA_ID, build_semantic_visualization_montages
from .montage_report import (
    SEMANTIC_MONTAGE_ARTIFACT_CONTRACT_ID,
    publish_semantic_montage_report,
)
from .models import (
    AnalysisFamilySpec,
    ArtifactSelector,
    EntityScope,
    PlanStatus,
    ProviderSpec,
    ReportPlan,
    SourceRequirement,
    VisualizationSpec,
)
from .planner import build_report_plan, plan_recording_report
from .report_registry import (
    AnalyticsExportBinding,
    check_report_manifest,
    index_report_manifest,
    query_indexed_reports,
    report_output_dir,
    resolve_analytics_export_binding,
    validate_report_id,
)
from .selection import query_report_recordings

__all__ = [
    "ANALYSIS_FAMILIES",
    "MATERIALIZATION_POLICIES",
    "PROVIDERS",
    "REPORT_EXPORT_SCHEMA_ID",
    "VISUALIZATIONS",
    "AnalysisFamilySpec",
    "AnalyticsExportBinding",
    "ArtifactSelector",
    "EntityScope",
    "PlanStatus",
    "ProviderSpec",
    "ReportPlan",
    "SourceRequirement",
    "SEMANTIC_MONTAGE_SCHEMA_ID",
    "SEMANTIC_MONTAGE_ARTIFACT_CONTRACT_ID",
    "VisualizationSpec",
    "build_report_plan",
    "build_semantic_visualization_montages",
    "check_report_manifest",
    "export_report_bundle",
    "index_report_manifest",
    "plan_recording_report",
    "publish_semantic_montage_report",
    "query_report_recordings",
    "query_indexed_reports",
    "report_output_dir",
    "report_plan_json",
    "report_plan_sha256",
    "report_plan_to_dict",
    "report_manifest_sha256",
    "resolve_analytics_export_binding",
    "validate_report_id",
    "verify_report_manifest_sha256",
]
