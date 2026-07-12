"""Declarative contracts and plan records for dataset reporting."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping


REPORT_PLAN_SCHEMA_ID = "palette.dataset_report_plan.v1"


class PlanStatus(str, Enum):
    """Read-only planner outcomes for one requested visualization."""

    READY = "ready"
    NEEDS_RENDER = "needs_render"
    NEEDS_ANALYSIS = "needs_analysis"
    CONTRACT_MISMATCH = "contract_mismatch"
    NOT_APPLICABLE = "not_applicable"
    BLOCKED_MISSING_SOURCE = "blocked_missing_source"
    ERROR = "error"


class EntityScope(str, Enum):
    """Cardinality axis over which a visualization is expanded."""

    RECORDING = "recording"
    TRACK = "track"
    STIMULUS_STEP = "stimulus_step"
    CHASER = "chaser"


@dataclass(frozen=True)
class SourceRequirement:
    """One upstream source capability satisfied by any listed Zarr group."""

    source_id: str
    any_group_paths: tuple[str, ...]


@dataclass(frozen=True)
class AnalysisFamilySpec:
    """Describe how a logical analysis family is found in a recording Zarr."""

    family_id: str
    label: str
    stage_id: str
    run_parent_paths: tuple[str, ...]
    prerequisites: tuple[str, ...] = ()
    entity_id_attrs: tuple[str, ...] = ()
    source_requirements: tuple[SourceRequirement, ...] = ()


@dataclass(frozen=True)
class ArtifactSelector:
    """Select one run-local artifact by a relative path pattern.

    ``path_pattern`` uses shell-style wildcards and may contain ``{entity_id}``.
    It is evaluated relative to the resolved analysis run.
    """

    path_pattern: str
    artifact_role: str = "snapshot"


@dataclass(frozen=True)
class VisualizationSpec:
    """A stable semantic visualization declaration."""

    visualization_id: str
    label: str
    provider_id: str
    analysis_family_id: str
    selector: ArtifactSelector
    entity_scope: EntityScope = EntityScope.RECORDING
    entity_source_family_id: str | None = None
    visualization_contract_id: str | None = None
    renderer: str | None = None
    renderer_version: str | None = None
    required_by_default: bool = True


@dataclass(frozen=True)
class ProviderSpec:
    """Compose related analyses and plots into a report provider."""

    provider_id: str
    label: str
    visualization_ids: tuple[str, ...]
    always_applicable: bool = False
    stimulus_modes: tuple[str, ...] = ()


@dataclass(frozen=True)
class SelectedRecording:
    dataset_id: str
    recording_id: str
    zarr_path: str
    protocol_name: str | None
    protocol_hash: str | None = None
    arena_id: str | None = None
    recording_started_utc: str | None = None


@dataclass(frozen=True)
class StimulusStep:
    step_index: int
    step_name: str
    stimulus_mode: str
    start_frame: int | None = None
    end_frame: int | None = None
    duration_s: float | None = None


@dataclass(frozen=True)
class ChaserEntity:
    chaser_index: int
    behavior_class: str


@dataclass(frozen=True)
class ResolvedRun:
    family_id: str
    run_id: str
    path: str
    selection: str
    schema_id: str | None = None
    schema_version: int | None = None
    method: str | None = None
    method_version: str | None = None
    source_fingerprint: str | None = None
    lineage_hash: str | None = None
    entity_id: str | None = None


@dataclass(frozen=True)
class ArtifactReference:
    path: str
    artifact_name: str
    artifact_role: str | None
    visualization_contract_id: str | None
    renderer: str | None
    renderer_version: str | None
    artifact_signature: str | None
    content_sha256: str | None


@dataclass(frozen=True)
class ProviderPlan:
    provider_id: str
    label: str
    applicable: bool
    applicability_source: str
    matching_stimulus_modes: tuple[str, ...] = ()


@dataclass(frozen=True)
class VisualizationPlanItem:
    visualization_id: str
    label: str
    provider_id: str
    required: bool
    entity_scope: str
    entity_id: str | None
    status: str
    reason: str
    expected_visualization_contract_id: str | None
    expected_renderer: str | None
    expected_renderer_version: str | None
    source_run: ResolvedRun | None = None
    artifact: ArtifactReference | None = None
    proposed_actions: tuple[str, ...] = ()


@dataclass(frozen=True)
class RecordingReportPlan:
    recording: SelectedRecording
    stimulus_run: ResolvedRun | None
    stimulus_steps: tuple[StimulusStep, ...]
    stimulus_modes: tuple[str, ...]
    track_ids: tuple[str, ...]
    chasers: tuple[ChaserEntity, ...]
    providers: tuple[ProviderPlan, ...]
    items: tuple[VisualizationPlanItem, ...]
    errors: tuple[str, ...] = ()


@dataclass(frozen=True)
class ReportPlan:
    schema_id: str
    schema_version: int
    created_at_utc: str
    tool: str
    registry_path: str
    query: Mapping[str, Any]
    requested_provider_ids: tuple[str, ...]
    recordings: tuple[RecordingReportPlan, ...]
    status_counts: Mapping[str, int] = field(default_factory=dict)
