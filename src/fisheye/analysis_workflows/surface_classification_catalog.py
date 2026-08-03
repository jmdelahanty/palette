"""Closed coverage catalog for analytics surfaces outside the storage catalog.

This module classifies current and historical analytics outputs without
claiming that they have adopted the derived-analysis storage contract,
byte-budget planner, registry projection, or atomic publication.  Scientific
run families enter ``storage_contract_catalog`` only after those stronger
contracts exist; this catalog makes the remaining migration boundary
executable in the meantime.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from importlib import import_module
import re
from string import Formatter
from typing import Any


class AnalyticsSurfaceClass(str, Enum):
    """Scientific and operational roles allowed by the coverage census."""

    MAINTAINED_SCIENTIFIC_AUTHORITY = "maintained_scientific_authority"
    EMBEDDED_COMPONENT = "embedded_component"
    VISUALIZATION_CACHE = "visualization_cache"
    EXPORT = "export"
    MAINTENANCE_OUTPUT = "maintenance_output"
    LEGACY = "legacy"


class AnalyticsSurfaceLifecycle(str, Enum):
    """Whether new maintained workflows still own the surface."""

    CURRENT = "current"
    CURRENT_LEGACY_SHAPED = "current_legacy_shaped"
    LEGACY = "legacy"


class AnalyticsStorageKind(str, Enum):
    """Physical namespace containing the output."""

    ZARR = "zarr"
    FILESYSTEM = "filesystem"


class AnalyticsMutationMode(str, Enum):
    """How the current owner changes persisted state."""

    IMMUTABLE_RUN = "immutable_run"
    EMBEDDED_RUN = "embedded_run"
    ARTIFACT_RUN = "artifact_run"
    FILESYSTEM_GENERATION = "filesystem_generation"
    IN_PLACE_MUTATION = "in_place_mutation"


class ExactStorageContractStatus(str, Enum):
    """Disposition of an exact array-storage contract for the surface."""

    REQUIRED = "required"
    NOT_APPLICABLE = "not_applicable"
    LEGACY_COMPATIBILITY_ONLY = "legacy_compatibility_only"


class AnalyticsLifecycleContext(str, Enum):
    """Exact persistence context that disambiguates similarly named surfaces."""

    TOP_LEVEL_RUN = "top_level_run"
    CHASER_EMBEDDED_COMPONENT = "chaser_embedded_component"
    POLICY_EVIDENCE_RUN = "policy_evidence_run"
    RECORDING_WIDE_QUALITY_SNAPSHOT = "recording_wide_quality_snapshot"
    SOURCE_DETECT_NESTED_QUALITY_REPORT = "source_detect_nested_quality_report"
    DIAGNOSTIC_EXPORT = "diagnostic_export"
    LEGACY_STANDALONE_RUN = "legacy_standalone_run"
    ACTIVE_LEGACY_NAMESPACE_COLLISION = "active_legacy_namespace_collision"
    IN_PLACE_COMPATIBILITY_REPAIR = "in_place_compatibility_repair"
    TRACK_BOUND_VISUALIZATION = "track_bound_visualization"


class AnalyticsConsumerScope(str, Enum):
    """Exact consumer scope for one classified persistence surface."""

    RECORDING_ANALYSIS = "recording_analysis"
    CHASER_COMPONENT_READER = "chaser_component_reader"
    REGISTERED_DETECTION_GATE = "registered_detection_gate"
    RECORDING_WIDE_DETECTION_QUALITY = "recording_wide_detection_quality"
    SOURCE_DETECT_LOCAL_QUALITY = "source_detect_local_quality"
    OPERATOR_DIAGNOSTIC = "operator_diagnostic"
    LEGACY_COMPATIBILITY = "legacy_compatibility"
    SHARED_SWIM_BOUT_SELECTOR = "shared_swim_bout_selector"
    STIMULUS_REPAIR = "stimulus_repair"
    TRACK_PRESENTATION = "track_presentation"


_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_]*$")
_ENTRYPOINT = re.compile(r"^_?[a-z][a-z0-9_]*$")
_MODULE = re.compile(r"^fisheye(?:\.[a-z][a-z0-9_]*)+$")
_SCHEMA_ID = re.compile(r"^palette\.[a-z0-9]+(?:[._][a-z0-9]+)*$")


@dataclass(frozen=True)
class AnalyticsSurfaceClassification:
    """One truthful classification of a persisted analytics surface."""

    surface_id: str
    classification: AnalyticsSurfaceClass
    lifecycle: AnalyticsSurfaceLifecycle
    lifecycle_context: AnalyticsLifecycleContext
    consumer_scope: AnalyticsConsumerScope
    stage_binding: str
    owner_module: str
    owner_entrypoint: str
    owner_path: str
    storage_kind: AnalyticsStorageKind
    mutation_mode: AnalyticsMutationMode
    array_bearing: bool
    exact_storage_contract_status: ExactStorageContractStatus
    central_storage_catalog_required: bool
    schema_id: str | None = None
    schema_version: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.surface_id, str) or not _IDENTIFIER.fullmatch(
            self.surface_id
        ):
            raise ValueError("surface_id must be one canonical identifier")
        for field, expected in (
            ("classification", AnalyticsSurfaceClass),
            ("lifecycle", AnalyticsSurfaceLifecycle),
            ("lifecycle_context", AnalyticsLifecycleContext),
            ("consumer_scope", AnalyticsConsumerScope),
            ("storage_kind", AnalyticsStorageKind),
            ("mutation_mode", AnalyticsMutationMode),
            ("exact_storage_contract_status", ExactStorageContractStatus),
        ):
            if not isinstance(getattr(self, field), expected):
                raise TypeError(f"{field} must use its closed enum")
        for field in (
            "array_bearing",
            "central_storage_catalog_required",
        ):
            if type(getattr(self, field)) is not bool:
                raise TypeError(f"{field} must be an exact bool")
        if not _IDENTIFIER.fullmatch(self.stage_binding):
            raise ValueError("stage_binding must be one canonical identifier")
        if not isinstance(self.owner_module, str) or not _MODULE.fullmatch(
            self.owner_module
        ):
            raise ValueError("owner_module must be one exact fisheye module")
        if not _ENTRYPOINT.fullmatch(self.owner_entrypoint):
            raise ValueError("owner_entrypoint must be one canonical identifier")
        if (
            not self.owner_path
            or self.owner_path != self.owner_path.strip()
            or self.owner_path.startswith("/")
            or self.owner_path.endswith("/")
            or "\\" in self.owner_path
            or any(char.isspace() or ord(char) == 127 for char in self.owner_path)
            or any(part in {"", ".", ".."} for part in self.owner_path.split("/"))
        ):
            raise ValueError("owner_path must be one canonical path pattern")
        try:
            parsed_path = tuple(Formatter().parse(self.owner_path))
        except ValueError as exc:
            raise ValueError("owner_path has malformed template braces") from exc
        for _literal, field_name, format_spec, conversion in parsed_path:
            if field_name is None:
                continue
            if (
                not _IDENTIFIER.fullmatch(field_name)
                or format_spec
                or conversion is not None
            ):
                raise ValueError("owner_path has a noncanonical template field")
        if (self.schema_id is None) != (self.schema_version is None):
            raise ValueError("schema_id and schema_version must be declared together")
        if self.schema_id is not None and (
            not isinstance(self.schema_id, str)
            or not _SCHEMA_ID.fullmatch(self.schema_id)
            or type(self.schema_version) is not int
            or self.schema_version <= 0
        ):
            raise ValueError("schema identity must be one positive Palette version")
        if self.central_storage_catalog_required and (
            self.exact_storage_contract_status
            is not ExactStorageContractStatus.REQUIRED
            or not self.array_bearing
            or self.classification
            is not AnalyticsSurfaceClass.MAINTAINED_SCIENTIFIC_AUTHORITY
            or self.lifecycle is not AnalyticsSurfaceLifecycle.CURRENT
        ):
            raise ValueError(
                "central catalog adoption is only valid for current array-bearing "
                "scientific authorities requiring an exact storage contract"
            )
        if (
            self.classification
            in {
                AnalyticsSurfaceClass.MAINTAINED_SCIENTIFIC_AUTHORITY,
                AnalyticsSurfaceClass.EMBEDDED_COMPONENT,
            }
            and self.lifecycle is not AnalyticsSurfaceLifecycle.CURRENT
        ):
            raise ValueError("scientific authorities and components must be current")
        if (
            self.classification is AnalyticsSurfaceClass.LEGACY
            and self.lifecycle
            not in {
                AnalyticsSurfaceLifecycle.LEGACY,
                AnalyticsSurfaceLifecycle.CURRENT_LEGACY_SHAPED,
            }
        ):
            raise ValueError(
                "legacy-shaped surfaces require an explicit legacy lifecycle"
            )
        if self.lifecycle is AnalyticsSurfaceLifecycle.CURRENT_LEGACY_SHAPED and (
            self.classification is not AnalyticsSurfaceClass.LEGACY
            or self.lifecycle_context
            is not AnalyticsLifecycleContext.ACTIVE_LEGACY_NAMESPACE_COLLISION
            or self.exact_storage_contract_status
            is not ExactStorageContractStatus.REQUIRED
        ):
            raise ValueError(
                "current legacy-shaped surfaces must be active namespace collisions "
                "requiring an exact contract"
            )
        if self.lifecycle is AnalyticsSurfaceLifecycle.CURRENT and (
            self.exact_storage_contract_status
            is ExactStorageContractStatus.LEGACY_COMPATIBILITY_ONLY
        ):
            raise ValueError("current surfaces cannot be legacy compatibility-only")
        if (
            self.exact_storage_contract_status
            is ExactStorageContractStatus.LEGACY_COMPATIBILITY_ONLY
            and self.lifecycle is not AnalyticsSurfaceLifecycle.LEGACY
        ):
            raise ValueError(
                "legacy compatibility-only contracts require the legacy lifecycle"
            )
        if (
            self.exact_storage_contract_status is ExactStorageContractStatus.REQUIRED
            and (
                not self.array_bearing
                or self.lifecycle
                not in {
                    AnalyticsSurfaceLifecycle.CURRENT,
                    AnalyticsSurfaceLifecycle.CURRENT_LEGACY_SHAPED,
                }
                or self.classification
                not in {
                    AnalyticsSurfaceClass.MAINTAINED_SCIENTIFIC_AUTHORITY,
                    AnalyticsSurfaceClass.EMBEDDED_COMPONENT,
                    AnalyticsSurfaceClass.LEGACY,
                }
            )
        ):
            raise ValueError(
                "required exact contracts are reserved for current array-bearing "
                "scientific authorities, embedded components, and active legacy "
                "namespace collisions"
            )
        if (
            self.lifecycle is AnalyticsSurfaceLifecycle.CURRENT
            and self.classification
            is AnalyticsSurfaceClass.MAINTAINED_SCIENTIFIC_AUTHORITY
            and self.array_bearing
            and self.exact_storage_contract_status
            is ExactStorageContractStatus.NOT_APPLICABLE
        ):
            raise ValueError(
                "current array-bearing scientific authorities require an exact "
                "storage-contract disposition"
            )
        if self.mutation_mode is AnalyticsMutationMode.IN_PLACE_MUTATION and (
            self.classification is not AnalyticsSurfaceClass.MAINTENANCE_OUTPUT
        ):
            raise ValueError("in-place mutation is reserved for maintenance outputs")

        expected_context_state = {
            AnalyticsLifecycleContext.TOP_LEVEL_RUN: (
                AnalyticsSurfaceClass.MAINTAINED_SCIENTIFIC_AUTHORITY,
                AnalyticsSurfaceLifecycle.CURRENT,
                AnalyticsConsumerScope.RECORDING_ANALYSIS,
                AnalyticsStorageKind.ZARR,
                AnalyticsMutationMode.IMMUTABLE_RUN,
                True,
            ),
            AnalyticsLifecycleContext.CHASER_EMBEDDED_COMPONENT: (
                AnalyticsSurfaceClass.EMBEDDED_COMPONENT,
                AnalyticsSurfaceLifecycle.CURRENT,
                AnalyticsConsumerScope.CHASER_COMPONENT_READER,
                AnalyticsStorageKind.ZARR,
                AnalyticsMutationMode.EMBEDDED_RUN,
                True,
            ),
            AnalyticsLifecycleContext.POLICY_EVIDENCE_RUN: (
                AnalyticsSurfaceClass.MAINTAINED_SCIENTIFIC_AUTHORITY,
                AnalyticsSurfaceLifecycle.CURRENT,
                AnalyticsConsumerScope.REGISTERED_DETECTION_GATE,
                AnalyticsStorageKind.ZARR,
                AnalyticsMutationMode.IMMUTABLE_RUN,
                True,
            ),
            AnalyticsLifecycleContext.RECORDING_WIDE_QUALITY_SNAPSHOT: (
                AnalyticsSurfaceClass.MAINTAINED_SCIENTIFIC_AUTHORITY,
                AnalyticsSurfaceLifecycle.CURRENT,
                AnalyticsConsumerScope.RECORDING_WIDE_DETECTION_QUALITY,
                AnalyticsStorageKind.ZARR,
                AnalyticsMutationMode.IMMUTABLE_RUN,
                True,
            ),
            AnalyticsLifecycleContext.SOURCE_DETECT_NESTED_QUALITY_REPORT: (
                AnalyticsSurfaceClass.MAINTAINED_SCIENTIFIC_AUTHORITY,
                AnalyticsSurfaceLifecycle.CURRENT,
                AnalyticsConsumerScope.SOURCE_DETECT_LOCAL_QUALITY,
                AnalyticsStorageKind.ZARR,
                AnalyticsMutationMode.IMMUTABLE_RUN,
                True,
            ),
            AnalyticsLifecycleContext.DIAGNOSTIC_EXPORT: (
                AnalyticsSurfaceClass.EXPORT,
                AnalyticsSurfaceLifecycle.CURRENT,
                AnalyticsConsumerScope.OPERATOR_DIAGNOSTIC,
                AnalyticsStorageKind.FILESYSTEM,
                AnalyticsMutationMode.FILESYSTEM_GENERATION,
                False,
            ),
            AnalyticsLifecycleContext.LEGACY_STANDALONE_RUN: (
                AnalyticsSurfaceClass.LEGACY,
                AnalyticsSurfaceLifecycle.LEGACY,
                AnalyticsConsumerScope.LEGACY_COMPATIBILITY,
                AnalyticsStorageKind.ZARR,
                AnalyticsMutationMode.IMMUTABLE_RUN,
                True,
            ),
            AnalyticsLifecycleContext.ACTIVE_LEGACY_NAMESPACE_COLLISION: (
                AnalyticsSurfaceClass.LEGACY,
                AnalyticsSurfaceLifecycle.CURRENT_LEGACY_SHAPED,
                AnalyticsConsumerScope.SHARED_SWIM_BOUT_SELECTOR,
                AnalyticsStorageKind.ZARR,
                AnalyticsMutationMode.IMMUTABLE_RUN,
                True,
            ),
            AnalyticsLifecycleContext.IN_PLACE_COMPATIBILITY_REPAIR: (
                AnalyticsSurfaceClass.MAINTENANCE_OUTPUT,
                AnalyticsSurfaceLifecycle.LEGACY,
                AnalyticsConsumerScope.STIMULUS_REPAIR,
                AnalyticsStorageKind.ZARR,
                AnalyticsMutationMode.IN_PLACE_MUTATION,
                True,
            ),
            AnalyticsLifecycleContext.TRACK_BOUND_VISUALIZATION: (
                AnalyticsSurfaceClass.VISUALIZATION_CACHE,
                AnalyticsSurfaceLifecycle.CURRENT,
                AnalyticsConsumerScope.TRACK_PRESENTATION,
                AnalyticsStorageKind.ZARR,
                AnalyticsMutationMode.ARTIFACT_RUN,
                True,
            ),
        }
        actual_context_state = (
            self.classification,
            self.lifecycle,
            self.consumer_scope,
            self.storage_kind,
            self.mutation_mode,
            self.array_bearing,
        )
        if actual_context_state != expected_context_state[self.lifecycle_context]:
            raise ValueError(
                "classification, lifecycle, consumer scope, storage kind, mutation "
                "mode, and array-bearing state must agree with lifecycle_context"
            )
        if self.central_storage_catalog_required and (
            self.lifecycle_context is not AnalyticsLifecycleContext.TOP_LEVEL_RUN
        ):
            raise ValueError("only top-level analysis runs enter the central catalog")
        if (
            self.lifecycle_context
            in {
                AnalyticsLifecycleContext.DIAGNOSTIC_EXPORT,
                AnalyticsLifecycleContext.TRACK_BOUND_VISUALIZATION,
            }
            and self.exact_storage_contract_status
            is not ExactStorageContractStatus.NOT_APPLICABLE
        ):
            raise ValueError(
                "artifact and export contexts do not use array storage contracts"
            )
        if (
            self.lifecycle_context
            in {
                AnalyticsLifecycleContext.LEGACY_STANDALONE_RUN,
                AnalyticsLifecycleContext.IN_PLACE_COMPATIBILITY_REPAIR,
            }
            and self.exact_storage_contract_status
            is not ExactStorageContractStatus.LEGACY_COMPATIBILITY_ONLY
        ):
            raise ValueError("compatibility contexts require compatibility-only status")
        if (
            self.lifecycle_context
            is AnalyticsLifecycleContext.RECORDING_WIDE_QUALITY_SNAPSHOT
        ):
            if (
                self.stage_binding != "detect_quality"
                or self.owner_path != "detect_quality_runs/{run}"
            ):
                raise ValueError(
                    "recording-wide quality context must bind the root quality family"
                )
        if (
            self.lifecycle_context
            is AnalyticsLifecycleContext.SOURCE_DETECT_NESTED_QUALITY_REPORT
        ):
            if (
                self.stage_binding != "detect_quality"
                or self.owner_path
                != "detect_runs/{detect_run}/quality_reports/{quality_run}"
            ):
                raise ValueError(
                    "source-local quality context must bind the nested quality family"
                )

    def resolves_owner_entrypoint(self) -> bool:
        """Return whether the classified current code owner is callable."""

        module: Any = import_module(self.owner_module)
        return callable(getattr(module, self.owner_entrypoint, None))

    @property
    def exact_storage_contract_required(self) -> bool:
        """Return the requirement as a value derived from the closed status."""

        return self.exact_storage_contract_status is ExactStorageContractStatus.REQUIRED

    def as_record(self) -> dict[str, object]:
        """Return a strict JSON-compatible coverage record."""

        return {
            "surface_id": self.surface_id,
            "classification": self.classification.value,
            "lifecycle": self.lifecycle.value,
            "lifecycle_context": self.lifecycle_context.value,
            "consumer_scope": self.consumer_scope.value,
            "stage_binding": self.stage_binding,
            "owner_module": self.owner_module,
            "owner_entrypoint": self.owner_entrypoint,
            "owner_path": self.owner_path,
            "storage_kind": self.storage_kind.value,
            "mutation_mode": self.mutation_mode.value,
            "array_bearing": self.array_bearing,
            "exact_storage_contract_status": self.exact_storage_contract_status.value,
            "central_storage_catalog_required": self.central_storage_catalog_required,
            "schema_id": self.schema_id,
            "schema_version": self.schema_version,
        }


def _scientific_run(
    *,
    surface_id: str,
    owner_module: str,
    owner_entrypoint: str,
    owner_path: str,
    schema_id: str,
    schema_version: int,
    consumer_scope: AnalyticsConsumerScope = AnalyticsConsumerScope.RECORDING_ANALYSIS,
    lifecycle_context: AnalyticsLifecycleContext = AnalyticsLifecycleContext.TOP_LEVEL_RUN,
    central_catalog: bool = True,
) -> AnalyticsSurfaceClassification:
    return AnalyticsSurfaceClassification(
        surface_id=surface_id,
        classification=AnalyticsSurfaceClass.MAINTAINED_SCIENTIFIC_AUTHORITY,
        lifecycle=AnalyticsSurfaceLifecycle.CURRENT,
        lifecycle_context=lifecycle_context,
        consumer_scope=consumer_scope,
        stage_binding=surface_id,
        owner_module=owner_module,
        owner_entrypoint=owner_entrypoint,
        owner_path=owner_path,
        storage_kind=AnalyticsStorageKind.ZARR,
        mutation_mode=AnalyticsMutationMode.IMMUTABLE_RUN,
        array_bearing=True,
        exact_storage_contract_status=ExactStorageContractStatus.REQUIRED,
        central_storage_catalog_required=central_catalog,
        schema_id=schema_id,
        schema_version=schema_version,
    )


def _chaser_component(
    *,
    surface_id: str,
    owner_module: str,
    owner_entrypoint: str,
    component_parent: str,
    schema_id: str,
    schema_version: int,
) -> AnalyticsSurfaceClassification:
    return AnalyticsSurfaceClassification(
        surface_id=surface_id,
        classification=AnalyticsSurfaceClass.EMBEDDED_COMPONENT,
        lifecycle=AnalyticsSurfaceLifecycle.CURRENT,
        lifecycle_context=AnalyticsLifecycleContext.CHASER_EMBEDDED_COMPONENT,
        consumer_scope=AnalyticsConsumerScope.CHASER_COMPONENT_READER,
        stage_binding=surface_id,
        owner_module=owner_module,
        owner_entrypoint=owner_entrypoint,
        owner_path=(
            "analysis/chaser_distance_runs/{run}/" f"{component_parent}/{{component}}"
        ),
        storage_kind=AnalyticsStorageKind.ZARR,
        mutation_mode=AnalyticsMutationMode.EMBEDDED_RUN,
        array_bearing=True,
        exact_storage_contract_status=ExactStorageContractStatus.REQUIRED,
        central_storage_catalog_required=False,
        schema_id=schema_id,
        schema_version=schema_version,
    )


ANALYTICS_SURFACE_CLASSIFICATIONS: tuple[AnalyticsSurfaceClassification, ...] = (
    _scientific_run(
        surface_id="stimulus_epochs",
        owner_module="fisheye.analysis.stimulus_epoch_runs",
        owner_entrypoint="write_stimulus_epoch_run",
        owner_path="analysis/stimulus_epoch_runs/{run}",
        schema_id="palette.stimulus_epoch_windows.v1",
        schema_version=1,
    ),
    _scientific_run(
        surface_id="detection_occupancy",
        owner_module="fisheye.analysis.detection_occupancy_runs",
        owner_entrypoint="write_detection_occupancy_run",
        owner_path="analysis/detection_occupancy_runs/{run}",
        schema_id="palette.detection_occupancy.v1",
        schema_version=1,
    ),
    _scientific_run(
        surface_id="session_occupancy",
        owner_module="fisheye.analysis.detection_occupancy_runs",
        owner_entrypoint="write_session_occupancy_run",
        owner_path="analysis/session_occupancy_runs/{run}",
        schema_id="palette.session_occupancy.v1",
        schema_version=1,
    ),
    _scientific_run(
        surface_id="chaser_distance",
        owner_module="fisheye.analysis.chaser_distance_runs",
        owner_entrypoint="write_chaser_distance_run",
        owner_path="analysis/chaser_distance_runs/{run}",
        schema_id="palette.chaser_distance.v1",
        schema_version=1,
    ),
    _chaser_component(
        surface_id="chaser_quadrant_occupancy",
        owner_module="fisheye.analysis.chaser_quadrant_occupancy",
        owner_entrypoint="write_chaser_quadrant_occupancy_component",
        component_parent="chaser_quadrant_occupancy",
        schema_id="palette.chaser.quadrant_occupancy.v1",
        schema_version=1,
    ),
    _chaser_component(
        surface_id="chaser_near_field_occupancy",
        owner_module="fisheye.analysis.chaser_near_field_occupancy",
        owner_entrypoint="write_chaser_near_field_occupancy_component",
        component_parent="chaser_near_field_occupancy",
        schema_id="palette.chaser.near_field_occupancy.v1",
        schema_version=1,
    ),
    _chaser_component(
        surface_id="chaser_epoch_behavior_summary",
        owner_module="fisheye.analysis.chaser_epoch_behavior_summary",
        owner_entrypoint="write_chaser_epoch_behavior_summary_component",
        component_parent="epoch_behavior_summary",
        schema_id="palette.chaser.epoch_behavior_summary.v1",
        schema_version=1,
    ),
    _chaser_component(
        surface_id="chaser_egocentric_bearing",
        owner_module="fisheye.analysis.chaser_egocentric_bearing",
        owner_entrypoint="write_chaser_egocentric_bearing_component",
        component_parent="egocentric_bearing",
        schema_id="palette.chaser_egocentric_bearing.v1",
        schema_version=1,
    ),
    _chaser_component(
        surface_id="chaser_gaze_tracking",
        owner_module="fisheye.analysis.chaser_gaze_tracking",
        owner_entrypoint="write_chaser_gaze_tracking_component",
        component_parent="gaze_tracking",
        schema_id="palette.chaser_gaze_tracking.v1",
        schema_version=1,
    ),
    _chaser_component(
        surface_id="chaser_bout_response",
        owner_module="fisheye.analysis.chaser_bout_response",
        owner_entrypoint="write_chaser_bout_response_component",
        component_parent="chaser_bout_response",
        schema_id="palette.chaser_bout_response.v1",
        schema_version=1,
    ),
    _chaser_component(
        surface_id="chaser_escape_events",
        owner_module="fisheye.analysis.chaser_escape_events",
        owner_entrypoint="write_chaser_escape_events_component",
        component_parent="chaser_escape_events",
        schema_id="palette.chaser_escape_events.v3",
        schema_version=3,
    ),
    _chaser_component(
        surface_id="chaser_radial_occupancy",
        owner_module="fisheye.analysis.chaser_radial_occupancy",
        owner_entrypoint="write_chaser_radial_occupancy_component",
        component_parent="chaser_radial_occupancy",
        schema_id="palette.chaser_radial_occupancy.v1",
        schema_version=1,
    ),
    _chaser_component(
        surface_id="chaser_response_regimes",
        owner_module="fisheye.analysis.chaser_response_regimes",
        owner_entrypoint="write_chaser_response_regimes_component",
        component_parent="chaser_response_regimes",
        schema_id="palette.chaser_response_regimes.v1",
        schema_version=1,
    ),
    _chaser_component(
        surface_id="chaser_escape_freeze_summary",
        owner_module="fisheye.analysis.chaser_escape_freeze_summary",
        owner_entrypoint="write_chaser_escape_freeze_summary_component",
        component_parent="chaser_escape_freeze",
        schema_id="palette.chaser.escape_freeze_summary.v1",
        schema_version=1,
    ),
    _scientific_run(
        surface_id="registered_detection_gate",
        owner_module=(
            "fisheye.analysis_workflows.materializers.registered_detection_gate"
        ),
        owner_entrypoint="publish_registered_detection_gate",
        owner_path="analysis/detection_gate_runs/{run}",
        schema_id="palette.registered_detection_gate_run",
        schema_version=1,
        consumer_scope=AnalyticsConsumerScope.REGISTERED_DETECTION_GATE,
        lifecycle_context=AnalyticsLifecycleContext.POLICY_EVIDENCE_RUN,
        central_catalog=False,
    ),
    AnalyticsSurfaceClassification(
        surface_id="detection_quality_collection",
        classification=AnalyticsSurfaceClass.MAINTAINED_SCIENTIFIC_AUTHORITY,
        lifecycle=AnalyticsSurfaceLifecycle.CURRENT,
        lifecycle_context=AnalyticsLifecycleContext.RECORDING_WIDE_QUALITY_SNAPSHOT,
        consumer_scope=AnalyticsConsumerScope.RECORDING_WIDE_DETECTION_QUALITY,
        stage_binding="detect_quality",
        owner_module="fisheye.refinement.detect_quality_collection",
        owner_entrypoint="run_collection_detect_quality",
        owner_path="detect_quality_runs/{run}",
        storage_kind=AnalyticsStorageKind.ZARR,
        mutation_mode=AnalyticsMutationMode.IMMUTABLE_RUN,
        array_bearing=True,
        exact_storage_contract_status=ExactStorageContractStatus.REQUIRED,
        central_storage_catalog_required=False,
        schema_id="palette.detect_quality_collection.v2",
        schema_version=2,
    ),
    AnalyticsSurfaceClassification(
        surface_id="detection_quality_nested_report",
        classification=AnalyticsSurfaceClass.MAINTAINED_SCIENTIFIC_AUTHORITY,
        lifecycle=AnalyticsSurfaceLifecycle.CURRENT,
        lifecycle_context=(
            AnalyticsLifecycleContext.SOURCE_DETECT_NESTED_QUALITY_REPORT
        ),
        consumer_scope=AnalyticsConsumerScope.SOURCE_DETECT_LOCAL_QUALITY,
        stage_binding="detect_quality",
        owner_module="fisheye.refinement.detect_quality",
        owner_entrypoint="save_quality_report",
        owner_path="detect_runs/{detect_run}/quality_reports/{quality_run}",
        storage_kind=AnalyticsStorageKind.ZARR,
        mutation_mode=AnalyticsMutationMode.IMMUTABLE_RUN,
        array_bearing=True,
        exact_storage_contract_status=ExactStorageContractStatus.REQUIRED,
        central_storage_catalog_required=False,
    ),
    AnalyticsSurfaceClassification(
        surface_id="registered_detection_gate_audit",
        classification=AnalyticsSurfaceClass.EXPORT,
        lifecycle=AnalyticsSurfaceLifecycle.CURRENT,
        lifecycle_context=AnalyticsLifecycleContext.DIAGNOSTIC_EXPORT,
        consumer_scope=AnalyticsConsumerScope.OPERATOR_DIAGNOSTIC,
        stage_binding="registered_detection_gate_audit",
        owner_module="fisheye.diagnostics.audit_arena_geometry_detection_gates",
        owner_entrypoint="run_audit",
        owner_path="{output_dir}/audit_report.json",
        storage_kind=AnalyticsStorageKind.FILESYSTEM,
        mutation_mode=AnalyticsMutationMode.FILESYSTEM_GENERATION,
        array_bearing=False,
        exact_storage_contract_status=ExactStorageContractStatus.NOT_APPLICABLE,
        central_storage_catalog_required=False,
        schema_id="palette.diagnostics.arena_geometry_detection_gate_audit",
        schema_version=1,
    ),
    AnalyticsSurfaceClassification(
        surface_id="speed_runs",
        classification=AnalyticsSurfaceClass.LEGACY,
        lifecycle=AnalyticsSurfaceLifecycle.LEGACY,
        lifecycle_context=AnalyticsLifecycleContext.LEGACY_STANDALONE_RUN,
        consumer_scope=AnalyticsConsumerScope.LEGACY_COMPATIBILITY,
        stage_binding="speed",
        owner_module="fisheye.analysis.compute_speed",
        owner_entrypoint="main",
        owner_path="analysis/speed_runs/{run}",
        storage_kind=AnalyticsStorageKind.ZARR,
        mutation_mode=AnalyticsMutationMode.IMMUTABLE_RUN,
        array_bearing=True,
        exact_storage_contract_status=(
            ExactStorageContractStatus.LEGACY_COMPATIBILITY_ONLY
        ),
        central_storage_catalog_required=False,
    ),
    AnalyticsSurfaceClassification(
        surface_id="swim_bout_statistics",
        classification=AnalyticsSurfaceClass.LEGACY,
        lifecycle=AnalyticsSurfaceLifecycle.CURRENT_LEGACY_SHAPED,
        lifecycle_context=(AnalyticsLifecycleContext.ACTIVE_LEGACY_NAMESPACE_COLLISION),
        consumer_scope=AnalyticsConsumerScope.SHARED_SWIM_BOUT_SELECTOR,
        stage_binding="swim_bout_statistics",
        owner_module="fisheye.analysis.swim_bout_statistics",
        owner_entrypoint="_save_report_to_zarr",
        owner_path="analysis/swim_bout_runs/{run}",
        storage_kind=AnalyticsStorageKind.ZARR,
        mutation_mode=AnalyticsMutationMode.IMMUTABLE_RUN,
        array_bearing=True,
        exact_storage_contract_status=ExactStorageContractStatus.REQUIRED,
        central_storage_catalog_required=False,
    ),
    AnalyticsSurfaceClassification(
        surface_id="chaser_state_interpolation",
        classification=AnalyticsSurfaceClass.MAINTENANCE_OUTPUT,
        lifecycle=AnalyticsSurfaceLifecycle.LEGACY,
        lifecycle_context=AnalyticsLifecycleContext.IN_PLACE_COMPATIBILITY_REPAIR,
        consumer_scope=AnalyticsConsumerScope.STIMULUS_REPAIR,
        stage_binding="chaser_state_interpolation",
        owner_module="fisheye.analysis.chaser_state_interpolator",
        owner_entrypoint="_interpolate_run",
        owner_path="analysis/stimulus_runs/{run}",
        storage_kind=AnalyticsStorageKind.ZARR,
        mutation_mode=AnalyticsMutationMode.IN_PLACE_MUTATION,
        array_bearing=True,
        exact_storage_contract_status=(
            ExactStorageContractStatus.LEGACY_COMPATIBILITY_ONLY
        ),
        central_storage_catalog_required=False,
    ),
    AnalyticsSurfaceClassification(
        surface_id="track_kinematics_visualization",
        classification=AnalyticsSurfaceClass.VISUALIZATION_CACHE,
        lifecycle=AnalyticsSurfaceLifecycle.CURRENT,
        lifecycle_context=AnalyticsLifecycleContext.TRACK_BOUND_VISUALIZATION,
        consumer_scope=AnalyticsConsumerScope.TRACK_PRESENTATION,
        stage_binding="track_kinematics_visualization",
        owner_module="fisheye.analysis.plot_track_kinematics",
        owner_entrypoint="publish_track_kinematics_plot_artifacts",
        owner_path=(
            "analysis/track_kinematics_visualization_runs/"
            "*/*/tracks/id_*/*/visualizations"
        ),
        storage_kind=AnalyticsStorageKind.ZARR,
        mutation_mode=AnalyticsMutationMode.ARTIFACT_RUN,
        array_bearing=True,
        exact_storage_contract_status=ExactStorageContractStatus.NOT_APPLICABLE,
        central_storage_catalog_required=False,
        schema_id="palette.track_kinematics_visualization_run",
        schema_version=1,
    ),
)


ANALYTICS_SURFACE_CLASSIFICATION_BY_ID = {
    entry.surface_id: entry for entry in ANALYTICS_SURFACE_CLASSIFICATIONS
}

CHASER_PROFILE_SURFACE_IDS = frozenset(
    {
        "stimulus_epochs",
        "detection_occupancy",
        "chaser_distance",
        "chaser_quadrant_occupancy",
        "chaser_near_field_occupancy",
        "chaser_epoch_behavior_summary",
        "chaser_egocentric_bearing",
        "chaser_gaze_tracking",
        "chaser_bout_response",
        "chaser_escape_events",
        "chaser_radial_occupancy",
        "chaser_response_regimes",
        "chaser_escape_freeze_summary",
    }
)


def classified_surface_records() -> tuple[dict[str, object], ...]:
    """Return stable JSON-compatible classification records."""

    return tuple(entry.as_record() for entry in ANALYTICS_SURFACE_CLASSIFICATIONS)


__all__ = [
    "ANALYTICS_SURFACE_CLASSIFICATION_BY_ID",
    "ANALYTICS_SURFACE_CLASSIFICATIONS",
    "CHASER_PROFILE_SURFACE_IDS",
    "AnalyticsConsumerScope",
    "AnalyticsLifecycleContext",
    "AnalyticsMutationMode",
    "AnalyticsStorageKind",
    "AnalyticsSurfaceClass",
    "AnalyticsSurfaceClassification",
    "AnalyticsSurfaceLifecycle",
    "ExactStorageContractStatus",
    "classified_surface_records",
]
