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
    IMPLEMENTED_INDEPENDENT = "implemented_independent"
    NOT_APPLICABLE = "not_applicable"
    LEGACY_COMPATIBILITY_ONLY = "legacy_compatibility_only"


_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_]*$")
_ENTRYPOINT = re.compile(r"^_?[a-z][a-z0-9_]*$")


@dataclass(frozen=True)
class AnalyticsSurfaceClassification:
    """One truthful classification of a persisted analytics surface."""

    surface_id: str
    classification: AnalyticsSurfaceClass
    lifecycle: AnalyticsSurfaceLifecycle
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
        if not self.owner_module.startswith("fisheye.") or (
            self.owner_module != self.owner_module.strip()
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
            or any(ord(char) < 32 or ord(char) == 127 for char in self.owner_path)
            or any(part in {"", ".", ".."} for part in self.owner_path.split("/"))
        ):
            raise ValueError("owner_path must be one canonical path pattern")
        if (self.schema_id is None) != (self.schema_version is None):
            raise ValueError("schema_id and schema_version must be declared together")
        if self.schema_id is not None and (
            not self.schema_id.startswith("palette.")
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
        if self.classification is AnalyticsSurfaceClass.LEGACY and (
            self.lifecycle is not AnalyticsSurfaceLifecycle.LEGACY
        ):
            raise ValueError("legacy surfaces must declare the legacy lifecycle")
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
                or self.lifecycle is not AnalyticsSurfaceLifecycle.CURRENT
                or self.classification
                not in {
                    AnalyticsSurfaceClass.MAINTAINED_SCIENTIFIC_AUTHORITY,
                    AnalyticsSurfaceClass.EMBEDDED_COMPONENT,
                }
            )
        ):
            raise ValueError(
                "required exact contracts are reserved for current array-bearing "
                "scientific authorities and embedded components"
            )
        if self.mutation_mode is AnalyticsMutationMode.IN_PLACE_MUTATION and (
            self.classification is not AnalyticsSurfaceClass.MAINTENANCE_OUTPUT
        ):
            raise ValueError("in-place mutation is reserved for maintenance outputs")

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
    central_catalog: bool = True,
) -> AnalyticsSurfaceClassification:
    return AnalyticsSurfaceClassification(
        surface_id=surface_id,
        classification=AnalyticsSurfaceClass.MAINTAINED_SCIENTIFIC_AUTHORITY,
        lifecycle=AnalyticsSurfaceLifecycle.CURRENT,
        owner_module=owner_module,
        owner_entrypoint=owner_entrypoint,
        owner_path=owner_path,
        storage_kind=AnalyticsStorageKind.ZARR,
        mutation_mode=AnalyticsMutationMode.IMMUTABLE_RUN,
        array_bearing=True,
        exact_storage_contract_status=(
            ExactStorageContractStatus.REQUIRED
            if central_catalog
            else ExactStorageContractStatus.IMPLEMENTED_INDEPENDENT
        ),
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
        central_catalog=False,
    ),
    AnalyticsSurfaceClassification(
        surface_id="detection_quality_collection",
        classification=AnalyticsSurfaceClass.MAINTAINED_SCIENTIFIC_AUTHORITY,
        lifecycle=AnalyticsSurfaceLifecycle.CURRENT,
        owner_module="fisheye.refinement.detect_quality_collection",
        owner_entrypoint="run_collection_detect_quality",
        owner_path="detect_quality_runs/{run}",
        storage_kind=AnalyticsStorageKind.ZARR,
        mutation_mode=AnalyticsMutationMode.IMMUTABLE_RUN,
        array_bearing=True,
        exact_storage_contract_status=(
            ExactStorageContractStatus.IMPLEMENTED_INDEPENDENT
        ),
        central_storage_catalog_required=False,
        schema_id="palette.detect_quality_collection.v2",
        schema_version=2,
    ),
    AnalyticsSurfaceClassification(
        surface_id="detection_quality_nested_report",
        classification=AnalyticsSurfaceClass.MAINTAINED_SCIENTIFIC_AUTHORITY,
        lifecycle=AnalyticsSurfaceLifecycle.CURRENT,
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
        lifecycle=AnalyticsSurfaceLifecycle.LEGACY,
        owner_module="fisheye.analysis.swim_bout_statistics",
        owner_entrypoint="_save_report_to_zarr",
        owner_path="analysis/swim_bout_runs/{run}",
        storage_kind=AnalyticsStorageKind.ZARR,
        mutation_mode=AnalyticsMutationMode.IMMUTABLE_RUN,
        array_bearing=True,
        exact_storage_contract_status=(
            ExactStorageContractStatus.LEGACY_COMPATIBILITY_ONLY
        ),
        central_storage_catalog_required=False,
    ),
    AnalyticsSurfaceClassification(
        surface_id="chaser_state_interpolation",
        classification=AnalyticsSurfaceClass.MAINTENANCE_OUTPUT,
        lifecycle=AnalyticsSurfaceLifecycle.LEGACY,
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
    "AnalyticsMutationMode",
    "AnalyticsStorageKind",
    "AnalyticsSurfaceClass",
    "AnalyticsSurfaceClassification",
    "AnalyticsSurfaceLifecycle",
    "ExactStorageContractStatus",
    "classified_surface_records",
]
