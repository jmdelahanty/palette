"""Executable catalog of maintained derived-analysis storage contracts.

The scientific writers remain the authority for schema and method constants.
This catalog records where those constants live, which run family owns the
result, and which production materializer or guarded direct writer owns
publication.  Resolving an entry therefore reads the writer constant instead
of duplicating its value here.

Physical policy remains stage-owned for now.  ``byte_planner_adopted`` makes
that migration boundary explicit so a fixed-row policy cannot be mistaken for
the shared byte-budgeted storage planner used by newer observation stages.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
import re
from typing import Any


_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_]*$")
_MODULE = re.compile(r"^[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)+$")
_CONSTANT_ATTR = re.compile(r"^[A-Z][A-Z0-9_]*$")
_CALLABLE_ATTR = re.compile(r"^_?[a-z][a-z0-9_]*$")
_PATH_SEGMENT = re.compile(r"^[a-z][a-z0-9_]*$")
_POLICY_OWNER = re.compile(r"^[A-Za-z0-9_.]+$")


@dataclass(frozen=True)
class DerivedAnalysisStorageContract:
    """One maintained array-bearing derived-analysis run family."""

    stage_id: str
    run_parent: str
    availability_parents: tuple[str, ...]
    schema_module: str
    schema_id_attr: str
    schema_version_attr: str
    method_version_attr: str
    layout_attr: str | None
    materializer_module: str | None
    physical_policy_owner: str
    registry_publication: str
    byte_planner_adopted: bool = False
    method_attr: str | None = None
    publication_owner_kind: str = "shared_atomic_materializer_v1"
    publication_owner_module: str | None = None
    publication_entrypoint_attr: str | None = None

    def __post_init__(self) -> None:
        """Reject contradictory or noncanonical catalog declarations."""

        required_text = {
            "stage_id": self.stage_id,
            "run_parent": self.run_parent,
            "schema_module": self.schema_module,
            "schema_id_attr": self.schema_id_attr,
            "schema_version_attr": self.schema_version_attr,
            "method_version_attr": self.method_version_attr,
            "physical_policy_owner": self.physical_policy_owner,
            "registry_publication": self.registry_publication,
        }
        for field, value in required_text.items():
            if type(value) is not str or not value or value != value.strip():
                raise ValueError(f"{field} must be one nonempty exact string")

        def require_relative_path(value: str, *, field: str) -> None:
            parts = value.split("/")
            if (
                value != value.strip()
                or value.startswith("/")
                or value.endswith("/")
                or any(not _PATH_SEGMENT.fullmatch(part) for part in parts)
            ):
                raise ValueError(f"{field} must be one canonical relative path")

        if not _IDENTIFIER.fullmatch(self.stage_id):
            raise ValueError("stage_id must be one canonical identifier")
        if not _MODULE.fullmatch(self.schema_module):
            raise ValueError("schema_module must be one canonical module path")
        for field in (
            "schema_id_attr",
            "schema_version_attr",
            "method_version_attr",
        ):
            if not _CONSTANT_ATTR.fullmatch(getattr(self, field)):
                raise ValueError(f"{field} must be one canonical constant attr")
        if not _POLICY_OWNER.fullmatch(self.physical_policy_owner):
            raise ValueError(
                "physical_policy_owner must be one canonical policy identifier"
            )
        require_relative_path(self.run_parent, field="run_parent")
        if (
            not isinstance(self.availability_parents, tuple)
            or not self.availability_parents
        ):
            raise ValueError("availability_parents must be one nonempty tuple")
        for parent in self.availability_parents:
            if type(parent) is not str:
                raise ValueError("availability parents must be exact strings")
            require_relative_path(parent, field="availability_parent")
            if parent != self.run_parent and not parent.startswith(
                f"{self.run_parent}/"
            ):
                raise ValueError(
                    "availability parents must equal or be nested below run_parent"
                )

        if type(self.byte_planner_adopted) is not bool:
            raise TypeError("byte_planner_adopted must be an exact bool")
        if self.registry_publication not in {
            "not_implemented",
            "serialized_finalizer_v1",
        }:
            raise ValueError("registry_publication must name one supported exact mode")
        for field, value in (
            ("layout_attr", self.layout_attr),
            ("method_attr", self.method_attr),
        ):
            if value is not None and (
                type(value) is not str or not _CONSTANT_ATTR.fullmatch(value)
            ):
                raise ValueError(f"{field} must be None or one canonical constant attr")

        allowed_owner_kinds = {
            "shared_atomic_materializer_v1",
            "guarded_direct_writer_v1",
        }
        if self.publication_owner_kind not in allowed_owner_kinds:
            raise ValueError(
                "publication_owner_kind must name one supported exact owner mode"
            )
        if self.publication_owner_kind == "shared_atomic_materializer_v1":
            if type(self.materializer_module) is not str or not _MODULE.fullmatch(
                self.materializer_module
            ):
                raise ValueError(
                    "shared atomic publication requires materializer_module"
                )
            if self.publication_owner_module is not None:
                raise ValueError(
                    "shared atomic publication derives its owner from materializer_module"
                )
            if self.publication_entrypoint_attr is not None:
                raise ValueError(
                    "shared atomic publication forbids a direct-writer entrypoint"
                )
        else:
            if self.materializer_module is not None:
                raise ValueError(
                    "guarded direct publication must not claim a materializer"
                )
            if type(self.publication_owner_module) is not str or not _MODULE.fullmatch(
                self.publication_owner_module
            ):
                raise ValueError(
                    "guarded direct publication requires an exact owner module"
                )
            if type(
                self.publication_entrypoint_attr
            ) is not str or not _CALLABLE_ATTR.fullmatch(
                self.publication_entrypoint_attr
            ):
                raise ValueError(
                    "guarded direct publication requires an exact entrypoint attr"
                )

    def _writer_module(self) -> Any:
        return import_module(self.schema_module)

    def resolved_schema(self) -> dict[str, object]:
        """Resolve the live writer constants for diagnostics and validation."""

        module = self._writer_module()
        schema_id = getattr(module, self.schema_id_attr)
        schema_version = getattr(module, self.schema_version_attr)
        method_version = getattr(module, self.method_version_attr)
        method = getattr(module, self.method_attr) if self.method_attr else None
        layout = getattr(module, self.layout_attr) if self.layout_attr else None
        return {
            "stage_id": self.stage_id,
            "run_parent": self.run_parent,
            "availability_parents": list(self.availability_parents),
            "schema_id": str(schema_id),
            "schema_version": int(schema_version),
            "method": method,
            "method_version": method_version,
            "layout": layout,
            "materializer_module": self.materializer_module,
            "publication_owner_module": (
                self.publication_owner_module or self.materializer_module
            ),
            "physical_policy_owner": self.physical_policy_owner,
            "registry_publication": self.registry_publication,
            "byte_planner_adopted": self.byte_planner_adopted,
            "publication_owner_kind": self.publication_owner_kind,
            "publication_entrypoint": self.publication_entrypoint_attr,
        }

    def uses_shared_atomic_publisher(self) -> bool:
        """Return whether the materializer imports the canonical publisher."""

        if self.materializer_module is None:
            return False
        materializer = import_module(self.materializer_module)
        publisher = import_module(
            "fisheye.analysis_workflows.materializers.atomic_run_publisher"
        )
        return (
            getattr(materializer, "atomic_publish_run_group", None)
            is publisher.atomic_publish_run_group
        )

    def resolves_publication_entrypoint(self) -> bool:
        """Return whether the declared direct publication entry point exists."""

        if self.publication_entrypoint_attr is None:
            return False
        owner_module = self.publication_owner_module or self.materializer_module
        if owner_module is None:
            return False
        owner = import_module(owner_module)
        return callable(getattr(owner, self.publication_entrypoint_attr, None))


DERIVED_ANALYSIS_STORAGE_CONTRACTS: tuple[DerivedAnalysisStorageContract, ...] = (
    DerivedAnalysisStorageContract(
        stage_id="track_kinematics",
        run_parent="analysis/track_kinematics_runs",
        availability_parents=("analysis/track_kinematics_runs/offline",),
        schema_module="fisheye.analysis.track_kinematics",
        schema_id_attr="TRACK_KINEMATICS_RUN_SCHEMA_ID",
        schema_version_attr="TRACK_KINEMATICS_RUN_SCHEMA_VERSION",
        method_version_attr="TRACK_KINEMATICS_METHOD_VERSION",
        layout_attr=None,
        materializer_module=(
            "fisheye.analysis_workflows.materializers.track_kinematics"
        ),
        physical_policy_owner="track_kinematics_rechunk_v3",
        registry_publication="serialized_finalizer_v1",
    ),
    DerivedAnalysisStorageContract(
        stage_id="swim_bouts",
        run_parent="analysis/swim_bout_runs",
        availability_parents=("analysis/swim_bout_runs",),
        schema_module="fisheye.analysis.detect_bouts_multi_level",
        schema_id_attr="SWIM_BOUT_RUN_SCHEMA_ID",
        schema_version_attr="SWIM_BOUT_RUN_SCHEMA_VERSION_FRAME_AXIS_REFERENCE",
        method_version_attr="METHOD_VERSION",
        layout_attr="SWIM_BOUT_STORED_LAYOUT_COMPACT_V2",
        materializer_module="fisheye.analysis_workflows.materializers.swim_bouts",
        physical_policy_owner="swim_bout_compact_tabular_v2",
        registry_publication="serialized_finalizer_v1",
    ),
    DerivedAnalysisStorageContract(
        stage_id="bout_kinematics",
        run_parent="analysis/bout_kinematics_runs",
        availability_parents=("analysis/bout_kinematics_runs",),
        schema_module="fisheye.analysis.bout_kinematics",
        schema_id_attr="SCHEMA_ID",
        schema_version_attr="SCHEMA_VERSION",
        method_version_attr="METHOD_VERSION",
        layout_attr="BOUT_KINEMATICS_LAYOUT_DEFAULT",
        materializer_module=(
            "fisheye.analysis_workflows.materializers.bout_kinematics"
        ),
        physical_policy_owner="shared_columnar_v1",
        registry_publication="serialized_finalizer_v1",
    ),
    DerivedAnalysisStorageContract(
        stage_id="eye_angles",
        run_parent="analysis/eye_angle_runs",
        availability_parents=("analysis/eye_angle_runs",),
        schema_module="fisheye.analysis.eye_angle_analysis",
        schema_id_attr="EYE_ANGLE_RUN_SCHEMA_ID",
        schema_version_attr="EYE_ANGLE_RUN_SCHEMA_VERSION",
        method_version_attr="EYE_ANGLE_METHOD_VERSION",
        layout_attr="EYE_ANGLE_LAYOUT_DEFAULT",
        materializer_module="fisheye.analysis_workflows.materializers.eye_angles",
        physical_policy_owner="eye_angle_semantic_dense_v2",
        registry_publication="serialized_finalizer_v1",
    ),
    DerivedAnalysisStorageContract(
        stage_id="subject_shape",
        run_parent="analysis/subject_shape_runs",
        availability_parents=("analysis/subject_shape_runs",),
        schema_module="fisheye.analysis.subject_shape_runs",
        schema_id_attr="SUBJECT_SHAPE_SCHEMA_ID",
        schema_version_attr="SUBJECT_SHAPE_SCHEMA_VERSION",
        method_version_attr="SUBJECT_SHAPE_METHOD_VERSION",
        layout_attr=None,
        materializer_module="fisheye.analysis_workflows.materializers.subject_shape",
        physical_policy_owner="subject_shape_indexed_shards_v1",
        registry_publication="serialized_finalizer_v1",
    ),
    DerivedAnalysisStorageContract(
        stage_id="tail_kinematics",
        run_parent="analysis/tail_kinematics_runs",
        availability_parents=("analysis/tail_kinematics_runs",),
        schema_module="fisheye.analysis.tail_kinematics_runs",
        schema_id_attr="TAIL_KINEMATICS_SCHEMA_ID",
        schema_version_attr="TAIL_KINEMATICS_SCHEMA_VERSION",
        method_version_attr="TAIL_KINEMATICS_METHOD_VERSION",
        layout_attr=None,
        materializer_module=(
            "fisheye.analysis_workflows.materializers.tail_kinematics"
        ),
        physical_policy_owner="tail_kinematics_process_shards_v1",
        registry_publication="serialized_finalizer_v1",
    ),
    DerivedAnalysisStorageContract(
        stage_id="tail_posture_view",
        run_parent="analysis/tail_posture_view_runs",
        availability_parents=("analysis/tail_posture_view_runs",),
        schema_module="fisheye.analysis.tail_posture_view_runs",
        schema_id_attr="TAIL_POSTURE_VIEW_SCHEMA_ID",
        schema_version_attr="TAIL_POSTURE_VIEW_SCHEMA_VERSION",
        method_version_attr="TAIL_POSTURE_VIEW_METHOD_VERSION",
        layout_attr=None,
        materializer_module=None,
        physical_policy_owner="refined_subject_mask_metric_row_chunk_compatibility",
        registry_publication="serialized_finalizer_v1",
        byte_planner_adopted=False,
        method_attr="TAIL_POSTURE_VIEW_METHOD",
        publication_owner_kind="guarded_direct_writer_v1",
        publication_owner_module="fisheye.analysis.tail_posture_view_runs",
        publication_entrypoint_attr="write_tail_posture_view_run",
    ),
    DerivedAnalysisStorageContract(
        stage_id="bout_classification",
        run_parent="analysis/bout_classification_runs",
        availability_parents=("analysis/bout_classification_runs",),
        schema_module="fisheye.analysis.megabouts_classifier",
        schema_id_attr="SCHEMA_ID",
        schema_version_attr="SCHEMA_VERSION",
        method_version_attr="ADAPTER_METHOD_VERSION",
        layout_attr=None,
        materializer_module=None,
        physical_policy_owner="columnar_store_array_v1",
        registry_publication="serialized_finalizer_v1",
        byte_planner_adopted=False,
        method_attr="ADAPTER_METHOD",
        publication_owner_kind="guarded_direct_writer_v1",
        publication_owner_module="fisheye.analysis.megabouts_classifier",
        publication_entrypoint_attr="write_megabouts_classification_run",
    ),
    DerivedAnalysisStorageContract(
        stage_id="stimulus_response",
        run_parent="analysis/stimulus_response_runs",
        availability_parents=("analysis/stimulus_response_runs",),
        schema_module="fisheye.analysis.stimulus_response",
        schema_id_attr="STIMULUS_RESPONSE_SCHEMA_ID",
        schema_version_attr="STIMULUS_RESPONSE_SCHEMA_VERSION",
        method_version_attr="STIMULUS_RESPONSE_METHOD_VERSION",
        layout_attr="STIMULUS_RESPONSE_LAYOUT_DEFAULT",
        materializer_module=(
            "fisheye.analysis_workflows.materializers.stimulus_response"
        ),
        physical_policy_owner="stimulus_response_compact_tabular_v2",
        registry_publication="serialized_finalizer_v1",
    ),
    DerivedAnalysisStorageContract(
        stage_id="stimulus_epochs",
        run_parent="analysis/stimulus_epoch_runs",
        availability_parents=("analysis/stimulus_epoch_runs",),
        schema_module="fisheye.analysis.stimulus_epoch_runs",
        schema_id_attr="SCHEMA_ID",
        schema_version_attr="SCHEMA_VERSION",
        method_version_attr="METHOD_VERSION",
        layout_attr=None,
        materializer_module=None,
        physical_policy_owner="stimulus_epoch_fixed_row_v1",
        registry_publication="serialized_finalizer_v1",
        byte_planner_adopted=False,
        method_attr="METHOD",
        publication_owner_kind="guarded_direct_writer_v1",
        publication_owner_module="fisheye.analysis.stimulus_epoch_runs",
        publication_entrypoint_attr="write_stimulus_epoch_run",
    ),
    DerivedAnalysisStorageContract(
        stage_id="detection_occupancy",
        run_parent="analysis/detection_occupancy_runs",
        availability_parents=("analysis/detection_occupancy_runs",),
        schema_module="fisheye.analysis.detection_occupancy_runs",
        schema_id_attr="SCHEMA_ID",
        schema_version_attr="SCHEMA_VERSION",
        method_version_attr="METHOD_VERSION",
        layout_attr=None,
        materializer_module=None,
        physical_policy_owner="detection_occupancy_fixed_row_v1",
        registry_publication="serialized_finalizer_v1",
        byte_planner_adopted=False,
        method_attr="METHOD",
        publication_owner_kind="guarded_direct_writer_v1",
        publication_owner_module="fisheye.analysis.detection_occupancy_runs",
        publication_entrypoint_attr="write_detection_occupancy_run",
    ),
    DerivedAnalysisStorageContract(
        stage_id="session_occupancy",
        run_parent="analysis/session_occupancy_runs",
        availability_parents=("analysis/session_occupancy_runs",),
        schema_module="fisheye.analysis.detection_occupancy_runs",
        schema_id_attr="SESSION_SCHEMA_ID",
        schema_version_attr="SCHEMA_VERSION",
        method_version_attr="METHOD_VERSION",
        layout_attr=None,
        materializer_module=None,
        physical_policy_owner="detection_occupancy_fixed_row_v1",
        registry_publication="serialized_finalizer_v1",
        byte_planner_adopted=False,
        method_attr="SESSION_METHOD",
        publication_owner_kind="guarded_direct_writer_v1",
        publication_owner_module="fisheye.analysis.detection_occupancy_runs",
        publication_entrypoint_attr="write_session_occupancy_run",
    ),
)


DERIVED_ANALYSIS_STORAGE_CONTRACT_BY_STAGE = {
    contract.stage_id: contract for contract in DERIVED_ANALYSIS_STORAGE_CONTRACTS
}

DERIVED_ANALYSIS_AVAILABILITY_RUN_PARENTS = {
    contract.stage_id: contract.availability_parents
    for contract in DERIVED_ANALYSIS_STORAGE_CONTRACTS
}

SERIALIZED_REGISTRY_STAGE_IDS = frozenset(
    contract.stage_id
    for contract in DERIVED_ANALYSIS_STORAGE_CONTRACTS
    if contract.registry_publication == "serialized_finalizer_v1"
)


def resolved_storage_contracts() -> tuple[dict[str, object], ...]:
    """Return current writer-backed declarations in stable catalog order."""

    return tuple(
        contract.resolved_schema() for contract in DERIVED_ANALYSIS_STORAGE_CONTRACTS
    )


__all__ = [
    "DERIVED_ANALYSIS_STORAGE_CONTRACT_BY_STAGE",
    "DERIVED_ANALYSIS_STORAGE_CONTRACTS",
    "DERIVED_ANALYSIS_AVAILABILITY_RUN_PARENTS",
    "SERIALIZED_REGISTRY_STAGE_IDS",
    "DerivedAnalysisStorageContract",
    "resolved_storage_contracts",
]
