"""Executable catalog of maintained derived-analysis storage contracts.

The scientific writers remain the authority for schema and method constants.
This catalog records where those constants live, which run family owns the
result, and which production materializer owns publication.  Resolving an entry
therefore reads the writer constant instead of duplicating its value here.

Physical policy remains stage-owned for now.  ``byte_planner_adopted`` makes
that migration boundary explicit so a fixed-row policy cannot be mistaken for
the shared byte-budgeted storage planner used by newer observation stages.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any


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
    materializer_module: str
    physical_policy_owner: str
    registry_publication: str
    byte_planner_adopted: bool = False

    def _writer_module(self) -> Any:
        return import_module(self.schema_module)

    def resolved_schema(self) -> dict[str, object]:
        """Resolve the live writer constants for diagnostics and validation."""

        module = self._writer_module()
        schema_id = getattr(module, self.schema_id_attr)
        schema_version = getattr(module, self.schema_version_attr)
        method_version = getattr(module, self.method_version_attr)
        layout = getattr(module, self.layout_attr) if self.layout_attr else None
        return {
            "stage_id": self.stage_id,
            "run_parent": self.run_parent,
            "availability_parents": list(self.availability_parents),
            "schema_id": str(schema_id),
            "schema_version": int(schema_version),
            "method_version": method_version,
            "layout": layout,
            "materializer_module": self.materializer_module,
            "physical_policy_owner": self.physical_policy_owner,
            "registry_publication": self.registry_publication,
            "byte_planner_adopted": self.byte_planner_adopted,
        }

    def uses_shared_atomic_publisher(self) -> bool:
        """Return whether the materializer imports the canonical publisher."""

        materializer = import_module(self.materializer_module)
        publisher = import_module(
            "fisheye.analysis_workflows.materializers.atomic_run_publisher"
        )
        return (
            getattr(materializer, "atomic_publish_run_group", None)
            is publisher.atomic_publish_run_group
        )


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
        registry_publication="not_implemented",
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
        registry_publication="not_implemented",
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
        registry_publication="not_implemented",
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
        registry_publication="not_implemented",
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
        registry_publication="not_implemented",
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
        contract.resolved_schema()
        for contract in DERIVED_ANALYSIS_STORAGE_CONTRACTS
    )


__all__ = [
    "DERIVED_ANALYSIS_STORAGE_CONTRACT_BY_STAGE",
    "DERIVED_ANALYSIS_STORAGE_CONTRACTS",
    "DERIVED_ANALYSIS_AVAILABILITY_RUN_PARENTS",
    "SERIALIZED_REGISTRY_STAGE_IDS",
    "DerivedAnalysisStorageContract",
    "resolved_storage_contracts",
]
