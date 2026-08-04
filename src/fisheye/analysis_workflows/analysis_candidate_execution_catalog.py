"""Typed execution-adapter catalog for derived-analysis storage candidates.

This catalog describes how each maintained family must enter the shared
writer/publication evidence runner.  It does not dispatch arbitrary keyword
arguments.  Until a dedicated typed runner is installed, the descriptor stays
``contract_only`` or explicitly blocked and cannot mint an execution request.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any

from fisheye.shared.zarr.manifest_digest import canonical_json_sha256

from .analysis_candidate_execution import (
    ANALYSIS_CANDIDATE_EXECUTION_ADAPTER_SCHEMA_ID,
    ANALYSIS_CANDIDATE_EXECUTION_ADAPTER_SCHEMA_VERSION,
    CandidateComputationMode,
    CandidateLogicalEqualityContract,
    CandidateRunnerStatus,
    CoordinateContractRole,
    CoordinateContractStatus,
    require_candidate_execution_adapter_manifest,
)
from .analysis_candidate_invocation import (
    CandidateInvocationContract,
    candidate_invocation_contract_is_frozen,
)
from .storage_candidate_catalog import (
    DERIVED_ANALYSIS_STORAGE_CANDIDATE_BY_STAGE,
    StorageCandidatePublicationMode,
)


@dataclass(frozen=True)
class AnalysisCandidateExecutionAdapter:
    """One exact family-to-runner contract."""

    stage_id: str
    invocation_contract: CandidateInvocationContract
    computation_mode: CandidateComputationMode
    runner_status: CandidateRunnerStatus
    coordinate_role: CoordinateContractRole
    coordinate_contract_status: CoordinateContractStatus
    logical_equality_contract: CandidateLogicalEqualityContract
    source_run_parent: str | None = None
    runner_module: str | None = None
    runner_entrypoint: str | None = None
    suite_validator_module: str | None = None
    suite_validator_entrypoint: str | None = None
    adapter_version: int = 1

    def __post_init__(self) -> None:
        if self.stage_id not in DERIVED_ANALYSIS_STORAGE_CANDIDATE_BY_STAGE:
            raise ValueError("execution adapter must own one cataloged candidate")
        for field, enum_type in (
            ("invocation_contract", CandidateInvocationContract),
            ("computation_mode", CandidateComputationMode),
            ("runner_status", CandidateRunnerStatus),
            ("coordinate_role", CoordinateContractRole),
            ("coordinate_contract_status", CoordinateContractStatus),
            ("logical_equality_contract", CandidateLogicalEqualityContract),
        ):
            if not isinstance(getattr(self, field), enum_type):
                raise TypeError(f"{field} must use {enum_type.__name__}")
        if type(self.adapter_version) is not int or self.adapter_version < 1:
            raise ValueError("adapter_version must be one positive exact integer")
        if self.runner_status is CandidateRunnerStatus.IMPLEMENTED:
            if not candidate_invocation_contract_is_frozen(self.invocation_contract):
                raise ValueError(
                    "implemented adapters require one frozen invocation grammar"
                )
            if (
                not self.runner_module
                or not self.runner_entrypoint
                or not self.suite_validator_module
                or not self.suite_validator_entrypoint
            ):
                raise ValueError(
                    "implemented adapters require one typed runner and suite validator"
                )
        elif any(
            value is not None
            for value in (
                self.runner_module,
                self.runner_entrypoint,
                self.suite_validator_module,
                self.suite_validator_entrypoint,
            )
        ):
            raise ValueError(
                "nonimplemented adapters must not claim a runner or suite validator"
            )

        candidate = DERIVED_ANALYSIS_STORAGE_CANDIDATE_BY_STAGE[self.stage_id]
        if (
            candidate.publication_mode is StorageCandidatePublicationMode.GUARDED_DIRECT
            and self.runner_status
            is not CandidateRunnerStatus.BLOCKED_DIRECT_PUBLICATION
        ):
            raise ValueError("guarded-direct candidate publication must remain blocked")
        if (
            self.coordinate_contract_status
            is CoordinateContractStatus.BLOCKED_CANONICAL_BINDING
            and self.runner_status
            is not CandidateRunnerStatus.BLOCKED_COORDINATE_AUTHORITY
        ):
            raise ValueError("unbound coordinate candidates must remain blocked")

    @property
    def adapter_id(self) -> str:
        return f"palette.analysis_candidate_execution.{self.stage_id}.v1"

    def as_manifest(self) -> dict[str, object]:
        candidate = DERIVED_ANALYSIS_STORAGE_CANDIDATE_BY_STAGE[self.stage_id]
        payload: dict[str, object] = {
            "stage_id": self.stage_id,
            "adapter_id": self.adapter_id,
            "adapter_version": self.adapter_version,
            "run_parent": candidate.run_parent,
            "source_run_parent": self.source_run_parent or candidate.run_parent,
            "profile_id": candidate.profile_id,
            "candidate_owner_module": candidate.owner_module,
            "candidate_owner_entrypoint": candidate.entrypoint_attr,
            "invocation_contract": self.invocation_contract.value,
            "computation_mode": self.computation_mode.value,
            "publication_mode": candidate.publication_mode.value,
            "runner_status": self.runner_status.value,
            "runner_module": self.runner_module,
            "runner_entrypoint": self.runner_entrypoint,
            "suite_validator_module": self.suite_validator_module,
            "suite_validator_entrypoint": self.suite_validator_entrypoint,
            "coordinate_role": self.coordinate_role.value,
            "coordinate_contract_status": self.coordinate_contract_status.value,
            "logical_equality_contract": self.logical_equality_contract.value,
        }
        result = {
            "schema_id": ANALYSIS_CANDIDATE_EXECUTION_ADAPTER_SCHEMA_ID,
            "schema_version": ANALYSIS_CANDIDATE_EXECUTION_ADAPTER_SCHEMA_VERSION,
            "payload": payload,
            "payload_digest": canonical_json_sha256(payload),
        }
        require_candidate_execution_adapter_manifest(result)
        return result

    def resolves_candidate_owner(self) -> bool:
        candidate = DERIVED_ANALYSIS_STORAGE_CANDIDATE_BY_STAGE[self.stage_id]
        module: Any = import_module(candidate.owner_module)
        return callable(getattr(module, candidate.entrypoint_attr, None))

    def resolves_runner(self) -> bool:
        if self.runner_module is None or self.runner_entrypoint is None:
            return False
        module: Any = import_module(self.runner_module)
        return callable(getattr(module, self.runner_entrypoint, None))

    def resolves_suite_validator(self) -> bool:
        if (
            self.suite_validator_module is None
            or self.suite_validator_entrypoint is None
        ):
            return False
        module: Any = import_module(self.suite_validator_module)
        return callable(getattr(module, self.suite_validator_entrypoint, None))


def _implemented_exact_tabular(
    stage_id: str,
    logical_equality_contract: CandidateLogicalEqualityContract,
) -> AnalysisCandidateExecutionAdapter:
    return AnalysisCandidateExecutionAdapter(
        stage_id=stage_id,
        invocation_contract=CandidateInvocationContract.EXACT_TABULAR_V1,
        computation_mode=CandidateComputationMode.LOGICAL_REMATERIALIZATION,
        runner_status=CandidateRunnerStatus.IMPLEMENTED,
        coordinate_role=CoordinateContractRole.BOUND_DERIVATIVE,
        coordinate_contract_status=(
            CoordinateContractStatus.BOUND_SOURCE_VALIDATION_IMPLEMENTED
        ),
        logical_equality_contract=logical_equality_contract,
        runner_module="fisheye.diagnostics.analysis_candidate_execution",
        runner_entrypoint="execute_exact_tabular_candidate",
        suite_validator_module=(
            "fisheye.analysis_workflows.analysis_candidate_suite_validation"
        ),
        suite_validator_entrypoint="require_exact_tabular_execution_suite",
    )


def _implemented_occupancy(
    stage_id: str,
    logical_equality_contract: CandidateLogicalEqualityContract,
) -> AnalysisCandidateExecutionAdapter:
    return AnalysisCandidateExecutionAdapter(
        stage_id=stage_id,
        invocation_contract=CandidateInvocationContract.OCCUPANCY_V1,
        computation_mode=CandidateComputationMode.LOGICAL_REMATERIALIZATION,
        runner_status=CandidateRunnerStatus.IMPLEMENTED,
        coordinate_role=CoordinateContractRole.BOUND_DERIVATIVE,
        coordinate_contract_status=(
            CoordinateContractStatus.BOUND_SOURCE_VALIDATION_IMPLEMENTED
        ),
        logical_equality_contract=logical_equality_contract,
        runner_module="fisheye.diagnostics.analysis_candidate_execution",
        runner_entrypoint="execute_exact_tabular_candidate",
        suite_validator_module=(
            "fisheye.analysis_workflows.analysis_candidate_suite_validation"
        ),
        suite_validator_entrypoint="require_exact_tabular_execution_suite",
    )


def _implemented_track_flat() -> AnalysisCandidateExecutionAdapter:
    return AnalysisCandidateExecutionAdapter(
        stage_id="track_kinematics",
        invocation_contract=CandidateInvocationContract.TRACK_FLAT_V1,
        computation_mode=CandidateComputationMode.LOGICAL_REMATERIALIZATION,
        runner_status=CandidateRunnerStatus.IMPLEMENTED,
        coordinate_role=CoordinateContractRole.CANONICAL_PRODUCER,
        coordinate_contract_status=CoordinateContractStatus.SOURCE_PRESERVATION_ONLY,
        logical_equality_contract=(
            CandidateLogicalEqualityContract.TRACK_FLAT_PROJECTION_V1
        ),
        runner_module="fisheye.diagnostics.track_kinematics_candidate_execution",
        runner_entrypoint="execute_track_flat_candidate",
        suite_validator_module=(
            "fisheye.analysis_workflows.track_kinematics_candidate_suite"
        ),
        suite_validator_entrypoint="require_track_flat_execution_suite",
    )


def _implemented_eye_angles() -> AnalysisCandidateExecutionAdapter:
    return AnalysisCandidateExecutionAdapter(
        stage_id="eye_angles",
        invocation_contract=CandidateInvocationContract.EYE_ANGLES_V1,
        computation_mode=CandidateComputationMode.SCIENTIFIC_COMPUTE,
        runner_status=CandidateRunnerStatus.IMPLEMENTED,
        coordinate_role=CoordinateContractRole.BOUND_DERIVATIVE,
        coordinate_contract_status=(
            CoordinateContractStatus.BOUND_SOURCE_VALIDATION_IMPLEMENTED
        ),
        logical_equality_contract=(
            CandidateLogicalEqualityContract.EYE_ANGLE_COMPACT_V7_ARRAYS_V1
        ),
        runner_module="fisheye.diagnostics.eye_angle_candidate_execution",
        runner_entrypoint="execute_eye_angle_candidate",
        suite_validator_module=(
            "fisheye.analysis_workflows.eye_angle_candidate_execution"
        ),
        suite_validator_entrypoint="require_eye_angle_execution_suite",
    )


def _implemented_stimulus_epochs() -> AnalysisCandidateExecutionAdapter:
    return AnalysisCandidateExecutionAdapter(
        stage_id="stimulus_epochs",
        invocation_contract=CandidateInvocationContract.STIMULUS_EPOCHS_V1,
        computation_mode=CandidateComputationMode.LOGICAL_REMATERIALIZATION,
        runner_status=CandidateRunnerStatus.IMPLEMENTED,
        coordinate_role=CoordinateContractRole.TEMPORAL_AXIS_ONLY,
        coordinate_contract_status=CoordinateContractStatus.TEMPORAL_AXIS_IMPLEMENTED,
        logical_equality_contract=(
            CandidateLogicalEqualityContract.STIMULUS_EPOCH_V2_ARRAYS_V1
        ),
        runner_module=("fisheye.diagnostics.stimulus_epoch_candidate_execution"),
        runner_entrypoint="execute_stimulus_epoch_candidate",
        suite_validator_module=(
            "fisheye.analysis_workflows.stimulus_epoch_candidate_execution"
        ),
        suite_validator_entrypoint="require_stimulus_epoch_execution_suite",
    )


def _implemented_chaser_distance() -> AnalysisCandidateExecutionAdapter:
    return AnalysisCandidateExecutionAdapter(
        stage_id="chaser_distance",
        invocation_contract=CandidateInvocationContract.CHASER_DISTANCE_BASE_V1,
        computation_mode=CandidateComputationMode.LOGICAL_REMATERIALIZATION,
        runner_status=CandidateRunnerStatus.IMPLEMENTED,
        coordinate_role=CoordinateContractRole.CANONICAL_PRODUCER,
        coordinate_contract_status=CoordinateContractStatus.SOURCE_PRESERVATION_ONLY,
        logical_equality_contract=(
            CandidateLogicalEqualityContract.CHASER_DISTANCE_SEALED_BASE_V2_ARRAYS_V1
        ),
        source_run_parent="analysis/chaser_distance_runs",
        runner_module="fisheye.diagnostics.chaser_distance_candidate_execution",
        runner_entrypoint="execute_chaser_distance_candidate",
        suite_validator_module=(
            "fisheye.analysis_workflows.chaser_distance_candidate_execution"
        ),
        suite_validator_entrypoint="require_chaser_distance_execution_suite",
    )


def _implemented_tail_kinematics() -> AnalysisCandidateExecutionAdapter:
    return AnalysisCandidateExecutionAdapter(
        stage_id="tail_kinematics",
        invocation_contract=CandidateInvocationContract.TAIL_KINEMATICS_V1,
        computation_mode=CandidateComputationMode.SCIENTIFIC_COMPUTE,
        runner_status=CandidateRunnerStatus.IMPLEMENTED,
        coordinate_role=CoordinateContractRole.CANONICAL_PRODUCER,
        coordinate_contract_status=(
            CoordinateContractStatus.CANONICAL_PUBLICATION_IMPLEMENTED
        ),
        logical_equality_contract=(
            CandidateLogicalEqualityContract.TAIL_KINEMATICS_DECLARED_ARRAYS_V1
        ),
        runner_module="fisheye.diagnostics.tail_kinematics_candidate_execution",
        runner_entrypoint="execute_tail_kinematics_candidate",
        suite_validator_module=(
            "fisheye.analysis_workflows.tail_kinematics_candidate_execution"
        ),
        suite_validator_entrypoint="require_tail_kinematics_execution_suite",
    )


def _implemented_subject_shape() -> AnalysisCandidateExecutionAdapter:
    return AnalysisCandidateExecutionAdapter(
        stage_id="subject_shape",
        invocation_contract=CandidateInvocationContract.SUBJECT_SHAPE_V1,
        computation_mode=CandidateComputationMode.SCIENTIFIC_COMPUTE,
        runner_status=CandidateRunnerStatus.IMPLEMENTED,
        coordinate_role=CoordinateContractRole.CANONICAL_PRODUCER,
        coordinate_contract_status=(
            CoordinateContractStatus.CANONICAL_PUBLICATION_IMPLEMENTED
        ),
        logical_equality_contract=(
            CandidateLogicalEqualityContract.SUBJECT_SHAPE_V4_ARRAYS_V1
        ),
        runner_module="fisheye.diagnostics.subject_shape_candidate_execution",
        runner_entrypoint="execute_subject_shape_candidate",
        suite_validator_module=(
            "fisheye.analysis_workflows.subject_shape_candidate_execution"
        ),
        suite_validator_entrypoint="require_subject_shape_execution_suite",
    )


def _implemented_bout_classification() -> AnalysisCandidateExecutionAdapter:
    return AnalysisCandidateExecutionAdapter(
        stage_id="bout_classification",
        invocation_contract=CandidateInvocationContract.BOUT_CLASSIFICATION_V1,
        computation_mode=CandidateComputationMode.LOGICAL_REMATERIALIZATION,
        runner_status=CandidateRunnerStatus.IMPLEMENTED,
        coordinate_role=CoordinateContractRole.BOUND_DERIVATIVE,
        coordinate_contract_status=(
            CoordinateContractStatus.BOUND_SOURCE_VALIDATION_IMPLEMENTED
        ),
        logical_equality_contract=(
            CandidateLogicalEqualityContract.BOUT_CLASSIFICATION_V2_ARRAYS_V1
        ),
        runner_module=("fisheye.diagnostics.bout_classification_candidate_execution"),
        runner_entrypoint="execute_bout_classification_candidate",
        suite_validator_module=(
            "fisheye.analysis_workflows." "bout_classification_candidate_execution"
        ),
        suite_validator_entrypoint=("require_bout_classification_execution_suite"),
    )


def _implemented_stimulus_response() -> AnalysisCandidateExecutionAdapter:
    return AnalysisCandidateExecutionAdapter(
        stage_id="stimulus_response",
        invocation_contract=CandidateInvocationContract.STIMULUS_RESPONSE_V1,
        computation_mode=CandidateComputationMode.SCIENTIFIC_COMPUTE,
        runner_status=CandidateRunnerStatus.IMPLEMENTED,
        coordinate_role=CoordinateContractRole.BOUND_DERIVATIVE,
        coordinate_contract_status=(
            CoordinateContractStatus.BOUND_SOURCE_VALIDATION_IMPLEMENTED
        ),
        logical_equality_contract=(
            CandidateLogicalEqualityContract.STIMULUS_RESPONSE_V3_ARRAYS_V1
        ),
        runner_module=("fisheye.diagnostics.stimulus_response_candidate_execution"),
        runner_entrypoint="execute_stimulus_response_candidate",
        suite_validator_module=(
            "fisheye.analysis_workflows.stimulus_response_candidate_execution"
        ),
        suite_validator_entrypoint=("require_stimulus_response_execution_suite"),
    )


def _implemented_tail_posture() -> AnalysisCandidateExecutionAdapter:
    return AnalysisCandidateExecutionAdapter(
        stage_id="tail_posture_view",
        invocation_contract=CandidateInvocationContract.TAIL_POSTURE_V1,
        computation_mode=CandidateComputationMode.SCIENTIFIC_COMPUTE,
        runner_status=CandidateRunnerStatus.IMPLEMENTED,
        coordinate_role=CoordinateContractRole.CANONICAL_PRODUCER,
        coordinate_contract_status=(
            CoordinateContractStatus.CANONICAL_PUBLICATION_IMPLEMENTED
        ),
        logical_equality_contract=(
            CandidateLogicalEqualityContract.TAIL_POSTURE_V3_ARRAYS_V1
        ),
        runner_module="fisheye.diagnostics.tail_posture_candidate_execution",
        runner_entrypoint="execute_tail_posture_candidate",
        suite_validator_module=(
            "fisheye.analysis_workflows.tail_posture_candidate_execution"
        ),
        suite_validator_entrypoint="require_tail_posture_execution_suite",
    )


ANALYSIS_CANDIDATE_EXECUTION_ADAPTERS: tuple[AnalysisCandidateExecutionAdapter, ...] = (
    _implemented_track_flat(),
    _implemented_exact_tabular(
        "swim_bouts",
        CandidateLogicalEqualityContract.SWIM_BOUTS_DECLARED_ARRAYS_V1,
    ),
    _implemented_exact_tabular(
        "bout_kinematics",
        CandidateLogicalEqualityContract.BOUT_KINEMATICS_DECLARED_ARRAYS_V1,
    ),
    _implemented_eye_angles(),
    _implemented_stimulus_epochs(),
    _implemented_subject_shape(),
    _implemented_tail_kinematics(),
    _implemented_stimulus_response(),
    _implemented_occupancy(
        "detection_occupancy",
        CandidateLogicalEqualityContract.DETECTION_OCCUPANCY_DECLARED_ARRAYS_V1,
    ),
    _implemented_occupancy(
        "session_occupancy",
        CandidateLogicalEqualityContract.SESSION_OCCUPANCY_DECLARED_ARRAYS_V1,
    ),
    _implemented_chaser_distance(),
    _implemented_tail_posture(),
    _implemented_bout_classification(),
)


ANALYSIS_CANDIDATE_EXECUTION_ADAPTER_BY_STAGE = {
    adapter.stage_id: adapter for adapter in ANALYSIS_CANDIDATE_EXECUTION_ADAPTERS
}

if set(ANALYSIS_CANDIDATE_EXECUTION_ADAPTER_BY_STAGE) != set(
    DERIVED_ANALYSIS_STORAGE_CANDIDATE_BY_STAGE
):
    raise RuntimeError("execution-adapter catalog must cover every storage candidate")


def resolved_candidate_execution_adapters() -> tuple[dict[str, object], ...]:
    return tuple(
        adapter.as_manifest() for adapter in ANALYSIS_CANDIDATE_EXECUTION_ADAPTERS
    )


__all__ = [
    "ANALYSIS_CANDIDATE_EXECUTION_ADAPTER_BY_STAGE",
    "ANALYSIS_CANDIDATE_EXECUTION_ADAPTERS",
    "AnalysisCandidateExecutionAdapter",
    "CandidateInvocationContract",
    "resolved_candidate_execution_adapters",
]
