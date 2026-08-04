"""Validated, normalized evidence from heterogeneous analytics read matrices.

The family benchmark runners intentionally retain their exact scientific
schemas.  Promotion review, however, needs one truthful projection of their
common identity and evidence boundaries.  This module validates a matrix
through its catalog-owned validator and then normalizes only facts that the
matrix actually proves.  Missing consumer or physical-I/O evidence remains
explicitly unavailable or unrecorded; it is never inferred from a successful
read matrix.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from importlib import import_module
import json
from pathlib import PurePosixPath
from typing import Any, Mapping

from fisheye.shared.zarr.manifest_digest import canonical_json_sha256

from .storage_benchmark_catalog import (
    DERIVED_ANALYSIS_STORAGE_BENCHMARK_BY_STAGE,
)
from .storage_candidate_catalog import (
    DERIVED_ANALYSIS_STORAGE_CANDIDATE_BY_STAGE,
)
from .storage_contract_catalog import (
    DERIVED_ANALYSIS_STORAGE_CONTRACT_BY_STAGE,
)


class BenchmarkEvidenceState(str, Enum):
    """A four-state evidence result that does not collapse missing evidence."""

    PASSED = "passed"
    FAILED = "failed"
    UNAVAILABLE = "unavailable"
    NOT_RECORDED = "not_recorded"


@dataclass(frozen=True)
class StorageBenchmarkIdentity:
    """Common evidence identity extracted from one validated family matrix."""

    stage_id: str
    family_id: str
    matrix_schema_id: str
    matrix_schema_version: int
    benchmark_id: str
    archive_path: str
    source_run_name: str
    candidate_run_name: str
    source_run_path: str
    candidate_run_path: str
    balanced_repetitions: BenchmarkEvidenceState
    decoded_equality: BenchmarkEvidenceState
    metadata_equivalence: BenchmarkEvidenceState
    physical_io: BenchmarkEvidenceState
    palette_source_consumer: BenchmarkEvidenceState
    palette_candidate_consumer: BenchmarkEvidenceState
    crimson_consumer: BenchmarkEvidenceState
    promotion_gate: BenchmarkEvidenceState

    def as_record(self) -> dict[str, object]:
        return {
            "stage_id": self.stage_id,
            "family_id": self.family_id,
            "matrix_schema_id": self.matrix_schema_id,
            "matrix_schema_version": self.matrix_schema_version,
            "benchmark_id": self.benchmark_id,
            "archive_path": self.archive_path,
            "source_run_name": self.source_run_name,
            "candidate_run_name": self.candidate_run_name,
            "source_run_path": self.source_run_path,
            "candidate_run_path": self.candidate_run_path,
            "balanced_repetitions": self.balanced_repetitions.value,
            "decoded_equality": self.decoded_equality.value,
            "metadata_equivalence": self.metadata_equivalence.value,
            "physical_io": self.physical_io.value,
            "palette_source_consumer": self.palette_source_consumer.value,
            "palette_candidate_consumer": self.palette_candidate_consumer.value,
            "crimson_consumer": self.crimson_consumer.value,
            "promotion_gate": self.promotion_gate.value,
        }


_EXACT_TABULAR = frozenset(
    {
        "swim_bouts",
        "bout_kinematics",
        "detection_occupancy",
        "session_occupancy",
    }
)


def _mapping(value: object, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be one object")
    return value


def _text(value: object, *, label: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise ValueError(f"{label} must be one nonempty exact string")
    return value


def _state(value: object) -> BenchmarkEvidenceState:
    if value is True:
        return BenchmarkEvidenceState.PASSED
    if value is False:
        return BenchmarkEvidenceState.FAILED
    raise ValueError("recorded evidence state must be one exact bool")


def _run_path(parent: str, name: str) -> str:
    path = PurePosixPath(parent) / name
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError("benchmark run path is not canonical")
    return path.as_posix()


def _require_envelope(
    stage_id: str,
    matrix: Mapping[str, Any],
) -> Mapping[str, Any]:
    record = DERIVED_ANALYSIS_STORAGE_BENCHMARK_BY_STAGE.get(stage_id)
    if record is None:
        raise ValueError(f"unknown analytics benchmark stage {stage_id!r}")
    if not isinstance(matrix, Mapping) or set(matrix) != {
        "schema_id",
        "schema_version",
        "payload",
        "payload_digest",
    }:
        raise ValueError("benchmark matrix envelope field set differs")
    if (
        matrix["schema_id"] != record.matrix_schema_id
        or matrix["schema_version"] != record.matrix_schema_version
    ):
        raise ValueError("benchmark matrix schema differs from its catalog entry")
    payload = _mapping(matrix["payload"], label="benchmark matrix payload")
    if matrix["payload_digest"] != canonical_json_sha256(payload):
        raise ValueError("benchmark matrix payload digest differs")
    try:
        json.dumps(matrix, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"benchmark matrix is not strict JSON: {exc}") from exc
    return payload


def _minimum_identity(
    stage_id: str,
    payload: Mapping[str, Any],
) -> tuple[str, str, str, str, str, str]:
    """Return family, archive, source/candidate names, and exact run paths."""

    contract = DERIVED_ANALYSIS_STORAGE_CONTRACT_BY_STAGE[stage_id]
    candidate = DERIVED_ANALYSIS_STORAGE_CANDIDATE_BY_STAGE[stage_id]
    if stage_id in _EXACT_TABULAR:
        family = _text(payload.get("family_id"), label="matrix family_id")
        archive = _text(payload.get("archive_path"), label="matrix archive_path")
        source = _text(payload.get("source_run_name"), label="source run")
        target = _text(payload.get("candidate_run_name"), label="candidate run")
        source_path = _run_path(contract.run_parent, source)
        target_path = _run_path(candidate.run_parent, target)
    elif stage_id == "chaser_distance":
        family = _text(payload.get("family_id"), label="matrix family_id")
        archive = _text(payload.get("archive_path"), label="matrix archive_path")
        source = _text(payload.get("source_run_name"), label="source run")
        target = _text(payload.get("candidate_run_name"), label="candidate run")
        source_path = _run_path(
            _text(payload.get("source_parent_path"), label="source parent"),
            source,
        )
        target_path = _run_path(
            _text(payload.get("candidate_parent_path"), label="candidate parent"),
            target,
        )
    elif stage_id == "stimulus_epochs":
        module_name = DERIVED_ANALYSIS_STORAGE_BENCHMARK_BY_STAGE[stage_id].adapter_module
        if module_name is None:
            raise ValueError("stimulus-epoch benchmark adapter is absent")
        family = _text(
            getattr(import_module(module_name), "FAMILY_ID", None),
            label="matrix family_id",
        )
        archive = _text(payload.get("archive_path"), label="matrix archive_path")
        source = _text(payload.get("source_run_name"), label="source run")
        target = _text(payload.get("candidate_run_name"), label="candidate run")
        workload = _mapping(payload.get("workload"), label="matrix workload")
        workload_payload = _mapping(
            workload.get("payload"), label="matrix workload payload"
        )
        source_path = _text(
            workload_payload.get("source_run_path"), label="source run path"
        )
        target_path = _text(
            workload_payload.get("candidate_run_path"), label="candidate run path"
        )
    elif stage_id in {"stimulus_response", "eye_angles"}:
        family = _text(payload.get("family_id"), label="matrix family_id")
        archive = _text(payload.get("archive"), label="matrix archive")
        source = _text(payload.get("source_run"), label="source run")
        target = _text(payload.get("candidate_run"), label="candidate run")
        workload = _mapping(payload.get("workload"), label="matrix workload")
        workload_payload = _mapping(
            workload.get("payload"), label="matrix workload payload"
        )
        source_path = _text(
            workload_payload.get("source_run_path"), label="source run path"
        )
        target_path = _text(
            workload_payload.get("candidate_run_path"), label="candidate run path"
        )
    elif stage_id == "track_kinematics":
        family = _text(payload.get("family_id"), label="matrix family_id")
        archive = _text(payload.get("archive_path"), label="matrix archive_path")
        source = _text(payload.get("source_run_name"), label="source run")
        target = _text(payload.get("candidate_run_name"), label="candidate run")
        pair = _mapping(payload.get("pair_validation"), label="pair validation")
        source_path = _text(pair.get("source_run_path"), label="source run path")
        target_path = _text(pair.get("candidate_run_path"), label="candidate run path")
    elif stage_id == "tail_kinematics":
        family = _text(payload.get("family_id"), label="matrix family_id")
        archive = _text(payload.get("archive"), label="matrix archive")
        source = _text(payload.get("source_run_name"), label="source run")
        target = _text(payload.get("candidate_run_name"), label="candidate run")
        trials = payload.get("trials")
        if not isinstance(trials, list) or not trials:
            raise ValueError("tail-kinematics matrix has no trials")
        trial_payload = _mapping(trials[0].get("payload"), label="trial payload")
        pair = _mapping(trial_payload.get("pair_validation"), label="pair validation")
        pair_payload = _mapping(pair.get("payload"), label="pair payload")
        source_path = _text(
            pair_payload.get("source_run_path"), label="source run path"
        )
        target_path = _text(
            pair_payload.get("candidate_run_path"), label="candidate run path"
        )
    elif stage_id == "subject_shape":
        family = _text(payload.get("family_id"), label="matrix family_id")
        archive = _text(payload.get("archive_path"), label="matrix archive_path")
        source = _text(payload.get("source_run_name"), label="source run")
        target = _text(payload.get("candidate_run_name"), label="candidate run")
        parent = _text(payload.get("parent_path"), label="matrix parent_path")
        source_path = _run_path(parent, source)
        target_path = _run_path(parent, target)
    elif stage_id == "tail_posture_view":
        family = _text(payload.get("family_id"), label="matrix family_id")
        archive_identity = _mapping(
            payload.get("archive_identity"), label="archive identity"
        )
        archive = _text(
            archive_identity.get("resolved_path"), label="matrix archive path"
        )
        source = _text(payload.get("source_run"), label="source run")
        target = _text(payload.get("candidate_run"), label="candidate run")
        source_path = _run_path(contract.run_parent, source)
        target_path = _run_path(candidate.run_parent, target)
    elif stage_id == "bout_classification":
        family = _text(payload.get("family_id"), label="matrix family_id")
        archive = _text(payload.get("archive_path"), label="matrix archive_path")
        source = _text(payload.get("source_run_name"), label="source run")
        target = _text(payload.get("candidate_run_name"), label="candidate run")
        pair = _mapping(payload.get("pair_validation"), label="pair validation")
        pair_payload = _mapping(pair.get("payload"), label="pair payload")
        source_path = _text(
            pair_payload.get("source_run_path"), label="source run path"
        )
        target_path = _text(
            pair_payload.get("candidate_run_path"), label="candidate run path"
        )
    else:  # pragma: no cover - exhaustive catalog guard
        raise AssertionError(f"unhandled benchmark stage {stage_id!r}")
    adapter_module = DERIVED_ANALYSIS_STORAGE_BENCHMARK_BY_STAGE[
        stage_id
    ].adapter_module
    declared_family = (
        getattr(import_module(adapter_module), "FAMILY_ID", None)
        if adapter_module is not None
        else None
    )
    expected_family = stage_id if declared_family is None else declared_family
    if family != expected_family:
        raise ValueError("benchmark matrix family differs from its catalog stage")
    if source == target or source_path == target_path:
        raise ValueError("benchmark source and candidate identities must differ")
    return family, archive, source, target, source_path, target_path


def validate_storage_benchmark_matrix(
    stage_id: str,
    matrix: Mapping[str, Any],
) -> None:
    """Run the stage-owned deep validator, including live replay where required."""

    payload = _require_envelope(stage_id, matrix)
    _family, archive, source, target, _source_path, _target_path = _minimum_identity(
        stage_id, payload
    )
    record = DERIVED_ANALYSIS_STORAGE_BENCHMARK_BY_STAGE[stage_id]
    if record.validator_module is None or record.validator_entrypoint is None:
        raise ValueError("benchmark stage has no matrix validator")
    validator = getattr(
        import_module(record.validator_module), record.validator_entrypoint, None
    )
    if not callable(validator):
        raise ValueError("benchmark matrix validator does not resolve")
    if stage_id == "tail_posture_view":
        validator(
            matrix,
            archive=archive,
            source_run=source,
            candidate_run=target,
        )
    else:
        validator(matrix)


def _balanced_state(
    stage_id: str,
    payload: Mapping[str, Any],
) -> BenchmarkEvidenceState:
    field = {
        **{stage: "balanced_read_matrix_complete" for stage in _EXACT_TABULAR},
        "stimulus_epochs": "balanced_fresh_process_matrix_complete",
        "stimulus_response": "balanced_fresh_process_matrix_complete",
        "eye_angles": "balanced_fresh_process_matrix_complete",
        "track_kinematics": "balanced_fresh_process_matrix_complete",
        "bout_classification": "balanced_read_matrix_complete",
    }.get(stage_id)
    if field is not None:
        return _state(payload.get(field))
    module_name = DERIVED_ANALYSIS_STORAGE_BENCHMARK_BY_STAGE[stage_id].adapter_module
    if module_name is None:
        return BenchmarkEvidenceState.NOT_RECORDED
    expected = getattr(import_module(module_name), "DEFAULT_REPETITIONS", None)
    repetitions = payload.get("repetitions")
    if type(expected) is not int or type(repetitions) is not int:
        return BenchmarkEvidenceState.NOT_RECORDED
    return _state(repetitions == expected)


def _evidence_states(
    stage_id: str,
    payload: Mapping[str, Any],
) -> tuple[
    BenchmarkEvidenceState,
    BenchmarkEvidenceState,
    BenchmarkEvidenceState,
    BenchmarkEvidenceState,
    BenchmarkEvidenceState,
]:
    """Return decoded, metadata, physical, Palette-source, Palette-candidate."""

    unavailable = BenchmarkEvidenceState.UNAVAILABLE
    absent = BenchmarkEvidenceState.NOT_RECORDED
    if stage_id in _EXACT_TABULAR:
        correctness = _mapping(payload.get("correctness"), label="correctness")
        return (
            _state(correctness.get("logical_equality")),
            _state(correctness.get("direct_consolidated_metadata_equivalence")),
            unavailable,
            absent,
            absent,
        )
    if stage_id in {"chaser_distance", "subject_shape"}:
        correctness = _mapping(payload.get("correctness"), label="correctness")
        return (
            _state(
                correctness.get("full_decoded_logical_equality") is True
                and correctness.get("primary_access_decoded_equality") is True
            ),
            _state(correctness.get("direct_consolidated_metadata_equivalence")),
            unavailable,
            absent,
            absent,
        )
    if stage_id == "stimulus_epochs":
        correctness = _mapping(payload.get("correctness"), label="correctness")
        return (
            _state(
                correctness.get("complete_decoded_array_equality") is True
                and correctness.get("decoded_segment_equality") is True
            ),
            _state(correctness.get("direct_consolidated_metadata_equivalence")),
            unavailable,
            BenchmarkEvidenceState.PASSED,
            BenchmarkEvidenceState.PASSED,
        )
    if stage_id == "stimulus_response":
        correctness = _mapping(payload.get("correctness"), label="correctness")
        return (
            _state(
                correctness.get("decoded_arrays_equal") is True
                and correctness.get("logical_reader_equal") is True
            ),
            _state(correctness.get("direct_consolidated_equal")),
            unavailable,
            BenchmarkEvidenceState.PASSED,
            BenchmarkEvidenceState.PASSED,
        )
    if stage_id == "eye_angles":
        correctness = _mapping(payload.get("correctness"), label="correctness")
        return (
            _state(
                correctness.get("decoded_arrays_equal") is True
                and correctness.get(
                    "source_public_reader_candidate_diagnostic_adapter_equal"
                )
                is True
            ),
            _state(correctness.get("direct_consolidated_equal")),
            unavailable,
            BenchmarkEvidenceState.PASSED,
            unavailable,
        )
    if stage_id == "track_kinematics":
        pair = _mapping(payload.get("pair_validation"), label="pair validation")
        return (
            _state(pair.get("complete_decoded_equality")),
            BenchmarkEvidenceState.PASSED,
            unavailable,
            _state(payload.get("public_source_consumer_implemented")),
            (
                BenchmarkEvidenceState.PASSED
                if payload.get("public_candidate_consumer_implemented") is True
                else unavailable
            ),
        )
    if stage_id == "tail_kinematics":
        return (
            BenchmarkEvidenceState.PASSED,
            BenchmarkEvidenceState.PASSED,
            unavailable,
            BenchmarkEvidenceState.PASSED,
            unavailable,
        )
    if stage_id == "tail_posture_view":
        pair = _mapping(payload.get("pair_validation"), label="pair validation")
        pair_payload = _mapping(pair.get("payload"), label="pair payload")
        logical = _mapping(
            pair_payload.get("logical_equality"), label="logical equality"
        )
        return (
            _state(logical.get("all_equal")),
            BenchmarkEvidenceState.PASSED,
            unavailable,
            BenchmarkEvidenceState.PASSED,
            unavailable,
        )
    if stage_id == "bout_classification":
        correctness = _mapping(payload.get("correctness"), label="correctness")
        return (
            _state(correctness.get("complete_decoded_equality")),
            _state(correctness.get("direct_consolidated_metadata_equivalence")),
            unavailable,
            BenchmarkEvidenceState.PASSED,
            unavailable,
        )
    raise AssertionError(f"unhandled benchmark stage {stage_id!r}")


def extract_storage_benchmark_identity(
    stage_id: str,
    matrix: Mapping[str, Any],
) -> StorageBenchmarkIdentity:
    """Validate and normalize one family matrix without inventing evidence."""

    payload = _require_envelope(stage_id, matrix)
    validate_storage_benchmark_matrix(stage_id, matrix)
    family, archive, source, target, source_path, target_path = _minimum_identity(
        stage_id, payload
    )
    decoded, metadata, physical, palette_source, palette_candidate = _evidence_states(
        stage_id, payload
    )
    record = DERIVED_ANALYSIS_STORAGE_BENCHMARK_BY_STAGE[stage_id]
    return StorageBenchmarkIdentity(
        stage_id=stage_id,
        family_id=family,
        matrix_schema_id=str(record.matrix_schema_id),
        matrix_schema_version=int(record.matrix_schema_version),
        benchmark_id=_text(payload.get("benchmark_id"), label="benchmark_id"),
        archive_path=archive,
        source_run_name=source,
        candidate_run_name=target,
        source_run_path=source_path,
        candidate_run_path=target_path,
        balanced_repetitions=_balanced_state(stage_id, payload),
        decoded_equality=decoded,
        metadata_equivalence=metadata,
        physical_io=physical,
        palette_source_consumer=palette_source,
        palette_candidate_consumer=palette_candidate,
        crimson_consumer=BenchmarkEvidenceState.NOT_RECORDED,
        promotion_gate=BenchmarkEvidenceState.NOT_RECORDED,
    )


__all__ = [
    "BenchmarkEvidenceState",
    "StorageBenchmarkIdentity",
    "extract_storage_benchmark_identity",
    "validate_storage_benchmark_matrix",
]
