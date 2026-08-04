from __future__ import annotations

from dataclasses import replace
from importlib import import_module
import json

import pytest

from fisheye.analysis_workflows.storage_benchmark_catalog import (
    DERIVED_ANALYSIS_STORAGE_BENCHMARK_BY_STAGE,
    DERIVED_ANALYSIS_STORAGE_BENCHMARKS,
    StorageBenchmarkAdapterStatus,
    resolved_storage_benchmarks,
)
from fisheye.analysis_workflows.storage_candidate_catalog import (
    DERIVED_ANALYSIS_STORAGE_CANDIDATE_BY_STAGE,
)

READ_MATRIX_STAGES = {
    "swim_bouts",
    "bout_kinematics",
    "detection_occupancy",
    "session_occupancy",
    "chaser_distance",
    "stimulus_epochs",
    "stimulus_response",
    "eye_angles",
    "track_kinematics",
    "tail_kinematics",
    "subject_shape",
    "tail_posture_view",
    "bout_classification",
}


def test_benchmark_catalog_covers_every_candidate_without_claiming_promotion() -> None:
    assert set(DERIVED_ANALYSIS_STORAGE_BENCHMARK_BY_STAGE) == set(
        DERIVED_ANALYSIS_STORAGE_CANDIDATE_BY_STAGE
    )
    assert len(DERIVED_ANALYSIS_STORAGE_BENCHMARKS) == len(
        DERIVED_ANALYSIS_STORAGE_BENCHMARK_BY_STAGE
    )
    assert {
        stage_id
        for stage_id, record in DERIVED_ANALYSIS_STORAGE_BENCHMARK_BY_STAGE.items()
        if record.adapter_status
        is StorageBenchmarkAdapterStatus.READ_MATRIX_IMPLEMENTED
    } == READ_MATRIX_STAGES
    assert all(
        not record.benchmark_coverage_complete
        for record in DERIVED_ANALYSIS_STORAGE_BENCHMARKS
    )
    assert all(
        record.crimson_consumer_required
        for record in DERIVED_ANALYSIS_STORAGE_BENCHMARKS
    )


def test_implemented_read_matrices_are_executable_and_truthful() -> None:
    for stage_id in READ_MATRIX_STAGES:
        record = DERIVED_ANALYSIS_STORAGE_BENCHMARK_BY_STAGE[stage_id]
        assert record.resolves_adapter()
        assert record.resolves_validator()
        assert isinstance(record.matrix_schema_id, str)
        assert record.matrix_schema_id.startswith("palette.")
        assert isinstance(record.matrix_schema_version, int)
        assert record.reader_workload_implemented is True
        assert record.decoded_equality_implemented is True
        assert record.metadata_equivalence_implemented is True
        assert record.writer_phase_measured is False
        assert record.publication_phase_measured is False
        assert record.physical_io_measured is False
        assert record.palette_consumer_implemented is False
        assert record.crimson_consumer_required is True
        assert record.crimson_consumer_implemented is False
        assert record.representative_short_executed is False
        assert record.representative_full_executed is False
        assert record.evidence_receipt_path is None
        assert record.evidence_receipt_sha256 is None
        assert record.gate_contract_id is None
        assert record.gate_contract_version is None
        assert record.gate_passed is False

    chaser = DERIVED_ANALYSIS_STORAGE_BENCHMARK_BY_STAGE["chaser_distance"]
    assert chaser.adapter_module == (
        "fisheye.diagnostics.benchmark_chaser_distance_base_candidate"
    )
    assert chaser.adapter_entrypoint == "run_benchmark_matrix"
    stimulus_epochs = DERIVED_ANALYSIS_STORAGE_BENCHMARK_BY_STAGE["stimulus_epochs"]
    assert stimulus_epochs.adapter_module == (
        "fisheye.diagnostics.benchmark_stimulus_epoch_reads"
    )
    assert stimulus_epochs.adapter_entrypoint == "run_benchmark_matrix"
    assert stimulus_epochs.validator_entrypoint == "require_matrix_result"
    assert stimulus_epochs.matrix_schema_id == "palette.stimulus_epoch.read_matrix"
    assert stimulus_epochs.matrix_schema_version == 1
    stimulus_response = DERIVED_ANALYSIS_STORAGE_BENCHMARK_BY_STAGE["stimulus_response"]
    assert stimulus_response.adapter_module == (
        "fisheye.diagnostics.benchmark_stimulus_response_reads"
    )
    assert stimulus_response.adapter_entrypoint == "run_benchmark_matrix"
    eye_angles = DERIVED_ANALYSIS_STORAGE_BENCHMARK_BY_STAGE["eye_angles"]
    assert eye_angles.adapter_module == (
        "fisheye.diagnostics.benchmark_eye_angle_v7_reads"
    )
    assert eye_angles.adapter_entrypoint == "run_benchmark_matrix"
    track_kinematics = DERIVED_ANALYSIS_STORAGE_BENCHMARK_BY_STAGE["track_kinematics"]
    assert track_kinematics.adapter_module == (
        "fisheye.diagnostics.benchmark_track_kinematics_v2_candidate"
    )
    assert track_kinematics.adapter_entrypoint == "run_benchmark_matrix"
    tail_kinematics = DERIVED_ANALYSIS_STORAGE_BENCHMARK_BY_STAGE["tail_kinematics"]
    assert tail_kinematics.adapter_module == (
        "fisheye.diagnostics.benchmark_tail_kinematics_candidate_reads"
    )
    assert tail_kinematics.adapter_entrypoint == "run_matrix"
    assert tail_kinematics.validator_entrypoint == "validate_matrix"
    subject_shape = DERIVED_ANALYSIS_STORAGE_BENCHMARK_BY_STAGE["subject_shape"]
    assert subject_shape.adapter_module == (
        "fisheye.diagnostics.benchmark_subject_shape_v4_candidate"
    )
    assert subject_shape.adapter_entrypoint == "run_benchmark_matrix"
    tail_posture = DERIVED_ANALYSIS_STORAGE_BENCHMARK_BY_STAGE["tail_posture_view"]
    assert tail_posture.adapter_module == (
        "fisheye.diagnostics.benchmark_tail_posture_view_v3_candidate"
    )
    assert tail_posture.adapter_entrypoint == "run_benchmark_matrix"
    assert tail_posture.validator_entrypoint == "validate_matrix_evidence"
    bout_classification = DERIVED_ANALYSIS_STORAGE_BENCHMARK_BY_STAGE[
        "bout_classification"
    ]
    assert bout_classification.adapter_module == (
        "fisheye.diagnostics.benchmark_bout_classification_v2_reads"
    )
    assert bout_classification.adapter_entrypoint == "run_benchmark_matrix"


def test_cataloged_matrix_schemas_equal_their_owning_modules() -> None:
    for record in DERIVED_ANALYSIS_STORAGE_BENCHMARKS:
        assert record.adapter_module is not None
        module = import_module(record.adapter_module)
        version = getattr(
            module,
            "MATRIX_SCHEMA_VERSION",
            getattr(module, "SCHEMA_VERSION", None),
        )
        assert record.matrix_schema_id == module.MATRIX_SCHEMA_ID
        assert record.matrix_schema_version == version


def test_all_cataloged_families_now_have_read_matrices() -> None:
    assert READ_MATRIX_STAGES == set(DERIVED_ANALYSIS_STORAGE_BENCHMARK_BY_STAGE)
    assert all(
        record.adapter_status is StorageBenchmarkAdapterStatus.READ_MATRIX_IMPLEMENTED
        for record in DERIVED_ANALYSIS_STORAGE_BENCHMARKS
    )


def test_resolved_records_are_strict_json() -> None:
    records = resolved_storage_benchmarks()
    assert [record["stage_id"] for record in records] == list(
        DERIVED_ANALYSIS_STORAGE_CANDIDATE_BY_STAGE
    )
    assert json.loads(json.dumps(records)) == list(records)


@pytest.mark.parametrize(
    ("changes", "error"),
    (
        ({"stage_id": "unknown_stage"}, "own one storage candidate"),
        ({"stage_id": "bad stage"}, "canonical identifier"),
        ({"adapter_status": "implemented"}, "adapter_status"),
        ({"writer_phase_measured": 1}, "exact bool"),
        ({"adapter_module": "bad module"}, "exact runner and validator"),
        ({"adapter_entrypoint": "bad-entry"}, "exact runner and validator"),
        ({"validator_module": "bad module"}, "exact runner and validator"),
        ({"validator_entrypoint": "bad-entry"}, "exact runner and validator"),
        ({"matrix_schema_id": "bad schema"}, "exact runner and validator"),
        ({"matrix_schema_version": 0}, "exact runner and validator"),
        ({"reader_workload_implemented": False}, "reader workload"),
        ({"writer_phase_measured": True}, "immutable evidence receipt"),
    ),
)
def test_implemented_declaration_fails_closed(
    changes: dict[str, object], error: str
) -> None:
    base = DERIVED_ANALYSIS_STORAGE_BENCHMARK_BY_STAGE["swim_bouts"]
    with pytest.raises((TypeError, ValueError), match=error):
        replace(base, **changes)


def test_plan_only_declaration_cannot_claim_adapter_or_execution() -> None:
    implemented = DERIVED_ANALYSIS_STORAGE_BENCHMARK_BY_STAGE["tail_kinematics"]
    base = replace(
        implemented,
        adapter_status=StorageBenchmarkAdapterStatus.PLAN_ONLY,
        adapter_module=None,
        adapter_entrypoint=None,
        validator_module=None,
        validator_entrypoint=None,
        matrix_schema_id=None,
        matrix_schema_version=None,
        reader_workload_implemented=False,
        decoded_equality_implemented=False,
        metadata_equivalence_implemented=False,
    )
    assert base.resolves_adapter() is False
    with pytest.raises(ValueError, match="must not claim an adapter or validator"):
        replace(base, adapter_module="fisheye.diagnostics.some_benchmark")
    with pytest.raises(ValueError, match="must not claim an adapter or validator"):
        replace(
            base,
            validator_module="fisheye.diagnostics.some_benchmark",
            validator_entrypoint="require_matrix_result",
        )
    with pytest.raises(ValueError, match="requires an implemented adapter"):
        replace(base, physical_io_measured=True)


def test_complete_coverage_requires_crimson_when_declared() -> None:
    base = DERIVED_ANALYSIS_STORAGE_BENCHMARK_BY_STAGE["swim_bouts"]
    completed = replace(
        base,
        writer_phase_measured=True,
        publication_phase_measured=True,
        physical_io_measured=True,
        palette_consumer_implemented=True,
        representative_short_executed=True,
        representative_full_executed=True,
        evidence_receipt_path="docs/diagnostics/swim_bouts/matrix.json",
        evidence_receipt_sha256="a" * 64,
        gate_contract_id="swim_bouts_storage_gate",
        gate_contract_version=1,
        gate_passed=True,
    )
    assert completed.crimson_consumer_required is True
    assert completed.crimson_consumer_implemented is False
    assert completed.benchmark_coverage_complete is False
    assert (
        replace(
            completed,
            crimson_consumer_implemented=True,
        ).benchmark_coverage_complete
        is True
    )


def test_unbound_or_partial_evidence_receipts_fail_closed() -> None:
    base = DERIVED_ANALYSIS_STORAGE_BENCHMARK_BY_STAGE["swim_bouts"]
    with pytest.raises(ValueError, match="evidence bindings require"):
        replace(base, evidence_receipt_path="evidence.json")
    with pytest.raises(ValueError, match="immutable evidence receipt"):
        replace(
            base,
            writer_phase_measured=True,
            evidence_receipt_path="evidence.json",
            evidence_receipt_sha256="not-a-digest",
            gate_contract_id="storage_gate",
            gate_contract_version=1,
        )
