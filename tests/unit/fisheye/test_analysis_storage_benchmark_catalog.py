from __future__ import annotations

from dataclasses import replace
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
    "subject_shape",
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
    stimulus_response = DERIVED_ANALYSIS_STORAGE_BENCHMARK_BY_STAGE[
        "stimulus_response"
    ]
    assert stimulus_response.adapter_module == (
        "fisheye.diagnostics.benchmark_stimulus_response_reads"
    )
    assert stimulus_response.adapter_entrypoint == "run_benchmark_matrix"
    eye_angles = DERIVED_ANALYSIS_STORAGE_BENCHMARK_BY_STAGE["eye_angles"]
    assert eye_angles.adapter_module == (
        "fisheye.diagnostics.benchmark_eye_angle_v7_reads"
    )
    assert eye_angles.adapter_entrypoint == "run_benchmark_matrix"
    subject_shape = DERIVED_ANALYSIS_STORAGE_BENCHMARK_BY_STAGE["subject_shape"]
    assert subject_shape.adapter_module == (
        "fisheye.diagnostics.benchmark_subject_shape_v4_candidate"
    )
    assert subject_shape.adapter_entrypoint == "run_benchmark_matrix"


def test_plan_only_families_have_no_fabricated_adapter_or_execution_evidence() -> None:
    for stage_id, record in DERIVED_ANALYSIS_STORAGE_BENCHMARK_BY_STAGE.items():
        if stage_id in READ_MATRIX_STAGES:
            continue
        assert record.adapter_status is StorageBenchmarkAdapterStatus.PLAN_ONLY
        assert record.adapter_module is None
        assert record.adapter_entrypoint is None
        assert record.resolves_adapter() is False
        assert record.as_record()["benchmark_coverage_complete"] is False


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
        ({"adapter_module": "bad module"}, "exact module"),
        ({"adapter_entrypoint": "bad-entry"}, "exact module"),
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
    base = DERIVED_ANALYSIS_STORAGE_BENCHMARK_BY_STAGE["tail_kinematics"]
    with pytest.raises(ValueError, match="must not claim an adapter"):
        replace(base, adapter_module="fisheye.diagnostics.some_benchmark")
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
    assert replace(
        completed,
        crimson_consumer_implemented=True,
    ).benchmark_coverage_complete is True


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
