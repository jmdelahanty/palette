from __future__ import annotations

from copy import deepcopy
from importlib import import_module

import pytest

from fisheye.analysis_workflows import storage_benchmark_evidence as evidence
from fisheye.analysis_workflows.storage_benchmark_catalog import (
    DERIVED_ANALYSIS_STORAGE_BENCHMARK_BY_STAGE,
)
from fisheye.analysis_workflows.storage_benchmark_evidence import (
    BenchmarkEvidenceState,
    extract_storage_benchmark_identity,
)
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256


def _matrix(stage_id: str, payload: dict[str, object]) -> dict[str, object]:
    record = DERIVED_ANALYSIS_STORAGE_BENCHMARK_BY_STAGE[stage_id]
    return {
        "schema_id": record.matrix_schema_id,
        "schema_version": record.matrix_schema_version,
        "payload": payload,
        "payload_digest": canonical_json_sha256(payload),
    }


def _workload(source: str, candidate: str) -> dict[str, object]:
    return {
        "schema_id": "fixture.workload",
        "schema_version": 1,
        "payload": {
            "source_run_path": source,
            "candidate_run_path": candidate,
        },
        "payload_digest": "fixture",
    }


def _pair(source: str, candidate: str, **extra: object) -> dict[str, object]:
    payload = {
        "source_run_path": source,
        "candidate_run_path": candidate,
        **extra,
    }
    return {
        "schema_id": "fixture.pair",
        "schema_version": 1,
        "payload": payload,
        "payload_digest": canonical_json_sha256(payload),
    }


def _payloads() -> dict[str, dict[str, object]]:
    exact_correctness = {
        "logical_equality": True,
        "direct_consolidated_metadata_equivalence": True,
    }
    chaser_correctness = {
        "full_decoded_logical_equality": True,
        "primary_access_decoded_equality": True,
        "direct_consolidated_metadata_equivalence": True,
    }
    return {
        "swim_bouts": {
            "benchmark_id": "exact_tabular_candidate_reads_v1",
            "family_id": "swim_bouts",
            "archive_path": "/tmp/benchmark/archive.zarr",
            "source_run_name": "source",
            "candidate_run_name": "candidate",
            "repetitions": 5,
            "balanced_read_matrix_complete": True,
            "correctness": exact_correctness,
        },
        "bout_kinematics": {
            "benchmark_id": "exact_tabular_candidate_reads_v1",
            "family_id": "bout_kinematics",
            "archive_path": "/tmp/benchmark/archive.zarr",
            "source_run_name": "source",
            "candidate_run_name": "candidate",
            "repetitions": 5,
            "balanced_read_matrix_complete": True,
            "correctness": exact_correctness,
        },
        "detection_occupancy": {
            "benchmark_id": "exact_tabular_candidate_reads_v1",
            "family_id": "detection_occupancy",
            "archive_path": "/tmp/benchmark/archive.zarr",
            "source_run_name": "source",
            "candidate_run_name": "candidate",
            "repetitions": 5,
            "balanced_read_matrix_complete": True,
            "correctness": exact_correctness,
        },
        "session_occupancy": {
            "benchmark_id": "exact_tabular_candidate_reads_v1",
            "family_id": "session_occupancy",
            "archive_path": "/tmp/benchmark/archive.zarr",
            "source_run_name": "source",
            "candidate_run_name": "candidate",
            "repetitions": 5,
            "balanced_read_matrix_complete": True,
            "correctness": exact_correctness,
        },
        "chaser_distance": {
            "benchmark_id": "chaser_distance_sealed_base_reads_v1",
            "family_id": "chaser_distance_sealed_base",
            "archive_path": "/tmp/benchmark/archive.zarr",
            "source_parent_path": "analysis/chaser_distance_runs",
            "candidate_parent_path": "analysis/chaser_distance_storage_candidates",
            "source_run_name": "source",
            "candidate_run_name": "candidate",
            "repetitions": 5,
            "correctness": chaser_correctness,
        },
        "stimulus_epochs": {
            "benchmark_id": "stimulus_epoch_reads_v1",
            "archive_path": "/tmp/benchmark/archive.zarr",
            "source_run_name": "source",
            "candidate_run_name": "candidate",
            "repetitions": 5,
            "balanced_fresh_process_matrix_complete": True,
            "workload": _workload(
                "analysis/stimulus_epoch_runs/source",
                "analysis/stimulus_epoch_runs/candidate",
            ),
            "correctness": {
                "complete_decoded_array_equality": True,
                "decoded_segment_equality": True,
                "direct_consolidated_metadata_equivalence": True,
            },
        },
        "stimulus_response": {
            "benchmark_id": "stimulus_response_reads_v1",
            "family_id": "stimulus_response_compact_v3",
            "archive": "/tmp/benchmark/archive.zarr",
            "source_run": "source",
            "candidate_run": "candidate",
            "repetitions": 5,
            "balanced_fresh_process_matrix_complete": True,
            "workload": _workload(
                "analysis/stimulus_response_runs/source",
                "analysis/stimulus_response_runs/candidate",
            ),
            "correctness": {
                "decoded_arrays_equal": True,
                "logical_reader_equal": True,
                "direct_consolidated_equal": True,
            },
        },
        "eye_angles": {
            "benchmark_id": "eye_angle_v7_reads_v1",
            "family_id": "eye_angle_compact_v7",
            "archive": "/tmp/benchmark/archive.zarr",
            "source_run": "source",
            "candidate_run": "candidate",
            "repetitions": 5,
            "balanced_fresh_process_matrix_complete": True,
            "workload": _workload(
                "analysis/eye_angle_runs/source",
                "analysis/eye_angle_runs/candidate",
            ),
            "correctness": {
                "decoded_arrays_equal": True,
                "source_public_reader_candidate_diagnostic_adapter_equal": True,
                "direct_consolidated_equal": True,
            },
        },
        "track_kinematics": {
            "benchmark_id": "track_kinematics_v2_reads_v1",
            "family_id": "track_kinematics_v2_flat_lineage",
            "archive_path": "/tmp/benchmark/archive.zarr",
            "source_run_name": "source",
            "candidate_run_name": "candidate",
            "repetitions": 5,
            "balanced_fresh_process_matrix_complete": True,
            "pair_validation": {
                "source_run_path": "analysis/track_kinematics_runs/offline/source",
                "candidate_run_path": (
                    "analysis/track_kinematics_runs/offline/candidate"
                ),
                "complete_decoded_equality": True,
                "metadata_equivalence": {"source": {}, "candidate": {}},
            },
            "public_source_consumer_implemented": True,
            "public_candidate_consumer_implemented": False,
        },
        "tail_kinematics": {
            "benchmark_id": "tail_kinematics_v2_reads_v1",
            "family_id": "tail_kinematics_v2",
            "archive": "/tmp/benchmark/archive.zarr",
            "source_run_name": "source",
            "candidate_run_name": "candidate",
            "repetitions": 5,
            "trials": [
                {
                    "payload": {
                        "pair_validation": _pair(
                            "analysis/tail_kinematics_runs/source",
                            "analysis/tail_kinematics_runs/candidate",
                        )
                    }
                }
            ],
        },
        "subject_shape": {
            "benchmark_id": "subject_shape_v4_reads_v1",
            "family_id": "subject_shape_v4",
            "archive_path": "/tmp/benchmark/archive.zarr",
            "parent_path": "analysis/subject_shape_runs",
            "source_run_name": "source",
            "candidate_run_name": "candidate",
            "repetitions": 5,
            "correctness": chaser_correctness,
        },
        "tail_posture_view": {
            "benchmark_id": "tail_posture_view_v3_reads_v1",
            "family_id": "tail_posture_view_v3",
            "archive_identity": {"resolved_path": "/tmp/benchmark/archive.zarr"},
            "source_run": "source",
            "candidate_run": "candidate",
            "repetitions": 5,
            "pair_validation": {
                "payload": {
                    "logical_equality": {"all_equal": True},
                    "metadata_equivalence": {"source": {}, "candidate": {}},
                }
            },
        },
        "bout_classification": {
            "benchmark_id": "bout_classification_v2_reads_v1",
            "family_id": "bout_classification",
            "archive_path": "/tmp/benchmark/archive.zarr",
            "source_run_name": "source",
            "candidate_run_name": "candidate",
            "repetitions": 5,
            "balanced_read_matrix_complete": True,
            "pair_validation": _pair(
                "analysis/bout_classification_runs/source",
                "analysis/bout_classification_runs/candidate",
            ),
            "correctness": {
                "complete_decoded_equality": True,
                "direct_consolidated_metadata_equivalence": True,
            },
        },
    }


@pytest.mark.parametrize("stage_id", sorted(_payloads()))
def test_normalized_identity_keeps_absent_evidence_explicit(
    stage_id: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        evidence,
        "validate_storage_benchmark_matrix",
        lambda observed_stage, _matrix: calls.append(observed_stage),
    )

    identity = extract_storage_benchmark_identity(
        stage_id,
        _matrix(stage_id, _payloads()[stage_id]),
    )

    assert calls == [stage_id]
    assert identity.stage_id == stage_id
    adapter_module = DERIVED_ANALYSIS_STORAGE_BENCHMARK_BY_STAGE[
        stage_id
    ].adapter_module
    assert adapter_module is not None
    declared_family = getattr(import_module(adapter_module), "FAMILY_ID", None)
    assert identity.family_id == (
        stage_id if declared_family is None else declared_family
    )
    assert identity.archive_path == "/tmp/benchmark/archive.zarr"
    assert identity.source_run_name == "source"
    assert identity.candidate_run_name == "candidate"
    assert identity.source_run_path.endswith("/source")
    assert identity.candidate_run_path.endswith("/candidate")
    assert identity.balanced_repetitions is BenchmarkEvidenceState.PASSED
    assert identity.decoded_equality is BenchmarkEvidenceState.PASSED
    assert identity.metadata_equivalence is BenchmarkEvidenceState.PASSED
    assert identity.physical_io is BenchmarkEvidenceState.UNAVAILABLE
    assert identity.crimson_consumer is BenchmarkEvidenceState.NOT_RECORDED
    assert identity.promotion_gate is BenchmarkEvidenceState.NOT_RECORDED


def test_public_candidate_consumer_gaps_are_not_reported_as_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        evidence,
        "validate_storage_benchmark_matrix",
        lambda _stage, _matrix: None,
    )
    payloads = _payloads()

    exact = extract_storage_benchmark_identity(
        "swim_bouts", _matrix("swim_bouts", payloads["swim_bouts"])
    )
    track = extract_storage_benchmark_identity(
        "track_kinematics",
        _matrix("track_kinematics", payloads["track_kinematics"]),
    )
    stimulus = extract_storage_benchmark_identity(
        "stimulus_response",
        _matrix("stimulus_response", payloads["stimulus_response"]),
    )

    assert exact.palette_candidate_consumer is BenchmarkEvidenceState.NOT_RECORDED
    assert track.palette_candidate_consumer is BenchmarkEvidenceState.UNAVAILABLE
    assert stimulus.palette_candidate_consumer is BenchmarkEvidenceState.PASSED


def test_schema_digest_and_stage_family_tampering_fail_before_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        evidence,
        "validate_storage_benchmark_matrix",
        lambda _stage, _matrix: pytest.fail("validator must not run"),
    )
    payload = _payloads()["swim_bouts"]
    matrix = _matrix("swim_bouts", payload)

    wrong_schema = deepcopy(matrix)
    wrong_schema["schema_id"] = "palette.wrong"
    with pytest.raises(ValueError, match="schema differs"):
        extract_storage_benchmark_identity("swim_bouts", wrong_schema)

    wrong_digest = deepcopy(matrix)
    wrong_digest["payload_digest"] = "0" * 64
    with pytest.raises(ValueError, match="payload digest"):
        extract_storage_benchmark_identity("swim_bouts", wrong_digest)


def test_stage_family_mismatch_fails_even_after_deep_validator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        evidence,
        "validate_storage_benchmark_matrix",
        lambda _stage, _matrix: None,
    )
    payload = deepcopy(_payloads()["swim_bouts"])
    payload["family_id"] = "bout_kinematics"
    with pytest.raises(ValueError, match="family differs"):
        extract_storage_benchmark_identity("swim_bouts", _matrix("swim_bouts", payload))


def test_projection_is_strict_json(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        evidence,
        "validate_storage_benchmark_matrix",
        lambda _stage, _matrix: None,
    )
    payload = _payloads()["bout_classification"]
    identity = extract_storage_benchmark_identity(
        "bout_classification",
        _matrix("bout_classification", payload),
    )
    assert canonical_json_sha256(identity.as_record())


def test_catalog_is_the_public_matrix_evidence_reader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        evidence,
        "validate_storage_benchmark_matrix",
        lambda _stage, _matrix: None,
    )
    payload = _payloads()["stimulus_epochs"]
    record = DERIVED_ANALYSIS_STORAGE_BENCHMARK_BY_STAGE["stimulus_epochs"]

    normalized = record.validated_matrix_identity(_matrix("stimulus_epochs", payload))

    assert normalized["stage_id"] == "stimulus_epochs"
    assert normalized["decoded_equality"] == "passed"
    assert normalized["physical_io"] == "unavailable"
    assert normalized["crimson_consumer"] == "not_recorded"
