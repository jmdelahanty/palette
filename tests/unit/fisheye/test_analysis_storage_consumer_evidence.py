from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from fisheye.analysis_workflows.storage_benchmark_catalog import (
    DERIVED_ANALYSIS_STORAGE_BENCHMARK_BY_STAGE,
    DerivedAnalysisStorageBenchmark,
)
from fisheye.analysis_workflows.storage_consumer_evidence import (
    ConsumerEvidenceScale,
    StorageConsumer,
    build_storage_consumer_evidence,
    require_storage_consumer_evidence,
)
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256


def _trial(*, repetition: int, position: int, role: str) -> dict[str, object]:
    return {
        "role": role,
        "order_position": position,
        "process_identity": f"r{repetition}-{role}-{position}",
        "exit_code": 0,
        "exact_schema_and_dtype": True,
        "direct_consolidated_metadata_equivalence": True,
        "explicit_run_selection": True,
        "dtype_probe_count": 0,
        "stale_publication_count": 0,
        "production_mutations": [],
        "decoded_logical_digest": "c" * 64,
        "workload_result_digest": "d" * 64,
        "measurements": {
            "full_scan_ms": 12.5,
            "peak_rss_bytes": 64 * 1024 * 1024,
            "physical_read_bytes": None,
            "physical_read_operations": None,
            "primary_read_p95_ms": 1.5,
            "readiness_ms": 20.0,
            "throughput_rows_per_second": 50_000.0,
        },
    }


def _repetitions() -> list[dict[str, object]]:
    return [
        {
            "repetition_index": 0,
            "order": ["source", "candidate"],
            "trials": [
                _trial(repetition=0, position=0, role="source"),
                _trial(repetition=0, position=1, role="candidate"),
            ],
        },
        {
            "repetition_index": 1,
            "order": ["candidate", "source"],
            "trials": [
                _trial(repetition=1, position=0, role="candidate"),
                _trial(repetition=1, position=1, role="source"),
            ],
        },
    ]


@pytest.fixture
def matrix_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, dict[str, object]]:
    payload = {"fixture": "consumer_evidence"}
    matrix = {
        "schema_id": "palette.exact_tabular_candidate_read_matrix",
        "schema_version": 2,
        "payload": payload,
        "payload_digest": canonical_json_sha256(payload),
    }
    path = tmp_path / "matrix.json"
    path.write_text(json.dumps(matrix), encoding="utf-8")
    normalized = {
        "stage_id": "swim_bouts",
        "family_id": "swim_bouts",
        "matrix_schema_id": matrix["schema_id"],
        "matrix_schema_version": matrix["schema_version"],
        "benchmark_id": "fixture",
        "archive_path": "/groups/source/archive.zarr",
        "source_run_name": "source",
        "candidate_run_name": "candidate",
        "source_run_path": "analysis/swim_bout_runs/source",
        "candidate_run_path": "analysis/swim_bout_runs/candidate",
        "balanced_repetitions": "passed",
        "decoded_equality": "passed",
        "metadata_equivalence": "passed",
        "physical_io": "not_recorded",
        "palette_source_consumer": "not_recorded",
        "palette_candidate_consumer": "not_recorded",
        "crimson_consumer": "not_recorded",
        "promotion_gate": "not_recorded",
    }

    def validated(
        self: DerivedAnalysisStorageBenchmark,
        supplied: object,
    ) -> dict[str, object]:
        if (
            self is DERIVED_ANALYSIS_STORAGE_BENCHMARK_BY_STAGE["swim_bouts"]
            and supplied == matrix
        ):
            return normalized
        pytest.fail("unexpected matrix validation request")

    monkeypatch.setattr(
        DerivedAnalysisStorageBenchmark,
        "validated_matrix_identity",
        validated,
    )
    return path, normalized


def _build(
    matrix_binding: tuple[Path, dict[str, object]],
    *,
    clean: bool = True,
    repetitions: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    matrix_path, _ = matrix_binding
    return build_storage_consumer_evidence(
        stage_id="swim_bouts",
        consumer=StorageConsumer.CRIMSON,
        scale=ConsumerEvidenceScale.REPRESENTATIVE_FULL,
        execution_id="sleepyfish-full-v1",
        matrix_path=matrix_path,
        consumer_archive_path="/Volumes/johnsonlab/fixture.zarr",
        producer_revision="a" * 40,
        producer_worktree_clean=clean,
        executable_sha256="b" * 64,
        command=["crimson-storage-consumer", "--headless"],
        workload_contract_id="derived-analytics-storage-consumer-v1",
        workload_contract_version=1,
        workload_contract_digest="e" * 64,
        cache_state="fresh_process_os_cache_uncontrolled",
        started_at_utc="2026-08-04T12:00:00+00:00",
        finished_at_utc="2026-08-04T12:05:00+00:00",
        platform_record={
            "operating_system": "macos",
            "architecture": "arm64",
            "runtime": "tensorstore-0.1.64",
        },
        repetitions=_repetitions() if repetitions is None else repetitions,
    )


def test_consumer_evidence_binds_matrix_balanced_trials_and_catalog_stage(
    matrix_binding: tuple[Path, dict[str, object]],
) -> None:
    receipt = _build(matrix_binding)

    require_storage_consumer_evidence(receipt, replay_matrix=True)
    DERIVED_ANALYSIS_STORAGE_BENCHMARK_BY_STAGE[
        "swim_bouts"
    ].validate_consumer_evidence(receipt, replay_matrix=True)
    payload = receipt["payload"]
    assert payload["matrix_binding"]["normalized_identity"] == matrix_binding[1]
    assert payload["consumer_paths"] == {
        "archive_path": "/Volumes/johnsonlab/fixture.zarr",
        "source_run_path": (
            "/Volumes/johnsonlab/fixture.zarr/analysis/swim_bout_runs/source"
        ),
        "candidate_run_path": (
            "/Volumes/johnsonlab/fixture.zarr/analysis/swim_bout_runs/candidate"
        ),
    }
    assert payload["gate"] == {
        "balanced_fresh_processes_complete": True,
        "decoded_equality": True,
        "workload_equality": True,
        "metadata_equivalence": True,
        "consumer_gate_passed": True,
        "evidence_eligible": True,
    }
    assert payload["promotion_authorized"] is False


def test_consumer_evidence_derives_failure_and_dirty_ineligibility(
    matrix_binding: tuple[Path, dict[str, object]],
) -> None:
    mismatched = _repetitions()
    mismatched[1]["trials"][0]["decoded_logical_digest"] = "f" * 64
    failed = _build(matrix_binding, repetitions=mismatched)
    assert failed["payload"]["gate"]["decoded_equality"] is False
    assert failed["payload"]["gate"]["consumer_gate_passed"] is False
    assert failed["payload"]["gate"]["evidence_eligible"] is False

    dirty = _build(matrix_binding, clean=False)
    assert dirty["payload"]["gate"]["consumer_gate_passed"] is True
    assert dirty["payload"]["gate"]["evidence_eligible"] is False


def test_consumer_evidence_rejects_rehashed_gate_and_process_tampering(
    matrix_binding: tuple[Path, dict[str, object]],
) -> None:
    receipt = _build(matrix_binding)
    tampered = deepcopy(receipt)
    tampered["payload"]["gate"]["evidence_eligible"] = False
    tampered["payload_digest"] = canonical_json_sha256(tampered["payload"])
    with pytest.raises(ValueError, match="gate differs"):
        require_storage_consumer_evidence(tampered)

    duplicated = _repetitions()
    duplicated[1]["trials"][1]["process_identity"] = duplicated[0]["trials"][0][
        "process_identity"
    ]
    with pytest.raises(ValueError, match="process identity"):
        _build(matrix_binding, repetitions=duplicated)

    reordered = _repetitions()
    reordered[1]["order"] = ["source", "candidate"]
    with pytest.raises(ValueError, match="order or cardinality"):
        _build(matrix_binding, repetitions=reordered)


def test_catalog_rejects_consumer_evidence_for_another_stage(
    matrix_binding: tuple[Path, dict[str, object]],
) -> None:
    receipt = _build(matrix_binding)
    with pytest.raises(ValueError, match="different catalog stage"):
        DERIVED_ANALYSIS_STORAGE_BENCHMARK_BY_STAGE[
            "bout_kinematics"
        ].validate_consumer_evidence(receipt)
