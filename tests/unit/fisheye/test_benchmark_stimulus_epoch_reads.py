from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest
import zarr

from fisheye.analysis.exact_tabular_storage import (
    ANALYSIS_STORAGE_PLAN_RECEIPT_ATTR,
)
from fisheye.analysis_workflows.materializers.stimulus_epochs import (
    materialize_stimulus_epoch_candidate,
)
from fisheye.diagnostics import benchmark_stimulus_epoch_reads as benchmark
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from fisheye.shared.zarr_helpers import consolidate_metadata_capture_expected_warnings

from .test_stimulus_epoch_schema import create_legacy_stimulus_epoch_archive


def _archive(tmp_path: Path, *, name: str = "stimulus") -> Path:
    archive = tmp_path / f"{name}_analysis.zarr"
    create_legacy_stimulus_epoch_archive(archive)
    result = materialize_stimulus_epoch_candidate(
        archive,
        source_run="source",
        run_name="candidate",
        scratch_root=tmp_path / f"scratch-{name}",
        copy_backend="python",
        apply=True,
    )
    assert result["status"] == "complete"
    return archive


def _preflight(archive: Path, *, repetitions: int = 1) -> dict[str, object]:
    return benchmark._preflight(
        archive,
        source_run_name="source",
        candidate_run_name="candidate",
        seed=23,
        repetitions=repetitions,
    )


def _rehash(envelope: dict[str, object]) -> None:
    envelope["payload_digest"] = canonical_json_sha256(envelope["payload"])


def test_preflight_binds_exact_source_candidate_and_complete_workload(
    tmp_path: Path,
) -> None:
    archive = _archive(tmp_path)

    result = _preflight(archive, repetitions=5)
    workload = result["workload"]
    assert isinstance(workload, dict)
    benchmark.require_workload(workload)
    payload = workload["payload"]
    assert payload["access"] == {
        "mode": "eager_whole_array_once",
        "array_order": list(benchmark.ARRAY_PATHS),
        "operation_count": 12,
    }
    assert set(payload["expected_arrays"]) == set(benchmark.ARRAY_PATHS)
    assert payload["metadata_equivalence"]["source"]["array_count"] == 12
    assert payload["metadata_equivalence"]["candidate"]["array_count"] == 12
    assert (
        payload["candidate_storage_receipt_payload_digest"]
        == result["candidate_storage_receipt_payload_digest"]
    )


def test_single_trials_use_strict_role_specific_consumers_and_equal_payloads(
    tmp_path: Path,
) -> None:
    archive = _archive(tmp_path)
    workload = _preflight(archive)["workload"]
    candidate = benchmark.run_single_trial(
        archive,
        source_run="source",
        candidate_run="candidate",
        role="candidate",
        repetition_index=0,
        order_position=0,
        seed=23,
        cache_state="pytest_uncontrolled_os_cache",
        workload=workload,
    )
    source = benchmark.run_single_trial(
        archive,
        source_run="source",
        candidate_run="candidate",
        role="source",
        repetition_index=0,
        order_position=1,
        seed=23,
        cache_state="pytest_uncontrolled_os_cache",
        workload=workload,
    )

    benchmark.require_trial_result(candidate, workload=workload)
    benchmark.require_trial_result(source, workload=workload)
    assert candidate["payload"]["validation"]["consumer_path"] == "strict_exact_v2"
    assert source["payload"]["validation"]["consumer_path"] == "explicit_legacy_v1"
    assert (
        candidate["payload"]["full_scan"]["arrays"]
        == source["payload"]["full_scan"]["arrays"]
    )
    assert candidate["payload"]["physical_io"]["transferred_bytes"] is None


def test_matrix_uses_distinct_fresh_processes_and_preserves_archive_metadata(
    tmp_path: Path,
) -> None:
    archive = _archive(tmp_path)
    output = tmp_path / ".palette_benchmarks" / "stimulus-epoch-read-matrix"

    result = benchmark.run_benchmark_matrix(
        archive,
        source_run="source",
        candidate_run="candidate",
        output_dir=output,
        cache_state="pytest_uncontrolled_os_cache",
        seed=23,
        repetitions=1,
    )

    benchmark.require_matrix_result(result)
    payload = result["payload"]
    assert payload["correctness"]["all_passed"] is True
    assert payload["archive_read_only_metadata_guard"]["unchanged"] is True
    assert payload["balanced_fresh_process_matrix_complete"] is False
    assert [trial["payload"]["role"] for trial in payload["trials"]] == [
        "candidate",
        "source",
    ]
    process_ids = [trial["payload"]["process_id"] for trial in payload["trials"]]
    assert len(set(process_ids)) == 2
    assert payload["driver_process_id"] not in process_ids
    assert (output / "read_workload.json").is_file()
    assert (output / "matrix_result.json").is_file()
    assert len(list((output / "trials").glob("*.json"))) == 2


def test_rehashed_evidence_tampering_fails_deep_validation(tmp_path: Path) -> None:
    archive = _archive(tmp_path)
    workload = _preflight(archive)["workload"]
    trial = benchmark.run_single_trial(
        archive,
        source_run="source",
        candidate_run="candidate",
        role="candidate",
        repetition_index=0,
        order_position=0,
        seed=23,
        cache_state="pytest",
        workload=workload,
    )

    fabricated_io = copy.deepcopy(trial)
    fabricated_io["payload"]["physical_io"]["transferred_bytes"] = 1
    _rehash(fabricated_io)
    with pytest.raises(ValueError, match="must not fabricate"):
        benchmark.require_trial_result(fabricated_io, workload=workload)

    wrong_receipt = copy.deepcopy(trial)
    wrong_receipt["payload"]["validation"][
        "candidate_storage_receipt_payload_digest"
    ] = ("f" * 64)
    _rehash(wrong_receipt)
    with pytest.raises(ValueError, match="receipt binding mismatch"):
        benchmark.require_trial_result(wrong_receipt, workload=workload)

    wrong_access = copy.deepcopy(workload)
    wrong_access["payload"]["access"]["operation_count"] = 11
    _rehash(wrong_access)
    with pytest.raises(ValueError, match="access declaration"):
        benchmark.require_workload(wrong_access)

    wrong_candidate_lineage = copy.deepcopy(workload)
    candidate_payload = wrong_candidate_lineage["payload"]
    candidate_payload["candidate_lineage_hash"] = "d" * 64
    candidate_payload["candidate_lineage_payload_sha256"] = "e" * 64
    manifest = candidate_payload["candidate_run_manifest"]
    manifest["payload"]["candidate_lineage"]["lineage_hash"] = "d" * 64
    manifest["payload"]["candidate_lineage"]["lineage_payload_sha256"] = "e" * 64
    manifest["payload_digest"] = canonical_json_sha256(manifest["payload"])
    candidate_payload["candidate_run_manifest_payload_digest"] = manifest[
        "payload_digest"
    ]
    _rehash(wrong_candidate_lineage)
    with pytest.raises(ValueError, match="candidate lineage executable digest"):
        benchmark.require_workload(wrong_candidate_lineage)

    wrong_materializer = copy.deepcopy(workload)
    wrong_materializer["payload"]["candidate_materializer_identity"][
        "git_commit"
    ] = "different_commit"
    _rehash(wrong_materializer)
    with pytest.raises(ValueError, match="lineage differs from run-manifest"):
        benchmark.require_workload(wrong_materializer)


def test_matrix_rejects_rehashed_nonpromotion_and_fresh_process_tampering(
    tmp_path: Path,
) -> None:
    archive = _archive(tmp_path)
    result = benchmark.run_benchmark_matrix(
        archive,
        source_run="source",
        candidate_run="candidate",
        output_dir=tmp_path / ".palette_benchmarks" / "matrix-tamper",
        cache_state="pytest",
        seed=23,
        repetitions=1,
    )

    promoted = copy.deepcopy(result)
    promoted["payload"]["benchmark_only"][
        "selector_or_profile_change_authorized"
    ] = True
    _rehash(promoted)
    with pytest.raises(ValueError, match="benchmark-only"):
        benchmark.require_matrix_result(promoted)

    reused_process = copy.deepcopy(result)
    reused_process["payload"]["trials"][1]["payload"]["process_id"] = reused_process[
        "payload"
    ]["trials"][0]["payload"]["process_id"]
    _rehash(reused_process["payload"]["trials"][1])
    _rehash(reused_process)
    with pytest.raises(ValueError, match="distinct fresh processes"):
        benchmark.require_matrix_result(reused_process)

    coordinated = copy.deepcopy(result)
    workload = coordinated["payload"]["workload"]
    workload_payload = workload["payload"]
    workload_payload["source_lineage_hash"] = "a" * 64
    workload_payload["source_lineage_payload_sha256"] = "b" * 64
    workload_payload["candidate_storage_receipt_payload_digest"] = "c" * 64
    manifest = workload_payload["candidate_run_manifest"]
    manifest["payload"]["source_epoch"]["lineage_hash"] = "a" * 64
    manifest["payload"]["source_epoch"]["lineage_payload_sha256"] = "b" * 64
    manifest["payload"]["schema_bindings"]["storage_receipt_payload_digest"] = "c" * 64
    manifest["payload_digest"] = canonical_json_sha256(manifest["payload"])
    workload_payload["candidate_run_manifest_payload_digest"] = manifest[
        "payload_digest"
    ]
    _rehash(workload)
    for trial in coordinated["payload"]["trials"]:
        trial["payload"]["workload_payload_digest"] = workload["payload_digest"]
        trial["payload"]["validation"]["source_lineage_hash"] = "a" * 64
        if trial["payload"]["role"] == "candidate":
            trial["payload"]["validation"][
                "candidate_storage_receipt_payload_digest"
            ] = ("c" * 64)
            trial["payload"]["validation"]["candidate_run_manifest_payload_digest"] = (
                manifest["payload_digest"]
            )
        _rehash(trial)
    _rehash(coordinated)
    with pytest.raises(ValueError, match="lineage executable digest mismatch"):
        benchmark.require_matrix_result(coordinated)


def test_stale_candidate_metadata_and_decoded_mismatch_fail_closed(
    tmp_path: Path,
) -> None:
    stale = _archive(tmp_path, name="stale")
    workload = _preflight(stale)["workload"]
    direct = zarr.open_group(str(stale), mode="a", use_consolidated=False)
    direct["analysis/stimulus_epoch_runs/candidate"]["windows"].attrs[
        "post_consolidation_tamper"
    ] = True
    with pytest.raises(RuntimeError, match="Direct/consolidated declaration differs"):
        benchmark.run_single_trial(
            stale,
            source_run="source",
            candidate_run="candidate",
            role="candidate",
            repetition_index=0,
            order_position=0,
            seed=23,
            cache_state="pytest",
            workload=workload,
        )

    mismatch = _archive(tmp_path, name="mismatch")
    direct = zarr.open_group(str(mismatch), mode="a", use_consolidated=False)
    direct["analysis/stimulus_epoch_runs/source/windows/window_id"][:] = np.asarray(
        [0, 2, 3], dtype=np.int32
    )
    consolidate_metadata_capture_expected_warnings(mismatch)
    with pytest.raises(ValueError, match="decoded arrays differ"):
        _preflight(mismatch)


def test_rehashed_persisted_candidate_receipt_and_unsafe_paths_are_rejected(
    tmp_path: Path,
) -> None:
    archive = _archive(tmp_path)
    root = zarr.open_group(str(archive), mode="a", use_consolidated=False)
    candidate = root["analysis/stimulus_epoch_runs/candidate"]
    receipt = copy.deepcopy(candidate.attrs[ANALYSIS_STORAGE_PLAN_RECEIPT_ATTR])
    receipt["payload"]["storage_profile"]["target_chunk_bytes"] += 1
    receipt["payload_digest"] = canonical_json_sha256(receipt["payload"])
    candidate.attrs[ANALYSIS_STORAGE_PLAN_RECEIPT_ATTR] = receipt
    consolidate_metadata_capture_expected_warnings(archive)
    with pytest.raises(ValueError, match="storage"):
        _preflight(archive)

    safe_archive = _archive(tmp_path, name="safe")
    with pytest.raises(ValueError, match="explicit immutable child name"):
        benchmark.run_benchmark_matrix(
            safe_archive,
            source_run="latest",
            candidate_run="candidate",
            output_dir=tmp_path / ".palette_benchmarks" / "alias",
            cache_state="pytest",
            repetitions=1,
        )
    with pytest.raises(ValueError, match="disjoint"):
        benchmark.run_benchmark_matrix(
            safe_archive,
            source_run="source",
            candidate_run="candidate",
            output_dir=safe_archive / "benchmark-output",
            cache_state="pytest",
            repetitions=1,
        )
    with pytest.raises(ValueError, match="benchmark-only"):
        benchmark.run_benchmark_matrix(
            safe_archive,
            source_run="source",
            candidate_run="candidate",
            output_dir=tmp_path / "results",
            cache_state="pytest",
            repetitions=1,
        )


def test_five_repetition_order_is_deterministic_and_rotated() -> None:
    assert [
        benchmark._trial_order(seed=23, repetition_index=index) for index in range(5)
    ] == [
        ("candidate", "source"),
        ("source", "candidate"),
        ("candidate", "source"),
        ("source", "candidate"),
        ("candidate", "source"),
    ]
