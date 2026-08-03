from __future__ import annotations

from copy import deepcopy
import os
from pathlib import Path
import shutil

import pytest
import zarr

from fisheye.analysis import eye_angle_io
from fisheye.diagnostics import benchmark_eye_angle_v7_reads as benchmark
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from fisheye.shared.zarr.metadata_equivalence import ZarrMetadataEquivalenceError
from tests.unit.fisheye.test_eye_angle_materializer import (
    _accept_synthetic_subject_shape_publication,
    _build_source,
    _materialize_storage_candidate,
    mod as materializer,
)


def _resign(value: dict[str, object], schema_id: str) -> dict[str, object]:
    payload = value["payload"]
    assert isinstance(payload, dict)
    return {
        "schema_id": schema_id,
        "schema_version": 1,
        "payload": payload,
        "payload_digest": canonical_json_sha256(payload),
    }


@pytest.fixture(scope="module")
def eye_angle_pair(tmp_path_factory: pytest.TempPathFactory) -> Path:
    directory = tmp_path_factory.mktemp("eye-angle-v7-benchmark")
    archive = directory / "fixture.zarr"
    monkeypatch = pytest.MonkeyPatch()
    try:
        _build_source(archive)
        _accept_synthetic_subject_shape_publication(monkeypatch, archive)
        monkeypatch.setattr(
            materializer,
            "emit_eye_angle_stage_completion",
            lambda *args, **kwargs: False,
        )
        result = materializer.materialize_eye_angles(
            archive,
            scratch_root=directory / "source-scratch",
            subject_shape_run="shape_1",
            keypoint_run="kp_raw_1",
            run_name="eye_source",
            chunk_rows=2,
            angle_chunk_rows=2,
            angle_chunk_columns=16,
            output_shard_rows=4,
            angle_shard_columns=16,
            execution_backend="serial_driver",
            scheduler="single-threaded",
            num_workers=1,
            shard_workers=1,
            fps=100.0,
            copy_backend="python",
            apply=True,
            keep_scratch=True,
            check_capacity=False,
            stage_command="eye-angle-benchmark-source-fixture",
        )
        assert result["status"] == "complete"
        candidate = _materialize_storage_candidate(
            monkeypatch,
            archive,
            directory / "candidate-scratch",
            run_name="eye_candidate",
        )
        assert candidate["status"] == "complete"
    finally:
        monkeypatch.undo()
    return archive


def test_preflight_binds_exact_pair_and_diagnostic_only_adapter(
    eye_angle_pair: Path,
) -> None:
    workload = benchmark._preflight(
        eye_angle_pair,
        source_run_name="eye_source",
        candidate_run_name="eye_candidate",
        seed=37,
        repetitions=1,
    )
    benchmark.require_workload(workload)
    payload = workload["payload"]
    assert payload["expected_arrays"]
    assert len(payload["expected_arrays"]) == 41
    assert payload["palette_consumer_implemented"] is False
    assert payload["candidate_adapter_scope"] == (
        "diagnostic_only_private_strict_payload_adapter"
    )
    assert payload["promotion_authorized"] is False


def test_fresh_process_matrix_is_balanced_read_only_and_nonpromoting(
    eye_angle_pair: Path,
    tmp_path: Path,
) -> None:
    output = tmp_path / "eye-angle-benchmark-results"
    result = benchmark.run_benchmark_matrix(
        eye_angle_pair,
        source_run="eye_source",
        candidate_run="eye_candidate",
        output_value=output,
        repetitions=1,
        seed=37,
    )
    benchmark.require_matrix_result(result)
    payload = result["payload"]
    assert len(payload["trials"]) == 2
    assert len({trial["payload"]["process_id"] for trial in payload["trials"]}) == 2
    assert os.getpid() not in {
        trial["payload"]["process_id"] for trial in payload["trials"]
    }
    assert payload["archive_read_only_metadata_guard"]["unchanged"] is True
    assert payload["physical_io_availability"] == (
        "not_collected_requires_external_trace"
    )
    assert payload["palette_consumer_implemented"] is False


def test_candidate_storage_receipt_coordinated_rehash_is_rejected(
    eye_angle_pair: Path,
) -> None:
    workload = benchmark._preflight(
        eye_angle_pair,
        source_run_name="eye_source",
        candidate_run_name="eye_candidate",
        seed=37,
        repetitions=1,
    )
    tampered = deepcopy(workload)
    payload = tampered["payload"]
    receipt = payload["candidate_storage_receipt"]
    receipt["payload"]["arrays"][0]["plan"]["chunk_nbytes"] += 1
    receipt["payload_digest"] = canonical_json_sha256(receipt["payload"])
    candidate_path = payload["candidate_run_path"]
    declaration = payload["candidate_metadata_declarations"][candidate_path]
    declaration["attributes"]["eye_angle_storage_plan"] = deepcopy(receipt)
    payload["candidate_metadata_equivalence"]["declarations_sha256"] = (
        canonical_json_sha256(payload["candidate_metadata_declarations"])
    )
    tampered = _resign(tampered, benchmark.WORKLOAD_SCHEMA_ID)
    with pytest.raises(ValueError, match="executable byte plan"):
        benchmark.require_workload(tampered)


def test_dependency_lineage_coordinated_rehash_is_rejected(
    eye_angle_pair: Path,
) -> None:
    workload = benchmark._preflight(
        eye_angle_pair,
        source_run_name="eye_source",
        candidate_run_name="eye_candidate",
        seed=37,
        repetitions=1,
    )
    tampered = deepcopy(workload)
    payload = tampered["payload"]
    for role in ("source", "candidate"):
        run_path = payload[f"{role}_run_path"]
        declarations = payload[f"{role}_metadata_declarations"]
        contracts = declarations[run_path]["attributes"]["eye_angle_source_contracts"]
        contracts["keypoints"]["method_version"] = "hostile.rewrite"
        payload[f"{role}_contract_digests"]["eye_angle_source_contracts"] = (
            canonical_json_sha256(contracts)
        )
        payload[f"{role}_metadata_equivalence"]["declarations_sha256"] = (
            canonical_json_sha256(declarations)
        )
    tampered = _resign(tampered, benchmark.WORKLOAD_SCHEMA_ID)
    with pytest.raises(ValueError, match="dependency identity differs"):
        benchmark.require_workload(tampered)


def test_trial_role_order_and_physical_io_claims_fail_closed(
    eye_angle_pair: Path,
) -> None:
    workload = benchmark._preflight(
        eye_angle_pair,
        source_run_name="eye_source",
        candidate_run_name="eye_candidate",
        seed=37,
        repetitions=1,
    )
    role = benchmark._trial_order(seed=37, repetition_index=0)[0]
    trial = benchmark.run_single_trial(
        eye_angle_pair,
        source_run="eye_source",
        candidate_run="eye_candidate",
        role=role,
        repetition_index=0,
        order_position=0,
        seed=37,
        cache_state="unit-test",
        workload=workload,
    )
    tampered = deepcopy(trial)
    tampered["payload"]["physical_io"]["transferred_bytes"] = 10
    tampered = _resign(tampered, benchmark.TRIAL_SCHEMA_ID)
    with pytest.raises(ValueError, match="fabricated physical I/O"):
        benchmark.require_trial_result(tampered, workload=workload)

    with pytest.raises(ValueError, match="rotation binding"):
        benchmark.run_single_trial(
            eye_angle_pair,
            source_run="eye_source",
            candidate_run="eye_candidate",
            role=role,
            repetition_index=0,
            order_position=1,
            seed=37,
            cache_state="unit-test",
            workload=workload,
        )


def test_public_consumer_flag_and_promotion_cannot_be_rehashed_true(
    eye_angle_pair: Path,
) -> None:
    workload = benchmark._preflight(
        eye_angle_pair,
        source_run_name="eye_source",
        candidate_run_name="eye_candidate",
        seed=37,
        repetitions=1,
    )
    for field in ("palette_consumer_implemented", "promotion_authorized"):
        tampered = deepcopy(workload)
        tampered["payload"][field] = True
        tampered = _resign(tampered, benchmark.WORKLOAD_SCHEMA_ID)
        with pytest.raises(ValueError, match="nonpromotion/telemetry"):
            benchmark.require_workload(tampered)


def test_unsafe_names_outputs_and_symlinked_runs_are_rejected(
    eye_angle_pair: Path,
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="explicit immutable"):
        benchmark._preflight(
            eye_angle_pair,
            source_run_name="latest",
            candidate_run_name="eye_candidate",
            seed=37,
            repetitions=1,
        )
    with pytest.raises(ValueError, match="must be a new disjoint"):
        benchmark._safe_output(
            eye_angle_pair / "benchmark-output", archive=eye_angle_pair
        )

    linked_archive = tmp_path / "linked.zarr"
    linked_archive.symlink_to(eye_angle_pair, target_is_directory=True)
    with pytest.raises(ValueError, match="must not be a symlink"):
        benchmark._safe_archive(linked_archive)


def test_candidate_normal_reader_still_rejects_selector_ineligible_run(
    eye_angle_pair: Path,
) -> None:
    root = zarr.open_group(str(eye_angle_pair), mode="r", use_consolidated=True)
    with pytest.raises(eye_angle_io.EyeAngleIOError, match="not selector-eligible"):
        eye_angle_io.load_eye_angle_run_tables(root, run_name="eye_candidate")


def test_stale_consolidated_metadata_and_run_descendant_symlink_fail_closed(
    eye_angle_pair: Path,
    tmp_path: Path,
) -> None:
    stale = tmp_path / "stale.zarr"
    shutil.copytree(eye_angle_pair, stale)
    direct = zarr.open_group(str(stale), mode="a", use_consolidated=False)
    direct["analysis/eye_angle_runs/eye_source"].attrs[
        "non_authoritative_test_marker"
    ] = "direct-only"
    with pytest.raises(ZarrMetadataEquivalenceError, match="declaration differs"):
        benchmark._preflight(
            stale,
            source_run_name="eye_source",
            candidate_run_name="eye_candidate",
            seed=37,
            repetitions=1,
        )

    linked = tmp_path / "linked-descendant.zarr"
    shutil.copytree(eye_angle_pair, linked)
    metadata = linked / "analysis" / "eye_angle_runs" / "eye_source" / "zarr.json"
    saved = tmp_path / "saved-eye-source-zarr.json"
    metadata.rename(saved)
    metadata.symlink_to(saved)
    with pytest.raises(ValueError, match="forbidden symlink"):
        benchmark._preflight(
            linked,
            source_run_name="eye_source",
            candidate_run_name="eye_candidate",
            seed=37,
            repetitions=1,
        )
