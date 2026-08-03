from __future__ import annotations

import copy
from pathlib import Path
import shutil

import pytest
import zarr

from fisheye.analysis.exact_tabular_storage import (
    ANALYSIS_STORAGE_PLAN_DIGEST_ATTR,
    ANALYSIS_STORAGE_PLAN_RECEIPT_ATTR,
)
from fisheye.analysis_workflows.materializers.chaser_distance_base import (
    materialize_chaser_distance_base_candidate,
)
from fisheye.diagnostics import benchmark_chaser_distance_base_candidate as benchmark
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from fisheye.shared.zarr_helpers import consolidate_metadata_capture_expected_warnings

from .test_chaser_distance_coordinate_publication import _publish_canonical


@pytest.fixture(scope="module")
def candidate_archive(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("chaser-distance-read-benchmark")
    archive, _zarr_root, _run = _publish_canonical(root)
    consolidate_metadata_capture_expected_warnings(archive)
    result = materialize_chaser_distance_base_candidate(
        archive,
        source_run="canonical_distance",
        run_name="candidate",
        scratch_root=root / "scratch",
        copy_backend="python",
        apply=True,
    )
    assert result["status"] == "complete"
    return archive


def _preflight(archive: Path, *, repetitions: int = 1) -> dict[str, object]:
    return benchmark._preflight(
        archive,
        source_path=(
            f"{benchmark.SOURCE_PARENT_PATH}/canonical_distance"
        ),
        candidate_path=f"{benchmark.CANDIDATE_PARENT_PATH}/candidate",
        seed=17,
        repetitions=repetitions,
    )


def _trial(
    archive: Path,
    *,
    role: str,
    order_position: int,
    suite: dict[str, object],
) -> dict[str, object]:
    return benchmark.run_single_trial(
        archive,
        source_parent=benchmark.SOURCE_PARENT_PATH,
        candidate_parent=benchmark.CANDIDATE_PARENT_PATH,
        source_run="canonical_distance",
        candidate_run="candidate",
        role=role,
        repetition_index=0,
        order_position=order_position,
        seed=17,
        cache_state="pytest_uncontrolled_os_cache",
        suite_manifest=suite,
    )


def test_preflight_binds_exact_source_authority_candidate_and_receipt(
    candidate_archive: Path,
) -> None:
    result = _preflight(candidate_archive, repetitions=5)

    suite = result["suite"]
    assert suite["payload"]["family_id"] == benchmark.FAMILY_ID
    assert suite["payload"]["repetitions"] == 5
    assert len(result["logical_hashes"]) == 30
    assert (
        suite["payload"]["storage_plan_receipt"]["payload_digest"]
        == result["candidate_storage_receipt_payload_digest"]
    )
    assert len(result["source_binding_sha256"]) == 64
    assert len(result["candidate_manifest_payload_digest"]) == 64


def test_source_and_candidate_trials_are_exact_and_truthful(
    candidate_archive: Path,
) -> None:
    preflight = _preflight(candidate_archive)
    candidate = _trial(
        candidate_archive,
        role="candidate",
        order_position=0,
        suite=preflight["suite"],
    )
    source = _trial(
        candidate_archive,
        role="source",
        order_position=1,
        suite=preflight["suite"],
    )

    benchmark.require_trial_result(candidate)
    benchmark.require_trial_result(source)
    assert candidate["payload"]["logical_arrays"] == source["payload"][
        "logical_arrays"
    ]
    assert candidate["payload"]["validation"]["source_binding_sha256"] == source[
        "payload"
    ]["validation"]["source_binding_sha256"]
    assert candidate["payload"]["publication_timing"]["availability"] == (
        "recorded_in_cluster_output_staging"
    )
    assert source["payload"]["publication_timing"]["availability"] == (
        "not_applicable_source"
    )
    assert candidate["payload"]["physical_io"]["transferred_bytes"] is None
    assert source["payload"]["physical_io"]["request_count"] is None
    assert candidate["payload"]["storage"]["scope"].startswith("exact_30_array")


def test_matrix_uses_fresh_processes_and_preserves_archive_metadata(
    tmp_path: Path,
    candidate_archive: Path,
) -> None:
    output = tmp_path / ".palette_benchmarks" / "chaser-distance-matrix"
    result = benchmark.run_benchmark_matrix(
        candidate_archive,
        source_parent=benchmark.SOURCE_PARENT_PATH,
        candidate_parent=benchmark.CANDIDATE_PARENT_PATH,
        source_run="canonical_distance",
        candidate_run="candidate",
        output_dir=output,
        cache_state="pytest_uncontrolled_os_cache",
        seed=17,
        repetitions=1,
    )

    benchmark.require_matrix_result(result)
    payload = result["payload"]
    assert [trial["payload"]["role"] for trial in payload["trials"]] == [
        "candidate",
        "source",
    ]
    assert payload["correctness"]["all_passed"] is True
    assert payload["archive_read_only_metadata_guard"]["unchanged"] is True
    assert payload["promotion_decision"]["authorized"] is False
    assert (output / "analysis_benchmark_suite.json").is_file()
    assert (output / "matrix_result.json").is_file()
    assert len(list((output / "trials").glob("*.json"))) == 2


def test_rehashed_trial_identity_and_physical_io_tampering_fail(
    candidate_archive: Path,
) -> None:
    preflight = _preflight(candidate_archive)
    trial = _trial(
        candidate_archive,
        role="candidate",
        order_position=0,
        suite=preflight["suite"],
    )
    identity = copy.deepcopy(trial)
    identity["payload"]["run_path"] = (
        f"{benchmark.CANDIDATE_PARENT_PATH}/other"
    )
    identity["payload_digest"] = canonical_json_sha256(identity["payload"])
    with pytest.raises(ValueError, match="role/run/path binding"):
        benchmark.require_trial_result(identity)

    physical = copy.deepcopy(trial)
    physical["payload"]["physical_io"]["transferred_bytes"] = 1
    physical["payload_digest"] = canonical_json_sha256(physical["payload"])
    with pytest.raises(ValueError, match="must not fabricate"):
        benchmark.require_trial_result(physical)

    availability = copy.deepcopy(trial)
    availability["payload"]["physical_io"]["availability"] = "measured"
    availability["payload_digest"] = canonical_json_sha256(availability["payload"])
    with pytest.raises(ValueError, match="must not fabricate"):
        benchmark.require_trial_result(availability)

    aggregate = copy.deepcopy(trial)
    aggregate["payload"]["primary_access"]["total_wall_seconds"] += 1.0
    aggregate["payload_digest"] = canonical_json_sha256(aggregate["payload"])
    with pytest.raises(ValueError, match="aggregate totals"):
        benchmark.require_trial_result(aggregate)


def test_rehashed_matrix_order_and_promotion_tampering_fail(
    tmp_path: Path,
    candidate_archive: Path,
) -> None:
    result = benchmark.run_benchmark_matrix(
        candidate_archive,
        source_parent=benchmark.SOURCE_PARENT_PATH,
        candidate_parent=benchmark.CANDIDATE_PARENT_PATH,
        source_run="canonical_distance",
        candidate_run="candidate",
        output_dir=tmp_path / ".palette_benchmarks" / "tamper-source",
        cache_state="pytest_uncontrolled_os_cache",
        repetitions=1,
    )
    order = copy.deepcopy(result)
    order["payload"]["trial_order"][0]["roles"] = ["source", "candidate"]
    order["payload_digest"] = canonical_json_sha256(order["payload"])
    with pytest.raises(ValueError, match="not deterministic"):
        benchmark.require_matrix_result(order)

    promotion = copy.deepcopy(result)
    promotion["payload"]["promotion_decision"]["authorized"] = True
    promotion["payload_digest"] = canonical_json_sha256(promotion["payload"])
    with pytest.raises(ValueError, match="cannot authorize promotion"):
        benchmark.require_matrix_result(promotion)

    binding = copy.deepcopy(result)
    binding["payload"]["trials"][0]["payload"]["validation"][
        "source_binding_sha256"
    ] = "f" * 64
    binding["payload"]["trials"][0]["payload_digest"] = canonical_json_sha256(
        binding["payload"]["trials"][0]["payload"]
    )
    binding["payload_digest"] = canonical_json_sha256(binding["payload"])
    with pytest.raises(ValueError, match="matrix/trial identity binding"):
        benchmark.require_matrix_result(binding)

    manifest = copy.deepcopy(result)
    candidate_trial = next(
        trial
        for trial in manifest["payload"]["trials"]
        if trial["payload"]["role"] == "candidate"
    )
    candidate_trial["payload"]["validation"][
        "candidate_manifest_payload_digest"
    ] = "f" * 64
    candidate_trial["payload_digest"] = canonical_json_sha256(
        candidate_trial["payload"]
    )
    manifest["payload_digest"] = canonical_json_sha256(manifest["payload"])
    with pytest.raises(ValueError, match="matrix/trial identity binding"):
        benchmark.require_matrix_result(manifest)

    metadata_guard = copy.deepcopy(result)
    for phase in ("before", "after"):
        metadata_guard["payload"]["archive_read_only_metadata_guard"][phase][
            "files"
        ][0]["size_bytes"] += 1
    metadata_guard["payload_digest"] = canonical_json_sha256(
        metadata_guard["payload"]
    )
    with pytest.raises(ValueError, match="inventory digest mismatch"):
        benchmark.require_matrix_result(metadata_guard)

    physical = copy.deepcopy(result)
    physical["payload"]["physical_io"]["availability"] = "measured"
    physical["payload_digest"] = canonical_json_sha256(physical["payload"])
    with pytest.raises(ValueError, match="fabricates physical I/O"):
        benchmark.require_matrix_result(physical)


def test_malformed_atomic_publication_receipt_fails_trial(
    tmp_path: Path,
) -> None:
    archive, _zarr_root, _run = _publish_canonical(tmp_path)
    consolidate_metadata_capture_expected_warnings(archive)
    materialized = materialize_chaser_distance_base_candidate(
        archive,
        source_run="canonical_distance",
        run_name="candidate",
        scratch_root=tmp_path / "publication-receipt-scratch",
        copy_backend="python",
        apply=True,
    )
    assert materialized["status"] == "complete"
    root = zarr.open_group(str(archive), mode="a", use_consolidated=False)
    candidate = root[f"{benchmark.CANDIDATE_PARENT_PATH}/candidate"]
    original = copy.deepcopy(candidate.attrs["cluster_output_staging"])
    owner_mismatch = copy.deepcopy(original)
    owner_mismatch["publication_owner_uuid"] = "replacement-owner"
    candidate.attrs["cluster_output_staging"] = owner_mismatch
    consolidate_metadata_capture_expected_warnings(archive)
    preflight = _preflight(archive)

    with pytest.raises(ValueError, match="receipt owner differs from the run"):
        _trial(
            archive,
            role="candidate",
            order_position=0,
            suite=preflight["suite"],
        )

    root = zarr.open_group(str(archive), mode="a", use_consolidated=False)
    candidate = root[f"{benchmark.CANDIDATE_PARENT_PATH}/candidate"]
    receipt = copy.deepcopy(original)
    receipt["final_validation"]["valid"] = False
    candidate.attrs["cluster_output_staging"] = receipt
    consolidate_metadata_capture_expected_warnings(archive)
    preflight = _preflight(archive)

    with pytest.raises(ValueError, match="final_validation did not pass exactly"):
        _trial(
            archive,
            role="candidate",
            order_position=0,
            suite=preflight["suite"],
        )


def test_candidate_receipt_tampering_fails_preflight(
    tmp_path: Path,
    candidate_archive: Path,
) -> None:
    archive = tmp_path / "receipt-tamper.zarr"
    shutil.copytree(candidate_archive, archive)
    root = zarr.open_group(str(archive), mode="a", use_consolidated=False)
    candidate = root[f"{benchmark.CANDIDATE_PARENT_PATH}/candidate"]
    receipt = copy.deepcopy(candidate.attrs[ANALYSIS_STORAGE_PLAN_RECEIPT_ATTR])
    receipt["payload"]["storage_profile"]["target_chunk_bytes"] += 1
    receipt["payload_digest"] = canonical_json_sha256(receipt["payload"])
    candidate.attrs[ANALYSIS_STORAGE_PLAN_RECEIPT_ATTR] = receipt
    candidate.attrs[ANALYSIS_STORAGE_PLAN_DIGEST_ATTR] = receipt["payload_digest"]
    consolidate_metadata_capture_expected_warnings(archive)

    with pytest.raises(ValueError, match="Invalid sealed-base candidate"):
        _preflight(archive)


def test_stale_consolidated_candidate_metadata_fails_trial(
    tmp_path: Path,
    candidate_archive: Path,
) -> None:
    archive = tmp_path / "stale.zarr"
    shutil.copytree(candidate_archive, archive)
    preflight = _preflight(archive)
    direct = zarr.open_group(str(archive), mode="a", use_consolidated=False)
    direct[f"{benchmark.CANDIDATE_PARENT_PATH}/candidate"].attrs[
        "post_consolidation_tamper"
    ] = True

    with pytest.raises(RuntimeError, match="Direct/consolidated declaration differs"):
        _trial(
            archive,
            role="candidate",
            order_position=0,
            suite=preflight["suite"],
        )


def test_explicit_parent_name_and_output_safety_fail_before_writing(
    tmp_path: Path,
    candidate_archive: Path,
) -> None:
    with pytest.raises(ValueError, match="source_parent must be exact"):
        benchmark.run_benchmark_matrix(
            candidate_archive,
            source_parent="analysis/other_runs",
            candidate_parent=benchmark.CANDIDATE_PARENT_PATH,
            source_run="canonical_distance",
            candidate_run="candidate",
            output_dir=tmp_path / ".palette_benchmarks" / "wrong-parent",
            cache_state="pytest",
            repetitions=1,
        )
    with pytest.raises(ValueError, match="explicit immutable child"):
        benchmark.run_benchmark_matrix(
            candidate_archive,
            source_parent=benchmark.SOURCE_PARENT_PATH,
            candidate_parent=benchmark.CANDIDATE_PARENT_PATH,
            source_run="latest",
            candidate_run="candidate",
            output_dir=tmp_path / ".palette_benchmarks" / "alias",
            cache_state="pytest",
            repetitions=1,
        )
    with pytest.raises(ValueError, match="must be different names"):
        benchmark.run_benchmark_matrix(
            candidate_archive,
            source_parent=benchmark.SOURCE_PARENT_PATH,
            candidate_parent=benchmark.CANDIDATE_PARENT_PATH,
            source_run="same_name",
            candidate_run="same_name",
            output_dir=tmp_path / ".palette_benchmarks" / "same-name",
            cache_state="pytest",
            repetitions=1,
        )
    unsafe = tmp_path / "results"
    with pytest.raises(ValueError, match="benchmark-only"):
        benchmark.run_benchmark_matrix(
            candidate_archive,
            source_parent=benchmark.SOURCE_PARENT_PATH,
            candidate_parent=benchmark.CANDIDATE_PARENT_PATH,
            source_run="canonical_distance",
            candidate_run="candidate",
            output_dir=unsafe,
            cache_state="pytest",
            repetitions=1,
        )
    assert not unsafe.exists()


def test_default_order_is_rotated_and_deterministic() -> None:
    observed = [
        benchmark._trial_order(seed=17, repetition_index=index) for index in range(5)
    ]
    assert observed == [
        ("candidate", "source"),
        ("source", "candidate"),
        ("candidate", "source"),
        ("source", "candidate"),
        ("candidate", "source"),
    ]
