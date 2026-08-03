from __future__ import annotations

import copy
import os
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest
import zarr

from fisheye.analysis.subject_shape_storage import (
    SUBJECT_SHAPE_ACCESS_AWARE_CANDIDATE_PROFILE_ID,
)
from fisheye.analysis_workflows.materializers import subject_shape as materializer
from fisheye.diagnostics import benchmark_subject_shape_v4_candidate as benchmark
from fisheye.shared.subject_shape_coordinate_publication import (
    SUBJECT_SHAPE_PUBLICATION_OWNER_ATTR,
    SUBJECT_SHAPE_STORAGE_PLAN_ATTR,
)
from fisheye.shared.coordinate_record import coordinate_record_sha256
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from fisheye.shared.zarr.metadata_equivalence import ZarrMetadataEquivalenceError
from tests.unit.fisheye.subject_shape_test_fixtures import (
    resolve_canonical_refined_archive_template,
)
from tests.unit.fisheye.test_subject_shape_runs import _patch_provenance


SOURCE_RUN = "shape_source"
CANDIDATE_RUN = "shape_candidate"


def _rehash(value: dict[str, object]) -> dict[str, object]:
    value["payload_digest"] = canonical_json_sha256(value["payload"])
    return value


@pytest.fixture(scope="session")
def subject_shape_pair(tmp_path_factory: pytest.TempPathFactory) -> Path:
    destination = tmp_path_factory.mktemp("subject-shape-benchmark") / "pair.zarr"
    shutil.copytree(resolve_canonical_refined_archive_template(), destination)
    with pytest.MonkeyPatch.context() as monkeypatch:
        _patch_provenance(monkeypatch)
        monkeypatch.setattr(
            materializer,
            "write_best_effort_run_lineage_attrs",
            lambda *args, **kwargs: None,
        )
        common = {
            "refined_run": "r1",
            "block_rows": 2,
            "execution_backend": "serial_driver",
            "scheduler": "single-threaded",
            "num_workers": 1,
            "shard_copy_workers": 1,
            "native_threads": 1,
            "copy_backend": "python",
            "apply": True,
            "keep_scratch": False,
            "check_capacity": False,
        }
        materializer.materialize_subject_shape(
            destination,
            scratch_root=destination.parent / "source-scratch",
            run_name=SOURCE_RUN,
            output_shard_rows=4,
            stage_command="subject-shape-benchmark-source-fixture",
            **common,
        )
        materializer.materialize_subject_shape(
            destination,
            scratch_root=destination.parent / "candidate-scratch",
            run_name=CANDIDATE_RUN,
            storage_profile=SUBJECT_SHAPE_ACCESS_AWARE_CANDIDATE_PROFILE_ID,
            stage_command="subject-shape-benchmark-candidate-fixture",
            **common,
        )
    return destination


def _preflight(path: Path, *, repetitions: int = 1) -> dict[str, object]:
    return benchmark._preflight(
        path,
        source_path=f"{benchmark.PARENT_PATH}/{SOURCE_RUN}",
        candidate_path=f"{benchmark.PARENT_PATH}/{CANDIDATE_RUN}",
        seed=benchmark.DEFAULT_SEED,
        repetitions=repetitions,
    )


def _trial(path: Path, *, role: str, repetitions: int = 1) -> dict[str, object]:
    preflight = _preflight(path, repetitions=repetitions)
    order = benchmark._trial_order(seed=benchmark.DEFAULT_SEED, repetition_index=0)
    return benchmark.run_single_trial(
        path,
        parent=benchmark.PARENT_PATH,
        source_run=SOURCE_RUN,
        candidate_run=CANDIDATE_RUN,
        role=role,
        repetition_index=0,
        order_position=order.index(role),
        driver_process_id=os.getppid(),
        seed=benchmark.DEFAULT_SEED,
        cache_state="uncontrolled_test_cache",
        suite_manifest=preflight["suite"],
    )


@pytest.fixture(scope="session")
def trial_pair(subject_shape_pair: Path) -> dict[str, dict[str, object]]:
    return {
        role: _trial(subject_shape_pair, role=role)
        for role in ("source", "candidate")
    }


@pytest.fixture(scope="session")
def matrix_evidence(
    subject_shape_pair: Path,
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[dict[str, object], Path]:
    output = tmp_path_factory.mktemp("subject-shape-matrix") / "benchmark-evidence"
    result = benchmark.run_benchmark_matrix(
        subject_shape_pair,
        parent=benchmark.PARENT_PATH,
        source_run=SOURCE_RUN,
        candidate_run=CANDIDATE_RUN,
        output_dir=output,
        cache_state="uncontrolled_test_cache",
        repetitions=1,
    )
    return result, output


def test_preflight_binds_separate_manifests_and_complete_decoded_equality(
    subject_shape_pair: Path,
) -> None:
    result = _preflight(subject_shape_pair)
    assert result["source_manifest_sha256"] != result["candidate_manifest_sha256"]
    assert result["candidate_source_manifest_link_sha256"]
    assert result["candidate_publication_receipt_sha256"]
    assert result["array_paths"] == sorted(result["logical_arrays"])
    assert result["source_refined_run"] == "r1"


def test_source_and_candidate_trials_are_exact_and_logically_equal(
    trial_pair: dict[str, dict[str, object]],
) -> None:
    source = trial_pair["source"]
    candidate = trial_pair["candidate"]
    benchmark.require_trial_result(source)
    benchmark.require_trial_result(candidate)
    assert source["payload"]["logical_arrays"] == candidate["payload"]["logical_arrays"]
    assert {
        path: record["selection_digest"]
        for path, record in source["payload"]["primary_access"]["arrays"].items()
    } == {
        path: record["selection_digest"]
        for path, record in candidate["payload"]["primary_access"]["arrays"].items()
    }
    assert candidate["payload"]["physical_io"]["transferred_bytes"] is None


def test_fresh_process_matrix_is_read_only_and_nonpromoting(
    matrix_evidence: tuple[dict[str, object], Path],
) -> None:
    result, output = matrix_evidence
    benchmark.require_matrix_result(result)
    assert result["payload"]["promotion_decision"]["authorized"] is False
    assert result["payload"]["archive_read_only_metadata_guard"]["unchanged"] is True
    assert (output / "matrix_result.json").is_file()
    assert len(list((output / "trials").glob("*.json"))) == 2


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda payload: payload.__setitem__("run_path", "analysis/subject_shape_runs/other"),
            "role/run/path",
        ),
        (
            lambda payload: payload["physical_io"].__setitem__("transferred_bytes", 1),
            "physical-I/O",
        ),
        (
            lambda payload: payload["full_scan"].__setitem__("total_decoded_bytes", 0),
            "aggregate totals",
        ),
        (
            lambda payload: payload["validation"].__setitem__(
                "run_manifest_sha256", "0" * 64
            ),
            "identity binding",
        ),
    ],
)
def test_rehashed_trial_tampering_fails(
    trial_pair: dict[str, dict[str, object]],
    mutation,
    message: str,
) -> None:
    evidence = copy.deepcopy(trial_pair["candidate"])
    mutation(evidence["payload"])
    _rehash(evidence)
    with pytest.raises(ValueError, match=message):
        benchmark.require_trial_result(evidence)


def test_coordinated_rehash_of_embedded_producer_seal_still_fails(
    trial_pair: dict[str, dict[str, object]],
) -> None:
    evidence = copy.deepcopy(trial_pair["candidate"])
    payload = evidence["payload"]
    artifacts = payload["contract_artifacts"]
    link = artifacts["candidate_retained_producer_manifest_link"]
    producer = link["source_manifest"]
    candidate_manifest = artifacts["candidate_final_manifest"]
    transformed = set(candidate_manifest["coordinate_descriptors"]) | set(
        candidate_manifest["scalar_surfaces"]
    )
    path = sorted(set(producer["arrays"]) - transformed)[0]
    producer["arrays"][path]["content_sha256"] = "0" * 64
    producer_sha256 = coordinate_record_sha256(producer)
    link["source_manifest_sha256"] = producer_sha256
    for role in ("source", "candidate"):
        consumed = artifacts[f"{role}_consumed_unbound_manifest"]
        consumed["arrays"][path]["content_sha256"] = "0" * 64
        consumed_sha256 = coordinate_record_sha256(consumed)
        final_manifest = artifacts[f"{role}_final_manifest"]
        final_manifest["consumed_unbound_stage"]["record_sha256"] = consumed_sha256
        final_manifest_sha256 = coordinate_record_sha256(final_manifest)
        payload[f"{role}_manifest_sha256"] = final_manifest_sha256
        run_name = SOURCE_RUN if role == "source" else CANDIDATE_RUN
        declarations = artifacts["metadata_documents"][role]
        run_attrs = declarations[f"{benchmark.PARENT_PATH}/{run_name}"]["attributes"]
        run_attrs[benchmark.SUBJECT_SHAPE_MANIFEST_ATTR] = final_manifest
        run_attrs[f"{benchmark.SUBJECT_SHAPE_MANIFEST_ATTR}_sha256"] = (
            final_manifest_sha256
        )
        consumed_attrs = declarations[
            f"{benchmark.PARENT_PATH}/{run_name}/"
            "coordinate_records/consumed_unbound_stage"
        ]["attributes"]
        consumed_attrs[benchmark.SUBJECT_SHAPE_CONSUMED_UNBOUND_STAGE_ATTR] = consumed
        if role == "candidate":
            payload["validation"]["run_manifest_sha256"] = final_manifest_sha256
            payload["metadata"]["declarations"] = copy.deepcopy(declarations)
            payload["metadata"]["subtree_declarations_digest"] = (
                canonical_json_sha256(payload["metadata"]["declarations"])
            )
    link_sha256 = coordinate_record_sha256(link)
    payload["candidate_source_manifest_link_sha256"] = link_sha256
    payload["validation"]["candidate_source_manifest_link_sha256"] = link_sha256
    candidate_declarations = artifacts["metadata_documents"]["candidate"]
    candidate_run_attrs = candidate_declarations[
        f"{benchmark.PARENT_PATH}/{CANDIDATE_RUN}"
    ]["attributes"]
    candidate_run_attrs[benchmark.SUBJECT_SHAPE_STORAGE_SOURCE_MANIFEST_ATTR] = link
    candidate_run_attrs[
        f"{benchmark.SUBJECT_SHAPE_STORAGE_SOURCE_MANIFEST_ATTR}_sha256"
    ] = link_sha256
    payload["metadata"]["declarations"] = copy.deepcopy(candidate_declarations)
    payload["metadata"]["subtree_declarations_digest"] = canonical_json_sha256(
        payload["metadata"]["declarations"]
    )
    _rehash(evidence)
    with pytest.raises(
        ValueError, match="producer-sealed array|consumed-stage array"
    ):
        benchmark.require_trial_result(evidence)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda payload: payload["trial_order"][0]["roles"].reverse(),
            "order",
        ),
        (
            lambda payload: payload["promotion_decision"].__setitem__("authorized", True),
            "cannot authorize",
        ),
        (
            lambda payload: payload.__setitem__("candidate_manifest_sha256", "0" * 64),
            "identity binding",
        ),
        (
            lambda payload: payload["physical_io"].__setitem__("request_count", 2),
            "physical-I/O",
        ),
        (
            lambda payload: payload["archive_read_only_metadata_guard"].__setitem__(
                "unchanged", False
            ),
            "immutability",
        ),
    ],
)
def test_rehashed_matrix_tampering_fails(
    matrix_evidence: tuple[dict[str, object], Path],
    mutation,
    message: str,
) -> None:
    result, _ = matrix_evidence
    evidence = copy.deepcopy(result)
    mutation(evidence["payload"])
    _rehash(evidence)
    with pytest.raises(ValueError, match=message):
        benchmark.require_matrix_result(evidence)


def test_publication_receipt_rejects_owner_and_final_validation_tampering(
    subject_shape_pair: Path,
) -> None:
    root = zarr.open_group(str(subject_shape_pair), mode="r", use_consolidated=True)
    run_path = f"{benchmark.PARENT_PATH}/{CANDIDATE_RUN}"
    facts = benchmark._validate_role(root, run_path=run_path, role="candidate")
    attrs = dict(facts["group"].attrs)
    attrs[SUBJECT_SHAPE_PUBLICATION_OWNER_ATTR] = "wrong-owner"
    with pytest.raises(ValueError, match="receipt contract differs"):
        benchmark._publication_receipt(
            SimpleNamespace(attrs=attrs),
            role="candidate",
            archive=subject_shape_pair,
            run_path=run_path,
            run_name=CANDIDATE_RUN,
            row_count=facts["row_count"],
            manifest_sha256=facts["manifest_sha256"],
            storage_receipt=facts["receipt_manifest"],
        )
    attrs = dict(facts["group"].attrs)
    receipt = copy.deepcopy(attrs["cluster_output_staging"])
    receipt["final_validation"]["valid"] = False
    attrs["cluster_output_staging"] = receipt
    with pytest.raises(ValueError, match="final validation did not pass"):
        benchmark._publication_receipt(
            SimpleNamespace(attrs=attrs),
            role="candidate",
            archive=subject_shape_pair,
            run_path=run_path,
            run_name=CANDIDATE_RUN,
            row_count=facts["row_count"],
            manifest_sha256=facts["manifest_sha256"],
            storage_receipt=facts["receipt_manifest"],
        )


def test_candidate_storage_receipt_tamper_fails_preflight(
    subject_shape_pair: Path,
    tmp_path: Path,
) -> None:
    archive = tmp_path / "tampered.zarr"
    shutil.copytree(subject_shape_pair, archive)
    root = zarr.open_group(str(archive), mode="a", use_consolidated=False)
    candidate = root[benchmark.PARENT_PATH][CANDIDATE_RUN]
    receipt = copy.deepcopy(candidate.attrs[SUBJECT_SHAPE_STORAGE_PLAN_ATTR])
    receipt["payload_digest"] = "0" * 64
    candidate.attrs[SUBJECT_SHAPE_STORAGE_PLAN_ATTR] = receipt
    zarr.consolidate_metadata(root.store)
    with pytest.raises(ValueError, match="storage receipt"):
        _preflight(archive)


def test_stale_consolidated_metadata_fails_trial(
    subject_shape_pair: Path,
    tmp_path: Path,
) -> None:
    archive = tmp_path / "stale.zarr"
    shutil.copytree(subject_shape_pair, archive)
    root = zarr.open_group(str(archive), mode="a", use_consolidated=False)
    root[benchmark.PARENT_PATH][CANDIDATE_RUN].attrs["diagnostic_only_tamper"] = True
    with pytest.raises(
        ZarrMetadataEquivalenceError,
        match="Direct/consolidated declaration differs",
    ):
        _trial(archive, role="candidate")


def test_paths_fail_closed(subject_shape_pair: Path, tmp_path: Path) -> None:
    preflight = _preflight(subject_shape_pair)
    with pytest.raises(ValueError, match="parent"):
        benchmark.run_single_trial(
            subject_shape_pair,
            parent="analysis/other_runs",
            source_run=SOURCE_RUN,
            candidate_run=CANDIDATE_RUN,
            role="candidate",
            repetition_index=0,
            order_position=0,
            driver_process_id=os.getppid(),
            seed=benchmark.DEFAULT_SEED,
            cache_state="explicit",
            suite_manifest=preflight["suite"],
        )
    with pytest.raises(ValueError, match="differ"):
        benchmark.run_benchmark_matrix(
            subject_shape_pair,
            parent=benchmark.PARENT_PATH,
            source_run=SOURCE_RUN,
            candidate_run=SOURCE_RUN,
            output_dir=tmp_path / "subject-shape-benchmark-same-name",
            cache_state="explicit",
            repetitions=1,
        )
    with pytest.raises(ValueError, match="disjoint"):
        benchmark._safe_output(subject_shape_pair / "benchmark-output", archive=subject_shape_pair)


def test_trial_order_is_rotated_and_deterministic() -> None:
    first = benchmark._trial_order(seed=17, repetition_index=0)
    second = benchmark._trial_order(seed=17, repetition_index=1)
    assert first == tuple(reversed(second))
    assert benchmark._trial_order(seed=17, repetition_index=0) == first


def test_matrix_rejects_replayed_fresh_process_identity(
    matrix_evidence: tuple[dict[str, object], Path],
) -> None:
    result, _ = matrix_evidence
    evidence = copy.deepcopy(result)
    trials = evidence["payload"]["trials"]
    trials[1]["payload"]["process_id"] = trials[0]["payload"]["process_id"]
    _rehash(trials[1])
    _rehash(evidence)
    with pytest.raises(ValueError, match="fresh-process identities"):
        benchmark.require_matrix_result(evidence)


def test_matrix_rejects_duplicate_role_position_evidence(
    matrix_evidence: tuple[dict[str, object], Path],
) -> None:
    result, _ = matrix_evidence
    evidence = copy.deepcopy(result)
    trials = evidence["payload"]["trials"]
    trials[1]["payload"]["order_position"] = trials[0]["payload"][
        "order_position"
    ]
    _rehash(trials[1])
    _rehash(evidence)
    with pytest.raises(ValueError, match="role/position"):
        benchmark.require_matrix_result(evidence)


def test_preflight_rejects_selected_run_symlink_before_zarr_open(
    subject_shape_pair: Path,
    tmp_path: Path,
) -> None:
    archive = tmp_path / "selected-symlink.zarr"
    shutil.copytree(subject_shape_pair, archive)
    selected = archive.joinpath(
        *benchmark.PARENT_PATH.split("/"), CANDIDATE_RUN
    )
    retained = selected.with_name(f"{selected.name}-real")
    selected.rename(retained)
    selected.symlink_to(retained.name, target_is_directory=True)
    with pytest.raises(ValueError, match="candidate run path contains a symlink"):
        _preflight(archive)


def test_preflight_rejects_refined_dependency_symlink_before_zarr_open(
    subject_shape_pair: Path,
    tmp_path: Path,
) -> None:
    archive = tmp_path / "dependency-symlink.zarr"
    shutil.copytree(subject_shape_pair, archive)
    dependency = archive / "refined_subject_masks_runs" / "r1"
    retained = dependency.with_name(f"{dependency.name}-real")
    dependency.rename(retained)
    dependency.symlink_to(retained.name, target_is_directory=True)
    with pytest.raises(ValueError, match="refined-mask source run path contains a symlink"):
        _preflight(archive)


def test_coordinated_rehash_of_embedded_metadata_still_fails(
    matrix_evidence: tuple[dict[str, object], Path],
) -> None:
    result, _ = matrix_evidence
    evidence = copy.deepcopy(result)
    candidate_run_path = f"{benchmark.PARENT_PATH}/{CANDIDATE_RUN}"
    absolute_array_path = f"{candidate_run_path}/body_frame/axis_valid"

    def mutate_documents(artifacts: dict[str, object]) -> None:
        declaration = artifacts["metadata_documents"]["candidate"][
            absolute_array_path
        ]
        declaration["shape"] = [3]

    mutate_documents(evidence["payload"]["contract_artifacts"])
    for trial in evidence["payload"]["trials"]:
        mutate_documents(trial["payload"]["contract_artifacts"])
        if trial["payload"]["role"] == "candidate":
            declarations = trial["payload"]["metadata"]["declarations"]
            declarations[absolute_array_path]["shape"] = [3]
            trial["payload"]["metadata"]["subtree_declarations_digest"] = (
                canonical_json_sha256(declarations)
            )
        _rehash(trial)
    _rehash(evidence)
    with pytest.raises(ValueError, match="candidate array metadata.*differs"):
        benchmark.require_matrix_result(evidence)


def test_manifest_cannot_reclassify_invariant_array_as_transformed(
    trial_pair: dict[str, dict[str, object]],
) -> None:
    evidence = copy.deepcopy(trial_pair["candidate"])
    payload = evidence["payload"]
    artifacts = payload["contract_artifacts"]
    invariant_path = "body_frame/failure_reason_bytes"
    for role, run_name in (
        ("source", SOURCE_RUN),
        ("candidate", CANDIDATE_RUN),
    ):
        manifest = artifacts[f"{role}_final_manifest"]
        manifest["coordinate_descriptors"][invariant_path] = {
            "record_ref": (
                f"/{benchmark.PARENT_PATH}/{run_name}/{invariant_path}"
                "@canonical_coordinate_descriptor"
            ),
            "descriptor_sha256": "0" * 64,
        }
        manifest_sha256 = coordinate_record_sha256(manifest)
        payload[f"{role}_manifest_sha256"] = manifest_sha256
        metadata_attrs = artifacts["metadata_documents"][role][
            f"{benchmark.PARENT_PATH}/{run_name}"
        ]["attributes"]
        metadata_attrs[benchmark.SUBJECT_SHAPE_MANIFEST_ATTR] = manifest
        metadata_attrs[f"{benchmark.SUBJECT_SHAPE_MANIFEST_ATTR}_sha256"] = (
            manifest_sha256
        )
        if role == "candidate":
            payload["validation"]["run_manifest_sha256"] = manifest_sha256
            payload["metadata"]["declarations"] = copy.deepcopy(
                artifacts["metadata_documents"][role]
            )
            payload["metadata"]["subtree_declarations_digest"] = (
                canonical_json_sha256(payload["metadata"]["declarations"])
            )
    _rehash(evidence)
    with pytest.raises(ValueError, match="transform vocabulary is not executable"):
        benchmark.require_trial_result(evidence)
