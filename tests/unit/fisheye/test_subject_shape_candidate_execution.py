from __future__ import annotations

# Imported pytest fixtures are intentionally rebound as test parameters.
# ruff: noqa: F401, F811

from copy import deepcopy
import json
from pathlib import Path
import shutil
import subprocess
import uuid

import pytest
import zarr

from fisheye.analysis_workflows.analysis_candidate_execution import (
    PhysicalIOScope,
    build_candidate_execution_request,
    require_candidate_execution_receipt,
)
from fisheye.analysis_workflows.analysis_candidate_execution_catalog import (
    ANALYSIS_CANDIDATE_EXECUTION_ADAPTER_BY_STAGE,
)
from fisheye.analysis_workflows.analysis_candidate_invocation import (
    CandidateInvocationContract,
    build_subject_shape_invocation,
    require_candidate_invocation_manifest,
)
from fisheye.analysis_workflows.materializers.subject_shape import (
    SUBJECT_SHAPE_EXECUTION_PHASE_ORDER,
    materialize_subject_shape_execution_candidate,
    tombstone_subject_shape_execution_candidate,
)
from fisheye.analysis_workflows.subject_shape_candidate_execution import (
    SUBJECT_SHAPE_EXECUTION_FAMILY_ID,
    build_subject_shape_execution_suite,
    compute_subject_shape_logical_hashes,
    require_subject_shape_execution_suite,
)
from fisheye.diagnostics.subject_shape_candidate_execution import (
    SubjectShapeCandidateExecutionFailed,
    run_subject_shape_candidate_fresh_process,
    subject_shape_selector_snapshot_sha256,
)
from fisheye.shared.subject_shape_coordinate_publication import (
    load_persisted_subject_shape_coordinate_publication,
)
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from tests.unit.fisheye.test_subject_shape_runs import (
    canonical_refined_template,
    canonical_subject_shape_template,
)

SOURCE_PATH = "analysis/subject_shape_runs/shape_001"


def _invocation() -> dict[str, object]:
    return build_subject_shape_invocation(
        source_manifest_sha256="1" * 64,
        source_refined_subject_masks_run="refined_masks_001",
        source_refined_authority_sha256="2" * 64,
        storage_profile_id="subject_shape_access_aware_candidate_v1",
        block_rows=16_384,
        output_shard_rows=131_072,
        execution_backend="serial_driver",
        scheduler="single-threaded",
        num_workers=1,
        shard_copy_workers=1,
        native_threads=1,
        copy_backend="python",
        keep_scratch=False,
        check_capacity=True,
    )


def test_subject_shape_invocation_grammar_is_exact_and_closed() -> None:
    invocation = _invocation()
    require_candidate_invocation_manifest(
        invocation,
        expected_contract=CandidateInvocationContract.SUBJECT_SHAPE_V1,
    )

    unexpected = deepcopy(invocation)
    unexpected["payload"]["parameters"]["unexpected"] = True
    unexpected["payload_digest"] = canonical_json_sha256(unexpected["payload"])
    with pytest.raises(ValueError, match="field set differs"):
        require_candidate_invocation_manifest(unexpected)

    bool_as_int = deepcopy(invocation)
    bool_as_int["payload"]["parameters"]["block_rows"] = True
    bool_as_int["payload_digest"] = canonical_json_sha256(bool_as_int["payload"])
    with pytest.raises(ValueError, match="block_rows"):
        require_candidate_invocation_manifest(bool_as_int)

    invalid_refined_run = deepcopy(invocation)
    invalid_refined_run["payload"]["parameters"][
        "source_refined_subject_masks_run"
    ] = "refined_subject_masks_runs/run"
    invalid_refined_run["payload_digest"] = canonical_json_sha256(
        invalid_refined_run["payload"]
    )
    with pytest.raises(ValueError, match="run name"):
        require_candidate_invocation_manifest(invalid_refined_run)


def _rehash_suite(value: dict[str, object]) -> None:
    value["payload"]["storage_plan_receipt"]["payload_digest"] = canonical_json_sha256(
        value["payload"]["storage_plan_receipt"]["payload"]
    )
    value["payload_digest"] = canonical_json_sha256(value["payload"])


def test_subject_shape_suite_replays_exact_v4_declarations(
    canonical_subject_shape_template: tuple[Path, dict[str, object]],
) -> None:
    archive, _summary = canonical_subject_shape_template
    root = zarr.open_group(str(archive), mode="r", use_consolidated=False)
    suite = build_subject_shape_execution_suite(
        root[SOURCE_PATH],
        repetitions=1,
    )

    require_subject_shape_execution_suite(SUBJECT_SHAPE_EXECUTION_FAMILY_ID, suite)
    receipt = suite["payload"]["storage_plan_receipt"]["payload"]
    assert receipt["storage_profile"]["profile_id"] == (
        "subject_shape_access_aware_candidate_v1"
    )
    assert len(receipt["arrays"]) >= 50


def test_subject_shape_suite_rejects_rehashed_declaration_tamper(
    canonical_subject_shape_template: tuple[Path, dict[str, object]],
) -> None:
    archive, _summary = canonical_subject_shape_template
    root = zarr.open_group(str(archive), mode="r", use_consolidated=False)
    suite = build_subject_shape_execution_suite(root[SOURCE_PATH], repetitions=1)
    tampered = deepcopy(suite)
    record = tampered["payload"]["storage_plan_receipt"]["payload"]["arrays"][0]
    record["declaration"]["null_semantics"] = "tampered but rehashed"
    _rehash_suite(tampered)

    with pytest.raises(ValueError, match="publication case is not bound"):
        require_subject_shape_execution_suite(
            SUBJECT_SHAPE_EXECUTION_FAMILY_ID,
            tampered,
        )


def test_execution_materializer_stages_computes_and_tombstones_owned_candidate(
    tmp_path: Path,
    canonical_subject_shape_template: tuple[Path, dict[str, object]],
) -> None:
    template, _summary = canonical_subject_shape_template
    benchmark = tmp_path / ".palette_benchmarks" / "subject-shape"
    benchmark.mkdir(parents=True)
    archive = benchmark / "fixture.zarr"
    shutil.copytree(template, archive)
    root = zarr.open_group(str(archive), mode="r", use_consolidated=False)
    expected_hashes = compute_subject_shape_logical_hashes(root[SOURCE_PATH])
    parent = root["analysis/subject_shape_runs"]
    selectors_before = {
        name: parent.attrs.get(name) for name in ("latest", "latest_complete")
    }
    binding = {
        "schema_id": "palette.analysis_candidate_execution_binding",
        "schema_version": 1,
        "execution_id": "subject_shape_test_execution",
        "request_payload_digest": "1" * 64,
    }
    accepted: dict[str, object] = {}

    def accept(_root, _parent, candidate):
        accepted["manifest"] = candidate.attrs["subject_shape_publication_manifest"]
        return {"accepted": True}

    result = materialize_subject_shape_execution_candidate(
        archive,
        source_run=SOURCE_PATH,
        run_name="typed_candidate",
        scratch_root=tmp_path / "subject-shape-execution-scratch",
        block_rows=2,
        output_shard_rows=8,
        execution_backend="serial_driver",
        scheduler="single-threaded",
        num_workers=1,
        shard_copy_workers=1,
        native_threads=1,
        copy_backend="python",
        keep_scratch=False,
        check_capacity=False,
        execution_binding=binding,
        expected_source_logical_hashes=expected_hashes,
        publication_acceptance_validator=accept,
    )

    assert result["status"] == "complete"
    assert result["source_logical_manifest_sha256"] == (
        result["published_logical_manifest_sha256"]
    )
    assert [phase["name"] for phase in result["runtime_telemetry"]["phases"]] == list(
        SUBJECT_SHAPE_EXECUTION_PHASE_ORDER
    )
    assert result["caller_acceptance"] == {"accepted": True}
    assert accepted
    assert not (tmp_path / "subject-shape-execution-scratch").exists()

    direct = zarr.open_group(
        str(archive), mode="r", zarr_format=3, use_consolidated=False
    )
    consolidated = zarr.open_group(
        str(archive), mode="r", zarr_format=3, use_consolidated=True
    )
    for view in (direct, consolidated):
        assert {
            name: view["analysis/subject_shape_runs"].attrs.get(name)
            for name in ("latest", "latest_complete")
        } == selectors_before
        candidate = view["analysis/subject_shape_runs/typed_candidate"]
        assert candidate.attrs["stage_selector_eligible"] is False
        assert (
            candidate.attrs["cluster_output_staging"][
                "analysis_candidate_execution_binding"
            ]
            == binding
        )

    tombstone = tombstone_subject_shape_execution_candidate(
        archive,
        run_name="typed_candidate",
        expected_execution_binding=binding,
        failure_phase="driver_receipt_publication",
        error_type="OSError",
        error_message="simulated sidecar failure",
    )
    assert tombstone["tombstoned"] is True
    for consolidated_mode in (False, True):
        failed = zarr.open_group(
            str(archive),
            mode="r",
            zarr_format=3,
            use_consolidated=consolidated_mode,
        )["analysis/subject_shape_runs/typed_candidate"]
        assert failed.attrs["palette_run_completion_status"] == "failed"
        assert failed.attrs["stage_selector_eligible"] is False
        assert (
            failed.attrs["analysis_candidate_execution_tombstone"]["execution_binding"]
            == binding
        )


def test_fresh_process_publishes_exact_subject_shape_candidate_and_receipt(
    tmp_path: Path,
    canonical_subject_shape_template: tuple[Path, dict[str, object]],
) -> None:
    template, _summary = canonical_subject_shape_template
    benchmark = tmp_path / ".palette_benchmarks" / "subject-shape-runner"
    benchmark.mkdir(parents=True)
    archive = benchmark / "fixture.zarr"
    shutil.copytree(template, archive)
    root = zarr.open_group(str(archive), mode="r", use_consolidated=False)
    source = root[SOURCE_PATH]
    publication = load_persisted_subject_shape_coordinate_publication(
        root,
        SOURCE_PATH,
    )
    source_hashes = compute_subject_shape_logical_hashes(source)
    suite = build_subject_shape_execution_suite(source, repetitions=1)
    invocation = build_subject_shape_invocation(
        source_manifest_sha256=publication.manifest.record_sha256,
        source_refined_subject_masks_run=source.attrs[
            "source_refined_subject_masks_run"
        ],
        source_refined_authority_sha256=(
            publication.source.scientific_manifest.record_sha256
        ),
        storage_profile_id="subject_shape_access_aware_candidate_v1",
        block_rows=2,
        output_shard_rows=8,
        execution_backend="serial_driver",
        scheduler="single-threaded",
        num_workers=1,
        shard_copy_workers=1,
        native_threads=1,
        copy_backend="python",
        keep_scratch=False,
        check_capacity=False,
    )
    adapter = ANALYSIS_CANDIDATE_EXECUTION_ADAPTER_BY_STAGE["subject_shape"]
    probe = benchmark / "protected.json"
    probe.write_text("{}\n", encoding="utf-8")
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    def build_request(*, identity: str, candidate: str) -> dict[str, object]:
        return build_candidate_execution_request(
            execution_id=f"subject_shape_{uuid.uuid4().hex}",
            adapter_manifest=adapter.as_manifest(),
            invocation=invocation,
            benchmark_suite=suite,
            archive_path=archive,
            source_run_path=SOURCE_PATH,
            candidate_run_path=f"analysis/subject_shape_runs/{candidate}",
            scratch_root=Path("/tmp") / f"palette-shape-runner-{uuid.uuid4().hex}",
            source_identity_sha256=identity,
            palette_commit=commit,
            repetition_index=0,
            candidate_order_index=0,
            candidate_order_count=1,
            cache_state="fresh",
            physical_io_scope=PhysicalIOScope.UNAVAILABLE,
            selector_before_sha256=subject_shape_selector_snapshot_sha256(archive),
            registry_probe_path=probe,
            production_profiles_probe_path=probe,
        )

    accepted = build_request(
        identity=canonical_json_sha256(source_hashes),
        candidate="typed_fresh_candidate",
    )
    request_path = benchmark / "request.json"
    request_path.write_text(
        json.dumps(accepted, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    receipt_path = benchmark / "receipt.json"
    attempt_path = benchmark / "attempt.json"

    receipt = run_subject_shape_candidate_fresh_process(
        request_path,
        receipt_path=receipt_path,
        attempt_path=attempt_path,
    )

    require_candidate_execution_receipt(
        receipt,
        expected_request_payload_digest=accepted["payload_digest"],
    )
    assert receipt_path.is_file()
    assert not attempt_path.exists()
    assert receipt["payload"]["logical_equality"]["equal"] is True
    assert receipt["payload"]["logical_equality"]["compared_array_count"] == len(
        source_hashes
    )
    final = zarr.open_group(str(archive), mode="r", use_consolidated=False)
    candidate = final["analysis/subject_shape_runs/typed_fresh_candidate"]
    assert candidate.attrs["stage_selector_eligible"] is False
    assert candidate.attrs["subject_shape_storage_candidate"]["promotion_status"] == (
        "unpromoted_candidate"
    )

    rejected = build_request(identity="0" * 64, candidate="typed_failed_candidate")
    rejected_path = benchmark / "rejected-request.json"
    rejected_path.write_text(
        json.dumps(rejected, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    rejected_receipt = benchmark / "rejected-receipt.json"
    rejected_attempt = benchmark / "rejected-attempt.json"
    with pytest.raises(
        SubjectShapeCandidateExecutionFailed,
        match="decoded identity differs",
    ):
        run_subject_shape_candidate_fresh_process(
            rejected_path,
            receipt_path=rejected_receipt,
            attempt_path=rejected_attempt,
        )
    assert rejected_attempt.is_file()
    assert not rejected_receipt.exists()


def test_execution_materializer_acceptance_failure_is_fail_closed(
    tmp_path: Path,
    canonical_subject_shape_template: tuple[Path, dict[str, object]],
) -> None:
    template, _summary = canonical_subject_shape_template
    benchmark = tmp_path / ".palette_benchmarks" / "subject-shape-failure"
    benchmark.mkdir(parents=True)
    archive = benchmark / "fixture.zarr"
    shutil.copytree(template, archive)
    root = zarr.open_group(str(archive), mode="r", use_consolidated=False)
    expected_hashes = compute_subject_shape_logical_hashes(root[SOURCE_PATH])
    binding = {
        "schema_id": "palette.analysis_candidate_execution_binding",
        "schema_version": 1,
        "execution_id": "subject_shape_failed_execution",
        "request_payload_digest": "2" * 64,
    }

    def reject(*_args):
        raise RuntimeError("simulated atomic acceptance rejection")

    with pytest.raises(RuntimeError, match="atomic acceptance rejection") as raised:
        materialize_subject_shape_execution_candidate(
            archive,
            source_run=SOURCE_PATH,
            run_name="rejected_candidate",
            scratch_root=tmp_path / "subject-shape-rejected-scratch",
            block_rows=2,
            output_shard_rows=8,
            execution_backend="serial_driver",
            scheduler="single-threaded",
            num_workers=1,
            shard_copy_workers=1,
            native_threads=1,
            copy_backend="python",
            keep_scratch=True,
            check_capacity=False,
            execution_binding=binding,
            expected_source_logical_hashes=expected_hashes,
            publication_acceptance_validator=reject,
        )
    telemetry = raised.value.palette_runtime_telemetry
    atomic_phase = next(
        phase for phase in telemetry["phases"] if phase["name"] == "atomic_publication"
    )
    assert atomic_phase["outcome"] == "error"
    direct = zarr.open_group(str(archive), mode="r", use_consolidated=False)
    rejected = direct["analysis/subject_shape_runs/rejected_candidate"]
    assert rejected.attrs["palette_run_completion_status"] == "failed"
    assert rejected.attrs["stage_selector_eligible"] is False
