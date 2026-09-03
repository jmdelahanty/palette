from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import uuid

import pytest
import zarr

from fisheye.analysis import subject_shape_runs
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
    candidate_invocation_contract_is_frozen,
)
from fisheye.analysis_workflows.materializers.tail_kinematics import (
    materialize_tail_kinematics,
)
from fisheye.analysis_workflows.tail_kinematics_candidate_execution import (
    build_tail_kinematics_execution_suite,
    build_tail_kinematics_invocation,
    compute_tail_kinematics_logical_hashes,
)
from fisheye.diagnostics.tail_kinematics_candidate_execution import (
    TailKinematicsCandidateExecutionFailed,
    run_tail_kinematics_candidate_fresh_process,
    tail_kinematics_selector_snapshot_sha256,
)
from fisheye.shared import tail_coordinate_publication
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from fisheye.shared.zarr_helpers import (
    consolidate_metadata_capture_expected_warnings,
)
from tests.unit.fisheye.subject_shape_test_fixtures import (
    resolve_canonical_refined_archive_template,
)
from tests.unit.fisheye.test_subject_shape_coordinate_publication import (
    _patch_provenance as patch_shape_provenance,
)

_TAIL_SHARED_INTEGRATED = bool(
    candidate_invocation_contract_is_frozen(
        CandidateInvocationContract.TAIL_KINEMATICS_V1
    )
    and ANALYSIS_CANDIDATE_EXECUTION_ADAPTER_BY_STAGE[
        "tail_kinematics"
    ].runner_status.value
    == "implemented"
)


@pytest.mark.skipif(
    not _TAIL_SHARED_INTEGRATED,
    reason="shared tail invocation/catalog integration is owned by the parent branch",
)
def test_fresh_process_publishes_exact_tail_candidate_and_receipt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    benchmark = tmp_path / ".palette_benchmarks" / "tail-execution"
    benchmark.mkdir(parents=True)
    archive = benchmark / "fixture.zarr"
    shutil.copytree(resolve_canonical_refined_archive_template(), archive)
    root = zarr.open_group(str(archive), mode="r+", use_consolidated=False)
    patch_shape_provenance(monkeypatch)
    subject_shape_runs.write_subject_shape_run_group(
        root,
        refined_run="r1",
        run_name="shape_1",
        chunk_size=2,
    )
    consolidate_metadata_capture_expected_warnings(archive)
    materialize_tail_kinematics(
        archive,
        scratch_root=tmp_path / "tail-source-scratch",
        shape_run="shape_1",
        run_name="tail_source",
        tail_angle_sample_count=10,
        block_rows=2,
        output_shard_rows=4,
        execution_backend="serial",
        num_workers=1,
        copy_backend="python",
        apply=True,
        keep_scratch=False,
        check_capacity=False,
        stage_command="tail execution source fixture",
    )
    consolidate_metadata_capture_expected_warnings(archive)
    root = zarr.open_group(str(archive), mode="r", use_consolidated=False)
    source = root["analysis/tail_kinematics_runs/tail_source"]
    source_publication = (
        tail_coordinate_publication.load_tail_kinematics_coordinate_publication(
            root,
            "analysis/tail_kinematics_runs/tail_source",
        )
    )
    source_hashes = compute_tail_kinematics_logical_hashes(source)
    suite = build_tail_kinematics_execution_suite(
        source,
        scale_id="unit_live_runner",
        description="real canonical tail execution fixture",
        repetitions=1,
    )
    invocation = build_tail_kinematics_invocation(
        source_subject_shape_run="shape_1",
        source_tail_coordinate_manifest_sha256=(
            source_publication.manifest.record_sha256
        ),
        source_subject_shape_manifest_sha256=(
            source_publication.source.manifest.record_sha256
        ),
        tail_angle_sample_count=10,
        block_rows=2,
        output_shard_rows=4,
        storage_profile_id="published_http_v1",
        copy_backend="python",
        keep_scratch=False,
        check_capacity=False,
    )
    adapter = ANALYSIS_CANDIDATE_EXECUTION_ADAPTER_BY_STAGE["tail_kinematics"]
    probe = benchmark / "protected.json"
    probe.write_text("{}\n", encoding="utf-8")
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    scratch = Path("/tmp") / f"palette-tail-runner-{uuid.uuid4().hex}"
    request = build_candidate_execution_request(
        execution_id=f"tail_{uuid.uuid4().hex}",
        adapter_manifest=adapter.as_manifest(),
        invocation=invocation,
        benchmark_suite=suite,
        archive_path=archive,
        source_run_path="analysis/tail_kinematics_runs/tail_source",
        candidate_run_path="analysis/tail_kinematics_runs/tail_candidate",
        scratch_root=scratch,
        source_identity_sha256=canonical_json_sha256(source_hashes),
        palette_commit=commit,
        repetition_index=0,
        candidate_order_index=0,
        candidate_order_count=1,
        cache_state="fresh",
        physical_io_scope=PhysicalIOScope.UNAVAILABLE,
        selector_before_sha256=tail_kinematics_selector_snapshot_sha256(archive),
        registry_probe_path=probe,
        production_profiles_probe_path=probe,
    )
    request_path = benchmark / "request.json"
    request_path.write_text(
        json.dumps(request, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    receipt_path = benchmark / "receipt.json"
    attempt_path = benchmark / "attempt.json"

    receipt = run_tail_kinematics_candidate_fresh_process(
        request_path,
        receipt_path=receipt_path,
        attempt_path=attempt_path,
    )

    require_candidate_execution_receipt(
        receipt,
        expected_request_payload_digest=request["payload_digest"],
    )
    assert receipt_path.is_file()
    assert not attempt_path.exists()
    assert receipt["payload"]["logical_equality"]["equal"] is True
    assert receipt["payload"]["logical_equality"]["compared_array_count"] in {
        21,
        23,
    }
    assert [phase["phase"] for phase in receipt["payload"]["phases"]] == [
        "plan",
        "source_staging",
        "scientific_compute",
        "local_validation",
        "local_consolidation",
        "local_direct_consolidated_comparison",
        "atomic_publication",
        "published_validation",
        "published_direct_consolidated_comparison",
        "decoded_equality",
        "physical_inventory",
        "publication_acceptance_validation",
    ]
    final = zarr.open_group(str(archive), mode="r", use_consolidated=False)
    parent = final["analysis/tail_kinematics_runs"]
    assert parent.attrs["latest"] == "tail_source"
    assert parent.attrs["latest_complete"] == "tail_source"
    candidate = parent["tail_candidate"]
    assert candidate.attrs["palette_run_completion_status"] == "complete"
    assert candidate.attrs["stage_selector_eligible"] is False
    assert candidate.attrs["storage_candidate_profile_promoted"] is False

    failed_request = build_candidate_execution_request(
        execution_id=f"tail_{uuid.uuid4().hex}",
        adapter_manifest=adapter.as_manifest(),
        invocation=invocation,
        benchmark_suite=suite,
        archive_path=archive,
        source_run_path="analysis/tail_kinematics_runs/tail_source",
        candidate_run_path="analysis/tail_kinematics_runs/tail_failed",
        scratch_root=Path("/tmp") / f"palette-tail-runner-{uuid.uuid4().hex}",
        source_identity_sha256="0" * 64,
        palette_commit=commit,
        repetition_index=0,
        candidate_order_index=0,
        candidate_order_count=1,
        cache_state="fresh",
        physical_io_scope=PhysicalIOScope.UNAVAILABLE,
        selector_before_sha256=tail_kinematics_selector_snapshot_sha256(archive),
        registry_probe_path=probe,
        production_profiles_probe_path=probe,
    )
    failed_request_path = benchmark / "failed-request.json"
    failed_request_path.write_text(
        json.dumps(failed_request, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    failed_receipt_path = benchmark / "failed-receipt.json"
    failed_attempt_path = benchmark / "failed-attempt.json"
    with pytest.raises(
        TailKinematicsCandidateExecutionFailed,
        match="immutable attempt record",
    ):
        run_tail_kinematics_candidate_fresh_process(
            failed_request_path,
            receipt_path=failed_receipt_path,
            attempt_path=failed_attempt_path,
        )
    assert not failed_receipt_path.exists()
    assert failed_attempt_path.is_file()
    attempt = json.loads(failed_attempt_path.read_text(encoding="utf-8"))
    assert attempt["payload"]["status"] == "failed"
    assert attempt["payload"]["failure_phase"] == "runner_preflight"
    assert attempt["payload"]["nonmutation_evidence"]["unchanged"] is True
