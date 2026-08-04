from __future__ import annotations

from copy import deepcopy
import multiprocessing
import os
from pathlib import Path
import subprocess
from typing import Any
import uuid

import pytest
import zarr

from fisheye.analysis.track_kinematics import (
    TRACK_MOTION_PUBLICATION_MANIFEST_DIGEST_ATTR,
)
from fisheye.analysis.track_kinematics_storage import (
    build_flat_candidate_declarations,
    source_flat_projection_hashes,
)
from fisheye.analysis_workflows import analysis_candidate_execution_catalog as catalog
from fisheye.analysis_workflows.analysis_candidate_execution import (
    CandidateComputationMode,
    CandidateLogicalEqualityContract,
    CandidateRunnerStatus,
    CoordinateContractRole,
    CoordinateContractStatus,
    PhysicalIOScope,
    build_candidate_execution_request,
    require_candidate_execution_receipt,
)
from fisheye.analysis_workflows.analysis_candidate_execution_catalog import (
    AnalysisCandidateExecutionAdapter,
)
from fisheye.analysis_workflows.analysis_candidate_invocation import (
    CandidateInvocationContract,
    build_track_flat_invocation,
)
from fisheye.analysis_workflows.materializers.track_kinematics_candidate import (
    EXECUTION_BINDING_ATTR,
    EXECUTION_FAILURE_TOMBSTONE_ATTR,
    TRACK_FLAT_EXECUTION_PHASE_ORDER,
    materialize_track_kinematics_flat_candidate,
    tombstone_track_kinematics_execution_candidate,
)
from fisheye.analysis_workflows.track_kinematics_candidate_suite import (
    build_track_flat_execution_suite,
    require_track_flat_execution_suite,
)
from fisheye.diagnostics.track_kinematics_candidate_execution import (
    TrackFlatCandidateExecutionFailed,
    execute_track_flat_candidate,
    require_track_flat_execution_attempt,
    track_flat_selector_snapshot_sha256,
)
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from fisheye.shared.zarr_run_completion import RUN_COMPLETION_STATUS_ATTR
from tests.unit.fisheye.test_benchmark_track_kinematics_v2_candidate import (
    SOURCE_RUN_PATH,
    _build_canonical_sealed_source,
)
from tests.unit.fisheye.test_track_kinematics_flat_candidate import (
    _build_source_archive,
    _populate_v1_run,
)


def _implemented_adapter() -> AnalysisCandidateExecutionAdapter:
    return AnalysisCandidateExecutionAdapter(
        stage_id="track_kinematics",
        invocation_contract=CandidateInvocationContract.TRACK_FLAT_V1,
        computation_mode=CandidateComputationMode.LOGICAL_REMATERIALIZATION,
        runner_status=CandidateRunnerStatus.IMPLEMENTED,
        coordinate_role=CoordinateContractRole.CANONICAL_PRODUCER,
        coordinate_contract_status=CoordinateContractStatus.SOURCE_PRESERVATION_ONLY,
        logical_equality_contract=(
            CandidateLogicalEqualityContract.TRACK_FLAT_PROJECTION_V1
        ),
        runner_module=("fisheye.diagnostics.track_kinematics_candidate_execution"),
        runner_entrypoint="execute_track_flat_candidate",
        suite_validator_module=(
            "fisheye.analysis_workflows.track_kinematics_candidate_suite"
        ),
        suite_validator_entrypoint="require_track_flat_execution_suite",
    )


def _patch_implemented_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, object]:
    adapter = _implemented_adapter()
    monkeypatch.setitem(
        catalog.ANALYSIS_CANDIDATE_EXECUTION_ADAPTER_BY_STAGE,
        "track_kinematics",
        adapter,
    )
    return adapter.as_manifest()


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _request(
    archive: Path,
    *,
    adapter: dict[str, object],
    scratch: Path,
    probe: Path,
    motion_digest: str | None = None,
) -> dict[str, object]:
    root = zarr.open_group(
        str(archive), mode="r", zarr_format=3, use_consolidated=False
    )
    source = root[SOURCE_RUN_PATH]
    declarations = build_flat_candidate_declarations(source)
    hashes = source_flat_projection_hashes(source, declarations)
    authority = str(source.attrs[TRACK_MOTION_PUBLICATION_MANIFEST_DIGEST_ATTR])
    suite = build_track_flat_execution_suite(
        source,
        storage_profile_id="published_http_v1",
        repetitions=1,
    )
    return build_candidate_execution_request(
        execution_id=f"track_flat_{uuid.uuid4().hex}",
        adapter_manifest=adapter,
        invocation=build_track_flat_invocation(
            source_motion_authority_sha256=motion_digest or authority,
            storage_profile_id="published_http_v1",
            copy_backend="python",
            keep_scratch=False,
        ),
        benchmark_suite=suite,
        archive_path=archive,
        source_run_path=SOURCE_RUN_PATH,
        candidate_run_path=(
            "analysis/track_kinematics_runs/offline/typed_candidate_v2"
        ),
        scratch_root=scratch,
        source_identity_sha256=canonical_json_sha256(hashes),
        palette_commit=_git_commit(),
        repetition_index=0,
        candidate_order_index=0,
        candidate_order_count=1,
        cache_state="fresh_process_os_cache_uncontrolled",
        physical_io_scope=PhysicalIOScope.UNAVAILABLE,
        selector_before_sha256=track_flat_selector_snapshot_sha256(archive),
        registry_probe_path=probe,
        production_profiles_probe_path=probe,
    )


def _execute_in_child(
    queue: Any,
    request: dict[str, object],
    driver_pid: int,
) -> None:
    catalog.ANALYSIS_CANDIDATE_EXECUTION_ADAPTER_BY_STAGE["track_kinematics"] = (
        _implemented_adapter()
    )
    try:
        queue.put(
            ("receipt", execute_track_flat_candidate(request, driver_pid=driver_pid))
        )
    except TrackFlatCandidateExecutionFailed as exc:
        queue.put(("attempt", exc.attempt))


def test_track_suite_is_exact_and_explicitly_excludes_physical_bundle(
    tmp_path: Path,
) -> None:
    source = zarr.open_group(
        str(tmp_path / "source"),
        mode="w",
        zarr_format=3,
        use_consolidated=False,
    )
    _populate_v1_run(source, track_rows=(3, 2), include_physical=False)
    suite = build_track_flat_execution_suite(
        source,
        storage_profile_id="published_http_v1",
        repetitions=2,
    )
    require_track_flat_execution_suite("track_kinematics", suite)
    assert suite["payload"]["repetitions"] == 2

    tampered = deepcopy(suite)
    record = tampered["payload"]["storage_plan_receipt"]["payload"]["arrays"][0]
    record["declaration"]["fill_semantics"] = "tampered"
    receipt_payload = tampered["payload"]["storage_plan_receipt"]["payload"]
    tampered["payload"]["storage_plan_receipt"]["payload_digest"] = (
        canonical_json_sha256(receipt_payload)
    )
    tampered["payload_digest"] = canonical_json_sha256(tampered["payload"])
    with pytest.raises(ValueError, match="storage plan|declaration|digest"):
        require_track_flat_execution_suite("track_kinematics", tampered)

    physical = zarr.open_group(
        str(tmp_path / "physical"),
        mode="w",
        zarr_format=3,
        use_consolidated=False,
    )
    _populate_v1_run(physical, track_rows=(3,), include_physical=True)
    with pytest.raises(ValueError, match="excludes the physical"):
        build_track_flat_execution_suite(
            physical,
            storage_profile_id="published_http_v1",
        )


def test_materializer_stages_binds_accepts_and_owner_tombstones(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "recording.zarr"
    scratch = tmp_path / "scratch"
    _build_source_archive(archive)
    binding = {
        "schema_id": "palette.analysis_candidate_execution_binding",
        "schema_version": 1,
        "execution_id": "track_test",
        "request_payload_digest": "a" * 64,
        "candidate_run_path": ("analysis/track_kinematics_runs/offline/candidate_v2"),
    }
    selector_before = track_flat_selector_snapshot_sha256(archive)

    def accept(root, _offline, candidate):
        assert candidate.attrs[EXECUTION_BINDING_ATTR] == binding
        assert track_flat_selector_snapshot_sha256(archive) == selector_before
        assert root["analysis/track_kinematics_runs/offline/source_v1"] is not None
        return {"execution_binding": binding, "accepted": True}

    result = materialize_track_kinematics_flat_candidate(
        archive,
        source_run="source_v1",
        run_name="candidate_v2",
        scratch_root=scratch,
        copy_backend="python",
        apply=True,
        stage_source_to_scratch=True,
        exclude_physical_bundle=True,
        execution_binding=binding,
        publication_acceptance_validator=accept,
    )
    assert result["caller_acceptance"] == {
        "execution_binding": binding,
        "accepted": True,
    }
    assert [phase["name"] for phase in result["runtime_telemetry"]["phases"]] == list(
        TRACK_FLAT_EXECUTION_PHASE_ORDER
    )
    assert not scratch.exists()
    assert track_flat_selector_snapshot_sha256(archive) == selector_before

    tombstone_track_kinematics_execution_candidate(
        archive,
        run_name="candidate_v2",
        expected_execution_binding=binding,
        failure_phase="driver_receipt_publication",
        error_type="OSError",
        error_message="injected receipt failure",
    )
    direct = zarr.open_group(
        str(archive), mode="r", zarr_format=3, use_consolidated=False
    )["analysis/track_kinematics_runs/offline/candidate_v2"]
    consolidated = zarr.open_group(
        str(archive), mode="r", zarr_format=3, use_consolidated=True
    )["analysis/track_kinematics_runs/offline/candidate_v2"]
    assert direct.attrs[RUN_COMPLETION_STATUS_ATTR] == "failed"
    assert (
        consolidated.attrs[EXECUTION_FAILURE_TOMBSTONE_ATTR]["execution_binding"]
        == binding
    )
    with pytest.raises(RuntimeError, match="another execution"):
        tombstone_track_kinematics_execution_candidate(
            archive,
            run_name="candidate_v2",
            expected_execution_binding={**binding, "execution_id": "foreign"},
            failure_phase="driver_receipt_publication",
            error_type="OSError",
            error_message="foreign",
        )


def test_acceptance_failure_stays_inside_atomic_tombstone_boundary(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "recording.zarr"
    scratch = tmp_path / "scratch"
    _build_source_archive(archive)
    selector_before = track_flat_selector_snapshot_sha256(archive)
    binding = {
        "schema_id": "palette.analysis_candidate_execution_binding",
        "schema_version": 1,
        "execution_id": "track_acceptance_failure",
        "request_payload_digest": "b" * 64,
        "candidate_run_path": ("analysis/track_kinematics_runs/offline/candidate_v2"),
    }

    def reject(_root, _offline, _candidate):
        raise ValueError("injected track acceptance failure")

    with pytest.raises(ValueError, match="injected track acceptance failure"):
        materialize_track_kinematics_flat_candidate(
            archive,
            source_run="source_v1",
            run_name="candidate_v2",
            scratch_root=scratch,
            copy_backend="python",
            apply=True,
            stage_source_to_scratch=True,
            exclude_physical_bundle=True,
            execution_binding=binding,
            publication_acceptance_validator=reject,
        )

    path = "analysis/track_kinematics_runs/offline/candidate_v2"
    direct = zarr.open_group(
        str(archive), mode="r", zarr_format=3, use_consolidated=False
    )[path]
    consolidated = zarr.open_group(
        str(archive), mode="r", zarr_format=3, use_consolidated=True
    )[path]
    assert direct.attrs[RUN_COMPLETION_STATUS_ATTR] == "failed"
    assert consolidated.attrs[RUN_COMPLETION_STATUS_ATTR] == "failed"
    assert direct.attrs[EXECUTION_BINDING_ATTR] == binding
    assert direct.attrs["stage_selector_eligible"] is False
    assert direct.attrs["storage_candidate_profile_promoted"] is False
    assert track_flat_selector_snapshot_sha256(archive) == selector_before


@pytest.mark.parametrize("valid_authority", [True, False])
def test_real_canonical_source_executes_or_fails_closed_in_fresh_child(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    valid_authority: bool,
) -> None:
    benchmark_root = tmp_path / ".palette_benchmarks" / "track-execution"
    benchmark_root.mkdir(parents=True)
    archive = _build_canonical_sealed_source(benchmark_root)
    probe = benchmark_root / "protected-probe.json"
    probe.write_text("{}\n", encoding="utf-8")
    adapter = _patch_implemented_adapter(monkeypatch)
    scratch = Path("/tmp") / f"palette-track-test-{uuid.uuid4().hex}"
    request = _request(
        archive,
        adapter=adapter,
        scratch=scratch,
        probe=probe,
        motion_digest=None if valid_authority else "f" * 64,
    )
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    process = context.Process(
        target=_execute_in_child,
        args=(queue, request, os.getpid()),
    )
    process.start()
    kind, evidence = queue.get(timeout=180)
    process.join(timeout=15)
    assert process.exitcode == 0
    queue.close()
    queue.join_thread()

    if valid_authority:
        assert kind == "receipt"
        require_candidate_execution_receipt(
            evidence,
            expected_request_payload_digest=request["payload_digest"],
        )
        assert evidence["payload"]["publication_gate_passed"] is False
        assert evidence["payload"]["coordinate_evidence"]["status"] == (
            "verified_source_preservation_nonminting"
        )
        assert evidence["payload"]["fresh_process"]["child_pid"] == process.pid
        run = zarr.open_group(
            str(archive), mode="r", zarr_format=3, use_consolidated=False
        )["analysis/track_kinematics_runs/offline/typed_candidate_v2"]
        assert run.attrs["stage_selector_eligible"] is False
        assert run.attrs["storage_candidate_profile_promoted"] is False
    else:
        assert kind == "attempt"
        require_track_flat_execution_attempt(
            evidence,
            expected_request_payload_digest=request["payload_digest"],
        )
        assert evidence["payload"]["failure_phase"] == "runner_preflight"
        assert evidence["payload"]["target_state"]["exists"] is False
    assert not scratch.exists()
