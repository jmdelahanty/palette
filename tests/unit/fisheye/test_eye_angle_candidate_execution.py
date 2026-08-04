from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from fisheye.analysis.eye_angle_schema import EyeAngleDimensions
from fisheye.analysis.eye_angle_storage import build_eye_angle_candidate_storage_plan
from fisheye.analysis_workflows.analysis_candidate_execution import (
    CandidateComputationMode,
    CandidateLogicalEqualityContract,
    CandidateRunnerStatus,
    CoordinateContractRole,
    CoordinateContractStatus,
    PhysicalIOScope,
    build_candidate_execution_request,
    require_candidate_execution_request,
)
from fisheye.analysis_workflows.analysis_candidate_execution_catalog import (
    ANALYSIS_CANDIDATE_EXECUTION_ADAPTER_BY_STAGE,
    AnalysisCandidateExecutionAdapter,
)
from fisheye.analysis_workflows.analysis_candidate_invocation import (
    CandidateInvocationContract,
    build_eye_angle_invocation,
)
from fisheye.analysis_workflows.eye_angle_candidate_execution import (
    EYE_ANGLE_EXECUTION_FAMILY_ID,
    build_eye_angle_bound_source_evidence,
    eye_angle_dimensions_from_suite,
    require_eye_angle_execution_suite,
)
from fisheye.shared.zarr.analysis_benchmark_suite import (
    AnalysisBenchmarkScale,
    build_analysis_benchmark_suite,
)
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256


def _suite() -> dict[str, object]:
    receipt = build_eye_angle_candidate_storage_plan(EyeAngleDimensions(23, 31, 16))
    return build_analysis_benchmark_suite(
        family_id=EYE_ANGLE_EXECUTION_FAMILY_ID,
        scale=AnalysisBenchmarkScale(
            scale_id="unit",
            dimensions=receipt.dimensions,
            description="small exact eye-angle execution fixture",
        ),
        storage_receipt=receipt,
        repetitions=1,
    )


def _authority(label: str) -> dict[str, object]:
    body = {
        "schema_id": f"palette.{label}",
        "schema_version": 1,
        "identity": label,
    }
    return {**body, "record_sha256": canonical_json_sha256(body)}


def _source_contracts() -> dict[str, object]:
    return {
        "eye_geometry": {
            "source_authority": _authority("subject_shape_authority"),
        },
        "keypoints": {
            "canonical_keypoint_authority": _authority("keypoint_authority"),
        },
        "diagnostic_base_keypoints": {},
        "resolved_arrays": {},
    }


def test_suite_validator_reconstructs_exact_41_array_plan() -> None:
    suite = _suite()

    require_eye_angle_execution_suite(EYE_ANGLE_EXECUTION_FAMILY_ID, suite)
    dimensions = eye_angle_dimensions_from_suite(suite)

    assert dimensions == EyeAngleDimensions(23, 31, 16)
    assert len(suite["payload"]["storage_plan_receipt"]["payload"]["arrays"]) == 41


def test_suite_validator_rejects_another_family() -> None:
    with pytest.raises(ValueError, match="family differs"):
        require_eye_angle_execution_suite("track_kinematics", _suite())


def test_bound_source_evidence_contains_only_live_source_authorities() -> None:
    evidence = build_eye_angle_bound_source_evidence(_source_contracts())

    assert evidence["status"] == "verified_bound_source"
    assert evidence["coordinate_gate_passed"] is True
    assert [item["role"] for item in evidence["source_authority_digests"]] == [
        "canonical_keypoints",
        "subject_shape_eye_geometry",
    ]
    assert evidence["published_authority_sha256"] is None
    assert evidence["published_authority_ref"] is None
    assert evidence["temporal_axis_sha256"] is None
    assert evidence["temporal_axis_ref"] is None


def test_bound_source_evidence_rejects_coordinated_authority_tampering() -> None:
    contracts = _source_contracts()
    tampered = deepcopy(contracts)
    tampered["keypoints"]["canonical_keypoint_authority"]["identity"] = "other"

    with pytest.raises(ValueError, match="self-digest differs"):
        build_eye_angle_bound_source_evidence(tampered)


def test_request_v2_binds_exact_eye_invocation_and_resolves_family_runner(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    adapter = AnalysisCandidateExecutionAdapter(
        stage_id="eye_angles",
        invocation_contract=CandidateInvocationContract.EYE_ANGLES_V1,
        computation_mode=CandidateComputationMode.SCIENTIFIC_COMPUTE,
        runner_status=CandidateRunnerStatus.IMPLEMENTED,
        coordinate_role=CoordinateContractRole.BOUND_DERIVATIVE,
        coordinate_contract_status=(
            CoordinateContractStatus.BOUND_SOURCE_VALIDATION_IMPLEMENTED
        ),
        logical_equality_contract=(
            CandidateLogicalEqualityContract.EYE_ANGLE_COMPACT_V7_ARRAYS_V1
        ),
        runner_module="fisheye.diagnostics.eye_angle_candidate_execution",
        runner_entrypoint="execute_eye_angle_candidate",
        suite_validator_module=(
            "fisheye.analysis_workflows.eye_angle_candidate_execution"
        ),
        suite_validator_entrypoint="require_eye_angle_execution_suite",
    )
    monkeypatch.setitem(
        ANALYSIS_CANDIDATE_EXECUTION_ADAPTER_BY_STAGE,
        "eye_angles",
        adapter,
    )
    benchmark_root = tmp_path / ".palette_benchmarks" / "eye"
    benchmark_root.mkdir(parents=True)
    registry = tmp_path / "registry.sqlite"
    profiles = tmp_path / "profiles.json"
    registry.write_bytes(b"registry")
    profiles.write_bytes(b"profiles")
    invocation = build_eye_angle_invocation(
        subject_shape_run="shape_1",
        keypoint_run="kp_raw_1",
        storage_profile_id="eye_angle_access_aware_candidate_v1",
        chunk_rows=8_192,
        angle_chunk_rows=2_048,
        angle_chunk_columns=16,
        output_shard_rows=131_072,
        angle_shard_columns=32,
        execution_backend="serial_driver",
        scheduler="single-threaded",
        num_workers=1,
        shard_workers=1,
        native_threads=1,
        fps=100.0,
        smoothing_window=5,
        copy_backend="python",
        keep_scratch=False,
        check_capacity=True,
    )
    request = build_candidate_execution_request(
        execution_id="eye-unit",
        adapter_manifest=adapter.as_manifest(),
        invocation=invocation,
        benchmark_suite=_suite(),
        archive_path=benchmark_root / "fixture.zarr",
        source_run_path="analysis/eye_angle_runs/source",
        candidate_run_path="analysis/eye_angle_runs/candidate",
        scratch_root=tmp_path / "scratch",
        source_identity_sha256="a" * 64,
        palette_commit="b" * 40,
        repetition_index=0,
        candidate_order_index=0,
        candidate_order_count=1,
        cache_state="fresh",
        physical_io_scope=PhysicalIOScope.UNAVAILABLE,
        selector_before_sha256="c" * 64,
        registry_probe_path=registry,
        production_profiles_probe_path=profiles,
    )

    require_candidate_execution_request(request)
    assert request["payload"]["invocation"] == invocation
    assert adapter.resolves_runner() is True
    assert adapter.resolves_suite_validator() is True
