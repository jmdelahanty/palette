from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from fisheye.analysis.swim_bout_schema import (
    _required_specs as _swim_bout_required_specs,
    build_swim_bout_array_declarations,
)
from fisheye.analysis_workflows.analysis_candidate_execution import (
    CandidateComputationMode,
    CandidateExecutionPhase,
    CandidatePhaseMeasurement,
    CoordinateEvidenceStatus,
    PhaseOutcome,
    PhysicalIOScope,
    build_candidate_execution_receipt,
    build_candidate_execution_request,
    protected_path_snapshot_sha256,
    require_candidate_execution_adapter_manifest,
    require_candidate_execution_receipt,
    require_candidate_execution_request,
    required_execution_phases,
)
from fisheye.analysis_workflows.analysis_candidate_execution_catalog import (
    ANALYSIS_CANDIDATE_EXECUTION_ADAPTER_BY_STAGE,
    ANALYSIS_CANDIDATE_EXECUTION_ADAPTERS,
)
from fisheye.analysis_workflows.analysis_candidate_invocation import (
    build_exact_tabular_invocation,
)
from fisheye.analysis_workflows.storage_candidate_catalog import (
    DERIVED_ANALYSIS_STORAGE_CANDIDATE_BY_STAGE,
)
from fisheye.shared.zarr.analysis_array_contracts import (
    AnalysisArrayDeclaration,
    AnalysisAuthorityRole,
)
from fisheye.shared.zarr.analysis_benchmark_suite import (
    AnalysisBenchmarkScale,
    build_analysis_benchmark_suite,
)
from fisheye.shared.zarr.analysis_storage_planning import (
    AnalysisArrayStorageFacts,
    plan_analysis_storage,
)
from fisheye.shared.zarr.array_contracts import FLOAT32, ArrayContract
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from fisheye.shared.zarr.storage_intent import AccessPattern, WriteMode
from fisheye.shared.zarr.storage_profiles import PUBLISHED_HTTP_V1, SCRATCH_COMPUTE_V1


class _Array:
    def __init__(self, *, shape, dtype):
        self.shape = tuple(shape)
        self.dtype = np.dtype(dtype)


class _Group:
    def __init__(self):
        self._arrays = {}
        self._groups = {}

    def add(self, path, array):
        current = self
        parts = path.split("/")
        for part in parts[:-1]:
            current = current._groups.setdefault(part, _Group())
        current._arrays[parts[-1]] = array

    def arrays(self):
        return tuple(self._arrays.items())

    def groups(self):
        return tuple(self._groups.items())


def test_protected_tree_snapshot_has_unambiguous_entry_framing(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    (first / "a").write_bytes(b"x\nb\0y")
    (second / "a").write_bytes(b"x")
    (second / "b").write_bytes(b"y")

    assert protected_path_snapshot_sha256(first) != protected_path_snapshot_sha256(
        second
    )

    before_empty_directory = protected_path_snapshot_sha256(second)
    (second / "empty").mkdir()
    assert protected_path_snapshot_sha256(second) != before_empty_directory


def _suite(*, profile=PUBLISHED_HTTP_V1):
    group = _Group()
    for path, spec in _swim_bout_required_specs().items():
        shape = tuple(
            (
                16
                if axis == "utf8_byte"
                else 2_048 if axis == "frame" else 2 if axis == "detector_signal" else 3
            )
            for axis in spec.axes
        )
        group.add(path, _Array(shape=shape, dtype=spec.dtype))
    declarations = build_swim_bout_array_declarations(
        group,
        byte_planner_adopted=True,
    )
    facts = {
        declaration.path: AnalysisArrayStorageFacts(
            path=declaration.path,
            shape=group_path(group, declaration.path).shape,
            dtype=group_path(group, declaration.path).dtype,
            access_unit_semantics="one complete compact logical record",
        )
        for declaration in declarations
    }
    receipt = plan_analysis_storage(
        declarations,
        facts,
        profile=profile,
    )
    return build_analysis_benchmark_suite(
        family_id="swim_bouts",
        scale=AnalysisBenchmarkScale(
            scale_id="rows_2048",
            dimensions=receipt.dimensions,
            description="Synthetic request scale.",
        ),
        storage_receipt=receipt,
        repetitions=5,
    )


def group_path(group, path):
    current = group
    parts = path.split("/")
    for part in parts[:-1]:
        current = current._groups[part]
    return current._arrays[parts[-1]]


def _unrelated_suite():
    declaration = AnalysisArrayDeclaration(
        path="values",
        contract=ArrayContract(
            schema_id="palette.test.execution_values",
            schema_version=1,
            dtype=FLOAT32,
            shape_template=("n_rows",),
            axis_names=("row",),
            description="Unrelated execution-contract values.",
        ),
        required=True,
        access_pattern=AccessPattern.WINDOWED,
        write_mode=WriteMode.IMMUTABLE,
        authority_role=AnalysisAuthorityRole.SCIENTIFIC_AUTHORITY,
        fill_semantics="every row is present",
        null_semantics="none",
        physical_policy_owner="test_execution",
        byte_planner_adopted=True,
    )
    receipt = plan_analysis_storage(
        (declaration,),
        {
            "values": AnalysisArrayStorageFacts(
                path="values",
                shape=(2_048,),
                dtype=np.dtype("float32"),
                access_unit_semantics="one logical row",
            )
        },
        profile=PUBLISHED_HTTP_V1,
        dimensions={"n_rows": 2_048},
    )
    return build_analysis_benchmark_suite(
        family_id="swim_bouts",
        scale=AnalysisBenchmarkScale(
            scale_id="unrelated_rows_2048",
            dimensions=receipt.dimensions,
            description="Deliberately unrelated suite.",
        ),
        storage_receipt=receipt,
        repetitions=5,
    )


@pytest.fixture
def implemented_adapter():
    return ANALYSIS_CANDIDATE_EXECUTION_ADAPTER_BY_STAGE["swim_bouts"].as_manifest()


def _exact_invocation():
    return build_exact_tabular_invocation(
        storage_profile_id="published_http_v1",
        copy_backend="python",
        keep_scratch=False,
    )


def _request(
    adapter_manifest,
    *,
    physical_io_scope=PhysicalIOScope.FILESYSTEM_OR_NETWORK_TRANSFER,
):
    return build_candidate_execution_request(
        execution_id="swim_bouts_rows_2048_rep0",
        adapter_manifest=adapter_manifest,
        invocation=_exact_invocation(),
        benchmark_suite=_suite(),
        archive_path="/tmp/.palette_benchmarks/execution/archive.zarr",
        source_run_path="analysis/swim_bout_runs/source_v1",
        candidate_run_path="analysis/swim_bout_runs/candidate_v1",
        scratch_root="/tmp/palette-candidate-execution-scratch",
        source_identity_sha256="b" * 64,
        palette_commit="a" * 40,
        repetition_index=0,
        candidate_order_index=0,
        candidate_order_count=1,
        cache_state="fresh_process_os_cache_uncontrolled",
        physical_io_scope=physical_io_scope,
        selector_before_sha256="f" * 64,
        registry_probe_path=Path(__file__).resolve(),
        production_profiles_probe_path=Path(__file__).resolve(),
    )


def _phases() -> list[CandidatePhaseMeasurement]:
    return [
        CandidatePhaseMeasurement(
            phase=phase,
            outcome=PhaseOutcome.SUCCEEDED,
            started_at_utc="2026-08-03T12:00:00+00:00",
            completed_at_utc="2026-08-03T12:00:01+00:00",
            wall_seconds=1.0,
            cpu_user_seconds=0.4,
            cpu_system_seconds=0.1,
            peak_process_tree_rss_bytes=1_024,
        )
        for phase in required_execution_phases(
            CandidateComputationMode.LOGICAL_REMATERIALIZATION
        )
    ]


def _receipt(
    adapter_manifest,
    *,
    physical_io_scope=PhysicalIOScope.FILESYSTEM_OR_NETWORK_TRANSFER,
):
    request = _request(
        adapter_manifest,
        physical_io_scope=physical_io_scope,
    )
    measured = physical_io_scope is PhysicalIOScope.FILESYSTEM_OR_NETWORK_TRANSFER
    unavailable = physical_io_scope is PhysicalIOScope.UNAVAILABLE
    array_count = len(
        request["payload"]["benchmark_suite"]["payload"]["storage_plan_receipt"][
            "payload"
        ]["arrays"]
    )
    protected = request["payload"]["protected_state_before"]
    return build_candidate_execution_receipt(
        request=request,
        status="complete",
        fresh_process={
            "driver_pid": 100,
            "child_pid": 101,
            "child_start_time_ticks": 123_456,
            "is_fresh": True,
        },
        environment={
            "hostname": "benchmark-host",
            "platform": "Linux",
            "python_version": "3.11.9",
            "python_executable": "/opt/palette/bin/python",
            "palette_commit": "a" * 40,
            "palette_git_dirty": False,
            "runner_ref": (
                "fisheye.diagnostics.analysis_candidate_execution:"
                "execute_exact_tabular_candidate"
            ),
            "runner_sha256": "c" * 64,
            "cache_state": "fresh_process_os_cache_uncontrolled",
        },
        phases=_phases(),
        coordinate_evidence={
            "role": "bound_derivative",
            "status": CoordinateEvidenceStatus.VERIFIED_BOUND_SOURCE.value,
            "source_authority_digests": [{"role": "track_motion", "sha256": "d" * 64}],
            "published_authority_sha256": None,
            "published_authority_ref": None,
            "temporal_axis_sha256": None,
            "temporal_axis_ref": None,
            "validator_ref": "fisheye.analysis.swim_bout_schema:validate",
            "validation_receipt_sha256": "3" * 64,
            "coordinate_gate_passed": True,
        },
        logical_equality={
            "contract_id": "swim_bouts_declared_arrays_v1",
            "compared_array_count": array_count,
            "source_logical_manifest_sha256": "b" * 64,
            "candidate_logical_manifest_sha256": "b" * 64,
            "equal": True,
        },
        metadata_equivalence={
            "local_array_count": array_count,
            "published_array_count": array_count,
            "local_equal": True,
            "published_equal": True,
        },
        physical_io={
            "scope": physical_io_scope.value,
            "physical_io_measured": measured,
            "read_bytes": None if unavailable else 100,
            "write_bytes": None if unavailable else 200,
            "read_operations": None if unavailable else 3,
            "write_operations": None if unavailable else 4,
            "measurement_ref": None if unavailable else "sidecar/io.json",
            "measurement_sha256": None if unavailable else "4" * 64,
        },
        output_storage={
            "metadata_file_count": array_count + 1,
            "payload_file_count": 5,
            "file_count": array_count + 6,
            "apparent_bytes": 1_000,
            "allocated_bytes": 4_096,
        },
        nonmutation_evidence={
            "selector_before_sha256": "f" * 64,
            "selector_after_sha256": "f" * 64,
            "registry_before_sha256": protected["registry_sha256"],
            "registry_after_sha256": protected["registry_sha256"],
            "production_profiles_before_sha256": protected[
                "production_profiles_sha256"
            ],
            "production_profiles_after_sha256": protected["production_profiles_sha256"],
            "snapshot_contract_id": "analysis_candidate_nonmutation_v1",
            "unchanged": True,
        },
    )


def test_execution_adapter_catalog_is_exact_and_fully_typed() -> None:
    assert len(ANALYSIS_CANDIDATE_EXECUTION_ADAPTERS) == 13
    assert set(ANALYSIS_CANDIDATE_EXECUTION_ADAPTER_BY_STAGE) == set(
        DERIVED_ANALYSIS_STORAGE_CANDIDATE_BY_STAGE
    )
    manifests = [
        adapter.as_manifest() for adapter in ANALYSIS_CANDIDATE_EXECUTION_ADAPTERS
    ]
    for manifest in manifests:
        require_candidate_execution_adapter_manifest(manifest)

    status = {
        manifest["payload"]["stage_id"]: manifest["payload"]["runner_status"]
        for manifest in manifests
    }
    assert set(status.values()) == {"implemented"}

    for stage in status:
        adapter = ANALYSIS_CANDIDATE_EXECUTION_ADAPTER_BY_STAGE[stage]
        assert adapter.resolves_candidate_owner() is True
        assert adapter.resolves_runner() is True
        assert adapter.resolves_suite_validator() is True


def test_execution_request_and_complete_receipt_are_strict_json(
    implemented_adapter,
) -> None:
    request = _request(implemented_adapter)
    receipt = _receipt(implemented_adapter)

    require_candidate_execution_request(request)
    require_candidate_execution_receipt(
        receipt,
        expected_request_payload_digest=request["payload_digest"],
    )
    assert receipt["payload"]["publication_gate_passed"] is True

    adapter_numeric_alias = deepcopy(implemented_adapter)
    adapter_numeric_alias["schema_version"] = 1.0
    with pytest.raises(ValueError, match="schema identity"):
        require_candidate_execution_adapter_manifest(adapter_numeric_alias)

    receipt_numeric_alias = deepcopy(receipt)
    receipt_numeric_alias["schema_version"] = 2.0
    with pytest.raises(ValueError, match="schema identity"):
        require_candidate_execution_receipt(
            receipt_numeric_alias,
            expected_request_payload_digest=request["payload_digest"],
        )

    legacy_receipt = deepcopy(receipt)
    legacy_receipt["schema_version"] = 1
    with pytest.raises(ValueError, match="schema identity"):
        require_candidate_execution_receipt(
            legacy_receipt,
            expected_request_payload_digest=request["payload_digest"],
        )


def test_execution_request_requires_exact_bound_invocation(
    implemented_adapter,
) -> None:
    request = _request(implemented_adapter)

    missing = deepcopy(request)
    missing["payload"].pop("invocation")
    missing["payload_digest"] = canonical_json_sha256(missing["payload"])
    with pytest.raises(ValueError, match="payload field set"):
        require_candidate_execution_request(missing)

    legacy_v1 = deepcopy(request)
    legacy_v1["schema_version"] = 1
    with pytest.raises(ValueError, match="schema identity"):
        require_candidate_execution_request(legacy_v1)

    numeric_alias = deepcopy(request)
    numeric_alias["schema_version"] = 2.0
    with pytest.raises(ValueError, match="schema identity"):
        require_candidate_execution_request(numeric_alias)

    changed_profile = deepcopy(request)
    invocation = changed_profile["payload"]["invocation"]
    invocation["payload"]["parameters"]["storage_profile_id"] = "scratch_compute_v1"
    invocation["payload_digest"] = canonical_json_sha256(invocation["payload"])
    changed_profile["payload_digest"] = canonical_json_sha256(
        changed_profile["payload"]
    )
    with pytest.raises(ValueError, match="storage profile differs"):
        require_candidate_execution_request(changed_profile)


def test_execution_request_rejects_nonbenchmark_archive(implemented_adapter) -> None:
    with pytest.raises(ValueError, match=r"\.palette_benchmarks"):
        build_candidate_execution_request(
            execution_id="unsafe_request",
            adapter_manifest=implemented_adapter,
            invocation=_exact_invocation(),
            benchmark_suite=_suite(),
            archive_path="/groups/recording_analysis.zarr",
            source_run_path="analysis/swim_bout_runs/source_v1",
            candidate_run_path="analysis/swim_bout_runs/candidate_v1",
            scratch_root="/tmp/palette-candidate-execution-scratch",
            source_identity_sha256="b" * 64,
            palette_commit="a" * 40,
            repetition_index=0,
            candidate_order_index=0,
            candidate_order_count=1,
            cache_state="fresh_process_os_cache_uncontrolled",
            physical_io_scope=PhysicalIOScope.UNAVAILABLE,
            selector_before_sha256="f" * 64,
            registry_probe_path=Path(__file__).resolve(),
            production_profiles_probe_path=Path(__file__).resolve(),
        )


def test_rehashed_adapter_tampering_fails(implemented_adapter) -> None:
    request = _request(implemented_adapter)
    tampered = deepcopy(request)
    adapter = tampered["payload"]["adapter_manifest"]
    adapter["payload"]["coordinate_contract_status"] = "source_preservation_only"
    adapter["payload_digest"] = canonical_json_sha256(adapter["payload"])
    tampered["payload_digest"] = canonical_json_sha256(tampered["payload"])

    with pytest.raises(
        ValueError,
        match="coordinate role|registered catalog entry",
    ):
        require_candidate_execution_request(tampered)


def test_rehashed_logical_equality_tampering_fails(implemented_adapter) -> None:
    receipt = _receipt(implemented_adapter)
    tampered = deepcopy(receipt)
    tampered["payload"]["logical_equality"]["candidate_logical_manifest_sha256"] = (
        "9" * 64
    )
    tampered["payload_digest"] = canonical_json_sha256(tampered["payload"])

    with pytest.raises(ValueError, match="decoded equality"):
        require_candidate_execution_receipt(
            tampered,
            expected_request_payload_digest=receipt["payload"][
                "request_payload_digest"
            ],
        )


def test_proc_self_io_is_recorded_but_not_called_physical_transfer(
    implemented_adapter,
) -> None:
    receipt = _receipt(
        implemented_adapter,
        physical_io_scope=PhysicalIOScope.PROCESS_SELF_PROC_IO,
    )
    assert receipt["payload"]["physical_io"]["physical_io_measured"] is False
    assert receipt["payload"]["publication_gate_passed"] is False


def test_unavailable_physical_io_requires_null_counters(implemented_adapter) -> None:
    receipt = _receipt(
        implemented_adapter,
        physical_io_scope=PhysicalIOScope.UNAVAILABLE,
    )
    require_candidate_execution_receipt(
        receipt,
        expected_request_payload_digest=receipt["payload"]["request_payload_digest"],
    )
    assert receipt["payload"]["publication_gate_passed"] is False

    tampered = deepcopy(receipt)
    tampered["payload"]["physical_io"]["read_bytes"] = 0
    tampered["payload_digest"] = canonical_json_sha256(tampered["payload"])
    with pytest.raises(ValueError, match="null counters"):
        require_candidate_execution_receipt(
            tampered,
            expected_request_payload_digest=receipt["payload"][
                "request_payload_digest"
            ],
        )


def test_phase_contract_uses_exactly_one_compute_mode() -> None:
    scientific = required_execution_phases(CandidateComputationMode.SCIENTIFIC_COMPUTE)
    rematerialized = required_execution_phases(
        CandidateComputationMode.LOGICAL_REMATERIALIZATION
    )
    assert CandidateExecutionPhase.SCIENTIFIC_COMPUTE in scientific
    assert CandidateExecutionPhase.LOGICAL_REMATERIALIZATION not in scientific
    assert CandidateExecutionPhase.LOGICAL_REMATERIALIZATION in rematerialized
    assert CandidateExecutionPhase.SCIENTIFIC_COMPUTE not in rematerialized

    with pytest.raises(ValueError, match="guarded-direct"):
        required_execution_phases(CandidateComputationMode.GUARDED_DIRECT_WRITER)


def test_execution_request_rejects_nonlocal_scratch(implemented_adapter) -> None:
    with pytest.raises(ValueError, match="node-local scratch"):
        build_candidate_execution_request(
            execution_id="nonlocal_scratch",
            adapter_manifest=implemented_adapter,
            invocation=_exact_invocation(),
            benchmark_suite=_suite(),
            archive_path="/groups/.palette_benchmarks/execution/archive.zarr",
            source_run_path="analysis/swim_bout_runs/source_v1",
            candidate_run_path="analysis/swim_bout_runs/candidate_v1",
            scratch_root="/groups/scratch/not_node_local",
            source_identity_sha256="b" * 64,
            palette_commit="a" * 40,
            repetition_index=0,
            candidate_order_index=0,
            candidate_order_count=1,
            cache_state="fresh_process_os_cache_uncontrolled",
            physical_io_scope=PhysicalIOScope.UNAVAILABLE,
            selector_before_sha256="f" * 64,
            registry_probe_path=Path(__file__).resolve(),
            production_profiles_probe_path=Path(__file__).resolve(),
        )


def test_execution_request_rejects_storage_profile_mismatch(
    implemented_adapter,
) -> None:
    with pytest.raises(ValueError, match="storage profile"):
        build_candidate_execution_request(
            execution_id="wrong_profile",
            adapter_manifest=implemented_adapter,
            invocation=_exact_invocation(),
            benchmark_suite=_suite(profile=SCRATCH_COMPUTE_V1),
            archive_path="/tmp/.palette_benchmarks/execution/archive.zarr",
            source_run_path="analysis/swim_bout_runs/source_v1",
            candidate_run_path="analysis/swim_bout_runs/candidate_v1",
            scratch_root="/tmp/palette-candidate-execution-scratch",
            source_identity_sha256="b" * 64,
            palette_commit="a" * 40,
            repetition_index=0,
            candidate_order_index=0,
            candidate_order_count=1,
            cache_state="fresh_process_os_cache_uncontrolled",
            physical_io_scope=PhysicalIOScope.UNAVAILABLE,
            selector_before_sha256="f" * 64,
            registry_probe_path=Path(__file__).resolve(),
            production_profiles_probe_path=Path(__file__).resolve(),
        )


def test_execution_request_rejects_unrelated_family_labeled_suite(
    implemented_adapter,
) -> None:
    with pytest.raises(ValueError, match="live family projection|compact arrays"):
        build_candidate_execution_request(
            execution_id="unrelated_suite",
            adapter_manifest=implemented_adapter,
            invocation=_exact_invocation(),
            benchmark_suite=_unrelated_suite(),
            archive_path="/tmp/.palette_benchmarks/execution/archive.zarr",
            source_run_path="analysis/swim_bout_runs/source_v1",
            candidate_run_path="analysis/swim_bout_runs/candidate_v1",
            scratch_root="/tmp/palette-candidate-execution-scratch",
            source_identity_sha256="b" * 64,
            palette_commit="a" * 40,
            repetition_index=0,
            candidate_order_index=0,
            candidate_order_count=1,
            cache_state="fresh_process_os_cache_uncontrolled",
            physical_io_scope=PhysicalIOScope.UNAVAILABLE,
            selector_before_sha256="f" * 64,
            registry_probe_path=Path(__file__).resolve(),
            production_profiles_probe_path=Path(__file__).resolve(),
        )


def test_coordinated_logical_replacement_cannot_escape_request(
    implemented_adapter,
) -> None:
    receipt = _receipt(implemented_adapter)
    tampered = deepcopy(receipt)
    equality = tampered["payload"]["logical_equality"]
    equality["source_logical_manifest_sha256"] = "9" * 64
    equality["candidate_logical_manifest_sha256"] = "9" * 64
    tampered["payload_digest"] = canonical_json_sha256(tampered["payload"])

    with pytest.raises(ValueError, match="decoded equality"):
        require_candidate_execution_receipt(
            tampered,
            expected_request_payload_digest=receipt["payload"][
                "request_payload_digest"
            ],
        )


def test_coordinate_role_cannot_claim_unowned_authority(
    implemented_adapter,
) -> None:
    receipt = _receipt(implemented_adapter)
    tampered = deepcopy(receipt)
    coordinate = tampered["payload"]["coordinate_evidence"]
    coordinate["published_authority_sha256"] = "9" * 64
    coordinate["published_authority_ref"] = "invented/authority.json"
    tampered["payload_digest"] = canonical_json_sha256(tampered["payload"])

    with pytest.raises(ValueError, match="only bind source authorities"):
        require_candidate_execution_receipt(
            tampered,
            expected_request_payload_digest=receipt["payload"][
                "request_payload_digest"
            ],
        )


def test_nonmutation_baseline_is_anchored_in_request(implemented_adapter) -> None:
    receipt = _receipt(implemented_adapter)
    tampered = deepcopy(receipt)
    evidence = tampered["payload"]["nonmutation_evidence"]
    evidence["selector_before_sha256"] = "8" * 64
    evidence["selector_after_sha256"] = "8" * 64
    tampered["payload_digest"] = canonical_json_sha256(tampered["payload"])

    with pytest.raises(ValueError, match="requested pre-state"):
        require_candidate_execution_receipt(
            tampered,
            expected_request_payload_digest=receipt["payload"][
                "request_payload_digest"
            ],
        )


def test_receipt_requires_external_request_anchor(implemented_adapter) -> None:
    receipt = _receipt(implemented_adapter)
    with pytest.raises(ValueError, match="request binding"):
        require_candidate_execution_receipt(
            receipt,
            expected_request_payload_digest="9" * 64,
        )


def test_execution_receipt_v1_rejects_failure_status(implemented_adapter) -> None:
    receipt = _receipt(implemented_adapter)
    tampered = deepcopy(receipt)
    tampered["payload"]["status"] = "failed"
    tampered["payload"]["publication_gate_passed"] = False
    tampered["payload_digest"] = canonical_json_sha256(tampered["payload"])

    with pytest.raises(ValueError, match="completed publications only"):
        require_candidate_execution_receipt(
            tampered,
            expected_request_payload_digest=receipt["payload"][
                "request_payload_digest"
            ],
        )


def test_completed_receipt_rejects_zero_storage_bytes(implemented_adapter) -> None:
    receipt = _receipt(implemented_adapter)
    tampered = deepcopy(receipt)
    tampered["payload"]["output_storage"]["apparent_bytes"] = 0
    tampered["payload"]["output_storage"]["allocated_bytes"] = 0
    tampered["payload_digest"] = canonical_json_sha256(tampered["payload"])

    with pytest.raises(ValueError, match="positive bytes"):
        require_candidate_execution_receipt(
            tampered,
            expected_request_payload_digest=receipt["payload"][
                "request_payload_digest"
            ],
        )


def test_not_applicable_phase_cannot_claim_timing() -> None:
    with pytest.raises(ValueError, match="must not claim measurements"):
        CandidatePhaseMeasurement(
            phase=CandidateExecutionPhase.ATOMIC_PUBLICATION,
            outcome=PhaseOutcome.NOT_APPLICABLE,
            started_at_utc=None,
            completed_at_utc=None,
            wall_seconds=0.0,
            cpu_user_seconds=None,
            cpu_system_seconds=None,
            peak_process_tree_rss_bytes=None,
            not_applicable_reason="guarded direct publication",
        )
