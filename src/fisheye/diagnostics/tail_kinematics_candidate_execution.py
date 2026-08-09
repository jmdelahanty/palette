"""Fresh-process execution for exact tail-kinematics storage candidates.

The runner recomputes one immutable candidate from the canonical subject-shape
authority named by a closed invocation.  It compares every decoded core array
and the optional source-revision bundle to an explicit complete source run,
accepts the candidate only inside the atomic publication lock, and emits either
one immutable receipt or one separate failed-attempt sidecar.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys
from typing import Any, Mapping, Optional, Sequence

import zarr

from fisheye.analysis import tail_kinematics_runs as tail_runs
from fisheye.analysis.tail_kinematics_schema import infer_tail_kinematics_dimensions
from fisheye.analysis.tail_kinematics_storage import (
    ANALYSIS_STORAGE_PLAN_RECEIPT_ATTR,
    build_tail_kinematics_storage_receipt,
    validate_tail_kinematics_storage_receipt,
)
from fisheye.analysis_workflows.analysis_candidate_execution import (
    CandidateComputationMode,
    CandidatePhaseMeasurement,
    PhaseOutcome,
    PhysicalIOScope,
    build_candidate_execution_receipt,
    protected_path_snapshot_sha256,
    require_candidate_execution_receipt,
    require_candidate_execution_request,
    required_execution_phases,
)
from fisheye.shared.runtime_telemetry import (
    require_runtime_telemetry,
)
from fisheye.analysis_workflows.materializers.tail_kinematics import (
    EXECUTION_BINDING_ATTR,
    TAIL_KINEMATICS_EXECUTION_PHASE_ORDER,
    materialize_tail_kinematics,
    tombstone_tail_kinematics_execution_candidate,
)
from fisheye.analysis_workflows.tail_kinematics_candidate_execution import (
    TAIL_KINEMATICS_EXECUTION_FAMILY_ID,
    TAIL_KINEMATICS_EXECUTION_PROFILE_ID,
    build_tail_kinematics_coordinate_evidence,
    compute_tail_kinematics_logical_hashes,
    require_tail_kinematics_execution_suite,
    require_tail_kinematics_invocation_manifest,
)
from fisheye.diagnostics.analysis_candidate_execution import (
    ATTEMPT_SCHEMA_ID,
    ATTEMPT_SCHEMA_VERSION,
    _build_failed_attempt,
    _child_start_time_ticks,
    _execution_binding,
    _git_identity,
    _physical_io_evidence,
    _proc_io,
    _process_start_time_ticks,
    _read_json,
    _target_state,
    _write_json_exclusive,
    require_exact_tabular_execution_attempt,
)
from fisheye.shared import tail_coordinate_publication as tail_publication
from fisheye.shared.json_safety import json_attr_safe
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from fisheye.shared.zarr.storage_profiles import get_storage_profile
from fisheye.shared.zarr_io import open_zarr_root

RUNNER_REF = (
    "fisheye.diagnostics.tail_kinematics_candidate_execution:"
    "execute_tail_kinematics_candidate"
)
RUN_PARENT = "analysis/tail_kinematics_runs"
_SELECTOR_SNAPSHOT_SCHEMA_ID = "palette.analysis_candidate_selector_snapshot"
_SELECTOR_SNAPSHOT_SCHEMA_VERSION = 1


class TailKinematicsCandidateExecutionFailed(RuntimeError):
    """Raised with one self-validating shared-v2 failed-attempt record."""

    def __init__(self, message: str, *, attempt: Mapping[str, Any]) -> None:
        super().__init__(message)
        self.attempt = dict(attempt)


def _attrs(group: Any) -> dict[str, Any]:
    attrs = group.attrs
    return dict(attrs.asdict() if hasattr(attrs, "asdict") else dict(attrs))


def _this_runner_sha256() -> str:
    digest = hashlib.sha256()
    with Path(__file__).open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def tail_kinematics_selector_snapshot_sha256(archive_path: str | Path) -> str:
    """Digest the complete selector-owning tail-kinematics parent surface."""

    root = open_zarr_root(Path(archive_path).expanduser().resolve(), mode="r")
    parent = root[RUN_PARENT]
    payload = {
        "schema_id": _SELECTOR_SNAPSHOT_SCHEMA_ID,
        "schema_version": _SELECTOR_SNAPSHOT_SCHEMA_VERSION,
        "stage_id": TAIL_KINEMATICS_EXECUTION_FAMILY_ID,
        "run_parent": RUN_PARENT,
        "parent_attributes": json_attr_safe(_attrs(parent)),
    }
    return canonical_json_sha256(payload)


def require_tail_kinematics_execution_request(value: Mapping[str, Any]) -> None:
    """Require shared request-v2 plus the family-owned closed invocation."""

    if not isinstance(value, Mapping) or not isinstance(value.get("payload"), Mapping):
        raise ValueError("tail-kinematics execution request is not an envelope")
    require_tail_kinematics_invocation_manifest(value["payload"].get("invocation"))
    # This becomes executable when the parent integrates the same exact grammar
    # and runner descriptor into the shared invocation/catalog foundations.
    require_candidate_execution_request(value)


def _source_preflight(
    root: Any,
    source_path: str,
    *,
    parameters: Mapping[str, Any],
    planned_storage_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    if (
        not source_path.startswith(f"{RUN_PARENT}/")
        or source_path.count("/") != 2
        or source_path.rsplit("/", 1)[1] in {"latest", "latest_complete"}
    ):
        raise ValueError("tail-kinematics source must be one explicit immutable run")
    source = root[source_path]
    attrs = _attrs(source)
    if (
        attrs.get("palette_run_completion_status") != "complete"
        or attrs.get("stage_selector_eligible") is not True
    ):
        raise ValueError("tail-kinematics source must be complete and eligible")
    publication = tail_publication.load_tail_kinematics_coordinate_publication(
        root,
        source_path,
    )
    if (
        publication.manifest.record_sha256
        != parameters["source_tail_coordinate_manifest_sha256"]
    ):
        raise ValueError("live tail publication differs from the invocation")
    if (
        publication.source.manifest.record_sha256
        != parameters["source_subject_shape_manifest_sha256"]
    ):
        raise ValueError("live subject-shape publication differs from the invocation")
    shape_run = str(parameters["source_subject_shape_run"])
    resolved_shape_run, _shape_group, resolved_sources = (
        tail_runs._resolve_tail_kinematics_sources(root, shape_run)
    )
    if (
        resolved_shape_run != shape_run
        or resolved_sources.source_publication_manifest_sha256
        != parameters["source_subject_shape_manifest_sha256"]
        or attrs.get("source_subject_shape_run") != shape_run
        or str(attrs.get("source_subject_shape_path") or "").strip("/")
        != f"analysis/subject_shape_runs/{shape_run}"
    ):
        raise ValueError("tail source is not bound to the invoked subject shape")
    dimensions = infer_tail_kinematics_dimensions(source)
    if dimensions.n_tail_samples != parameters["tail_angle_sample_count"]:
        raise ValueError("tail sample count differs from the invocation")
    expected_receipt = build_tail_kinematics_storage_receipt(
        dimensions,
        profile=get_storage_profile(TAIL_KINEMATICS_EXECUTION_PROFILE_ID),
    ).as_manifest()
    if expected_receipt != dict(planned_storage_receipt):
        raise ValueError("live tail dimensions differ from the benchmark suite")
    logical_hashes = compute_tail_kinematics_logical_hashes(source)
    return {
        "source_group": source,
        "publication": publication,
        "dimensions": dimensions,
        "logical_hashes": logical_hashes,
        "logical_manifest_sha256": canonical_json_sha256(logical_hashes),
        "source_authority_sha256": resolved_sources.source_authority.get(
            "record_sha256"
        ),
    }


def _validate_published_candidate(
    candidate: Any,
    *,
    expected_hashes: Mapping[str, Any],
    expected_storage_receipt: Mapping[str, Any],
) -> None:
    errors = list(validate_tail_kinematics_storage_receipt(candidate))
    attrs = _attrs(candidate)
    if (
        attrs.get("palette_run_completion_status") != "complete"
        or attrs.get("stage_selector_eligible") is not False
        or attrs.get("storage_candidate_profile_promoted") is not False
        or attrs.get(ANALYSIS_STORAGE_PLAN_RECEIPT_ATTR)
        != dict(expected_storage_receipt)
    ):
        errors.append("published candidate lifecycle or storage binding differs")
    if compute_tail_kinematics_logical_hashes(candidate) != dict(expected_hashes):
        errors.append("published candidate decoded values differ")
    if errors:
        raise ValueError(
            "published tail-kinematics candidate validation failed: "
            + "; ".join(errors)
        )


def _load_ineligible_publication(root: Any, candidate_path: str) -> Any:
    """Freshly load the complete ineligible publication without rewriting it."""

    return tail_publication._load_tail_coordinate_publication(
        root,
        candidate_path,
        expected_selector_eligible=False,
        expected_kind="tail_kinematics",
        require_complete=True,
    )


def _translate_phases(runtime: Mapping[str, Any]) -> list[CandidatePhaseMeasurement]:
    require_runtime_telemetry(
        runtime,
        expected_materializer="tail_kinematics_candidate",
        allowed_phase_order=TAIL_KINEMATICS_EXECUTION_PHASE_ORDER,
        require_error_phase=False,
    )
    records = runtime.get("phases")
    expected = required_execution_phases(CandidateComputationMode.SCIENTIFIC_COMPUTE)
    if not isinstance(records, list) or [record.get("name") for record in records] != [
        phase.value for phase in expected
    ]:
        raise ValueError("tail-kinematics materializer phase sequence differs")
    result: list[CandidatePhaseMeasurement] = []
    for phase, record in zip(expected, records, strict=True):
        if record.get("outcome") != "ok":
            raise ValueError("successful tail materializer has a failed phase")
        cpu = record.get("cpu_seconds")
        if not isinstance(cpu, Mapping):
            raise ValueError("tail-kinematics phase CPU telemetry is invalid")
        result.append(
            CandidatePhaseMeasurement(
                phase=phase,
                outcome=PhaseOutcome.SUCCEEDED,
                started_at_utc=record.get("started_at_utc"),
                completed_at_utc=record.get("finished_at_utc"),
                wall_seconds=record.get("wall_seconds"),
                cpu_user_seconds=float(cpu.get("own_user_cpu_seconds", 0.0))
                + float(cpu.get("child_user_cpu_seconds", 0.0)),
                cpu_system_seconds=float(cpu.get("own_system_cpu_seconds", 0.0))
                + float(cpu.get("child_system_cpu_seconds", 0.0)),
                peak_process_tree_rss_bytes=max(
                    int(record.get("process_peak_rss_bytes_at_end", 0)),
                    int(record.get("children_peak_rss_bytes_at_end", 0)),
                ),
            )
        )
    return result


def require_tail_kinematics_execution_attempt(
    value: Mapping[str, Any],
    *,
    expected_request_payload_digest: str,
) -> None:
    """Validate shared attempt-v2 plus tail-specific telemetry."""

    if (
        not isinstance(value, Mapping)
        or set(value) != {"schema_id", "schema_version", "payload", "payload_digest"}
        or value.get("schema_id") != ATTEMPT_SCHEMA_ID
        or value.get("schema_version") != ATTEMPT_SCHEMA_VERSION
        or value.get("payload_digest") != canonical_json_sha256(value.get("payload"))
    ):
        raise ValueError("tail-kinematics execution-attempt envelope differs")
    runtime = value["payload"].get("runtime_telemetry")
    compatibility = deepcopy(dict(value))
    compatibility["payload"]["runtime_telemetry"] = None
    compatibility["payload_digest"] = canonical_json_sha256(compatibility["payload"])
    require_exact_tabular_execution_attempt(
        compatibility,
        expected_request_payload_digest=expected_request_payload_digest,
    )
    if runtime is not None:
        require_runtime_telemetry(
            runtime,
            expected_materializer="tail_kinematics_candidate",
            allowed_phase_order=TAIL_KINEMATICS_EXECUTION_PHASE_ORDER,
            require_error_phase=value["payload"]["failure_phase"] == "materializer",
        )


def _failed_attempt(
    *,
    request: Mapping[str, Any],
    driver_pid: int,
    child_start_time_ticks: int,
    child_pid: int,
    error: BaseException,
    failure_phase: str,
    runtime_telemetry: Mapping[str, Any] | None,
    selector_after_sha256: str | None,
    registry_after_sha256: str | None,
    production_profiles_after_sha256: str | None,
    observation_errors: Mapping[str, str] | None = None,
    reported_error_type: str | None = None,
    reported_error_message: str | None = None,
) -> dict[str, Any]:
    result = _build_failed_attempt(
        request=request,
        driver_pid=driver_pid,
        child_start_time_ticks=child_start_time_ticks,
        child_pid=child_pid,
        error=error,
        failure_phase=failure_phase,
        runtime_telemetry=None,
        selector_after_sha256=selector_after_sha256,
        registry_after_sha256=registry_after_sha256,
        production_profiles_after_sha256=production_profiles_after_sha256,
        observation_errors=observation_errors,
        reported_error_type=reported_error_type,
        reported_error_message=reported_error_message,
    )
    result["payload"]["runtime_telemetry"] = (
        None if runtime_telemetry is None else json_attr_safe(runtime_telemetry)
    )
    result["payload_digest"] = canonical_json_sha256(result["payload"])
    require_tail_kinematics_execution_attempt(
        result,
        expected_request_payload_digest=str(request["payload_digest"]),
    )
    return result


def execute_tail_kinematics_candidate(
    request: Mapping[str, Any],
    *,
    driver_pid: int,
) -> dict[str, Any]:
    """Execute one digest-bound tail candidate in a direct child."""

    require_tail_kinematics_execution_request(request)
    payload = request["payload"]
    adapter = payload["adapter_manifest"]["payload"]
    invocation = payload["invocation"]["payload"]
    if (
        adapter["stage_id"] != TAIL_KINEMATICS_EXECUTION_FAMILY_ID
        or adapter["computation_mode"] != "scientific_compute"
        or invocation["contract_id"] != "tail_kinematics_v1"
    ):
        raise ValueError("tail runner received another adapter or invocation")
    if type(driver_pid) is not int or driver_pid <= 0 or driver_pid != os.getppid():
        raise ValueError("tail execution must run in one direct child")
    scope = PhysicalIOScope(payload["physical_io_scope"])
    if scope not in {PhysicalIOScope.UNAVAILABLE, PhysicalIOScope.PROCESS_SELF_PROC_IO}:
        raise ValueError("external transfer scope requires a traced-process driver")

    parameters = invocation["parameters"]
    archive = Path(payload["archive_path"])
    source_path = str(payload["source_run_path"])
    candidate_path = str(payload["candidate_run_path"])
    candidate_name = candidate_path.rsplit("/", 1)[1]
    protected = payload["protected_state_before"]
    selector_before = str(protected["selector_sha256"])
    execution_binding = _execution_binding(request)
    child_start = _child_start_time_ticks()
    io_before = _proc_io()
    failure_phase = "runner_preflight"
    runtime: Mapping[str, Any] | None = None
    planned_receipt = payload["benchmark_suite"]["payload"]["storage_plan_receipt"]
    try:
        if tail_kinematics_selector_snapshot_sha256(archive) != selector_before:
            raise ValueError("live tail selector differs from the request")
        commit, dirty = _git_identity()
        if commit != payload["palette_commit"]:
            raise ValueError("runner Git commit differs from the request")
        require_tail_kinematics_execution_suite(
            TAIL_KINEMATICS_EXECUTION_FAMILY_ID,
            payload["benchmark_suite"],
        )
        root = open_zarr_root(archive, mode="r")
        source = _source_preflight(
            root,
            source_path,
            parameters=parameters,
            planned_storage_receipt=planned_receipt,
        )
        source_hashes = source["logical_hashes"]
        source_identity = source["logical_manifest_sha256"]
        if source_identity != payload["source_identity_sha256"]:
            raise ValueError("tail decoded source identity differs from request")
        source_publication = source["publication"]

        def accept_published_candidate(
            published_root: zarr.Group,
            _published_parent: zarr.Group,
            candidate: zarr.Group,
        ) -> Mapping[str, Any]:
            if candidate.attrs.get(EXECUTION_BINDING_ATTR) != execution_binding:
                raise ValueError("published tail execution binding differs")
            fresh = _source_preflight(
                published_root,
                source_path,
                parameters=parameters,
                planned_storage_receipt=planned_receipt,
            )
            if (
                fresh["logical_hashes"] != source_hashes
                or fresh["logical_manifest_sha256"] != source_identity
                or fresh["source_authority_sha256"] != source["source_authority_sha256"]
            ):
                raise ValueError("tail source changed during publication")
            _validate_published_candidate(
                candidate,
                expected_hashes=source_hashes,
                expected_storage_receipt=planned_receipt,
            )
            candidate_publication = _load_ineligible_publication(
                published_root,
                candidate_path,
            )
            coordinate = build_tail_kinematics_coordinate_evidence(
                source_publication=source_publication,
                candidate_publication=candidate_publication,
            )
            selector_after = tail_kinematics_selector_snapshot_sha256(archive)
            if selector_after != selector_before:
                raise ValueError("tail execution changed selector state")
            return {
                "coordinate_evidence": coordinate,
                "selector_after_sha256": selector_after,
                "execution_binding": execution_binding,
                "source_identity_sha256": source_identity,
            }

        failure_phase = "materializer"
        materialized = materialize_tail_kinematics(
            archive,
            scratch_root=payload["scratch_root"],
            shape_run=str(parameters["source_subject_shape_run"]),
            run_name=candidate_name,
            tail_angle_sample_count=int(parameters["tail_angle_sample_count"]),
            block_rows=int(parameters["block_rows"]),
            output_shard_rows=int(parameters["output_shard_rows"]),
            execution_backend=str(parameters["execution_backend"]),
            num_workers=int(parameters["num_workers"]),
            copy_backend=str(parameters["copy_backend"]),
            apply=True,
            keep_scratch=bool(parameters["keep_scratch"]),
            check_capacity=bool(parameters["check_capacity"]),
            storage_profile=get_storage_profile(str(parameters["storage_profile_id"])),
            execution_binding=execution_binding,
            expected_source_logical_hashes=source_hashes,
            publication_acceptance_validator=accept_published_candidate,
        )
        runtime = materialized["runtime_telemetry"]
        phases = _translate_phases(runtime)

        failure_phase = "runner_receipt_assembly"
        acceptance = materialized.get("caller_acceptance")
        if (
            not isinstance(acceptance, Mapping)
            or acceptance.get("execution_binding") != execution_binding
            or acceptance.get("selector_after_sha256") != selector_before
            or acceptance.get("source_identity_sha256") != source_identity
        ):
            raise ValueError("atomic tail acceptance evidence differs")
        coordinate = acceptance["coordinate_evidence"]
        compared_count = len(source_hashes["arrays"])
        receipt = build_candidate_execution_receipt(
            request=request,
            status="complete",
            fresh_process={
                "driver_pid": driver_pid,
                "child_pid": os.getpid(),
                "child_start_time_ticks": child_start,
                "is_fresh": True,
            },
            environment={
                "hostname": socket.gethostname(),
                "platform": platform.platform(),
                "python_version": platform.python_version(),
                "python_executable": sys.executable,
                "palette_commit": commit,
                "palette_git_dirty": dirty,
                "runner_ref": RUNNER_REF,
                "runner_sha256": _this_runner_sha256(),
                "cache_state": payload["cache_state"],
            },
            phases=phases,
            publication_runtime_telemetry=materialized["publish"][
                "runtime_telemetry"
            ],
            coordinate_evidence=coordinate,
            logical_equality={
                "contract_id": adapter["logical_equality_contract"],
                "compared_array_count": compared_count,
                "source_logical_manifest_sha256": source_identity,
                "candidate_logical_manifest_sha256": materialized[
                    "published_logical_manifest_sha256"
                ],
                "equal": True,
            },
            metadata_equivalence={
                "local_array_count": materialized[
                    "local_direct_consolidated_array_count"
                ],
                "published_array_count": materialized[
                    "published_direct_consolidated_array_count"
                ],
                "local_equal": True,
                "published_equal": True,
            },
            physical_io=_physical_io_evidence(
                request=request,
                before=io_before,
                after=_proc_io(),
            ),
            output_storage=materialized["output_storage"],
            nonmutation_evidence={
                "selector_before_sha256": selector_before,
                "selector_after_sha256": selector_before,
                "registry_before_sha256": protected["registry_sha256"],
                "registry_after_sha256": protected["registry_sha256"],
                "production_profiles_before_sha256": protected[
                    "production_profiles_sha256"
                ],
                "production_profiles_after_sha256": protected[
                    "production_profiles_sha256"
                ],
                "snapshot_contract_id": protected["snapshot_contract_id"],
                "unchanged": True,
            },
        )
        require_candidate_execution_receipt(
            receipt,
            expected_request_payload_digest=str(request["payload_digest"]),
        )
        return receipt
    except BaseException as exc:
        attached = getattr(exc, "palette_runtime_telemetry", None)
        if isinstance(attached, Mapping):
            runtime = attached
        target = _target_state(archive, candidate_path)
        if target.get("completion_status") == "complete":
            tombstone_tail_kinematics_execution_candidate(
                archive,
                run_name=candidate_name,
                expected_execution_binding=execution_binding,
                failure_phase=failure_phase,
                error_type=type(exc).__name__,
                error_message=str(exc),
            )
        attempt = _failed_attempt(
            request=request,
            driver_pid=driver_pid,
            child_start_time_ticks=child_start,
            child_pid=os.getpid(),
            error=exc,
            failure_phase=failure_phase,
            runtime_telemetry=runtime,
            selector_after_sha256=tail_kinematics_selector_snapshot_sha256(archive),
            registry_after_sha256=str(protected["registry_sha256"]),
            production_profiles_after_sha256=str(
                protected["production_profiles_sha256"]
            ),
        )
        raise TailKinematicsCandidateExecutionFailed(
            f"Tail-kinematics execution failed during {failure_phase}: {exc}",
            attempt=attempt,
        ) from exc


def run_tail_kinematics_candidate_fresh_process(
    request_path: str | Path,
    *,
    receipt_path: str | Path,
    attempt_path: str | Path,
) -> dict[str, Any]:
    """Launch once and expose evidence only after protected-state observation."""

    request_file = Path(request_path).expanduser().resolve()
    receipt_file = Path(receipt_path).expanduser().resolve()
    attempt_file = Path(attempt_path).expanduser().resolve()
    if receipt_file == attempt_file:
        raise ValueError("receipt and failed-attempt paths must differ")
    if receipt_file.exists() or attempt_file.exists():
        raise FileExistsError("fresh-process evidence path already exists")
    request = _read_json(request_file)
    require_tail_kinematics_execution_request(request)
    payload = request["payload"]
    archive = Path(payload["archive_path"])
    protected = payload["protected_state_before"]
    for path in (receipt_file, attempt_file):
        if ".palette_benchmarks" not in path.parts or path.is_relative_to(archive):
            raise ValueError("execution evidence must be outside the Zarr benchmark")
    if (
        protected_path_snapshot_sha256(protected["registry_probe_path"])
        != protected["registry_sha256"]
        or protected_path_snapshot_sha256(protected["production_profiles_probe_path"])
        != protected["production_profiles_sha256"]
        or tail_kinematics_selector_snapshot_sha256(archive)
        != protected["selector_sha256"]
    ):
        raise ValueError("protected pre-state differs from execution request")

    hidden_receipt = receipt_file.with_name(f".{receipt_file.name}.child.{os.getpid()}")
    hidden_attempt = attempt_file.with_name(f".{attempt_file.name}.child.{os.getpid()}")
    if hidden_receipt.exists() or hidden_attempt.exists():
        raise FileExistsError("fresh-process hidden evidence path already exists")
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            __name__,
            "child",
            "--request",
            str(request_file),
            "--receipt",
            str(hidden_receipt),
            "--attempt",
            str(hidden_attempt),
            "--driver-pid",
            str(os.getpid()),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    child_pid = int(process.pid)
    try:
        child_start = _process_start_time_ticks(child_pid)
    except (FileNotFoundError, ProcessLookupError):
        child_start = 1
    stdout, stderr = process.communicate()

    after: dict[str, str | None] = {}
    errors: dict[str, str] = {}
    observers = {
        "selector_after_sha256": lambda: tail_kinematics_selector_snapshot_sha256(
            archive
        ),
        "registry_after_sha256": lambda: protected_path_snapshot_sha256(
            protected["registry_probe_path"]
        ),
        "production_profiles_after_sha256": lambda: protected_path_snapshot_sha256(
            protected["production_profiles_probe_path"]
        ),
    }
    for field, observer in observers.items():
        try:
            after[field] = observer()
        except BaseException as exc:
            after[field] = None
            errors[field] = f"{type(exc).__name__}: {exc}"
    unchanged = bool(
        not errors
        and after["selector_after_sha256"] == protected["selector_sha256"]
        and after["registry_after_sha256"] == protected["registry_sha256"]
        and after["production_profiles_after_sha256"]
        == protected["production_profiles_sha256"]
    )

    def terminal_attempt(
        error: BaseException,
        *,
        phase: str,
        runtime: Mapping[str, Any] | None = None,
        reported_error_type: str | None = None,
        reported_error_message: str | None = None,
    ) -> dict[str, Any]:
        target = _target_state(archive, str(payload["candidate_run_path"]))
        if target.get("completion_status") == "complete":
            tombstone_tail_kinematics_execution_candidate(
                archive,
                run_name=str(payload["candidate_run_path"]).rsplit("/", 1)[1],
                expected_execution_binding=_execution_binding(request),
                failure_phase=phase,
                error_type=type(error).__name__,
                error_message=str(error),
            )
        attempt = _failed_attempt(
            request=request,
            driver_pid=os.getpid(),
            child_start_time_ticks=child_start,
            child_pid=child_pid,
            error=error,
            failure_phase=phase,
            runtime_telemetry=runtime,
            selector_after_sha256=after["selector_after_sha256"],
            registry_after_sha256=after["registry_after_sha256"],
            production_profiles_after_sha256=after["production_profiles_after_sha256"],
            observation_errors=errors,
            reported_error_type=reported_error_type,
            reported_error_message=reported_error_message,
        )
        _write_json_exclusive(attempt_file, attempt)
        return attempt

    try:
        if process.returncode == 0:
            if not hidden_receipt.is_file() or hidden_attempt.exists():
                error = RuntimeError("successful child wrote invalid evidence")
                attempt = terminal_attempt(error, phase="driver_child_evidence")
                raise TailKinematicsCandidateExecutionFailed(
                    str(error), attempt=attempt
                )
            try:
                provisional = _read_json(hidden_receipt)
                require_candidate_execution_receipt(
                    provisional,
                    expected_request_payload_digest=str(request["payload_digest"]),
                )
                if provisional["payload"]["fresh_process"] != {
                    "driver_pid": os.getpid(),
                    "child_pid": child_pid,
                    "child_start_time_ticks": child_start,
                    "is_fresh": True,
                }:
                    raise ValueError("child process identity differs")
            except BaseException as error:
                attempt = terminal_attempt(error, phase="driver_child_evidence")
                raise TailKinematicsCandidateExecutionFailed(
                    str(error), attempt=attempt
                ) from error
            if not unchanged:
                error = RuntimeError("protected state changed after child execution")
                attempt = terminal_attempt(
                    error,
                    phase="driver_protected_poststate",
                )
                raise TailKinematicsCandidateExecutionFailed(
                    str(error), attempt=attempt
                )
            try:
                _write_json_exclusive(receipt_file, provisional)
            except BaseException as error:
                receipt_file.unlink(missing_ok=True)
                attempt = terminal_attempt(
                    error,
                    phase="driver_receipt_publication",
                )
                raise TailKinematicsCandidateExecutionFailed(
                    str(error), attempt=attempt
                ) from error
            return provisional

        runtime: Mapping[str, Any] | None = None
        phase = "driver_child_failure"
        reported_type: str | None = None
        reported_message: str | None = None
        child_error: BaseException = RuntimeError(
            "fresh tail child exited nonzero: "
            f"returncode={process.returncode}, stdout={stdout!r}, stderr={stderr!r}"
        )
        if hidden_attempt.is_file() and not hidden_receipt.exists():
            try:
                child_attempt = _read_json(hidden_attempt)
                require_tail_kinematics_execution_attempt(
                    child_attempt,
                    expected_request_payload_digest=str(request["payload_digest"]),
                )
            except BaseException as error:
                phase = "driver_child_evidence"
                child_error = error
            else:
                child_payload = child_attempt["payload"]
                runtime = child_payload.get("runtime_telemetry")
                phase = str(child_payload["failure_phase"])
                reported_type = str(child_payload["error_type"])
                reported_message = str(child_payload["error_message"])
                child_error = RuntimeError(f"{reported_type}: {reported_message}")
        attempt = terminal_attempt(
            child_error,
            phase=phase,
            runtime=runtime,
            reported_error_type=reported_type,
            reported_error_message=reported_message,
        )
        raise TailKinematicsCandidateExecutionFailed(
            "Fresh tail child failed; see immutable attempt record.",
            attempt=attempt,
        )
    finally:
        hidden_receipt.unlink(missing_ok=True)
        hidden_attempt.unlink(missing_ok=True)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    child = subparsers.add_parser("child")
    child.add_argument("--request", type=Path, required=True)
    child.add_argument("--receipt", type=Path, required=True)
    child.add_argument("--attempt", type=Path, required=True)
    child.add_argument("--driver-pid", type=int, required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    request = _read_json(args.request.resolve())
    try:
        receipt = execute_tail_kinematics_candidate(
            request,
            driver_pid=args.driver_pid,
        )
    except TailKinematicsCandidateExecutionFailed as exc:
        _write_json_exclusive(args.attempt.resolve(), exc.attempt)
        return 1
    _write_json_exclusive(args.receipt.resolve(), receipt)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "RUNNER_REF",
    "TailKinematicsCandidateExecutionFailed",
    "execute_tail_kinematics_candidate",
    "require_tail_kinematics_execution_attempt",
    "require_tail_kinematics_execution_request",
    "run_tail_kinematics_candidate_fresh_process",
    "tail_kinematics_selector_snapshot_sha256",
]
