"""Strict execution evidence for derived-analysis storage candidates.

The benchmark-suite manifest plans deterministic work.  This module binds one
fresh-process execution to that plan and records the exact phase, coordinate,
publication, equality, storage, and physical-I/O evidence produced by the
family adapter.  It deliberately contains no dynamic ``**kwargs`` dispatcher
and grants no selector, registry, profile, or production mutation authority.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
from typing import Any, Mapping, Sequence

from fisheye.shared.zarr.analysis_benchmark_suite import (
    require_analysis_benchmark_suite_manifest,
)
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256

from .analysis_candidate_invocation import (
    CandidateInvocationContract,
    require_candidate_invocation_manifest,
)
from .materializers.atomic_run_publisher import (
    ATOMIC_RUN_PUBLISHER_OPTIONAL_SUCCESS_PHASES,
    ATOMIC_RUN_PUBLISHER_PHASE_ORDER,
)
from .materializers.runtime_telemetry import require_runtime_telemetry

ANALYSIS_CANDIDATE_EXECUTION_REQUEST_SCHEMA_ID = (
    "palette.analysis_candidate_execution_request"
)
ANALYSIS_CANDIDATE_EXECUTION_REQUEST_SCHEMA_VERSION = 2
ANALYSIS_CANDIDATE_EXECUTION_RECEIPT_SCHEMA_ID = (
    "palette.analysis_candidate_execution_receipt"
)
ANALYSIS_CANDIDATE_EXECUTION_RECEIPT_SCHEMA_VERSION = 4
ANALYSIS_CANDIDATE_EXECUTION_RECEIPT_LEGACY_SCHEMA_VERSION = 2
ANALYSIS_CANDIDATE_EXECUTION_RECEIPT_HIERARCHICAL_SCHEMA_VERSION = 3
ANALYSIS_CANDIDATE_EXECUTION_ADAPTER_SCHEMA_ID = (
    "palette.analysis_candidate_execution_adapter"
)
ANALYSIS_CANDIDATE_EXECUTION_ADAPTER_SCHEMA_VERSION = 2

_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_.-]*$")
_MODULE = re.compile(r"^[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)+$")
_CALLABLE = re.compile(r"^_?[a-z][a-z0-9_]*$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")
_KNOWN_INVOCATION_CONTRACTS = {
    contract.value for contract in CandidateInvocationContract
}


class CandidateLogicalEqualityContract(str, Enum):
    """Closed decoded-equality surface owned by each candidate family."""

    TRACK_FLAT_PROJECTION_V1 = "track_flat_projection_v1"
    SWIM_BOUTS_DECLARED_ARRAYS_V1 = "swim_bouts_declared_arrays_v1"
    BOUT_KINEMATICS_DECLARED_ARRAYS_V1 = "bout_kinematics_declared_arrays_v1"
    EYE_ANGLE_COMPACT_V7_ARRAYS_V1 = "eye_angle_compact_v7_arrays_v1"
    SUBJECT_SHAPE_V4_ARRAYS_V1 = "subject_shape_v4_arrays_v1"
    TAIL_KINEMATICS_DECLARED_ARRAYS_V1 = "tail_kinematics_declared_arrays_v1"
    STIMULUS_RESPONSE_V3_ARRAYS_V1 = "stimulus_response_v3_arrays_v1"
    STIMULUS_EPOCH_V2_ARRAYS_V1 = "stimulus_epoch_v2_arrays_v1"
    DETECTION_OCCUPANCY_DECLARED_ARRAYS_V1 = "detection_occupancy_declared_arrays_v1"
    SESSION_OCCUPANCY_DECLARED_ARRAYS_V1 = "session_occupancy_declared_arrays_v1"
    CHASER_DISTANCE_SEALED_BASE_V2_ARRAYS_V1 = (
        "chaser_distance_sealed_base_v2_arrays_v1"
    )
    TAIL_POSTURE_V3_ARRAYS_V1 = "tail_posture_v3_arrays_v1"
    BOUT_CLASSIFICATION_V2_ARRAYS_V1 = "bout_classification_v2_arrays_v1"


class CandidateComputationMode(str, Enum):
    """How one candidate obtains its scientific payload."""

    LOGICAL_REMATERIALIZATION = "logical_rematerialization"
    SCIENTIFIC_COMPUTE = "scientific_compute"
    GUARDED_DIRECT_WRITER = "guarded_direct_writer"


class CandidateRunnerStatus(str, Enum):
    """Whether the shared runner may execute an adapter today."""

    CONTRACT_ONLY = "contract_only"
    IMPLEMENTED = "implemented"
    BLOCKED_DIRECT_PUBLICATION = "blocked_direct_publication"
    BLOCKED_COORDINATE_AUTHORITY = "blocked_coordinate_authority"


class CoordinateContractRole(str, Enum):
    """Scientific coordinate responsibility of one derived family."""

    CANONICAL_PRODUCER = "canonical_producer"
    BOUND_DERIVATIVE = "bound_derivative"
    TEMPORAL_AXIS_ONLY = "temporal_axis_only"
    CANONICAL_BINDING_REQUIRED = "canonical_binding_required"


class CoordinateContractStatus(str, Enum):
    """Current executable coordinate gate behind one adapter."""

    CANONICAL_PUBLICATION_IMPLEMENTED = "canonical_publication_implemented"
    BOUND_SOURCE_VALIDATION_IMPLEMENTED = "bound_source_validation_implemented"
    TEMPORAL_AXIS_IMPLEMENTED = "temporal_axis_implemented"
    SOURCE_PRESERVATION_ONLY = "source_preservation_only"
    BLOCKED_CANONICAL_BINDING = "blocked_canonical_binding"


class CoordinateEvidenceStatus(str, Enum):
    """Observed coordinate result from one execution."""

    VERIFIED_CANONICAL_PUBLICATION = "verified_canonical_publication"
    VERIFIED_BOUND_SOURCE = "verified_bound_source"
    VERIFIED_TEMPORAL_AXIS = "verified_temporal_axis"
    VERIFIED_SOURCE_PRESERVATION_NONMINTING = "verified_source_preservation_nonminting"
    BLOCKED = "blocked"


class CandidateExecutionPhase(str, Enum):
    """Exact phase vocabulary for writer/publication evidence."""

    PLAN = "plan"
    SOURCE_STAGING = "source_staging"
    SCIENTIFIC_COMPUTE = "scientific_compute"
    LOGICAL_REMATERIALIZATION = "logical_rematerialization"
    LOCAL_VALIDATION = "local_validation"
    LOCAL_CONSOLIDATION = "local_consolidation"
    LOCAL_DIRECT_CONSOLIDATED_COMPARISON = "local_direct_consolidated_comparison"
    ATOMIC_PUBLICATION = "atomic_publication"
    PUBLISHED_VALIDATION = "published_validation"
    PUBLISHED_DIRECT_CONSOLIDATED_COMPARISON = (
        "published_direct_consolidated_comparison"
    )
    DECODED_EQUALITY = "decoded_equality"
    PHYSICAL_INVENTORY = "physical_inventory"
    PUBLICATION_ACCEPTANCE_VALIDATION = "publication_acceptance_validation"


_ATOMIC_PUBLICATION_CHILD_PHASES = frozenset(
    {
        CandidateExecutionPhase.PUBLISHED_VALIDATION,
        CandidateExecutionPhase.PUBLISHED_DIRECT_CONSOLIDATED_COMPARISON,
        CandidateExecutionPhase.DECODED_EQUALITY,
        CandidateExecutionPhase.PHYSICAL_INVENTORY,
        CandidateExecutionPhase.PUBLICATION_ACCEPTANCE_VALIDATION,
    }
)


class PhaseOutcome(str, Enum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    NOT_APPLICABLE = "not_applicable"


class PhysicalIOScope(str, Enum):
    """Provenance for I/O counters; scopes are not interchangeable."""

    UNAVAILABLE = "unavailable"
    PROCESS_SELF_PROC_IO = "process_self_proc_io"
    PROCESS_TREE_FILE_SYSCALLS = "process_tree_file_syscalls"
    FILESYSTEM_OR_NETWORK_TRANSFER = "filesystem_or_network_transfer"
    TENSORSTORE_FILE_METRICS = "tensorstore_file_metrics"


TRANSFER_PHYSICAL_IO_SCOPES = frozenset(
    {
        PhysicalIOScope.FILESYSTEM_OR_NETWORK_TRANSFER,
    }
)

_NODE_LOCAL_SCRATCH_ROOTS = (
    Path("/tmp"),
    Path("/var/tmp"),
    Path("/scratch"),
    Path("/dev/shm"),
    Path("/local"),
    Path("/lscratch"),
)
_NONMUTATION_SNAPSHOT_CONTRACT_ID = "analysis_candidate_nonmutation_v1"
_PROTECTED_PATH_SNAPSHOT_CONTRACT_ID = "palette.protected_path_content_sha256.v1"


def _json_copy(value: object) -> Any:
    try:
        return json.loads(
            json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            )
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"value is not strict JSON: {exc}") from exc


def _require_identifier(value: object, *, label: str) -> str:
    if type(value) is not str or not _IDENTIFIER.fullmatch(value):
        raise ValueError(f"{label} must be one canonical identifier")
    return value


def _require_sha256(value: object, *, label: str) -> str:
    if type(value) is not str or not _SHA256.fullmatch(value):
        raise ValueError(f"{label} must be one lowercase SHA-256 digest")
    return value


def _require_utc_timestamp(value: object, *, label: str) -> datetime:
    if type(value) is not str or not value:
        raise ValueError(f"{label} must be one UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{label} must be one ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ValueError(f"{label} must carry an explicit UTC offset")
    return parsed


def _require_relative_run_path(value: object, *, label: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise ValueError(f"{label} must be one nonempty relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"{label} must be one canonical relative path")
    if any(not re.fullmatch(r"[A-Za-z0-9_.-]+", part) for part in path.parts):
        raise ValueError(f"{label} contains an unsafe path component")
    return value


def _absolute_path(value: object, *, label: str) -> Path:
    if type(value) is not str or not value:
        raise ValueError(f"{label} must be one absolute path string")
    path = Path(value)
    if not path.is_absolute() or path != path.resolve():
        raise ValueError(f"{label} must already be one resolved absolute path")
    return path


def _require_node_local_scratch_path(value: object, *, label: str) -> Path:
    path = _absolute_path(value, label=label)
    if not any(
        path == root or path.is_relative_to(root) for root in _NODE_LOCAL_SCRATCH_ROOTS
    ):
        raise ValueError(f"{label} must be below a recognized node-local scratch root")
    return path


def _adapter_payload(value: Mapping[str, Any]) -> Mapping[str, Any]:
    require_candidate_execution_adapter_manifest(value)
    return value["payload"]


def require_candidate_execution_adapter_manifest(
    value: Mapping[str, Any],
) -> None:
    """Deeply validate one typed family-adapter declaration."""

    if not isinstance(value, Mapping) or set(value) != {
        "schema_id",
        "schema_version",
        "payload",
        "payload_digest",
    }:
        raise ValueError("execution adapter envelope field set differs")
    if (
        value["schema_id"] != ANALYSIS_CANDIDATE_EXECUTION_ADAPTER_SCHEMA_ID
        or type(value["schema_version"]) is not int
        or value["schema_version"]
        != ANALYSIS_CANDIDATE_EXECUTION_ADAPTER_SCHEMA_VERSION
    ):
        raise ValueError("execution adapter schema identity differs")
    payload = value["payload"]
    if not isinstance(payload, Mapping) or set(payload) != {
        "stage_id",
        "adapter_id",
        "adapter_version",
        "source_run_parent",
        "run_parent",
        "profile_id",
        "candidate_owner_module",
        "candidate_owner_entrypoint",
        "invocation_contract",
        "computation_mode",
        "publication_mode",
        "runner_status",
        "runner_module",
        "runner_entrypoint",
        "suite_validator_module",
        "suite_validator_entrypoint",
        "coordinate_role",
        "coordinate_contract_status",
        "logical_equality_contract",
    }:
        raise ValueError("execution adapter payload field set differs")
    if value["payload_digest"] != canonical_json_sha256(payload):
        raise ValueError("execution adapter payload digest differs")
    _json_copy(value)
    _require_identifier(payload["stage_id"], label="adapter stage_id")
    _require_identifier(payload["adapter_id"], label="adapter_id")
    if type(payload["adapter_version"]) is not int or payload["adapter_version"] < 1:
        raise ValueError("adapter_version must be one positive exact integer")
    _require_relative_run_path(payload["run_parent"], label="adapter run_parent")
    _require_relative_run_path(
        payload["source_run_parent"], label="adapter source_run_parent"
    )
    _require_identifier(payload["profile_id"], label="adapter profile_id")
    if payload["invocation_contract"] not in _KNOWN_INVOCATION_CONTRACTS:
        raise ValueError("adapter invocation_contract is unsupported")
    try:
        CandidateLogicalEqualityContract(payload["logical_equality_contract"])
    except (TypeError, ValueError) as exc:
        raise ValueError("adapter logical_equality_contract is unsupported") from exc
    try:
        CandidateComputationMode(payload["computation_mode"])
        CandidateRunnerStatus(payload["runner_status"])
        CoordinateContractRole(payload["coordinate_role"])
        CoordinateContractStatus(payload["coordinate_contract_status"])
    except (TypeError, ValueError) as exc:
        raise ValueError("execution adapter contains an unsupported enum") from exc
    if payload["publication_mode"] not in {
        "shared_atomic_nonpromoting_v1",
        "guarded_direct_nonpromoting_v1",
    }:
        raise ValueError("execution adapter publication_mode differs")
    if (
        type(payload["candidate_owner_module"]) is not str
        or not _MODULE.fullmatch(payload["candidate_owner_module"])
        or type(payload["candidate_owner_entrypoint"]) is not str
        or not _CALLABLE.fullmatch(payload["candidate_owner_entrypoint"])
    ):
        raise ValueError("execution adapter requires one exact candidate owner")
    runner_module = payload["runner_module"]
    runner_entrypoint = payload["runner_entrypoint"]
    suite_validator_module = payload["suite_validator_module"]
    suite_validator_entrypoint = payload["suite_validator_entrypoint"]
    if runner_module is not None and (
        type(runner_module) is not str or not _MODULE.fullmatch(runner_module)
    ):
        raise ValueError("runner_module must be None or one module path")
    if runner_entrypoint is not None and (
        type(runner_entrypoint) is not str or not _CALLABLE.fullmatch(runner_entrypoint)
    ):
        raise ValueError("runner_entrypoint must be None or one callable name")
    if suite_validator_module is not None and (
        type(suite_validator_module) is not str
        or not _MODULE.fullmatch(suite_validator_module)
    ):
        raise ValueError("suite_validator_module must be None or one module path")
    if suite_validator_entrypoint is not None and (
        type(suite_validator_entrypoint) is not str
        or not _CALLABLE.fullmatch(suite_validator_entrypoint)
    ):
        raise ValueError("suite_validator_entrypoint must be None or one callable name")
    status = CandidateRunnerStatus(payload["runner_status"])
    if status is CandidateRunnerStatus.IMPLEMENTED:
        if any(
            value is None
            for value in (
                runner_module,
                runner_entrypoint,
                suite_validator_module,
                suite_validator_entrypoint,
            )
        ):
            raise ValueError(
                "implemented adapters require one typed runner and suite validator"
            )
    elif any(
        value is not None
        for value in (
            runner_module,
            runner_entrypoint,
            suite_validator_module,
            suite_validator_entrypoint,
        )
    ):
        raise ValueError(
            "nonimplemented adapters must not claim a runner or suite validator"
        )
    if (
        payload["publication_mode"] == "guarded_direct_nonpromoting_v1"
        and status is not CandidateRunnerStatus.BLOCKED_DIRECT_PUBLICATION
    ):
        raise ValueError("guarded-direct adapters must remain publication blocked")
    if (
        CoordinateContractStatus(payload["coordinate_contract_status"])
        is CoordinateContractStatus.BLOCKED_CANONICAL_BINDING
        and status is not CandidateRunnerStatus.BLOCKED_COORDINATE_AUTHORITY
    ):
        raise ValueError("unbound coordinate adapters must remain execution blocked")
    mode = CandidateComputationMode(payload["computation_mode"])
    if (payload["publication_mode"] == "guarded_direct_nonpromoting_v1") != (
        mode is CandidateComputationMode.GUARDED_DIRECT_WRITER
    ):
        raise ValueError("adapter computation and publication modes disagree")
    role = CoordinateContractRole(payload["coordinate_role"])
    coordinate_status = CoordinateContractStatus(payload["coordinate_contract_status"])
    allowed_statuses = {
        CoordinateContractRole.CANONICAL_PRODUCER: {
            CoordinateContractStatus.CANONICAL_PUBLICATION_IMPLEMENTED,
            CoordinateContractStatus.SOURCE_PRESERVATION_ONLY,
        },
        CoordinateContractRole.BOUND_DERIVATIVE: {
            CoordinateContractStatus.BOUND_SOURCE_VALIDATION_IMPLEMENTED,
        },
        CoordinateContractRole.TEMPORAL_AXIS_ONLY: {
            CoordinateContractStatus.TEMPORAL_AXIS_IMPLEMENTED,
        },
        CoordinateContractRole.CANONICAL_BINDING_REQUIRED: {
            CoordinateContractStatus.BLOCKED_CANONICAL_BINDING,
        },
    }[role]
    if coordinate_status not in allowed_statuses:
        raise ValueError("adapter coordinate role and implementation status disagree")


def _require_registered_adapter_manifest(value: Mapping[str, Any]) -> Mapping[str, Any]:
    """Require the embedded adapter to equal the live versioned catalog entry."""

    payload = _adapter_payload(value)
    from .analysis_candidate_execution_catalog import (  # local to avoid a cycle
        ANALYSIS_CANDIDATE_EXECUTION_ADAPTER_BY_STAGE,
    )

    stage_id = str(payload["stage_id"])
    registered = ANALYSIS_CANDIDATE_EXECUTION_ADAPTER_BY_STAGE.get(stage_id)
    if registered is None or registered.as_manifest() != _json_copy(value):
        raise ValueError("execution adapter differs from the registered catalog entry")
    return payload


def _require_suite_matches_adapter(
    adapter: Mapping[str, Any],
    benchmark_suite: Mapping[str, Any],
) -> None:
    module_name = adapter["suite_validator_module"]
    entrypoint = adapter["suite_validator_entrypoint"]
    if type(module_name) is not str or type(entrypoint) is not str:
        raise ValueError("implemented adapter lacks its suite validator")
    from importlib import import_module

    validator = getattr(import_module(module_name), entrypoint, None)
    if not callable(validator):
        raise ValueError("execution adapter suite validator is not callable")
    validator(str(adapter["stage_id"]), benchmark_suite)


def _require_immediate_run_child(
    path: str,
    *,
    run_parent: str,
    label: str,
) -> str:
    """Require exactly ``<catalog parent>/<one immutable run name>``."""

    prefix = f"{run_parent}/"
    if not path.startswith(prefix):
        raise ValueError(f"{label} must use the adapter run parent")
    run_name = path[len(prefix) :]
    if not run_name or "/" in run_name:
        raise ValueError(f"{label} must name one immediate run child")
    return path


def protected_path_snapshot_sha256(path: str | Path) -> str:
    """Digest one explicit protected file/tree with unambiguous entry framing."""

    def frame(digest: Any, tag: bytes, payload: bytes) -> None:
        digest.update(tag)
        digest.update(len(payload).to_bytes(8, byteorder="big", signed=False))
        digest.update(payload)

    def file_digest(candidate: Path) -> bytes:
        content = hashlib.sha256()
        with candidate.open("rb") as stream:
            while block := stream.read(1024 * 1024):
                content.update(block)
        return content.digest()

    supplied = Path(path).expanduser()
    if not supplied.is_absolute():
        raise ValueError("Protected-state probe path must be absolute.")
    if supplied.is_symlink():
        raise ValueError(f"Protected-state probe target is a symlink: {supplied}.")
    target = supplied.resolve(strict=True)
    digest = hashlib.sha256()
    if target.is_file():
        digest.update(b"palette-protected-file-v2\0")
        frame(digest, b"content\0", file_digest(target))
        return digest.hexdigest()
    if not target.is_dir():
        raise ValueError(f"Protected-state probe target is not a file/tree: {target}.")
    digest.update(b"palette-protected-tree-v2\0")
    entries = sorted(target.rglob("*"), key=lambda candidate: candidate.as_posix())
    for candidate in entries:
        if candidate.is_symlink():
            raise ValueError(
                f"Protected-state probe tree contains a symlink: {candidate}."
            )
        relative = candidate.relative_to(target).as_posix().encode("utf-8")
        if candidate.is_dir():
            frame(digest, b"directory\0", relative)
            continue
        if candidate.is_file():
            frame(digest, b"file-path\0", relative)
            frame(digest, b"file-content-sha256\0", file_digest(candidate))
            continue
        raise ValueError(
            f"Protected-state probe tree has an unsupported node: {candidate}."
        )
    return digest.hexdigest()


def build_candidate_execution_request(
    *,
    execution_id: str,
    adapter_manifest: Mapping[str, Any],
    invocation: Mapping[str, Any],
    benchmark_suite: Mapping[str, Any],
    archive_path: str | Path,
    source_run_path: str,
    candidate_run_path: str,
    scratch_root: str | Path,
    source_identity_sha256: str,
    palette_commit: str,
    repetition_index: int,
    candidate_order_index: int,
    candidate_order_count: int,
    cache_state: str,
    physical_io_scope: PhysicalIOScope,
    selector_before_sha256: str,
    registry_probe_path: str | Path,
    production_profiles_probe_path: str | Path,
    nonmutation_snapshot_contract_id: str = _NONMUTATION_SNAPSHOT_CONTRACT_ID,
) -> dict[str, object]:
    """Build one executable, nonproduction, digest-bound request."""

    adapter = _require_registered_adapter_manifest(adapter_manifest)
    if (
        CandidateRunnerStatus(adapter["runner_status"])
        is not CandidateRunnerStatus.IMPLEMENTED
    ):
        raise ValueError("execution requests require an implemented typed adapter")
    require_analysis_benchmark_suite_manifest(benchmark_suite, require_current=True)
    _require_suite_matches_adapter(adapter, benchmark_suite)
    require_candidate_invocation_manifest(
        invocation,
        expected_contract=str(adapter["invocation_contract"]),
        expected_profile_id=str(adapter["profile_id"]),
    )
    suite_payload = benchmark_suite["payload"]
    if suite_payload["family_id"] != adapter["stage_id"]:
        raise ValueError("benchmark suite and execution adapter families differ")
    archive = Path(archive_path).expanduser().resolve()
    scratch = _require_node_local_scratch_path(
        str(Path(scratch_root).expanduser().resolve()), label="scratch_root"
    )
    if not archive.is_absolute() or ".palette_benchmarks" not in archive.parts:
        raise ValueError("execution archive must live below .palette_benchmarks")
    if not scratch.is_absolute():
        raise ValueError("scratch_root must be one absolute path")
    if (
        archive == scratch
        or archive.is_relative_to(scratch)
        or scratch.is_relative_to(archive)
    ):
        raise ValueError("scratch and benchmark archive must not contain one another")
    source_path = _require_relative_run_path(source_run_path, label="source_run_path")
    candidate_path = _require_relative_run_path(
        candidate_run_path, label="candidate_run_path"
    )
    if source_path == candidate_path:
        raise ValueError("source and candidate run paths must differ")
    source_run_parent = str(adapter["source_run_parent"])
    candidate_run_parent = str(adapter["run_parent"])
    _require_immediate_run_child(
        source_path,
        run_parent=source_run_parent,
        label="source_run_path",
    )
    _require_immediate_run_child(
        candidate_path,
        run_parent=candidate_run_parent,
        label="candidate_run_path",
    )
    _require_identifier(execution_id, label="execution_id")
    _require_sha256(source_identity_sha256, label="source_identity_sha256")
    if type(palette_commit) is not str or not _GIT_SHA.fullmatch(palette_commit):
        raise ValueError("palette_commit must be one full lowercase Git SHA")
    repetitions = suite_payload["repetitions"]
    if (
        type(repetition_index) is not int
        or repetition_index < 0
        or repetition_index >= repetitions
    ):
        raise ValueError("repetition_index is outside the benchmark suite")
    if type(candidate_order_count) is not int or candidate_order_count < 1:
        raise ValueError("candidate_order_count must be one positive exact integer")
    if (
        type(candidate_order_index) is not int
        or candidate_order_index < 0
        or candidate_order_index >= candidate_order_count
    ):
        raise ValueError("candidate_order_index is outside the candidate order")
    _require_identifier(cache_state, label="cache_state")
    if not isinstance(physical_io_scope, PhysicalIOScope):
        raise TypeError("physical_io_scope must use PhysicalIOScope")
    _require_sha256(selector_before_sha256, label="selector_before_sha256")
    registry_supplied = Path(registry_probe_path).expanduser()
    profiles_supplied = Path(production_profiles_probe_path).expanduser()
    registry_before_sha256 = protected_path_snapshot_sha256(registry_supplied)
    production_profiles_before_sha256 = protected_path_snapshot_sha256(
        profiles_supplied
    )
    registry_probe = registry_supplied.resolve(strict=True)
    profiles_probe = profiles_supplied.resolve(strict=True)
    if nonmutation_snapshot_contract_id != _NONMUTATION_SNAPSHOT_CONTRACT_ID:
        raise ValueError("nonmutation snapshot contract differs")
    payload: dict[str, object] = {
        "execution_id": execution_id,
        "adapter_manifest": _json_copy(adapter_manifest),
        "invocation": _json_copy(invocation),
        "benchmark_suite": _json_copy(benchmark_suite),
        "archive_path": str(archive),
        "source_run_path": source_path,
        "candidate_run_path": candidate_path,
        "scratch_root": str(scratch),
        "source_identity_sha256": source_identity_sha256,
        "palette_commit": palette_commit,
        "repetition_index": repetition_index,
        "candidate_order_index": candidate_order_index,
        "candidate_order_count": candidate_order_count,
        "cache_state": cache_state,
        "physical_io_scope": physical_io_scope.value,
        "protected_state_before": {
            "snapshot_contract_id": nonmutation_snapshot_contract_id,
            "selector_sha256": selector_before_sha256,
            "registry_sha256": registry_before_sha256,
            "production_profiles_sha256": production_profiles_before_sha256,
            "probe_contract_id": _PROTECTED_PATH_SNAPSHOT_CONTRACT_ID,
            "registry_probe_path": str(registry_probe),
            "production_profiles_probe_path": str(profiles_probe),
        },
        "execution_policy": {
            "benchmark_only": True,
            "node_local_compute": True,
            "selector_mutation_authorized": False,
            "registry_mutation_authorized": False,
            "production_profile_promotion_authorized": False,
            "canonical_data_mutation_authorized": False,
        },
    }
    result = {
        "schema_id": ANALYSIS_CANDIDATE_EXECUTION_REQUEST_SCHEMA_ID,
        "schema_version": ANALYSIS_CANDIDATE_EXECUTION_REQUEST_SCHEMA_VERSION,
        "payload": payload,
        "payload_digest": canonical_json_sha256(payload),
    }
    require_candidate_execution_request(result)
    return result


def require_candidate_execution_request(value: Mapping[str, Any]) -> None:
    """Deeply validate one request, including suite and adapter digests."""

    if not isinstance(value, Mapping) or set(value) != {
        "schema_id",
        "schema_version",
        "payload",
        "payload_digest",
    }:
        raise ValueError("execution request envelope field set differs")
    if (
        value["schema_id"] != ANALYSIS_CANDIDATE_EXECUTION_REQUEST_SCHEMA_ID
        or type(value["schema_version"]) is not int
        or value["schema_version"]
        != ANALYSIS_CANDIDATE_EXECUTION_REQUEST_SCHEMA_VERSION
    ):
        raise ValueError("execution request schema identity differs")
    payload = value["payload"]
    expected = {
        "execution_id",
        "adapter_manifest",
        "invocation",
        "benchmark_suite",
        "archive_path",
        "source_run_path",
        "candidate_run_path",
        "scratch_root",
        "source_identity_sha256",
        "palette_commit",
        "repetition_index",
        "candidate_order_index",
        "candidate_order_count",
        "cache_state",
        "physical_io_scope",
        "protected_state_before",
        "execution_policy",
    }
    if not isinstance(payload, Mapping) or set(payload) != expected:
        raise ValueError("execution request payload field set differs")
    if value["payload_digest"] != canonical_json_sha256(payload):
        raise ValueError("execution request payload digest differs")
    _json_copy(value)
    adapter = _require_registered_adapter_manifest(payload["adapter_manifest"])
    if (
        CandidateRunnerStatus(adapter["runner_status"])
        is not CandidateRunnerStatus.IMPLEMENTED
    ):
        raise ValueError("execution request adapter is not implemented")
    require_candidate_invocation_manifest(
        payload["invocation"],
        expected_contract=str(adapter["invocation_contract"]),
        expected_profile_id=str(adapter["profile_id"]),
    )
    suite = payload["benchmark_suite"]
    require_analysis_benchmark_suite_manifest(suite)
    _require_suite_matches_adapter(adapter, suite)
    suite_payload = suite["payload"]
    if suite_payload["family_id"] != adapter["stage_id"]:
        raise ValueError("execution request suite family differs from adapter")
    receipt_payload = suite_payload["storage_plan_receipt"]["payload"]
    if receipt_payload["storage_profile"]["profile_id"] != adapter["profile_id"]:
        raise ValueError("execution request storage profile differs from adapter")
    _require_identifier(payload["execution_id"], label="execution_id")
    archive = _absolute_path(payload["archive_path"], label="archive_path")
    scratch = _require_node_local_scratch_path(
        payload["scratch_root"], label="scratch_root"
    )
    if ".palette_benchmarks" not in archive.parts:
        raise ValueError("execution archive is outside .palette_benchmarks")
    if (
        archive == scratch
        or archive.is_relative_to(scratch)
        or scratch.is_relative_to(archive)
    ):
        raise ValueError("scratch and benchmark archive overlap")
    source_path = _require_relative_run_path(
        payload["source_run_path"], label="source_run_path"
    )
    candidate_path = _require_relative_run_path(
        payload["candidate_run_path"], label="candidate_run_path"
    )
    source_run_parent = str(adapter["source_run_parent"])
    candidate_run_parent = str(adapter["run_parent"])
    if source_path == candidate_path:
        raise ValueError("execution request run paths differ from adapter ownership")
    _require_immediate_run_child(
        source_path,
        run_parent=source_run_parent,
        label="source_run_path",
    )
    _require_immediate_run_child(
        candidate_path,
        run_parent=candidate_run_parent,
        label="candidate_run_path",
    )
    _require_sha256(payload["source_identity_sha256"], label="source_identity_sha256")
    if type(payload["palette_commit"]) is not str or not _GIT_SHA.fullmatch(
        payload["palette_commit"]
    ):
        raise ValueError("execution request palette_commit is invalid")
    repetitions = suite_payload["repetitions"]
    repetition_index = payload["repetition_index"]
    if (
        type(repetition_index) is not int
        or repetition_index < 0
        or repetition_index >= repetitions
    ):
        raise ValueError("execution request repetition_index is outside its suite")
    order_count = payload["candidate_order_count"]
    order_index = payload["candidate_order_index"]
    if (
        type(order_count) is not int
        or order_count < 1
        or type(order_index) is not int
        or order_index < 0
        or order_index >= order_count
    ):
        raise ValueError("execution request candidate order is invalid")
    _require_identifier(payload["cache_state"], label="cache_state")
    try:
        PhysicalIOScope(payload["physical_io_scope"])
    except (TypeError, ValueError) as exc:
        raise ValueError("execution request physical_io_scope is unsupported") from exc
    protected = _require_exact_fields(
        payload["protected_state_before"],
        {
            "snapshot_contract_id",
            "selector_sha256",
            "registry_sha256",
            "production_profiles_sha256",
            "probe_contract_id",
            "registry_probe_path",
            "production_profiles_probe_path",
        },
        label="protected pre-state",
    )
    if protected["snapshot_contract_id"] != _NONMUTATION_SNAPSHOT_CONTRACT_ID:
        raise ValueError("protected pre-state snapshot contract differs")
    if protected["probe_contract_id"] != _PROTECTED_PATH_SNAPSHOT_CONTRACT_ID:
        raise ValueError("protected pre-state path-probe contract differs")
    for field in (
        "selector_sha256",
        "registry_sha256",
        "production_profiles_sha256",
    ):
        _require_sha256(protected[field], label=f"protected pre-state {field}")
    _absolute_path(protected["registry_probe_path"], label="registry_probe_path")
    _absolute_path(
        protected["production_profiles_probe_path"],
        label="production_profiles_probe_path",
    )
    if payload["execution_policy"] != {
        "benchmark_only": True,
        "node_local_compute": True,
        "selector_mutation_authorized": False,
        "registry_mutation_authorized": False,
        "production_profile_promotion_authorized": False,
        "canonical_data_mutation_authorized": False,
    }:
        raise ValueError("execution request safety policy differs")


@dataclass(frozen=True)
class CandidatePhaseMeasurement:
    """One exact phase timing and process-tree resource observation."""

    phase: CandidateExecutionPhase
    outcome: PhaseOutcome
    started_at_utc: str | None
    completed_at_utc: str | None
    wall_seconds: float | None
    cpu_user_seconds: float | None
    cpu_system_seconds: float | None
    peak_process_tree_rss_bytes: int | None
    parent_phase: CandidateExecutionPhase | None = None
    not_applicable_reason: str | None = None
    error_type: str | None = None
    error_message: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.phase, CandidateExecutionPhase):
            raise TypeError("phase must use CandidateExecutionPhase")
        if not isinstance(self.outcome, PhaseOutcome):
            raise TypeError("outcome must use PhaseOutcome")
        if self.parent_phase is None and self.phase in _ATOMIC_PUBLICATION_CHILD_PHASES:
            object.__setattr__(
                self,
                "parent_phase",
                CandidateExecutionPhase.ATOMIC_PUBLICATION,
            )
        if self.parent_phase is not None and not isinstance(
            self.parent_phase, CandidateExecutionPhase
        ):
            raise TypeError("parent_phase must use CandidateExecutionPhase or None")
        if self.parent_phase is self.phase:
            raise ValueError("a phase cannot be its own parent")
        timing = (
            self.started_at_utc,
            self.completed_at_utc,
            self.wall_seconds,
            self.cpu_user_seconds,
            self.cpu_system_seconds,
            self.peak_process_tree_rss_bytes,
        )
        if self.outcome is PhaseOutcome.NOT_APPLICABLE:
            if any(value is not None for value in timing):
                raise ValueError("not-applicable phases must not claim measurements")
            if (
                type(self.not_applicable_reason) is not str
                or not self.not_applicable_reason.strip()
                or self.error_type is not None
                or self.error_message is not None
            ):
                raise ValueError(
                    "not-applicable phase reason is missing or inconsistent"
                )
            return
        started = _require_utc_timestamp(
            self.started_at_utc, label="phase started_at_utc"
        )
        completed = _require_utc_timestamp(
            self.completed_at_utc, label="phase completed_at_utc"
        )
        if completed < started:
            raise ValueError("phase completion timestamp precedes its start")
        for field in ("wall_seconds", "cpu_user_seconds", "cpu_system_seconds"):
            observed = getattr(self, field)
            if (
                type(observed) not in {int, float}
                or not math.isfinite(float(observed))
                or observed < 0
            ):
                raise ValueError(f"{field} must be one nonnegative finite number")
        if (
            type(self.peak_process_tree_rss_bytes) is not int
            or self.peak_process_tree_rss_bytes < 0
        ):
            raise ValueError("executed phases require nonnegative peak RSS")
        if self.not_applicable_reason is not None:
            raise ValueError("executed phases must not have a not-applicable reason")
        if self.outcome is PhaseOutcome.SUCCEEDED:
            if self.error_type is not None or self.error_message is not None:
                raise ValueError("successful phases must not contain an error")
        elif (
            type(self.error_type) is not str
            or not self.error_type
            or type(self.error_message) is not str
            or not self.error_message
        ):
            raise ValueError("failed phases require exact error type and message")

    def as_manifest(self) -> dict[str, object]:
        return {
            "phase": self.phase.value,
            "outcome": self.outcome.value,
            "started_at_utc": self.started_at_utc,
            "completed_at_utc": self.completed_at_utc,
            "wall_seconds": self.wall_seconds,
            "cpu_user_seconds": self.cpu_user_seconds,
            "cpu_system_seconds": self.cpu_system_seconds,
            "peak_process_tree_rss_bytes": self.peak_process_tree_rss_bytes,
            "parent_phase": (
                None if self.parent_phase is None else self.parent_phase.value
            ),
            "not_applicable_reason": self.not_applicable_reason,
            "error_type": self.error_type,
            "error_message": self.error_message,
        }


def required_execution_phases(
    computation_mode: CandidateComputationMode,
) -> tuple[CandidateExecutionPhase, ...]:
    """Return the exact ordered phase list for one computation mode."""

    if not isinstance(computation_mode, CandidateComputationMode):
        raise TypeError("computation_mode must use CandidateComputationMode")
    if computation_mode is CandidateComputationMode.GUARDED_DIRECT_WRITER:
        raise ValueError("guarded-direct candidates cannot enter the shared runner")
    compute = (
        CandidateExecutionPhase.SCIENTIFIC_COMPUTE
        if computation_mode is CandidateComputationMode.SCIENTIFIC_COMPUTE
        else CandidateExecutionPhase.LOGICAL_REMATERIALIZATION
    )
    return (
        CandidateExecutionPhase.PLAN,
        CandidateExecutionPhase.SOURCE_STAGING,
        compute,
        CandidateExecutionPhase.LOCAL_VALIDATION,
        CandidateExecutionPhase.LOCAL_CONSOLIDATION,
        CandidateExecutionPhase.LOCAL_DIRECT_CONSOLIDATED_COMPARISON,
        CandidateExecutionPhase.ATOMIC_PUBLICATION,
        CandidateExecutionPhase.PUBLISHED_VALIDATION,
        CandidateExecutionPhase.PUBLISHED_DIRECT_CONSOLIDATED_COMPARISON,
        CandidateExecutionPhase.DECODED_EQUALITY,
        CandidateExecutionPhase.PHYSICAL_INVENTORY,
        CandidateExecutionPhase.PUBLICATION_ACCEPTANCE_VALIDATION,
    )


def _require_exact_fields(
    value: object,
    fields: set[str],
    *,
    label: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError(f"{label} field set differs")
    return value


def _require_nonnegative_int(value: object, *, label: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{label} must be one nonnegative exact integer")
    return value


def _require_nonnegative_number(value: object, *, label: str) -> float:
    if type(value) not in {int, float} or not math.isfinite(float(value)) or value < 0:
        raise ValueError(f"{label} must be one nonnegative finite number")
    return float(value)


def _require_current_phase_chronology(
    phases: Sequence[CandidatePhaseMeasurement],
) -> None:
    """Require a nonoverlapping sibling timeline with contained child phases."""

    by_phase = {phase.phase: phase for phase in phases}
    if len(by_phase) != len(phases):
        raise ValueError("execution receipt phase identities are duplicated")
    children_by_parent: dict[
        CandidateExecutionPhase, list[CandidatePhaseMeasurement]
    ] = {}
    top_level: list[CandidatePhaseMeasurement] = []
    for phase in phases:
        if phase.parent_phase is None:
            top_level.append(phase)
        else:
            if phase.parent_phase not in by_phase:
                raise ValueError("execution receipt phase parent is absent")
            children_by_parent.setdefault(phase.parent_phase, []).append(phase)

    def executed_interval(
        phase: CandidatePhaseMeasurement,
    ) -> tuple[datetime, datetime] | None:
        if phase.outcome is PhaseOutcome.NOT_APPLICABLE:
            return None
        return (
            _require_utc_timestamp(
                phase.started_at_utc,
                label=f"{phase.phase.value} started_at_utc",
            ),
            _require_utc_timestamp(
                phase.completed_at_utc,
                label=f"{phase.phase.value} completed_at_utc",
            ),
        )

    def require_nonoverlapping_siblings(
        siblings: Sequence[CandidatePhaseMeasurement],
    ) -> None:
        prior_finished: datetime | None = None
        for phase in siblings:
            interval = executed_interval(phase)
            if interval is None:
                continue
            started, finished = interval
            if prior_finished is not None and started < prior_finished:
                raise ValueError("execution receipt sibling phase intervals overlap")
            prior_finished = finished

    require_nonoverlapping_siblings(top_level)
    for parent_phase, children in children_by_parent.items():
        parent = by_phase[parent_phase]
        parent_interval = executed_interval(parent)
        if parent_interval is None:
            raise ValueError("executed child phase has a non-applicable parent")
        parent_started, parent_finished = parent_interval
        require_nonoverlapping_siblings(children)
        child_wall_sum = 0.0
        for child in children:
            interval = executed_interval(child)
            if interval is None:
                continue
            child_started, child_finished = interval
            if child_started < parent_started or child_finished > parent_finished:
                raise ValueError("execution receipt child phase escapes its parent")
            assert child.wall_seconds is not None
            child_wall_sum += float(child.wall_seconds)
        assert parent.wall_seconds is not None
        if child_wall_sum > float(parent.wall_seconds) + 1e-9:
            raise ValueError("execution receipt child wall time exceeds its parent")


def _require_atomic_publication_runtime_telemetry(
    value: Mapping[str, Any],
    *,
    phases: Sequence[CandidatePhaseMeasurement],
) -> None:
    """Validate and bind the mechanical publisher trace to its parent phase."""

    require_runtime_telemetry(
        value,
        expected_materializer="atomic_run_publisher",
        allowed_phase_order=ATOMIC_RUN_PUBLISHER_PHASE_ORDER,
        require_error_phase=False,
        require_current=True,
    )
    observed_names = tuple(str(phase["name"]) for phase in value["phases"])
    required_names = set(ATOMIC_RUN_PUBLISHER_PHASE_ORDER).difference(
        ATOMIC_RUN_PUBLISHER_OPTIONAL_SUCCESS_PHASES
    )
    if set(observed_names).difference(ATOMIC_RUN_PUBLISHER_PHASE_ORDER):
        raise ValueError("atomic publication telemetry contains an unknown phase")
    if not required_names.issubset(observed_names):
        raise ValueError("atomic publication telemetry omits a required success phase")
    if any(phase["outcome"] != "ok" for phase in value["phases"]):
        raise ValueError("successful execution contains failed publication telemetry")

    parent = next(
        (
            phase
            for phase in phases
            if phase.phase is CandidateExecutionPhase.ATOMIC_PUBLICATION
        ),
        None,
    )
    if parent is None or parent.outcome is not PhaseOutcome.SUCCEEDED:
        raise ValueError("atomic publication telemetry lacks a successful parent phase")
    parent_started = _require_utc_timestamp(
        parent.started_at_utc,
        label="atomic publication parent started_at_utc",
    )
    parent_finished = _require_utc_timestamp(
        parent.completed_at_utc,
        label="atomic publication parent completed_at_utc",
    )
    nested_started = _require_utc_timestamp(
        value["started_at_utc"],
        label="atomic publisher started_at_utc",
    )
    nested_finished = _require_utc_timestamp(
        value["finished_at_utc"],
        label="atomic publisher finished_at_utc",
    )
    if nested_started < parent_started or nested_finished > parent_finished:
        raise ValueError("atomic publication telemetry escapes its execution parent")

    assert parent.wall_seconds is not None
    nested_wall = _require_nonnegative_number(
        value["wall_seconds"], label="atomic publisher wall_seconds"
    )
    if nested_wall > float(parent.wall_seconds) + 1e-9:
        raise ValueError("atomic publisher wall time exceeds its execution parent")

    assert parent.cpu_user_seconds is not None
    assert parent.cpu_system_seconds is not None
    cpu = value["cpu_seconds"]
    nested_user = float(cpu["own_user_cpu_seconds"]) + float(
        cpu["child_user_cpu_seconds"]
    )
    nested_system = float(cpu["own_system_cpu_seconds"]) + float(
        cpu["child_system_cpu_seconds"]
    )
    if nested_user > float(parent.cpu_user_seconds) + 1e-9:
        raise ValueError("atomic publisher user CPU exceeds its execution parent")
    if nested_system > float(parent.cpu_system_seconds) + 1e-9:
        raise ValueError("atomic publisher system CPU exceeds its execution parent")

    assert parent.peak_process_tree_rss_bytes is not None
    nested_peak = max(
        int(value["process_peak_rss_bytes"]),
        int(value["children_peak_rss_bytes"]),
    )
    if nested_peak > parent.peak_process_tree_rss_bytes:
        raise ValueError("atomic publisher peak RSS exceeds its execution parent")


def require_candidate_execution_receipt(
    value: Mapping[str, Any],
    *,
    expected_request_payload_digest: str,
) -> None:
    """Deeply validate one execution receipt and every nested evidence surface."""

    if not isinstance(value, Mapping) or set(value) != {
        "schema_id",
        "schema_version",
        "payload",
        "payload_digest",
    }:
        raise ValueError("execution receipt envelope field set differs")
    schema_version = value["schema_version"]
    if (
        value["schema_id"] != ANALYSIS_CANDIDATE_EXECUTION_RECEIPT_SCHEMA_ID
        or type(schema_version) is not int
        or schema_version
        not in {
            ANALYSIS_CANDIDATE_EXECUTION_RECEIPT_LEGACY_SCHEMA_VERSION,
            ANALYSIS_CANDIDATE_EXECUTION_RECEIPT_HIERARCHICAL_SCHEMA_VERSION,
            ANALYSIS_CANDIDATE_EXECUTION_RECEIPT_SCHEMA_VERSION,
        }
    ):
        raise ValueError("execution receipt schema identity differs")
    payload_fields = {
        "status",
        "request",
        "request_payload_digest",
        "fresh_process",
        "environment",
        "phases",
        "coordinate_evidence",
        "logical_equality",
        "metadata_equivalence",
        "physical_io",
        "output_storage",
        "nonmutation_evidence",
        "publication_gate_passed",
    }
    if schema_version == ANALYSIS_CANDIDATE_EXECUTION_RECEIPT_SCHEMA_VERSION:
        payload_fields.add("publication_runtime_telemetry")
    payload = _require_exact_fields(
        value["payload"],
        payload_fields,
        label="execution receipt payload",
    )
    if value["payload_digest"] != canonical_json_sha256(payload):
        raise ValueError("execution receipt payload digest differs")
    _json_copy(value)
    request = payload["request"]
    require_candidate_execution_request(request)
    _require_sha256(
        expected_request_payload_digest,
        label="expected_request_payload_digest",
    )
    if (
        payload["request_payload_digest"] != request["payload_digest"]
        or payload["request_payload_digest"] != expected_request_payload_digest
    ):
        raise ValueError("execution receipt request binding differs")
    adapter = _require_registered_adapter_manifest(
        request["payload"]["adapter_manifest"]
    )
    suite_payload = request["payload"]["benchmark_suite"]["payload"]
    planned_arrays = suite_payload["storage_plan_receipt"]["payload"]["arrays"]
    planned_array_count = len(planned_arrays)
    mode = CandidateComputationMode(adapter["computation_mode"])
    required_phases = required_execution_phases(mode)
    if schema_version == ANALYSIS_CANDIDATE_EXECUTION_RECEIPT_LEGACY_SCHEMA_VERSION:
        required_phases = tuple(
            phase
            for phase in required_phases
            if phase is not CandidateExecutionPhase.PUBLICATION_ACCEPTANCE_VALIDATION
        )
    phases = payload["phases"]
    if not isinstance(phases, list) or len(phases) != len(required_phases):
        raise ValueError("execution receipt phase count differs")
    outcomes: list[PhaseOutcome] = []
    measurements: list[CandidatePhaseMeasurement] = []
    for expected_phase, record in zip(required_phases, phases, strict=True):
        current_fields = set(
            CandidatePhaseMeasurement(
                phase=expected_phase,
                outcome=PhaseOutcome.NOT_APPLICABLE,
                started_at_utc=None,
                completed_at_utc=None,
                wall_seconds=None,
                cpu_user_seconds=None,
                cpu_system_seconds=None,
                peak_process_tree_rss_bytes=None,
                not_applicable_reason="schema probe",
            ).as_manifest()
        )
        expected_fields = (
            current_fields
            if schema_version
            in {
                ANALYSIS_CANDIDATE_EXECUTION_RECEIPT_HIERARCHICAL_SCHEMA_VERSION,
                ANALYSIS_CANDIDATE_EXECUTION_RECEIPT_SCHEMA_VERSION,
            }
            else current_fields - {"parent_phase"}
        )
        phase_record = _require_exact_fields(
            record, expected_fields, label="execution phase"
        )
        try:
            observed_phase = CandidateExecutionPhase(phase_record["phase"])
            outcome = PhaseOutcome(phase_record["outcome"])
        except (TypeError, ValueError) as exc:
            raise ValueError("execution phase enum differs") from exc
        if observed_phase is not expected_phase:
            raise ValueError("execution receipt phase order differs")
        parent_value = phase_record.get("parent_phase")
        try:
            parent_phase = (
                None
                if parent_value is None
                else CandidateExecutionPhase(parent_value)
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("execution phase parent enum differs") from exc
        measurement = CandidatePhaseMeasurement(
            phase=observed_phase,
            outcome=outcome,
            started_at_utc=phase_record["started_at_utc"],
            completed_at_utc=phase_record["completed_at_utc"],
            wall_seconds=phase_record["wall_seconds"],
            cpu_user_seconds=phase_record["cpu_user_seconds"],
            cpu_system_seconds=phase_record["cpu_system_seconds"],
            peak_process_tree_rss_bytes=phase_record["peak_process_tree_rss_bytes"],
            parent_phase=parent_phase,
            not_applicable_reason=phase_record["not_applicable_reason"],
            error_type=phase_record["error_type"],
            error_message=phase_record["error_message"],
        )
        if (
            schema_version
            in {
                ANALYSIS_CANDIDATE_EXECUTION_RECEIPT_HIERARCHICAL_SCHEMA_VERSION,
                ANALYSIS_CANDIDATE_EXECUTION_RECEIPT_SCHEMA_VERSION,
            }
            and phase_record["parent_phase"]
            != measurement.as_manifest()["parent_phase"]
        ):
            raise ValueError("execution phase parent differs from the canonical hierarchy")
        measurements.append(measurement)
        outcomes.append(outcome)

    if schema_version in {
        ANALYSIS_CANDIDATE_EXECUTION_RECEIPT_HIERARCHICAL_SCHEMA_VERSION,
        ANALYSIS_CANDIDATE_EXECUTION_RECEIPT_SCHEMA_VERSION,
    }:
        _require_current_phase_chronology(measurements)
    if schema_version == ANALYSIS_CANDIDATE_EXECUTION_RECEIPT_SCHEMA_VERSION:
        _require_atomic_publication_runtime_telemetry(
            payload["publication_runtime_telemetry"],
            phases=measurements,
        )

    fresh = _require_exact_fields(
        payload["fresh_process"],
        {"driver_pid", "child_pid", "child_start_time_ticks", "is_fresh"},
        label="fresh-process evidence",
    )
    for field in ("driver_pid", "child_pid", "child_start_time_ticks"):
        _require_nonnegative_int(fresh[field], label=field)
    if (
        fresh["driver_pid"] <= 0
        or fresh["child_pid"] <= 0
        or fresh["child_start_time_ticks"] <= 0
        or fresh["driver_pid"] == fresh["child_pid"]
        or fresh["is_fresh"] is not True
    ):
        raise ValueError("execution receipt does not prove one fresh child process")

    environment = _require_exact_fields(
        payload["environment"],
        {
            "hostname",
            "platform",
            "python_version",
            "python_executable",
            "palette_commit",
            "palette_git_dirty",
            "runner_ref",
            "runner_sha256",
            "cache_state",
        },
        label="execution environment",
    )
    for field in (
        "hostname",
        "platform",
        "python_version",
        "python_executable",
        "runner_ref",
        "cache_state",
    ):
        if type(environment[field]) is not str or not environment[field]:
            raise ValueError(f"execution environment {field} is invalid")
    if (
        type(environment["palette_commit"]) is not str
        or not _GIT_SHA.fullmatch(environment["palette_commit"])
        or environment["palette_commit"] != request["payload"]["palette_commit"]
        or type(environment["palette_git_dirty"]) is not bool
    ):
        raise ValueError("execution environment Git identity differs")
    _require_sha256(environment["runner_sha256"], label="runner_sha256")
    if environment["cache_state"] != request["payload"]["cache_state"]:
        raise ValueError("execution environment cache state differs")
    expected_runner_ref = f"{adapter['runner_module']}:{adapter['runner_entrypoint']}"
    if environment["runner_ref"] != expected_runner_ref:
        raise ValueError("execution environment runner identity differs")

    coordinate = _require_exact_fields(
        payload["coordinate_evidence"],
        {
            "role",
            "status",
            "source_authority_digests",
            "published_authority_sha256",
            "published_authority_ref",
            "temporal_axis_sha256",
            "temporal_axis_ref",
            "validator_ref",
            "validation_receipt_sha256",
            "coordinate_gate_passed",
        },
        label="coordinate evidence",
    )
    try:
        role = CoordinateContractRole(coordinate["role"])
        coordinate_status = CoordinateEvidenceStatus(coordinate["status"])
    except (TypeError, ValueError) as exc:
        raise ValueError("coordinate evidence enum differs") from exc
    if role.value != adapter["coordinate_role"]:
        raise ValueError("coordinate evidence role differs from its adapter")
    digests = coordinate["source_authority_digests"]
    if not isinstance(digests, list):
        raise ValueError("coordinate source authority digests must be one array")
    labels: list[str] = []
    for item in digests:
        record = _require_exact_fields(
            item, {"role", "sha256"}, label="coordinate source authority"
        )
        labels.append(_require_identifier(record["role"], label="authority role"))
        _require_sha256(record["sha256"], label="authority sha256")
    if labels != sorted(set(labels)):
        raise ValueError("coordinate source authorities must be unique and sorted")
    for field in ("published_authority_sha256", "temporal_axis_sha256"):
        observed = coordinate[field]
        if observed is not None:
            _require_sha256(observed, label=field)
    for field in ("published_authority_ref", "temporal_axis_ref"):
        observed = coordinate[field]
        if observed is not None and (type(observed) is not str or not observed):
            raise ValueError(f"{field} must be None or one nonempty reference")
    if (
        type(coordinate["validator_ref"]) is not str
        or ":" not in coordinate["validator_ref"]
    ):
        raise ValueError("coordinate validator_ref is invalid")
    _require_sha256(
        coordinate["validation_receipt_sha256"],
        label="coordinate validation_receipt_sha256",
    )
    if type(coordinate["coordinate_gate_passed"]) is not bool:
        raise ValueError("coordinate_gate_passed must be an exact bool")
    expected_coordinate_status = CoordinateContractStatus(
        adapter["coordinate_contract_status"]
    )
    expected_evidence = {
        CoordinateContractStatus.CANONICAL_PUBLICATION_IMPLEMENTED: (
            CoordinateEvidenceStatus.VERIFIED_CANONICAL_PUBLICATION
        ),
        CoordinateContractStatus.BOUND_SOURCE_VALIDATION_IMPLEMENTED: (
            CoordinateEvidenceStatus.VERIFIED_BOUND_SOURCE
        ),
        CoordinateContractStatus.TEMPORAL_AXIS_IMPLEMENTED: (
            CoordinateEvidenceStatus.VERIFIED_TEMPORAL_AXIS
        ),
        CoordinateContractStatus.SOURCE_PRESERVATION_ONLY: (
            CoordinateEvidenceStatus.VERIFIED_SOURCE_PRESERVATION_NONMINTING
        ),
        CoordinateContractStatus.BLOCKED_CANONICAL_BINDING: (
            CoordinateEvidenceStatus.BLOCKED
        ),
    }[expected_coordinate_status]
    if coordinate_status is not expected_evidence:
        raise ValueError("coordinate evidence status differs from its adapter")
    if coordinate_status is CoordinateEvidenceStatus.VERIFIED_CANONICAL_PUBLICATION:
        if (
            not digests
            or coordinate["published_authority_sha256"] is None
            or coordinate["published_authority_ref"] is None
            or coordinate["temporal_axis_sha256"] is not None
            or coordinate["temporal_axis_ref"] is not None
        ):
            raise ValueError("canonical coordinate publication evidence is incomplete")
    elif coordinate_status is CoordinateEvidenceStatus.VERIFIED_TEMPORAL_AXIS:
        if (
            coordinate["temporal_axis_sha256"] is None
            or coordinate["temporal_axis_ref"] is None
            or coordinate["published_authority_sha256"] is not None
            or coordinate["published_authority_ref"] is not None
        ):
            raise ValueError("temporal coordinate evidence lacks its axis digest")
    elif coordinate_status in {
        CoordinateEvidenceStatus.VERIFIED_BOUND_SOURCE,
        CoordinateEvidenceStatus.VERIFIED_SOURCE_PRESERVATION_NONMINTING,
    }:
        if (
            not digests
            or coordinate["published_authority_sha256"] is not None
            or coordinate["published_authority_ref"] is not None
            or coordinate["temporal_axis_sha256"] is not None
            or coordinate["temporal_axis_ref"] is not None
        ):
            raise ValueError(
                "coordinate dependency evidence must only bind source authorities"
            )
    elif coordinate_status is CoordinateEvidenceStatus.BLOCKED and (
        digests
        or coordinate["published_authority_sha256"] is not None
        or coordinate["published_authority_ref"] is not None
        or coordinate["temporal_axis_sha256"] is not None
        or coordinate["temporal_axis_ref"] is not None
    ):
        raise ValueError("blocked coordinate evidence must not claim an authority")
    coordinate_can_promote = coordinate_status not in {
        CoordinateEvidenceStatus.VERIFIED_SOURCE_PRESERVATION_NONMINTING,
        CoordinateEvidenceStatus.BLOCKED,
    }
    if coordinate["coordinate_gate_passed"] is not coordinate_can_promote:
        raise ValueError("coordinate gate result differs from its evidence status")

    equality = _require_exact_fields(
        payload["logical_equality"],
        {
            "contract_id",
            "compared_array_count",
            "source_logical_manifest_sha256",
            "candidate_logical_manifest_sha256",
            "equal",
        },
        label="logical equality",
    )
    if equality["contract_id"] != adapter["logical_equality_contract"]:
        raise ValueError("logical equality projection differs from its adapter")
    _require_nonnegative_int(
        equality["compared_array_count"], label="compared_array_count"
    )
    if equality["compared_array_count"] != planned_array_count:
        raise ValueError("logical equality array count differs from its storage plan")
    for field in (
        "source_logical_manifest_sha256",
        "candidate_logical_manifest_sha256",
    ):
        _require_sha256(equality[field], label=field)
    if (
        equality["equal"] is not True
        or equality["source_logical_manifest_sha256"]
        != equality["candidate_logical_manifest_sha256"]
        or equality["source_logical_manifest_sha256"]
        != request["payload"]["source_identity_sha256"]
    ):
        raise ValueError("execution candidate lacks exact decoded equality")

    metadata = _require_exact_fields(
        payload["metadata_equivalence"],
        {
            "local_array_count",
            "published_array_count",
            "local_equal",
            "published_equal",
        },
        label="metadata equivalence",
    )
    for field in ("local_array_count", "published_array_count"):
        _require_nonnegative_int(metadata[field], label=field)
        if metadata[field] != planned_array_count:
            raise ValueError("metadata array count differs from its storage plan")
    if metadata["local_equal"] is not True or metadata["published_equal"] is not True:
        raise ValueError("direct/consolidated metadata equivalence did not pass")

    physical = _require_exact_fields(
        payload["physical_io"],
        {
            "scope",
            "physical_io_measured",
            "read_bytes",
            "write_bytes",
            "read_operations",
            "write_operations",
            "measurement_ref",
            "measurement_sha256",
        },
        label="physical I/O evidence",
    )
    try:
        scope = PhysicalIOScope(physical["scope"])
    except (TypeError, ValueError) as exc:
        raise ValueError("physical I/O scope is unsupported") from exc
    if scope.value != request["payload"]["physical_io_scope"]:
        raise ValueError("physical I/O scope differs from its request")
    expected_measured = scope in TRANSFER_PHYSICAL_IO_SCOPES
    if physical["physical_io_measured"] is not expected_measured:
        raise ValueError("physical_io_measured differs from its measurement scope")
    counters = (
        "read_bytes",
        "write_bytes",
        "read_operations",
        "write_operations",
    )
    if scope is PhysicalIOScope.UNAVAILABLE:
        if any(physical[field] is not None for field in counters):
            raise ValueError("unavailable physical I/O must use null counters")
        if (
            physical["measurement_ref"] is not None
            or physical["measurement_sha256"] is not None
        ):
            raise ValueError("unavailable physical I/O must not claim a receipt")
    else:
        for field in counters:
            if physical[field] is not None:
                _require_nonnegative_int(physical[field], label=field)
        if not any(physical[field] is not None for field in counters):
            raise ValueError("available physical I/O lacks every counter")
        if (
            type(physical["measurement_ref"]) is not str
            or not physical["measurement_ref"]
        ):
            raise ValueError("measured I/O lacks a measurement receipt reference")
        _require_sha256(
            physical["measurement_sha256"], label="physical I/O measurement_sha256"
        )

    storage = _require_exact_fields(
        payload["output_storage"],
        {
            "file_count",
            "metadata_file_count",
            "payload_file_count",
            "apparent_bytes",
            "allocated_bytes",
        },
        label="output storage",
    )
    for field, observed in storage.items():
        _require_nonnegative_int(observed, label=field)
    if storage["file_count"] != (
        storage["metadata_file_count"] + storage["payload_file_count"]
    ):
        raise ValueError("output storage file counts do not add up")
    if (
        storage["file_count"] < 1
        or storage["metadata_file_count"] < planned_array_count
    ):
        raise ValueError("output storage does not contain the planned array metadata")
    if storage["apparent_bytes"] < 1 or storage["allocated_bytes"] < 1:
        raise ValueError("completed output storage must occupy positive bytes")

    nonmutation = _require_exact_fields(
        payload["nonmutation_evidence"],
        {
            "selector_before_sha256",
            "selector_after_sha256",
            "registry_before_sha256",
            "registry_after_sha256",
            "production_profiles_before_sha256",
            "production_profiles_after_sha256",
            "snapshot_contract_id",
            "unchanged",
        },
        label="nonmutation evidence",
    )
    pairs = (
        ("selector_before_sha256", "selector_after_sha256"),
        ("registry_before_sha256", "registry_after_sha256"),
        ("production_profiles_before_sha256", "production_profiles_after_sha256"),
    )
    for before, after in pairs:
        _require_sha256(nonmutation[before], label=before)
        _require_sha256(nonmutation[after], label=after)
        if nonmutation[before] != nonmutation[after]:
            raise ValueError("benchmark execution changed protected production state")
    protected = request["payload"]["protected_state_before"]
    if (
        nonmutation["snapshot_contract_id"] != protected["snapshot_contract_id"]
        or nonmutation["selector_before_sha256"] != protected["selector_sha256"]
        or nonmutation["registry_before_sha256"] != protected["registry_sha256"]
        or nonmutation["production_profiles_before_sha256"]
        != protected["production_profiles_sha256"]
    ):
        raise ValueError("nonmutation evidence differs from the requested pre-state")
    if nonmutation["unchanged"] is not True:
        raise ValueError("nonmutation evidence did not pass")
    _require_identifier(
        nonmutation["snapshot_contract_id"],
        label="nonmutation snapshot_contract_id",
    )

    status = payload["status"]
    if status != "complete":
        raise ValueError(
            "execution receipts represent completed publications only; "
            "failed attempts require a separate attempt record"
        )
    if any(outcome is not PhaseOutcome.SUCCEEDED for outcome in outcomes):
        raise ValueError("complete execution receipts require every phase to succeed")
    expected_publication_gate = (
        coordinate["coordinate_gate_passed"] is True
        and physical["physical_io_measured"] is True
        and environment["palette_git_dirty"] is False
    )
    if payload["publication_gate_passed"] is not expected_publication_gate:
        raise ValueError("publication gate result differs from execution evidence")


def build_candidate_execution_receipt(
    *,
    request: Mapping[str, Any],
    status: str,
    fresh_process: Mapping[str, Any],
    environment: Mapping[str, Any],
    phases: Sequence[CandidatePhaseMeasurement],
    publication_runtime_telemetry: Mapping[str, Any],
    coordinate_evidence: Mapping[str, Any],
    logical_equality: Mapping[str, Any],
    metadata_equivalence: Mapping[str, Any],
    physical_io: Mapping[str, Any],
    output_storage: Mapping[str, Any],
    nonmutation_evidence: Mapping[str, Any],
) -> dict[str, object]:
    """Build and validate one immutable execution receipt."""

    require_candidate_execution_request(request)
    if status != "complete":
        raise ValueError("execution receipts represent completed publications only")
    payload: dict[str, object] = {
        "status": status,
        "request": _json_copy(request),
        "request_payload_digest": request["payload_digest"],
        "fresh_process": _json_copy(fresh_process),
        "environment": _json_copy(environment),
        "phases": [phase.as_manifest() for phase in phases],
        "publication_runtime_telemetry": _json_copy(
            publication_runtime_telemetry
        ),
        "coordinate_evidence": _json_copy(coordinate_evidence),
        "logical_equality": _json_copy(logical_equality),
        "metadata_equivalence": _json_copy(metadata_equivalence),
        "physical_io": _json_copy(physical_io),
        "output_storage": _json_copy(output_storage),
        "nonmutation_evidence": _json_copy(nonmutation_evidence),
        "publication_gate_passed": False,
    }
    coordinate_passed = bool(payload["coordinate_evidence"]["coordinate_gate_passed"])
    io_scope = PhysicalIOScope(payload["physical_io"]["scope"])
    payload["publication_gate_passed"] = (
        status == "complete"
        and coordinate_passed
        and io_scope in TRANSFER_PHYSICAL_IO_SCOPES
        and payload["environment"]["palette_git_dirty"] is False
    )
    result = {
        "schema_id": ANALYSIS_CANDIDATE_EXECUTION_RECEIPT_SCHEMA_ID,
        "schema_version": ANALYSIS_CANDIDATE_EXECUTION_RECEIPT_SCHEMA_VERSION,
        "payload": payload,
        "payload_digest": canonical_json_sha256(payload),
    }
    require_candidate_execution_receipt(
        result,
        expected_request_payload_digest=str(request["payload_digest"]),
    )
    return result


__all__ = [
    "ANALYSIS_CANDIDATE_EXECUTION_ADAPTER_SCHEMA_ID",
    "ANALYSIS_CANDIDATE_EXECUTION_ADAPTER_SCHEMA_VERSION",
    "ANALYSIS_CANDIDATE_EXECUTION_RECEIPT_SCHEMA_ID",
    "ANALYSIS_CANDIDATE_EXECUTION_RECEIPT_HIERARCHICAL_SCHEMA_VERSION",
    "ANALYSIS_CANDIDATE_EXECUTION_RECEIPT_LEGACY_SCHEMA_VERSION",
    "ANALYSIS_CANDIDATE_EXECUTION_RECEIPT_SCHEMA_VERSION",
    "ANALYSIS_CANDIDATE_EXECUTION_REQUEST_SCHEMA_ID",
    "ANALYSIS_CANDIDATE_EXECUTION_REQUEST_SCHEMA_VERSION",
    "CandidateComputationMode",
    "CandidateExecutionPhase",
    "CandidateLogicalEqualityContract",
    "CandidatePhaseMeasurement",
    "CandidateRunnerStatus",
    "CoordinateContractRole",
    "CoordinateContractStatus",
    "CoordinateEvidenceStatus",
    "PhaseOutcome",
    "PhysicalIOScope",
    "TRANSFER_PHYSICAL_IO_SCOPES",
    "build_candidate_execution_receipt",
    "build_candidate_execution_request",
    "protected_path_snapshot_sha256",
    "require_candidate_execution_adapter_manifest",
    "require_candidate_execution_receipt",
    "require_candidate_execution_request",
    "required_execution_phases",
]
