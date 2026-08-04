"""Fresh-process typed execution for diagnostic track-flat candidates."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys
from typing import Any, Mapping, Optional, Sequence
import uuid

from fisheye.analysis.direct_writer_storage import (
    ANALYSIS_STORAGE_PLAN_RECEIPT_ATTR,
)
from fisheye.analysis.track_kinematics import load_bound_track_motion_run
from fisheye.analysis.track_kinematics_storage import (
    TRACK_KINEMATICS_FLAT_CANDIDATE_MANIFEST_DIGEST_ATTR,
    build_flat_candidate_declarations,
    build_flat_candidate_storage_receipt,
    source_flat_projection_hashes,
    validate_flat_candidate,
)
from fisheye.analysis_workflows.analysis_candidate_execution import (
    CandidateComputationMode,
    CandidatePhaseMeasurement,
    CoordinateEvidenceStatus,
    PhaseOutcome,
    PhysicalIOScope,
    build_candidate_execution_receipt,
    protected_path_snapshot_sha256,
    require_candidate_execution_receipt,
    require_candidate_execution_request,
    required_execution_phases,
)
from fisheye.analysis_workflows.materializers.atomic_run_publisher import (
    ATOMIC_PUBLICATION_TOMBSTONE_ATTR,
)
from fisheye.analysis_workflows.materializers.runtime_telemetry import (
    require_runtime_telemetry,
)
from fisheye.analysis_workflows.materializers.track_kinematics_candidate import (
    EXECUTION_BINDING_ATTR,
    EXECUTION_FAILURE_TOMBSTONE_ATTR,
    TRACK_FLAT_EXECUTION_PHASE_ORDER,
    materialize_track_kinematics_flat_candidate,
    tombstone_track_kinematics_execution_candidate,
)
from fisheye.analysis_workflows.track_kinematics_candidate_suite import (
    require_track_flat_execution_suite,
)
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from fisheye.shared.zarr.storage_profiles import get_storage_profile
from fisheye.shared.zarr_io import open_zarr_root
from fisheye.shared.zarr_run_completion import (
    RUN_COMPLETION_STATUS_ATTR,
    RUN_STATUS_COMPLETE,
)

ATTEMPT_SCHEMA_ID = "palette.analysis_candidate_execution_attempt"
ATTEMPT_SCHEMA_VERSION = 2
RUNNER_REF = (
    "fisheye.diagnostics.track_kinematics_candidate_execution:"
    "execute_track_flat_candidate"
)
PARENT_PATH = "analysis/track_kinematics_runs"
OFFLINE_PATH = f"{PARENT_PATH}/offline"
_SELECTOR_SNAPSHOT_SCHEMA_ID = "palette.track_candidate_selector_snapshot"
_SELECTOR_SNAPSHOT_SCHEMA_VERSION = 1
_FAILURE_PHASES = frozenset(
    {
        "runner_preflight",
        "materializer",
        "runner_receipt_assembly",
        "driver_child_evidence",
        "driver_protected_poststate",
        "driver_receipt_publication",
        "driver_child_failure",
    }
)


class TrackFlatCandidateExecutionFailed(RuntimeError):
    """Raised with a self-validating terminal failed-attempt record."""

    def __init__(self, message: str, *, attempt: Mapping[str, Any]) -> None:
        super().__init__(message)
        self.attempt = dict(attempt)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _strict_json_copy(value: object) -> Any:
    return json.loads(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    )


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _group_attrs(group: Any) -> dict[str, Any]:
    attrs = group.attrs
    return dict(attrs.asdict() if hasattr(attrs, "asdict") else dict(attrs))


def _group_at_path(root: Any, path: str) -> Any:
    node = root
    for component in path.split("/"):
        node = node[component]
    return node


def _selector_snapshot_from_root(root: Any) -> dict[str, Any]:
    parent = _group_at_path(root, PARENT_PATH)
    offline = parent["offline"]
    return {
        "schema_id": _SELECTOR_SNAPSHOT_SCHEMA_ID,
        "schema_version": _SELECTOR_SNAPSHOT_SCHEMA_VERSION,
        "parents": {
            PARENT_PATH: _strict_json_copy(_group_attrs(parent)),
            OFFLINE_PATH: _strict_json_copy(_group_attrs(offline)),
        },
    }


def track_flat_selector_snapshot_sha256(archive_path: str | Path) -> str:
    """Digest both selector-owning track parent attribute surfaces."""

    root = open_zarr_root(Path(archive_path).expanduser().resolve(), mode="r")
    return canonical_json_sha256(_selector_snapshot_from_root(root))


def _execution_binding(request: Mapping[str, Any]) -> dict[str, Any]:
    payload = request["payload"]
    return {
        "schema_id": "palette.analysis_candidate_execution_binding",
        "schema_version": 1,
        "execution_id": payload["execution_id"],
        "request_payload_digest": request["payload_digest"],
        "candidate_run_path": payload["candidate_run_path"],
    }


def _proc_io() -> dict[str, int]:
    try:
        lines = Path("/proc/self/io").read_text(encoding="utf-8").splitlines()
    except OSError:
        return {}
    result: dict[str, int] = {}
    for line in lines:
        key, separator, raw = line.partition(":")
        if not separator:
            continue
        try:
            result[key.strip()] = int(raw.strip())
        except ValueError:
            continue
    return result


def _nonnegative_delta(
    after: Mapping[str, int], before: Mapping[str, int], key: str
) -> int | None:
    if key not in before or key not in after:
        return None
    return max(0, int(after[key]) - int(before[key]))


def _process_start_time_ticks(pid: int | None = None) -> int:
    stat_path = Path("/proc/self/stat") if pid is None else Path(f"/proc/{pid}/stat")
    text = stat_path.read_text(encoding="utf-8")
    closing = text.rfind(")")
    if closing < 0:
        raise RuntimeError("Could not parse /proc process command boundary.")
    value = int(text[closing + 2 :].split()[19])
    if value <= 0:
        raise RuntimeError("Process start-time ticks are not positive.")
    return value


def _git_identity() -> tuple[str, bool]:
    repository = Path(__file__).resolve().parents[3]
    commit = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "-C", str(repository), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    return commit, dirty


def _runner_sha256() -> str:
    digest = hashlib.sha256()
    with Path(__file__).open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _source_preflight(
    root: Any,
    source_path: str,
    *,
    expected_motion_authority_sha256: str,
    storage_profile_id: str,
) -> dict[str, Any]:
    if not source_path.startswith(f"{OFFLINE_PATH}/"):
        raise ValueError("Track-flat source must be one explicit offline run.")
    source = _group_at_path(root, source_path)
    if (
        source.attrs.get("schema_id") != "analysis.track_kinematics_runs"
        or source.attrs.get("schema_version") != 1
        or source.attrs.get(RUN_COMPLETION_STATUS_ATTR) != RUN_STATUS_COMPLETE
        or source.attrs.get("stage_selector_eligible") is not True
    ):
        raise ValueError("Track-flat source is not one completed eligible v1 run.")
    bound = load_bound_track_motion_run(root, source)
    if bound.manifest_sha256 != expected_motion_authority_sha256:
        raise ValueError(
            "Live track full-motion authority differs from the invocation digest."
        )
    declarations = build_flat_candidate_declarations(source)
    if any(declaration.path.endswith("/positions_mm") for declaration in declarations):
        raise ValueError("track_flat_v1 excludes the physical track bundle.")
    hashes = source_flat_projection_hashes(source, declarations)
    storage_receipt = build_flat_candidate_storage_receipt(
        source,
        profile=get_storage_profile(storage_profile_id),
    ).as_manifest()
    return {
        "source_group": source,
        "motion_authority_sha256": bound.manifest_sha256,
        "declarations": declarations,
        "logical_hashes": hashes,
        "logical_manifest_sha256": canonical_json_sha256(hashes),
        "storage_receipt": storage_receipt,
    }


def validate_track_flat_source_preservation(
    *,
    source_run_path: str,
    source_motion_authority_sha256: str,
    source_logical_manifest_sha256: str,
    candidate_logical_manifest_sha256: str,
    candidate_manifest_sha256: str,
) -> dict[str, Any]:
    for label, digest in (
        ("source motion authority", source_motion_authority_sha256),
        ("source logical manifest", source_logical_manifest_sha256),
        ("candidate logical manifest", candidate_logical_manifest_sha256),
        ("candidate manifest", candidate_manifest_sha256),
    ):
        if not _is_sha256(digest):
            raise ValueError(f"Track-flat {label} digest is invalid.")
    if source_logical_manifest_sha256 != candidate_logical_manifest_sha256:
        raise ValueError("Track-flat source and candidate logical identities differ.")
    if not source_run_path.startswith(f"{OFFLINE_PATH}/"):
        raise ValueError("Track-flat source-preservation run is not offline.")
    validation_payload = {
        "schema_id": "palette.track_flat_source_preservation_validation",
        "schema_version": 1,
        "source_run_ref": f"/{source_run_path}",
        "source_motion_authority_sha256": source_motion_authority_sha256,
        "source_logical_manifest_sha256": source_logical_manifest_sha256,
        "candidate_logical_manifest_sha256": candidate_logical_manifest_sha256,
        "candidate_manifest_sha256": candidate_manifest_sha256,
        "physical_bundle_mode": "excluded_from_flat_candidate_v1",
        "authority_minting": "forbidden_diagnostic_projection",
    }
    return {
        "role": "canonical_producer",
        "status": (
            CoordinateEvidenceStatus.VERIFIED_SOURCE_PRESERVATION_NONMINTING.value
        ),
        "source_authority_digests": [
            {
                "role": "source_track_motion_manifest",
                "sha256": source_motion_authority_sha256,
            }
        ],
        "published_authority_sha256": None,
        "published_authority_ref": None,
        "temporal_axis_sha256": None,
        "temporal_axis_ref": None,
        "validator_ref": f"{__name__}:validate_track_flat_source_preservation",
        "validation_receipt_sha256": canonical_json_sha256(validation_payload),
        "coordinate_gate_passed": False,
    }


def _translate_phases(
    runtime: Mapping[str, Any],
) -> list[CandidatePhaseMeasurement]:
    require_runtime_telemetry(
        runtime,
        expected_materializer="track_kinematics_flat_candidate",
        allowed_phase_order=TRACK_FLAT_EXECUTION_PHASE_ORDER,
        require_error_phase=False,
    )
    records = runtime.get("phases")
    expected = required_execution_phases(
        CandidateComputationMode.LOGICAL_REMATERIALIZATION
    )
    if not isinstance(records, list) or [record.get("name") for record in records] != [
        phase.value for phase in expected
    ]:
        raise ValueError("Track-flat materializer phase sequence differs.")
    result: list[CandidatePhaseMeasurement] = []
    for phase, record in zip(expected, records, strict=True):
        if record.get("outcome") != "ok":
            raise ValueError("A successful track-flat materializer has a failed phase.")
        cpu = record.get("cpu_seconds")
        if not isinstance(cpu, Mapping):
            raise ValueError("Track-flat phase CPU telemetry is invalid.")
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


def _physical_io_evidence(
    *,
    request: Mapping[str, Any],
    before: Mapping[str, int],
    after: Mapping[str, int],
) -> dict[str, Any]:
    scope = PhysicalIOScope(request["payload"]["physical_io_scope"])
    if scope is PhysicalIOScope.UNAVAILABLE:
        return {
            "scope": scope.value,
            "physical_io_measured": False,
            "read_bytes": None,
            "write_bytes": None,
            "read_operations": None,
            "write_operations": None,
            "measurement_ref": None,
            "measurement_sha256": None,
        }
    if scope is not PhysicalIOScope.PROCESS_SELF_PROC_IO:
        raise ValueError(
            "Track direct-child execution supports only unavailable or process-self "
            "I/O; transfer evidence requires an external tracer."
        )
    measurement = {
        "schema_id": "palette.process_self_proc_io_measurement",
        "schema_version": 1,
        "request_payload_digest": request["payload_digest"],
        "before": dict(before),
        "after": dict(after),
    }
    return {
        "scope": scope.value,
        "physical_io_measured": False,
        "read_bytes": _nonnegative_delta(after, before, "read_bytes"),
        "write_bytes": _nonnegative_delta(after, before, "write_bytes"),
        "read_operations": _nonnegative_delta(after, before, "syscr"),
        "write_operations": _nonnegative_delta(after, before, "syscw"),
        "measurement_ref": "inline:/proc/self/io",
        "measurement_sha256": canonical_json_sha256(measurement),
    }


def _target_state(archive: Path, candidate_path: str) -> dict[str, Any]:
    target = archive.joinpath(*candidate_path.split("/"))
    if not target.exists():
        return {
            "exists": False,
            "completion_status": None,
            "selector_eligible": None,
            "atomic_tombstone_present": False,
            "inspection_error": None,
        }
    try:
        attrs = _group_attrs(open_zarr_root(target, mode="r"))
    except Exception as exc:
        return {
            "exists": True,
            "completion_status": None,
            "selector_eligible": None,
            "atomic_tombstone_present": False,
            "inspection_error": type(exc).__name__,
        }
    return {
        "exists": True,
        "completion_status": attrs.get(RUN_COMPLETION_STATUS_ATTR),
        "selector_eligible": attrs.get("stage_selector_eligible"),
        "atomic_tombstone_present": (
            ATOMIC_PUBLICATION_TOMBSTONE_ATTR in attrs
            or EXECUTION_FAILURE_TOMBSTONE_ATTR in attrs
        ),
        "inspection_error": None,
    }


def _build_failed_attempt(
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
    payload = request["payload"]
    protected = payload["protected_state_before"]
    target_state = _target_state(
        Path(payload["archive_path"]),
        str(payload["candidate_run_path"]),
    )
    if target_state.get("selector_eligible") is True:
        raise RuntimeError(
            "Failed track-flat execution left a selector-eligible target."
        ) from error
    nonmutation = {
        "selector_before_sha256": protected["selector_sha256"],
        "selector_after_sha256": selector_after_sha256,
        "registry_before_sha256": protected["registry_sha256"],
        "registry_after_sha256": registry_after_sha256,
        "production_profiles_before_sha256": protected["production_profiles_sha256"],
        "production_profiles_after_sha256": production_profiles_after_sha256,
        "snapshot_contract_id": protected["snapshot_contract_id"],
        "observation_errors": dict(observation_errors or {}),
    }
    nonmutation["unchanged"] = bool(
        not nonmutation["observation_errors"]
        and selector_after_sha256 is not None
        and registry_after_sha256 is not None
        and production_profiles_after_sha256 is not None
        and nonmutation["selector_before_sha256"]
        == nonmutation["selector_after_sha256"]
        and nonmutation["registry_before_sha256"]
        == nonmutation["registry_after_sha256"]
        and nonmutation["production_profiles_before_sha256"]
        == nonmutation["production_profiles_after_sha256"]
    )
    attempt_payload = {
        "status": "failed",
        "request": _strict_json_copy(request),
        "request_payload_digest": request["payload_digest"],
        "driver_pid": int(driver_pid),
        "child_pid": int(child_pid),
        "child_start_time_ticks": int(child_start_time_ticks),
        "failed_at_utc": _utc_now(),
        "failure_phase": failure_phase,
        "error_type": reported_error_type or type(error).__name__,
        "error_message": reported_error_message or str(error),
        "runtime_telemetry": (
            None if runtime_telemetry is None else _strict_json_copy(runtime_telemetry)
        ),
        "target_state": target_state,
        "nonmutation_evidence": nonmutation,
    }
    result = {
        "schema_id": ATTEMPT_SCHEMA_ID,
        "schema_version": ATTEMPT_SCHEMA_VERSION,
        "payload": attempt_payload,
        "payload_digest": canonical_json_sha256(attempt_payload),
    }
    require_track_flat_execution_attempt(
        result,
        expected_request_payload_digest=str(request["payload_digest"]),
    )
    return result


def require_track_flat_execution_attempt(
    value: Mapping[str, Any],
    *,
    expected_request_payload_digest: str,
) -> None:
    """Deeply validate the shared-v2 terminal failure envelope."""

    if not isinstance(value, Mapping) or set(value) != {
        "schema_id",
        "schema_version",
        "payload",
        "payload_digest",
    }:
        raise ValueError("Track execution-attempt envelope field set differs.")
    if (
        value["schema_id"] != ATTEMPT_SCHEMA_ID
        or type(value["schema_version"]) is not int
        or value["schema_version"] != ATTEMPT_SCHEMA_VERSION
    ):
        raise ValueError("Track execution-attempt schema identity differs.")
    payload = value["payload"]
    expected = {
        "status",
        "request",
        "request_payload_digest",
        "driver_pid",
        "child_pid",
        "child_start_time_ticks",
        "failed_at_utc",
        "failure_phase",
        "error_type",
        "error_message",
        "runtime_telemetry",
        "target_state",
        "nonmutation_evidence",
    }
    if not isinstance(payload, Mapping) or set(payload) != expected:
        raise ValueError("Track execution-attempt payload field set differs.")
    if value["payload_digest"] != canonical_json_sha256(payload):
        raise ValueError("Track execution-attempt payload digest differs.")
    _strict_json_copy(value)
    require_candidate_execution_request(payload["request"])
    if (
        payload["status"] != "failed"
        or payload["request_payload_digest"] != payload["request"]["payload_digest"]
        or payload["request_payload_digest"] != expected_request_payload_digest
        or not _is_sha256(expected_request_payload_digest)
    ):
        raise ValueError("Track execution-attempt request binding differs.")
    for field in ("driver_pid", "child_pid", "child_start_time_ticks"):
        if type(payload[field]) is not int or payload[field] <= 0:
            raise ValueError(f"Track execution-attempt {field} is invalid.")
    if payload["driver_pid"] == payload["child_pid"]:
        raise ValueError("Track execution-attempt does not identify a fresh child.")
    try:
        failed_at = datetime.fromisoformat(payload["failed_at_utc"])
    except (TypeError, ValueError) as exc:
        raise ValueError("Track execution-attempt timestamp is invalid.") from exc
    if failed_at.tzinfo is None or failed_at.utcoffset() != timezone.utc.utcoffset(
        failed_at
    ):
        raise ValueError("Track execution-attempt timestamp is not UTC.")
    if (
        payload["failure_phase"] not in _FAILURE_PHASES
        or type(payload["error_type"]) is not str
        or not payload["error_type"]
        or type(payload["error_message"]) is not str
        or not payload["error_message"]
    ):
        raise ValueError("Track execution-attempt failure identity is incomplete.")

    target = payload["target_state"]
    if not isinstance(target, Mapping) or set(target) != {
        "exists",
        "completion_status",
        "selector_eligible",
        "atomic_tombstone_present",
        "inspection_error",
    }:
        raise ValueError("Track execution-attempt target state differs.")
    if (
        type(target["exists"]) is not bool
        or type(target["atomic_tombstone_present"]) is not bool
        or target["selector_eligible"] is True
    ):
        raise ValueError("Failed track execution target is selector eligible.")
    if (
        target["selector_eligible"] is not None
        and target["selector_eligible"] is not False
    ):
        raise ValueError("Track execution-attempt eligibility is not exact.")
    for field in ("completion_status", "inspection_error"):
        if target[field] is not None and (
            type(target[field]) is not str or not target[field]
        ):
            raise ValueError(f"Track execution-attempt target {field} is invalid.")
    if target["atomic_tombstone_present"] and (
        not target["exists"]
        or target["completion_status"] != "failed"
        or target["selector_eligible"] is not False
        or target["inspection_error"] is not None
    ):
        raise ValueError("Track execution-attempt tombstone state is inconsistent.")
    if not target["exists"] and any(
        (
            target["completion_status"] is not None,
            target["selector_eligible"] is not None,
            target["atomic_tombstone_present"],
            target["inspection_error"] is not None,
        )
    ):
        raise ValueError("Absent track execution target claims persisted state.")

    runtime = payload["runtime_telemetry"]
    if runtime is not None:
        require_runtime_telemetry(
            runtime,
            expected_materializer="track_kinematics_flat_candidate",
            allowed_phase_order=TRACK_FLAT_EXECUTION_PHASE_ORDER,
            require_error_phase=payload["failure_phase"] == "materializer",
        )
    nonmutation = payload["nonmutation_evidence"]
    if not isinstance(nonmutation, Mapping) or set(nonmutation) != {
        "selector_before_sha256",
        "selector_after_sha256",
        "registry_before_sha256",
        "registry_after_sha256",
        "production_profiles_before_sha256",
        "production_profiles_after_sha256",
        "snapshot_contract_id",
        "observation_errors",
        "unchanged",
    }:
        raise ValueError("Track execution-attempt nonmutation evidence differs.")
    protected = payload["request"]["payload"]["protected_state_before"]
    if (
        nonmutation["selector_before_sha256"] != protected["selector_sha256"]
        or nonmutation["registry_before_sha256"] != protected["registry_sha256"]
        or nonmutation["production_profiles_before_sha256"]
        != protected["production_profiles_sha256"]
        or nonmutation["snapshot_contract_id"] != protected["snapshot_contract_id"]
    ):
        raise ValueError("Track execution-attempt protected pre-state differs.")
    errors = nonmutation["observation_errors"]
    if not isinstance(errors, Mapping) or any(
        field
        not in {
            "selector_after_sha256",
            "registry_after_sha256",
            "production_profiles_after_sha256",
        }
        or type(message) is not str
        or not message
        for field, message in errors.items()
    ):
        raise ValueError("Track execution-attempt observation errors are invalid.")
    for field in (
        "selector_after_sha256",
        "registry_after_sha256",
        "production_profiles_after_sha256",
    ):
        observed = nonmutation[field]
        if observed is None:
            if field not in errors:
                raise ValueError(f"Missing track {field} lacks an observation error.")
        elif not _is_sha256(observed) or field in errors:
            raise ValueError(f"Track execution-attempt {field} is invalid.")
    expected_unchanged = bool(
        not errors
        and nonmutation["selector_before_sha256"]
        == nonmutation["selector_after_sha256"]
        and nonmutation["registry_before_sha256"]
        == nonmutation["registry_after_sha256"]
        and nonmutation["production_profiles_before_sha256"]
        == nonmutation["production_profiles_after_sha256"]
    )
    if nonmutation["unchanged"] is not expected_unchanged:
        raise ValueError("Track execution-attempt nonmutation result differs.")


def execute_track_flat_candidate(
    request: Mapping[str, Any],
    *,
    driver_pid: int,
) -> dict[str, Any]:
    """Execute one exact diagnostic track-flat publication in a direct child."""

    require_candidate_execution_request(request)
    payload = request["payload"]
    adapter = payload["adapter_manifest"]["payload"]
    invocation = payload["invocation"]["payload"]
    if adapter["stage_id"] != "track_kinematics":
        raise ValueError("Track-flat runner owns only track_kinematics.")
    if adapter["computation_mode"] != "logical_rematerialization":
        raise ValueError("Track-flat runner requires logical rematerialization.")
    if adapter["coordinate_contract_status"] != "source_preservation_only":
        raise ValueError("Track-flat runner is not a canonical authority minter.")
    if invocation["contract_id"] != "track_flat_v1":
        raise ValueError("Track-flat runner requires track_flat_v1 invocation.")
    require_track_flat_execution_suite("track_kinematics", payload["benchmark_suite"])
    parameters = invocation["parameters"]
    if (
        parameters["source_schema_id"] != "analysis.track_kinematics_runs"
        or parameters["source_schema_version"] != 1
        or parameters["source_run_type"] != "offline"
        or parameters["physical_bundle_mode"] != "excluded_from_flat_candidate_v1"
    ):
        raise ValueError("Track-flat invocation contract differs.")
    if type(driver_pid) is not int or driver_pid <= 0 or driver_pid != os.getppid():
        raise ValueError("Track-flat execution must be a direct child of its driver.")
    scope = PhysicalIOScope(payload["physical_io_scope"])
    if scope not in {
        PhysicalIOScope.UNAVAILABLE,
        PhysicalIOScope.PROCESS_SELF_PROC_IO,
    }:
        raise ValueError("External transfer scopes require a traced-process driver.")

    child_start = _process_start_time_ticks()
    archive = Path(payload["archive_path"])
    source_path = str(payload["source_run_path"])
    candidate_path = str(payload["candidate_run_path"])
    source_name = source_path.rsplit("/", 1)[1]
    candidate_name = candidate_path.rsplit("/", 1)[1]
    protected = payload["protected_state_before"]
    selector_before = str(protected["selector_sha256"])
    execution_binding = _execution_binding(request)
    storage_profile_id = str(parameters["storage_profile_id"])
    copy_backend = str(parameters["copy_backend"])
    keep_scratch = bool(parameters["keep_scratch"])
    motion_digest = str(parameters["source_motion_authority_sha256"])
    io_before = _proc_io()
    failure_phase = "runner_preflight"
    runtime_telemetry: Mapping[str, Any] | None = None
    try:
        live_selector_before = track_flat_selector_snapshot_sha256(archive)
        if live_selector_before != selector_before:
            raise ValueError("Live track selector pre-state differs from the request.")
        commit, dirty = _git_identity()
        if commit != payload["palette_commit"]:
            raise ValueError("Track runner Git commit differs from the request.")
        root = open_zarr_root(archive, mode="r")
        source = _source_preflight(
            root,
            source_path,
            expected_motion_authority_sha256=motion_digest,
            storage_profile_id=storage_profile_id,
        )
        source_hashes = source["logical_hashes"]
        source_identity = source["logical_manifest_sha256"]
        if source_identity != payload["source_identity_sha256"]:
            raise ValueError("Track source logical identity differs from the request.")
        planned_receipt = payload["benchmark_suite"]["payload"]["storage_plan_receipt"]
        if source["storage_receipt"] != planned_receipt:
            raise ValueError("Live track storage plan differs from the suite.")

        def accept_published_candidate(
            published_root: Any,
            _published_offline: Any,
            candidate_group: Any,
        ) -> Mapping[str, Any]:
            if candidate_group.attrs.get(EXECUTION_BINDING_ATTR) != execution_binding:
                raise ValueError("Published track execution binding differs.")
            if candidate_group.attrs.get("physical_bundle_mode") != (
                "excluded_from_flat_candidate_v1"
            ):
                raise ValueError("Published track physical-bundle policy differs.")
            persisted_receipt = candidate_group.attrs.get(
                ANALYSIS_STORAGE_PLAN_RECEIPT_ATTR
            )
            if persisted_receipt != planned_receipt:
                raise ValueError("Published track storage receipt differs from suite.")
            live = _source_preflight(
                published_root,
                source_path,
                expected_motion_authority_sha256=motion_digest,
                storage_profile_id=storage_profile_id,
            )
            if (
                live["logical_hashes"] != source_hashes
                or live["logical_manifest_sha256"] != source_identity
                or live["storage_receipt"] != planned_receipt
            ):
                raise ValueError("Track source changed during candidate publication.")
            validation = validate_flat_candidate(
                candidate_group,
                source_group=live["source_group"],
            )
            if not validation["valid"]:
                raise ValueError(
                    f"Published track candidate validation failed: {validation}."
                )
            candidate_identity = canonical_json_sha256(validation["logical_hashes"])
            if candidate_identity != source_identity:
                raise ValueError("Published track candidate logical identity differs.")
            selector_after = canonical_json_sha256(
                _selector_snapshot_from_root(published_root)
            )
            if selector_after != selector_before:
                raise ValueError("Track candidate changed root or offline selectors.")
            candidate_manifest_sha256 = candidate_group.attrs.get(
                TRACK_KINEMATICS_FLAT_CANDIDATE_MANIFEST_DIGEST_ATTR
            )
            if not _is_sha256(candidate_manifest_sha256):
                raise ValueError("Track candidate manifest digest is absent.")
            coordinate = validate_track_flat_source_preservation(
                source_run_path=source_path,
                source_motion_authority_sha256=motion_digest,
                source_logical_manifest_sha256=source_identity,
                candidate_logical_manifest_sha256=candidate_identity,
                candidate_manifest_sha256=str(candidate_manifest_sha256),
            )
            return {
                "coordinate_evidence": coordinate,
                "selector_after_sha256": selector_after,
                "execution_binding": execution_binding,
                "storage_plan_receipt_sha256": canonical_json_sha256(persisted_receipt),
            }

        failure_phase = "materializer"
        materialized = materialize_track_kinematics_flat_candidate(
            archive,
            source_run=source_name,
            run_name=candidate_name,
            scratch_root=payload["scratch_root"],
            profile_id=storage_profile_id,
            copy_backend=copy_backend,
            apply=True,
            keep_scratch=keep_scratch,
            stage_source_to_scratch=True,
            exclude_physical_bundle=True,
            execution_binding=execution_binding,
            publication_acceptance_validator=accept_published_candidate,
        )
        runtime_telemetry = materialized["runtime_telemetry"]
        phases = _translate_phases(runtime_telemetry)

        failure_phase = "runner_receipt_assembly"
        caller_acceptance = materialized.get("caller_acceptance")
        if not isinstance(caller_acceptance, Mapping):
            raise ValueError("Atomic track publication omitted runner acceptance.")
        if caller_acceptance.get("execution_binding") != execution_binding:
            raise ValueError("Atomic track acceptance binding differs.")
        coordinate = caller_acceptance.get("coordinate_evidence")
        if not isinstance(coordinate, Mapping):
            raise ValueError("Atomic track acceptance omitted coordinate evidence.")
        if coordinate.get("coordinate_gate_passed") is not False:
            raise ValueError("Diagnostic track projection cannot pass coordinate gate.")
        selector_after = caller_acceptance.get("selector_after_sha256")
        if selector_after != selector_before:
            raise ValueError("Atomic track selector receipt differs.")
        io_after = _proc_io()
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
                "runner_sha256": _runner_sha256(),
                "cache_state": payload["cache_state"],
            },
            phases=phases,
            publication_runtime_telemetry=materialized["publication"][
                "runtime_telemetry"
            ],
            coordinate_evidence=coordinate,
            logical_equality={
                "contract_id": adapter["logical_equality_contract"],
                "compared_array_count": int(
                    materialized["published_validation"]["array_count"]
                ),
                "source_logical_manifest_sha256": materialized[
                    "source_logical_manifest_sha256"
                ],
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
                after=io_after,
            ),
            output_storage=materialized["output_storage"],
            nonmutation_evidence={
                "selector_before_sha256": selector_before,
                "selector_after_sha256": selector_after,
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
        if receipt["payload"]["publication_gate_passed"] is not False:
            raise ValueError("Diagnostic track-flat receipt unexpectedly passed gate.")
        return receipt
    except BaseException as exc:
        attached = getattr(exc, "palette_runtime_telemetry", None)
        if isinstance(attached, Mapping):
            runtime_telemetry = attached
        target = _target_state(archive, candidate_path)
        if target.get("completion_status") == RUN_STATUS_COMPLETE:
            tombstone_track_kinematics_execution_candidate(
                archive,
                run_name=candidate_name,
                expected_execution_binding=execution_binding,
                failure_phase=failure_phase,
                error_type=type(exc).__name__,
                error_message=str(exc),
            )
        selector_after = track_flat_selector_snapshot_sha256(archive)
        attempt = _build_failed_attempt(
            request=request,
            driver_pid=driver_pid,
            child_start_time_ticks=child_start,
            child_pid=os.getpid(),
            error=exc,
            failure_phase=failure_phase,
            runtime_telemetry=runtime_telemetry,
            selector_after_sha256=selector_after,
            registry_after_sha256=str(protected["registry_sha256"]),
            production_profiles_after_sha256=str(
                protected["production_profiles_sha256"]
            ),
        )
        raise TrackFlatCandidateExecutionFailed(
            f"Track-flat execution failed during {failure_phase}: {exc}",
            attempt=attempt,
        ) from exc


def _write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(
                value,
                stream,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected one JSON object at {path}.")
    return value


def run_track_flat_candidate_fresh_process(
    request_path: str | Path,
    *,
    receipt_path: str | Path,
    attempt_path: str | Path,
) -> dict[str, Any]:
    """Run once, then verify protected post-state before exposing evidence."""

    request_file = Path(request_path).expanduser().resolve()
    receipt_file = Path(receipt_path).expanduser().resolve()
    attempt_file = Path(attempt_path).expanduser().resolve()
    if receipt_file == attempt_file:
        raise ValueError("Receipt and failed-attempt paths must differ.")
    if receipt_file.exists() or attempt_file.exists():
        raise FileExistsError("Track fresh-process evidence paths already exist.")
    request = _read_json(request_file)
    require_candidate_execution_request(request)
    payload = request["payload"]
    adapter = payload["adapter_manifest"]["payload"]
    if adapter["stage_id"] != "track_kinematics":
        raise ValueError("Track fresh-process driver received another family.")
    archive = Path(payload["archive_path"])
    protected = payload["protected_state_before"]
    for path in (receipt_file, attempt_file):
        if ".palette_benchmarks" not in path.parts or path.is_relative_to(archive):
            raise ValueError(
                "Track execution evidence must be a benchmark sidecar outside Zarr."
            )
    if (
        protected_path_snapshot_sha256(protected["registry_probe_path"])
        != protected["registry_sha256"]
        or protected_path_snapshot_sha256(protected["production_profiles_probe_path"])
        != protected["production_profiles_sha256"]
        or track_flat_selector_snapshot_sha256(archive) != protected["selector_sha256"]
    ):
        raise ValueError("Track protected pre-state differs from its request.")

    hidden_receipt = receipt_file.with_name(f".{receipt_file.name}.child.{os.getpid()}")
    hidden_attempt = attempt_file.with_name(f".{attempt_file.name}.child.{os.getpid()}")
    if hidden_receipt.exists() or hidden_attempt.exists():
        raise FileExistsError("Track hidden evidence path already exists.")
    command = [
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
    ]
    process = subprocess.Popen(
        command,
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

    after_values: dict[str, str | None] = {}
    observation_errors: dict[str, str] = {}
    for field, observer in (
        (
            "selector_after_sha256",
            lambda: track_flat_selector_snapshot_sha256(archive),
        ),
        (
            "registry_after_sha256",
            lambda: protected_path_snapshot_sha256(protected["registry_probe_path"]),
        ),
        (
            "production_profiles_after_sha256",
            lambda: protected_path_snapshot_sha256(
                protected["production_profiles_probe_path"]
            ),
        ),
    ):
        try:
            after_values[field] = observer()
        except BaseException as exc:
            after_values[field] = None
            observation_errors[field] = f"{type(exc).__name__}: {exc}"
    unchanged = bool(
        not observation_errors
        and after_values["selector_after_sha256"] == protected["selector_sha256"]
        and after_values["registry_after_sha256"] == protected["registry_sha256"]
        and after_values["production_profiles_after_sha256"]
        == protected["production_profiles_sha256"]
    )

    def fail_driver(
        error: BaseException,
        *,
        phase: str,
        runtime: Mapping[str, Any] | None = None,
        reported_error_type: str | None = None,
        reported_error_message: str | None = None,
    ) -> dict[str, Any]:
        target = _target_state(archive, str(payload["candidate_run_path"]))
        if target.get("completion_status") == RUN_STATUS_COMPLETE:
            tombstone_track_kinematics_execution_candidate(
                archive,
                run_name=str(payload["candidate_run_path"]).rsplit("/", 1)[1],
                expected_execution_binding=_execution_binding(request),
                failure_phase=phase,
                error_type=type(error).__name__,
                error_message=str(error),
            )
        attempt = _build_failed_attempt(
            request=request,
            driver_pid=os.getpid(),
            child_start_time_ticks=child_start,
            child_pid=child_pid,
            error=error,
            failure_phase=phase,
            runtime_telemetry=runtime,
            selector_after_sha256=after_values["selector_after_sha256"],
            registry_after_sha256=after_values["registry_after_sha256"],
            production_profiles_after_sha256=after_values[
                "production_profiles_after_sha256"
            ],
            observation_errors=observation_errors,
            reported_error_type=reported_error_type,
            reported_error_message=reported_error_message,
        )
        _write_json_exclusive(attempt_file, attempt)
        return attempt

    try:
        if process.returncode == 0:
            if not hidden_receipt.is_file() or hidden_attempt.exists():
                error = RuntimeError(
                    "Successful track child did not write one provisional receipt."
                )
                attempt = fail_driver(error, phase="driver_child_evidence")
                raise TrackFlatCandidateExecutionFailed(str(error), attempt=attempt)
            try:
                provisional = _read_json(hidden_receipt)
                require_candidate_execution_receipt(
                    provisional,
                    expected_request_payload_digest=str(request["payload_digest"]),
                )
                expected_fresh = {
                    "driver_pid": os.getpid(),
                    "child_pid": child_pid,
                    "child_start_time_ticks": child_start,
                    "is_fresh": True,
                }
                if provisional["payload"]["fresh_process"] != expected_fresh:
                    raise ValueError(
                        "Track child process evidence differs from parent."
                    )
                if provisional["payload"]["publication_gate_passed"] is not False:
                    raise ValueError(
                        "Diagnostic track receipt passed publication gate."
                    )
            except BaseException as error:
                attempt = fail_driver(error, phase="driver_child_evidence")
                raise TrackFlatCandidateExecutionFailed(
                    str(error), attempt=attempt
                ) from error
            if not unchanged:
                error = RuntimeError(
                    "Track selector, registry, or production-profile state changed "
                    "or could not be observed after child execution."
                )
                attempt = fail_driver(error, phase="driver_protected_poststate")
                raise TrackFlatCandidateExecutionFailed(str(error), attempt=attempt)
            try:
                _write_json_exclusive(receipt_file, provisional)
            except BaseException as error:
                receipt_file.unlink(missing_ok=True)
                attempt = fail_driver(error, phase="driver_receipt_publication")
                raise TrackFlatCandidateExecutionFailed(
                    str(error), attempt=attempt
                ) from error
            return provisional

        runtime: Mapping[str, Any] | None = None
        reported_phase = "driver_child_failure"
        reported_error_type: str | None = None
        reported_error_message: str | None = None
        child_error: BaseException = RuntimeError(
            "Fresh track child exited nonzero: "
            f"returncode={process.returncode}, stdout={stdout!r}, stderr={stderr!r}."
        )
        if hidden_attempt.is_file() and not hidden_receipt.exists():
            try:
                child_attempt = _read_json(hidden_attempt)
                require_track_flat_execution_attempt(
                    child_attempt,
                    expected_request_payload_digest=str(request["payload_digest"]),
                )
            except BaseException as error:
                reported_phase = "driver_child_evidence"
                child_error = error
            else:
                child_payload = child_attempt["payload"]
                runtime = child_payload.get("runtime_telemetry")
                reported_phase = str(child_payload["failure_phase"])
                reported_error_type = str(child_payload["error_type"])
                reported_error_message = str(child_payload["error_message"])
                child_error = RuntimeError(
                    f"{reported_error_type}: {reported_error_message}"
                )
        attempt = fail_driver(
            child_error,
            phase=reported_phase,
            runtime=runtime,
            reported_error_type=reported_error_type,
            reported_error_message=reported_error_message,
        )
        raise TrackFlatCandidateExecutionFailed(str(child_error), attempt=attempt)
    finally:
        hidden_receipt.unlink(missing_ok=True)
        hidden_attempt.unlink(missing_ok=True)


def _child_command(args: argparse.Namespace) -> int:
    request = _read_json(Path(args.request))
    try:
        receipt = execute_track_flat_candidate(
            request,
            driver_pid=int(args.driver_pid),
        )
    except TrackFlatCandidateExecutionFailed as exc:
        _write_json_exclusive(Path(args.attempt), exc.attempt)
        return 1
    _write_json_exclusive(Path(args.receipt), receipt)
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    child = subparsers.add_parser("child")
    child.add_argument("--request", type=Path, required=True)
    child.add_argument("--receipt", type=Path, required=True)
    child.add_argument("--attempt", type=Path, required=True)
    child.add_argument("--driver-pid", type=int, required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--request", type=Path, required=True)
    run.add_argument("--receipt", type=Path, required=True)
    run.add_argument("--attempt", type=Path, required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "child":
        return _child_command(args)
    try:
        run_track_flat_candidate_fresh_process(
            args.request,
            receipt_path=args.receipt,
            attempt_path=args.attempt,
        )
    except TrackFlatCandidateExecutionFailed:
        return 1
    return 0


__all__ = [
    "ATTEMPT_SCHEMA_ID",
    "ATTEMPT_SCHEMA_VERSION",
    "TrackFlatCandidateExecutionFailed",
    "execute_track_flat_candidate",
    "require_track_flat_execution_attempt",
    "run_track_flat_candidate_fresh_process",
    "track_flat_selector_snapshot_sha256",
    "validate_track_flat_source_preservation",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
