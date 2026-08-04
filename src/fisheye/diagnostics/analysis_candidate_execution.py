"""Fresh-process execution for exact-tabular analytics storage candidates.

This runner currently owns only the compact swim-bout and bout-kinematics
logical-rematerialization families.  It accepts the closed shared request,
replays its exact storage suite against an explicit source run, stages the
source on node-local scratch, and delegates publication to the atomic
selector-ineligible candidate materializer.

The child writes either a complete execution receipt or a separate failed
attempt record.  ``/proc/self/io`` counters are retained as application
telemetry and never mislabeled as filesystem/network transfer evidence.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys
from typing import Any, Callable, Mapping, Optional, Sequence
import uuid

from fisheye.analysis.bout_kinematics_schema import (
    build_bout_kinematics_array_declarations,
    validate_bout_kinematics_array_manifest,
)
from fisheye.analysis.exact_tabular_storage import (
    ANALYSIS_STORAGE_PLAN_RECEIPT_ATTR,
    build_exact_tabular_storage_receipt,
)
from fisheye.analysis.swim_bout_frame_axis import (
    FRAME_AXIS_CONTRACT_ATTR,
    FRAME_AXIS_CONTRACT_SHA256_ATTR,
    resolve_swim_bout_frame_axis,
)
from fisheye.analysis.swim_bout_schema import (
    build_swim_bout_array_declarations,
    validate_swim_bout_array_manifest,
)
from fisheye.analysis.track_kinematics_io import load_track_kinematics_track
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
from fisheye.analysis_workflows.materializers.exact_tabular_candidate import (
    EXACT_TABULAR_EXECUTION_PHASE_ORDER,
    EXECUTION_BINDING_ATTR,
    EXECUTION_FAILURE_TOMBSTONE_ATTR,
    compute_exact_tabular_logical_hashes,
    materialize_exact_tabular_candidate,
    tombstone_exact_tabular_execution_candidate,
)
from fisheye.analysis_workflows.materializers.runtime_telemetry import (
    require_runtime_telemetry,
)
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from fisheye.shared.zarr.storage_profiles import get_storage_profile
from fisheye.shared.zarr_io import open_zarr_root


ATTEMPT_SCHEMA_ID = "palette.analysis_candidate_execution_attempt"
ATTEMPT_SCHEMA_VERSION = 1
RUNNER_REF = (
    "fisheye.diagnostics.analysis_candidate_execution:execute_exact_tabular_candidate"
)
_SHA256_LENGTH = 64
_SELECTOR_SNAPSHOT_SCHEMA_ID = "palette.analysis_candidate_selector_snapshot"
_SELECTOR_SNAPSHOT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class _Family:
    stage_id: str
    run_parent: str
    declarations: Callable[..., tuple[Any, ...]]
    validate: Callable[..., tuple[str, ...]]


_FAMILIES = {
    "swim_bouts": _Family(
        stage_id="swim_bouts",
        run_parent="analysis/swim_bout_runs",
        declarations=build_swim_bout_array_declarations,
        validate=validate_swim_bout_array_manifest,
    ),
    "bout_kinematics": _Family(
        stage_id="bout_kinematics",
        run_parent="analysis/bout_kinematics_runs",
        declarations=build_bout_kinematics_array_declarations,
        validate=validate_bout_kinematics_array_manifest,
    ),
}

_TRACK_AUTHORITY_FIELDS = frozenset(
    {
        "schema_id",
        "schema_version",
        "run_ref",
        "track_ref",
        "track_id",
        "motion_manifest_ref",
        "motion_manifest_sha256",
        "positions_px_ref",
        "positions_px_coordinate_descriptor_sha256",
        "positions_mm_ref",
        "positions_mm_coordinate_descriptor_sha256",
        "track_sample_key_ref",
        "source_acquisition_frame_index_ref",
    }
)


class ExactTabularCandidateExecutionFailed(RuntimeError):
    """Raised with a self-validating failed-attempt record."""

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
        and len(value) == _SHA256_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    )


def _family(stage_id: str) -> _Family:
    try:
        return _FAMILIES[stage_id]
    except KeyError as exc:
        raise ValueError(
            f"The exact-tabular runner does not own stage {stage_id!r}."
        ) from exc


def _group_attrs(group: Any) -> dict[str, Any]:
    attrs = group.attrs
    return dict(attrs.asdict() if hasattr(attrs, "asdict") else dict(attrs))


def _group_at_path(root: Any, path: str) -> Any:
    node = root
    for component in path.split("/"):
        node = node[component]
    return node


def _canonical_absolute_ref(value: object, *, label: str) -> tuple[str, ...]:
    if type(value) is not str or not value.startswith("/") or "\\" in value:
        raise ValueError(f"{label} is not one canonical absolute Zarr reference.")
    components = tuple(value[1:].split("/"))
    if not components or any(part in {"", ".", ".."} for part in components):
        raise ValueError(f"{label} is not one canonical absolute Zarr reference.")
    return components


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
        raise RuntimeError("Could not parse /proc/self/stat command boundary.")
    fields_from_state = text[closing + 2 :].split()
    # Field 22 (starttime); this list starts with field 3 (state).
    value = int(fields_from_state[19])
    if value <= 0:
        raise RuntimeError("Process start-time ticks are not positive.")
    return value


def _child_start_time_ticks() -> int:
    return _process_start_time_ticks()


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


def exact_tabular_selector_snapshot_sha256(
    archive_path: str | Path,
    *,
    stage_id: str,
) -> str:
    """Digest the complete target-parent attribute surface."""

    family = _family(stage_id)
    root = open_zarr_root(Path(archive_path).resolve(), mode="r")
    parent = _group_at_path(root, family.run_parent)
    return _selector_snapshot_sha256_from_parent(
        parent,
        stage_id=family.stage_id,
        run_parent=family.run_parent,
    )


def _selector_snapshot_sha256_from_parent(
    parent: Any,
    *,
    stage_id: str,
    run_parent: str,
) -> str:
    payload = {
        "schema_id": _SELECTOR_SNAPSHOT_SCHEMA_ID,
        "schema_version": _SELECTOR_SNAPSHOT_SCHEMA_VERSION,
        "stage_id": stage_id,
        "run_parent": run_parent,
        "parent_attributes": _strict_json_copy(_group_attrs(parent)),
    }
    return canonical_json_sha256(payload)


def _execution_binding(request: Mapping[str, Any]) -> dict[str, Any]:
    payload = request["payload"]
    return {
        "schema_id": "palette.analysis_candidate_execution_binding",
        "schema_version": 1,
        "execution_id": payload["execution_id"],
        "request_payload_digest": request["payload_digest"],
        "candidate_run_path": payload["candidate_run_path"],
    }


def _require_track_motion_authority(value: object, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _TRACK_AUTHORITY_FIELDS:
        raise ValueError(f"{label} track-motion authority field set differs.")
    record = _strict_json_copy(value)
    if (
        record["schema_id"] != "palette.track_motion_read_authority"
        or record["schema_version"] != 1
        or type(record["track_id"]) is not int
        or record["track_id"] < 0
    ):
        raise ValueError(f"{label} track-motion authority identity differs.")
    for field in (
        "run_ref",
        "track_ref",
        "motion_manifest_ref",
        "positions_px_ref",
        "track_sample_key_ref",
        "source_acquisition_frame_index_ref",
    ):
        if type(record[field]) is not str or not record[field].startswith("/"):
            raise ValueError(f"{label} track-motion authority {field} is invalid.")
    for field in (
        "motion_manifest_sha256",
        "positions_px_coordinate_descriptor_sha256",
    ):
        if not _is_sha256(record[field]):
            raise ValueError(f"{label} track-motion authority {field} is invalid.")
    mm_ref = record["positions_mm_ref"]
    mm_digest = record["positions_mm_coordinate_descriptor_sha256"]
    if (mm_ref is None) != (mm_digest is None):
        raise ValueError(f"{label} millimetre coordinate authority is incomplete.")
    if mm_ref is not None and (
        type(mm_ref) is not str
        or not mm_ref.startswith("/")
        or not _is_sha256(mm_digest)
    ):
        raise ValueError(f"{label} millimetre coordinate authority is invalid.")
    expected_track_ref = f"{record['run_ref']}/tracks/id_{int(record['track_id'])}"
    expected_refs = {
        "track_ref": expected_track_ref,
        "motion_manifest_ref": (
            f"{record['run_ref']}@track_motion_publication_manifest"
        ),
        "positions_px_ref": f"{expected_track_ref}/positions_px",
        "track_sample_key_ref": f"{expected_track_ref}/track_sample_key",
        "source_acquisition_frame_index_ref": (
            f"{expected_track_ref}/source_acquisition_frame_index"
        ),
    }
    if mm_ref is not None:
        expected_refs["positions_mm_ref"] = f"{expected_track_ref}/positions_mm"
    for field, expected in expected_refs.items():
        if record[field] != expected:
            raise ValueError(
                f"{label} track-motion authority {field} disagrees with its run."
            )
    return record


def _resolve_live_track_motion_authority(
    root: Any,
    authority: Mapping[str, Any],
    *,
    label: str,
) -> dict[str, Any]:
    """Resolve the public canonical reader and require its exact authority."""

    components = _canonical_absolute_ref(authority["run_ref"], label=f"{label} run_ref")
    if (
        len(components) != 4
        or components[:2] != ("analysis", "track_kinematics_runs")
        or components[2] not in {"offline", "online"}
        or components[3] == "latest"
    ):
        raise ValueError(f"{label} run_ref is not one immutable canonical track run.")
    for field in (
        "track_ref",
        "positions_px_ref",
        "track_sample_key_ref",
        "source_acquisition_frame_index_ref",
    ):
        _canonical_absolute_ref(authority[field], label=f"{label} {field}")
    if authority["positions_mm_ref"] is not None:
        _canonical_absolute_ref(
            authority["positions_mm_ref"], label=f"{label} positions_mm_ref"
        )
    tables = load_track_kinematics_track(
        root,
        run_name=components[3],
        scope=components[2],
        track_id=int(authority["track_id"]),
        required_speed_levels=(),
    )
    live = tables.authority_record()
    if live != dict(authority):
        raise ValueError(f"{label} differs from the live canonical track authority.")
    return live


def _detector_frame_count(run_group: Any) -> int:
    signal = _group_at_path(run_group, "signals/detector_signal_mm_s")
    if len(signal.shape) != 2:
        raise ValueError("Swim-bout detector signal must be rank two.")
    return int(signal.shape[1])


def _validate_exact_tabular_coordinate_binding(
    root: Any,
    run_group: Any,
    *,
    stage_id: str,
) -> dict[str, Any]:
    attrs = _group_attrs(run_group)
    if stage_id == "swim_bouts":
        authority = _require_track_motion_authority(
            attrs.get("source_track_motion_authority"),
            label="swim-bout source",
        )
        authority = _resolve_live_track_motion_authority(
            root,
            authority,
            label="swim-bout source",
        )
        if (
            attrs.get("source_track_motion_manifest_sha256")
            != authority["motion_manifest_sha256"]
        ):
            raise ValueError("Swim-bout source motion-manifest digest binding differs.")
        frame_contract = attrs.get(FRAME_AXIS_CONTRACT_ATTR)
        if not isinstance(frame_contract, Mapping):
            raise ValueError("Swim-bout source lacks a typed frame-axis contract.")
        frame_digest = canonical_json_sha256(frame_contract)
        if attrs.get(FRAME_AXIS_CONTRACT_SHA256_ATTR) != frame_digest:
            raise ValueError("Swim-bout frame-axis contract digest differs.")
        frame_values = resolve_swim_bout_frame_axis(
            root,
            run_group,
            expected_length=_detector_frame_count(run_group),
        )
        if frame_values is None:
            raise ValueError("Swim-bout frame-axis authority did not resolve.")
        authorities = [
            {
                "role": "track_motion",
                "sha256": canonical_json_sha256(authority),
            }
        ]
        validation_payload = {
            "stage_id": stage_id,
            "source_authority_digests": authorities,
            "frame_axis_contract_sha256": frame_digest,
            "resolved_frame_count": int(frame_values.size),
        }
    elif stage_id == "bout_kinematics":
        source_refs = attrs.get("source_refs")
        if not isinstance(source_refs, Mapping):
            raise ValueError("Bout-kinematics source_refs is absent or invalid.")
        track = _require_track_motion_authority(
            source_refs.get("source_track_motion_authority"),
            label="bout-kinematics source",
        )
        swim = _require_track_motion_authority(
            source_refs.get("source_swim_bout_track_motion_authority"),
            label="bout-kinematics swim-bout source",
        )
        track = _resolve_live_track_motion_authority(
            root,
            track,
            label="bout-kinematics source",
        )
        swim = _resolve_live_track_motion_authority(
            root,
            swim,
            label="bout-kinematics swim-bout source",
        )
        if swim != track:
            raise ValueError(
                "Bout-kinematics track and swim-bout coordinate authorities differ."
            )
        swim_path = source_refs.get("source_swim_bout_path")
        if type(swim_path) is not str or not swim_path:
            raise ValueError("Bout-kinematics source swim-bout path is not immutable.")
        swim_components = tuple(swim_path.split("/"))
        if (
            len(swim_components) < 3
            or swim_components[:2] != ("analysis", "swim_bout_runs")
            or swim_components[2] in {"", ".", "..", "latest"}
            or any(part in {"", ".", "..", "latest"} for part in swim_components)
        ):
            raise ValueError(
                "Bout-kinematics source swim-bout path is not canonical and immutable."
            )
        try:
            _group_at_path(root, swim_path)
            swim_run = _group_at_path(root, "/".join(swim_components[:3]))
        except (KeyError, TypeError) as exc:
            raise ValueError(
                "Bout-kinematics source swim-bout path does not resolve."
            ) from exc
        swim_errors = _FAMILIES["swim_bouts"].validate(swim_run)
        if swim_errors:
            raise ValueError(
                "Bout-kinematics source swim-bout run is invalid: "
                + "; ".join(swim_errors)
            )
        swim_coordinate = _validate_exact_tabular_coordinate_binding(
            root,
            swim_run,
            stage_id="swim_bouts",
        )
        live_swim_authorities = {
            item["role"]: item["sha256"]
            for item in swim_coordinate["source_authority_digests"]
        }
        if live_swim_authorities.get("track_motion") != canonical_json_sha256(track):
            raise ValueError(
                "Bout-kinematics source swim-bout run uses another track authority."
            )
        authorities = [
            {
                "role": "source_swim_bout_track_motion",
                "sha256": canonical_json_sha256(swim),
            },
            {"role": "track_motion", "sha256": canonical_json_sha256(track)},
        ]
        validation_payload = {
            "stage_id": stage_id,
            "source_authority_digests": authorities,
            "source_swim_bout_path": swim_path,
            "source_swim_bout_coordinate_receipt_sha256": swim_coordinate[
                "validation_receipt_sha256"
            ],
        }
    else:  # pragma: no cover - guarded by _family
        raise ValueError(f"Unsupported coordinate-binding stage {stage_id!r}.")
    return {
        "role": "bound_derivative",
        "status": CoordinateEvidenceStatus.VERIFIED_BOUND_SOURCE.value,
        "source_authority_digests": authorities,
        "published_authority_sha256": None,
        "published_authority_ref": None,
        "temporal_axis_sha256": None,
        "temporal_axis_ref": None,
        "validator_ref": f"{__name__}:_validate_exact_tabular_coordinate_binding",
        "validation_receipt_sha256": canonical_json_sha256(validation_payload),
        "coordinate_gate_passed": True,
    }


def _translate_phases(runtime: Mapping[str, Any]) -> list[CandidatePhaseMeasurement]:
    require_runtime_telemetry(
        runtime,
        expected_materializer="exact_tabular_candidate",
        allowed_phase_order=EXACT_TABULAR_EXECUTION_PHASE_ORDER,
        require_error_phase=False,
    )
    records = runtime.get("phases")
    if not isinstance(records, list):
        raise ValueError("Materializer runtime telemetry has no phase list.")
    expected = required_execution_phases(
        CandidateComputationMode.LOGICAL_REMATERIALIZATION
    )
    if [record.get("name") for record in records] != [
        phase.value for phase in expected
    ]:
        raise ValueError("Materializer runtime phase sequence differs.")
    result: list[CandidatePhaseMeasurement] = []
    for phase, record in zip(expected, records, strict=True):
        if record.get("outcome") != "ok":
            raise ValueError("A successful materializer contains a failed phase.")
        cpu = record.get("cpu_seconds")
        if not isinstance(cpu, Mapping):
            raise ValueError("Materializer phase CPU telemetry is invalid.")
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
            "The direct child runner supports only unavailable or process-self "
            "I/O; filesystem/network transfer requires an external tracer."
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
        group = open_zarr_root(target, mode="r")
        attrs = _group_attrs(group)
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
        "completion_status": attrs.get("palette_run_completion_status"),
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
            "Failed candidate execution left a selector-eligible public target."
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
    require_exact_tabular_execution_attempt(
        result,
        expected_request_payload_digest=str(request["payload_digest"]),
    )
    return result


def require_exact_tabular_execution_attempt(
    value: Mapping[str, Any],
    *,
    expected_request_payload_digest: str,
) -> None:
    """Deeply validate the terminal failure envelope."""

    if not isinstance(value, Mapping) or set(value) != {
        "schema_id",
        "schema_version",
        "payload",
        "payload_digest",
    }:
        raise ValueError("Execution-attempt envelope field set differs.")
    if (
        value["schema_id"] != ATTEMPT_SCHEMA_ID
        or value["schema_version"] != ATTEMPT_SCHEMA_VERSION
    ):
        raise ValueError("Execution-attempt schema identity differs.")
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
        raise ValueError("Execution-attempt payload field set differs.")
    if value["payload_digest"] != canonical_json_sha256(payload):
        raise ValueError("Execution-attempt payload digest differs.")
    _strict_json_copy(value)
    require_candidate_execution_request(payload["request"])
    if (
        payload["status"] != "failed"
        or payload["request_payload_digest"] != payload["request"]["payload_digest"]
        or payload["request_payload_digest"] != expected_request_payload_digest
    ):
        raise ValueError("Execution-attempt request binding differs.")
    if not _is_sha256(expected_request_payload_digest):
        raise ValueError("Expected execution-attempt request digest is invalid.")
    for field in ("driver_pid", "child_pid", "child_start_time_ticks"):
        if type(payload[field]) is not int or payload[field] <= 0:
            raise ValueError(f"Execution-attempt {field} is invalid.")
    if payload["driver_pid"] == payload["child_pid"]:
        raise ValueError("Execution-attempt does not identify a distinct child.")
    try:
        failed_at = datetime.fromisoformat(payload["failed_at_utc"])
    except (TypeError, ValueError) as exc:
        raise ValueError("Execution-attempt timestamp is invalid.") from exc
    if failed_at.tzinfo is None or failed_at.utcoffset() != timezone.utc.utcoffset(
        failed_at
    ):
        raise ValueError("Execution-attempt timestamp is not UTC.")
    if (
        payload["failure_phase"]
        not in {
            "runner_preflight",
            "materializer",
            "runner_receipt_assembly",
            "driver_child_evidence",
            "driver_protected_poststate",
            "driver_receipt_publication",
            "driver_child_failure",
        }
        or type(payload["error_type"]) is not str
        or not payload["error_type"]
        or type(payload["error_message"]) is not str
        or not payload["error_message"]
    ):
        raise ValueError("Execution-attempt failure identity is incomplete.")
    target = payload["target_state"]
    if not isinstance(target, Mapping) or set(target) != {
        "exists",
        "completion_status",
        "selector_eligible",
        "atomic_tombstone_present",
        "inspection_error",
    }:
        raise ValueError("Execution-attempt target-state field set differs.")
    if (
        type(target["exists"]) is not bool
        or type(target["atomic_tombstone_present"]) is not bool
        or target["selector_eligible"] is True
    ):
        raise ValueError("Failed execution target is selector eligible.")
    for field in ("completion_status", "inspection_error"):
        if target[field] is not None and (
            type(target[field]) is not str or not target[field]
        ):
            raise ValueError(f"Execution-attempt target {field} is invalid.")
    if (
        target["selector_eligible"] is not None
        and target["selector_eligible"] is not False
    ):
        raise ValueError("Execution-attempt target eligibility is invalid.")
    if not target["exists"] and (
        target["completion_status"] is not None
        or target["selector_eligible"] is not None
        or target["atomic_tombstone_present"]
        or target["inspection_error"] is not None
    ):
        raise ValueError("Absent execution-attempt target claims persisted state.")
    if target["atomic_tombstone_present"] and (
        not target["exists"]
        or target["completion_status"] != "failed"
        or target["selector_eligible"] is not False
        or target["inspection_error"] is not None
    ):
        raise ValueError("Execution-attempt tombstone state is inconsistent.")
    if target["inspection_error"] is not None and not target["exists"]:
        raise ValueError("Execution-attempt inspection error lacks a target.")
    runtime = payload["runtime_telemetry"]
    if runtime is not None:
        require_runtime_telemetry(
            runtime,
            expected_materializer="exact_tabular_candidate",
            allowed_phase_order=EXACT_TABULAR_EXECUTION_PHASE_ORDER,
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
        raise ValueError("Execution-attempt nonmutation evidence is absent.")
    protected = payload["request"]["payload"]["protected_state_before"]
    if (
        nonmutation.get("selector_before_sha256") != protected["selector_sha256"]
        or nonmutation.get("registry_before_sha256") != protected["registry_sha256"]
        or nonmutation.get("production_profiles_before_sha256")
        != protected["production_profiles_sha256"]
        or nonmutation.get("snapshot_contract_id") != protected["snapshot_contract_id"]
    ):
        raise ValueError("Execution-attempt pre-state binding differs.")
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
        raise ValueError("Execution-attempt observation errors are invalid.")
    for field in (
        "selector_after_sha256",
        "registry_after_sha256",
        "production_profiles_after_sha256",
    ):
        observed = nonmutation.get(field)
        if observed is None:
            if field not in errors:
                raise ValueError(f"Execution-attempt missing {field} lacks an error.")
        elif not _is_sha256(observed) or field in errors:
            raise ValueError(f"Execution-attempt {field} is invalid.")
    expected_unchanged = bool(
        not errors
        and nonmutation["selector_before_sha256"]
        == nonmutation["selector_after_sha256"]
        and nonmutation["registry_before_sha256"]
        == nonmutation["registry_after_sha256"]
        and nonmutation["production_profiles_before_sha256"]
        == nonmutation["production_profiles_after_sha256"]
    )
    if nonmutation.get("unchanged") is not expected_unchanged:
        raise ValueError("Execution-attempt nonmutation result differs.")


def execute_exact_tabular_candidate(
    request: Mapping[str, Any],
    *,
    driver_pid: int,
    copy_backend: str = "python",
    keep_scratch: bool = False,
) -> dict[str, Any]:
    """Execute scientific publication in one direct child.

    Registry/profile post-state is deliberately not observed here.  The
    controlling parent probes it after this process terminates before making
    this provisional receipt externally visible.
    """

    require_candidate_execution_request(request)
    payload = request["payload"]
    adapter = payload["adapter_manifest"]["payload"]
    family = _family(str(adapter["stage_id"]))
    if adapter["computation_mode"] != "logical_rematerialization":
        raise ValueError("Exact-tabular runner requires logical rematerialization.")
    if type(driver_pid) is not int or driver_pid <= 0 or driver_pid != os.getppid():
        raise ValueError(
            "Exact-tabular execution must run in a direct child of its driver."
        )
    scope = PhysicalIOScope(payload["physical_io_scope"])
    if scope not in {
        PhysicalIOScope.UNAVAILABLE,
        PhysicalIOScope.PROCESS_SELF_PROC_IO,
    }:
        raise ValueError(
            "External transfer scopes require the future traced-process driver."
        )

    child_start = _child_start_time_ticks()
    archive = Path(payload["archive_path"])
    source_path = str(payload["source_run_path"])
    candidate_path = str(payload["candidate_run_path"])
    source_name = source_path.rsplit("/", 1)[1]
    candidate_name = candidate_path.rsplit("/", 1)[1]
    protected = payload["protected_state_before"]
    selector_before = str(protected["selector_sha256"])
    execution_binding = _execution_binding(request)
    io_before = _proc_io()
    failure_phase = "runner_preflight"
    runtime_telemetry: Mapping[str, Any] | None = None
    try:
        live_selector_before = exact_tabular_selector_snapshot_sha256(
            archive,
            stage_id=family.stage_id,
        )
        if live_selector_before != selector_before:
            raise ValueError(
                "Live selector pre-state differs from the execution request."
            )
        commit, dirty = _git_identity()
        if commit != payload["palette_commit"]:
            raise ValueError("Runner Git commit differs from the execution request.")
        root = open_zarr_root(archive, mode="r")
        source_group = _group_at_path(root, source_path)
        source_errors = family.validate(source_group)
        if source_errors:
            raise ValueError(
                "Exact-tabular source manifest is invalid: " + "; ".join(source_errors)
            )
        source_declarations = family.declarations(
            source_group,
            byte_planner_adopted=False,
        )
        source_hashes = compute_exact_tabular_logical_hashes(
            source_group,
            source_declarations,
        )
        source_identity = canonical_json_sha256(source_hashes)
        if source_identity != payload["source_identity_sha256"]:
            raise ValueError("Source logical identity differs from the request.")
        candidate_declarations = family.declarations(
            source_group,
            byte_planner_adopted=True,
        )
        actual_receipt = build_exact_tabular_storage_receipt(
            source_group,
            declarations=candidate_declarations,
            profile=get_storage_profile(str(adapter["profile_id"])),
        ).as_manifest()
        planned_receipt = payload["benchmark_suite"]["payload"]["storage_plan_receipt"]
        if actual_receipt != planned_receipt:
            raise ValueError(
                "Live exact-tabular storage plan differs from the benchmark suite."
            )
        source_coordinate = _validate_exact_tabular_coordinate_binding(
            root,
            source_group,
            stage_id=family.stage_id,
        )

        def accept_published_candidate(
            published_root: Any,
            published_parent: Any,
            candidate_group: Any,
        ) -> Mapping[str, Any]:
            if candidate_group.attrs.get(EXECUTION_BINDING_ATTR) != execution_binding:
                raise ValueError(
                    "Published execution binding differs from its request."
                )
            candidate_coordinate = _validate_exact_tabular_coordinate_binding(
                published_root,
                candidate_group,
                stage_id=family.stage_id,
            )
            if candidate_coordinate != source_coordinate:
                raise ValueError(
                    "Published coordinate binding differs from its validated source."
                )
            persisted_receipt = candidate_group.attrs.get(
                ANALYSIS_STORAGE_PLAN_RECEIPT_ATTR
            )
            if persisted_receipt != planned_receipt:
                raise ValueError(
                    "Published storage receipt differs from the execution suite."
                )
            selector_after = _selector_snapshot_sha256_from_parent(
                published_parent,
                stage_id=family.stage_id,
                run_parent=family.run_parent,
            )
            if selector_after != selector_before:
                raise ValueError(
                    "Exact-tabular execution changed parent selector state."
                )
            return {
                "coordinate_evidence": candidate_coordinate,
                "selector_after_sha256": selector_after,
                "execution_binding": execution_binding,
                "storage_plan_receipt_sha256": canonical_json_sha256(persisted_receipt),
            }

        failure_phase = "materializer"
        materialized = materialize_exact_tabular_candidate(
            archive,
            family_id=family.stage_id,
            source_run=source_name,
            run_name=candidate_name,
            scratch_root=payload["scratch_root"],
            profile_id=str(adapter["profile_id"]),
            copy_backend=copy_backend,
            apply=True,
            keep_scratch=keep_scratch,
            stage_source_to_scratch=True,
            execution_binding=execution_binding,
            publication_acceptance_validator=accept_published_candidate,
        )
        runtime_telemetry = materialized["runtime_telemetry"]
        phases = _translate_phases(runtime_telemetry)

        failure_phase = "runner_receipt_assembly"
        caller_acceptance = materialized.get("caller_acceptance")
        if not isinstance(caller_acceptance, Mapping):
            raise ValueError("Atomic publication omitted runner acceptance evidence.")
        if caller_acceptance.get("execution_binding") != execution_binding:
            raise ValueError("Atomic publication acceptance binding differs.")
        if caller_acceptance.get("coordinate_evidence") != source_coordinate:
            raise ValueError("Atomic publication coordinate receipt differs.")
        selector_after = caller_acceptance.get("selector_after_sha256")
        if selector_after != selector_before:
            raise ValueError("Atomic publication selector receipt differs.")
        io_after = _proc_io()
        physical = _physical_io_evidence(
            request=request,
            before=io_before,
            after=io_after,
        )
        array_count = int(materialized["published_validation"]["array_count"])
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
            coordinate_evidence=source_coordinate,
            logical_equality={
                "contract_id": adapter["logical_equality_contract"],
                "compared_array_count": array_count,
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
            physical_io=physical,
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
        return receipt
    except BaseException as exc:
        attached = getattr(exc, "palette_runtime_telemetry", None)
        if isinstance(attached, Mapping):
            runtime_telemetry = attached
        target = _target_state(archive, candidate_path)
        if target.get("completion_status") == "complete":
            tombstone_exact_tabular_execution_candidate(
                archive,
                family_id=family.stage_id,
                run_name=candidate_name,
                expected_execution_binding=execution_binding,
                failure_phase=failure_phase,
                error_type=type(exc).__name__,
                error_message=str(exc),
            )
        selector_after = exact_tabular_selector_snapshot_sha256(
            archive,
            stage_id=family.stage_id,
        )
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
        raise ExactTabularCandidateExecutionFailed(
            f"Exact-tabular candidate execution failed during {failure_phase}: {exc}",
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


def run_exact_tabular_candidate_fresh_process(
    request_path: str | Path,
    *,
    receipt_path: str | Path,
    attempt_path: str | Path,
    copy_backend: str = "python",
    keep_scratch: bool = False,
) -> dict[str, Any]:
    """Launch once, then observe protected after-state before exposing evidence."""

    request_file = Path(request_path).expanduser().resolve()
    receipt_file = Path(receipt_path).expanduser().resolve()
    attempt_file = Path(attempt_path).expanduser().resolve()
    if receipt_file == attempt_file:
        raise ValueError("Receipt and failed-attempt paths must differ.")
    if receipt_file.exists() or attempt_file.exists():
        raise FileExistsError("Fresh-process evidence paths must not already exist.")
    request = _read_json(request_file)
    require_candidate_execution_request(request)
    payload = request["payload"]
    archive = Path(payload["archive_path"])
    family = _family(str(payload["adapter_manifest"]["payload"]["stage_id"]))
    protected = payload["protected_state_before"]
    for path in (receipt_file, attempt_file):
        if ".palette_benchmarks" not in path.parts or path.is_relative_to(archive):
            raise ValueError(
                "Execution evidence must be a benchmark sidecar outside the Zarr."
            )
    live_registry_before = protected_path_snapshot_sha256(
        protected["registry_probe_path"]
    )
    live_profiles_before = protected_path_snapshot_sha256(
        protected["production_profiles_probe_path"]
    )
    if (
        live_registry_before != protected["registry_sha256"]
        or live_profiles_before != protected["production_profiles_sha256"]
    ):
        raise ValueError("Protected path pre-state differs from the execution request.")
    live_selector_before = exact_tabular_selector_snapshot_sha256(
        archive,
        stage_id=family.stage_id,
    )
    if live_selector_before != protected["selector_sha256"]:
        raise ValueError("Selector pre-state differs from the execution request.")

    hidden_receipt = receipt_file.with_name(f".{receipt_file.name}.child.{os.getpid()}")
    hidden_attempt = attempt_file.with_name(f".{attempt_file.name}.child.{os.getpid()}")
    if hidden_receipt.exists() or hidden_attempt.exists():
        raise FileExistsError("Fresh-process hidden evidence path already exists.")
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
        "--copy-backend",
        copy_backend,
    ]
    if keep_scratch:
        command.append("--keep-scratch")
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
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
            lambda: exact_tabular_selector_snapshot_sha256(
                archive,
                stage_id=family.stage_id,
            ),
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
        if target.get("completion_status") == "complete":
            tombstone_exact_tabular_execution_candidate(
                archive,
                family_id=family.stage_id,
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
                    "Successful child did not write exactly one provisional receipt."
                )
                attempt = fail_driver(error, phase="driver_child_evidence")
                raise ExactTabularCandidateExecutionFailed(str(error), attempt=attempt)
            try:
                provisional = _read_json(hidden_receipt)
                require_candidate_execution_receipt(
                    provisional,
                    expected_request_payload_digest=str(request["payload_digest"]),
                )
                expected_fresh_process = {
                    "driver_pid": os.getpid(),
                    "child_pid": child_pid,
                    "child_start_time_ticks": child_start,
                    "is_fresh": True,
                }
                if provisional["payload"]["fresh_process"] != expected_fresh_process:
                    raise ValueError(
                        "Child fresh-process claims differ from parent observations."
                    )
            except BaseException as error:
                attempt = fail_driver(error, phase="driver_child_evidence")
                raise ExactTabularCandidateExecutionFailed(
                    str(error), attempt=attempt
                ) from error
            if not unchanged:
                error = RuntimeError(
                    "Protected selector, registry, or production-profile state changed "
                    "or could not be observed after child execution."
                )
                attempt = fail_driver(error, phase="driver_protected_poststate")
                raise ExactTabularCandidateExecutionFailed(str(error), attempt=attempt)
            try:
                _write_json_exclusive(receipt_file, provisional)
            except BaseException as error:
                # The destination did not exist before this attempt; if the
                # hard-link commit succeeded but durability acknowledgement
                # failed, remove that unacknowledged success record before
                # publishing the terminal failure evidence.
                receipt_file.unlink(missing_ok=True)
                attempt = fail_driver(error, phase="driver_receipt_publication")
                raise ExactTabularCandidateExecutionFailed(
                    str(error), attempt=attempt
                ) from error
            return provisional

        runtime: Mapping[str, Any] | None = None
        reported_phase = "driver_child_failure"
        reported_error_type: str | None = None
        reported_error_message: str | None = None
        child_error: BaseException = RuntimeError(
            "Fresh child exited nonzero: "
            f"returncode={process.returncode}, stdout={stdout!r}, stderr={stderr!r}."
        )
        if hidden_attempt.is_file() and not hidden_receipt.exists():
            try:
                child_attempt = _read_json(hidden_attempt)
                require_exact_tabular_execution_attempt(
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
                    f"{child_payload['error_type']}: {child_payload['error_message']}"
                )
        elif hidden_attempt.exists() or hidden_receipt.exists():
            reported_phase = "driver_child_evidence"
            child_error = RuntimeError(
                "Failed child wrote an invalid combination of provisional evidence."
            )
        attempt = fail_driver(
            child_error,
            phase=reported_phase,
            runtime=runtime,
            reported_error_type=reported_error_type,
            reported_error_message=reported_error_message,
        )
        raise ExactTabularCandidateExecutionFailed(
            "Fresh exact-tabular child failed; see its immutable attempt record.",
            attempt=attempt,
        )
    finally:
        for hidden in (hidden_receipt, hidden_attempt):
            try:
                hidden.unlink(missing_ok=True)
            except OSError:
                pass


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    child = subparsers.add_parser("child")
    child.add_argument("--request", type=Path, required=True)
    child.add_argument("--receipt", type=Path, required=True)
    child.add_argument("--attempt", type=Path, required=True)
    child.add_argument("--driver-pid", type=int, required=True)
    child.add_argument("--copy-backend", choices=("python", "rsync"), default="python")
    child.add_argument("--keep-scratch", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    request = _read_json(args.request.resolve())
    try:
        receipt = execute_exact_tabular_candidate(
            request,
            driver_pid=args.driver_pid,
            copy_backend=args.copy_backend,
            keep_scratch=args.keep_scratch,
        )
    except ExactTabularCandidateExecutionFailed as exc:
        _write_json_exclusive(args.attempt.resolve(), exc.attempt)
        return 1
    _write_json_exclusive(args.receipt.resolve(), receipt)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "ATTEMPT_SCHEMA_ID",
    "ATTEMPT_SCHEMA_VERSION",
    "ExactTabularCandidateExecutionFailed",
    "exact_tabular_selector_snapshot_sha256",
    "execute_exact_tabular_candidate",
    "require_exact_tabular_execution_attempt",
    "run_exact_tabular_candidate_fresh_process",
]
