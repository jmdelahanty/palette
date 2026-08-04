"""Cross-language consumer evidence for derived-analysis storage candidates.

Family read matrices prove that Palette's benchmark adapters can decode two
physical layouts.  They do not prove that a maintained Palette reader or a
Crimson adapter consumed those layouts.  This module defines the portable,
fail-closed evidence envelope for that separate gate.

One receipt binds a deeply validated family matrix to balanced fresh-process
source/candidate trials from exactly one consumer and representative scale.
The validator derives the compatibility verdict from typed-open, metadata,
decoded-value, workload, selection, and nonmutation facts.  Neither a passing
receipt nor this module authorizes a selector or physical-profile change.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
from typing import Any, Mapping, Sequence

from fisheye.shared.zarr.manifest_digest import canonical_json_sha256

from .storage_benchmark_catalog import (
    DERIVED_ANALYSIS_STORAGE_BENCHMARK_BY_STAGE,
)

CONSUMER_EVIDENCE_SCHEMA_ID = "palette.derived_analysis_storage_consumer_evidence"
CONSUMER_EVIDENCE_SCHEMA_VERSION = 1

_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_.-]*$")
_REVISION = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REQUIRED_MEASUREMENTS = (
    "full_scan_ms",
    "peak_rss_bytes",
    "physical_read_bytes",
    "physical_read_operations",
    "primary_read_p95_ms",
    "readiness_ms",
    "throughput_rows_per_second",
)


class StorageConsumer(str, Enum):
    """Consumer implementations eligible to produce a v1 receipt."""

    PALETTE = "palette"
    CRIMSON = "crimson"


class ConsumerEvidenceScale(str, Enum):
    """Required representative execution scales."""

    REPRESENTATIVE_SHORT = "representative_short"
    REPRESENTATIVE_FULL = "representative_full"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _strict_json_file(path: Path, *, label: str) -> Mapping[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} is unsafe or absent")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not strict JSON") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must contain one object")
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} is not strict JSON") from exc
    return value


def _absolute_path(value: object, *, label: str) -> str:
    if type(value) is not str or not value or not Path(value).is_absolute():
        raise ValueError(f"{label} must be one absolute path")
    return str(Path(value))


def _relative_run_path(value: object, *, label: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{label} must be one nonempty relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"{label} is not canonical")
    return path.as_posix()


def _utc_timestamp(value: object, *, label: str) -> str:
    if type(value) is not str:
        raise ValueError(f"{label} must be one UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{label} must be one UTC timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ValueError(f"{label} must use UTC")
    return value


def _measurement(value: object, *, name: str) -> int | float | None:
    if value is None:
        return None
    if type(value) not in {int, float} or value < 0:
        raise ValueError(f"consumer measurement {name!r} is invalid")
    return value


def _normalized_matrix_binding(
    *,
    stage_id: str,
    matrix_path: Path,
) -> tuple[Mapping[str, Any], dict[str, object]]:
    if stage_id not in DERIVED_ANALYSIS_STORAGE_BENCHMARK_BY_STAGE:
        raise ValueError(f"unknown derived-analysis benchmark stage {stage_id!r}")
    supplied = matrix_path.expanduser()
    if supplied.is_symlink() or not supplied.is_file():
        raise ValueError("consumer evidence matrix artifact is unsafe or absent")
    resolved = supplied.resolve(strict=True)
    matrix = _strict_json_file(resolved, label="consumer evidence matrix artifact")
    normalized = DERIVED_ANALYSIS_STORAGE_BENCHMARK_BY_STAGE[
        stage_id
    ].validated_matrix_identity(matrix)
    binding = {
        "path": str(resolved),
        "sha256": _sha256_file(resolved),
        "bytes": resolved.stat().st_size,
        "payload_digest": matrix["payload_digest"],
        "normalized_identity": normalized,
        "matrix_validated": True,
    }
    return matrix, binding


def _gate_facts(
    *,
    repetitions: Sequence[Mapping[str, Any]],
    producer_clean: bool,
) -> dict[str, bool]:
    trials = [trial for repetition in repetitions for trial in repetition["trials"]]
    decoded = {trial["decoded_logical_digest"] for trial in trials}
    workloads = {trial["workload_result_digest"] for trial in trials}
    universal = all(
        trial["exit_code"] == 0
        and trial["exact_schema_and_dtype"] is True
        and trial["direct_consolidated_metadata_equivalence"] is True
        and trial["explicit_run_selection"] is True
        and trial["dtype_probe_count"] == 0
        and trial["stale_publication_count"] == 0
        and trial["production_mutations"] == []
        for trial in trials
    )
    compatible = universal and len(decoded) == 1 and len(workloads) == 1
    return {
        "balanced_fresh_processes_complete": True,
        "decoded_equality": len(decoded) == 1,
        "workload_equality": len(workloads) == 1,
        "metadata_equivalence": all(
            trial["direct_consolidated_metadata_equivalence"] is True
            for trial in trials
        ),
        "consumer_gate_passed": compatible,
        "evidence_eligible": compatible and producer_clean,
    }


def build_storage_consumer_evidence(
    *,
    stage_id: str,
    consumer: StorageConsumer,
    scale: ConsumerEvidenceScale,
    execution_id: str,
    matrix_path: str | Path,
    consumer_archive_path: str | Path,
    producer_revision: str,
    producer_worktree_clean: bool,
    executable_sha256: str,
    command: Sequence[str],
    workload_contract_id: str,
    workload_contract_version: int,
    workload_contract_digest: str,
    cache_state: str,
    started_at_utc: str,
    finished_at_utc: str,
    platform_record: Mapping[str, str],
    repetitions: Sequence[Mapping[str, Any]],
) -> dict[str, object]:
    """Build one immutable source/candidate consumer-compatibility receipt."""

    if not isinstance(consumer, StorageConsumer):
        raise TypeError("consumer must use StorageConsumer")
    if not isinstance(scale, ConsumerEvidenceScale):
        raise TypeError("scale must use ConsumerEvidenceScale")
    matrix_file = Path(matrix_path)
    _, matrix_binding = _normalized_matrix_binding(
        stage_id=stage_id,
        matrix_path=matrix_file,
    )
    normalized = matrix_binding["normalized_identity"]
    assert isinstance(normalized, Mapping)
    archive_path = str(Path(consumer_archive_path).expanduser())
    if not Path(archive_path).is_absolute():
        raise ValueError("consumer archive path must be absolute")
    source_relative = _relative_run_path(
        normalized["source_run_path"], label="normalized source run path"
    )
    candidate_relative = _relative_run_path(
        normalized["candidate_run_path"], label="normalized candidate run path"
    )
    payload: dict[str, object] = {
        "stage_id": stage_id,
        "consumer": consumer.value,
        "scale": scale.value,
        "execution_id": execution_id,
        "matrix_binding": matrix_binding,
        "consumer_paths": {
            "archive_path": archive_path,
            "source_run_path": str(Path(archive_path) / source_relative),
            "candidate_run_path": str(Path(archive_path) / candidate_relative),
        },
        "producer": {
            "repository": consumer.value,
            "revision": producer_revision,
            "worktree_clean": producer_worktree_clean,
            "executable_sha256": executable_sha256,
        },
        "execution": {
            "command": list(command),
            "workload_contract_id": workload_contract_id,
            "workload_contract_version": workload_contract_version,
            "workload_contract_digest": workload_contract_digest,
            "cache_state": cache_state,
            "started_at_utc": started_at_utc,
            "finished_at_utc": finished_at_utc,
            "platform": dict(platform_record),
        },
        "repetitions": [dict(item) for item in repetitions],
        "gate": {},
        "promotion_authorized": False,
    }
    _validate_consumer_payload_shape(payload, validate_gate=False)
    payload["gate"] = _gate_facts(
        repetitions=payload["repetitions"],
        producer_clean=producer_worktree_clean,
    )
    receipt = {
        "schema_id": CONSUMER_EVIDENCE_SCHEMA_ID,
        "schema_version": CONSUMER_EVIDENCE_SCHEMA_VERSION,
        "payload": payload,
        "payload_digest": canonical_json_sha256(payload),
    }
    require_storage_consumer_evidence(receipt)
    return receipt


def _validate_trial(
    trial: object,
    *,
    repetition_index: int,
    order: Sequence[str],
) -> str:
    expected = {
        "role",
        "order_position",
        "process_identity",
        "exit_code",
        "exact_schema_and_dtype",
        "direct_consolidated_metadata_equivalence",
        "explicit_run_selection",
        "dtype_probe_count",
        "stale_publication_count",
        "production_mutations",
        "decoded_logical_digest",
        "workload_result_digest",
        "measurements",
    }
    if not isinstance(trial, Mapping) or set(trial) != expected:
        raise ValueError("consumer trial field set differs")
    role = trial["role"]
    position = trial["order_position"]
    if role not in {"source", "candidate"} or type(position) is not int:
        raise ValueError("consumer trial role or order position is invalid")
    if position not in {0, 1} or order[position] != role:
        raise ValueError("consumer trial order binding differs")
    process_identity = trial["process_identity"]
    if (
        type(process_identity) is not str
        or not process_identity
        or not process_identity.startswith(f"r{repetition_index}-")
    ):
        raise ValueError("consumer trial process identity is invalid")
    if type(trial["exit_code"]) is not int:
        raise ValueError("consumer trial exit code is invalid")
    for field in (
        "exact_schema_and_dtype",
        "direct_consolidated_metadata_equivalence",
        "explicit_run_selection",
    ):
        if type(trial[field]) is not bool:
            raise ValueError(f"consumer trial {field} must be one exact bool")
    for field in ("dtype_probe_count", "stale_publication_count"):
        if type(trial[field]) is not int or trial[field] < 0:
            raise ValueError(f"consumer trial {field} is invalid")
    if not isinstance(trial["production_mutations"], list) or any(
        type(item) is not str or not item for item in trial["production_mutations"]
    ):
        raise ValueError("consumer trial production mutations are invalid")
    for field in ("decoded_logical_digest", "workload_result_digest"):
        if type(trial[field]) is not str or not _SHA256.fullmatch(trial[field]):
            raise ValueError(f"consumer trial {field} is invalid")
    measurements = trial["measurements"]
    if not isinstance(measurements, Mapping) or set(measurements) != set(
        _REQUIRED_MEASUREMENTS
    ):
        raise ValueError("consumer trial measurement field set differs")
    for name, value in measurements.items():
        _measurement(value, name=name)
    if any(
        measurements[name] is None
        for name in (
            "full_scan_ms",
            "peak_rss_bytes",
            "primary_read_p95_ms",
            "readiness_ms",
            "throughput_rows_per_second",
        )
    ):
        raise ValueError("consumer trial omits a required performance measurement")
    if (measurements["physical_read_bytes"] is None) != (
        measurements["physical_read_operations"] is None
    ):
        raise ValueError(
            "consumer physical-read measurements are only partially present"
        )
    return process_identity


def _validate_consumer_payload_shape(
    payload: Mapping[str, Any],
    *,
    validate_gate: bool,
) -> None:
    expected = {
        "stage_id",
        "consumer",
        "scale",
        "execution_id",
        "matrix_binding",
        "consumer_paths",
        "producer",
        "execution",
        "repetitions",
        "gate",
        "promotion_authorized",
    }
    if set(payload) != expected:
        raise ValueError("consumer evidence payload field set differs")
    stage_id = payload["stage_id"]
    if stage_id not in DERIVED_ANALYSIS_STORAGE_BENCHMARK_BY_STAGE:
        raise ValueError("consumer evidence stage is unknown")
    if payload["consumer"] not in {item.value for item in StorageConsumer}:
        raise ValueError("consumer evidence consumer is unknown")
    if payload["scale"] not in {item.value for item in ConsumerEvidenceScale}:
        raise ValueError("consumer evidence scale is unknown")
    execution_id = payload["execution_id"]
    if type(execution_id) is not str or not _IDENTIFIER.fullmatch(execution_id):
        raise ValueError("consumer evidence execution ID is invalid")

    binding = payload["matrix_binding"]
    if not isinstance(binding, Mapping) or set(binding) != {
        "path",
        "sha256",
        "bytes",
        "payload_digest",
        "normalized_identity",
        "matrix_validated",
    }:
        raise ValueError("consumer matrix binding field set differs")
    _absolute_path(binding["path"], label="consumer matrix artifact path")
    if (
        type(binding["sha256"]) is not str
        or not _SHA256.fullmatch(binding["sha256"])
        or type(binding["bytes"]) is not int
        or binding["bytes"] < 0
        or type(binding["payload_digest"]) is not str
        or not _SHA256.fullmatch(binding["payload_digest"])
        or binding["matrix_validated"] is not True
        or not isinstance(binding["normalized_identity"], Mapping)
        or binding["normalized_identity"].get("stage_id") != stage_id
    ):
        raise ValueError("consumer matrix binding is invalid")
    normalized = binding["normalized_identity"]
    source_relative = _relative_run_path(
        normalized.get("source_run_path"), label="normalized source run path"
    )
    candidate_relative = _relative_run_path(
        normalized.get("candidate_run_path"), label="normalized candidate run path"
    )
    paths = payload["consumer_paths"]
    if not isinstance(paths, Mapping) or set(paths) != {
        "archive_path",
        "source_run_path",
        "candidate_run_path",
    }:
        raise ValueError("consumer path field set differs")
    archive = _absolute_path(paths["archive_path"], label="consumer archive path")
    if paths != {
        "archive_path": archive,
        "source_run_path": str(Path(archive) / source_relative),
        "candidate_run_path": str(Path(archive) / candidate_relative),
    }:
        raise ValueError("consumer paths differ from normalized matrix identity")

    producer = payload["producer"]
    if not isinstance(producer, Mapping) or set(producer) != {
        "repository",
        "revision",
        "worktree_clean",
        "executable_sha256",
    }:
        raise ValueError("consumer producer field set differs")
    if (
        producer["repository"] != payload["consumer"]
        or type(producer["revision"]) is not str
        or not _REVISION.fullmatch(producer["revision"])
        or type(producer["worktree_clean"]) is not bool
        or type(producer["executable_sha256"]) is not str
        or not _SHA256.fullmatch(producer["executable_sha256"])
    ):
        raise ValueError("consumer producer identity is invalid")

    execution = payload["execution"]
    if not isinstance(execution, Mapping) or set(execution) != {
        "command",
        "workload_contract_id",
        "workload_contract_version",
        "workload_contract_digest",
        "cache_state",
        "started_at_utc",
        "finished_at_utc",
        "platform",
    }:
        raise ValueError("consumer execution field set differs")
    command = execution["command"]
    if (
        not isinstance(command, list)
        or not command
        or any(type(item) is not str or not item for item in command)
        or type(execution["workload_contract_id"]) is not str
        or not _IDENTIFIER.fullmatch(execution["workload_contract_id"])
        or type(execution["workload_contract_version"]) is not int
        or execution["workload_contract_version"] < 1
        or type(execution["workload_contract_digest"]) is not str
        or not _SHA256.fullmatch(execution["workload_contract_digest"])
        or type(execution["cache_state"]) is not str
        or not execution["cache_state"]
    ):
        raise ValueError("consumer execution identity is invalid")
    started = datetime.fromisoformat(
        _utc_timestamp(execution["started_at_utc"], label="consumer start time")
    )
    finished = datetime.fromisoformat(
        _utc_timestamp(execution["finished_at_utc"], label="consumer finish time")
    )
    if finished < started:
        raise ValueError("consumer execution finishes before it starts")
    platform_record = execution["platform"]
    if (
        not isinstance(platform_record, Mapping)
        or set(platform_record)
        != {
            "operating_system",
            "architecture",
            "runtime",
        }
        or any(
            type(value) is not str or not value for value in platform_record.values()
        )
    ):
        raise ValueError("consumer execution platform record is invalid")

    repetitions = payload["repetitions"]
    if not isinstance(repetitions, list) or len(repetitions) < 2:
        raise ValueError("consumer evidence requires at least two repetitions")
    process_ids: set[str] = set()
    for index, repetition in enumerate(repetitions):
        if not isinstance(repetition, Mapping) or set(repetition) != {
            "repetition_index",
            "order",
            "trials",
        }:
            raise ValueError("consumer repetition field set differs")
        expected_order = (
            ["source", "candidate"] if index % 2 == 0 else ["candidate", "source"]
        )
        if (
            repetition["repetition_index"] != index
            or repetition["order"] != expected_order
            or not isinstance(repetition["trials"], list)
            or len(repetition["trials"]) != 2
        ):
            raise ValueError("consumer repetition order or cardinality differs")
        role_set = set()
        for trial in repetition["trials"]:
            process_id = _validate_trial(
                trial,
                repetition_index=index,
                order=expected_order,
            )
            if process_id in process_ids:
                raise ValueError("consumer trial process identity is not fresh")
            process_ids.add(process_id)
            role_set.add(trial["role"])
        if role_set != {"source", "candidate"}:
            raise ValueError("consumer repetition does not cover both roles")

    expected_gate = _gate_facts(
        repetitions=repetitions,
        producer_clean=producer["worktree_clean"],
    )
    if validate_gate and payload["gate"] != expected_gate:
        raise ValueError("consumer evidence gate differs from derived facts")
    if payload["promotion_authorized"] is not False:
        raise ValueError("consumer evidence cannot authorize promotion")


def require_storage_consumer_evidence(
    value: Mapping[str, Any],
    *,
    replay_matrix: bool = False,
) -> None:
    """Deeply validate one receipt and optionally replay its matrix binding."""

    if not isinstance(value, Mapping) or set(value) != {
        "schema_id",
        "schema_version",
        "payload",
        "payload_digest",
    }:
        raise ValueError("consumer evidence envelope field set differs")
    if (
        value["schema_id"] != CONSUMER_EVIDENCE_SCHEMA_ID
        or value["schema_version"] != CONSUMER_EVIDENCE_SCHEMA_VERSION
    ):
        raise ValueError("consumer evidence schema identity differs")
    payload = value["payload"]
    if not isinstance(payload, Mapping):
        raise ValueError("consumer evidence payload must be one object")
    if value["payload_digest"] != canonical_json_sha256(payload):
        raise ValueError("consumer evidence payload digest differs")
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"consumer evidence is not strict JSON: {exc}") from exc
    _validate_consumer_payload_shape(payload, validate_gate=True)
    if replay_matrix:
        binding = payload["matrix_binding"]
        matrix_path = Path(binding["path"])
        matrix, expected_binding = _normalized_matrix_binding(
            stage_id=payload["stage_id"],
            matrix_path=matrix_path,
        )
        del matrix
        if expected_binding != binding:
            raise ValueError("consumer evidence matrix binding differs on replay")


__all__ = [
    "CONSUMER_EVIDENCE_SCHEMA_ID",
    "CONSUMER_EVIDENCE_SCHEMA_VERSION",
    "ConsumerEvidenceScale",
    "StorageConsumer",
    "build_storage_consumer_evidence",
    "require_storage_consumer_evidence",
]
