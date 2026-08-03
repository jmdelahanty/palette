"""Deterministic benchmark-suite planning for derived-analysis storage.

The suite binds real benchmark workloads to an exact logical declaration and
resolved physical-plan receipt.  It plans evidence; it does not create data,
choose a winning profile, or authorize production publication.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from typing import Any, Mapping

from fisheye.shared.zarr.analysis_storage_planning import (
    ANALYSIS_STORAGE_PLAN_SCHEMA_ID,
    ANALYSIS_STORAGE_PLAN_SCHEMA_VERSION,
    AnalysisArrayStoragePlanReceipt,
    AnalysisStoragePlanReceipt,
    analysis_storage_plan_receipt_from_manifest,
)
from fisheye.shared.zarr.benchmark_contracts import (
    EAGER_FULL_READ_V1,
    FULL_SCAN_READ_V1,
    INDEXED_RANGE_READ_V1,
    PER_ROW_RANDOM_READ_V1,
    PUBLISH_VALIDATE_V1,
    WINDOWED_ROWS_READ_V1,
    WRITE_MATERIALIZATION_V1,
    BenchmarkPhase,
    BenchmarkWorkloadContract,
    StorageBenchmarkCase,
)
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from fisheye.shared.zarr.storage_intent import AccessPattern


ANALYSIS_BENCHMARK_SUITE_SCHEMA_ID = "palette.analysis_storage_benchmark_suite"
ANALYSIS_BENCHMARK_SUITE_SCHEMA_VERSION = 1

_IDENTIFIER = re.compile(r"^[a-z0-9][a-z0-9_.-]*$")
_PRIMARY_READ_WORKLOAD = {
    AccessPattern.EAGER: EAGER_FULL_READ_V1,
    AccessPattern.WINDOWED: WINDOWED_ROWS_READ_V1,
    AccessPattern.PER_ROW: PER_ROW_RANDOM_READ_V1,
    AccessPattern.INDEXED: INDEXED_RANGE_READ_V1,
}
_KNOWN_WORKLOADS = {
    workload.workload_id: workload
    for workload in (
        WRITE_MATERIALIZATION_V1,
        PUBLISH_VALIDATE_V1,
        EAGER_FULL_READ_V1,
        WINDOWED_ROWS_READ_V1,
        PER_ROW_RANDOM_READ_V1,
        INDEXED_RANGE_READ_V1,
        FULL_SCAN_READ_V1,
    )
}


@dataclass(frozen=True)
class AnalysisBenchmarkScale:
    """One immutable logical scale included in a benchmark matrix."""

    scale_id: str
    dimensions: tuple[tuple[str, int], ...]
    description: str

    def __post_init__(self) -> None:
        if type(self.scale_id) is not str or not _IDENTIFIER.fullmatch(self.scale_id):
            raise ValueError("scale_id must be one canonical benchmark identifier")
        if (
            not isinstance(self.dimensions, tuple)
            or self.dimensions != tuple(sorted(self.dimensions))
            or len(self.dimensions) != len(dict(self.dimensions))
        ):
            raise ValueError("scale dimensions must be one unique sorted tuple")
        for name, extent in self.dimensions:
            if type(name) is not str or not _IDENTIFIER.fullmatch(name):
                raise ValueError("scale dimension names must be canonical identifiers")
            if type(extent) is not int or extent < 0:
                raise ValueError("scale dimension extents must be nonnegative integers")
        if type(self.description) is not str or not self.description.strip():
            raise ValueError("scale description must be nonempty")

    def as_manifest(self) -> dict[str, object]:
        return {
            "scale_id": self.scale_id,
            "dimensions": dict(self.dimensions),
            "description": self.description,
        }


def _case_prefix(family_id: str, scale_id: str, path: str) -> str:
    return f"{family_id}__{scale_id}__{path.replace('/', '__')}"


def _deterministic_rows(
    *,
    path: str,
    n_rows: int,
    seed: int,
    count: int,
) -> list[int]:
    if n_rows <= 0 or count <= 0:
        return []
    target = min(n_rows, count)
    if target == n_rows:
        return list(range(n_rows))
    rows: list[int] = []
    seen: set[int] = set()
    attempt = 0
    while len(rows) < target:
        payload = f"{seed}\0{path}\0{attempt}".encode()
        row = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % n_rows
        if row not in seen:
            seen.add(row)
            rows.append(row)
        attempt += 1
    return rows


def _primary_read_selection_from_facts(
    *,
    path: str,
    shape: tuple[int, ...],
    access: AccessPattern,
    seed: int,
) -> dict[str, object]:
    n_rows = int(shape[0]) if shape else 1
    if access is AccessPattern.EAGER:
        return {"mode": "whole_array"}
    if access is AccessPattern.WINDOWED:
        if n_rows == 0:
            return {"mode": "bounded_row_windows", "window_rows": 0, "ranges": []}
        window_rows = min(4_096, n_rows)
        maximum_start = n_rows - window_rows
        starts = sorted(
            {(maximum_start * index) // 7 if maximum_start else 0 for index in range(8)}
        )
        return {
            "mode": "bounded_row_windows",
            "window_rows": window_rows,
            "ranges": [[start, start + window_rows] for start in starts],
        }
    rows = _deterministic_rows(
        path=path,
        n_rows=n_rows,
        seed=seed,
        count=128,
    )
    if access is AccessPattern.PER_ROW:
        return {"mode": "random_complete_rows", "row_indices": rows}
    return {
        "mode": "indexed_row_resolution",
        "index_rows": rows,
        "value_ranges": "resolve_from_persisted_index_during_execution",
    }


def _primary_read_selection(
    entry: AnalysisArrayStoragePlanReceipt,
    *,
    seed: int,
) -> dict[str, object]:
    return _primary_read_selection_from_facts(
        path=entry.declaration.path,
        shape=entry.plan.logical_shape,
        access=AccessPattern(entry.plan.access_pattern),
        seed=seed,
    )


def _array_case(
    *,
    family_id: str,
    scale_id: str,
    entry: AnalysisArrayStoragePlanReceipt,
    phase: BenchmarkPhase,
    workload: BenchmarkWorkloadContract,
    selection: Mapping[str, object],
) -> dict[str, object]:
    case = StorageBenchmarkCase(
        case_id=(
            f"{_case_prefix(family_id, scale_id, entry.declaration.path)}"
            f"__{workload.workload_id.rsplit('.', 2)[-2]}"
        ),
        phase=phase,
        array_contract=entry.declaration.contract,
        storage_plan=entry.plan,
        workload=workload,
    )
    return {
        "array_path": entry.declaration.path,
        "case": case.as_manifest(),
        "selection": dict(selection),
    }


def build_analysis_benchmark_suite(
    *,
    family_id: str,
    scale: AnalysisBenchmarkScale,
    storage_receipt: AnalysisStoragePlanReceipt,
    seed: int = 17,
    repetitions: int = 5,
) -> dict[str, object]:
    """Build one strict writer/publisher/reader benchmark-suite manifest."""

    if type(family_id) is not str or not _IDENTIFIER.fullmatch(family_id):
        raise ValueError("family_id must be one canonical benchmark identifier")
    if not isinstance(scale, AnalysisBenchmarkScale):
        raise TypeError("scale must be an AnalysisBenchmarkScale")
    if not isinstance(storage_receipt, AnalysisStoragePlanReceipt):
        raise TypeError("storage_receipt must be an AnalysisStoragePlanReceipt")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be one nonnegative exact integer")
    if type(repetitions) is not int or repetitions < 1:
        raise ValueError("repetitions must be one positive exact integer")
    if dict(scale.dimensions) != dict(storage_receipt.dimensions):
        raise ValueError("benchmark scale dimensions must equal the storage receipt")

    receipt_manifest = storage_receipt.as_manifest()
    cases: list[dict[str, object]] = []
    for entry in storage_receipt.entries:
        cases.extend(
            (
                _array_case(
                    family_id=family_id,
                    scale_id=scale.scale_id,
                    entry=entry,
                    phase=BenchmarkPhase.WRITE,
                    workload=WRITE_MATERIALIZATION_V1,
                    selection={"mode": "complete_materialization"},
                ),
                _array_case(
                    family_id=family_id,
                    scale_id=scale.scale_id,
                    entry=entry,
                    phase=BenchmarkPhase.READ,
                    workload=_PRIMARY_READ_WORKLOAD[
                        AccessPattern(entry.plan.access_pattern)
                    ],
                    selection=_primary_read_selection(entry, seed=seed),
                ),
                _array_case(
                    family_id=family_id,
                    scale_id=scale.scale_id,
                    entry=entry,
                    phase=BenchmarkPhase.READ,
                    workload=FULL_SCAN_READ_V1,
                    selection={"mode": "full_scan"},
                ),
            )
        )
    cases.sort(key=lambda record: str(record["case"]["case_id"]))

    payload: dict[str, object] = {
        "family_id": family_id,
        "scale": scale.as_manifest(),
        "seed": seed,
        "repetitions": repetitions,
        "storage_plan_receipt": receipt_manifest,
        "array_cases": cases,
        "publication_case": {
            "case_id": f"{family_id}__{scale.scale_id}__whole_run__publish_validate",
            "scope": "complete_immutable_run",
            "phase": BenchmarkPhase.PUBLISH.value,
            "workload": PUBLISH_VALIDATE_V1.as_manifest(),
            "storage_plan_receipt_digest": receipt_manifest["payload_digest"],
            "additional_required_metrics": [
                "manifest_validation_seconds",
                "direct_consolidated_comparison_seconds",
                "payload_object_count",
                "apparent_bytes",
                "allocated_bytes",
                "peak_rss_bytes",
            ],
        },
        "execution_policy": {
            "benchmark_only": True,
            "selector_eligible": False,
            "registry_registered": False,
            "node_local_compute": True,
            "publish_method": "exclusive_copy_validate_atomic_rename",
            "candidate_order": "rotate_by_repetition",
            "cache_state_must_be_declared": True,
            "decoded_equality_required": True,
            "direct_consolidated_metadata_equivalence_required": True,
            "production_mutation_authorized": False,
        },
    }
    result = {
        "schema_id": ANALYSIS_BENCHMARK_SUITE_SCHEMA_ID,
        "schema_version": ANALYSIS_BENCHMARK_SUITE_SCHEMA_VERSION,
        "payload": payload,
        "payload_digest": canonical_json_sha256(payload),
    }
    require_analysis_benchmark_suite_manifest(result)
    return result


def _manifest_error(message: str) -> ValueError:
    return ValueError(f"Invalid analysis benchmark suite: {message}")


def require_analysis_benchmark_suite_manifest(value: Mapping[str, Any]) -> None:
    """Deeply validate a suite, including rehashed linkage tampering."""

    if not isinstance(value, Mapping) or set(value) != {
        "schema_id",
        "schema_version",
        "payload",
        "payload_digest",
    }:
        raise _manifest_error("unexpected envelope field set")
    if value["schema_id"] != ANALYSIS_BENCHMARK_SUITE_SCHEMA_ID:
        raise _manifest_error("unsupported schema ID")
    if value["schema_version"] != ANALYSIS_BENCHMARK_SUITE_SCHEMA_VERSION:
        raise _manifest_error("unsupported schema version")
    payload = value["payload"]
    if not isinstance(payload, Mapping) or set(payload) != {
        "family_id",
        "scale",
        "seed",
        "repetitions",
        "storage_plan_receipt",
        "array_cases",
        "publication_case",
        "execution_policy",
    }:
        raise _manifest_error("unexpected payload field set")
    if value["payload_digest"] != canonical_json_sha256(payload):
        raise _manifest_error("payload digest mismatch")
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise _manifest_error(f"not strict JSON: {exc}") from exc

    family_id = payload["family_id"]
    if type(family_id) is not str or not _IDENTIFIER.fullmatch(family_id):
        raise _manifest_error("noncanonical family ID")
    scale = payload["scale"]
    if not isinstance(scale, Mapping) or set(scale) != {
        "scale_id",
        "dimensions",
        "description",
    }:
        raise _manifest_error("invalid scale declaration")
    scale_id = scale["scale_id"]
    if type(scale_id) is not str or not _IDENTIFIER.fullmatch(scale_id):
        raise _manifest_error("noncanonical scale ID")
    if type(payload["seed"]) is not int or payload["seed"] < 0:
        raise _manifest_error("invalid seed")
    if type(payload["repetitions"]) is not int or payload["repetitions"] < 1:
        raise _manifest_error("invalid repetition count")

    receipt = payload["storage_plan_receipt"]
    if not isinstance(receipt, Mapping) or set(receipt) != {
        "schema_id",
        "schema_version",
        "payload",
        "payload_digest",
    }:
        raise _manifest_error("invalid storage-plan receipt envelope")
    if (
        receipt["schema_id"] != ANALYSIS_STORAGE_PLAN_SCHEMA_ID
        or receipt["schema_version"] != ANALYSIS_STORAGE_PLAN_SCHEMA_VERSION
        or receipt["payload_digest"] != canonical_json_sha256(receipt["payload"])
    ):
        raise _manifest_error("invalid storage-plan receipt identity or digest")
    try:
        analysis_storage_plan_receipt_from_manifest(receipt)
    except (TypeError, ValueError) as exc:
        raise _manifest_error(f"storage-plan receipt is not executable: {exc}") from exc
    receipt_payload = receipt["payload"]
    if not isinstance(receipt_payload, Mapping):
        raise _manifest_error("storage-plan receipt payload is not an object")
    if scale["dimensions"] != receipt_payload.get("dimensions"):
        raise _manifest_error("scale and storage-plan dimensions differ")
    receipt_arrays = receipt_payload.get("arrays")
    if not isinstance(receipt_arrays, list):
        raise _manifest_error("storage-plan receipt arrays are missing")
    plan_by_path: dict[str, Mapping[str, Any]] = {}
    schema_by_path: dict[str, Mapping[str, Any]] = {}
    for record in receipt_arrays:
        if (
            not isinstance(record, Mapping)
            or not isinstance(record.get("plan"), Mapping)
            or not isinstance(record.get("declaration"), Mapping)
        ):
            raise _manifest_error("invalid storage-plan array entry")
        path = record.get("path")
        if type(path) is not str or path in plan_by_path:
            raise _manifest_error("invalid or duplicate storage-plan array path")
        plan_by_path[path] = record["plan"]
        logical_contract = record["declaration"].get("logical_contract")
        if not isinstance(logical_contract, Mapping):
            raise _manifest_error("storage-plan declaration lacks a logical contract")
        schema_by_path[path] = logical_contract

    cases = payload["array_cases"]
    if not isinstance(cases, list):
        raise _manifest_error("array_cases must be an array")
    case_ids: set[str] = set()
    workload_ids_by_path: dict[str, set[str]] = {path: set() for path in plan_by_path}
    case_count_by_path: dict[str, int] = {path: 0 for path in plan_by_path}
    for record in cases:
        if not isinstance(record, Mapping) or set(record) != {
            "array_path",
            "case",
            "selection",
        }:
            raise _manifest_error("invalid array-case field set")
        path = record["array_path"]
        case = record["case"]
        if path not in plan_by_path or not isinstance(case, Mapping):
            raise _manifest_error("array case references an unknown array")
        case_id = case.get("case_id")
        if type(case_id) is not str or case_id in case_ids:
            raise _manifest_error("invalid or duplicate case ID")
        case_ids.add(case_id)
        case_count_by_path[path] += 1
        if case.get("storage_plan") != plan_by_path[path]:
            raise _manifest_error("case storage plan differs from its receipt")
        schema = schema_by_path[path]
        if case.get("logical_schema") != {
            "id": schema.get("schema_id"),
            "version": schema.get("schema_version"),
        }:
            raise _manifest_error("case logical schema differs from its declaration")
        workload = case.get("workload")
        if not isinstance(workload, Mapping):
            raise _manifest_error("case workload is missing")
        workload_id = workload.get("workload_id")
        expected_workload = _KNOWN_WORKLOADS.get(workload_id)
        if expected_workload is None or workload != expected_workload.as_manifest():
            raise _manifest_error("case workload is unknown or noncanonical")
        phase = case.get("phase")
        if phase not in workload.get("phases", []):
            raise _manifest_error("case phase is incompatible with its workload")
        plan_access = plan_by_path[path].get("access_pattern")
        if plan_access not in workload.get("access_patterns", []):
            raise _manifest_error("case workload is incompatible with array access")
        expected_phase = (
            BenchmarkPhase.WRITE.value
            if workload_id == WRITE_MATERIALIZATION_V1.workload_id
            else BenchmarkPhase.READ.value
        )
        if phase != expected_phase:
            raise _manifest_error("case phase differs from its required suite phase")
        expected_case_id = (
            f"{_case_prefix(family_id, scale_id, path)}"
            f"__{workload_id.rsplit('.', 2)[-2]}"
        )
        if case_id != expected_case_id:
            raise _manifest_error("case ID is not canonical for its workload")
        if workload_id == WRITE_MATERIALIZATION_V1.workload_id:
            expected_selection: Mapping[str, object] = {
                "mode": "complete_materialization"
            }
        elif workload_id == FULL_SCAN_READ_V1.workload_id:
            expected_selection = {"mode": "full_scan"}
        else:
            expected_selection = _primary_read_selection_from_facts(
                path=path,
                shape=tuple(plan_by_path[path]["logical_shape"]),
                access=AccessPattern(plan_access),
                seed=payload["seed"],
            )
        if record["selection"] != expected_selection:
            raise _manifest_error("case selection differs from deterministic v1")
        workload_ids_by_path[path].add(workload_id)
    for path, workload_ids in workload_ids_by_path.items():
        access = AccessPattern(plan_by_path[path]["access_pattern"])
        expected = {
            WRITE_MATERIALIZATION_V1.workload_id,
            FULL_SCAN_READ_V1.workload_id,
            _PRIMARY_READ_WORKLOAD[access].workload_id,
        }
        if workload_ids != expected or case_count_by_path[path] != 3:
            raise _manifest_error(f"array {path!r} does not have the exact workloads")

    publication = payload["publication_case"]
    if not isinstance(publication, Mapping) or set(publication) != {
        "case_id",
        "scope",
        "phase",
        "workload",
        "storage_plan_receipt_digest",
        "additional_required_metrics",
    }:
        raise _manifest_error("publication case has an unexpected field set")
    if publication.get("storage_plan_receipt_digest") != receipt.get("payload_digest"):
        raise _manifest_error("publication case is not bound to the storage plan")
    if (
        publication.get("case_id")
        != f"{family_id}__{scale_id}__whole_run__publish_validate"
        or publication.get("scope") != "complete_immutable_run"
        or publication.get("phase") != BenchmarkPhase.PUBLISH.value
    ):
        raise _manifest_error("publication case identity is noncanonical")
    if publication.get("workload") != PUBLISH_VALIDATE_V1.as_manifest():
        raise _manifest_error("publication workload is noncanonical")
    if publication.get("additional_required_metrics") != [
        "manifest_validation_seconds",
        "direct_consolidated_comparison_seconds",
        "payload_object_count",
        "apparent_bytes",
        "allocated_bytes",
        "peak_rss_bytes",
    ]:
        raise _manifest_error("publication metrics differ from v1")
    policy = payload["execution_policy"]
    required_policy = {
        "benchmark_only": True,
        "selector_eligible": False,
        "registry_registered": False,
        "node_local_compute": True,
        "publish_method": "exclusive_copy_validate_atomic_rename",
        "candidate_order": "rotate_by_repetition",
        "cache_state_must_be_declared": True,
        "decoded_equality_required": True,
        "direct_consolidated_metadata_equivalence_required": True,
        "production_mutation_authorized": False,
    }
    if policy != required_policy:
        raise _manifest_error("execution safety policy differs from v1")


__all__ = [
    "ANALYSIS_BENCHMARK_SUITE_SCHEMA_ID",
    "ANALYSIS_BENCHMARK_SUITE_SCHEMA_VERSION",
    "AnalysisBenchmarkScale",
    "build_analysis_benchmark_suite",
    "require_analysis_benchmark_suite_manifest",
]
