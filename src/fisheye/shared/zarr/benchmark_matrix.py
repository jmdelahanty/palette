"""Pure, versioned planning for byte-based storage benchmark matrices.

The matrix planner never opens or writes a Zarr store. Dataset adapters provide
resolved stage-plan manifests; this module fingerprints their effective
physical layouts, removes duplicate labels, and emits deterministic balanced
trial orders and exact destinations.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
import hashlib
import json
from pathlib import Path
import random
from typing import Callable, Iterable, Mapping, Sequence

from fisheye.shared.zarr.benchmark_contracts import BenchmarkWorkloadContract
from fisheye.shared.zarr.storage_intent import AccessPattern


MATRIX_SCHEMA_ID = "palette.storage_benchmark_matrix"
MATRIX_SCHEMA_VERSION = 1
PHYSICAL_FINGERPRINT_SCHEMA_ID = "palette.storage_physical_plan_fingerprint"
PHYSICAL_FINGERPRINT_SCHEMA_VERSION = 1


class BenchmarkLayout(str, Enum):
    """Physical object layout requested for an immutable candidate."""

    REGULAR = "regular"
    SHARDED = "sharded"


def _require_identifier(value: str, *, field: str) -> str:
    identifier = str(value).strip()
    if not identifier:
        raise ValueError(f"{field} cannot be empty.")
    if any(character not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for character in identifier):
        raise ValueError(
            f"{field} must contain only lowercase letters, digits, '_' or '-'."
        )
    return identifier


def _canonical_sha256(value: Mapping[str, object]) -> str:
    payload = json.dumps(
        value,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class BenchmarkScale:
    """One named logical dataset scale with exact dimensions."""

    scale_id: str
    dimensions: tuple[tuple[str, int], ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "scale_id",
            _require_identifier(self.scale_id, field="scale_id"),
        )
        dimensions = tuple(
            (str(name).strip(), int(value)) for name, value in self.dimensions
        )
        if not dimensions or any(not name for name, _value in dimensions):
            raise ValueError("Benchmark scale dimensions require non-empty names.")
        if len({name for name, _value in dimensions}) != len(dimensions):
            raise ValueError("Benchmark scale dimension names must be unique.")
        if any(value < 0 for _name, value in dimensions):
            raise ValueError("Benchmark scale dimensions cannot be negative.")
        object.__setattr__(self, "dimensions", dimensions)

    @classmethod
    def from_mapping(
        cls,
        scale_id: str,
        dimensions: Mapping[str, int],
    ) -> BenchmarkScale:
        """Create a scale while preserving the adapter's dimension order."""

        return cls(scale_id=scale_id, dimensions=tuple(dimensions.items()))

    @property
    def dimension_map(self) -> dict[str, int]:
        return dict(self.dimensions)

    def as_manifest(self) -> dict[str, object]:
        return {
            "schema_id": "palette.storage_benchmark_scale",
            "schema_version": 1,
            "scale_id": self.scale_id,
            "dimensions": self.dimension_map,
        }


@dataclass(frozen=True)
class StorageCandidateRequest:
    """One byte-budget request; raw chunk/shard row overrides do not exist."""

    layout: BenchmarkLayout
    target_chunk_bytes: int
    target_shard_bytes: int | None = None
    target_chunk_bytes_by_access: tuple[tuple[str, int], ...] = ()

    def __post_init__(self) -> None:
        layout = BenchmarkLayout(self.layout)
        object.__setattr__(self, "layout", layout)
        chunk_bytes = int(self.target_chunk_bytes)
        if chunk_bytes <= 0:
            raise ValueError("target_chunk_bytes must be positive.")
        object.__setattr__(self, "target_chunk_bytes", chunk_bytes)
        normalized_overrides: dict[str, int] = {}
        for raw_access, raw_target in self.target_chunk_bytes_by_access:
            access = AccessPattern(raw_access).value
            if access in normalized_overrides:
                raise ValueError(
                    f"Duplicate access-specific chunk target for {access!r}."
                )
            target = int(raw_target)
            if target <= 0:
                raise ValueError(
                    "Access-specific target chunk bytes must be positive."
                )
            normalized_overrides[access] = target
        object.__setattr__(
            self,
            "target_chunk_bytes_by_access",
            tuple(
                (access.value, normalized_overrides[access.value])
                for access in AccessPattern
                if access.value in normalized_overrides
            ),
        )
        if layout is BenchmarkLayout.REGULAR:
            if self.target_shard_bytes is not None:
                raise ValueError(
                    "Regular candidates cannot declare target_shard_bytes."
                )
            return
        if self.target_shard_bytes is None:
            raise ValueError("Sharded candidates require target_shard_bytes.")
        shard_bytes = int(self.target_shard_bytes)
        if shard_bytes < chunk_bytes:
            raise ValueError(
                "target_shard_bytes cannot be smaller than target_chunk_bytes."
            )
        if any(
            target > shard_bytes
            for _access, target in self.target_chunk_bytes_by_access
        ):
            raise ValueError(
                "target_shard_bytes cannot be smaller than an access-specific "
                "target chunk size."
            )
        object.__setattr__(self, "target_shard_bytes", shard_bytes)

    @property
    def label(self) -> str:
        chunk = f"chunk_{self.target_chunk_bytes}"
        access = "".join(
            f"__{name}_chunk_{value}"
            for name, value in self.target_chunk_bytes_by_access
        )
        if self.layout is BenchmarkLayout.REGULAR:
            return f"regular__{chunk}{access}"
        return (
            f"sharded__{chunk}{access}__shard_{self.target_shard_bytes}"
        )

    def as_manifest(self) -> dict[str, object]:
        return {
            "schema_id": "palette.storage_benchmark_candidate_request",
            "schema_version": 2,
            "label": self.label,
            "layout": self.layout.value,
            "target_chunk_bytes": self.target_chunk_bytes,
            "target_shard_bytes": self.target_shard_bytes,
            "target_chunk_bytes_by_access": dict(
                self.target_chunk_bytes_by_access
            ),
            "row_overrides_supported": False,
        }


@dataclass(frozen=True)
class MatrixWorkload:
    """A versioned workload included in a matrix execution."""

    workload_id: str
    phases: tuple[str, ...]
    access_patterns: tuple[str, ...]

    @classmethod
    def from_contract(
        cls,
        contract: BenchmarkWorkloadContract,
    ) -> MatrixWorkload:
        manifest = contract.as_manifest()
        return cls(
            workload_id=contract.workload_id,
            phases=tuple(str(value) for value in manifest["phases"]),
            access_patterns=tuple(
                str(value) for value in manifest["access_patterns"]
            ),
        )

    def as_manifest(self) -> dict[str, object]:
        return {
            "schema_id": "palette.storage_benchmark_matrix_workload",
            "schema_version": 1,
            "workload_id": self.workload_id,
            "phases": list(self.phases),
            "access_patterns": list(self.access_patterns),
        }


def physical_stage_plan_payload(
    *,
    scale: BenchmarkScale,
    stage_plan: Mapping[str, object],
) -> dict[str, object]:
    """Project a resolved stage plan to its effective physical contract."""

    storage_profile = stage_plan.get("storage_profile")
    arrays = stage_plan.get("arrays")
    logical_schema = stage_plan.get("logical_stage_schema")
    if not isinstance(storage_profile, Mapping):
        raise ValueError("Stage plan is missing storage_profile.")
    if not isinstance(logical_schema, Mapping):
        raise ValueError("Stage plan is missing logical_stage_schema.")
    if not isinstance(arrays, list) or not arrays:
        raise ValueError("Stage plan must contain resolved arrays.")

    physical_arrays: list[dict[str, object]] = []
    for index, item in enumerate(arrays):
        if not isinstance(item, Mapping) or not isinstance(item.get("plan"), Mapping):
            raise ValueError(f"Stage plan arrays[{index}] lacks a resolved plan.")
        plan = item["plan"]
        physical_arrays.append(
            {
                "path": item.get("path"),
                "policy_version": plan.get("policy_version"),
                "codec_profile_id": plan.get("codec_profile_id"),
                "logical_schema_id": plan.get("logical_schema_id"),
                "logical_schema_version": plan.get("logical_schema_version"),
                "logical_shape": plan.get("logical_shape"),
                "logical_dtype": plan.get("logical_dtype"),
                "access_unit_shape": plan.get("access_unit_shape"),
                "growth_axis": plan.get("growth_axis"),
                "shard_axes": plan.get("shard_axes"),
                "access_pattern": plan.get("access_pattern"),
                "write_mode": plan.get("write_mode"),
                "chunk_shape": plan.get("chunk_shape"),
                "shard_shape": plan.get("shard_shape"),
                "write_ownership": plan.get("write_ownership"),
            }
        )
    return {
        "schema_id": PHYSICAL_FINGERPRINT_SCHEMA_ID,
        "schema_version": PHYSICAL_FINGERPRINT_SCHEMA_VERSION,
        "zarr_format": 3,
        "scale": scale.as_manifest(),
        "logical_stage_schema": dict(logical_schema),
        "codec_profile_id": storage_profile.get("codec_profile_id"),
        "metadata_open_contract": stage_plan.get("metadata_open_contract"),
        "arrays": physical_arrays,
    }


@dataclass(frozen=True)
class ResolvedStorageCandidate:
    """One retained unique physical stage plan."""

    candidate_id: str
    scale_id: str
    request: StorageCandidateRequest
    requested_labels: tuple[str, ...]
    physical_fingerprint: str
    stage_plan: Mapping[str, object]

    def as_manifest(self) -> dict[str, object]:
        return {
            "schema_id": "palette.storage_benchmark_resolved_candidate",
            "schema_version": 1,
            "candidate_id": self.candidate_id,
            "scale_id": self.scale_id,
            "request": self.request.as_manifest(),
            "requested_labels": list(self.requested_labels),
            "physical_fingerprint": self.physical_fingerprint,
            "stage_plan": dict(self.stage_plan),
        }


@dataclass(frozen=True)
class DuplicateStorageCandidate:
    """One requested label removed because its physical plan already exists."""

    scale_id: str
    removed_label: str
    retained_candidate_id: str
    physical_fingerprint: str

    def as_manifest(self) -> dict[str, object]:
        return {
            "scale_id": self.scale_id,
            "removed_label": self.removed_label,
            "retained_candidate_id": self.retained_candidate_id,
            "physical_fingerprint": self.physical_fingerprint,
            "reason": "identical_effective_physical_stage_plan",
        }


@dataclass(frozen=True)
class BenchmarkTrial:
    """One ordered candidate execution and its exclusive destination."""

    position: int
    candidate_id: str
    layout: BenchmarkLayout
    destination: str
    destination_collision: bool

    def as_manifest(self) -> dict[str, object]:
        return {
            "position": self.position,
            "candidate_id": self.candidate_id,
            "layout": self.layout.value,
            "destination": self.destination,
            "destination_collision": self.destination_collision,
            "destination_mode": "exclusive_create",
        }


@dataclass(frozen=True)
class BenchmarkRepetition:
    """One balanced execution block for a single scale."""

    scale_id: str
    repetition_index: int
    trials: tuple[BenchmarkTrial, ...]

    def as_manifest(self) -> dict[str, object]:
        return {
            "schema_id": "palette.storage_benchmark_repetition",
            "schema_version": 1,
            "scale_id": self.scale_id,
            "repetition_index": self.repetition_index,
            "trials": [trial.as_manifest() for trial in self.trials],
        }


@dataclass(frozen=True)
class StorageBenchmarkMatrix:
    """Complete plan-only result for one benchmark family."""

    matrix_id: str
    seed: int
    scales: tuple[BenchmarkScale, ...]
    workloads: tuple[MatrixWorkload, ...]
    candidates: tuple[ResolvedStorageCandidate, ...]
    duplicates: tuple[DuplicateStorageCandidate, ...]
    repetitions: tuple[BenchmarkRepetition, ...]
    correctness_gates: Mapping[str, object]
    performance_tolerances: Mapping[str, object]

    @property
    def collision_count(self) -> int:
        return sum(
            trial.destination_collision
            for repetition in self.repetitions
            for trial in repetition.trials
        )

    def as_manifest(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema_id": MATRIX_SCHEMA_ID,
            "schema_version": MATRIX_SCHEMA_VERSION,
            "matrix_id": self.matrix_id,
            "seed": self.seed,
            "scales": [scale.as_manifest() for scale in self.scales],
            "workloads": [workload.as_manifest() for workload in self.workloads],
            "candidates": [candidate.as_manifest() for candidate in self.candidates],
            "duplicates": [duplicate.as_manifest() for duplicate in self.duplicates],
            "repetitions": [
                repetition.as_manifest() for repetition in self.repetitions
            ],
            "correctness_gates": dict(self.correctness_gates),
            "performance_tolerances": dict(self.performance_tolerances),
            "summary": {
                "requested_candidate_labels": (
                    len(self.candidates) + len(self.duplicates)
                ),
                "unique_physical_candidates": len(self.candidates),
                "removed_duplicate_labels": len(self.duplicates),
                "planned_trials": sum(
                    len(repetition.trials) for repetition in self.repetitions
                ),
                "destination_collisions": self.collision_count,
                "payload_io_performed": False,
            },
        }
        payload["matrix_fingerprint"] = _canonical_sha256(payload)
        return payload


def require_storage_benchmark_matrix_manifest(
    manifest: Mapping[str, object],
) -> None:
    """Reject a serialized benchmark matrix whose declared digest has drifted."""

    if manifest.get("schema_id") != MATRIX_SCHEMA_ID:
        raise ValueError("Unsupported storage benchmark matrix schema.")
    if manifest.get("schema_version") != MATRIX_SCHEMA_VERSION:
        raise ValueError("Unsupported storage benchmark matrix schema version.")
    declared = manifest.get("matrix_fingerprint")
    if not isinstance(declared, str) or not declared:
        raise ValueError("Storage benchmark matrix lacks a fingerprint.")
    payload = dict(manifest)
    payload.pop("matrix_fingerprint", None)
    actual = _canonical_sha256(payload)
    if actual != declared:
        raise ValueError(
            "Storage benchmark matrix fingerprint does not match its contents."
        )
    _require_identifier(str(manifest.get("matrix_id", "")), field="matrix_id")
    scales = manifest.get("scales")
    candidates = manifest.get("candidates")
    repetitions = manifest.get("repetitions")
    if not isinstance(scales, list) or not isinstance(candidates, list):
        raise ValueError("Storage benchmark matrix lacks scales or candidates.")
    if not isinstance(repetitions, list):
        raise ValueError("Storage benchmark matrix lacks repetitions.")
    for scale in scales:
        if not isinstance(scale, Mapping):
            raise ValueError("Storage benchmark matrix scale must be an object.")
        _require_identifier(str(scale.get("scale_id", "")), field="scale_id")
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            raise ValueError("Storage benchmark matrix candidate must be an object.")
        _require_identifier(
            str(candidate.get("candidate_id", "")),
            field="candidate_id",
        )
    for repetition in repetitions:
        if not isinstance(repetition, Mapping):
            raise ValueError("Storage benchmark matrix repetition must be an object.")
        _require_identifier(
            str(repetition.get("scale_id", "")),
            field="scale_id",
        )
        repetition_index = repetition.get("repetition_index")
        if type(repetition_index) is not int or repetition_index < 0:
            raise ValueError("repetition_index must be a nonnegative exact integer.")
        trials = repetition.get("trials")
        if not isinstance(trials, list):
            raise ValueError("Storage benchmark repetition lacks trials.")
        for trial in trials:
            if not isinstance(trial, Mapping):
                raise ValueError("Storage benchmark trial must be an object.")
            _require_identifier(
                str(trial.get("candidate_id", "")),
                field="candidate_id",
            )


def _balanced_order(
    candidates: Sequence[ResolvedStorageCandidate],
    *,
    repetition_index: int,
    seed: int,
) -> tuple[ResolvedStorageCandidate, ...]:
    buckets: dict[BenchmarkLayout, list[ResolvedStorageCandidate]] = {
        layout: [] for layout in BenchmarkLayout
    }
    for candidate in candidates:
        buckets[candidate.request.layout].append(candidate)

    for layout, bucket in buckets.items():
        random.Random(f"{seed}:{layout.value}").shuffle(bucket)
        if bucket:
            offset = repetition_index % len(bucket)
            buckets[layout] = bucket[offset:] + bucket[:offset]

    merged: list[ResolvedStorageCandidate] = []
    active_layouts = [layout for layout in BenchmarkLayout if buckets[layout]]
    index = 0
    while any(buckets.values()):
        layout = active_layouts[index % len(active_layouts)]
        if buckets[layout]:
            merged.append(buckets[layout].pop(0))
        index += 1
    if repetition_index % 2:
        merged.reverse()
    return tuple(merged)


def plan_storage_benchmark_matrix(
    *,
    matrix_id: str,
    scales: Sequence[BenchmarkScale],
    candidate_requests: Sequence[StorageCandidateRequest],
    workloads: Sequence[MatrixWorkload],
    repetitions: int,
    repetition_start: int = 0,
    seed: int,
    destination_root: Path,
    resolve_stage_plan: Callable[
        [BenchmarkScale, StorageCandidateRequest], Mapping[str, object]
    ],
    occupied_destinations: Iterable[Path] = (),
    correctness_gates: Mapping[str, object],
    performance_tolerances: Mapping[str, object],
) -> StorageBenchmarkMatrix:
    """Resolve, deduplicate, and order a storage matrix without payload I/O."""

    resolved_matrix_id = _require_identifier(matrix_id, field="matrix_id")
    if type(repetitions) is not int or repetitions <= 0:
        raise ValueError("repetitions must be positive.")
    if type(repetition_start) is not int or repetition_start < 0:
        raise ValueError("repetition_start must be a nonnegative exact integer.")
    scales_tuple = tuple(scales)
    requests_tuple = tuple(candidate_requests)
    workloads_tuple = tuple(workloads)
    if not scales_tuple or not requests_tuple or not workloads_tuple:
        raise ValueError("Matrix scales, candidates, and workloads cannot be empty.")
    if len({scale.scale_id for scale in scales_tuple}) != len(scales_tuple):
        raise ValueError("Matrix scale IDs must be unique.")
    if len({request.label for request in requests_tuple}) != len(requests_tuple):
        raise ValueError("Matrix candidate labels must be unique.")

    destination_base = destination_root.expanduser().resolve() / resolved_matrix_id
    occupied = {path.expanduser().resolve() for path in occupied_destinations}
    retained: list[ResolvedStorageCandidate] = []
    duplicates: list[DuplicateStorageCandidate] = []
    by_fingerprint: dict[str, int] = {}
    for scale in scales_tuple:
        for request in requests_tuple:
            stage_plan = dict(resolve_stage_plan(scale, request))
            physical_payload = physical_stage_plan_payload(
                scale=scale,
                stage_plan=stage_plan,
            )
            fingerprint = _canonical_sha256(physical_payload)
            retained_index = by_fingerprint.get(fingerprint)
            if retained_index is not None:
                original = retained[retained_index]
                retained[retained_index] = replace(
                    original,
                    requested_labels=original.requested_labels + (request.label,),
                )
                duplicates.append(
                    DuplicateStorageCandidate(
                        scale_id=scale.scale_id,
                        removed_label=request.label,
                        retained_candidate_id=original.candidate_id,
                        physical_fingerprint=fingerprint,
                    )
                )
                continue
            candidate_id = (
                f"{scale.scale_id}__{request.label}__{fingerprint[:12]}"
            )
            by_fingerprint[fingerprint] = len(retained)
            retained.append(
                ResolvedStorageCandidate(
                    candidate_id=candidate_id,
                    scale_id=scale.scale_id,
                    request=request,
                    requested_labels=(request.label,),
                    physical_fingerprint=fingerprint,
                    stage_plan=stage_plan,
                )
            )

    repetition_models: list[BenchmarkRepetition] = []
    for scale in scales_tuple:
        scale_candidates = [
            candidate for candidate in retained if candidate.scale_id == scale.scale_id
        ]
        for repetition_index in range(
            repetition_start,
            repetition_start + repetitions,
        ):
            ordered = _balanced_order(
                scale_candidates,
                repetition_index=repetition_index,
                seed=int(seed),
            )
            trials: list[BenchmarkTrial] = []
            for position, candidate in enumerate(ordered):
                destination = (
                    destination_base
                    / scale.scale_id
                    / f"repetition_{repetition_index:03d}"
                    / f"{candidate.candidate_id}.zarr"
                )
                trials.append(
                    BenchmarkTrial(
                        position=position,
                        candidate_id=candidate.candidate_id,
                        layout=candidate.request.layout,
                        destination=str(destination),
                        destination_collision=destination in occupied,
                    )
                )
            repetition_models.append(
                BenchmarkRepetition(
                    scale_id=scale.scale_id,
                    repetition_index=repetition_index,
                    trials=tuple(trials),
                )
            )

    return StorageBenchmarkMatrix(
        matrix_id=resolved_matrix_id,
        seed=int(seed),
        scales=scales_tuple,
        workloads=workloads_tuple,
        candidates=tuple(retained),
        duplicates=tuple(duplicates),
        repetitions=tuple(repetition_models),
        correctness_gates=dict(correctness_gates),
        performance_tolerances=dict(performance_tolerances),
    )


__all__ = [
    "MATRIX_SCHEMA_ID",
    "MATRIX_SCHEMA_VERSION",
    "BenchmarkLayout",
    "BenchmarkRepetition",
    "BenchmarkScale",
    "BenchmarkTrial",
    "DuplicateStorageCandidate",
    "MatrixWorkload",
    "ResolvedStorageCandidate",
    "StorageBenchmarkMatrix",
    "StorageCandidateRequest",
    "physical_stage_plan_payload",
    "plan_storage_benchmark_matrix",
    "require_storage_benchmark_matrix_manifest",
]
