"""Common schema-linked contracts for Palette storage benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping

from fisheye.shared.zarr.array_contracts import ArrayContract
from fisheye.shared.zarr.storage_intent import AccessPattern, StoragePlan


class BenchmarkPhase(str, Enum):
    WRITE = "write"
    PUBLISH = "publish"
    READ = "read"


@dataclass(frozen=True)
class BenchmarkWorkloadContract:
    """Stable identity and compatibility rules for one benchmark workload."""

    workload_id: str
    phases: tuple[BenchmarkPhase, ...]
    access_patterns: tuple[AccessPattern, ...]
    description: str
    required_metrics: tuple[str, ...]

    def __post_init__(self) -> None:
        workload_id = str(self.workload_id).strip()
        if not workload_id:
            raise ValueError("workload_id cannot be empty.")
        object.__setattr__(self, "workload_id", workload_id)
        object.__setattr__(
            self,
            "phases",
            tuple(BenchmarkPhase(phase) for phase in self.phases),
        )
        object.__setattr__(
            self,
            "access_patterns",
            tuple(AccessPattern(access) for access in self.access_patterns),
        )
        if not self.phases:
            raise ValueError("Benchmark workload must support at least one phase.")
        if not self.access_patterns:
            raise ValueError(
                "Array benchmark workload must support at least one access pattern."
            )
        if not self.required_metrics:
            raise ValueError("Benchmark workload must declare required metrics.")

    def as_manifest(self) -> dict[str, object]:
        return {
            "workload_id": self.workload_id,
            "phases": [phase.value for phase in self.phases],
            "access_patterns": [access.value for access in self.access_patterns],
            "description": self.description,
            "required_metrics": list(self.required_metrics),
        }


ALL_ACCESS_PATTERNS = tuple(AccessPattern)

WRITE_MATERIALIZATION_V1 = BenchmarkWorkloadContract(
    workload_id="palette.storage_workload.write_materialization.v1",
    phases=(BenchmarkPhase.WRITE,),
    access_patterns=ALL_ACCESS_PATTERNS,
    description="Create and completely materialize a planned array.",
    required_metrics=(
        "write_seconds",
        "logical_bytes",
        "physical_bytes",
        "peak_rss_bytes",
        "payload_object_count",
    ),
)

PUBLISH_VALIDATE_V1 = BenchmarkWorkloadContract(
    workload_id="palette.storage_workload.publish_validate.v1",
    phases=(BenchmarkPhase.PUBLISH,),
    access_patterns=ALL_ACCESS_PATTERNS,
    description="Copy, validate, consolidate, and publish an immutable array.",
    required_metrics=(
        "copy_seconds",
        "validation_seconds",
        "consolidation_seconds",
        "publication_seconds",
    ),
)

EAGER_FULL_READ_V1 = BenchmarkWorkloadContract(
    workload_id="palette.storage_workload.eager_full_read.v1",
    phases=(BenchmarkPhase.READ,),
    access_patterns=(AccessPattern.EAGER,),
    description="Read a complete eager array.",
    required_metrics=("read_seconds", "logical_bytes", "transferred_bytes"),
)

WINDOWED_ROWS_READ_V1 = BenchmarkWorkloadContract(
    workload_id="palette.storage_workload.windowed_rows_read.v1",
    phases=(BenchmarkPhase.READ,),
    access_patterns=(AccessPattern.WINDOWED,),
    description="Read bounded contiguous row windows.",
    required_metrics=(
        "read_seconds",
        "requested_rows",
        "decoded_bytes",
        "transferred_bytes",
        "request_count",
    ),
)

PER_ROW_RANDOM_READ_V1 = BenchmarkWorkloadContract(
    workload_id="palette.storage_workload.per_row_random_read.v1",
    phases=(BenchmarkPhase.READ,),
    access_patterns=(AccessPattern.PER_ROW,),
    description="Read random complete row/component access units.",
    required_metrics=(
        "read_seconds",
        "requested_rows",
        "decoded_bytes",
        "transferred_bytes",
        "request_count",
    ),
)

INDEXED_RANGE_READ_V1 = BenchmarkWorkloadContract(
    workload_id="palette.storage_workload.indexed_range_read.v1",
    phases=(BenchmarkPhase.READ,),
    access_patterns=(AccessPattern.INDEXED,),
    description="Resolve indexes and read the selected flat value ranges.",
    required_metrics=(
        "read_seconds",
        "selected_value_count",
        "decoded_bytes",
        "transferred_bytes",
        "request_count",
    ),
)

FULL_SCAN_READ_V1 = BenchmarkWorkloadContract(
    workload_id="palette.storage_workload.full_scan_read.v1",
    phases=(BenchmarkPhase.READ,),
    access_patterns=ALL_ACCESS_PATTERNS,
    description="Read every logical value in deterministic order.",
    required_metrics=("read_seconds", "logical_bytes", "peak_rss_bytes"),
)


CORE_BENCHMARK_WORKLOADS = {
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
class StorageBenchmarkCase:
    """One schema-, plan-, phase-, and workload-locked benchmark case."""

    case_id: str
    phase: BenchmarkPhase
    array_contract: ArrayContract
    storage_plan: StoragePlan
    workload: BenchmarkWorkloadContract

    def __post_init__(self) -> None:
        case_id = str(self.case_id).strip()
        if not case_id:
            raise ValueError("case_id cannot be empty.")
        object.__setattr__(self, "case_id", case_id)
        phase = BenchmarkPhase(self.phase)
        object.__setattr__(self, "phase", phase)

        if phase not in self.workload.phases:
            raise ValueError(
                f"Workload {self.workload.workload_id} does not support phase "
                f"{phase.value}."
            )
        plan_access = AccessPattern(self.storage_plan.access_pattern)
        if plan_access not in self.workload.access_patterns:
            raise ValueError(
                f"Workload {self.workload.workload_id} does not support access "
                f"pattern {plan_access.value}."
            )
        if (
            self.storage_plan.logical_schema_id != self.array_contract.schema_id
            or self.storage_plan.logical_schema_version
            != self.array_contract.schema_version
        ):
            raise ValueError(
                "Benchmark storage plan logical schema identity does not match "
                "the array contract."
            )
        expected_dtype = self.array_contract.dtype.numpy_dtype
        if (
            expected_dtype is not None
            and self.storage_plan.logical_dtype != expected_dtype
        ):
            raise ValueError(
                f"Benchmark plan dtype {self.storage_plan.logical_dtype!r} does "
                f"not match contract dtype {expected_dtype!r}."
            )
        shape_errors = self.array_contract.validate_shape(
            self.storage_plan.logical_shape
        )
        if shape_errors:
            raise ValueError(
                "Benchmark plan shape violates array contract: "
                + "; ".join(shape_errors)
            )

    def as_manifest(self) -> dict[str, object]:
        return {
            "case_id": self.case_id,
            "phase": self.phase.value,
            "logical_schema": {
                "id": self.array_contract.schema_id,
                "version": self.array_contract.schema_version,
            },
            "workload": self.workload.as_manifest(),
            "storage_plan": self.storage_plan.as_dict(),
        }


def benchmark_result_envelope(
    case: StorageBenchmarkCase,
    *,
    source_identity: Mapping[str, object],
    environment: Mapping[str, object],
    trials: list[Mapping[str, object]],
    summary: Mapping[str, object],
    validation: Mapping[str, object],
) -> dict[str, object]:
    """Build the common JSON result envelope used by benchmark adapters."""

    return {
        "schema_id": "palette.storage_benchmark",
        "schema_version": 1,
        **case.as_manifest(),
        "source_identity": dict(source_identity),
        "environment": dict(environment),
        "trials": [dict(trial) for trial in trials],
        "summary": dict(summary),
        "validation": dict(validation),
    }
