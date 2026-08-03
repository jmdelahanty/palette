"""Executable coverage catalog for derived-analysis storage benchmarks.

Storage candidates and benchmark evidence are deliberately separate.  A
candidate can exist without a runnable source/candidate matrix, and a runnable
matrix is not promotion evidence until representative executions and real
consumer gates have been recorded.  This catalog makes those gaps explicit for
every current derived-analysis storage contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from importlib import import_module
import re
from typing import Any

from .storage_candidate_catalog import (
    DERIVED_ANALYSIS_STORAGE_CANDIDATE_BY_STAGE,
)


_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_]*$")
_MODULE = re.compile(r"^[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)+$")
_CALLABLE = re.compile(r"^_?[a-z][a-z0-9_]*$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class StorageBenchmarkAdapterStatus(str, Enum):
    """Current executable benchmark-adapter state."""

    PLAN_ONLY = "plan_only"
    READ_MATRIX_IMPLEMENTED = "read_matrix_implemented"


@dataclass(frozen=True)
class DerivedAnalysisStorageBenchmark:
    """Truthful benchmark coverage for one cataloged storage candidate."""

    stage_id: str
    adapter_status: StorageBenchmarkAdapterStatus
    crimson_consumer_required: bool
    adapter_module: str | None = None
    adapter_entrypoint: str | None = None
    writer_phase_measured: bool = False
    publication_phase_measured: bool = False
    reader_workload_implemented: bool = False
    decoded_equality_implemented: bool = False
    metadata_equivalence_implemented: bool = False
    physical_io_measured: bool = False
    palette_consumer_implemented: bool = False
    crimson_consumer_implemented: bool = False
    representative_short_executed: bool = False
    representative_full_executed: bool = False
    evidence_receipt_path: str | None = None
    evidence_receipt_sha256: str | None = None
    gate_contract_id: str | None = None
    gate_contract_version: int | None = None
    gate_passed: bool = False

    def __post_init__(self) -> None:
        if type(self.stage_id) is not str or not _IDENTIFIER.fullmatch(
            self.stage_id
        ):
            raise ValueError("stage_id must be one canonical identifier")
        if self.stage_id not in DERIVED_ANALYSIS_STORAGE_CANDIDATE_BY_STAGE:
            raise ValueError("benchmark stage must own one storage candidate")
        if not isinstance(self.adapter_status, StorageBenchmarkAdapterStatus):
            raise TypeError("adapter_status must use StorageBenchmarkAdapterStatus")
        for field in (
            "writer_phase_measured",
            "publication_phase_measured",
            "reader_workload_implemented",
            "decoded_equality_implemented",
            "metadata_equivalence_implemented",
            "physical_io_measured",
            "palette_consumer_implemented",
            "crimson_consumer_required",
            "crimson_consumer_implemented",
            "representative_short_executed",
            "representative_full_executed",
            "gate_passed",
        ):
            if type(getattr(self, field)) is not bool:
                raise TypeError(f"{field} must be an exact bool")

        has_adapter = self.adapter_status is not StorageBenchmarkAdapterStatus.PLAN_ONLY
        if has_adapter:
            if (
                type(self.adapter_module) is not str
                or not _MODULE.fullmatch(self.adapter_module)
                or type(self.adapter_entrypoint) is not str
                or not _CALLABLE.fullmatch(self.adapter_entrypoint)
            ):
                raise ValueError(
                    "implemented benchmark adapters require an exact module and "
                    "entrypoint"
                )
            if not self.reader_workload_implemented:
                raise ValueError(
                    "an implemented read adapter must declare its reader workload"
                )
        elif self.adapter_module is not None or self.adapter_entrypoint is not None:
            raise ValueError("plan-only benchmark coverage must not claim an adapter")

        if (
            self.physical_io_measured
            or self.palette_consumer_implemented
            or self.crimson_consumer_implemented
            or self.representative_short_executed
            or self.representative_full_executed
        ) and not has_adapter:
            raise ValueError(
                "executed or consumer evidence requires an implemented adapter"
            )

        evidence_claimed = any(
            (
                self.writer_phase_measured,
                self.publication_phase_measured,
                self.physical_io_measured,
                self.representative_short_executed,
                self.representative_full_executed,
                self.gate_passed,
            )
        )
        binding_values = (
            self.evidence_receipt_path,
            self.evidence_receipt_sha256,
            self.gate_contract_id,
            self.gate_contract_version,
        )
        if evidence_claimed:
            if (
                type(self.evidence_receipt_path) is not str
                or not self.evidence_receipt_path.strip()
                or type(self.evidence_receipt_sha256) is not str
                or not _SHA256.fullmatch(self.evidence_receipt_sha256)
                or type(self.gate_contract_id) is not str
                or not _IDENTIFIER.fullmatch(self.gate_contract_id)
                or type(self.gate_contract_version) is not int
                or self.gate_contract_version < 1
            ):
                raise ValueError(
                    "measured/executed benchmark claims require one immutable "
                    "evidence receipt and versioned gate binding"
                )
        elif any(value is not None for value in binding_values):
            raise ValueError(
                "benchmark evidence bindings require a measured/executed claim"
            )

    def resolves_adapter(self) -> bool:
        """Return whether the declared adapter entrypoint is importable."""

        if self.adapter_module is None or self.adapter_entrypoint is None:
            return False
        module: Any = import_module(self.adapter_module)
        return callable(getattr(module, self.adapter_entrypoint, None))

    @property
    def benchmark_coverage_complete(self) -> bool:
        """Return whether every declared evidence category is bound and passed.

        Complete coverage is deliberately not profile-promotion authorization;
        activation remains a separate versioned policy decision.
        """

        return all(
            (
                self.writer_phase_measured,
                self.publication_phase_measured,
                self.reader_workload_implemented,
                self.decoded_equality_implemented,
                self.metadata_equivalence_implemented,
                self.physical_io_measured,
                self.palette_consumer_implemented,
                (
                    not self.crimson_consumer_required
                    or self.crimson_consumer_implemented
                ),
                self.representative_short_executed,
                self.representative_full_executed,
                self.evidence_receipt_path is not None,
                self.evidence_receipt_sha256 is not None,
                self.gate_contract_id is not None,
                self.gate_contract_version is not None,
                self.gate_passed,
            )
        )

    def as_record(self) -> dict[str, object]:
        return {
            "stage_id": self.stage_id,
            "adapter_status": self.adapter_status.value,
            "adapter_module": self.adapter_module,
            "adapter_entrypoint": self.adapter_entrypoint,
            "writer_phase_measured": self.writer_phase_measured,
            "publication_phase_measured": self.publication_phase_measured,
            "reader_workload_implemented": self.reader_workload_implemented,
            "decoded_equality_implemented": self.decoded_equality_implemented,
            "metadata_equivalence_implemented": self.metadata_equivalence_implemented,
            "physical_io_measured": self.physical_io_measured,
            "palette_consumer_implemented": self.palette_consumer_implemented,
            "crimson_consumer_required": self.crimson_consumer_required,
            "crimson_consumer_implemented": self.crimson_consumer_implemented,
            "representative_short_executed": self.representative_short_executed,
            "representative_full_executed": self.representative_full_executed,
            "evidence_receipt_path": self.evidence_receipt_path,
            "evidence_receipt_sha256": self.evidence_receipt_sha256,
            "gate_contract_id": self.gate_contract_id,
            "gate_contract_version": self.gate_contract_version,
            "gate_passed": self.gate_passed,
            "benchmark_coverage_complete": self.benchmark_coverage_complete,
        }


_CRIMSON_REQUIRED_STAGES = frozenset(
    DERIVED_ANALYSIS_STORAGE_CANDIDATE_BY_STAGE
)


def _read_matrix(stage_id: str) -> DerivedAnalysisStorageBenchmark:
    adapter_module, adapter_entrypoint = _READ_MATRIX_ADAPTERS[stage_id]
    return DerivedAnalysisStorageBenchmark(
        stage_id=stage_id,
        adapter_status=StorageBenchmarkAdapterStatus.READ_MATRIX_IMPLEMENTED,
        crimson_consumer_required=stage_id in _CRIMSON_REQUIRED_STAGES,
        adapter_module=adapter_module,
        adapter_entrypoint=adapter_entrypoint,
        reader_workload_implemented=True,
        decoded_equality_implemented=True,
        metadata_equivalence_implemented=True,
    )


def _plan_only(stage_id: str) -> DerivedAnalysisStorageBenchmark:
    return DerivedAnalysisStorageBenchmark(
        stage_id=stage_id,
        adapter_status=StorageBenchmarkAdapterStatus.PLAN_ONLY,
        crimson_consumer_required=stage_id in _CRIMSON_REQUIRED_STAGES,
    )


_EXACT_TABULAR_ADAPTER = (
    "fisheye.diagnostics.benchmark_exact_tabular_candidates",
    "run_benchmark_matrix",
)
_READ_MATRIX_ADAPTERS = {
    "swim_bouts": _EXACT_TABULAR_ADAPTER,
    "bout_kinematics": _EXACT_TABULAR_ADAPTER,
    "detection_occupancy": _EXACT_TABULAR_ADAPTER,
    "session_occupancy": _EXACT_TABULAR_ADAPTER,
    "chaser_distance": (
        "fisheye.diagnostics.benchmark_chaser_distance_base_candidate",
        "run_benchmark_matrix",
    ),
    "stimulus_epochs": (
        "fisheye.diagnostics.benchmark_stimulus_epoch_reads",
        "run_benchmark_matrix",
    ),
    "stimulus_response": (
        "fisheye.diagnostics.benchmark_stimulus_response_reads",
        "run_benchmark_matrix",
    ),
    "eye_angles": (
        "fisheye.diagnostics.benchmark_eye_angle_v7_reads",
        "run_benchmark_matrix",
    ),
    "track_kinematics": (
        "fisheye.diagnostics.benchmark_track_kinematics_v2_candidate",
        "run_benchmark_matrix",
    ),
    "subject_shape": (
        "fisheye.diagnostics.benchmark_subject_shape_v4_candidate",
        "run_benchmark_matrix",
    ),
}

DERIVED_ANALYSIS_STORAGE_BENCHMARKS: tuple[
    DerivedAnalysisStorageBenchmark, ...
] = tuple(
    (
        _read_matrix(stage_id)
        if stage_id in _READ_MATRIX_ADAPTERS
        else _plan_only(stage_id)
    )
    for stage_id in DERIVED_ANALYSIS_STORAGE_CANDIDATE_BY_STAGE
)

DERIVED_ANALYSIS_STORAGE_BENCHMARK_BY_STAGE = {
    benchmark.stage_id: benchmark
    for benchmark in DERIVED_ANALYSIS_STORAGE_BENCHMARKS
}


def resolved_storage_benchmarks() -> tuple[dict[str, object], ...]:
    return tuple(benchmark.as_record() for benchmark in DERIVED_ANALYSIS_STORAGE_BENCHMARKS)


__all__ = [
    "DERIVED_ANALYSIS_STORAGE_BENCHMARK_BY_STAGE",
    "DERIVED_ANALYSIS_STORAGE_BENCHMARKS",
    "DerivedAnalysisStorageBenchmark",
    "StorageBenchmarkAdapterStatus",
    "resolved_storage_benchmarks",
]
