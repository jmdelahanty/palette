"""Non-authoritative phase timing for exact analytics export publishers.

Runtime telemetry is returned to the invoking process only.  It is never part
of an immutable export manifest, source binding, content digest, selector, or
registry identity.  The recorder rejects nested and duplicate phases so the
reported phase sum remains interpretable.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
import math
import time


EXPORT_RUNTIME_TELEMETRY_SCHEMA_ID = "palette.analytics_export.runtime_telemetry"
EXPORT_RUNTIME_TELEMETRY_SCHEMA_VERSION = 1
EXPORT_RUNTIME_PHASES = (
    "source_binding_before",
    "scratch_parquet_write",
    "source_binding_after",
    "scratch_to_staging_copy",
    "staged_decoded_validation",
    "manifest_validation",
    "publication_staged_validation",
    "publication_generation_rename",
    "publication_manifest_commit",
    "published_payload_validation",
)


class ExportRuntimePhaseRecorder:
    """Record exact non-overlapping monotonic phase durations."""

    def __init__(self, *, clock: Callable[[], float] = time.perf_counter) -> None:
        self._clock = clock
        self._started = float(clock())
        self._durations: dict[str, float] = {}
        self._active: str | None = None

    @contextmanager
    def measure(self, phase: str) -> Iterator[None]:
        if phase not in EXPORT_RUNTIME_PHASES:
            raise ValueError(f"Unknown analytics-export runtime phase: {phase!r}")
        if self._active is not None:
            raise RuntimeError(
                f"Cannot nest export phase {phase!r} inside {self._active!r}."
            )
        if phase in self._durations:
            raise RuntimeError(
                f"Analytics-export phase {phase!r} was already measured."
            )
        self._active = phase
        started = float(self._clock())
        try:
            yield
        finally:
            finished = float(self._clock())
            duration = finished - started
            if not math.isfinite(duration) or duration < 0:
                raise RuntimeError("Analytics-export phase clock moved backwards.")
            self._durations[phase] = duration
            self._active = None

    def snapshot(self) -> dict[str, object]:
        if self._active is not None:
            raise RuntimeError("Cannot snapshot an active analytics-export phase.")
        if tuple(self._durations) != EXPORT_RUNTIME_PHASES:
            raise RuntimeError(
                "Successful analytics export omitted or reordered runtime phases."
            )
        wall = float(self._clock()) - self._started
        measured = float(sum(self._durations.values()))
        if not math.isfinite(wall) or wall < measured:
            raise RuntimeError("Analytics-export runtime totals are invalid.")
        return {
            "schema_id": EXPORT_RUNTIME_TELEMETRY_SCHEMA_ID,
            "schema_version": EXPORT_RUNTIME_TELEMETRY_SCHEMA_VERSION,
            "clock": "time.perf_counter_monotonic_seconds",
            "phases_are_nonoverlapping": True,
            "phase_order": list(EXPORT_RUNTIME_PHASES),
            "phases_seconds": dict(self._durations),
            "measured_phase_seconds": measured,
            "wall_elapsed_seconds": wall,
            "unmeasured_overhead_seconds": wall - measured,
            "persistence_policy": "process_result_only_not_manifest_or_scientific_identity",
        }


def validate_export_runtime_telemetry(payload: Mapping[str, object]) -> None:
    expected_fields = {
        "schema_id",
        "schema_version",
        "clock",
        "phases_are_nonoverlapping",
        "phase_order",
        "phases_seconds",
        "measured_phase_seconds",
        "wall_elapsed_seconds",
        "unmeasured_overhead_seconds",
        "persistence_policy",
    }
    if set(payload) != expected_fields:
        raise ValueError("Analytics-export runtime telemetry has unexpected fields.")
    if (
        payload.get("schema_id") != EXPORT_RUNTIME_TELEMETRY_SCHEMA_ID
        or payload.get("schema_version") != EXPORT_RUNTIME_TELEMETRY_SCHEMA_VERSION
        or payload.get("clock") != "time.perf_counter_monotonic_seconds"
        or payload.get("phases_are_nonoverlapping") is not True
        or payload.get("persistence_policy")
        != "process_result_only_not_manifest_or_scientific_identity"
        or payload.get("phase_order") != list(EXPORT_RUNTIME_PHASES)
    ):
        raise ValueError("Analytics-export runtime telemetry contract is invalid.")
    phases = payload.get("phases_seconds")
    if not isinstance(phases, Mapping) or set(phases) != set(EXPORT_RUNTIME_PHASES):
        raise ValueError("Analytics-export runtime phase set is invalid.")
    # JSON object member order is not semantic, and canonical/sorted JSON
    # publication deliberately reorders this mapping. The separately required
    # phase_order field owns sequence; totals must follow that explicit order.
    values = tuple(phases[phase] for phase in EXPORT_RUNTIME_PHASES)
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0
        for value in values
    ):
        raise ValueError("Analytics-export runtime phase duration is invalid.")
    measured = payload.get("measured_phase_seconds")
    wall = payload.get("wall_elapsed_seconds")
    overhead = payload.get("unmeasured_overhead_seconds")
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0
        for value in (measured, wall, overhead)
    ):
        raise ValueError("Analytics-export runtime total is invalid.")
    if not math.isclose(
        float(measured),
        sum(float(value) for value in values),
        rel_tol=0.0,
        abs_tol=1e-9,
    ) or not math.isclose(
        float(wall),
        float(measured) + float(overhead),
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise ValueError("Analytics-export runtime totals do not reconcile.")


__all__ = [
    "EXPORT_RUNTIME_PHASES",
    "EXPORT_RUNTIME_TELEMETRY_SCHEMA_ID",
    "EXPORT_RUNTIME_TELEMETRY_SCHEMA_VERSION",
    "ExportRuntimePhaseRecorder",
    "validate_export_runtime_telemetry",
]
