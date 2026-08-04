"""Lightweight report-only phase telemetry for analysis materializers.

The records produced here describe one execution.  They are deliberately kept
out of scientific payload identity, storage digests, and selector contracts.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from datetime import datetime, timezone
import math
import os
import resource
import socket
import time
from typing import Any


RUNTIME_TELEMETRY_SCHEMA_ID = "palette.materializer_phase_telemetry"
RUNTIME_TELEMETRY_SCHEMA_VERSION = 1
RUNTIME_TELEMETRY_IDENTITY_POLICY = (
    "report_only_excluded_from_scientific_identity_and_payload_digests"
)

_CPU_FIELDS = (
    "own_user_cpu_seconds",
    "own_system_cpu_seconds",
    "child_user_cpu_seconds",
    "child_system_cpu_seconds",
)
_ROOT_FIELDS = frozenset(
    {
        "schema_id",
        "schema_version",
        "identity_policy",
        "materializer",
        "started_at_utc",
        "finished_at_utc",
        "wall_seconds",
        "cpu_seconds",
        "average_effective_cpu_cores",
        "process_peak_rss_bytes",
        "children_peak_rss_bytes",
        "phase_wall_seconds_sum",
        "phases",
        "execution",
    }
)
_PHASE_BASE_FIELDS = frozenset(
    {
        "name",
        "started_at_utc",
        "finished_at_utc",
        "outcome",
        "wall_seconds",
        "cpu_seconds",
        "average_effective_cpu_cores",
        "process_peak_rss_bytes_at_end",
        "children_peak_rss_bytes_at_end",
        "process_io_delta",
    }
)
_EXECUTION_BASE_FIELDS = frozenset(
    {"host", "pid", "lsb_jobid", "lsb_jobname", "lsb_queue", "allocated_slots"}
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _rss_bytes(value: int | float) -> int:
    # Palette materializers run on Linux, where ru_maxrss is reported in KiB.
    return max(0, int(value)) * 1024


def _proc_io() -> dict[str, int]:
    path = "/proc/self/io"
    try:
        with open(path, encoding="utf-8") as stream:
            lines = stream.read().splitlines()
    except OSError:
        return {}
    values: dict[str, int] = {}
    for line in lines:
        key, separator, raw_value = line.partition(":")
        if not separator:
            continue
        try:
            values[key.strip()] = int(raw_value.strip())
        except ValueError:
            continue
    return values


def _snapshot() -> dict[str, Any]:
    own = resource.getrusage(resource.RUSAGE_SELF)
    children = resource.getrusage(resource.RUSAGE_CHILDREN)
    return {
        "perf_counter": float(time.perf_counter()),
        "own_user_cpu_seconds": float(own.ru_utime),
        "own_system_cpu_seconds": float(own.ru_stime),
        "child_user_cpu_seconds": float(children.ru_utime),
        "child_system_cpu_seconds": float(children.ru_stime),
        "own_peak_rss_bytes": _rss_bytes(own.ru_maxrss),
        "child_peak_rss_bytes": _rss_bytes(children.ru_maxrss),
        "proc_io": _proc_io(),
    }


def _nonnegative_delta(after: float | int, before: float | int) -> float:
    return max(0.0, float(after) - float(before))


def _require_timestamp(value: object, *, label: str) -> datetime:
    if type(value) is not str or not value:
        raise ValueError(f"{label} must be one nonempty ISO-8601 timestamp.")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{label} is not an ISO-8601 timestamp.") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{label} must include a timezone.")
    return parsed


def _require_nonnegative_number(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be one finite nonnegative number.")
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{label} must be one finite nonnegative number.")
    return number


def _require_nonnegative_int(value: object, *, label: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{label} must be one nonnegative exact integer.")
    return value


def _require_cpu_record(value: object, *, label: str) -> dict[str, float]:
    fields = {*_CPU_FIELDS, "total"}
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError(f"{label} CPU field set differs.")
    record = {
        name: _require_nonnegative_number(value[name], label=f"{label}.{name}")
        for name in fields
    }
    expected_total = sum(record[name] for name in _CPU_FIELDS)
    if not math.isclose(record["total"], expected_total, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError(f"{label}.total differs from its CPU components.")
    return record


def _require_average_cpu(
    value: object,
    *,
    total_cpu: float,
    wall_seconds: float,
    label: str,
) -> None:
    observed = _require_nonnegative_number(value, label=label)
    expected = total_cpu / wall_seconds if wall_seconds > 0.0 else 0.0
    if not math.isclose(observed, expected, rel_tol=1e-9, abs_tol=1e-12):
        raise ValueError(f"{label} differs from CPU/wall measurements.")


def require_runtime_telemetry(
    value: Mapping[str, Any],
    *,
    expected_materializer: str | None = None,
    allowed_phase_order: tuple[str, ...] | None = None,
    require_error_phase: bool = False,
) -> None:
    """Deeply validate report-only materializer telemetry.

    ``allowed_phase_order`` permits a failure-prefix/subsequence while still
    rejecting duplicate, unknown, or reordered phase claims.
    """

    if not isinstance(value, Mapping) or set(value) != _ROOT_FIELDS:
        raise ValueError("Runtime telemetry root field set differs.")
    if (
        value["schema_id"] != RUNTIME_TELEMETRY_SCHEMA_ID
        or value["schema_version"] != RUNTIME_TELEMETRY_SCHEMA_VERSION
        or value["identity_policy"] != RUNTIME_TELEMETRY_IDENTITY_POLICY
    ):
        raise ValueError("Runtime telemetry contract identity differs.")
    if type(value["materializer"]) is not str or not value["materializer"]:
        raise ValueError("Runtime telemetry materializer is invalid.")
    if expected_materializer is not None and value["materializer"] != expected_materializer:
        raise ValueError("Runtime telemetry materializer differs.")
    started = _require_timestamp(value["started_at_utc"], label="runtime started_at_utc")
    finished = _require_timestamp(value["finished_at_utc"], label="runtime finished_at_utc")
    if finished < started:
        raise ValueError("Runtime telemetry finishes before it starts.")
    wall = _require_nonnegative_number(value["wall_seconds"], label="runtime wall_seconds")
    cpu = _require_cpu_record(value["cpu_seconds"], label="runtime")
    _require_average_cpu(
        value["average_effective_cpu_cores"],
        total_cpu=cpu["total"],
        wall_seconds=wall,
        label="runtime average_effective_cpu_cores",
    )
    _require_nonnegative_int(value["process_peak_rss_bytes"], label="runtime process_peak_rss_bytes")
    _require_nonnegative_int(value["children_peak_rss_bytes"], label="runtime children_peak_rss_bytes")
    phase_sum = _require_nonnegative_number(
        value["phase_wall_seconds_sum"], label="runtime phase_wall_seconds_sum"
    )
    phases = value["phases"]
    if not isinstance(phases, list):
        raise ValueError("Runtime telemetry phases must be one array.")
    order = (
        {name: index for index, name in enumerate(allowed_phase_order)}
        if allowed_phase_order is not None
        else None
    )
    prior_index = -1
    observed_names: set[str] = set()
    observed_phase_wall = 0.0
    saw_error = False
    for index, phase in enumerate(phases):
        if not isinstance(phase, Mapping):
            raise ValueError("Runtime telemetry phase must be one object.")
        outcome = phase.get("outcome")
        expected_fields = _PHASE_BASE_FIELDS | ({"error_type"} if outcome == "error" else set())
        if set(phase) != expected_fields or outcome not in {"ok", "error"}:
            raise ValueError("Runtime telemetry phase field set or outcome differs.")
        name = phase["name"]
        if type(name) is not str or not name or name in observed_names:
            raise ValueError("Runtime telemetry phase name is invalid or duplicated.")
        observed_names.add(name)
        if order is not None:
            if name not in order or order[name] <= prior_index:
                raise ValueError("Runtime telemetry phase names are unknown or reordered.")
            prior_index = order[name]
        phase_started = _require_timestamp(
            phase["started_at_utc"], label=f"runtime phase {index} started_at_utc"
        )
        phase_finished = _require_timestamp(
            phase["finished_at_utc"], label=f"runtime phase {index} finished_at_utc"
        )
        if phase_finished < phase_started:
            raise ValueError("Runtime telemetry phase finishes before it starts.")
        phase_wall = _require_nonnegative_number(
            phase["wall_seconds"], label=f"runtime phase {name} wall_seconds"
        )
        observed_phase_wall += phase_wall
        phase_cpu = _require_cpu_record(phase["cpu_seconds"], label=f"runtime phase {name}")
        _require_average_cpu(
            phase["average_effective_cpu_cores"],
            total_cpu=phase_cpu["total"],
            wall_seconds=phase_wall,
            label=f"runtime phase {name} average_effective_cpu_cores",
        )
        _require_nonnegative_int(
            phase["process_peak_rss_bytes_at_end"],
            label=f"runtime phase {name} process_peak_rss_bytes_at_end",
        )
        _require_nonnegative_int(
            phase["children_peak_rss_bytes_at_end"],
            label=f"runtime phase {name} children_peak_rss_bytes_at_end",
        )
        io = phase["process_io_delta"]
        if not isinstance(io, Mapping) or any(
            type(key) is not str or type(item) is not int or item < 0
            for key, item in io.items()
        ):
            raise ValueError("Runtime telemetry phase I/O counters are invalid.")
        if outcome == "error":
            saw_error = True
            if type(phase["error_type"]) is not str or not phase["error_type"]:
                raise ValueError("Runtime telemetry error phase lacks an error type.")
    if not math.isclose(phase_sum, observed_phase_wall, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError("Runtime telemetry phase-wall sum differs.")
    if require_error_phase and not saw_error:
        raise ValueError("Runtime telemetry contains no failed phase.")
    execution = value["execution"]
    if not isinstance(execution, Mapping) or not _EXECUTION_BASE_FIELDS.issubset(execution):
        raise ValueError("Runtime telemetry execution context is incomplete.")
    if type(execution["host"]) is not str or not execution["host"]:
        raise ValueError("Runtime telemetry execution host is invalid.")
    if type(execution["pid"]) is not int or execution["pid"] <= 0:
        raise ValueError("Runtime telemetry execution pid is invalid.")
    for field in ("lsb_jobid", "lsb_jobname", "lsb_queue", "allocated_slots"):
        if execution[field] is not None and type(execution[field]) is not str:
            raise ValueError(f"Runtime telemetry execution {field} is invalid.")
    # This also rejects NaN/infinity and non-JSON context values.
    import json

    json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":"))


class PhaseTelemetry:
    """Collect ordered wall/CPU/I/O measurements around named phases."""

    def __init__(
        self,
        *,
        materializer: str,
        context: Mapping[str, Any] | None = None,
    ) -> None:
        self.materializer = str(materializer)
        self.context = dict(context or {})
        self.started_at_utc = _utc_now()
        self._started = _snapshot()
        self._phases: list[dict[str, Any]] = []

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        started_at_utc = _utc_now()
        before = _snapshot()
        outcome = "ok"
        error_type: str | None = None
        try:
            yield
        except BaseException as exc:
            outcome = "error"
            error_type = type(exc).__name__
            raise
        finally:
            after = _snapshot()
            wall_seconds = _nonnegative_delta(
                after["perf_counter"], before["perf_counter"]
            )
            cpu = {
                key: _nonnegative_delta(after[key], before[key])
                for key in (
                    "own_user_cpu_seconds",
                    "own_system_cpu_seconds",
                    "child_user_cpu_seconds",
                    "child_system_cpu_seconds",
                )
            }
            total_cpu_seconds = float(sum(cpu.values()))
            io_before = dict(before["proc_io"])
            io_after = dict(after["proc_io"])
            io_delta = {
                key: int(_nonnegative_delta(io_after[key], io_before.get(key, 0)))
                for key in sorted(io_after)
                if key in io_before
            }
            record: dict[str, Any] = {
                "name": str(name),
                "started_at_utc": started_at_utc,
                "finished_at_utc": _utc_now(),
                "outcome": outcome,
                "wall_seconds": wall_seconds,
                "cpu_seconds": {
                    **cpu,
                    "total": total_cpu_seconds,
                },
                "average_effective_cpu_cores": (
                    total_cpu_seconds / wall_seconds if wall_seconds > 0.0 else 0.0
                ),
                "process_peak_rss_bytes_at_end": int(after["own_peak_rss_bytes"]),
                "children_peak_rss_bytes_at_end": int(
                    after["child_peak_rss_bytes"]
                ),
                "process_io_delta": io_delta,
            }
            if error_type is not None:
                record["error_type"] = error_type
            self._phases.append(record)

    def duration_seconds(self, name: str) -> float | None:
        matches = [
            float(item["wall_seconds"])
            for item in self._phases
            if item["name"] == name
        ]
        return float(sum(matches)) if matches else None

    def to_json(self) -> dict[str, Any]:
        finished = _snapshot()
        wall_seconds = _nonnegative_delta(
            finished["perf_counter"], self._started["perf_counter"]
        )
        cpu = {
            key: _nonnegative_delta(finished[key], self._started[key])
            for key in (
                "own_user_cpu_seconds",
                "own_system_cpu_seconds",
                "child_user_cpu_seconds",
                "child_system_cpu_seconds",
            )
        }
        total_cpu_seconds = float(sum(cpu.values()))
        return {
            "schema_id": RUNTIME_TELEMETRY_SCHEMA_ID,
            "schema_version": RUNTIME_TELEMETRY_SCHEMA_VERSION,
            "identity_policy": RUNTIME_TELEMETRY_IDENTITY_POLICY,
            "materializer": self.materializer,
            "started_at_utc": self.started_at_utc,
            "finished_at_utc": _utc_now(),
            "wall_seconds": wall_seconds,
            "cpu_seconds": {**cpu, "total": total_cpu_seconds},
            "average_effective_cpu_cores": (
                total_cpu_seconds / wall_seconds if wall_seconds > 0.0 else 0.0
            ),
            "process_peak_rss_bytes": int(finished["own_peak_rss_bytes"]),
            "children_peak_rss_bytes": int(finished["child_peak_rss_bytes"]),
            "phase_wall_seconds_sum": float(
                sum(float(item["wall_seconds"]) for item in self._phases)
            ),
            "phases": list(self._phases),
            "execution": {
                "host": socket.gethostname(),
                "pid": int(os.getpid()),
                "lsb_jobid": os.environ.get("LSB_JOBID"),
                "lsb_jobname": os.environ.get("LSB_JOBNAME"),
                "lsb_queue": os.environ.get("LSB_QUEUE"),
                "allocated_slots": os.environ.get("LSB_DJOB_NUMPROC"),
                **self.context,
            },
        }


__all__ = [
    "PhaseTelemetry",
    "RUNTIME_TELEMETRY_IDENTITY_POLICY",
    "RUNTIME_TELEMETRY_SCHEMA_ID",
    "RUNTIME_TELEMETRY_SCHEMA_VERSION",
    "require_runtime_telemetry",
]
