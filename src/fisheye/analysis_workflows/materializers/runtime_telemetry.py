"""Lightweight report-only phase telemetry for analysis materializers.

The records produced here describe one execution.  They are deliberately kept
out of scientific payload identity, storage digests, and selector contracts.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from datetime import datetime, timezone
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
]
