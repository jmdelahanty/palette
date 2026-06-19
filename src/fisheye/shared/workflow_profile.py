"""Small JSONL workflow profiler for coarse pipeline phase timings."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


class WorkflowProfilePhase:
    def __init__(self, profiler: "WorkflowProfiler", phase: str, metadata: Mapping[str, Any]) -> None:
        self._profiler = profiler
        self._phase = str(phase)
        self._metadata = dict(metadata)
        self._started_perf = 0.0
        self._started_at_utc = ""
        self.details: dict[str, Any] = {}

    def __enter__(self) -> dict[str, Any]:
        self._started_perf = time.perf_counter()
        self._started_at_utc = datetime.now(timezone.utc).isoformat()
        self._profiler.emit(
            "start",
            self._phase,
            started_at_utc=self._started_at_utc,
            **self._metadata,
        )
        return self.details

    def __exit__(self, exc_type: object, exc: BaseException | None, traceback: object) -> bool:
        payload = {
            "started_at_utc": self._started_at_utc,
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "duration_seconds": float(time.perf_counter() - self._started_perf),
            "status": "error" if exc is not None else "ok",
            **self._metadata,
            **self.details,
        }
        if exc is not None:
            payload["error"] = repr(exc)
        self._profiler.record_finish(self._phase, payload)
        return False


class WorkflowProfiler:
    """Collect coarse phase timings and optionally append start/finish JSONL."""

    def __init__(self, path: Path | None, *, schema_prefix: str = "palette_workflow") -> None:
        self.path = path
        self.schema_prefix = str(schema_prefix)
        self.events: list[dict[str, Any]] = []
        self.phase_seconds: dict[str, float] = {}
        self.phase_counts: dict[str, int] = {}
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, event: str, phase: str, **payload: Any) -> None:
        row = {
            "schema": f"{self.schema_prefix}_profile_event_v1",
            "event": str(event),
            "phase": str(phase),
            "emitted_at_utc": datetime.now(timezone.utc).isoformat(),
            **payload,
        }
        safe_row = json_safe(row)
        self.events.append(safe_row)
        if self.path is not None:
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(safe_row, sort_keys=True) + "\n")

    def record_finish(self, phase: str, payload: Mapping[str, Any]) -> None:
        seconds = float(payload.get("duration_seconds") or 0.0)
        self.phase_seconds[str(phase)] = float(self.phase_seconds.get(str(phase), 0.0) + seconds)
        self.phase_counts[str(phase)] = int(self.phase_counts.get(str(phase), 0) + 1)
        self.emit("finish", phase, **dict(payload))

    def phase(self, phase: str, **metadata: Any) -> WorkflowProfilePhase:
        return WorkflowProfilePhase(self, phase, metadata)

    def summary(self) -> dict[str, Any]:
        return {
            "schema": f"{self.schema_prefix}_profile_summary_v1",
            "profile_jsonl": str(self.path) if self.path is not None else None,
            "phase_seconds": dict(sorted(self.phase_seconds.items())),
            "phase_counts": dict(sorted(self.phase_counts.items())),
            "event_count": int(len(self.events)),
        }
