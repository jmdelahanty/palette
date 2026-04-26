"""Shared timing helpers for ROI-model inference diagnostics."""

from __future__ import annotations

import time
from contextlib import contextmanager
from threading import Lock
from typing import Any, Iterator, Sequence


class InferenceTimingProfiler:
    """Collect per-stage timing summaries for inference loops.

    This profiler is intentionally lightweight and JSON-friendly so callers can
    persist the summary into run attrs and benchmark payloads.
    """

    def __init__(self, *, enabled: bool = False) -> None:
        self.enabled = bool(enabled)
        self._order: list[str] = []
        self._seconds: dict[str, float] = {}
        self._calls: dict[str, int] = {}
        self._items: dict[str, int] = {}
        self._lock = Lock()

    def record(self, stage: str, seconds: float, *, items: int = 0) -> None:
        if not self.enabled:
            return
        name = str(stage)
        with self._lock:
            if name not in self._seconds:
                self._order.append(name)
                self._seconds[name] = 0.0
                self._calls[name] = 0
                self._items[name] = 0
            self._seconds[name] += float(max(0.0, seconds))
            self._calls[name] += 1
            self._items[name] += int(max(0, items))

    @contextmanager
    def time(self, stage: str, *, items: int = 0) -> Iterator[None]:
        started = time.perf_counter()
        try:
            yield
        finally:
            self.record(stage, time.perf_counter() - started, items=items)

    def summary(
        self,
        *,
        total_items: int | None = None,
        wall_seconds: float | None = None,
        notes: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        if not self.enabled:
            return {"enabled": False}

        accounted_seconds = float(sum(self._seconds.values()))
        resolved_wall_seconds = float(wall_seconds) if wall_seconds is not None else accounted_seconds
        unaccounted_seconds = max(0.0, resolved_wall_seconds - accounted_seconds)

        stages: dict[str, dict[str, Any]] = {}
        for name in self._order:
            total_seconds = float(self._seconds.get(name, 0.0))
            calls = int(self._calls.get(name, 0))
            items = int(self._items.get(name, 0))
            stages[name] = {
                "total_seconds": total_seconds,
                "calls": calls,
                "items": items,
                "avg_ms_per_call": (total_seconds * 1000.0 / calls) if calls > 0 else 0.0,
                "avg_ms_per_item": (total_seconds * 1000.0 / items) if items > 0 else None,
                "share_of_wall_time_percent": (
                    (total_seconds / resolved_wall_seconds) * 100.0 if resolved_wall_seconds > 0 else 0.0
                ),
            }

        payload: dict[str, Any] = {
            "enabled": True,
            "total_items": int(total_items) if total_items is not None else None,
            "wall_seconds": resolved_wall_seconds,
            "accounted_seconds": accounted_seconds,
            "unaccounted_seconds": unaccounted_seconds,
            "unaccounted_share_of_wall_time_percent": (
                (unaccounted_seconds / resolved_wall_seconds) * 100.0 if resolved_wall_seconds > 0 else 0.0
            ),
            "stages": stages,
        }
        if notes:
            payload["notes"] = [str(note) for note in notes if str(note).strip()]
        return payload

    def render_lines(
        self,
        *,
        total_items: int | None = None,
        wall_seconds: float | None = None,
        limit: int | None = None,
    ) -> list[str]:
        summary = self.summary(total_items=total_items, wall_seconds=wall_seconds)
        if not summary.get("enabled"):
            return []
        stages = summary.get("stages", {})
        ordered = sorted(
            stages.items(),
            key=lambda item: float(item[1].get("total_seconds", 0.0)),
            reverse=True,
        )
        if limit is not None:
            ordered = ordered[: max(0, int(limit))]
        lines: list[str] = []
        for name, payload in ordered:
            total_seconds = float(payload.get("total_seconds", 0.0))
            share_pct = float(payload.get("share_of_wall_time_percent", 0.0))
            avg_ms_per_call = float(payload.get("avg_ms_per_call", 0.0))
            avg_ms_per_item = payload.get("avg_ms_per_item")
            if avg_ms_per_item is None:
                item_text = "n/a"
            else:
                item_text = f"{float(avg_ms_per_item):.3f} ms/item"
            lines.append(
                f"{name}: {total_seconds:.2f}s ({share_pct:.1f}% wall, {avg_ms_per_call:.2f} ms/call, {item_text})"
            )
        return lines
