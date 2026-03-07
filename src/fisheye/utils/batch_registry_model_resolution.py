from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Protocol, Sequence


@dataclass(frozen=True)
class ResolvedModel:
    model_path: str
    payload: Dict[str, Any]


class PlanLike(Protocol):
    recording_dir: Path
    zarr_path: Path
    camera_id: Optional[str]


ResolvePlanFn = Callable[[PlanLike], ResolvedModel]
EventFn = Callable[[Dict[str, Any]], None]


def resolve_registry_models_for_plans(
    *,
    plans: Sequence[PlanLike],
    resolve_plan: ResolvePlanFn,
    on_event: Optional[EventFn] = None,
) -> tuple[Dict[str, ResolvedModel], Dict[str, str]]:
    resolved: Dict[str, ResolvedModel] = {}
    errors: Dict[str, str] = {}
    resolved_by_recording: Dict[str, ResolvedModel] = {}
    errors_by_recording: Dict[str, str] = {}
    total = len(plans)

    for index, plan in enumerate(plans, start=1):
        key = str(plan.zarr_path.resolve())
        recording_key = str(plan.recording_dir.resolve())
        base_event: Dict[str, Any] = {
            "index": index,
            "total": total,
            "recording": str(plan.recording_dir),
            "zarr": str(plan.zarr_path),
            "camera_id": plan.camera_id,
        }
        cached = resolved_by_recording.get(recording_key)
        if cached is not None:
            resolved[key] = cached
            selected = cached.payload.get("selected")
            selected_payload = selected if isinstance(selected, dict) else {}
            if on_event is not None:
                on_event(
                    {
                        **base_event,
                        "event": "model_resolution_cached",
                        "selected_model_path": cached.model_path,
                        "selected_run_id": selected_payload.get("run_id"),
                        "selected_set_id": selected_payload.get("set_id"),
                    }
                )
            continue

        cached_error = errors_by_recording.get(recording_key)
        if cached_error is not None:
            errors[key] = cached_error
            if on_event is not None:
                on_event(
                    {
                        **base_event,
                        "event": "model_resolution_failed",
                        "error": cached_error,
                    }
                )
            continue

        if on_event is not None:
            on_event({**base_event, "event": "model_resolution_start"})

        try:
            model = resolve_plan(plan)
            resolved[key] = model
            resolved_by_recording[recording_key] = model
            selected = model.payload.get("selected")
            selected_payload = selected if isinstance(selected, dict) else {}
            if on_event is not None:
                on_event(
                    {
                        **base_event,
                        "event": "model_resolution_ok",
                        "selected_model_path": model.model_path,
                        "selected_run_id": selected_payload.get("run_id"),
                        "selected_set_id": selected_payload.get("set_id"),
                    }
                )
        except Exception as exc:
            message = str(exc)
            errors[key] = message
            errors_by_recording[recording_key] = message
            if on_event is not None:
                on_event(
                    {
                        **base_event,
                        "event": "model_resolution_failed",
                        "error": message,
                    }
                )

    return resolved, errors
