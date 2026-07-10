"""Public method-neutral tracking dispatch API."""

from __future__ import annotations

from typing import Any, Mapping

from .contracts import TrackingObservations, TrackingResult


TRACKING_METHOD_SINGLE_SUBJECT_PER_ARENA = "single_subject_per_arena"


def available_tracking_methods() -> tuple[str, ...]:
    """Return tracking methods supported by the current runtime."""

    return (TRACKING_METHOD_SINGLE_SUBJECT_PER_ARENA,)


def build_tracking(
    observations: TrackingObservations,
    *,
    method: str,
    parameters: Mapping[str, Any] | None = None,
) -> TrackingResult:
    """Run one registered tracker against the shared observation contract."""

    normalized_method = str(method).strip()
    normalized_parameters = dict(parameters or {})
    if normalized_method == TRACKING_METHOD_SINGLE_SUBJECT_PER_ARENA:
        # Local import keeps the method implementation dependent on the shared
        # contracts without creating a module-import cycle.
        from .single_subject_per_arena import build_single_subject_per_arena_tracking

        unexpected = set(normalized_parameters) - {"conflict_policy"}
        if unexpected:
            raise ValueError(
                "Unsupported single_subject_per_arena parameters: "
                + ", ".join(sorted(unexpected))
            )
        result = build_single_subject_per_arena_tracking(
            observations.arena_ids,
            observations.frame_indices,
            conflict_policy=str(normalized_parameters.get("conflict_policy", "fail")),
        )
    else:
        raise ValueError(
            f"Unknown tracking method {normalized_method!r}; "
            f"available methods: {', '.join(available_tracking_methods())}."
        )
    result.validate_against(observations)
    return result


def write_tracking_run(**kwargs: Any) -> Any:
    """Build and persist a run through the shared tracking writer contract.

    The local import avoids a cycle while keeping callers independent of the
    currently registered method implementation module.
    """

    from .single_subject_per_arena import write_tracking_run as _write_tracking_run

    return _write_tracking_run(**kwargs)


__all__ = [
    "TRACKING_METHOD_SINGLE_SUBJECT_PER_ARENA",
    "TrackingObservations",
    "TrackingResult",
    "available_tracking_methods",
    "build_tracking",
    "write_tracking_run",
]
