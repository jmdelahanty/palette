"""Component review status for subject-mask tasks, including deferred requests.

While background Apply effects are owed on a mask run, the run lock is held
by the effects worker and review-status writes would wait on it.  A labeler
may still record a non-approval review state (``needs_review``): it is kept as
a durable task event and applied by the worker, under the run lock, once the
effects finish.  Approval stays gated on complete effects, because approval
certifies the derived products.

``apply_review_status_locked`` is the single writer used by the browser route
and by the worker.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

DEFERRED_REVIEW_EVENT = "subject_mask_review_status_deferred"
REVIEW_STATUS_EVENT = "set_review_status"
DEFERRABLE_REVIEW_STATES = frozenset({"needs_review"})


def _review_target(runtime) -> dict[str, str]:
    return {
        "component_name": str(runtime.component_name),
        "refined_run": str(runtime.refined.run_name),
    }


def pending_deferred_review(store, runtime) -> dict[str, object] | None:
    """The newest deferred review request not yet superseded by a written status."""

    if store is None:
        return None
    target = _review_target(runtime)
    deferred = store.get_event_for_target(
        task_id=str(runtime.task_id), event_type=DEFERRED_REVIEW_EVENT, target=target
    )
    if not deferred:
        return None
    written = store.get_event_for_target(
        task_id=str(runtime.task_id), event_type=REVIEW_STATUS_EVENT, target=target
    )
    if written and str(written.get("created_at_utc") or "") >= str(
        deferred.get("created_at_utc") or ""
    ):
        return None
    request = deferred.get("after")
    return dict(request) if isinstance(request, Mapping) else None


def record_deferred_review(store, runtime, *, user: str, request: Mapping[str, object]) -> dict[str, object]:
    """Durably record a review state to apply once owed effects complete."""

    state = str(request.get("state") or "")
    if state not in DEFERRABLE_REVIEW_STATES:
        raise ValueError(f"Review state {state!r} cannot be deferred past owed Apply effects.")
    payload = {**dict(request), "reviewer": user}
    store.record_event(
        task_id=runtime.task_id,
        recording_id=runtime.recording_id,
        user=user,
        event_type=DEFERRED_REVIEW_EVENT,
        target=_review_target(runtime),
        after=payload,
    )
    return payload


def apply_review_status_locked(
    store,
    runtime,
    *,
    user: str,
    request: Mapping[str, object],
    refresh_registry: Callable[..., bool],
    registry_scope: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    """Write one component review status; the caller holds the refined run lock.

    Returns the component payload, run payload, and the recorded audit event.
    """

    from fisheye.tune import refined_subject_mask_review as review_mod

    before_component_reviews = runtime.refined.group.attrs.get("component_review_statuses")
    before_run_review = runtime.refined.group.attrs.get("refined_subject_mask_review_status")
    component_payload, run_payload = review_mod.apply_component_review_status(
        runtime.refined.parent,
        str(runtime.refined.run_name),
        runtime.refined.group,
        component_name=runtime.component_name,
        state=str(request["state"]),
        method=str(request.get("method") or runtime.review_method or "manual"),
        intended_use=str(request.get("intended_use") or runtime.review_intended_use or "training"),
        reviewer=user,
        notes=str(request.get("notes") or runtime.review_notes or "").strip() or None,
        zarr_path=runtime.zarr_path,
    )
    event = store.record_event(
        task_id=runtime.task_id,
        recording_id=runtime.recording_id,
        user=user,
        event_type=REVIEW_STATUS_EVENT,
        target=_review_target(runtime),
        before={
            "component_review_statuses": dict(before_component_reviews) if isinstance(before_component_reviews, Mapping) else None,
            "run_review_status": dict(before_run_review) if isinstance(before_run_review, Mapping) else None,
        },
        after={
            "component_review_status": component_payload,
            "run_review_status": run_payload,
        },
    )
    refresh_registry(
        store=store,
        task_id=runtime.task_id,
        recording_id=runtime.recording_id,
        user=user,
        workflow_kind="subject_mask_component",
        scope=registry_scope.get("scope") or {},
        zarr_path=runtime.zarr_path,
        dataset_id=registry_scope.get("dataset_id"),
        zarr_use=registry_scope.get("zarr_use"),
    )
    return component_payload, run_payload, event


def apply_pending_deferred_review_locked(
    store,
    runtime,
    *,
    refresh_registry: Callable[..., bool],
    registry_scope: Mapping[str, object],
) -> dict[str, object] | None:
    """Apply a deferred review request after effects complete (run lock held)."""

    request = pending_deferred_review(store, runtime)
    if request is None:
        return None
    component_payload, _run_payload, _event = apply_review_status_locked(
        store,
        runtime,
        user=str(request.get("reviewer") or ""),
        request=request,
        refresh_registry=refresh_registry,
        registry_scope=registry_scope,
    )
    return component_payload


__all__ = [
    "DEFERRABLE_REVIEW_STATES",
    "DEFERRED_REVIEW_EVENT",
    "apply_pending_deferred_review_locked",
    "apply_review_status_locked",
    "pending_deferred_review",
    "record_deferred_review",
]
