"""Run-wide mask Apply ownership and durable, source-bound successor offers."""

from __future__ import annotations

from collections.abc import Mapping

TAIL_SUCCESSOR_EVENT = "mask_apply_tail_successor"
TAIL_OFFER_KEYS = (
    "tail_refresh_status",
    "tail_refresh_version",
    "tail_refresh_tasks",
    "tail_refresh_failures",
    "tail_refresh_valid_rows",
    "tail_refresh_training_eligible_rows",
    "tail_refresh_manual_point_count",
    "tail_refresh_mask_revision",
)


def pending_mask_run_effects(store, runtime):
    if store is None:
        return []
    return store.list_pending_subject_mask_run_effects(
        zarr_path=runtime.zarr_path,
        refined_run=runtime.refined.run_name,
        task_id=runtime.task_id,
    )


def require_mask_apply_ownership(store, runtime, apply_id):
    """Recheck under the mask lock before pixels or secondary effects mutate."""
    for pending in pending_mask_run_effects(store, runtime):
        if pending["task_id"] != runtime.task_id or pending["apply_id"] != apply_id:
            raise RuntimeError(
                "Finish the pending mask Apply in task "
                + str(pending["task_id"])
                + " before changing this mask run."
            )


def tail_successor_event(store, runtime, *, apply_id=None, expected_mask_revision=None):
    """Resolve display evidence by exact Apply ID or latest event for this task.

    This offers a task, not scientific authority. Executing a pending effect
    additionally validates the recorded version before reusing its publication.
    """
    if store is None:
        return None
    if apply_id is not None:
        event = store.get_event_for_target(
            task_id=runtime.task_id,
            event_type=TAIL_SUCCESSOR_EVENT,
            target={"apply_id": str(apply_id)},
        )
    else:
        events = store.list_events(
            task_id=runtime.task_id,
            event_type=TAIL_SUCCESSOR_EVENT,
            limit=1,
        )
        event = events[0] if events else None
    if event is None:
        return None
    after = event.get("after") or {}
    binding = after.get("source_bindings") or {}
    proof = binding.get("mask_apply_refresh") or {}
    offer = after.get("tail_refresh")
    revision = expected_mask_revision
    if revision is None:
        revision = int(runtime.refined.group.attrs.get("edit_revision", 0))
    if proof.get("source_mask_edit_revision") != revision:
        if apply_id is None:
            return None  # An older offer must not represent the current edit.
        raise RuntimeError("Tail successor receipt has a different mask revision")
    if (
        not isinstance(offer, Mapping)
        or not all(key in offer for key in TAIL_OFFER_KEYS)
        or proof.get("source_mask_run")
        != f"refined_subject_masks_runs/{runtime.refined.run_name}"
        or proof.get("apply_id") != event.get("target", {}).get("apply_id")
        or (apply_id is not None and proof.get("apply_id") != str(apply_id))
        or offer.get("tail_refresh_mask_revision") != revision
        or offer.get("tail_refresh_status") != "complete"
    ):
        raise RuntimeError("Tail successor receipt has a conflicting source binding")
    return event


def tail_successor_offer(store, runtime, *, apply_id=None, expected_mask_revision=None):
    event = tail_successor_event(
        store,
        runtime,
        apply_id=apply_id,
        expected_mask_revision=expected_mask_revision,
    )
    if event is None:
        return {}
    offer = event["after"]["tail_refresh"]
    return {key: offer[key] for key in TAIL_OFFER_KEYS}
