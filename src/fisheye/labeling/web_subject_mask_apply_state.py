"""Run-wide mask Apply ownership and durable, source-bound successor offers."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
from types import SimpleNamespace

import numpy as np

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


def committed_by_this_apply(runtime, *, apply_id, checkpoint_revision, edit_revision, committed_mask, checkpoint_mask):
    """True when a retried apply finds its own committed canonical write.

    A crash between the revision write and SQLite finalization leaves the
    run one revision past the checkpoint and stamped with this apply id.
    The row must still hold exactly the checkpoint's mask.
    """

    if not (
        runtime.refined.group.attrs.get("edit_revision_last_apply_id") == apply_id
        and int(checkpoint_revision) + 1 == int(edit_revision)
    ):
        return False
    if not np.array_equal(np.asarray(committed_mask) > 0, np.asarray(checkpoint_mask) > 0):
        raise ValueError(f"Apply {apply_id} revision is committed but a row differs from its checkpoint.")
    return True


def classify_apply_checkpoints(runtime, checkpoints, *, apply_id, edit_revision, target_path, source_rowset_path):
    """Validate claimed checkpoints and split them into write, stale, and committed rows.

    Rows at the current revision get their edited stack prepared (one masks_roi
    chunk read per chunk).  Rows one revision behind that this apply already
    committed are verified pixel-exact; other revision mismatches are stale.
    """

    from . import web_mask_tail_border as tail_border
    from .web_runtimes import _subject_mask_checkpoint_mask, _subject_mask_row_identity

    checkpoint_ids: list[str] = []
    applied_rows: list[int] = []
    edited_stacks: list[np.ndarray] = []
    before_area_total = 0
    after_area_total = 0
    stale_checkpoint_ids: list[str] = []
    stale_rows: list[int] = []
    committed_checkpoint_ids: list[str] = []
    committed_rows: list[int] = []
    tail_border_actions: list[dict[str, object]] = []
    masks_array = runtime.refined.group["masks_roi"]
    tail_border.preflight_run(runtime)
    row_chunk: int | None = None
    cached_chunk_index = -1
    cached_chunk: np.ndarray | None = None
    scoped_row_set = set(int(value) for value in runtime.roi_indices.tolist())
    for checkpoint in checkpoints:
        checkpoint_target_path = str(checkpoint.get("target_run_path") or "")
        if checkpoint_target_path != target_path:
            raise ValueError(
                f"checkpoint target mismatch: expected {target_path}, got {checkpoint_target_path}"
            )
        checkpoint_source_rowset = str(checkpoint.get("source_rowset_path") or "")
        if checkpoint_source_rowset and checkpoint_source_rowset != source_rowset_path:
            raise ValueError(
                f"checkpoint source rowset mismatch: expected {source_rowset_path}, got {checkpoint_source_rowset}"
            )
        checkpoint_revision = int(checkpoint.get("target_edit_revision") or 0)
        roi_idx = int(checkpoint.get("roi_idx") or 0)
        if roi_idx not in scoped_row_set:
            raise ValueError(f"checkpoint row {roi_idx} is outside the active task row scope.")
        if checkpoint_revision != edit_revision:
            if committed_by_this_apply(
                runtime, apply_id=apply_id, checkpoint_revision=checkpoint_revision,
                edit_revision=edit_revision, committed_mask=masks_array[roi_idx, runtime.comp_idx],
                checkpoint_mask=_subject_mask_checkpoint_mask(checkpoint),
            ):
                committed_checkpoint_ids.append(str(checkpoint.get("checkpoint_id") or ""))
                committed_rows.append(roi_idx)
                continue
            stale_checkpoint_ids.append(str(checkpoint.get("checkpoint_id") or ""))
            stale_rows.append(roi_idx)
            continue
        metadata = checkpoint.get("metadata")
        if isinstance(metadata, Mapping):
            expected_identity = metadata.get("row_identity")
            if isinstance(expected_identity, Mapping):
                current_identity = _subject_mask_row_identity(runtime, roi_idx)
                for key, expected_value in expected_identity.items():
                    if key not in current_identity:
                        continue
                    if str(current_identity.get(key)) != str(expected_value):
                        raise ValueError(
                            f"checkpoint row identity mismatch for row {roi_idx}, field {key}: "
                            f"expected {expected_value}, got {current_identity.get(key)}"
                        )
        edited_mask = _subject_mask_checkpoint_mask(checkpoint)
        if row_chunk is None:
            row_chunk = int(masks_array.chunks[0])
        chunk_index = roi_idx // row_chunk
        if chunk_index != cached_chunk_index:
            chunk_start = chunk_index * row_chunk
            cached_chunk = np.asarray(
                masks_array[chunk_start:min(chunk_start + row_chunk, int(masks_array.shape[0]))],
                dtype=np.uint8,
            )
            cached_chunk_index = chunk_index
        assert cached_chunk is not None
        current_stack = cached_chunk[roi_idx - chunk_index * row_chunk]
        before_mask = (np.asarray(current_stack[runtime.comp_idx], dtype=np.uint8) > 0).astype(np.uint8)
        edited_stack, tail_action = tail_border.prepare_apply_row(runtime, checkpoint, roi_idx=roi_idx, current_stack=current_stack, before_mask=before_mask, edited_mask=edited_mask)
        if tail_action is not None:
            tail_border_actions.append(tail_action)
        checkpoint_ids.append(str(checkpoint.get("checkpoint_id") or ""))
        applied_rows.append(roi_idx)
        edited_stacks.append(edited_stack)
        before_area_total += int(before_mask.sum())
        after_area_total += int(edited_mask.sum())
    return SimpleNamespace(
        checkpoint_ids=checkpoint_ids,
        applied_rows=applied_rows,
        edited_stacks=edited_stacks,
        before_area_total=before_area_total,
        after_area_total=after_area_total,
        stale_checkpoint_ids=stale_checkpoint_ids,
        stale_rows=stale_rows,
        committed_checkpoint_ids=committed_checkpoint_ids,
        committed_rows=committed_rows,
        tail_border_actions=tail_border_actions,
    )


def commit_mask_edit_revision(runtime, *, apply_id, revision):
    """Stamp the revision and its apply id in one metadata write."""

    attrs = {
        "edit_revision": int(revision),
        "edit_revision_updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "edit_revision_last_apply_id": apply_id,
    }
    if "mask_rle" in runtime.refined.group:
        attrs["mask_rle_stale_since_edit_revision"] = int(revision)
    runtime.refined.group.attrs.update(attrs)


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
    result = {key: offer[key] for key in TAIL_OFFER_KEYS}
    if "tail_refresh_visible_endpoint_rows" in offer:
        result["tail_refresh_visible_endpoint_rows"] = offer["tail_refresh_visible_endpoint_rows"]
    return result
