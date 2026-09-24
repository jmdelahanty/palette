"""Canonical apply and crash recovery for browser keypoint checkpoints."""

from __future__ import annotations

import copy
from datetime import datetime, timezone
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from fisheye.shared.zarr_helpers import archive_metadata_publication_lock

from .web_keypoint_checkpoints import (
    KEYPOINT_APPLY_INFLIGHT_ATTR,
    KEYPOINT_APPLY_RECEIPTS_ATTR,
    KEYPOINT_CHECKPOINT_COMPONENT,
    KEYPOINT_CHECKPOINT_SAVE_MODE,
    KeypointCheckpointConflict,
    _APPLY_SNAPSHOT_LIMIT,
    _RECEIPT_HISTORY_LIMIT,
    _bindings,
    _checkpoint_metadata,
    _checkpoint_payload,
    _decoded_json_value,
    _edit_revision,
    _fail_closed_recovered_rows,
    _json_value,
    _mark_changed_rows_stale,
    _row_state_documents,
    _write_intended_rows,
    checkpoint_snapshot_digest,
    keypoint_browser_save_mode,
    validate_keypoint_checkpoint,
)

KEYPOINT_APPLY_INFLIGHT_SCHEMA = "palette.keypoint_checkpoint_apply_inflight.v2"


def _receipts(session: object) -> dict[str, object]:
    attrs = getattr(getattr(session, "refined"), "attrs")
    value = attrs.get(KEYPOINT_APPLY_RECEIPTS_ATTR)
    return dict(value) if isinstance(value, Mapping) else {}


def _fresh_session_for_apply(
    runtime: object, backend_module: object
) -> tuple[object, object]:
    """Resolve a fresh unconsolidated apply view when the archive is on disk."""

    current = getattr(runtime, "review_session")
    archive = Path(str(getattr(current, "zarr_path"))).expanduser()
    resolver = getattr(backend_module, "resolve_review_session", None)
    if not callable(resolver) or not (archive / "zarr.json").exists():
        return current, runtime
    rows = getattr(runtime, "task_roi_indices", None)
    target_rows = (
        np.asarray(rows, dtype=np.int64).tolist() if rows is not None else None
    )
    fresh = resolver(
        str(archive),
        refined_run=str(getattr(current, "refined_run")),
        crop_run=str(getattr(current, "crop_run")),
        include_all=True,
        target_roi_indices=target_rows,
    )
    # Preserve the presentation queue; apply addresses validated ROI identities
    # through a private one-row session view.
    fresh.failures = np.asarray(getattr(current, "failures"), dtype=np.int64).copy()
    fresh_runtime = copy.copy(runtime)
    fresh_runtime.review_session = fresh
    return fresh, fresh_runtime


def _history_result(
    history: Sequence[Mapping[str, object]], *, apply_id: str, expected_digest: str
) -> dict[str, object]:
    digest = checkpoint_snapshot_digest(history)
    if digest != expected_digest:
        raise KeypointCheckpointConflict(
            "The apply_id belongs to a different keypoint checkpoint snapshot."
        )
    before_values = [int(row.get("edit_revision_before") or 0) for row in history]
    after_values = [int(row.get("edit_revision_after") or 0) for row in history]
    row_results = []
    for row in history:
        payload = _checkpoint_payload(row)
        row_results.append(
            _applied_row_result(
                row,
                payload,
                {"changed": True, "reason_updated": True, "stale_touched": 0},
            )
        )
    return {
        "apply_id": apply_id,
        "checkpoint_snapshot_sha256": digest,
        "already_applied": True,
        "applied_checkpoint_count": len(history),
        "rows": [int(row.get("roi_idx") or 0) for row in history],
        "row_results": row_results,
        "edit_revision_before": before_values[0] if before_values else 0,
        "edit_revision_after": after_values[0] if after_values else 0,
        "canonical_zarr_mutated": False,
        "saved": True,
        "applied": True,
    }


def _applied_row_result(
    checkpoint: Mapping[str, object],
    payload: Mapping[str, object],
    operation_result: Mapping[str, object],
) -> dict[str, object]:
    metadata = _checkpoint_metadata(checkpoint)
    binding = metadata.get("binding")
    identity = (
        dict(binding.get("row_identity"))
        if isinstance(binding, Mapping)
        and isinstance(binding.get("row_identity"), Mapping)
        else {"roi_idx": int(checkpoint.get("roi_idx") or 0)}
    )
    intended = metadata.get("intended_row_state")
    fields = intended.get("fields") if isinstance(intended, Mapping) else None
    fields = fields if isinstance(fields, Mapping) else {}
    status_field_names = {
        "heading": "heading",
        "refined_success": "refined_success",
        "usable_keypoints": "usable_keypoints",
        "edit_applied": "edit_applied",
        "confidence_valid": "confidence_valid",
        "geometry_valid": "geometry_valid",
        "heading_finite": "heading_finite",
        "heading_usable": "heading_usable",
    }
    status = {
        status_name: _decoded_json_value(fields[field_name])
        for field_name, status_name in status_field_names.items()
        if fields.get(field_name) is not None
    }
    heading_value = status.get("heading")
    if isinstance(heading_value, float) and not math.isfinite(heading_value):
        status["heading"] = None
    for source_name in ("source_refined_row_id", "source_detect_row_index"):
        if identity.get(source_name) is not None:
            status[source_name] = identity[source_name]
    readback = {
        "roi_idx": int(checkpoint.get("roi_idx") or 0),
        "frame_idx": identity.get("frame_idx"),
        "reason": str(_decoded_json_value(fields.get("reason")) or ""),
        "status": status,
    }
    binding = metadata.get("binding")
    base_state = (
        binding.get("expected_row_state")
        if isinstance(binding, Mapping)
        else None
    )
    intended_changed = bool(
        isinstance(base_state, Mapping)
        and isinstance(intended, Mapping)
        and canonical_json_sha256(base_state) != canonical_json_sha256(intended)
    )
    return {
        "checkpoint_id": str(checkpoint.get("checkpoint_id") or ""),
        "operation": str(payload.get("operation") or ""),
        "row_identity": identity,
        "roi_idx": int(checkpoint.get("roi_idx") or 0),
        "frame_idx": identity.get("frame_idx"),
        "changed": intended_changed,
        "reason_updated": bool(
            isinstance(base_state, Mapping)
            and isinstance(base_state.get("fields"), Mapping)
            and base_state["fields"].get("reason") != fields.get("reason")  # type: ignore[index,union-attr]
        ),
        "stale_touched": int(operation_result.get("stale_touched") or 0),
        "readback": _json_value(readback),
    }


def apply_keypoint_checkpoints(
    store: object,
    runtime: object,
    backend_module: object,
    *,
    apply_id: str,
    checkpoint_snapshot_sha256: str,
) -> dict[str, object]:
    """Apply one immutable claimed checkpoint snapshot under the archive lock.

    Mutable Zarr arrays are not transactionally published as a generation.  If
    a write becomes uncertain, the claimed rows and an inflight Zarr receipt
    remain durable for a same-``apply_id`` recovery; they are never reported as
    applied or released for a different writer.
    """

    apply_id_value = str(apply_id).strip()
    expected_digest = str(checkpoint_snapshot_sha256).strip()
    if not apply_id_value:
        raise ValueError("Missing apply_id.")
    if not expected_digest:
        raise ValueError("Missing checkpoint_snapshot_sha256.")
    session = getattr(runtime, "review_session")
    if keypoint_browser_save_mode(session) != KEYPOINT_CHECKPOINT_SAVE_MODE:
        raise ValueError("Immutable keypoint delta review has no checkpoint apply step.")
    task_id = str(getattr(runtime, "task_id"))

    history = store.get_applied_session_checkpoints_by_apply_id(
        task_id=task_id, apply_id=apply_id_value
    )
    if history:
        fallback_result = _history_result(
            history, apply_id=apply_id_value, expected_digest=expected_digest
        )
        # Reconcile the narrow crash window where the durable receipt/store
        # finalization succeeded but clearing the matching inflight attr did
        # not.  A different current apply is never disturbed.
        with archive_metadata_publication_lock(str(getattr(session, "zarr_path"))):
            fresh_session, _fresh_runtime = _fresh_session_for_apply(
                runtime, backend_module
            )
            fresh_attrs = getattr(getattr(fresh_session, "refined"), "attrs")
            durable_receipt = _receipts(fresh_session).get(apply_id_value)
            if isinstance(durable_receipt, Mapping):
                if (
                    str(durable_receipt.get("checkpoint_snapshot_sha256") or "")
                    != expected_digest
                ):
                    raise KeypointCheckpointConflict(
                        "The apply_id Zarr receipt belongs to a different snapshot."
                    )
                result = {
                    **dict(durable_receipt),
                    "already_applied": True,
                    "canonical_zarr_mutated": False,
                    "saved": True,
                    "applied": True,
                }
            else:
                result = fallback_result
            stale_inflight = fresh_attrs.get(KEYPOINT_APPLY_INFLIGHT_ATTR)
            if isinstance(stale_inflight, Mapping) and (
                str(stale_inflight.get("apply_id") or "") == apply_id_value
                and str(stale_inflight.get("checkpoint_snapshot_sha256") or "")
                == expected_digest
            ):
                fresh_attrs.pop(KEYPOINT_APPLY_INFLIGHT_ATTR, None)
            if fresh_session is not getattr(runtime, "review_session"):
                runtime.review_session = fresh_session
        return result

    pending_effects = store.list_pending_session_checkpoint_apply_effects(
        task_id=task_id,
        component_name=KEYPOINT_CHECKPOINT_COMPONENT,
        limit=1,
    )
    if pending_effects:
        raise KeypointCheckpointConflict(
            "Finish the prior keypoint Apply record before applying another snapshot."
        )

    checkpoints = store.claim_session_checkpoints_for_apply(
        task_id=task_id,
        component_name=KEYPOINT_CHECKPOINT_COMPONENT,
        apply_id=apply_id_value,
        limit=_APPLY_SNAPSHOT_LIMIT,
        checkpoint_snapshot_sha256=expected_digest,
    )
    if not checkpoints:
        # Compatibility with sidecars created before same-ID claim replay was
        # added atomically to LabelingStore.
        checkpoints = store.list_session_checkpoints(
            task_id=task_id,
            state="applying",
            component_name=KEYPOINT_CHECKPOINT_COMPONENT,
            apply_id=apply_id_value,
            limit=_APPLY_SNAPSHOT_LIMIT,
        )
    if not checkpoints:
        raise KeypointCheckpointConflict(
            "No checkpoint snapshot was claimed; another apply may own the pending rows."
        )
    actual_digest = checkpoint_snapshot_digest(checkpoints)
    attrs = getattr(getattr(session, "refined"), "attrs")
    if actual_digest != expected_digest:
        inflight_before = attrs.get(KEYPOINT_APPLY_INFLIGHT_ATTR)
        if not (
            isinstance(inflight_before, Mapping)
            and str(inflight_before.get("apply_id") or "") == apply_id_value
        ):
            store.release_session_checkpoints_apply(
                task_id=task_id, apply_id=apply_id_value
            )
        raise KeypointCheckpointConflict(
            "The pending keypoint checkpoint snapshot changed before apply."
        )

    write_started = False
    rows = [int(row.get("roi_idx") or 0) for row in checkpoints]
    try:
        with archive_metadata_publication_lock(str(getattr(session, "zarr_path"))):
            session, apply_runtime = _fresh_session_for_apply(runtime, backend_module)
            attrs = getattr(getattr(session, "refined"), "attrs")
            history = store.get_applied_session_checkpoints_by_apply_id(
                task_id=task_id, apply_id=apply_id_value
            )
            if history:
                return _history_result(
                    history,
                    apply_id=apply_id_value,
                    expected_digest=expected_digest,
                )
            receipts = _receipts(session)
            receipt = receipts.get(apply_id_value)
            inflight = attrs.get(KEYPOINT_APPLY_INFLIGHT_ATTR)
            owned_inflight = isinstance(inflight, Mapping)
            if owned_inflight and (
                str(inflight.get("apply_id") or "") != apply_id_value
                or str(inflight.get("checkpoint_snapshot_sha256") or "")
                != actual_digest
            ):
                raise KeypointCheckpointConflict(
                    "The durable keypoint receipt conflicts with another inflight apply."
                    if isinstance(receipt, Mapping)
                    else "A different uncertain keypoint apply owns the canonical target."
                )
            intended = _intended_row_states(checkpoints)

            if isinstance(receipt, Mapping):
                if str(receipt.get("checkpoint_snapshot_sha256") or "") != actual_digest:
                    raise KeypointCheckpointConflict(
                        "The apply_id Zarr receipt belongs to a different snapshot."
                    )
                # The canonical write already has a durable receipt.  A store
                # finalization failure must keep this same apply_id claimed so
                # restart recovery cannot release it to a different writer.
                write_started = True
                if bool(getattr(session, "recovered_roi_only", False)):
                    current = _validated_current_rows(
                        checkpoints, apply_runtime, session, intended, uncertain=True
                    )[1]
                    _write_intended_rows(session, intended, current)
                    _verify_rows(
                        session,
                        intended,
                        message="Recovered keypoint receipt {} did not restore its intended row.",
                        checkpoints=checkpoints,
                    )
                _finalize_store(
                    store,
                    checkpoints,
                    apply_id=apply_id_value,
                    edit_revision_before=int(receipt.get("edit_revision_before") or 0),
                    edit_revision_after=int(receipt.get("edit_revision_after") or 0),
                )
                if owned_inflight:
                    attrs.pop(KEYPOINT_APPLY_INFLIGHT_ATTR, None)
                if session is not getattr(runtime, "review_session"):
                    runtime.review_session = session
                return {
                    **dict(receipt),
                    "already_applied": True,
                    "canonical_zarr_mutated": False,
                    "saved": True,
                    "applied": True,
                }

            # Without an owned inflight marker every row must still be at its
            # exact staged base.  With one, a prior attempt of this same
            # apply_id may have left any snapshot row partially written.
            write_started = owned_inflight
            payloads, current = _validated_current_rows(
                checkpoints, apply_runtime, session, intended, uncertain=owned_inflight
            )
            # The global revision is context, not row freshness.  A checkpoint
            # for an unchanged row remains valid after a different row commits.
            edit_revision_before = (
                int(inflight.get("edit_revision_before") or 0)
                if owned_inflight
                else _edit_revision(session)
            )
            if not owned_inflight:
                attrs[KEYPOINT_APPLY_INFLIGHT_ATTR] = {
                    "schema": KEYPOINT_APPLY_INFLIGHT_SCHEMA,
                    "apply_id": apply_id_value,
                    "checkpoint_snapshot_sha256": actual_digest,
                    "checkpoint_ids": [
                        str(row.get("checkpoint_id") or "") for row in checkpoints
                    ],
                    "edit_revision_before": edit_revision_before,
                    "started_at_utc": datetime.now(timezone.utc).isoformat(),
                }
            write_started = True
            _write_intended_rows(session, intended, current)
            changed_operations: dict[int, str] = {}
            for checkpoint in checkpoints:
                roi_idx = int(checkpoint.get("roi_idx") or 0)
                binding = _checkpoint_metadata(checkpoint).get("binding")
                base_row_state = (
                    binding.get("expected_row_state")
                    if isinstance(binding, Mapping)
                    else None
                )
                if not isinstance(base_row_state, Mapping):
                    raise KeypointCheckpointConflict(
                        "Keypoint checkpoint base row state is missing."
                    )
                if canonical_json_sha256(base_row_state) != canonical_json_sha256(
                    intended[roi_idx]
                ):
                    changed_operations[roi_idx] = str(
                        payloads[roi_idx].get("operation") or ""
                    )
            stale_touched = _mark_changed_rows_stale(session, changed_operations)
            _verify_rows(
                session,
                intended,
                message="Keypoint apply did not converge to its durable intended row state.",
            )
            row_results = [
                _applied_row_result(
                    checkpoint,
                    payloads[int(checkpoint.get("roi_idx") or 0)],
                    {
                        "stale_touched": stale_touched.get(
                            int(checkpoint.get("roi_idx") or 0), 0
                        )
                    },
                )
                for checkpoint in checkpoints
            ]
            edit_revision_after = int(edit_revision_before) + 1
            attrs["edit_revision"] = int(edit_revision_after)
            attrs["edit_revision_updated_at_utc"] = datetime.now(
                timezone.utc
            ).isoformat()
            attrs["edit_revision_last_apply_id"] = apply_id_value
            receipt_payload: dict[str, object] = {
                "schema": "palette.keypoint_checkpoint_apply_receipt.v1",
                "apply_id": apply_id_value,
                "checkpoint_snapshot_sha256": actual_digest,
                "applied_checkpoint_count": len(checkpoints),
                "rows": rows,
                "checkpoint_ids": [
                    str(row.get("checkpoint_id") or "") for row in checkpoints
                ],
                "row_results": row_results,
                "edit_revision_before": edit_revision_before,
                "edit_revision_after": edit_revision_after,
                "applied_at_utc": datetime.now(timezone.utc).isoformat(),
            }
            receipts[apply_id_value] = receipt_payload
            if len(receipts) > _RECEIPT_HISTORY_LIMIT:
                ordered = sorted(
                    receipts.items(),
                    key=lambda pair: str(
                        pair[1].get("applied_at_utc")
                        if isinstance(pair[1], Mapping)
                        else ""
                    ),
                )
                receipts = dict(ordered[-_RECEIPT_HISTORY_LIMIT:])
            attrs[KEYPOINT_APPLY_RECEIPTS_ATTR] = receipts
            _finalize_store(
                store,
                checkpoints,
                apply_id=apply_id_value,
                edit_revision_before=edit_revision_before,
                edit_revision_after=edit_revision_after,
            )
            attrs.pop(KEYPOINT_APPLY_INFLIGHT_ATTR, None)
            if session is not getattr(runtime, "review_session"):
                runtime.review_session = session
            return {
                **receipt_payload,
                "already_applied": False,
                "canonical_zarr_mutated": any(
                    bool(result.get("changed")) for result in row_results
                ),
                "saved": True,
                "applied": True,
            }
    except Exception:
        if write_started:
            _fail_closed_recovered_rows(session, rows)
        else:
            store.release_session_checkpoints_apply(
                task_id=task_id, apply_id=apply_id_value
            )
        raise


def _intended_row_states(
    checkpoints: Sequence[Mapping[str, object]],
) -> dict[int, Mapping[str, object]]:
    intended: dict[int, Mapping[str, object]] = {}
    for checkpoint in checkpoints:
        document = _checkpoint_metadata(checkpoint).get("intended_row_state")
        if not isinstance(document, Mapping):
            raise KeypointCheckpointConflict(
                "Keypoint checkpoint intended row state changed."
            )
        roi_idx = int(checkpoint.get("roi_idx") or 0)
        if roi_idx in intended:
            raise KeypointCheckpointConflict(
                "A keypoint row appears twice in one snapshot."
            )
        intended[roi_idx] = document
    return intended


def _validated_current_rows(
    checkpoints: Sequence[Mapping[str, object]],
    apply_runtime: object,
    session: object,
    intended: Mapping[int, Mapping[str, object]],
    *,
    uncertain: bool,
) -> tuple[dict[int, Mapping[str, object]], dict[int, Mapping[str, object]]]:
    """Validate every checkpoint before the first write, reading each array once.

    ``uncertain`` admits rows left at their intended state, or partway between
    base and intended, by an interrupted attempt of the same apply_id.
    """

    rows = sorted(intended)
    current = _row_state_documents(session, rows)
    bindings = _bindings(session, rows, row_states=current)
    payloads: dict[int, Mapping[str, object]] = {}
    for checkpoint in checkpoints:
        roi_idx = int(checkpoint.get("roi_idx") or 0)
        target = intended[roi_idx]
        payloads[roi_idx] = validate_keypoint_checkpoint(
            checkpoint,
            apply_runtime,
            allowed_row_state_sha256=(
                [canonical_json_sha256(target)] if uncertain else []
            ),
            allowed_intermediate_row_state=target if uncertain else None,
            current_binding=bindings[roi_idx],
        )
    return payloads, current


def _verify_rows(
    session: object,
    intended: Mapping[int, Mapping[str, object]],
    *,
    message: str,
    checkpoints: Sequence[Mapping[str, object]] = (),
) -> None:
    actual = _row_state_documents(session, sorted(intended))
    ids = {
        int(row.get("roi_idx") or 0): str(row.get("checkpoint_id") or "")
        for row in checkpoints
    }
    for roi_idx, document in intended.items():
        if canonical_json_sha256(actual[roi_idx]) != canonical_json_sha256(document):
            raise RuntimeError(message.format(ids.get(roi_idx, roi_idx)))


def _finalize_store(
    store: object,
    checkpoints: Sequence[Mapping[str, object]],
    *,
    apply_id: str,
    edit_revision_before: int,
    edit_revision_after: int,
) -> None:
    updated = store.mark_session_checkpoints_applied(  # type: ignore[attr-defined]
        checkpoint_ids=[str(row.get("checkpoint_id") or "") for row in checkpoints],
        apply_id=apply_id,
        edit_revision_before=int(edit_revision_before),
        edit_revision_after=int(edit_revision_after),
        require_secondary_effects=True,
    )
    if int(updated) != len(checkpoints):
        raise RuntimeError(
            "The keypoint apply receipt could not finalize every claimed checkpoint."
        )


__all__ = ["apply_keypoint_checkpoints"]
