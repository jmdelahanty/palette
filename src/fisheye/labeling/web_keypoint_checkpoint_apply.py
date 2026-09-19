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
    _apply_operation,
    _checkpoint_metadata,
    _checkpoint_payload,
    _converge_row_to_intended_state,
    _decoded_json_value,
    _edit_revision,
    _ensure_downstream_stale_after_row_apply,
    _json_value,
    _row_state_document,
    checkpoint_snapshot_digest,
    keypoint_browser_save_mode,
    validate_keypoint_checkpoint,
)


def _receipts(session: object) -> dict[str, object]:
    attrs = getattr(getattr(session, "refined"), "attrs")
    value = attrs.get(KEYPOINT_APPLY_RECEIPTS_ATTR)
    return dict(value) if isinstance(value, Mapping) else {}


def _fail_closed_recovered_training_row(session: object, *, roi_idx: int) -> None:
    if not bool(getattr(session, "recovered_roi_only", False)):
        return
    refined = getattr(session, "refined")
    getter = getattr(refined, "get", None)
    eligible = getter("training_eligible") if callable(getter) else None
    if eligible is not None:
        eligible[int(roi_idx)] = False


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

    pending_effect_reader = getattr(
        store, "list_pending_session_checkpoint_apply_effects", None
    )
    if callable(pending_effect_reader):
        pending_effects = pending_effect_reader(
            task_id=task_id,
            component_name=KEYPOINT_CHECKPOINT_COMPONENT,
            limit=1,
        )
        if pending_effects:
            raise KeypointCheckpointConflict(
                "Finish the prior keypoint apply audit and registry effects before applying another snapshot."
            )

    checkpoints = store.claim_session_checkpoints_for_apply(
        task_id=task_id,
        component_name=KEYPOINT_CHECKPOINT_COMPONENT,
        apply_id=apply_id_value,
        limit=_APPLY_SNAPSHOT_LIMIT,
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
            if isinstance(receipt, Mapping):
                if str(receipt.get("checkpoint_snapshot_sha256") or "") != actual_digest:
                    raise KeypointCheckpointConflict(
                        "The apply_id Zarr receipt belongs to a different snapshot."
                    )
                # The canonical write already has a durable receipt.  A store
                # finalization failure must keep this same apply_id claimed so
                # restart recovery cannot release it to a different writer.
                write_started = True
                receipt_inflight = attrs.get(KEYPOINT_APPLY_INFLIGHT_ATTR)
                if isinstance(receipt_inflight, Mapping) and (
                    str(receipt_inflight.get("apply_id") or "") != apply_id_value
                    or str(
                        receipt_inflight.get("checkpoint_snapshot_sha256") or ""
                    )
                    != actual_digest
                ):
                    raise KeypointCheckpointConflict(
                        "The durable keypoint receipt conflicts with another inflight apply."
                    )
                for checkpoint in checkpoints:
                    if not bool(getattr(session, "recovered_roi_only", False)):
                        break
                    checkpoint_id = str(checkpoint.get("checkpoint_id") or "")
                    metadata = _checkpoint_metadata(checkpoint)
                    intended = metadata.get("intended_row_state")
                    if not isinstance(intended, Mapping):
                        raise KeypointCheckpointConflict(
                            "The recovered keypoint receipt lost its intended row state."
                        )
                    validate_keypoint_checkpoint(
                        checkpoint,
                        apply_runtime,
                        allowed_row_state_sha256=[canonical_json_sha256(intended)],
                        allowed_intermediate_row_state=intended,
                    )
                    roi_idx = int(checkpoint.get("roi_idx") or 0)
                    _converge_row_to_intended_state(
                        session, roi_idx=roi_idx, intended=intended
                    )
                    if canonical_json_sha256(
                        _row_state_document(session, roi_idx)
                    ) != canonical_json_sha256(intended):
                        raise RuntimeError(
                            f"Recovered keypoint receipt {checkpoint_id} did not restore its intended row."
                        )
                updated = store.mark_session_checkpoints_applied(
                    checkpoint_ids=[
                        str(row.get("checkpoint_id") or "") for row in checkpoints
                    ],
                    apply_id=apply_id_value,
                    edit_revision_before=int(receipt.get("edit_revision_before") or 0),
                    edit_revision_after=int(receipt.get("edit_revision_after") or 0),
                )
                if int(updated) != len(checkpoints):
                    raise RuntimeError(
                        "The keypoint apply receipt could not finalize every claimed checkpoint."
                    )
                if isinstance(receipt_inflight, Mapping):
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

            inflight = attrs.get(KEYPOINT_APPLY_INFLIGHT_ATTR)
            recovering = isinstance(inflight, Mapping)
            write_started = bool(recovering)
            if recovering and (
                str(inflight.get("apply_id") or "") != apply_id_value
                or str(inflight.get("checkpoint_snapshot_sha256") or "")
                != actual_digest
            ):
                raise KeypointCheckpointConflict(
                    "A different uncertain keypoint apply owns the canonical target."
                )
            completed_digests: dict[str, str] = {}
            completed_row_results: dict[str, dict[str, object]] = {}
            if recovering:
                completed = inflight.get("completed_row_state_sha256")
                if isinstance(completed, Mapping):
                    completed_digests = {
                        str(key): str(value)
                        for key, value in completed.items()
                        if str(key) and str(value)
                    }
                recorded_results = inflight.get("completed_row_results")
                if isinstance(recorded_results, Mapping):
                    completed_row_results = {
                        str(key): dict(value)
                        for key, value in recorded_results.items()
                        if str(key) and isinstance(value, Mapping)
                    }

            current_checkpoint_id = (
                str(inflight.get("current_checkpoint_id") or "")
                if recovering
                else ""
            )
            current_intended: Mapping[str, object] | None = None
            if current_checkpoint_id:
                candidate = inflight.get("current_expected_row_state")
                candidate_digest = str(
                    inflight.get("current_expected_row_state_sha256") or ""
                )
                if (
                    not isinstance(candidate, Mapping)
                    or candidate_digest != canonical_json_sha256(candidate)
                ):
                    raise KeypointCheckpointConflict(
                        "The uncertain keypoint apply row intent changed."
                    )
                current_intended = candidate

            payloads: list[Mapping[str, object]] = []
            for checkpoint in checkpoints:
                roi_idx = int(checkpoint.get("roi_idx") or 0)
                checkpoint_id = str(checkpoint.get("checkpoint_id") or "")
                metadata = _checkpoint_metadata(checkpoint)
                recovered_completed_intended = (
                    metadata.get("intended_row_state")
                    if checkpoint_id in completed_digests
                    and bool(getattr(session, "recovered_roi_only", False))
                    else None
                )
                payloads.append(
                    validate_keypoint_checkpoint(
                        checkpoint,
                        apply_runtime,
                        allowed_row_state_sha256=(
                            [completed_digests[checkpoint_id]]
                            if checkpoint_id in completed_digests
                            else []
                        ),
                        allowed_intermediate_row_state=(
                            current_intended
                            if checkpoint_id == current_checkpoint_id
                            else (
                                recovered_completed_intended
                                if isinstance(
                                    recovered_completed_intended, Mapping
                                )
                                else None
                            )
                        ),
                    )
                )
                if checkpoint_id in completed_digests and (
                    canonical_json_sha256(_row_state_document(session, roi_idx))
                    != completed_digests[checkpoint_id]
                ):
                    completed_digests.pop(checkpoint_id, None)
                    completed_row_results.pop(checkpoint_id, None)
            intended_states: dict[str, dict[str, object]] = {}
            for checkpoint, payload in zip(checkpoints, payloads):
                checkpoint_id = str(checkpoint.get("checkpoint_id") or "")
                if checkpoint_id in completed_digests:
                    continue
                metadata = _checkpoint_metadata(checkpoint)
                intended = metadata.get("intended_row_state")
                assert isinstance(intended, Mapping)
                if checkpoint_id == current_checkpoint_id and _json_value(
                    current_intended
                ) != _json_value(intended):
                    raise KeypointCheckpointConflict(
                        "The uncertain keypoint row intent no longer matches its checkpoint."
                    )
                intended_states[checkpoint_id] = dict(intended)
            # The global revision is context, not row freshness.  A checkpoint
            # for an unchanged row remains valid after a different row commits.
            edit_revision_before = (
                int(inflight.get("edit_revision_before") or 0)
                if recovering
                else _edit_revision(session)
            )
            inflight_payload = (
                dict(inflight)
                if recovering
                else {
                    "schema": "palette.keypoint_checkpoint_apply_inflight.v1",
                    "apply_id": apply_id_value,
                    "checkpoint_snapshot_sha256": actual_digest,
                    "checkpoint_ids": [
                        str(row.get("checkpoint_id") or "")
                        for row in checkpoints
                    ],
                    "edit_revision_before": edit_revision_before,
                    "started_at_utc": datetime.now(timezone.utc).isoformat(),
                    "completed_row_state_sha256": {},
                    "completed_row_results": {},
                }
            )
            attrs[KEYPOINT_APPLY_INFLIGHT_ATTR] = inflight_payload
            write_started = True
            results: list[Mapping[str, object]] = []
            for checkpoint, payload in zip(checkpoints, payloads):
                checkpoint_id = str(checkpoint.get("checkpoint_id") or "")
                if checkpoint_id in completed_digests:
                    continue
                roi_idx = int(checkpoint.get("roi_idx") or 0)
                intended = intended_states[checkpoint_id]
                inflight_payload["current_checkpoint_id"] = checkpoint_id
                inflight_payload["current_roi_idx"] = roi_idx
                inflight_payload["current_expected_row_state"] = intended
                inflight_payload["current_expected_row_state_sha256"] = (
                    canonical_json_sha256(intended)
                )
                attrs[KEYPOINT_APPLY_INFLIGHT_ATTR] = dict(inflight_payload)
                try:
                    operation_result = _apply_operation(
                        backend_module,
                        session,
                        roi_idx=roi_idx,
                        payload=payload,
                    )
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
                    _converge_row_to_intended_state(
                        session, roi_idx=roi_idx, intended=intended
                    )
                    actual_row_state = _row_state_document(session, roi_idx)
                    actual_row_digest = canonical_json_sha256(actual_row_state)
                    intended_row_digest = canonical_json_sha256(intended)
                    if actual_row_digest != intended_row_digest:
                        raise RuntimeError(
                            "Keypoint apply did not converge to its durable intended row state."
                        )
                    stale_touched = _ensure_downstream_stale_after_row_apply(
                        session,
                        roi_idx=roi_idx,
                        payload=payload,
                        base_row_state=base_row_state,
                        intended_row_state=intended,
                    )
                except Exception:
                    _fail_closed_recovered_training_row(session, roi_idx=roi_idx)
                    raise
                operation_result = {
                    **dict(operation_result),
                    "stale_touched": max(
                        int(operation_result.get("stale_touched") or 0),
                        int(stale_touched),
                    ),
                }
                results.append(operation_result)
                completed_digests[checkpoint_id] = actual_row_digest
                completed_row_results[checkpoint_id] = _applied_row_result(
                    checkpoint, payload, operation_result
                )
                inflight_payload["completed_row_state_sha256"] = dict(
                    completed_digests
                )
                inflight_payload["completed_row_results"] = dict(
                    completed_row_results
                )
                inflight_payload.pop("current_checkpoint_id", None)
                inflight_payload.pop("current_roi_idx", None)
                inflight_payload.pop("current_expected_row_state", None)
                inflight_payload.pop("current_expected_row_state_sha256", None)
                attrs[KEYPOINT_APPLY_INFLIGHT_ATTR] = dict(inflight_payload)
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
                "rows": [int(row.get("roi_idx") or 0) for row in checkpoints],
                "checkpoint_ids": [
                    str(row.get("checkpoint_id") or "") for row in checkpoints
                ],
                "row_results": [
                    completed_row_results[str(row.get("checkpoint_id") or "")]
                    for row in checkpoints
                ],
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

            updated = store.mark_session_checkpoints_applied(
                checkpoint_ids=[
                    str(row.get("checkpoint_id") or "") for row in checkpoints
                ],
                apply_id=apply_id_value,
                edit_revision_before=edit_revision_before,
                edit_revision_after=edit_revision_after,
            )
            if int(updated) != len(checkpoints):
                raise RuntimeError(
                    "The keypoint apply receipt could not finalize every claimed checkpoint."
                )
            attrs.pop(KEYPOINT_APPLY_INFLIGHT_ATTR, None)
            if session is not getattr(runtime, "review_session"):
                runtime.review_session = session
            return {
                **receipt_payload,
                "already_applied": False,
                "canonical_zarr_mutated": any(
                    bool(result.get("changed"))
                    for result in completed_row_results.values()
                ),
                "saved": True,
                "applied": True,
            }
    except Exception:
        if write_started and bool(getattr(session, "recovered_roi_only", False)):
            for checkpoint in checkpoints:
                _fail_closed_recovered_training_row(
                    session, roi_idx=int(checkpoint.get("roi_idx") or 0)
                )
        if not write_started:
            store.release_session_checkpoints_apply(
                task_id=task_id, apply_id=apply_id_value
            )
        raise


__all__ = ["apply_keypoint_checkpoints"]
