"""HTTP response orchestration for browser keypoint checkpoint mutations."""

from __future__ import annotations

from http import HTTPStatus
from typing import Callable, Mapping

from fisheye.shared.zarr_helpers import archive_metadata_publication_lock

from .web_authorization_metadata import _browser_mutation_response_metadata
from .web_keypoint_checkpoints import (
    KEYPOINT_CHECKPOINT_SAVE_MODE,
    KEYPOINT_IMMUTABLE_DIRECT_SAVE_MODE,
    KeypointCheckpointConflict,
    apply_keypoint_checkpoints,
    current_keypoint_payload,
    keypoint_browser_save_mode,
    stage_keypoint_checkpoint,
)
from .web_responses import _format_error
from .web_runtimes import (
    _advance_keypoint,
    _browser_runtime_target_token,
    _keypoint_runtime_request_lock,
    _keypoint_runtime_state,
    _labeler_safe_error_details,
    _redact_labeler_runtime_payload,
    _require_browser_mutation_target_token,
    _session_scope,
)


RefreshRegistry = Callable[..., bool]


def _refresh_keypoint_registry(
    refresh_registry: RefreshRegistry,
    *,
    store: object,
    runtime: object,
    session: Mapping[str, object],
    user: str,
) -> bool:
    return bool(
        refresh_registry(
            store=store,
            task_id=runtime.task_id,
            recording_id=runtime.recording_id,
            user=user,
            workflow_kind="keypoints",
            scope=_session_scope(session),
            zarr_path=str(runtime.review_session.zarr_path),
            dataset_id=str(session.get("dataset_id") or "") or None,
            zarr_use=str(session.get("zarr_use") or "") or None,
        )
    )


def save_keypoint_request(
    *,
    store: object,
    runtime: object,
    backend_module: object,
    session: Mapping[str, object],
    user: str,
    body: Mapping[str, object],
    operator_validation_mutation_gate: Mapping[str, object],
    refresh_registry: RefreshRegistry,
) -> tuple[dict[str, object], HTTPStatus]:
    """Stage mutable Save or retain the immutable delta compatibility path."""

    try:
        with _keypoint_runtime_request_lock(runtime):
            _require_browser_mutation_target_token(runtime, body)
            save_mode = keypoint_browser_save_mode(runtime.review_session)
            if save_mode == KEYPOINT_CHECKPOINT_SAVE_MODE:
                result = stage_keypoint_checkpoint(
                    store,
                    runtime,
                    user=user,
                    operation="replace_points",
                    points=body.get("points"),
                )
                before = {"target_edit_revision": result.get("target_edit_revision")}
                event_type = "checkpoint_keypoints"
            else:
                before = dict(
                    backend_module.load_roi_payload(
                        runtime.review_session, position=runtime.position
                    )
                )
                with archive_metadata_publication_lock(
                    str(runtime.review_session.zarr_path)
                ):
                    _require_browser_mutation_target_token(runtime, body)
                    direct_result = backend_module.save_roi_correction(
                        runtime.review_session,
                        position=runtime.position,
                        points=body.get("points"),
                    )
                result = {
                    **direct_result,
                    "saved": True,
                    "applied": True,
                    "canonical_zarr_mutated": bool(direct_result.get("changed")),
                    "save_mode": KEYPOINT_IMMUTABLE_DIRECT_SAVE_MODE,
                }
                before = {
                    "roi_idx": before.get("roi_idx"),
                    "frame_idx": before.get("frame_idx"),
                    "points": before.get("points"),
                    "reason": before.get("reason"),
                    "status": before.get("status"),
                }
                event_type = "save_keypoints"
            mutation_event = store.record_event(
                task_id=runtime.task_id,
                recording_id=runtime.recording_id,
                user=user,
                event_type=event_type,
                target={
                    "roi_idx": result.get("roi_idx"),
                    "frame_idx": result.get("frame_idx"),
                    "refined_run": str(runtime.review_session.refined_run),
                    "crop_run": str(runtime.review_session.crop_run),
                },
                before=before,
                after=result,
            )
            runtime.request_generation = int(
                getattr(runtime, "request_generation", 0)
            ) + 1
            if save_mode == KEYPOINT_IMMUTABLE_DIRECT_SAVE_MODE:
                runtime.summary_cache = None
                _refresh_keypoint_registry(
                    refresh_registry,
                    store=store,
                    runtime=runtime,
                    session=session,
                    user=user,
                )
            _advance_keypoint(
                runtime, advance=bool(body.get("advance", runtime.auto_advance_on_save))
            )
            response_state = _keypoint_runtime_state(runtime, backend_module, store=store)
            roi_payload = current_keypoint_payload(
                store,
                runtime,
                backend_module,
                state_payload=response_state,
            )
    except Exception as exc:
        return (
            _format_error(
                "save_error",
                details=_labeler_safe_error_details(exc),
                status=HTTPStatus.BAD_REQUEST,
            ),
            HTTPStatus.BAD_REQUEST,
        )
    return (
        _redact_labeler_runtime_payload(
            {
                "ok": True,
                "result": result,
                "mutation": _browser_mutation_response_metadata(
                    workflow_kind="keypoints",
                    session=session,
                    mutation_event=mutation_event,
                    operator_validation_mutation_gate=operator_validation_mutation_gate,
                ),
                "state": response_state,
                "roi": roi_payload,
            }
        ),
        HTTPStatus.OK,
    )


def apply_keypoint_request(
    *,
    store: object,
    runtime: object,
    backend_module: object,
    session: Mapping[str, object],
    user: str,
    body: Mapping[str, object],
    operator_validation_mutation_gate: Mapping[str, object],
    refresh_registry: RefreshRegistry,
) -> tuple[dict[str, object], HTTPStatus]:
    """Apply one server-selected checkpoint snapshot and report retry ownership."""

    apply_finished = False
    result: dict[str, object] | None = None
    try:
        with _keypoint_runtime_request_lock(runtime):
            _require_browser_mutation_target_token(runtime, body)
            captured_position = int(runtime.position)
            captured_target_token = _browser_runtime_target_token(runtime)
            captured_request_generation = int(
                getattr(runtime, "request_generation", 0)
            )
        result = apply_keypoint_checkpoints(
            store,
            runtime,
            backend_module,
            apply_id=str(body.get("apply_id") or ""),
            checkpoint_snapshot_sha256=str(
                body.get("checkpoint_snapshot_sha256") or ""
            ),
        )
        apply_finished = True
        runtime.summary_cache = None
        mutation_event = store.record_event(
            task_id=runtime.task_id,
            recording_id=runtime.recording_id,
            user=user,
            event_type="apply_keypoint_session_checkpoints",
            target={
                "apply_id": result.get("apply_id"),
                "refined_run": str(runtime.review_session.refined_run),
                "rows": result.get("rows"),
            },
            after=result,
        )
        if not _refresh_keypoint_registry(
            refresh_registry,
            store=store,
            runtime=runtime,
            session=session,
            user=user,
        ):
            raise RuntimeError("Registry refresh failed after canonical keypoint apply.")
        effects_completer = getattr(
            store, "mark_session_checkpoint_apply_effects_complete", None
        )
        if callable(effects_completer):
            effects_completer(
                task_id=runtime.task_id,
                component_name="keypoints",
                apply_id=str(result.get("apply_id") or ""),
            )
        with _keypoint_runtime_request_lock(runtime):
            response_state = _keypoint_runtime_state(
                runtime, backend_module, store=store
            )
            target_unchanged = (
                int(runtime.position) == captured_position
                and _browser_runtime_target_token(runtime) == captured_target_token
                and int(getattr(runtime, "request_generation", 0))
                == captured_request_generation
            )
            response: dict[str, object] = {
                "ok": True,
                "result": result,
                "mutation": _browser_mutation_response_metadata(
                    workflow_kind="keypoints",
                    session=session,
                    mutation_event=mutation_event,
                    operator_validation_mutation_gate=operator_validation_mutation_gate,
                ),
                "state": response_state,
            }
            if target_unchanged:
                response["roi"] = current_keypoint_payload(
                    store,
                    runtime,
                    backend_module,
                    state_payload=response_state,
                )
            else:
                response["current_target_changed"] = True
        return _redact_labeler_runtime_payload(response), HTTPStatus.OK
    except Exception as exc:
        try:
            failure_state = _keypoint_runtime_state(runtime, backend_module, store=store)
        except Exception:
            failure_state = {}
        if apply_finished and result is not None:
            return (
                _format_error(
                    "keypoint_apply_secondary_effect_error",
                    details=_labeler_safe_error_details(exc),
                    status=HTTPStatus.CONFLICT,
                    extra={
                        "state": failure_state,
                        "result": result,
                        "canonical_apply_succeeded": True,
                        "apply_retry_disposition": "retry_same_snapshot",
                        "safe_prewrite_rejection": False,
                        "retain_apply_id": True,
                    },
                ),
                HTTPStatus.CONFLICT,
            )
        requested_apply_id = str(body.get("apply_id") or "").strip()
        retain_apply_id = bool(
            requested_apply_id
            and str(failure_state.get("resumable_apply_id") or "")
            == requested_apply_id
        )
        is_conflict = isinstance(exc, KeypointCheckpointConflict)
        status = (
            HTTPStatus.CONFLICT
            if is_conflict or retain_apply_id
            else HTTPStatus.BAD_REQUEST
        )
        return (
            _format_error(
                "keypoint_apply_conflict" if is_conflict else "keypoint_apply_error",
                details=_labeler_safe_error_details(exc),
                status=status,
                extra={
                    "state": failure_state,
                    "apply_retry_disposition": (
                        "retry_same_snapshot"
                        if retain_apply_id
                        else "fresh_snapshot_required"
                    ),
                    "safe_prewrite_rejection": not retain_apply_id,
                    "retain_apply_id": retain_apply_id,
                },
            ),
            status,
        )


def action_keypoint_request(
    *,
    store: object,
    runtime: object,
    backend_module: object,
    session: Mapping[str, object],
    user: str,
    body: Mapping[str, object],
    operator_validation_mutation_gate: Mapping[str, object],
    refresh_registry: RefreshRegistry,
) -> tuple[dict[str, object], HTTPStatus]:
    """Stage mutable actions or retain immutable delta action compatibility."""

    action = str(body.get("action") or "").strip()
    try:
        with _keypoint_runtime_request_lock(runtime):
            _require_browser_mutation_target_token(runtime, body)
            save_mode = keypoint_browser_save_mode(runtime.review_session)
            if action not in {
                "mark_no_keypoints",
                "mark_detection_issue",
                "clear_failure_label",
            }:
                raise ValueError(f"Unsupported keypoint action: {action}")
            if save_mode == KEYPOINT_CHECKPOINT_SAVE_MODE:
                result = stage_keypoint_checkpoint(
                    store,
                    runtime,
                    user=user,
                    operation=action,
                )
                event_type = f"checkpoint_keypoint_{action}"
            else:
                with archive_metadata_publication_lock(
                    str(runtime.review_session.zarr_path)
                ):
                    _require_browser_mutation_target_token(runtime, body)
                    direct_result = getattr(backend_module, action)(
                        runtime.review_session, position=runtime.position
                    )
                result = {
                    **direct_result,
                    "saved": True,
                    "applied": True,
                    "canonical_zarr_mutated": bool(direct_result.get("changed")),
                    "save_mode": KEYPOINT_IMMUTABLE_DIRECT_SAVE_MODE,
                }
                event_type = f"keypoint_{action}"
            mutation_event = store.record_event(
                task_id=runtime.task_id,
                recording_id=runtime.recording_id,
                user=user,
                event_type=event_type,
                target={
                    "roi_idx": result.get("roi_idx"),
                    "frame_idx": result.get("frame_idx"),
                },
                after=result,
            )
            runtime.request_generation = int(
                getattr(runtime, "request_generation", 0)
            ) + 1
            if save_mode == KEYPOINT_IMMUTABLE_DIRECT_SAVE_MODE:
                runtime.summary_cache = None
                _refresh_keypoint_registry(
                    refresh_registry,
                    store=store,
                    runtime=runtime,
                    session=session,
                    user=user,
                )
            _advance_keypoint(
                runtime, advance=bool(body.get("advance", runtime.auto_advance_on_save))
            )
            response_state = _keypoint_runtime_state(runtime, backend_module, store=store)
            roi_payload = current_keypoint_payload(
                store,
                runtime,
                backend_module,
                state_payload=response_state,
            )
    except Exception as exc:
        return (
            _format_error(
                "keypoint_action_error",
                details=_labeler_safe_error_details(exc),
                status=HTTPStatus.BAD_REQUEST,
            ),
            HTTPStatus.BAD_REQUEST,
        )
    return (
        _redact_labeler_runtime_payload(
            {
                "ok": True,
                "result": result,
                "mutation": _browser_mutation_response_metadata(
                    workflow_kind="keypoints",
                    session=session,
                    mutation_event=mutation_event,
                    operator_validation_mutation_gate=operator_validation_mutation_gate,
                ),
                "state": response_state,
                "roi": roi_payload,
            }
        ),
        HTTPStatus.OK,
    )
