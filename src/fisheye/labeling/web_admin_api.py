"""Admin JSON routes served through the Flask strangler."""

from __future__ import annotations

from http import HTTPStatus
from types import SimpleNamespace
from typing import Any, Mapping
from urllib.parse import unquote

import numpy as np
from flask import Flask, Response, request

from .assignment_store import ADMIN_REVIEW_STATES
from .admin_registry import (
    _admin_completed_work_payload,
    _admin_dataset_export_csv,
    _admin_dataset_export_rows,
    _admin_task_review_payload,
)
from .admin_dashboard import (
    _admin_datasets_payload,
    _admin_recording_payload,
    _admin_recording_session_summary,
    _admin_summary_payload,
    _admin_user_payload,
    _admin_users_payload,
    _server_safety_payload,
)
from .web_app import claimed_route
from .web_auth import _is_admin_user, _resolve_user
from .web_auth_errors import _authentication_required_error_details
from .web_responses import _format_error, _json_response


def _json(payload: object, *, status: HTTPStatus = HTTPStatus.OK) -> Response:
    data, response_status, content_type = _json_response(payload, status=status)
    return Response(data, status=int(response_status), content_type=content_type)


def _request_adapter() -> SimpleNamespace:
    path = request.full_path
    if path.endswith("?"):
        path = request.path
    return SimpleNamespace(headers=request.headers, path=path)


def _admin_user_or_error(state: Any) -> tuple[str | None, Response | None]:
    user, source = _resolve_user(_request_adapter(), state.config)
    if not user:
        return None, _json(
            _format_error(
                "authentication_required",
                details=_authentication_required_error_details(source, state.config),
                status=HTTPStatus.UNAUTHORIZED,
            ),
            status=HTTPStatus.UNAUTHORIZED,
        )
    if not _is_admin_user(user, state.config):
        return None, _json(
            _format_error("admin_required", status=HTTPStatus.FORBIDDEN),
            status=HTTPStatus.FORBIDDEN,
        )
    return user, None


def _last_arg(name: str) -> str:
    values = request.args.getlist(name)
    return str(values[-1]).strip() if values else ""


def _last_arg_any(*names: str) -> str:
    for name in names:
        value = _last_arg(name)
        if value:
            return value
    return ""


def _truthy_arg(name: str) -> bool:
    return _last_arg(name).lower() in {"1", "true", "yes", "on"}


def _session_closure_support(event: Mapping[str, object] | None) -> dict[str, object] | None:
    if not event:
        return None
    return {
        "event_id": str(event.get("event_id") or ""),
        "event_type": str(event.get("event_type") or ""),
        "event_user": str(event.get("user") or ""),
        "created_at_utc": str(event.get("created_at_utc") or ""),
        "task_id": str(event.get("task_id") or ""),
        "recording_id": str(event.get("recording_id") or ""),
    }


def register_admin_api_routes(app: Flask, state: Any) -> None:
    """Register admin JSON endpoints on ``app``."""

    @claimed_route(app, "/api/health", methods=["GET"])
    def health() -> Response:
        return _json(
            {
                "ok": True,
                "store_path": str(state.config.store_path),
                "preflight": _server_safety_payload(
                    state.config,
                    include_admin_details=False,
                ),
            }
        )

    @claimed_route(app, "/api/admin/summary", methods=["GET"])
    def admin_summary() -> Response:
        _user, error = _admin_user_or_error(state)
        if error is not None:
            return error
        try:
            admin = _admin_summary_payload(state.store, config=state.config)
        except Exception as exc:
            return _json(
                _format_error(
                    "admin_summary_failed",
                    details=str(exc),
                    status=HTTPStatus.INTERNAL_SERVER_ERROR,
                ),
                status=HTTPStatus.INTERNAL_SERVER_ERROR,
            )
        return _json({"ok": True, "admin": admin})

    @claimed_route(app, "/api/admin/datasets", methods=["GET"])
    def admin_datasets() -> Response:
        _user, error = _admin_user_or_error(state)
        if error is not None:
            return error
        try:
            payload = _admin_datasets_payload(
                state.store,
                config=state.config,
                dataset_id=_last_arg("dataset_id") or None,
                recording_id=_last_arg("recording_id") or None,
                assignee_user=_last_arg_any("user", "assignee_user") or None,
                status=_last_arg("status") or None,
                warnings_only=_truthy_arg("warnings"),
            )
        except Exception as exc:
            return _json(
                _format_error(
                    "admin_datasets_failed",
                    details=str(exc),
                    status=HTTPStatus.INTERNAL_SERVER_ERROR,
                ),
                status=HTTPStatus.INTERNAL_SERVER_ERROR,
            )
        return _json(payload)

    @claimed_route(app, "/api/admin/datasets/export", methods=["GET"])
    def admin_datasets_export() -> Response:
        _user, error = _admin_user_or_error(state)
        if error is not None:
            return error
        try:
            payload = _admin_datasets_payload(
                state.store,
                config=state.config,
                dataset_id=_last_arg("dataset_id") or None,
                recording_id=_last_arg("recording_id") or None,
                assignee_user=_last_arg_any("user", "assignee_user") or None,
                status=_last_arg("status") or None,
                warnings_only=_truthy_arg("warnings"),
            )
            rows = _admin_dataset_export_rows(payload)
            export_format = _last_arg("format").lower() or "csv"
        except Exception as exc:
            return _json(
                _format_error(
                    "admin_datasets_export_failed",
                    details=str(exc),
                    status=HTTPStatus.INTERNAL_SERVER_ERROR,
                ),
                status=HTTPStatus.INTERNAL_SERVER_ERROR,
            )
        if export_format == "json":
            response = _json(
                {
                    "ok": True,
                    "schema": "palette.web_labeling_admin_dataset_export.v1",
                    "format": "json",
                    "generated_at_utc": payload.get("generated_at_utc"),
                    "store_path": payload.get("store_path"),
                    "registry": payload.get("registry", {}),
                    "filters": payload.get("filters", {}),
                    "counts": payload.get("counts", {}),
                    "warning_count": payload.get("warning_count", 0),
                    "warnings": payload.get("warnings", []),
                    "row_count": len(rows),
                    "rows": rows,
                },
                status=HTTPStatus.OK,
            )
            response.headers["Content-Disposition"] = (
                'attachment; filename="palette-admin-datasets.json"'
            )
            return response
        if export_format != "csv":
            return _json(
                _format_error(
                    "payload_validation",
                    details="Unsupported export format. Use format=csv or format=json.",
                    status=HTTPStatus.BAD_REQUEST,
                ),
                status=HTTPStatus.BAD_REQUEST,
            )
        return Response(
            _admin_dataset_export_csv(rows).encode("utf-8"),
            status=int(HTTPStatus.OK),
            content_type="text/csv; charset=utf-8",
            headers={
                "Content-Disposition": 'attachment; filename="palette-admin-datasets.csv"'
            },
        )

    @claimed_route(app, "/api/admin/completed-work", methods=["GET"])
    def admin_completed_work() -> Response:
        _user, error = _admin_user_or_error(state)
        if error is not None:
            return error
        try:
            payload = _admin_completed_work_payload(
                state.store,
                assignee_user=_last_arg_any("user", "assignee_user") or None,
                workflow_kind=_last_arg("workflow_kind") or None,
                recording_id=_last_arg("recording_id") or None,
                admin_review_state=_last_arg("admin_review_state") or "pending",
            )
        except Exception as exc:
            return _json(
                _format_error(
                    "admin_completed_work_failed",
                    details=str(exc),
                    status=HTTPStatus.INTERNAL_SERVER_ERROR,
                ),
                status=HTTPStatus.INTERNAL_SERVER_ERROR,
            )
        return _json(payload)

    @claimed_route(app, "/api/admin/admin-reviews", methods=["POST"])
    def admin_review_update() -> Response:
        user, error = _admin_user_or_error(state)
        if error is not None:
            return error
        body = request.get_json(silent=True)
        if not isinstance(body, Mapping):
            return _json(
                _format_error(
                    "payload_validation",
                    details="Expected a JSON object body.",
                    status=HTTPStatus.BAD_REQUEST,
                ),
                status=HTTPStatus.BAD_REQUEST,
            )
        task_id = str(body.get("task_id") or "").strip()
        review_state = str(body.get("state") or body.get("admin_review_state") or "").strip()
        notes = str(body.get("notes") or "").strip() or None
        if not task_id:
            return _json(
                _format_error(
                    "payload_validation",
                    details="Missing task_id.",
                    status=HTTPStatus.BAD_REQUEST,
                ),
                status=HTTPStatus.BAD_REQUEST,
            )
        if not review_state:
            return _json(
                _format_error(
                    "payload_validation",
                    details="Missing admin review state.",
                    status=HTTPStatus.BAD_REQUEST,
                ),
                status=HTTPStatus.BAD_REQUEST,
            )
        try:
            review = state.store.upsert_admin_review(
                task_id=task_id,
                state=review_state,
                reviewer_user=str(user or ""),
                notes=notes,
                metadata={
                    "source_route": "/api/admin/admin-reviews",
                    "source": "admin_task_detail_page",
                },
            )
            payload = _admin_task_review_payload(state.store, task_id)
        except KeyError:
            return _json(
                _format_error("task_not_found", status=HTTPStatus.NOT_FOUND),
                status=HTTPStatus.NOT_FOUND,
            )
        except Exception as exc:
            return _json(
                _format_error(
                    "admin_review_update_failed",
                    details=str(exc),
                    status=HTTPStatus.BAD_REQUEST,
                ),
                status=HTTPStatus.BAD_REQUEST,
            )
        return _json(
            {
                "ok": True,
                "schema": "palette.web_labeling_admin_review_update.v1",
                "admin_review": review,
                "admin_task_review": payload or {},
                "operator_action": (
                    "The admin review state was recorded in the labeling sidecar and an audit event was appended."
                ),
            }
        )

    @claimed_route(app, "/api/admin/keypoint-preview", methods=["GET"])
    def admin_keypoint_preview() -> Response:
        _user, error = _admin_user_or_error(state)
        if error is not None:
            return error
        task_id = _last_arg("task_id")
        roi_idx_text = _last_arg("roi_idx")
        if not task_id:
            return _json(
                _format_error(
                    "payload_validation",
                    details="Missing task_id.",
                    status=HTTPStatus.BAD_REQUEST,
                ),
                status=HTTPStatus.BAD_REQUEST,
            )
        if not roi_idx_text:
            return _json(
                _format_error(
                    "payload_validation",
                    details="Missing roi_idx.",
                    status=HTTPStatus.BAD_REQUEST,
                ),
                status=HTTPStatus.BAD_REQUEST,
            )
        try:
            roi_idx = int(roi_idx_text)
        except ValueError:
            return _json(
                _format_error(
                    "payload_validation",
                    details="roi_idx must be an integer.",
                    status=HTTPStatus.BAD_REQUEST,
                ),
                status=HTTPStatus.BAD_REQUEST,
            )
        task = state.store.get_task(task_id)
        if task is None:
            return _json(
                _format_error("task_not_found", status=HTTPStatus.NOT_FOUND),
                status=HTTPStatus.NOT_FOUND,
            )
        if str(task.get("workflow_kind") or "") != "keypoints":
            return _json(
                _format_error(
                    "unsupported_workflow",
                    details="Admin keypoint preview is only available for keypoint tasks.",
                    status=HTTPStatus.BAD_REQUEST,
                ),
                status=HTTPStatus.BAD_REQUEST,
            )
        scope = task.get("scope") if isinstance(task.get("scope"), Mapping) else {}
        zarr_path = str(scope.get("zarr_path") or "").strip()
        if not zarr_path:
            return _json(
                _format_error(
                    "task_scope_missing_zarr_path",
                    status=HTTPStatus.BAD_REQUEST,
                ),
                status=HTTPStatus.BAD_REQUEST,
            )
        try:
            from fisheye.tune import keypoint_review_backend as backend_module
            from .web_responses import _raw_array_payload

            _root, refined, crop, resolved_refined_run, resolved_crop_run = backend_module.resolve_latest_refined_and_crop(
                zarr_path,
                refined_run=str(scope.get("refined_run") or "").strip() or None,
                crop_run=str(scope.get("crop_run") or "").strip() or None,
                mode="r",
            )
            frame_indices = np.asarray(crop["frame_indices"][:], dtype=np.int64)
            if roi_idx < 0 or roi_idx >= int(frame_indices.shape[0]):
                raise IndexError("roi_idx is out of range.")
            points = np.asarray(refined["keypoints_roi"][roi_idx], dtype=float)
            image = np.asarray(crop["roi_images"][roi_idx], dtype=np.uint8)
            labels = list(refined.attrs.get("keypoint_labels", []))
            if not labels:
                labels = [str(index + 1) for index in range(int(points.shape[0]))]
            reason = ""
            if "reason" in refined:
                reason = str(refined["reason"][roi_idx])
            def _json_scalar(value: object) -> object:
                try:
                    scalar = np.asarray(value).item()
                except Exception:
                    return str(value)
                if isinstance(scalar, np.generic):
                    scalar = scalar.item()
                if isinstance(scalar, (bytes, bytearray, memoryview)):
                    try:
                        return bytes(scalar).decode("utf-8")
                    except Exception:
                        return str(scalar)
                if isinstance(scalar, float):
                    return scalar if np.isfinite(scalar) else None
                if isinstance(scalar, (bool, int, str)) or scalar is None:
                    return scalar
                return str(scalar)
            status: dict[str, object] = {}
            for name in (
                "heading",
                "refined_success",
                "usable_keypoints",
                "edit_applied",
                "confidence_valid",
                "geometry_valid",
                "heading_finite",
                "heading_usable",
            ):
                if name not in refined:
                    continue
                status[name] = _json_scalar(refined[name][roi_idx])
            for name in ("source_refined_row_ids", "source_detect_row_index"):
                if name not in crop:
                    continue
                status[name] = _json_scalar(crop[name][roi_idx])
            def _json_point(point: object) -> list[float | None]:
                pair = np.asarray(point, dtype=float).reshape(-1)
                x = float(pair[0])
                y = float(pair[1])
                return [
                    x if np.isfinite(x) else None,
                    y if np.isfinite(y) else None,
                ]
            roi_payload = {
                "roi_idx": roi_idx,
                "position": roi_idx,
                "total": int(frame_indices.shape[0]),
                "frame_idx": int(frame_indices[roi_idx]),
                "labels": [str(label) for label in labels],
                "points": [_json_point(point) for point in points],
                "reason": reason,
                "status": status,
                "roi_image": _raw_array_payload(image),
            }
            summary = {
                "zarr_path": zarr_path,
                "refined_run": str(resolved_refined_run),
                "crop_run": str(resolved_crop_run),
                "total_rois": int(frame_indices.shape[0]),
            }
        except Exception as exc:
            return _json(
                _format_error(
                    "admin_keypoint_preview_failed",
                    details=str(exc),
                    status=HTTPStatus.BAD_REQUEST,
                ),
                status=HTTPStatus.BAD_REQUEST,
            )
        return _json(
            {
                "ok": True,
                "schema": "palette.web_labeling_admin_keypoint_preview.v1",
                "task_id": task_id,
                "recording_id": str(task.get("recording_id") or ""),
                "dataset_id": str(task.get("dataset_id") or ""),
                "assignee_user": str(task.get("assignee_user") or ""),
                "workflow_kind": str(task.get("workflow_kind") or ""),
                "read_only": True,
                "refined_run": str(summary.get("refined_run") or ""),
                "crop_run": str(summary.get("crop_run") or ""),
                "roi": roi_payload,
                "summary": summary,
                "operator_action": (
                    "This preview is read-only. Use the separate correction route only after explicitly enabling correction mode."
                ),
            }
        )

    @claimed_route(app, "/api/admin/keypoint-corrections", methods=["POST"])
    def admin_keypoint_correction() -> Response:
        user, error = _admin_user_or_error(state)
        if error is not None:
            return error
        body = request.get_json(silent=True)
        if not isinstance(body, Mapping):
            return _json(
                _format_error(
                    "payload_validation",
                    details="Expected a JSON object body.",
                    status=HTTPStatus.BAD_REQUEST,
                ),
                status=HTTPStatus.BAD_REQUEST,
            )
        task_id = str(body.get("task_id") or "").strip()
        roi_idx_raw = body.get("roi_idx")
        points = body.get("points")
        notes = str(body.get("notes") or "").strip() or None
        set_review_state = str(body.get("set_review_state") or "accepted_with_corrections").strip()
        if not task_id:
            return _json(
                _format_error("payload_validation", details="Missing task_id.", status=HTTPStatus.BAD_REQUEST),
                status=HTTPStatus.BAD_REQUEST,
            )
        try:
            roi_idx = int(roi_idx_raw)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return _json(
                _format_error("payload_validation", details="roi_idx must be an integer.", status=HTTPStatus.BAD_REQUEST),
                status=HTTPStatus.BAD_REQUEST,
            )
        if not isinstance(points, list):
            return _json(
                _format_error("payload_validation", details="Missing points list.", status=HTTPStatus.BAD_REQUEST),
                status=HTTPStatus.BAD_REQUEST,
            )
        if set_review_state and set_review_state not in ADMIN_REVIEW_STATES:
            return _json(
                _format_error(
                    "payload_validation",
                    details=(
                        "set_review_state must be empty or one of: "
                        + ", ".join(ADMIN_REVIEW_STATES)
                    ),
                    status=HTTPStatus.BAD_REQUEST,
                ),
                status=HTTPStatus.BAD_REQUEST,
            )
        task = state.store.get_task(task_id)
        if task is None:
            return _json(_format_error("task_not_found", status=HTTPStatus.NOT_FOUND), status=HTTPStatus.NOT_FOUND)
        if str(task.get("workflow_kind") or "") != "keypoints":
            return _json(
                _format_error(
                    "unsupported_workflow",
                    details="Admin keypoint correction is only available for keypoint tasks.",
                    status=HTTPStatus.BAD_REQUEST,
                ),
                status=HTTPStatus.BAD_REQUEST,
            )
        if str(task.get("state") or "") != "complete":
            return _json(
                _format_error(
                    "task_not_complete",
                    details="Admin correction writes require a completed task.",
                    status=HTTPStatus.BAD_REQUEST,
                ),
                status=HTTPStatus.BAD_REQUEST,
            )
        scope = task.get("scope") if isinstance(task.get("scope"), Mapping) else {}
        zarr_path = str(scope.get("zarr_path") or "").strip()
        if not zarr_path:
            return _json(
                _format_error("task_scope_missing_zarr_path", status=HTTPStatus.BAD_REQUEST),
                status=HTTPStatus.BAD_REQUEST,
            )
        try:
            from fisheye.tune import keypoint_review_backend as backend_module

            review_session = backend_module.resolve_review_session(
                zarr_path,
                refined_run=str(scope.get("refined_run") or "").strip() or None,
                crop_run=str(scope.get("crop_run") or "").strip() or None,
                include_all=True,
                target_roi_indices=[roi_idx],
            )
            before = dict(backend_module.load_roi_payload(review_session, position=0))
            before_points = before.get("points") if isinstance(before.get("points"), list) else []
            result = backend_module.save_roi_correction(
                review_session,
                position=0,
                points=points,  # type: ignore[arg-type]
            )
            readback = result.get("readback") if isinstance(result.get("readback"), Mapping) else {}
            after_points = points
            labels = before.get("labels") if isinstance(before.get("labels"), list) else []
            deltas: list[dict[str, object]] = []
            for index, point_after in enumerate(after_points):
                point_before = before_points[index] if index < len(before_points) else None
                if (
                    not isinstance(point_before, list)
                    or not isinstance(point_after, list)
                    or len(point_before) < 2
                    or len(point_after) < 2
                    or point_before[0] is None
                    or point_before[1] is None
                    or point_after[0] is None
                    or point_after[1] is None
                ):
                    continue
                bx = float(point_before[0])
                by = float(point_before[1])
                ax = float(point_after[0])
                ay = float(point_after[1])
                dx = ax - bx
                dy = ay - by
                deltas.append(
                    {
                        "index": index,
                        "label": str(labels[index] if index < len(labels) else index + 1),
                        "before": [bx, by],
                        "after": [ax, ay],
                        "dx": dx,
                        "dy": dy,
                        "distance_px": float(np.hypot(dx, dy)),
                    }
                )
            correction_event = state.store.record_event(
                task_id=task_id,
                recording_id=str(task.get("recording_id") or ""),
                user=str(user or ""),
                event_type="admin_save_keypoint_correction",
                target={
                    "roi_idx": result.get("roi_idx"),
                    "frame_idx": result.get("frame_idx"),
                    "refined_run": str(review_session.refined_run),
                    "crop_run": str(review_session.crop_run),
                    "admin_review_state": set_review_state,
                },
                before={
                    "labeler_user": str(task.get("assignee_user") or ""),
                    "roi_idx": before.get("roi_idx"),
                    "frame_idx": before.get("frame_idx"),
                    "points": before_points,
                    "reason": before.get("reason"),
                    "status": before.get("status"),
                },
                after={
                    "admin_reviewer_user": str(user or ""),
                    "points": after_points,
                    "deltas": deltas,
                    "changed": result.get("changed"),
                    "reason_updated": result.get("reason_updated"),
                    "readback": readback,
                    "notes": notes,
                },
            )
            admin_review = None
            if set_review_state:
                correction_event_count = state.store.count_admin_correction_events(task_id)
                admin_review = state.store.upsert_admin_review(
                    task_id=task_id,
                    state=set_review_state,
                    reviewer_user=str(user or ""),
                    notes=notes,
                    correction_event_count=correction_event_count,
                    metadata={
                        "source_route": "/api/admin/keypoint-corrections",
                        "latest_correction_event_id": str(correction_event.get("event_id") or ""),
                    },
                )
        except Exception as exc:
            return _json(
                _format_error(
                    "admin_keypoint_correction_failed",
                    details=str(exc),
                    status=HTTPStatus.BAD_REQUEST,
                ),
                status=HTTPStatus.BAD_REQUEST,
            )
        return _json(
            {
                "ok": True,
                "schema": "palette.web_labeling_admin_keypoint_correction.v1",
                "task_id": task_id,
                "roi_idx": result.get("roi_idx"),
                "frame_idx": result.get("frame_idx"),
                "result": result,
                "delta_count": len(deltas),
                "deltas": deltas,
                "correction_event": correction_event,
                "admin_review": admin_review or {},
                "operator_action": (
                    "Admin keypoint correction was applied to the refined keypoint run and recorded as a distinct audit event."
                ),
            }
        )

    @claimed_route(app, "/api/admin/users", methods=["GET"])
    def admin_users() -> Response:
        _user, error = _admin_user_or_error(state)
        if error is not None:
            return error
        return _json(_admin_users_payload(state.store, config=state.config))

    @claimed_route(app, "/api/admin/preflight", methods=["GET"])
    def admin_preflight() -> Response:
        _user, error = _admin_user_or_error(state)
        if error is not None:
            return error
        return _json(
            {
                "ok": True,
                "preflight": _server_safety_payload(
                    state.config,
                    include_admin_details=True,
                ),
            }
        )

    @claimed_route(
        app,
        "/api/admin/events/<path:event_id>",
        claim="prefix",
        claim_prefix_value="/api/admin/events",
        methods=["GET"],
    )
    def admin_event(event_id: str) -> Response:
        _user, error = _admin_user_or_error(state)
        if error is not None:
            return error
        resolved_event_id = unquote(str(event_id or "").strip("/"))
        if not resolved_event_id or "/" in resolved_event_id:
            return _json(
                _format_error("event_not_found", status=HTTPStatus.NOT_FOUND),
                status=HTTPStatus.NOT_FOUND,
            )
        event = state.store.get_event(resolved_event_id)
        if event is None:
            return _json(
                _format_error("event_not_found", status=HTTPStatus.NOT_FOUND),
                status=HTTPStatus.NOT_FOUND,
            )
        return _json(
            {
                "ok": True,
                "event_id": resolved_event_id,
                "event": event,
                "retry_promotion_url": (
                    f"/api/admin/events/{resolved_event_id}/retry-promotion"
                    if str(event.get("event_type") or "") == "promotion_failed"
                    else ""
                ),
                "operator_action": (
                    "Use this audit event to reconcile a labeler-provided save reference with the assigned task, recording, user, target, and mutation outcome."
                ),
            }
        )

    @claimed_route(
        app,
        "/api/admin/sessions/<path:session_id>/closure",
        claim="prefix",
        claim_prefix_value="/api/admin/sessions",
        methods=["GET"],
    )
    def admin_session_closure(session_id: str) -> Response:
        _user, error = _admin_user_or_error(state)
        if error is not None:
            return error
        resolved_session_id = unquote(str(session_id or "").strip("/"))
        if not resolved_session_id:
            return _json(
                _format_error("missing_session_id", status=HTTPStatus.NOT_FOUND),
                status=HTTPStatus.NOT_FOUND,
            )
        session = state.store.get_session(resolved_session_id)
        if session is None:
            return _json(
                _format_error("session_not_found", status=HTTPStatus.NOT_FOUND),
                status=HTTPStatus.NOT_FOUND,
            )
        closure_event = state.store.get_session_closure_event(resolved_session_id)
        closure_support = _session_closure_support(closure_event)
        return _json(
            {
                "ok": True,
                "session_id": resolved_session_id,
                "session": _admin_recording_session_summary(session),
                "has_closure_event": closure_support is not None,
                "session_closure_event": closure_support,
                "operator_action": (
                    "Use this closure event to explain stale-tab, reassignment, completion, or cleanup failures to the labeler."
                    if closure_support is not None
                    else "No closure event is recorded for this session; inspect the session and task state directly."
                ),
            }
        )

    @claimed_route(
        app,
        "/api/admin/tasks/<path:task_id>",
        claim="prefix",
        claim_prefix_value="/api/admin/tasks",
        methods=["GET"],
    )
    def admin_task_review(task_id: str) -> Response:
        _user, error = _admin_user_or_error(state)
        if error is not None:
            return error
        resolved_task_id = unquote(str(task_id or "").strip("/"))
        if not resolved_task_id:
            return _json(
                _format_error("missing_task_id", status=HTTPStatus.NOT_FOUND),
                status=HTTPStatus.NOT_FOUND,
            )
        payload = _admin_task_review_payload(state.store, resolved_task_id)
        if payload is None:
            return _json(
                _format_error("task_not_found", status=HTTPStatus.NOT_FOUND),
                status=HTTPStatus.NOT_FOUND,
            )
        return _json(payload)

    @claimed_route(
        app,
        "/api/admin/recordings/<path:recording_id>",
        claim="prefix",
        claim_prefix_value="/api/admin/recordings",
        methods=["GET"],
    )
    def admin_recording(recording_id: str) -> Response:
        _user, error = _admin_user_or_error(state)
        if error is not None:
            return error
        resolved_recording_id = unquote(str(recording_id or "").strip("/"))
        if not resolved_recording_id:
            return _json(
                _format_error("missing_recording_id", status=HTTPStatus.NOT_FOUND),
                status=HTTPStatus.NOT_FOUND,
            )
        return _json(
            {
                "ok": True,
                "admin_recording": _admin_recording_payload(
                    state.store,
                    recording_id=resolved_recording_id,
                ),
            }
        )

    @claimed_route(
        app,
        "/api/admin/users/<path:target_user>",
        claim="prefix",
        claim_prefix_value="/api/admin/users",
        methods=["GET"],
    )
    def admin_user(target_user: str) -> Response:
        _user, error = _admin_user_or_error(state)
        if error is not None:
            return error
        resolved_target_user = unquote(str(target_user or "").strip("/"))
        if not resolved_target_user:
            return _json(
                _format_error("missing_user", status=HTTPStatus.NOT_FOUND),
                status=HTTPStatus.NOT_FOUND,
            )
        return _json(
            {
                "ok": True,
                "admin_user": _admin_user_payload(
                    state.store,
                    user=resolved_target_user,
                ),
            }
        )


__all__ = ["register_admin_api_routes"]
