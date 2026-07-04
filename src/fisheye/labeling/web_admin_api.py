"""Read-only admin JSON routes served through the Flask strangler."""

from __future__ import annotations

from http import HTTPStatus
from types import SimpleNamespace
from typing import Any, Mapping
from urllib.parse import unquote

from flask import Flask, Response, request

from .admin_registry import (
    _admin_dataset_export_csv,
    _admin_dataset_export_rows,
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
    """Register read-only admin JSON endpoints on ``app``."""

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
