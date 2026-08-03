"""Read-only admin HTML routes served through the Flask strangler."""

from __future__ import annotations

from http import HTTPStatus
from types import SimpleNamespace
from typing import Any, Callable, Mapping
from urllib.parse import parse_qs, unquote

from flask import Flask, Response, request

from .admin_registry import _admin_completed_work_payload, _admin_task_review_payload
from .admin_dashboard import (
    _admin_recording_payload,
    _admin_user_payload,
    _admin_users_payload,
)
from .web_admin_renderers import (
    _admin_completed_work_html,
    _admin_datasets_html,
    _admin_html,
    _admin_recording_html,
    _admin_task_html,
    _admin_user_html,
    _admin_users_html,
)
from .web_app import claimed_route
from .web_auth import _is_admin_user, _resolve_user
from .web_auth_errors import _authentication_required_error_details
from .web_error_pages import _browser_error_html
from .web_responses import _format_error

AdminPageResponder = Callable[..., tuple[bytes, HTTPStatus, str]]


def _last_query_arg(request_adapter: object, *names: str, default: str = "") -> str:
    path = str(getattr(request_adapter, "path", "") or "")
    query = path.partition("?")[2]
    if not query:
        return default
    values_by_name = parse_qs(query, keep_blank_values=True)
    for name in names:
        values = values_by_name.get(name)
        if values:
            return str(values[-1]).strip()
    return default


def _admin_page_response_payload(
    state: Any,
    *,
    path: str,
    request_adapter: object,
) -> tuple[bytes, HTTPStatus, str]:
    user, auth_source = _resolve_user(request_adapter, state.config)  # type: ignore[arg-type]
    if user is None:
        payload = _format_error(
            "authentication_required",
            details=_authentication_required_error_details(auth_source, state.config),
            status=HTTPStatus.UNAUTHORIZED,
        )
        return (
            _browser_error_html(payload),
            HTTPStatus.UNAUTHORIZED,
            "text/html; charset=utf-8",
        )
    if not _is_admin_user(user, state.config):
        payload = _format_error(
            "admin_required",
            status=HTTPStatus.FORBIDDEN,
        )
        return (
            _browser_error_html(payload),
            HTTPStatus.FORBIDDEN,
            "text/html; charset=utf-8",
        )

    if path == "/admin":
        return _admin_html(), HTTPStatus.OK, "text/html; charset=utf-8"
    if path == "/admin/datasets":
        return _admin_datasets_html(), HTTPStatus.OK, "text/html; charset=utf-8"
    if path == "/admin/completed-work":
        return (
            _admin_completed_work_html(
                _admin_completed_work_payload(
                    state.store,
                    assignee_user=_last_query_arg(request_adapter, "user", "assignee_user") or None,
                    workflow_kind=_last_query_arg(request_adapter, "workflow_kind") or None,
                    recording_id=_last_query_arg(request_adapter, "recording_id") or None,
                    admin_review_state=_last_query_arg(
                        request_adapter,
                        "admin_review_state",
                        default="pending",
                    )
                    or "pending",
                )
            ),
            HTTPStatus.OK,
            "text/html; charset=utf-8",
        )
    if path == "/admin/users":
        return (
            _admin_users_html(_admin_users_payload(state.store, config=state.config)),
            HTTPStatus.OK,
            "text/html; charset=utf-8",
        )
    if path.startswith("/admin/recordings/"):
        recording_id = unquote(path[len("/admin/recordings/") :].strip("/"))
        if not recording_id:
            payload = _format_error(
                "missing_recording_id",
                status=HTTPStatus.NOT_FOUND,
            )
            return (
                _browser_error_html(payload),
                HTTPStatus.NOT_FOUND,
                "text/html; charset=utf-8",
            )
        return (
            _admin_recording_html(
                _admin_recording_payload(state.store, recording_id=recording_id)
            ),
            HTTPStatus.OK,
            "text/html; charset=utf-8",
        )
    if path.startswith("/admin/users/"):
        target_user = unquote(path[len("/admin/users/") :].strip("/"))
        if not target_user:
            payload = _format_error("missing_user", status=HTTPStatus.NOT_FOUND)
            return (
                _browser_error_html(payload),
                HTTPStatus.NOT_FOUND,
                "text/html; charset=utf-8",
            )
        payload = _admin_user_payload(state.store, user=target_user)
        return (
            _admin_user_html(
                user=target_user,
                work=payload["work"] if isinstance(payload.get("work"), Mapping) else {},
                dashboard_row=payload["dashboard_user"] if isinstance(payload.get("dashboard_user"), Mapping) else None,
            ),
            HTTPStatus.OK,
            "text/html; charset=utf-8",
        )
    if path.startswith("/admin/tasks/"):
        task_id = path[len("/admin/tasks/") :].strip("/")
        if not task_id:
            payload = _format_error(
                "missing_task_id",
                status=HTTPStatus.NOT_FOUND,
            )
            return (
                _browser_error_html(payload),
                HTTPStatus.NOT_FOUND,
                "text/html; charset=utf-8",
            )
        payload = _admin_task_review_payload(state.store, task_id)
        if payload is None:
            payload = _format_error("task_not_found", status=HTTPStatus.NOT_FOUND)
            return (
                _browser_error_html(payload),
                HTTPStatus.NOT_FOUND,
                "text/html; charset=utf-8",
            )
        return_to = _last_query_arg(request_adapter, "return_to")
        if return_to.startswith("/admin/completed-work"):
            payload = {**payload, "admin_queue_return_url": return_to}
        return (
            _admin_task_html(payload),
            HTTPStatus.OK,
            "text/html; charset=utf-8",
        )

    payload = _format_error("not_found", status=HTTPStatus.NOT_FOUND)
    return (
        _browser_error_html(payload),
        HTTPStatus.NOT_FOUND,
        "text/html; charset=utf-8",
    )

def _request_adapter() -> SimpleNamespace:
    path = request.full_path
    if path.endswith("?"):
        path = request.path
    return SimpleNamespace(headers=request.headers, path=path)


def _respond(
    state: Any,
    response_builder: AdminPageResponder,
    *,
    path: str,
) -> Response:
    body, status, content_type = response_builder(
        state,
        path=path,
        request_adapter=_request_adapter(),
    )
    return Response(body, status=int(status), content_type=content_type)


def register_admin_page_routes(
    app: Flask,
    state: Any,
    response_builder: AdminPageResponder,
) -> None:
    """Register admin HTML pages on ``app``."""

    @claimed_route(app, "/admin", methods=["GET"])
    def admin_page_home() -> Response:
        return _respond(state, response_builder, path="/admin")

    @claimed_route(app, "/admin/datasets", methods=["GET"])
    def admin_page_datasets() -> Response:
        return _respond(state, response_builder, path="/admin/datasets")

    @claimed_route(app, "/admin/completed-work", methods=["GET"])
    def admin_page_completed_work() -> Response:
        return _respond(state, response_builder, path="/admin/completed-work")

    @claimed_route(app, "/admin/users", methods=["GET"])
    def admin_page_users() -> Response:
        return _respond(state, response_builder, path="/admin/users")

    @claimed_route(
        app,
        "/admin/recordings/<path:recording_id>",
        claim="prefix",
        claim_prefix_value="/admin/recordings",
        methods=["GET"],
    )
    def admin_page_recording(recording_id: str) -> Response:
        return _respond(
            state,
            response_builder,
            path=f"/admin/recordings/{recording_id}",
        )

    @claimed_route(
        app,
        "/admin/users/<path:user>",
        claim="prefix",
        claim_prefix_value="/admin/users",
        methods=["GET"],
    )
    def admin_page_user(user: str) -> Response:
        return _respond(state, response_builder, path=f"/admin/users/{user}")

    @claimed_route(
        app,
        "/admin/tasks/<path:task_id>",
        claim="prefix",
        claim_prefix_value="/admin/tasks",
        methods=["GET"],
    )
    def admin_page_task(task_id: str) -> Response:
        return _respond(state, response_builder, path=f"/admin/tasks/{task_id}")


__all__ = ["_admin_page_response_payload", "register_admin_page_routes"]
