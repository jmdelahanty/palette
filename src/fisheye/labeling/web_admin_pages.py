"""Read-only admin HTML routes served through the Flask strangler."""

from __future__ import annotations

from http import HTTPStatus
from types import SimpleNamespace
from typing import Any, Callable

from flask import Flask, Response, request

from .web_app import claimed_route

AdminPageResponder = Callable[..., tuple[bytes, HTTPStatus, str]]


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
    """Register read-only admin HTML pages on ``app``."""

    @claimed_route(app, "/admin", methods=["GET"])
    def admin_page_home() -> Response:
        return _respond(state, response_builder, path="/admin")

    @claimed_route(app, "/admin/datasets", methods=["GET"])
    def admin_page_datasets() -> Response:
        return _respond(state, response_builder, path="/admin/datasets")

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


__all__ = ["register_admin_page_routes"]
