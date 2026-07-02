"""Read-only labeler-facing JSON routes served through the Flask strangler."""

from __future__ import annotations

from http import HTTPStatus
from types import SimpleNamespace
from typing import Any, Callable

from flask import Flask, Response, request

from .web_app import claimed_route
from .web_responses import _json_response

PersonalApiResponder = Callable[..., tuple[dict[str, object], HTTPStatus]]


def _json(payload: object, *, status: HTTPStatus = HTTPStatus.OK) -> Response:
    data, response_status, content_type = _json_response(payload, status=status)
    return Response(data, status=int(response_status), content_type=content_type)


def _request_adapter() -> SimpleNamespace:
    path = request.full_path
    if path.endswith("?"):
        path = request.path
    return SimpleNamespace(headers=request.headers, path=path)


def _respond(
    state: Any,
    response_builder: PersonalApiResponder,
    *,
    path: str,
) -> Response:
    payload, status = response_builder(
        state,
        path=path,
        request_adapter=_request_adapter(),
    )
    return _json(payload, status=status)


def register_personal_api_routes(
    app: Flask,
    state: Any,
    response_builder: PersonalApiResponder,
) -> None:
    """Register read-only personal work JSON endpoints on ``app``."""

    @claimed_route(app, "/api/me/identity", methods=["GET"])
    def personal_identity() -> Response:
        return _respond(state, response_builder, path="/api/me/identity")

    @claimed_route(app, "/api/me/tasks", methods=["GET"])
    def personal_tasks() -> Response:
        return _respond(state, response_builder, path="/api/me/tasks")

    @claimed_route(app, "/api/me/datasets", methods=["GET"])
    def personal_datasets() -> Response:
        return _respond(state, response_builder, path="/api/me/datasets")


__all__ = ["register_personal_api_routes"]
