"""Read-only labeler-facing HTML routes served through the Flask strangler."""

from __future__ import annotations

from http import HTTPStatus
from types import SimpleNamespace
from typing import Any, Callable

from flask import Flask, Response, request

from .web_app import claimed_route

PersonalPageResponder = Callable[..., tuple[bytes, HTTPStatus, str]]


def _request_adapter() -> SimpleNamespace:
    path = request.full_path
    if path.endswith("?"):
        path = request.path
    return SimpleNamespace(headers=request.headers, path=path)


def _respond(
    state: Any,
    response_builder: PersonalPageResponder,
    *,
    path: str,
) -> Response:
    body, status, content_type = response_builder(
        state,
        path=path,
        request_adapter=_request_adapter(),
    )
    return Response(body, status=int(status), content_type=content_type)


def register_personal_page_routes(
    app: Flask,
    state: Any,
    response_builder: PersonalPageResponder,
) -> None:
    """Register read-only personal dashboard and dataset queue pages on ``app``."""

    @claimed_route(app, "/", methods=["GET"])
    def labeler_landing_home() -> Response:
        return _respond(state, response_builder, path="/")

    @claimed_route(app, "/me", methods=["GET"])
    def labeler_landing_me() -> Response:
        return _respond(state, response_builder, path="/me")

    @claimed_route(app, "/labeling", methods=["GET"])
    def labeler_landing_labeling() -> Response:
        return _respond(state, response_builder, path="/labeling")

    @claimed_route(app, "/identity", methods=["GET"])
    def labeler_identity_probe() -> Response:
        return _respond(state, response_builder, path="/identity")

    @claimed_route(app, "/work", methods=["GET"])
    def personal_work_dashboard() -> Response:
        return _respond(state, response_builder, path="/work")

    @claimed_route(app, "/my-work", methods=["GET"])
    def personal_work_dashboard_alias() -> Response:
        return _respond(state, response_builder, path="/my-work")

    @claimed_route(app, "/datasets", methods=["GET"])
    def personal_dataset_queue() -> Response:
        return _respond(state, response_builder, path="/datasets")

    @claimed_route(app, "/my-datasets", methods=["GET"])
    def personal_dataset_queue_alias() -> Response:
        return _respond(state, response_builder, path="/my-datasets")


__all__ = ["register_personal_page_routes"]
