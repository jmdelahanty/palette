"""Read-only labeler-facing HTML routes served through the Flask strangler."""

from __future__ import annotations

from http import HTTPStatus
from types import SimpleNamespace
from typing import Any, Callable, Mapping
from urllib.parse import parse_qs, urlparse

from flask import Flask, Response, request

from .admin_dashboard import _known_labeler_status, _store_consistency_report
from .web_app import claimed_route
from .web_auth import (
    DASHBOARD_PATH,
    DATASET_QUEUE_PATH,
    PERSONAL_DATASET_QUEUE_PATH,
    _resolve_user,
)
from .web_auth_errors import _authentication_required_error_details
from .web_authorization_metadata import _labeler_read_authorization_denial_metadata
from .web_error_pages import _browser_error_html
from .web_identity import (
    _identity_probe_html,
    _identity_probe_payload,
    _mark_identity_probe_unknown_labeling_user,
)
from .web_personal_renderers import _dashboard_html, _datasets_html
from .web_policy import IDENTITY_PROBE_PATH, LABELING_HOME_PATH, PERSONAL_WORK_PATH
from .web_responses import _format_error

PersonalPageResponder = Callable[..., tuple[bytes, HTTPStatus, str]]


def _personal_page_response_payload(
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

    request_path = str(getattr(request_adapter, "path", path) or path)
    query = parse_qs(urlparse(request_path).query, keep_blank_values=True)
    expected_user = str((query.get("expected_user") or [""])[-1]).strip()
    if path == IDENTITY_PROBE_PATH:
        identity_check_report = _store_consistency_report(state.store)
        identity_assignment_ownership_integrity = (
            identity_check_report.get("assignment_ownership_integrity")
            if isinstance(
                identity_check_report.get("assignment_ownership_integrity"),
                Mapping,
            )
            else {}
        )
        payload = _identity_probe_payload(
            user=user,
            auth_source=auth_source,
            expected_user=expected_user,
            config=state.config,
            store=state.store,
            assignment_ownership_integrity=identity_assignment_ownership_integrity,
        )
        known_user_status = _known_labeler_status(state.store, user)
        payload["known_user_status"] = known_user_status
        if bool(payload.get("ok")) and not bool(known_user_status.get("is_known_labeler")):
            _mark_identity_probe_unknown_labeling_user(payload)
        return (
            _identity_probe_html(payload),
            HTTPStatus.OK if bool(payload.get("ok")) else HTTPStatus.FORBIDDEN,
            "text/html; charset=utf-8",
        )

    if expected_user and str(user) != expected_user:
        payload = _format_error(
            "dashboard_user_mismatch",
            details=(
                f"This dashboard link is for {expected_user}, "
                f"but the browser is authenticated as {user}. "
                "Stop and contact the operator before labeling."
            ),
            status=HTTPStatus.FORBIDDEN,
            extra=_labeler_read_authorization_denial_metadata(
                user=user,
                expected_user=expected_user,
                route_path=path,
                response_kind="html",
            ),
        )
        return (
            _browser_error_html(payload),
            HTTPStatus.FORBIDDEN,
            "text/html; charset=utf-8",
        )

    known_user_status = _known_labeler_status(state.store, user)
    if not bool(known_user_status.get("is_active_labeling_user")):
        error = (
            "unknown_labeling_user"
            if not bool(known_user_status.get("registry_row_present"))
            else "inactive_labeling_user"
        )
        details = (
            "This browser identity is not present in the labeling user registry. "
            "Ask the operator to add or activate this user before labeling."
            if error == "unknown_labeling_user"
            else "This browser identity is present in the labeling user registry but is inactive. "
            "Ask the operator to reactivate this user before labeling."
        )
        payload = _format_error(
            error,
            details=details,
            status=HTTPStatus.FORBIDDEN,
            extra={
                **_labeler_read_authorization_denial_metadata(
                    user=user,
                    expected_user=expected_user or user,
                    route_path=path,
                    response_kind="html",
                ),
                "known_user_status": known_user_status,
            },
        )
        return (
            _browser_error_html(payload),
            HTTPStatus.FORBIDDEN,
            "text/html; charset=utf-8",
        )

    if path in {DASHBOARD_PATH, PERSONAL_WORK_PATH}:
        return _dashboard_html(), HTTPStatus.OK, "text/html; charset=utf-8"
    if path in {"/", "/me", LABELING_HOME_PATH, DATASET_QUEUE_PATH, PERSONAL_DATASET_QUEUE_PATH}:
        return _datasets_html(), HTTPStatus.OK, "text/html; charset=utf-8"

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


__all__ = ["_personal_page_response_payload", "register_personal_page_routes"]
