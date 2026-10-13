"""Admin-only, read-only inspection page and API for any labeler's work.

GET /admin/inspect                     page (Preact, static assets)
GET /api/admin/inspect/tasks           mask and keypoint tasks with row counts
GET /api/admin/inspect/task?task_id=   one task's rows: applied / saved / untouched
GET /api/admin/inspect/row?task_id=&roi_idx=
                                       crop image, applied label, saved label

Everything is read through fisheye.labeling.admin_inspect, which opens the
store and archives read-only and never creates sessions. There are no
mutation routes here.
"""

from __future__ import annotations

from http import HTTPStatus
from pathlib import Path
from typing import Any

from flask import Flask, Response

from .admin_inspect import InspectError, inspect_row, inspect_task, inspect_tasks
from .template_assets import render_labeling_template
from .web_admin_api import _admin_user_or_error, _json, _last_arg
from .web_app import claimed_route
from .web_responses import _format_error
from .web_static import static_url

ADMIN_INSPECT_PAGE_PATH = "/admin/inspect"


def _store_path(state: Any) -> Path:
    return Path(state.store.path)


def _inspect_error(exc: Exception) -> Response:
    status = HTTPStatus.NOT_FOUND if "Unknown task_id" in str(exc) else HTTPStatus.BAD_REQUEST
    return _json(_format_error("inspect_unavailable", details=str(exc), status=status), status=status)


def register_admin_inspect_routes(app: Flask, state: Any) -> None:
    @claimed_route(app, ADMIN_INSPECT_PAGE_PATH, methods=["GET"])
    def admin_inspect_page() -> Response:
        _user, error = _admin_user_or_error(state)
        if error is not None:
            return error
        body = render_labeling_template(
            "admin/inspect.html",
            {
                "palette_css": static_url("css/palette.css"),
                "inspect_css": static_url("css/inspect.css"),
                "inspect_js": static_url("js/inspect_page.js"),
                "keypoint_style_js": static_url("js/keypoint_style.js"),
            },
        )
        return Response(body.encode("utf-8"), status=200, content_type="text/html; charset=utf-8")

    @claimed_route(app, "/api/admin/inspect/tasks", methods=["GET"])
    def admin_inspect_tasks() -> Response:
        _user, error = _admin_user_or_error(state)
        if error is not None:
            return error
        return _json(inspect_tasks(_store_path(state)))

    @claimed_route(app, "/api/admin/inspect/task", methods=["GET"])
    def admin_inspect_task() -> Response:
        _user, error = _admin_user_or_error(state)
        if error is not None:
            return error
        try:
            return _json(inspect_task(_store_path(state), _last_arg("task_id")))
        except (InspectError, KeyError, OSError) as exc:
            return _inspect_error(exc)

    @claimed_route(app, "/api/admin/inspect/row", methods=["GET"])
    def admin_inspect_row() -> Response:
        _user, error = _admin_user_or_error(state)
        if error is not None:
            return error
        try:
            roi_idx = int(_last_arg("roi_idx"))
        except ValueError:
            return _inspect_error(InspectError("roi_idx must be an integer."))
        try:
            return _json(inspect_row(_store_path(state), _last_arg("task_id"), roi_idx))
        except (InspectError, KeyError, OSError) as exc:
            return _inspect_error(exc)


__all__ = ["ADMIN_INSPECT_PAGE_PATH", "register_admin_inspect_routes"]
