"""Read-only peer review page and API, plus advisory frame flags.

GET  /peer                                  page (Preact, static assets)
GET  /api/peer/tasks                        mask and keypoint tasks: owner, applied rows, open flags
GET  /api/peer/task?task_id=                rows applied at least once, with open flags
GET  /api/peer/row?task_id=&roi_idx=        image, applied label, and flags for one applied row
POST /api/peer/flags                        add a flag: task_id, roi_idx, note
POST /api/peer/flags/<flag_id>/resolve      resolve: task owner, flag author, or admin

Any active labeling user (or an admin) may use these routes. Reads go through
fisheye.labeling.peer_review, which never returns a saved, unapplied payload
and never opens a Zarr for writing. The two POST routes write only the
labeling_frame_flags table.
"""

from __future__ import annotations

from http import HTTPStatus
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

from flask import Flask, Response, request

from .admin_dashboard import _known_labeler_status
from .admin_inspect import InspectError
from .peer_review import (
    FlagError,
    FlagForbidden,
    add_frame_flag,
    peer_row,
    peer_task,
    peer_tasks,
    resolve_frame_flag,
)
from .template_assets import render_labeling_template
from .web_admin_api import _json, _last_arg, _request_adapter
from .web_app import claimed_route
from .web_auth import _is_admin_user, _resolve_user
from .web_auth_errors import _authentication_required_error_details
from .web_responses import _format_error, request_has_same_origin
from .web_static import static_url

PEER_REVIEW_PAGE_PATH = "/peer"


def _error(code: str, status: HTTPStatus, details: str | None = None) -> Response:
    return _json(_format_error(code, details=details, status=status), status=status)


def _peer_user_or_error(state: Any) -> tuple[str | None, Response | None]:
    user, source = _resolve_user(_request_adapter(), state.config)
    if not user:
        return None, _error(
            "authentication_required",
            HTTPStatus.UNAUTHORIZED,
            _authentication_required_error_details(source, state.config),
        )
    if _is_admin_user(user, state.config):
        return user, None
    known = _known_labeler_status(state.store, user)
    if not bool(known.get("is_active_labeling_user")):
        code = "unknown_labeling_user" if not bool(known.get("registry_row_present")) else "inactive_labeling_user"
        return None, _error(code, HTTPStatus.FORBIDDEN, "Ask the operator to add or activate this user.")
    return user, None


def _same_origin_or_error(state: Any) -> Response | None:
    if state.config.csrf_same_origin and not request_has_same_origin(SimpleNamespace(headers=request.headers)):
        return _error("same_origin_required", HTTPStatus.FORBIDDEN, "Cross-origin mutating requests are rejected.")
    return None


def _store_path(state: Any) -> Path:
    return Path(state.store.path)


def _read_error(exc: Exception) -> Response:
    status = HTTPStatus.NOT_FOUND if "Unknown task_id" in str(exc) else HTTPStatus.BAD_REQUEST
    return _error("peer_review_unavailable", status, str(exc))


def register_peer_review_routes(app: Flask, state: Any) -> None:
    @claimed_route(app, PEER_REVIEW_PAGE_PATH, methods=["GET"])
    def peer_review_page() -> Response:
        _user, error = _peer_user_or_error(state)
        if error is not None:
            return error
        body = render_labeling_template(
            "personal/peer.html",
            {
                "palette_css": static_url("css/palette.css"),
                "inspect_css": static_url("css/inspect.css"),
                "peer_js": static_url("js/peer_page.js"),
                "keypoint_style_js": static_url("js/keypoint_style.js"),
            },
        )
        return Response(body.encode("utf-8"), status=200, content_type="text/html; charset=utf-8")

    @claimed_route(app, "/api/peer/tasks", methods=["GET"])
    def peer_review_tasks() -> Response:
        user, error = _peer_user_or_error(state)
        if error is not None:
            return error
        return _json({**peer_tasks(_store_path(state)), "user": user})

    @claimed_route(app, "/api/peer/task", methods=["GET"])
    def peer_review_task() -> Response:
        _user, error = _peer_user_or_error(state)
        if error is not None:
            return error
        try:
            return _json(peer_task(_store_path(state), _last_arg("task_id")))
        except (InspectError, KeyError, OSError) as exc:
            return _read_error(exc)

    @claimed_route(app, "/api/peer/row", methods=["GET"])
    def peer_review_row() -> Response:
        user, error = _peer_user_or_error(state)
        if error is not None:
            return error
        try:
            roi_idx = int(_last_arg("roi_idx"))
        except ValueError:
            return _read_error(InspectError("roi_idx must be an integer."))
        try:
            row = peer_row(_store_path(state), _last_arg("task_id"), roi_idx)
        except (InspectError, KeyError, OSError) as exc:
            return _read_error(exc)
        admin = _is_admin_user(user, state.config)
        for flag in row["flags"]:
            flag["can_resolve"] = flag["state"] == "open" and (
                admin or user in {row.get("owner"), flag["flagged_by"]}
            )
        return _json(row)

    @claimed_route(app, "/api/peer/flags", methods=["POST"])
    def peer_review_add_flag() -> Response:
        user, error = _peer_user_or_error(state)
        if error is not None:
            return error
        error = _same_origin_or_error(state)
        if error is not None:
            return error
        body = request.get_json(silent=True)
        if not isinstance(body, Mapping):
            return _error("payload_validation", HTTPStatus.BAD_REQUEST, "Expected a JSON object body.")
        try:
            roi_idx = int(body.get("roi_idx"))
        except (TypeError, ValueError):
            return _error("payload_validation", HTTPStatus.BAD_REQUEST, "roi_idx must be an integer.")
        try:
            flag = add_frame_flag(
                state.store,
                task_id=str(body.get("task_id") or "").strip(),
                roi_idx=roi_idx,
                flagged_by=str(user),
                note=str(body.get("note") or ""),
            )
        except KeyError as exc:
            return _error("task_not_found", HTTPStatus.NOT_FOUND, str(exc))
        except (FlagError, InspectError) as exc:
            return _error("flag_rejected", HTTPStatus.BAD_REQUEST, str(exc))
        return _json({"ok": True, "flag": flag})

    @claimed_route(
        app,
        "/api/peer/flags/<flag_id>/resolve",
        methods=["POST"],
        claim="prefix",
        claim_prefix_value="/api/peer/flags/",
    )
    def peer_review_resolve_flag(flag_id: str) -> Response:
        user, error = _peer_user_or_error(state)
        if error is not None:
            return error
        error = _same_origin_or_error(state)
        if error is not None:
            return error
        try:
            flag = resolve_frame_flag(
                state.store, flag_id=flag_id, user=str(user), is_admin=_is_admin_user(user, state.config)
            )
        except KeyError as exc:
            return _error("flag_not_found", HTTPStatus.NOT_FOUND, str(exc))
        except FlagForbidden as exc:
            return _error("flag_resolve_forbidden", HTTPStatus.FORBIDDEN, str(exc))
        return _json({"ok": True, "flag": flag})


__all__ = ["PEER_REVIEW_PAGE_PATH", "register_peer_review_routes"]
