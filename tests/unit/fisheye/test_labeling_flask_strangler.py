from __future__ import annotations

import io
import json
from email.message import Message
from http import HTTPStatus
from types import SimpleNamespace

import pytest

pytest.importorskip("flask")

from flask import Response, request

from fisheye.labeling.web_app import claimed_route, create_labeling_app
from fisheye.labeling.web_personal_api import register_personal_api_routes
from fisheye.labeling.web_policy import BROWSER_RESPONSE_SECURITY_HEADERS
from fisheye.labeling.web_wsgi_adapter import handle_with_flask_if_claimed


class FakeHandler:
    def __init__(
        self,
        method: str,
        path: str,
        *,
        headers: dict[str, str] | None = None,
        body: bytes = b"",
    ) -> None:
        self.command = method
        self.path = path
        self.request_version = "HTTP/1.1"
        self.client_address = ("127.0.0.1", 43210)
        self.server = SimpleNamespace(server_name="legacy.test", server_port=8765)
        self.headers = Message()
        for name, value in (headers or {}).items():
            self.headers[name] = value
        if body and "Content-Length" not in self.headers:
            self.headers["Content-Length"] = str(len(body))
        self.rfile = io.BytesIO(body)
        self.wfile = io.BytesIO()
        self.response_code: int | None = None
        self.response_message: str | None = None
        self.sent_headers: list[tuple[str, str]] = []
        self.ended_headers = False

    def send_response(self, code: int, message: str | None = None) -> None:
        self.response_code = code
        self.response_message = message

    def send_header(self, name: str, value: str) -> None:
        self.sent_headers.append((name, value))

    def end_headers(self) -> None:
        self.ended_headers = True

    def header_values(self, name: str) -> list[str]:
        lowered = name.lower()
        return [value for header_name, value in self.sent_headers if header_name.lower() == lowered]


def test_wsgi_adapter_preserves_request_and_response_headers_for_claimed_prefix() -> None:
    app = create_labeling_app(import_name=__name__)

    @claimed_route(
        app,
        "/claimed/prefix/<path:tail>",
        claim="prefix",
        claim_prefix_value="/claimed/prefix",
        methods=["GET"],
    )
    def prefix_headers(tail: str) -> Response:
        response = Response(f"tail={tail}", status=202)
        response.headers["X-Echo-Token"] = request.headers["X-Token"]
        response.headers["X-Echo-Query"] = request.args["mode"]
        response.headers.add("X-Multi-Value", "one")
        response.headers.add("X-Multi-Value", "two")
        return response

    handler = FakeHandler(
        "GET",
        "/claimed/prefix/a/b?mode=fast",
        headers={"Host": "example.test:9999", "X-Token": "abc123"},
    )

    assert handle_with_flask_if_claimed(handler, app) is True
    assert handler.response_code == 202
    assert handler.wfile.getvalue() == b"tail=a/b"
    assert handler.header_values("X-Echo-Token") == ["abc123"]
    assert handler.header_values("X-Echo-Query") == ["fast"]
    assert handler.header_values("X-Multi-Value") == ["one", "two"]


def test_wsgi_adapter_passes_request_body_and_status_code() -> None:
    app = create_labeling_app(import_name=__name__)

    @claimed_route(app, "/claimed/body", methods=["POST"])
    def body_echo() -> Response:
        payload = request.get_data()
        return Response(
            payload,
            status=201,
            content_type=request.headers.get("Content-Type", "application/octet-stream"),
            headers={"X-Body-Length": str(len(payload))},
        )

    body = b'{"ok": true}'
    handler = FakeHandler(
        "POST",
        "/claimed/body",
        headers={"Content-Type": "application/json", "Content-Length": str(len(body))},
        body=body,
    )

    assert handle_with_flask_if_claimed(handler, app) is True
    assert handler.response_code == 201
    assert handler.header_values("X-Body-Length") == [str(len(body))]
    assert handler.header_values("Content-Type") == ["application/json"]
    assert handler.wfile.getvalue() == body


def test_unclaimed_path_falls_through_without_touching_flask() -> None:
    app = create_labeling_app(import_name=__name__)
    touched = {"before_request": 0, "view": 0}

    @app.before_request
    def count_before_request() -> None:
        touched["before_request"] += 1

    @claimed_route(app, "/claimed/only", methods=["GET"])
    def claimed_only() -> Response:
        touched["view"] += 1
        return Response("claimed")

    handler = FakeHandler("GET", "/legacy/still-owned")

    assert handle_with_flask_if_claimed(handler, app) is False
    assert touched == {"before_request": 0, "view": 0}
    assert handler.response_code is None
    assert handler.sent_headers == []
    assert handler.wfile.getvalue() == b""


def test_route_claims_are_method_aware_for_prefixes() -> None:
    app = create_labeling_app(import_name=__name__)

    @claimed_route(
        app,
        "/admin/read/<path:tail>",
        claim="prefix",
        claim_prefix_value="/admin/read",
        methods=["GET"],
    )
    def admin_read(tail: str) -> Response:
        return Response(f"tail={tail}", status=200)

    get_handler = FakeHandler("GET", "/admin/read/item")
    post_handler = FakeHandler("POST", "/admin/read/item")

    assert handle_with_flask_if_claimed(get_handler, app) is True
    assert get_handler.response_code == 200
    assert handle_with_flask_if_claimed(post_handler, app) is False
    assert post_handler.response_code is None


def test_claimed_flask_response_security_headers_match_existing_policy() -> None:
    app = create_labeling_app(import_name=__name__)

    @claimed_route(app, "/claimed/security", methods=["GET"])
    def security() -> Response:
        return Response("secure", status=204)

    handler = FakeHandler("GET", "/claimed/security")

    assert handle_with_flask_if_claimed(handler, app) is True
    assert handler.response_code == 204
    for name, expected in BROWSER_RESPONSE_SECURITY_HEADERS.items():
        assert handler.header_values(name) == [expected]


def test_personal_api_routes_are_get_only_and_use_shared_responder() -> None:
    app = create_labeling_app(import_name=__name__)
    state = SimpleNamespace(name="state")
    calls: list[tuple[object, str, str, str]] = []

    def response_builder(
        received_state: object,
        *,
        path: str,
        request_adapter: object,
    ) -> tuple[dict[str, object], HTTPStatus]:
        calls.append(
            (
                received_state,
                path,
                str(getattr(request_adapter, "path", "")),
                str(getattr(request_adapter, "headers", {}).get("X-User", "")),
            )
        )
        return {"ok": True, "path": path}, HTTPStatus.ACCEPTED

    register_personal_api_routes(app, state, response_builder)

    get_handler = FakeHandler(
        "GET",
        "/api/me/datasets?expected_user=alice",
        headers={"X-User": "alice"},
    )
    post_handler = FakeHandler("POST", "/api/me/datasets")

    assert handle_with_flask_if_claimed(get_handler, app) is True
    assert get_handler.response_code == int(HTTPStatus.ACCEPTED)
    assert json.loads(get_handler.wfile.getvalue().decode("utf-8")) == {
        "ok": True,
        "path": "/api/me/datasets",
    }
    assert calls == [
        (
            state,
            "/api/me/datasets",
            "/api/me/datasets?expected_user=alice",
            "alice",
        )
    ]

    assert handle_with_flask_if_claimed(post_handler, app) is False
    assert post_handler.response_code is None
