"""WSGI adapter for routing claimed labeling paths through Flask.

The adapter is intentionally conservative: it checks the Flask route-claim
registry before building a WSGI environ or invoking Flask. A ``False`` return
means the legacy ``BaseHTTPRequestHandler`` code should continue handling the
request exactly as it does today.
"""

from __future__ import annotations

import io
import sys
from collections.abc import Iterable
from typing import Any
from urllib.parse import unquote, urlsplit

from flask import Flask

from .web_app import is_path_claimed


def request_path_from_handler(handler: Any) -> str:
    """Return the URL path component from a BaseHTTPRequestHandler-like object."""

    raw_target = str(getattr(handler, "path", "/") or "/")
    return urlsplit(raw_target).path or "/"


def _handler_header_items(handler: Any) -> Iterable[tuple[str, str]]:
    headers = getattr(handler, "headers", None)
    if headers is None:
        return ()
    return ((str(name), str(value)) for name, value in headers.items())


def _handler_header(handler: Any, name: str, default: str = "") -> str:
    headers = getattr(handler, "headers", None)
    if headers is None:
        return default
    value = headers.get(name)
    if value is None:
        return default
    return str(value)


def _server_name_and_port(handler: Any) -> tuple[str, str]:
    server = getattr(handler, "server", None)
    name = str(getattr(server, "server_name", "localhost") or "localhost")
    port = str(getattr(server, "server_port", "80") or "80")
    host = _handler_header(handler, "Host")
    if not host:
        return name, port
    try:
        parsed = urlsplit(f"//{host}")
    except ValueError:
        return name, port
    if parsed.hostname:
        name = parsed.hostname
    if parsed.port is not None:
        port = str(parsed.port)
    return name, port


def build_wsgi_environ(handler: Any) -> dict[str, Any]:
    """Build a WSGI environ from a BaseHTTPRequestHandler-like request."""

    raw_target = str(getattr(handler, "path", "/") or "/")
    parsed = urlsplit(raw_target)
    server_name, server_port = _server_name_and_port(handler)
    scheme = _handler_header(handler, "X-Forwarded-Proto", "http").split(",", 1)[0].strip() or "http"
    client_address = getattr(handler, "client_address", ("", ""))
    remote_addr = str(client_address[0]) if client_address else ""

    environ: dict[str, Any] = {
        "REQUEST_METHOD": str(getattr(handler, "command", "GET") or "GET").upper(),
        "SCRIPT_NAME": "",
        "PATH_INFO": unquote(parsed.path or "/"),
        "QUERY_STRING": parsed.query,
        "SERVER_NAME": server_name,
        "SERVER_PORT": server_port,
        "SERVER_PROTOCOL": str(getattr(handler, "request_version", "HTTP/1.1") or "HTTP/1.1"),
        "REMOTE_ADDR": remote_addr,
        "wsgi.version": (1, 0),
        "wsgi.url_scheme": scheme,
        "wsgi.input": getattr(handler, "rfile", None) or io.BytesIO(),
        "wsgi.errors": sys.stderr,
        "wsgi.multithread": True,
        "wsgi.multiprocess": False,
        "wsgi.run_once": False,
    }

    for name, value in _handler_header_items(handler):
        environ_key = name.upper().replace("-", "_")
        if environ_key in {"CONTENT_TYPE", "CONTENT_LENGTH"}:
            target_key = environ_key
        else:
            target_key = f"HTTP_{environ_key}"
        if target_key in environ:
            environ[target_key] = f"{environ[target_key]},{value}"
        else:
            environ[target_key] = value
    return environ


def _write_wsgi_response(handler: Any, status: str, headers: list[tuple[str, str]], body_chunks: list[bytes]) -> None:
    status_code_text, _, reason = status.partition(" ")
    status_code = int(status_code_text)
    try:
        handler.send_response(status_code, reason or None)
    except TypeError:
        handler.send_response(status_code)
    for name, value in headers:
        handler.send_header(str(name), str(value))
    handler.end_headers()
    if str(getattr(handler, "command", "GET") or "GET").upper() == "HEAD":
        return
    for chunk in body_chunks:
        if chunk:
            handler.wfile.write(chunk)


def handle_with_flask_if_claimed(handler: Any, app: Flask) -> bool:
    """Dispatch to Flask only when the request path is explicitly claimed.

    Returns ``True`` when Flask wrote a response. Returns ``False`` without
    touching Flask when the path is unclaimed and should fall through to the
    legacy handler branch.
    """

    if not is_path_claimed(
        app,
        request_path_from_handler(handler),
        method=str(getattr(handler, "command", "GET") or "GET").upper(),
    ):
        return False

    environ = build_wsgi_environ(handler)
    response_status: str | None = None
    response_headers: list[tuple[str, str]] = []
    body_chunks: list[bytes] = []

    def start_response(status: str, headers: list[tuple[str, str]], exc_info: object = None) -> Any:
        nonlocal response_status, response_headers
        if exc_info is not None:
            exc_type, exc_value, traceback = exc_info  # type: ignore[misc]
            raise exc_value.with_traceback(traceback)
        response_status = status
        response_headers = list(headers)

        def write(data: bytes) -> None:
            body_chunks.append(bytes(data))

        return write

    result = app.wsgi_app(environ, start_response)
    try:
        for chunk in result:
            body_chunks.append(bytes(chunk))
    finally:
        close = getattr(result, "close", None)
        if callable(close):
            close()

    if response_status is None:
        raise RuntimeError("Flask WSGI app did not call start_response.")
    _write_wsgi_response(handler, response_status, response_headers, body_chunks)
    return True


__all__ = [
    "build_wsgi_environ",
    "handle_with_flask_if_claimed",
    "request_path_from_handler",
]
