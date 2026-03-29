"""HTTP handlers for the recording status page."""

from __future__ import annotations

import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Optional
from urllib.parse import parse_qs, unquote, urlparse

from .query import (
    build_health_report,
    query_dataset_status,
    query_status_heartbeat,
    query_status_history,
    query_status_summary,
    query_status_wide,
)

_CONTENT_TYPES: Mapping[str, str] = MappingProxyType(
    {
        ".css": "text/css; charset=utf-8",
        ".html": "text/html; charset=utf-8",
        ".js": "application/javascript; charset=utf-8",
        ".json": "application/json; charset=utf-8",
        ".png": "image/png",
        ".svg": "image/svg+xml",
        ".txt": "text/plain; charset=utf-8",
    }
)


def _param_first(query_params: Mapping[str, list[str]], key: str) -> Optional[str]:
    values = query_params.get(key)
    if not values:
        return None
    value = values[0].strip()
    return value or None


def _parse_int_param(
    query_params: Mapping[str, list[str]],
    key: str,
    *,
    default: int,
) -> int:
    value = _param_first(query_params, key)
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid integer for '{key}': {value}") from exc


def _parse_bool_param(
    query_params: Mapping[str, list[str]],
    key: str,
    *,
    default: bool = False,
) -> bool:
    value = _param_first(query_params, key)
    if value is None:
        return default
    lowered = value.lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Invalid boolean for '{key}': {value}")


def build_handler(*, registry_path: Path, static_dir: Path) -> type[BaseHTTPRequestHandler]:
    static_root = static_dir.resolve()

    class StatusPageRequestHandler(BaseHTTPRequestHandler):
        server_version = "PaletteStatusPage/0.2"
        sys_version = ""

        def _write_bytes(
            self,
            payload: bytes,
            *,
            status: HTTPStatus = HTTPStatus.OK,
            content_type: str = "application/octet-stream",
        ) -> None:
            self.send_response(int(status))
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def _write_json(self, payload: object, *, status: HTTPStatus = HTTPStatus.OK) -> None:
            data = json.dumps(payload, sort_keys=True).encode("utf-8")
            self._write_bytes(data, status=status, content_type="application/json; charset=utf-8")

        def _write_error(self, status: HTTPStatus, *, error: str, details: Optional[str] = None) -> None:
            payload: dict[str, Any] = {
                "ok": False,
                "error": error,
            }
            if details:
                payload["details"] = details
            self._write_json(payload, status=status)

        def _write_not_found(self, message: str = "Not found") -> None:
            self._write_error(HTTPStatus.NOT_FOUND, error=message)

        def _serve_static(self, relative_path: str) -> None:
            candidate = (static_root / relative_path).resolve()
            if not candidate.is_relative_to(static_root) or not candidate.is_file():
                self._write_not_found("Static asset not found")
                return
            suffix = candidate.suffix.lower()
            content_type = _CONTENT_TYPES.get(suffix, "application/octet-stream")
            self._write_bytes(candidate.read_bytes(), content_type=content_type)

        def _handle_api_get(self, path: str, query_params: Mapping[str, list[str]]) -> bool:
            if path == "/api/status/summary":
                summary = query_status_summary(registry_path)
                self._write_json({"ok": True, "summary": summary})
                return True

            if path == "/api/status/heartbeat":
                heartbeat = query_status_heartbeat(registry_path)
                self._write_json({"ok": True, "heartbeat": heartbeat})
                return True

            if path == "/api/status/wide":
                q = _param_first(query_params, "q")
                zarr_use = _param_first(query_params, "zarr_use")
                only_blocking = _parse_bool_param(query_params, "only_blocking", default=False)
                limit = _parse_int_param(query_params, "limit", default=200)
                offset = _parse_int_param(query_params, "offset", default=0)
                result = query_status_wide(
                    registry_path,
                    q=q,
                    zarr_use=zarr_use,
                    only_blocking=only_blocking,
                    limit=limit,
                    offset=offset,
                )
                self._write_json({"ok": True, **result})
                return True

            dataset_prefix = "/api/status/dataset/"
            if path.startswith(dataset_prefix):
                encoded_dataset = path[len(dataset_prefix) :]
                dataset_id = unquote(encoded_dataset).strip()
                if not dataset_id:
                    raise ValueError("dataset_id path segment is required.")
                result = query_dataset_status(registry_path, dataset_id=dataset_id)
                self._write_json({"ok": True, **result})
                return True

            if path == "/api/status/history":
                dataset_id = _param_first(query_params, "dataset_id")
                if dataset_id is None:
                    raise ValueError("dataset_id query parameter is required.")
                step_name = _param_first(query_params, "step_name")
                limit = _parse_int_param(query_params, "limit", default=200)
                result = query_status_history(
                    registry_path,
                    dataset_id=dataset_id,
                    step_name=step_name,
                    limit=limit,
                )
                self._write_json({"ok": True, **result})
                return True

            return False

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            path = parsed.path or "/"
            query_params = parse_qs(parsed.query, keep_blank_values=False)

            if path == "/":
                self._serve_static("index.html")
                return
            if path == "/healthz":
                report = build_health_report(registry_path)
                status = HTTPStatus.OK if report.ok else HTTPStatus.SERVICE_UNAVAILABLE
                self._write_json(report.to_dict(), status=status)
                return
            if path.startswith("/static/"):
                rel = path.removeprefix("/static/").lstrip("/")
                if not rel:
                    self._write_not_found()
                    return
                self._serve_static(rel)
                return
            if path.startswith("/api/"):
                try:
                    handled = self._handle_api_get(path, query_params)
                except ValueError as exc:
                    self._write_error(HTTPStatus.BAD_REQUEST, error="invalid_request", details=str(exc))
                    return
                except Exception as exc:
                    self._write_error(HTTPStatus.INTERNAL_SERVER_ERROR, error="internal_error", details=str(exc))
                    return
                if handled:
                    return
            self._write_not_found()

        def do_HEAD(self) -> None:
            parsed = urlparse(self.path)
            path = parsed.path or "/"
            if path in {"/", "/healthz"} or path.startswith("/api/"):
                self.send_response(HTTPStatus.OK)
                self.end_headers()
                return
            if path.startswith("/static/"):
                rel = path.removeprefix("/static/").lstrip("/")
                candidate = (static_root / rel).resolve() if rel else static_root
                if rel and candidate.is_relative_to(static_root) and candidate.is_file():
                    self.send_response(HTTPStatus.OK)
                    self.end_headers()
                    return
            self.send_response(HTTPStatus.NOT_FOUND)
            self.end_headers()

        def log_message(self, format: str, *args: object) -> None:
            # Keep server output quiet by default.
            return

    return StatusPageRequestHandler
