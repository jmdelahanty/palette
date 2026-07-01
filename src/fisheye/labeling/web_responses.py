"""HTTP response and lightweight request helpers for the labeling web server."""

from __future__ import annotations

import base64
import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler
from typing import Mapping, Sequence
from urllib.parse import urlparse

import numpy as np


def json_response(payload: object, *, status: HTTPStatus = HTTPStatus.OK) -> tuple[bytes, HTTPStatus, str]:
    return json.dumps(payload, allow_nan=False, sort_keys=True).encode("utf-8"), status, "application/json; charset=utf-8"


def format_error(
    error: str,
    *,
    details: str | None = None,
    status: HTTPStatus = HTTPStatus.BAD_REQUEST,
    extra: Mapping[str, object] | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {"ok": False, "error": error, "status": int(status)}
    if details:
        payload["details"] = details
    if extra:
        payload.update(dict(extra))
    return payload


def read_json_body(handler: BaseHTTPRequestHandler) -> dict[str, object]:
    raw_len = handler.headers.get("Content-Length")
    try:
        length = int(raw_len or "0")
    except ValueError:
        length = 0
    if length <= 0:
        return {}
    payload = json.loads(handler.rfile.read(length).decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("JSON request body must be an object.")
    return payload


def is_loopback_host(host: str) -> bool:
    return str(host).strip().lower() in {"127.0.0.1", "localhost", "::1", "[::1]"}


def request_has_same_origin(handler: BaseHTTPRequestHandler) -> bool:
    host = str(handler.headers.get("X-Forwarded-Host") or handler.headers.get("Host") or "").strip()
    if not host:
        return True
    origin = str(handler.headers.get("Origin") or "").strip()
    if origin:
        parsed_origin = urlparse(origin)
        if not parsed_origin.netloc or parsed_origin.netloc != host:
            return False
    referer = str(handler.headers.get("Referer") or "").strip()
    if referer:
        parsed_referer = urlparse(referer)
        if parsed_referer.netloc and parsed_referer.netloc != host:
            return False
    return True


def raw_array_payload(array: object) -> dict[str, object]:
    arr = np.asarray(array, dtype=np.uint8)
    return {
        "shape": [int(dim) for dim in arr.shape],
        "channels": int(arr.shape[-1]) if arr.ndim >= 3 else 1,
        "dtype": str(arr.dtype),
        "encoding": "base64_raw",
        "pixels": base64.b64encode(arr.tobytes()).decode("ascii"),
    }


def decode_uint8_payload(payload: Mapping[str, object]) -> np.ndarray:
    if str(payload.get("encoding") or "") != "base64_raw":
        raise ValueError("Expected base64_raw payload.")
    shape_raw = payload.get("shape")
    if not isinstance(shape_raw, Sequence):
        raise ValueError("Payload missing shape.")
    shape = tuple(int(dim) for dim in shape_raw)
    pixels = payload.get("pixels")
    if not isinstance(pixels, str):
        raise ValueError("Payload missing pixels.")
    arr = np.frombuffer(base64.b64decode(pixels.encode("ascii")), dtype=np.uint8)
    expected = int(np.prod(shape, dtype=np.int64))
    if int(arr.size) != expected:
        raise ValueError(f"Payload size mismatch: expected {expected}, got {arr.size}.")
    return arr.reshape(shape)


_json_response = json_response
_format_error = format_error
_read_json_body = read_json_body
_is_loopback_host = is_loopback_host
_request_has_same_origin = request_has_same_origin
_raw_array_payload = raw_array_payload
_decode_uint8_payload = decode_uint8_payload
