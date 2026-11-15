"""Cacheable static assets for the labeling web UI: GET /static/<build>/<path>.

Every page used to inline its scripts and styles. Pages can instead reference
`static_url("js/x.js")`, which names the file under a build id: the sha256 of
the whole static tree (relative paths plus contents). Because the build id is a
directory, ES modules can import siblings by relative path, and any change to
any static file changes every URL, so responses can be cached as immutable.

A request for any other build id is a 404 rather than the current bytes, so a
browser never caches new content under an old immutable URL. Pages themselves
stay no-store, so a reload after a deploy picks up the new build id.

Assets are served without a labeling login: they are the application's own
code and styles, and carry no user or dataset data.
"""

from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path
from typing import Any

from flask import Flask, Response, abort, g

from .web_app import claimed_route

STATIC_ROUTE_PREFIX = "/static"
STATIC_ROOT = Path(__file__).resolve().parent / "static"
STATIC_CACHE_CONTROL = "public, max-age=31536000, immutable"
_BUILD_ID_LENGTH = 16

STATIC_CONTENT_TYPES: dict[str, str] = {
    ".js": "text/javascript; charset=utf-8",
    ".mjs": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".svg": "image/svg+xml",
    ".woff2": "font/woff2",
}


def _servable(relative: Path) -> bool:
    return relative.suffix in STATIC_CONTENT_TYPES and not any(
        part.startswith(".") for part in relative.parts
    )


@lru_cache(maxsize=None)
def _static_files() -> dict[str, bytes]:
    files: dict[str, bytes] = {}
    for path in sorted(STATIC_ROOT.rglob("*")):
        relative = path.relative_to(STATIC_ROOT)
        if path.is_file() and _servable(relative):
            files[relative.as_posix()] = path.read_bytes()
    return files


@lru_cache(maxsize=None)
def static_build_id() -> str:
    """The sha256 prefix of every servable static path and its bytes."""

    digest = hashlib.sha256()
    for relative, data in _static_files().items():
        digest.update(relative.encode("utf-8") + b"\0")
        digest.update(hashlib.sha256(data).digest())
    return digest.hexdigest()[:_BUILD_ID_LENGTH]


def static_url(relative_path: str) -> str:
    """The cacheable URL for one file under labeling/static/."""

    relative = str(relative_path).lstrip("/")
    if relative not in _static_files():
        raise KeyError(f"unknown labeling static asset: {relative_path!r}")
    return f"{STATIC_ROUTE_PREFIX}/{static_build_id()}/{relative}"


def register_static_routes(app: Flask) -> None:
    @claimed_route(
        app,
        f"{STATIC_ROUTE_PREFIX}/<build_id>/<path:relative_path>",
        claim="prefix",
        claim_prefix_value=STATIC_ROUTE_PREFIX,
        methods=["GET"],
    )
    def static_asset(build_id: str, relative_path: str) -> Any:
        data = _static_files().get(relative_path)
        if build_id != static_build_id() or data is None:
            abort(404)
        g.palette_labeling_cache_control = STATIC_CACHE_CONTROL
        return Response(
            data,
            status=200,
            content_type=STATIC_CONTENT_TYPES[Path(relative_path).suffix],
        )


__all__ = [
    "STATIC_CACHE_CONTROL",
    "STATIC_ROUTE_PREFIX",
    "register_static_routes",
    "static_build_id",
    "static_url",
]
