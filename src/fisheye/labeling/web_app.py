"""Flask application factory for the labeling web strangler.

This module intentionally owns only framework scaffolding:

- an explicit route-claim registry,
- reusable security-header hooks,
- and an application factory with no migrated route families.

Legacy ``BaseHTTPRequestHandler`` dispatch remains the source of truth until a
route family is deliberately claimed and ported.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal, TypeVar

from flask import Flask, abort, g, request

from .web_policy import BROWSER_RESPONSE_SECURITY_HEADERS


_ROUTE_CLAIMS_EXTENSION = "palette_labeling_route_claims"
_SECURITY_HOOKS_EXTENSION = "palette_labeling_security_hooks_installed"

_ViewFunc = TypeVar("_ViewFunc", bound=Callable[..., Any])
ClaimMode = Literal["path", "prefix", "none"]


def normalize_claim_path(path: str) -> str:
    """Normalize a request or claim path for exact/prefix matching."""

    text = str(path or "/")
    text = text.split("?", 1)[0].split("#", 1)[0]
    if not text.startswith("/"):
        text = f"/{text}"
    return text or "/"


@dataclass
class RouteClaims:
    """Explicit path/prefix claims for Flask-owned labeling routes."""

    paths: set[str] = field(default_factory=set)
    prefixes: set[str] = field(default_factory=set)

    def claim_path(self, path: str) -> "RouteClaims":
        self.paths.add(normalize_claim_path(path))
        return self

    def claim_prefix(self, prefix: str) -> "RouteClaims":
        normalized = normalize_claim_path(prefix)
        if normalized != "/":
            normalized = normalized.rstrip("/")
        self.prefixes.add(normalized or "/")
        return self

    def is_claimed(self, path: str) -> bool:
        normalized = normalize_claim_path(path)
        if normalized in self.paths:
            return True
        for prefix in self.prefixes:
            if prefix == "/" or normalized == prefix or normalized.startswith(f"{prefix}/"):
                return True
        return False


def install_route_claims(app: Flask, claims: RouteClaims | None = None) -> RouteClaims:
    """Attach route claims to a Flask app and return the active registry."""

    active = claims if claims is not None else RouteClaims()
    app.extensions[_ROUTE_CLAIMS_EXTENSION] = active
    return active


def get_route_claims(app: Flask) -> RouteClaims:
    """Return the app route-claim registry, creating one if needed."""

    claims = app.extensions.get(_ROUTE_CLAIMS_EXTENSION)
    if isinstance(claims, RouteClaims):
        return claims
    return install_route_claims(app)


def is_path_claimed(app: Flask, path: str) -> bool:
    """Return whether Flask is allowed to handle ``path``."""

    return get_route_claims(app).is_claimed(path)


def claim_path(app: Flask, path: str) -> None:
    """Claim one exact path for Flask handling."""

    get_route_claims(app).claim_path(path)


def claim_prefix(app: Flask, prefix: str) -> None:
    """Claim one route-family prefix for Flask handling."""

    get_route_claims(app).claim_prefix(prefix)


def claimed_route(
    app: Flask,
    rule: str,
    *,
    claim: ClaimMode = "path",
    claim_path_value: str | None = None,
    claim_prefix_value: str | None = None,
    **options: Any,
) -> Callable[[_ViewFunc], _ViewFunc]:
    """Register a Flask route and explicitly claim the matching path family.

    ``claim="path"`` is appropriate for static rules. Dynamic route families
    should use ``claim="prefix"`` with ``claim_prefix_value`` so the legacy
    adapter can decide ownership before Flask routing runs.
    """

    def decorator(view_func: _ViewFunc) -> _ViewFunc:
        registered = app.route(rule, **options)(view_func)
        claims = get_route_claims(app)
        if claim == "path":
            claims.claim_path(claim_path_value or rule)
        elif claim == "prefix":
            claims.claim_prefix(claim_prefix_value or rule)
        elif claim != "none":
            raise ValueError(f"Unsupported Flask route claim mode: {claim!r}")
        return registered

    return decorator


def browser_response_security_headers() -> dict[str, str]:
    """Return the canonical labeling browser response-security headers."""

    return dict(BROWSER_RESPONSE_SECURITY_HEADERS)


def apply_browser_response_security_headers(response: Any) -> Any:
    """Apply the existing labeling response-security policy to a Flask response."""

    for name, value in browser_response_security_headers().items():
        response.headers[name] = value
    return response


def install_security_hooks(app: Flask) -> Flask:
    """Install synchronous request hooks shared by future Flask route families."""

    if app.extensions.get(_SECURITY_HOOKS_EXTENSION):
        return app

    @app.before_request
    def _palette_labeling_require_claimed_route() -> None:
        if not is_path_claimed(app, request.path):
            abort(404)
        g.palette_labeling_claimed_path = request.path

    @app.after_request
    def _palette_labeling_apply_security_headers(response: Any) -> Any:
        return apply_browser_response_security_headers(response)

    app.extensions[_SECURITY_HOOKS_EXTENSION] = True
    return app


def create_labeling_app(
    *,
    import_name: str = __name__,
    claims: RouteClaims | None = None,
    config: dict[str, Any] | None = None,
    **flask_options: Any,
) -> Flask:
    """Create the Flask app used by the labeling strangler adapter.

    No production route family is registered here yet. Future route modules
    should register with :func:`claimed_route`, :func:`claim_path`, or
    :func:`claim_prefix`; unclaimed paths remain legacy-owned.
    """

    app = Flask(import_name, **flask_options)
    if config:
        app.config.update(config)
    install_route_claims(app, claims)
    install_security_hooks(app)
    return app


__all__ = [
    "RouteClaims",
    "apply_browser_response_security_headers",
    "browser_response_security_headers",
    "claim_path",
    "claim_prefix",
    "claimed_route",
    "create_labeling_app",
    "get_route_claims",
    "install_route_claims",
    "install_security_hooks",
    "is_path_claimed",
    "normalize_claim_path",
]
