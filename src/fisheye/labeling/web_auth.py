"""Authentication and expected-user URL helpers for Palette web labeling."""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler
from typing import Mapping, Sequence
from urllib.parse import parse_qs, quote, unquote_plus, urlparse

DASHBOARD_PATH = "/work"

DATASET_QUEUE_PATH = "/datasets"

PERSONAL_DATASET_QUEUE_PATH = "/my-datasets"

SIGNED_INVITE_COOKIE_NAME = "palette_labeling_invite"

SIGNED_INVITE_SCOPE_PERSONAL_QUEUE = "personal_queue"

def _resolve_user(handler: BaseHTTPRequestHandler, config: ServerConfig) -> tuple[str | None, str]:
    invite_user, invite_source = _resolve_invite_user(handler, config)
    if invite_source:
        return invite_user, invite_source
    if config.fixed_user:
        return config.fixed_user, "fixed"
    if not config.trust_auth_header:
        return None, "auth_header_not_trusted"
    header_value = handler.headers.get(config.auth_header)
    if header_value:
        user = header_value.strip()
        if user:
            return user, f"header:{config.auth_header}"
    return None, f"missing_header:{config.auth_header}"

def _is_admin_user(user: str | None, config: ServerConfig) -> bool:
    if not user:
        return False
    return str(user) in {str(item) for item in config.admin_users}

def _b64url_decode(data: str) -> bytes:
    padding = "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode((data + padding).encode("ascii"))

def _verify_signed_invite_token(token: str, *, secret: str) -> dict[str, object]:
    parts = str(token or "").split(".", 1)
    if len(parts) != 2:
        raise ValueError("Malformed signed invite token.")
    payload_bytes = _b64url_decode(parts[0])
    expected = hmac.new(str(secret).encode("utf-8"), payload_bytes, hashlib.sha256).digest()
    provided = _b64url_decode(parts[1])
    if not hmac.compare_digest(expected, provided):
        raise ValueError("Invalid signed invite token.")
    payload = json.loads(payload_bytes.decode("utf-8"))
    if not isinstance(payload, dict) or int(payload.get("v") or 0) != 1:
        raise ValueError("Unsupported signed invite token.")
    if str(payload.get("kind") or "") != "labeler_invite":
        raise ValueError("Signed token is not a labeler invite.")
    if str(payload.get("scope") or "") != SIGNED_INVITE_SCOPE_PERSONAL_QUEUE:
        raise ValueError("Signed invite token has unsupported scope.")
    if int(payload.get("exp") or 0) < int(time.time()):
        raise ValueError("Signed invite token has expired.")
    user = str(payload.get("user") or "").strip()
    if not user:
        raise ValueError("Signed invite token is missing user.")
    return payload

def _invite_query_token_from_request(handler: BaseHTTPRequestHandler) -> str:
    query = parse_qs(urlparse(handler.path).query, keep_blank_values=True)
    return str((query.get("invite") or [""])[-1]).strip()

def _cookie_value(handler: BaseHTTPRequestHandler, name: str) -> str:
    cookie_header = str(handler.headers.get("Cookie") or "")
    if not cookie_header:
        return ""
    for raw_part in cookie_header.split(";"):
        if "=" not in raw_part:
            continue
        key, value = raw_part.split("=", 1)
        if key.strip() == name:
            return value.strip()
    return ""

def _invite_token_from_request(handler: BaseHTTPRequestHandler) -> str:
    query_token = _invite_query_token_from_request(handler)
    if query_token:
        return query_token
    return _cookie_value(handler, SIGNED_INVITE_COOKIE_NAME)

def _resolve_invite_user(handler: BaseHTTPRequestHandler, config: ServerConfig) -> tuple[str | None, str | None]:
    token = _invite_token_from_request(handler)
    if not token:
        return None, None
    if not config.link_secret:
        return None, "signed_invites_disabled"
    try:
        payload = _verify_signed_invite_token(token, secret=config.link_secret)
        revocation_reason = _signed_task_link_revocation_reason(
            payload,
            not_before_utc=config.link_not_before_utc,
        )
        if revocation_reason:
            return None, revocation_reason
        invite_user = str(payload.get("user") or "").strip()
        query = parse_qs(urlparse(handler.path).query, keep_blank_values=True)
        expected_user = str((query.get("expected_user") or [""])[-1]).strip()
        if expected_user and expected_user != invite_user:
            return None, "invite_expected_user_mismatch"
        return invite_user, f"invite:{payload.get('scope') or SIGNED_INVITE_SCOPE_PERSONAL_QUEUE}"
    except Exception as exc:
        return None, f"invite_error:{type(exc).__name__}"

def _utc_timestamp(value: str | int | float | None) -> int | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(float(text))
    except ValueError:
        pass
    normalized = text[:-1] + "+00:00" if text.endswith("Z") else text
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return int(parsed.timestamp())

def _signed_task_link_revocation_reason(payload: Mapping[str, object], *, not_before_utc: str | None) -> str | None:
    floor = _utc_timestamp(not_before_utc)
    if floor is None:
        return None
    issued_at = payload.get("iat")
    if issued_at is None:
        return "Signed link was created before issuance timestamps were recorded."
    try:
        issued_at_unix = int(float(str(issued_at)))
    except ValueError:
        return "Signed link has an invalid issuance timestamp."
    if issued_at_unix < floor:
        return "Signed link was created before the configured not-before timestamp."
    return None

def _expected_user_query_value_from_url(url: str) -> str:
    from urllib.parse import unquote_plus

    text = str(url or "")
    if "?" not in text:
        return ""
    query_text = text.split("?", 1)[1].split("#", 1)[0]
    for query_part in query_text.split("&"):
        query_key, separator, query_value = query_part.partition("=")
        if separator and unquote_plus(query_key) == "expected_user":
            return unquote_plus(query_value)
    return ""

def _dataset_queue_url_for_base(base_url: str | None) -> str:
    if not base_url:
        return ""
    return f"{str(base_url).rstrip('/')}{DATASET_QUEUE_PATH}"

def _personal_dataset_queue_url_for_base(base_url: str | None) -> str:
    if not base_url:
        return ""
    return f"{str(base_url).rstrip('/')}{PERSONAL_DATASET_QUEUE_PATH}"

def _dashboard_url_for_expected_user(dashboard_url: str | None, user: str | None) -> str:
    base = str(dashboard_url or "").strip()
    expected_user = str(user or "").strip()
    if not base or not expected_user:
        return base
    separator = "&" if "?" in base else "?"
    return f"{base}{separator}expected_user={quote(expected_user, safe='')}"

def _dataset_queue_url_for_dashboard(dashboard_url: str | None, user: str | None) -> str:
    dashboard = str(dashboard_url or "").strip()
    if not dashboard:
        return ""
    if dashboard.endswith(DASHBOARD_PATH):
        queue_url = f"{dashboard[:-len(DASHBOARD_PATH)]}{DATASET_QUEUE_PATH}"
    else:
        queue_url = DATASET_QUEUE_PATH if dashboard == DASHBOARD_PATH else ""
    return _dashboard_url_for_expected_user(queue_url, user) if queue_url else ""

def _personal_dataset_queue_url_for_dashboard(
    dashboard_url: str | None,
    user: str | None,
) -> str:
    dashboard = str(dashboard_url or "").strip()
    if not dashboard:
        return ""
    if dashboard.endswith(DASHBOARD_PATH):
        queue_url = f"{dashboard[:-len(DASHBOARD_PATH)]}{PERSONAL_DATASET_QUEUE_PATH}"
    else:
        queue_url = PERSONAL_DATASET_QUEUE_PATH if dashboard == DASHBOARD_PATH else ""
    return _dashboard_url_for_expected_user(queue_url, user) if queue_url else ""
