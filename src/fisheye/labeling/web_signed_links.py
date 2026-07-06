"""Signed task and invite link helpers for web labeling."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
import uuid
from datetime import datetime, timezone
from urllib.parse import quote

from .web_auth import PERSONAL_DATASET_QUEUE_PATH, SIGNED_INVITE_SCOPE_PERSONAL_QUEUE, _b64url_decode


def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def _effective_signed_link_ttl_seconds(ttl_seconds: int) -> int:
    return max(60, int(ttl_seconds))


def _signed_task_link_token_info(
    *,
    task_id: str,
    secret: str,
    ttl_seconds: int,
    expected_user: str | None = None,
) -> dict[str, object]:
    issued_at_unix = int(time.time())
    effective_ttl_seconds = _effective_signed_link_ttl_seconds(ttl_seconds)
    expires_at_unix = issued_at_unix + effective_ttl_seconds
    normalized_expected_user = str(expected_user or "").strip()
    payload = {
        "v": 1,
        "task_id": str(task_id),
        "iat": issued_at_unix,
        "exp": expires_at_unix,
        "nonce": uuid.uuid4().hex,
    }
    if normalized_expected_user:
        payload["expected_user"] = normalized_expected_user
    payload_bytes = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    signature = hmac.new(str(secret).encode("utf-8"), payload_bytes, hashlib.sha256).digest()
    token = f"{_b64url_encode(payload_bytes)}.{_b64url_encode(signature)}"
    return {
        "token": token,
        "issued_at_unix": issued_at_unix,
        "expires_at_unix": expires_at_unix,
        "issued_at_utc": datetime.fromtimestamp(issued_at_unix, tz=timezone.utc).isoformat(),
        "expires_at_utc": datetime.fromtimestamp(expires_at_unix, tz=timezone.utc).isoformat(),
        "ttl_seconds": effective_ttl_seconds,
        "expected_user": normalized_expected_user,
    }


def _signed_task_link_token(
    *,
    task_id: str,
    secret: str,
    ttl_seconds: int,
    expected_user: str | None = None,
) -> str:
    return str(
        _signed_task_link_token_info(
            task_id=task_id,
            secret=secret,
            ttl_seconds=ttl_seconds,
            expected_user=expected_user,
        )["token"]
    )


def _signed_invite_token_info(
    *,
    user: str,
    secret: str,
    ttl_seconds: int,
    scope: str = SIGNED_INVITE_SCOPE_PERSONAL_QUEUE,
) -> dict[str, object]:
    normalized_user = str(user or "").strip()
    if not normalized_user:
        raise ValueError("Signed invite tokens require a user.")
    issued_at_unix = int(time.time())
    effective_ttl_seconds = _effective_signed_link_ttl_seconds(ttl_seconds)
    expires_at_unix = issued_at_unix + effective_ttl_seconds
    payload = {
        "v": 1,
        "kind": "labeler_invite",
        "scope": str(scope or SIGNED_INVITE_SCOPE_PERSONAL_QUEUE),
        "user": normalized_user,
        "iat": issued_at_unix,
        "exp": expires_at_unix,
        "nonce": uuid.uuid4().hex,
    }
    payload_bytes = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    signature = hmac.new(str(secret).encode("utf-8"), payload_bytes, hashlib.sha256).digest()
    token = f"{_b64url_encode(payload_bytes)}.{_b64url_encode(signature)}"
    return {
        "token": token,
        "issued_at_unix": issued_at_unix,
        "expires_at_unix": expires_at_unix,
        "issued_at_utc": datetime.fromtimestamp(issued_at_unix, tz=timezone.utc).isoformat(),
        "expires_at_utc": datetime.fromtimestamp(expires_at_unix, tz=timezone.utc).isoformat(),
        "ttl_seconds": effective_ttl_seconds,
        "user": normalized_user,
        "scope": payload["scope"],
    }


def _signed_invite_token(
    *,
    user: str,
    secret: str,
    ttl_seconds: int,
    scope: str = SIGNED_INVITE_SCOPE_PERSONAL_QUEUE,
) -> str:
    return str(
        _signed_invite_token_info(
            user=user,
            secret=secret,
            ttl_seconds=ttl_seconds,
            scope=scope,
        )["token"]
    )


def _signed_invite_path(token: str, *, user: str) -> str:
    return (
        f"{PERSONAL_DATASET_QUEUE_PATH}"
        f"?expected_user={quote(str(user), safe='')}"
        f"&invite={quote(str(token), safe='')}"
    )


def _verify_signed_task_link_token(token: str, *, secret: str) -> dict[str, object]:
    parts = str(token or "").split(".", 1)
    if len(parts) != 2:
        raise ValueError("Malformed signed link token.")
    payload_bytes = _b64url_decode(parts[0])
    expected = hmac.new(str(secret).encode("utf-8"), payload_bytes, hashlib.sha256).digest()
    provided = _b64url_decode(parts[1])
    if not hmac.compare_digest(expected, provided):
        raise ValueError("Invalid signed link token.")
    payload = json.loads(payload_bytes.decode("utf-8"))
    if not isinstance(payload, dict) or int(payload.get("v") or 0) != 1:
        raise ValueError("Unsupported signed link token.")
    if int(payload.get("exp") or 0) < int(time.time()):
        raise ValueError("Signed link token has expired.")
    task_id = str(payload.get("task_id") or "").strip()
    if not task_id:
        raise ValueError("Signed link token is missing task_id.")
    return payload
