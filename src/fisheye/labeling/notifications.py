"""Email/outbox notifications for Palette web labeling workflows."""

from __future__ import annotations

import json
import os
import re
import smtplib
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from email.message import EmailMessage
from email.utils import formatdate, make_msgid
from pathlib import Path
from typing import Mapping
from urllib.parse import quote


NOTIFICATION_MODES = ("disabled", "outbox", "smtp")
NOTIFICATION_MODE_ENV_VAR = "PALETTE_LABELING_NOTIFICATION_MODE"
NOTIFICATION_FROM_ENV_VAR = "PALETTE_LABELING_NOTIFICATION_FROM"
NOTIFICATION_BASE_URL_ENV_VAR = "PALETTE_LABELING_BASE_URL"
NOTIFICATION_OUTBOX_ENV_VAR = "PALETTE_LABELING_NOTIFICATION_OUTBOX"
SMTP_HOST_ENV_VAR = "PALETTE_LABELING_SMTP_HOST"
SMTP_PORT_ENV_VAR = "PALETTE_LABELING_SMTP_PORT"
SMTP_USERNAME_ENV_VAR = "PALETTE_LABELING_SMTP_USERNAME"
SMTP_PASSWORD_ENV_VAR = "PALETTE_LABELING_SMTP_PASSWORD"
SMTP_STARTTLS_ENV_VAR = "PALETTE_LABELING_SMTP_STARTTLS"
SMTP_SSL_ENV_VAR = "PALETTE_LABELING_SMTP_SSL"

DEFAULT_NOTIFICATION_MODE = "outbox"
DEFAULT_NOTIFICATION_FROM = "Palette Labeling <palette-labeling@localhost>"
DEFAULT_NOTIFICATION_OUTBOX = "~/.palette/labeling_email_outbox"
PERSONAL_DATASET_QUEUE_PATH = "/my-datasets"


@dataclass(frozen=True)
class LabelingNotificationConfig:
    mode: str = DEFAULT_NOTIFICATION_MODE
    sender: str = DEFAULT_NOTIFICATION_FROM
    base_url: str | None = None
    outbox_dir: Path = Path(DEFAULT_NOTIFICATION_OUTBOX).expanduser()
    smtp_host: str | None = None
    smtp_port: int = 587
    smtp_username: str | None = None
    smtp_password: str | None = None
    smtp_starttls: bool = True
    smtp_ssl: bool = False

    @classmethod
    def from_env(
        cls,
        *,
        mode: str | None = None,
        base_url: str | None = None,
    ) -> "LabelingNotificationConfig":
        mode_value = str(mode or os.environ.get(NOTIFICATION_MODE_ENV_VAR) or DEFAULT_NOTIFICATION_MODE).strip().lower()
        if mode_value not in NOTIFICATION_MODES:
            raise ValueError(f"notification mode must be one of: {', '.join(NOTIFICATION_MODES)}")
        try:
            smtp_port = int(os.environ.get(SMTP_PORT_ENV_VAR) or "587")
        except ValueError:
            smtp_port = 587
        return cls(
            mode=mode_value,
            sender=str(os.environ.get(NOTIFICATION_FROM_ENV_VAR) or DEFAULT_NOTIFICATION_FROM),
            base_url=_normalize_base_url(base_url if base_url is not None else os.environ.get(NOTIFICATION_BASE_URL_ENV_VAR)),
            outbox_dir=Path(os.environ.get(NOTIFICATION_OUTBOX_ENV_VAR) or DEFAULT_NOTIFICATION_OUTBOX).expanduser(),
            smtp_host=_normalize_optional_text(os.environ.get(SMTP_HOST_ENV_VAR)),
            smtp_port=smtp_port,
            smtp_username=_normalize_optional_text(os.environ.get(SMTP_USERNAME_ENV_VAR)),
            smtp_password=_normalize_optional_text(os.environ.get(SMTP_PASSWORD_ENV_VAR)),
            smtp_starttls=_env_truthy(os.environ.get(SMTP_STARTTLS_ENV_VAR), default=True),
            smtp_ssl=_env_truthy(os.environ.get(SMTP_SSL_ENV_VAR), default=False),
        )


@dataclass(frozen=True)
class LabelingNotification:
    kind: str
    to_email: str
    to_user: str
    subject: str
    text_body: str


def send_labeler_added_notification(
    *,
    user: Mapping[str, object],
    actor_user: str | None = None,
    config: LabelingNotificationConfig | None = None,
) -> dict[str, object]:
    """Notify a user that their Palette labeling user row is active/available."""

    config = config or LabelingNotificationConfig.from_env()
    notification = _build_labeler_added_notification(user, base_url=config.base_url)
    return send_labeling_notification(
        notification,
        actor_user=actor_user,
        config=config,
        context={
            "user_id": notification.to_user,
            "role": str(user.get("role") or ""),
            "status": str(user.get("status") or ""),
        },
    )


def send_assignment_available_notification(
    *,
    user: Mapping[str, object] | None,
    assignment: Mapping[str, object],
    actor_user: str | None = None,
    config: LabelingNotificationConfig | None = None,
) -> dict[str, object]:
    """Notify a labeler that a recording assignment is available."""

    config = config or LabelingNotificationConfig.from_env()
    assignment_status = str(assignment.get("status") or "").strip()
    if assignment_status and assignment_status != "active":
        return _skipped_result(
            kind="assignment_available",
            to_user=str(assignment.get("assignee_user") or ""),
            reason="assignment_not_active",
            details="Assignment notifications are only sent for active assignments.",
        )
    notification = _build_assignment_available_notification(
        user=user,
        assignment=assignment,
        base_url=config.base_url,
    )
    return send_labeling_notification(
        notification,
        actor_user=actor_user,
        config=config,
        context={
            "recording_id": str(assignment.get("recording_id") or ""),
            "assignee_user": notification.to_user,
            "status": assignment_status,
        },
    )


def send_labeling_notification(
    notification: LabelingNotification,
    *,
    actor_user: str | None = None,
    config: LabelingNotificationConfig | None = None,
    context: Mapping[str, object] | None = None,
) -> dict[str, object]:
    config = config or LabelingNotificationConfig.from_env()
    if config.mode == "disabled":
        return _skipped_result(
            kind=notification.kind,
            to_user=notification.to_user,
            reason="notifications_disabled",
            details=f"{NOTIFICATION_MODE_ENV_VAR}=disabled.",
            context=context,
        )
    if not notification.to_email:
        return _skipped_result(
            kind=notification.kind,
            to_user=notification.to_user,
            reason="missing_recipient_email",
            details="The labeling user row does not have an email address.",
            context=context,
        )

    message = _build_email_message(notification, config=config, actor_user=actor_user)
    if config.mode == "outbox":
        return _write_outbox_message(message, notification=notification, config=config, context=context)
    if config.mode == "smtp":
        return _send_smtp_message(message, notification=notification, config=config, context=context)
    return _skipped_result(
        kind=notification.kind,
        to_user=notification.to_user,
        reason="unsupported_notification_mode",
        details=f"Unsupported notification mode: {config.mode}",
        context=context,
    )


def _build_labeler_added_notification(
    user: Mapping[str, object],
    *,
    base_url: str | None,
) -> LabelingNotification:
    user_id = str(user.get("user_id") or "").strip()
    role = str(user.get("role") or "labeler").strip() or "labeler"
    status = str(user.get("status") or "active").strip() or "active"
    display_name = str(user.get("display_name") or user_id).strip() or user_id
    email = str(user.get("email") or "").strip()
    queue_url = _personal_dataset_queue_url(user_id, base_url=base_url)
    subject = f"Palette labeling access added: {user_id}"
    body = "\n".join(
        [
            f"Hi {display_name},",
            "",
            "You have been added to the Palette web labeling system.",
            "",
            f"Username: {user_id}",
            f"Role: {role}",
            f"Status: {status}",
            "",
            "Use this username when opening the labeling website.",
            f"Assigned datasets queue: {queue_url}",
            "",
            "If the page says your browser identity does not match, contact the operator and include the username above.",
        ]
    )
    return LabelingNotification(
        kind="labeler_added",
        to_email=email,
        to_user=user_id,
        subject=subject,
        text_body=body,
    )


def _build_assignment_available_notification(
    *,
    user: Mapping[str, object] | None,
    assignment: Mapping[str, object],
    base_url: str | None,
) -> LabelingNotification:
    assignee_user = str(assignment.get("assignee_user") or (user or {}).get("user_id") or "").strip()
    display_name = str((user or {}).get("display_name") or assignee_user).strip() or assignee_user
    email = str((user or {}).get("email") or "").strip()
    recording_id = str(assignment.get("recording_id") or "").strip()
    notes = str(assignment.get("notes") or "").strip()
    queue_url = _personal_dataset_queue_url(assignee_user, base_url=base_url)
    subject = f"Palette dataset available for labeling: {recording_id}"
    lines = [
        f"Hi {display_name},",
        "",
        "A Palette labeling dataset has been assigned to you.",
        "",
        f"Username: {assignee_user}",
        f"Recording: {recording_id}",
        f"Assigned datasets queue: {queue_url}",
    ]
    if notes:
        lines.extend(["", f"Operator notes: {notes}"])
    lines.extend(
        [
            "",
            "Open the assigned datasets queue and start the available review tasks from there.",
            "If the page says your browser identity does not match, contact the operator and include the username above.",
        ]
    )
    return LabelingNotification(
        kind="assignment_available",
        to_email=email,
        to_user=assignee_user,
        subject=subject,
        text_body="\n".join(lines),
    )


def _build_email_message(
    notification: LabelingNotification,
    *,
    config: LabelingNotificationConfig,
    actor_user: str | None,
) -> EmailMessage:
    message = EmailMessage()
    message["From"] = config.sender
    message["To"] = notification.to_email
    message["Subject"] = notification.subject
    message["Date"] = formatdate(localtime=False)
    message["Message-ID"] = make_msgid(domain="palette-labeling.local")
    message["X-Palette-Labeling-Notification-Kind"] = notification.kind
    message["X-Palette-Labeling-User"] = notification.to_user
    if actor_user:
        message["X-Palette-Labeling-Actor"] = str(actor_user)
    message.set_content(notification.text_body)
    return message


def _write_outbox_message(
    message: EmailMessage,
    *,
    notification: LabelingNotification,
    config: LabelingNotificationConfig,
    context: Mapping[str, object] | None,
) -> dict[str, object]:
    outbox_dir = config.outbox_dir
    outbox_dir.mkdir(parents=True, exist_ok=True)
    notification_id = str(uuid.uuid4())
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_user = _safe_file_token(notification.to_user or "unknown")
    stem = f"{stamp}_{notification.kind}_{safe_user}_{notification_id}"
    eml_path = outbox_dir / f"{stem}.eml"
    json_path = outbox_dir / f"{stem}.json"
    eml_path.write_bytes(bytes(message))
    payload = _result_payload(
        ok=True,
        notification_id=notification_id,
        status="queued",
        transport="outbox",
        notification=notification,
        context=context,
        extra={
            "outbox_eml_path": str(eml_path),
            "outbox_json_path": str(json_path),
            "sent": False,
            "queued": True,
        },
    )
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def _send_smtp_message(
    message: EmailMessage,
    *,
    notification: LabelingNotification,
    config: LabelingNotificationConfig,
    context: Mapping[str, object] | None,
) -> dict[str, object]:
    if not config.smtp_host:
        return _failed_result(
            kind=notification.kind,
            to_user=notification.to_user,
            reason="missing_smtp_host",
            details=f"Set {SMTP_HOST_ENV_VAR} or use {NOTIFICATION_MODE_ENV_VAR}=outbox.",
            context=context,
        )
    notification_id = str(uuid.uuid4())
    try:
        smtp_cls = smtplib.SMTP_SSL if config.smtp_ssl else smtplib.SMTP
        with smtp_cls(config.smtp_host, config.smtp_port, timeout=30) as server:
            if config.smtp_starttls and not config.smtp_ssl:
                server.starttls()
            if config.smtp_username:
                server.login(config.smtp_username, config.smtp_password or "")
            server.send_message(message)
    except Exception as exc:
        return _failed_result(
            kind=notification.kind,
            to_user=notification.to_user,
            reason="smtp_send_failed",
            details=str(exc),
            context=context,
        )
    return _result_payload(
        ok=True,
        notification_id=notification_id,
        status="sent",
        transport="smtp",
        notification=notification,
        context=context,
        extra={"sent": True, "queued": False},
    )


def _result_payload(
    *,
    ok: bool,
    notification_id: str | None,
    status: str,
    transport: str,
    notification: LabelingNotification,
    context: Mapping[str, object] | None,
    extra: Mapping[str, object] | None = None,
) -> dict[str, object]:
    return {
        "ok": bool(ok),
        "schema": "palette.web_labeling_notification_result.v1",
        "notification_id": notification_id,
        "status": status,
        "transport": transport,
        "kind": notification.kind,
        "to_user": notification.to_user,
        "to_email": notification.to_email,
        "subject": notification.subject,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "context": dict(context or {}),
        **dict(extra or {}),
    }


def _skipped_result(
    *,
    kind: str,
    to_user: str,
    reason: str,
    details: str,
    context: Mapping[str, object] | None = None,
) -> dict[str, object]:
    return {
        "ok": False,
        "schema": "palette.web_labeling_notification_result.v1",
        "notification_id": None,
        "status": "skipped",
        "transport": "none",
        "kind": kind,
        "to_user": to_user,
        "reason": reason,
        "details": details,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "context": dict(context or {}),
        "sent": False,
        "queued": False,
    }


def _failed_result(
    *,
    kind: str,
    to_user: str,
    reason: str,
    details: str,
    context: Mapping[str, object] | None = None,
) -> dict[str, object]:
    return {
        "ok": False,
        "schema": "palette.web_labeling_notification_result.v1",
        "notification_id": None,
        "status": "failed",
        "transport": "smtp",
        "kind": kind,
        "to_user": to_user,
        "reason": reason,
        "details": details,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "context": dict(context or {}),
        "sent": False,
        "queued": False,
    }


def _personal_dataset_queue_url(user_id: str, *, base_url: str | None = None) -> str:
    path = f"{PERSONAL_DATASET_QUEUE_PATH}?expected_user={quote(str(user_id or '').strip())}"
    base_url = _normalize_base_url(base_url if base_url is not None else os.environ.get(NOTIFICATION_BASE_URL_ENV_VAR))
    return f"{base_url}{path}" if base_url else path


def _normalize_base_url(value: object) -> str | None:
    text = str(value or "").strip().rstrip("/")
    return text or None


def _normalize_optional_text(value: object) -> str | None:
    text = str(value or "").strip()
    return text or None


def _env_truthy(value: object, *, default: bool = False) -> bool:
    if value is None:
        return default
    text = str(value).strip().lower()
    if text in {"", "0", "false", "no", "off"}:
        return False
    if text in {"1", "true", "yes", "on"}:
        return True
    return default


def _safe_file_token(value: str) -> str:
    token = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return token.strip("._") or "unknown"
