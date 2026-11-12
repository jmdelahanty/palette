"""Build, file and send plain email messages over an SMTP relay.

One implementation of the mechanics every Palette email uses: labeling
notifications (``fisheye.labeling.notifications``) and the general
``fisheye.utils.send_email`` CLI. Callers own their recipients, wording,
outbox naming and result records; this module only builds the
``EmailMessage``, writes its exact bytes to an outbox, and hands it to the
relay.
"""

from __future__ import annotations

import smtplib
from dataclasses import dataclass
from email.message import EmailMessage
from email.utils import formatdate, make_msgid, parseaddr
from pathlib import Path
from typing import Iterable, Mapping, Sequence

SMTP_TIMEOUT_SECONDS = 30


@dataclass(frozen=True)
class SmtpConfig:
    host: str
    port: int = 25
    username: str | None = None
    password: str | None = None
    starttls: bool = False
    ssl: bool = False
    timeout: int = SMTP_TIMEOUT_SECONDS


def address_is_valid(value: str) -> bool:
    """A single address (optionally ``Name <addr>``) with a local part and domain."""

    _name, address = parseaddr(str(value or ""))
    if not address or any(char.isspace() for char in address):
        return False
    local, sep, domain = address.rpartition("@")
    return bool(sep and local and domain)


def bare_address(value: str) -> str:
    return parseaddr(str(value or ""))[1].strip().lower()


def build_message(
    *,
    sender: str,
    to: Sequence[str] | str,
    subject: str,
    text_body: str,
    cc: Sequence[str] = (),
    reply_to: str | None = None,
    html_body: str | None = None,
    headers: Mapping[str, str] | None = None,
    message_id_domain: str,
    localtime: bool = False,
) -> EmailMessage:
    """Header order: From, To, Cc, Reply-To, Subject, Date, Message-ID, extras."""

    message = EmailMessage()
    message["From"] = sender
    message["To"] = to if isinstance(to, str) else ", ".join(to)
    if cc:
        message["Cc"] = ", ".join(cc)
    if reply_to:
        message["Reply-To"] = reply_to
    message["Subject"] = subject
    message["Date"] = formatdate(localtime=localtime)
    message["Message-ID"] = make_msgid(domain=message_id_domain)
    for name, value in (headers or {}).items():
        message[name] = value
    message.set_content(text_body)
    if html_body is not None:
        message.add_alternative(html_body, subtype="html")
    return message


def write_outbox(message: EmailMessage, outbox_dir: Path, stem: str) -> Path:
    """Write the message's exact bytes to ``<outbox_dir>/<stem>.eml``."""

    outbox_dir.mkdir(parents=True, exist_ok=True)
    path = outbox_dir / f"{stem}.eml"
    path.write_bytes(bytes(message))
    return path


def send(message: EmailMessage, config: SmtpConfig) -> dict[str, object]:
    """Send through the relay; return the recipients it refused (empty when none)."""

    smtp_cls = smtplib.SMTP_SSL if config.ssl else smtplib.SMTP
    with smtp_cls(config.host, config.port, timeout=config.timeout) as server:
        if config.starttls and not config.ssl:
            server.starttls()
        if config.username:
            server.login(config.username, config.password or "")
        refused = server.send_message(message)
    return dict(refused or {})


def unique_addresses(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        key = bare_address(value)
        if key and key not in seen:
            seen.add(key)
            out.append(value)
    return out


__all__ = [
    "SMTP_TIMEOUT_SECONDS",
    "SmtpConfig",
    "address_is_valid",
    "bare_address",
    "build_message",
    "send",
    "unique_addresses",
    "write_outbox",
]
