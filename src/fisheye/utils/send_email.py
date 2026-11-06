"""Preview, then send, a one-off email through the lab SMTP relay.

Usage::

    scripts/py -m fisheye.utils.send_email \\
        --to someone@janelia.hhmi.org --subject "Data for you" --body-file note.md
    # prints the full preview and sends nothing; add --send to send.

The plain-text body file is the source. A ``.md`` body also gets a simple HTML
alternative (headings, paragraphs, bullet lists, inline code). Unless
switched off, the AI-authorship disclosure is prepended and the sender is
cc'd. On ``--send`` the exact message is filed in the outbox before it is
handed to the relay; the command exits nonzero if any recipient is refused.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import textwrap
import uuid
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import Sequence

from fisheye.shared.email_delivery import (
    SmtpConfig,
    address_is_valid,
    bare_address,
    build_message,
    send,
    unique_addresses,
    write_outbox,
)

DEFAULT_FROM = "Jeremy Delahanty <delahantyj@hhmi.org>"
DEFAULT_REPLY_TO = "delahantyj@hhmi.org"
DEFAULT_SMTP_HOST = "proofpointrelayjrc.hhmi.org"
DEFAULT_SMTP_PORT = 25
DEFAULT_OUTBOX = "~/.palette/email_outbox"
MESSAGE_ID_DOMAIN = "hhmi.org"
AI_DISCLOSURE = (
    "This email was written by Claude, an AI assistant, at Jeremy's request. "
    "Jeremy asked for it to be sent; he is cc'd. Please reply to him with any questions."
)
RECORD_SCHEMA = "palette.send_email_record.v1"
COW = (
    "        \\   ^__^",
    "         \\  (oo)\\_______",
    "            (__)\\       )\\/\\",
    "                ||----w |",
    "                ||     ||",
)
COWSAY_WIDTH = 40


def cowsay(message: str, *, width: int = COWSAY_WIDTH) -> str:
    """The cowsay cow saying ``message``, wrapped to ``width`` columns (no dependency)."""

    lines = textwrap.wrap(" ".join(message.split()), width=width) or [""]
    inner = max(len(line) for line in lines)
    bubble = [" " + "_" * (inner + 2)]
    if len(lines) == 1:
        bubble.append(f"< {lines[0]} >")
    else:
        for index, line in enumerate(lines):
            left, right = ("/", "\\") if index == 0 else ("\\", "/") if index == len(lines) - 1 else ("|", "|")
            bubble.append(f"{left} {line.ljust(inner)} {right}")
    bubble.append(" " + "-" * (inner + 2))
    return "\n".join([*bubble, *COW])


def _inline(text: str) -> str:
    parts = re.split(r"(`[^`]+`)", text)
    out = []
    for part in parts:
        if len(part) >= 2 and part.startswith("`") and part.endswith("`"):
            out.append(f"<code>{escape(part[1:-1])}</code>")
        else:
            out.append(escape(part))
    return "".join(out)


def markdown_to_html(text: str) -> str:
    """A deliberately small Markdown subset: headings, paragraphs, bullets, inline code."""

    html: list[str] = []
    paragraph: list[str] = []
    items: list[str] = []

    def flush() -> None:
        if paragraph:
            html.append(f"<p>{_inline(' '.join(paragraph))}</p>")
            paragraph.clear()
        if items:
            html.append("<ul>" + "".join(f"<li>{_inline(item)}</li>" for item in items) + "</ul>")
            items.clear()

    for raw in text.splitlines():
        line = raw.rstrip()
        heading = re.match(r"^(#{1,6})\s+(.*)$", line)
        bullet = re.match(r"^\s*[-*]\s+(.*)$", line)
        if not line.strip():
            flush()
        elif heading:
            flush()
            level = len(heading.group(1))
            html.append(f"<h{level}>{_inline(heading.group(2))}</h{level}>")
        elif bullet:
            if paragraph:
                flush()
            items.append(bullet.group(1))
        elif items and raw[:1].isspace():
            items[-1] = f"{items[-1]} {line.strip()}"
        else:
            if items:
                flush()
            paragraph.append(line.strip())
    flush()
    return '<html><body style="font-family:sans-serif;line-height:1.45;max-width:760px">' + "".join(html) + "</body></html>"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--to", action="append", required=True, help="Recipient (repeatable).")
    parser.add_argument("--cc", action="append", default=[], help="Cc recipient (repeatable).")
    parser.add_argument("--subject", required=True)
    parser.add_argument("--body-file", required=True, type=Path, help="Plain-text body; .md also renders HTML.")
    parser.add_argument("--html-file", type=Path, help="Explicit HTML alternative (overrides .md rendering).")
    parser.add_argument("--from", dest="sender", default=DEFAULT_FROM)
    parser.add_argument("--reply-to", default=DEFAULT_REPLY_TO)
    parser.add_argument("--smtp-host", default=DEFAULT_SMTP_HOST)
    parser.add_argument("--smtp-port", type=int, default=DEFAULT_SMTP_PORT)
    parser.add_argument("--outbox", type=Path, default=Path(DEFAULT_OUTBOX))
    parser.add_argument("--no-ai-disclosure", action="store_true", help="Omit the AI-authorship disclosure line.")
    parser.add_argument("--no-cc-sender", action="store_true", help="Do not cc the From address.")
    parser.add_argument("--cowsay", help="What the cow says (default: the subject).")
    parser.add_argument("--no-cowsay", action="store_true", help="Omit the cowsay header.")
    parser.add_argument("--send", action="store_true", help="Actually send (default: preview only).")
    return parser


def _stem(subject: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    token = re.sub(r"[^A-Za-z0-9_.-]+", "_", subject.strip()).strip("._")[:60] or "email"
    return f"{stamp}_{token}_{uuid.uuid4()}"


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    for value in [*args.to, *args.cc, args.sender, args.reply_to]:
        if value and not address_is_valid(value):
            print(f"error: not a valid email address: {value!r}", file=sys.stderr)
            return 2
    to = unique_addresses(args.to)
    taken = {bare_address(value) for value in to}
    cc = [value for value in unique_addresses(args.cc) if bare_address(value) not in taken]
    if not args.no_cc_sender and bare_address(args.sender) not in taken | {bare_address(v) for v in cc}:
        cc.append(args.sender)

    body = args.body_file.read_text(encoding="utf-8").strip("\n")
    disclosure = None if args.no_ai_disclosure else AI_DISCLOSURE
    cow = None if args.no_cowsay else cowsay(args.cowsay or args.subject)
    text_body = "\n\n".join(part for part in (disclosure, cow, body) if part) + "\n"
    if args.html_file is not None:
        html_body = args.html_file.read_text(encoding="utf-8")
    elif args.body_file.suffix.lower() == ".md":
        header = ""
        if disclosure:
            header += f'<p style="color:#555"><em>{escape(disclosure)}</em></p>'
        if cow:
            header += (
                '<pre style="white-space:pre;font-family:monospace;font-size:13px;line-height:1.1">'
                f"{escape(cow)}</pre>"
            )
        rendered = markdown_to_html(body)
        opening = rendered.index(">") + 1 + rendered[rendered.index(">") + 1 :].index(">") + 1
        html_body = rendered[:opening] + header + rendered[opening:]
    else:
        html_body = None

    message = build_message(
        sender=args.sender,
        to=to,
        cc=cc,
        reply_to=args.reply_to or None,
        subject=args.subject,
        text_body=text_body,
        html_body=html_body,
        message_id_domain=MESSAGE_ID_DOMAIN,
        localtime=True,
    )

    if not args.send:
        for name in ("From", "To", "Cc", "Reply-To", "Subject"):
            if message[name]:
                print(f"{name}: {message[name]}")
        print(f"HTML alternative: {'yes' if html_body is not None else 'no'}")
        print()
        print(text_body, end="")
        print("\n[preview only: nothing sent; re-run with --send to send]")
        return 0

    outbox = args.outbox.expanduser()
    stem = _stem(args.subject)
    eml_path = write_outbox(message, outbox, stem)
    record_path = outbox / f"{stem}.json"
    record = {
        "schema": RECORD_SCHEMA,
        "message_id": message["Message-ID"],
        "from": args.sender,
        "to": to,
        "cc": cc,
        "subject": args.subject,
        "smtp_host": args.smtp_host,
        "smtp_port": args.smtp_port,
        "outbox_eml_path": str(eml_path),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "pending",
    }
    record_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    try:
        refused = send(message, SmtpConfig(host=args.smtp_host, port=args.smtp_port))
    except Exception as exc:  # report and record; never claim a send that failed
        record.update(status="failed", error=str(exc))
        record_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"error: send failed: {exc}", file=sys.stderr)
        return 1
    record.update(status="refused" if refused else "sent", refused={k: str(v) for k, v in refused.items()})
    record_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"sent {message['Message-ID']} refused: {refused}")
    print(f"outbox: {eml_path}")
    return 1 if refused else 0


if __name__ == "__main__":
    raise SystemExit(main())
