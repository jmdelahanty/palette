"""Shared email mechanics: labeling notifications keep their exact bytes; the
send-email CLI previews by default and files before it sends. No network."""

from __future__ import annotations

import email
import json
import re
from email import policy
from pathlib import Path

import pytest

import fisheye.shared.email_delivery as delivery
from fisheye.labeling import notifications
from fisheye.utils import send_email

# Captured from fisheye.labeling.notifications._build_email_message at
# origin/main 8b3acb08, before the extraction, with Date and Message-ID fixed.
GOLDEN_PLAIN = (
    b"From: Palette Labeling <palette-labeling@localhost>\nTo: alice@example.org\n"
    b"Subject: Palette labeling access added: alice\nDate: Thu, 05 Nov 2026 12:00:00 -0000\n"
    b"Message-ID: <fixed@palette-labeling.local>\nX-Palette-Labeling-Notification-Kind: labeler_added\n"
    b"X-Palette-Labeling-User: alice\n{actor}Content-Type: text/plain; charset=\"utf-8\"\n"
    b"Content-Transfer-Encoding: 7bit\nMIME-Version: 1.0\n\nHi Alice,\n\nLine two.\n"
)
GOLDEN_HTML = (
    b"From: Palette Labeling <palette-labeling@localhost>\nTo: alice@example.org\n"
    b"Subject: Palette labeling access added: alice\nDate: Thu, 05 Nov 2026 12:00:00 -0000\n"
    b"Message-ID: <fixed@palette-labeling.local>\nX-Palette-Labeling-Notification-Kind: labeler_added\n"
    b"X-Palette-Labeling-User: alice\n{actor}MIME-Version: 1.0\nContent-Type: multipart/alternative;\n"
    b" boundary=\"B\"\n\n--B\nContent-Type: text/plain; charset=\"utf-8\"\nContent-Transfer-Encoding: 7bit\n\n"
    b"Hi Alice,\n\nLine two.\n\n--B\nContent-Type: text/html; charset=\"utf-8\"\n"
    b"Content-Transfer-Encoding: 7bit\nMIME-Version: 1.0\n\n<p>Hi</p>\n\n--B--\n"
)


class FakeSMTP:
    instances: list["FakeSMTP"] = []
    refuse: dict = {}

    def __init__(self, host, port, timeout):
        self.host, self.port, self.timeout = host, port, timeout
        self.sent = []
        self.calls = []
        FakeSMTP.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def starttls(self):
        self.calls.append("starttls")

    def login(self, user, password):
        self.calls.append(("login", user, password))

    def send_message(self, message):
        self.sent.append(message)
        return dict(FakeSMTP.refuse)


@pytest.fixture
def fake_smtp(monkeypatch):
    FakeSMTP.instances = []
    FakeSMTP.refuse = {}
    monkeypatch.setattr("smtplib.SMTP", FakeSMTP)
    monkeypatch.setattr("smtplib.SMTP_SSL", lambda *a, **k: pytest.fail("SSL not expected"))
    return FakeSMTP


# --- labeling: behaviour-preserving extraction ------------------------------


@pytest.mark.parametrize("html", [None, "<p>Hi</p>"])
@pytest.mark.parametrize("actor", [None, "bob"])
def test_labeling_message_bytes_are_unchanged(monkeypatch, html, actor):
    monkeypatch.setattr(delivery, "formatdate", lambda localtime=False: "Thu, 05 Nov 2026 12:00:00 -0000")
    monkeypatch.setattr(delivery, "make_msgid", lambda domain=None: f"<fixed@{domain}>")
    config = notifications.LabelingNotificationConfig(mode="outbox")
    note = notifications.LabelingNotification(
        kind="labeler_added",
        to_email="alice@example.org",
        to_user="alice",
        subject="Palette labeling access added: alice",
        text_body="Hi Alice,\n\nLine two.",
        html_body=html,
    )
    data = bytes(notifications._build_email_message(note, config=config, actor_user=actor))
    golden = (GOLDEN_PLAIN if html is None else GOLDEN_HTML).replace(
        b"{actor}", b"X-Palette-Labeling-Actor: bob\n" if actor else b""
    )
    if html is not None:
        data = re.sub(rb"=+\d+==", b"B", data)
    assert data == golden


def test_labeling_smtp_mode_uses_relay_settings(tmp_path, fake_smtp):
    config = notifications.LabelingNotificationConfig(
        mode="smtp", sender="p@example.org", smtp_host="smtp.example.org", smtp_port=2525,
        smtp_username="u", smtp_password="pw", smtp_starttls=True,
    )
    result = notifications.send_labeler_added_notification(
        user={"user_id": "alice", "email": "alice@example.org"}, config=config
    )
    assert result["status"] == "sent"
    (server,) = fake_smtp.instances
    assert (server.host, server.port, server.timeout) == ("smtp.example.org", 2525, 30)
    assert server.calls == ["starttls", ("login", "u", "pw")]
    assert len(server.sent) == 1


def test_labeling_outbox_writes_eml_and_json(tmp_path):
    config = notifications.LabelingNotificationConfig(mode="outbox", outbox_dir=tmp_path)
    result = notifications.send_labeler_added_notification(
        user={"user_id": "alice", "email": "alice@example.org"}, config=config
    )
    eml, record = Path(result["outbox_eml_path"]), Path(result["outbox_json_path"])
    assert eml.parent == tmp_path and eml.suffix == ".eml" and record.suffix == ".json"
    assert json.loads(record.read_text())["status"] == "queued"


# --- send-email CLI ---------------------------------------------------------


def _body(tmp_path: Path, name="note.md", text="# Hello\n\nSome `code` here.\n\n- one\n- two\n") -> Path:
    path = tmp_path / name
    path.write_text(text)
    return path


def _args(tmp_path, *extra):
    return ["--to", "yurii@example.org", "--subject", "Data for you",
            "--body-file", str(_body(tmp_path)), "--outbox", str(tmp_path / "outbox"), *extra]


def test_preview_sends_nothing(tmp_path, fake_smtp, capsys):
    assert send_email.main(_args(tmp_path)) == 0
    out = capsys.readouterr().out
    assert fake_smtp.instances == []
    assert not (tmp_path / "outbox").exists()
    assert "To: yurii@example.org" in out
    assert "Cc: Jeremy Delahanty <delahantyj@hhmi.org>" in out
    assert send_email.AI_DISCLOSURE in out
    assert "preview only" in out


def test_send_files_then_sends_once_with_headers(tmp_path, fake_smtp, capsys, monkeypatch):
    order = []
    real_write = send_email.write_outbox

    def write(message, outbox, stem):
        order.append("outbox")
        return real_write(message, outbox, stem)

    def relay(message, config):
        order.append("send")
        return delivery.send(message, config)

    monkeypatch.setattr(send_email, "write_outbox", write)
    monkeypatch.setattr(send_email, "send", relay)
    assert send_email.main(_args(tmp_path, "--send")) == 0
    assert order == ["outbox", "send"]
    (server,) = fake_smtp.instances
    assert (server.host, server.port, server.timeout) == ("proofpointrelayjrc.hhmi.org", 25, 30)
    assert server.calls == []  # plain relay: no STARTTLS, no login
    (message,) = server.sent
    assert message["To"] == "yurii@example.org"
    assert message["Cc"] == "Jeremy Delahanty <delahantyj@hhmi.org>"
    assert message["Reply-To"] == "delahantyj@hhmi.org"
    assert message.get_body(("plain",)).get_content().startswith(send_email.AI_DISCLOSURE)
    html = message.get_body(("html",)).get_content()
    assert "<h1>Hello</h1>" in html and "<code>code</code>" in html and "<li>one</li>" in html
    (eml,) = (tmp_path / "outbox").glob("*.eml")
    filed = email.message_from_bytes(eml.read_bytes(), policy=policy.default)
    assert filed["Message-ID"] == message["Message-ID"]
    (record,) = (tmp_path / "outbox").glob("*.json")
    assert json.loads(record.read_text())["status"] == "sent"


def test_refused_recipient_exits_nonzero_and_is_recorded(tmp_path, fake_smtp):
    fake_smtp.refuse = {"yurii@example.org": (550, b"no such user")}
    assert send_email.main(_args(tmp_path, "--send")) == 1
    (record,) = (tmp_path / "outbox").glob("*.json")
    assert json.loads(record.read_text())["status"] == "refused"


def test_relay_failure_is_recorded_not_claimed(tmp_path, monkeypatch):
    def boom(*_a, **_k):
        raise OSError("relay down")

    monkeypatch.setattr("smtplib.SMTP", boom)
    assert send_email.main(_args(tmp_path, "--send")) == 1
    (record,) = (tmp_path / "outbox").glob("*.json")
    assert json.loads(record.read_text())["status"] == "failed"


def test_flags_and_validation(tmp_path, fake_smtp, capsys):
    plain = _body(tmp_path, "note.txt", "Hello.\n")
    args = ["--to", "yurii@example.org", "--subject", "s", "--body-file", str(plain),
            "--outbox", str(tmp_path / "o"), "--no-ai-disclosure", "--no-cc-sender", "--send"]
    assert send_email.main(args) == 0
    (message,) = fake_smtp.instances[0].sent
    assert message["Cc"] is None
    assert not message.is_multipart()
    assert message.get_content() == "Hello.\n"
    assert send_email.main(["--to", "not an address", "--subject", "s", "--body-file", str(plain)]) == 2


def test_sender_already_addressed_is_not_cced_twice(tmp_path, fake_smtp):
    args = ["--to", "delahantyj@hhmi.org", "--cc", "x@example.org", "--cc", "X@example.org",
            "--subject", "s", "--body-file", str(_body(tmp_path)), "--outbox", str(tmp_path / "o"), "--send"]
    assert send_email.main(args) == 0
    (message,) = fake_smtp.instances[0].sent
    assert message["Cc"] == "x@example.org"


def test_address_validation():
    assert delivery.address_is_valid("Name <a@b.org>")
    assert delivery.address_is_valid("a@b.org")
    assert not delivery.address_is_valid("a b@c.org")
    assert not delivery.address_is_valid("nobody")
    assert not delivery.address_is_valid("")
