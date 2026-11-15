"""Labeling emails send labelers to /queue, with the SSH tunnel step for local servers."""

from __future__ import annotations

import email
from email import policy

import pytest

from fisheye.labeling.notifications import (
    LabelingNotificationConfig,
    send_assignment_available_notification,
    send_labeler_added_notification,
)

USER = {
    "user_id": "alice",
    "display_name": "Alice Example",
    "email": "alice@example.org",
    "role": "labeler",
    "status": "active",
}
ASSIGNMENT = {"recording_id": "rec-1", "assignee_user": "alice", "status": "active"}


@pytest.fixture(autouse=True)
def _no_tunnel_env(monkeypatch):
    for name in (
        "PALETTE_LABELING_BASE_URL",
        "PALETTE_LABELING_TUNNEL_LOCAL_PORT",
        "PALETTE_LABELING_TUNNEL_REMOTE_PORT",
        "PALETTE_LABELING_SSH_TARGET",
        "PALETTE_LABELING_REMOTE_HOST",
        "PALETTE_LABELING_REMOTE_BIND_HOST",
    ):
        monkeypatch.delenv(name, raising=False)


def _body(result) -> str:
    assert result["ok"] is True
    with open(result["outbox_eml_path"], "rb") as handle:
        message = email.message_from_binary_file(handle, policy=policy.default)
    return message.get_content()


def _config(tmp_path, base_url):
    return LabelingNotificationConfig(mode="outbox", base_url=base_url, outbox_dir=tmp_path)


def test_assignment_email_links_queue_and_tunnel_for_local_server(tmp_path, monkeypatch):
    monkeypatch.setenv("PALETTE_LABELING_REMOTE_HOST", "ws1.example.org")

    body = _body(
        send_assignment_available_notification(
            user=USER, assignment=ASSIGNMENT, config=_config(tmp_path, "http://127.0.0.1:8798")
        )
    )

    assert "Your labeling queue: http://127.0.0.1:8798/queue?expected_user=alice" in body
    assert "ssh -N -L 8798:127.0.0.1:8798 alice@ws1.example.org" in body
    assert "4. Open http://127.0.0.1:8798/queue in your browser." in body
    assert "/my-datasets" not in body


def test_assignment_email_without_base_url_has_relative_queue_link_and_no_tunnel(tmp_path):
    body = _body(
        send_assignment_available_notification(user=USER, assignment=ASSIGNMENT, config=_config(tmp_path, None))
    )

    assert "Your labeling queue: /queue?expected_user=alice" in body
    assert "ssh -N -L" not in body


def test_labeler_added_email_links_queue(tmp_path):
    body = _body(send_labeler_added_notification(user=USER, config=_config(tmp_path, "http://127.0.0.1:8798")))

    assert "http://127.0.0.1:8798/queue?expected_user=alice" in body
    assert "/my-datasets" not in body
