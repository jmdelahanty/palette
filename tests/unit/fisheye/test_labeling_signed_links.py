from __future__ import annotations

import pytest

from fisheye.labeling import web as labeling_web
from fisheye.labeling.assignment_store import LabelingStore


def test_signed_task_link_roundtrip_and_expiry(monkeypatch):
    monkeypatch.setattr(labeling_web.time, "time", lambda: 1_000)
    token = labeling_web._signed_task_link_token(task_id="task-a", secret="secret-a", ttl_seconds=60)

    payload = labeling_web._verify_signed_task_link_token(token, secret="secret-a")
    assert payload["task_id"] == "task-a"
    assert payload["iat"] == 1_000
    assert payload["exp"] == 1_060

    with pytest.raises(ValueError, match="Invalid signed link token"):
        labeling_web._verify_signed_task_link_token(token, secret="wrong-secret")

    monkeypatch.setattr(labeling_web.time, "time", lambda: 1_061)
    with pytest.raises(ValueError, match="expired"):
        labeling_web._verify_signed_task_link_token(token, secret="secret-a")


def test_signed_task_link_revocation_floor(monkeypatch):
    monkeypatch.setattr(labeling_web.time, "time", lambda: 1_000)
    old_token = labeling_web._signed_task_link_token(task_id="task-a", secret="secret-a", ttl_seconds=600)
    old_payload = labeling_web._verify_signed_task_link_token(old_token, secret="secret-a")

    monkeypatch.setattr(labeling_web.time, "time", lambda: 2_000)
    new_token = labeling_web._signed_task_link_token(task_id="task-a", secret="secret-a", ttl_seconds=600)
    new_payload = labeling_web._verify_signed_task_link_token(new_token, secret="secret-a")

    assert labeling_web._signed_task_link_revocation_reason(
        old_payload,
        not_before_utc="1970-01-01T00:25:00Z",
    )
    assert (
        labeling_web._signed_task_link_revocation_reason(
            new_payload,
            not_before_utc="1970-01-01T00:25:00Z",
        )
        is None
    )
    assert labeling_web._signed_task_link_revocation_reason(
        {"v": 1, "task_id": "task-a", "exp": 3_000},
        not_before_utc="1970-01-01T00:25:00Z",
    )


def test_signed_link_task_still_requires_current_assignment(tmp_path):
    store = LabelingStore(tmp_path / "labeling_work.sqlite")
    try:
        store.initialize()
        store.assign_recording(recording_id="rec-a", assignee_user="alice")
        store.upsert_task(task_id="task-a", recording_id="rec-a", workflow_kind="detect_training")

        lease = store.create_session(task_id="task-a", user="alice", ttl_seconds=600)
        assert lease.task_id == "task-a"

        with pytest.raises(PermissionError):
            store.create_session(task_id="task-a", user="bob", ttl_seconds=600)

        store.assign_recording(recording_id="rec-a", assignee_user="bob", status="active")

        with pytest.raises(PermissionError):
            store.create_session(task_id="task-a", user="alice", ttl_seconds=600)

        bob_lease = store.create_session(task_id="task-a", user="bob", ttl_seconds=600)
        assert bob_lease.user == "bob"
    finally:
        store.close()
