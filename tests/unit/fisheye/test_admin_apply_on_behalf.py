"""Operator Apply of a labeler's saved mask edits, through the editor's route."""

from __future__ import annotations

import json

import numpy as np
import pytest

from fisheye.labeling import web
from fisheye.labeling.admin_apply_on_behalf import (
    ADMIN_APPLY_EVENT,
    AdminApplyRefused,
    apply_on_behalf,
    apply_plan,
    main,
)
from tests.unit.fisheye.test_labeling_web_routes import _running_server
from tests.unit.fisheye.test_mask_tail_apply_refresh import browser_context, reviewed_archive  # noqa: F401
from tests.unit.fisheye.test_web_mask_tail_apply_integration import request


@pytest.fixture
def saved_but_unapplied(reviewed_archive, tmp_path):
    """The labeler saved one mask row in the editor and left without Apply."""

    path, root, initial = reviewed_archive
    store, _runtime = browser_context(reviewed_archive, tmp_path)
    store.upsert_labeling_user(user_id="reviewer", status="active")
    mask_run = root[initial["paths"]["mask_edit"]]
    edited = np.asarray(mask_run["masks_roi"][1, 0]).copy()
    edited[4:9, 100:107] = 1  # outside the synthetic body, so the change is visible
    session = store.create_session(task_id="original-mask", user="reviewer")
    route = f"/api/sessions/{session.session_id}/subject-mask"
    with _running_server(store, user="reviewer") as base:
        status, nav = request(base, route + "/nav", {"position": 1})
        assert status == 200, nav
        status, saved = request(base, route + "/save", {
            "mask": web._raw_array_payload(edited), "target_token": nav["state"]["target_token"],
        })
        assert status == 200, saved
    store.close_session(session_id=session.session_id, user="reviewer")
    before = np.asarray(mask_run["masks_roi"][1, 0]).copy()
    assert not np.array_equal(before, edited)
    try:
        yield store, root, initial, edited
    finally:
        store.close()


def _checkpoints(store, task_id):
    return [dict(r) for r in store.conn.execute(
        "SELECT user, state, apply_id, roi_idx FROM labeling_session_checkpoints WHERE task_id = ?;", (task_id,))]


def test_apply_on_behalf_runs_the_editor_apply_and_credits_the_labeler(saved_but_unapplied, tmp_path):
    store, root, initial, edited = saved_but_unapplied
    plan = apply_plan(store, "original-mask")
    assert plan["ok"] and plan["assignee"] == "reviewer" and plan["pending_rows"] == 1

    report = apply_on_behalf(store.path, "original-mask", actor="operator", backup_dir=tmp_path / "backups")

    outcome = report["outcome"]
    assert outcome["ok"], outcome
    assert outcome["applied_checkpoint_count"] == 1
    assert outcome["edit_revision_after"] == outcome["edit_revision_before"] + 1
    # The pixels reached the archive exactly as the labeler saved them.
    mask_run = root[initial["paths"]["mask_edit"]]
    np.testing.assert_array_equal(np.asarray(mask_run["masks_roi"][1, 0]), edited)
    # The checkpoint is applied and still the labeler's; the operator is on the event.
    rows = _checkpoints(store, "original-mask")
    assert [(r["user"], r["state"], r["apply_id"]) for r in rows] == [("reviewer", "applied", outcome["apply_id"])]
    event = store.conn.execute(
        "SELECT user, target_json, after_json FROM labeling_task_events WHERE event_type = ?;", (ADMIN_APPLY_EVENT,)
    ).fetchone()
    assert event["user"] == "operator"
    assert json.loads(event["target_json"])["labeler"] == "reviewer"
    # A validated store backup was taken first, and the private session is closed.
    assert report["backup"]["validation"]["integrity_check"] == "ok"
    assert apply_plan(store, "original-mask")["open_sessions"] == []


def test_refuses_while_the_labeler_has_the_task_open(saved_but_unapplied, tmp_path):
    store, *_ = saved_but_unapplied
    store.create_session(task_id="original-mask", user="reviewer")
    plan = apply_plan(store, "original-mask")
    assert not plan["ok"] and any("open editor session" in r for r in plan["refusals"])
    with pytest.raises(AdminApplyRefused, match="open editor session"):
        apply_on_behalf(store.path, "original-mask", actor="operator", backup_dir=tmp_path / "b")
    assert [r["state"] for r in _checkpoints(store, "original-mask")] == ["active"]


def test_refuses_keypoint_tasks_and_tasks_with_nothing_saved(saved_but_unapplied):
    store, *_ = saved_but_unapplied
    keypoints = apply_plan(store, "original-pose")
    assert any("not supported" in r for r in keypoints["refusals"])
    assert any("no saved, unapplied rows" in r for r in keypoints["refusals"])


def test_dry_run_cli_reports_the_plan_without_applying(saved_but_unapplied, capsys):
    store, *_ = saved_but_unapplied
    code = main(["--store", str(store.path), "--task-id", "original-mask", "--actor", "operator"])
    out = json.loads(capsys.readouterr().out)
    assert code == 0 and out["dry_run"] is True and out["plan"]["pending_rows"] == 1
    assert [r["state"] for r in _checkpoints(store, "original-mask")] == ["active"]
