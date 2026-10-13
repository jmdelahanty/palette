"""Re-save mask checkpoints made stale by another component's Apply on the same run."""

from __future__ import annotations

import json
import shutil

import numpy as np
import pytest
import zarr

from fisheye.labeling import web
from fisheye.labeling.admin_apply_on_behalf import apply_on_behalf
from fisheye.labeling.rebase_stale_mask_saves import (
    PROOF_RECEIPT_CHAIN,
    REBASE_EVENT,
    REBASE_REASON,
    RebaseRefused,
    main,
    rebase_plan,
    rebase_stale_saves,
)
from tests.unit.fisheye.test_labeling_web_routes import _running_server
from tests.unit.fisheye.test_mask_tail_apply_refresh import browser_context, reviewed_archive  # noqa: F401
from tests.unit.fisheye.test_web_mask_tail_apply_integration import request


def _save(base, session_id, roi, mask):
    route = f"/api/sessions/{session_id}/subject-mask"
    status, nav = request(base, route + "/nav", {"roi_idx": roi})
    assert status == 200, nav
    status, saved = request(base, route + "/save", {"mask": web._raw_array_payload(mask), "target_token": nav["state"]["target_token"], "advance": False})
    assert status == 200, saved


@pytest.fixture
def stale_body_save(reviewed_archive, tmp_path):
    """Body and eye rows saved at revision R; the eye task applied first (R -> R+1)."""

    path, root, initial = reviewed_archive
    store, _runtime = browser_context(reviewed_archive, tmp_path)
    store.upsert_labeling_user(user_id="reviewer", status="active")
    run_rel = initial["paths"]["mask_edit"]
    task = store.get_task("original-mask")
    store.upsert_task(recording_id="rec", task_id="eyes-task", workflow_kind="subject_mask_component",
                      run_name=task["run_name"], component_name="eyes_union",
                      scope={**task["scope"], "component_name": "eyes_union"})
    base_backup = tmp_path / "base-run"
    shutil.copytree(path / run_rel, base_backup)
    run = root[run_rel]
    body = np.asarray(run["masks_roi"][1, 0]).copy()
    body[4:9, 100:107] = 1
    eyes = np.asarray(run["masks_roi"][1, 1]).copy()
    eyes[110:114, 10:14] = 1
    for task_id, mask in (("original-mask", body), ("eyes-task", eyes)):
        session = store.create_session(task_id=task_id, user="reviewer")
        with _running_server(store, user="reviewer") as base:
            _save(base, session.session_id, 1, mask)
        store.close_session(session_id=session.session_id, user="reviewer")
    # The body save stands for one made before saves recorded their base
    # component digest (the case this tool exists for): without that digest,
    # Apply cannot judge it by content and keeps the revision rule.
    legacy = store.get_session_checkpoint(task_id="original-mask", roi_idx=1, component_name="subject_body")
    metadata = {k: v for k, v in legacy["metadata"].items() if k != "base_component_sha256"}
    store.upsert_session_checkpoint(
        session_id=legacy["session_id"], task_id="original-mask", recording_id="rec", user="reviewer",
        workflow_kind="subject_mask_component", target_run_path=legacy["target_run_path"],
        target_edit_revision=legacy["target_edit_revision"], source_rowset_path=legacy["source_rowset_path"],
        roi_idx=1, component_name="subject_body", payload=legacy["payload"], metadata=metadata,
    )
    first = apply_on_behalf(store.path, "eyes-task", actor="operator", skip_backup=True)["outcome"]
    assert first["ok"] and first["applied_checkpoint_count"] == 1, first
    try:
        yield store, path, run_rel, base_backup, body
    finally:
        store.close()


def test_stale_body_save_is_reported_rebased_with_audit_and_then_applied(stale_body_save):
    store, path, run_rel, base_backup, body = stale_body_save
    original = dict(store.conn.execute(
        "SELECT checkpoint_id, target_edit_revision, updated_at_utc FROM labeling_session_checkpoints WHERE task_id='original-mask';"
    ).fetchone())

    # The editor route skips the body save as stale; the admin command now says so.
    skipped = apply_on_behalf(store.path, "original-mask", actor="operator", skip_backup=True)["outcome"]
    assert not skipped["ok"] and skipped["stale_checkpoint_count"] == 1 and skipped["applied_checkpoint_count"] == 0

    plan = rebase_plan(store.path, "original-mask", base_backup)
    assert plan["ok"] and len(plan["stale_rows"]) == 1 and plan["conflict_rows"] == []
    assert plan["current_revision"] == plan["base_revision"] + 1

    report = rebase_stale_saves(store.path, "original-mask", base_backup, actor="operator", skip_backup=True)
    assert report["outcome"] == {"ok": True, "rebased_rows": 1, "row_count": 1, "failures": []}
    now = dict(store.conn.execute(
        "SELECT user, state, target_edit_revision, payload_json FROM labeling_session_checkpoints WHERE task_id='original-mask';"
    ).fetchone())
    assert now["user"] == "reviewer" and now["state"] == "active"
    assert now["target_edit_revision"] == plan["current_revision"]
    audit = json.loads(store.conn.execute(
        "SELECT before_json FROM labeling_task_events WHERE event_type = ? AND user = 'operator';", (REBASE_EVENT,)
    ).fetchone()[0])
    row = audit["rows"][0]
    assert audit["reason"] == REBASE_REASON and audit["base_backup"] == str(base_backup)
    assert row["checkpoint_id"] == original["checkpoint_id"]
    assert row["original_target_edit_revision"] == original["target_edit_revision"]
    assert row["original_saved_at_utc"] and row["base_component_sha256"] == row["current_component_sha256"]

    applied = apply_on_behalf(store.path, "original-mask", actor="operator", skip_backup=True)["outcome"]
    assert applied["ok"] and applied["applied_checkpoint_count"] == 1, applied
    run = zarr.open_group(str(path), mode="r", use_consolidated=False)[run_rel]
    np.testing.assert_array_equal(np.asarray(run["masks_roi"][1, 0]), (body > 0).astype(np.uint8))


def test_refuses_when_the_component_pixels_changed_after_the_save(stale_body_save):
    store, path, run_rel, base_backup, _body = stale_body_save
    run = zarr.open_group(str(path), mode="a", use_consolidated=False)[run_rel]
    changed = np.asarray(run["masks_roi"][1, 0]).copy()
    changed[0:3, 0:3] = 1 - changed[0:3, 0:3]
    run["masks_roi"][1, 0] = changed
    plan = rebase_plan(store.path, "original-mask", base_backup)
    assert not plan["ok"] and len(plan["conflict_rows"]) == 1
    with pytest.raises(RebaseRefused, match="pixels changed"):
        rebase_stale_saves(store.path, "original-mask", base_backup, actor="operator", skip_backup=True)
    assert store.conn.execute(
        "SELECT target_edit_revision FROM labeling_session_checkpoints WHERE task_id='original-mask';"
    ).fetchone()[0] == plan["base_revision"]


def test_refuses_open_sessions_and_a_backup_at_another_revision(stale_body_save, tmp_path):
    store, path, run_rel, base_backup, _body = stale_body_save
    wrong = tmp_path / "current-run"
    shutil.copytree(path / run_rel, wrong)  # already at the newer revision
    assert any("revision" in r for r in rebase_plan(store.path, "original-mask", wrong)["refusals"])
    store.create_session(task_id="original-mask", user="reviewer")
    assert any("open editor session" in r for r in rebase_plan(store.path, "original-mask", base_backup)["refusals"])


def test_receipt_chain_proves_the_stale_body_save_and_it_then_applies(stale_body_save):
    store, path, run_rel, _base_backup, body = stale_body_save
    plan = rebase_plan(store.path, "original-mask", None)
    assert plan["ok"], plan["refusals"]
    assert plan["proof"] == PROOF_RECEIPT_CHAIN and plan["base_backup"] is None and len(plan["stale_rows"]) == 1
    assert [(r["task_id"], r["component_name"]) for r in plan["receipt_chain"]] == [("eyes-task", "eyes_union")]

    report = rebase_stale_saves(store.path, "original-mask", None, actor="operator", skip_backup=True)
    assert report["outcome"] == {"ok": True, "rebased_rows": 1, "row_count": 1, "failures": []}
    audit = json.loads(store.conn.execute(
        "SELECT before_json FROM labeling_task_events WHERE event_type = ? AND user = 'operator';", (REBASE_EVENT,)
    ).fetchone()[0])
    assert audit["proof"] == PROOF_RECEIPT_CHAIN and audit["receipt_chain"] == plan["receipt_chain"]
    applied = apply_on_behalf(store.path, "original-mask", actor="operator", skip_backup=True)["outcome"]
    assert applied["ok"] and applied["applied_checkpoint_count"] == 1, applied
    run = zarr.open_group(str(path), mode="r", use_consolidated=False)[run_rel]
    np.testing.assert_array_equal(np.asarray(run["masks_roi"][1, 0]), (body > 0).astype(np.uint8))


def _refusals_after(store, path, run_rel, change):
    change(store, zarr.open_group(str(path), mode="a", use_consolidated=False)[run_rel])
    plan = rebase_plan(store.path, "original-mask", None)
    assert not plan["ok"]
    with pytest.raises(RebaseRefused):
        rebase_stale_saves(store.path, "original-mask", None, actor="operator", skip_backup=True)
    assert store.conn.execute(
        "SELECT target_edit_revision FROM labeling_session_checkpoints WHERE task_id='original-mask';"
    ).fetchone()[0] == plan["base_revision"]
    return " | ".join(plan["refusals"])


def test_receipt_chain_refuses_a_revision_step_this_store_did_not_apply(stale_body_save):
    store, path, run_rel, _b, _m = stale_body_save
    bump = lambda _s, run: run.attrs.update(edit_revision=int(run.attrs["edit_revision"]) + 1)  # noqa: E731
    assert "0 Apply receipt(s)" in _refusals_after(store, path, run_rel, bump)


def test_receipt_chain_refuses_when_the_run_names_another_last_apply(stale_body_save):
    store, path, run_rel, _b, _m = stale_body_save
    other = lambda _s, run: run.attrs.update(edit_revision_last_apply_id="written-elsewhere")  # noqa: E731
    assert "last Apply" in _refusals_after(store, path, run_rel, other)


def test_receipt_chain_refuses_a_step_that_wrote_this_component(stale_body_save):
    store, path, run_rel, _b, _m = stale_body_save

    def same_component(s, _run):
        s.conn.execute("UPDATE labeling_checkpoint_apply_receipts SET component_name = 'subject_body' WHERE task_id = 'eyes-task';")
        s.conn.commit()

    assert "wrote subject_body itself" in _refusals_after(store, path, run_rel, same_component)


def test_cli_requires_exactly_one_proof(stale_body_save, tmp_path):
    store = stale_body_save[0]
    with pytest.raises(SystemExit):
        main(["--store", str(store.path), "--task-id", "original-mask", "--actor", "operator"])
    with pytest.raises(SystemExit):
        main(["--store", str(store.path), "--task-id", "original-mask", "--actor", "operator",
              "--receipt-chain", "--base-backup", str(tmp_path)])
    assert main(["--store", str(store.path), "--task-id", "original-mask", "--actor", "operator", "--receipt-chain"]) == 0
