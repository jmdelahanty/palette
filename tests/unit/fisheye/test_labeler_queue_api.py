"""GET /api/me/queue: the lean labeler queue (schema palette.labeler_queue.v1).

It is a projection of the same work summary as /api/me/datasets, so access
checks and the Start decision must match the full payload exactly.
"""

from __future__ import annotations

import json

from fisheye.labeling.assignment_store import LabelingStore
from fisheye.labeling.web_labeler_queue import LABELER_QUEUE_SCHEMA

from .test_labeling_web_routes import (
    _add_active_labeling_users,
    _assert_browser_response_security_headers,
    _headers_request,
    _json_request,
    _running_server,
)

TOP_KEYS = {"ok", "schema", "user", "expected_user", "include_completed", "progress",
            "labeler", "blockers", "links", "datasets"}
DATASET_KEYS = {"dataset_id", "label", "open_task_count", "task_count", "recording_count",
                "workflow_counts", "work_url", "recordings"}
RECORDING_KEYS = {"recording_id", "open_task_count", "task_count", "blocked_reason", "work_url", "tasks"}
TASK_KEYS = {"task_id", "title", "workflow_kind", "component_name", "state", "priority",
             "notes", "work_url", "start", "progress"}
START_KEYS = {"ready", "endpoint", "method", "not_ready_reason", "operator_action"}
PROGRESS_KEYS = {"row_total", "saved_row_count", "applied_row_count", "unapplied_row_count",
                 "carried_row_count"}


def _store(tmp_path) -> LabelingStore:
    store = LabelingStore(tmp_path / "labeling_work.sqlite")
    store.initialize()
    _add_active_labeling_users(store, "alice")
    store.assign_recording(recording_id="rec-a", assignee_user="alice", assigned_by="admin")
    store.upsert_task(recording_id="rec-a", task_id="task-open", workflow_kind="keypoints",
                      title="Fix fins", priority=5, notes="Rows 3-9",
                      scope={"zarr_path": "/nowhere.zarr"})
    store.upsert_task(recording_id="rec-a", task_id="task-blocked", workflow_kind="subject_mask_component",
                      component_name="subject_body", state="blocked",
                      scope={"zarr_path": "/nowhere.zarr"})
    return store


def _tasks(payload, *, full: bool):
    datasets = payload["dataset_queue"] if full else payload["datasets"]
    return {t["task_id"]: t for d in datasets for r in d["recordings"] for t in r["tasks"]}


def test_queue_payload_shape_is_pinned(tmp_path):
    store = _store(tmp_path)
    try:
        with _running_server(store, user="alice") as base_url:
            status, payload = _json_request(base_url, "/api/me/queue?expected_user=alice")
            header_status, headers = _headers_request(base_url, "/api/me/queue")
        assert status == 200 and header_status == 200
        _assert_browser_response_security_headers(headers)
        assert set(payload) == TOP_KEYS
        assert payload["schema"] == LABELER_QUEUE_SCHEMA
        assert payload["user"] == "alice"
        for dataset in payload["datasets"]:
            assert set(dataset) == DATASET_KEYS
            for recording in dataset["recordings"]:
                assert set(recording) == RECORDING_KEYS
                for task in recording["tasks"]:
                    assert set(task) == TASK_KEYS
                    assert set(task["start"]) == START_KEYS
                    assert set(task["progress"]) == PROGRESS_KEYS
        assert payload["links"]["diagnostics"] == "/api/me/datasets?expected_user=alice"
    finally:
        store.close()


def test_queue_matches_the_full_payload_and_is_much_smaller(tmp_path):
    store = _store(tmp_path)
    try:
        with _running_server(store, user="alice") as base_url:
            _, lean = _json_request(base_url, "/api/me/queue?expected_user=alice")
            _, full = _json_request(base_url, "/api/me/datasets?expected_user=alice")
        lean_tasks, full_tasks = _tasks(lean, full=False), _tasks(full, full=True)
        assert set(lean_tasks) == set(full_tasks) == {"task-open", "task-blocked"}
        states = set(full["dataset_queue_direct_start_policy"]["startable_task_states"])
        for task_id, task in full_tasks.items():
            expected = (
                task["direct_browser_start_authorization_contract_ready"] in (True, "true")
                and bool(task["labeler_start_ready"])
                and task["state"] in states
                and bool(task["direct_browser_start_endpoint"])
            )
            assert lean_tasks[task_id]["start"]["ready"] is expected, task_id
        assert lean_tasks["task-open"]["start"]["ready"] is True
        assert lean_tasks["task-open"]["start"]["endpoint"] == "/api/tasks/task-open/open"
        assert lean_tasks["task-open"]["notes"] == "Rows 3-9" and lean_tasks["task-open"]["priority"] == 5
        assert lean_tasks["task-blocked"]["start"]["ready"] is False
        assert lean_tasks["task-blocked"]["start"]["endpoint"] == ""
        assert lean["progress"]["open_task_count"] == full["dataset_queue_state"]["counts"]["open_task_count"]
        assert len(json.dumps(lean)) * 10 < len(json.dumps(full))
    finally:
        store.close()


def test_queue_keeps_the_same_access_checks(tmp_path):
    store = _store(tmp_path)
    try:
        with _running_server(store, user="alice") as base_url:
            mismatch_status, mismatch = _json_request(base_url, "/api/me/queue?expected_user=bob")
            full_status, full = _json_request(base_url, "/api/me/datasets?expected_user=bob")
        assert mismatch_status == full_status == 403
        assert mismatch["error"] == full["error"] == "dashboard_user_mismatch"
        with _running_server(store, user="mallory") as base_url:
            unknown_status, unknown = _json_request(base_url, "/api/me/queue")
            full_unknown_status, full_unknown = _json_request(base_url, "/api/me/datasets")
        assert unknown_status == full_unknown_status == 403
        assert unknown["error"] == full_unknown["error"]
        with _running_server(store, user=None) as base_url:
            anon_status, anon = _json_request(base_url, "/api/me/queue")
        assert anon_status == 401 and anon["error"] == "authentication_required"
    finally:
        store.close()


def test_blocking_operator_validation_gate_is_reported_and_blocks_start(tmp_path, monkeypatch):
    from fisheye.labeling.web_labeler_queue import labeler_queue_payload

    work = {
        "expected_user": "alice",
        "dataset_queue_direct_start_policy": {"startable_task_states": ["pending"]},
        "operator_validation_start_gate": {"blocks_task_open": True, "not_ready_reason": "evidence_missing",
                                           "operator_action": "Record launch evidence."},
        "dataset_queue": [{"dataset_id": "d", "recordings": [{"recording_id": "r", "tasks": [{
            "task_id": "t", "state": "pending", "labeler_start_ready": True,
            "direct_browser_start_authorization_contract_ready": True,
            "direct_browser_start_endpoint": "/api/tasks/t/open"}]}]}],
    }
    payload = labeler_queue_payload(work, user="alice")
    start = payload["datasets"][0]["recordings"][0]["tasks"][0]["start"]
    assert start == {"ready": False, "endpoint": "", "method": "POST",
                     "not_ready_reason": "evidence_missing", "operator_action": "Record launch evidence."}
    assert payload["blockers"] == [{"code": "evidence_missing", "message": "Record launch evidence."}]


def test_queue_reports_store_row_progress_per_task(tmp_path):
    store = _store(tmp_path)
    store.upsert_task(recording_id="rec-a", task_id="task-open", workflow_kind="keypoints",
                      title="Fix fins", priority=5, notes="Rows 3-9",
                      scope={"zarr_path": "/nowhere.zarr", "target_roi_indices": [3, 4, 5]})
    session = store.create_session(task_id="task-open", user="alice").session_id
    store.upsert_session_checkpoint(
        session_id=session, task_id="task-open", recording_id="rec-a", user="alice",
        workflow_kind="keypoints", target_run_path="refined_keypoints_runs/r",
        target_edit_revision=0, source_rowset_path=None, roi_idx=4,
        component_name="keypoints", payload={},
    )
    try:
        with _running_server(store, user="alice") as base_url:
            status, payload = _json_request(base_url, "/api/me/queue?expected_user=alice")
        assert status == 200
        tasks = _tasks(payload, full=False)
        assert tasks["task-open"]["progress"] == {
            "row_total": 3, "saved_row_count": 1, "applied_row_count": 0,
            "unapplied_row_count": 1, "carried_row_count": 0,
        }
        assert tasks["task-blocked"]["progress"]["row_total"] is None
        assert tasks["task-blocked"]["progress"]["saved_row_count"] == 0
    finally:
        store.close()
