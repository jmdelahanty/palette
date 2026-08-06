from __future__ import annotations

import json
import sys
import threading
import urllib.error
import urllib.request
from contextlib import contextmanager
from http.server import ThreadingHTTPServer
from types import ModuleType, SimpleNamespace

import pytest

from fisheye.labeling import web as labeling_web
from fisheye.labeling.assignment_store import LabelingStore


@contextmanager
def _running_server(
    store: LabelingStore,
    *,
    user: str,
    admin_users: tuple[str, ...] = (),
):
    config = labeling_web.ServerConfig(
        store_path=store.path,
        host="127.0.0.1",
        port=0,
        fixed_user=user,
        auth_header="X-Forwarded-User",
        session_ttl_seconds=600,
        admin_users=admin_users,
    )
    state = labeling_web.ServerState(store=store, config=config)
    server = ThreadingHTTPServer(("127.0.0.1", 0), labeling_web._make_handler(state))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        yield f"http://{host}:{port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _json_request(
    base_url: str,
    path: str,
    *,
    method: str = "GET",
    payload: dict[str, object] | None = None,
) -> tuple[int, dict[str, object]]:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"} if payload is not None else {}
    request = urllib.request.Request(
        f"{base_url}{path}",
        data=body,
        headers=headers,
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))


def _text_request(base_url: str, path: str) -> tuple[int, str]:
    request = urllib.request.Request(f"{base_url}{path}", method="GET")
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            return response.status, response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8")


def _fake_module(monkeypatch, name: str, **attrs: object) -> ModuleType:
    module = ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    monkeypatch.setitem(sys.modules, name, module)
    parent_name, _, child_name = name.rpartition(".")
    parent = sys.modules.get(parent_name)
    if parent is not None and child_name:
        monkeypatch.setattr(parent, child_name, module, raising=False)
    return module


def _completed_keypoint_task(store: LabelingStore) -> None:
    store.initialize()
    store.upsert_labeling_user(user_id="alice", status="active")
    store.assign_recording(recording_id="rec-a", assignee_user="alice")
    store.upsert_task(
        task_id="task-a",
        recording_id="rec-a",
        workflow_kind="keypoints",
        title="Review these keypoints",
        state="complete",
        scope={
            "zarr_path": "/server-owned/task.zarr",
            "refined_run": "refined-a",
            "crop_run": "crop-a",
        },
    )


def test_admin_review_state_is_durable_audited_and_idempotent(tmp_path):
    store = LabelingStore(tmp_path / "labeling_work.sqlite")
    try:
        _completed_keypoint_task(store)
        store.record_event(
            task_id="task-a",
            recording_id="rec-a",
            user="admin",
            event_type="admin_save_keypoint_correction",
        )

        review = store.upsert_admin_review(
            task_id="task-a",
            state="accepted_with_corrections",
            reviewer_user="admin",
            notes="Corrected one point.",
            metadata={"source": "unit_test"},
        )
        repeated = store.upsert_admin_review(
            task_id="task-a",
            state="accepted_with_corrections",
            reviewer_user="admin",
            notes="Corrected one point.",
            metadata={"source": "unit_test"},
        )

        assert repeated["review_id"] == review["review_id"]
        assert repeated["state"] == "accepted_with_corrections"
        assert repeated["reviewer_user"] == "admin"
        assert repeated["correction_event_count"] == 1
        assert repeated["metadata"] == {"source": "unit_test"}
        assert len(
            store.list_events(
                task_id="task-a",
                event_type="admin_review_state_changed",
            )
        ) == 1
    finally:
        store.close()


def test_admin_review_decisions_require_completed_tasks(tmp_path):
    store = LabelingStore(tmp_path / "labeling_work.sqlite")
    try:
        store.initialize()
        store.upsert_labeling_user(user_id="alice", status="active")
        store.assign_recording(recording_id="rec-a", assignee_user="alice")
        store.upsert_task(
            task_id="task-a",
            recording_id="rec-a",
            workflow_kind="keypoints",
        )

        with pytest.raises(ValueError, match="completed task"):
            store.upsert_admin_review(
                task_id="task-a",
                state="accepted_as_is",
                reviewer_user="admin",
            )

        pending = store.upsert_admin_review(
            task_id="task-a",
            state="pending",
            reviewer_user="admin",
        )
        assert pending["state"] == "pending"
    finally:
        store.close()


def test_admin_completed_work_queue_and_review_decision(tmp_path):
    store = LabelingStore(tmp_path / "labeling_work.sqlite")
    try:
        _completed_keypoint_task(store)
        store.record_event(
            task_id="task-a",
            recording_id="rec-a",
            user="alice",
            event_type="save_keypoints",
            target={"roi_idx": 7, "frame_idx": 12},
            before={"roi_idx": 7, "points": [[1.0, 2.0]]},
            after={"readback": {"roi_idx": 7, "frame_idx": 12}},
        )

        with _running_server(
            store,
            user="admin",
            admin_users=("admin",),
        ) as base_url:
            pending_status, pending = _json_request(
                base_url,
                "/api/admin/completed-work",
            )
            page_status, page_html = _text_request(base_url, "/admin/completed-work")
            review_status, review = _json_request(
                base_url,
                "/api/admin/admin-reviews",
                method="POST",
                payload={
                    "task_id": "task-a",
                    "state": "accepted_as_is",
                    "notes": "Checked against the source frame.",
                },
            )
            pending_after_status, pending_after = _json_request(
                base_url,
                "/api/admin/completed-work",
            )
            accepted_status, accepted = _json_request(
                base_url,
                "/api/admin/completed-work?admin_review_state=accepted_as_is",
            )
            task_status, task_payload = _json_request(
                base_url,
                "/api/admin/tasks/task-a",
            )

        assert pending_status == 200
        assert pending["counts"]["completed_work_count"] == 1
        assert pending["completed_work"][0]["task_id"] == "task-a"
        assert pending["completed_work"][0]["save_event_count"] == 1
        assert page_status == 200
        assert "Completed work review" in page_html
        assert "task-a" in page_html
        assert review_status == 200
        assert review["admin_review"]["state"] == "accepted_as_is"
        assert review["admin_review"]["reviewer_user"] == "admin"
        assert pending_after_status == 200
        assert pending_after["completed_work"] == []
        assert accepted_status == 200
        assert accepted["completed_work"][0]["task_id"] == "task-a"
        assert task_status == 200
        assert task_payload["admin_review"]["state"] == "accepted_as_is"
        assert len(
            store.list_events(
                task_id="task-a",
                event_type="admin_review_state_changed",
            )
        ) == 1
    finally:
        store.close()


def test_admin_keypoint_correction_uses_task_scope_and_records_audit(
    tmp_path,
    monkeypatch,
):
    calls: list[tuple[str, object]] = []

    def resolve_review_session(zarr_path, **kwargs):
        calls.append(("resolve", {"zarr_path": zarr_path, **kwargs}))
        return SimpleNamespace(refined_run="refined-a", crop_run="crop-a")

    def load_roi_payload(_session, *, position):
        calls.append(("load", position))
        return {
            "roi_idx": 7,
            "frame_idx": 12,
            "labels": ["snout", "tail"],
            "points": [[1.0, 2.0], [3.0, 4.0]],
            "reason": "",
            "status": {"usable_keypoints": True},
        }

    def save_roi_correction(_session, *, position, points):
        calls.append(("save", {"position": position, "points": points}))
        return {
            "roi_idx": 7,
            "frame_idx": 12,
            "changed": True,
            "reason_updated": False,
            "readback": {
                "roi_idx": 7,
                "frame_idx": 12,
                "points": points,
                "status": {"usable_keypoints": True},
            },
        }

    _fake_module(
        monkeypatch,
        "fisheye.tune.keypoint_review_backend",
        resolve_review_session=resolve_review_session,
        load_roi_payload=load_roi_payload,
        save_roi_correction=save_roi_correction,
    )
    store = LabelingStore(tmp_path / "labeling_work.sqlite")
    try:
        _completed_keypoint_task(store)
        corrected_points = [[2.0, 3.0], [3.0, 4.0]]

        with _running_server(
            store,
            user="admin",
            admin_users=("admin",),
        ) as base_url:
            invalid_status, _invalid = _json_request(
                base_url,
                "/api/admin/keypoint-corrections",
                method="POST",
                payload={
                    "task_id": "task-a",
                    "roi_idx": 7,
                    "points": corrected_points,
                    "set_review_state": "not-a-review-state",
                },
            )
            assert calls == []
            status, payload = _json_request(
                base_url,
                "/api/admin/keypoint-corrections",
                method="POST",
                payload={
                    "task_id": "task-a",
                    "roi_idx": 7,
                    "points": corrected_points,
                    "notes": "Moved the snout.",
                },
            )

        assert invalid_status == 400
        assert status == 200
        assert calls[0] == (
            "resolve",
            {
                "zarr_path": "/server-owned/task.zarr",
                "refined_run": "refined-a",
                "crop_run": "crop-a",
                "include_all": True,
                "target_roi_indices": [7],
            },
        )
        assert calls[1] == ("load", 0)
        assert calls[2] == (
            "save",
            {"position": 0, "points": corrected_points},
        )
        assert payload["delta_count"] == 2
        assert payload["admin_review"]["state"] == "accepted_with_corrections"
        correction_events = store.list_events(
            task_id="task-a",
            event_type="admin_save_keypoint_correction",
        )
        assert len(correction_events) == 1
        assert correction_events[0]["user"] == "admin"
        assert correction_events[0]["target"]["roi_idx"] == 7
        assert store.count_admin_correction_events("task-a") == 1
    finally:
        store.close()
