"""Background subject-mask Apply effects (``serve --background-apply-effects``).

Real HTTP Apply, real run lock, real QC refresh, and runtimes rebuilt from the
store by the same builder browser sessions use.  Flag-off behavior is pinned
by the existing route tests.
"""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from http.server import ThreadingHTTPServer

import numpy as np
import pytest

from fisheye.labeling import web as labeling_web
from fisheye.labeling import web_subject_mask_apply_effects as effects_mod
from fisheye.labeling import web_subject_mask_apply_effects_worker as worker_mod
from fisheye.labeling.assignment_store import LabelingStore

RUN = "refined_subject_masks_001"


def _request(base_url, path, payload=None, *, timeout=60):
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"} if payload is not None else {}
    req = urllib.request.Request(
        f"{base_url}{path}", data=body, headers=headers, method="GET" if payload is None else "POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))


@contextmanager
def _server(store, *, background=True, wait_seconds=300.0, admin_users=(), start_worker=False):
    config = labeling_web.ServerConfig(
        store_path=store.path, host="127.0.0.1", port=0, fixed_user="alice",
        auth_header="X-Forwarded-User", session_ttl_seconds=600, admin_users=tuple(admin_users),
        background_apply_effects=background, background_apply_wait_seconds=wait_seconds,
    )
    state = labeling_web.ServerState(store=store, config=config)
    if start_worker:
        _start_worker(state)
    server = ThreadingHTTPServer(("127.0.0.1", 0), labeling_web._make_handler(state))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        yield f"http://{host}:{port}", state
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        if state.apply_effects_worker is not None:
            state.apply_effects_worker.stop(timeout=30)


def _start_worker(state, **overrides):
    worker = worker_mod.start_worker_for_state(
        state, refresh_registry=lambda **kwargs: labeling_web._refresh_registry_for_scope(**kwargs),
    )
    for key, value in overrides.items():
        setattr(worker, key, value)
    return worker


def _wait_until(predicate, timeout=30.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return predicate()


@pytest.fixture
def mask_store(tmp_path, monkeypatch):
    from fisheye.tune import refined_subject_mask_review as review_mod
    from tests.unit.fisheye.test_refined_subject_mask_review import _build_subject_review_root

    monkeypatch.setenv("PALETTE_LABELING_NOTIFICATION_MODE", "outbox")
    monkeypatch.setenv("PALETTE_LABELING_NOTIFICATION_OUTBOX", str(tmp_path / "outbox"))
    zarr_path = tmp_path / "subject.zarr"
    root = _build_subject_review_root(zarr_path=zarr_path)
    review_mod.prepare_refined_subject_run(
        root, subject_run="subject_masks_001", refined_run=RUN,
        components=("subject_body", "swim_bladder"),
    )
    store = LabelingStore(tmp_path / "labeling_work.sqlite")
    store.initialize()
    store.assign_recording(recording_id="rec-a", assignee_user="alice")
    store.upsert_labeling_user(user_id="admin", email="admin@example.org", role="admin")
    store.upsert_task(
        task_id="task-a", recording_id="rec-a", workflow_kind="subject_mask_component",
        component_name="subject_body", run_name=RUN,
        scope={"zarr_path": str(zarr_path), "subject_run": "subject_masks_001", "refined_run": RUN,
               "component_name": "subject_body"},
    )
    lease = store.create_session(task_id="task-a", user="alice", ttl_seconds=600)
    try:
        yield store, lease, zarr_path
    finally:
        store.close()


def _route(lease, suffix):
    return f"/api/sessions/{lease.session_id}/subject-mask{suffix}"


def _save_row(base, lease, position, mask):
    status, nav = _request(base, _route(lease, "/nav"), {"position": position})
    assert status == 200, nav
    status, saved = _request(base, _route(lease, "/save"), {
        "mask": labeling_web._raw_array_payload(mask), "target_token": nav["state"]["target_token"],
    })
    assert status == 200, saved
    return saved["state"]["target_token"]


def _apply(base, lease, apply_id, token, *, timeout=60):
    return _request(base, _route(lease, "/apply"), {"apply_id": apply_id, "target_token": token}, timeout=timeout)


def _run_attrs(zarr_path):
    from fisheye.tune import refined_subject_mask_review as review_mod

    return review_mod.open_zarr_root(zarr_path, mode="r", use_consolidated=False)[
        f"refined_subject_masks_runs/{RUN}"
    ]


def _edited(value=1):
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[1:1 + 3 * value, 2:7] = 1
    return mask


def test_apply_returns_before_effects_and_worker_completes_them(mask_store):
    store, lease, zarr_path = mask_store
    with _server(store) as (base, state):
        status, reviewed = _request(base, _route(lease, "/review-status"), {
            "state": "needs_review", "target_token": _request(base, _route(lease, "/state"))[1]["state"]["target_token"],
        })
        assert status == 200, reviewed
        token = _save_row(base, lease, 0, _edited())
        status, applied = _apply(base, lease, "bg-1", token)
        assert status == 200, applied
        assert applied["result"]["effects"] == "queued"
        assert applied["result"]["applied_checkpoint_count"] == 1
        assert applied["state"]["pending_apply_effect_count"] == 1
        assert applied["state"]["apply_effects_background"] is True
        assert applied["state"]["apply_effects_status"]["state"] == "queued"
        run = _run_attrs(zarr_path)
        assert int(run.attrs["edit_revision"]) == 1
        assert bool(run.attrs["metrics_stale"]) is True  # QC has not run yet
        np.testing.assert_array_equal(run["masks_roi"][0, 0] > 0, _edited() > 0)

        # Approval stays gated; completion is allowed and records owed effects.
        status, blocked = _request(base, _route(lease, "/review-status"), {
            "state": "approved", "target_token": applied["state"]["target_token"],
        })
        assert status == 409 and blocked["error"] == "pending_apply_effects", blocked

        _start_worker(state)
        assert _wait_until(lambda: store.count_pending_session_checkpoint_apply_effects(task_id="task-a") == 0)
        status, current = _request(base, _route(lease, "/state"))
        assert status == 200, current
        assert current["state"]["pending_apply_effect_count"] == 0
        assert current["state"]["apply_effects_status"] is None
        assert current["state"]["qc_status"] == "complete"  # live runtime reopened by the worker
    run = _run_attrs(zarr_path)
    assert bool(run.attrs["metrics_stale"]) is False
    assert int(run.attrs["edit_revision"]) == 1
    attempts = store.list_events(task_id="task-a", event_type=effects_mod.ATTEMPT_EVENT)
    assert sorted(event["after"]["status"] for event in attempts) == ["complete", "running"]
    assert store.list_events(task_id="task-a", event_type=worker_mod.BACKGROUND_EFFECTS_EVENT)


def test_completion_allowed_while_effects_pending_with_flag_on(mask_store):
    store, lease, _zarr_path = mask_store
    with _server(store) as (base, _state):
        state_token = _request(base, _route(lease, "/state"))[1]["state"]["target_token"]
        status, _ = _request(base, _route(lease, "/review-status"), {"state": "needs_review", "target_token": state_token})
        assert status == 200
        token = _save_row(base, lease, 0, _edited())
        status, applied = _apply(base, lease, "bg-complete", token)
        assert status == 200, applied
        assert store.count_pending_session_checkpoint_apply_effects(task_id="task-a") == 1
        status, completed = _request(base, "/api/tasks/task-a/complete", {
            "session_id": lease.session_id, "expected_user": "alice",
        })
        assert status == 200, completed
    event = store.list_events(task_id="task-a", event_type="task_completed")[0]
    assert event["after"]["apply_effects_pending"] is True
    assert event["after"]["pending_apply_effect_count"] == 1
    assert store.count_pending_session_checkpoint_apply_effects(task_id="task-a") == 1


def test_restart_completes_pending_receipt_on_startup(mask_store):
    store, lease, zarr_path = mask_store
    with _server(store) as (base, _state):  # flag on, but no worker: a crash before effects
        token = _save_row(base, lease, 0, _edited())
        status, applied = _apply(base, lease, "bg-restart", token)
        assert status == 200, applied
    assert store.count_pending_session_checkpoint_apply_effects(task_id="task-a") == 1
    with _server(store, start_worker=True):
        assert _wait_until(lambda: store.count_pending_session_checkpoint_apply_effects(task_id="task-a") == 0)
    assert bool(_run_attrs(zarr_path).attrs["metrics_stale"]) is False


def test_second_apply_waits_for_prior_effects_then_processes_in_order(mask_store):
    store, lease, zarr_path = mask_store
    with _server(store, wait_seconds=0.2) as (base, state):
        token = _save_row(base, lease, 0, _edited(1))
        status, first = _apply(base, lease, "bg-first", token)
        assert status == 200, first
        token = _save_row(base, lease, 1, _edited(2))
        status, refused = _apply(base, lease, "bg-second", token)
        assert status == 409, refused
        assert refused["error"] == "previous_update_still_running"
        assert store.count_unapplied_session_checkpoints(task_id="task-a", component_name="subject_body") == 1

    with _server(store, wait_seconds=60.0) as (base, state):
        token = _request(base, _route(lease, "/state"))[1]["state"]["target_token"]
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(_apply, base, lease, "bg-second", token)
            time.sleep(0.5)
            assert not future.done()  # waiting, not refused
            _start_worker(state)
            status, second = future.result(timeout=60)
        assert status == 200, second
        assert second["result"]["edit_revision_before"] == 1
        assert second["result"]["edit_revision_after"] == 2
        assert _wait_until(lambda: store.count_pending_session_checkpoint_apply_effects(task_id="task-a") == 0)
    completed = [
        event["target"]["apply_id"]
        for event in reversed(store.list_events(task_id="task-a", event_type=worker_mod.BACKGROUND_EFFECTS_EVENT))
    ]
    assert completed == ["bg-first", "bg-second"]
    assert int(_run_attrs(zarr_path).attrs["edit_revision"]) == 2


def test_transient_failure_retries_with_backoff_and_is_surfaced(mask_store, monkeypatch, tmp_path):
    store, lease, zarr_path = mask_store
    calls = []

    def flaky_registry(**kwargs):
        calls.append(kwargs)
        return len(calls) > 1

    monkeypatch.setattr(labeling_web, "_refresh_registry_for_scope", flaky_registry)
    with _server(store, admin_users=("admin",)) as (base, state):
        token = _save_row(base, lease, 0, _edited())
        assert _apply(base, lease, "bg-transient", token)[0] == 200
        worker = worker_mod.ApplyEffectsWorker(
            store, refresh_registry=lambda **kwargs: labeling_web._refresh_registry_for_scope(**kwargs),
            notify=lambda receipt, after: worker_mod.notify_admins_of_failure(store, ("admin",), receipt, after),
            backoff_seconds=(30.0, 120.0),
        )
        state.apply_effects_worker = worker
        worker.run_once()
        status, current = _request(base, _route(lease, "/state"))
        assert current["state"]["apply_effects_status"]["state"] == "retrying"
        assert "registry refresh remains pending" in current["state"]["apply_effects_status"]["reason"]
        assert current["state"]["apply_effects_status"]["next_attempt_at_utc"]
        worker.run_once()  # not yet due: no second attempt
        assert len(calls) == 1
        worker.wake(force_retry=True)  # e.g. a new Apply is waiting on this run
        worker.run_once()
        assert len(calls) == 2
    assert store.count_pending_session_checkpoint_apply_effects(task_id="task-a") == 0
    attempts = [event["after"] for event in store.list_events(task_id="task-a", event_type=effects_mod.ATTEMPT_EVENT)]
    failed = [after for after in attempts if after["status"] == "failed"]
    assert len(failed) == 1 and failed[0]["failure_kind"] == "transient"
    notices = store.list_events(task_id="task-a", event_type="apply_effects_failure_notification_queued")
    assert len(notices) == 1 and notices[0]["after"]["to_email"] == "admin@example.org"
    assert len(list((tmp_path / "outbox").glob("*.eml"))) == 1


def test_refusal_is_not_retried_and_blocks_next_apply_until_manual_retry(mask_store, monkeypatch):
    from fisheye.labeling import web_subject_mask_apply_qc as qc_mod

    store, lease, zarr_path = mask_store
    original = qc_mod.refresh_subject_mask_apply_qc_locked
    attempts = []

    def refuse(**kwargs):
        attempts.append(kwargs)
        raise RuntimeError("Browser Apply QC cannot change declared mask_labels: ['x'].")

    monkeypatch.setattr(qc_mod, "refresh_subject_mask_apply_qc_locked", refuse)
    with _server(store, admin_users=("admin",)) as (base, state):
        token = _save_row(base, lease, 0, _edited())
        assert _apply(base, lease, "bg-refused", token)[0] == 200
        worker = worker_mod.ApplyEffectsWorker(
            store, refresh_registry=lambda **kwargs: labeling_web._refresh_registry_for_scope(**kwargs),
        )
        state.apply_effects_worker = worker
        worker.run_once()
        worker.wake(force_retry=True)
        worker.run_once()
        assert len(attempts) == 1  # refusals are never retried automatically
        status, current = _request(base, _route(lease, "/state"))
        assert current["state"]["apply_effects_status"]["state"] == "failed"
        assert "cannot change declared mask_labels" in current["state"]["apply_effects_status"]["reason"]

        token = _save_row(base, lease, 1, _edited(2))
        status, blocked = _apply(base, lease, "bg-next", token)
        assert status == 409 and blocked["error"] == "previous_update_failed", blocked

        # A person retrying the refused apply_id runs today's inline path.
        monkeypatch.setattr(qc_mod, "refresh_subject_mask_apply_qc_locked", original)
        status, retried = _apply(base, lease, "bg-refused", token)
        assert status == 200, retried
        assert retried["result"]["already_applied"] is True
        assert retried["result"]["qc_status"] == "complete"
    assert store.count_pending_session_checkpoint_apply_effects(task_id="task-a") == 0


def test_retry_of_queued_apply_id_defers_to_worker(mask_store):
    store, lease, _zarr_path = mask_store
    with _server(store) as (base, _state):
        token = _save_row(base, lease, 0, _edited())
        assert _apply(base, lease, "bg-queued", token)[0] == 200
        status, retried = _apply(base, lease, "bg-queued", token)
        assert status == 200, retried
        assert retried["result"]["already_applied"] is True
        assert retried["result"]["effects"] == "queued"
    assert store.count_pending_session_checkpoint_apply_effects(task_id="task-a") == 1


def test_serve_flag_defaults_off():
    parser = labeling_web.build_parser()
    assert parser.parse_args(["serve"]).background_apply_effects is False
    assert parser.parse_args(["serve", "--background-apply-effects"]).background_apply_effects is True


def test_failure_classification():
    assert worker_mod.classify_failure(TimeoutError("lock")) == "transient"
    assert worker_mod.classify_failure(OSError("nfs")) == "transient"
    assert worker_mod.classify_failure(RuntimeError("Subject-mask registry refresh remains pending.")) == "transient"
    assert worker_mod.classify_failure(ValueError("identity mismatch")) == "refused"
    assert worker_mod.classify_failure(RuntimeError("Tail successor receipt has a conflicting source binding")) == "refused"
    assert worker_mod.classify_failure(worker_mod.ApplyEffectsRefused("x")) == "refused"


def _component_review_state(zarr_path, component="subject_body"):
    reviews = _run_attrs(zarr_path).attrs.get("component_review_statuses") or {}
    entry = reviews.get(component) if isinstance(reviews, dict) else None
    return entry.get("state") if isinstance(entry, dict) else None


def test_needs_review_is_deferred_and_task_completes_without_waiting(mask_store):
    store, lease, zarr_path = mask_store
    with _server(store) as (base, state):
        token = _save_row(base, lease, 0, _edited())
        status, applied = _apply(base, lease, "bg-deferred", token)
        assert status == 200 and applied["result"]["effects"] == "queued", applied
        token = applied["state"]["target_token"]
        assert applied["state"]["component_review_completion_ready"] is False

        # Approval still waits for the owed effects.
        status, blocked = _request(base, _route(lease, "/review-status"), {"state": "approved", "target_token": token})
        assert status == 409 and blocked["error"] == "pending_apply_effects", blocked
        assert "set needs_review now" in blocked["details"]
        assert "Retry the saved" not in blocked["details"]

        # needs_review is recorded now and written later, without taking the run lock.
        status, deferred = _request(base, _route(lease, "/review-status"), {"state": "needs_review", "target_token": token})
        assert status == 202 and deferred["deferred"] is True, deferred
        assert deferred["state"]["component_review_completion_guard"]["review_state_deferred"] is True
        assert deferred["state"]["component_review_completion_ready"] is True
        assert _component_review_state(zarr_path) != "needs_review"

        status, completed = _request(base, "/api/tasks/task-a/complete", {
            "session_id": lease.session_id, "expected_user": "alice",
        })
        assert status == 200, completed

        _start_worker(state)
        assert _wait_until(lambda: store.count_pending_session_checkpoint_apply_effects(task_id="task-a") == 0)
    assert _component_review_state(zarr_path) == "needs_review"
    written = store.list_events(task_id="task-a", event_type="set_review_status")
    assert written and written[0]["user"] == "alice"


def test_failed_deferred_review_write_keeps_effects_owed(mask_store, monkeypatch):
    from fisheye.tune import refined_subject_mask_review as review_mod

    store, lease, zarr_path = mask_store
    real_write = review_mod.apply_component_review_status
    fail_once = {"pending": True}

    def flaky_write(*args, **kwargs):
        if fail_once["pending"]:
            fail_once["pending"] = False
            raise OSError("injected review write failure")
        return real_write(*args, **kwargs)

    monkeypatch.setattr(review_mod, "apply_component_review_status", flaky_write)
    with _server(store) as (base, state):
        token = _save_row(base, lease, 0, _edited())
        status, applied = _apply(base, lease, "bg-deferred-fail", token)
        assert status == 200, applied
        status, _ = _request(base, _route(lease, "/review-status"), {
            "state": "needs_review", "target_token": applied["state"]["target_token"],
        })
        assert status == 202
        worker = worker_mod.ApplyEffectsWorker(
            store=store, refresh_registry=lambda **kwargs: labeling_web._refresh_registry_for_scope(**kwargs),
            backoff_seconds=(0.0,),
        )
        state.apply_effects_worker = worker
        worker.run_once()
        # The status write failed before the receipt was completed: effects stay owed.
        assert store.count_pending_session_checkpoint_apply_effects(task_id="task-a") == 1
        worker.wake(force_retry=True)
        worker.run_once()
    assert store.count_pending_session_checkpoint_apply_effects(task_id="task-a") == 0
    assert _component_review_state(zarr_path) == "needs_review"
