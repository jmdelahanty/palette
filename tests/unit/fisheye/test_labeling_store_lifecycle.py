from __future__ import annotations

import base64
import hashlib
import json
import sqlite3
import threading
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Mapping

import pytest

from fisheye.labeling.assignment_store import (
    SCHEMA_VERSION,
    LabelingStore,
    session_checkpoint_snapshot_row_sha256,
)
from fisheye.labeling.web import ServerConfig, ServerState, _make_handler


def _active_file_connection_count(store: LabelingStore) -> int:
    with store._connection_lock:  # noqa: SLF001 - lifecycle regression assertion
        return len(store._connections)  # noqa: SLF001


def _wait_for_file_connections(store: LabelingStore, expected: int) -> None:
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        if _active_file_connection_count(store) == expected:
            return
        time.sleep(0.01)
    assert _active_file_connection_count(store) == expected


@contextmanager
def _running_server(store: LabelingStore, *, user: str):
    config = ServerConfig(
        store_path=store.path,
        host="127.0.0.1",
        port=0,
        fixed_user=user,
        auth_header="X-Forwarded-User",
        session_ttl_seconds=600,
        admin_users=(user,),
    )
    server = ThreadingHTTPServer(
        ("127.0.0.1", 0),
        _make_handler(ServerState(store=store, config=config)),
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        yield f"http://{host}:{port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _get_status(url: str) -> int:
    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            response.read()
            return int(response.status)
    except urllib.error.HTTPError as exc:
        exc.read()
        return int(exc.code)


def _checkpoint_store(tmp_path: Path) -> tuple[LabelingStore, object]:
    store = LabelingStore(tmp_path / "labeling.sqlite")
    store.assign_recording(recording_id="rec-a", assignee_user="alice")
    store.upsert_task(
        task_id="task-a",
        recording_id="rec-a",
        workflow_kind="keypoints",
    )
    lease = store.create_session(task_id="task-a", user="alice", ttl_seconds=600)
    return store, lease


def _checkpoint_kwargs(lease: object, *, roi_idx: int = 7, value: int = 1) -> dict[str, object]:
    return {
        "session_id": lease.session_id,
        "task_id": lease.task_id,
        "recording_id": lease.recording_id,
        "user": lease.user,
        "workflow_kind": "keypoints",
        "target_run_path": "/runs/refined-keypoints-a",
        "target_edit_revision": 3,
        "source_rowset_path": "/runs/source-keypoints-a",
        "roi_idx": roi_idx,
        "component_name": "keypoints",
        "payload": {"schema": "palette.keypoint_checkpoint.v1", "value": value},
        "metadata": {"row_identity": {"roi_idx": roi_idx}},
    }


def _snapshot_row_sha256(row: Mapping[str, object]) -> str:
    snapshot = {
        "schema": "palette.labeling_session_checkpoint_snapshot_row.v1",
        "checkpoint_id": str(row["checkpoint_id"]),
        "task_id": str(row["task_id"]),
        "recording_id": str(row["recording_id"]),
        "user": str(row["user"]),
        "workflow_kind": str(row["workflow_kind"]),
        "target_run_path": str(row["target_run_path"]),
        "target_edit_revision": int(row["target_edit_revision"] or 0),
        "source_rowset_path": str(row["source_rowset_path"] or ""),
        "roi_idx": int(row["roi_idx"] or 0),
        "component_name": str(row["component_name"]),
        "payload": row["payload"],
        "metadata": row["metadata"],
    }
    canonical = json.dumps(
        snapshot,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def test_initialize_bootstraps_once_and_assignment_still_creates_user(tmp_path, monkeypatch):
    store = LabelingStore(tmp_path / "labeling.sqlite")
    calls = 0
    original = store._initialize_schema  # noqa: SLF001

    def counted_initialize_schema(conn):
        nonlocal calls
        calls += 1
        return original(conn)

    monkeypatch.setattr(store, "_initialize_schema", counted_initialize_schema)
    try:
        store.assign_recording(recording_id="rec-a", assignee_user="alice")
        assignment = store.get_assignment("rec-a")
        user = store.get_labeling_user("alice")
        store.list_assignments()
        store.list_labeling_users()

        assert calls == 1
        assert assignment is not None
        assert user is not None
        assert user["status"] == "active"
        assert user["notes"] == "Auto-created from existing recording assignments."
    finally:
        store.close()


def test_initialize_is_safe_when_threads_first_open_store_together(tmp_path, monkeypatch):
    store = LabelingStore(tmp_path / "labeling.sqlite")
    calls = 0
    calls_lock = threading.Lock()
    original = store._initialize_schema  # noqa: SLF001

    def counted_initialize_schema(conn):
        nonlocal calls
        with calls_lock:
            calls += 1
        return original(conn)

    monkeypatch.setattr(store, "_initialize_schema", counted_initialize_schema)
    barrier = threading.Barrier(8)
    errors: list[BaseException] = []

    def worker() -> None:
        try:
            barrier.wait()
            assert store.list_assignments() == []
        except BaseException as exc:  # pragma: no cover - assertion reports collected failures
            errors.append(exc)
        finally:
            store.close_thread_connection()

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)
    try:
        assert not errors
        assert all(not thread.is_alive() for thread in threads)
        assert calls == 1
        assert _active_file_connection_count(store) == 0
    finally:
        store.close()


def test_initialize_upgrades_old_assignment_store_and_rejects_future_schema(tmp_path):
    old_path = tmp_path / "old.sqlite"
    with sqlite3.connect(old_path) as conn:
        conn.executescript(
            """
            CREATE TABLE labeling_schema_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            INSERT INTO labeling_schema_meta(key, value) VALUES ('schema_version', '1');
            CREATE TABLE recording_assignments (
                recording_id TEXT PRIMARY KEY,
                assignee_user TEXT NOT NULL,
                assigned_by TEXT,
                assigned_at_utc TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'active',
                notes TEXT
            );
            INSERT INTO recording_assignments(
                recording_id, assignee_user, assigned_at_utc, status
            ) VALUES ('rec-old', 'legacy-user', '2026-01-01T00:00:00+00:00', 'active');
            """
        )

    old_store = LabelingStore(old_path)
    try:
        old_store.initialize()
        assert old_store.get_labeling_user("legacy-user")["status"] == "active"
        version = old_store.conn.execute(
            "SELECT value FROM labeling_schema_meta WHERE key = 'schema_version'"
        ).fetchone()["value"]
        assert version == str(SCHEMA_VERSION)
    finally:
        old_store.close()

    future_path = tmp_path / "future.sqlite"
    with sqlite3.connect(future_path) as conn:
        conn.executescript(
            "CREATE TABLE labeling_schema_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);"
            f"INSERT INTO labeling_schema_meta(key, value) VALUES ('schema_version', '{SCHEMA_VERSION + 1}');"
        )
    future_store = LabelingStore(future_path)
    try:
        with pytest.raises(RuntimeError, match="newer than supported"):
            future_store.initialize()
        version = future_store.conn.execute(
            "SELECT value FROM labeling_schema_meta WHERE key = 'schema_version'"
        ).fetchone()["value"]
        assert version == str(SCHEMA_VERSION + 1)
    finally:
        future_store.close()


def test_close_thread_connection_is_bounded_and_does_not_close_other_threads(tmp_path):
    store = LabelingStore(tmp_path / "labeling.sqlite")
    store.initialize()
    store.close_thread_connection()
    barrier = threading.Barrier(12)
    errors: list[BaseException] = []

    def worker(index: int) -> None:
        try:
            barrier.wait()
            store.list_assignments()
            store.record_assignment_event(
                recording_id=f"rec-{index}",
                actor_user="test",
                event_type="connection_lifecycle_test",
            )
        except BaseException as exc:  # pragma: no cover - assertion reports collected failures
            errors.append(exc)
        finally:
            store.close_thread_connection()

    threads = [threading.Thread(target=worker, args=(index,)) for index in range(12)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)
    try:
        assert not errors
        assert all(not thread.is_alive() for thread in threads)
        assert _active_file_connection_count(store) == 0
        assert len(store.list_assignment_events(event_type="connection_lifecycle_test")) == 12
    finally:
        store.close()


def test_http_success_and_permission_failure_release_request_connections(tmp_path):
    store = LabelingStore(tmp_path / "labeling.sqlite")
    store.assign_recording(recording_id="rec-a", assignee_user="alice")
    store.upsert_task(
        task_id="task-a",
        recording_id="rec-a",
        workflow_kind="keypoints",
    )
    store.close_thread_connection()
    try:
        with _running_server(store, user="alice") as base_url:
            assert _get_status(f"{base_url}/api/me/datasets?expected_user=alice") == 200
            _wait_for_file_connections(store, 0)

            store.deactivate_labeling_user("alice", actor_user="operator")
            store.close_thread_connection()
            assert _get_status(f"{base_url}/api/me/datasets?expected_user=alice") == 403
            _wait_for_file_connections(store, 0)
    finally:
        store.close()


def test_memory_store_request_cleanup_keeps_shared_connection_until_store_close():
    store = LabelingStore(":memory:")
    store.initialize()
    connection = store.conn

    store.close_thread_connection()

    assert store.conn is connection
    assert store.list_assignments() == []
    store.close()
    with pytest.raises(RuntimeError, match="closed"):
        _ = store.conn


def test_hot_path_optimization_preserves_session_permission_checks(tmp_path):
    store = LabelingStore(tmp_path / "labeling.sqlite")
    try:
        store.assign_recording(recording_id="rec-a", assignee_user="alice")
        store.upsert_task(
            task_id="task-a",
            recording_id="rec-a",
            workflow_kind="keypoints",
        )
        with pytest.raises(PermissionError, match="not assigned"):
            store.create_session(task_id="task-a", user="bob", ttl_seconds=600)

        lease = store.create_session(task_id="task-a", user="alice", ttl_seconds=600)
        store.assign_recording(
            recording_id="rec-a",
            assignee_user="bob",
            allow_stale_open_sessions=True,
        )
        with pytest.raises(PermissionError, match="previous-owner sessions"):
            store.create_session(task_id="task-a", user="bob", ttl_seconds=600)
        assert store.get_session(lease.session_id) is not None
    finally:
        store.close()


def test_checkpoint_snapshot_descriptors_match_claim_and_same_row_save_changes_digest(tmp_path):
    store, lease = _checkpoint_store(tmp_path)
    try:
        later = store.upsert_session_checkpoint(
            **_checkpoint_kwargs(lease, roi_idx=7, value=1),
        )
        earlier = store.upsert_session_checkpoint(
            **_checkpoint_kwargs(lease, roi_idx=8, value=2),
        )
        store.conn.execute(
            "UPDATE labeling_session_checkpoints SET updated_at_utc = ? WHERE checkpoint_id = ?;",
            ("2026-09-19T12:00:02+00:00", later["checkpoint_id"]),
        )
        store.conn.execute(
            "UPDATE labeling_session_checkpoints SET updated_at_utc = ? WHERE checkpoint_id = ?;",
            ("2026-09-19T12:00:01+00:00", earlier["checkpoint_id"]),
        )
        store.conn.commit()

        before = store.list_session_checkpoint_snapshot_descriptors(
            task_id="task-a",
            state="active",
            component_name="keypoints",
        )
        assert [row["checkpoint_id"] for row in before] == [
            earlier["checkpoint_id"],
            later["checkpoint_id"],
        ]
        assert all(
            set(row)
            == {
                "checkpoint_id",
                "roi_idx",
                "updated_at_utc",
                "state",
                "apply_id",
                "snapshot_row_sha256",
            }
            for row in before
        )
        assert store.count_session_checkpoints(
            task_id="task-a",
            state="active",
            component_name="keypoints",
        ) == 2
        assert store.count_session_checkpoints(
            task_id="task-a",
            state="applying",
            component_name="keypoints",
        ) == 0

        changed = store.upsert_session_checkpoint(
            **{
                **_checkpoint_kwargs(lease, roi_idx=7, value=3),
                "payload": {
                    "schema": "palette.keypoint_checkpoint.v1",
                    "label": "café",
                    "nullable": None,
                    "value": 3,
                },
            }
        )
        assert changed["checkpoint_id"] == later["checkpoint_id"]
        assert changed["snapshot_row_sha256"] != later["snapshot_row_sha256"]
        assert changed["snapshot_row_sha256"] == _snapshot_row_sha256(changed)
        assert changed["snapshot_row_sha256"] == session_checkpoint_snapshot_row_sha256(
            changed
        )

        descriptors = store.list_session_checkpoint_snapshot_descriptors(
            task_id="task-a",
            state="active",
            component_name="keypoints",
        )
        with pytest.raises(ValueError, match="lowercase SHA-256"):
            store.claim_session_checkpoints_for_apply(
                task_id="task-a",
                component_name="keypoints",
                apply_id="invalid-digest-apply",
                checkpoint_snapshot_sha256="ABC",
            )
        claimed = store.claim_session_checkpoints_for_apply(
            task_id="task-a",
            component_name="keypoints",
            apply_id="descriptor-apply",
        )
        assert [
            (row["checkpoint_id"], row["snapshot_row_sha256"])
            for row in claimed
        ] == [
            (row["checkpoint_id"], row["snapshot_row_sha256"])
            for row in descriptors
        ]
        assert all(row["snapshot_row_sha256"] == _snapshot_row_sha256(row) for row in claimed)
        applying = store.list_session_checkpoint_snapshot_descriptors(
            task_id="task-a",
            state="applying",
            component_name="keypoints",
        )
        assert [row["checkpoint_id"] for row in applying] == [
            row["checkpoint_id"] for row in claimed
        ]
        assert store.count_session_checkpoints(
            task_id="task-a",
            state="active",
            component_name="keypoints",
        ) == 0
        assert store.count_session_checkpoints(
            task_id="task-a",
            state="applying",
            component_name="keypoints",
        ) == 2
    finally:
        store.close()


def test_checkpoint_snapshot_digest_preserves_generic_nonfinite_json_grammar(tmp_path):
    store, lease = _checkpoint_store(tmp_path)
    try:
        checkpoint = store.upsert_session_checkpoint(
            **{
                **_checkpoint_kwargs(lease),
                "payload": {
                    "schema": "palette.generic_checkpoint.v1",
                    "confidence": float("nan"),
                },
            }
        )
        assert checkpoint["payload"]["confidence"] != checkpoint["payload"]["confidence"]
        assert checkpoint["snapshot_row_sha256"] == session_checkpoint_snapshot_row_sha256(
            checkpoint
        )
        assert store.count_unapplied_session_checkpoints(task_id="task-a") == 1
    finally:
        store.close()


def test_dense_mask_checkpoint_digest_preserves_payload_bytes(tmp_path):
    store = LabelingStore(tmp_path / "labeling.sqlite")
    store.assign_recording(recording_id="rec-mask", assignee_user="alice")
    store.upsert_task(
        task_id="task-mask",
        recording_id="rec-mask",
        workflow_kind="subject_mask_component",
        component_name="body",
    )
    lease = store.create_session(task_id="task-mask", user="alice", ttl_seconds=600)
    pixels = bytes((row + column) % 2 for row in range(256) for column in range(256))
    payload = {
        "schema": "palette.web_labeling_subject_mask_checkpoint_payload.v1",
        "payload_kind": "dense_roi_replacement_mask",
        "mask": {
            "shape": [256, 256],
            "channels": 1,
            "dtype": "uint8",
            "encoding": "base64_raw",
            "pixels": base64.b64encode(pixels).decode("ascii"),
        },
    }
    metadata = {
        "schema": "palette.web_labeling_subject_mask_checkpoint_metadata.v1",
        "row_identity": {"roi_idx": 17, "source_frame_idx": 41},
        "component_name": "body",
        "target_run_path": "/runs/refined-subject-mask-a",
        "source_rowset_path": "/runs/source-subject-mask-a",
    }
    expected_payload_json = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    try:
        checkpoint = store.upsert_session_checkpoint(
            session_id=lease.session_id,
            task_id="task-mask",
            recording_id="rec-mask",
            user="alice",
            workflow_kind="subject_mask_component",
            target_run_path="/runs/refined-subject-mask-a",
            target_edit_revision=9,
            source_rowset_path="/runs/source-subject-mask-a",
            roi_idx=17,
            component_name="body",
            payload=payload,
            metadata=metadata,
        )
        stored = store.conn.execute(
            "SELECT payload_json FROM labeling_session_checkpoints WHERE checkpoint_id = ?;",
            (checkpoint["checkpoint_id"],),
        ).fetchone()
        assert stored["payload_json"] == expected_payload_json
        assert checkpoint["snapshot_row_sha256"] == _snapshot_row_sha256(checkpoint)
        descriptors = store.list_session_checkpoint_snapshot_descriptors(
            task_id="task-mask",
            state="active",
            component_name="body",
        )
        assert descriptors == [
            {
                "checkpoint_id": checkpoint["checkpoint_id"],
                "roi_idx": 17,
                "updated_at_utc": checkpoint["updated_at_utc"],
                "state": "active",
                "apply_id": None,
                "snapshot_row_sha256": checkpoint["snapshot_row_sha256"],
            }
        ]
    finally:
        store.close()


def test_checkpoint_claim_is_exclusive_and_same_apply_id_replays_exact_snapshot(tmp_path):
    store, lease = _checkpoint_store(tmp_path)
    try:
        first = store.upsert_session_checkpoint(**_checkpoint_kwargs(lease, roi_idx=7))
        second = store.upsert_session_checkpoint(**_checkpoint_kwargs(lease, roi_idx=8))
        barrier = threading.Barrier(2)
        claims: dict[str, list[dict[str, object]]] = {}
        errors: list[BaseException] = []

        def claim(apply_id: str) -> None:
            try:
                barrier.wait()
                claims[apply_id] = store.claim_session_checkpoints_for_apply(
                    task_id="task-a",
                    component_name="keypoints",
                    apply_id=apply_id,
                )
            except BaseException as exc:  # pragma: no cover - assertion reports collected failures
                errors.append(exc)
            finally:
                store.close_thread_connection()

        threads = [threading.Thread(target=claim, args=(value,)) for value in ("apply-a", "apply-b")]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)

        assert not errors
        winners = [apply_id for apply_id, rows in claims.items() if rows]
        assert len(winners) == 1
        winner = winners[0]
        assert {row["checkpoint_id"] for row in claims[winner]} == {
            first["checkpoint_id"],
            second["checkpoint_id"],
        }
        assert store.claim_session_checkpoints_for_apply(
            task_id="task-a",
            component_name="keypoints",
            apply_id="different-apply",
        ) == []
        replay = store.claim_session_checkpoints_for_apply(
            task_id="task-a",
            component_name="keypoints",
            apply_id=winner,
        )
        assert [row["checkpoint_id"] for row in replay] == [
            row["checkpoint_id"] for row in claims[winner]
        ]
    finally:
        store.close()


def test_checkpoint_save_refuses_claimed_row_without_overwriting_payload(tmp_path):
    store, lease = _checkpoint_store(tmp_path)
    try:
        checkpoint = store.upsert_session_checkpoint(**_checkpoint_kwargs(lease, value=1))
        claimed = store.claim_session_checkpoints_for_apply(
            task_id="task-a",
            component_name="keypoints",
            apply_id="apply-a",
        )
        assert [row["checkpoint_id"] for row in claimed] == [checkpoint["checkpoint_id"]]

        with pytest.raises(RuntimeError, match="being applied"):
            store.upsert_session_checkpoint(**_checkpoint_kwargs(lease, value=2))

        persisted = store.get_session_checkpoint(
            task_id="task-a",
            roi_idx=7,
            component_name="keypoints",
            state="applying",
        )
        assert persisted is not None
        assert persisted["payload"] == checkpoint["payload"]
        assert persisted["apply_id"] == "apply-a"
    finally:
        store.close()


def test_checkpoint_finalize_requires_all_requested_rows_owned_by_apply(tmp_path):
    store, lease = _checkpoint_store(tmp_path)
    try:
        checkpoint = store.upsert_session_checkpoint(**_checkpoint_kwargs(lease))
        store.claim_session_checkpoints_for_apply(
            task_id="task-a",
            component_name="keypoints",
            apply_id="apply-a",
        )

        with pytest.raises(RuntimeError, match="owned applying checkpoints"):
            store.mark_session_checkpoints_applied(
                checkpoint_ids=[checkpoint["checkpoint_id"], "missing-checkpoint"],
                apply_id="apply-a",
                edit_revision_before=3,
                edit_revision_after=4,
            )
        assert store.get_session_checkpoint(
            task_id="task-a",
            roi_idx=7,
            component_name="keypoints",
            state="applying",
        ) is not None

        assert store.mark_session_checkpoints_applied(
            checkpoint_ids=[checkpoint["checkpoint_id"]],
            apply_id="apply-a",
            edit_revision_before=3,
            edit_revision_after=4,
        ) == 1
    finally:
        store.close()


def test_applied_checkpoint_receipt_survives_reopen_and_later_row_save(tmp_path):
    store, lease = _checkpoint_store(tmp_path)
    store_path = store.path
    checkpoint = store.upsert_session_checkpoint(**_checkpoint_kwargs(lease, value=1))
    store.claim_session_checkpoints_for_apply(
        task_id="task-a",
        component_name="keypoints",
        apply_id="apply-durable",
    )
    assert store.mark_session_checkpoints_applied(
        checkpoint_ids=[checkpoint["checkpoint_id"]],
        apply_id="apply-durable",
        edit_revision_before=3,
        edit_revision_after=4,
    ) == 1
    assert store.count_pending_session_checkpoint_apply_effects(
        task_id="task-a",
        component_name="keypoints",
    ) == 0
    store.close()

    reopened = LabelingStore(store_path)
    try:
        prior = reopened.get_applied_session_checkpoints_by_apply_id(
            task_id="task-a",
            apply_id="apply-durable",
        )
        assert len(prior) == 1
        assert prior[0]["payload"] == {"schema": "palette.keypoint_checkpoint.v1", "value": 1}
        assert prior[0]["secondary_effects_state"] == "complete"
        reopened.upsert_session_checkpoint(**_checkpoint_kwargs(lease, value=2))

        replay = reopened.get_applied_session_checkpoints_by_apply_id(
            task_id="task-a",
            apply_id="apply-durable",
        )
        assert replay == prior
        assert reopened.claim_session_checkpoints_for_apply(
            task_id="task-a",
            component_name="keypoints",
            apply_id="apply-durable",
        ) == []
        active = reopened.get_session_checkpoint(
            task_id="task-a",
            roi_idx=7,
            component_name="keypoints",
            state="active",
        )
        assert active is not None
        assert active["payload"]["value"] == 2
    finally:
        reopened.close()


def test_applied_checkpoint_secondary_effects_are_durable_strict_and_idempotent(tmp_path):
    store, lease = _checkpoint_store(tmp_path)
    store_path = store.path
    checkpoint = store.upsert_session_checkpoint(**_checkpoint_kwargs(lease))
    expected_snapshot_sha256 = "a" * 64
    claimed = store.claim_session_checkpoints_for_apply(
        task_id="task-a",
        component_name="keypoints",
        apply_id="effects-apply",
        checkpoint_snapshot_sha256=expected_snapshot_sha256,
    )
    assert [row["checkpoint_id"] for row in claimed] == [checkpoint["checkpoint_id"]]
    replayed_claim = store.claim_session_checkpoints_for_apply(
        task_id="task-a",
        component_name="keypoints",
        apply_id="effects-apply",
        checkpoint_snapshot_sha256=expected_snapshot_sha256,
    )
    assert [row["checkpoint_id"] for row in replayed_claim] == [
        checkpoint["checkpoint_id"]
    ]
    with pytest.raises(RuntimeError, match="different checkpoint snapshot digest"):
        store.claim_session_checkpoints_for_apply(
            task_id="task-a",
            component_name="keypoints",
            apply_id="effects-apply",
            checkpoint_snapshot_sha256="b" * 64,
        )
    assert store.mark_session_checkpoints_applied(
        checkpoint_ids=[checkpoint["checkpoint_id"]],
        apply_id="effects-apply",
        edit_revision_before=3,
        edit_revision_after=4,
        require_secondary_effects=True,
    ) == 1
    assert store.count_pending_session_checkpoint_apply_effects(
        task_id="task-a",
        component_name="keypoints",
    ) == 1
    pending = store.list_pending_session_checkpoint_apply_effects(
        task_id="task-a",
        component_name="keypoints",
    )
    assert len(pending) == 1
    assert pending[0]["apply_id"] == "effects-apply"
    assert pending[0]["secondary_effects_state"] == "pending"
    assert pending[0]["checkpoint_snapshot_sha256"] == expected_snapshot_sha256
    assert pending[0]["checkpoint_count"] == 1
    assert "checkpoints_json" not in pending[0]
    store.close()

    reopened = LabelingStore(store_path)
    try:
        assert reopened.count_pending_session_checkpoint_apply_effects(
            task_id="task-a",
            component_name="keypoints",
        ) == 1
        replay = reopened.get_applied_session_checkpoints_by_apply_id(
            task_id="task-a",
            apply_id="effects-apply",
        )
        assert [row["checkpoint_id"] for row in replay] == [checkpoint["checkpoint_id"]]
        with pytest.raises(RuntimeError, match="different checkpoint task"):
            reopened.mark_session_checkpoint_apply_effects_complete(
                task_id="wrong-task",
                component_name="keypoints",
                apply_id="effects-apply",
            )
        with pytest.raises(RuntimeError, match="different checkpoint component"):
            reopened.mark_session_checkpoint_apply_effects_complete(
                task_id="task-a",
                component_name="body",
                apply_id="effects-apply",
            )
        assert reopened.mark_session_checkpoint_apply_effects_complete(
            task_id="task-a",
            component_name="keypoints",
            apply_id="effects-apply",
        ) is True
        assert reopened.mark_session_checkpoint_apply_effects_complete(
            task_id="task-a",
            component_name="keypoints",
            apply_id="effects-apply",
        ) is False
        assert reopened.count_pending_session_checkpoint_apply_effects(
            task_id="task-a",
            component_name="keypoints",
        ) == 0
    finally:
        reopened.close()


def test_mixed_fresh_and_stale_apply_receipt_records_only_applied_subset(tmp_path):
    store, lease = _checkpoint_store(tmp_path)
    try:
        fresh = store.upsert_session_checkpoint(**_checkpoint_kwargs(lease, roi_idx=7, value=7))
        stale = store.upsert_session_checkpoint(**_checkpoint_kwargs(lease, roi_idx=8, value=8))
        claimed = store.claim_session_checkpoints_for_apply(
            task_id="task-a",
            component_name="keypoints",
            apply_id="apply-mixed",
        )
        assert {row["checkpoint_id"] for row in claimed} == {
            fresh["checkpoint_id"],
            stale["checkpoint_id"],
        }

        assert store.mark_session_checkpoints_applied(
            checkpoint_ids=[fresh["checkpoint_id"]],
            apply_id="apply-mixed",
            edit_revision_before=3,
            edit_revision_after=4,
        ) == 1
        assert store.get_session_checkpoint(
            task_id="task-a",
            roi_idx=8,
            component_name="keypoints",
            state="active",
        ) is not None
        assert store.release_session_checkpoints_apply(
            task_id="task-a",
            apply_id="apply-mixed",
        ) == 1
        assert store.release_session_checkpoints_apply(
            task_id="task-a",
            apply_id="apply-mixed",
        ) == 1

        receipt = store.get_applied_session_checkpoints_by_apply_id(
            task_id="task-a",
            apply_id="apply-mixed",
        )
        assert [row["checkpoint_id"] for row in receipt] == [fresh["checkpoint_id"]]
        assert store.get_session_checkpoint(
            task_id="task-a",
            roi_idx=8,
            component_name="keypoints",
            state="active",
        ) is not None
        assert store.claim_session_checkpoints_for_apply(
            task_id="task-a",
            component_name="keypoints",
            apply_id="apply-mixed",
        ) == []
    finally:
        store.close()


def test_checkpoint_finalize_sql_failure_rolls_back_receipt_and_rows(tmp_path):
    store, lease = _checkpoint_store(tmp_path)
    try:
        checkpoint = store.upsert_session_checkpoint(**_checkpoint_kwargs(lease))
        store.claim_session_checkpoints_for_apply(
            task_id="task-a",
            component_name="keypoints",
            apply_id="apply-failure",
        )
        store.conn.execute(
            """
            CREATE TRIGGER reject_checkpoint_finalize
            BEFORE UPDATE OF state ON labeling_session_checkpoints
            WHEN NEW.state = 'applied'
            BEGIN
                SELECT RAISE(ABORT, 'injected finalize failure');
            END;
            """
        )
        store.conn.commit()

        with pytest.raises(sqlite3.IntegrityError, match="injected finalize failure"):
            store.mark_session_checkpoints_applied(
                checkpoint_ids=[checkpoint["checkpoint_id"]],
                apply_id="apply-failure",
                edit_revision_before=3,
                edit_revision_after=4,
            )

        receipt = store.conn.execute(
            "SELECT state, secondary_effects_state "
            "FROM labeling_checkpoint_apply_receipts WHERE apply_id = 'apply-failure';"
        ).fetchone()
        assert dict(receipt) == {
            "state": "applying",
            "secondary_effects_state": "not_ready",
        }
        assert store.get_applied_session_checkpoints_by_apply_id(
            task_id="task-a",
            apply_id="apply-failure",
        ) == []
        applying = store.get_session_checkpoint(
            task_id="task-a",
            roi_idx=7,
            component_name="keypoints",
            state="applying",
        )
        assert applying is not None
        assert applying["apply_id"] == "apply-failure"
    finally:
        store.close()


def test_v6_sidecar_migration_preserves_mask_checkpoint_and_keypoint_audit_bytes(tmp_path):
    store = LabelingStore(tmp_path / "labeling.sqlite")
    store_path = store.path
    store.assign_recording(recording_id="rec-a", assignee_user="alice")
    store.upsert_task(
        task_id="task-mask",
        recording_id="rec-a",
        workflow_kind="subject_mask_component",
        component_name="body",
    )
    mask_lease = store.create_session(task_id="task-mask", user="alice", ttl_seconds=600)
    checkpoint = store.upsert_session_checkpoint(
        session_id=mask_lease.session_id,
        task_id="task-mask",
        recording_id="rec-a",
        user="alice",
        workflow_kind="subject_mask_component",
        target_run_path="/runs/refined-subject-mask-a",
        target_edit_revision=9,
        source_rowset_path="/runs/source-subject-mask-a",
        roi_idx=17,
        component_name="body",
        payload={"schema": "palette.subject_mask_checkpoint.v1", "mask_rle": [1, 4, 2]},
        metadata={"row_identity": {"roi_idx": 17, "source_frame_idx": 41}},
    )
    store.claim_session_checkpoints_for_apply(
        task_id="task-mask",
        component_name="body",
        apply_id="legacy-mask-apply",
    )
    store.mark_session_checkpoints_applied(
        checkpoint_ids=[checkpoint["checkpoint_id"]],
        apply_id="legacy-mask-apply",
        edit_revision_before=9,
        edit_revision_after=10,
    )
    store.upsert_task(
        task_id="task-keypoints",
        recording_id="rec-a",
        workflow_kind="keypoints",
    )
    event = store.record_event(
        task_id="task-keypoints",
        recording_id="rec-a",
        user="alice",
        event_type="save_keypoints",
        target={"roi_idx": 23, "source_frame_idx": 52},
        before={"left_eye": [1.25, 2.5]},
        after={"left_eye": [1.5, 2.75]},
    )
    identity_sql = {
        "task": ("SELECT * FROM labeling_tasks WHERE task_id = ?", ("task-mask",)),
        "session": (
            "SELECT * FROM labeling_sessions WHERE session_id = ?",
            (mask_lease.session_id,),
        ),
        "checkpoint": (
            "SELECT * FROM labeling_session_checkpoints WHERE checkpoint_id = ?",
            (checkpoint["checkpoint_id"],),
        ),
        "event": (
            "SELECT * FROM labeling_task_events WHERE event_id = ?",
            (event["event_id"],),
        ),
    }
    store.conn.execute("DROP TABLE labeling_checkpoint_apply_receipts;")
    store.conn.execute("DROP INDEX idx_labeling_session_checkpoints_snapshot_order;")
    store.conn.execute(
        "ALTER TABLE labeling_session_checkpoints DROP COLUMN snapshot_row_sha256;"
    )
    before = {
        name: dict(store.conn.execute(sql, params).fetchone())
        for name, (sql, params) in identity_sql.items()
    }
    store.conn.execute(
        "UPDATE labeling_schema_meta SET value = '6' WHERE key = 'schema_version';"
    )
    store.conn.commit()
    store.close()

    reopened = LabelingStore(store_path)
    try:
        reopened.initialize()
        after = {
            name: dict(reopened.conn.execute(sql, params).fetchone())
            for name, (sql, params) in identity_sql.items()
        }
        assert {
            name: {key: row[key] for key in before[name]}
            for name, row in after.items()
        } == before
        assert after["event"]["target_json"] == before["event"]["target_json"]
        assert after["event"]["before_json"] == before["event"]["before_json"]
        assert after["event"]["after_json"] == before["event"]["after_json"]
        replay = reopened.get_applied_session_checkpoints_by_apply_id(
            task_id="task-mask",
            apply_id="legacy-mask-apply",
        )
        assert len(replay) == 1
        assert replay[0]["checkpoint_id"] == checkpoint["checkpoint_id"]
        assert replay[0]["payload"] == {
            "schema": "palette.subject_mask_checkpoint.v1",
            "mask_rle": [1, 4, 2],
        }
        assert replay[0]["metadata"]["row_identity"] == {
            "roi_idx": 17,
            "source_frame_idx": 41,
        }
        assert replay[0]["snapshot_row_sha256"] == _snapshot_row_sha256(replay[0])
        receipt = reopened.conn.execute(
            "SELECT state, checkpoint_count, secondary_effects_state "
            "FROM labeling_checkpoint_apply_receipts "
            "WHERE apply_id = 'legacy-mask-apply';"
        ).fetchone()
        assert dict(receipt) == {
            "state": "applied",
            "checkpoint_count": 1,
            "secondary_effects_state": "complete",
        }
        assert reopened.count_pending_session_checkpoint_apply_effects(
            task_id="task-mask",
            component_name="body",
        ) == 0
        version = reopened.conn.execute(
            "SELECT value FROM labeling_schema_meta WHERE key = 'schema_version';"
        ).fetchone()["value"]
        assert version == str(SCHEMA_VERSION)
    finally:
        reopened.close()


def test_v7_sidecar_migration_backfills_current_and_historical_snapshot_digests(tmp_path):
    store, lease = _checkpoint_store(tmp_path)
    store_path = store.path
    checkpoint = store.upsert_session_checkpoint(
        **{
            **_checkpoint_kwargs(lease),
            "payload": {
                "schema": "palette.keypoint_checkpoint.v1",
                "point": [12.5, None],
            },
        }
    )
    store.claim_session_checkpoints_for_apply(
        task_id="task-a",
        component_name="keypoints",
        apply_id="v7-applied",
    )
    store.mark_session_checkpoints_applied(
        checkpoint_ids=[checkpoint["checkpoint_id"]],
        apply_id="v7-applied",
        edit_revision_before=3,
        edit_revision_after=4,
    )
    receipt = store.conn.execute(
        "SELECT checkpoints_json FROM labeling_checkpoint_apply_receipts WHERE apply_id = ?;",
        ("v7-applied",),
    ).fetchone()
    historical = json.loads(receipt["checkpoints_json"])
    historical[0].pop("snapshot_row_sha256", None)
    legacy_receipt_json = json.dumps(
        historical,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    store.conn.execute(
        "UPDATE labeling_checkpoint_apply_receipts SET checkpoints_json = ? WHERE apply_id = ?;",
        (legacy_receipt_json, "v7-applied"),
    )
    store.conn.execute("DROP INDEX idx_labeling_session_checkpoints_snapshot_order;")
    store.conn.execute("DROP INDEX idx_labeling_checkpoint_apply_effects_pending;")
    store.conn.execute(
        "ALTER TABLE labeling_checkpoint_apply_receipts "
        "DROP COLUMN checkpoint_snapshot_sha256;"
    )
    store.conn.execute(
        "ALTER TABLE labeling_checkpoint_apply_receipts "
        "DROP COLUMN secondary_effects_completed_at_utc;"
    )
    store.conn.execute(
        "ALTER TABLE labeling_checkpoint_apply_receipts "
        "DROP COLUMN secondary_effects_state;"
    )
    store.conn.execute(
        "ALTER TABLE labeling_session_checkpoints DROP COLUMN snapshot_row_sha256;"
    )
    store.conn.execute(
        "UPDATE labeling_schema_meta SET value = '7' WHERE key = 'schema_version';"
    )
    store.conn.commit()
    store.close()

    reopened = LabelingStore(store_path)
    try:
        reopened.initialize()
        current = reopened.get_session_checkpoint(
            task_id="task-a",
            roi_idx=7,
            component_name="keypoints",
            state="applied",
        )
        assert current is not None
        assert current["snapshot_row_sha256"] == _snapshot_row_sha256(current)
        replay = reopened.get_applied_session_checkpoints_by_apply_id(
            task_id="task-a",
            apply_id="v7-applied",
        )
        assert len(replay) == 1
        assert replay[0]["snapshot_row_sha256"] == _snapshot_row_sha256(replay[0])
        stored_receipt_json = reopened.conn.execute(
            "SELECT checkpoints_json FROM labeling_checkpoint_apply_receipts WHERE apply_id = ?;",
            ("v7-applied",),
        ).fetchone()["checkpoints_json"]
        assert stored_receipt_json == legacy_receipt_json
        receipt_effects_state = reopened.conn.execute(
            "SELECT secondary_effects_state FROM labeling_checkpoint_apply_receipts "
            "WHERE apply_id = ?;",
            ("v7-applied",),
        ).fetchone()["secondary_effects_state"]
        assert receipt_effects_state == "complete"
        version = reopened.conn.execute(
            "SELECT value FROM labeling_schema_meta WHERE key = 'schema_version';"
        ).fetchone()["value"]
        assert version == str(SCHEMA_VERSION)
    finally:
        reopened.close()


def test_v7_nonfinite_checkpoint_migration_remains_openable(tmp_path):
    store, lease = _checkpoint_store(tmp_path)
    store_path = store.path
    checkpoint = store.upsert_session_checkpoint(**_checkpoint_kwargs(lease))
    store.conn.execute(
        "UPDATE labeling_session_checkpoints SET payload_json = ? WHERE checkpoint_id = ?;",
        (
            '{"confidence":NaN,"schema":"palette.legacy_checkpoint.v1"}',
            checkpoint["checkpoint_id"],
        ),
    )
    store.conn.execute("DROP INDEX idx_labeling_session_checkpoints_snapshot_order;")
    store.conn.execute("DROP INDEX idx_labeling_checkpoint_apply_effects_pending;")
    store.conn.execute(
        "ALTER TABLE labeling_checkpoint_apply_receipts "
        "DROP COLUMN checkpoint_snapshot_sha256;"
    )
    store.conn.execute(
        "ALTER TABLE labeling_checkpoint_apply_receipts "
        "DROP COLUMN secondary_effects_completed_at_utc;"
    )
    store.conn.execute(
        "ALTER TABLE labeling_checkpoint_apply_receipts "
        "DROP COLUMN secondary_effects_state;"
    )
    store.conn.execute(
        "ALTER TABLE labeling_session_checkpoints DROP COLUMN snapshot_row_sha256;"
    )
    store.conn.execute(
        "UPDATE labeling_schema_meta SET value = '7' WHERE key = 'schema_version';"
    )
    store.conn.commit()
    store.close()

    reopened = LabelingStore(store_path)
    try:
        reopened.initialize()
        current = reopened.get_session_checkpoint(
            task_id="task-a",
            roi_idx=7,
            component_name="keypoints",
        )
        assert current is not None
        assert current["payload"]["confidence"] != current["payload"]["confidence"]
        assert current["snapshot_row_sha256"] == session_checkpoint_snapshot_row_sha256(
            current
        )
        descriptor = reopened.list_session_checkpoint_snapshot_descriptors(
            task_id="task-a",
            state="active",
            component_name="keypoints",
        )
        assert descriptor[0]["snapshot_row_sha256"] == current["snapshot_row_sha256"]
    finally:
        reopened.close()


def test_apply_id_cannot_be_reused_for_another_task(tmp_path):
    store, lease = _checkpoint_store(tmp_path)
    try:
        store.upsert_session_checkpoint(**_checkpoint_kwargs(lease))
        store.claim_session_checkpoints_for_apply(
            task_id="task-a",
            component_name="keypoints",
            apply_id="globally-bound-apply",
        )
        store.assign_recording(recording_id="rec-b", assignee_user="bob")
        store.upsert_task(
            task_id="task-b",
            recording_id="rec-b",
            workflow_kind="keypoints",
        )
        second_lease = store.create_session(task_id="task-b", user="bob", ttl_seconds=600)
        store.upsert_session_checkpoint(
            **_checkpoint_kwargs(second_lease, roi_idx=9),
        )

        with pytest.raises(RuntimeError, match="different task"):
            store.claim_session_checkpoints_for_apply(
                task_id="task-b",
                component_name="keypoints",
                apply_id="globally-bound-apply",
            )
    finally:
        store.close()


def test_checkpoint_batches_over_default_limit_match_preview_and_cover_all_rows(tmp_path):
    store, lease = _checkpoint_store(tmp_path)
    now = "2026-09-19T12:00:00+00:00"
    rows = []
    for roi_idx in range(1005):
        checkpoint_id = f"checkpoint-{roi_idx:04d}"
        snapshot = {
            "checkpoint_id": checkpoint_id,
            "task_id": "task-a",
            "recording_id": "rec-a",
            "user": "alice",
            "workflow_kind": "keypoints",
            "target_run_path": "/runs/refined-keypoints-a",
            "target_edit_revision": 3,
            "source_rowset_path": "/runs/source-keypoints-a",
            "roi_idx": roi_idx,
            "component_name": "keypoints",
            "payload": {"schema": "palette.keypoint_checkpoint.v1", "value": 1},
            "metadata": {},
        }
        rows.append(
            (
                checkpoint_id,
                lease.session_id,
                "task-a",
                "rec-a",
                "alice",
                "keypoints",
                "/runs/refined-keypoints-a",
                3,
                "/runs/source-keypoints-a",
                roi_idx,
                "keypoints",
                '{"schema":"palette.keypoint_checkpoint.v1","value":1}',
                "{}",
                _snapshot_row_sha256(snapshot),
                "active",
                now,
                now,
            )
        )
    try:
        store.conn.executemany(
            """
            INSERT INTO labeling_session_checkpoints (
                checkpoint_id, session_id, task_id, recording_id, user, workflow_kind,
                target_run_path, target_edit_revision, source_rowset_path,
                roi_idx, component_name, payload_json, metadata_json,
                snapshot_row_sha256, state,
                created_at_utc, updated_at_utc, applied_at_utc, apply_id,
                edit_revision_before, edit_revision_after
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL, NULL, NULL);
            """,
            rows,
        )
        store.conn.commit()

        first_preview = store.list_session_checkpoints(
            task_id="task-a",
            state="active",
            component_name="keypoints",
            limit=1000,
        )
        first_claim = store.claim_session_checkpoints_for_apply(
            task_id="task-a",
            component_name="keypoints",
            apply_id="apply-batch-1",
        )
        assert [row["checkpoint_id"] for row in first_claim] == [
            row["checkpoint_id"] for row in first_preview
        ]
        assert len(first_claim) == 1000
        assert store.count_session_checkpoints(
            task_id="task-a",
            state="active",
            component_name="keypoints",
        ) == 5
        assert store.count_session_checkpoints(
            task_id="task-a",
            state="applying",
            component_name="keypoints",
        ) == 1000
        assert store.mark_session_checkpoints_applied(
            checkpoint_ids=[str(row["checkpoint_id"]) for row in first_claim],
            apply_id="apply-batch-1",
            edit_revision_before=3,
            edit_revision_after=4,
        ) == 1000
        assert store.count_unapplied_session_checkpoints(
            task_id="task-a",
            component_name="keypoints",
        ) == 5

        second_preview = store.list_session_checkpoints(
            task_id="task-a",
            state="active",
            component_name="keypoints",
            limit=1000,
        )
        second_claim = store.claim_session_checkpoints_for_apply(
            task_id="task-a",
            component_name="keypoints",
            apply_id="apply-batch-2",
        )
        assert [row["checkpoint_id"] for row in second_claim] == [
            row["checkpoint_id"] for row in second_preview
        ]
        assert len(second_claim) == 5
        assert {
            int(row["roi_idx"]) for row in first_claim + second_claim
        } == set(range(1005))
    finally:
        store.close()
