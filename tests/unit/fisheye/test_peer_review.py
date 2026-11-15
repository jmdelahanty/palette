"""Peer review: any active labeler sees others' applied work read-only and flags frames."""

from __future__ import annotations

import base64
import hashlib
import json
import re
import shutil
import sqlite3
import subprocess
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pytest

from fisheye.labeling import web, web_static
from fisheye.labeling.admin_inspect import InspectError
from fisheye.labeling.peer_review import FlagError, add_frame_flag, peer_row, peer_task, peer_tasks
from tests.unit.fisheye.test_labeling_web_routes import _running_server
from tests.unit.fisheye.test_mask_tail_apply_refresh import browser_context, reviewed_archive  # noqa: F401
from tests.unit.fisheye.test_web_mask_tail_apply_integration import request

JS = web_static.STATIC_ROOT / "js"
SAVED_KEYS = {"saved_mask", "saved_points", "saved_operation", "payload", "payload_json"}


def _tree_digest(path: Path) -> str:
    digest = hashlib.sha256()
    for file in sorted(p for p in Path(path).rglob("*") if p.is_file()):
        digest.update(str(file.relative_to(path)).encode() + b"\0" + file.read_bytes())
    return digest.hexdigest()


def _store_digest(path: Path, *, skip: tuple[str, ...] = ()) -> str:
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        lines = [line for line in conn.iterdump() if not any(f'"{name}"' in line for name in skip)]
    finally:
        conn.close()
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


def _mask(payload) -> np.ndarray:
    return np.frombuffer(base64.b64decode(payload["pixels"]), dtype=np.uint8).reshape(payload["shape"])


def _post(base, route, payload, headers=None):
    req = urllib.request.Request(
        base + route, data=json.dumps(payload).encode(), method="POST",
        headers={"Content-Type": "application/json", **(headers or {})},
    )
    try:
        response = urllib.request.urlopen(req, timeout=60)
    except urllib.error.HTTPError as exc:
        response = exc
    with response:
        return response.status, json.loads(response.read())


@pytest.fixture
def labeled(reviewed_archive, tmp_path):
    """Owner 'reviewer': row 0 applied and then saved again, row 1 saved only."""

    path, root, initial = reviewed_archive
    store, _runtime = browser_context(reviewed_archive, tmp_path)
    for user in ("reviewer", "peer", "other", "admin"):
        store.upsert_labeling_user(user_id=user, status="active")
    store.upsert_labeling_user(user_id="gone", status="inactive")
    mask_run = root[initial["paths"]["mask_edit"]]

    def edited(roi, rows, cols):
        mask = np.asarray(mask_run["masks_roi"][roi, 0]).copy()
        mask[rows, cols] = 1
        return mask

    applied0 = edited(0, slice(2, 6), slice(2, 8))
    resaved0, saved1 = edited(0, slice(20, 30), slice(20, 40)), edited(1, slice(4, 9), slice(100, 107))
    session = store.create_session(task_id="original-mask", user="reviewer")
    route = f"/api/sessions/{session.session_id}/subject-mask"

    def save(base, position, mask):
        status, nav = request(base, route + "/nav", {"position": position})
        assert status == 200, nav
        status, out = request(base, route + "/save", {"mask": web._raw_array_payload(mask), "target_token": nav["state"]["target_token"]})
        assert status == 200, out
        return out

    with _running_server(store, user="reviewer") as base:
        out = save(base, 0, applied0)
        status, out = request(base, route + "/apply", {"apply_id": "a1", "target_token": out["state"]["target_token"]})
        assert status == 200, out
        # Visible through the current applied checkpoint before any later edit.
        assert [r["roi_idx"] for r in peer_task(store.path, "original-mask")["rows"]] == [0]
        assert {t["task_id"]: t for t in peer_tasks(store.path)["tasks"]}["original-mask"]["applied_rows"] == 1
        save(base, 0, resaved0)
        save(base, 1, saved1)
    store.close_session(session_id=session.session_id, user="reviewer")
    try:
        yield store, Path(path), applied0, resaved0
    finally:
        store.close()


def test_peers_see_rows_applied_at_least_once_including_resaved(labeled):
    store, _path, applied0, resaved0 = labeled
    task = peer_task(store.path, "original-mask")
    assert [r["roi_idx"] for r in task["rows"]] == [0]
    assert task["owner"] == "reviewer" and task["read_only"] is True
    row0 = peer_row(store.path, "original-mask", 0)
    # The applied mask from the Zarr, not the later saved edit.
    np.testing.assert_array_equal(_mask(row0["applied_mask"]), (applied0 > 0).astype(np.uint8))
    assert not np.array_equal(_mask(row0["applied_mask"]), (resaved0 > 0).astype(np.uint8))
    assert row0["apply_id"] == "a1" and row0["labeler"] == "reviewer"
    assert not SAVED_KEYS & set(row0)
    with pytest.raises(InspectError):
        peer_row(store.path, "original-mask", 1)
    listed = {t["task_id"]: t for t in peer_tasks(store.path)["tasks"]}
    # The list counts rows whose latest checkpoint is applied; row 0 was edited again.
    assert listed["original-mask"]["applied_rows"] == 0
    assert listed["original-mask"]["assignee"] == "reviewer"
    assert peer_task(store.path, "original-pose")["rows"] == []

def test_peer_routes_return_applied_work_and_write_nothing(labeled):
    store, path, *_ = labeled
    archive_before, store_before = _tree_digest(path), _store_digest(store.path)
    with _running_server(store, user="peer") as base:
        status, tasks = request(base, "/api/peer/tasks")
        assert status == 200 and tasks["user"] == "peer"
        status, task = request(base, "/api/peer/task?task_id=original-mask")
        assert status == 200 and [r["roi_idx"] for r in task["rows"]] == [0]
        for roi in (0,):
            status, row = request(base, f"/api/peer/row?task_id=original-mask&roi_idx={roi}")
            assert status == 200 and not SAVED_KEYS & set(row)
        status, refused = request(base, "/api/peer/row?task_id=original-mask&roi_idx=1")
        assert status == 400 and "no applied work" in refused["details"]
        with urllib.request.urlopen(base + "/peer", timeout=10) as response:
            assert response.status == 200
    assert _tree_digest(path) == archive_before
    assert _store_digest(store.path) == store_before


@pytest.mark.parametrize("user,expected", [("nobody", 403), ("gone", 403), (None, 401)])
def test_peer_routes_refuse_unknown_inactive_and_anonymous_users(tmp_path, user, expected):
    from fisheye.labeling.assignment_store import LabelingStore

    store = LabelingStore(tmp_path / "store.sqlite")
    store.initialize()
    store.upsert_labeling_user(user_id="gone", status="inactive")
    try:
        _assert_refused(store, user, expected)
    finally:
        store.close()


def _assert_refused(store, user, expected):
    with _running_server(store, user=user) as base:
        for route in ("/api/peer/tasks", "/api/peer/task?task_id=original-mask",
                      "/api/peer/row?task_id=original-mask&roi_idx=0"):
            assert request(base, route)[0] == expected, route
        assert _post(base, "/api/peer/flags", {"task_id": "original-mask", "roi_idx": 0, "note": "x"})[0] == expected
        assert _post(base, "/api/peer/flags/any/resolve", {})[0] == expected
        try:
            urllib.request.urlopen(base + "/peer", timeout=10)
            raise AssertionError("/peer was served")
        except urllib.error.HTTPError as exc:
            assert exc.code == expected


def test_flags_add_list_resolve_and_show_in_the_owners_queue(labeled):
    store, path, *_ = labeled
    label_tables = ("labeling_frame_flags",)
    archive_before, labels_before = _tree_digest(path), _store_digest(store.path, skip=label_tables)
    with _running_server(store, user="peer") as base:
        status, out = _post(base, "/api/peer/flags", {"task_id": "original-mask", "roi_idx": 0, "note": "Tail cut off"})
        assert status == 200, out
        flag = out["flag"]
        assert flag["state"] == "open" and flag["flagged_by"] == "peer" and flag["apply_id"] == "a1"
        assert _post(base, "/api/peer/flags", {"task_id": "original-mask", "roi_idx": 1, "note": "x"})[0] == 400
        assert _post(base, "/api/peer/flags", {"task_id": "original-mask", "roi_idx": 0, "note": "  "})[0] == 400
        assert _post(base, "/api/peer/flags", {"task_id": "original-mask", "roi_idx": 0, "note": "y" * 1001})[0] == 400
        status, row = request(base, "/api/peer/row?task_id=original-mask&roi_idx=0")
        assert [f["note"] for f in row["flags"]] == ["Tail cut off"] and row["flags"][0]["can_resolve"] is True
        status, task = request(base, "/api/peer/task?task_id=original-mask")
        assert {r["roi_idx"]: r["open_flag_count"] for r in task["rows"]} == {0: 1}
    with _running_server(store, user="reviewer") as base:
        status, queue = request(base, "/api/me/queue?expected_user=reviewer")
        assert status == 200
        counts = {t["task_id"]: t["open_flag_count"] for d in queue["datasets"] for r in d["recordings"] for t in r["tasks"]}
        assert counts["original-mask"] == 1
        assert queue["links"]["peer_review"] == "/peer"
    with _running_server(store, user="other") as base:
        status, row = request(base, "/api/peer/row?task_id=original-mask&roi_idx=0")
        assert row["flags"][0]["can_resolve"] is False
        status, out = _post(base, f"/api/peer/flags/{flag['flag_id']}/resolve", {})
        assert status == 403, out
        assert _post(base, "/api/peer/flags/missing/resolve", {})[0] == 404
    with _running_server(store, user="reviewer") as base:
        status, out = _post(base, f"/api/peer/flags/{flag['flag_id']}/resolve", {})
        assert status == 200 and out["flag"]["state"] == "resolved" and out["flag"]["resolved_by"] == "reviewer"
    assert {t["task_id"]: t for t in peer_tasks(store.path)["tasks"]}["original-mask"]["open_flag_count"] == 0
    # Flags never touch labels, checkpoints, receipts, or the archive.
    assert _tree_digest(path) == archive_before
    assert _store_digest(store.path, skip=label_tables) == labels_before


def test_flag_author_and_admin_may_resolve(labeled):
    store, *_ = labeled
    first = add_frame_flag(store, task_id="original-mask", roi_idx=0, flagged_by="peer", note="a")
    second = add_frame_flag(store, task_id="original-mask", roi_idx=0, flagged_by="other", note="b")
    with _running_server(store, user="peer") as base:
        assert _post(base, f"/api/peer/flags/{first['flag_id']}/resolve", {})[0] == 200
    with _running_server(store, user="admin", admin_users=("admin",)) as base:
        assert _post(base, f"/api/peer/flags/{second['flag_id']}/resolve", {})[0] == 200
    with pytest.raises(FlagError):
        add_frame_flag(store, task_id="original-mask", roi_idx=1, flagged_by="peer", note="never applied")


def test_flag_posts_require_same_origin(labeled):
    store, *_ = labeled
    with _running_server(store, user="peer") as base:
        status, out = _post(base, "/api/peer/flags", {"task_id": "original-mask", "roi_idx": 0, "note": "x"},
                            headers={"Origin": "http://evil.example"})
        assert status == 403 and out["error"] == "same_origin_required"
    assert peer_task(store.path, "original-mask")["rows"][0]["open_flag_count"] == 0


def test_flag_table_is_additive_and_keeps_the_schema_version(tmp_path):
    from fisheye.labeling.assignment_store import SCHEMA_VERSION, LabelingStore

    path = tmp_path / "store.sqlite"
    store = LabelingStore(path)
    store.initialize()
    store.close()
    conn = sqlite3.connect(path)
    before = sorted(conn.execute("SELECT name, sql FROM sqlite_master;").fetchall())
    assert any(name == "labeling_frame_flags" for name, _ in before)
    assert conn.execute("SELECT value FROM labeling_schema_meta WHERE key = 'schema_version';").fetchone()[0] == str(SCHEMA_VERSION)
    conn.close()
    store = LabelingStore(path)
    store.initialize()
    store.close()
    conn = sqlite3.connect(path)
    assert sorted(conn.execute("SELECT name, sql FROM sqlite_master;").fetchall()) == before
    conn.close()


def test_peer_page_assets_resolve_and_modules_parse(labeled, tmp_path):
    store, *_ = labeled
    with _running_server(store, user="peer") as base:
        with urllib.request.urlopen(base + "/peer", timeout=10) as response:
            html = response.read().decode()
        for url in re.findall(r'(?:href|src)="(/static/[^"]+)"', html):
            with urllib.request.urlopen(base + url, timeout=10) as asset:
                assert asset.status == 200, url
    assert "@@" not in html
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is required to parse the page modules")
    text = (JS / "peer_page.js").read_text()
    assert all(s.startswith(("./", "../")) for s in re.findall(r'from\s+"([^"]+)"', text))
    copy = tmp_path / "peer_page.mjs"
    copy.write_text(text)
    assert subprocess.run([node, "--check", str(copy)], capture_output=True, timeout=60).returncode == 0


def test_applied_keypoint_rows_show_the_zarr_points(labeled):
    store, *_ = labeled
    store.conn.execute("UPDATE labeling_tasks SET state = 'pending' WHERE task_id = 'original-pose';")
    store.conn.commit()
    lease = store.create_session(task_id="original-pose", user="reviewer")
    store.upsert_session_checkpoint(
        session_id=lease.session_id, task_id="original-pose", recording_id="rec", user="reviewer",
        workflow_kind="keypoints", target_run_path="refined_keypoints_runs/r", target_edit_revision=0,
        source_rowset_path=None, roi_idx=0, component_name="keypoints",
        payload={"operation": "set", "points": [[1.0, 2.0]]},
    )
    store.conn.execute(
        "UPDATE labeling_session_checkpoints SET state = 'applied', apply_id = 'k1', applied_at_utc = '2026-10-14T00:00:00Z' "
        "WHERE task_id = 'original-pose';"
    )
    store.conn.commit()
    row = peer_row(store.path, "original-pose", 0)
    assert row["apply_id"] == "k1" and len(row["applied_points"]) == len(row["labels"])
    assert not SAVED_KEYS & set(row)
