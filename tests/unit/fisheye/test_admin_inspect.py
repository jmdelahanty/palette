"""Admin inspect view: read-only look at any labeler's applied and saved work."""

from __future__ import annotations

import base64
import hashlib
import json
import re
import shutil
import sqlite3
import subprocess
from pathlib import Path

import numpy as np
import pytest

from fisheye.labeling import web, web_static
from fisheye.labeling.admin_inspect import InspectError, inspect_row, inspect_task, inspect_tasks
from tests.unit.fisheye.test_labeling_web_routes import _running_server
from tests.unit.fisheye.test_mask_tail_apply_refresh import browser_context, reviewed_archive  # noqa: F401
from tests.unit.fisheye.test_web_mask_tail_apply_integration import request

JS = web_static.STATIC_ROOT / "js"


def _tree_digest(path: Path) -> str:
    digest = hashlib.sha256()
    for file in sorted(p for p in Path(path).rglob("*") if p.is_file()):
        digest.update(str(file.relative_to(path)).encode() + b"\0" + file.read_bytes())
    return digest.hexdigest()


def _store_digest(path: Path) -> str:
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        dump = "\n".join(conn.iterdump())
    finally:
        conn.close()
    return hashlib.sha256(dump.encode()).hexdigest()


def _mask(payload) -> np.ndarray:
    return np.frombuffer(base64.b64decode(payload["pixels"]), dtype=np.uint8).reshape(payload["shape"])


@pytest.fixture
def labeled(reviewed_archive, tmp_path):
    """Row 0 saved and applied; row 1 saved but never applied."""

    path, root, initial = reviewed_archive
    store, _runtime = browser_context(reviewed_archive, tmp_path)
    store.upsert_labeling_user(user_id="reviewer", status="active")
    store.upsert_labeling_user(user_id="admin", status="active")
    mask_run = root[initial["paths"]["mask_edit"]]
    applied = np.asarray(mask_run["masks_roi"][0, 0]).copy()
    applied[2:6, 2:8] = 1
    saved = np.asarray(mask_run["masks_roi"][1, 0]).copy()
    saved[4:9, 100:107] = 1
    session = store.create_session(task_id="original-mask", user="reviewer")
    route = f"/api/sessions/{session.session_id}/subject-mask"
    with _running_server(store, user="reviewer") as base:
        for position, mask in ((0, applied),):
            status, nav = request(base, route + "/nav", {"position": position})
            status, out = request(base, route + "/save", {"mask": web._raw_array_payload(mask), "target_token": nav["state"]["target_token"]})
            assert status == 200, out
        status, out = request(base, route + "/apply", {"apply_id": "a1", "target_token": out["state"]["target_token"]})
        assert status == 200, out
        status, nav = request(base, route + "/nav", {"position": 1})
        status, out = request(base, route + "/save", {"mask": web._raw_array_payload(saved), "target_token": nav["state"]["target_token"]})
        assert status == 200, out
    store.close_session(session_id=session.session_id, user="reviewer")
    try:
        yield store, Path(path), applied, saved
    finally:
        store.close()


def test_task_rows_report_applied_saved_and_untouched(labeled):
    store, *_ = labeled
    task = inspect_task(store.path, "original-mask")
    status = {r["roi_idx"]: r["status"] for r in task["rows"]}
    assert status[0] == "applied" and status[1] == "saved"
    assert task["counts"]["applied"] == 1 and task["counts"]["saved"] == 1
    listed = {t["task_id"]: t for t in inspect_tasks(store.path)["tasks"]}
    assert listed["original-mask"]["applied_rows"] == 1 and listed["original-mask"]["saved_rows"] == 1
    assert listed["original-mask"]["labelers"] == ["reviewer"]


def test_rows_show_the_applied_and_the_saved_mask(labeled):
    store, _path, applied, saved = labeled
    row0 = inspect_row(store.path, "original-mask", 0)
    np.testing.assert_array_equal(_mask(row0["applied_mask"]), (applied > 0).astype(np.uint8))
    assert row0["saved_mask"] is None and row0["status"] == "applied"
    row1 = inspect_row(store.path, "original-mask", 1)
    np.testing.assert_array_equal(_mask(row1["saved_mask"]), (saved > 0).astype(np.uint8))
    assert not np.array_equal(_mask(row1["applied_mask"]), _mask(row1["saved_mask"]))
    assert row1["status"] == "saved" and row1["labeler"] == "reviewer" and row1["read_only"] is True
    assert _mask(row1["image"]).shape[:2] == _mask(row1["applied_mask"]).shape


def test_keypoint_rows_are_inspectable(labeled):
    store, *_ = labeled
    task = inspect_task(store.path, "original-pose")
    assert task["rows"] and all(r["status"] == "untouched" for r in task["rows"])
    row = inspect_row(store.path, "original-pose", task["rows"][0]["roi_idx"])
    assert len(row["applied_points"]) == len(row["labels"]) and row["saved_points"] is None


def test_inspecting_changes_nothing_in_the_archive_or_the_store(labeled):
    store, path, *_ = labeled
    archive_before, store_before = _tree_digest(path), _store_digest(store.path)
    with _running_server(store, user="admin", admin_users=("admin",)) as base:
        status, tasks = request(base, "/api/admin/inspect/tasks")
        assert status == 200
        for task_id in ("original-mask", "original-pose"):
            status, task = request(base, f"/api/admin/inspect/task?task_id={task_id}")
            assert status == 200, task
            for r in task["rows"][:3]:
                status, row = request(base, f"/api/admin/inspect/row?task_id={task_id}&roi_idx={r['roi_idx']}")
                assert status == 200, row
    assert _tree_digest(path) == archive_before
    assert _store_digest(store.path) == store_before


def test_inspect_is_admin_only_and_has_no_write_routes(labeled):
    store, *_ = labeled
    with _running_server(store, user="reviewer") as base:
        for route in ("/api/admin/inspect/tasks", "/api/admin/inspect/task?task_id=original-mask",
                      "/api/admin/inspect/row?task_id=original-mask&roi_idx=0", "/admin/inspect"):
            status, _ = (lambda r: (r[0], r[1]))(request(base, route))
            assert status == 403, route
    with _running_server(store, user="admin", admin_users=("admin",)) as base:
        status, _ = request(base, "/api/admin/inspect/row?task_id=original-mask&roi_idx=0", {"anything": 1})
        assert status in (404, 405)
    with pytest.raises(InspectError):
        inspect_row(store.path, "missing-task", 0)


def test_inspect_page_assets_resolve_and_scripts_parse(labeled, tmp_path):
    store, *_ = labeled
    import urllib.request
    with _running_server(store, user="admin", admin_users=("admin",)) as base:
        with urllib.request.urlopen(base + "/admin/inspect", timeout=10) as response:
            html = response.read().decode()
        for url in re.findall(r'(?:href|src)="(/static/[^"]+)"', html):
            with urllib.request.urlopen(base + url, timeout=10) as asset:
                assert asset.status == 200, url
    assert "@@" not in html
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is required to parse the page modules")
    for name in ("inspect_page.js", "inspect_model.js"):
        text = (JS / name).read_text()
        assert all(s.startswith(("./", "../")) for s in re.findall(r'from\s+"([^"]+)"', text)), name
        copy = tmp_path / f"{name}.mjs"
        copy.write_text(text)
        assert subprocess.run([node, "--check", str(copy)], capture_output=True, timeout=60).returncode == 0, name
    script = tmp_path / "model.mjs"
    script.write_text(f"""
import * as m from {json.dumps((JS / "inspect_model.js").as_uri())};
const enc = (arr, shape) => ({{shape, pixels: Buffer.from(Uint8Array.from(arr)).toString("base64")}});
globalThis.atob = (v) => Buffer.from(v, "base64").toString("binary");
const a = enc([1,1,0,0], [2,2]), s = enc([0,1,1,1], [2,2]);
console.log(JSON.stringify({{diff: m.maskDifference(a, s),
  rows: m.filterRows([{{roi_idx:1,status:"saved"}},{{roi_idx:2,status:"applied"}}], "saved").map(r => r.roi_idx),
  next: m.adjacentRow([{{roi_idx:4}},{{roi_idx:9}}], 4, 1)}}));
""")
    out = json.loads(subprocess.run([node, str(script)], capture_output=True, text=True, timeout=60).stdout)
    assert out == {"diff": {"added": 2, "removed": 1}, "rows": [1], "next": 9}
