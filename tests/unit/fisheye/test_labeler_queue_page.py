"""GET /queue: the Preact labeler queue page built on /api/me/queue.

The page is a static shell plus ES modules served from /static/<build>/. It
must keep the same identity guards as /my-work, and its view model must take
the Start decision from the API rather than recompute it.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import urllib.error
import urllib.request

import pytest

pytest.importorskip("flask")

from fisheye.labeling import web_static
from fisheye.labeling.web_static import STATIC_CACHE_CONTROL, static_url

from .test_labeler_queue_api import _store
from .test_labeling_web_routes import (
    _assert_browser_response_security_headers,
    _json_request,
    _running_server,
)

STATIC_JS = web_static.STATIC_ROOT / "js"


def _get(base_url: str, path: str):
    request = urllib.request.Request(f"{base_url}{path}", method="GET")
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            return response.status, dict(response.headers), response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        return exc.code, dict(exc.headers), exc.read().decode("utf-8")


def _node():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is required to check the queue page modules")
    return node


def test_queue_page_shell_references_cacheable_assets_that_resolve(tmp_path):
    store = _store(tmp_path)
    try:
        with _running_server(store, user="alice") as base_url:
            status, headers, body = _get(base_url, "/queue?expected_user=alice")
            assert status == 200
            _assert_browser_response_security_headers(headers)
            assets = re.findall(r'(?:href|src)="(/static/[^"]+)"', body)
            assert assets == [static_url("css/palette.css"), static_url("js/queue_page.js")]
            assert "@@" not in body
            # Every module the page imports, transitively, is served from the same build.
            for relative in ("js/queue_page.js", "js/queue_model.js", "vendor/preact.module.js",
                             "vendor/preact-hooks.module.js", "vendor/htm.module.js", "css/palette.css"):
                asset_status, asset_headers, _ = _get(base_url, static_url(relative))
                assert asset_status == 200, relative
                assert asset_headers["Cache-Control"] == STATIC_CACHE_CONTROL
    finally:
        store.close()


def test_queue_page_keeps_the_personal_page_identity_guards(tmp_path):
    store = _store(tmp_path)
    try:
        with _running_server(store, user="alice") as base_url:
            mismatch, _, mismatch_body = _get(base_url, "/queue?expected_user=bob")
            full_mismatch, _, _ = _get(base_url, "/my-work?expected_user=bob")
        assert mismatch == full_mismatch == 403
        assert "dashboard_user_mismatch" in mismatch_body
        with _running_server(store, user="mallory") as base_url:
            unknown, _, _ = _get(base_url, "/queue")
            full_unknown, _, _ = _get(base_url, "/my-work")
        assert unknown == full_unknown == 403
        with _running_server(store, user=None) as base_url:
            anon, _, _ = _get(base_url, "/queue")
        assert anon == 401
    finally:
        store.close()


def test_page_modules_parse_and_import_only_relative_paths(tmp_path):
    node = _node()
    for name in ("queue_page.js", "queue_model.js"):
        text = (STATIC_JS / name).read_text()
        specifiers = re.findall(r'from\s+"([^"]+)"', text)
        assert all(s.startswith("./") or s.startswith("../") for s in specifiers), name
        copy = tmp_path / f"{name}.mjs"
        copy.write_text(text)
        result = subprocess.run([node, "--check", str(copy)], capture_output=True, text=True, timeout=60)
        assert result.returncode == 0, (name, result.stderr)


def test_view_model_takes_start_decision_from_the_real_api_payload(tmp_path):
    node = _node()
    store = _store(tmp_path)
    try:
        with _running_server(store, user="alice") as base_url:
            status, payload = _json_request(base_url, "/api/me/queue?expected_user=alice")
        assert status == 200
    finally:
        store.close()
    script = tmp_path / "model.mjs"
    script.write_text(
        f"""
import * as m from {json.dumps((STATIC_JS / "queue_model.js").as_uri())};
const payload = {json.dumps(payload)};
const rows = m.queueRows(payload);
console.log(JSON.stringify({{
  rows: rows.map((r) => [r.taskId, r.kindLabel, r.stateLabel, r.action, r.canStart, r.startEndpoint, r.notes]),
  filters: m.queueFilters(rows).map((f) => f.id),
  keypointsOnly: m.filterRows(rows, "keypoints").map((r) => r.taskId),
  summary: m.queueSummary(payload, rows),
  params: m.authParams("?expected_user=alice&invite=abc&other=1").toString(),
  merged: m.withParams("/api/tasks/t/open?x=1", m.authParams("?expected_user=alice")),
}}));
"""
    )
    result = subprocess.run([node, str(script)], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    model = json.loads(result.stdout)
    tasks = {t["task_id"]: t for d in payload["datasets"] for r in d["recordings"] for t in r["tasks"]}
    # Highest priority first; Start comes from start.ready, never recomputed.
    assert model["rows"] == [
        ["task-open", "Keypoints", "Not started", "Start", tasks["task-open"]["start"]["ready"],
         "/api/tasks/task-open/open", "Rows 3-9"],
        ["task-blocked", "Mask · body", "Blocked", "", False, "", ""],
    ]
    assert model["filters"] == ["", "keypoints", "subject_mask_component"]
    assert model["keypointsOnly"] == ["task-open"]
    assert model["summary"] == "1 open task across 1 recording."
    assert model["params"] == "expected_user=alice&invite=abc"
    assert model["merged"] == "/api/tasks/t/open?x=1&expected_user=alice"
