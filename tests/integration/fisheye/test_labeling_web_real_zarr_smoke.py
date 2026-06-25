from __future__ import annotations

import json
import os
import threading
import urllib.error
import urllib.request
from contextlib import contextmanager
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from fisheye.labeling import web as labeling_web
from fisheye.labeling.assignment_store import LabelingStore


SPEC_ENV = "PALETTE_LABELING_WEB_REAL_ZARR_SMOKE_SPEC"


@contextmanager
def _running_server(store: LabelingStore, *, user: str):
    config = labeling_web.ServerConfig(
        store_path=store.path,
        host="127.0.0.1",
        port=0,
        fixed_user=user,
        auth_header="X-Forwarded-User",
        session_ttl_seconds=600,
        admin_users=(user,),
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


def _json_request(base_url: str, path: str, *, method: str = "GET", payload: object | None = None):
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"} if payload is not None else {}
    request = urllib.request.Request(f"{base_url}{path}", data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))


def _load_spec() -> dict[str, object]:
    raw_path = os.environ.get(SPEC_ENV)
    if not raw_path:
        pytest.skip(f"Set {SPEC_ENV} to run real-zarr labeling web smoke tests.")
    path = Path(raw_path).expanduser()
    if not path.is_file():
        pytest.fail(f"{SPEC_ENV} does not point to a file: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        pytest.fail("Smoke spec must be a JSON object.")
    return data


def _route_prefix(workflow_kind: str) -> str:
    if workflow_kind == "keypoints":
        return "keypoints"
    if workflow_kind == "detect_training":
        return "detect"
    if workflow_kind == "detect_analysis":
        return "detect-analysis"
    if workflow_kind == "subject_mask_component":
        return "subject-mask"
    raise AssertionError(f"Unsupported workflow_kind in smoke spec: {workflow_kind}")


def test_labeling_web_real_zarr_smoke_spec(tmp_path):
    spec = _load_spec()
    cases = spec.get("cases")
    if not isinstance(cases, list) or not cases:
        pytest.fail("Smoke spec must contain a non-empty cases list.")

    store = LabelingStore(tmp_path / "labeling_work.sqlite")
    try:
        store.initialize()
        for idx, raw_case in enumerate(cases):
            if not isinstance(raw_case, dict):
                pytest.fail(f"Smoke case {idx} must be a JSON object.")
            name = str(raw_case.get("name") or f"case-{idx}")
            user = str(raw_case.get("user") or "labeler")
            recording_id = str(raw_case.get("recording_id") or f"recording-{idx}")
            task_id = str(raw_case.get("task_id") or f"task-{idx}")
            workflow_kind = str(raw_case.get("workflow_kind") or "")
            scope = raw_case.get("scope") or {}
            requests = raw_case.get("requests") or []
            if not workflow_kind:
                pytest.fail(f"{name}: workflow_kind is required.")
            if not isinstance(scope, dict):
                pytest.fail(f"{name}: scope must be a JSON object.")
            if not isinstance(requests, list) or not requests:
                pytest.fail(f"{name}: requests must be a non-empty list.")

            store.assign_recording(recording_id=recording_id, assignee_user=user)
            store.upsert_task(
                task_id=task_id,
                recording_id=recording_id,
                workflow_kind=workflow_kind,
                dataset_id=raw_case.get("dataset_id"),
                zarr_use=raw_case.get("zarr_use"),
                stage_group=raw_case.get("stage_group"),
                run_name=raw_case.get("run_name"),
                component_name=raw_case.get("component_name"),
                title=raw_case.get("title"),
                scope=scope,
                state=str(raw_case.get("state") or "pending"),
            )

            with _running_server(store, user=user) as base_url:
                open_status, open_payload = _json_request(base_url, f"/api/tasks/{task_id}/open", method="POST", payload={})
                assert open_status == 200, f"{name}: session open failed: {open_payload}"
                session_id = open_payload["session"]["session_id"]
                prefix = _route_prefix(workflow_kind)
                for request_idx, request_spec in enumerate(requests):
                    if not isinstance(request_spec, dict):
                        pytest.fail(f"{name}: request {request_idx} must be a JSON object.")
                    method = str(request_spec.get("method") or "GET").upper()
                    suffix = str(request_spec.get("path") or "").strip()
                    if not suffix.startswith("/"):
                        pytest.fail(f"{name}: request {request_idx} path must start with '/'.")
                    payload = request_spec.get("body") if method != "GET" else None
                    expected_status = int(request_spec.get("expect_status") or 200)
                    status, response = _json_request(
                        base_url,
                        f"/api/sessions/{session_id}/{prefix}{suffix}",
                        method=method,
                        payload=payload,
                    )
                    assert status == expected_status, f"{name}: request {request_idx} failed: {response}"
                    if bool(request_spec.get("expect_ok", True)):
                        assert response.get("ok") is True, f"{name}: request {request_idx} returned non-ok: {response}"
    finally:
        store.close()
