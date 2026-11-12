from __future__ import annotations

from fisheye.labeling import web as labeling_web


class _Handler:
    def __init__(self, headers: dict[str, str]) -> None:
        self.headers = headers


def test_same_origin_allows_requests_without_browser_origin_headers():
    assert labeling_web._request_has_same_origin(_Handler({}))
    assert labeling_web._request_has_same_origin(_Handler({"Host": "labeling.example.org"}))


def test_same_origin_accepts_matching_origin_or_referer():
    assert labeling_web._request_has_same_origin(
        _Handler({"Host": "labeling.example.org", "Origin": "https://labeling.example.org"})
    )
    assert labeling_web._request_has_same_origin(
        _Handler({"Host": "labeling.example.org", "Referer": "https://labeling.example.org/r/session-a"})
    )
    assert labeling_web._request_has_same_origin(
        _Handler({"Host": "127.0.0.1:8795", "Origin": "http://127.0.0.1:8795"})
    )


def test_same_origin_rejects_mismatched_origin_or_referer():
    assert not labeling_web._request_has_same_origin(
        _Handler({"Host": "labeling.example.org", "Origin": "https://evil.example.org"})
    )
    assert not labeling_web._request_has_same_origin(
        _Handler({"Host": "labeling.example.org", "Referer": "https://evil.example.org/r/session-a"})
    )
    assert not labeling_web._request_has_same_origin(
        _Handler({"Host": "labeling.example.org", "Origin": "null"})
    )


def test_same_origin_uses_forwarded_host_when_present():
    assert labeling_web._request_has_same_origin(
        _Handler(
            {
                "Host": "127.0.0.1:8795",
                "X-Forwarded-Host": "labeling.example.org",
                "Origin": "https://labeling.example.org",
            }
        )
    )
    assert not labeling_web._request_has_same_origin(
        _Handler(
            {
                "Host": "127.0.0.1:8795",
                "X-Forwarded-Host": "labeling.example.org",
                "Origin": "https://127.0.0.1:8795",
            }
        )
    )


def _flask_mutating_routes(monkeypatch, tmp_path):
    """Every non-GET rule on the Flask app a real server is built with."""

    from fisheye.labeling import web_app
    from fisheye.labeling.assignment_store import LabelingStore

    captured = []
    original = labeling_web.create_labeling_app

    def capture(**kwargs):
        app = original(**kwargs)
        captured.append(app)
        return app

    monkeypatch.setattr(labeling_web, "create_labeling_app", capture)
    store = LabelingStore(tmp_path / "labeling_work.sqlite")
    store.initialize()
    config = labeling_web.ServerConfig(
        store_path=store.path, host="127.0.0.1", port=0, fixed_user="admin",
        auth_header="X-Forwarded-User", session_ttl_seconds=600, admin_users=("admin",),
    )
    labeling_web._make_handler(labeling_web.ServerState(store=store, config=config))
    (app,) = captured
    routes = sorted(
        (rule.rule, method)
        for rule in app.url_map.iter_rules()
        for method in rule.methods - web_app.SAFE_METHODS
    )
    return store, routes


def test_every_mutating_flask_route_rejects_cross_origin_requests(monkeypatch, tmp_path):
    import json
    import threading
    import urllib.error
    import urllib.request
    from http.server import ThreadingHTTPServer

    store, routes = _flask_mutating_routes(monkeypatch, tmp_path)
    monkeypatch.undo()
    # The two admin routes had no check before the app-wide guard existed.
    assert ("/api/admin/admin-reviews", "POST") in routes
    assert ("/api/admin/keypoint-corrections", "POST") in routes
    config = labeling_web.ServerConfig(
        store_path=store.path, host="127.0.0.1", port=0, fixed_user="admin",
        auth_header="X-Forwarded-User", session_ttl_seconds=600, admin_users=("admin",),
    )
    server = ThreadingHTTPServer(
        ("127.0.0.1", 0), labeling_web._make_handler(labeling_web.ServerState(store=store, config=config))
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        for rule, method in routes:
            path = rule.replace("<flag_id>", "missing-flag")
            assert "<" not in path, f"add a sample value for {rule}"
            for origin, refused in (("http://evil.example", True), (f"http://{host}:{port}", False)):
                request = urllib.request.Request(
                    f"http://{host}:{port}{path}", data=b"{}", method=method,
                    headers={"Content-Type": "application/json", "Origin": origin},
                )
                try:
                    with urllib.request.urlopen(request, timeout=10) as response:
                        status, body = response.status, response.read()
                except urllib.error.HTTPError as exc:
                    status, body = exc.code, exc.read()
                error = json.loads(body).get("error") if body else None
                if refused:
                    assert (status, error) == (403, "same_origin_required"), (rule, method)
                else:
                    assert error != "same_origin_required", (rule, method)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        store.close()
