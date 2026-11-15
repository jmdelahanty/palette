"""GET /static/<build>/<path> serves labeling assets as immutable, cacheable files."""

from __future__ import annotations

import urllib.error
import urllib.request

import pytest

pytest.importorskip("flask")

from fisheye.labeling import web_static
from fisheye.labeling.assignment_store import LabelingStore
from fisheye.labeling.web_policy import BROWSER_RESPONSE_SECURITY_HEADERS
from fisheye.labeling.web_static import STATIC_CACHE_CONTROL, static_build_id, static_url

from .test_labeling_web_routes import _running_server


def _get(base_url: str, path: str):
    request = urllib.request.Request(f"{base_url}{path}", method="GET")
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            return response.status, dict(response.headers), response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, dict(exc.headers), exc.read()


@pytest.fixture
def base_url(tmp_path):
    # No labeling user: assets need no login and carry no user data.
    with _running_server(LabelingStore(tmp_path / "labeling.sqlite"), user=None) as url:
        yield url


def test_static_url_names_the_file_under_the_tree_build_id():
    url = static_url("js/keypoint_editor.js")
    assert url == f"/static/{static_build_id()}/js/keypoint_editor.js"
    with pytest.raises(KeyError):
        static_url("js/not_a_file.js")


def test_static_asset_is_served_immutable_with_security_headers(base_url):
    status, headers, body = _get(base_url, static_url("js/keypoint_editor.js"))
    assert status == 200
    assert body == (web_static.STATIC_ROOT / "js" / "keypoint_editor.js").read_bytes()
    assert headers["Content-Type"] == "text/javascript; charset=utf-8"
    assert headers["Cache-Control"] == STATIC_CACHE_CONTROL
    assert "Pragma" not in headers and "Expires" not in headers
    for name, value in BROWSER_RESPONSE_SECURITY_HEADERS.items():
        if name not in {"Cache-Control", "Pragma", "Expires"}:
            assert headers[name] == value

    status, headers, _ = _get(base_url, static_url("css/session_operator_support.css"))
    assert status == 200
    assert headers["Content-Type"] == "text/css; charset=utf-8"


@pytest.mark.parametrize(
    "path",
    [
        "/static/0000000000000000/js/keypoint_editor.js",  # stale or wrong build
        "/static/{build}/js/not_a_file.js",
        "/static/{build}/../web.py",
        "/static/{build}/%2e%2e/web.py",
        "/static/{build}/js/../../web.py",
        "/static/{build}",
        "/static",
    ],
)
def test_unknown_stale_or_escaping_paths_are_not_found_and_not_cached(base_url, path):
    status, headers, _ = _get(base_url, path.format(build=static_build_id()))
    assert status == 404
    assert headers["Cache-Control"] == BROWSER_RESPONSE_SECURITY_HEADERS["Cache-Control"]


def test_build_id_changes_when_any_static_file_changes(tmp_path, monkeypatch):
    (tmp_path / "js").mkdir()
    (tmp_path / "js" / "a.js").write_text("export const a = 1;\n")
    (tmp_path / "js" / ".hidden.js").write_text("secret\n")
    (tmp_path / "js" / "notes.txt").write_text("not servable\n")
    monkeypatch.setattr(web_static, "STATIC_ROOT", tmp_path)

    def build_id():
        web_static._static_files.cache_clear()
        web_static.static_build_id.cache_clear()
        return static_build_id()

    try:
        first = build_id()
        assert list(web_static._static_files()) == ["js/a.js"]
        (tmp_path / "js" / "a.js").write_text("export const a = 2;\n")
        second = build_id()
        (tmp_path / "js" / "b.js").write_text("export const b = 1;\n")
        third = build_id()
        assert len({first, second, third}) == 3
    finally:
        monkeypatch.undo()
        web_static._static_files.cache_clear()
        web_static.static_build_id.cache_clear()
