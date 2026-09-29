"""Every editor page carries the elements and handlers its scripts use.

Editor scripts find controls by id and templates call script functions from
inline handlers, so a layout change that drops or renames one breaks editing
silently. This renders each real editor page and checks both directions,
plus the static assets the page links.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from fisheye.labeling import web_session_renderers as renderers
from fisheye.labeling import web_static

JS_ROOT = web_static.STATIC_ROOT / "js"
SHARED_JS = ["browser_mutation_status.js", "operator_support.js", "image_canvas_viewport.js"]
EDITORS = {
    "keypoints": ("_keypoint_session_html", "keypoint_editor.js"),
    "subject_mask_component": ("_subject_mask_session_html", "subject_mask_editor.js"),
    "detect_training": ("_detect_session_html", "detect_editor.js"),
    "detect_analysis": ("_video_detect_session_html", "video_detect_editor.js"),
}
SESSION = {
    "session_id": "sess-1",
    "task_id": "task-1",
    "recording_id": "rec-1",
    "user": "alice",
    "title": "Review",
    "expires_at_utc": "2026-09-30T00:00:00Z",
}


def _page(kind: str) -> str:
    renderer, _ = EDITORS[kind]
    return getattr(renderers, renderer)({**SESSION, "workflow_kind": kind}).decode("utf-8")


@pytest.mark.parametrize("kind", sorted(EDITORS))
def test_every_element_the_scripts_look_up_exists(kind):
    html = _page(kind)
    _, editor_js = EDITORS[kind]
    source = "".join((JS_ROOT / name).read_text() for name in [editor_js, *SHARED_JS])
    used = set(re.findall(r'getElementById\("([^"]+)"\)', source))
    ids = re.findall(r'\bid="([^"]+)"', html)
    assert used - set(ids) == set(), kind
    assert len(ids) == len(set(ids)), f"duplicate ids on {kind}"


@pytest.mark.parametrize("kind", sorted(EDITORS))
def test_every_inline_handler_calls_a_defined_function(kind):
    html = _page(kind)
    handlers = set(re.findall(r'\son(?:click|change|input|keydown)="(?:if \([^)]*\) )?([A-Za-z_]\w*)\(', html))
    defined = set(re.findall(r"function\s+([A-Za-z_]\w*)\s*\(", html))
    assert handlers and handlers - defined == set(), kind


@pytest.mark.parametrize("kind", sorted(EDITORS))
def test_linked_static_assets_resolve_and_placeholders_are_filled(kind):
    html = _page(kind)
    assert "@@" not in html
    for url in re.findall(r'(?:href|src)="(/static/[^"]+)"', html):
        build, relative = url.removeprefix("/static/").split("/", 1)
        assert build == web_static.static_build_id(), url
        assert relative in web_static._static_files(), url


@pytest.mark.parametrize("kind", sorted(EDITORS))
def test_session_links_include_the_queue_page(kind):
    html = _page(kind)
    assert 'href="/queue?expected_user=alice"' in html
    assert 'data-session-return="queue"' in html


def test_mask_editor_uses_the_editor_shell():
    html = _page("subject_mask_component")
    assert '<body class="editor">' in html
    assert static_links(html) == ["css/palette.css", "css/editor.css", "js/canvas_stage_fit.js"]
    # The editor's own nav buttons are now present, so first/last-row disabling works.
    assert 'id="nav-prev-button"' in html and 'id="nav-next-button"' in html


def static_links(html: str) -> list[str]:
    return [url.split("/", 3)[3] for url in re.findall(r'(?:href|src)="(/static/[^"]+)"', html)]


def test_canvas_fit_script_parses():
    import shutil
    import subprocess

    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is required")
    result = subprocess.run(
        [node, "--check", str(JS_ROOT / "canvas_stage_fit.js")], capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, result.stderr


def test_shared_session_banner_has_no_inline_styles():
    banner = renderers._session_status_banner({**SESSION, "workflow_kind": "keypoints"})
    assert "style=" not in banner and 'class="session-banner"' in banner
    assert Path(web_static.STATIC_ROOT / "css" / "session_operator_support.css").exists()
