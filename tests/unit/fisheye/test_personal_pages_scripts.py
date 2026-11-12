"""The /my-work page's JavaScript parses and builds support text lazily.

The page scripts lived inside Python strings until 2026-09-28, where a single
"\\n" became a real newline and silently broke the whole script (as it did on
/my-work from 2026-07-04 until 2026-09-27). They now live in
templates/personal/*.html; the pinned digest below proves that move served
identical bytes. The /my-datasets page was retired for /queue on 2026-10-14.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess

import pytest

from fisheye.labeling.web_personal_renderers import _dashboard_html

SCRIPT_RE = re.compile(r"<script(?![^>]*\bsrc=)[^>]*>([\s\S]*?)</script>")

# sha256 of the served pages when they moved out of Python (PR #233). An
# intentional page edit must update the digest here in the same change.
PINNED_PAGE_SHA256 = {
    "my-work": "fd445371c3aac4997effa49426902b892df0d34cd3696fb59deecd8fce59fb5d",
}


@pytest.mark.parametrize(
    ("page_id", "page"),
    [("my-work", _dashboard_html)],
)
def test_served_page_bytes_match_pinned_digest(page_id, page):
    assert hashlib.sha256(page()).hexdigest() == PINNED_PAGE_SHA256[page_id], (
        f"/{page_id} bytes changed; if intentional, update PINNED_PAGE_SHA256"
    )


def _node():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is required to check the page scripts")
    return node


@pytest.mark.parametrize("page", [_dashboard_html], ids=["my-work"])
def test_every_inline_script_parses(page, tmp_path):
    node = _node()
    scripts = SCRIPT_RE.findall(page().decode("utf-8"))
    assert scripts
    for index, script in enumerate(scripts):
        path = tmp_path / f"script_{index}.js"
        path.write_text(script)
        result = subprocess.run([node, "--check", str(path)], capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
