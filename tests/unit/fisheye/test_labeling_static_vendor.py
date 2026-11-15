"""The vendored Preact and htm modules are the recorded files and load as ES modules.

Provenance and the update procedure live in labeling/static/vendor/README.md.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess

import pytest

from fisheye.labeling import web_static
from fisheye.labeling.web_static import static_url

VENDOR = web_static.STATIC_ROOT / "vendor"

# Update together with the table in vendor/README.md.
PINNED_VENDOR_SHA256 = {
    "preact.module.js": "25a5df7e9f628a587743c4641a368737a3b218f28ef738ba0376a2d5bdfc948c",
    "preact-hooks.module.js": "9e7eff58e0ae604583461eba1344da69a6894eab575eb591c635cc0d3f9ec57d",
    "htm.module.js": "ab33dd3f38059b9be4d5f5350128eefb2356639c4e0bbe9d9e8b3ba75847e9e4",
}


def test_vendored_files_match_pinned_digests_and_ship_licenses():
    served = sorted(p.name for p in VENDOR.glob("*.js"))
    assert served == sorted(PINNED_VENDOR_SHA256)
    for name, digest in PINNED_VENDOR_SHA256.items():
        assert hashlib.sha256((VENDOR / name).read_bytes()).hexdigest() == digest, name
    assert "MIT" in (VENDOR / "LICENSE-preact").read_text()
    assert "Apache License" in (VENDOR / "LICENSE-htm").read_text()


def test_vendored_modules_have_no_bare_imports_or_missing_source_maps():
    for name in PINNED_VENDOR_SHA256:
        text = (VENDOR / name).read_text()
        assert "sourceMappingURL" not in text, name
        assert 'from"preact"' not in text and "from 'preact'" not in text, name


def test_vendored_modules_are_served_by_the_static_route():
    for name in PINNED_VENDOR_SHA256:
        assert static_url(f"vendor/{name}").endswith(f"/vendor/{name}")


def test_htm_builds_preact_vnodes_and_hooks_resolve(tmp_path):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is required to load the vendored modules")
    script = tmp_path / "smoke.mjs"
    script.write_text(
        f"""
import {{ h }} from {json.dumps((VENDOR / "preact.module.js").as_uri())};
import {{ useState, useEffect }} from {json.dumps((VENDOR / "preact-hooks.module.js").as_uri())};
import htm from {json.dumps((VENDOR / "htm.module.js").as_uri())};
const html = htm.bind(h);
const tree = html`<ul class=${{"queue"}}>${{["a", "b"].map((t) => html`<li key=${{t}}>${{t}}</li>`)}}</ul>`;
console.log(JSON.stringify({{
  type: tree.type,
  cls: tree.props.class,
  items: tree.props.children.map((c) => [c.type, c.key, c.props.children]),
  hooks: [typeof useState, typeof useEffect],
}}));
"""
    )
    result = subprocess.run([node, str(script)], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {
        "type": "ul",
        "cls": "queue",
        "items": [["li", "a", "a"], ["li", "b", "b"]],
        "hooks": ["function", "function"],
    }
