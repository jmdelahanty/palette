"""The personal queue pages' JavaScript parses and builds support text lazily.

The page scripts lived inside Python strings until 2026-09-28, where a single
"\\n" became a real newline and silently broke the whole script (as it did on
/my-work from 2026-07-04 until 2026-09-27). They now live in
templates/personal/*.html; the pinned digests below prove that move served
identical bytes.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess

import pytest

from fisheye.labeling.web_personal_renderers import _dashboard_html, _datasets_html

SCRIPT_RE = re.compile(r"<script(?![^>]*\bsrc=)[^>]*>([\s\S]*?)</script>")

# sha256 of the served pages when they moved out of Python (PR #233). An
# intentional page edit must update the digest here in the same change.
PINNED_PAGE_SHA256 = {
    "my-work": "fd445371c3aac4997effa49426902b892df0d34cd3696fb59deecd8fce59fb5d",
    "my-datasets": "0f71791494a91ae05abacdea3d5ef2f6f0516dc079dcd7042e98ad91e7a6c06f",
}


@pytest.mark.parametrize(
    ("page_id", "page"),
    [("my-work", _dashboard_html), ("my-datasets", _datasets_html)],
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


@pytest.mark.parametrize("page", [_dashboard_html, _datasets_html], ids=["my-work", "my-datasets"])
def test_every_inline_script_parses(page, tmp_path):
    node = _node()
    scripts = SCRIPT_RE.findall(page().decode("utf-8"))
    assert scripts
    for index, script in enumerate(scripts):
        path = tmp_path / f"script_{index}.js"
        path.write_text(script)
        result = subprocess.run([node, "--check", str(path)], capture_output=True, text=True)
        assert result.returncode == 0, result.stderr


HARNESS = r"""
const vm = require("vm");
const {scripts, payload} = JSON.parse(require("fs").readFileSync(0, "utf8"));
const els = new Map(), copied = [];
function el(id) {
  if (!els.has(id)) els.set(id, new Proxy({id, innerHTML:"", textContent:"", value:"", className:"", dataset:{}, style:{},
    classList:{add(){}, remove(){}, toggle(){}, contains(){return false}}, setAttribute(){}, getAttribute(){return null},
    addEventListener(){}, querySelector(){return null}, querySelectorAll(){return []}, appendChild(){}, remove(){}, closest(){return null}},
    {get:(t,k)=>k in t ? t[k] : (typeof k === "string" ? ()=>{} : undefined), set:(t,k,v)=>{t[k]=v; return true;}}));
  return els.get(id);
}
const loc = {search:"?expected_user=u", pathname:"/my-datasets", href:"http://x/my-datasets?expected_user=u", origin:"http://x", hash:""};
const ctx = vm.createContext({console, JSON, Math, Number, String, Object, Array, Set, Map, Promise, Date,
  encodeURIComponent, decodeURIComponent, URL, URLSearchParams, setTimeout, clearTimeout,
  window:{location:loc, addEventListener(){}, setTimeout:()=>0, history:{replaceState(){}}}, location:loc,
  navigator:{clipboard:{writeText:async t=>{copied.push(t);}}},
  document:{getElementById:el, querySelector:()=>el("q"), querySelectorAll:()=>[], createElement:()=>el("tmp"+Math.random()),
    body:el("body"), addEventListener(){}, readyState:"complete", title:""},
  fetch:async()=>({ok:true, status:200, json:async()=>payload, text:async()=>JSON.stringify(payload)})});
for (const s of scripts) vm.runInContext(s, ctx);
(async () => {
  await new Promise(r => setTimeout(r, 30));
  const html = [...els.values()].map(e => e.innerHTML || "").join("");
  const keys = [...html.matchAll(/data-support-key="(\d+)"/g)].map(m => m[1]);
  for (const k of keys) await vm.runInContext(`copySupportByKey({dataset:{supportKey:"${k}"}, textContent:""}, "x")`, ctx);
  process.stdout.write(JSON.stringify({html, keys, copied}));
})().catch(e => { console.error(e); process.exitCode = 1; });
"""


def test_my_datasets_builds_support_text_only_when_copied():
    node = _node()
    task = {"task_id": "task-7", "recording_id": "rec-1", "workflow_kind": "keypoints", "state": "pending",
            "operator_support": {"task_id": "task-7", "dataset_id": "ds-1", "recording_id": "rec-1"}}
    recording = {"recording_id": "rec-1", "open_task_count": 1, "tasks": [task],
                 "operator_support": {"recording_id": "rec-1", "dataset_id": "ds-1"}}
    dataset = {"dataset_id": "ds-1", "open_task_count": 1, "task_count": 1, "recording_count": 1,
               "recordings": [recording], "operator_support": {"dataset_id": "ds-1"}}
    payload = {"ok": True, "user": "u", "dataset_queue": [dataset], "datasets": [dataset]}
    scripts = SCRIPT_RE.findall(_datasets_html().decode("utf-8"))
    result = subprocess.run([node, "-e", HARNESS], input=json.dumps({"scripts": scripts, "payload": payload}),
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    out = json.loads(result.stdout)
    assert len(out["keys"]) == 3  # dataset, recording and task Copy buttons
    assert len(out["copied"]) == 3 and all(text for text in out["copied"])
    assert any("task-7" in text for text in out["copied"])
    for text in out["copied"]:
        assert text not in out["html"]  # built on click, never embedded in the page
    assert "data-support-details" not in out["html"]
