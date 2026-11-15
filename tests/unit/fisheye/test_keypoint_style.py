"""keypoint_style.js: category colours from landmark names, left solid / right hollow."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from fisheye.labeling import web_static

ROOT = Path(__file__).resolve().parents[3]
SCHEMA = ROOT / "configs" / "fisheye" / "pose_schemas" / "head_tail11_fins_v2.json"


def _styles(labels):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is required")
    script = f"""
const vm = require("vm");
const context = {{window: {{}}}};
vm.runInNewContext(require("fs").readFileSync({json.dumps(str(web_static.STATIC_ROOT / "js" / "keypoint_style.js"))}, "utf8"), context);
const styles = context.window.keypointStyles({json.dumps(labels)});
console.log(JSON.stringify({{styles, groups: context.window.keypointGroups(styles)}}));
"""
    result = subprocess.run([node, "-e", script], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def test_real_schema_gets_category_families_and_paired_sides():
    labels = [node["name"] for node in json.loads(SCHEMA.read_text())["nodes"]]
    out = _styles(labels)
    by_name = dict(zip(labels, out["styles"]))
    assert {s["category"] for s in out["styles"]} == {"head", "tail", "fin", "snout"}
    # Pairs share a colour; left is solid, right is a hollow ring.
    for left, right in [("eye_left", "eye_right"),
                        ("left_pectoral_fin_tip", "right_pectoral_fin_tip"),
                        ("left_pectoral_fin_insertion", "right_pectoral_fin_insertion")]:
        assert by_name[left]["color"] == by_name[right]["color"]
        assert (by_name[left]["hollow"], by_name[right]["hollow"]) == (False, True)
    assert by_name["left_pectoral_fin_tip"]["color"] != by_name["left_pectoral_fin_insertion"]["color"]
    assert by_name["swim_bladder"]["hollow"] is False and by_name["snout_tip"]["category"] == "snout"
    # Tail: 11 distinct shades from dark base to light tip, brightness increasing.
    tail = [by_name[name]["color"] for name in labels if name.startswith("tail")]
    assert len(tail) == 11 and len(set(tail)) == 11
    assert tail[0] == "#7a3a00" and tail[-1] == "#ffb35c"
    brightness = [sum(int(c[i:i + 2], 16) for i in (1, 3, 5)) for c in tail]
    assert brightness == sorted(brightness)
    assert [g["name"] for g in out["groups"]] == ["Head", "Snout", "Tail", "Fins"]
    assert sum(len(g["indices"]) for g in out["groups"]) == len(labels) == 19


def test_unknown_names_fall_back_to_distinct_palette_colours():
    out = _styles(["alpha", "beta", "gamma"])
    assert [s["category"] for s in out["styles"]] == ["other"] * 3
    assert len({s["color"] for s in out["styles"]}) == 3
    assert [g["name"] for g in out["groups"]] == ["Other"]
