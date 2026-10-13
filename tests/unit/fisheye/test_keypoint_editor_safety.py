"""Keypoint editor: the review control shows the saved state, and unsaved points are never dropped silently."""

import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[3] / "src/fisheye/labeling/static/js/keypoint_editor.js"

HARNESS = r"""
const fs = require("fs"), vm = require("vm"), assert = require("assert");
const nodes = new Map(), requests = [], errors = [], confirms = [], listeners = {};
let confirmAnswer = true;
const pointPayload = Array.from({length:18}, (_,i) => [30+i,40+i]);
function getNode(id) {
  if (!nodes.has(id)) nodes.set(id, {width:640, height:640, innerHTML:"", textContent:"", events:{},
    value: id === "review-state" ? "approved" : "", disabled:false,
    getContext: () => new Proxy({}, {get: () => () => {}}),
    addEventListener(type, handler) {this.events[type] = handler;}});
  return nodes.get(id);
}
const response = {ok:true, roi_idx:0, frame_idx:15, labels:pointPayload.map((_,i)=>`p${i}`),
  frame_index_domain:"legacy_training_sample_row", points:pointPayload,
  roi_image:{shape:[128,128], pixels:Buffer.alloc(128*128).toString("base64")},
  state:{position:0,total:5,refined_run:"seed",review_status:{state:"needs_review"}}};
const context = vm.createContext({console, Number, Math, JSON, Uint8Array,
  window:{PALETTE_KEYPOINT_SESSION_ID:"test",
    addEventListener(type, fn) {(listeners[type]=listeners[type]||[]).push(fn);},
    confirm:(msg)=>{confirms.push(msg); return confirmAnswer;}},
  document:{getElementById:getNode},
  ImageData:class {constructor(w,h) {this.width=w;this.height=h;this.data=new Uint8Array(w*h*4);}},
  atob:v=>Buffer.from(v,"base64").toString("binary"),
  clearOperatorSupport() {}, showOperatorSupport:e=>errors.push(e.message),
  readApiPayload:async r=>r.body, mutationStatusSuffix:()=>"",
  createImageCanvasViewport:()=>({imageWidth:128,imageHeight:128,view:{scale:1},
    hasImage:()=>false, setImageData() {}, fit() {}, constrain() {},
    beginPan:()=>false, panMove:()=>false, endPan() {},
    canvasPoint:e=>[e.x,e.y],canvasToImage:(x,y)=>[x,y]}),
  fetch:async (url, options={})=>{requests.push({url,...options});return {ok:true,body:
    url.endsWith("/save")?{ok:true,result:{roi_idx:0}}:response};}});
const run = code => vm.runInContext(code, context);
const navCount = () => requests.filter(r=>r.url.endsWith("/nav")).length;
const unloadBlocked = () => {let b=false; for (const fn of listeners.beforeunload||[]) fn({preventDefault(){b=true;}}); return b;};
function place(index,x,y) {
  getNode("points").events.click({type:"click",preventDefault(){},
    target:{closest:()=>({dataset:{pointIndex:String(index)}})}});
  getNode("canvas").events.mousedown({x,y,preventDefault(){}});
}
vm.runInContext(fs.readFileSync(process.argv[1],"utf8"), context);
(async()=>{
  await new Promise(resolve=>setImmediate(resolve));
  // Review control shows the saved state and nothing is preselected to apply.
  assert.strictEqual(getNode("review-state").value, "needs_review");
  assert.strictEqual(getNode("review-current").textContent, "Current: needs_review");
  assert.strictEqual(getNode("set-review-button").disabled, true);
  getNode("review-state").value = "approved"; run("reviewSelectChanged()");
  assert.strictEqual(getNode("set-review-button").disabled, false);
  // Clean row: no question asked.
  assert.strictEqual(unloadBlocked(), false);
  await run("nav(1)");
  assert.strictEqual(confirms.length, 0); assert.strictEqual(navCount(), 1);
  // Unsaved point: Cancel keeps it; OK discards and moves.
  place(3, 90, 91);
  assert.strictEqual(unloadBlocked(), true);
  confirmAnswer = false; await run("nav(1)");
  assert.strictEqual(confirms.length, 1); assert.strictEqual(navCount(), 1);
  assert.strictEqual(run("JSON.stringify(points[3])"), "[90,91]");
  confirmAnswer = true; await run("nav(1)");
  assert.strictEqual(confirms.length, 2); assert.strictEqual(navCount(), 2);
  // A saved row is clean again.
  place(3, 92, 93);
  await run("save(false)");
  assert.strictEqual(unloadBlocked(), false);
  await run("nav(1)");
  assert.strictEqual(confirms.length, 2); assert.strictEqual(navCount(), 3);
  assert.strictEqual(errors.length, 0, errors.join("; "));
})().catch(e=>{console.error(e);process.exitCode=1;});
"""


def test_review_control_and_unsaved_point_guard():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is required to execute the browser editor regression")
    result = subprocess.run([node, "-e", HARNESS, str(SCRIPT)], capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stdout + result.stderr
