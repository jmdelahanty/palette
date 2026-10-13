"""Execute the shipped editor JS: null landmarks must never become (0,0)."""

import shutil
import subprocess
from pathlib import Path

import pytest


def test_browser_missing_points_place_reset_and_save():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is required to execute the browser editor regression")
    script = (
        Path(__file__).resolve().parents[3]
        / "src/fisheye/labeling/static/js/keypoint_editor.js"
    )
    harness = r"""
const fs = require("fs"), vm = require("vm"), assert = require("assert");
const nodes = new Map(), requests = [], errors = [];
const pointPayload = Array.from({length:18}, (_,i) => i<14 ? [30+i,40+i] : [null,null]);
function getNode(id) {
  if (!nodes.has(id)) nodes.set(id, {width:640, height:640, innerHTML:"", events:{},
    getContext: () => new Proxy({}, {get: () => () => {}}),
    addEventListener(type, handler) {this.events[type] = handler;}});
  return nodes.get(id);
}
const response = {ok:true, roi_idx:0, frame_idx:15, labels:pointPayload.map((_,i)=>`p${i}`),
  frame_index_domain:"legacy_training_sample_row", points:pointPayload,
  roi_image:{shape:[128,128], pixels:Buffer.alloc(128*128).toString("base64")},
  state:{position:0,total:1,refined_run:"seed"}};
const context = vm.createContext({console, Number, Math, JSON, Uint8Array,
  window:{PALETTE_KEYPOINT_SESSION_ID:"test", addEventListener() {}},
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
vm.runInContext(fs.readFileSync(process.argv[1],"utf8"), context);
(async()=>{
  await new Promise(resolve=>setImmediate(resolve));
  assert(getNode("points").innerHTML.includes("missing"));
  assert(getNode("summary").innerHTML.includes("source training row"));
  await vm.runInContext("save(false)", context);
  assert(errors.pop().includes("missing landmark"));
  assert(!requests.some(r=>r.url.endsWith("/save")));
  function place(index,x,y) {
    getNode("points").events.click({type:"click",preventDefault(){},
      target:{closest:()=>({dataset:{pointIndex:String(index)}})}});
    getNode("canvas").events.mousedown({x,y,preventDefault(){}});
  }
  // A missing selected fin must be placed even when the click is near a head point.
  place(14,31,41);
  vm.runInContext("resetPoints()",context);
  assert(vm.runInContext("Number.isNaN(points[14][0])",context));
  for(let i=14;i<18;i++) place(i,100+i-14,70);
  await vm.runInContext("save(false)",context);
  const saved=JSON.parse(requests.find(r=>r.url.endsWith("/save")).body).points;
  assert.deepStrictEqual(saved.slice(0,14),pointPayload.slice(0,14));
  assert.deepStrictEqual(saved.slice(14),[[100,70],[101,70],[102,70],[103,70]]);
  assert.strictEqual(errors.length,0);
})().catch(e=>{console.error(e);process.exitCode=1;});
"""
    result = subprocess.run(
        [node, "-e", harness, str(script)], capture_output=True, text=True, timeout=20
    )
    assert result.returncode == 0, result.stdout + result.stderr
