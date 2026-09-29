"""Copy the previous row's mask onto the current row in the shipped mask editor.

Training rows are sampled frames, often far apart, but an animal can hold the
same position for a long time; then the last row's mask is a good start.
"""

from __future__ import annotations

import base64
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

SCRIPT = (
    Path(__file__).resolve().parents[3]
    / "src/fisheye/labeling/static/js/subject_mask_editor.js"
)

# A minimal browser stand-in (as in test_subject_mask_editor_pieces) serving
# several rows: /nav moves the position, /roi/current returns that row, and
# /save records the mask and advances when asked.
HARNESS = r"""
const fs=require("fs"),vm=require("vm");
const nodes=new Map(),frames=[],errors=[],saves=[],listeners={};
const press=(key)=>{for(const fn of listeners.keydown||[])fn({key,target:{tagName:"canvas"},preventDefault(){},ctrlKey:false,metaKey:false,altKey:false,shiftKey:false});};
function surface(){const t={width:0,height:0};const ctx=new Proxy({},{get:(o,k)=>o[k]||(()=>{})});
  return Object.assign(t,{getContext:()=>ctx,style:{},addEventListener(){}});}
function getNode(id){if(!nodes.has(id))nodes.set(id,Object.assign(surface(),{innerHTML:"",textContent:"",value:"overlay",
  disabled:false,dataset:{},classes:new Set(),classList:{toggle(c,on){const n=nodes.get(id);(on===undefined?!n.classes.has(c):on)?n.classes.add(c):n.classes.delete(c);}}}));return nodes.get(id);}
const input=JSON.parse(fs.readFileSync(0,"utf8"));
let position=0;
const rowPayload=()=>{const r=input.rows[position];const [h,w]=r.shape;return {ok:true,roi_idx:r.roi,frame_idx:r.roi,mask_area_px:0,
  roi_image:{shape:[h,w],pixels:Buffer.alloc(h*w,70).toString("base64")},mask:{shape:[h,w],pixels:r.mask},
  state:{position,total:input.rows.length,component_review_completion_guard:{ready:true}}};};
const viewport={imageWidth:0,imageHeight:0,view:{scale:1,offsetX:0,offsetY:0},hasImage:()=>true,setImageData(){},
  drawImage(){},drawCanvas(){},imageToCanvas:(x,y)=>[x,y],pointerEvent:e=>e,canvasPoint:e=>[e.x,e.y],
  canvasToImage:(x,y)=>[x,y],beginPan:()=>false,panMove:()=>false,endPan(){},fit(){}};
const context=vm.createContext({console,Number,Math,JSON,Uint8Array,Int32Array,
  window:{PALETTE_SUBJECT_MASK_SESSION_ID:"test",addEventListener(type,fn){(listeners[type]=listeners[type]||[]).push(fn);},
    requestAnimationFrame:fn=>frames.push(fn),confirm:()=>true},
  document:{getElementById:getNode,createElement:surface,querySelectorAll:()=>[]},
  ImageData:class{constructor(a,b){this.width=a;this.height=b;this.data=new Uint8Array(a*b*4);}},
  atob:v=>Buffer.from(v,"base64").toString("binary"),btoa:v=>Buffer.from(v,"binary").toString("base64"),
  clearOperatorSupport(){},showOperatorSupport:e=>errors.push(e.message),readApiPayload:async r=>r.body,
  mutationStatusSuffix:()=>"",createImageCanvasViewport:()=>viewport,
  fetch:async(url,options={})=>{
    const body=options.body?JSON.parse(options.body):{};
    if(url.endsWith("/nav"))position+=Number(body.delta||0);
    if(url.endsWith("/save")){saves.push({position,mask:body.mask.pixels});
      input.rows[position].mask=body.mask.pixels;if(body.advance)position+=1;
      return {ok:true,body:{ok:true,result:{checkpoint_area_px:0}}};}
    return {ok:true,body:rowPayload()};}});
const run=code=>vm.runInContext(code,context);
const flush=()=>{while(frames.length)frames.shift()();};
vm.runInContext(fs.readFileSync(process.argv[1],"utf8"),context);
const settle=()=>new Promise(r=>setImmediate(r));
(async()=>{
  await settle();await settle();flush();
  const out={};
  for(const step of input.steps){
    if(step.run){await run(step.run);await settle();await settle();}
    if(step.press){press(step.press);await settle();await settle();}
    flush();
    const button=getNode("copy-previous-button");
    out[step.name]={mask:run("Array.from(mask)"),roi:run("payload.roi_idx"),status:getNode("status").textContent,
      copyDisabled:button.disabled,copyLabel:button.textContent,undoDisabled:getNode("undo-bulk-button").disabled,
      unsaved:run("hasUnsavedEdits()"),saves:saves.length};
  }
  out.errors=errors;
  process.stdout.write(JSON.stringify(out));
})().catch(e=>{console.error(e);process.exitCode=1;});
"""


def _row(roi: int, mask: np.ndarray) -> dict:
    return {"roi": roi, "shape": list(mask.shape),
            "mask": base64.b64encode(mask.astype(np.uint8).tobytes()).decode()}


def _run(rows, steps):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is required for the mask editor regression")
    result = subprocess.run(
        [node, "-e", HARNESS, str(SCRIPT)], input=json.dumps({"rows": rows, "steps": steps}),
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    out = json.loads(result.stdout)
    assert out["errors"] == []
    return out


def _mask(shape, box):
    m = np.zeros(shape, np.uint8)
    y0, y1, x0, x1 = box
    m[y0:y1, x0:x1] = 1
    return m


def test_copy_previous_row_mask_is_local_undoable_and_saved_only_on_save():
    a = _mask((12, 16), (2, 6, 3, 9))
    b = _mask((12, 16), (7, 10, 10, 14))
    out = _run([_row(40, a), _row(95, b)], [
        {"name": "first"},
        {"name": "next", "run": "nav(1)"},
        {"name": "copied", "press": "c"},
        {"name": "undone", "run": "undoBulkEdit()"},
        {"name": "again", "run": "copyPreviousMask()"},
        {"name": "saved", "run": "save(false)"},
    ])
    # Nothing to copy on the first row shown.
    assert out["first"]["copyDisabled"] is True and out["first"]["copyLabel"] == "Copy previous mask"
    # On the next row (a non-adjacent sample), the button names the row it copies from.
    assert out["next"]["roi"] == 95 and out["next"]["copyDisabled"] is False
    assert out["next"]["copyLabel"] == "Copy mask from ROI 40"
    # Copy replaces this row's mask with ROI 40's, locally and undoably.
    assert out["copied"]["mask"] == a.ravel().tolist()
    assert out["copied"]["unsaved"] is True and out["copied"]["undoDisabled"] is False
    assert out["copied"]["saves"] == 0 and "ROI 40" in out["copied"]["status"]
    assert out["undone"]["mask"] == b.ravel().tolist() and out["undone"]["unsaved"] is False
    # Only Save persists it.
    assert out["again"]["saves"] == 0
    assert out["saved"]["saves"] == 1 and out["saved"]["mask"] == a.ravel().tolist()
    assert out["saved"]["unsaved"] is False


def test_copy_source_is_the_saved_mask_not_discarded_strokes():
    a = _mask((10, 10), (1, 4, 1, 4))
    b = _mask((10, 10), (6, 9, 6, 9))
    out = _run([_row(3, a), _row(8, b), _row(20, b)], [
        # Save + Next on row 3 after an edit: the saved mask is the source.
        {"name": "edited", "run": "mask[99]=1"},
        {"name": "saved_next", "run": "save(true)"},
        {"name": "copied", "press": "c"},
        # Discard an edit on ROI 8 by moving on: the source is ROI 8 as loaded.
        {"name": "scribble", "run": "mask.fill(1)"},
        {"name": "moved", "run": "nav(1)"},
        {"name": "copied_again", "press": "c"},
    ])
    saved_a = a.ravel().tolist()
    saved_a[99] = 1
    assert out["saved_next"]["roi"] == 8 and out["saved_next"]["copyLabel"] == "Copy mask from ROI 3"
    assert out["copied"]["mask"] == saved_a
    assert out["copied_again"]["roi"] == 20 and out["moved"]["copyLabel"] == "Copy mask from ROI 8"
    assert out["copied_again"]["mask"] == b.ravel().tolist()


def test_different_crop_sizes_are_refused():
    out = _run([_row(1, _mask((10, 12), (1, 3, 1, 3))), _row(2, _mask((14, 12), (5, 8, 5, 8)))], [
        {"name": "next", "run": "nav(1)"},
        {"name": "attempt", "press": "c"},
    ])
    assert out["next"]["copyDisabled"] is True
    assert out["attempt"]["mask"] == _mask((14, 12), (5, 8, 5, 8)).ravel().tolist()
    assert "can't be copied" in out["attempt"]["status"] and out["attempt"]["undoDisabled"] is True
