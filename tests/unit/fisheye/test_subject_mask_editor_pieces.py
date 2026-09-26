"""Stray-piece detection, removal, and hole fill in the shipped mask editor."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from scipy import ndimage

SCRIPT = (
    Path(__file__).resolve().parents[3]
    / "src/fisheye/labeling/static/js/subject_mask_editor.js"
)

# Loads the editor in a minimal browser stand-in (as the binary-view test
# does) and runs the JSON commands read from stdin against it.
HARNESS = r"""
const fs=require("fs"),vm=require("vm");
const nodes=new Map(),frames=[],errors=[],requests=[];
function surface(){const t={width:0,height:0};const ctx=new Proxy({},{get:(o,k)=>o[k]||(()=>{})});
  return Object.assign(t,{getContext:()=>ctx,style:{},addEventListener(){}});}
function getNode(id){if(!nodes.has(id))nodes.set(id,Object.assign(surface(),{innerHTML:"",textContent:"",value:"overlay",
  disabled:false,dataset:{},classes:new Set(),classList:{toggle(c,on){const n=nodes.get(id);(on===undefined?!n.classes.has(c):on)?n.classes.add(c):n.classes.delete(c);}}}));return nodes.get(id);}
const input=JSON.parse(fs.readFileSync(0,"utf8"));
const [h,w]=input.shape;
let response={ok:true,roi_idx:0,frame_idx:0,mask_area_px:0,
  roi_image:{shape:[h,w],pixels:Buffer.alloc(h*w,70).toString("base64")},
  mask:{shape:[h,w],pixels:Buffer.from(input.mask).toString("base64")},
  state:{position:0,total:1,component_review_completion_guard:{ready:true}}};
const viewport={imageWidth:w,imageHeight:h,view:{scale:1,offsetX:0,offsetY:0},hasImage:()=>true,setImageData(){},
  drawImage(){},drawCanvas(){},imageToCanvas:(x,y)=>[x,y],pointerEvent:e=>e,canvasPoint:e=>[e.x,e.y],
  canvasToImage:(x,y)=>[x,y],beginPan:()=>false,panMove:()=>false,endPan(){},fit(){}};
const context=vm.createContext({console,Number,Math,JSON,Uint8Array,Int32Array,
  window:{PALETTE_SUBJECT_MASK_SESSION_ID:"test",addEventListener(){},requestAnimationFrame:fn=>frames.push(fn)},
  document:{getElementById:getNode,createElement:surface,querySelectorAll:()=>[]},
  ImageData:class{constructor(a,b){this.width=a;this.height=b;this.data=new Uint8Array(a*b*4);}},
  atob:v=>Buffer.from(v,"base64").toString("binary"),btoa:v=>Buffer.from(v,"binary").toString("base64"),
  clearOperatorSupport(){},showOperatorSupport:e=>errors.push(e.message),readApiPayload:async r=>r.body,
  createImageCanvasViewport:()=>viewport,
  fetch:async(url,options={})=>{requests.push({url,...options});return {ok:true,body:response};}});
const run=code=>vm.runInContext(code,context);
const flush=()=>{while(frames.length)frames.shift()();};
const maskBytes=()=>Array.from(Buffer.from(run("encodeMaskPayload().pixels"),"base64"));
vm.runInContext(fs.readFileSync(process.argv[1],"utf8"),context);
(async()=>{
  await new Promise(r=>setImmediate(r));flush();
  const out={};
  for(const step of input.steps){
    if(step.run)run(step.run);
    flush();
    out[step.name]={mask:maskBytes(),pieces:getNode("mask-pieces").textContent,
      warn:getNode("mask-pieces").classes.has("warn"),removeDisabled:getNode("remove-stray-button").disabled,
      undoDisabled:getNode("undo-bulk-button").disabled,status:getNode("status").textContent};
  }
  if(input.pure)out.pure=run(input.pure);
  out.errors=errors;
  process.stdout.write(JSON.stringify(out));
})().catch(e=>{console.error(e);process.exitCode=1;});
"""


def _node():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is required for the mask editor regression")
    return node


def _run(mask: np.ndarray, steps, pure=None):
    payload = {"shape": list(mask.shape), "mask": mask.astype(np.uint8).ravel().tolist(),
               "steps": steps, "pure": pure}
    result = subprocess.run(
        [_node(), "-e", HARNESS, str(SCRIPT)], input=json.dumps(payload),
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    out = json.loads(result.stdout)
    assert out["errors"] == []
    return out


def _as_mask(values, shape):
    return np.asarray(values, dtype=np.uint8).reshape(shape)


def test_piece_count_and_hole_fill_match_the_server_definitions():
    rng = np.random.default_rng(7)
    for trial in range(6):
        mask = (rng.random((23, 31)) < 0.45).astype(np.uint8)
        labels, count = ndimage.label(mask, structure=np.ones((3, 3)))
        seed = tuple(int(v) for v in np.argwhere(labels == np.bincount(labels.ravel())[1:].argmax() + 1)[0])
        pure = (
            "(()=>{const info=maskPieces(mask,maskWidth,maskHeight);"
            "const filled=mask.slice();"
            f"const n=fillPieceHoles(filled,maskWidth,maskHeight,{seed[1]},{seed[0]});"
            "return {count:info.count,sizes:info.sizes.slice(1),filled:Array.from(filled),n};})()"
        )
        out = _run(mask, [], pure=pure)["pure"]
        assert out["count"] == count
        assert sorted(out["sizes"]) == sorted(np.bincount(labels.ravel())[1:].tolist())
        piece = labels == labels[seed]
        expected = mask.astype(bool) | ndimage.binary_fill_holes(piece)
        np.testing.assert_array_equal(_as_mask(out["filled"], mask.shape).astype(bool), expected)
        assert out["n"] == int(expected.sum() - mask.astype(bool).sum())


def test_stray_pixel_is_reported_removed_and_undone():
    mask = np.zeros((40, 50), np.uint8)
    mask[10:30, 15:35] = 1
    mask[2, 47] = 1  # A one-pixel stray piece far from the body.
    mask[37, 3] = 1
    out = _run(mask, [
        {"name": "loaded"},
        {"name": "removed", "run": "removeStrayPiecesAction()"},
        {"name": "undone", "run": "undoBulkEdit()"},
    ])
    assert out["loaded"]["warn"] and out["loaded"]["pieces"].startswith("2 stray pieces")
    assert out["loaded"]["removeDisabled"] is False and out["loaded"]["undoDisabled"] is True
    body = mask.copy(); body[2, 47] = 0; body[37, 3] = 0
    np.testing.assert_array_equal(_as_mask(out["removed"]["mask"], mask.shape), body)
    assert out["removed"]["pieces"] == "Mask is one piece." and out["removed"]["removeDisabled"] is True
    assert out["removed"]["undoDisabled"] is False
    np.testing.assert_array_equal(_as_mask(out["undone"]["mask"], mask.shape), mask)
    assert out["undone"]["undoDisabled"] is True


def test_fill_holes_tool_fills_only_the_clicked_piece():
    mask = np.zeros((30, 40), np.uint8)
    mask[5:20, 5:20] = 1
    mask[9:14, 9:14] = 0  # Hole in the body.
    mask[22:28, 30:38] = 1
    mask[24:26, 33:35] = 0  # Hole in a separate piece: left alone.
    out = _run(mask, [
        {"name": "miss", "run": "setTool('fill'); beginCanvasEdit({x:1,y:1,preventDefault(){}})"},
        {"name": "filled", "run": "beginCanvasEdit({x:6,y:6,preventDefault(){}})"},
        {"name": "undone", "run": "undoBulkEdit()"},
    ])
    np.testing.assert_array_equal(_as_mask(out["miss"]["mask"], mask.shape), mask)
    assert "Click on a mask piece" in out["miss"]["status"]
    expected = mask.copy(); expected[9:14, 9:14] = 1
    np.testing.assert_array_equal(_as_mask(out["filled"]["mask"], mask.shape), expected)
    assert "Filled 25 px" in out["filled"]["status"]
    np.testing.assert_array_equal(_as_mask(out["undone"]["mask"], mask.shape), mask)
