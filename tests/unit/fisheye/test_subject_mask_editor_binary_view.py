"""Exercise display switching and edits through the shipped mask editor."""

import shutil
import subprocess
from pathlib import Path

import pytest


def test_binary_pixels_edits_and_view_switch_preserve_mask():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is required for the mask editor rendering regression")
    script = (
        Path(__file__).resolve().parents[3]
        / "src/fisheye/labeling/static/js/subject_mask_editor.js"
    )
    harness = r"""
const fs = require("fs"), vm = require("vm"), assert = require("assert");
const nodes = new Map(), errors = [], requests = [], frames = [];
let rendered;
function surface() {
  const target = {width:0,height:0,pixels:null};
  const ctx = new Proxy({
    putImageData(image,x,y) {
      if (!target.pixels || target.pixels.length !== target.width*target.height*4)
        target.pixels = new Uint8Array(target.width*target.height*4);
      for (let row=0; row<image.height; row++) {
        const start=((y+row)*target.width+x)*4;
        target.pixels.set(image.data.subarray(row*image.width*4,(row+1)*image.width*4),start);
      }
    }
  }, {get:(obj,key)=>obj[key] || (()=>{})});
  return Object.assign(target,{getContext:()=>ctx,style:{},events:{},
    addEventListener(type,handler) {this.events[type]=handler;}});
}
function getNode(id) {
  if (!nodes.has(id)) nodes.set(id,Object.assign(surface(),{
    innerHTML:"",value:"overlay",dataset:{},classList:{toggle(){}}}));
  return nodes.get(id);
}
const pixels = [1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1];
let response = {ok:true,roi_idx:0,frame_idx:0,mask_area_px:3,
  roi_image:{shape:[4,4],pixels:Buffer.alloc(16,70).toString("base64")},
  mask:{shape:[4,4],pixels:Buffer.from(pixels).toString("base64")},
  state:{position:0,total:2,component_review_completion_guard:{ready:true}}};
const viewport = {imageWidth:4,imageHeight:4,view:{scale:2,offsetX:1,offsetY:1},
  hasImage:()=>true,setImageData(){},drawImage(){},drawCanvas(c){rendered=c.pixels.slice();},
  imageToCanvas:(x,y)=>[x,y],pointerEvent:e=>e,canvasPoint:e=>[e.x,e.y],
  canvasToImage:(x,y)=>[x,y],beginPan:()=>false,panMove:()=>false,endPan(){}};
const context=vm.createContext({console,Number,Math,JSON,Uint8Array,
  window:{PALETTE_SUBJECT_MASK_SESSION_ID:"test",addEventListener(){},
    requestAnimationFrame:fn=>frames.push(fn)},
  document:{getElementById:getNode,createElement:surface,querySelectorAll:()=>[]},
  ImageData:class {constructor(w,h){this.width=w;this.height=h;this.data=new Uint8Array(w*h*4);}},
  atob:v=>Buffer.from(v,"base64").toString("binary"),
  btoa:v=>Buffer.from(v,"binary").toString("base64"),
  clearOperatorSupport(){},showOperatorSupport:e=>errors.push(e.message),
  readApiPayload:async r=>r.body,createImageCanvasViewport:()=>viewport,
  fetch:async (url,options={})=>{requests.push({url,...options});return {ok:true,body:response};}});
const run=code=>vm.runInContext(code,context);
function flush(){while(frames.length)frames.shift()();}
function rgba(index){return Array.from(rendered.slice(index*4,index*4+4));}
function maskBytes(){return Array.from(Buffer.from(run("encodeMaskPayload().pixels"),"base64"));}
vm.runInContext(fs.readFileSync(process.argv[1],"utf8"),context);
(async()=>{
  await new Promise(resolve=>setImmediate(resolve));flush();
  assert.deepStrictEqual(rgba(0),[0,200,148,118]);
  assert.deepStrictEqual(rgba(1),[0,0,0,0]);
  const viewBefore=JSON.stringify(viewport.view), requestCount=requests.length;
  run("setMaskView('binary')");flush();
  for(let i=0;i<16;i++)assert.deepStrictEqual(rgba(i),pixels[i]?[255,255,255,255]:[0,0,0,255]);
  assert.deepStrictEqual(maskBytes(),pixels);
  assert.strictEqual(JSON.stringify(viewport.view),viewBefore);
  assert.strictEqual(requests.length,requestCount); // switching never checkpoints
  // Actual brush path must update partial rectangles in binary mode.
  run("brushSize=1; tool='erase'; paintAt({x:0,y:0,shiftKey:false})");flush();
  assert.deepStrictEqual(rgba(0),[0,0,0,255]);
  assert.deepStrictEqual(rgba(15),[255,255,255,255]);
  run("tool='paint'; paintAt({x:2,y:1,shiftKey:false})");flush();
  assert.deepStrictEqual(rgba(6),[255,255,255,255]);
  const edited=maskBytes();
  run("setMaskView('overlay')");flush();
  assert.deepStrictEqual(rgba(0),[0,0,0,0]);
  assert.deepStrictEqual(rgba(6),[0,200,148,118]);
  assert.deepStrictEqual(maskBytes(),edited);
  run("setMaskView('binary')");flush();
  assert.deepStrictEqual(maskBytes(),edited);
  // A newly loaded ROI rebuilds every background pixel without resetting mode.
  response={...response,roi_idx:1,mask:{shape:[4,4],pixels:Buffer.alloc(16).toString("base64")}};
  await run("loadCurrent()");flush();
  for(let i=0;i<16;i++)assert.deepStrictEqual(rgba(i),[0,0,0,255]);
  assert.strictEqual(getNode("mask-view").value,"binary");
  assert.strictEqual(errors.length,0);
})().catch(e=>{console.error(e);process.exitCode=1;});
"""
    result = subprocess.run(
        [node, "-e", harness, str(script)], capture_output=True, text=True, timeout=20
    )
    assert result.returncode == 0, result.stdout + result.stderr
