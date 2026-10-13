"""Execute the shipped mask editor to protect in-flight brush edits."""

import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[3]
    / "src/fisheye/labeling/static/js/subject_mask_editor.js"
)


def browser(case):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is required for the browser regression")
    harness = r"""
const fs=require('fs'),vm=require('vm'),assert=require('assert');
const nodes=new Map(),requests=[];
function el(id){
 if(!nodes.has(id)) nodes.set(id,{id,width:8,height:8,value:'',innerHTML:'',textContent:'',
 style:{},dataset:{},events:{},getContext:()=>new Proxy({}, {get:()=>()=>{}}),
 addEventListener(kind,handler){this.events[kind]=handler;}});
 return nodes.get(id);
}
function roi(position=0,state={}){
 const pixels=Buffer.alloc(64).toString('base64');
 return {ok:true,roi_idx:position,frame_idx:position+10,mask_area_px:0,
 component_name:'subject_body',refined_run:'mask',
 roi_image:{shape:[8,8],pixels},mask:{shape:[8,8],pixels},
 state:{position,total:2,target_token:'token-'+position,edit_revision:0,
 unapplied_session_edit_count:1,pending_apply_effect_count:0,...state}};
}
let current=roi(),fetchImpl=async()=>({ok:true,body:current});
const context=vm.createContext({console,Number,Math,JSON,Uint8Array,setTimeout,clearTimeout,
 window:{PALETTE_SUBJECT_MASK_SESSION_ID:'mask-session',crypto:{randomUUID:()=> 'fixed-apply'},addEventListener(){}},
 document:{getElementById:el,createElement:()=>el('offscreen'),querySelectorAll:()=>[]},
 ImageData:class{constructor(w,h){this.width=w;this.height=h;this.data=new Uint8Array(w*h*4);}},
 requestAnimationFrame:()=>1,atob:v=>Buffer.from(v,'base64').toString('binary'),btoa:v=>Buffer.from(v,'binary').toString('base64'),
 clearOperatorSupport(){},showOperatorSupport(){},mutationStatusSuffix:()=>'',handleTaskCompletionSuccess(){},
 escapeSupportText:v=>String(v).replaceAll('<','&lt;').replaceAll('>','&gt;'),
 readApiPayload:async r=>r.body,apiFailure:(r,d)=>Object.assign(new Error(d.details||d.error),{operatorSupport:{error:d.error}}),
 createImageCanvasViewport:()=>({imageWidth:8,imageHeight:8,view:{scale:1},setImageData(){},fit(){},drawImage(){},
  beginPan:()=>false,panMove:()=>false,endPan(){},handleWheel(){},
  canvasPoint:e=>[e.x,e.y],canvasToImage:(x,y)=>[x,y],imageToCanvas:(x,y)=>[x,y]}),
 fetch:async(url,options={})=>{requests.push({url,...options});return fetchImpl(url,options);}
});
function deferred(){let resolve;const promise=new Promise(r=>resolve=r);return {promise,resolve};}
function tick(){return new Promise(r=>setImmediate(r));}
const offer={tail_refresh_status:'complete',tail_refresh_version:'v2',tail_refresh_tasks:[{task_id:'new-tail-task',workflow_kind:'keypoints'}],
 tail_refresh_valid_rows:2,tail_refresh_training_eligible_rows:2,tail_refresh_manual_point_count:8,tail_refresh_failures:[],tail_refresh_mask_revision:1};
vm.runInContext(fs.readFileSync(process.argv[1],'utf8'),context);
(async()=>{await tick(); __CASE__ })().catch(e=>{console.error(e);process.exitCode=1;});
""".replace("__CASE__", case)
    result = subprocess.run(
        [node, "-e", harness, str(SCRIPT)], capture_output=True, text=True, timeout=20
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_background_apply_preserves_brush_pixels_and_offers_new_task():
    browser(r"""
 const pending=deferred();fetchImpl=async()=>pending.promise;
 const applying=vm.runInContext('applySavedEdits()',context);await tick();
 vm.runInContext('mask[0]=1',context);
 pending.resolve({ok:true,body:{ok:true,result:{...offer,apply_id:'fixed-apply',applied_checkpoint_count:1,qc_status:'complete'},
 state:{...current.state,edit_revision:1,unapplied_session_edit_count:0,qc_status:'complete',tail_refresh:offer}}});
 await applying;
 assert.strictEqual(vm.runInContext('mask[0]',context),1);
 assert.strictEqual(requests.filter(r=>r.url.endsWith('/roi/current')).length,1);
 assert(el('summary').innerHTML.includes('new-tail-task'));
 assert(el('summary').innerHTML.includes('/my-work'));
 assert.strictEqual(vm.runInContext('payload.state.unapplied_session_edit_count',context),0);
""")


def test_apply_cannot_overwrite_newer_navigation_or_checkpoint_state():
    browser(r"""
 const pending=deferred();fetchImpl=async(url)=>{
  if(url.endsWith('/apply'))return pending.promise;
  if(url.endsWith('/nav'))current=roi(1,{unapplied_session_edit_count:3});
  return {ok:true,body:current};
 };
 const applying=vm.runInContext('applySavedEdits()',context);await tick();
 await vm.runInContext('nav(1)',context);
 vm.runInContext('mask[0]=1',context);
 pending.resolve({ok:true,body:{ok:true,result:{...offer,applied_checkpoint_count:1,qc_status:'complete'},
 state:{...roi().state,unapplied_session_edit_count:0,tail_refresh:offer}}});
 await applying;
 assert.strictEqual(vm.runInContext('mask[0]',context),1);
 assert.strictEqual(vm.runInContext('payload.roi_idx',context),1);
 assert.strictEqual(vm.runInContext('payload.state.target_token',context),'token-1');
 assert.strictEqual(vm.runInContext('payload.state.unapplied_session_edit_count',context),3);
""")


def test_pending_effect_message_keeps_reason_and_retry_id_without_pixel_reload():
    browser(r"""
 fetchImpl=async()=>({ok:false,status:400,body:{ok:false,error:'subject_mask_apply_effects_pending',
 details:'Apply the paired keypoint saved checkpoints first.',state:{...current.state,qc_status:'complete',pending_apply_effect_count:1,resumable_apply_id:'fixed-apply'}}});
 vm.runInContext('mask[0]=1',context);
 await vm.runInContext('applySavedEdits()',context);
 assert.strictEqual(vm.runInContext('mask[0]',context),1);
 assert(el('status').textContent.includes('paired keypoint'));
 assert(el('status').textContent.includes('pending'));
 assert.strictEqual(requests.filter(r=>r.url.endsWith('/roi/current')).length,1);
 fetchImpl=async()=>({ok:true,body:{ok:true,result:{...offer,already_applied:true,qc_status:'complete'},
 state:{...current.state,pending_apply_effect_count:0,tail_refresh:offer}}});
 await vm.runInContext('applySavedEdits()',context);
 const attempts=requests.filter(r=>r.url.endsWith('/apply'));
 assert.strictEqual(JSON.parse(attempts[0].body).apply_id,JSON.parse(attempts[1].body).apply_id);
 // The server-provided offer also renders on a fresh page/current-ROI load.
 current=roi(0,{edit_revision:1,tail_refresh:offer});fetchImpl=async()=>({ok:true,body:current});
 vm.runInContext('tailRefreshResult=null',context);await vm.runInContext('loadCurrent()',context);
 assert(el('summary').innerHTML.includes('new-tail-task'));
""")


def test_background_effects_status_line_polls_and_shows_failure_banner():
    browser(r"""
 const timers=[];context.setTimeout=(fn)=>{timers.push(fn);return timers.length;};
 const bgState={...current.state,apply_effects_background:true,edit_revision:1,unapplied_session_edit_count:0,
  pending_apply_effect_count:1,resumable_apply_id:'queued-apply',apply_effects_status:{state:'queued'}};
 fetchImpl=async()=>({ok:true,body:{ok:true,result:{apply_id:'fixed-apply',applied_checkpoint_count:1,effects:'queued',qc_status:'pending'},state:bgState}});
 await vm.runInContext('applySavedEdits()',context);
 assert(el('summary').innerHTML.includes('Updating QC and tail versions'));
 assert(el('status').textContent.includes('background'));
 assert.strictEqual(timers.length,1);
 fetchImpl=async()=>({ok:true,body:{ok:true,state:{...bgState,apply_effects_status:{state:'retrying',reason:'registry <busy>'}}}});
 await timers[0]();
 assert(requests.some(r=>r.url.endsWith('/subject-mask/state')));
 assert(el('summary').innerHTML.includes('Background update failed'));
 assert(el('summary').innerHTML.includes('&lt;busy&gt;'));
 assert(el('summary').innerHTML.includes('Retrying automatically'));
 assert.strictEqual(timers.length,2);
 // A new Apply while effects are owed must not reuse the queued apply_id.
 fetchImpl=async()=>({ok:true,body:{ok:true,result:{apply_id:'fixed-apply',applied_checkpoint_count:1,effects:'queued'},state:bgState}});
 await vm.runInContext('applySavedEdits()',context);
 const applies=requests.filter(r=>r.url.endsWith('/apply'));
 assert.strictEqual(JSON.parse(applies[applies.length-1].body).apply_id,'fixed-apply');
 fetchImpl=async()=>({ok:true,body:{ok:true,state:{...bgState,pending_apply_effect_count:0,apply_effects_status:null,qc_status:'complete'}}});
 await timers[timers.length-1]();
 assert(!el('summary').innerHTML.includes('Updating QC'));
 assert(!el('summary').innerHTML.includes('Background update failed'));
 const settled=timers.length;
 vm.runInContext('renderSummary()',context);
 assert.strictEqual(timers.length,settled);
""")


def test_background_refused_effects_banner_and_retry_reuses_apply_id():
    browser(r"""
 context.setTimeout=()=>1;
 current=roi(0,{apply_effects_background:true,pending_apply_effect_count:1,resumable_apply_id:'refused-apply',
  unapplied_session_edit_count:0,apply_effects_status:{state:'failed',reason:'QC cannot change declared mask_labels'}});
 fetchImpl=async()=>({ok:true,body:current});
 await vm.runInContext('loadCurrent()',context);
 assert(el('summary').innerHTML.includes('Background update failed'));
 assert(el('summary').innerHTML.includes('will not retry automatically'));
 fetchImpl=async()=>({ok:true,body:{ok:true,result:{apply_id:'refused-apply',already_applied:true,qc_status:'complete'},
  state:{...current.state,pending_apply_effect_count:0,apply_effects_status:null}}});
 await vm.runInContext('applySavedEdits()',context);
 const applies=requests.filter(r=>r.url.endsWith('/apply'));
 assert.strictEqual(JSON.parse(applies[0].body).apply_id,'refused-apply');
 assert(!el('summary').innerHTML.includes('Background update failed'));
""")


def test_review_controls_offer_only_needs_review_while_background_update_runs():
    browser(r"""
 const select=el('review-state');
 select.options=['approved','pending','needs_review','rejected'].map(value=>({value,textContent:value,disabled:false}));
 select.value='approved';
 Object.defineProperty(select,'selectedOptions',{get(){return select.options.filter(o=>o.value===select.value);}});
 const bgState={...current.state,apply_effects_background:true,edit_revision:1,unapplied_session_edit_count:0,
  pending_apply_effect_count:1,apply_effects_status:{state:'running'}};
 context.__payload={...current,state:bgState};
 vm.runInContext('payload=__payload; renderSummary()',context);
 const byValue=v=>select.options.find(o=>o.value===v);
 assert.strictEqual(byValue('needs_review').disabled,false);
 for (const v of ['approved','pending','rejected']) assert.strictEqual(byValue(v).disabled,true,v);
 assert.strictEqual(select.value,'needs_review');
 assert(byValue('approved').textContent.includes('after update'));
 assert(el('summary').innerHTML.includes('You can set <b>needs_review</b> now'));
 context.__payload={...current,state:{...bgState,pending_apply_effect_count:0,apply_effects_status:null}};
 vm.runInContext('payload=__payload; renderSummary()',context);
 for (const o of select.options) assert.strictEqual(o.disabled,false,o.value);
 assert.strictEqual(byValue('approved').textContent,'approved');
""")
