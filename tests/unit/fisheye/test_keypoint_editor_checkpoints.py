"""Execute checkpoint save/apply behavior in the shipped browser editor JS."""

import shutil
import subprocess
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[3]
    / "src/fisheye/labeling/static/js/keypoint_editor.js"
)


def _run_browser_case(case: str) -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is required to execute the browser editor regression")
    harness = r"""
const fs = require("fs"), vm = require("vm"), assert = require("assert");
const nodes = new Map(), windowEvents = {}, requests = [], errors = [];
function getNode(id) {
  if (!nodes.has(id)) nodes.set(id, {
    id, width:640, height:640, innerHTML:"", textContent:"", className:"",
    hidden:false, disabled:false, value:id === "review-state" ? "approved" : "",
    events:{}, getContext:()=>new Proxy({}, {get:()=>()=>{}}),
    addEventListener(type, handler) { this.events[type] = handler; }
  });
  return nodes.get(id);
}
function roi(position, token, state={}) {
  return {
    ok:true, roi_idx:position, frame_idx:100 + position,
    labels:Array.from({length:19}, (_,i)=>i === 18 ? "snout_19" : `p${i + 1}`),
    points:Array.from({length:19}, (_,i)=>[30+i, 40+i]),
    roi_image:{shape:[128,128], pixels:Buffer.alloc(128*128).toString("base64")},
    state:{position,total:3,refined_run:"seed",target_token:token,
      checkpoint_save_supported:true,save_mode:"checkpoint_v1",
      unapplied_session_edit_count:0,active_session_edit_count:0,
      applying_session_edit_count:0,checkpoint_snapshot_sha256:null,
      apply_available:false,...state}
  };
}
let initial = roi(0, "token-0");
let fetchImpl = async (url) => ({ok:true, body:initial});
const context = vm.createContext({
  console, Number, Math, JSON, Uint8Array, setTimeout, clearTimeout, initial, roi,
  window:{PALETTE_KEYPOINT_SESSION_ID:"test", crypto:{randomUUID:()=>"apply-fixed"},
    addEventListener(type, handler) { windowEvents[type] = handler; }},
  document:{getElementById:getNode},
  ImageData:class {constructor(w,h) {this.width=w;this.height=h;this.data=new Uint8Array(w*h*4);}},
  atob:v=>Buffer.from(v,"base64").toString("binary"),
  clearOperatorSupport() {}, showOperatorSupport:e=>errors.push(e.message),
  apiFailure:(response, data, fallback)=>new Error(data?.error || fallback),
  readApiPayload:async response=>response.body,
  mutationStatusSuffix:()=>"", handleTaskCompletionSuccess() {},
  createImageCanvasViewport:()=>({imageWidth:128,imageHeight:128,view:{scale:1},
    hasImage:()=>true,setImageData() {},fit() {},constrain() {},drawImage() {},
    beginPan:()=>false,panMove:()=>false,endPan() {},handleWheel() {},
    canvasPoint:e=>[e.x,e.y],canvasToImage:(x,y)=>[x,y],imageToCanvas:(x,y)=>[x,y]}),
  fetch:async (url, options={}) => {
    const request = {url, ...options}; requests.push(request);
    return fetchImpl(url, options, request);
  }
});
function deferred() {
  let resolve, reject;
  const promise = new Promise((yes,no)=>{resolve=yes;reject=no;});
  return {promise,resolve,reject};
}
function tick() { return new Promise(resolve=>setImmediate(resolve)); }
vm.runInContext(fs.readFileSync(process.argv[1], "utf8"), context);
(async()=>{
  await tick();
  __CASE__
})().catch(error=>{console.error(error);process.exitCode=1;});
""".replace("__CASE__", case)
    result = subprocess.run(
        [node, "-e", harness, str(SCRIPT)],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_checkpoint_save_is_single_flight_and_restores_controls_after_failure() -> None:
    _run_browser_case(
        r"""
  const pending = deferred();
  fetchImpl = async url => {
    if (url.endsWith("/save")) return pending.promise;
    return {ok:true,body:initial};
  };
  vm.runInContext(`payload.state = Object.assign(payload.state, {
    unapplied_session_edit_count:1,active_session_edit_count:1,
    checkpoint_snapshot_sha256:${JSON.stringify("a".repeat(64))},apply_available:true
  }); renderSummary()`,context);
  const first = vm.runInContext("save(false)", context);
  const second = vm.runInContext("save(false)", context);
  vm.runInContext("applyCheckpoints()",context);
  windowEvents.keydown({key:"s",preventDefault(){},target:{tagName:"BODY"}});
  assert.strictEqual(requests.filter(r=>r.url.endsWith("/save")).length, 1);
  assert.strictEqual(getNode("save-button").disabled, true);
  assert.strictEqual(getNode("nav-next-button").disabled, true);
  // Foreground Save also guards navigation and canvas edits.
  vm.runInContext("nav(1)", context);
  getNode("canvas").events.mousedown({x:90,y:91,preventDefault(){}});
  assert.strictEqual(requests.filter(r=>r.url.endsWith("/nav")).length, 0);
  assert.strictEqual(requests.filter(r=>r.url.endsWith("/apply")).length, 0);
  assert.deepStrictEqual(vm.runInContext("points[0]", context), [30,40]);
  pending.reject(new Error("connection lost"));
  await Promise.all([first, second]); await tick();
  assert.strictEqual(getNode("save-button").disabled, false);
  assert.strictEqual(getNode("nav-next-button").disabled, false);
  assert(errors.includes("connection lost"));
"""
    )


def test_checkpoint_save_folds_returned_roi_and_reports_pending_without_get() -> None:
    _run_browser_case(
        r"""
  requests.length = 0;
  const next = roi(1,"token-1",{
    unapplied_session_edit_count:1,active_session_edit_count:1,
    checkpoint_snapshot_sha256:"a".repeat(64),apply_available:true
  });
  fetchImpl = async url => {
    assert(url.endsWith("/save"));
    return {ok:true,body:{ok:true,result:{roi_idx:0,saved:true,applied:false,
      canonical_zarr_mutated:false,checkpoint_id:"checkpoint-1",operation:"points"},
      state:next.state,roi:next}};
  };
  vm.runInContext("setActivePoint(18); viewport.view.scale = 3",context);
  await vm.runInContext("save(true)", context);
  assert.strictEqual(requests.length,1);
  assert(requests[0].url.endsWith("/save"));
  assert.strictEqual(vm.runInContext("payload.roi_idx",context),1);
  assert.strictEqual(vm.runInContext("activePoint",context),18);
  assert.strictEqual(vm.runInContext("viewport.view.scale",context),3);
  assert(getNode("points").innerHTML.includes("snout_19"));
  assert(getNode("status").textContent.includes("Checkpoint saved"));
  assert(getNode("status").textContent.includes("not"));
  assert(getNode("checkpoint-status").textContent.includes("1"));
  assert.strictEqual(getNode("apply-button").hidden,false);
  assert.strictEqual(getNode("apply-button").disabled,false);
  assert.strictEqual(getNode("set-review-button").disabled,true);
  assert.strictEqual(getNode("complete-task-button").disabled,true);
  await vm.runInContext("setReviewStatus()",context);
  await vm.runInContext("completeTask()",context);
  assert(!requests.some(r=>r.url.endsWith("/review-status")));
  assert(!requests.some(r=>r.url.endsWith("/complete")));
"""
    )


def test_background_apply_never_clobbers_new_navigation_or_edits() -> None:
    _run_browser_case(
        r"""
  initial.state = {...initial.state,unapplied_session_edit_count:1,
    active_session_edit_count:1,checkpoint_snapshot_sha256:"a".repeat(64),apply_available:true};
  // Re-render the state changed by this case without another request.
  vm.runInContext("payload.state = Object.assign(payload.state, initial.state); renderSummary()", context);
  requests.length = 0;
  const applying = deferred();
  fetchImpl = async (url, options) => {
    if (url.endsWith("/apply")) return applying.promise;
    if (url.endsWith("/nav")) return {ok:true,body:{ok:true,result:{}}};
    if (url.endsWith("/roi/current")) return {ok:true,body:roi(1,"token-1",{
      unapplied_session_edit_count:1,active_session_edit_count:0,
      applying_session_edit_count:1,checkpoint_snapshot_sha256:"b".repeat(64),
      apply_available:true})};
    if (url.endsWith("/save")) {
      const saved=roi(1,"token-1",{unapplied_session_edit_count:2,
        active_session_edit_count:1,applying_session_edit_count:1,
        checkpoint_snapshot_sha256:"b".repeat(64),apply_available:true});
      return {ok:true,body:{ok:true,
        result:{roi_idx:1,saved:true,applied:false,canonical_zarr_mutated:false},
        state:saved.state,roi:saved}};
    }
    throw new Error(url);
  };
  const applyPromise = vm.runInContext("applyCheckpoints()", context);
  assert.strictEqual(getNode("apply-button").disabled,true);
  await vm.runInContext("nav(1)", context);
  // A checkpoint for another row is permitted while Apply remains in flight.
  await vm.runInContext("save(false)", context);
  vm.runInContext("setActivePoint(0)",context);
  getNode("canvas").events.mousedown({x:91,y:92,preventDefault(){}});
  const edited = vm.runInContext("points[0].slice()",context);
  const old = roi(0,"token-0",{unapplied_session_edit_count:0,
    active_session_edit_count:0,applying_session_edit_count:0,
    checkpoint_snapshot_sha256:null,apply_available:false});
  applying.resolve({ok:true,body:{ok:true,result:{apply_id:"apply-fixed",
    checkpoint_snapshot_sha256:"a".repeat(64),applied_checkpoint_count:1,
    saved:true,applied:true,canonical_zarr_mutated:true},state:old.state,roi:old}});
  await applyPromise;
  assert.strictEqual(vm.runInContext("payload.roi_idx",context),1);
  assert.strictEqual(vm.runInContext("payload.state.target_token",context),"token-1");
  assert.strictEqual(vm.runInContext("payload.state.unapplied_session_edit_count",context),2);
  assert.strictEqual(vm.runInContext("payload.state.checkpoint_snapshot_sha256",context),"b".repeat(64));
  assert.deepStrictEqual(vm.runInContext("points[0]",context),edited);
  assert.strictEqual(requests.filter(r=>r.url.endsWith("/apply")).length,1);
  assert.strictEqual(requests.filter(r=>r.url.endsWith("/save")).length,1);
  assert.strictEqual(getNode("apply-button").disabled,false);
  assert(getNode("status").textContent.includes("Applied"));
  assert(getNode("status").textContent.includes("2 pending"));
"""
    )


def test_apply_response_during_save_cannot_break_folded_save_adoption() -> None:
    _run_browser_case(
        r"""
  initial.state = {...initial.state,unapplied_session_edit_count:1,
    active_session_edit_count:1,checkpoint_snapshot_sha256:"a".repeat(64),apply_available:true};
  vm.runInContext("payload.state = Object.assign(payload.state, initial.state); renderSummary()",context);
  requests.length=0;
  const applying=deferred(), saving=deferred();
  fetchImpl=async url => {
    if (url.endsWith("/apply")) return applying.promise;
    if (url.endsWith("/save")) return saving.promise;
    throw new Error(url);
  };
  const applyPromise=vm.runInContext("applyCheckpoints()",context);
  const savePromise=vm.runInContext("save(true)",context);
  const old=roi(0,"token-0",{unapplied_session_edit_count:0,
    active_session_edit_count:0,checkpoint_snapshot_sha256:null,apply_available:false});
  applying.resolve({ok:true,body:{ok:true,result:{apply_id:"apply-fixed",
    checkpoint_snapshot_sha256:"a".repeat(64),applied_checkpoint_count:1,
    saved:true,applied:true,canonical_zarr_mutated:true},state:old.state,roi:old}});
  await applyPromise;
  assert.strictEqual(vm.runInContext("payload.state.target_token",context),"token-0");
  const next=roi(1,"token-1",{unapplied_session_edit_count:1,
    active_session_edit_count:1,checkpoint_snapshot_sha256:"b".repeat(64),apply_available:true});
  saving.resolve({ok:true,body:{ok:true,result:{roi_idx:0,saved:true,applied:false,
    canonical_zarr_mutated:false},state:next.state,roi:next}});
  await savePromise;
  assert.strictEqual(vm.runInContext("payload.roi_idx",context),1);
  assert.strictEqual(vm.runInContext("payload.state.target_token",context),"token-1");
  assert.strictEqual(vm.runInContext("payload.state.checkpoint_snapshot_sha256",context),"b".repeat(64));
"""
    )


def test_checkpoint_actions_fold_roi_and_report_server_selected_save_mode() -> None:
    _run_browser_case(
        r"""
  requests.length=0;
  const checkpointed=roi(1,"token-1",{unapplied_session_edit_count:1,
    active_session_edit_count:1,checkpoint_snapshot_sha256:"a".repeat(64),apply_available:true});
  fetchImpl=async url => {
    assert(url.endsWith("/action"));
    return {ok:true,body:{ok:true,result:{roi_idx:0,saved:true,applied:false,
      canonical_zarr_mutated:false,operation:"mark_no_keypoints"},
      state:checkpointed.state,roi:checkpointed}};
  };
  await vm.runInContext("action('mark_no_keypoints')",context);
  assert.strictEqual(requests.length,1);
  assert.strictEqual(vm.runInContext("payload.roi_idx",context),1);
  assert(getNode("status").textContent.includes("Checkpoint saved"));
  assert(getNode("status").textContent.includes("not changed"));

  const direct=roi(2,"token-2",{checkpoint_save_supported:false,
    save_mode:"immutable_delta_direct_v1",immutable_base:true,
    edit_storage:"delta_generation",apply_available:false});
  context.direct=direct;
  vm.runInContext("payload.state = Object.assign(payload.state, direct.state); renderSummary()",context);
  requests.length=0;
  fetchImpl=async url => ({ok:true,body:{ok:true,result:{roi_idx:1,saved:true,
    applied:true,canonical_zarr_mutated:false,operation:"mark_detection_issue"},
    state:direct.state,roi:direct}});
  await vm.runInContext("action('mark_detection_issue')",context);
  assert.strictEqual(requests.length,1);
  assert(getNode("status").textContent.includes("immutable delta"));
  assert.strictEqual(getNode("apply-controls").hidden,true);
  assert.strictEqual(getNode("immutable-direct-save-note").hidden,false);
"""
    )


def test_apply_retry_keeps_bound_id_and_snapshot_and_recovers_after_reload() -> None:
    _run_browser_case(
        r"""
  initial.state = {...initial.state,unapplied_session_edit_count:1,
    active_session_edit_count:1,checkpoint_snapshot_sha256:"a".repeat(64),apply_available:true};
  vm.runInContext("payload.state = Object.assign(payload.state, initial.state); renderSummary()", context);
  requests.length = 0;
  let attempts = 0;
  fetchImpl = async (url, options) => {
    if (!url.endsWith("/apply")) throw new Error(url);
    attempts += 1;
    if (attempts === 1) throw new Error("response lost");
    const requestBody=JSON.parse(options.body);
    return {ok:true,body:{ok:true,result:{apply_id:requestBody.apply_id,
      checkpoint_snapshot_sha256:requestBody.checkpoint_snapshot_sha256,
      applied_checkpoint_count:1,saved:true,
      applied:true,canonical_zarr_mutated:true},state:{...initial.state,
      unapplied_session_edit_count:0,active_session_edit_count:0,
      checkpoint_snapshot_sha256:null,apply_available:false}}};
  };
  await vm.runInContext("applyCheckpoints()", context);
  assert.strictEqual(getNode("apply-button").disabled,false);
  await vm.runInContext("applyCheckpoints()", context);
  const bodies=requests.map(r=>JSON.parse(r.body));
  assert.strictEqual(bodies[0].apply_id,bodies[1].apply_id);
  assert.strictEqual(bodies[0].checkpoint_snapshot_sha256,bodies[1].checkpoint_snapshot_sha256);
  assert.strictEqual(bodies[0].checkpoint_snapshot_sha256,"a".repeat(64));

  // Recovery state pairs an old apply ID with its own digest, even if a new
  // active checkpoint changed the current snapshot digest.
  vm.runInContext(`payload.state = Object.assign(payload.state, {
    unapplied_session_edit_count:2,active_session_edit_count:1,
    pending_apply_effect_count:1,selected_session_edit_count:1,
    checkpoint_snapshot_sha256:${JSON.stringify("b".repeat(64))},apply_available:true,
    resumable_apply_id:"apply-recovered",
    resumable_checkpoint_snapshot_sha256:${JSON.stringify("a".repeat(64))}
  }); renderSummary()`,context);
  assert.strictEqual(getNode("checkpoint-status").textContent,
    "Labels are applied; finish recording this Apply.");
  assert.strictEqual(getNode("apply-button").textContent,"Finish Apply");
  assert(getNode("apply-help").textContent.includes("already applied"));
  requests.length=0;
  await vm.runInContext("applyCheckpoints()",context);
  const recovered=JSON.parse(requests[0].body);
  assert.strictEqual(recovered.apply_id,"apply-recovered");
  assert.strictEqual(recovered.checkpoint_snapshot_sha256,"a".repeat(64));
"""
    )


def test_safe_prewrite_rejection_starts_fresh_apply_id_and_snapshot() -> None:
    _run_browser_case(
        r"""
  initial.state = {...initial.state,unapplied_session_edit_count:1,
    active_session_edit_count:1,checkpoint_snapshot_sha256:"a".repeat(64),apply_available:true};
  vm.runInContext("payload.state = Object.assign(payload.state, initial.state); renderSummary()",context);
  const ids=["apply-stale","apply-fresh"];
  context.window.crypto.randomUUID=()=>ids.shift();
  requests.length=0;
  let attempts=0;
  fetchImpl=async (url,options) => {
    attempts += 1;
    const body=JSON.parse(options.body);
    if (attempts === 1) return {ok:false,status:409,body:{ok:false,
      error:"keypoint_checkpoint_snapshot_conflict",
      details:"checkpoint snapshot changed before the canonical write",
      apply_retry_disposition:"fresh_snapshot_required",
      safe_prewrite_rejection:true,retain_apply_id:false,
      state:{...initial.state,unapplied_session_edit_count:2,
        active_session_edit_count:2,checkpoint_snapshot_sha256:"b".repeat(64),
        apply_available:true}}};
    return {ok:true,body:{ok:true,result:{apply_id:body.apply_id,
      checkpoint_snapshot_sha256:body.checkpoint_snapshot_sha256,
      applied_checkpoint_count:2,saved:true,applied:true,
      canonical_zarr_mutated:true},state:{...initial.state,
      unapplied_session_edit_count:0,active_session_edit_count:0,
      checkpoint_snapshot_sha256:null,apply_available:false}}};
  };
  await vm.runInContext("applyCheckpoints()",context);
  assert.strictEqual(vm.runInContext("payload.state.checkpoint_snapshot_sha256",context),"b".repeat(64));
  assert.strictEqual(getNode("apply-button").disabled,false);
  assert(getNode("status").textContent.includes("Start Apply again"));
  await vm.runInContext("applyCheckpoints()",context);
  const bodies=requests.map(request=>JSON.parse(request.body));
  assert.strictEqual(bodies[0].apply_id,"apply-stale");
  assert.strictEqual(bodies[0].checkpoint_snapshot_sha256,"a".repeat(64));
  assert.strictEqual(bodies[1].apply_id,"apply-fresh");
  assert.strictEqual(bodies[1].checkpoint_snapshot_sha256,"b".repeat(64));
"""
    )


def test_wrong_digest_for_owned_apply_adopts_server_recovery_pair() -> None:
    _run_browser_case(
        r"""
  initial.state = {...initial.state,unapplied_session_edit_count:1,
    applying_session_edit_count:1,checkpoint_snapshot_sha256:"a".repeat(64),
    apply_available:true,resumable_apply_id:"apply-owned",
    resumable_checkpoint_snapshot_sha256:"a".repeat(64)};
  vm.runInContext("payload.state = Object.assign(payload.state, initial.state);"
    + " uncertainApplyAttempt={applyId:'apply-owned',snapshotDigest:'0'.repeat(64)};"
    + " renderSummary()",context);
  requests.length=0;
  let attempts=0;
  fetchImpl=async (url,options) => {
    attempts += 1;
    const body=JSON.parse(options.body);
    if (attempts === 1) return {ok:false,status:409,body:{ok:false,
      error:"keypoint_apply_conflict",details:"apply ID belongs to another snapshot",
      apply_retry_disposition:"fresh_snapshot_required",
      safe_prewrite_rejection:true,retain_apply_id:false,state:initial.state}};
    return {ok:true,body:{ok:true,result:{apply_id:body.apply_id,
      checkpoint_snapshot_sha256:body.checkpoint_snapshot_sha256,
      applied_checkpoint_count:1,saved:true,applied:true,
      canonical_zarr_mutated:true},state:{...initial.state,
      unapplied_session_edit_count:0,applying_session_edit_count:0,
      checkpoint_snapshot_sha256:null,apply_available:false,
      resumable_apply_id:null,resumable_checkpoint_snapshot_sha256:null}}};
  };
  await vm.runInContext("applyCheckpoints()",context);
  assert.strictEqual(getNode("apply-button").disabled,false);
  await vm.runInContext("applyCheckpoints()",context);
  const bodies=requests.map(request=>JSON.parse(request.body));
  assert.strictEqual(bodies[0].apply_id,"apply-owned");
  assert.strictEqual(bodies[0].checkpoint_snapshot_sha256,"0".repeat(64));
  assert.strictEqual(bodies[1].apply_id,"apply-owned");
  assert.strictEqual(bodies[1].checkpoint_snapshot_sha256,"a".repeat(64));
"""
    )


def test_immutable_direct_delta_hides_apply_and_labels_save_as_direct() -> None:
    _run_browser_case(
        r"""
  initial.state = {...initial.state,checkpoint_save_supported:false,
    save_mode:"immutable_delta_direct_v1",
    immutable_base:true,edit_storage:"delta_generation",delta_run:"review",
    delta_generation:"g1",checkpoint_snapshot_sha256:null,apply_available:false};
  vm.runInContext("payload.state = Object.assign(payload.state, initial.state); renderSummary()", context);
  assert.strictEqual(getNode("apply-controls").hidden,true);
  assert.strictEqual(getNode("immutable-direct-save-note").hidden,false);
  assert.strictEqual(getNode("save-button").textContent,"Save direct delta");
  assert(getNode("checkpoint-status").textContent.includes("direct immutable delta"));
  requests.length=0;
  fetchImpl = async url => ({ok:true,body:{ok:true,
    result:{roi_idx:0,saved:true,applied:true,canonical_zarr_mutated:false,operation:"points"},
    state:initial.state,roi:initial}});
  await vm.runInContext("save(false)",context);
  assert(getNode("status").textContent.includes("immutable delta"));
  assert(!requests.some(r=>r.url.endsWith("/apply")));
"""
    )
