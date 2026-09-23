const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");

const script = fs.readFileSync("src/fisheye/labeling/static/js/subject_mask_editor.js", "utf8");
const nodes = new Map();
function node(id) {
  if (!nodes.has(id)) nodes.set(id, {
    id, style: {}, dataset: {}, innerHTML: "", textContent: "", value: "",
    hidden: false, disabled: false, addEventListener() {},
    getContext() { return {putImageData() {}}; },
  });
  return nodes.get(id);
}
const raw = (values, shape) => ({pixels: Buffer.from(values).toString("base64"), shape});
const initial = {
  ok: true, roi_idx: 0, frame_idx: 15, component_name: "subject_body",
  refined_run: "mask_tail_edit_v1", mask_area_px: 2,
  roi_image: raw([0, 0, 0, 0], [2, 2]), mask: raw([0, 0, 1, 1], [2, 2]),
  tail_crop_border: {
    roi_idx: 0, mask_revision: 1, original_queued_reason: "tail_derivation_failed:body_touches_crop_border",
    accepted: false, stale_acceptance: false, pending_action: {action: "accept", reason: "tiny crop"},
    checkpoint_state: "active", latest_outcome: null,
  },
  state: {position: 0, total: 1, edit_revision: 1, target_token: "row-token", qc_status: "complete"},
};
let applyResolve;
const calls = [];
const context = {
  console, Uint8Array, Uint8ClampedArray, Buffer, setTimeout,
  atob: (value) => Buffer.from(value, "base64").toString("binary"),
  btoa: (value) => Buffer.from(value, "binary").toString("base64"),
  ImageData: class { constructor(width, height) { this.width = width; this.height = height; this.data = new Uint8ClampedArray(width * height * 4); } },
  document: {
    getElementById: node,
    createElement: () => ({width: 0, height: 0, getContext() { return {putImageData() {}}; }}),
    querySelectorAll: () => [],
  },
  window: {
    PALETTE_SUBJECT_MASK_SESSION_ID: "session-one",
    addEventListener() {},
    requestAnimationFrame() {},
    crypto: {randomUUID: () => "apply-one"},
  },
  createImageCanvasViewport: () => ({
    imageWidth: 2, imageHeight: 2, view: {scale: 1},
    setImageData() {}, hasImage: () => false, handleWheel() {}, endPan() {},
  }),
  readApiPayload: async (response) => response.body,
  apiFailure: () => new Error("API failed"),
  escapeSupportText: (text) => String(text),
  mutationStatusSuffix: () => "",
  showOperatorSupport: (error) => { throw error; },
  clearOperatorSupport() {},
  fetch: async (url) => {
    calls.push(url);
    if (url.endsWith("/roi/current")) return {ok: true, body: initial};
    if (url.endsWith("/roi/status")) return {ok: true, body: {
      ok: true, roi_idx: 0,
      tail_crop_border: {
        ...initial.tail_crop_border, mask_revision: 2, accepted: true,
        acceptance: {accepted_by: "reviewer", accepted_at_mask_revision: 2, reason: "tiny crop"},
        pending_action: null, checkpoint_state: null,
        latest_outcome: {mask_revision: 2, status: "derived", reason: "tail_tip_is_visible_crop_endpoint", visible_endpoint: true},
      },
    }};
    if (url.endsWith("/apply")) return new Promise((resolve) => { applyResolve = () => resolve({ok: true, body: {
      ok: true, result: {applied_checkpoint_count: 1, edit_revision_before: 1, edit_revision_after: 2, qc_status: "complete"},
      state: {...initial.state, edit_revision: 2, tail_refresh: {tail_refresh_mask_revision: 2}},
    }}); });
    throw new Error("Unexpected URL: " + url);
  },
};
vm.createContext(context);
vm.runInContext(script, context, {filename: "subject_mask_editor.js"});

(async () => {
  await new Promise(setImmediate);
  node("tail-border-reason").value = "reason currently being typed";
  vm.runInContext("mask[0] = 1; markMaskOverlayDirty();", context);
  assert.match(node("tail-border-status").innerHTML, /unsaved painted pixels/i);
  assert.equal(node("tail-border-reason").value, "reason currently being typed");
  const apply = vm.runInContext("applySavedEdits()", context);
  await new Promise(setImmediate);
  applyResolve();
  await apply;
  assert.equal(vm.runInContext("mask[0]", context), 1, "background Apply must preserve local paint");
  assert.equal(vm.runInContext("payload.state.edit_revision", context), 2);
  assert.match(node("tail-border-status").innerHTML, /visible crop-edge endpoint/);
  assert.match(node("tail-border-status").innerHTML, /unsaved painted pixels/i);
  assert.equal(node("tail-border-reason").value, "reason currently being typed");
  assert(calls.some((url) => url.endsWith("/roi/status")), "Apply must refresh row outcome");
})().catch((error) => { console.error(error); process.exitCode = 1; });
