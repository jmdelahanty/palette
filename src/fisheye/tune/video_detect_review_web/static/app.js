const state = {
  bboxNorm: null,
  bboxRect: null,
  framePayload: null,
  serverState: null,
  currentVideoUrl: null,
  messageTimeout: null,
  renderer: null,
  rendererLabel: "Renderer: detecting...",
  rendererPreference: new URLSearchParams(window.location.search).get("renderer") || "webgpu",
  frameCache: new Map(),
  pendingFrameFetches: new Map(),
  pendingEdits: new Map(),
  clipOptionsSignature: "",
  isSaving: false,
  saveProgressText: "",
  playback: {
    isPlaying: false,
    rafId: null,
    videoId: null,
    clipId: null,
    parentOffset: 0,
    lastFrame: null,
    lastRenderedFrame: null,
  },
  image: {
    width: 1,
    height: 1,
  },
  viewport: {
    scale: 1,
    offsetX: 0,
    offsetY: 0,
    isPanning: false,
    panStartX: 0,
    panStartY: 0,
    drawStart: null,
    moveStart: null,
    moveStartBox: null,
  },
};

const pane = document.getElementById("viewer-pane");
const video = document.getElementById("source-video");
const gpuCanvas = document.getElementById("gpu-canvas");
const overlay = document.getElementById("overlay");
const overlayCtx = overlay.getContext("2d");
const stateLine = document.getElementById("state-line");
const statusLine = document.getElementById("status-line");
const readbackLine = document.getElementById("readback-line");
const dirtyLine = document.getElementById("dirty-line");
const saveProgressLine = document.getElementById("save-progress-line");
const editModeLine = document.getElementById("edit-mode-line");
const messages = document.getElementById("messages");
const rendererLine = document.getElementById("renderer-line");
const clipControls = document.getElementById("clip-controls");
const clipSelect = document.getElementById("clip-select");
const frameInput = document.getElementById("frame-input");
const playBtn = document.getElementById("play-btn");
const saveBtn = document.getElementById("save-btn");
const clearBtn = document.getElementById("clear-btn");
const prevIssueBtn = document.getElementById("prev-issue-btn");
const nextIssueBtn = document.getElementById("next-issue-btn");
const nextLowConfBtn = document.getElementById("next-low-conf-btn");
const nextManualBtn = document.getElementById("next-manual-btn");
const lowConfidenceInput = document.getElementById("low-confidence-input");
const speedSelect = document.getElementById("speed-select");
const autoAdvanceBox = document.getElementById("auto-advance-box");
const PLAYBACK_RATES = [0.25, 0.5, 1, 1.5, 2, 3, 4, 8];

function setRendererLabel(text) {
  state.rendererLabel = text;
  updateRendererLine();
}

function currentPlaybackRate() {
  const rate = Number(speedSelect?.value || video.playbackRate || 1);
  if (!Number.isFinite(rate) || rate <= 0) {
    return 1;
  }
  return Math.min(Math.max(rate, 0.05), 16);
}

function applyPlaybackRate() {
  const rate = currentPlaybackRate();
  video.playbackRate = rate;
  updateRendererLine();
}

function stepPlaybackRate(delta) {
  const current = currentPlaybackRate();
  let index = PLAYBACK_RATES.findIndex((rate) => rate >= current - 1e-6);
  if (index < 0) {
    index = PLAYBACK_RATES.indexOf(1);
  }
  const nextIndex = Math.min(Math.max(index + delta, 0), PLAYBACK_RATES.length - 1);
  const next = PLAYBACK_RATES[nextIndex];
  if (speedSelect) {
    speedSelect.value = String(next);
  }
  applyPlaybackRate();
}

function updateRendererLine() {
  const viewport = state.viewport;
  rendererLine.textContent = [
    state.rendererLabel,
    `video=${video.videoWidth || 0}x${video.videoHeight || 0}`,
    `t=${Number(video.currentTime || 0).toFixed(3)}s`,
    `rate=${Number(video.playbackRate || 1).toFixed(2)}x`,
    `ready=${video.readyState}`,
    `dpr=${Number(devicePixelRatio || 1).toFixed(2)}`,
    `view=${state.image.width}x${state.image.height}@${viewport.scale.toFixed(3)}+${viewport.offsetX.toFixed(1)},${viewport.offsetY.toFixed(1)}`,
  ].join(" | ");
}

function setMessage(text, isError = false) {
  if (state.messageTimeout) {
    clearTimeout(state.messageTimeout);
  }
  messages.textContent = text;
  messages.style.color = isError ? "#fca5a5" : "#86efac";
  state.messageTimeout = setTimeout(() => {
    messages.textContent = "";
  }, 2600);
}

function isEditable() {
  return Boolean(state.serverState?.summary?.editable);
}

function isPromotionEnabled() {
  return Boolean(state.serverState?.promotion_hook?.enabled);
}

function currentFrameNumber() {
  return Math.trunc(Number(state.framePayload?.parent_frame_index ?? frameInput.value ?? 0));
}

function numberOrNull(value) {
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

function availableClipSources() {
  const videos = Array.isArray(state.serverState?.videos) ? state.serverState.videos : [];
  return videos
    .filter((source) => source && (source.clip_id || source.parent_frame_start !== undefined))
    .slice()
    .sort((a, b) => {
      const aStart = numberOrNull(a.parent_frame_start);
      const bStart = numberOrNull(b.parent_frame_start);
      if (aStart !== null && bStart !== null && aStart !== bStart) {
        return aStart - bStart;
      }
      return String(a.clip_id || a.video_id || "").localeCompare(String(b.clip_id || b.video_id || ""));
    });
}

function frameRangeText(startValue, endValue, prefix) {
  const start = numberOrNull(startValue);
  const end = numberOrNull(endValue);
  if (start === null && end === null) {
    return null;
  }
  if (start !== null && end !== null && start !== end) {
    return `${prefix} ${start}-${end}`;
  }
  return `${prefix} ${start ?? end}`;
}

function formatClipOption(source, index) {
  const label = source.clip_id || source.video_id || `video_${index}`;
  const camera = source.camera_serial ? `cam ${source.camera_serial}` : null;
  const parentRange = frameRangeText(source.parent_frame_start, source.parent_frame_end, "parent");
  const localRange = frameRangeText(source.source_frame_start, source.source_frame_end, "local");
  return [label, camera, parentRange, localRange].filter(Boolean).join(" | ");
}

function updateClipSelector(payload) {
  if (!clipControls || !clipSelect) {
    return;
  }
  const summary = state.serverState?.summary || {};
  const sources = availableClipSources();
  const visible = summary.mode === "clipped" && sources.length > 1;
  clipControls.hidden = !visible;
  if (!visible) {
    state.clipOptionsSignature = "";
    return;
  }

  const signature = sources
    .map((source) => [
      source.video_id,
      source.clip_id,
      source.camera_serial,
      source.parent_frame_start,
      source.parent_frame_end,
      source.source_frame_start,
      source.source_frame_end,
    ].join(":"))
    .join("|");
  if (signature !== state.clipOptionsSignature) {
    clipSelect.replaceChildren();
    sources.forEach((source, index) => {
      const option = document.createElement("option");
      option.value = String(source.video_id || "");
      option.textContent = formatClipOption(source, index);
      clipSelect.appendChild(option);
    });
    state.clipOptionsSignature = signature;
  }

  const payloadVideoId = String(payload.video_id || "");
  if (payloadVideoId) {
    clipSelect.value = payloadVideoId;
  }
}

function cloneBboxNorm(bboxNorm) {
  if (!bboxNorm) {
    return null;
  }
  return bboxNorm.map((value) => Number(value));
}

function pendingEditCount() {
  return state.pendingEdits.size;
}

function updateDirtyControls() {
  const count = pendingEditCount();
  const editable = isEditable();
  if (dirtyLine) {
    dirtyLine.classList.toggle("dirty", count > 0);
    dirtyLine.textContent = count > 0 ? `Unsaved edits: ${count} frame${count === 1 ? "" : "s"}` : "";
  }
  if (saveProgressLine) {
    saveProgressLine.classList.toggle("saving", state.isSaving);
    saveProgressLine.textContent = state.isSaving ? state.saveProgressText || "Saving..." : "";
  }
  if (saveBtn) {
    saveBtn.classList.toggle("saving", state.isSaving);
    saveBtn.setAttribute("aria-busy", state.isSaving ? "true" : "false");
    saveBtn.disabled = !editable || state.isSaving;
    if (state.isSaving) {
      saveBtn.textContent = count > 0 ? `Saving ${count}...` : "Saving...";
      saveBtn.title = "Save is already in progress.";
    } else {
      saveBtn.textContent = count > 0 ? `Save Pending (${count})` : "Save";
      saveBtn.title = editable
        ? (count > 0 ? `Save ${count} pending frame edit${count === 1 ? "" : "s"}.` : "Save the current box.")
        : "Read-only mode. Restart with --edit to save boxes.";
    }
  }
}

function setSavingState(isSaving, text = "") {
  state.isSaving = Boolean(isSaving);
  state.saveProgressText = state.isSaving ? String(text || "Saving...") : "";
  updateEditableControls();
}

function markCurrentDirty() {
  if (!isEditable()) {
    return;
  }
  const frame = currentFrameNumber();
  state.pendingEdits.set(frame, {
    frame,
    bbox_norm: cloneBboxNorm(state.bboxNorm),
  });
  updateDirtyControls();
}

function updateEditableControls() {
  const editable = isEditable();
  const promotionEnabled = isPromotionEnabled();
  if (saveBtn) {
    saveBtn.disabled = !editable || state.isSaving;
  }
  if (clearBtn) {
    clearBtn.disabled = !editable || state.isSaving;
    clearBtn.title = state.isSaving
      ? "Save is in progress."
      : (editable ? "Clear the current box locally; save to persist." : "Read-only mode. Restart with --edit to clear boxes.");
  }
  if (!editModeLine) {
    return;
  }
  editModeLine.classList.toggle("editable", editable);
  editModeLine.classList.toggle("readonly", !editable);
  editModeLine.textContent = editable
    ? `Editable mode: box edits save to the analysis Zarr${promotionEnabled ? " and promote to training." : "."}`
    : "Read-only mode: launch with --edit to enable save/clear writes.";
  updateDirtyControls();
}

function isFormControl(target) {
  const tagName = target && target.tagName ? target.tagName.toUpperCase() : "";
  return tagName === "INPUT" || tagName === "SELECT" || tagName === "TEXTAREA";
}

function waitForEvent(target, eventName, timeoutMs = 6000) {
  return new Promise((resolve, reject) => {
    let finished = false;
    const timer = setTimeout(() => {
      if (!finished) {
        finished = true;
        target.removeEventListener(eventName, onEvent);
        reject(new Error(`Timed out waiting for ${eventName}`));
      }
    }, timeoutMs);
    function onEvent() {
      if (finished) {
        return;
      }
      finished = true;
      clearTimeout(timer);
      target.removeEventListener(eventName, onEvent);
      resolve();
    }
    target.addEventListener(eventName, onEvent, { once: true });
  });
}

function waitForVideoData(timeoutMs = 10000) {
  if (video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA) {
    return Promise.resolve();
  }
  return new Promise((resolve, reject) => {
    let finished = false;
    const events = ["loadeddata", "canplay", "seeked", "timeupdate"];
    const cleanup = () => {
      clearTimeout(timer);
      for (const eventName of events) {
        video.removeEventListener(eventName, onReady);
      }
      video.removeEventListener("error", onError);
    };
    const finish = (callback) => {
      if (finished) {
        return;
      }
      finished = true;
      cleanup();
      callback();
    };
    function onReady() {
      if (video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA) {
        finish(resolve);
      }
    }
    function onError() {
      finish(() => reject(new Error(video.error ? video.error.message : "Video decode failed.")));
    }
    const timer = setTimeout(() => {
      finish(() => reject(new Error("Timed out waiting for decoded video data.")));
    }, timeoutMs);
    for (const eventName of events) {
      video.addEventListener(eventName, onReady);
    }
    video.addEventListener("error", onError, { once: true });
  });
}

function resizeCanvases() {
  const rect = pane.getBoundingClientRect();
  const width = Math.max(1, Math.floor(rect.width * devicePixelRatio));
  const height = Math.max(1, Math.floor(rect.height * devicePixelRatio));
  let changed = false;
  for (const canvas of [gpuCanvas, overlay]) {
    if (canvas.width !== width) {
      canvas.width = width;
      changed = true;
    }
    if (canvas.height !== height) {
      canvas.height = height;
      changed = true;
    }
  }
  return { rect, width, height, changed };
}

function setViewportToFit() {
  const rect = pane.getBoundingClientRect();
  const scaleX = rect.width / state.image.width;
  const scaleY = rect.height / state.image.height;
  state.viewport.scale = Math.min(scaleX, scaleY) * 0.96;
  state.viewport.offsetX = (rect.width - state.image.width * state.viewport.scale) / 2;
  state.viewport.offsetY = (rect.height - state.image.height * state.viewport.scale) / 2;
  draw();
}

function imgToView(point) {
  return [
    state.viewport.offsetX + point[0] * state.viewport.scale,
    state.viewport.offsetY + point[1] * state.viewport.scale,
  ];
}

function viewToImg(x, y) {
  return [
    (x - state.viewport.offsetX) / state.viewport.scale,
    (y - state.viewport.offsetY) / state.viewport.scale,
  ];
}

function eventToOverlayPoint(event) {
  const rect = overlay.getBoundingClientRect();
  return [
    event.clientX - rect.left,
    event.clientY - rect.top,
  ];
}

function clampImgPoint(point) {
  return [
    Math.min(Math.max(point[0], 0), Math.max(0, state.image.width - 1)),
    Math.min(Math.max(point[1], 0), Math.max(0, state.image.height - 1)),
  ];
}

function normToRectPx(bbox) {
  if (!bbox) {
    return null;
  }
  const [cx, cy, bw, bh] = bbox;
  return [
    (cx - bw * 0.5) * state.image.width,
    (cy - bh * 0.5) * state.image.height,
    (cx + bw * 0.5) * state.image.width,
    (cy + bh * 0.5) * state.image.height,
  ];
}

function rectPxToNorm(rect) {
  const xMin = Math.min(rect[0], rect[2]);
  const xMax = Math.max(rect[0], rect[2]);
  const yMin = Math.min(rect[1], rect[3]);
  const yMax = Math.max(rect[1], rect[3]);
  const cx = ((xMin + xMax) * 0.5) / state.image.width;
  const cy = ((yMin + yMax) * 0.5) / state.image.height;
  const bw = (xMax - xMin) / state.image.width;
  const bh = (yMax - yMin) / state.image.height;
  if (bw <= 0 || bh <= 0) {
    return null;
  }
  return [
    Math.min(Math.max(cx, 0), 1),
    Math.min(Math.max(cy, 0), 1),
    Math.min(Math.max(bw, 0), 1),
    Math.min(Math.max(bh, 0), 1),
  ];
}

function pointInBox(imgPoint) {
  const rect = state.bboxRect || normToRectPx(state.bboxNorm);
  if (!rect) {
    return false;
  }
  const xMin = Math.min(rect[0], rect[2]);
  const xMax = Math.max(rect[0], rect[2]);
  const yMin = Math.min(rect[1], rect[3]);
  const yMax = Math.max(rect[1], rect[3]);
  return imgPoint[0] >= xMin && imgPoint[0] <= xMax && imgPoint[1] >= yMin && imgPoint[1] <= yMax;
}

async function initWebGpuRenderer() {
  if (!window.isSecureContext) {
    throw new Error("WebGPU requires a secure context; use an SSH tunnel to http://localhost or serve over HTTPS.");
  }
  if (!navigator.gpu) {
    throw new Error("navigator.gpu is unavailable in this browser/session.");
  }
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) {
    throw new Error("No WebGPU adapter was available.");
  }
  const device = await adapter.requestDevice();
  const context = gpuCanvas.getContext("webgpu");
  if (!context) {
    throw new Error("Could not create a WebGPU canvas context.");
  }
  const format = navigator.gpu.getPreferredCanvasFormat();
  let configuredWidth = 0;
  let configuredHeight = 0;
  let lastRenderError = "";

  device.lost.then((info) => {
    state.renderer = { type: "canvas2d" };
    pane.classList.remove("webgpu");
    setRendererLabel(`Renderer: Canvas2D fallback (WebGPU device lost: ${info.message || info.reason})`);
  });

  function configureIfNeeded() {
    if (configuredWidth === gpuCanvas.width && configuredHeight === gpuCanvas.height) {
      return;
    }
    context.configure({ device, format, alphaMode: "opaque" });
    configuredWidth = gpuCanvas.width;
    configuredHeight = gpuCanvas.height;
  }

  const shader = device.createShaderModule({
    code: `
      struct VertexOut {
        @builtin(position) position: vec4f,
        @location(0) uv: vec2f,
      };

      @vertex
      fn vs(@location(0) position: vec2f, @location(1) uv: vec2f) -> VertexOut {
        var out: VertexOut;
        out.position = vec4f(position, 0.0, 1.0);
        out.uv = uv;
        return out;
      }

      @group(0) @binding(0) var frameSampler: sampler;
      @group(0) @binding(1) var frameTexture: texture_external;

      @fragment
      fn fs(in: VertexOut) -> @location(0) vec4f {
        return textureSampleBaseClampToEdge(frameTexture, frameSampler, in.uv);
      }
    `,
  });
  const pipeline = device.createRenderPipeline({
    layout: "auto",
    vertex: {
      module: shader,
      entryPoint: "vs",
      buffers: [
        {
          arrayStride: 16,
          attributes: [
            { shaderLocation: 0, offset: 0, format: "float32x2" },
            { shaderLocation: 1, offset: 8, format: "float32x2" },
          ],
        },
      ],
    },
    fragment: {
      module: shader,
      entryPoint: "fs",
      targets: [{ format }],
    },
    primitive: { topology: "triangle-list" },
  });
  const sampler = device.createSampler({ magFilter: "linear", minFilter: "linear" });
  const vertexBuffer = device.createBuffer({
    size: 6 * 4 * Float32Array.BYTES_PER_ELEMENT,
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
  });

  function writeVertices() {
    const rect = pane.getBoundingClientRect();
    const x1 = state.viewport.offsetX;
    const y1 = state.viewport.offsetY;
    const x2 = x1 + state.image.width * state.viewport.scale;
    const y2 = y1 + state.image.height * state.viewport.scale;
    const ndc = (x, y) => [
      (x / rect.width) * 2 - 1,
      1 - (y / rect.height) * 2,
    ];
    const [ax, ay] = ndc(x1, y1);
    const [bx, by] = ndc(x2, y1);
    const [cx, cy] = ndc(x1, y2);
    const [dx, dy] = ndc(x2, y2);
    const vertices = new Float32Array([
      ax, ay, 0, 0,
      bx, by, 1, 0,
      cx, cy, 0, 1,
      cx, cy, 0, 1,
      bx, by, 1, 0,
      dx, dy, 1, 1,
    ]);
    device.queue.writeBuffer(vertexBuffer, 0, vertices);
  }

  function render() {
    if (video.readyState < 2) {
      return false;
    }
    configureIfNeeded();
    writeVertices();
    let bindGroup;
    try {
      bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: sampler },
          { binding: 1, resource: device.importExternalTexture({ source: video }) },
        ],
      });
    } catch (error) {
      const message = error && error.message ? error.message : String(error);
      if (message !== lastRenderError) {
        lastRenderError = message;
        setRendererLabel(`Renderer: Canvas2D fallback (WebGPU video texture import failed: ${message})`);
      }
      return false;
    }
    if (lastRenderError) {
      lastRenderError = "";
      setRendererLabel("Renderer: WebGPU video texture + Canvas overlay");
    }
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginRenderPass({
      colorAttachments: [
        {
          view: context.getCurrentTexture().createView(),
          clearValue: { r: 0.008, g: 0.024, b: 0.055, a: 1 },
          loadOp: "clear",
          storeOp: "store",
        },
      ],
    });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.setVertexBuffer(0, vertexBuffer);
    pass.draw(6);
    pass.end();
    device.queue.submit([encoder.finish()]);
    return true;
  }

  return { type: "webgpu", render };
}

function drawBox(ctx) {
  const rect = state.bboxRect || normToRectPx(state.bboxNorm);
  if (!rect) {
    return;
  }
  const [x1, y1] = imgToView([rect[0], rect[1]]);
  const [x2, y2] = imgToView([rect[2], rect[3]]);
  const x = Math.min(x1, x2);
  const y = Math.min(y1, y2);
  const w = Math.abs(x2 - x1);
  const h = Math.abs(y2 - y1);
  ctx.strokeStyle = "#f97316";
  ctx.lineWidth = 2;
  ctx.setLineDash([]);
  ctx.strokeRect(x, y, w, h);
  ctx.fillStyle = "rgba(249, 115, 22, 0.14)";
  ctx.fillRect(x, y, w, h);
}

function draw() {
  resizeCanvases();
  const rect = pane.getBoundingClientRect();
  overlayCtx.setTransform(devicePixelRatio, 0, 0, devicePixelRatio, 0, 0);
  overlayCtx.clearRect(0, 0, rect.width, rect.height);

  const renderedWithWebGpu = state.renderer && state.renderer.type === "webgpu" && state.renderer.render();
  if (!renderedWithWebGpu) {
    overlayCtx.fillStyle = "#020617";
    overlayCtx.fillRect(0, 0, rect.width, rect.height);
    if (video.readyState >= 2) {
      overlayCtx.drawImage(
        video,
        state.viewport.offsetX,
        state.viewport.offsetY,
        state.image.width * state.viewport.scale,
        state.image.height * state.viewport.scale,
      );
    }
  }
  drawBox(overlayCtx);
  updateRendererLine();
}

function renderText(payload) {
  const st = payload.state || state.serverState || {};
  const summary = st.summary || {};
  const status = payload.status || {};
  frameInput.value = String(payload.parent_frame_index ?? 0);
  frameInput.max = String(Math.max(0, (summary.total_frames || 1) - 1));
  stateLine.textContent = [
    `parent=${payload.parent_frame_index ?? "-"}`,
    payload.clip_id ? `clip=${payload.clip_id}` : null,
    payload.source_frame_index !== undefined ? `local=${payload.source_frame_index}` : null,
    payload.recording_frame_id !== null && payload.recording_frame_id !== undefined ? `recording_frame_id=${payload.recording_frame_id}` : null,
    `mode=${summary.mode || "-"}`,
    `editable=${Boolean(summary.editable)}`,
  ].filter(Boolean).join(" | ");
  statusLine.textContent = [
    status.status_label ? `status=${status.status_label}` : null,
    status.source_kind_label ? `source=${status.source_kind_label}` : null,
    status.reason_label ? `reason=${status.reason_label}` : null,
    status.manual_edit !== undefined ? `manual=${Boolean(status.manual_edit)}` : null,
    status.confidence_score !== null && status.confidence_score !== undefined ? `score=${Number(status.confidence_score).toFixed(3)}` : null,
    payload.media_width && payload.source_width ? `media=${payload.media_width}x${payload.media_height}` : null,
    payload.source_width ? `source=${payload.source_width}x${payload.source_height}` : null,
    payload.bbox_media_xyxy ? `box_media=[${payload.bbox_media_xyxy.map((v) => Number(v).toFixed(1)).join(",")}]` : null,
    payload.refined_group_path ? `run=${payload.refined_group_path}` : null,
  ].filter(Boolean).join(" | ");
}

function updateBoxFromPayload(payload) {
  const frame = Number(payload.parent_frame_index);
  const pending = state.pendingEdits.get(frame);
  state.bboxNorm = cloneBboxNorm(pending ? pending.bbox_norm : (payload.bbox_norm || null));
  state.bboxRect = pending
    ? normToRectPx(state.bboxNorm)
    : (payload.bbox_media_xyxy ? [...payload.bbox_media_xyxy] : normToRectPx(state.bboxNorm));
}

function cacheFramePayload(payload) {
  const frame = Number(payload.parent_frame_index);
  if (!Number.isFinite(frame)) {
    return;
  }
  state.frameCache.set(frame, payload);
  while (state.frameCache.size > 1800) {
    const oldest = state.frameCache.keys().next().value;
    state.frameCache.delete(oldest);
  }
}

async function fetchFramePayload(frameIndex, { updateCurrent = true } = {}) {
  const frame = Number(frameIndex);
  if (state.frameCache.has(frame)) {
    return state.frameCache.get(frame);
  }
  if (state.pendingFrameFetches.has(frame)) {
    return state.pendingFrameFetches.get(frame);
  }
  const url = updateCurrent ? `/api/frame/${frame}` : `/api/frame/${frame}?update_current=false`;
  const request = fetch(url).then(async (response) => {
    const payload = await response.json();
    if (!payload.ok) {
      throw new Error(payload.details || "Frame load failed.");
    }
    if (updateCurrent) {
      state.serverState = payload.state || state.serverState;
    } else if (payload.state && !state.serverState) {
      state.serverState = payload.state;
    }
    cacheFramePayload(payload);
    return payload;
  }).finally(() => {
    state.pendingFrameFetches.delete(frame);
  });
  state.pendingFrameFetches.set(frame, request);
  return request;
}

function prefetchPlaybackWindow(centerFrame) {
  const totalFrames = Number(state.serverState?.summary?.total_frames || 0);
  const fps = Number(state.framePayload?.fps || 30);
  const lookaheadFrames = Math.min(240, Math.max(45, Math.ceil(fps * currentPlaybackRate() * 2)));
  const start = Math.max(0, Math.trunc(centerFrame) - 2);
  const end = Math.trunc(centerFrame) + lookaheadFrames;
  for (let frame = start; frame <= end; frame += 1) {
    if (totalFrames > 0 && frame >= totalFrames) {
      break;
    }
    if (state.frameCache.has(frame) || state.pendingFrameFetches.has(frame)) {
      continue;
    }
    fetchFramePayload(frame, { updateCurrent: false }).catch(() => {
      // Playback can continue with already-cached frames; exact fetch errors
      // are surfaced when the frame becomes current.
    });
  }
}

function nearestCachedPayload(parentFrame, maxDistance = 2) {
  if (state.frameCache.has(parentFrame)) {
    return state.frameCache.get(parentFrame);
  }
  for (let offset = 1; offset <= maxDistance; offset += 1) {
    const previous = state.frameCache.get(parentFrame - offset);
    if (previous && previous.video_id === state.playback.videoId) {
      return previous;
    }
    const next = state.frameCache.get(parentFrame + offset);
    if (next && next.video_id === state.playback.videoId) {
      return next;
    }
  }
  return null;
}

function applyFramePayload(payload, { fit = false } = {}) {
  state.framePayload = payload;
  state.serverState = payload.state || state.serverState;
  state.image.width = Number(payload.media_width || payload.width || 1);
  state.image.height = Number(payload.media_height || payload.height || 1);
  updateBoxFromPayload(payload);
  cacheFramePayload(payload);
  updateClipSelector(payload);
  updateEditableControls();
  if (fit || state.viewport.scale <= 0) {
    setViewportToFit();
  } else {
    draw();
  }
  renderText(payload);
}

async function ensureVideo(payload) {
  if (state.currentVideoUrl !== payload.media_url) {
    state.currentVideoUrl = payload.media_url;
    video.preload = "auto";
    video.src = payload.media_url;
    video.load();
    await waitForEvent(video, "loadedmetadata");
  }
  applyPlaybackRate();
  const targetTime = Number(payload.video_time_s || 0);
  if (Math.abs(video.currentTime - targetTime) > Math.max(0.002, 0.25 / Number(payload.fps || 30))) {
    const seekPromise = waitForEvent(video, "seeked", 10000).catch(() => null);
    video.currentTime = targetTime;
    await seekPromise;
  }
  await waitForVideoData();
  updateRendererLine();
}

async function loadFrame(frameIndex, { fit = false } = {}) {
  pausePlayback();
  let payload;
  try {
    payload = await fetchFramePayload(frameIndex);
  } catch (error) {
    setMessage(error.message || "Frame load failed.", true);
    return;
  }
  await ensureVideo(payload);
  applyFramePayload(payload, { fit });
}

async function loadCurrentFrame({ fit = false } = {}) {
  pausePlayback();
  const response = await fetch("/api/frame/current");
  const payload = await response.json();
  if (!payload.ok) {
    setMessage(payload.details || "Frame load failed.", true);
    return;
  }
  await ensureVideo(payload);
  applyFramePayload(payload, { fit });
}

async function navigate(delta) {
  const current = Number(state.framePayload?.parent_frame_index || 0);
  await loadFrame(current + delta);
}

async function goToFrame() {
  pausePlayback();
  const value = Number(frameInput.value || 0);
  await loadFrame(value, { fit: false });
}

async function goToSelectedClip() {
  pausePlayback();
  const selectedVideoId = String(clipSelect?.value || "");
  const source = availableClipSources().find((candidate) => String(candidate.video_id || "") === selectedVideoId);
  if (!source) {
    setMessage("Selected clip is not available.", true);
    return;
  }
  const firstFrame = numberOrNull(source.parent_frame_start);
  if (firstFrame === null) {
    setMessage("Selected clip has no parent-frame range.", true);
    return;
  }
  await loadFrame(firstFrame, { fit: false });
}

function lowConfidenceThreshold() {
  const value = Number(lowConfidenceInput?.value || 0.5);
  if (!Number.isFinite(value)) {
    return 0.5;
  }
  return Math.min(Math.max(value, 0), 1);
}

async function searchFrame(target, direction = "next") {
  pausePlayback();
  const start = Math.trunc(Number(state.framePayload?.parent_frame_index ?? frameInput.value ?? 0));
  const params = new URLSearchParams({
    target,
    direction,
    start: String(start),
    low_confidence_threshold: String(lowConfidenceThreshold()),
  });
  const response = await fetch(`/api/search?${params.toString()}`);
  const payload = await response.json();
  if (!payload.ok) {
    setMessage(payload.details || payload.error || "No matching frame found.", true);
    return;
  }
  await ensureVideo(payload);
  applyFramePayload(payload);
  const search = payload.search || {};
  setMessage(`Found ${search.target || target} at frame ${payload.parent_frame_index}.`);
}

async function saveCurrent() {
  pausePlayback();
  if (state.isSaving) {
    setMessage("Save already in progress; wait for it to finish.", true);
    return;
  }
  if (!isEditable()) {
    setMessage("Read-only mode. Restart with --edit to save boxes.", true);
    return;
  }
  const pending = Array.from(state.pendingEdits.values()).sort((a, b) => a.frame - b.frame);
  if (pending.length > 0) {
    setSavingState(true, `Saving ${pending.length} pending frame${pending.length === 1 ? "" : "s"}...`);
    try {
      await savePendingEdits(pending);
    } catch (error) {
      setMessage(`Save failed: ${error.message || error}`, true);
    } finally {
      setSavingState(false);
    }
    return;
  }
  const visibleFrame = Math.trunc(Number(state.framePayload?.parent_frame_index ?? frameInput.value ?? 0));
  setSavingState(true, `Saving frame ${visibleFrame}...`);
  try {
    const response = await fetch(`/api/frame/${visibleFrame}/save`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        bbox_norm: state.bboxNorm,
        advance: Boolean(autoAdvanceBox && autoAdvanceBox.checked),
      }),
    });
    const payload = await response.json();
    if (!payload.ok) {
      setMessage(payload.details || "Save failed.", true);
      return;
    }
    const result = payload.result || {};
    const promotion = payload.promotion || null;
    const promotionError = payload.promotion_error || null;
    const promotionCounts = promotion && promotion.result ? promotion.result.action_counts || {} : {};
    const timing = payload.timing || {};
    readbackLine.textContent = [
      result.action ? `action=${result.action}` : null,
      result.parent_frame_index !== undefined ? `parent=${result.parent_frame_index}` : null,
      result.status && result.status.status_label ? `status=${result.status.status_label}` : null,
      result.status && result.status.reason_label ? `reason=${result.status.reason_label}` : null,
      promotion ? `promotion=${JSON.stringify(promotionCounts)}` : null,
      promotionError ? `promotion_error=${promotionError.details || promotionError.error}` : null,
      Number.isFinite(Number(timing.analysis_write_s)) ? `analysis=${Number(timing.analysis_write_s).toFixed(3)}s` : null,
      Number.isFinite(Number(timing.promotion_s)) ? `promotion_s=${Number(timing.promotion_s).toFixed(3)}s` : null,
      Number.isFinite(Number(timing.total_save_s)) ? `total=${Number(timing.total_save_s).toFixed(3)}s` : null,
    ].filter(Boolean).join(" | ");
    state.frameCache.delete(visibleFrame);
    if (promotionError) {
      setMessage(`Saved, but promotion failed: ${promotionError.details || promotionError.error}`, true);
    } else if (promotion && promotion.ok === false) {
      const status = promotion.result && promotion.result.status ? promotion.result.status : "not_ok";
      setMessage(`Saved, but promotion returned ${status}.`, true);
    } else if (promotion) {
      setMessage("Saved and promoted.");
    } else {
      setMessage("Saved.");
    }
    await loadCurrentFrame();
  } catch (error) {
    setMessage(`Save failed: ${error.message || error}`, true);
  } finally {
    setSavingState(false);
  }
}

async function savePendingEdits(edits) {
  const response = await fetch("/api/frames/save_batch", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ edits }),
  });
  const payload = await response.json();
  if (!payload.ok) {
    setMessage(payload.details || "Batch save failed.", true);
    return;
  }
  const items = Array.isArray(payload.items) ? payload.items : [];
  const okItems = items.filter((item) => item.ok);
  const completedItems = okItems.filter((item) => {
    if (item.promotion_error) {
      return false;
    }
    if (item.promotion && item.promotion.ok === false) {
      return false;
    }
    return true;
  });
  for (const item of completedItems) {
    state.pendingEdits.delete(Number(item.frame));
    state.frameCache.delete(Number(item.frame));
  }
  const summary = payload.summary || {};
  const promotionFailures = Number(summary.promotion_failed || 0);
  const failed = Number(summary.failed || 0);
  const timing = payload.timing || {};
  const slowestFrames = Array.isArray(timing.slowest_frames) ? timing.slowest_frames.slice(0, 3) : [];
  const slowestLabel = slowestFrames.map((row) => {
    const frame = Number(row.frame);
    const total = Number(row.total_save_s);
    if (!Number.isFinite(frame) || !Number.isFinite(total)) {
      return null;
    }
    return `${frame}:${total.toFixed(2)}s`;
  }).filter(Boolean).join(",");
  readbackLine.textContent = [
    `batch_saved=${Number(summary.saved || okItems.length)}`,
    `failed=${failed}`,
    promotionFailures ? `promotion_failed=${promotionFailures}` : null,
    Number.isFinite(Number(timing.total_batch_s)) ? `batch_total=${Number(timing.total_batch_s).toFixed(3)}s` : null,
    Number.isFinite(Number(timing.analysis_write_total_s)) ? `analysis_total=${Number(timing.analysis_write_total_s).toFixed(3)}s` : null,
    Number.isFinite(Number(timing.analysis_write_mean_s)) ? `analysis_mean=${Number(timing.analysis_write_mean_s).toFixed(3)}s` : null,
    Number.isFinite(Number(timing.analysis_write_max_s)) ? `analysis_max=${Number(timing.analysis_write_max_s).toFixed(3)}s` : null,
    Number.isFinite(Number(timing.promotion_total_s)) ? `promotion_total=${Number(timing.promotion_total_s).toFixed(3)}s` : null,
    Number.isFinite(Number(timing.promotion_mean_s)) ? `promotion_mean=${Number(timing.promotion_mean_s).toFixed(3)}s` : null,
    Number.isFinite(Number(timing.promotion_decode_total_s)) ? `decode_total=${Number(timing.promotion_decode_total_s).toFixed(3)}s` : null,
    Number.isFinite(Number(timing.promotion_image_write_s)) ? `image_write=${Number(timing.promotion_image_write_s).toFixed(3)}s` : null,
    Number.isFinite(Number(timing.promotion_zarr_metadata_write_s)) ? `metadata_write=${Number(timing.promotion_zarr_metadata_write_s).toFixed(3)}s` : null,
    Number.isFinite(Number(timing.promotion_decode_group_count)) ? `decode_groups=${Number(timing.promotion_decode_group_count)}` : null,
    slowestLabel ? `slowest=${slowestLabel}` : null,
  ].filter(Boolean).join(" | ");
  updateDirtyControls();
  if (failed > 0 || promotionFailures > 0) {
    setMessage(`Saved ${Number(summary.saved || okItems.length)} pending edits; ${failed} failed, ${promotionFailures} promotion failures.`, true);
  } else {
    setMessage(`Saved ${Number(summary.saved || okItems.length)} pending edits.`);
  }
  const currentWasSaved = completedItems.some((item) => Number(item.frame) === currentFrameNumber());
  if (currentWasSaved && autoAdvanceBox && autoAdvanceBox.checked) {
    await navigate(1);
  } else {
    await loadCurrentFrame();
  }
}

function clearBox() {
  pausePlayback();
  if (!isEditable()) {
    setMessage("Read-only mode. Restart with --edit to clear boxes.", true);
    return;
  }
  state.bboxNorm = null;
  state.bboxRect = null;
  markCurrentDirty();
  draw();
  setMessage("Box cleared locally; press Save to persist.");
}

function currentVideoSource() {
  const videos = state.serverState && Array.isArray(state.serverState.videos) ? state.serverState.videos : [];
  return videos.find((source) => source.video_id === state.playback.videoId) || null;
}

function currentPlaybackFrame() {
  const payload = state.framePayload || {};
  const fps = Number(payload.fps || 30);
  const localFrame = Math.max(0, Math.round(Number(video.currentTime || 0) * fps));
  return {
    localFrame,
    parentFrame: Math.trunc(state.playback.parentOffset + localFrame),
  };
}

function pausePlayback() {
  if (state.playback.rafId !== null) {
    cancelAnimationFrame(state.playback.rafId);
  }
  state.playback.isPlaying = false;
  state.playback.rafId = null;
  state.playback.videoId = null;
  state.playback.clipId = null;
  state.playback.lastFrame = null;
  state.playback.lastRenderedFrame = null;
  if (!video.paused) {
    video.pause();
  }
  if (playBtn) {
    playBtn.textContent = "Play";
  }
}

function playbackLoop() {
  if (!state.playback.isPlaying) {
    return;
  }
  const { localFrame, parentFrame } = currentPlaybackFrame();
  const source = currentVideoSource();
  if (source && source.frame_count !== null && source.frame_count !== undefined && localFrame >= Number(source.frame_count)) {
    pausePlayback();
    return;
  }
  const totalFrames = Number(state.serverState?.summary?.total_frames || 0);
  if (totalFrames > 0 && (parentFrame < 0 || parentFrame >= totalFrames)) {
    pausePlayback();
    return;
  }
  if (parentFrame !== state.playback.lastFrame) {
    state.playback.lastFrame = parentFrame;
    frameInput.value = String(parentFrame);
    prefetchPlaybackWindow(parentFrame);
    const cached = nearestCachedPayload(parentFrame);
    if (cached) {
      if (cached.video_id !== state.playback.videoId) {
        pausePlayback();
        return;
      }
      if (Number(cached.parent_frame_index) !== state.playback.lastRenderedFrame) {
        state.playback.lastRenderedFrame = Number(cached.parent_frame_index);
        applyFramePayload(cached);
      } else {
        draw();
      }
    } else {
      fetchFramePayload(parentFrame)
        .then((payload) => {
          if (!state.playback.isPlaying) {
            return;
          }
          const current = currentPlaybackFrame();
          if (payload.video_id !== state.playback.videoId) {
            pausePlayback();
            return;
          }
          if (Number(payload.parent_frame_index) === current.parentFrame) {
            state.playback.lastRenderedFrame = Number(payload.parent_frame_index);
            applyFramePayload(payload);
          }
        })
        .catch((error) => {
          pausePlayback();
          setMessage(error.message || "Playback frame load failed.", true);
        });
    }
  }
  draw();
  state.playback.rafId = requestAnimationFrame(playbackLoop);
}

async function togglePlayback() {
  if (state.playback.isPlaying) {
    pausePlayback();
    return;
  }
  const payload = state.framePayload;
  if (!payload) {
    setMessage("No frame loaded.", true);
    return;
  }
  await ensureVideo(payload);
  state.playback.isPlaying = true;
  state.playback.videoId = payload.video_id;
  state.playback.clipId = payload.clip_id || null;
  state.playback.parentOffset = Number(payload.parent_frame_index || 0) - Number(payload.source_frame_index || 0);
  state.playback.lastFrame = null;
  state.playback.lastRenderedFrame = null;
  applyPlaybackRate();
  prefetchPlaybackWindow(Number(payload.parent_frame_index || 0));
  if (playBtn) {
    playBtn.textContent = "Pause";
  }
  try {
    await video.play();
  } catch (error) {
    pausePlayback();
    setMessage(error.message || "Video playback failed.", true);
    return;
  }
  state.playback.rafId = requestAnimationFrame(playbackLoop);
}

function ensureEvents() {
  overlay.addEventListener("contextmenu", (event) => {
    event.preventDefault();
  });

  overlay.addEventListener("mousedown", (event) => {
    pausePlayback();
    const [x, y] = eventToOverlayPoint(event);
    const imgPoint = clampImgPoint(viewToImg(x, y));
    if (event.shiftKey || event.button === 1 || event.button === 2) {
      state.viewport.isPanning = true;
      state.viewport.panStartX = x;
      state.viewport.panStartY = y;
      overlay.style.cursor = "grabbing";
      event.preventDefault();
      return;
    }
    if (event.button !== 0) {
      return;
    }
    if (!isEditable()) {
      setMessage("Read-only mode. Restart with --edit to draw or move boxes.", true);
      event.preventDefault();
      return;
    }
    if (pointInBox(imgPoint)) {
      state.viewport.moveStart = imgPoint;
      state.viewport.moveStartBox = state.bboxRect ? [...state.bboxRect] : null;
      overlay.style.cursor = "move";
      event.preventDefault();
      return;
    }
    state.viewport.drawStart = imgPoint;
    state.bboxRect = [imgPoint[0], imgPoint[1], imgPoint[0] + 1, imgPoint[1] + 1];
    state.bboxNorm = rectPxToNorm(state.bboxRect);
    draw();
    event.preventDefault();
  });

  window.addEventListener("mousemove", (event) => {
    const rect = overlay.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    if (x < 0 || y < 0 || x > rect.width || y > rect.height) {
      return;
    }
    const imgPoint = clampImgPoint(viewToImg(x, y));

    if (state.viewport.isPanning) {
      const dx = x - state.viewport.panStartX;
      const dy = y - state.viewport.panStartY;
      state.viewport.offsetX += dx;
      state.viewport.offsetY += dy;
      state.viewport.panStartX = x;
      state.viewport.panStartY = y;
      draw();
      return;
    }

    if (state.viewport.moveStart && state.viewport.moveStartBox) {
      const dx = imgPoint[0] - state.viewport.moveStart[0];
      const dy = imgPoint[1] - state.viewport.moveStart[1];
      const moved = [...state.viewport.moveStartBox];
      const width = Math.abs(moved[2] - moved[0]);
      const height = Math.abs(moved[3] - moved[1]);
      const xMin = Math.min(Math.max(Math.min(moved[0], moved[2]) + dx, 0), Math.max(0, state.image.width - width));
      const yMin = Math.min(Math.max(Math.min(moved[1], moved[3]) + dy, 0), Math.max(0, state.image.height - height));
      state.bboxRect = [xMin, yMin, xMin + width, yMin + height];
      state.bboxNorm = rectPxToNorm(state.bboxRect);
      draw();
      return;
    }

    if (state.viewport.drawStart) {
      const start = state.viewport.drawStart;
      state.bboxRect = [start[0], start[1], imgPoint[0], imgPoint[1]];
      state.bboxNorm = rectPxToNorm(state.bboxRect);
      draw();
    }
  });

  window.addEventListener("mouseup", () => {
    const completedBoxEdit = Boolean(state.viewport.drawStart || state.viewport.moveStart);
    state.viewport.isPanning = false;
    state.viewport.drawStart = null;
    state.viewport.moveStart = null;
    state.viewport.moveStartBox = null;
    overlay.style.cursor = "crosshair";
    if (completedBoxEdit) {
      state.bboxNorm = rectPxToNorm(state.bboxRect);
      markCurrentDirty();
      draw();
    }
  });

  overlay.addEventListener("wheel", (event) => {
    event.preventDefault();
    const [cursorX, cursorY] = eventToOverlayPoint(event);
    const scaleBefore = state.viewport.scale;
    const delta = event.deltaY < 0 ? 1.12 : 0.9;
    const scaleAfter = Math.max(0.02, Math.min(40, scaleBefore * delta));
    const [imgX, imgY] = viewToImg(cursorX, cursorY);
    state.viewport.scale = scaleAfter;
    state.viewport.offsetX = cursorX - imgX * scaleAfter;
    state.viewport.offsetY = cursorY - imgY * scaleAfter;
    draw();
  }, { passive: false });

  window.addEventListener("resize", () => {
    resizeCanvases();
    draw();
  });

  window.addEventListener("keydown", async (event) => {
    const key = event.key;
    if (event.target === frameInput && key === "Enter") {
      event.preventDefault();
      await goToFrame();
      return;
    }
    if (isFormControl(event.target)) {
      return;
    }
    if (key === "s" || key === "S") {
      event.preventDefault();
      await saveCurrent();
      return;
    }
    if (key === "c" || key === "C") {
      event.preventDefault();
      clearBox();
      return;
    }
    if (key === "n") {
      event.preventDefault();
      await navigate(1);
      return;
    }
    if (key === "p") {
      event.preventDefault();
      await navigate(-1);
      return;
    }
    if (key === "f" || key === "F") {
      event.preventDefault();
      setViewportToFit();
      return;
    }
    if (key === "g" || key === "G") {
      event.preventDefault();
      await goToFrame();
      return;
    }
    if (key === "m") {
      event.preventDefault();
      await searchFrame("missing_or_filtered", "next");
      return;
    }
    if (key === "M") {
      event.preventDefault();
      await searchFrame("missing_or_filtered", "prev");
      return;
    }
    if (key === "l") {
      event.preventDefault();
      await searchFrame("low_confidence", "next");
      return;
    }
    if (key === "L") {
      event.preventDefault();
      await searchFrame("low_confidence", "prev");
      return;
    }
    if (key === "e") {
      event.preventDefault();
      await searchFrame("manual_edit", "next");
      return;
    }
    if (key === "E") {
      event.preventDefault();
      await searchFrame("manual_edit", "prev");
      return;
    }
    if (key === " " || key === "k" || key === "K") {
      event.preventDefault();
      await togglePlayback();
      return;
    }
    if (key === "[" || key === "{") {
      event.preventDefault();
      stepPlaybackRate(-1);
      return;
    }
    if (key === "]" || key === "}") {
      event.preventDefault();
      stepPlaybackRate(1);
    }
  });

  document.getElementById("fit-btn").addEventListener("click", setViewportToFit);
  saveBtn.addEventListener("click", saveCurrent);
  clearBtn.addEventListener("click", clearBox);
  document.getElementById("next-btn").addEventListener("click", () => navigate(1));
  document.getElementById("prev-btn").addEventListener("click", () => navigate(-1));
  document.getElementById("go-btn").addEventListener("click", goToFrame);
  if (clipSelect) {
    clipSelect.addEventListener("change", goToSelectedClip);
  }
  prevIssueBtn.addEventListener("click", () => searchFrame("missing_or_filtered", "prev"));
  nextIssueBtn.addEventListener("click", () => searchFrame("missing_or_filtered", "next"));
  nextLowConfBtn.addEventListener("click", () => searchFrame("low_confidence", "next"));
  nextManualBtn.addEventListener("click", () => searchFrame("manual_edit", "next"));
  playBtn.addEventListener("click", togglePlayback);
  speedSelect.addEventListener("change", applyPlaybackRate);
  video.addEventListener("ended", pausePlayback);
}

async function bootstrap() {
  resizeCanvases();
  ensureEvents();
  try {
    const webgpuRenderer = state.rendererPreference === "webgpu" ? await initWebGpuRenderer() : null;
    if (webgpuRenderer) {
      state.renderer = webgpuRenderer;
      pane.classList.add("webgpu");
      setRendererLabel("Renderer: WebGPU video texture + Canvas overlay");
    } else {
      state.renderer = { type: "canvas2d" };
      pane.classList.remove("webgpu");
      video.style.opacity = "0";
      setRendererLabel(state.rendererPreference === "webgpu" ? "Renderer: Canvas2D fallback" : "Renderer: Canvas2D default");
    }
  } catch (error) {
    state.renderer = { type: "canvas2d" };
    pane.classList.remove("webgpu");
    video.style.opacity = "0";
    setRendererLabel(`Renderer: Canvas2D fallback (${error.message || error})`);
  }
  await loadCurrentFrame({ fit: true });
}

bootstrap();
