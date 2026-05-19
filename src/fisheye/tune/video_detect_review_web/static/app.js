const state = {
  bboxNorm: null,
  bboxRect: null,
  framePayload: null,
  serverState: null,
  currentVideoUrl: null,
  messageTimeout: null,
  renderer: null,
  rendererLabel: "Renderer: detecting...",
  rendererPreference: new URLSearchParams(window.location.search).get("renderer") || "canvas",
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
const messages = document.getElementById("messages");
const rendererLine = document.getElementById("renderer-line");
const frameInput = document.getElementById("frame-input");
const autoAdvanceBox = document.getElementById("auto-advance-box");

function setRendererLabel(text) {
  state.rendererLabel = text;
  updateRendererLine();
}

function updateRendererLine() {
  const viewport = state.viewport;
  rendererLine.textContent = [
    state.rendererLabel,
    `video=${video.videoWidth || 0}x${video.videoHeight || 0}`,
    `t=${Number(video.currentTime || 0).toFixed(3)}s`,
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
  for (const canvas of [gpuCanvas, overlay]) {
    canvas.width = width;
    canvas.height = height;
  }
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
  if (!navigator.gpu) {
    return null;
  }
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) {
    return null;
  }
  const device = await adapter.requestDevice();
  const context = gpuCanvas.getContext("webgpu");
  const format = navigator.gpu.getPreferredCanvasFormat();
  context.configure({ device, format, alphaMode: "opaque" });

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
    resizeCanvases();
    context.configure({ device, format, alphaMode: "opaque" });
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
    } catch (_error) {
      return false;
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
  state.bboxNorm = payload.bbox_norm || null;
  state.bboxRect = payload.bbox_media_xyxy ? [...payload.bbox_media_xyxy] : normToRectPx(state.bboxNorm);
}

async function ensureVideo(payload) {
  if (state.currentVideoUrl !== payload.media_url) {
    state.currentVideoUrl = payload.media_url;
    video.preload = "auto";
    video.src = payload.media_url;
    video.load();
    await waitForEvent(video, "loadedmetadata");
  }
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
  const response = await fetch(`/api/frame/${frameIndex}`);
  const payload = await response.json();
  if (!payload.ok) {
    setMessage(payload.details || "Frame load failed.", true);
    return;
  }
  state.framePayload = payload;
  state.serverState = payload.state;
  state.image.width = Number(payload.media_width || payload.width || 1);
  state.image.height = Number(payload.media_height || payload.height || 1);
  updateBoxFromPayload(payload);
  await ensureVideo(payload);
  if (fit || state.viewport.scale <= 0) {
    setViewportToFit();
  } else {
    draw();
  }
  renderText(payload);
}

async function loadCurrentFrame({ fit = false } = {}) {
  const response = await fetch("/api/frame/current");
  const payload = await response.json();
  if (!payload.ok) {
    setMessage(payload.details || "Frame load failed.", true);
    return;
  }
  state.framePayload = payload;
  state.serverState = payload.state;
  state.image.width = Number(payload.media_width || payload.width || 1);
  state.image.height = Number(payload.media_height || payload.height || 1);
  updateBoxFromPayload(payload);
  await ensureVideo(payload);
  if (fit || state.viewport.scale <= 0) {
    setViewportToFit();
  } else {
    draw();
  }
  renderText(payload);
}

async function navigate(delta) {
  const current = Number(state.framePayload?.parent_frame_index || 0);
  await loadFrame(current + delta);
}

async function goToFrame() {
  const value = Number(frameInput.value || 0);
  await loadFrame(value, { fit: false });
}

async function saveCurrent() {
  const response = await fetch("/api/frame/current/save", {
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
  readbackLine.textContent = [
    result.action ? `action=${result.action}` : null,
    result.parent_frame_index !== undefined ? `parent=${result.parent_frame_index}` : null,
    result.status && result.status.status_label ? `status=${result.status.status_label}` : null,
    result.status && result.status.reason_label ? `reason=${result.status.reason_label}` : null,
  ].filter(Boolean).join(" | ");
  setMessage("Saved.");
  await loadCurrentFrame();
}

function clearBox() {
  state.bboxNorm = null;
  state.bboxRect = null;
  draw();
  setMessage("Box cleared locally; press Save to persist.");
}

function ensureEvents() {
  overlay.addEventListener("contextmenu", (event) => {
    event.preventDefault();
  });

  overlay.addEventListener("mousedown", (event) => {
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
    state.viewport.isPanning = false;
    state.viewport.drawStart = null;
    state.viewport.moveStart = null;
    state.viewport.moveStartBox = null;
    overlay.style.cursor = "crosshair";
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
    }
  });

  document.getElementById("fit-btn").addEventListener("click", setViewportToFit);
  document.getElementById("save-btn").addEventListener("click", saveCurrent);
  document.getElementById("clear-btn").addEventListener("click", clearBox);
  document.getElementById("next-btn").addEventListener("click", () => navigate(1));
  document.getElementById("prev-btn").addEventListener("click", () => navigate(-1));
  document.getElementById("go-btn").addEventListener("click", goToFrame);
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
