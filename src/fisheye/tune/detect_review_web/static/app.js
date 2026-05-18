const state = {
  bboxNorm: null,
  image: {
    width: 1,
    height: 1,
    sourceCanvas: null,
  },
  frameState: null,
  messageTimeout: null,
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

const canvas = document.getElementById("viewer");
const ctx = canvas.getContext("2d");
const stateLine = document.getElementById("state-line");
const statusLine = document.getElementById("status-line");
const readbackLine = document.getElementById("readback-line");
const messages = document.getElementById("messages");
const autoAdvanceBox = document.getElementById("auto-advance-box");

function setMessage(text, isError = false) {
  if (state.messageTimeout) {
    clearTimeout(state.messageTimeout);
  }
  messages.textContent = text;
  messages.style.color = isError ? "#fca5a5" : "#86efac";
  state.messageTimeout = setTimeout(() => {
    messages.textContent = "";
  }, 2200);
}

function ensureCanvasScale() {
  const rect = canvas.getBoundingClientRect();
  if (rect.width > 0 && rect.height > 0) {
    canvas.style.width = `${rect.width}px`;
    canvas.style.height = `${rect.height}px`;
  }
}

function setViewportFromImage() {
  const rect = canvas.getBoundingClientRect();
  const scaleX = rect.width / state.image.width;
  const scaleY = rect.height / state.image.height;
  state.viewport.scale = Math.min(scaleX, scaleY) * 0.95;
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

function clampImgPoint(point) {
  return [
    Math.min(Math.max(point[0], 0), state.image.width - 1),
    Math.min(Math.max(point[1], 0), state.image.height - 1),
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
  const rect = normToRectPx(state.bboxNorm);
  if (!rect) {
    return false;
  }
  const xMin = Math.min(rect[0], rect[2]);
  const xMax = Math.max(rect[0], rect[2]);
  const yMin = Math.min(rect[1], rect[3]);
  const yMax = Math.max(rect[1], rect[3]);
  return imgPoint[0] >= xMin && imgPoint[0] <= xMax && imgPoint[1] >= yMin && imgPoint[1] <= yMax;
}

function drawBox() {
  const rect = normToRectPx(state.bboxNorm);
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
  ctx.fillStyle = "rgba(249, 115, 22, 0.12)";
  ctx.fillRect(x, y, w, h);
}

function draw() {
  const rect = canvas.getBoundingClientRect();
  canvas.width = Math.max(1, Math.floor(rect.width * devicePixelRatio));
  canvas.height = Math.max(1, Math.floor(rect.height * devicePixelRatio));
  ctx.setTransform(devicePixelRatio, 0, 0, devicePixelRatio, 0, 0);
  ctx.clearRect(0, 0, rect.width, rect.height);
  ctx.fillStyle = "#030712";
  ctx.fillRect(0, 0, rect.width, rect.height);

  if (state.image.sourceCanvas) {
    ctx.drawImage(
      state.image.sourceCanvas,
      state.viewport.offsetX,
      state.viewport.offsetY,
      state.image.width * state.viewport.scale,
      state.image.height * state.viewport.scale,
    );
  }
  drawBox();
}

function parseImagePayload(payload) {
  const shape = payload.shape;
  const [h, w, channels = 1] = shape;
  const binary = atob(payload.pixels || "");
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) {
    bytes[i] = binary.charCodeAt(i);
  }

  const pixels = new Uint8ClampedArray(w * h * 4);
  const source = state.image.sourceCanvas || document.createElement("canvas");
  source.width = w;
  source.height = h;
  for (let i = 0; i < h * w; i += 1) {
    const j = i * 4;
    const b = i * channels;
    if (channels === 1) {
      const v = bytes[i] ?? 0;
      pixels[j] = v;
      pixels[j + 1] = v;
      pixels[j + 2] = v;
    } else {
      pixels[j] = bytes[b] ?? 0;
      pixels[j + 1] = bytes[b + 1] ?? 0;
      pixels[j + 2] = bytes[b + 2] ?? 0;
    }
    pixels[j + 3] = 255;
  }
  const srcCtx = source.getContext("2d");
  if (srcCtx) {
    srcCtx.putImageData(new ImageData(pixels, w, h), 0, 0);
  }
  state.image = { width: w, height: h, sourceCanvas: source };
}

function renderState(payload) {
  const st = payload.state || state.frameState || {};
  const summary = st.summary || {};
  stateLine.textContent = `Frame ${payload.frame_idx ?? "-"} | item ${(payload.position ?? 0) + 1}/${payload.total ?? st.total ?? 0} | refined=${st.refined_run || "-"}`;
  const status = payload.status || {};
  const pieces = [
    status.status_label ? `status=${status.status_label}` : null,
    status.source_kind_label ? `source=${status.source_kind_label}` : null,
    status.reason_label ? `reason=${status.reason_label}` : null,
    status.manual_edit !== undefined ? `manual=${Boolean(status.manual_edit)}` : null,
    status.confidence_score !== null && status.confidence_score !== undefined ? `score=${Number(status.confidence_score).toFixed(3)}` : null,
    summary.reviewable_frames !== undefined ? `reviewable=${summary.reviewable_frames}` : null,
  ].filter(Boolean);
  statusLine.textContent = pieces.join(" | ");
}

async function loadCurrentFrame() {
  const response = await fetch("/api/frame/current");
  const payload = await response.json();
  if (!payload.ok) {
    setMessage(payload.details || "No frame available.", true);
    return;
  }
  state.frameState = payload.state;
  state.bboxNorm = payload.bbox_norm || null;
  parseImagePayload(payload.frame_image);
  setViewportFromImage();
  renderState(payload);
}

async function navigate(delta) {
  const response = await fetch("/api/nav", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ delta }),
  });
  const payload = await response.json();
  if (!payload.ok) {
    setMessage(payload.details || "Navigation failed.", true);
    return;
  }
  await loadCurrentFrame();
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
    result.frame_idx !== undefined ? `frame=${result.frame_idx}` : null,
    result.status && result.status.status_label ? `status=${result.status.status_label}` : null,
    result.status && result.status.reason_label ? `reason=${result.status.reason_label}` : null,
  ].filter(Boolean).join(" | ");
  setMessage("Saved.");
  await loadCurrentFrame();
}

function clearBox() {
  state.bboxNorm = null;
  draw();
  setMessage("Box cleared locally; press Save to persist.");
}

function ensureEvents() {
  canvas.addEventListener("contextmenu", (event) => {
    event.preventDefault();
  });

  canvas.addEventListener("mousedown", (event) => {
    const x = event.offsetX;
    const y = event.offsetY;
    const imgPoint = clampImgPoint(viewToImg(x, y));
    if (event.shiftKey || event.button === 1 || event.button === 2) {
      state.viewport.isPanning = true;
      state.viewport.panStartX = x;
      state.viewport.panStartY = y;
      canvas.style.cursor = "grabbing";
      event.preventDefault();
      return;
    }
    if (event.button !== 0) {
      return;
    }
    if (pointInBox(imgPoint)) {
      state.viewport.moveStart = imgPoint;
      state.viewport.moveStartBox = state.bboxNorm ? [...state.bboxNorm] : null;
      canvas.style.cursor = "move";
      event.preventDefault();
      return;
    }
    state.viewport.drawStart = imgPoint;
    state.bboxNorm = rectPxToNorm([imgPoint[0], imgPoint[1], imgPoint[0] + 1, imgPoint[1] + 1]);
    draw();
    event.preventDefault();
  });

  window.addEventListener("mousemove", (event) => {
    if (!canvas.matches(":hover")) {
      return;
    }
    const rect = canvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
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
      const dx = (imgPoint[0] - state.viewport.moveStart[0]) / state.image.width;
      const dy = (imgPoint[1] - state.viewport.moveStart[1]) / state.image.height;
      const moved = [...state.viewport.moveStartBox];
      moved[0] = Math.min(Math.max(moved[0] + dx, moved[2] * 0.5), 1 - moved[2] * 0.5);
      moved[1] = Math.min(Math.max(moved[1] + dy, moved[3] * 0.5), 1 - moved[3] * 0.5);
      state.bboxNorm = moved;
      draw();
      return;
    }

    if (state.viewport.drawStart) {
      const start = state.viewport.drawStart;
      state.bboxNorm = rectPxToNorm([start[0], start[1], imgPoint[0], imgPoint[1]]);
      draw();
    }
  });

  window.addEventListener("mouseup", () => {
    state.viewport.isPanning = false;
    state.viewport.drawStart = null;
    state.viewport.moveStart = null;
    state.viewport.moveStartBox = null;
    canvas.style.cursor = "crosshair";
  });

  canvas.addEventListener("wheel", (event) => {
    event.preventDefault();
    const cursorX = event.offsetX;
    const cursorY = event.offsetY;
    const scaleBefore = state.viewport.scale;
    const delta = event.deltaY < 0 ? 1.12 : 0.9;
    const scaleAfter = Math.max(0.05, Math.min(30, scaleBefore * delta));
    const [imgX, imgY] = viewToImg(cursorX, cursorY);
    state.viewport.scale = scaleAfter;
    state.viewport.offsetX = cursorX - imgX * scaleAfter;
    state.viewport.offsetY = cursorY - imgY * scaleAfter;
    draw();
  }, { passive: false });

  window.addEventListener("resize", () => {
    ensureCanvasScale();
    setViewportFromImage();
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
      setViewportFromImage();
      return;
    }
    if (key === "q" || key === "Q") {
      window.close();
    }
  });

  document.getElementById("fit-btn").addEventListener("click", setViewportFromImage);
  document.getElementById("save-btn").addEventListener("click", saveCurrent);
  document.getElementById("clear-btn").addEventListener("click", clearBox);
  document.getElementById("next-btn").addEventListener("click", () => navigate(1));
  document.getElementById("prev-btn").addEventListener("click", () => navigate(-1));
}

async function bootstrap() {
  ensureCanvasScale();
  ensureEvents();
  await loadCurrentFrame();
}

bootstrap();
