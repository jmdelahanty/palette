const COLORS = ["#22c55e", "#1a66f3", "#f85151", "#f59e0b", "#a855f7", "#14b8a6", "#e11d48", "#0ea5e9", "#84cc16", "#f97316"];

const state = {
  labels: [],
  points: [],
  activeIndex: 0,
  showText: true,
  image: {
    width: 1,
    height: 1,
    sourceCanvas: null,
  },
  status: {},
  roiState: null,
  messageTimeout: null,
  viewport: {
    scale: 1,
    offsetX: 0,
    offsetY: 0,
    isPanning: false,
    panStartX: 0,
    panStartY: 0,
    draggingPoint: null,
    dragOffsetX: 0,
    dragOffsetY: 0,
  },
};

const canvas = document.getElementById("viewer");
const ctx = canvas.getContext("2d");
const statusLine = document.getElementById("status-line");
const reasonLine = document.getElementById("reason-line");
const pointsList = document.getElementById("points-list");
const stateLine = document.getElementById("state-line");
const readbackLine = document.getElementById("readback-line");
const messages = document.getElementById("messages");
const filterMode = document.getElementById("filter-mode");
const searchBox = document.getElementById("search-box");
const autoAdvanceBox = document.getElementById("auto-advance-box");
const datasetPanel = document.getElementById("dataset-panel");
const datasetSelect = document.getElementById("dataset-select");
const datasetRefreshBtn = document.getElementById("dataset-refresh-btn");
const datasetStatusLine = document.getElementById("dataset-status-line");

function setMessage(text, isError = false) {
  if (state.messageTimeout) {
    clearTimeout(state.messageTimeout);
  }
  messages.textContent = text;
  messages.style.color = isError ? "#fca5a5" : "#86efac";
  state.messageTimeout = setTimeout(() => {
    messages.textContent = "";
  }, 2000);
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

function clampPoint(point) {
  const x = Math.min(Math.max(point[0], 0), state.image.width - 1);
  const y = Math.min(Math.max(point[1], 0), state.image.height - 1);
  return [x, y];
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

function drawPoints() {
  for (let i = 0; i < state.points.length; i += 1) {
    if (!Number.isFinite(state.points[i][0]) || !Number.isFinite(state.points[i][1])) {
      continue;
    }
    const [vx, vy] = imgToView(state.points[i]);
    const size = i === state.activeIndex ? 6 : 4;
    ctx.fillStyle = COLORS[i % COLORS.length];
    ctx.strokeStyle = "#0f172a";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.arc(vx, vy, size, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
    if (state.showText) {
      ctx.fillStyle = "#f8fafc";
      ctx.fillText(state.labels[i] || String(i + 1), vx + 7, vy - 7);
    }
  }
}

function draw() {
  const rect = canvas.getBoundingClientRect();
  canvas.width = Math.max(1, Math.floor(rect.width * devicePixelRatio));
  canvas.height = Math.max(1, Math.floor(rect.height * devicePixelRatio));
  ctx.setTransform(devicePixelRatio, 0, 0, devicePixelRatio, 0, 0);
  ctx.clearRect(0, 0, rect.width, rect.height);
  ctx.fillStyle = "#020617";
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

  if (state.points.length > 0) {
    drawPoints();
  }
}

function ensureCanvasScale() {
  const rect = canvas.getBoundingClientRect();
  if (rect.width > 0 && rect.height > 0) {
    canvas.style.width = `${rect.width}px`;
    canvas.style.height = `${rect.height}px`;
  }
}

function findNearestPoint(imgX, imgY) {
  let nearest = -1;
  let nearestDist = Number.POSITIVE_INFINITY;
  for (let i = 0; i < state.points.length; i += 1) {
    const [px, py] = state.points[i];
    if (!Number.isFinite(px) || !Number.isFinite(py)) {
      continue;
    }
    const d2 = (px - imgX) * (px - imgX) + (py - imgY) * (py - imgY);
    if (d2 < nearestDist) {
      nearestDist = d2;
      nearest = i;
    }
  }
  if (nearest < 0) {
    return -1;
  }
  const maxDistImg = Math.max(8, 14 / Math.max(0.05, state.viewport.scale));
  if (nearestDist > maxDistImg * maxDistImg) {
    return -1;
  }
  return nearest;
}

function setActiveLabelLine() {
  const active = state.labels[state.activeIndex] || `kp_${state.activeIndex + 1}`;
  const activeState = state.status || {};
  const heading = activeState.heading;
  const items = [];
  if (heading !== undefined && heading !== null && Number.isFinite(Number(heading))) {
    items.push(`heading=${Number(heading).toFixed(3)}`);
  }
  if (activeState.refined_success !== undefined) {
    items.push(`refined_success=${Boolean(activeState.refined_success)}`);
  }
  if (activeState.usable_keypoints !== undefined) {
    items.push(`usable=${Boolean(activeState.usable_keypoints)}`);
  }
  if (activeState.edit_applied !== undefined) {
    items.push(`edit_applied=${Boolean(activeState.edit_applied)}`);
  }
  statusLine.textContent = `Active ${state.activeIndex + 1}/${state.labels.length}: ${active} | ${items.join(" | ")}`;
}

function cycleActive(delta) {
  const total = state.points.length;
  if (total <= 0) {
    return;
  }
  state.activeIndex = (state.activeIndex + delta + total) % total;
  setActiveLabelLine();
  renderPointsList();
  draw();
}

function renderPointsList() {
  const rows = [];
  for (let i = 0; i < state.labels.length; i += 1) {
    const point = state.points[i] || [NaN, NaN];
    const isActive = i === state.activeIndex;
    const color = COLORS[i % COLORS.length];
    const pointText = Number.isFinite(point[0]) && Number.isFinite(point[1])
      ? `${point[0].toFixed(2)}, ${point[1].toFixed(2)}`
      : "unset";
    rows.push(
      `<div class="row">
          <span class="dot" style="background:${color};"></span>
          <span style="font-weight:${isActive ? "700" : "400"};">${i + 1}. ${state.labels[i] || "kp_" + (i + 1)}</span>
          <span>${pointText}</span>
        </div>`,
    );
  }
  pointsList.innerHTML = rows.join("");
}

function parseImagePayload(payload) {
  const shape = payload.shape;
  const [h, w, channels = 1] = shape;
  const raw = payload.pixels || "";
  const binary = atob(raw);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) {
    bytes[i] = binary.charCodeAt(i);
  }

  const pixels = new Uint8ClampedArray(w * h * 4);
  const source = state.image.sourceCanvas || document.createElement("canvas");
  source.width = w;
  source.height = h;
  if (channels === 1) {
    for (let i = 0; i < h * w; i += 1) {
      const v = bytes[i] ?? 0;
      const j = i * 4;
      pixels[j] = v;
      pixels[j + 1] = v;
      pixels[j + 2] = v;
      pixels[j + 3] = 255;
    }
  } else if (channels === 2) {
    for (let i = 0; i < h * w; i += 1) {
      const j = i * 4;
      const b = i * 2;
      const v = bytes[b] ?? 0;
      pixels[j] = v;
      pixels[j + 1] = bytes[b + 1] ?? v;
      pixels[j + 2] = v;
      pixels[j + 3] = 255;
    }
  } else if (channels >= 3) {
    for (let i = 0; i < h * w; i += 1) {
      const j = i * 4;
      const b = i * channels;
      pixels[j] = bytes[b] ?? 0;
      pixels[j + 1] = bytes[b + 1] ?? 0;
      pixels[j + 2] = bytes[b + 2] ?? 0;
      pixels[j + 3] = channels >= 4 ? bytes[b + 3] ?? 255 : 255;
    }
  }

  const imgData = new ImageData(pixels, w, h);
  const srcCtx = source.getContext("2d");
  if (srcCtx) {
    srcCtx.putImageData(imgData, 0, 0);
  }
  state.image = {
    width: w,
    height: h,
    sourceCanvas: source,
  };
}

function renderStateLine() {
  if (!state.roiState) {
    stateLine.textContent = "No ROI loaded.";
    return;
  }
  const { position, total, frame_idx, roi_idx } = state.roiState;
  const filter = state.roiState.filter_mode || "unknown";
  const search = state.roiState.search ? ` | search=${state.roiState.search}` : "";
  const base = `ROI ${position + 1}/${total} | frame=${frame_idx} | roi=${roi_idx} | filter=${filter}${search}`;
  stateLine.textContent = base;
  if (filterMode && state.roiState.filter_mode) {
    filterMode.value = state.roiState.filter_mode;
  }
  if (searchBox && state.roiState.search !== undefined) {
    searchBox.value = state.roiState.search || "";
  }
  if (autoAdvanceBox && state.roiState.auto_advance_on_save !== undefined) {
    autoAdvanceBox.checked = Boolean(state.roiState.auto_advance_on_save);
  }
}

function renderDatasetStatus() {
  if (!state.roiState || !state.roiState.registry_enabled) {
    if (datasetPanel) {
      datasetPanel.classList.add("hidden");
    }
    return;
  }
  if (datasetPanel) {
    datasetPanel.classList.remove("hidden");
  }
  const summary = state.roiState.dataset_summary || {};
  const dataset = state.roiState.dataset || {};
  const reviewStatus = summary.review_status || state.roiState.review_status || {};
  const pieces = [];
  if (state.roiState.dataset_id) {
    pieces.push(`dataset=${state.roiState.dataset_id}`);
  }
  if (dataset.recording_id) {
    pieces.push(`recording=${dataset.recording_id}`);
  }
  if (summary.total_rois !== undefined) {
    pieces.push(`total=${summary.total_rois}`);
  }
  if (summary.remaining_failures !== undefined) {
    pieces.push(`remaining=${summary.remaining_failures}`);
  }
  if (summary.reviewable_failures !== undefined) {
    pieces.push(`reviewable=${summary.reviewable_failures}`);
  }
  if (summary.usable_keypoints !== undefined) {
    pieces.push(`usable=${summary.usable_keypoints}`);
  }
  if (summary.manual_corrections !== undefined) {
    pieces.push(`manual=${summary.manual_corrections}`);
  }
  if (reviewStatus.state) {
    pieces.push(`status=${reviewStatus.state}`);
  }
  if (summary.error) {
    pieces.push(`summary_error=${summary.error}`);
  }
  if (datasetStatusLine) {
    datasetStatusLine.textContent = pieces.join(" | ");
  }
  if (datasetSelect && state.roiState.dataset_id) {
    datasetSelect.value = state.roiState.dataset_id;
  }
}

async function loadState() {
  const response = await fetch("/api/state");
  const payload = await response.json();
  if (!payload.ok) {
    setMessage("Failed to load state.", true);
    return;
  }
  state.labels = payload.state.labels || [];
  state.status = payload.state.status || {};
  state.roiState = payload.state;
  reasonLine.textContent = payload.state.reason || "";
  setActiveLabelLine();
  renderDatasetStatus();
}

function updateFromPayload(payload) {
  state.roiState = payload.state;
  state.labels = payload.labels || [];
  state.points = (payload.points || []).map((pt) => {
    const x = pt && pt[0] !== null && pt[0] !== undefined ? Number(pt[0]) : NaN;
    const y = pt && pt[1] !== null && pt[1] !== undefined ? Number(pt[1]) : NaN;
    return [x, y];
  });
  state.status = payload.status || {};
  reasonLine.textContent = payload.reason || "";
  state.activeIndex = Math.min(state.activeIndex, Math.max(0, state.points.length - 1));
  if (!Number.isFinite(state.activeIndex) || state.activeIndex < 0) {
    state.activeIndex = 0;
  }
  parseImagePayload(payload.roi_image);
  setViewportFromImage();
  setActiveLabelLine();
  renderPointsList();
  renderStateLine();
  renderDatasetStatus();
  draw();
}

function datasetOptionLabel(dataset) {
  const parts = [];
  if (dataset.dataset_id) {
    parts.push(dataset.dataset_id);
  }
  if (dataset.recording_id) {
    parts.push(dataset.recording_id);
  }
  if (dataset.camera_serial) {
    parts.push(`cam${dataset.camera_serial}`);
  }
  if (dataset.status) {
    parts.push(dataset.status);
  }
  if (dataset.keypoint_review_state) {
    parts.push(`review=${dataset.keypoint_review_state}`);
  } else if (dataset.keypoint_review_reason) {
    parts.push(`review=${dataset.keypoint_review_reason}`);
  }
  return parts.join(" | ") || "(unnamed dataset)";
}

async function loadRegistryDatasets() {
  if (!datasetPanel || !datasetSelect) {
    return;
  }
  const response = await fetch("/api/registry/datasets");
  const payload = await response.json();
  if (!payload.ok || !payload.enabled) {
    datasetPanel.classList.add("hidden");
    return;
  }
  datasetPanel.classList.remove("hidden");
  const current = payload.dataset_id || (state.roiState && state.roiState.dataset_id) || "";
  datasetSelect.innerHTML = "";
  for (const dataset of payload.datasets || []) {
    const option = document.createElement("option");
    option.value = dataset.dataset_id || "";
    option.textContent = datasetOptionLabel(dataset);
    if (option.value === current) {
      option.selected = true;
    }
    datasetSelect.appendChild(option);
  }
}

async function selectDataset(datasetId) {
  if (!datasetId) {
    return;
  }
  setMessage("Switching dataset...");
  const response = await fetch("/api/registry/select", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ dataset_id: datasetId }),
  });
  const payload = await response.json();
  if (!payload.ok) {
    setMessage(payload.details || "Dataset switch failed.", true);
    return;
  }
  state.roiState = payload.state;
  renderDatasetStatus();
  setMessage("Dataset loaded.");
  await loadRegistryDatasets();
  await loadCurrentRoi();
}

function renderReadback(result) {
  if (!result) {
    readbackLine.textContent = "";
    return;
  }
  const readback = result.readback || {};
  const status = readback.status || {};
  const items = [
    result.action ? `action=${result.action}` : "action=save",
    `changed=${Boolean(result.changed)}`,
    readback.roi_idx !== undefined ? `roi=${readback.roi_idx}` : null,
    readback.frame_idx !== undefined ? `frame=${readback.frame_idx}` : null,
    result.stale_touched !== undefined ? `stale=${result.stale_touched}` : null,
    status.refined_success !== undefined ? `success=${Boolean(status.refined_success)}` : null,
    status.usable_keypoints !== undefined ? `usable=${Boolean(status.usable_keypoints)}` : null,
    readback.reason ? `reason=${readback.reason}` : null,
  ].filter(Boolean);
  readbackLine.textContent = items.join(" | ");
}

async function loadCurrentRoi() {
  const response = await fetch("/api/roi/current");
  const payload = await response.json();
  if (!payload.ok) {
    reasonLine.textContent = payload.error || "No ROIs remain.";
    state.points = [];
    draw();
    setMessage("No ROI available.", true);
    return;
  }
  updateFromPayload(payload);
}

async function navigate(delta) {
  const response = await fetch("/api/nav", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ delta }),
  });
  const payload = await response.json();
  if (!payload.ok) {
    setMessage("Navigation failed.", true);
    return;
  }
  if (payload.moved) {
    setMessage("Navigated.");
  } else {
    setMessage("At edge.");
  }
  await loadCurrentRoi();
}

async function saveCurrent() {
  const advance = Boolean(autoAdvanceBox && autoAdvanceBox.checked);
  const response = await fetch("/api/roi/current/save", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ points: state.points, advance }),
  });
  const payload = await response.json();
  if (!payload.ok) {
    setMessage("Save failed.", true);
    return;
  }
  const result = payload.result || {};
  renderReadback(result);
  if (result.changed) {
    setMessage("Saved.");
    await loadCurrentRoi();
  } else {
    setMessage("No changes.");
  }
}

async function runAction(action) {
  const advance = Boolean(autoAdvanceBox && autoAdvanceBox.checked);
  const response = await fetch("/api/roi/current/action", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ action, advance }),
  });
  const payload = await response.json();
  if (!payload.ok) {
    setMessage(payload.details || "Action failed.", true);
    return;
  }
  renderReadback(payload.result || {});
  setMessage(`Action saved: ${action}.`);
  await loadCurrentRoi();
}

async function applyFilter() {
  const response = await fetch("/api/filter", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      filter_mode: filterMode ? filterMode.value : "failed",
      search: searchBox ? searchBox.value : "",
    }),
  });
  const payload = await response.json();
  if (!payload.ok) {
    setMessage(payload.details || "Filter failed.", true);
    return;
  }
  setMessage("Filter applied.");
  await loadCurrentRoi();
}

async function jumpToTarget() {
  const roiRaw = document.getElementById("jump-roi").value;
  const frameRaw = document.getElementById("jump-frame").value;
  const payloadBody = {};
  if (roiRaw !== "") {
    payloadBody.roi_idx = Number.parseInt(roiRaw, 10);
  }
  if (frameRaw !== "") {
    payloadBody.frame_idx = Number.parseInt(frameRaw, 10);
  }
  const response = await fetch("/api/jump", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payloadBody),
  });
  const payload = await response.json();
  if (!payload.ok) {
    setMessage(payload.details || "Jump failed.", true);
    return;
  }
  setMessage("Jumped.");
  await loadCurrentRoi();
}

async function applyReviewStatus(reviewState) {
  const response = await fetch("/api/review_status", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ state: reviewState }),
  });
  const payload = await response.json();
  if (!payload.ok) {
    setMessage(payload.details || "Review status failed.", true);
    return;
  }
  renderReadback(payload.result || {});
  setMessage(`Review status: ${reviewState}.`);
  await loadState();
}

function ensureEvents() {
  canvas.addEventListener("contextmenu", (event) => {
    event.preventDefault();
  });

  canvas.addEventListener("mousedown", (event) => {
    const x = event.offsetX;
    const y = event.offsetY;
    const [imgX, imgY] = viewToImg(x, y);
    const idx = findNearestPoint(imgX, imgY);
    if (event.shiftKey || event.button === 1 || event.button === 2) {
      state.viewport.isPanning = true;
      state.viewport.panStartX = x;
      state.viewport.panStartY = y;
      canvas.style.cursor = "grabbing";
      event.preventDefault();
      return;
    }
    if (idx >= 0 && state.points[idx]) {
      state.activeIndex = idx;
      setActiveLabelLine();
      const [vx, vy] = imgToView(state.points[idx]);
      state.viewport.draggingPoint = idx;
      state.viewport.dragOffsetX = vx - x;
      state.viewport.dragOffsetY = vy - y;
      event.preventDefault();
      return;
    }
    if (state.points.length > 0) {
      const active = Math.min(Math.max(0, state.activeIndex), state.points.length - 1);
      state.points[active] = clampPoint([imgX, imgY]);
      state.viewport.draggingPoint = active;
      state.viewport.dragOffsetX = 0;
      state.viewport.dragOffsetY = 0;
      renderPointsList();
      setActiveLabelLine();
      draw();
      event.preventDefault();
      return;
    }
    state.viewport.isPanning = true;
    state.viewport.panStartX = x;
    state.viewport.panStartY = y;
    canvas.style.cursor = "grabbing";
    event.preventDefault();
  });

  window.addEventListener("mousemove", (event) => {
    if (!canvas.matches(":hover")) {
      return;
    }
    const rect = canvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;

    if (state.viewport.draggingPoint !== null) {
      const [imgX, imgY] = viewToImg(x + state.viewport.dragOffsetX, y + state.viewport.dragOffsetY);
      state.points[state.viewport.draggingPoint] = clampPoint([imgX, imgY]);
      renderPointsList();
      setActiveLabelLine();
      draw();
      return;
    }

    if (state.viewport.isPanning) {
      const dx = x - state.viewport.panStartX;
      const dy = y - state.viewport.panStartY;
      state.viewport.offsetX += dx;
      state.viewport.offsetY += dy;
      state.viewport.panStartX = x;
      state.viewport.panStartY = y;
      draw();
    }
  });

  window.addEventListener("mouseup", () => {
    state.viewport.draggingPoint = null;
    state.viewport.isPanning = false;
    canvas.style.cursor = "grab";
  });

  canvas.addEventListener("dblclick", (event) => {
    const rect = canvas.getBoundingClientRect();
    const [imgX, imgY] = viewToImg(event.offsetX, event.offsetY);
    const idx = findNearestPoint(imgX, imgY);
    if (idx >= 0 && state.points[idx]) {
      state.activeIndex = idx;
      setActiveLabelLine();
      draw();
    }
  });

  canvas.addEventListener("wheel", (event) => {
    event.preventDefault();
    const rect = canvas.getBoundingClientRect();
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
    if (key === "f" || key === "F") {
      setViewportFromImage();
      return;
    }
    if (key === "[") {
      event.preventDefault();
      cycleActive(-1);
      return;
    }
    if (key === "]") {
      event.preventDefault();
      cycleActive(1);
      return;
    }
    if (key === "t" || key === "T") {
      event.preventDefault();
      state.showText = !state.showText;
      draw();
      return;
    }
    if (key === "1" && !event.ctrlKey) {
      const idx = 0;
      if (idx < state.points.length) {
        state.activeIndex = idx;
        setActiveLabelLine();
      }
      return;
    }
    const digit = Number.parseInt(key, 10);
    if (Number.isInteger(digit) && digit >= 1 && digit <= 9) {
      const idx = digit - 1;
      if (idx < state.points.length) {
        state.activeIndex = idx;
        setActiveLabelLine();
      }
      return;
    }
    if (key === "s" || key === "S") {
      event.preventDefault();
      await saveCurrent();
      return;
    }
    if (key === "x" || key === "X") {
      event.preventDefault();
      await runAction("mark_no_keypoints");
      return;
    }
    if (key === "d" || key === "D") {
      event.preventDefault();
      await runAction("mark_detection_issue");
      return;
    }
    if (key === "c" || key === "C") {
      event.preventDefault();
      await runAction("clear_failure_label");
      return;
    }
    if (key === "b" || key === "B") {
      event.preventDefault();
      await runAction("flag_followup");
      return;
    }
    if (key === "a") {
      event.preventDefault();
      await applyReviewStatus("approved");
      return;
    }
    if (key === "N") {
      event.preventDefault();
      await applyReviewStatus("needs_review");
      return;
    }
    if (key === "R") {
      event.preventDefault();
      await applyReviewStatus("rejected");
      return;
    }
    if (key === "P") {
      event.preventDefault();
      await applyReviewStatus("pending");
      return;
    }
    if (key === "r") {
      event.preventDefault();
      await loadCurrentRoi();
      setMessage("Reset points from current data.");
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
    if (key === "q" || key === "Q") {
      window.close();
    }
  });

  document.getElementById("fit-btn").addEventListener("click", setViewportFromImage);
  document.getElementById("save-btn").addEventListener("click", saveCurrent);
  document.getElementById("next-btn").addEventListener("click", () => navigate(1));
  document.getElementById("prev-btn").addEventListener("click", () => navigate(-1));
  document.getElementById("no-keypoints-btn").addEventListener("click", () => runAction("mark_no_keypoints"));
  document.getElementById("detection-issue-btn").addEventListener("click", () => runAction("mark_detection_issue"));
  document.getElementById("clear-label-btn").addEventListener("click", () => runAction("clear_failure_label"));
  document.getElementById("followup-btn").addEventListener("click", () => runAction("flag_followup"));
  document.getElementById("apply-filter-btn").addEventListener("click", applyFilter);
  document.getElementById("jump-btn").addEventListener("click", jumpToTarget);
  document.getElementById("approve-btn").addEventListener("click", () => applyReviewStatus("approved"));
  document.getElementById("needs-review-btn").addEventListener("click", () => applyReviewStatus("needs_review"));
  document.getElementById("reject-btn").addEventListener("click", () => applyReviewStatus("rejected"));
  document.getElementById("pending-btn").addEventListener("click", () => applyReviewStatus("pending"));
  if (datasetRefreshBtn) {
    datasetRefreshBtn.addEventListener("click", loadRegistryDatasets);
  }
  if (datasetSelect) {
    datasetSelect.addEventListener("change", () => selectDataset(datasetSelect.value));
  }
}

async function bootstrap() {
  ensureCanvasScale();
  ensureEvents();
  await loadRegistryDatasets();
  await loadState();
  await loadCurrentRoi();
}

bootstrap();
