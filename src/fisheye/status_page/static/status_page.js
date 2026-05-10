"use strict";

const DEFAULT_LIMIT = 50;
const HEARTBEAT_POLL_MS = 15000;

const STATE = {
  wideRows: [],
  wideColumns: [],
  selectedDatasetId: null,
  selectedWideRowIndex: null,
  heartbeatSignature: null,
  heartbeatTimer: null,
  refreshing: false,
};

const STATUS_COLUMN_NAMES = new Set([
  "Zarr",
  "Import",
  "BG Full",
  "BG DS",
  "Detect",
  "Detect Quality",
  "Refine Detect",
  "Crop",
  "Keypoints",
  "Refined Keypoints (analysis/train)",
  "Eye Masks",
  "Refined Eye Masks",
  "Arena Assignment",
  "Track",
  "Stimulus",
  "Calib",
  "Tuning",
  "status",
]);

function setHealthLine(ok, text) {
  const line = document.getElementById("health-line");
  line.textContent = text;
  line.className = ok ? "ok" : "error";
}

async function fetchJson(url) {
  const response = await fetch(url, { cache: "no-store" });
  const payload = await response.json();
  if (!response.ok || payload.ok === false) {
    const details = payload.details ? `: ${payload.details}` : "";
    throw new Error(`Request failed (${response.status})${details}`);
  }
  return payload;
}

function escapeHtml(value) {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function valueToText(value) {
  if (value === null || value === undefined) {
    return "";
  }
  if (typeof value === "object") {
    try {
      return JSON.stringify(value);
    } catch (_err) {
      return String(value);
    }
  }
  return String(value);
}

function statusClassForText(text) {
  const upper = text.trim().toUpperCase();
  if (!upper) {
    return "";
  }
  if (upper.startsWith("OK")) {
    return "status-ok";
  }
  if (upper.startsWith("MISS")) {
    return "status-miss";
  }
  if (upper.startsWith("STALE")) {
    return "status-stale";
  }
  if (upper.startsWith("UNVER")) {
    return "status-unver";
  }
  if (upper.startsWith("ERR") || upper.startsWith("FAIL")) {
    return "status-error";
  }
  if (upper === "NA" || upper === "N/A") {
    return "status-na";
  }
  return "";
}

function renderRowsAsTable({
  containerId,
  rows,
  columns,
  emptyMessage,
  selectable = false,
  selectedRowIndex = null,
}) {
  const wrap = document.getElementById(containerId);
  if (!rows || rows.length === 0) {
    wrap.innerHTML = `<p class="muted">${escapeHtml(emptyMessage)}</p>`;
    return;
  }

  const visibleColumns = columns.length > 0 ? columns : Object.keys(rows[0]);
  let html = "<table><thead><tr>";
  for (const col of visibleColumns) {
    html += `<th>${escapeHtml(col)}</th>`;
  }
  html += "</tr></thead><tbody>";

  rows.forEach((row, index) => {
    const rowClasses = [];
    if (selectable) {
      rowClasses.push("selectable");
    }
    if (selectedRowIndex !== null && selectedRowIndex === index) {
      rowClasses.push("selected");
    }
    const classAttr = rowClasses.length > 0 ? ` class="${rowClasses.join(" ")}"` : "";
    const dataAttr = selectable ? ` data-row-index="${index}"` : "";
    html += `<tr${classAttr}${dataAttr}>`;
    for (const col of visibleColumns) {
      const text = valueToText(row[col]);
      const statusClass =
        STATUS_COLUMN_NAMES.has(col) || text.length <= 24 ? statusClassForText(text) : "";
      const tdClass = statusClass ? ` class="${statusClass}"` : "";
      html += `<td${tdClass}>${escapeHtml(text)}</td>`;
    }
    html += "</tr>";
  });
  html += "</tbody></table>";
  wrap.innerHTML = html;
}

function clearDetailViews(message) {
  document.getElementById("detail-meta").textContent = message;
  document.getElementById("history-meta").textContent = message;
  renderRowsAsTable({
    containerId: "detail-wrap",
    rows: [],
    columns: [],
    emptyMessage: "No detail rows loaded.",
  });
  renderRowsAsTable({
    containerId: "history-wrap",
    rows: [],
    columns: [],
    emptyMessage: "No history rows loaded.",
  });
}

async function loadHealth() {
  const block = document.getElementById("health-json");
  try {
    const response = await fetch("/healthz", { cache: "no-store" });
    const data = await response.json();
    if (data.ok) {
      setHealthLine(true, "Healthy");
    } else {
      setHealthLine(false, "Unhealthy");
    }
    block.textContent = JSON.stringify(data, null, 2);
  } catch (error) {
    setHealthLine(false, "Health check failed");
    block.textContent = String(error);
  }
}

async function loadSummary() {
  const block = document.getElementById("summary-json");
  try {
    const payload = await fetchJson("/api/status/summary");
    block.textContent = JSON.stringify(payload.summary, null, 2);
  } catch (error) {
    block.textContent = String(error);
  }
}

function buildWideUrl() {
  const q = document.getElementById("filter-q").value.trim();
  const zarrUse = document.getElementById("filter-use").value.trim();
  const onlyBlocking = document.getElementById("filter-blocking").checked;

  const params = new URLSearchParams();
  params.set("limit", String(DEFAULT_LIMIT));
  params.set("offset", "0");
  if (q) {
    params.set("q", q);
  }
  if (zarrUse) {
    params.set("zarr_use", zarrUse);
  }
  if (onlyBlocking) {
    params.set("only_blocking", "1");
  }
  return `/api/status/wide?${params.toString()}`;
}

function visibleWideColumns(payload) {
  const hidden = new Set(payload.hidden_columns || []);
  if (Array.isArray(payload.columns) && payload.columns.length > 0) {
    return payload.columns.filter((col) => !hidden.has(col));
  }
  if (payload.rows && payload.rows.length > 0) {
    return Object.keys(payload.rows[0]).filter((col) => !hidden.has(col) && !col.startsWith("_"));
  }
  return [];
}

function attachWideRowHandlers() {
  const wrap = document.getElementById("wide-table-wrap");
  for (const tr of wrap.querySelectorAll("tbody tr[data-row-index]")) {
    tr.addEventListener("click", () => {
      const idx = Number.parseInt(tr.getAttribute("data-row-index"), 10);
      if (Number.isFinite(idx)) {
        selectWideRow(idx);
      }
    });
  }
}

async function loadWide() {
  const meta = document.getElementById("wide-meta");
  meta.textContent = "Loading...";
  try {
    const payload = await fetchJson(buildWideUrl());
    STATE.wideRows = payload.rows || [];
    STATE.wideColumns = visibleWideColumns(payload);

    let selectedIndex = null;
    if (STATE.selectedDatasetId) {
      selectedIndex = STATE.wideRows.findIndex((row) => row._dataset_id === STATE.selectedDatasetId);
      if (selectedIndex < 0) {
        selectedIndex = null;
      }
    }
    STATE.selectedWideRowIndex = selectedIndex;

    renderRowsAsTable({
      containerId: "wide-table-wrap",
      rows: STATE.wideRows,
      columns: STATE.wideColumns,
      emptyMessage: "No rows matched the current filters.",
      selectable: true,
      selectedRowIndex: STATE.selectedWideRowIndex,
    });
    attachWideRowHandlers();
    meta.textContent = `Showing ${payload.returned_rows}/${payload.total_rows} rows (limit=${payload.limit})`;
  } catch (error) {
    meta.textContent = String(error);
    STATE.wideRows = [];
    STATE.selectedWideRowIndex = null;
    renderRowsAsTable({
      containerId: "wide-table-wrap",
      rows: [],
      columns: [],
      emptyMessage: "Failed to load wide status rows.",
    });
  }
}

async function loadDatasetDetails(datasetId) {
  const meta = document.getElementById("detail-meta");
  try {
    const payload = await fetchJson(`/api/status/dataset/${encodeURIComponent(datasetId)}`);
    const rows = payload.rows || [];
    meta.textContent = `Dataset ${datasetId}: ${payload.row_count} step rows`;
    renderRowsAsTable({
      containerId: "detail-wrap",
      rows,
      columns: [
        "step_name",
        "status",
        "run_name",
        "method",
        "coverage_pct",
        "updated_utc",
        "source",
      ],
      emptyMessage: "No step rows for selected dataset.",
    });
  } catch (error) {
    meta.textContent = String(error);
    renderRowsAsTable({
      containerId: "detail-wrap",
      rows: [],
      columns: [],
      emptyMessage: "Failed to load dataset details.",
    });
  }
}

async function loadHistory(datasetId) {
  const meta = document.getElementById("history-meta");
  try {
    const payload = await fetchJson(
      `/api/status/history?dataset_id=${encodeURIComponent(datasetId)}&limit=120`,
    );
    const rows = payload.rows || [];
    meta.textContent = `Dataset ${datasetId}: ${payload.row_count} history rows`;
    renderRowsAsTable({
      containerId: "history-wrap",
      rows,
      columns: [
        "recorded_utc",
        "step_name",
        "status",
        "run_name",
        "method",
        "coverage_pct",
        "updated_utc",
        "source",
      ],
      emptyMessage: "No history rows for selected dataset.",
    });
  } catch (error) {
    meta.textContent = String(error);
    renderRowsAsTable({
      containerId: "history-wrap",
      rows: [],
      columns: [],
      emptyMessage: "Failed to load history.",
    });
  }
}

async function selectWideRow(index) {
  if (index < 0 || index >= STATE.wideRows.length) {
    return;
  }
  STATE.selectedWideRowIndex = index;
  const row = STATE.wideRows[index];
  const datasetId = row._dataset_id || null;
  STATE.selectedDatasetId = datasetId;

  renderRowsAsTable({
    containerId: "wide-table-wrap",
    rows: STATE.wideRows,
    columns: STATE.wideColumns,
    emptyMessage: "No rows matched the current filters.",
    selectable: true,
    selectedRowIndex: STATE.selectedWideRowIndex,
  });
  attachWideRowHandlers();

  if (!datasetId) {
    const ids = Array.isArray(row._dataset_ids) ? row._dataset_ids : [];
    if (ids.length > 1) {
      clearDetailViews(`Ambiguous row selection: ${ids.length} datasets match this row.`);
    } else {
      clearDetailViews("No dataset_id mapping found for this row.");
    }
    return;
  }

  await Promise.all([loadDatasetDetails(datasetId), loadHistory(datasetId)]);
}

async function refreshAfterHeartbeatChange() {
  if (STATE.refreshing) {
    return;
  }
  STATE.refreshing = true;
  try {
    await Promise.all([loadSummary(), loadWide()]);
    if (STATE.selectedDatasetId) {
      await Promise.all([loadDatasetDetails(STATE.selectedDatasetId), loadHistory(STATE.selectedDatasetId)]);
    }
  } finally {
    STATE.refreshing = false;
  }
}

async function pollHeartbeat() {
  const line = document.getElementById("refresh-meta");
  try {
    const payload = await fetchJson("/api/status/heartbeat");
    const hb = payload.heartbeat;
    const signature = `${hb.latest_updated_utc || ""}|${hb.status_rows_total}|${hb.wide_rows_total}`;
    line.textContent =
      `Heartbeat: latest=${hb.latest_updated_utc || "-"} rows=${hb.status_rows_total} wide=${hb.wide_rows_total}`;

    if (STATE.heartbeatSignature === null) {
      STATE.heartbeatSignature = signature;
      return;
    }
    if (signature !== STATE.heartbeatSignature) {
      STATE.heartbeatSignature = signature;
      await refreshAfterHeartbeatChange();
    }
  } catch (error) {
    line.textContent = `Heartbeat failed: ${error}`;
  }
}

function setupFilters() {
  const form = document.getElementById("wide-filter-form");
  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    STATE.selectedWideRowIndex = null;
    await loadWide();
    if (STATE.selectedDatasetId) {
      await Promise.all([loadDatasetDetails(STATE.selectedDatasetId), loadHistory(STATE.selectedDatasetId)]);
    }
  });
}

function startHeartbeatLoop() {
  if (STATE.heartbeatTimer) {
    clearInterval(STATE.heartbeatTimer);
  }
  STATE.heartbeatTimer = setInterval(() => {
    pollHeartbeat();
  }, HEARTBEAT_POLL_MS);
}

async function bootstrap() {
  setupFilters();
  clearDetailViews("Click a wide-row entry to load detail/history.");
  await loadHealth();
  await loadSummary();
  await loadWide();
  await pollHeartbeat();
  startHeartbeatLoop();
}

bootstrap();
