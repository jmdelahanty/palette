// Admin inspect page: read-only view of any labeler's applied and saved work.
import { h, render } from "../vendor/preact.module.js";
import { useEffect, useRef, useState } from "../vendor/preact-hooks.module.js";
import htm from "../vendor/htm.module.js";
import {
  STATUS_LABELS,
  adjacentRow,
  filterRows,
  imageRgba,
  maskDifference,
  maskOverlayRgba,
  taskGroups,
  taskLabel,
} from "./inspect_model.js";

const html = htm.bind(h);

async function getJson(url) {
  const response = await fetch(url, { cache: "no-store" });
  let body;
  try { body = await response.json(); } catch { body = { ok: false, details: `Server answered ${response.status}.` }; }
  if (!response.ok || !body.ok) throw new Error(body.details || body.error || `Request failed (${response.status}).`);
  return body;
}

function pixelCanvas({ rgba, width, height }) {
  const surface = document.createElement("canvas");
  surface.width = width; surface.height = height;
  surface.getContext("2d").putImageData(new ImageData(rgba, width, height), 0, 0);
  return surface;
}

function Viewer({ row, view }) {
  const canvasRef = useRef(null);
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !row) return;
    const image = imageRgba(row.image);
    const box = canvas.parentElement.getBoundingClientRect();
    const scale = Math.max(1, Math.min(box.width / image.width, (box.height || box.width) / image.height));
    const dpr = window.devicePixelRatio || 1;
    canvas.style.width = Math.floor(image.width * scale) + "px";
    canvas.style.height = Math.floor(image.height * scale) + "px";
    canvas.width = Math.floor(image.width * scale * dpr);
    canvas.height = Math.floor(image.height * scale * dpr);
    const g = canvas.getContext("2d");
    g.imageSmoothingEnabled = false;
    g.drawImage(pixelCanvas(image), 0, 0, canvas.width, canvas.height);
    const k = canvas.width / image.width;
    if (row.workflow_kind === "subject_mask_component") {
      const overlay = maskOverlayRgba(row.applied_mask, row.saved_mask, view);
      if (overlay) g.drawImage(pixelCanvas(overlay), 0, 0, canvas.width, canvas.height);
      return;
    }
    const styles = typeof keypointStyles === "function" ? keypointStyles(row.labels) : [];
    const r = 5 * dpr;
    const point = (p) => (p && p[0] !== null && p[1] !== null ? [p[0] * k, p[1] * k] : null);
    row.applied_points.forEach((p, i) => {
      const a = point(p);
      const s = row.saved_points ? point(row.saved_points[i]) : null;
      const color = styles[i]?.color || "#ffd166";
      if (a && s && view === "both" && (a[0] !== s[0] || a[1] !== s[1])) {
        g.strokeStyle = "rgba(255,255,255,0.7)"; g.lineWidth = 1.5 * dpr;
        g.beginPath(); g.moveTo(a[0], a[1]); g.lineTo(s[0], s[1]); g.stroke();
      }
      if (a && view !== "saved") {
        g.beginPath(); g.arc(a[0], a[1], r, 0, Math.PI * 2);
        g.fillStyle = styles[i]?.hollow ? "#0f1411" : color; g.fill();
        g.lineWidth = (styles[i]?.hollow ? 2.5 : 1) * dpr; g.strokeStyle = styles[i]?.hollow ? color : "#0f1411"; g.stroke();
      }
      if (s && view !== "applied") {
        g.beginPath(); g.arc(s[0], s[1], r + 2 * dpr, 0, Math.PI * 2);
        g.lineWidth = 2 * dpr; g.strokeStyle = "#ffffff"; g.stroke();
      }
    });
  }, [row, view]);
  return html`<div class="inspect-stage"><canvas ref=${canvasRef}></canvas></div>`;
}

function App() {
  const [tasks, setTasks] = useState(null);
  const [taskId, setTaskId] = useState(new URLSearchParams(location.search).get("task_id") || "");
  const [task, setTask] = useState(null);
  const [status, setStatus] = useState("");
  const [roi, setRoi] = useState(null);
  const [row, setRow] = useState(null);
  const [view, setView] = useState("both");
  const [problem, setProblem] = useState("");

  useEffect(() => { getJson("/api/admin/inspect/tasks").then((b) => setTasks(b.tasks)).catch((e) => setProblem(e.message)); }, []);
  useEffect(() => {
    if (!taskId) return;
    setTask(null); setRow(null); setProblem("");
    history.replaceState({}, "", `?task_id=${encodeURIComponent(taskId)}`);
    getJson(`/api/admin/inspect/task?task_id=${encodeURIComponent(taskId)}`)
      .then((b) => { setTask(b); const first = b.rows.find((r) => r.status === "saved") || b.rows[0]; setRoi(first ? first.roi_idx : null); })
      .catch((e) => setProblem(e.message));
  }, [taskId]);
  useEffect(() => {
    if (!task || roi === null) return;
    setRow(null);
    getJson(`/api/admin/inspect/row?task_id=${encodeURIComponent(task.task_id)}&roi_idx=${roi}`)
      .then(setRow).catch((e) => setProblem(e.message));
  }, [task, roi]);

  const rows = task ? filterRows(task.rows, status) : [];
  useEffect(() => {
    const onKey = (event) => {
      if (event.ctrlKey || event.metaKey || event.altKey || event.target.tagName === "SELECT") return;
      if (event.key === "n") setRoi((current) => adjacentRow(rows, current, 1));
      if (event.key === "p") setRoi((current) => adjacentRow(rows, current, -1));
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [rows]);

  const diff = row && row.workflow_kind === "subject_mask_component" ? maskDifference(row.applied_mask, row.saved_mask) : null;
  return html`
    <header class="topbar"><span class="brand">Palette admin · Inspect labels</span><span class="spacer"></span>
      <span class="muted small">Read-only: nothing here changes labels or sessions</span></header>
    <main class="inspect">
      ${problem && html`<div class="notice notice-danger" role="alert">${problem}</div>`}
      <label class="inspect-picker">Task
        <select value=${taskId} onChange=${(e) => setTaskId(e.target.value)}>
          <option value="">${tasks ? "Choose a task…" : "Loading…"}</option>
          ${(tasks ? taskGroups(tasks) : []).map((group) => html`<optgroup label=${group.assignee}>
            ${group.tasks.map((t) => html`<option value=${t.task_id}>${taskLabel(t)}</option>`)}</optgroup>`)}
        </select></label>
      ${task && html`<div class="inspect-body">
        <aside class="card inspect-rows">
          <div class="inspect-counts">
            ${["", "saved", "applied", "untouched"].map((s) => html`<button type="button" class="btn btn-small" aria-pressed=${status === s}
              onClick=${() => setStatus(s)}>${s ? STATUS_LABELS[s] : "All"} ${s ? task.counts[s] : task.rows.length}</button>`)}
          </div>
          <ul>${rows.map((r) => html`<li><button type="button" class=${`row-item status-${r.status}`} aria-current=${r.roi_idx === roi}
            onClick=${() => setRoi(r.roi_idx)}><span class="mono">ROI ${r.roi_idx}</span><span>${STATUS_LABELS[r.status]}</span></button></li>`)}</ul>
        </aside>
        <section class="card inspect-view">
          <div class="inspect-head">
            <div><b>${task.recording_id}</b> · ${task.workflow_kind === "keypoints" ? "Keypoints" : `Mask · ${task.component_name}`} · ROI ${roi}
              <div class="muted small">${row ? `${STATUS_LABELS[row.status]}${row.labeler ? ` · ${row.labeler}` : ""}${row.saved_at_utc ? ` · ${row.saved_at_utc}` : ""}` : "Loading…"}</div></div>
            <span class="spacer"></span>
            <div class="segmented">${["both", "applied", "saved"].map((v) => html`<button type="button" class="btn btn-small"
              aria-pressed=${view === v} onClick=${() => setView(v)}>${v === "both" ? "Both" : v === "applied" ? "Applied" : "Saved"}</button>`)}</div>
          </div>
          ${row && html`<${Viewer} row=${row} view=${view} />`}
          <p class="muted small">
            ${row && row.workflow_kind === "subject_mask_component"
              ? html`Green: applied mask · amber: saved, not applied · teal: both agree.${diff ? ` Saved edit adds ${diff.added} px and removes ${diff.removed} px.` : ""}`
              : html`Dots: applied landmarks · white rings: saved, not applied (lines show moved points).`}
            ${" "}<kbd>N</kbd>/<kbd>P</kbd> next / previous row.</p>
        </section>
      </div>`}
    </main>`;
}

render(html`<${App} />`, document.getElementById("app"));
