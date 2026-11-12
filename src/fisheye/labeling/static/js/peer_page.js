// Peer review page: read-only view of other labelers' applied work, with advisory frame flags.
import { h, render } from "../vendor/preact.module.js";
import { useEffect, useRef, useState } from "../vendor/preact-hooks.module.js";
import htm from "../vendor/htm.module.js";
import { adjacentRow, imageRgba, maskOverlayRgba, taskGroups } from "./inspect_model.js";
import { authParams, withParams } from "./queue_model.js";

const html = htm.bind(h);
const params = authParams(window.location.search);
const NOTE_MAX = 1000;

async function readJson(response) {
  let body;
  try { body = await response.json(); } catch { body = { ok: false, details: `Server answered ${response.status}.` }; }
  if (!response.ok || !body.ok) throw new Error(body.details || body.error || `Request failed (${response.status}).`);
  return body;
}

const getJson = (url) => fetch(withParams(url, params), { cache: "no-store" }).then(readJson);
const postJson = (url, body) =>
  fetch(withParams(url, params), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  }).then(readJson);

function taskLabel(task) {
  const kind = task.workflow_kind === "keypoints"
    ? "Keypoints"
    : "Mask · " + String(task.component_name || "").replace(/^subject_/, "").replaceAll("_", " ");
  const flags = task.open_flag_count ? ` · ${task.open_flag_count} flagged` : "";
  return `${task.recording_id} · ${kind} · ${task.applied_rows} applied${flags}`;
}

function pixelCanvas({ rgba, width, height }) {
  const surface = document.createElement("canvas");
  surface.width = width; surface.height = height;
  surface.getContext("2d").putImageData(new ImageData(rgba, width, height), 0, 0);
  return surface;
}

function Viewer({ row }) {
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
    if (row.workflow_kind === "subject_mask_component") {
      const overlay = maskOverlayRgba(row.applied_mask, null, "applied");
      if (overlay) g.drawImage(pixelCanvas(overlay), 0, 0, canvas.width, canvas.height);
      return;
    }
    const k = canvas.width / image.width;
    const styles = typeof keypointStyles === "function" ? keypointStyles(row.labels) : [];
    const r = 5 * dpr;
    row.applied_points.forEach((p, i) => {
      if (!p || p[0] === null || p[1] === null) return;
      const color = styles[i]?.color || "#ffd166";
      g.beginPath(); g.arc(p[0] * k, p[1] * k, r, 0, Math.PI * 2);
      g.fillStyle = styles[i]?.hollow ? "#0f1411" : color; g.fill();
      g.lineWidth = (styles[i]?.hollow ? 2.5 : 1) * dpr; g.strokeStyle = styles[i]?.hollow ? color : "#0f1411"; g.stroke();
    });
  }, [row]);
  return html`<div class="inspect-stage"><canvas ref=${canvasRef}></canvas></div>`;
}

function Flags({ row, onChanged }) {
  const [note, setNote] = useState("");
  const [busy, setBusy] = useState(false);
  const [problem, setProblem] = useState("");
  async function add(event) {
    event.preventDefault();
    setBusy(true); setProblem("");
    try {
      await postJson("/api/peer/flags", { task_id: row.task_id, roi_idx: row.roi_idx, note });
      setNote("");
      onChanged();
    } catch (error) { setProblem(error.message); } finally { setBusy(false); }
  }
  async function resolve(flag) {
    setBusy(true); setProblem("");
    try {
      await postJson(`/api/peer/flags/${encodeURIComponent(flag.flag_id)}/resolve`, {});
      onChanged();
    } catch (error) { setProblem(error.message); } finally { setBusy(false); }
  }
  return html`<div class="peer-flags">
    <h2 class="small">Flags on this frame</h2>
    ${problem && html`<div class="notice notice-danger" role="alert">${problem}</div>`}
    ${row.flags.length === 0 && html`<p class="muted small">No flags yet.</p>`}
    <ul>${row.flags.map((flag) => html`<li class=${`flag flag-${flag.state}`}>
      <div><b>${flag.flagged_by}</b> <span class="muted small">${flag.created_at_utc}</span>
        ${flag.state === "resolved" && html` <span class="muted small">· resolved by ${flag.resolved_by}</span>`}</div>
      <p>${flag.note}</p>
      ${flag.can_resolve && html`<button type="button" class="btn btn-small" disabled=${busy} onClick=${() => resolve(flag)}>Mark resolved</button>`}
    </li>`)}</ul>
    <form onSubmit=${add}>
      <label class="small">Flag this frame for ${row.owner || "its labeler"}
        <textarea rows="3" maxlength=${NOTE_MAX} value=${note} onInput=${(e) => setNote(e.target.value)}
          placeholder="What looks wrong?"></textarea></label>
      <button type="submit" class="btn btn-primary btn-small" disabled=${busy || !note.trim()}>Add flag</button>
      <span class="muted small"> Flags are notes only; they never change labels.</span>
    </form>
  </div>`;
}

function App() {
  const [tasks, setTasks] = useState(null);
  const [user, setUser] = useState("");
  const [taskId, setTaskId] = useState(new URLSearchParams(location.search).get("task_id") || "");
  const [task, setTask] = useState(null);
  const [roi, setRoi] = useState(null);
  const [row, setRow] = useState(null);
  const [revision, setRevision] = useState(0);
  const [problem, setProblem] = useState("");

  useEffect(() => {
    getJson("/api/peer/tasks").then((b) => { setTasks(b.tasks); setUser(b.user || ""); }).catch((e) => setProblem(e.message));
  }, [revision]);
  useEffect(() => {
    if (!taskId) return;
    setProblem("");
    const query = new URLSearchParams(params);
    query.set("task_id", taskId);
    history.replaceState({}, "", `?${query}`);
    getJson(`/api/peer/task?task_id=${encodeURIComponent(taskId)}`)
      .then((b) => {
        setTask(b);
        setRoi((current) => (b.rows.some((r) => r.roi_idx === current) ? current : b.rows[0]?.roi_idx ?? null));
      })
      .catch((e) => setProblem(e.message));
  }, [taskId, revision]);
  useEffect(() => {
    if (!task || roi === null) { setRow(null); return; }
    getJson(`/api/peer/row?task_id=${encodeURIComponent(task.task_id)}&roi_idx=${roi}`)
      .then(setRow).catch((e) => setProblem(e.message));
  }, [task, roi]);

  const rows = task ? task.rows : [];
  useEffect(() => {
    const onKey = (event) => {
      if (event.ctrlKey || event.metaKey || event.altKey) return;
      if (["SELECT", "TEXTAREA", "INPUT"].includes(event.target.tagName)) return;
      if (event.key === "n") setRoi((current) => adjacentRow(rows, current, 1));
      if (event.key === "p") setRoi((current) => adjacentRow(rows, current, -1));
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [rows]);

  return html`
    <header class="topbar"><span class="brand">Palette labeling · Review others' work</span><span class="spacer"></span>
      ${user && html`<span class="muted small">Signed in as <b class="ink">${user}</b> · </span>`}
      <a class="small" href=${withParams("/queue", params)}>Back to your queue</a></header>
    <main class="inspect">
      ${problem && html`<div class="notice notice-danger" role="alert">${problem}</div>`}
      <p class="muted small">Read-only: you see applied work only, and nothing here changes labels or sessions.</p>
      <label class="inspect-picker">Task
        <select value=${taskId} onChange=${(e) => setTaskId(e.target.value)}>
          <option value="">${tasks ? "Choose a task…" : "Loading…"}</option>
          ${(tasks ? taskGroups(tasks) : []).map((group) => html`<optgroup label=${group.assignee}>
            ${group.tasks.map((t) => html`<option value=${t.task_id}>${taskLabel(t)}</option>`)}</optgroup>`)}
        </select></label>
      ${task && html`<div class="inspect-body">
        <aside class="card inspect-rows">
          ${rows.length === 0 && html`<p class="muted small">No applied rows yet.</p>`}
          <ul>${rows.map((r) => html`<li><button type="button" class="row-item status-applied" aria-current=${r.roi_idx === roi}
            onClick=${() => setRoi(r.roi_idx)}><span class="mono">ROI ${r.roi_idx}</span>
            <span>${r.open_flag_count ? `${r.open_flag_count} flagged` : "Applied"}</span></button></li>`)}</ul>
        </aside>
        <section class="card inspect-view">
          <div class="inspect-head">
            <div><b>${task.recording_id}</b> · ${task.workflow_kind === "keypoints" ? "Keypoints" : `Mask · ${task.component_name}`} · ROI ${roi ?? "–"}
              <div class="muted small">${row ? `Applied by ${row.labeler || task.owner || "unknown"}${row.applied_at_utc ? ` · ${row.applied_at_utc}` : ""}` : ""}</div></div>
          </div>
          ${row && html`<${Viewer} row=${row} />`}
          ${row && html`<${Flags} row=${row} onChanged=${() => setRevision((n) => n + 1)} />`}
          <p class="muted small"><kbd>N</kbd>/<kbd>P</kbd> next / previous row.</p>
        </section>
      </div>`}
    </main>`;
}

render(html`<${App} />`, document.getElementById("app"));
