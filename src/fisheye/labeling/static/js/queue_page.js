// Labeler queue page: GET /api/me/queue rendered with Preact + htm.
import { h, render } from "../vendor/preact.module.js";
import { useEffect, useState } from "../vendor/preact-hooks.module.js";
import htm from "../vendor/htm.module.js";
import {
  QUEUE_SCHEMA,
  authParams,
  filterRows,
  queueFilters,
  queueRows,
  queueSummary,
  withParams,
} from "./queue_model.js";

const html = htm.bind(h);
const params = authParams(window.location.search);

async function readJson(response) {
  try {
    return await response.json();
  } catch {
    return { ok: false, error: "invalid_json_response", details: `The server answered ${response.status}.` };
  }
}

function problemFrom(payload, fallback) {
  return {
    title: fallback,
    details: payload.details || payload.error || "",
    code: payload.error || "",
  };
}

function Problem({ problem }) {
  if (!problem) return null;
  return html`<div class="notice notice-danger" role="alert">
    <strong>${problem.title}</strong>
    ${problem.details && html`<p>${problem.details}</p>`}
    ${problem.code && html`<p class="mono muted">Reference: ${problem.code}</p>`}
  </div>`;
}

function Progress({ progress }) {
  return html`<div class="queue-progress">
    <div class="progress-line">
      ${progress.percent !== null &&
      html`<div class="progress-track" role="progressbar" aria-valuemin="0" aria-valuemax="100"
        aria-valuenow=${progress.percent}><div class="progress-fill" style=${`width: ${progress.percent}%`}></div></div>`}
      <span class="mono muted">${progress.text}</span>
    </div>
    ${progress.details.length > 0 &&
    html`<span class=${`small ${progress.awaitingApply ? "text-pending" : "muted"}`}>${progress.details.join(" · ")}</span>`}
  </div>`;
}

function TaskRow({ row, busy, onStart }) {
  return html`<div class="queue-row" data-task-id=${row.taskId}>
    <div class="queue-cell-main">
      <span class="queue-recording" title=${row.dataset ? `Dataset: ${row.dataset}` : ""}>${row.recording}</span>
      ${row.title !== row.taskId && html`<span class="ink">${row.title}</span>`}
      ${row.notes && html`<span class="muted small clamp-2" title=${row.notes}>${row.notes}</span>`}
    </div>
    <span class=${`chip chip-${row.kind}`}>${row.kindLabel}</span>
    <${Progress} progress=${row.progress} />
    <span class=${`queue-state state-${row.shownState}`}>${row.stateLabel}</span>
    <div class="queue-action">
      ${row.action &&
      (row.canStart
        ? html`<button type="button" class="btn btn-primary" disabled=${busy} onClick=${() => onStart(row)}>
            ${busy ? "Opening…" : row.action}
          </button>`
        : html`<button type="button" class="btn" disabled title=${row.notReadyReason}>${row.action}</button>
            ${row.notReadyReason && html`<span class="muted small">${row.notReadyReason}</span>`}`)}
    </div>
  </div>`;
}

function OperatorDetails({ payload }) {
  const [open, setOpen] = useState(false);
  const links = payload.links || {};
  return html`<section class="card">
    <button type="button" class="disclosure" aria-expanded=${open} onClick=${() => setOpen(!open)}>
      <span class="mono muted">${open ? "▾" : "▸"}</span> Operator details
      <span class="muted small">full page, identity check, diagnostics</span>
    </button>
    ${open &&
    html`<div class="disclosure-body">
      <a href=${withParams(links.personal_work || "/my-work", params)}>Full work page</a>
      ${links.identity_probe && html`<a href=${links.identity_probe}>Identity check</a>`}
      <a href=${links.diagnostics}>Diagnostics (JSON)</a>
      ${payload.labeler.status && html`<span class="muted">Labeler status: ${payload.labeler.status}</span>`}
    </div>`}
  </section>`;
}

function QueuePage() {
  const [payload, setPayload] = useState(null);
  const [problem, setProblem] = useState(null);
  const [kind, setKind] = useState("");
  const [starting, setStarting] = useState("");

  useEffect(() => {
    (async () => {
      const response = await fetch(withParams("/api/me/queue", params), { cache: "no-store" });
      const body = await readJson(response);
      if (!response.ok || !body.ok || body.schema !== QUEUE_SCHEMA) {
        setProblem(problemFrom(body, "Palette could not load your queue."));
        return;
      }
      setPayload(body);
    })().catch((error) => setProblem({ title: "Palette could not load your queue.", details: String(error) }));
  }, []);

  async function start(row) {
    setStarting(row.taskId);
    setProblem(null);
    try {
      const response = await fetch(withParams(row.startEndpoint, params), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ client_label: navigator.userAgent, expected_user: params.get("expected_user") || "" }),
      });
      const body = await readJson(response);
      if (!response.ok || !body.ok || !body.session || !body.session.url) {
        setProblem(problemFrom(body, "Palette could not open that task."));
        setStarting("");
        return;
      }
      window.location.href = body.session.url;
    } catch (error) {
      setProblem({ title: "Palette could not open that task.", details: String(error) });
      setStarting("");
    }
  }

  const rows = payload ? queueRows(payload) : [];
  const shown = filterRows(rows, kind);
  return html`
    <header class="topbar">
      <span class="brand">Palette labeling</span>
      <span class="spacer"></span>
      ${payload && html`<span class="muted small">Signed in as <b class="ink">${payload.user}</b></span>`}
    </header>
    <main class="page">
      <div class="page-head">
        <div>
          <h1>Your queue</h1>
          <p class="muted">${payload ? queueSummary(payload, rows) : problem ? "" : "Loading your queue…"}</p>
        </div>
        <span class="spacer"></span>
        ${rows.length > 0 &&
        html`<div class="segmented" role="group" aria-label="Filter by task type">
          ${queueFilters(rows).map(
            (f) => html`<button type="button" class="btn btn-small" aria-pressed=${f.id === kind} onClick=${() => setKind(f.id)}>
              ${f.label}
            </button>`,
          )}
        </div>`}
      </div>
      <${Problem} problem=${problem} />
      ${payload &&
      payload.blockers.map(
        (b) => html`<div class="notice notice-danger" role="alert">
          <strong>Labeling is paused for you.</strong>
          <p>${b.message || "Ask the operator to clear this before labeling."}</p>
          <p class="mono muted">Reference: ${b.code}</p>
        </div>`,
      )}
      ${payload &&
      !payload.labeler.start_ready &&
      payload.labeler.message &&
      html`<div class="notice notice-pending">${payload.labeler.message}</div>`}
      ${payload &&
      html`<section class="card queue-table" aria-label="Tasks">
        <div class="queue-row queue-head"><span>Recording</span><span>Task</span><span>Progress</span><span>Status</span><span></span></div>
        ${shown.length
          ? shown.map(
              (row) => html`<${TaskRow} key=${row.key} row=${row} busy=${starting === row.taskId} onStart=${start} />`,
            )
          : html`<p class="empty muted">No tasks here.</p>`}
      </section>`}
      ${payload && html`<${OperatorDetails} payload=${payload} />`}
    </main>
  `;
}

render(html`<${QueuePage} />`, document.getElementById("app"));
