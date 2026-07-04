"""Read-only personal HTML renderers for labeling web surfaces."""

from __future__ import annotations

__all__ = ["_dashboard_html", "_datasets_html"]

def _dashboard_html() -> bytes:
    return b"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Palette labeling work</title>
  <style>
    :root {
      --ink: #17201a;
      --muted: #5e6b61;
      --line: #d7ded5;
      --paper: #f7f5ec;
      --card: #fffdf5;
      --accent: #0f6f5c;
      --warn: #a15c00;
      --bad: #9a3027;
      --shadow: rgba(23, 32, 26, 0.12);
    }
    body {
      margin: 0;
      background:
        radial-gradient(circle at 12% 8%, rgba(15, 111, 92, 0.13), transparent 28rem),
        linear-gradient(135deg, #f6f0dc 0%, #edf5ee 58%, #f8f6ee 100%);
      color: var(--ink);
      font-family: Georgia, "Times New Roman", serif;
    }
    main {
      max-width: 1120px;
      margin: 0 auto;
      padding: 44px 20px 72px;
    }
    header {
      display: flex;
      align-items: end;
      justify-content: space-between;
      gap: 18px;
      margin-bottom: 24px;
    }
    h1 {
      font-size: clamp(2.1rem, 5vw, 4.4rem);
      line-height: 0.92;
      margin: 0;
      letter-spacing: -0.055em;
      max-width: 760px;
    }
    .subhead {
      color: var(--muted);
      font-family: "Trebuchet MS", Verdana, sans-serif;
      max-width: 680px;
      margin: 12px 0 0;
    }
    .pill {
      display: inline-flex;
      align-items: center;
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 9px 13px;
      background: rgba(255, 253, 245, 0.72);
      box-shadow: 0 8px 24px var(--shadow);
      font-family: "Trebuchet MS", Verdana, sans-serif;
      white-space: nowrap;
    }
    .summary {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(135px, 1fr));
      gap: 12px;
      margin: 24px 0;
    }
    .safety {
      display: grid;
      grid-template-columns: 1.2fr 1fr;
      gap: 14px;
      margin: 18px 0;
      border: 1px solid var(--line);
      border-left: 6px solid var(--accent);
      border-radius: 22px;
      background: rgba(255, 253, 245, 0.82);
      box-shadow: 0 14px 32px var(--shadow);
      padding: 16px 18px;
      font-family: "Trebuchet MS", Verdana, sans-serif;
    }
    .safety h2 {
      margin: 0 0 8px;
      font-family: Georgia, "Times New Roman", serif;
      font-size: 1.25rem;
    }
    .safety p {
      margin: 0 0 8px;
    }
    .safety ul {
      margin: 0;
      padding-left: 1.1rem;
    }
    .entry-actions {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin: 18px 0;
      align-items: center;
      font-family: "Trebuchet MS", Verdana, sans-serif;
    }
    .entry-actions a {
      display: inline-block;
      border-radius: 999px;
      background: var(--ink);
      color: white;
      padding: 10px 14px;
      text-decoration: none;
      font-weight: 700;
    }
    .stat, .recording {
      background: rgba(255, 253, 245, 0.86);
      border: 1px solid var(--line);
      border-radius: 22px;
      box-shadow: 0 18px 40px var(--shadow);
    }
    .stat {
      padding: 18px;
    }
    .stat b {
      display: block;
      font-size: 2rem;
      letter-spacing: -0.04em;
    }
    .stat span, .task-meta, .empty, button {
      font-family: "Trebuchet MS", Verdana, sans-serif;
    }
    .recording {
      padding: 20px;
      margin: 16px 0;
    }
    .recording h2 {
      margin: 0 0 14px;
      font-size: 1.55rem;
      letter-spacing: -0.025em;
      overflow-wrap: anywhere;
    }
    .task {
      border-top: 1px solid var(--line);
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 16px;
      padding: 15px 0;
      align-items: center;
    }
    .task:first-of-type {
      border-top: 0;
    }
    .task-title {
      font-size: 1.12rem;
      font-weight: 700;
    }
    .task-meta {
      color: var(--muted);
      font-size: 0.92rem;
      margin-top: 5px;
    }
    .assignment-notes {
      margin: 10px 0 4px;
      border-left: 4px solid var(--accent);
      padding: 8px 12px;
      background: rgba(255, 253, 245, 0.72);
      color: var(--ink);
      overflow-wrap: anywhere;
    }
    .task-notes {
      margin-top: 8px;
      border-left: 3px solid var(--line);
      padding: 6px 10px;
      background: rgba(255, 253, 245, 0.62);
      color: var(--ink);
      font-family: "Trebuchet MS", Verdana, sans-serif;
      font-size: 0.92rem;
      overflow-wrap: anywhere;
    }
    .progress-line {
      margin-top: 6px;
      color: var(--muted);
      font-family: "Trebuchet MS", Verdana, sans-serif;
      font-size: 0.92rem;
    }
    .state {
      display: inline-block;
      border-radius: 999px;
      border: 1px solid var(--line);
      padding: 2px 8px;
      margin-right: 6px;
      background: #ffffff99;
    }
    .filters {
      display: grid;
      grid-template-columns: minmax(220px, 1fr) minmax(180px, 260px) auto auto auto auto;
      gap: 12px;
      align-items: end;
      margin: 18px 0;
      border: 1px solid var(--line);
      background: rgba(255, 253, 245, 0.68);
      border-radius: 20px;
      padding: 14px;
      box-shadow: 0 12px 30px var(--shadow);
      font-family: "Trebuchet MS", Verdana, sans-serif;
    }
    input, select {
      box-sizing: border-box;
      width: 100%;
      margin-top: 4px;
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 9px 12px;
      background: #fffdf8;
      color: var(--ink);
      font: inherit;
    }
    .inline-checkbox input {
      width: auto;
      margin-right: 0.45rem;
    }
    .filter-status {
      color: var(--muted);
      font-size: 0.9rem;
      white-space: nowrap;
    }
    button {
      border: 0;
      border-radius: 999px;
      background: var(--accent);
      color: white;
      padding: 10px 14px;
      cursor: pointer;
      font-weight: 700;
    }
    button:disabled {
      opacity: 0.55;
      cursor: wait;
    }
    .empty {
      border: 1px dashed var(--line);
      background: rgba(255, 253, 245, 0.55);
      border-radius: 20px;
      padding: 24px;
      color: var(--muted);
    }
    .failed-promotions {
      margin: 18px 0;
      border: 1px solid rgba(154, 48, 39, 0.32);
      background: rgba(255, 253, 245, 0.82);
      border-radius: 20px;
      padding: 16px;
      box-shadow: 0 12px 30px var(--shadow);
    }
    .failed-promotions h2 {
      margin: 0 0 10px;
      font-size: 1.15rem;
    }
    .failed-row {
      border-top: 1px solid var(--line);
      padding: 9px 0;
      font-family: "Trebuchet MS", Verdana, sans-serif;
      font-size: 0.9rem;
      overflow-wrap: anywhere;
    }
    .failed-row:first-of-type {
      border-top: 0;
    }
    .dataset-queue {
      margin: 18px 0;
      border: 1px solid var(--line);
      background: rgba(255, 253, 245, 0.68);
      border-radius: 20px;
      padding: 16px;
      box-shadow: 0 12px 30px var(--shadow);
    }
    .dataset-queue h2 {
      margin: 0 0 10px;
      font-size: 1.15rem;
    }
    .dataset-row {
      border-top: 1px solid var(--line);
      padding: 10px 0;
      font-family: "Trebuchet MS", Verdana, sans-serif;
      font-size: 0.92rem;
      overflow-wrap: anywhere;
    }
    .dataset-row:first-of-type {
      border-top: 0;
    }
    .dataset-row b {
      font-family: Georgia, "Times New Roman", serif;
      font-size: 1rem;
    }
    .operator-error {
      border: 1px solid #f0b9ad;
      background: #fff8f5;
      border-radius: 18px;
      padding: 16px 18px;
      margin: 18px 0;
      color: var(--ink);
      box-shadow: 0 18px 45px rgba(112, 35, 22, 0.08);
    }
    .operator-error h2 {
      margin: 0 0 8px;
      font-size: 1.1rem;
      color: var(--bad);
    }
    .operator-error p {
      margin: 8px 0;
      color: var(--muted);
    }
    .operator-error pre {
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      border: 1px solid #f0d7cf;
      background: #fffdf8;
      border-radius: 12px;
      padding: 10px 12px;
      margin: 10px 0;
      color: var(--ink);
      font-size: 0.88rem;
    }
    .error {
      color: var(--bad);
    }
    @media (max-width: 720px) {
      header, .task {
        display: block;
      }
      .pill {
        margin-top: 18px;
      }
      .summary {
        grid-template-columns: 1fr;
      }
      .filters {
        grid-template-columns: 1fr;
      }
      .safety {
        grid-template-columns: 1fr;
      }
      button {
        margin-top: 12px;
        width: 100%;
      }
    }
  </style>
</head>
<body>
  <main>
    <header>
      <div>
        <h1>Work waiting for completion</h1>
        <p class="subhead">Your assigned recordings and open review/editing tasks. Higher-priority tasks are shown first within each recording. Browser saves run through Palette's server-side assigned task/training Zarr writers, and CSV, HTML, JSON, and handoff files are metadata only. You do not need a local Palette or Crimson installation.</p>
      </div>
      <div class="pill" id="user-pill">Loading user...</div>
    </header>
    <section class="summary">
      <div class="stat"><b id="recording-count">-</b><span>recordings</span></div>
      <div class="stat"><b id="dataset-count">-</b><span>datasets waiting</span></div>
      <div class="stat"><b id="task-count">-</b><span>startable / total tasks</span></div>
      <div class="stat"><b id="complete-count">-</b><span>complete tasks</span></div>
      <div class="stat"><b id="waiting-recording-count">-</b><span>waiting recordings</span></div>
      <div class="stat"><b id="blocked-recording-count">-</b><span>blocked / no-open recordings</span></div>
      <div class="stat"><b id="session-count">0</b><span>sessions opened here</span></div>
      <div class="stat"><b id="failed-promotion-count">-</b><span>failed promotions</span></div>
    </section>
    <section class="safety">
      <div>
        <h2>Browser-only labeling</h2>
        <p>Open one task at a time from this page or from a signed link. Start with the highest-priority open task unless the operator told you otherwise. Save through the browser controls only.</p>
        <p>Recording instructions appear above each recording; task-specific notes appear under the relevant task.</p>
        <p>This page is personalized to the user shown at the top. If that user is not you, stop before opening work and ask the operator to fix authentication or the assignment.</p>
        <p>Each recording has one active assigned owner. Only that current assignee can open or save browser labeling work for the recording.</p>
        <p>Use <code>include_completed=1</code> when you need the API response to include completed tasks for completion/reopen support.</p>
        <p><b>Supported browser workflows:</b> keypoints, detect_training, detect_analysis, subject_mask_component. Detection analysis tasks may be review-only unless the task scope enables edits.</p>
        <p>If the expected recording is missing, a link expired, or a save fails, stop and ask the operator to regenerate or inspect your handoff.</p>
      </div>
      <ul>
        <li>Do not edit zarr files directly.</li>
        <li>Do not forward links or handoff files.</li>
        <li>Do not keep working after a recording is reassigned or paused.</li>
      </ul>
    </section>
    <section class="entry-actions">
      <a id="landing-link" href="/">Open landing page</a>
      <button type="button" id="copy-landing-link">Copy start link</button>
    </section>
    <section class="filters">
      <label>Search<br><input id="task-search" type="search" placeholder="recording, dataset, component, priority, notes"></label>
      <label>Workflow<br><select id="workflow-filter"><option value="">All workflows</option></select></label>
      <label class="inline-checkbox"><input id="include-completed" type="checkbox">Show completed tasks</label>
      <button type="button" id="clear-filters">Clear filters</button>
      <button type="button" id="refresh-work">Refresh work</button>
      <div class="filter-status" id="filtered-count">No filter</div>
    </section>
    <section id="dashboard-error" aria-live="polite"></section>
    <section id="failed-promotions"></section>
    <section id="dataset-queue" class="dataset-queue" aria-live="polite">
      <h2>Datasets waiting for completion</h2>
      <p class="muted">Loading personalized dataset queue...</p>
    </section>
    <section id="content" class="empty">Loading assignments...</section>
  </main>
  <script>
    let openedSessions = 0;
    let currentWork = null;

    function escapeText(value) {
      return String(value == null ? "" : value)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;");
    }

    async function readApiPayload(response) {
      try {
        return await response.json();
      } catch (error) {
        return {
          ok: false,
          error: "invalid_json_response",
          details: `The server returned a non-JSON response with status ${response.status || "unknown"}.`
        };
      }
    }

    function dashboardFailure(response, payload, fallbackError) {
      const errorCode = String((payload && payload.error) || fallbackError);
      const details = String((payload && (payload.details || payload.message)) || response.statusText || errorCode);
      const failure = new Error(details);
      failure.operatorSupport = {
        error: errorCode,
        status: response.status || "unknown",
        details,
        personalized_launch_readiness: payload && payload.personalized_launch_readiness,
        task_open_authorization_contract: payload && payload.task_open_authorization_contract,
        authorization_context: payload && payload.authorization_context
      };
      return failure;
    }

    function normalizedFailure(error, fallbackError) {
      if (error && error.operatorSupport) return error;
      const message = String((error && error.message) || error || fallbackError);
      const failure = new Error(message);
      failure.operatorSupport = {
        error: fallbackError,
        status: "client",
        details: message
      };
      return failure;
    }

    function operatorSupportText(error, fallbackError) {
      const failure = normalizedFailure(error, fallbackError);
      const support = failure.operatorSupport || {};
      const contract = support.task_open_authorization_contract || {};
      const context = support.authorization_context || {};
      const readiness = support.personalized_launch_readiness || {};
      return [
        `error=${support.error || fallbackError}`,
        `status=${support.status || "client"}`,
        `details=${support.details || failure.message || ""}`,
        `personalized_launch_readiness=${JSON.stringify(readiness || {})}`,
        `personalized_launch_readiness_schema=${readiness.schema || ""}`,
        `personalized_launch_readiness_personalized_labeler_entry_url=${readiness.personalized_labeler_entry_url || ""}`,
        `personalized_launch_readiness_browser_label_write_target=${readiness.browser_label_write_target || ""}`,
        `personalized_launch_readiness_browser_writes_csv_or_handoff_files=${readiness.browser_writes_csv_or_handoff_files ?? ""}`,
        `personalized_launch_readiness_browser_has_direct_zarr_write_authority=${readiness.browser_has_direct_zarr_write_authority ?? ""}`,
        `authorization_resolved_user=${context.resolved_user || ""}`,
        `authorization_expected_user=${context.expected_user || ""}`,
        `authorization_return_expected_user=${context.return_expected_user || ""}`,
        `authorization_return_personal_dataset_queue_url=${context.return_personal_dataset_queue_url || ""}`,
        `authorization_return_personal_dataset_queue_expected_user_guarded=${context.return_personal_dataset_queue_expected_user_guarded ?? ""}`,
        `authorization_return_personal_work_url=${context.return_personal_work_url || ""}`,
        `authorization_return_personal_work_expected_user_guarded=${context.return_personal_work_expected_user_guarded ?? ""}`,
        `authorization_task_id=${context.task_id || ""}`,
        `authorization_recording_id=${context.recording_id || ""}`,
        `task_open_authorization_contract_schema=${contract.schema || ""}`,
        `task_open_authorization_contract_ready=${contract.ready ?? ""}`,
        `task_open_authorization_contract_not_ready_reason=${contract.not_ready_reason || ""}`,
        `task_open_expected_user_guard_checked_server_side=${contract.expected_user_guard_checked_server_side ?? ""}`,
        `task_open_expected_user_guard_present=${contract.expected_user_guard_present ?? ""}`,
        `task_open_expected_user_matches_resolved_user=${contract.expected_user_matches_resolved_user ?? ""}`,
        `task_open_active_assignment_present=${contract.active_assignment_present ?? ""}`,
        `task_open_task_assigned_to_resolved_user=${contract.task_assigned_to_resolved_user ?? ""}`,
        `task_open_assignment_status_active=${contract.assignment_status_active ?? ""}`,
        `task_open_task_state_startable=${contract.task_state_startable ?? ""}`,
        `task_open_reassignment_session_safety_checked_server_side=${contract.reassignment_session_safety_checked_server_side ?? ""}`,
        `task_open_reassignment_session_safety_passed=${contract.reassignment_session_safety_passed ?? ""}`,
        `task_open_session_created_server_side=${contract.session_created_server_side ?? ""}`,
        `task_open_server_authorizes_open=${contract.server_authorizes_open ?? ""}`,
        `task_open_operator_validation_start_gate_required=${contract.operator_validation_start_gate_required ?? ""}`,
        `task_open_operator_validation_start_gate_ready=${contract.operator_validation_start_gate_ready ?? ""}`,
        `task_open_operator_validation_start_gate_blocks_task_open=${contract.operator_validation_start_gate_blocks_task_open ?? ""}`,
        `task_open_operator_validation_start_gate_not_ready_reason=${contract.operator_validation_start_gate_not_ready_reason || ""}`,
        `task_open_operator_validation_status=${contract.operator_validation_status || ""}`,
        `task_open_operator_validation_pending_gate_ids=${JSON.stringify(contract.operator_validation_pending_gate_ids || [])}`,
        `task_open_operator_validation_required_missing_evidence_gate_ids=${JSON.stringify(contract.operator_validation_required_missing_evidence_gate_ids || [])}`,
        `task_open_browser_label_write_target=${contract.browser_label_write_target || ""}`,
        `task_open_browser_writes_csv_or_handoff_files=${contract.browser_writes_csv_or_handoff_files ?? ""}`,
        `task_open_browser_has_direct_zarr_write_authority=${contract.browser_has_direct_zarr_write_authority ?? ""}`
      ].join("\n");
    }

    function clearDashboardError() {
      const target = document.getElementById("dashboard-error");
      target.className = "";
      target.innerHTML = "";
    }

    function copySupportDetails(button) {
      const block = button.closest(".operator-error");
      const text = block && block.querySelector("pre") ? block.querySelector("pre").textContent : "";
      const markCopied = () => {
        button.textContent = "Copied";
        window.setTimeout(() => { button.textContent = "Copy support details"; }, 1800);
      };
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text).then(markCopied).catch(() => {
          const textarea = document.createElement("textarea");
          textarea.value = text;
          document.body.appendChild(textarea);
          textarea.select();
          document.execCommand("copy");
          textarea.remove();
          markCopied();
        });
        return;
      }
      const textarea = document.createElement("textarea");
      textarea.value = text;
      document.body.appendChild(textarea);
      textarea.select();
      document.execCommand("copy");
      textarea.remove();
      markCopied();
    }

    function showDashboardError(error, title, fallbackError) {
      const failure = normalizedFailure(error, fallbackError);
      const supportText = operatorSupportText(failure, fallbackError);
      const target = document.getElementById("dashboard-error");
      target.className = "operator-error";
      target.innerHTML = `
        <h2>${escapeText(title)}</h2>
        <p>Stop and send these support details to the operator. Do not edit zarr files directly or keep working from an old tab.</p>
        <details>
          <summary>What to send the operator</summary>
          <pre>${escapeText(supportText)}</pre>
          <button type="button" onclick="copySupportDetails(this)">Copy support details</button>
        </details>
      `;
    }

    function taskTitle(task) {
      if (task.title) return task.title;
      const parts = [task.workflow_kind || "task"];
      if (task.component_name) parts.push(task.component_name);
      if (task.run_name) parts.push(task.run_name);
      return parts.join(" / ");
    }

    function taskPriority(task) {
      const value = Number(task.priority == null || task.priority === "" ? 0 : task.priority);
      return Number.isFinite(value) ? value : 0;
    }

    const initialWorkQuery = new URLSearchParams(window.location.search);
    const expectedUserGuardParam = initialWorkQuery.get("expected_user") || "";
    const inviteTokenParam = initialWorkQuery.get("invite") || "";
    let pendingInitialWorkflowFilter = initialWorkQuery.get("workflow") || "";
    let activeLinkFilters = {
      dataset_id: initialWorkQuery.get("dataset_id") || "",
      recording_id: initialWorkQuery.get("recording_id") || "",
      task_id: initialWorkQuery.get("task_id") || ""
    };

    function authQueryParams() {
      const params = new URLSearchParams();
      if (expectedUserGuardParam) params.set("expected_user", expectedUserGuardParam);
      if (inviteTokenParam) params.set("invite", inviteTokenParam);
      return params;
    }

    function withAuthQuery(path) {
      const url = new URL(path, window.location.href);
      for (const [key, value] of authQueryParams().entries()) {
        url.searchParams.set(key, value);
      }
      return url.pathname + url.search;
    }

    function guardedWorkPath(path) {
      return withAuthQuery(path);
    }

    function setDashboardEntryLinks() {
      const link = document.getElementById("landing-link");
      if (link) link.href = guardedWorkPath("/");
    }

    async function copyDashboardText(button, text, resetText) {
      if (!text) return;
      const markCopied = () => {
        button.textContent = "Copied";
        window.setTimeout(() => { button.textContent = resetText; }, 1800);
      };
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text).then(markCopied).catch(() => {
          const textarea = document.createElement("textarea");
          textarea.value = text;
          document.body.appendChild(textarea);
          textarea.select();
          document.execCommand("copy");
          textarea.remove();
          markCopied();
        });
        return;
      }
      const textarea = document.createElement("textarea");
      textarea.value = text;
      document.body.appendChild(textarea);
      textarea.select();
      document.execCommand("copy");
      textarea.remove();
      markCopied();
    }

    async function copyDashboardLandingLink(button) {
      const link = document.getElementById("landing-link");
      const href = link ? new URL(link.getAttribute("href") || "/", window.location.href).href : window.location.href;
      await copyDashboardText(button, href, "Copy start link");
    }

    function taskSearchText(recording, task) {
      return [
        recording.recording_id,
        recording.assignment_notes,
        taskTitle(task),
        task.task_id,
        task.workflow_kind,
        task.dataset_id,
        task.zarr_use,
        task.stage_group,
        task.run_name,
        task.component_name,
        task.state,
        task.priority,
        task.notes
      ].filter(Boolean).join(" ").toLowerCase();
    }

    function recordingSearchText(recording) {
      return [
        recording.recording_id,
        recording.assignment_notes
      ].filter(Boolean).join(" ").toLowerCase();
    }

    function updateWorkflowOptions(payload) {
      const select = document.getElementById("workflow-filter");
      const current = select.value || pendingInitialWorkflowFilter;
      const workflows = new Set();
      for (const recording of payload.recordings || []) {
        for (const task of recording.tasks || []) {
          if (task.workflow_kind) workflows.add(task.workflow_kind);
        }
      }
      select.innerHTML = `<option value="">All workflows</option>` +
        Array.from(workflows).sort().map((workflow) =>
          `<option value="${escapeText(workflow)}">${escapeText(workflow)}</option>`
        ).join("");
      if (current && workflows.has(current)) {
        select.value = current;
      }
      pendingInitialWorkflowFilter = "";
    }

    function filteredRecordings(payload) {
      const workflow = document.getElementById("workflow-filter").value;
      const search = document.getElementById("task-search").value.trim().toLowerCase();
      return (payload.recordings || []).map((recording) => {
        if (activeLinkFilters.recording_id && String(recording.recording_id || "") !== activeLinkFilters.recording_id) {
          return {...recording, tasks: [], keep_empty_assigned_recording: false};
        }
        const sourceTasks = recording.tasks || [];
        const tasks = sourceTasks.filter((task) => {
          if (activeLinkFilters.dataset_id && String(task.dataset_id || "") !== activeLinkFilters.dataset_id) return false;
          if (activeLinkFilters.task_id && String(task.task_id || "") !== activeLinkFilters.task_id) return false;
          if (workflow && task.workflow_kind !== workflow) return false;
          if (search && !taskSearchText(recording, task).includes(search)) return false;
          return true;
        });
        tasks.sort((a, b) => {
          const priorityDelta = taskPriority(b) - taskPriority(a);
          if (priorityDelta) return priorityDelta;
          const titleDelta = taskTitle(a).localeCompare(taskTitle(b));
          if (titleDelta) return titleDelta;
          return String(a.task_id || "").localeCompare(String(b.task_id || ""));
        });
        const keepEmptyAssignedRecording =
          sourceTasks.length === 0 &&
          !activeLinkFilters.dataset_id &&
          !activeLinkFilters.recording_id &&
          !activeLinkFilters.task_id &&
          !workflow &&
          (!search || recordingSearchText(recording).includes(search));
        return {...recording, tasks, keep_empty_assigned_recording: keepEmptyAssignedRecording};
      }).filter((recording) => recording.tasks.length > 0 || recording.keep_empty_assigned_recording);
    }

    function clearStructuredFilters() {
      activeLinkFilters = {dataset_id: "", recording_id: "", task_id: ""};
      pendingInitialWorkflowFilter = "";
      if (window.history && window.history.replaceState) {
        const params = authQueryParams();
        const query = params.toString();
        window.history.replaceState({}, "", `${window.location.pathname}${query ? "?" + query : ""}`);
      }
    }

    async function openTask(taskId, button) {
      button.disabled = true;
      clearDashboardError();
      try {
        const response = await fetch(withAuthQuery(`/api/tasks/${encodeURIComponent(taskId)}/open`), {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({client_label: navigator.userAgent, expected_user: expectedUserGuardParam || ""})
        });
        const payload = await readApiPayload(response);
        if (!response.ok || !payload.ok) {
          throw dashboardFailure(response, payload, "task_open_failed");
        }
        openedSessions += 1;
        document.getElementById("session-count").textContent = String(openedSessions);
        window.location.href = payload.session.url;
      } catch (error) {
        showDashboardError(error, "Palette could not open that task.", "task_open_failed");
      } finally {
        button.disabled = false;
      }
    }

    const OPERATOR_VALIDATION_GATE_IDS = [
      "mutable_zarr_backup_confirmation",
      "browser_response_security_headers",
      "identity_probe_verification",
      "browser_smoke",
      "disposable_zarr_mutation_smoke",
      "operator_recovery_contract",
    ];
    const OPERATOR_VALIDATION_GATE_FIELD_SUFFIXES = [
      "status",
      "pending",
      "missing_evidence",
      "needs_review",
      "passed",
    ];

    function operatorValidationGateSupportLines(source) {
      const item = source || {};
      return OPERATOR_VALIDATION_GATE_IDS.flatMap((gateId) => {
        const prefix = `operator_validation_gate_${gateId}`;
        return OPERATOR_VALIDATION_GATE_FIELD_SUFFIXES.map((suffix) => (
          `${prefix}_${suffix}=${item[`${prefix}_${suffix}`] ?? ""}`
        ));
      });
    }

    function safeShareExternalLaunchEvidenceGapSupportLines(source) {
      const item = source || {};
      return [
        `safe_share_external_launch_evidence_gap_gate_ids=${JSON.stringify(item.safe_share_external_launch_evidence_gap_gate_ids || [])}`,
        `safe_share_external_launch_evidence_gap_count=${item.safe_share_external_launch_evidence_gap_count ?? ""}`,
        `safe_share_external_launch_evidence_gap_statuses=${JSON.stringify(item.safe_share_external_launch_evidence_gap_statuses || {})}`,
        `safe_share_external_launch_evidence_gap_action_required=${item.safe_share_external_launch_evidence_gap_action_required ?? ""}`,
        `safe_share_external_launch_evidence_gap_summary=${item.safe_share_external_launch_evidence_gap_summary || ""}`,
        `safe_share_external_launch_evidence_gap_todos=${JSON.stringify(item.safe_share_external_launch_evidence_gap_todos || [])}`,
        `safe_share_external_launch_evidence_gap_todo_count=${item.safe_share_external_launch_evidence_gap_todo_count ?? ""}`,
        `safe_share_external_launch_evidence_gap_todo_fields=${JSON.stringify(item.safe_share_external_launch_evidence_gap_todo_fields || [])}`,
        `safe_share_external_launch_evidence_gap_template_paths_by_gate_id=${JSON.stringify(item.safe_share_external_launch_evidence_gap_template_paths_by_gate_id || {})}`,
        `safe_share_external_launch_evidence_gap_record_command_ids_by_gate_id=${JSON.stringify(item.safe_share_external_launch_evidence_gap_record_command_ids_by_gate_id || {})}`,
      ];
    }

    function personalizedLaunchReadinessSupportLines(source) {
      const readiness = (source && source.personalized_launch_readiness) || {};
      return [
        `personalized_launch_readiness=${JSON.stringify(readiness || {})}`,
        `personalized_launch_readiness_schema=${readiness.schema || ""}`,
        `personalized_launch_readiness_field_count=${readiness.field_count ?? ""}`,
        `personalized_launch_readiness_personalized_labeler_entry_url=${readiness.personalized_labeler_entry_url || ""}`,
        `personalized_launch_readiness_labeler_start_ready=${readiness.labeler_start_ready ?? ""}`,
        `personalized_launch_readiness_labeler_work_completion_status=${readiness.labeler_work_completion_status || ""}`,
        `personalized_launch_readiness_external_launch_evidence_gap_count=${readiness.external_launch_evidence_gap_count ?? ""}`,
        `personalized_launch_readiness_external_launch_evidence_gap_gate_ids=${JSON.stringify(readiness.external_launch_evidence_gap_gate_ids || [])}`,
        `personalized_launch_readiness_external_launch_evidence_gap_todo_count=${readiness.external_launch_evidence_gap_todo_count ?? ""}`,
        `personalized_launch_readiness_external_launch_evidence_gap_todos=${JSON.stringify(readiness.external_launch_evidence_gap_todos || [])}`,
        `personalized_launch_readiness_browser_label_write_target=${readiness.browser_label_write_target || ""}`,
        `personalized_launch_readiness_browser_writes_csv_or_handoff_files=${readiness.browser_writes_csv_or_handoff_files ?? ""}`,
        `personalized_launch_readiness_browser_has_direct_zarr_write_authority=${readiness.browser_has_direct_zarr_write_authority ?? ""}`,
      ];
    }

    function noOpenTaskMessage(recording) {
      if (recording.no_open_task_message) {
        return String(recording.no_open_task_message);
      }
      const total = Number(recording.total_task_count || 0);
      const complete = Number(recording.complete_task_count || 0);
      if (total > 0 && complete >= total) {
        return "All tasks for this recording are complete. Ask the operator before reopening or continuing work.";
      }
      if (total > 0) {
        return "This recording is assigned to you, but no startable tasks match the current view. Clear filters or ask the operator to inspect the batch.";
      }
      return "This recording is assigned to you, but no browser-labeling tasks have been generated yet. If you expected work here, ask the operator to generate tasks or inspect the batch.";
    }

    function dashboardQueueSupportText(row) {
      const item = row || {};
      const operatorSupport = item.operator_support || {};
      const operatorSupportValue = (name, fallback = "") => (
        operatorSupport[name] !== undefined ? operatorSupport[name] : (item[name] !== undefined ? item[name] : fallback)
      );
      const directStartPolicy = currentWork && currentWork.dataset_queue_direct_start_policy
        ? currentWork.dataset_queue_direct_start_policy
        : {};
      const mutationWriteChecklist = currentWork && currentWork.browser_mutation_write_checklist
        ? currentWork.browser_mutation_write_checklist
        : {};
      const routeAuthorizationChecklist = currentWork && currentWork.labeler_route_authorization_checklist
        ? currentWork.labeler_route_authorization_checklist
        : {};
      const runtimeGateCliPolicy = currentWork && currentWork.runtime_operator_validation_gate_cli_policy
        ? currentWork.runtime_operator_validation_gate_cli_policy
        : {};
      const operatorValidationStartGate = currentWork && currentWork.operator_validation_start_gate
        ? currentWork.operator_validation_start_gate
        : {};
      const operatorValidationMutationGate = currentWork && currentWork.operator_validation_mutation_gate
        ? currentWork.operator_validation_mutation_gate
        : {};
      const operatorValidationVisibilityPolicy = currentWork && currentWork.operator_validation_visibility_policy
        ? currentWork.operator_validation_visibility_policy
        : {};
      const operatorValidationCommandTemplates = currentWork && currentWork.operator_validation_command_templates
        ? currentWork.operator_validation_command_templates
        : {};
      const singleOwnerPolicy = currentWork && currentWork.single_owner_policy
        ? currentWork.single_owner_policy
        : {};
      return [
        "page_context=work_dashboard_dataset_queue",
        `user=${currentWork && currentWork.user ? currentWork.user : ""}`,
        `expected_user=${expectedUserGuardParam || (currentWork && currentWork.expected_user) || ""}`,
        `single_owner_policy_assignment_scope=${currentWork ? currentWork.single_owner_policy_assignment_scope ?? singleOwnerPolicy.assignment_scope ?? "" : ""}`,
        `single_owner_policy_recording_assignment_key=${currentWork ? currentWork.single_owner_policy_recording_assignment_key ?? singleOwnerPolicy.recording_assignment_key ?? "" : ""}`,
        `single_owner_policy_one_current_assignment_row_per_recording=${currentWork ? currentWork.single_owner_policy_one_current_assignment_row_per_recording ?? singleOwnerPolicy.one_current_assignment_row_per_recording ?? "" : ""}`,
        `single_owner_policy_one_active_owner=${currentWork ? currentWork.single_owner_policy_one_active_owner ?? singleOwnerPolicy.one_active_owner ?? "" : ""}`,
        `single_owner_policy_multiple_labelers_per_recording_allowed=${currentWork ? currentWork.single_owner_policy_multiple_labelers_per_recording_allowed ?? singleOwnerPolicy.multiple_labelers_per_recording_allowed ?? "" : ""}`,
        `single_owner_policy_assignment_user_match_required_for_mutation=${currentWork ? currentWork.single_owner_policy_assignment_user_match_required_for_mutation ?? singleOwnerPolicy.assignment_user_match_required_for_mutation ?? "" : ""}`,
        `single_owner_policy_browser_mutation_requires_current_assignment_owner=${currentWork ? currentWork.single_owner_policy_browser_mutation_requires_current_assignment_owner ?? singleOwnerPolicy.browser_mutation_requires_current_assignment_owner ?? "" : ""}`,
        `single_owner_policy_browser_mutation_target_resolved_server_side=${currentWork ? currentWork.single_owner_policy_browser_mutation_target_resolved_server_side ?? singleOwnerPolicy.browser_mutation_target_resolved_server_side ?? "" : ""}`,
        `single_owner_policy_browser_mutation_target_source=${currentWork ? currentWork.single_owner_policy_browser_mutation_target_source ?? singleOwnerPolicy.browser_mutation_target_source ?? "" : ""}`,
        `single_owner_policy_labelers_mutate_assigned_training_zarrs=${currentWork ? currentWork.single_owner_policy_labelers_mutate_assigned_training_zarrs ?? singleOwnerPolicy.labelers_mutate_assigned_training_zarrs ?? "" : ""}`,
        `single_owner_policy_labelers_mutate_intermediate_csvs=${currentWork ? currentWork.single_owner_policy_labelers_mutate_intermediate_csvs ?? singleOwnerPolicy.labelers_mutate_intermediate_csvs ?? "" : ""}`,
        `assignment_ownership_contract_store_single_owner_assignment_contract_present=${currentWork ? currentWork.assignment_ownership_contract_store_single_owner_assignment_contract_present ?? "" : ""}`,
        `assignment_ownership_contract_store_single_owner_assignment_contract_ready=${currentWork ? currentWork.assignment_ownership_contract_store_single_owner_assignment_contract_ready ?? "" : ""}`,
        `assignment_ownership_contract_store_single_owner_assignment_contract_met=${currentWork ? currentWork.assignment_ownership_contract_store_single_owner_assignment_contract_met ?? "" : ""}`,
        `assignment_ownership_contract_store_single_owner_assignment_contract_schema=${currentWork ? currentWork.assignment_ownership_contract_store_single_owner_assignment_contract_schema ?? "" : ""}`,
        `assignment_ownership_contract_browser_mutation_target_resolved_server_side=${currentWork ? currentWork.assignment_ownership_contract_browser_mutation_target_resolved_server_side ?? "" : ""}`,
        `assignment_ownership_contract_labelers_mutate_assigned_training_zarrs=${currentWork ? currentWork.assignment_ownership_contract_labelers_mutate_assigned_training_zarrs ?? "" : ""}`,
        `assignment_ownership_contract_labelers_mutate_intermediate_csvs=${currentWork ? currentWork.assignment_ownership_contract_labelers_mutate_intermediate_csvs ?? "" : ""}`,
        `handoff_status=${currentWork && currentWork.status ? currentWork.status : ""}`,
        `handoff_ready_to_send=${currentWork && currentWork.ready_to_send !== undefined ? currentWork.ready_to_send : ""}`,
        `handoff_sendability_reasons=${JSON.stringify(currentWork && currentWork.sendability_reasons ? currentWork.sendability_reasons : [])}`,
        `handoff_sendability_actions=${JSON.stringify(currentWork && currentWork.sendability_actions ? currentWork.sendability_actions : [])}`,
        `runtime_operator_validation_gate_cli_policy_preferred_require_flag=${runtimeGateCliPolicy.preferred_require_flag || ""}`,
        `runtime_operator_validation_gate_cli_policy_legacy_require_flag=${runtimeGateCliPolicy.legacy_require_flag || ""}`,
        `runtime_operator_validation_gate_cli_policy_validation_checklist_flag=${runtimeGateCliPolicy.validation_checklist_flag || ""}`,
        `runtime_operator_validation_gate_cli_policy_protects_browser_start_open=${runtimeGateCliPolicy.protects_browser_start_open ?? ""}`,
        `runtime_operator_validation_gate_cli_policy_protects_browser_mutations=${runtimeGateCliPolicy.protects_browser_mutations ?? ""}`,
        `runtime_operator_validation_gate_cli_policy_blocks_before_target_token_check=${runtimeGateCliPolicy.blocks_before_target_token_check ?? ""}`,
        `runtime_operator_validation_gate_cli_policy_blocks_before_zarr_write=${runtimeGateCliPolicy.blocks_before_zarr_write ?? ""}`,
        `runtime_operator_validation_gate_cli_policy_blocks_before_audit_event_creation=${runtimeGateCliPolicy.blocks_before_audit_event_creation ?? ""}`,
        `operator_validation_start_gate_required=${operatorValidationStartGate.required_for_browser_start ?? ""}`,
        `operator_validation_start_gate_ready=${operatorValidationStartGate.ready ?? ""}`,
        `operator_validation_start_gate_blocks_task_open=${operatorValidationStartGate.blocks_task_open ?? ""}`,
        `operator_validation_start_gate_not_ready_reason=${operatorValidationStartGate.not_ready_reason || ""}`,
        `operator_validation_mutation_gate_required=${operatorValidationMutationGate.required_for_browser_mutation ?? ""}`,
        `operator_validation_mutation_gate_ready=${operatorValidationMutationGate.ready ?? ""}`,
        `operator_validation_mutation_gate_blocks_browser_mutation=${operatorValidationMutationGate.blocks_browser_mutation ?? ""}`,
        `operator_validation_mutation_gate_not_ready_reason=${operatorValidationMutationGate.not_ready_reason || ""}`,
        `operator_validation_mutation_gate_pending_gate_ids=${JSON.stringify(operatorValidationMutationGate.operator_validation_pending_gate_ids || [])}`,
        `operator_validation_mutation_gate_required_missing_evidence_gate_ids=${JSON.stringify(operatorValidationMutationGate.operator_validation_required_missing_evidence_gate_ids || [])}`,
        `operator_validation_start_gate_operator_validation_status=${operatorValidationStartGate.operator_validation_status || ""}`,
        `operator_validation_start_gate_pending_gate_ids=${JSON.stringify(operatorValidationStartGate.operator_validation_pending_gate_ids || [])}`,
        `operator_validation_start_gate_required_missing_evidence_gate_ids=${JSON.stringify(operatorValidationStartGate.operator_validation_required_missing_evidence_gate_ids || [])}`,
        `operator_validation_status=${currentWork && currentWork.operator_validation_status ? currentWork.operator_validation_status : ""}`,
        `operator_validation_source=${currentWork && currentWork.operator_validation_source ? currentWork.operator_validation_source : ""}`,
        `operator_validation_gate_count=${currentWork && currentWork.operator_validation_gate_count !== undefined ? currentWork.operator_validation_gate_count : ""}`,
        `operator_validation_pending_gate_ids=${JSON.stringify(currentWork && currentWork.operator_validation_pending_gate_ids ? currentWork.operator_validation_pending_gate_ids : [])}`,
        `operator_validation_required_missing_evidence_gate_ids=${JSON.stringify(currentWork && currentWork.operator_validation_required_missing_evidence_gate_ids ? currentWork.operator_validation_required_missing_evidence_gate_ids : [])}`,
        `operator_validation_gate_status_values=${JSON.stringify(operatorValidationVisibilityPolicy.operator_validation_gate_status_values || [])}`,
        `operator_validation_gate_ids=${JSON.stringify(OPERATOR_VALIDATION_GATE_IDS)}`,
        `operator_validation_gate_flat_field_suffixes=${JSON.stringify(OPERATOR_VALIDATION_GATE_FIELD_SUFFIXES)}`,
        ...operatorValidationGateSupportLines(currentWork || {}),
        ...safeShareExternalLaunchEvidenceGapSupportLines(currentWork || {}),
        `operator_validation_public_fields=${JSON.stringify(operatorValidationVisibilityPolicy.public_fields || [])}`,
        `operator_validation_operator_only_fields=${JSON.stringify(operatorValidationVisibilityPolicy.operator_only_fields || [])}`,
        `operator_validation_operator_action_fields=${JSON.stringify(operatorValidationVisibilityPolicy.operator_action_fields || [])}`,
        `operator_validation_operator_action_fields_are_labeler_instructions=${operatorValidationVisibilityPolicy.operator_action_fields_are_labeler_instructions ?? ""}`,
        `operator_validation_labeler_visible_payloads_may_include_operator_action_fields_for_support=${operatorValidationVisibilityPolicy.labeler_visible_payloads_may_include_operator_action_fields_for_support ?? ""}`,
        `operator_validation_labeler_visible_payloads_include_operator_only_fields=${operatorValidationVisibilityPolicy.labeler_visible_payloads_include_operator_only_fields ?? ""}`,
        `operator_validation_per_user_payloads_use_public_fields_only=${operatorValidationVisibilityPolicy.per_user_payloads_use_public_fields_only ?? ""}`,
        `operator_validation_top_level_operator_reports_may_include_operator_only_fields=${operatorValidationVisibilityPolicy.top_level_operator_reports_may_include_operator_only_fields ?? ""}`,
        `operator_validation_required_pending_gate_count=${currentWork && currentWork.operator_validation_required_pending_gate_count !== undefined ? currentWork.operator_validation_required_pending_gate_count : ""}`,
        `operator_validation_needs_review_gate_count=${currentWork && currentWork.operator_validation_needs_review_gate_count !== undefined ? currentWork.operator_validation_needs_review_gate_count : ""}`,
        `operator_validation_required_missing_evidence_gate_count=${currentWork && currentWork.operator_validation_required_missing_evidence_gate_count !== undefined ? currentWork.operator_validation_required_missing_evidence_gate_count : ""}`,
        `operator_validation_operator_action=${currentWork && currentWork.operator_validation_operator_action ? currentWork.operator_validation_operator_action : ""}`,
        `operator_validation_external_evidence_required=${currentWork && currentWork.operator_validation_external_evidence_required !== undefined ? currentWork.operator_validation_external_evidence_required : ""}`,
        `operator_validation_external_evidence_required_gate_ids=${JSON.stringify(currentWork && currentWork.operator_validation_external_evidence_required_gate_ids ? currentWork.operator_validation_external_evidence_required_gate_ids : [])}`,
        `operator_validation_external_evidence_required_gate_count=${currentWork && currentWork.operator_validation_external_evidence_required_gate_count !== undefined ? currentWork.operator_validation_external_evidence_required_gate_count : ""}`,
        `operator_validation_external_evidence_template_fields_by_gate_id=${JSON.stringify(currentWork && currentWork.operator_validation_external_evidence_template_fields_by_gate_id ? currentWork.operator_validation_external_evidence_template_fields_by_gate_id : {})}`,
        `operator_validation_external_evidence_template_paths_by_gate_id=${JSON.stringify(currentWork && currentWork.operator_validation_external_evidence_template_paths_by_gate_id ? currentWork.operator_validation_external_evidence_template_paths_by_gate_id : {})}`,
        `operator_validation_checklist_only_required_gate_ids=${JSON.stringify(currentWork && currentWork.operator_validation_checklist_only_required_gate_ids ? currentWork.operator_validation_checklist_only_required_gate_ids : [])}`,
        `operator_validation_checklist_only_required_gate_count=${currentWork && currentWork.operator_validation_checklist_only_required_gate_count !== undefined ? currentWork.operator_validation_checklist_only_required_gate_count : ""}`,
        `operator_validation_command_template_schema=${operatorValidationCommandTemplates.schema || ""}`,
        `operator_validation_command_template_command_count=${operatorValidationCommandTemplates.command_count ?? ""}`,
        `operator_validation_command_template_command_ids=${JSON.stringify(operatorValidationCommandTemplates.command_ids || [])}`,
        `operator_validation_command_template_template_backed_gate_ids=${JSON.stringify(operatorValidationCommandTemplates.template_backed_gate_ids || [])}`,
        `operator_validation_command_template_validation_checklist_gate_ids=${JSON.stringify(operatorValidationCommandTemplates.validation_checklist_gate_ids || [])}`,
        `operator_validation_command_template_apply_required_gate_ids=${JSON.stringify(operatorValidationCommandTemplates.apply_required_gate_ids || [])}`,
        `operator_validation_command_template_evidence_template_fields_by_gate_id=${JSON.stringify(operatorValidationCommandTemplates.evidence_template_fields_by_gate_id || {})}`,
        `operator_validation_command_template_evidence_template_paths_by_gate_id=${JSON.stringify(operatorValidationCommandTemplates.evidence_template_paths_by_gate_id || {})}`,
        `operator_validation_command_template_missing_command_gate_ids=${JSON.stringify(operatorValidationCommandTemplates.missing_command_gate_ids || [])}`,
        `operator_validation_command_template_launch_evidence_collection_plan_schema=${operatorValidationCommandTemplates.launch_evidence_collection_plan_schema || ""}`,
        `operator_validation_command_template_launch_evidence_collection_step_count=${operatorValidationCommandTemplates.launch_evidence_collection_step_count ?? ""}`,
        `operator_validation_command_template_launch_evidence_collection_gate_ids=${JSON.stringify(operatorValidationCommandTemplates.launch_evidence_collection_gate_ids || [])}`,
        `operator_validation_command_template_launch_evidence_collection_record_command_ids=${JSON.stringify(operatorValidationCommandTemplates.launch_evidence_collection_record_command_ids || [])}`,
        `operator_validation_command_template_launch_evidence_collection_operator_only=${operatorValidationCommandTemplates.launch_evidence_collection_operator_only ?? ""}`,
        `operator_validation_command_template_launch_evidence_collection_required_final_field=${operatorValidationCommandTemplates.launch_evidence_collection_required_final_field || ""}`,
        `operator_validation_command_template_launch_evidence_collection_required_final_value=${operatorValidationCommandTemplates.launch_evidence_collection_required_final_value ?? ""}`,
        `operator_validation_command_template_launch_evidence_collection_final_inspection_command=${operatorValidationCommandTemplates.launch_evidence_collection_final_inspection_command || ""}`,
        `operator_validation_command_template_commands_are_operator_only=${operatorValidationCommandTemplates.commands_are_operator_only ?? ""}`,
        `operator_validation_command_template_commands_are_labeler_instructions=${operatorValidationCommandTemplates.commands_are_labeler_instructions ?? ""}`,
        `operator_validation_command_template_labelers_must_not_run_commands=${operatorValidationCommandTemplates.labelers_must_not_run_commands ?? ""}`,
        `links_expire_at_utc=${currentWork && currentWork.links_expire_at_utc ? currentWork.links_expire_at_utc : ""}`,
        `seconds_until_expiration=${currentWork && currentWork.seconds_until_expiration !== undefined ? currentWork.seconds_until_expiration : ""}`,
        `dataset_id=${item.dataset_id || ""}`,
        `dataset_label=${item.dataset_label || ""}`,
        `recording_id=${item.recording_id || ""}`,
        `task_id=${item.task_id || ""}`,
        `workflow_kind=${item.workflow_kind || ""}`,
        `state=${item.state || ""}`,
        `operator_support_expected_user_personal_dataset_queue_url=${operatorSupportValue("expected_user_personal_dataset_queue_url")}`,
        `operator_support_personalized_labeler_entry_url=${operatorSupportValue("personalized_labeler_entry_url")}`,
        `operator_support_personal_dataset_queue_link_role=${operatorSupportValue("personal_dataset_queue_link_role")}`,
        `operator_support_canonical_dataset_queue_link_role=${operatorSupportValue("canonical_dataset_queue_link_role")}`,
        `operator_support_browser_label_write_target=${operatorSupportValue("browser_label_write_target")}`,
        `operator_support_browser_writes_csv_or_handoff_files=${operatorSupportValue("browser_writes_csv_or_handoff_files")}`,
        `operator_support_browser_has_direct_zarr_write_authority=${operatorSupportValue("browser_has_direct_zarr_write_authority")}`,
        `operator_support_csv_handoff_artifact_role=${operatorSupportValue("csv_handoff_artifact_role")}`,
        `labeler_start_ready=${item.labeler_start_ready}`,
        `labeler_action=${item.labeler_action || ""}`,
        `labeler_work_completion_status=${currentWork && currentWork.labeler_work_completion ? currentWork.labeler_work_completion.status || "" : ""}`,
        `labeler_work_completion_completed=${currentWork && currentWork.labeler_work_completion ? currentWork.labeler_work_completion.completed ?? "" : ""}`,
        `labeler_work_completion_has_waiting_work=${currentWork && currentWork.labeler_work_completion ? currentWork.labeler_work_completion.has_waiting_work ?? "" : ""}`,
        `labeler_work_completion_ready_for_more_labeling=${currentWork && currentWork.labeler_work_completion ? currentWork.labeler_work_completion.ready_for_more_labeling ?? "" : ""}`,
        `labeler_work_completion_operator_action_required=${currentWork && currentWork.labeler_work_completion ? currentWork.labeler_work_completion.operator_action_required ?? "" : ""}`,
        `browser_mutation_target_contract_met=${currentWork ? currentWork.browser_mutation_target_contract_met ?? "" : ""}`,
        `browser_mutation_target_mismatch_count=${currentWork ? currentWork.browser_mutation_target_mismatch_count ?? "" : ""}`,
        `direct_browser_start_contract_met=${currentWork ? currentWork.direct_browser_start_contract_met ?? "" : ""}`,
        `direct_browser_start_mismatch_count=${currentWork ? currentWork.direct_browser_start_mismatch_count ?? "" : ""}`,
        `single_owner_policy_contract_met=${currentWork ? currentWork.single_owner_policy_contract_met ?? "" : ""}`,
        `data_plane_write_target=${item.data_plane_write_target || ""}`,
        `authoritative_label_state=${item.authoritative_label_state || ""}`,
        `mutable_label_data_plane=${item.mutable_label_data_plane || ""}`,
        `browser_mutation_write_checklist_schema=${mutationWriteChecklist.schema || ""}`,
        `browser_mutation_write_checklist_ready=${mutationWriteChecklist.ready ?? ""}`,
        `browser_mutation_write_checklist_label_mutation_target_kind=${mutationWriteChecklist.label_mutation_target_kind || ""}`,
        `browser_mutation_write_checklist_browser_label_write_target=${mutationWriteChecklist.browser_label_write_target || ""}`,
        `browser_mutation_write_checklist_csv_handoff_artifact_role=${mutationWriteChecklist.csv_handoff_artifact_role || ""}`,
        `browser_mutation_write_checklist_csv_handoff_artifacts_are_label_write_targets=${mutationWriteChecklist.csv_handoff_artifacts_are_label_write_targets ?? ""}`,
        `browser_mutation_write_checklist_handoff_csv_artifacts_are_label_write_targets=${mutationWriteChecklist.handoff_csv_artifacts_are_label_write_targets ?? ""}`,
        `browser_mutation_write_checklist_intermediate_csv_artifacts_are_label_write_targets=${mutationWriteChecklist.intermediate_csv_artifacts_are_label_write_targets ?? ""}`,
        `labeler_route_authorization_checklist_schema=${routeAuthorizationChecklist.schema || ""}`,
        `labeler_route_authorization_checklist_ready=${routeAuthorizationChecklist.ready ?? ""}`,
        `labeler_route_authorization_expected_user_must_match_resolved_user=${routeAuthorizationChecklist.expected_user_must_match_resolved_user ?? ""}`,
        `labeler_route_authorization_expected_user_matches_resolved_user=${routeAuthorizationChecklist.expected_user_matches_resolved_user ?? ""}`,
        `labeler_route_authorization_known_assignment_store_user_required=${routeAuthorizationChecklist.known_assignment_store_user_required ?? ""}`,
        `labeler_route_authorization_known_assignment_store_user=${routeAuthorizationChecklist.known_assignment_store_user ?? ""}`,
        `labeler_route_authorization_active_assignment_required=${routeAuthorizationChecklist.active_assignment_required ?? ""}`,
        `labeler_route_authorization_active_assignment_count=${routeAuthorizationChecklist.active_assignment_count ?? ""}`,
        `labeler_route_authorization_has_active_assignment=${routeAuthorizationChecklist.has_active_assignment ?? ""}`,
        `labeler_route_authorization_single_owner_store_contract_required=${routeAuthorizationChecklist.single_owner_store_contract_required ?? ""}`,
        `labeler_route_authorization_single_owner_store_contract_present=${routeAuthorizationChecklist.single_owner_store_contract_present ?? ""}`,
        `labeler_route_authorization_single_owner_store_contract_ready=${routeAuthorizationChecklist.single_owner_store_contract_ready ?? ""}`,
        `labeler_route_authorization_single_owner_store_contract_met=${routeAuthorizationChecklist.single_owner_store_contract_met ?? ""}`,
        `labeler_route_authorization_single_owner_store_proof_ready=${routeAuthorizationChecklist.single_owner_store_proof_ready ?? ""}`,
        `labeler_route_authorization_assignment_ownership_integrity_ok=${routeAuthorizationChecklist.assignment_ownership_integrity_ok ?? ""}`,
        `labeler_route_authorization_duplicate_active_owner_count=${routeAuthorizationChecklist.duplicate_active_owner_count ?? ""}`,
        `labeler_route_authorization_browser_mutation_target_resolved_server_side=${routeAuthorizationChecklist.browser_mutation_target_resolved_server_side ?? ""}`,
        `labeler_route_authorization_labelers_mutate_assigned_training_zarrs=${routeAuthorizationChecklist.labelers_mutate_assigned_training_zarrs ?? ""}`,
        `labeler_route_authorization_labelers_mutate_intermediate_csvs=${routeAuthorizationChecklist.labelers_mutate_intermediate_csvs ?? ""}`,
        `labeler_route_authorization_task_open_requires_active_assignment=${routeAuthorizationChecklist.task_open_requires_active_assignment ?? ""}`,
        `labeler_route_authorization_task_open_requires_task_assigned_to_resolved_user=${routeAuthorizationChecklist.task_open_requires_task_assigned_to_resolved_user ?? ""}`,
        `labeler_route_authorization_task_open_requires_startable_task_state=${routeAuthorizationChecklist.task_open_requires_startable_task_state ?? ""}`,
        `labeler_route_authorization_mutation_requires_current_session=${routeAuthorizationChecklist.mutation_requires_current_session ?? ""}`,
        `labeler_route_authorization_mutation_requires_active_assignment=${routeAuthorizationChecklist.mutation_requires_active_assignment ?? ""}`,
        `labeler_route_authorization_mutation_requires_current_target_token=${routeAuthorizationChecklist.mutation_requires_current_target_token ?? ""}`,
        `labeler_route_authorization_signed_links_are_entry_hints_not_authorization=${routeAuthorizationChecklist.signed_links_are_entry_hints_not_authorization ?? ""}`,
        `labeler_route_authorization_forwarded_expected_user_links_recheck_identity=${routeAuthorizationChecklist.forwarded_expected_user_links_recheck_identity ?? ""}`,
        `labeler_route_authorization_forwarded_signed_links_recheck_runtime_operator_validation_start_gate=${routeAuthorizationChecklist.forwarded_signed_links_recheck_runtime_operator_validation_start_gate ?? ""}`,
        `direct_start_policy_enabled=${directStartPolicy.enabled}`,
        `direct_start_policy_endpoint_route_template=${directStartPolicy.endpoint_route_template || ""}`,
        `direct_start_policy_same_origin_only=${directStartPolicy.same_origin_only}`,
        `direct_start_policy_exact_route_required=${directStartPolicy.exact_route_required}`,
        `direct_start_policy_endpoint_task_segment_must_match_row_task_id=${directStartPolicy.endpoint_task_segment_must_match_row_task_id}`,
        `direct_start_policy_post_body_expected_user_required=${directStartPolicy.post_body_expected_user_required}`,
        `direct_start_policy_post_body_expected_user_field=${directStartPolicy.post_body_expected_user_field || ""}`,
        `direct_start_policy_denied_start_returns_task_open_authorization_contract=${directStartPolicy.denied_start_returns_task_open_authorization_contract}`,
        `direct_start_policy_denied_start_support_preserves_task_open_authorization_contract=${directStartPolicy.denied_start_support_preserves_task_open_authorization_contract}`,
        `direct_start_policy_denied_start_support_includes_authorization_context=${directStartPolicy.denied_start_support_includes_authorization_context}`,
        `direct_start_policy_denied_start_contract_reports_no_session_created=${directStartPolicy.denied_start_contract_reports_no_session_created}`,
        `direct_start_policy_denied_start_contract_reports_server_authorizes_open_false=${directStartPolicy.denied_start_contract_reports_server_authorizes_open_false}`,
        `direct_start_policy_startable_task_states=${JSON.stringify(directStartPolicy.startable_task_states || [])}`,
        `direct_start_policy_csv_handoff_artifact_role=${directStartPolicy.csv_handoff_artifact_role || ""}`,
        `direct_start_policy_csv_handoff_artifacts_are_label_write_targets=${directStartPolicy.csv_handoff_artifacts_are_label_write_targets}`,
        `direct_start_policy_handoff_csv_artifacts_are_label_write_targets=${directStartPolicy.handoff_csv_artifacts_are_label_write_targets}`,
        `direct_start_policy_intermediate_csv_artifacts_are_label_write_targets=${directStartPolicy.intermediate_csv_artifacts_are_label_write_targets}`,
        `direct_start_policy_browser_writes_csv_or_handoff_files=${directStartPolicy.browser_writes_csv_or_handoff_files}`,
        `direct_start_policy_browser_writes_handoff_csv=${directStartPolicy.browser_writes_handoff_csv}`,
        `direct_start_policy_browser_writes_intermediate_csv=${directStartPolicy.browser_writes_intermediate_csv}`,
        `direct_start_policy_browser_receives_zarr_write_authority=${directStartPolicy.browser_receives_zarr_write_authority}`,
        `direct_start_policy_browser_has_direct_zarr_write_authority=${directStartPolicy.browser_has_direct_zarr_write_authority}`,
        `label_mutation_target_kind=${item.label_mutation_target_kind || ""}`,
        `browser_label_write_target=${item.browser_label_write_target || ""}`,
        `training_zarr_mutations_are_server_owned=${item.training_zarr_mutations_are_server_owned}`,
        `handoff_artifacts_are_metadata_only=${item.handoff_artifacts_are_metadata_only}`,
        `csv_handoff_artifact_role=${item.csv_handoff_artifact_role || ""}`,
        `csv_handoff_artifacts_are_label_write_targets=${item.csv_handoff_artifacts_are_label_write_targets}`,
        `handoff_csv_artifacts_are_label_write_targets=${item.handoff_csv_artifacts_are_label_write_targets}`,
        `intermediate_csv_artifacts_are_label_write_targets=${item.intermediate_csv_artifacts_are_label_write_targets}`,
        `browser_writes_csv_or_handoff_files=${item.browser_writes_csv_or_handoff_files}`,
        `browser_writes_handoff_csv=${item.browser_writes_handoff_csv}`,
        `browser_writes_intermediate_csv=${item.browser_writes_intermediate_csv}`,
        `browser_receives_zarr_write_authority=${item.browser_receives_zarr_write_authority}`,
        `browser_has_direct_zarr_write_authority=${item.browser_has_direct_zarr_write_authority}`,
        `open_task_count=${item.open_task_count ?? ""}`,
        `non_startable_task_count=${item.non_startable_task_count ?? ""}`,
        `task_count=${item.task_count ?? ""}`,
        `recording_count=${item.recording_count ?? ""}`,
        `preferred_labeler_entrypoint=${currentWork && currentWork.preferred_labeler_entrypoint ? currentWork.preferred_labeler_entrypoint : ""}`,
        `preferred_labeler_entry_url=${currentWork && currentWork.preferred_labeler_entry_url ? currentWork.preferred_labeler_entry_url : ""}`,
        `personalized_labeler_entrypoint=${currentWork && currentWork.personalized_labeler_entrypoint ? currentWork.personalized_labeler_entrypoint : ""}`,
        `personalized_labeler_entry_url=${currentWork && currentWork.personalized_labeler_entry_url ? currentWork.personalized_labeler_entry_url : ""}`,
        ...personalizedLaunchReadinessSupportLines(currentWork),
        `queue_first_entry_contract_schema=${currentWork && currentWork.queue_first_entry_contract ? currentWork.queue_first_entry_contract.schema || "" : ""}`,
        `queue_first_entry_contract_ready=${currentWork && currentWork.queue_first_entry_contract ? currentWork.queue_first_entry_contract.ready ?? "" : ""}`,
        `queue_first_entry_contract_preferred_labeler_entrypoint=${currentWork && currentWork.queue_first_entry_contract ? currentWork.queue_first_entry_contract.preferred_labeler_entrypoint || "" : ""}`,
        `queue_first_entry_contract_preferred_labeler_entry_url=${currentWork && currentWork.queue_first_entry_contract ? currentWork.queue_first_entry_contract.preferred_labeler_entry_url || "" : ""}`,
        `queue_first_entry_contract_personalized_labeler_entrypoint=${currentWork && currentWork.queue_first_entry_contract ? currentWork.queue_first_entry_contract.personalized_labeler_entrypoint || "" : ""}`,
        `queue_first_entry_contract_personalized_labeler_entry_url=${currentWork && currentWork.queue_first_entry_contract ? currentWork.queue_first_entry_contract.personalized_labeler_entry_url || "" : ""}`,
        `queue_first_entry_contract_personalized_entry_required=${currentWork && currentWork.queue_first_entry_contract ? currentWork.queue_first_entry_contract.personalized_entry_required ?? "" : ""}`,
        `queue_first_entry_contract_personalized_labeler_entry_url_matches_personal_dataset_queue=${currentWork && currentWork.queue_first_entry_contract ? currentWork.queue_first_entry_contract.personalized_labeler_entry_url_matches_personal_dataset_queue ?? "" : ""}`,
        `queue_first_entry_contract_preferred_labeler_entry_url_matches_personal_dataset_queue=${currentWork && currentWork.queue_first_entry_contract ? currentWork.queue_first_entry_contract.preferred_labeler_entry_url_matches_personal_dataset_queue ?? "" : ""}`,
        `queue_first_entry_contract_preferred_labeler_entry_url_is_expected_user_guarded=${currentWork && currentWork.queue_first_entry_contract ? currentWork.queue_first_entry_contract.preferred_labeler_entry_url_is_expected_user_guarded ?? "" : ""}`,
        `queue_first_entry_contract_personalized_labeler_entry_url_is_expected_user_guarded=${currentWork && currentWork.queue_first_entry_contract ? currentWork.queue_first_entry_contract.personalized_labeler_entry_url_is_expected_user_guarded ?? "" : ""}`,
        `queue_first_entry_contract_landing_ready=${currentWork && currentWork.queue_first_entry_contract ? currentWork.queue_first_entry_contract.landing_ready ?? "" : ""}`,
        `queue_first_entry_contract_labeling_home_ready=${currentWork && currentWork.queue_first_entry_contract ? currentWork.queue_first_entry_contract.labeling_home_ready ?? "" : ""}`,
        `queue_first_entry_contract_dataset_queue_ready=${currentWork && currentWork.queue_first_entry_contract ? currentWork.queue_first_entry_contract.dataset_queue_ready ?? "" : ""}`,
        `queue_first_entry_contract_personal_dataset_queue_ready=${currentWork && currentWork.queue_first_entry_contract ? currentWork.queue_first_entry_contract.personal_dataset_queue_ready ?? "" : ""}`,
        `queue_first_entry_contract_personal_work_ready=${currentWork && currentWork.queue_first_entry_contract ? currentWork.queue_first_entry_contract.personal_work_ready ?? "" : ""}`,
        `queue_first_entry_contract_queue_first_paths_ready=${currentWork && currentWork.queue_first_entry_contract ? currentWork.queue_first_entry_contract.queue_first_paths_ready ?? "" : ""}`,
        `queue_first_entry_contract_datasets_waiting_aliases_ready=${currentWork && currentWork.queue_first_entry_contract ? currentWork.queue_first_entry_contract.datasets_waiting_aliases_ready ?? "" : ""}`,
        `queue_first_entry_contract_expected_user_landing_guard=${currentWork && currentWork.queue_first_entry_contract ? currentWork.queue_first_entry_contract.expected_user_landing_guard ?? "" : ""}`,
        `queue_first_entry_contract_expected_user_queue_guard=${currentWork && currentWork.queue_first_entry_contract ? currentWork.queue_first_entry_contract.expected_user_queue_guard ?? "" : ""}`,
        `queue_first_entry_contract_expected_user_dashboard_guard=${currentWork && currentWork.queue_first_entry_contract ? currentWork.queue_first_entry_contract.expected_user_dashboard_guard ?? "" : ""}`,
        `preferred_labeler_entry_url_matches_dataset_queue=${currentWork && currentWork.preferred_labeler_entry_url_matches_dataset_queue !== undefined ? currentWork.preferred_labeler_entry_url_matches_dataset_queue : ""}`,
        `preferred_labeler_entry_url_matches_personal_dataset_queue=${currentWork && currentWork.preferred_labeler_entry_url_matches_personal_dataset_queue !== undefined ? currentWork.preferred_labeler_entry_url_matches_personal_dataset_queue : ""}`,
        `personalized_labeler_entry_url_matches_personal_dataset_queue=${currentWork && currentWork.personalized_labeler_entry_url_matches_personal_dataset_queue !== undefined ? currentWork.personalized_labeler_entry_url_matches_personal_dataset_queue : ""}`,
        `labeler_landing_link_role=${currentWork && currentWork.labeler_landing_link_role ? currentWork.labeler_landing_link_role : ""}`,
        `personal_dataset_queue_link_role=${currentWork && currentWork.personal_dataset_queue_link_role ? currentWork.personal_dataset_queue_link_role : ""}`,
        `dataset_queue_link_role=${currentWork && currentWork.dataset_queue_link_role ? currentWork.dataset_queue_link_role : ""}`,
        `canonical_dataset_queue_link_role=${currentWork && currentWork.canonical_dataset_queue_link_role ? currentWork.canonical_dataset_queue_link_role : ""}`,
        `dataset_queue_preview_url=${currentWork && currentWork.dataset_queue_preview_url ? currentWork.dataset_queue_preview_url : currentWork && currentWork.expected_user_personal_dataset_queue_url ? currentWork.expected_user_personal_dataset_queue_url : ""}`,
        `canonical_dataset_queue_preview_url=${currentWork && currentWork.canonical_dataset_queue_preview_url ? currentWork.canonical_dataset_queue_preview_url : currentWork && currentWork.expected_user_dataset_queue_url ? currentWork.expected_user_dataset_queue_url : ""}`,
        `dashboard_link_role=${currentWork && currentWork.dashboard_link_role ? currentWork.dashboard_link_role : ""}`,
        `identity_probe_link_role=${currentWork && currentWork.identity_probe_link_role ? currentWork.identity_probe_link_role : ""}`,
        `task_links_role=${currentWork && currentWork.task_links_role ? currentWork.task_links_role : ""}`,
        `expected_user_dataset_queue_url=${currentWork && currentWork.expected_user_dataset_queue_url ? currentWork.expected_user_dataset_queue_url : ""}`,
        `expected_user_labeling_home_url=${currentWork && currentWork.expected_user_labeling_home_url ? currentWork.expected_user_labeling_home_url : ""}`,
        `expected_user_dashboard_url=${currentWork && currentWork.expected_user_dashboard_url ? currentWork.expected_user_dashboard_url : ""}`,
        `expected_user_personal_dataset_queue_url=${currentWork && currentWork.expected_user_personal_dataset_queue_url ? currentWork.expected_user_personal_dataset_queue_url : ""}`,
        `expected_user_personal_work_url=${currentWork && currentWork.expected_user_personal_work_url ? currentWork.expected_user_personal_work_url : ""}`,
        `expected_user_work_url=${item.expected_user_work_url || item.work_url || ""}`
      ].join("\\n");
    }

    function renderDatasetQueue(payload) {
      const queue = payload.dataset_queue || [];
      const target = document.getElementById("dataset-queue");
      if (!queue.length) {
        target.className = "dataset-queue";
        target.innerHTML = `<h2>Datasets waiting for completion</h2><p class="muted">No open dataset work is currently waiting for completion.</p>`;
        return;
      }
      target.className = "dataset-queue";
      target.innerHTML = `<h2>Datasets waiting for completion</h2>` + queue.map((dataset) => {
        const workflowText = Object.entries(dataset.workflow_counts || {}).map(([workflow, count]) =>
          `${workflow}:${count}`
        ).join(", ");
        const recordingText = (dataset.recordings || []).map((recording) =>
          `<a href="${escapeText(recording.expected_user_work_url || recording.work_url || dataset.expected_user_work_url || dataset.work_url || "/work")}">${escapeText(recording.recording_id)}</a> (${escapeText(recording.open_task_count || 0)} startable - ${escapeText(recording.labeler_action || "open_recording")})`
        ).join(", ");
        const workUrl = dataset.expected_user_work_url || dataset.work_url || "/work";
        const supportDetails = dashboardQueueSupportText(dataset);
        return `<div class="dataset-row">
          <b><a href="${escapeText(workUrl)}">${escapeText(dataset.dataset_label || dataset.dataset_id || "Unspecified dataset")}</a></b>
          <br><span class="muted">${escapeText(dataset.open_task_count || 0)} startable / ${escapeText(dataset.task_count || 0)} shown tasks - ${escapeText(dataset.recording_count || 0)} recordings - action ${escapeText(dataset.labeler_action || "open_dataset")}${workflowText ? " - workflows " + escapeText(workflowText) : ""}</span>
          <br><span class="muted">Writes: ${escapeText(dataset.data_plane_write_target || "server_owned_assigned_task_zarr_scope")} - handoff metadata only: ${escapeText(dataset.handoff_artifacts_are_metadata_only)}</span>
          <br><span>${recordingText}</span>
          <details class="operator-error"><summary>Dataset support details</summary><pre>${escapeText(supportDetails)}</pre><button type="button" onclick="copySupportDetails(this)">Copy support details</button></details>
        </div>`;
      }).join("");
    }

    function render(payload) {
      clearDashboardError();
      document.getElementById("user-pill").innerHTML =
        `${escapeText(payload.user)} (${escapeText(payload.auth_source)})` +
        (payload.is_admin ? ` &nbsp; <a href="/admin">Admin</a>` : "");
      const landingLink = document.getElementById("landing-link");
      if (landingLink) landingLink.href = payload.expected_user_labeler_landing_url || guardedWorkPath("/");
      document.getElementById("recording-count").textContent = payload.recording_count;
      const datasetSummary = payload.dataset_queue_summary || {};
      document.getElementById("dataset-count").textContent = datasetSummary.waiting_dataset_count ?? (payload.dataset_queue || []).length;
      document.getElementById("task-count").textContent = `${payload.startable_task_count || 0}/${payload.total_task_count || payload.task_count || 0}`;
      const progressSummary = payload.progress_summary || {};
      document.getElementById("complete-count").textContent = progressSummary.complete_task_count ?? payload.complete_task_count ?? 0;
      document.getElementById("waiting-recording-count").textContent = progressSummary.waiting_recording_count ?? 0;
      document.getElementById("blocked-recording-count").textContent = progressSummary.blocked_recording_count ?? 0;
      document.getElementById("failed-promotion-count").textContent = payload.failed_promotion_count || 0;
      renderDatasetQueue(payload);
      const visibleRecordings = filteredRecordings(payload);
      const visibleTaskCount = visibleRecordings.reduce((total, recording) => total + recording.tasks.length, 0);
      const startableTaskStates = new Set(((payload.task_state_policy || {}).startable_task_states || []).map(String));
      const operatorValidationStartGate = payload.operator_validation_start_gate || {};
      const operatorValidationBlocksStart = operatorValidationStartGate.blocks_task_open === true || operatorValidationStartGate.blocks_task_open === "true";
      const activeFilterLabels = [
        activeLinkFilters.dataset_id ? `dataset=${activeLinkFilters.dataset_id}` : "",
        activeLinkFilters.recording_id ? `recording=${activeLinkFilters.recording_id}` : "",
        activeLinkFilters.task_id ? `task=${activeLinkFilters.task_id}` : "",
        document.getElementById("workflow-filter").value ? `workflow=${document.getElementById("workflow-filter").value}` : "",
        document.getElementById("task-search").value.trim() ? "search" : ""
      ].filter(Boolean);
      document.getElementById("filtered-count").textContent = activeFilterLabels.length
        ? `${visibleTaskCount} visible tasks | ${activeFilterLabels.join(" | ")}`
        : "No filter";
      const failedPromotions = document.getElementById("failed-promotions");
      if (payload.failed_promotions && payload.failed_promotions.length) {
        failedPromotions.className = "failed-promotions";
        failedPromotions.innerHTML = `<h2>Failed promotions needing retry</h2>` + payload.failed_promotions.map((event) => {
          const target = event.target || {};
          const after = event.after || {};
          return `<div class="failed-row">
            <b>${escapeText(event.recording_id)}</b> / ${escapeText(event.workflow_kind)}
            <br>${escapeText(event.created_at_utc)} | frame=${escapeText(target.source_frame_index ?? target.parent_frame_index ?? "")}
            <br><span class="error">${escapeText(after.details || after.error || "promotion failed")}</span>
            <br><span>Ask the operator to inspect and retry this promotion from the admin recovery view after repair.</span>
          </div>`;
        }).join("");
      } else {
        failedPromotions.className = "";
        failedPromotions.innerHTML = "";
      }
      const content = document.getElementById("content");
      if (!payload.recordings.length) {
        content.className = "empty";
        content.textContent = (payload.empty_state && payload.empty_state.message)
          ? String(payload.empty_state.message)
          : "No active labeling recordings are assigned to you right now. If you expected work, ask the operator to check your recording assignment.";
        return;
      }
      if (!visibleRecordings.length) {
        const filterSupport = [
          "error=work_filter_no_matches",
          `user=${payload.user || ""}`,
          `expected_user=${expectedUserGuardParam || payload.expected_user || ""}`,
          `filters=${activeFilterLabels.join(" | ") || "none"}`,
          `dataset_id=${activeLinkFilters.dataset_id || ""}`,
          `recording_id=${activeLinkFilters.recording_id || ""}`,
          `task_id=${activeLinkFilters.task_id || ""}`,
          `workflow=${document.getElementById("workflow-filter").value || ""}`,
          `search=${document.getElementById("task-search").value.trim()}`
        ].join("\\n");
        content.className = "empty operator-error";
        content.innerHTML = `
          <h2>No assigned tasks match the current filters.</h2>
          <p>Clear filters, open the full dashboard, or send these support details to the operator if this came from a handoff link.</p>
          <details>
            <summary>What to send the operator</summary>
            <pre>${escapeText(filterSupport)}</pre>
            <button type="button" onclick="copySupportDetails(this)">Copy support details</button>
          </details>
        `;
        return;
      }
      content.className = "";
      content.innerHTML = visibleRecordings.map((recording) => {
        const workflowStateCounts = recording.workflow_state_counts || {};
        const componentCounts = recording.component_counts || {};
        const progressMeta = `${recording.startable_task_count || 0} startable / ${recording.total_task_count || 0} total; ${recording.non_startable_task_count || 0} non-startable; ${recording.complete_task_count || 0} complete`;
        const workflowMeta = Object.entries(workflowStateCounts).map(([key, value]) =>
          `${key}: ${value.startable || 0} startable, ${value.non_startable || 0} non-startable, ${value.complete || 0} complete`
        ).join(" | ");
        const countMeta = [
          progressMeta,
          workflowMeta,
          Object.entries(componentCounts).length ? "components " + Object.entries(componentCounts).map(([key, value]) => `${key}:${value}`).join(", ") : null
        ].filter(Boolean).map(escapeText).join(" | ");
        const assignmentNotes = recording.assignment_notes
          ? `<div class="assignment-notes"><b>Instructions:</b> ${escapeText(recording.assignment_notes)}</div>`
          : "";
        const tasks = recording.tasks.map((task) => {
          const meta = [
            task.workflow_kind,
            task.dataset_id ? `dataset=${task.dataset_id}` : null,
            task.zarr_use ? `zarr_use=${task.zarr_use}` : null,
            task.stage_group ? `stage=${task.stage_group}` : null,
            task.component_name ? `component=${task.component_name}` : null,
            task.priority != null && task.priority !== "" ? `priority=${task.priority}` : null
          ].filter(Boolean).map(escapeText).join(" | ");
          const taskNotes = task.notes
            ? `<div class="task-notes"><b>Task note:</b> ${escapeText(task.notes)}</div>`
            : "";
          const taskState = String(task.state || "");
          const taskOpenContractReadyRaw = task.direct_browser_start_authorization_contract_ready;
          const taskOpenContractReadyKnown = taskOpenContractReadyRaw !== undefined && taskOpenContractReadyRaw !== null && taskOpenContractReadyRaw !== "";
          const taskOpenContractReady = taskOpenContractReadyRaw === true || taskOpenContractReadyRaw === "true";
          const taskOpenNotReadyReason = operatorValidationBlocksStart
            ? (operatorValidationStartGate.not_ready_reason || "operator_validation_start_blocked")
            : task.direct_browser_start_not_ready_reason || ((task.direct_browser_start_authorization_contract || {}).not_ready_reason || "");
          const taskOpenOperatorAction = operatorValidationBlocksStart
            ? (operatorValidationStartGate.operator_action || "Complete required operator validation evidence before browser Start/Open.")
            : task.direct_browser_start_operator_action || ((task.direct_browser_start_authorization_contract || {}).operator_action || "");
          const canOpenTaskFromDashboard = !operatorValidationBlocksStart && startableTaskStates.has(taskState) && (!taskOpenContractReadyKnown || taskOpenContractReady);
          const actionCell = canOpenTaskFromDashboard
            ? `<button type="button" onclick="openTask('${escapeText(task.task_id)}', this)">Open task</button>`
            : taskState === "complete"
            ? `<span class="muted">Complete; ask the operator to reopen this task if more labeling is required.</span>`
            : `<span class="muted">${operatorValidationBlocksStart ? "Start is waiting for operator validation" : "Task is not startable from the dashboard"}${taskOpenNotReadyReason ? ": " + escapeText(taskOpenNotReadyReason) : ""}; ${operatorValidationBlocksStart ? "ask the operator to complete launch validation." : "ask the operator to move it to pending/in_progress before labeling."}</span>`;
          return `<div class="task">
            <div>
              <div class="task-title">${escapeText(taskTitle(task))}</div>
              <div class="task-meta"><span class="state">${escapeText(task.state)}</span>${meta}</div>
              ${taskOpenContractReadyKnown ? `<div class="task-meta">Task-open authorization contract: ${escapeText(taskOpenContractReady ? "ready" : "not ready")}</div>` : ""}
              ${taskOpenNotReadyReason ? `<div class="task-meta">Task-open not-ready reason: ${escapeText(taskOpenNotReadyReason)}</div>` : ""}
              ${taskOpenOperatorAction ? `<div class="task-meta">Task-open operator action: ${escapeText(taskOpenOperatorAction)}</div>` : ""}
              ${taskNotes}
            </div>
            ${actionCell}
          </div>`;
        }).join("") || `<div class="empty">${escapeText(noOpenTaskMessage(recording))}</div>`;
        return `<article class="recording">
          <h2>${escapeText(recording.recording_id)}</h2>
          ${assignmentNotes}
          <div class="progress-line">${countMeta}</div>
          ${tasks}
        </article>`;
      }).join("");
    }

    async function load() {
      const refreshButton = document.getElementById("refresh-work");
      const content = document.getElementById("content");
      refreshButton.disabled = true;
      clearDashboardError();
      content.className = "empty";
      content.textContent = "Refreshing assigned work...";
      try {
        const includeCompleted = document.getElementById("include-completed").checked;
        const params = authQueryParams();
        if (includeCompleted) params.set("include_completed", "1");
        const query = params.toString();
        const response = await fetch(`/api/me/tasks${query ? "?" + query : ""}`);
        const payload = await readApiPayload(response);
        if (!response.ok || !payload.ok) {
          throw dashboardFailure(response, payload, "dashboard_load_failed");
        }
        currentWork = payload.work;
        updateWorkflowOptions(currentWork);
        render(currentWork);
      } catch (error) {
        content.className = "empty error";
        content.textContent = "Assigned work could not be loaded. Send the support details above to the operator.";
        showDashboardError(error, "Palette could not load your assigned work.", "dashboard_load_failed");
      } finally {
        refreshButton.disabled = false;
      }
    }

    document.getElementById("task-search").addEventListener("input", () => {
      if (currentWork) render(currentWork);
    });
    document.getElementById("workflow-filter").addEventListener("change", () => {
      if (currentWork) render(currentWork);
    });
    document.getElementById("clear-filters").addEventListener("click", () => {
      document.getElementById("task-search").value = "";
      document.getElementById("workflow-filter").value = "";
      document.getElementById("include-completed").checked = false;
      clearStructuredFilters();
      if (currentWork) render(currentWork);
    });
    document.getElementById("include-completed").addEventListener("change", () => {
      load();
    });
    document.getElementById("refresh-work").addEventListener("click", () => {
      load();
    });
    document.getElementById("copy-landing-link").addEventListener("click", (event) => {
      copyDashboardLandingLink(event.currentTarget);
    });
    setDashboardEntryLinks();
    load();
  </script>
</body>
</html>
"""


def _datasets_html() -> bytes:
    body = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Palette dataset queue</title>
  <style>
    :root {
      --paper: #fffaf0;
      --ink: #252118;
      --muted: #695f50;
      --line: #d9c7a7;
      --accent: #2f6f73;
      --accent-soft: #dff1ed;
      --warn: #9a3027;
      --shadow: rgba(73, 54, 25, 0.15);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "Avenir Next", "Trebuchet MS", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at 8% 12%, rgba(47, 111, 115, 0.14), transparent 28rem),
        linear-gradient(135deg, #fffdf7 0%, var(--paper) 45%, #f0ead9 100%);
      min-height: 100vh;
    }
    main {
      width: min(1100px, calc(100vw - 32px));
      margin: 0 auto;
      padding: 28px 0 42px;
    }
    header {
      display: flex;
      justify-content: space-between;
      gap: 18px;
      align-items: flex-start;
      margin-bottom: 18px;
    }
    h1 {
      margin: 0;
      font-size: clamp(2rem, 6vw, 4.2rem);
      letter-spacing: -0.06em;
      line-height: 0.95;
    }
    .subhead {
      max-width: 720px;
      color: var(--muted);
      font-size: 1.05rem;
    }
    .pill, .card, .dataset {
      border: 1px solid var(--line);
      border-radius: 20px;
      background: rgba(255, 253, 246, 0.78);
      box-shadow: 0 16px 36px var(--shadow);
    }
    .pill {
      padding: 10px 14px;
      white-space: nowrap;
      color: var(--muted);
    }
    .actions {
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      margin: 18px 0;
    }
    a.button, button {
      border: 0;
      border-radius: 999px;
      background: var(--ink);
      color: white;
      padding: 10px 14px;
      text-decoration: none;
      font: inherit;
      cursor: pointer;
    }
    a.button.secondary, button.secondary {
      background: var(--accent-soft);
      color: var(--accent);
    }
    .summary {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
      gap: 12px;
      margin: 16px 0 20px;
    }
    .card {
      padding: 14px;
    }
    .card b {
      display: block;
      font-size: 1.8rem;
      line-height: 1;
    }
    .muted {
      color: var(--muted);
    }
    .notice {
      border: 1px solid var(--line);
      border-left: 5px solid var(--accent);
      border-radius: 18px;
      background: rgba(255, 253, 246, 0.82);
      padding: 12px 14px;
      margin: 14px 0 18px;
    }
    .notice b {
      display: block;
      margin-bottom: 4px;
    }
    #blocked-recordings, #reassignment-session-safety {
      display: none;
      border-left-color: var(--warn);
    }
    #blocked-recordings.active, #reassignment-session-safety.active {
      display: block;
    }
    #error {
      display: none;
      border: 1px solid rgba(154, 48, 39, 0.3);
      border-radius: 16px;
      background: #fff8f5;
      padding: 12px;
      color: var(--warn);
    }
    #error.active {
      display: block;
    }
    #error pre {
      margin: 10px 0;
      padding: 10px;
      border-radius: 12px;
      background: #fffef9;
      color: var(--ink);
      white-space: pre-wrap;
      overflow-x: auto;
    }
    .notice pre {
      margin: 10px 0;
      padding: 10px;
      border-radius: 12px;
      background: #fffef9;
      color: var(--ink);
      white-space: pre-wrap;
      overflow-x: auto;
    }
    .dataset {
      padding: 18px;
      margin: 14px 0;
    }
    .dataset h2 {
      margin: 0 0 6px;
      font-size: 1.35rem;
    }
    .recordings {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      margin-top: 12px;
    }
    .recording-chip {
      display: inline-flex;
      gap: 6px;
      align-items: center;
      flex-wrap: wrap;
    }
    .recordings a {
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 7px 10px;
      color: var(--accent);
      background: #fffef9;
      text-decoration: none;
    }
    .tasks {
      display: grid;
      gap: 8px;
      margin-top: 14px;
    }
    .task-row {
      border: 1px solid var(--line);
      border-radius: 14px;
      background: rgba(255, 254, 249, 0.9);
      padding: 10px;
    }
    .task-row a {
      color: var(--accent);
      font-weight: 800;
      text-decoration: none;
    }
    .task-meta, .task-note {
      margin-top: 3px;
      color: var(--muted);
      font-size: 0.92rem;
    }
    @media (max-width: 680px) {
      header { display: block; }
      .pill { display: inline-block; margin-top: 12px; white-space: normal; }
    }
  </style>
</head>
<body>
  <main>
    <header>
      <div>
        <p class="eyebrow">Datasets waiting for completion</p>
        <h1>Your datasets waiting for completion</h1>
        <p class="subhead">A personalized queue of assigned dataset and recording work. Each recording has one active assigned owner. Open a startable task directly from this page, or use the full browser labeling dashboard as a fallback. Browser saves stay on assigned task/training Zarr writers; CSV, HTML, JSON, and handoff files are metadata only.</p>
        <p class="muted">Support fields include <code>identity_personal_queue_evidence_status</code> and <code>operator_validation_identity_personal_queue_evidence_status_values</code>.</p>
      </div>
      <div class="pill" id="user-pill">Loading user...</div>
    </header>
    <section class="actions">
      <a class="button secondary" id="landing-link" href="/">Open landing page</a>
      <a class="button secondary" id="work-link" href="/work">Open full work dashboard</a>
      <a class="button secondary" id="identity-link" href="/identity">Open identity check</a>
      <button type="button" id="copy-landing-link">Copy start link</button>
      <button type="button" id="refresh">Refresh queue</button>
    </section>
    <section class="notice">
      <b>Check your identity before opening work.</b>
      Confirm this page shows your expected user, use only work assigned to you, and stop if a copied link shows another user. No local Palette or Crimson installation is needed; do not edit zarr files directly or forward handoff links.
    </section>
    <section class="summary">
      <div class="card"><b id="dataset-count">-</b><span class="muted">waiting datasets</span></div>
      <div class="card"><b id="task-count">-</b><span class="muted">startable queue tasks</span></div>
      <div class="card"><b id="complete-count">-</b><span class="muted">completed tasks</span></div>
      <div class="card"><b id="blocked-count">-</b><span class="muted">blocked/no-open recordings</span></div>
    </section>
    <section id="queue-state" class="notice"></section>
    <section id="reassignment-session-safety" class="notice"></section>
    <section id="blocked-recordings" class="notice"></section>
    <section id="backup-policy" class="notice"></section>
    <section id="audit-policy" class="notice"></section>
    <section id="session-guard-policy" class="notice"></section>
    <section id="error"></section>
    <section id="queue" class="muted">Loading personalized dataset queue...</section>
  </main>
  <script>
    const initialQuery = new URLSearchParams(window.location.search);
    const expectedUserGuardParam = initialQuery.get("expected_user") || "";
    const inviteTokenParam = initialQuery.get("invite") || "";

    function escapeText(value) {
      return String(value ?? "").replace(/[&<>"']/g, (ch) => ({
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
        "'": "&#39;"
      }[ch]));
    }

    function guardedPath(path) {
      const url = new URL(path, window.location.href);
      if (expectedUserGuardParam) url.searchParams.set("expected_user", expectedUserGuardParam);
      if (inviteTokenParam) url.searchParams.set("invite", inviteTokenParam);
      return url.pathname + url.search;
    }

    function setGuardedEntryLinks() {
      document.getElementById("landing-link").href = guardedPath("/");
      document.getElementById("work-link").href = guardedPath("/work");
      document.getElementById("identity-link").href = guardedPath("/identity");
    }

    async function copyText(button, text, resetText) {
      if (!text) return;
      try {
        await navigator.clipboard.writeText(text);
        button.textContent = "Copied";
        window.setTimeout(() => { button.textContent = resetText; }, 1800);
      } catch (_error) {
        const scratch = document.createElement("textarea");
        scratch.value = text;
        scratch.setAttribute("readonly", "readonly");
        scratch.style.position = "fixed";
        scratch.style.left = "-9999px";
        document.body.appendChild(scratch);
        scratch.focus();
        scratch.select();
        document.execCommand("copy");
        scratch.remove();
        button.textContent = "Copied";
        window.setTimeout(() => { button.textContent = resetText; }, 1800);
      }
    }

    async function copyLandingLink(button) {
      const link = document.getElementById("landing-link");
      const href = link ? new URL(link.getAttribute("href") || "/", window.location.href).href : window.location.href;
      await copyText(button, href, "Copy start link");
    }

    async function readApiPayload(response) {
      const text = await response.text();
      if (!text) return {};
      try {
        return JSON.parse(text);
      } catch (error) {
        return {ok: false, error: "invalid_json", details: text.slice(0, 1000)};
      }
    }

    function supportText(payload, fallback) {
      const contract = (payload && payload.task_open_authorization_contract) || {};
      const context = (payload && payload.authorization_context) || {};
      return [
        fallback,
        `url=${window.location.href}`,
        `expected_user=${expectedUserGuardParam || ""}`,
        `error=${payload.error || "unknown"}`,
        `status=${payload.status || "client"}`,
        `details=${payload.details || ""}`,
        `authorization_resolved_user=${context.resolved_user || ""}`,
        `authorization_expected_user=${context.expected_user || ""}`,
        `authorization_return_expected_user=${context.return_expected_user || ""}`,
        `authorization_return_personal_dataset_queue_url=${context.return_personal_dataset_queue_url || ""}`,
        `authorization_return_personal_dataset_queue_expected_user_guarded=${context.return_personal_dataset_queue_expected_user_guarded ?? ""}`,
        `authorization_return_personal_work_url=${context.return_personal_work_url || ""}`,
        `authorization_return_personal_work_expected_user_guarded=${context.return_personal_work_expected_user_guarded ?? ""}`,
        `authorization_task_id=${context.task_id || ""}`,
        `authorization_recording_id=${context.recording_id || ""}`,
        `task_open_authorization_contract_schema=${contract.schema || ""}`,
        `task_open_authorization_contract_ready=${contract.ready ?? ""}`,
        `task_open_authorization_contract_not_ready_reason=${contract.not_ready_reason || ""}`,
        `task_open_expected_user_guard_checked_server_side=${contract.expected_user_guard_checked_server_side ?? ""}`,
        `task_open_expected_user_guard_present=${contract.expected_user_guard_present ?? ""}`,
        `task_open_expected_user_matches_resolved_user=${contract.expected_user_matches_resolved_user ?? ""}`,
        `task_open_active_assignment_present=${contract.active_assignment_present ?? ""}`,
        `task_open_task_assigned_to_resolved_user=${contract.task_assigned_to_resolved_user ?? ""}`,
        `task_open_assignment_status_active=${contract.assignment_status_active ?? ""}`,
        `task_open_task_state_startable=${contract.task_state_startable ?? ""}`,
        `task_open_reassignment_session_safety_checked_server_side=${contract.reassignment_session_safety_checked_server_side ?? ""}`,
        `task_open_reassignment_session_safety_passed=${contract.reassignment_session_safety_passed ?? ""}`,
        `task_open_session_created_server_side=${contract.session_created_server_side ?? ""}`,
        `task_open_server_authorizes_open=${contract.server_authorizes_open ?? ""}`,
        `task_open_operator_validation_start_gate_required=${contract.operator_validation_start_gate_required ?? ""}`,
        `task_open_operator_validation_start_gate_ready=${contract.operator_validation_start_gate_ready ?? ""}`,
        `task_open_operator_validation_start_gate_blocks_task_open=${contract.operator_validation_start_gate_blocks_task_open ?? ""}`,
        `task_open_operator_validation_start_gate_not_ready_reason=${contract.operator_validation_start_gate_not_ready_reason || ""}`,
        `task_open_operator_validation_status=${contract.operator_validation_status || ""}`,
        `task_open_operator_validation_pending_gate_ids=${JSON.stringify(contract.operator_validation_pending_gate_ids || [])}`,
        `task_open_operator_validation_required_missing_evidence_gate_ids=${JSON.stringify(contract.operator_validation_required_missing_evidence_gate_ids || [])}`,
        `task_open_browser_label_write_target=${contract.browser_label_write_target || ""}`,
        `task_open_browser_writes_csv_or_handoff_files=${contract.browser_writes_csv_or_handoff_files ?? ""}`,
        `task_open_browser_has_direct_zarr_write_authority=${contract.browser_has_direct_zarr_write_authority ?? ""}`
      ].filter(Boolean).join("\\n");
    }

    function showError(payload, fallback) {
      const target = document.getElementById("error");
      const text = supportText(payload, fallback);
      target.className = "active";
      target.innerHTML = `<b>${escapeText(fallback)}</b>
        <p class="muted">Stop and send these support details to the operator if the identity or assignment looks wrong.</p>
        <details>
          <summary>What to send the operator</summary>
          <pre>${escapeText(text)}</pre>
          <button type="button" onclick="copyDatasetSupport(this)">Copy support details</button>
        </details>`;
    }

    async function copyDatasetSupport(button) {
      const block = button.closest("#error") || button.closest("#queue-state") || button.closest("#reassignment-session-safety") || button.closest("#blocked-recordings") || button.closest("#backup-policy") || button.closest("#audit-policy") || button.closest("#session-guard-policy");
      const text = block ? (block.querySelector("pre")?.textContent || "") : "";
      if (!text) return;
      const resetText = button.dataset.copyReset || "Copy support details";
      try {
        await navigator.clipboard.writeText(text);
        button.textContent = "Copied";
        window.setTimeout(() => { button.textContent = resetText; }, 1800);
      } catch (_error) {
        const range = document.createRange();
        const pre = block ? block.querySelector("pre") : null;
        if (!pre) return;
        range.selectNodeContents(pre);
        const selection = window.getSelection();
        if (selection) {
          selection.removeAllRanges();
          selection.addRange(range);
        }
        document.execCommand("copy");
        button.textContent = "Copied";
        window.setTimeout(() => { button.textContent = resetText; }, 1800);
      }
    }

    let currentDatasetQueuePayload = null;

    const OPERATOR_VALIDATION_GATE_IDS = [
      "mutable_zarr_backup_confirmation",
      "browser_response_security_headers",
      "identity_probe_verification",
      "browser_smoke",
      "disposable_zarr_mutation_smoke",
      "operator_recovery_contract",
    ];
    const OPERATOR_VALIDATION_GATE_FIELD_SUFFIXES = [
      "status",
      "pending",
      "missing_evidence",
      "needs_review",
      "passed",
    ];

    function operatorValidationGateSupportLines(source) {
      const item = source || {};
      return OPERATOR_VALIDATION_GATE_IDS.flatMap((gateId) => {
        const prefix = `operator_validation_gate_${gateId}`;
        return OPERATOR_VALIDATION_GATE_FIELD_SUFFIXES.map((suffix) => (
          `${prefix}_${suffix}=${item[`${prefix}_${suffix}`] ?? ""}`
        ));
      });
    }

    function safeShareExternalLaunchEvidenceGapSupportLines(source) {
      const item = source || {};
      return [
        `safe_share_external_launch_evidence_gap_gate_ids=${JSON.stringify(item.safe_share_external_launch_evidence_gap_gate_ids || [])}`,
        `safe_share_external_launch_evidence_gap_count=${item.safe_share_external_launch_evidence_gap_count ?? ""}`,
        `safe_share_external_launch_evidence_gap_statuses=${JSON.stringify(item.safe_share_external_launch_evidence_gap_statuses || {})}`,
        `safe_share_external_launch_evidence_gap_action_required=${item.safe_share_external_launch_evidence_gap_action_required ?? ""}`,
        `safe_share_external_launch_evidence_gap_summary=${item.safe_share_external_launch_evidence_gap_summary || ""}`,
        `safe_share_external_launch_evidence_gap_todos=${JSON.stringify(item.safe_share_external_launch_evidence_gap_todos || [])}`,
        `safe_share_external_launch_evidence_gap_todo_count=${item.safe_share_external_launch_evidence_gap_todo_count ?? ""}`,
        `safe_share_external_launch_evidence_gap_todo_fields=${JSON.stringify(item.safe_share_external_launch_evidence_gap_todo_fields || [])}`,
        `safe_share_external_launch_evidence_gap_template_paths_by_gate_id=${JSON.stringify(item.safe_share_external_launch_evidence_gap_template_paths_by_gate_id || {})}`,
        `safe_share_external_launch_evidence_gap_record_command_ids_by_gate_id=${JSON.stringify(item.safe_share_external_launch_evidence_gap_record_command_ids_by_gate_id || {})}`,
      ];
    }

    function personalizedLaunchReadinessSupportLines(source) {
      const readiness = (source && source.personalized_launch_readiness) || {};
      return [
        `personalized_launch_readiness=${JSON.stringify(readiness || {})}`,
        `personalized_launch_readiness_schema=${readiness.schema || ""}`,
        `personalized_launch_readiness_field_count=${readiness.field_count ?? ""}`,
        `personalized_launch_readiness_personalized_labeler_entry_url=${readiness.personalized_labeler_entry_url || ""}`,
        `personalized_launch_readiness_labeler_start_ready=${readiness.labeler_start_ready ?? ""}`,
        `personalized_launch_readiness_labeler_work_completion_status=${readiness.labeler_work_completion_status || ""}`,
        `personalized_launch_readiness_external_launch_evidence_gap_count=${readiness.external_launch_evidence_gap_count ?? ""}`,
        `personalized_launch_readiness_external_launch_evidence_gap_gate_ids=${JSON.stringify(readiness.external_launch_evidence_gap_gate_ids || [])}`,
        `personalized_launch_readiness_external_launch_evidence_gap_todo_count=${readiness.external_launch_evidence_gap_todo_count ?? ""}`,
        `personalized_launch_readiness_external_launch_evidence_gap_todos=${JSON.stringify(readiness.external_launch_evidence_gap_todos || [])}`,
        `personalized_launch_readiness_browser_label_write_target=${readiness.browser_label_write_target || ""}`,
        `personalized_launch_readiness_browser_writes_csv_or_handoff_files=${readiness.browser_writes_csv_or_handoff_files ?? ""}`,
        `personalized_launch_readiness_browser_has_direct_zarr_write_authority=${readiness.browser_has_direct_zarr_write_authority ?? ""}`,
      ];
    }

    function supportDetailsText(support) {
      const row = support || {};
      const operatorSupport = row.operator_support || {};
      const operatorSupportValue = (name, fallback = "") => (
        operatorSupport[name] !== undefined ? operatorSupport[name] : (row[name] !== undefined ? row[name] : fallback)
      );
      const datasetQueuePayload = currentDatasetQueuePayload || {};
      const directStartPolicy = currentDatasetQueuePayload && currentDatasetQueuePayload.dataset_queue_direct_start_policy
        ? currentDatasetQueuePayload.dataset_queue_direct_start_policy
        : {};
      const runtimeGateCliPolicy = currentDatasetQueuePayload && currentDatasetQueuePayload.runtime_operator_validation_gate_cli_policy
        ? currentDatasetQueuePayload.runtime_operator_validation_gate_cli_policy
        : {};
      const mutationWriteChecklist = currentDatasetQueuePayload && currentDatasetQueuePayload.browser_mutation_write_checklist
        ? currentDatasetQueuePayload.browser_mutation_write_checklist
        : {};
      const routeAuthorizationChecklist = currentDatasetQueuePayload && currentDatasetQueuePayload.labeler_route_authorization_checklist
        ? currentDatasetQueuePayload.labeler_route_authorization_checklist
        : {};
      const reassignmentSessionSafety = currentDatasetQueuePayload && currentDatasetQueuePayload.reassignment_session_safety
        ? currentDatasetQueuePayload.reassignment_session_safety
        : {};
      const operatorValidationStartGate = currentDatasetQueuePayload && currentDatasetQueuePayload.operator_validation_start_gate
        ? currentDatasetQueuePayload.operator_validation_start_gate
        : {};
      const operatorValidationMutationGate = currentDatasetQueuePayload && currentDatasetQueuePayload.operator_validation_mutation_gate
        ? currentDatasetQueuePayload.operator_validation_mutation_gate
        : {};
      const operatorValidationVisibilityPolicy = currentDatasetQueuePayload && currentDatasetQueuePayload.operator_validation_visibility_policy
        ? currentDatasetQueuePayload.operator_validation_visibility_policy
        : {};
      const operatorValidationCommandTemplates = currentDatasetQueuePayload && currentDatasetQueuePayload.operator_validation_command_templates
        ? currentDatasetQueuePayload.operator_validation_command_templates
        : {};
      const singleOwnerPolicy = currentDatasetQueuePayload && currentDatasetQueuePayload.single_owner_policy
        ? currentDatasetQueuePayload.single_owner_policy
        : {};
      return [
        "page_context=dataset_queue",
        `user=${row.user || ""}`,
        `expected_user=${expectedUserGuardParam || ""}`,
        `single_owner_policy_assignment_scope=${datasetQueuePayload.single_owner_policy_assignment_scope ?? singleOwnerPolicy.assignment_scope ?? ""}`,
        `single_owner_policy_recording_assignment_key=${datasetQueuePayload.single_owner_policy_recording_assignment_key ?? singleOwnerPolicy.recording_assignment_key ?? ""}`,
        `single_owner_policy_one_current_assignment_row_per_recording=${datasetQueuePayload.single_owner_policy_one_current_assignment_row_per_recording ?? singleOwnerPolicy.one_current_assignment_row_per_recording ?? ""}`,
        `single_owner_policy_one_active_owner=${datasetQueuePayload.single_owner_policy_one_active_owner ?? singleOwnerPolicy.one_active_owner ?? ""}`,
        `single_owner_policy_multiple_labelers_per_recording_allowed=${datasetQueuePayload.single_owner_policy_multiple_labelers_per_recording_allowed ?? singleOwnerPolicy.multiple_labelers_per_recording_allowed ?? ""}`,
        `single_owner_policy_assignment_user_match_required_for_mutation=${datasetQueuePayload.single_owner_policy_assignment_user_match_required_for_mutation ?? singleOwnerPolicy.assignment_user_match_required_for_mutation ?? ""}`,
        `single_owner_policy_browser_mutation_requires_current_assignment_owner=${datasetQueuePayload.single_owner_policy_browser_mutation_requires_current_assignment_owner ?? singleOwnerPolicy.browser_mutation_requires_current_assignment_owner ?? ""}`,
        `single_owner_policy_browser_mutation_target_resolved_server_side=${datasetQueuePayload.single_owner_policy_browser_mutation_target_resolved_server_side ?? singleOwnerPolicy.browser_mutation_target_resolved_server_side ?? ""}`,
        `single_owner_policy_browser_mutation_target_source=${datasetQueuePayload.single_owner_policy_browser_mutation_target_source ?? singleOwnerPolicy.browser_mutation_target_source ?? ""}`,
        `single_owner_policy_labelers_mutate_assigned_training_zarrs=${datasetQueuePayload.single_owner_policy_labelers_mutate_assigned_training_zarrs ?? singleOwnerPolicy.labelers_mutate_assigned_training_zarrs ?? ""}`,
        `single_owner_policy_labelers_mutate_intermediate_csvs=${datasetQueuePayload.single_owner_policy_labelers_mutate_intermediate_csvs ?? singleOwnerPolicy.labelers_mutate_intermediate_csvs ?? ""}`,
        `assignment_ownership_contract_store_single_owner_assignment_contract_present=${datasetQueuePayload.assignment_ownership_contract_store_single_owner_assignment_contract_present ?? ""}`,
        `assignment_ownership_contract_store_single_owner_assignment_contract_ready=${datasetQueuePayload.assignment_ownership_contract_store_single_owner_assignment_contract_ready ?? ""}`,
        `assignment_ownership_contract_store_single_owner_assignment_contract_met=${datasetQueuePayload.assignment_ownership_contract_store_single_owner_assignment_contract_met ?? ""}`,
        `assignment_ownership_contract_store_single_owner_assignment_contract_schema=${datasetQueuePayload.assignment_ownership_contract_store_single_owner_assignment_contract_schema ?? ""}`,
        `assignment_ownership_contract_browser_mutation_target_resolved_server_side=${datasetQueuePayload.assignment_ownership_contract_browser_mutation_target_resolved_server_side ?? ""}`,
        `assignment_ownership_contract_labelers_mutate_assigned_training_zarrs=${datasetQueuePayload.assignment_ownership_contract_labelers_mutate_assigned_training_zarrs ?? ""}`,
        `assignment_ownership_contract_labelers_mutate_intermediate_csvs=${datasetQueuePayload.assignment_ownership_contract_labelers_mutate_intermediate_csvs ?? ""}`,
        `handoff_status=${datasetQueuePayload.status || ""}`,
        `handoff_ready_to_send=${datasetQueuePayload.ready_to_send ?? ""}`,
        `handoff_sendability_reasons=${JSON.stringify(datasetQueuePayload.sendability_reasons || [])}`,
        `handoff_sendability_actions=${JSON.stringify(datasetQueuePayload.sendability_actions || [])}`,
        `runtime_operator_validation_gate_cli_policy_preferred_require_flag=${runtimeGateCliPolicy.preferred_require_flag || ""}`,
        `runtime_operator_validation_gate_cli_policy_legacy_require_flag=${runtimeGateCliPolicy.legacy_require_flag || ""}`,
        `runtime_operator_validation_gate_cli_policy_validation_checklist_flag=${runtimeGateCliPolicy.validation_checklist_flag || ""}`,
        `runtime_operator_validation_gate_cli_policy_protects_browser_start_open=${runtimeGateCliPolicy.protects_browser_start_open ?? ""}`,
        `runtime_operator_validation_gate_cli_policy_protects_browser_mutations=${runtimeGateCliPolicy.protects_browser_mutations ?? ""}`,
        `runtime_operator_validation_gate_cli_policy_blocks_before_target_token_check=${runtimeGateCliPolicy.blocks_before_target_token_check ?? ""}`,
        `runtime_operator_validation_gate_cli_policy_blocks_before_zarr_write=${runtimeGateCliPolicy.blocks_before_zarr_write ?? ""}`,
        `runtime_operator_validation_gate_cli_policy_blocks_before_audit_event_creation=${runtimeGateCliPolicy.blocks_before_audit_event_creation ?? ""}`,
        `operator_validation_start_gate_required=${operatorValidationStartGate.required_for_browser_start ?? ""}`,
        `operator_validation_start_gate_ready=${operatorValidationStartGate.ready ?? ""}`,
        `operator_validation_start_gate_blocks_task_open=${operatorValidationStartGate.blocks_task_open ?? ""}`,
        `operator_validation_start_gate_not_ready_reason=${operatorValidationStartGate.not_ready_reason || ""}`,
        `operator_validation_mutation_gate_required=${operatorValidationMutationGate.required_for_browser_mutation ?? ""}`,
        `operator_validation_mutation_gate_ready=${operatorValidationMutationGate.ready ?? ""}`,
        `operator_validation_mutation_gate_blocks_browser_mutation=${operatorValidationMutationGate.blocks_browser_mutation ?? ""}`,
        `operator_validation_mutation_gate_not_ready_reason=${operatorValidationMutationGate.not_ready_reason || ""}`,
        `operator_validation_mutation_gate_pending_gate_ids=${JSON.stringify(operatorValidationMutationGate.operator_validation_pending_gate_ids || [])}`,
        `operator_validation_mutation_gate_required_missing_evidence_gate_ids=${JSON.stringify(operatorValidationMutationGate.operator_validation_required_missing_evidence_gate_ids || [])}`,
        `operator_validation_start_gate_operator_validation_status=${operatorValidationStartGate.operator_validation_status || ""}`,
        `operator_validation_start_gate_pending_gate_ids=${JSON.stringify(operatorValidationStartGate.operator_validation_pending_gate_ids || [])}`,
        `operator_validation_start_gate_required_missing_evidence_gate_ids=${JSON.stringify(operatorValidationStartGate.operator_validation_required_missing_evidence_gate_ids || [])}`,
        `operator_validation_status=${datasetQueuePayload.operator_validation_status || ""}`,
        `operator_validation_source=${datasetQueuePayload.operator_validation_source || ""}`,
        `operator_validation_gate_count=${datasetQueuePayload.operator_validation_gate_count ?? ""}`,
        `operator_validation_pending_gate_ids=${JSON.stringify(datasetQueuePayload.operator_validation_pending_gate_ids || [])}`,
        `operator_validation_required_missing_evidence_gate_ids=${JSON.stringify(datasetQueuePayload.operator_validation_required_missing_evidence_gate_ids || [])}`,
        `operator_validation_gate_status_values=${JSON.stringify(operatorValidationVisibilityPolicy.operator_validation_gate_status_values || [])}`,
        `operator_validation_gate_ids=${JSON.stringify(OPERATOR_VALIDATION_GATE_IDS)}`,
        `operator_validation_gate_flat_field_suffixes=${JSON.stringify(OPERATOR_VALIDATION_GATE_FIELD_SUFFIXES)}`,
        ...operatorValidationGateSupportLines(datasetQueuePayload),
        ...safeShareExternalLaunchEvidenceGapSupportLines(datasetQueuePayload),
        `operator_validation_public_fields=${JSON.stringify(operatorValidationVisibilityPolicy.public_fields || [])}`,
        `operator_validation_operator_only_fields=${JSON.stringify(operatorValidationVisibilityPolicy.operator_only_fields || [])}`,
        `operator_validation_operator_action_fields=${JSON.stringify(operatorValidationVisibilityPolicy.operator_action_fields || [])}`,
        `operator_validation_operator_action_fields_are_labeler_instructions=${operatorValidationVisibilityPolicy.operator_action_fields_are_labeler_instructions ?? ""}`,
        `operator_validation_labeler_visible_payloads_may_include_operator_action_fields_for_support=${operatorValidationVisibilityPolicy.labeler_visible_payloads_may_include_operator_action_fields_for_support ?? ""}`,
        `operator_validation_labeler_visible_payloads_include_operator_only_fields=${operatorValidationVisibilityPolicy.labeler_visible_payloads_include_operator_only_fields ?? ""}`,
        `operator_validation_per_user_payloads_use_public_fields_only=${operatorValidationVisibilityPolicy.per_user_payloads_use_public_fields_only ?? ""}`,
        `operator_validation_top_level_operator_reports_may_include_operator_only_fields=${operatorValidationVisibilityPolicy.top_level_operator_reports_may_include_operator_only_fields ?? ""}`,
        `operator_validation_required_pending_gate_count=${datasetQueuePayload.operator_validation_required_pending_gate_count ?? ""}`,
        `operator_validation_needs_review_gate_count=${datasetQueuePayload.operator_validation_needs_review_gate_count ?? ""}`,
        `operator_validation_required_missing_evidence_gate_count=${datasetQueuePayload.operator_validation_required_missing_evidence_gate_count ?? ""}`,
        `operator_validation_operator_action=${datasetQueuePayload.operator_validation_operator_action || ""}`,
        `operator_validation_external_evidence_required=${datasetQueuePayload.operator_validation_external_evidence_required ?? ""}`,
        `operator_validation_external_evidence_required_gate_ids=${JSON.stringify(datasetQueuePayload.operator_validation_external_evidence_required_gate_ids || [])}`,
        `operator_validation_external_evidence_required_gate_count=${datasetQueuePayload.operator_validation_external_evidence_required_gate_count ?? ""}`,
        `operator_validation_external_evidence_template_fields_by_gate_id=${JSON.stringify(datasetQueuePayload.operator_validation_external_evidence_template_fields_by_gate_id || {})}`,
        `operator_validation_external_evidence_template_paths_by_gate_id=${JSON.stringify(datasetQueuePayload.operator_validation_external_evidence_template_paths_by_gate_id || {})}`,
        `operator_validation_checklist_only_required_gate_ids=${JSON.stringify(datasetQueuePayload.operator_validation_checklist_only_required_gate_ids || [])}`,
        `operator_validation_checklist_only_required_gate_count=${datasetQueuePayload.operator_validation_checklist_only_required_gate_count ?? ""}`,
        `operator_validation_command_template_schema=${operatorValidationCommandTemplates.schema || ""}`,
        `operator_validation_command_template_command_count=${operatorValidationCommandTemplates.command_count ?? ""}`,
        `operator_validation_command_template_command_ids=${JSON.stringify(operatorValidationCommandTemplates.command_ids || [])}`,
        `operator_validation_command_template_template_backed_gate_ids=${JSON.stringify(operatorValidationCommandTemplates.template_backed_gate_ids || [])}`,
        `operator_validation_command_template_validation_checklist_gate_ids=${JSON.stringify(operatorValidationCommandTemplates.validation_checklist_gate_ids || [])}`,
        `operator_validation_command_template_apply_required_gate_ids=${JSON.stringify(operatorValidationCommandTemplates.apply_required_gate_ids || [])}`,
        `operator_validation_command_template_evidence_template_fields_by_gate_id=${JSON.stringify(operatorValidationCommandTemplates.evidence_template_fields_by_gate_id || {})}`,
        `operator_validation_command_template_evidence_template_paths_by_gate_id=${JSON.stringify(operatorValidationCommandTemplates.evidence_template_paths_by_gate_id || {})}`,
        `operator_validation_command_template_missing_command_gate_ids=${JSON.stringify(operatorValidationCommandTemplates.missing_command_gate_ids || [])}`,
        `operator_validation_command_template_launch_evidence_collection_plan_schema=${operatorValidationCommandTemplates.launch_evidence_collection_plan_schema || ""}`,
        `operator_validation_command_template_launch_evidence_collection_step_count=${operatorValidationCommandTemplates.launch_evidence_collection_step_count ?? ""}`,
        `operator_validation_command_template_launch_evidence_collection_gate_ids=${JSON.stringify(operatorValidationCommandTemplates.launch_evidence_collection_gate_ids || [])}`,
        `operator_validation_command_template_launch_evidence_collection_record_command_ids=${JSON.stringify(operatorValidationCommandTemplates.launch_evidence_collection_record_command_ids || [])}`,
        `operator_validation_command_template_launch_evidence_collection_operator_only=${operatorValidationCommandTemplates.launch_evidence_collection_operator_only ?? ""}`,
        `operator_validation_command_template_launch_evidence_collection_required_final_field=${operatorValidationCommandTemplates.launch_evidence_collection_required_final_field || ""}`,
        `operator_validation_command_template_launch_evidence_collection_required_final_value=${operatorValidationCommandTemplates.launch_evidence_collection_required_final_value ?? ""}`,
        `operator_validation_command_template_launch_evidence_collection_final_inspection_command=${operatorValidationCommandTemplates.launch_evidence_collection_final_inspection_command || ""}`,
        `operator_validation_command_template_commands_are_operator_only=${operatorValidationCommandTemplates.commands_are_operator_only ?? ""}`,
        `operator_validation_command_template_commands_are_labeler_instructions=${operatorValidationCommandTemplates.commands_are_labeler_instructions ?? ""}`,
        `operator_validation_command_template_labelers_must_not_run_commands=${operatorValidationCommandTemplates.labelers_must_not_run_commands ?? ""}`,
        `links_expire_at_utc=${datasetQueuePayload.links_expire_at_utc || ""}`,
        `seconds_until_expiration=${datasetQueuePayload.seconds_until_expiration ?? ""}`,
        `dataset_id=${row.dataset_id || ""}`,
        `dataset_label=${row.dataset_label || ""}`,
        `recording_id=${row.recording_id || ""}`,
        `task_id=${row.task_id || ""}`,
        `workflow_kind=${row.workflow_kind || ""}`,
        `state=${row.state || ""}`,
        `zarr_use=${row.zarr_use || ""}`,
        `stage_group=${row.stage_group || ""}`,
        `run_name=${row.run_name || ""}`,
        `component_name=${row.component_name || ""}`,
        `operator_support_expected_user_personal_dataset_queue_url=${operatorSupportValue("expected_user_personal_dataset_queue_url")}`,
        `operator_support_personalized_labeler_entry_url=${operatorSupportValue("personalized_labeler_entry_url")}`,
        `operator_support_personal_dataset_queue_link_role=${operatorSupportValue("personal_dataset_queue_link_role")}`,
        `operator_support_canonical_dataset_queue_link_role=${operatorSupportValue("canonical_dataset_queue_link_role")}`,
        `operator_support_browser_label_write_target=${operatorSupportValue("browser_label_write_target")}`,
        `operator_support_browser_writes_csv_or_handoff_files=${operatorSupportValue("browser_writes_csv_or_handoff_files")}`,
        `operator_support_browser_has_direct_zarr_write_authority=${operatorSupportValue("browser_has_direct_zarr_write_authority")}`,
        `operator_support_csv_handoff_artifact_role=${operatorSupportValue("csv_handoff_artifact_role")}`,
        `labeler_start_ready=${row.labeler_start_ready}`,
        `labeler_action=${row.labeler_action || ""}`,
        `reassignment_session_safety_ok=${datasetQueuePayload.reassignment_session_safety_ok ?? reassignmentSessionSafety.ok ?? ""}`,
        `reassignment_session_safety_blocks_labeler_mutation=${datasetQueuePayload.reassignment_session_safety_blocks_labeler_mutation ?? reassignmentSessionSafety.blocks_labeler_mutation ?? ""}`,
        `reassignment_session_safety_active_session_assignment_mismatch_count=${datasetQueuePayload.reassignment_session_safety_active_session_assignment_mismatch_count ?? reassignmentSessionSafety.active_session_assignment_mismatch_count ?? ""}`,
        `reassignment_session_safety_active_session_assignment_mismatch_session_ids=${JSON.stringify(datasetQueuePayload.reassignment_session_safety_active_session_assignment_mismatch_session_ids || reassignmentSessionSafety.active_session_assignment_mismatch_session_ids || [])}`,
        `reassignment_session_safety_active_session_assignment_mismatch_recording_ids=${JSON.stringify(datasetQueuePayload.reassignment_session_safety_active_session_assignment_mismatch_recording_ids || reassignmentSessionSafety.active_session_assignment_mismatch_recording_ids || [])}`,
        `reassignment_session_safety_requires_operator_recovery=${datasetQueuePayload.reassignment_session_safety_requires_operator_recovery ?? reassignmentSessionSafety.requires_operator_recovery ?? ""}`,
        `reassignment_session_safety_operator_action=${datasetQueuePayload.reassignment_session_safety_operator_action || reassignmentSessionSafety.operator_action || ""}`,
        `browser_mutation_target_contract_met=${datasetQueuePayload.browser_mutation_target_contract_met ?? ""}`,
        `browser_mutation_target_mismatch_count=${datasetQueuePayload.browser_mutation_target_mismatch_count ?? ""}`,
        `direct_browser_start_contract_met=${datasetQueuePayload.direct_browser_start_contract_met ?? ""}`,
        `direct_browser_start_mismatch_count=${datasetQueuePayload.direct_browser_start_mismatch_count ?? ""}`,
        `single_owner_policy_contract_met=${datasetQueuePayload.single_owner_policy_contract_met ?? ""}`,
        `data_plane_write_target=${row.data_plane_write_target || ""}`,
        `authoritative_label_state=${row.authoritative_label_state || ""}`,
        `mutable_label_data_plane=${row.mutable_label_data_plane || ""}`,
        `browser_mutation_write_checklist_schema=${mutationWriteChecklist.schema || ""}`,
        `browser_mutation_write_checklist_ready=${mutationWriteChecklist.ready ?? ""}`,
        `browser_mutation_write_checklist_label_mutation_target_kind=${mutationWriteChecklist.label_mutation_target_kind || ""}`,
        `browser_mutation_write_checklist_browser_label_write_target=${mutationWriteChecklist.browser_label_write_target || ""}`,
        `browser_mutation_write_checklist_csv_handoff_artifact_role=${mutationWriteChecklist.csv_handoff_artifact_role || ""}`,
        `browser_mutation_write_checklist_csv_handoff_artifacts_are_label_write_targets=${mutationWriteChecklist.csv_handoff_artifacts_are_label_write_targets ?? ""}`,
        `browser_mutation_write_checklist_handoff_csv_artifacts_are_label_write_targets=${mutationWriteChecklist.handoff_csv_artifacts_are_label_write_targets ?? ""}`,
        `browser_mutation_write_checklist_intermediate_csv_artifacts_are_label_write_targets=${mutationWriteChecklist.intermediate_csv_artifacts_are_label_write_targets ?? ""}`,
        `labeler_route_authorization_checklist_schema=${routeAuthorizationChecklist.schema || ""}`,
        `labeler_route_authorization_checklist_ready=${routeAuthorizationChecklist.ready ?? ""}`,
        `labeler_route_authorization_expected_user_must_match_resolved_user=${routeAuthorizationChecklist.expected_user_must_match_resolved_user ?? ""}`,
        `labeler_route_authorization_expected_user_matches_resolved_user=${routeAuthorizationChecklist.expected_user_matches_resolved_user ?? ""}`,
        `labeler_route_authorization_known_assignment_store_user_required=${routeAuthorizationChecklist.known_assignment_store_user_required ?? ""}`,
        `labeler_route_authorization_known_assignment_store_user=${routeAuthorizationChecklist.known_assignment_store_user ?? ""}`,
        `labeler_route_authorization_active_assignment_required=${routeAuthorizationChecklist.active_assignment_required ?? ""}`,
        `labeler_route_authorization_active_assignment_count=${routeAuthorizationChecklist.active_assignment_count ?? ""}`,
        `labeler_route_authorization_has_active_assignment=${routeAuthorizationChecklist.has_active_assignment ?? ""}`,
        `labeler_route_authorization_single_owner_store_contract_required=${routeAuthorizationChecklist.single_owner_store_contract_required ?? ""}`,
        `labeler_route_authorization_single_owner_store_contract_present=${routeAuthorizationChecklist.single_owner_store_contract_present ?? ""}`,
        `labeler_route_authorization_single_owner_store_contract_ready=${routeAuthorizationChecklist.single_owner_store_contract_ready ?? ""}`,
        `labeler_route_authorization_single_owner_store_contract_met=${routeAuthorizationChecklist.single_owner_store_contract_met ?? ""}`,
        `labeler_route_authorization_single_owner_store_proof_ready=${routeAuthorizationChecklist.single_owner_store_proof_ready ?? ""}`,
        `labeler_route_authorization_assignment_ownership_integrity_ok=${routeAuthorizationChecklist.assignment_ownership_integrity_ok ?? ""}`,
        `labeler_route_authorization_duplicate_active_owner_count=${routeAuthorizationChecklist.duplicate_active_owner_count ?? ""}`,
        `labeler_route_authorization_browser_mutation_target_resolved_server_side=${routeAuthorizationChecklist.browser_mutation_target_resolved_server_side ?? ""}`,
        `labeler_route_authorization_labelers_mutate_assigned_training_zarrs=${routeAuthorizationChecklist.labelers_mutate_assigned_training_zarrs ?? ""}`,
        `labeler_route_authorization_labelers_mutate_intermediate_csvs=${routeAuthorizationChecklist.labelers_mutate_intermediate_csvs ?? ""}`,
        `labeler_route_authorization_task_open_requires_active_assignment=${routeAuthorizationChecklist.task_open_requires_active_assignment ?? ""}`,
        `labeler_route_authorization_task_open_requires_task_assigned_to_resolved_user=${routeAuthorizationChecklist.task_open_requires_task_assigned_to_resolved_user ?? ""}`,
        `labeler_route_authorization_task_open_requires_startable_task_state=${routeAuthorizationChecklist.task_open_requires_startable_task_state ?? ""}`,
        `labeler_route_authorization_mutation_requires_current_session=${routeAuthorizationChecklist.mutation_requires_current_session ?? ""}`,
        `labeler_route_authorization_mutation_requires_active_assignment=${routeAuthorizationChecklist.mutation_requires_active_assignment ?? ""}`,
        `labeler_route_authorization_mutation_requires_current_target_token=${routeAuthorizationChecklist.mutation_requires_current_target_token ?? ""}`,
        `labeler_route_authorization_signed_links_are_entry_hints_not_authorization=${routeAuthorizationChecklist.signed_links_are_entry_hints_not_authorization ?? ""}`,
        `labeler_route_authorization_forwarded_expected_user_links_recheck_identity=${routeAuthorizationChecklist.forwarded_expected_user_links_recheck_identity ?? ""}`,
        `labeler_route_authorization_forwarded_signed_links_recheck_runtime_operator_validation_start_gate=${routeAuthorizationChecklist.forwarded_signed_links_recheck_runtime_operator_validation_start_gate ?? ""}`,
        `label_mutation_target_kind=${row.label_mutation_target_kind || ""}`,
        `browser_label_write_target=${row.browser_label_write_target || ""}`,
        `training_zarr_mutations_are_server_owned=${row.training_zarr_mutations_are_server_owned}`,
        `handoff_artifacts_are_metadata_only=${row.handoff_artifacts_are_metadata_only}`,
        `csv_handoff_artifact_role=${row.csv_handoff_artifact_role || ""}`,
        `csv_handoff_artifacts_are_label_write_targets=${row.csv_handoff_artifacts_are_label_write_targets}`,
        `handoff_csv_artifacts_are_label_write_targets=${row.handoff_csv_artifacts_are_label_write_targets}`,
        `intermediate_csv_artifacts_are_label_write_targets=${row.intermediate_csv_artifacts_are_label_write_targets}`,
        `browser_writes_csv_or_handoff_files=${row.browser_writes_csv_or_handoff_files}`,
        `browser_writes_handoff_csv=${row.browser_writes_handoff_csv}`,
        `browser_writes_intermediate_csv=${row.browser_writes_intermediate_csv}`,
        `browser_receives_zarr_write_authority=${row.browser_receives_zarr_write_authority}`,
        `browser_has_direct_zarr_write_authority=${row.browser_has_direct_zarr_write_authority}`,
        `task_count=${row.task_count ?? ""}`,
        `open_task_count=${row.open_task_count ?? ""}`,
        `non_startable_task_count=${row.non_startable_task_count ?? ""}`,
        `recording_count=${row.recording_count ?? ""}`,
        `workflow_counts=${JSON.stringify(row.workflow_counts || {})}`,
        `preferred_labeler_entrypoint=${currentDatasetQueuePayload && currentDatasetQueuePayload.preferred_labeler_entrypoint ? currentDatasetQueuePayload.preferred_labeler_entrypoint : ""}`,
        `preferred_labeler_entry_url=${currentDatasetQueuePayload && currentDatasetQueuePayload.preferred_labeler_entry_url ? currentDatasetQueuePayload.preferred_labeler_entry_url : ""}`,
        `personalized_labeler_entrypoint=${currentDatasetQueuePayload && currentDatasetQueuePayload.personalized_labeler_entrypoint ? currentDatasetQueuePayload.personalized_labeler_entrypoint : ""}`,
        `personalized_labeler_entry_url=${currentDatasetQueuePayload && currentDatasetQueuePayload.personalized_labeler_entry_url ? currentDatasetQueuePayload.personalized_labeler_entry_url : ""}`,
        ...personalizedLaunchReadinessSupportLines(currentDatasetQueuePayload),
        `queue_first_entry_contract_schema=${currentDatasetQueuePayload && currentDatasetQueuePayload.queue_first_entry_contract ? currentDatasetQueuePayload.queue_first_entry_contract.schema || "" : ""}`,
        `queue_first_entry_contract_ready=${currentDatasetQueuePayload && currentDatasetQueuePayload.queue_first_entry_contract ? currentDatasetQueuePayload.queue_first_entry_contract.ready ?? "" : ""}`,
        `queue_first_entry_contract_preferred_labeler_entrypoint=${currentDatasetQueuePayload && currentDatasetQueuePayload.queue_first_entry_contract ? currentDatasetQueuePayload.queue_first_entry_contract.preferred_labeler_entrypoint || "" : ""}`,
        `queue_first_entry_contract_preferred_labeler_entry_url=${currentDatasetQueuePayload && currentDatasetQueuePayload.queue_first_entry_contract ? currentDatasetQueuePayload.queue_first_entry_contract.preferred_labeler_entry_url || "" : ""}`,
        `queue_first_entry_contract_personalized_labeler_entrypoint=${currentDatasetQueuePayload && currentDatasetQueuePayload.queue_first_entry_contract ? currentDatasetQueuePayload.queue_first_entry_contract.personalized_labeler_entrypoint || "" : ""}`,
        `queue_first_entry_contract_personalized_labeler_entry_url=${currentDatasetQueuePayload && currentDatasetQueuePayload.queue_first_entry_contract ? currentDatasetQueuePayload.queue_first_entry_contract.personalized_labeler_entry_url || "" : ""}`,
        `queue_first_entry_contract_personalized_entry_required=${currentDatasetQueuePayload && currentDatasetQueuePayload.queue_first_entry_contract ? currentDatasetQueuePayload.queue_first_entry_contract.personalized_entry_required ?? "" : ""}`,
        `queue_first_entry_contract_personalized_labeler_entry_url_matches_personal_dataset_queue=${currentDatasetQueuePayload && currentDatasetQueuePayload.queue_first_entry_contract ? currentDatasetQueuePayload.queue_first_entry_contract.personalized_labeler_entry_url_matches_personal_dataset_queue ?? "" : ""}`,
        `queue_first_entry_contract_preferred_labeler_entry_url_matches_personal_dataset_queue=${currentDatasetQueuePayload && currentDatasetQueuePayload.queue_first_entry_contract ? currentDatasetQueuePayload.queue_first_entry_contract.preferred_labeler_entry_url_matches_personal_dataset_queue ?? "" : ""}`,
        `queue_first_entry_contract_preferred_labeler_entry_url_is_expected_user_guarded=${currentDatasetQueuePayload && currentDatasetQueuePayload.queue_first_entry_contract ? currentDatasetQueuePayload.queue_first_entry_contract.preferred_labeler_entry_url_is_expected_user_guarded ?? "" : ""}`,
        `queue_first_entry_contract_personalized_labeler_entry_url_is_expected_user_guarded=${currentDatasetQueuePayload && currentDatasetQueuePayload.queue_first_entry_contract ? currentDatasetQueuePayload.queue_first_entry_contract.personalized_labeler_entry_url_is_expected_user_guarded ?? "" : ""}`,
        `queue_first_entry_contract_landing_ready=${currentDatasetQueuePayload && currentDatasetQueuePayload.queue_first_entry_contract ? currentDatasetQueuePayload.queue_first_entry_contract.landing_ready ?? "" : ""}`,
        `queue_first_entry_contract_labeling_home_ready=${currentDatasetQueuePayload && currentDatasetQueuePayload.queue_first_entry_contract ? currentDatasetQueuePayload.queue_first_entry_contract.labeling_home_ready ?? "" : ""}`,
        `queue_first_entry_contract_dataset_queue_ready=${currentDatasetQueuePayload && currentDatasetQueuePayload.queue_first_entry_contract ? currentDatasetQueuePayload.queue_first_entry_contract.dataset_queue_ready ?? "" : ""}`,
        `queue_first_entry_contract_personal_dataset_queue_ready=${currentDatasetQueuePayload && currentDatasetQueuePayload.queue_first_entry_contract ? currentDatasetQueuePayload.queue_first_entry_contract.personal_dataset_queue_ready ?? "" : ""}`,
        `queue_first_entry_contract_personal_work_ready=${currentDatasetQueuePayload && currentDatasetQueuePayload.queue_first_entry_contract ? currentDatasetQueuePayload.queue_first_entry_contract.personal_work_ready ?? "" : ""}`,
        `queue_first_entry_contract_queue_first_paths_ready=${currentDatasetQueuePayload && currentDatasetQueuePayload.queue_first_entry_contract ? currentDatasetQueuePayload.queue_first_entry_contract.queue_first_paths_ready ?? "" : ""}`,
        `queue_first_entry_contract_datasets_waiting_aliases_ready=${currentDatasetQueuePayload && currentDatasetQueuePayload.queue_first_entry_contract ? currentDatasetQueuePayload.queue_first_entry_contract.datasets_waiting_aliases_ready ?? "" : ""}`,
        `queue_first_entry_contract_expected_user_landing_guard=${currentDatasetQueuePayload && currentDatasetQueuePayload.queue_first_entry_contract ? currentDatasetQueuePayload.queue_first_entry_contract.expected_user_landing_guard ?? "" : ""}`,
        `queue_first_entry_contract_expected_user_queue_guard=${currentDatasetQueuePayload && currentDatasetQueuePayload.queue_first_entry_contract ? currentDatasetQueuePayload.queue_first_entry_contract.expected_user_queue_guard ?? "" : ""}`,
        `queue_first_entry_contract_expected_user_dashboard_guard=${currentDatasetQueuePayload && currentDatasetQueuePayload.queue_first_entry_contract ? currentDatasetQueuePayload.queue_first_entry_contract.expected_user_dashboard_guard ?? "" : ""}`,
        `preferred_labeler_entry_url_matches_dataset_queue=${currentDatasetQueuePayload && currentDatasetQueuePayload.preferred_labeler_entry_url_matches_dataset_queue !== undefined ? currentDatasetQueuePayload.preferred_labeler_entry_url_matches_dataset_queue : ""}`,
        `preferred_labeler_entry_url_matches_personal_dataset_queue=${currentDatasetQueuePayload && currentDatasetQueuePayload.preferred_labeler_entry_url_matches_personal_dataset_queue !== undefined ? currentDatasetQueuePayload.preferred_labeler_entry_url_matches_personal_dataset_queue : ""}`,
        `personalized_labeler_entry_url_matches_personal_dataset_queue=${currentDatasetQueuePayload && currentDatasetQueuePayload.personalized_labeler_entry_url_matches_personal_dataset_queue !== undefined ? currentDatasetQueuePayload.personalized_labeler_entry_url_matches_personal_dataset_queue : ""}`,
        `labeler_landing_link_role=${currentDatasetQueuePayload && currentDatasetQueuePayload.labeler_landing_link_role ? currentDatasetQueuePayload.labeler_landing_link_role : ""}`,
        `personal_dataset_queue_link_role=${currentDatasetQueuePayload && currentDatasetQueuePayload.personal_dataset_queue_link_role ? currentDatasetQueuePayload.personal_dataset_queue_link_role : ""}`,
        `dataset_queue_link_role=${currentDatasetQueuePayload && currentDatasetQueuePayload.dataset_queue_link_role ? currentDatasetQueuePayload.dataset_queue_link_role : ""}`,
        `canonical_dataset_queue_link_role=${currentDatasetQueuePayload && currentDatasetQueuePayload.canonical_dataset_queue_link_role ? currentDatasetQueuePayload.canonical_dataset_queue_link_role : ""}`,
        `dataset_queue_preview_url=${currentDatasetQueuePayload && currentDatasetQueuePayload.dataset_queue_preview_url ? currentDatasetQueuePayload.dataset_queue_preview_url : ""}`,
        `canonical_dataset_queue_preview_url=${currentDatasetQueuePayload && currentDatasetQueuePayload.canonical_dataset_queue_preview_url ? currentDatasetQueuePayload.canonical_dataset_queue_preview_url : ""}`,
        `personalized_dataset_queue_preview_users=${JSON.stringify(currentDatasetQueuePayload && currentDatasetQueuePayload.personalized_dataset_queue_preview_users ? currentDatasetQueuePayload.personalized_dataset_queue_preview_users : [])}`,
        `canonical_dataset_queue_preview_users=${JSON.stringify(currentDatasetQueuePayload && currentDatasetQueuePayload.canonical_dataset_queue_preview_users ? currentDatasetQueuePayload.canonical_dataset_queue_preview_users : [])}`,
        `missing_personalized_dataset_queue_preview_users=${JSON.stringify(currentDatasetQueuePayload && currentDatasetQueuePayload.missing_personalized_dataset_queue_preview_users ? currentDatasetQueuePayload.missing_personalized_dataset_queue_preview_users : [])}`,
        `all_users_have_personalized_dataset_queue_preview=${currentDatasetQueuePayload && currentDatasetQueuePayload.all_users_have_personalized_dataset_queue_preview !== undefined ? currentDatasetQueuePayload.all_users_have_personalized_dataset_queue_preview : ""}`,
        `preferred_personal_queue_match_users=${JSON.stringify(currentDatasetQueuePayload && currentDatasetQueuePayload.preferred_personal_queue_match_users ? currentDatasetQueuePayload.preferred_personal_queue_match_users : [])}`,
        `missing_preferred_personal_queue_match_users=${JSON.stringify(currentDatasetQueuePayload && currentDatasetQueuePayload.missing_preferred_personal_queue_match_users ? currentDatasetQueuePayload.missing_preferred_personal_queue_match_users : [])}`,
        `all_users_have_preferred_personal_queue_match=${currentDatasetQueuePayload && currentDatasetQueuePayload.all_users_have_preferred_personal_queue_match !== undefined ? currentDatasetQueuePayload.all_users_have_preferred_personal_queue_match : ""}`,
        `personalized_personal_queue_match_users=${JSON.stringify(currentDatasetQueuePayload && currentDatasetQueuePayload.personalized_personal_queue_match_users ? currentDatasetQueuePayload.personalized_personal_queue_match_users : [])}`,
        `missing_personalized_personal_queue_match_users=${JSON.stringify(currentDatasetQueuePayload && currentDatasetQueuePayload.missing_personalized_personal_queue_match_users ? currentDatasetQueuePayload.missing_personalized_personal_queue_match_users : [])}`,
        `all_users_have_personalized_personal_queue_match=${currentDatasetQueuePayload && currentDatasetQueuePayload.all_users_have_personalized_personal_queue_match !== undefined ? currentDatasetQueuePayload.all_users_have_personalized_personal_queue_match : ""}`,
        `dataset_queue_preferred_entrypoint_counts=${JSON.stringify(currentDatasetQueuePayload && currentDatasetQueuePayload.dataset_queue_preferred_entrypoint_counts ? currentDatasetQueuePayload.dataset_queue_preferred_entrypoint_counts : {})}`,
        `dataset_queue_link_role_counts=${JSON.stringify(currentDatasetQueuePayload && currentDatasetQueuePayload.dataset_queue_link_role_counts ? currentDatasetQueuePayload.dataset_queue_link_role_counts : {})}`,
        `dashboard_link_role=${currentDatasetQueuePayload && currentDatasetQueuePayload.dashboard_link_role ? currentDatasetQueuePayload.dashboard_link_role : ""}`,
        `identity_probe_link_role=${currentDatasetQueuePayload && currentDatasetQueuePayload.identity_probe_link_role ? currentDatasetQueuePayload.identity_probe_link_role : ""}`,
        `task_links_role=${currentDatasetQueuePayload && currentDatasetQueuePayload.task_links_role ? currentDatasetQueuePayload.task_links_role : ""}`,
        `expected_user_dataset_queue_url=${currentDatasetQueuePayload && currentDatasetQueuePayload.expected_user_dataset_queue_url ? currentDatasetQueuePayload.expected_user_dataset_queue_url : ""}`,
        `expected_user_labeling_home_url=${currentDatasetQueuePayload && currentDatasetQueuePayload.expected_user_labeling_home_url ? currentDatasetQueuePayload.expected_user_labeling_home_url : ""}`,
        `expected_user_dashboard_url=${currentDatasetQueuePayload && currentDatasetQueuePayload.expected_user_dashboard_url ? currentDatasetQueuePayload.expected_user_dashboard_url : ""}`,
        `expected_user_personal_dataset_queue_url=${currentDatasetQueuePayload && currentDatasetQueuePayload.expected_user_personal_dataset_queue_url ? currentDatasetQueuePayload.expected_user_personal_dataset_queue_url : ""}`,
        `expected_user_personal_work_url=${currentDatasetQueuePayload && currentDatasetQueuePayload.expected_user_personal_work_url ? currentDatasetQueuePayload.expected_user_personal_work_url : ""}`,
        `expected_user_work_url=${row.expected_user_work_url || ""}`,
        `direct_start_policy_enabled=${directStartPolicy.enabled}`,
        `direct_start_policy_endpoint_route_template=${directStartPolicy.endpoint_route_template || ""}`,
        `direct_start_policy_same_origin_only=${directStartPolicy.same_origin_only}`,
        `direct_start_policy_exact_route_required=${directStartPolicy.exact_route_required}`,
        `direct_start_policy_endpoint_task_segment_must_match_row_task_id=${directStartPolicy.endpoint_task_segment_must_match_row_task_id}`,
        `direct_start_policy_post_body_expected_user_required=${directStartPolicy.post_body_expected_user_required}`,
        `direct_start_policy_post_body_expected_user_field=${directStartPolicy.post_body_expected_user_field || ""}`,
        `direct_start_policy_denied_start_returns_task_open_authorization_contract=${directStartPolicy.denied_start_returns_task_open_authorization_contract}`,
        `direct_start_policy_denied_start_support_preserves_task_open_authorization_contract=${directStartPolicy.denied_start_support_preserves_task_open_authorization_contract}`,
        `direct_start_policy_denied_start_support_includes_authorization_context=${directStartPolicy.denied_start_support_includes_authorization_context}`,
        `direct_start_policy_denied_start_contract_reports_no_session_created=${directStartPolicy.denied_start_contract_reports_no_session_created}`,
        `direct_start_policy_denied_start_contract_reports_server_authorizes_open_false=${directStartPolicy.denied_start_contract_reports_server_authorizes_open_false}`,
        `direct_start_policy_startable_task_states=${JSON.stringify(directStartPolicy.startable_task_states || [])}`,
        `direct_start_policy_csv_handoff_artifact_role=${directStartPolicy.csv_handoff_artifact_role || ""}`,
        `direct_start_policy_csv_handoff_artifacts_are_label_write_targets=${directStartPolicy.csv_handoff_artifacts_are_label_write_targets}`,
        `direct_start_policy_handoff_csv_artifacts_are_label_write_targets=${directStartPolicy.handoff_csv_artifacts_are_label_write_targets}`,
        `direct_start_policy_intermediate_csv_artifacts_are_label_write_targets=${directStartPolicy.intermediate_csv_artifacts_are_label_write_targets}`,
        `direct_start_policy_browser_writes_csv_or_handoff_files=${directStartPolicy.browser_writes_csv_or_handoff_files}`,
        `direct_start_policy_browser_writes_handoff_csv=${directStartPolicy.browser_writes_handoff_csv}`,
        `direct_start_policy_browser_writes_intermediate_csv=${directStartPolicy.browser_writes_intermediate_csv}`,
        `direct_start_policy_browser_receives_zarr_write_authority=${directStartPolicy.browser_receives_zarr_write_authority}`,
        `direct_start_policy_browser_has_direct_zarr_write_authority=${directStartPolicy.browser_has_direct_zarr_write_authority}`,
        `direct_browser_start_endpoint=${row.direct_browser_start_endpoint || ""}`,
        `direct_browser_start_method=${row.direct_browser_start_method || ""}`,
        `direct_browser_start_uses_existing_task_open_api=${row.direct_browser_start_uses_existing_task_open_api ?? ""}`,
        `direct_browser_start_authorization_contract_ready=${row.direct_browser_start_authorization_contract_ready ?? ""}`,
        `direct_browser_start_expected_user_guard_required=${row.direct_browser_start_expected_user_guard_required ?? ""}`,
        `direct_browser_start_expected_user_guard_enforced_by_api=${row.direct_browser_start_expected_user_guard_enforced_by_api ?? ""}`,
        `direct_browser_start_server_rechecks_on_post=${row.direct_browser_start_server_rechecks_on_post ?? ""}`,
        `direct_browser_start_not_ready_reason=${row.direct_browser_start_not_ready_reason || ""}`,
        `direct_browser_start_not_ready_reasons=${JSON.stringify(row.direct_browser_start_not_ready_reasons || [])}`,
        `direct_browser_start_operator_action=${row.direct_browser_start_operator_action || ""}`,
        `direct_browser_start_requires_expected_user_guard=${row.direct_browser_start_requires_expected_user_guard ?? ""}`
      ].join("\\n");
    }

    function emptyQueueSupportText(payload, summary, progress, queueState) {
      const emptyState = payload.empty_state || {};
      const state = queueState || {};
      const directStartPolicy = payload.dataset_queue_direct_start_policy || {};
      const runtimeGateCliPolicy = payload.runtime_operator_validation_gate_cli_policy || {};
      const mutationWriteChecklist = payload.browser_mutation_write_checklist || {};
      const routeAuthorizationChecklist = payload.labeler_route_authorization_checklist || {};
      const reassignmentSessionSafety = payload.reassignment_session_safety || {};
      const operatorValidationStartGate = payload.operator_validation_start_gate || {};
      const operatorValidationMutationGate = payload.operator_validation_mutation_gate || {};
      const operatorValidationVisibilityPolicy = payload.operator_validation_visibility_policy || {};
      const operatorValidationCommandTemplates = payload.operator_validation_command_templates || {};
      const singleOwnerPolicy = payload.single_owner_policy || {};
      return [
        "page_context=dataset_queue_empty",
        "empty_queue=dataset_queue_no_open_work",
        `user=${payload.user || ""}`,
        `expected_user=${payload.expected_user || expectedUserGuardParam || ""}`,
        `single_owner_policy_assignment_scope=${payload.single_owner_policy_assignment_scope ?? singleOwnerPolicy.assignment_scope ?? ""}`,
        `single_owner_policy_recording_assignment_key=${payload.single_owner_policy_recording_assignment_key ?? singleOwnerPolicy.recording_assignment_key ?? ""}`,
        `single_owner_policy_one_current_assignment_row_per_recording=${payload.single_owner_policy_one_current_assignment_row_per_recording ?? singleOwnerPolicy.one_current_assignment_row_per_recording ?? ""}`,
        `single_owner_policy_one_active_owner=${payload.single_owner_policy_one_active_owner ?? singleOwnerPolicy.one_active_owner ?? ""}`,
        `single_owner_policy_multiple_labelers_per_recording_allowed=${payload.single_owner_policy_multiple_labelers_per_recording_allowed ?? singleOwnerPolicy.multiple_labelers_per_recording_allowed ?? ""}`,
        `single_owner_policy_assignment_user_match_required_for_mutation=${payload.single_owner_policy_assignment_user_match_required_for_mutation ?? singleOwnerPolicy.assignment_user_match_required_for_mutation ?? ""}`,
        `single_owner_policy_browser_mutation_requires_current_assignment_owner=${payload.single_owner_policy_browser_mutation_requires_current_assignment_owner ?? singleOwnerPolicy.browser_mutation_requires_current_assignment_owner ?? ""}`,
        `single_owner_policy_browser_mutation_target_resolved_server_side=${payload.single_owner_policy_browser_mutation_target_resolved_server_side ?? singleOwnerPolicy.browser_mutation_target_resolved_server_side ?? ""}`,
        `single_owner_policy_browser_mutation_target_source=${payload.single_owner_policy_browser_mutation_target_source ?? singleOwnerPolicy.browser_mutation_target_source ?? ""}`,
        `single_owner_policy_labelers_mutate_assigned_training_zarrs=${payload.single_owner_policy_labelers_mutate_assigned_training_zarrs ?? singleOwnerPolicy.labelers_mutate_assigned_training_zarrs ?? ""}`,
        `single_owner_policy_labelers_mutate_intermediate_csvs=${payload.single_owner_policy_labelers_mutate_intermediate_csvs ?? singleOwnerPolicy.labelers_mutate_intermediate_csvs ?? ""}`,
        `assignment_ownership_contract_store_single_owner_assignment_contract_present=${payload.assignment_ownership_contract_store_single_owner_assignment_contract_present ?? ""}`,
        `assignment_ownership_contract_store_single_owner_assignment_contract_ready=${payload.assignment_ownership_contract_store_single_owner_assignment_contract_ready ?? ""}`,
        `assignment_ownership_contract_store_single_owner_assignment_contract_met=${payload.assignment_ownership_contract_store_single_owner_assignment_contract_met ?? ""}`,
        `assignment_ownership_contract_store_single_owner_assignment_contract_schema=${payload.assignment_ownership_contract_store_single_owner_assignment_contract_schema ?? ""}`,
        `assignment_ownership_contract_browser_mutation_target_resolved_server_side=${payload.assignment_ownership_contract_browser_mutation_target_resolved_server_side ?? ""}`,
        `assignment_ownership_contract_labelers_mutate_assigned_training_zarrs=${payload.assignment_ownership_contract_labelers_mutate_assigned_training_zarrs ?? ""}`,
        `assignment_ownership_contract_labelers_mutate_intermediate_csvs=${payload.assignment_ownership_contract_labelers_mutate_intermediate_csvs ?? ""}`,
        `handoff_status=${payload.status || ""}`,
        `handoff_ready_to_send=${payload.ready_to_send ?? ""}`,
        `handoff_sendability_reasons=${JSON.stringify(payload.sendability_reasons || [])}`,
        `handoff_sendability_actions=${JSON.stringify(payload.sendability_actions || [])}`,
        `runtime_operator_validation_gate_cli_policy_preferred_require_flag=${runtimeGateCliPolicy.preferred_require_flag || ""}`,
        `runtime_operator_validation_gate_cli_policy_legacy_require_flag=${runtimeGateCliPolicy.legacy_require_flag || ""}`,
        `runtime_operator_validation_gate_cli_policy_validation_checklist_flag=${runtimeGateCliPolicy.validation_checklist_flag || ""}`,
        `runtime_operator_validation_gate_cli_policy_protects_browser_start_open=${runtimeGateCliPolicy.protects_browser_start_open ?? ""}`,
        `runtime_operator_validation_gate_cli_policy_protects_browser_mutations=${runtimeGateCliPolicy.protects_browser_mutations ?? ""}`,
        `runtime_operator_validation_gate_cli_policy_blocks_before_target_token_check=${runtimeGateCliPolicy.blocks_before_target_token_check ?? ""}`,
        `runtime_operator_validation_gate_cli_policy_blocks_before_zarr_write=${runtimeGateCliPolicy.blocks_before_zarr_write ?? ""}`,
        `runtime_operator_validation_gate_cli_policy_blocks_before_audit_event_creation=${runtimeGateCliPolicy.blocks_before_audit_event_creation ?? ""}`,
        `operator_validation_start_gate_required=${operatorValidationStartGate.required_for_browser_start ?? ""}`,
        `operator_validation_start_gate_ready=${operatorValidationStartGate.ready ?? ""}`,
        `operator_validation_start_gate_blocks_task_open=${operatorValidationStartGate.blocks_task_open ?? ""}`,
        `operator_validation_start_gate_not_ready_reason=${operatorValidationStartGate.not_ready_reason || ""}`,
        `operator_validation_mutation_gate_required=${operatorValidationMutationGate.required_for_browser_mutation ?? ""}`,
        `operator_validation_mutation_gate_ready=${operatorValidationMutationGate.ready ?? ""}`,
        `operator_validation_mutation_gate_blocks_browser_mutation=${operatorValidationMutationGate.blocks_browser_mutation ?? ""}`,
        `operator_validation_mutation_gate_not_ready_reason=${operatorValidationMutationGate.not_ready_reason || ""}`,
        `operator_validation_mutation_gate_pending_gate_ids=${JSON.stringify(operatorValidationMutationGate.operator_validation_pending_gate_ids || [])}`,
        `operator_validation_mutation_gate_required_missing_evidence_gate_ids=${JSON.stringify(operatorValidationMutationGate.operator_validation_required_missing_evidence_gate_ids || [])}`,
        `operator_validation_start_gate_operator_validation_status=${operatorValidationStartGate.operator_validation_status || ""}`,
        `operator_validation_start_gate_pending_gate_ids=${JSON.stringify(operatorValidationStartGate.operator_validation_pending_gate_ids || [])}`,
        `operator_validation_start_gate_required_missing_evidence_gate_ids=${JSON.stringify(operatorValidationStartGate.operator_validation_required_missing_evidence_gate_ids || [])}`,
        `operator_validation_status=${payload.operator_validation_status || ""}`,
        `operator_validation_source=${payload.operator_validation_source || ""}`,
        `operator_validation_gate_count=${payload.operator_validation_gate_count ?? ""}`,
        `operator_validation_pending_gate_ids=${JSON.stringify(payload.operator_validation_pending_gate_ids || [])}`,
        `operator_validation_required_missing_evidence_gate_ids=${JSON.stringify(payload.operator_validation_required_missing_evidence_gate_ids || [])}`,
        `operator_validation_gate_status_values=${JSON.stringify(operatorValidationVisibilityPolicy.operator_validation_gate_status_values || [])}`,
        `operator_validation_gate_ids=${JSON.stringify(OPERATOR_VALIDATION_GATE_IDS)}`,
        `operator_validation_gate_flat_field_suffixes=${JSON.stringify(OPERATOR_VALIDATION_GATE_FIELD_SUFFIXES)}`,
        ...operatorValidationGateSupportLines(payload),
        ...safeShareExternalLaunchEvidenceGapSupportLines(payload),
        `operator_validation_public_fields=${JSON.stringify(operatorValidationVisibilityPolicy.public_fields || [])}`,
        `operator_validation_operator_only_fields=${JSON.stringify(operatorValidationVisibilityPolicy.operator_only_fields || [])}`,
        `operator_validation_operator_action_fields=${JSON.stringify(operatorValidationVisibilityPolicy.operator_action_fields || [])}`,
        `operator_validation_operator_action_fields_are_labeler_instructions=${operatorValidationVisibilityPolicy.operator_action_fields_are_labeler_instructions ?? ""}`,
        `operator_validation_labeler_visible_payloads_may_include_operator_action_fields_for_support=${operatorValidationVisibilityPolicy.labeler_visible_payloads_may_include_operator_action_fields_for_support ?? ""}`,
        `operator_validation_labeler_visible_payloads_include_operator_only_fields=${operatorValidationVisibilityPolicy.labeler_visible_payloads_include_operator_only_fields ?? ""}`,
        `operator_validation_per_user_payloads_use_public_fields_only=${operatorValidationVisibilityPolicy.per_user_payloads_use_public_fields_only ?? ""}`,
        `operator_validation_top_level_operator_reports_may_include_operator_only_fields=${operatorValidationVisibilityPolicy.top_level_operator_reports_may_include_operator_only_fields ?? ""}`,
        `operator_validation_required_pending_gate_count=${payload.operator_validation_required_pending_gate_count ?? ""}`,
        `operator_validation_needs_review_gate_count=${payload.operator_validation_needs_review_gate_count ?? ""}`,
        `operator_validation_required_missing_evidence_gate_count=${payload.operator_validation_required_missing_evidence_gate_count ?? ""}`,
        `operator_validation_operator_action=${payload.operator_validation_operator_action || ""}`,
        `operator_validation_external_evidence_required=${payload.operator_validation_external_evidence_required ?? ""}`,
        `operator_validation_external_evidence_required_gate_ids=${JSON.stringify(payload.operator_validation_external_evidence_required_gate_ids || [])}`,
        `operator_validation_external_evidence_required_gate_count=${payload.operator_validation_external_evidence_required_gate_count ?? ""}`,
        `operator_validation_external_evidence_template_fields_by_gate_id=${JSON.stringify(payload.operator_validation_external_evidence_template_fields_by_gate_id || {})}`,
        `operator_validation_external_evidence_template_paths_by_gate_id=${JSON.stringify(payload.operator_validation_external_evidence_template_paths_by_gate_id || {})}`,
        `operator_validation_checklist_only_required_gate_ids=${JSON.stringify(payload.operator_validation_checklist_only_required_gate_ids || [])}`,
        `operator_validation_checklist_only_required_gate_count=${payload.operator_validation_checklist_only_required_gate_count ?? ""}`,
        `operator_validation_command_template_schema=${operatorValidationCommandTemplates.schema || ""}`,
        `operator_validation_command_template_command_count=${operatorValidationCommandTemplates.command_count ?? ""}`,
        `operator_validation_command_template_command_ids=${JSON.stringify(operatorValidationCommandTemplates.command_ids || [])}`,
        `operator_validation_command_template_template_backed_gate_ids=${JSON.stringify(operatorValidationCommandTemplates.template_backed_gate_ids || [])}`,
        `operator_validation_command_template_validation_checklist_gate_ids=${JSON.stringify(operatorValidationCommandTemplates.validation_checklist_gate_ids || [])}`,
        `operator_validation_command_template_apply_required_gate_ids=${JSON.stringify(operatorValidationCommandTemplates.apply_required_gate_ids || [])}`,
        `operator_validation_command_template_evidence_template_fields_by_gate_id=${JSON.stringify(operatorValidationCommandTemplates.evidence_template_fields_by_gate_id || {})}`,
        `operator_validation_command_template_evidence_template_paths_by_gate_id=${JSON.stringify(operatorValidationCommandTemplates.evidence_template_paths_by_gate_id || {})}`,
        `operator_validation_command_template_missing_command_gate_ids=${JSON.stringify(operatorValidationCommandTemplates.missing_command_gate_ids || [])}`,
        `operator_validation_command_template_launch_evidence_collection_plan_schema=${operatorValidationCommandTemplates.launch_evidence_collection_plan_schema || ""}`,
        `operator_validation_command_template_launch_evidence_collection_step_count=${operatorValidationCommandTemplates.launch_evidence_collection_step_count ?? ""}`,
        `operator_validation_command_template_launch_evidence_collection_gate_ids=${JSON.stringify(operatorValidationCommandTemplates.launch_evidence_collection_gate_ids || [])}`,
        `operator_validation_command_template_launch_evidence_collection_record_command_ids=${JSON.stringify(operatorValidationCommandTemplates.launch_evidence_collection_record_command_ids || [])}`,
        `operator_validation_command_template_launch_evidence_collection_operator_only=${operatorValidationCommandTemplates.launch_evidence_collection_operator_only ?? ""}`,
        `operator_validation_command_template_launch_evidence_collection_required_final_field=${operatorValidationCommandTemplates.launch_evidence_collection_required_final_field || ""}`,
        `operator_validation_command_template_launch_evidence_collection_required_final_value=${operatorValidationCommandTemplates.launch_evidence_collection_required_final_value ?? ""}`,
        `operator_validation_command_template_launch_evidence_collection_final_inspection_command=${operatorValidationCommandTemplates.launch_evidence_collection_final_inspection_command || ""}`,
        `operator_validation_command_template_commands_are_operator_only=${operatorValidationCommandTemplates.commands_are_operator_only ?? ""}`,
        `operator_validation_command_template_commands_are_labeler_instructions=${operatorValidationCommandTemplates.commands_are_labeler_instructions ?? ""}`,
        `operator_validation_command_template_labelers_must_not_run_commands=${operatorValidationCommandTemplates.labelers_must_not_run_commands ?? ""}`,
        `links_expire_at_utc=${payload.links_expire_at_utc || ""}`,
        `seconds_until_expiration=${payload.seconds_until_expiration ?? ""}`,
        `labeler_start_ready=${payload.labeler_start_ready}`,
        `labeler_start_status=${payload.labeler_start_status || state.code || ""}`,
        `labeler_action=${payload.labeler_action || ""}`,
        `labeler_work_completion_status=${payload.labeler_work_completion ? payload.labeler_work_completion.status || "" : ""}`,
        `labeler_work_completion_completed=${payload.labeler_work_completion ? payload.labeler_work_completion.completed ?? "" : ""}`,
        `labeler_work_completion_has_waiting_work=${payload.labeler_work_completion ? payload.labeler_work_completion.has_waiting_work ?? "" : ""}`,
        `labeler_work_completion_ready_for_more_labeling=${payload.labeler_work_completion ? payload.labeler_work_completion.ready_for_more_labeling ?? "" : ""}`,
        `labeler_work_completion_operator_action_required=${payload.labeler_work_completion ? payload.labeler_work_completion.operator_action_required ?? "" : ""}`,
        `reassignment_session_safety_ok=${payload.reassignment_session_safety_ok ?? reassignmentSessionSafety.ok ?? ""}`,
        `reassignment_session_safety_blocks_labeler_mutation=${payload.reassignment_session_safety_blocks_labeler_mutation ?? reassignmentSessionSafety.blocks_labeler_mutation ?? ""}`,
        `reassignment_session_safety_active_session_assignment_mismatch_count=${payload.reassignment_session_safety_active_session_assignment_mismatch_count ?? reassignmentSessionSafety.active_session_assignment_mismatch_count ?? ""}`,
        `reassignment_session_safety_active_session_assignment_mismatch_session_ids=${JSON.stringify(payload.reassignment_session_safety_active_session_assignment_mismatch_session_ids || reassignmentSessionSafety.active_session_assignment_mismatch_session_ids || [])}`,
        `reassignment_session_safety_active_session_assignment_mismatch_recording_ids=${JSON.stringify(payload.reassignment_session_safety_active_session_assignment_mismatch_recording_ids || reassignmentSessionSafety.active_session_assignment_mismatch_recording_ids || [])}`,
        `reassignment_session_safety_requires_operator_recovery=${payload.reassignment_session_safety_requires_operator_recovery ?? reassignmentSessionSafety.requires_operator_recovery ?? ""}`,
        `reassignment_session_safety_operator_action=${payload.reassignment_session_safety_operator_action || reassignmentSessionSafety.operator_action || ""}`,
        `direct_start_policy_enabled=${directStartPolicy.enabled}`,
        `direct_start_policy_endpoint_route_template=${directStartPolicy.endpoint_route_template || ""}`,
        `direct_start_policy_same_origin_only=${directStartPolicy.same_origin_only}`,
        `direct_start_policy_exact_route_required=${directStartPolicy.exact_route_required}`,
        `direct_start_policy_endpoint_task_segment_must_match_row_task_id=${directStartPolicy.endpoint_task_segment_must_match_row_task_id}`,
        `direct_start_policy_post_body_expected_user_required=${directStartPolicy.post_body_expected_user_required}`,
        `direct_start_policy_post_body_expected_user_field=${directStartPolicy.post_body_expected_user_field || ""}`,
        `direct_start_policy_denied_start_returns_task_open_authorization_contract=${directStartPolicy.denied_start_returns_task_open_authorization_contract}`,
        `direct_start_policy_denied_start_support_preserves_task_open_authorization_contract=${directStartPolicy.denied_start_support_preserves_task_open_authorization_contract}`,
        `direct_start_policy_denied_start_support_includes_authorization_context=${directStartPolicy.denied_start_support_includes_authorization_context}`,
        `direct_start_policy_denied_start_contract_reports_no_session_created=${directStartPolicy.denied_start_contract_reports_no_session_created}`,
        `direct_start_policy_denied_start_contract_reports_server_authorizes_open_false=${directStartPolicy.denied_start_contract_reports_server_authorizes_open_false}`,
        `direct_start_policy_startable_task_states=${JSON.stringify(directStartPolicy.startable_task_states || [])}`,
        `direct_start_policy_csv_handoff_artifact_role=${directStartPolicy.csv_handoff_artifact_role || ""}`,
        `direct_start_policy_csv_handoff_artifacts_are_label_write_targets=${directStartPolicy.csv_handoff_artifacts_are_label_write_targets}`,
        `direct_start_policy_browser_writes_csv_or_handoff_files=${directStartPolicy.browser_writes_csv_or_handoff_files}`,
        `direct_start_policy_browser_writes_handoff_csv=${directStartPolicy.browser_writes_handoff_csv}`,
        `direct_start_policy_browser_writes_intermediate_csv=${directStartPolicy.browser_writes_intermediate_csv}`,
        `direct_start_policy_browser_receives_zarr_write_authority=${directStartPolicy.browser_receives_zarr_write_authority}`,
        `direct_start_policy_browser_has_direct_zarr_write_authority=${directStartPolicy.browser_has_direct_zarr_write_authority}`,
        `browser_mutation_target_contract_met=${payload.browser_mutation_target_contract_met ?? ""}`,
        `browser_mutation_target_mismatch_count=${payload.browser_mutation_target_mismatch_count ?? ""}`,
        `direct_browser_start_contract_met=${payload.direct_browser_start_contract_met ?? ""}`,
        `direct_browser_start_mismatch_count=${payload.direct_browser_start_mismatch_count ?? ""}`,
        `single_owner_policy_contract_met=${payload.single_owner_policy_contract_met ?? ""}`,
        `browser_mutation_write_checklist_schema=${mutationWriteChecklist.schema || ""}`,
        `browser_mutation_write_checklist_ready=${mutationWriteChecklist.ready ?? ""}`,
        `browser_mutation_write_checklist_label_mutation_target_kind=${mutationWriteChecklist.label_mutation_target_kind || ""}`,
        `browser_mutation_write_checklist_browser_label_write_target=${mutationWriteChecklist.browser_label_write_target || ""}`,
        `browser_mutation_write_checklist_csv_handoff_artifact_role=${mutationWriteChecklist.csv_handoff_artifact_role || ""}`,
        `browser_mutation_write_checklist_csv_handoff_artifacts_are_label_write_targets=${mutationWriteChecklist.csv_handoff_artifacts_are_label_write_targets ?? ""}`,
        `browser_mutation_write_checklist_handoff_csv_artifacts_are_label_write_targets=${mutationWriteChecklist.handoff_csv_artifacts_are_label_write_targets ?? ""}`,
        `browser_mutation_write_checklist_intermediate_csv_artifacts_are_label_write_targets=${mutationWriteChecklist.intermediate_csv_artifacts_are_label_write_targets ?? ""}`,
        `labeler_route_authorization_checklist_schema=${routeAuthorizationChecklist.schema || ""}`,
        `labeler_route_authorization_checklist_ready=${routeAuthorizationChecklist.ready ?? ""}`,
        `labeler_route_authorization_expected_user_must_match_resolved_user=${routeAuthorizationChecklist.expected_user_must_match_resolved_user ?? ""}`,
        `labeler_route_authorization_expected_user_matches_resolved_user=${routeAuthorizationChecklist.expected_user_matches_resolved_user ?? ""}`,
        `labeler_route_authorization_known_assignment_store_user_required=${routeAuthorizationChecklist.known_assignment_store_user_required ?? ""}`,
        `labeler_route_authorization_known_assignment_store_user=${routeAuthorizationChecklist.known_assignment_store_user ?? ""}`,
        `labeler_route_authorization_active_assignment_required=${routeAuthorizationChecklist.active_assignment_required ?? ""}`,
        `labeler_route_authorization_active_assignment_count=${routeAuthorizationChecklist.active_assignment_count ?? ""}`,
        `labeler_route_authorization_has_active_assignment=${routeAuthorizationChecklist.has_active_assignment ?? ""}`,
        `labeler_route_authorization_single_owner_store_contract_required=${routeAuthorizationChecklist.single_owner_store_contract_required ?? ""}`,
        `labeler_route_authorization_single_owner_store_contract_present=${routeAuthorizationChecklist.single_owner_store_contract_present ?? ""}`,
        `labeler_route_authorization_single_owner_store_contract_ready=${routeAuthorizationChecklist.single_owner_store_contract_ready ?? ""}`,
        `labeler_route_authorization_single_owner_store_contract_met=${routeAuthorizationChecklist.single_owner_store_contract_met ?? ""}`,
        `labeler_route_authorization_single_owner_store_proof_ready=${routeAuthorizationChecklist.single_owner_store_proof_ready ?? ""}`,
        `labeler_route_authorization_assignment_ownership_integrity_ok=${routeAuthorizationChecklist.assignment_ownership_integrity_ok ?? ""}`,
        `labeler_route_authorization_duplicate_active_owner_count=${routeAuthorizationChecklist.duplicate_active_owner_count ?? ""}`,
        `labeler_route_authorization_browser_mutation_target_resolved_server_side=${routeAuthorizationChecklist.browser_mutation_target_resolved_server_side ?? ""}`,
        `labeler_route_authorization_labelers_mutate_assigned_training_zarrs=${routeAuthorizationChecklist.labelers_mutate_assigned_training_zarrs ?? ""}`,
        `labeler_route_authorization_labelers_mutate_intermediate_csvs=${routeAuthorizationChecklist.labelers_mutate_intermediate_csvs ?? ""}`,
        `labeler_route_authorization_task_open_requires_active_assignment=${routeAuthorizationChecklist.task_open_requires_active_assignment ?? ""}`,
        `labeler_route_authorization_task_open_requires_task_assigned_to_resolved_user=${routeAuthorizationChecklist.task_open_requires_task_assigned_to_resolved_user ?? ""}`,
        `labeler_route_authorization_task_open_requires_startable_task_state=${routeAuthorizationChecklist.task_open_requires_startable_task_state ?? ""}`,
        `labeler_route_authorization_mutation_requires_current_session=${routeAuthorizationChecklist.mutation_requires_current_session ?? ""}`,
        `labeler_route_authorization_mutation_requires_active_assignment=${routeAuthorizationChecklist.mutation_requires_active_assignment ?? ""}`,
        `labeler_route_authorization_mutation_requires_current_target_token=${routeAuthorizationChecklist.mutation_requires_current_target_token ?? ""}`,
        `labeler_route_authorization_signed_links_are_entry_hints_not_authorization=${routeAuthorizationChecklist.signed_links_are_entry_hints_not_authorization ?? ""}`,
        `labeler_route_authorization_forwarded_expected_user_links_recheck_identity=${routeAuthorizationChecklist.forwarded_expected_user_links_recheck_identity ?? ""}`,
        `labeler_route_authorization_forwarded_signed_links_recheck_runtime_operator_validation_start_gate=${routeAuthorizationChecklist.forwarded_signed_links_recheck_runtime_operator_validation_start_gate ?? ""}`,
        `labeler_start_message=${payload.labeler_start_message || state.message || ""}`,
        `labeler_start_operator_action=${payload.labeler_start_operator_action || state.operator_action || emptyState.operator_action || ""}`,
        `queue_state_code=${state.code || ""}`,
        `queue_state_title=${state.title || ""}`,
        `queue_state_blocks_labeler_start=${state.blocks_labeler_start}`,
        `empty_state_code=${emptyState.code || ""}`,
        `empty_state_message=${emptyState.message || ""}`,
        `operator_action=${state.operator_action || emptyState.operator_action || ""}`,
        `waiting_dataset_count=${summary.waiting_dataset_count ?? 0}`,
        `open_task_count=${summary.open_task_count ?? 0}`,
        `non_startable_task_count=${summary.non_startable_task_count ?? 0}`,
        `complete_task_count=${progress.complete_task_count ?? summary.complete_task_count ?? 0}`,
        `waiting_recording_count=${progress.waiting_recording_count ?? 0}`,
        `complete_recording_count=${progress.complete_recording_count ?? 0}`,
        `blocked_recording_count=${progress.blocked_recording_count ?? 0}`,
        `blocked_recordings_by_reason=${JSON.stringify(progress.blocked_recordings_by_reason || {})}`
      ].join("\\n");
    }

    function renderQueueState(payload, summary, progress, queueState) {
      const state = queueState || {};
      const target = document.getElementById("queue-state");
      const code = state.code || "unknown";
      const completion = payload.labeler_work_completion || {};
      const labelerStartReady = Boolean(payload.labeler_start_ready);
      const startStatus = payload.labeler_start_status || code;
      const labelerAction = payload.labeler_action || (labelerStartReady ? "open_dataset_queue" : "wait_for_work");
      const title = state.title || "Dataset queue state";
      const message = payload.labeler_start_message || state.message || "Refresh the queue or ask the operator to inspect this assignment if work was expected.";
      const action = payload.labeler_start_operator_action || state.operator_action || (payload.empty_state && payload.empty_state.operator_action) || "";
      const blocksStart = !labelerStartReady || Boolean(state.blocks_labeler_start);
      const statusCopy = labelerQueueStatusCopy(payload, summary, progress, state, completion);
      const support = emptyQueueSupportText(payload, summary, progress, state);
      target.className = "notice active";
      target.innerHTML = `<b>Queue state: ${escapeText(title)}</b>
        <p><b>${escapeText(statusCopy.heading)}</b> ${escapeText(statusCopy.message)}</p>
        <p class="muted">${escapeText(statusCopy.dataPlane)}</p>
        <p>${escapeText(message)}</p>
        <p class="muted">State code: ${escapeText(code)} - Start status: ${escapeText(startStatus)} - Labeler action: ${escapeText(labelerAction)} - Labeler start ${blocksStart ? "blocked" : "allowed"}.</p>
        ${blocksStart ? "<p><b>Do not start new labeling from this queue until the operator resolves this state.</b></p>" : ""}
        ${action ? `<p class="muted">Operator action: ${escapeText(action)}</p>` : ""}
        <details>
          <summary>What to send the operator</summary>
          <pre>${escapeText(support)}</pre>
          <button type="button" data-copy-reset="Copy queue state" onclick="copyDatasetSupport(this)">Copy queue state</button>
        </details>`;
    }

    function labelerQueueStatusCopy(payload, summary, progress, queueState, completion) {
      const state = queueState || {};
      const completionState = completion || {};
      const status = String(completionState.status || "");
      const code = String(state.code || payload.labeler_start_status || "");
      const openTaskCount = Number(summary.open_task_count ?? 0);
      const waitingDatasetCount = Number(summary.waiting_dataset_count ?? 0);
      const completeTaskCount = Number(progress.complete_task_count ?? summary.complete_task_count ?? 0);
      const totalTaskCount = Number(progress.total_task_count ?? summary.task_count ?? completeTaskCount);
      const blockedCount = Number(progress.blocked_recording_count || 0);
      const dataPlane = "Browser saves go through Palette's server-side assigned task/training-Zarr writers. CSV, HTML, JSON, handoff, and roster files are metadata only, not label-write targets.";
      if (status === "waiting" || payload.labeler_start_ready || openTaskCount > 0 || waitingDatasetCount > 0) {
        return {
          heading: "Assigned datasets waiting for browser labeling.",
          message: "These datasets belong to your current assignment and still have startable tasks waiting for completion.",
          dataPlane
        };
      }
      if (status === "complete" || code === "all_assigned_work_complete" || (totalTaskCount > 0 && completeTaskCount >= totalTaskCount && blockedCount === 0)) {
        return {
          heading: "All assigned work complete.",
          message: "There are no startable browser-labeling tasks waiting for you unless an operator reopens work or assigns another recording.",
          dataPlane
        };
      }
      if (status === "blocked" || blockedCount > 0 || state.blocks_labeler_start) {
        return {
          heading: "Assigned work is blocked.",
          message: "Your assignment exists, but no startable browser-labeling task is available until the operator resolves this queue state.",
          dataPlane
        };
      }
      if (status === "unassigned" || code === "no_active_assignments") {
        return {
          heading: "No active recording assignment.",
          message: "This page is personalized, but no active recording is currently assigned to this expected user.",
          dataPlane
        };
      }
      return {
        heading: "No startable browser-labeling task is waiting.",
        message: "If you expected work here, send the queue state to the operator instead of editing local files or CSV artifacts.",
        dataPlane
      };
    }

    async function startDatasetQueueTask(button) {
      const taskId = button && button.dataset ? String(button.dataset.taskId || "") : "";
      const openEndpoint = button && button.dataset ? String(button.dataset.openEndpoint || "") : "";
      if (!taskId || !openEndpoint) {
        showError({error: "missing_task_open_endpoint", details: "The dataset queue row did not include a task_id and direct browser-start endpoint."}, "Palette could not open that task.");
        return;
      }
      const openEndpointMatch = /^\/api\/tasks\/([^/?#]+)\/open$/.exec(openEndpoint);
      if (!openEndpointMatch) {
        showError({error: "invalid_task_open_endpoint", details: "The dataset queue direct browser-start endpoint was not an exact same-origin /api/tasks/{task_id}/open route."}, "Palette could not open that task.");
        return;
      }
      let endpointTaskId = "";
      try {
        endpointTaskId = decodeURIComponent(openEndpointMatch[1]);
      } catch (_error) {
        showError({error: "invalid_task_open_endpoint", details: "The dataset queue direct browser-start endpoint task segment was not valid percent-encoding."}, "Palette could not open that task.");
        return;
      }
      if (endpointTaskId !== taskId) {
        showError({error: "task_open_endpoint_mismatch", details: "The dataset queue direct browser-start endpoint did not match the task_id on this row."}, "Palette could not open that task.");
        return;
      }
      button.disabled = true;
      try {
        const response = await fetch(guardedPath(openEndpoint), {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({client_label: navigator.userAgent, expected_user: expectedUserGuardParam || ""})
        });
        const payload = await readApiPayload(response);
        if (!response.ok || !payload.ok || !payload.session || !payload.session.url) {
          showError(payload, "Palette could not open that task.");
          return;
        }
        window.location.href = payload.session.url;
      } catch (error) {
        showError({error: "task_open_failed", details: String(error)}, "Palette could not open that task.");
      } finally {
        button.disabled = false;
      }
    }

    function emptyQueueCopy(payload, summary, progress, queueState) {
      const emptyState = payload.empty_state || {};
      const state = queueState || {};
      if (state.title || state.message) {
        return {
          title: state.title || "No open dataset work is currently waiting for completion.",
          message: state.message || emptyState.message || "Refresh the queue or ask the operator to inspect your assignment if you expected work here."
        };
      }
      const code = String(emptyState.code || "");
      const blockedCount = Number(progress.blocked_recording_count || 0);
      const completeTaskCount = Number(progress.complete_task_count ?? summary.complete_task_count ?? 0);
      const totalTaskCount = Number(progress.total_task_count ?? summary.task_count ?? completeTaskCount);
      if (code === "all_tasks_complete" || (totalTaskCount > 0 && completeTaskCount >= totalTaskCount && blockedCount === 0)) {
        return {
          title: "All assigned dataset work is complete.",
          message: emptyState.message || "There is no open dataset work waiting for you. If more labeling is needed, the operator must reopen or assign new tasks."
        };
      }
      if (blockedCount > 0) {
        return {
          title: "Assigned recordings need operator action before more labeling.",
          message: emptyState.message || "Your assigned recordings currently have no startable queue tasks. The operator may need to generate tasks, reopen completed work, move blocked tasks to pending/in_progress, or inspect task visibility."
        };
      }
      if (code === "no_active_assignments") {
        return {
          title: "No active labeling recordings are assigned to you.",
          message: emptyState.message || "If you expected work, ask the operator to check your recording assignment and browser identity."
        };
      }
      return {
        title: "No open dataset work is currently waiting for completion.",
        message: emptyState.message || "Refresh the queue or ask the operator to inspect your assignment if you expected work here."
      };
    }

    function render(payload) {
      currentDatasetQueuePayload = payload || {};
      const errorBlock = document.getElementById("error");
      errorBlock.className = "";
      errorBlock.textContent = "";
      const expectedUserText = payload.expected_user ? ` - expected ${payload.expected_user}` : "";
      document.getElementById("user-pill").textContent = `${payload.user || "unknown"} (${payload.auth_source || "auth"})${expectedUserText}`;
      setGuardedEntryLinks();
      document.getElementById("landing-link").href = payload.expected_user_labeler_landing_url || guardedPath("/");
      document.getElementById("identity-link").href = payload.expected_user_identity_probe_url || guardedPath("/identity");
      const summary = payload.dataset_queue_summary || {};
      const progress = payload.progress_summary || {};
      const queueState = payload.dataset_queue_state || {};
      document.getElementById("dataset-count").textContent = summary.waiting_dataset_count ?? (payload.dataset_queue || payload.datasets || []).length;
      document.getElementById("task-count").textContent = summary.open_task_count ?? 0;
      document.getElementById("complete-count").textContent = progress.complete_task_count ?? summary.complete_task_count ?? 0;
      document.getElementById("blocked-count").textContent = progress.blocked_recording_count ?? 0;
      renderQueueState(payload, summary, progress, queueState);
      const reassignmentSessionSafety = payload.reassignment_session_safety || {};
      const reassignmentSafetyBlocked = Boolean(
        payload.reassignment_session_safety_blocks_labeler_mutation ||
        reassignmentSessionSafety.blocks_labeler_mutation ||
        payload.reassignment_session_safety_ok === false ||
        reassignmentSessionSafety.ok === false
      );
      const reassignmentBlock = document.getElementById("reassignment-session-safety");
      if (reassignmentSafetyBlocked) {
        const reassignmentRecordings = payload.reassignment_session_safety_active_session_assignment_mismatch_recording_ids || reassignmentSessionSafety.active_session_assignment_mismatch_recording_ids || [];
        const reassignmentSessions = payload.reassignment_session_safety_active_session_assignment_mismatch_session_ids || reassignmentSessionSafety.active_session_assignment_mismatch_session_ids || [];
        const reassignmentOperatorAction = payload.reassignment_session_safety_operator_action || reassignmentSessionSafety.operator_action || queueState.operator_action || "";
        const reassignmentSupport = [
          "page_context=dataset_queue_reassignment_session_safety",
          `user=${payload.user || ""}`,
          `expected_user=${payload.expected_user || expectedUserGuardParam || ""}`,
          `labeler_start_ready=${payload.labeler_start_ready}`,
          `labeler_start_status=${payload.labeler_start_status || queueState.code || ""}`,
          `reassignment_session_safety_ok=${payload.reassignment_session_safety_ok ?? reassignmentSessionSafety.ok ?? ""}`,
          `reassignment_session_safety_blocks_labeler_mutation=${payload.reassignment_session_safety_blocks_labeler_mutation ?? reassignmentSessionSafety.blocks_labeler_mutation ?? ""}`,
          `reassignment_session_safety_active_session_assignment_mismatch_count=${payload.reassignment_session_safety_active_session_assignment_mismatch_count ?? reassignmentSessionSafety.active_session_assignment_mismatch_count ?? ""}`,
          `reassignment_session_safety_active_session_assignment_mismatch_session_ids=${JSON.stringify(reassignmentSessions)}`,
          `reassignment_session_safety_active_session_assignment_mismatch_recording_ids=${JSON.stringify(reassignmentRecordings)}`,
          `reassignment_session_safety_requires_operator_recovery=${payload.reassignment_session_safety_requires_operator_recovery ?? reassignmentSessionSafety.requires_operator_recovery ?? ""}`,
          `reassignment_session_safety_operator_action=${reassignmentOperatorAction}`
        ].join("\\n");
        reassignmentBlock.className = "notice active";
        reassignmentBlock.innerHTML = `<b>Operator recovery required before labeling.</b>
          <p>Stale previous-owner sessions are still open for one or more assigned recordings, so this queue will not start browser labeling until the operator resolves them.</p>
          ${reassignmentRecordings.length ? `<p class="muted">Affected recordings: ${escapeText(reassignmentRecordings.join(", "))}</p>` : ""}
          ${reassignmentOperatorAction ? `<p class="muted">Operator action: ${escapeText(reassignmentOperatorAction)}</p>` : ""}
          <details>
            <summary>What to send the operator</summary>
            <pre>${escapeText(reassignmentSupport)}</pre>
            <button type="button" data-copy-reset="Copy reassignment safety" onclick="copyDatasetSupport(this)">Copy reassignment safety</button>
          </details>`;
      } else {
        reassignmentBlock.className = "notice";
        reassignmentBlock.textContent = "";
      }
      const blocked = document.getElementById("blocked-recordings");
      const blockedCount = Number(progress.blocked_recording_count || 0);
      if (blockedCount > 0) {
        const reasons = Object.entries(progress.blocked_recordings_by_reason || {}).map(([reason, count]) =>
          `${reason}: ${count}`
        ).join(", ");
        const recordings = (progress.blocked_recordings || []).join(", ");
        const blockedSupport = [
          "page_context=dataset_queue_blocked_recordings",
          `user=${payload.user || ""}`,
          `expected_user=${payload.expected_user || expectedUserGuardParam || ""}`,
          `blocked_recording_count=${blockedCount}`,
          `blocked_recordings=${recordings}`,
          `blocked_recordings_by_reason=${JSON.stringify(progress.blocked_recordings_by_reason || {})}`
        ].join("\\n");
        blocked.className = "notice active";
        blocked.innerHTML = `<b>Assigned recordings need operator action.</b>
          <p>${escapeText(blockedCount)} assigned recording${blockedCount === 1 ? "" : "s"} have no open queue task right now.${reasons ? " Reasons: " + escapeText(reasons) + "." : ""}</p>
          ${recordings ? `<p class="muted">Recordings: ${escapeText(recordings)}</p>` : ""}
          <p>Open the full work dashboard for details or send this state to the operator.</p>
          <details>
            <summary>What to send the operator</summary>
            <pre>${escapeText(blockedSupport)}</pre>
            <button type="button" data-copy-reset="Copy blocked details" onclick="copyDatasetSupport(this)">Copy blocked details</button>
          </details>`;
      } else {
        blocked.className = "notice";
        blocked.textContent = "";
      }
      const backupPolicy = payload.zarr_backup_policy || {};
      const backupSupport = [
        "page_context=dataset_queue_backup_policy",
        `user=${payload.user || ""}`,
        `expected_user=${payload.expected_user || expectedUserGuardParam || ""}`,
        `labeler_start_ready=${payload.labeler_start_ready}`,
        `labeler_action=${payload.labeler_action || ""}`,
        `backup_validation_gate=${backupPolicy.validation_gate || "mutable_zarr_backup_confirmation"}`,
        `copy_before_labeling=${backupPolicy.copy_before_labeling}`,
        `labelers_do_not_receive_backup_paths=${backupPolicy.labelers_do_not_receive_backup_paths}`,
        `rollback_owner=${backupPolicy.rollback_owner || "operator"}`
      ].join("\\n");
      const backupBlock = document.getElementById("backup-policy");
      backupBlock.className = "notice active";
      backupBlock.innerHTML = `<b>Backup and rollback are operator-owned.</b>
        <p class="muted">This queue does not expose backup paths or raw Zarr paths. Operators must satisfy the ${escapeText(backupPolicy.validation_gate || "mutable_zarr_backup_confirmation")} gate before broad launch.</p>
        <details>
          <summary>What to send the operator</summary>
          <pre>${escapeText(backupSupport)}</pre>
          <button type="button" data-copy-reset="Copy backup policy" onclick="copyDatasetSupport(this)">Copy backup policy</button>
        </details>`;
      const auditPolicy = payload.mutation_audit_policy || {};
      const auditSupport = [
        "page_context=dataset_queue_audit_policy",
        `user=${payload.user || ""}`,
        `expected_user=${payload.expected_user || expectedUserGuardParam || ""}`,
        `labeler_start_ready=${payload.labeler_start_ready}`,
        `labeler_action=${payload.labeler_action || ""}`,
        `audit_event_store=${auditPolicy.event_store || "labeling_task_events"}`,
        `server_records_events=${auditPolicy.server_records_events}`,
        `append_only=${auditPolicy.append_only}`,
        `browser_records_events_directly=${auditPolicy.browser_records_events_directly}`,
        `validation_gate=${auditPolicy.validation_gate || "disposable_zarr_mutation_smoke"}`
      ].join("\\n");
      const auditBlock = document.getElementById("audit-policy");
      auditBlock.className = "notice active";
      auditBlock.innerHTML = `<b>Mutation audit trail is server-side.</b>
        <p class="muted">Browser saves are expected to append server-side audit events in ${escapeText(auditPolicy.event_store || "labeling_task_events")}.</p>
        <details>
          <summary>What to send the operator</summary>
          <pre>${escapeText(auditSupport)}</pre>
          <button type="button" data-copy-reset="Copy audit policy" onclick="copyDatasetSupport(this)">Copy audit policy</button>
        </details>`;
      const sessionGuardPolicy = payload.session_guard_policy || {};
      const sessionGuardSupport = [
        "page_context=dataset_queue_session_guard_policy",
        `user=${payload.user || ""}`,
        `expected_user=${payload.expected_user || expectedUserGuardParam || ""}`,
        `labeler_start_ready=${payload.labeler_start_ready}`,
        `labeler_action=${payload.labeler_action || ""}`,
        `requires_current_session=${sessionGuardPolicy.requires_current_session}`,
        `requires_unexpired_session=${sessionGuardPolicy.requires_unexpired_session}`,
        `stale_tab_save_rejected=${sessionGuardPolicy.stale_tab_save_rejected}`,
        `session_closure_event_support=${sessionGuardPolicy.session_closure_event_support}`,
        `reopen_authority=${sessionGuardPolicy.reopen_authority || "operator"}`
      ].join("\\n");
      const sessionGuardBlock = document.getElementById("session-guard-policy");
      sessionGuardBlock.className = "notice active";
      sessionGuardBlock.innerHTML = `<b>Stale browser tabs are guarded.</b>
        <p class="muted">Only the current, unexpired session can save. Reassigned, completed, closed, expired, or superseded sessions are rejected before mutation.</p>
        <details>
          <summary>What to send the operator</summary>
          <pre>${escapeText(sessionGuardSupport)}</pre>
          <button type="button" data-copy-reset="Copy session guard policy" onclick="copyDatasetSupport(this)">Copy session guard policy</button>
        </details>`;
      const queue = payload.dataset_queue || payload.datasets || [];
      const target = document.getElementById("queue");
      if (!queue.length) {
        const emptyCopy = emptyQueueCopy(payload, summary, progress, queueState);
        const emptySupport = emptyQueueSupportText(payload, summary, progress, queueState);
        const dashboardUrl = payload.expected_user_dashboard_url || guardedPath("/work");
        const action = queueState.operator_action || (payload.empty_state && payload.empty_state.operator_action) || "";
        target.className = "notice active";
        target.innerHTML = `<b>${escapeText(emptyCopy.title)}</b>
          <p>${escapeText(emptyCopy.message)}</p>
          ${action ? `<p class="muted">Operator action: ${escapeText(action)}</p>` : ""}
          <p><a class="button secondary" href="${escapeText(dashboardUrl)}">Open full work dashboard</a></p>
          <details>
            <summary>What to send the operator</summary>
            <pre>${escapeText(emptySupport)}</pre>
            <button type="button" class="secondary" onclick="copyText(this, this.dataset.supportDetails || '', 'Copy queue state')" data-support-details="${escapeText(emptySupport)}">Copy queue state</button>
          </details>`;
        return;
      }
      target.className = "";
      const startableTaskStates = new Set(((payload.dataset_queue_direct_start_policy || {}).startable_task_states || []).map(String));
      const operatorValidationStartGate = payload.operator_validation_start_gate || {};
      const operatorValidationBlocksStart = operatorValidationStartGate.blocks_task_open === true || operatorValidationStartGate.blocks_task_open === "true";
      target.innerHTML = queue.map((dataset) => {
        const workUrl = dataset.expected_user_work_url || dataset.work_url || guardedPath("/work");
        const workflows = Object.entries(dataset.workflow_counts || {}).map(([workflow, count]) => `${workflow}:${count}`).join(", ");
        const datasetSupportDetails = supportDetailsText({...dataset.operator_support, ...dataset});
        const recordings = (dataset.recordings || []).map((recording) => {
          const url = recording.expected_user_work_url || recording.work_url || workUrl;
          const recordingSupportDetails = supportDetailsText({...recording.operator_support, ...recording});
          return `<span class="recording-chip">
            <a href="${escapeText(url)}">${escapeText(recording.recording_id)} - ${escapeText(recording.open_task_count || 0)} open - ${escapeText(recording.labeler_action || "open_recording")}</a>
            <button type="button" class="secondary" onclick="copyText(this, this.dataset.supportDetails || '', 'Copy recording support')" data-support-details="${escapeText(recordingSupportDetails)}">Copy recording support</button>
          </span>`;
        }).join("");
        const taskRows = (dataset.recordings || []).flatMap((recording) =>
          (recording.tasks || []).map((task) => {
            const taskUrl = task.expected_user_work_url || task.work_url || recording.expected_user_work_url || recording.work_url || workUrl;
            const taskIdText = task.task_id ? `task ${task.task_id} - ` : "";
            const priority = task.priority !== null && task.priority !== undefined && task.priority !== "" ? ` - priority ${task.priority}` : "";
            const zarrUse = task.zarr_use ? ` - ${task.zarr_use}` : "";
            const support = task.operator_support || {};
            const taskId = String(task.task_id || "");
            const supportId = [
              support.dataset_id ? `dataset ${support.dataset_id}` : "dataset unspecified",
              support.recording_id ? `recording ${support.recording_id}` : "",
              support.task_id ? `task ${support.task_id}` : "",
              support.workflow_kind ? `workflow ${support.workflow_kind}` : ""
            ].filter(Boolean).join(" - ");
            const supportDetails = supportDetailsText({...support, ...task});
            const notes = task.notes ? `<div class="task-note">${escapeText(task.notes)}</div>` : "";
            const directStartEndpoint = task.direct_browser_start_endpoint || "";
            const directStartContractReady = task.direct_browser_start_authorization_contract_ready === true || task.direct_browser_start_authorization_contract_ready === "true";
            const directStartNotReadyReason = operatorValidationBlocksStart
              ? (operatorValidationStartGate.not_ready_reason || "operator_validation_start_blocked")
              : task.direct_browser_start_not_ready_reason || ((task.direct_browser_start_authorization_contract || {}).not_ready_reason || "");
            const directStartOperatorAction = operatorValidationBlocksStart
              ? (operatorValidationStartGate.operator_action || "Complete required operator validation evidence before browser Start/Open.")
              : task.direct_browser_start_operator_action || ((task.direct_browser_start_authorization_contract || {}).operator_action || "");
            const canStartTask = !operatorValidationBlocksStart && directStartContractReady && Boolean(task.labeler_start_ready) && startableTaskStates.has(String(task.state || "")) && Boolean(taskId) && Boolean(directStartEndpoint);
            const startAction = canStartTask
              ? `<button type="button" data-task-id="${escapeText(taskId)}" data-open-endpoint="${escapeText(directStartEndpoint)}" onclick="startDatasetQueueTask(this)">Start browser task</button>`
              : `<span class="muted">${operatorValidationBlocksStart ? "Start is waiting for operator validation" : "Task is not startable from the queue"}${directStartNotReadyReason ? ": " + escapeText(directStartNotReadyReason) : ""}; ${operatorValidationBlocksStart ? "ask the operator to complete launch validation." : "use the dashboard fallback or ask the operator to reopen it."}</span>`;
            return `<div class="task-row">
              ${startAction}
              <a class="button secondary" href="${escapeText(taskUrl)}">Open dashboard fallback</a>
              <div class="task-meta">${escapeText(taskIdText)}${escapeText(recording.recording_id || "")} - ${escapeText(task.workflow_kind || "workflow")}${escapeText(priority)}${escapeText(zarrUse)} - ${escapeText(task.state || "open")} - ${escapeText(task.labeler_action || "open_task")}</div>
              <div class="task-meta">Direct start authorization contract: ${escapeText(directStartContractReady ? "ready" : "not ready")}</div>
              ${directStartNotReadyReason ? `<div class="task-meta">Direct start not-ready reason: ${escapeText(directStartNotReadyReason)}</div>` : ""}
              ${directStartOperatorAction ? `<div class="task-meta">Direct start operator action: ${escapeText(directStartOperatorAction)}</div>` : ""}
              <div class="task-meta">Title: ${escapeText(task.title || task.task_id || "Open task")}</div>
              <div class="task-meta">Support: ${escapeText(supportId)}</div>
              <button type="button" class="secondary" onclick="copyText(this, this.dataset.supportDetails || '', 'Copy task support details')" data-support-details="${escapeText(supportDetails)}">Copy task support details</button>
              ${notes}
            </div>`;
          })
        ).join("");
        return `<article class="dataset">
          <h2><a href="${escapeText(workUrl)}">${escapeText(dataset.dataset_label || dataset.dataset_id || "Unspecified dataset")}</a></h2>
          <div class="muted">${escapeText(dataset.open_task_count || 0)} startable / ${escapeText(dataset.task_count || 0)} shown tasks - ${escapeText(dataset.recording_count || 0)} recordings - action ${escapeText(dataset.labeler_action || "open_dataset")}${workflows ? " - workflows " + escapeText(workflows) : ""}</div>
          <button type="button" class="secondary" onclick="copyText(this, this.dataset.supportDetails || '', 'Copy dataset support')" data-support-details="${escapeText(datasetSupportDetails)}">Copy dataset support</button>
          <div class="recordings">${recordings}</div>
          <div class="tasks">${taskRows}</div>
        </article>`;
      }).join("");
    }

    async function load() {
      const refresh = document.getElementById("refresh");
      refresh.disabled = true;
      try {
        const params = new URLSearchParams();
        if (expectedUserGuardParam) params.set("expected_user", expectedUserGuardParam);
        if (inviteTokenParam) params.set("invite", inviteTokenParam);
        const query = params.toString();
        const response = await fetch(`/api/me/datasets${query ? "?" + query : ""}`);
        const payload = await readApiPayload(response);
        if (!response.ok || !payload.ok) {
          showError(payload, "Palette could not load your dataset queue.");
          return;
        }
        render(payload);
      } catch (error) {
        showError({error: "dataset_queue_load_failed", details: String(error)}, "Palette could not load your dataset queue.");
      } finally {
        refresh.disabled = false;
      }
    }

    document.getElementById("refresh").addEventListener("click", load);
    document.getElementById("copy-landing-link").addEventListener("click", (event) => copyLandingLink(event.currentTarget));
    setGuardedEntryLinks();
    load();
  </script>
</body>
</html>
"""
    return body.encode("utf-8")


_SESSION_OPERATOR_SUPPORT_CSS = read_labeling_asset("static/css/session_operator_support.css")
_SESSION_OPERATOR_SUPPORT_HTML = read_labeling_asset("templates/partials/session_operator_support.html")
_SESSION_OPERATOR_SUPPORT_JS = read_labeling_asset("static/js/operator_support.js")
_BROWSER_MUTATION_STATUS_JS = read_labeling_asset("static/js/browser_mutation_status.js")
_IMAGE_CANVAS_VIEWPORT_JS = read_labeling_asset("static/js/image_canvas_viewport.js")


def _session_status_banner(session: Mapping[str, object]) -> str:
    session_id = html.escape(str(session.get("session_id") or ""))
    expires_at = html.escape(str(session.get("expires_at_utc") or "unknown"))
    task_id = html.escape(str(session.get("task_id") or ""))
    recording_id = html.escape(str(session.get("recording_id") or ""))
    personal_queue_url = html.escape(_session_return_url(session, PERSONAL_DATASET_QUEUE_PATH))
    personal_work_url = html.escape(_session_return_url(session, PERSONAL_WORK_PATH))
    closed_at = str(session.get("closed_at_utc") or "").strip()
    state_text = "closed" if closed_at else "active"
    closed_bits = f" Closed at {html.escape(closed_at)}." if closed_at else ""
    return f"""
    <section style="border:1px solid #d7ded5;border-radius:18px;background:rgba(255,253,245,.82);padding:12px 14px;margin:-4px 0 18px;color:#5f6d62;box-shadow:0 10px 28px rgba(23,32,26,.08);">
      <b style="color:#17201a;">Session {state_text}</b>
      <span>Task <code>{task_id}</code> for recording <code>{recording_id}</code> expires at <code>{expires_at}</code>.{closed_bits}</span>
      <span style="display:block;margin-top:4px;">If this tab reports a superseded session, expired session, or completed task, return to <a href="{personal_queue_url}">your personalized dataset queue</a> or <a href="{personal_work_url}">your personalized work dashboard</a> and reopen the task.</span>
      <span style="display:block;margin-top:4px;font-size:.88rem;">Session <code>{session_id}</code></span>
    </section>
"""

