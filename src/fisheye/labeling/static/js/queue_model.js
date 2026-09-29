// Pure view model for the labeler queue page (no DOM), built from
// GET /api/me/queue (schema palette.labeler_queue.v1). The server decides
// whether a task may start; this module only arranges what it returned.

export const QUEUE_SCHEMA = "palette.labeler_queue.v1";

const WORKFLOW_LABELS = {
  keypoints: "Keypoints",
  subject_mask_component: "Mask",
  detect_training: "Detection",
  detect_analysis: "Detection review",
};

const STATE_LABELS = {
  pending: "Not started",
  in_progress: "In progress",
  blocked: "Blocked",
  complete: "Complete",
  superseded: "Replaced",
};

const COMPONENT_LABELS = {
  subject_body: "body",
  subject_swim_bladder: "swim bladder",
  subject_eyes: "eyes",
};

export function workflowLabel(kind) {
  return WORKFLOW_LABELS[kind] || String(kind || "Task").replaceAll("_", " ");
}

export function stateLabel(state) {
  return STATE_LABELS[state] || String(state || "");
}

function kindLabel(task) {
  const base = workflowLabel(task.workflow_kind);
  const component = task.component_name
    ? COMPONENT_LABELS[task.component_name] || task.component_name.replace(/^subject_/, "").replaceAll("_", " ")
    : "";
  return component ? `${base} · ${component}` : base;
}

function actionLabel(state) {
  if (state === "in_progress") return "Continue";
  if (state === "pending") return "Start";
  return "";
}

export function queueRows(payload) {
  const rows = [];
  for (const dataset of payload.datasets || []) {
    for (const recording of dataset.recordings || []) {
      for (const task of recording.tasks || []) {
        const start = task.start || {};
        rows.push({
          key: task.task_id,
          taskId: task.task_id,
          dataset: dataset.label || dataset.dataset_id || "",
          recording: recording.recording_id || "",
          recordingBlocked: recording.blocked_reason || "",
          title: task.title || task.task_id,
          notes: task.notes || "",
          kind: task.workflow_kind || "",
          kindLabel: kindLabel(task),
          state: task.state || "",
          stateLabel: stateLabel(task.state),
          priority: Number(task.priority) || 0,
          action: actionLabel(task.state),
          canStart: start.ready === true && Boolean(start.endpoint),
          startEndpoint: start.ready === true ? start.endpoint || "" : "",
          notReadyReason: start.ready === true ? "" : start.operator_action || start.not_ready_reason || "",
        });
      }
    }
  }
  rows.sort(
    (a, b) =>
      b.priority - a.priority ||
      a.recording.localeCompare(b.recording) ||
      a.title.localeCompare(b.title) ||
      a.taskId.localeCompare(b.taskId),
  );
  return rows;
}

export function queueFilters(rows) {
  const kinds = [...new Set(rows.map((row) => row.kind))].sort((a, b) =>
    workflowLabel(a).localeCompare(workflowLabel(b)),
  );
  return [{ id: "", label: "All" }, ...kinds.map((kind) => ({ id: kind, label: workflowLabel(kind) }))];
}

export function filterRows(rows, kind) {
  return kind ? rows.filter((row) => row.kind === kind) : rows;
}

export function queueSummary(payload, rows) {
  const open = rows.filter((row) => row.state === "pending" || row.state === "in_progress");
  const recordings = new Set(open.map((row) => row.recording)).size;
  if (!open.length) return "Nothing to label right now.";
  const tasks = open.length === 1 ? "1 open task" : `${open.length} open tasks`;
  const across = recordings === 1 ? "1 recording" : `${recordings} recordings`;
  return `${tasks} across ${across}.`;
}

// The page's own identity guards, forwarded on every API call.
export function authParams(search) {
  const source = new URLSearchParams(search || "");
  const params = new URLSearchParams();
  for (const key of ["expected_user", "invite"]) {
    const value = source.get(key);
    if (value) params.set(key, value);
  }
  return params;
}

export function withParams(path, params) {
  const [base, query = ""] = String(path).split("?", 2);
  const merged = new URLSearchParams(query);
  for (const [key, value] of params.entries()) merged.set(key, value);
  const text = merged.toString();
  return text ? `${base}?${text}` : base;
}
