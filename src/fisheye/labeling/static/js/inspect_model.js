// Pure helpers for the admin inspect page (no DOM). Payloads come from the
// read-only /api/admin/inspect/* routes.

export const STATUS_LABELS = { applied: "Applied", saved: "Saved, not applied", untouched: "No labeler edit" };

export function decodeBytes(payload) {
  const raw = atob(payload.pixels);
  const bytes = new Uint8Array(raw.length);
  for (let i = 0; i < raw.length; i++) bytes[i] = raw.charCodeAt(i);
  return { bytes, height: payload.shape[0], width: payload.shape[1], channels: payload.shape[2] || 1 };
}

// RGBA pixels for a grayscale or RGB crop.
export function imageRgba(payload) {
  const { bytes, height, width, channels } = decodeBytes(payload);
  const out = new Uint8ClampedArray(width * height * 4);
  for (let i = 0; i < width * height; i++) {
    const s = i * channels;
    out[i * 4] = bytes[s];
    out[i * 4 + 1] = channels > 1 ? bytes[s + 1] : bytes[s];
    out[i * 4 + 2] = channels > 2 ? bytes[s + 2] : bytes[s];
    out[i * 4 + 3] = 255;
  }
  return { rgba: out, width, height };
}

// A mask overlay: applied in green, saved in amber, both (agreeing) in teal.
export function maskOverlayRgba(applied, saved, view) {
  const a = applied ? decodeBytes(applied) : null;
  const s = saved ? decodeBytes(saved) : null;
  const base = a || s;
  if (!base) return null;
  const n = base.width * base.height;
  const out = new Uint8ClampedArray(n * 4);
  for (let i = 0; i < n; i++) {
    const on_a = view !== "saved" && a && a.bytes[i] > 0;
    const on_s = view !== "applied" && s && s.bytes[i] > 0;
    if (!on_a && !on_s) continue;
    const both = on_a && on_s;
    const color = both ? [0, 200, 180] : on_a ? [0, 200, 120] : [255, 176, 32];
    out[i * 4] = color[0]; out[i * 4 + 1] = color[1]; out[i * 4 + 2] = color[2]; out[i * 4 + 3] = both ? 120 : 140;
  }
  return { rgba: out, width: base.width, height: base.height };
}

// Pixels that differ between the applied and saved masks.
export function maskDifference(applied, saved) {
  if (!applied || !saved) return null;
  const a = decodeBytes(applied).bytes;
  const s = decodeBytes(saved).bytes;
  let added = 0, removed = 0;
  for (let i = 0; i < Math.min(a.length, s.length); i++) {
    const on_a = a[i] > 0, on_s = s[i] > 0;
    if (on_s && !on_a) added++;
    if (on_a && !on_s) removed++;
  }
  return { added, removed };
}

export function filterRows(rows, status) {
  return status ? rows.filter((row) => row.status === status) : rows;
}

export function taskGroups(tasks) {
  const groups = new Map();
  for (const task of tasks) {
    const key = task.assignee || "unassigned";
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(task);
  }
  return [...groups.entries()].map(([assignee, items]) => ({ assignee, tasks: items }));
}

export function taskLabel(task) {
  const kind = task.workflow_kind === "keypoints" ? "Keypoints" : "Mask · " + String(task.component_name || "").replace(/^subject_/, "").replaceAll("_", " ");
  const total = task.row_total === null || task.row_total === undefined ? "" : ` of ${task.row_total}`;
  return `${task.recording_id} · ${kind} · ${task.applied_rows} applied, ${task.saved_rows} saved${total}`;
}

export function adjacentRow(rows, roi, delta) {
  const index = rows.findIndex((row) => row.roi_idx === roi);
  const next = rows[index + delta];
  return next ? next.roi_idx : roi;
}
