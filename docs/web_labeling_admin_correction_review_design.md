# Web labeling admin correction review design

Date: 2026-07-07
Status: design, not yet implemented

## Goal

Palette needs an admin review layer for completed web-labeling work. Labeler
completion should mean "the labeler says this task is done"; it should not mean
"the work is accepted for training without operator inspection."

The admin review layer should let an admin inspect completed labeler work, edit
keypoints when needed, record those edits as correction examples, and then mark
the work as accepted, accepted-with-corrections, needs-revision, or rejected.

This design is initially scoped to RedScare keypoint review, but the same state
model should later apply to masks and detection boxes.

## Non-goals for v1

- Do not replace the existing keypoint editor.
- Do not introduce a second canonical keypoint surface.
- Do not overwrite audit history to make admin edits look like labeler edits.
- Do not require webKnossos or another external annotation tool.
- Do not implement automatic grading or labeler scoring as the first slice.

## State model

Keep task completion and admin acceptance separate.

```text
task_state = pending | in_progress | complete | canceled
admin_review_state = pending | accepted_as_is | accepted_with_corrections | needs_revision | rejected
```

`task_state=complete` means the labeler completed their assigned work.

`admin_review_state=accepted_as_is` means an admin inspected the completed task
and accepted it without changing labels.

`admin_review_state=accepted_with_corrections` means an admin inspected the task,
changed one or more labels, and accepted the corrected result.

`admin_review_state=needs_revision` means the task should go back to a labeler
with feedback.

`admin_review_state=rejected` means the task result should not be used as-is and
needs operator follow-up.

## Data plane and audit plane

The canonical label data plane remains the assigned training Zarr review run.
Admin corrections that update keypoints should mutate the same server-owned,
task-scoped training Zarr target used by labeler saves.

The audit/education plane lives in the labeling SQLite sidecar. It records what
changed, who changed it, why it changed, and whether the correction is useful as
an onboarding example.

This split is intentional:

```text
canonical training label = current keypoint value in Zarr
correction history       = before/after rows in SQLite
```

Canonical consumers use the Zarr value. Admin dashboards and onboarding tools use
the SQLite correction history.

## Labeler snapshot semantics

When an admin opens a completed task for review, the system needs a stable
"labeler submitted" baseline. Admin edits are compared against that baseline.

For v1, create or materialize the baseline at the admin-review boundary:

```text
labeler_baseline_xy = current canonical keypoints at first admin review open
```

This is safe because the task is already complete. The baseline should be stored
in SQLite or a compact sidecar payload so it remains stable even if the admin
later moves points.

If a completed task has existing admin corrections, reopen should show the same
stored baseline plus the current corrected Zarr values.

## Proposed SQLite tables

### `labeling_admin_reviews`

One row per task-level admin review decision.

```text
review_id TEXT PRIMARY KEY
created_at_utc TEXT NOT NULL
updated_at_utc TEXT NOT NULL
task_id TEXT NOT NULL
recording_id TEXT NOT NULL
dataset_id TEXT
workflow_kind TEXT NOT NULL
run_name TEXT
labeler_user TEXT NOT NULL
admin_user TEXT NOT NULL
state TEXT NOT NULL
summary_json TEXT
notes TEXT
```

Recommended `state` values:

```text
pending
accepted_as_is
accepted_with_corrections
needs_revision
rejected
```

`summary_json` can include aggregate counts such as changed rows, changed
keypoints, mean correction distance, max correction distance, and reason-code
counts.

### `labeling_keypoint_corrections`

One row per corrected keypoint instance.

```text
correction_id TEXT PRIMARY KEY
review_id TEXT NOT NULL
task_id TEXT NOT NULL
recording_id TEXT NOT NULL
dataset_id TEXT
run_name TEXT NOT NULL
row_index INTEGER NOT NULL
keypoint_index INTEGER NOT NULL
keypoint_name TEXT
labeler_user TEXT NOT NULL
admin_user TEXT NOT NULL
before_x REAL
before_y REAL
after_x REAL
after_y REAL
delta_px REAL
reason_code TEXT
admin_note TEXT
training_example INTEGER NOT NULL DEFAULT 0
created_at_utc TEXT NOT NULL
```

The before coordinate is the stored labeler baseline. The after coordinate is the
admin-corrected canonical value.

### Optional `labeling_keypoint_review_baselines`

If baseline payloads become large or need independent versioning, use a separate
baseline table.

```text
baseline_id TEXT PRIMARY KEY
task_id TEXT NOT NULL
recording_id TEXT NOT NULL
dataset_id TEXT
run_name TEXT NOT NULL
labeler_user TEXT NOT NULL
created_by_admin_user TEXT NOT NULL
created_at_utc TEXT NOT NULL
baseline_payload_json TEXT NOT NULL
```

For 200-row RedScare keypoint tasks, a JSON payload is acceptable for v1. If this
scales to much larger tasks, move to a compact binary sidecar or per-row table.

## Admin UI surfaces

### `/admin/completed-work`

Read-only queue of completed tasks awaiting admin review.

Columns:

```text
task_id
recording_id
assignee / labeler
workflow_kind
run_name
task_completed_at
save_count
issue_count
admin_review_state
open_admin_review_link
accept_as_is_action
needs_revision_action
```

Default filter:

```text
task_state=complete AND admin_review_state in (missing, pending)
```

Useful additional filters:

```text
labeler
workflow_kind
recording_id
completed date
has issue flags
has admin corrections
admin_review_state
```

### Admin keypoint review editor

Reuse the existing keypoint editor in an admin-review mode.

Admin-review mode should show:

```text
admin mode badge
labeler submitted points as ghost/original markers
current editable points as normal markers
arrows from labeler baseline position to admin corrected position
changed-point list for the current ROI
reason-code selector for changed points
optional admin note
training-example checkbox
accept / needs revision / reject actions
```

Useful toggles:

```text
show labeler baseline points
show correction arrows
show only changed points
show labels
show skeleton
```

Point rendering:

```text
labeler baseline point = faint / hollow marker
admin current point    = solid editable marker
correction arrow       = from before_xy to after_xy
```

### `/admin/correction-examples`

Gallery of curated correction examples for onboarding and feedback.

Each example should show:

```text
ROI image
labeler baseline point
admin corrected point
arrow from before to after
keypoint name
reason_code
admin note
recording_id optional
labeler optional or anonymized
```

This page should start as admin-only. Later, selected examples can be exported or
shown to labelers as training material.

## Reason codes

Suggested v1 keypoint correction reason codes:

```text
wrong_landmark
left_right_swap
tail_midpoint_error
fin_insertion_error
fin_tip_error
off_body
low_confidence_guess
missed_visible_point
ambiguous_frame
other
```

Reason codes are optional for saving a correction but should be encouraged when
marking a correction as a training example.

## Metrics

Per task:

```text
rows_corrected
keypoints_corrected
mean_delta_px
median_delta_px
max_delta_px
corrections_by_keypoint_name
corrections_by_reason_code
training_example_count
```

Per labeler:

```text
tasks_completed
tasks_admin_reviewed
tasks_accepted_as_is
tasks_accepted_with_corrections
tasks_needing_revision
total_keypoint_corrections
mean_corrections_per_task
mean_delta_px
corrections_by_keypoint_name
corrections_by_reason_code
```

Use these metrics carefully. They are quality-control and onboarding aids, not a
standalone performance score. Some recordings are harder than others.

## Save/apply behavior

Admin edits should use the same server-owned mutation boundary as labeler edits:

```text
same active assignment / admin authorization checks
same task-scoped training Zarr target resolution
same edit_revision validation where applicable
same audit-event append behavior
```

Additional admin correction behavior:

1. Ensure a labeler baseline exists before the first admin edit.
2. Save the admin-corrected keypoints to the canonical Zarr review run.
3. Compare labeler baseline vs corrected current values.
4. Upsert one correction row per changed keypoint.
5. Update the admin review summary.
6. Do not mark the task accepted until the admin explicitly chooses an admin
   decision.

## Revision and idempotency

Admin correction save should be idempotent per `(task_id, row_index,
keypoint_index, baseline_revision, admin_revision)` where possible.

If the admin moves the same keypoint repeatedly before accepting, v1 can either:

```text
store latest correction only
```

or:

```text
append every admin correction movement as event history, plus maintain latest summary
```

Preferred v1: maintain latest correction rows for review metrics, and append
normal audit events for each save. This keeps the correction table useful without
turning it into high-frequency mouse-move history.

## Needs-revision flow

When admin marks a task `needs_revision`:

```text
admin_review_state=needs_revision
admin note required
optional correction examples flagged
original task may be reopened or a follow-up task may be created
```

V1 can start with reopening the same task:

```text
task_state=pending
admin_review_state=needs_revision
```

Later, a follow-up task model may be cleaner:

```text
parent_task_id
followup_task_id
reason=admin_needs_revision
```

## Phased implementation checklist

### Phase 1: read-only completed-work queue

- Add admin completed-work payload builder.
- Add `/admin/completed-work` HTML page or admin dashboard section.
- List completed tasks with missing/pending admin review.
- Add links to open the existing editor in admin-review mode.
- No mutation routes yet.

### Phase 2: admin-review mode baseline overlay

- Add server endpoint to create/read labeler baseline for a completed task.
- Add editor admin-review mode flag.
- Render labeler baseline points as ghost markers.
- Render current points as editable markers.
- Draw before/after arrows when points differ.
- Add toggles for baseline/arrows/changed-only.

### Phase 3: correction save

- Allow admin to edit completed task keypoints.
- Save admin-corrected points to the canonical training Zarr review run.
- Record correction rows in SQLite.
- Record admin correction audit events.
- Keep task complete; do not implicitly accept.

### Phase 4: admin decisions

- Add `accepted_as_is`, `accepted_with_corrections`, `needs_revision`, and
  `rejected` actions.
- Store task-level admin review state.
- Require notes for `needs_revision` and `rejected`.
- Summarize correction counts in admin completed-work queue.

### Phase 5: correction examples gallery

- Add `training_example` checkbox to correction rows.
- Add `/admin/correction-examples` page.
- Show before/after image overlays with arrows.
- Add filters by keypoint, reason code, labeler, and recording.

### Phase 6: labeler feedback and onboarding

- Build per-labeler correction summaries.
- Export curated examples as onboarding material.
- Optionally show selected examples to labelers before they start new work.
- Avoid exposing private per-labeler metrics broadly until policy is clear.

## Open questions

- Should admin acceptance write to Zarr attrs immediately, or first stay in the
  SQLite sidecar until the flow is proven?
- Should admin-corrected tasks stay assigned to the original labeler, or should
  admin corrections create an operator-owned session/task?
- Should `needs_revision` reopen the original task or create a follow-up task?
- What correction distance threshold should count as a meaningful correction
  rather than harmless jitter?
- Should correction examples show labeler identity by default, or anonymize for
  onboarding exports?
