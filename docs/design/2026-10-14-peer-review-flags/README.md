# Labeling: read-only peer review with frame flags

- **Status:** accepted (scope decided by the user on 2026-10-14)
- **Owner:** labeling web app (session for the user); last reviewed 2026-10-14
- **Requested by:** the lab manager, so labelers can look at each other's applied work and flag frames for each other
- **Builds on:** `/admin/inspect` (merged 2026-10-02): `fisheye.labeling.admin_inspect` and `web_admin_inspect.py`

## Decisions

| Question | Decision |
|---|---|
| Who can see whose work | Every active labeling user can see every other labeler's tasks. No reviewer pairs. |
| What a flag means | An advisory note. It never changes a label, a checkpoint, an Apply, or a publication, and it blocks nothing. |
| Which work is visible | Applied work only. Saved but unapplied edits are never sent to a peer. |
| Task kinds | Mask (`subject_mask_component`) and keypoint tasks, the two that inspect already reads. |

## What a peer sees

A row is visible to peers when it has **at least one applied checkpoint**. The admin inspect status uses the
*latest* checkpoint, so a row the owner has applied and then edited again reads as `saved` there. For peers,
that row stays visible and shows the applied label from the training Zarr, which is still the applied value. The
pending edit is not shown.

Rows with no applied checkpoint (untouched, or saved only) are not listed and the row endpoint refuses them, so
model output and in-progress edits are not presented as someone's work.

The labels shown come from the same read-only path as admin inspect: the store through a `mode=ro` SQLite
connection, and the Zarr opened with `mode="r"`. No session, checkpoint, or task event is created by viewing.

## Routes

| Route | Purpose |
|---|---|
| `GET /peer` | Page: tasks by labeler, then one task's applied rows, then one row with image, applied label, and flags |
| `GET /api/peer/tasks` | Mask and keypoint tasks with owner, applied-row count, and open-flag count |
| `GET /api/peer/task?task_id=` | The task's peer-visible rows (applied at least once), each with its open-flag count |
| `GET /api/peer/row?task_id=&roi_idx=` | Image and applied label for one peer-visible row, plus its flags |
| `POST /api/peer/flags` | Add a flag: `task_id`, `roi_idx`, `note` |
| `POST /api/peer/flags/<flag_id>/resolve` | Resolve a flag. Only the task owner, the flag's author, or an admin may resolve. |

Authorization for all of them is "an active labeling user", the same `_known_labeler_status` check the personal
pages use. Inactive and unknown users get 403 and anonymous requests get 401. The POST routes use the app's
existing same-origin CSRF check. The page and APIs have no route that reaches the editors, the Apply path, or a
Zarr write.

## Flag storage

A new table in the labeling store, created additively with `CREATE TABLE IF NOT EXISTS` (no existing table is
rebuilt):

```sql
CREATE TABLE IF NOT EXISTS labeling_frame_flags (
    flag_id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL REFERENCES labeling_tasks(task_id) ON DELETE CASCADE,
    recording_id TEXT NOT NULL,
    roi_idx INTEGER NOT NULL,
    flagged_by TEXT NOT NULL,
    note TEXT NOT NULL,
    applied_checkpoint_id TEXT,
    apply_id TEXT,
    state TEXT NOT NULL DEFAULT 'open' CHECK (state IN ('open', 'resolved')),
    created_at_utc TEXT NOT NULL,
    resolved_by TEXT,
    resolved_at_utc TEXT
);
```

`applied_checkpoint_id` and `apply_id` record which applied version the flagger was looking at, so the owner can
tell whether a flag predates a later correction. Notes are limited to 1,000 characters. Flags are not part of
any receipt, digest, training export, or publication.

## Where the owner sees flags

- `/queue`: a task row shows "N frames flagged" when it has open flags.
- `/peer`: the owner sees flags on their own tasks there, and resolves them there.

Showing flags inside the mask and keypoint editors is a later step, not part of this change.

## Tests

- Peer APIs return only rows with an applied checkpoint, and never a saved payload. This includes a row that
  was applied and then saved again.
- A peer request creates no session, checkpoint, or task event, and opens no Zarr for writing.
- Unknown, inactive, and anonymous users are refused on every peer route.
- Flags: add, list, and resolve. Resolve is refused for a user who is not the owner, the author, or an admin.
- A flag changes no label row, checkpoint, receipt, or Apply state.
- The `/queue` payload carries the open-flag count.

## Decision log

- 2026-10-14: built; the notes below record where the build differs from, or adds to, the plan above.


- **Applied history comes from the apply receipts.** Saving a row again resets its checkpoint to `active` and
  clears `apply_id` and `applied_at_utc`, so the current checkpoint cannot show that a row was applied. The task
  and row views also read the applied receipts' `checkpoints_json` (SQLite `json_each`). That is 0.13 s for one
  task on a copy of the live store.
- **The task list counts differently.** Scanning every receipt took 2.1 s on that copy, so the task list's
  `applied_rows` counts rows whose latest checkpoint is applied (the admin inspect rule, one cheap query). A row
  applied and then edited again appears in the task view but not in that count.
- **`SCHEMA_VERSION` stays 9.** The table and its index (`idx_labeling_frame_flags_task`) are additive and older
  code ignores them. A version bump would make servers on older commits refuse a shared store.
- **Admins may also use the peer routes**, even without a labeling user row.
- **Same-origin checks run inside the two POST routes.** Flask handles claimed routes before the stdlib
  handler's same-origin check, so these routes call `request_has_same_origin` themselves. The existing Flask
  POST routes `/api/admin/admin-reviews` and `/api/admin/keypoint-corrections` do not, which is a separate,
  pre-existing gap.
- **`/queue` additions:** a "Review others' work" link to `/peer`; `links.peer_review`; per-task
  `open_flag_count`; and a "N frames flagged by reviewers" link to `/peer?task_id=...`.
