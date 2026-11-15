# Background Apply effects for the labeling server

- **Status:** accepted
- **Owner:** labeling/Apply work (session palette-12, for the user); last reviewed 2026-09-24
- **Builds on:** PR #194 (batch keypoint Apply), PR #202 (mask Apply effects
  speed-up and committed-write recovery), PR #180 (durable mask Apply effects)

## Problem

A labeler who applies subject-mask edits waits about two minutes before they can
continue. The pixels are durable after about 2 s. The remaining time goes to
*derived* work that runs inside the same HTTP request:

| Mask Apply (live receipts) | Rows | Pixel write | Effects (QC + tail + registry) |
|---|---:|---:|---:|
| `e0898046` | 1 | 1.9 s | 112 s |
| `a0d3cb18` | 10 | 3.0 s | 144 s |
| `10a1c0a7` | 3 | 1.8 s | 140 s |
| `6a88fda0` | 2 | 1.7 s | 149 s |

PR #202 cuts the modelled `/groups` cost of the effects from about 86 s to about
41 s. That is still a wait, and it grows with archive size. Keypoint Apply is
already fast (about 3 s for 98 rows after PR #194) and its only effect, a
registry refresh, takes about 0.6 s. It does not need this change.

## Goal

A labeler never waits on derived work. After **Apply** returns (the pixel write,
about 2 s), they can keep editing, move to another row, and **complete the
task**. Derived products follow automatically, and failures are visible and
retried rather than silent.

**Non-goals**
- Making the pixel write itself asynchronous. It is durable, fast, and what the
  labeler is actually saving.
- Changing QC, tail-derivation science, successor identity, or receipt formats.
- A general job system for the pipeline.

## What already exists

The pieces are mostly there; the gap is who runs them and when.

- **A durable record of owed work.** After the pixel write, the Apply receipt
  in SQLite (`labeling_checkpoint_apply_receipts`) stays in
  `secondary_effects_state != 'complete'` until QC, the tail refresh, and the
  registry refresh all succeed. `list_pending_subject_mask_run_effects` finds
  them per run.
- **Idempotent completion.** Retrying the same `apply_id` reruns the effects
  without rewriting pixels. PR #202 extends this to a crash between the pixel
  commit and SQLite finalization.
- **Gates.** While effects are pending, review-status changes and task
  completion return `409 pending_apply_effects`, and the browser shows
  `pending_apply_effect_count`.
- **Serialization.** Effects run under the refined-run write lock, and the tail
  refresh also takes the archive publication lock.
- **Threads.** The server is a `ThreadingHTTPServer`.

Today the only thing that runs owed effects is the HTTP request itself, or a
manual retry by the labeler.

## Design

### 1. The Apply request returns after the pixel commit

`/subject-mask/apply` does exactly what it does today up to and including
`mark_session_checkpoints_applied(require_secondary_effects=True)`. That is the
durable "effects owed" record. Instead of running QC, the tail refresh, and the
registry refresh inline, it enqueues the Apply and returns:

```json
{"applied": true, "effects": "queued", "apply_id": "...", "edit_revision_after": 7}
```

### 2. One effects worker per server

A single daemon thread, started with the server, drains owed effects:

- **Source of truth is SQLite, not memory.** The worker's queue is
  `list_pending_*_apply_effects`. The in-memory queue is only a wake-up signal.
  On startup, the worker scans for pending receipts, so a restart or crash
  resumes owed work instead of stranding it.
- **One archive at a time, in Apply order.** Effects for a run are processed in
  `applied_at_utc` order under the existing locks. A new Apply on the same run
  simply waits for the run lock, as concurrent requests do today.
- **Exactly the existing code path.** The worker calls the same function the
  same-`apply_id` retry calls today. No second implementation of the effects.

### 3. What the labeler sees

- **After Apply:** the row shows "Saved". A small status line shows
  "Updating QC and tail versions…" while effects for this task are owed.
- **On success:** the status line clears, and new successor tasks appear in the
  queue as they do today.
- **On failure:** a persistent banner: "Background update failed: <reason>.
  Retrying automatically." Retries back off (for example 30 s, 2 min, 10 min,
  then hourly). Failures are emailed to the admin (the user) through the
  existing notification email path; there is no separate admin page.
- **The browser polls** the existing state endpoint every few seconds while
  `pending_apply_effect_count > 0`. No new push channel is needed.

### 4. Task completion while effects are owed (**decision needed**)

The labeler's goal is to finish the task, not to wait for derived products.

- **Proposed:** the labeler may complete the task while effects are pending. The
  task records `completed_at` and `effects_pending`, and becomes fully complete
  when the worker finishes. **Scientific approval** (review status `approved`,
  training export, selector activation) stays gated on effects being complete
  and QC fresh, as today. The derived products are what approval certifies.
- **Alternative:** keep blocking completion until effects finish. This is
  simpler, but the labeler waits again, just later.

### 5. Coalescing repeated Applies (**decision needed**)

Several quick Applies to the same run queue several effect jobs.

- **QC** recomputes the whole run, so only the latest revision matters. The
  worker can run QC once for the newest owed revision and mark older receipts
  complete as superseded, recording which receipt covered them.
- **Tail successors** are versioned per Apply (the successor version hashes a
  proof that includes the `apply_id`), so each Apply currently yields its own
  successor version and tasks.
  - **Proposed:** keep one successor per Apply for now (no identity change).
    After PR #202 each takes about 40 s on `/groups`, and the queue drains while
    the labeler works.
  - **Later, if backlogs grow:** a versioned successor policy keyed by mask
    revision rather than Apply. That is an identity change, so it needs its own
    design.

### 6. Failure and safety

- **Fail closed, unchanged.** Pixels are applied, and derived caches, QC and
  stale flags already mark derived products stale until refreshed. Nothing
  downstream can use unrefreshed QC or tail versions.
- **Refusals stay refusals.** An effect that refuses (source changed, identity
  mismatch) is recorded as failed with its reason and is not retried blindly;
  it needs an admin action. Transient errors (I/O, locks) are retried.
- **Shutdown:** the worker stops taking new jobs and lets the current one
  finish, bounded by a timeout. Anything interrupted is resumed on the next
  start, because the receipt is still pending.

## Invariants to preserve (and the tests that pin them)

- Pixels are never rewritten by effects; retries are idempotent
  (`test_subject_mask_http_qc_failure_retries_effects_without_rewriting_pixels`).
- The same-`apply_id` retry finishes effects without a second pixel write, and
  now also after a crash before SQLite finalization
  (`test_subject_mask_http_retry_finalizes_write_committed_before_store_finalize`,
  PR #202).
- Tail refresh refuses a source that changes during publication
  (`test_source_change_during_publication_refuses_stale_snapshot`).
- Review-status changes remain blocked while effects are owed.
- Receipt formats and successor version identity are unchanged.

## Test plan

- The Apply request returns before effects run, and the receipt is pending.
- The worker completes pending effects. A server restart with a pending receipt
  completes it on startup.
- Two Applies on one run are processed in order, never concurrently.
- A transient failure is retried with backoff and surfaced in state. A refusal
  is not retried and is surfaced for admin action.
- Task completion while effects are pending (if accepted) records
  `effects_pending`. Approval stays blocked until effects complete.
- Browser: the status line appears, clears on success, and shows the failure
  banner on failure.

## Rollout

1. Land PR #202 (faster effects plus crash recovery).
2. Land the worker behind a server flag (`--background-apply-effects`), default
   off. Enable it on the preview and measure.
3. Default it on. Keep the inline path only for the flag-off case, then delete it.

## Decisions needed

1. ~~Task completion while effects are pending~~ Decided: allowed; approval,
   export, and activation stay gated on complete effects.
2. ~~Coalescing~~ Decided: QC coalesces to the latest owed revision; one tail
   successor per Apply for now.
3. ~~Who is notified of failures~~ Decided: email to the user only.

## Follow-up: row-scoped QC

The QC science is per-row: metrics, eye ellipses, eye-pair separation, and
contours each depend only on that row's mask. The whole-run recompute comes
from bookkeeping, not science:
- staleness is one run-level flag;
- contours are stored as one packed array with per-row offsets;
- the check recomputes everything a second time.

A row-scoped refresh would recompute only rows whose `row_revision` changed
since the last QC, splice their contours into the packed arrays, and check only
those rows. It needs a v2 QC policy stamp and a parity test against a full
recompute, and it keeps a full recompute at approval or export. Estimated QC
time for a 1–10 row Apply: about 12–22 s down to 1–2 s on `/groups`. Separate
change, after the worker.

## Decision log

- 2026-09-24: effect failures are emailed to the user only.
- 2026-09-24: accepted by the user: completion while effects are pending,
  QC coalescing, one tail successor per Apply.

- 2026-09-24: draft opened after live receipts showed about 2 minutes of effects
  per mask Apply. The user asked for saving to be fast or transparent so
  labelers can apply and complete tasks without waiting.
