# Manual Add-Row Propagation Design

Date anchored: 2026-07-08

Status: design. This records the target design for making a user-added
artifact row (a new bounding box, keypoints for a row that lacks them, or a
subject mask for a row that lacks one) flow safely and asynchronously through
`detect -> keypoints -> subject masks`. It is not the implemented write path.

Companion to [mutable_review_runs_contract.md](mutable_review_runs_contract.md)
(the edit-vs-observation split) and
[realtime_sparse_row_index_contract.md](realtime_sparse_row_index_contract.md)
(stable row identity + frame-index lookup). This doc is about the case those
two reserve for "future": genuinely *adding* a row, not editing an existing
one. The SLEAP comparison and resulting storage assessment are recorded in
[sleap_palette_storage_assessment.md](sleap_palette_storage_assessment.md).

## Motivation

An operator inspecting an analysis recording will sometimes need to:

1. add a bounding box where the detector produced none (or missed a second
   subject in an occupied frame);
2. add keypoints to a row that has none because pose prediction failed;
3. add a subject-mask to a row that has none because segmentation failed.

The desired end state is that the operator's edit is durable, is promoted to
the recording's training Zarr, and that keypoints / segmentations for the new
row are generated **asynchronously in the background**, producing derived runs
that reference the source — without the operator hand-orchestrating each
downstream cluster job.

## Current Reality (why this doc exists)

A code audit (2026-07-08) established the following. These are the constraints
the design must move.

### Editing an existing row works; adding a new row is the wall

Row identity is logical (`refined_row_ids`), not physical position, and every
stage writer rewrites its arrays wholesale on each save. So mid-array insertion
is never attempted and the classic "insert row i shifts every later row"
trapdoor does not occur. The problem is the opposite: there is no *append*
path either.

- **Detect.** `apply_manual_edit` ->
  `_write_dense_curated_edit_payload` -> `write_curated_refined_detect_surfaces`
  rebuilds the whole `instances/` group each save (identity is correct: new
  rows get `max(existing)+1`). But the dense edit payload is strictly one row
  per frame, `entity_id == 0`: `_load_dense_curated_edit_payload` hard-fails on
  `np.any(entity_ids != 0)`
  (`src/fisheye/tune/detect_review.py`). Adding a **second** subject to a frame
  that already holds one is not expressible through the review UI or promotion.
- **Keypoints.** Keypoint arrays are positionally row-aligned to the crop run's
  `total_rois`. A failed row already exists (box present, keypoints empty), so
  upgrading it is an in-place value write — supported by
  `patch_keypoints_from_crops.py` and `keypoint_tuner.py`. But there is no
  append/grow path: a new detect row leaves keypoints one row short and
  `keypoint_retry.py` raises
  `"Retry indices exceed crop ROI count; source/refined runs are misaligned."`
- **Subject masks.** `masks_roi` dense `(N, C, H, W)` is the write authority;
  `mask_rle/` (ragged CSR: contiguous `counts` blob + `indptr`) and
  `mask_bitpacked/` are derived caches. Editing an existing empty channel is a
  cheap in-place write (`refined_subject_mask_review.py::_apply_refined_subject_roi_rows`,
  bounds-checked against a fixed row count). There is **no add-a-row code path
  at all**: adding a row means regenerating the whole run and touching ~15
  position-aligned arrays (lineage + metrics + per-component groups) plus an
  O(n) rewrite of the RLE `counts`/`indptr` blob.

### The N+1 problem wedges the pipeline

When detect gains a row (N+1) but keypoints/masks still have N:

- the dense boolean marker `source_row_stale[roi]=True` only updates in-bounds
  rows, so it cannot represent a new appended row in an N-row downstream
  payload (`src/fisheye/shared/subject_mask_stale.py`);
- `source_update_pending_rows` and the run-level
  `source_subject_mask_stale` payload can preserve the requested row id, but
  no reconciler currently acts on that pending signal;
- the next downstream stage hits `assert_row_alignment`
  (`src/fisheye/shared/row_alignment.py`) and hard-raises on the leading-dim
  mismatch.

This is fail-closed (a crash, not silent corruption), which is good, but it
means one added box wedges the pipeline until every downstream stage is
regenerated to N+1 or the pending row is explicitly materialized. There is also
no `mark_downstream_keypoint_runs_stale` edge; detect->keypoint staleness only
propagates when a whole new detect run is emitted through the registry step
cascade.

### There is no background worker

No queue, no daemon, no dispatcher exists in the pipeline path. Downstream
generation is a human running `scripts/submit_*_bsub.sh` per stage with the
upstream run name wired in by hand. The pending signals a worker would consume
-- `source_update_pending_rows`, `source_subject_mask_stale`, registry
`status="missing"` (from `src/fisheye/registry/step_cascade.py`) -- are all
written and then read only by dashboards/diagnostics. Nothing acts on them.

### Staleness is three uncoordinated systems, none row-count aware

1. Registry step cascade (`step_cascade.py`): push, step-level, fires only when
   a *new run* is produced, not on in-place row edits.
2. Zarr stale markers (`subject_mask_stale.py`, `keypoint_stale.py`): push,
   row-level, keypoint->mask only. Attr-level pending payloads can record an
   out-of-range appended row, but dense boolean row markers cannot, and no
   actuator consumes the pending state.
3. Lineage fingerprint (`run_lineage_fingerprint.py`): nominally pull, written
   `best_effort`, **contains no `row_count` or rowset digest** and is never a
   read-time gate — so it cannot detect an added row. Consumed only by the
   read-only `audit_analysis_staleness.py` diagnostic.

### What already works

- Row identity model conforms to contract and is independently validated
  (`refined_detect_identity.py`).
- Bbox promotion analysis->training is real and clean: a tail append/upsert into
  the training Zarr (`detect_training_promotion_backend.py`), mirrored into
  `refined_detect_runs/<run>/instances`. It is detect-only; no keypoint/mask
  promotion path exists.
- Alignment is fail-closed.
- The ragged tail-append primitive needed for masks already exists and works
  for contours (`refined_subject_component_contours.py::write_component_contour_row`
  / `_append_points`) — it is simply not applied to mask rows.

## Design Principles

1. **Two layers, two rules.** Observation rowsets
   (`detect_runs`, `crop_runs`, raw model outputs) stay immutable. Review /
   assignment surfaces (`refined_detect_runs/<run>/instances`, refined keypoint
   and refined subject-mask review surfaces) are mutable and patched in place
   with an `edit_revision` counter + audit event. This is the
   `mutable_review_runs_contract` rule; adding a row obeys it too — an add is a
   mutation of the review surface, not a new run per click.

2. **Append, never mid-insert.** A new row is appended at the tail of every
   row-aligned array, receives a fresh non-reused stable id, and is located by
   a frame-index lookup rather than physical order. Derived caches (RLE,
   bitpacked, contours, metrics) are rebuilt or marked stale, never spliced.

3. **A row is not "done" until its whole column is present.** Adding a detect
   row creates a downstream hole. That hole is an explicit, queryable
   `pending_generation` state — not a silent no-op and not a crash.

4. **A reconciler actuates; it does not infer.** A background loop turns
   pending state into submitted stage jobs with explicitly resolved upstream
   run names. It never guesses inputs from `latest`.

## Target Architecture

### 1. Row-append primitive per stage

Add one guarded "extend run" writer per stage that owns the full list of
row-aligned arrays and updates them atomically (as atomically as NFS + no-WAL
allows: write-new-then-swap, never partial). It must:

- append to every lineage, metric, and per-component array by tail `resize`
  (mirror `refined_subject_component_contours::_append_points`);
- mint a fresh stable id from an allocator that never reuses a retired id
  (`refined_row_id` for detect; a new `subject_row_id` for masks;
  keypoint rows inherit the crop/detect row id);
- set the new row's downstream-relevant fields to a `pending_generation`
  sentinel rather than zero/NaN that reads as "real but empty";
- treat RLE/bitpacked/contours as fully derived — rebuild or flag stale, per
  the existing `mark_derived_mask_caches_stale_attrs` mechanism.

A partial append that touches some arrays but not others must be impossible to
commit; readers already infer `n_rows` from whichever array they find first
(`refined_subject_masks_io.py`), so a half-extended run is loadable-but-corrupt
today. One writer owning the array list closes that.

### 2. Per-row downstream state on the canonical surface

Add artifact-state columns to `refined_detect_runs/<run>/instances` (they
currently live only on the legacy dense root):

- `keypoints_state_codes` and `subject_mask_state_codes`, per row, drawn from
  `{ present, pending_generation, failed, not_applicable }`.

An appended detect row is written with both set to `pending_generation`. This
is the single source of truth the reconciler reads, and it is what makes "the
new row orphaned its downstream" a queryable fact instead of an alignment
crash.

### 3. Frame-index lookup so appended rows are findable

Implement the refined-run `frame_index/` CSR
(`frame_numbers / row_start / row_count / row_indices`) from
`realtime_sparse_row_index_contract`, rebuilt on every append, for detect and
subject-mask runs. Without it a tail-appended row is invisible to viewers and
consumers that scan by frame.

### 4. Row-append propagation contract (the N+1 fix)

Enforce one policy repo-wide. Target recommendation: **row-count change
enqueues an explicit incremental generate-for-rows-[i..j] unit**, and
downstream stages learn to tolerate a source with more rows than they have by
treating the extra rows as `pending_generation` rather than asserting equality.

The important distinction from a broad storage rewrite is that keyed sparse
lineage already exists in `row_lineage.py`: when both sources provide
`instance_key`, row-lineage comparison sorts by key instead of trusting physical
order. The missing behavior is not "ignore row-count mismatches"; it is to make
downstream payload absence explicit before payload consumers require physical
row equality.

- Relax `assert_row_alignment` at the specific boundary where the downstream
  stage is *known incomplete*: allow `len(source) >= len(downstream)` when the
  surplus source rows are all `pending_generation`, and fail as today on any
  other mismatch. Everywhere else keep the hard assert — it is the reason a bad
  edit crashes instead of corrupts.
- Add the missing `mark_downstream_keypoint_runs_stale` edge so a detect edit
  propagates to keypoints at row level, not only via a full new-run cascade.
- Ensure stale marker APIs record an appended (out-of-range) row in
  attr-level pending state even when dense boolean row markers cannot be
  extended yet.

Crimson-side implementation note: **any source row-count change forces full
downstream regen** remains the safest first runnable v1 if the incremental
surplus-row relaxation is too invasive to land with the reconciler. This
sidesteps the per-row machinery entirely at the cost of recomputing whole runs
for one added box. The important requirement is not which policy wins first;
it is that the policy is explicit, enforced, and consumed by the reconciler
instead of leaving N+1 rows wedged between stages.

### 5. Reconcile-and-dispatch loop (the "worker")

The highest-leverage missing piece. It need not be a daemon — a
cron / `/loop`-driven reconciler is consistent with the existing
subprocess-runner + completion-marker orchestration. Each pass:

1. scans registry `recording_step_status` for `status="missing"` rows and the
   Zarr per-row `pending_generation` / `source_update_pending_rows` markers;
2. joins each pending unit to the resolved upstream run name (never `latest`);
3. submits the correct stage job (`submit_keypoints_batches_bsub.sh`,
   `submit_subject_mask_batches_bsub.sh`, ...) scoped to the pending rows;
4. on stage completion, the existing completion-marker path clears the pending
   state and appends the new derived run referencing the source.

The lineage substrate to do this already exists (`source_*` refs, completion
markers, run-name resolution). Only the actuator is missing. Build the
actuator before anything else in this list — every other change here produces
pending signals that currently fall on the floor.

### 5a. Crimson-side implementation gates

Crimson can tolerate either downstream policy as long as the state is explicit
and frame-local. The UI should not infer pending rows by noticing missing
keypoints or masks. It should read explicit per-row state from the canonical
review surface and display whether an artifact is materialized, pending,
failed, stale, or rejected.

Implementation gates before this becomes reliable for interactive review:

- `frame_index/` CSR is mandatory for any run that can accept appended rows.
  Crimson should not scan an append tail or rely on physical row locality to
  discover new annotations for the displayed frame.
- Pending-generation rows need stable identities before worker output exists.
  Jobs should be keyed by stable row ids plus source run/edit revision, not by
  physical row offsets alone.
- Every dispatched job must record the exact upstream run name, row ids, and
  rowset digest it consumed. It must never resolve `latest` at execution time.
- A partially extended run must fail validation before Crimson can select it as
  active. Prefer temp-group/write-new-then-swap patterns and a post-write row
  alignment validator over piecemeal resize commits.
- Training promotion should consume reviewed materialized rows. For subject
  masks, the promoted training surface remains dense `masks_roi`; compact masks,
  contours, and metrics are regenerated derived arrays, not label authority.

### 6. Rowset digest in the lineage fingerprint

Fold `row_count` + an `instance_key` digest into
`build_run_lineage_payload`, and add a read-time gate (promote the logic in
`audit_analysis_staleness.py` from diagnostic to gate). This gives an automatic
backstop that catches row-count drift the push markers miss, and reconciles the
three staleness systems around one content-aware signal.

### 7. Extend promotion to keypoints and masks

Promotion is detect-bbox-only today. Once the append primitive and per-row
state exist, add analogous append/upsert promotion for keypoint and subject-mask
rows so a corrected row reaches the training Zarr, not only the analysis Zarr.

### 8. Unblock multi-instance add

Remove the one-row-per-frame / `entity_id == 0` restriction in
`_load_dense_curated_edit_payload` and in the promotion backend. Until this
lands, "add a bbox that doesn't currently exist" is only half-possible — a
second subject in an occupied frame is unreachable. This is orthogonal to the
worker but is the other thing that makes the headline use case fully work.

## Sequencing

This design describes the full target, but implementing keyed sparse lineage
plus explicit downstream pending rows everywhere is a cross-cutting refactor.
The safe first implementation should remove the N+1 wedge while preserving
today's complete-payload invariant.

1. Source row-count change policy: force full downstream keypoint/mask
   regeneration for any detect row-count change. This is the smallest runnable
   behavior that avoids partial materialization semantics.
2. Rowset fingerprints: add `row_count` + `instance_key` digest to source
   lineage payloads and expose the result as a read-time freshness gate.
3. Reconcile-and-dispatch loop over the pending signals that already exist
   (`status="missing"` + `source_update_pending_rows`). Its first policy can be
   full-run downstream regeneration; incremental row generation can land behind
   the same actuator later.
4. Per-row downstream state columns + row-append primitive on detect, behind
   the append-not-insert + append-safety rules.
5. Row-append propagation contract (N+1 fix) + missing detect->keypoint edge +
   `assert_row_alignment` surplus tolerance.
6. Subject-mask append primitive + `frame_index/` CSR (reuse the contour
   append pattern).
7. Keypoint/mask promotion; multi-instance add.

The explicit per-row `pending_generation` path should come after the reconciler
exists. Until then, pending state is easy to write and easy to forget.

## Non-goals

- A general append-only Zarr edit-log format (tracked by
  `mutable_review_runs_contract`).
- Changing the observation-layer immutability rule.
- Replacing the human-launchable CLI/bsub path; the reconciler wraps it, it
  does not remove it.

## Key References

- `docs/mutable_review_runs_contract.md` — edit-vs-observation split,
  `edit_revision` target.
- `docs/realtime_sparse_row_index_contract.md` — stable row id + `frame_index/`
  CSR.
- `docs/refined_detect_row_identity_contract.md` — detect row identity rules.
- `docs/keypoint_late_correction_contract.md`,
  `docs/keypoint_merged_row_gate_contract.md` — keypoint correction + row gate.
- `docs/refined_subject_masks_runs_contract.md`,
  `docs/subject_mask_runs_contract.md` — refined mask stage.
- `docs/analysis_to_training_promotion_contract.md` — implemented bbox
  promotion.
- Code anchors: `src/fisheye/shared/refined_detect_curation.py`,
  `src/fisheye/tune/detect_review.py`,
  `src/fisheye/utils/patch_keypoints_from_crops.py`,
  `src/fisheye/utils/keypoint_retry.py`,
  `src/fisheye/shared/mask_store.py`,
  `src/fisheye/tune/refined_subject_mask_review.py`,
  `src/fisheye/shared/refined_subject_component_contours.py`,
  `src/fisheye/shared/subject_mask_stale.py`,
  `src/fisheye/shared/keypoint_stale.py`,
  `src/fisheye/shared/row_alignment.py`,
  `src/fisheye/shared/run_lineage_fingerprint.py`,
  `src/fisheye/registry/step_cascade.py`,
  `src/fisheye/tune/detect_training_promotion_backend.py`.
