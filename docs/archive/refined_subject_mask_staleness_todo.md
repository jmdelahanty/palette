# Refined Subject-Mask Staleness TODO

<!-- todo-meta
status: active
last_updated: 2026-04-06
-->

## Goal

Make refined subject-mask staleness a first-class, repo-native concept that:

- preserves the current row-level repair workflow for swim bladder and future
  subject-mask components
- exposes stale state through the same canonical surfaces the rest of the repo
  uses for review, registry queries, and lifecycle gating
- keeps source-drift semantics separate from ordinary review-state semantics

This is a design and integration TODO, not a claim that the current
swim-bladder flow is unusable. The current flow is operationally helpful and
should be preserved where it is already solving real operator pain.

## Current Practical Workflow

Today the refined subject-mask path can already do the following for
`swim_bladder`:

- refresh selected source rows in `subject_mask_runs/<run>`
- check selected refined rows against the current source rows
- auto-sync untouched refined rows
- preserve manually overridden refined rows and mark them stale
- open a stale-only batch-review queue for the affected ROIs

This is practically good because it avoids forcing whole-run regeneration and
it avoids silently overwriting curated refined rows.

## Why This Needs Follow-Up

The repository already has a stronger stale/invalidation vocabulary elsewhere:

- runtime cascade invalidation for new upstream runs marks downstream steps
  `missing` in the registry
- eye-mask stale-after-keypoint-correction is stored as an explicit artifact
  payload with lineage, reason, timestamp, ROI/frame indices, and explicit
  resolution
- registry lifecycle derivation already knows how to surface
  `source_subject_mask_stale_*` when such a payload exists

The current refined subject-mask stale flow does not yet fully align with that
model.

## Findings

### 1. The current stale queue is operationally real but registry-invisible

The new flow stores row-local stale state in component attrs such as:

- `manual_override`
- `source_row_stale`
- `source_update_pending_rows`

inside `refined_subject_masks_runs/<run>/components/<component>`.

That is sufficient for local stale review, but not for registry-backed
selection. As a result:

- the swim-bladder batch reviewer has to bypass the registry for `--status stale`
- standard subject-mask registry rows do not currently expose this stale state

The repo already has a canonical top-level subject-mask stale surface available
to the registry:

- `run.attrs["source_subject_mask_stale"]`

but the current swim-bladder stale workflow does not write it yet.

### 2. Stale is currently being collapsed into ordinary review state

When source rows change and a refined row must be preserved, the current flow:

- marks the row stale locally
- then sets the component review state to `needs_review`

This is pragmatic, but it loses semantics. The repo already distinguishes:

- stale source lineage
- ordinary in-progress review

Those are not the same thing. A run that was previously approved and then
became stale because its source changed is different from a run that was never
reviewed or is simply still under active cleanup.

### 3. Swim-bladder stale provenance is weaker than the eye-mask precedent

The eye-mask stale contract keeps a structured stale payload with:

- state
- timestamp
- reason
- source lineage
- ROI/frame indices
- later resolution metadata

The current refined subject-mask stale workflow keeps:

- row fingerprints
- row flags
- a pending-row list

and partly encodes stale reason in review notes. That is weaker and easier to
overwrite during later review-state changes.

### 4. `manual_override` is not yet a clean provenance bit

The current preserve-versus-auto-sync decision uses `manual_override`, but that
bit is not yet independent enough.

Today it is effectively derived from:

- `edit_applied`
- or current mask-vs-source difference

This is useful as a bootstrap heuristic, but it is not the same as:

- "a human intentionally curated this row and it should be preserved when the
  source changes"

If source-update behavior is going to be relied on long-term, the repo should
have a true curated/preserve bit rather than only a difference-from-source bit.

### 5. The row-level queue is good and should stay

This review is not arguing for removal of the stale-row queue.

The row-level stale queue is the right operator UX for subject-mask refinement
because:

- subject-mask edits are ROI-local
- partial source refresh is now possible
- re-reviewing only touched stale rows is much cheaper than re-cleaning an
  entire run

The problem is not the existence of the queue. The problem is that the queue is
not yet projected into the repo’s broader stale lifecycle model.

## Recommended Direction

### 1. Keep the current row-level stale queue

Do not remove:

- `source_row_stale`
- `source_update_pending_rows`
- stale-only ROI review

Those are the right local repair primitives.

### 2. Add a canonical top-level subject-mask stale payload

Mirror the eye-mask stale pattern by writing a structured
`source_subject_mask_stale` payload on affected refined subject-mask runs.

Minimum payload should include:

- `state = "stale"`
- `timestamp_utc`
- `reason`
- `source_subject_mask_run`
- affected `component_names`
- affected `roi_indices`
- optional source-update method metadata

The row-level arrays should remain the detailed local queue, while the top-level
payload becomes the canonical stale surface for lifecycle and registry sync.

### 3. Keep stale separate from review

Review status should continue to answer:

- approved?
- pending?
- rejected?
- needs manual QC?

Stale status should answer:

- did the approved/curated artifact drift from its upstream source?

The repo should not have to overload `needs_review` to mean both.

### 4. Add an explicit stale resolution path

The eye-mask precedent is good here.

Refined subject-mask stale handling should support an explicit operator action
analogous to:

- accept preserved refined rows as still valid ground truth
- record resolution metadata
- clear stale state without pretending the artifact was never stale

This should be a first-class path, not an implicit side effect of toggling the
component review state.

### 5. Project subject-mask stale into the registry

Once the canonical top-level stale payload exists, registry sync/query paths
should surface it so that:

- subject-mask lifecycle can distinguish stale from in-progress review
- stale-aware batch selection can use the registry when appropriate
- dashboards and training gates can reason about stale refined subject masks
  without zarr-only special cases

### 6. Split curated-preserve semantics from `edit_applied`

Keep `edit_applied` with its existing contract:

- row differs from source in a meaningful way

Add a separate explicit semantic bit for something like:

- `manual_override`
- `preserve_on_source_update`
- `curated_override`

whichever name is chosen should mean:

- preserve this refined row rather than auto-syncing it when the upstream source
  changes

That decision should not rely only on current difference-from-source.

### 7. Improve stale visibility in the review UI

Once stale is made first-class, the operator surfaces should show it directly.

At minimum:

- stale queue screens should show why the row is stale
- component summaries should expose row-level stale/manual-override state
- run/component views should distinguish stale-from-source from ordinary
  `pending` / `needs_review`

## Near-Term Acceptance Criteria

This TODO can be considered materially addressed when all of the following are
true:

1. refined subject-mask runs can emit a canonical `source_subject_mask_stale`
   payload
2. subject-mask registry rows surface that stale state
3. stale and review remain distinct lifecycle concepts
4. stale rows can still be queued and reviewed ROI-locally
5. operators have an explicit stale-resolution action
6. preserve-on-source-update semantics no longer depend only on
   `edit_applied`

## References

- [repo_wide_staleness_checklist.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/repo_wide_staleness_checklist.md)
- [repo_wide_staleness_gap_matrix.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/repo_wide_staleness_gap_matrix.md)
- [repo_wide_staleness_policy.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/repo_wide_staleness_policy.md)
- [swim_bladder_review_policy.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/swim_bladder_review_policy.md)
- [subject_mask_refinement_todo.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/subject_mask_refinement_todo.md)
- [mask_review_save_approval_policy.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/mask_review_save_approval_policy.md)
- [keypoint_late_correction_contract.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/keypoint_late_correction_contract.md)
- [refined_subject_masks_runs_contract.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/refined_subject_masks_runs_contract.md)
- [subject_mask_registry_contract.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/subject_mask_registry_contract.md)
