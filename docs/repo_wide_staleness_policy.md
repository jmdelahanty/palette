# Repo-Wide Staleness And Upstream-Correction Policy

<!-- design-meta
status: draft
last_updated: 2026-04-06
-->

## Purpose

Define a repo-wide policy for what should happen when an upstream artifact is
corrected after downstream stages already exist.

This note is intended to unify the current patterns used for:

- detect refinement and manual correction
- keypoint late correction and downstream eye-mask stale marking
- subject-mask and swim-bladder partial source refresh
- registry lifecycle derivation and step invalidation

For the concrete per-stage checklist, see
[repo_wide_staleness_checklist.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/repo_wide_staleness_checklist.md).
For the current gap summary, see
[repo_wide_staleness_gap_matrix.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/repo_wide_staleness_gap_matrix.md).
For the current implementation priority list, see
[repo_wide_staleness_implementation_todo.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/repo_wide_staleness_implementation_todo.md).
For the crop mixed-mode design background, see
[crop_live_view_vs_materialized_stream_design.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/archive/crop_live_view_vs_materialized_stream_design.md).

## Short Answer

Yes: as much as possible, do not mutate the pre-refinement "raw" run.

That should be the default design rule.

More precisely:

- raw provenance runs should be append-only
- refined/manual artifacts should be the editable working surface
- downstream drift from an upstream correction should be modeled explicitly as
  `stale`, not hidden inside ordinary review state

The main exception is a non-curated materialized cache whose only purpose is to
mirror an immutable source for downstream use. Those artifacts may be safely
refreshed in place if:

- they are not the provenance authority
- they do not contain human edits
- their lineage is explicit
- downstream consumers are told that the source changed

## Core Principles

### 1. Separate provenance, refinement, and review

These are different concepts:

- provenance: what the upstream algorithm or import originally produced
- refinement/manual correction: the editable working artifact
- review: whether a human has accepted the refined artifact for a given use

The repo should not rely on one concept to stand in for another.

### 2. Separate `missing`, `stale`, and review state

- `missing` means the downstream artifact is absent or was invalidated by a new
  upstream run
- `stale` means the downstream artifact still exists, but an upstream source
  changed underneath it
- review state means human QC status such as `approved`, `pending`,
  `needs_review`, or `rejected`

`stale` is not the same as `needs_review`.

### 3. Prefer immutable runs

When an upstream algorithm changes materially, prefer creating a new run rather
than rewriting an old one.

This is already the repo’s detect/refinement stance:

- raw detect is append-only
- refined detect is immutable
- manual detect corrections live in a separate subgroup rather than rewriting
  the raw detect run

### 4. Preserve curated downstream work

If a downstream artifact has manual curation, an upstream correction should not
silently overwrite it.

Instead:

- untouched downstream rows may auto-sync
- curated downstream rows should be preserved and marked stale for targeted
  review

### 5. Keep crop geometry canonical, even when crop pixels become optional

Raw detect boxes are not a sufficient downstream provenance contract.

Best-practice policy:

- `detect_runs` answer where an object was detected
- `crop_runs` answer what exact ROI patch downstream stages used
- `roi_images` may become optional in some archive classes
- crop geometry/provenance should remain canonical even when pixels are not
  persisted

Why this matters:

- a bbox alone does not fully encode ROI size, centering, clipping/padding,
  and source-selection lineage
- downstream keypoints and masks are usually written in ROI coordinates and
  need a stable mapping back to full-frame space
- row-local stale repair depends on stable crop identity fields such as
  `frame_indices`, `source_refined_row_ids` when refined detections are the
  source, `detection_indices` for physical source-row addressing,
  `roi_coordinates_full`, and crop signature/revision

So the clean long-term policy is not "detections only." It is "shared
crop-stage geometry/provenance remains canonical, while persisted ROI pixels
may be optional in mixed-mode analysis workflows."

## Artifact Classes

The repo should reason about upstream changes by artifact class rather than by
stage name alone.

### A. Append-only provenance runs

Examples:

- `detect_runs/<run>`
- imported/raw source-aligned stage outputs

Policy:

- never rewrite in place
- if the algorithm or source changes, create a new run
- if manual correction is needed, attach it to the refined/manual layer rather
  than mutating the provenance run

### B. Refined/manual working artifacts

Examples:

- `refined_detect_runs/<run>/<manual_group>`
- `refined_keypoints_runs/<run>`
- `refined_subject_masks_runs/<run>`

Policy:

- these are the canonical editable surfaces
- edits may be row-local
- approval stays explicit and separate from saving
- downstream artifacts should receive explicit stale markers when upstream rows
  change

### C. Non-curated materialized caches

Examples:

- runtime mask/materialization outputs that are regenerated from immutable
  upstream lineage and do not contain human edits
- materialized crop pixels in archive classes where `crop_runs` geometry is the
  real authority and `roi_images` acts as a persisted acceleration/cache layer

Policy:

- these may be refreshed in place when the source changes
- they are not the provenance authority
- they should still emit explicit source-drift signals for downstream refined
  artifacts

This is the narrow exception to the "do not mutate raw runs" rule.

## Decision Rule For Upstream Changes

When an upstream artifact changes, classify the event first.

### Case 1. New upstream run

Example:

- new detect run
- new refined detect run
- new refined keypoint run produced from different params/model

Policy:

- downstream steps should be invalidated through the runtime cascade and
  treated as `missing`
- this is a run identity change, not a row-level stale event

### Case 2. Correction inside an existing editable artifact

Example:

- manual correction in `refined_keypoints_runs/<run>`
- manual correction in `refined_subject_masks_runs/<run>`
- manual subgroup update under a refined detect run

Policy:

- keep the artifact identity
- mark dependent downstream artifacts `stale`
- preserve curated downstream edits
- provide targeted re-review only for affected rows/components when lineage is
  stable

### Case 3. Refresh of a non-curated materialized cache

Example:

- partial recompute of coarse swim-bladder masks after a keypoint correction

Policy:

- in-place refresh is acceptable
- downstream refined artifacts should still treat this as a source-change event
- stale metadata should record that the source cache changed underneath the
  refined artifact

## Row-Level Versus Run-Level Repair

Row-level stale repair is safe only when row identity is stable.

Minimum expectation:

- stable `frame_indices`
- stable `source_refined_row_ids` for current refined-detect sources, or
  `detection_indices` only for source rowsets whose physical row order is known
  stable
- no ambiguous reordering, merge, or split of the underlying source object

For current refined detections, the equivalent row identity is
`refined_detect_runs/<run>/instances/refined_row_ids`, copied onto crop runs as
`crop_runs/<run>/source_refined_row_ids`; raw lineage is carried by
`instances/source_detect_row_index` and optionally by
`crop_runs/<run>/source_detect_row_index`, and audited through
`source_detections/resolved_refined_row_id`. See
[refined_detect_row_identity_contract.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/refined_detect_row_identity_contract.md).

If those conditions fail, the repo should escalate from row-level stale repair
to broader rerun or invalidation.

## Bounding-Box Example

Bounding-box correction is the right stress test because it can affect crop
geometry, keypoints, masks, and later derived metrics.

### Safe row-local case

Example:

- one refined/manual bbox is nudged
- the detection still refers to the same fish on the same frame
- row identity is stable

Recommended behavior:

1. keep the raw detect run unchanged
2. store the correction on the refined/manual detect surface
3. mark dependent crop/keypoint/mask artifacts stale for the affected rows
4. auto-refresh uncurated downstream caches when possible
5. preserve curated downstream refined rows and queue only those rows for stale
   review

### Identity-breaking case

Example:

- a bbox is added, removed, split, merged, or reassigned to a different fish
- row order or detection identity changes

Recommended behavior:

- do not pretend this is a safe row-local refresh
- downstream lineage should be invalidated more aggressively
- rerun from the affected stage downward, or create a new upstream run and let
  cascade invalidation mark downstream work `missing`

## Canonical Stale Payload Policy

Stages that support downstream stale handling should expose a top-level stale
payload on the affected artifact family.

Minimum shape:

```json
{
  "state": "stale",
  "timestamp_utc": "2026-04-06T12:00:00Z",
  "reason": "upstream_manual_correction",
  "source_group": "refined_keypoints_runs",
  "source_run": "refined_keypoints_...",
  "roi_indices": [56, 77],
  "frame_indices": [1234, 1300]
}
```

Recommended extensions:

- component names
- resolution metadata
- stale method
- source lineage fingerprints

Detailed row-local queues may live deeper in component attrs, but the top-level
payload should be the canonical registry/query surface.

## Preserve-On-Source-Update Policy

The repo should not infer long-term preservation semantics only from
`edit_applied`.

Preferred rule:

- `edit_applied` means "differs meaningfully from source"
- a separate explicit bit should mean "preserve this row if the source changes"

This matters because:

- deterministic refinement can differ from source without being human curation
- source changes can make an untouched row look edited after the fact

## Registry Policy

Registry/query tooling should surface stale explicitly.

That means:

- top-level stale payloads should be projected into registry rows
- lifecycle derivation should keep `stale` separate from review-state
  transitions
- batch tools should prefer registry-backed stale selection where practical

Local zarr-only stale queues may still exist for detailed row targeting, but
they should not be the only way to discover stale work.

## Near-Term Recommended Adoption

1. Keep raw provenance runs append-only by default.
2. Keep refined/manual artifacts as the editable surfaces.
3. Allow in-place refresh only for non-curated materialized caches.
4. Represent upstream drift with explicit stale payloads.
5. Preserve curated downstream rows and review only the affected subset.
6. Escalate to rerun/invalidation when row identity is no longer stable.
7. Project stale into the registry rather than hiding it inside review notes or
   component-local attrs only.

## References

- [detection_refinement_workflow.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/detection_refinement_workflow.md)
- [keypoint_late_correction_contract.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/keypoint_late_correction_contract.md)
- [mask_review_save_approval_policy.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/mask_review_save_approval_policy.md)
- [repo_wide_staleness_checklist.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/repo_wide_staleness_checklist.md)
- [repo_wide_staleness_gap_matrix.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/repo_wide_staleness_gap_matrix.md)
- [refined_detect_row_identity_contract.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/refined_detect_row_identity_contract.md)
- [refined_subject_mask_staleness_todo.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/refined_subject_mask_staleness_todo.md)
