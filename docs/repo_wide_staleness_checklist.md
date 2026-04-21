# Repo-Wide Staleness Checklist By Stage Family

<!-- design-meta
status: draft
last_updated: 2026-04-06
-->

## Purpose

Turn the high-level policy in
[repo_wide_staleness_policy.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/repo_wide_staleness_policy.md)
into a concrete checklist for the main Palette stage families.

For the current implementation/contract gaps against this checklist, see
[repo_wide_staleness_gap_matrix.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/repo_wide_staleness_gap_matrix.md).

This note is intentionally operational. It answers:

- what artifact is the provenance authority?
- what artifact is the editable working surface?
- when is row-local stale repair safe?
- when should the repo escalate to rerun/invalidation?

## Shared Checklist

For every stage family, answer these questions explicitly:

1. What is the append-only provenance artifact?
2. What is the editable/refined artifact?
3. What stale payload should downstream consumers see?
4. Can downstream caches auto-sync?
5. Can curated downstream rows be preserved?
6. What row identity fields make row-local stale repair safe?
7. What kinds of changes force a rerun or run-level invalidation?
8. How is stale projected into registry/query surfaces?
9. How does an operator explicitly resolve stale without pretending it never
   happened?

## Detect

### Provenance authority

- `detect_runs/<run>`

### Editable working surface

- `refined_detect_runs/<run>/instances`
- run-level review/status metadata on `refined_detect_runs/<run>`

### Default policy

- never mutate `detect_runs/<run>` in place
- create a new detect run when the model, threshold family, or algorithm
  changes materially
- store curated/manual corrections in the refined run's canonical sparse
  surface, not in the raw detect run

### Row-local stale safety

Safe only when:

- the edited detection still refers to the same fish on the same frame
- downstream row identity can still be matched by `frame_indices` plus stable
  detection identity

### Escalate to rerun/invalidation when

- a detection is added or deleted
- one box becomes two, or two become one
- fish identity changes
- row order or detection identity can no longer be matched safely downstream

### Downstream consequences

- crop, keypoints, eye masks, subject masks, and swim-bladder masks may all be
  affected
- safe row-local edits should produce targeted stale events downstream
- identity-breaking edits should invalidate downstream work more aggressively

## Crop

### Provenance authority

- `crop_runs/<run>`

### Editable working surface

- currently limited compared with detect/keypoints/masks
- crop is mainly a derived geometry stage, not a primary manual-authoring
  surface

### Default policy

- treat crop as a derived artifact selected from detect lineage
- if the upstream detect selection changes materially, prefer regenerating crop
  from the chosen source rather than hand-editing crop data
- do not collapse downstream lineage to raw detect boxes alone; `crop_runs`
  remains the canonical ROI geometry/provenance layer
- in mixed-mode archives, persisted crop pixels may be optional, but crop
  geometry and row identity remain required

### Row-local stale safety

Safe only when:

- the crop row still maps to the same frame/detection identity
- bbox/crop geometry can be recomputed without changing row identity
- the canonical crop identity fields remain stable enough to reproduce the same
  downstream ROI mapping (`frame_indices`, `detection_indices`,
  `roi_coordinates_full`, ROI size, signature/revision)

### Escalate to rerun/invalidation when

- detect identity changes
- crop row alignment to downstream keypoint or mask rows can no longer be
  trusted

### Downstream consequences

- keypoints, eye masks, subject masks, and swim-bladder masks all depend on
  crop geometry
- crop changes are often more disruptive than they first appear, so row-local
  repair should be conservative here
- analysis archives may eventually use `geometry_only` crop runs, but training
  artifacts should continue to persist materialized crop pixels by default

## Keypoints

### Provenance authority

- `keypoints_runs/<run>`

### Editable working surface

- `refined_keypoints_runs/<run>`

### Default policy

- keep raw keypoint inference runs as provenance snapshots
- write late corrections into `refined_keypoints_runs/<run>`
- do not mutate historical raw keypoint runs by default

### Existing precedent

- downstream eye-mask runs already use explicit stale payloads after keypoint
  correction

### Row-local stale safety

Safe only when:

- the edited row still maps to the same crop/detection row
- `frame_indices` and `detection_indices` remain stable

### Escalate to rerun/invalidation when

- keypoint row alignment to crop or detect lineage is broken
- upstream crop identity changed
- the correction actually reflects a different ROI identity rather than a
  better annotation of the same ROI

### Downstream consequences

- eye masks should surface explicit stale
- subject masks and swim-bladder masks should follow the same pattern
- untouched downstream rows may auto-sync if they are non-curated caches
- curated downstream refined rows should be preserved and marked stale

## Eye Masks

### Provenance authority

- `eye_masks_runs/<run>`

### Editable working surface

- historical standalone eye editing: `refined_eye_masks_runs/<run>`
- canonical long-term editable surface for eye components:
  `refined_subject_masks_runs/<run>`

### Default policy

- treat raw `eye_masks_runs/<run>` as ROI-local source masks, not the place for
  manual curation
- keep save separate from approval
- prefer canonical eye authoring on the refined subject-mask surface as the
  repo continues to unify mask refinement

### Existing stale precedent

- explicit `source_keypoint_stale` payload with resolution path

### Row-local stale safety

Safe only when:

- source keypoint and crop lineage still resolve to the same ROI row
- eye identity is still the same anatomical pair for that row

### Escalate to rerun/invalidation when

- upstream crop or keypoint identity changes
- left/right eye identity can no longer be trusted
- the source row was replaced, not corrected

### Downstream consequences

- eye-mask profiles, merged training exports, and model-selection surfaces
  should treat stale distinctly from ordinary review state
- preserved refined eye edits need an explicit stale-resolution path, not just
  a review-state toggle

## Subject Masks

### Provenance authority

- `subject_mask_runs/<run>`

### Editable working surface

- `refined_subject_masks_runs/<run>`

### Default policy

- raw subject-mask runs are runtime/materialization outputs, not the canonical
  manual-authoring surface
- refined subject masks are the editable component-aware artifact
- component review and run review remain separate from source-drift state

### Row-local stale safety

Safe only when:

- `frame_indices` and `detection_indices` remain stable
- the changed source row still refers to the same ROI identity
- the affected component can be reconciled without changing row alignment

### Escalate to rerun/invalidation when

- source row identity changes
- component alignment across the run is no longer trustworthy
- an upstream change invalidates the assembled subject-mask row structure, not
  just component pixels

### Downstream consequences

- component-local stale queues are good and should remain
- but the canonical stale surface should be a top-level
  `source_subject_mask_stale` payload that registry/query tooling can see

## Swim-Bladder Masks

Swim bladder deserves its own checklist because it currently exists in two
different practical forms:

- coarse source masks materialized into `subject_mask_runs/<run>`
- refined curated swim-bladder component state inside
  `refined_subject_masks_runs/<run>/components/swim_bladder`

### Provenance authority

- the hand-tuned method/tuning lives in `analysis_metadata`
- the coarse materialized mask rows live in `subject_mask_runs/<run>`
- the canonical curated review surface lives in refined subject masks

### Editable working surface

- `refined_subject_masks_runs/<run>` component `swim_bladder`

### Default policy

- do not use the coarse materialized swim-bladder run as the human authority
- it is a non-curated source cache that may be partially refreshed in place
- human edits belong on the refined subject-mask component

### Row-local stale safety

Safe only when:

- the swim-bladder source row still maps to the same ROI identity
- crop and keypoint lineage remain aligned
- the source update changes only the mask content, not row identity

### Escalate to rerun/invalidation when

- the selected keypoint/crop lineage changes row identity
- the swim-bladder source run no longer aligns one-to-one with the refined row
- a broader subject-mask source reshuffle occurs

### Downstream consequences

- coarse swim-bladder masks may refresh in place because they are a
  non-curated cache
- refined swim-bladder rows should auto-sync only when untouched
- manually cleaned refined swim-bladder rows should be preserved and marked
  stale for targeted review

### Important distinction

Swim bladder is a subject-mask component, but operationally it should not be
collapsed into generic "subject masks" for stale handling.

Why:

- it has its own source materialization workflow
- it may be refreshed from keypoint-driven updates without recomputing the full
  subject-mask stack
- it benefits from row-local stale review even when the rest of the refined
  subject-mask run is fine

## Eye Masks Versus Subject Masks Versus Swim-Bladder Masks

These should remain distinct in policy discussions:

- `eye masks` are their own historical/runtime stage family with legacy and
  compatibility surfaces
- `subject masks` are the unified multi-component family
- `swim_bladder` is one subject-mask component, but it also has a dedicated
  coarse source materialization path that behaves like a cache feeding refined
  subject-mask review

Repo-wide policy should unify the stale vocabulary across them, but it should
not pretend they are operationally identical.

## Registry Expectations

For every family above, the registry should eventually be able to answer:

- is the selected artifact present?
- is it stale?
- why is it stale?
- what source run made it stale?
- what review state does it have?
- does it have targeted stale rows pending?

Run-level lifecycle and row-level repair queue are complementary, not
interchangeable.

## Recommended Near-Term Follow-Through

1. Keep raw provenance runs append-only across detect, keypoints, eye masks,
   and subject masks.
2. Keep refined/manual artifacts as the editable surface.
3. Treat coarse swim-bladder materialization as a refreshable cache, not the
   curation authority.
4. Standardize top-level stale payloads across eye-mask and subject-mask
   families.
5. Preserve row-local stale queues where lineage is stable.
6. Escalate to rerun/invalidation when identity changes, especially for detect
   and crop.

## References

- [repo_wide_staleness_policy.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/repo_wide_staleness_policy.md)
- [repo_wide_staleness_gap_matrix.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/repo_wide_staleness_gap_matrix.md)
- [detection_refinement_workflow.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/detection_refinement_workflow.md)
- [keypoint_late_correction_contract.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/keypoint_late_correction_contract.md)
- [mask_review_save_approval_policy.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/mask_review_save_approval_policy.md)
- [refined_subject_mask_staleness_todo.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/refined_subject_mask_staleness_todo.md)
