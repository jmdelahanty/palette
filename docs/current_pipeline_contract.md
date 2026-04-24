# Current Palette Pipeline Contract

<!-- contract-meta
version: 1
status: active
last_verified: 2026-04-24
-->

Purpose: define the current operator-facing source-of-truth contract for Palette
pipeline artifacts while the codebase is still migrating from eye-specific mask
stages to unified subject-mask components.

This document is the short current-state contract. Deeper design notes and TODOs
remain useful, but operator/query behavior should be judged against this file
first.

## Design Philosophy

Palette should keep these artifact roles distinct:

- Raw provenance snapshots: append-only inference or materialization outputs.
  They record what a model or method produced for a row-aligned source.
- Refined authority artifacts: curated or editable surfaces that operators and
  exports should trust for reviewed state.
- Compatibility surfaces: historical or adapter layouts kept readable for older
  archives, old viewers, and migration tooling. They are not competing
  canonical authorities.
- Refreshable caches: derived materializations that can be regenerated when
  sources change. They are not human curation authority.
- Training/export products: out-of-band products built from the current
  authority. They should be rebuilt after authority changes rather than treated
  as live pipeline state.

The default rule is:

- raw stages preserve provenance
- refined stages carry reviewed/editable authority
- registry and operator summaries should prefer the current authority and label
  compatibility state explicitly
- stale state is separate from review state

## Current Stage Contract

| Family | Raw provenance | Current refined authority | Compatibility/cache surfaces | Operator/query truth |
| --- | --- | --- | --- | --- |
| Import/video metadata | imported analysis/training zarr metadata | none | downsampled video products | imported archive metadata and manifest status |
| Detect | `detect_runs/<run>` | `refined_detect_runs/<run>/instances` | legacy refined subgroups such as `filtered`, `interpolated`, and `manual_*` | canonical curated refined rows when present; raw detect only as fallback or provenance |
| Detect quality | detect-run quality reports | refined detect review/status metadata | legacy detect-quality aliases | quality labels feed refine; review state belongs to refined detect |
| Crop | `crop_runs/<run>` | none in normal operation | geometry-only or repaired crop variants | current crop run that still matches selected detect/refined lineage |
| Keypoints | `keypoints_runs/<run>` | `refined_keypoints_runs/<run>` | legacy keypoint attrs such as singular `source_keypoint_run` | refined keypoints when present; metadata-driven pose and heading semantics |
| Raw segmentation | `subject_mask_runs/<run>` | none | `eye_masks_runs/<run>` remains a legacy eye-specific raw stage during migration | unified subject-mask component availability for current mask state |
| Refined subject masks | `subject_mask_runs/<run>` sources plus component provenance | `refined_subject_masks_runs/<run>` | none; this is the canonical refined component surface | component availability, review state, and lifecycle from refined subject-mask component rows |
| Refined eye masks | `eye_masks_runs/<run>` or projected subject-mask sources | `refined_subject_masks_runs/<run>` for current eye review | `refined_eye_masks_runs/<run>` is historical or derived compatibility layout | active eye geometry/export should prefer refined subject-mask eye components and fall back to refined-eye only for historical archives |
| Swim bladder | coarse masks materialized in `subject_mask_runs/<run>` | `refined_subject_masks_runs/<run>/components/swim_bladder` | coarse swim-bladder materialization is a refreshable source cache | refined subject-mask swim-bladder component state |
| Arena assignment/tracking | selected detect/refined lineage outputs | tracking QC/status metadata | older raw-detect-aligned assignments | assignment/tracking rows whose source lineage matches the selected detect/refined state |

## Mask-Specific Rules

The mask transition is the main place where the contract is not yet a finished
steady state.

Current rules:

- `subject_mask_runs` is the canonical raw multi-component mask family.
- `refined_subject_masks_runs` is the canonical editable/refined component mask
  family for body, eyes, and swim bladder.
- `eye_masks_runs` remains writable during migration because current eye
  producers and workflows still depend on it.
- New eye orchestration should project or companion-write eye outputs into
  `subject_mask_runs` when possible.
- `refined_eye_masks_runs` remains readable and may be materialized as a
  derived compatibility artifact, but it should not become a second manual
  review authority for new operator-facing eye state.
- Active eye geometry and export consumers should use the shared resolver that
  prefers `refined_subject_masks_runs` and falls back to `refined_eye_masks_runs`
  for historical archives.

The target steady state is one segmentation orchestration surface that writes
one coherent `subject_mask_runs/<run>` snapshot with component-scoped method and
provenance metadata. Until that exists, old eye-specific stages are tolerated as
transition surfaces, not design precedent.

## Registry And Operator Surface Rules

Registry and operator output should answer current availability and review state
from unified component surfaces wherever they exist.

Required behavior:

- Subject-mask component rows should rank available
  `refined_subject_masks_runs` components ahead of legacy eye rows.
- Partial refined-subject runs are valid. A refined run with only one reviewed
  component should be visible for that component and should not be hidden by a
  broader raw or legacy run.
- Stale lifecycle state must remain visible even when a row is otherwise the
  preferred component row.
- Subject-mask component registry views should project refined-run
  `source_subject_mask_stale_*` fields so query, training, and operator
  surfaces can explain why a component is stale without treating stale as a
  review label.
- `check_recording_steps` should label `eye_masks` and `refined_eye_masks` as
  legacy compatibility and report unified component summaries separately.
- Query tools may expose legacy eye-mask filters for diagnostics and historical
  training compatibility, but current mask-state filters should use subject-mask
  component views.

## Staleness Rules

Stale state is a source-lineage problem, not a review-state problem.

Required behavior:

- Raw provenance runs should not be edited to hide source changes.
- Curated refined rows may be preserved after a source change only when row
  identity is still stable.
- Preserved curated rows should be marked stale or queued for targeted review,
  not silently treated as clean.
- Refreshable caches can be regenerated or partially refreshed when lineage is
  stable.
- Identity-breaking changes should escalate to rerun or run-level invalidation.

For masks, the intended canonical stale payload is visible at the refined
subject-mask level and projected into registry/query/operator surfaces. Existing
eye-mask stale payloads are the precedent, not the final mask-wide answer.

## Current Known Gaps

These gaps are allowed transition state, but they should not be treated as the
desired design:

- `src/fisheye/core/pipeline.py` still exposes `eye_masks` and
  `refined_eye_masks` as first-class stages.
- There is no completed top-level `segmentation` orchestration step that writes
  one coherent `subject_mask_runs/<run>` snapshot across body, eyes, and swim
  bladder.
- Some registry query and training-prep paths still expose legacy eye-mask
  filters as primary-looking options.
- Subject/refined-subject stale repair is not yet as complete as the eye-mask
  stale precedent, though component registry/query/operator projection now
  exposes refined-run source-stale state.
- Older operator docs may list component-specific segmentation commands before
  describing the unified target.
- Detect row-local stale repair now has a contract and validator, but downstream
  crop/keypoint/mask consumers still need a full audit for positional or legacy
  manual-subgroup assumptions.

## Review Checklist

When reviewing new pipeline work, ask:

1. What is the raw provenance artifact?
2. What is the refined or operator-authoritative artifact?
3. Is any legacy surface clearly labeled as compatibility?
4. Does the registry/query surface prefer the same authority as runtime/export?
5. Are partial component runs handled without hiding stale state?
6. Does source drift become explicit stale state rather than a review toggle?
7. Does the change move code toward one raw segmentation snapshot and one
   refined component authority?

## Related Documents

- [recording_analysis_pipeline_contract.md](recording_analysis_pipeline_contract.md)
- [segmentation_pipeline_step_todo.md](segmentation_pipeline_step_todo.md)
- [subject_mask_refinement_todo.md](subject_mask_refinement_todo.md)
- [subject_mask_stage_unification_todo.md](subject_mask_stage_unification_todo.md)
- [repo_wide_staleness_checklist.md](repo_wide_staleness_checklist.md)
- [repo_wide_staleness_gap_matrix.md](repo_wide_staleness_gap_matrix.md)
- [refined_detect_row_identity_contract.md](refined_detect_row_identity_contract.md)
- [src/fisheye/docs/provenance_workflow.md](../src/fisheye/docs/provenance_workflow.md)
