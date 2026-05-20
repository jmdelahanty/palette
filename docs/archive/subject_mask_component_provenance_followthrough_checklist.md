# Subject-Mask Component Provenance Follow-Through Checklist
<!-- contract-meta
version: 1
status: draft
last_verified: 2026-04-14
-->

Purpose: define the next Palette-only implementation passes needed to bring
`subject_mask_runs` and component-scoped subject-mask provenance up to the same
crop-snapshot standard now used by keypoints and eye masks.

This is an implementation checklist, not a historical note. It is intended to
be used as the working target for future code changes.

## Scope

In scope:

- raw `subject_mask_runs` writers
- component-scoped subject-mask provenance writers
- merged subject-mask assembly
- provenance diagnostics for subject-mask crop snapshot drift
- focused runtime/docs/tests for the above

Out of scope for this checklist:

- registry schema changes
- `refined_subject_masks_runs` finalization semantics beyond crop-lineage carry-through
- training/export data cards
- eye-mask or keypoint lineage (already handled elsewhere)

## Current Verified State

As of 2026-04-14:

- Active subject-mask writers already carry:
  - `source_crop_run`
  - `source_crop_storage_mode`
  - `source_crop_signature`
  - `source_crop_revision`
- The active writers do **not** yet carry:
  - `source_detect_review_status_ref`
- The active writers currently hand-roll crop snapshot attrs instead of using
  the shared helper:
  - `src/fisheye/segmentation/subject_segmentation.py`
  - `src/fisheye/segmentation/infer_unet_subject_masks.py`
  - `src/fisheye/utils/run_sam_subject_masks.py`
  - `src/fisheye/segmentation/swim_bladder_segmentation.py`
- Component provenance under
  `components/<component>/provenance` currently records only stage/run/method/
  channel metadata via
  `src/fisheye/shared/subject_mask_component_provenance.py`.
- `merge_subject_mask_runs.py` still writes only `source_crop_run` at the
  merged-run level and does not validate crop snapshot equivalence across its
  source runs.
- `check_provenance_consistency.py` does not yet audit `subject_mask_runs`
  against current crop snapshot fields.

## Canonical Target

### Run-level subject-mask crop snapshot contract

Every new `subject_mask_runs/<run>` writer should record the same canonical crop
snapshot field set used by keypoints and eye masks:

- `source_crop_run`
- `source_crop_storage_mode`
- `source_crop_signature`
- `source_crop_revision`
- `source_detect_review_status_ref`

The same field set should appear in:

- run attrs
- `provenance.inputs`
- any step-status or audit payloads that already expose crop provenance

### Component provenance contract

Each `components/<component>/provenance` subgroup should remain the canonical
component-local lineage record. It should continue to store its current source
stage/run/method/channel metadata, and should also gain the crop snapshot fields
 needed to relate the component back to the crop surface it was derived from:

- `source_crop_run`
- `source_crop_storage_mode`
- `source_crop_signature`
- `source_crop_revision`
- `source_detect_review_status_ref`

Component provenance should stay component-local and should not be replaced by
top-level run attrs.

### Normalization policy

Crop snapshot fields must be written through the shared helper:

- `build_source_crop_snapshot_attrs(...)` in
  `src/fisheye/shared/provenance_attrs.py`

Specific expectations:

- `source_crop_signature` should use the helper’s normalized representation.
  Do not write raw dicts in some writers and strings in others.
- `source_detect_review_status_ref` should be copied from the crop run when
  present, and omitted when absent.
- Writers should not manually duplicate field-by-field crop snapshot logic once
  the helper is available.

## Implementation Checklist

### 1. Standardize active subject-mask writers

- [x] Update `src/fisheye/segmentation/subject_segmentation.py`
  to use `build_source_crop_snapshot_attrs(...)` for both run attrs and
  `provenance.inputs`.
- [x] Update `src/fisheye/segmentation/infer_unet_subject_masks.py`
  the same way.
- [x] Update `src/fisheye/utils/run_sam_subject_masks.py`
  the same way.
- [x] Update `src/fisheye/segmentation/swim_bladder_segmentation.py`
  the same way.

Acceptance criteria:

- All four writers expose the full crop snapshot quintet.
- All four writers write the same normalized `source_crop_signature` type.
- All four writers include `source_detect_review_status_ref` when the crop run
  has it.

### 2. Extend component-scoped provenance

- [x] Extend `src/fisheye/shared/subject_mask_component_provenance.py`
  to accept the crop snapshot field set in a structured way.
- [x] Update all subject-mask writers that call
  `write_subject_mask_component_provenance(...)` to pass the crop snapshot
  fields through.

Acceptance criteria:

- `components/<component>/provenance.attrs` exposes both:
  - existing source-stage/source-run/source-channel metadata
  - the canonical crop snapshot field set
- No writer manually writes component provenance attrs outside the helper for
  these same fields.

### 3. Fix merged subject-mask assembly

- [x] Update `src/fisheye/utils/merge_subject_mask_runs.py`
  to validate crop snapshot equivalence across source runs, not just
  `source_crop_run` and row alignment.
- [x] If the body-source and eye-source crop snapshot fields disagree, fail
  assembly with an explicit error listing the mismatched fields.
- [x] When the sources agree, write the shared crop snapshot field set to the
  merged run attrs and provenance.
- [x] Pass the crop snapshot field set into each component provenance entry for
  the merged run.

Acceptance criteria:

- Merge does not silently choose one source run’s crop snapshot when the other
  disagrees.
- Successful merged runs expose the same crop snapshot fields as native
  subject-mask runs.

### 4. Add subject-mask provenance audit

- [x] Extend `src/fisheye/diagnostics/check_provenance_consistency.py`
  to inspect the latest `subject_mask_runs/<run>` entry.
- [x] Reuse the same crop snapshot comparison semantics already used for
  keypoints and eye masks:
  - wrong `source_crop_run`
  - stale `source_crop_signature`
  - stale `source_crop_revision`
  - stale or missing `source_detect_review_status_ref`
  - stale `source_crop_storage_mode`
- [x] Add a distinct issue bucket for subject-mask crop snapshot drift rather
  than overloading crop-vs-source drift.

Acceptance criteria:

- Healthy subject-mask runs do not produce provenance issues.
- Stale subject-mask runs produce field-specific messages.
- The checker clearly separates:
  - crop run drift from upstream detect/refined source
  - downstream subject-mask crop snapshot drift

### 5. Keep `refined_subject_masks_runs` decisions explicit

- [x] Decide whether `refined_subject_masks_runs` should adopt the same crop
  snapshot quintet in this pass or in a follow-up.
- [x] Carry the same crop snapshot contract through `refined_subject_masks_runs`
  in this pass.

Required decision rule:

- Do not partially implement `refined_subject_masks_runs` in an ad hoc way.
- Decision taken on 2026-04-15:
  current `refined_subject_masks_runs/<run>` writers preserve the same
  `source_crop_*` + `source_detect_review_status_ref` contract as their
  upstream `subject_mask_runs/<run>` source, and provenance/audit surfaces
  should treat refined subject-mask runs as crop-snapshot consumers.

### 6. Update docs and contracts

- [x] Update `src/fisheye/docs/provenance_workflow.md`
  to list subject-mask crop snapshot fields explicitly.
- [x] Update `docs/subject_mask_runs_contract.md`
  so the required/optional attrs match the actual crop snapshot target.
- [x] Update active subject-mask contract/reference docs that describe component
  provenance if the helper schema changes.

Acceptance criteria:

- Active docs describe the same field set the runtime writes.
- No active doc claims subject-mask component provenance is only stage/run
  metadata if the implementation now carries crop snapshot fields too.

### 7. Add focused tests

- [x] Extend `tests/unit/fisheye/test_subject_segmentation.py`
  to assert crop signature/revision/review linkage on attrs and provenance.
- [x] Extend `tests/unit/fisheye/test_infer_unet_subject_masks_source.py`
  the same way.
- [x] Extend `tests/unit/fisheye/test_run_sam_subject_masks.py`
  to assert crop snapshot propagation into run attrs, provenance, and component
  provenance.
- [x] Extend `tests/unit/fisheye/test_merge_subject_mask_runs.py`
  to assert:
  - crop snapshot fields on successful merged runs
  - explicit failure on mismatched source crop snapshots
- [x] Add or extend tests for
  `src/fisheye/shared/subject_mask_component_provenance.py`.
- [x] Extend
  `tests/unit/fisheye/test_check_provenance_consistency.py`
  to cover subject-mask crop snapshot drift reporting.

Test policy:

- Prefer deterministic fake-group/in-memory tests for provenance logic.
- Only use real-zarr tests where the writer path genuinely requires them.
- If a real-zarr sandbox test hangs, defer it as local validation with the
  exact command.

## Implementation Rules

- Keep `source_detect_review_status_ref` as a separate stable provenance field.
  Do not fold mutable review payloads into `source_crop_signature`.
- Do not change `crop_signature` semantics in this pass.
- Do not widen this pass into registry schema work unless a runtime change
  forces it.
- Prefer shared helpers over repeated attr assembly.
- Fail loudly on mixed-source ambiguity in merge code instead of encoding
  unclear precedence rules.

## Suggested Execution Order

1. Standardize the four active subject-mask writers.
2. Extend the component provenance helper and migrate its callers.
3. Tighten `merge_subject_mask_runs.py`.
4. Add subject-mask checks to `check_provenance_consistency.py`.
5. Update docs and tests together.

## Done Definition

This checklist is complete when:

- all active `subject_mask_runs` writers use the shared crop snapshot helper,
- subject-mask component provenance carries crop snapshot lineage,
- merged subject-mask runs either prove source equivalence or fail cleanly,
- provenance diagnostics report stale subject-mask crop snapshots,
- active docs and focused tests reflect the new contract.
