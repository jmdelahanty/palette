# Training Label-Origin Provenance TODO

## Goal

Make merged training datasets self-describing enough that, for any exported row,
we can answer:

- which source dataset/frame it came from
- whether the label content was automatic, manually corrected, manually created,
  or interpolated/synthetic
- what supervision mode the row carries for training

This should be answerable from the merged training dataset itself, without
requiring a join back into the source Zarr for routine auditing.

## Why this is needed

Current merged training exports already preserve source-row lineage:

- `source_index/source_dataset_idx`
- `source_index/source_frame_idx`
- `source_index/source_dataset_id`
- `source_index/source_zarr_path`

That is enough to trace a row back to its source frame, but not enough to answer
"was this label manual or automatic?" directly from the merged dataset.

Current gaps:

- Detect merged export now writes the canonical
  `refined_detect_runs/<run>/instances` surface and preserves
  `source_kind_codes` plus `manual_edit_flags`; this distinguishes raw-detect,
  manual, and manually edited positive rows better than the old crop
  `detection_source` field. A cross-task `label_origin` / `supervision_mode`
  field is still not standardized across detect, pose, and masks.
- Keypoint merged export preserves run-level provenance
  (`refined_keypoint_run`, `quality_registry_refined_run`,
  `keypoint_review_status`) and row-level supervision mode
  (`keypoint_box_only`), but not a canonical per-row manual-vs-auto origin.
- Eye-mask merged export preserves source lineage and profile review metadata,
  but not a per-row label-origin field.

## Success Criteria

- A merged training row can be classified without source-Zarr lookups.
- The classification is stable across detect, pose, and eye-mask exports.
- Validation fails if a claimed row-origin encoding is malformed or invalid.
- Training cards and registry summaries can report counts by label origin.
- Existing exports remain readable during migration.

## Proposed Model

Use two separate concepts instead of one overloaded field:

1. `label_origin`
   Purpose: where the annotation content came from.

   Proposed stable vocabulary:

   - `auto`
   - `manual_review`
   - `manual_training`
   - `interpolated`
   - `synthetic`
   - `unknown`

2. `supervision_mode`
   Purpose: how the row should be interpreted by training.

   Proposed stable vocabulary:

   - `dense`
   - `box_only`
   - `explicit_negative`
   - `no_supervision`

Rationale:

- `label_origin` answers the provenance question.
- `supervision_mode` answers the training semantics question.
- This avoids conflating "manual vs auto" with "full vs box-only vs negative".

## Storage Proposal

Preferred:

- Add row-level arrays to merged exports under `source_index/`:
  - `source_index/label_origin`
  - `source_index/supervision_mode`

Optional optimization:

- use small integer codes on disk with attrs documenting the enum mapping
- keep string labels in summaries/manifests for readability

Why `source_index/`:

- these are row-lineage fields, not model-output arrays
- all training tasks already read `source_index/*`
- cards/audits can consume them uniformly

## Task-by-Task Mapping Work

### Detect

- Split current "real/manual-reviewed" bucket into explicit row-level origin.
- Derive `label_origin` from the actual resolved source subgroup when possible:
  - `detect`
  - `filtered`
  - `manual`
  - `interpolated`
- Preserve current `detection_source` for backward compatibility during
  migration, but stop treating it as the only provenance field.

### Keypoints

- Identify the authoritative row-level signal for:
  - untouched pass-through rows
  - manual corrections
  - interpolated rows
  - future manual training labels
- Export that as `label_origin`.
- Continue exporting `keypoint_box_only`, but also map it into
  `supervision_mode=box_only`.

### Eye masks

- Determine whether row-level manual-vs-auto provenance exists in source runs or
  whether only run-level review/profile metadata is available today.
- If only run-level metadata exists, document the limitation and emit
  `label_origin=unknown` rather than inventing per-row precision.
- Map explicit negatives into `supervision_mode=explicit_negative`.

## Implementation Plan

### Phase 1: Source-Signal Audit

- See the concrete checklist in
  [training_label_origin_phase1_audit.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/training_label_origin_phase1_audit.md).
- [ ] Detect: identify the exact row-level source signal available in source
  runs and refined/manual subgroups.
- [ ] Keypoints: identify the exact row-level source signal available in refined
  runs (`clean`, `manual`, `interpolated`, or equivalent reason/status arrays).
- [ ] Eye masks: identify whether row-level manual-vs-auto provenance exists, or
  whether only run-level review/profile metadata is available.
- [ ] Record edge cases where only dataset-level provenance is available.

### Phase 2: Contract

- [ ] Write a small contract doc for merged training row provenance:
  - field names
  - allowed values
  - enum/code mapping
  - backward-compatibility rules
- [ ] Decide whether arrays are stored as strings or integer codes plus attrs.
- [ ] Decide whether `manual_review` and `manual_training` must remain distinct.

### Phase 3: Exporters

- [ ] Update [export_detect_training_zarr.py](/home/delahantyj@hhmi.org/gitrepos/palette/src/fisheye/utils/export_detect_training_zarr.py)
  to write row-level label provenance.
- [ ] Update [export_keypoint_training_zarr.py](/home/delahantyj@hhmi.org/gitrepos/palette/src/fisheye/utils/export_keypoint_training_zarr.py)
  to write row-level label provenance.
- [ ] Update [export_eye_mask_training_zarr.py](/home/delahantyj@hhmi.org/gitrepos/palette/src/fisheye/utils/export_eye_mask_training_zarr.py)
  to write row-level label provenance where support exists.
- [ ] Add validation rules to merged-export validators.

### Phase 4: Prepare/Manifest/Registry Surfaces

- [ ] Include aggregate `label_origin` counts in merged `training_export` attrs.
- [ ] Include per-source and merged counts in manifests/summary JSON.
- [ ] Surface counts in training registry checks.
- [ ] Surface counts in training data cards.

### Phase 5: Backfill and Migration

- [ ] Decide whether legacy merged exports need backfill support or only
  forward-compatible readers.
- [ ] If backfill is required, implement a non-destructive backfill tool that
  reconstructs row provenance only when the source signal is unambiguous.

## Validation

- [ ] Unit tests for each exporter covering manual, auto, interpolated, and
  unknown cases.
- [ ] Validator tests for malformed codes/labels and length mismatches.
- [ ] One end-to-end smoke test per task verifying provenance survives merged
  export and summary generation.

## Open Questions

- Should `filtered` remain an explicit `label_origin`, or should it collapse into
  `auto` with separate run metadata?
- Should `interpolated` and `synthetic` remain distinct, or share one bucket?
- For tasks where only run-level provenance exists, do we permit dataset-wide
  fill values, or require `unknown`?
- Do we want one cross-task enum, or task-specific enums normalized only in
  higher-level summaries?

## Non-Goals

- Reconstruct hidden per-row provenance when the source run does not encode it.
- Rewrite historical source Zarr layouts just to satisfy merged-export metadata.
- Change current row-gating semantics as part of this work.

## Related Docs

- [provenance_todo.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/provenance_todo.md)
- [training_label_origin_phase1_audit.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/training_label_origin_phase1_audit.md)
- [training_quality_gate_contract.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/training_quality_gate_contract.md)
- [detection_merged_export_contract.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/detection_merged_export_contract.md)
- [keypoint_merged_row_gate_contract.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/keypoint_merged_row_gate_contract.md)
- [keypoint_training_workflow.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/keypoint_training_workflow.md)
