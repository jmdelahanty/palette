# Training Label-Origin Phase 1 Audit

## Purpose

Turn the training label-origin provenance work into a concrete audit against the
source arrays and attrs that exist today.

This phase is not about changing exporters yet. It is about answering, for each
training task:

- what row-level source signals already exist
- whether manual vs automatic vs interpolated provenance is actually
  recoverable today
- which signal should be treated as authoritative
- where the current ambiguity is real and cannot be inferred away

## Required Output Per Task

For detect, keypoints, and eye masks, Phase 1 should produce:

1. the authoritative row-level provenance signal, if one exists
2. the fallback signal(s), if the authoritative signal is absent
3. the cases where only dataset-level provenance is available
4. the proposed mapping to `label_origin`
5. the proposed mapping to `supervision_mode`

## Cross-Task Audit Questions

- Does the source run expose a row-level signal, or only a run-level/source-path
  label?
- If a manual subgroup exists, does it contain only edited rows or a full merged
  rowset with edited and untouched rows?
- If `retune_id` exists, does `retune_id > -1` reliably mean "manually tuned"?
- If `reason` exists, is its vocabulary stable enough to drive provenance, or is
  it primarily diagnostic?
- Does the merged exporter already preserve the signal, or does it drop it?

## Detect

### Current Source Signals Confirmed

- Refined detect group resolution already distinguishes dataset/run-level source
  subgroup preference via `manual`, `interpolated`, `filtered`, `raw` in
  [refined_detect_review.py](/home/delahantyj@hhmi.org/gitrepos/palette/src/fisheye/shared/refined_detect_review.py).
- Registry prepare flow already resolves the selected refined subgroup using
  `manual_review_latest`, then `interpolated`, then `filtered` in
  [prepare_detect_training_from_registry.py](/home/delahantyj@hhmi.org/gitrepos/palette/src/fisheye/utils/prepare_detect_training_from_registry.py).
- Refined detect source groups expose row-level `detection_source`,
  `reason_bytes`, and `reason` in
  [refine_detect.py](/home/delahantyj@hhmi.org/gitrepos/palette/src/fisheye/refinement/refine_detect.py)
  and the schema in
  [stage_arrays.py](/home/delahantyj@hhmi.org/gitrepos/palette/src/fisheye/shared/zarr/stage_arrays.py).
- Manual detect review output groups additionally expose `retune_id`, and the
  writer preserves kept rows while assigning fresh `retune_id` values to newly
  retuned rows in
  [detect_review.py](/home/delahantyj@hhmi.org/gitrepos/palette/src/fisheye/tune/detect_review.py).
- The merged detect exporter currently preserves `detection_source` and source
  lineage, but not a normalized manual-vs-auto row-origin field, in
  [export_detect_training_zarr.py](/home/delahantyj@hhmi.org/gitrepos/palette/src/fisheye/utils/export_detect_training_zarr.py).

### Preliminary Assessment

- `interpolated` is clearly recoverable row-level today via
  `detection_source == 1`.
- Dataset/run-level source selection (`manual`, `filtered`, `detect`,
  `interpolated`) is recoverable today.
- Per-row manual edit provenance may be recoverable from `retune_id` and/or
  manual-group `reason`, but that must be verified against actual manual groups.
- Current merged detect training exports collapse all non-interpolated rows into
  the same bucket, so the merged dataset alone cannot distinguish untouched real
  rows from manually reviewed ones.

### Audit Checklist

- [ ] Verify actual manual review groups contain `retune_id` for all supported
  detect-review outputs.
- [ ] Confirm whether `retune_id > -1` means "edited/inserted during manual
  review" and `-1` means "carried through unchanged".
- [ ] Inspect the manual-group `reason` vocabulary:
  - expected candidates: `retune`, inherited `clean`, inherited
    `interpolated`, or `kept`
- [ ] Confirm whether a manual subgroup stores a full rowset or only the edited
  subset.
- [ ] Confirm whether rows copied from an interpolated base keep
  `detection_source == 1` after manual review.
- [ ] Decide whether detect `label_origin` should be derived from:
  - subgroup label alone
  - subgroup label + `retune_id`
  - subgroup label + `reason`
- [ ] Record the exact detect mapping table to use later:
  - `manual_review`
  - `auto`
  - `interpolated`
  - `unknown`

## Keypoints

### Current Source Signals Confirmed

- Refined keypoint runs already expose strong row-level candidates:
  `detection_source`, `retune_id`, `quality_labels`, `refined_success`,
  `source_success`, `usable_keypoints`, `reason_bytes`, and `reason` in
  [stage_arrays.py](/home/delahantyj@hhmi.org/gitrepos/palette/src/fisheye/shared/zarr/stage_arrays.py).
- Base refinement writes `reason` tags such as `clean`, `detection_failed`,
  `flip_corrected`, `confidence_missing`, `low_confidence`, and
  `geometry_issue` in
  [refine_keypoints.py](/home/delahantyj@hhmi.org/gitrepos/palette/src/fisheye/refinement/refine_keypoints.py).
- Manual keypoint tuning writes `retune_id` on updated rows and merges new tags
  into `reason`; it also appends `detection_issue` on failed/manual-review
  escalation paths in
  [keypoint_tuner.py](/home/delahantyj@hhmi.org/gitrepos/palette/src/fisheye/tune/keypoint_tuner.py).
- Review summaries currently look for `manual_correction` in reason counts in
  [keypoint_review.py](/home/delahantyj@hhmi.org/gitrepos/palette/src/fisheye/tune/keypoint_review.py),
  so Phase 1 must verify whether that token still exists in real data or whether
  the summary logic has drifted from the writer.
- The merged keypoint exporter preserves source lineage, `detection_source`, and
  `keypoint_box_only`, but not `reason` or `retune_id`, in
  [export_keypoint_training_zarr.py](/home/delahantyj@hhmi.org/gitrepos/palette/src/fisheye/utils/export_keypoint_training_zarr.py).

### Preliminary Assessment

- `interpolated` is likely recoverable row-level through `detection_source`.
- Box-only supervision is already recoverable through `keypoint_box_only`; that
  should feed `supervision_mode`, not `label_origin`.
- Manual tuning is likely recoverable at the source refined-run level via
  `retune_id` and possibly `reason`, but the merged export currently drops that
  information.
- There is a likely vocabulary mismatch to resolve:
  `keypoint_review.py` counts `manual_correction`, while the current tuner logic
  obviously appends `detection_issue` and preserves/merges existing tags.

### Audit Checklist

- [ ] Inspect real refined keypoint runs and enumerate the actual `reason`
  vocabulary present after:
  - base refinement
  - manual keypoint tuning
  - manual failure escalation
- [ ] Confirm whether `retune_id > -1` reliably marks manually tuned rows.
- [ ] Confirm whether manual tuning adds a dedicated provenance tag to `reason`,
  or only modifies diagnostic tags.
- [ ] Resolve the `manual_correction` vs `detection_issue` vocabulary question:
  - writer truth
  - reviewer summary truth
  - historical-data compatibility
- [ ] Confirm whether `detection_source == 1` is preserved on interpolated
  source rows after manual keypoint tuning.
- [ ] Confirm which field should be authoritative for `label_origin`:
  - `retune_id`
  - `reason`
  - `detection_source`
  - some combination
- [ ] Confirm `keypoint_box_only` should map to
  `supervision_mode=box_only` independently of `label_origin`.

## Eye Masks

### Current Source Signals Confirmed

- Eye-mask export can source from `eye_masks_runs` or `refined_eye_masks_runs`
  via the selected stage/run in
  [prepare_eye_mask_training_from_registry.py](/home/delahantyj@hhmi.org/gitrepos/palette/src/fisheye/utils/prepare_eye_mask_training_from_registry.py)
  and
  [export_eye_mask_training_zarr.py](/home/delahantyj@hhmi.org/gitrepos/palette/src/fisheye/utils/export_eye_mask_training_zarr.py).
- The exporter explicitly resolves row-level `reason` labels using this fallback:
  metrics-group reason -> run-group reason -> `detection_source`-derived labels,
  in
  [export_eye_mask_training_zarr.py](/home/delahantyj@hhmi.org/gitrepos/palette/src/fisheye/utils/export_eye_mask_training_zarr.py)
  and
  [detect_reason_codec.py](/home/delahantyj@hhmi.org/gitrepos/palette/src/fisheye/shared/detect_reason_codec.py).
- Explicit negatives are already defined operationally via the
  `fish_present_no_keypoints` reason tag in
  [export_eye_mask_training_zarr.py](/home/delahantyj@hhmi.org/gitrepos/palette/src/fisheye/utils/export_eye_mask_training_zarr.py).
- Refined eye-mask tuning writes `retune_id` and merges `retuned` into `reason`
  on updated rows in
  [eye_mask_tuner.py](/home/delahantyj@hhmi.org/gitrepos/palette/src/fisheye/tune/eye_mask_tuner.py).
- The merged eye-mask exporter already preserves `detection_source` and
  `reason`, but not `retune_id`, in
  [export_eye_mask_training_zarr.py](/home/delahantyj@hhmi.org/gitrepos/palette/src/fisheye/utils/export_eye_mask_training_zarr.py).

### Preliminary Assessment

- `interpolated` is recoverable row-level today.
- Explicit-negative supervision is already recoverable row-level today via
  `reason` containing `fish_present_no_keypoints`.
- Manual retunes are likely recoverable in refined-eye-mask source runs via
  `retune_id` and/or `reason=...|retuned`, but that needs to be verified across
  actual refined-eye-mask archives.
- Eye masks are currently ahead of the other tasks on `supervision_mode`, but
  still do not emit a normalized row-level `label_origin`.

### Audit Checklist

- [ ] Inspect real `refined_eye_masks_runs` examples and confirm `retune_id > -1`
  marks manually retuned rows.
- [ ] Enumerate the actual refined eye-mask `reason` vocabulary, especially the
  presence/absence of `retuned`.
- [ ] Confirm whether the exporter selects `metrics/reason` when present and
  whether that preserves manual-retune provenance.
- [ ] Confirm whether explicit-negative rows always carry the
  `fish_present_no_keypoints` tag when included.
- [ ] Decide whether eye-mask `label_origin` should be derived from:
  - `retune_id`
  - `reason`
  - `detection_source`
  - source stage (`eye_masks_runs` vs `refined_eye_masks_runs`)
- [ ] Confirm `supervision_mode=explicit_negative` mapping from the explicit
  negative tag.

## Phase 1 Exit Criteria

Phase 1 is complete when each task has:

- one documented authoritative row-level signal, or an explicit statement that
  row-level provenance does not exist today
- one documented fallback chain
- one documented ambiguity list that Phase 2 must not paper over
- one proposed mapping table from current source signals to
  `label_origin` / `supervision_mode`

## Recommended Deliverables After This Audit

- a small contract doc for normalized merged-training row provenance
- exporter changes only for signals that are actually trustworthy
- no guessed backfill logic for cases where the source data is ambiguous

## Related Docs

- [training_label_origin_provenance_todo.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/training_label_origin_provenance_todo.md)
- [provenance_todo.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/provenance_todo.md)
- [detection_merged_export_contract.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/detection_merged_export_contract.md)
- [keypoint_merged_row_gate_contract.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/keypoint_merged_row_gate_contract.md)
- [provenance_workflow.md](/home/delahantyj@hhmi.org/gitrepos/palette/src/fisheye/docs/provenance_workflow.md)
