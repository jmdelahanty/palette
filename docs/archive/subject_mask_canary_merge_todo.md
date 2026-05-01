# Subject Mask Canary Merge TODO

<!-- design-meta
status: superseded
last_verified: 2026-04-01
-->

## Status

This TODO was superseded by the direct assembled-refined canary path.

The original plan here was to create one merged raw
`subject_mask_runs/<run>` entry for the canary archive before refinement.
That is no longer the preferred next step for sparse multi-source workflows.

## Resolved Direction

For sparse body/eye/swim assembly, the preferred workflow is now:

```text
component/raw sources
  -> refined_subject_masks_runs/<run>
     (assembly + subject-mask finalization in one command)
```

Implemented entrypoint:

- [src/fisheye/refinement/assemble_refined_subject_masks.py](/home/delahantyj@hhmi.org/gitrepos/palette/src/fisheye/refinement/assemble_refined_subject_masks.py)

The first canary assembled refined run was created on 2026-04-01:

- archive:
  - `/nvme1/recordings/2026-01-28T22-15-03Z_arena_1_DefaultScreen/zarr/2026-01-28T22-15-03Z_arena_1_DefaultScreen_training.zarr`
- body source run:
  - `subject_masks_canary_sam_points_body_eyes_001`
- eye source run:
  - `subject_masks_from_refined_eye_masks_2026-02-12_19-51-24`
- swim source run:
  - `traditional_swim_bladder_masks_canary_001`
- refined output run:
  - `refined_subject_masks_canary_body_eyes_swim_001`

## Why The Raw-Merge Plan Was Superseded

- it introduced an extra snapshot-like artifact that sparse workflows would
  immediately turn into a refined working run
- it duplicated lineage without improving the actual review/edit UX
- the refined stage already carries the component-local provenance, metrics,
  QC, reasons, and review scaffolding that the canary needs
- direct assembly plus finalization matches the semantics of other Palette
  refined artifacts better than a merged-raw-only milestone

## What Still Matters From The Old Merge Plan

The old merge TODO had valid requirements that still apply to direct assembly:

- validate source alignment:
  - `source_crop_run`
  - row count
  - ROI shape
  - `frame_indices`
  - `detection_indices`
  - `detection_source`
- keep component provenance explicit rather than encoding ancestry in run names
- preserve canonical channel identity and `available_channels`
- treat unavailable channels as unavailable, not as negative labels

Those checks now belong to the direct assembled-refined path rather than a
required raw merge step.

## Remaining Follow-Up

- continue reviewing the assembled canary at component level rather than only
  validating its creation
- keep eye editing in `refined_eye_masks_runs` during transition, even though
  assembled refined runs can already seed eye components from the eye-derived
  source run
- leave a raw merged utility as optional future work for dense raw snapshots or
  export-oriented use cases, not as the required canary milestone
