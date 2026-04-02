# Segmentation Stage Split Review

<!-- review-meta
status: active
last_verified: 2026-03-28
-->

Purpose: record the current segmentation-stage layout in `palette`, confirm
that multiple raw mask authoring paths still exist, and identify the narrowest
unification seam for moving eye-only segmentation into the subject-mask stage
family.

## Executive Summary

Yes, `palette` currently has multiple runtime mask-creation paths.

- raw eye segmentation still writes to `eye_masks_runs/<run>`
- raw body, swim bladder, and SAM3 subject segmentation write to
  `subject_mask_runs/<run>`
- eye refinement still writes to `refined_eye_masks_runs/<run>`
- subject-mask review/editing writes to `refined_subject_masks_runs/<run>`

The main bridge already exists:

- `src/fisheye/utils/backfill_subject_mask_runs.py`

That module can read eye-mask stages and project them into
`subject_mask_runs/<run>` with `available_channels`, `mask_probs_roi`, and
lineage/provenance. This is the lowest-risk seam for unifying runtime storage
without breaking the eye-specific refinement and training stack.

## Current Runtime Writers

### Raw Eye-Mask Writers

| Surface | Entrypoint | Current write target | Notes |
| --- | --- | --- | --- |
| Traditional eye segmentation | `src/fisheye/segmentation/eye_segmentation.py` -> `segment_eye_masks(...)` | `eye_masks_runs/<run>` | Writes `masks_roi`, ellipses, contours, eye separation, and eye-specific provenance. |
| YOLO eye segmentation | `src/fisheye/segmentation/eye_segmentation_yolo.py` -> `segment_eye_masks_yolo(...)` | `eye_masks_runs/<run>` | Writes binary and probability masks plus eye-specific metadata. Uses `eye_labels = ["eye_0", "eye_1"]`. |
| U-Net eye segmentation | `src/fisheye/segmentation/infer_unet_eye_masks.py` -> `main(...)` | `eye_masks_runs/<run>` | Always writes `mask_probs_roi`, optionally `masks_roi`. Can emit union or LR channel layouts depending on `label_mode`. |

### Eye-Mask Orchestration That Still Targets Legacy Eye Stages

| Surface | Entrypoint | Effect |
| --- | --- | --- |
| Registry-resolved eye inference | `src/fisheye/utils/run_eye_masks_with_registry_model.py` | Resolves a model from the registry, dispatches YOLO or U-Net, then annotates the resulting `eye_masks_runs/<run>`. |
| Batch eye runner | `src/fisheye/utils/run_eye_masks_batch.py` | Dispatches `traditional`, `yolo`, or `unet`, then optionally runs refinement into `refined_eye_masks_runs/<run>`. |
| Core pipeline | `src/fisheye/core/pipeline.py` | Raw `eye_masks` stage now delegates to the shared eye orchestration path, which also materializes a unified eye-only `subject_mask_runs/<run>` companion. Refined eye masks still run as a separate stage. |

### Raw Subject-Mask Writers

| Surface | Entrypoint | Current write target | Notes |
| --- | --- | --- | --- |
| Traditional subject body segmentation | `src/fisheye/segmentation/subject_segmentation.py` -> `segment_subject_masks_from_root(...)` | `subject_mask_runs/<run>` | Body-only run. Uses `available_channels = (True, False, False)`. |
| Traditional swim bladder segmentation | `src/fisheye/segmentation/swim_bladder_segmentation.py` -> `segment_swim_bladder_masks_from_root(...)` | `subject_mask_runs/<run>` | Swim-bladder-only run. Uses `available_channels = (False, False, True)`. |
| SAM3 subject segmentation | `src/fisheye/utils/run_sam_subject_masks.py` -> `run_sam_subject_mask_inference(...)` | `subject_mask_runs/<run>` | Currently body-only in practice. |
| Eye-to-subject projection/backfill | `src/fisheye/utils/backfill_subject_mask_runs.py` -> `backfill_subject_mask_run(...)` | `subject_mask_runs/<run>` | Not fresh segmentation. Projects `eye_masks_runs` or `refined_eye_masks_runs` into the subject-mask schema. |

### Refined / Editable Writers

| Surface | Entrypoint | Current write target | Notes |
| --- | --- | --- | --- |
| Eye refinement | `src/fisheye/refinement/refine_eye_masks.py` -> `refine_eye_masks(...)` | `refined_eye_masks_runs/<run>` | Still the authoritative refined eye stage for geometry, ellipses, contours, and eye QA in v1. |
| Subject-mask review/editor | `src/fisheye/tune/refined_subject_mask_review.py` -> `prepare_refined_subject_run(...)` | `refined_subject_masks_runs/<run>` | Default new-run component selection now follows the available components in the source `subject_mask_runs` input, including eyes when present. New run creation still seeds from `subject_mask_runs` sources, not directly from `refined_eye_masks_runs`. |

Implementation note as of 2026-04-02:

- direct `refined_eye_masks_runs` -> `refined_subject_masks_runs` seeding is
  not yet shipped
- `scripts/py -m fisheye.tune.eye_mask_review --manual` now routes canonical
  eye review/edit into `refined_subject_masks_runs` through a compatibility
  `subject_mask_runs` projection, while `--legacy-manual` retains the old
  refined-eye failure-review UI
- new raw eye orchestration now dual-writes a compatibility
  `subject_mask_runs/<run>` companion immediately after successful raw
  `eye_masks_runs/<run>` completion in:
  - `src/fisheye/utils/run_eye_masks_batch.py`
  - `src/fisheye/utils/run_eye_masks_with_registry_model.py`
  - `src/fisheye/core/pipeline.py` `eye_masks` stage
- the implemented unified path for legacy eye data is:
  `eye_masks_runs` or `refined_eye_masks_runs`
  -> `subject_mask_runs/<compat_run>` via backfill/projection
  -> `refined_subject_masks_runs/<run>` via review preparation or multi-source
  assembly

## Current Mismatches

### 1. Raw Runtime Storage Still Has Two Physical Stage Families

New eye inference still lands first in `eye_masks_runs`, while new
body/swim-bladder and SAM3 inference land in `subject_mask_runs`.
However, the shipped eye orchestration now also materializes an immediate
eye-only `subject_mask_runs/<run>` companion using `subject_v1_union`.

This means downstream tooling has to know whether a component came from:

- `eye_masks_runs`
- `refined_eye_masks_runs`
- `subject_mask_runs`
- `refined_subject_masks_runs`

rather than reading one canonical runtime subject-mask snapshot.

### 2. Eye Output Semantics Are Heterogeneous

Current eye producers do not all emit the same channel semantics:

- traditional eye segmentation writes anatomical left/right semantics
- YOLO currently writes `eye_0` and `eye_1`, not guaranteed anatomical LR
- U-Net may write a single union eye channel or anatomical LR, depending on
  checkpoint `label_mode`

Because of that, the safest immediate unified storage target for eye-only
runtime writes is:

- `label_schema_id = "subject_v1_union"`
- `mask_labels = ["subject_body", "eyes_union", "swim_bladder"]`
- `available_channels = [false, true, false]`

Using `subject_v1_lr` as the default immediate bridge would be unsafe unless
the producer can guarantee anatomical left/right identity.

### 3. The Existing Subject-Mask Contracts Already Anticipate This Transition

`docs/subject_mask_runs_contract.md` already describes a migration model where:

1. historical `eye_masks_runs` and `refined_eye_masks_runs` stay supported
2. eye content can be projected into `subject_mask_runs`
3. `refined_eye_masks_runs` remains the eye-specific refined stage in v1
4. only later should new raw `eye_masks_runs` creation be deprecated

`docs/refined_subject_masks_runs_contract.md` also explicitly keeps
`refined_eye_masks_runs` authoritative for refined eye geometry and QA in v1.

### 4. Some Downstream Consumers Already Prefer `subject_mask_runs`

`src/fisheye/utils/export_subject_mask_training_zarr.py` already consumes
`subject_mask_runs` as the subject-mask training source. That means routing new
eye inference into `subject_mask_runs` immediately improves downstream
consistency even before the eye-specific refinement stack is migrated.

## Recommended Immediate Unification Slice

The narrowest safe change is:

1. keep the existing raw eye writers unchanged as producers
2. extract the projection/write logic from
   `src/fisheye/utils/backfill_subject_mask_runs.py` into a reusable helper
3. call that helper immediately after successful raw eye-mask completion in:
   - `src/fisheye/utils/run_eye_masks_batch.py`
   - `src/fisheye/utils/run_eye_masks_with_registry_model.py`
4. create a new `subject_mask_runs/<run>` eye-only snapshot using
   `subject_v1_union` by default
5. set `available_channels` so only the eye component is marked present
6. preserve source eye-run lineage on the projected subject-mask run

Status:

- shipped
- the runtime projection currently records
  `run_semantics = "eye_mask_runtime_projection"`
- explicit historical utility/backfill still records
  `run_semantics = "legacy_eye_mask_projection"`

This gives new eye inference a canonical subject-mask surface immediately while
avoiding a forced rewrite of:

- `refined_eye_masks_runs`
- eye-specific QA summaries
- ellipses
- contours
- eye separation
- left/right assignment logic
- eye-training data preparation that still depends on eye-specific stages

## Bridge Now vs Final Target

The existing TODO in `docs/subject_mask_stage_unification_todo.md` argues that
the eventual canonical runtime/refined schema should be:

- `subject_v1_lr`
- `["subject_body", "eye_left", "eye_right", "swim_bladder"]`

That can still be the longer-term destination.

However, the codebase is not ready to make that the default write target for
all new eye inference today because some current producers do not guarantee
anatomical LR identity.

Recommended interpretation:

- immediate bridge: write eye-only subject-mask runs as `subject_v1_union`
- later canonicalization: migrate all eye producers to trustworthy LR semantics
  and then evaluate switching the canonical runtime schema to `subject_v1_lr`

## What Should Stay Separate For Now

These should remain valid and first-class during the transition:

- `eye_masks_runs/<run>` as the raw eye-specific inference artifact
- `refined_eye_masks_runs/<run>` as the refined eye-specific artifact

Reason:

- the refined eye workflow still reads from `eye_masks_runs`
- eye-specific geometry and QA still live there
- current training and review tooling still depend on those eye-specific
  surfaces

So the recommended near-term model is additive, not destructive:

- keep eye-specific stages for compatibility and specialized workflows
- also write eye outputs into `subject_mask_runs` for canonical runtime subject
  storage

## Open Follow-Up Questions

1. When eye-only subject-mask runs become standard, should runtime naming move
   fully to native `subject_masks_*` identities without preserving any
   eye-stage ancestry hints?

## Related Docs

- `docs/subject_mask_runs_contract.md`
- `docs/refined_subject_masks_runs_contract.md`
- `docs/subject_mask_training_artifact_contract.md`
- `docs/subject_mask_stage_unification_todo.md`
