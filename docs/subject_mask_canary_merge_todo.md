# Subject Mask Canary Merge TODO

## Goal

Create one canonical raw `subject_mask_runs/<run>` entry for the canary archive
that combines:

- body from the current SAM subject-mask run
- eyes from the projected refined-eye subject-mask run
- swim bladder present in schema but unavailable

This is the next concrete step toward the unified subject-mask direction
described in
[subject_mask_stage_unification_todo.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/subject_mask_stage_unification_todo.md).

## Why This Step Is Needed

The canary archive currently has the right pieces, but not yet the right
canonical raw artifact:

- one `subject_mask_runs/<run>` that is body-only
- one `subject_mask_runs/<run>` that is eye-only

That is acceptable during migration, but it is not the desired steady state.

We want the canary to prove the future authoring model:

- one canonical raw subject-mask run
- component provenance recorded in metadata
- no need for downstream tools to merge body and eyes on every read

## Current Canary Inputs

Primary archive:

- `/nvme1/recordings/2026-01-28T22-15-03Z_arena_1_DefaultScreen/zarr/2026-01-28T22-15-03Z_arena_1_DefaultScreen_training.zarr`

Known raw inputs:

- body source:
  - `subject_mask_runs/sam_subject_masks_canary_001`
- eye source:
  - `subject_mask_runs/subject_masks_from_refined_eye_masks_2026-02-12_19-51-24`

Expected semantics:

- body source currently provides `subject_body`
- eye source currently provides `eye_left` and `eye_right`
- swim bladder is not yet populated and should remain unavailable

## Desired Output

Create one new raw run under `subject_mask_runs/` with:

- `label_schema_id = "subject_v1_lr"`
- `mask_labels = ["subject_body", "eye_left", "eye_right", "swim_bladder"]`
- `available_channels = [true, true, true, false]`

Channel meaning:

- `subject_body` copied from the body source run
- `eye_left` copied from the eye source run
- `eye_right` copied from the eye source run
- `swim_bladder` left unavailable

Important semantic rule:

- unavailable `swim_bladder` is not a negative label
- it must remain zero-filled/implicit-zero with `available_channels=false`

## Non-goals

- changing historical `eye_masks_runs` or `refined_eye_masks_runs`
- moving refined eye geometry into `refined_subject_masks_runs`
- introducing a full generalized component-merging framework for all future
  cases before the canary is proven
- automatically creating a refined run as part of this step

## Merge Policy

The merge target is a raw run, not a refined run.

That means:

- this step creates a new `subject_mask_runs/<run>`
- the result is snapshot-like and immutable once written
- future editing should happen against `refined_subject_masks_runs`

## Required Preconditions

Before writing the merged run, validate:

- same `source_crop_run`
- same row count
- same ROI spatial shape
- same `frame_indices`
- same `detection_indices`
- same `detection_source`

If any of those fail, the merge should stop with a clear error.

## Required Arrays

The merged run should write or preserve:

- `frame_indices`
- `frame_counts`
- `detection_indices`
- `detection_source`
- `masks_roi`
- `mask_probs_roi`
- `available_channels`
- `metrics/prob_max`
- `metrics/mask_present`

Recommended if available:

- `metrics/area_px`
- `metrics/centroid_xy`
- `metrics/centroid_valid`
- `metrics/bbox_xyxy`
- `metrics/bbox_valid`

## Probability Policy

Use component-native probabilities when they exist.

For this canary:

- body probability should come from the SAM body source
- eye probabilities should come from the eye-derived source
- swim bladder remains implicit zero

If a source component lacks probabilities:

- fall back to binary mask semantics for that component
- record the probability provenance clearly in attrs

## Storage Policy

The merged run should keep the canonical fixed channel schema while avoiding
needless materialization of unavailable channels.

That means:

- channel axis remains present in `masks_roi` and `mask_probs_roi`
- unavailable channels use zero fill semantics
- chunking should keep channel chunk size `1`
- only available channel slices need to be written explicitly

This should match the newer subject-mask writer behavior rather than the older
whole-array materialization path.

## Provenance Policy

Do not encode component ancestry into the run name.

The merged run name should describe the artifact itself, for example:

- `subject_masks_canary_body_eyes_001`

Component provenance should instead live in attrs, ideally under a
component-scoped payload such as:

```json
{
  "components": {
    "subject_body": {
      "source_stage": "subject_mask_runs",
      "source_run": "sam_subject_masks_canary_001",
      "source_channel": "subject_body"
    },
    "eye_left": {
      "source_stage": "subject_mask_runs",
      "source_run": "subject_masks_from_refined_eye_masks_2026-02-12_19-51-24",
      "source_channel": "eye_left"
    },
    "eye_right": {
      "source_stage": "subject_mask_runs",
      "source_run": "subject_masks_from_refined_eye_masks_2026-02-12_19-51-24",
      "source_channel": "eye_right"
    },
    "swim_bladder": {
      "source_stage": null,
      "source_run": null,
      "source_channel": "swim_bladder"
    }
  }
}
```

Exact attr naming can still be finalized, but the provenance needs to be
component-scoped.

## Merge Utility Direction

Recommended near-term implementation:

- add a small Palette utility dedicated to merging component-compatible
  `subject_mask_runs`
- keep it explicit rather than trying to fold this into the eye-mask backfill
  utility

Suggested inputs:

- zarr path
- body source run
- eye source run
- target run name
- optional `--overwrite`

Suggested behavior:

1. open both source runs
2. validate alignment and schema compatibility
3. allocate the target `subject_v1_lr` run
4. copy body channel from the body source
5. copy `eye_left` / `eye_right` from the eye source
6. leave swim bladder unavailable
7. compute run metrics and provenance attrs
8. write the merged run and set parent `latest`

## Acceptance Criteria

This step is done when:

- the canary archive contains one merged raw run with:
  - body available
  - left eye available
  - right eye available
  - swim bladder unavailable
- the merged run validates against the runtime subject-mask contract
- the merged run is row-aligned with the canary crop/keypoint lineage
- downstream tools can open one raw subject-mask run instead of separately
  resolving body and eye runs

## Follow-up After The Merge

Once the merged raw run exists and validates:

1. use it as the preferred raw source for the canary
2. use `refined_subject_masks_runs` as the corresponding canonical editable
   target
3. leave the eye-only projected run in place only as a migration artifact
4. decide whether to backfill or compose additional canaries the same way

## Open Questions

1. Should the merged canary run become the archive `subject_mask_runs.attrs["latest"]`
   immediately, or should that wait until one round of downstream validation?
2. Should the merged utility require `subject_v1_lr` eye channels, or also
   accept a union-eye source and collapse it into `eye_left` / `eye_right`
   placeholders only for special cases?
3. Do we want component provenance recorded as attrs immediately in this step,
   or is a temporary simpler payload acceptable for the first canary merge?
