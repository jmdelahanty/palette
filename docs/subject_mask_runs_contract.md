# Subject Mask Runs Contract (Draft v1)
<!-- contract-meta
version: 1
status: draft
last_verified: 2026-03-10
-->

Purpose: define the runtime/storage contract for a generalized ROI-local
multilabel mask stage that can represent subject body, eyes, and swim bladder
without overloading the existing eye-specific stages.

## Scope

- Define `subject_mask_runs/<run>` for runtime and curated source zarrs.
- Support model-native subject-mask runs.
- Support explicit projection/backfill from legacy `eye_masks_runs` or
  `refined_eye_masks_runs`.
- Keep eye-specific refinement and geometry in `refined_eye_masks_runs`.

## Non-goals

- Replacing `refined_eye_masks_runs`.
- Storing eye ellipses, eye separation, or eye-specific QA summaries here.
- Storing subject centerlines, splines, or tail kinematics here.
- Defining merged training dataset layout. See
  [subject_mask_training_artifact_contract.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/subject_mask_training_artifact_contract.md).

## Intended Stage Relationship

```text
crop_runs/<run>
  -> subject_mask_runs/<run>
  -> refined_eye_masks_runs/<run>      # eye-specific refinement
  -> subject_shape_runs/<run>          # future deterministic geometry stage
```

Legacy compatibility path:

```text
eye_masks_runs/<run> or refined_eye_masks_runs/<run>
  -> subject_mask_runs/<run>           # explicit projection/backfill
```

## Canonical Label Schema

Recommended default schema for v1:

- `label_schema_id = "subject_v1_union"`
- `mask_labels = ["subject_body", "eyes_union", "swim_bladder"]`

Future schemas may extend this contract, for example:

- `subject_v1_lr`
- `mask_labels = ["subject_body", "eye_left", "eye_right", "swim_bladder"]`

Operational note:

- model-native inference may still prefer `subject_v1_union`
- curated/training-oriented backfills should preserve `subject_v1_lr` whenever
  the source stage provides anatomical left/right eye semantics

Readers must never infer channel meaning from channel index alone.

## Evolution Policy

This contract is intended to support three different runtime shapes over time:

1. Sparse compatibility runs
   Example: legacy eye-mask backfills where only `eyes_union` is available.
2. Dense multi-component runs
   Example: future subject-mask models that emit `subject_body`,
   `eyes_union`, and `swim_bladder` together.
3. Component-scoped runs
   Example: future workflows where an operator or model only materializes one
   subset of the canonical labels.

V1 fully supports cases 1 and 2.

Case 3 is intentionally deferred, but the contract is shaped to allow it later
through:

- canonical `mask_labels`
- run-level `available_channels`
- explicit `label_schema_id`

So the same stage family can represent eye-only runs now and fuller subject
segmentation later without another schema reset.

## Output Layout

```text
subject_mask_runs/
  attrs:
    latest                     "<run_id>"
  <run_id>/
    frame_indices              (N,) int32           # new runs should include
    frame_counts               (F,) int32           # new runs should include
    detection_indices          (N,) int32           # new runs should include
    detection_source           (N,) int8
    masks_roi                  (N, C, H, W) uint8
    mask_probs_roi             (N, C, H, W) float16/float32/uint8
    available_channels         (C,) bool
    metrics/
      prob_max                 (N, C) float32
      mask_present             (N, C) bool
```

## `subject_mask_runs/<latest>`

Required arrays:

- `detection_source`
  - shape: `(N,)`
  - expected to match the source crop run
- `masks_roi`
  - shape: `(N, C, H, W)`
  - binary values `{0, 1}`
- `mask_probs_roi`
  - shape: `(N, C, H, W)`
  - finite semantic probabilities in `[0, 1]`
  - physical dtype may be `float16`, `float32`, or quantized `uint8`
- `available_channels`
  - shape: `(C,)`
  - run-level boolean declaration of which channels contain semantically valid
    data in this run

Recommended lineage arrays:

- `frame_indices`
- `frame_counts`
- `detection_indices`

Required `metrics/` subgroup arrays:

- `prob_max`
  - shape: `(N, C)`
  - per-row per-channel maximum decoded probability in `[0, 1]`
- `mask_present`
  - shape: `(N, C)`
  - true when the binary mask contains at least one positive pixel

Recommended `metrics/` subgroup arrays:

- `area_px`
  - shape: `(N, C)`
- `centroid_xy`
  - shape: `(N, C, 2)`
- `centroid_valid`
  - shape: `(N, C)`
- `bbox_xyxy`
  - shape: `(N, C, 4)`
- `bbox_valid`
  - shape: `(N, C)`

Method-specific recommended `metrics/` arrays:

- `sam_quality_score`
  - shape: `(N, C)`
  - for SAM/SAM2/SAM3-style runs, the selected candidate's predicted mask
    quality score for each row/channel

## Required attrs

- `source_crop_run`
- `label_schema_id`
- `mask_labels`
- `output_semantics = "multilabel"`
- `overlap_policy = "independent_sigmoid"`
- `method`
- `run_semantics`
- `duration_seconds`

Required probability-storage attrs:

- `probabilities_dtype`
- `probabilities_encoding`

`probabilities_encoding` allowed values:

- `unit_float`
- `linear_uint8_0_255`

Optional attrs:

- `source_keypoints_run`
- `source_keypoint_run`
- `source_keypoint_group`
- `source_eye_masks_run`
- `source_refined_eye_masks_run`
- `source_subject_mask_run`
- `projection_mode`
- `model_info`
- `thresholds_by_label`
- `summary_statistics`

### Traditional subject-body inference attrs

When a `subject_mask_runs/<run>` entry is produced by the traditional
background-subtraction body segmenter, the run should also record:

- `source_background_run`
- `source_background_array`
- `source_dish_mask_array`
- `probability_semantics = "normalized_background_diff"`
- `tuning_source`
- `tuning_timestamp`

Recommended `run_semantics` for this path:

- `traditional_subject_body_inference`

### Traditional swim-bladder inference attrs

When a `subject_mask_runs/<run>` entry is produced by the traditional
keypoint-centered swim-bladder segmenter, the run should also record:

- `source_keypoints_run`
- `source_keypoint_run`
- `source_keypoint_group`
- `probability_semantics = "normalized_patch_darkness"`
- `tuning_source`
- `tuning_timestamp`

Recommended `run_semantics` for this path:

- `traditional_swim_bladder_inference`

### SAM subject-mask inference attrs

When a `subject_mask_runs/<run>` entry is produced by a promptable
SAM/SAM2/SAM3-style segmentation path, the run should also record:

- `probability_semantics = "sigmoid_selected_mask_logits"`
- `sam_quality_score_semantics = "predicted_mask_quality"`
- `sam_multimask_output`
- prompt-policy attrs describing the point/box prompting strategy
- checkpoint/runtime attrs such as `sam_checkpoint_path`
- structured `model_info` when available

Recommended `run_semantics` for the current body-only path:

- `sam_body_mask_inference`

Interpretation:

- SAM returns candidate mask logits and separate per-candidate quality scores
- Palette currently chooses the candidate with the highest predicted quality
  score
- `masks_roi` stores the selected candidate thresholded at logit `> 0`
- `mask_probs_roi` stores `sigmoid(selected_candidate_logits)`, not a directly
  emitted calibrated semantic probability map from SAM
- `metrics/sam_quality_score` stores the selected candidate's separate quality
  score and should be interpreted separately from `mask_probs_roi`

Interpretation:

- `method` names the concrete algorithm or tuning family used to produce the
  run
- `run_semantics` names the artifact meaning
- `probability_semantics` tells readers how to interpret `mask_probs_roi`

This distinction matters because multiple methods may produce the same kind of
raw subject-mask run, while downstream tools and registry views often need to
query the artifact meaning rather than the exact algorithm label.

## `available_channels` semantics

`available_channels` is a runtime/source-stage availability declaration, not a
training supervision mask.

Meaning:

- `available_channels[c] == true` means channel `c` contains semantically valid
  predictions or projected labels in this run
- `available_channels[c] == false` means channel `c` is a placeholder channel
  and readers must not treat it as a true negative

Required invariant:

- if `available_channels[c] == false`, then `masks_roi[:, c]` must be all-zero
- if `available_channels[c] == false`, then decoded `mask_probs_roi[:, c]` must
  be all-zero

This is intentionally different from training
`target_valid_channels`, which is row-level supervision metadata.

Operational note:

- unavailable channels may represent compatibility backfills today
- or intentionally component-scoped runs in the future

In both cases, readers must treat them as unavailable rather than absent.

## Legacy projection/backfill policy

Legacy `eye_masks_runs` and `refined_eye_masks_runs` remain supported source
stages.

Recommended migration model:

1. Preserve historical `eye_masks_runs` and `refined_eye_masks_runs`.
2. Allow explicit projection/backfill into `subject_mask_runs`.
3. Keep `refined_eye_masks_runs` as the eye-specific derived stage.
4. Move future eye refinement to read from `subject_mask_runs` plus keypoints
   when that path is stable.
5. Only then deprecate creation of new raw `eye_masks_runs`.

## Projection rules from legacy eye-mask stages

### Projection into `subject_v1_union`

If the source stage is `eye_masks_runs` or `refined_eye_masks_runs` and the
target schema is `subject_v1_union`:

- if source channels are `["eye_left", "eye_right"]`:
  - `masks_roi[:, eyes_union] = union(left, right)`
  - `mask_probs_roi[:, eyes_union] = max(prob_left, prob_right)`
  - `projection_mode = "eyes_union_from_lr"`
- if source channels are a non-anatomical two-eye pair (for example
  `["eye_0", "eye_1"]`):
  - `masks_roi[:, eyes_union] = union(channel_0, channel_1)`
  - `mask_probs_roi[:, eyes_union] = max(prob_0, prob_1)`
  - `projection_mode = "eyes_union_from_pair"`
- if source already provides an eye union channel:
  - copy that channel directly
  - `projection_mode = "eyes_union_from_union"`

In both cases:

- `available_channels = [false, true, false]`
- `subject_body` is written as a zero placeholder channel
- `swim_bladder` is written as a zero placeholder channel
- attrs should record the source eye stage and source run name

### Projection into `subject_v1_lr`

If the source stage is `eye_masks_runs` or `refined_eye_masks_runs` and the
target schema is `subject_v1_lr`:

- source channels must carry anatomical left/right identity
- `masks_roi[:, eye_left] = source_left`
- `masks_roi[:, eye_right] = source_right`
- if probability masks also preserve left/right identity:
  - copy them channel-for-channel
- otherwise:
  - write binary-mask probability fallback derived from the refined/source masks
  - `probabilities_encoding = "unit_float"`
- `projection_mode = "eye_lr_from_lr"`

In both cases:

- `available_channels = [false, true, true, false]`
- `subject_body` is written as a zero placeholder channel
- `swim_bladder` is written as a zero placeholder channel
- writers must reject `subject_v1_lr` projection from unlabeled two-eye pair
  channels such as `["eye_0", "eye_1"]`

## Reader contract

- `masks_roi` and `mask_probs_roi` are ROI-local, not full-frame.
- Readers should decode `mask_probs_roi` into semantic probabilities in
  `[0, 1]` regardless of physical dtype.
- Readers must consult `available_channels` before interpreting an all-zero
  channel as biological absence.
- Full-frame placement must be derived from the source crop geometry.

## Related Contracts

- [subject_mask_training_artifact_contract.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/subject_mask_training_artifact_contract.md)
- [eye_mask_training_artifact_contract.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/eye_mask_training_artifact_contract.md)
