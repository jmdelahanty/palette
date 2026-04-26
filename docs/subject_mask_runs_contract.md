# Subject Mask Runs Contract (Draft v1)
<!-- contract-meta
version: 1
status: draft
last_verified: 2026-04-25
-->

Purpose: define the runtime/storage contract for a generalized ROI-local
multilabel mask stage that can represent subject body, eyes, and swim bladder
without overloading the existing eye-specific stages.

## Scope

- Define `subject_mask_runs/<run>` for runtime and curated source zarrs.
- Support model-native subject-mask runs.
- Support explicit projection/backfill from legacy `eye_masks_runs` or
  `refined_eye_masks_runs`.
- Keep refined geometry and review authority out of raw `subject_mask_runs`.
  Modern refined eye geometry belongs in `refined_subject_masks_runs` when
  `eye_left` and `eye_right` are present; `refined_eye_masks_runs` remains a
  compatibility/historical layout.

## Non-goals

- Replacing `refined_eye_masks_runs`.
- Storing eye ellipses, eye separation, or eye-specific QA summaries here.
- Storing subject centerlines, splines, or tail kinematics here.
- Storing thresholded model masks as canonical raw model output.
- Defining merged training dataset layout. See
  [subject_mask_training_artifact_contract.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/subject_mask_training_artifact_contract.md).

## Intended Stage Relationship

```text
crop_runs/<run>
  -> subject_mask_runs/<run>
  -> refined_subject_masks_runs/<run>  # canonical refined component masks
  -> subject_shape_runs/<run>          # future deterministic geometry stage
```

Legacy compatibility path:

```text
eye_masks_runs/<run> or refined_eye_masks_runs/<run>
  -> subject_mask_runs/<run>           # explicit projection/backfill
  -> refined_subject_masks_runs/<run>   # canonical refined target
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
   Example: current U-Net subject-mask models that emit `subject_body`,
   `eyes_union`, and `swim_bladder` together.
3. Component-scoped runs
   Example: future workflows where an operator or model only materializes one
   subset of the canonical labels.

V1 fully supports cases 1 and 2. The shipped U-Net subject-mask inference path
writes probability-first dense multi-component `subject_v1_union` runs with
`mask_probs_roi` as the canonical raw model output.

Case 3 is intentionally deferred, but the contract is shaped to allow it later
through:

- canonical `mask_labels`
- run-level `available_channels`
- explicit `label_schema_id`

So the same stage family can represent eye-only runs now and fuller subject
segmentation later without another schema reset.

## Output Layout

Native raw model-output runs should store probability surfaces as the
operator-independent raw evidence. Thresholded or morphologically repaired masks
belong in `refined_subject_masks_runs/<run>` after refinement/finalization, not
as canonical payload in the raw run.

```text
subject_mask_runs/
  attrs:
    latest                     "<run_id>"
  <run_id>/
    frame_indices              (N,) int32           # new runs should include
    frame_counts               (F,) int32           # new runs should include
    detection_indices          (N,) int32           # new runs should include
    detection_source           (N,) int8
    mask_probs_roi             (N, C, H, W) float16/float32/uint8
    available_channels         (C,) bool
    metrics/
      prob_max                 (N, C) float32
      probability_present      (N, C) bool          # recommended
    components/
      <component_name>/
        provenance/            # attrs-only subgroup describing component origin
        metrics/               # optional component-local extension point
```

## `subject_mask_runs/<latest>`

Required arrays:

- `detection_source`
  - shape: `(N,)`
  - expected to match the source crop run
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

Compatibility arrays:

- `masks_roi`
  - allowed for legacy projection/backfill runs and existing raw producers
    during migration
  - if present in a native raw model-output run, it is a derived compatibility
    cache, not the canonical raw evidence
  - future native raw model-output writers should omit it
  - threshold policy attrs must make any generated cache reproducible

Recommended `metrics/` subgroup arrays:

- `probability_present`
  - shape: `(N, C)`
  - true when the probability surface has meaningful foreground evidence under
    the run's declared probability semantics
- `mask_present`
  - shape: `(N, C)`
  - compatibility/cache metric derived from a binary compatibility mask or an
    explicitly recorded threshold policy
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
- `source_crop_storage_mode`
- `source_crop_signature`
- `source_crop_revision`
- `label_schema_id`
- `mask_labels`
- `output_semantics = "multilabel"`
- `overlap_policy = "independent_sigmoid"`
- `method`
- `run_semantics`
- `duration_seconds`

Required when the source crop run exposes detect-review linkage:

- `source_detect_review_status_ref`

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
- `assignment_keypoints_run`
- `assignment_keypoint_group`
- `assignment_keypoint_contract = "subject_eyes_union_assignment_keypoints_v1"`
- `assignment_keypoint_role = "eyes_union_lr_assignment"`
- `assignment_keypoint_selection`
- `source_eye_masks_run`
- `source_refined_eye_masks_run`
- `source_subject_mask_run`
- `projection_mode`
- `model_info`
- `thresholds_by_label`
  - required only when a writer materializes optional compatibility binary masks
    from `mask_probs_roi`
- `summary_statistics`

Crop-snapshot semantics:

- `source_crop_run` + `source_crop_storage_mode` + `source_crop_signature` +
  `source_crop_revision` form the portable crop snapshot for downstream ROI
  consumers.
- `source_detect_review_status_ref` remains a separate stable lineage field and
  must not be folded into `source_crop_signature`.
- Downstream writers should preserve this crop snapshot contract rather than
  re-deriving it ad hoc from the latest crop run.
- Current `refined_subject_masks_runs/<run>` writers carry the same crop
  snapshot fields forward from their `subject_mask_runs/<run>` source.

Assignment-keypoint semantics:

- `source_keypoints_run` / `source_keypoint_group` mean the raw
  `subject_mask_runs/<run>` producer consumed that keypoint run as an input.
- `assignment_keypoints_run` / `assignment_keypoint_group` mean the run has a
  declared keypoint source for later deterministic post-segmentation
  assignment, currently `eyes_union` -> `eye_left` / `eye_right`.
- raw U-Net subject-mask inference does not need keypoints to segment masks, so
  it should record `assignment_*` attrs rather than pretending keypoints were a
  raw segmentation input.
- `assignment_keypoint_group` and `assignment_keypoints_run` must be written as
  a pair and must row-align with the source crop run.

## Component Provenance

`components/<component>/provenance` is the canonical component-local lineage
record for `subject_mask_runs/<run>`.

Required attrs for populated component provenance:

- `source_stage`
- `source_run`
- `source_method`
- `source_channels`
- `source_crop_run`
- `source_crop_storage_mode`
- `source_crop_signature`
- `source_crop_revision`

Required when the crop source exposes detect-review linkage:

- `source_detect_review_status_ref`

Semantics:

- `source_*` identifies the upstream artifact that seeded the component-local
  mask payload.
- The crop snapshot fields identify the crop surface that component was derived
  from.
- Component provenance stays component-local; it does not replace the run-level
  crop snapshot attrs.

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
- native raw output stores `sigmoid(selected_candidate_logits)`, not a directly
  emitted calibrated semantic probability map from SAM
- `metrics/sam_quality_score` stores the selected candidate's separate quality
  score and should be interpreted separately from `mask_probs_roi`
- existing SAM/raw writers may still materialize `masks_roi` as a compatibility
  cache thresholded at logit `> 0`; that cache is not the canonical raw
  artifact and should be moved behind refined/finalized outputs over time

Interpretation:

- `method` names the concrete algorithm or tuning family used to produce the
  run
- `run_semantics` names the artifact meaning
- `probability_semantics` tells readers how to interpret `mask_probs_roi`

This distinction matters because multiple methods may produce the same kind of
raw subject-mask run, while downstream tools and registry views often need to
query the artifact meaning rather than the exact algorithm label.

### U-Net subject-mask inference attrs

When a `subject_mask_runs/<run>` entry is produced by the unified U-Net
subject-mask inference path, the run should also record:

- `probability_semantics = "sigmoid_multilabel_logits"`
- `unet_checkpoint_path`
- structured `model_info` when available
- the crop-read provenance attrs already used by other ROI-driven stages, such
  as `source_crop_storage_mode` and `source_roi_read_mode`

Recommended `run_semantics` for the shipped path:

- `unet_subject_mask_inference`

Current implementation note:

- the shipped U-Net subject-mask path currently supports
  `label_schema_id = "subject_v1_union"` only
- it predicts `["subject_body", "eyes_union", "swim_bladder"]` together
- fully supervised checkpoints may write
  `available_channels = [true, true, true]`
- partially supervised checkpoints must still preserve the canonical schema,
  but any channel without semantic support in that checkpoint should be written
  as an unavailable placeholder:
  - `available_channels[c] = false`
  - decoded `mask_probs_roi[:, c] = 0`
  - optional compatibility `masks_roi[:, c] = 0` if that cache exists
- future U-Net variants may support other schemas such as `subject_v1_lr`, but
  readers must continue relying on `label_schema_id` and `mask_labels`, not
  channel position assumptions

Default analysis-inference mode:

- U-Net subject-mask inference is probability-first by default:
  `mask_probs_roi` is the canonical raw model output and thresholded binary
  `masks_roi` is not materialized unless `--write-masks-roi` is passed
- async output is enabled by default to overlap GPU inference with CPU
  spatial-metric derivation and Zarr writes through a bounded output queue
- `--output-queue-size 2` is the conservative default; larger values can
  increase memory pressure because each queued item contains dense probability
  and thresholded working-mask batches
- Rich progress rendering is disabled by default for agent/log-friendly runs;
  pass `--progress` for interactive progress output
- runs produced this way should record `masks_roi_materialized`,
  `async_output`, and `output_queue_size` attrs so readers and benchmarks know
  which surfaces were materialized
- compatibility/debug options remain available:
  - `--write-masks-roi` to materialize the dense binary threshold cache
  - `--no-async-output` to force the serial writer path
  - `--progress` for interactive terminal progress

Recommended operator command shape:

```bash
scripts/py -m fisheye.segmentation.infer_unet_subject_masks \
  <analysis.zarr> \
  --resolve-model-from-registry \
  --registry /nvme1/palette_registry.sqlite \
  --model-coverage-class dense_all_components \
  --model-component-coverage-key body+eyes+swim_bladder \
  --model-label-schema-id subject_v1_union \
  --crop-run <crop_run> \
  --assignment-keypoint-group refined_keypoints_runs \
  --assignment-keypoint-run <refined_keypoints_run> \
  --device 0 \
  --batch-size 128 \
  --mask-probs-dtype uint8 \
  --mask-probs-chunk-rois 32
```

Observed canary timings on `2026-01-28T23-15-10Z_arena_4_Feeding`
(`19,235` ROIs, `512x512`, 3 channels, `uint8` probabilities, CUDA device 0):

- dense probabilities plus dense `masks_roi`, profiled: `170.4s`
- probability-only without async output: `135.3s`
- probability-only async default output mode: `100.2s`
- probability-only async with on-device spatial metrics on arena 2:
  `89.8s`; `metric_compute` dropped from `30.1s` to `0.003s` in the
  profiled comparison run

Treat these numbers as local workstation/storage guidance, not a contract.
The contract is the artifact behavior: probability-first raw runs may omit
`masks_roi`, and async output must not change array contents or row alignment.

KvikIO / GPUDirect Storage note:

- KvikIO/GDS is experimental for this stage and is not an operator backend yet.
- Use `scripts/py -m fisheye.diagnostics.benchmark_kvikio_gds` outside the
  Codex sandbox before considering a GDS-backed writer on a workstation.
- Current local findings and backend criteria are tracked in
  `docs/kvikio_gds_subject_mask_experiment.md`.

## `available_channels` semantics

`available_channels` is a runtime/source-stage availability declaration, not a
training supervision mask.

Meaning:

- `available_channels[c] == true` means channel `c` contains semantically valid
  predictions or projected labels in this run
- `available_channels[c] == false` means channel `c` is a placeholder channel
  and readers must not treat it as a true negative

Required invariant:

- if `available_channels[c] == false`, then decoded `mask_probs_roi[:, c]` must
  be all-zero
- if an optional compatibility `masks_roi` array exists and
  `available_channels[c] == false`, then `masks_roi[:, c]` must be all-zero

This is intentionally different from training
`target_valid_channels`, which is row-level supervision metadata.

Operational note:

- unavailable channels may represent compatibility backfills today
- intentionally component-scoped runs in the future
- or partially supervised model outputs whose checkpoint does not provide
  semantically trustworthy predictions for every schema channel

In both cases, readers must treat them as unavailable rather than absent.

## Component-Scoped Provenance

Run-level `source_*_run` attrs remain useful coarse lineage pointers, but they
are not sufficient once one `subject_mask_runs/<run>` entry may contain
components materialized from different upstream stages or channels.

Canonical home:

- `components/<component_name>/provenance/`

The provenance subgroup should be attrs-only in v1 unless a later contract
needs per-row component lineage.

Writers should persist component-scoped provenance for every semantically
available component. This becomes effectively required for:

- mixed-source runs
- projection/backfill runs
- future unified runs that combine components from different model families

Required provenance attrs for an available component:

- `source_stage`
  - source stage family such as `subject_mask_runs`, `eye_masks_runs`,
    `refined_eye_masks_runs`
- `source_run`
  - source run id within that stage family
- `source_method`
  - upstream run `method` used to generate the component payload
- `source_channels`
  - list of source channel names used to generate this component

Recommended provenance attrs:

- `source_label_schema_id`
  - the source run's `label_schema_id`
- `projection_mode`
  - required when this component is projected or collapsed from another schema
- `source_created_at_utc`
  - source run creation timestamp when available

Semantics:

- `source_channels` is a list because some projections consume multiple source
  channels
- for single-channel lineage, writers should still use a one-element list
- use anatomical names such as `eye_left` / `eye_right` only when the source
  artifact explicitly carries anatomical left/right identity
- if the source artifact is only an unlabeled positional pair, writers should
  use positional channel identifiers such as `channel_0` / `channel_1` instead
  of anatomical names
- for placeholder components where `available_channels[c] == false`, the
  provenance subgroup may be omitted

Examples:

- A SAM3 body channel should typically record:
  - `source_stage = "subject_mask_runs"`
  - `source_run = "sam_subject_masks_..."`
  - `source_method = "sam_body_mask_inference"`
  - `source_channels = ["subject_body"]`
- An `eyes_union` compatibility projection from left/right eyes should record:
  - `source_stage = "refined_eye_masks_runs"`
  - `source_channels = ["eye_left", "eye_right"]`
  - `projection_mode = "eyes_union_from_lr"`

## Legacy projection/backfill policy

Legacy `eye_masks_runs` and `refined_eye_masks_runs` remain supported source
stages.

Recommended migration model:

1. Preserve historical `eye_masks_runs` and `refined_eye_masks_runs`.
2. Allow explicit projection/backfill into `subject_mask_runs`.
3. Keep `refined_eye_masks_runs` as the eye-specific derived/compatibility
   stage.
4. Prefer new eye refinement through `subject_mask_runs` plus declared
   assignment keypoint lineage, finalized into `refined_subject_masks_runs`.
5. Only then deprecate creation of new raw `eye_masks_runs`.

Current implementation note:

- the explicit migration utility/backfill path records
  `run_semantics = "legacy_eye_mask_projection"`
- `scripts/py -m fisheye.utils.backfill_subject_mask_runs --source-stage prefer_refined`
  now provides a one-pass migration mode that prefers `refined_eye_masks_runs`
  and falls back to `eye_masks_runs`
- raw eye orchestration may also materialize an immediate compatibility
  `subject_mask_runs/<run>` companion after successful `eye_masks_runs/<run>`
  completion
- that fresh runtime companion records
  `run_semantics = "eye_mask_runtime_projection"`
- current runtime eye projection defaults to `subject_v1_union`, even when the
  eye producer may carry richer left/right semantics, so the canonical raw
  bridge remains safe across traditional, YOLO, and U-Net eye producers

## Projection rules from legacy eye-mask stages

Legacy projection/backfill runs may include binary `masks_roi` because the
source artifact is already a binary mask surface. These arrays are
compatibility labels/caches, not native raw model-output masks.

### Projection into `subject_v1_union`

If the source stage is `eye_masks_runs` or `refined_eye_masks_runs` and the
target schema is `subject_v1_union`:

- if source channels are `["eye_left", "eye_right"]`:
  - optional compatibility `masks_roi[:, eyes_union] = union(left, right)`
  - `mask_probs_roi[:, eyes_union] = max(prob_left, prob_right)`
  - `projection_mode = "eyes_union_from_lr"`
- if source channels are a non-anatomical two-eye pair (for example
  `["eye_0", "eye_1"]`):
  - optional compatibility `masks_roi[:, eyes_union] = union(channel_0, channel_1)`
  - `mask_probs_roi[:, eyes_union] = max(prob_0, prob_1)`
  - `components/eyes_union/provenance/source_channels = ["channel_0", "channel_1"]`
  - `projection_mode = "eyes_union_from_pair"`
- if source already provides an eye union channel:
  - copy that channel directly
  - `components/eyes_union/provenance/source_channels = ["eyes_union"]`
  - `projection_mode = "eyes_union_from_union"`

In both cases:

- `available_channels = [false, true, false]`
- `subject_body` probability payload is written as a zero placeholder channel
- `swim_bladder` probability payload is written as a zero placeholder channel
- attrs should record the source eye stage and source run name

### Projection into `subject_v1_lr`

If the source stage is `eye_masks_runs` or `refined_eye_masks_runs` and the
target schema is `subject_v1_lr`:

- source channels must carry anatomical left/right identity
- optional compatibility `masks_roi[:, eye_left] = source_left`
- optional compatibility `masks_roi[:, eye_right] = source_right`
- `components/eye_left/provenance/source_channels = ["eye_left"]`
- `components/eye_right/provenance/source_channels = ["eye_right"]`
- if probability masks also preserve left/right identity:
  - copy them channel-for-channel
- otherwise:
  - write binary-mask probability fallback derived from the refined/source masks
  - `probabilities_encoding = "unit_float"`
- `projection_mode = "eye_lr_from_lr"`

In both cases:

- `available_channels = [false, true, true, false]`
- `subject_body` probability payload is written as a zero placeholder channel
- `swim_bladder` probability payload is written as a zero placeholder channel
- writers must reject `subject_v1_lr` projection from unlabeled two-eye pair
  channels such as `["eye_0", "eye_1"]`

## Reader contract

- `mask_probs_roi` is ROI-local, not full-frame.
- Readers should decode `mask_probs_roi` into semantic probabilities in
  `[0, 1]` regardless of physical dtype.
- Native raw model-output readers should not require `masks_roi`; thresholding
  belongs in an explicit refinement/finalization step.
- If optional compatibility `masks_roi` exists, readers should treat it as a
  derived cache or projected label surface, not as the raw model authority.
- Readers must consult `available_channels` before interpreting an all-zero
  channel as biological absence.
- Full-frame placement must be derived from the source crop geometry.

## Related Contracts

- [subject_mask_training_artifact_contract.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/subject_mask_training_artifact_contract.md)
- [eye_mask_training_artifact_contract.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/eye_mask_training_artifact_contract.md)
