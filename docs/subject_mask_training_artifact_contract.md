# Subject Mask Training Artifact Contract (Draft v1)
<!-- contract-meta
version: 2
status: active
last_verified: 2026-07-01
-->

Purpose: define the merged training artifact for a generalized ROI-local
subject-mask task that can supervise fish body, eyes, and swim bladder without
discarding existing eye-mask training data.

Current implementation status: the registry preflight, merged export,
validation, zarr loader, U-Net trainer, model-registry logging/discovery, and
runtime inference path are implemented for the `subject_v1_union` dense
body/eyes-union/swim-bladder workflow. Refined output is produced by the
separate smart finalizer described in
[refined_subject_mask_smart_finalizer_design.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/refined_subject_mask_smart_finalizer_design.md).

## Scope

- Output one merged `.zarr` artifact for subject-mask training.
- Preserve row-aligned crop lineage and per-channel supervision provenance.
- Support partial supervision per channel.
- Keep existing merged eye-mask training artifacts first-class and unchanged as
  historical training datasets.

## Goals

- Allow one target schema such as `["subject_body", "eyes_union",
  "swim_bladder"]`.
- Allow one target schema such as `["subject_body", "eye_left", "eye_right",
  "swim_bladder"]` when the export explicitly targets LR semantics.
- Allow mixed source stages at the contract level:
  - `subject_mask_runs/<run>`
  - `refined_subject_masks_runs/<run>`
  - `eye_masks_runs/<run>`
  - `refined_eye_masks_runs/<run>`
- Avoid fabricating negatives for channels that were never labeled.
- Make it possible to train a new subject-mask model before full relabeling is
  complete.

## Non-goals

- Rewriting historical `eye_masks` merged training datasets.
- Requiring historical eye-mask datasets to be rewritten or backfilled.
- Reconstructing missing full-body or swim-bladder labels from old eye-mask
  datasets.
- Defining runtime inference storage for `subject_mask_runs` itself. This
  contract is for merged training artifacts only. See
  [subject_mask_runs_contract.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/subject_mask_runs_contract.md).

## Compatibility Policy

Historical eye-mask training artifacts remain valid and supported:

- `training_task = "eye_masks"` artifacts remain registry-trackable merged
  training datasets.
- No migration is required for existing merged eye-mask training datasets.

The new subject-mask artifact is additive:

- `training_task = "subject_masks"` is a separate training task.
- Exporters may project eye-mask labels into the subject-mask schema when that
  projection is explicit and loss-masked.
- Current implementation ingests `subject_mask_runs` and coherent
  `refined_subject_masks_runs` directly. Refined-source reads go through the
  mask-store boundary, so analysis inputs may provide dense `masks_roi`,
  compact editable `mask_bitpacked`, or compact final/read-mostly `mask_rle`.
  The exported training artifact still writes dense
  `subject_mask_runs/<run>/masks_roi` with
  `mask_storage_format = "dense_uint8"` and
  `mask_storage_surface = "masks_roi"`.
- Legacy `eye_masks_runs` / `refined_eye_masks_runs` should be projected or
  materialized into a subject-mask source first until direct eye-mask adapters
  are implemented.

Runtime/source-stage compatibility note:

- runtime `subject_mask_runs` uses run-level `available_channels`
- merged training artifacts use per-row `target_valid_channels`
- writers and readers must not treat these as interchangeable names

The distinction is intentional:

- `available_channels` means a runtime source run contains semantically valid
  data for that channel at all
- `target_valid_channels` means a specific training row-channel pair is
  supervised and should contribute to loss

## Crop Pixel Contract

For Orange monochrome camera recordings, the accepted crop pixel contract for
new subject-mask/keypoint training exports is `pynvvc_luma_v1`:

```text
shape: [roi, roi_height, roi_width]
dtype: uint8
source: decoded NV12 Y/luma plane from the source MP4
semantics: mono camera intensity before model-specific tensorization
```

This matches the current Orange TensorRT deployment boundary: Orange detection
and pose paths start from single-channel mono/luma, then perform
engine-specific preprocessing outside TensorRT by resizing/letterboxing,
replicating luma into three planar channels, dividing by 255, and feeding FP32
NCHW tensors to the engine. Training artifacts should therefore preserve the
luma ROI crop as the canonical image surface and leave RGB replication,
normalization, and input-size details to the trainer/runtime for the selected
model.

## Source Parity Audit

The preflight manifest is the canonical source-selection contract for a merged
subject-mask training export. It records the exact source zarr, source stage,
source run, crop run, label schema, sample count, available components, and
registry component-quality rows selected for export.

Before export or training, operators should be able to verify that those
registry-surfaced rows still match the on-disk source representation:

```bash
scripts/py -m fisheye.utils.audit_subject_mask_training_sources \
  /path/to/subject_mask_training.manifest.json
```

The audit checks the manifest `selected_sources` rows against source zarrs:

- selected `source_stage_group/source_subject_mask_run` exists
- selected crop run exists and row-counts match the source physical mask store
  (`masks_roi`, compact `mask_bitpacked`, or compact `mask_rle` through
  `MaskStore`)
- `label_schema_id`, `mask_labels`, `available_channels`, and
  `available_components` agree
- component review state/intended-use from registry agrees with
  `component_review_statuses`
- component mask-present rates agree when `metrics/mask_present` is present
- `refined_subject_masks_runs` sources have approved available components by
  default, matching the merged exporter policy

Use `--allow-unapproved-refined` only for draft/QA exports that intentionally
mirror the exporter override. Use `--read-masks-for-rates` when
`metrics/mask_present` is missing and heavier dense materialization through
`MaskStore` is acceptable.

## Evolution Policy

This training contract is intended to support:

1. sparse supervision from legacy eye-mask sources,
2. dense supervision from future full subject-mask annotations, and
3. later component-scoped exports where a dataset intentionally supervises only
   one subset of subject-mask labels.

V1 artifact semantics support cases 1 and 2. The current exporter supports
those cases when the source has already been materialized as
`subject_mask_runs`.

Case 3 is intentionally deferred at the exporter/workflow level, but the
contract is already compatible with it because supervision is expressed per
row-channel via `target_valid_channels` rather than by assuming all channels
are always labeled.

## Migration Policy

Recommended transition plan:

1. Keep historical `eye_masks_runs` and `refined_eye_masks_runs` in place for
   provenance and compatibility.
2. Allow explicit backfill/projection of legacy eye-mask runs into
   `subject_mask_runs`.
3. Treat `refined_eye_masks_runs` as an eye-specific compatibility/derived
   stage during the transition to unified subject-mask refinement.
4. Prefer `refined_subject_masks_runs` as the canonical refined source for new
   eye-capable exports when unified LR eye components are available there.
5. Deprecate authoring of new raw `eye_masks_runs` only after the
   `subject_mask_runs` path is stable.

This contract only governs the merged training artifact, but exporters should
support that runtime migration path rather than requiring a destructive rewrite
of historical eye-mask data.

## Relationship To Canonical Runtime/Refined Schemas

The canonical runtime/refined eye-capable authoring target is moving toward:

- `subject_mask_runs` for raw snapshots
- `refined_subject_masks_runs` for refined/editable snapshots
- `label_schema_id = "subject_v1_lr"` for canonical refined eye-capable work

This training contract remains more permissive than the runtime/refined
contracts because exporters need to support:

- legacy eye-only sources
- union-eye supervision
- partial-supervision datasets
- compatibility projections into `subject_v1_union`

So the training export default may remain `subject_v1_union` even while
canonical runtime/refined authoring trends toward `subject_v1_lr`.

## Canonical Target Schema

Recommended default target schema for v1:

- `label_schema_id = "subject_v1_union"`
- `mask_labels = ["subject_body", "eyes_union", "swim_bladder"]`

Exporters should also support explicit LR targets, for example:

- `subject_v1_lr`
- `mask_labels = ["subject_body", "eye_left", "eye_right", "swim_bladder"]`

Policy:

- `subject_v1_union` remains the recommended default training export schema for
  v1 compatibility
- `subject_v1_lr` is the canonical refined eye-capable runtime/refined authoring
  target, but not yet required as the default export schema

Writers must always persist:

- `label_schema_id`
- `mask_labels`

Readers must never infer channel meaning from channel index alone.

## Partial-Supervision Rule

This contract is channel-aware and partial-supervision aware.

Required rule:

- every row-channel pair must declare whether that channel is supervised.

Implications:

- an unsupervised channel is not a negative
- zero-filled arrays for unsupervised channels are storage placeholders only
- training loss and metrics must ignore unsupervised channels

## Output Layout

```text
<merged>.zarr/
  crop_runs/
    attrs:
      latest                   "<run_id>"
    <run_id>/
      roi_images               (N, H, W) uint8 or (N, H, W, 3) uint8
      bbox_norm_coords         (N, 4) float32
      crop_bbox_norm_coords    (N, 4) float32
      frame_indices            (N,) int32/int64
      detection_source         (N,) int8

  subject_mask_runs/
    attrs:
      latest                   "<run_id>"
    <run_id>/
      masks_roi                (N, C, H, W) uint8
      target_valid_channels    (N, C) bool
      attrs:
        label_schema_id
        mask_labels
        mask_storage_format    "dense_uint8"
        mask_storage_surface   "masks_roi"
        mask_store_encoding    "dense_uint8"
        allow_partial_supervision

  splits/
    train_indices              (Nt,) int64
    val_indices                (Nv,) int64
    test_indices               (Ntest,) int64      # empty allowed

  source_index/
    source_dataset_idx         (N,) int32
    source_frame_idx           (N,) int64
    source_roi_idx             (N,) int64          # recommended
    label_origin_codes         (N, C) uint8
    supervision_mode_codes     (N, C) uint8
    source_dataset_id          (M,) string
    source_zarr_path           (M,) string
    source_stage_group         (M,) string
    source_run_name            (M,) string
    source_crop_run            (M,) string
    source_label_schema_id     (M,) string
    source_projection_mode     (M,) string
    source_mask_store_encoding (M,) string
    source_mask_storage_surface (M,) string

  attrs:
    zarr_purpose               "training"
    training_task              "subject_masks"
    training_export            {...}
```

## Root Contract

Required root attrs:

- `zarr_purpose = "training"`
- `training_task = "subject_masks"`
- `training_export` dict with:
  - `tool`
  - `created_at_utc`
  - `input_format` (`gray` or `rgb`)
  - `label_schema_id`
  - `mask_labels`
  - `mask_storage_format = "dense_uint8"`
  - `mask_storage_surface = "masks_roi"`
  - `allow_partial_supervision` (`true`)
  - `source_stage` (`subject_mask_runs`, `refined_subject_masks_runs`,
    `eye_masks_runs`, `refined_eye_masks_runs`, or `mixed`)
  - `source_count`
  - `source_zarr_paths`
  - `source_mask_store_encoding`
  - `source_mask_store_encodings`
  - `source_mask_storage_surface`
  - `source_mask_storage_surfaces`
  - `split_seed`

Required root groups:

- `crop_runs/`
- `subject_mask_runs/`
- `splits/`
- `source_index/`

## `crop_runs/<latest>`

Required arrays:

- `roi_images`
  - shape: `(N, H, W)` for gray or `(N, H, W, 3)` for rgb
- `bbox_norm_coords`
  - shape: `(N, 4)`
- `crop_bbox_norm_coords`
  - shape: `(N, 4)`
- `frame_indices`
  - shape: `(N,)`
  - required to be local indexing `0..N-1`
- `detection_source`
  - shape: `(N,)`
  - integer codes, currently expected in `{0, 1}`

Required attrs:

- `source_crop_run`
- `source_zarr_path`

## `subject_mask_runs/<latest>`

Required arrays:

- `masks_roi`
  - shape: `(N, C, H, W)`
  - binary values `{0, 1}`
- `target_valid_channels`
  - shape: `(N, C)`
  - boolean mask indicating which target channels are supervised

Required attrs:

- `label_schema_id`
- `mask_labels`
- `mask_storage_format = "dense_uint8"`
- `mask_storage_surface = "masks_roi"`
- `mask_store_encoding = "dense_uint8"`
- `allow_partial_supervision = true`
- `source_mask_stage`
- `source_crop_run`
- `source_zarr_path`

Dense `masks_roi` is the training artifact compatibility contract. Compact
binary storage options such as bitpacked masks and RLE are valid analysis-source
storage only when read through the mask-store materialization API. A training
artifact must not store `subject_mask_runs/<run>/mask_bitpacked` or
`subject_mask_runs/<run>/mask_rle`; if a compact analysis source was used, that
source encoding is recorded in `source_mask_store_encoding(s)` and the physical
source surface is recorded in `source_mask_storage_surface(s)` while the
training run itself remains dense. See
[`mask_rle_storage_design_and_benchmark_plan.md`](mask_rle_storage_design_and_benchmark_plan.md).

Current readers tolerate older dense training artifacts that predate
`mask_storage_format` and `mask_storage_surface`, but if those attrs are present
they must match the dense training contract. Readers reject training artifacts
that expose compact `subject_mask_runs/<run>/mask_rle`.

Recommended attrs:

- `projection_summary`
- `valid_channel_counts`
- `dense_channel_counts`
- `explicit_negative_channel_counts`
- `source_mask_store_encoding`
- `source_mask_store_encodings`
- `source_mask_storage_surface`
- `source_mask_storage_surfaces`

### Channel validity invariant

For every row `n` and channel `c`:

- if `target_valid_channels[n, c] == false`, the trainer must ignore
  `masks_roi[n, c]`
- if `target_valid_channels[n, c] == false`, exporters should write an all-zero
  mask for storage stability
- if `target_valid_channels[n, c] == true`, the corresponding provenance arrays
  must not encode `no_supervision`

## `source_index/`

Required sample-aligned arrays:

- `source_dataset_idx`
  - shape: `(N,)`
- `source_frame_idx`
  - shape: `(N,)`
- `label_origin_codes`
  - shape: `(N, C)`
- `supervision_mode_codes`
  - shape: `(N, C)`

Recommended sample-aligned array:

- `source_roi_idx`
  - shape: `(N,)`
- `source_refined_row_ids`
  - shape: `(N,)`
  - stable refined-detection row identity when available, or `-1`
- `source_detect_row_index`
  - shape: `(N,)`
  - raw detect row lineage when available, or `-1`

Required source-table arrays:

- `source_dataset_id`
  - shape: `(M,)`
- `source_zarr_path`
  - shape: `(M,)`
- `source_stage_group`
  - shape: `(M,)`
- `source_run_name`
  - shape: `(M,)`
- `source_crop_run`
  - shape: `(M,)`
- `source_label_schema_id`
  - shape: `(M,)`
- `source_projection_mode`
  - shape: `(M,)`
- `source_mask_store_encoding`
  - shape: `(M,)`
  - source analysis mask-store encoding used by the exporter, for example
    `dense_uint8`, `bitpacked_binary_v1`, or `component_rle_v1`
- `source_mask_storage_surface`
  - shape: `(M,)`
  - physical source mask surface used by the exporter: `masks_roi` for dense
    sources, `mask_bitpacked` for compact editable analysis sources, or
    `mask_rle` for compact final/read-mostly analysis sources

Required attrs:

- `mapping_version`
- `source_count`
- `label_origin_codebook`
- `supervision_mode_codebook`

### `label_origin_codes`

Stable vocabulary:

- `0 = unknown`
- `1 = auto`
- `2 = manual_review`
- `3 = manual_training`
- `4 = interpolated`
- `5 = synthetic`

### `supervision_mode_codes`

Stable vocabulary:

- `0 = no_supervision`
- `1 = dense`
- `2 = explicit_negative`
- `3 = box_only`

`box_only` is reserved for future compatibility and should be uncommon for
segmentation tasks.

## `splits/`

Required arrays:

- `train_indices`
- `val_indices`
- `test_indices` (empty allowed)

All split arrays must:

- be 1D integer arrays
- have indices in `[0, N-1]`
- have no duplicates internally
- be pairwise disjoint
- exactly cover all `N` rows across train/val/test

## Source Projection Rules

### 1. Identity projection from subject-mask stage families

If the source run comes from `subject_mask_runs` or
`refined_subject_masks_runs` and already matches the requested target schema:

- masks are copied channel-for-channel
- `target_valid_channels` is derived from the source-run channel availability
  and supervision provenance
- `source_projection_mode = "schema_identity"`

Required rule:

- exporters must interpret runtime `available_channels` as source availability,
  not as row-level supervision by itself
- if a source run marks a channel unavailable, exporters must set
  `target_valid_channels[:, channel] = false`

### 1b. Projection from `subject_v1_lr` into `subject_v1_union`

If the source run is `subject_mask_runs` or `refined_subject_masks_runs` with:

- `label_schema_id = "subject_v1_lr"`
- channels `["subject_body", "eye_left", "eye_right", "swim_bladder"]`

and the requested training target schema is `subject_v1_union`:

- `eyes_union = union(eye_left, eye_right)`
- `target_valid_channels[:, eyes_union]` is true only where both:
  - the source `available_channels` marks `eye_left` and `eye_right` available
  - row-level supervision provenance marks those channels valid
- `source_projection_mode = "eyes_union_from_subject_lr"`

Required rule:

- exporters must not lose row-level provenance when collapsing LR channels into
  union
- exporters must not mark `subject_body` or `swim_bladder` valid unless those
  channels were explicitly supervised in the source rows
- if a refined-subject source is available for a unified LR refined run,
  exporters should prefer that canonical refined source over a compatibility
  `refined_eye_masks_runs` artifact for the same semantic labels

### 2. Eye-mask projection into `subject_v1_union`

If the source stage is `eye_masks_runs` or `refined_eye_masks_runs` and the
requested target schema is `subject_v1_union`:

- target `eyes_union` is allowed
- target `subject_body` is unsupervised unless an explicit body mask source is
  also provided
- target `swim_bladder` is unsupervised unless an explicit swim-bladder mask
  source is also provided

Projection behavior:

- if source eye masks have two channels (`eye_left`, `eye_right`):
  - `eyes_union = union(left, right)`
  - `source_projection_mode = "eyes_union_from_lr"`
- if source eye masks have two non-anatomical channels (for example
  `eye_0`, `eye_1`):
  - `eyes_union = union(channel_0, channel_1)`
  - `source_projection_mode = "eyes_union_from_pair"`
- if source eye masks already have one union channel:
  - `eyes_union = copy(source_union)`
  - `source_projection_mode = "eyes_union_from_union"`

Required rule:

- exporters must not mark `subject_body` or `swim_bladder` as valid when only
  eye masks are available

Implementation status:

- Direct `eye_masks_runs` / `refined_eye_masks_runs` ingestion is not currently
  implemented by `fisheye.utils.export_subject_mask_training_zarr`.
- For now, project or backfill those sources into `subject_mask_runs` first,
  then export the subject-mask training artifact from that adapter run.

### 3. No fabricated negatives

Missing labels must remain unsupervised.

Invalid behavior:

- writing `subject_body = 0` and `target_valid_channels=true` for a row that
  came only from eye-mask annotation
- writing `swim_bladder = 0` and treating it as a dense negative without an
  explicit swim-bladder label source

## Explicit-negative handling

If a source row carries an explicit-negative signal for a target channel:

- `target_valid_channels[n, c] = true`
- `supervision_mode_codes[n, c] = explicit_negative`
- `masks_roi[n, c]` must be all-zero

This is different from `no_supervision`.

`explicit_negative` means:

- the annotator or source contract intentionally supervised that channel as
  absent

`no_supervision` means:

- the channel is unknown and must be ignored

## Required Invariants

- All sample-aligned arrays share identical first dimension `N`.
- `masks_roi.shape[1] == target_valid_channels.shape[1] == C`.
- `label_origin_codes.shape == supervision_mode_codes.shape == (N, C)`.
- `source_dataset_idx[i]` is within `[0, M-1]`.
- `source_dataset_id`, `source_zarr_path`, `source_stage_group`,
  `source_run_name`, `source_label_schema_id`, and `source_projection_mode`
  share identical first dimension `M`.
- When `target_valid_channels[n, c] == false`:
  - `supervision_mode_codes[n, c]` must equal `0`
- When `target_valid_channels[n, c] == true`:
  - `supervision_mode_codes[n, c]` must not equal `0`

## Training Behavior Contract

Training loaders consuming this artifact must:

- apply loss only where `target_valid_channels == true`
- compute per-channel metrics only over valid rows for that channel
- treat `explicit_negative` as supervised zero-target rows
- never infer supervision from mask values alone

## Migration Rules

1. Existing `eye_masks` merged training datasets remain valid historical
   artifacts and should not be rewritten as part of runtime eye-mask severance.
2. New `subject_masks` exports may ingest old eye-mask source runs.
3. Old eye-mask rows become partial-supervision rows in the new schema.
4. Exporters must preserve the strongest row-level provenance available from
   source runs and use `unknown` when the source signal is ambiguous.
5. Exporters must not require complete relabeling before producing a valid
   `subject_masks` artifact.

## Example conversion from existing eye-mask data

Given source eye-mask labels only:

- source stage: `refined_eye_masks_runs`
- source channels: `["eye_left", "eye_right"]`
- target schema: `["subject_body", "eyes_union", "swim_bladder"]`

Exported row semantics:

- `subject_body`
  - `target_valid_channels = false`
  - `supervision_mode = no_supervision`
- `eyes_union`
  - `target_valid_channels = true`
  - `supervision_mode = dense` or `explicit_negative`
  - mask value = union of source eye channels
- `swim_bladder`
  - `target_valid_channels = false`
  - `supervision_mode = no_supervision`

This preserves valuable eye supervision without pretending body or
swim-bladder labels exist.

## Implemented Entrypoints

Implemented:

- `scripts/py -m fisheye.utils.validate_subject_mask_training_zarr <merged>.zarr`
- `scripts/py -m fisheye.utils.export_subject_mask_training_zarr <source>.zarr <merged>.zarr`
- `scripts/py -m fisheye.utils.prepare_subject_mask_training_from_registry --registry <registry.sqlite> --out-manifest <manifest.json>`
- `scripts/py -m fisheye.utils.run_subject_mask_training_pipeline --manifest <manifest.json> --config <config.yaml> --export-merged --train`
- `scripts/py -m fisheye.segmentation.train_unet_subject_masks <config.yaml> --manifest <manifest.json> --registry <registry.sqlite>`
- `scripts/py -m fisheye.segmentation.infer_unet_subject_masks <analysis.zarr> --resolve-model-from-registry --registry <registry.sqlite>`

The registry preflight is prepare-only. It selects exportable unified
`subject_mask_runs` or coherent `refined_subject_masks_runs` sources from
`subject_mask_component_quality_*` registry surfaces, writes
`selected_sources`, and flags split refined latest component truth that needs
explicit assembly. The pipeline wrapper consumes that manifest, exports one
merged subject-mask training zarr, rewrites the training config/manifest to the
merged `crop_runs/<run>` and `subject_mask_runs/<run>`, and can launch
`fisheye.segmentation.train_unet_subject_masks`. The trainer derives `names` and
`nc` from the merged artifact schema, can write validation preview PNGs at
multiple thresholds, can log TensorBoard scalars when configured, and records
subject-mask model coverage metadata in the registry. Subject-mask data-card
aggregation is still future parity work. Refined-source export is approved-only
by default: every available component in a `refined_subject_masks_runs/<run>`
must have `component_review_statuses[component].state == "approved"`, with
`--allow-unapproved-refined` reserved for draft/QA exports. If registry latest
component truth is split across multiple `refined_subject_masks_runs`, first
assemble a new coherent approved refined run with
`fisheye.refinement.assemble_refined_subject_masks`; the exporter should not
silently combine split refined sources.
For historical refined-eye plus refined swim-bladder consolidation, follow the
additive unified eye/swim migration procedure in
`refined_subject_masks_runs_contract.md`. Assembly now records source review
state as provenance but only promotes approved source review onto the target
when `--promote-source-review` is passed explicitly.

## Related Contracts

- [refined_subject_masks_runs_contract.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/refined_subject_masks_runs_contract.md)
- [eye_subject_mask_unification_design.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/eye_subject_mask_unification_design.md)
- [subject_mask_runs_contract.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/subject_mask_runs_contract.md)
- [training_label_origin_provenance_todo.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/training_label_origin_provenance_todo.md)
- [training_quality_gate_contract.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/training_quality_gate_contract.md)
