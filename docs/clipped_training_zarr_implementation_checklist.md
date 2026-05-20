# Clipped Training Zarr Implementation Checklist
<!-- contract-meta
status: working_checklist
last_verified: 2026-05-20
purpose: Define and track the next implementation slice for creating sampled training Zarrs from Orange-style clipped recordings.
-->

## Purpose

Palette can now materialize long recordings into Orange-style clips, build a
recording-level frame index, and create a metadata-only clipped analysis-Zarr
shell. The next slice is to create sampled training Zarrs from that clipped
layout without treating each clip as a separate biological recording.

This document records the frame-map design, the Parquet decision, and the
implementation checklist for that slice.

Consumer-facing frame semantics are defined in
`docs/clipped_recording_consumer_mapping_contract.md`. That document is the
place to check before changing how `frame_indices`,
`raw_video/original_frame_indices`, `parent_frame_index`, or
`recording_frame_id` are exposed to readers.

Storage-root relocation semantics are defined in
`docs/recording_store_relocation_components.md`. That document is the place to
check before moving clipped recordings or clipped training Zarrs from local
paths such as `/nvme1/recordings` to durable `/groups` storage.

## Current Prerequisites

- [x] `fisheye.utils.plan_orange_style_clips` plans keyframe-aligned clip
  boundaries.
- [x] `fisheye.utils.materialize_orange_style_clips` writes Orange-style clip
  folders.
- [x] `fisheye.utils.verify_orange_style_clips` validates clip structure and
  optional packet counts.
- [x] `fisheye.utils.build_recording_frame_index` writes
  `recording_frame_index.parquet` plus
  `recording_frame_index_manifest.json`.
- [x] `fisheye.utils.create_clipped_analysis_zarr` writes a metadata-only
  parent analysis shell with clip-camera namespaces.
- [x] Real sleepyfish smoke has 22 clips, 22 clip-camera rows, and 1,188,000
  indexed parent frames.
- [x] `fisheye.utils.create_clipped_training_zarr` can materialize sampled
  clipped training Zarrs from `recording_frame_index.parquet`.
- [x] Real sleepyfish cam2010093 dry run selected 238 `frame_step=5000`
  samples across 22 clips and preflighted exact-copy compatibility with
  `detect_seed_v007_20260513` and
  `refined_detect_2026-05-13_16-00-31_review_migrated`.
- [x] Real sleepyfish cam2010093 apply smoke wrote 3 sampled clipped frames
  to `/tmp`, copied dish-mask metadata, and wrote `source_frame_index.parquet`.
- [x] Real sleepyfish cams 2010093-2010096 have full clipped training Zarrs
  created with `frame_step=5000`: 238 sampled frames each, 22 source clips
  each, `raw_video/images_full`, `raw_video/images_ds`, and
  `source_frame_index.parquet`.
- [x] Real sleepyfish cams 2010093-2010096 copied cleaned
  `detect_seed_v007_20260513` and
  `refined_detect_2026-05-13_16-00-31_review_migrated` after exact
  `raw_video/original_frame_indices == parent_frame_index` preflight.
- [x] Detection-label promotion from clipped analysis finalized collections into
  per-recording training Zarrs is implemented through
  `fisheye.tune.detect_training_promotion_backend` and
  `fisheye.utils.promote_analysis_detect_to_training`.
- [x] Detection-label promotion mirrors positive labels into the canonical
  `refined_detect_runs/<run>/instances` table. Crop-run bbox fields remain
  materialized image/support metadata, not the authoritative detection-label
  surface for exporters or downstream consumers.
- [x] The video-backed detection review UI can call the same promotion backend
  on save when launched with `--edit --promote-training-zarr`; batch saves call
  promotion once per saved batch and clipped appends decode once per touched
  source clip.

## Design Decisions

### Recording Identity

A clipped recording remains one recording/session. Clips are child artifacts,
not separate registry recordings. A training Zarr sampled from a clipped
recording should preserve:

- parent `recording_id` / `session_id`;
- `camera_serial`;
- parent-frame identity through `parent_frame_index` and
  `recording_frame_id`;
- clip source identity through `clip_id` and `clip_local_frame_index`.

### Parquet Decision

Use Parquet for row-oriented frame maps and Zarr arrays for image/model data.

`recording_frame_index.parquet` at the recording root is the full source map.
It is derived from `recording_clip_index.json` plus per-clip metadata CSVs and
is safe to regenerate from recording metadata.

This full map is not limited to analysis Zarrs. It is a recording-level sidecar
for any clipped recording and may be used by analysis shells, training-Zarr
creation, diagnostics, or future viewers. For single-video recordings the same
sidecar is optional because the mapping is trivial, but keeping the same table
shape can simplify generic tooling.

The sampled training Zarr should additionally carry a small row-aligned
snapshot of the selected source rows, preferably as:

```text
<training>.zarr/
  source_frame_index.parquet
```

Recommended root attrs:

- `source_frame_index_path = "source_frame_index.parquet"`
- `source_frame_index_schema = "palette.training_source_frame_index.v1"`
- `source_recording_frame_index_path = "/path/to/recording_frame_index.parquet"`
- `source_layout = "rolling_clips"`
- `source_video_path = null` or omitted when multiple clips contribute frames
- `source_video_paths = [...]` for inspection only, not row mapping authority

Rationale:

- Zarr attrs are the wrong shape for hundreds to millions of mixed-type rows.
- Zarr arrays are awkward for path/string-heavy frame-map tables.
- Parquet supports fast filtered reads by `clip_id`, `camera_serial`, or frame
  range.
- A sampled `source_frame_index.parquet` makes the training Zarr portable even
  if the source recording later moves.

### Compatibility With Existing Training Readers

Keep `raw_video/original_frame_indices` for compatibility. For clipped
training Zarrs it must contain `parent_frame_index`, not clip-local frame
indices.

That means `original_frame_indices` is the sampled-Zarr bridge back to the
parent recording timeline. It is not the clip lookup table. Exact clip lookup
uses either the full `recording_frame_index.parquet` or the sampled
`source_frame_index.parquet`.

Existing label arrays should remain sample-local:

- `detect_runs/<run>/frame_indices` indexes rows in `raw_video/images_*`.
- `refined_detect_runs/<run>/instances/frame_indices` indexes rows in
  `raw_video/images_*`.
- crop/keypoint/mask run `frame_indices` remain sample-local if those stages
  are later added to the training Zarr.

The row-aligned source table bridges sample-local rows back to source clips:

```text
sample_index -> parent_frame_index -> (clip_id, clip_local_frame_index, video_path)
```

Crimson impact:

- Existing Crimson detection/keypoint/mask readers can treat a clipped training
  Zarr like a normal sampled training Zarr because images are materialized and
  stage `frame_indices` remain sample-local.
- Crimson should not use `source_video_path` as a single-video authority for
  clipped training Zarrs; source-video lookup must go through
  `source_frame_index.parquet` when exact provenance is needed.
- This does not make clipped analysis shells Crimson-ready. Clip-local
  analysis review requires a finalized collection resolver first.

### Dish Mask And Instance Mapping

Dish masks are spatial camera metadata, not clip metadata. Orange acquisition
guarantees that dish locations and camera geometry do not move within a
recording. For rolling clips from one camera stream, one camera-specific dish
mask applies across every clip for that recording. The saved mask payload may
be tuned on `images_ds` or `images_full`; consumers should preserve the source
`tuned_on_array` and normalized metrics rather than assuming the raw numbers
are always `images_full` pixels.

The clipped training-Zarr creator should therefore:

- copy or derive the recording/camera dish mask into
  `analysis_metadata.attrs["dish_mask"]` when available;
- record the dish-mask source path or source Zarr plus recording-camera frame
  shape in the training-Zarr manifest;
- preserve the copied mask's `tuned_on_array`, source metrics, and normalized
  geometry fields;
- keep dish masks keyed by `camera_serial` when a future multi-camera clipped
  recording exists;
- fail only when the operator explicitly requires a dish mask; otherwise mark
  the mask as missing and let later refinement run with its normal no-mask
  behavior.

This matters because current refinement uses
`analysis_metadata.attrs["dish_mask"]` as a hard spatial gate. Outside-dish
source detections are preserved in `source_detections/` for audit but excluded
from accepted `instances/`.

For new clipped training Zarrs sampled from raw clips, detections and refined
detections are created after import. Their `frame_indices` are sample-local and
their `refined_row_ids` are local to that training Zarr run. There is no
cross-clip instance identity to preserve at import time beyond the
`source_frame_index.parquet` row map.

For future label import from already processed clipped analysis runs, row
identity must include the source run path. `refined_row_ids` are stable within a
single refined run, but they are not globally unique across clip-local runs.
The safe source identity is a composite key such as:

```text
(recording_id, camera_serial, clip_id, refined_run_path, refined_row_id)
```

or equivalently:

```text
(source_refined_run_path, source_refined_row_id)
```

Label-import provenance should preserve:

- `source_recording_id`
- `source_camera_serial`
- `source_clip_id`
- `source_parent_frame_index`
- `source_clip_local_frame_index`
- `source_refined_run_path`
- `source_refined_row_ids`
- `source_detect_row_index`

Do not merge or compare `refined_row_ids` from different clip-local runs by
integer value alone.

### Proposed `source_frame_index.parquet` Columns

Required columns:

- `sample_index`
- `session_id`
- `recording_id`
- `camera_serial`
- `parent_frame_index`
- `recording_frame_id`
- `clip_index`
- `clip_id`
- `clip_local_frame_index`
- `timestamp`
- `timestamp_sys`
- `video_path`
- `metadata_path`
- `keyframe_path`
- `clip_manifest_path`
- `source_recording_frame_index_path`

Optional columns:

- `sample_reason`
- `frame_step`
- `sample_plan_id`
- `source_manifest_sha256`
- `video_sha256` if a future integrity pass computes it

### Sampling And Decode Policy

Sample on the parent recording clock, then group selected rows by
`(camera_serial, clip_id)` for efficient decode.

For each clip group:

1. open the clip video once;
2. decode selected `clip_local_frame_index` rows;
3. write samples into the output Zarr at their assigned `sample_index`;
4. append the same rows to `source_frame_index.parquet`.

For sparse sampling from short clips, random access may be acceptable. For
larger samples, sort selected rows within each clip and prefer sequential
decode where practical. Do not reopen one video per sample.

## Proposed CLI

Current CLI:

```bash
scripts/py -m fisheye.utils.create_clipped_training_zarr \
  /path/to/recording_folder \
  --frame-step 5000 \
  --camera-serial 2010093 \
  --output-zarr /path/to/recording/zarr/<recording>_training.zarr \
  --copy-analysis-metadata-from /path/to/source_training.zarr \
  --require-dish-mask \
  --write-manifest \
  --dry-run
```

Apply mode should require an explicit flag:

```bash
scripts/py -m fisheye.utils.create_clipped_training_zarr \
  /path/to/recording_folder \
  --frame-step 5000 \
  --camera-serial 2010093 \
  --output-zarr /path/to/recording/zarr/<recording>_training.zarr \
  --copy-analysis-metadata-from /path/to/source_training.zarr \
  --require-dish-mask \
  --write-manifest \
  --apply \
  --overwrite
```

When an existing sampled training Zarr has cleaned detection labels for the
same selected parent frames, pass:

```bash
--copy-existing-detections-from /path/to/source_training.zarr
```

This copies `analysis_metadata`, `detect_runs`, and `refined_detect_runs` only
after verifying exact equality between source
`raw_video/original_frame_indices` and the new clipped training sample's
`parent_frame_index` array. Mismatched frame maps fail closed.

## Implementation Checklist

### A. Planner

- [x] Resolve `recording_frame_index_manifest.json` and
  `recording_frame_index.parquet`.
- [x] Filter by `camera_serial` when requested; fail closed if multiple
  cameras exist and no camera is specified.
- [x] Support `--frame-step` and `--max-frames` / `--target-count` sampling.
- [x] Produce deterministic `sample_index` ordering by parent frame index.
- [x] Group selected rows by `(camera_serial, clip_id)` for decode.
- [x] Emit dry-run JSON with selected row count, clips touched, frame range,
  and output paths.

### B. Writer

- [x] Create a training Zarr with `zarr_purpose="training"`.
- [x] Write `raw_video/images_full` and, if requested, downsampled arrays using
  the same semantics as current sampled imports.
- [x] Write `raw_video/original_frame_indices = parent_frame_index`.
- [x] Write `source_frame_index.parquet` row-aligned to samples.
- [x] Record root attrs that point to the full recording frame index and the
  sampled source-frame index.
- [x] Copy or derive a camera-specific `analysis_metadata.attrs["dish_mask"]`
  when available.
- [x] Record dish-mask source, coordinate space, and shape in the manifest.
- [x] Preserve the clipped shell dish-mask policy attrs:
  `dish_mask_scope="recording_camera"` and
  `dish_mask_clip_policy="single_camera_mask_applies_to_all_clips"`.
- [x] Preserve source recording metadata, dish design, experiment labels, and
  recording-only context.
- [x] Write a manifest JSON with command, git/platform provenance, sampling
  parameters, selected clips, and output checks.
- [x] Optionally copy existing `detect_runs` and `refined_detect_runs` when
  source and destination sampled parent-frame maps are exactly identical.

### C. Validation

- [x] Unit-test planner behavior with a synthetic clipped recording and
  synthetic `recording_frame_index.parquet`.
- [x] Unit-test dry-run no-mutation behavior.
- [x] Unit-test `raw_video/original_frame_indices` equals selected
  `parent_frame_index`.
- [x] Unit-test `source_frame_index.parquet` row count equals imported sample
  count.
- [x] Unit-test dish-mask metadata copy and require-mask behavior.
- [x] Unit-test that clipped training stage `frame_indices` are sample-local
  while source provenance maps to parent/clip-local frames.
- [x] Real dry run on one sleepyfish clipped recording with `frame_step=5000`.
- [x] Real apply smoke on one sleepyfish clipped recording with 3 frames.
- [x] Real full apply on sleepyfish cams 2010093-2010096 with `frame_step=5000`.
- [x] Verify decoded frame count, selected clips, and source row mapping.
- [x] Verify full sleepyfish clipped training Zarrs have dish masks before copied
  detection groups, 238 `source_frame_index.parquet` rows, and exact source
  parent-frame mapping.
- [ ] Verify the resulting full clipped training Zarr can run the existing dish-mask,
  detection prediction, refinement, and review workflow.
- [ ] Verify Crimson/operator readers can inspect the resulting clipped
  training Zarr as a sampled training archive without interpreting source
  clip paths directly.

### D. Registry And Operator Workflow

- [x] Creation leaves registry scanning as an explicit follow-up. This keeps
  import/materialization deterministic and lets operators inspect a clipped
  training Zarr before it becomes discoverable for export.
- [ ] Add operator docs with dry-run and apply commands.
- [x] Confirm registry reports `zarr_use="training"` and preserves source
  context for clipped sources. The datasets table and
  `dataset_context_current` expose `source_layout`,
  `source_frame_index_path`, `source_recording_frame_index_path`, and
  `source_frame_index_schema`.
- [x] Registry-driven detection training preparation defaults to
  `--training-sample-duplicate-policy prefer-clipped`, which skips original
  full-video sampled training Zarrs when a clipped training Zarr has the same
  recording, camera, and `raw_video/original_frame_indices` fingerprint.
- [x] The duplicate policy can be set to `error` for audit runs or `keep-all`
  only when deliberate double-counting is required.
- [ ] Add a training-image profile smoke so source-frame metadata is visible in
  registry summaries.

### E. Deferred

- [ ] Clip-aware analysis writers and run-group importers.
- [x] Finalized detect/refined-detect clip collections under
  `experiment_index/finalized_runs/<workflow_id>`.
- [ ] Temporal boundary policy for track kinematics and bout detection.
- [x] Detection bbox promotion from existing clipped analysis refined runs using
  composite clipped source identity and parent-frame mapping.
- [ ] Keypoint, mask, and multi-instance label import from existing clipped
  analysis runs.
- [ ] Multi-camera clipped recordings.

## Staleness Notes From 2026-05-16 Docs Pass

- Existing sampled-import docs describe one source `cams/*.mp4`; clipped
  training imports need a source-frame table because samples may come from many
  MP4s.
- Existing detection/keypoint/mask training exporters can remain sample-local
  if clipped training Zarrs materialize `raw_video/images_*` and keep
  `frame_indices` sample-local.
- Existing cluster docs should use the clip-camera namespace
  `clips/<clip_id>/cameras/<camera_serial>/<family>/<run>` for future
  clip-local run groups.
- `recording_frame_index.parquet` is not a review ledger and should not contain
  mutable labels, approvals, stale flags, or latest-run decisions.
- Existing single-run lineage docs correctly describe `refined_row_ids` as
  stable row identity within one run. Clipped workflows must make that scope
  explicit by pairing row ids with the concrete clip-camera run path.
