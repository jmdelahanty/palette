# Clipped Recording Consumer Mapping Contract
<!-- contract-meta
status: design_contract
last_verified: 2026-05-16
purpose: Define how clipped-recording frame mappings affect Palette readers, training consumers, and future Crimson integration.
-->

## Purpose

Clipped recordings introduce two frame clocks:

- the parent recording clock, which is continuous for the biological recording;
- the clip-local video clock, which starts at zero inside each MP4 clip.

Existing Palette and Crimson consumers mostly assume one video clock per Zarr.
This contract makes the transition explicit so new clipped workflows do not
silently break review, editing, or analysis readers.

## Source Of Truth

For Orange-style clipped recordings:

- `recording_clip_index.json` plus per-clip `Cam*_meta.csv` are the
  authoritative source artifacts.
- `recording_frame_index.parquet` is a recording-level derived convenience
  table that maps parent frames to source clip frames. It normally lives at the
  recording folder root, not inside a training Zarr or analysis Zarr.
- Zarr run groups store scientific outputs and review/edit state.
- Registry and `experiment_index` views are rebuildable projections.

The frame index is not a review ledger. It must not store approvals, manual
edits, stale flags, latest choices, or model output status.

## Frame Index Spaces

### Parent Recording Index

`parent_frame_index` is the zero-based index for the continuous recording
timeline. For native Orange metadata with one-based `recording_frame_id`, this
is normally:

```text
parent_frame_index = recording_frame_id - 1
```

This is the timeline a UI should use for whole-recording navigation and plots.

### Recording Frame ID

`recording_frame_id` is the Orange session-continuous frame id from
`Cam*_meta.csv` `frame_id`. It is the semantic key together with
`camera_serial`:

```text
(camera_serial, recording_frame_id)
```

### Clip-Local Frame Index

`clip_local_frame_index` is the zero-based row number in one clip's metadata
CSV and the frame index into that clip's MP4.

### Sample Index

Training Zarrs materialize selected frames into `raw_video/images_*`.
Stage arrays inside a training Zarr should normally use `frame_indices` as
sample-local indices into those materialized arrays.

## Consumer Rules By Archive Type

### Traditional Single-Video Analysis Zarr

Current consumers are unchanged.

- Top-level run families such as `detect_runs/<run>` and
  `refined_detect_runs/<run>` are real run groups.
- `frame_indices` refers to the archive's source/import frame index.
- `source_video_path` can identify the one backing video when live decode is
  needed.
- Crimson's current read contracts apply.

### Sampled Training Zarr

Current training consumers remain sample-local.

- `raw_video/images_*[sample_index]` is the materialized image.
- `raw_video/original_frame_indices[sample_index]` maps to the source frame in
  the source recording context.
- `detect_runs/<run>/frame_indices` and
  `refined_detect_runs/<run>/instances/frame_indices` refer to sample indices,
  not necessarily source video frame numbers.
- Merged/exported training datasets should keep their own `source_index/`
  lineage arrays.

For a single-video sampled training Zarr, `original_frame_indices` is often
enough to reopen the one source video because there is only one backing MP4.
For a clipped training Zarr, it is intentionally not enough: it gives the
parent-frame coordinate, and a Parquet source map supplies the clip path and
clip-local frame.

### Clipped Training Zarr

Clipped training Zarrs should look like sampled training Zarrs to existing
label/model consumers, but they must carry a richer source map.

Required behavior:

- `frame_indices` inside detect/refined/crop/keypoint/mask runs remain
  sample-local.
- `raw_video/original_frame_indices` contains `parent_frame_index`.
- `source_frame_index.parquet` maps each `sample_index` to
  `(camera_serial, recording_frame_id, parent_frame_index, clip_id,
  clip_local_frame_index, video_path)`.
- Root attrs point to both the full recording frame index and the sampled
  source-frame index.

This lets existing training exporters work without becoming clip-aware, while
new provenance/debug tools can recover the exact source MP4 and local frame.

Mapping chain:

```text
raw_video/images_full[sample_index]
  -> raw_video/original_frame_indices[sample_index] = parent_frame_index
  -> source_frame_index.parquet row for sample_index
  -> clip_id + clip_local_frame_index + video_path
```

The sampled `source_frame_index.parquet` is a row-aligned subset/snapshot for
the training Zarr. The full `recording_frame_index.parquet` remains the
recording-level map for all frames, including frames not imported into the
training Zarr.

### Dish Masks And Spatial Metadata

Dish masks are camera/static spatial metadata. They are not clip-local review
state. Orange acquisition guarantees that dish locations and camera geometry do
not move within a recording. For rolling clips cut from one camera stream, the
same camera-specific dish mask is used for every clip in that recording.
The persisted mask payload may be expressed in the coordinate system of the
array on which it was tuned, commonly `images_ds`; consumers should preserve
`tuned_on_array` and use normalized mask metrics when applying the mask to
normalized detections.

Consumer rules:

- Materialized clipped training Zarrs should carry
  `analysis_metadata.attrs["dish_mask"]` copied from the parent/source tuning
  Zarr before detections are copied or predicted.
- Clip-local analysis runs should record the source dish mask or inherited
  camera spatial metadata in provenance, rather than duplicating conflicting
  masks per clip.
- Multi-camera recordings must treat dish masks as keyed by `camera_serial`.
- Clipped analysis shells declare this invariant with
  `analysis_metadata.attrs["dish_mask_scope"] == "recording_camera"` and
  `analysis_metadata.attrs["dish_mask_clip_policy"] ==
  "single_camera_mask_applies_to_all_clips"`.
- If a mask is missing, readers should not synthesize one silently. Refinement
  and quality tools should report that the run was produced without a
  dish-mask gate.

Current refinement treats the dish mask as a spatial gate: outside-mask
candidate detections may remain in audit surfaces, but accepted refined
instances should be inside the mask unless manually overridden by a documented
workflow.

### Instance Row Identity Across Clips

`refined_row_ids` are stable logical row IDs within one concrete refined run.
They are not globally unique across a clipped recording, and they are not fish
identity.

For traditional archives and materialized training Zarrs, the run path is
usually implicit because there is one active top-level refined run. For clipped
analysis archives, the run path must be part of identity:

```text
(recording_id, camera_serial, clip_id, refined_run_path, refined_row_id)
```

or:

```text
(source_refined_run_path, source_refined_row_id)
```

Crimson and Palette repair tools must not join, edit, or mark stale rows from
different clip-local runs by `refined_row_id` alone. Downstream outputs that
copy `source_refined_row_ids` from a clipped source must also preserve the
source clip/run path or a manifest that resolves it.

### Clipped Analysis Zarr Shell

The metadata-only clipped analysis shell is not a drop-in replacement for a
traditional analysis Zarr.

Important constraints:

- Top-level run-family groups are placeholders for future finalized or
  aggregated views.
- Top-level `latest` attrs should not be interpreted as real run selections
  until a finalize stage writes explicit finalized collections.
- Physical model outputs live under clip-camera namespaces:

```text
clips/<clip_id>/cameras/<camera_serial>/<family>/<run_name>
```

- Inside those clip-local run groups, `frame_indices` should be clip-local
  indices unless the run explicitly declares a parent-frame coordinate system.
- Consumers that need whole-recording navigation must map through
  `recording_frame_index.parquet` or a finalized collection manifest.

In analysis shells, the recording-level parquet table is not merely optional
provenance. Until a finalized collection exists, it is the only general map
from parent timeline frame to the physical clip file and local video frame.

## Crimson Impact

Crimson's current Palette contracts are valid for traditional top-level
analysis Zarrs and materialized training Zarrs. They are not sufficient for
clipped analysis shells.

Before Crimson can treat clipped analysis archives as first-class review
targets, Palette needs a resolver layer with this behavior:

1. Select a finalized workflow collection from
   `experiment_index/finalized_runs/<workflow_id>`.
2. Resolve each logical stage to concrete clip-camera run paths.
3. Convert clip-local `frame_indices` to parent timeline positions for display.
4. Route edits back to the owning clip-camera run group, not a top-level
   placeholder.
5. Mark downstream clip-camera outputs stale when an upstream clip-camera
   refined surface changes.
6. Present parent-wide status from the finalized collection, not from a single
   top-level `latest` attr.

Until that resolver exists, Crimson should either:

- open traditional single-video analysis/training Zarrs; or
- open materialized clipped training Zarrs whose frame/image arrays are
  sample-local and self-contained.

It should not assume `clips/clip_*/...` can be flattened by directory order.
The parent timeline order comes from `recording_frame_index.parquet` or a
finalized collection manifest.

## Editing And Stale-State Rules

Manual edits should target the same coordinate namespace as the reviewed run.

For traditional archives:

- edits target `refined_detect_runs/<run>/instances` or the current accepted
  write surface.

For clipped analysis archives:

- edits target
  `clips/<clip_id>/cameras/<camera_serial>/refined_detect_runs/<run>/instances`
  or the corresponding clip-local refined surface.
- downstream stale checks compare clip-local upstream fingerprints and stable
  row ids.
- parent-level finalized status is recomputed after clip-local edits.

The frame index does not change when detections, masks, or keypoints are
edited.

## Reader Migration Checklist

- [ ] Add a shared Palette resolver that can return either a traditional
  top-level run path or a finalized clipped run collection.
- [ ] Define a finalized collection manifest schema under
  `experiment_index/finalized_runs/<workflow_id>`.
- [ ] Update status/check tools to report clipped shell state as
  `metadata_only` until finalized runs exist.
- [ ] Update Crimson contracts after the resolver exists, rather than making
  Crimson independently discover clip directories.
- [ ] Add tests for mapping clip-local detect rows to parent timeline frames.
- [ ] Add tests that top-level clipped shell placeholders are not mistaken for
  real latest runs.

## Non-Goals

- Do not put millions of frame-map rows in Zarr attrs.
- Do not make `recording_frame_index.parquet` mutable review state.
- Do not require every legacy reader to understand clipped analysis shells
  before clipped training Zarr creation.
- Do not make one global concatenated run group the primary cluster write
  target just to preserve old reader assumptions.
