# Clipped Finalized Detect Collection Contract
<!-- contract-meta
version: 1
status: active
implementation: implemented
last_verified: 2026-05-28
purpose: Define how consumers resolve finalized clipped refined-detect collections to parent frames, clip-local video frames, and refined detection rows.
-->

## Purpose

Clipped analysis recordings do not have one top-level video clock. A parent
recording timeline is assembled from many clip videos, and each finalized
detect/refine workflow selects one refined run per `(clip_id, camera_serial)`.

This document is the consumer contract for resolving that layout. Consumers
include:

- `fisheye.tune.video_detect_review_web`;
- `fisheye.utils.resolve_clipped_refined_detect_collection`;
- analysis-to-training promotion tools;
- Crimson or other non-Python viewers.

## Authoritative Surfaces

The authoritative collection lives in the analysis Zarr:

```text
experiment_index/finalized_runs/<collection_id>
```

This does not make flat `refined_detect_runs/<run>/instances` obsolete in
general. Flat top-level refined-detection runs remain the current canonical
surface for non-clipped recordings and materialized training Zarrs. In clipped
analysis archives, the source-preserving authority is the finalized collection
plus the selected clip-local refined runs. If a clipped archive also carries a
top-level flat refined-detection table, consumers should treat it as a derived
projection or compatibility/convenience view unless its provenance explicitly
declares otherwise.

Collection discovery order:

1. Use an explicit caller-provided `collection_id`.
2. Otherwise use `refined_detect_runs.attrs.latest_collection`.
3. If neither exists, fail closed. Do not infer collections by scanning clip
   directories.

The collection group must carry `attrs.selected_runs`, a list with one selected
run per `(camera_serial, clip_id)` pair. Consumers must treat duplicate selected
pairs as invalid.

Each selected run entry should include:

- `camera_serial`
- `clip_id`
- `detect_run`
- `detect_quality_run`, when produced by the workflow
- `refined_detect_run`
- `detect_group_path`
- `refined_group_path`
- `source.video_path`
- `source.metadata_path`, when available
- `source.keyframe_path`, when available

Group paths are Zarr-internal relative paths and must remain relative, for
example:

```text
clips/clip_000017/cameras/2010093/refined_detect_runs/<run>
```

## Frame Index Contract

Consumers resolve the parent timeline through `recording_frame_index.parquet`.

Frame-index path discovery order:

1. Use an explicit `--recording-frame-index` or equivalent caller override.
2. Use root attr `recording_frame_index_path`, if present.
3. Use collection attr `plan_path`, then the plan's `recording_dir`, then
   `recording_frame_index_manifest.json`.
4. Fall back to `<recording_dir>/recording_frame_index.parquet`, where
   `<recording_dir>` is inferred from the analysis Zarr path.

Required columns:

- `camera_serial`
- `clip_id`
- `clip_local_frame_index`
- `recording_frame_id`
- `video_path`

Recommended columns:

- `parent_frame_index`
- `metadata_path`
- `keyframe_path`
- `timestamp`
- `timestamp_sys`

Frame key semantics:

- `parent_frame_index` is the zero-based parent recording frame shown by
  review UIs.
- `recording_frame_id` is the acquisition frame id and is one-based for the
  current clipped recording products.
- `clip_local_frame_index` is the zero-based frame number inside the clip MP4.
- If `parent_frame_index` is present, consumers must use it and require it to
  be row-aligned for random access by parent frame.
- If `parent_frame_index` is absent, consumers may derive
  `parent_frame_index = recording_frame_id - 1` only as a compatibility
  fallback and should log that fallback.

Consumers must not use sorted clip directory order as a frame map.

## Detection Lookup

For a parent frame:

1. Read the row at `parent_frame_index` from `recording_frame_index.parquet`.
2. Extract `(camera_serial, clip_id, clip_local_frame_index)`.
3. Find the selected run for `(camera_serial, clip_id)` in
   `selected_runs`.
4. Open `selected.refined_group_path`.
5. Resolve the refined row whose `frame_indices` equals
   `clip_local_frame_index`.

The preferred refined surface is the canonical sparse instances surface under:

```text
<refined_group_path>/instances
```

Consumers that still support older dense or staged layouts must prefer
`instances` when present. New clipped finalized collections should not require
top-level `refined_detect_runs/<run>` arrays.

## Coordinate Contract

Canonical review payloads expose normalized boxes:

```text
bbox_norm_coords = [cx, cy, width, height]
```

The normalized coordinates are the storage-independent edit surface used for
manual saves and promotion. Consumers should derive display coordinates from
normalized boxes using the source-video dimensions recorded on the session or
run metadata. This follows Palette's canonical full-frame-normalized
`bbox_norm_coords` convention: center coordinates plus box size, all normalized
to the source full-frame image. ROI-local or crop-video-local boxes must use a
more specific name such as `bbox_crop_norm_coords` or `bbox_roi_xyxy`.

Modern clip-local raw detection runs must store full-frame dimensions as
`source_video_width` and `source_video_height`. Recording-ordered detection
snapshots using `palette.clipped_detect_quality_source.v2` must carry the same
two attrs after verifying that every selected clip agrees. Consumers should
resolve full-frame geometry in this order:

1. the explicitly selected run or recording-ordered source;
2. root `source_video_width`/`source_video_height` or root `width`/`height` when
   they are a validated recording invariant;
3. `raw_video` source/full-frame attrs or stored full-frame array shape;
4. an explicit caller override used only when canonical metadata is absent.

Conflicting positive values must fail closed. `inference_width` and
`inference_height` describe model input resizing and are never a substitute for
source full-frame geometry.

Completed historical v1 recording-order sources may receive a metadata-only
geometry repair only after revalidating their exact raw-run lineage, canonical
frame coverage, source slices, array contracts, and stored decoded digests.
Such a repair must retain the v1 `schema_id`, record
`full_frame_geometry_repair`, explicitly state that array payloads were not
rewritten, and stamp only compatible root/`raw_video` geometry. It must not
promote an incompletely validated artifact to v2 by attribute substitution.

When pixel boxes are present, `bbox_img_xyxy` must be interpreted only with its
declared coordinate metadata. For current fixed clipped detect/refine outputs,
pixel boxes are expected to be source-image pixels for the source clip frame.
If a consumer sees pixel boxes that numerically match an inference resolution,
that is a metadata/data mismatch and should be reported rather than silently
becoming the canonical contract.

Review proxy videos do not change canonical coordinates. A proxy manifest
provides media dimensions for display only. The source detection geometry
remains source-image geometry; frontends scale from source dimensions to proxy
media dimensions for rendering.

## Editing Contract

Editable consumers must write through the curated refined-detect writer
semantics, not by patching only one array.

For clipped finalized collections, a manual edit should update:

- the selected per-clip refined run;
- the canonical sparse `instances` rows;
- source-surface review fields when present;
- manual flags, reason labels, confidence, class id, and status labels
  consistently.

Manual edits are scoped to the selected collection and clip run. They do not
modify `recording_frame_index.parquet`, source videos, or review proxy videos.

Training promotion is a separate write surface. The browser reviewer may call a
promotion hook after an analysis edit, but the analysis edit remains valid even
if promotion fails.

## Validation Checklist

Before treating a clipped finalized collection as reviewable:

- The collection group exists at
  `experiment_index/finalized_runs/<collection_id>`.
- `selected_runs` is non-empty.
- Every selected run has `camera_serial`, `clip_id`, and `refined_group_path`.
- There is at most one selected run per `(camera_serial, clip_id)`.
- `recording_frame_index.parquet` exists and has the required columns.
- Parent frame rows resolve to selected `(camera_serial, clip_id)` pairs.
- Each selected `source.video_path` exists or a valid review-proxy manifest is
  supplied.
- If a review-proxy manifest is supplied, it has an entry for every selected
  `(camera_serial, clip_id)` pair.
- Refined runs expose canonical `instances` for new clipped collections.
- Box coordinates are finite for `present` rows and dimensions are positive.

## Known Consumers

- `fisheye.utils.resolve_clipped_refined_detect_collection` builds an Arrow
  frame/run table from a finalized collection and the recording frame index.
- `fisheye.tune.video_detect_review_web` uses the same contract for video-backed
  inspection, editing, issue navigation, and optional training promotion.
- Crimson should use this resolver contract rather than directory scans or
  top-level detect-run assumptions.
