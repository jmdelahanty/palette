# Acquisition Crop-Video ROI Provider Plan
<!-- contract-meta
status: design
last_verified: 2026-06-29
purpose: Plan how acquisition crop videos become direct ROI pixel providers for keypoint and subject-mask workflows without breaking crop-run lineage contracts.
-->

## Purpose

Palette already supports geometry-only crop runs and temporary ROI caches for
analysis workflows. RedScare adds a related but distinct pixel source:
Orange-produced acquisition crop videos under
`derived/external_crop_recorder/`.

The goal is to let keypoint and subject-mask inference consume acquisition crop
video frames directly when they are valid, while still falling back to recovered
full-frame crops when realtime crop detection missed a frame. Downstream
consumers should continue to see normal Palette row lineage, crop placement, and
full-frame coordinate outputs.

This plan complements:

- `docs/acquisition_video_stream_source_policy.md`
- `docs/geometry_only_crop_workflow_cache_design.md`
- `docs/acquisition_crop_pose_training_workflow.md`
- `docs/keypoint_pixel_contract_registry_todo.md`

## Current State

### Already Implemented

- Analysis archives may use `crop_storage_mode=geometry_only`. Geometry-only
  crop runs store crop row lineage and placement, not permanent ROI pixels.
- Keypoint and subject-mask YOLO/U-Net workflows can consume temporary or flat
  ROI caches instead of permanent `crop_runs/<run>/roi_images`.
- Training Zarrs remain materialized. They should contain dense crop images
  because training/review artifacts are meant to be self-contained.
- Acquisition stream inventory is mirrored into analysis Zarrs as
  `analysis/acquisition_video_streams/`. That surface means the media exists; it
  does not mean any model stage used those pixels.
- `append_acquisition_crop_video_training` can decode sampled acquisition crop
  video frames into a normal materialized `crop_runs/<run>` inside
  `*_training.zarr`.
- `import_acquisition_detections_to_detect_run` can retain acquisition-time crop
  recorder boxes only as an explicit, selector-free
  `detection_artifact_runs/<run>`. It cannot create a normal `detect_runs` child
  until an exact canonical acquisition identity/coordinate binding path exists.
- `CropImageSource` can read acquisition crop-video-backed geometry-only crop
  runs directly from `source_crop_video_frame_indices`.
- `CropImageSource` can also read hybrid acquisition crop runs where some rows
  come from the crop MP4 and some rows come from a supplemental flat ROI cache.
- `build_hybrid_acquisition_offline_crop_run` can create one logical crop
  rowset backed by acquisition crop video plus pre-decoded offline-recovered
  ROIs.

### Observed RedScare State On 2026-06-29

- RedScare has 28 active analysis Zarr rows and 28 active training Zarr rows in
  the `/groups` registry.
- The 28 analysis Zarrs have `analysis/acquisition_video_streams`.
- The 28 analysis Zarrs do not currently have `crop_runs`, `keypoints_runs`, or
  `refined_keypoints_runs`.
- The 28 training Zarrs have acquisition crop-video-backed materialized
  `crop_runs`, plus keypoint/refined-keypoint and subject-mask/refined-subject
  mask surfaces.

### Orange Crop Metadata Clarification On 2026-06-29

Orange's external crop metadata carries both crop-window geometry and selected
live detection geometry:

- `crop_x,crop_y,crop_w,crop_h` are the actual clamped full-frame source ROI
  copied into the crop video. They are canonical crop-window geometry, not fish
  bboxes.
- `detection_x,detection_y,detection_w,detection_h` are the single selected
  postprocessed live detection used to center the crop when
  `has_detection=true`. They are useful provenance, but not the full live
  detection stream and not proof of full-fish crop containment.
- `has_detection=false` and `blank_frame=true` means Orange encoded an explicit
  blank crop frame. The default zero `crop_*` fields on those rows must be
  treated as invalid crop geometry.
- If `Cam<serial>_yolo_events.jsonl` is present, its `detections[]` rows are
  the preferred source for an imported Orange live-detection run. Crop-meta
  `detection_*` is only the selected crop-controller bbox.

The Palette consequence is:

- build analysis `crop_runs` from `crop_*`
- run crop-sufficiency checks against offline refined detections
- import `yolo_events` or crop-meta `detection_*` as a separate online
  detection run only when online-vs-offline bbox quality is explicitly needed
- never run normal detection quality on `crop_*` crop-window geometry

### Gap

There is no direct, validated cluster path that says:

```text
analysis zarr + acquisition crop MP4 + crop-meta CSV
  -> keypoints_runs/refined_keypoints_runs in the analysis zarr
```

The current validated path is:

```text
acquisition crop MP4
  -> materialized crop_runs/<run> in training zarr
  -> training-review bootstrap keypoints/masks
```

For analysis Zarrs, the current cluster keypoint and subject-mask tooling expects
a `crop_runs/<run>` rowset and an image provider. The provider can now be
materialized `roi_images`, a flat/temporary ROI cache, an acquisition crop video,
or a hybrid acquisition crop video plus supplemental flat cache. The remaining
gap is cluster orchestration: building the hybrid crop run/cache and then
submitting downstream model jobs as one reproducible workflow.

## Design Principle

Keep `crop_runs/<run>` as the canonical ROI rowset and lineage surface.

Do not make downstream keypoint or mask consumers reason directly about crop
videos, crop-meta CSV quirks, or realtime-vs-offline missing-frame recovery.
Those details belong in the ROI pixel-provider layer.

In practical terms:

- `crop_runs/<run>` answers: which ROI rows exist, which parent frames they map
  to, where the ROI sits in full-frame coordinates, and which upstream rowset
  produced the geometry.
- ROI pixel providers answer: how to produce the image tensor for each crop row.
- Keypoints and masks write normal Palette outputs with `source_crop_run`,
  `source_crop_row_ids`, full-frame `keypoints_img`, and normal mask placement.

## Pixel Provider Types

### Existing Providers

`materialized_roi_images`

- Reads `crop_runs/<run>/roi_images`.
- Appropriate for training Zarrs and small self-contained review artifacts.

`flat_roi_cache`

- Reads a flat binary cache generated from source video and crop geometry.
- Appropriate for high-throughput cluster inference and repeated downstream
  stages.

`geometry_only_live`

- Decodes source video and crops on demand.
- Useful for small diagnostics but not the preferred high-throughput path.

### New Providers

`acquisition_crop_video`

- Reads frames directly from Orange's acquisition crop MP4.
- Row identity comes from `source_crop_video_frame_indices`.
- Full-frame placement comes from `source_crop_xywh` and
  `roi_coordinates_full`.
- Valid only for rows where crop metadata reports a real crop:
  `has_detection=true` and `blank_frame=false`.

`offline_recovered_full_frame_crop`

- Reads the full-frame camera video and crops using offline refined-detection
  geometry.
- Used for frames where offline detection succeeded but realtime acquisition
  crop detection was missing or blank.
- Should usually be materialized into a temporary flat ROI cache before
  keypoints/masks so repeated downstream stages do not re-decode full video.

`hybrid_acquisition_then_recovered`

- One logical crop run with per-row source kind.
- Uses acquisition crop-video pixels when valid.
- Uses recovered full-frame crops for rows missing from the acquisition crop
  stream.
- Preserves one downstream rowset for keypoints/masks.

## Proposed Crop-Run Contract Extension

A crop-video-backed or hybrid analysis crop run should still live under:

```text
crop_runs/<run>/
```

Required row arrays:

```text
frame_indices                         # zero-based parent recording frame index
source_recording_frame_ids             # acquisition 1-based frame id when available
roi_coordinates_full                   # x,y top-left in full-frame pixels
source_crop_xywh                       # x,y,w,h in full-frame pixels
source_pixel_kind_codes                # integer enum per row; attrs carry labels
supplemental_cache_row_indices         # row in supplemental flat cache, or -1
source_refined_row_ids                 # optional refined-detect row ids for recovered crops
source_crop_meta_row_indices           # required for acquisition crop-video rows
source_crop_video_frame_indices        # required for acquisition crop-video rows
source_crop_local_frame_ids            # Orange local ids, provenance only
```

Recommended QC arrays:

```text
realtime_detection_present
realtime_crop_blank
offline_recovered
recovery_reason_code
bbox_img_xyxy
bbox_norm_coords
bbox_crop_norm_coords
```

Run attrs should include:

```text
crop_storage_mode = "geometry_only"
source_pixels = "hybrid_acquisition_crop_video_offline_supplement" | "acquisition_crop_video"
roi_pixel_provider = "acquisition_crop_video" | "hybrid_acquisition_crop_video_offline_supplement"
source_crop_video_path
source_crop_meta_path or source_crop_meta_array_path
source_video_path
source_video_fingerprint
supplemental_roi_cache_manifest
source_crop_xywh_coordinate_space = "source_image_xywh"
roi_coordinates_full_coordinate_space = "source_image_xy"
source_crop_video_frame_indices_semantics = "zero_based_frame_index_in_acquisition_crop_video"
source_crop_local_frame_ids_semantics = "orange_acquisition_local_frame_id_not_video_frame_index"
supplemental_cache_row_indices_semantics = "row_index_in_supplemental_flat_roi_cache_or_-1_for_acquisition_video_rows"
```

For hybrid runs, `source_pixel_kind_codes` should use a small explicit
vocabulary stored in `source_pixel_kind_code_map`:

```text
acquisition_crop_video = 0
offline_full_frame_supplemental_flat_cache = 1
```

Rows marked `missing_unrecoverable` should either be excluded from model
inference or produce failed output rows with explicit reason codes. They should
not silently reuse stale crop-video pixels.

## Why This Direction

### Storage

Acquisition crop videos already contain useful ROI pixels. Re-decoding the full
frame for every valid realtime crop wastes time and creates unnecessary cache
or Zarr payloads.

### Runtime

For RedScare-like recordings, most rows should be served directly from the crop
MP4. Only realtime-missed rows need full-frame recovery. That should reduce
decode bandwidth and keep the expensive full-frame crop path focused on gaps.

### Contract Stability

Downstream outputs already depend on `crop_runs/<run>` lineage. Keeping that
rowset as the stable boundary avoids teaching every keypoint, mask, review, and
Crimson consumer a separate crop-video contract.

### Review And Training

Training Zarrs can remain dense/materialized. Analysis Zarrs can remain lean.
If an analysis result later becomes a training artifact, the export step can
materialize dense crop images from the declared provider through the same
contract.

## Implementation Checklist

### Phase 0: State Audit

- [x] Verify RedScare analysis Zarrs have acquisition stream inventory but no
  analysis `crop_runs`.
- [x] Verify RedScare training Zarrs already have acquisition crop-video-backed
  materialized crop runs and downstream review surfaces.
- [x] Add a registry/report command that summarizes, per recording:
  acquisition crop stream available, analysis crop run present, training crop
  run present, crop-video-backed keypoints present, and offline detection
  coverage.
- [x] Identify recordings where crop video dimensions are large enough for the
  current keypoint/subject-mask model input policy.
- [x] Clarify Orange crop-meta semantics for crop-window geometry versus
  selected live detection bbox geometry.

Phase 0 RedScare report command:

```bash
scripts/py -m fisheye.utils.report_acquisition_crop_video_roi_readiness \
  --registry /groups/johnson/johnsonlab/jeremy/registries/palette_registry.sqlite \
  --path-contains RedScare \
  --output-jsonl /tmp/redscare_acquisition_crop_video_roi_readiness_20260629.jsonl \
  --limit 40
```

Observed Phase 0 result:

- 28/28 RedScare recordings have analysis and training Zarr rows.
- 28/28 analysis Zarrs have acquisition crop-video stream inventory.
- 28/28 crop videos and crop-meta CSVs are present.
- 28/28 crop videos meet the current `384x384 >= 348` size policy.
- 28/28 analysis Zarrs have offline detect and refined-detect surfaces.
- 28/28 training Zarrs have crop/keypoint/refined-keypoint/subject-mask/refined
  subject-mask review surfaces.
- 0/28 analysis Zarrs have analysis `crop_runs`.

Uniform next action from the report:

```text
build_analysis_acquisition_crop_run
```

### Phase 1: Analysis Crop-Run Builder

- [x] Add a dry-run-first tool that builds an analysis `crop_runs/<run>` rowset
  from acquisition crop metadata without writing `roi_images`.
- [x] The builder must reject or mark rows with `blank_frame=true`,
  `has_detection=false`, non-finite crop geometry, or crop-video frame-index
  gaps.
- [x] Use `crop_x,crop_y,crop_w,crop_h` as canonical crop-window geometry.
  Do not treat those fields as fish bboxes.
- [x] Preserve `detection_x,detection_y,detection_w,detection_h` separately as
  selected live detection provenance when present.
- [x] Write row arrays for crop-video frame indices, crop-meta row indices,
  source crop geometry, and full-frame placement.
- [x] Use canonical `bbox_norm_coords` only for full-frame-normalized boxes.
  Keep crop-frame-normalized boxes under `bbox_crop_norm_coords`.
- [x] Mark the run as `crop_storage_mode=geometry_only`.
- [x] Record source video, crop video, crop metadata, pixel-provider identity,
  source dimensions, and crop-coordinate semantics in attrs/provenance.
- [ ] Refresh registry crop quality/status after apply.

Implemented builder:

```bash
scripts/py -m fisheye.utils.build_analysis_acquisition_crop_run
```

RedScare registry dry-run on 2026-06-29:

```bash
scripts/py -m fisheye.utils.build_analysis_acquisition_crop_run \
  --source registry \
  --registry /groups/johnson/johnsonlab/jeremy/registries/palette_registry.sqlite \
  --path-contains RedScare \
  --output-jsonl /tmp/redscare_build_analysis_acquisition_crop_run_dryrun_20260629.jsonl
```

Dry-run result:

```text
records: 28
applied: False
selected_rows_total: 3,870,334
rejected_blank_crop_frame_total: 39,302
rejected_crop_has_no_detection_total: 0
rejected_nonfinite_crop_geometry_total: 0
rejected_nonfinite_detection_geometry_total: 0
source_shape: 4512x4512 for all 28 recordings
```

The builder writes `source_pixel_kind_codes` and `crop_state_codes` integer
arrays with attrs maps rather than variable-length string arrays.

Approved RedScare registry apply on 2026-06-29:

```bash
scripts/py -m fisheye.utils.build_analysis_acquisition_crop_run \
  --source registry \
  --registry /groups/johnson/johnsonlab/jeremy/registries/palette_registry.sqlite \
  --path-contains RedScare \
  --run-name crop_acquisition_crop_video_analysis_redscare_20260629_01 \
  --output-jsonl /tmp/redscare_build_analysis_acquisition_crop_run_apply_20260629.jsonl \
  --apply
```

Apply result:

```text
records: 28
applied: True
selected_rows_total: 3,870,334
rejected_blank_crop_frame_total: 39,302
rejected_crop_has_no_detection_total: 0
rejected_nonfinite_crop_geometry_total: 0
rejected_nonfinite_detection_geometry_total: 0
run_name: crop_acquisition_crop_video_analysis_redscare_20260629_01
```

Post-apply readiness report moved all 28 RedScare recordings to:

```text
run_analysis_keypoints_from_roi_provider
```

Spot-check on `2026-06-23T16-01-09Z_arena_1_RedScare` confirmed
`crop_runs.latest_any = crop_acquisition_crop_video_analysis_redscare_20260629_01`,
`crop_storage_mode=geometry_only`, `source_pixels=acquisition_crop_video`, and
`palette_run_completion_status=complete`. `latest` and `latest_materialized`
remain unset for this analysis crop parent because the new run has no
materialized `roi_images`.

### Phase 2: Acquisition Crop-Video Pixel Provider

- [x] Use the existing shared `CropImageSource` as the first provider interface
  rather than adding a parallel abstraction.
- [x] Implement provider mode `acquisition_crop_video`.
- [x] Decode selected crop-video frames using
  `source_crop_video_frame_indices`, not Orange local frame IDs.
- [x] Process acquisition crop-video rows as provider-local batches, not by
  switching decoders row-by-row inside model inference.
- [x] Validate decoded frame dimensions against crop-run `roi_size` and crop
  stream metadata.
- [x] Return the same tensor/image shape expected by current keypoint and
  subject-mask model paths.
- [x] Record `source_roi_read_mode=acquisition_crop_video` and
  `source_roi_pixel_contract_name` in keypoint/mask run attrs and registry
  performance tables.
- [ ] Add parity diagnostics comparing a sampled materialized training crop run
  to the direct acquisition crop-video provider for the same recording.

Implementation note, 2026-06-29:

`CropImageSource` now detects geometry-only crop runs with
`source_pixels=acquisition_crop_video` / `roi_pixel_provider=acquisition_crop_video`,
opens `source_crop_video_path` through the sequential PyNvVideoCodec luma reader,
and maps crop rows through `source_crop_video_frame_indices`. It returns
`[N,H,W] uint8` luma ROI pixels directly from the acquisition crop MP4; it does
not apply a second full-frame crop to those pixels. Temporary ROI cache promotion
is intentionally disabled for acquisition crop-video-only runs.

The provider records the `orange_mono_pynvvc_luma_uint8_v1` pixel contract through
the existing `CropImageSource` provenance path. Unit tests cover the row mapping
with a fake PyNvVC reader. A workstation real-video smoke reached the provider
path but could not create an NVDEC decoder because CUDA was not visible in that
session (`torch.cuda.is_available() == False`); the real-video smoke should be
run on a GPU node.

### Phase 3: Offline Recovery Provider

- [x] Compare realtime crop metadata coverage to offline refined detections.
- [x] Define recovery row selection:
  offline detection exists, realtime crop is missing/blank, and offline bbox is
  inside a valid dish/ROI policy.
- [x] Add `offline_full_frame_supplemental_flat_cache` rows using full-frame
  refined-detection geometry.
- [x] Pre-stage recovered rows into a supplemental flat ROI cache before model
  inference.
- [x] Prefer PyNvVideoCodec indexed decode for sparse recovery frames, with
  sequential full-frame decode as the safe fallback when indexed decode is
  unavailable or inconsistent.
- [ ] Prefer node-local scratch for the recovery cache in cluster jobs; publish
  only downstream model outputs and cache manifests unless the user explicitly
  requests persistent recovered crop pixels.
- [x] Record recovery source refined-detect run, row ids, source video
  fingerprint, crop policy, and cache manifest.
- [ ] Ensure recovered rows produce the same downstream coordinate contract as
  acquisition crop-video rows.

Implemented tool:

```bash
scripts/py -m fisheye.utils.build_hybrid_acquisition_offline_crop_run \
  <analysis.zarr> \
  --acquisition-crop-run <crop_video_run> \
  --refined-detect-run <refined_detect_run> \
  --run-name <hybrid_crop_run> \
  --decode-mode auto \
  --apply
```

`--decode-mode auto` chooses indexed PyNvVideoCodec reads when recovery frames
are sparse. Indexed mode requests exact frame indices through
`SimpleDecoder.get_batch_frames_by_index(...)`; if that path fails in `auto`
mode, the tool falls back to sequential decode. Explicit `--decode-mode indexed`
fails fast instead of silently changing the access pattern.

### Phase 4: Hybrid Provider

- [x] Build one logical crop run that combines acquisition crop rows and
  recovered rows.
- [x] Add `source_pixel_kind_codes`, `supplemental_cache_row_indices`, and
  per-kind row counts to run attrs and summary.
- [x] Classify rows before inference into provider groups:
  `acquisition_crop_video` and
  `offline_full_frame_supplemental_flat_cache`.
- [x] Let the shared crop reader return mixed rows in requested crop-run order:
  acquisition crop-video rows from the crop MP4 and recovered rows from the
  supplemental flat cache.
- [ ] Ensure keypoint and subject-mask inference can read mixed rows without
  reordering or dropping source row identity by writing explicit
  `source_crop_row_ids` / crop-run row ids in outputs.
- [x] Fail clearly if a row's declared provider cannot produce pixels.
- [x] Add a smoke test where some rows are served from crop video and some rows
  from recovered full-frame crops.

Decoder scheduling rule:

Do not alternate between crop-video decoding and full-frame decoding inside the
same tight model-inference row loop. That would force decoder seeks/context
switching and make the stage slow. The intended execution is:

1. Build the crop-run rowset and classify row provider kind.
2. If recovered rows exist, build their supplemental flat cache first. Prefer
   indexed PyNvVideoCodec reads for sparse frame recovery; fall back to
   sequential full-frame decode only when indexed decode is unavailable or
   unsafe.
3. Run acquisition crop-video rows as a crop-MP4 provider block.
4. Run recovered rows from the flat cache provider block.
5. Merge outputs by crop-run row id so downstream contracts stay row-stable.

For the initial RedScare analysis crop runs created on 2026-06-29, the crop run
contains only usable acquisition crop-video rows. Blank rows were excluded, so
the first implementation can be crop-video-provider-only. The recovery/hybrid
path is the next extension when we intentionally add offline-recovered rows for
blank/insufficient crop-video frames.

### Phase 5: Cluster Integration

- [ ] Add a submitter for analysis crop-video crop-run creation.
- [ ] Add keypoint and subject-mask submitter flags for
  `--roi-pixel-provider acquisition_crop_video|hybrid`.
- [ ] Preserve the existing node-local scratch policy for temporary caches.
- [ ] Add timing/provenance fields for crop-video decode, recovered-crop cache
  build, model inference, and finalization.
- [ ] Refresh registry step status and keypoint/mask performance views after
  successful runs.

### Phase 6: Review And Export

- [ ] Confirm Crimson can place keypoints/masks from crop-video-backed analysis
  runs using `source_crop_row_ids` and `roi_coordinates_full`.
- [ ] Confirm web review can assign and edit outputs without caring whether ROI
  pixels came from crop video or recovered full-frame crops.
- [ ] Add export support that materializes dense training crops from
  acquisition/hybrid providers when promoting analysis outputs to training
  artifacts.
- [ ] Keep training Zarr outputs dense by default and record source provider
  provenance in the training manifest.

## Acceptance Criteria

- A RedScare analysis Zarr can hold a geometry-only acquisition crop-video crop
  run without permanent `roi_images`.
- Keypoint inference can run from that crop run by decoding the acquisition crop
  MP4 directly.
- Subject-mask inference can use the same rowset/provider.
- Offline-recovered rows can be mixed into the same logical rowset without
  breaking `frame_indices`, `source_crop_row_ids`, or full-frame coordinate
  outputs.
- Registry performance rows expose the ROI pixel contract and read mode used by
  the model.
- Training exports remain dense/materialized, even when the analysis source was
  compact or provider-backed.

## Non-Goals

- Do not make `analysis/acquisition_video_streams` imply model input use.
- Do not make every consumer parse crop-meta CSVs.
- Do not change the meaning of `bbox_norm_coords`.
- Do not make training Zarrs geometry-only.
- Do not remove the existing materialized crop-run path.
- Do not require direct crop-video support in legacy traditional keypoint code
  before the YOLO/cache-backed path is validated.

## Open Questions

- Should crop-video provider rows be allowed to include blank frames as negative
  training samples, or should analysis inference rowsets exclude them by
  default?
- Should recovered full-frame crops be stored in the same flat cache as
  acquisition crop-video decoded rows, or should cache manifests remain
  provider-specific?
- Should a hybrid crop run keep all acquisition crop-video valid rows, or should
  it be restricted to the rowset selected by offline refined detections?
- How much crop-video/full-frame pixel parity is required before this becomes a
  default RedScare analysis path?
