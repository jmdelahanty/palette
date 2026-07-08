# Clipped Collection ROI Cache Model Workflow

Date: 2026-07-07

Status: design and readiness checkpoint. Do not persist production clipped
collection ROI caches until the blockers below are closed or explicitly waived.

## Goal

Long clipped recordings already have per-clip detection quality and refined
detection outputs. The desired downstream path is to reuse those finalized
refined detections as crop geometry, build temporary flat ROI caches in clip
shards, then run keypoint and subject-mask models without redoing full-frame
detection/refinement.

The target model input path is:

1. Resolve `refined_detect_runs.latest_collection` or an explicit finalized
   clipped collection id.
2. Build one flat ROI cache per clip or shard from the collection row geometry.
3. Run keypoints from the shard cache.
4. Run refined keypoints.
5. Run subject-mask inference from the same shard cache.
6. Run refined subject-mask finalization.
7. Preserve enough lineage that outputs can be mapped back to clip-local frames,
   parent recording frames, refined-detection rows, and source clip videos.

## Current Implementation Surface

Existing code already supports the most important pixel operation:

- `src/fisheye/shared/clipped_collection_flat_roi_cache.py` builds a
  `flat_bin_v1` ROI cache from a finalized clipped refined-detect collection.
- `src/fisheye/utils/build_clipped_collection_flat_roi_cache.py` is the CLI
  wrapper.
- `scripts/submit_clipped_collection_flat_roi_cache_bsub.sh` submits a single
  LSF job that builds on node-local scratch and publishes payload, row-index
  Parquet, and manifest in manifest-last order.
- `scripts/submit_clipped_collection_flat_roi_cache_bundle_bsub.sh` submits one
  LSF job that runs several independent clip-cache builders concurrently on the
  same GPU allocation. Each child writes its own flat cache, row-index parquet,
  progress log, status JSON, and manifest; the parent job writes a bundle
  summary JSON.
- `scripts/submit_clipped_collection_flat_roi_cache_bundles_bsub.sh` resolves a
  clip set, splits it into multiple bundle jobs, and submits the whole set to
  LSF. This is the scheduler-facing wrapper for an entire clipped recording.

The cache row index already carries most required lineage:

- `work_unit_id`
- `camera_serial`
- `clip_id`
- `clip_index`
- `clip_local_frame_index`
- `recording_frame_id`
- `parent_frame_index`
- `timestamp`
- `timestamp_sys`
- `refined_group_path`
- `refined_detect_run`
- `refined_instance_row_index`
- `refined_row_id`
- `source_detect_row_index`
- `bbox_norm_*`
- `roi_x`, `roi_y`, `roi_w`, `roi_h`
- `video_path`

The flat cache manifest records:

- `source.source_kind = finalized_clipped_refined_detect_collection`
- `source.collection_id`
- `source.collection_path`
- `source.recording_frame_index`
- `source.selected_run_count`
- `source.clip_count`
- `source.selection.clip_ids`
- `source.selection.work_unit_ids`
- `source.crop_storage_mode = geometry_only_collection_derived`
- `source.frame_source_kind = clip_source_video_path`
- `row_index.path`
- `builder.pixel_contract`

## Gaps Found

### 1. Clip-shard filter is partial

The builder accepts `--collection-id`, `--clip-id`, `--work-unit-id`, and
`--limit-rows`. It can build an exact per-clip or per-work-unit cache and
records the selection in the cache key and manifest. It still does not support
generic modulo sharding through `--shard-index` / `--shard-count`.

This matters because a full Sleepyfish-sized collection can be hundreds of GiB
as raw `512x512 uint8` ROI rows. Clip-sharded caches are the right unit for
parallel execution, failure recovery, and cleanup.

### 1a. Small smoke throughput is not steady-state throughput

The 2026-07-07 Sleepyfish clipped-cache cluster smokes showed why tiny
`--limit-rows` runs should not be interpreted as steady-state performance:

| Rows | Payload | Overall builder rate | Decode/read rate | Note |
| ---: | ---: | ---: | ---: | --- |
| 256 | 64 MiB | 24.1 ROI/s | 176.8 frames/s | dominated by startup and per-batch progress logging |
| 1024 | 256 MiB | 100.7 ROI/s | 190.8 frames/s | startup amortized; closer to steady state |

Earlier full-recording GoodCopBadCop flat-cache builds were about
175-180 ROI/s, with `decode_seconds_total` as the dominant timing bucket. The
Sleepyfish 1024-row smoke is consistent with that once fixed startup/Torch/NVDEC
initialization is separated from per-frame decode. Downstream keypoint numbers
such as 212-276 poses/s are not cache-build rates; they measure model inference
after a flat cache already exists.

### 2. Downstream consumers still require a crop run

`CropImageSource.open()` resolves a real `crop_runs/<run>` before activating a
flat cache. If `crop_run` is omitted and a flat-cache manifest is supplied, the
current helper tries to infer `source.crop_run_name` from the manifest. Clipped
collection cache manifests intentionally do not have `source.crop_run_name`.

Therefore a clipped collection cache cannot currently be passed directly to the
keypoint or subject-mask runners unless a compatible crop-run-like surface also
exists.

### 3. Output lineage is copied from crop runs, not cache row indexes

Keypoint and subject-mask writers call `copy_row_lineage_arrays(...)` against
the resolved crop run. They do not yet promote clipped cache row-index fields
into output arrays.

For clipped runs, downstream outputs should be self-describing, not dependent on
an external cache manifest remaining available forever. At minimum, keypoint and
mask outputs from collection caches should persist:

- `frame_indices` in the parent recording frame domain.
- `source_frame_indices` in the parent recording frame domain.
- `source_clip_indices`.
- `source_clip_local_frame_indices`.
- `source_refined_row_ids`.
- `source_detect_row_index`.
- `source_crop_row_ids` only when a real or virtual crop run exists.

Future additions should also consider string/code arrays for `clip_id`,
`work_unit_id`, and `refined_group_path`, or a compact lookup table plus integer
codes.

### 4. Registry stage rows lack collection-cache targeting fields

The registry can identify clipped datasets through `datasets.source_layout` and
`datasets.source_recording_frame_index_path`, but keypoint/mask performance
extractors currently record only general crop/cache fields such as:

- `source_crop_run`
- `source_roi_read_mode`
- `roi_cache_policy`
- `source_roi_cache_used`
- `source_roi_cache_backend`

They do not record:

- `source_collection_id`
- `source_collection_path`
- `source_clip_id`
- `source_clip_index`
- `source_work_unit_id`
- `source_roi_cache_manifest`
- `source_roi_cache_row_index_path`
- `source_roi_cache_source_kind`
- `source_recording_frame_index_path`

Without those fields, registry queries cannot reliably answer which clipped
collection or shard produced a keypoint/mask run.

### 5. Clipped metadata is incomplete or stale on existing datasets

Read-only inspection of the PRFS registry showed:

- `datasets.source_layout = rolling_clips` exists and can target clipped data.
- Some clipped training rows still have
  `source_recording_frame_index_path` pointing to `/nvme1`.
- The Sleepyfish cam2010095 analysis Zarr row exists but does not currently
  carry `source_layout = rolling_clips` in the registry.
- The Sleepyfish cam2010095 analysis Zarr root attrs lack modern clipped attrs:
  `recording_id`, `zarr_use`, `source_layout`,
  `source_recording_frame_index_path`, and `source_frame_index_schema`.
- The recording root contains `recording_frame_index.parquet` and
  `recording_frame_index_manifest.json`, so this is backfillable.

The finalized collection selected-run `source.video_path` entries for the
inspected cam2010095 collection point to `/groups`, so the cache builder should
decode the correct MP4s. The stale `/nvme1` frame-index paths remain a registry
and relocation hygiene issue.

## Recommended Architecture

Use a crop-run-compatible proxy layer rather than forcing every model runner to
understand finalized clipped collections immediately.

For each clip shard, create a lightweight geometry-only proxy crop run:

```text
crop_runs/<proxy_run>/
  frame_indices
  roi_coordinates_full
  detection_indices
  source_frame_indices
  source_clip_indices
  source_clip_local_frame_indices
  source_refined_row_ids
  source_detect_row_index
```

Suggested attrs:

```text
crop_storage_mode = geometry_only
source_kind = finalized_clipped_refined_detect_collection
source_collection_id = <collection id>
source_collection_path = experiment_index/finalized_runs/<collection id>
source_clip_id = <clip id or shard label>
source_clip_index = <clip index if single clip>
source_work_unit_id = <work unit if single work unit>
source_recording_frame_index_path = <recording_frame_index.parquet>
source_roi_cache_required = true
source_roi_cache_manifest = <published flat cache manifest, optional after build>
source_roi_cache_row_index_path = <published rows.parquet, optional after build>
crop_policy = centered_refined_bbox
```

Benefits:

- Existing keypoint/mask code can continue to use `CropImageSource`.
- `source.crop_run_name` can be present in the flat cache manifest.
- Existing row-lineage copy helpers can propagate canonical row arrays.
- Crimson/registry tooling can reason about a normal crop-run source while
  still seeing clipped collection provenance.

### Proxy crop-run plus manifest-alias bridge

The current keypoint and subject-mask runners cannot consume a clipped
collection cache manifest by itself. They first resolve a real
`crop_runs/<run>` and then validate any supplied flat cache against that run.
The validation checks `source.crop_run_name` in the cache manifest, while the
output writers copy row lineage from the crop run rather than from the cache
row-index parquet.

The minimal bridge therefore has two artifacts per cache shard:

1. A geometry-only proxy crop run in the analysis Zarr.
2. A small manifest alias that points at the existing cache `.bin` payload but
   stamps `source.crop_run_name = <proxy_run>`.

The alias should not copy the binary payload. It should only rewrite enough
manifest metadata for the standard `open_flat_roi_cache(...,
expected_crop_run=<proxy_run>)` path to validate the cache against the proxy
crop run. The original clipped manifest remains the provenance source; the
alias exists to adapt it to the standard Palette crop-run contract.

Proxy crop runs should be generated directly from the cache row-index parquet:

```text
frame_indices                  = parent_frame_index
source_frame_indices           = parent_frame_index
source_clip_indices            = clip_index
source_clip_local_frame_indices= clip_local_frame_index
source_refined_row_ids         = refined_row_id or refined_instance_row_index
source_detect_row_index        = source_detect_row_index
detection_indices              = roi_row_index
source_crop_row_ids            = arange(n_rois)
roi_coordinates_full           = [roi_x, roi_y]
```

Required proxy attrs:

```text
crop_storage_mode = geometry_only
source_kind = finalized_clipped_refined_detect_collection_proxy
source_collection_id = <collection id>
source_collection_path = experiment_index/finalized_runs/<collection id>
source_clip_id = <clip id>
source_clip_index = <clip index>
source_roi_cache_manifest = <original manifest>
source_roi_cache_alias_manifest = <alias manifest>
source_roi_cache_row_index_path = <rows.parquet>
source_roi_cache_required = true
crop_policy = centered_refined_bbox
roi_shape = [512, 512]
```

The proxy run is intentionally metadata-only. It does not own pixels and should
be treated as invalid for model inference unless the corresponding cache alias
is supplied.

Proxy crop runs are written under `crop_runs` for compatibility with existing
`source_crop_run` attrs and `CropImageSource`, but they are not ordinary crop
stage outputs. They should set `stage_selector_eligible=false`,
`proxy_crop_complete=true`, and a non-complete run-completion status such as
`palette_run_completion_status=auxiliary`. They may set legacy
`status=completed` so old cleanup scripts do not remove them, but standard
completion resolvers must not treat them as `latest_complete` crop runs.

### Sharded model runs and latest-pointer safety

Running one model job per clip cache shard is the right compute strategy, but
those shard outputs are not equivalent to a normal whole-recording
`keypoints_runs.latest_complete` or `subject_mask_runs.latest_complete`.

Current code publishes normal selectors as soon as a keypoint run starts and
completes:

- `detect_keypoints_yolo._prepare_run_group()` calls `note_pending_latest(...)`.
- `detect_keypoints_yolo.detect_keypoints_yolo()` writes
  `root.attrs["current_keypoint_group_path"]`.
- `zarr_run_completion.mark_run_complete(..., parent_group=..., run_name=...)`
  sets both `latest_complete` and `latest` on the parent.

The same pattern exists in subject-mask paths:

- `infer_unet_subject_masks._prepare_run_group()` calls
  `note_pending_latest(...)`.
- `subject_segmentation._prepare_run_group()` calls
  `note_pending_latest(...)`.
- `run_sam_subject_masks` calls `note_pending_latest(...)` when creating
  `subject_mask_runs/<run>`.
- `refined_subject_mask_review` calls `note_pending_latest(...)` when creating
  `refined_subject_masks_runs/<run>`.
- These writers complete through `mark_run_complete(..., parent_group=...,
  run_name=...)`, so normal completion publishes parent selectors.

Therefore, if 22 clipped-cache shards are written as ordinary top-level
`keypoints_runs/<run>`, `subject_mask_runs/<run>`, or
`refined_subject_masks_runs/<run>` groups, the last shard to finish becomes the
apparent latest complete run for the whole analysis Zarr. That is incorrect:
the run covers only one clip shard, and downstream consumers that resolve
`latest_complete` would silently see a partial collection.

Preferred rule:

- Shard outputs must be complete runs, but they must not publish global
  parent selectors.
- Shard outputs must not be discoverable as ordinary latest-complete stage
  runs through resolver fallback scans.
- A collection-level finalizer, not an arbitrary shard, is responsible for
  publishing the selector that ordinary consumers resolve.

Important resolver caveat:

`resolve_latest_complete_run_name()` first checks `latest` and
`latest_complete`, then falls back to scanning completed child groups in reverse
name order. Therefore, simply skipping `note_pending_latest()` and calling
`mark_run_complete(..., parent_group=None)` is not enough if shard groups live
directly under `keypoints_runs` or `subject_mask_runs`: a store with no valid
parent selector could still resolve the newest completed shard as the latest
run.

Recommended implementation:

1. Prefer writing shard outputs under explicit shard parents such as
   `keypoint_shard_runs`, `subject_mask_shard_runs`, and
   `refined_subject_mask_shard_runs`, not under the ordinary stage parents.
   These shard parents can use the same completion contract internally without
   interfering with normal stage resolution.
2. If shards must live under ordinary stage parents, add and enforce a generic
   selector-eligibility attr such as `stage_selector_eligible=false`, and update
   all standard run resolvers to skip non-eligible children during fallback.
   This is broader and riskier than a separate shard parent because custom
   readers may still scan child groups directly.
3. In shard mode, create the run and stamp completion status, but do not update
   root `current_*_group_path` attrs or ordinary stage parent selectors.
4. Stamp shard attrs:
   `is_collection_shard=true`, `source_collection_id`, `source_clip_id`,
   `source_clip_index`, `source_crop_run`, `source_roi_cache_alias_manifest`,
   and `source_roi_cache_row_index_path`.
5. Add a collection finalizer that validates all expected shard runs exist,
   verifies row-lineage coverage, and then writes either:
   - a collection index run that lists the shard run paths, or
   - a merged whole-collection run if downstream tools require a single flat
     row surface.
6. Only the finalizer may update `latest_complete`, `latest`, and any
   authoritative pointer for the stage.

Open design decision:

- A collection index run is cheaper and avoids rewriting millions of rows, but
  consumers must learn to follow shard manifests.
- A merged whole-collection run is more compatible with existing consumers but
  duplicates arrays and may be expensive for masks.

For the first implementation, prefer separate shard parents plus a
collection-level finalizer. For keypoints, a merged whole-collection
`keypoints_runs/<run>` is likely cheap enough and maximally compatible. For
subject masks, a collection index or compact merged representation is safer than
materializing a huge dense whole-collection mask run.

### Keypoint shard mode and finalizer design

Keypoint inference should be split into two responsibilities:

1. GPU shard inference against one proxy crop run plus one cache alias.
2. CPU finalization that publishes a normal whole-collection keypoint run.

Shard inference output:

```text
keypoint_shard_runs/<run>/
  keypoints_roi
  keypoints_img
  keypoints_norm
  keypoint_confidences
  confidence
  detection_success
  heading
  heading_finite
  heading_usable
  pose_bbox_xyxy_roi
  effective_threshold
  effective_se2_radius
  frame_indices
  source_frame_indices
  source_clip_indices
  source_clip_local_frame_indices
  source_crop_row_ids
  source_refined_row_ids
  source_detect_row_index
  detection_indices
```

Shard attrs:

```text
is_collection_shard = true
stage_selector_eligible = false
source_collection_id = <collection id>
source_clip_id = <clip id>
source_clip_index = <clip index>
source_crop_run = <proxy crop run>
source_roi_cache_alias_manifest = <alias manifest>
source_roi_cache_row_index_path = <rows.parquet>
```

The shard writer should reuse the existing keypoint inference logic, but it
must make the output parent explicit. In shard mode it writes to
`keypoint_shard_runs`, does not call `note_pending_latest()` on
`keypoints_runs`, and does not update `root.attrs["current_keypoint_group_path"]`.
The shard parent may have its own `latest`/`latest_complete` attrs if useful for
debugging, because normal Palette consumers do not resolve that parent. Keypoint
shard mode also suppresses the canonical keypoint registry/status refresh; a
shard is a staging artifact until a finalizer publishes a normal
`keypoints_runs/<collection_run>`.

Implemented v1 utility:

```bash
scripts/py -m fisheye.utils.finalize_keypoint_shards \
  /path/to/recording_analysis.zarr \
  --shard-run keypoint_shard_... \
  --output-run keypoints_collection_... \
  --json
```

For whole-collection output, first merge per-clip proxy crop runs into one
collection-level proxy crop run:

```bash
scripts/py -m fisheye.utils.merge_clipped_proxy_crop_runs \
  /path/to/recording_analysis.zarr \
  --source-crop-run crop_proxy_clip_000000 \
  --source-crop-run crop_proxy_clip_000001 \
  --output-run crop_proxy_collection_... \
  --json
```

Then finalize keypoint shards against that merged proxy:

```bash
scripts/py -m fisheye.utils.finalize_keypoint_shards \
  /path/to/recording_analysis.zarr \
  --shard-run keypoint_shard_clip_000000 \
  --shard-run keypoint_shard_clip_000001 \
  --target-crop-run crop_proxy_collection_... \
  --output-run keypoints_collection_... \
  --json
```

The finalizer writes a normal `keypoints_runs/<collection_run>` from completed
`keypoint_shard_runs`. Without `--target-crop-run`, it still requires all
shards to reference the same `source_crop_run`. With `--target-crop-run`, it
allows mixed per-clip proxy crop runs by mapping each shard row onto the merged
proxy crop run using clip/refined/frame row identity. The published keypoint run
then carries one downstream-compatible `source_crop_run`, which keeps current
`refine_keypoints.py`, Crimson, registry extractors, and review/export tools on
the ordinary single-crop-run contract.

The implemented v1 finalizer:

1. Resolves explicit `--shard-run` names or a JSON `--shard-runs-file`.
2. Verifies every shard is complete when completion attrs are present.
3. Requires core model-output arrays, `frame_counts`, `n_rois`, `n_keypoints`,
   `source_crop_run`, and `source_crop_row_ids`.
4. Requires schema/model compatibility attrs to match across shards.
5. Fails on mixed `source_crop_run` values unless `--target-crop-run` is set.
6. When rebasing, verifies each source proxy crop row maps to exactly one row in
   the target merged proxy crop run.
7. Fails on duplicate rebased `source_crop_row_ids`.
8. Sorts rows by target `source_crop_row_ids` for deterministic crop-row order.
9. Concatenates all per-ROI arrays and copied row-lineage arrays.
10. Sums frame-domain `frame_counts` and `n_rois`, then recomputes
   `n_keypoints` from merged `frame_indices` and `detection_success`.
11. Stamps `source_keypoint_shard_runs`, `source_keypoint_shard_run_paths`,
    `source_crop_run`, `source_keypoint_shard_crop_runs`, and
    `collection_finalizer_schema`.
12. Publishes `keypoints_runs.latest_complete`, `keypoints_runs.latest`, and
    `root.attrs["current_keypoint_group_path"]` only after validation passes.

This preserves the existing downstream contract while retaining clip-local
lineage in arrays such as `source_clip_indices`,
`source_clip_local_frame_indices`, `source_refined_row_ids`, and
`source_detect_row_index`.

This merged keypoint run is intentionally compatible with existing refinement,
review, Crimson, registry extractors, and training exporters. Current
`refine_keypoints.py` hard-codes `root["keypoints_runs"][run]`, so asking it to
consume `keypoint_shard_runs` directly would spread collection-awareness into
more code than needed.

Subject masks should not blindly copy this finalizer shape. Merging keypoints is
cheap; merging dense masks can be enormous. For masks, prefer shard outputs plus
a collection index or compact merged representation unless a concrete consumer
requires a dense whole-collection run.

## Registry Targeting Policy

Clipped recordings should remain singleton recording/dataset entities. Clips are
child artifacts and execution shards, not separate recordings.

Recommended targeting fields:

- `datasets.source_layout = rolling_clips`
- `datasets.source_recording_frame_index_path`
- `datasets.source_frame_index_schema`
- stage rows:
  - `source_collection_id`
  - `source_collection_path`
  - `source_clip_id` or `source_shard_id`
  - `source_roi_cache_manifest`
  - `source_roi_cache_row_index_path`
  - `source_roi_cache_source_kind`

If cache lifecycle tracking becomes important, add a small
`roi_cache_artifacts` registry table. Do not force temporary workflow caches
into `recording_artifacts`, which is recording-folder-artifact oriented.

## Implementation Checklist

### Phase 0: readiness audit and metadata repair

- [x] Add a read-only audit for clipped collection cache readiness.
- [x] Check dataset registry rows for clipped attrs.
- [x] Check Zarr root attrs for clipped attrs.
- [x] Check `recording_frame_index.parquet` existence.
- [x] Check finalized collection existence and selected-run source paths.
- [x] Report stale `/nvme1` paths separately from blockers.
- [ ] Backfill active clipped-analysis root attrs before building caches:
  `recording_id`, `zarr_purpose=analysis`, `source_layout=rolling_clips`,
  `recording_frame_index_path`, `source_recording_frame_index_path`, and
  `source_frame_index_schema`.
- [ ] Backfill the canonical registry `datasets` row for the same analysis Zarr
  with `zarr_use=analysis`, `source_layout=rolling_clips`,
  `source_recording_frame_index_path`, and `source_frame_index_schema`.
- [ ] Rewrite active path columns in `recording_frame_index.parquet` and
  `recording_frame_index_manifest.json` when they still point at `/nvme1`
  after relocation. Preserve historical provenance elsewhere.
- [ ] Rerun the readiness audit after repair. Cache/model execution should start
  only after blockers are zero and stale active path warnings are resolved.

### Phase 1: clip-sharded cache build

- [x] Add builder filters: `clip_ids`, `work_unit_ids`.
- [ ] Add generic modulo filters: `shard_index`, `shard_count`.
- [x] Include clip/work-unit selection in the cache key and manifest.
- [x] Add wrapper flags: `--clip-id`, `--work-unit-id`.
- [ ] Add wrapper flags: `--shard-index`, `--shard-count`.
- [x] Add a bundle LSF submitter that launches multiple clip-filtered builders
  in one GPU job.
- [x] Add tests that clip/work-unit filters select deterministic rowsets and
  fail closed for missing values.

### Bundle Decode Parallelism

NVIDIA L4 has multiple NVDEC engines, but a single clipped-cache build uses one
sequential PyNvVideoCodec reader. The safe way to expose decoder parallelism is
not to write a shared flat cache from multiple processes. Instead:

- Request one GPU allocation.
- Launch up to `--max-workers 4` child builders inside that job.
- Give each child exactly one `--clip-id` selection.
- Publish one manifest/bin/rows triplet per child clip.
- Treat the bundle status JSON as an orchestration artifact, not as a merged
  cache contract.

Example:

```bash
scripts/submit_clipped_collection_flat_roi_cache_bundle_bsub.sh \
  --zarr /groups/johnson/johnsonlab/jeremy/recordings/sleepyfish_2026_05_05_17_45_30_cam2010095/zarr/sleepyfish_2026_05_05_17_45_30_cam2010095_analysis.zarr \
  --collection-id sleepyfish_cam2010095_allclips_pynvvc_fixed_20260522_01 \
  --clip-id clip_000000 \
  --clip-id clip_000001 \
  --clip-id clip_000002 \
  --clip-id clip_000003 \
  --public-cache-dir /nrs/ahrens/palette_staging/clipped_collection_flat_roi_cache/sleepyfish_cam2010095_bundle_0000_0003 \
  --log-dir /groups/johnson/johnsonlab/jeremy/recordings/logs/clipped_collection_flat_roi_cache_bundle_bsub \
  --run-id sleepyfish_cam2010095_bundle_0000_0003_20260707 \
  --run-label sleepyfish_cam2010095_bundle_0000_0003 \
  --queue gpu_l4 \
  --ncores 8 \
  --mem-gb 64 \
  --gpus 1 \
  --max-workers 4 \
  --walltime 4:00 \
  --progress-interval-s 30 \
  --overwrite
```

The bundle submitter also supports `--all-clips`, which resolves clip IDs from
the finalized collection at submission time. Use explicit clip IDs for first
validation runs so the output directory and expected child manifests are easy to
audit.

On Janelia LSF, the default `-gpu num=1` request has been observed to resolve to
`mode=exclusive_process:mps=no:j_exclusive=yes`. That mode allows only one child
process to create a CUDA decoder/context; additional PyNvVideoCodec children can
fail with `CUDA_ERROR_DEVICE_UNAVAILABLE`. The bundle submitter therefore
exposes `--gpu-resource` as a raw LSF override for experiments such as:

```bash
--gpu-resource 'num=1:mode=shared:j_exclusive=no'
```

Only use multi-worker bundles when the effective GPU resource is non-exclusive.
If the cluster cannot provide shared CUDA contexts on one L4, use one clip per
job or a lower-level single-process multi-decoder implementation instead.

Measured validation on 2026-07-07:

- Default `-gpu num=1` resolved to
  `mode=exclusive_process:mps=no:j_exclusive=yes`; one child completed and the
  other three failed in PyNvVideoCodec with `CUDA_ERROR_DEVICE_UNAVAILABLE`.
- Shared request `--gpu-resource 'num=1:mode=shared:j_exclusive=no'` allowed
  four child decoder processes to run concurrently on one L4.
- Four full sleepyfish clip caches (`clip_000000`..`clip_000003`) completed as
  job `152007051` with `215,357` total ROI rows and `56,454,545,408` payload
  bytes.
- Each child built at roughly `132-133 ROI/s` over `~404.5 s`; aggregate build
  throughput was roughly `532 ROI/s`.
- End-to-end job runtime was `460 s`, including publication of four payloads to
  NRS, for roughly `468 ROI/s` overall.
- Per-child payload publish took `~17-18 s` at roughly `746-779 MiB/s`.

### Whole Collection Scheduling

Use `submit_clipped_collection_flat_roi_cache_bundles_bsub.sh` when the goal is
to schedule every clip in a finalized collection. It discovers clip IDs from
`experiment_index/finalized_runs/<collection_id>/selected_runs`, groups them
with `--clips-per-job`, and submits one bundle job per group.

Recommended L4 starting point:

```bash
scripts/submit_clipped_collection_flat_roi_cache_bundles_bsub.sh \
  --zarr /groups/johnson/johnsonlab/jeremy/recordings/sleepyfish_2026_05_05_17_45_30_cam2010095/zarr/sleepyfish_2026_05_05_17_45_30_cam2010095_analysis.zarr \
  --collection-id sleepyfish_cam2010095_allclips_pynvvc_fixed_20260522_01 \
  --all-clips \
  --public-cache-dir-root /nrs/ahrens/palette_staging/clipped_collection_flat_roi_cache/sleepyfish_cam2010095_allclips \
  --log-dir /groups/johnson/johnsonlab/jeremy/recordings/logs/clipped_collection_flat_roi_cache_bundle_bsub \
  --run-id-prefix sleepyfish_cam2010095_allclips_20260707 \
  --run-label-prefix sleepyfish_cam2010095_allclips \
  --queue gpu_l4 \
  --ncores 8 \
  --mem-gb 64 \
  --gpus 1 \
  --gpu-resource 'num=1:mode=shared:j_exclusive=no' \
  --clips-per-job 4 \
  --max-workers 4 \
  --walltime 1:00 \
  --progress-interval-s 60
```

Operational notes:

- `--clips-per-job` controls how many clip caches a bundle job owns.
- `--max-workers` controls how many child builders run concurrently inside one
  bundle job.
- For L4 shared-mode decode, keep `--clips-per-job` and `--max-workers` aligned
  at `4` unless benchmarking suggests otherwise.
- Use `--start-bundle-index` and `--limit-bundles` for retries, canaries, or
  partial scheduling.
- Use `--dry-run` first; it prints every child bundle submission and output
  directory without submitting to LSF.
- Each bundle publishes into its own directory under `--public-cache-dir-root`.
  Downstream model scheduling should treat those child manifests as independent
  cache shards until a later collection/proxy-crop layer is implemented.

### Model Orchestration DAG

The full clipped keypoint workflow should be submitted as an explicit LSF DAG,
not as a single long-running monolithic job. Each step should write a status
artifact and the next step should depend on successful completion of the prior
required jobs.

Recommended dependency graph:

```text
cache_build[clip] or cache_already_exists
  -> proxy_create[clip]
  -> keypoint_shard[clip]
  -> merge_proxy_and_finalize_keypoints
  -> refine_keypoints
  -> optional_cleanup

keypoint_shard[clip]
  -> optional shard status audit
```

LSF dependency policy:

- Use `done(<jobid>)` dependencies for finalization and refinement.
- On Janelia LSF, CPU-only finalizer/refinement jobs should default to the
  `short` queue. The cluster does not provide a `normal` queue, and using that
  name causes `bsub` to reject the dependent job after shard jobs have already
  been submitted.
- Do not use `ended(<jobid>)` for required upstream work, because a failed
  shard must not trigger a partial collection finalizer.
- Per-clip keypoint shard jobs should depend only on that clip's cache/proxy
  readiness, not on all clips. This preserves parallelism and makes retries
  clip-local.
- The collection finalizer should run only after every expected shard job has
  completed successfully and should independently validate that every expected
  `keypoint_shard_runs/<run>` exists, is complete, and maps to the merged proxy
  crop run.
- Refinement should depend on the keypoint collection finalizer, not on the
  individual shard jobs.

Cleanup policy:

- Do not delete smoke or intermediate shard runs merely because jobs finish.
- Delete smoke runs only after the full collection `keypoints_runs/<run>` and
  `refined_keypoints_runs/<run>` exist and pass lineage validation.
- Production shard groups may remain as provenance/debug artifacts until a
  deliberate retention policy exists. Temporary node-local scratch must still be
  cleaned on job exit via shell `trap`, as current keypoint and cache wrappers
  already do.

Existing Palette precedent:

- `submit_subject_mask_batches_bsub.sh` submits inference and finalization as
  separate jobs with `bsub -w "done(<inference_job_id>)"`.
- `submit_crop_flat_roi_cache_batches_bsub.sh` builds a dependency expression
  for a registry/cache finalizer after per-recording jobs complete.

### Phase 2: proxy crop runs

- [x] Add a tool to create geometry-only proxy crop runs from a clipped
  collection row index.
- [x] Ensure proxy crop runs carry all row-lineage arrays needed by
  keypoints/masks.
- [x] Write per-shard manifest aliases that add `source.crop_run_name` for the
  proxy crop run while reusing the existing cache `.bin` payload.
- [x] Validate one proxy crop run plus manifest alias with
  `open_flat_roi_cache(..., expected_crop_run=<proxy_run>)`.
- [x] Smoke `CropImageSource.open(..., roi_cache_manifest=<alias>)` against one
  real clipped proxy.
- [x] Add a tool to merge per-clip proxy crop runs into one collection-level
  proxy crop run for downstream finalization/refinement.

### Phase 3: model runners

- [x] Add shard-output-parent support or an equivalent shard mode to keypoint
  runners.
- [x] Expose keypoint shard output through the LSF batch submitter via
  `--output-parent keypoint_shard_runs`.
- [x] Ensure keypoint LSF jobs clean staged flat-cache scratch on job exit when
  `--stage-roi-cache-to-scratch` is enabled.
- [ ] Add shard-output-parent support or an equivalent shard mode to subject-mask
  runners.
- [x] Smoke keypoints against one clip proxy + cache alias manifest.
- [ ] Smoke subject masks against the same proxy + cache alias manifest.
- [x] Verify output arrays contain parent-frame and clip-local lineage.
- [ ] Verify refined keypoints and refined subject masks preserve lineage.
- [x] Verify keypoint shard runs do not live under ordinary stage parents or, if they do,
  are excluded from resolver fallback.
- [x] Verify keypoint shard runs do not update ordinary stage `latest`,
  `latest_complete`, root `current_*_group_path` selectors, or canonical
  keypoint registry/status rows.
- [x] Add a v1 keypoint shard finalizer that publishes a canonical
  `keypoints_runs/<run>` only after same-source-crop shards validate.
- [x] Add a whole-clipped-collection keypoint finalizer path that safely rebases
  shards spanning multiple proxy crop runs onto a merged proxy crop run.
- [x] Add a clipped keypoint orchestration dry-run planner that resolves cache
  manifests, proxy runs, shard runs, finalizer/refinement runs, and LSF
  dependency templates.
- [x] Add clipped keypoint orchestration apply mode that creates proxy runs,
  submits per-clip shard jobs, parses LSF job ids, and submits collection
  finalizer/refinement jobs with explicit `done(<jobid>)` dependencies.
- [x] Smoke clipped keypoint orchestration apply mode on two clips.

Two-clip apply smoke on 2026-07-07:

- Source collection:
  `sleepyfish_cam2010095_allclips_pynvvc_fixed_20260522_01`
- Clips: `clip_000004`, `clip_000005`
- Shard jobs: `152019149`, `152019150` on `gpu_l4`
- Finalizer/refine jobs: `152019156`, `152019157` on `short`
- Merged proxy crop run:
  `crop_proxy_sleepyfish_kp_apply_smoke_20260707_01_collection`
- Canonical keypoint run:
  `keypoints_sleepyfish_kp_apply_smoke_20260707_01`
- Refined keypoint run:
  `refined_keypoints_sleepyfish_kp_apply_smoke_20260707_01`
- Row count: `103,637`
- Keypoint success: `103,582/103,637` (`99.95%`)
- Refined usable keypoints: `103,569/103,637`
- Finalizer duration: `6.94 s`
- Refine duration: `36.24 s`
- Validation: crop, keypoint, and refined-keypoint row counts matched, and
  `frame_indices`, `source_frame_indices`, `source_clip_indices`,
  `source_clip_local_frame_indices`, `source_crop_row_ids`,
  `source_refined_row_ids`, `source_detect_row_index`, and
  `detection_indices` matched across crop -> keypoints -> refined keypoints.

The first apply attempt exposed two operational bugs: LSF can report `bsub` job
IDs on stderr, and `normal` is not a valid Janelia CPU queue. The submitter now
parses job IDs from stdout or stderr, persists partial `submission.json`
snapshots during submission, and defaults CPU finalizer/refinement jobs to
`short`.

Full-collection apply smoke on 2026-07-07:

- Source collection:
  `sleepyfish_cam2010095_allclips_pynvvc_fixed_20260522_01`
- Clips: `clip_000000` through `clip_000021`
- Run label: `sleepyfish_kp_allclips_20260708_01`
- Shard jobs: `152019167` through `152019188` on `gpu_l4`
- Finalizer/refine jobs: `152019189`, `152019190` on `short`
- First-bundle cache note: clips `000000` through `000003` were available from
  the earlier shared-mode cache smoke under
  `/nrs/ahrens/palette_staging/clipped_collection_flat_roi_cache_smoke/`;
  symlink aliases were added under the all-clips cache root
  `sleepyfish_cam2010095_allclips_b0000` so the production planner can resolve
  all 22 manifests from one root.
- Merged proxy crop run:
  `crop_proxy_sleepyfish_kp_allclips_20260708_01_collection`
- Canonical keypoint run:
  `keypoints_sleepyfish_kp_allclips_20260708_01`
- Refined keypoint run:
  `refined_keypoints_sleepyfish_kp_allclips_20260708_01`
- Row count: `1,169,010`
- Keypoint success: `1,168,869/1,169,010` (`99.99%`)
- Refined usable keypoints: `1,168,798/1,169,010`
- Geometry issues: `71`
- Finalizer LSF runtime: `108 s`; stage finalization duration was reported in
  the keypoint finalizer output.
- Refine stage duration: `385.55 s`; LSF runtime was `1,050 s`, including
  startup, Zarr write/flush, and LSF accounting overhead.
- Validation: crop, keypoint, and refined-keypoint row counts matched, and
  `frame_indices`, `source_frame_indices`, `source_clip_indices`,
  `source_clip_local_frame_indices`, `source_crop_row_ids`,
  `source_refined_row_ids`, `source_detect_row_index`, and
  `detection_indices` matched across crop -> keypoints -> refined keypoints.

### Phase 4: registry

- [ ] Add stage-row fields for collection/cache source targeting.
- [ ] Update keypoint and subject-mask extractors.
- [ ] Backfill clipped dataset rows and Zarr root attrs.
- [ ] Add registry queries for missing clipped cache/model stages.

### Phase 5: cluster operation

- [ ] Submit one clip-shard smoke.
- [ ] Submit all clip shards.
- [ ] Keep caches on NRS or a clearly temporary PRFS cache root.
- [ ] Record cleanup policy and cache TTL in manifests or registry artifacts.

## Current Recommendation

Keypoints are now safe to run across clipped flat-cache collections using the
explicit orchestration DAG: per-clip proxy crop runs, non-publishing
`keypoint_shard_runs`, collection-level proxy merge/finalization, and a normal
`refined_keypoints_runs/<run>` output. Use the dry-run planner first, then
submit with `--apply` once all clip cache manifests resolve.

Do not submit full subject-mask inference across clipped caches yet. The
subject-mask runner still needs equivalent shard-output semantics before it can
consume per-clip proxy crop/cache inputs without publishing unsafe ordinary
whole-recording selectors.

## Read-Only Audit Smoke

Added:

```bash
scripts/py -m fisheye.utils.audit_clipped_collection_cache_readiness \
  /groups/johnson/johnsonlab/jeremy/recordings/sleepyfish_2026_05_05_17_45_30_cam2010095/zarr/sleepyfish_2026_05_05_17_45_30_cam2010095_analysis.zarr \
  --registry /groups/johnson/johnsonlab/jeremy/registries/palette_registry.sqlite \
  --sample-video-paths 3
```

Result on 2026-07-07:

- `status = warning`
- `blocker_count = 0`
- `warning_count = 7`
- collection resolved:
  `sleepyfish_cam2010095_allclips_pynvvc_fixed_20260522_01`
- selected run count: `22`
- clip count: `22`
- all sampled `selected_runs[*].source.video_path` entries existed on
  `/groups`
- recording frame index row count: `1,188,000`

Warnings:

- Zarr root missing `recording_id`.
- Zarr root missing `source_layout = rolling_clips`.
- Zarr root missing `source_recording_frame_index_path`.
- sampled `recording_frame_index.parquet::video_path` values point to `/nvme1`.
- sampled `recording_frame_index.parquet::video_path` values do not exist.
- registry dataset row has empty `source_layout`.
- registry dataset row has empty `source_recording_frame_index_path`.

Interpretation:

The collection itself is decode-ready for the inspected camera because its
selected-run source paths point to `/groups`. The archive and registry are not
yet clipped-workflow-ready because root attrs, registry targeting fields, and
frame-index path columns still need relocation/backfill cleanup.

## Proxy Crop-Run Smoke

Added:

```bash
scripts/py -m fisheye.utils.create_clipped_collection_proxy_crop_run \
  /groups/johnson/johnsonlab/jeremy/recordings/sleepyfish_2026_05_05_17_45_30_cam2010095/zarr/sleepyfish_2026_05_05_17_45_30_cam2010095_analysis.zarr \
  /nrs/ahrens/palette_staging/clipped_collection_flat_roi_cache/sleepyfish_cam2010095_allclips/sleepyfish_cam2010095_allclips_b0001/sleepyfish_cam2010095_allclips_b0001__clip_000004.flat_roi_cache.json \
  --proxy-run crop_proxy_sleepyfish_cam2010095_clip_000004_20260707 \
  --json
```

Result:

```text
proxy_crop_run_path = crop_runs/crop_proxy_sleepyfish_cam2010095_clip_000004_20260707
row_count = 53,293
cache_shape = [53293, 512, 512]
source_clip_id = clip_000004
alias_manifest_path = /nrs/ahrens/palette_staging/clipped_collection_flat_roi_cache/sleepyfish_cam2010095_allclips/sleepyfish_cam2010095_allclips_b0001/sleepyfish_cam2010095_allclips_b0001__clip_000004.flat_roi_cache__crop_proxy_sleepyfish_cam2010095_clip_000004_20260707.alias.json
```

Read smoke:

```text
CropImageSource.open(..., crop_run=<proxy>, roi_cache_manifest=<alias>)
total_rois = 53,293
roi_shape = (512, 512)
read_shape = (2, 512, 512)
read_dtype = uint8
roi_cache_used = true
read_mode = flat_bin_roi_cache
```

## Two-Clip Keypoint Finalizer Smoke

Date: 2026-07-07

Validated a true multi-proxy path against Sleepyfish cam2010095 clips
`000004` and `000005`.

Inputs:

```text
crop_proxy_sleepyfish_cam2010095_clip_000004_20260707
crop_proxy_sleepyfish_cam2010095_clip_000005_20260707
keypoint_shard_2026-07-07_17-53-56
keypoint_shard_2026-07-07_22-00-16
```

The clip `000005` shard ran on LSF `gpu_l4` job `152016505` with
node-local scratch staging:

```text
total_rois = 50,344
successful = 50,328
failed = 16
duration = 248.4 s
rate = 202.6 poses/s
scratch_cleanup = /scratch/delahantyj/152016505/palette_roi_cache_stage
```

Merged proxy crop run:

```text
crop_runs/crop_proxy_sleepyfish_cam2010095_clips_000004_000005_collection_proxy_smoke_20260707
source_proxy_crop_run_count = 2
row_count = 103,637
source_row_counts = [53,293, 50,344]
```

Finalized keypoint run:

```text
keypoints_runs/keypoints_sleepyfish_cam2010095_clips_000004_000005_target_proxy_smoke_20260707
source_crop_run = crop_proxy_sleepyfish_cam2010095_clips_000004_000005_collection_proxy_smoke_20260707
source_crop_rebased_from_shards = true
total_rois = 103,637
successful_detections = 103,582
failed_detections = 55
success_rate_percent = 99.95
finalization_duration_seconds = 5.34
```

Refined keypoint run:

```text
refined_keypoints_runs/refined_keypoints_sleepyfish_cam2010095_clips_000004_000005_target_proxy_smoke_20260707
source_keypoints_run = keypoints_sleepyfish_cam2010095_clips_000004_000005_target_proxy_smoke_20260707
source_crop_run = crop_proxy_sleepyfish_cam2010095_clips_000004_000005_collection_proxy_smoke_20260707
refined_success = 103,582
source_failures = 55
geometry_issues = 13
usable_keypoints = 103,569
pass_rate_percent = 99.95
duration_seconds = 33.05
```

Readback validation:

```text
clip_indices_unique = [4, 5]
source_crop_row_ids = 0..103636
frame_indices matched merged proxy crop = true
source_frame_indices matched merged proxy crop = true
source_clip_indices matched merged proxy crop = true
source_clip_local_frame_indices matched merged proxy crop = true
source_refined_row_ids matched merged proxy crop = true
source_detect_row_index matched merged proxy crop = true
detection_indices matched merged proxy crop = true
```

This proves the current merged-proxy design preserves the ordinary downstream
single-`source_crop_run` contract while retaining per-clip lineage arrays.
