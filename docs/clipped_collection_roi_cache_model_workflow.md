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

### Phase 2: proxy crop runs

- [ ] Add a tool to create geometry-only proxy crop runs from a clipped
  collection row index.
- [ ] Ensure proxy crop runs carry all row-lineage arrays needed by
  keypoints/masks.
- [ ] Make cache manifests include `source.crop_run_name` when built for a proxy
  crop run.

### Phase 3: model runners

- [ ] Smoke keypoints against one clip proxy + cache manifest.
- [ ] Smoke subject masks against the same proxy + cache manifest.
- [ ] Verify output arrays contain parent-frame and clip-local lineage.
- [ ] Verify refined keypoints and refined subject masks preserve lineage.

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

Do not build full persistent collection caches yet. Start with Phase 0 and
Phase 1, then run a one-clip smoke. Persisting large caches before proxy crop
runs and registry lineage are in place would create outputs that are usable only
by external manifest knowledge, not self-describing Palette stage contracts.

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
