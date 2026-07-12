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
  bbox_norm_coords
```

Suggested attrs:

```text
crop_storage_mode = geometry_only
source_kind = finalized_clipped_refined_detect_collection
detection_source_type = finalized_clipped_refined_detect_collection_proxy
source_detect_run = finalized_clipped_refined_detect_collection_proxy:<collection id>
source_detect_run_semantics = synthetic_collection_rowset_label_not_detect_runs_child
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
bbox_norm_coords_semantics = bbox_xywh_normalized_to_full_frame
bbox_norm_coords_source = clipped_collection_row_index.bbox_norm_cxcywh
```

`bbox_norm_coords` uses Palette's canonical full-frame-normalized
`[cx, cy, w, h]` convention. The proxy source label is intentionally synthetic:
it identifies the finalized clipped collection rowset, not a child under
`detect_runs/`. This lets arena assignment and tracking key lineage against
`source_detect_run` plus the exact `source_rowset_path` without pretending the
collection is a normal raw detect run.

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

Modern per-clip proxies already store `bbox_norm_coords`. If older per-clip
proxies predate that array, the merge utility can repair the merged output from
each proxy's `source_roi_cache_row_index_path`, provided the row-index parquet
contains `bbox_norm_cx`, `bbox_norm_cy`, `bbox_norm_w`, and `bbox_norm_h`.
The merged run records `legacy_bbox_norm_coords_repair_count` so operators can
see whether the output was built entirely from modern proxy arrays or repaired
legacy inputs.

Existing proxy crop runs can also be repaired in place before rerunning
downstream tracking/kinematics:

```bash
scripts/py -m fisheye.utils.repair_clipped_proxy_crop_contract \
  /path/to/recording_analysis.zarr \
  --crop-run crop_proxy_collection_... \
  --apply
```

Without `--apply`, the command is a dry-run and reports the arrays/attrs it
would write. The repair only targets clipped proxy crop runs and leaves ordinary
crop runs untouched.

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

### Subject-mask clipped collection readiness

Read-only audit on 2026-07-08 found that the ordinary subject-mask pipeline is
production-capable for standard recording-level crop runs, but not yet safe for
clipped collection shards.

What works today:

- `infer_unet_subject_masks.py` can read normal `crop_runs/<run>` inputs and
  optional flat ROI cache manifests.
- Raw U-Net output is probability-first: `subject_mask_runs/<run>/mask_probs_roi`
  is canonical, and dense binary `masks_roi` is optional compatibility output.
- Raw probabilities can opt into `2,048`-row Zarr indexed shards with
  `--mask-probs-shard-rois 2048`. Inference still writes ordinary inner chunks
  to a private working array, then packs and exact-validates complete shards
  before the run can complete. Dense `masks_roi` remains ordinarily chunked.
- The raw writer copies row lineage from the crop run, including
  `frame_indices`, `source_frame_indices`, `source_clip_indices`,
  `source_clip_local_frame_indices`, `source_crop_row_ids`,
  `source_refined_row_ids`, `source_detect_row_index`, and `instance_key` when
  present in the source crop run.
- The raw writer records ROI cache provenance, crop pixel/source snapshot attrs,
  model resolution attrs, assignment-keypoint attrs, timing profile attrs, and
  summary statistics.
- `submit_subject_mask_batches_bsub.sh` already supports the normal
  one-recording-per-job cluster workflow with ROI cache staging, local output
  staging, split GPU inference and CPU finalization jobs, progress artifacts,
  and scratch cleanup through shell `trap`.
- `finalize_subject_masks.py` requires modern `source_crop_row_ids`, preserves
  source crop lineage, supports `process_shards`, emits progress JSONL, writes
  eye geometry and component contours, and can publish dense, bitpacked, RLE, or
  combined refined mask stores.

What this slice fixed:

- `infer_unet_subject_masks.py` now accepts `--output-parent` with
  `subject_mask_runs` as the default and `subject_mask_shard_runs` as the
  shard-safe parent.
- Shard outputs are completed under `subject_mask_shard_runs`, stamp
  `is_collection_shard=true` and `stage_selector_eligible=false`, and do not
  mutate ordinary `subject_mask_runs.latest` / `latest_complete` selectors.
- Shard outputs suppress canonical registry/status refresh and record
  collection/clip/cache alias metadata when supplied.
- `run_subject_mask_batch_pipeline.py` and
  `submit_subject_mask_batches_bsub.sh` now expose shard output mode for raw
  inference. Selecting `subject_mask_shard_runs` is inference-only and disables
  the dependent refined-finalization submission in the bsub wrapper.
- The batch runner, clipped-collection planner, and bsub wrapper also propagate
  the independent `--mask-probs-shard-rois` physical-storage option. This is
  orthogonal to the logical `subject_mask_shard_runs` parent: it controls how
  each raw run's canonical probability array is stored, not which rows belong
  to that run.
- Focused tests pin that a subject-mask shard does not change canonical
  `subject_mask_runs` selectors or a root current-pointer attr.
- `finalize_subject_masks.py` now supports a low-level collection-finalizer
  mode. It accepts repeated `--subject-shard-run` values or a
  `--subject-shard-runs-file`, validates compatible completed
  `subject_mask_shard_runs`, optionally rebases shard-local crop rows into a
  merged proxy crop run via `--target-crop-run`, and writes one canonical
  `refined_subject_masks_runs/<collection_run>` without materializing a merged
  raw `subject_mask_runs` probability surface.

What is still not safe yet:

- The collection finalizer has now been smoked against two real clipped proxy
  crop runs plus two real subject-mask shard outputs for `subject_body` and
  `swim_bladder`. Eye-component collection finalization is still pending
  because it requires a refined-keypoint run aligned to the same collection
  proxy crop surface.
- The existing subject-mask bsub wrapper can submit a shard-mode inference job,
  but it is still one target list / one shared shard metadata set. A full
  collection-aware wrapper that enumerates clips and passes per-clip metadata is
  still pending.
- A merged whole-collection raw `subject_mask_runs` would duplicate dense
  probability surfaces. For Sleepyfish-scale collections this can be much larger
  than keypoint merging and should not be the default design.

Preferred direction:

1. Add a shard-safe raw subject-mask parent, for example
   `subject_mask_shard_runs/<run>`.
2. Make shard mode explicit in the inference CLI and cluster wrapper. Shard
   mode must not update `subject_mask_runs.latest`, `latest_complete`, root
   `current_*` attrs, or canonical registry/status rows.
3. Keep per-clip raw probability outputs as staging artifacts. They may remain
   useful for debugging and recomputation, but ordinary consumers should not
   resolve them as recording-level mask runs.
4. Add a collection subject-mask finalizer that validates every expected shard,
   verifies row lineage, and writes one canonical
   `refined_subject_masks_runs/<collection_run>`.
5. Prefer compact refined storage for collection outputs:
   `bitpacked_v1` when edit/review is expected, `dense_and_bitpacked` for
   validation or compatibility canaries, and `rle_v1` only for final/read-mostly
   products whose readers support compact stores.
6. Avoid a monolithic raw `subject_mask_runs/<collection_run>` unless a concrete
   downstream consumer requires a single raw probability surface.

Implementation checklist for subject-mask clipped collections:

- [x] Add `--output-parent` or `--shard-mode` to
  `infer_unet_subject_masks.py`.
- [x] In shard mode, write to `subject_mask_shard_runs` and stamp
  `is_collection_shard=true`, `stage_selector_eligible=false`,
  `source_collection_id`, `source_clip_id`, `source_clip_index`,
  `source_crop_run`, `source_roi_cache_alias_manifest`, and
  `source_roi_cache_row_index_path`.
- [x] Ensure shard mode still calls `mark_run_started` /
  `mark_run_complete` for the shard run itself, but never mutates ordinary
  `subject_mask_runs` parent selectors.
- [x] Suppress canonical subject-mask registry/status refresh for shard runs.
- [x] Expose shard-mode raw inference through
  `run_subject_mask_batch_pipeline.py` and
  `submit_subject_mask_batches_bsub.sh`.
- [x] Add opt-in post-inference indexed sharding for immutable raw
  `mask_probs_roi`, with complete-shard writes and exact decoded-byte
  validation before completion.
- [x] Add focused tests proving a subject-mask shard does not change
  `subject_mask_runs.latest`, `subject_mask_runs.latest_complete`, or root
  current-pointer attrs.
- [x] Add a single-clip smoke using the existing clipped proxy crop run plus
  cache alias path that keypoints already validated.
- [x] Validate raw shard output row lineage against the proxy crop run.
- [x] Add a collection finalizer that reads explicit
  `subject_mask_shard_runs/<run>` names or a shard-runs JSON file.
- [x] Finalizer validation must require every expected shard to be complete,
  model/schema-compatible, and row-lineage compatible with the target merged
  proxy crop run.
- [x] Decide whether the collection finalizer writes directly to
  `refined_subject_masks_runs/<collection_run>` or creates an intermediate
  collection index. Default should be direct refined output unless a consumer
  proves it needs merged raw probabilities.
- [ ] Preserve assignment-keypoint lineage by pointing finalization at the
  canonical clipped collection refined-keypoint run.
- [x] Write refined collection masks with explicit `source_crop_run` pointing to
  the merged proxy crop run and required `source_crop_row_ids`.
- [x] Validate refined mask lineage against crop and raw mask shards.
- [ ] Validate refined mask lineage against canonical clipped collection
  refined keypoints in an eye-component collection smoke.
- [x] Smoke compact refined storage on two clips with `dense_and_bitpacked`.
- [ ] Smoke a production candidate with `bitpacked_v1`.
- [ ] Add a collection-aware subject-mask orchestration wrapper that enumerates
  clipped work units, passes per-clip shard provenance, and depends on the
  collection finalizer after shard inference completes.

Low-level collection finalizer command shape:

```bash
scripts/py -m fisheye.refinement.finalize_subject_masks \
  /path/to/collection_analysis.zarr \
  --subject-shard-runs-file /path/to/subject_mask_shards.json \
  --target-crop-run crop_proxy_<collection_or_workflow_id> \
  --run-name refined_subject_masks_<collection_or_workflow_id> \
  --components subject_body eyes_union swim_bladder \
  --assignment-keypoint-group refined_keypoints_runs \
  --assignment-keypoints-run <collection_refined_keypoint_run> \
  --mask-storage dense_and_bitpacked \
  --execution-backend process_shards \
  --num-workers 8 \
  --write-eye-geometry \
  --write-component-contours \
  --overwrite \
  --json
```

The shard-runs JSON may be either a list of run names or an object with
`subject_mask_shard_runs`, `shard_runs`, or `runs`. Mixed per-clip crop runs
must pass `--target-crop-run`; the finalizer maps shard-local
`source_crop_row_ids` onto that merged proxy crop run using
`source_clip_indices`, `source_clip_local_frame_indices`,
`source_refined_row_ids`, `source_detect_row_index`, `frame_indices`, and
`roi_coordinates_full`.

Subject-mask shard smoke on 2026-07-08:

- Source collection:
  `sleepyfish_cam2010095_allclips_pynvvc_fixed_20260522_01`
- Clip: `clip_000004`
- A 256-row derived smoke cache was created from the already-built full
  `clip_000004` flat cache to avoid re-decoding video with a CPU-only local
  Torch install.
- Smoke cache:
  `/nrs/ahrens/palette_staging/clipped_collection_flat_roi_cache_smoke/subject_mask_shard_smoke_20260708_clip_000004_limit256/subject_mask_shard_smoke_clip_000004_limit256.flat_roi_cache.json`
- Proxy crop run:
  `crop_proxy_subject_mask_shard_smoke_20260708_clip_000004_limit256`
- Raw subject-mask shard run:
  `subject_mask_shard_runs/subject_masks_unet_shard_smoke_20260708_clip_000004_limit256`
- Model:
  `subject_masks_union_all_components_v001`
- Execution: CPU smoke, `batch_size=8`, `mask_probs_dtype=uint8`,
  `masks_roi_materialized=false`
- Duration: `54.5 s` for `256` ROIs; timing profile was dominated by CPU
  `model_forward` (`52.1 s`).
- Output shape: `mask_probs_roi = (256, 3, 512, 512)`.
- Summary: `rows_with_nonempty_masks = 256/256`.
- Validation: `frame_indices`, `source_frame_indices`, `source_clip_indices`,
  `source_clip_local_frame_indices`, `source_crop_row_ids`,
  `source_refined_row_ids`, `source_detect_row_index`, and
  `detection_indices` all matched the proxy crop run.
- Selector safety: `subject_mask_shard_runs.latest` and
  `latest_complete` point at the smoke shard; canonical
  `subject_mask_runs.latest` and `latest_complete` remained unset.
- Shard attrs stamped `is_collection_shard=true`,
  `stage_selector_eligible=false`, and
  `registry_status_deferred_reason=collection_shard_not_canonical_stage_output`.
- Model-resolution provenance is present under the current attr contract:
  `model_resolution_selected_model_path`, `model_resolution_registry_path`,
  `model_resolution_selected_run_id`,
  `model_resolution_selected_component_coverage_key`,
  `model_resolution_selected_metric_name`,
  `model_resolution_selected_metric_value`, and
  `model_resolution_candidates_json`. The legacy-looking names
  `checkpoint_path` and `model_registry_resolution` are not the contract.

Subject-mask collection-finalizer smoke on 2026-07-08:

- Source collection:
  `sleepyfish_cam2010095_allclips_pynvvc_fixed_20260522_01`
- Clips: `clip_000004` and `clip_000005`
- Per-clip proxy crop runs:
  - `crop_proxy_subject_mask_shard_smoke_20260708_clip_000004_limit256`
  - `crop_proxy_subject_mask_shard_smoke_20260708_clip_000005_limit256`
- Per-clip raw subject-mask shard runs:
  - `subject_mask_shard_runs/subject_masks_unet_shard_smoke_20260708_clip_000004_limit256`
  - `subject_mask_shard_runs/subject_masks_unet_shard_smoke_20260708_clip_000005_limit256`
- The `clip_000005` shard used the same model and settings as the `clip_000004`
  shard: `subject_masks_union_all_components_v001`, CPU, `batch_size=8`,
  `mask_probs_dtype=uint8`, `mask_probs_chunk_rois=16`, and
  `masks_roi_materialized=false`.
- The `clip_000005` raw shard processed `256` ROIs in `52.8 s`; timing was
  dominated by CPU `model_forward` (`52.0 s`).
- Merged proxy crop run:
  `crop_proxy_subject_mask_shard_smoke_20260708_clips_000004_000005_limit256_collection`
  with `512` rows and source row counts `[256, 256]`.
- Refined collection run:
  `refined_subject_masks_runs/refined_subject_masks_shard_collection_smoke_20260708_clips_000004_000005_limit256_body_swim`
- Finalized components: `subject_body` and `swim_bladder`. Eye-left/right
  assignment was intentionally not part of this smoke because no refined
  keypoint run was aligned to the limited 512-row merged proxy crop surface.
- Finalizer execution: `process_shards`, `num_workers=2`, `chunk_size=256`,
  `postcompute_backend=process_shards`, `mask_storage=dense_and_bitpacked`,
  `write_component_contours=true`.
- Duration: `20.0 s` for `512` ROIs (`25.6 rows/s` end-to-end). The
  postcompute contour phase processed `512` ROIs at `254 rows/s`.
- Persisted refined dense shape:
  `masks_roi = (512, 2, 512, 512)`.
- Persisted compact editable shape:
  `mask_bitpacked/masks_packed = (512, 2, 512, 64)`.
- Bitpacked validation passed in full round-trip mode for both row chunks and
  both channels.
- Component contours were written for both components:
  `subject_body` (`239,426` points) and `swim_bladder` (`26,507` points).
- Refined run attrs stamp
  `collection_finalizer_schema=palette_subject_mask_shard_collection_finalizer_v1`,
  `finalized_from_subject_mask_shards=true`,
  `source_crop_rebased_from_shards=true`, and
  `source_crop_rebase_target_run=<merged proxy crop run>`.
- Validation confirmed that the refined run's `frame_indices`,
  `source_frame_indices`, `source_clip_indices`,
  `source_clip_local_frame_indices`, `source_refined_row_ids`,
  `source_detect_row_index`, and `source_crop_row_ids` exactly match the merged
  target proxy crop run. Clip row counts were `256` rows from clip index `4`
  and `256` rows from clip index `5`.

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
- [x] Add shard-output-parent support or an equivalent shard mode to subject-mask
  runners.
- [x] Smoke keypoints against one clip proxy + cache alias manifest.
- [x] Smoke subject masks against the same proxy + cache alias manifest.
- [x] Verify output arrays contain parent-frame and clip-local lineage.
- [x] Verify refined keypoints preserve lineage.
- [x] Verify refined subject masks preserve crop/shard lineage for a two-clip
  `subject_body`/`swim_bladder` collection smoke.
- [ ] Verify refined subject masks preserve assignment-keypoint lineage in an
  eye-component collection smoke.
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
- [x] Add a clipped subject-mask orchestration dry-run planner that resolves
  cache manifests, proxy runs, raw subject-mask shard runs, finalizer run names,
  and LSF dependency templates.
- [x] Add clipped subject-mask orchestration apply mode that creates proxy runs,
  submits per-clip raw shard jobs, parses LSF job ids, and submits the
  collection finalizer with explicit `done(<jobid>)` dependencies.
- [ ] Add node-local flat-cache staging and cleanup to clipped subject-mask
  shard jobs.
- [ ] Smoke clipped subject-mask orchestration apply mode on two clips.

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

Two-clip subject-mask finalizer smoke on 2026-07-08:

- Source collection:
  `sleepyfish_cam2010095_allclips_pynvvc_fixed_20260522_01`
- Clips: `clip_000004`, `clip_000005`
- Per-clip proxy crop runs:
  - `crop_proxy_subject_mask_shard_smoke_20260708_clip_000004_limit256`
  - `crop_proxy_subject_mask_shard_smoke_20260708_clip_000005_limit256`
- Per-clip raw subject-mask shard runs:
  - `subject_mask_shard_runs/subject_masks_unet_shard_smoke_20260708_clip_000004_limit256`
  - `subject_mask_shard_runs/subject_masks_unet_shard_smoke_20260708_clip_000005_limit256`
- Merged proxy crop run:
  `crop_proxy_subject_mask_shard_smoke_20260708_clips_000004_000005_limit256_collection`
- Refined subject-mask run:
  `refined_subject_masks_shard_collection_smoke_20260708_clips_000004_000005_limit256_body_swim`
- Row count: `512` (`256` rows from clip index `4`, `256` from clip index `5`)
- Components finalized: `subject_body`, `swim_bladder`
- Storage: `dense_and_bitpacked`
- Dense shape: `masks_roi = (512, 2, 512, 512)`
- Bitpacked shape: `mask_bitpacked/masks_packed = (512, 2, 512, 64)`
- Finalizer duration: `20.0 s`; end-to-end finalization rate
  `25.6 rows/s`; postcompute contour phase `254 rows/s`
- Component contours written:
  `subject_body` (`239,426` points) and `swim_bladder` (`26,507` points)
- Validation: the refined run's `frame_indices`, `source_frame_indices`,
  `source_clip_indices`, `source_clip_local_frame_indices`,
  `source_refined_row_ids`, `source_detect_row_index`, and
  `source_crop_row_ids` exactly matched the merged target proxy crop run.
- Model-resolution provenance for raw shards is under the current
  `model_resolution_*` attr contract, including
  `model_resolution_selected_model_path`, `model_resolution_registry_path`,
  `model_resolution_selected_run_id`,
  `model_resolution_selected_component_coverage_key`,
  `model_resolution_selected_metric_name`,
  `model_resolution_selected_metric_value`, and
  `model_resolution_candidates_json`.

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

Subject masks now have equivalent shard-output semantics and a collection-aware
planner. Use the dry-run planner first, and run a two-clip apply smoke before a
full collection apply. If finalization includes `eyes_union`, pass the
collection refined-keypoint run explicitly with `--assignment-keypoints-run`.
The v1 subject-mask planner reads published cache alias manifests directly from
shared storage; node-local flat-cache staging is still a pending performance
hardening step.

Example subject-mask dry run:

```bash
scripts/submit_clipped_collection_subject_masks_bsub.sh \
  --zarr /path/to/collection_analysis.zarr \
  --collection-id <finalized_collection_id> \
  --cache-dir-root /nrs/ahrens/palette_staging/clipped_collection_flat_roi_cache/<cache_root> \
  --all-clips \
  --run-label <workflow_label> \
  --assignment-keypoints-run <collection_refined_keypoint_run> \
  --components subject_body eyes_union swim_bladder \
  --queue gpu_l4 \
  --ncores 8 \
  --mem-gb 32 \
  --finalizer-queue short \
  --finalizer-ncores 8 \
  --finalizer-mem-gb 32 \
  --mask-storage dense_and_bitpacked \
  --dry-run
```

For a body/swim-only smoke that does not require keypoint-based eye assignment,
use:

```bash
--components subject_body swim_bladder
```

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
