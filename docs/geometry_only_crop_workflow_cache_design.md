# Geometry-Only Crop Workflow Cache Design
<!-- contract-meta
status: design
last_verified: 2026-06-18
purpose: Define the target policy for geometry-only analysis crop runs, shared workflow ROI caches, and optional flat binary crop-cache transport.
-->

## Purpose

Palette is moving toward lean analysis archives where crop runs store geometry,
lineage, and provenance, while ROI pixels are materialized only when they are
needed for training artifacts, review caches, or high-throughput inference.

This document defines the intended workflow shape before implementation.

It complements:

- `docs/archive/crop_live_view_vs_materialized_stream_design.md`
- `docs/crop_storage_mode_migration_todo.md`
- `docs/lsf_submission_framework_design.md`
- `docs/cluster_run_group_artifact_workflow.md`

## Core Policy

Target policy:

- Analysis crop runs should default to `crop_storage_mode=geometry_only` after
  the major readers are migrated.
- Training Zarrs should continue to default to materialized crops and should
  reject geometry-only crop writes.
- Materialized analysis crops remain an explicit operator/debug/performance
  override, not the default long-term storage policy.
- Temporary ROI caches are runtime or workflow artifacts, not canonical
  analysis data.
- Source videos must remain addressable and fingerprinted when analysis crops
  are geometry-only.

The canonical analysis crop run should be small and durable:

```text
analysis.zarr/
  crop_runs/
    crop_<run>/
      roi_coordinates_full
      roi_coordinates_ds
      bbox_norm_coords
      frame_indices
      frame_counts
      detection_indices
      detection_source
      attrs:
        crop_storage_mode = "geometry_only"
        bbox_norm_coords_semantics = "bbox_xywh_normalized_to_full_frame"
        roi_size
        crop_signature
        source_detect_run
        detection_source_path
        source_video_path
        source_video_fingerprint
```

`bbox_norm_coords` is always Palette's canonical full-frame-normalized
`[cx, cy, w, h]` geometry, regardless of whether ROI pixels come from full-frame
decode, acquisition crop video, or a clipped-collection flat cache. Local
crop-frame or ROI-frame boxes need explicit local names; do not overload
`bbox_norm_coords`.

The canonical training crop run should stay self-contained:

```text
training.zarr/
  crop_runs/
    crop_<run>/
      roi_images
      roi_coordinates_full
      frame_indices
      detection_indices
      attrs:
        crop_storage_mode = "materialized"
```

## Latest Pointer Migration

During migration, keep pointer semantics conservative:

```text
crop_runs.attrs["latest"]              -> latest materialized-compatible run
crop_runs.attrs["latest_materialized"] -> latest materialized run
crop_runs.attrs["latest_any"]          -> latest valid crop run of any storage mode
```

Once major readers are migrated to mixed-mode reads, `latest` can become
latest-any:

```text
crop_runs.attrs["latest"]              -> same as latest_any
crop_runs.attrs["latest_materialized"] -> compatibility pointer for old readers
crop_runs.attrs["latest_any"]          -> latest valid crop run of any storage mode
```

Until that cutover, cluster and mixed-mode analysis workflows should pass
explicit `--crop-run <run>` or resolve `latest_any`. Materialized-only tools
should use `latest_materialized` and fail clearly if no materialized run
exists.

## Workflow Cache Concept

A geometry-only crop run intentionally avoids storing ROI pixels permanently in
the analysis archive. But downstream pose and segmentation stages still need ROI
pixels. The cache policy is therefore:

1. Build ROI pixels once per workflow/crop run.
2. Store them in a shared workflow cache artifact.
3. Reuse that cache across pose, eye-mask, subject-mask, and related jobs.
4. Keep the cache outside canonical analysis archives.
5. Delete or expire it independently of canonical data.

Logical workflow:

```text
detect/refine
  -> crop geometry job
  -> crop geometry import/validation
  -> ROI cache build job
  -> keypoints job
  -> eye/subject/swim-bladder segmentation jobs
  -> optional review job
```

The ROI cache is keyed by immutable inputs:

- archive identity
- crop run name
- crop signature
- ROI shape
- source video fingerprint
- crop image conversion policy
- cache backend/schema version

If any of those values change, the cache must not be reused.

## Storage Placement

There are three storage tiers with different roles.

### Node-Local Job Scratch

Example:

```text
/scratch/$USER/$LSB_JOBID/palette_cache/
```

Use this for hot writes and hot reads during a single job. It is the preferred
place to build ROI pixels because it avoids high-concurrency writes to shared
storage.

Limitations:

- It is usually node-local.
- It may disappear after the job.
- A downstream job on another node cannot assume it is still available.

### PRFS Workflow Scratch

Example shape:

```text
<prfs_scratch_root>/palette_workflows/<workflow_id>/
  roi_cache/
  jobs/
  manifests/
```

At Janelia, a practical PRFS scratch root may live under `misc/public` or an
equivalent site-managed shared scratch location. Treat that path as a shared
workflow cache and transfer area, not as canonical analysis storage.

Current Janelia smoke default:

```text
/nrs/johnson/palette_staging/flat_roi_cache/
```

This path was used as the shared workflow cache root for the 2026-05-16
sickyfish smoke. If a different site or project allocation is used, pass
`--public-cache-root` explicitly.

Use this for:

- sharing one ROI cache across multiple downstream jobs;
- preserving cache artifacts after the builder job exits;
- allowing later jobs to stage the cache into their own node-local scratch.

Risk:

- PRFS/NFS is still shared storage. If downstream jobs perform heavy random ROI
  reads directly from PRFS, the bottleneck may simply move from canonical Zarr
  to shared-cache files.

Preferred policy:

- Build cache on node-local scratch.
- Validate and publish it to PRFS workflow scratch.
- Downstream jobs either read it directly only after benchmarking, or first
  stage/unpack it into their own node-local scratch for hot inference.

### Canonical Analysis Zarr

Canonical analysis Zarrs should store geometry-only crop runs and downstream
outputs, not temporary ROI cache pixels.

Do not write workflow ROI caches under `analysis.zarr/crop_runs/<run>/` unless
the operator explicitly asks to materialize that crop run.

## Cache Backends

### Zarr ROI Cache

This is the nearest-term implementation because Palette already has a temporary
ROI cache path.

Shape:

```text
roi_cache_<key>.zarr/
  roi_images
  zarr.json
  cache_manifest attrs
```

Advantages:

- Existing code path.
- Chunked access.
- Easy shape/dtype metadata.
- Similar reader semantics to existing materialized crops.

Costs:

- Many files and metadata operations.
- Shared-storage random reads can be expensive.
- Publishing to PRFS can be slower than moving one large file.

Use it first because it is already integrated, then benchmark whether PRFS
direct reads are acceptable.

### Flat Binary ROI Cache

A flat binary cache is a plausible optimization for workflow transfer and
node-local hot reads.

Shape:

```text
<cache_root>/
  <archive>__<crop_run>__<key>.flat_roi_cache.json
  <archive>__<crop_run>__<key>.flat_roi_cache.bin
```

Manifest example:

```json
{
  "schema": "palette_roi_cache_flat_bin_v1",
  "layout": "flat_bin_v1",
  "cache_complete": true,
  "cache_key": "...",
  "source": {
    "archive_path": "/groups/.../recording_analysis.zarr",
    "crop_run_name": "crop_...",
    "source_crop_storage_mode": "geometry_only",
    "crop_signature": "...",
    "frame_source_kind": "source_video_path",
    "frame_source_path": "/groups/.../Cam2010093.mp4"
  },
  "array": {
    "bin_path": "<archive>__<crop_run>__<key>.flat_roi_cache.bin",
    "dtype": "uint8",
    "shape": [336451, 512, 512],
    "order": "C",
    "row_stride_bytes": 262144,
    "total_bytes": 88264171520,
    "sha256": "optional"
  }
}
```

Advantages:

- One large data file plus one small manifest.
- Fast sequential copy to and from PRFS workflow scratch.
- Easy memory mapping for fixed-shape random row reads.
- No Zarr metadata overhead.

Costs and constraints:

- Only appropriate when ROI images are fixed shape and dtype.
- No native chunk metadata or partial-array semantics.
- Concurrent writes need explicit partitioning or a single writer.
- Validation must check file size, shape, dtype, row count, and checksum.
- Readers need a new adapter; this is not a drop-in Zarr array.

Flat binary should be treated as an experimental cache backend, not a canonical
format. The canonical source of truth remains:

```text
geometry-only crop run + source video + cache manifest
```

The flat binary cache may be worth implementing if benchmarks show that Zarr
cache publication or PRFS reads are a bottleneck.

Current implementation preference: implement the flat binary cache backend
early enough to benchmark it against the Zarr cache backend. The expected value
is not as a canonical format, but as a low-overhead transfer and hot-read cache
for fixed-shape ROI tensors.

Implementation shape:

```bash
scripts/py -m fisheye.utils.build_flat_roi_cache \
  /path/to/recording_analysis.zarr \
  --crop-run crop_<run> \
  --output-dir /nrs/johnson/palette_staging/flat_roi_cache/<workflow_id>/roi_cache \
  --batch-size 1024
```

For LSF cluster jobs, prefer the submit wrapper so the large payload is built on
node-local scratch and only the completed artifact pair is published to shared
workflow cache storage:

```bash
scripts/submit_flat_roi_cache_bsub.sh \
  --zarr /path/to/recording_analysis.zarr \
  --crop-run crop_<run> \
  --workflow-id <workflow_id> \
  --public-cache-root /nrs/johnson/palette_staging/flat_roi_cache
```

For the common "create crop geometry, then materialize a flat ROI cache" case,
use the two-job wrapper:

```bash
scripts/submit_crop_flat_roi_cache_bsub.sh \
  --zarr /groups/johnson/johnsonlab/jeremy/palette_smoke/<recording>/zarr/<recording>_analysis.zarr \
  --source-type refined \
  --workflow-id smoke_crop_flat_roi_cache_20260516 \
  --public-cache-root /nrs/johnson/palette_staging/flat_roi_cache \
  --run-label sickyfish_cam2010093_crop_cache \
  --crop-queue short \
  --cache-queue gpu_l4 \
  --cache-gpus 1 \
  --roi-live-acceleration gpu \
  --crop-walltime 1:00 \
  --cache-walltime 2:00
```

The wrapper submits two jobs. The crop job writes or reuses the geometry-only
crop run on a CPU queue. The cache job is submitted immediately with an LSF
dependency equivalent to `-w done(<crop_jobid>)`, so it remains pending until
the crop job exits successfully. If the crop job fails, the cache job does not
start.

For multi-recording LSF batches, use the fan-out wrapper:

```bash
scripts/submit_crop_flat_roi_cache_batches_bsub.sh \
  --file-list /path/to/analysis_zarrs.txt \
  --workflow-id <workflow_id> \
  --public-cache-root /nrs/johnson/palette_staging/flat_roi_cache \
  --cache-queue gpu_l4 \
  --cache-gpus 1
```

The fan-out wrapper defers crop-job registry writes by default. Each crop job
sets `PALETTE_DISABLE_REGISTRY_WRITES=1`, writes its crop run into the analysis
Zarr, and emits a per-recording crop status JSON. Each dependent cache job
publishes the flat cache payload and emits a cache status JSON. After all cache
jobs finish, the wrapper submits one CPU-only registry finalizer job:

```bash
scripts/py -m fisheye.utils.finalize_crop_flat_roi_cache_batch_registry \
  <batch-run-root> \
  --registry /groups/johnson/johnsonlab/jeremy/registries/palette_registry.sqlite \
  --apply
```

This serial finalizer verifies the crop/cache pairing, verifies the crop run is
complete, refreshes `crop_quality`, and marks `crop=ok` in
`recording_step_status`. Avoid `--inline-registry` for multi-recording batches:
parallel SQLite writes to the shared PRFS registry can corrupt the database.

For local workstation batches, use the Python wrapper instead of submitting LSF
jobs:

```bash
scripts/py -m fisheye.utils.crop_flat_roi_cache_batch \
  /nvme1/recordings \
  --source registry \
  --registry /nvme1/palette_registry.sqlite \
  --path-contains GoodCopBadCop \
  --source-type refined \
  --selection-policy full_recording \
  --crop-storage-mode geometry_only \
  --workflow-id goodcopbadcop_crop_cache_20260604 \
  --cache-root /nvme1/palette_roi_cache \
  --cache-decode-backend pynvvc_luma \
  --roi-live-acceleration gpu \
  --apply
```

`crop_flat_roi_cache_batch` is dry-run by default. In apply mode it processes
archives serially: create or reuse the crop run, resolve the resulting crop
run, then call `build_flat_roi_cache` into
`<cache-root>/<workflow-id>/roi_cache`. Existing matching crop runs are not
recomputed unless `--force-new` is passed, and existing matching flat-cache
manifests are reused unless `--overwrite-cache` is passed.

For finalized clipped refined-detect collections, skip synthetic crop-run
creation and submit the collection-aware cache builder directly:

```bash
scripts/submit_clipped_collection_flat_roi_cache_bsub.sh \
  --zarr /groups/johnson/johnsonlab/jeremy/palette_smoke/<recording>/zarr/<recording>_analysis.zarr \
  --collection-id <workflow_id> \
  --public-cache-root /nrs/johnson/palette_staging/flat_roi_cache \
  --run-label <recording>_<workflow_id>_roi_cache \
  --queue gpu_l4 \
  --gpus 1 \
  --walltime 2:00
```

Use `--limit-rows 1024` for the first LSF smoke. The wrapper writes a job
script, submission context, stdout/stderr, status JSON, progress JSONL, and a
builder manifest snapshot. It builds under `$PALETTE_JOB_CACHE`, publishes
`.bin` and `.rows.parquet` first, then publishes the `.json` manifest last.

Observed Janelia queue note: `normal` is not a valid queue on the checked
cluster. The wrapper defaults now use `short`; GPU cache builds should pass an
explicit GPU queue such as `--cache-queue gpu_l4 --cache-gpus 1`.

The wrapper writes:

```text
/scratch/$USER/$LSB_JOBID/palette_cache/flat_roi_cache/<label>.flat_roi_cache.{json,bin}
```

During cache construction the wrapper also writes progress telemetry in the run
directory:

```text
runs/diagnostics/.../<label>.cache.<jobid>.progress.jsonl
```

The flat-cache builder has a dedicated decode backend selection:

```bash
scripts/py -m fisheye.utils.build_flat_roi_cache \
  /path/to/recording_analysis.zarr \
  --decode-backend auto \
  --roi-live-acceleration cpu \
  --output-dir /nrs/johnson/palette_staging/flat_roi_cache/<workflow_id>/roi_cache
```

For geometry-only crop runs backed by an external video, `auto` prefers
`pynvvc_luma`: a sequential PyNvVideoCodec/NVDEC path that streams frames from
the beginning of the file, crops only frames with ROI rows, and writes those rows
directly into the flat binary payload. This avoids Decord `VideoReader` startup
and random-access indexing costs. If PyNvVideoCodec cannot be used and the
backend was not explicitly forced, `auto` falls back to the generic `read_slice`
path. Use `--decode-backend pynvvc_luma` when the job should fail instead of
falling back.

`read_slice` is retained as a compatibility/reference backend, not the intended
long-video cluster materialization path.

Each JSONL record reports rows and bytes written, elapsed time, ETA, aggregate
ROI throughput, and per-batch timing for decode/read, crop, contiguous
conversion, serialization, and file write. The submit wrapper also passes
`--progress-stderr`, so compact progress summaries appear in the LSF stderr file
while the builder is running. A non-empty cache stderr file is therefore not
necessarily a failure; check the final status JSON and LSF exit state.

then publishes payload first and manifest last:

```text
/nrs/johnson/palette_staging/flat_roi_cache/<workflow_id>/roi_cache/<label>.flat_roi_cache.bin
/nrs/johnson/palette_staging/flat_roi_cache/<workflow_id>/roi_cache/<label>.flat_roi_cache.json
```

Downstream pose/segmentation stages should not parse binary payloads directly.
They pass the manifest to `CropImageSource`:

```bash
scripts/py -m fisheye.detection.detect_keypoints_yolo \
  /path/to/recording_analysis.zarr \
  --model /path/to/best.pt \
  --roi-cache-manifest /nrs/johnson/palette_staging/flat_roi_cache/<workflow_id>/roi_cache/<cache>.json
```

`CropImageSource` is the adapter boundary. It validates the manifest against the
selected archive/crop run, memory-maps the payload as `[roi, height, width]`
`uint8`, and exposes the same batch/slice interface used for materialized Zarr
crops and geometry-only live reads. This keeps pose and segmentation code
cache-format agnostic.

Performance remains an empirical question. Flat binary is expected to reduce
metadata overhead and make sequential copy/staging cheap. Zarr may still win for
tooling compatibility, chunk-local random access, or compressed/sharded storage.
Benchmarks should compare both backends on PRFS direct reads and node-local
staged reads before changing workflow defaults.

## Crop Pixel Contract

Downstream pose and segmentation jobs should receive the same logical crop data
they received from local materialized crop runs:

```text
shape: [roi, roi_height, roi_width]
dtype: uint8
layout: C-order
semantics: grayscale ROI pixels
coordinates: crop_runs/<run>/roi_coordinates_full top-left coordinates
padding: zero outside source-frame bounds
row order: crop_runs/<run> row order
```

The cache format is an implementation detail. Pose and segmentation consumers
should continue to read through `CropImageSource` and should not parse flat
binary payloads directly.

Future crop runs record this in attrs as:

- `roi_image_representation`: currently `uint8_grayscale_roi_v1`.
- `roi_pixel_contract`: structured conversion metadata with contract name,
  source-frame representation, dtype/shape/order, padding behavior, and
  production status.
- `roi_pixel_contract_name`: scalar copy of `roi_pixel_contract.name`, used as
  the cheap registry/export filter key.

Downstream pose and segmentation runs copy the effective reader/cache contract
as:

- `source_roi_image_representation`
- `source_roi_pixel_contract_name`
- `source_roi_pixel_contract`

Those downstream attrs describe the pixels actually consumed. For example, a
geometry-only crop run may have a deferred contract, while a flat cache consumed
through `--roi-cache-manifest` records `nv12_luma_plane_uint8` from the cache
manifest.

The current flat-cache PyNvVideoCodec backend is `pynvvc_luma`. It crops the
decoded NV12 Y plane directly. For Orange monochrome camera recordings, this is
the accepted production crop/cache/training pixel contract:

```text
contract: pynvvc_luma_v1
shape: [roi, roi_height, roi_width]
dtype: uint8
source: decoded NV12 Y/luma plane from Orange mono camera MP4
semantics: mono camera intensity before model-specific resize/letterbox
```

The Orange runtime audit on 2026-05-16 found that current Orange TensorRT
detection and pose deployments both start from single-channel mono/luma. Orange
then performs preprocessing outside TensorRT:

- detection: luma -> resize/letterbox -> replicate to 3 planar channels -> /255
  -> FP32 NCHW `1x3x640x640`
- pose: luma ROI crop -> resize/letterbox -> replicate to 3 planar channels ->
  /255 -> FP32 NCHW, currently documented as `1x3x256x256` for the first real
  pose engine
- no mean/std normalization was reported
- no deployed TensorRT segmentation path was found in Orange at that time

Therefore Palette should cache/train from `[N,H,W] uint8` luma crops for mono
Orange recordings and perform model-specific tensorization later. Normalized CHW
tensors or replicated RGB tensors should be considered runtime/model-input
products, not canonical cache artifacts, because their size, channel layout, and
padding depend on the selected engine.

This contract is intentionally not byte-identical to every historical Palette
materialization path:

- OpenCV CPU crop paths have used `cv2.COLOR_BGR2GRAY`.
- Decord GPU crop paths have used RGB channel mean.
- `pynvvc_luma` uses the raw NV12 luma plane.
- Detection inference uses `pynvvc_nv12_rgb` as the correctness-oriented PyNv
  backend, because YOLO expects RGB-like input and fixed-frame parity favored
  NV12-to-RGB over luma replication.

Historical crop parity checks remain useful for quantifying the migration, but
the production question is now consistency with the explicit `pynvvc_luma_v1`
contract, not strict equality with old OpenCV/Decord-derived crop pixels.

### PyNvVideoCodec Surface Lifetime

The `pynvvc_luma` cache builder must not retain decoded frame tensors returned
by PyNvVideoCodec across decoder advancement. Palette receives those tensors via
`torch.from_dlpack(frame)`, which should be treated as a view over
decoder-owned GPU memory. Empirical GoodCopBadCop validation on 2026-06-05
showed that collecting a batch of decoded frame tensors with `decode_next(N)`
and cropping them later can corrupt cache rows: earlier tensors may point at
surfaces reused for later frames.

Accepted safe policy:

- Iterate decoded frames one at a time.
- For frames with no ROI rows, decode and skip without copying/cropping.
- For frames with ROI rows, crop immediately while the decoder surface is still
  current.
- Clone/copy only the derived ROI tensors into owned staging memory.
- Batch only owned ROI payloads for host transfer and disk writes.

Rejected optimization:

- "Decode many full frames, then batch-crop later" is unsafe unless the decoded
  full frames are first cloned into owned GPU memory or PyNvVideoCodec provides
  a documented retained-surface guarantee.
- Cloning full 4512x4512 luma frames was tested conceptually during the
  2026-06-05 GoodCopBadCop investigation and is the wrong tradeoff for this
  cache: it preserves correctness but loses the benefit of copying only
  512x512 ROI pixels.

The current safe persisted-cache path is therefore "immediate ROI clone, then
batched owned-ROI writes." A future direct inference path may keep ROI tensors
on GPU and avoid this flat-cache disk write, but persisted flat caches must own
their bytes before decoder advancement.

Transfer/write policy for the persisted flat cache:

- Concatenate only owned ROI tensors, not borrowed decoder frame tensors.
- Prefer a small ring of reusable pinned host buffers for GPU-to-host ROI
  payload copies.
- Hand each pinned host buffer directly to the asynchronous writer and do not
  reuse that buffer until its write future completes.
- Write contiguous ROI row runs as a single file write; retain sparse-row
  sorting as a fallback for non-sequential cache construction.
- Track `gpu_cat_seconds_total`, `gpu_to_host_seconds_total`, pinned/pageable
  transfer counts, and writer wait time in the manifest timing block so cache
  builds can be evaluated from artifacts rather than anecdotes.

### Finalized Clipped Collection Cache

For clipped recordings, downstream pose and segmentation should not first merge
all clip-local detections into a parent dense crop run just to build a runtime
cache. The collection-aware builder consumes the finalized clipped
refined-detect collection directly:

```bash
scripts/py -m fisheye.utils.build_clipped_collection_flat_roi_cache \
  /path/to/recording_analysis.zarr \
  --collection-id <workflow_id> \
  --output-dir /nrs/johnson/palette_staging/flat_roi_cache/<workflow_id>/roi_cache \
  --progress-jsonl /path/to/cache.progress.jsonl \
  --progress-stderr
```

The builder resolves `experiment_index/finalized_runs/<workflow_id>`, joins it
to `recording_frame_index.parquet`, reads each selected
`clips/<clip>/cameras/<camera>/refined_detect_runs/<run>/instances` group, and
derives the same centered fixed-size ROI coordinates as `crop_batch`.

Root attrs policy:

- Root path/provenance attrs such as `recording_path` and `source_video_path`
  remain current for single-video archives and migrated smoke copies.
- Root image-dimension attrs such as `width` and `height` remain valid for
  single-video archives. They may also be valid parent-level invariants for
  clipped recordings when all clips/cameras share the same resolution.
- Clipped collection cache builders should prefer refined pixel-space
  `instances/bbox_img_xyxy` for ROI geometry because those boxes are already the
  refined row-level pixel coordinates. If only normalized boxes are present,
  root `width`/`height` may be used as a compatibility fallback when the
  dimensions are unambiguous; otherwise the builder should fail rather than
  silently invent geometry.

It writes three sibling artifacts:

```text
<label>.flat_roi_cache.json
<label>.flat_roi_cache.bin
<label>.flat_roi_cache.rows.parquet
```

The `.bin` payload remains `[N,H,W] uint8` flat luma crops. The sidecar
`.rows.parquet` is required for clipped collection caches because there is no
single `crop_runs/<run>` row table. It records at least:

- `roi_row_index`
- `clip_id`, `clip_index`, `camera_serial`
- `clip_local_frame_index`, `recording_frame_id`, `parent_frame_index`
- `refined_group_path`, `refined_detect_run`, `refined_row_id`
- `source_detect_row_index`
- `bbox_norm_*`, `roi_x`, `roi_y`, `roi_w`, `roi_h`
- `video_path`, `metadata_path`, `keyframe_path`

Downstream pose/segmentation code should still read ROI pixels through a cache
adapter and should use the row-index parquet for lineage/output placement. It
should not parse `.bin` directly and should not infer parent frame identity from
row number.

## Crop Pixel Parity Checklist

Before making a PyNv flat ROI cache the default input surface for pose or
segmentation:

- [x] Define and record explicit crop/cache pixel-contract metadata for future
  crop runs and downstream pose/segmentation runs.
- [x] Decide the accepted production grayscale contract for runtime analysis
  crops: OpenCV weighted grayscale, Decord historical mean grayscale, raw NV12
  luma, or a new explicit conversion.
- [x] Add a fixed-row parity utility that compares the canonical
  `CropImageSource` reader path against a flat ROI cache for the same
  archive/crop run.
- [x] Add a training-zarr parity utility that compares stored
  `crop_runs/<run>/roi_images` against crops reconstructed from the original
  MP4 with the sequential PyNv luma path.
- [ ] Sample rows from early, middle, and late frames, plus at least one
  near-boundary padded crop if available.
- [ ] Report byte equality, max absolute difference, mean absolute difference,
  p95 absolute difference, and a small image-diff artifact for mismatches.
- [x] Record `pixel_contract`, `decode_backend_effective`, source video
  identity, crop run, and conversion details in the flat-cache manifest.
- [x] Decide whether an alternative production backend such as
  `pynvvc_legacy_gray` or `pynvvc_nv12_gray` is required. It is not required for
  mono Orange recordings while `pynvvc_luma_v1` is the selected contract.
- [ ] If strict parity requires a different conversion, update
  `--decode-backend auto` to prefer the parity-accepted sequential PyNv backend,
  not the slow `read_slice` path.
- [ ] Add a wrapper validation mode that can fail a cache job when parity
  exceeds configured tolerances.
- [ ] Run one downstream pose/keypoint smoke and one segmentation smoke using
  `--roi-cache-manifest` from the flat cache.
- [ ] Keep `read_slice`/materialized Zarr as fallback until parity and downstream
  consumer smokes pass on more than one recording.

Working interpretation as of 2026-05-16: for Orange monochrome camera
recordings, `pynvvc_luma_v1` is the accepted crop/cache/training pixel contract.
Flat-cache mechanics and downstream model smokes still need validation, but the
pixel representation decision is no longer blocked on historical crop parity.

Strict parity check:

```bash
scripts/py -m fisheye.diagnostics.check_flat_roi_cache_pixel_parity \
  /path/to/recording_analysis.zarr \
  --roi-cache-manifest /nrs/johnson/palette_staging/flat_roi_cache/<workflow>/roi_cache/<label>.flat_roi_cache.json \
  --reference-roi-live-acceleration gpu \
  --sample-count 64 \
  --output-json /path/to/parity_report.json
```

The default thresholds are strict byte equality. A nonzero result means the flat
cache does not match the currently configured `CropImageSource` path. Use this
to catch implementation drift, but do not require new `pynvvc_luma_v1` crops to
match historical OpenCV/Decord materializations byte-for-byte.

## Training Zarr Comparison

It is possible to compare existing training crops against the new PyNv workflow,
but only when the training rows can be mapped back to the original video frames
and crop geometry.

Straightforward case:

- A per-recording training zarr still has `crop_runs/<run>/roi_images`,
  `frame_indices`, `roi_coordinates_full`, ROI size, and a readable
  `source_video_path`.
- If `raw_video/original_frame_indices` is present, crop `frame_indices` are
  local sampled-frame indices and must be mapped through that array before
  decoding the source video.
- Use the training parity diagnostic to decode the source video with
  PyNvVideoCodec, reconstruct the same ROI rows, and compare against stored
  `roi_images`.

Example:

```bash
scripts/py -m fisheye.diagnostics.check_training_crop_pynvvc_pixel_parity \
  /path/to/recording_training.zarr \
  --crop-run crop_2026-02-03_23-34-39 \
  --video-path /path/to/source_video.mp4 \
  --sample-count 64 \
  --output-json /tmp/training_crop_pynvvc_parity.json
```

`--video-path` is optional when the zarr's `source_video_path` is readable from
the current machine. Use it for copied PRFS smoke inputs whose stored attrs
still point at the original workstation path.

For small per-recording training zarrs, `--all-rows` is appropriate. For long
sampled videos, use indexed PyNvVC access rather than sequentially decoding up
to late source frames. For rolling-clip training zarrs, prefer
`source_frame_index.parquet` so crop rows decode from `video_path +
clip_local_frame_index` instead of reopening/seeking through the parent MP4.

Detection-only sampled training zarrs are different. They may have
`detect_runs` and `refined_detect_runs` but no `crop_runs`; this is expected for
detector-only training because bounding boxes are read from the refined-detect
surface directly. `fisheye.utils.regenerate_training_crops_pynvvc` requires an
existing crop run with `frame_indices` and `roi_coordinates_full`, so it should
not be run on detection-only zarrs. If one of those recordings later becomes a
pose/keypoint or segmentation source, first create crop geometry from the
approved refined detections, then materialize a new `pynvvc_luma_v1` crop run.

Observed inventory on 2026-05-16, updated after the first clipped-source crop
migration on 2026-05-17:

- 60 approved detector-training source zarrs exist under `/nvme1/recordings`.
- 52 have `crop_runs`; all 52 already have a `pynvvc_luma_v1` crop run.
- 4 `sickyfish` sampled training zarrs now have `pynvvc_luma_v1` crop runs
  sourced from their single-camera MP4s.
- 4 `sleepyfish` clipped training zarrs now have `pynvvc_luma_v1` crop runs
  sourced from rolling clips through `source_frame_index.parquet`.
- The 4 non-clipped `sleepyfish` sampled training zarrs still carry their
  original grayscale crop runs. Avoid regenerating those from the parent MP4;
  use the clipped training zarrs or add an explicit clip-frame mapping first.

Merged/exported training zarrs need more care:

- The exported `roi_images` are copied from source training zarrs.
- If the export preserved source archive/run/row lineage, compare by grouping
  rows back to each source archive and then running the same row-level parity
  check.
- If that lineage is absent or source videos are no longer reachable, strict
  pixel parity cannot be reconstructed; only weaker distributional checks or
  model-output comparisons are possible.

Recommended acceptance gate:

- First run pixel parity: byte equality when the intended contract is identical,
  or explicit tolerances when accepting a different grayscale contract such as
  NV12 luma.
- Then run model-output parity on a fixed row sample: pose keypoint coordinates,
  confidences, and segmentation probabilities/masks. This catches cases where
  small pixel differences matter to the trained model.

## Source Video Path Portability

Geometry-only crop runs still need a readable video source during creation.
They skip ROI pixel extraction, but the crop writer resolves and validates the
frame source so the crop run can record durable video provenance for later
live reads or cache materialization.

Copied archives can therefore fail on the cluster if their root metadata still
points at a workstation-only path such as `/nvme1/recordings/...`, even when
the MP4 has been copied beside the archive under `/groups/...`. The 2026-05-16
sickyfish smoke hit this exact failure:

```text
Video path in metadata not found:
/nvme1/recordings/.../cams/Cam2010093_....mp4
```

The archive copy was repaired by updating the copied Zarr's root attrs and
`raw_video` attrs to the cluster-visible MP4:

```text
root.attrs["recording_path"] = "/groups/.../<recording>"
root.attrs["source_video_path"] = "/groups/.../<recording>/cams/<camera>.mp4"
root.attrs["source_path"] = "/groups/.../<recording>/cams/<camera>.mp4"
root.attrs["source_video_metadata"]["source_path"] = "/groups/.../<recording>/cams/<camera>.mp4"
raw_video.attrs["source_path"] = "/groups/.../<recording>/cams/<camera>.mp4"
```

For future migrated archives, this should be handled during migration/import:
`source_video_path` must be a path readable from the compute nodes. For ad-hoc
smoke copies, repair the copied archive metadata or add an explicit
source-video override before submitting crop/cache jobs. Do not edit the
original workstation archive merely to satisfy a cluster smoke copy.

## NFS/PRFS Read Policy

The main risk of geometry-only crops is accidentally replacing permanent Zarr
storage cost with repeated remote decode and random reads.

Avoid this anti-pattern:

```text
keypoints job:
  reads source video over PRFS and builds its own ROI cache

eye-mask job:
  reads source video over PRFS and builds its own ROI cache again

subject-mask job:
  reads source video over PRFS and builds its own ROI cache again
```

Prefer:

```text
ROI cache build job:
  reads source video once
  writes cache to node-local scratch
  publishes validated cache to PRFS workflow scratch

downstream GPU jobs:
  stage cache to node-local scratch when possible
  run inference from local cache
  write outputs as normal stage artifacts
```

Direct PRFS cache reads are allowed only when a benchmark shows they are not
the bottleneck for that workload.

Current policy from the 2026-06-18 GoodCopBadCop L4 benchmark:

- large flat-bin caches should be staged to node-local scratch before GPU
  keypoint/segmentation inference;
- direct PRFS reads are acceptable for small caches or explicit diagnostic
  comparisons;
- use `--stage-roi-cache-to-scratch` when an explicit
  `--roi-cache-manifest` is known.

Measured result for one 33.4 GiB flat cache:

| Mode | Cache copy | Inference | Throughput | End-to-end |
| --- | ---: | ---: | ---: | ---: |
| Direct PRFS flat cache | 0s | 643.9s | 212.2 poses/s | 643.9s |
| Node-local staged flat cache | 45.8s | 495.5s | 275.8 poses/s | 541.3s |

This was a roughly 30% inference-throughput improvement and remained faster
after paying the one-time copy cost. Treat 5 GiB as the initial policy threshold
where staging should be expected unless a workflow-specific benchmark says
otherwise.

Initial benchmark question: compare direct reads from PRFS workflow cache
against staging the same cache to node-local scratch. The prior video benchmark
showed PRFS video reads can be close to local reads for sequential decode, but
ROI cache access is a different pattern. Do not assume video-read parity means
cache-read parity.

## Provenance Requirements

Downstream runs that consume a cache should record:

- `source_crop_run`
- `source_crop_storage_mode`
- `source_crop_signature`
- `source_video_path`
- `source_video_fingerprint`
- `roi_cache_policy`
- `roi_cache_backend`
- `roi_cache_key`
- `roi_cache_manifest_path`
- `roi_cache_source_tier` (`node_scratch`, `prfs_workflow_scratch`, or
  `canonical_materialized`)
- `roi_cache_staged_to_node_scratch`
- `roi_cache_staging_policy` (`node_scratch_staged_flat_cache` or
  `direct_manifest_read`)
- `roi_cache_validation_status`
- staging recommendation fields for explicit manifests:
  `staging_recommended`, `staging_recommendation_min_bytes`,
  `staging_recommendation_reason`, and `staging_recommendation_basis`
- requested and effective device metadata:
  `requested_device`, `normalized_torch_device`, `initial_model_device`, and
  `resolved_model_device`
- scheduler placement for cluster runs: `execution_hostname`,
  `scheduler_job_id`, `scheduler_job_index`, `scheduler_queue`,
  `scheduler_hosts`, `scheduler_mcpu_hosts`, `scheduler_gpu_request`, and
  `scheduler_cuda_visible_devices`

The cache manifest should record:

- builder command
- git commit
- LSF job id and host
- source archive path
- crop run path
- crop signature
- source video fingerprint
- backend schema
- shape/dtype/order
- checksum
- timing for decode, crop, write, package, and publish

Workflow logs should also record where each cache was written:

- node-local scratch path;
- PRFS workflow scratch path, when published;
- cache backend (`zarr` or `flat_binary`);
- whether each downstream stage read directly from PRFS or staged locally;
- downstream stage success/failure status.

The registry does not need to track temporary ROI caches in the near-term.
Cache lifecycle can be managed from workflow manifests/logs.

Current implementation status:

- Flat cache manifests record schema/layout, source archive, crop run,
  source frame path, ROI shape, payload byte count, optional checksum, builder
  batch size, pixel contract, and phase timing summaries.
- Finalized clipped refined-detect collections can also be cached without a
  `crop_runs/<run>` source. These manifests record collection identity and a
  sidecar `.rows.parquet` path for row lineage.
- Submit wrappers write stdout/stderr plus crop/cache status JSON files with
  job id, host, status, published paths, byte size, and source/array metadata.
- Downstream readers that accept `--roi-cache-manifest` record cache policy,
  backend, key, and manifest path in run attrs/provenance.
- Per-phase timings for decode/read, ROI extraction, contiguous conversion,
  serialization, local `.bin` write, PRFS payload copy, manifest-last publish,
  and validation are captured in flat-cache builder and wrapper telemetry.
- The registry-model keypoint wrapper can stage an explicit flat-cache manifest
  and payload to node-local scratch before inference. The output keypoint run
  records the requested manifest path, effective manifest path, source tier,
  staging policy, staging recommendation, copy timing, and validation status.
  Multi-recording manifest-directory resolution remains a separate
  workflow-planning task.
- GPU keypoint jobs can be submitted with `--gpus N`; when a GPU is requested
  and `--device` is not set, the LSF wrapper passes `--device 0` into the
  per-recording command. Output keypoint runs mirror device and LSF placement
  metadata into run attrs, stage provenance, and registry status details, so
  direct PRFS reads versus staged-cache reads can be compared by actual host/GPU
  allocation.

## Cache Lifecycle

Temporary workflow caches should have a cleanup policy, but they should not
become registry-managed canonical data.

Recommended near-term behavior:

- A cache builder writes a cache manifest.
- Each downstream stage logs whether it consumed the cache and whether it
  succeeded.
- If all requested downstream stages succeed, the workflow can mark the cache
  eligible for TTL cleanup.
- If a downstream stage fails, retain the cache until the workflow is retried or
  the operator explicitly cleans it.
- Cleanup tools should operate on workflow-cache manifests, not on registry
  rows.

Default TTL remains an implementation decision. The important policy is that
successful workflow caches are disposable, while failed/incomplete workflow
caches are retained long enough for retry/debugging.

## Job Placement And Multi-Step Reporting

Keeping cache builder and downstream pose/segmentation jobs on the same node is
attractive because it avoids shared-cache staging. It is not always the right
default because:

- CPU-only cache/import/validation jobs should not hold a GPU allocation;
- Dask-capable stages may need their own scheduler/resource shape;
- LSF may schedule dependent jobs on different nodes unless explicitly
  constrained;
- a long multi-step allocation makes failure/retry coarser.

Near-term policy:

- Design the workflow as cross-node safe.
- Record every stage as a separate job/report with explicit dependencies.
- Use shared PRFS workflow cache as the handoff artifact.
- Use optional node-local staging for downstream hot inference when a completed
  flat-cache manifest is already known.
- Revisit same-node placement only after measuring cache staging overhead.

Workflow reporting should make partial failure obvious. A workflow manifest
should track:

- planned stages;
- submitted LSF job ids;
- dependency edges;
- cache manifest paths;
- per-stage status JSON paths;
- final status: `complete`, `failed`, or `partial`.

This gives the operator a stable answer even if a multi-step submission fails
halfway through.

## Crimson And Review Tools

The long-term review direction may be Crimson-first. That is compatible with
geometry-only canonical crops, but Crimson and Palette review tools need an
explicit cache/read policy.

Acceptable review modes:

- live video + geometry read for sparse interactive inspection;
- workflow ROI cache read for bulk review;
- on-demand materialized review cache for a selected run;
- explicit materialized crop run when portability is more important than
  storage cost.

Review tools should not assume `crop_runs/<run>/roi_images` exists on analysis
archives once geometry-only becomes the default.

## Rollout Plan

1. Keep training archives materialized.
2. Update docs and defaults so analysis archives can default to geometry-only.
3. Update batch planners to treat `latest_any` as crop-ready for mixed-mode
   consumers.
4. Keep materialized-only readers on `latest_materialized`.
5. Inventory Palette and Crimson readers/review tools that still require
   materialized `roi_images`.
6. Add a workflow ROI cache builder that writes to node-local scratch and
   publishes to PRFS workflow scratch.
7. Implement both Zarr and flat binary cache backends for benchmark parity.
8. Add cache staging support for downstream GPU jobs.
9. Benchmark Zarr-cache and flat-binary-cache reads from PRFS versus node-local
   staging.
10. After major readers are migrated, change `crop_runs.latest` to latest-any.

## Open Questions

- What TTL should successful workflow caches use?
- Are PRFS workflow-cache reads fast enough for ROI inference, or should
  downstream jobs always stage caches to node-local scratch?
- Is same-node placement worth the scheduler complexity after cache staging is
  benchmarked?
- Which Palette and Crimson review surfaces must support geometry-only before
  `crop_runs.latest` becomes latest-any?
- Should the crop/cache submitter grow a first-class `--source-video-path`
  override or repair/preflight helper for copied archives whose metadata still
  points at workstation-only paths?

## Required Reader Inventory

Before changing `crop_runs.latest` to latest-any, run a dedicated inventory of
all readers that touch crop pixels. The inventory should classify each reader:

- mixed-mode safe through `CropImageSource`;
- materialized-only by design;
- stale direct `crop_group["roi_images"]` access that should migrate;
- review/tooling path where geometry-only support can be deferred;
- Crimson consumer needing a contract update.

Current Palette inventory: `docs/archive/crop_reader_geometry_only_inventory_2026-05-16.md`.
Future updates should keep a checklist with owner, migration requirement, and
blocking status for the `latest` cutover.
