# Geometry-Only Crop Workflow Cache Design
<!-- contract-meta
status: design
last_verified: 2026-05-16
purpose: Define the target policy for geometry-only analysis crop runs, shared workflow ROI caches, and optional flat binary crop-cache transport.
-->

## Purpose

Palette is moving toward lean analysis archives where crop runs store geometry,
lineage, and provenance, while ROI pixels are materialized only when they are
needed for training artifacts, review caches, or high-throughput inference.

This document defines the intended workflow shape before implementation.

It complements:

- `docs/crop_live_view_vs_materialized_stream_design.md`
- `docs/crop_storage_mode_migration_todo.md`
- `docs/cluster_workflow_orchestration.md`
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
        roi_size
        crop_signature
        detection_source_path
        source_video_path
        source_video_fingerprint
```

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

Current proposed default:

```text
misc/public/palette_cache/
```

Use the site-qualified absolute path for actual jobs once confirmed from the
cluster login/compute environment.

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
  --output-dir /misc/public/palette_cache/<workflow_id>/roi_cache \
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
  --public-cache-root /misc/public/palette_cache
```

The wrapper writes:

```text
/scratch/$USER/$LSB_JOBID/palette_cache/flat_roi_cache/<label>.flat_roi_cache.{json,bin}
```

then publishes payload first and manifest last:

```text
/misc/public/palette_cache/<workflow_id>/roi_cache/<label>.flat_roi_cache.bin
/misc/public/palette_cache/<workflow_id>/roi_cache/<label>.flat_roi_cache.json
```

Downstream pose/segmentation stages should not parse binary payloads directly.
They pass the manifest to `CropImageSource`:

```bash
scripts/py -m fisheye.detection.detect_keypoints_yolo \
  /path/to/recording_analysis.zarr \
  --model /path/to/best.pt \
  --roi-cache-manifest /misc/public/palette_cache/<workflow_id>/roi_cache/<cache>.json
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
- `roi_cache_validation_status`

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
- Add optional node-local staging for downstream hot inference.
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

- What is the site-qualified absolute path for `misc/public/palette_cache` from
  Janelia login and compute nodes?
- What TTL should successful workflow caches use?
- Are PRFS workflow-cache reads fast enough for ROI inference, or should
  downstream jobs always stage caches to node-local scratch?
- Is same-node placement worth the scheduler complexity after cache staging is
  benchmarked?
- Which Palette and Crimson review surfaces must support geometry-only before
  `crop_runs.latest` becomes latest-any?

## Required Reader Inventory

Before changing `crop_runs.latest` to latest-any, run a dedicated inventory of
all readers that touch crop pixels. The inventory should classify each reader:

- mixed-mode safe through `CropImageSource`;
- materialized-only by design;
- stale direct `crop_group["roi_images"]` access that should migrate;
- review/tooling path where geometry-only support can be deferred;
- Crimson consumer needing a contract update.

Current Palette inventory: `docs/crop_reader_geometry_only_inventory_2026-05-16.md`.
Future updates should keep a checklist with owner, migration requirement, and
blocking status for the `latest` cutover.
