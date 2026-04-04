# Zarr Transfer Benchmark Plan

## Purpose

Define the benchmark matrix for comparing three off-machine movement formats for
the same Palette archive:

- raw `.zarr` directory copy
- packed `.tar.zst` transfer artifact
- rewritten sharded `.zarr` clone

This plan is intentionally based on **real** archives, not synthetic stores.
The goal is to measure the actual mix of:

- bulk frame data
- dense derived arrays
- metadata fanout
- historical run baggage

## Source Selection

Do **not** generate a fake benchmark dataset first.

Instead:

1. Pick one representative real archive.
2. Pick one worst file-count archive.
3. Derive alternate transport layouts from those exact inputs.

Recommended first pair:

- representative archive:
  `2026-01-28T22-15-03Z_arena_1_DefaultScreen_training.zarr`
- worst file-count archive:
  `2026-01-28T20-41-59Z_arena_1_DefaultScreen_training.zarr`

This keeps bytes, file count, chunking, and metadata realistic.

## Benchmark Variants

For each source archive, benchmark these three variants:

### 1. Raw Directory

The source archive copied as-is.

Measures:

- copy time to destination
- destination file count
- destination size
- open/read timing on destination

### 2. Packed Transfer Artifact

Use the existing packer:

`scripts/py -m fisheye.utils.pack_zarr_transfer_artifact <archive>.zarr --apply`

Measures:

- pack time
- artifact size
- copy time to destination
- unpack time on destination
- open/read timing after unpack

Optional variant:

- repeat with regenerable compatibility groups excluded, for example
  `--exclude-top-level refined_eye_masks_runs`

### 3. Sharded Zarr Clone

Rewrite the real archive into a new benchmark-only `.zarr` store where selected
arrays are sharded.

Measures:

- rewrite time
- resulting file count
- resulting size
- copy time to destination
- open/read timing on destination

No unpack step is needed because the result remains a browseable `.zarr`.

## Benchmark Questions

The benchmark should answer three separate questions:

### A. Best transport format

If the goal is just moving data off-machine, which wins:

- raw directory
- prebuilt `.tar.zst`
- sharded clone

### B. Best move-and-use format

If the goal is to move data and then browse or run read-mostly tools on the
destination, which wins:

- raw directory copied directly
- `.tar.zst` copied then unpacked
- sharded clone copied directly

### C. Best long-lived online format

If the goal is a browseable online store on NFS, is the sharded clone better
than the raw working layout for:

- open latency
- single-frame reads
- ROI image reads
- subject-mask reads

## Metrics To Record

For every run, record:

- source archive path
- archive variant (`raw`, `tar_zst`, `sharded_clone`)
- source bytes
- source file count
- destination bytes
- destination file count
- build/export time
- copy time
- unpack time, if applicable
- open-root time
- one representative frame-read time
- one representative ROI image-read time
- one representative subject-mask read time

The benchmark should keep raw timing data in a machine-readable manifest rather
than only terminal notes.

### Destination Read Benchmark Utility

Use:

`scripts/py -m fisheye.utils.benchmark_zarr_destination_reads <archive>.zarr --variant <label> --json`

This utility records:

- open-root timing
- one `raw_video/images_full` frame read
- one `crop_runs/*/roi_images` row read
- one `subject_mask_runs/*/masks_roi` row read

## Existing Commands

### Raw inspection

Use:

`scripts/py -m fisheye.utils.report_zarr_storage <archive>.zarr`

### Destination read benchmark

Use:

`scripts/py -m fisheye.utils.benchmark_zarr_destination_reads <archive>.zarr --variant <label> --json`

### Packed artifact build

Use:

`scripts/py -m fisheye.utils.pack_zarr_transfer_artifact <archive>.zarr --apply`

### Raw directory copy

Use `cp` or `rsync`, but keep the command stable across variants for a fair
comparison.

## Sharded Export Prototype

Prototype utility:

`scripts/py -m fisheye.utils.export_sharded_zarr_clone <source>.zarr --dest <dest>.zarr --policy <policy> --apply`

### Current prototype contract

- never mutate the source archive
- always write to a new destination store
- preserve group structure and attrs
- preserve non-sharded arrays as normal chunked arrays
- preserve array chunks unless an explicit chunk override is added later
- add shard shapes only for arrays selected by the benchmark policy
- emit a manifest describing per-array source and destination layout

### First benchmark policies

The prototype does not need to support arbitrary policies first.

Start with:

- `raw_only`
  - shard only `raw_video/images_full`
  - shard only `raw_video/images_ds`
- `raw_and_crops`
  - plus `crop_runs/*/roi_images`
- `dense_readmostly_v1`
  - plus `subject_mask_runs/*/masks_roi`
  - plus `subject_mask_runs/*/mask_probs_roi`
  - plus `refined_subject_masks_runs/*/masks_roi`
  - plus `refined_subject_masks_runs/*/mask_probs_roi`
- `dense_readmostly_rechunk_v1`
  - same selection as `dense_readmostly_v1`
  - but rechunk dense subject-mask arrays to full-ROI chunks before sharding

Everything else should stay chunked for the prototype.

### First shard-shape rule

For the prototype:

- preserve existing chunk shape
- compute shard depth along axis 0 only
- target shard size in MB, for example `128 MB`
- clamp shard depth so it is at least one chunk and an integer multiple of the
  axis-0 chunk size

This keeps the prototype simple and avoids inventing stage-specific shard logic
too early.

### First rechunk rule

For `dense_readmostly_rechunk_v1`:

- keep raw video chunking unchanged
- keep crop ROI image chunking unchanged unless a later benchmark shows it
  should change
- rechunk `subject_mask_runs/*/{masks_roi,mask_probs_roi}` to
  `(min(16, n_rows), 1, H, W)`
- rechunk `refined_subject_masks_runs/*/{masks_roi,mask_probs_roi}` to
  `(min(16, n_rows), 1, H, W)`

This is intentionally aligned with the current raw subject-mask full-ROI chunk
policy so that the archival benchmark can answer whether old tiled historical
arrays are the main reason sharding alone underperforms.

### Arrays the prototype should avoid sharding initially

- scalar metadata arrays
- small metrics arrays
- lineage/index arrays
- string/object arrays
- manual-review authority arrays that are still actively edited

## Read Checks On Destination

The benchmark should include basic destination usability checks, not just copy
timings.

Suggested minimum checks:

1. open root group
2. read one `raw_video/images_full` frame
3. read one `crop_runs/*/roi_images` row
4. read one `subject_mask_runs/*/masks_roi` row

If a layout copies fast but reads badly, it should not be the preferred online
format.

## Expected Interpretation

The likely outcomes are:

- `.tar.zst` wins for pure transport
- sharded clone may win for browseable/read-mostly online storage
- raw working store remains best for active mutation

That is the result this benchmark is intended to confirm or reject.

## Representative Archive Results (2026-04-03)

Representative archive:

- `2026-01-28T22-15-03Z_arena_1_DefaultScreen_training.zarr`

Measured movement timings:

- raw `.zarr` directory copy to `/groups/...`: `136.74 s`
- prebuilt `.tar.zst` copy to `/groups/...`: `5.99 s`
- unpack on `/groups/...`: `117.75 s`
- sharded clone copy (`dense_readmostly_v1`): `116.03 s`
- sharded clone copy (`dense_readmostly_rechunk_v1`): `111.22 s`

Measured build/export timings:

- local transfer-artifact pack: `43.25 s`
- sharded clone export (`dense_readmostly_v1`): `135.65 s`
- sharded clone export (`dense_readmostly_rechunk_v1`): `466.03 s`

Measured destination open/read timings:

- raw
  - open root: `0.0098 s`
  - `raw_video/images_full` frame read: `1.1522 s`
  - latest crop ROI row read: `0.0340 s`
  - latest subject-mask row read: `0.0248 s`
- tar unpacked
  - open root: `0.0090 s`
  - `raw_video/images_full` frame read: `1.1873 s`
  - latest crop ROI row read: `0.0342 s`
  - latest subject-mask row read: `0.0290 s`
- sharded dense
  - open root: `0.0087 s`
  - `raw_video/images_full` frame read: `1.4438 s`
  - latest crop ROI row read: `0.0380 s`
  - latest subject-mask row read: `0.0333 s`
- sharded rechunk
  - open root: `0.0136 s`
  - `raw_video/images_full` frame read: `1.4601 s`
  - latest crop ROI row read: `0.0295 s`
  - latest subject-mask row read: `0.0201 s`

Interpretation:

- prebuilt `.tar.zst` is the best default for off-machine movement
- unpacked tar reads essentially like the raw directory on the destination
- sharded browseable exports are not a general read win on this representative
  archive
- archival rechunking improves subject-mask row reads, but the rewrite cost is
  high and the overall read profile still does not justify making sharded
  exports the default transport path

## Recommended Next Step

1. Implement `export_sharded_zarr_clone` as a narrow benchmark prototype.
2. Run the three-way benchmark on the representative archive.
3. Repeat on the worst file-count archive.
4. Decide whether sharded export earns a place beside the existing `.tar.zst`
   transfer artifact path.
