# Shard training and review archive runs

- **Status:** draft
- **Owner:** labeling/Apply work (session palette-12, for the user); last reviewed 2026-09-25
- **Builds on:**
  - `docs/archive/zarr_sharding_design_note.md` (policy: shard when file
    counts become large, measure first);
  - the shared planner `src/fisheye/shared/zarr/storage_planner.py`
    (`StorageProfile`, `ArrayIntent`, `_should_shard`) and
    `src/fisheye/shared/zarr/columnar.py` (`pick_shards`);
  - production sharding already in use: detection
    (`immutable_yolo_storage`, `detection_storage`), keypoints
    (`keypoint_publication_profile`), crops (`crop_storage`), and
    kinematics and subject shape (`analysis_workflows/materializers`), with
    benchmarks under `docs/archive/*sharding*`.
- **Related:** `docs/design/2026-09-25-tail-successor-crop-reference/README.md`

## Problem

Training and review archives (`*_recovered_training.zarr` and similar) are
written chunk-only. Their runs are small, but each is split into many files:

- `keypoints_roi` uses one chunk per row, so a 229-row run has 229 chunk
  files.
- Every per-row array is a separate file with its own `zarr.json`.

One mask Apply's five successor runs total 1,039 files for about 26 MB, and
the arena 2 archive holds 4,376 files for about 196 MB.

On `/groups` NFS, cost tracks file count (measured 2026-09-24 and 09-25):

- publishing about 1,039 files takes 60 s;
- one source-fingerprint check stats 1,039 files in 3.5 s, and runs about 15
  times per Apply;
- inventory and hash passes re-read every file;
- concurrent I/O did not help on this mount (114 s to 108 s).

Keypoint Apply showed the same pattern: a 98-row batch Apply took 84 s on NFS
and 11 s locally, because of per-file operations.

## Design

Use the existing storage planner with a **training/review profile**, not a
new sharding helper:

1. **Whole-run shards for small runs.** For review and training runs (at most
   tens of thousands of rows), each array gets a single shard covering the
   full row extent. The inner chunk grid keeps today's chunk shapes, including
   one-row inner chunks for `keypoints_roi`, so reads stay row-granular. A
   229-row run's `keypoints_roi` becomes 1 file instead of 229.
2. **Large arrays keep size-bounded shards** from the planner's existing byte
   targets. Examples are `masks_roi` and crop `roi_images` (when crops are
   still materialized).
3. **Mutable review surfaces:** an edit rewrites the containing shard. For
   small runs that is a few KB to MB per write, and batch Apply already writes
   whole chunks. Shard layout must respect the Dask/parallel-write rule: only
   one writer owns a shard.
4. **Recorded in metadata:** requested and effective shard shapes are
   recorded, as the planner already does for production runs.

## Adoption

1. **New successor versions and newly built review archives only.** They
   ride the next policy version, alongside the crop-reference change.
   Existing runs are not rewritten.
2. **Every reader stays layout-agnostic,** because Zarr sharding is
   transparent to readers. Add a conformance test that reads each review
   family from sharded and chunk-only fixtures and compares values.
3. **Measure on `/groups`** with the same replay harness before and after:
   files per Apply, publish time and total Apply time.
4. **Optional later:** an explicit, authorized repack of existing review
   archives, written as new runs. Not part of this design.

## Risks

- **Write amplification on mutable edits** (rewriting a whole shard). Measure
  it on the largest mutable review run before adopting there.
- **Tools that walk the filesystem by chunk path** (such as the tail refresh's
  fingerprint and physical inventories) see fewer, larger files. They stay
  correct (atomic shard rewrites still change inode and mtime), and they get
  much faster.
- **The Zarr v3 sharding codec's partial-read performance** on NFS. Production
  benchmarks already cover detection and keypoints; add one for review-size
  runs.

## Expected effect (estimate until measured)

One mask Apply's successor goes from about 1,039 files to about 50 or fewer,
with about 830 files once crop referencing lands. The fingerprint, inventory,
copy and validation passes shrink roughly in proportion to file count. That
should take publish from about 50–60 s to a few seconds on `/groups`.

## Decisions needed

1. Adopt a training/review storage profile in the shared planner, with
   whole-run shards for small runs?
2. Apply it only to new runs and versions (proposed), leaving existing runs
   as they are?

## Decision log

- 2026-09-25: opened. The user agreed to shard, noting that production already
  uses sharding widely.
