# Zarr Storage Lifecycle Policy

## Purpose

Define how Palette Zarr stores should behave across three distinct phases:

- active processing on local scratch
- finalized online storage for inspection and downstream reads
- transfer/archive movement off-machine

The goal is to avoid forcing one storage layout to serve all three jobs.

## Why This Policy Exists

Sharding helps when the problem is "too many tiny files are slow to move or
slow to serve over NFS." It does **not** automatically make the live mutable
editing path better:

- chunking keeps partial writes and parallel writes simple
- sharding reduces file count by packing many chunks into fewer files
- updating one chunk inside a shard usually means rewriting the shard payload

That means sharding is a good fit for large immutable or read-mostly arrays, but
it is not the default answer for hot refinement/review outputs.

## Current Measurements (2026-04-03)

Measured on the current `/nvme1/recordings` training set:

- 52 training zarrs scanned
- average archive size: about `4.8 GB`
- median archive size: about `5.4 GB`
- average file count: about `8.4k`
- median file count: `6.9k`
- max file count: `28,970`

Representative findings:

- `/nvme1/recordings/2026-01-28T22-15-03Z_arena_1_DefaultScreen/..._training.zarr`
  - total: `6.0 GB`, `10,131` files
  - `raw_video`: `5.0 GB`, `18` files
  - `subject_mask_runs`: `884 MB`, `2,038` files
  - `refined_eye_masks_runs`: `46 MB`, `5,773` files
- `/nvme1/recordings/2026-01-28T20-41-59Z_arena_1_DefaultScreen/..._training.zarr`
  - total: `5.6 GB`, `28,970` files
  - `raw_video`: `5.0 GB`, `18` files
  - `refined_eye_masks_runs`: `216 MB`, `27,440` files

Conclusion:

- `raw_video` size dominates bytes
- derived run outputs dominate file count
- movement pain is already more about metadata/file-open overhead than bulk
  bandwidth

## Storage Modes

### 1. Working Scratch Store

This is the canonical live store during import, inference, refinement, review,
and tuning.

Properties:

- local NVMe / scratch
- optimized for chunked parallel writes
- may contain in-progress and experimental runs
- disposable after export + verification

Policy:

- keep actively written arrays chunked
- allow sharding only where writes are effectively write-once and bulk
- do not optimize this mode primarily for portability

### 2. Finalized Online Store

This is the browseable/read-mostly store that remains online after a run is
considered stable enough for normal use.

Properties:

- should keep canonical lineage and canonical outputs
- should avoid retaining every exploratory or compatibility artifact forever
- may compact selected immutable arrays more aggressively than the scratch store

Policy:

- keep canonical outputs
- prefer one promoted/canonical run per stage family where practical
- keep review-authority outputs chunked unless they are truly frozen
- omit temporary caches and regenerable artifacts by default

### 3. Transfer/Archive Artifact

This is the thing moved to external drives or nearline storage.

Properties:

- optimized for movement and verification
- not edited in place
- should minimize per-file overhead

Policy:

- first implementation should be a packed artifact, likely `tar.zst`
- keep checksum + manifest alongside the artifact
- do not require the live working store itself to be portable

## Group Policy

### Immutable Bulk Arrays

Examples:

- `raw_video/images_full`
- `raw_video/images_ds`

Policy:

- shard by default when the import path supports it
- treat as immutable after import
- good candidates for more aggressive compaction in finalized/export modes

### Hot Dense Derived Arrays

Examples:

- `crop_runs/*/roi_images`
- `subject_mask_runs/*/masks_roi`
- `subject_mask_runs/*/mask_probs_roi`
- other dense stage outputs that are still being iterated on

Policy:

- chunk while active
- choose chunking for the dominant runtime access pattern, not for portability
- revisit spatial tiling if it multiplies file count without helping actual
  edits or reads

### Review / Manual-Authority Outputs

Examples:

- `refined_subject_masks_runs`
- refined keypoint/detect review outputs

Policy:

- keep chunked while edits are still expected
- do not make sharded mutation the normal operator path
- compact only once the artifact is genuinely finalized

### Compatibility / Regenerable Outputs

Examples:

- `refined_eye_masks_runs` when derived from canonical
  `refined_subject_masks_runs`
- other compatibility projections or cache-like materializations

Policy:

- keep only when needed for active compatibility
- exclude from finalized/export artifacts by default once regeneration is
  reliable and documented

### Temporary Runtime Caches

Examples:

- external ROI caches
- other derived performance caches

Policy:

- keep outside canonical archives by default
- treat as disposable

## Large Analysis Archive Guidance

For future large analysis archives (for example `100k` frames), do not assume
that one archive should permanently hold:

- raw imported imagery
- every historical inference run
- every compatibility artifact
- every manual refinement variant

That will scale poorly even if some arrays are sharded.

Current example:

- `subject_mask_runs/*/masks_roi` is currently chunked like
  `[64, 1, 128, 128]` over `[T, 3, 512, 512]`
- at `T = 100,000`, that is about `75,024` chunks for `masks_roi` alone

Implications:

- file budget needs to be treated as a first-class design constraint
- historical runs should not accumulate forever in the canonical online store
- dense outputs may need different canonical chunk shapes than today
- small scalar metrics should eventually be audited for packing/aggregation

## Provisional File-Count Budgets

These are operational warning thresholds, not hard guarantees:

- target finalized online stores to stay below about `10k` files when practical
- investigate any single group that contributes more than about `5k` files
- treat stores above about `20k` files as portability/NFS-risk candidates

These thresholds come from the current measured movement pain and should be
revisited after a transfer-artifact benchmark exists.

## Recommended Lifecycle

1. Write pipeline outputs to a chunked scratch store on local NVMe.
2. Run refinement/review against that working store.
3. Promote only canonical outputs into the finalized online profile.
4. Create a transfer artifact for off-machine movement.
5. Verify checksum/manifest before deleting scratch copies.

## Recommended Near-Term Implementation Order

1. Add a transfer-artifact tool (`tar.zst` first).
2. Benchmark directory copy vs packed-artifact copy to external storage.
3. Define keep-vs-regenerate rules for compatibility artifacts.
4. Audit chunk layouts for the largest dense derived arrays.
5. Revisit selective sharding only after those measurements.

## Non-Goals

This policy does **not** say:

- every array should be sharded
- live refinement should happen against packed artifacts
- every existing archive must be rewritten immediately

It does say:

- working, finalized, and transfer storage should be treated as different
  products
- file count must be designed explicitly, not left to emerge accidentally
