# Zarr Storage Lifecycle Policy

## Purpose

Define how Palette Zarr stores should behave across three distinct phases:

- active processing on local scratch
- finalized online storage for inspection and downstream reads
- transfer/archive movement off-machine

The goal is to avoid forcing one storage layout to serve all three jobs.

## Why This Policy Exists

Palette deliberately uses a hybrid storage model rather than treating one file
format as the answer to every layer:

- Citrus/Orange acquisition snapshots may remain H5/HDF5 where the acquisition
  stack already emits them.
- Palette recording analysis archives remain Zarr because they contain chunked,
  heterogeneous arrays that need partial reads by frame, ROI, track, channel,
  and time window.
- Cross-recording analytics should use Parquet/DuckDB exports because those
  questions are table-shaped and query-oriented.
- Cluster transfer and storage-tier movement should use packed artifacts or
  run-group packages when many small Zarr files would be inefficient to move.

HDF5 would make single-file transfer and whole-archive checksums simpler, but
it would push Palette toward single-writer bottlenecks and coarse-grained
mutation exactly where recording-level distributed processing and run-family
imports matter. Zarr remains the right canonical analysis store, provided the
repository controls metadata fanout, chunk/shard policy, and mutation
lifecycle.

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

Representative off-machine benchmark conclusion on `/groups/...`:

- raw directory copy: `136.74 s`
- prebuilt `.tar.zst` copy: `5.99 s`
- unpacked tar reads are close to raw for representative open/frame/ROI/mask
  reads
- sharded browseable exports help only modestly for transfer on the
  representative archive
- archival rechunk improves unified subject-mask reads, but not enough to
  outweigh export cost for general movement

Policy implication:

- use `tar.zst` as the default off-machine transfer format
- treat sharded browseable exports as a niche read-mostly option, not the
  default transport path

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
- do not treat them as the primary long-term chunk-policy target once canonical
  unified subject-mask stores are in place

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

Historical example:

- some `subject_mask_runs/*/masks_roi` arrays were chunked like
  `[64, 1, 128, 128]` over `[T, 3, 512, 512]`
- at `T = 100,000`, that is about `75,024` chunks for `masks_roi` alone

Current raw subject-mask write policy:

- dense `subject_mask_runs/*/masks_roi` and `mask_probs_roi` should use
  full-ROI spatial chunks with modest row depth:
  `[min(16, T), 1, H, W]`
- for a `512x512` ROI archive at `T = 100,000`, that is about `18,750`
  chunks for one dense mask array

Implications:

- file budget needs to be treated as a first-class design constraint
- historical runs should not accumulate forever in the canonical online store
- dense outputs need canonical chunk shapes chosen for ROI-level access,
  not tile-level portability
- small scalar metrics should eventually be audited for packing/aggregation

For analysis-run schema direction specifically, see
`docs/analysis_zarr_object_count_schema_direction.md`. That note covers the
schema-level object-count issue: parameter-sweep fanout, representation/alias
materialization, and component-per-group metric mirrors. Those patterns can
create many `zarr.json` metadata objects even when chunking/sharding choices are
otherwise reasonable.

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

## Cluster / Network Storage Workflow

For cluster jobs, avoid streaming many small Zarr chunk writes directly over
network storage. The preferred pattern is to write complete derived run groups
on node-local scratch, pack them, transfer the packed artifact, and promote the
complete run group near the destination archive.

Recommended flow:

1. Read or stage required inputs onto node-local NVMe/SSD scratch when
   practical.
2. Compute outputs into a local scratch Zarr run group.
3. Validate the local run group before transfer.
4. Pack the run group as a transfer artifact, preferably `tar.zst`, with a
   manifest and checksums.
5. Transfer the packed artifact over the network.
6. Unpack into a staging path on the destination side, near the canonical Zarr.
7. Validate the unpacked staging run group.
8. Promote the complete run group into the destination hierarchy with an atomic
   rename when the filesystem supports it.
9. Update small metadata surfaces last, such as `latest` attrs and consolidated
   metadata.

Do not update `latest` attrs, consolidated metadata, or other reader-visible
selection metadata until the destination run group is fully present and
validated. Avoid multiple jobs promoting into the same destination archive at
the same time unless a writer lock or equivalent coordination mechanism exists.

This pattern avoids the worst case of many small random chunk writes over NFS or
other shared storage. It also keeps mutable Dask writes on local scratch, where
they can follow the chunk-ownership safety rules in
`docs/dask_zarr_write_safety.md`.

Prefer this:

```text
local scratch Zarr run -> validate -> tar.zst -> transfer -> stage -> validate -> atomic promote
```

Avoid this for bulk outputs:

```text
many workers inserting individual chunks directly into the home Zarr over the network
```

## Deferred Consolidated-Metadata Policy

Direct child metadata remains the correctness baseline for mutable Palette
stores. Readers that need correctness against actively edited local stores must
be able to discover groups and attrs from direct `zarr.json` metadata, or by
opening with consolidated metadata disabled. Consolidated metadata is a
performance and portability surface, not the only source of truth.

The preferred writer policy is to refresh consolidated metadata at stable
single-writer finalization boundaries, after all arrays, groups, direct attrs,
indexes, and parent `latest` attrs have been written and validated.
Consolidation is available through the shared
`fisheye.shared.zarr_helpers.reconsolidate_zarr_metadata()` helper and the
operator CLI:

```bash
scripts/py -m fisheye.utils.reconsolidate_zarr_metadata /path/to/archive.zarr
scripts/py -m fisheye.utils.reconsolidate_zarr_metadata /path/to/archive.zarr \
  --group-path detect_runs/detect_001/quality_reports
```

Helper behavior:

- write direct metadata first and make direct readers correct before
  consolidation runs;
- consolidate only after a complete run group is present and selected metadata
  such as `latest` has been updated;
- record consolidation provenance such as `metadata_consolidation_policy`,
  `metadata_consolidated_at_utc`, `metadata_consolidation_status`, and any
  warning/error text;
- treat consolidation failure as a warning when direct metadata is valid, not as
  a reason to roll back otherwise valid analysis data;
- keep external consumers on a fallback path that can read direct metadata when
  consolidated metadata is stale or absent.

Do not run consolidation from parallel workers that share an archive. For
clipped workflows, per-clip workers should leave consolidated metadata alone and
the recording-level finalizer should refresh it once shared writes are complete.
The near-term rule remains: do not trust consolidated metadata for correctness
on mutable analysis stores, but make finalized stores fresh when a single
finalization step can safely do so.

## Recommended Near-Term Implementation Order

1. Add a transfer-artifact tool (`tar.zst` first).
   - Prototype utility:
     `scripts/py -m fisheye.utils.pack_zarr_transfer_artifact <archive>.zarr --apply`
2. Benchmark directory copy vs packed-artifact copy to external storage.
3. Define keep-vs-regenerate rules for compatibility artifacts.
4. Audit chunk layouts for the largest dense derived arrays.

## Next Chunk-Policy Audit

The next chunk-policy pass should validate chunk choices empirically across the
major workflow families instead of only fixing one storage hotspot at a time.

Priority order:

1. Canonical unified mask stores.
   - `subject_mask_runs`
   - `refined_subject_masks_runs`
2. Dense image-like supporting stages.
   - `crop_runs`
3. Row-oriented structured stages.
   - `detect` / `refined_detect`
   - `keypoints_runs` / `refined_keypoints_runs`
4. Transitional compatibility stores only as needed for migration safety.
   - `eye_masks_runs`
   - `refined_eye_masks_runs`

Expected evaluation style by family:

- `subject_mask_runs` / `refined_subject_masks_runs`
  - validate ROI-level read/write/edit latency
  - validate file-count growth on representative training and large analysis
    archives
  - validate transfer/export behavior for canonical runs
- `crop_runs`
  - validate ROI read patterns and file-count contribution
  - decide whether scratch and finalized/export modes should diverge
- `detect` / `refined_detect`
  - audit array fanout and row-chunk depth
  - do not assume chunking alone is the main lever if schema fanout dominates
- `keypoints_runs` / `refined_keypoints_runs`
  - audit row chunk depth, run fanout, and retention policy
  - prefer simple row-oriented chunk contracts over mask-like tuning

Guardrails:

- do not blindly inherit upstream chunks into canonical refined outputs
- prefer one explicit helper per stage family over scattered literal chunk
  tuples
- do not overinvest in long-term chunk optimization for eye-only compatibility
  stores if canonical unified subject-mask data is the future
- keep transfer-only optimizations separate from the mutable working-store path
5. Use the transfer benchmark runbook in
   `docs/zarr_transfer_benchmark_plan.md` to compare raw vs packed vs sharded
   export layouts.
   - Prototype utility:
     `scripts/py -m fisheye.utils.export_sharded_zarr_clone <source>.zarr --dest <dest>.zarr --policy <policy> --apply`
   - Include the archival rechunk benchmark path:
     `--policy dense_readmostly_rechunk_v1`
6. Revisit selective sharding only after those measurements.

## Non-Goals

This policy does **not** say:

- every array should be sharded
- live refinement should happen against packed artifacts
- every existing archive must be rewritten immediately

It does say:

- working, finalized, and transfer storage should be treated as different
  products
- file count must be designed explicitly, not left to emerge accidentally
