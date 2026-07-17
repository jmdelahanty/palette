<!-- ARCHIVED 2026-07-17: measured conclusions moved to zarr_storage_lifecycle_policy.md and diagnostics. -->

# Zarr on NFS: Audit + Design TODO

## Context (current assumptions)

- Filesystem: NFS.
- Workload: single node, write-once read-many after import.
- Archiving: rare (not a primary requirement yet).
- Sharding is optional (not required) while the pipeline is still evolving.

Related policy sketch:

- [zarr_storage_lifecycle_policy.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/zarr_storage_lifecycle_policy.md)

## Current behavior in this repo

- Zarr v3 `LocalStore` is the default for new imports.
- Standard import path always writes **sharded** `raw_video` arrays (frame shards).
- kvikIO/GDS import path uses `use_sharding`; default `false` means **chunk-only**.
- Derived arrays (crops, detections, keypoints, masks, etc.) are chunked, not sharded.
- Many call sites open Zarr directly instead of using `open_zarr_root`.
- No explicit consolidated-metadata finalization step is run after most writes.
- Direct `zarr.json` metadata is currently the correctness path for mutable
  local stores; consolidated metadata may be stale and readers need a direct
  metadata fallback.

## Current measured observations (2026-04-03)

Training-set scan under `/nvme1/recordings`:

- 52 training zarrs
- average archive size: about `4.8 GB`
- median archive size: about `5.4 GB`
- average file count: about `8.4k`
- median file count: `6.9k`
- max file count: `28,970`

Important finding:

- `raw_video` is already low-file-count in the sampled training stores
  (for example `5.0 GB` in `18` files),
- but derived groups can dominate file count
  (for example `refined_eye_masks_runs` at `216 MB` and `27,440` files).

Implication:

- movement/NFS pain is not just a raw-video sharding problem,
- it is also a derived-run retention/layout problem.

## Design goals for NFS

- Minimize metadata IOPS and inode count.
- Preserve good read performance for common access patterns (frame + ROI slices).
- Keep the pipeline UX simple and predictable.

## What I need to learn / measure

- NFS limits and behavior: metadata latency, inode caps, mount options.
- File counts per dataset: chunk files per array, per run, total.
- Access patterns in practice: how many chunks do visualizers read per action?
- Zarr v3 consolidated metadata support (if any) and its impact on open times.
- Sharding impact on read/write speed for `raw_video` and `crop_runs`.
- Transfer-artifact impact on off-machine copy speed vs raw directory copies.

## TODO: technical evaluation

1. Benchmark open + read on a representative dataset and log timings.
2. Count chunk files/inodes per dataset and per stage.
3. Test sharding on the kvikIO path for 4512x4512 frames.
4. Explore sharding for `crop_runs` (large, high-count arrays).
5. Decide on a default sharding policy for NFS (target shard size in MB).
6. Evaluate consolidated metadata support and add optional finalize step.
   - Add a shared `finalize_zarr_mutation`-style helper for writers.
   - Consolidate only after arrays, direct attrs, indexes, and parent `latest`
     attrs are complete.
   - Record consolidation freshness/status in provenance.
   - Ensure consolidation is never done by parallel workers writing chunks.
   - Add an audit command that reports missing/stale consolidated metadata but
     validates direct metadata as the authoritative mutable-store surface.
7. Route all open calls through `open_zarr_root` for future store changes.
8. Benchmark movement to external storage:
   - raw `.zarr` directory copy
   - packed transfer artifact copy
   - unpack time on destination
   - rewritten sharded clone copy
   - destination open/read timing for each layout
9. Record whether transfer artifacts should include all groups or omit
   regenerable compatibility artifacts by default.
10. Implement and benchmark a narrow sharded export clone prototype.
    - Runbook: `docs/zarr_transfer_benchmark_plan.md`
    - Utility:
      `scripts/py -m fisheye.utils.export_sharded_zarr_clone ...`
    - Include archival rechunk variant:
      `--policy dense_readmostly_rechunk_v1`
11. Validate chunk-policy choices empirically across the major workflow
    families instead of treating masks as the only storage concern.
    - Canonical priority:
      `subject_mask_runs`, `refined_subject_masks_runs`, `crop_runs`,
      `detect` / `refined_detect`, `keypoints_runs` / `refined_keypoints_runs`
    - Transitional compatibility priority only as needed:
      `eye_masks_runs`, `refined_eye_masks_runs`
    - Measure stage-specific behavior:
      hot-write latency, review/edit latency, file-count growth, transfer cost,
      and destination read/open behavior
    - Add explicit stage-family chunk helpers where ad hoc literals still exist
    - Avoid blindly inheriting upstream chunks into canonical refined outputs

## TODO: design changes (if needed)

- Align config semantics: either honor `use_sharding` everywhere or remove it and
  document always-sharded defaults.
- Record chunk/shard sizes for all large arrays in run metadata.
- Add an optional "pack/export" step for portability (`tar.zst` first; rewritten
  sharded export only if we prove it is worth the complexity).
- Keep sharded export scoped to read-mostly benchmark policies first; do not
  turn it into the default mutable working-store path.
- Allow archival export to rechunk dense read-mostly arrays when that is the
  only way to make sharding materially reduce file count.
- Document storage modes and NFS-recommended settings.
- Define keep-vs-regenerate policy for compatibility artifacts in finalized and
  transfer modes.
- Define canonical stage-family chunk contracts for crops, detections,
  keypoints, and unified subject masks.
- Add an audit/report path that flags arrays which violate the canonical chunk
  contract for their stage family.
- Define file-count warning thresholds for finalized online stores.
- Define consolidated-metadata freshness policy for finalized and transfer
  artifacts, while keeping direct metadata authoritative for mutable working
  stores.

## Success criteria

- Predictable open times and stable read throughput on NFS.
- File counts stay within NFS inode/metadata constraints.
- Storage layout is explicit and auditable in metadata.
- Off-machine movement is dominated by bulk bandwidth rather than tiny-file
  overhead.

## Checklist (short version)

- [ ] Capture baseline timings (open + read + crop access).
- [ ] Capture file counts per dataset (raw_video + derived runs).
- [ ] Verify whether GDS path can shard (or fails) and document outcome.
- [ ] Pick shard targets for raw_video and crops based on measurements.
- [ ] Decide how sharding is enabled (config flag vs auto policy).
- [x] Build a transfer-artifact prototype.
  - Utility: `scripts/py -m fisheye.utils.pack_zarr_transfer_artifact ...`
- [x] Benchmark raw-dir copy vs packed copy.
- [x] Benchmark raw-dir vs packed vs sharded-clone copy.
- [x] Benchmark `dense_readmostly_v1` vs `dense_readmostly_rechunk_v1`.
- [x] Implement `export_sharded_zarr_clone` benchmark prototype.
- [x] Run the representative-archive three-way benchmark from
  `docs/zarr_transfer_benchmark_plan.md`.
- [x] Capture destination open/read timings for all three layouts.
- [x] Add a destination open/read benchmark utility.
  - Utility:
    `scripts/py -m fisheye.utils.benchmark_zarr_destination_reads ...`
- [ ] Decide which compatibility/regenerable groups belong in transfer artifacts.
- [ ] Run a workflow-wide chunk-policy audit across canonical stage families.
- [ ] Decide the next stage family to standardize after unified subject masks.
- [ ] Add stage-family compliance checks for canonical chunk layouts.
- [ ] Document the final policy and update schema docs.
