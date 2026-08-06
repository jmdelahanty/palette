# Tabular Delta And Compaction Contract

**Status:** keypoint/general v1 partition prototype retained; its detection
payload is superseded for future work by
`refined_detection_delta_v2_contract.md`. The maintained Palette keypoint
reviewer now resolves and appends verified keypoint-v1 partitions. Crimson
overlay reads and keypoint compaction/scheduling remain follow-up integration
work.

The v1 detection payload below is historical implementation context. It cannot
represent a complete manual detection row and must not be extended into the v2
contract. New detection work uses the exact detection-specific v2 schema and
resolver; keypoint v1 behavior is unchanged.

## Decision

Raw and canonical refined keypoint/detection runs are immutable snapshots.
They use Zarr v3 indexed sharding with `131072` requested outer rows while
retaining bounded inner chunks. Manual and other genuinely sparse corrections
are written to partition-owned delta generations instead of modifying those
shards in place.

This separates three concerns:

1. inference and refinement produce complete evidence or canonical snapshots;
2. interactive review writes small sparse deltas;
3. a maintenance compactor periodically publishes another complete immutable
snapshot.

`262144` remains an explicit option for strictly immutable publication where
the additional object-count reduction is worth doubling a shard rewrite unit.

Automated refinement that changes a substantial fraction of rows should write
a complete snapshot directly. It should not encode a dense rewrite as millions
of nominally sparse delta rows.

## Stable Target Identity

`instance_key` is the durable edit target. `row_index_hint` is only an
acceleration hint and must resolve to the same key in the bound base run.
Compaction and readers must reject a hint/key mismatch.

New clipped detections mint keys from:

- canonical recording identity from the analysis-Zarr root;
- recording-level `parent_frame_index` from
  `recording_frame_index.parquet`;
- class and quantized bounding-box content;
- the existing duplicate-ordinal policy for exact same-frame duplicates.

The raw detection run records the frame domain, frame-index manifest path, and
SHA-256 of the dense clip-local-to-parent mapping. Clip-local `frame_indices`
remain available for clip-local processing, but are not the frame input to the
stable key hash.

The clipped flat ROI-cache row-index schema is
`palette_clipped_collection_flat_roi_cache_rows_v2`. It requires
`instance_key`, and the per-clip proxy, merged proxy, raw keypoint collection,
and refined keypoint snapshot must preserve it exactly.

Historical clipped shells may contain keys minted with the scratch-store
identity (`detect_output`) and clip-local frames. The
`fisheye.utils.backfill_legacy_instance_keys` repair resolves each clip through
the registered `recording_frame_index.parquet`, verifies any existing key array
against that exact legacy recipe, and then plans recording-global replacement
keys. It bridges historical proxies through
`source_refined_row_ids -> instances/refined_row_ids -> instance_key`; stable
row IDs are lookup keys, never dense positions. Replacement is permitted only
for a byte-for-byte verified legacy payload. New proxy writers carry
`instance_key` directly and do not need this compatibility join.

## Storage Hierarchy

```text
edit_delta_runs/<delta_run>/
  attrs:
    schema = palette.tabular_delta_run.v1
    target_kind = keypoints | detections
    base_run_path
    base_instance_key_count
    base_instance_key_sha256
    active_generation

  generations/<generation>/
    attrs:
      schema = palette.tabular_delta_generation.v1
      generation_ordinal
      status = open | frozen | compacted

    partitions/<worker-or-batch-id>/
      attrs:
        schema = palette.tabular_delta_partition.v1
        editor
        partition_sha256
        operation_code_map
        reason_code_map

      instance_key       uint64[N]
      row_index_hint     int64[N]
      operation_codes    uint8[N]
      revision           uint64[N]
      timestamp_ns       int64[N]
      reason_codes       uint16[N]
```

Keypoint partitions additionally contain:

```text
keypoint_index  int16[N]
new_xy          float64[N,2]
valid           bool[N]
```

Detection partitions additionally contain:

```text
new_bbox_norm_coords  float64[N,4]
valid                 bool[N]
```

Operation codes are versioned by
`palette.tabular_delta_operation_codes.v1`. The current maps live in
`fisheye.shared.tabular_deltas` and are copied into partition attributes.
Free-form reason text should remain exceptional; ordinary reasons use a
versioned numeric map to keep partitions compact.

Complete base and compacted runs persist explanatory row labels only as
fixed-width `reason_bytes`. The variable-length `reason` array is a historical
read fallback: delta writers must not target it, and snapshot publishers and
compactors omit it. A legacy base containing only `reason` must first be
deterministically canonicalized to `reason_bytes`; publication fails closed
rather than silently dropping the labels.

Each partition is immutable and owned by one writer. Writers must never append
concurrently to a shared Zarr array. A partition uses one ordinary chunk per
column because it is expected to be small; sharding sparse edit partitions
would add complexity without meaningful file-count savings.

## Read Resolution

A delta-aware reader resolves a displayed field by:

1. opening the immutable base selected by the delta run binding;
2. verifying `base_instance_key_sha256`;
3. loading partition indexes for the requested generation set;
4. resolving edits by `instance_key` and field identity;
5. selecting the newest applicable edit by
   `(revision, timestamp_ns, partition, partition_row_index)`;
6. falling back to the base value when no applicable delta exists.

Crimson already preloads selected tabular keypoint/detection arrays. It should
also preload the small resolved delta index. Masks and other large pixel
surfaces remain lazy.

## Generation Freeze

Compaction needs a fixed input. The scheduler freezes generation `G`, records
its sorted partition names and aggregate digest, and immediately opens
generation `G+1` for new edits. Frozen generations reject new partitions.

This guarantees that edits arriving while compaction runs are not lost and do
not make the compactor's input move underneath it.

## Future Compaction Job

Yes: compaction should be an explicit LSF maintenance job, not a login-node
process and not an implicit side effect of a browser save.

The job will:

1. acquire a recording/run maintenance lock;
2. freeze the active generation and open its successor;
3. verify the base run, key digest, partition digests, operation schema, and
   deterministic merge order;
4. assign workers whole, non-overlapping output storage shards;
5. stream one base shard at a time, apply applicable deltas, and write one new
   shard once;
6. validate decoded equality for untouched rows and exact expected values for
   edited rows;
7. write provenance binding the base and frozen generation digest;
8. mark the new run complete and atomically promote `latest` and
   `latest_complete`;
9. mark the frozen generation compacted with the resulting run path while
   retaining every partition for audit;
10. refresh registry projections after publication.

The compactor must never rewrite or delete the current approved base. Failure
leaves the old pointer authoritative and the frozen deltas readable.

Compaction should be triggered by policy rather than only a wall-clock timer,
for example:

- delta row density exceeds a configured fraction of the base;
- partition count or overlay-open latency exceeds a budget;
- a review milestone is approved;
- an operator requests archival/final publication.

## Current Implementation Surfaces

- `fisheye.utils.publish_tabular_snapshot` clones a completed keypoint or
  refined-detection run into an exact immutable sharded sibling. It is dry-run
  by default and promotes only after decoded SHA-256 validation.
- `fisheye.shared.tabular_deltas` creates base-bound generations, writes
  immutable keypoint/detection partitions, validates key/hint identity, and
  freezes generations. Its keypoint resolver additionally validates the exact
  partition envelope, recomputes payload digests, enforces landmark operation
  semantics, and applies the frozen deterministic merge order.
- `fisheye.tune.keypoint_review_backend` treats an
  `artifact_mutability=immutable_snapshot` run as read-only, requires one bound
  open keypoint generation, renders base plus the verified overlay, and writes
  one immutable partition per accepted review action. It refuses in-place
  approval until compaction publishes a successor snapshot. The existing
  direct writer remains only for non-immutable compatibility runs.
- `fisheye.shared.zarr.refined_keypoint_manifest` now validates exact
  successor ancestry: recording and lineage identity, a new snapshot UUID,
  immediate-parent run/manifest/snapshot binding, unchanged instance-key
  order, and unchanged retired-key evidence. This removes the former
  manifest-gate blocker to a keypoint compactor; live-generation rollover and
  final reviewed-publication orchestration are still pending.
- `fisheye.utils.compact_refined_keypoint_deltas_v2` consumes one already
  frozen, digest-verified keypoint generation, regenerates every dependent
  refined array, and publishes a selector-ineligible sharded successor plus a
  sidecar receipt. It deliberately does not freeze a live generation, mutate
  the source archive, import the output, or activate selectors. Generation
  rollover and reviewed publication orchestration remain pending.
  Manual coordinate replacements receive confidence `1.0`; cleared landmarks
  receive `NaN`. The current selector-ineligible compactor treats a complete
  finite manually reviewed pose as confidence- and geometry-valid. Production
  activation remains blocked until the reviewed-publication gate binds and
  replays the exact review-QC policy rather than relying on that provisional
  manual-acceptance rule.
- `fisheye.utils.publish_reviewed_training_artifact_candidate` is the combined
  training-artifact boundary. It snapshots the mutable review package, imports
  the receipt-bound compacted keypoint run, seals approved dense subject masks
  in the copy, consolidates the completed immutable artifact, and atomically
  publishes it selector-ineligible. The active review package remains mutable
  and unchanged; pending mask review, stale receipts, non-frozen generations,
  or concurrent source writes fail closed.
- `fisheye.utils.finalize_keypoint_shards` now publishes canonical clipped
  keypoints directly as indexed shards.
- `fisheye.utils.publish_clipped_refined_detect_snapshot` materializes one
  recording-level refined-detection snapshot from a finalized clip collection,
  retaining both parent-video and clip-local frame/row lineage. The detailed
  contract is in `docs/clipped_refined_detection_snapshot_contract.md`.
- `fisheye.utils.backfill_refined_subject_mask_instance_keys` performs an
  additive mask-key repair only after exact blockwise equality across the ten
  shared keypoint/mask lineage arrays.

Crimson still needs its own overlay reader, and the keypoint compactor remains
required before review approval. Existing in-place review writers remain
compatibility paths and must not be used to edit an
`artifact_mutability=immutable_snapshot` run.

## Sleepyfish Production Snapshot Canary (2026-07-15)

The full Cam2010095 clipped collection was used as the first production
identity and immutable-snapshot canary. No inference or refinement was rerun.

- The historical analysis shell lacked the newer root
  `analysis_layout=clipped_recording_shell` marker but had a valid finalized
  refined-detection collection pointer. The identity repair now treats that
  pointer as an unambiguous clipped-lineage contract instead of falling into
  unrelated legacy root runs.
- Deterministic `instance_key` lineage repair wrote 135 arrays across all 22
  raw/refined detection lineages, crop proxies, the merged crop, raw
  keypoints, and refined keypoints. Post-write validation checked all 135
  arrays with zero mismatches.
- The selected refined-keypoint rowset contains 1,169,010 unique observation
  rows and records
  `instance_key_policy=copied_from_merged_proxy_crop_rows`.
- The immutable snapshot
  `refined_keypoints_sleepyfish_kp_allclips_sharded_20260715_01` copied 46
  canonical arrays. Every decoded source/destination SHA-256 matched. The
  legacy `reason` mirror was omitted and `reason_bytes` remained authoritative.
- Publication completed on one LSF compute-node slot in 49 seconds with 242 MB
  peak RSS. The run was streamed by outer shard; the full table was not loaded
  into memory.
- Physical files in the selected refined-keypoint run fell from 28,277 to 365
  (about 77x fewer). `keypoints_img` retains 1,024-row inner chunks and uses
  131,072-row outer shards; the backfilled identity array retains 16,384-row
  inner chunks on the same outer-shard grid.
- Completion, exact-validation metadata, `latest`, and `latest_complete` all
  select the new snapshot. The shared registry refresh completed from the same
  canonical Zarr after taking a SQLite backup.

The original selected refined detections remain a finalized 22-clip collection
and immutable source evidence. A recording-level snapshot publisher was added
after this first canary to expose the same collection as one validated,
indexed-sharded consumer table without deleting or rewriting those source
runs. Its production application is tracked separately from this completed
keypoint canary.

`immutable_snapshot` is an application contract, not a filesystem write lock.
Low-level Zarr or PRFS writes remain physically possible for an account with
permission. Supported review writers must refuse in-place changes and route
them to `edit_delta_runs`. No synthetic scientific correction was added to the
production recording merely to exercise that path. A real production delta
canary remains blocked on the Crimson overlay reader, review-writer routing,
and parity-tested derived-metric refresh during compaction.
