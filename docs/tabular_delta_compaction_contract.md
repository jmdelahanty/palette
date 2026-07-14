# Tabular Delta And Compaction Contract

**Status:** sparse delta storage and immutable snapshot publication implemented;
Crimson overlay reads, review-writer routing, and compaction scheduling remain
follow-up integration work.

## Decision

Raw and canonical refined keypoint/detection runs are immutable snapshots.
They use Zarr v3 indexed sharding with `262144` requested outer rows while
retaining bounded inner chunks. Manual and other genuinely sparse corrections
are written to partition-owned delta generations instead of modifying those
shards in place.

This separates three concerns:

1. inference and refinement produce complete evidence or canonical snapshots;
2. interactive review writes small sparse deltas;
3. a maintenance compactor periodically publishes another complete immutable
   snapshot.

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
  freezes generations.
- `fisheye.utils.finalize_keypoint_shards` now publishes canonical clipped
  keypoints directly as indexed shards.

Until Crimson and the review backends are routed through this layer, existing
in-place review writers remain compatibility paths and must not be used to edit
an `artifact_mutability=immutable_snapshot` run.
