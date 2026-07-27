# Refined Detection Delta v2 Contract

Date: 2026-07-27

Status: frozen Palette logical and persisted-partition contract with executable
validation, resolution, immutable writes, and frozen-generation reads;
compactor, consumer overlay, and production routing pending

## Decision

Manual additions and interactive corrections do not mutate an immutable raw or
refined-detection snapshot. They are immutable, single-writer delta partitions
bound to one exact refined-v1 base. A deterministic resolver overlays frozen
partitions and produces the complete logical inputs for a new immutable
refined-v1 snapshot.

```text
immutable refined-v1 base
  + immutable ordered delta-v2 partitions
  -> deterministic resolved logical state
  -> validated immutable refined-v1 successor
```

The executable logical contract is
`src/fisheye/shared/zarr/refined_detection_delta.py`. The isolated persistence
boundary is `refined_detection_delta_storage.py`. Neither changes selectors or
registries.

## Scope

The first profile targets full-acquisition refined authorities, including a
clip-local authority whose frame domain is its own complete video. A
recording-level clipped aggregate remains derived: edits are committed to its
bound clip-local authority and the aggregate is rebuilt from the finalized
source collection. Delta v2 does not attempt to inject incomplete clip/media
lineage into the recording aggregate.

Raw `detect_runs` remain immutable evidence. A missing detection is an
`add_instance` event against refined state, never an append to raw detections.
Zero, one, or many active instances may resolve in any frame.

## Persisted Envelope

Each partition binds:

| Field | Contract |
| --- | --- |
| `schema_id` | Exact `palette.refined_detection.delta`. |
| `schema_version` | Exact integer `2`. |
| `delta_lineage_id` | Canonical UUID shared by every generation over one exact base. |
| `base_snapshot_id` | Canonical UUID from the bound refined-v1 manifest. |
| `base_manifest_digest` | Exact lowercase SHA-256 of the bound manifest payload. |
| `generation_ordinal` | Nonnegative integer; event order cannot move backward across generations. |
| `partition_id` | Path-safe immutable single-writer partition identity. |
| `actor_id` | Nonempty author/service identity. |
| `reason_code_map` | Exact partition-local `uint16 -> lowercase_snake_case` registry; code zero is `none`. |

The stable event identity is `(delta_lineage_id, event_sequence)`. An authoring
service must allocate `event_sequence` globally and monotonically within the
lineage. It should allocate new `refined_row_id` values at the same commit
boundary. Partition names and timestamps are provenance only and never break
ordering ties.

## Exact Event Arrays

Every partition contains exactly these arrays with common first dimension
`D = n_events`:

| Array | Shape | Dtype | Meaning |
| --- | ---: | --- | --- |
| `event_sequence` | `[D]` | `uint64` | Positive globally unique total order. |
| `expected_previous_event_sequence` | `[D]` | `uint64` | Optimistic predecessor for this instance; zero means unedited base/absent add target. |
| `operation_codes` | `[D]` | `uint8` | Exact operation registry below. |
| `instance_key` | `[D]` | `uint64` | Durable observation/edit identity. |
| `refined_row_ids` | `[D]` | `int64` | Stable, non-reused refined-lineage identity. |
| `row_index_hint` | `[D]` | `int64` | Exact base position or `-1`; never authoritative. |
| `timestamp_ns` | `[D]` | `int64` | Nonnegative provenance time, excluded from merge order. |
| `reason_codes` | `[D]` | `uint16` | Code in the partition-local registry. |
| `payload_valid` | `[D]` | `bool` | True exactly for add and replace. |
| `frame_indices` | `[D]` | `int32` | Complete post-operation frame, or `-1` sentinel. |
| `source_acquisition_frame_index` | `[D]` | `int64` | Sealed frame identity, or `-1` sentinel. |
| `bbox_norm_coords` | `[D,4]` | `float32` | Complete authoritative box, or all-zero sentinel. |
| `scores` | `[D]` | `float32` | Complete score, or exact zero sentinel. |
| `score_valid` | `[D]` | `bool` | Complete score validity, or false sentinel. |
| `class_ids` | `[D]` | `int32` | Complete class, or `-1` sentinel. |
| `source_kind_codes` | `[D]` | `uint8` | Complete source kind, or zero sentinel. |
| `manual_edit_flags` | `[D]` | `bool` | True for add/replace, false sentinel otherwise. |
| `source_detect_row_index` | `[D]` | `int64` | Raw-audit lineage or exact `-1`. |

All columns use exact dtypes; readers and writers must not probe alternatives.
Sparse partitions use one complete ordinary chunk per array and no shards.
They remain unconsolidated while the generation is open because each partition
is small and independently owned. Consolidation belongs to the immutable
compacted snapshot, not live authoring.

## Physical Persistence

Persisted v2 state lives only under:

```text
refined_detection_delta_runs/<delta_lineage_uuid>/
  generations/generation_<20-digit-uint64>/
    partitions/<partition_id>/
```

The lineage, generation, and partition groups contain exactly one manifest
attribute each. Every manifest is a strict `payload` plus
`payload_digest` envelope using canonical JSON with non-finite values
forbidden. The partition payload binds the complete logical batch manifest,
storage profile, codec profile, per-array physical plan, per-array logical
digest, and aggregate content digest. Recomputing only the outer digest after
nested tampering does not make the partition valid: the reader reconstructs
the complete expected payload from the arrays and registered policies.

Each partition is limited to 65,536 events. The complete 18-column logical row
is 99 uncompressed bytes, so the bound is about 6.19 MiB across all arrays and
the largest single array chunk (`bbox_norm_coords`) is exactly 1 MiB. Every
array therefore uses one ordinary unsharded chunk through
`editable_local_v1`; its Zarr v3 codec chain is little-endian `bytes` followed
by `zstd_fast_v1` (level 0, no codec checksum). The partition is immutable once
its manifest is installed. An incomplete group has no valid manifest and fails
closed.

Frozen generation manifests contain sorted partition receipts and an aggregate
generation digest. Successive generations also bind the immediately preceding
frozen generation's manifest digest and maximum event sequence. A new
generation cannot open until its predecessor validates as frozen, and every
new event sequence must advance beyond the preceding generation. This makes
the total-order rule executable across generation boundaries. Loading frozen
generation `G` recursively verifies the digest-linked prefix and returns base
resolution inputs for every generation through `G`; it never resolves only the
tip while dropping predecessor edits.

## Operation Semantics

The exact registry is:

| Code | Name | Precondition | Result |
| ---: | --- | --- | --- |
| 1 | `add_instance` | Key absent from all base/source/history identities; row ID at or above the allocator high-water mark; predecessor zero. | Creates one active manual row from a complete payload. |
| 2 | `replace_instance` | Target active; expected predecessor equals its latest event. | Replaces geometry/class/reason while preserving frame, acquisition identity, key, row ID, source lineage, and model-score semantics. |
| 3 | `delete_instance` | Target active; expected predecessor matches. | Tombstones the row; a raw-backed source audit row becomes `manual_clear` with no resolved refined row. |
| 4 | `restore_instance` | Target's latest event in this uncompacted lineage is delete; expected predecessor identifies that delete. | Reactivates the retained pre-delete payload and restores the raw source-audit join when present. |

Add and replace carry a complete post-operation payload. This avoids nullable
field masks and makes each event independently auditable. A replace sets
`manual_edit_flags=true` but cannot rewrite sealed frame, source, or model-score
lineage. Its bbox and class may change while `instance_key` remains stable.

An add is always manual-origin:

- `source_kind_codes=3`;
- `source_detect_row_index=-1`;
- `manual_edit_flags=true`;
- `score_valid=false` and exact `scores=0.0`;
- new non-reused `refined_row_id`; and
- `instance_key` exactly minted from recording identity, row ID, frame, bbox,
  and class by the frozen manual-correction allocator.

Delete and restore carry no positive-row payload and must use all exact
sentinels. Restore is intentionally bounded: once a compacted successor omits
a row, refined-v1 considers that identity retired. Recreating it later is a new
add with a new row ID and key, not reuse of the retired identity.

## Deterministic Resolution

`resolve_refined_detection_deltas()`:

1. validates the exact refined-v1 base and both reason registries;
2. checks every batch against the same base snapshot and manifest digest;
3. rejects duplicate global event sequences or generation-order inversion;
4. resolves targets by `instance_key`, checking any physical-row hint;
5. applies events solely by `event_sequence`;
6. rejects stale `expected_previous_event_sequence` values;
7. preserves identities and allocator non-reuse;
8. updates raw source decisions for delete/restore;
9. sorts active rows by `(frame_indices, refined_row_ids)`;
10. derives pixel boxes and centers from authoritative float32 normalized boxes;
11. rebuilds the exact `F+1` `frame_row_offsets` index;
12. deterministically re-encodes instance/source reason registries; and
13. runs the complete refined-v1 schema validator on the result.

The resolver reports added/deleted/touched keys, operation counts, generation
ordinals, the successor allocator high-water mark, and whether the rowset
changed. It never creates a selector or claims a partial result is publishable.

## Conflict And Retry Rules

- `event_sequence` is the sole merge order and must never be duplicated.
- `expected_previous_event_sequence` is per-instance optimistic concurrency.
- An edit based on stale state fails rather than winning by timestamp.
- The persisted writer makes retry of an already committed partition ID
  idempotent only when its complete batch manifest and event arrays are
  identical. Reusing an event sequence in another partition, relocating an
  event during retry, or changing any payload is a conflict.
- Generation freeze must bind sorted partition identities and content digests.
- Compaction freezes generation `G` and opens `G+1` before materialization so
  edits arriving during the job are not lost.

## Downstream Boundary

Any net rowset change requires a complete replacement artifact before
selection. “Complete replacement” does not mean rerunning every model: the
materializer may copy forward unchanged keyed rows and compute only new or
invalidated observations. Crop, keypoint, subject-mask, tracking, and training
outputs must bind the exact new snapshot/delta revision and validate complete
keyed coverage before publication.

## Implementation Checklist

- [x] Freeze exact v2 operation codes, arrays, dtypes, sentinels, and envelope.
- [x] Freeze stable add identity and non-reusing allocator behavior.
- [x] Freeze total ordering and optimistic conflict semantics.
- [x] Implement deterministic in-memory add/replace/delete/restore resolution.
- [x] Rebuild `frame_row_offsets`, update source audit, and validate the exact
      refined-v1 result.
- [x] Cover multi-instance frames, manual additions, corrections, tombstones,
      restore, stale conflicts, duplicate sequences, wrong dtypes, and invalid
      allocator/key use with in-memory tests.
- [x] Add the persisted partition writer with strict JSON manifest and content
      digest; do not extend the legacy v1 detection payload in place.
- [x] Add a frozen-generation reader and recompute every partition/generation
      digest before resolution.
- [ ] Add the immutable compactor using the shared detection storage planner,
      consolidated metadata, and complete publication validator.
- [ ] Add crash/retry tests around generation freeze, successor opening,
      scratch construction, and atomic publication.
- [ ] Define and implement downstream copy-forward/invalidation planning for
      crop, keypoints, subject masks, tracking, and training.
- [ ] Add Palette/Crimson base-plus-delta overlays only if pre-compaction reads
      are needed; compacted refined-v1 remains the public compatibility path.
- [ ] Route review saves and selectors only after the persisted writer,
      compactor, downstream completeness gate, and cross-consumer tests pass.

## Explicit Non-Goals Of This Checkpoint

- No production archive or canonical dataset is written; persistence is
  exercised only in isolated test stores.
- No production reviewer is rerouted.
- No current refined run is mutated.
- No selector, registry, or training artifact changes.
- No recording-level clipped aggregate is edited directly.
- No compactor or cluster job is authorized yet.
