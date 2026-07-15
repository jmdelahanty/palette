# Clipped Refined-Detection Recording Snapshot Contract

## Decision

Per-clip detection and refinement runs are immutable production evidence. A
finalized clipped collection may additionally be materialized as one canonical
recording-level refined-detection snapshot under:

```text
refined_detect_runs/<recording_snapshot>/
```

The recording snapshot is the consumer-facing table. It does not delete,
rewrite, or hide the selected clip runs or their finalized collection
manifest. Publication is dry-run by default, validates decoded output before
completion, and changes `latest` / `latest_complete` only after the new run is
complete.

The snapshot schema is
`palette.refined_detect_collection_snapshot.v1`. All metadata remains Zarr v3
metadata in each node's `zarr.json`; the workflow does not create Zarr v2
`.zattrs`, `.zarray`, or `.zgroup` files.

## Frame Domains

Every row retains both recording-level and clip-level frame identity:

| Array | Domain |
| --- | --- |
| `frame_indices` | Canonical zero-based `parent_frame_index` on the complete recording timeline. |
| `source_frame_indices` | Explicit recording/acquisition-frame alias; equal to `frame_indices` for this snapshot contract. |
| `source_recording_frame_ids` | Exact one-based acquisition `recording_frame_id` from `recording_frame_index.parquet`. |
| `source_clip_indices` | Selected clip ordinal. |
| `source_clip_local_frame_indices` | Zero-based decoded frame within that clip. |

The publisher reads the canonical recording frame-index Parquet, verifies every
selected clip-local mapping, and requires the selected one-camera collection to
cover the complete contiguous parent timeline. `instances/frame_counts` is
therefore sized to the complete recording, and `instances/frame_offsets` is
rebuilt against the recording-level frame axis.

## Observation and Source-Row Identity

`instances/instance_key` is the durable observation identity and must remain
unique across the complete recording snapshot. It is copied exactly from the
selected clip runs.

Per-clip stable row IDs are not globally unique. The snapshot consequently
separates snapshot-local IDs from source provenance:

| Snapshot array | Meaning |
| --- | --- |
| `instances/refined_row_ids` | Unique row ID within the recording snapshot. |
| `instances/source_refined_row_ids` | Original refined row ID within the source clip run. |
| `instances/source_detect_row_index` | Unique row into the snapshot's aggregated `source_detections` projection. |
| `instances/source_clip_detect_row_index` | Original detection row index within the source clip run. |

`source_detections/resolved_refined_row_id` is rebased to the snapshot-local
`instances/refined_row_ids`. Its original clip-local value remains in
`source_resolved_refined_row_id`. The source projection similarly carries both
recording-snapshot and clip-local detection row IDs.

These columns make all joins explicit:

- modern consumers join observations by `instance_key`;
- source audits join by `(source_clip_indices,
  source_clip_detect_row_index)` or `(source_clip_indices,
  source_refined_row_ids)`;
- video resolution uses `frame_indices` for the complete recording and the
  clip columns for direct clip reads.

## Storage and Publication

All canonical numeric arrays use Zarr v3 indexed sharding. The default outer
grid is 131,072 rows. Coordinate, confidence, and fixed-width reason payloads
retain 1,024-row inner chunks; small identity and lineage columns retain
16,384-row inner chunks.

The publisher assembles and writes one complete outer shard at a time. It does
not load the complete scientific table into memory and never allows parallel
writers to share one physical output shard. `reason_bytes` is canonical and
the legacy variable-length `reason` mirror is omitted.

Before promotion the publisher validates:

1. all selected clip runs are complete;
2. array dtypes and trailing shapes agree across clips;
3. the recording frame map is complete and non-overlapping;
4. every copied or rebased array matches its decoded write digest;
5. `instance_key` is globally unique;
6. frame counts and offsets match the recording-level rows;
7. the resulting sparse refined-detection identity contract passes.

The selected collection remains recorded through
`latest_collection`/`latest_collection_path`, while `latest` and
`latest_complete` select the recording-level snapshot.

## Refined Subject-Mask Key Repair

Historical refined subject-mask runs may predate `instance_key`. The additive
repair `fisheye.utils.backfill_refined_subject_mask_instance_keys` compares the
selected refined-mask run against the selected refined-keypoint run across all
ten shared lineage arrays:

```text
detection_indices
detection_source
frame_counts
frame_indices
source_clip_indices
source_clip_local_frame_indices
source_crop_row_ids
source_detect_row_index
source_frame_indices
source_refined_row_ids
```

Only an exact blockwise match permits key publication. The repair never reads
`masks_roi`. It writes a temporary indexed-sharded `instance_key` array,
validates it, and atomically renames it into the completed mask run. Existing
historical assignment/provenance attributes are retained.

## Delta Compaction

Future detection deltas bind to this immutable recording snapshot by
`instance_key`. A compaction/finalizer job freezes a delta generation, streams
the base one outer shard at a time, applies keyed edits, and publishes a new
complete snapshot. Coordinate-only edits retain row topology. Insertions or
deletions rebuild snapshot-local row IDs, source projection links,
`frame_counts`, and `frame_offsets`; durable identity remains the
`instance_key`. The previous snapshot, per-clip evidence, and delta partitions
remain available for provenance and rollback.

## Operator Surface

Use the compute-node wrapper:

```bash
scripts/submit_clipped_refined_detect_snapshot_bsub.sh \
  --run-id <id> \
  --zarr-path <analysis.zarr> \
  --collection-id <finalized_collection> \
  --output-run <recording_snapshot> \
  --submit
```

Add `--apply` only after reviewing the dry-run reports. The same job performs
the exact-lineage refined-mask key repair unless
`--skip-mask-backfill` is specified. The login host only submits `bsub`; all
Zarr and Parquet work occurs in the LSF allocation.
