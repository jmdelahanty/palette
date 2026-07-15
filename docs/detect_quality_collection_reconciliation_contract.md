# Collection Detection-Quality Reconciliation Contract

## Status and scope

This contract defines the modern recording-level `detect_quality` stage for an
immutable, recording-ordered detection surface. It applies to both ordinary
recording runs and clipped collections after their raw detection rows have been
assembled into one canonical row order.

The writer schema is `palette.detect_quality_collection.v2`. Historical
`detect_runs/<run>/quality_reports/<quality_run>` groups remain readable
compatibility surfaces. New collection-aware outputs live under:

```text
detect_quality_runs/<quality_run>/
```

`detect_quality` labels raw detection artifacts. It does not replace refined
detection approval, sparse correction deltas, or immutable refined snapshots.

## Source prerequisites

The selected source group must contain:

- `frame_indices: int[N]`, ordered by canonical zero-based recording frame;
- `bbox_norm_coords: numeric[N,4]`;
- `instance_key: uint64[N]`, unique within the recording surface.

Modern sources fail closed when `instance_key` is missing, has the wrong type,
or contains duplicates. `instance_key` supplies row identity; it does not
supply temporal order. The canonical recording frame index supplies order.

Clipped runs must first be represented as one recording-ordered source surface.
Clip identifiers and local row/frame indices remain lineage, but quality
workers do not independently infer collection order from directory names.

## Parallel worker phase

Each worker owns a complete source row shard and writes only a temporary trace
on node-local scratch. Workers do not write canonical Zarr arrays.

Each trace contains:

- canonical frame indices;
- exact `instance_key` values;
- `float32` pixel centroids;
- compact bounding-box validation inputs;
- a schema and quality-parameter hash.

Workers may compute shard-local coverage, box, count, and gap information in
parallel. Temporary trace ranges must be complete, non-overlapping, and
contiguous in source row order.

## Collection reconciler/finalizer

The finalizer orders traces by their declared row ranges and performs one
deterministic recording-wide reconciliation. Its result must be bit-for-bit
equivalent regardless of worker count or source shard boundaries.

The finalizer:

1. verifies trace schemas, parameter hashes, row ranges, frame bounds, and key
   uniqueness;
2. reconstructs the complete frame-count domain, including empty frames;
3. merges leading/trailing frame gaps and frames spanning row shards;
4. runs the versioned temporal state machine in canonical frame order;
5. applies label precedence deterministically;
6. writes canonical arrays once, one complete output shard at a time;
7. rereads the live source identity/order and completed outputs;
8. promotes `latest` and `latest_complete` only after validation succeeds.

The compact finalizer trace is intentionally small. At 1.2 million detections,
frame index plus two `float32` centroid coordinates is approximately 19 MB;
keys and compact box-validation fields add only tens of MB. Bounding boxes and
other source columns are never loaded as one full table by the finalizer.

## Temporal policy v2

The policy identifier is
`palette.detect_quality.temporal_reacquisition.v2`.

For single-subject quality:

- a valid observation within the jump threshold advances the baseline;
- an observation beyond the threshold starts a provisional relocation;
- an isolated provisional relocation remains `jump`;
- a stable cluster of `relocation_confirm_count` observations is accepted and
  its provisional rows are relabeled `clean`;
- a gap at least `blip_gap_threshold` frames labels the next valid observation
  `blip` and resets the baseline;
- over-expected frames do not contribute temporal representatives.

The default confirmation count is three. The default relocation-cluster radius
is one half of the effective jump threshold. Both values are explicit,
versioned provenance.

For `expected_subject_count > 1`, global jump/blip analysis remains disabled.
Raw rows interleave subjects before arena/identity assignment, so global
single-trajectory displacement is not meaningful. Count, coverage, gap, and
box validation remain parallel and recording-wide.

## Canonical arrays

All canonical arrays use Zarr v3 indexed sharding:

| Array | Shape | DType | Identity/index space |
| --- | --- | --- | --- |
| `quality_flags` | `(n_recording_frames,)` | `int8` | canonical recording frame |
| `detection_quality_labels` | `(n_detections,)` | `int8` | source detection row |
| `instance_key` | `(n_detections,)` | `uint64` | stable detection identity |

Default outer shards contain 131,072 rows. Inner chunks default to 16,384 rows
because these immutable `int8`/`uint64` arrays are scan-oriented and small.

Label schema `palette.detect_quality_labels.v1` retains existing codes:

| Code | Meaning |
| --- | --- |
| `-1` | no detection (frame array only) |
| `0` | clean |
| `2` | blip/reacquisition after a long gap |
| `3` | unconfirmed jump |
| `4` | over expected detection count |

Precedence is `over-expected` over temporal labels, temporal labels over clean,
and no-detection only for frames without source rows.

## Failure and publication contract

Completion fails closed on:

- missing, non-`uint64`, duplicate, or changed `instance_key` values;
- source row-count or frame-order changes during the job;
- incomplete or overlapping worker ranges;
- out-of-range or unresolved recording frames;
- mixed trace schemas, enum schemas, or quality parameters;
- stored label counts or key hashes that disagree with validation;
- detection labels that do not exactly map from frame flags.

Failure may leave a marked failed run for audit, but it may not change the
completed selection pointer. A successful output is immutable. Algorithm or
parameter changes create a new quality run.

Manual detection corrections belong in sparse refined-detection deltas keyed
by `instance_key`. They do not mutate raw quality labels. Compaction produces a
new refined snapshot and refreshes registry summaries for that snapshot.
