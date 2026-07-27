# Refined Detection Snapshot And Storage Contract v1

Status: frozen implementation target; no production writer is routed through
this contract yet.

This contract defines the future-facing immutable refined-detection snapshot.
It freezes logical arrays, dtypes, identities, frame lookup, source-audit
semantics, storage access classes, codecs, metadata, and publication gates
before Palette designs the next detection delta schema or compactor.

The executable definitions are:

- `src/fisheye/shared/zarr/refined_detection_schema.py`
- `src/fisheye/shared/zarr/refined_detection_storage.py`
- `src/fisheye/shared/zarr/array_contracts.py`

Current refined-detection writers remain transition surfaces. This checkpoint
does not change selectors, registries, reviewers, training exporters, or
production archives.

## Scope And Lifecycle

`detect_runs/<run>` remains immutable raw detector evidence.

`refined_detect_runs/<run>` is a complete immutable snapshot containing:

1. `instances`: every accepted or manually added positive instance;
2. `source_detections`: every candidate from the bound raw detection rowset,
   including rejected candidates and the disposition of each source row.

The snapshot allows zero, one, or many instances in a frame. A missing
detection is represented by no row. An eventual manual addition becomes an
ordinary `instances` row in the next compacted snapshot; it never mutates the
raw detect run.

Interactive edits must eventually use external sparse delta generations.
Compaction must create and validate a new immutable snapshot. Neither the delta
schema nor compactor is part of v1 implementation work yet.

## Dimensions

- `F = n_frames`
- `N = n_instances`
- `S = n_source_detections`

Every array is required for its active lineage profile. Canonical groups have
an exact array set: adding a field requires a new schema version.

## `instances` Arrays

| Array | Shape | Dtype | Contract |
| --- | ---: | --- | --- |
| `frame_indices` | `[N]` | `int32` | Zero-based camera frame; rows sorted by frame then `refined_row_ids`. |
| `source_acquisition_frame_index` | `[N]` | `int64` | Sealed acquisition frame identity; equals `frame_indices` as `int64` in a recording-level snapshot. |
| `instance_key` | `[N]` | `uint64` | Required durable observation/edit identity; unique in the table. |
| `refined_row_ids` | `[N]` | `int64` | Nonnegative, unique, non-reused logical identity in one refined lineage. |
| `bbox_norm_coords` | `[N,4]` | `float32` | Authoritative finite contained `cx,cy,w,h` geometry. |
| `bbox_img_xyxy` | `[N,4]` | `float32` | Exact source-camera pixel-edge projection of the normalized box. |
| `centers_img_xy` | `[N,2]` | `float32` | Exact continuous-pixel midpoint of `bbox_img_xyxy`. |
| `scores` | `[N]` | `float32` | Finite value in `[0,1]`; paired with `score_valid`. |
| `score_valid` | `[N]` | `bool` | False for a row without a model score; its physical score must then be exact zero. |
| `class_ids` | `[N]` | `int32` | Nonnegative taxonomy index, currently bounded by `uint16` range. |
| `source_kind_codes` | `[N]` | `uint8` | `1=raw_detect`, `3=manual`. Interpolation is not a canonical v1 source kind. |
| `manual_edit_flags` | `[N]` | `bool` | True when a person created or corrected the row. |
| `source_detect_row_index` | `[N]` | `int64` | Row in `source_detections`; exact `-1` for a manual row without a raw candidate. |
| `reason_codes` | `[N]` | `uint16` | `0=no additional reason`; all nonzero meanings come from an exact versioned map and digest in the run manifest. |
| `frame_row_offsets` | `[F+1]` | `int64` | Authoritative CSR frame-to-row index. |

For a raw-backed row, key, frame identity, and score remain joined to its
source candidate. When `manual_edit_flags=false`, normalized geometry and class
must also equal the source candidate. A correction may change geometry or class
only with `manual_edit_flags=true` while retaining raw lineage.

A manual row must have a newly minted durable `instance_key` that collides with
neither another refined row nor any bound source-candidate key, a new non-reused
`refined_row_id`, `source_detect_row_index=-1`, `source_kind_codes=3`,
`manual_edit_flags=true`, `score_valid=false`, and `scores=0.0`.

## `source_detections` Arrays

| Array | Shape | Dtype | Contract |
| --- | ---: | --- | --- |
| `source_detect_row_index` | `[S]` | `int64` | Exact contiguous identity `0..S-1`. |
| `frame_indices` | `[S]` | `int32` | Zero-based camera frame; nondecreasing. |
| `source_acquisition_frame_index` | `[S]` | `int64` | Sealed acquisition frame identity. |
| `instance_key` | `[S]` | `uint64` | Stable key copied from the bound raw candidate. |
| `bbox_norm_coords` | `[S,4]` | `float32` | Authoritative raw candidate geometry. |
| `bbox_img_xyxy` | `[S,4]` | `float32` | Exact pixel-edge projection. |
| `centers_img_xy` | `[S,2]` | `float32` | Exact continuous-pixel center. |
| `scores` | `[S]` | `float32` | Finite raw model score in `[0,1]`. |
| `class_ids` | `[S]` | `int32` | Raw model class. |
| `decision_codes` | `[S]` | `uint8` | `0=accepted`, `1=filtered`, `2=duplicate`, `3=manual_clear`. |
| `resolved_refined_row_id` | `[S]` | `int64` | Refined row for an accepted source; exact `-1` otherwise. |
| `reason_codes` | `[S]` | `uint16` | Versioned compact decision reason. |
| `frame_row_offsets` | `[F+1]` | `int64` | Authoritative CSR frame-to-source-row index. |

Accepted source rows and raw-backed refined rows are one-to-one. Every
accepted source row resolves to the corresponding present `refined_row_id`,
and every nonaccepted source row resolves to `-1`.

## Frame Lookup

Both tables persist `frame_row_offsets` with shape `F+1`:

```text
rows for frame f = table[offsets[f]:offsets[f + 1]]
```

Offsets must start at zero, be nondecreasing, exactly match the table's sorted
`frame_indices`, and end at the table row count. `frame_counts`, `n_detections`,
and `frame_offsets` are not canonical v1 arrays. Counts are derived as
`diff(frame_row_offsets)`.

This representation supports empty frames and multiple detections without a
one-detection-per-frame assumption. Crimson may read and retain each selected
table's complete offset vector once, then perform O(1) frame-to-row lookup.

## Clipped Recording Snapshot Profile

The `clipped_recording_snapshot` profile adds five required lineage columns to
each table:

| Table | Added arrays |
| --- | --- |
| `instances` | `source_recording_frame_ids:int64`, `source_clip_indices:int32`, `source_clip_local_frame_indices:int32`, `source_clip_detect_row_index:int64`, `source_refined_row_ids:int64` |
| `source_detections` | `source_recording_frame_ids:int64`, `source_clip_indices:int32`, `source_clip_local_frame_indices:int32`, `source_clip_detect_row_index:int64`, `source_resolved_refined_row_id:int64` |

Recording frame IDs are one-based and equal acquisition frame indices plus
one. Manual instance rows use `source_clip_detect_row_index=-1`; raw-backed rows
require a nonnegative clip-local source row. Clip membership and frame mapping
must also be bound by exact manifest identities and digests when a publisher is
implemented.

## Excluded Transition Fields

Canonical v1 groups reject:

- `frame_counts`, `frame_offsets`, and `n_detections`;
- `confidence_scores` in place of `scores` plus `score_valid`;
- variable-length `reason`, `reason_bytes`, and `review_notes` arrays.

Human notes belong in a separate review/audit artifact. Compact numeric reason
codes remain in the hot canonical table. Current float64 refined boxes, int8
source codes, int32 source row indices, whole-run rewrite behavior, and legacy
offset names are compatibility surfaces, not this contract.

## Storage Intent Contract

Physical row depths are computed from uncompressed bytes per complete row, not
from one frame count or one row constant shared across dtypes.

| Array class | Access | Inner unit | Immutable publication |
| --- | --- | --- | --- |
| Both `frame_row_offsets` arrays | `EAGER` | one boundary offset; reader normally loads the complete selected index | Shard when large; a genuinely small array may remain one ordinary object. |
| `instances/*` except offsets | `WINDOWED` | one complete instance row, with all trailing axes intact | Indexed shards along row axis only. |
| `source_detections/*` except offsets | `INDEXED` | one complete source row, with all trailing axes intact | Indexed shards along row axis only. |

Every sharded write must assign a complete outer shard to one writer. Logical
row slices that share a physical chunk or shard may not be written in parallel.
Unsharded immutable arrays use one materializing writer.

The general shared baseline remains `published_http_v1`: approximately 1 MiB
uncompressed inner chunks and 32 MiB target shards. It is useful as a control,
not evidence that every access class should use the same physical chunk bytes.

The frozen access-aware candidate is
`REFINED_DETECTION_ACCESS_AWARE_CANDIDATE_V1`:

- `WINDOWED` and `INDEXED`: exact 128 KiB uncompressed inner chunks;
- `EAGER`: exact 1 MiB uncompressed inner chunks;
- outer shard target and cap: 8 MiB uncompressed;
- maximum estimated payload-object budget: 4,096;
- sharding only for immutable publication;
- small arrays remain a single/few objects when sharding adds no value.

The candidate reproduces the layout family supported by the existing canonical
detection evidence. It is deliberately not the planner default and is not yet
a promoted Palette writer profile.

The paired gate's anchor is separately frozen as
`REFINED_DETECTION_REGULAR_CONTROL_V1`: exact 1 MiB uncompressed inner chunks,
no outer sharding, and otherwise the same codec contract. The general
`published_http_v1` profile is sharded and therefore must not be mislabeled as
the regular control.

## Zarr And Codec Contract

All canonical v1 snapshot arrays use Zarr v3 with codec profile
`zstd_fast_v1`:

1. little-endian `bytes` serializer;
2. Zstandard compressor, level `0`, compressor checksum disabled;
3. for sharded arrays, `sharding_indexed` with the index at the end;
4. little-endian shard-index bytes plus `crc32c`.

This is an exact compatibility contract, not an invitation for per-writer
codec selection. A codec change requires a new named profile and reader
compatibility evidence.

## Metadata And Manifest Contract

Mutable or incomplete construction uses direct metadata. Before a snapshot can
be visible to a selector, publication must:

- validate every logical invariant and decoded value;
- validate direct array/group metadata;
- write strict-JSON consolidated metadata at the archive root;
- prove consolidated declarations equal direct `zarr.json` declarations;
- include exact schema, dtype, code-map, reason-map, storage-profile, codec,
  source-run, dimensions, and lineage identities/digests in the run manifest.

Crimson should consume the exact manifest and consolidated schema rather than
probe candidate dtypes. Direct metadata remains the fail-closed validation and
mutable-construction path.

## Practical Physical-Profile Gate

Before the candidate becomes a production default, publish one immutable
refined snapshot as a regular 1 MiB control and as the access-aware candidate,
then run the frozen Crimson workload. Promotion requires:

- exact decoded equality and zero stale publications;
- exactly one retained offset read per selected table;
- zero playback deadline misses;
- no meaningful readiness or current-frame regression;
- at least 4x fewer payload objects;
- at least 20% less traversal transfer;
- no material peak-RSS regression;
- direct/consolidated metadata equivalence and full codec/CRC validation.

If it passes, version and promote the profile, publish one canary refined run,
validate it in Palette and Crimson, and retain the old profile for rollback.
The deferred three-candidate matrix is unnecessary unless this paired check
exposes a material problem.

## Implementation Checklist

Contract freeze:

- [x] Separate immutable raw evidence from immutable refined snapshots.
- [x] Freeze exact array paths, shapes, dtypes, identities, code maps, and
      sentinels for full-acquisition snapshots.
- [x] Freeze the conditional clipped-recording lineage extension.
- [x] Make both `F+1` offset indexes required and authoritative.
- [x] Make manual additions representable in a compacted snapshot.
- [x] Freeze byte-based access classes, physical ownership, Zarr v3 codecs,
      and consolidated-metadata requirements.
- [x] Preserve the access-aware hybrid as an explicit unpromoted candidate.
- [x] Preserve a genuine 1 MiB unsharded paired-gate control.
- [x] Add deterministic schema and storage-plan tests without Zarr I/O.

Before production routing:

- [ ] Review the exact v1 contract jointly with Palette and Crimson.
- [ ] Add an immutable shadow writer that consumes only these declarations.
- [ ] Validate a real current refined run against a deliberate transition
      adapter and report every lossy or unavailable field.
- [ ] Publish the paired regular/candidate refined canary and apply the
      practical gate above.
- [ ] Promote a versioned profile only after that gate passes.
- [ ] Route no selector until decoded values, manifest, direct metadata, and
      consolidated metadata all validate.

Deferred until this contract is accepted:

- [ ] Define a complete detection delta v2 payload for add, replace, delete,
      restore, and class/geometry correction.
- [ ] Implement base-plus-delta resolution.
- [ ] Implement the whole-shard immutable compactor.
- [ ] Route manual review through delta partitions.
- [ ] Add rowset-change invalidation/regeneration for crops, keypoints, masks,
      and training exports.
