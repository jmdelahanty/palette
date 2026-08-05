# Refined Detection Snapshot And Storage Contract v1

Status: frozen implementation target; no production writer is routed through
this contract yet.

This contract defines the future-facing immutable refined-detection snapshot.
It freezes logical arrays, dtypes, identities, frame lookup, source-audit
semantics, storage access classes, codecs, metadata, and publication gates
and is now the immutable output contract used by the isolated delta-v2
compactor.

The executable definitions are:

- `src/fisheye/shared/zarr/refined_detection_schema.py`
- `src/fisheye/shared/zarr/refined_detection_storage.py`
- `src/fisheye/shared/zarr/refined_detection_manifest.py`
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

Interactive edits use external sparse delta generations. Compaction creates and
validates a new immutable snapshot through the shared storage planner. The
implemented compactor remains local and selector-ineligible; no current review
writer or production selector is routed through it yet.

## Dimensions

- `F = n_frames`
- `N = n_instances`
- `S = n_source_detections`

Published snapshots require `F >= 1`. A zero-frame result is not a presentable
refined-detection snapshot and must remain an incomplete/absent artifact rather
than publishing `[0]` offsets that Crimson cannot present.

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

The run manifest makes this identity enforceable across snapshots. It records a
stable UUID `lineage_id`, a unique UUID `snapshot_id`, the parent run and parent
manifest digest, and `next_refined_row_id` under the exact
`monotonic_int64_nonreuse_v1` allocator. A successor must:

- retain the parent's lineage ID;
- retain the parent's recording identity and use a new snapshot ID;
- bind the exact parent run and manifest digest;
- never decrease `next_refined_row_id`;
- preserve `instance_key` for every surviving `refined_row_id`;
- reject any ID below the parent high-water mark that was absent from the
  parent rowset;
- allocate new IDs at or above the parent high-water mark.

Manual keys are not a second monotonic integer sequence. They use Palette's
existing deterministic `palette.blake2b64.recording_frame_bbox_class_v1`
algorithm, the `manual_curation_refined_row_id_v1` namespace, the immutable
recording identity, and the newly allocated non-reused `refined_row_id`.
Collisions with refined or source-candidate keys fail publication. Once minted,
a surviving key is copied rather than recomputed after geometry/class edits.

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

The persisted `clipped_binding` uses schema
`palette.refined_detection.clipped_binding` v1 and requires:

- one exact finalized collection ID and manifest digest;
- exactly one camera serial per recording snapshot;
- one video identity and manifest digest;
- the canonical recording-frame-index digest;
- a complete ordered clip table with contiguous global ordinals within that
  single camera;
- per clip: media identity/digest, parent half-open frame interval,
  frame-map digest, and source refined run/manifest digest.

Source authority is discriminated by lineage profile. A `full_acquisition`
snapshot binds one `authority_kind=canonical_run` raw detection run. A
`clipped_recording_snapshot` binds
`authority_kind=clipped_collection`: the same finalized collection identity
and digest as `clipped_binding`, plus one ordered member per clip. Each member
binds the exact per-clip refined run/manifest and the canonical raw detection
run recorded by that refined manifest. A single raw-run identity is invalid
for a multi-clip recording snapshot.

The clip intervals cover `[0,F)` exactly, independently of detection rows, so
Crimson can resolve media for empty frames. Both tables' clip ordinals and local
frames must be in range and must map exactly to their parent frames. A
raw-backed instance must copy clip ordinal, local frame, and clip-local source
row from the addressed source-audit row. Its `source_refined_row_ids` value must
equal that source row's `source_resolved_refined_row_id`.

## Excluded Transition Fields

Canonical v1 groups reject:

- `frame_counts`, `frame_offsets`, and `n_detections`;
- `confidence_scores` in place of `scores` plus `score_valid`;
- variable-length `reason`, `reason_bytes`, and `review_notes` arrays.

Human notes belong in a separate review/audit artifact. Compact numeric reason
codes remain in the hot canonical table. Current float64 refined boxes, int8
source codes, int32 source row indices, whole-run rewrite behavior, and legacy
offset names are compatibility surfaces, not this contract.

The two numeric reason columns deliberately use separate registries:

- `run_manifest.payload.reason_registries.instances`;
- `run_manifest.payload.reason_registries.source_detections`.

Each registry is `palette.refined_detection.reason_registry` v1, encodes codes
as canonical decimal JSON-object keys and lowercase snake-case labels, requires
exact `"0":"none"`, and carries a SHA-256 canonical-JSON digest. The fixed
source-decision code map remains separate from source-decision reason codes.

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

The evidence candidate is frozen as
`REFINED_DETECTION_ACCESS_AWARE_CANDIDATE_V1`:

- `WINDOWED` and `INDEXED`: exact 128 KiB uncompressed inner chunks;
- `EAGER`: exact 1 MiB uncompressed inner chunks;
- outer shard target and cap: 8 MiB uncompressed;
- maximum estimated payload-object budget: 4,096 per array (the resolved
  manifest also records the summed whole-stage estimate);
- sharding only for immutable publication;
- small arrays remain a single/few objects when sharding adds no value.

The candidate reproduces the layout family supported by the existing canonical
detection evidence. Following the full-duration Palette and mounted-macOS
Crimson gates, those exact physical values are promoted under the new stable
profile ID `detection_published_access_aware_v1`. The candidate ID remains
frozen historical evidence; it is not rewritten in place.

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

The manifest has one exact persisted location:

```text
refined_detect_runs/<run>/zarr.json.attributes.run_manifest
```

It is a `palette.refined_detection.run_manifest` v1 envelope with:

- `digest_algorithm = sha256_canonical_json_v1`;
- a strict-JSON `payload` serialized with sorted keys, no whitespace, UTF-8,
  and no NaN/Infinity;
- `payload_digest`, the SHA-256 of those canonical payload bytes;
- exact completion contract/status and selector eligibility;
- the complete logical schema and resolved physical storage/codec plan;
- immutable raw source run, manifest digest, and logical-content digest;
- snapshot lineage and allocation state;
- separate reason registries and digests;
- the clipped binding when that profile is active;
- a normalized direct/consolidated declaration digest.

The metadata-declaration digest covers normalized group/array declarations
relative to the refined run group. The exact path set is the run root (`""`),
`instances`, `source_detections`, and every active 28- or 38-array schema
binding. The publisher extracts the same subtree from archive-root inline
consolidated metadata and rebases it to those run-relative paths.
`normalize_refined_detection_metadata_declarations()` first requires exact
direct/consolidated path sets and structural equality for every declaration.
There is one explicit representation normalization: Zarr-Python 3.1.3 writes
`consolidated_metadata: null` in a leaf group's direct `zarr.json`, while its
archive-root consolidation deliberately represents that same leaf group as
`{"kind":"inline","must_understand":false,"metadata":{}}`. Consumers must
treat an omitted field, `null`, and that exact empty group envelope as
equivalent. Any non-empty, wrong-kind, array-level, or otherwise different
consolidation envelope fails validation.
Arrays and every other group field must match exactly, including attributes.
The normalizer then requires exact Zarr-v3 group/array field sets and removes
only top-level `attributes` and the already-validated empty group consolidation
envelope from the digest surface. The result is a
`palette.refined_detection.metadata_declarations` v1 document;
`refined_detection_metadata_declarations_digest()` hashes its canonical JSON.
This avoids a circular digest through the `run_manifest` attribute itself.
All excluded ordinary node attributes are non-authoritative under v1. The
`run_manifest` is authoritative through its own canonical payload digest, and
the parent selection envelope is validated by its separate versioned digest.
Any future authoritative attribute must receive its own versioned digest or a
new metadata-declaration schema; consumers must not infer authority from an
attribute merely because it is present.
The manifest builder accepts both declaration maps and computes the digest;
callers cannot inject an arbitrary digest. A second nested consolidation at the
refined run group is neither required nor implied.

`validate_refined_detection_run_manifest()` reconstructs and exactly compares
the frozen logical schema and the complete byte-planned storage manifest,
including codec profile, per-array chunk/shard plans, object estimates, and
write ownership. It also deeply validates the persisted lineage, source
authority, complete clipped bindings, and canonical reason registries. Adding
or changing a nested field and recomputing the outer digest is insufficient.
A successful document parse alone does not make a snapshot contract-valid.
Before visibility, `validate_refined_detection_publication()` must also:

- recompute the metadata-declaration digest from the exact direct/consolidated
  tree;
- validate all logical arrays and cross-array invariants;
- read both `uint16` reason-code arrays and prove every persisted code exists in
  its corresponding registry.
- validate root or successor snapshot identity, including deterministic manual
  keys, allocator high-water marks, nonreuse, and surviving-key preservation;
- for clipped snapshots, consume the exact bound per-clip manifests and arrays,
  prove every source-audit row is present exactly once, and prove exact
  membership and payload identity for every `(clip_index,
  source_refined_row_id)` pair.

Successor publication must supply its parent manifest and identity arrays to
the named gate. Clipped publication must supply the per-clip evidence. Missing
external evidence fails closed; callers cannot obtain a promotable result from
local array consistency alone.

Crimson should consume the exact manifest and consolidated schema rather than
probe candidate dtypes. Direct metadata remains the fail-closed validation and
mutable-construction path. Direct/consolidated comparison must use the exact
empty-leaf-group normalization above rather than raw JSON-object or byte
equality.

## Refined Selection Contract

Production selection uses `palette.refined_detection.selection_contract` v1.
The request is typed by `stage`, `run`, and `raw_fallback_policy`; raw fallback
defaults to `forbid`.

Selection order is:

1. an explicit refined v1 run;
2. the approved authoritative refined v1 run;
3. a canonical raw run only when the request explicitly permits raw fallback
   and no refined authority was requested or selected.

An explicit refined run must exist, have a valid manifest, be complete,
selector-eligible, and have validated direct/consolidated metadata. Approval is
not required for explicit inspection/review. Any invalid explicit refined run
is a terminal error and never falls back to raw.

Implicit refined selection uses these exact parent attributes:

```text
refined_detect_runs/zarr.json.attributes.authoritative_run
refined_detect_runs/zarr.json.attributes.authoritative_run_provenance
```

The provenance value is a
`palette.refined_detection.authoritative_selection` v1 envelope. It binds the
selected run and run-manifest digest and records `review_state=approved`, review
method, approved actor/time, and one exact intended use: `analysis`, `training`,
or `analysis_and_training`. Crimson may present all three approved uses;
training promotion still applies its own intended-use gate. A present but
invalid authoritative pointer is a terminal error, not an absent authority.

Current unversioned `authoritative_run_provenance` payloads and silent refined
to raw preference chains are transition behavior. The shadow publisher and
future Crimson adapter must use the versioned envelope. Selector-ineligible
benchmark paths require a separate explicit benchmark-only API, not a
production selection exception.

Activation uses two distinct declarations deliberately. The run manifest
records the candidate's final eligibility intent, while the run-group
`stage_selector_eligible` attribute is the visibility commit bit. A completed
selector-ineligible candidate begins with both values false. Before acquiring
or committing selectors, the publisher calls
`build_refined_detection_activation_candidate_manifest()`, installs the
returned manifest with intent `true`, regenerates consolidated metadata, and
revalidates the complete candidate while the run attribute remains false. The
owner- and generation-guarded activation transaction changes the run attribute
to true as its literal final write. A selected run is valid only when both
values are true; the temporary mismatch is permitted only inside the owned,
still-invisible activation staging state.

Because the eligibility attribute is the literal final write, the archive-root
consolidated declaration intentionally retains the immediately preceding
`stage_selector_eligible=false` value. No reconsolidation may follow the commit
bit. Direct/consolidated equivalence therefore has one narrow, executable
exception: the run-root direct declaration may contain eligibility `true` while
the consolidated declaration contains `false` only when both declarations bind
the same final-intent run manifest whose publication eligibility is `true`.
`normalize_refined_detection_metadata_declarations()` enforces this exception;
all other attribute or declaration drift remains invalid. Selection reads the
authority, provenance, and eligibility attributes from direct metadata.

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
- [x] Freeze the exact persisted run-manifest and parent authority envelopes.
- [x] Freeze fail-closed refined-first selection and explicit raw fallback.
- [x] Make cross-snapshot row/key allocation and parent binding enforceable.
- [x] Bind clipped snapshots to one exact collection, camera, media table, and
      complete frame map.
- [x] Prohibit zero-frame published snapshots.
- [x] Separate and digest instance/source reason-code registries.
- [x] Normalize and digest the exact direct/consolidated Zarr-v3 declaration
      tree with executable code rather than accepting a caller digest.
- [x] Deep-parse clipped manifests and require complete interval coverage.
- [x] Deep-parse reason registries and validate persisted array-code coverage.
- [x] Reconstruct and exactly compare the frozen logical schema and complete
      byte-planned storage/codec/write-ownership manifest.
- [x] Make the named publication gate enforce root/successor snapshot identity.
- [x] Replace the ambiguous single raw authority for clipped snapshots with an
      ordered per-clip refined/raw source collection.
- [x] Require bound per-clip evidence for exact source-row coverage and
      `(clip_index, source_refined_row_id)` membership.
- [x] Add recomputed-digest tampering and multiple-subject-per-frame identity
      tests.
- [x] Add deterministic schema and storage-plan tests without Zarr I/O.
- [x] Generalize the shared immutable snapshot publisher to the clipped
      recording profile while requiring both the exact clipped binding and
      every bound per-clip artifact at publication time.
- [x] Implement the recording finalizer that preserves recording-stable
      `instance_key`, requires globally allocated non-overlapping
      `refined_row_ids`, rebuilds both `F+1` indexes, and proves the complete
      source-audit table equals the strict recording-level canonical run.
- [x] Add maintained selector-ineligible CLIs and an LSF fragment for strict
      refined-detection then crop-v2 recording publication.

Before production routing:

- [x] Complete Crimson's first read-only review; incorporate all six required
      contract changes before shadow-writer work.
- [x] Complete Crimson's second read-only review; retain three narrow
      fail-closed validation gaps as blockers to contract-valid publication.
- [x] Complete the third adversarial read-only review. It accepted the logical
      design but found exact nested-manifest, named identity-gate, and external
      clipped-lineage evidence gaps.
- [x] Ask Crimson and the DAG reviewer to verify the exact reconstruction,
      combined identity gate, and bound per-clip evidence close those gaps
      without reopening physical-layout tuning.
- [x] Add a standalone immutable shadow writer that consumes only these
      declarations, writes below an explicit `/tmp` or `.palette_benchmarks`
      namespace, and cannot update a selector or registry.
- [x] Add a read-only current-run transition adapter that reports every lossy,
      unavailable, excluded, or blocked field without mutating its source.
- [x] Validate representative full-acquisition and clipped current refined runs
      through that transition adapter. The full run is contract-ready only
      with explicit historical key initialization; the clipped aggregate fails
      closed until the bound clipped transition is implemented.
- [x] Define and validate an exact canonical raw-source run manifest and
      logical-content digest rather than accepting caller-supplied hashes.
- [x] Require the refined shadow publisher to prove all nine source-audit
      projections match the bound canonical source artifact before writing.
- [x] Validate each emitted array declaration against its resolved chunk,
      shard, codec, checksum, fill, axis, dtype, and reserved-attribute plan.
- [x] Publish and fresh-process reopen a full 23,287-frame standalone raw plus
      refined shadow pair under `/tmp`, with no selectors or production-state
      changes.
- [x] After Crimson reported its refined-v1 consumer ready, publish the same
      full snapshot as a fresh canonical/refined pair under the shared
      `.palette_benchmarks` namespace. Record exact server/macOS paths, run and
      manifest identities, direct/consolidated equality, strict receipts, and
      zero production-state changes in an immutable handoff manifest.
- [x] Have Crimson open the shared refined run through its dedicated
      selector-ineligible benchmark option, retain `instance_key` and
      `refined_row_id`, and publish a result bound to both the Palette handoff
      digest and exact Crimson commits. The immutable result passed at
      implementation `28537f64b...`, evidence `57693f8cf...`, with evidence
      SHA-256 `b0221cfb...`.
- [x] Implement the paired full-duration canary publisher with verified
      source-to-node-local staging, accept-all canonical root initialization,
      scratch construction, exclusive atomic candidate copyback, full shared
      reopen validation, and a required 4x planned and actual object gate.
- [x] Publish and independently reopen the full-duration paired
      regular/candidate refined canary. Palette commit `a94abaea...`, LSF job
      `153190577`, and canary payload digest `2c00649c...` passed exact decoded
      equality and reduced observed payload objects from 220 to 42 (5.24x).
- [x] Have Crimson apply the mounted-macOS physical transfer, latency,
      readiness, deadline, and RSS gate to the immutable pair. Crimson reported
      equal decoded/traversal digests, 86.8% less traversal transfer, 68.5%
      less whole-process transfer, zero deadline misses, and current-frame p95
      improving from 144.2 ms to 48.0 ms.
- [x] Promote the unchanged physical values as the versioned
      `detection_published_access_aware_v1` default for new canonical/refined
      v1 snapshots. Preserve `detection_regular_rollback_v1` as explicit
      rollback and do not change selectors in this step.
- [x] Publish and independently reopen one fresh selector-ineligible pair whose
      manifests carry the promoted profile ID. LSF job `153192642` validated
      every canonical/refined declaration, exact hashes, codecs, offsets,
      direct/consolidated equality, and physical equivalence to the frozen
      candidate.
- [ ] Have Crimson perform an exact-schema open-only smoke against the named
      profile canary; no new performance matrix is required.
- [ ] Route no selector until decoded values, manifest, direct metadata, and
      consolidated metadata all validate.
- [x] Add a bounded strict clip-evidence publisher that converts complete
      compatibility detect/refine groups, proves recording-stable keys against
      the native canonical slice, and allocates automated raw-backed refined
      IDs from recording canonical source-row positions. It rejects manual
      clip rows so manual identity remains owned by delta compaction.
- [ ] Invoke that strict clip-evidence fragment from every maintained clipped
      campaign. The recording finalizer continues to reject overlapping or
      unproven identities.
- [ ] Insert the strict recording finalizer into the maintained clipped
      campaign and run one selector-ineligible full-recording canary before
      archive import or selection.

Deferred until this contract is accepted:

- [x] Define the complete `palette.refined_detection.delta` v2 payload for add,
      replace, delete, restore, and class/geometry correction. Exact executable
      declarations and semantics are in
      `refined_detection_delta_v2_contract.md` and
      `refined_detection_delta.py`.
- [x] Implement deterministic in-memory base-plus-delta resolution, including
      optimistic conflicts, manual-key validation, source-audit updates,
      sorted rows, rebuilt offsets, and final refined-v1 validation.
- [x] Persist bounded v2 partitions through the shared array factory and add a
      frozen-generation reader that reconstructs manifests, physical plans,
      array digests, partition digests, and the lineage-global generation
      chain before resolution.
- [x] Implement the selector-ineligible local whole-shard immutable compactor
      with complete refined-v1 validation and phase-aware benchmark receipts.
- [ ] Bind delta-compaction provenance inside a future authoritative manifest
      revision before production promotion; the current exact v1 manifest is
      intentionally unchanged for Crimson compatibility.
- [ ] Route manual review through delta partitions.
- [ ] Add rowset-change invalidation/regeneration for crops, keypoints, masks,
      and training exports.
