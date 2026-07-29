# Shared Zarr Storage Policy Design

Status: proposed

## Purpose

Define one owner for Palette Zarr physical layout decisions across analysis,
review, publication, and training stores. Array producers continue to own
scientific meaning and logical schemas; the shared storage module owns chunks,
shards, codecs, layout provenance, and write-safety constraints.

This design complements:

- [`zarr_storage_lifecycle_policy.md`](zarr_storage_lifecycle_policy.md)
- [`dask_zarr_write_safety.md`](dask_zarr_write_safety.md)
- [`tabular_delta_compaction_contract.md`](tabular_delta_compaction_contract.md)
- [`shared_coordinate_storage_contract_v1.md`](shared_coordinate_storage_contract_v1.md)

## Decision Summary

1. Plan chunks from uncompressed bytes and access shape, not a universal row
   count.
2. Use one shared planner for heterogeneous arrays. Array shape and dtype are
   planner inputs, not reasons to fork policy.
3. Keep randomly editable authorities unsharded by default. Store tabular edits
   as sparse deltas over immutable sharded bases where practical.
4. Require indexed sharding for immutable multi-chunk publication and training
   arrays unless an explicit exception applies.
5. Keep metadata initialization work separate from payload layout work:
   consolidation, manifests, deferred opens, and selector safety remain required.
6. Make the shared storage module the only production array-creation path and
   enforce that boundary in CI.

The initial byte targets are hypotheses, not permanent format constants:

- inner chunk target: approximately 1 MiB uncompressed
- normal acceptable inner range: 512 KiB to 2 MiB
- narrow immutable shard target: 8 MiB to 32 MiB
- large dense immutable shard target: 128 MiB to 1 GiB
- object-count target: profile- and archive-scale dependent

Zarr recommends starting around 1 MiB uncompressed while emphasizing that
access shape is more important than a universal size. It also defines inner
chunks as independently readable units and shards as storage-object/write units:

- <https://zarr.readthedocs.io/en/v3.1.3/user-guide/performance.html>
- <https://zarr.readthedocs.io/en/latest/user-guide/glossary/>

## Scope And Non-Goals

In scope:

- recording analysis Zarrs
- task-scoped review/labeling Zarrs
- immutable trainer-facing Zarr exports
- chunk, shard, codec, fill, ordering, and write-ownership policy
- storage provenance, validation, reporting, and migration planning

Not in scope:

- changing scientific array meanings or coordinate authorities
- replacing Zarr with a different canonical recording format
- treating rechunking as the primary fix for Crimson initialization latency
- forcing unlike arrays into one physical representation

Coordinate semantics are nevertheless a required input boundary. The shared
storage layer consumes exact logical array contracts and the versioned
coordinate-surface catalog; it does not infer coordinate meaning from an array
name or choose a scientific representation. Detection normalized boxes,
source-camera pixel keypoints, ROI-local masks, and integer crop extents may all
use different natural representations while sharing one explicit semantic
catalog and source-camera presentation mapping.

## Organizing Principle

The core API is a pure planner:

```python
plan_storage(
    *,
    shape,
    dtype,
    axes,
    access,
    write_mode,
    profile,
    atomic_axes=(),
    access_unit=None,
) -> StoragePlan
```

For an ordinary row-aligned array:

```text
bytes_per_row = dtype.itemsize * product(per_row_shape)
ideal_rows = target_chunk_bytes // bytes_per_row
```

The planner then rounds, clamps, and aligns the result using the access class,
write mode, worker ownership, shard budget, and object-count budget. Writers do
not choose per-array row constants.

`axes`, `atomic_axes`, and `access_unit` extend the same arithmetic to:

- row records such as keypoints `[row, keypoint, xy]`
- component-addressed masks `[row, component, y, x]`
- semantic column bundles such as eye-angle matrices
- video frames
- flat ragged values such as contour points or RLE counts
- static lookup and enum arrays

## Storage Intent Types

### Access

| Class | Typical arrays | Inner-chunk rule | Shard rule |
| --- | --- | --- | --- |
| `EAGER` | small metadata tables, enums, run-level lookups | whole array below a configured cap | none |
| `WINDOWED` | timelines, keypoints, lineage, counts, offsets | byte-budget along the row axis; preserve atomic trailing axes | shard when immutable and multi-chunk |
| `PER_ROW` | masks and other per-observation display data | byte-budgeted multiples of one declared row/component access unit | shard when immutable |
| `INDEXED` | contour points, RLE counts, other CSR value arrays | byte-budget along the flat value axis | shard when immutable |

The planner may also accept `max_decode_bytes_per_query` when a workload needs a
stricter bound than the normal byte target.

### Write Mode

| Mode | Rule |
| --- | --- |
| `RANDOM_UPDATE` | no shards unless an explicit serialized whole-shard ownership contract exists |
| `APPEND_ONLY` | shards only when the writer buffers and commits complete non-overlapping shards |
| `IMMUTABLE` | shard every multi-chunk array; one appropriate chunk remains regular |

This axis enforces the distinction between an edit surface and a published
snapshot. It also preserves the repository rule that parallel writers must own
whole, non-overlapping physical chunks and, when applicable, whole shards.

## Storage Profiles

| Profile | Purpose | Default behavior |
| --- | --- | --- |
| `scratch_compute_v1` | node-local intermediate output | align with compute/write ownership; publication layout is not required |
| `editable_local_v1` | mutable authoritative analysis or review surface | optimize edit unit; regular chunks by default |
| `published_http_v1` | immutable Crimson-facing snapshot | indexed shards, consolidated metadata, transport manifest, selector eligibility |
| `detection_published_access_aware_v1` | promoted immutable canonical/refined detection snapshot | 128 KiB windowed/indexed chunks, 1 MiB eager chunks, 8 MiB indexed shards |
| `detection_regular_rollback_v1` | explicit immutable detection rollback | exact 1 MiB chunks, no outer sharding; never selected by default |
| `training_immutable_v1` | immutable trainer-facing dataset | indexed shards; inner chunks tuned for random minibatch/sample access |

The logical schema may be identical across profiles. The resolved physical
layout is allowed to differ and must be recorded in provenance.

## Keypoint Editing, Promotion, And Training

Yes: promoted keypoint datasets and trainer-facing training Zarrs should be
sharded. Interactive edits should not rewrite large shards in place.

The preferred lifecycle is:

```text
immutable sharded keypoint base
              |
              +-- sparse task/review deltas or an editable task-scoped review run
                              |
                              v
                  validate identity, review state,
                  skeleton, coordinates, and provenance
                              |
                              v
                 compact/promote a new immutable
                    sharded keypoint snapshot
                              |
                              v
                  export an immutable sharded
                     trainer-facing Zarr
```

Rules:

- Never modify an immutable canonical keypoint shard for one interactive edit.
- The review system must identify the authoritative current label and preserve
  before/after audit history.
- Promotion resolves base plus accepted deltas into a complete new snapshot and
  validates decoded equality for unchanged rows.
- Only approved, training-eligible snapshots may enter a training manifest.
- Training export is a new immutable artifact; it is not the live edit surface.
- Training arrays are sharded, but their inner chunks follow training access:
  full per-sample records, random minibatches, and trainer prefetch behavior.
- Split indices, source indexes, labels, and image arrays all use the shared
  planner rather than fixed row constants.

Current Palette already has much of this lifecycle shape:

- keypoint delta compaction publishes an immutable indexed-sharded snapshot
- merged training export validates source identity, skeleton, row gates, and
  provenance, but its physical layout still uses writer-local chunk choices and
  does not consistently shard output arrays

The storage-policy rollout should preserve the former and migrate the latter to
`training_immutable_v1`.

## Representative Derived Layouts

These are benchmark candidates, not frozen contracts.

| Array | Access unit | Candidate inner chunk | Candidate immutable shard |
| --- | ---: | ---: | ---: |
| `frame_offsets`, `int64` | 8 B | 131,072 rows = 1 MiB | whole array or at least 8 MiB |
| `frame_counts`, `int32` | 4 B | 262,144 rows = 1 MiB | whole array or at least 8 MiB |
| Boolean status | 1 B | 262,144 to 1,048,576 rows | whole array or 8–32 MiB |
| `keypoints_img`, `float64[5,2]` | 80 B | 8,192 or 16,384 rows | 131,072 rows = 10 MiB |
| sampled body contour, `float32[128,2]` | 1 KiB | 1,024 rows = 1 MiB | 32–128 MiB |
| dense mask component, `uint8[512,512]` | 256 KiB | 4 rows = 1 MiB | 128 MiB to 1 GiB |

Shard planning must consider both target bytes and target object count. For
example, a 32 MiB mask shard improves per-frame decode amplification but creates
roughly the same number of storage objects as today's unsharded 32 MiB mask
chunks. A much larger immutable shard is needed to improve both dimensions.

## Codec Policy

Codec selection is explicit and versioned. It must not depend silently on the
installed Zarr default.

Initial named profiles should include:

- `zstd_fast_v1`
- `blosc_lz4_bitshuffle_v1`
- `uncompressed_gds_v1`

Each codec profile must declare:

- exact serializer, compressor, level, shuffle, and checksum settings
- supported Palette datatypes and representations
- Crimson decoder compatibility
- benchmark evidence
- a stable profile identifier

## Common Frame-To-Row Index

Frame-sorted stages should share this contract:

```text
frame_counts   int32[F]
frame_offsets  int64[F + 1]
row_frames     int32[R]       optional when otherwise derivable
```

Required invariants:

```text
frame_offsets[0] == 0
diff(frame_offsets) == frame_counts
frame_offsets[-1] == R
rows for frame f == [frame_offsets[f], frame_offsets[f + 1])
```

Whether offsets use `EAGER` or `WINDOWED` must be decided from Crimson request
traces. Full loading at open implies `EAGER`; range reads during playback or
scrubbing imply `WINDOWED`.

## Shared Module Boundary

Proposed package structure:

```text
src/fisheye/shared/zarr/
  array_contracts.py
  coordinate_contracts.py
  storage_intent.py
  storage_profiles.py
  storage_planner.py
  codec_profiles.py
  array_factory.py
  storage_manifest.py
  storage_validation.py
  storage_report.py
```

Existing helpers become clients or adapters:

- `columnar.py` retains structured-field and string representation logic but
  delegates physical planning.
- `chunk_profiles.py` becomes intent presets or is retired.
- `subject_mask_chunks.py` loses physical row constants; mask code continues to
  define mask semantics and representations.
- `zarr_sharded_copy.py` remains the mechanical copy engine and consumes an
  explicit destination `StoragePlan`.

### Array Creation Fence

Production code creates arrays only through `array_factory`.

The factory:

- requires storage intent and profile
- refuses raw caller-supplied `chunks=` and `shards=`
- creates the array from a resolved `StoragePlan`
- records the resolved contract and estimates
- validates worker ownership requirements

CI uses an AST-based check to forbid direct array-creation calls outside:

- the shared factory
- an explicit migration/diagnostic allowlist
- tests that intentionally exercise raw Zarr behavior

An explicit `create_array_exact()` escape hatch supports migrations that must
preserve an existing physical layout. Its use requires a reason and validation.

## Resolved Storage Contract

Every array records enough information to explain and reproduce its layout:

```json
{
  "schema_id": "palette.array_storage_contract",
  "schema_version": 1,
  "profile_id": "published_http_v1",
  "policy_version": "palette.storage_planner.v1",
  "access": "windowed",
  "write_mode": "immutable",
  "chunk_shape": [16384, 5, 2],
  "chunk_bytes_uncompressed": 1310720,
  "shard_shape": [131072, 5, 2],
  "shard_bytes_uncompressed": 10485760,
  "estimated_payload_objects": 10,
  "codec_profile": "zstd_fast_v1",
  "write_ownership": "whole_shard_single_writer"
}
```

Run-level manifests summarize capabilities, index paths, and layout digests.
Root publication metadata identifies selector-visible runs without requiring a
consumer to probe every optional array.

## Metadata And Selector Track

Payload planning does not solve slow archive initialization by itself.

Required parallel work:

- consolidate metadata after every successful selector-visible publication
- stamp a metadata generation/digest
- provide a root transport/capability manifest
- make Crimson defer optional arrays and use consolidated typed metadata
- prevent `latest` from selecting canary, smoke, incomplete, ineligible, or
  profile-incompatible runs

A run may become `latest` only when it is:

```text
complete
selector_eligible
non-canary
profile-compatible
validated
published
```

## Benchmark Contract

Benchmark inner targets of 128 KiB, 512 KiB, 1 MiB, and 2 MiB. Benchmark shard
targets of 8 MiB, 32 MiB, 128 MiB, and 512 MiB where applicable.

Measure:

- cold and warm random lookup
- forward playback and 700 FPS traversal
- real Crimson window sizes
- random training minibatches and prefetch
- full-array eager loading
- time to first usable frame
- HTTP and range-request counts
- compressed bytes transferred
- uncompressed bytes decoded
- cache hit rate and peak memory
- write/publication throughput and peak memory
- final payload-object and metadata-object counts

Use both a request-logging HTTP Range server and the actual Mac/VPN path.

## Implementation Goals And Checklists

### Goal 0: Approve The Contract

- [ ] Approve access classes and write modes.
- [ ] Approve the four storage profiles.
- [ ] Confirm codec support in Crimson.
- [ ] Capture Crimson traces for offsets, keypoints, masks, and initialization.
- [ ] Decide initial shard object-count budgets by profile.
- [ ] Reconcile this design with existing lifecycle and delta-compaction docs.

### Goal 1: Build A Pure Planner Without Changing Writers

- [ ] Add `Access`, `WriteMode`, `StorageProfile`, and `StoragePlan` types.
- [ ] Implement byte-based inner-chunk planning.
- [ ] Implement aligned shard planning with object-count constraints.
- [ ] Implement full-array handling for `EAGER`.
- [ ] Implement access-unit preservation for `PER_ROW`.
- [ ] Implement worker/chunk/shard ownership validation.
- [ ] Add versioned codec profiles.
- [ ] Add golden tests for counts, offsets, keypoints, masks, sampled contours,
      ragged contour points, and strings.
- [ ] Add a read-only report comparing actual and proposed layouts.

Exit criteria:

- [ ] Planning is deterministic for identical inputs.
- [ ] No planner test requires a real filesystem Zarr.
- [ ] Plans report chunk bytes, shard bytes, object count, and rationale.

### Goal 2: Adopt Low-Risk Immutable Row Arrays

- [ ] Add the common `frame_counts`/`frame_offsets` contract.
- [ ] Route detection, crop, keypoint, and mask lineage arrays through the
      planner for new runs.
- [ ] Route new YOLO keypoint arrays through byte-based planning.
- [ ] Preserve full keypoint and coordinate trailing axes in each inner chunk.
- [ ] Validate decoded values and frame-index invariants.
- [ ] Enable the direct-creation AST check in warning mode.

Exit criteria:

- [ ] No adopted writer supplies a raw row chunk or shard constant.
- [ ] Immutable multi-chunk outputs are sharded.
- [ ] Existing readers remain compatible.

### Goal 3: Complete Keypoint Edit And Training Lifecycles

- [ ] Declare the authoritative task/review label surface.
- [ ] Use sparse deltas or an explicitly editable task-scoped review run.
- [ ] Preserve before/after audit history for edits.
- [ ] Compact base plus accepted deltas into a new immutable sharded snapshot.
- [ ] Require approval, intended-use, skeleton, identity, and coordinate checks
      before training promotion.
- [ ] Add `training_immutable_v1` to keypoint training export.
- [ ] Plan and shard ROI images, keypoints, bboxes, status, split indices, and
      source-index arrays.
- [ ] Benchmark real trainer random sampling, batching, and prefetch.
- [ ] Validate a promoted training artifact before registry activation.
- [ ] Prohibit interactive mutation of promoted training artifacts.

Exit criteria:

- [ ] Interactive edits never rewrite canonical large shards.
- [ ] Promoted keypoint snapshots are immutable and sharded.
- [ ] Trainer-facing Zarrs are immutable, sharded, validated, and reproducible
      from a manifest.

### Goal 4: Adopt Dense Masks And Indexed Values

- [ ] Move refined dense-mask inner chunks toward the per-row/component byte
      budget.
- [ ] Keep live dense edit authorities regular-chunked.
- [ ] Materialize immutable sharded mask publication snapshots where required.
- [ ] Shard sampled contours without changing their already appropriate inner
      body-contour chunk.
- [ ] Re-plan full contour points and RLE counts as `INDEXED` flat arrays.
- [ ] Keep CSR pointer/length invariants exact across migration.
- [ ] Validate per-frame reads, full finalization, and edit amplification.

Exit criteria:

- [ ] Per-frame mask decode is bounded by the declared plan.
- [ ] Published dense products meet the object-count budget.
- [ ] Editable authorities and derived caches retain correct stale semantics.

### Goal 5: Enforce Creation And Publication

- [ ] Route remaining production writers through `array_factory`.
- [ ] Convert columnar, geometry-preload, and sharded-copy helpers to planner
      adapters.
- [ ] Enable the direct-creation AST check as a required CI gate.
- [ ] Add the narrow migration/diagnostic allowlist.
- [ ] Require storage-contract validation before run completion.
- [ ] Require consolidated metadata and manifest generation at publication.
- [ ] Enforce selector eligibility and canary/smoke exclusion.

Exit criteria:

- [ ] New production arrays cannot bypass the policy accidentally.
- [ ] Every selector-visible array has a valid resolved storage contract.
- [ ] Crimson can discover published capabilities without probing optional
      arrays individually.

### Goal 6: Census And Migration

- [ ] Inventory existing archives by actual chunks, shards, codecs, metadata
      state, selector eligibility, and estimated object count.
- [ ] Classify archives by required migration family.
- [ ] Extend dry-run migration planning beyond YOLO detection/keypoints.
- [ ] Stage, validate, and atomically publish migrated immutable runs.
- [ ] Never rewrite mutable authorities without an explicit lifecycle-specific
      migration contract.
- [ ] Re-consolidate metadata after migration.

Exit criteria:

- [ ] Migration reports prove decoded equality or a documented semantic
      transformation.
- [ ] Old and new layouts remain distinguishable by profile and policy version.
- [ ] Registry state identifies which archives still require migration.

## First Pull Request Checklist

The first pull request should change no production writer behavior.

- [ ] Add intent/profile/plan dataclasses and enums.
- [ ] Add the pure planner.
- [ ] Add byte, alignment, and object-count calculations.
- [ ] Add golden in-memory tests for representative array families.
- [ ] Add the read-only storage report.
- [ ] Document planned AST enforcement without enabling it.
- [ ] Record open benchmark and codec questions.

## Open Decisions

- [ ] Does Crimson read frame offsets eagerly or by range?
- [ ] Which codec profiles can Crimson decode natively and efficiently?
- [ ] What object-count budget is appropriate for multi-terabyte mask products?
- [ ] Should published HTTP masks live in the analysis archive or a separately
      materialized publication store?
- [ ] What training minibatch and sampler patterns should determine the
      `training_immutable_v1` inner chunk target?
- [ ] Which task-scoped training review Zarr is authoritative before promotion,
      and when is it frozen?
- [ ] Which existing direct array-creation utilities require a permanent escape
      hatch rather than migration to the factory?
