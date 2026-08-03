# Eye-angle byte-planner candidate adoption — 2026-08-03

Status: implemented as an explicit, selector-ineligible candidate. No
production storage profile or selector default is changed.

## Goal and boundary

The maintained compact-v7 eye-angle writer now has a first real adoption path
for the shared analysis byte planner. The path uses the same exact 41 logical
arrays, values, dtypes, shapes, row identity, channel indexes, and validation as
the established writer. Only physical chunks, indexed shards, codecs, and the
associated digest-bound receipt differ.

The established `legacy_explicit_chunks` path remains the CLI default and keeps
its existing selector/registry behavior. The candidate must be requested with:

```text
--layout compact_dense_v2 \
--storage-profile eye_angle_access_aware_candidate_v1 \
--execution-backend serial_driver
```

Candidate runs persist `stage_selector_eligible = false`, never update
`latest`/`latest_complete`, and never emit registry completion. They carry an
explicit `eye_angle_storage_candidate` envelope stating that activation is not
allowed. Hierarchical output and Dask worker writes are rejected for this
profile.

## Candidate policy

- Inner target: approximately 1 MiB uncompressed, bounded to 512 KiB–2 MiB.
- Outer target: approximately 32 MiB uncompressed, at most 64 MiB.
- Small eager semantic tables: one ordinary chunk/object, no shard.
- Codec: Zarr v3 bytes + Zstd level 0; indexed shards use bytes + CRC32C for
  the index at the end.
- Access unit: one complete logical record. Fixed trailing semantic axes are
  never split merely to hit a byte target.
- Ownership: immutable whole-shard, single serial writer. Parallel logical row
  writes are forbidden because they may share a physical shard.

For one representative 1,000,000-row / 1,188,000-frame plan:

| Array | Inner chunk | Outer shard | Rationale |
|---|---:|---:|---|
| `roi_angles` | `(2,048, 141)` | `(61,440, 141)` | complete angle rows; about 1.1 MiB/chunk |
| `frame_angles` | `(2,048, 141)` | `(61,440, 141)` | windowed frame rows |
| `support/instance_key` | `(131,072,)` | `(1,048,576,)` | 1 MiB int64/uint64 chunks |
| `support/body_frame/valid` | whole 1,000,000-row array | none | the full bool array is under the eager/small-array boundary |
| `angle_channel_index/name` | `(141, 256)` | none | small eager semantic table |

The exact values are re-derived from concrete dimensions and dtype item sizes;
they are not writer row-count constants.

## Receipts and validation

The run stores `eye_angle_storage_plan`, an exact
`palette.analysis_storage_plan_receipt@1`. Its SHA-256 binds:

- the candidate profile and codec identity;
- all 41 logical declarations;
- resolved dimensions, shapes, and fixed dtypes;
- access units and lifecycle classifications;
- resolved inner chunks and outer shards;
- object estimates and whole-shard write ownership.

Completion recomputes the plan from the executable schema and runtime
dimensions, requires exact receipt equality even after an attacker recomputes
the payload digest, and validates every direct Zarr array declaration against
the resolved plan. Ordinary compact-v7 logical and value-alias validation still
runs unchanged.

The active writer does not consolidate the recording root because it is
mutating an existing archive and the candidate is not selector-visible. A
separate validation helper proves direct/consolidated declaration equivalence
after a benchmark or atomic publisher explicitly consolidates its immutable
artifact. A future promotion gate must perform that consolidation check before
visibility.

## Implementation checklist

- [x] Preserve established writer defaults and activation behavior.
- [x] Require explicit candidate selection.
- [x] Feed all 41 exact declarations and concrete shapes/dtypes to the shared
  byte planner.
- [x] Create candidate arrays only through the shared plan-aware Zarr factory.
- [x] Persist and deeply recompute the digest-bound plan receipt.
- [x] Validate direct physical metadata for every array.
- [x] Validate direct/consolidated equivalence in a real-Zarr fixture.
- [x] Reject unsafe Dask worker ownership and legacy hierarchical layout.
- [x] Prove candidate and established writers produce identical logical arrays.
- [ ] Run an immutable full-duration candidate publication benchmark.
- [ ] Obtain mounted Crimson read/object/RSS evidence.
- [ ] Promote a versioned shared profile only through a separate reviewed
  change; do not mutate this candidate ID in place.
