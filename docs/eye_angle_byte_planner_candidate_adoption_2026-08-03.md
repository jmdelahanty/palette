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

The production materializer accepts the same explicit profile and preserves
that boundary end to end:

```text
scripts/py -m fisheye.analysis_workflows.materializers.eye_angles \
  <recording>_analysis.zarr \
  --run-name <immutable-candidate-name> \
  --storage-profile eye_angle_access_aware_candidate_v1 \
  --execution-backend serial_driver \
  --apply
```

It stages the exact sealed source subset to node-local scratch, asks the
maintained direct writer to create the final byte-planned arrays there, and
does not run the legacy post-hoc resharing copy over those arrays.

## Candidate policy

- Inner target: approximately 1 MiB uncompressed, bounded to 512 KiB–2 MiB.
- Outer target: approximately 32 MiB uncompressed, at most 64 MiB.
- Small eager semantic tables: one ordinary chunk/object, no shard.
- Codec: Zarr v3 bytes + Zstd level 0; indexed shards use bytes + CRC32C for
  the index at the end.
- Physical fill is a closed path-aware contract across all 41 arrays: NaN for
  float payload/support arrays whose frozen invalid or unavailable sentinel is
  NaN; false for boolean availability/validity arrays; and zero for fixed-width
  text, QA bit fields, and mandatory identity/time coordinates.
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
the resolved plan, including the exact semantic fill class. Zarr-v3's direct
JSON string `"NaN"` and its decoded in-memory float NaN are normalized only for
that top-level metadata field; other metadata remains exact. Ordinary
compact-v7 logical and value-alias validation still runs unchanged.

The direct writer does not consolidate while it is mutating the node-local
archive. The materializer writes all local provenance, consolidates the
node-local root, and proves direct/consolidated declaration equivalence before
publication. It then uses the common atomic run-group publisher: copy to a
same-parent hidden sibling, validate the copy, and `os.replace` it into the
immutable named path. After the publisher's final metadata write, the
materializer refreshes the authoritative recording-root consolidated metadata
and proves the direct and consolidated run attributes and 41 array
declarations agree. Publication and root consolidation run under the one
archive-wide metadata/publication lock shared by every atomic stage, rather
than an eye-angle-only lock.

This final callback is a metadata-visibility boundary only. It does not make
the candidate selector eligible, change `latest` or `latest_complete`, or emit
a registry completion event. Existing parent pointer values are recorded in
the read-only plan for evidence, then snapshotted again under the publication
lock and checked unchanged in both metadata views.

A pre-rename interruption leaves no public child and the retained node-local
run may be retried with the same name. Any post-rename failure retains an
immutable public failed/ineligible tombstone whose recovery policy requires a
new run name. If failure occurs after authoritative consolidation, the atomic
publisher invokes the eye-angle failure-visibility repair while still holding
the shared archive lock. The repair reconsolidates the root and requires the
complete direct and consolidated run attributes—including the exact atomic
tombstone—to agree before returning. Existing public names are never replaced.

Node-local scratch and the authoritative archive must also be disjoint after
resolving symlinks. Equality, scratch nested under the archive, and the archive
nested under scratch are all rejected before source metadata is opened. This
keeps the successful recursive scratch cleanup incapable of reaching the
authoritative store through either direct or aliased paths.

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
- [x] Build the candidate on node-local scratch through the direct byte planner.
- [x] Publish through hidden-copy validation and immutable rename.
- [x] Refresh authoritative consolidated metadata after final publisher writes.
- [x] Preserve selectors and suppress registry completion/activation.
- [x] Prove pre-rename retry, terminal failed-tombstone evidence, stale metadata
  refresh, and no same-name replacement.
- [x] Serialize publication/consolidation with the archive-wide lock and repair
  direct/consolidated failed-tombstone visibility after post-consolidation
  failure.
- [x] Reject equality and both resolved containment directions between source
  and scratch, including symlink aliases.
- [ ] Run an immutable full-duration candidate publication benchmark.
- [ ] Obtain mounted Crimson read/object/RSS evidence.
- [ ] Promote a versioned shared profile only through a separate reviewed
  change; do not mutate this candidate ID in place.
