# Subject-shape v4 byte-planned physical candidate — 2026-08-03

Status: implemented as an explicit, immutable, selector-ineligible candidate.
This document does not promote a production profile or authorize registry or
selector changes.

## Scope

The logical contract remains
`analysis.subject_shape.full_anatomy_v4`: four ordered anatomy components,
three ordered relations, the closed row-index bundle, body frame, source-mask
revision bundle, and the canonical source-camera binding surface. No array,
dtype, shape, fill/null interpretation, identity, or coordinate semantic was
removed or renamed.

The candidate changes only the physical Zarr v3 representation:

- profile ID: `subject_shape_access_aware_candidate_v1`;
- per-row inner target: 128 KiB uncompressed;
- eager fixed-axis inner target: 1 MiB uncompressed;
- immutable outer target and cap: 8 MiB uncompressed;
- codec: pinned `zstd_fast_v1` (`bytes` + Zstd level 0, no data checksum);
- indexed shard metadata: `bytes` + CRC32C, index at end;
- record integrity: every fixed trailing anatomy axis remains indivisible;
- planner ownership: `fisheye.analysis.subject_shape_storage`.

Chunk and shard row counts are therefore derived independently for every array
from dtype item size and the complete per-row record shape. They are not copied
from `block_rows`, frame count, or the former 131,072-row resharing constant.

## Lifecycle

1. Read the canonical refined subject-mask authority without mutation.
2. Compute the complete ROI-local v4 numeric stage in node-local scratch.
3. Materialize every exact array through the shared byte planner and Zarr-v3
   factory into a second node-local run.
4. Bind and revalidate the original producer-sealed unbound manifest, then
   compare every destination dtype, shape, fill contract, and decoded payload
   against those sealed entries in bounded row blocks.
5. Atomically copy the complete candidate to a new immutable run name.
6. At the authoritative path, revalidate source revisions, consume a refreshed
   exact unbound receipt only after preserving and rechecking the original
   producer-manifest linkage, transform/bind canonical coordinates, and create
   the final aliases through the same byte planner.
7. Recompute and persist the final bound-array storage receipt.
8. Complete the run with `stage_selector_eligible = false`, preserving the
   pre-publication `latest` and `latest_complete` states exactly.
9. Consolidate the recording root as the final metadata visibility step and
   require the exact direct/consolidated subtree node inventory, group
   declarations/attributes, and array declarations.

If a fallible check fails after consolidation, the common atomic publisher
retains an owner-bound failed/ineligible public tombstone. The family callback
reconsolidates and proves that direct and consolidated tombstone attrs are
identical. The failed name is never reused.

## Safety boundary

- Source Zarr and scratch root must be disjoint after symlink resolution in
  both containment directions.
- Existing target names are rejected; `overwrite=True` remains unsupported.
- Candidate publication does not update production pointers, profiles,
  registries, or selectors.
- Zarr writes are serial during physical conversion. Parallel scientific
  computation retains its established whole-physical-chunk ownership rules.
- The candidate is not a promotion claim. Crimson mounted-read and
  full-duration publication benchmarks remain required before any default
  changes.

## Implementation checklist

- [x] Keep the exact full-anatomy v4 logical schema unchanged.
- [x] Classify the complete unbound and bound array inventories.
- [x] Derive chunks/shards from bytes and complete access-unit shapes.
- [x] Create all candidate arrays through the shared Zarr-v3 factory.
- [x] Preserve the pinned codec and indexed-shard chain.
- [x] Prove bounded-block decoded equality during node-local conversion.
- [x] Preserve the producer-sealed source manifest and reject stale dtype,
  shape, payload, or schema inventory before physical restamping.
- [x] Preserve every logical fill contract, including the body-spline `-1`
  sentinel and its receipt semantic `minus_one_means_invalid`.
- [x] Plan final-path binding aliases through the same candidate policy.
- [x] Persist a digest-bound executable storage-plan receipt.
- [x] Reject receipt, profile, codec, chunk, shard, and array-attr tampering.
- [x] Preserve parent pointers and selector ineligibility.
- [x] Require exact direct/consolidated subtree node inventory plus group and
  array declaration equivalence.
- [x] Repair consolidated failed-tombstone visibility.
- [x] Reject equality/containment/symlink and existing-name hazards.
- [ ] Run a full-duration node-local compute/publication benchmark.
- [ ] Run Crimson mounted random-row, playback-window, eager-axis, transfer,
      cache, and RSS gates.
- [ ] Make a separate, evidence-bound promotion decision if those gates pass.

## Code and focused evidence

- `src/fisheye/analysis/subject_shape_storage.py`
- `src/fisheye/analysis_workflows/materializers/subject_shape.py`
- `src/fisheye/analysis/subject_shape_runs.py`
- `src/fisheye/shared/subject_shape_coordinate_publication.py`
- `tests/unit/fisheye/test_subject_shape_runs.py`

No production archive, selector, registry, or shared profile is changed by this
checkpoint.
