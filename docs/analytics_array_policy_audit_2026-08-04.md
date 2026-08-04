# Analytics Array Policy Audit — 2026-08-04

Status: implemented as a selector-ineligible candidate-execution gate. This
checkpoint does not promote a storage profile, change a writer default, move a
selector, or register an authority.

## Decision

Every maintained analytics candidate execution now crosses two validators in
order:

1. its family validator proves that the benchmark suite is the exact live
   logical inventory for that family; and
2. the shared `analysis_array_policy_audit` proves the common logical and
   physical policy for every array in that inventory.

The common audit rebuilds its evidence from the digest-bound benchmark suite.
A caller cannot make tampered evidence valid merely by recomputing the outer
digest.

## Enforced Contract

For every observed candidate array the audit requires:

- one exact fixed-width dtype, symbolic shape, concrete shape, axis inventory,
  description, units or explicit null, coordinate space or explicit null,
  fill/null semantics, access class, write mode, and authority role;
- `byte_planner_adopted=true` and the shared storage-policy version;
- an access unit that is not split by the inner chunk;
- chunks derived from dtype bytes and complete access-unit shape;
- shards composed of complete inner chunks, with immutable or append-only
  whole-shard single-writer ownership and no random-update sharding;
- the exact registered `zstd_fast_v1` codec profile: Zarr v3, little-endian
  bytes, Zstd level 0 without the compressor checksum, and an end-located
  little-endian-bytes plus CRC32C shard index;
- effective chunk and shard shapes and bytes, inner-chunk count, payload-object
  estimate, and object/shard budget results; and
- exactly one full-scan benchmark case for the array.

The requested byte and object budgets live in the exact storage-profile
manifest. Effective shapes, byte sizes, ownership, and object estimates live in
each array plan. The audit binds both.

## Access-Class Boundary

The executable Zarr access classes remain `eager`, `windowed`, `per_row`, and
`indexed`. Full or bulk scan is a benchmark workload applied to every array,
not a competing inner-chunk shape. Artifact byte streams such as PNG or report
files are non-Zarr surfaces and remain classified outside `ArrayIntent`.

This distinction avoids introducing fictitious Zarr access classes solely to
mirror higher-level workflows.

## Coverage

All 13 maintained candidate adapters enter the shared gate through
`_require_suite_matches_adapter`. Family validation runs first, so the common
audit cannot substitute a partial or fabricated inventory. Optional family
bundles remain governed by their exact family schema: when present in the live
suite they receive the same common audit; when absent they cannot be invented
by the benchmark request.

The audit receipt is evidence only. Its policy scope explicitly denies profile
promotion and selector or registry mutation.

## Validation

- Shared policy, candidate-execution, benchmark-suite, and storage-planner
  matrix: 52/52 passed.
- Broader maintained-family matrix: 68/70 passed. The two failures are the two
  parameterizations of one pre-existing real-track fixture. That fixture
  constructs a legacy canonical-v2 float64 detection/crop coordinate authority;
  the corrected maintained track schema requires float32 positions and
  correctly rejects it before candidate execution. It is not evidence against
  the shared policy gate and was not weakened or silently cast.
- Python compilation, Ruff, and `git diff --check` pass for the checkpoint.

The maintained float32 track decision and the explicit float64 compatibility
boundary remain recorded in
`docs/track_coordinate_precision_contract_correction_2026-08-04.md`.

## Remaining Gates

This checkpoint closes catalog-wide declaration and planning enforcement. It
does not replace the remaining representative-short/full-duration benchmarks,
physical-I/O tracing, real Palette/Crimson consumer gates, profile-promotion
decisions, rollback readers, or deliberate legacy retirement.
