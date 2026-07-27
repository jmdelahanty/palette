# Detection Promoted-Profile Canary

Date: 2026-07-27

Status: passed; selector-ineligible; no production-state changes

## Scope

This is the resolution-only final gate for
`detection_published_access_aware_v1`. It is not another performance matrix.
It verifies that the stable profile ID resolves to the exact physical layout
already measured by Crimson and accepted by Palette.

## Immutable Provenance

- Palette commit:
  `1d96158393c735f01d6fbaec64d585f2a9a4b5b3`;
- Crimson benchmark implementation:
  `9cf04acee9682a6f4f5fae005c0af6077ec5cc4b`;
- Crimson evidence commit:
  `258f25811c76dd48e206183d73f9807e140e7264`; and
- Crimson aggregate SHA-256:
  `0be6f191b0d684914cdd48bc938267cd8bb2fb6e066e158ba65cf2339d466d32`.

The cluster ran from the clean detached worktree:

`/groups/johnson/johnsonlab/jeremy/gitrepos/palette-worktrees/shared-zarr-storage-policy-20260723-1d961583`

The shared `sun` checkout remained unchanged.

## First Resolution Run

LSF job `153192513` proved the named profile's physical resolution and all
logical/metadata invariants. Its immutable manifest retained the pre-promotion
field `crimson_physical_measurement_required=true`, however. That field is
incorrect for this post-measurement verification even though it does not alter
the arrays, profile, selectors, or result. The artifact remains unchanged and
selector-ineligible; a new versioned run supersedes its envelope.

## Published Evidence

LSF job `153192642` completed successfully on `h07u28`. The immutable workflow
is:

`/groups/johnson/johnsonlab/jeremy/recordings/.palette_benchmarks/refined_detection_storage/profile_canary/sleepyfish_promoted_default_verification_20260727_v2`

The corresponding macOS path is:

`/Volumes/johnsonlab/jeremy/recordings/.palette_benchmarks/refined_detection_storage/profile_canary/sleepyfish_promoted_default_verification_20260727_v2`

The strict `canary_manifest.json` has:

- canonical payload digest:
  `7fb7e28d99bc36bf8cfa1b7cfa1a4a2c01b28e59208a964ee8479d56e798b9f0`;
- file SHA-256:
  `2b302d40875d02c787d6511205d0f46c3aab793292d76917f52b4fa85af5d441`;
- `profile_promoted=true`;
- `crimson_physical_measurement_required=false`;
- `selector_eligible=false`;
- `registry_registered=false`; and
- no production-state changes.

## Result

The full-duration snapshot contains 1,188,000 frames, 1,187,087 instance rows,
1,187,087 source-audit rows, and 1,188,001 offset boundaries.

The persisted canonical and refined run manifests both carry
`detection_published_access_aware_v1`. The refined manifest additionally marks
the role `promoted_detection_snapshot_default` and status
`promoted_production_default`.

The publication gate passed:

- exact decoded logical hashes between regular and access-aware stores;
- exact canonical source-audit equality;
- offsets begin at zero, end at row count, and agree with row grouping;
- direct and consolidated metadata declarations are equivalent;
- codec and CRC declarations validate;
- planned payload objects are 240 regular versus 48 promoted, ratio 0.20;
- observed payload objects are 220 regular versus 42 promoted, ratio 0.1909;
- source copy took 5.15 seconds and the core workflow took 48.38 seconds; and
- process peak RSS was 690,626,560 bytes.

A separate fresh process then reopened the shared stores and validated all 9
canonical and 28 refined arrays. It proved:

- the promoted planners have the same chunk, shard, and codec signatures as
  the frozen access-aware evidence candidate;
- every persisted refined array's direct physical declaration matches the
  previously measured full-duration candidate;
- all logical hashes match the prior canary; and
- selector attributes remain absent.

The promoted profile therefore resolves to the approved physical layout. A
Crimson exact-schema open-only smoke may be run against this path, but another
performance matrix is neither required nor authorized by this checkpoint.
