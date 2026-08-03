# Guarded direct-writer byte-planning candidates (2026-08-03)

## Scope and decision

The maintained tail-posture v3 and bout-classification v2 direct writers now
have an explicit, opt-in byte-planned candidate path. The legacy path remains
the default and retains its existing physical layout and selector activation.
No analytics storage profile or production selector is promoted by this
change.

Passing a `StorageProfile` through the Python API, or one of the explicit CLI
`--storage-profile` choices, changes only that new run:

- every exact fixed-width array is planned by
  `fisheye.shared.zarr.analysis_storage_planning` from its actual dtype and
  complete logical row shape;
- arrays are created through the shared Zarr v3 array factory with the resolved
  bytes/Zstd/checksum/indexed-sharding metadata;
- the complete profile, declarations, observed shapes/dtypes, access units,
  chunks, shards, write ownership, object estimates, and canonical digest are
  persisted in `analysis_storage_plan_receipt`;
- validation recomputes the plan from the physical arrays and persisted profile,
  verifies the receipt digest and redundant profile/digest bindings, and
  compares every direct array declaration with the resolved plan; and
- the completed run remains `stage_selector_eligible=false` and does not update
  `latest` or `latest_complete`.

The persisted role is `explicit_unpromoted_candidate`. Omitting the profile
continues to use the established tail-posture mask-row chunks or shared
columnar chunks and then follows the existing guarded activation policy.

## Access units

Tail-posture arrays are windowed by observation. One access unit is one whole
observation record, including all keypoints, angles, or fixed-width reason
bytes in its trailing axes. Bout-classification arrays remain eager by
swim-bout row; wide 64- and 128-byte text records remain indivisible.

Consequently, a 1 MiB candidate target produces different row counts from
actual uncompressed bytes. Examples include 131,072 `uint64` identity rows,
16,384 64-byte rows, 8,192 128-byte rows, and 8,192 `float32[11,2]`
tail-keypoint rows. There is no family-wide row-count literal in the candidate
path.

## Validation evidence

Focused coverage includes:

- empty, 200,000-row, and 1,000,000-row plans;
- narrow scalar columns and wide fixed-record arrays;
- deterministic stage object estimates;
- a rehashed but tampered receipt that still fails executable-plan validation;
- a real Zarr v3 bout-classification candidate round trip; and
- equality of direct and consolidated array metadata and decoded values.

The compatibility lifecycle suites remain the rollback evidence: default
writes still activate only after their existing proof, and injected failures
retain immutable, selector-ineligible tombstones without changing a prior
authority.

## Remaining gate

These candidates are implementation evidence, not a profile recommendation.
Benchmark publication/read latency, bytes transferred, object counts, and the
real Palette/Crimson access workloads before authorizing either candidate
profile for selector-visible production runs.
