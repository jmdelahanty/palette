# Recording-Level Sampled Subject-Mask Contours

Date: 2026-08-01

Status: Palette full-duration selector-ineligible canary passed; Crimson
mounted-read and visual-equivalence gate pending.

## Decision

Fixed-count sampled contours are the canonical **derived presentation surface**
for subject-mask outlines. They are not a pixel, edit, scientific, or training
authority. Dense `refined_subject_masks_runs/<run>/masks_roi` remains the sole
modern authority for those uses.

The cache is an independent immutable run:

```text
subject_mask_cache_runs/<cache_run>/
  components/<component>/sampled_contours/
    points_xy          float32 [N,K,2]
    valid                 bool [N]
    source_point_count   int32 [N]
```

The default component sample counts are:

| Component | K |
|---|---:|
| `subject_body` | 128 |
| `eye_left` | 64 |
| `eye_right` | 64 |
| `swim_bladder` | 32 |

Each valid row is a clockwise, canonical-start, closed-arc-length sample of the
largest external contour in ROI pixel coordinates. Invalid rows use
`valid=false` and all-NaN points. `source_point_count` always records the
observed finite source-vertex count (zero when no contour exists); a count below
two cannot produce a valid sample. Full ragged contours remain an optional cold
inspection/export cache and are not required for the viewer profile.

## Why It Is a Separate Run

A refined dense run is immutable once sealed. Adding contours later would
mutate its metadata and invalidate its publication proof. An independent cache
run can instead be regenerated, benchmarked, replaced, or omitted without
changing the dense scientific identity.

Every cache manifest binds:

- the exact refined run and manifest digests;
- the dense `masks_roi` logical-value digest;
- the component-registry digest;
- `instance_key`, crop-row, acquisition-frame, `frame_row_offsets`, and crop
  placement digests;
- all twelve cache-array logical hashes;
- the exact storage plan and direct/consolidated metadata declarations;
- one freshness receipt per component proving full dense-derived generation.

The recording-level publisher currently computes contours in bounded dense-row
blocks on local scratch, stages the 2.53 GiB logical result through disk-backed
memmaps, and writes complete immutable output shards from a single owner.
Dense-row contour extraction may use bounded worker processes, but workers
return only disjoint row blocks to node-local memmaps; they never write Zarr.
It never materializes the full contour surface in heap memory.

The intended successor avoids repeating that extraction when refinement
workers already produced exact fixed-count contours. Each worker will bind its
row-local contour arrays to the exact dense-mask unit digest, component
registry, sampling algorithm/version, sample count, winding, and canonical
start rule. The recording finalizer will assemble those rows in canonical crop
order and bind the cache run to the final dense authority. It will regenerate
only missing/stale rows, or all rows after a dense edit or sampling-contract
change. There is no cross-clip scientific reducer for an observation-local
contour.

## Physical Candidate

`subject_mask_presentation_candidate_v1` derives layout from uncompressed
bytes, not from one shared row constant:

- inner target: 128 KiB;
- outer indexed shard target/ceiling: 8 MiB;
- codec: little-endian bytes + Zstd level 0 with compressor checksum disabled,
  plus a little-endian/CRC32C shard index at end;
- Zarr format: v3;
- sharding axis: observation rows only.

For the 1,169,010-row Sleepyfish candidate:

| Array class | Inner shape | Outer shape | Payload objects |
|---|---:|---:|---:|
| body points, K=128 | `[128,128,2]` | `[8192,128,2]` | 143 |
| each eye points, K=64 | `[256,64,2]` | `[16384,64,2]` | 72 each |
| swim-bladder points, K=32 | `[512,32,2]` | `[32768,32,2]` | 36 |
| each `valid` | `[131072]` | `[1179648]` | 1 |
| each `source_point_count` | `[32768]` | `[1179648]` | 1 |

Total logical bytes are 2,716,779,240 (2.53 GiB), represented by an estimated
331 payload objects plus group/array metadata. The profile remains a candidate
until Crimson measures mounted reads; changing the physical profile later does
not change contour semantics.

## Bundle Lifecycle

- Bundle v1/v2 remains readable and contains raw, refined, and quality members.
- Bundle v3 contains exactly four members: raw, refined, quality, and
  `presentation_cache`.
- Cache generation is opt-in through `--cache-run` while the profile is under
  evaluation; existing publication commands continue to emit v2.
- A v3 bundle proves the cache and refined dense member share exact dimensions,
  components, dense hash, manifest identity, and row identity before importing
  any member.
- The cache and bundle remain selector-ineligible unless the existing explicit
  bundle activation transaction succeeds.
- A dense edit creates a new refined snapshot. Its old contour cache remains
  immutable historical evidence and cannot be rebound; a new cache run must be
  generated for the new dense authority.

## Crimson Read Contract

For frame `f`, Crimson should:

1. use the refined member's retained `frame_row_offsets[f:f+2]`;
2. read the corresponding row window from the requested component's
   `points_xy` and `valid` arrays;
3. retain every row in the interval, including multiple subjects in one frame;
4. transform ROI-pixel points through the crop-v2 placement bound by the same
   `instance_key`/crop-row identity;
5. draw the contour as closed without requiring a duplicated closing point.

`source_point_count` is primarily an inspection/QC companion and need not be
opened for ordinary overlays. Crimson must not fall back from an invalid
explicit bundle-v3 cache to an unrelated contour source.

## Implementation Checklist

- [x] Freeze fixed-K dtype, shape, coordinate, invalid-row, winding, and start
  semantics.
- [x] Add a byte-derived shared storage planner for all twelve arrays.
- [x] Add bounded dense-to-contour generation on local scratch.
- [x] Parallelize only disjoint node-local contour blocks while retaining one
  whole-shard Zarr writer.
- [x] Create arrays only through the shared policy-owned Zarr factory.
- [x] Add per-component dense-source receipts and logical hashes.
- [x] Validate exact direct/consolidated metadata and boundary samples.
- [x] Add bundle-v3 import, cross-binding, activation, and rollback behavior.
- [x] Preserve v1/v2 read compatibility and opt-in publication.
- [x] Add recomputed-digest tampering and wrong-authority rejection tests.
- [x] Publish one full-duration selector-ineligible cache canary without
  duplicating the already-accepted bundle-v2 source.
- [x] Measure cache publication time, peak RSS, physical bytes, and object
  count.
- [ ] Run Crimson exact-schema/open, random frame, 70-frame window, traversal,
  cancellation, RSS, and Metal visual-equivalence gates.
- [ ] Promote or revise the physical profile from that evidence.
- [ ] Make bundle v3 the production publisher default only after promotion.
- [ ] Carry worker-produced sampled contours through strict terminal receipts
  and assemble them without a second dense-mask extraction pass.
- [ ] Make full ragged contour publication opt-in in the new default profile
  while retaining historical read/migration compatibility.
- [ ] Add compact bitpacked/RLE recording-level members in a later independent
  cache profile; do not couple them to contour promotion.

## Explicit Non-Goals

- No mutation of the completed 2026-07-31/2026-08-01 full-duration canary.
- No selector or registry change in this checkpoint.
- No claim that a largest-external sampled outline preserves holes or every
  disconnected island. Exact topology remains available from dense masks until
  a future multi-ring contour contract is justified.
