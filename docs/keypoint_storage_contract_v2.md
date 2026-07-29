# Keypoint, Body-Frame, And QC Storage Contract v2

<!-- contract-meta
status: decision-frozen-implementation-pending
schema: palette.stage.keypoint_observations v2
date: 2026-07-29
owner: jeremy
depends_on: docs/shared_coordinate_storage_contract_v1.md,
  docs/keypoint_heading_computation_contract.md,
  docs/body_frame_contract.md,
  docs/crop_pixel_work_package_contract.md
-->

## Decision

Palette separates landmark authority from reusable derived orientation while
keeping intrinsic and review QC with the snapshot it qualifies:

```text
keypoints_runs/<run>            immutable model observations + source QC
refined_keypoints_runs/<run>    reviewed landmark authority + review QC
analysis/body_frame_runs/<run>  derived orientation geometry
```

The keypoint families do not persist heading arrays in v2. The skeleton retains
`pose_schema.metadata.heading_computation`, because that payload defines which
labeled landmarks determine anatomical orientation. A body-frame producer uses
that exact recipe and one exact raw or refined keypoint snapshot to materialize
orientation.

This boundary prevents a threshold change from rewriting scientific landmark
coordinates, prevents an accepted keypoint edit from leaving an embedded
heading cache silently stale, and keeps every acceptance flag beside the
coordinate snapshot whose fitness it describes.

Palette does not introduce a required catch-all `keypoint_quality_runs` family
in this contract. Longitudinal diagnostics may be computed on demand or later
published under an explicitly scoped `analysis/pose_qc_runs/<run>` contract
when a real consumer and retention policy justify it. Such a future run is not
landmark authority and cannot silently replace snapshot-local review state.

## Current Compatibility Surface

Historical and current v1 runs persist some or all of:

- `heading`, `heading_finite`, and `heading_usable`;
- `heading_delta_prev_deg`, `heading_delta_next_deg`, and
  `heading_temporal_outlier`;
- triangle geometry and quality arrays beside landmark coordinates; and
- `float64` keypoint coordinates, confidences, and heading.

Those arrays remain readable through explicit v1 adapters. They are not part of
the v2 keypoint schema and must not be copied into a v2 keypoint snapshot merely
to preserve layout compatibility. Migration recomputes and validates derived
products into their new families; it never rewrites a historical run in place.

## Keypoint Observation v2

### Authority and coordinates

`keypoints_roi` is the authoritative landmark coordinate payload. It uses
continuous ROI-local pixels and is bound to one immutable crop snapshot and its
exact ROI-to-source-camera transform. `keypoints_img` is a required exact
source-camera-pixel projection cache for consumers and validation.

`keypoints_norm` is omitted from v2. A consumer that requires normalized values
derives them from the exact source-camera extent. Heading must never be computed
from normalized keypoints because non-square normalization can distort angles.

The initial v2 dtype is exact `float32`. Current `float64` remains the v1
representation. Writer activation requires a numerical comparison proving that
the float32 projection and editing round trip remain comfortably below the
accepted pixel-error budget.

### Core arrays

| Path | Exact dtype and shape | Role |
| --- | --- | --- |
| `instance_key` | `uint64[N]` | Stable observation/edit-lineage identity; not a subject ID |
| `source_crop_row_ids` | `int64[N]` | Exact row in the bound crop snapshot |
| `source_acquisition_frame_index` | `int64[N]` | Full acquisition-frame identity |
| `frame_indices` | `int64[N]` | Nondecreasing recording-frame lookup domain |
| `frame_row_offsets` | `int64[F+1]` | Exact CSR frame-to-keypoint-row index |
| `source_crop_row_signature` | `uint8[N,32]` | Exact crop-input compatibility signature |
| `keypoint_row_signature` | `uint8[N,32]` | Signature of the landmark row, skeleton, validity, and source binding |
| `keypoints_roi` | `float32[N,K,2]` | Authoritative ROI-local continuous coordinates |
| `keypoints_img` | `float32[N,K,2]` | Exact derived source-camera continuous coordinates |
| `keypoint_confidences` | `float32[N,K]` | Per-landmark source confidence in skeleton order |
| `keypoint_valid` | `bool[N,K]` | Explicit landmark validity; invalid coordinates/confidences use NaN |
| `pose_confidence` | `float32[N]` | Source model's row-level pose score |
| `pose_success` | `bool[N]` | Whether the source producer resolved a pose row |
| `pose_bbox_xyxy_roi` | `float32[N,4]` | Source pose box in ROI-local continuous half-open edges |
| `pose_bbox_xyxy_img` | `float32[N,4]` | Exact source-camera projection of the pose box |

All row arrays have the same order and `N`. `instance_key` values are unique.
Rows are contiguous in nondecreasing `frame_indices` order. Repeated adjacent
offsets represent empty frames; a range longer than one represents multiple
subjects or observations in the frame.

The run manifest binds the ordered landmark labels, skeleton ID and digest,
coordinate catalog, crop manifest, pixel contract/package when used, model and
preprocessing identities, exact logical schema, physical storage plan, and
consolidated/direct metadata equivalence.

### Refined keypoint snapshots

A compact `refined_keypoints_runs/<run>` v2 snapshot reuses the shared identity,
lineage, coordinate, confidence, validity, bbox, and signature fields. It
replaces raw-only `pose_success` with the unambiguous pair `source_success` and
`refined_success`, then adds authoring provenance including exact parent
snapshot identity, per-landmark edit flags, review state, acceptance reasons,
and accepted edit/delta digests. It does not add heading. It does retain the
exact QC facts used to validate or approve that snapshot.

The initial refined QC surface is:

| Path | Exact dtype and shape | Role |
| --- | --- | --- |
| `source_success` | `bool[N]` | Source observation was usable before refinement |
| `refined_success` | `bool[N]` | Refined row is accepted as a usable pose |
| `keypoint_edit_flags` | `bool[N,K]` | Landmark coordinates changed from the parent |
| `flip_corrected` | `bool[N]` | Anatomical label/polarity correction was applied |
| `confidence_valid` | `bool[N]` | Snapshot's declared confidence rule passed |
| `geometry_valid` | `bool[N]` | Snapshot's declared skeleton-geometry rule passed |
| `usable_keypoints` | `bool[N]` | Combined review/promotion usability result |
| `review_state_codes` | `uint8[N]` | Controlled review-state registry |
| `reason_codes` | `uint16[N]` | Controlled acceptance/rejection reason registry; zero means none |

The manifest contains exact controlled maps and digests for both code arrays.
If promotion depends on quantitative skeleton measurements such as edge
lengths, angles, or named derived metrics, those measurements are also
snapshot-local arrays bound to an exact metric schema. Profile-specific
triangle arrays remain a v1/traditional compatibility surface rather than
universal v2 fields.

The observation remains identified by `instance_key`; the landmark within a row
is identified by the digest-bound ordered skeleton label. A separate refined
row ID is unnecessary while the contract permits exactly one keypoint row per
observation. Detection additions first create a new detection/crop observation
and therefore a new `instance_key` before keypoint inference or manual labeling.

Edits are keyed deltas. Accepted deltas compact into a new immutable sharded
snapshot. Training exports cite that compact snapshot and materialize their own
dense image/label representation; they do not treat an edit delta as training
authority.

## Body-Frame v1

Reuse pressure from Crimson rendering, eye angles, tracking, subject shape, and
bout analyses now justifies a reusable `analysis/body_frame_runs/<run>` family.
A body-frame run is immutable and binds exactly one source keypoint snapshot or
another approved estimator source.

### Arrays

| Path | Exact dtype and shape | Role |
| --- | --- | --- |
| `instance_key` | `uint64[N]` | Exact source observation identity |
| `source_keypoint_row_ids` | `int64[N]` | Exact row in the bound keypoint snapshot |
| `source_keypoint_row_signature` | `uint8[N,32]` | Exact input landmark-row signature |
| `frame_indices` | `int64[N]` | Recording-frame domain |
| `frame_row_offsets` | `int64[F+1]` | CSR frame-to-body-frame-row index |
| `origin_xy` | `float32[N,2]` | Anatomical origin in source-camera pixels |
| `forward_axis_xy` | `float32[N,2]` | Unit vector from posterior toward anterior |
| `left_axis_xy` | `float32[N,2]` | Unit vector toward anatomical left |
| `axis_valid` | `bool[N]` | Geometry and polarity resolved by the estimator |
| `heading_deg` | `float32[N]` | Required derived cache: `atan2(-fy, fx)` in degrees |

Invalid rows use `axis_valid=false` and NaN for all geometry and heading arrays.
`heading_deg` is not body-frame authority by itself: publication recomputes it
from `forward_axis_xy`, validates the fixed angular convention, and digest-binds
both arrays and row identity.

For the keypoint estimator, the manifest binds the exact source snapshot,
`pose_schema.metadata.heading_computation`, its dependency labels and digest,
the source-camera coordinate descriptor, estimator version, and all output
digests. A run may use mask, spline, or hybrid inputs only through another
explicit estimator profile; it cannot relabel those outputs as keypoint-derived.

## QC Boundary

QC is classified by what it describes, not by placing every metric in one
generic stage:

1. **Source observations** live with the raw keypoint row: per-keypoint
   confidence and validity, pose confidence, and producer success.
2. **Review and promotion decisions** live with the refined snapshot they
   qualify: edit flags, source/refined success, review state and reason,
   confidence validity, geometry validity, and combined usability.
3. **Orientation geometry** lives in the body-frame run: axis validity and the
   mechanically derived heading cache. `axis_valid` says that the estimator
   resolved geometry; it is not a substitute for refined-pose acceptance.
4. **Longitudinal diagnostics** are recomputable analysis: coordinate or
   heading jumps, speed/acceleration/jerk, skeleton-length or angle drift,
   dropout streaks, deviations from a temporally smoothed trajectory, and
   identity/track discontinuities. They may be computed on demand. If later
   persisted, they require a narrowly named, versioned analysis contract with
   exact source snapshot/body-frame/track bindings and policy digests.

Snapshot-local QC must be available without locating a second run. A changed
acceptance policy creates a new refined snapshot or an explicit review
artifact; it does not mutate an immutable snapshot. A changed exploratory
temporal threshold can produce a new diagnostic result without changing the
refined landmark authority.

The current embedded `heading_delta_prev_deg`, `heading_delta_next_deg`, and
`heading_temporal_outlier` arrays remain v1 compatibility fields. Before v2
production activation, their real consumers must be censused. A diagnostic
that is part of the acceptance gate moves into the refined snapshot under an
exact policy; an exploratory diagnostic is recomputed or published later under
the optional analysis boundary.

Model evaluation against labeled ground truth is a separate level of QC. It
includes per-keypoint pixel error/RMSE, confidence-thresholded error, PCK or
OKS-style measures, and precision/recall summaries. Those are dataset/model
evaluation artifacts, not per-observation authority fields.

## Consumer Contract

- Ordinary Crimson rendering opens keypoints and the selected body-frame run.
- Review UI reads intrinsic/review QC from the selected raw or refined
  keypoint snapshot. Optional longitudinal diagnostic products remain lazy.
- A body-frame selection must be explicitly bound to the selected keypoint
  snapshot. An invalid explicit selection fails; it never falls back silently.
- Track kinematics may consume the observation body frame, then persist its own
  track-sample heading/interpolation products. Observation `instance_key` is not
  a longitudinal subject or track identity.
- Training exports use reviewed keypoints. Heading is recomputed from the bound
  skeleton recipe or included only as an explicitly derived auxiliary target.

## Storage And Publication

- Raw keypoint runs, compact refined snapshots, and body frames are immutable
  and use byte-planned indexed shards.
- Edit deltas are small append/write-optimized artifacts; compaction produces a
  new immutable sharded snapshot.
- Inner chunks are selected by uncompressed bytes and access class, not one
  global frame/row count.
- Row-aligned hot columns should share chunk boundaries when consumers read
  them together; the exact row size remains benchmark-gated.
- `frame_row_offsets` is classified eager and retained once by Crimson.
- Snapshot QC columns share the row grid used by the keypoint payload they
  qualify. Body-frame hot columns are eligible for byte-budgeted background
  residency.
- Every current publication is Zarr v3 with the approved codec/checksum profile,
  exact consolidated metadata, a versioned run manifest, and no dtype probing.
- Whole-shard writer ownership is required for parallel publication.

## Edit And Invalidation Lifecycle

1. Write a keyed keypoint edit delta; never edit heading directly.
2. Validate source snapshot, `instance_key`, landmark label, and row signature.
3. Compact into a new immutable refined-keypoint snapshot.
4. Recompute and validate the refined snapshot's intrinsic/review QC.
5. Mark body-frame and optional diagnostic products bound to the old snapshot
   stale.
6. Publish a new body-frame run from the new snapshot.
7. Publish any explicitly required longitudinal diagnostic artifact.
8. Activate selectors only after logical, storage, consolidation, and consumer
   gates pass.

If landmarks are scientifically correct but anatomical polarity requires a
manual override, that is a separately versioned body-frame correction input.
It must not be encoded by silently changing `heading_deg`.

## Implementation Checklist

### Logical schemas

- [x] Add exact float32 keypoint-v2 array contracts and coordinate bindings.
- [x] Implement the raw/refined shared stage schema and cross-array invariants.
- [ ] Freeze exact snapshot-local raw/refined QC arrays, code maps, and policy
      digests.
- [x] Implement body-frame-v1 logical contracts and derivation validation.
- [ ] Require exact manifest field sets and canonical digests.

### Writer and lifecycle

- [ ] Add selector-ineligible shadow writers; do not change current defaults.
- [ ] Add keyed refined-keypoint deltas and immutable compaction.
- [ ] Recompute snapshot-local QC during compaction and body frame after it.
- [ ] Preserve source crop, pixel-package, skeleton, model, and coordinate
      provenance through every publication.
- [ ] Keep production selectors unchanged until all gates pass.

### Numerical and storage gates

- [ ] Compare v1 float64 and v2 float32 landmark/source-camera projections,
      editing round trips, and derived heading.
- [ ] Benchmark writer, compaction, publication, random-frame, window, full-read,
      and training-export workloads.
- [ ] Benchmark common-row chunk alignment for hot keypoint/body-frame columns.
- [ ] Record requested and effective chunk/shard shapes and object estimates.

### Consumer and migration

- [ ] Add Crimson exact-schema adapters for keypoint v2, body-frame v1, and
      the snapshot-local QC surface.
- [ ] Census consumers of current embedded temporal-heading diagnostics and
      decide which are acceptance inputs versus optional analysis.
- [ ] Prove ordinary playback performs zero optional diagnostic reads.
- [ ] Publish a selector-ineligible cross-language canary.
- [ ] Add a migration tool that recomputes legacy embedded heading into a new
      body-frame run and proves value equivalence within the declared dtype
      tolerance.
- [ ] Retain explicit v1 compatibility adapters without weakening v2.

## Promotion Gate

No production writer or selector changes in the contract-foundation phase.
Promotion requires exact logical validation, stable multi-observation frame
lookup, edit/compaction correctness, numerical acceptance, direct/consolidated
metadata equivalence, supported codecs, Crimson correctness, and benchmarked
read/write/publication behavior on representative short and full recordings.
