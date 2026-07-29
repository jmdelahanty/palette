# Keypoint, Body-Frame, And QC Storage Contract v2

<!-- contract-meta
status: logical-contract-foundation-implemented
schema: palette.stage.keypoint_observations v2
date: 2026-07-29
owner: jeremy
depends_on: docs/shared_coordinate_storage_contract_v1.md,
  docs/keypoint_heading_computation_contract.md,
  docs/body_frame_contract.md,
  docs/crop_pixel_work_package_contract.md
-->

## Decision

Palette separates landmark authority, source-bound diagnostic results, accepted
review state, and reusable derived orientation:

```text
keypoints_runs/<run>             immutable model observations + source facts
keypoint_quality_runs/<run>      immutable diagnostic metrics + policy proposals
refined_keypoints_runs/<run>     reviewed landmark authority + accepted review QC
analysis/body_frame_runs/<run>   derived orientation geometry including heading
```

The keypoint families do not persist heading arrays in v2. The skeleton retains
`pose_schema.metadata.heading_computation`, because that payload defines which
labeled landmarks determine anatomical orientation. A body-frame producer uses
that exact recipe and one exact raw or refined keypoint snapshot to materialize
orientation.

This boundary prevents a metric or threshold change from rewriting scientific
landmark coordinates, prevents an accepted keypoint edit from leaving an
embedded heading cache silently stale, and distinguishes an automated quality
proposal from the accepted review decision persisted by a refined snapshot.

`keypoint_quality_runs` is deliberately narrow rather than a catch-all. Version
1 permits only observation-local metrics that can be evaluated independently
for each raw keypoint row. Heading, frame-adjacent temporal metrics, trajectory
metrics, and track/subject continuity metrics are excluded because
`instance_key` identifies an observation, not a longitudinal animal. A later
temporal-quality schema must bind an explicit predecessor or track lineage.

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

The selector-ineligible raw-v2 publisher now enforces that envelope. Its
YOLO-facing preparation adapter converts the current float64 payload to the
exact float32 schema, derives `frame_row_offsets`, validity and row signatures,
recomputes source-camera projections from the bound crop-v2 geometry, and
records the maximum conversion/reprojection error. It deliberately drops
legacy heading, normalized-coordinate, count-alias, and embedded-QC families.
This is a canary boundary, not a production selector change.

## Keypoint Quality v1

`keypoint_quality_runs/<run>` is an immutable, selector-independent diagnostic
snapshot bound to exactly one raw keypoint-v2 run. It contains every source row
exactly once and in the source row order. It neither copies nor replaces
coordinates. A new quality algorithm, metric definition, or threshold policy
creates a new quality run; it never mutates keypoints or an older quality run.

### Core arrays

| Path | Exact dtype and shape | Role |
| --- | --- | --- |
| `instance_key` | `uint64[N]` | Exact source observation identity |
| `source_keypoint_row_ids` | `int64[N]` | Exact source row IDs; v1 is precisely `arange(N)` |
| `source_keypoint_row_signature` | `uint8[N,32]` | Exact source landmark-row signatures |
| `frame_indices` | `int64[N]` | Exact source recording-frame index |
| `frame_row_offsets` | `int64[F+1]` | Exact CSR frame-to-quality-row index |
| `keypoint_metric_values` | `float32[N,K,Q]` | Ordered observation-local per-landmark metrics |
| `keypoint_metric_valid` | `bool[N,K,Q]` | Exact finite-value mask; invalid values are NaN |
| `pose_metric_values` | `float32[N,P]` | Ordered observation-local row metrics |
| `pose_metric_valid` | `bool[N,P]` | Exact finite-value mask; invalid values are NaN |
| `keypoint_quality_flags` | `uint16[N,K]` | Bitwise findings from the keypoint flag registry |
| `pose_quality_flags` | `uint16[N]` | Bitwise findings from the pose flag registry |
| `proposed_keypoint_valid` | `bool[N,K]` | Automated policy proposal; cannot resurrect an invalid source landmark |
| `proposed_pose_usable` | `bool[N]` | Automated policy proposal; cannot resurrect a failed source pose |

The digest-bound profile declares the exact ordered keypoint and pose metric
catalogs, each metric's version, units, directionality, and description, both
single-bit flag registries, and the policy digest. The profile document has its
own canonical SHA-256 digest. Zero flags mean no finding; undeclared bits are
invalid. Metric IDs containing heading, temporal, trajectory, or track terms
are rejected by v1 rather than being computed using adjacent sparse rows.

The first profile may include observation-local confidence-margin,
single-view pose-plausibility, ensemble-disagreement, geometry, or valid-point
coverage metrics. Lightning Pose-style temporal norm is intentionally deferred
until a source provides stable longitudinal lineage. Pixel error against
labeled ground truth remains a model/training evaluation artifact rather than
a recording-level observation-quality field.

The quality manifest must bind the exact source keypoint run and manifest
digest, source skeleton and row signatures, complete metric/policy profile,
logical schema, physical storage plan, array digests, and direct/consolidated
metadata equivalence. A quality run is not landmark authority and cannot be a
training label source by itself.

### Initial implemented producer

The selector-ineligible v1 producer implements one deliberately small
`observation_local_baseline` profile:

- `confidence_margin` per keypoint is source confidence minus the exact policy
  threshold;
- `valid_landmark_fraction` per pose is the fraction retained by that policy;
- keypoint flags distinguish low confidence from source invalidity;
- pose flags distinguish source failure from insufficient retained landmarks;
- proposed validity can only remove source validity, never create it.

The policy document and complete profile are independently digest-bound. The
publisher creates arrays only through the shared byte planner and array
factory, writes whole physical units, validates decoded values and source
signatures, consolidates metadata, persists the manifest at
`keypoint_quality_runs/<run>/zarr.json.attributes.run_manifest`, reconsolidates,
and reopens the result through the complete publication gate.

This first publisher writes only standalone selector-ineligible shadows. It
does not create a registry status row, selector, production authority, or
training artifact. It initially uses `published_http_v1`; a representative
benchmark must decide whether keypoint-quality deserves a distinct promoted
profile.

## Refined Keypoint v2

A compact `refined_keypoints_runs/<run>` v2 snapshot binds the exact raw
keypoint run and the quality run whose proposals were reviewed. It reuses the
shared identity, lineage, coordinate, confidence, validity, bbox, and signature
fields. It replaces raw-only `pose_success` with the unambiguous pair
`source_success` and `refined_success`, then adds authoring provenance including
exact parent snapshot identity, per-landmark edit flags, review state,
acceptance reasons, and accepted edit/delta digests. It does not add heading.
It does retain the exact QC facts used to validate or approve that snapshot.

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
The v1 keypoint body-frame producer resolves landmark indices from the ordered
pose-schema labels. It does not accept a run override, deprecated alias, or
caller-supplied hard-coded positions as canonical provenance.

Masks are not required for that estimator. A future mask/spline or hybrid
body-frame estimator may publish the same logical output arrays only under a
new estimator ID/version with exact mask inputs, polarity evidence, validity
rules, and digests. Different estimators are never silently substituted or
declared scientifically equivalent merely because their output shapes match.

## QC Boundary

QC is classified by what it describes, not by placing every metric in one
generic stage:

1. **Source observations** live with the raw keypoint row: per-keypoint
   confidence and validity, pose confidence, and producer success.
2. **Automated diagnostic measurements and proposals** live in the immutable
   quality run. They remain source-bound evidence and do not become accepted
   review state merely because a threshold passed.
3. **Review and promotion decisions** live with the refined snapshot they
   qualify: edit flags, source/refined success, review state and reason,
   confidence validity, geometry validity, and combined usability.
4. **Orientation geometry** lives in the body-frame run: axis validity and the
   mechanically derived heading cache. `axis_valid` says that the estimator
   resolved geometry; it is not a substitute for refined-pose acceptance.
5. **Longitudinal diagnostics** are recomputable analysis: coordinate or
   heading jumps, speed/acceleration/jerk, skeleton-length or angle drift,
   dropout streaks, deviations from a temporally smoothed trajectory, and
   identity/track discontinuities. They may be computed on demand. If later
   persisted, they require a narrowly named, versioned analysis contract with
   exact source snapshot/body-frame/track bindings and policy digests.

Accepted snapshot-local QC must be available without locating the quality run.
The quality run explains and reproduces the automated proposal; the refined
snapshot freezes what was accepted. A changed quality policy creates a new
quality run and, if accepted decisions change, a new refined snapshot. It does
not mutate either older artifact.

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
- Review UI may lazily read the quality run for diagnostic detail, while the
  selected refined snapshot supplies the final accepted review state.
- A body-frame selection must be explicitly bound to the selected keypoint
  snapshot. An invalid explicit selection fails; it never falls back silently.
- Track kinematics may consume the observation body frame, then persist its own
  track-sample heading/interpolation products. Observation `instance_key` is not
  a longitudinal subject or track identity.
- Training exports use reviewed keypoints. Heading is recomputed from the bound
  skeleton recipe or included only as an explicitly derived auxiliary target.

### DAG and registry boundary

The first archive-native quality node must accept a run name, not caller-built
source dictionaries. It resolves and validates the raw-keypoint-v2 manifest,
crop/model/skeleton bindings, policy, and destination itself, then returns a
receipt containing the source, quality-manifest, policy, profile, and logical
content digests plus phase timings. Until raw-v2 is persisted, the quality
publisher remains a standalone benchmark shadow rather than a production DAG
edge.

The existing SQLite `keypoint_quality` table is a legacy summary extracted from
`refined_keypoints_runs`; it is not an index of the new
`keypoint_quality_runs` artifacts. Do not write new quality runs through that
table or silently change its meaning. If operational discovery later needs an
artifact index, add a distinctly named, manifest-keyed surface such as
`keypoint_quality_artifacts`.

## Storage And Publication

- Raw keypoint runs, quality runs, compact refined snapshots, and body frames
  are immutable and use byte-planned indexed shards.
- Edit deltas are small append/write-optimized artifacts; compaction produces a
  new immutable sharded snapshot.
- Inner chunks are selected by uncompressed bytes and access class, not one
  global frame/row count.
- Row-aligned hot columns should share chunk boundaries when consumers read
  them together; the exact row size remains benchmark-gated.
- `frame_row_offsets` is classified eager and retained once by Crimson.
- Quality columns use the exact raw-keypoint row grid. Refined accepted-QC
  columns use the refined row grid. Body-frame hot columns are eligible for
  byte-budgeted background residency.
- Every current publication is Zarr v3 with the approved codec/checksum profile,
  exact consolidated metadata, a versioned run manifest, and no dtype probing.
- Whole-shard writer ownership is required for parallel publication.

## Edit And Invalidation Lifecycle

Initial production order is raw keypoints, one bound quality snapshot, one
reviewed refined snapshot, then body-frame materialization from the selected
refined authority. Quality runs are not authority selectors themselves.

1. Write a keyed keypoint edit delta; never edit heading directly.
2. Validate source snapshot, `instance_key`, landmark label, and row signature.
3. Compact into a new immutable refined-keypoint snapshot.
4. Recompute and validate the refined snapshot's accepted intrinsic/review QC;
   do not rewrite the source-bound raw quality run.
5. Mark body-frame and optional longitudinal products bound to the old snapshot
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
- [x] Freeze and implement exact source-bound keypoint-quality arrays, ordered
      metric catalogs, bit registries, policy digest, and cross-array checks.
- [x] Register the keypoint-quality artifact family and dependency/invalidation
      edges without activating production status or selection.
- [ ] Freeze exact refined accepted-QC code maps and manifest bindings to the
      quality run used during review.
- [x] Implement body-frame-v1 logical contracts and derivation validation.
- [x] Require exact quality manifest field sets, reconstruct the logical and
      storage builders, and enforce canonical digests.
- [x] Add the exact body-frame run manifest and publication reconstruction
      gate.
- [x] Add the exact raw-keypoint-v2 run manifest and publication reconstruction
      gate.
- [ ] Add the equivalent exact run manifest for refined keypoints.

### Writer and lifecycle

- [x] Add the selector-ineligible keypoint-quality shadow writer without
      changing current defaults.
- [x] Add the observation-local keypoint-quality producer and immutable
      publication gate.
- [x] Freeze the exact raw-keypoint-v2 run manifest, byte-derived storage plan,
      selector-ineligible publisher, and legacy-YOLO preparation adapter.
- [x] Validate a deterministic selector-ineligible YOLO-shaped raw-v2 canary,
      including a mask-free keypoint-to-body-frame chain.
- [ ] Publish and measure one representative selector-ineligible YOLO canary
      from a real completed run before inserting the quality DAG node.
- [ ] Make clipped finalization and later delta compaction publish the same
      raw/refined v2 contracts rather than parallel layouts.
- [ ] Add a bounded-row DAG materializer that accepts only a validated
      raw-keypoint-v2 manifest, writes complete destination physical units,
      and does not retain duplicate full source/output tables.
- [x] Add the body-frame-v1 producer, exact manifest, and byte-planned
      selector-ineligible shadow publisher.
- [ ] Add the bounded-row body-frame DAG node after refined-keypoint selection.
- [ ] Keep legacy embedded heading readable but never copy it into v2; migrate
      by recomputation into a new body-frame run after numerical/consumer gates.
- [ ] Add keyed refined-keypoint deltas and immutable compaction.
- [ ] Recompute accepted snapshot-local QC during compaction and body frame
      after it.
- [ ] Preserve source crop, pixel-package, skeleton, model, and coordinate
      provenance through every publication.
- [ ] Keep production selectors unchanged until all gates pass.

### Numerical and storage gates

- [ ] Compare v1 float64 and v2 float32 landmark/source-camera projections,
      editing round trips, and derived heading.
- [x] Add one deterministic compute, publication, retained-offset,
      random-frame, 70-frame-window, and full-scan integration benchmark.
- [ ] Repeat publication/read measurement on selector-ineligible
      representative short and full-duration raw-keypoint-v2 sources.
- [ ] Benchmark refined compaction and training-export workloads.
- [ ] Benchmark common-row chunk alignment for hot keypoint/body-frame columns.
- [x] Record requested and effective chunk/shard shapes, object estimates,
      publication phases, physical file statistics, and peak process RSS.

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

## Deterministic Publication/Read Checkpoint

The first bounded integration benchmark ran on 2026-07-29 using 23,287 frames,
22,926 observations, five landmarks, 365 empty frames, and four frames with two
observations. It passed publication validation, direct/consolidated manifest
equality, exact retained offsets, deterministic random-frame and 70-frame
window digests, and all-array full-scan digests.

Local `/tmp` results on the Palette workstation were:

| Measurement | Result |
| --- | ---: |
| Source validation + quality computation | 72.2 ms |
| Validated publication | 225.1 ms |
| Physical payload objects | 13 |
| Apparent archive bytes | 1,596,377 |
| Retained offset load | 1.61 ms / 186,304 bytes / exactly one read |
| Random-frame median / p95 | 8.92 / 9.73 ms |
| 70-frame-window median / p95 | 8.96 / 9.61 ms |
| Full scan | 14.23 ms / 172.27 MiB/s |
| Peak RSS delta from fresh driver entry | 30,273,536 bytes |

This is integration evidence only. All 13 quality arrays fit in one payload
object apiece at this row count, so the run does not exercise indexed sharding,
remote filesystem behavior, cache pressure, or full-duration object counts.
It cannot promote `published_http_v1` or choose a distinct quality profile.
Representative evidence first requires a persisted raw-keypoint-v2 source with
an exact manifest; using a legacy float64/embedded-heading run as though it were
v2 would weaken the contract. The benchmark driver intentionally marks its
deterministic source and result as selector-ineligible and promotion-ineligible.

The same driver also passed a synthetic full-duration-shape checkpoint with
1,188,000 frames, 1,187,087 observations, 929 empty frames, and 16 two-row
frames. This run did exercise the physical plan: all 13 arrays were
indexed-sharded into 14 payload objects. It wrote 132,961,056 logical bytes as
68,234,804 apparent bytes and completed validated publication in 4.01 seconds
on local ext4. The 9,504,008-byte retained offset index loaded once in 9.96 ms;
random-frame median/p95 was 33.76/35.20 ms, 70-frame-window median/p95 was
32.28/34.46 ms, and a complete scan took 751.7 ms at 168.69 MiB/s.

That checkpoint is physical-scale integration evidence, not representative
scientific-data or PRFS/macOS evidence. It also exposed a workflow concern:
the fresh process peaked at 1,089,241,088 bytes RSS, 915,832,832 bytes above
driver entry. Production DAG execution should read and compute bounded row
blocks instead of retaining the synthetic source, copied source evidence, and
complete prepared output simultaneously. Streaming must preserve complete
row units and single-writer ownership of each physical shard.
