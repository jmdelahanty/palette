# Position, Body-Frame, and Motion Provider Design

<!-- contract-meta
version: 1
status: draft
implementation: partial
last_updated: 2026-08-17
-->

Purpose: define how Palette should materialize and select subject position,
heading/body frame, speed, and angular-motion surfaces without treating a
detection bounding-box centroid as the only possible biological position.

This design extends the composable stimulus-analysis work. It does not change
existing immutable detection, keypoint, subject-mask, body-frame, or
track-kinematics runs. Existing detection-centroid products remain valid under
their producing contracts.

## Decisions

1. Position, body frame, and motion are separate scientific surfaces.
2. A shared, versioned anatomy profile defines stable biological roles and
   reusable named point or axis recipes independently of measurement modality.
3. A pose, mask, or other source schema explicitly binds its native labels to
   anatomy-profile roles and may advertise recipe capability. Matching label
   strings alone do not prove semantic compatibility, and a source schema does
   not redefine a shared recipe.
4. A versioned estimator profile selects an exact recipe, source modality,
   validity policy, and coordinate authority.
5. Materialized values live in derived runs and bind the exact estimator,
   source arrays, row identity, timeline, coordinate descriptors, and semantic
   schema digests.
6. Speed is derived from one exact position series. Angular velocity and
   angular speed are derived from one exact heading/body-frame series.
7. Position and body frame may use different source modalities when their row,
   time, and coordinate authorities are explicitly compatible.
8. A body-frame origin is not automatically the subject position used for
   movement. The current eye-pair midpoint body-frame origin and a
   three-anchor position mean are different points.
9. A mean of component centroids and a centroid of the corresponding pixel
   union are different estimators and require different IDs.
10. Position-provider selection is explicit and lineage-bearing. A consumer
   must not silently fall back from masks to keypoints to detections.
11. Existing runs are migrated through immutable successors or explicit legacy
    adapters, never by relabeling their position or heading semantics.
12. Position products share one discovery family but use distinct row-axis
    namespaces:
    `analysis/subject_position_runs/observation/<run>` for exact
    `instance_key` rows and
    `analysis/subject_position_runs/track_sample/<run>` for exact
    `track_sample_key` rows. A run belongs to exactly one row axis, selectors
    are row-axis-specific, and cross-axis projection publishes a new derived
    product with its exact mapping and source digests.
13. Version 1 point expressions support keypoint, component-centroid,
    bounding-box-centroid, midpoint, and equal-per-point mean operations.
    Arbitrary numeric weights and caller-defined formulas are not accepted.
    A scientifically justified weighted center requires a new named,
    versioned estimator profile.
14. Version 1 anatomical estimators require every declared anchor. Missing,
    invalid, non-finite, empty, or authority-rejected anchors invalidate the
    row; they are not dropped, renormalized, interpolated, or replaced from a
    different modality. Structural source incompatibility blocks publication,
    while an ordinary row measurement failure is retained as an all-NaN
    position with explicit validity and reason evidence.
15. Observation-position runs are unsmoothed geometric measurements. Temporal
    smoothing requires ordered track identity and may publish a separate,
    explicitly identified `track_sample` position derivative. Smoothing a
    position and smoothing a derived speed are different operations and remain
    separately named and lineage-bound.
16. There is no global implicit position default or fallback order. Automated
    analyses use a named, versioned policy that explicitly selects an
    estimator and acceptable source authority. Existing detection-centroid
    behavior remains available through an explicit compatibility policy. A
    new organism/protocol/analysis-scoped default requires reviewed canary
    evidence and a timestamped promotion decision.
17. Materialized position storage follows
    `subject_position_storage_contract_v1`: evaluator arithmetic uses float64,
    authoritative continuous source-camera positions are published as exact
    float32, validity is explicit bool, failure state uses compact uint16
    controlled codes, and invalid positions use canonical paired float32 NaNs.
18. Version 1 motion keeps its declared nominal-FPS arithmetic. Scientific
    readiness late-binds that arithmetic to the exact recording, canonical
    `source_video_metadata.v2`, and selected acquisition frame-clock digest in
    the same archive. It does not copy camera timestamps or publish a second
    timestamp-delta array. A future variable-delta computation requires a new
    versioned motion policy and evidence that nominal timing is inadequate.

## Relationship to Existing Heading and Body-Frame Contracts

The active keypoint heading contract stores a declarative recipe in:

```text
pose_schema.metadata.heading_computation
```

For the current zebrafish skeleton, the forward direction is:

```text
swim_bladder -> midpoint(eye_left, eye_right)
```

The richer body-frame contract already separates semantic anchors, estimator
modality and formula, materialized geometry, validity, and exact provenance.
That separation is the model for generalized position.

Position differs from heading in one important way. An anatomy profile can
define one canonical anterior construction while supporting several
scientifically useful subject positions. A source schema proves that it can
supply the required roles, but an analysis must still bind one exact
position-estimator profile.

Existing `pose_schema.metadata.heading_computation` declarations remain
authoritative for the schemas and immutable runs that already use them. They
are exposed through an explicit source-local compatibility adapter rather than
rewritten. New schemas may reference a shared anatomy profile. If an inline
source declaration and a shared-profile reference are both present, their
canonical normalized semantics must agree or validation fails closed.

## Anatomy Profiles and Source Bindings

A shared profile such as `zebrafish_larva_anatomy.v1` owns modality-neutral
roles such as `swim_bladder`, `eye_left`, `eye_right`, and `subject_body`, plus
named constructions such as `head_triad_equal_mean`, `eye_pair_midpoint`, and
`anterior_axis`.

A pose schema binds those roles to exact keypoint labels. A subject-mask schema
binds them to exact component labels and availability semantics. A source with
only `eyes_union` cannot advertise support for a recipe requiring independently
labeled left and right eyes. String equality between labels in two schemas is
not a semantic join; consumers validate the profile identity, source-schema
identity, explicit role bindings, and their digests.

The estimator profile selects the shared recipe plus an exact source modality
and validity policy. The materialized run binds the anatomy profile, source
schema, role bindings, estimator, and source arrays. This lets keypoint and mask
estimators share biological meaning without claiming that their observations
or validity contracts are identical.

## Conceptual Layers

```text
shared anatomy profile
├── stable biological roles
└── reusable named point/axis recipes
        ↓
semantic source schema and explicit role binding
├── keypoint skeleton with bound landmark roles
├── subject-mask set with bound component roles
└── detection geometry without anatomical roles
        ↓
position or body-frame estimator profile
├── required semantic roles
├── point/axis expression and weighting
├── source modality
├── confidence and missing-source policy
└── coordinate and row/time compatibility requirements
        ↓
materialized derived surface
├── exact values
├── validity and reason codes
├── source identities and content digests
└── estimator/profile identity and digest
        ↓
motion derivation
├── speed/path/acceleration from position
└── heading change/angular velocity from body frame
```

## Initial Position Estimator Profiles

### `detection_bbox_centroid.v1`

```text
position = midpoint(bbox_img_xyxy.min, bbox_img_xyxy.max)
```

This is the current detection/crop position behavior. It is not anatomical,
but remains useful when keypoints or masks are unavailable. It binds the exact
box array, instance identity, acquisition frames, coordinate descriptor, and
selected detection/refinement run.

### `keypoint_anatomical_triad_mean.v1`

```text
position = mean(
    keypoint(swim_bladder),
    keypoint(eye_left),
    keypoint(eye_right),
)
```

Each required landmark contributes equal weight. Labels resolve through the
exact pose schema rather than numeric channel indices.

### `mask_component_anatomical_triad_mean.v1`

```text
position = mean(
    component_centroid(swim_bladder),
    component_centroid(eye_left),
    component_centroid(eye_right),
)
```

Each anatomical component contributes equal weight after its own centroid is
computed. This is intentionally not the centroid of the union of the three
masks, which weights components according to pixel area.

For modern subject-mask runs, the mask publication owns that computation and
its ROI-to-source-camera projection. The position evaluator consumes the
exact canonical source-camera `centroid_xy` and `centroid_valid` surfaces; it
does not reopen a dense ROI mask, guess an ROI origin, or threshold mask
probabilities a second time. A future source without an authoritative centroid
surface needs an explicit coordinate-bound adapter before it can advertise
this estimator.

### `subject_body_mask_centroid.v1`

```text
position = component_centroid(subject_body)
```

This estimator describes the projected whole-body area center. It can differ
systematically from both head-triad and detection-box centers and remains a
separate method rather than a fallback alias.

## Controlled Point Expressions

The first expression vocabulary should remain narrow:

- `keypoint(label)`;
- `component_centroid(component_id)`;
- `bbox_centroid(array_ref)`;
- `midpoint(point_a, point_b)`; and
- `mean_points(points, weighting="equal_per_point")`.

The evaluator canonicalizes and digests the complete expression. Unknown
operations, roles, weighting policies, or coordinate types fail closed.
`midpoint` is the canonical two-point equal mean. Point order is semantically
irrelevant for a mean and canonicalizes consistently. Version 1 does not accept
caller-supplied numeric weights or custom formulas. Pixel-union centroid is a
distinct operation and is not represented as `mean_points`; adding it requires
a separately controlled operation and estimator profile.

The same semantic recipe may be supported by more than one modality, but the
materialized estimator remains modality-specific because keypoints and masks
have different source and validity contracts.

## Validity and Confidence Policy

Structural incompatibility fails the run before publication. This includes a
required anatomy role absent from the source schema, a required run-level mask
channel marked unavailable, missing/duplicated/reordered source identities,
incompatible coordinates, stale source or schema digests, and a source that
requires but does not identify an authoritative validity policy.

An ordinary per-row measurement failure preserves row coverage but writes
`position_xy = [NaN, NaN]`, `valid = false`, and a controlled failure reason.
Support evidence retains per-anchor validity so multiple failures remain
inspectable even when the row has one deterministic primary reason code.

Version 1 estimator validity is:

- detection-box centroid: the bound source observation is accepted and its box
  is finite and non-degenerate;
- keypoint anatomical-triad mean: swim bladder, left eye, and right eye are all
  source-authority-valid and finite;
- mask-component anatomical-triad mean: all three bound components are
  available and each row has a valid, nonempty mask with a finite centroid; and
- whole-body mask centroid: the bound `subject_body` component is available
  and the row has a valid, nonempty mask with a finite centroid.

Position evaluation consumes each source's exact authoritative validity and
binds its policy identity and digest. Canonical detection rows bind the
canonical-detection schema-v1 invariant that every published row is a finite,
positive-area, in-extent observation. Keypoints and masks bind their persisted
validity surfaces. The position layer does not apply another guessed numeric
confidence threshold. A future raw source with a threshold decision must
reference an explicit versioned policy; there is no implicit default. No
estimator computes a partial mean, renormalizes remaining anchors, falls back
across modalities, or interpolates at the observation-position stage.

## Proposed Materialized Position Surface

The accepted namespace separates measurement rows from identity-resolved
trajectory rows without splitting provider discovery across unrelated
families:

```text
analysis/subject_position_runs/observation/<run>/
  position_xy                     float32 [N, 2]
  valid                           bool [N]
  failure_reason_codes            uint16 [N]
  instance_key                    uint64 [N]
  source_acquisition_frame_index  int64 [N]
  source_row_index                int64 [N]
  support/
    source_points_xy              optional [N, P, 2]
    source_points_valid           optional [N, P]
```

An observation run uses row axis `observation_instance` and binds the exact
ordered `instance_key` source rowset. Phase 1 implements this form first.

A future independently useful track-resolved provider uses:

```text
analysis/subject_position_runs/track_sample/<run>/
  position_xy                     float32 [N, 2]
  valid                           bool [N]
  failure_reason_codes            uint16 [N]
  track_sample_key                int64 [N, 2]
  source_acquisition_frame_index  int64 [N]
```

A track-sample run uses row axis `track_sample` and binds the exact ordered
`track_sample_key = (track_id, acquisition_frame_index)` rowset. It is a new
derived publication, not an observation run reinterpreted after tracking.
Cross-axis conversion requires a sealed, digest-bound mapping; equal row counts
are never evidence of compatibility. Selectors are scoped by row axis, such as
`latest_observation` and `latest_track_sample`; there is no ambiguous universal
`latest` selector.

Initially, track kinematics consumes an observation-position run, joins it
through exact `instance_key` lineage, and stores its resolved track-sample
positions in the track-kinematics output. The `track_sample` position namespace
is reserved for independently reusable products such as an explicitly
interpolated, smoothed, or externally supplied trajectory position series.

Required metadata includes:

- position schema and estimator IDs, versions, canonical payload, and digest;
- semantic skeleton or mask-set identity and digest when applicable;
- exact source run, arrays, row identity, and source-content digests;
- coordinate descriptor and applied transform lineage;
- source acquisition-frame authority;
- missing/invalid/confidence policy;
- selection and publication state; and
- immutable manifest and completion evidence.

Support arrays are diagnostic evidence. Consumers use `position_xy` and
`valid` and must not reconstruct a different position from support arrays
while claiming the same estimator.

The exact logical dtypes, float64-to-float32 computation boundary, canonical
NaN invariants, coordinate descriptor, reason-code registry, physical-plan
requirements, and reader checks are defined in
`subject_position_storage_contract_v1.md`.

## Body-Frame and Heading Selection

Reusable orientation continues to use the body-frame contract:

```text
analysis/body_frame_runs/<run>/
  origin_xy
  forward_axis_xy
  left_axis_xy
  axis_valid
```

Compatible estimator families include `keypoint_head_axis`,
`mask_component_axis`, and `body_spline_with_anchor_polarity`. Scalar heading
derives from the selected forward axis under the fixed coordinate convention,
or from a separately row-bound heading surface when its contract permits it.
Position and body-frame selection remain independent.

## Motion Derivation

A future track-motion input authority binds independent handles:

```yaml
position_source:
  run: analysis/subject_position_runs/observation/<run>
  row_axis: observation_instance
  estimator_id: keypoint_anatomical_triad_mean.v1
  manifest_sha256: ...

body_frame_source:
  run: analysis/body_frame_runs/<run>
  estimator_id: keypoint_head_axis.v1
  manifest_sha256: ...
```

The resolver verifies exact row identity or a sealed join, compatible
acquisition-frame authority, compatible coordinates and transforms, completed
eligible sources, and current source digests.

Linear outputs bind position plus temporal/scale authorities: displacement,
path distance, speed, acceleration, occupancy, and spatial trajectory metrics.
Angular outputs bind body frame or heading plus temporal authority: heading
change, angular velocity, angular speed, and body-relative bearing.

### Temporal position transforms

Observation-position providers never smooth or interpolate. A temporal
position transform first resolves exact track identity and order, then may
publish a new immutable
`analysis/subject_position_runs/track_sample/<run>` derivative. That run binds
the source observation-position run, sealed observation-to-track mapping, time
authority, smoothing/interpolation algorithm, requested and effective window,
alignment, edge handling, and gap policy.

Raw and transformed track-sample positions remain independently selectable.
A motion algorithm must declare which exact position it differentiates. An
algorithm that smooths an already derived speed publishes a filtered-speed
surface, not a smoothed-position provider. Occupancy and trajectory recipes
also bind the exact raw or transformed provider they consume.

Phase 1 implements unsmoothed observation positions and their exact projection
into track kinematics. A generic smoothed-position publisher is deferred, but
the separate track-sample contract is reserved now. Existing track-kinematics
smoothing remains readable under its original explicit compatibility contract
and is not relabeled as a new provider.

## Composable Analytics Integration

Generic offers require semantic capabilities instead of one hardcoded path:

| Metric | Required provider capability |
| --- | --- |
| occupancy | valid position series |
| path/trajectory | ordered valid position series |
| speed | position plus temporal authority |
| acceleration | speed-compatible position and time |
| heading | valid body-frame or heading series |
| angular speed | body frame or heading plus time |
| body-relative stimulus bearing | position, body frame, and stimulus geometry |

Multiple compatible providers produce multiple explicit offers. A saved
analysis request binds the selected provider ID and run digest. Cross-provider
comparisons are separate products; they do not overwrite or average sources.

## Provider Selection and Promotion

An estimator defines mathematical position semantics. A position-selection
policy chooses an estimator, acceptable source authority, row axis, and
validity policy. A broader analysis policy may then combine that position
policy with exact tracking, temporal transforms, motion, and plotting choices.

For example, existing behavior is preserved explicitly rather than treated as
an unnamed default:

```yaml
policy_id: detection_centroid_compatibility.v1
position_estimator: detection_bbox_centroid.v1
source_requirement: selected_refined_detection
row_axis: observation_instance
validity_policy: upstream_authority_required.v1
fallback: none
```

A direct request names a provider or a named policy that names one. `latest`
cannot choose a scientific method. If the selected provider is unavailable,
the request stops with a structured reason instead of falling back across
detections, keypoints, or masks. Multiple providers may remain visible as
separate offers. Any recommended default is scoped to a defined organism,
protocol, and analysis context rather than declared globally.

Before a provider such as `keypoint_anatomical_triad_mean.v1` may become a
GoodBatBadBat default, a reviewed canary must establish:

- coordinate and row-lineage correctness;
- valid coverage and failure patterns across representative cameras and
  recordings;
- visual agreement with reviewed anatomy;
- stability across pre-, chaser-, and post-periods;
- measured differences among detection, keypoint, and mask alternatives;
- consequences for occupancy heatmaps, speed, acceleration, bouts, and
  pre/post contrasts;
- absence of systematic behavioral-state-dependent or camera-dependent bias;
- reproducibility from the same immutable inputs; and
- a timestamped promotion decision binding the estimator, evidence set, and
  policy version.

Before promotion, Palette may materialize and compare selector-ineligible
providers but must not silently install a new default.

## Current Implementation Assessment

- Keypoint heading semantics are declarative through
  `pose_schema.metadata.heading_computation`.
- The heading evaluator supports `keypoint` and two-label `midpoint`; a general
  multi-point mean operation is not implemented.
- The body-frame contract already separates anchors, estimators, and outputs
  across keypoint, mask-component, and spline families.
- Current offline track kinematics binds detection/crop source-camera centers
  for position and a keypoint-run heading array for orientation.
- Current online track kinematics uses its selected position rowset and an
  optional row-sibling visual angle.
- Track kinematics therefore demonstrates mixed position/heading modalities,
  but its accepted source families are tightly coded.
- Existing composable stimulus analytics assume a trajectory but do not
  identify which position estimator produced it.

## Implementation Checklist

### Phase 0: decisions and compatibility evidence

- [ ] Freeze current detection/crop-centroid plus keypoint-heading track
      fixtures.
- [ ] Freeze numeric fixtures distinguishing eye midpoint, triad mean,
      equal-component mask mean, pixel-union centroid, and body-mask centroid.
- [ ] Inventory position, heading, body-frame, speed, occupancy, and stimulus
      consumers and classify their implicit source assumptions.
- [x] Select one position-run family with distinct `observation` and
      `track_sample` row-axis namespaces; implement observation rows first.
- [x] Place reusable named point recipes in a shared anatomy profile; require
      pose and mask schemas to bind their native labels to profile roles.
- [x] Freeze the v1 expression vocabulary and equal-per-point weighting; do
      not permit arbitrary caller-supplied weights or formulas.
- [x] Require every estimator anchor; distinguish run-fatal structural source
      incompatibility from retained row-level invalid measurements, and reuse
      exact upstream validity policies without guessed thresholds.
- [x] Keep observation positions unsmoothed; reserve temporal smoothing for a
      separately identified track-sample position derivative and distinguish
      it from filtered speed.
- [x] Require explicit named provider-selection policies with no fallback;
      preserve detection centroid through a compatibility policy and require
      reviewed canary evidence before any scoped default promotion.
- [x] Freeze the v1 logical storage contract: exact float32 position arrays,
      explicit bool validity, uint16 reason codes, canonical invalid NaNs,
      exact identity/frame/source-row types, and coordinate authority.

### Phase 1: pure contracts and evaluators

- [x] Define versioned position-estimator and materialized-position schemas.
- [x] Implement canonical point-expression models and digest helpers.
- [x] Implement pure evaluators for keypoint, component-centroid, box-centroid,
      midpoint, and equal-per-point means.
- [x] Validate roles through exact skeleton or mask-set identities.
- [x] Define stable validity and failure-reason vocabulary.
- [x] Test reordered labels/components, missing roles, invalid points, empty
      masks, unknown operations, unsupported weighting policies, and stale
      schema digests.
- [x] Prove equal-component mean differs from pixel-union centroid when
      component areas differ.

Phase 1 is implemented by `fisheye.shared.anatomy_profile`,
`subject_position_types`, `subject_position_expression`,
`subject_position_contract`, and `subject_position_storage`, with the initial
zebrafish profile at
`configs/fisheye/anatomy_profiles/zebrafish_larva_v1.json`. These are pure
contracts, evaluators, and logical validators; they do not publish Zarr runs or
install a provider-selection default.

A Phase 1 `PositionEvaluationResult` is deliberately unbound numeric output.
It is not source-camera authority and is never selector-eligible by itself.
Before publication, Phase 2 must prove that every input row has the exact
source identity, canonical coordinate descriptor, and required directed
transform. Equal row counts, matching label strings, and caller assertions are
not sufficient.

The initial Phase 2 source-currentness policy is deliberately narrow. A
position adapter consumes one explicitly named run only when that run is the
current, complete, selector-eligible canonical coordinate publication for its
source family. Keypoint production-bundle members that remain
selector-ineligible are a different authority and are not an implicit fallback.
Supporting those members later requires a separately named adapter and policy.

The four initial estimator profiles are detection bounding-box centroid,
keypoint anatomical-triad equal mean, subject-mask anatomical-component-triad
equal mean, and subject-body-mask centroid. A pixelwise mask-union centroid is
not one of these four. Its operands, overlap semantics, threshold authority,
and validity policy must be defined in a new versioned estimator before it can
be materialized.

Subject-mask adapters consume the producer's validated `centroid_xy` and
`centroid_valid` surfaces and apply their exact bound ROI-to-source-camera
transform. They do not reopen masks and independently recompute centroids in
the position layer.

All Phase 2 publications bind
`subject_position_canary_no_default.v1`. This policy permits only an explicitly
named provider, has no default or fallback, requires later promotion evidence,
and requires the materialized run to remain selector-ineligible. It is
independent of the estimator validity policy that determines row-level
success and failure reasons.

### Phase 2: immutable position publication

- [x] Implement selector-ineligible position-run planning and publication.
- [x] Add the four initial estimator profiles.
- [x] Bind exact row identity, acquisition frames, coordinate descriptors,
      source arrays, semantic schemas, and manifests.
- [x] Add a keypoint source adapter that validates the persisted canonical
      skeleton identity and projects ROI keypoints through the exact row-wise
      crop transform into source-camera coordinates.
- [x] Add a subject-mask source adapter that validates component-label and
      availability authority, exact `centroid_xy`/`centroid_valid` surfaces,
      row identity, derivation records, and source-camera coordinate binding.
- [x] Validate direct and consolidated metadata before completion.
- [x] Preserve sources and publish retries as new immutable attempts.

Phase 2 is implemented by the strict detection, keypoint, and subject-mask
source adapters in `fisheye.shared.subject_position_*_source`, the sealed
source-to-evaluator bridge in `subject_position_preparation`, and the generic
publisher in
`fisheye.analysis_workflows.materializers.subject_position`. The publisher
writes the canonical descriptor on `position_xy`, binds every source and
policy record in the immutable manifest, publishes through a hidden atomic
attempt, refreshes consolidated metadata only after final validation, and
leaves all selectors unchanged. The subject-mask adapter has estimator-specific
entry points so the anatomical-triad provider does not require an unrelated
whole-body channel and the whole-body provider does not require eye or swim
bladder channels.

`subject_body_centroid` is now an explicit anatomy recipe advertised only by
the subject-mask binding. No pixel-union operation or estimator was added.

### Phase 3: body-frame and motion integration

- [x] Define typed position-source and keypoint-body-frame-source handles.
- [x] Add an exact content manifest to newly written tracking runs and make
      keyed tracking identity explicit.
- [x] Define a typed tracking-run handle that rejects selector lookup,
      keyless legacy rows, stale manifests, post-seal mutation, stale
      consolidation, and cross-archive composition.
- [x] Check row identity, time, coordinates, transforms, completion, and
      staleness before composition.
- [x] Generalize track-motion authority to consume one explicit position and
      one explicit body-frame/heading source.
- [x] Validate the traditional-v3 inline heading declaration against the
      shared `anterior_axis` semantics before exposing a compatibility adapter.
- [x] Keep linear and angular motion lineage independent.
- [x] Preserve detection-centroid/keypoint-heading behavior as an explicit
      compatibility profile.
- [x] Refuse implicit fallback and same-length-only joins.
- [x] Publish selector-ineligible successors instead of mutating existing track
      runs.

Phase 3 deliberately does not infer heading from the outline of a full-body
mask. PCA, ellipse, centerline, spline, or another subject-body shape
orientation estimator remains deferred and requires a separate controlled
recipe, polarity policy, validity contract, and canary evidence.

The narrower `mask_component_axis` calculation is distinct from full-body
shape orientation: it uses only the explicitly labeled `eye_left`,
`eye_right`, and `swim_bladder` component centroids. Its array-level producer
and source validation are implemented, but the current immutable
`body_frame_runs` manifest and reader are keypoint-source-specific. Therefore
the component-mask adapter remains non-publishing in this phase; it must gain
a genuinely mask-aware manifest and typed reader before it can feed motion
publication. It must not be published by relabeling mask rows as keypoint
rows.

The new provider-motion publication is a selector-ineligible canary surface.
It now accepts only a loader-minted tracking-run handle; callers can no longer
supply a path, digest, key array, or track-ID array independently. Newly
written tracking runs carry an exact decoded-content manifest and explicit
selector eligibility. The handle reopens and re-hashes the named run during
composition, motion preparation, run planning, and immediately before atomic
publication. It also proves that tracking and the position/body-frame
authority belong to the same archive. Keyless or manifestless historical
tracking remains available through the legacy compatibility reader but cannot
become modern provider-motion authority without an immutable keyed successor.

This closes the Phase 3 tracking-authority implementation blocker. Production
activation remains blocked by required CI and Phase 5 canary/promotion
evidence, not by another implicit tracking input surface.

Provider-motion successors require the archive's typed source-camera physical
authority by default. The authority's source-camera frame digest must equal the
position provider's frame digest. A calibrated run publishes exact paired
`px`/`mm`, `px/s`/`mm/s`, and `px/s^2`/`mm/s^2` arrays and validates each
physical array as its float32 pixel peer multiplied by the bound
`mm_per_pixel`. Pixel-only publication is permitted only through an explicit
selector-ineligible canary exception and records that omission in the immutable
computation manifest.

Existing provider-motion runs preserve exact acquisition-frame indices and
compute `time_seconds` from a caller-supplied FPS. The Phase 4 binding layer now
late-binds an existing immutable run to the recording authority without
rewriting it. The strict loader requires canonical `source_video_metadata.v2`,
the exact selected acquisition frame-clock record and array digests, matching
recording/camera/frame-count/FPS metadata, the complete zero-based acquisition
frame domain, source indices within that domain, and direct/consolidated
metadata equality. A provider-motion run additionally has to declare the same
FPS as the recording authority. The resulting authority digest is shared by
position, body-frame, motion, and temporal-selection identities; an offer is
not `ready` when any required identity is missing it or the digests disagree.

This is intentionally a read-time, no-write bridge. It validates the existing
clock publication and array digests, but neither copies camera or system
timestamp values into provider runs nor changes the numerical values of
existing motion products. A legacy archive without an acquisition frame clock
remains explicit `legacy_missing`/`blocked_temporal_authority`; no nominal
clock is guessed.

#### Nominal-timebase evidence decision (2026-08-17)

A read-only audit covered all 84 canonical GoodBatBadBat analysis Zarrs and
14,202,392 acquisition frames. Camera timestamps were present and valid for
100% of frames; every adjacent delta was strictly positive, with no duplicate
or decreasing timestamps. At the declared 100 FPS, the nominal interval is
10,000,000 ns. The median recording p99 absolute deviation from nominal was
25 ns, the maximum recording p99 was 25 ns, and full-recording span drift was
between -26 ns and +26 ns.

After implementation, the strict no-write authority loader also bound all 84
canonical archives successfully (`84 discovered`, `84 bound`, `0 failed`),
including direct/consolidated equality and existing clock payload-digest
validation.

That evidence does not justify a second timestamp-derived motion product or
additional copied delta arrays. Version 1 therefore retains
`frame_index_difference / nominal_fps`, while exact acquisition-clock lineage
is required for scientific readiness. Revisit variable-delta motion only if a
future bounded audit finds meaningful non-monotonicity, missing coverage,
jitter, or accumulated drift, and introduce it as a separately identified
policy rather than changing version 1 in place.

### Phase 4: composable stimulus analytics

Phase 4 may now build on the explicit provider-motion authority. Its first
implementations must remain selector-ineligible offers and must not imply that
one position or body-frame provider has been promoted as the scientific
default.

The shared Phase 4A foundation is implemented by
`provider_analysis_offers`, `provider_analysis_bindings`,
`provider_recording_timing_authority`,
`provider_track_motion_source_handle`, and `resolved_epoch_selection`.
Metric-specific occupancy, motion, and heading offers remain pending.

- [x] Late-bind provider and temporal identities to one exact recording,
      source-video metadata record, and acquisition frame-clock authority.
- [ ] Add provider requirements to occupancy, trajectory, speed,
      acceleration, heading, and angular-speed offers.
- [ ] Include provider IDs and digests in metric, contrast, and plot recipes.
- [ ] Permit multiple explicit offers when several providers are available.
- [ ] Add strict cross-provider comparison products.
- [ ] Reject incompatible providers in ordinary scientific contrasts.
- [ ] Expose provider identity and readiness in reporting and review UIs.

### Phase 5: canaries, migration, and activation

- [ ] Materialize all initial providers for one reviewed zebrafish recording.
- [ ] Compare offsets, valid coverage, speed, occupancy, and bout sensitivity
      without selecting a production default.
- [ ] Review representative frames where providers disagree.
- [ ] Publish selector-ineligible track successors for at least two position
      methods using the same body-frame source.
- [ ] Validate source preservation, exact lineage, and consolidated visibility.
- [ ] Define promotion policy only after canary evidence exists.
- [ ] Add legacy adapters without relabeling historical estimator semantics.

## Acceptance Criteria

- Every metric or motion run identifies its exact position and body-frame
  estimators.
- Detection, keypoint, and mask methods remain separate immutable products.
- Skeleton and mask roles resolve by controlled IDs, not array indices.
- Equal-component triad mean and pixel-union centroid remain distinct.
- Invalid anchors produce explicit invalid rows and reason codes.
- Speed can be recomputed from different positions without changing its
  algorithm or stimulus-selection contract.
- Angular outputs can independently select a compatible body frame.
- A scientifically ready offer binds every provider and its temporal selection
  to the same exact recording/timebase authority digest.
- Existing detection-centroid tracks remain unchanged and readable.
- No provider is selected through an undocumented fallback order.
- Composable offers expose provider identity and readiness.

## Resolved Decisions Before Implementation

The initial namespace, row-axis, anatomy-profile ownership, expression,
weighting, validity, smoothing, source-selection, and promotion-policy
decisions are resolved above. Implementation may proceed only through the
phased checklist and remains selector-ineligible until its required canary and
promotion evidence are complete.

## Related Documents

- [Composable Stimulus Analysis and Plot Recipe Design](composable_stimulus_analysis_and_plot_recipes_design.md)
- [Subject Position Storage Contract v1](subject_position_storage_contract_v1.md)
- [Keypoint Heading Computation Contract](keypoint_heading_computation_contract.md)
- [Body Frame Contract](body_frame_contract.md)
- [Derived Analysis Run Contract](derived_analysis_run_contract.md)
- [Future Track Motion Storage Layout](future_track_motion_storage_layout.md)
