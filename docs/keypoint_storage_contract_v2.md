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

### Pose model-input contract v1

The model graph does not own source crop geometry. Strict whole-recording
inference therefore requires a separate
`palette.pose_model_input_contract` document bound to one exact model set,
run, weights digest, training-manifest digest, training-report digest, and
training-args digest. The contract declares the training source ROI shape,
network `imgsz`, pixel contract, Ultralytics version, maximum model stride,
training-time rectangular/multi-scale settings, runtime adapter,
channel/normalization behavior, and result-coordinate space.
Planner and worker both validate it; the worker also verifies the actual model
selected through the registry. Runtime versions are separately allowlisted and
must reproduce a digest-bound deterministic preprocessing tensor before pixels
are staged; matching a version string alone is not sufficient.

Historical model backfill uses
`fisheye.utils.build_pose_model_input_contract`. It reconstructs only claims
supported by immutable package evidence and refuses disagreement between the
training manifest, report, and arguments. Future training workflows should
emit the same document directly rather than require reconstruction.

For the first Batman diagnostic, the contract derives two distinct stages:

1. center-pad the native 348x348 cache row with constant zero to the 512x512
   training source extent;
2. submit that 512x512 luma-repeated image through the reviewed Ultralytics
   runtime adapter at
   `imgsz=256`, `rect=false`, OpenCV linear letterbox preprocessing, and
   uint8-to-float `/255` normalization.

Ultralytics reports detections in the original submitted 512-pixel extent, so
the canonical coordinate transform remains the existing exact 512-to-348
padding inverse. The internal 512-to-256 resize is library preprocessing, not
a new public coordinate space. A smaller native crop padded to 512 is labeled
`scale_matched_diagnostic_not_training_context`: it matches the trained scale
but does not pretend that synthetic padding contains the real surrounding
camera pixels.

The raw-keypoint-v2 array schema is unchanged. The terminal receipt binds the
input-contract file digest, payload digest, derived runtime plan, and observed
runtime attributes and probe result before strict v2 finalization. The current
historical model was trained with Ultralytics `8.3.214`; runtime versions
`8.3.169` and `8.3.214` produce the exact reviewed probe SHA-256
`d141f8e12a791d6b4b0c99ae3dfc24c6d6c11b63f9739df755d1d7bbe4b1d35a`.

### Terminal inference failure evidence

Production-shaped YOLO inference persists a terminal-only
`pose_failure_codes: uint8[N]` array before strict raw-v2 preparation. Code zero
must coincide exactly with `detection_success == true`; every failed row has
one declared nonzero terminal outcome:

| Code | Label | Meaning |
| ---: | --- | --- |
| 0 | `none` | A pose row was resolved |
| 1 | `no_pose_detection_above_threshold` | Postprocessing returned no pose detection at the recorded confidence threshold |
| 2 | `keypoint_payload_missing` | A selected pose detection lacked its keypoint payload |
| 3 | `keypoint_payload_empty` | The keypoint payload was empty or had no detection axis |
| 4 | `insufficient_keypoint_count` | A legacy terminal result contained fewer landmarks than the bound skeleton |

Unknown codes, non-`uint8` arrays, or disagreement with the success mask fail
terminal validation. Structural incompatibilities in canonical inference—such
as a model/skeleton cardinality mismatch—remain job-level failures rather than
being downgraded to a row-level code.

The immutable terminal receipt binds the code map, complete histogram, and
array digest. Strict raw-v2 preparation validates this evidence but does not
copy `pose_failure_codes` into `keypoints_runs`: v2 is an already frozen public
schema. Promoting these diagnostics to the public archive requires an explicit
raw-keypoint schema revision rather than an optional array or an in-place v2
change.

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

Refined source-bindings v2 persist the complete canonical skeleton-semantics
document, not merely its ID and digest. The document contains the ordered
labels, nodes, edges, `[K,2]` shape, and heading recipe. Its canonical JSON
SHA-256 must equal the bound skeleton digest, its nodes must reproduce the
ordered labels exactly, and its cardinality must equal the keypoint array's
second axis. Consumers resolve anatomical roles such as `eye_left` and
`eye_right` from these labels at runtime; positional or model-specific index
fallbacks are forbidden. Historical refined source-bindings v1, which contain
only `skeleton_id` and `skeleton_digest`, are insufficient for an anatomical
consumer unless a separate explicitly bound semantics document is available.

Edits are keyed deltas. Accepted deltas compact into a new immutable sharded
snapshot. Training exports cite that compact snapshot and materialize their own
dense image/label representation; they do not treat an edit delta as training
authority.

The first-snapshot refined-v2 producer and selector-ineligible publisher are
implemented. The producer consumes one completely validated raw-keypoint-v2
snapshot, its exact quality-v1 snapshot, the bound crop geometry, and explicit
decisions keyed by `instance_key`. It preserves row order and source facts,
recomputes image projections and row signatures after accepted landmark edits,
and supports rejection, recovery of source failures, validity edits, and exact
skeleton permutations. Every destination buffer is independent of its inputs.

The persisted manifest binds the complete raw, quality, and crop manifests;
recording, skeleton, coordinate-catalog, and row-signature identities; exact
review/reason registries; immutable snapshot identity; retired-key evidence;
the reconstructed byte-derived storage plan; all decoded array hashes; and
direct/consolidated declarations. The publication gate requires refined keys
to preserve the raw key order and forbids live/retired overlap. Parent/successor
validation remains deliberately deferred to the delta compactor rather than
being approximated by this initial-snapshot writer.

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

Names such as `D2`, `C2`, `Kraw2`, and `Kref2` are explanatory generation
labels only. They are not persisted array or group names. Concrete artifacts
remain versioned runs under `refined_detect_runs/<run>`, `crop_runs/<run>`,
`keypoints_runs/<run>`, and `refined_keypoints_runs/<run>`, with exact parent
run and manifest-digest bindings.

When a detection successor changes the observation row set, the first
implemented successor chain is:

```text
immutable refined-detection successor
  -> complete geometry-only crop successor
  -> complete raw-keypoint successor
```

Crop reconciliation is keyed by `instance_key`. An unchanged observation and
unchanged crop geometry may reuse its parent pose payload. A new observation or
a surviving observation whose crop geometry changed must have one terminal
inference result. A successful attempt persists finite pose payload; a failed
attempt persists a real row with `pose_success=false` and exact NaN pose
payloads. An absent result means pending work and blocks successor publication.
A retired detection and crop row is omitted from the new keypoint snapshot.

This rule keeps the raw snapshot total over the target crop row set: one raw
keypoint row exists for every crop observation, including attempted failures.
Later manual keypoint recovery edits that failed row through the refined
keypoint lifecycle; they do not synthesize a new detection or crop row.

The current successor publisher is deliberately selector-ineligible and
unregistered. It writes a fresh immutable Zarr and a digest-bound receipt that
records reused, inference-success, inference-failure, and retired key sets.
Atomic archive import, production DAG registration, and selector activation
remain separate gates. The cross-application ownership, pluggable inference,
commit, compaction, and promotion decisions are frozen in the
[Crimson inference, commit, and Palette compaction decision](crimson_inference_commit_compaction_decision_2026-07-29.md).

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

## Clipped Compute and Recording-Level Finalization (2026-07-29)

Clip boundaries are bounded compute partitions, not public storage
partitions. The maintained execution shape remains:

1. materialize one immutable pixel package per clip;
2. run clip-local keypoint inference in an LSF array;
3. write one terminal sidecar per clip; and
4. rematerialize complete recording-level snapshots with the shared planners.

`ClipTerminalKeypointResult` now makes the clip boundary exact. The existing
clip shard deliberately remains `coordinate_contract_mode=legacy_noncanonical`;
it cannot self-certify keypoint-v2. Every sidecar binds the clip index, complete
`instance_key` set, source crop-row signatures, crop manifest and coordinate-
catalog digests, keypoint coordinate-catalog digest, pose-model binding,
preprocessing contract, input-package manifest, exact hashes of the eight
legacy YOLO result arrays, four proxy-crop lineage/geometry arrays, and the
fixed proxy ROI shape. The package path must also equal the package recorded by
the clip shard, and finalization rehashes that file. The adapter proves proxy
instance keys, frame identities,
crop-row mapping, integer origins, and sizes equal the strict crop-v2 rows before
narrowing legacy float64 pose values to canonical float32. Acquisition-frame
identity is taken from crop-v2 rather than invented by the legacy shard.

The preprocessing reference must include an exact `clip_source_contract`
object containing the legacy coordinate mode, `input_mode_effective`, and
`model_input_transform` copied from the completed clip shard. A completed clip
artifact contains one result for every expected row. Model failure is a real
terminal row with `pose_success=false` and NaN pose payload; a missing row is
pending/incomplete and blocks finalization.

The recording finalizer joins by `instance_key`, verifies the clip keysets
exactly partition the crop-v2 rowset, restores crop row order, and recomputes
source-camera projections and row signatures. It never copies chunk or shard
metadata from clip outputs. Raw keypoints, keypoint quality, refined
keypoints, and body frame are each written through their byte-based
`published_http_v1` planner. One canonical JSON receipt binds every clip
sidecar, the crop-v2 manifest, and all four finalized run manifests.

The first integration surface is deliberately selector-ineligible and
unregistered. It writes a direct-path bundle rather than importing the four
keypoint families into a recording archive. This is sufficient to test exact
contracts and physical rematerialization without changing production. The
later archive-import transaction must reuse these same builders and receipts.

Detections, refined detections, and crops follow the same ownership rule:
per-clip detect/refine results are compute evidence; canonical/refined
detection snapshots and crop-v2 are complete recording-level publications.
Palette already has strict selector-ineligible snapshot publishers for the
detection pair and crop-v2. The legacy clipped campaign has not yet composed
those publishers into its main path, so the new keypoint finalizer must remain
an opt-in fragment until that upstream recording-level detection/crop boundary
is present. Detect quality may use a collection-wide intermediate, but it is
not a substitute for the final canonical/refined snapshot pair.

## Implementation Checklist

### Logical schemas

- [x] Add exact float32 keypoint-v2 array contracts and coordinate bindings.
- [x] Implement the raw/refined shared stage schema and cross-array invariants.
- [x] Freeze and implement exact source-bound keypoint-quality arrays, ordered
      metric catalogs, bit registries, policy digest, and cross-array checks.
- [x] Register the keypoint-quality artifact family and dependency/invalidation
      edges without activating production status or selection.
- [x] Freeze exact refined accepted-QC code maps and manifest bindings to the
      quality run used during review.
- [x] Implement body-frame-v1 logical contracts and derivation validation.
- [x] Require exact quality manifest field sets, reconstruct the logical and
      storage builders, and enforce canonical digests.
- [x] Add the exact body-frame run manifest and publication reconstruction
      gate.
- [x] Add the exact raw-keypoint-v2 run manifest and publication reconstruction
      gate.
- [x] Add the equivalent exact run manifest for refined keypoints.

### Writer and lifecycle

- [x] Add the selector-ineligible keypoint-quality shadow writer without
      changing current defaults.
- [x] Add the observation-local keypoint-quality producer and immutable
      publication gate.
- [x] Freeze the exact raw-keypoint-v2 run manifest, byte-derived storage plan,
      selector-ineligible publisher, and legacy-YOLO preparation adapter.
- [x] Validate a deterministic selector-ineligible YOLO-shaped raw-v2 canary,
      including a mask-free keypoint-to-body-frame chain.
- [x] Freeze the representative 23,287-frame / 22,926-row crop-v2 canary
      inputs, durable-cache namespace, node-scratch relocation semantics, and
      atomic selector-ineligible publication driver.
- [x] Materialize and validate the reusable `uint8[22926,512,512]`
      `flat_bin_v1` cache under NRS, including a complete payload SHA-256.
- [x] Publish and measure one representative selector-ineligible YOLO canary
      from a real completed run before inserting the quality DAG node.
- [x] Add the pure refined-v2 producer, byte-derived storage planner, and
      first-snapshot selector-ineligible publication/reopen gate.
- [x] Add exact refined-detection-to-crop successor reconciliation and a
      selector-ineligible immutable crop-successor publisher.
- [x] Add complete raw-keypoint successor preparation and selector-ineligible
      publication: reuse unchanged crop rows, require terminal inference for
      added/changed rows, persist attempted failures, and omit retired rows.
- [x] Make clipped finalization publish raw, quality, refined, and body-frame
      contracts through the same byte planners used by standalone v2
      publications, never through inherited per-clip shard metadata.
- [x] Add exact terminal clip sidecars, a composable LSF finalization fragment,
      and one receipt binding every clip plus crop/raw/quality/refined/body
      manifests; keep the integration bundle selector-ineligible.
- [x] Add composable strict recording-level refined-detection and crop-v2
      publication fragments downstream of the existing canonical publisher,
      and bind their standalone crop output into the recording-level keypoint
      finalizer.
- [ ] Insert those fragments into the maintained clipped campaign around
      pixel-package creation and clip inference. The strict clip-evidence and
      binding fragment is now implemented; final insertion remains coordinated
      with the detection-DAG refactor.
- [ ] Import the four finalized keypoint-family candidates atomically into the
      recording archive while preserving the standalone receipts as evidence.
- [ ] Add a bounded-row DAG materializer that accepts only a validated
      raw-keypoint-v2 manifest, writes complete destination physical units,
      and does not retain duplicate full source/output tables.
- [x] Add the body-frame-v1 producer, exact manifest, and byte-planned
      selector-ineligible shadow publisher.
- [ ] Add the bounded-row body-frame DAG node after refined-keypoint selection.
- [ ] Keep legacy embedded heading readable but never copy it into v2; migrate
      by recomputation into a new body-frame run after numerical/consumer gates.
- [x] Add keyed refined-keypoint deltas and selector-ineligible immutable
      compaction.
- [x] Bind one skeleton-specific manual-QC policy and recompute accepted
      snapshot-local QC through the same evaluator used by live review.
- [ ] Recompute the downstream body-frame snapshot after approved compaction.
- [x] Preserve source crop, pixel-package, skeleton, model, preprocessing, and
      coordinate identities through every clip receipt and the final bundle
      receipt.
- [ ] Keep production selectors unchanged until all gates pass.

The representative canary uses the geometry-only crop snapshot at
`crop_runs/crop_geometry_publication_read_crimson_20260729_v2` in the
2026-01-28 coordinate-catalog fixture. That snapshot intentionally forbids
`roi_images`; its exact pixel authority is the 23,287-frame Cam2010093 source
video and the Orange PyNvVideoCodec luma `uint8` contract. The reusable cache
target is:

```text
/nrs/johnson/palette_staging/flat_roi_cache/
  keypoint_v2_cropv2_20260729_v2/roi_cache/
  20260128_arena1_cam2010093_cropv2.flat_roi_cache.{json,bin}
```

Cache construction accepts an explicitly relocated byte-identical video while
retaining the original crop archive as the cache authority. YOLO may consume
that cache from a byte-identical node-scratch archive copy only when it passes
the original authority archive as the expected cache binding. The canary
builds raw keypoints, observation-local quality, and body frame on node-local
scratch, validates every exact schema and manifest there, copies the complete
workflow to a hidden shared temporary directory, then reveals it with one
same-filesystem rename. It never writes a production selector or registry row.

### Representative crop-v2/keypoint-v2 canary evidence

The real 22,926-row checkpoint completed on LSF job `153230652` at Palette
commit `79e8108f4705e9627888d21b1e4192b345b47722`. The reusable NRS cache is
complete and has this exact identity:

| Field | Result |
| --- | --- |
| Shape and dtype | `uint8[22926,512,512]` |
| Payload bytes | `6,009,913,344` (5.60 GiB) |
| Payload SHA-256 | `f635aab60e840f29f286be786fd103271b2270fa510e045cd8501a0736cb44e0` |
| Manifest SHA-256 | `2fcdd4c7f3bb25fa5517b3e654b8b72a4e2c62a5b87913cc288034d6c38911e6` |
| First materialization | 205.66 s; 111.48 ROI rows/s; 27.87 MiB/s |
| Decoded acquisition frames | 23,287, including 361 frames with no crop row |

The successful cache-reuse workflow is published at:

```text
/groups/johnson/johnsonlab/jeremy/recordings/.palette_benchmarks/
  keypoint_storage/integration/20260128_cropv2_keypoint_v2_20260729_v4/
```

It contains exact raw-keypoint-v2, keypoint-quality-v1, and body-frame-v1
snapshots with manifest digests
`227f0c80065a38d77604b0638bb16a22cd513b383609d364b4481a4fb0cf8db6`,
`3d0af6dab6ca0ddc478c80755c040c2af2381e00166ee8a4cab7f8d9cb920e81`,
and `a8b12539669174bf20ebaf181b0c341148903588fc5ae27af46f94e24a2ab1af`,
respectively. All three reopened publication gates passed. The exact direct
metadata census found 15 raw, 13 quality, and 10 body-frame arrays with the
documented dtypes. At this bounded size every array is one unsharded chunk and
uses the `bytes -> zstd(level=0)` chain; this is the intentional
single/few-object case, not evidence against sharding longer recordings.

Cache reuse and compute/publication timing was:

| Phase | Seconds |
| --- | ---: |
| Stage crop archive to node scratch | 0.21 |
| Stage existing 5.60 GiB cache to node scratch | 1.82 |
| YOLO inference for 22,926 rows | 79.18 |
| Convert legacy producer boundary to raw v2 | 0.34 |
| Validate and publish raw keypoint v2 | 1.14 |
| Validate and publish keypoint quality v1 | 0.51 |
| Validate and publish body frame v1 | 0.54 |

The 1.82-second cache stage was an immediate same-day reuse measurement and is
not a cold-NRS transfer gate; cold staging still needs its own repetitions.
Inference resolved 22,858 poses and retained 68 explicit failed rows. The
float32 conversion introduced zero ROI-coordinate error for this result; the
largest source-camera reprojection difference was 0.000244140625 pixels, below
the frozen 0.001-pixel tolerance. `/usr/bin/time` measured a 7,989,207,040-byte
peak process RSS, which remains an optimization target: reading the durable
flat cache can make mapped cache pages resident even though the model works in
bounded batches.

The source crop archive's metadata fingerprint was identical before and after
the run. The handoff explicitly records no source mutation, selector,
registry, training artifact, or production-state write. The cache is a durable
derived accelerator bound to the exact crop manifest and source-video
identity; it is not pixel authority and is not added to the analysis Zarr.

### Deferred flat-cache RSS optimization

The observed 7.99 GiB peak RSS does not block the logical contracts,
selector-ineligible publication, or Crimson interoperability checkpoint. The
flat cache is read through a read-only NumPy memory map, so RSS may include
reclaimable file-backed pages from the complete 5.60 GiB sequential scan in
addition to anonymous Python, NumPy, PyTorch, and pinned-transfer memory. It
does not establish an 8 GiB heap requirement.

Defer this optimization until the raw-keypoint, keypoint-quality, body-frame,
and refined-keypoint contracts have exact persisted envelopes and Crimson has
typed readers for their selector-ineligible canaries. Before enabling broad
concurrent production, Palette must then:

- sample `/proc/<pid>/smaps_rollup` by workflow phase and distinguish
  anonymous, file-backed, shared, and pinned-host resident memory;
- reconcile process `ru_maxrss`, `/usr/bin/time`, and LSF accounting;
- compare the current memory map with bounded range reads and/or explicit
  release of already-consumed file-backed pages;
- repeat cold and warm NRS-to-scratch staging and inference measurements; and
- freeze a concurrency-aware memory gate for multi-camera execution.

The optimization target is a working set proportional to the active inference
window rather than total cache size while preserving the existing sequential
throughput. It is a production scaling and resource-efficiency gate, not a
reason to weaken or delay the storage schemas.

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

- [x] Publish selector-ineligible raw-keypoint-v2, keypoint-quality-v1,
      and body-frame-v1 fixtures for Crimson. The immutable paths, exact typed
      surfaces, object receipt, and consumer gate are frozen in
      `docs/keypoint_v2_crimson_fixture_contract.md`.
- [x] Publish the selector-ineligible refined-keypoint-v2 and refined-derived
      body-frame-v1 follow-on canary with correction, rejection, and recovery
      cases; freeze its gate in
      `docs/refined_keypoint_v2_crimson_fixture_contract.md`.
- [ ] Add Crimson exact-schema adapters for keypoint v2, body-frame v1, and
      the snapshot-local QC surface.
- [ ] Census consumers of current embedded temporal-heading diagnostics and
      decide which are acceptance inputs versus optional analysis.
- [ ] Prove ordinary playback performs zero optional diagnostic reads.
- [ ] Run Crimson correctness, retained-offset, lazy-optional-read, and bounded
      window benchmarks against the selector-ineligible fixtures.
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
