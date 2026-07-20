# Crimson coordinate implementation work package

Date: 2026-07-19

Status: implementation handoff draft. This document does not authorize changes
to Crimson or production archives. The authoritative cross-repository read
contracts remain normative; this package orders the Crimson work after the
corresponding Palette publishers and contracts pass final review.

## Outcome

Crimson should have one future-normal coordinate boundary that:

- validates Palette's compact array-owned canonical descriptor and digest;
- resolves every referenced identity, frame, extent, transform, derivation, and
  publication record by exact path and digest;
- reads only complete, selector-eligible, freshly validated immutable children;
- transforms supported geometry to source-camera pixels only through an ordered,
  direction-labelled chain; and
- creates viewport/display coordinates only as ephemeral renderer state.

New recordings must not invoke a legacy adapter. Historical inspection remains
a separately named, visibly unverified mode and cannot feed normal scientific
or presentation publication.

## Deployment prerequisites

Do not enable a Crimson future-normal reader until all of these are true for its
surface family:

1. the Palette writer emits canonical descriptors directly rather than adapting
   `camera`, `texture`, dimensions, names, or numerical ranges;
2. the Palette strict loader freshly reconstructs and validates the same closed
   publication that Crimson will validate;
3. the authoritative contract is merged with exact schema IDs, versions,
   controlled profiles, identity domains, and activation gates;
4. a real producer-to-reader fixture and hostile mutation fixtures pass in both
   repositories; and
5. any required schema-field addition, including `collection_axis`, is deployed
   coherently so an older strict parser fails closed instead of silently dropping
   semantics.

## Shared native validation layer

Implement one native Crimson module for digest-bound Palette coordinate
evidence. Avoid separate ad hoc parsers in bbox, keypoint, track, mask, and shape
views.

The module must:

- parse canonical JSON with exact required/allowed fields and reject duplicate,
  unknown, or incorrectly typed fields according to the declared schema version;
- recompute canonical SHA-256 digests rather than trusting stored digest strings;
- resolve canonical archive-root references without traversal or selector
  ambiguity;
- bind one exact archive identity and reject cross-archive references;
- validate row identity, optional collection-axis identity, reference extent,
  frame authority, overlay status, and ordered transform endpoints;
- validate array dtype, shape, payload digest, controlled attrs, and publication
  inventory before returning values;
- pin the exact selected child, copy requested values, and revalidate it after
  the read; and
- return typed unavailable/unsupported/stale errors without coordinate fallback.

The scientific and rendering code should receive ordinary typed arrays only
after this boundary succeeds.

## Reader implementation order

### 1. Track motion

Implement `palette-crimson/track_motion_read.md` first because it is both a
source-camera overlay and a shared scientific input.

- Resolve an explicit scope and candidate run. Treat `latest` only as discovery.
- Require completion, selector eligibility, canonical binding status, the exact
  full-motion manifest, and publication commit.
- Build row lookup from `track_sample_key` and
  `source_acquisition_frame_index`; do not use row offset or
  `source_instance_key` as sample identity.
- Read pixel and optional physical positions from their array descriptors.
- Expose derived motion only through sealed logical-surface records, including
  destination-sample transition semantics and gap validity.
- Permit source-video overlay only for direct source-camera coordinates or a
  fully supported persisted chain to that exact camera frame.

Palette's scoped producer/strict-reader review is now `GO` on the remediation
branch. Do not expose canonical track-motion support until the authoritative
contract is merged and the same hostile fixtures pass in Crimson.

### 2. Detection bounding boxes

Replace bbox-name and dimension inference with the final
`palette-crimson/detect_bbox_read.md` contract.

- Require source-image `bbox_img_xyxy` with `bbox_xyxy` geometry and
  `pixel_edge_half_open` convention for source-camera display.
- Bind the exact observation `instance_key`, acquisition-frame mapping, source
  camera extent, and bbox projection/center derivation.
- Treat normalized or model-input boxes as different profiles. Never scale them
  with video, viewport, or nearby run dimensions.
- Apply no conversion unless its exact directed transform and endpoints validate.

### 3. Keypoints and manual edits

Implement canonical raw-keypoint reading as its own first slice after the final
raw contract is merged. Palette's raw publisher/loader now binds exact model
content to the ordered pose schema; refined/manual reading remains a separate,
later capability and must not block or silently replace raw selection.

- Preserve the distinction among ROI-local, model-input, source-image, and
  normalized arrays.
- Bind label/collection identity separately from observation `instance_key`.
- Require the exact model-to-pose binding, including ordered labels, model and
  manifest digests, and every populated registry schema field.
- Require the exact ROI placement for ROI-to-camera presentation.
- Replace direct Crimson mutation with the final edit-request/successor protocol.
  Crimson must not overwrite a completed Palette coordinate authority in place.

The final raw-keypoint contract must make the point/edge split mechanical:

- `keypoints_roi`, `keypoints_img`, and `keypoints_norm` are collected
  `point_xy` surfaces with continuous convention. Each descriptor owns
  `collection_axis = {axis: 1, role: "keypoint", cardinality: K,
  label_authority: {record_ref, record_sha256}}`.
- `pose_bbox_xyxy_roi` and `pose_bbox_xyxy_img` are `bbox_xyxy` surfaces with
  `pixel_edge_half_open` convention. `pose_bbox_xyxy_norm` is numerically
  continuous normalized geometry, but its transform chain must terminate at
  the half-open source-camera bbox frame, never the continuous point frame.
- `source_crop_xywh` is half-open `bbox_xywh`. Bbox and crop-placement arrays
  do not carry the keypoint collection axis.
- The ROI point chain begins at the continuous `roi_images` authority. The ROI
  bbox chain begins at the distinct run-local
  `coordinate_frames/roi_bbox_edge` authority. Sharing dimensions does not
  make those records interchangeable.

Model identity lives at
`keypoint_coordinate_context.model_artifact.pose_schema_binding`, not in a
numeric `K`, package default, run summary, or familiar label order. Crimson
must validate schema/version/canonicalization, binding digest, exact model SHA,
ordered labels/nodes/edges, model `[K,D]`, runtime `[K,2]`, and equality to the
collection-axis label authority. A registered binding uses
`registered_training_manifest_v1` with manifest-primary/all-populated-registry-
fields-agree policy. An explicit reviewed binding uses
`explicit_digest_bound_assertion_v1`, requires an assertion ID, and must not
claim registry or manifest provenance. Crimson validates the sealed in-archive
binding; it does not reach back into Palette's registry or training filesystem.

Renderer heading is available only when the validated pose schema carries a
non-null controlled `heading_computation`; named keypoints by themselves do not
authorize an anatomical heading. Hostile fixtures must include same-`K` label
reordering, wrong model SHA, binding-digest tampering, point/bbox frame swaps,
and the boundary distinction that bbox `x_max == width` is valid while point
`x == width` is not.

### 4. Subject masks and subject shape

Palette's scoped mask/shape producer and strict-reader review is `GO` on the
remediation branch. External support still waits for the authoritative contract
and cross-repository fixtures.

- Treat dense refined `masks_roi` as the scientific/edit authority and compact
  bitpacked/RLE surfaces as derived caches only.
- Validate observation identity, subject-component collection identity, ROI
  extent, and ROI-to-source-camera transform before display.
- Interpret bounding boxes as half-open pixel-edge geometry.
- Distinguish one point per observation (`point_xy`), one point collected per
  component (`point_xy` plus `collection_axis`), and multiple points per
  observation (`points_xy`).
- Validate body origins as points and anatomical axes as unitless `vector_xy`;
  never translate vectors with point transforms.

### 5. Stimulus and chaser-distance surfaces

The remediation branch removes the generic homography/scale fallback from the
canonical chaser-distance writer and requires exact typed preflight in normal
consumers. Enable Crimson only after the corresponding authoritative contract
is merged. Derived chaser components and dashboard artifacts remain unavailable
until they receive their own sealed semantic authorities.

- Require canonical stimulus-state identity and exact acquisition-frame mapping.
- Require direction-labelled source-camera/canvas/arena transforms bound to the
  active camera and selected calibration.
- Require a sealed projector-pixels-per-mm authority for arena-canvas distances;
  never use a median state-row value or infer scale from a resolution ratio.
- Validate fish and chaser inputs in the same arena-relative frame before using
  distance results.

## Renderer boundary

The coordinate validator may return either source-camera geometry or an exact
supported chain that produces it. The renderer may then compute:

```text
source-camera pixels -> fitted video rectangle -> viewport/device pixels
```

That last transform belongs to the live Crimson view. It must not be written
back into Palette descriptors, lineage, registry metadata, or scientific
results. Resizing a window must therefore have no effect on persisted
coordinate authority.

## Required cross-repository conformance fixtures

Use small immutable fixtures generated by real Palette writers, plus hostile
copies that change exactly one fact. At minimum cover:

1. plausible ROI-local values rejected as direct camera points;
2. non-zero crop offsets and unequal ROI/source extents;
3. a non-self-inverse projective matrix in both correct and reversed direction;
4. source or target extent substitution;
5. stale descriptor, frame, lineage, payload, attrs, and publication digests;
6. swapped/missing observation, track-sample, acquisition-frame, and collection
   identities;
7. `point_xy` versus `points_xy` versus collected `point_xy` shape rules;
8. half-open bbox edges and pixel-center point semantics;
9. pixels/mm versus mm/pixel inversion;
10. unsupported texture/canvas/projector/arena/body profiles;
11. exact-child replacement during a read while an unrelated selector change
    leaves an already pinned child valid;
12. a future recording that succeeds with the legacy module disabled; and
13. a viewport resize that changes only ephemeral renderer state.

Palette and Crimson should compare canonical descriptor/manifest JSON and
digests for the same fixtures, not merely compare rendered numerical output.

## Historical inspection quarantine

If retained, historical support must live behind an explicitly named command or
UI mode such as `unverified_legacy_inspection_only`. It may show arrays and
their uncertainty, but it must not:

- participate in normal run selection;
- mint canonical descriptors;
- authorize source-camera overlays from `camera`/`texture` labels;
- synthesize physical coordinates;
- feed a new Palette or Crimson scientific publication; or
- be required by a future recording.

This quarantine lets future acquisition remain simple while preserving a
deliberate path for archive triage. It is not a permanent adapter dependency for
normal Crimson operation.

## Acceptance and release gates

For each reader family, require:

- authoritative contract merged;
- Palette producer, strict loader, and independent review GO;
- Crimson native parser and hostile conformance suite green;
- no fallback from a failed canonical read to a legacy path;
- explicit user-facing unavailable/unsupported/stale states;
- legacy module disabled in the future-recording integration test; and
- no production archive mutation during validation.

Roll out per surface family. A reader that has not met these gates stays
unavailable; another validated family need not wait for it.

## Out of scope for this package

- applying the Palette migration manifest;
- rewriting or deleting historical arrays;
- modifying the production registry;
- recreating scientific outputs whose historical lineage is ambiguous; and
- defining viewport coordinates as a persisted Palette space.
