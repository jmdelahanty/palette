# Subject-Mask Recording Coordinate Aggregation

Date: 2026-08-04

Status: selector-ineligible contract and strict bundle-authority reader
implemented; no production archive, selector, registry authority, physical
profile, or canonical data was changed. A future recording-level canary and
downstream subject-shape adapter remain required.

## Decision

Canonical coordinate ownership belongs to recording-level subject-mask bundle
publication, not to individual clip/shard finalization.

`subject_mask_shard_runs` and their refined counterparts are noncanonical work
packages. They do not contain the rich historical coordinate-publication nodes
needed by the old single-run publisher, and the collection finalizer does not
know the authoritative whole-recording crop axis. Changing
`future_canonical` in `finalize_subject_masks.py` would therefore assert proof
that the collection does not possess.

The safe publication sequence is:

1. Each worker emits a complete scientific identity, immutable attempt, and
   semantic receipt over its whole owned row interval.
2. Recording assembly proves the ordered shard union is exactly `[0, R)` and
   reconstructs identity, frame-offset, and placement arrays from crop v2.
3. The raw recording core binds that exact crop-v2 manifest and retained worker
   evidence in subject-mask core manifest v3.
4. The refined recording core binds the same crop authority and the exact raw
   core from the same publication attempt.
5. Bundle cross-validation binds both coordinate catalogs, both recording
   assembly identities, the crop authority, and the refined-to-raw edge.
6. Members remain selector-ineligible until the single bundle authority is
   explicitly activated.

Historical core manifest v2 and validation receipt v1 remain readable only as
explicit compatibility surfaces. They are not silently upgraded and cannot be
retrofitted into coordinate authority because they discarded semantic evidence.

## Implemented Contract

The raw and refined logical schemas now expose an exact
`palette.array_coordinate_catalog` derived from their frozen array contracts.
The catalogs correctly distinguish raw probability raster authority from dense
refined raster authority while sharing ROI placement and derived metric surface
semantics.

Coordinate-aware core manifest v3 adds two closed fields:

- `coordinate_contract`: the exact digest-bound schema catalog;
- `coordinate_dependencies`: the exact crop-v2 identity, logical-content and
  coordinate-catalog digests, the recording source-receipt and producer-evidence
  digests, and—on refined cores—the exact raw-core manifest/catalog binding.

The dependency builder re-hashes the live crop `instance_key`,
`source_acquisition_frame_index`, `frame_row_offsets`, and `source_crop_xywh`
arrays and requires equality with the crop-v2 logical-content manifest. A valid
JSON envelope alone is insufficient.

Recording source-validation receipt v2 retains the complete recording assembly
identity. Each ordered worker entry contains its full scientific identity,
attempt, and semantic receipt rather than only three opaque digests. Validation
reconstructs every nested digest, run binding, local/global interval, and exact
`[0, R)` coverage.

The recording identity also freezes a common scientific-authority projection.
Worker-local pixel hashes, package IDs, row identities, and clip paths may vary;
model/refinement policy, preprocessing/coordinate semantics, component policy,
ROI pixel domain, and other recording-wide semantics may not. Conflicting
worker model/policy evidence fails before publication.

`publish_recording_subject_mask_bundle()` now requires crop-v2 by default.
`legacy_allow_missing` exists only for explicit compatibility callers and unit
fixtures. When crop v2 is present, the publisher emits raw/refined core v3 and
the bundle cross-binding proves their shared crop and exact refined-to-raw edge.

The backend-neutral
`load_recording_subject_mask_coordinate_authority()` reader accepts only one
of two explicit policies: the single activated root bundle authority, or a
named inactive bundle when the caller opts into `allow_inactive=True` for a
benchmark/canary. It revalidates the complete bundle, both core-v3 members, the
crop-v2 manifest, both source-receipt sidecars, every retained worker record,
and their digest chain before returning a sealed authority object. It never
uses family `latest` pointers and never treats an unselected run as implicit
authority.

The sealed object also exposes only the subject-shape inputs that can be
proved from this contract: exact `instance_key`, crop-row, acquisition-frame,
frame-offset, and placement nodes; recording/camera/source-frame dimensions;
and the dense ROI raster dimensions. Its translation-only accessor rejects a
crop whose source width/height differs from the dense ROI extent. That makes
the current ROI-to-camera translation safe without granting authority for a
future resize, pad, affine, or projective transform.

Refined core dependencies additionally retain an ordered recording-level
assignment-keypoint collection. Each worker interval binds its exact keypoint
group/run, success surface, crop-row selection proof, and optional canonical
coordinate records. Recording-wide semantic comparison deliberately removes
those worker-local identifiers, but the collection keeps them under the core
digest. This supports either one shared recording keypoint authority or exact
clip-local authorities without silently pretending they are the same run.

`load_subject_shape_bundle_source()` is the next sealed boundary. It converts
only a validated bundle-v3 authority into a versioned subject-shape source
record, retains the exact component-channel order and assignment-keypoint
collection, and exposes typed point/box translation helpers. The adapter does
not construct or masquerade as the historical
`BoundRefinedSubjectMaskCoordinateSurfaces` type.

The subject-shape logical array layout may remain v4 only if the publisher
reconstructs every existing semantic guarantee from this new source record.
Because multi-clip assignment may legitimately bind distinct keypoint runs,
the conservative publication target is a new source-binding/derivation
manifest version and, unless a recording-level equivalence proof restores the
old single-source semantics, subject-shape profile v5/method v12. The output
camera-pixel/body-frame coordinate contract itself need not change.

## Validation

The focused outside-sandbox real-Zarr matrix passed 41 tests covering the
publisher foundation, followed by a dedicated active/inactive strict-reader
integration gate covering:

- retained worker evidence and exact assembly coverage;
- conflicting worker scientific authority rejection;
- legacy core-v2/receipt-v1 compatibility;
- raw and refined coordinate-core-v3 publication;
- crop live-array-to-manifest equality;
- refined-to-raw binding;
- recomputed-digest coordinate-catalog tampering;
- crop-v2-required fail-closed behavior;
- inactive recording-level bundle import, cross-binding, activation, and
  recovery behavior;
- explicit inactive authorization, default rejection of unselected bundles,
  exact activated-root-authority resolution, and exact translation-only
  subject-shape source geometry;
- distinct clip-local keypoint-assignment authorities with complete ordered
  crop-row coverage.

Static validation also passed Ruff, Python compilation, and `git diff --check`.

## Still Open

This checkpoint does not make the existing full-duration Sleepyfish mask canary
canonical. Its v2 manifests do not retain the missing evidence, so it must be
republished from the still-verifiable worker/crop inputs or recomputed.

The remaining safe sequence is:

- add the versioned subject-shape publication/derivation path that consumes
  the sealed bundle source alongside the unchanged historical rich-coordinate
  reader;
- bind the recording/camera/frame-axis identity required by subject-shape v4;
- prove a realistic multi-clip canary, including an empty-only frame window;
- materialize a selector-ineligible subject-shape-v4 source;
- resume eye-angle-v7 and tail-kinematics-v2 short/full query-export matrices;
- obtain Palette and, where user-facing, Crimson consumer evidence before any
  activation or physical-profile promotion.

Large dense raster arrays remain governed by whole-shard publication ownership.
Parallel logical row slices are not sufficient when two writers could touch the
same physical shard.
