# Crop-Only Recording Storage Profile

<!-- contract-meta
status: future-design
last_verified: 2026-08-15
purpose: Define the evidence, certification, and irreversible-loss contract required before a recording may retain acquisition crop video without retaining its continuous full-frame video.
-->

## Status and scope

This is a future storage profile. It is not approved for current production
recordings, and it does not authorize deletion of any existing full-frame video.

The profile applies to recordings where Orange produces a frame-continuous
acquisition crop stream from live detector output and an operator eventually
wants to retain that crop stream without retaining the continuous full-frame
camera stream.

This document separates three decisions that must not be collapsed:

1. Which pixels were recorded during acquisition.
2. Which detection observations are scientific inputs to analysis.
3. Which recording artifacts may be removed after validation and archival.

The acquisition crop stream is primary data. It is not a cache: current Orange
crop videos may be lossless and can contain higher-fidelity fish pixels than the
lossy full-frame recording. The full-frame video remains uniquely valuable for
recovering live-detector misses, changing the crop policy, examining context,
and independently validating dish geometry.

## Storage profile vocabulary

Every recording must declare one storage profile. Absence of a declaration is
legacy/unknown and must never be interpreted as permission to remove media.

### `full_plus_crop`

The continuous full-frame stream and acquisition crop stream are both retained.

- Offline full-frame detection can remain the normal detection authority.
- Acquisition detections remain independent acquisition evidence unless
  explicitly promoted through a versioned binding policy.
- Acquisition crop pixels may be used by downstream models when a row-level
  routing contract proves they contain the canonical analysis ROI.
- Missing, blank, or insufficient acquisition crops may be recovered from the
  full-frame stream.
- Independent early/middle/late dish fitting remains possible.

This is the required profile for current GoodBatBadBat production work.

## Crop geometry profile

The recording storage profile and crop geometry profile are separate choices.
For zebrafish, the intended new default is:

```text
zebrafish_crop_384_v1
roi_width_px = 384
roi_height_px = 384
```

The persisted crop geometry remains native 384 x 384. Palette's existing
reversible `ModelInputTransform` may center-pad it to a larger submitted model
extent, such as 512 x 512, and map outputs back by removing the recorded pad.
That transform intentionally does not shrink pixels. Any later framework/network
resize is a separate recorded preprocessing stage and must not change the
full-frame placement or meaning of the source crop.

For acquisition-backed rows, the recorded 384 x 384 crop window and its exact
full-frame origin are the candidate ROI. For full-frame recovery rows, Palette
generates a 384 x 384 ROI from the selected refined detection using the same
versioned zebrafish crop policy. Existing 348 x 348 runs remain immutable legacy
artifacts and are not rewritten.

### `crop_only_certified`

The continuous full-frame stream was initially recorded, the crop stream was
validated against it, an immutable retention decision was approved, and the
continuous full-frame stream was then removed or moved outside the retained
recording package.

- The crop video and its complete frame ledger become irreplaceable primary
  data.
- Live acquisition detections or crop-local offline detections become the only
  pixel-supported fish observations.
- Frames where the crop is blank, missing, or pointed at the wrong location are
  not recoverable.
- Full-frame context cannot be reconstructed from crop pixels.
- A sparse, full-resolution full-frame audit stream remains required for the
  independent dish-geometry policy and temporal camera/arena checks.

No numerical certification threshold is defined here. Thresholds must be
versioned and justified by a bounded canary across cameras, arenas, and repeated
recordings before this profile can become production-eligible.

### `crop_only_direct_experimental`

No continuous full-frame video was retained long enough to certify the crop
stream against an offline full-frame analysis.

This profile is experimental and fail-closed:

- It cannot claim equivalence to `full_plus_crop` or `crop_only_certified`.
- It cannot estimate live-detector misses outside the retained crop.
- It cannot recover blank or misregistered crops.
- It remains selector-ineligible for production analyses that require certified
  continuous fish-position coverage unless a separate reviewed policy permits
  it.
- Sparse full-frame audit imagery is still required for dish-geometry review.

## Acquisition observations are not pre-refined detections

The live acquisition workflow is a separate observation path, not an offline
pipeline with quality and refinement silently skipped:

```text
ephemeral full-frame camera image
  -> live raw detector observations
  -> crop-controller selection
  -> retained crop-video frame and placement ledger
```

Import must preserve the raw observations without claiming that detection
quality, temporal refinement, review, or subject selection has occurred.

When exact frame, camera, coordinate, and model identity are bound, a later
publisher may create a canonical immutable detection run with an explicit
producer such as `orange_live_acquisition`. Detection quality and refinement can
then run after acquisition. Those stages may reject, classify, or impute from
existing evidence; they cannot recreate pixels that were never retained.

Any imputed position must remain distinguishable from a pixel-supported
observation. Interpolation must never create a synthetic raw detection row or
convert an irrecoverable crop miss into observed evidence.

## Required retained artifacts

A crop-only package is incomplete unless all required artifacts below are
present and bound to one recording, camera, arena, frame clock, and native
full-frame extent.

### Crop pixels

- Acquisition crop MP4 or a versioned successor encoding.
- Exact crop-video dimensions, codec, pixel contract, frame count, and frame
  rate.
- Keyframe/index information sufficient for deterministic frame access.
- Recorder summary and terminal status.
- A strong source-media identity. Prefer a producer or transfer content checksum.
  If a bounded fingerprint is used instead, its weaker guarantee must be named
  explicitly and must not be represented as a content hash.

### Complete frame ledger

The consumer-facing Zarr import must preserve every crop-stream frame, including
blank and no-detection outcomes. At minimum it must retain row-aligned values
equivalent to:

```text
source_recording_frame_ids
source_acquisition_frame_index
source_crop_meta_row_indices
source_crop_video_frame_indices
source_crop_local_frame_ids
source_camera_frame_ids, when supplied
acquisition timestamps, when supplied
has_detection
blank_frame
crop_xywh
selected_detection_xywh
selected_detection_confidence
```

`crop_xywh` is the full-frame camera region encoded into the crop frame.
`selected_detection_xywh` is only the live detection selected by the crop
controller. These fields must remain semantically distinct.

The original ledger sidecar must be checksummed and retained even after its
columns are mirrored into Zarr. The Zarr arrays are the stable consumer
contract; the producer CSV remains immutable acquisition evidence.

### Complete live detector evidence

When Orange emits a complete live detection event stream, retain it separately
from the crop-controller selection. The crop ledger may contain only one
selected detection per frame and is not necessarily the complete detector
output.

Required provenance includes:

- model and weights identity;
- preprocessing and pixel contract;
- class mapping;
- confidence/NMS thresholds;
- all recorded candidate or postprocessed detections, as declared by the
  producer contract;
- crop selection policy;
- software and configuration identity;
- per-frame detector/recorder failure status.

### Geometry and coordinate authority

- Recording-bound full-precision dish geometry contract.
- Camera serial, arena identity, native width/height, and native camera pixel
  frame.
- Physical inner-rim boundary and forgiving centroid gate kept separate.
- No presentation reflection, heuristic flip, or mutable current-rig geometry.

### Sparse full-frame audit imagery

Crop-only retention must keep enough native full-frame imagery to run the
independent blind dish fit and inspect temporal stability. The sampling schedule
must be versioned and must cover early, middle, and late recording windows.

The audit imagery does not restore continuous recovery. It exists to preserve:

- independent dish-boundary corroboration;
- camera/arena extent and orientation checks;
- gross crop-placement inspection;
- temporal drift evidence;
- an operator-readable record of full-frame context.

The schedule and minimum image count require canary evidence. This document does
not invent those values.

## Canonical Zarr layers

Crop-only and full-plus-crop recordings should expose the same logical raw
acquisition contract:

```text
analysis/acquisition_video_streams/
  streams/
    crop/
      immutable stream manifest
      ledger_runs/<run>/
        complete per-frame metadata arrays
      source-media identity
      producer-sidecar identities
```

The large MP4 may remain beside the analysis Zarr. Zarr is the canonical index,
coordinate, lineage, and source-selection surface; it does not need to duplicate
the encoded video bytes.

Derived analysis remains separate:

```text
crop_runs/<run>/
  canonical analysis ROI rows
  exact source detection/refinement binding
  per-row pixel-provider routing
  crop-video row lineage or full-frame recovery lineage
```

Stream availability does not select a crop run or prove that any model consumed
crop-video pixels. Each downstream run must record the exact crop run, provider
manifest, source row identities, and media identity it consumed.

## Certification workflow

`crop_only_certified` requires a complete, immutable certification chain:

1. Import the full and crop streams without altering producer artifacts.
2. Canonicalize the complete crop ledger into the analysis Zarr.
3. Validate recording/camera/frame-clock/native-extent identity.
4. Verify crop-video decode, frame count, dimensions, keyframe declarations,
   ledger cardinality, and blank/no-detection semantics.
5. Import and bind complete live detection evidence when available.
6. Run normal full-video offline detection, quality, and refinement.
7. Compare live detections and crop placement against the exact offline refined
   rowset.
8. Measure crop containment, missingness, blank frames, temporal clusters of
   failure, coordinate disagreement, and downstream keypoint/mask parity.
9. Complete independent early/middle/late dish validation from retained
   full-frame audit imagery.
10. Apply a versioned certification policy derived from canary evidence.
11. Require explicit operator approval for non-automatic or borderline cases.
12. Publish a retention decision receipt before any destructive action.
13. Verify the retained crop-only package and its archive copies after the
   continuous full-frame stream is removed.

The retention decision is separate from analysis approval. A successful
detection/crop comparison does not by itself authorize deletion.

## Retention decision receipt

The immutable receipt must include at least:

- recording, camera, arena, and dataset identities;
- previous and selected storage profiles;
- exact full and crop stream identities;
- crop ledger and live-detection identities;
- geometry contract and audit-frame identities;
- certification policy ID/version and all thresholds;
- measured comparison and downstream parity results;
- unresolved warnings and review outcome;
- operator identity, decision, reason, and timestamp;
- exact artifacts authorized for removal;
- exact artifacts required to remain;
- archive/backup verification;
- a human-readable statement of what becomes irrecoverable;
- software commit and command identity.

No deletion tool should infer targets from globs, mutable selectors, or a
recording root. It must consume one approved receipt, revalidate every exact
target, and fail closed on any mismatch. Destructive tooling is outside the
scope of the initial implementation.

## Registry and consumer behavior

The registry should expose storage and evidence state independently:

```text
recording_storage_profile
crop_stream_inventory_status
crop_stream_canonicalization_status
live_detection_binding_status
crop_certification_status
full_frame_recovery_available
geometry_audit_status
retention_decision_status
```

Crimson and other consumers should use two interfaces:

- the raw acquisition stream for playback, crop-placement inspection, and
  acquisition evidence;
- ordinary Palette crop/keypoint/mask run contracts for derived analysis.

Consumers must display an explicit warning for crop-only recordings that
continuous full-frame recovery is unavailable. They must not parse Orange CSVs
or infer crop-video use from stream availability.

## Promotion requirements

Before `crop_only_certified` becomes production-eligible:

- validate multiple cameras, arenas, recording days, and behavior conditions;
- include hard cases with blank crops, edge-of-dish fish, rapid motion, and live
  detector disagreement;
- measure positional, keypoint, subject-mask, and scientific-analysis parity;
- validate the sparse full-frame audit schedule against the blind dish fitter;
- set numerical thresholds from evidence rather than convenience;
- complete Crimson/read-tool compatibility checks;
- prove backup, restoration, and retention-receipt behavior;
- pass required CI and a commit-pinned canary deployment.

## Non-goals

- This profile does not authorize deleting current GoodBatBadBat full videos.
- It does not make live detections equivalent to offline refined detections.
- It does not treat blank crop frames as reviewed negative biological evidence.
- It does not permit acquisition-only dish geometry without independent review.
- It does not require storing encoded video bytes inside Zarr.
- It does not define certification thresholds before canary measurements exist.

## Related contracts

- [Acquisition video stream source policy](acquisition_video_stream_source_policy.md)
- [Orange runtime video artifact contract](orange_runtime_video_artifact_contract.md)
- [Acquisition crop-video ROI provider plan](acquisition_crop_video_roi_provider_plan.md)
- [Crop geometry storage contract v1](crop_geometry_storage_contract_v1.md)
- [Raw video storage tiering proposal](raw_video_storage_tiering_proposal.md)
- [Recording-bound geometry import and validation design](recording_bound_geometry_import_and_validation_design.md)
