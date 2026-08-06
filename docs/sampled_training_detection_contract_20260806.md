# Sampled Training Detection Contract — 2026-08-06

Status: implemented for selector-ineligible training canaries.

## Decision

A sampled training Zarr does not share the recording-level canonical detection
frame axis. Its review UI uses local rows `0..F-1`, while scientific identity
must remain tied to acquisition frames. Palette therefore publishes a distinct
sampled-training detection contract rather than weakening or pretending to
satisfy canonical detection v1.

The strict table is `detect_runs/<run>/instances` with the same nine exact
typed arrays used by canonical detection:

- `frame_indices` is the local sampled-row index.
- `source_acquisition_frame_index` equals
  `raw_video/original_frame_indices[frame_indices]` exactly.
- `instance_key` is derived from recording identity, acquisition frame,
  authoritative normalized box, and class.
- `frame_row_offsets` has shape `F+1` and indexes zero, one, or many detections
  per sampled frame.
- normalized boxes are authoritative source-camera geometry; pixel boxes and
  centers are exact derived float32 projections.

One-fish-per-frame is a dataset expectation, not a destructive writer rule.
The publisher records zero/one/multiple counts and never forces `max_det=1` or
drops duplicate candidates during inference. A later refined seed may apply
top-1 while retaining every source row in its audit table.

## Lifecycle

Checklist:

- [x] Copy and content-authenticate the complete training base on node-local
  scratch.
- [x] Run the registered detector into an immutable, selector-free
  `detection_artifact_runs/<run>`.
- [x] Atomically import that artifact without selectors.
- [x] Recompute and validate the frame mapping, geometry, stable keys, logical
  hashes, direct metadata declarations, codecs, chunks, and shards.
- [x] Build an access-aware bound candidate on node-local scratch.
- [x] Atomically publish it under `detect_runs/<run>`.
- [x] Keep both runs selector-ineligible and unregistered.
- [x] Keep the still-editable training root unconsolidated.
- [ ] Initialize a selector-ineligible refined review seed after inspecting
  zero/one/multiple detector cardinality.
- [ ] Review/correct detections, approve for training, then publish crop,
  keypoint, and subject-mask training surfaces.
- [ ] Consolidate only when the complete training artifact becomes immutable
  and selector-visible.

Implementation:

- `fisheye.utils.run_sampled_training_detection_canary`
- `fisheye.shared.zarr.sampled_training_detection_publication`
- `fisheye.shared.detection_tables`

The compatibility reader resolves modern columns from `instances/` and legacy
columns from the run root. New writers do not create root aliases. Manual
curation preserves copied keys and mints new keys using mapped acquisition
frames, preventing sampled datasets from inventing incompatible identities.
