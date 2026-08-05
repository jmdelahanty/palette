# Sampled Training Detection Source Contract

Status: implemented reader support for maintained, already-published sampled
training runs.

This contract exists for training archives where a conservative subset of
full-camera detections was selected to define new materialized instance crops.
It does not turn that subset into a normal detector stage result and does not
claim anything about unsampled camera frames.

## Source and consumer surfaces

`sampled_detection_runs/<run>` is a canonical geometry source, not a
stage-selectable detection run. It must be complete, use
`coordinate_contract == "canonical_v2"`, and remain explicitly
`stage_selector_eligible == false`.

A canonical `crop_runs/<run>` may select this rowset through the same
`instance_key`-bound crop selection record used for an ordinary
`detect_runs/<run>` source. The crop is a separate, complete,
selector-eligible materialized surface. Keypoint and subject-mask readers load
the crop without copying its pixels or geometry arrays; the crop loader follows
the sealed source reference and validates the sampled source first.

The family names are intentionally distinct. A sampled source must not be
renamed, copied, or linked under `detect_runs` merely to satisfy a reader.

## Row identity and future multi-subject use

Each row is one detection instance. Its `instance_key` is minted from the
recording identity, parent acquisition-frame index, detector bbox, and class
ID. It is not a biological-subject identity or a track identity.

Later `track_id`, association, or subject-identity corrections may therefore
refer to these instances without rewriting the crop, keypoint, or mask
observation identity. Multiple instances in a camera frame are compatible with
the observation model. The current strong-single sampled subset is
deliberately conservative: ambiguous multi-candidate rows remain excluded
rather than silently assigning an identity.

## Required proof

The rowset schema is `palette.sampled_training_detection_run.v1` with
`source_kind == "strong_single_full_frame_detection_selection"`. Its
`sampled_training_detection_selection` record must digest-bind and exactly
recompute:

- the source `detect_runs/<run>` frame, bbox, score, and class arrays;
- the proposal `crop_runs/<run>` frame, bbox, and crop-placement arrays;
- the canonical strong-single policy;
- the all-proposal-row reason-coded selection receipt; and
- every accepted instance, frame, bbox, score, class, and source-row array.

Policy v1 remains loadable for existing publications. Policy v2 additionally
requires the full target crop window to lie inside the camera frame, preventing
padded pixels from being presented as an ordinary camera-frame crop.

A reader fails closed if the run is incomplete or selector-eligible, the
schema or policy is not canonical, any source or output array differs from
recomputation, the stored proof is stale, or the crop selection no longer
refers to the exact sampled `instance_key` array.

## Publication scope

This change makes the shared strict publication and loading primitives
available and allows canonical crop consumers to follow an existing sampled
source. It does not publish, rewrite, promote, or repair any Zarr run. A future
writer or migration must stage the sampled rowset and crop as new immutable
runs, validate them through a fresh read, and expose only the crop as
selector-eligible after all proof records are complete.
