# Chaser gaze-tracking workflow

## Decision

Chaser gaze tracking is measured entirely in the fish body frame. The response
is `left/right_gaze_signed_deg`, where zero is fish-forward and positive is
anatomical left. The predictor is egocentric chaser `bearing_deg` under the same
convention.

Do not use world-frame gaze alignment as evidence of eye tracking: a fish that
turns its body toward a chaser can create apparent world-frame alignment without
moving either eye. Do not regress the nasal-positive
`left/right_eye_angle_deg` presentation fields against chaser bearing; those
fields intentionally have different left- and right-eye sign transforms.

The recording-level component schema is
`palette.chaser_gaze_tracking.v1`. It is stored under the exact immutable
chaser-distance run that supplies object positions, roles, epochs, and the
egocentric-bearing component:

```text
analysis/chaser_distance_runs/<distance-run>/gaze_tracking/<component>/
├── frames/
├── objects/
├── epochs/
├── recording_summary/
├── virtual_controls/
├── object_vs_virtual/
├── distance_bearing_summary/
├── lock_on_events/
└── visualizations/
```

The chaser axis is variable length and role labels are protocol-derived. The
analysis supports one, two, or more chasers and does not infer role from array
position.

## Required gates

Run `fisheye.analysis.gaze_convention_validation` before computing gaze error.
It samples bounded physical row windows from a compact eye-angle run and checks:

- canonical body-frame and gaze metadata;
- unit, orthogonal, correctly handed forward/left axes;
- heading reconstructed from the forward axis;
- left and right nasal-positive sign identities;
- vergence as the sum of nasal-positive eye angles;
- directed gaze from the resolved ellipse major axes; and
- stored gaze vectors reconstructed in the fish body frame.

The gaze analysis separately compares the eye run's stored body heading with
the track heading used to compute egocentric bearing. It refuses output if the
median absolute mismatch exceeds 20 degrees or the circular resultant length is
below 0.8.

Ellipse axes are directionless. Palette directs the eye axis after resolving
the major axis into the fish-forward half-plane. The numeric identities above
verify internal consistency but cannot independently establish that biological
direction. Every newly introduced acquisition family therefore requires a
bounded visual overlay review. The review image is a gate artifact, not a claim
that all frames were manually inspected.

Modern `compact_dense_v2` eye runs expose two row semantics. `roi_angles` is a
sparse keypoint-detection-row table joined to camera frames through
`support/frame_indices`. `frame_angles` is a dense camera-frame projection and
is the source used for gaze tracking. That projection can end at the last frame
represented by an eye-detection row, before the recording's final frame. Gaze
tracking bounds-checks camera frame IDs and marks any uncovered tail frames
unavailable; it never indexes beyond the shorter projection.

## Metrics and controls

Eye-accessible bearing ranges are estimated separately for each recording and
eye from the 1st and 99th percentiles of valid gaze. Static tracking gain is the
OLS slope of body-frame gaze on body-frame chaser bearing within that range.
The dynamic summaries regress wrapped frame-to-frame gaze changes on wrapped
bearing changes at zero lag and at causal lags only. A positive lag means that
the eye follows the bearing change.

A lock-on frame must be valid, within the empirical eye range, no farther than
50 mm from the chaser, and have absolute wrapped gaze error no greater than 10
degrees. Persisted lock-on events last at least 0.10 seconds. These defaults are
stored in component provenance and can be changed explicitly.

For each real chaser, virtual references rotate its position about the circular
arena centre. This preserves radial position and wall proximity while moving
the reference to an empty location. References that collide too often with a
real chaser are excluded. The primary recording summary reports real-minus-mean
virtual gain and lock occupancy, plus improvement in median absolute error.

Frame rows are descriptive measurements, not independent replicates. Cohort
inference uses one summary per recording/fish, with session or fish-level
handling when repeated measurements exist. No frame-pooled p-values are
produced.

## Commands

Read-only convention validation and preview:

```bash
scripts/py -m fisheye.analysis.gaze_convention_validation \
  /path/to/recording_analysis.zarr \
  --eye-angle-run latest \
  --review-png /tmp/gaze_convention_review.png \
  --json-output /tmp/gaze_convention_validation.json \
  --fail-on-error

scripts/py -m fisheye.analysis.chaser_gaze_tracking \
  /path/to/recording_analysis.zarr \
  --eye-angle-run latest \
  --chaser-distance-run latest \
  --egocentric-component latest \
  --preview-png /tmp/chaser_gaze_tracking.png
```

An exact protocol cohort can be snapshotted from the registry and submitted
through the Citrus poller. This example selects the 32 source recordings in the
current GoodCopBadCop analysis cohort that have the modern movement and geometry
prerequisites:

```bash
scripts/submit_chaser_protocol_analytics_bsub.sh \
  --source registry \
  --registry /groups/johnson/johnsonlab/jeremy/registries/palette_registry.sqlite \
  --zarr-use analysis \
  --zarr-origin source \
  --protocol-name GoodCopBadCop \
  --require-step-ok track_kinematics \
  --require-step-ok refined_subject_masks \
  --require-step-ok refined_keypoints \
  --eye-and-gaze-only \
  --preset goodcopbadcop \
  --run-id goodcopbadcop_gaze_v1_20260714
```

The submitter writes the exact selected paths and normalized filters into an
immutable manifest before submission. Each task writes one Zarr only, validates
the eye conventions, and marks the gaze component complete only after all
arrays and the bounded summary PNG have succeeded.
