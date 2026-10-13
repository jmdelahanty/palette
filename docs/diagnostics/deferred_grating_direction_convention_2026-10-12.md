# Deferred: grating alignment mixes two angle conventions

Status: **deferred** (Jeremy, 2026-10-12). The contract work for the incoming
behaviour cohorts comes first. Nothing here blocks intake.

## Defect

`src/fisheye/analysis/stimulus_response_grating.py`

- `resolve_grating_direction` (line 25) returns the Citrus canvas angle plus a
  fixed offset and calls it "camera-space". That canvas angle is the retired
  `canvas_y_down_clockwise_legacy` convention, where +90 points down the screen.
- `compute_grating_per_frame` (line 122) computes
  `heading_step - grating_dir_deg`. `heading_step` is Palette's heading
  (`atan2_negative_y_x_degrees`, counter-clockwise), so one subtraction mixes a
  counter-clockwise angle with a clockwise one.
- The fixed offset ignores reflection. The omnifin0 arena 1 canvas→camera
  homography has a y reflection, and agent-contracts #91 says directions must
  be mapped through the homography Jacobian. Angles must never be subtracted
  across rasters.

**Effect:** the alignment angle is correct only for directions that a sign
flip leaves unchanged (0° and 180°, i.e. left and right gratings). Any other
direction comes out mirrored. agent-contracts #91 lists this module as a known
violation.

Callers: `stimulus_response.py:79`, `stimulus_response_omr.py:14`.

## Work when picked up

1. **Scope first, no code.** Find the recordings with `MOVING_GRATING` steps
   whose direction is not 0 or 180, and the published runs and exports that
   consumed `compute_grating_per_frame` on them. If there are none, the fix
   carries no science impact.
2. **Fix.**
   - Map the grating direction into source-camera pixels through the
     recording's canvas→camera homography (two projected points, or the
     Jacobian), using `fisheye.shared.heading_convention`.
   - On unified files, select the convention from the admission binding: angle_v1
     reads the stimulus angles natively. Legacy files go through
     `legacy_canvas_angle_to_degrees`.
   - Remove the fixed offset. Bump the output schema.
3. **Remove the CI allowlist entry**, if the heading ratchet gains one for this
   module.
4. **Republish** whatever step 1 found, if anything.

## Why deferred is safe for now

The new cohorts land on admission revision 5 (angle_v1). Their raw angles are
stored correctly, so the bug only affects this derived alignment metric, and a
recompute can repair it later. It does not affect intake, admission or
the frame clock.
