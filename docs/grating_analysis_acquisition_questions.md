# Grating Analysis: Acquisition Validation Checklist

Purpose: confirm the spatial/angular relationship between the projector stimulus
and the camera image so that grating response analysis can correctly compare fish
heading (camera space) to grating drift direction (texture/projector space).

This doc is intended to be shared with the acquisition agent / Citrus developer
to get definitive answers. Items marked **[CONFIRM]** need a yes/no or value.
Items marked **[REQUEST]** are things we'd like logged or exposed that aren't today.

---

## 1. Angular Relationship: Projector vs Camera

### What we know
- Fish heading is computed in **camera space** (4512×4512) using
  `atan2(-dy, dx)` from swim bladder → eye midpoint. Convention: 0° = right,
  90° = up (standard math, y-axis inverted from image coords).
- Grating `orientation_degrees` is specified in **texture space** (358×358)
  in the protocol JSON. It represents the drift direction of the grating.
- A `coordinate_transform` attribute stores a linear scale factor
  (4512/358 ≈ 12.6×) but **no rotation or flip information**.
- A 3×3 **homography matrix** exists in
  `/calibration_snapshot/<camera_id>/homography_matrix_yml` that maps
  projector→camera, but it's an opaque matrix — we haven't decomposed it.

### What we need confirmed

- **[CONFIRM] Is there a mirror in the optical path between the projector
  and the camera?** (e.g., projector projects downward onto a mirror, camera
  looks up through the dish, or similar). If yes:
  - Which axis is flipped (horizontal, vertical, or both)?
  - Is this flip consistent across both rigs (omnifin0 and omnifin1)?

- **[CONFIRM] What is the physical orientation of the projector relative to
  the camera?** Specifically:
  - Is the projector's "right" (positive x in texture space) aligned with
    the camera's "right" (positive x in image space)?
  - Is the projector's "up" (negative y in texture space) aligned with the
    camera's "up" (negative y in image space)?
  - Or is there a 90°/180°/270° rotation between them?

- **[CONFIRM] What does `orientation_degrees = 0` mean for a MOVING_GRATING?**
  - Which direction do the bars drift on screen? (rightward? upward? other?)
  - What angle convention is used? (0° = rightward and increasing
    counter-clockwise? clockwise?)

- **[CONFIRM] Is the angular relationship between projector and camera
  identical for omnifin0 and omnifin1?** Or does each rig have a different
  rotation/flip due to physical setup differences?

- **[CONFIRM] Does the homography matrix stored in the calibration snapshot
  faithfully capture all of the above (rotation, flip, scale)?** If so, we
  can decompose it programmatically rather than hardcoding offsets. But we
  need to know whether it's been validated for angular accuracy (not just
  positional/scale accuracy).

---

## 2. Orientation Convention for Grating Drift

### What we know
- The protocol parameter is `orientation_degrees` (or fallback names
  `angle_degrees`, `grating_orientation`).
- It's a float, 0–360.

### What we need confirmed

- **[CONFIRM] Does `orientation_degrees` specify the direction of drift
  (motion direction of the bars), or the orientation of the bars themselves
  (perpendicular to drift)?** For example, if bars are vertical and drifting
  rightward, is `orientation_degrees` = 0° (drift direction) or 90° (bar
  orientation)?

- **[CONFIRM] What is the angle convention?**
  - 0° = which direction?
  - Does the angle increase clockwise or counter-clockwise?
  - Is this documented anywhere in the Citrus source?

---

## 3. Multi-Arena / Multi-Dish Behavior

### What we know
- Arena config defines sub-arena positions (`sub_arena_x_px`, `sub_arena_y_px`,
  width, height) within the full projector canvas.
- Protocol steps define stimulus parameters globally (one `orientation_degrees`
  per step).

### What we need confirmed

- **[CONFIRM] In multi-dish setups, does each dish/sub-arena receive the same
  grating orientation?** Or can different sub-arenas have independent
  orientations within the same protocol step?

- **[CONFIRM] If a grating covers multiple sub-arenas, is the texture tiled
  independently per sub-arena, or is it one continuous texture across the
  full projector canvas?** This affects whether `orientation_degrees` is
  relative to the sub-arena or the full canvas.

---

## 4. Reactive Grating Modules

### What we know
- Some protocol steps use `reactive_logic_module_name` (e.g.,
  `"OrientationMirrorsXPosition"`) which dynamically adjusts grating
  parameters based on fish position/behavior.
- For these steps, `orientation_degrees` in the protocol is only the
  **initial value** — the actual orientation changes frame-by-frame.
- Currently, **frame-level grating parameters are NOT logged** in the
  stimulus metadata. Only `PARAMS_APPLIED` events are logged when parameters
  change, but these are sparse (not per-frame).

### What we need confirmed

- **[CONFIRM] Which reactive logic modules modify `orientation_degrees`
  dynamically?** Please list all module names that affect grating direction
  (not just speed or spatial frequency).

- **[CONFIRM] For reactive modules, how frequently do parameters change?**
  Every frame? Every N frames? On specific triggers?

- **[REQUEST] Can per-frame grating orientation be logged in the stimulus
  metadata?** For the grating analysis to work with reactive gratings, we
  need a time series of the actual orientation at each frame (or at least
  at each parameter change event). Ideally this would be:
  - An array in the stimulus metadata: `reactive_orientation_degrees[n_frames]`
  - Or at minimum: denser `PARAMS_APPLIED` events with the updated
    `orientation_degrees` value and the corresponding camera frame ID.

- **[REQUEST] Can each `PARAMS_APPLIED` event include a
  `triggering_camera_frame_id` field?** Currently events have
  `relative_timestamp_ns` but mapping to camera frames requires
  interpolation. A direct frame ID would simplify alignment.

---

## 5. Additional Frame-Level Data Requests

These are not blockers for the initial implementation (which will handle
static gratings only), but would enable richer analysis in the future.

- **[REQUEST] Grating phase per frame.** The instantaneous phase of the
  grating at each frame would allow computing the expected optic flow at
  the fish's retinal position. Not critical for heading alignment analysis,
  but useful for future optic flow models.

- **[REQUEST] Actual stimulus render timestamp per frame.** The
  `relative_timestamp_ns` is relative to session start — is this the time
  the stimulus frame was actually displayed (projector VSync), or when the
  render command was issued? The distinction matters for precise latency
  analysis.

---

## 6. Calibration Validation Plan

Once the above questions are answered, we can validate the transform with
a simple empirical check:

1. Run a recording with a known grating (e.g., `orientation_degrees = 0`)
   and a stationary fish (or artificial marker).
2. Measure the apparent drift direction in the camera image.
3. Compare to the heading convention (0° = rightward in camera space).
4. Compute the angular offset and flip state.
5. Repeat for both rigs.

This gives us a **ground-truth offset per rig** that we store as a
calibration constant. Alternatively, if the homography matrix is confirmed
accurate for angles, we decompose it to extract the rotation component
automatically.

### Deliverable
A per-rig constant:
```python
RIG_ANGULAR_CALIBRATION = {
    "omnifin0": {"rotation_offset_deg": 0.0, "flip_x": False, "flip_y": False},
    "omnifin1": {"rotation_offset_deg": 0.0, "flip_x": False, "flip_y": False},
}
```
This gets stored as a parameter in the grating analysis run and applied to
all `orientation_degrees` values before computing alignment with fish heading.

---

## Summary of Blockers

| Item | Type | Blocking? | Notes |
|------|------|-----------|-------|
| Mirror/flip in optical path | CONFIRM | **Yes** | Wrong flip = alignment metric is inverted |
| orientation_degrees convention (drift vs bars, CW vs CCW) | CONFIRM | **Yes** | Wrong convention = 90° systematic error |
| Projector-camera rotation | CONFIRM | **Yes** | Any rotation = systematic angular offset |
| Homography captures full transform? | CONFIRM | Partial | If yes, can skip hardcoded offsets |
| Multi-arena orientation | CONFIRM | No (v1) | Can start with single-arena recordings |
| Reactive module list | CONFIRM | No (v1) | Will exclude reactive steps initially |
| Per-frame reactive orientation logging | REQUEST | No (v1) | Needed for future reactive grating analysis |
| Per-frame grating phase | REQUEST | No | Future enhancement |
| Render timestamp semantics | REQUEST | No | Future latency analysis |
