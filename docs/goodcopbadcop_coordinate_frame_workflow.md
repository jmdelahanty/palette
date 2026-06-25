# GoodCopBadCop Coordinate-Frame Workflow
<!-- contract-meta
status: draft
last_updated: 2026-06-22
-->

Purpose: document the current, active coordinate-frame contract for
GoodCopBadCop chaser analyses. This is a reference for deciding whether an
analysis is using raw camera pixels, registered camera-derived positions,
projector/stimulus coordinates, or arena-relative coordinates.

This document describes the workflow implemented by
`src/fisheye/analysis/chaser_distance_runs.py` and consumed by the CRA primary
endpoint and egocentric-bearing components.

## Short Version

GoodCopBadCop chaser metrics are not computed in raw camera space. They are
computed in a shared arena-relative canvas frame:

```text
fish detection centroid: source_image_px
  -> stored homography
  -> canvas/projector px
  -> subtract active arena origin
  -> arena_relative_canvas_px

chaser position: stimulus log / chaser_states
  -> already arena_relative_canvas_px

distance_px = norm(chaser_arena_xy - fish_arena_xy)
distance_mm = distance_px / pixels_per_mm_projector
```

The camera still provides the fish position. The stimulus controller provides
the chaser position. The homography is the bridge that puts those two sources
into one coordinate frame before distance, occupancy, or egocentric-bearing
metrics are computed.

## Source Authorities

The active workflow has three source authorities:

- `refined_detect_runs/<run>/instances`
  - source for fish detection centers;
  - native frame is `source_image_px`;
  - row identity is camera frame.
- `analysis/stimulus_runs/<run>/tracking_data/chaser_states`
  - source for chaser positions;
  - current GoodCopBadCop external-IPC imports expose chaser positions in
    `arena_relative_canvas_px`;
  - row identity is stimulus/logged state, aligned onto camera frames by the
    chaser-distance writer.
- `analysis/calibration`
  - source for the stored homography and `pixels_per_mm_projector`;
  - supplies the transform used to map fish camera detections into the chaser
    coordinate frame.

Epoch definitions are owned separately by:

```text
analysis/stimulus_epoch_runs/<run>/windows
```

The chaser-distance writer copies window IDs and labels into summaries for
convenience, but the stimulus epoch run remains the authority for window
boundaries.

## Coordinate Frames

### `source_image_px`

This is the raw camera or detection-source image coordinate frame.

Current use:

```text
analysis/chaser_distance_runs/<run>/positions/fish_centroid_img_xy
```

Attrs written by the chaser-distance writer:

```text
fish_centroid_img_xy_coordinate_frame = "source_image_px"
```

This frame is useful for detection QC and video overlays. It is not the
canonical frame for chaser metrics because chaser positions are logged by the
stimulus controller, not measured from the camera image.

### Canvas / Projector Pixels

The stimulus controller represents chasers in a canvas/projector-like pixel
frame. Current imported GoodCopBadCop chaser states are already arena-relative,
not global raw camera pixels.

The current chaser state contract is:

```text
coordinate_frame = "arena_relative_canvas_px"
coordinate_origin = "top_left_of_active_arena"
```

### `arena_relative_canvas_px`

This is the shared analysis frame for current chaser metrics. It is a
per-arena coordinate frame whose origin is the active arena's top-left corner
in the stimulus canvas.

Current use:

```text
analysis/chaser_distance_runs/<run>/positions/fish_centroid_arena_xy
analysis/chaser_distance_runs/<run>/positions/chaser_arena_xy
analysis/chaser_distance_runs/<run>/distances/distance_px
```

Attrs written by the chaser-distance writer:

```text
fish_centroid_arena_xy_coordinate_frame = "arena_relative_canvas_px"
chaser_arena_xy_coordinate_frame = "arena_relative_canvas_px"
chaser_arena_xy_coordinate_origin = "top_left_of_active_arena"
distance_px_coordinate_frame = "arena_relative_canvas_px"
```

This frame is currently the canonical frame for:

- fish-to-chaser distance;
- CRA distance-to-object summaries;
- CRA object-relative quadrant occupancy;
- egocentric chaser bearing;
- per-recording and merged group exports built from those components.

## Active Transform Workflow

The current implementation path is:

1. Load fish detections:

   ```text
   frames, centers_img = _read_detection_centers(...)
   fish_img, fish_valid = _dense_fish_positions(...)
   ```

2. Load chaser state and frame alignment:

   ```text
   camera_frame_id, stimulus_frame_num, timestamp_ns, stim_to_camera =
       _load_frame_alignment(...)

   chaser_indices, chaser_xy, chaser_valid, coordinate_frame, coordinate_origin =
       _dense_chaser_positions_from_group(...)
   ```

3. Require chaser coordinates to already be arena-relative:

   ```text
   coordinate_frame == "arena_relative_canvas_px"
   ```

4. Load calibration:

   ```text
   calibration = load_calibration_transform(root, stimulus_run=stim_name)
   homography = calibration["homography"]
   pixels_per_mm_projector = _resolve_pixels_per_mm_projector(...)
   arena_origin = _resolve_arena_origin(stim_group)
   ```

5. Transform fish detection centers into arena-relative canvas pixels:

   ```text
   fish_arena = _camera_to_arena_xy(
       fish_img,
       homography_camera_to_canvas=homography,
       arena_origin_in_canvas_xy=arena_origin,
   )
   ```

6. Compute distances:

   ```text
   delta = chaser_xy - fish_arena
   distance_px = norm(delta)
   distance_mm = distance_px / pixels_per_mm_projector
   ```

The resulting arrays are written under:

```text
analysis/chaser_distance_runs/<run>/positions/
analysis/chaser_distance_runs/<run>/distances/
```

## Homography Direction Caveat

The historical helper in `fisheye.shared.coordinate_transform` is named
`projector_to_camera_px`, and that module's docstring describes a
projector-to-camera homography. The active GoodCopBadCop chaser-distance code
uses the helper as a generic homogeneous point transformer.

The implementation comment in `chaser_distance_runs.py` is the current source
of truth for this workflow:

```text
Current Citrus calibration snapshots store the matrix that maps camera pixels
into projector/canvas pixels. The helper applies an arbitrary 3x3 homogeneous
transform despite its historical projector->camera name.
```

Therefore, for current GoodCopBadCop chaser-distance runs:

```text
analysis/calibration/homography_matrix
```

is treated as the transform from fish detection image coordinates into
canvas/projector coordinates. The writer then subtracts
`arena_origin_in_canvas_xy` to produce `arena_relative_canvas_px`.

Do not infer transform direction from the helper name alone. Consumers should
trust the stored output attrs on the analysis run and the writer provenance.

## Why Not Raw Camera Space?

Raw camera space is not wrong; it is just not the canonical metric frame for
the current analysis.

Chaser positions are authored and logged by the stimulus controller. The
camera does not directly measure the dot/chaser position. Computing distances
in raw camera pixels would require mapping every logged chaser position into
camera pixels first. That is possible for QC overlays, but it would make the
canonical metric camera-specific and less directly tied to the stimulus
controller's object position.

The current policy is:

```text
analysis metrics: registered shared arena frame
video/debug overlays: camera frame as needed
```

This keeps the confirmatory and exploratory metrics object-relative while still
preserving camera-space positions for QC.

## Distance Units

`distance_px` is an arena-relative canvas pixel distance, not a raw camera
pixel distance.

`distance_mm` is computed as:

```text
distance_mm = distance_px / pixels_per_mm_projector
```

The conversion factor is resolved from:

1. `analysis/calibration.attrs["pixels_per_mm_projector"]`, if present and
   positive;
2. otherwise the median positive `pixels_per_mm` value in
   `tracking_data/chaser_states`, if available.

Because the current metric frame is canvas/projector-like, the writer uses
`pixels_per_mm_projector`, not camera `pixel_to_mm`.

## Downstream Components

### CRA Primary Endpoint

The CRA primary endpoint consumes:

```text
analysis/chaser_distance_runs/<run>/positions/fish_centroid_arena_xy
analysis/chaser_distance_runs/<run>/positions/chaser_arena_xy
analysis/chaser_distance_runs/<run>/distances/distance_mm
```

Its object-relative distance and occupancy summaries inherit the
`arena_relative_canvas_px` frame from the chaser-distance run.

The key analysis rule is that pre/post occupancy is computed relative to the
object's current location in each phase, not relative to a fixed raw camera
quadrant.

### Egocentric Chaser Bearing

Egocentric bearing consumes:

```text
fish_centroid_arena_xy
chaser_arena_xy
track_kinematics heading
```

The object vector is:

```text
object_vector_arena_xy = chaser_arena_xy - fish_centroid_arena_xy
```

Angles are then converted from image-style y-down positions into math-y-up
bearings before subtracting fish heading:

```text
egocentric_bearing_deg = wrap(object_bearing_world_deg - fish_heading_deg)
```

Current convention:

```text
0 deg = chaser in front of fish
positive = anatomical left
```

### Group Exports

Merged GoodCopBadCop exports do not recompute coordinate transforms. They
export rows derived from the per-recording zarr arrays and preserve provenance
back to the source run/component.

Therefore, any coordinate-frame correction must happen at the per-recording
analysis-run level before group export.

## QC Expectations

For each chaser-distance run, QC should be able to answer:

- Does `fish_centroid_arena_xy` overlay sensibly against transformed video
  landmarks or arena bounds?
- Do `chaser_arena_xy` positions match expected pre/post chaser locations?
- Is `distance_px_coordinate_frame` explicitly
  `arena_relative_canvas_px`?
- Is `distance_mm_conversion` explicitly
  `distance_px / pixels_per_mm_projector`?
- Does the run record the source detection run, stimulus run, epoch run,
  calibration-derived `pixels_per_mm_projector`, and arena origin?

Camera-space overlays remain useful for registration QC, but the canonical
distance and bearing arrays should be checked in the registered arena frame.

## Known Naming Debt

The active code and some historical helper names do not use the same language:

- `projector_to_camera_px` currently applies an arbitrary 3x3 homogeneous
  transform and is used by GoodCopBadCop as camera/source-image to
  canvas/projector.
- `homography_camera_to_canvas` is the local parameter name in
  `chaser_distance_runs.py` and better reflects the active use.
- `fish_centroid_arena_xy` means arena-relative canvas pixels, not physical
  millimetres.

Future cleanup should separate transform helpers by explicit direction, but
until that happens, the persisted array attrs and run provenance are the
contract readers should trust.
