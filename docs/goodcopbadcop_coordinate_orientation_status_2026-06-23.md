# GoodCopBadCop Coordinate and Orientation Status
<!-- status-meta
status: audit
last_updated: 2026-06-23
-->

Purpose: summarize whether the current GoodCopBadCop/CRA analyses and marimo
visualizations use consistent coordinate systems, axis directions, and polar
angle conventions.

This is a status document, not a new contract. The active contract remains
`docs/goodcopbadcop_coordinate_frame_workflow.md`.

## Bottom Line

No active sign-flip or axis-swap bug was found in the current
`chaser_distance_runs` -> CRA/near-field/egocentric-bearing ->
`palette_explorer` path.

Some older or adjacent chaser tools do not fully follow the newer contract and
should be treated as legacy/diagnostic until patched. In particular,
`compute_chaser_fish_metrics.py` mixes image-space displacement with math-style
heading without the y-flip used by the newer egocentric component.

The current workflow is internally consistent if the reader keeps one distinction
clear:

- stored position arrays are image-like: `x` right, `y` down;
- angle calculations convert image vectors to math coordinates by negating `dy`;
- arena-position plots reverse the display y-axis so the visual still looks like
  the image/stimulus arena;
- polar plots use their own explicitly labeled angular conventions.

The main remaining risks are legacy/adjacent modules and homography naming debt,
not observed calculation errors in the current GoodCopBadCop path. As of the
follow-up patch from this audit, chaser-distance outputs write explicit axis
direction attrs, CRA primary endpoint rejects non-canonical chaser-distance
frames, and escape/freeze trajectory metadata distinguishes chaser-centric from
fish-centered columns.

## Canonical Position Frame

Current GoodCopBadCop chaser metrics are computed in
`arena_relative_canvas_px`, not raw camera pixels.

Workflow:

```text
fish detection centroid: source_image_px
  -> stored homography
  -> canvas/projector px
  -> subtract active arena origin
  -> arena_relative_canvas_px

chaser position: stimulus log / chaser_states
  -> arena_relative_canvas_px

distance_px = norm(chaser_arena_xy - fish_arena_xy)
distance_mm = distance_px / pixels_per_mm_projector
```

Evidence:

- `docs/goodcopbadcop_coordinate_frame_workflow.md:18` documents the shared
  arena-relative canvas frame.
- `docs/goodcopbadcop_coordinate_frame_workflow.md:104` defines the origin as
  the active arena top-left.
- `src/fisheye/analysis/chaser_distance_runs.py:532` rejects chaser states that
  are not `arena_relative_canvas_px`.
- `src/fisheye/analysis/chaser_distance_runs.py:548` maps fish detections into
  the same frame before distance calculation.
- `src/fisheye/analysis/chaser_distance_runs.py:557` computes
  `chaser_xy - fish_arena`.
- `src/fisheye/analysis/chaser_distance_runs.py:559` converts distance pixels to
  mm with `pixels_per_mm_projector`.
- `src/fisheye/analysis/chaser_distance_runs.py:870` writes position
  coordinate-frame attrs.
- `src/fisheye/analysis/chaser_distance_runs.py:890` writes the distance-frame
  and distance-conversion attrs.

Important wording: `fish_centroid_arena_xy` and `chaser_arena_xy` are arena-local
canvas pixels. They are not stored as absolute millimetre coordinates. Analyses
convert deltas/distances to mm using `pixels_per_mm_projector`.

## Homography Direction Debt

There is known naming debt around `projector_to_camera_px`.

The shared helper docstring still describes projector-to-camera use:

- `src/fisheye/shared/coordinate_transform.py:1`
- `src/fisheye/shared/coordinate_transform.py:172`

The active GoodCopBadCop path intentionally treats the stored calibration matrix
as camera/source-image to canvas/projector for current Citrus snapshots:

- `src/fisheye/analysis/chaser_distance_runs.py:269`
- `docs/goodcopbadcop_coordinate_frame_workflow.md:196`

Status: consistent in the active writer, but the helper name and docstring remain
misleading. Consumers should trust the persisted output attrs and writer
provenance, not infer direction from the helper name.

## Global Angle Rule

Palette's global heading rule is:

```text
heading_deg = atan2(-dy, dx)
```

This means:

- `0 deg` points image +x/right;
- `90 deg` points up in the image;
- `-90 deg` points down in the image;
- positive rotation is counter-clockwise in math coordinates.

Evidence:

- `docs/analytics_math_primer.md:58`
- `docs/body_frame_contract.md:63`
- `docs/keypoint_heading_computation_contract.md:209`
- `docs/goodcopbadcop_egocentric_bearing_implementation_checklist.md:46`

This rule explains why angle code negates `y` while image-layout plots reverse
the display axis instead.

## Egocentric Chaser Bearing

Egocentric chaser bearing is computed from the chaser-distance position arrays
and offline track-kinematics heading.

Formula:

```text
object_vector_arena_xy = chaser_arena_xy - fish_centroid_arena_xy
object_bearing_world_deg = atan2(-object_vector_y, object_vector_x)
egocentric_bearing_deg = wrap(object_bearing_world_deg - fish_heading_deg)
```

Interpretation:

- `0 deg`: chaser directly in front of fish;
- `+90 deg`: chaser on fish anatomical left;
- `-90 deg`: chaser on fish anatomical right;
- `+/-180 deg`: chaser behind fish.

Evidence:

- `docs/goodcopbadcop_egocentric_bearing_implementation_checklist.md:38`
- `src/fisheye/analysis/chaser_egocentric_bearing.py:51` records the angle
  convention string.
- `src/fisheye/analysis/chaser_egocentric_bearing.py:143` documents arena
  positions as y-down and headings as math-y-up.
- `src/fisheye/analysis/chaser_egocentric_bearing.py:204` computes the object
  vector.
- `src/fisheye/analysis/chaser_egocentric_bearing.py:208` converts y-down
  vectors to math-y-up bearings.
- `src/fisheye/analysis/chaser_egocentric_bearing.py:209` subtracts fish heading.
- `tests/unit/fisheye/test_chaser_egocentric_bearing.py:59` tests right/up/left/down
  behavior.

Display:

- Static matplotlib polar plots put `0/front` at north and positive angles
  counter-clockwise: `src/fisheye/analysis/chaser_egocentric_bearing.py:715`.
- Marimo Plotly polar plots use `rotation=90`, `direction="counterclockwise"`,
  and labels `behind/right/front/left/behind`:
  `apps/marimo/components/goodcopbadcop_chaser.py:1297` and
  `apps/marimo/components/goodcopbadcop_chaser.py:1465`.

Status: analysis and display conventions match.

## Chaser-Centric Escape/Freeze Frame

The escape/freeze canary uses a chaser-centric frame for active pursuit trials.

Source arena positions are y-down. The transform converts to y-up metric deltas
and rotates so the chaser's travel heading points along `+y`.

Evidence:

- `docs/goodcopbadcop_escape_freeze_canary_checklist.md:208` describes the
  chaser-centric transform.
- `src/fisheye/analysis/chaser_escape_freeze.py:418` computes chaser motion in
  math-y-up coordinates by negating image `dy`.
- `src/fisheye/analysis/chaser_escape_freeze.py:452` documents the transform.
- `src/fisheye/analysis/chaser_escape_freeze.py:469` computes fish-minus-chaser
  deltas in mm with y flipped.
- `src/fisheye/analysis/chaser_escape_freeze.py:470` rotates chaser heading to
  `+y`.
- `src/fisheye/analysis/chaser_escape_freeze.py:476` stores chaser-frame bearing
  as `atan2(x_prime, y_prime)`.
- `tests/unit/fisheye/test_chaser_escape_freeze.py:16` tests that chaser heading
  maps to positive y.

Important distinction: this bearing is not the fish egocentric bearing. In the
chaser frame, `0 deg` is chaser-forward and positive angles follow positive
`x_prime`. With the current transform, positive `x_prime` is right relative to
the chaser's travel direction. That differs from egocentric bearing, where
positive means fish anatomical left.

Status: code is internally consistent, but the docs should explicitly define
`escape_bearing_deg` zero and sign.

## Fish-Centered Escape/Freeze Diagnostic Frame

The fish-centered escape/freeze diagnostic is a world/math view centered on the
fish. It is not rotated by fish heading.

Formula:

```text
chaser_x_fish_centered_mm = (chaser_x - fish_x) / pixels_per_mm_projector
chaser_y_fish_centered_mm = -(chaser_y - fish_y) / pixels_per_mm_projector
```

Interpretation:

- fish is at the origin;
- `+x` is right;
- `+y` is up;
- `0 deg` is right;
- `90 deg` is up.

Evidence:

- `docs/goodcopbadcop_escape_freeze_canary_checklist.md:212`
- `src/fisheye/analysis/chaser_escape_freeze.py:910`
- `src/fisheye/analysis/chaser_escape_freeze.py:911`
- `src/fisheye/analysis/chaser_escape_freeze.py:1208`
- `src/fisheye/analysis/chaser_escape_freeze.py:1397`
- `src/fisheye/analysis/chaser_escape_freeze.py:1424`

Display:

- The per-trial fish-centered diagnostic uses ordinary Cartesian axes with
  `+y` visually upward: `src/fisheye/analysis/chaser_escape_freeze.py:1191`.
- The polar approach scatter and density set zero to east/right and positive
  direction counter-clockwise:
  `src/fisheye/analysis/chaser_escape_freeze.py:1424`.

Status: analysis and display conventions match. The key display distinction is
that this polar plot is fish-centered/world-oriented, while egocentric polar
plots are fish-heading-oriented.

## CRA Quadrants and Arena Layout

CRA primary endpoint quadrant logic uses arena-local top-left origin with y
increasing down.

Quadrant codes:

```text
0 = top_left
1 = top_right
2 = bottom_left
3 = bottom_right
```

Midline ownership:

- `x >= width / 2` belongs to right;
- `y >= height / 2` belongs to bottom.

Evidence:

- `src/fisheye/analysis/cra_primary_endpoint.py:339`
- `src/fisheye/analysis/cra_primary_endpoint.py:344`
- `tests/unit/fisheye/test_cra_primary_endpoint.py:209`

Object-relative occupancy uses each object's current phase location rather than
a fixed physical quadrant:

- `src/fisheye/analysis/cra_primary_endpoint.py:496` summarizes phase-specific
  chaser/object position.
- `src/fisheye/analysis/cra_primary_endpoint.py:514` assigns each object to a
  phase-specific quadrant.
- `src/fisheye/analysis/cra_primary_endpoint.py:529` computes occupancy in that
  object's phase-specific quadrant.

Display:

- Static CRA overview reverses y with `ax.set_ylim(height, 0)`:
  `src/fisheye/analysis/cra_primary_endpoint.py:841`.

Status: computation and display are consistent.

## Arena Heatmaps and Occupancy Displays

Arena-position heatmaps intentionally preserve image/stimulus layout by reversing
the y-axis in display.

Evidence:

- Current marimo component:
  `apps/marimo/components/goodcopbadcop_chaser.py:1696` builds the heatmap and
  `apps/marimo/components/goodcopbadcop_chaser.py:1749` reverses y.
- The older standalone notebook does the same:
  `apps/marimo/goodcopbadcop_explorer.py:315` and
  `apps/marimo/goodcopbadcop_explorer.py:351`.
- Tests assert this behavior:
  `tests/unit/fisheye/test_marimo_palette_explorer_components.py:643`.

Status: display-side y reversal is intentional and should not be copied into
analysis arrays.

## Legacy and Adjacent Tools

The active GoodCopBadCop marimo component delegates orientation-sensitive plots
to component renderers. The older standalone app also reverses y for arena and
occupancy displays. However, several legacy/adjacent tools have weaker
orientation guarantees.

### `compute_chaser_fish_metrics.py`

This module predates the current chaser-distance/CRA stack. It uses crop-derived
source-image fish centroids and chaser positions rather than
`arena_relative_canvas_px`.

It also builds the heading vector as:

```text
[cos(heading), sin(heading)]
```

and computes signed angle against the image-space displacement vector without
negating `dy`.

Evidence:

- `src/fisheye/analysis/compute_chaser_fish_metrics.py:168` builds the heading
  vector.
- `src/fisheye/analysis/compute_chaser_fish_metrics.py:377` computes
  `chaser_point - fish_point` in image coordinates.
- `src/fisheye/analysis/compute_chaser_fish_metrics.py:391` computes the signed
  angle from those vectors.

Status: likely incompatible with the current math-y-up heading convention for
signed angle orientation. Do not use it as the authority for GoodCopBadCop
egocentric analyses.

### Legacy Static Visualizers

`visualize_chaser_vs_fish.py` plots image/source pixel positions without
reversing y, so the scatter display is likely vertically inverted relative to
the current arena-display convention:

- `src/fisheye/visualization/visualize_chaser_vs_fish.py:56`
- `src/fisheye/visualization/visualize_chaser_vs_fish.py:86`

`chaser_phase_analysis.py` uses `origin="lower"` for heatmap display, which is
also likely flipped relative to y-down image-layout plots:

- `src/fisheye/analysis/chaser_phase_analysis.py:1027`

Status: these are legacy visualization risks, not active `palette_explorer`
risks.

### Detection Occupancy Metadata

Detection occupancy spatial zone sets do record their coordinate frame and axis
directions:

- `src/fisheye/analysis/detection_occupancy_runs.py:343`
- `src/fisheye/analysis/detection_occupancy_runs.py:550`

The heatmap arrays themselves record axis order and bin edges, but do not repeat
`coordinate_origin` or y-axis direction at the heatmap group:

- `src/fisheye/analysis/detection_occupancy_runs.py:700`

The current renderers display them correctly with `origin="upper"`/reversed y:

- `src/fisheye/analysis/detection_occupancy_runs.py:617`
- `apps/marimo/components/goodcopbadcop_chaser.py:1749`

Status: current display is consistent. Future generic consumers should read the
spatial occupancy zone metadata or the heatmap group should repeat origin/axis
attrs to avoid upside-down rendering.

## Current Risks and Cleanup Items

1. Keep explicit axis direction attrs on chaser-distance runs.

   Current outputs now record:

   ```text
   x_axis_direction = "right"
   y_axis_direction = "down"
   ```

2. Keep escape/freeze trajectory metadata split by coordinate family.

   `trial_trajectories` contains both chaser-centric columns and fish-centered
   columns. The writer now adds per-family attrs for:

   ```text
   chaser_centric_mm
   fish_centered_world_mm
   ```

   along with a structured `column_coordinate_frames` attr.

3. Keep escape/freeze doc wording strict.

   The selected chaser-distance run stores registered arena-local pixels and mm
   distances. Escape/freeze should use registered arena-local canvas pixel
   positions and convert position deltas to mm with `pixels_per_mm_projector`.

4. Keep CRA primary endpoint rejection of non-canonical coordinate frames.

   CRA primary endpoint now matches CRA near-field: it fails if its source
   chaser-distance run is not `arena_relative_canvas_px` with
   `top_left_of_active_arena`.

5. Clarify circular arena geometry frame.

   Near-field and epoch behavior summaries read circular arena center/radius
   geometry. Those geometry attrs should be explicitly declared to be in the same
   active-arena top-left frame as `fish_centroid_arena_xy` and `chaser_arena_xy`;
   otherwise center-distance and wall-band metrics could shift if future metadata
   stores full-canvas/global centers.

6. Explicitly distinguish polar plot families in UI captions.

   Egocentric polar plots:

   ```text
   0 deg = fish front
   +90 deg = fish anatomical left
   ```

   Fish-centered approach polar plots:

   ```text
   0 deg = image/world right
   90 deg = image/world up
   no fish-heading rotation
   ```

   This is already mostly captioned, but it is important enough to keep visible.

7. Clarify `object_phase_x_mm` and `object_phase_y_mm`.

   CRA primary endpoint computes object x/y in mm by dividing arena-local canvas
   pixels by `pixels_per_mm_projector`. These are arena-local top-left-origin
   canvas-mm coordinates, not a separate physical coordinate registration
   surface. The docs should say this explicitly.

8. Keep homography direction naming debt visible.

   The active GoodCopBadCop writer is explicit and consistent, but the shared
   helper name/docstring is still historically inverted for this use. Future
   cleanup should split the generic homogeneous transform helper from directional
   projector/camera helpers.

9. Multi-zone future risk.

   CRA primary endpoint currently resolves quadrant bounds from
   `stimulus_coordinates/arena_*`, with `arena_1` as the preferred path. This is
   acceptable for the current inspected GoodCopBadCop data, but future Citrus
   topology should tie active arena/zone identity to `zone_id`, `stream_id`, and a
   `homography_ref`.

10. Patch or retire legacy chaser metric/display tools.

    `compute_chaser_fish_metrics.py`, `visualize_chaser_vs_fish.py`, and
    `chaser_phase_analysis.py` should either be updated to the current
    coordinate/angle contract or clearly labeled legacy so they are not confused
    with the current GoodCopBadCop outputs.

## Status Table

| Surface | Coordinate/angle convention | Status |
| --- | --- | --- |
| Chaser-distance positions | `arena_relative_canvas_px`, top-left origin, x right, y down | Consistent; axis attrs written |
| Chaser-distance distances | `distance_mm = distance_px / pixels_per_mm_projector` | Consistent |
| CRA quadrants | top-left/top-right/bottom-left/bottom-right in y-down arena layout | Consistent |
| CRA static plots | y reversed for image-layout display | Consistent |
| Near-field distances | consumes authoritative `distance_mm` | Consistent |
| Egocentric bearing | y flipped for angle math; 0 front; positive fish-left | Consistent |
| Egocentric polar display | 0/front at top; positive counter-clockwise | Consistent |
| Escape/freeze chaser frame | chaser heading to +y; metric deltas; positive bearing follows +x | Consistent; document sign |
| Escape/freeze fish-centered frame | fish at origin; x right; y up; 0 right; 90 up | Consistent |
| Arena heatmaps | y-down data displayed with reversed y-axis | Consistent |
| Homography helper naming | helper name says projector-to-camera, active use is generic/camera-to-canvas | Known naming debt |
| Legacy chaser-fish metrics | source-image px and signed angles without current y-flip | Inconsistent risk |
| Legacy static chaser plots | some use y-up display for y-down data | Inconsistent risk |

## Recommended Next Steps

1. Add migration/backfill logic if old chaser-distance runs without explicit
   axis attrs need to be made self-describing.
2. Add focused tests or deprecation notes for remaining legacy static display
   tools if they are still used operationally.
3. Later, split `projector_to_camera_px` into a direction-neutral homogeneous
   transform helper plus explicitly directional wrappers.
