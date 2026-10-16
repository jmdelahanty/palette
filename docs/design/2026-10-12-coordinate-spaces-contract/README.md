# Shared coordinate spaces: Orange, Citrus, Palette

- **Status:** draft proposal. Jeremy asked for it on 2026-10-12. Palette drafts
  it here; Orange and Citrus answer the questions in §7. Once they agree, it
  moves to agent-contracts as a three-party contract (proposed home:
  `systems/coordinate_spaces_v1/`).
- **Owner:** Jeremy Delahanty. **Last reviewed:** 2026-10-12.
- **Builds on:**
  - the heading convention, agent-contracts `citrus-palette/heading/heading_convention_v1.json` (PR #86, merged `6b63aa2`);
  - `citrus.pose_observation` v2 (same PR);
  - the stimulus angle convention, agent-contracts `citrus-palette/angles/stimulus_angle_convention_v1.json` (PR #91, merged `f3d97d46`): every stimulus angle is `atan2_negative_y_x_degrees`, the retired form is `canvas_y_down_clockwise_legacy`, and directions cross rasters through the homography Jacobian, never by subtracting angles;
  - the 2026-10-12 coordinate audit (agent report, summarised below).
- **Code base:** palette main `43bdff5f`; agent-contracts main `6b63aa2`. Paths are relative to `src/fisheye/` unless they start with `tests/` or `AC:` (agent-contracts).

---

## 1. Problem

We need one chain to hold: **image-left on the camera is the same left on the
projector canvas and on the fish.** Today no shared contract can establish it.

1. **A real rig mirrors.** Citrus measured the active camera→canvas homography of
   omnifin0 arena 1. It carries a y reflection: camera 90° maps to canvas 270°.
   Nothing declares *why*: rear projection, camera mounted below, an optical
   mirror, or a camera flip setting. So nothing can confirm that the reflection
   is intended rather than a mis-wired calibration.
2. **Palette's recorded display orientation is not the orientation.**
   `selected_output_transform_token` is the first word inside the xrandr header's
   parentheses (`shared/selected_calibration.py:196-201`, `:1534-1545`).
   - In non-verbose xrandr output that list holds the *supported* rotations and
     reflections, not the current one. The test fixture shows it:
     `"DP-3 connected 1920x1080+3840+0 (normal left inverted right x axis y axis)"`
     (`tests/unit/fisheye/test_selected_calibration.py:133-143`). So the token is
     always `normal`.
   - A rotated output prints its rotation *before* the parenthesis, so the regex
     fails. That fails closed only by accident, and no test covers it.
   - `--transform` (keystone) matrices appear only in verbose output, which
     Palette never parses.
3. **Homography handedness is never checked.** Palette checks only that the
   matrix is invertible:
   - `abs(det) > 1e-12` in `shared/unified_h5/geometry.py:66-73`;
   - `abs(np.linalg.det(values)) <= 1e-12` in `shared/stimulus_coordinate_contract.py:2526`.

   Citrus folds fixed mount and optical orientation *into* that matrix on
   purpose: `"mount_orientation_rule": "Fixed projector/display/optical orientation is calibration evidence captured by the accepted camera-to-logical-canvas homography, never a stimulus-local axis flip."`
   (`shared/unified_h5/presentation.py:105`). So the matrix is the only carrier
   of physical mirroring, and its sign goes unchecked.
4. **The fish's left assumes an undeclared camera view.** The body frame
   hard-codes `"left_axis": "fixed_clockwise_90_degrees_in_camera_xy"` and
   `"axis_handedness": "determinant_negative_one_camera_xy"`
   (`shared/zarr/body_frame_producer.py:218-219`). That is correct only for a
   dorsal, non-mirrored camera image.
   - The new heading convention now says so: `AC:citrus-palette/heading/heading_convention_v1.json:33-34`, `"camera_view_assumed": "dorsal_unmirrored"`.
   - It is *assumed* per session, not measured. A ventral view or a ReverseX
     camera would swap anatomical left and right on every row without failing
     any check.
5. **The camera raster is tied to its calibration only by size.** Palette
   matches calibration and recording on `native_{width,height}_px` alone
   (`shared/unified_h5/geometry.py:167-173`).
   - Orange's `coordinate_frame` declares `top_left_pixel` at (0,0), right/down
     (`shared/recording_geometry.py:400`), but not *top-left of what*: sensor or
     AOI.
   - A changed AOI offset, flip or binning at the same output size passes every
     check today.

Phototaxis (a left/right half-field stimulus split on the fish's body
midline) depends on items 1, 3 and 4 directly.

## 2. Space registry

One canonical id per space. Every space is 2-D, +x right, +y down, units px unless stated.

**Pixel conventions** apply to every space below:
- `pixel_center`: integer index `i` names the pixel whose centre is at `i`. Its
  area is `[i − 0.5, i + 0.5)`. Used for raster indices and OpenCV-style fits.
- `continuous_pixel_edge`: real coordinate `x`. The left edge of pixel 0 is at
  `x = 0` and its centre at `x = 0.5`; the raster spans `[0, W] × [0, H]`.
  Convert with `x_edge = i_center + 0.5`.
  **This is the precise meaning of Palette's `continuous`**
  (`shared/coordinate_descriptor.py:144-152`, never defined there) and of
  Citrus's `continuous_pixel_edge_coordinates` (`presentation.py:31`).
- `pixel_edge_half_open`: boxes `[x0, x1) × [y0, y1)` in `continuous_pixel_edge`
  coordinates. Palette's bbox surfaces use this
  (`shared/coordinate_surface_contract.py:34-37`).

| Canonical id | Owner | Origin (top-left of …) | Notes |
|---|---|---|---|
| `camera.sensor_px.v1` | Orange | full physical sensor, as read out before AOI, flip or binning | Width/height are the sensor's. Evidence only, never an analysis frame. |
| `camera.recorded_px.v1` | Orange | the **recorded video raster** after AOI, ReverseX/Y, binning/decimation and any rotation | The frame every camera-side product lives in. Its relation to the sensor is a declared transform (§3), not an assumption. |
| `camera.crop_px.v1` | Orange | one crop window inside `camera.recorded_px` | A pure translation by the window origin. Orange clamps the window inside the frame, and both origin ints must be declared as the crop's top-left. |
| `palette.roi_px.v1` | Palette | Palette's requested ROI window inside `camera.recorded_px` | A translation, zero-padded outside the frame (`requested_window_zero_padded_v1`). Not the same rule as Orange's clamping; each declares its own. |
| `palette.model_input_px.v1` | Palette | the detector or pose model input tensor | A scale and translation from `palette.roi_px` or `camera.recorded_px`, declared per run. |
| `citrus.display_canvas_px.v1` | Citrus | the logical final display canvas Citrus renders into (= one xrandr output's `W×H`) | The homography's destination. Extent comes from the session's `stimulus_output_{width,height}`, which must equal xrandr `W×H` (§5). |
| `citrus.framebuffer_px.v1` | Citrus | the OpenGL framebuffer | Declared by Citrus as the canvas reflected once vertically (`presentation.py:132-139`). |
| `citrus.stimulus_video_px.v1` | Citrus | the encoded stimulus video raster | Declared as reflected once (`presentation.py:123-130`). |
| `citrus.arena_canvas_px.v1` | Citrus | the active arena rectangle inside `citrus.display_canvas_px` | A translation by the arena origin. The origin must be integer px (§6c). |
| `physical.camera_mm.v1` | Palette | coincident with `camera.recorded_px` top-left (`coordinate_frame_record.py:142`) | Isotropic scale `1 / pixels_per_mm_camera`, one plane (dish top rim). y stays down. |

There is one pixel convention per space and kind:
- points: `continuous_pixel_edge`;
- boxes: `pixel_edge_half_open`;
- raster indices: `pixel_center`.

A transform may join only spaces whose convention it declares (§3).

### Alias table (today's names → canonical)

| Today's name | Where | Canonical |
|---|---|---|
| `source_camera_px` | Citrus geometry matrices (`shared/unified_h5/contracts/experimental_h5_geometry_v1.json:37,48`) | `camera.recorded_px.v1` |
| `camera_view_px` | calibration attrs (`selected_calibration.py:140`, `unified_h5/geometry.py:152`) | `camera.recorded_px.v1` |
| `camera_native_pixels` | Orange `coordinate_frame`, rim observations (`recording_geometry.py:287,323`) | `camera.recorded_px.v1` (to confirm: Q-O1) |
| `full_frame_pixels` | Orange crop outputs (`contracts/orange_recording_crop_output_v1.schema.json:75-77`) | `camera.recorded_px.v1` |
| `source_frame_pixels` | Orange pose events (`contracts/orange_pose_event_v2.schema.json:212,228`) | `camera.recorded_px.v1` |
| `crop_pixels` | Orange pose events (`:199`) | `camera.crop_px.v1` |
| `source_camera_image_px.top_left_y_down.v1` | Palette descriptor registry (`coordinate_descriptor.py:1248`) | `camera.recorded_px.v1` |
| `roi_local_px.top_left_y_down.v1` | Palette | `palette.roi_px.v1` |
| `detector_model_input_px…` | Palette (`coordinate_descriptor.py:1297`) | `palette.model_input_px.v1` |
| `final_display_canvas_px` | Citrus matrices, calibration attrs | `citrus.display_canvas_px.v1` |
| `stimulus_canvas_px.top_left_y_down.v1` | Palette (`coordinate_descriptor.py:1339`) | `citrus.display_canvas_px.v1` |
| `projector_px` | Palette, marked unavailable | no canonical space; it is the physical projector panel, outside this contract until a photodiode/projector calibration exists |
| `arena_relative_canvas_px` | Citrus catalog, Palette | `citrus.arena_canvas_px.v1` |
| `physical_mm.source_camera_y_down.v1` | Palette | `physical.camera_mm.v1` |

Producers keep their field names. The contract adds a `space_id` next to each
existing name. Palette resolves through this table and nowhere else, which
replaces today's hard-coded string equalities.

## 3. Transform records

Every transform that crosses spaces is a record with these fields:

| Field | Rule |
|---|---|
| `from_space`, `to_space` | Canonical ids from §2. |
| `direction` | The direction the matrix maps (`from → to`). The inverse is a separate record or is derived and checked. |
| `kind` | `translation`, `scale_translation`, `affine` or `homography`. |
| `matrix` | 3×3, `<f8`, finite. |
| `fit_pixel_convention` | The convention of the point pairs the matrix was fitted on (`pixel_center` for OpenCV fits). Palette converts to `continuous_pixel_edge` with the ±0.5 shift, never silently. |
| `from_extent_px`, `to_extent_px` | `[W, H]` of both spaces at fit time. |
| `handedness_declared` | `preserved` or `reflected`. |
| `handedness_reference_point` | A point in `from_space` (default: the centre of the arena region, mapped back to the camera). |
| `orientation_basis` | A list of the declared physical causes (§4) whose parity product gives `handedness_declared`. |

**Handedness check.** For a homography `H`, the Jacobian determinant at a point
`p` is `det(H) / w(p)^3`, with `w(p) = h31·x + h32·y + h33`. The consumer:
1. requires `w(p) > 0` over the whole `from_extent`, checked at the four corners and the reference point;
2. computes `sign(det(H))`;
3. requires `reflected ⇔ sign < 0`.

For affine and translation kinds this reduces to the sign of the linear part's determinant.

## 4. Producer evidence (new recordings)

### Orange, per recording, per camera

These are sealed in the start snapshot; the record is
`camera_geometry_v1` under `camera_runtime.<serial>`.

| Field | Why |
|---|---|
| `sensor_width_px`, `sensor_height_px` | Defines `camera.sensor_px`. |
| `aoi_offset_x_px`, `aoi_offset_y_px`, `aoi_width_px`, `aoi_height_px` | Places `camera.recorded_px` in the sensor. |
| `reverse_x`, `reverse_y` (bool, as applied by the camera) | Each is one reflection. |
| `binning_x`, `binning_y`, `decimation_x`, `decimation_y` | Scale between sensor and recorded raster. |
| `rotation_deg` (0/90/180/270, any software rotation applied before recording) | Rotation is not a reflection, but it changes which way is "up". |
| `viewing_side` (`dorsal` or `ventral`) | Relative to the dish. Declared by rig configuration, not inferred. |
| `optical_mirror_count` (int) | Fold mirrors in the camera path. An odd count is one reflection. |
| `recorded_space_parity` (`+1`/`−1`) | The product of the above, computed by Orange. It is the handedness of `camera.recorded_px` relative to the dish seen from above. |

Orange also emits the `camera.sensor_px → camera.recorded_px` transform record (§3).

### Citrus, per session, per output

| Field | Why |
|---|---|
| `xrandr_verbose_output_block` (raw text) | The real current rotation, reflection and `Transform:` matrix. Palette parses it (§6a). |
| `output_rotation`, `output_reflection` (parsed and declared by Citrus) | Cross-checked against the raw text. |
| `output_transform_matrix` (3×3, from verbose) | Must be identity unless declared. |
| `projection_mode` (`front` or `rear`) | Rear projection is one reflection. |
| `projector_mount` (`table` or `ceiling`) | Ceiling mount is a 180° rotation in projector firmware, not a reflection. Declared because firmware settings vary (Q-C3). |
| `projector_firmware_flip` (`none`, `horizontal`, `vertical`, `both`) | Projector-menu mirroring. |
| `optical_mirror_count` (projection path) | An odd count is one reflection. |
| `canvas_space_parity` (`+1`/`−1`) | The product of the above, with the canvas as it lands on the dish seen from above. |
| `expected_camera_to_canvas_handedness` | `preserved` when `recorded_space_parity × canvas_space_parity = +1`, else `reflected`. |

Citrus also emits the `camera.recorded_px → citrus.display_canvas_px` homography
as a §3 record, with `handedness_declared` equal to the expected value and
`orientation_basis` naming the fields that produced it.

## 5. Palette consumer checks (fail closed)

For every recording admitted under this contract:
1. **Handedness agrees three ways.**
   - The homography's computed sign (§3) must equal `handedness_declared`.
   - `handedness_declared` must equal `expected_camera_to_canvas_handedness`.
   - That must equal `recorded_space_parity × canvas_space_parity`.

   Any disagreement refuses the recording's geometry: stimulus coordinates,
   physical mm and egocentric products.
2. **Body frame and heading.** `viewing_side` and `recorded_space_parity` must
   imply `dorsal_unmirrored`, or Palette refuses the body frame and heading.
   Citrus's `pose_observation v2.camera_view_assumed` must match Orange's
   declaration, or Palette refuses the comparison.
   - A ventral or mirrored rig needs a declared `left_axis` construction of its
     own: a new handedness id in the heading convention, not a silent flip.
3. **Raster identity.** Calibration and recording must match on sensor size,
   AOI, flips, binning and rotation, not only width and height.
4. **Display extent.** `stimulus_output_{width,height}` must equal the xrandr
   `W×H` of the selected output. The unified path checks only the session attrs
   today (`unified_h5/geometry.py:167-173`), while the legacy path uses xrandr
   (`shared/stimulus_frame_transform.py:206-213`).
5. **Missing evidence.** A new recording (after the contract's effective date)
   without the §4 fields is refused at intake.

**Legacy policy.** Recordings made before the contract are admitted with
`orientation_status = "orientation_unverified"` stamped on every geometry-derived
run. Analyses that depend on left/right or on canvas direction must check the
status and refuse unverified rows unless the cohort explicitly opts in.
Proximity and speed do not depend on handedness and are unaffected. For rigs
that still exist, the status is lifted by measurement, not by assertion:
- run a one-off calibration check that shows a known asymmetric pattern and
  locates it in the camera image;
- record that check as evidence, keyed by rig and date range.

## 6. Palette-side fixes that do not wait for the contract

a. **xrandr token.** Stop deriving `selected_output_transform_token` from the
   supported-rotations list.
   - Parse the current rotation and reflection from the header text before the
     parenthesis (`connected [primary] WxH+X+Y [rotation] [X axis|Y axis|X and Y axis] (`),
     and from verbose output when present.
   - Accept `primary`, which the regex rejects today.
   - Add tests for a rotated output and a reflected one.
   - Until this lands, the existing token proves nothing; treat it as such.

b. **Stale agent-contract.** `AC:palette-crimson/detect_bbox_read.md:128,151`
   still requires `pixel_convention == "continuous"` for boxes. Palette
   publishes `pixel_edge_half_open` (`shared/coordinate_surface_contract.py:35`;
   `docs/crimson_detect_bbox_read_contract.md:82`), and
   `AC:palette-crimson/subject_shape_read.md:207` already says half-open.
   Fix the two lines in agent-contracts. This was flagged in
   `docs/diagnostics/coordinate_contract_audit_2026-07-19.md` and is still open.

c. **Arena origin type.**
   - Citrus declares `effective_arena_origin_{x,y}_px` as `<f8`
     (`shared/unified_h5/contracts/experimental_h5_geometry_v1.json:17-18`).
   - Palette's canonical arena frame requires exact ints
     (`shared/stimulus_coordinate_contract.py:759-762`).
   - Decide: Citrus writes integer-valued floats and Palette accepts `float.is_integer()`
     values, refusing others. Or the field becomes integer in the next schema.
     Either way the rule is written down, not left to whichever reader runs.

d. **Two mm definitions.** The GoodCopBadCop workflow computes canvas mm from
   `pixels_per_mm_projector`, with a median fallback from `chaser_states`
   (`docs/goodcopbadcop_coordinate_frame_workflow.md:255-266`). Canonical mm is
   camera-derived. Mark the canvas route as legacy-only in that doc. New
   analyses use `physical.camera_mm.v1`.

## 7. Open questions

### For Orange

- **Q-O1.** Is `camera_native_pixels` (in `coordinate_frame` and rim observations) the recorded video raster after AOI, flips and binning, or the full sensor? Are `full_frame_pixels` and `source_frame_pixels` the same raster?
- **Q-O2.** Which camera settings that change geometry can be set on the rigs today: AOI OffsetX/Y, ReverseX/ReverseY, binning, decimation, any software rotation? Which of them are already in `camera_runtime` or `sensor_pipeline`, under which keys?
- **Q-O3.** Can Orange seal `camera_geometry_v1` (§4) in the start snapshot, including `viewing_side` and `optical_mirror_count` from rig configuration? Who maintains the rig configuration?
- **Q-O4.** Are any current or past rigs viewing from below (ventral), or through a fold mirror?
- **Q-O5.** Orange clamps crop windows inside the frame. Confirm that a crop's `x_px`/`y_px` are its top-left in `camera.recorded_px`, as integer pixel edges.

### For Citrus

- **Q-C1.** What physically explains the y reflection in omnifin0 arena 1's camera→canvas homography: rear projection, projector menu flip, a mirror, or the camera? Is the same true for its other arenas and for the other rigs?
- **Q-C2.** Can Citrus capture `xrandr --verbose` for the selected output each session (current rotation, reflection, `Transform:`), and declare `output_rotation`, `output_reflection` and `output_transform_matrix`?
- **Q-C3.** Can Citrus record `projection_mode`, `projector_mount`, `projector_firmware_flip` and the projection-path `optical_mirror_count` per rig? Where would they live: rig config, session metadata, or the geometry receipt?
- **Q-C4.** What pixel convention were the camera→canvas calibrations fitted in? Were the point pairs integer pixel centres from OpenCV, or edge coordinates?
- **Q-C5.** Can the homography record carry `handedness_declared` and `orientation_basis` (§3)? Can Citrus refuse its own calibration when the matrix sign disagrees?
- **Q-C6.** For the arena origin: are integer-valued floats guaranteed today, or can a fractional origin occur?
- **Q-C7.** Is `pose_observation v2.camera_view_assumed` set from rig configuration, or constant? Can it come from Orange's `viewing_side` once that exists?

## 8. Rollout

1. **Now, Palette only.**
   - The §6a xrandr parser fix.
   - The §6b agent-contracts two-line fix.
   - Stamp `orientation_status = "orientation_unverified"` on geometry runs from all current recordings, so nothing claims verified orientation.
2. **Answers to §7,** then the agent-contracts PR `systems/coordinate_spaces_v1/`, containing:
   - the registry and alias table;
   - the transform-record schema;
   - the producer evidence schemas;
   - fixtures with one preserved rig and one reflected rig.
3. **Producers.** Orange seals `camera_geometry_v1`. Citrus captures verbose xrandr, projection evidence and handedness-declared homographies. Each producer ships behind a schema version, so Palette can tell the old form from the new.
4. **Palette.**
   - Pin the contract.
   - Run the §5 checks at intake.
   - Resolve all space names through the alias table.
   - Replace `|det|`-only checks with the sign check.
5. **Effective date.** From the first producer release that carries the evidence, missing evidence is refused (§5 item 5).

### What blocks phototaxis

A left/right split on the fish's midline needs every link of the chain to be verified, not assumed:
- the camera parity (Q-O2, Q-O4);
- the projector and canvas parity (Q-C1–Q-C3);
- the homography sign check (§5 item 1);
- `camera_view_assumed` backed by declared `viewing_side` (§5 item 2).

Until those land, phototaxis sessions are acquired under `orientation_unverified`. Each needs the one-off asymmetric-pattern check on its rig before its left/right results are trusted. **The minimum to start phototaxis safely:** steps 1 and 2, plus Citrus's Q-C1 answer and that check run on the phototaxis rig.
