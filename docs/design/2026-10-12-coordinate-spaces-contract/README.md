# Shared coordinate spaces: Orange, Citrus, Palette

- **Status:** draft proposal. Jeremy asked for it on 2026-10-12. Palette drafts
  it here. Orange and Citrus answered the questions in §7 (2026-10-12/13,
  updated 2026-11-14). Jeremy decided the four open items on 2026-11-15
  (§10). Once the producers agree, it
  moves to agent-contracts as a three-party contract (proposed home:
  `systems/coordinate_spaces_v1/`).
- **Owner:** Jeremy Delahanty. **Last reviewed:** 2026-11-14 (citations
  re-verified against palette main `b04ec0d4` and agent-contracts main
  `de21c037`; see §9).
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
   - `abs(det) > 1e-12` in `shared/unified_h5/geometry.py:79-82`;
   - `abs(np.linalg.det(values)) <= 1e-12` in `shared/stimulus_coordinate_contract.py:2526`.

   Citrus folds fixed mount and optical orientation *into* that matrix on
   purpose: `"mount_orientation_rule": "Fixed projector/display/optical orientation is calibration evidence captured by the accepted camera-to-logical-canvas homography, never a stimulus-local axis flip."`
   (`shared/unified_h5/presentation.py:105`). So the matrix is the only carrier
   of physical mirroring, and its sign goes unchecked.
4. **The fish's left assumes an undeclared camera view.** The body frame
   hard-codes `"left_axis": "fixed_clockwise_90_degrees_in_camera_xy"` and
   `"axis_handedness": "determinant_negative_one_camera_xy"`
   (`shared/zarr/body_frame_producer.py:219-220`). That is correct only for a
   dorsal, non-mirrored camera image.
   - The new heading convention now says so: `AC:citrus-palette/heading/heading_convention_v1.json:33-34`, `"camera_view_assumed": "dorsal_unmirrored"`.
   - It is *assumed* per session, not measured. A ventral view or a ReverseX
     camera would swap anatomical left and right on every row without failing
     any check.
5. **The camera raster is tied to its calibration only by size.** Palette
   matches calibration and recording on `native_{width,height}_px` alone
   (`shared/unified_h5/geometry.py:175-181`).
   - Orange's `coordinate_frame` declares `top_left_pixel` at (0,0), right/down
     (`shared/recording_geometry.py:446-451`), but not *top-left of what*: sensor or
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
| `citrus.display_canvas_px.v1` | Citrus | the logical final display canvas Citrus renders into (= one xrandr output's `W×H`), **y-reflected relative to scan-out** (measured 2026-11-15, §10) | The homography's destination. Extent comes from the session's `stimulus_output_{width,height}`, which must equal xrandr `W×H` (§5). Canvas `(x, y)` is shown at scan-out `(x, H − 1 − y)` (pixel-centre indices), so canvas `(0, 0)` is the scan-out's **bottom-left**. |
| `citrus.framebuffer_px.v1` | Citrus | the OpenGL framebuffer | Declared by Citrus as the canvas reflected once vertically (`presentation.py:132-140`). |
| `citrus.stimulus_video_px.v1` | Citrus | the encoded stimulus video raster | Declared as reflected once (`presentation.py:123-131`). |
| `citrus.arena_canvas_px.v1` | Citrus | the active arena rectangle inside `citrus.display_canvas_px` | A translation by the arena origin. The origin must be integer px (§6c). |
| `physical.camera_mm.v1` | Palette | coincident with `camera.recorded_px` top-left (`coordinate_frame_record.py:143`) | Isotropic scale `1 / pixels_per_mm_camera`, one plane (dish top rim). y stays down. |

There is one pixel convention per space and kind:
- points: `continuous_pixel_edge`;
- boxes: `pixel_edge_half_open`;
- raster indices: `pixel_center`.

A transform may join only spaces whose convention it declares (§3).

### Alias table (today's names → canonical)

| Today's name | Where | Canonical |
|---|---|---|
| `source_camera_px` | Citrus geometry matrices (`shared/unified_h5/contracts/experimental_h5_geometry_v1.json:37,48`) | `camera.recorded_px.v1` |
| `camera_view_px` | calibration attrs (`selected_calibration.py:140`, `unified_h5/geometry.py:160`) | `camera.recorded_px.v1` |
| `camera_native_pixels` | Orange `coordinate_frame`, rim observations (`recording_geometry.py:446,374`) | `camera.recorded_px.v1` (to confirm: Q-O1) |
| `full_frame_pixels` | Orange crop outputs (`contracts/orange_recording_crop_output_v1.schema.json:75-77`) | `camera.recorded_px.v1` |
| `source_frame_pixels` | Orange pose events (`contracts/orange_pose_event_v2.schema.json:212,228`; same lines in `orange_pose_event_v3.schema.json`) | `camera.recorded_px.v1` |
| `crop_pixels` | Orange pose events (`:199`) | `camera.crop_px.v1` |
| `source_camera_image_px.top_left_y_down.v1` | Palette descriptor registry (`coordinate_descriptor.py:1249`) | `camera.recorded_px.v1` |
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
   today (`unified_h5/geometry.py:175-181`), while the legacy path uses xrandr
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

b. **Stale agent-contract.** `AC:palette-crimson/detect_bbox_read.md:128`
   still requires `pixel_convention == "continuous"` for the source-camera
   `bbox_img_xyxy` box. Palette publishes `pixel_edge_half_open`
   (`shared/coordinate_surface_contract.py:35`;
   `docs/crimson_detect_bbox_read_contract.md:82`), and
   `AC:palette-crimson/subject_shape_read.md:207` already says half-open.
   Fix that one line in agent-contracts. This was flagged in
   `docs/diagnostics/coordinate_contract_audit_2026-07-19.md` and is still open.
   - *Correction (2026-11-14):* the earlier draft also cited line 151. That
     line is the normalized `bbox_norm_coords` (`bbox_cxcywh`) descriptor, and
     `continuous` is correct there: Palette publishes it
     (`shared/coordinate_surface_contract.py:176`) and the descriptor validator
     requires it for normalized spaces (`shared/coordinate_descriptor.py:2015-2016`).
     Line 151 must not change.

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

## 7. Questions and answers

Both producers answered on this PR in October:
- Citrus, 2026-10-12: <https://github.com/jmdelahanty/palette/pull/377#issuecomment-6130745356> (evidence at Citrus `5feca8d`);
- Orange, 2026-10-13: <https://github.com/jmdelahanty/palette/pull/377#issuecomment-6150487989> (evidence at Orange `009eaf1`).

Both sent updates on 2026-11-14. Citrus's was checked against Citrus main
`be2cd47` and `/etc/pancakebatter/rigs/omnifin0.yml` rev 15.

> *Correction (2026-11-14):* an earlier revision of this section said the
> questions were sent to the owners on 2026-11-14 and still unanswered. They
> had already been answered in the October comments above. The 2026-11-14
> re-send was redundant, and it is what prompted the updates recorded here.

**Status key:**
- **answered:** nothing more is needed from the producer for the contract text;
- **answered, build pending:** the answer is settled, but the producer still has to build something;
- **open:** a decision or measurement is still missing.

Palette has independently checked the following against producer main:
- Citrus `be2cd47`: `src/core/display_snapshot.cpp:312-318` still takes the token from inside the parenthesis, and `src/ipc/pose_metadata.h:37` still sets `kCameraViewAssumed = "dorsal_unmirrored"`.
- Orange `ec15c94` (2026-10-27) is the commit that defines `camera_native_pixels` once in `docs/recording_metadata.md`.

Palette has **not** seen `rig.yml`: it is not on this workstation, and
`jmdelahanty/pancakebatter` main has no `rigs/` tree. Its contents below are as
reported by Citrus.

### For Orange

- **Q-O1.** Is `camera_native_pixels` (in `coordinate_frame` and rim observations) the recorded video raster after AOI, flips and binning, or the full sensor? Are `full_frame_pixels` and `source_frame_pixels` the same raster?
  - **Answered** (10-13; confirmed 11-14). All three are one raster, `camera.recorded_px.v1`. It is the AOI stream: unbinned, with no flip and no rotation. On pancake0 the AOI is the full sensor (4512×4512 at offset 0), so `camera.sensor_px` and `camera.recorded_px` coincide there. Orange defined the space once in `docs/recording_metadata.md` (`ec15c94`). Citrus's `camera_native_px` also belongs on this row of the alias table.
- **Q-O2.** Which camera settings that change geometry can be set on the rigs today: AOI OffsetX/Y, ReverseX/ReverseY, binning, decimation, any software rotation? Which of them are already in `camera_runtime` or `sensor_pipeline`, under which keys?
  - **Answered** (10-13; updated 11-14). Settable: AOI Width, Height, OffsetX and OffsetY, plus `Bin` (Orange requests 1x1).
  - The HB-20000SBM has no ReverseX/Y, Binning or Decimation nodes, and Orange applies no software rotation.
  - The inventory is recorded in `camera_runtime.<serial>.sensor_pipeline` (schema v3) and summarised in `raster_geometry`. Orange verified this on the rig on 2026-11-10.
  - In October `Bin` was neither read nor set. The 11-14 update says Orange now requests it.
- **Q-O3.** Can Orange seal `camera_geometry_v1` (§4) in the start snapshot, including `viewing_side` and `optical_mirror_count` from rig configuration? Who maintains the rig configuration?
  - **Answered, build pending (Jeremy).** Pancake Batter's `rigs/<rig_id>/rig.yml` owns the declared physical facts: `viewing_side`, mirrors and `recorded_space_parity`. Since 2026-11-03 the start snapshot pins the rig documents under `session.rig_config`.
  - The per-camera `camera_geometry_v1` record, built from the readback and refusing on a mismatch, is designed (10-13) but not built. It waits for Jeremy.
- **Q-O4.** Are any current or past rigs viewing from below (ventral), or through a fold mirror?
  - **Answered.** No rig views from below or through a fold mirror. omnifin0 is top-down and dorsal.
  - `rig.yml` lists the mirror count as `unknown` until someone reads it off the rig.
  - The undercolor colour camera has no recorded orientation (10-13 answer).
- **Q-O5.** Orange clamps crop windows inside the frame. Confirm that a crop's `x_px`/`y_px` are its top-left in `camera.recorded_px`, as integer pixel edges.
  - **Answered.** Confirmed: the origin is `clamp((int)cx − size/2, 0, W − size)`. The clamp shifts the window and never truncates it, and the rule is the same on every path.
  - Caveat for Palette: on the device path, pose events' `crop.x_px` is the **192 px pose window**, not the 384 px video crop in `crop_meta.csv`. Orange will add a `crop.role` field to say which.
  - Orange's implicit convention is `continuous_pixel_edge` for points and `pixel_edge_half_open` for boxes (10-13).

### For Citrus

- **Q-C1.** What physically explains the y reflection in omnifin0 arena 1's camera→canvas homography: rear projection, projector menu flip, a mirror, or the camera? Is the same true for its other arenas and for the other rigs?
  - **Answered for omnifin0** (11-14, as reported). `rig.yml` rev 15 declares:
    - `projection_mode` rear;
    - `projector_firmware_flip` vertical (BenQ "Rear Ceiling");
    - mount rotation 180°;
    - `optical_mirror_count` 0;
    - `viewing_side` dorsal.
  - That gives `canvas_space_parity` +1. Composed with Citrus's OpenGL presentation reflection, the signed chain is −1, which matches det < 0 on all four arenas. The rev-4 commissioning homographies (2026-11-13) are still det < 0.
  - In October one reflection was unaccounted for; the declared projector menu closes it.
  - Open: other rigs, and omnifin0 fits made before 2026-08-10, whose det > 0 is not evidence about the optics.
- **Q-C2.** Can Citrus capture `xrandr --verbose` for the selected output each session (current rotation, reflection, `Transform:`), and declare `output_rotation`, `output_reflection` and `output_transform_matrix`?
  - **Answered, build pending (Citrus).** The raw `--verbose` block is already stored per session under `/metadata/display`.
  - The token bug starts in Citrus's parser (`src/core/display_snapshot.cpp:312-318` at `be2cd47`), which still takes the wrong word. Citrus owns that fix; it is not made yet. The §6a Palette fix still applies to the value Palette derives.
  - The selected output is now DP-2 1080p240, on its own X screen.
- **Q-C3.** Can Citrus record `projection_mode`, `projector_mount`, `projector_firmware_flip` and the projection-path `optical_mirror_count` per rig? Where would they live: rig config, session metadata, or the geometry receipt?
  - **Answered, build pending.** The facts live in Pancake Batter's `rig.yml`.
  - Citrus pins the rig file (rig_id, revision, sha) and reads `media_stack`. It does not yet copy the parity fields into the geometry receipt.
- **Q-C4.** What pixel convention were the camera→canvas calibrations fitted in? Were the point pairs integer pixel centres from OpenCV, or edge coordinates?
  - **Answered.** The fits use pixel centres on both sides, with no ±0.5 shift, so `fit_pixel_convention = pixel_center`.
  - Only candidates fitted after 2026-10-05 declare it (`coordinate_contract` v2). The active 2026-09-14 artifacts carry v1 and no declaration.
  - Palette applies the +0.5 shift itself.
- **Q-C5.** Can the homography record carry `handedness_declared` and `orientation_basis` (§3)? Can Citrus refuse its own calibration when the matrix sign disagrees?
  - **Open (Jeremy).** The sign is measured today, but nothing declares or refuses it.
  - Design: record it at Fit; refuse at Promote with `declared_parity_mismatch` or `declared_parity_undeclared`; recheck at START. Not built; it waits for Jeremy.
- **Q-C6.** For the arena origin: are integer-valued floats guaranteed today, or can a fractional origin occur?
  - **Answered.** The origin is an integer by construction (int arithmetic) and becomes float64 only at the H5 boundary.
  - Citrus proposed the §6c rule: Palette accepts `float.is_integer()` and refuses anything else, and the field becomes an integer in the next schema. Adopting it is Jeremy's call.
- **Q-C7.** Is `pose_observation v2.camera_view_assumed` set from rig configuration, or constant? Can it come from Orange's `viewing_side` once that exists?
  - **Answered.** It is the compile-time constant `"dorsal_unmirrored"` (`src/ipc/pose_metadata.h:37`), and pose observation v3 pins it as a schema `const` (§9).
  - Deriving it from `viewing_side` needs a pose contract bump. Until then, Palette's §5 item 2 check compares against a constant. That still fails loudly the day a ventral or mirrored rig appears.

### What still blocks the move to agent-contracts (2026-11-14)

> *Update (2026-11-15):* Jeremy decided all four items in point 1 below. See §10
> for the decisions and what still remains.

No producer question blocks the contract text any more. What remains:
1. **Decisions for Jeremy:**
   - the Q-C5 design (declare at Fit, refuse at Promote, recheck at START);
   - building Orange's per-camera `camera_geometry_v1` (Q-O3);
   - the §6c arena-origin rule;
   - what to do with the phototaxis sessions already registered without orientation evidence (§9).
2. **Fixtures:** one preserved rig and one reflected rig. omnifin0 can serve as the reflected one once its `rig.yml` is readable by Palette.
3. **Read-off and verification:** omnifin0's mirror count is still `unknown`. Palette has not seen `rig.yml` rev 15 itself.

Producer build items (Citrus's parser fix in Q-C2, the parity fields in the geometry receipt in Q-C3, Orange's `crop.role`) belong to §8 step 3. They do not block the contract text.

## 8. Rollout

1. **Now, Palette only.**
   - The §6a xrandr parser fix.
   - The §6b agent-contracts one-line fix.
   - Stamp `orientation_status = "orientation_unverified"` on geometry runs from all current recordings, so nothing claims verified orientation.
     *Superseded 2026-11-15 (§10):* the status is derived per recording by one
     resolver. Existing runs are not stamped.
2. **Answers to §7** (received 2026-10-12/13 and updated 2026-11-14; what still
   blocks is listed at the end of §7), then the agent-contracts PR
   `systems/coordinate_spaces_v1/`, containing:
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

As of 2026-11-14 the omnifin0 parity chain is declared and accounted for
(Q-C1, Q-O2, Q-O4). The sign check and a `viewing_side`-backed
`camera_view_assumed` are still not built.

Until those land, phototaxis sessions are acquired under `orientation_unverified`. Each needs the one-off asymmetric-pattern check on its rig before its left/right results are trusted. **The minimum to start phototaxis safely:** steps 1 and 2, plus Citrus's Q-C1 answer and that check run on the phototaxis rig.

## 9. Re-verification and changes since 2026-10-13 (checked 2026-11-14)

**Citations fixed.** Every file:line citation was rechecked against palette
main `b04ec0d4` and agent-contracts main `de21c037`. These had drifted and are
updated above:
- `unified_h5/geometry.py`: determinant check 66-73 → 79-82, `camera_view_px`
  152 → 160, extent check 167-173 → 175-181;
- `zarr/body_frame_producer.py` 218-219 → 219-220;
- `recording_geometry.py`: `top_left_pixel` 400 → 446-451, and `camera_native_pixels` 287,323 → 446,374;
- `unified_h5/presentation.py` 123-130 → 123-131 and 132-139 → 132-140;
- `coordinate_frame_record.py` 142 → 143;
- `coordinate_descriptor.py` 1248 → 1249;
- §6b: line 151 dropped (see the correction there).

All other citations, including every `AC:` one, still hold.

**Merged since 2026-10-13 that bears on this doc:**
- **None of the §6 fixes has landed.** The xrandr token still comes from the
  parenthesised list (`selected_calibration.py:196-201`, `:1544`). Both
  `|det|`-only checks are unchanged. Nothing stamps `orientation_unverified`,
  and no code mentions it.
- **Pose observation v3** (`6988c81f`, 2026-10-14, admission revision 7)
  carries `camera_view_assumed`, but as a schema `const: "dorsal_unmirrored"`
  (`shared/contracts/citrus_pose_observation_contract.v3.schema.json:487-489`).
  This matches Citrus's Q-C7 answer: the value is a constant, not rig
  configuration. Under v3 a ventral or mirrored rig cannot declare itself at
  all; it would need a schema change. The semantic-hash v4 cases
  (`citrus_palette_semantic_hash_v4_cases.json`) also embed the constant.
- **Phototaxis is now admitted.** Admission revision 6 (`decdf91a`,
  2026-10-14) admits reactive heading-split `phototaxis_v1`, and real
  2026-10-17/18 phototaxis recordings have been registered since. The
  "What blocks phototaxis" gate in §8 was therefore not applied before
  acquisition. Those sessions carry no orientation evidence, and no
  asymmetric-pattern check is recorded for their rig. Their left/right
  results are `orientation_unverified` in substance, even though nothing
  stamps them yet.
- **Orange pose event v3** (`f0e12c8f`, frame_identity_v1) keeps
  `crop_pixels` and `source_frame_pixels` at the same lines, so the alias table
  is unchanged.
- **Dish geometry** (`1bad4def`): `exact_match_applied` now comes from Citrus's
  runtime geometry contract. That concerns registration status, not
  orientation, so it does not change any point here.

## 10. Decisions and status (2026-11-15)

Palette owner of this contract from 2026-11-15: the `coordinate_spaces`
session, handed over by PR-monitor.

**Jeremy's decisions (2026-11-15, given directly in the Palette owner session):**

| Item | Decision |
|---|---|
| Q-C5 handedness | **Approved** as designed: declare at Fit, refuse at Promote (`declared_parity_mismatch`, `declared_parity_undeclared`), recheck at START. Palette does not trust the producer's `status` alone: it requires `"agrees"`, recomputes sign(det) from the active homography (with the `w(p) > 0` check), parses the xrandr reflection itself, and requires the record's `rig_file_sha256` and `rig_geometry_revision` to equal the recording's `/metadata/rig/pancakebatter_rig_file` pin. Approving the design does not approve Citrus #115 for merge. |
| Q-O3 `camera_geometry_v1` | **Not built for now.** Palette computes `recorded_space_parity` from `camera_runtime.<serial>.sensor_pipeline` (v3) plus the `rig.yml` pinned in `session.rig_config`. The record gets built when a camera with ReverseX/Y, binning or decimation nodes, or a ventral or mirrored rig, appears. |
| §6c arena origin | **Accept an integral float (`640.0`) and refuse a fractional one (`640.5`); never round.** The field becomes integer in Citrus's next geometry schema version. The legacy path already enforced this. The unified-H5 path is palette#451. |
| 10-17/18 phototaxis sessions | omnifin0's camera cannot see the projection, so they cannot be verified from their own frames. Jeremy states the projector has not flipped or changed orientation since October. Run the §5 asymmetric-pattern check on omnifin0 while the NIR filters are off. That check plus this statement lifts `orientation_unverified` for those sessions. |
| `orientation_unverified` (§8 step 1) | **Derived, not stamped.** One resolver reports `orientation_status` per recording: verified only when a checked orientation evidence record exists for it, otherwise unverified. It may be projected into the registry. No existing run is mutated. New runs record the status in provenance when produced. |

**§6 fixes (status 2026-11-16):**
- §6a xrandr token: palette#450, merged `729a34cc`; live since intake re-pin `d190dd22` (2026-11-15 23:26Z).
- §6b `detect_bbox_read.md:128`: agent-contracts#131, merged `1362d753` (head `b33b63e`). Crimson main `01b424b` has no bbox `pixel_convention` check, so no reader changes.
- §6c: palette#451, merged `c1667a1c`; live since `d190dd22`.
- §6d: not started.

**Q-C5 declared orientation (status 2026-11-16):**
- Contract: agent-contracts#135, merged `3480e3e` (`citrus-palette/coordinate_spaces_v1/`, schema `citrus.declared_orientation_check` v1, sha256 `1bf06621…fa82`, rules R1–R4, consumer checks 1–4, admission by install-time cutoff).
- Citrus #115 was reverted (Citrus #125, `fec43d0`): its START check read the homography before it was resolved. The redo merged as Citrus #126 (`027d292a`) and was installed on omnifin0 at 2026-11-15T23:17:22.675Z.
- Palette: palette#458, merged `8f961c82`, checks the group in unified H5 admission. It is required on omnifin0 from **2026-11-15T23:17:22Z** (rounded down; no Citrus ran 23:17:07–23:18:17Z), kept in `contracts/palette_citrus_declared_orientation_cutoffs_v1.json`. Not live until an intake re-pin at or after `8f961c82`.
- Citrus's `selected_output_transform_token` now names the current rotation, matching palette#450. Files sealed before it need no change: Palette never trusted that attr alone, and all 236 legacy H5s re-derive `normal`/unreflected from their sealed raw block.

**Rim v4 / dish-mask follow-ups merged 2026-11-15 (not part of §6):** palette#453 (geometry contract v3, rim v4, H5 scope v3; live `d190dd22`), #454 (Palette recomputes Orange's `orange_dish_rim_auto_accept_v1` gate; live `d190dd22`), #455 (exact Citrus daily match needs `translation_matches_runtime` and `canonical_geometry_preserved`; live `d190dd22`), #456 (Orange physical-registration masks with the 24 h freshness window; live `73ccf2c5`) and #457 (acquisition candidate record v2 for physical masks; live `73ccf2c5`). Open: the Citrus H5 path still reads daily registration only.

**Presentation parity: resolved by measurement (see "Orientation check" below).** Kept for the record:
- #115's rule is `sign(det) = handedness × presentation_parity × xrandr_reflection`, with `presentation_parity = −1`. That value is a Citrus code constant (`presentation_mapping_contract.cpp`, `display_reflection = [[1,0,0],[0,−1,H],[0,0,1]]`), not a measurement.
- omnifin0's `rig.yml` also declares a vertical firmware flip. So "−1 × vertical flip" and "+1 × no flip" give the same det, and the det cannot tell them apart.
- If the presentation reflection really is applied at scan-out, then `citrus.display_canvas_px` is bottom-left-origin in xrandr output pixels, and the §2 row ("top-left of … one xrandr output") is wrong.
- omnifin0 fits made before 2026-08-10 have det > 0 under the same presentation code, and nothing explains that yet.
- Check proposed to Citrus: render a marker at canvas (0, 0) and capture DP-2's X screen. It needs no camera or filters, but it is a rig session, so it needs Jeremy's OK after Orange's hold clears.

### Orientation check, omnifin0, 2026-11-15 (NIR filters off)

Run on Jeremy's go. Rig `omnifin0`, `rig_geometry_revision` 16, rig file
`sha256:1934857fe156e7d3aa3873602760b67ce3ae1e74be181a8ddbd0e9df3f0c0332`.

**Pattern.** Citrus `static_dots`, white on black, in every arena: a 6 mm disk at the `top_left` preset and a 3 mm disk at `top_right`.

**Evidence (held by its producers; nothing is transferred yet, which is Jeremy's to start):**
- **Camera:** Orange guided capture, calibration session `calsess_2026_11_15T17_29_47Z_shadow`, commissioning record `omnifin0_20261115T172434Z_orientation_check_nir_filters_off_static_dots` step 02; 10 frames per camera at 10 ms, plus `black_reference`. On pancake0, read-only:
  - result JSON `/home/jeremy/orange_data/calibrations/commissioning/omnifin0_20261115T172434Z_orientation_check_nir_filters_off_static_dots/steps/02_static_dots_wet_tank_projected_surface_omnifin0_/orange_gui_guided_capture_20261115T172921Z.json`, sha256 `c8a07d0cebbbb351562b582e0146bf741e4564a412774eb089bee8bbb000f839`. The commanded `center_canvas_px`/`radius_px` are at `samples[1].workflow.citrus_scene_post_capture.static_dots`; large radius about 12.7 px, so about 4.23 canvas px/mm;
  - calibration session `/home/jeremy/orange_data/calibrations/sessions/calsess_2026_11_15T17_29_47Z_shadow`, `session_index.json` sha256 `0b4ec4121822aa49ac8af7466a3cbf995d8665a3f9e212495bc1aee8fc970185`;
  - `static_dots` capture sha256s (`black_reference` in brackets):
    - 2010093 `c40a0d2e…` (`864d2455…`);
    - 2010094 `302161d1…` (`07bf641f…`);
    - 2010095 `1176435b…` (`bd7f865a…`);
    - 2010096 `af9058c4…` (`c48385b4…`).
  - Orange's camera centroids were thresholded at 153 DN on saturated disks, so they may be biased by up to one pixel. The result JSON does not pin the rig revision; revision 16 comes from the host at 17:29Z.
- **Scan-out:** `xwd` of X screen `:1.1` (1920×1080) at 17:33:52Z, scene presented at 17:33:51.351Z; rig GUI `fde5467`. Files on pancake0: `screen_1_1.xwd` sha256 `74d30722bbe78f51a871751ec4ccc74607caa85f1096867ab8ec932c91c4e0de`, a PNG copy `ab1978e99ef085165e398d41cfef281d91eb3064f9677bf3f334fc60b9e33130`, and the scene status. Staged durably on pancake0 at `/mnt/Data2/palette_delivery/omnifin0-orientation-check-20261115/` with the scene status (`scene_status.json` sha256 `387f1400e333b844fb7647e28a1a4ca6fc08ee06d0d887d2d25e5cf34bbcb393`), Citrus logs, `README.md` and `SHA256SUMS` (12 entries). Copying them to `/groups` is Jeremy's call.
- **Active homographies:** set `calgrp_2026_11_13T00_27_59Z` (`homography_grid_projected_surface_g4`). Per arena, `homography_arena_N_<serial>.yml`, sha256s:
  - arena 1: `033f7b05…`, det −6.397e−3;
  - arena 2: `7da8a26f…`, det −6.379e−3;
  - arena 3: `a8e7dea6…`, det −6.391e−3;
  - arena 4: `720d5c13…`, det −6.381e−3.

**Per-dot measurements** (canvas centre from the scene status; scan-out centroid from the xwd; camera centroid from Orange):

| Arena (camera) | Dot | Canvas centre | Scan-out | Camera px |
|---|---|---|---|---|
| 1 (2010093) | tl | (186.7, 264.7) | (186.8, 814.2) | (1390, 3114) |
| 1 (2010093) | tr | (324.0, 262.0) | (323.9, 816.9) | (3119, 3149) |
| 2 (2010094) | tl | (642.7, 270.7) | (642.8, 808.2) | (1405, 3088) |
| 2 (2010094) | tr | (780.0, 268.0) | (780.0, 811.0) | (3127, 3124) |
| 3 (2010095) | tl | (1107.6, 264.6) | (1107.7, 814.3) | (1368, 3108) |
| 3 (2010095) | tr | (1245.0, 262.0) | (1245.2, 817.2) | (3092, 3139) |
| 4 (2010096) | tl | (1579.5, 255.5) | (1579.6, 823.4) | (1378, 3099) |
| 4 (2010096) | tr | (1717.2, 252.8) | (1717.2, 826.2) | (3099, 3125) |

**Findings:**
1. **Presentation parity is −1, measured.** On scan-out, every blob sits at `y = 1079 − canvas_y` (±0.1 px) with x unchanged, and the blob pixel counts equal the scene's. §2 is corrected above. #115's `kPresentationParity = −1` is confirmed, and the vertical firmware flip is now pinned separately from it.
2. **Camera ↔ canvas handedness: x preserved, y reflected (det < 0) on all four arenas.**
   - Through the active 11-13 homographies, every camera blob lands within 1.6 px of its canvas centre, with `w` = 0.996–1.000 (Citrus).
   - Palette checked independently with the homography sealed in the 10-17/18 phototaxis H5s, which is the same matrix in every Citrus H5 from 2026-10-09 to 2026-11-06, det < 0. It maps today's camera blobs onto the commanded dots within 0.2–8 px (≤ about 2 mm at 4.23 canvas px/mm). Every dot lands in its commanded corner.
   - The residual is the known November projector move (0.65–2.0 mm), a translation rather than a flip.
3. **10-17/18 phototaxis sessions.** Per Jeremy's decision, this check plus his statement that projector orientation did not change since October lifts `orientation_unverified` for the camera ↔ canvas link of those sessions.
4. **Still declared only: camera parity relative to the dish.** This check shows camera and projected canvas agree. It does not show the camera image is unmirrored relative to the fish. If it were, stimuli would still land on the body-frame "left", but that would be anatomical right. That link rests on Q-O2/Q-O4 (no ReverseX/Y nodes, dorsal view, `rig.yml` mirror count 0). A printed asymmetric target in a dish would measure it. Whether to require that is Jeremy's call.
5. *Unrelated:* Cam2010096 has a saturated spot near (754, 345) that is also in `black_reference`, so it is not from the projector.

**Still blocking the move to agent-contracts:**
- the presentation-parity check above and the asymmetric-pattern check;
- fixtures with one preserved rig and one reflected rig;
- a physical read-off of omnifin0's mirror count;
- a home for the three-party registry and transform-record schemas. palette-agent-contracts notes that `systems/` holds links only, so the `systems/coordinate_spaces_v1/` proposed in the Status line needs replacing.
