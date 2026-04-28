# Eye-Angle Metrics: Data Layout and Computation

This note summarizes where the eye-angle products are written inside a Palette archive and how each quantity is derived from the upstream detections, keypoints, and refined eye geometry. It reflects the v4 eye-angle schema where the biologically preferred gaze axis is explicit, signed angles are resolved through a body-frame support contract, and BEAST/Johnson-style mean per-eye vergence is available without changing the existing total-vergence surface.

## Where the data lives

Eye-angle analysis runs are stored under:

```
analysis/eye_angle_runs/<run>/
    angles/
        roi/            # per-detection signals
        frame/          # optional frame-aligned signals
    qa/
        roi/
        frame/
    support/
```

Key datasets:

- `angles/roi/left_gaze_deg`, `right_gaze_deg`, `left_gaze_signed_deg`, `right_gaze_signed_deg`, `vergence_gaze_deg`, `vergence_gaze_signed_deg`, and `version_gaze_deg` are the preferred biological eye-angle surface. These are derived from the ellipse minor axis, which matches the apparent gaze/look direction in the current overhead fish imagery.
- `angles/roi/left_nasal_gaze_deg`, `right_nasal_gaze_deg`, and `mean_eye_vergence_gaze_deg` are v4 BEAST/Johnson-comparable outputs. The per-eye nasal fields estimate inward/nasal rotation from the lateral eye-axis baseline, and the mean field is `0.5 * (left_nasal_gaze_deg + right_nasal_gaze_deg)`.
- `angles/roi/left_deg`, `right_deg`, `left_signed_deg`, `right_signed_deg`, `vergence_deg`, and `version_deg` are legacy major-axis ellipse outputs retained for compatibility and geometry QA.
- `angles/roi/left_minor_signed_deg`, `right_minor_signed_deg`, `vergence_minor_signed_deg`, and `version_minor_deg` are the legacy names for the gaze-axis values. In v2, the explicit `*_gaze_*` names should be preferred by new readers.
- `angles/roi/left_centroid_deg`, `right_centroid_deg`, `vergence_centroid_deg` hold centroid-position angles in fish-frame coordinates. These are useful pose context, but they are not a replacement for ellipse-derived gaze/vergence.
- `_delta_deg` and `_delta_deg_smoothed` arrays contain absolute frame-to-frame changes.
- `qa/roi/valid_left`, `valid_right`, `valid_frame`, and `reason_codes` provide flags and bitmasks that explain any exclusions.
- `support/time_seconds`, `frame_indices`, `ellipse_*` expose timing metadata and ellipse diagnostics used by the visualizations.

Current runs resolve eye geometry through `fisheye.shared.eye_geometry_source`.
The preferred source is `analysis/subject_shape_runs/<run>` when it contains
`eye_left` and `eye_right` component ellipse geometry. If no subject-shape
geometry is available, the resolver falls back to
`refined_subject_masks_runs/<run>` with eye component geometry, then to
historical `refined_eye_masks_runs/<run>` data as a compatibility fallback.

The referenced sources are captured in run attributes:

- `schema_id = "analysis.eye_angle_runs"` and `schema_version = 4`: stable
  run-level contract for this analysis product.
- `method = "ellipse_and_centroid_eye_angles"`: the writer computes both
  ellipse-axis and centroid-position eye-angle families.
- `row_axis = "keypoint_detection_rows"`: ROI outputs are row-aligned to the
  refined keypoint/eye-geometry detection rows.
- `eye_angle_output_schema`: machine-readable summary of output groups,
  row axes, units, suffix conventions, derivative outputs, and QA reason-code
  linkage.
- `source_eye_geometry_stage` and `source_eye_geometry_run`: the actual stage
  and run used for geometry.
- `source_geometry_kind`: normalized geometry role, one of
  `subject_shape_eye_geometry`, `refined_subject_eye_geometry`, or
  `legacy_refined_eye_geometry`; unknown future stages are recorded as
  `unknown_eye_geometry`.
- `source_subject_shape_run`: subject-shape source when analysis-facing shape
  geometry was used.
- `source_refined_subject_masks_run`: canonical refined-subject source when
  available.
- `source_refined_eye_run`: compatibility refined-eye source when one was used
  or mapped.
- `source_keypoints_run`: canonical refined keypoint source. The legacy
  `source_keypoint_run` alias may be mirrored during migration.

The raw ROIs sampled by the viewer live under `keypoints_runs/<run>/roi_images`.

Schema boundary:

- `schema_id` / `schema_version` identify the run family contract.
- `eye_angle_output_schema` describes the current output layout, units, suffix
  conventions, derivative arrays, reason-code links, and mixed support row
  axes. It is not a replacement for source lineage attrs.
- `source_geometry_kind` records which eye-geometry authority was actually
  consumed so readers do not have to infer semantics from path names alone.
- `preferred_angle_family = "gaze"` and `preferred_eye_axis = "ellipse_minor"`
  tell readers which arrays to use for biological eye orientation by default.

Body-frame boundary:

- Current v4 eye-angle runs materialize a keypoint-derived body-frame support
  group under `support/body_frame/`.
- Future schema updates should prefer `analysis/subject_shape_runs/<run>/body_frame/`
  when a coherent mask/spline/keypoint body frame exists and fall back to
  `pose_schema.metadata.heading_computation` for keypoint-only datasets.
- See `docs/body_frame_contract.md`.

## Execution model

`fisheye.analysis.eye_angle_analysis` writes base ROI outputs directly into the
target zarr run in row chunks. The default `serial_driver` backend processes
those chunks in the driver process. `--execution-backend dask_worker_chunks`
lets Dask workers reopen the archive and write disjoint ROI row chunks; use
`--scheduler threads`, `--scheduler processes`, or `--scheduler distributed`
depending on the workload.

Smoothing, deltas, speed, acceleration, and frame-level projection are computed
after the base ROI pass. Those second-pass products are intentionally
driver-side for now because smoothing and derivatives need adjacent rows across
chunk boundaries.

## Angle conventions

Angles are generated inside `fisheye.analysis.eye_angle_analysis._process_chunk`, which receives:

1. Keypoint ROIs (swim bladder, left/right eye centers).
2. Eye-geometry ellipse fits, preferably from `analysis/subject_shape_runs`.
3. A keypoint-derived body-frame support group computed from the same keypoints.

### Per-eye angles

For each detection and eye:

1. We materialize `support/body_frame/` from the swim bladder and left/right eye keypoints. The forward axis is `swim_bladder -> midpoint(eye_left, eye_right)` and the left axis is resolved from the labeled eye pair.
2. The ellipse major-axis direction (`theta_deg`) is converted to a unit vector. To remove the 180° axis ambiguity, the vector is flipped so it points generally forward in the body frame.
3. The signed major-axis angle is `atan2(dot(axis, left_axis), dot(axis, forward_axis))`. Positive values point toward anatomical left. The unsigned major-axis magnitude is `abs(left/right_signed_deg)`.
4. The minor-axis direction is produced by rotating the major axis by 90°. It is also disambiguated toward the forward half-plane before the same body-frame `atan2` computation, producing the minor/gaze per-eye signed angles.
5. In schema v3 and later, the minor-axis family is aliased into explicit gaze names: `left_gaze_signed_deg = left_minor_signed_deg`, `right_gaze_signed_deg = right_minor_signed_deg`, `vergence_gaze_signed_deg = vergence_minor_signed_deg`, and `version_gaze_deg = version_minor_deg`.
6. In schema v4, per-eye BEAST-comparable nasal gaze is additionally computed as `left/right_nasal_gaze_deg = 90 - abs(left/right_gaze_signed_deg)`, and `mean_eye_vergence_gaze_deg = 0.5 * (left_nasal_gaze_deg + right_nasal_gaze_deg)`.

Invalid or near-circular fits are rejected early; reason bits (`REASON_*`) mark any failure so consumers can down-weight those detections.

### Binocular aggregates

Once left and right signed angles are available:

- Per-eye signed angles are body-frame anatomical-left-positive.
- Ellipse axes are directionless, even after we choose a forward-facing representative vector for each eye. Therefore **gaze/minor-axis vergence** is the smaller angle between the two undirected eye-axis lines: `min(abs(left - right), 180 - abs(left - right))`.
- `vergence_gaze_deg` and `vergence_minor_signed_deg` store that nonnegative axis separation. The `*_signed` suffix is retained for compatibility with the older output family, not because this binocular aggregate is a directed vector delta.
- **Version** is `0.5 * (left_signed_deg + right_signed_deg)`, the shared anatomical-left/right component.
- Gaze/minor-axis variants follow the same algebra: `vergence_gaze_deg = undirected_axis_separation(left_gaze_signed_deg, right_gaze_signed_deg)` and `version_gaze_deg = 0.5*(left_gaze_signed_deg + right_gaze_signed_deg)`. The legacy name `version_minor_deg` contains the same version values.
- **Mean per-eye vergence** is `mean_eye_vergence_gaze_deg = 0.5 * (left_nasal_gaze_deg + right_nasal_gaze_deg)`. This is the Palette field intended for BEAST/Johnson-style comparisons because BEAST stores per-eye vergence angles and commonly plots their mean rather than the bilateral axis-separation total.
- `vergence_gaze_deg` is retained as the v3-compatible total/axis-separation aggregate. Under the expected outward anatomical eye-axis polarity it is equivalent to the sum of left/right nasal gaze, but consumers should use `mean_eye_vergence_gaze_deg` when they want a per-eye mean comparable to BEAST `iEyes`/`bEyes` traces.

### Relationship to Johnson et al. style eye tracking

Johnson-style hunting analyses register the fish, fit ellipses to the eyes, and use ellipse-derived looking/vergence angles for hunting-state classification. Palette follows the same broad geometry strategy, but OpenCV-style ellipse parameters expose major/minor axes explicitly and, in current imagery, the minor axis is the axis that visually tracks where the eye is looking. Therefore schema v3 made the minor-axis family the explicit `gaze` family rather than asking readers to infer this from `minor` names.

Schema v4 keeps those existing gaze fields and adds explicit per-eye nasal gaze plus `mean_eye_vergence_gaze_deg`. The local BEAST data examined in `/nvme1/beast_data` stores two per-eye eye-angle columns; the median of the per-frame mean is approximately half the sum, which matches the Johnson-style plots more closely than Palette's older total axis-separation field.

The centroid outputs are auxiliary pose-position measurements, not the Johnson-style gaze signal. They measure the *position* of each eye centroid relative to the fish's heading, rather than the *orientation* of the eye ellipse.

For each detection:

1. Compute `head_center = mean(swim_bladder, left_eye, right_eye)`.
2. Build vectors from `head_center` to each eye centroid.
3. Convert to math coordinates (`y` flipped to point up) to match the heading convention.
4. Rotate into the fish frame by `-heading_rad` so the heading aligns with `+x`.
5. Compute per-eye angles: `theta_L = atan2(Ly, Lx)`, `theta_R = atan2(Ry, Rx)`.
6. Compute vergence: `vergence_centroid = |theta_L| + |theta_R|`.

Centroid-based vergence can be useful as a diagnostic or covariate, but downstream hunting-state classification should prefer the gaze-axis family. Use `mean_eye_vergence_gaze_deg` for BEAST/Johnson-style mean per-eye convergence, and use `vergence_gaze_deg` when a v3-compatible total/axis-separation aggregate is required.

Outputs:
- `angles/roi/left_centroid_deg`, `right_centroid_deg`, `vergence_centroid_deg`
- Smoothed and delta variants follow the same naming pattern as ellipse-based angles.
- Frame-level equivalents in `angles/frame/`.

### Deltas and smoothing

Absolute per-step changes (`*_delta_deg`) are computed with `_compute_delta`, preserving NaNs. When smoothing windows are configured, rolling averages are applied after the base computation, and the same delta routine is run on the smoothed series.

## Auxiliary products

- `support/ellipse_major`, `ellipse_minor`, and `ellipse_ratio` capture the geometric properties of the fitted ellipses and are useful for QA thresholds.
- Heading (in degrees) is re-serialized alongside the angles so downstream viewers can overlay the fish’s forward axis (`heading_deg`).
- Frame-level outputs repeat the same signals after the detections are resampled onto the video frame timeline; they live under `angles/frame/` with matching schema.

## Visual tools

- `fisheye.visualization.visualize_eye_angles` renders dashboards from an eye-angle run, including the unsigned/signed series, delta plots, and QA summaries.
- `fisheye.visualization.visualize_eye_angle_overlays` overlays masks, headings, and signed/unsigned values on the original ROIs.

Both tools default to the gaze family when available and fall back to legacy minor-axis arrays for older runs. The overlay viewer starts in gaze/minor-axis mode because that is the biologically preferred axis for the current imagery.

## Bout-level summaries

`analysis/eye_angle_runs/<run>` owns frame- and ROI-aligned eye geometry and gaze
traces. It should not be mutated to add bout-derived summaries.

Bout-level Johnson-style summaries belong in
`analysis/bout_kinematics_runs/<run>/eye_gaze/per_bout_metrics/`. That derived
table links to an exact eye-angle run and an exact swim-bout candidate, then
stores only per-bout pre/post/within-window aggregates such as vergence means,
valid fractions, and optional convergence fractions. This keeps eye-angle traces
reusable while allowing bout windows and segmentation candidates to be compared
without rewriting the eye-angle source.
