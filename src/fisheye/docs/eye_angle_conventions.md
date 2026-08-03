# Eye-Angle Metrics: Data Layout and Computation

This note summarizes where the eye-angle products are written inside a Palette archive and how each quantity is derived from the upstream detections, keypoints, and subject-mask eye geometry. It reflects the exact compact v7 eye-angle run schema and v9 output schema: the ellipse major axis is the canonical stored eye-orientation axis, gaze/minor direction is derived from that resolved major axis, BEAST/Johnson-style and Bianco/Engert-style vergence surfaces are available without changing the existing total-vergence surface, a machine-readable variant schema classifies those surfaces for UI selection, and versioned algorithm/source contracts make the exact computation reproducible. For a field-by-field user guide to every angle variant, see `docs/eye_angle_variants.md`.

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

- `angles/roi/left_major_signed_deg` and `right_major_signed_deg` are the canonical per-eye orientation surface. They store the resolved ellipse major axis in the fish body frame, with `0 deg` aligned to the body forward axis and positive rotation toward anatomical left.
- `angles/roi/left_eye_angle_deg`, `right_eye_angle_deg`, and `vergence_eye_angle_deg` are Bianco/Engert-style eye-frame angles derived from the canonical major-axis fields. Per-eye values are nasal-positive for each eye: `left_eye_angle_deg = -left_major_signed_deg`, `right_eye_angle_deg = right_major_signed_deg`, and `vergence_eye_angle_deg = left_eye_angle_deg + right_eye_angle_deg`. Positive vergence means convergence, negative means divergence, and zero means no convergence component.
- `angles/roi/left_gaze_deg`, `right_gaze_deg`, `left_gaze_signed_deg`, `right_gaze_signed_deg`, `vergence_gaze_deg`, `vergence_gaze_signed_deg`, and `version_gaze_deg` are the gaze surface. These are derived by rotating the resolved major axis by eye-specific 90 degree offsets, not by independently resolving the minor-axis half-plane.
- `angles/roi/left_gaze_xy` and `right_gaze_xy` store ROI/image-space unit vectors for the same derived gaze directions so visualizers can draw rays without re-deriving the axis.
- `angles/roi/left_nasal_gaze_deg`, `right_nasal_gaze_deg`, and `mean_eye_vergence_gaze_deg` are BEAST/Johnson-comparable outputs. The per-eye nasal fields estimate inward/nasal rotation from the lateral eye-axis baseline, and the mean field is `0.5 * (left_nasal_gaze_deg + right_nasal_gaze_deg)`.
- `angles/roi/left_deg`, `right_deg`, `left_signed_deg`, `right_signed_deg`, `vergence_deg`, and `version_deg` are legacy major-axis outputs retained for compatibility. `left/right_signed_deg` are aliases of `left/right_major_signed_deg` in v5.
- `angles/roi/left_minor_signed_deg`, `right_minor_signed_deg`, `vergence_minor_signed_deg`, and `version_minor_deg` are the legacy names for the gaze-axis values. The explicit `*_gaze_*` names should be preferred by new readers.
- `angles/roi/left_centroid_deg`, `right_centroid_deg`, `vergence_centroid_deg` hold centroid-position angles in fish-frame coordinates. These are useful pose context, but they are not a replacement for ellipse-derived gaze/vergence.
- `_delta_deg` and `_delta_deg_smoothed` arrays contain absolute frame-to-frame changes.
- `qa/roi/valid_left`, `valid_right`, `valid_frame`, and `reason_codes` provide flags and bitmasks that explain any exclusions.
- `qa/roi/left_major_axis_marginal`, `right_major_axis_marginal`, and `major_axis_marginal` are non-fatal warnings for the rare case where the major axis is close to the forward half-plane boundary and 180 degree ambiguity resolution is therefore less certain.
- `support/instance_key` is the ordered observation identity copied from the
  exact base-keypoint publication sealed by subject-shape assignment.
- `support/source_acquisition_frame_index` is the canonical acquisition-frame
  coordinate for each ROI row. `support/frame_indices` is a compatibility alias
  and must be byte-for-byte equal to it.
- `support/time_seconds` and `ellipse_*` expose timing metadata and ellipse
  diagnostics used by the visualizations.

Canonical runs resolve eye geometry through
`fisheye.shared.eye_geometry_source` from a completed
`analysis/subject_shape_runs/<run>` publication containing `eye_left` and
`eye_right` ellipse geometry. The same subject-shape publication must seal an
assignment proof for one exact completed `keypoints_runs/<run>` child. Palette
reloads that child and fails closed unless crop placement, keypoint labels,
success mask, ordered `instance_key`, ordered acquisition-frame index, and
source frame count agree with the sealed proof.

Historical refined keypoints are not a canonical fallback. They may be used
only with the explicit `--diagnostic-refined-keypoint-run` option, and the
result is permanently marked nonselector with diagnostic-only publication
scope. Ordinary `--keypoint-run` is only an assertion that the caller expected
the exact base-keypoint child already sealed by the selected subject-shape run.

The referenced sources are captured in run attributes:

- `schema_id = "analysis.eye_angle_runs"` and `schema_version = 7`: exact
  compact-dense-v2 array contract. The closed schema-v2-v6 legacy-layout
  allowlist remains behind explicit compatibility. This is the stable
  run-level contract for this analysis product.
- `method = "ellipse_and_centroid_eye_angles"`: the writer computes both
  ellipse-axis and centroid-position eye-angle families.
- `method_version = "eye_angle_analysis.v5"`: gaze direction is derived from
  the resolved major axis and is no longer clipped to +/-90 deg.
- `row_axis = "keypoint_detection_rows"`: ROI outputs are row-aligned to the
  subject-shape-sealed base-keypoint rows. The ordered identity is explicit in
  `support/instance_key`; the acquisition-time coordinate is explicit in
  `support/source_acquisition_frame_index`.
- `eye_angle_output_schema`: machine-readable summary of output groups,
  row axes, units, suffix conventions, derivative outputs, QA reason-code
  linkage, and `variant_schema`. Output schema v6 adds Bianco/Engert-style
  eye-frame fields, and output schema v7 adds the UI-facing representation
  registry while leaving the run schema and v5 method semantics intact. Output
  schema v8 adds the versioned algorithm-contract link and exact temporal
  operator identities. Output schema v9 adds canonical row identity and
  acquisition-frame support surfaces.
- `support/frame_time_seconds` is required for every maintained v7 run. The
  dense `roi_angles` channel named `heading_deg` is a compatibility alias only;
  its values must exactly equal authoritative
  `support/body_frame/heading_deg`.
- `eye_angle_variant_schema`: mirror of
  `eye_angle_output_schema.variant_schema`. Consumers can use it to present
  selectable `eye_frame`, `gaze`, `nasal_gaze`, `major`, `centroid`, and
  `legacy` angle representations without hardcoding field-name groups.
- `eye_angle_algorithm_contract`: versioned, machine-readable record of the
  exact ellipse parameter order and rejection rule, resolved keypoint indices,
  body-frame construction, 180-degree axis resolution, eye-frame/gaze
  transforms, smoothing, delta and derivative operators, frame projection,
  FPS source, and every numerical threshold. Its contract identity is
  `analysis.eye_angle_algorithm_contract`, version 1. This is separate from
  `method_version` because adding more precise provenance does not change the
  v5 scientific calculation.

Maintained discovery and ordinary reads accept only compact run schema v7.
Historical run schemas v2-v6 are available solely through an explicit
`legacy_compatibility=True` policy. Open/read validation checks exact arrays,
semantic indexes, group attributes, and the reconstructed output/algorithm
manifests without scanning all scientific values. Publication validation adds
chunked identity/alias checks, including exact heading-alias equality; exhaustive
scientific fill/null audits remain canary or maintenance work.
- `eye_angle_source_contracts`: resolved paths and available schema, method,
  completion, git, and lineage-fingerprint attrs for the eye geometry and exact
  keypoint source. Its canonical keypoint authority binds the subject-shape
  assignment proof to `keypoints_roi`, `detection_success`, `instance_key`,
  `source_acquisition_frame_index`, crop placement, labels, and temporal
  authority. Source-shape component attrs preserve the upstream ellipse
  estimator, such as `cv2.fitEllipse_component_contour_v1`.
- `source_eye_geometry_stage` and `source_eye_geometry_run`: the actual stage
  and run used for geometry.
- `source_geometry_kind`: normalized geometry role, one of
  `subject_shape_eye_geometry` or `refined_subject_eye_geometry`; unknown future
  or historical stages are recorded as
  `unknown_eye_geometry`.
- `source_subject_shape_run`: subject-shape source when analysis-facing shape
  geometry was used.
- `source_refined_subject_masks_run`: canonical refined-subject source when
  available.
- `source_refined_eye_run`: historical lineage only when a subject-mask source
  was seeded from compatibility refined-eye data.
- `source_base_keypoints_run`: exact canonical base-keypoint source sealed by
  subject shape.
- `source_refined_keypoints_diagnostic_run`: present only on an explicitly
  requested historical diagnostic run; such a run is never selector eligible.

The raw ROIs sampled by the viewer live under `keypoints_runs/<run>/roi_images`.

Schema boundary:

- `schema_id` / `schema_version` identify the run family contract.
- `eye_angle_output_schema` describes the current output layout, units, suffix
  conventions, derivative arrays, reason-code links, and mixed support row
  axes. Its nested `variant_schema` describes UI-selectable angle
  representations. It is not a replacement for source lineage attrs.
- `source_geometry_kind` records which eye-geometry authority was actually
  consumed so readers do not have to infer semantics from path names alone.
- `preferred_angle_family = "gaze"` and `preferred_eye_axis = "ellipse_major"`
  tell readers that the run's canonical orientation axis is major-axis based
  while the gaze family remains the historical preferred biological viewing
  surface. UI angle-trace selectors should use
  `eye_angle_variant_schema.default_representation` instead.

Body-frame boundary:

- Current v6 eye-angle runs materialize a keypoint-derived body-frame support
  group under `support/body_frame/`.
- The body frame is recomputed from the same sealed keypoint payload used by
  the analysis. A separately persisted upstream heading is not an input.
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
3. The exact success mask, ordered `instance_key`, and ordered acquisition-frame
   coordinate bound to those keypoints by the subject-shape assignment proof.

`_process_chunk` computes the body frame from the keypoints and success mask;
it does not accept a separately persisted heading input.

### Per-eye angles

For each detection and eye:

1. We materialize `support/body_frame/` from the swim bladder and left/right eye keypoints. The forward axis is `swim_bladder -> midpoint(eye_left, eye_right)` and the left axis is resolved from the labeled eye pair.
2. The ellipse major-axis direction (`theta_deg`) is converted to a unit vector. To remove the 180° axis ambiguity, the vector is flipped so it points generally forward in the body frame.
3. The signed major-axis angle is `atan2(dot(axis, left_axis), dot(axis, forward_axis))`. Positive values point toward anatomical left. The unsigned major-axis magnitude is `abs(left/right_signed_deg)`.
4. The gaze/minor-axis direction is produced by rotating the resolved major axis in body-frame coordinates. For the left eye, `gaze = major + 90 deg`; for the right eye, `gaze = major - 90 deg`. The old independent minor-axis half-plane test and +/-90 deg clipping are not used in v5.
5. The derived gaze family is aliased into explicit gaze names: `left_gaze_signed_deg = left_minor_signed_deg`, `right_gaze_signed_deg = right_minor_signed_deg`, `vergence_gaze_signed_deg = vergence_minor_signed_deg`, and `version_gaze_deg = version_minor_deg`.
6. Per-eye BEAST-comparable nasal gaze is computed as `left/right_nasal_gaze_deg = 90 - abs(left/right_gaze_signed_deg)`, and `mean_eye_vergence_gaze_deg = 0.5 * (left_nasal_gaze_deg + right_nasal_gaze_deg)`.

Invalid or near-circular fits are rejected early; reason bits (`REASON_*`) mark any failure so consumers can down-weight those detections. Major-axis marginal rows are warnings only and do not invalidate the frame.

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

Johnson-style hunting analyses register the fish, fit ellipses to the eyes, and use ellipse-derived looking/vergence angles for hunting-state classification. Palette follows the same broad geometry strategy, but OpenCV-style ellipse parameters expose major/minor axes explicitly. Schema v5 stores the resolved major axis as the canonical eye orientation and derives the gaze/minor axis from it so 180 degree ambiguity is resolved on the geometrically stable axis.

Schema v5 keeps the existing gaze fields and explicit per-eye nasal gaze plus `mean_eye_vergence_gaze_deg`. The local BEAST data examined in `/nvme1/beast_data` stores two per-eye eye-angle columns; the median of the per-frame mean is approximately half the sum, which matches the Johnson-style plots more closely than Palette's older total axis-separation field.

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

Absolute per-step changes (`*_delta_deg`) are
`abs(value[row] - value[row - 1])`; the first row and any pair containing a
non-finite value are `NaN`. These deltas are not time-normalized. Smoothing is
a centered, NaN-aware boxcar implemented with `numpy.convolve(mode="same")`;
edge windows are partial and normalized by their finite sample count. The
requested window is capped to the sequence length, made odd by decrementing an
even value, and disabled below three samples. Smoothed deltas apply the same
absolute adjacent-difference operator to the smoothed series.

Angular speeds use a backward difference to the previous valid sample. A
sample is left `NaN` when `dt <= 0` or the valid-sample gap exceeds 0.25 s.
Angular acceleration applies the same operator to angular speed. These
details, including requested/effective smoothing windows and the actual FPS
source, are persisted in `eye_angle_algorithm_contract`.

## Auxiliary products

- `support/ellipse_major`, `ellipse_minor`, and `ellipse_ratio` capture the geometric properties of the fitted ellipses and are useful for QA thresholds.
- Heading is derived from the sealed keypoints and re-serialized alongside the
  angles so downstream viewers can overlay the fish's forward axis
  (`support/body_frame/heading_deg` and its angle-output compatibility field).
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
