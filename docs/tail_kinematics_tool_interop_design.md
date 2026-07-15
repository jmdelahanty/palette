# Tail Kinematics Tool Interop Design
<!-- contract-meta
version: 1
status: draft
last_verified: 2026-05-01
-->

Purpose: define how Palette should make its Zarr datastores compatible with
tail-kinematics and behavior-classification tools such as Stytra, ZebraZoom,
Megabouts, and BEAST-style analyses without making any external tool format the
canonical Palette schema.

## Executive Summary

The goal is defensible and worth designing for.

Palette should own:

- reviewed source masks and keypoints
- body-frame and tail geometry provenance
- ordered, sampled body/tail curves
- tail-angle and curvature arrays with explicit units and conventions
- per-bout or per-frame classifier outputs with source refs

External packages should be treated as methods or adapters:

- Stytra-like and ZebraZoom-like methods can extract or import tail tracking
  points and tail angles.
- Megabouts can consume Palette-derived tail keypoints, tail angles,
  trajectory, and bout windows for preprocessing and classification. The first
  Palette target is classifier-only: Palette supplies the bout windows and
  Megabouts classifies them.
- BEAST/Johnson-style analyses can consume bout-aligned tail-shape, heading,
  and eye-gaze arrays.

The interoperability layer should therefore be:

```text
Palette canonical source data
  -> Palette tail-kinematics run
  -> Palette tail-posture view/export
  -> external method/classifier
  -> Palette classifier or imported-analysis run
```

It should not be:

```text
external tool output
  -> silent mutation of refined masks, keypoints, or subject-shape geometry
```

## Terminology

### Skeletonization

In image processing, skeletonization usually means thinning a binary mask to a
one-pixel-wide medial structure. Palette currently uses this as one possible
intermediate for extracting `centerline_xy`.

This is not the final scientific representation for tail behavior. Raw skeleton
pixels are branch-prone, jagged, and sensitive to attached artifacts. They are
useful for candidate extraction, QC, and debugging.

### Ordered Tail Curve

An ordered tail curve is a sequence of points from `tail_base_xy` to
`tail_tip_xy`, sampled along anatomical arclength. This is the first stable
representation that can be compared across frames.

Palette should produce this as either:

- a sampled centerline
- a B-spline sample
- keypoint-derived tail points
- an imported tool-derived tail trace

For mask-derived workflows, the preferred mature representation is a B-spline:
a fitted continuous curve model of the ordered centerline. Smoothing is a
parameterized choice within the spline fit, not the definition of the spline.
Palette should record whether a spline is interpolating the source centerline or
using smoothing/regularization, because downstream tail angles and curvature
depend on that choice.

### Tail Angles

Tail angles are computed from ordered tail points or a fitted curve. They can be
stored as:

- absolute tangent angle per segment
- angle relative to the body frame
- cumulative segment angle
- curvature over normalized tail arclength

Tools differ in exact convention, so Palette must record the convention, units,
orientation, sample count, and source geometry.

### Native Versus Tool-Specific Tail Angles

Palette's native `analysis/tail_kinematics_runs/<run>/tail_angle_rad` is a
behavior-facing body-frame tangent representation. Each channel answers:

```text
At this normalized tail position, what is the local tail tangent angle relative
to the fish's caudal body axis?
```

Megabouts' `tail_angle` is different. Given ordered tail keypoints, Megabouts
computes cumulative signed segment angles. Each channel answers:

```text
After walking from tail base through this segment, what cumulative bend has
accumulated relative to the body-to-tail-base vector?
```

These are related geometric summaries of the same tail curve, but they are not
the same dataset:

- Palette samples local tangent directions on a subject-shape curve or spline.
- Megabouts samples discrete keypoint-to-keypoint segment directions.
- Palette references its anatomical body frame.
- Megabouts references the vector from head to the first tail point, then
  accumulates relative segment rotations.
- Palette's default native run currently has `K=10` angle samples.
- Megabouts keypoint input uses `K=11` ordered keypoints to produce `K=10`
  cumulative segment-angle channels.

Therefore, tool-specific angles should be generated as adapter views rather
than by redefining Palette's native `tail_angle_rad`.

## What The Existing Literature And Tools Do

### Stytra

Stytra describes larval tail posture as a curve discretized into 7-10 segments.
Its tail tracker finds the next segment from the previous segment position and
orientation using image intensity or center-of-mass sampling windows. It also
notes that tail shape can be interpolated to a fixed segment count for
cross-setup comparison.

Implication for Palette: a sampled ordered tail curve is the compatible
surface, not a raw skeleton pixel image.

Source:

- <https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1006699>

### pi_tailtrack

pi_tailtrack uses the word `skeletonized`, but its practical output is a set of
up to 10 ordered tracking points. It iteratively places points from a starting
coordinate using a semicircle-intersection rule and depends on consistent fish
orientation in the imaging setup.

Implication for Palette: if Palette uses the term skeleton, it should
distinguish raw medial-axis pixels from ordered tail tracking points.

Source:

- <https://journals.biologists.com/jeb/article/226/22/jeb246335/335476/pi-tailtrack-A-compact-inexpensive-and-open-source>

### ZebraZoom

ZebraZoom tracks core/head geometry and tail points, computes tail-bending
angle, fits a spline to the tail midline for curvature, and reports movement
and bout-level parameters. Its data API exposes tail-angle heatmaps per bout or
time interval and plotting helpers for tracking points.

Implication for Palette: curvature should be derived from an ordered/smoothed
midline or spline, while bout classifiers should consume tail-angle traces and
validated bout windows.

Sources:

- <https://www.frontiersin.org/journals/neural-circuits/articles/10.3389/fncir.2013.00107/full>
- <https://zebrazoom.org/documentation/docs/behaviorAnalysis/dataapi/>
- <https://zebrazoom.org/documentation/docs/behaviorAnalysis/behaviorAnalysisGUI/>

### Megabouts

Megabouts is mostly downstream of tracking. It accepts tracking data such as
head position, heading, tail keypoints, or tail angles. Its preprocessing
includes missing-value interpolation, PCA/eigen-fish denoising, Savitzky-Golay
smoothing, baseline correction, and tail vigor. Its segmentation/classification
APIs extract bout-aligned tail arrays and trajectory arrays.

Implication for Palette: Megabouts compatibility requires a clean adapter from
Palette arrays to:

- `head_x`, `head_y`, `head_yaw`
- `tail_x`, `tail_y` or `tail_angle`
- `fps`
- `mm_per_unit`
- optional bout onset/offset windows

For Megabouts keypoint input, Palette should provide 11 ordered tail keypoints
from tail base / swim-bladder side to tail tip. Megabouts converts those into
10 cumulative tail-angle segments. This is distinct from Palette's current
`analysis/tail_kinematics_runs` default `K=10` behavior-facing tail-angle
samples, though a K=11 Palette candidate is a reasonable comparison target.

Sources:

- <https://megabouts.ai/api/tracking_data.html>
- <https://megabouts.ai/api/preprocessing.html>
- <https://megabouts.ai/api/segmentation.html>

Palette's direct in-memory integration plan is documented in
[megabouts_direct_integration_design.md](megabouts_direct_integration_design.md).

#### Dependency And Attribution Policy

Palette should interoperate with Megabouts without requiring Megabouts for the
default Palette install. The default output should be a
Megabouts-compatible posture view derived from Palette geometry. Actual
Megabouts preprocessing and classification should be optional third-party
execution.

This distinction matters because Megabouts is distributed under a
non-commercial research and academic-use license. Palette should not copy or
vendor Megabouts source code or model weights. If Palette calls Megabouts APIs,
the derived run must record the package version or checkout commit, the
non-commercial license, and the citation:

```text
Jouary et al., "Megabouts: a flexible pipeline for zebrafish locomotion
analysis", bioRxiv, doi:10.1101/2024.09.14.613078
```

Palette-native preprocessing can use standard operations such as interpolation,
PCA denoising, Savitzky-Golay smoothing, baseline subtraction, and vigor
estimation, but that method must be named and attributed as Palette-native
unless it directly calls Megabouts.

### BEAST / Johnson-Style Analyses

BEAST-style behavior modeling treats larval behavior as sequences of bouts and
interbout intervals, with features such as heading change, eye vergence, prey
geometry, and tail-shape change. The public `beast` repository currently checked
out locally mainly points to compressed data and a Colab demo:

```text
~/gitrepos/beast/README.md
```

Implication for Palette: the useful target is not a full BEAST clone inside
Palette. The useful target is a clean per-bout feature table with source refs
for heading, eye angles, tail-shape arrays, prey/object detections, and bout
boundaries.

Source:

- <https://www.sciencedirect.com/science/article/pii/S0960982219314654>

## Palette-Compatible Data Model

### Canonical Inputs

Palette should support tail-kinematics extraction from either masks or
keypoints:

```text
refined_subject_masks_runs/<run>
  subject_body
  swim_bladder
  eye_left
  eye_right

analysis/subject_shape_runs/<run>
  body_frame/
  components/subject_body/centerline_xy
  components/subject_body/tail_base_xy
  components/subject_body/tail_tip_xy
  components/subject_body/tail_sample_xy
  components/subject_body/tail_tangent_xy
  components/subject_body/tail_curvature_px_inv

refined_keypoints_runs/<run>
  optional tail keypoints or pose-schema tail_tip

analysis/track_kinematics_runs/<run>
  position, heading, speed, frame/time base

analysis/swim_bout_runs/<run>
  bout candidate windows
```

Mask-derived and keypoint-derived tails are comparable, but they are different
estimators. They should not overwrite each other.

### Tool-Ready Tail Posture View

To interoperate with external tools, Palette provides a tool-ready view family:
`analysis/tail_posture_view_runs`. This family is a regenerated compatibility
surface, not a replacement for `analysis/tail_kinematics_runs`.

The v1 writer implemented in `fisheye.analysis.tail_posture_view_runs` is
Megabouts-compatible and Palette-owned. It requires no Megabouts installation,
does not copy Megabouts code, and records that boundary in attrs. Future
ZebraZoom, Stytra, or Palette-native views can use the same sibling family with
different `view_family` and convention attrs.

```text
analysis/tail_posture_view_runs/<run>/
  attrs:
    schema_id                         "analysis.tail_posture_view_runs"
    schema_version                    1
    method                            "tail_posture_view_from_subject_shape"
    method_version                    1
    row_axis                          "roi_rows"
    view_family                       "megabouts_compatible"
    compatible_tool                   "megabouts"
    dependency_policy                 "no_megabouts_dependency_required"
    source_subject_shape_run
    source_subject_shape_path
    source_refined_subject_masks_run
    source_tail_kinematics_run        optional comparison source
    source_tail_geometry_kind         "subject_shape_tail_curve_resample"
    head_source                       "head_endpoint_xy" | "snout_tip_xy"
    keypoint_count                    11
    angle_count                       10
    keypoint_order                    "tail_base_to_tail_tip"
    tail_base_definition
    tail_tip_definition
    angle_convention                  "megabouts_cumulative_segment_angle"
    angle_units_primary               "rad"
    frame_index_source
    algorithm_provenance

  valid                               (N,)
  failure_reason_bytes                (N, width)
  frame_index                         (N,)
  row_index/                          copied source row lineage when present
  head_xy                             (N, 2)
  head_yaw_rad                        (N,)
  tail_keypoints_xy                   (N, 11, 2)
  tail_angle_rad                      (N, 10)
  tail_angle_deg                      (N, 10)
```

This is not the primary canonical tail-metrics family. Palette-native
behavior-facing tail metrics live in `analysis/tail_kinematics_runs/<run>`;
tool-ready views are regenerated compatibility surfaces derived from those
arrays plus source subject-shape geometry. For tools whose angle semantics
differ from Palette's native tangent angles, the view should derive directly
from the source ordered curve/keypoints and record that conversion. See
[tail_kinematics_run_design.md](tail_kinematics_run_design.md).

Execution note: the v1 writer retains a serial backend and now also supports
node-local `process_shards` execution. Each task owns one complete,
non-overlapping physical output shard across every row-aligned tail array. The
driver alone creates groups, copies lineage, writes attrs, validates the run,
and marks completion. Shared-storage publication remains serialized and
atomic. The ownership rules are documented in
[dask_zarr_write_safety.md](dask_zarr_write_safety.md).

### External Classification Outputs

Behavior labels from Megabouts, ZebraZoom, Stytra, or BEAST-like models should
land in a separate classifier run, not in the source geometry run.

Recommended future shape:

```text
analysis/bout_classification_runs/<run>/
  attrs:
    schema_id                         "analysis.bout_classification_runs"
    schema_version                    1
    classifier_family                 "megabouts" | "zebrazoom" | "stytra" | "beast_style" | "palette_native"
    classifier_name
    classifier_version
    source_tail_posture_run_or_export
    source_track_kinematics_run
    source_swim_bout_run              optional
    source_eye_angle_run              optional
    source_object_detection_run       optional
    config_json
    created_at_utc

  per_bout/
    source_bout_id
    start_frame
    end_frame
    class_id
    class_label_bytes
    confidence                        optional
    classifier_score                  optional
    failure_reason_bytes              optional

  features/                           optional, method-specific but documented
```

The classifier run may store compact method-specific features, but it should
not duplicate dense video, masks, or full tail-posture arrays unless required
for reproducibility and explicitly named as an export artifact.

### Tail Posture Preprocessing Runs

Preprocessing is an algorithmic transform and should be separate from
tool-compatible posture views. This lets users compare different preprocessing
methods without regenerating geometry or classifier labels.

Recommended future shape:

```text
analysis/tail_posture_preprocessing_runs/<run>/
  attrs:
    schema_id                         "analysis.tail_posture_preprocessing_runs"
    schema_version                    1
    preprocessing_family              "palette_standard_tail_preprocessing" | "megabouts"
    source_tail_posture_view_run
    config_json
    api_entrypoint                    optional
    package_version                   optional
    package_git_commit                optional
    license                           optional
    citation                          optional

  frame_index
  angle_raw_rad
  angle_processed_rad
  angle_baseline_rad
  tail_vigor
  no_tracking
  valid
  failure_reason_bytes
```

## Adapter Strategy

### Megabouts Adapter

Megabouts is the most natural first adapter because it expects already-tracked
tail and trajectory arrays. Palette can provide these from `subject_shape_runs`
or keypoint-derived tail posture.

Megabouts compatibility does not mean Megabouts owns Palette's tail schema.
Palette should compute reusable tail primitives first, then map them into a
Megabouts-compatible view or export. Megabouts-derived classifications should
return as imported classifier outputs with source refs, not as mutations to
Palette-native tail metrics.

Initial adapter responsibilities:

- resolve frame/time base and FPS
- resolve calibrated units when available
- produce `head_x`, `head_y`, `head_yaw`
- produce `tail_x`, `tail_y` with 11 ordered tail keypoints for Megabouts
  keypoint input, or produce a separately audited Megabouts-compatible
  `tail_angle`
- map Palette's signed body-frame tail angles into Megabouts' expected
  convention, with the conversion recorded in the export manifest
- map invalid Palette rows to `NaN` values or a no-tracking mask
- record exact source refs and conversion parameters

The first implementation should be a read-only classifier adapter over
Palette-selected `analysis/swim_bout_runs` windows. File exports remain useful
for debugging and cross-tool inspection, but they should not be required for
the initial Palette-to-Megabouts path.

### ZebraZoom Adapter

ZebraZoom can be used in two ways:

- as an external tracker/classifier run on video, then imported into Palette
- as a conceptual schema target for tail-angle heatmaps, bends, TBF, and
  bout-level metrics

Because ZebraZoom is more end-to-end than Megabouts, Palette should not assume
it will always consume Palette-derived tail curves directly. An importer is
probably as useful as an exporter.

### Stytra Adapter

Stytra is especially relevant for head-restrained or closed-loop experiments.
For Palette, the useful compatibility target is ordered segment geometry and
tail-angle traces. If Stytra runs online during acquisition, Palette should
import its traced tail points with source refs rather than recomputing them
silently.

### BEAST-Style Adapter

BEAST-style modeling should be treated as a downstream feature/classification
consumer. Palette should provide:

- bout onset/offset and interbout windows
- heading change metrics
- eye vergence metrics
- tail-shape change metrics
- prey/object positions in body-centered coordinates when object detections
  exist

This belongs downstream of current `bout_kinematics_runs`,
`eye_angle_runs`, and future tail-posture/pose-kinematics outputs.

## Design Rules

1. Do not persist raw skeleton pixels as the primary scientific output.
2. Persist or export ordered tail points, spline samples, tail angles,
   curvature, and validity.
3. Always record body-frame, polarity, angle convention, sample count, units,
   spline degree/parameterization, interpolation versus smoothing mode,
   smoothing/filtering parameters, and source refs.
4. Keep third-party classifier labels in separate classifier/import runs.
5. Do not mutate refined masks, refined keypoints, subject-shape runs, or
   swim-bout segmentation outputs when running a classifier.
6. Make tool adapters optional and versioned.
7. Allow multiple classifier runs for the same source data so Megabouts,
   ZebraZoom, Stytra, and Palette-native methods can be compared.
8. Keep keypoint-only and mask-only workflows both valid.

## Relationship To Existing Palette Docs

- [body_spline_tail_anchor_design.md](body_spline_tail_anchor_design.md)
  defines the mask/spline-derived tail geometry and tail landmark conventions.
- [subject_shape_runs_contract.md](subject_shape_runs_contract.md) defines
  where interpreted mask-derived shape geometry lives.
- [tail_kinematics_run_design.md](tail_kinematics_run_design.md) defines the
  first Palette-native tail-angle, tail-deflection, and curvature metric run.
- [pose_kinematics_run_design.md](pose_kinematics_run_design.md) defines where
  keypoint/skeleton-derived geometry should live.
- [bout_kinematics_run_design.md](bout_kinematics_run_design.md) defines
  downstream per-bout heading and movement metrics.

This document sits one layer above those contracts. It defines how those
Palette-native arrays should be exported to, or compared against, external
behavior-analysis packages.

## Implementation Checklist

### Documentation

- [x] Document skeletonization versus ordered tail-curve versus tail-angle
  analysis.
- [x] Document external tool interoperability goal.
- [x] Document that Megabouts is the most natural first adapter because it
  consumes tracked arrays rather than requiring end-to-end video tracking.

### Palette Geometry

- [x] Finish first mask-derived tail sampling arrays in `subject_shape_runs`.
- [x] Add a validated B-spline method, with sampled-centerline fallback if
  spline fitting fails.
- [x] Add tail tangent and curvature outputs.
- [x] Add overlay visualization for tail samples, tangents, normals, and
  B-spline review.
- [ ] Add width-profile outputs.
- [ ] Add persisted summary plots for subject-shape tail validity and length
  distributions.

### Palette Tail Kinematics

- [x] Define a dedicated `analysis/tail_kinematics_runs` design.
- [x] Implement frame-level tail angles, tail-tip angles, lateral deflections,
  and curvature summaries from subject-shape tail samples.
- [x] Add tests for angle convention, left/right sign, straight-tail zero angle,
  and invalid-row propagation.
- [ ] Add visualization artifacts and Marimo loading after the run schema
  stabilizes.

### Tool-Ready View

- [x] Define exact `analysis/tail_posture_view_runs` v1 fields after the
  first tail-kinematics canary.
- [x] Run the direct Megabouts convention audit from
  [megabouts_direct_integration_design.md](megabouts_direct_integration_design.md)
  before treating Palette `tail_angle_rad` as Megabouts-compatible.
- [x] Implement a read-only posture-view writer from `subject_shape_runs`.
- [x] Run the feeding canary posture-view writer:
  `tail_posture_view_megabouts_compatible_canary_20260501` wrote 17,495 valid
  rows and 1,740 invalid rows from 19,235 ROI rows.
- [x] Prototype classifier-only Megabouts integration using Palette
  `swim_bout_runs` windows and K=11 Megabouts tail keypoints.
- [ ] Implement a keypoint-derived exporter from `pose_kinematics_runs` or
  refined keypoints when tail labels exist.
- [x] Add tests for angle convention, tail point ordering, and invalid-row
  handling for the subject-shape posture view.

### External Tool Adapters

- [x] Prototype Megabouts classifier-only execution using Palette canary data.
- [ ] Decide whether Megabouts should run as an installed optional dependency,
  an explicitly configured command, or both.
- [ ] Define ZebraZoom import/export mapping after inspecting real output files.
- [ ] Define Stytra import mapping if acquisition or online tracking emits
  Stytra tail traces.

### Classification Runs

- [x] Define `analysis/bout_classification_runs` once first external classifier
  output is available.
- [x] Store classifier labels, confidence/scores, source refs, and exact config.
- [x] Keep classifier outputs independent from `swim_bout_runs` and
  `bout_kinematics_runs`.
- [ ] Add comparison tooling to evaluate multiple classifier candidates on the
  same source bouts.

## Resolved Initial Direction

The first `analysis/bout_classification_runs` implementation should depend on
existing Palette `swim_bout_runs`. Future tool-segmented candidates may write
their own bout boundaries into classifier/import runs for comparison, but they
should not replace Palette swim-bout candidates by default.

## Open Questions

- Should the tool-ready posture view be persisted as a run or generated on
  demand from `subject_shape_runs` and `pose_kinematics_runs`?
- Should Palette's first tail-angle convention match Megabouts exactly, or
  should Palette store both its native convention and a Megabouts-compatible
  exported convention?
- How should classifier confidence be normalized across tools that expose
  different score semantics?
- What minimum canary review should be required before a tail-posture source is
  allowed into downstream classifiers?
