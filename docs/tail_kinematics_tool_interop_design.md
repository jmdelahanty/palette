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
  trajectory, and bout windows for preprocessing and classification.
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

Sources:

- <https://megabouts.ai/api/tracking_data.html>
- <https://megabouts.ai/api/preprocessing.html>
- <https://megabouts.ai/api/segmentation.html>

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

To interoperate with external tools, Palette should provide a tool-ready view or
export with the following logical fields:

```text
tail_posture_view/
  attrs:
    schema_id                         "analysis.tail_posture_view"
    schema_version                    1
    source_estimator_family           "subject_shape" | "pose_keypoints" | "external_import"
    source_run
    source_component_or_labels
    frame_rate_hz
    units_xy                          "px" | "mm"
    mm_per_unit                       optional
    body_frame_convention
    tail_base_definition
    tail_tip_definition
    tail_sample_domain                "tail_segment_normalized_arclength"
    tail_sample_count
    angle_convention
    angle_units                       "rad" | "deg"

  frame_indices                       (N,)
  time_s                              (N,) optional
  valid                               (N,)
  failure_reason_bytes                (N, width)

  head_x                              (N,) optional
  head_y                              (N,) optional
  head_yaw                            (N,) optional
  tail_x                              (N, K)
  tail_y                              (N, K)
  tail_tangent_angle                  (N, K - 1) optional
  tail_angle                          (N, K - 1) optional
  tail_curvature                      (N, K) optional
```

This does not need to be the primary permanent run family. Palette-native
behavior-facing tail metrics should live in
`analysis/tail_kinematics_runs/<run>` first, then tool-ready views can be
generated from those arrays plus the source subject-shape geometry. See
[tail_kinematics_run_design.md](tail_kinematics_run_design.md).

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
- produce `tail_x`, `tail_y` or `tail_angle`
- map Palette's signed body-frame tail angles into Megabouts' expected
  convention, with the conversion recorded in the export manifest
- map invalid Palette rows to `NaN` values or a no-tracking mask
- record exact source refs and conversion parameters

The first implementation can be export-only. Running Megabouts classification
inside Palette should remain optional because it introduces a third-party
runtime dependency and model/version concerns.

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

- [ ] Define exact `tail_posture_view` or `tool_views/megabouts` fields after
  first tail-kinematics canary.
- [ ] Implement a read-only exporter from `subject_shape_runs`.
- [ ] Implement a keypoint-derived exporter from `pose_kinematics_runs` or
  refined keypoints when tail labels exist.
- [ ] Add tests for angle convention, tail point ordering, and invalid-row
  handling.

### External Tool Adapters

- [ ] Prototype Megabouts export using Palette canary data.
- [ ] Decide whether Megabouts should run inside Palette or remain a separate
  optional command.
- [ ] Define ZebraZoom import/export mapping after inspecting real output files.
- [ ] Define Stytra import mapping if acquisition or online tracking emits
  Stytra tail traces.

### Classification Runs

- [ ] Define `analysis/bout_classification_runs` once first external classifier
  output is available.
- [ ] Store classifier labels, confidence/scores, source refs, and exact config.
- [ ] Keep classifier outputs independent from `swim_bout_runs` and
  `bout_kinematics_runs`.
- [ ] Add comparison tooling to evaluate multiple classifier candidates on the
  same source bouts.

## Open Questions

- Should the tool-ready posture view be persisted as a run or generated on
  demand from `subject_shape_runs` and `pose_kinematics_runs`?
- Should Palette's first tail-angle convention match Megabouts exactly, or
  should Palette store both its native convention and a Megabouts-compatible
  exported convention?
- Should `analysis/bout_classification_runs` depend on existing
  `swim_bout_runs`, or should tools that segment their own bouts write their
  own bout boundaries into the classifier/import run?
- How should classifier confidence be normalized across tools that expose
  different score semantics?
- What minimum canary review should be required before a tail-posture source is
  allowed into downstream classifiers?
