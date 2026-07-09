# Subject-Mask Component Contour Sampling Check

**Date:** 2026-07-08
**Status:** empirical diagnostic on the RedScare training canary

## Context

Refined subject-mask runs store component contours as ragged point lists. For
display, compact review surfaces, and future spline-like summaries, we need a
fixed-size representation that is small but still visually faithful to the
original contour.

This check compared original persisted component contours against closed
arc-length resampled contours at fixed `K` values using real ROI image underlays
from:

```text
/groups/johnson/johnsonlab/jeremy/recordings/2026-06-23T16-01-09Z_arena_1_RedScare/zarr/2026-06-23T16-01-09Z_arena_1_RedScare_training.zarr
```

Run:

```text
refined_subject_masks_sam3_body_existing_eye_swim_red_scare_v3_canary_20260628_01
```

Calibration was resolved from `analysis/calibration.attrs[pixel_to_mm]`:

```text
pixel_to_mm = 0.018788045498284132
```

## Diagnostic Method

Tool:

```bash
scripts/py -m fisheye.diagnostics.plot_sampled_component_contours
```

The diagnostic renders:

- original contour over the ROI image
- sampled contour over the same ROI image
- optional extra fixed-K sampled panels

It also reports nearest-vertex similarity metrics:

- `raw_to_sampled_p95_px`: p95 distance from original contour vertices to the
  sampled contour vertices, in ROI pixels
- `symmetric_hausdorff_px`: worst bidirectional nearest-vertex distance
- calibrated millimeter equivalents when `pixel_to_mm` is available
- p95 distance normalized by contour bbox diagonal and perimeter

These are vertex-set approximation metrics, not biological error metrics. They
are useful for judging whether a fixed-K contour visibly preserves the stored
boundary at the pixel scale used by the archive.

## Findings

### Subject Body

Tested on high-vertex body examples:

| rows | raw vertices | K | p95 px | p95 mm | symmetric Hausdorff px | bbox-diagonal fraction |
|---|---:|---:|---:|---:|---:|---:|
| 9, 156 | 475-483 | 128 | 1.94-1.96 | 0.036-0.037 | 2.08-2.14 | 0.0088-0.0091 |

Interpretation:

- `K=128` visually preserved the body outline on the tested RedScare examples.
- The p95 approximation error was under `1%` of the component bbox diagonal.
- This is a reasonable compact body contour candidate for display and derived
  summaries.
- Do not treat `K=128` as a replacement for dense masks during review/editing;
  dense `masks_roi` remains the authoritative editable mask surface.

Recommendation:

- Use `K=128` as the current body contour sampling candidate.
- Re-test before lowering body below `K=128`, especially for curled tails,
  strong bends, and large body masks with fine tail geometry.

### Swim Bladder

Tested on rows with longer swim-bladder contours:

| rows | raw vertices | K | p95 px | p95 mm | symmetric Hausdorff px | bbox-diagonal fraction |
|---|---:|---:|---:|---:|---:|---:|
| 183, 189 | 63 | 96 | 0.34-0.36 | 0.006-0.007 | 0.69 | 0.0109-0.0117 |
| 183, 189 | 63 | 32 | 1.00-1.05 | 0.019-0.020 | 1.07-1.09 | 0.0316-0.0348 |

Interpretation:

- `K=96` is effectively oversampling these examples and preserves the contour
  very tightly.
- `K=32` still looked visually close on the ROI image, with p95 error around
  one pixel.
- Because the swim bladder is a small rigid component, a one-pixel boundary
  difference is more noticeable as a fraction of component size than it is for
  the body.

Recommendation:

- Use `K=32` for compact display/overview if storage width matters.
- Use `K=64` or higher for geometry-sensitive review or derived shape metrics
  until more recordings are sampled.
- `K=96` is safe but likely more than needed for this component.

### Eyes

Tested separately for left and right eyes:

| component | rows | raw vertices | K | p95 px | p95 mm | symmetric Hausdorff px | bbox-diagonal fraction |
|---|---|---:|---:|---:|---:|---:|---:|
| `eye_left` | 38, 15 | 45 | 64 | 0.36-0.37 | 0.0068-0.0069 | 0.65-0.71 | 0.0171-0.0172 |
| `eye_left` | 38, 15 | 45 | 32 | 0.75-0.77 | 0.0142-0.0144 | 0.78-0.80 | 0.0358-0.0359 |
| `eye_right` | 196, 134 | 46 | 64 | 0.36-0.37 | 0.0069-0.0070 | 0.70-0.71 | 0.0162-0.0170 |
| `eye_right` | 196, 134 | 46 | 32 | 0.77-0.78 | 0.0144-0.0147 | 0.80-0.82 | 0.0339-0.0355 |

Interpretation:

- `K=32` was visually acceptable for both eyes in these examples.
- `K=64` roughly halves the p95 vertex approximation error relative to `K=32`.
- Eye geometry is a biologically important downstream surface, so preserving
  subtle boundary and ellipse behavior matters more here than it does for a
  thumbnail display overlay.

Recommendation:

- Use `K=64` as the safer default for eye review, geometry, and analysis-facing
  sampled contours.
- `K=32` is acceptable for compact display/overview paths where the contour is
  not used to recompute eye geometry.
- Do not use sampled contours as the source of truth for eye angle or ellipse
  metrics unless that metric path is explicitly validated against dense masks.

## Current Safe Defaults

| component | recommended K | safe use |
|---|---:|---|
| `subject_body` | 128 | display, compact derived contour, body outline summary |
| `swim_bladder` | 32 for display; 64+ for geometry-sensitive work | compact display at 32; review/shape validation at 64+ |
| `eye_left` | 64 | review, geometry-facing sampled contour, display |
| `eye_right` | 64 | review, geometry-facing sampled contour, display |

These fixed-K contours are derived caches. They should not replace dense
`refined_subject_masks_runs/<run>/masks_roi` for editing, training export, or
canonical mask-pixel authority.

## Caveats

- This was sampled on one RedScare canary training run. Treat these values as
  current candidates, not universal guarantees.
- The distance metric compares contour vertices, not filled mask overlap.
  Future validation should add IoU or signed-distance-mask checks if sampled
  contours will drive geometry computation.
- Small components should be judged in real units and as a fraction of their own
  bbox diagonal. One pixel is negligible for a full body outline but meaningful
  for an eye.
- The eye contour is not just a display object. Eye orientation and ellipse
  geometry are downstream analysis signals, so eye sampled contours should stay
  more conservative than thumbnail-only overlays.

## Reproduction Examples

Body:

```bash
scripts/py -m fisheye.diagnostics.plot_sampled_component_contours \
  --zarr /groups/johnson/johnsonlab/jeremy/recordings/2026-06-23T16-01-09Z_arena_1_RedScare/zarr/2026-06-23T16-01-09Z_arena_1_RedScare_training.zarr \
  --run refined_subject_masks_sam3_body_existing_eye_swim_red_scare_v3_canary_20260628_01 \
  --component subject_body \
  --rows 9,156 \
  --component-k subject_body=128 \
  --layout comparison \
  --image-source crop \
  --output /tmp/palette_contour_real_roi_redscare_subject_body_k128.png \
  --json
```

Swim bladder:

```bash
scripts/py -m fisheye.diagnostics.plot_sampled_component_contours \
  --zarr /groups/johnson/johnsonlab/jeremy/recordings/2026-06-23T16-01-09Z_arena_1_RedScare/zarr/2026-06-23T16-01-09Z_arena_1_RedScare_training.zarr \
  --run refined_subject_masks_sam3_body_existing_eye_swim_red_scare_v3_canary_20260628_01 \
  --component swim_bladder \
  --rows 183,189 \
  --component-k swim_bladder=32 \
  --layout comparison \
  --image-source crop \
  --output /tmp/palette_contour_real_roi_redscare_swim_bladder_k32.png \
  --json
```

Eyes:

```bash
scripts/py -m fisheye.diagnostics.plot_sampled_component_contours \
  --zarr /groups/johnson/johnsonlab/jeremy/recordings/2026-06-23T16-01-09Z_arena_1_RedScare/zarr/2026-06-23T16-01-09Z_arena_1_RedScare_training.zarr \
  --run refined_subject_masks_sam3_body_existing_eye_swim_red_scare_v3_canary_20260628_01 \
  --component eye_left \
  --rows 38,15 \
  --component-k eye_left=32 \
  --layout comparison \
  --image-source crop \
  --output /tmp/palette_contour_real_roi_redscare_eye_left_k32.png \
  --json
```
