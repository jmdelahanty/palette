# Tuning and Masks

Several pipeline stages depend on per-recording (or per-camera) parameters
that need to be tuned interactively before batch processing can work well.
This guide covers the tuning tools, what order to use them, and how to
propagate settings across recordings.

## Overview

```
  dish mask          detect threshold        sub-arena masks
  (camera FOV)       (fish blob params)      (multi-fish only)
       |                    |                       |
       v                    v                       v
   background ──────> detection ──────────> arena assignment
                            |
                            v
                          crop
                            |
              ┌─────────────┼──────────────┐
              v             v              v
         keypoints        subject masks / components
              |                     |
              v                     v
       swim bladder          body, eyes, swim bladder
```

Tuning happens at the top of each branch. Once you tune one recording per
camera, you can propagate those settings to all other recordings from the same
camera.

---

## Dish mask

The dish mask defines the circular boundary of the petri dish in the camera
frame. Detection and segmentation stages use it to ignore everything outside
the dish.

**Future direction:** Citrus will eventually create the dish mask at runtime
during acquisition, embedding it directly in the recording. When that lands,
the organize step will carry the mask into the recording automatically and
this entire tune/propagate/review cycle goes away. Until then, the workflow
below is how dish masks get created. The acquisition-side contract is tracked in
`docs/operator_guide/citrus_dish_mask_handoff.md`.

### Tune one recording

```bash
scripts/py -m fisheye.tune.mask_tuner \
  /path/to/zarr/..._analysis.zarr \
  --registry /nvme1/palette_registry.sqlite
```

This opens an interactive GUI with OpenCV trackbars. Adjust the circle until
it fits the dish edge, then save and close.

Options:
- `--full` — use full-resolution frames (default: downsampled)
- `--frame N` — tune on a specific frame index
- `--mode circle|rectangle|auto` — mask shape (default: auto-detect)

The tuning is saved to `analysis_metadata.attrs["dish_mask"]` — metadata only,
no arrays. It stores the circle center/radius and the Hough parameters used to
find it. When `--registry` is provided, a successful save also upserts
`recording_step_status.step_name="dish_mask"` as `ok` for the matching registry
dataset row. Without `--registry`, the Zarr write is still complete and the
registry can be refreshed later by maintenance/backfill.

For production Zarrs that keep only `raw_video` metadata, the tuner falls back
to the recorded `source_video_path` and tunes in the analysis/inference
coordinate space by default. Use `--full` only when you intentionally want to
save full-resolution coordinates.

### Tune missing masks from the registry

You can omit the positional zarr path and let `mask_tuner` query the registry
for analysis zarrs that still need masks:

```bash
scripts/py -m fisheye.tune.mask_tuner \
  --registry /groups/johnson/johnsonlab/jeremy/registries/palette_registry.sqlite \
  --path-contains GoodCopBadCop \
  --missing-only
```

This opens the same interactive tuner one zarr at a time. Closing one tuner
window returns to the terminal, where Enter advances to the next candidate and
`q` exits the batch. Add `--list-only` to print the selected zarr paths without
opening the GUI:

```bash
scripts/py -m fisheye.tune.mask_tuner \
  --registry /groups/johnson/johnsonlab/jeremy/registries/palette_registry.sqlite \
  --path-contains GoodCopBadCop \
  --missing-only \
  --list-only
```

Shell continuation matters: the trailing `\` must be at the end of the line it
continues. A line containing only `\` does not continue the previous command.

### Propagate to other recordings

Dish masks are grouped by `experimental_chamber` (dish type). The batch apply
re-detects the circle on each target recording using the Hough parameters from
your tuned source:

```bash
# Dry-run
scripts/py -m fisheye.utils.apply_dish_mask_by_chamber /nvme1/recordings \
  --recursive \
  --source /path/to/tuned/zarr/..._analysis.zarr \
  --registry /nvme1/palette_registry.sqlite

# Apply
scripts/py -m fisheye.utils.apply_dish_mask_by_chamber /nvme1/recordings \
  --recursive \
  --source /path/to/tuned/zarr/..._analysis.zarr \
  --registry /nvme1/palette_registry.sqlite \
  --apply
```

### Review the batch results

The batch apply gets you close, but dish center/scale can drift between
sessions. Always review:

```bash
scripts/py -m fisheye.utils.review_dish_masks /nvme1/recordings \
  --recursive \
  --chamber cedar \
  --registry /nvme1/palette_registry.sqlite
```

This steps through each recording one at a time, launching the tuner for each.
Useful flags:
- `--only-present` — verify batch-applied masks
- `--only-missing` — find recordings without masks
- `--start N` / `--limit N` — partial review passes

To drive the review list from the registry rather than from Zarr attrs on disk,
use `--source registry`. This is the preferred mode after recordings have been
registered because every save marks `recording_step_status.dish_mask=ok`:

```bash
scripts/py -m fisheye.utils.review_dish_masks /nvme1/recordings \
  --source registry \
  --only-missing \
  --registry /nvme1/palette_registry.sqlite
```

If registry mode shows fewer recordings than expected, first confirm the
analysis zarrs have dataset rows in that registry. Managed imports should be
run with `--registry`; for already-created zarrs, repair registry visibility
with:

```bash
scripts/py -m fisheye.utils.registry_rescan /path/to/recordings \
  --recursive \
  --registry /nvme1/palette_registry.sqlite
```

---

## Tuning is camera-specific

All tuning parameters — dish mask coordinates, detection thresholds,
segmentation settings — are tied to a specific physical camera. Pixel
positions, lighting, and optics differ between cameras, so tuning from one
camera will produce wrong results on another.

The propagation tools enforce this:

- `apply_tuning_by_camera` matches on `camera_id` and only copies to
  recordings from the **same** camera.
- `apply_dish_mask_by_chamber` re-detects the circle per target (it copies
  Hough detection parameters, not raw pixel coordinates), but still within
  the same chamber grouping.

**Never manually copy tuning metadata between cameras.** If you have a new
camera, tune one of its recordings from scratch and propagate from there.

---

## Detection threshold

Controls the blob detection parameters used by the traditional (non-YOLO)
detector: threshold, morphology kernel sizes, and area filters.

```bash
scripts/py -m fisheye.tune.detect_threshold_tuner \
  /path/to/zarr/..._analysis.zarr \
  --config configs/fisheye/default.yaml
```

Saved to `analysis_metadata.attrs["detection_tuning"]`.

If you're using YOLO detection (the default in the pipeline), you may not need
this step — it's primarily for the traditional blob detector.

---

## Sub-arena masks (multi-fish only)

If your dish contains multiple arenas (sub-regions with one fish each), define
them so arena assignment can track fish per-arena.

```bash
scripts/py -m fisheye.tune.subdish_mask_tuner \
  /path/to/zarr/..._analysis.zarr
```

Draw rectangles around each sub-arena in the interactive GUI. Each rectangle
gets an arena ID.

Saved to `analysis_metadata.attrs["subdish_mask_tuning"]` as a list of
rectangle definitions.

---

## Background

Backgrounds are computed (not tuned), but they're needed by several
segmentation stages. Compute them before tuning subject masks or eye masks.

```bash
# Single recording
scripts/py -m fisheye.preprocessing.background \
  /path/to/zarr/..._analysis.zarr

# Batch
scripts/py -m fisheye.utils.compute_backgrounds_batch /nvme1/recordings \
  --recursive \
  --apply
```

Options:
- `--method mode|median` — aggregation method (default: mode)
- `--sample-size N` — number of frames to sample (default: 500)
- `--full` / `--ds` — which resolutions to compute

Writes to `background_runs/` in the Zarr.

---

## Keypoint tuning

Controls the 3-point keypoint detector (swim bladder, left eye, right eye)
when using the traditional geometry-based method.

```bash
scripts/py -m fisheye.tune.keypoint_tuner \
  /path/to/zarr/..._analysis.zarr
```

Adjust blob detection thresholds, minimum area, and triangle geometry
constraints. Saved to `analysis_metadata.attrs["keypoint_tuning"]`.

---

## Eye mask tuning

Controls traditional (threshold + Sobel) eye segmentation parameters.

```bash
scripts/py -m fisheye.tune.eye_mask_tuner \
  /path/to/zarr/..._analysis.zarr
```

Options:
- `--crop-run NAME` — specific crop run (default: latest)
- `--keypoint-run NAME` — keypoint run for eye center hints
- `--roi-index N` — start on a specific ROI

Saved to `analysis_metadata.attrs["eye_mask_tuning"]`.

---

## Subject body mask tuning

Controls traditional body segmentation: background subtraction threshold,
morphology (closing/opening radii), and minimum area filtering.

```bash
scripts/py -m fisheye.tune.subject_mask_tuner \
  /path/to/zarr/..._analysis.zarr \
  --component subject_body
```

The tuner shows the crop ROI with a live preview of the mask using your
current parameters. Adjust until the body is cleanly segmented, then save.

Saved to `analysis_metadata.attrs["subject_mask_tuning"].components["subject_body"]`.

### Propagate by camera

Tuning for the same camera usually transfers well. Copy it to all recordings
from the same camera:

```bash
# Dry-run
scripts/py -m fisheye.utils.apply_tuning_by_camera /nvme1/recordings \
  --source /path/to/tuned/zarr/..._analysis.zarr \
  --recursive \
  --keys subject_mask_tuning \
  --merge-dicts

# Apply
scripts/py -m fisheye.utils.apply_tuning_by_camera /nvme1/recordings \
  --source /path/to/tuned/zarr/..._analysis.zarr \
  --recursive \
  --keys subject_mask_tuning \
  --merge-dicts \
  --apply
```

`--merge-dicts` is important — it merges the tuning into the target instead of
replacing the whole attribute, so you won't clobber unrelated component tuning
(like swim bladder) that may already exist on the target.

To propagate only a specific component:

```bash
scripts/py -m fisheye.utils.apply_tuning_by_camera /nvme1/recordings \
  --source /path/to/tuned/zarr/..._analysis.zarr \
  --recursive \
  --keys subject_mask_tuning \
  --subject-mask-components subject_body \
  --merge-dicts \
  --apply
```

### Materialize the masks

After tuning is propagated, run segmentation to produce actual mask arrays:

```bash
scripts/py -m fisheye.segmentation.subject_segmentation \
  /path/to/zarr/..._analysis.zarr \
  --run-name traditional_subject_masks_001
```

Use `--overwrite` if re-running after re-tuning.

---

## Swim bladder tuning

The swim bladder has a dedicated tuner that works on a patch centered on the
swim bladder keypoint (so keypoints must exist first).

```bash
scripts/py -m fisheye.tune.swim_bladder_mask_tuner \
  /path/to/zarr/..._analysis.zarr
```

Two method families are available:
- **threshold** (default): blob threshold + morphology on a local patch
- **polar_boundary**: radial gradient analysis, better for ring-like boundaries

```bash
# Use polar boundary method
scripts/py -m fisheye.tune.swim_bladder_mask_tuner \
  /path/to/zarr/..._analysis.zarr \
  --method-family polar_boundary
```

Saved to `analysis_metadata.attrs["subject_mask_tuning"].components["swim_bladder"]`.

### Propagate and materialize

Same propagation workflow as subject body:

```bash
scripts/py -m fisheye.utils.apply_tuning_by_camera /nvme1/recordings \
  --source /path/to/tuned/zarr/..._analysis.zarr \
  --recursive \
  --keys subject_mask_tuning \
  --subject-mask-components swim_bladder \
  --merge-dicts \
  --apply
```

Then materialize in batch:

```bash
# Dry-run
scripts/py -m fisheye.utils.run_swim_bladder_segmentation_batch /nvme1/recordings \
  --recursive

# Apply
scripts/py -m fisheye.utils.run_swim_bladder_segmentation_batch /nvme1/recordings \
  --recursive \
  --apply
```

---

## Reviewing and refining masks

After materialization, review and correct masks with the interactive editor.

### Subject mask review (body + swim bladder)

```bash
scripts/py -m fisheye.tune.refined_subject_mask_review \
  /path/to/zarr/..._analysis.zarr \
  --subject-run traditional_subject_masks_001 \
  --components subject_body swim_bladder
```

This opens a paint/erase GUI where you can edit masks per ROI. Hotkeys:
- **s** — save current ROI edits
- **a** — approve the current component
- **Space** — next ROI
- **[** / **]** — cycle between components

Edits go to `refined_subject_masks_runs/`.

### Eye component review

Standalone eye-mask review has been retired. Edit eyes through the refined
subject-mask review surface:

```bash
scripts/py -m fisheye.tune.refined_subject_mask_review \
  /path/to/zarr/..._analysis.zarr \
  --components eye_left eye_right
```

### Inspecting masks without editing

To compare raw vs refined masks without entering edit mode:

```bash
scripts/py -m fisheye.visualization.subject_mask_inspector \
  /path/to/zarr/..._analysis.zarr \
  --subject-run traditional_subject_masks_001 \
  --refined-run refined_subject_masks_001 \
  --component subject_body
```

Shows contours, convex hulls, and QC metrics side by side.

---

## Checking tuning status

To see which recordings have which tuning and pipeline steps completed:

```bash
scripts/py -m fisheye.utils.check_recording_steps /nvme1/recordings --recursive
```

This prints a summary table showing, per recording:
- dish mask, detection tuning, keypoint tuning, subject mask tuning (with
  per-component breakdown), eye mask tuning, sub-arena masks
- which pipeline stages have run (detect, crop, keypoints, masks, etc.)

---

## Typical workflow for a new camera

When you get recordings from a camera you haven't processed before:

1. **Dish mask** — tune one recording, batch apply by chamber, review
2. **Background** — compute in batch
3. **Detection + crop** — run the pipeline (or traditional blob detector with
   tuned thresholds)
4. **Keypoints** — tune if using traditional method, then run
5. **Subject body** — tune, propagate by camera, materialize, review
6. **Swim bladder** — tune, propagate by camera, materialize, review
7. **Eye masks** — tune, run segmentation, review

After the first recording from a camera is tuned, subsequent recordings from
the same camera only need propagation + review — no manual tuning.
