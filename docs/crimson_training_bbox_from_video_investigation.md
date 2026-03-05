# Investigation: Bounding Box Training Labels from Crimson

Date: 2026-03-05

## Goal

Allow users to create bounding box training labels directly from video in Crimson,
without requiring an existing palette detection pipeline run, and have those labels
be consumable by palette's YOLO detection training pipeline.

## Current State

### What Crimson already has

**Bbox editing primitives** (all implemented):
- Select box by click, move by drag, draw new box (`N` mode + click-drag),
  delete selected box (`Del`)
- Per-frame in-memory dirty-state tracking with frame overrides
- Visual feedback: `[S]` selected, `[M]` modified, `[A]` added

Source: `crimson/src/zarr_bbox_edit.h:14` (`ZarrBBoxEditState` struct),
`crimson/src/zarr_bbox_edit.cpp` (full implementation).

**Detection data loading** from all palette run types:
- `detect_runs/<run>` (raw)
- `refined_detect_runs/<run>/{filtered,interpolated,manual}` (refined)
- Fallback resolution chain per palette spec

Source: `crimson/src/zarr_loader_detections.cpp`,
`crimson/src/zarr_loader.h:88` (`ZarrDetectionData` struct, fields at lines 88-144).

**Zarr writeback** to refined manual subgroup:
- `writeManualRefinedDetections()` at `crimson/src/zarr_loader.h:738-750`
- Writes arrays: `frame_indices`, `bbox_norm_coords`, `scores`, `class_ids`,
  `frame_counts`, `n_detections`, `frame_mapping`, `detection_source`,
  `reason_bytes`, `retune_id`
- Implementation: `crimson/src/zarr_loader_write.cpp:6-435`
- UI trigger: "Write Manual Payload to Zarr" button at `crimson/src/red.cpp:2642`
- Write target: `refined_detect_runs/<latest>/manual`
  (see `crimson/src/red.cpp:2718`)

**Write contract**:
`crimson/docs/crimson_refined_detect_manual_contract.md` (full spec).

### What palette already has

**Detection training dataset loader**:
- `palette/src/fisheye/training/zarr_yolo_dataset_loader.py`
- `SingleDatasetConfig` at line 38: `source_type` accepts `'detect'`,
  `'filtered'`, `'interpolated'`, or `'manual'`
- Resolution logic uses `resolve_refined_detect_group()` from
  `palette/src/fisheye/shared/refined_detect_review.py:28`
- Preference chain: `("manual", "interpolated", "filtered", "raw")` (line 9)
- `__getitem__` loads: frame image from `raw_video/images_ds`, bbox from
  `bbox_norm_coords`, returns `[cx, cy, w, h]` normalized tensors

**Training manifest builder**:
- `palette/src/fisheye/diagnostics/prepare_detect_training.py`
- Resolves bbox arrays from `crop_runs`, `detect_runs`, or
  `refined_detect_runs` (lines 312, 962-984)
- Validates bbox integrity (NaN, out-of-range, invalid dimensions)
- Outputs JSON manifest listing each dataset's `bbox_array_path` and
  `detection_source_type`

**Training data export**:
- `palette/src/fisheye/utils/export_detect_training_zarr.py`
- Copies selected arrays (frames + bboxes) into standalone training zarrs

**Training config** (`palette/configs/fisheye/detect_config.yaml`):
- Per-dataset `source_type`, `input_format`, split ratios
- Sampling strategy: balanced / proportional / weighted
- Augmentation config

**Required bbox format** (shared across all palette consumers):
- `bbox_norm_coords`: `float`, shape `(n_detections, 4)`,
  values `[center_x, center_y, width, height]` normalized to `[0, 1]`
- `frame_indices`: `int32`, shape `(n_detections,)`
- `frame_counts`: `int32`, shape `(n_frames,)`
- `scores`: `float32`, shape `(n_detections,)`
- `class_ids`: `int32`, shape `(n_detections,)`

This is the same format Crimson already writes for manual refined detections.

### Existing design document

`crimson/docs/crimson_video_bbox_training_labeling_plan.md` (dated 2026-03-05)
outlines an earlier version of this vision. That document proposed a separate
`training_bbox_label_runs/` namespace and a separate "Training Label Mode" UI.

This investigation supersedes that plan on two points:
- **Namespace**: renamed to `manual_training_labels` (clearer name).
- **UI approach**: no separate mode. The editing UI is unified; provenance is
  carried by run-level attrs (`source_detect_run`, `label_status`, etc.) and
  palette routes based on metadata rather than namespace conventions alone.

## The Gap

### 1. No "start from scratch" path

Crimson's detection loader assumes `detect_runs` or `refined_detect_runs` already
exist. Opening a video with no prior detections provides no bbox editing canvas.

Relevant code: `crimson/src/zarr_loader.h:730-736` (`hasDetectionData()` checks
for existing data before enabling detection display).

### 2. Write target requires existing refined detect run

`writeManualRefinedDetections()` writes to `refined_detect_runs/<latest>/manual`.
It assumes a refined run already exists and looks up `refined_detect_runs.attrs["latest"]`
to find the write target.

Source: `crimson/src/zarr_loader_write.cpp:66-74` (manual group resolution),
`crimson/src/zarr_loader_write.cpp:404` (refined run path construction).

### 3. Training labels conflated with review edits

The existing manual write path serves review/refinement workflows. Using it for
training annotation would mix ground-truth labels with operational review edits
in the same namespace, making provenance ambiguous.

### 4. Palette has no `manual_training_labels` source type

The training dataset loader (`zarr_yolo_dataset_loader.py:41`) recognizes
`detect`, `filtered`, `interpolated`, and `manual` — but not a dedicated
training-label source. The manifest builder (`prepare_detect_training.py`)
similarly resolves from `crop_runs`, `detect_runs`, and `refined_detect_runs`
only (lines 962-984).

### 5. No class ontology or frame-status workflow

Crimson has no class picker / hotkey UI, no frame-status state machine
(unlabeled / partial / complete / skip), and no train/val/test split assignment.
These are needed for systematic dataset curation.

### 6. No direct export to YOLO/COCO from Crimson

`crimson/data_exporter/red3d2yolo.py` exists but exports from the legacy Red3D
HDF5 format, not from the Zarr training label namespace.

## Design Decision: Unified UI, Metadata-Driven Routing

A key design choice is to **not** split Crimson into separate "review mode" and
"training label mode" UIs. The editing interaction is identical in both cases —
draw, move, delete boxes. What differs is provenance and intent, and those are
properties of the data, not the interface.

The `manual_training_labels` namespace carries run-level attributes that tell
downstream consumers (palette) everything they need:

- `source_detect_run`: the model run labels were seeded from, or **null** if
  authored from scratch on raw video
- `label_status`: `"in_progress"` | `"complete"` | `"locked"`
- `class_map`: ontology reference (e.g. `{0: "fish"}`)
- `annotator_id`, `created_at`
- `source_video_path`

Palette's dataset loader and manifest builder read these attrs to decide how
to consume the data — no mode flag needed from the user side.

This means a single save action in Crimson writes to `manual_training_labels/`
regardless of whether the user started from scratch or from pre-existing model
detections. The `source_detect_run` attr records the difference.

## What Needs to Change

### Crimson changes (C++)

**A. New Zarr namespace: `manual_training_labels/<run_name>/`**

Arrays (identical schema to what palette already consumes):
- `frame_indices` (int32)
- `bbox_norm_coords` (float64, shape `(n, 4)`, normalized `[cx, cy, w, h]`)
- `class_ids` (int32)
- `scores` (float32, default 1.0 for human labels)
- `frame_counts` (int32, shape `(n_frames,)`)

Run attrs:
- `source_detect_run`: str or null (null = authored from scratch)
- `label_status`: `"in_progress"` | `"complete"` | `"locked"`
- `class_map`: ontology reference (e.g. `{0: "fish"}`)
- `source_video_path`: path to source video
- `annotator_id`, `created_at`, `label_schema_version`

Files to modify:
- `crimson/src/zarr_loader.h` — new struct or fields for training label runs
- `crimson/src/zarr_loader.cpp` — read path for training label runs
- `crimson/src/zarr_loader_write.cpp` — write path (parallel to existing
  `writeManualRefinedDetections` but targeting new namespace, without
  requiring a pre-existing refined detect run)

**B. Bootstrap from video without existing detections**

Allow `ZarrBBoxEditState` to initialize with an empty detection set when
no `detect_runs` exist. Frame count comes from video metadata (fps + duration)
or from `raw_video/images_ds` shape.

Files to modify:
- `crimson/src/zarr_loader.h` — `hasDetectionData()` or new
  `hasLabelCapability()` that returns true when video is loaded
- `crimson/src/red.cpp` — enable bbox editing canvas even without detections

**C. Unified save path**

The existing "Write Manual Payload to Zarr" button (`crimson/src/red.cpp:2642`)
gains a save target selector: write to `refined_detect_runs/.../manual`
(existing review workflow) or to `manual_training_labels/<run>/` (training
labels). The UI is otherwise the same — no mode switch, just a destination
choice. Crimson auto-populates `source_detect_run` based on whether
detections were loaded from a model run or the canvas started empty.

**D. Export helper** (can be Python script initially)

Update or add a script in `crimson/tools/` or `crimson/data_exporter/` that
reads `manual_training_labels/<run>/` and exports to YOLO format.

### Palette changes (Python)

**E. Add `manual_training_labels` source type to dataset loader**

In `zarr_yolo_dataset_loader.py`:
- Add `'manual_training_labels'` to `SingleDatasetConfig.source_type` (line 41)
- Add resolution logic: when `source_type == 'manual_training_labels'`, look up
  `manual_training_labels/<latest>/` instead of `detect_runs` or
  `refined_detect_runs`
- Palette can also auto-detect: if `manual_training_labels/` exists in a zarr
  and the run's `label_status` is `"complete"` or `"locked"`, prefer it
  over model-generated detections when building training sets

**F. Add resolution to manifest builder**

In `prepare_detect_training.py`:
- Add `manual_training_labels` as a recognized group alongside
  `detect_runs`, `refined_detect_runs`, `crop_runs` (around lines 962-984)
- Validate arrays using existing bbox integrity checks
- Read `source_detect_run` attr to record provenance in the manifest

**G. Document new namespace in zarr structure**

Update `palette/src/fisheye/docs/zarr_structure.md` and
`crimson/tools/palette_zarr_layout_v3` to include `manual_training_labels`.

### Shared contract docs (new)

- `crimson_manual_training_labels_write_contract.md` — what Crimson writes
- `crimson_manual_training_labels_read_contract.md` — what Crimson reads back
- Palette-side: update `detect_config.yaml` docs to show
  `source_type: manual_training_labels` option

## Minimum Viable Path

The fastest route to "label boxes in Crimson, train in Palette":

1. **Crimson**: Add a `writeManualTrainingLabels()` function that writes to
   `manual_training_labels/<run>/` using the same array format as
   `writeManualRefinedDetections()` but without requiring a pre-existing
   refined detect run. Wire the existing bbox-edit save action to target
   this namespace, with `source_detect_run` set to null or to the loaded
   detection run name depending on context.

2. **Crimson**: Allow the bbox edit canvas to initialize empty when no
   detections exist (bootstrap from video frame count).

3. **Palette**: Add `source_type: "manual_training_labels"` to
   `SingleDatasetConfig` and teach the dataset loader + manifest builder
   to resolve arrays from `manual_training_labels/`.

4. **Config**: Point `detect_config.yaml` at the labeled zarr with
   `source_type: manual_training_labels`.

This skips class picker UI, frame-status state machine, and YOLO export
tooling — those can be layered on incrementally.

## Format Compatibility

The bbox array format is already identical between what Crimson writes and
what palette's training loader consumes:

| Field | Crimson write type | Palette read type | Match? |
|---|---|---|---|
| `bbox_norm_coords` | float64, `[cx,cy,w,h]` normalized | float (any), `[cx,cy,w,h]` normalized | Yes |
| `frame_indices` | int32 | int32 | Yes |
| `frame_counts` | int32 | int32 | Yes |
| `scores` | float32 | float32 | Yes |
| `class_ids` | int32 | int32 | Yes |

No format translation is needed — only a new namespace and resolution path.

## Risks

- **Coordinate space**: Crimson normalizes to video frame dimensions. Palette
  normalizes to inference space (which may differ if video was resized for
  detection). For training labels authored directly from video, the
  normalization reference is the raw video frame, which must match the
  training image dimensions. This should be validated at export time.

- **Frame indexing**: Training labels index into raw video frames. The palette
  loader currently expects frame indices to align with `raw_video/images_ds`
  or `crop_runs/.../roi_images`. Labels from Crimson must use the same
  frame indexing as the video's downsampled image array.

- **Zarr version**: Crimson writes Zarr v3 via TensorStore. Palette reads
  via the `zarr` Python library (primarily v2 API with v3 compatibility).
  The existing refined-detect write path already works across this boundary,
  so the new namespace should too, but this should be verified.

- **Save destination ambiguity**: With a unified UI (no mode switch), the user
  must choose where to save — `refined_detect_runs/.../manual` (review) or
  `manual_training_labels/<run>/` (training). This could be a simple dropdown
  or radio next to the save button. The risk is a user accidentally saving
  training labels to the review namespace or vice versa. Mitigation: default
  to `manual_training_labels` when no refined detect run exists (from-scratch
  case), default to `refined_detect_runs` when editing loaded model detections,
  and always show the active save target in the UI.
