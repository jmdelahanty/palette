# Paintera Native Palette Editing And SAM3 Subject-Mask Canary

## Purpose

Capture the current state of the `palette` <-> `paintera` workflow for:

- opening Palette training Zarr stores directly in Paintera,
- editing binary subject/eye masks against ROI crops, and
- generating new `subject_mask_runs` with SAM3 from refined keypoints.

This note records what was verified on a real canary archive rather than what
is only planned.

## Canary Archive

Primary archive:

- `/nvme1/recordings/2026-01-28T22-15-03Z_arena_1_DefaultScreen/zarr/2026-01-28T22-15-03Z_arena_1_DefaultScreen_training.zarr`

Working copy used for destructive tests:

- `/tmp/2026-01-28T22-15-03Z_arena_1_DefaultScreen_training.test.zarr`
- `/tmp/2026-01-28T22-15-03Z_arena_1_DefaultScreen_training.sam3.zarr`

Relevant runs present in the canary archive:

- crop run:
  - `crop_runs/crop_2026-02-03_23-32-21`
- refined subject-mask run:
  - `refined_subject_masks_runs/refined_subject_masks_2026-03-11_07-07-55`
- refined eye-mask run:
  - `refined_eye_masks_runs/refined_eye_masks_2026-02-12_19-51-24`
- refined 5-keypoint run:
  - `refined_keypoints_runs/refined_keypoints_traditional_v2_seed_001`

## What Exists In The Paintera Repo

The `paintera` repo now has a Palette-specific native backend for Palette
training Zarr stores.

Main integration points:

- `src/main/kotlin/org/janelia/saalfeldlab/util/n5/palette/PaletteTrainingN5.kt`
  - opens a Palette training store as a synthetic N5/Paintera container
  - exposes ROI raw images and mask channels as editable virtual datasets
  - writes label edits back into the underlying `masks_roi` channel stacks
- `src/main/kotlin/org/janelia/saalfeldlab/util/n5/zarrv3/ZarrV3ShimKeyValueAccess.kt`
  - provides the Zarr v3 key/value shim needed for local Palette stores
- `src/main/kotlin/org/janelia/saalfeldlab/util/n5/universe/N5FactoryWithCache.kt`
  - recognizes `zarr.json`-style containers
- `src/main/kotlin/org/janelia/saalfeldlab/util/n5/palette/PaletteLinkedRawSupport.kt`
  - resolves the matching raw ROI underlay for a synthetic label dataset
- `src/main/java/org/janelia/saalfeldlab/paintera/PainteraCommandLineArgs.java`
- `src/main/java/org/janelia/saalfeldlab/paintera/ui/dialogs/open/menu/n5/N5OpenSourceHelper.java`
- `src/main/kotlin/org/janelia/saalfeldlab/paintera/ui/dialogs/open/OpenSourceState.kt`
  - ensure the linked ROI raw source is opened together with the label source

Bridge tooling still exists in `paintera` and remains useful as a fallback:

- `tools/palette_zarr_bridge.py`
- `tools/launch_palette_bridge_in_paintera.sh`
- `docs/palette_zarr_bridge.md`

The bridge is no longer the only path, but it is still useful for debugging or
legacy workflows.

## Native Dataset Mapping In Paintera

Palette training data is exposed as synthetic datasets inside Paintera.

Patterns that are expected to work:

- raw ROI crops:
  - `raw/<crop_run>`
- subject-mask runs:
  - `labels/subject_mask/<run>/subject_body`
  - `labels/subject_mask/<run>/eyes_union`
  - `labels/subject_mask/<run>/swim_bladder`
- refined subject masks:
  - `labels/refined_subject_masks/<run>/subject_body`
  - `labels/refined_subject_masks/<run>/swim_bladder`
- refined eye masks:
  - `labels/refined_eye_masks/<run>/eye_left`
  - `labels/refined_eye_masks/<run>/eye_right`

For Palette-backed label datasets, Paintera also carries a `sourceRawPath`
attribute so opening the label source automatically opens the matching ROI raw
underlay.

## Verified Paintera Workflow

### Launch

Paintera requires JDK 21 and Maven in the shell used for launch.

Example launch against a copied training archive:

```bash
cd ~/gitrepos/paintera

DATASET='labels/refined_subject_masks/refined_subject_masks_2026-03-11_07-07-55/subject_body'
STORE='/tmp/2026-01-28T22-15-03Z_arena_1_DefaultScreen_training.test.zarr'

MAVEN_OPTS="-Djava.io.tmpdir=/tmp -Djna.tmpdir=/tmp" \
mvn javafx:run \
  "-Dpaintera.commandline.args=--add-n5-container=${STORE} -d ${DATASET}"
```

Verified behavior:

- the label source opens,
- the matching ROI raw image also opens automatically,
- `maxId=1` is provided automatically for these binary mask datasets.

### Editing Behavior

What worked in practice:

- `M` maximizes the active 2D viewer and is the best approximation to a 2D-only
  mode
- disabling `Settings -> Navigation -> Rotations` makes slice editing much less
  awkward
- increasing label alpha in the source panel makes committed paint visible
  enough for binary-mask cleanup

Paint semantics:

- `Space` + left drag: paint
- `Space` + right drag: erase from the temporary canvas only
- `Shift` + `Space` + right drag: erase already committed label back to
  background

Save semantics:

- releasing `Space` submits paint into Paintera's in-memory canvas
- `Ctrl+C` then `Commit` writes the changed mask blocks back to the underlying
  Palette Zarr store
- `Ctrl+S` saves Paintera project/UI state only

Important distinction:

- mask edits go back to the training Zarr
- viewer settings, layout, rotation toggle, and other UI state do **not** go
  back to the training Zarr

### Binary Writeback Contract

Palette-backed Paintera mask datasets are currently binary.

That means:

- any nonzero painted label is written back as `1`
- erased/background pixels are written back as `0`

This is correct for current body/eye mask editing, but it is not a general
multi-instance labeling path.

## Current Paintera Limitations

The current native backend is useful, but it is not feature-complete.

Known limits:

- it writes mask pixels, not full Palette review provenance
- it does **not** currently update sibling metadata such as:
  - `edit_applied`
  - `reason` / `reason_bytes`
  - `summary_statistics.postprocess.*`
  - `*_review_status`
- it is binary-mask oriented rather than arbitrary integer label editing
- Paintera still creates the 3D view; `M` is the practical workaround rather
  than a true 2D-only mode
- UI/project settings are stored in the Paintera project directory, not in the
  Palette training archive

So the current contract is:

- useful for direct pixel editing of existing mask channels
- not yet sufficient to produce fully Palette-native review provenance on its
  own

### Refined-Subject Metadata Saveback Hook

Palette now provides a thin helper executable for Paintera refined-subject
saveback:

- `/home/delahantyj@hhmi.org/gitrepos/palette/scripts/sync_refined_subject_mask_metadata`

It wraps:

- [sync_refined_subject_mask_metadata.py](/home/delahantyj@hhmi.org/gitrepos/palette/src/fisheye/utils/sync_refined_subject_mask_metadata.py)

Paintera can point its external saveback hook at that executable via:

- Java property:
  - `paintera.palette.refinedSubjectMetadataSync.executable`
- environment variable:
  - `PAINTERA_PALETTE_REFINED_SUBJECT_METADATA_SYNC_EXECUTABLE`

Current scope:

- refined subject-mask runs only
- component-scoped sync using the existing Palette refined-subject review
  semantics
- no new run-level `reason` / `reason_bytes` reconciliation beyond what the
  current Palette review code already owns

## SAM3 Canary Results

The same canary archive was copied to:

- `/tmp/2026-01-28T22-15-03Z_arena_1_DefaultScreen_training.sam3.zarr`

### Keypoint Source

`fisheye.utils.run_sam_subject_masks` prefers refined keypoints when
`--keypoint-group auto` is used. For the canary, the explicit run used was:

- `refined_keypoints_runs/refined_keypoints_traditional_v2_seed_001`

This run carries five keypoints:

- `swim_bladder`
- `eye_left`
- `eye_right`
- `snout_tip`
- `tail_tip`

### Inspect Summary

This command was used first:

```bash
~/gitrepos/palette/scripts/py -m fisheye.utils.run_sam_subject_masks \
  /tmp/2026-01-28T22-15-03Z_arena_1_DefaultScreen_training.sam3.zarr \
  --keypoint-group refined_keypoints_runs \
  --keypoint-run refined_keypoints_traditional_v2_seed_001 \
  --output-run sam_subject_masks_from_refined_keypoints_traditional_v2_seed_001 \
  --json
```

Observed summary:

- alignment status: `ok`
- eligible rows: `227`
- failed prompt rows: `0`
- positive prompt count per row: `5`
- SAM3 runtime available: `true`

### Box + Points Run

Command:

```bash
~/gitrepos/palette/scripts/py -m fisheye.utils.run_sam_subject_masks \
  /tmp/2026-01-28T22-15-03Z_arena_1_DefaultScreen_training.sam3.zarr \
  --keypoint-group refined_keypoints_runs \
  --keypoint-run refined_keypoints_traditional_v2_seed_001 \
  --output-run sam_subject_masks_from_refined_keypoints_traditional_v2_seed_001 \
  --sam3-root ~/gitrepos/sam3 \
  --apply
```

Observed result:

- segmented `227/227`
- nonempty `227`
- duration about `33s`
- device `cuda`

This run uses:

- five positive keypoints
- plus the detect-derived box prompt

### Points-Only Run

Command:

```bash
~/gitrepos/palette/scripts/py -m fisheye.utils.run_sam_subject_masks \
  /tmp/2026-01-28T22-15-03Z_arena_1_DefaultScreen_training.sam3.zarr \
  --keypoint-group refined_keypoints_runs \
  --keypoint-run refined_keypoints_traditional_v2_seed_001 \
  --output-run sam_subject_masks_points_only_from_refined_keypoints_traditional_v2_seed_001 \
  --sam3-root ~/gitrepos/sam3 \
  --no-box-prompt \
  --apply
```

Observed result:

- segmented `227/227`
- nonempty `227`
- duration about `33s`
- device `cuda`

Subjective canary note:

- the points-only run worked unexpectedly well and is a strong baseline worth
  comparing against the detect-box-assisted run

### Output Contract

The SAM3 subject-mask runs were written under:

- `subject_mask_runs/sam_subject_masks_from_refined_keypoints_traditional_v2_seed_001`
- `subject_mask_runs/sam_subject_masks_points_only_from_refined_keypoints_traditional_v2_seed_001`

These runs follow the current Palette subject-mask convention:

- `label_schema_id = "subject_v1_union"`
- `mask_labels = ["subject_body", "eyes_union", "swim_bladder"]`
- `available_channels = [true, false, false]`
- `masks_roi.shape = (227, 3, 512, 512)`

Only `subject_body` is populated in these phase-1 SAM3 runs.

## Useful Palette Viewer Commands

### Inspect SAM prompts and stored subject-body masks

```bash
~/gitrepos/palette/scripts/py -m fisheye.visualization.visualize_sam_subject_prompts \
  /tmp/2026-01-28T22-15-03Z_arena_1_DefaultScreen_training.sam3.zarr \
  --keypoint-group refined_keypoints_runs \
  --keypoint-run refined_keypoints_traditional_v2_seed_001 \
  --subject-run sam_subject_masks_from_refined_keypoints_traditional_v2_seed_001
```

Useful keys:

- `n` / `p`: next / previous ROI
- `j` / `k`: jump `-10` / `+10`
- `q` or `Esc`: quit

### Open a subject-mask run in the Palette review UI

```bash
~/gitrepos/palette/scripts/py -m fisheye.tune.refined_subject_mask_review \
  /tmp/2026-01-28T22-15-03Z_arena_1_DefaultScreen_training.sam3.zarr \
  --subject-run sam_subject_masks_from_refined_keypoints_traditional_v2_seed_001
```

## Recommended Next Steps

Short-term:

- compare the detect-box-assisted and points-only SAM3 runs quantitatively
- decide whether points-only should be the default prompt policy for the next
  canary
- add a Palette-native provenance sync step for Paintera edits

Paintera-side:

- update Palette review metadata when pixels are committed
- keep the current native path focused on binary mask cleanup

Palette-side:

- add a compact compare utility for subject-mask runs if repeated SAM3 canaries
  become common
- decide whether Paintera-origin edits need a distinct provenance tag such as
  `paintera_edit`
