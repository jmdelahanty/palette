# Analysis-to-Training Promotion Contract

<!-- contract-meta
status: active
last_updated: 2026-05-20
scope: detect bbox promotion from analysis zarr to per-recording training zarr
-->

## Goal

Support operator-driven promotion of hand-labeled detection examples from a
recording analysis Zarr into that recording's training Zarr.

This is intended for workflows where Crimson or Palette lets an operator inspect
an analysis recording, add or correct bounding boxes, and immediately preserve
those supervised examples in the per-recording training archive.

## Current Position

As of 2026-05-20, the practical review/edit split is:

- Crimson remains the preferred full-stack desktop path for high-performance
  analysis-video inspection.
- Palette now has two browser review surfaces:
  - `detect_review_web` for materialized image archives such as training Zarrs
    with `raw_video/images_*`;
  - `video_detect_review_web` for analysis-video detection review, including
    clipped finalized collections when launched with the resolver inputs and,
    preferably, a review-proxy manifest.
- Promotion is implemented for detection bounding boxes, including clipped
  finalized-collection sources. The remaining integration work is exporter
  adoption, Crimson save-hook wiring, and performance/telemetry hardening for
  online batch saves.

The current Palette-side promotion contract is: after an operator edits or
confirms analysis labels, Palette can dry-run and apply an upsert into the
recording's per-recording training Zarr. Review tools may call the same backend
online after save.

The preferred future operator workflow is automatic promotion on save: Crimson
or another review UI saves the analysis-Zarr edit first, then invokes the
Palette promotion backend for that edited frame and reports the result to the
operator. The standalone promotion CLI should remain the same backend path used
for dry-runs, backfills, audits, repairs, and batch promotion after older review
sessions.

Current implementation status:

- Implemented backend: `fisheye.tune.detect_training_promotion_backend`.
- Implemented CLI: `fisheye.utils.promote_analysis_detect_to_training`.
- Implemented backend scope: non-clipped top-level refined detections plus
  clipped finalized-collection refined detections when the caller supplies the
  resolved clip context.
- Implemented CLI scope: non-clipped top-level refined detections and clipped
  finalized-collection backfill from the resolved recording frame map.
- Implemented post-save hook: `fisheye.tune.video_detect_review_web` can
  promote traditional and clipped analysis saves when launched with
  `--edit --promote-training-zarr`.
- Implemented online batch behavior: `video_detect_review_web` groups analysis
  writes and calls the promotion backend once per save batch; clipped promotion
  groups appended frames by source clip/video for a single decode pass per clip.
- Not yet wired: automatic promotion from Crimson save.
- Implemented exporter contract: promoted positive detection rows are mirrored
  into `refined_detect_runs/<run>/instances`, so normal refined-source training
  exports can consume promoted labels without a crop-run-specific manifest
  override.

Therefore, saving an edited box in the current review UI writes the analysis
Zarr only unless the server was explicitly launched with the promotion hook.
With `--promote-training-zarr`, the save path writes the analysis edit first,
then promotes that same source frame into the recording's training Zarr. The
web UI can buffer edits across multiple frames and submit them as one batch;
the analysis write is grouped by refined-detect output group, then the
promotion backend is called once for the saved batch. If promotion fails, the
analysis edit remains valid and the response reports the promotion failure
separately.

For clipped analysis sessions, promotion uses the parent-frame index as the
training-row identity and decodes image data from the real source clip at the
resolved clip-local frame. If a review proxy video is active, it is used only
for browser display and never as the promoted training image source.

Online save telemetry is intentionally split by stage. The `/api/frames/save_batch`
response and server log include the total batch time, analysis-Zarr write time,
promotion time, clipped decode group count/time, image-array write time, and
promotion metadata write time. This is the primary diagnostic for distinguishing
slow source-video decode from slow training-Zarr writes.

Example video-review server with post-save promotion:

```bash
scripts/py -m fisheye.tune.video_detect_review_web \
  <recording>/zarr/<recording>_analysis.zarr \
  --edit \
  --promote-training-zarr <recording>/zarr/<recording>_training.zarr \
  --host 0.0.0.0 \
  --port 8790
```

To promote saved frames outside a review session, run the promotion CLI
explicitly.

Example dry-run:

```bash
scripts/py -m fisheye.utils.promote_analysis_detect_to_training \
  <recording>/zarr/<recording>_analysis.zarr \
  --frames 12345
```

Example apply:

```bash
scripts/py -m fisheye.utils.promote_analysis_detect_to_training \
  <recording>/zarr/<recording>_analysis.zarr \
  --frames 12345 \
  --apply
```

For migrated/clipped recordings, prefer resolving the target from the registry:

```bash
scripts/py -m fisheye.utils.promote_analysis_detect_to_training \
  <recording>/zarr/<recording>_analysis.zarr \
  --use-registry-target \
  --collection-id <finalized_collection_id> \
  --apply
```

This queries the registry for the analysis dataset's `recording_id`, then
requires exactly one active `zarr_use='training'` dataset for that recording.
The JSON result records `training_zarr_source: registry` and the matched
analysis/training dataset IDs. If the registry target is ambiguous or missing,
the command fails closed.

For standard recording layouts without a registry target, the CLI infers the
sibling target `<recording>_training.zarr` from `<recording>_analysis.zarr`.
For smoke runs, migration repairs, or nonstandard targets, pass the target
explicitly:

```bash
scripts/py -m fisheye.utils.promote_analysis_detect_to_training \
  <recording>/zarr/<recording>_analysis.zarr \
  --training-zarr <target_training.zarr> \
  --frames 12345 \
  --apply
```

Example clipped finalized-collection manual backfill:

```bash
scripts/py -m fisheye.utils.promote_analysis_detect_to_training \
  <recording>/zarr/<recording>_analysis.zarr \
  --use-registry-target \
  --collection-id <finalized_collection_id> \
  --apply
```

For clipped mode, omitting `--frames` means "discover manually edited frames
from the finalized collection." This is the default backfill mode. A manually
cleared frame is promoted as a `negative` row unless `--no-negative` is passed.

## Two-Level Dataset Model

Palette should keep two concepts separate:

1. Per-recording training Zarr
   - Mutable curated example store for one recording.
   - Receives immediate upserts from manual review/promotion.
   - Stores frame image data, bbox labels, label state, and source lineage.
   - Does not assign global train/val/test splits.

2. Unified/exported training artifact
   - Immutable model-training artifact built later from many per-recording
     training Zarrs.
   - Owns train/val/test split assignment.
   - Owns global dataset versioning, manifests, cards, and registry training-set
     records.

Promotion should mutate only the per-recording training Zarr. Retraining should
build a new unified/exported dataset artifact from selected per-recording
training Zarrs.

## Source of Truth

The analysis Zarr remains the editable source of truth for the reviewed
recording state.

For detection labels, the current source surface is:

- `refined_detect_runs/<run>/instances`

Crimson should render from the current refined-detect read contract and save
manual detection edits through the current refined-detect write/review contract.
Promotion then copies the selected final supervised frame state into the
recording training Zarr.

Related contracts:

- `docs/crimson_detect_bbox_read_contract.md`
- `docs/crimson_detect_review_acceptance_contract.md`
- `docs/detection_review_web_todo.md`
- `docs/training_label_origin_provenance_todo.md`
- `docs/training_dataset_versioning_todo.md`

## Promotion Unit

The minimum promotion unit is one final supervised frame for one recording
context.

For non-clipped recordings, identity should include:

- `source_analysis_zarr_path` or registry `analysis_dataset_id`
- `source_refined_detect_run`
- `source_frame_index`
- `source_entity_id` or instance identity, currently expected to be `0` for
  single-subject detection training

For clipped recordings, identity must additionally preserve:

- `source_clip_id`
- `source_clip_index`
- `source_camera_serial`
- `source_clip_local_frame_index`
- `source_parent_frame_index` or `recording_frame_id`

Promotion must be an upsert, not blind append. Re-promoting the same source
identity updates the existing training row.

## Label Semantics

Training rows should represent the current supervised truth, not the full edit
history.

Allowed per-frame label states:

- `positive`: one or more valid bboxes are exported with the image.
- `negative`: no valid bbox remains; this is an explicit negative frame.
- `inactive`: retained for audit but excluded from future exports.

For sparse refined-detection review, an already-empty frame's explicit label is
stored in the versioned sibling surface defined by
`docs/detect_frame_decision_storage_contract_v1.md`; it is not represented by a
fake row in `refined_detect_runs/<run>/instances`.

Policy:

- Moving or resizing a bbox produces/updates a `positive` row.
- Adding a bbox to a previously absent frame produces/updates a `positive` row.
- Clearing/deleting the final bbox produces/updates a `negative` row only when
  no valid bbox remains for that promotion identity.
- Do not separately encode "false positive then corrected" in the training
  label surface. If the operator moves a box, the final row is simply the
  corrected positive bbox.
- Edit history may be recorded in provenance/audit metadata, but training labels
  should remain final-state labels.

## Required Per-Recording Training Fields

The canonical detection label table is:

- `refined_detect_runs/<run>/instances`: sparse, frame-sorted positive
  detection instances. Promotion must upsert positive supervised labels here so
  normal refined-source exporters, registry quality checks, and downstream
  consumers have one authoritative label surface.

The materialized training-image/support surface is:

- `raw_video/images_ds`: materialized frame image used by training.
- `crop_runs/<run>/bbox_norm_coords`: normalized `[cx, cy, w, h]`, finite for
  `positive`, NaN or ignored for `negative`. This mirrors the canonical refined
  label state for per-recording crop/image consumers but is not the primary
  detection label authority and is not required in new merged training exports.
- `crop_runs/<run>/frame_indices`: row index into `raw_video/images_ds`, usually
  sequential in the per-recording training Zarr.
- `crop_runs/<run>/label_state`: enum/string for `positive | negative |
  inactive`.
- `crop_runs/<run>/label_origin`: e.g. `manual_training` or `manual_review`.
- `source_index/source_analysis_zarr_path` or equivalent source dataset id.
- `source_index/source_refined_detect_run`.
- `source_index/source_frame_idx`.
- `source_index/source_refined_row_ids` when available.
- `source_index/source_detect_row_index` when available.
- clipped mapping fields when the source is clipped.

The per-recording training Zarr should not create or update global
`splits/train_indices`, `splits/val_indices`, or `splits/test_indices`. Those
belong to the unified/exported training artifact.

## Image Representation

The promoted image must match the active detection-training representation for
the project.

Current preferred representation for mono Orange recordings:

- `uint8` luma/mono image data, compatible with downstream YOLO preprocessing.

The promotion result should record:

- image source path
- decode/copy method
- pixel representation contract
- source frame mapping

This allows later unified exports to avoid guessing whether promoted rows came
from legacy grayscale, PyNvVideoCodec luma, clipped frames, or another source.

## Upsert Behavior

Promotion should expose a dry-run first.

For each requested source identity, dry-run should report:

- `append`: no matching training row exists.
- `update`: matching row exists and will be changed.
- `no_change`: matching row already matches the source final label.
- `conflict`: multiple matching rows exist, source mapping is ambiguous, or the
  requested source cannot be resolved.
- `skip`: source state is not promotable under current policy.

Apply mode should fail closed on `conflict` unless an explicit repair option is
provided.

## Negative Examples

The sparse refined-instance table does not contain negative rows. Explicit
negative frames live in the bound `detect_frame_decision_runs` surface. The
unified exporter includes them only after the review scope is complete.

For YOLO detection export:

- `positive` rows produce normal bbox labels.
- `negative` frames produce image-only samples with empty labels.
- `inactive` rows are ignored.

The immutable export records positive/negative state for every sample. Later
model-training runs may deterministically subsample those already identified
negative frames without changing label authority.

## Registry Responsibilities

Initial implementation can work without registry writes, but should emit enough
metadata to support later registry synchronization.

Eventually, registry sync should be able to answer:

- which analysis dataset produced each promoted training row
- which per-recording training dataset contains promoted examples
- counts by `label_state` and `label_origin`
- whether the per-recording training Zarr has pending promoted edits not yet
  included in any unified/exported training artifact

## Implemented First Slice

Implemented in `fisheye.utils.promote_analysis_detect_to_training`:

- Dry-run CLI:
  - Input: analysis zarr, training zarr, refined run, frame list.
  - Output: `append` / `update` / `no_change` / `conflict` / `skip` plan.

- Apply mode:
  - Supports non-clipped, single-subject frame-axis refined detect rows through
    `promote_detection_frames`.
  - Supports clipped single-subject refined detect rows through
    `promote_clipped_detection_frames`.
  - The CLI can resolve clipped promotion frames from
    `experiment_index/finalized_runs/<collection_id>` plus
    `recording_frame_index.parquet`; it defaults to manual-only backfill and
    keeps manual clears as explicit negative examples.
  - Copies `raw_video/images_ds[source_frame]` into the per-recording training
    Zarr when appending.
  - Can decode a traditional source video through OpenCV when analysis
    `raw_video/images_ds` is absent and a source video path is available.
  - For clipped appends, decodes the real source clip at
    `clip_local_frame_index`, writes `raw_video/images_ds`, and writes
    `raw_video/images_full`. If the target training Zarr does not already have
    `images_full`, clipped promotion creates it from the decoded source-frame
    luma shape.
  - Clipped appends are grouped by source clip. The backend prefers one
    sequential PyNvVideoCodec luma decode pass per clip, falls back to OpenCV
    random access if PyNvVideoCodec is unavailable, resizes output image arrays
    once for the append batch, and keeps `raw_video/images_full` populated.
    This is intentional because later crop, keypoint, and segmentation
    workflows need a high-resolution source image without re-decoding the
    original video.
  - Updates bbox and label-state fields when updating.
  - Writes `source_index` lineage.
  - Mirrors all positive promoted/materialized rows into
    `refined_detect_runs/<run>/instances`, using the active refined run when one
    exists or `refined_detect_promoted_manual` for new per-recording training
    stores.
  - Invalidates stale inline Zarr v3 consolidated child metadata on mutated
    refined-detect groups so later readers see the repaired arrays.

- Validation behavior:
  - duplicate source identity rows become `conflict`;
  - positive rows require finite bbox;
  - negative rows store NaN bbox;
  - image row count must match active crop-run row count;
  - source lineage is written for promoted rows.

Still deferred:

- Crimson handoff:
  - Crimson saves refined detect edit in analysis Zarr.
  - Crimson invokes the same promotion backend for the edited frame.
  - Crimson displays promotion result.
- Region-level hard-negative boxes remain deferred. Frame-level reviewed
  negatives and multiple positive detections per frame are implemented by the
  frame-supervision export bridge. New merged detection exports continue to use
  canonical refined instances as the positive label surface and do not
  forward-write crop-run label mirrors.

## Automatic-On-Save Workflow

Automatic promotion should be implemented as an explicit post-save hook, not as
hidden mutation inside a bbox editor.

This is implemented for `video_detect_review_web` traditional and clipped
sessions when launched with `--edit --promote-training-zarr <training.zarr>`.
The same sequence remains the target for Crimson.

Expected sequence:

1. The review UI writes the corrected detection state to the analysis Zarr.
2. The review UI calls the Palette promotion backend for the edited source
   identity.
3. Promotion performs the same conflict checks as the standalone CLI.
4. The review UI reports one of `append`, `update`, `no_change`, `negative`,
   `skip`, or `conflict`.
5. If promotion fails, the analysis edit remains valid and the UI reports that
   the training Zarr was not updated.

This keeps analysis review authority separate from training-store mutation while
still giving operators the desired "save once, training row updated" behavior.
The standalone CLI remains required for non-interactive backfill, repair, and
auditing.

The implemented Palette web path is still deliberately narrower than the full
future workflow:

- detection bounding boxes only;
- one subject per frame;
- no registry writes required;
- clipped promotion requires either an explicit caller-provided frame context or
  a finalized collection plus recording frame map for CLI discovery.

## Non-Goals

- Do not mutate unified/exported training artifacts in place.
- Do not assign train/val/test splits during promotion.
- Do not encode full edit history as training labels.
- Do not require registry migration for the first slice.
- Do not infer clipped promotion identity from directory names; use the resolved
  collection/frame-map context.
