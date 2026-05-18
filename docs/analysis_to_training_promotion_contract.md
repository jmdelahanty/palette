# Analysis-to-Training Promotion Contract

<!-- contract-meta
status: draft
last_updated: 2026-05-18
scope: detect bbox promotion from analysis zarr to per-recording training zarr
-->

## Goal

Support operator-driven promotion of hand-labeled detection examples from a
recording analysis Zarr into that recording's training Zarr.

This is intended for workflows where Crimson or Palette lets an operator inspect
an analysis recording, add or correct bounding boxes, and immediately preserve
those supervised examples in the per-recording training archive.

## Current Position

As of 2026-05-18, the practical review/edit split is:

- Crimson is the preferred path for analysis-video inspection, especially for
  clipped recordings once its Parquet-backed resolver lands.
- Palette's current web review tools are best suited to materialized training
  Zarrs because they can read persisted `raw_video/images_*` arrays directly.
- A video-backed web reviewer for analysis Zarrs is possible, but it requires
  a separate decoder/resolver layer and is not the next Palette-side blocker.

The next Palette-side implementation slice is therefore promotion: after an
operator edits or confirms analysis labels, Palette should be able to dry-run
and then apply an upsert into the recording's per-recording training Zarr.

The preferred future operator workflow is automatic promotion on save: Crimson
or another review UI saves the analysis-Zarr edit first, then invokes the
Palette promotion backend for that edited frame and reports the result to the
operator. The standalone promotion CLI should remain the same backend path used
for dry-runs, backfills, audits, repairs, and batch promotion after older review
sessions.

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

Allowed per-row label states:

- `positive`: one valid bbox should be exported as a labeled detection example.
- `negative`: no valid bbox remains; this is an explicit negative frame.
- `inactive`: retained for audit but excluded from future exports.

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

The exact storage group can be finalized during implementation, but the promoted
detect examples need these logical fields:

- `raw_video/images_ds`: materialized frame image used by training.
- `crop_runs/<run>/bbox_norm_coords`: normalized `[cx, cy, w, h]`, finite for
  `positive`, NaN or ignored for `negative`.
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

Explicit negative rows are allowed in the per-recording training Zarr, but the
unified/exported training artifact decides whether to include them.

For YOLO detection export:

- `positive` rows produce normal bbox labels.
- `negative` rows may produce image-only samples with empty labels if the export
  configuration includes negatives.
- `inactive` rows are ignored.

This keeps manual curation flexible while letting each model-training run choose
its negative-sampling policy.

## Registry Responsibilities

Initial implementation can work without registry writes, but should emit enough
metadata to support later registry synchronization.

Eventually, registry sync should be able to answer:

- which analysis dataset produced each promoted training row
- which per-recording training dataset contains promoted examples
- counts by `label_state` and `label_origin`
- whether the per-recording training Zarr has pending promoted edits not yet
  included in any unified/exported training artifact

## Recommended First Implementation Slice

1. Write a dry-run CLI:
   - Input: analysis zarr, training zarr, refined run, frame list.
   - Output: append/update/no_change/conflict/skip plan.

2. Implement apply for detection bbox promotion:
   - Supports non-clipped, single-subject frame-axis refined detect rows.
   - Copies `raw_video/images_ds[source_frame]` into the per-recording training
     Zarr when appending.
   - Updates bbox and label-state fields when updating.
   - Writes source-index lineage.

3. Add validation:
   - no duplicate source identity rows
   - positive rows have finite bbox
   - negative rows have no finite bbox
   - image row count matches label row count
   - source lineage is complete

4. Add Crimson handoff:
   - Crimson saves refined detect edit in analysis Zarr.
   - Crimson invokes the same promotion backend for the edited frame.
   - Crimson displays promotion result.

## Future Automatic-On-Save Workflow

Automatic promotion should be implemented as an explicit post-save hook, not as
hidden mutation inside a bbox editor.

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

For the first implementation pass, keep this deliberately narrower than the
full clipped workflow:

- non-clipped or already materialized training Zarr targets only;
- detection bounding boxes only;
- one subject per frame;
- no registry writes required;
- `--dry-run` required for initial smoke validation before `--apply`.

Clipped analysis promotion should wait until the read-only Crimson clipped
resolver is working, because clipped apply mode must route source identity
through `(clip_id, camera_serial, clip_local_frame_index, refined_group_path)`
rather than a single top-level frame index.

## Non-Goals

- Do not mutate unified/exported training artifacts in place.
- Do not assign train/val/test splits during promotion.
- Do not encode full edit history as training labels.
- Do not require registry migration for the first slice.
- Do not attempt clipped multi-camera promotion until the non-clipped identity
  contract is tested.
