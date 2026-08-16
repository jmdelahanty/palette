# Refined Detect Row Identity Contract

<!-- contract-meta
version: 1
status: draft
implementation: implemented
last_updated: 2026-04-24
-->

## Purpose

This contract defines how Palette should interpret stable row identity for
current refined detection data. It is the detect/crop/keypoint counterpart to the
repo-wide stale policy: row-local stale repair is safe only when refined detect
row identity survives an edit.

## Canonical Surface

Current refined detection authority lives at:

- `refined_detect_runs/<run>/instances`

The run root carries review/status metadata. `source_detections/` is the
raw-candidate audit surface. Legacy sparse subgroups such as `manual`,
`interpolated`, and `filtered` remain readable for older archives, but they are
not the current operator-facing authority when `instances/` exists.

## Identity Fields

Required identity fields on `instances/`:

- `refined_row_ids`: stable logical row IDs for curated refined detection rows.
- `frame_indices`: frame for each curated row.
- `frame_offsets` and `frame_counts`: read acceleration for frame-local slices.
- `source_detect_row_index`: raw detect row lineage when the row is backed by a
  raw candidate; `-1` for manual additions or rows without raw lineage.

Required linkage fields on `source_detections/`:

- `source_detect_row_index`: raw detect row ID in the bound source detect run.
- `decision_codes`: accepted/filtered/duplicate/manual-clear decision.
- `resolved_refined_row_id`: the accepted refined row ID, or `-1` when no
  refined row resolves from that raw candidate.

## Rules

- `refined_row_ids` are artifact row identity, not fish identity, arena identity,
  track identity, or physical array position.
- `refined_row_ids` must be unique and nonnegative in written `instances/`
  surfaces.
- Physical rows should be sorted by `frame_indices` then `refined_row_ids`.
- `frame_offsets` must match `frame_counts`, and `frame_counts` must match
  `frame_indices`.
- A raw-backed instance row must retain `source_detect_row_index`.
- Accepted `source_detections` rows must resolve to an existing
  `instances/refined_row_ids` value.
- If an instance row points at a raw source row, that source row must be accepted
  and its `resolved_refined_row_id` must match the instance row ID.
- Deleted refined rows are omitted from the next `instances/` surface, but their
  old IDs must not be reused.
- Edited refined rows keep their existing IDs.
- Added rows receive new IDs.

## Downstream Stale Semantics

Crop runs sourced from canonical refined detections must preserve this logical
identity as `crop_runs/<run>/source_refined_row_ids`, aligned one-to-one with
`crop_runs/<run>/frame_indices`, `bbox_norm_coords`, and `detection_indices`.
`detection_indices` remains the physical row index into the resolved
`detection_source_path`; it is useful for array addressing but is not stable
enough by itself for row-local repair after insertions or deletions.
Downstream ROI-aligned stages should preserve `source_refined_row_ids` and
`source_detect_row_index` when copying row lineage from crop/keypoint/mask
sources.

Merged training exports should carry the same row lineage under
`source_index/source_refined_row_ids` and
`source_index/source_detect_row_index`, alongside `source_dataset_idx`,
`source_frame_idx`, and `source_roi_idx`. The stable sample key for review,
repair, or cross-artifact joins is therefore source dataset plus refined row
ID when present, with raw detect row index as the legacy/raw-backed fallback.

Row-local repair and review consumers should resolve user-facing flag entries
by `source_refined_row_id` first, `source_detect_row_index` second, and only
then by legacy `frame_idx`/`roi_idx`. In-place crop repair may still use
`detection_indices` to address the resolved source row after the stable row ID
selects the target crop row.

Legacy crop runs that predate row-lineage propagation can be audited or
backfilled from their refined-detect `detection_source_path` when that source
points at an `instances/` surface:

```bash
scripts/py -m fisheye.utils.backfill_crop_row_lineage /nvme1/recordings --recursive
scripts/py -m fisheye.utils.backfill_crop_row_lineage /nvme1/recordings --recursive --apply --consolidate-metadata
```

The utility maps each crop row's physical `detection_indices` value into
`instances/refined_row_ids` and `instances/source_detect_row_index`. Rows whose
physical index no longer resolves are written as `-1`; they must not be guessed
from frame-local ordinal position.

When a bbox edit preserves `refined_row_id`, downstream crop/keypoint/mask rows
may be preserved and marked stale for targeted review or refresh.

When an edit adds, deletes, splits, merges, or reassigns rows, Palette should
preserve any surviving rows by `refined_row_id` but treat the changed row set as
a topology change. Consumers must not assume frame-local ordinal position is
stable enough for row-local stale repair.

When `refined_row_ids` or `source_detect_row_index` cannot be trusted, downstream
repair should escalate to broader rerun or invalidation.

## Validation

Use `fisheye.shared.refined_detect_identity.validate_refined_detect_identity` for
programmatic checks, or:

```bash
scripts/py -m fisheye.utils.inspect_refined_detect_run /path/to/archive.zarr --fail-on-validation-error
```

This validation is stricter than legacy read compatibility. It answers whether a
current sparse refined-detect surface is safe to use for row-local downstream
stale decisions.
