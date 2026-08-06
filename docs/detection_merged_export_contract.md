# Detection Merged Export Contract
<!-- contract-meta
version: 4
status: active
last_verified: 2026-08-06
-->

Purpose: define the current contract for exporting one merged detection-training
Zarr per training set while preserving source traceability and using the same
canonical refined-detection label surface as normal per-recording training
Zarrs.

## Scope

- Output a single merged Zarr for detection training.
- Preserve source traceability for each exported sample.
- Persist deterministic split membership in the exported Zarr.
- Use `refined_detect_runs/<run>/instances` as the only forward detection-label
  surface.

## Non-Goals

- Replacing registry/manifest workflow.
- Supporting pose/segmentation labels in this exporter version.
- Rewriting existing source Zarr layout.
- Forward-writing `crop_runs` detection-label mirrors in merged exports.

## Compatibility Position

Current readers should prefer:

- `raw_video/images_ds` or `raw_video/images_ds_rgb`
- `refined_detect_runs/<run>/instances/bbox_norm_coords`
- `refined_detect_runs/<run>/instances/frame_indices`
- `refined_detect_runs/<run>/instances/source_kind_codes`
- `refined_detect_runs/<run>/instances/manual_edit_flags`

Legacy readers may still support crop-only training archives, but new merged
exports must not depend on `crop_runs/<run>/bbox_norm_coords` as the label
authority. `crop_runs` remains a materialized image/support concept for older
and per-recording stores, not the merged-export contract.

## Output Layout

```text
<merged>.zarr/
  raw_video/
    images_ds                  (F, H, W) uint8
    images_ds_rgb              (F, H, W, 3) uint8   # optional
    attrs:
      downsample_formats       ["gray"] or ["gray","rgb"]
      downsampled_resolution   [H, W]
      fps                      float (optional if mixed sources)

  refined_detect_runs/
    attrs:
      latest                   "<run_id>"
    <run_id>/
      attrs:
        curated_primary_surface "instances"
        refined_storage_semantics "sparse_instances_v1"
        interpolation_enabled  false
        interpolation_policy   "forbidden_for_merged_training_export"
      instances/
        refined_row_ids        (N,) int64
        frame_indices          (N,) int32           # sorted merged frame identity
        frame_offsets          (F+1,) int64          # zero/one/many instance lookup
        frame_counts           (F,) int32
        bbox_img_xyxy          (N, 4) float64
        bbox_norm_coords       (N, 4) float64       # normalized [cx, cy, w, h]
        source_kind_codes      (N,) int8            # raw_detect/manual only for new exports
        manual_edit_flags      (N,) bool
        source_detect_row_index (N,) int32
        reason_bytes           (N, width) uint8

  detection_training_supervision/
    label_state_codes          (F,) uint8           # 1 positive, 2 negative
    reason_codes               (F,) uint16

  splits/
    train_indices              (Nt,) int64
    val_indices                (Nv,) int64
    test_indices               (Ntest,) int64        # optional
    attrs:
      split_mode               "fixed_indices"
      train_ratio              float
      val_ratio                float
      test_ratio               float or null
      seed                     int
      strategy                 "global_random"
      created_at_utc           ISO-8601

  source_index/
    source_dataset_idx         (F,) int32
    source_frame_idx           (F,) int64
    source_instance_dataset_idx (N,) int32
    source_roi_idx             (N,) int64
    source_refined_row_ids     (N,) int64
    source_detect_row_index    (N,) int32
    source_dataset_id          (M,) UTF-8 string
    source_zarr_path           (M,) UTF-8 string
    attrs:
      mapping_version          2
      source_count             M

  attrs:
    zarr_purpose               "training"
    total_frames               F
    width                      W
    height                     H
    training_export            {...}
```

## Root `training_export` Attr Contract

Required keys:

- `schema_version`: `"2.0.0"`
- `task`: `"detect"`
- `set_id`: training set id string
- `set_name`: string
- `set_version`: int
- `source_type_requested`: string
- `source_type_resolved`: `"refined"`
- `canonical_label_path`: `refined_detect_runs/<run>/instances`
- `sample_axis`: `"frame"`
- `zero_instance_frame_semantics`: `"explicit_reviewed_negative"`
- `total_supervised_frames`: integer `F`
- `total_instances`: integer `N`
- `positive_frame_count` and `negative_frame_count`: integers summing to `F`
- `source_frame_decisions`: exact source decision paths and content digests
- `interpolation_policy`: `"forbidden_for_merged_training_export"`
- `input_format`: `"gray"` or `"rgb"` or `"both"`
- `include_rgb`: bool
- `created_at_utc`: ISO-8601 UTC
- `manifest_path`: path string if available
- `manifest_sha256`: sha256 hex if available
- `query_filter`: JSON object
- `invocation`: JSON object
- `source_dataset_ids`: list[str]
- `source_zarr_paths`: list[str]

## Invariants

- Frame/sample-aligned arrays share first dimension `F`:
  - `raw_video/images_ds` or `raw_video/images_ds_rgb`
  - `detection_training_supervision/label_state_codes`
  - `detection_training_supervision/reason_codes`
  - `source_index/source_dataset_idx`
  - `source_index/source_frame_idx`
- Instance-aligned arrays share first dimension `N`:
  - `refined_detect_runs/<run>/instances/bbox_norm_coords`
  - `refined_detect_runs/<run>/instances/frame_indices`
  - `refined_detect_runs/<run>/instances/source_kind_codes`
  - `refined_detect_runs/<run>/instances/manual_edit_flags`
  - `source_index/source_instance_dataset_idx`
  - `source_index/source_roi_idx`
  - `source_index/source_refined_row_ids`
  - `source_index/source_detect_row_index`
- `instances/frame_indices` is sorted, lies in `[0, F)`, and may repeat.
- `instances/frame_offsets` has shape `F+1`, starts at zero, ends at `N`,
  and its differences exactly equal `instances/frame_counts`.
- A positive supervision frame has one or more instances; a negative frame has
  none. A frame must never be both positive and negative.
- `train_indices`, `val_indices`, and `test_indices` are disjoint.
- Union of split indices equals `{0..F-1}`. Splits are therefore by image, not
  by instance row.
- `source_dataset_idx[i]` is within `[0, M-1]`.
- `source_frame_idx[f]` is the original frame index for merged sample `f`.
- `source_roi_idx[i]` is the source-local instance row index before merge.
- `source_refined_row_ids[i]` is the stable refined-detection row identity
  when available, or `-1` for legacy/unmapped rows.
- `source_detect_row_index[i]` is the raw detect row lineage when available,
  or `-1` for rows without raw-detect backing.
- New merged exports must not contain `source_kind_codes == interpolated`.

## Interpolation Policy

Current refined-detection training data is sparse and final-state based.
Interpolation is not allowed in new merged exports.

If a source dataset contains legacy interpolated rows, the exporter must fail
closed by default with a clear error. The caller should review, migrate, or
exclude those rows before creating a new merged artifact. Reader fallback for
old archives may remain, but forward exports must stay interpolation-free.

## Determinism Rules

Given identical ordered input dataset list, identical seed, identical split
ratios, and identical filtering rules:

- Exported sample order is deterministic.
- Split indices are deterministic.
- Source-index lineage is deterministic.
- Metadata hashes are stable except explicit time fields.

## CLI Contract

- `--merge`
- `--out-zarr <path>`
- `--split 0.8/0.2` or `--split 0.8/0.1/0.1`
- `--seed 42`
- `--include-rgb`
- `--registry <path>` optional

Behavior:

- Default mode writes gray only.
- The merged training config should use `source_type: refined`, regardless of
  the historical source type used to build the input manifest.
- If RGB export is requested and any source lacks RGB downsample arrays, fail
  fast.
- `--split` writes fixed index arrays in `splits/`.

## Validation Checklist

- Validate all input datasets have required arrays for their manifest-selected
  source path.
- Validate output has canonical `refined_detect_runs/<run>/instances`.
- Validate no interpolated source-kind rows are present.
- Validate the frame and instance axes independently, including exact
  `frame_offsets`/`frame_counts`/`frame_indices` agreement.
- When a source has a bound frame-decision run, reject unresolved frames,
  positive/negative collisions, or review state that changes during export.
- Validate bbox coordinates are finite and in normalized range rules already
  used by preflight.
- Validate split coverage/disjointness.
- Validate source-index lookup integrity.
- Write a summary JSON next to output with counts:
  - total samples
  - total instances and positive/negative frame counts
  - per-source dataset counts
  - source-kind counts
  - manual-edited counts when available
  - split counts

## Registry Integration

When `--registry` is provided:

- Upsert merged dataset row into `datasets` with `zarr_purpose="training"`.
- Upsert provenance/context if available from dominant or homogeneous source
  metadata.
- Add `training_sets.dataset_ids_json` linkage to include merged dataset id.
- Persist invocation metadata in `training_sets.invocation_json`.
