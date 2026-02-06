# Detection Merged Export Contract (Proposed)

Purpose: define an implementation-ready contract for exporting a single merged detection-training Zarr per training set, while preserving provenance and compatibility with current training loaders.

## Scope

- Output a single merged Zarr for detection training.
- Preserve source traceability for each exported sample.
- Persist deterministic split membership in the exported Zarr.
- Keep compatibility with existing detection loader expectations.

## Non-goals

- Replacing registry/manifest workflow.
- Supporting pose/segmentation labels in this exporter version.
- Rewriting existing source Zarr layout.

## Compatibility Targets

The merged Zarr must remain compatible with current assumptions used by:

- `src/fisheye/training/zarr_yolo_dataset_loader.py`
- `src/fisheye/diagnostics/prepare_detect_training.py`

Compatibility requirements:

- `raw_video/images_ds` exists.
- `crop_runs/<run_id>/bbox_norm_coords` exists.
- `crop_runs.attrs["latest"]` points to `<run_id>`.
- `crop_runs/<run_id>/detection_source` exists when `includes_interpolated=true`.
- `crop_runs/<run_id>/frame_indices` exists.

## Output Layout (Zarr v3)

```
<merged>.zarr/
  raw_video/
    images_ds                  (N, H, W) uint8
    images_ds_rgb              (N, H, W, 3) uint8   # optional
    attrs:
      downsample_formats       ["gray"] or ["gray","rgb"]
      downsampled_resolution   [H, W]
      fps                      float (optional if mixed sources)

  crop_runs/
    attrs:
      latest                   "<run_id>"
    <run_id>/
      bbox_norm_coords         (N, 4) float32
      frame_indices            (N,) int64
      detection_source         (N,) int8             # 0 real, 1 interpolated
      attrs:
        detection_source_type  "manual|filtered|detect|interpolated"
        includes_interpolated  bool
        detection_source_path  "merged://source_index"
        n_real_detections      int
        n_interpolated_detections int

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
    source_dataset_idx         (N,) int32
    source_frame_idx           (N,) int64
    source_dataset_id          (M,) unicode string
    source_zarr_path           (M,) unicode string
    attrs:
      mapping_version          1
      source_count             M

  attrs:
    zarr_purpose               "training"
    training_export            {...}
```

## Root `training_export` Attr Contract

Required keys:

- `schema_version`: `"1.0.0"`
- `task`: `"detect"`
- `set_id`: training set id string
- `set_name`: string
- `set_version`: int
- `source_type_requested`: string
- `input_format`: `"gray"` or `"rgb"` or `"both"`
- `include_rgb`: bool
- `created_at_utc`: ISO-8601 UTC
- `manifest_path`: path string if available
- `manifest_sha256`: sha256 hex if available
- `query_filter`: JSON object (same normalized payload style as `prepare_detect_training`)
- `invocation`: JSON object (same style as `build_invocation_record`)
- `source_dataset_ids`: list[str]
- `source_zarr_paths`: list[str]

## Invariants

- All sample-aligned arrays share identical first dimension `N`:
  - `raw_video/images_ds`
  - `crop_runs/<run_id>/bbox_norm_coords`
  - `crop_runs/<run_id>/frame_indices`
  - `crop_runs/<run_id>/detection_source`
  - `source_index/source_dataset_idx`
  - `source_index/source_frame_idx`
- `train_indices`, `val_indices`, `test_indices` are disjoint.
- Union of split indices equals `{0..N-1}` when test split is enabled.
- `source_dataset_idx[i]` is within `[0, M-1]`.
- `source_frame_idx[i]` is the original frame index in the source Zarr context.
- `detection_source` encoding is stable:
  - `0` real/manual-reviewed sample
  - `1` interpolated/synthetic sample

## Determinism Rules

Given identical ordered input dataset list, identical seed, identical split ratios, and identical filtering rules:

- Exported sample order is deterministic.
- Split indices are deterministic.
- `manifest_sha256` and exported metadata hashes are stable except for explicit time fields.

## CLI Contract (Exporter)

Proposed CLI surface:

- `--merge`
- `--out-zarr <path>`
- `--split 0.8/0.2` or `--split 0.8/0.1/0.1`
- `--seed 42`
- `--input-format gray|rgb|both`
- `--include-rgb` (alias for `--input-format both`)
- `--source-type manual|filtered|detect|interpolated`
- `--set-name <name>`
- `--set-version <int>`
- `--set-id <id>` (optional override)
- `--registry <path>` (optional)

Behavior:

- Default mode writes gray only.
- If `--input-format both` is requested and any source lacks RGB downsample arrays, fail fast.
- `--split` writes fixed index arrays in `splits/`.

## Validation Checklist (Exporter)

- Validate all input datasets have required arrays for selected source type.
- Validate merged array lengths are consistent.
- Validate bbox coordinates are finite and in normalized range rules already used by preflight.
- Validate split coverage/disjointness.
- Validate source index lookup integrity.
- Write a summary JSON next to output with counts:
  - total samples
  - per-source dataset counts
  - real vs interpolated counts
  - split counts

## Registry Integration (Recommended)

When `--registry` is provided:

- Upsert merged dataset row into `datasets` with `zarr_purpose="training"`.
- Upsert provenance/context if available from dominant or homogeneous source metadata.
- Add `training_sets.dataset_ids_json` linkage to include merged dataset id.
- Persist invocation metadata in `training_sets.invocation_json`.

## Open Decision Defaults

Recommended defaults for initial implementation:

- Include `splits/` arrays: yes.
- Single merged Zarr per training set: yes.
- Default output modality: gray only.
- Optional RGB export: yes via `--include-rgb` or `--input-format both`.
