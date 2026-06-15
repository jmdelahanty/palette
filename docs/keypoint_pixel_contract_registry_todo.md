# Keypoint Pixel-Contract Registry Todo

Last verified: 2026-06-15

Purpose: make keypoint runs queryable and enforceable by the exact ROI pixel
surface/model-input representation they consumed. This follows the crop-contract
work: crop runs now expose `roi_pixel_contract_name` in `crop_quality`; keypoint
runs should expose their consumed source ROI contract in `keypoint_performance`.

## Current State

Writers already carry some of the right attrs:

- `src/fisheye/detection/detect_keypoints_yolo.py` writes crop snapshot attrs,
  active ROI pixel attrs, ROI cache/read-mode attrs, and `input_mode_*` attrs
  into `keypoints_runs/<run>` and step-status `details_json`.
- `src/fisheye/detection/detect_keypoints_traditional.py` writes crop snapshot
  attrs from the source crop run. This is appropriate for materialized
  `roi_images`, but it does not carry cache/read-mode attrs because that path is
  not cache-backed.
- `src/fisheye/utils/run_keypoints_batch.py` refreshes
  `keypoint_performance` and `keypoint_quality` after batch runs, but the
  performance extractor does not yet preserve pixel-contract fields.

Registry audit against `/nvme1/palette_registry.sqlite`:

- `keypoint_performance`: 478 rows, 167 datasets, 58 recordings.
- methods: `yolo_pose` 365, `traditional_pose` 105, `merged_export` 6, blank 2.
- `keypoint_performance` columns currently stop at source run, model, runtime,
  thresholds, `imgsz`, and summary JSON. It has no
  `source_roi_pixel_contract_name`, `source_roi_read_mode`, cache backend, or
  `input_mode_*` columns.
- `recording_step_status.details_json` for `step_name='keypoints'` already has
  newer pixel-contract telemetry, but only for recent runs:
  `source_roi_pixel_contract_name`: 171 missing, 8
  `orange_mono_pynvvc_luma_uint8_v1`, 4 `nv12_luma_plane_uint8`.
- Current GoodCopBadCop keypoint status rows report
  `source_roi_pixel_contract_name=nv12_luma_plane_uint8` and
  `source_roi_read_mode=flat_bin_roi_cache`; one sweep run also reports
  `input_mode_effective=tensor`.

Conclusion: writer-side provenance is partially present, but the stable registry
surface is not queryable enough. Consumers that need to filter by keypoint model
input currently have to parse `recording_step_status.details_json`, which is the
wrong long-term surface.

## Target Registry Fields

Add these nullable columns to `keypoint_performance` and current/latest views:

- `source_crop_storage_mode`
- `source_crop_signature`
- `source_crop_revision`
- `source_roi_image_representation`
- `source_roi_pixel_contract_name`
- `source_roi_pixel_contract_json`
- `source_roi_read_mode`
- `roi_cache_policy`
- `source_roi_cache_used`
- `source_roi_cache_backend`
- `source_roi_live_acceleration_effective`
- `source_roi_live_gpu_chunk_frames`
- `input_mode_requested`
- `input_mode_effective`

Keep `source_roi_cache_key` and `source_roi_cache_path` out of the default
latest views unless a consumer needs them; they are useful provenance but are
less stable as query dimensions.

## Implementation Checklist

1. Add a numbered registry migration for the new `keypoint_performance` columns
   and refresh `keypoint_performance_latest` plus
   `recording_keypoint_performance_latest`.
2. Extend `registry/extractors/keypoint_performance.py` to extract the fields
   from run attrs first, then fall back to `provenance.inputs` where available.
3. Add tests for the extractor and views using both YOLO/cache-backed attrs and
   traditional/materialized-crop attrs.
4. Add an inline refresh path for direct keypoint writers, mirroring crop:
   successful keypoint completion should refresh `keypoint_performance` for the
   current Zarr without requiring a later batch wrapper or full registry scan.
5. Re-scan or targeted-refresh `/nvme1` after the migration so recent status
   telemetry is reflected in `keypoint_performance`.
6. Update keypoint training/export registry filters to use
   `keypoint_performance.source_roi_pixel_contract_name` when selecting source
   runs. Mixed contracts should be detected and reported by default; allow them
   only through an explicit compatibility group or override recorded in the
   exported manifest.
7. Document the allowed current contracts:
   `orange_mono_pynvvc_luma_uint8_v1` for materialized PyNvVC luma crop runs and
   `nv12_luma_plane_uint8` for the current flat ROI cache path until that cache
   contract is renamed or normalized.
8. After one production refresh, add a registry audit query that reports
   keypoint runs with missing `source_roi_pixel_contract_name` among current
   source-analysis datasets.

## Non-Goals For This Slice

- Do not change keypoint model preprocessing yet.
- Do not rewrite existing keypoint arrays.
- Do not infer a pixel contract for old keypoint runs unless the source crop run
  has an explicit, current crop contract.
- Do not assume different contract names are always incompatible; first make
  mixed-contract use visible, then validate whether specific contracts are
  empirically equivalent enough to share a training/export set.
