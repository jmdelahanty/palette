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
  `keypoint_performance` and `keypoint_quality` after batch runs.

Pre-migration registry audit against `/nvme1/palette_registry.sqlite`:

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

Post-migration live refresh against `/nvme1/palette_registry.sqlite`:

- Applied schema migration 55.
- Refreshed 166/167 datasets selected from existing `keypoint_performance`
  rows; the remaining dataset path was missing.
- Refreshed 167/179 datasets selected from `recording_step_status` keypoint OK
  rows; 12 dataset paths were missing.
- `keypoint_performance`: 485 rows.
- methods: `yolo_pose` 372, `traditional_pose` 105, `merged_export` 6, blank 2.
- `source_roi_pixel_contract_name`: 410 missing, 67
  `orange_mono_pynvvc_luma_uint8_v1`, 8 `nv12_luma_plane_uint8`.
- `source_roi_read_mode`: 440 missing, 19 `temporary_cache`, 18
  `materialized_crop_run`, 8 `flat_bin_roi_cache`.
- `input_mode_effective`: 481 missing, 4 `tensor`.
- GoodCopBadCop current rows now appear in
  `recording_keypoint_performance_latest` with
  `source_roi_pixel_contract_name=nv12_luma_plane_uint8` and
  `source_roi_read_mode=flat_bin_roi_cache`.

Conclusion: the stable registry surface is now queryable for current/newer runs,
but historical rows remain under-labeled. Consumers should treat missing
`source_roi_pixel_contract_name` as unknown, not equivalent.

Coverage reporting added on 2026-06-15:

- `scripts/py -m fisheye.utils.report_keypoint_contract_coverage` reports
  `keypoint_performance` contract coverage without writing to the registry.
- It can inspect all keypoint rows, latest-per-dataset rows, or
  latest-per-recording rows, and can emit JSONL for downstream triage.
- Group status separates explicit mixing from unknown data:
  `mixed_explicit`, `mixed_with_unknown`, `explicit_with_unknown`,
  `unknown_only`, and `explicit_single`.
- The current Orange/luma pair,
  `orange_mono_pynvvc_luma_uint8_v1` plus `nv12_luma_plane_uint8`, is treated as
  a candidate-compatible set for reporting. This does not make them globally
  equivalent; training/export code should still record any explicit override in
  the exported manifest.

Keypoint training/export preflight gate added on 2026-06-16:

- `prepare_keypoint_training_from_registry` now matches each selected
  `keypoint_run_resolved` to `keypoint_performance` and refuses source runs with
  missing `source_roi_pixel_contract_name`.
- Single-contract source sets pass by default and write
  `required_roi_pixel_contract_name` plus `keypoint_contract_policy` into the
  manifest.
- Mixed explicit contracts fail by default. They are allowed only with repeated
  `--compatible-keypoint-contract` values, and the manifest records the
  compatibility group, contract counts, read-mode counts, cache-backend counts,
  and input-mode counts.
- Per-dataset manifest rows now carry the selected keypoint run's ROI contract,
  read mode, cache backend, input mode, crop signature, and crop revision from
  `keypoint_performance`.
- Live GoodCopBadCop keypoint rows are currently single-contract
  `nv12_luma_plane_uint8` with `flat_bin_roi_cache`/`flat_bin_v1`, so they do not
  require a mixed-contract override.

Remaining consumer mismatch:

- GoodCopBadCop is not selectable through the existing prepare preflight yet
  because `query_datasets(..., model_input='gray')` requires registry
  `has_images_ds`/`downsample_formats_json` metadata. These external-video /
  crop-cache analysis Zarrs have keypoint/crop sources but no stored
  `raw_video/images_ds` surface. That is separate from the keypoint contract
  gate and should be handled by a later model-input/source-selection cleanup.

## Target Registry Fields

Implemented on 2026-06-15:

- nullable `keypoint_performance` columns for the fields below,
- extraction from `keypoints_runs/<run>.attrs` with `provenance.inputs`
  fallback,
- projection through `keypoint_performance_latest` and
  `recording_keypoint_performance_latest`,
- focused tests for cache-backed YOLO-style attrs and traditional
  materialized-crop attrs,
- inline `keypoint_performance` refresh from successful direct YOLO and
  traditional keypoint completion status emission.

Fields:

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

1. [x] Add a numbered registry migration for the new `keypoint_performance` columns
   and refresh `keypoint_performance_latest` plus
   `recording_keypoint_performance_latest`.
2. [x] Extend `registry/extractors/keypoint_performance.py` to extract the fields
   from run attrs first, then fall back to `provenance.inputs` where available.
3. [x] Add tests for the extractor and views using both YOLO/cache-backed attrs and
   traditional/materialized-crop attrs.
4. [x] Add an inline refresh path for direct keypoint writers, mirroring crop:
   successful keypoint completion should refresh `keypoint_performance` for the
   current Zarr without requiring a later batch wrapper or full registry scan.
5. [x] Re-scan or targeted-refresh `/nvme1` after the migration so recent status
   telemetry is reflected in `keypoint_performance`.
6. [x] Update keypoint training/export registry filters to use
   `keypoint_performance.source_roi_pixel_contract_name` when selecting source
   runs. Mixed contracts should be detected and reported by default; allow them
   only through an explicit compatibility group or override recorded in the
   exported manifest.
7. [x] Document the allowed current contracts:
   `orange_mono_pynvvc_luma_uint8_v1` for materialized PyNvVC luma crop runs and
   `nv12_luma_plane_uint8` for the current flat ROI cache path until that cache
   contract is renamed or normalized.
8. [x] After one production refresh, add a registry audit query that reports
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
