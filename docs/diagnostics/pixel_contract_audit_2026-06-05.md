# Pixel Contract Audit
<!-- contract-meta
status: current
last_verified: 2026-06-14
purpose: Record persisted-pixel/decode-contract audit results and the implementation checklist for making training and recording Zarr pixel metadata enforceable.
-->

## Summary

Palette should distinguish persisted pixel artifacts from model-input tensors.
The current audit supports that distinction: no persisted pixel surface was found
with `pynvvc_nv12_rgb` metadata. That backend is still best understood as the
current full-frame detection inference tensor path, not as a persisted-image
writer.

The actual problem is under-labeled persisted pixels:

- Recording and training Zarr raw-video surfaces often store only `gray`,
  resolution, and writer/import hints.
- Current `pynvvc_luma` crop runs usually carry a structured
  `roi_pixel_contract`, but many do not also stamp the scalar
  `roi_pixel_contract_name`.
- Merged training exports do not yet aggregate or enforce source pixel
  contracts at the export root or copied raw-video surfaces.

Encoded source-video metadata is better covered than pixel-contract metadata for
traditional single-video recordings, but it is not yet sufficient for strict
regeneration/parity enforcement across clipped and merged artifacts. Treat source
video provenance as a related but separate contract: it describes the compressed
video inputs, while pixel contracts describe the decoded/stored pixels derived
from those inputs.

## Audit Inputs

Tool:

```bash
scripts/py -m fisheye.utils.audit_zarr_pixel_contracts
```

Use `--include-source-video-metadata` to add encoded source-video coverage rows
to the same JSONL report. The flag is opt-in so existing pixel-contract report
counts remain stable.

Use `--apply-safe-scalar-name-backfill` only for the narrow crop-run metadata
repair that copies structured `roi_pixel_contract.name` to scalar
`roi_pixel_contract_name`. This does not rewrite pixel arrays.

Use `--source-video-backfill-plan-jsonl <path>` for a report-only source-video
metadata backfill plan. The plan identifies which rows need `ffprobe` fields,
fingerprints, or path repair; it does not mutate Zarrs or registry rows.

Reports generated on 2026-06-05:

```text
/tmp/palette_pixel_contract_audit_recordings_20260605.jsonl
/tmp/palette_pixel_contract_audit_recordings_20260605.summary.json
/tmp/palette_pixel_contract_audit_merged_training_20260605.jsonl
/tmp/palette_pixel_contract_audit_merged_training_20260605.summary.json
```

Recording/registry command:

```bash
scripts/py -m fisheye.utils.audit_zarr_pixel_contracts \
  --registry /nvme1/palette_registry.sqlite \
  --path-contains /recordings/ \
  --include-source-video-metadata \
  --skip-missing-zarrs \
  --output-jsonl /tmp/palette_pixel_contract_audit_recordings_20260605.jsonl \
  --summary-json /tmp/palette_pixel_contract_audit_recordings_20260605.summary.json \
  --summary-to-stderr
```

Merged-training command:

```bash
scripts/py -m fisheye.utils.audit_zarr_pixel_contracts \
  /nvme1/training/datasets \
  --recursive \
  --skip-missing-zarrs \
  --output-jsonl /tmp/palette_pixel_contract_audit_merged_training_20260605.jsonl \
  --summary-json /tmp/palette_pixel_contract_audit_merged_training_20260605.summary.json \
  --summary-to-stderr
```

## Findings

Recording Zarr audit:

```text
zarr_count: 134
row_count: 632
surface_counts:
  raw_video: 122
  raw_video_array: 120
  crop_run: 198
  crop_roi_images: 192
missing_field_counts:
  decode_backend: 242
  pixel_contract: 242
  roi_pixel_contract: 248
  roi_pixel_contract_name: 73
```

Recording backfill categories:

```text
present: 69
safe_scalar_name_backfill: 73
infer_from_crop_run_attrs: 125
infer_legacy_opencv_gray_from_writer: 12
legacy_import_gray_under_labeled: 218
parent_crop_contract_missing: 123
unknown_raw_video_contract: 12
```

Merged training audit:

```text
zarr_count: 19
row_count: 65
surface_counts:
  merged_training_root: 19
  raw_video: 8
  raw_video_array: 8
  crop_run: 19
  crop_roi_images: 11
missing_field_counts:
  decode_backend: 35
  pixel_contract: 35
  roi_pixel_contract: 30
```

Merged-training backfill categories:

```text
missing_export_contract: 19
requires_source_contract_audit: 16
unknown_crop_contract: 19
parent_crop_contract_missing: 11
```

Observed explicit decode backends:

```text
recording surfaces: pynvvc_luma only
merged training surfaces: none
```

Observed explicit pixel contracts:

```text
orange_mono_pynvvc_luma_uint8_v1: 122 surfaces
raw_video_images_full_to_uint8_grayscale: 16 surfaces
geometry_only_deferred_uint8_grayscale: 4 surfaces
```

Encoded source-video metadata spot check:

```text
active /recordings/ registry zarr rows: 194
rows with video_codec: 171
rows with video_pix_fmt: 171
rows with format_comment: 166
rows with encoder_params_json: 166
rolling_clips/clipped-training rows with registry codec coverage: 0/4
```

Traditional single-video `raw_video` attrs usually carry useful provenance such
as `source_path`, `video_width`, `video_height`, `video_codec`,
`video_pix_fmt`, `source_video_total_frames`, `format_comment`, and parsed
encoder params. Clipped training `raw_video` attrs are much thinner; the rich
mapping lives in sidecars such as `recording_clip_index.json`,
per-clip manifests, keyframe JSON, and the recording snapshot, not in a
standalone Zarr/registry source-video manifest.

## Follow-Up Run: 2026-06-08

Reports:

```text
/tmp/palette_metadata_backfill_preview_20260608.jsonl
/tmp/palette_metadata_backfill_preview_20260608.summary.json
/tmp/palette_metadata_backfill_apply_20260608.jsonl
/tmp/palette_metadata_backfill_apply_20260608.summary.json
/tmp/palette_metadata_backfill_verify_20260608.jsonl
/tmp/palette_metadata_backfill_verify_20260608.summary.json
/tmp/palette_source_video_backfill_plan_verify_20260608.jsonl
/tmp/palette_source_video_stat_fingerprint_apply_20260608.jsonl
/tmp/palette_source_video_stat_fingerprint_apply_20260608.summary.json
/tmp/palette_source_video_stat_fingerprint_verify_20260608.jsonl
/tmp/palette_source_video_stat_fingerprint_verify_20260608.summary.json
/tmp/palette_goodcopbadcop_colorimetry_apply_20260609.jsonl
/tmp/palette_goodcopbadcop_colorimetry_apply_20260609.summary.json
/tmp/palette_goodcopbadcop_colorimetry_verify_20260609.jsonl
/tmp/palette_goodcopbadcop_colorimetry_verify_20260609.summary.json
/tmp/palette_source_video_colorimetry_verify_20260609.jsonl
/tmp/palette_source_video_colorimetry_verify_20260609.summary.json
```

The first registry scan hit an inaccessible `/groups/...` path with
`OSError: [Errno 127] Key has expired`. The audit tool now treats inaccessible
paths as absent when `--skip-missing-zarrs` is enabled, so local `/nvme1`
coverage can continue without aborting on stale PRFS credentials.

Safe scalar-name backfill result:

```text
preview safe_scalar_name_backfill: 65
apply safe_scalar_name_backfill_action_counts: {"updated": 65}
verify safe_scalar_name_backfill: 0
verify missing roi_pixel_contract_name: 0
```

No pixel arrays were rewritten. The applied change was limited to crop-run
metadata: `roi_pixel_contract.name -> roi_pixel_contract_name`.

Remaining recording source-video metadata gaps after verification:

```text
source_video rows: 129
missing only fingerprint: 113
missing colorimetry + fingerprint: 4
missing source path + codec/pix_fmt/dimensions/fps/frame count/colorimetry/fingerprint: 12
```

The four rows missing colorimetry plus fingerprint are the GoodCopBadCop
analysis Zarrs. They already have codec, pix_fmt, dimensions, fps, and frame
count. The twelve severe source-path gaps are stale-looking sickyfish
cross-camera training rows and should be verified/removed from active registry
coverage before any metadata repair is attempted.

Stat fingerprint backfill result:

```text
preview source-video rows with existing videos and missing fingerprint: 117
unique source videos represented by those rows: 60
apply source_video_stat_fingerprint_action_counts: {"updated": 117, "skipped": 12}
verify source_video_backfill_status_counts:
  present: 113
  missing_colorimetry_or_fingerprint: 4
  missing_source_video_path: 12
verify fingerprint missing count: 12
```

The applied fingerprint is deliberately not a full-file content hash. It is a
fast source identity guard:

```text
source_video_fingerprint_strategy = "stat_v1"
source_video_fingerprint = sha256(path + size_bytes + mtime_ns + codec + pix_fmt + width + height + fps + frame_count)
source_video_size_bytes = stat(source_video).st_size
source_video_mtime_ns = stat(source_video).st_mtime_ns
```

Use full-file `source_video_sha256` only for explicit archival/integrity audits.
For routine training/export enforcement, `stat_v1` is the current fast guard
against accidental path/file replacement.

GoodCopBadCop colorimetry backfill result:

```text
preview GoodCopBadCop rows missing colorimetry: 4
ffprobe reported for all four source videos:
  color_range: tv
  color_space: <absent>
  color_transfer: <absent>
  color_primaries: <absent>
apply source_video_ffprobe_colorimetry_action_counts: {"updated": 4}
verify GoodCopBadCop source_video_backfill_status_counts: {"present": 4}
verify all accessible recording source_video_backfill_status_counts:
  present: 117
  missing_source_video_path: 12
```

Only ffprobe-reported fields were stamped. Palette did not invent color matrix,
transfer, or primaries for monochrome Orange videos. The updated attrs are:

```text
color_range = "tv"
source_video_colorimetry_source = "ffprobe_stream"
```

Merged training source-video metadata remains under-labeled:

```text
merged training zarrs scanned: 19
merged source-video rows missing source_video_path: 19
```

Treat merged exports as derived artifacts: repair source Zarr metadata first,
then re-export rather than manually patching merged artifacts.

## Follow-Up Run: 2026-06-14

Reports:

```text
/tmp/palette_pixel_contract_audit_recordings_20260614.jsonl
/tmp/palette_pixel_contract_audit_recordings_20260614.summary.json
/tmp/palette_pixel_contract_audit_recordings_crop_focus_20260614.jsonl
/tmp/palette_pixel_contract_audit_recordings_crop_focus_20260614.summary.json
/tmp/palette_current_crop_contract_report_20260614.json
/tmp/palette_current_crop_contract_backfill_apply_nvme1_20260614.jsonl
/tmp/palette_current_crop_contract_backfill_apply_nvme1_20260614.summary.json
/tmp/palette_current_crop_contract_report_nvme1_apply_20260614.json
/tmp/palette_current_crop_contract_backfill_verify_nvme1_20260614.jsonl
/tmp/palette_current_crop_contract_backfill_verify_nvme1_20260614.summary.json
/tmp/palette_current_crop_contract_report_nvme1_verify_20260614.json
/tmp/palette_pixel_contract_audit_merged_training_20260614.jsonl
/tmp/palette_pixel_contract_audit_merged_training_20260614.summary.json
```

Recording/registry command:

```bash
scripts/py -m fisheye.utils.audit_zarr_pixel_contracts \
  --registry /nvme1/palette_registry.sqlite \
  --path-contains /recordings/ \
  --include-source-video-metadata \
  --skip-missing-zarrs \
  --output-jsonl /tmp/palette_pixel_contract_audit_recordings_20260614.jsonl \
  --summary-json /tmp/palette_pixel_contract_audit_recordings_20260614.summary.json \
  --summary-to-stderr
```

Focused current-crop command:

```bash
scripts/py -m fisheye.utils.audit_zarr_pixel_contracts \
  --registry /nvme1/palette_registry.sqlite \
  --path-contains /recordings/ \
  --skip-missing-zarrs \
  --output-jsonl /tmp/palette_pixel_contract_audit_recordings_crop_focus_20260614.jsonl \
  --summary-json /tmp/palette_pixel_contract_audit_recordings_crop_focus_20260614.summary.json \
  --crop-contract-report-json /tmp/palette_current_crop_contract_report_20260614.json \
  --summary-to-stderr
```

Merged-training command:

```bash
scripts/py -m fisheye.utils.audit_zarr_pixel_contracts \
  /nvme1/training/datasets \
  --recursive \
  --skip-missing-zarrs \
  --output-jsonl /tmp/palette_pixel_contract_audit_merged_training_20260614.jsonl \
  --summary-json /tmp/palette_pixel_contract_audit_merged_training_20260614.summary.json \
  --summary-to-stderr
```

Current `/nvme1` registry recording audit:

```text
zarr_count: 129
row_count: 732
surface_counts:
  raw_video: 117
  raw_video_array: 112
  crop_run: 190
  crop_roi_images: 184
source_video_scope_counts:
  single_video: 129
source_video_backfill_status_counts:
  present: 117
  missing_source_video_path: 12
missing_field_counts:
  decode_backend: 229
  pixel_contract: 229
  roi_pixel_contract: 248
```

Current pixel-contract backfill categories:

```text
present: 126
legacy_import_gray_under_labeled: 217
infer_from_crop_run_attrs: 125
parent_crop_contract_missing: 123
unknown_raw_video_contract: 12
```

Current-crop focused report:

```text
zarrs_with_crop_runs: 117
zarrs_with_current_crop_run: 117
zarrs_missing_current_crop_selector: 0
crop_run_rows_scanned: 198
current_crop_run_rows: 117
current_crop_runs_with_contract: 8
current_crop_runs_missing_contract: 109
contract_counts:
  orange_mono_pynvvc_luma_uint8_v1: 8
  geometry_only_deferred_uint8_grayscale: 4
  missing: 105
crop_storage_mode_counts:
  geometry_only: 4
  materialized: 113
backfill_status_counts:
  present: 8
  safe_scalar_name_backfill: 4
  infer_from_crop_run_attrs: 105
```

The current-crop report uses `crop_runs.attrs.latest`, `latest_complete`, or
`latest_any` as the current-run selector. As of this scan, every Zarr with crop
runs has one of those selectors. The four `safe_scalar_name_backfill` rows are
clipped-training crop runs on PRFS with a structured
`roi_pixel_contract.name` but no scalar `roi_pixel_contract_name`; these are
safe metadata-only repairs when those stores are in the active write scope. The
105 `infer_from_crop_run_attrs` rows are current historical materialized crop
runs and should remain legacy-labeled unless regenerated.

Current-crop contract apply on `/nvme1/recordings`:

```bash
scripts/py -m fisheye.utils.audit_zarr_pixel_contracts \
  --registry /nvme1/palette_registry.sqlite \
  --path-contains /nvme1/recordings \
  --skip-missing-zarrs \
  --apply-safe-scalar-name-backfill \
  --apply-inferred-legacy-crop-contracts \
  --apply-current-crop-runs-only \
  --output-jsonl /tmp/palette_current_crop_contract_backfill_apply_nvme1_20260614.jsonl \
  --summary-json /tmp/palette_current_crop_contract_backfill_apply_nvme1_20260614.summary.json \
  --crop-contract-report-json /tmp/palette_current_crop_contract_report_nvme1_apply_20260614.json \
  --summary-to-stderr
```

Apply result:

```text
inferred_legacy_crop_contract_action_counts:
  updated: 105
safe_scalar_name_backfill_action_counts: {}
```

Verify command:

```bash
scripts/py -m fisheye.utils.audit_zarr_pixel_contracts \
  --registry /nvme1/palette_registry.sqlite \
  --path-contains /nvme1/recordings \
  --skip-missing-zarrs \
  --output-jsonl /tmp/palette_current_crop_contract_backfill_verify_nvme1_20260614.jsonl \
  --summary-json /tmp/palette_current_crop_contract_backfill_verify_nvme1_20260614.summary.json \
  --crop-contract-report-json /tmp/palette_current_crop_contract_report_nvme1_verify_20260614.json \
  --summary-to-stderr
```

Verified `/nvme1` current-crop state after the apply:

```text
zarrs_with_crop_runs: 113
zarrs_with_current_crop_run: 113
zarrs_missing_current_crop_selector: 0
current_crop_run_rows: 113
current_crop_runs_with_contract: 113
current_crop_runs_missing_contract: 0
contract_counts:
  decord_rgb_channel_mean_uint8: 53
  raw_video_images_full_to_uint8_grayscale: 52
  geometry_only_deferred_uint8_grayscale: 4
  orange_mono_pynvvc_luma_uint8_v1: 4
```

The apply was metadata-only. It wrote `roi_pixel_contract` and
`roi_pixel_contract_name` to current crop-run groups whose contract could be
inferred from existing `crop_storage_mode`/`video_source_type`/acceleration
attrs. It did not rewrite `roi_images`, did not stamp historical crops as
`orange_mono_pynvvc_luma_uint8_v1`, and did not touch older non-current crop
runs because `--apply-current-crop-runs-only` was used.

The broad `/nvme1` post-verify audit still reports 38 crop-contract missing
fields (`infer_from_crop_run_attrs: 20`, `parent_crop_contract_missing: 18`).
Those are older non-current crop runs and their child `roi_images`, not current
training candidates.

Existing `/nvme1/palette_registry.sqlite` rows still need a crop-quality refresh
to make the new contract fields queryable from registry views. The code now
extracts/stores `crop_storage_mode`, `roi_image_representation`,
`roi_pixel_contract_name`, and `roi_pixel_contract_json`, but the existing
SQLite registry was not refreshed as part of this metadata-only Zarr apply.

Registry crop-quality refresh on `/nvme1/recordings`:

```bash
scripts/py -m fisheye.registry.maintenance \
  /nvme1/recordings \
  --registry /nvme1/palette_registry.sqlite \
  --refresh-crop-quality \
  --dry-run

scripts/py -m fisheye.registry.maintenance \
  /nvme1/recordings \
  --registry /nvme1/palette_registry.sqlite \
  --refresh-crop-quality
```

The first apply exposed a maintenance bug: crop-quality refresh attempted
`zarr.open_group(..., consolidated=False)`, but the active zarr version expects
`use_consolidated=False`. The TypeError fallback opened with default metadata,
which could read stale consolidated attrs and miss the freshly stamped crop
contracts. `maintenance._backfill_crop_quality` now uses the repository's
non-consolidated opener helper.

Patched refresh result:

```text
dry-run after opener fix:
  scanned: 117
  missing: 0
  errors: 0
  no_quality: 4
  inserted: 0
  updated: 105
  deleted: 0
  unchanged: 38

apply after opener fix:
  scanned: 117
  missing: 0
  errors: 0
  no_quality: 4
  inserted: 0
  updated: 105
  deleted: 0
  unchanged: 38
```

Registry verification after refresh:

```text
crop_quality rows: 281
rows with roi_pixel_contract_name: 113
rows with roi_pixel_contract_json: 113

active non-smoke /nvme1 source-analysis crop_quality_current rows:
  total: 112
  named: 112
  decord_rgb_channel_mean_uint8: 104
  geometry_only_deferred_uint8_grayscale: 8
```

The one remaining unnamed `/nvme1/recordings` source-analysis current row is the
smoke analysis Zarr under `/nvme1/recordings/smoke/`, which is excluded from
active coverage.

Surface-level breakdown:

```text
raw_video:
  legacy_import_gray_under_labeled: 113
  unknown_raw_video_contract: 4
raw_video_array:
  legacy_import_gray_under_labeled: 104
  unknown_raw_video_contract: 8
crop_run:
  present: 65
  infer_from_crop_run_attrs: 125
crop_roi_images:
  present: 61
  parent_crop_contract_missing: 123
```

Interpretation of the 2026-06-14 recording scan:

- The safe crop scalar-name repair remains complete: no remaining
  `safe_scalar_name_backfill` actions were reported.
- The source-video metadata backfill remains complete for accessible active
  single-video recordings except the same twelve stale-looking cross-camera
  `sickyfish_2026_02_23...*_training.zarr` rows with no source-video path.
- The remaining raw-video gaps are historical contract-labeling gaps, not
  evidence that current writers failed to stamp new `pynvvc_luma` crop/cache
  contracts.
- The 125 `infer_from_crop_run_attrs` rows are historical materialized crop runs
  that can be labeled with medium-confidence legacy contracts if strict
  reproduction requires it, but they should not be promoted to the current
  `orange_mono_pynvvc_luma_uint8_v1` contract without regeneration or parity
  evidence.

The twelve source-video metadata gaps are cross-camera training rows under:

```text
/nvme1/recordings/sickyfish_2026_02_23_16_23_35_cam2010093/zarr/*_training.zarr
/nvme1/recordings/sickyfish_2026_02_23_16_23_35_cam2010094/zarr/*_training.zarr
/nvme1/recordings/sickyfish_2026_02_23_16_23_35_cam2010095/zarr/*_training.zarr
/nvme1/recordings/sickyfish_2026_02_23_16_23_35_cam2010096/zarr/*_training.zarr
```

The four `unknown_raw_video_contract` zarrs are same-camera
`sickyfish_2026_02_23...*_training.zarr` artifacts. Treat these as legacy
training artifacts unless a separate writer audit proves their raw-video
decoder/conversion contract.

Current merged-training audit:

```text
zarr_count: 19
row_count: 65
surface_counts:
  merged_training_root: 19
  raw_video: 8
  raw_video_array: 8
  crop_run: 19
  crop_roi_images: 11
missing_field_counts:
  decode_backend: 35
  pixel_contract: 35
  roi_pixel_contract: 30
backfill_status_counts:
  missing_export_contract: 19
  requires_source_contract_audit: 16
  unknown_crop_contract: 19
  parent_crop_contract_missing: 11
```

Existing merged training exports remain under-labeled legacy artifacts. New
detection merged exports now aggregate source raw-video pixel contracts and can
enforce `--required-pixel-contract-name`; old merged exports should be re-exported
from source-contract-aware inputs rather than manually patched.

## Interpretation

Do not backfill old raw-video images as `pynvvc_luma` unless they are regenerated
or a writer-specific audit proves that contract. Most historical raw-video gray
arrays are not contract-equivalent to the current `pynvvc_luma_v1` crop/cache
contract.

The safe backfill class is narrow:

- If a crop run has a structured `roi_pixel_contract.name`, copying that name to
  `roi_pixel_contract_name` is safe and high confidence.

The medium-confidence historical classes should remain visibly historical:

- Clipped training raw frames created by `create_clipped_training_zarr` can be
  labeled as OpenCV `VideoCapture` plus `cv2.COLOR_BGR2GRAY` if needed, but the
  preferred strict fix is regenerating/copying from explicit source pixels.
- Legacy materialized crop runs can be labeled from
  `crop_storage_mode`/`video_source_type`/`acceleration`, but those labels should
  not be treated as the new canonical `pynvvc_luma_v1` contract.

Merged training exports need source-contract aggregation before they can be
considered enforceable. Existing merged exports should be treated as under-labeled
legacy artifacts until re-exported or audited against source rows.

## Implementation Checklist

1. Normalize existing `pynvvc_luma` crop metadata.

- [x] Extend the audit tool with an apply mode that only handles
  `safe_scalar_name_backfill`.
- [x] For each affected accessible crop run, copy `roi_pixel_contract.name` to
  `roi_pixel_contract_name`.
- [x] Do not rewrite pixel arrays.
- [x] Re-run the audit and require `safe_scalar_name_backfill: 0` for accessible
  `/nvme1` recording Zarrs.

2. Stamp new crop runs consistently at write time.

- [x] Update `regenerate_training_crops_pynvvc.py` to write both
  `roi_pixel_contract` and `roi_pixel_contract_name` at the crop-run top level.
- [x] Verify materialized crop, geometry-only crop, flat ROI cache, and clipped flat
  ROI cache writers all stamp a contract name and structured contract.
- [x] Add unit coverage for scalar-name stamping.

2a. Label current historical crop runs without promoting them to the canonical
    PyNvVC-luma contract.

- [x] Add an explicit `--apply-inferred-legacy-crop-contracts` mode for
  medium-confidence legacy crop contracts inferred from
  `crop_storage_mode`/`video_source_type`/acceleration attrs.
- [x] Add `--apply-current-crop-runs-only` so metadata repair can target only the
  current crop run selected by `latest`, `latest_complete`, or `latest_any`.
- [x] Refuse to infer `orange_mono_pynvvc_luma_uint8_v1`; canonical luma
  contract still requires regeneration or parity evidence.
- [x] Apply current-only inferred legacy crop contracts to `/nvme1/recordings`.
- [x] Verify current crop runs in `/nvme1/recordings` are now fully labeled:
  `113/113` current crop runs have a contract.
- [ ] Apply the safe scalar-name repair to active PRFS clipped-training crop runs
  if those stores remain in active use.
- [x] Refresh `/nvme1/palette_registry.sqlite` crop-quality rows so the new crop
  contract columns are queryable from registry views.

3. Add raw-video pixel contract fields to future imports and clipped builders.

- Define raw-frame contracts separately from ROI contracts. Do not reuse
  `orange_mono_pynvvc_luma_uint8_v1` for legacy raw frames unless the writer
  actually produced that representation.
- For future PyNvVideoCodec luma imports, stamp `raw_video.decode_backend`,
  `raw_video.pixel_contract_name`, and per-array image representation attrs.
- For future clipped training builders, either copy from an already stamped source
  pixel surface or stamp the exact decoder/conversion used.
- Keep legacy `opencv_bgr2gray_uint8` and `legacy_import_gray_uint8` labels
  visibly non-canonical.

4. Update merged training exporters to aggregate source contracts.

- [x] Detection exporter: aggregate source `raw_video/images_ds` pixel contracts into
  the merged root `training_export` and merged `raw_video` attrs.
- Keypoint/pose/mask exporters: aggregate source crop-run
  `roi_pixel_contract_name` values into merged crop attrs and training-export
  payloads.
- [x] If all detection sources match, stamp the single contract name; if detection
  sources differ or are partially labeled, stamp an explicit mixed/partial label.
- If keypoint/pose/mask sources differ, either refuse by default or stamp an explicit mixed-contract
  label only when a caller opts into legacy/mixed export.

5. Add exporter enforcement switches.

- [x] Add detection `--required-pixel-contract-name` enforcement for merged raw-video exports.
- Add `--required-pixel-contract-name` or task-specific equivalents for remaining exporters.
- Default new high-confidence exports to refuse incompatible source contracts.
- Allow `--allow-mixed-pixel-contracts` only for legacy audit/reproduction
  workflows.
- Record the enforcement decision in `training_export`.

6. Re-export current training artifacts that should be used for new model
   training.

- Prefer regenerating merged datasets after source contract metadata is present.
- For keypoint/mask datasets, use source crop runs carrying
  `orange_mono_pynvvc_luma_uint8_v1`.
- For detection datasets, keep current production detection preprocessing
  separate from persisted image contracts until a luma-trained detector is
  validated.

7. Add a recurring audit gate.

- Run `audit_zarr_pixel_contracts` in dry-run/report mode before model training
  exports.
- Fail training-export pipelines if required pixel-contract fields are absent or
  mixed without explicit opt-in.
- Keep JSONL reports as run artifacts alongside training manifests.

8. Add encoded source-video metadata audit and backfill.

- [x] Add a source-video metadata audit surface separate from decoded pixel-contract
  rows. It should report one row per source video or clipped source segment,
  including `source_video_path`, codec, pixel format, width, height, fps, frame
  count, encoder settings, color range/matrix/transfer/primaries when available,
  and a stable source-video fingerprint.
- [x] Read existing metadata from registry provenance, `raw_video` attrs,
  `recording_clip_index.json`, per-clip manifests, keyframe JSON, and recording
  snapshots before falling back to live `ffprobe`.
- [x] Backfill traditional single-video recordings from existing `raw_video` attrs
  and registry provenance where fields are already present for the fast
  `stat_v1` fingerprint case.
- For clipped/rolling-clip recordings, promote sidecar metadata into a
  structured Zarr/registry source-video manifest so clipped training Zarrs are
  not dependent on out-of-band sidecars for basic source provenance.
- Extend future import, clipped-training, and review-proxy writers to stamp
  colorimetry (`color_range`, `color_space`/matrix, `color_transfer`,
  `color_primaries`) in addition to codec/pix_fmt/dimensions/fps/frame counts.
- Make strict regeneration/export tooling require both a decoded pixel contract
  and sufficient encoded source-video metadata. The two contracts should be
  validated independently.

## Open Decisions

- Whether to maintain a historical raw-video contract taxonomy, or simply mark
  old raw-video images as `legacy_unknown_gray_uint8` and require regeneration
  for strict training.
- Whether detection training should continue using legacy gray `images_ds` until a
  luma-replicated detector is trained, or whether the next detection dataset should
  be rebuilt from an explicit `pynvvc_luma` full-frame contract.
- Whether merged datasets should ever allow mixed pixel contracts, or whether
  mixed export should be blocked except for explicit archival reproduction.
- Whether clipped source-video manifests should live under `raw_video`, a new
  `source_videos` group, or registry provenance with a mirrored Zarr copy.
- Which source-video fingerprint is required for enforcement: full-file hash,
  size/mtime plus partial hash, or a video-stream hash derived from ffprobe/packet
  metadata.

## References

- `docs/video_pixel_model_input_contract.md`
- `docs/geometry_only_crop_workflow_cache_design.md`
- `docs/diagnostics/flat_roi_cache_pynvvc_surface_reuse_2026-06-05.md`
- `src/fisheye/utils/audit_zarr_pixel_contracts.py`
- `src/fisheye/shared/roi_pixel_contract.py`
