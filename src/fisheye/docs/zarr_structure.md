# Palette Zarr Layout (v3)

This reference summarizes the structure produced by the modern Palette
pipeline. It is the **authoritative** human-readable spec.

Machine-readable per-stage array contracts (required/optional arrays, dtypes,
and shape templates) live in `fisheye.shared.zarr.stage_arrays` and should be
treated as the runtime-validation counterpart to this document.

---

## Import profiles and bootstrap requirements

Palette Zarrs are sparse, stage-built archives. A valid archive does not need
every run family listed in this document at creation time. The machine-readable
singleton import-profile authority is
`fisheye.shared.import_profile_contract`; inspect an archive without mutation
using:

```bash
scripts/py -m fisheye.utils.check_import_profile /path/to/archive.zarr
```

| Profile | Active writer authority | Required bootstrap surface | Allowed omissions | Frame universe |
| --- | --- | --- | --- | --- |
| `metadata_only_analysis` | `fisheye.utils.import_recording_analysis` (`ensure_analysis_archive` plus acquisition-metadata publication) | `raw_video/`; canonical source-video locator; `import_method`, `import_stage`, positive `total_frames`, and positive `fps` across root/`raw_video` attrs | All frame arrays, acquisition clock when no camera clock source exists, and all unexecuted stage/run families | Acquisition video frames described by `source_video_metadata`; no stored-frame row domain exists until a later artifact creates one |
| `sampled_training_pynvvc_luma` | `fisheye.utils.import_sampled_training_pynvvc` | `raw_video/images_full`, `raw_video/original_frame_indices`; source-video locator; PyNvVC import/decode/pixel-contract attrs; positive source frame count and frame step | Downsampled pixels, acquisition clock when no camera clock source exists, stimulus runs, and all later analysis families | Stored row `i` maps to acquisition frame `original_frame_indices[i]`; stored row count is not the acquisition frame count |
| `legacy_decord_training_or_full` | None; historical read compatibility only | Classified from historical attrs and stored pixel arrays | No new writes, extensions, or backfills are supported | Interpret only through the persisted historical frame mapping/attrs; regenerate with PyNvVC for a new canonical artifact |

The classifier reports two different severities:

- `required_missing` means the claimed profile is incomplete and consumers
  must fail closed for operations that depend on that authority.
- `recommended_missing` means the profile is structurally valid but provenance
  is degraded, for example a missing source fingerprint or colorimetry field.

Consumer rule: absence of `detect_runs/`, `crop_runs/`, `analysis/`, or any
other optional stage family is normal for a valid newly imported archive.
Consumers must not infer corruption from those omissions or from filenames.
Missing `raw_video/` or a profile-required field is an incomplete import.

The old broad `create_palette_zarr`/Decord bootstrap has been retired. Do not
create empty run-family placeholders merely to make a metadata-only archive
look like a fully processed one.

---

## Root Group

**Attributes**

- `schema_version`, `zarr_format`
- `created_at`, `pipeline_version`, `command_line_args`
- `git_info`, `platform_info`
- `source_video_metadata` (versioned width, height, fps, frames, codec, and
  recording-relative locator; see `docs/source_video_metadata_contract.md`)
- `session_uuid`, `recording_id`, `recording_name`, `recording_path`
- `recording_type`, `recording_subtype`, `behavior_mode`,
  `artifact_schema_id`
- `experiment_context_status` (`"present"` when an H5/protocol source is
  available, `"absent"` for recording-only/video-only archives)
- `experiment_context_source` (`"h5"` or `"none"`)
- `stimulus_runs_available` (`false` for recording-only archives with no
  stimulus/protocol source)
- `experiment_context_status_detail` *(optional human-readable explanation
  when context is absent or degraded)*
- `source_h5`, `source_h5_path` *(when `experiment_context_source == "h5"`)*
- `acquisition_frame_clock_available`, `acquisition_frame_clock_status`
- `acquisition_frame_clock_ref`, `acquisition_frame_clock_sha256`,
  `acquisition_frame_clock_camera_id`, `acquisition_frame_clock_row_count`
  *(when a camera clock source is available)*
- `processing_history` *(optional ordered list)*

`Registry.scan_zarr` treats these root attrs as sufficient recording context:
it indexes both video-only training Zarrs and recording-only analysis Zarrs in
`recordings` and exposes the experiment-context fields through
`dataset_context_current`.

**Possible immediate children (profile- and stage-dependent)**

Only `raw_video/` is required at bootstrap for the active singleton profiles.
Other children appear when their owning stage or metadata publisher runs.

- `raw_video/`
- `background_runs/`
- `detect_runs/`
- `crop_runs/`
- `keypoints_runs/`
- `eye_masks_runs/`
- `subject_mask_runs/`
- `refined_detect_runs/`
- `refined_keypoints_runs/`
- `refined_eye_masks_runs/`
- `refined_subject_masks_runs/`
- `refined_online_runs/`
- `tracking_runs/`
- `arena_assignment_runs/`
- `calibration/`
- `analysis/`
- `analysis_metadata/`

Most `*_runs` groups carry:

- `attrs["latest"]` → most recent run name.
- `attrs["latest_complete"]` → most recent run name whose run-completion
  marker is complete, when the writer uses the modern completion protocol.
- Child run groups named `<stage>_<YYYY-MM-DD_hh-mm-ss>` (possibly with a
  `_NNN` suffix if repeated inside the same second).
- Run attributes capturing provenance (`provenance` dict with command,
  timestamps, git/environment snapshots), upstream run references
  (`source_detect_run`, `source_crop_run`, etc.), configuration snapshots,
  and often `duration_seconds`.
- Per-run completion attrs written by `fisheye.shared.zarr_run_completion`
  (`palette_run_completion_status`, `palette_run_completed_at_utc`, and
  related audit fields) for modern writers. A run must have a complete marker before
  `emit_stage_completion(..., status="ok", run_name=...)` will write an `ok`
  registry status row.

Lightweight stages such as `tracking_runs/` may omit some of the broader
timing-oriented attrs while still recording canonical lineage and summary
statistics.

Machine-readable `StageSpec` array contracts are validated at status-write time
by `fisheye.registry.stage_complete.emit_stage_completion`. Completion-marker
validation is hard for `ok` writes. Array validation is shadow-mode by default:
validation details are recorded in `recording_step_status.details_json`, but a
stage only blocks on array-contract failures after it is explicitly added to
`_ENFORCE_STAGE_ARRAY_VALIDATION_FOR`.

### `analysis/subject_metadata_runs/` and `analysis/experiment_setup_runs/`

New H5-backed imports store immutable acquisition subject snapshots as
versioned `palette.subject_metadata.v1` runs under
`analysis/subject_metadata_runs/<run>` and publish the declared acquisition plan as a
versioned `palette.experiment_setup.v2` run under
`analysis/experiment_setup_runs/<run>`. The setup distinguishes expected and
assigned subjects from the upstream source-dish population. Detection quality
and refinement bind the selected setup path and digest. Root
`experiment_setup`/`subject_count` attrs and stimulus-run subject snapshots are
historical compatibility projections, not the modern authority. See
`docs/experiment_setup_contract.md`.

### `analysis/acquisition_frame_clock_runs/`

New analysis and sampled-training imports publish a compact, immutable
`palette.acquisition_frame_clock.v1` authority when Orange camera timing is
available. The selected run contains recording-wide numeric arrays:

| Array | DType | Meaning |
| --- | --- | --- |
| `recording_frame_id` | `int64` | Session-continuous producer frame identifier |
| `parent_frame_index` | `int64` | Complete, ordered zero-based source-video frame domain |
| `camera_timestamp_ns` | `int64` | Orange `timestamp`, embedded camera-frame clock |
| `system_timestamp_ns` | `int64` | Orange `timestamp_sys`, host `CLOCK_REALTIME` |
| `camera_timestamp_valid` | `bool` | Whether the camera timestamp row is present |
| `system_timestamp_valid` | `bool` | Whether the system timestamp row is present |

The arrays retain exact integer nanoseconds and use explicit validity arrays;
missing clock values are not represented as floating-point NaNs. Runs bind
array digests, camera identity, source metadata, and the exact temporal-domain
semantics in an immutable record. They use 4,096-row logical chunks and request
131,072-row indexed outer shards through the shared columnar writer.

Each clock surface declares `unit`, `clock_domain`, `time_reference_kind`,
`origin`, `timescale`, and `semantic_status`; names ending in `_ns` alone are
not sufficient temporal semantics. `system_timestamp_ns` is producer-declared
POSIX time from `clock_gettime(CLOCK_REALTIME)`: nanoseconds since
`1970-01-01T00:00:00 UTC`, excluding leap seconds, sampled immediately after
`EVT_CameraGetFrame` returns. It is not time since process, stream, or recording
start.

Orange passes `camera_timestamp_ns` through unchanged from
`Emergent::CEmergentFrame.timestamp`. It is classified as an absolute
`IEEE-1588_PTP_TAI` clock only when that recording provides combined evidence:
PTP synchronization configuration, valid low-offset PTP samples, agreement
between the latched PTP clock and embedded frame timestamps, and a stable
camera-minus-host offset near the TAI-UTC difference in force for the
recording. The semantic status remains
`inferred_from_recording_evidence_not_sdk_declared`. Merely enabling PTP is not
enough. If the combined evidence is absent or contradictory, the camera clock
is labeled `device_defined_unknown_epoch` with unspecified origin/timescale;
it is never silently called epoch or recording-relative time.

For a sampled training archive, `raw_video/original_frame_indices[i]` joins to
`parent_frame_index`; `raw_video` records the selected clock reference and
digest. The writer does not create a redundant sampled `timestamps` array.

Source precedence is the manifest-declared full-stream
`frame_clock_metadata`, then a conventional sibling `Cam*_meta.csv`, then an
existing `recording_frame_index.parquet`. H5 stimulus timestamps remain under
`analysis/stimulus_runs`: they are not relabeled as camera acquisition time.
When no camera clock source exists, import explicitly records
`acquisition_frame_clock_available=false` and continues.

Shadow-mode registry telemetry can be inspected with:

```bash
scripts/py -m fisheye.utils.report_stage_array_validation_shadow \
  --registry /nvme1/palette_registry.sqlite \
  --include-no-spec
```

### Future Clipped Analysis Archives

Current readers and writers mostly support the first-class single-video camera
layout with top-level run families such as `detect_runs/`, `crop_runs/`, and
`keypoints_runs/`. Long rolling-clip recordings should not extend that model by
having many cluster jobs append into one giant global run group.

`cams/` is not a legacy layout. It is the canonical single-file-per-camera
recording representation and remains appropriate for short or moderate
recordings, such as 20-30 minute videos. `clips/` is a second first-class
representation for recordings that are intentionally split into rolling video
segments.

The intended future layout is one parent analysis Zarr with clip-local physical
run groups:

```text
<recording>_analysis.zarr/
  # root attrs point to recording-level sidecars:
  # recording_frame_index.parquet
  # recording_frame_index_manifest.json
  detect_runs/                 # parent finalized/aggregated placeholder
  refined_detect_runs/
  crop_runs/
  keypoints_runs/
  subject_mask_runs/
  clips/
    clip_000000/
      cameras/
        2010093/
          source/
            frame_map/
          detect_runs/
          refined_detect_runs/
          crop_runs/
          keypoints_runs/
          subject_mask_runs/
    clip_000001/
      ...
  experiment_index/
    workflow_manifests/
    finalized_runs/
```

In that layout, clip-local run groups are the cluster compute/import target.
`experiment_index/finalized_runs/<workflow_id>` maps each stage to a
collection of concrete clip-local run paths. Compatibility readers should learn
to resolve either a traditional top-level run group or a finalized clip
collection. Materialized global concatenated arrays are optional exports, not
the default durable write path.

The prototype shell creator is
`scripts/py -m fisheye.utils.create_clipped_analysis_zarr`. It creates
metadata-only clip-camera namespaces and parent placeholders; it does not run
analysis stages, import run-group artifacts, or update registry projections.

Temporal stages require an explicit boundary policy before they can be treated
as clip-local. Detection, crop geometry, keypoint inference, and mask inference
are image-local and fit this layout directly. Track kinematics, bout detection,
smoothing, and temporal state-machine stages need parent-wide finalization or
documented overlap/state handoff.

Dish masks in clipped archives are camera/static spatial metadata. Orange
acquisition guarantees fixed dish location and fixed camera geometry within a
recording, so one dish mask per `(recording_id, camera_serial)` applies to all
clips from that camera. The mask should be inherited from the source
recording/camera. The mask payload may retain the source `tuned_on_array`
coordinate system such as `images_ds`; consumers should use normalized mask
metrics when applying it to normalized detections. It is not per-clip mutable
review state.

Run-row identity is scoped to a concrete run path. In clipped archives,
`refined_row_ids` and downstream `source_refined_row_ids` must be interpreted
with the owning clip-camera run path, for example
`(source_refined_run_path, source_refined_row_id)`. Do not treat row-id
integers from different clip-local runs as globally unique.

See `docs/orange_rolling_clip_recording_contract.md` and
`docs/cluster_run_group_artifact_workflow.md` for the storage design. See
`docs/clipped_recording_consumer_mapping_contract.md` for the reader-facing
frame-index semantics and Crimson impact.

---

## String/Text Encoding Conventions

String-like data should follow these conventions across runtime writers:

1. Reason/status labels that must be TensorStore/C++ compatible:
- Canonical encoding: `reason_bytes` as `uint8[N,width]`, null-terminated UTF-8.
- New writers, review updates, republishers, and compactors must not create or
  maintain a variable-length `reason` mirror.
- Required attrs when reason bytes are present:
  - `reason_encoding="utf8-null-terminated"`
  - `reason_authority="reason_bytes"`
  - `reason_bytes_width=<int>`
  - `reason_bytes_null_terminated=true`
  - `reason_fallback_order=["reason_bytes","detection_source"]`

2. General text columns/arrays:
- Canonical runtime encoding is variable-length UTF-8.
- Avoid new fixed-width Unicode writes (`<U...`) in runtime code paths.

3. Read compatibility:
- Readers should tolerate legacy fixed-width string arrays where they exist.
- Historical read compatibility remains `reason_bytes` -> `reason` -> labels
  derived from `detection_source`. The legacy `reason` step is a reader
  fallback, not a current write contract.

See also: `docs/zarr_string_encoding_todo.md`.

---

## `raw_video/`

Arrays written during import (kvikIO or standard path):

| Array | Shape | DType | Notes |
| ----- | ----- | ----- | ----- |
| `images_full` | `(n_frames, H, W)` | `uint8` | Full-resolution frames (optional) |
| `images_ds` | `(n_frames, H_ds, W_ds)` | `uint8` | Downsampled frames (optional) |
| `images_ds_rgb` | `(n_frames, H_ds, W_ds, 3)` | `uint8` | Downsampled RGB frames (optional) |
| `original_frame_indices` | `(n_import_frames,)` | `int32` | Present for sampled imports (`frame_step > 1`) |
| `timestamps` | `(n_frames,)` | `float64` | Historical/derived compatibility surface; active sampled PyNvVC imports omit it in favor of the acquisition-frame-clock authority |

Attributes include import method, device, chunk/shard sizes, duration,
throughput, and source video metadata.

`raw_video/original_frame_indices` is an array-level mapping for sampled
imports. It is useful because it is compact and aligned to imported frames, but
it is not a complete recording provenance table.

Clipped training imports should also carry root attrs:
`source_layout="rolling_clips"`,
`source_frame_index_path="source_frame_index.parquet"`,
`source_frame_index_schema="palette.training_source_frame_index.v1"`, and
`source_recording_frame_index_path=<recording-root frame index>`. The registry
stores these attrs on `datasets` and exposes them through
`dataset_context_current` / `query_datasets()`. Registry-driven detection
training preparation uses them, together with a fingerprint of
`raw_video/original_frame_indices`, to prefer clipped replacements over
original full-video sampled training Zarrs when both represent the same parent
frames.

For clipped training Zarrs, `raw_video/original_frame_indices` should contain
`parent_frame_index` values. Stage-level `frame_indices` in that training Zarr
remain sample-local indices into `raw_video/images_*`; consumers that need the
exact source MP4 and local frame should read the row-aligned
`source_frame_index.parquet` sidecar.

Put differently, `original_frame_indices` is the clipped training Zarr's
compatibility bridge to the parent recording timeline. It is not the full
clip-level map. The clip-level map is either the recording-root
`recording_frame_index.parquet` or the training-local sampled
`source_frame_index.parquet`.

For full recording-level frame provenance, Palette should use a sidecar table
such as `recording_frame_index.parquet` plus a small manifest or root attrs
that point to it. This applies to both future clipped recordings and current
single-video recordings. In single-video archives, the local source-frame index
is normally equal to `parent_frame_index`; in clipped archives, the table maps
`parent_frame_index` to `(clip_id, clip_local_frame_index, video_path)`.

Keep large row-oriented metadata out of Zarr attrs. Use Zarr for arrays and
Parquet for frame-index/query tables.

The frame-index sidecar is not a review ledger. Mutable edits, review status,
stale-state detection, source fingerprints, and downstream lineage remain in
Zarr run groups and derived registry/finalize views. The frame index should be
safe to regenerate from recording metadata without losing scientific curation.

---

## `background_runs/`

Run arrays vary by method; common layout:

| Array | Shape | DType | Notes |
| ----- | ----- | ----- | ----- |
| `background_full` | `(H, W)` | `uint8` | Full-resolution background |
| `background_ds` | `(H_ds, W_ds)` | `uint8` | Downsampled background |
| `frame_indices` | `(n_samples,)` | `int32` | Frames sampled for background |

Attributes: `method`, `parameters`, `num_samples`, `duration_seconds`,
provenance, GPU/system info.

---

## `detect_runs/`

Outputs from blob/YOLO detection stages.

| Array | Shape | DType | Notes |
| ----- | ----- | ----- | ----- |
| `frame_indices` | `(n_detections,)` | `int32` | Corresponding frame per detection |
| `frame_counts` | `(n_frames,)` | `int32` | Number of detections per frame |
| `n_detections` | `(n_frames,)` | `int32` | Alias of `frame_counts` (kept for legacy consumers) |
| `bbox_norm_coords` | `(n_detections, 4)` | `float32/float64` | Normalized `[cx, cy, w, h]`; current YOLO writer uses `float64` |
| `scores` | `(n_detections,)` | `float32` | Confidence scores |
| `class_ids` | `(n_detections,)` | `int32` | Detector class labels |
| `instance_key` | `(n_detections,)` | `uint64` | Stable content-derived detection identity |
| `centers_px` *(optional)* | `(n_detections, 2)` | `float32` | Pixel centers (blob) |

Attributes store detector `method`, model identifiers, thresholds, duration,
and upstream background information. Standard values:

| Method | When used |
| ------ | --------- |
| `blob` | Traditional background-subtraction detector |
| `yolo_detect` | YOLO object detector |
| `yolo_pose` | YOLO pose detector |

YOLO detection storage note:

- the serial YOLO detector defaults to indexed Zarr v3 shards while retaining
  its existing inner chunk grid; the default detection- and frame-domain outer
  row targets are both `131072`
- detection inference already materializes the complete result table before
  saving, so the sharded path writes complete outer shards directly and does
  not allocate a second streaming buffer
- `frame_indices`, `bbox_norm_coords`, `scores`, `class_ids`, and
  `instance_key` share the detection-row grid; `frame_counts` and
  `n_detections` share the independent frame-row grid
- the writer rereads all seven arrays and requires exact decoded SHA-256 parity
  before completion
- runs record `detect_storage_layout`, `detect_storage_policy`,
  `detect_row_shard_rows`, `detect_frame_shard_rows`, and
  `detect_shard_write` in attrs/provenance
- CUDA runs record the PyTorch allocator high-water mark under
  `timing_summary.pytorch_cuda_peak_memory` and in query-friendly
  `pytorch_cuda_peak_*` attrs. This measures PyTorch allocated/reserved memory,
  not CUDA-context, NVDEC, driver, or other non-PyTorch device allocations
- use `--no-detect-sharding` for an explicit ordinary-chunk compatibility or
  benchmark run; blob/traditional detection writers remain unchanged

---

## `crop_runs/`

Each run stores crop-stage geometry/provenance and, in materialized mode, the
cropped ROI tensors needed by downstream consumers.

| Array | Shape | DType | Notes |
| ----- | ----- | ----- | ----- |
| `roi_images` | `(n_rois, h, w)` | `uint8` | Cropped grayscale patches. Required for future ordinary runs; absent only on historical `geometry_only` compatibility runs. |
| `roi_coordinates_full` | `(n_rois, 2)` | `int32` | Source-camera `points_xy` compatibility surface for each ROI top-left. Canonical ordinary runs bind it to the observation `instance_key` rows and exact `source_crop_xywh[:, :2]` payload with a digest-checked derivation; it is directly suitable for source-camera overlay. |
| `roi_coordinates_ds` *(historical only)* | `(n_rois, 2)` | `int32` | Ambiguous downsampled offsets retained only for legacy reads; future ordinary writers do not reconstruct them from a resolution ratio. |
| `instance_key` | `(n_rois,)` | `uint64` | Exact observation identity copied from the selected canonical detection rowset. |
| `source_acquisition_frame_index` | `(n_rois,)` | `int64` | Exact source-acquisition frame identity. |
| `bbox_norm_coords` | `(n_rois, 4)` | source floating dtype | Exact normalized detector geometry (`[cx, cy, w, h]`) copied from the source. |
| `bbox_img_xyxy` | `(n_rois, 4)` | source floating dtype | Exact source-camera bounding box. |
| `centers_img_xy` | `(n_rois, 2)` | source floating dtype | Exact source-camera center. |
| `source_crop_xywh` | `(n_rois, 4)` | source floating dtype | Source-camera placement of each materialized ROI. |
| `bbox_roi_xyxy` | `(n_rois, 4)` | source floating dtype | ROI-local bbox with an explicit ROI-to-source-camera transform. |
| `frame_indices` | `(n_rois,)` | `int64` | Decode index; exactly equal to `source_acquisition_frame_index` for admitted sources. |
| `frame_counts` | `(n_frames,)` | `int32` | Count of ROIs per frame |
| `detection_source` *(historical only)* | `(n_rois,)` | `int8` | Legacy/support label for sparse/refined rows. |
| `detection_indices` | `(n_rois,)` | `int64` | Exact physical row index into the admitted `detect_runs/<run>` rowset. |
| `source_refined_row_ids` *(optional)* | `(n_rois,)` | `int64` | Stable logical refined-detection row IDs copied from `refined_detect_runs/<run>/instances/refined_row_ids` |
| `source_detect_row_index` *(optional)* | `(n_rois,)` | `int32` | Raw detect row lineage copied from refined instances when available; `-1` for manual rows |

Attributes:

- `source_detect_run`, `source_background_run`, `detection_source_type`,
  `detection_source_path`, `includes_interpolated`, `n_real_detections`,
  `n_interpolated_detections`, ROI size, scaling factors.
- `crop_storage_mode` declares whether the run is `materialized` or
  `geometry_only`.
- `detect_review_status` (snapshot of refined review status when crop ran)
- `detect_review_status_ref` (refined run path where review status lives)
- `detection_selection_policy` (policy label used for auto source selection)
- `crop_signature` (signature of crop inputs: source path/type, ROI size, parameters hash)
- `roi_image_representation`, `roi_pixel_contract`, and
  `roi_pixel_contract_name` describe the model-facing ROI pixel surface. The
  current representation is `uint8_grayscale_roi_v1`; the structured contract
  records the conversion path, for example OpenCV BGR-to-gray, Decord channel
  mean, geometry-only deferred pixels, or PyNv flat-cache luma. The scalar
  contract name is duplicated intentionally for registry/query/export filters.
- `crop_review_status` (review status payload for this crop run, optional)
- `crop_review_signature` (signature snapshot stored when crop review was set)
- `source_refined_row_ids_available`, `source_refined_row_id_policy`, and
  `source_detect_row_index_available` describe optional row-identity lineage.
- `summary_statistics` (frames with crops, total ROIs, percentage coverage).
- GPU/environment provenance.

Parent-group pointer semantics for historical mixed-mode archives:

- `crop_runs.attrs["latest"]` remains materialized-compatible for backward
  compatibility.
- `crop_runs.attrs["latest_materialized"]` tracks the latest materialized run.
- `crop_runs.attrs["latest_any"]` tracks the latest run regardless of storage
  mode.

Future ordinary-writer policy:

- Direct, batch, Palette CLI, launcher, and flat-cache workflows default to
  materialized output from an exact `detect_runs/<run>` source.
- New ordinary crop runs reject `geometry_only` before run creation. Historical
  geometry-only groups remain explicit read-compatibility surfaces.
- New ordinary crop runs reject refined and legacy sparse sources until those
  producers publish an exact canonical row-selection contract.
- Canonical ordinary crop runs persist `roi_images`, instance identity,
  acquisition-frame identity, source-camera geometry, ROI-local geometry, and
  a direction-labelled ROI-to-source-camera transform.
- A new run remains selector-ineligible while it is written and marked
  complete. Palette freshly reloads the full persisted ordinary-crop contract,
  publishes parent selectors while the candidate is still ineligible, and
  flips eligibility only as the final publication mutation.
- Many traditional/training/export consumers still require materialized
  `roi_images` even though mixed-mode readers now exist for some ROI-model
  workflows.
- New merged detection-training exports do not use `crop_runs` as the forward
  label authority. They write positive detection labels only to
  `refined_detect_runs/<run>/instances`; crop-run label arrays remain
  compatibility/support surfaces for per-recording and historical stores.

Source-resolution helpers can still inspect `refined`, `filtered`,
`interpolated`, `manual`, and `auto` surfaces for historical tooling. The
future ordinary writer admits only an exact `detect_runs/<run>` source and
records it in `detection_source_path`; it does not adapt a refined/sparse rowset
or silently fall back.

For `geometry_only` crop runs, `source_video_path` or embedded
`raw_video/images_full` must remain readable from the environment where ROI
pixels will be reconstructed. Copied analysis archives should update copied
metadata paths to the cluster-visible source video before crop/cache jobs run.

`detection_indices` is intentionally an ordinal mapping into the resolved source
rowset, not stable logical identity. When crops are sourced from canonical
refined detections, consumers that need row-local stale repair should use
`source_refined_row_ids`; `source_detect_row_index` is raw-candidate lineage and
may be `-1` for manual additions.

---

## `keypoints_runs/`

Produced by the keypoint detection stage (traditional or YOLO-based).

| Array | Shape | DType | Notes |
| ----- | ----- | ----- | ----- |
| `frame_indices` | `(n_rois,)` | `int32` | Inherit from corresponding crops |
| `frame_counts` | `(n_frames,)` | `int32` | ROIs per frame (mirrors `n_rois`) |
| `n_rois` | `(n_frames,)` | `int32` | Alias maintained for legacy callers |
| `detection_indices` | `(n_rois,)` | `int32` | Copied from source crop run |
| `source_refined_row_ids` *(optional)* | `(n_rois,)` | `int64` | Copied from source crop run when refined detect lineage exists |
| `source_detect_row_index` *(optional)* | `(n_rois,)` | `int32` | Copied from source crop run when raw detect lineage exists |
| `keypoints_roi` | `(n_rois, n_keypoints, 2)` | `float64` | Coordinates in ROI pixels |
| `keypoints_img` | `(n_rois, n_keypoints, 2)` | `float64` | Full-image pixels |
| `keypoints_norm` | `(n_rois, n_keypoints, 2)` | `float64` | Normalized [0,1] |
| `heading` | `(n_rois,)` | `float64` | Degrees, NaN when unavailable |
| `confidence` | `(n_rois,)` | `float64` | Overall score |
| `keypoint_confidences` | `(n_rois, n_keypoints)` | `float64` | Per-keypoint confidences in `keypoint_labels` order |
| `effective_threshold` | `(n_rois,)` | `float64` | Per-ROI threshold used |
| `effective_se2_radius` | `(n_rois,)` | `float64` | Search radius actually applied |
| `detection_success` | `(n_rois,)` | `bool` | True if keypoints converged |
| `detection_source` | `(n_rois,)` | `int8` | 0=real, 1=interpolated (from crop source) |
| `heading_finite` | `(n_rois,)` | `bool` | True when `heading` is finite |
| `heading_usable` | `(n_rois,)` | `bool` | True when source is real, detection succeeded, and heading is finite |
| `n_keypoints` | `(n_frames,)` | `int32` | Successful keypoints per frame |
| `triangle_angles` | `(n_rois, 3)` | `float64` | Triangle angles in canonical order (swim_bladder, left, right) |
| `triangle_angles_raw` | `(n_rois, 3)` | `float64` | Triangle angles in candidate order (largest -> smallest blob) |
| `triangle_area` | `(n_rois,)` | `float64` | Triangle area (pixels^2) |

Attributes: `source_crop_run`, `source_crop_storage_mode`,
`source_crop_signature`, `source_crop_revision`,
`source_detect_review_status_ref`, `source_background_run`,
`source_detect_run`, `source_refined_run` (if available), `method`,
`parameter_source`, `parameters`, `skeleton_id`, `kpt_shape`, `pose_schema`, `keypoint_labels`,
`keypoint_confidence_labels`, `triangle_angle_order`,
`triangle_angle_raw_order`, `heading_computation_override`,
requested/resolved device metadata, scheduler placement, timing, QA summaries.

Cluster/device provenance note:

- GPU keypoint runs should prefer explicit device selection. The cluster
  submitter uses `--gpus N` to request LSF GPUs and defaults to `--device 0`
  when a GPU is requested and no device override is supplied.
- Run attrs and `attrs["provenance"]` record `requested_device`,
  `normalized_torch_device`, `initial_model_device`, and
  `resolved_model_device`. Some engine formats cannot introspect a final model
  parameter device; in that case `resolved_model_device` may fall back to the
  normalized/requested device.
- Cluster placement attrs include `execution_hostname`, `scheduler`,
  `scheduler_job_id`, `scheduler_job_name`, `scheduler_job_index`,
  `scheduler_queue`, `scheduler_hosts`, `scheduler_mcpu_hosts`,
  `scheduler_cuda_visible_devices`, and `scheduler_gpu_request`.
- Explicit flat-cache runs include `source_roi_cache_staging_policy` at the
  run-attr level and `roi_cache_staging_policy` in parameters/provenance/status
  details. Current policies are `node_scratch_staged_flat_cache` and
  `direct_manifest_read`.
- The same compact device/scheduler fields are mirrored into registry status
  details for performance triage without opening the zarr store.

Keypoint storage note:

- the canonical datastore contract remains dense numeric arrays such as
  `keypoints_roi (N, K, 2)` and `keypoint_confidences (N, K)`
- semantic meaning comes from `keypoint_labels` and `pose_schema`, not from
  hard-coded positional assumptions
- consumers that need specific landmarks should build a label-to-index helper
  view at runtime and then read from the dense arrays
- per-row key/value keypoint storage is not the Palette datastore contract
- fixed-width triangle diagnostic arrays are compatibility/QC outputs for the
  traditional triangle and are not the general skeleton geometry contract
- the serial YOLO writer defaults to indexed Zarr v3 shards while retaining its
  existing inner chunk grid: ROI outer rows default to `131072` and frame-domain
  outer rows default to `131072`; use `--no-keypoint-sharding` for an explicit
  ordinary-chunk compatibility or benchmark run
- sharded YOLO writes use exactly two buffers and write complete outer shards;
  the inference-produced ROI arrays share one aligned outer grid, copied ROI
  lineage uses that same grid, and frame-count arrays use the independent frame
  grid
- variable-width string arrays remain ordinarily chunked rather than sharded
- runs record `keypoint_storage_layout`, `keypoint_storage_policy`,
  `keypoint_roi_shard_rows`, `keypoint_frame_shard_rows`, and a
  `keypoint_shard_write` validation summary in run attrs/provenance
- `keypoint_storage_policy` distinguishes `default_indexed_sharding_v1` from
  `explicit_regular_chunks_override`
- completed refined keypoint outputs may be published as immutable
  `131072`-row indexed-sharded snapshots; sparse review changes belong in
  `edit_delta_runs` partitions and are compacted into a new snapshot
- strictly immutable publication jobs may explicitly choose `262144` rows per
  shard when minimizing object count is more important than smaller maintenance
  rewrite units
- both ordinary working runs and sharded snapshot runs use the same Zarr array API
- traditional/Dask writers must not concurrently write disjoint logical slices
  inside the same physical shard

Skeleton-identity metadata note:

- new keypoint runs are expected to persist explicit `skeleton_id` and
  `kpt_shape` attrs, not only `pose_schema`
- readers should resolve skeleton identity in this order:
  1. explicit run attr `skeleton_id`
  2. `pose_schema.skeleton_id`
  3. fallback `pose_schema:<name>`
- historical archives can be normalized with
  `fisheye.utils.backfill_keypoint_skeleton_attrs`

Heading metadata note:

- `pose_schema.metadata.heading_computation` is the canonical heading
  definition for the skeleton.
- `heading_computation_override` is an optional run-level override/disable
  payload.
- Readers should resolve in this order:
  1. run attr `heading_computation_override`
  2. `pose_schema.metadata.heading_computation`
  3. deprecated run attr `heading_computation`
  4. heading semantics unavailable
- See `docs/keypoint_heading_computation_contract.md`.

---

## `detect_runs/<run>/quality_reports/<qrun>/`

Produced by `fisheye.refinement.detect_quality`.
This is a raw detect artifact-label surface used by `refine_detect`; it is not
the refined detect review/approval surface.

| Array | Shape | DType | Notes |
| ----- | ----- | ----- | ----- |
| `quality_flags` | `(n_frames,)` | `int8` | Frame-level quality labels |
| `detection_quality_labels` | `(n_detections,)` | `int8` | Detection-level quality labels |

Attributes include thresholds/modes used for artifact detection, summary counts
of jumps/blips/gaps, and provenance references for the analyzed detect run.

---

## `eye_masks_runs/`

Generated by segmentation inference (`infer_unet_eye_masks.py`,
`eye_segmentation_yolo.py`, etc.).
Row lineage (`frame_indices`, `detection_indices`, `frame_counts`,
`source_refined_row_ids`, `source_detect_row_index`) follows:
`docs/archive/eye_mask_row_mapping_contract.md`.

| Array | Shape | DType | Notes |
| ----- | ----- | ----- | ----- |
| `frame_indices` *(legacy-optional)* | `(n_rois,)` | `int32` | ROI frame mapping (new runs should copy from source crop run; missing in some legacy runs) |
| `frame_counts` *(legacy-optional)* | `(n_frames,)` | `int32` | Count of ROIs per frame (new runs should match source crop run) |
| `detection_indices` *(legacy-optional)* | `(n_rois,)` | `int32` | Index into source detect rows via `crop_runs/<run>/detection_indices` |
| `source_refined_row_ids` *(optional)* | `(n_rois,)` | `int64` | Copied from source crop/keypoint lineage when present |
| `source_detect_row_index` *(optional)* | `(n_rois,)` | `int32` | Copied from source crop/keypoint lineage when present |
| `masks_roi` *(optional)* | `(n_rois, C, H, W)` | `uint8` | Binary ROI-local masks. `C=1` for union, `C=2` for left/right. |
| `mask_probs_roi` *(optional)* | `(n_rois, C, H, W)` | `float16/float32/uint8` | Semantic ROI-local probabilities in `[0,1]`; `C=1` for union, `C=2` for left/right. Analysis U-Net runs default to quantized `uint8`; if stored as `uint8`, decode with `p = stored / 255`. |
| `mask_scores` *(optional)* | `(n_rois,)` | `float32` | Confidence per ROI |
| `ellipse_params` | `(n_rois, 2, 5)` | `float32` | `[cx, cy, major, minor, angle]` in ROI pixels |
| `ellipse_success` | `(n_rois, 2)` | `bool` | Ellipse fit success per eye |
| `eye_separation` | `(n_rois,)` | `float32` | Centroid distance (ROI pixels) |
| `detection_source` | `(n_rois,)` | `int8` | 0=real, 1=interpolated (copied from crop run) |
| `contour_left_ptr` | `(n_rois,)` | `int32` | Pointer into `contours_left` |
| `contour_left_len` | `(n_rois,)` | `int32` | Number of points for left eye contour |
| `contour_right_ptr` | `(n_rois,)` | `int32` | Pointer into `contours_right` |
| `contour_right_len` | `(n_rois,)` | `int32` | Number of points for right eye contour |
| `contours_left` | `(n_points, 2)` | `float32` | Concatenated left eye contours (x, y) |
| `contours_right` | `(n_points, 2)` | `float32` | Concatenated right eye contours (x, y) |
| `reason` | `(n_rois,)` | `string` | Per-ROI labels (`clean`, `keypoint_fail`, `no_region`, `overlap`, `too_close`, `too_far`, `incomplete`) |

Attributes: `source_crop_run`, `source_crop_storage_mode`,
`source_crop_signature`, `source_crop_revision`,
`source_detect_review_status_ref`, canonical `source_keypoints_run`
*(legacy alias: `source_keypoint_run` may be present for migration
compatibility)*, `source_keypoint_group` (defaults to
`refined_keypoints_runs` when present), `method`,
model info, thresholds, separation limits, `successful_eyes`,
`successful_roi_pairs`, `reason_counts`, `ellipse_angle_units`,
`ellipse_fit_backend` (currently `opencv` for refined runs),
`ellipse_fit_method` (currently `cv2.fitEllipse` for refined runs),
`ellipse_contour_mode` (currently `external_pixel_contour` for refined runs), `eye_labels`, `segmenter_label_mode`,
`mask_probs_chunk_rois`, `duration_seconds`,
optional threshold-calibration metadata:
- `recommended_probability_threshold`
- `recommended_probability_threshold_review`

Probability-storage attrs when `mask_probs_roi` exists:

- `probabilities_dtype`: physical storage dtype for `mask_probs_roi`
- `probabilities_encoding`:
  - `unit_float` for direct float probabilities
  - `linear_uint8_0_255` for quantized `uint8` probabilities

Reader contract:

- `mask_probs_roi` is ROI-local, not full-frame.
- `masks_roi` is ROI-local, not full-frame.
- Readers should interpret the dataset semantically as probabilities in
  `[0,1]` regardless of physical dtype.
- Full-frame placement should be derived from `source_crop_run` geometry.
- Analysis-oriented runs may use `segmenter_label_mode="union"` with a single
  channel and let refinement assign left/right identity later.

Lineage compatibility policy for eye-mask runs:

- New runs should treat `source_keypoints_run` as required canonical lineage.
- Legacy `source_keypoint_run` is read-compatible fallback only.
- Diagnostics/readers resolve canonical first, then legacy fallback.

---

## `refined_detect_runs/`

Created by `fisheye.refinement.refine_detect`.

Each refined run contains the canonical curated detect surface. Primary writes
land on the sparse refined subgroups. Shared readers should prefer the
`instances/` sparse surface when it exists.

### Run-level attributes

Common attrs on `refined_detect_runs/<run>`:
- `source_detect_run`, `source_quality_run`
- `refinement_timestamp`, `processing_time_seconds`
- `operations`
- `parameters` (includes curated selection policy such as `dish_mask_gate` and
  `top_k_selection` when used)
- `coverage_frames_total` (frame universe used for coverage percent)
- `coverage_frame_source` (`full` or `sampled`)
- `coverage_frames_full` (full frame count when sampled coverage is used)
- `detect_review_status` (review metadata dict; see below)
- `summary_statistics`
- `curated_row_storage`
- `entity_assignment_policy`
- `coordinate_space`
- `row_identity_policy`
- status/source/review/artifact code maps
- provenance/environment metadata

Coordinate attrs on current refined runs and their `instances/` /
`source_detections/` surfaces:

- `bbox_img_xyxy_coordinate_space`: `source_image_xyxy`
- `bbox_img_xyxy_reference_width`, `bbox_img_xyxy_reference_height`
- `bbox_norm_coords_format`: `cxcywh`
- `bbox_norm_coords_coordinate_space`: `normalized`
- `bbox_norm_reference_width`, `bbox_norm_reference_height`
- `bbox_norm_reference_space`: usually `inference_image` for YOLO-generated
  detections, or `source_image` when normalized boxes are source-frame relative
- `bbox_coordinate_contract_version`: current writer value is
  `refined_detect_bbox_coordinates_v2`

`coordinate_space = "full_image_xyxy"` is retained as a legacy run-level alias.
Consumers should prefer the explicit `bbox_*` attrs above.

Parent attrs on `refined_detect_runs/`:
- `latest`
- `detect_review_status_latest` (historical detect-review lineage pointer written by migration tools; no reader consults it for current run resolution)

`detect_review_status` payload fields (may be extended over time):
- `state` (e.g., approved/needs_review)
- `method` (manual/algorithmic/hybrid/spotcheck)
- `intended_use` (training/analysis/etc.)
- `timestamp`
- `resolved_group` (`refined` for current runs; legacy runs may still use manual/interpolated/filtered/raw)
- `preference_chain` (ordered list used for resolution)
- optional `reviewer`, `notes`

### Metadata-Only Run Root

The run root is now metadata-only for current sparse refined detect runs.
Canonical bbox data lives in the `instances/` and `source_detections/`
subgroups below.

Explicit downstream overrides should target `instances/` when they need a
stable current-run curated bbox path.

Current review/runtime note:

- `fisheye.tune.detect_review` edits the sparse refined surfaces directly
- when fixed sub-arena ROI definitions are present, review operates on one slot
  per `(frame, arena_id)`
- unconstrained multiple curated detections inside the same arena/ROI are not
  yet supported by the manual detect-review UI
- `ambiguous` in dense/single-slot views means the frame cannot be represented
  as one obvious detection, usually because multiple source candidates or
  multiple curated instances exist for that frame

### Sparse Curated Read Surfaces

When present, the active curated refined-detect surfaces are:

- `refined_detect_runs/<run>/instances`
- `refined_detect_runs/<run>/source_detections`

#### `instances/`

Primary curated bbox read surface for current runs.

Required arrays:

- `refined_row_ids`
- `frame_indices`
- `frame_offsets`
- `bbox_img_xyxy`: source-image pixel-space `[x1, y1, x2, y2]`
- `bbox_norm_coords`: normalized `[cx, cy, w, h]`; reference dimensions are
  recorded in `bbox_norm_reference_width` / `bbox_norm_reference_height`
- `source_kind_codes`
- `manual_edit_flags`

Common optional arrays:

- `instance_key`: stable observation identity; required for modern keyed runs
- `instance_key_origin_codes`: `0` for copied detector identity and `1` for a
  key minted when a manual observation is first created
- `row_identity_mode`: run attr set to `instance_key` for modern keyed runs or
  `legacy_positional` for explicitly labeled historical compatibility
- `row_identity_mode_schema`: run attr set to `palette.row_identity_mode.v1`
  whenever `row_identity_mode` is present
- `confidence_scores`
- `class_ids`
- `source_detect_row_index`
- `reason_bytes`
- `review_notes`

Historical runs may additionally contain a variable-length `reason` mirror;
readers may consume it only when `reason_bytes` is absent.

Reader rule:

- rows in `instances/` are already the curated accepted detections; render only
  rows with finite bbox geometry
- `refined_row_ids` are stable logical row identity and must not be treated as
  physical row positions or biological identity
- in clipped analysis archives, `refined_row_ids` are local to the concrete
  clip-camera refined run; parent-wide consumers must pair them with the run
  path or finalized collection identity
- current sparse rows should be sorted by `frame_indices` then
  `refined_row_ids`; `frame_offsets` and `frame_counts` must match that order
- surviving `refined_row_ids` retain their stored `instance_key` across bbox,
  class, review, note, and physical-order changes
- `next_refined_row_id` on the refined run is the monotonic allocator high-water
  mark; IDs below it that are absent from `instances/` are retired and must not
  be reused
- a new manual observation uses its newly allocated `refined_row_id` in the
  key-minting namespace; split and merge outputs therefore receive fresh row
  IDs and fresh keys

#### `source_detections/`

Candidate-audit surface mirroring the exact bound raw detect rowset for current
refined runs.

Required arrays:

- `source_detect_row_index`
- `frame_indices`
- `bbox_norm_coords`: normalized `[cx, cy, w, h]`; reference dimensions are
  recorded in `bbox_norm_reference_width` / `bbox_norm_reference_height`
- `decision_codes`
- `resolved_refined_row_id`

Common optional arrays:

- `bbox_img_xyxy`: source-image pixel-space `[x1, y1, x2, y2]`
- `confidence_scores`
- `class_ids`
- `reason_bytes`
- `review_notes`

Historical runs may additionally contain a variable-length `reason` mirror;
readers may consume it only when `reason_bytes` is absent.

Reader rule:

- treat `source_detections/` as an audit/provenance surface, not the primary
  bbox render surface
- accepted source rows should resolve to current `instances/refined_row_ids`
  values; stale or missing mappings mean row-local downstream repair is unsafe

### Legacy Sparse Subgroups

Older archives may still contain sparse subgroups such as:

- `filtered`
- `interpolated`
- `manual`
- `manual_*`

These remain compatibility/provenance artifacts for legacy runs. They are no
longer the primary detect contract for new runs.

---

## `refined_keypoints_runs/`

Outputs from `fisheye.refinement.refine_keypoints`.

| Array | Shape | DType | Notes |
| ----- | ----- | ----- | ----- |
| `frame_indices` | `(n_rois,)` | `int32` | Copied from source keypoint run |
| `frame_counts` | `(n_frames,)` | `int32` | Copied from source keypoint run |
| `n_rois` | `(n_frames,)` | `int32` | Alias maintained for legacy callers |
| `detection_indices` *(optional)* | `(n_rois,)` | `int32` | Copied from source when present |
| `source_refined_row_ids` *(optional)* | `(n_rois,)` | `int64` | Copied from source when present |
| `source_detect_row_index` *(optional)* | `(n_rois,)` | `int32` | Copied from source when present |
| `detection_source` | `(n_rois,)` | `int8` | 0=real, 1=interpolated |
| `retune_id` | `(n_rois,)` | `int32` | Batch retune parameter set label (`-1` = none) |
| `keypoints_roi` | `(n_rois, n_keypoints, 2)` | `float64` | Refined keypoints (ROI pixels) |
| `keypoints_img` | `(n_rois, n_keypoints, 2)` | `float64` | Refined keypoints (full image) |
| `keypoints_norm` | `(n_rois, n_keypoints, 2)` | `float64` | Refined keypoints (normalized) |
| `heading` | `(n_rois,)` | `float64` | Heading after refinement |
| `confidence` | `(n_rois,)` | `float64` | Overall score (copied from source) |
| `keypoint_confidences` *(optional)* | `(n_rois, n_keypoints)` | `float64` | Per-keypoint confidences in `keypoint_labels` order |
| `effective_threshold` *(optional)* | `(n_rois,)` | `float64` | Copied from source if present |
| `effective_se2_radius` *(optional)* | `(n_rois,)` | `float64` | Copied from source if present |
| `triangle_area` | `(n_rois,)` | `float64` | Triangle area |
| `min_angle` | `(n_rois,)` | `float64` | Minimum triangle angle (deg) |
| `triangle_angles` | `(n_rois, 3)` | `float64` | Triangle angles (deg) in canonical order |
| `quality_labels` | `(n_rois,)` | `int8` | 0=clean, 4=source_failed, 6=flip_corrected |
| `refined_success` | `(n_rois,)` | `bool` | Refinement executed successfully |
| `source_success` | `(n_rois,)` | `bool` | Source keypoint success mask |
| `flip_corrected` | `(n_rois,)` | `bool` | True if left/right eyes were swapped |
| `heading_finite` | `(n_rois,)` | `bool` | True when `heading` is finite |
| `heading_usable` | `(n_rois,)` | `bool` | True when refined succeeded, source is real, and heading is finite |
| `heading_delta_prev_deg` *(optional)* | `(n_rois,)` | `float32` | Circular absolute heading delta to previous usable temporal neighbor; omitted when temporal-heading review is disabled |
| `heading_delta_next_deg` *(optional)* | `(n_rois,)` | `float32` | Circular absolute heading delta to next usable temporal neighbor; omitted when temporal-heading review is disabled |
| `heading_temporal_outlier` *(optional)* | `(n_rois,)` | `bool` | True when both temporal deltas exceed the configured outlier threshold; omitted when temporal-heading review is disabled |
| `confidence_valid` | `(n_rois,)` | `bool` | All per-keypoint confidences >= threshold |
| `geometry_valid` | `(n_rois,)` | `bool` | Triangle angle/area pass thresholds |
| `usable_keypoints` | `(n_rois,)` | `bool` | Confidence + geometry valid |
| `reason_bytes` | `(n_rois, width)` | `uint8` | Null-terminated UTF-8 reason labels (TensorStore-safe primary encoding) |
| `reason` | `(n_rois,)` | `string` | Pipe-delimited tags (e.g., `flip_corrected|geometry_issue`) |
| `failure_indices` | `(n_failures,)` | `int32` | ROI indices where source keypoints failed |

Attributes: `source_keypoints_run`, `source_crop_run`,
`source_crop_storage_mode`, `source_crop_signature`,
`source_crop_revision`, `source_detect_review_status_ref`,
`source_detect_run`, `skeleton_id`, `kpt_shape`, `pose_schema`,
`heading_computation_override`, `derived_metrics_schema`,
refinement parameters (thresholds),
`summary_statistics`, `retune_params`, `keypoint_signature`,
`keypoint_review_status`, `keypoint_review_signature`, scheduler config,
environment/provenance metadata.

Keypoint storage note:

- the canonical datastore contract remains dense numeric arrays such as
  `keypoints_roi (N, K, 2)` and `keypoint_confidences (N, K)`
- semantic meaning comes from `keypoint_labels` and `pose_schema`, not from
  hard-coded positional assumptions
- consumers that need specific landmarks should build a label-to-index helper
  view at runtime and then read from the dense arrays
- per-row key/value keypoint storage is not the Palette datastore contract
- fixed-width triangle diagnostic arrays are compatibility/QC outputs for the
  traditional triangle and are not the general skeleton geometry contract

Skeleton-identity metadata note:

- new refined keypoint runs are expected to persist explicit `skeleton_id` and
  `kpt_shape` attrs, not only `pose_schema`
- readers should resolve skeleton identity in this order:
  1. explicit run attr `skeleton_id`
  2. `pose_schema.skeleton_id`
  3. fallback `pose_schema:<name>`
- historical archives can be normalized with
  `fisheye.utils.backfill_keypoint_skeleton_attrs`

Heading metadata note:

- `pose_schema.metadata.heading_computation` is the canonical heading
  definition for the skeleton.
- `heading_computation_override` is an optional run-level override/disable
  payload for exceptional cases.
- Readers should resolve in this order:
  1. run attr `heading_computation_override`
  2. `pose_schema.metadata.heading_computation`
  3. deprecated run attr `heading_computation`
  4. unavailable
- If the override payload exists and `enabled=false`, that disables heading
  semantics for the run even if `pose_schema.metadata.heading_computation`
  exists.
- See `docs/keypoint_heading_computation_contract.md`.

Derived-metrics metadata note:

- `derived_metrics_schema` is a run-level semantic contract for derived arrays
  and boolean/status gates.
- For current refined keypoint runs, it declares the triangle geometry metric
  semantics behind `triangle_area`, `triangle_angles`, `min_angle`, and
  `geometry_valid`.
- It is separate from the entity schema (`pose_schema`, `keypoint_labels`) and
  separate from heading semantics.
- See `docs/derived_metrics_schema_contract.md`.

Temporal-heading policy notes:

- Temporal-heading review is intended for temporally contiguous refined runs.
- Sampled imports (for example `raw_video/import_mode="sampled"` or archives
  with `original_frame_indices`) disable temporal-heading review.
- When disabled, the optional temporal arrays above are omitted and
  `summary_statistics.postprocess` records:
  - `temporal_heading_status`
  - `temporal_heading_disabled_reason`

Post-refinement coordinate-space diagnostics attrs (when audit is enabled):

- `post_refinement_audit_json`: absolute path to `<dataset>_audit.json`
- `post_refinement_audit_generated_utc`: UTC timestamp of audit generation
- `post_refinement_audit_status_counts`: status-count snapshot from the audit report

Optional overlap-analysis attrs (when overlap analysis is enabled):

- `post_refinement_overlap_json`: absolute path to `<dataset>_overlap.json`
- `post_refinement_overlap_generated_utc`: UTC timestamp of overlap report generation
- `post_refinement_overlap_bad_row_count`: bad-row count from overlap analysis

Reason-label attrs on refined keypoint runs:

- `reason_encoding="utf8-null-terminated"`
- `reason_authority="reason_bytes"`
- `reason_bytes_width=<int>`
- `reason_bytes_null_terminated=true`
- `reason_fallback_order=["reason_bytes","detection_source"]`

Current consumers should read `reason_bytes`, then derive labels from
`detection_source` when it is absent. Historical readers may insert legacy
`reason` between those steps for old archives only
(`0=clean`, `1=interpolated`).

`summary_statistics` is a dict with a refine-time snapshot plus optional
postprocess counts, e.g.:

- `summary_statistics.refine`: counts written by `refine_keypoints`
- `summary_statistics.postprocess`: counts recomputed after retune/manual review
- `summary_statistics.postprocess_updated_utc`: timestamp of the last audit

`retune_params` maps `retune_id` values to the keypoint-tuning parameter sets
applied during batch retuning.

---

## `refined_eye_masks_runs/`

Compatibility and historical eye-specific refined layout. Current canonical
reviewed eye geometry should be read from `refined_subject_masks_runs/<run>`
when that run contains `eye_left` and `eye_right` components with geometry
arrays. Active consumers that need old `(N, 2, ...)` arrays should use
`fisheye.shared.eye_geometry_source`, which adapts canonical refined-subject
geometry and falls back to this group for historical archives.

See `fisheye.refinement.refine_eye_masks`.  Key arrays:
Row lineage (`frame_indices`, `detection_indices`, `frame_counts`,
`source_refined_row_ids`, `source_detect_row_index`) is copied from
`source_eye_masks_run`; see `docs/archive/eye_mask_row_mapping_contract.md`.

| Array | Shape | DType | Notes |
| ----- | ----- | ----- | ----- |
| `masks_roi` | `(n_rois, 2, H, W)` | `uint8` | Refined masks |
| `ellipse_params` | `(n_rois, 2, 5)` | `float32` | `[cx, cy, major, minor, angle]` |
| `ellipse_success` | `(n_rois, 2)` | `bool` | Fit success per eye |
| `eye_separation` | `(n_rois,)` | `float32` | Centroid distance |
| `retune_id` *(optional)* | `(n_rois,)` | `int32` | Batch retune parameter set label (`-1` = none) |
| `frame_indices` *(legacy-optional)* | `(n_rois,)` | `int32` | Copied from source eye-mask run (new runs should include) |
| `frame_counts` *(legacy-optional)* | `(n_frames,)` | `int32` | Copied from source eye-mask run (new runs should include) |
| `detection_indices` *(legacy-optional)* | `(n_rois,)` | `int32` | Copied from source eye-mask run (new runs should include when upstream exists) |
| `source_refined_row_ids` *(optional)* | `(n_rois,)` | `int64` | Copied from source eye-mask run when present |
| `source_detect_row_index` *(optional)* | `(n_rois,)` | `int32` | Copied from source eye-mask run when present |
| `mask_probs_roi_refined` *(optional)* | `(n_rois, 2, H, W)` | `float16` | Optional refined ROI-local left/right probabilities in `[0,1]`; new refined runs are binary-first and only write this dataset when explicitly requested. |
| `contour_left_ptr` | `(n_rois,)` | `int32` | Pointer into `contours_left` |
| `contour_left_len` | `(n_rois,)` | `int32` | Number of points for left eye contour |
| `contour_right_ptr` | `(n_rois,)` | `int32` | Pointer into `contours_right` |
| `contour_right_len` | `(n_rois,)` | `int32` | Number of points for right eye contour |
| `contours_left` | `(n_points, 2)` | `float32` | Concatenated left eye contours (x, y) |
| `contours_right` | `(n_points, 2)` | `float32` | Concatenated right eye contours (x, y) |

`metrics/` subgroup:

- Per-eye area arrays: `area_refined`, `area_source`, `area_zscore`,
  `area_delta_vs_source`, `area_ratio_vs_source`.
- Eye-pair area/symmetry arrays: `area_union_refined`,
  `area_union_source`, `area_ratio_left_right`, `area_diff_left_right`,
  `area_union_delta`, `area_union_ratio`, `symmetry_sum`,
  `symmetry_abs_diff`.
- Geometry/QC arrays: `centroid_error`, `symmetry_offsets`,
  `separation_refined`, `separation_keypoint`, `separation_delta`,
  `axis_ratio`, `circularity`, `connectivity_flags`, `smoothing_flags`,
  `filter_flags`, `pixels_reassigned`.
- Probability arrays: `probabilities_used`, `probability_mean`,
  `probability_max`, `probability_var`, `probability_high_fraction`.
- Reason array: canonical `reason_bytes` (tags include `refined`,
  `copied_original`, `filtered_*`, `retuned`, `manual_correction`).
  Historical runs may retain a read-only `reason` mirror.

Attributes expose `metrics_summary`, configuration snapshots, per-eye filter
thresholds, `summary_statistics`, `retune_params`, and links to source runs
(`source_eye_masks_run`, `source_eye_masks_method`, `source_keypoint_group`,
canonical `source_keypoints_run` with optional legacy alias
`source_keypoint_run`, `source_crop_run`).
`traditional_fast_path=true` indicates masks/ellipses were copied from the
source (used for traditional segmentation unless
`force_refine_traditional=true`).

Important refined-output policy:

- `masks_roi` is the canonical artifact for this compatibility layout, not the
  canonical reviewed eye-geometry authority for modern unified subject-mask
  runs.
- `mask_probs_roi_refined` is optional debug/high-fidelity output.
- Refined runs record the threshold used to derive binary masks from
  probability inputs via `mask_probability_threshold`.
- Refined runs also record `mask_probability_threshold_source` describing
  whether that threshold came from:
  - explicit CLI input
  - source-run recommendation metadata
  - the default fallback

Semantic note:

- Refined eye-mask outputs remain ROI-local.
- A refined run may promote a raw union-mask source into explicit left/right
  channels using keypoint-informed refinement.

Refinement keypoint-source binding is strict by default:

- Resolution order is `--keypoint-run` -> source lineage attrs on source eye-mask run
  (`source_keypoint_group` + canonical/legacy keypoint-run attr) -> error.
- `--allow-latest-keypoint-fallback` enables legacy compatibility fallback and
  should be treated as temporary recovery behavior.

`summary_statistics` mirrors refined keypoints: the `refine` snapshot is written
by `refine_eye_masks`, and `postprocess` is updated by the review tooling
(`eye_mask_review --retune/--manual/--audit`). The postprocess stats include
manual correction counts, retune totals, and reason tag counts. `retune_params`
maps `retune_id` values to the parameter sets applied during batch retuning.

---

## `subject_mask_runs/`

Unified subject-mask runs produced by the traditional body segmenter, U-Net,
SAM/SAM2/SAM3 write-back, and swim-bladder refresh flows.

| Array | Shape | DType | Notes |
| ----- | ----- | ----- | ----- |
| `frame_indices` *(recommended)* | `(n_rois,)` | `int32` | Copied from the source crop run when available |
| `frame_counts` *(recommended)* | `(n_frames,)` | `int32` | Copied from the source crop run when available |
| `detection_indices` *(recommended)* | `(n_rois,)` | `int32` | Copied from the source crop run when available |
| `source_refined_row_ids` *(recommended)* | `(n_rois,)` | `int64` | Copied from the source crop run when available |
| `source_detect_row_index` *(recommended)* | `(n_rois,)` | `int32` | Copied from the source crop run when available |
| `detection_source` | `(n_rois,)` | `int8` | Expected to align with the source crop run |
| `masks_roi` *(optional)* | `(n_rois, C, H, W)` | `uint8` | Thresholded binary multilabel masks. Raw probability-first U-Net runs may omit this dense convenience copy and set `masks_roi_materialized=false`. |
| `mask_probs_roi` | `(n_rois, C, H, W)` | `float16/float32/uint8` | Decoded or quantized semantic probabilities in `[0,1]`; probability-first raw runs treat this as the canonical model output. |
| `available_channels` | `(C,)` | `bool` | Run-level declaration of which channels are semantically valid |

Compact binary mask storage is currently implemented for refined subject-mask
runs, not raw probability-first subject-mask runs. Raw subject-mask readers
should continue to treat `mask_probs_roi` as the canonical model-output surface
and dense `masks_roi` as an optional thresholded compatibility cache. The
shared compact-mask reader/writer design is documented in
`docs/mask_rle_storage_design_and_benchmark_plan.md`.

New U-Net runs store `mask_probs_roi` as Zarr v3 indexed shards by default,
using `32`-row independently readable inner chunks and `2,048`-row physical
shards. Two channel-major host buffers accumulate inference batches while one
background writer publishes each complete outer shard once. The writer hashes
source values before buffer reuse, rereads and exact-validates the completed
destination, and only then completes the run. `--mask-probs-shard-rois` may
select another valid outer size; `--no-mask-probs-sharding` is the explicit
ordinary-chunk compatibility override. Readers continue to address
`mask_probs_roi` normally and must not depend on its physical layout. Dense
editable/refined `masks_roi` is not covered by this raw-probability storage
policy.

`metrics/` subgroup:

- required: `prob_max`, `mask_present`
- recommended: `area_px`, `centroid_xy`, `centroid_valid`, `bbox_xyxy`,
  `bbox_valid`

Important attrs:

- `source_crop_run`
- `source_crop_storage_mode`
- `source_crop_signature`
- `source_crop_revision`
- `source_detect_review_status_ref` *(when the crop source exposes detect-review linkage)*
- `label_schema_id`
- `mask_labels`
- `output_semantics="multilabel"`
- `overlap_policy="independent_sigmoid"`
- `method`
- `run_semantics`
- `probabilities_dtype`
- `probabilities_encoding`
- `mask_probs_chunk_rois`
- `mask_probs_shard_rois` *(present for indexed-sharded probability runs)*
- `mask_probs_storage_layout` (`regular_chunks_v1` or `indexed_sharding_v1`)
- `mask_probs_storage_policy` (`default_indexed_sharding_v1` or `explicit_regular_chunks_override`)
- `mask_probs_default_shard_rois` (currently `2048`)
- `mask_probs_shard_write` *(double-buffer, exact digest, write, and validation summary for new indexed-sharded runs)*
- `mask_probs_postpack` *(legacy post-pack summary for indexed-sharded runs written before the direct writer)*
- `summary_statistics`

Component-local lineage lives under `components/<component>/provenance`.
Those provenance payloads mirror the canonical crop snapshot fields so merged
and refined subject-mask stages can preserve the exact crop surface they
consumed.

---

## `refined_subject_masks_runs/`

Canonical refined/editable subject-mask runs produced by
`fisheye.tune.refined_subject_mask_review`,
`fisheye.refinement.assemble_refined_subject_masks`, and
`fisheye.refinement.finalize_subject_masks`.

For modern archives, this is also the canonical reviewed eye-geometry surface
when `mask_labels` includes `eye_left` and `eye_right`. Eye masks are stored in
`masks_roi` by semantic component channel, per-eye geometry lives under
`components/eye_left|eye_right/geometry/`, and cross-eye metrics live under
`relations/eye_pair/metrics/`.

Refined subject-mask runs own direct mask-local geometry primitives: component
contours, centroids, areas, bboxes, validity flags, and simple component shape
descriptors that are recomputable from one refined mask channel. Interpreted
biological geometry that requires a coordinate convention, anatomical polarity,
temporal context, or cross-component relationship belongs in
`analysis/subject_shape_runs/<run>` or a specialized downstream analysis run.

| Array | Shape | DType | Notes |
| ----- | ----- | ----- | ----- |
| `frame_indices` *(recommended)* | `(n_rois,)` | `int32` | Copied from the upstream subject-mask/crop lineage when available |
| `frame_counts` *(recommended)* | `(n_frames,)` | `int32` | Copied from the upstream subject-mask/crop lineage when available |
| `detection_indices` *(recommended)* | `(n_rois,)` | `int32` | Copied from the upstream subject-mask/crop lineage when available |
| `source_refined_row_ids` *(recommended)* | `(n_rois,)` | `int64` | Copied from the upstream subject-mask/crop lineage when available |
| `source_detect_row_index` *(recommended)* | `(n_rois,)` | `int32` | Copied from the upstream subject-mask/crop lineage when available |
| `detection_source` | `(n_rois,)` | `int8` | Expected to align with the source crop run |
| `masks_roi` | `(n_rois, C, H, W)` | `uint8` | Dense refined binary masks; authoritative editable pixel surface for modern runs |
| `mask_bitpacked/` *(optional)* | `(n_rois, C, H, ceil(W/8))` | `uint8` | Fixed-size width-bitpacked exact binary masks; derived display/publication cache mirrored from `masks_roi` |
| `mask_rle/` *(optional)* | component groups | typed arrays | Compact component-separated exact binary masks; derived archive/fallback display cache mirrored from `masks_roi` |
| `available_channels` | `(C,)` | `bool` | Declares which refined components are intentionally present |
| `edit_applied` | `(n_rois, C)` | `bool` | True when the refined channel differs from the source subject-mask run |
| `reason_bytes` *(optional)* | `(n_rois, width)` | `uint8` | Null-terminated UTF-8 reason labels |

Historical refined subject-mask runs may contain a variable-length `reason`
array. It is read-only compatibility data and is not emitted by current writers.

Modern editable refined subject-mask runs use dense `masks_roi` as the storage
authority. New provenance should record the explicit dense encoding name
`dense_uint8_v1`; existing `dense_uint8` attrs and CLI values are compatibility
spellings for the same v1 dense binary contract. Compact encodings such as
`bitpacked_binary_v1` and `component_rle_v1` are derived caches for display,
publication, or archive fallback, not training or edit authority. Legacy
compact-only readers should materialize masks through
`fisheye.shared.mask_store.open_mask_store(...)`; see
`docs/mask_rle_storage_design_and_benchmark_plan.md`.

Dense masks are ROI-local and adapt to the crop image shape. The default modern
chunk policy is `(min(128, n_rois), 1, H, W)`, so the common 512x512 refined
layout uses chunks `[128, 1, 512, 512]`; a 348x348 crop-video-backed run should
use `[128, 1, 348, 348]`. The default modern bitpacked cache policy is
`(min(512, n_rois), min(4, C), H, ceil(W/8))`, so a 512x512 four-component cache
uses chunks `[512, 4, 512, 64]`.
Use `scripts/py -m fisheye.utils.materialize_refined_subject_mask_store` to
dry-run, create, or repair the dense authoritative `masks_roi` surface for
historical compact-only refined-subject runs. The same utility refreshes compact mirrors
from edited dense masks: `--refresh-bitpacked --components <name> --rows <idx>`
updates fixed-size bitpacked row/channel cells, while `--refresh-rle
--components <name>` rebuilds selected component RLE groups after the edit path
marks RLE stale.

`metrics/` subgroup:

- required common mask geometry: `mask_present`, `area_px`,
  `centroid_xy`, `centroid_valid`, `bbox_xyxy`, `bbox_valid`
- component-local QC lives under `components/<component>/metrics/` and
  includes `component_count`, `largest_component_fraction`, `hole_count`,
  `hole_area_fraction`, `sigma_noise`, `curvature_var`, `ipr`, and
  `solidity`. Fast smart-finalizer runs may set expensive shape metrics such as
  `sigma_noise`, `curvature_var`, `ipr`, and `solidity` to deferred/NaN values
  when `metric_level="cheap"` is recorded.
- component metric attrs should include
  `schema_id="refined_subject_component_mask_metrics_v1"`,
  `qc_schema_id="refined_subject_component_metric_qc_reasons_v1"`, and a
  `qc_policy` payload. Generated metric-QC reason tags use the
  `needs_review_metric_*` prefix and can be refreshed with
  `fisheye.utils.backfill_refined_subject_mask_metrics`. Refresh attrs record
  the metric level, execution backend, timing summary, chunk timings, and
  per-component review counts.
- eye-pair relation metrics live under
  `relations/eye_pair/metrics/{separation_px,separation_valid}` when both
  `eye_left` and `eye_right` are present
- finalization-specific cleanup metrics may live under
  `components/<component>/finalization_metrics/`

Important attrs:

- `source_subject_mask_run`
- `source_crop_run`
- `source_crop_storage_mode`
- `source_crop_signature`
- `source_crop_revision`
- `source_detect_review_status_ref` *(when the crop source exposes detect-review linkage)*
- `label_schema_id`
- `mask_labels`
- `output_semantics="multilabel"`
- `refinement_semantics="canonical_component_masks"`
- `method`
- `refined_subject_mask_review_status`
- `component_review_statuses`
- `summary_statistics`
- `component_summary_statistics`
- `eye_geometry_status` (`computed` when refined-subject eye geometry arrays and
  relation metrics are present; `deferred` when intentionally skipped)
- `smart_finalizer_timing_summary` and `smart_finalizer_chunk_timings` for runs
  created by `fisheye.refinement.finalize_subject_masks`
- `execution_backend`, `process_shard_execution_enabled`,
  `worker_process_count`, `requested_chunk_size`, `worker_chunk_size`, and
  `chunk_alignment` for smart-finalizer runs

Refined subject-mask runs preserve the same portable crop snapshot contract as
their upstream `subject_mask_runs/<run>` source rather than re-deriving lineage
from the latest crop run later. Component-local lineage continues to live under
`components/<component>/provenance`.

Eye geometry arrays are read through `fisheye.shared.eye_geometry_source`, which
exposes `eye_left`/`eye_right` component data as legacy-compatible
`(N, 2, ...)` array-like views for callers that have not moved to
component-native reads yet. Geometry-only consumers such as eye-angle analysis
have consumer-specific authority rules: future-normal eye-angle publication
requires a canonical `analysis/subject_shape_runs` publication and its nested
assignment-keypoint proof. Mask/export consumers continue using refined-subject
geometry unless they explicitly need derived shape primitives.

---

## `arena_assignment_runs/`

Generated by `fisheye.tracking.arena_assignment` and friends.

Common arrays:

| Array | Shape | DType | Notes |
| ----- | ----- | ----- | ----- |
| `arena_ids` | `(n_detections,)` | `int32` | Assigned arena/ROI IDs |
| `n_detections_per_arena` | `(n_frames, n_arenas)` | `int32` | Per-frame counts for each arena |

Run attrs include the assignment source, arena definitions, and summary counts.

Attributes describe the assignment strategy, arena definitions used,
expected counts, and QA tallies (`assigned`, `unassigned`).

---

## `tracking_runs/`

Generated by `fisheye.tracking.single_subject_per_arena` through
`fisheye.tracking.arena_assignment` for the current one-subject-per-arena
workflow.

Common arrays:

| Array | Shape | DType | Notes |
| ----- | ----- | ----- | ----- |
| `track_ids` | `(n_detections,)` | `int32` | Run-local track ID per source row. `-1` means unassigned. |
| `arena_ids` | `(n_detections,)` | `int32` | Arena assignment mirrored from the bound arena-assignment run. |
| `frame_indices` | `(n_detections,)` | `int32` | Source frame index per row for auditing/debugging. |
| `source_row_indices` | `(n_detections,)` | `int32` | `0..n_rows-1` index into the exact tracked rowset. |
| `track_ids_present` | `(n_tracks,)` | `int32` | Sorted list of emitted real track IDs. |
| `track_arena_ids` | `(n_tracks,)` | `int32` | Arena ID for each emitted track. Parallel to `track_ids_present`. |

Important attrs:

- `tracking_method`: currently `single_subject_per_arena`
- `source_detect_run`
- `source_refined_run` when applicable
- `source_arena_assignment_run`
- `source_rowset_path`
- `track_namespace`: currently `local_per_run`
- `unassigned_track_id`: currently `-1`
- `conflict_policy`: currently `fail`
- `num_tracks`
- `n_assigned_rows`
- `n_unassigned_rows`
- `unassigned_row_rate_percent`
- `tracking_qc_state`: current runtime values are `ok` or `warn`
- `tracking_warn_threshold_rows`, `tracking_warn_threshold_percent`
- `tracking_block_threshold_rows`, `tracking_block_threshold_percent`
- `summary_statistics`

Unassigned rows remain explicit in `tracking_runs/` as `track_id == -1` for QA
and provenance. They are excluded from public `analysis/track_kinematics_runs`
outputs by default unless the diagnostic `--include-unassigned` path is used.

During active review, `tracking_runs/<run>` may be a mutable review surface.
Small corrections should patch touched rows, increment an `edit_revision`, and
append an edit event while keeping `source_rowset_path` fixed. See
`docs/mutable_review_runs_contract.md`.

### Incremental materialization source signatures

New copy-forward materializations store row-aligned source compatibility as:

```text
source_row_signature       uint8[N, 32]
```

The run attrs use the `source_row_signature_` prefix and include
`schema_id=palette.row_source_signature`, schema version, algorithm,
canonicalization, stage, basis, specification digest, and the complete
JSON-safe specification. The specification names every content/revision
component and the global compatibility context used to create the digests.
The array must share the output row axis and be joined by `instance_key` rather
than physical position across runs.

This is a source-compatibility surface, not a replacement observation ID and
not an output-content checksum. Writers may compute it one bounded storage
shard at a time. In immutable tabular snapshots it follows the existing
lineage grid: full 32-byte trailing dimension, 16,384-row inner chunks, and a
requested 131,072-row indexed outer shard. That is about 38 MB logical for 1.2
million observations and roughly nine data shards, before compression. Small
single-owner delta partitions remain ordinarily chunked rather than indexed-
sharded. See
`docs/stable_identity_incremental_materialization_decision.md`.

---

## `calibration/`

Calibration metadata imported from stimulus H5 files. This group stores calibration
parameters for both camera and projector coordinate spaces.

**Attributes**:

| Attribute | Description | Units | Notes |
|-----------|-------------|-------|-------|
| `pixel_to_mm` | Camera-space calibration | pixels/mm | For camera coordinates (4512×4512) |
| `pixels_per_mm_camera` | Alias for pixel_to_mm | pixels/mm | Camera space |
| `pixels_per_mm_projector` | Projector-space calibration | pixels/mm | **For texture/stimulus coordinates (358×358)** |
| `z_eff_mm` | Effective viewing distance | mm | Accounts for refraction through media |
| `measured_stimulus_fps` | Measured stimulus frame rate | fps | Computed from `/video_metadata/frame_metadata` timestamps |
| `measured_fps` | Legacy alias for measured_stimulus_fps | fps | Maintained for backward compatibility |
| `arena_shape` | Arena shape | - | "CIRCLE" or "RECTANGLE" |
| `arena_center_x_px` | Arena center X | pixels | Camera space |
| `arena_center_y_px` | Arena center Y | pixels | Camera space |
| `arena_radius_px` | Arena radius (if circle) | pixels | Camera space |
| `arena_width_px` | Arena width (if rectangle) | pixels | Camera space |
| `arena_height_px` | Arena height (if rectangle) | pixels | Camera space |

**CRITICAL: Dual Calibration System**

There are **two separate calibrations** because the camera and projector operate in
different coordinate spaces:

1. **Camera Space** (4512×4512 pixels):
   - Used for: Offline detection/tracking, bounding boxes, keypoint positions
   - Calibration: `pixels_per_mm_camera` (or `pixel_to_mm`)
   - Typical value: ~5.8 pixels/mm

2. **Texture/Projector Space** (358×358 pixels):
   - Used for: Online stimulus positions (chaser, target), stimulus rendering
   - Calibration: `pixels_per_mm_projector`
   - Typical value: ~0.44 pixels/mm

**Why Two Calibrations?**
- Camera and projector have vastly different resolutions
- Camera may view the arena at an angle (requires homography)
- Scaling between spaces (factor ~12.6) does NOT preserve physical distances
- 1 pixel in texture space ≠ 12.6 pixels in camera space (in real-world distance)

**Usage Guidelines**:
- **Online/stimulus data**: Use `pixels_per_mm_projector`
- **Offline/detection data**: Use `pixels_per_mm_camera`
- **Never mix calibrations**: Apply the calibration matching the coordinate space

---

## `refined_online_runs/`

Refined online target positions from stimulus runs. These are the smoothed, outlier-removed,
and gap-interpolated positions used for accurate movement analysis.

**Structure**: `refined_online_runs/<run_name>/`

See `analysis/refined_online_runs/` section below for detailed structure.

---

## `analysis/`

Organized by analyzer:

### `analysis/stimulus_runs/`

Stimulus run data imported from Citrus H5 files. Each run contains:

**Structure**: `analysis/stimulus_runs/<run_name>/`

**Run Attributes**:
- `created_at_utc`: Import timestamp
- `source_h5`: Path to source H5 file
- `source_stimulus_video_path`: Path to rendered stimulus video next to source H5 (when present; analysis stimulus runs only)
- `import_version`: Import script version
- `protocol_json`: Protocol definition (JSON string)
- `arena_config_json`: Arena/calibration configuration (JSON string)
- `coordinate_transform`: Legacy run-level coordinate system info (JSON string; omitted on new imports when child groups declare local coordinate frames)
  - `texture_dimensions`: Stimulus texture size (typically [358, 358])
  - `camera_dimensions`: Camera resolution (typically [4512, 4512])
  - `texture_to_camera_scale`: Scale factor (~12.6)
  - `coordinate_note`: Description of coordinate spaces
- `legacy_texture_to_camera_transform`: Legacy texture-to-camera transform retained for older consumers when child groups declare their own coordinate metadata.
- `coordinate_transform_status`: Indicates whether the run-level transform is legacy authoritative or suppressed because child-group coordinate metadata is authoritative.
- Gap analysis stats: `total_frames`, `missing_frames`, `max_gap_size`, etc.

**Arrays**:

| Array/Group | Description |
|-------------|-------------|
| `video_metadata/frame_metadata` | Columnar table with frame timing and IDs (see below) |
| `interpolation_mask` | Boolean mask indicating interpolated frames |
| `frame_alignment/camera_to_metadata_index` | Absolute camera frame → metadata index map. Length = max camera frame + 1. Slots with no stimulus data are `-1`. |
| `frame_alignment/camera_interpolation_mask` | Boolean mask for the above indices (`True` only when every metadata row for that camera frame was recorded, not interpolated). Frames with `-1` entries are `False`. |
| `frame_alignment` attrs | `camera_frame_offset` *(legacy)*: original minimum triggering frame retained for backward compatibility. |
| `tracking_data/chaser_states` | Per-frame chaser position/state data (columnar) |
| `tracking_data/bounding_boxes` | Detection bounding boxes from tracking system (columnar) |
| `events/` | Experimental events (columnar storage, see below) |
| `steps/step_<i>/` | Canonical per-step stimulus metadata materialized from events plus protocol JSON |
| `stimulus_coordinates/` | Mirrored H5 stimulus-coordinate metadata when present |

**Step Metadata** (`steps/step_<i>/`):
- `metadata_schema_version`: Step metadata schema version.
- `step_index`, `step_name`, `stimulus_mode_id`, `stimulus_mode`.
- `start_camera_frame`, `end_camera_frame`, `duration_s`.
- `raw_protocol_params_json`: Strict-JSON string containing the source protocol step params.

Stimulus-specific source metadata is nested under optional subgroups:

- `moving_grating/`: `orientation_degrees_authored`,
  `grating_direction_camera_deg`, `direction_mapping_source`,
  `direction_mapping_status`, speed/frequency attrs, and derived temporal
  frequency when inputs are available.
- `concentric_grating/`: authored radial polarity/sign, polarity validation
  status, source-resolved center attrs, speed/frequency attrs, stimulus role,
  and optional centering target annulus attrs.

These step groups are source-derived stimulus metadata. Fish-response metrics
remain under `analysis/stimulus_response_runs/<run>/`.

**Frame Metadata Fields** (`video_metadata/frame_metadata`):
- Stored as separate datasets (columnar layout) for Zarr v3 compatibility.
- `stimulus_frame_num`: Stimulus frame counter (uint64)
- `triggering_camera_frame_id`: Corresponding camera frame ID (uint64)
- `timestamp_relative_ns`: Time since session start (int64)

**Bounding Box Fields** (`tracking_data/bounding_boxes`):
- Stored column-wise. Geometry columns (`x_min`, `y_min`, `width`, `height`) come directly from the tracker.
- `centroid_x`, `centroid_y`: Computed during import as the midpoint of each bounding box in camera pixels.
- `payload_timestamp_ns_epoch`, `received_timestamp_ns_epoch`: Tracker timing fields (int64)
- `payload_frame_id`, `payload_camera_id`: Tracker identifiers for frame/camera (uint64/uint16)
- `box_index_in_payload`: Index of the detection in the tracker payload (uint8)
- `class_id`, `confidence`: Detector metadata (uint16/float32)

**Chaser States Fields** (`tracking_data/chaser_states`):
All fields stored as separate columnar arrays. Key fields include:
- `frame_number`: Stimulus frame counter (uint64)
- `camera_frame_id`: Camera frame ID (uint64)
- `relative_timestamp_ns`: Time since session start (int64)
- `chaser_index`: Chaser agent index (int32)
- `pos_x_px`, `pos_y_px`: Legacy chaser position fields; interpret using the
  group-local coordinate attrs below.
- `target_x_px`, `target_y_px`: Legacy target position fields; interpret using
  the group-local coordinate attrs below.
- `target_visible`: Whether target is tracked (bool)
- `current_radius_px`: Chaser radius (float32)
- `distance_to_target_px`, `distance_to_target_mm`: Distance to target (float32)
- `chase_speed_px_per_s`, `chase_speed_mm_per_s`: Speed (float32)
- `visual_angle_deg`: Visual angle at target (float32)
- `angular_velocity_deg_s`: Angular expansion rate (float32)
- `tau_ms`: Time to collision (float32)
- `loom_mode`, `loom_phase`: Loom behavior state (uint8)
- `trial_state`: PRE/TRAINING/POST period (uint8)
- `chase_sequence_active`: Whether chase is active (bool)

**IMPORTANT**: Chaser position coordinate space is group-local. Prefer
`tracking_data/chaser_states.attrs.coordinate_frame`,
`coordinate_origin`, and `position_fields` over run-level metadata. Legacy runs
without group-local coordinate attrs used texture-space positions
(358×358 pixels) and may expose `coordinate_transform`. New external-IPC runs
can use `coordinate_frame="arena_relative_canvas_px"`, where positions are
relative to the active arena origin; convert to canvas pixels by adding
`calibration/arena_geometry.attrs.arena_origin_in_canvas_{x,y}_px`. Use
`target_clamped_pos_x/y` for arena-constrained target positions; `target_pos_x/y`
may intentionally lie outside the active arena.

**Events Structure** (`events/`):
Events are stored in columnar format with separate arrays for each field:
- `relative_timestamp_ns`: Event timestamp (int64)
- `event_type`: Event type ID (int32)
- `step_index`: Protocol step index (int32)
- `event_name`: Event name (variable-length UTF-8)
- `stimulus_mode`: Active stimulus mode (int32)
- `details`: Event details JSON (variable-length UTF-8)
- `stimulus_frame_num`: Stimulus frame (uint64)
- `camera_frame_id`: Camera frame (uint64)

**Enum Definitions** (`analysis/enums/`):
Enumeration tables for event types, stimulus modes, etc. Each enum is stored as:
- `enums/<enum_name>/id`: Enum ID values (int32)
- `enums/<enum_name>/name`: Enum names (variable-length UTF-8)

Common enums:
- `event_types`: All experimental event types
- `stimulus_modes`: All stimulus mode types
- `chaser_loom_modes`: Chaser looming behaviors
- `chaser_trial_states`: PRE/TRAINING/POST periods

### `analysis/calibration/`

Calibration metadata normalized from H5 `/calibration_snapshot` data at stimulus
import time. The raw/mirrored H5 calibration snapshot remains under
`analysis/stimulus_runs/<run>/calibration/<camera_id>`, while this group is the
machine-readable analysis surface.

**Attributes**:
- `pixel_to_mm`: Camera-space conversion in millimetres per raw camera pixel, derived as `1 / pixels_per_mm_camera`
- `pixels_per_mm_camera`: Citrus camera calibration in raw camera pixels per physical millimetre in the dish/arena plane
- `pixels_per_mm_projector`: Citrus projector/canvas calibration in stimulus pixels per physical millimetre in the displayed arena plane
- `z_eff_mm`: Effective viewing distance through media
- `z_eff_status`: Present when acquisition provided a non-usable value, for example `unusable_nonpositive`
- `homography_status`: Present when no numeric homography matrix was available, for example `missing_numeric_matrix`
- `source_h5`, `source_stimulus_run`: Source lineage for the normalized calibration
- `measured_stimulus_fps`: Measured stimulus frame rate (from H5 frame metadata timestamps)
- `measured_fps`: Legacy alias for `measured_stimulus_fps`
- `arena_shape`: CIRCLE or RECTANGLE
- `arena_center_x_px`, `arena_center_y_px`: Arena center
- `arena_radius_px` or `arena_width_px`, `arena_height_px`: Arena dimensions

**Arrays**:
- `homography_matrix`: 3×3 projector/texture → camera transform when the H5
  includes a numeric `homography_matrix_yml`. Archives with only homography PNG
  buffers must mark the matrix absent rather than infer one.

**CRITICAL CALIBRATION NOTE**:
There are **two separate calibrations** for the two coordinate spaces:
- **`pixels_per_mm_camera`**: For camera-space coordinates (4512×4512 pixels)
  - Used for offline detection/tracking data
  - Used for bounding boxes from detection system
- **`pixels_per_mm_projector`**: For projector/texture-space coordinates (358×358 pixels)
  - Used for online stimulus positions (chaser, target)
  - Used for stimulus-related measurements
  - **This is the authoritative calibration for distance calculations on online/stimulus data**

These are different because:
1. Camera and projector have different resolutions
2. Camera may view the arena at an angle
3. Scaling positions between spaces does NOT preserve physical distances
4. A 1-pixel movement in texture space ≠ 12.6× the distance of 1 pixel in camera space

### `analysis/track_kinematics_runs/`

Track kinematics results organized by type:

**Structure**: `analysis/track_kinematics_runs/<online|offline>/<run_name>/`

**Future-normal run authority**:

- completion gates: `palette_run_completion_status == "complete"`,
  `stage_selector_eligible == true`, and
  `coordinate_binding_status == "bound_canonical_v2"`;
- `track_motion_publication_manifest`, its SHA-256, and
  `track_motion_publication_commit`: closed live array/group/attrs and
  derivation authority;
- `parameters.coordinate_space`: a direction-explicit controlled space such as
  `source_camera_image_px` or `arena_relative_canvas_px`, never legacy
  `"camera"`/`"texture"`;
- manifest v1 offline `inputs`: exactly `detection_path`, `position_source_path`,
  `position_source_rowset_path`, `position_source_kind`, `crop_run`,
  `keypoint_path`, and `tracking_path`, plus only the controlled optional
  `chaser_metrics` and `swim_bout_run` inputs;
- manifest v2 is selected only by the immutable root attr
  `track_motion_publication_manifest_schema_version == 2` and requires an
  explicit `position_lineage_mode`. `direct_crop_selection_v1` seals the exact
  crop-selection record and detection rowset; `verified_collection_proxy_successor_v1`
  seals the current successor-mapping record and digest, its historical rowset,
  and the exact keypoint/tracking sources. Omitting the version attr always
  selects the frozen v1 contract; v1 does not accept v2 inputs or lineage fields;
- offline `keypoint_path` and `tracking_path`: exact two-component run paths
  (`keypoints_runs/<run>` or `refined_keypoints_runs/<run>`, and
  `tracking_runs/<run>`); variant/run-name pairs, nested tracking metadata, and
  reconstructed usability selectors are not future authority;
- `source_refs`: the exact mechanical projection of `inputs`, validated against
  the selected crop/detection/keypoint/tracking arrays;
- `track_motion_input_authority`: digest-bound row alignment and exact input
  array authority;
- `physical_coordinate_authority`: the only authority for `positions_mm` and
  other physical surfaces. A root scalar `pixel_to_mm`, duplicated
  `physical_calibration`, or resolution ratio cannot authorize them.

Historical run-level `coordinate_space`, `pixel_to_mm`, source-run aliases, and
flat layouts are inspection/migration metadata only. They are not accepted by
the normal future reader.

**Shared Root Arrays (offline runs only)**:
- `camera_frame_ids` (`int64`): Master frame index aligned to all chaser metrics.
- `stimulus_frame_nums` (`int64`), `timestamp_ns` (`int64`), `trial_state` (`int16`): Optional context per frame.
- `metadata_mask` (`bool`, optional): Propagated interpolation/original mask when available.
- `has_offline` (`bool`): Indicates frames with valid chaser metrics.
- `distance_to_target_px`, `distance_to_target_mm` (`float32`): Chaser→target separation per frame.
- `distance_to_target_interpolated_px`, `distance_to_target_interpolated_mm` (`float32`, optional): Raw distances with short NaN gaps (duration ≤ `distance_interpolation_seconds` × FPS) filled via linear interpolation.
- `distance_to_target_smoothed_px`, `distance_to_target_smoothed_mm` (`float32`, optional): Moving-average smoothing applied to the interpolated series using the track-kinematics run's `fps` and `smoothing_seconds`.
- `chaser_position_px`, `chaser_positions_px` (`float32`, `[N, 2]`): Chaser centroid in camera pixels (duplicate naming retained for compatibility).
- `fish_centroid_px`, `fish_centroids_px` (`float32`, `[N, 2]`): Target centroid in camera pixels.
- `angle_signed_deg`, `angle_unsigned_deg`, `heading_deg` (`float32`): Legacy per-frame angular metrics from historical `analysis/chaser_fish_metrics` runs. New GoodCopBadCop/CRA analyses should use `analysis/chaser_distance_runs` plus egocentric-bearing outputs instead.

Consumers map from track-level `frame_indices` into these arrays using `camera_frame_ids` and the `has_offline` mask.

**Per-Track Data** (`tracks/id_<track>/`):
Each track stores ordered samples for that ID. `track_sample_key = (track_id,
source_acquisition_frame_index)` is primary identity; `source_instance_key` is
nullable observation lineage, and `source_row_index` is the exact selected
upstream row. `frame_indices` is only a sealed exact alias of the acquisition
frame column. Historical `detection_indices` is not in the future-normal
schema.

- `positions_px`, optional `positions_mm` (`float32`, `[N, 2]`): each array owns
  a canonical coordinate descriptor and digest. Pixel positions retain the
  exact selected native frame; millimetres require the typed physical authority.
- `speed_raw_px`, `speed_raw_mm`: Gap-aware raw speed from validity-filtered frame path-distance increments
- `speed_filtered_px`, `speed_filtered_mm`: Speed after hysteresis filtering
- `speed_smoothed_px`, `speed_smoothed_mm`: Speed after temporal smoothing
- `speed_averaged_px`, `speed_averaged_mm`: Optional longer-window averaged speed
- `movement/speed/<raw|filtered|smoothed|averaged>/`: Preferred v2 grouped
  movement layout. Each level stores `px`, `mm`, `acceleration_px`,
  `acceleration_mm`, `smoothed_acceleration_px`,
  `smoothed_acceleration_mm`, and, where defined for that level,
  `frame_path_distance_px` and `frame_path_distance_mm`.
  These paths aid discovery only; the sealed surface record supplies units,
  axis domain, operation, and exact inputs. Flat speed names may remain as
  sealed compatibility aliases during transition but are never a fallback for
  the normal logical reader.
- `speed_derivatives/`: Transitional source-scoped acceleration mirror. Each child group is
  keyed by source speed level: `speed_raw`, `speed_filtered`,
  `speed_smoothed`, `speed_averaged`.
  - `speed_derivatives/<level>/acceleration_px`, `acceleration_mm`: Framewise
    first difference of the named source speed trace divided by `delta_seconds`.
    Undefined samples, including the first sample and transitions involving
    NaN source speed values, remain NaN.
  - `speed_derivatives/<level>/smoothed_acceleration_px`,
    `smoothed_acceleration_mm`: Centered moving-average smoothing applied after
    differentiation.
  - Child attrs include `source_speed_level`, `source_speed_px_array`,
    `source_speed_mm_array`, `time_delta_array`, `derivative_method`,
    `post_smoothing_method`, `post_smoothing_alignment`,
    `post_smoothing_window_frames`, and `post_smoothing_window_s`.
- `heading_degrees`, `heading_radians`, `delta_heading_degrees`, `angular_velocity_deg_s`
- `angular_velocity_raw_deg_s`, `angular_speed_raw_deg_s`
- `delta_heading_smoothed_degrees`, `angular_velocity_smoothed_deg_s`, `angular_speed_smoothed_deg_s`
- `smoothed_heading_degrees`, `smoothed_heading_radians`
- `acceleration_px`, `acceleration_mm`, `smoothed_acceleration_px`,
  `smoothed_acceleration_mm`: Compatibility aliases for
  `movement/speed/smoothed/*` and `speed_derivatives/speed_smoothed/*`. New
  consumers should read the grouped movement layout so the source speed trace
  is unambiguous.
- `frame_path_distance_raw_px`, `frame_path_distance_raw_mm`: Gap-aware pre-hysteresis frame path-distance increments
- `frame_path_distance_filtered_px`, `frame_path_distance_filtered_mm`: Gap-aware hysteresis-filtered frame path-distance increments
- `frame_path_distance_smoothed_px`, `frame_path_distance_smoothed_mm`: Gap-aware temporally smoothed frame path-distance increments
- `cumulative_path_distance_px`, `cumulative_path_distance_mm`: Cumulative gap-aware path distance
- `second_indices`, `speed_per_second_px`, `speed_per_second_mm`, `heading_per_second_degrees`, `heading_per_second_resultant`
- `keypoint_success`, `detection_source`, plus per-track manifest metadata in subgroup attributes
- `swim_bouts/`: legacy compatibility mirror of selected
  `analysis/swim_bout_runs` bout rows (e.g., `bout_id`, `start_time_s`,
  `end_time_s`, `start_frame`, `end_frame`, `duration_s`, `path_length_mm`,
  `net_displacement_mm`, `mean_speed_mm_s`, `peak_detection_signal_mm_s`,
  `peak_physical_speed_mm_s`, ...). Treat this as a deprecated convenience
  copy only; new consumers should resolve authoritative bout boundaries and
  metrics from `analysis/swim_bout_runs` through `swim_bout_io.py`.

**Preferred v2 movement layout**:
The flat speed arrays plus `speed_derivatives/` hierarchy are retained for
compatibility. New runs also group speed values, path-distance increments, and
speed-derived acceleration together:

```text
tracks/id_<track>/movement/speed/<raw|filtered|smoothed|averaged>/
  px
  mm
  frame_path_distance_px        # where defined for this level
  frame_path_distance_mm        # where defined for this level
  acceleration_px
  acceleration_mm
  smoothed_acceleration_px
  smoothed_acceleration_mm
```

New reader code should prefer `movement/speed/<level>/...`, then fall back to
the v1 `speed_derivatives/<level>/...` and flat speed arrays, then fall back to
historical flat acceleration arrays only for older archives.

Track-level arrays remain unchanged between online and offline runs; only the root-level chaser metrics are added for offline runs.

Current run attrs include `schema_id = "analysis.track_kinematics_runs"`,
`schema_version = 1`, `method = "track_kinematics"`,
`method_version = "track_kinematics.v1"`, `row_axis = "track_samples"`, and
exact `source_refs` for tracking, keypoint, and refined-keypoint inputs.

### `analysis/swim_bout_runs/`

Bout segmentation candidates derived from track-kinematics speed traces.

**Logical reader contract**: consumers should read swim-bout runs through
`fisheye.analysis.swim_bout_io`, not by assuming a single physical tree shape.
The resolver exposes candidates, signal variants, bout tables, intervals,
histograms, summary metrics, and detector-response series for both historical
hierarchical runs and compact tabular runs.

**Hierarchical v1 structure**:
`analysis/swim_bout_runs/<run_name>/<speed_level>/`

Each `<speed_level>/` subgroup answers "what bouts did this speed-level detector
find?" and stores:

- `bouts`: columnar bout-boundary table with `bout_id`, start/end frames and
  times, optional core start/end fields, duration, observed duration,
  path-length, net-displacement, and gap-coverage fields
- `inter_bout_intervals`: columnar table between adjacent bouts
- optional transformed detector-signal arrays, such as
  `detection_signal_mm_s` for the causal exponential response candidate
- run and speed-level attrs describing the source track-kinematics run,
  `track_id`, speed source, thresholding, hysteresis, boundary mode, smoothing,
  and overwrite/review parameters
- frame-resolved duration attrs:
  `min_bout_duration_s`, `resolved_min_bout_frames`,
  `effective_min_bout_duration_s`, `min_gap_duration_s`,
  optional explicit `min_gap_frames`, `resolved_min_gap_frames`,
  `effective_min_gap_duration_s`, `min_gap_frame_source`, and
  `duration_frame_rounding_policy`

Known speed-level subgroups include `speed_raw`, `speed_filtered`,
`speed_smoothed`, `speed_averaged`, and derived `speed_exponential`. Each
subgroup records a generic detector-signal contract through attrs such as
`detection_signal_transform_type`, `detection_signal_source_path`,
`detection_signal_source_level`, and `movement_metric_source_level`. Identity
levels point directly at their track-kinematics speed arrays; transformed levels
store the derived signal as `detection_signal_mm_s`.

`speed_exponential` is a derived response trace for segmentation comparison, not
an independent measured speed. Its causal smoothing/decay can soften rises,
lower peaks, and extend tails relative to `speed_filtered`. Biological speed
measurements should remain grounded in the source speed/path-distance arrays and
the selected bout boundaries.

Within `bouts`, `peak_detection_signal_mm_s` is the maximum of the signal used
to define that subgroup's bout boundaries. `peak_physical_speed_mm_s` is the
maximum of the declared physical movement source inside the same boundaries.
For identity levels these can be equal; for transformed detector signals they
are intentionally distinct.

**Compact v2 structure**:
`analysis/swim_bout_runs/<run_name>/` with
`attrs["layout"] == "compact_tabular_v2"` stores the same logical surfaces as
tables and indexes:

```text
indexes/candidates
indexes/signal_variants
tables/bouts
tables/peak_events
tables/inter_bout_intervals
tables/summary_metrics
tables/histograms
signals/detector_signal_mm_s
signals/detector_signal_signal_ids
signals/frame_indices                    # schema-7 or declared embedded fallback
```

Compact v2 replaces physical `<speed_level>` subgroups with `candidate_id` and
`signal_id` columns. A selected signal's `speed_level`, `role`,
`source_level`, and `path_distance_source_level` come from
`indexes/signal_variants`. A detector response such as `speed_exponential`
therefore remains selectable, but its dense trace is stored as one row in
`signals/detector_signal_mm_s` keyed by `detector_signal_signal_ids`. The
logical array remains signal-major with shape `(detector_signal_id, frame)` for
reader compatibility. New runs store this small dense payload as one regular
chunk (`palette_storage_policy="single_regular_chunk_v1"`). This measured
exception favors the common complete-trace read; it does not apply to large
framewise eye, tail, or track arrays. Readers should still resolve the signal
row before requesting values so the logical lookup remains future-compatible.

New compact runs use `palette.swim_bout_runs` schema version 8 and persist a
run-level `frame_axis_contract` with schema id
`palette.swim_bout_frame_axis_reference`, version 1. The contract pins the
exact same-Zarr track-kinematics `frame_indices` path, source run, track id,
shape, source dtype, and canonical content hash. The default
`frame_axis_storage=reference` does not duplicate the axis.
`frame_axis_storage=embedded` additionally writes `signals/frame_indices` as a
portable fallback, but current Palette readers still prefer the authoritative
path while it exists. Historical schema-7 compact runs have no reference
contract and continue to read their embedded axis unchanged. A reference must
name a concrete source run rather than a `latest` pointer; readers fail closed
on shape, dtype, run, or track mismatch.

This reference rule applies only when the downstream array retains the exact
upstream row domain, ordering, and length. Coordinate axes such as this track
frame axis may then be referenced directly. Identity and mapping arrays serve a
different purpose: `instance_key` identifies a biological/detection instance,
`source_row_indices` maps local rows to upstream rows, and `frame_offsets`
indexes sparse frame groups. A stage that filters, duplicates, or reorders rows
must persist the appropriate local identity or mapping even when its metadata
also names an upstream authority.

Run attrs also include `swim_bout_algorithm_contract`, a versioned,
self-contained scientific contract (`analysis.swim_bout_algorithm_contract`,
schema version 1). It records source paths and axes, the exact causal
exponential recurrence and reset rules, the active threshold/peak algorithm,
duration and boundary rounding, overlap/gap handling, interval conventions,
validity rules, and the physical sources used for bout metrics. The identical
contract and its SHA-256 are nested in stage provenance.

Do not add v1-style compatibility mirrors under compact runs unless a concrete
external reader requires them. New Palette Python readers, Marimo notebooks,
stimulus-response analysis, bout kinematics, and exports should use
`swim_bout_io.py` and treat the physical layout as an implementation detail.

**Source-speed selection rule**:
`analysis/swim_bout_runs` is a separate event-candidate surface, not a child of
`track_kinematics_runs/.../movement/speed`. Viewers should connect the two by
matching lineage metadata. Given a selected track-kinematics run, `track_id`,
and speed source, a compatible swim-bout subgroup must either:

- directly represent that speed level, for example selected `filtered` speed
  maps to subgroup `speed_filtered`, or
- represent a transformed detector signal whose attrs point back to that
  selected speed source, for example `speed_exponential` with
  `detection_signal_source_level="filtered"`.

Consumers should auto-select direct matches first and present transformed
matches as additional candidates. They should not auto-use a bout subgroup whose
detector source points at a different speed trace than the one selected by the
operator.

Duration parameters are operator-facing seconds by default, but bout
segmentation is frame-discrete. New runs resolve positive second durations with
`duration_frame_rounding_policy="ceil_seconds_times_fps"` and persist the
resolved frame counts. Operators may also set explicit `min_gap_frames`; when
present, it overrides `min_gap_duration_s` and is recorded with
`min_gap_frame_source="explicit_frames"`.

Gap merging is an explicit persisted policy:

- `gap_merge_policy="sampled_frame_gap"`: the default; merge
  threshold-separated segments using sampled below-threshold frame counts and
  `resolved_min_gap_frames`
- `gap_merge_policy="interpolated_core_gap"`: compare linearly interpolated
  core threshold-crossing times and `gap_merge_min_gap_duration_s`, falling
  back to the sampled frame rule when interpolation is not valid
- `gap_merge_policy_active`: false for `detection_method="peak_event"` because
  peak-event detection uses peak spacing and width envelopes instead of
  threshold-region gap merging
- `gap_merge_min_gap_duration_s`: the seconds threshold used by the
  interpolated policy; equals `min_gap_duration_s` unless explicit
  `min_gap_frames` was provided
- `gap_merge_min_gap_source`: `seconds` or `explicit_frames`

Swim-bout schema v5 adds `detection_method="peak_event"` as an additive
segmentation family. Peak-event runs still write normal `bouts` rows, but each
speed-level subgroup also contains an aligned `peak_events/` columnar table.
This table stores the peak-finding metadata used to create each event:

- `peak_frame`, `peak_time_s`
- `peak_signal_value_mm_s`
- `peak_prominence_mm_s`
- `peak_width_samples`, `peak_width_s`, `peak_width_height_mm_s`
- `left_ips`, `right_ips`
- `left_width_frame_interpolated`, `right_width_frame_interpolated`
- `left_base_frame`, `right_base_frame`
- `left_base_signal_value_mm_s`, `right_base_signal_value_mm_s`
- `boundary_mode`
- `shape_split_policy`

The first peak-event implementation supports
`peak_event_boundary_mode="relative_prominence_width"` and
`shape_split_policy="none"`. Valley-depth splitting remains future work.

Frame boundary fields (`start_frame`, `end_frame`, `core_start_frame`,
`core_end_frame`) are authoritative for row alignment and array slicing.
Swim-bout schema v3 and newer stores optional sub-frame threshold timing
annotations:

- `core_start_time_s_interpolated`
- `core_end_time_s_interpolated`
- `core_duration_s_interpolated`
- `core_start_time_interpolated_valid`
- `core_end_time_interpolated_valid`
- `threshold_crossing_interpolation`

These annotations estimate threshold crossings for `core_*` boundaries only.
They are finite only when a finite adjacent sample pair brackets the threshold
crossing, and they must not replace frame-index boundaries. Peak-event
interpolated width boundaries live in the aligned `peak_events/` table instead.
`boundary_mode=local_minimum` can still use `start/end` as expanded event
envelope boundaries while `core_*` and interpolated core times describe the
detection criterion.

Use this surface for operator review and comparison of bout definitions. Do not
store downstream heading-change or Johnson-style pre/post position measurements
here; those belong in linked `analysis/bout_kinematics_runs` outputs.

### `analysis/bout_kinematics_runs/`

Per-bout movement, heading, and eye-gaze metrics linked to exact
track-kinematics, swim-bout segmentation, and eye-angle runs. Eye-gaze metrics
are computed by default and are omitted only for an explicitly opted-out
compatibility run.

Use this surface for downstream per-bout biological measurements after a bout
candidate has been selected. The primary table is `per_bout_metrics/`, aligned
to source swim-bout rows, and records both metric values and validity state for
the measurement logic used.

**Structure**: `analysis/bout_kinematics_runs/<run_name>/`

The default physical layout is `compact_tabular_v2`. Logical movement,
heading, and eye-gaze tables are resolved through
`fisheye.analysis.bout_kinematics_io`; the hierarchical paths below describe
the compatibility layout and logical table names, not a required on-disk tree.

**Run Attributes**:
- `schema_id`: `"analysis.bout_kinematics_runs"`
- `schema_version`: Current schema is `7`
- `method`: `"heading_window_and_within_bout_metrics"`
- `method_version`: Current implementation is `"bout_kinematics.v7"`
- `row_axis`: `"swim_bout_rows"`
- `source_track_kinematics_run`, `source_track_id`
- `source_swim_bout_run`, `source_swim_bout_speed_level`
- `source_refs`: Exact source archive/path mapping consumed by the run,
  including `zarr_path`, `source_heading_arrays`, and optional
  `source_peak_events_path`
- `parameters`: Heading levels, `pre_post_mode`, fixed pre/post windows,
  within-bout window policy, physical-active measurement policy, copied source
  boundary field lists, and optional dominant-frequency settings
- `default_heading_level`: Usually `heading_smoothed`

**Movement group**:

```text
movement/per_bout_metrics/
```

`movement/per_bout_metrics/` is stored in columnar form and is aligned to the
source swim-bout rows. It is the physical movement estimator layer; it preserves
source detector-window durations and writes separate active-motion measurements
from a declared physical speed source, usually `speed_filtered_mm`.

Key fields include:

- `bout_id`, `source_start_frame`, `source_end_frame`,
  `source_core_start_frame`, `source_core_end_frame`
- copied detector-boundary durations: `detector_duration_s`,
  `detector_observed_duration_s`, `detector_core_duration_s`
- physical active sampled boundaries: `physical_active_start_frame`,
  `physical_active_end_frame`, `physical_active_start_time_s`,
  `physical_active_end_time_s`, `physical_active_duration_s`
- optional interpolated threshold-crossing boundaries:
  `physical_active_start_time_s_interpolated`,
  `physical_active_end_time_s_interpolated`,
  `physical_active_duration_s_interpolated`,
  `physical_active_start_time_interpolated_valid`, and
  `physical_active_end_time_interpolated_valid`
- physical movement summaries: `physical_active_observed_duration_s`,
  `physical_active_path_length_mm`, `physical_active_path_length_px`,
  `physical_active_mean_speed_mm_s`, and
  `physical_active_peak_speed_mm_s`
- policy/provenance fields: `physical_active_threshold_mm_s`,
  `physical_active_boundary_margin_s`,
  `physical_active_boundary_policy_bytes`, and
  `physical_active_boundary_constraint_bytes`
- validity fields: `physical_active_valid`,
  `physical_active_valid_transition_count`,
  `physical_active_valid_transition_fraction`, and `failure_reason_bytes`

The movement group attrs record
`physical_active_boundary_policy="physical_active"`,
`physical_active_boundary_constraint`, `physical_active_threshold_mm_s`,
`physical_active_boundary_margin_s`, `physical_active_signal_level`, and
`physical_active_signal_array`.

**Heading-level groups**:

```text
heading_smoothed/per_bout_metrics/
heading_raw/per_bout_metrics/
```

`per_bout_metrics/` is stored in columnar form. Key fields include:

- `bout_id`, `source_start_frame`, `source_end_frame`,
  `source_core_start_frame`, `source_core_end_frame`
- optional source interpolated core-threshold timing copied from the source
  swim-bout row: `source_core_start_time_s_interpolated`,
  `source_core_end_time_s_interpolated`,
  `source_core_duration_s_interpolated`,
  `source_core_start_time_interpolated_valid`, and
  `source_core_end_time_interpolated_valid`
- optional source peak-event boundary context copied from aligned
  `peak_events/` rows: `source_peak_frame`, `source_peak_time_s`,
  `source_peak_signal_value_mm_s`, `source_peak_prominence_mm_s`,
  `source_peak_width_s`, `source_peak_width_height_mm_s`,
  `source_peak_left_width_frame_interpolated`,
  `source_peak_right_width_frame_interpolated`,
  `source_peak_left_width_time_s`, `source_peak_right_width_time_s`,
  `source_peak_boundary_mode_bytes`, and
  `source_peak_shape_split_policy_bytes`
- `pre_epoch_start_frame`, `pre_epoch_end_frame`
- `post_epoch_start_frame`, `post_epoch_end_frame`
  (`*_end_frame` values are inclusive; `-1` means no resolved epoch)
- `pre_heading_mean_deg`, `post_heading_mean_deg`
- `net_delta_heading_deg`, `abs_net_delta_heading_deg`
- `pre_position_mean_x_mm`, `pre_position_mean_y_mm`
- `post_position_mean_x_mm`, `post_position_mean_y_mm`
- `interbout_epoch_displacement_mm`
- optional pixel-space mirrors: `pre_position_mean_x_px`,
  `pre_position_mean_y_px`, `post_position_mean_x_px`,
  `post_position_mean_y_px`, `interbout_epoch_displacement_px`
- `within_heading_range_deg`, `within_heading_peak_to_peak_deg`
- `within_heading_path_deg`, `within_heading_std_deg`
- `within_heading_zero_crossings`
- `within_heading_dominant_frequency_hz` plus `dominant_frequency_valid`
- `within_angular_velocity_mean_deg_s`, `within_angular_speed_mean_deg_s`,
  `within_angular_speed_max_deg_s`, `within_angular_velocity_std_deg_s`
- `within_angular_velocity_valid`,
  `within_angular_velocity_transition_count`
- `pre_window_valid`, `post_window_valid`, `pre_position_valid`,
  `post_position_valid`, `within_window_valid`
- `failure_reason_bytes`

These runs must not mutate `analysis/swim_bout_runs`; they are independently
recomputable derived analyses linked to immutable segmentation candidates. See
`docs/bout_kinematics_run_design.md`.

When visualization artifacts are requested, `visualizations/` may contain:

- `bout_movement_summary_track_<id>_png` and
  `bout_movement_summary_track_<id>_interactive` for physical movement
  histograms from `movement/per_bout_metrics`
- `bout_kinematics_summary_track_<id>_png` and
  `bout_kinematics_summary_track_<id>_interactive` for heading/turning
  histograms
- `bout_eye_gaze_summary_track_<id>_png` and
  `bout_eye_gaze_summary_track_<id>_interactive` for default eye-gaze metrics

### `analysis/refined_online_runs/`

Refined online target positions from stimulus runs (smoothed, outlier-removed, gap-filled).

**Structure**: `analysis/refined_online_runs/<run_name>/`

**Run Attributes**:
- `source_stimulus_run`: Source stimulus run name
- `chaser_index`: Which chaser was tracked
- `texture_to_camera_scale`: Scale factor between coordinate spaces
- `coordinate_space`: "texture" (positions in texture space)
- `pixels_per_mm_projector`: Texture-space calibration (pixels/mm)
- `refinement_timestamp`: Processing timestamp
- `processing_time_seconds`: Time to process
- `operations`: List of operations performed (["smooth", "outlier_removal", "interpolate"])
- `parameters`: Refinement parameters (window_length, polyorder, displacement_threshold, max_gap)
- `coverage_stats`: Coverage before/after refinement
- `outlier_stats`: Outliers detected and removed
- `interpolation_stats`: Gaps filled and frames interpolated

**Arrays**:
- `camera_frame_ids`: Camera frame IDs (N,) int32
- `original_valid_mask`: Original validity mask (N,) bool
- `smoothed_mask`: Valid after smoothing (N,) bool
- `outlier_mask`: Detected outliers (N,) bool

**Subgroups**:

`filtered/`: After smoothing and outlier removal
- `camera_frame_ids`: Frame IDs
- `positions_px`: Positions in **texture space** (N, 2) float64
- `valid_mask`: Valid positions mask

`interpolated/`: Final refined positions (after gap filling)
- `camera_frame_ids`: Frame IDs
- `positions_px`: Positions in **texture space** (N, 2) float64
- `valid_mask`: Valid positions mask
- `interpolation_mask`: Identifies interpolated positions (bool)

**IMPORTANT**: All positions are in **texture space** (358×358 pixels).
Use `pixels_per_mm_projector` for distance calculations.

### `analysis/eye_angle_runs/`

Eye angle analysis results:
- Per-ROI and per-frame eye-angle metrics
- QA masks and quality indicators
- `reason_codes` for data quality classification
- The default physical layout is `compact_dense_v2`: comprehensive
  `roi_angles` and `frame_angles` matrices plus a name index. Current
  production storage uses semantic channel order, approximately `(4096, 16)`
  inner chunks and `(131072, 32)` outer shards; consumers resolve named
  channels through `fisheye.analysis.eye_angle_io` rather than persisting
  numeric column positions.
- Run attrs: `schema_id = "analysis.eye_angle_runs"`, `schema_version = 7`,
  `method = "ellipse_and_centroid_eye_angles"`,
  `method_version = "eye_angle_analysis.v5"`,
  `row_axis = "keypoint_detection_rows"`, and `eye_angle_output_schema` for
  machine-readable output groups, units, suffixes, derivative arrays, and QA
  reason-code linkage. Output schema v9 includes
  `variant_schema`, mirrored as `eye_angle_variant_schema` in run attrs, for
  UI-selectable angle representations, and links the versioned algorithm
  contract. Run-schema v7/output-schema v9 additionally make ordered row
  identity and acquisition time explicit.
- `eye_angle_algorithm_contract`: versioned computation metadata covering the
  ellipse source/parameter convention, body-frame formulas and resolved
  keypoint indices, axis-ambiguity rule, angle transforms, QA thresholds,
  temporal operators, effective smoothing windows, derivative gap limit, FPS
  source, and unique-detection frame projection.
- `eye_angle_source_contracts`: exact source paths plus available schema,
  method, completion, git, and lineage identities for the canonical
  subject-shape eye geometry and the exact base-keypoint publication sealed by
  its assignment proof. The canonical authority binds keypoint values,
  detection success, crop placement, labels, ordered `instance_key`, ordered
  `source_acquisition_frame_index`, and source frame count. Canonical readers
  must not reconstruct or select a different keypoint source.
- The v5 scientific method exposes canonical major-axis arrays:
  `left_major_signed_deg`, `right_major_signed_deg`,
  `vergence_major_signed_deg`, and `version_major_deg`. The major axis is
  resolved into the fish forward half-plane, with `0 deg` aligned to body
  forward and positive values toward anatomical left.
- Output schema v6 adds Bianco/Engert-style eye-frame arrays:
  `left_eye_angle_deg`, `right_eye_angle_deg`, and
  `vergence_eye_angle_deg`, plus smoothed and delta variants. These are derived
  from the canonical major-axis fields with per-eye nasal-positive signs:
  `left_eye_angle_deg = -left_major_signed_deg`,
  `right_eye_angle_deg = right_major_signed_deg`, and
  `vergence_eye_angle_deg = left_eye_angle_deg + right_eye_angle_deg`.
  Positive `vergence_eye_angle_deg` means convergence; negative means
  divergence.
- Output schema v7 adds `eye_angle_variant_schema`, which groups fields into
  UI-facing representations: `eye_frame`, `gaze`, `nasal_gaze`, `major`,
  `centroid`, and `legacy`. UIs should prefer this registry over hardcoded
  angle-field lists when offering representation selectors.
- Output schema v8 adds the `eye_angle_algorithm_contract` link and exact
  smoothing, delta, and derivative operator identities without changing the
  v5 scientific method.
- Output schema v9 adds `support/instance_key` and
  `support/source_acquisition_frame_index`. `support/frame_indices` remains an
  equality-required compatibility alias of the acquisition-frame array; it is
  not a second temporal authority.
- The v5 scientific method exposes explicit gaze arrays derived from the
  resolved major axis:
  `left_gaze_deg`, `right_gaze_deg`, `left_gaze_signed_deg`,
  `right_gaze_signed_deg`, `vergence_gaze_deg`,
  `vergence_gaze_signed_deg`, and `version_gaze_deg`, plus smoothed, delta,
  speed, and acceleration variants where applicable. New consumers should use
  these over legacy `left_deg` / `right_deg` fields for gaze surfaces.
- `left_gaze_xy` and `right_gaze_xy` are ROI/image-space unit vectors for
  drawing the derived gaze direction.
- The v5 scientific method retains `vergence_gaze_deg` as the v3-compatible
  total/axis separation and adds `left_nasal_gaze_deg`,
  `right_nasal_gaze_deg`, and
  `mean_eye_vergence_gaze_deg` for BEAST/Johnson-style mean per-eye vergence.
- Run attrs include `preferred_angle_family = "gaze"`,
  `preferred_eye_axis = "ellipse_major"`, and
  `gaze_angle_source = "ellipse_minor_derived_from_resolved_major_axis"`.
  `preferred_angle_family` is the historical preferred biological viewing
  surface; UI selectors should use
  `eye_angle_variant_schema.default_representation` for angle-representation
  defaults.
- Run-schema v7 materializes keypoint-derived body-frame support arrays under
  `support/body_frame/` so signed eye angles are anatomical-left-positive and
  convergence polarity is not conflated with ellipse-axis orientation
  disambiguation. These arrays are recomputed from the exact sealed keypoint
  payload and success mask; a separately persisted upstream heading is not an
  input.
- `qa/roi/*major_axis_marginal` and `qa/frame/major_axis_marginal` are
  non-fatal warning flags for rare cases where the major axis is close to the
  half-plane boundary used for 180 degree ambiguity resolution.
- Future-normal eye geometry is a complete, selector-eligible
  `analysis/subject_shape_runs/<run>` publication with left/right eye ellipse
  geometry and a used assignment-keypoint proof. Run attrs record
  `source_geometry_kind`, `source_subject_shape_run`,
  `source_refined_subject_masks_run`, and `source_refined_eye_run` as
  applicable. Refined-subject geometry is accepted only on the explicitly
  requested historical refined-keypoint diagnostic path; that output is
  permanently `stage_selector_eligible = false` with
  `publication_scope = "historical_diagnostic_only"`.
- Canonical keypoint lineage names the exact sealed child in
  `source_base_keypoints_run` (and canonical `source_keypoints_run`).
  `source_refined_keypoints_diagnostic_run` is present only for the explicit
  nonselector diagnostic path. There is no latest-keypoint or refined-keypoint
  fallback for canonical publication.
- `source_geometry_kind` normalizes the consumed geometry source as
  `subject_shape_eye_geometry`, `refined_subject_eye_geometry`,
  `legacy_refined_eye_geometry`, or `unknown_eye_geometry`.

### `analysis/subject_shape_runs/`

Active deterministic derived analysis stage for biological shape outputs
computed from canonical refined subject masks.

**Structure**: `analysis/subject_shape_runs/<run_name>/`

Expected source:

- exact `refined_subject_masks_runs/<run>` source
- optional refined-subject mask-local geometry primitives
- optional keypoint, heading, tracking, or temporal context inputs

Current run attrs:

- `schema_id`: `"analysis.subject_shape_runs"`
- `schema_version`: `3`
- `method`: the declared subject-shape method
- `method_version`: current implementation version `10`
- `created_at_utc`
- `row_axis`: initially `"refined_subject_mask_rows"`
- `source_refs`: exact input runs and paths
- `source_refined_subject_masks_run`
- optional `body_frame_schema_id`, `body_frame_schema_version`,
  `body_frame_estimator`, `body_frame_coordinate_space`,
  `body_frame_angle_convention`, and `body_frame_source_refs` when the run
  materializes a shared fish anatomical body frame

Expected row-aligned index group:

- `row_index/frame_indices`
- `row_index/detection_indices` when available
- `row_index/source_refined_row_ids` when available

`row_index/frame_indices` is the canonical subject-shape row-to-frame mapping.
Do not require a root-level `frame_indices` array for subject-shape runs; current
canary runs may omit that compatibility shortcut. Realtime viewers may add or
consume a derived top-level `frame_index` alias or CSR-style `frame_index/`
lookup cache, but those are convenience surfaces over the stable row axis.

Expected body-frame group when materialized:

- `body_frame/origin_xy`: `(N, 2)` anatomical frame origin in ROI or image pixels
- `body_frame/forward_axis_xy`: `(N, 2)` unit vector pointing posterior-to-anterior
- `body_frame/left_axis_xy`: `(N, 2)` unit vector pointing anatomical left
- `body_frame/heading_deg`: `(N,)` heading from `atan2(-dy, dx)`
- `body_frame/valid`: `(N,)` body-frame validity
- `body_frame/failure_reason_bytes`: `(N, width)` optional uint8 utf8-null-terminated stable reason tags
- `body_frame/midline_xy`: `(N, P, 2)` optional centerline/spline samples
- `body_frame/arclength_px`: `(N,)` optional body midline/spline arc length

Expected component groups:

- `components/subject_body/`: interpreted body geometry such as centerline,
  B-spline, body length, axis, curvature, and body-shape validity.
- `components/swim_bladder/`: swim-bladder centroid/blob/ellipse summaries and
  component-specific validity.
- `components/eye_left/` and `components/eye_right/`: first-class
  analysis-facing eye component geometry, ellipse/axis summaries, and
  component-specific validity consumed by coherent body/eyes/swim
  subject-shape analysis.

Current centerline/tail-anchor arrays:

- `components/subject_body/centerline_xy`: `(N, P, 2)` sampled head-to-tail
  centerline in ROI pixels.
- `components/subject_body/centerline_valid`: `(N,)` centerline validity.
- `components/subject_body/centerline_failure_reason_bytes`: `(N, width)`
  stable reason tags.
- `components/subject_body/snout_tip_xy`: `(N, 2)` semantic rostral/nasal
  landmark, distinct from body-frame origin and centerline head endpoint.
- `components/subject_body/snout_tip_valid`: `(N,)` validity for
  `snout_tip_xy`.
- `components/subject_body/snout_tip_failure_reason_bytes`: `(N, width)`
  stable reason tags for rostral/snout estimation.
- `components/subject_body/head_endpoint_to_snout_distance_px`: `(N,)`
  Euclidean distance between `head_endpoint_xy` and `snout_tip_xy`. In current
  schema v3/method v8 runs this should be approximately zero for valid
  centerlines because `head_endpoint_xy` is snout-anchored.
- `components/subject_body/centerline_reaches_snout`: `(N,)` true when
  `head_endpoint_to_snout_distance_px` is within the run's declared threshold.
- `components/subject_body/centerline_snout_check_reason_bytes`: `(N, width)`
  reason tags for the intermediate centerline-to-snout check.
- `components/subject_body/head_endpoint_xy`: `(N, 2)` anterior endpoint of
  the selected centerline/spline estimator. In current schema v3/method v8 runs
  this is the validated `snout_tip_xy` for rows with `centerline_valid = true`.
  Older schema v2/method v5 runs may have a skeleton-derived endpoint that does
  not reach the semantic snout.
- `components/subject_body/tail_tip_xy`: `(N, 2)` posterior centerline
  endpoint; source-specific measurement of semantic `tail_tip`.
- `components/subject_body/tail_base_xy`: `(N, 2)` centerline projection of
  the caudal swim-bladder contour anchor.
- `components/subject_body/tail_base_valid`: `(N,)` tail-base projection
  validity.
- `components/subject_body/tail_base_arclength_px`: `(N,)` arclength from head
  endpoint to tail base.
- `components/subject_body/tail_segment_arclength_px`: `(N,)` arclength from
  tail base to tail tip.
- `components/subject_body/body_arclength_px`: `(N,)` current centerline
  arclength from head endpoint to tail tip.
- `components/swim_bladder/caudal_contour_point_xy`: `(N, 2)` swim-bladder
  contour point with minimum projection on the body-frame forward axis.
- `components/swim_bladder/caudal_contour_projection_px`: `(N,)` projection of
  the caudal contour point in body-frame coordinates.
- `components/swim_bladder/caudal_contour_valid`: `(N,)` caudal-anchor
  validity.

`components/subject_body/bspline_sample_xy` and
`components/subject_body/tail_sample_xy` are subject-shape geometry outputs.
They may be denser than the behavior-facing representation used for tail-angle
analysis. Low-dimensional tail-angle vectors, currently defaulting to `K=10`,
belong in `analysis/tail_kinematics_runs` as `tail_angle_sample_*` arrays.

Expected relation groups:

- `relations/eye_pair/`: cross-eye metrics such as separation.
- `relations/swim_bladder_to_body/`: swim-bladder position relative to body
  axis or centerline.
- `relations/eyes_to_body/`: eye angles or offsets relative to body/head
  heading.

Component groups in `analysis/subject_shape_runs` are derived geometry groups,
not approval surfaces. Source mask approval remains owned by
`refined_subject_masks_runs/components/<component>`.

Eye geometry in `analysis/subject_shape_runs` is the analysis-facing coherent
subject-shape surface. Eye contours, ellipse fits, and eye-pair checks may also
live in `refined_subject_masks_runs` when they are mask-local QC/source
primitives. `analysis/eye_angle_runs` remains a specialized downstream
time-series or behavior-facing analysis. Canonical runs require this
subject-shape eye geometry and the exact base keypoint child sealed by its
assignment proof; refined-subject geometry is diagnostic-only for this
consumer.

Examples that belong here instead of in `refined_subject_masks_runs` as
canonical outputs:

- body centerline/spline used as an anatomical coordinate frame
- canonical body B-spline fits with smoothing/knot/parameterization policy
- canonical body length from centerline/B-spline arc length
- head/tail-polarized body axis or mask-derived heading
- body curvature, bend, or width profile
- swim-bladder position relative to body axis or centerline
- swim-bladder distance to eye pair, body centroid, or anatomical landmarks
- eye angles relative to body/head heading
- temporally smoothed or track-aligned shape metrics

The first writer is `fisheye.analysis.subject_shape_runs`. It writes
row-aligned body/eyes/swim component summaries, body principal-axis estimates,
eye/swim ellipse summaries, eye-pair relations, swim/eye-to-body relations,
mask-component body-frame arrays, caudal swim-bladder contour anchors,
snout-anchored subject-body centerlines, B-spline samples/control points,
tail-segment samples/tangents/normals/curvature, and body/tail length metrics
using serial or Dask worker-chunk execution. Mask-width profiles and richer
shape-QC summaries remain follow-up derived-shape methods. The storage and
provenance contract is documented in `docs/subject_shape_runs_contract.md`; the
shared derived-run contract is documented in
`docs/derived_analysis_run_contract.md`. Shared fish-relative frame semantics
are documented in `docs/body_frame_contract.md`.

### `analysis/tail_kinematics_runs/`

Frame-level tail-angle, tail-deflection, and tail-curvature metrics derived
from an exact ordered tail-geometry source, usually
`analysis/subject_shape_runs/<run>/components/subject_body`.

This run family is behavior-facing. It should consume subject-shape geometry
and write derived traces, but it should not mutate subject-shape geometry,
refined masks, swim-bout segmentations, or classifier outputs. The first design
contract is documented in `docs/tail_kinematics_run_design.md`.

**Structure**: `analysis/tail_kinematics_runs/<run_name>/`

Expected run attrs:

- `schema_id`: `"analysis.tail_kinematics_runs"`
- `schema_version`: initial design is `1`
- `method`: e.g. `"tail_metrics_from_subject_shape"`
- `method_version`
- `row_axis`: `"roi_rows"`
- `source_subject_shape_run`
- `source_refined_subject_masks_run`
- `source_tail_geometry_kind`: e.g. `"subject_shape_bspline_tail_resample"`
- `body_frame_convention`
- `tail_angle_reference_axis`: `"caudal_axis=-forward_axis"`
- `tail_angle_positive_direction`: `"anatomical_left"`
- `tail_sample_domain`: `"tail_segment_normalized_arclength"`
- `tail_angle_sample_count`: default `10`
- `source_geometry_tail_sample_count`: optional count from the source
  subject-shape geometry

Expected arrays:

- `frame_index`: `(N,)`
- `time_s`: `(N,)` optional
- `valid`: `(N,)`
- `failure_reason_bytes`: `(N, width)`
- `tail_angle_sample_s`: `(K,)` low-dimensional normalized tail positions used
  for behavior-facing tail-angle vectors.
- `tail_angle_sample_xy`: `(N, K, 2)` evaluated positions at
  `tail_angle_sample_s`.
- `tail_angle_rad`: `(N, K)` signed body-frame tail tangent angle.
- `tail_angle_deg`: `(N, K)` optional plotting mirror.
- `tail_tip_angle_rad`: `(N,)`
- `tail_tip_angle_deg`: `(N,)` optional plotting mirror.
- `tail_lateral_deflection_px`: `(N, K)` signed lateral displacement from
  `tail_base_xy` along the body-frame anatomical-left axis.
- `tail_tip_lateral_deflection_px`: `(N,)`
- `tail_lateral_deflection_mm`: `(N, K)` optional when calibrated.
- `max_abs_tail_angle_rad`: `(N,)`
- `max_abs_tail_angle_deg`: `(N,)` optional plotting mirror.
- `tail_angle_rms_rad`: `(N,)`
- `integrated_abs_tail_angle_rad`: `(N,)`
- `tail_curvature_px_inv`: `(N, K)` mirrored or converted from the selected
  subject-shape geometry source with source attrs.
- `max_abs_tail_curvature_px_inv`: `(N,)`
- `integrated_abs_tail_curvature`: `(N,)`

When available from the source subject-shape run, the writer should also copy
`source_refined_subject_masks/row_revision` and
`source_refined_subject_masks/row_revision_available` into the tail run so
refined-mask lineage remains auditable from the selected tail-kinematics
surface.

Tool-specific posture views, such as Megabouts-ready arrays, should be stored
in `analysis/tail_posture_view_runs/<run>` with explicit source attrs. They
should not be nested under or overwrite Palette-native tail traces. Third-party
classifier labels should land in a separate
`analysis/bout_classification_runs/<run>` family.

Dense whole-body B-spline samples, B-spline control points, and geometry/QC
tail samples remain in `analysis/subject_shape_runs`. The tail-kinematics
surface intentionally defaults to a lower-dimensional `K=10` behavior vector
for plotting, bout summaries, and Megabouts-like adapters.

### `analysis/tail_posture_view_runs/`

Tool-compatible tail-posture views derived from Palette source geometry. These
runs are regenerated compatibility artifacts and are not canonical replacements
for `analysis/tail_kinematics_runs`.

**Structure**: `analysis/tail_posture_view_runs/<run_name>/`

**Run Attributes**:

- `schema_id`: `"analysis.tail_posture_view_runs"`
- `schema_version`: `1`
- `method`: e.g. `"tail_posture_view_from_subject_shape"`
- `method_version`
- `row_axis`: `"roi_rows"`
- `view_family`: e.g. `"megabouts_compatible"`
- `compatible_tool`: e.g. `"megabouts"`
- `dependency_policy`: e.g. `"no_megabouts_dependency_required"`
- `source_subject_shape_run`
- `source_subject_shape_path`
- `source_refined_subject_masks_run`
- `source_tail_kinematics_run`: optional comparison source
- `source_tail_geometry_kind`: e.g. `"subject_shape_tail_curve_resample"`
- `head_source`: e.g. `"head_endpoint_xy"` or `"snout_tip_xy"`
- `keypoint_count`: e.g. `11`
- `angle_count`: e.g. `10`
- `angle_convention`: e.g. `"megabouts_cumulative_segment_angle"`
- `keypoint_order`: e.g. `"tail_base_to_tail_tip"`
- `frame_index_source`
- `row_lineage_copied`, `row_lineage_missing`
- `algorithm_provenance`
- standard stage `provenance`

**Arrays**:

- `frame_index`: `(N,)`
- `valid`: `(N,)`
- `failure_reason_bytes`: `(N, width)`
- `head_xy`: `(N, 2)`
- `head_yaw_rad`: `(N,)`
- `tail_keypoints_xy`: `(N, 11, 2)` for the current Megabouts-compatible view
- `tail_angle_rad`: `(N, 10)` canonical angle units for the current view
- `tail_angle_deg`: `(N, 10)` plotting mirror

**Groups**:

- `row_index/`: copied source row-lineage arrays when available, including
  `frame_indices`, `detection_indices`, `source_refined_row_ids`, and
  `source_detect_row_index`.

The first implemented view is `megabouts_compatible`: it resamples
subject-shape tail geometry to 11 ordered tail keypoints and writes 10
cumulative segment-angle channels. It records that it is a Palette-owned
compatibility implementation and does not require or import Megabouts.

### `analysis/stimulus_response_runs/`

Per-step behavioral metrics across stimulus types. Consumes identity-resolved
track data from `track_kinematics_runs` and stimulus metadata from
`stimulus_runs`. See `docs/stimulus_response_run_design.md` for full metric
definitions and `docs/archive/stimulus_response_implementation_plan.md` for design
decisions.

**Structure**: `analysis/stimulus_response_runs/<run_name>/`

Schema `palette.stimulus_response` version 2 defaults to
`compact_tabular_v2`, with summary, bout, window, and trial tables resolved by
`fisheye.analysis.stimulus_response_io`. The hierarchical groups below are the
explicit compatibility/debug layout; compact runs intentionally omit dense
per-frame/time-series copies that can be reconstructed from exact upstream
sources.

**Run Attributes**:
- `provenance`: Stage provenance contract (`palette_stage_provenance`)
- `source_track_kinematics_run`: Source kinematics run name
- `source_track_kinematics_type`: `"online"` or `"offline"`
- `source_stimulus_run`: Source stimulus run name
- `source_bout_run`: Source bout run name (optional, present when bout data used)
- `n_steps`, `n_fish`, `fish_ids`: Recording summary

**`global/`** — recording-wide per-fish movement summary:

| Array | Shape | DType | Notes |
|-------|-------|-------|-------|
| `fish_id` | `(n_fish,)` | `int32` | Track IDs |
| `total_distance_mm` | `(n_fish,)` | `float32` | |
| `mean_speed_mm_s` | `(n_fish,)` | `float32` | |
| `total_active_s` | `(n_fish,)` | `float32` | Time above moving threshold |
| `fraction_moving` | `(n_fish,)` | `float32` | |

**`steps/step_{i}/`** — one group per protocol step:

Step attributes: `step_index`, `step_name`, `stimulus_mode`, `stimulus_mode_id`,
`start_frame`, `end_frame`, `duration_s`, `stimulus_params`.

**`steps/step_{i}/per_fish/`** — base movement metrics (all stimulus types):

| Array | Shape | DType | Notes |
|-------|-------|-------|-------|
| `fish_id` | `(n_fish,)` | `int32` | |
| `total_distance_mm` | `(n_fish,)` | `float32` | |
| `mean_speed_mm_s` | `(n_fish,)` | `float32` | |
| `median_speed_mm_s` | `(n_fish,)` | `float32` | |
| `max_speed_mm_s` | `(n_fish,)` | `float32` | |
| `fraction_moving` | `(n_fish,)` | `float32` | |
| `coverage` | `(n_fish,)` | `float32` | Fraction of step frames with valid detection |
| `num_bouts` | `(n_fish,)` | `int32` | Optional, present when bout data available |
| `mean_bout_duration_s` | `(n_fish,)` | `float32` | Optional |
| `mean_interbout_interval_s` | `(n_fish,)` | `float32` | Optional |

**`steps/step_{i}/per_bout/`** — bout-level metrics (optional):

| Array | Shape | DType | Notes |
|-------|-------|-------|-------|
| `fish_id` | `(n_bouts,)` | `int32` | |
| `bout_id` | `(n_bouts,)` | `int32` | |
| `start_frame` | `(n_bouts,)` | `int64` | |
| `end_frame` | `(n_bouts,)` | `int64` | |
| `duration_s` | `(n_bouts,)` | `float32` | |
| `mean_speed_mm_s` | `(n_bouts,)` | `float32` | |
| `peak_physical_speed_mm_s` | `(n_bouts,)` | `float32` | |

**`steps/step_{i}/grating/`** — MOVING_GRATING steps only:

This group only exists when `stimulus_mode == "MOVING_GRATING"`.

`grating/per_frame/`:

| Array | Shape | DType | Notes |
|-------|-------|-------|-------|
| `frame_indices` | `(n_step_frames,)` | `int64` | |
| `alignment_angle_deg` | `(n_fish, n_step_frames)` | `float32` | 0 = following, ±180 = opposing |
| `alignment_cos` | `(n_fish, n_step_frames)` | `float32` | +1 = following, -1 = opposing |
| `speed_along_grating_mm_s` | `(n_fish, n_step_frames)` | `float32` | Speed projected onto grating direction |
| `angular_velocity_deg_s` | `(n_fish, n_step_frames)` | `float32` | |

`grating/per_fish/`:

| Array | Shape | DType | Notes |
|-------|-------|-------|-------|
| `mean_alignment_cos` | `(n_fish,)` | `float32` | |
| `resultant_vector_length` | `(n_fish,)` | `float32` | Circular mean resultant (0=random, 1=consistent) |
| `fraction_following` | `(n_fish,)` | `float32` | |
| `fraction_opposing` | `(n_fish,)` | `float32` | |
| `fraction_perpendicular` | `(n_fish,)` | `float32` | |
| `speed_weighted_alignment` | `(n_fish,)` | `float32` | |
| `optomotor_gain` | `(n_fish,)` | `float32` | mean(speed_along) / grating_speed |
| `drift_along_grating_mm` | `(n_fish,)` | `float32` | Net displacement in grating direction |
| `drift_perp_grating_mm` | `(n_fish,)` | `float32` | Net displacement perpendicular to grating |
| `latency_to_follow_s` | `(n_fish,)` | `float32` | Time to first sustained following (NaN if never) |

`grating/time_series/`:

| Array | Shape | DType | Notes |
|-------|-------|-------|-------|
| `bin_center_s` | `(n_bins,)` | `float32` | |
| `alignment_cos` | `(n_fish, n_bins)` | `float32` | |
| `speed_mm_s` | `(n_fish, n_bins)` | `float32` | |
| `fraction_following` | `(n_fish, n_bins)` | `float32` | |
| `optomotor_gain` | `(n_fish, n_bins)` | `float32` | |

`grating/omr/`:

Moving-grating OMR responsiveness indices. This subgroup is present on
`MOVING_GRATING` steps when OMR metrics are enabled.

Important attrs include `method_version`, `stimulus_direction_deg`,
`grating_direction_camera_deg`, `orientation_degrees_authored`,
`camera_to_projector_offset_deg`, `direction_mapping_source`,
`direction_mapping_status`, `direction_mapping_validated`,
`detector_estimator_policy`, `position_source_array`, `position_anchor`,
`speed_source_array`, `projection_deadzone`,
`projection_speed_deadzone_mm_s`, `moving_threshold_mm_s`,
`window_lengths_s`, and `early_response_window_lengths_s`. Optional speed,
frequency, and arena geometry attrs use JSON `null` when unavailable; metadata
must not contain JSON-invalid `NaN` or `Infinity`.

| Subgroup | Key Arrays | Notes |
|----------|------------|-------|
| `per_fish/` | `fish_id`, `omr_path_index`, `omr_net_direction_index`, `bout_fraction_correct_classified`, `bout_choice_index`, `bout_path_index`, `time_choice_index`, `first_aligned_bout_latency_s`, `quality_flag` | Step-level per-fish OMR summaries |
| `per_bout/` | `fish_id`, `bout_id`, `start_frame`, `end_frame`, `per_bout_omr_score`, `parallel_displacement_mm`, `bout_displacement_mm`, `bout_path_length_mm`, `correct_label`, `quality_flag` | Bout boundaries come from swim-bout detector runs; physical scores come from track positions |
| `windows/` | `window_id`, `fish_id`, `start_frame`, `end_frame`, `window_length_s`, `omr_path_index`, `time_choice_index`, `coverage_fraction`, `quality_flag` | Non-overlapping response windows |
| `early_windows/` | `window_id`, `fish_id`, `window_length_s`, `actual_window_length_s`, `omr_path_index`, `bout_path_index`, `time_choice_index`, `quality_flag` | Onset-anchored first-response windows |

**`steps/step_{i}/concentric_grating/`** — CONCENTRIC_GRATING steps only:

This group only exists when `stimulus_mode == "CONCENTRIC_GRATING"` and a
center can be resolved.

| Subgroup | Key Arrays | Notes |
|----------|------------|-------|
| `per_frame/` | `frame_indices`, `distance_to_center_mm`, `radial_heading_angle_deg`, `radial_speed_mm_s`, `tangential_speed_mm_s` | Centering/polar decomposition |
| `per_fish/` | `fish_id`, `mean_distance_to_center_mm`, `initial_distance_to_center_mm`, `final_distance_to_center_mm`, `fraction_approaching`, `mean_radial_speed_mm_s`, `mean_tangential_speed_mm_s` | Step-level centering summaries |
| `time_series/` | `bin_center_s`, `distance_to_center_mm`, `radial_speed_mm_s`, `radial_heading_cos`, `fraction_approaching` | Binned centering summaries |

`concentric_grating/radial_omr/`:

Stimulus-aligned radial/tangential OMR indices for concentric gratings. This
subgroup preserves outward-positive physical radial components separately from
stimulus-aligned components:

```text
stimulus_aligned = stimulus_radial_sign * outward_radial
```

Important attrs include `method_version`,
`coordinate_system = "camera_mm_polar_about_stimulus_center"`,
`stimulus_center_mm`, `stimulus_center_source`,
`stimulus_radial_polarity`, `stimulus_radial_sign`,
`stimulus_radial_polarity_authored`,
`stimulus_radial_polarity_observed`,
`stimulus_radial_polarity_source`,
`stimulus_radial_polarity_validated`,
`effective_stimulus_radial_polarity_source`,
`radial_singularity_epsilon_mm`, `projection_deadzone`,
`projection_speed_deadzone_mm_s`, `moving_threshold_mm_s`,
`window_lengths_s`, `early_response_window_lengths_s`,
`concentric_grating_role`, and detector-vs-estimator source attrs.

| Subgroup | Key Arrays | Notes |
|----------|------------|-------|
| `per_frame/` | `frame_indices`, `valid_radial_basis`, `radius_mm`, `radial_speed_outward_mm_s`, `tangential_speed_ccw_mm_s`, `stimulus_aligned_radial_speed_mm_s` | Frame-level radial basis and speed decomposition |
| `per_fish/` | `fish_id`, `omr_path_index`, `radial_path_index`, `omr_net_direction_index`, `tangential_bias_index`, `start_radius_mm`, `end_radius_mm`, `bout_fraction_correct_classified`, `time_choice_index`, `first_aligned_bout_latency_s`, `quality_flag` | Step-level radial OMR summaries |
| `per_bout/` | `fish_id`, `bout_id`, `start_frame`, `end_frame`, `radial_omr_score`, `radial_net_direction_score`, `tangential_bias_score`, `omr_label`, `quality_flag` | Bout-level radial/tangential scores |
| `windows/` | `window_id`, `fish_id`, `window_length_s`, `omr_path_index`, `time_choice_index`, `tangential_bias_index`, `coverage_fraction`, `quality_flag` | Non-overlapping radial OMR windows |
| `early_windows/` | Same structure as `windows/` | Onset-anchored radial OMR windows |

### Additional Analysis Groups

Other analyzers follow the same `analysis/<analysis_type>_runs/<run_name>/` pattern with
analyzer-specific arrays and provenance attributes. New derived analysis run
families should follow `docs/derived_analysis_run_contract.md` for source refs,
row-axis semantics, and validity/failure state.

`analysis/stimulus_epoch_runs/<run>` is the preferred place for reusable
event-aligned window definitions derived from `analysis/stimulus_runs/<run>`.
Detection occupancy, keypoint summaries, mask summaries, tracking summaries,
and stimulus-response analyses should reference that epoch run when they compute
per-window behavior summaries instead of redefining event windows independently.

`analysis/detection_occupancy_runs/<run>` is the implemented surface for
event-aligned detection heatmaps and coverage summaries. It should reference a
`source_stimulus_epoch_run` and a `source_detection_path` rather than owning
stimulus-window semantics itself. Core child groups are `windows/`,
`coverage/`, `heatmaps/`, and `visualizations/`.

`analysis/chaser_distance_runs/<run>` stores framewise offline fish-to-chaser
distances and epoch summaries for chaser protocols such as GoodCopBadCop. It
consumes refined detection centroids, `analysis/stimulus_runs/<run>`
chaser-state/alignment/calibration data, and
`analysis/stimulus_epoch_runs/<run>` window definitions. Core child groups are
`frames/`, `chasers/`, `positions/`, `distances/`, `epoch_summary/`,
`epoch_distributions/`, and `visualizations/`. The `epoch_distributions/`
group stores fixed-bin distance histograms (`hist_counts`, `hist_density`,
`bin_edges_mm`, `bin_centers_mm`) for fast distribution-shape viewers. See
`docs/chaser_distance_run_contract.md`.

---

## `analysis_metadata/`

Lightweight store for metadata generated during tuning or diagnostics.
Examples:

- `attrs["dish_mask"]` – saved dish mask parameters from the tuner
  (circle center/radius with Hough params or rectangle ROI).
- `attrs["subdish_mask_tuning"]` – multi-dish ROI definitions.
- `attrs["keypoint_tuning"]` – parameters saved from the keypoint tuner
  (`roi_thresh`, `se1_radius`, `se2_radius`, `min_area`, `min_valid_angle`,
  `max_valid_angle`, `min_triangle_area`, plus `tuned_on_frame` and
  `tuned_on_detection`).
  These are stage-local tuned overrides; packaged traditional defaults now
  live under `configs/fisheye/pose_heuristics/traditional_pose/` and should be
  treated as the shared baseline when no tuning block is present.
- `attrs["eye_mask_tuning"]` – parameters saved from the eye mask tuner:
  - `method`: `"global_threshold_otsu"`
  - `version`: Schema version (e.g., `"1.0"`)
  - `tuned_timestamp`: ISO 8601 UTC timestamp
  - `tuned_parameters`:
    - `roi_padding`: Padding around keypoint (int)
    - `pre_threshold`: Pre-threshold cutoff (int or null)
    - `sobel_strength`: Edge subtraction strength 0–1 (float)
    - `min_area`, `max_area`: Region area bounds (int, max may be null)
    - `closing_radius`, `opening_radius`: Morphological radii (int)
    - `min_eye_separation`, `max_eye_separation`: Eye gap bounds (float or null)
  - `context`: ROI index, crop/keypoint runs used, success flag
  - Note: this metadata is expected for traditional eye-mask parameter tuning; YOLO/U-Net model-based runs (including `_analysis.zarr`) may not include it.
- Other agents may add read-only metadata blocks here.

---

## Coordinate Spaces & Calibration Quick Reference

**Question: Which calibration should I use?**

| Data Source | Coordinate Space | Calibration to Use | Location |
|-------------|-----------------|-------------------|----------|
| Offline detections (YOLO, blob) | Camera (4512×4512) | `pixels_per_mm_camera` | `calibration` attrs |
| Keypoint positions | Camera (4512×4512) | `pixels_per_mm_camera` | `calibration` attrs |
| Bounding boxes | Camera (4512×4512) | `pixels_per_mm_camera` | `calibration` attrs |
| Online stimulus positions (chaser, target) | Texture (358×358) | `pixels_per_mm_projector` | `calibration` attrs |
| Refined online positions | Texture (358×358) | `pixels_per_mm_projector` | `refined_online_runs/<run>` attrs |
| Movement runs (online_refined) | Texture (358×358) | `pixels_per_mm_projector` | Already in `pixel_to_mm` attr |
| Movement runs (online, offline) | Camera (4512×4512) | `pixels_per_mm_camera` | Already in `pixel_to_mm` attr |

**Common Pitfalls**:
1. ❌ Don't scale texture positions to camera space and use camera calibration
2. ❌ Don't assume all pixel distances are equal (they're not!)
3. ✅ Always use the calibration matching your coordinate space
4. ✅ Check the `coordinate_space` attribute to know which space you're in

**Typical Values** (for reference):
- `pixels_per_mm_camera`: ~5.8 pixels/mm (camera space)
- `pixels_per_mm_projector`: ~0.44 pixels/mm (texture space)
- `texture_to_camera_scale`: ~12.6 (spatial scaling factor, NOT distance scaling!)

---

## Provenance & Access Tips

- Always inspect run attributes first — they encode the upstream run names,
  configuration, quality summaries, and time spent.
- **Check `coordinate_space` attribute** to determine which calibration to use.
- Array chunking follows the detection/ROI axis for efficient sequential
  reads.  Use `zarr.open_group(path, mode="r")` and slice natively.
- `fisheye.shared.zarr.schema.get_run_group(root, stage)` resolves the run
  path respecting `attrs["latest"]`.
- QA-sensitive tooling should filter using stage-specific reason/metrics arrays.
  For current refined detect/keypoint groups, use `reason_bytes`, then
  `detection_source`. Historical readers may fall back through legacy `reason`.
- **For distance calculations**: Always verify you're using the correct
  `pixels_per_mm` value for your coordinate space.

This document should remain in sync with the schema module and the stage
implementations.  When a new run group or array is added, update both places.
