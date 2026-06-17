# Cluster Pipeline Migration Checklist
<!-- contract-meta
status: working_checklist
last_verified: 2026-06-17
purpose: Track what remains to migrate Palette detect, pose, segmentation, and refinement workflows to Janelia cluster execution.
-->

## Purpose

This document is the implementation checklist for moving Palette processing
from workstation-first execution to a defensible cluster workflow.

It is based on a read-through of:

- `docs/operator_guide/pipeline_workflow.md`
- `docs/cluster_batching_guide.md`
- `docs/cluster_workflow_orchestration.md`
- `docs/cluster_run_group_artifact_workflow.md`
- `scripts/submit_detect_batches_bsub.sh`
- `scripts/submit_crop_batches_bsub.sh`
- `scripts/submit_keypoints_batches_bsub.sh`
- `scripts/submit_eye_masks_batches_bsub.sh`
- batch utilities under `src/fisheye/utils/`
- writers under `src/fisheye/detection/`, `src/fisheye/refinement/`, and
  `src/fisheye/segmentation/`

The near-term goal is not to redesign every writer. The near-term goal is to
run small, observable, registry-scoped cluster jobs safely, then close the
gaps needed for larger production runs.

For scheduler-level decisions about splitting GPU inference, CPU refinement,
artifact import, validation, and registry projection into separate LSF jobs,
see `docs/cluster_workflow_orchestration.md`.

## Current State

Palette already has the first layer of cluster support:

| Area | Status | Notes |
|------|--------|-------|
| Environment validation | present | `scripts/validate_cluster_palette_env.sh` checks Python, CUDA, PyTorch, Decord, FFmpeg linkage, reports PyNvVideoCodec/NVIDIA video-library availability, and can require PyNv with `--require-pynvvc`. |
| Detect submitter | present | `scripts/submit_detect_batches_bsub.sh` wraps `fisheye.utils.run_detections_batch`. |
| Detect-quality-refine submitter | present | `scripts/submit_detect_quality_refine_bsub.sh` chains detect, detect_quality, and refined_detect through LSF `done(<jobid>)` dependencies. |
| Detect artifact-quality-refine submitter | present | `scripts/submit_detect_artifact_quality_refine_bsub.sh` submits per-recording scratch artifact detect jobs plus dependent import/validate/quality/refine CPU jobs. |
| Crop submitter | present | `scripts/submit_crop_batches_bsub.sh` wraps `fisheye.utils.crop_batch`. |
| Crop + flat ROI cache submitter | present | `scripts/submit_crop_flat_roi_cache_bsub.sh` submits crop geometry and dependent flat-cache publish jobs. |
| Clipped collection flat ROI cache submitter | present | `scripts/submit_clipped_collection_flat_roi_cache_bsub.sh` submits finalized clipped collection cache materialization and manifest-last publish. |
| Keypoint submitter | present | `scripts/submit_keypoints_batches_bsub.sh` wraps `fisheye.utils.run_keypoints_batch`. |
| Eye-mask submitter | present | `scripts/submit_eye_masks_batches_bsub.sh` wraps `fisheye.utils.run_eye_masks_batch`. |
| Registry discovery | present for first four stages | Registry mode can prefilter by `recording_step_status` and path/camera/rig filters. |
| Model registry resolution | present for detect, keypoints, eye masks | Batch runners can resolve registry models and record candidate provenance. |
| Video decode benchmark | present | `fisheye.diagnostics.benchmark_video_decode` showed PRFS streaming is acceptable for single-pass Decord-GPU detection. |
| Run-group artifact design | documented | `docs/cluster_run_group_artifact_workflow.md` defines the target architecture. |
| Whole-Zarr transfer packing | prototype present | `fisheye.utils.pack_zarr_transfer_artifact` packs whole archives, not individual run groups. |
| Rolling-clip planning/materialization | active pilot present | Planner/materializer/verifier, frame-index builder, metadata-only analysis-Zarr shell creator, clip-local detect artifact import, detect-quality/refined-detect chaining, and finalized collection writing have all passed the sleepyfish all-clips smoke. Core reader/editor support remains in migration. |

Palette also has batch utilities for additional stages:

| Utility | Status | Cluster gap |
|---------|--------|-------------|
| `fisheye.utils.refine_detect_batch` | present | Covered by the chained detect-quality-refine submitter for registry-discovered targets. |
| `fisheye.utils.refine_keypoints_batch` | present | No dedicated LSF submitter yet. |
| `fisheye.utils.run_subject_mask_batch_pipeline` | present | No dedicated LSF submitter yet; artifact/import policy not implemented. |
| subject-mask review/finalization utilities | present | Some tools are operator-driven and need cluster-safe boundaries. |
| eye-mask refinement | present inside `run_eye_masks_batch --refine` and standalone code | Cluster docs focus on raw eye-mask inference more than refined surfaces. |

## Immediate Pilot Goal

Run one or a few recordings end-to-end enough to prove the cluster can process
real data without changing the canonical architecture yet.

For the pilot, direct writes to each recording's analysis Zarr are acceptable
if all of these are true:

- jobs are limited to one writer per target analysis Zarr;
- `--max-active` is conservative;
- run names are unique or overwrite is explicit;
- outputs are validated immediately after the job;
- registry updates are treated as projections from Zarr state, not as the
  source of truth.

This is intentionally lower than the final production bar. The run-group
artifact workflow remains the target for larger, high-concurrency production
runs.

For detection, the intended production shape is: stream the source video from
PRFS/NRS, write the new detect run group to
`/scratch/$USER/$LSB_JOBID`, package that run group as a transfer artifact, and
promote it into the canonical analysis Zarr through a serialized importer. Do
not treat `/tmp` as the cluster scratch target, and do not make the tarball the
canonical data format.

For future clipped recordings, the intended production shape is similar but the
target namespace is clip-local:

```text
clips/<clip_id>/cameras/<camera_serial>/<run_family>/<run_name>
```

The cluster should parallelize model compute by clip, then run a serialized
finalize stage that owns shared experiment metadata, logical latest aliases,
collection manifests, consolidated metadata, and registry projections. Do not
make many clip jobs append into one top-level global run group.

## Stage Checklist

### 0. Environment And Inputs

- [x] Build a cluster-compatible `palette-py311` environment.
- [x] Validate CUDA PyTorch on L4 GPU.
- [x] Build Decord against conda FFmpeg and CUDA/NVDEC.
- [x] Add `scripts/validate_cluster_palette_env.sh`.
- [x] Confirm Decord GPU smoke returns CUDA tensors.
- [x] Add explicit PyNvVideoCodec/NVIDIA video-library validation for PyNv parity jobs.
- [x] Benchmark PRFS versus local `/tmp` video decode throughput.
- [ ] Add a short operator runbook for rebuilding Decord on cluster nodes.
- [ ] Decide whether this environment should remain conda-based or move to
  Apptainer after the first production smoke.

### 0b. Rolling-Clip Archive Shape

Current implementation:

- `fisheye.utils.plan_orange_style_clips`
- `fisheye.utils.materialize_orange_style_clips`
- `fisheye.utils.verify_orange_style_clips`

Ready:

- [x] Orange rolling-clip layout audited.
- [x] Keyframe-aligned planning utility exists.
- [x] Retroactive materializer exists for one camera stream at a time.
- [x] Verifier exists for structural checks and optional ffprobe packet counts.
- [x] Sleepyfish long recordings were materialized and verified as
  Orange-style clips.
- [x] `fisheye.utils.build_recording_frame_index` exists and supports clipped
  and single-video layouts.
- [x] `fisheye.utils.create_clipped_analysis_zarr` exists and creates a
  metadata-only clipped analysis shell with clip-camera namespaces.
- [x] Real smoke on
  `/nvme1/recordings/sleepyfish_2026_05_05_17_45_30_cam2010093` wrote
  `recording_frame_index.parquet` and manifest with 1,188,000 rows and
  zero validation failures.
- [x] Real shell smoke on the same sleepyfish recording wrote
  `/tmp/palette_clipped_analysis_shell_smoke/sleepyfish_2026_05_05_17_45_30_cam2010093_analysis.zarr`
  with 22 clips, 22 clip-camera rows, and 1,188,000 indexed frames.
- [x] Consumer mapping contract exists:
  `docs/clipped_recording_consumer_mapping_contract.md`.

Remaining:

- [x] Build `recording_frame_index.parquet` and
  `recording_frame_index_manifest.json` from `recording_clip_index` plus
  per-clip metadata CSVs.
- [x] Create parent analysis-Zarr shell for clipped recordings.
- [x] Extend run-group artifact manifests/importer to target
  `clips/<clip_id>/cameras/<camera_serial>/<family>/<run_name>`.
- [x] Add a clip workflow finalizer that validates expected clip coverage and
  writes `experiment_index/finalized_runs/<workflow_id>`.
- [x] Run a full clipped detect -> import -> detect-quality -> refined-detect
  -> validate -> finalize cluster smoke on the PRFS sleepyfish clipped
  recording. Workflow `sleepyfish_cam2010093_allclips_20260517_01` completed
  `133/133` stages, finalized 22 clip-camera refined-detect runs, and resolved
  1,188,000 frame mappings with no unselected frame pairs.
- [ ] Teach core readers to resolve finalized clip collections in addition to
  traditional top-level run groups.
- [ ] Teach Crimson-facing readers to use Palette's finalized collection
  resolver instead of independently flattening `clips/` directories.
- [ ] Define temporal boundary policies for track kinematics, bout detection,
  and other stateful stages before enabling clip-local execution for them.

Read-only stale-assumption audit from 2026-05-16:

- Batch submitters still discover whole `*_analysis.zarr` stores.
- Existing writers generally update top-level run-family `latest` attrs
  directly.
- The current run-group importer only accepts top-level
  `<run_family>/<run_name>` targets.
- Reader/status tools mostly resolve top-level latest runs and do not yet know
  about finalized clip collections.
- Current Zarr schema helpers are still single-video-first for normal import
  paths; clipped analysis shell creation is implemented separately but core
  readers/writers do not yet target clip-camera namespaces.

### 1. Detect

Current implementation:

- `src/fisheye/detection/detect_yolo.py`
- `src/fisheye/utils/run_detections_batch.py`
- `scripts/submit_detect_batches_bsub.sh`

Ready for pilot:

- [x] Registry-scoped discovery exists.
- [x] Registry model resolution exists.
- [x] LSF submitter exists.
- [x] Decord-GPU tensor resize bug fixed.
- [x] Decode/storage policy documented.
- [x] Explicit `--model` path is supported for cluster smoke runs that do not
  have registry model metadata available on the cluster.
- [x] Compute-only detection smoke CLI exists:
  `scripts/py -m fisheye.diagnostics.detect_compute_smoke`.
- [x] Detection run-group artifact runner exists:
  `scripts/py -m fisheye.utils.run_detection_artifact`.
- [x] Detection artifact runner and LSF submitter can record clip-camera
  context (`workflow_id`, `recording_id`, `clip_id`, `clip_index`, and
  `camera_serial`) in submission logs, job stdout, artifact summaries, and
  manifests.
- [x] Detection artifact importer apply mode exists:
  `scripts/py -m fisheye.utils.import_run_group_artifact --apply`.
- [x] Post-import validator exists:
  `scripts/py -m fisheye.utils.validate_imported_run_group`.
- [x] Full-recording artifact-chain submitter exists:
  `scripts/submit_detect_artifact_quality_refine_bsub.sh`.
- [x] Artifact-chain submitter can use registry model resolution during dry-run
  planning (`run_detections_batch --resolve-models`) or an explicit `--model`.
- [x] Artifact-chain postprocess pins deterministic detect and detect-quality
  run names before invoking refined detect, avoiding mutable `latest`
  selection in the safe broad-run path.
- [x] Cluster detect submitters expose explicit inference resize controls and
  the chained submitters default to `pynvvc_nv12_rgb` with `640x640` input so
  broad runs do not silently use full-frame tensor inference.
- [x] Runtime detection rejects GPU tensor decoder paths without explicit
  resize dims, covering both PyNvVideoCodec and Decord GPU.

Remaining:

- [x] Run a compute-only cluster detection smoke on the sickyfish PRFS
  recording: PRFS video open, model load, small-batch decode, inference, and no
  canonical `detect_runs` writes.
- [x] Add minimal scratch run-group artifact/import path for detection before a
  full output-writing cluster smoke.
- [x] Run one real cluster detection artifact/import smoke on the sickyfish PRFS
  recording.
- [x] Verify detect output strict JSON and schema.
- [x] Verify run provenance includes model, decode backend, timings, git
  commit, host, and CUDA device.
- [ ] Verify registry `detect` status refresh after cluster run.
- [x] Decide whether detect should be the first stage converted to scratch
  run-group artifacts.
- [ ] Run one registry-scoped full-recording batch through
  `scripts/submit_detect_artifact_quality_refine_bsub.sh` and audit the
  imported detect, detect-quality, and refined-detect outputs before promoting
  this as the default broad production path.

### 2. Detect Quality And Refined Detect

Current implementation:

- `src/fisheye/refinement/detect_quality.py`
- `src/fisheye/refinement/refine_detect.py`
- `src/fisheye/utils/detect_quality_batch.py`
- `src/fisheye/utils/refine_detect_batch.py`

Ready for pilot:

- [x] Batch utilities exist.
- [x] Refined detect writes sparse canonical `instances` and
  `source_detections`.
- [x] Raw `detect_runs` are treated as immutable model outputs.
- [x] Registry-free smoke from an imported cluster detect run succeeded:
  detect quality wrote `quality_reports/<run>` under the imported detect run,
  and `refine_detect` consumed that explicit quality run to produce a curated
  `refined_detect_runs/<run>`.
- [x] Clip-local smoke from an imported cluster detect artifact succeeded:
  `import_run_group_artifact --use-intended-target` promoted the run under
  `clips/<clip_id>/cameras/<camera_serial>/detect_runs/<run>`, detect quality
  wrote clip-local `quality_reports/<run>`, and `refine_detect` wrote
  `clips/<clip_id>/cameras/<camera_serial>/refined_detect_runs/<run>`.
- [x] The imported-detect validator preserves the imported model-output core
  fingerprint while allowing the known mutable derived child
  `quality_reports/`.

Execution model note:

- `detect_quality` and `refine_detect` are single-process, single-writer stages
  today. Do not add Dask inside these writers just to run them on the cluster;
  scale them first by recording or clip namespace. If they later become
  artifact-produced stages, package/import the completed run group and keep
  shared metadata updates serialized.

Remaining:

- [x] Add a chained detect-quality/refine submitter for registry-discovered
  targets: `scripts/submit_detect_quality_refine_bsub.sh`.
- [x] Add clip-aware validation/reporting utilities for imported detect and
  refined-detect paths so operators do not need to inspect nested `zarr.json`
  metadata manually.
- [x] Verify all-clips cluster fan-out performance on a long clipped recording:
  22 concurrent GPU detect jobs finished in `7m37s` wall time versus
  `~2h39m` summed one-GPU detect/artifact time, with full workflow completion
  in `~9m03s` from submission to finalizer.
- [ ] Add an explicit detect-quality report validator if the quality report
  becomes a hard contract separate from `refine_detect` consumption.
- [ ] Ensure registry discovery can select recordings with `detect='ok'` and
  missing or stale `refined_detect`.
- [ ] Ensure `refine_detect_batch` emits cluster-friendly JSONL logs and
  records LSF context through shared provenance.
- [ ] Define validation: curated row count, outside-dish-mask filtering,
  source candidate audit rows, `latest`, strict JSON, registry status.

### 3. Crop

Current implementation:

- `src/fisheye/utils/crop_batch.py`
- `src/fisheye/tracking/crop.py`
- `scripts/submit_crop_batches_bsub.sh`

Ready for pilot:

- [x] Registry-scoped discovery exists.
- [x] LSF submitter exists.
- [x] Geometry-only analysis crop policy is implemented in batch planning.
- [x] Two-job crop + flat ROI cache submitter exists; the cache job depends on
  successful crop completion.
- [x] Flat ROI cache builder writes to node-local scratch and publishes
  payload-first/manifest-last to `/misc/public/palette_cache/<workflow_id>/`.
- [x] GPU decode path exists for external video crop generation.
- [x] `decode_seconds`, `compute_seconds`, and `write_seconds` style profiling
  exists in crop code paths.
- [x] Sequential PyNvVideoCodec flat-cache materialization path exists
  (`pynvvc_luma`) and avoids the Decord open/random-access behavior that made
  the first long-video cache smoke impractical.
- [x] Finalized clipped refined-detect collections can be materialized directly
  into a flat ROI cache with a sidecar row-index parquet via
  `fisheye.utils.build_clipped_collection_flat_roi_cache`. This avoids creating
  a synthetic parent-level crop run before pose/segmentation cache generation.

Remaining:

- [x] Run the clipped-collection flat-cache builder on the all-clips sleepyfish
  finalized collection with a small local `--limit-rows` smoke.
- [ ] Run the clipped-collection flat-cache LSF wrapper on the all-clips
  sleepyfish finalized collection with `--limit-rows 1024`, then without the
  limit if the row-index and payload validation pass.
- [x] Add an LSF submit wrapper for clipped-collection flat-cache generation
  that builds on node-local scratch, publishes `.bin`, `.rows.parquet`, then
  `.json` manifest to PRFS workflow cache storage.
- [ ] Rerun the legacy crop-run + flat ROI cache cluster smoke after repairing
  copied smoke archive `source_video_path` attrs to point at the PRFS MP4.
- [ ] Verify crop source resolution prefers the refined canonical surface.
- [ ] Verify the crop status JSON or collection cache progress JSONL, flat
  cache manifest, row-index parquet, published payload size, and
  `open_flat_roi_cache` validation.
- [x] Add detailed phase telemetry for cache build and publish: video open,
  decode, crop extraction, local write, PRFS copy, manifest publish, and
  validation.
- [x] Add explicit crop/cache pixel-contract metadata for future crop runs and
  downstream pose/segmentation runs (`roi_image_representation`,
  `roi_pixel_contract`, `source_roi_image_representation`,
  `source_roi_pixel_contract_name`, `source_roi_pixel_contract`).
- [x] Decide the accepted production crop pixel contract for pose/segmentation
  input caches. Current materialized readers expose grayscale `uint8` ROI crops,
  but historical paths differ in conversion details: OpenCV weighted grayscale,
  Decord GPU channel mean, and current PyNv luma-plane cropping.
  2026-05-16 Orange runtime audit selected `pynvvc_luma_v1` for mono Orange
  camera recordings: `[N,H,W] uint8` decoded NV12 Y/luma crops, with
  engine-specific resize/letterbox/channel replication/normalization performed
  by the model runtime.
- [x] Add a crop-pixel parity utility for the canonical `CropImageSource` path
  versus flat ROI caches. It reports byte equality, max/mean/p95 absolute
  difference, and top mismatched rows for fixed ROI rows.
- [x] Add a training-zarr parity utility for comparing stored
  `crop_runs/<run>/roi_images` against PyNvVideoCodec luma crops reconstructed
  from the original video, including `raw_video/original_frame_indices` mapping
  for sampled training zarrs.
- [x] Audit current training-zarr crop migration coverage. As of 2026-05-16,
  52/60 approved detector-training source zarrs were crop-bearing and all 52
  had a `pynvvc_luma_v1` crop run. The remaining 8 `sickyfish`/`sleepyfish`
  sampled zarrs were initially detection-only, but that snapshot is now
  historical: those archives are being promoted into crop/keypoint training
  sources as explicit crop geometry and PyNvVC-luma crops are added. Inspect
  each archive's current `crop_runs.latest`, `keypoints_runs.latest`, and
  `refined_keypoints_runs.latest` before treating it as detection-only.
- [ ] Run `fisheye.diagnostics.check_flat_roi_cache_pixel_parity` on each new
  flat ROI cache before using it for pose/segmentation quality validation.
- [x] Run `fisheye.diagnostics.check_training_crop_pynvvc_pixel_parity` on at
  least one existing per-recording training zarr before accepting `pynvvc_luma`
  as a model-facing crop cache contract. The first quick check was not
  byte-identical to historical crop pixels, which is expected because
  `pynvvc_luma_v1` is an explicit raw-luma contract rather than the old
  OpenCV/Decord grayscale conversion.
- [x] Decide whether `pynvvc_luma` is acceptable for production
  pose/segmentation caches or implement a canonical backend such as
  `pynvvc_legacy_gray` / `pynvvc_nv12_gray`. For mono Orange recordings,
  `pynvvc_luma_v1` is the accepted contract.
- [ ] If `pynvvc_luma` fails strict crop-pixel parity, implement a parity-safe
  sequential PyNv backend and make `--decode-backend auto` prefer that backend
  rather than falling back to slow `read_slice` for long-video workflows.
- [ ] Benchmark direct PRFS flat-cache reads versus staging the flat cache to
  node-local scratch before downstream GPU inference.
- [ ] Decide whether materialized crop outputs still need scratch package/import
  when operators explicitly request `crop_storage_mode=materialized`.

#### Clipped Collection ROI Cache To Pose/Segmentation Slice

Goal: consume the finalized clipped refined-detect collection directly, build a
shared runtime ROI cache, and prove pose/segmentation can use that cache without
creating a synthetic parent-level crop run.

Implementation checklist:

- [x] Build collection-aware flat ROI cache writer:
  `fisheye.utils.build_clipped_collection_flat_roi_cache`.
- [x] Write `.flat_roi_cache.bin`, `.flat_roi_cache.json`, and required
  `.flat_roi_cache.rows.parquet` sidecar.
- [x] Record `pynvvc_luma_v1` pixel contract and source collection provenance
  in the cache manifest.
- [x] Make refined `instances/bbox_img_xyxy` the preferred row-level geometry
  source; root `width`/`height` remain valid metadata and fallback dimensions
  when normalized-only rows are encountered.
- [x] Define the refined bbox coordinate contract: `bbox_img_xyxy` is source-image
  pixel-space; `bbox_norm_coords` remains normalized `cxcywh` with explicit
  reference-dimension attrs, often the detector inference image.
- [x] Add dry-run/apply backfill support for older refined runs whose
  `bbox_img_xyxy` was materialized in inference-image pixels.
- [x] Add focused unit coverage for clipped cache pixels and row-index lineage.
- [x] Run a limited real-data smoke against
  `sleepyfish_cam2010093_allclips_20260517_01`.
- [x] Add `scripts/submit_clipped_collection_flat_roi_cache_bsub.sh`.
- [x] Make the LSF wrapper write a job script, stdout/stderr, status JSON,
  progress JSONL, submission context, and final manifest path.
- [x] Make the wrapper build under `$PALETTE_JOB_CACHE`/node scratch and publish
  payload, row-index parquet, and manifest to
  `/misc/public/palette_cache/<workflow_id>/roi_cache` in manifest-last order.
- [ ] Run an LSF limited smoke with `--limit-rows 1024` and validate manifest,
  payload size, row-index parquet, `open_flat_roi_cache`, and progress
  telemetry.
- [ ] Run the full all-clips sleepyfish cache build if the limited smoke passes.
- [ ] Add a row-index-aware ROI cache source/adapter for downstream
  pose/segmentation jobs so those jobs do not parse `.bin` or assume row number
  equals parent frame identity.
- [ ] Add pose smoke over the clipped collection cache using a small row limit.
- [ ] Add segmentation smoke over the clipped collection cache using a small row
  limit.
- [ ] Decide whether downstream jobs should read PRFS cache directly or stage
  `.bin`/`.rows.parquet` to node scratch before GPU inference.
- [ ] Record downstream output provenance with cache manifest path, row-index
  schema, pixel contract, source finalized collection id, LSF job id, host, GPU,
  and git commit.

### 4. Keypoints And Refined Keypoints

Current implementation:

- `src/fisheye/detection/detect_keypoints_yolo.py`
- `src/fisheye/refinement/refine_keypoints.py`
- `src/fisheye/utils/run_keypoints_batch.py`
- `src/fisheye/utils/refine_keypoints_batch.py`
- `scripts/submit_keypoints_batches_bsub.sh`

Ready for pilot:

- [x] Registry-scoped keypoint discovery exists.
- [x] Registry pose-model resolution exists.
- [x] LSF keypoint submitter exists.
- [x] `run_keypoints_batch` can delegate refinement.

Remaining:

- [ ] Run one cluster keypoint smoke after crop is available.
- [ ] Decide whether keypoint refinement is always part of cluster keypoint
  jobs or a separate job family.
- [ ] Add `scripts/submit_refine_keypoints_batches_bsub.sh` if refinement is
  split.
- [ ] Verify registry refresh covers raw keypoint performance and refined
  keypoint quality.
- [ ] Verify provenance records source crop run, model resolution, run command,
  LSF job id, host, CUDA device, and git commit.

### 5. Eye Masks And Refined Eye Masks

Current implementation:

- `src/fisheye/segmentation/eye_segmentation_yolo.py`
- `src/fisheye/segmentation/infer_unet_eye_masks.py`
- `src/fisheye/refinement/refine_eye_masks.py`
- `src/fisheye/utils/run_eye_masks_batch.py`
- `scripts/submit_eye_masks_batches_bsub.sh`

Ready for pilot:

- [x] Registry-scoped discovery exists.
- [x] Registry model resolution exists for YOLO/U-Net modes.
- [x] LSF submitter exists.
- [x] Runner can refine as part of the eye-mask path.

Remaining:

- [ ] Run one cluster eye-mask smoke after crop/keypoints are available.
- [ ] Decide current default method for cluster runs: U-Net versus YOLO.
- [ ] Ensure refined eye-mask compatibility surfaces are either generated or
  explicitly deferred.
- [ ] Verify registry refresh covers eye-mask performance, eye-mask quality,
  and any subject-mask metrics produced by this runner.
- [ ] Verify cluster provenance captures model resolution and whether crop
  cache paths were used.

### 6. Subject Masks And Refined Subject Masks

Current implementation:

- `src/fisheye/segmentation/infer_unet_subject_masks.py`
- `src/fisheye/segmentation/subject_segmentation.py`
- `src/fisheye/refinement/finalize_subject_masks.py`
- `src/fisheye/refinement/refine_subject_masks.py`
- `src/fisheye/tune/refined_subject_mask_review.py`
- `src/fisheye/utils/run_subject_mask_batch_pipeline.py`
- `scripts/run_subject_mask_batch_pipeline`

Ready for pilot:

- [x] Batch pipeline entry point exists.
- [x] Writers use shared stage provenance in major subject-mask paths.
- [x] Review/finalization tooling exists.

Remaining:

- [ ] Add `scripts/submit_subject_mask_batches_bsub.sh`.
- [ ] Decide which subject-mask path is the cluster default:
  U-Net inference, SAM, traditional, or existing smart finalizer inputs.
- [ ] Separate operator review surfaces from pure cluster compute surfaces.
- [ ] Ensure finalization writes cluster timing/provenance consistently.
- [ ] Decide whether subject-mask outputs require run-group artifact import
  before broad cluster use, because mask stores are large and file-heavy.
- [ ] Add registry-scoped discovery and skip-existing behavior if missing from
  the subject-mask batch pipeline.
- [ ] Add validation: component presence, geometry metrics, profile artifacts,
  strict JSON, and registry projection.

## Cross-Cutting Implementation Checklist

### A. Cluster Provenance

Shared stage provenance is already widely used, but cluster-specific fields
should be consistent across every cluster-capable writer.

Required cluster fields:

- `palette_git_commit`
- `command`
- `hostname`
- `cluster_scheduler`
- `cluster_job_id`
- `cluster_task_id`
- `allocated_slots`
- `cuda_visible_devices`
- `torch_cuda_device`
- `decoder_backend` when video decode is used
- `wall_seconds`
- stage-specific phase timings

Implementation tasks:

- [ ] Add a shared helper that reads LSF environment variables and returns a
  JSON-safe cluster context payload.
- [ ] Use that helper from detect, crop, keypoints, eye masks, subject masks,
  and refinement/finalization writers.
- [ ] Add tests that the helper handles unset variables for workstation runs.
- [ ] Extend flat ROI cache submitters beyond coarse status JSONs with
  per-phase timings and LSF resource context.

### B. Registry And Stage Status

The cluster docs currently define the cleanest DAG for the first four batch
stages:

```text
detect -> crop -> keypoints -> eye_masks
```

The operator workflow includes additional stages:

```text
detect_quality -> refined_detect -> refined_keypoints -> subject_masks
```

Implementation tasks:

- [ ] Decide the canonical stage-status vocabulary for cluster queries.
- [ ] Ensure registry status rows exist for refined stages that cluster jobs
  should schedule independently.
- [ ] Add registry discovery filters for `refined_detect`,
  `refined_keypoints`, `subject_masks`, and `refined_subject_masks`.
- [ ] Keep registry as a fast query projection, not the authoritative source
  of the data. Zarr run groups remain authoritative.

### B2. Clipped Recording Orchestration

Policy:

- Treat `(recording_id, camera_serial, clip_id)` as the cluster work unit.
- Run per-clip detect, import, validation, detect quality, and refined detect
  independently in clip-local namespaces.
- Use CPU-only jobs for import, validation, detect quality, and refined detect;
  do not hold a GPU after model inference completes.
- Fan in with one recording-level finalizer that writes the collection
  manifest, logical latest alias, consolidated metadata refresh, and registry
  projection when enabled.

Implementation tasks:

- [x] Add a dry-run clip inventory/planner that emits one work item per
  clip-camera:
  `scripts/py -m fisheye.utils.plan_clipped_detect_refine_workflow`.
- [x] Add optional deterministic run-name controls for detect artifact,
  detect-quality, and refined-detect stages so dependent jobs do not need to
  discover timestamped names from previous outputs.
- [x] Add a dry-run-safe submitter that consumes the plan and creates per-clip
  dependent job scripts/manifests:
  `scripts/py -m fisheye.utils.submit_clipped_detect_refine_plan_bsub`.
- [ ] Promote the submitter from one-clip smoke mode to broad fan-out after the
  one-clip chain is validated on PRFS.
- [x] Add a refined-detect validator after the per-clip chain:
  `scripts/py -m fisheye.utils.validate_refined_detect_run`.
- [x] Add a real recording-level finalizer after the per-clip chain:
  detect artifact -> import -> validate -> detect quality -> refine detect ->
  validate refined detect -> finalize collection.
  `scripts/py -m fisheye.utils.finalize_clipped_detect_refine_workflow`
  validates every planned clip-camera refined run and writes
  `experiment_index/finalized_runs/<workflow_id>` only when all checks pass.
- [x] Add per-stage JSON reports that include clip id, camera serial, run
  names, target group paths, job id, queue, host, timing, and validation state.
- [x] Add a recording-level finalizer that verifies expected clip-camera
  refined outputs, per-clip `frame_counts` length, and
  `recording_frame_index.parquet` clip/frame continuity before writing a
  finalized collection manifest.
- [x] Add retry/idempotence guardrails: submit-mode preflight refuses existing
  planned detect, detect-quality, refined, or finalized collection targets
  unless `--allow-existing-outputs` is explicitly passed.
- [x] Add an operator-facing submission checker:
  `scripts/py -m fisheye.utils.check_clipped_detect_refine_submission`
  summarizes detect artifact, CPU stage, and finalizer status from
  `submission_manifest.json` and can be used with `--require-complete` as the
  one-clip smoke gate.
- [ ] Run the one-clip PRFS smoke and require this gate to pass before enabling
  all-clip fan-out:
  `scripts/py -m fisheye.utils.check_clipped_detect_refine_submission <run_dir>/submission_manifest.json --require-complete`.
- [x] Add a collection resolver for downstream readers so they can map
  `recording_frame_id` to `(clip_id, clip_local_frame_index, run_path)`.
  `scripts/py -m fisheye.utils.resolve_clipped_refined_detect_collection`
  resolves a finalized collection and can export a frame/run mapping parquet.
- [ ] Add optional resume/skip-valid-output mode after one-clip finalizer smoke
  proves the fail-closed path on PRFS.

### C. Direct Write Versus Artifact Import

Short-term:

- use direct writes for one-recording or low-concurrency pilot jobs;
- never run two jobs that write the same run family/run name into the same
  archive;
- keep `--max-active` conservative.

Production target:

- cluster job writes complete run-group artifact on
  `/scratch/$USER/$LSB_JOBID`;
- serialized importer validates and promotes into canonical Zarr;
- importer updates `latest`, consolidated metadata, and registry projection.

For future long-running experiments split into clips, the importer should lock
the smallest mutable namespace that protects correctness. Current
single-recording archives usually need archive-level import serialization.
Clip-partitioned experiment stores can import disjoint clip-local run groups in
parallel, then run a short serialized experiment-level finalize step for shared
indexes, `latest`, consolidated metadata, and registry projection.

Implementation tasks:

- [ ] Add run-group packer. Existing `pack_zarr_transfer_artifact` packs whole
  archives and is not enough for per-run import.
- [ ] Add run-group artifact validator.
- [ ] Add serialized run-group importer with `.incoming` and `.failed`
  handling.
- [ ] Add file or SQLite locks so one importer owns a mutable namespace
  (archive-level for current stores; clip-level plus experiment-level finalize
  for future clip-partitioned stores).
- [ ] Add tests for failed validation leaving normal namespaces untouched.

### D. Validation Gates

Every cluster stage should have the same validation shape:

- dry-run discovery output;
- apply smoke on one recording;
- strict JSON metadata check;
- stage-specific required array check;
- source-run lineage/fingerprint check;
- registry projection refresh check;
- consumer smoke where relevant.

Initial consumer checks:

- Crimson load smoke for detect, refined detect, keypoints, masks, and compact
  analysis surfaces.
- Marimo read smoke for analysis surfaces used by notebooks.

### E. Logs And Observability

Implementation tasks:

- [ ] Make every submitter produce a manifest summary, per-batch path list,
  stdout/stderr, and JSONL stage log.
- [ ] Standardize log directory selection for PRFS versus fallback scratch.
- [x] Add a small status summarizer for clipped detect/refine LSF submission
  manifests:
  `scripts/py -m fisheye.utils.check_clipped_detect_refine_submission`.
- [ ] Add a failure classifier that distinguishes environment failure,
  registry/model-resolution failure, data-validation failure, and writer
  failure.

### B3. Shared Artifact-Orchestration Extraction

Policy:

- Do not build a generic DAG framework before the detect artifact-chain path
  has completed at least one real full-recording batch and one clipped batch.
- Extract shared helpers from working submitters only when repeated code has
  stable semantics: target planning, deterministic run naming, job-script
  writing, LSF job-id parsing, dependency expression construction,
  per-target submission manifests, and post-import validation.
- Keep stage-specific writers and validators explicit. A shared orchestrator
  should call stage-specific artifact builders/import validators; it should not
  hide stage contracts behind stringly-typed generic commands.

Implementation tasks:

- [ ] Audit `scripts/submit_detect_artifact_quality_refine_bsub.sh` after a
  real batch and record which generated logs/operators were useful.
- [ ] Compare repeated pieces with
  `src/fisheye/utils/submit_clipped_detect_refine_plan_bsub.py`.
- [x] Extract only stable shell pieces into `scripts/lib/palette_lsf.sh`
  (`bsub` job-id parsing, command printing, and dry-run-vs-submit handling).
- [ ] Extract additional helpers only after a real artifact-chain batch shows
  the target logs and manifests are sufficient.
- [ ] Leave direct stage submitters available as pilot/debug paths until the
  artifact-chain path has enough successful production mileage.

## Recommended Implementation Order

1. Run a compute-only cluster detect smoke from PRFS on the sickyfish recording
   without writing `detect_runs` chunks to canonical Zarr.
2. Implement the minimal detect run-group artifact/import path using
   `/scratch/$USER/$LSB_JOBID`, `.incoming`, and `.failed`.
3. Run one manual cluster detect artifact/import smoke from PRFS on the
   sickyfish recording.
4. Validate the detect run on the workstation: strict JSON, registry, and
   viewer/notebook readiness.
5. Add the shared LSF provenance helper.
6. Add `submit_refine_detect_batches_bsub.sh`.
7. Add `submit_refine_keypoints_batches_bsub.sh`.
8. Add `submit_subject_mask_batches_bsub.sh`.
9. Add registry discovery/status for refined stages that are missing from the
   first-four-stage DAG.
10. Run a full-recording registry-scoped detect artifact-quality-refine batch.
11. Audit the generated artifact, import, validation, detect-quality, and
    refined-detect logs.
12. Extract stable orchestration helpers from the detect-specific submitter and
    clipped submitter; do not introduce a broad DAG framework before this audit.
13. Generalize run-group artifact packer and validator beyond detection.
14. Generalize serialized run-group importer beyond detection.
15. Switch broad production cluster jobs from direct write to artifact/import.

For clipped sleepyfish-style recordings, use this narrower order before broad
parallel runs:

1. Build a clip-camera work-item inventory from the clipped analysis Zarr and
   frame-index sidecars.
2. Run one more clip end-to-end through detect artifact -> import -> validate
   -> detect quality -> refined detect -> validate refined detect.
3. Add the workflow submitter that can submit the same chain for all clips with
   bounded active GPU jobs.
4. Add the recording-level finalizer and collection manifest.
5. Run the full clipped recording only after the finalizer can fail closed on
   missing, failed, duplicated, or stale clip outputs.

## Open Questions

1. Should detect quality be a separate cluster stage, or should it be folded
   into detect/refine-detect jobs for now?
2. Should keypoint refinement always run inside keypoint jobs, or be scheduled
   independently?
3. Which subject-mask implementation should be the default cluster path?
4. Which stages should be first converted to run-group artifact/import?
   Dense crop and mask stages are the likely first candidates.
5. Should broad production cluster runs update registry rows from workers, or
   should registry updates happen only during the serialized import step?

## Near-Term Definition Of Done

The migration is ready for a small production batch when:

- one real recording has passed detect, crop, keypoints, eye masks, and the
  necessary refined stages on the cluster;
- every stage output passes strict JSON and stage-specific schema validation;
- registry status and performance projections match the Zarr state;
- Crimson or Marimo can read the resulting outputs;
- cluster provenance includes scheduler, host, GPU, command, git, model, and
  timing information;
- documented submit commands exist for every stage used in the batch;
- no stage requires repeated full-video copies to local scratch for normal
  single-pass processing.
