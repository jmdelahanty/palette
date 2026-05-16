# Cluster Pipeline Migration Checklist
<!-- contract-meta
status: working_checklist
last_verified: 2026-05-16
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
| Crop submitter | present | `scripts/submit_crop_batches_bsub.sh` wraps `fisheye.utils.crop_batch`. |
| Crop + flat ROI cache submitter | present | `scripts/submit_crop_flat_roi_cache_bsub.sh` submits crop geometry and dependent flat-cache publish jobs. |
| Keypoint submitter | present | `scripts/submit_keypoints_batches_bsub.sh` wraps `fisheye.utils.run_keypoints_batch`. |
| Eye-mask submitter | present | `scripts/submit_eye_masks_batches_bsub.sh` wraps `fisheye.utils.run_eye_masks_batch`. |
| Registry discovery | present for first four stages | Registry mode can prefilter by `recording_step_status` and path/camera/rig filters. |
| Model registry resolution | present for detect, keypoints, eye masks | Batch runners can resolve registry models and record candidate provenance. |
| Video decode benchmark | present | `fisheye.diagnostics.benchmark_video_decode` showed PRFS streaming is acceptable for single-pass Decord-GPU detection. |
| Run-group artifact design | documented | `docs/cluster_run_group_artifact_workflow.md` defines the target architecture. |
| Whole-Zarr transfer packing | prototype present | `fisheye.utils.pack_zarr_transfer_artifact` packs whole archives, not individual run groups. |

Palette also has batch utilities for additional stages:

| Utility | Status | Cluster gap |
|---------|--------|-------------|
| `fisheye.utils.refine_detect_batch` | present | No dedicated LSF submitter yet. |
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
- [x] Detection artifact importer apply mode exists:
  `scripts/py -m fisheye.utils.import_run_group_artifact --apply`.
- [x] Post-import validator exists:
  `scripts/py -m fisheye.utils.validate_imported_run_group`.

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

- [ ] Add `scripts/submit_detect_quality_batches_bsub.sh`, or document that
  detect quality remains workstation/local for now.
- [ ] Add `scripts/submit_refine_detect_batches_bsub.sh`.
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

Remaining:

- [ ] Rerun the crop + flat ROI cache cluster smoke after repairing copied
  smoke archive `source_video_path` attrs to point at the PRFS MP4.
- [ ] Verify crop source resolution prefers the refined canonical surface.
- [ ] Verify the crop status JSON, flat cache manifest, published payload size,
  and `open_flat_roi_cache` validation.
- [ ] Add detailed phase telemetry for cache build and publish: video open,
  decode, crop extraction, local write, PRFS copy, manifest publish, and
  validation.
- [ ] Benchmark direct PRFS flat-cache reads versus staging the flat cache to
  node-local scratch before downstream GPU inference.
- [ ] Decide whether materialized crop outputs still need scratch package/import
  when operators explicitly request `crop_storage_mode=materialized`.

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
- [ ] Add a small status summarizer for LSF run directories.
- [ ] Add a failure classifier that distinguishes environment failure,
  registry/model-resolution failure, data-validation failure, and writer
  failure.

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
10. Generalize run-group artifact packer and validator beyond detection.
11. Generalize serialized run-group importer beyond detection.
12. Switch broad production cluster jobs from direct write to artifact/import.

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
