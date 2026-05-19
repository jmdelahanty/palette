# Cluster Run-Group Artifact Workflow
<!-- contract-meta
status: design
last_verified: 2026-05-16
purpose: Define how Palette cluster jobs should produce Zarr-compatible run-group artifacts on node-local scratch and safely import them into canonical analysis archives.
-->

## Purpose

Palette analysis archives are Zarr stores with many chunk and metadata files.
That layout is useful for local partial reads, but direct high-concurrency
mutation over shared storage is a poor fit for cluster jobs that decode large
videos or write many derived arrays.

The cluster-safe pattern is:

1. run compute on node-local scratch;
2. write complete Zarr-compatible output run groups on scratch;
3. validate the scratch result;
4. package the result as an immutable transfer artifact;
5. import it into the canonical analysis Zarr with a serialized importer.

This document defines that workflow. It is a storage and provenance contract,
not a new analysis schema.

For the broader detect, pose, segmentation, and refinement migration checklist,
see `docs/cluster_pipeline_migration_checklist.md`.

For scheduler-level orchestration across GPU, CPU, import, validation, and
registry-projection jobs, see `docs/cluster_workflow_orchestration.md`.

For geometry-only crop runs and temporary ROI cache placement across cluster
pose/segmentation workflows, see
`docs/geometry_only_crop_workflow_cache_design.md`.

For Orange's native rolling-clip recording layout and Palette's future
clip-table import shape, see
`docs/orange_rolling_clip_recording_contract.md`.

For clipped training-Zarr creation from that layout, including the Parquet
frame-map decision, see
`docs/clipped_training_zarr_implementation_checklist.md`.

## Design Decision

Cluster workers should not write directly into canonical analysis Zarrs on
shared storage during active compute. A worker may read source metadata and
arrays, but the durable mutation of the canonical archive should happen in a
separate import step.

The unit of exchange is a complete run-group package. Examples:

```text
detect_runs/detect_...
analysis/swim_bout_runs/bouts_...
analysis/bout_kinematics_runs/bk_...
analysis/eye_angle_runs/eye_angle_...
```

The package is immutable after creation. If the run needs to be regenerated,
create a new run name and a new package.

## Clip-Partitioned Experiment Stores

Future long-running experiments may be recorded as multiple shorter clips
instead of one very large video. This is a first-class scaling target, not an
exception to the cluster artifact workflow.

Orange already supports headless rolling clips. The native Orange artifact
shape is one parent recording folder with `recording_clip_index.{json,csv}` and
per-clip directories under `clips/clip_%06d/`. Each clip has clip-local MP4,
metadata CSV, keyframe JSON, and clip manifest files. Palette should preserve
Orange's `clip_id` and session-continuous `recording_frame_id` mapping rather
than treating clips as unrelated recordings.

The design decision is: clip compute should parallelize by independent clip
namespace, while shared experiment-level metadata is finalized separately.
Importer locking protects shared mutable metadata; it should not serialize
independent clip-local compute or clip-local package promotion when the target
paths are disjoint.

Preferred layout shape:

```text
experiment.zarr/
  clips/
    clip_0000/
      cameras/
        2010093/
          source/
          detect_runs/detect_...
          crop_runs/crop_...
    clip_0001/
      cameras/
        2010093/
          source/
          detect_runs/detect_...
          crop_runs/crop_...
  experiment_index/
    clip_table
    run_manifest
```

Each cluster job should own one clip, or a small set of clips, and write a
complete clip-local run group on node-local scratch before packaging it. The
importer can then promote into a clip-local target such as
`clips/clip_0017/cameras/2010093/detect_runs/detect_...` without touching
arrays owned by other clips or cameras.

Avoid a design where many jobs append into one giant run group such as
`detect_runs/detect_full_experiment/frame_indices`. That shape requires global
row allocation, chunk-aligned writes, frame-offset coordination, and careful
metadata synchronization. It may be useful for a generated query/export layer,
but it should not be the primary concurrent write target.

Readers should treat an experiment as a logical concatenation of clip-local
outputs using explicit identifiers:

- `clip_id`
- `clip_local_frame_index`
- `recording_frame_id`
- optional generated `global_frame_index` alias when it is equal to
  `recording_frame_id`
- source video path or clip source id

A short serialized finalize step may build or refresh `experiment_index`,
`latest` pointers, consolidated metadata, and registry projections after all
clip-local imports complete.

### Clip Finalize Stage

For clipped recordings, finalization is a separate job family from model
compute and run-group import. It should not hold a GPU unless the specific
finalizer needs one.

The finalize stage should run after the required clip-local imports for a
workflow succeed. Its responsibilities are shared metadata and logical views:

- verify expected clip coverage from `recording_clip_index` or
  `recording_frame_index.parquet`;
- verify each clip-local run references the expected upstream run names and
  source fingerprints;
- write or refresh `experiment_index/workflow_manifests/<workflow_id>`;
- write run-collection manifests that map `(clip_id, camera_serial)` to
  concrete clip-local run paths;
- update logical latest aliases only after coverage and validation pass;
- refresh consolidated metadata when policy requires it;
- update registry projections from the finalized canonical Zarr state.

The finalize stage is also where temporal boundary policy belongs. Image-local
outputs can be collected directly, but temporal analyses such as track
kinematics, bout detection, smoothing, hysteresis, and state-machine outputs
must either be computed parent-wide in the finalize phase or record explicit
clip-overlap/state-handoff semantics. Silent per-clip concatenation is not a
valid temporal policy.

One giant global array may be generated as an explicit compatibility/export
product after finalization, but it should not be the active cluster write
target. The durable source of truth remains the parent recording clock plus
validated clip-local run groups.

## Janelia Cluster Assumptions

This design is written for the Janelia compute cluster guidance at
`https://hpc.int.janelia.org/docs/ai-agent-hints` (checked 2026-05-14; page
content reports last update 2026-04-22). If that live guidance changes, it
takes precedence over this cached interpretation.

Important environment assumptions:

- Scheduler is IBM Spectrum LSF, not Slurm. Use `bsub`, `bjobs`, and `bkill`
  terminology in job examples.
- Shared storage is NFS-backed PRFS/NRS, not Lustre or GPFS. This reinforces
  the rule that many small concurrent Zarr metadata/chunk writes should not be
  sprayed directly at shared storage.
- Node-local scratch is `/scratch/$USER/`. Cluster jobs should create a
  job-specific scratch directory such as `/scratch/$USER/$LSB_JOBID`.
- `/tmp` on compute nodes is not the correct scratch target.
- Docker is not available for cluster jobs. If containerization is needed, use
  Apptainer and include `--nv` for GPU jobs.
- LSF sets `CUDA_VISIBLE_DEVICES`; Palette job scripts should read it for
  provenance but should not override it by default.
- Jobs should set a wall-time limit with `-W` and write stdout/stderr with `-o`
  and `-e`.
- Python/ML threading variables should be set explicitly from the allocated
  slot count. Use `$LSB_DJOB_NUMPROC` as the starting point, then reduce values
  when the job has layered parallelism such as process pools, Dask workers, or
  PyTorch dataloader workers.

For Palette detection and analysis jobs, the immediate policy is
recording-level parallelism with conservative per-job threading. A job array is
appropriate only when each array task owns distinct recordings and writes
distinct scratch packages.

For single-pass detection specifically, do not copy full source videos to
scratch by default. Stream the camera video from PRFS/NRS and write only the
new detection output run group to job-local scratch. Copying a large MP4 to
scratch is reserved for workflows that repeatedly reopen the same video, perform
heavy random seeking, or have benchmark evidence that shared-storage streaming
is the bottleneck.

## Non-Goals

- Do not make tarballs the canonical data format.
- Do not replace per-recording Zarr archives.
- Do not make every analysis stage internally Dask-aware.
- Do not allow many cluster jobs to concurrently update `latest`,
  consolidated metadata, or the SQLite registry.
- Do not use this workflow to hide incomplete or failed runs in canonical
  archives.

## Roles

### Cluster Job

The cluster job owns compute and scratch output. It should:

- stage required inputs to node-local scratch when practical;
- run one recording, or a small independent batch of recordings;
- write output run groups into a scratch Zarr or scratch directory;
- capture full provenance;
- validate the output;
- package the run group and manifest;
- write a job report.

The cluster job must not:

- update canonical parent attrs such as `analysis/<family>.attrs["latest"]`;
- update consolidated metadata on the canonical archive;
- write directly into an existing canonical run group;
- update the registry as the source of truth for the completed run.

For detection, the desired cluster job flow is:

```text
PRFS video + canonical analysis metadata
  -> compute on cluster node
  -> /scratch/$USER/$LSB_JOBID/palette_run_group_artifact/run_group/
  -> validate scratch run group
  -> package artifact as tar.zst or tar.gz
  -> transfer package to PRFS
  -> serialized importer promotes into detect_runs/
```

The current direct-write detection runner is a pilot path, not the final
production path for broad cluster arrays.

If the immediate question is "can the cluster decode and run the model?", use a
compute-only smoke before writing any predicted chunks to PRFS/NRS. That smoke
should open the PRFS video, load the selected model, decode a small batch of
frames, run inference, and discard the predictions or write only a small JSON
report. It must not write `detect_runs` chunks to the canonical analysis Zarr.

Palette's concrete command for this is:

```bash
scripts/py -m fisheye.diagnostics.detect_compute_smoke \
  /groups/johnson/johnsonlab/jeremy/palette_smoke/<recording>/cams/<camera>.mp4 \
  --model /groups/johnson/johnsonlab/jeremy/palette_models/<model>/weights/best.pt \
  --decode-backend auto \
  --batch-size 4 \
  --max-batches 1 \
  --output-json /groups/johnson/johnsonlab/jeremy/palette_smoke/logs/detect_compute_smoke.json
```

For LSF, use `scripts/submit_detect_compute_smoke_bsub.sh` rather than a long
quoted `bsub` command. It writes a job script first, sets
`PALETTE_JOB_CACHE=/scratch/$USER/$LSB_JOBID/palette_cache`, and stores the JSON
as `<run_dir>/<run_label>.<LSB_JOBID>.json`.

The smoke honors `detection.resize_dims` from the detection config when no
explicit `--resize` is supplied. That keeps the compute path aligned with the
normal detector and avoids accidentally using full camera-frame resolution.
For LSF jobs, set `PALETTE_JOB_CACHE=/scratch/$USER/$LSB_JOBID/palette_cache`
so Ultralytics and other headless tools do not write into `$HOME`.

If the immediate question is "can we safely produce cluster detection outputs?",
the smoke should use the artifact/import path below. Avoid a full direct-write
detect smoke because it exercises the exact NFS chunk-write pattern this design
is intended to eliminate.

### Importer

The importer owns canonical archive mutation. It should run serially, or under a
lock that guarantees one importer per mutable namespace. For today's
single-recording archives, that usually means one importer per target archive.
For future clip-partitioned experiment stores, that can be one importer per
clip namespace for clip-local run groups, plus a separate experiment-level lock
for shared indexes and parent metadata. The importer should:

- unpack the package into an incoming/staging path on the storage node;
- validate the package again against the current canonical archive;
- verify upstream fingerprints still match;
- atomically promote the run group into its final path when possible;
- update parent attrs such as `latest` only after the run group is complete;
- refresh consolidated metadata when policy requires it;
- update registry projections from the final archive state.

## Package Layout

A run-group artifact should have a single package root:

```text
palette_run_group_artifact/
  artifact_manifest.json
  run_group/
    zarr.json
    ...
  validation/
    strict_json_report.json
    array_presence_report.json
  logs/
    command.log
```

The `run_group/` directory contains the exact Zarr group that will be imported
under the target parent group. For example, if the manifest target is:

```text
analysis/swim_bout_runs/bouts_tk_hyst4_low2_latch_s005_peak_event...
```

then `run_group/` should contain the contents of that `bouts_...` group, not
the entire analysis archive.

Transport format can be `tar.zst`, `tar.gz`, or an unpacked directory. Prefer a
compressed tarball for network transfer because it converts many small Zarr
chunk files into one sequential transfer. The tarball is only a transport
artifact; the canonical store remains Zarr.

## Manifest

Each artifact must include `artifact_manifest.json`. Required fields:

```json
{
  "artifact_schema": "palette_run_group_artifact_v1",
  "created_at": "2026-05-14T00:00:00Z",
  "archive_id": "optional-stable-recording-or-archive-id",
  "target_archive_path": "/path/to/recording_analysis.zarr",
  "target_group_path": "analysis/swim_bout_runs/run_name",
  "run_family": "swim_bout_runs",
  "run_name": "run_name",
  "layout": "compact_tabular_v2",
  "schema_version": "optional-stage-schema-version",
  "source_inputs": [
    {
      "path": "analysis/track_kinematics_runs/offline/tk_...",
      "role": "source_track_kinematics_run",
      "expected_fingerprint": "..."
    }
  ],
  "provenance": {
    "palette_git_commit": "...",
    "command": "scripts/py -m ...",
    "hostname": "...",
    "cluster": {
      "LSB_JOBID": "...",
      "LSB_JOBNAME": "...",
      "LSB_QUEUE": "...",
      "LSB_DJOB_NUMPROC": "...",
      "CUDA_VISIBLE_DEVICES": "..."
    },
    "runtime": {
      "platform": {
        "hostname": "...",
        "system": "Linux",
        "release": "...",
        "machine": "x86_64",
        "lsf": {"job_id": "...", "queue": "..."}
      },
      "gpu": {
        "available": true,
        "backend": "cuda",
        "devices": [{"index": 0, "name": "NVIDIA L4"}]
      },
      "environment": {
        "environment_name": "palette-py311",
        "python_executable": "/path/to/env/bin/python",
        "key_packages": {}
      },
      "env_vars": {"CUDA_VISIBLE_DEVICES": "..."}
    },
    "cuda_device": "...",
    "decoder_backend": "decord_gpu"
  },
  "timing": {
    "wall_seconds": 0.0,
    "decode_read_seconds": null,
    "inference_seconds": null,
    "write_seconds": null
  },
  "checksums": {
    "run_group_tree_hash": "..."
  },
  "validation": {
    "strict_json": "pass",
    "required_arrays": "pass",
    "upstream_fingerprints": "pass"
  }
}
```

Numeric fields that are not applicable should be `null`, not `NaN` or
`Infinity`.

The run group itself should also carry canonical `provenance` attrs. For
cluster-produced detection artifacts this includes the normal local provenance
fields plus scheduler details, richer machine/platform details, and GPU device
metadata so the imported run remains self-describing after the transfer package
is discarded.

## Import Algorithm

The importer should use a two-phase import:

Current dry-run implementation:

```bash
scripts/py -m fisheye.utils.import_run_group_artifact \
  /groups/johnson/johnsonlab/jeremy/palette_smoke/detect_artifacts/<run>/<label>.<JOBID>.tar.gz
```

For rolling-clip artifacts, use the manifest's clip-local intended target:

```bash
scripts/py -m fisheye.utils.import_run_group_artifact \
  /groups/johnson/johnsonlab/jeremy/palette_smoke/detect_artifacts/<run>/<label>.<JOBID>.tar.gz \
  --use-intended-target
```

This command extracts only to a temporary validation directory and prints a
strict JSON plan. It validates the manifest shape, strict JSON, required arrays,
run-group tree hash, source input paths, target archive path, final target path,
and planned `.incoming`/`.failed` paths. It does not create `.incoming`, does
not modify `latest`, and does not mutate the canonical Zarr.

Apply-mode implementation:

```bash
scripts/py -m fisheye.utils.import_run_group_artifact \
  /groups/johnson/johnsonlab/jeremy/palette_smoke/detect_artifacts/<run>/<label>.<JOBID>.tar.gz \
  --apply
```

For rolling-clip artifacts, add `--use-intended-target --apply` so the run lands
under `clips/<clip_id>/cameras/<camera_serial>/detect_runs/<run_name>` instead
of the compatibility top-level `detect_runs/<run_name>` path.

For clustered clip DAGs, prefer deterministic run names instead of timestamped
names. The detection artifact submitter accepts `--detect-run-name`, which is
passed through to `fisheye.utils.run_detection_artifact --run-name` and then to
the scratch `detect_yolo` run. This makes the downstream import, validation,
quality, and refine commands fully known before submission.

Apply mode first runs the same dry-run validations. If they pass, it copies the
packaged run group to `<family_parent>/.incoming/<run_name>/`, revalidates the
incoming copy, promotes it to `<family_parent>/<run_name>/`, and writes an
import receipt sidecar at
`<family_parent>/.imports/<run_name>_import_receipt.json`. The receipt is kept
outside the run group so the imported run-group bytes still match the package
tree hash. If apply-time validation fails after `.incoming` is created, the
incoming directory is moved to `<family_parent>/.failed/<run_name>_<timestamp>/`.
`latest` is updated only when the package `latest_policy` requests it.

Post-import validation:

```bash
scripts/py -m fisheye.utils.validate_imported_run_group \
  <recording_analysis.zarr> \
  --target-group-path detect_runs/<run_name>
```

The validator checks the import receipt, strict JSON, required arrays, source
inputs, source tarball checksum, run provenance, and the imported run-group
fingerprint. For `detect_yolo_sparse_v1`, the fingerprint is interpreted as
the immutable imported model-output core. Existing Palette detect-quality
reports are appended under `detect_runs/<run>/quality_reports/`; after those
reports exist, the validator allows that known mutable child while still
requiring the imported core tree hash to match the original artifact manifest.

Registry-free downstream smoke:

```bash
scripts/py -m fisheye.refinement.detect_quality \
  <recording_analysis.zarr> \
  --run <imported_detect_run> \
  --save

scripts/py -m fisheye.refinement.refine_detect \
  <recording_analysis.zarr> \
  --detect-run <imported_detect_run> \
  --quality-run <detect_quality_run> \
  --config configs/fisheye/default.yaml \
  --per-frame-top-k 1
```

For a clip-local imported detect run, use explicit family paths:

```bash
scripts/py -m fisheye.refinement.detect_quality \
  <recording_analysis.zarr> \
  --detect-family-path clips/<clip_id>/cameras/<camera_serial>/detect_runs \
  --run <imported_detect_run> \
  --quality-run-name <quality_run_name>

scripts/py -m fisheye.refinement.refine_detect \
  <recording_analysis.zarr> \
  --detect-family-path clips/<clip_id>/cameras/<camera_serial>/detect_runs \
  --refined-family-path clips/<clip_id>/cameras/<camera_serial>/refined_detect_runs \
  --detect-run <imported_detect_run> \
  --quality-run <quality_run_name> \
  --run-name <refined_run_name> \
  --per-frame-top-k 1

scripts/py -m fisheye.utils.validate_refined_detect_run \
  <recording_analysis.zarr> \
  --target-group-path clips/<clip_id>/cameras/<camera_serial>/refined_detect_runs/<refined_run_name>
```

Dry-run planning for all clip-camera chains:

```bash
scripts/py -m fisheye.utils.plan_clipped_detect_refine_workflow \
  /groups/johnson/johnsonlab/jeremy/palette_smoke/<recording> \
  --model /groups/johnson/johnsonlab/jeremy/palette_models/detect/<run>/weights/best.pt \
  --workflow-id sleepyfish_detect_refine_smoke_YYYYMMDD \
  --output-json /groups/johnson/johnsonlab/jeremy/palette_smoke/<recording>/derived/cluster_artifacts/detect_refine_plan.json
```

The planner does not submit jobs or mutate Zarrs. It reads
`recording_clip_index.json`, emits one work item per `(recording_id,
camera_serial, clip_id)`, generates deterministic detect/quality/refined run
names, and records the exact commands for detect artifact submission, import,
post-import validation, detect quality, refined detect, and refined-detect
validation.

To prepare the dependent LSF scripts from that plan without submitting:

```bash
scripts/py -m fisheye.utils.submit_clipped_detect_refine_plan_bsub \
  /groups/johnson/johnsonlab/jeremy/palette_smoke/<recording>/derived/cluster_artifacts/detect_refine_plan.json \
  --limit 1
```

The submitter writes a local submission bundle containing one script per CPU
stage, one finalizer script, and a `submission_manifest.json` with the planned
dependency chain. The CPU chain is import, imported-detect validation, detect
quality, refined detect, refined-detect validation, and final collection
validation. It is dry-run by default. Use `--submit --limit 1` for the first
cluster smoke; use `--allow-multiple` only after the one-clip chain has passed
import validation, detect-quality, refined-detect, and finalizer checks.

Before submission, the submitter preflights planned target paths in the
analysis Zarr. Dry-runs report collisions in `target_preflight`; submit mode
refuses to proceed when planned detect run, detect-quality report, refined run,
or finalized collection paths already exist. Prefer a new `--workflow-id` for
retries. `--allow-existing-outputs` exists for deliberate expert recovery, not
normal operation.

After submission, summarize the chain from the generated manifest:

```bash
scripts/py -m fisheye.utils.check_clipped_detect_refine_submission \
  <submission_bundle>/submission_manifest.json
```

This is the operator-facing progress gate. It checks the detect artifact summary
and tarball, every CPU stage status JSON, and the finalizer status JSON. A
running or blocked chain reports `incomplete`; explicit failed stages report
`failed`; a fully completed chain reports `ok`. Use `--require-complete` in
scripts or handoff notes when the next step must not proceed until every stage
and the finalizer are complete. For missing or failed stages, the human output
also reports stdout/stderr file presence and sizes so operators can jump
directly to the useful LSF logs.

The finalizer can also be run manually against a completed plan:

```bash
scripts/py -m fisheye.utils.finalize_clipped_detect_refine_workflow \
  <recording_dir>/derived/cluster_artifacts/detect_refine_plan.json \
  --submission-manifest <submission_manifest.json> \
  --apply
```

It fails closed if any planned clip-camera refined run is missing, if the
refined validator fails, if the refined `instances/frame_counts` length does
not match the planned clip `frame_count`, if `recording_frame_index.parquet`
has missing/non-contiguous planned clip-camera rows, or if required stage status
JSONs are missing/non-ok. Successful apply writes
`experiment_index/finalized_runs/<workflow_id>` and updates the parent
`refined_detect_runs.latest_collection` pointer.

Resolve a finalized collection to a parent-frame mapping:

```bash
scripts/py -m fisheye.utils.resolve_clipped_refined_detect_collection \
  <recording_analysis.zarr> \
  --collection-id <workflow_id> \
  --output-parquet <recording_dir>/derived/cluster_artifacts/<workflow_id>_frame_run_map.parquet
```

The resolver joins `recording_frame_index.parquet` to the finalized
`selected_runs` manifest and emits rows with `recording_frame_id`,
`clip_local_frame_index`, `clip_id`, `camera_serial`, and the selected
detect/refined run paths. Readers should use this resolver instead of scanning
clip directories independently.

Build a runtime ROI cache from a finalized clipped collection:

```bash
scripts/py -m fisheye.utils.build_clipped_collection_flat_roi_cache \
  <recording_analysis.zarr> \
  --collection-id <workflow_id> \
  --output-dir /misc/public/palette_cache/<workflow_id>/roi_cache \
  --progress-jsonl <recording_dir>/derived/cluster_artifacts/<workflow_id>_roi_cache.progress.jsonl \
  --progress-stderr
```

For LSF execution, use the submit wrapper so the cache is built on node-local
scratch and published to PRFS in manifest-last order:

```bash
scripts/submit_clipped_collection_flat_roi_cache_bsub.sh \
  --zarr <recording_analysis.zarr> \
  --collection-id <workflow_id> \
  --public-cache-root /misc/public/palette_cache \
  --run-label <recording>_<workflow_id>_roi_cache \
  --queue gpu_l4 \
  --gpus 1 \
  --walltime 2:00
```

For the first cluster smoke, add `--limit-rows 1024`. The wrapper writes a
submission context, generated job script, LSF stdout/stderr, progress JSONL,
status JSON, and final published manifest/payload/row-index paths.

This builder consumes clip-local refined `instances` rows directly and writes a
flat ROI cache plus a required `.rows.parquet` sidecar. The payload rows are
`pynvvc_luma_v1` `[N,H,W] uint8` crops. The row-index parquet maps each cache row
back to `clip_id`, `camera_serial`, `clip_local_frame_index`,
`recording_frame_id`, `refined_group_path`, and `refined_row_id`, so downstream
pose and segmentation jobs can remain clip/parent-frame aware without merging
all clip runs into one parent crop run first.

Root `width`/`height` attrs remain valid for single-video archives and can be
valid parent-level invariants for clipped recordings when every clip/camera has
the same resolution. The collection cache path should still prefer refined
`instances/bbox_img_xyxy` as the row-level geometry source, using root
dimensions only as an unambiguous compatibility fallback for normalized-only
rows. Root source-path attrs are still valid provenance and migration metadata,
especially for single-video archives and smoke-copy repair.

Measured all-clips smoke, 2026-05-17:

- Recording:
  `/groups/johnson/johnsonlab/jeremy/palette_smoke/sleepyfish_2026_05_05_17_45_30_cam2010093`.
- Workflow id: `sleepyfish_cam2010093_allclips_20260517_01`.
- Scope: 22 clip-camera work units, `clip_000000..clip_000021`,
  camera `2010093`, 1,188,000 frames.
- Decode backend: explicit `pynvvc_luma_rgb`.
- Model:
  `/groups/johnson/johnsonlab/jeremy/palette_models/detect/detect_all_available_detect_training_v003/detect_all_available_detect_training_v003_yolo11n_trt_20260516_retry1/weights/best.pt`.
- Submission checker result: `133/133` stages `ok`, including the finalizer.
- Finalized collection:
  `experiment_index/finalized_runs/sleepyfish_cam2010093_allclips_20260517_01`.
- Resolver result: `mapped_frame_count=1188000`,
  `selected_run_count=22`, `unselected_frame_pair_count=0`.

Performance from the actual cluster artifacts:

| Measurement | Value |
|-------------|-------|
| First detect job start to last detect job finish | `7m37s` |
| First detect job start to finalizer complete | `~8m54s` |
| Submission to finalizer complete | `~9m03s` |
| Sum of per-clip YOLO/detection time | `9308s` (`~2h35m`) |
| Sum of per-clip artifact time | `9563s` (`~2h39m`) |
| Sum of LSF detect job runtimes | `9680s` (`~2h41m`) |
| Mean per-clip artifact time | `435s` |
| Parallel aggregate detect throughput | `~2600 fps` |
| Serial-equivalent one-GPU throughput | `~128 fps` |
| Parallel speedup over summed one-GPU detect time | `~20x` |

Important caveat, added 2026-05-18: this measured collection was produced before
the PyNvVideoCodec surface-materialization fix in `detect_yolo`. The
pre-fix `pynvvc_luma_rgb` batch path could retain decoder-owned reusable CUDA
surfaces until after a full batch was decoded, producing batch-window bbox
plateaus even though `frame_indices` were continuous. The historical
`sleepyfish_cam2010093_allclips_20260517_01` refined detections are useful for
workflow and scheduler timing, but should not be treated as a trusted visual
prediction product.

Regenerate clipped detect/refine outputs with a run that records
`pynvvc_surface_materialization=stream_event_owned_batch_v2` or a later
materialization marker. The event/preallocated-batch implementation is
correctness-safe, but the measured full-clip smoke was not faster than the
previous conservative paths: `clip_000000` processed at `125.6 fps` with
`--batch-size 16`, versus historical fixed-path measurements of `127.8 fps` for
batch `16` and `131.0 fps` for batch `1`. For correctness-sensitive production
clipped `pynvvc_luma_rgb` jobs, keep using fixed/current PyNvVC outputs and
prefer `--batch-size 1` until a true overlap or buffer-pool design proves faster
end-to-end.

Cluster rerun readiness, added 2026-05-18:

- We are ready to rerun clipped detection/refinement on the cluster after pulling
  a commit that contains the `stream_event_owned_batch_v2` materialization path.
- Use a new `--workflow-id`; do not overwrite
  `sleepyfish_cam2010093_allclips_20260517_01`, because that collection remains
  scheduler-valid but visually suspect.
- Use `--decode-backend pynvvc_luma_rgb --batch-size 1` for the first production
  rerun. Current batch 16 is correctness-safe, but the measured full-clip smoke
  did not show an end-to-end speed advantage.
- After submission, require the checker to report every stage `ok`, then inspect
  one imported detect run's `timing_summary.pynvvc_surface_materialization` and
  `parameters.pynvvc_surface_materialization`; both should be
  `stream_event_owned_batch_v2`.
- Treat any PyNvVC batch-size-greater-than-1 run without a fixed materialization
  marker as suspect and regenerate rather than trying to repair boxes in place.

Template for the next all-clips rerun:

```bash
REC=/groups/johnson/johnsonlab/jeremy/palette_smoke/sleepyfish_2026_05_05_17_45_30_cam2010093
MODEL=/groups/johnson/johnsonlab/jeremy/palette_models/detect/detect_all_available_detect_training_v003/detect_all_available_detect_training_v003_yolo11n_trt_20260516_retry1/weights/best.pt
WORKFLOW_ID=sleepyfish_cam2010093_allclips_pynvvc_fixed_YYYYMMDD_01
PLAN="$REC/derived/cluster_artifacts/detect_refine_plan_${WORKFLOW_ID}.json"
RUN_DIR="$REC/derived/cluster_artifacts/detect_refine_submission_${WORKFLOW_ID}"

scripts/py -m fisheye.utils.plan_clipped_detect_refine_workflow \
  "$REC" \
  --model "$MODEL" \
  --workflow-id "$WORKFLOW_ID" \
  --decode-backend pynvvc_luma_rgb \
  --batch-size 1 \
  --output-json "$PLAN"

scripts/py -m fisheye.utils.submit_clipped_detect_refine_plan_bsub \
  "$PLAN" \
  --run-dir "$RUN_DIR" \
  --submit \
  --allow-multiple \
  --output-json "$RUN_DIR/submission_manifest.json"

scripts/py -m fisheye.utils.check_clipped_detect_refine_submission \
  "$RUN_DIR/submission_manifest.json" \
  --require-complete
```

Interpretation: the `~2600 fps` number is aggregate wall-clock throughput
from running 22 independent clip jobs concurrently; it is not per-GPU model
speed. The serial-equivalent numbers are the better estimate for how long this
recording would take if processed as one long video or as clips on a single
GPU. The fan-out result supports clip-camera GPU jobs as the default scaling
unit for long recordings, while the small CPU follow-up stages remain good
candidates for a future fused per-clip postprocess job to reduce scheduler
overhead.

For browser review of these clipped outputs, do not treat the original
`4512x4512` HEVC clip MP4s as the long-term web-serving substrate. They are
large analysis sources, not browser review proxies. The planned web review cache
is documented in `docs/detection_review_web_todo.md`: derived faststart H.264
proxy videos under `derived/review_proxy/video_detect/<proxy_run_id>/`, plus a
manifest that maps proxy clips back to the source clip timeline. The analysis
Zarr remains the canonical detection/curation store; proxy videos are
regenerable display artifacts.

The batch wrappers can also plan against a direct Zarr directory path without
registry discovery:

```bash
scripts/py -m fisheye.utils.detect_quality_batch <recording_analysis.zarr> \
  --detect-run <imported_detect_run> --no-skip-existing --json

scripts/py -m fisheye.utils.refine_detect_batch <recording_analysis.zarr> \
  --detect-run <imported_detect_run> --quality-run <detect_quality_run> \
  --no-skip-existing --config configs/fisheye/default.yaml
```

1. Unpack to an incoming path under the target run-family parent:

   ```text
   <family_parent>/.incoming/<run_name>/
   ```

   For detection this is `detect_runs/.incoming/<run_name>/`. For analysis
   run families it is typically `analysis/<family>/.incoming/<run_name>/`.

2. Validate:

   - strict JSON parse of all `zarr.json` files;
   - required arrays and attrs for the declared layout;
   - no unexpected non-finite JSON attrs;
   - source input paths exist in the target archive;
   - expected upstream fingerprints match current upstream fingerprints;
   - final target path does not already exist unless explicit overwrite is
     requested.

3. Promote:

   ```text
   <family_parent>/.incoming/<run_name>/
   <family_parent>/<run_name>/
   ```

4. Finalize parent metadata:

   - update parent `latest` only if requested;
   - update run-family indexes if present;
   - refresh consolidated metadata according to current Zarr policy;
   - rescan or update registry rows from the final archive.

If validation fails, leave the incoming directory untouched for diagnosis or
move it to:

```text
<family_parent>/.failed/<run_name>_<timestamp>/
```

Do not leave failed packages under the normal run-family namespace.

## Latest Policy

The package should declare whether it requests promotion to latest:

```json
{
  "latest_policy": "do_not_set_latest"
}
```

Allowed values:

- `do_not_set_latest`: import the run but leave current defaults unchanged.
- `set_latest_if_newer`: set latest only if validation passes and the importer
  determines this package supersedes the current latest.
- `set_latest_explicit`: set latest after successful import. This should be
  used sparingly and logged.

The cluster job itself never updates latest.

## Registry Policy

The registry is a query index, not the authoritative writer during cluster
compute. Cluster jobs should emit job reports and package manifests. A serial
reconciliation step should update registry projections after import.

This avoids concurrent SQLite writes and keeps the registry rebuildable from
canonical Zarr state plus artifact manifests.

## Concurrency Rules

- Parallelize by recording first.
- For long experiments represented as clips, parallelize by clip namespace when
  each job writes disjoint clip-local run groups.
- Use one writer per target run group.
- Use one importer per mutable namespace. Use archive-level locking for current
  single-recording archives; clip-level locking is acceptable for future
  stores when the import does not update shared experiment-level metadata.
- Do not run two imports that write the same parent attrs, `latest` pointer,
  consolidated metadata, experiment index, or registry projection concurrently.
- Do not unpack directly into the final target path.
- Do not share one scratch output directory across jobs unless each job owns a
  unique subdirectory.

Internal Dask remains valid only when the stage obeys
`docs/dask_zarr_write_safety.md`: workers must write disjoint physical chunks
or write temporary outputs that are merged by one owner.

## Failure Handling

Cluster jobs should be safely retryable:

- The output run name should include a deterministic parameter signature and a
  run timestamp or job suffix.
- A failed package must not be imported.
- Retrying should create a new package or overwrite only scratch-local outputs.
- Canonical overwrite requires explicit importer approval and should preserve
  an audit trail.

Importer failures should be recoverable:

- If unpack fails, delete or quarantine the incomplete incoming path.
- If validation fails, quarantine and report.
- If promotion succeeds but metadata finalization fails, rerun finalization from
  the canonical run group rather than recomputing the analysis.

## Relationship To Existing Docs

- `docs/dask_zarr_write_safety.md` defines the low-level chunk-write safety
  rule.
- `docs/cluster_batching_guide.md` defines current cluster batching guidance.
- `docs/zarr_storage_lifecycle_policy.md` defines storage-tier and archival
  policy.
- `docs/analytics_query_layer_design.md` defines how exported analytics should
  query registry-selected Zarr archives and Parquet products.

This document fills the gap between cluster compute and canonical Zarr mutation.

## First Implementation Slice

1. Use `fisheye.diagnostics.detect_compute_smoke` to verify video open, model
   load, small-batch decode, and inference without writing `detect_runs`
   chunks.
2. Done for detection: `fisheye.utils.run_detection_artifact` writes YOLO
   predictions into a scratch-only temporary Zarr, extracts the completed
   `detect_runs/<run_name>` group into `palette_run_group_artifact/run_group/`,
   writes manifest/validation reports, and packages the result as `.tar.gz`.
3. Add a dry-run importer that validates a package and prints the planned
   canonical mutations.
4. Add an apply mode that imports one package into one analysis archive.
5. Add strict JSON and source-fingerprint checks.
6. Add registry reconciliation after import, not during cluster compute.

Detection is the most valuable first target for cluster execution because large
video decode and inference are recording-local and expensive. Swim-bout and
bout-kinematics packages are useful second targets because they exercise compact
tabular run layouts with smaller transfer artifacts.

The first detection implementation targets only the run group under
`detect_runs/<run_name>`, not the entire analysis archive. It should
read input video from PRFS/NRS, write that run group on
`/scratch/$USER/$LSB_JOBID`, package the run group, and leave `latest`,
consolidated metadata, and registry projection for the importer.

Current detection artifact command shape:

```bash
scripts/py -m fisheye.utils.run_detection_artifact \
  <camera.mp4> \
  --target-zarr <recording_analysis.zarr> \
  --model <weights/best.pt> \
  --config configs/fisheye/yolo_detect_config.yaml \
  --decode-backend auto \
  --artifact-dir /scratch/$USER/$LSB_JOBID/palette_run_group_artifact \
  --work-dir /scratch/$USER/$LSB_JOBID/work \
  --tarball-output /scratch/$USER/$LSB_JOBID/<recording>.<jobid>.tar.gz
```

LSF wrapper:

```bash
scripts/submit_detect_artifact_bsub.sh \
  --zarr <recording_analysis.zarr> \
  --video <camera.mp4> \
  --model <weights/best.pt> \
  --output-dir /groups/johnson/johnsonlab/jeremy/palette_smoke/detect_artifacts
```

The artifact runner prints a strict JSON summary to stdout. That summary
includes `artifact_timing` for the detection call, run-group copy, validation,
hashing, and tarball creation. The LSF wrapper captures it as
`<label>.<JOBID>.summary.json` and writes a sibling
`<label>.<JOBID>.transfer.json` with the scratch-to-PRFS tarball copy timing.

For rolling-clip smoke runs, `scripts/submit_detect_artifact_bsub.sh` accepts
`--workflow-id`, `--recording-id`, `--clip-id`, `--clip-index`, and
`--camera-serial`. These values are written to `submission_context.json`, echoed
in the job stdout, carried into the artifact runner, and stored in
`artifact_manifest.json`, `artifact_summary.json`, and the transfer log. Clip
packages keep an importer-compatible top-level `target_group_path` and also
record
`intended_target_group_path=clips/<clip_id>/cameras/<camera_serial>/detect_runs/<run>`.
Use `import_run_group_artifact --use-intended-target` for the clip-local
canonical import path.
