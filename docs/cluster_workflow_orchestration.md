# Cluster Workflow Orchestration
<!-- contract-meta
status: design
last_verified: 2026-05-16
purpose: Define how Palette should chain cluster jobs across GPU, CPU, storage-import, validation, and registry-projection stages without wasting scheduler resources or corrupting shared Zarr stores.
-->

## Purpose

This document describes the workflow layer above individual Palette writers.
It answers a different question than the run-group artifact design:

- This document: which jobs should be submitted, what resources they request,
  and how outputs flow between jobs.
- `docs/cluster_run_group_artifact_workflow.md`: how one cluster-produced run
  group is packaged, validated, imported, and promoted into a canonical Zarr.
- `docs/cluster_batching_guide.md`: how to choose batching and concurrency
  units for existing batch runners.
- `docs/geometry_only_crop_workflow_cache_design.md`: how geometry-only crop
  runs and shared workflow ROI caches should be handled across pose and
  segmentation jobs.

The core decision is that Palette cluster execution should be a DAG of small
stage jobs, not one monolithic job that holds a GPU across CPU-only work.

## Design Decision

Use the scheduler for coarse workflow orchestration and reserve internal
parallelism for stages that explicitly support it.

In practical terms:

- GPU jobs run model inference and produce scratch run-group artifacts.
- CPU/storage jobs import artifacts, validate outputs, run detect quality,
  run refinement, and refresh projections.
- Each downstream job consumes explicit run names and report files from its
  upstream job. It should not infer inputs from `latest`.
- Broad parallelism should first be by recording or clip namespace.
- Dask should only be used inside a stage when that stage has a documented
  chunk-safe write contract.

This keeps GPU allocation time close to actual GPU work and makes retry
behavior easier to reason about.

## Local Batch Runners Versus Cluster Submitters

Palette should keep a clean separation between discovery, local execution, and
cluster submission.

The current `*_batch.py` runners are still useful. They are the right tool for
workstation runs, small pilots, and simple serial or mildly parallel execution
where one process can discover archives and process them directly.

Cluster production workflows need a different shape:

- A planner discovers recordings or clips, resolves explicit source runs,
  models, crop runs, cache paths, and output run names, then writes a durable
  plan.
- A local batch runner can consume that plan in-process for workstation or
  smoke testing.
- A cluster submitter consumes the same plan and submits independent LSF jobs,
  usually one recording or clip per stage.
- A stage job command processes exactly one recording or one clip for exactly
  one stage and writes a JSON report.

The natural cluster scheduling unit is:

```text
(recording_id, camera_serial, clip_id, stage)
```

For unclipped recordings, `clip_id` can be omitted or set to a sentinel such as
`full_recording`. For current one-camera recordings, `camera_serial` is still
part of the identity so the same plan shape can support future multi-camera
clip-camera artifacts. For future clipped experiments, each clip-camera
namespace should own independent outputs so detect, crop, pose, and
segmentation jobs can run in parallel without appending into the same physical
Zarr arrays.

Batching still matters, but it moves to the planning and submission layer. A
cluster submitter may group small CPU-only tasks into one job, cap active job
counts with LSF array limits, or submit one job per recording for large video
stages. It should not submit one monolithic GPU job that loops over a broad
batch of heterogeneous stages and then holds the GPU while CPU-only work runs.

Downstream handoff must use explicit JSON reports and run names. Cluster jobs
should not discover their inputs by reading `latest`, because `latest` may
intentionally lag while imports, validation, or registry projection are still
pending.

## Why Not One Large Job?

A single end-to-end job is attractive because it is simple to submit, but it is
the wrong default for production cluster processing.

Problems with a monolithic job:

- It keeps a GPU allocated while CPU-only stages run.
- A late CPU validation or import failure forces the whole GPU stage to be
  rerun unless artifacts were preserved carefully.
- It makes backfilling harder for the scheduler because the requested resource
  envelope must cover the largest stage, not the current stage.
- It mixes compute, storage mutation, validation, and registry projection into
  one failure domain.

The better default is a DAG where each stage requests only the resources it
needs.

## Stage Classes

| Stage class | Resource shape | Writes | Notes |
|-------------|----------------|--------|-------|
| Model inference | GPU, modest CPU, node scratch | scratch run-group artifact | Detect, pose, and mask inference belong here. |
| Artifact import | CPU/storage, no GPU | canonical Zarr `.incoming` then final namespace | Serialized per mutable namespace. |
| Structural validation | CPU, no GPU | JSON report only | Checks strict JSON, required arrays, provenance, fingerprints, row counts. |
| Detect quality | CPU, no GPU | `detect_runs/<run>/quality_reports/<quality_run>` | Single-process single-writer today. |
| Refined detect | CPU, no GPU | `refined_detect_runs/<refined_run>` | Consumes explicit detect and quality runs. |
| Registry projection | CPU, no GPU | SQLite registry or future registry backend | Projection from canonical Zarr state, not source of truth. |
| Consumer smoke | CPU or workstation GUI | reports/logs only | Crimson and Marimo checks should be explicit gates when needed. |

For crop-backed pose and segmentation jobs, geometry-only crop runs should not
cause each downstream job to independently reread the source video. Use a
shared workflow ROI cache as described in
`docs/geometry_only_crop_workflow_cache_design.md`.

For very short CPU stages, it is acceptable to combine adjacent CPU steps in a
single CPU-only job. Do not combine them into the preceding GPU job just for
convenience.

## Canonical Detect-To-Refine DAG

For the current detection pilot, the workflow should look like this:

```text
detect artifact job (GPU)
  -> import detect artifact job (CPU/storage)
  -> validate imported detect job (CPU)
  -> detect quality job (CPU)
  -> refined detect job (CPU)
  -> validate refined detect job (CPU)
  -> registry projection job (CPU, deferred until registry path is migrated)
```

The detect job owns expensive model inference. Everything after import is
either structural validation, quality scoring, deterministic filtering, or
metadata projection.

## Clipped Recording DAG

For rolling-clip recordings, the primary parallelism unit is one
`(recording_id, camera_serial, clip_id)` namespace. Each clip-camera should run
the detect-to-refine chain independently, then one recording-level finalizer
should fan in those results.

Per clip-camera:

```text
detect artifact job (GPU)
  -> import detect artifact job (CPU/storage)
  -> validate imported detect job (CPU)
  -> detect quality job (CPU)
  -> refined detect job (CPU)
  -> validate refined detect job (CPU)
```

Per recording-camera after all clip-camera chains succeed:

```text
recording-level finalizer job (CPU)
  -> verify clip coverage and frame-index continuity
  -> write collection manifest / logical latest aliases
  -> refresh consolidated metadata if policy requires it
  -> project registry/status rows when the registry is cluster-visible
```

The per-clip jobs may run in parallel because they write disjoint namespaces:

```text
clips/<clip_id>/cameras/<serial>/detect_runs/<run>
clips/<clip_id>/cameras/<serial>/refined_detect_runs/<run>
```

They should not write recording-level collection metadata, parent logical
latest aliases, registry rows, or consolidated metadata. Those shared updates
belong to the finalizer.

Clip-local `latest` attrs are acceptable inside an isolated clip-camera run
family because only that clip chain owns that namespace. Recording-level
consumers should not infer a complete recording surface from per-clip `latest`
attrs. They should consume an explicit finalized collection manifest that maps
each clip to the selected detect/refined runs.

The first implementation slice is a dry-run planner:

```bash
scripts/py -m fisheye.utils.plan_clipped_detect_refine_workflow \
  <recording_dir> \
  --model <detect_model.pt> \
  --workflow-id <stable_workflow_id> \
  --output-json <recording_dir>/derived/cluster_artifacts/detect_refine_plan.json
```

It emits deterministic names for the clip-local detect, detect-quality, and
refined-detect runs. Deterministic names are preferred for scheduled DAGs
because dependency jobs can use explicit paths instead of reading `latest` or
parsing a previous job's timestamped output.

The second implementation slice consumes that plan and prepares the LSF
dependency chain:

```bash
scripts/py -m fisheye.utils.submit_clipped_detect_refine_plan_bsub \
  <recording_dir>/derived/cluster_artifacts/detect_refine_plan.json \
  --limit 1
```

This command is dry-run by default. It writes a submission bundle with per-stage
job scripts, expected status JSON paths, exact `bsub` commands, and a finalizer
job that depends on every `validate_refined_detect` stage. The CPU dependency
chain is import, imported-detect validation, detect quality, refined detect,
refined-detect validation, and final collection validation. To submit a single
clip smoke, add `--submit --limit 1`. Submit mode refuses more than one work
unit unless `--allow-multiple` is explicitly passed; broad fan-out should wait
until the one-clip chain and finalizer checks are green.

Submit mode also performs a Zarr target preflight. If any planned clip-local
detect run, detect-quality report, refined run, or finalized collection already
exists, submission fails unless `--allow-existing-outputs` is explicitly set.
Use a new workflow id for ordinary retries; only bypass this guard during
intentional manual recovery.

### Finalizer Responsibilities

The current finalizer is intentionally a small CPU job:
`scripts/py -m fisheye.utils.finalize_clipped_detect_refine_workflow`. It:

- read all per-clip stage reports and require `status=ok`;
- validate that every expected clip-camera from `recording_clip_index` or
  `recording_frame_index.parquet` has a selected refined-detect run;
- check unexpected frame-count changes by comparing each refined
  `instances/frame_counts` vector length to the planned clip `frame_count`;
- audit `recording_frame_index.parquet` for planned clip-camera row counts,
  contiguous `clip_local_frame_index` values, duplicate `recording_frame_id`
  values, and camera-level `recording_frame_id` continuity;
- write a recording-level collection manifest with selected run paths;
- write `experiment_index/finalized_runs/<workflow_id>`;
- update parent-level `refined_detect_runs.latest_collection` only for the
  finalized collection, not while individual clips are still running;
- optionally refresh consolidated metadata and registry projections after the
  Zarr state is complete.

The finalizer should not concatenate large per-clip arrays unless a downstream
consumer explicitly requires a materialized parent-level array. The preferred
first representation is a manifest-backed collection plus the frame-index
mapping that already resolves `(recording_frame_id, clip_id,
clip_local_frame_index)`.

Downstream readers should use
`scripts/py -m fisheye.utils.resolve_clipped_refined_detect_collection` or the
underlying `build_collection_frame_map()` helper to resolve a finalized
collection into `(recording_frame_id, clip_id, clip_local_frame_index,
refined_group_path)`. This keeps collection semantics centralized and avoids
each consumer independently scanning `clips/`.

### Scheduler Pattern For Clips

For a small number of clips, a submitter can create explicit LSF dependency
chains per clip and a finalizer that depends on every refined validation job.
For many clips, prefer a generated workflow manifest plus job arrays or grouped
CPU jobs, with the finalizer depending on the array/group completion and then
validating report files.

Rules:

- Use GPU queues only for model inference or GPU decode/cache stages.
- Use CPU queues such as `short` for import, validation, detect quality, and
  refined detect.
- Cap active clip GPU jobs to avoid flooding shared storage with simultaneous
  large-video reads.
- Do not hold a GPU while detect quality or refined detect runs.
- Do not let multiple jobs write the same clip-local run family/run name.
- Keep downstream inputs explicit: archive path, clip id, camera serial,
  detect run, quality run, refined run, and target group path should all come
  from reports or the workflow manifest, not from broad `latest` discovery.

### Failure And Retry For Clips

Clip workflows should be idempotent by default:

- A failed detect job leaves only scratch artifacts and logs.
- A failed import leaves `.incoming` or moves it to `.failed` in that clip-local
  run family.
- A failed quality/refine job writes a failed report and does not update
  recording-level collection metadata.
- Retrying a clip should either use a new run name or require explicit
  overwrite/cleanup.
- The finalizer should fail closed if any expected clip is missing, failed, or
  points to a stale source run.

This pattern allows partial reruns: only the failed clips need to be retried,
and successful clip-local outputs remain usable as long as the finalizer has
not selected a different collection.

## LSF Dependency Pattern

LSF dependencies should express the stage graph explicitly. A concrete
workflow submitter can generate these commands and capture the returned job
ids in a manifest.

Conceptual shape:

```bash
detect_job=$(bsub -q gpu_l4 -gpu "num=1" -n 8 -W 4:00 \
  -J palette_detect_artifact \
  -oo logs/detect.%J.out -eo logs/detect.%J.err \
  scripts/run_detection_artifact_job.sh)

import_job=$(bsub -q short -n 2 -W 1:00 \
  -w "done(${detect_job})" \
  -J palette_import_detect \
  -oo logs/import.%J.out -eo logs/import.%J.err \
  scripts/run_import_detect_artifact_job.sh)

validate_import_job=$(bsub -q short -n 2 -W 1:00 \
  -w "done(${import_job})" \
  -J palette_validate_detect \
  -oo logs/validate_detect.%J.out -eo logs/validate_detect.%J.err \
  scripts/run_validate_imported_detect_job.sh)

quality_job=$(bsub -q short -n 2 -W 1:00 \
  -w "done(${validate_import_job})" \
  -J palette_detect_quality \
  -oo logs/quality.%J.out -eo logs/quality.%J.err \
  scripts/run_detect_quality_job.sh)

refine_job=$(bsub -q short -n 4 -W 1:00 \
  -w "done(${quality_job})" \
  -J palette_refine_detect \
  -oo logs/refine.%J.out -eo logs/refine.%J.err \
  scripts/run_refine_detect_job.sh)

validate_refined_job=$(bsub -q short -n 2 -W 1:00 \
  -w "done(${refine_job})" \
  -J palette_validate_refined_detect \
  -oo logs/validate_refined.%J.out -eo logs/validate_refined.%J.err \
  scripts/run_validate_refined_detect_job.sh)
```

The exact queue names and wall times are deployment choices. On the Janelia LSF
cluster checked on 2026-05-16, there is no `normal` queue; use `short` for
small CPU jobs and an explicit GPU queue such as `gpu_l4` for L4 jobs. The
important part is the dependency boundary:

- GPU job stops after producing a validated artifact and report.
- CPU jobs continue only after upstream success.
- Failed imports or validation failures do not consume GPU time.

## Janelia Queue Notes

Queue details should be checked from the cluster with `bqueues -l <queue>`.
Observed on 2026-05-16:

| Queue | Observed description | Palette use |
|-------|----------------------|-------------|
| `short` | "For short jobs that run less than 1 hour. This is the default queue." Default run limit `60 min`, maximum `61 min`, `NO_INTERACTIVE`. | Small CPU-only batch jobs: crop geometry, validation, import, detect quality, refinement, metadata repair/projection. |
| `gpu_l4` | "Nodes with Tesla L4 gpus and 8 slots per gpu." Default run limit `120 min`, much longer maximum allowed when requested. | Normal single-node L4 GPU jobs: detection, pose/segmentation inference, GPU decode, flat ROI cache materialization. |
| `gpu_l4_parallel` | "L4 nodes for multinode/MPI jobs." | Avoid for current Palette jobs unless a true multinode/MPI workflow is implemented. |

Rules of thumb:

- Use `short` for CPU jobs under an hour.
- Use `gpu_l4` with `-gpu "num=1"` for the current Palette GPU workloads.
- Do not use `gpu_l4_parallel` for ordinary one-recording jobs.
- If a CPU job needs more than one hour, inspect `bqueues -l local` or
  `bqueues -l cpu_parallel` and choose an appropriate CPU queue explicitly.
- If a GPU job needs more than two hours, keep using `gpu_l4` but request an
  explicit `-W` value within the queue maximum.

When a job array is used, each array task must own distinct recordings or clip
namespaces. Do not use one array to make many tasks append into the same Zarr
run group.

## Handoff Contract

Each job should write a small JSON report. The report is the machine-readable
handoff to the next stage.

Minimum fields:

```json
{
  "status": "ok",
  "stage": "detect_artifact",
  "job_id": "149843128",
  "archive_path": "/groups/.../recording_analysis.zarr",
  "run_family": "detect_runs",
  "run_name": "detect_2026-05-15_19-12-22",
  "target_group_path": "detect_runs/detect_2026-05-15_19-12-22",
  "artifact_path": "/groups/.../detect_2026-05-15_19-12-22.tar.gz",
  "stdout_path": "/groups/.../logs/149843128.out",
  "stderr_path": "/groups/.../logs/149843128.err"
}
```

Downstream jobs should consume these fields directly. They should not select
the newest run by reading `latest`, because `latest` may be intentionally
unchanged or may be updated by a later import/finalize step.

For detect quality and refined detect, the handoff should become:

```json
{
  "status": "ok",
  "stage": "refine_detect",
  "archive_path": "/groups/.../recording_analysis.zarr",
  "detect_run": "detect_2026-05-15_19-12-22",
  "quality_run": "detect_quality_2026-05-15_21-48-08",
  "refined_detect_run": "refined_detect_2026-05-15_21-48-43"
}
```

This explicit naming is the guardrail that lets cluster workflows run without
depending on registry state during the first migration slice.

## Resource Policy

### GPU Stages

GPU jobs should:

- request exactly the number of GPUs needed by the stage;
- set `PALETTE_JOB_CACHE=/scratch/$USER/$LSB_JOBID/palette_cache`;
- write stage outputs to `/scratch/$USER/$LSB_JOBID` first;
- stream source videos from PRFS unless a benchmark shows scratch staging is
  faster for that workload;
- emit timing for video open, decode, preprocess, inference, postprocess,
  artifact write, and artifact packaging;
- stop before canonical Zarr import.

GPU jobs should not:

- run detect quality or refinement after inference just because those steps
  are next in the biological workflow;
- update the registry;
- update `latest`;
- perform broad direct writes to canonical Zarr stores.

### CPU And Storage Stages

CPU jobs should:

- request no GPU;
- use modest CPU slots unless profiling shows the stage is CPU-bound;
- run the importer, validator, detect quality, refined detect, and registry
  projection as separate or combined CPU-only stages;
- preserve upstream JSON reports and write a downstream JSON report.

For import jobs, use the run-group artifact workflow:

```text
artifact tarball
  -> <family_parent>/.incoming/<run_name>
  -> validation
  -> <family_parent>/<run_name>
  -> import receipt
```

For current single-recording archives, import should be serialized at the
archive or run-family mutable namespace. For future clip-partitioned stores,
independent clip-local imports may run in parallel if they do not update shared
experiment metadata.

## Dask And Scheduler Boundaries

LSF and Dask solve different problems.

Use LSF for:

- selecting machines and queues;
- allocating GPUs and CPU slots;
- retrying failed stages;
- chaining stage jobs with dependencies;
- limiting concurrent recording or clip jobs.

Use Dask only inside a stage when:

- the stage has a real internal partitioning strategy;
- each worker writes disjoint physical Zarr chunks, or workers write temporary
  outputs that are merged by one owner;
- provenance records requested and effective chunking.

Do not add Dask to `detect_quality` or `refine_detect` just because they are
running on the cluster. They are currently single-process single-writer stages;
scale them first by recording or clip namespace.

See `docs/dask_zarr_write_safety.md` before adding internal Dask writes.

## Registry Policy

During the initial cluster migration, workflows may run registry-free after
input selection. The canonical handoff is the JSON report plus the final Zarr
state.

The registry should become involved at two boundaries:

- input discovery, when the registry is available on the same storage fabric
  and paths are valid from cluster nodes;
- projection after successful import and validation, when registry rows can be
  refreshed from canonical Zarr state.

Cluster workers should not treat registry rows as the authoritative completion
record while they are still writing artifacts. The authoritative state is:

1. successful stage report;
2. successful import receipt for imported run groups;
3. successful post-import or post-stage validation;
4. registry projection rebuilt from that state.

## Failure And Retry Policy

Every stage should be retryable without manual cleanup of normal namespaces.

Recommended behavior:

- Failed GPU inference leaves scratch artifacts and logs only.
- Failed import leaves `.incoming` or moves it to `.failed`.
- Failed validation writes a failed JSON report and does not update `latest`.
- Retrying with the same final run name should require explicit overwrite or
  cleanup.
- Retrying with a new run name should be the default for production data.

Downstream jobs should depend on upstream success using `done(<jobid>)`. A
workflow submitter should also record job ids and report paths so failed stages
can be inspected without searching LSF output manually.

## Logging And Provenance

A workflow run directory should contain:

```text
workflow_manifest.json
jobs/
  detect_artifact.<jobid>.json
  import_detect.<jobid>.json
  validate_detect.<jobid>.json
  detect_quality.<jobid>.json
  refine_detect.<jobid>.json
  validate_refined_detect.<jobid>.json
logs/
  detect_artifact.<jobid>.out
  detect_artifact.<jobid>.err
  import_detect.<jobid>.out
  import_detect.<jobid>.err
  validate_refined_detect.<jobid>.out
  validate_refined_detect.<jobid>.err
```

The workflow manifest should include:

- requested archive path;
- source video path;
- model path or registry model identity;
- planned run names;
- submitted job ids;
- dependency edges;
- queue, wall time, CPU slots, GPU request;
- artifact paths;
- validation report paths.

Stage provenance written into Zarr attrs should still include command, git
commit, host, LSF context, GPU context where relevant, and timing. Workflow
manifests complement stage provenance; they do not replace it.

## First Implementation Slice

The next implementation slice should avoid solving the whole pipeline at once.

Recommended single-recording scope:

1. Add CPU-only LSF submitter for detect quality.
2. Add CPU-only LSF submitter for refined detect.
3. Make both submitters accept explicit archive path, detect run, and quality
   run names.
4. Make both submitters write run directories with stdout, stderr, and JSON
   reports.
5. Add a small workflow submitter or documented operator pattern that chains:
   imported detect validation -> detect quality -> refined detect -> refined
   validation.

Recommended clipped-recording scope:

1. Add a clip inventory/planner that emits one work item per
   `(recording_id, camera_serial, clip_id)`.
2. Submit clip-local detect artifact jobs with explicit `--workflow-id`,
   `--recording-id`, `--clip-id`, `--clip-index`, and `--camera-serial`.
3. Submit dependent CPU jobs per clip for import, validation, detect quality,
   refined detect, and refined validation.
4. Write one JSON report per stage containing clip identity, run names, target
   group paths, queue, job id, timing, and validation status.
5. Add a recording-level finalizer that reads those reports and writes the
   finalized collection manifest.
6. Run the one-clip chain through the finalizer before enabling
   `--allow-multiple`.
7. Keep registry refresh out of the clip fan-out path; project registry rows
   only from the finalizer once the collection is complete.

Keep registry refresh out of this slice unless the cluster-visible registry
path is already decided. The workflow can remain explicit-path and explicit-run
based until the registry is migrated to the same storage environment.

## Definition Of Done

This orchestration layer is ready for a small production pilot when:

- GPU inference jobs stop after producing validated artifacts.
- CPU jobs handle import, validation, quality, and refinement without holding a
  GPU.
- Every stage writes a JSON report with explicit run-name outputs.
- LSF dependencies can chain at least detect import -> quality -> refinement.
- A failed stage can be retried without corrupting normal Zarr namespaces.
- Registry updates are either explicitly deferred or run as a final projection
  job from validated Zarr state.
