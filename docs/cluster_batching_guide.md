# Cluster Batching Guide

This guide covers running Palette pipeline stages as batch jobs on an HPC
cluster with LSF. The current primary stages — **detect**, **crop**,
**keypoints**, and **subject masks** — support both filesystem discovery and registry-backed SQL
pre-filtering.

Related contract:
- `docs/detect_batch_analysis_zarr_parallel_agents_contract.md`
- `docs/cluster_workflow_orchestration.md`
- `docs/cluster_run_group_artifact_workflow.md`
- `docs/cluster_pipeline_migration_checklist.md`
- `docs/clipped_training_zarr_implementation_checklist.md`
- `docs/environment_setup.md`

Janelia-specific cluster policy source:
- `https://hpc.int.janelia.org/docs/ai-agent-hints`

For Palette environment creation, use the portable `environment.yml` workflow
documented in `docs/environment_setup.md`. Do not use exact workstation export
snapshots as the default cluster install input.

---

## Why batch jobs?

Zarr writes generate many small files and metadata updates. Submitting hundreds
of tiny jobs causes heavy metadata churn and poor performance on networked
filesystems. HPC admins often recommend **fewer, longer jobs** with **sustained
writes** instead of many short tasks.

## Recommended strategy

- **Batch multiple recordings per job** (e.g., 10–30 recordings per job).
- **Limit concurrent jobs** (1–2 at a time per user / per node).
- **Keep per‑job CPU modest** unless the workflow is CPU‑bound.
- Prefer **threads** for IO‑heavy steps (background/detect on sampled imports).
- Avoid parallelizing more than the filesystem can sustain.

For future long experiments split into shorter clips, treat the clip as the
first parallelism unit when the storage layout gives each clip an independent
namespace. Multiple jobs may process and import disjoint clip-local run groups
in parallel; only experiment-level metadata updates such as clip indexes,
`latest` pointers, consolidated metadata, and registry projections need to be
serialized. See `docs/cluster_run_group_artifact_workflow.md` and
`docs/orange_rolling_clip_recording_contract.md`.

## Local Batch Versus Cluster Workflow Scope

This guide documents the current batch runners. They remain appropriate for
local workstation processing, small pilots, conservative archive-level runs,
and smoke testing.

Path scope: current production examples should use PRFS-visible paths:
`/groups/johnson/johnsonlab/jeremy/recordings` and the canonical registry
`/groups/johnson/johnsonlab/jeremy/registries/palette_registry.sqlite`.
Workstation-local paths such as `/nvme1/recordings` and
`/nvme1/palette_registry.sqlite` are legacy/local-cache paths and should not be
used for new cluster submissions.

For production cluster workflows, treat batching as a planning and submission
concern rather than as one long-lived Python process that owns many stages. The
preferred structure is:

- Use a planner to create explicit per-recording or per-clip work items.
- Use local batch runners to execute those work items directly when running on
  a workstation.
- Use a cluster submitter to turn those work items into LSF jobs with stage
  dependencies, resource-specific queues, logs, and JSON reports.

For video-heavy GPU stages, the default cluster unit should be one recording or
one clip per stage job. For short CPU-only stages, grouping several work items
into one CPU job can be reasonable when filesystem pressure is controlled.

At Janelia, checked 2026-05-16, `short` is the default queue for CPU jobs under
one hour, and `gpu_l4` is the normal single-node L4 GPU queue. `gpu_l4_parallel`
is for multinode/MPI jobs and should not be used for ordinary Palette
single-recording stages. See `docs/cluster_workflow_orchestration.md` for the
queue table.

Do not treat `--jobs`, `--batch-size`, or a loop over many archives as the
final cluster abstraction. Those controls are useful locally, but the cluster
workflow layer should schedule explicit stage jobs and pass explicit run names
instead of relying on `latest`.

For clipped recordings, the batch abstraction should produce explicit
`(recording_id, camera_serial, clip_id, stage)` work items. A separate finalize
work item should depend on the relevant clip-local imports and should be the
only job that updates shared experiment-level indexes, logical latest aliases,
consolidated metadata, or registry projections.

Use `scripts/py -m fisheye.utils.plan_clipped_detect_refine_workflow` as the
first conservative step for a clipped recording. It is dry-run only: it reads
`recording_clip_index.json`, creates deterministic run names, and prints the
exact per-clip commands for detection artifact submission, import, validation,
detect quality, and refined detect. Actual LSF submission should consume this
plan rather than rediscovering clip rows or relying on mutable `latest` attrs.

2026-05-17 all-clips smoke result: the
`sleepyfish_2026_05_05_17_45_30_cam2010093` PRFS recording was submitted as 22
clip-camera GPU detect jobs plus dependent CPU postprocess/finalizer jobs.
The workflow completed with `133/133` stages `ok`, finalized 22 selected
refined-detect runs, and resolved all 1,188,000 frame mappings. The detect
fan-out finished in `7m37s` wall time from first GPU job start to last GPU job
finish. Summed one-GPU detect/artifact time was `~2h39m`, implying roughly
`20x` wall-clock speedup when enough L4 slots were available.

This supports clip-camera fan-out for GPU detection on long recordings. It
does not prove that every small CPU stage should stay as a separate LSF job:
in this run the CPU stages were generally seconds long. The conservative
separate-stage chain is useful while validating contracts and logs; a future
`--fuse-cpu-postprocess` mode could combine per-clip import, validation,
detect-quality, refine, and refined validation into one CPU job while keeping
the GPU detect jobs and recording-level finalizer separate.

## Video Decode Storage Policy

For single-pass detection, do not copy full source videos to node-local scratch
by default. A 2026-05-14 L4-node smoke benchmark on a `172 GB`, `4512x4512`
sickyfish MP4 found sustained Decord-GPU decode throughput was effectively the
same from PRFS and from a local `/tmp` copy:

| Source path | Decord GPU single-frame decode |
|-------------|--------------------------------|
| PRFS `/groups/...` | `100.2 fps` |
| local `/tmp/...` copy | `100.7 fps` |

The full copy to `/tmp` took `3m15s`, so copying the source video was net
negative for this single-pass workload. For Decord-based jobs, the current
default should be to stream from PRFS while keeping one `VideoReader` open per
video/job.

The `/tmp` path in that benchmark was only a workstation-style comparison
point. On Janelia compute nodes, use `/scratch/$USER/$LSB_JOBID` for
node-local scratch outputs. Do not design cluster jobs around `/tmp`.

Use node-local scratch for video payloads only when the workflow repeatedly
reopens the same video, performs heavy random seeking, or a benchmark shows
shared-storage throughput is limiting that stage. Scratch remains the preferred
place for temporary outputs and run-group artifacts; this policy is only about
pre-copying large input videos.

For very large MP4s, Decord can still have a large one-time open cost because
it indexes keyframes before returning a `VideoReader`. For sequential
start-at-frame-0 compute smokes on grayscale detection models, the
`pynvvc_nv12_rgb` backend avoids that startup cost by streaming through
PyNvVideoCodec/NVDEC and doing NV12-to-RGB tensor preprocessing on CUDA. The
production `auto` backend now prefers `pynvvc_nv12_rgb` when CUDA,
PyNvVideoCodec, and resize dims are available, and falls back to Decord/OpenCV
otherwise. Request `pynvvc_nv12_rgb` explicitly only when you want to force the
candidate path during controlled validation runs.
`pynvvc_luma_rgb` remains available as an explicit faster diagnostic variant,
but it should not be the default correctness path unless parity is accepted for
that recording family.

When testing PyNvVideoCodec on the cluster, `--pipeline-mode producer` can be
used to overlap sequential decode with YOLO inference. Current smoke results
show the simple sequential PyNvVideoCodec path is faster than producer mode, so
treat producer mode as a diagnostic only. The honest comparison metric is
end-to-end wall-clock FPS, not per-stage timings, because per-batch global CUDA
synchronization is disabled to allow overlap.

Persisted production `detect_yolo` runs include a `timing_summary` attr and
flat timing attrs for read/decode, preprocess/resize, predict, postprocess,
array assembly, and Zarr write. Use those persisted timings to confirm that a
cluster production run has the same bottleneck profile as the compute smoke.

For the benchmark protocol and current measurements, see
`docs/detect_decode_backend_benchmark_todo.md`.

## Analysis-Stage Scaling Policy

For downstream analysis stages such as track kinematics, swim-bout detection,
bout kinematics, eye angles, and stimulus response metrics, prefer
**recording-level parallelism** as the first scaling layer. For clipped
long-running experiments, the equivalent unit is a clip-local namespace. On the
cluster this means submitting one recording, one clip, or a small independent
batch per job, rather than assuming every analysis stage is internally
Dask-aware.

This is intentional:

- A single analysis Zarr should normally have one writer per target run group.
- Independent recordings can safely run in parallel because they write to
  independent Zarr stores.
- Internal Dask is appropriate only when the stage has a well-defined
  chunk-boundary contract and workers write disjoint physical Zarr chunks.
- Avoid multiplying parallelism layers blindly. If a job processes several
  recordings concurrently, do not also launch a high-worker internal Dask stage
  for each recording unless the filesystem budget is explicit.

Current status:

- `eye_angle_analysis` has a Dask worker-chunk backend and records Dask
  scheduler/worker provenance.
- `track_kinematics` is currently single-process within one recording. It has
  cross-frame state such as hysteresis, smoothing, and derivatives, so any
  future Dask implementation must define chunk-boundary state handoff.
- `detect_bouts_multi_level` is currently single-process within one recording.
  Peak-event detection, threshold regions, and gap merging can cross chunk
  edges, so any future Dask implementation needs a reconciliation pass.
- `bout_kinematics` is the best first candidate for future internal Dask:
  per-bout rows are mostly independent and can be partitioned more cleanly,
  with a single-writer or chunk-aligned write phase.
- `stimulus_response` metrics should usually remain recording-level or
  step/window-level first. Add internal Dask only after the metric writer has a
  clear disjoint-slice contract.
- `detect_quality` and `refine_detect` are single-process single-writer stages
  today. On the cluster, run them by recording or clip namespace first, then run
  `validate_refined_detect_run` before any fan-in finalizer consumes the output.
  Do not wrap their internal writes in Dask unless the writer is redesigned
  around disjoint physical Zarr chunks or scratch artifacts plus serialized
  import.

When running movement/bout analysis locally, `scripts/run_movement_bout_batch_pipeline`
supports conservative archive-level scheduling. The default remains serial
execution. Use `--jobs N` only for independent archives and keep `N` small on
networked filesystems. Do not run two jobs that write the same run names into
the same analysis Zarr.

Before adding internal Dask to any analysis stage, follow
`docs/dask_zarr_write_safety.md`.

## Heuristics

- If a task is **<5–10 minutes**, batch it.
- If a task is **>30 minutes**, batch fewer recordings or submit one per job.
- Use **max active jobs** to cap concurrency (LSF array `%` syntax).

---

## Pipeline stages and prerequisites

The primary batch pipelines form a DAG. Each stage requires its predecessors to
have `recording_step_status = 'ok'` before it will process a recording:

```
detect  ->  crop  ->  keypoints  ->  subject_masks  ->  refined_subject_masks
```

| Stage      | Prerequisite steps          | Step name      | Skip-existing flag |
|------------|----------------------------|----------------|-------------------|
| detect     | *(none)*                   | `detect`       | `--overwrite`     |
| crop       | `detect`                   | `crop`         | `--force-new`     |
| keypoints  | `detect`, `crop`           | `keypoints`    | `--overwrite`     |
| subject masks | `crop`, `keypoints`     | `subject_masks`, `refined_subject_masks` | `--force-inference`, `--force-finalization` |

When using `--source registry`, the batch runner queries the registry's
`recording_step_status` table with SQL-level pre-filtering:

- **`require_steps_ok`** — INNER JOIN ensures only recordings where all
  prerequisite steps have status `'ok'` are returned.
- **`exclude_step_ok`** — LEFT JOIN excludes recordings where this stage
  already has status `'ok'` (skip-existing behavior). Disabled when the
  overwrite/force-new flag is set.

---

## Discovery modes

The primary batch runners support two discovery modes via `--source`:

### Filesystem mode (default)

```bash
scripts/py -m fisheye.utils.run_detections_batch /groups/johnson/johnsonlab/jeremy/recordings --recursive --dry-run --json
```

Recursively finds `*_analysis.zarr` targets under the given root directory.
No registry interaction at discovery time.

For copied recording layouts, the detection planner prefers the local
`<recording>/cams/*.mp4` beside the analysis Zarr before falling back to any
`source_video_path` attrs stored inside the Zarr. This allows PRFS smoke copies
to run even when their embedded source attrs still point at workstation paths.

Crop geometry and ROI-cache materialization are stricter today: `crop_batch`
resolves the archive's video source metadata and requires that path to exist
from the compute node, even for `crop_storage_mode=geometry_only`. Geometry-only
crop creation skips ROI pixel extraction but still records valid source-video
provenance for later live reads and cache building. For ad-hoc PRFS smoke
copies, repair copied archive attrs such as `source_video_path`, `source_path`,
and `raw_video/source_path` to the cluster-visible `cams/*.mp4` path before
submitting crop/cache jobs.

### Registry mode

```bash
scripts/py -m fisheye.utils.run_detections_batch /groups/johnson/johnsonlab/jeremy/recordings \
  --source registry \
  --registry /groups/johnson/johnsonlab/jeremy/registries/palette_registry.sqlite \
  --dry-run --json
```

Queries the registry database for analysis zarrs matching the scope, applies
SQL-level prerequisite and skip-existing filters, then builds plans only for
the resulting paths. This is faster than filesystem discovery on large trees
and ensures only "ready" recordings are processed.

**Registry filters** (available on the primary runners):

| Flag               | Description                                 |
|--------------------|---------------------------------------------|
| `--registry PATH`  | Path to the registry SQLite file            |
| `--rig-id ID`      | Filter by rig identifier                    |
| `--arena-id ID`    | Filter by arena identifier                  |
| `--camera-id ID` / `--camera-id-filter ID` | Filter by camera identifier (see note) |
| `--path-contains STR` | Substring match on zarr_path             |

> **Note:** Detection and crop use `--camera-id`. Keypoints and subject masks use
> `--camera-id-filter` to avoid ambiguity with the per-recording camera_id
> argument those runners also accept.

### Emit-paths mode

All runners support `--emit-paths` which prints discovered zarr paths to
stdout and exits immediately. The LSF submit scripts use this to build
batch manifests:

```bash
scripts/py -m fisheye.utils.run_detections_batch /groups/johnson/johnsonlab/jeremy/recordings \
  --source registry --emit-paths \
  --registry /groups/johnson/johnsonlab/jeremy/registries/palette_registry.sqlite
```

---

## 1. Detection batch

### Python runner

```bash
# Dry run — preview plans
scripts/py -m fisheye.utils.run_detections_batch /groups/johnson/johnsonlab/jeremy/recordings \
  --recursive --dry-run --json

# Apply — run detections with registry-backed model resolution
scripts/py -m fisheye.utils.run_detections_batch /groups/johnson/johnsonlab/jeremy/recordings \
  --recursive --apply \
  --registry /groups/johnson/johnsonlab/jeremy/registries/palette_registry.sqlite

# Apply — run detections with an explicit model path, bypassing model registry
scripts/py -m fisheye.utils.run_detections_batch /groups/johnson/johnsonlab/jeremy/palette_smoke \
  --recursive --apply \
  --model /groups/johnson/johnsonlab/jeremy/palette_models/detect/best.pt

# Registry mode — only process recordings not yet detected
scripts/py -m fisheye.utils.run_detections_batch /groups/johnson/johnsonlab/jeremy/recordings \
  --source registry \
  --registry /groups/johnson/johnsonlab/jeremy/registries/palette_registry.sqlite \
  --apply
```

It plans directly against `*_analysis.zarr` targets and skips archives that
already have `detect_runs/latest` unless `--overwrite` is set.

### LSF submit script

```bash
./scripts/submit_detect_batches_bsub.sh \
  --root /groups/johnson/johnsonlab/jeremy/recordings \
  --source registry \
  --batch-size 15 \
  --max-active 2 \
  --queue short \
  --ncores 4 \
  --mem-gb 16 \
  --registry /groups/johnson/johnsonlab/jeremy/registries/palette_registry.sqlite \
  --require-tuning \
  --dry-run
```

**Key options:**

| Flag               | Default | Description                                      |
|--------------------|---------|--------------------------------------------------|
| `--root`           | `/groups/johnson/johnsonlab/jeremy/recordings` | Root recordings directory             |
| `--source`         | `filesystem` | Discovery source (`filesystem` or `registry`) |
| `--batch-size`     | `10`    | Analysis zarrs per batch job                     |
| `--max-active`     | `2`     | Max concurrent jobs in array                     |
| `--queue`          | *(default)* | LSF queue name                               |
| `--ncores`         | `4`     | Cores per job                                    |
| `--mem-gb`         | `16`    | Memory per job in GB                             |
| `--registry`       | `/groups/johnson/johnsonlab/jeremy/registries/palette_registry.sqlite` | Registry path       |
| `--config`         | `configs/fisheye/default.yaml` | Detection config     |
| `--model`          | *(none)* | Explicit detect model path; bypasses registry model resolution |
| `--set-id`         | *(none)* | Detect model set filter                         |
| `--require-tuning` | off     | Skip zarrs without detection_tuning              |
| `--overwrite`      | off     | Rerun even if detect_runs/latest exists          |
| `--dry-run`        | off     | Print manifests + commands; do not submit        |

**Execution model:** Each batch job calls `run_detections_batch --apply` with
a batch of zarr paths. By default, model resolution happens inside the batch
runner through the registry. Passing `--model PATH` uses that explicit model
for every batch target and does not require registry-backed model resolution;
registry discovery still requires `--source registry` and a readable registry.

**Current pilot write policy:** `run_detections_batch` currently writes detect
run groups directly into the target analysis Zarr. This is acceptable only for
low-concurrency smoke runs where one job owns one target archive at a time and
the output is validated immediately.

Do not use the direct-write path for the storage-behavior smoke. If the goal is
to verify cluster throughput without NFS chunk-write pressure, run a compute-only
smoke first: open the PRFS video, load the model, decode a small frame batch,
and execute inference without writing predictions to the canonical analysis
Zarr.

Compute-only detection smoke:

```bash
scripts/py -m fisheye.diagnostics.detect_compute_smoke \
  /groups/johnson/johnsonlab/jeremy/palette_smoke/<recording>/cams/<camera>.mp4 \
  --model /groups/johnson/johnsonlab/jeremy/palette_models/<model>/weights/best.pt \
  --decode-backend auto \
  --batch-size 4 \
  --max-batches 1 \
  --output-json /groups/johnson/johnsonlab/jeremy/palette_smoke/logs/detect_compute_smoke.json
```

For LSF submission, prefer the wrapper instead of embedding the full command in
one quoted `bsub` string:

```bash
scripts/submit_detect_compute_smoke_bsub.sh \
  --video /groups/johnson/johnsonlab/jeremy/palette_smoke/<recording>/cams/<camera>.mp4 \
  --model /groups/johnson/johnsonlab/jeremy/palette_models/<model>/weights/best.pt \
  --config configs/fisheye/yolo_detect_config.yaml \
  --log-dir /groups/johnson/johnsonlab/jeremy/palette_smoke/logs \
  --batch-size 16 \
  --max-batches 100 \
  --run-label <camera>_aligned
```

The wrapper writes a per-run job script and uses an output path of the form
`<run_dir>/<run_label>.<LSB_JOBID>.json`. This avoids shell line-continuation
failures where `--output-json` receives only the log directory and the intended
JSON filename is executed as a separate command.

The smoke writes only the JSON report. It must report
`canonical_outputs_written=false`; if it writes `detect_runs` chunks, it is no
longer the compute-only smoke. By default, the smoke honors
`detection.resize_dims` from the detection config, so it should not accidentally
run YOLO over full-resolution camera frames unless that is explicitly requested.
For headless jobs, set `PALETTE_JOB_CACHE=/scratch/$USER/$LSB_JOBID/palette_cache`
before running Palette commands; the smoke uses that location for Ultralytics
config if `YOLO_CONFIG_DIR` is not already set.

Production detection accepts
`--decode-backend auto|pynvvc_nv12_rgb|pynvvc_luma_rgb|decord_gpu|decord_cpu|opencv`.
Use `auto` for normal cluster jobs; it prefers `pynvvc_nv12_rgb` when
PyNvVideoCodec, CUDA, and resize dims are available, then falls back to
Decord/OpenCV. Use explicit `pynvvc_nv12_rgb` when you want to force the
sequential NVDEC/NV12-RGB path for a controlled smoke or batch.
Compare a PyNv candidate against a Decord/OpenCV reference on explicit fixed
frames before treating it as accepted for a recording family:

```bash
scripts/submit_detect_decode_backend_parity_bsub.sh \
  --video /groups/johnson/johnsonlab/jeremy/palette_smoke/<recording>/cams/<camera>.mp4 \
  --model /groups/johnson/johnsonlab/jeremy/palette_models/<model>/weights/best.pt \
  --config configs/fisheye/yolo_detect_config.yaml \
  --backend-a decord_gpu \
  --backend-b pynvvc_nv12_rgb \
  --frames 0 100 500 1000 1500 \
  --batch-size 16 \
  --max-bbox-diff 0.01 \
  --max-score-diff 0.05 \
  --log-dir /groups/johnson/johnsonlab/jeremy/palette_smoke/logs \
  --run-label <camera>_decode_parity
```

For `pynvvc_*` candidates, the submitter runs
`scripts/validate_cluster_palette_env.sh --require-pynvvc` inside the LSF job
before prediction comparison. This catches missing PyNvVideoCodec or NVIDIA
video-driver libraries before the parity script starts.

Validate the parity report with:

```bash
scripts/py scripts/check_detect_decode_backend_parity.py \
  /groups/johnson/johnsonlab/jeremy/palette_smoke/logs/<run-dir>/<camera>_decode_parity.<JOBID>.json
```

When comparing compute-smoke or Crimson/native decode-smoke runs, use the
multi-report formatter rather than reading individual JSON files by hand:

```bash
scripts/py scripts/report_detect_compute_smokes.py \
  /groups/johnson/johnsonlab/jeremy/palette_smoke/logs/<decord-run>/*.json \
  /groups/johnson/johnsonlab/jeremy/palette_smoke/logs/<pynvvc-run>/*.json \
  /groups/johnson/johnsonlab/jeremy/palette_smoke/logs/<crimson-decode-run>/*.json
```

After writing two persisted detect runs, compare the stored artifacts:

```bash
scripts/py -m fisheye.diagnostics.compare_detection_runs \
  /groups/johnson/johnsonlab/jeremy/palette_smoke/<recording>/zarr/<recording>_analysis.zarr \
  --run-a detect_decord_reference \
  --run-b detect_pynvvc_candidate \
  --frames 0 100 500 1000 1500 \
  --fail-on-count-mismatch
```

**Production write policy:** detection jobs should stream the input video from
PRFS, write the new `detect_runs/detect_...` run group under a
job-specific `/scratch/$USER/$LSB_JOBID` directory, validate it there, package
that complete run group as a transfer artifact, and then use a serialized
importer to promote the run group into the canonical analysis Zarr. The
tarball is only a network-transfer optimization; the durable store remains
Zarr. See `docs/cluster_run_group_artifact_workflow.md`.

First implementation slice:

```bash
scripts/py -m fisheye.utils.run_detection_artifact \
  /groups/johnson/johnsonlab/jeremy/palette_smoke/<recording>/cams/<camera>.mp4 \
  --target-zarr /groups/johnson/johnsonlab/jeremy/palette_smoke/<recording>/zarr/<recording>_analysis.zarr \
  --model /groups/johnson/johnsonlab/jeremy/palette_models/<model>/weights/best.pt \
  --config configs/fisheye/yolo_detect_config.yaml \
  --decode-backend auto \
  --batch-size 16 \
  --artifact-dir /scratch/$USER/$LSB_JOBID/palette_run_group_artifact \
  --work-dir /scratch/$USER/$LSB_JOBID/work \
  --tarball-output /scratch/$USER/$LSB_JOBID/<recording>.<jobid>.tar.gz
```

The artifact runner writes predictions into a scratch-only temporary Zarr,
copies only the completed `detect_runs/<run_name>` group into
`palette_run_group_artifact/run_group/`, writes `artifact_manifest.json` plus
validation reports, and creates a `.tar.gz`. It records
`latest_policy=do_not_set_latest` and never mutates the canonical analysis
Zarr. The stdout summary is strict JSON and includes `artifact_timing` for the
scratch Zarr detection call, run-group copy, validation, hashing, and tarball
creation. The artifact manifest records the command, git state, LSF job
metadata, GPU/device details, runtime Python executable/prefix, key packages,
and selected scheduler/GPU environment variables. To submit this as one LSF job
and copy the resulting tarball back to PRFS:

```bash
scripts/submit_detect_artifact_bsub.sh \
  --zarr /groups/johnson/johnsonlab/jeremy/palette_smoke/<recording>/zarr/<recording>_analysis.zarr \
  --video /groups/johnson/johnsonlab/jeremy/palette_smoke/<recording>/cams/<camera>.mp4 \
  --model /groups/johnson/johnsonlab/jeremy/palette_models/<model>/weights/best.pt \
  --output-dir /groups/johnson/johnsonlab/jeremy/palette_smoke/detect_artifacts \
  --decode-backend auto \
  --batch-size 16
```

For registry-scoped full-recording runs, prefer the artifact-chain submitter
over the direct-write detect array. It discovers targets with
`run_detections_batch --source registry --json`, submits one GPU artifact job
per recording, then submits one dependent CPU postprocess job per recording.
The CPU job imports the artifact, validates the imported run, runs
`detect_quality_batch`, and runs `refine_detect_batch` against the explicit
detect run name. It also uses a deterministic quality run name and passes that
name into refinement. This preserves per-recording parallelism while avoiding
GPU jobs writing `detect_runs` chunks directly to PRFS/NRS:

```bash
scripts/submit_detect_artifact_quality_refine_bsub.sh \
  --root /groups/johnson/johnsonlab/jeremy/recordings \
  --registry /groups/johnson/johnsonlab/jeremy/registries/palette_registry.sqlite \
  --path-contains GoodCopBadCop \
  --detect-queue gpu_l4 \
  --detect-decode-backend pynvvc_nv12_rgb \
  --detect-resize-dims 640 640 \
  --detect-batch-size 16 \
  --post-queue short \
  --run-id goodcopbadcop_detect_artifact_refine_$(date -u +%Y%m%dT%H%M%SZ)
```

Add `--submit` on an LSF login node after inspecting the dry-run output. Logs,
target JSONL, generated job scripts, imported-run validation reports, and
artifact tarballs live under
`<root>/logs/detect_artifact_quality_refine_bsub/detect_artifact_quality_refine_<run_id>/`.
By default, the submitter asks `run_detections_batch --dry-run --json
--resolve-models` to resolve the registry-selected detect model for each
recording, then writes that selected path into `targets.jsonl` and
`submissions.tsv`. Pass `--model /path/to/best.pt` only when you want to bypass
registry model resolution deliberately. `--detect-set-id`,
`--detect-require-unique`, `--detect-top-k`, and
`--detect-include-non-success` are forwarded to the registry resolver.
Both the registry file and the selected model path must be readable from the
LSF submit host. On Janelia login/compute nodes, use the canonical PRFS
registry at `/groups/johnson/johnsonlab/jeremy/registries/palette_registry.sqlite`.
Do not pass a workstation-local registry such as `/nvme1/palette_registry.sqlite`
unless you are deliberately debugging a local-only snapshot. If registry model
rows still point at workstation-local paths such as `/nvme1/models/...`,
pass an explicit PRFS/NRS-visible model path under `/groups/...` or refresh the
registry model artifact paths before submitting cluster jobs.
The submitter records `latest_policy=set_latest_explicit` by default so a
successfully imported artifact leaves the normal `detect_runs/latest` surface
consistent with direct-write detection. Downstream CPU steps still pass the
explicit detect and quality run names and do not rely on `latest`.

The cluster submitter defaults to `--detect-decode-backend pynvvc_nv12_rgb` and
`--detect-resize-dims 640 640`. Keep those settings visible in operator
commands. If `resize_dims` is omitted and the detector takes a tensor input,
Ultralytics can receive full `4512x4512` frames, which has been observed to
drop L4 throughput to roughly `5 fps` instead of the expected near-realtime
path. Runtime detection now fails fast for GPU tensor decoder paths
(`pynvvc_*` and Decord GPU) when no explicit resize dims are resolved.

For rolling-clip smoke runs, pass the clip context explicitly and point
`--video` at the per-clip MP4. The submitter writes
`submission_context.json`, includes the context in stdout, and passes it to the
artifact manifest and summary:

```bash
scripts/submit_detect_artifact_bsub.sh \
  --zarr /groups/johnson/johnsonlab/jeremy/palette_smoke/<recording>/zarr/<recording>_analysis.zarr \
  --video /groups/johnson/johnsonlab/jeremy/palette_smoke/<recording>/clips/clip_000000/Cam<serial>_<recording>.mp4 \
  --model /groups/johnson/johnsonlab/jeremy/palette_models/<model>/weights/best.pt \
  --output-dir /groups/johnson/johnsonlab/jeremy/palette_smoke/detect_artifacts/<recording> \
  --workflow-id <workflow_id> \
  --recording-id <recording> \
  --clip-id clip_000000 \
  --clip-index 0 \
  --camera-serial <serial> \
  --decode-backend auto \
  --batch-size 16
```

The importer can promote detection artifacts into either top-level
`detect_runs/<run>` or a clip-local intended target. Clip smoke packages record
`intended_target_group_path=clips/<clip_id>/cameras/<camera_serial>/detect_runs/<run>`;
use `--use-intended-target` when importing those packages into clipped analysis
archives.

Apply-mode importer support is a separate serialized step. Until apply has
completed successfully, packages are durable transfer artifacts, not canonical
completed analysis runs. The LSF wrapper also writes `<label>.<JOBID>.transfer.json`
next to the tarball with the scratch-to-PRFS copy timing.

Dry-run importer validation:

```bash
scripts/py -m fisheye.utils.import_run_group_artifact \
  /groups/johnson/johnsonlab/jeremy/palette_smoke/detect_artifacts/<run>/<label>.<JOBID>.tar.gz \
  --use-intended-target
```

The dry-run importer extracts to a temporary validation directory, recomputes
the run-group tree hash, checks the target archive/final path, and prints the
planned `.incoming` promotion without mutating the canonical Zarr.

Apply mode:

```bash
scripts/py -m fisheye.utils.import_run_group_artifact \
  /groups/johnson/johnsonlab/jeremy/palette_smoke/detect_artifacts/<run>/<label>.<JOBID>.tar.gz \
  --use-intended-target \
  --apply
```

Apply mode copies the packaged run group to
`clips/<clip_id>/cameras/<camera_serial>/detect_runs/.incoming/<run_name>/`,
revalidates it there, promotes it to
`clips/<clip_id>/cameras/<camera_serial>/detect_runs/<run_name>/`, and writes an
import receipt at
`clips/<clip_id>/cameras/<camera_serial>/detect_runs/.imports/<run_name>_import_receipt.json`.
For first use on a new cluster path, test with `--target-zarr` pointing at a
disposable Zarr before applying to the canonical archive.

**Logs:** `<root>/logs/run_detections_batch/bsub_submissions/detect_<run_id>/`

---

## 2. Crop batch

### Python runner

```bash
# Dry run
scripts/py -m fisheye.utils.crop_batch /groups/johnson/johnsonlab/jeremy/recordings --recursive --dry-run

# Apply
scripts/py -m fisheye.utils.crop_batch /groups/johnson/johnsonlab/jeremy/recordings --recursive --apply

# Registry mode — only crop recordings where detect is 'ok'
scripts/py -m fisheye.utils.crop_batch /groups/johnson/johnsonlab/jeremy/recordings \
  --source registry \
  --registry /groups/johnson/johnsonlab/jeremy/registries/palette_registry.sqlite \
  --apply
```

Notes:
- Defaults to `source_type=auto` (uses review status or the refined fallback chain).
- Skips when the latest crop run already matches the resolved detection source
  and ROI size (use `--force-new` to always create a new run).
- No ML model resolution — crop is a deterministic operation using existing
  detections.

### LSF submit script

```bash
./scripts/submit_crop_batches_bsub.sh \
  --root /groups/johnson/johnsonlab/jeremy/recordings \
  --source registry \
  --batch-size 10 \
  --max-active 2 \
  --queue gpu_l4 \
  --gpus 1 \
  --mem-gb 32 \
  --registry /groups/johnson/johnsonlab/jeremy/registries/palette_registry.sqlite \
  --dry-run
```

**Key options:**

| Flag                       | Default | Description                               |
|----------------------------|---------|-------------------------------------------|
| `--root`                   | `/groups/johnson/johnsonlab/jeremy/recordings` | Root recordings directory     |
| `--source`                 | `filesystem` | Discovery source                     |
| `--batch-size`             | `10`    | Analysis zarrs per batch job              |
| `--max-active`             | `2`     | Max concurrent jobs                       |
| `--mem-gb`                 | `32`    | Memory per job in GB                      |
| `--registry`               | `/groups/johnson/johnsonlab/jeremy/registries/palette_registry.sqlite` | Registry path  |
| `--config`                 | *(none)* | Crop config YAML                         |
| `--source-type`            | *(auto)* | Detection source type                    |
| `--acceleration`           | *(auto)* | `auto`, `gpu`, or `cpu`                  |
| `--external-write-backend` | *(standard)* | `standard` or `kvikio`              |
| `--external-roi-storage`   | *(compressed)* | `compressed` or `uncompressed`    |
| `--force-new`              | off     | Always create new crop run (disables skip-existing) |
| `--zarr-use`               | `analysis` | Zarr use filter                        |
| `--dry-run`                | off     | Print manifests + commands; do not submit |

**Execution model:** Each batch job calls `crop_batch --apply` with a batch
of zarr paths. No per-recording model resolution — crop works directly on
the zarr.

**Logs:** `<root>/logs/crop_batch/bsub_submissions/crop_<run_id>/`

---

## 3. Keypoints batch

### Python runner

```bash
# Registry mode — only run keypoints where detect + crop are 'ok'
scripts/py -m fisheye.utils.run_keypoints_batch /groups/johnson/johnsonlab/jeremy/recordings \
  --source registry \
  --registry /groups/johnson/johnsonlab/jeremy/registries/palette_registry.sqlite \
  --dry-run --json
```

### LSF submit script

```bash
./scripts/submit_keypoints_batches_bsub.sh \
  --root /groups/johnson/johnsonlab/jeremy/recordings \
  --source registry \
  --batch-size 10 \
  --max-active 2 \
  --queue short \
  --mem-gb 32 \
  --registry /groups/johnson/johnsonlab/jeremy/registries/palette_registry.sqlite \
  --dry-run
```

**Key options:**

| Flag                  | Default | Description                                  |
|-----------------------|---------|----------------------------------------------|
| `--root`              | `/groups/johnson/johnsonlab/jeremy/recordings` | Root recordings directory        |
| `--source`            | `filesystem` | Discovery source                        |
| `--batch-size`        | `10`    | Analysis zarrs per batch job                 |
| `--max-active`        | `2`     | Max concurrent jobs                          |
| `--mem-gb`            | `32`    | Memory per job in GB                         |
| `--gpus`              | `0`     | GPUs per job; when >0 the submitter requests LSF GPUs and defaults `--device 0` |
| `--registry`          | `/groups/johnson/johnsonlab/jeremy/registries/palette_registry.sqlite` | Registry path     |
| `--set-id`            | *(none)* | Pose model set filter                       |
| `--pose-schema`       | *(detector default)* | Pose schema to pair with the selected model, e.g. `traditional_v2` for 5-keypoint models |
| `--top-k`             | `5`     | Candidate provenance depth                   |
| `--require-unique`    | off     | Fail if top model scores tie                 |
| `--include-non-success` | off   | Include non-success runs in model resolution |
| `--crop-run`          | *(auto)* | Explicit crop run name                      |
| `--batch-size-kp`     | `256`   | Keypoint inference batch size                |
| `--device`            | *(auto)* | Torch device override                       |
| `--roi-cache-dir`     | *(none)* | Scratch directory for temporary Zarr ROI caches |
| `--roi-cache-manifest` | *(none)* | Explicit `flat_bin_v1` ROI cache manifest; accepted only when exactly one target is selected |
| `--stage-roi-cache-to-scratch` | off | Copy the explicit flat-cache manifest/payload to node-local scratch before inference |
| `--roi-cache-staging-dir` | *(auto)* | Override staging directory; default prefers `/scratch/$USER/$LSB_JOBID` |
| `--cpu`               | off     | Force CPU inference                          |
| `--overwrite`         | off     | Rerun even if keypoints run exists           |
| `--camera-id-filter`  | *(none)* | Filter by camera_id (registry source only)  |
| `--dry-run`           | off     | Print manifests + commands; do not submit    |

**Execution model:** Each batch job iterates over zarr paths in its batch
file. For each zarr, it derives the recording directory and calls
`run_keypoints_with_registry_model --recording-dir <dir>`. Model resolution
happens per-recording at runtime.

For GPU keypoint jobs, pass `--queue gpu_l4 --gpus 1`. The submitter will request
`-gpu num=1` from LSF and, unless `--device` is already supplied, will pass
`--device 0` into the per-recording keypoint command. Keypoint runs record the
requested device, normalized torch device, resolved model device where available,
execution hostname/FQDN, LSF job id/name/index/queue, allocated hosts, LSF GPU
request, and CUDA-visible devices in run attrs/provenance. This is the provenance
surface to use when comparing single-recording jobs against packed multi-recording
jobs on the same L4 node.

`--roi-cache-dir` and `--roi-cache-manifest` are intentionally different:
`--roi-cache-dir` is a scratch root where the reader may build/reuse a
temporary Zarr cache, while `--roi-cache-manifest` points at a completed flat
binary cache manifest. Use `--stage-roi-cache-to-scratch` only with an explicit
manifest. The batch submitter refuses an explicit manifest when more than one
analysis Zarr is selected, because one manifest cannot safely describe multiple
recordings.

For large flat-cache manifests, stage before GPU inference:

```bash
./scripts/submit_keypoints_batches_bsub.sh \
  --root /groups/johnson/johnsonlab/jeremy/recordings \
  --source filesystem \
  --batch-size 1 \
  --max-active 1 \
  --queue gpu_l4 \
  --gpus 1 \
  --registry /groups/johnson/johnsonlab/jeremy/registries/palette_registry.sqlite \
  --roi-cache-manifest /groups/.../<recording>.flat_roi_cache.json \
  --stage-roi-cache-to-scratch \
  --pose-schema traditional_v2
```

The 2026-06-18 GoodCopBadCop benchmark showed a 33.4 GiB flat cache improved
from 212.2 poses/s direct from PRFS to 275.8 poses/s after staging to node-local
scratch. Even including the 45.8s cache copy, staged execution was faster
end-to-end. Use staging as the default policy for large caches; direct PRFS
reads are mainly for small caches or explicit comparisons.

After jobs finish, use the registry performance view plus step-status details
to compare throughput against actual cluster placement:

```bash
sqlite3 /groups/johnson/johnsonlab/jeremy/registries/palette_registry.sqlite <<'SQL'
.headers on
.mode column
SELECT
  k.recording_id,
  k.keypoint_run,
  ROUND(k.keypoints_per_second, 1) AS poses_per_s,
  json_extract(s.details_json, '$.scheduler_hosts') AS hosts,
  json_extract(s.details_json, '$.scheduler_gpu_request') AS gpu_request,
  json_extract(s.details_json, '$.scheduler_cuda_visible_devices') AS cuda_visible,
  json_extract(s.details_json, '$.roi_cache_source_tier') AS cache_tier,
  json_extract(s.details_json, '$.roi_cache_staging_policy') AS staging_policy,
  json_extract(s.details_json, '$.roi_cache_staged_to_node_scratch') AS staged
FROM recording_keypoint_performance_latest AS k
LEFT JOIN recording_step_status AS s
  ON s.dataset_id = k.dataset_id
 AND s.step_name = 'keypoints'
ORDER BY k.updated_utc DESC
LIMIT 20;
SQL
```

**Logs:** `<root>/logs/run_keypoints_batch/bsub_submissions/kp_<run_id>/`

---

## 4. Subject masks batch

### Python runner

```bash
# Registry mode - only select recordings where crop + keypoints are ok and
# refined_subject_masks is not already ok.
scripts/py -m fisheye.utils.run_subject_mask_batch_pipeline \
  /groups/johnson/johnsonlab/jeremy/recordings \
  --source registry \
  --registry /groups/johnson/johnsonlab/jeremy/registries/palette_registry.sqlite \
  --emit-paths
```

```bash
# Apply one recording locally/inside an allocated job. Prefer an explicit
# flat ROI cache manifest so subject-mask inference does not re-decode crops.
scripts/py -m fisheye.utils.run_subject_mask_batch_pipeline \
  /groups/johnson/johnsonlab/jeremy/recordings/<recording>/zarr/<recording>_analysis.zarr \
  --apply \
  --source registry \
  --registry /groups/johnson/johnsonlab/jeremy/registries/palette_registry.sqlite \
  --device 0 \
  --batch-size 128 \
  --roi-cache-manifest /groups/.../<recording>.flat_roi_cache.json
```

### LSF submit script

```bash
./scripts/submit_subject_mask_batches_bsub.sh \
  --root /groups/johnson/johnsonlab/jeremy/recordings \
  --source registry \
  --registry /groups/johnson/johnsonlab/jeremy/registries/palette_registry.sqlite \
  --roi-cache-root /groups/johnson/johnsonlab/jeremy/cache/<workflow>/roi_cache \
  --queue gpu_l4 \
  --gpus 1 \
  --max-active 4 \
  --dry-run
```

**Key options:**

| Flag                       | Default | Description                              |
|----------------------------|---------|------------------------------------------|
| `--root`                   | `/groups/johnson/johnsonlab/jeremy/recordings` | Root recordings directory    |
| `--source`                 | `registry` | Discovery source                    |
| `--max-active`             | `4`     | Max concurrent one-recording jobs        |
| `--mem-gb`                 | `48`    | Memory per job in GB                     |
| `--gpus`                   | `1`     | GPUs per job                             |
| `--registry`               | `/groups/johnson/johnsonlab/jeremy/registries/palette_registry.sqlite` | Registry path |
| `--roi-cache-root`         | *(none)* | Directory containing per-recording `*.flat_roi_cache.json` manifests |
| `--roi-cache-manifest`     | *(none)* | Explicit flat ROI cache manifest for exactly one selected recording |
| `--no-stage-roi-cache-to-scratch` | off | Disable node-local cache staging |
| `--allow-missing-roi-cache` | off    | Allow fallback when no flat cache manifest is found |
| `--batch-size-sm`          | `128`   | Subject-mask inference batch size        |
| `--device`                 | `0` when `--gpus > 0` | Torch device override       |
| `--overwrite`              | off     | Pass overwrite through to child stages   |
| `--camera-id-filter`       | *(none)* | Filter by camera_id (registry only)     |
| `--dry-run`                | off     | Print manifests + commands; do not submit|

**Execution model:** one LSF array task per recording. Each task resolves that
recording's flat ROI cache manifest, stages the manifest and `.bin` payload to
node-local scratch by default, passes the staged manifest into
`run_subject_mask_batch_pipeline`, and removes the staged local cache with an
`EXIT` trap on success or failure.

**Logs:** `<root>/logs/run_subject_mask_batch/bsub_submissions/sm_<run_id>/`

The historical `scripts/submit_eye_masks_batches_bsub.sh` remains available for
legacy `eye_masks_runs` compatibility, but new full-component mask jobs should
target `subject_mask_runs` and `refined_subject_masks_runs`.

---

## Running the full pipeline with registry mode

To process all four stages sequentially using registry-backed discovery:

```bash
# 1. Detection — no prerequisites
./scripts/submit_detect_batches_bsub.sh \
  --source registry --registry /groups/johnson/johnsonlab/jeremy/registries/palette_registry.sqlite \
  --batch-size 15 --max-active 2

# 2. Crop — requires detect='ok'
./scripts/submit_crop_batches_bsub.sh \
  --source registry --registry /groups/johnson/johnsonlab/jeremy/registries/palette_registry.sqlite \
  --batch-size 10 --max-active 2

# 3. Keypoints — requires detect='ok' and crop='ok'
./scripts/submit_keypoints_batches_bsub.sh \
  --source registry --registry /groups/johnson/johnsonlab/jeremy/registries/palette_registry.sqlite \
  --batch-size 10 --max-active 2

# 4. Subject masks — requires crop='ok' and keypoints='ok'
./scripts/submit_subject_mask_batches_bsub.sh \
  --source registry --registry /groups/johnson/johnsonlab/jeremy/registries/palette_registry.sqlite \
  --roi-cache-root /groups/johnson/johnsonlab/jeremy/cache/<workflow>/roi_cache \
  --max-active 4
```

Each stage automatically filters to only recordings that have completed all
prerequisites. You can safely submit all four in quick succession — later
stages will simply find zero targets if earlier stages haven't finished yet,
then can be re-submitted.

### Scoping to a subset

All submit scripts accept registry filters:

```bash
./scripts/submit_detect_batches_bsub.sh \
  --source registry \
  --rig-id rig_01 \
  --arena-id arena_A \
  --path-contains 2025-01
```

---

## How submit scripts work

The submit scripts follow the same broad pattern:

1. **Discovery** — Run the Python batch module with `--emit-paths` to get a
   list of zarr paths (either via filesystem glob or registry SQL query).
2. **Manifest** — Split or enumerate discovered paths into per-job files
   (`batch_0001.txt` or `target_0001.tsv`, depending on the stage) and write
   `recordings.txt` plus `manifest_summary.json`.
3. **Job script** — Generate a `run_batch.sh` script that reads a batch file
   and processes each recording.
4. **Submit** — Submit an LSF job array with `bsub`.

### Run directory structure

```
<log_dir>/<prefix>_<run_id>/
├── recordings.txt           # All discovered zarr paths
├── discovered_paths.txt     # Raw output from --emit-paths (registry mode)
├── manifest_summary.json    # Source, counts, batch size
├── batch_0001.txt           # First batch of zarr paths
├── batch_0002.txt           # Second batch
├── ...
├── run_batch.sh             # Generated job script
├── <jobid>_1.out            # LSF stdout for batch 1
├── <jobid>_1.err            # LSF stderr for batch 1
└── ...
```

### Dry-run mode

All scripts support `--dry-run` which performs discovery and manifest creation
but does not submit to LSF. Use this to verify the target list before
committing:

```bash
./scripts/submit_detect_batches_bsub.sh --source registry --dry-run
```

### Stable run IDs

Use `--run-id` for deterministic reruns:

```bash
./scripts/submit_detect_batches_bsub.sh --run-id my_rerun_001 --dry-run
```

### Chained Detect, Quality, And Refine Submission

For registry-discovered recordings that are missing detections completely, use
the chained submitter for conservative pilot runs, or use
`scripts/submit_detect_artifact_quality_refine_bsub.sh` when you want the safer
scratch-artifact/import path. The direct-write chained submitter discovers the
zarr target set once, submits detect as an LSF array, then submits
`detect_quality_batch` and `refine_detect_batch` as dependent CPU jobs using
`done(<jobid>)`.

The direct-write chained submitter below is a convenience path, not the
preferred broad production path. It writes detection output directly into the
canonical Zarr from GPU jobs, then schedules CPU quality/refine jobs that use
the archive's latest run selection at postprocess time. Use it only for
low-concurrency pilots or one-off local/cluster runs where no other writer is
mutating the same archives. For broad registry batches, use
`submit_detect_artifact_quality_refine_bsub.sh`.

Dry-run first:

```bash
./scripts/submit_detect_quality_refine_bsub.sh \
  --root /groups/johnson/johnsonlab/jeremy/recordings \
  --registry /groups/johnson/johnsonlab/jeremy/registries/palette_registry.sqlite \
  --path-contains GoodCopBadCop \
  --detect-decode-backend pynvvc_nv12_rgb \
  --detect-resize-dims 640 640 \
  --run-id goodcopbadcop_detect_quality_refine_$(date -u +%Y%m%dT%H%M%SZ)
```

Submit on an LSF login node:

```bash
./scripts/submit_detect_quality_refine_bsub.sh \
  --root /groups/johnson/johnsonlab/jeremy/recordings \
  --registry /groups/johnson/johnsonlab/jeremy/registries/palette_registry.sqlite \
  --path-contains GoodCopBadCop \
  --detect-queue gpu_l4 \
  --detect-gpu 'num=1' \
  --detect-decode-backend pynvvc_nv12_rgb \
  --detect-resize-dims 640 640 \
  --detect-batch-size 4 \
  --detect-max-active 2 \
  --quality-queue short \
  --quality-walltime 1:00 \
  --refine-queue short \
  --refine-walltime 1:00 \
  --run-id goodcopbadcop_detect_quality_refine_$(date -u +%Y%m%dT%H%M%SZ) \
  --submit
```

The dependency chain is fail-closed: if the detect array does not finish with
`DONE`, the quality job remains pending; if quality fails, refine remains
pending. Logs and the exact discovered `recordings.txt` live under
`<root>/logs/detect_quality_refine_bsub/detect_quality_refine_<run_id>/`.
On the Janelia `short` queue, keep CPU postprocess walltimes at or below
`1:00`; longer requests can be rejected by LSF before the dependency chain is
fully submitted.

---

## How to check which scheduler you have

On a login node:

```bash
which bsub    # LSF
which sbatch  # Slurm
which qsub    # PBS/Torque
```

If `bsub` exists → LSF (the submit scripts require this).

## Notes from HPC engineers (Zarr I/O)

- Prefer **bigger sustained writes** over many tiny jobs.
- Keep the number of simultaneous writers low.
- If local scratch is available, consider writing locally and copying back
  in large chunks.
- For long-running cluster jobs that create new analysis runs, prefer the
  run-group artifact workflow in
  `docs/cluster_run_group_artifact_workflow.md`: write complete run groups on
  scratch, package them, and import into the canonical Zarr with a serialized
  importer.

## Headless logs (Rich output)

Batch scripts use Rich progress bars by default. On headless schedulers this
is safe, but the control characters can make log files noisy. To keep logs
clean, disable Rich rendering:

```bash
RICH_DISABLE=1 scripts/py -m fisheye.utils.crop_batch ...
```

Or force a dumb terminal:

```bash
TERM=dumb scripts/py -m fisheye.utils.crop_batch ...
```

## Suggested defaults

| Parameter       | Detect | Crop  | Keypoints | Eye Masks |
|-----------------|--------|-------|-----------|-----------|
| `--batch-size`  | 10–30  | 10    | 10        | 10        |
| `--max-active`  | 1–2    | 1–2   | 1–2       | 1–2       |
| `--mem-gb`      | 16     | 32    | 32        | 32        |
| `--ncores`      | 4      | 4     | 4         | 4         |

Adjust upward only after monitoring I/O (`iostat`, `iotop`) and queue health.
