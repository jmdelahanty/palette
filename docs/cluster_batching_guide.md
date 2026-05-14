# Cluster Batching Guide

This guide covers running Palette pipeline stages as batch jobs on an HPC
cluster with LSF. All four stages — **detect**, **crop**, **keypoints**, and
**eye masks** — support both filesystem discovery and registry-backed SQL
pre-filtering.

Related contract:
- `docs/detect_batch_analysis_zarr_parallel_agents_contract.md`
- `docs/cluster_run_group_artifact_workflow.md`
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

## Analysis-Stage Scaling Policy

For downstream analysis stages such as track kinematics, swim-bout detection,
bout kinematics, eye angles, and stimulus response metrics, prefer
**recording-level parallelism** as the first scaling layer. On the cluster this
means submitting one recording or a small batch of recordings per job, rather
than assuming every analysis stage is internally Dask-aware.

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

The four batch pipelines form a DAG. Each stage requires its predecessors to
have `recording_step_status = 'ok'` before it will process a recording:

```
detect  →  crop  →  keypoints  →  eye_masks
```

| Stage      | Prerequisite steps          | Step name      | Skip-existing flag |
|------------|----------------------------|----------------|-------------------|
| detect     | *(none)*                   | `detect`       | `--overwrite`     |
| crop       | `detect`                   | `crop`         | `--force-new`     |
| keypoints  | `detect`, `crop`           | `keypoints`    | `--overwrite`     |
| eye masks  | `crop`, `keypoints`        | `eye_masks`    | `--overwrite`     |

When using `--source registry`, the batch runner queries the registry's
`recording_step_status` table with SQL-level pre-filtering:

- **`require_steps_ok`** — INNER JOIN ensures only recordings where all
  prerequisite steps have status `'ok'` are returned.
- **`exclude_step_ok`** — LEFT JOIN excludes recordings where this stage
  already has status `'ok'` (skip-existing behavior). Disabled when the
  overwrite/force-new flag is set.

---

## Discovery modes

All four batch runners support two discovery modes via `--source`:

### Filesystem mode (default)

```bash
scripts/py -m fisheye.utils.run_detections_batch /nvme1/recordings --recursive --dry-run --json
```

Recursively finds `*_analysis.zarr` targets under the given root directory.
No registry interaction at discovery time.

### Registry mode

```bash
scripts/py -m fisheye.utils.run_detections_batch /nvme1/recordings \
  --source registry \
  --registry /nvme1/palette_registry.sqlite \
  --dry-run --json
```

Queries the registry database for analysis zarrs matching the scope, applies
SQL-level prerequisite and skip-existing filters, then builds plans only for
the resulting paths. This is faster than filesystem discovery on large trees
and ensures only "ready" recordings are processed.

**Registry filters** (available on all four runners):

| Flag               | Description                                 |
|--------------------|---------------------------------------------|
| `--registry PATH`  | Path to the registry SQLite file            |
| `--rig-id ID`      | Filter by rig identifier                    |
| `--arena-id ID`    | Filter by arena identifier                  |
| `--camera-id ID` / `--camera-id-filter ID` | Filter by camera identifier (see note) |
| `--path-contains STR` | Substring match on zarr_path             |

> **Note:** Detection and crop use `--camera-id`. Keypoints and eye masks use
> `--camera-id-filter` to avoid ambiguity with the per-recording camera_id
> argument those runners also accept.

### Emit-paths mode

All runners support `--emit-paths` which prints discovered zarr paths to
stdout and exits immediately. The LSF submit scripts use this to build
batch manifests:

```bash
scripts/py -m fisheye.utils.run_detections_batch /nvme1/recordings \
  --source registry --emit-paths \
  --registry /nvme1/palette_registry.sqlite
```

---

## 1. Detection batch

### Python runner

```bash
# Dry run — preview plans
scripts/py -m fisheye.utils.run_detections_batch /nvme1/recordings \
  --recursive --dry-run --json

# Apply — run detections with registry-backed model resolution
scripts/py -m fisheye.utils.run_detections_batch /nvme1/recordings \
  --recursive --apply \
  --registry /nvme1/palette_registry.sqlite

# Registry mode — only process recordings not yet detected
scripts/py -m fisheye.utils.run_detections_batch /nvme1/recordings \
  --source registry \
  --registry /nvme1/palette_registry.sqlite \
  --apply
```

It plans directly against `*_analysis.zarr` targets and skips archives that
already have `detect_runs/latest` unless `--overwrite` is set.

### LSF submit script

```bash
./scripts/submit_detect_batches_bsub.sh \
  --root /nvme1/recordings \
  --source registry \
  --batch-size 15 \
  --max-active 2 \
  --queue short \
  --ncores 4 \
  --mem-gb 16 \
  --registry /nvme1/palette_registry.sqlite \
  --require-tuning \
  --dry-run
```

**Key options:**

| Flag               | Default | Description                                      |
|--------------------|---------|--------------------------------------------------|
| `--root`           | `/nvme1/recordings` | Root recordings directory             |
| `--source`         | `filesystem` | Discovery source (`filesystem` or `registry`) |
| `--batch-size`     | `10`    | Analysis zarrs per batch job                     |
| `--max-active`     | `2`     | Max concurrent jobs in array                     |
| `--queue`          | *(default)* | LSF queue name                               |
| `--ncores`         | `4`     | Cores per job                                    |
| `--mem-gb`         | `16`    | Memory per job in GB                             |
| `--registry`       | `/nvme1/palette_registry.sqlite` | Registry path       |
| `--config`         | `configs/fisheye/default.yaml` | Detection config     |
| `--set-id`         | *(none)* | Detect model set filter                         |
| `--require-tuning` | off     | Skip zarrs without detection_tuning              |
| `--overwrite`      | off     | Rerun even if detect_runs/latest exists          |
| `--dry-run`        | off     | Print manifests + commands; do not submit        |

**Execution model:** Each batch job calls `run_detections_batch --apply` with
a batch of zarr paths. Model resolution happens inside the batch runner.

**Logs:** `<root>/logs/run_detections_batch/bsub_submissions/detect_<run_id>/`

---

## 2. Crop batch

### Python runner

```bash
# Dry run
scripts/py -m fisheye.utils.crop_batch /nvme1/recordings --recursive --dry-run

# Apply
scripts/py -m fisheye.utils.crop_batch /nvme1/recordings --recursive --apply

# Registry mode — only crop recordings where detect is 'ok'
scripts/py -m fisheye.utils.crop_batch /nvme1/recordings \
  --source registry \
  --registry /nvme1/palette_registry.sqlite \
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
  --root /nvme1/recordings \
  --source registry \
  --batch-size 10 \
  --max-active 2 \
  --queue short \
  --mem-gb 32 \
  --registry /nvme1/palette_registry.sqlite \
  --dry-run
```

**Key options:**

| Flag                       | Default | Description                               |
|----------------------------|---------|-------------------------------------------|
| `--root`                   | `/nvme1/recordings` | Root recordings directory     |
| `--source`                 | `filesystem` | Discovery source                     |
| `--batch-size`             | `10`    | Analysis zarrs per batch job              |
| `--max-active`             | `2`     | Max concurrent jobs                       |
| `--mem-gb`                 | `32`    | Memory per job in GB                      |
| `--registry`               | `/nvme1/palette_registry.sqlite` | Registry path  |
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
scripts/py -m fisheye.utils.run_keypoints_batch /nvme1/recordings \
  --source registry \
  --registry /nvme1/palette_registry.sqlite \
  --dry-run --json
```

### LSF submit script

```bash
./scripts/submit_keypoints_batches_bsub.sh \
  --root /nvme1/recordings \
  --source registry \
  --batch-size 10 \
  --max-active 2 \
  --queue short \
  --mem-gb 32 \
  --registry /nvme1/palette_registry.sqlite \
  --dry-run
```

**Key options:**

| Flag                  | Default | Description                                  |
|-----------------------|---------|----------------------------------------------|
| `--root`              | `/nvme1/recordings` | Root recordings directory        |
| `--source`            | `filesystem` | Discovery source                        |
| `--batch-size`        | `10`    | Analysis zarrs per batch job                 |
| `--max-active`        | `2`     | Max concurrent jobs                          |
| `--mem-gb`            | `32`    | Memory per job in GB                         |
| `--registry`          | `/nvme1/palette_registry.sqlite` | Registry path     |
| `--set-id`            | *(none)* | Pose model set filter                       |
| `--top-k`             | `5`     | Candidate provenance depth                   |
| `--require-unique`    | off     | Fail if top model scores tie                 |
| `--include-non-success` | off   | Include non-success runs in model resolution |
| `--crop-run`          | *(auto)* | Explicit crop run name                      |
| `--batch-size-kp`     | `256`   | Keypoint inference batch size                |
| `--device`            | *(auto)* | Torch device override                       |
| `--cpu`               | off     | Force CPU inference                          |
| `--overwrite`         | off     | Rerun even if keypoints run exists           |
| `--camera-id-filter`  | *(none)* | Filter by camera_id (registry source only)  |
| `--dry-run`           | off     | Print manifests + commands; do not submit    |

**Execution model:** Each batch job iterates over zarr paths in its batch
file. For each zarr, it derives the recording directory and calls
`run_keypoints_with_registry_model --recording-dir <dir>`. Model resolution
happens per-recording at runtime.

**Logs:** `<root>/logs/run_keypoints_batch/bsub_submissions/kp_<run_id>/`

---

## 4. Eye masks batch

### Python runner

```bash
# Registry mode — only run eye masks where crop + keypoints are 'ok'
scripts/py -m fisheye.utils.run_eye_masks_batch /nvme1/recordings \
  --source registry \
  --registry /nvme1/palette_registry.sqlite \
  --no-log \
  --dry-run --json
```

```bash
# Registry mode + registry model resolution (U-Net example)
scripts/py -m fisheye.utils.run_eye_masks_batch /nvme1/recordings \
  --source registry \
  --registry /nvme1/palette_registry.sqlite \
  --model-source registry \
  --method unet \
  --model-top-k 5 \
  --dry-run --json
```

```bash
# Apply with registry model resolution (YOLO example)
scripts/py -m fisheye.utils.run_eye_masks_batch /nvme1/recordings \
  --source registry \
  --registry /nvme1/palette_registry.sqlite \
  --model-source registry \
  --method yolo \
  --model-set-id eye_mask_cedar_shadow_omnifin0_auto_gray_union_b9164009_v001 \
  --overwrite \
  --apply --json
```

```bash
# Recommended (your current workflow): U-Net + registry model resolution
scripts/py -m fisheye.utils.run_eye_masks_batch /nvme1/recordings \
  --source registry \
  --registry /nvme1/palette_registry.sqlite \
  --model-source registry \
  --method unet \
  --model-set-id eye_mask_cedar_shadow_omnifin0_auto_gray_union_b9164009_v001 \
  --overwrite \
  --apply --json
```

### LSF submit script

```bash
./scripts/submit_eye_masks_batches_bsub.sh \
  --root /nvme1/recordings \
  --source registry \
  --batch-size 10 \
  --max-active 2 \
  --queue short \
  --mem-gb 32 \
  --registry /nvme1/palette_registry.sqlite \
  --dry-run
```

**Key options:**

| Flag                       | Default | Description                              |
|----------------------------|---------|------------------------------------------|
| `--root`                   | `/nvme1/recordings` | Root recordings directory    |
| `--source`                 | `filesystem` | Discovery source                    |
| `--batch-size`             | `10`    | Analysis zarrs per batch job             |
| `--max-active`             | `2`     | Max concurrent jobs                      |
| `--mem-gb`                 | `32`    | Memory per job in GB                     |
| `--registry`               | `/nvme1/palette_registry.sqlite` | Registry path |
| `--set-id`                 | *(none)* | Eye mask model set filter               |
| `--top-k`                  | `5`     | Candidate provenance depth               |
| `--require-unique`         | off     | Fail if top model scores tie             |
| `--include-non-success`    | off     | Include non-success runs in resolution   |
| `--method`                 | *(auto)* | `yolo` or `unet`                        |
| `--crop-run`               | *(auto)* | Explicit crop run name                  |
| `--keypoints-run`          | *(auto)* | Explicit keypoints run name             |
| `--batch-size-em`          | `128`   | Eye mask inference batch size            |
| `--device`                 | *(auto)* | Torch device override                   |
| `--cpu`                    | off     | Force CPU inference                      |
| `--overwrite`              | off     | Rerun even if eye_masks run exists       |
| `--camera-id-filter`       | *(none)* | Filter by camera_id (registry only)     |
| `--dry-run`                | off     | Print manifests + commands; do not submit|

**YOLO-specific options:** `--resize-dims` (canonical), `--imgsz` (legacy alias),
`--conf`, `--iou`, `--max-det`,
`--mask-threshold`, `--adaptive-scale`, `--adaptive-cap`, `--no-retina-masks`,
`--proto-upsample-factor`, `--legacy-masks`, `--verbose`

**U-Net-specific options:** `--label-mode`, `--write-binary-masks`,
`--no-use-crop`

**Python runner registry-model options:** `--model-source {config,registry}` (default: `config`),
`--model-set-id`, `--model-top-k`, `--model-require-unique`,
`--model-include-non-success`

**Option name mapping (LSF submit script → Python runner):**
`--set-id` → `--model-set-id`, `--top-k` → `--model-top-k`,
`--require-unique` → `--model-require-unique`,
`--include-non-success` → `--model-include-non-success`

**Execution model:** Same as keypoints — each batch job iterates over zarr
paths, derives the recording directory, and calls
`run_eye_masks_with_registry_model --recording-dir <dir>`. Model resolution
happens per-recording at runtime.

**Logs:** `<root>/logs/run_eye_masks_batch/bsub_submissions/em_<run_id>/`

---

## Running the full pipeline with registry mode

To process all four stages sequentially using registry-backed discovery:

```bash
# 1. Detection — no prerequisites
./scripts/submit_detect_batches_bsub.sh \
  --source registry --registry /nvme1/palette_registry.sqlite \
  --batch-size 15 --max-active 2

# 2. Crop — requires detect='ok'
./scripts/submit_crop_batches_bsub.sh \
  --source registry --registry /nvme1/palette_registry.sqlite \
  --batch-size 10 --max-active 2

# 3. Keypoints — requires detect='ok' and crop='ok'
./scripts/submit_keypoints_batches_bsub.sh \
  --source registry --registry /nvme1/palette_registry.sqlite \
  --batch-size 10 --max-active 2

# 4. Eye masks — requires crop='ok' and keypoints='ok'
./scripts/submit_eye_masks_batches_bsub.sh \
  --source registry --registry /nvme1/palette_registry.sqlite \
  --batch-size 10 --max-active 2
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

All four submit scripts follow the same pattern:

1. **Discovery** — Run the Python batch module with `--emit-paths` to get a
   list of zarr paths (either via filesystem glob or registry SQL query).
2. **Manifest** — Split the discovered paths into batch files
   (`batch_0001.txt`, `batch_0002.txt`, ...) and write `recordings.txt` +
   `manifest_summary.json`.
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
