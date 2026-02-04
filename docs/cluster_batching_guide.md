# Cluster Batching Guide (Detection / Background)

This guide summarizes best practices for running Palette detection/background
jobs on a shared HPC filesystem and includes an example LSF `bsub` wrapper.

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

## Heuristics that work well

- If a task is **<5–10 minutes**, batch it.
- If a task is **>30 minutes**, batch fewer recordings or submit one per job.
- Use **max active jobs** to cap concurrency (LSF array `%` syntax).

## Detect jobs (batch script)

We added a batch runner:

```
python src/fisheye/utils/run_detections_batch.py /nvme1/recordings --recursive --apply --no-dask-progress
```

It skips zarrs that already have `detect_runs/latest`, unless `--overwrite` is
set.

## Crop jobs (batch script)

We added a batch crop runner:

```
python -m fisheye.utils.crop_batch /nvme1/recordings --recursive --apply
```

Notes:
- Defaults to `source_type=preferred` (uses review status or preferred chain).
- Skips when the latest crop run already matches the resolved detection source
  and ROI size (use `--force-new` to always create a new run).

## LSF (bsub) wrapper

Script: `scripts/submit_detect_batches_bsub.sh`

Example:

```
./scripts/submit_detect_batches_bsub.sh \
  --root /nvme1/recordings \
  --batch-size 15 \
  --max-active 2 \
  --queue short \
  --ncores 4 \
  --mem-gb 16 \
  --scheduler threads \
  --require-tuning
```

This:
- Finds all recordings under `--root`
- Splits into batches of `--batch-size`
- Submits an LSF job array with at most `--max-active` running at once

Logs and batch files go under:
```
/nvme1/recordings/logs/run_detections_batch/bsub_submissions/
```

## How to check which scheduler you have

On a login node:

```
which bsub
which sbatch
which qsub
```

If `bsub` exists → LSF.
If `sbatch` exists → Slurm.
If `qsub` exists → PBS/Torque.

## Notes from HPC engineers (Zarr I/O)

- Prefer **bigger sustained writes** over many tiny jobs.
- Keep the number of simultaneous writers low.
- If local scratch is available, consider writing locally and copying back
  in large chunks.

## Headless logs (Rich output)

Batch scripts use Rich progress bars by default. On headless schedulers this
is safe, but the control characters can make log files noisy. To keep logs
clean, disable Rich rendering:

```
RICH_DISABLE=1 python -m fisheye.utils.crop_batch ...
```

Or force a dumb terminal:

```
TERM=dumb python -m fisheye.utils.crop_batch ...
```

## Suggested defaults

- `--batch-size 10–30`
- `--max-active 1–2`
- `--scheduler threads`
- `--num-workers 4–8`

Adjust upward only after monitoring I/O (`iostat`, `iotop`) and queue health.
