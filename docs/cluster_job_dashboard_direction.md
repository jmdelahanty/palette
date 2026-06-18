# Cluster Job Dashboard Direction
<!-- contract-meta
status: design
last_verified: 2026-06-18
purpose: Capture the near-term direction for live and post-hoc visibility into Palette LSF jobs.
-->

## Purpose

Palette cluster jobs now span LSF scheduler state, per-stage stdout/stderr,
workflow JSON files, progress JSONL streams, Zarr run attrs, and registry
status/performance rows. Operators need one place to answer:

- what jobs are pending/running/done/failed;
- which recording/stage/run each job owns;
- where the job ran;
- what phase it is in;
- whether it is making progress;
- where stdout/stderr/status artifacts live;
- what throughput it achieved after completion.

This document records the intended dashboard direction. It is not implemented
yet.

## Data Sources

Use three layers, in this order:

1. **LSF state** from the login node:
   - `bjobs` for live state, queue, job id, job name, host allocation, runtime;
   - `bhist -l` for completed jobs and resource summaries.
2. **Workflow/job artifacts** under `/groups/.../logs/...`:
   - submission manifests;
   - stdout/stderr;
   - status JSON;
   - progress JSONL where available;
   - cache publish manifests and copy timing.
3. **Palette registry/Zarr state**:
   - `recording_step_status.details_json`;
   - stage performance views such as `recording_keypoint_performance_latest`;
   - run attrs/provenance for final durable metadata.

LSF alone is insufficient. It can report `RUN` on `h08u14`, but it cannot say
whether the job is copying a flat cache, loading a model, processing ROI batch
400/535, writing output arrays, or stuck.

## Target Tools

Near-term terminal tools:

```text
scripts/py -m fisheye.utils.cluster_job_watch ...
scripts/py -m fisheye.utils.cluster_job_summary ...
```

`cluster_job_watch` should be a polling Rich terminal dashboard. It should use
the configured login-node SSH alias for LSF state and read `/groups` logs
directly from the workstation.

`cluster_job_summary` should produce a non-live table/report for recent jobs,
combining LSF state, job artifacts, registry status, and throughput.

A web dashboard can come later if the terminal dashboard proves useful. The
first implementation should not require a service or database.

## Progress Contract

Do not make dashboards depend primarily on scraping human stdout. Stdout is a
fallback only.

Each long-running stage should emit a structured progress stream:

```text
<run_dir>/<stage>.<jobid>.progress.jsonl
```

Recommended event fields:

- `event`: `started`, `cache_staging`, `model_load`, `inference_batch`,
  `output_write`, `complete`, `error`;
- `timestamp_utc`;
- `job_id`;
- `job_index`;
- `hostname`;
- `scheduler_hosts`;
- `stage`;
- `recording_id`;
- `zarr_path`;
- `run_name`;
- `phase`;
- `items_done`;
- `items_total`;
- `elapsed_seconds`;
- `rate_items_per_second`;
- phase-specific payload, e.g. cache bytes copied or manifest path.

Existing flat-cache builders already emit useful progress JSONL. Keypoint and
mask inference should add per-batch JSONL events so a dashboard can show real
progress while models run.

## Display Shape

A practical terminal dashboard should group by workflow/run directory:

| Job | State | Stage | Recording | Host | Phase | Progress | Rate | Runtime | Logs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `151421477` | `DONE` | `keypoints` | `...arena_3...` | `h08u14` | `complete` | `136777/136777` | `275.8/s` | `9.6m` | stdout/stderr |

For completed jobs, show summary metrics:

- exit state;
- output run name;
- total items;
- success/failure counts;
- throughput;
- cache source tier;
- staged versus direct cache;
- copy time and copy throughput;
- scheduler host/GPU allocation.

For failed jobs, show:

- first and last stderr lines;
- last progress event;
- stage/phase at failure;
- output paths for diagnostics.

## First Implementation Slice

1. Define a shared progress JSONL helper.
2. Add keypoint per-batch progress events:
   - startup metadata;
   - cache staging event when applicable;
   - model load event;
   - one event every N batches;
   - completion summary.
3. Add `cluster_job_summary` for a log directory:
   - parse submission manifests;
   - query LSF for job state/history via `ssh login1-citrus-poller`;
   - read latest progress JSONL event;
   - join registry performance rows when available.
4. Add `cluster_job_watch` as a polling Rich table over one or more run dirs.

## Non-Goals

- Do not build a web server first.
- Do not require a central database before the JSONL/log-dir contract is stable.
- Do not make scheduler state the source of truth for scientific output; Zarr
  run attrs/provenance and registry projections remain the durable output state.
