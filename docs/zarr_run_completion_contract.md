# Zarr Run Completion Contract

Status: active  
Last verified: 2026-05-20

Palette run parents such as `detect_runs`, `crop_runs`,
`refined_detect_runs`, `keypoints_runs`, and nested `quality_reports` must not
publish an incomplete run as the preferred input for downstream stages.

## Attributes

Run groups that opt into this contract set:

- `palette_run_completion_contract = "palette.zarr_run_completion.v1"`
- `palette_run_completion_status = "running" | "complete" | "failed"`
- `palette_run_started_at_utc`
- `palette_run_completed_at_utc` when complete
- `palette_run_name`
- `palette_run_stage`

Run parent groups may set:

- `latest`: backward-compatible pointer to the latest complete run.
- `latest_complete`: explicit latest complete run pointer.
- `latest_pending`: newest started run that is not complete yet.

New contract-aware writers should create the run group, immediately mark the
run `running`, and set `latest_pending`. They should not move `latest` to the
new run until the run is complete. On success, writers call
`mark_run_complete(..., parent_group=<parent>, run_name=<name>)`, which updates
both `latest_complete` and `latest`.

## Read Rule

Contract-aware readers should resolve default inputs with
`resolve_latest_complete_run_name(parent, legacy_default=True)`.

Legacy runs without completion attrs are treated as complete for compatibility.
Runs that opt into `palette.zarr_run_completion.v1` are considered usable only
when `palette_run_completion_status == "complete"`.

## Safety Check

Scan an archive for unsafe latest pointers:

```bash
scripts/py -m fisheye.utils.check_zarr_run_completion /path/to/archive.zarr --fail-on-unsafe
```

Useful outputs:

- `unsafe_parent_count > 0`: at least one run parent has `latest` or
  `latest_complete` pointing at a missing or incomplete opted-in run.
- `pending_parent_count > 0`: one or more parents have `latest_pending` or
  incomplete opted-in runs. This is acceptable while a job is running, but
  should be investigated if the job is finished.

## Current Limitation

This contract is attr-based and protects default resolution, registry
completion, and diagnostics. It does not yet implement physical atomic
`<run>.partial` to `<run>` renames. That future hardening would reduce the
chance of consumers seeing a partially-populated group by explicit path.
