# Agent Prompt: Implement Citrus Batch Import Submit Tool

## Goal

Implement the real batch import tool used by the Citrus staging marker poller.
The workstation cron poller already detects completed Citrus transfers and
submits a placeholder LSF job through `login1-citrus-poller`. Replace that
placeholder with a conservative import/registry workflow that can run one
completed session at a time.

## Existing Pieces

Workstation poller:

```text
/home/delahantyj@hhmi.org/bin/citrus_staging_marker_poller.sh
```

Production staging root:

```text
/groups/johnson/johnsonlab/jeremy/staging
```

Citrus completion marker:

```text
_citrus_transfer_complete.json
```

Marker validation already requires:

```text
schema_id = citrus.transfer_completion_marker.v1
status = transfer_complete
```

State/log directories:

```text
/groups/johnson/johnsonlab/jeremy/staging/.processing_state
/groups/johnson/johnsonlab/jeremy/staging/.processing_logs
```

Remote submit host:

```text
login1-citrus-poller
```

Remote Palette checkout used by the poller:

```text
/groups/johnson/johnsonlab/jeremy/gitrepos/palette
```

Current remote submit wrapper:

```text
scripts/submit_citrus_session_import_bsub.sh
```

Current wrapper behavior:

- accepts `--session-dir`
- submits one LSF job through `bsub`
- writes LSF stdout/stderr under `.processing_logs/bsub_submissions`
- prints parseable `job_id=...`, `lsf_stdout=...`, `lsf_stderr=...`,
  and `status_file=...`
- generated LSF job currently only logs `would process <session_dir>`

## Required Design

Keep the poller short-lived:

1. Poller validates marker.
2. Poller creates per-marker `.claimed`.
3. Poller SSHes to `login1-citrus-poller`.
4. Login-node wrapper submits LSF job.
5. Poller writes `.submitted` with job id and LSF log paths.
6. Long-running import work happens only inside LSF, not inside the poller.

Do not rely on inotify or file events. Polling is intentional because the
staging path is on shared/network storage.

## Exact-Once / Failure Policy

Preserve conservative exactly-once dispatch semantics:

- If marker validation fails, do not claim or submit.
- If remote submission succeeds, write `.submitted`.
- If remote submission fails after `.claimed`, do not blindly resubmit if
  success is ambiguous. Prefer leaving `.claimed` plus a failure/unknown state
  for operator review.
- Do not delete or modify transferred session data.
- Small state/log/job-script files under `.processing_state` and
  `.processing_logs` are acceptable.

## Implementation Target

Update `scripts/submit_citrus_session_import_bsub.sh` so the generated LSF job
runs the real import flow instead of the current placeholder.

The implementation should likely create or call a repo-managed helper rather
than embedding a large command directly in `bsub`. Follow existing LSF wrapper
style in scripts such as:

```text
scripts/submit_detect_batches_bsub.sh
scripts/submit_review_proxy_videos_bsub.sh
scripts/submit_flat_roi_cache_bsub.sh
scripts/submit_crop_flat_roi_cache_bsub.sh
```

## Candidate Existing Import/Organize Surfaces

Inspect these before implementing:

```text
docs/operator_guide/organize_recordings.md
docs/organize_recordings_logging_schema.md
docs/staging_recording_only_review_2026-05-11.md
docs/analysis_zarr_creation_todo.md
docs/recording_analysis_pipeline_contract.md
scripts/organize_staging.sh
src/fisheye/utils/organize_recordings.py
src/fisheye/utils/import_recording_analysis.py
src/fisheye/utils/import_recordings_analysis.py
src/fisheye/utils/import_organized_recordings_analysis.py
src/fisheye/utils/run_recording_analysis_pipeline.py
```

Important repo rule:

```text
Use scripts/py for Python commands. Do not use bare python or conda activate.
```

`scripts/organize_staging.sh` currently defaults to `PYTHON=python`, so if that
wrapper is used from LSF, call it with:

```bash
PYTHON=/groups/johnson/johnsonlab/jeremy/gitrepos/palette/scripts/py
```

or update the wrapper carefully.

## Open Questions To Resolve

1. Does a Citrus marker session directory correspond to one organizer batch
   root, or can it contain multiple recordings/arenas?
2. Should the import job organize files into `/nvme1/recordings`,
   `/groups/johnson/...`, or another canonical recordings root?
3. Should the job call `organize_recordings` only, or also create/import
   analysis Zarrs and update the registry?
4. What registry path should the job use? Current common path appears to be:

   ```text
   /nvme1/palette_registry.sqlite
   ```

   Confirm before writing.

5. Should the LSF job run on `short`, or should import/registry steps be split
   into dependent CPU/GPU jobs?
6. Should staging cleanup be enabled? The initial requirement says not to
   delete/modify transferred session data, so default should be no cleanup until
   explicitly approved.

## Suggested First Implementation Slice

Keep the first real slice CPU-only and reversible:

1. Add `--apply` / `--dry-run` behavior to the submit wrapper if needed.
2. In the generated LSF job:
   - record environment, host, job id, repo commit, session dir, marker key
   - run a dry-run organization/import plan first and save it under the run dir
   - if configured for apply, run the apply command
   - write a final status JSON/TXT with `ok`, command, outputs, and error info
3. Do not enable staging cleanup.
4. Do not run GPU detection/refinement as part of the first import submitter
   unless the owner explicitly asks for it.

## Validation

Use focused validation before touching production markers:

```bash
bash -n scripts/submit_citrus_session_import_bsub.sh
scripts/submit_citrus_session_import_bsub.sh --session-dir "/tmp/session with spaces" --marker-key dryrun --log-dir /tmp/citrus-poller-wrapper-dryrun --dry-run
```

Then validate from the workstation through SSH:

```bash
ssh login1-citrus-poller '
  cd /groups/johnson/johnsonlab/jeremy/gitrepos/palette &&
  scripts/submit_citrus_session_import_bsub.sh \
    --session-dir "/tmp/session with spaces" \
    --marker-key dryrun \
    --log-dir /tmp/citrus-poller-wrapper-dryrun \
    --dry-run
'
```

For an end-to-end dispatch smoke, create a fake marker under a shared scratch
path visible to both workstation and login node, not under production staging.
The previous successful smoke used:

```text
/groups/ahrens/home/delahantyj/citrus_poller_smoke_<timestamp>
```

The placeholder job submitted through LSF and completed with:

```text
Job ID: 151313147
Status: DONE
Queue: short
Exec host: h07u20
```

Do not run production apply until the owner confirms the exact destination root,
registry path, and cleanup policy.

## Related Handoff

Before changing the login-node checkout state, read:

```text
docs/diagnostics/login_node_palette_branch_divergence_2026-06-16.md
```
