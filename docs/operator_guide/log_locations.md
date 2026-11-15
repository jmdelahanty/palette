# Where Palette's logs are

Every log location below is stable: code writes there by default, and readers
(the intake registrar, operators, the workflow runner) find results by these
paths. Short names used below:

- `J` = `/groups/johnson/johnsonlab/jeremy`
- `ws1` = the registry writer workstation (`delahantyj-ws1.hhmi.org`)

Logs say what happened. They are never the authority for what *is* true. That
lives in each analysis Zarr (import receipts, run provenance, completion
markers) and in the registry. When a log and a receipt disagree, the receipt
wins.

## At a glance

| Stage | Where | What one item's logs look like |
|---|---|---|
| Intake: discovery and submission (cron) | `ws1:~/.palette/logs/citrus_v2_poller.log` | one line per poll and per delivery submitted or refused |
| Intake: registration (cron) | `ws1:~/.palette/logs/citrus_v2_registrar.log` | one line per delivery: pending, attached, registered, refused, failed |
| Intake: per-delivery state | `J/staging/.processing_state_v2/` | `<key>.claimed`, `.submitted`, `.registered`, `.import_failed`, `.registration_failed`, `.registration_refused` |
| Intake: per-delivery LSF job | `J/staging/.processing_logs_v2/bsub_submissions/citrus_import_<UTC>_<session>_<key>/` | job script, `<job>.out`/`.err`, `<session>.<job>.status.txt`, `.payload.out`/`.err`, `workflow-<job>…/citrus_session_import.status.json` |
| Intake: launcher dedupe record | `J/staging/.processing_logs_v2/bsub_submissions/by_marker/<key>.job` | the LSF job id first submitted for a delivery |
| Intake: durable organizer journal | `J/recordings/.transfer_intake/<snapshot sha>/organization_state.json` | the organize → import → retire state machine for one delivery |
| Downstream stage jobs | `J/recordings/logs/<stage or launcher name>/` | one folder per launcher (e.g. `run_keypoints_batch`, `subject_mask_finalization_bsub`, `chaser_analytics`), usually with `bsub_submissions/` run folders inside |
| Older per-recording processing | `J/recordings/.processing_logs/`, `J/logs/` | older launchers; read-only history |
| One-off and named operations | `J/operations/<operation name>/` | one folder per deliberate operation (canaries, cohort materializations, successor runs), named by hand with a date |
| Registry changes | `J/registries/backups/` (pre-change copies), `J/registries/audits/` (reports) | one backup per registry publication, named for the operation |
| Workstation maintenance (cron) | `ws1:~/.palette/logs/` | `palette_registry_backup.log`, `labeling_store_backup.log`, `recording_step_status_smoke.log` |
| Retired v1 intake | `J/staging/.processing_logs/`, `J/staging/.processing_state/` | history of the removed v1 poller; nothing writes here now |

## Following one delivery through intake

A delivery is one sealed transfer folder in `J/staging/<experiment folder>/`.
Its **key** is the sha256 of its marker path and marker content. It appears in
every intake file name.

1. **Was it picked up?** `grep <experiment folder> ws1:~/.palette/logs/citrus_v2_poller.log`
   shows `submitting …` (with the full `bsub` command) or `refused marker=…`
   (with the reason). The key is in the `--marker-key` argument.
2. **What did it submit?** `J/staging/.processing_state_v2/<key>.submitted`
   holds the launcher's output, including `job_id=` and `run_dir=`. The last
   `job_id=` line is the job the registrar follows.
3. **What did the job do?** In the run folder
   `J/staging/.processing_logs_v2/bsub_submissions/citrus_import_*_<key>/`:
   - `<session>.<job>.status.txt`: host, job id and `payload_returncode`
     (0 done, 65 refused, 75 held by another attempt, 1 failed and retryable).
   - `<session>.<job>.payload.err`: the import's own stderr. Start here when a
     job failed.
   - `workflow-<job>…/citrus_session_import.status.json`: the structured
     result (status, per-parent results, Zarr paths, error).
   - `<job>.out`: LSF's own report, including run time and memory.
4. **Was it registered?** `J/staging/.processing_state_v2/<key>.registered`
   (with dataset ids), or `.registration_failed` (retried) /
   `.registration_refused` (terminal, needs an operator) /
   `.import_failed` (terminal). Each registrar run is also logged in
   `ws1:~/.palette/logs/citrus_v2_registrar.log`.
5. **Where did the data go?** One folder per camera under
   `J/recordings/<session>_Cam<serial>/`. Each has an analysis Zarr in `zarr/`
   and its import receipt inside the Zarr at `.imports/`. The organizer's
   journal for the whole delivery is
   `J/recordings/.transfer_intake/<snapshot sha>/organization_state.json`.

## Conventions for new stages

- Write a stage's cluster job logs under `J/recordings/logs/<stage name>/`,
  with one run folder per submission, and allow an environment-variable
  override of the root. Every current downstream launcher does this.
- Use `J/operations/<descriptive name>_<YYYYMMDD>/` only for deliberate,
  one-off operations that a person names and reviews later.
- Put an item's identity (session, camera, key or run id) in folder or file
  names, so `ls` and `grep` find it without opening files.
- Keep authoritative results in the Zarr or the registry, never only in a log.
