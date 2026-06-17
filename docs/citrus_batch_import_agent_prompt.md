# Citrus Batch Import Submit Workflow

## Current Status

The workstation cron poller detects completed Citrus transfers and submits one
LSF job through `login1-citrus-poller`. The submitted job now runs a
conservative import-only workflow for one completed session:

1. organize the session into the recordings store;
2. create/update analysis Zarrs from the organizer JSONL log;
3. scan imported or skipped-existing analysis Zarrs into a registry when
   explicitly requested.

The job does **not** run detect, refine, crops, keypoints, or masks.

## Existing Pieces

Workstation poller:

```text
/home/delahantyj@hhmi.org/bin/citrus_staging_marker_poller.sh
```

The cron entry for this poller belongs on the local desktop/workstation only.
Do not run polling cron jobs on the login node. The login node is used only as
an SSH-accessible LSF submit host after the workstation poller has validated and
claimed a completion marker.

Production staging root:

```text
/groups/johnson/johnsonlab/jeremy/staging
```

Citrus completion marker:

```text
_citrus_transfer_complete.json
```

Marker validation requires:

```text
schema_id = citrus.transfer_completion_marker.v1
status = transfer_complete
local_target = true
verify_mode = quick
dest_dir = <absolute session directory path>
```

The poller normalizes `dest_dir` and requires it to match the marker parent
directory exactly.

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
- generated LSF job runs
  `fisheye.utils.run_citrus_session_import --apply` by default
- default destination root is
  `/groups/johnson/johnsonlab/jeremy/recordings` unless `--dest-root`
  overrides it
- staging cleanup is intentionally not enabled by this wrapper

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

## Implemented Job Payload

The wrapper calls a repo-managed helper:

```text
src/fisheye/utils/run_citrus_session_import.py
```

The helper runs:

```bash
scripts/py -m fisheye.utils.organize_recordings \
  "$SESSION_DIR" \
  --dest-root "$DEST_ROOT" \
  --log-dir "$RUN_DIR/organize_recordings" \
  --recursive \
  --write-manifest \
  --apply \
  --rename-cams
```

This organize step writes `recording_manifest.json` with
`preflight.status="not_run"` unless the wrapper is extended to pass
`--run-video-diagnostics` and/or `--run-h5-diagnostics`; the downstream import
gate only blocks stored `preflight.status="fail"`.

then:

```bash
scripts/py -m fisheye.utils.import_organized_recordings_analysis \
  --organize-log "$ORGANIZE_LOG" \
  --log-dir "$RUN_DIR/import_organized_recordings_analysis" \
  --apply
```

The helper parses the organizer JSONL before running import. If the organizer
log has no `recording_applied` entries, it skips import/registry rather than
letting `import_organized_recordings_analysis` fall back to scanning a broad
recordings root.

Optional registry refresh is explicit:

```bash
scripts/submit_citrus_session_import_bsub.sh \
  --session-dir "$SESSION_DIR" \
  --register \
  --registry /path/to/palette_registry.sqlite
```

If enabled, the submitted job passes `--registry` to
`fisheye.utils.import_organized_recordings_analysis`. The import wrapper scans
successful imports and skipped existing analysis Zarrs before reporting each
recording complete.

Important policy choices:

- `--dry-run` on the submit wrapper does not submit an LSF job.
- poller `--dry-run` validates and logs but does not create `.claimed`,
  `.submitted`, `.failed`, or lock files under `.processing_state`.
- `--job-dry-run` submits an LSF job that runs the organize/import planners
  without writes.
- Cleanup flags are not passed to `organize_recordings`; transfer session
  directories may be left partially empty after files are moved, but the wrapper
  does not remove staging directories.
- Registry scanning is skipped unless `--register --registry ...` is passed,
  because the cluster-visible registry path is deployment-specific.

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

To submit a cluster planning job instead of a local wrapper dry-run:

```bash
scripts/submit_citrus_session_import_bsub.sh \
  --session-dir "$SESSION_DIR" \
  --marker-key smoke \
  --log-dir /tmp/citrus-poller-wrapper-smoke \
  --job-dry-run
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

Production apply is the default poller behavior now. Before a production
transfer, confirm:

- the remote checkout at
  `/groups/johnson/johnsonlab/jeremy/gitrepos/palette` is on the desired
  branch/commit;
- the desired destination root is
  `/groups/johnson/johnsonlab/jeremy/recordings` or is provided via
  `--dest-root`;
- registry refresh is either intentionally skipped or explicitly configured via
  `--register --registry ...`;
- staging cleanup remains disabled unless a later operator-approved workflow
  changes that policy.

## Related Handoff

Before changing the login-node checkout state, read:

```text
docs/diagnostics/login_node_palette_branch_divergence_2026-06-16.md
```
