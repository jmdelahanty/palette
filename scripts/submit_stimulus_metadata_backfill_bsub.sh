#!/usr/bin/env bash
set -euo pipefail
umask 0002

REGISTRY="${PALETTE_REGISTRY_PATH:-/groups/johnson/johnsonlab/jeremy/registries/palette_registry.sqlite}"
OUTPUT_ROOT="${PALETTE_REGISTRY_AUDIT_ROOT:-/groups/johnson/johnsonlab/jeremy/registries/audits/stimulus_metadata}"
PALETTE_REPO="${PALETTE_GROUPS_REPO:-/groups/johnson/johnsonlab/jeremy/gitrepos/palette}"
SUBMIT_HOST="${PALETTE_LSF_SUBMIT_HOST:-login1-citrus-poller}"
RUN_ID=""
QUEUE=""
MEM_GB=8
WALLTIME="1:00"
APPLY=0
ALLOW_ISSUES=0
ALL_RECORDINGS=0
LIMIT=""
SUBMIT=0
RECORDING_IDS=()

usage() {
  cat <<'USAGE'
Usage: submit_stimulus_metadata_backfill_bsub.sh --run-id ID [scope] [options]

Render or submit one sequential CPU LSF job that reads recording analysis Zarr
metadata and writes a census. With --apply it creates a SQLite backup, then
replaces only normalized stimulus registry tables. No Zarr is opened writable.

Scope (one required):
  --all-recordings            All active recording-owned analysis datasets
  --recording-id ID           Repeatable recording scope

Options:
  --run-id ID                 Unique census/apply run identifier
  --registry PATH             Shared registry path
  --output-root PATH          Reports, backups, logs, and submission metadata
  --palette-repo PATH         Cluster-visible Palette checkout
  --submit-host HOST          SSH submission host if bsub is unavailable locally
  --limit N                   Deterministic recording limit for a canary
  --apply                     Back up and replace normalized stimulus tables
  --allow-issues              Required to apply a census containing issues
  --queue NAME
  --mem-gb N                  Default: 8
  --walltime H:MM             Default: 1:00
  --submit                    Submit; otherwise render only
  -h, --help
USAGE
}

fail() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-id) RUN_ID="$2"; shift 2;;
    --registry) REGISTRY="$2"; shift 2;;
    --output-root) OUTPUT_ROOT="$2"; shift 2;;
    --palette-repo) PALETTE_REPO="$2"; shift 2;;
    --submit-host) SUBMIT_HOST="$2"; shift 2;;
    --all-recordings) ALL_RECORDINGS=1; shift;;
    --recording-id) RECORDING_IDS+=("$2"); shift 2;;
    --limit) LIMIT="$2"; shift 2;;
    --apply) APPLY=1; shift;;
    --allow-issues) ALLOW_ISSUES=1; shift;;
    --queue) QUEUE="$2"; shift 2;;
    --mem-gb) MEM_GB="$2"; shift 2;;
    --walltime) WALLTIME="$2"; shift 2;;
    --submit) SUBMIT=1; shift;;
    -h|--help) usage; exit 0;;
    *) fail "unknown argument: $1";;
  esac
done

[[ -n "$RUN_ID" ]] || fail "--run-id is required"
[[ "$RUN_ID" =~ ^[A-Za-z0-9._-]+$ ]] || fail "unsafe --run-id: $RUN_ID"
if [[ "$ALL_RECORDINGS" == "1" && "${#RECORDING_IDS[@]}" -gt 0 ]]; then
  fail "do not combine --all-recordings with --recording-id"
fi
if [[ "$ALL_RECORDINGS" != "1" && "${#RECORDING_IDS[@]}" -eq 0 ]]; then
  fail "provide --all-recordings or at least one --recording-id"
fi
[[ "$MEM_GB" =~ ^[1-9][0-9]*$ ]] || fail "--mem-gb must be positive"
if [[ -n "$LIMIT" ]]; then
  [[ "$LIMIT" =~ ^[1-9][0-9]*$ ]] || fail "--limit must be positive"
fi
[[ -f "$REGISTRY" ]] || fail "registry not found: $REGISTRY"
[[ -d "$PALETTE_REPO/.git" ]] || fail "Palette checkout not found: $PALETTE_REPO"
[[ -x "$PALETTE_REPO/scripts/py" ]] || fail "Palette scripts/py is not executable"

RUN_DIR="${OUTPUT_ROOT}/bsub_${RUN_ID}"
REPORT="${OUTPUT_ROOT}/${RUN_ID}.json"
BACKUP="${OUTPUT_ROOT}/${RUN_ID}.registry_backup.sqlite"
[[ ! -e "$RUN_DIR" ]] || fail "run directory already exists: $RUN_DIR"
[[ ! -e "$REPORT" ]] || fail "report already exists: $REPORT"
if [[ "$APPLY" == "1" ]]; then
  [[ ! -e "$BACKUP" ]] || fail "backup already exists: $BACKUP"
fi
mkdir -p "$RUN_DIR"

EXPECTED_COMMIT="$(git -C "$PALETTE_REPO" rev-parse HEAD)"
JOB_SCRIPT="${RUN_DIR}/run_stimulus_metadata_backfill.sh"
STATUS_FILE="${RUN_DIR}/status.txt"
SUBMISSION_FILE="${RUN_DIR}/submission.txt"

q_repo="$(printf '%q' "$PALETTE_REPO")"
q_registry="$(printf '%q' "$REGISTRY")"
q_report="$(printf '%q' "$REPORT")"
q_backup="$(printf '%q' "$BACKUP")"
q_commit="$(printf '%q' "$EXPECTED_COMMIT")"
q_status="$(printf '%q' "$STATUS_FILE")"

cat >"$JOB_SCRIPT" <<JOBSCRIPT
#!/usr/bin/env bash
set -euo pipefail
umask 0002

PALETTE_REPO=${q_repo}
REGISTRY=${q_registry}
REPORT=${q_report}
BACKUP=${q_backup}
EXPECTED_COMMIT=${q_commit}
STATUS_FILE=${q_status}

cd "\${PALETTE_REPO}"
ACTUAL_COMMIT="\$(git rev-parse HEAD)"
if [[ "\${ACTUAL_COMMIT}" != "\${EXPECTED_COMMIT}" ]]; then
  printf 'Palette commit mismatch: expected %s, found %s\n' \
    "\${EXPECTED_COMMIT}" "\${ACTUAL_COMMIT}" >&2
  exit 2
fi

cmd=(
  scripts/py -m fisheye.registry.stimulus_metadata_backfill
  --registry "\${REGISTRY}"
  --output "\${REPORT}"
)
JOBSCRIPT

if [[ "$ALL_RECORDINGS" == "1" ]]; then
  printf 'cmd+=(--all-recordings)\n' >>"$JOB_SCRIPT"
else
  for recording_id in "${RECORDING_IDS[@]}"; do
    printf 'cmd+=(--recording-id %q)\n' "$recording_id" >>"$JOB_SCRIPT"
  done
fi
if [[ -n "$LIMIT" ]]; then
  printf 'cmd+=(--limit %q)\n' "$LIMIT" >>"$JOB_SCRIPT"
fi
if [[ "$APPLY" == "1" ]]; then
  printf 'cmd+=(--apply --backup "$BACKUP")\n' >>"$JOB_SCRIPT"
fi
if [[ "$ALLOW_ISSUES" == "1" ]]; then
  printf 'cmd+=(--allow-issues)\n' >>"$JOB_SCRIPT"
fi

cat >>"$JOB_SCRIPT" <<'JOBSCRIPT'
printf 'command='; printf '%q ' "${cmd[@]}"; printf '\n'
"${cmd[@]}"

{
  printf 'status=complete\n'
  printf 'completed_at_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'host=%s\n' "$(hostname)"
  printf 'job_id=%s\n' "${LSB_JOBID:-manual}"
  printf 'palette_commit=%s\n' "${ACTUAL_COMMIT}"
  printf 'report=%s\n' "${REPORT}"
} >"${STATUS_FILE}"
JOBSCRIPT
chmod +x "$JOB_SCRIPT"

BSUB_ARGS=(
  -J "stimulus_metadata_${RUN_ID}"
  -n 1
  -W "$WALLTIME"
  -R "rusage[mem=${MEM_GB}G]"
  -oo "${RUN_DIR}/%J.out"
  -eo "${RUN_DIR}/%J.err"
)
if [[ -n "$QUEUE" ]]; then BSUB_ARGS+=(-q "$QUEUE"); fi
BSUB_COMMAND=(bsub "${BSUB_ARGS[@]}" bash "$JOB_SCRIPT")

printf 'mode=%s\n' "$([[ "$SUBMIT" == "1" ]] && printf submit || printf render-only)"
printf 'operation=%s\n' "$([[ "$APPLY" == "1" ]] && printf apply || printf census)"
printf 'palette_commit=%s\n' "$EXPECTED_COMMIT"
printf 'report=%s\n' "$REPORT"
printf 'backup=%s\n' "$([[ "$APPLY" == "1" ]] && printf '%s' "$BACKUP" || printf none)"
printf 'job_script=%s\n' "$JOB_SCRIPT"
printf 'bsub_command='; printf '%q ' "${BSUB_COMMAND[@]}"; printf '\n'

if [[ "$SUBMIT" == "1" ]]; then
  if command -v bsub >/dev/null 2>&1; then
    submit_mode="local_bsub"
    submit_output="$("${BSUB_COMMAND[@]}")"
  else
    [[ -n "$SUBMIT_HOST" ]] || fail "bsub unavailable and --submit-host is empty"
    submit_mode="ssh_bsub"
    printf -v remote_command '%q ' "${BSUB_COMMAND[@]}"
    submit_output="$(ssh "$SUBMIT_HOST" "$remote_command")"
  fi
  printf '%s\n' "$submit_output"
  job_id="$(printf '%s\n' "$submit_output" | sed -n 's/^Job <\([0-9][0-9]*\)>.*/\1/p' | head -n 1)"
  [[ -n "$job_id" ]] || fail "could not parse LSF job ID"
  {
    printf 'submitted_at_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'submit_mode=%s\n' "$submit_mode"
    printf 'submit_host=%s\n' "$SUBMIT_HOST"
    printf 'job_id=%s\n' "$job_id"
    printf 'palette_commit=%s\n' "$EXPECTED_COMMIT"
    printf 'report=%s\n' "$REPORT"
  } >"$SUBMISSION_FILE"
  printf 'job_id=%s\n' "$job_id"
  printf 'submission_file=%s\n' "$SUBMISSION_FILE"
fi
