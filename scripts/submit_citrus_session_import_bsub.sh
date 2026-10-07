#!/usr/bin/env bash
set -euo pipefail

SESSION_DIR=""
MARKER_KEY=""
LOG_DIR=""
QUEUE="short"
NCORES=1
MEM_GB=4
WALLTIME="1:00"
RUN_ID=""
DRY_RUN=0
DEST_ROOT="/groups/johnson/johnsonlab/jeremy/recordings"
JOB_DRY_RUN=0
RESUME_TRANSFER_PLAN=""
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

usage() {
  cat <<'USAGE'
Usage: submit_citrus_session_import_bsub.sh --session-dir PATH [options]

Submit one LSF job that ingests a completed Citrus transfer through
transfer-v2 parent intake (the only ingest path; see
fisheye.utils.run_citrus_session_import and fisheye.intake.import_delivery).
The job never writes the registry: job-mode registration is retired, and the
writer host registers with `python -m fisheye.intake register-delivery`.

Options:
  --session-dir PATH             Completed Citrus session directory

Options:
  --marker-key HASH              Stable marker key from the poller
  --log-dir PATH                 Submission logs/job scripts
                                (default: <session-parent>/.processing_logs/bsub_submissions)
  --queue NAME                   LSF queue (default: short)
  --ncores N                     CPU slots (default: 1)
  --mem-gb N                     Memory request in GB (default: 4)
  --walltime H:MM                LSF wall time (default: 1:00)
  --run-id ID                    Stable run id (default: UTC timestamp)
  --dest-root PATH               Organized recordings root
                                (default: /groups/johnson/johnsonlab/jeremy/recordings)
  --job-dry-run                  Submit a cluster job that plans but does not
                                modify recordings/Zarrs
  --no-register                  Accepted for compatibility; the job never registers
  --resume-transfer-plan PATH    Exact saved organization plan for a retry
  --dry-run                      Print files and submit command; do not submit
  -h, --help                     Show this message
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --session-dir) SESSION_DIR="$2"; shift 2;;
    --marker-key) MARKER_KEY="$2"; shift 2;;
    --log-dir) LOG_DIR="$2"; shift 2;;
    --queue) QUEUE="$2"; shift 2;;
    --ncores) NCORES="$2"; shift 2;;
    --mem-gb) MEM_GB="$2"; shift 2;;
    --walltime) WALLTIME="$2"; shift 2;;
    --run-id) RUN_ID="$2"; shift 2;;
    --dest-root) DEST_ROOT="$2"; shift 2;;
    --job-dry-run) JOB_DRY_RUN=1; shift;;
    --no-register) shift;;
    --register|--registry|--writer-host)
      echo "$1: job-mode registration is retired; the job never writes the registry." >&2
      echo "Register on the writer host: python -m fisheye.intake register-delivery." >&2
      exit 2;;
    --resume-transfer-plan) RESUME_TRANSFER_PLAN="$2"; shift 2;;
    --dry-run) DRY_RUN=1; shift;;
    -h|--help) usage; exit 0;;
    --*) echo "Unknown arg: $1" >&2; usage; exit 2;;
    *)
      if [[ -z "$SESSION_DIR" ]]; then
        SESSION_DIR="$1"
        shift
      else
        echo "Unexpected positional arg: $1" >&2
        usage
        exit 2
      fi
      ;;
  esac
done

if [[ -z "$SESSION_DIR" ]]; then
  echo "Missing required --session-dir PATH" >&2
  usage
  exit 2
fi

if [[ "$DRY_RUN" != "1" && ! -d "$SESSION_DIR" ]]; then
  echo "Session directory not found: $SESSION_DIR" >&2
  exit 2
fi

SESSION_PARENT="$(dirname -- "$SESSION_DIR")"
SESSION_NAME="$(basename -- "$SESSION_DIR")"
if [[ -z "$RUN_ID" ]]; then
  RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
fi
if [[ -z "$MARKER_KEY" ]]; then
  if command -v sha256sum >/dev/null 2>&1; then
    MARKER_KEY="$(printf '%s' "$SESSION_DIR" | sha256sum | awk '{print $1}')"
  else
    MARKER_KEY="$RUN_ID"
  fi
fi
if [[ -z "$LOG_DIR" ]]; then
  LOG_DIR="${SESSION_PARENT}/.processing_logs/bsub_submissions"
fi

SAFE_SESSION_NAME="$(printf '%s' "$SESSION_NAME" | tr -c 'A-Za-z0-9_.-' '_')"
SAFE_RUN_ID="$(printf '%s' "$RUN_ID" | tr -c 'A-Za-z0-9_.-' '_')"
SAFE_MARKER_KEY="$(printf '%s' "$MARKER_KEY" | tr -c 'A-Za-z0-9_.-' '_')"
RUN_DIR="${LOG_DIR}/citrus_import_${SAFE_RUN_ID}_${SAFE_SESSION_NAME}_${SAFE_MARKER_KEY}"
# One delivery (marker key) is submitted at most once, however often the
# caller retries: a retry after an ambiguous failure (ssh dropped, bsub output
# unparseable after LSF accepted the job) finds the earlier job instead of
# submitting a duplicate import. To resubmit deliberately, remove the record.
JOB_NAME="citrus_import_${SAFE_MARKER_KEY}"
SUBMITTED_RECORD="${LOG_DIR}/by_marker/${SAFE_MARKER_KEY}.job"

record_submission() {
  mkdir -p "$(dirname -- "$SUBMITTED_RECORD")"
  local tmp="${SUBMITTED_RECORD}.$$"
  printf 'job_id=%s\njob_name=%s\nsession_dir=%s\nrecorded_utc=%s\n' \
    "$1" "$JOB_NAME" "$SESSION_DIR" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >"$tmp"
  mv -f -- "$tmp" "$SUBMITTED_RECORD"
}

if [[ -s "$SUBMITTED_RECORD" ]]; then
  echo "already_submitted=1"
  echo "submitted_record=$SUBMITTED_RECORD"
  cat -- "$SUBMITTED_RECORD"
  exit 0
fi
if [[ "$DRY_RUN" != "1" ]] && command -v bjobs >/dev/null 2>&1; then
  # Every job LSF still holds is the existing submission EXCEPT a finished one:
  # PEND, RUN, the SUSP states, UNKWN, PROV, WAIT, ZOMBI and any state this
  # script does not know all count as live. bjobs -a also lists finished
  # DONE/EXIT jobs for LSF's clean period; those are no reason to skip, and
  # resubmitting is safe because import_delivery is idempotent and resumes
  # from durable state. The by_marker record above stays the authoritative
  # "already submitted" record. If bjobs fails or its output cannot be
  # parsed, exit 1 (the poller retries) rather than risk a duplicate import.
  bjobs_stderr="$(mktemp)"
  set +e
  bjobs_output="$(bjobs -a -J "$JOB_NAME" -o "jobid stat" -noheader 2>"$bjobs_stderr")"
  bjobs_rc=$?
  set -e
  bjobs_errors="$(cat -- "$bjobs_stderr")"
  rm -f -- "$bjobs_stderr"
  existing_job=""
  bjobs_none=0
  if grep -qiE 'is not found|no (unfinished )?job found' <<<"$bjobs_output"$'\n'"$bjobs_errors"; then
    bjobs_none=1  # LSF's "no such job" answer (exit 0 or 255 by version)
  elif [[ "$bjobs_rc" -ne 0 ]]; then
    echo "bjobs failed (rc=$bjobs_rc); not submitting: $bjobs_output $bjobs_errors" >&2
    exit 1
  fi
  if [[ "$bjobs_none" -eq 0 ]]; then
    while IFS= read -r line; do
      [[ -z "${line// /}" ]] && continue
      if [[ ! "$line" =~ ^[[:space:]]*([0-9]+)[[:space:]]+([A-Z]+)[[:space:]]*$ ]]; then
        echo "unparseable bjobs output; not submitting: $line" >&2
        exit 1
      fi
      job="${BASH_REMATCH[1]}"
      state="${BASH_REMATCH[2]}"
      if [[ "$state" != "DONE" && "$state" != "EXIT" && -z "$existing_job" ]]; then
        existing_job="$job"
      fi
    done <<<"$bjobs_output"
  fi
  if [[ "$existing_job" =~ ^[0-9]+$ ]]; then
    record_submission "$existing_job"
    echo "already_submitted=1"
    echo "submitted_record=$SUBMITTED_RECORD"
    echo "job_id=$existing_job"
    exit 0
  fi
fi

if [[ -e "$RUN_DIR" ]]; then
  echo "Run directory already exists: $RUN_DIR" >&2
  echo "Choose a different --run-id or --log-dir." >&2
  exit 2
fi
mkdir -p "$RUN_DIR"

JOB_SCRIPT="${RUN_DIR}/run_citrus_session_import.sh"
JOB_STATUS_TEMPLATE="${RUN_DIR}/${SAFE_SESSION_NAME}.JOBID.status.txt"

quoted_session_dir="$(printf '%q' "$SESSION_DIR")"
quoted_session_name="$(printf '%q' "$SESSION_NAME")"
quoted_run_dir="$(printf '%q' "$RUN_DIR")"
quoted_dest_root="$(printf '%q' "$DEST_ROOT")"
quoted_repo_root="$(printf '%q' "$REPO_ROOT")"
quoted_resume_plan="$(printf '%q' "$RESUME_TRANSFER_PLAN")"

cat >"$JOB_SCRIPT" <<JOBSCRIPT
#!/usr/bin/env bash
set -euo pipefail

SESSION_DIR=${quoted_session_dir}
SESSION_NAME=${quoted_session_name}
RUN_DIR=${quoted_run_dir}
DEST_ROOT=${quoted_dest_root}
REPO_ROOT=${quoted_repo_root}
JOB_DRY_RUN=${JOB_DRY_RUN}
RESUME_TRANSFER_PLAN=${quoted_resume_plan}
JOB_ID="\${LSB_JOBID:-manual}"
# An LSF requeue reuses LSB_JOBID, and the workflow refuses an existing run
# directory: every attempt gets its own workflow directory and payload logs.
ATTEMPT="\${LSB_JOBINDEX:-0}-\$(date -u +%Y%m%dT%H%M%S%NZ)-\$\$"
STATUS_FILE="\${RUN_DIR}/${SAFE_SESSION_NAME}.\${JOB_ID}.status.txt"
# The workflow creates and owns its run directory (it refuses an existing one),
# only after taking the delivery's lock, and writes its status at the standard
# name inside it. A job that finds the delivery held exits 75 and creates none.
WORKFLOW_DIR="\${RUN_DIR}/workflow-\${JOB_ID}-\${ATTEMPT}"
STATUS_JSON="\${WORKFLOW_DIR}/citrus_session_import.status.json"
PAYLOAD_STDOUT="\${RUN_DIR}/${SAFE_SESSION_NAME}.\${JOB_ID}.\${ATTEMPT}.payload.out"
PAYLOAD_STDERR="\${RUN_DIR}/${SAFE_SESSION_NAME}.\${JOB_ID}.\${ATTEMPT}.payload.err"

mkdir -p "\${RUN_DIR}"

cmd=(
  "\${REPO_ROOT}/scripts/py"
  -m
  fisheye.utils.run_citrus_session_import
  "\${SESSION_DIR}"
  --dest-root "\${DEST_ROOT}"
  --run-dir "\${WORKFLOW_DIR}"
)
if [[ "\${JOB_DRY_RUN}" == "1" ]]; then
  cmd+=(--dry-run)
else
  cmd+=(--apply)
fi
if [[ -n "\${RESUME_TRANSFER_PLAN}" ]]; then
  cmd+=(--resume-transfer-plan "\${RESUME_TRANSFER_PLAN}")
fi

printf 'payload_command='
printf '%q ' "\${cmd[@]}"
printf '\n'

set +e
"\${cmd[@]}" >"\${PAYLOAD_STDOUT}" 2>"\${PAYLOAD_STDERR}"
payload_rc=\$?
set -e

{
  printf 'started_at=%s\n' "\$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'host=%s\n' "\$(hostname)"
  printf 'job_id=%s\n' "\${JOB_ID}"
  printf 'attempt=%s\n' "\${ATTEMPT}"
  printf 'session_name=%s\n' "\${SESSION_NAME}"
  printf 'session_dir=%s\n' "\${SESSION_DIR}"
  printf 'dest_root=%s\n' "\${DEST_ROOT}"
  printf 'action=%s\n' 'citrus_session_import'
  printf 'payload_returncode=%s\n' "\${payload_rc}"
  printf 'payload_stdout=%s\n' "\${PAYLOAD_STDOUT}"
  printf 'payload_stderr=%s\n' "\${PAYLOAD_STDERR}"
  printf 'status_json=%s\n' "\${STATUS_JSON}"
} >"\${STATUS_FILE}"

printf 'payload_returncode=%s\n' "\${payload_rc}"
printf 'payload_stdout=%s\n' "\${PAYLOAD_STDOUT}"
printf 'payload_stderr=%s\n' "\${PAYLOAD_STDERR}"
printf 'status_file=%s\n' "\${STATUS_FILE}"
printf 'status_json=%s\n' "\${STATUS_JSON}"
exit "\${payload_rc}"
JOBSCRIPT
chmod +x "$JOB_SCRIPT"

BSUB_ARGS=(
  -J "$JOB_NAME"
  -n "$NCORES"
  -W "$WALLTIME"
  -R "rusage[mem=${MEM_GB}G]"
  -oo "${RUN_DIR}/%J.out"
  -eo "${RUN_DIR}/%J.err"
)
if [[ -n "$QUEUE" ]]; then
  BSUB_ARGS+=(-q "$QUEUE")
fi

printf -v BSUB_ARGS_SHELL '%q ' "${BSUB_ARGS[@]}"
BSUB_CMD="bsub ${BSUB_ARGS_SHELL}bash $(printf '%q' "$JOB_SCRIPT")"

echo "session_dir=$SESSION_DIR"
echo "session_name=$SESSION_NAME"
echo "run_dir=$RUN_DIR"
echo "job_script=$JOB_SCRIPT"
echo "expected_status=$JOB_STATUS_TEMPLATE"
echo "dest_root=$DEST_ROOT"
echo "job_dry_run=$JOB_DRY_RUN"
echo "registration=writer_host_only"
echo "submit_command=$BSUB_CMD"

if [[ "$DRY_RUN" == "1" ]]; then
  echo "dry_run=1"
  exit 0
fi

if ! command -v bsub >/dev/null 2>&1; then
  echo "bsub not found in PATH. Is this an LSF login/submit host?" >&2
  exit 2
fi

submit_output="$(bsub "${BSUB_ARGS[@]}" bash "$JOB_SCRIPT")"
echo "$submit_output"
job_id="$(printf '%s\n' "$submit_output" | sed -n 's/^Job <\([0-9][0-9]*\)>.*/\1/p' | head -n 1)"
if [[ -z "$job_id" ]]; then
  echo "Could not parse job id from bsub output." >&2
  exit 1
fi
record_submission "$job_id"
echo "job_id=$job_id"
echo "lsf_stdout=${RUN_DIR}/${job_id}.out"
echo "lsf_stderr=${RUN_DIR}/${job_id}.err"
echo "status_file=${RUN_DIR}/${SAFE_SESSION_NAME}.${job_id}.status.txt"
