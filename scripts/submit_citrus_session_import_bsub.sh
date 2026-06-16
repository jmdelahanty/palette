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

usage() {
  cat <<'USAGE'
Usage: submit_citrus_session_import_bsub.sh --session-dir PATH [options]

Submit one LSF job for a completed Citrus transfer session.

The current job payload is intentionally conservative: it only logs
"would process <session_dir>". Replace the placeholder inside the generated job
script once the real import command is settled.

Required:
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

cat >"$JOB_SCRIPT" <<JOBSCRIPT
#!/usr/bin/env bash
set -euo pipefail

SESSION_DIR=${quoted_session_dir}
SESSION_NAME=${quoted_session_name}
RUN_DIR=${quoted_run_dir}
JOB_ID="\${LSB_JOBID:-manual}"
STATUS_FILE="\${RUN_DIR}/${SAFE_SESSION_NAME}.\${JOB_ID}.status.txt"

mkdir -p "\${RUN_DIR}"
{
  printf 'started_at=%s\n' "\$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'host=%s\n' "\$(hostname)"
  printf 'job_id=%s\n' "\${JOB_ID}"
  printf 'session_name=%s\n' "\${SESSION_NAME}"
  printf 'session_dir=%s\n' "\${SESSION_DIR}"
  printf 'action=%s\n' 'placeholder'
  printf 'message=%s\n' "would process \${SESSION_DIR}"
} >"\${STATUS_FILE}"

printf 'would process %s\n' "\${SESSION_DIR}"
printf 'status_file=%s\n' "\${STATUS_FILE}"
JOBSCRIPT
chmod +x "$JOB_SCRIPT"

BSUB_ARGS=(
  -J "citrus_import_${SAFE_SESSION_NAME}"
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
echo "job_id=$job_id"
echo "lsf_stdout=${RUN_DIR}/${job_id}.out"
echo "lsf_stderr=${RUN_DIR}/${job_id}.err"
echo "status_file=${RUN_DIR}/${SAFE_SESSION_NAME}.${job_id}.status.txt"
