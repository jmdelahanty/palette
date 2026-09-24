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
REGISTER=1
REGISTRY="${PALETTE_REGISTRY:-/groups/johnson/johnsonlab/jeremy/registries/palette_registry.sqlite}"
WRITER_HOST="${PALETTE_REGISTRY_WRITER_HOST:-}"
WRITER_LOCK_PATH="${PALETTE_REGISTRY_WRITER_LOCK_PATH:-/tmp/palette-registry-writer.lock}"
SHADOW_TEMP_ROOT="${PALETTE_REGISTRY_SHADOW_TEMP_ROOT:-/tmp/palette-registry-shadows}"
SHADOW_BACKUP_DIR="${PALETTE_REGISTRY_SHADOW_BACKUP_DIR:-}"
RECORDING_ONLY=0
RECORDING_TYPE=""
RECORDING_SUBTYPE=""
BEHAVIOR_MODE=""
RESUME_TRANSFER_PLAN=""
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

usage() {
  cat <<'USAGE'
Usage: submit_citrus_session_import_bsub.sh --session-dir PATH [options]

Submit one LSF job that ingests a completed Citrus transfer through
transfer-v2 parent intake (the only ingest path; see
fisheye.utils.run_citrus_session_import).

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
  --register                     Scan imported/skipped analysis Zarrs into registry (default)
  --no-register                  Do not scan imported/skipped analysis Zarrs
  --registry PATH                Registry SQLite path used with --register
                                (default: $PALETTE_REGISTRY or /groups/.../palette_registry.sqlite)
  --writer-host HOST             Designated registry writer host; required with --register
  --recording-only               Import camera-video-only recordings without stimulus
  --recording-type TYPE          Recording context (required unless resuming)
  --recording-subtype SUBTYPE    Recording subtype (required unless resuming)
  --behavior-mode MODE           free, embedded or none (required unless resuming)
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
    --register) REGISTER=1; shift;;
    --no-register) REGISTER=0; shift;;
    --registry) REGISTRY="$2"; shift 2;;
    --writer-host) WRITER_HOST="$2"; shift 2;;
    --recording-only) RECORDING_ONLY=1; shift;;
    --recording-type) RECORDING_TYPE="$2"; shift 2;;
    --recording-subtype) RECORDING_SUBTYPE="$2"; shift 2;;
    --behavior-mode) BEHAVIOR_MODE="$2"; shift 2;;
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

if [[ "$REGISTER" == "1" && -z "$REGISTRY" ]]; then
  echo "--register requires --registry PATH" >&2
  exit 2
fi
if [[ -z "$SHADOW_BACKUP_DIR" ]]; then
  SHADOW_BACKUP_DIR="$(dirname -- "$REGISTRY")/backups"
fi
if [[ "$REGISTER" == "1" && -z "$WRITER_HOST" ]]; then
  echo "--register requires --writer-host or PALETTE_REGISTRY_WRITER_HOST" >&2
  exit 2
fi
if [[ -n "$WRITER_HOST" && ! "$WRITER_HOST" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "Unsafe --writer-host value: $WRITER_HOST" >&2
  exit 2
fi

if [[ "$DRY_RUN" != "1" && ! -d "$SESSION_DIR" ]]; then
  echo "Session directory not found: $SESSION_DIR" >&2
  exit 2
fi

if [[ -z "$RESUME_TRANSFER_PLAN" && ( -z "$RECORDING_TYPE" || -z "$RECORDING_SUBTYPE" || -z "$BEHAVIOR_MODE" ) ]]; then
  echo "Transfer-v2 intake requires --recording-type, --recording-subtype and --behavior-mode (or --resume-transfer-plan)" >&2
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
quoted_dest_root="$(printf '%q' "$DEST_ROOT")"
quoted_repo_root="$(printf '%q' "$REPO_ROOT")"
quoted_registry="$(printf '%q' "$REGISTRY")"
quoted_writer_host="$(printf '%q' "$WRITER_HOST")"
quoted_writer_lock_path="$(printf '%q' "$WRITER_LOCK_PATH")"
quoted_shadow_temp_root="$(printf '%q' "$SHADOW_TEMP_ROOT")"
quoted_shadow_backup_dir="$(printf '%q' "$SHADOW_BACKUP_DIR")"
quoted_recording_type="$(printf '%q' "$RECORDING_TYPE")"
quoted_recording_subtype="$(printf '%q' "$RECORDING_SUBTYPE")"
quoted_behavior_mode="$(printf '%q' "$BEHAVIOR_MODE")"
quoted_resume_plan="$(printf '%q' "$RESUME_TRANSFER_PLAN")"

cat >"$JOB_SCRIPT" <<JOBSCRIPT
#!/usr/bin/env bash
set -euo pipefail

SESSION_DIR=${quoted_session_dir}
SESSION_NAME=${quoted_session_name}
RUN_DIR=${quoted_run_dir}
DEST_ROOT=${quoted_dest_root}
REPO_ROOT=${quoted_repo_root}
REGISTRY=${quoted_registry}
export PALETTE_REGISTRY_WRITER_HOST=${quoted_writer_host}
export PALETTE_REGISTRY_WRITER_LOCK_PATH=${quoted_writer_lock_path}
export PALETTE_REGISTRY_SHADOW_TEMP_ROOT=${quoted_shadow_temp_root}
export PALETTE_REGISTRY_SHADOW_BACKUP_DIR=${quoted_shadow_backup_dir}
JOB_DRY_RUN=${JOB_DRY_RUN}
REGISTER=${REGISTER}
RECORDING_ONLY=${RECORDING_ONLY}
RECORDING_TYPE=${quoted_recording_type}
RECORDING_SUBTYPE=${quoted_recording_subtype}
BEHAVIOR_MODE=${quoted_behavior_mode}
RESUME_TRANSFER_PLAN=${quoted_resume_plan}
JOB_ID="\${LSB_JOBID:-manual}"
STATUS_FILE="\${RUN_DIR}/${SAFE_SESSION_NAME}.\${JOB_ID}.status.txt"
STATUS_JSON="\${RUN_DIR}/${SAFE_SESSION_NAME}.\${JOB_ID}.status.json"
PAYLOAD_STDOUT="\${RUN_DIR}/${SAFE_SESSION_NAME}.\${JOB_ID}.payload.out"
PAYLOAD_STDERR="\${RUN_DIR}/${SAFE_SESSION_NAME}.\${JOB_ID}.payload.err"

mkdir -p "\${RUN_DIR}"

cmd=(
  "\${REPO_ROOT}/scripts/py"
  -m
  fisheye.utils.run_citrus_session_import
  "\${SESSION_DIR}"
  --dest-root "\${DEST_ROOT}"
  --run-dir "\${RUN_DIR}"
  --status-json "\${STATUS_JSON}"
)
if [[ "\${JOB_DRY_RUN}" == "1" ]]; then
  cmd+=(--dry-run)
else
  cmd+=(--apply)
fi
if [[ "\${REGISTER}" == "1" ]]; then
  cmd+=(--register --registry "\${REGISTRY}")
fi
if [[ "\${RECORDING_ONLY}" == "1" ]]; then
  cmd+=(--recording-only)
fi
if [[ -n "\${RESUME_TRANSFER_PLAN}" ]]; then
  cmd+=(--resume-transfer-plan "\${RESUME_TRANSFER_PLAN}")
else
  cmd+=(--recording-type "\${RECORDING_TYPE}" --recording-subtype "\${RECORDING_SUBTYPE}" --behavior-mode "\${BEHAVIOR_MODE}")
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
  -J "citrus_import_${SAFE_SESSION_NAME}"
  -n "$NCORES"
  -W "$WALLTIME"
  -R "rusage[mem=${MEM_GB}G]"
  -oo "${RUN_DIR}/%J.out"
  -eo "${RUN_DIR}/%J.err"
)
if [[ "$REGISTER" == "1" ]]; then
  BSUB_ARGS+=(-R "select[hname==${WRITER_HOST}] span[hosts=1]")
fi
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
echo "register=$REGISTER"
if [[ -n "$REGISTRY" ]]; then
  echo "registry=$REGISTRY"
fi
if [[ -n "$WRITER_HOST" ]]; then
  echo "registry_writer_host=$WRITER_HOST"
fi
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
