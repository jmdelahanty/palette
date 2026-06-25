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
RECORDING_ONLY=0
ALLOW_PREFLIGHT_FAILURES=0
RUN_VIDEO_DIAGNOSTICS=1
RUN_H5_DIAGNOSTICS=1
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

usage() {
  cat <<'USAGE'
Usage: submit_citrus_session_import_bsub.sh --session-dir PATH [options]

Submit one LSF job for a completed Citrus transfer session.

The job payload is intentionally conservative: it organizes one completed
session into the recordings store, creates/imports analysis Zarrs, and
optionally scans those Zarrs into a registry during import. It does not run
detect/refine.

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
  --dest-root PATH               Organized recordings root
                                (default: /groups/johnson/johnsonlab/jeremy/recordings)
  --job-dry-run                  Submit a cluster job that plans but does not
                                modify recordings/Zarrs
  --register                     Scan imported/skipped analysis Zarrs into registry (default)
  --no-register                  Do not scan imported/skipped analysis Zarrs
  --registry PATH                Registry SQLite path used with --register
                                (default: $PALETTE_REGISTRY or /groups/.../palette_registry.sqlite)
  --recording-only               Import camera-video-only recordings without H5
  --allow-preflight-failures     Do not block import on manifest preflight fail
  --run-video-diagnostics        Persist video preflight diagnostics in manifests (default)
  --no-run-video-diagnostics     Skip video preflight diagnostics
  --run-h5-diagnostics           Persist H5 preflight diagnostics in manifests (default)
  --no-run-h5-diagnostics        Skip H5 preflight diagnostics
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
    --recording-only) RECORDING_ONLY=1; shift;;
    --allow-preflight-failures) ALLOW_PREFLIGHT_FAILURES=1; shift;;
    --run-video-diagnostics) RUN_VIDEO_DIAGNOSTICS=1; shift;;
    --no-run-video-diagnostics) RUN_VIDEO_DIAGNOSTICS=0; shift;;
    --run-h5-diagnostics) RUN_H5_DIAGNOSTICS=1; shift;;
    --no-run-h5-diagnostics) RUN_H5_DIAGNOSTICS=0; shift;;
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
quoted_dest_root="$(printf '%q' "$DEST_ROOT")"
quoted_repo_root="$(printf '%q' "$REPO_ROOT")"
quoted_registry="$(printf '%q' "$REGISTRY")"

cat >"$JOB_SCRIPT" <<JOBSCRIPT
#!/usr/bin/env bash
set -euo pipefail

SESSION_DIR=${quoted_session_dir}
SESSION_NAME=${quoted_session_name}
RUN_DIR=${quoted_run_dir}
DEST_ROOT=${quoted_dest_root}
REPO_ROOT=${quoted_repo_root}
REGISTRY=${quoted_registry}
JOB_DRY_RUN=${JOB_DRY_RUN}
REGISTER=${REGISTER}
RECORDING_ONLY=${RECORDING_ONLY}
ALLOW_PREFLIGHT_FAILURES=${ALLOW_PREFLIGHT_FAILURES}
RUN_VIDEO_DIAGNOSTICS=${RUN_VIDEO_DIAGNOSTICS}
RUN_H5_DIAGNOSTICS=${RUN_H5_DIAGNOSTICS}
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
if [[ "\${ALLOW_PREFLIGHT_FAILURES}" == "1" ]]; then
  cmd+=(--allow-preflight-failures)
fi
if [[ "\${RUN_VIDEO_DIAGNOSTICS}" == "1" ]]; then
  cmd+=(--run-video-diagnostics)
fi
if [[ "\${RUN_H5_DIAGNOSTICS}" == "1" ]]; then
  cmd+=(--run-h5-diagnostics)
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
