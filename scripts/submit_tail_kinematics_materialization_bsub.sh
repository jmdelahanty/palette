#!/usr/bin/env bash
set -euo pipefail
umask 0002

ZARR_PATH=""
SHAPE_RUN=""
RUN_NAME=""
PALETTE_REPO="${PALETTE_GROUPS_REPO:-/groups/johnson/johnsonlab/jeremy/gitrepos/palette}"
SUBMIT_HOST="${PALETTE_LSF_SUBMIT_HOST:-login1-citrus-poller}"
LOG_DIR=""
SCRATCH_BASE=""
QUEUE=""
NCORES=8
MEM_GB=32
WALLTIME="4:00"
BLOCK_ROWS=16384
OUTPUT_SHARD_ROWS=262144
TAIL_ANGLE_SAMPLE_COUNT=10
KEEP_SCRATCH=0
SUBMIT=0

usage() {
  cat <<'USAGE'
Usage: submit_tail_kinematics_materialization_bsub.sh \
  --zarr PATH --shape-run RUN --run-name RUN [options]

Render or submit one fail-closed LSF job for a recording's tail-kinematics
materialization. The compute job copies the complete required subject-shape
surface to node-local storage in one manifest-driven transfer, computes there,
and atomically publishes one completed run group to the authoritative Zarr.
No Zarr array data are read on the login host.

Required:
  --zarr PATH                  Palette analysis Zarr on shared storage
  --shape-run RUN             Exact immutable subject-shape source run
  --run-name RUN              New immutable tail-kinematics run name

Options:
  --palette-repo PATH         Clean cluster-visible Palette checkout
  --submit-host HOST          SSH poller used when bsub is unavailable locally
  --log-dir PATH              Default: <recording>/.processing_logs/tail_kinematics
  --scratch-base PATH         Node-local base; default: /scratch/$USER/$LSB_JOBID
                              with $TMPDIR as a node-local fallback
  --queue NAME                LSF queue; default is the cluster default
  --ncores N                  CPU slots and process-shard workers (default: 8)
  --mem-gb N                  Memory request (default: 32)
  --walltime H:MM             Walltime (default: 4:00)
  --block-rows N              Bounded compute rows (default: 16384)
  --output-shard-rows N       Physical output shard rows (default: 262144)
  --tail-angle-sample-count N Resampled tail points (default: 10)
  --keep-scratch              Retain node-local files after successful publish
  --submit                    Submit; otherwise render only
  -h, --help                  Show this help

Each worker owns one complete, non-overlapping output shard and computes it in
bounded sub-blocks. The driver alone creates/finalizes run metadata and performs
atomic publication.
USAGE
}

fail() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --zarr) ZARR_PATH="$2"; shift 2;;
    --shape-run) SHAPE_RUN="$2"; shift 2;;
    --run-name) RUN_NAME="$2"; shift 2;;
    --palette-repo) PALETTE_REPO="$2"; shift 2;;
    --submit-host) SUBMIT_HOST="$2"; shift 2;;
    --log-dir) LOG_DIR="$2"; shift 2;;
    --scratch-base) SCRATCH_BASE="$2"; shift 2;;
    --queue) QUEUE="$2"; shift 2;;
    --ncores) NCORES="$2"; shift 2;;
    --mem-gb) MEM_GB="$2"; shift 2;;
    --walltime) WALLTIME="$2"; shift 2;;
    --block-rows) BLOCK_ROWS="$2"; shift 2;;
    --output-shard-rows) OUTPUT_SHARD_ROWS="$2"; shift 2;;
    --tail-angle-sample-count) TAIL_ANGLE_SAMPLE_COUNT="$2"; shift 2;;
    --keep-scratch) KEEP_SCRATCH=1; shift;;
    --submit) SUBMIT=1; shift;;
    -h|--help) usage; exit 0;;
    *) fail "unknown argument: $1";;
  esac
done

[[ -n "$ZARR_PATH" ]] || fail "--zarr is required"
[[ -n "$SHAPE_RUN" ]] || fail "--shape-run is required"
[[ -n "$RUN_NAME" ]] || fail "--run-name is required"
[[ "$SHAPE_RUN" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || \
  fail "unsafe --shape-run: $SHAPE_RUN"
[[ "$RUN_NAME" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || \
  fail "unsafe --run-name: $RUN_NAME"
[[ "$MEM_GB" =~ ^[1-9][0-9]*$ ]] || fail "--mem-gb must be a positive integer"
[[ "$NCORES" =~ ^[1-9][0-9]*$ ]] || fail "--ncores must be a positive integer"
[[ "$BLOCK_ROWS" =~ ^[1-9][0-9]*$ ]] || fail "--block-rows must be a positive integer"
[[ "$OUTPUT_SHARD_ROWS" =~ ^[1-9][0-9]*$ ]] || \
  fail "--output-shard-rows must be a positive integer"
[[ "$TAIL_ANGLE_SAMPLE_COUNT" =~ ^[0-9]+$ ]] || \
  fail "--tail-angle-sample-count must be an integer"
(( TAIL_ANGLE_SAMPLE_COUNT >= 2 )) || \
  fail "--tail-angle-sample-count must be at least 2"
[[ -f "$ZARR_PATH/zarr.json" || -f "$ZARR_PATH/.zgroup" ]] || \
  fail "analysis Zarr metadata not found: $ZARR_PATH"
[[ -d "$PALETTE_REPO/.git" ]] || fail "Palette checkout not found: $PALETTE_REPO"
[[ -x "$PALETTE_REPO/scripts/py" ]] || fail "Palette scripts/py is not executable"
[[ -f "$PALETTE_REPO/src/fisheye/analysis_workflows/materializers/tail_kinematics.py" ]] || \
  fail "Palette checkout lacks the tail-kinematics materializer"
[[ -z "$(git -C "$PALETTE_REPO" status --porcelain)" ]] || \
  fail "Palette checkout must be clean before rendering a cluster job: $PALETTE_REPO"

ZARR_PARENT="$(dirname -- "$ZARR_PATH")"
RECORDING_DIR="$(dirname -- "$ZARR_PARENT")"
if [[ -z "$LOG_DIR" ]]; then
  LOG_DIR="${RECORDING_DIR}/.processing_logs/tail_kinematics"
fi
SAFE_ZARR_NAME="$(basename -- "$ZARR_PATH" | tr -c 'A-Za-z0-9_.-' '_')"
RUN_DIR="${LOG_DIR}/${RUN_NAME}_${SAFE_ZARR_NAME}"
[[ ! -e "$RUN_DIR" ]] || fail "submission directory already exists: $RUN_DIR"
mkdir -p "$RUN_DIR"

EXPECTED_COMMIT="$(git -C "$PALETTE_REPO" rev-parse HEAD)"
JOB_SCRIPT="${RUN_DIR}/run_tail_kinematics_materialization.sh"
REPORT_PATH="${RUN_DIR}/materialization_report.json"
STATUS_FILE="${RUN_DIR}/status.txt"
SUBMISSION_FILE="${RUN_DIR}/submission.txt"

q_repo="$(printf '%q' "$PALETTE_REPO")"
q_zarr="$(printf '%q' "$ZARR_PATH")"
q_shape_run="$(printf '%q' "$SHAPE_RUN")"
q_run_name="$(printf '%q' "$RUN_NAME")"
q_expected_commit="$(printf '%q' "$EXPECTED_COMMIT")"
q_report="$(printf '%q' "$REPORT_PATH")"
q_status="$(printf '%q' "$STATUS_FILE")"
q_scratch_base="$(printf '%q' "$SCRATCH_BASE")"

cat >"$JOB_SCRIPT" <<JOBSCRIPT
#!/usr/bin/env bash
set -euo pipefail
umask 0002

PALETTE_REPO=${q_repo}
ZARR_PATH=${q_zarr}
SHAPE_RUN=${q_shape_run}
RUN_NAME=${q_run_name}
EXPECTED_COMMIT=${q_expected_commit}
REPORT_PATH=${q_report}
STATUS_FILE=${q_status}
CONFIGURED_SCRATCH_BASE=${q_scratch_base}
BLOCK_ROWS=${BLOCK_ROWS}
OUTPUT_SHARD_ROWS=${OUTPUT_SHARD_ROWS}
TAIL_ANGLE_SAMPLE_COUNT=${TAIL_ANGLE_SAMPLE_COUNT}
NCORES=${NCORES}
KEEP_SCRATCH=${KEEP_SCRATCH}

[[ -n "\${LSB_JOBID:-}" ]] || {
  printf 'Refusing tail-kinematics execution outside an LSF allocation.\n' >&2
  exit 2
}
cd "\${PALETTE_REPO}"
ACTUAL_COMMIT="\$(git rev-parse HEAD)"
if [[ "\${ACTUAL_COMMIT}" != "\${EXPECTED_COMMIT}" ]]; then
  printf 'Palette commit mismatch: expected %s, found %s\n' \
    "\${EXPECTED_COMMIT}" "\${ACTUAL_COMMIT}" >&2
  exit 2
fi
if [[ -n "\$(git status --porcelain)" ]]; then
  printf 'Refusing dirty Palette checkout on the execution host.\n' >&2
  exit 2
fi

if [[ -n "\${CONFIGURED_SCRATCH_BASE}" ]]; then
  scratch_base="\${CONFIGURED_SCRATCH_BASE}"
elif [[ -d "/scratch/\${USER}" && -w "/scratch/\${USER}" ]]; then
  scratch_base="/scratch/\${USER}/\${LSB_JOBID}"
elif [[ -n "\${TMPDIR:-}" && -d "\${TMPDIR}" && -w "\${TMPDIR}" ]]; then
  scratch_base="\${TMPDIR}/palette/\${LSB_JOBID}"
else
  printf 'No writable node-local scratch root is available.\n' >&2
  exit 2
fi
case "\${scratch_base}" in
  /groups/*)
    printf 'Refusing shared /groups path as node-local scratch: %s\n' \
      "\${scratch_base}" >&2
    exit 2
    ;;
esac
scratch_root="\${scratch_base}/tail_kinematics_\${RUN_NAME}"

cmd=(
  scripts/py -m fisheye.analysis_workflows.materializers.tail_kinematics
  "\${ZARR_PATH}"
  --shape-run "\${SHAPE_RUN}"
  --run-name "\${RUN_NAME}"
  --scratch-root "\${scratch_root}"
  --block-rows "\${BLOCK_ROWS}"
  --output-shard-rows "\${OUTPUT_SHARD_ROWS}"
  --tail-angle-sample-count "\${TAIL_ANGLE_SAMPLE_COUNT}"
  --execution-backend process_shards
  --num-workers "\${NCORES}"
  --copy-backend rsync
  --report "\${REPORT_PATH}"
  --apply
  --json
)
if [[ "\${KEEP_SCRATCH}" == "1" ]]; then cmd+=(--keep-scratch); fi

printf 'materialization_command='; printf '%q ' "\${cmd[@]}"; printf '\n'
set +e
"\${cmd[@]}"
payload_rc=\$?
set -e

status_tmp="\${STATUS_FILE}.tmp.\$\$"
{
  if [[ "\${payload_rc}" == "0" ]]; then
    printf 'status=complete\n'
  else
    printf 'status=failed\n'
  fi
  printf 'completed_at_utc=%s\n' "\$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'host=%s\n' "\$(hostname)"
  printf 'job_id=%s\n' "\${LSB_JOBID}"
  printf 'palette_commit=%s\n' "\${ACTUAL_COMMIT}"
  printf 'zarr_path=%s\n' "\${ZARR_PATH}"
  printf 'shape_run=%s\n' "\${SHAPE_RUN}"
  printf 'run_name=%s\n' "\${RUN_NAME}"
  printf 'scratch_root=%s\n' "\${scratch_root}"
  printf 'payload_returncode=%s\n' "\${payload_rc}"
  printf 'materialization_report=%s\n' "\${REPORT_PATH}"
} >"\${status_tmp}"
mv "\${status_tmp}" "\${STATUS_FILE}"
exit "\${payload_rc}"
JOBSCRIPT
chmod +x "$JOB_SCRIPT"

BSUB_ARGS=(
  -J "tail_kinematics_${RUN_NAME}"
  -n "$NCORES"
  -W "$WALLTIME"
  -R "rusage[mem=${MEM_GB}G]"
  -oo "${RUN_DIR}/%J.out"
  -eo "${RUN_DIR}/%J.err"
)
if [[ -n "$QUEUE" ]]; then BSUB_ARGS+=(-q "$QUEUE"); fi
BSUB_COMMAND=(bsub "${BSUB_ARGS[@]}" bash "$JOB_SCRIPT")

printf 'mode=%s\n' "$([[ "$SUBMIT" == "1" ]] && printf submit || printf render-only)"
printf 'palette_commit=%s\n' "$EXPECTED_COMMIT"
printf 'zarr_path=%s\n' "$ZARR_PATH"
printf 'shape_run=%s\n' "$SHAPE_RUN"
printf 'run_name=%s\n' "$RUN_NAME"
printf 'run_dir=%s\n' "$RUN_DIR"
printf 'job_script=%s\n' "$JOB_SCRIPT"
printf 'materialization_report=%s\n' "$REPORT_PATH"
printf 'bsub_command='; printf '%q ' "${BSUB_COMMAND[@]}"; printf '\n'

if [[ "$SUBMIT" == "1" ]]; then
  if command -v bsub >/dev/null 2>&1; then
    submit_mode="local_bsub"
    submit_output="$("${BSUB_COMMAND[@]}")"
  else
    [[ -n "$SUBMIT_HOST" ]] || fail "bsub is unavailable and --submit-host is empty"
    submit_mode="ssh_bsub"
    printf -v remote_command '%q ' "${BSUB_COMMAND[@]}"
    submit_output="$(ssh "$SUBMIT_HOST" "$remote_command")"
  fi
  printf '%s\n' "$submit_output"
  job_id="$(printf '%s\n' "$submit_output" | sed -n 's/^Job <\([0-9][0-9]*\)>.*/\1/p' | head -n 1)"
  [[ -n "$job_id" ]] || fail "could not parse an LSF job ID"
  submission_tmp="${SUBMISSION_FILE}.tmp.$$"
  {
    printf 'submitted_at_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'submit_mode=%s\n' "$submit_mode"
    printf 'submit_host=%s\n' "$SUBMIT_HOST"
    printf 'job_id=%s\n' "$job_id"
    printf 'job_script=%s\n' "$JOB_SCRIPT"
    printf 'lsf_stdout=%s\n' "${RUN_DIR}/${job_id}.out"
    printf 'lsf_stderr=%s\n' "${RUN_DIR}/${job_id}.err"
    printf 'status_file=%s\n' "$STATUS_FILE"
    printf 'materialization_report=%s\n' "$REPORT_PATH"
    printf 'palette_commit=%s\n' "$EXPECTED_COMMIT"
  } >"$submission_tmp"
  mv "$submission_tmp" "$SUBMISSION_FILE"
  printf 'job_id=%s\n' "$job_id"
  printf 'lsf_stdout=%s\n' "${RUN_DIR}/${job_id}.out"
  printf 'lsf_stderr=%s\n' "${RUN_DIR}/${job_id}.err"
  printf 'status_file=%s\n' "$STATUS_FILE"
  printf 'submission_file=%s\n' "$SUBMISSION_FILE"
fi
