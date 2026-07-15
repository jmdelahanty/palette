#!/usr/bin/env bash
set -euo pipefail
umask 0002

SOURCE_RUN_PATH=""
BENCHMARK_ID=""
PALETTE_REPO="${PALETTE_GROUPS_REPO:-/groups/johnson/johnsonlab/jeremy/gitrepos/palette}"
SUBMIT_HOST="${PALETTE_LSF_SUBMIT_HOST:-login1-citrus-poller}"
LOG_DIR=""
SCRATCH_BASE=""
QUEUE=""
NCORES=16
MEM_GB=64
WALLTIME="8:00"
KEEP_SCRATCH=0
BENCHMARK_TRANSFER=0
SUBMIT=0
SHARD_ROWS=()

usage() {
  cat <<'USAGE'
Usage: submit_subject_shape_sharding_benchmark_bsub.sh \
  --source-run-path PATH --benchmark-id ID [options]

Stage one immutable subject-shape run to node-local storage, then clone it into
indexed-sharding candidates while preserving every logical chunk and decoded
array value. The authoritative source is read-only.

Required:
  --source-run-path PATH   analysis/subject_shape_runs/<run> directory
  --benchmark-id ID        Immutable processing-log directory identifier

Options:
  --shard-rows N           Candidate physical row shard; repeatable
                           Defaults: 16384 through 1048576
  --ncores N               Parallel shard-copy workers (default: 16)
  --mem-gb N               Memory request (default: 64)
  --walltime H:MM          Walltime (default: 8:00)
  --palette-repo PATH      Clean cluster-visible Palette checkout
  --submit-host HOST       SSH poller when bsub is unavailable locally
  --log-dir PATH           Defaults below the recording processing logs
  --scratch-base PATH      Node-local base override
  --queue NAME             LSF queue; default is cluster default
  --benchmark-transfer     Time checksum-validated publication copies to groups
  --keep-scratch           Retain staged source and candidates
  --submit                 Submit; otherwise render only
  -h, --help               Show this help
USAGE
}

fail() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --source-run-path) SOURCE_RUN_PATH="$2"; shift 2;;
    --benchmark-id) BENCHMARK_ID="$2"; shift 2;;
    --shard-rows) SHARD_ROWS+=("$2"); shift 2;;
    --ncores) NCORES="$2"; shift 2;;
    --mem-gb) MEM_GB="$2"; shift 2;;
    --walltime) WALLTIME="$2"; shift 2;;
    --palette-repo) PALETTE_REPO="$2"; shift 2;;
    --submit-host) SUBMIT_HOST="$2"; shift 2;;
    --log-dir) LOG_DIR="$2"; shift 2;;
    --scratch-base) SCRATCH_BASE="$2"; shift 2;;
    --queue) QUEUE="$2"; shift 2;;
    --benchmark-transfer) BENCHMARK_TRANSFER=1; shift;;
    --keep-scratch) KEEP_SCRATCH=1; shift;;
    --submit) SUBMIT=1; shift;;
    -h|--help) usage; exit 0;;
    *) fail "unknown argument: $1";;
  esac
done

[[ -n "$SOURCE_RUN_PATH" ]] || fail "--source-run-path is required"
[[ -n "$BENCHMARK_ID" ]] || fail "--benchmark-id is required"
[[ "$BENCHMARK_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || fail "unsafe --benchmark-id"
[[ "$NCORES" =~ ^[1-9][0-9]*$ ]] || fail "--ncores must be positive"
[[ "$MEM_GB" =~ ^[1-9][0-9]*$ ]] || fail "--mem-gb must be positive"
[[ -f "$SOURCE_RUN_PATH/zarr.json" || -f "$SOURCE_RUN_PATH/.zgroup" ]] || fail "source run metadata not found"
if [[ ${#SHARD_ROWS[@]} -eq 0 ]]; then
  SHARD_ROWS=(16384 32768 65536 131072 262144 524288 1048576)
fi
for value in "${SHARD_ROWS[@]}"; do
  [[ "$value" =~ ^[1-9][0-9]*$ ]] || fail "invalid --shard-rows: $value"
done
git -C "$PALETTE_REPO" rev-parse --git-dir >/dev/null 2>&1 || \
  fail "Palette checkout not found: $PALETTE_REPO"
[[ -x "$PALETTE_REPO/scripts/py" ]] || fail "Palette scripts/py is not executable"
[[ -f "$PALETTE_REPO/src/fisheye/diagnostics/benchmark_subject_shape_sharding.py" ]] || \
  fail "Palette checkout lacks the subject-shape sharding benchmark"
[[ -z "$(git -C "$PALETTE_REPO" status --porcelain)" ]] || fail "Palette checkout must be clean"

case "$SOURCE_RUN_PATH" in
  */zarr/*) RECORDING_DIR="${SOURCE_RUN_PATH%%/zarr/*}";;
  *) fail "--source-run-path must be inside a recording's zarr directory";;
esac
if [[ -z "$LOG_DIR" ]]; then
  LOG_DIR="${RECORDING_DIR}/.processing_logs/subject_shape_benchmarks"
fi
RUN_DIR="${LOG_DIR}/${BENCHMARK_ID}"
[[ ! -e "$RUN_DIR" ]] || fail "submission directory already exists: $RUN_DIR"
mkdir -p "$RUN_DIR"

EXPECTED_COMMIT="$(git -C "$PALETTE_REPO" rev-parse HEAD)"
JOB_SCRIPT="${RUN_DIR}/run_subject_shape_sharding_benchmark.sh"
REPORT_PATH="${RUN_DIR}/sharding_benchmark_report.json"
STATUS_FILE="${RUN_DIR}/status.txt"
SUBMISSION_FILE="${RUN_DIR}/submission.txt"
SHARD_ROWS_FILE="${RUN_DIR}/shard_rows.txt"
printf '%s\n' "${SHARD_ROWS[@]}" >"$SHARD_ROWS_FILE"

q_repo="$(printf '%q' "$PALETTE_REPO")"
q_source="$(printf '%q' "$SOURCE_RUN_PATH")"
q_benchmark="$(printf '%q' "$BENCHMARK_ID")"
q_expected="$(printf '%q' "$EXPECTED_COMMIT")"
q_report="$(printf '%q' "$REPORT_PATH")"
q_status="$(printf '%q' "$STATUS_FILE")"
q_rows="$(printf '%q' "$SHARD_ROWS_FILE")"
q_run_dir="$(printf '%q' "$RUN_DIR")"
q_scratch="$(printf '%q' "$SCRATCH_BASE")"

cat >"$JOB_SCRIPT" <<JOBSCRIPT
#!/usr/bin/env bash
set -euo pipefail
umask 0002
PALETTE_REPO=${q_repo}
SOURCE_RUN_PATH=${q_source}
BENCHMARK_ID=${q_benchmark}
EXPECTED_COMMIT=${q_expected}
REPORT_PATH=${q_report}
STATUS_FILE=${q_status}
SHARD_ROWS_FILE=${q_rows}
RUN_DIR=${q_run_dir}
CONFIGURED_SCRATCH_BASE=${q_scratch}
NCORES=${NCORES}
KEEP_SCRATCH=${KEEP_SCRATCH}
BENCHMARK_TRANSFER=${BENCHMARK_TRANSFER}

[[ -n "\${LSB_JOBID:-}" ]] || { printf 'Refusing execution outside LSF.\n' >&2; exit 2; }
cd "\${PALETTE_REPO}"
ACTUAL_COMMIT="\$(git rev-parse HEAD)"
[[ "\${ACTUAL_COMMIT}" == "\${EXPECTED_COMMIT}" ]] || { printf 'Palette commit mismatch.\n' >&2; exit 2; }
[[ -z "\$(git status --porcelain)" ]] || { printf 'Refusing dirty Palette checkout.\n' >&2; exit 2; }

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
case "\${scratch_base}" in /groups/*) printf 'Refusing shared scratch.\n' >&2; exit 2;; esac
scratch_root="\${scratch_base}/subject_shape_sharding_\${BENCHMARK_ID}"
staged_source="\${scratch_root}/source.zarr"
output_root="\${scratch_root}/variants"
transfer_root="\${RUN_DIR}/transfer_tmp"
mkdir -p "\${scratch_root}"
cleanup() {
  rm -rf -- "\${transfer_root}"
  if [[ "\${KEEP_SCRATCH}" != "1" ]]; then rm -rf -- "\${scratch_root}"; fi
}
trap cleanup EXIT

SECONDS=0
rsync -a -- "\${SOURCE_RUN_PATH}/" "\${staged_source}/"
staging_seconds=\${SECONDS}
verification_output="\${scratch_root}/source_verification.txt"
rsync -a -n -c --delete --itemize-changes -- "\${SOURCE_RUN_PATH}/" "\${staged_source}/" >"\${verification_output}"
[[ ! -s "\${verification_output}" ]] || { printf 'Staged source checksum verification failed.\n' >&2; exit 2; }

cmd=(
  scripts/py -m fisheye.diagnostics.benchmark_subject_shape_sharding
  "\${staged_source}"
  --output-root "\${output_root}"
  --workers "\${NCORES}"
  --read-repeats 2
  --report "\${REPORT_PATH}"
  --apply
)
while IFS= read -r rows; do [[ -n "\${rows}" ]] && cmd+=(--shard-rows "\${rows}"); done <"\${SHARD_ROWS_FILE}"
if [[ "\${BENCHMARK_TRANSFER}" == "1" ]]; then cmd+=(--transfer-root "\${transfer_root}"); fi
printf 'benchmark_command='; printf '%q ' "\${cmd[@]}"; printf '\n'
set +e
"\${cmd[@]}"
payload_rc=\$?
set -e

status_tmp="\${STATUS_FILE}.tmp.\$\$"
{
  if [[ "\${payload_rc}" == "0" ]]; then printf 'status=complete\n'; else printf 'status=failed\n'; fi
  printf 'completed_at_utc=%s\n' "\$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'host=%s\n' "\$(hostname)"
  printf 'job_id=%s\n' "\${LSB_JOBID}"
  printf 'palette_commit=%s\n' "\${ACTUAL_COMMIT}"
  printf 'source_run_path=%s\n' "\${SOURCE_RUN_PATH}"
  printf 'benchmark_id=%s\n' "\${BENCHMARK_ID}"
  printf 'source_staging_seconds=%s\n' "\${staging_seconds}"
  printf 'source_checksum_verified=true\n'
  printf 'report=%s\n' "\${REPORT_PATH}"
  printf 'payload_returncode=%s\n' "\${payload_rc}"
} >"\${status_tmp}"
mv "\${status_tmp}" "\${STATUS_FILE}"
exit "\${payload_rc}"
JOBSCRIPT
chmod +x "$JOB_SCRIPT"

BSUB_ARGS=(
  -J "subject_shape_sharding_${BENCHMARK_ID}"
  -n "$NCORES"
  -W "$WALLTIME"
  -R "span[hosts=1] rusage[mem=${MEM_GB}G]"
  -oo "${RUN_DIR}/%J.out"
  -eo "${RUN_DIR}/%J.err"
)
if [[ -n "$QUEUE" ]]; then BSUB_ARGS+=(-q "$QUEUE"); fi
BSUB_COMMAND=(bsub "${BSUB_ARGS[@]}" bash "$JOB_SCRIPT")

printf 'mode=%s\n' "$([[ "$SUBMIT" == "1" ]] && printf submit || printf render-only)"
printf 'palette_commit=%s\n' "$EXPECTED_COMMIT"
printf 'benchmark_id=%s\n' "$BENCHMARK_ID"
printf 'run_dir=%s\n' "$RUN_DIR"
printf 'job_script=%s\n' "$JOB_SCRIPT"
printf 'report=%s\n' "$REPORT_PATH"
printf 'bsub_command='; printf '%q ' "${BSUB_COMMAND[@]}"; printf '\n'

if [[ "$SUBMIT" == "1" ]]; then
  if command -v bsub >/dev/null 2>&1; then
    submit_mode="local_bsub"
    submit_output="$("${BSUB_COMMAND[@]}")"
  else
    [[ -n "$SUBMIT_HOST" ]] || fail "bsub unavailable and submit host empty"
    submit_mode="ssh_bsub"
    printf -v remote_command '%q ' "${BSUB_COMMAND[@]}"
    submit_output="$(ssh "$SUBMIT_HOST" "$remote_command")"
  fi
  printf '%s\n' "$submit_output"
  job_id="$(printf '%s\n' "$submit_output" | sed -n 's/^Job <\([0-9][0-9]*\)>.*/\1/p' | head -n 1)"
  [[ -n "$job_id" ]] || fail "could not parse an LSF job ID"
  {
    printf 'submitted_at_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'submit_mode=%s\n' "$submit_mode"
    printf 'submit_host=%s\n' "$SUBMIT_HOST"
    printf 'job_id=%s\n' "$job_id"
    printf 'job_script=%s\n' "$JOB_SCRIPT"
    printf 'lsf_stdout=%s\n' "${RUN_DIR}/${job_id}.out"
    printf 'lsf_stderr=%s\n' "${RUN_DIR}/${job_id}.err"
    printf 'status_file=%s\n' "$STATUS_FILE"
    printf 'report=%s\n' "$REPORT_PATH"
    printf 'palette_commit=%s\n' "$EXPECTED_COMMIT"
  } >"$SUBMISSION_FILE"
  printf 'job_id=%s\n' "$job_id"
  printf 'lsf_stdout=%s\n' "${RUN_DIR}/${job_id}.out"
  printf 'lsf_stderr=%s\n' "${RUN_DIR}/${job_id}.err"
  printf 'status_file=%s\n' "$STATUS_FILE"
  printf 'submission_file=%s\n' "$SUBMISSION_FILE"
fi
