#!/usr/bin/env bash
set -euo pipefail
umask 0002

SOURCE_RUN=""
RUN_ID="tail_kinematics_sharding_$(date +%Y%m%d_%H%M%S)"
PALETTE_REPO="${PALETTE_GROUPS_REPO:-/groups/johnson/johnsonlab/jeremy/gitrepos/palette}"
SUBMIT_HOST="${PALETTE_LSF_SUBMIT_HOST:-login1-citrus-poller}"
LOG_ROOT=""
SCRATCH_BASE=""
QUEUE=""
NCORES=8
MEM_GB=32
WALLTIME="4:00"
READ_REPEATS=3
RANDOM_ROWS=32
WINDOW_ROWS=1024
WINDOW_COUNT=8
SCAN_ROWS=16384
KEEP_SCRATCH=0
SUBMIT=0
SHARD_ROWS=(16384 65536 131072 262144)

usage() {
  cat <<'USAGE'
Usage: submit_tail_kinematics_sharding_benchmark_bsub.sh \
  --source-run PATH [options]

Render or submit one CPU job that stages a completed immutable tail-kinematics
run once, then clones it into disposable node-local Zarr variants with 16,384,
65,536, 131,072, and 262,144-row outer shards. Logical chunks and decoded data
remain unchanged. The shared recording is read-only; only the JSON/timing report
is retained on shared storage.

Required:
  --source-run PATH      Completed analysis/tail_kinematics_runs/<run> directory

Options:
  --run-id ID            Benchmark identifier
  --palette-repo PATH    Clean cluster-visible Palette checkout
  --submit-host HOST     SSH poller used when bsub is unavailable locally
  --log-root PATH        Shared report parent (default: recording processing logs)
  --scratch-base PATH    Node-local scratch base
  --queue NAME           LSF queue; default is the cluster default
  --ncores N             Copy workers and allocated CPU slots (default: 8)
  --mem-gb N             Memory request in GiB (default: 32)
  --walltime H:MM        LSF walltime (default: 4:00)
  --read-repeats N       Read-pattern repetitions (default: 3)
  --random-rows N        Random one-row reads per representative array (default: 32)
  --window-rows N        Rows per contiguous read window (default: 1024)
  --window-count N       Windows per representative array (default: 8)
  --scan-rows N          Bounded rows per full-scan read (default: 16384)
  --shard-rows CSV       Override candidates, e.g. 16384,65536,131072,262144
  --keep-scratch         Retain node-local variants after success
  --submit               Submit; otherwise render only
  -h, --help             Show this help

Workers write complete, non-overlapping outer row-shard stripes. The driver is
the only process that creates arrays/groups/attrs and writes the benchmark
report. Results are cache-warm or mixed node-local timings, not cold-cache PRFS
measurements.
USAGE
}

fail() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --source-run) SOURCE_RUN="$2"; shift 2;;
    --run-id) RUN_ID="$2"; shift 2;;
    --palette-repo) PALETTE_REPO="$2"; shift 2;;
    --submit-host) SUBMIT_HOST="$2"; shift 2;;
    --log-root) LOG_ROOT="$2"; shift 2;;
    --scratch-base) SCRATCH_BASE="$2"; shift 2;;
    --queue) QUEUE="$2"; shift 2;;
    --ncores) NCORES="$2"; shift 2;;
    --mem-gb) MEM_GB="$2"; shift 2;;
    --walltime) WALLTIME="$2"; shift 2;;
    --read-repeats) READ_REPEATS="$2"; shift 2;;
    --random-rows) RANDOM_ROWS="$2"; shift 2;;
    --window-rows) WINDOW_ROWS="$2"; shift 2;;
    --window-count) WINDOW_COUNT="$2"; shift 2;;
    --scan-rows) SCAN_ROWS="$2"; shift 2;;
    --shard-rows)
      IFS=',' read -r -a SHARD_ROWS <<<"$2"
      shift 2
      ;;
    --keep-scratch) KEEP_SCRATCH=1; shift;;
    --submit) SUBMIT=1; shift;;
    -h|--help) usage; exit 0;;
    *) fail "unknown argument: $1";;
  esac
done

[[ -n "$SOURCE_RUN" ]] || fail "--source-run is required"
[[ "$RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || fail "unsafe --run-id: $RUN_ID"
for value in "$NCORES" "$MEM_GB" "$READ_REPEATS" "$RANDOM_ROWS" \
  "$WINDOW_ROWS" "$WINDOW_COUNT" "$SCAN_ROWS"; do
  [[ "$value" =~ ^[1-9][0-9]*$ ]] || fail "expected a positive integer, got: $value"
done
(( ${#SHARD_ROWS[@]} > 0 )) || fail "at least one --shard-rows candidate is required"
for value in "${SHARD_ROWS[@]}"; do
  [[ "$value" =~ ^[1-9][0-9]*$ ]] || fail "invalid shard-row candidate: $value"
done
[[ -f "$SOURCE_RUN/zarr.json" || -f "$SOURCE_RUN/.zgroup" ]] || \
  fail "source run metadata not found: $SOURCE_RUN"
[[ -d "$PALETTE_REPO/.git" ]] || fail "Palette checkout not found: $PALETTE_REPO"
[[ -x "$PALETTE_REPO/scripts/py" ]] || fail "Palette scripts/py is not executable"
[[ -f "$PALETTE_REPO/src/fisheye/diagnostics/benchmark_tail_kinematics_sharding.py" ]] || \
  fail "Palette checkout lacks the tail-kinematics sharding benchmark"
[[ -z "$(git -C "$PALETTE_REPO" status --porcelain)" ]] || \
  fail "Palette checkout must be clean before rendering a cluster job: $PALETTE_REPO"

if [[ -z "$LOG_ROOT" ]]; then
  case "$SOURCE_RUN" in
    */zarr/*.zarr/analysis/tail_kinematics_runs/*)
      ZARR_ROOT="${SOURCE_RUN%%/analysis/tail_kinematics_runs/*}"
      RECORDING_DIR="$(dirname -- "$(dirname -- "$ZARR_ROOT")")"
      LOG_ROOT="${RECORDING_DIR}/.processing_logs/tail_kinematics_sharding_benchmarks"
      ;;
    *)
      LOG_ROOT="/groups/johnson/johnsonlab/jeremy/recordings/logs/tail_kinematics_sharding_benchmarks"
      ;;
  esac
fi

RUN_DIR="${LOG_ROOT%/}/${RUN_ID}"
[[ ! -e "$RUN_DIR" ]] || fail "benchmark run directory already exists: $RUN_DIR"
mkdir -p "$RUN_DIR"

EXPECTED_COMMIT="$(git -C "$PALETTE_REPO" rev-parse HEAD)"
JOB_SCRIPT="${RUN_DIR}/run_tail_kinematics_sharding_benchmark.sh"
REPORT_PATH="${RUN_DIR}/benchmark_report.json"
STATUS_FILE="${RUN_DIR}/status.txt"
SUBMISSION_FILE="${RUN_DIR}/submission.txt"

q_repo="$(printf '%q' "$PALETTE_REPO")"
q_source="$(printf '%q' "$SOURCE_RUN")"
q_run_id="$(printf '%q' "$RUN_ID")"
q_expected_commit="$(printf '%q' "$EXPECTED_COMMIT")"
q_run_dir="$(printf '%q' "$RUN_DIR")"
q_report="$(printf '%q' "$REPORT_PATH")"
q_status="$(printf '%q' "$STATUS_FILE")"
q_scratch_base="$(printf '%q' "$SCRATCH_BASE")"
q_shard_rows="$(printf '%q' "${SHARD_ROWS[*]}")"

cat >"$JOB_SCRIPT" <<JOBSCRIPT
#!/usr/bin/env bash
set -euo pipefail
umask 0002

PALETTE_REPO=${q_repo}
SOURCE_RUN=${q_source}
RUN_ID=${q_run_id}
EXPECTED_COMMIT=${q_expected_commit}
RUN_DIR=${q_run_dir}
REPORT_PATH=${q_report}
STATUS_FILE=${q_status}
CONFIGURED_SCRATCH_BASE=${q_scratch_base}
SHARD_ROWS_TEXT=${q_shard_rows}
NCORES=${NCORES}
READ_REPEATS=${READ_REPEATS}
RANDOM_ROWS=${RANDOM_ROWS}
WINDOW_ROWS=${WINDOW_ROWS}
WINDOW_COUNT=${WINDOW_COUNT}
SCAN_ROWS=${SCAN_ROWS}
KEEP_SCRATCH=${KEEP_SCRATCH}

[[ -n "\${LSB_JOBID:-}" ]] || {
  printf 'Refusing tail-sharding benchmark execution outside an LSF allocation.\n' >&2
  exit 2
}
cd "\${PALETTE_REPO}"
ACTUAL_COMMIT="\$(git rev-parse HEAD)"
[[ "\${ACTUAL_COMMIT}" == "\${EXPECTED_COMMIT}" ]] || {
  printf 'Palette commit mismatch: expected %s, found %s\n' \
    "\${EXPECTED_COMMIT}" "\${ACTUAL_COMMIT}" >&2
  exit 2
}
[[ -z "\$(git status --porcelain)" ]] || {
  printf 'Refusing dirty Palette checkout on the execution host.\n' >&2
  exit 2
}

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
    printf 'Refusing shared /groups path as node-local scratch: %s\n' "\${scratch_base}" >&2
    exit 2
    ;;
esac
scratch_root="\${scratch_base}/tail_sharding_\${RUN_ID}"
staged_source="\${scratch_root}/source_tail_run.zarr"
variant_root="\${scratch_root}/variants"
transfer_root="\${RUN_DIR}/.prfs_transfer_scratch"

cleanup() {
  local status=\$?
  trap - EXIT INT TERM
  if [[ "\${KEEP_SCRATCH}" != "1" && -d "\${scratch_root}" ]]; then
    rm -rf "\${scratch_root}"
  fi
  exit "\${status}"
}
trap cleanup EXIT INT TERM
mkdir -p "\${scratch_root}" "\${RUN_DIR}"

printf 'host=%s\n' "\$(hostname)"
printf 'job_id=%s\n' "\${LSB_JOBID}"
printf 'allocated_slots=%s\n' "\${LSB_DJOB_NUMPROC:-1}"
printf 'source_run=%s\n' "\${SOURCE_RUN}"
printf 'scratch_root=%s\n' "\${scratch_root}"

/usr/bin/time -f \$'elapsed_seconds=%e\nmaximum_rss_kib=%M\nfilesystem_inputs=%I\nfilesystem_outputs=%O' \
  -o "\${RUN_DIR}/stage.time.txt" \
  rsync -a -- "\${SOURCE_RUN}/" "\${staged_source}/"

cmd=(
  scripts/py -m fisheye.diagnostics.benchmark_tail_kinematics_sharding
  "\${staged_source}"
  --output-root "\${variant_root}"
  --workers "\${NCORES}"
  --read-repeats "\${READ_REPEATS}"
  --random-rows "\${RANDOM_ROWS}"
  --window-rows "\${WINDOW_ROWS}"
  --window-count "\${WINDOW_COUNT}"
  --scan-rows "\${SCAN_ROWS}"
  --report "\${REPORT_PATH}"
  --transfer-root "\${transfer_root}"
  --apply
)
read -r -a shard_rows_values <<<"\${SHARD_ROWS_TEXT}"
for value in "\${shard_rows_values[@]}"; do
  cmd+=(--shard-rows "\${value}")
done

printf 'benchmark_command='; printf '%q ' "\${cmd[@]}"; printf '\n'
set +e
/usr/bin/time -f \$'elapsed_seconds=%e\nmaximum_rss_kib=%M\nfilesystem_inputs=%I\nfilesystem_outputs=%O' \
  -o "\${RUN_DIR}/benchmark.time.txt" \
  "\${cmd[@]}"
payload_rc=\$?
set -e

if [[ "\${payload_rc}" == "0" ]]; then
  scripts/py -c 'import json,sys; p=json.load(open(sys.argv[1])); assert p["status"] == "complete" and p["all_variants_exact"]' "\${REPORT_PATH}"
fi

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
  printf 'source_run=%s\n' "\${SOURCE_RUN}"
  printf 'scratch_root=%s\n' "\${scratch_root}"
  printf 'payload_returncode=%s\n' "\${payload_rc}"
  printf 'benchmark_report=%s\n' "\${REPORT_PATH}"
} >"\${status_tmp}"
mv "\${status_tmp}" "\${STATUS_FILE}"
exit "\${payload_rc}"
JOBSCRIPT
chmod +x "$JOB_SCRIPT"

BSUB_ARGS=(
  -J "tail_shard_${RUN_ID}"
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
printf 'source_run=%s\n' "$SOURCE_RUN"
printf 'shard_rows=%s\n' "${SHARD_ROWS[*]}"
printf 'run_dir=%s\n' "$RUN_DIR"
printf 'job_script=%s\n' "$JOB_SCRIPT"
printf 'benchmark_report=%s\n' "$REPORT_PATH"
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
    printf 'benchmark_report=%s\n' "$REPORT_PATH"
    printf 'palette_commit=%s\n' "$EXPECTED_COMMIT"
  } >"$submission_tmp"
  mv "$submission_tmp" "$SUBMISSION_FILE"
  printf 'job_id=%s\n' "$job_id"
  printf 'lsf_stdout=%s\n' "${RUN_DIR}/${job_id}.out"
  printf 'lsf_stderr=%s\n' "${RUN_DIR}/${job_id}.err"
  printf 'status_file=%s\n' "$STATUS_FILE"
  printf 'submission_file=%s\n' "$SUBMISSION_FILE"
fi
