#!/usr/bin/env bash
set -euo pipefail
umask 0002

ZARR_PATH=""
REFINED_RUN=""
BENCHMARK_ID=""
PALETTE_REPO="${PALETTE_GROUPS_REPO:-/groups/johnson/johnsonlab/jeremy/gitrepos/palette}"
SUBMIT_HOST="${PALETTE_LSF_SUBMIT_HOST:-login1-citrus-poller}"
LOG_DIR=""
SCRATCH_BASE=""
QUEUE=""
NCORES=32
MEM_GB=64
WALLTIME="4:00"
SOURCE_START_ROW=524288
ROW_COUNT=32768
KEEP_SCRATCH=0
SUBMIT=0
VARIANTS=()

usage() {
  cat <<'USAGE'
Usage: submit_subject_shape_compute_benchmark_bsub.sh \
  --zarr PATH --refined-run RUN --benchmark-id ID [options]

Render or submit one bounded, source-read-only subject-shape compute benchmark.
Every variant writes only to node-local disposable Zarr outputs. No canonical
analysis run is created or modified.

Required:
  --zarr PATH              Palette analysis Zarr on shared storage
  --refined-run RUN        Exact immutable refined-subject-mask input run
  --benchmark-id ID        Immutable processing-log directory identifier

Options:
  --variant SPEC           NAME:WORKERS:BLOCK_ROWS:NATIVE_THREADS[:FLAGS]
                           Repeatable. FLAGS: crop,per-task-open
  --source-start-row N     Aligned first source row (default: 524288)
  --row-count N            Bounded rows (default: 32768)
  --ncores N               Allocated slots (default: 32)
  --mem-gb N               Memory request (default: 64)
  --walltime H:MM          Walltime (default: 4:00)
  --palette-repo PATH      Clean cluster-visible Palette checkout
  --submit-host HOST       SSH poller when bsub is unavailable locally
  --log-dir PATH           Default: <recording>/.processing_logs/subject_shape_benchmarks
  --scratch-base PATH      Node-local base override
  --queue NAME             LSF queue; default is cluster default
  --keep-scratch           Retain disposable outputs after completion
  --submit                 Submit; otherwise render only
  -h, --help               Show this help

Default variants compare per-task versus persistent inputs, 256/512/1024-row
compute blocks, native thread limits, 16/32 workers, and foreground-cropped
centerline extraction.
USAGE
}

fail() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --zarr) ZARR_PATH="$2"; shift 2;;
    --refined-run) REFINED_RUN="$2"; shift 2;;
    --benchmark-id) BENCHMARK_ID="$2"; shift 2;;
    --variant) VARIANTS+=("$2"); shift 2;;
    --source-start-row) SOURCE_START_ROW="$2"; shift 2;;
    --row-count) ROW_COUNT="$2"; shift 2;;
    --ncores) NCORES="$2"; shift 2;;
    --mem-gb) MEM_GB="$2"; shift 2;;
    --walltime) WALLTIME="$2"; shift 2;;
    --palette-repo) PALETTE_REPO="$2"; shift 2;;
    --submit-host) SUBMIT_HOST="$2"; shift 2;;
    --log-dir) LOG_DIR="$2"; shift 2;;
    --scratch-base) SCRATCH_BASE="$2"; shift 2;;
    --queue) QUEUE="$2"; shift 2;;
    --keep-scratch) KEEP_SCRATCH=1; shift;;
    --submit) SUBMIT=1; shift;;
    -h|--help) usage; exit 0;;
    *) fail "unknown argument: $1";;
  esac
done

[[ -n "$ZARR_PATH" ]] || fail "--zarr is required"
[[ -n "$REFINED_RUN" ]] || fail "--refined-run is required"
[[ -n "$BENCHMARK_ID" ]] || fail "--benchmark-id is required"
[[ "$REFINED_RUN" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || fail "unsafe --refined-run"
[[ "$BENCHMARK_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || fail "unsafe --benchmark-id"
for value_name in "ncores:$NCORES" "mem-gb:$MEM_GB" "source-start-row:$SOURCE_START_ROW" "row-count:$ROW_COUNT"; do
  value="${value_name#*:}"
  [[ "$value" =~ ^[0-9]+$ ]] || fail "${value_name%%:*} must be an integer"
done
(( NCORES > 0 && MEM_GB > 0 && ROW_COUNT > 0 )) || fail "resource and row counts must be positive"
(( SOURCE_START_ROW % 256 == 0 )) || fail "--source-start-row must align to 256 rows"

if [[ ${#VARIANTS[@]} -eq 0 ]]; then
  VARIANTS=(
    "per_task_256_w8_t1:8:256:1:per-task-open"
    "persistent_256_w8_t1:8:256:1"
    "persistent_512_w8_t1:8:512:1"
    "persistent_1024_w8_t1:8:1024:1"
    "persistent_1024_w8_t2:8:1024:2"
    "persistent_1024_w16_t1:16:1024:1"
    "persistent_1024_w32_t1:32:1024:1"
    "crop_1024_w32_t1:32:1024:1:crop"
  )
fi
for spec in "${VARIANTS[@]}"; do
  [[ "$spec" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*:[1-9][0-9]*:[1-9][0-9]*:[1-9][0-9]*(:[A-Za-z0-9,-]+)?$ ]] || \
    fail "invalid --variant: $spec"
  IFS=: read -r _name workers block native _flags <<<"$spec"
  (( block % 256 == 0 )) || fail "variant block rows must align to 256: $spec"
  (( workers <= NCORES )) || fail "variant workers exceed --ncores: $spec"
  (( workers * native <= NCORES )) || fail "variant native thread budget exceeds --ncores: $spec"
done

[[ -f "$ZARR_PATH/zarr.json" || -f "$ZARR_PATH/.zgroup" ]] || fail "analysis Zarr metadata not found"
git -C "$PALETTE_REPO" rev-parse --git-dir >/dev/null 2>&1 || \
  fail "Palette checkout not found: $PALETTE_REPO"
[[ -x "$PALETTE_REPO/scripts/py" ]] || fail "Palette scripts/py is not executable"
[[ -f "$PALETTE_REPO/src/fisheye/diagnostics/benchmark_subject_shape_compute.py" ]] || \
  fail "Palette checkout lacks the subject-shape compute benchmark"
[[ -z "$(git -C "$PALETTE_REPO" status --porcelain)" ]] || fail "Palette checkout must be clean"

ZARR_PARENT="$(dirname -- "$ZARR_PATH")"
RECORDING_DIR="$(dirname -- "$ZARR_PARENT")"
if [[ -z "$LOG_DIR" ]]; then
  LOG_DIR="${RECORDING_DIR}/.processing_logs/subject_shape_benchmarks"
fi
RUN_DIR="${LOG_DIR}/${BENCHMARK_ID}"
[[ ! -e "$RUN_DIR" ]] || fail "submission directory already exists: $RUN_DIR"
mkdir -p "$RUN_DIR"

EXPECTED_COMMIT="$(git -C "$PALETTE_REPO" rev-parse HEAD)"
JOB_SCRIPT="${RUN_DIR}/run_subject_shape_compute_benchmark.sh"
REPORT_PATH="${RUN_DIR}/compute_benchmark_report.json"
STATUS_FILE="${RUN_DIR}/status.txt"
SUBMISSION_FILE="${RUN_DIR}/submission.txt"
VARIANTS_FILE="${RUN_DIR}/variants.txt"
printf '%s\n' "${VARIANTS[@]}" >"$VARIANTS_FILE"

q_repo="$(printf '%q' "$PALETTE_REPO")"
q_zarr="$(printf '%q' "$ZARR_PATH")"
q_refined="$(printf '%q' "$REFINED_RUN")"
q_benchmark="$(printf '%q' "$BENCHMARK_ID")"
q_expected="$(printf '%q' "$EXPECTED_COMMIT")"
q_report="$(printf '%q' "$REPORT_PATH")"
q_status="$(printf '%q' "$STATUS_FILE")"
q_variants="$(printf '%q' "$VARIANTS_FILE")"
q_scratch="$(printf '%q' "$SCRATCH_BASE")"

cat >"$JOB_SCRIPT" <<JOBSCRIPT
#!/usr/bin/env bash
set -euo pipefail
umask 0002
PALETTE_REPO=${q_repo}
ZARR_PATH=${q_zarr}
REFINED_RUN=${q_refined}
BENCHMARK_ID=${q_benchmark}
EXPECTED_COMMIT=${q_expected}
REPORT_PATH=${q_report}
STATUS_FILE=${q_status}
VARIANTS_FILE=${q_variants}
CONFIGURED_SCRATCH_BASE=${q_scratch}
SOURCE_START_ROW=${SOURCE_START_ROW}
ROW_COUNT=${ROW_COUNT}
KEEP_SCRATCH=${KEEP_SCRATCH}

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
scratch_root="\${scratch_base}/subject_shape_compute_\${BENCHMARK_ID}"
output_root="\${scratch_root}/outputs"
mkdir -p "\${scratch_root}"
cleanup() {
  if [[ "\${KEEP_SCRATCH}" != "1" ]]; then rm -rf -- "\${scratch_root}"; fi
}
trap cleanup EXIT

cmd=(
  scripts/py -m fisheye.diagnostics.benchmark_subject_shape_compute
  "\${ZARR_PATH}"
  --refined-run "\${REFINED_RUN}"
  --source-start-row "\${SOURCE_START_ROW}"
  --row-count "\${ROW_COUNT}"
  --output-root "\${output_root}"
  --report "\${REPORT_PATH}"
  --apply
)
while IFS= read -r variant; do [[ -n "\${variant}" ]] && cmd+=(--variant "\${variant}"); done <"\${VARIANTS_FILE}"
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
  printf 'zarr_path=%s\n' "\${ZARR_PATH}"
  printf 'refined_run=%s\n' "\${REFINED_RUN}"
  printf 'benchmark_id=%s\n' "\${BENCHMARK_ID}"
  printf 'source_start_row=%s\n' "\${SOURCE_START_ROW}"
  printf 'row_count=%s\n' "\${ROW_COUNT}"
  printf 'report=%s\n' "\${REPORT_PATH}"
  printf 'payload_returncode=%s\n' "\${payload_rc}"
} >"\${status_tmp}"
mv "\${status_tmp}" "\${STATUS_FILE}"
exit "\${payload_rc}"
JOBSCRIPT
chmod +x "$JOB_SCRIPT"

BSUB_ARGS=(
  -J "subject_shape_compute_${BENCHMARK_ID}"
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
