#!/usr/bin/env bash
set -euo pipefail
umask 0002

ZARR_PATH=""
SUBJECT_SHAPE_RUN=""
KEYPOINT_RUN=""
BENCHMARK_ID=""
PALETTE_REPO="${PALETTE_GROUPS_REPO:-/groups/johnson/johnsonlab/jeremy/gitrepos/palette}"
SUBMIT_HOST="${PALETTE_LSF_SUBMIT_HOST:-login1-citrus-poller}"
LOG_DIR=""
SCRATCH_BASE=""
QUEUE=""
NCORES=8
MEM_GB=32
WALLTIME="1:00"
ORDER="all_columns,semantic_16,semantic_16,all_columns"
KEEP_SCRATCH=0
SUBMIT=0

usage() {
  cat <<'USAGE'
Usage: submit_eye_angle_materialization_layout_benchmark_bsub.sh \
  --zarr PATH --subject-shape-run RUN --keypoint-run RUN \
  --benchmark-id ID [options]

Render or submit one full-duration, source-read-only eye-angle materialization
benchmark. The exact source inputs are staged once and both layouts execute in
A/B/B/A order inside one LSF allocation. All outputs are disposable and remain
on node-local scratch; only the report and LSF logs use shared storage.

Required:
  --zarr PATH                Palette analysis Zarr on shared storage
  --subject-shape-run RUN    Exact completed subject-shape input run
  --keypoint-run RUN         Exact completed refined-keypoint input run
  --benchmark-id ID          Immutable processing-log identifier

Options:
  --order CSV                Default: all_columns,semantic_16,semantic_16,all_columns
  --ncores N                 Dask worker and allocated slot count (default: 8)
  --mem-gb N                 Approximate total memory request (default: 32)
  --walltime H:MM            LSF walltime (default: 1:00)
  --palette-repo PATH        Clean cluster-visible Palette checkout
  --submit-host HOST         Citrus SSH poller when bsub is unavailable locally
  --log-dir PATH             Default: <recording>/.processing_logs/eye_angle_benchmarks
  --scratch-base PATH        Node-local scratch base override
  --queue NAME               LSF queue; default is cluster default
  --keep-scratch             Retain disposable staged inputs and trial outputs
  --submit                   Submit; otherwise render only
  -h, --help                 Show this help
USAGE
}

fail() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --zarr) ZARR_PATH="$2"; shift 2;;
    --subject-shape-run) SUBJECT_SHAPE_RUN="$2"; shift 2;;
    --keypoint-run) KEYPOINT_RUN="$2"; shift 2;;
    --benchmark-id) BENCHMARK_ID="$2"; shift 2;;
    --order) ORDER="$2"; shift 2;;
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
[[ -n "$SUBJECT_SHAPE_RUN" ]] || fail "--subject-shape-run is required"
[[ -n "$KEYPOINT_RUN" ]] || fail "--keypoint-run is required"
[[ -n "$BENCHMARK_ID" ]] || fail "--benchmark-id is required"
for value in "$SUBJECT_SHAPE_RUN" "$KEYPOINT_RUN" "$BENCHMARK_ID"; do
  [[ "$value" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || fail "unsafe identifier: $value"
done
[[ "$NCORES" =~ ^[1-9][0-9]*$ ]] || fail "--ncores must be positive"
[[ "$MEM_GB" =~ ^[1-9][0-9]*$ ]] || fail "--mem-gb must be positive"
[[ "$ORDER" =~ ^(all_columns|semantic_16)(,(all_columns|semantic_16))*$ ]] || \
  fail "--order contains an unsupported layout"
[[ -f "$ZARR_PATH/zarr.json" || -f "$ZARR_PATH/.zgroup" ]] || \
  fail "analysis Zarr metadata not found"
git -C "$PALETTE_REPO" rev-parse --git-dir >/dev/null 2>&1 || \
  fail "Palette checkout not found: $PALETTE_REPO"
[[ -x "$PALETTE_REPO/scripts/py" ]] || fail "Palette scripts/py is not executable"
[[ -f "$PALETTE_REPO/src/fisheye/diagnostics/benchmark_eye_angle_materialization_layout.py" ]] || \
  fail "Palette checkout lacks the eye-angle materialization benchmark"
[[ -z "$(git -C "$PALETTE_REPO" status --porcelain)" ]] || \
  fail "Palette checkout must be clean"

MEM_GB_PER_SLOT=$(( (MEM_GB + NCORES - 1) / NCORES ))
ZARR_PARENT="$(dirname -- "$ZARR_PATH")"
RECORDING_DIR="$(dirname -- "$ZARR_PARENT")"
if [[ -z "$LOG_DIR" ]]; then
  LOG_DIR="${RECORDING_DIR}/.processing_logs/eye_angle_benchmarks"
fi
RUN_DIR="${LOG_DIR}/${BENCHMARK_ID}"
[[ ! -e "$RUN_DIR" ]] || fail "submission directory already exists: $RUN_DIR"
mkdir -p "$RUN_DIR"

EXPECTED_COMMIT="$(git -C "$PALETTE_REPO" rev-parse HEAD)"
JOB_SCRIPT="${RUN_DIR}/run_eye_angle_materialization_layout_benchmark.sh"
REPORT_PATH="${RUN_DIR}/report.json"
STATUS_FILE="${RUN_DIR}/status.txt"
SUBMISSION_FILE="${RUN_DIR}/submission.txt"

q_repo="$(printf '%q' "$PALETTE_REPO")"
q_zarr="$(printf '%q' "$ZARR_PATH")"
q_subject_shape="$(printf '%q' "$SUBJECT_SHAPE_RUN")"
q_keypoint="$(printf '%q' "$KEYPOINT_RUN")"
q_benchmark="$(printf '%q' "$BENCHMARK_ID")"
q_expected="$(printf '%q' "$EXPECTED_COMMIT")"
q_report="$(printf '%q' "$REPORT_PATH")"
q_status="$(printf '%q' "$STATUS_FILE")"
q_scratch="$(printf '%q' "$SCRATCH_BASE")"
q_order="$(printf '%q' "$ORDER")"
q_requested_queue="$(printf '%q' "${QUEUE:-<cluster-default>}")"

cat >"$JOB_SCRIPT" <<JOBSCRIPT
#!/usr/bin/env bash
set -euo pipefail
umask 0002
PALETTE_REPO=${q_repo}
ZARR_PATH=${q_zarr}
SUBJECT_SHAPE_RUN=${q_subject_shape}
KEYPOINT_RUN=${q_keypoint}
BENCHMARK_ID=${q_benchmark}
EXPECTED_COMMIT=${q_expected}
REPORT_PATH=${q_report}
STATUS_FILE=${q_status}
CONFIGURED_SCRATCH_BASE=${q_scratch}
ORDER=${q_order}
REQUESTED_QUEUE=${q_requested_queue}
NCORES=${NCORES}
MEM_GB_PER_SLOT=${MEM_GB_PER_SLOT}
WALLTIME=${WALLTIME}
KEEP_SCRATCH=${KEEP_SCRATCH}

[[ -n "\${LSB_JOBID:-}" ]] || { printf 'Refusing execution outside LSF.\n' >&2; exit 2; }
cd "\${PALETTE_REPO}"
ACTUAL_COMMIT="\$(git rev-parse HEAD)"
[[ "\${ACTUAL_COMMIT}" == "\${EXPECTED_COMMIT}" ]] || { printf 'Palette commit mismatch.\n' >&2; exit 2; }
[[ -z "\$(git status --porcelain)" ]] || { printf 'Refusing dirty Palette checkout.\n' >&2; exit 2; }
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

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
scratch_root="\${scratch_base}/eye_angle_layout_\${BENCHMARK_ID}"
[[ ! -e "\${scratch_root}" ]] || { printf 'Refusing existing scratch root.\n' >&2; exit 2; }
export PYTHONPYCACHEPREFIX="\${scratch_root}.pycache"
cleanup() {
  if [[ "\${KEEP_SCRATCH}" != "1" ]]; then
    rm -rf -- "\${scratch_root}" "\${PYTHONPYCACHEPREFIX}"
  fi
}
trap cleanup EXIT

printf 'requested_queue=%s\n' "\${REQUESTED_QUEUE}"
printf 'effective_queue=%s\n' "\${LSB_QUEUE:-unknown}"
printf 'execution_host=%s\n' "\$(hostname)"
printf 'allocated_slots=%s\n' "\${LSB_DJOB_NUMPROC:-\${NCORES}}"

cmd=(
  scripts/py -m fisheye.diagnostics.benchmark_eye_angle_materialization_layout
  "\${ZARR_PATH}"
  --output-root "\${scratch_root}"
  --report "\${REPORT_PATH}"
  --benchmark-id "\${BENCHMARK_ID}"
  --subject-shape-run "\${SUBJECT_SHAPE_RUN}"
  --keypoint-run "\${KEYPOINT_RUN}"
  --order "\${ORDER}"
  --num-workers "\${NCORES}"
  --shard-workers "\${NCORES}"
  --native-threads 1
  --copy-backend rsync
  --apply
)
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
  printf 'requested_queue=%s\n' "\${REQUESTED_QUEUE}"
  printf 'effective_queue=%s\n' "\${LSB_QUEUE:-unknown}"
  printf 'allocated_slots=%s\n' "\${LSB_DJOB_NUMPROC:-\${NCORES}}"
  printf 'job_id=%s\n' "\${LSB_JOBID}"
  printf 'palette_commit=%s\n' "\${ACTUAL_COMMIT}"
  printf 'benchmark_id=%s\n' "\${BENCHMARK_ID}"
  printf 'order=%s\n' "\${ORDER}"
  printf 'report=%s\n' "\${REPORT_PATH}"
  printf 'payload_returncode=%s\n' "\${payload_rc}"
} >"\${status_tmp}"
mv "\${status_tmp}" "\${STATUS_FILE}"
exit "\${payload_rc}"
JOBSCRIPT
chmod +x "$JOB_SCRIPT"

BSUB_ARGS=(
  -J "eye_angle_layout_${BENCHMARK_ID}"
  -n "$NCORES"
  -W "$WALLTIME"
  -R "span[hosts=1] rusage[mem=${MEM_GB_PER_SLOT}G]"
  -oo "${RUN_DIR}/%J.out"
  -eo "${RUN_DIR}/%J.err"
)
if [[ -n "$QUEUE" ]]; then BSUB_ARGS+=(-q "$QUEUE"); fi
BSUB_COMMAND=(bsub "${BSUB_ARGS[@]}" bash "$JOB_SCRIPT")

printf 'mode=%s\n' "$([[ "$SUBMIT" == "1" ]] && printf submit || printf render-only)"
printf 'palette_commit=%s\n' "$EXPECTED_COMMIT"
printf 'benchmark_id=%s\n' "$BENCHMARK_ID"
printf 'requested_queue=%s\n' "${QUEUE:-<cluster-default>}"
printf 'requested_ncores=%s\n' "$NCORES"
printf 'requested_mem_gb_total=%s\n' "$MEM_GB"
printf 'requested_mem_gb_per_slot=%s\n' "$MEM_GB_PER_SLOT"
printf 'requested_walltime=%s\n' "$WALLTIME"
printf 'order=%s\n' "$ORDER"
printf 'run_dir=%s\n' "$RUN_DIR"
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
    printf 'requested_queue=%s\n' "${QUEUE:-<cluster-default>}"
    printf 'requested_ncores=%s\n' "$NCORES"
    printf 'requested_mem_gb_total=%s\n' "$MEM_GB"
    printf 'requested_mem_gb_per_slot=%s\n' "$MEM_GB_PER_SLOT"
    printf 'requested_walltime=%s\n' "$WALLTIME"
    printf 'order=%s\n' "$ORDER"
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
