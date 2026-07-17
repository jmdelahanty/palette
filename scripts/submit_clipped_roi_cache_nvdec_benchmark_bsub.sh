#!/usr/bin/env bash
set -euo pipefail
umask 0002

ZARR_PATH=""
COLLECTION_ID=""
RECORDING_FRAME_INDEX=""
BENCHMARK_ID=""
PALETTE_REPO="${PALETTE_GROUPS_REPO:-/groups/johnson/johnsonlab/jeremy/gitrepos/palette}"
SUBMIT_HOST="${PALETTE_LSF_SUBMIT_HOST:-login1-citrus-poller}"
LOG_DIR=""
PUBLIC_ROOT=""
QUEUE="gpu_l4"
NCORES=8
MEM_GB=64
WALLTIME="2:00"
LIMIT_ROWS=8192
GPU_CHUNK_FRAMES=32
KEEP_PAYLOADS=0
SUBMIT=0
CLIP_IDS=()
SOURCE_VIDEOS=()
TRIAL_ORDERS=()

usage() {
  cat <<'USAGE'
Usage: submit_clipped_roi_cache_nvdec_benchmark_bsub.sh \
  --zarr PATH --collection-id ID --recording-frame-index PATH \
  --benchmark-id ID --clip-id ID --source-video PATH [options]

Render or submit a bounded production-path ROI-cache concurrency benchmark on
one NVIDIA L4. Repeat paired --clip-id/--source-video arguments; at least eight
clips are recommended when testing eight simultaneous decoder sessions.

The job runs every trial sequentially inside one L4 allocation. Each trial
uses the production bundle builder, writes caches to node-local scratch,
publishes them to an isolated NRS benchmark directory, validates them, records
nvidia-smi decoder telemetry, and removes the payloads. Reports and logs remain.

Required:
  --zarr PATH                    Canonical analysis Zarr (read-only)
  --collection-id ID            Finalized refined-detection collection
  --recording-frame-index PATH   Canonical recording frame-index parquet
  --benchmark-id ID              Immutable benchmark identifier
  --clip-id ID                   Selected clip; repeatable
  --source-video PATH            Source clip video paired by argument order

Options:
  --trial-order CSV              Concurrency order; repeat for another pass
                                 (default: 4,1,8,2,6 then 6,2,8,1,4)
  --limit-rows N                 Cache rows per clip per trial (default: 8192)
  --gpu-chunk-frames N           Production decoder batch size (default: 32)
  --ncores N                     Allocated CPU slots (default: 8; gpu_l4 queue ratio)
  --mem-gb N                     Approximate total memory (default: 64)
  --walltime H:MM                LSF walltime (default: 2:00)
  --queue NAME                   LSF queue (default: gpu_l4)
  --palette-repo PATH            Clean cluster-visible Palette checkout
  --submit-host HOST             Citrus poller host
  --log-dir PATH                 Default: <recording>/.processing_logs/roi_cache_nvdec_benchmarks
  --public-root PATH             Isolated temporary NRS root
  --keep-payloads                Do not remove bounded benchmark cache payloads
  --submit                       Submit through bsub/Citrus; otherwise render
  -h, --help                     Show this help
USAGE
}

fail() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --zarr) ZARR_PATH="$2"; shift 2;;
    --collection-id) COLLECTION_ID="$2"; shift 2;;
    --recording-frame-index) RECORDING_FRAME_INDEX="$2"; shift 2;;
    --benchmark-id) BENCHMARK_ID="$2"; shift 2;;
    --clip-id) CLIP_IDS+=("$2"); shift 2;;
    --source-video) SOURCE_VIDEOS+=("$2"); shift 2;;
    --trial-order) TRIAL_ORDERS+=("$2"); shift 2;;
    --limit-rows) LIMIT_ROWS="$2"; shift 2;;
    --gpu-chunk-frames) GPU_CHUNK_FRAMES="$2"; shift 2;;
    --ncores) NCORES="$2"; shift 2;;
    --mem-gb) MEM_GB="$2"; shift 2;;
    --walltime) WALLTIME="$2"; shift 2;;
    --queue) QUEUE="$2"; shift 2;;
    --palette-repo) PALETTE_REPO="$2"; shift 2;;
    --submit-host) SUBMIT_HOST="$2"; shift 2;;
    --log-dir) LOG_DIR="$2"; shift 2;;
    --public-root) PUBLIC_ROOT="$2"; shift 2;;
    --keep-payloads) KEEP_PAYLOADS=1; shift;;
    --submit) SUBMIT=1; shift;;
    -h|--help) usage; exit 0;;
    *) fail "unknown argument: $1";;
  esac
done

[[ -n "$ZARR_PATH" ]] || fail "--zarr is required"
[[ -n "$COLLECTION_ID" ]] || fail "--collection-id is required"
[[ -n "$RECORDING_FRAME_INDEX" ]] || fail "--recording-frame-index is required"
[[ -n "$BENCHMARK_ID" ]] || fail "--benchmark-id is required"
[[ "$BENCHMARK_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || fail "unsafe --benchmark-id"
[[ ${#CLIP_IDS[@]} -gt 0 ]] || fail "at least one --clip-id is required"
[[ ${#CLIP_IDS[@]} -eq ${#SOURCE_VIDEOS[@]} ]] || \
  fail "provide exactly one --source-video for every --clip-id"
for value_name in "ncores:$NCORES" "mem-gb:$MEM_GB" "limit-rows:$LIMIT_ROWS" "gpu-chunk-frames:$GPU_CHUNK_FRAMES"; do
  value="${value_name#*:}"
  [[ "$value" =~ ^[1-9][0-9]*$ ]] || fail "${value_name%%:*} must be positive"
done
[[ -f "$ZARR_PATH/zarr.json" || -f "$ZARR_PATH/.zgroup" ]] || fail "analysis Zarr metadata not found"
[[ -f "$RECORDING_FRAME_INDEX" ]] || fail "recording frame index not found"
for source_video in "${SOURCE_VIDEOS[@]}"; do [[ -f "$source_video" ]] || fail "source video not found: $source_video"; done
git -C "$PALETTE_REPO" rev-parse --git-dir >/dev/null 2>&1 || fail "Palette checkout not found: $PALETTE_REPO"
[[ -x "$PALETTE_REPO/scripts/py" ]] || fail "Palette scripts/py is not executable"
[[ -x "$PALETTE_REPO/scripts/submit_clipped_collection_flat_roi_cache_bundle_bsub.sh" ]] || \
  fail "Palette checkout lacks the production ROI-cache bundle runner"
[[ -f "$PALETTE_REPO/src/fisheye/diagnostics/benchmark_clipped_roi_cache_nvdec.py" ]] || \
  fail "Palette checkout lacks the NVDEC benchmark module"
[[ -z "$(git -C "$PALETTE_REPO" status --porcelain)" ]] || fail "Palette checkout must be clean"

if [[ ${#TRIAL_ORDERS[@]} -eq 0 ]]; then
  TRIAL_ORDERS=("4,1,8,2,6" "6,2,8,1,4")
fi
for order in "${TRIAL_ORDERS[@]}"; do
  [[ "$order" =~ ^[1-9][0-9]*(,[1-9][0-9]*)*$ ]] || fail "invalid --trial-order: $order"
done

MEM_GB_PER_SLOT=$(( (MEM_GB + NCORES - 1) / NCORES ))
ZARR_PARENT="$(dirname -- "$ZARR_PATH")"
RECORDING_DIR="$(dirname -- "$ZARR_PARENT")"
if [[ -z "$LOG_DIR" ]]; then
  LOG_DIR="${RECORDING_DIR}/.processing_logs/roi_cache_nvdec_benchmarks"
fi
RUN_DIR="${LOG_DIR}/${BENCHMARK_ID}"
[[ ! -e "$RUN_DIR" ]] || fail "benchmark directory already exists: $RUN_DIR"
if [[ -z "$PUBLIC_ROOT" ]]; then
  PUBLIC_ROOT="/nrs/johnson/palette_staging/benchmarks/clipped_roi_cache_nvdec/${BENCHMARK_ID}"
fi
case "$PUBLIC_ROOT" in
  /nrs/johnson/palette_staging/benchmarks/clipped_roi_cache_nvdec/*) ;;
  *) fail "--public-root must be under the isolated clipped_roi_cache_nvdec benchmark root";;
esac
mkdir -p "$RUN_DIR"

EXPECTED_COMMIT="$(git -C "$PALETTE_REPO" rev-parse HEAD)"
JOB_SCRIPT="${RUN_DIR}/run_clipped_roi_cache_nvdec_benchmark.sh"
REPORT_PATH="${RUN_DIR}/results/report.json"
STATUS_FILE="${RUN_DIR}/status.txt"
SUBMISSION_FILE="${RUN_DIR}/submission.txt"

printf '%s\n' "${CLIP_IDS[@]}" >"${RUN_DIR}/clip_ids.txt"
printf '%s\n' "${SOURCE_VIDEOS[@]}" >"${RUN_DIR}/source_videos.txt"
printf '%s\n' "${TRIAL_ORDERS[@]}" >"${RUN_DIR}/trial_orders.txt"

q_repo="$(printf '%q' "$PALETTE_REPO")"
q_zarr="$(printf '%q' "$ZARR_PATH")"
q_collection="$(printf '%q' "$COLLECTION_ID")"
q_frame_index="$(printf '%q' "$RECORDING_FRAME_INDEX")"
q_benchmark="$(printf '%q' "$BENCHMARK_ID")"
q_run_dir="$(printf '%q' "$RUN_DIR")"
q_public_root="$(printf '%q' "$PUBLIC_ROOT")"
q_expected="$(printf '%q' "$EXPECTED_COMMIT")"
q_status="$(printf '%q' "$STATUS_FILE")"

cat >"$JOB_SCRIPT" <<JOBSCRIPT
#!/usr/bin/env bash
set -euo pipefail
umask 0002
PALETTE_REPO=${q_repo}
ZARR_PATH=${q_zarr}
COLLECTION_ID=${q_collection}
RECORDING_FRAME_INDEX=${q_frame_index}
BENCHMARK_ID=${q_benchmark}
RUN_DIR=${q_run_dir}
PUBLIC_ROOT=${q_public_root}
EXPECTED_COMMIT=${q_expected}
STATUS_FILE=${q_status}
LIMIT_ROWS=${LIMIT_ROWS}
GPU_CHUNK_FRAMES=${GPU_CHUNK_FRAMES}
KEEP_PAYLOADS=${KEEP_PAYLOADS}

[[ -n "\${LSB_JOBID:-}" ]] || { printf 'Refusing execution outside LSF.\n' >&2; exit 2; }
cd "\${PALETTE_REPO}"
ACTUAL_COMMIT="\$(git rev-parse HEAD)"
[[ "\${ACTUAL_COMMIT}" == "\${EXPECTED_COMMIT}" ]] || { printf 'Palette commit mismatch.\n' >&2; exit 2; }
[[ -z "\$(git status --porcelain)" ]] || { printf 'Refusing dirty Palette checkout.\n' >&2; exit 2; }
export PYTHONPATH="\${PALETTE_REPO}/src:\${PALETTE_REPO}\${PYTHONPATH:+:\${PYTHONPATH}}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

mapfile -t CLIP_IDS <"\${RUN_DIR}/clip_ids.txt"
mapfile -t SOURCE_VIDEOS <"\${RUN_DIR}/source_videos.txt"
mapfile -t TRIAL_ORDERS <"\${RUN_DIR}/trial_orders.txt"
cmd=(
  scripts/py -m fisheye.diagnostics.benchmark_clipped_roi_cache_nvdec
  "\${ZARR_PATH}"
  --collection-id "\${COLLECTION_ID}"
  --recording-frame-index "\${RECORDING_FRAME_INDEX}"
  --benchmark-id "\${BENCHMARK_ID}"
  --run-dir "\${RUN_DIR}/results"
  --public-root "\${PUBLIC_ROOT}"
  --limit-rows "\${LIMIT_ROWS}"
  --gpu-chunk-frames "\${GPU_CHUNK_FRAMES}"
)
for clip_id in "\${CLIP_IDS[@]}"; do cmd+=(--clip-id "\${clip_id}"); done
for source_video in "\${SOURCE_VIDEOS[@]}"; do cmd+=(--source-video "\${source_video}"); done
for trial_order in "\${TRIAL_ORDERS[@]}"; do cmd+=(--trial-order "\${trial_order}"); done
if [[ "\${KEEP_PAYLOADS}" == "1" ]]; then cmd+=(--keep-payloads); fi

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
  printf 'benchmark_id=%s\n' "\${BENCHMARK_ID}"
  printf 'report=%s\n' "\${RUN_DIR}/results/report.json"
  printf 'payload_returncode=%s\n' "\${payload_rc}"
} >"\${status_tmp}"
mv "\${status_tmp}" "\${STATUS_FILE}"
exit "\${payload_rc}"
JOBSCRIPT
chmod +x "$JOB_SCRIPT"

BSUB_ARGS=(
  -J "roi_cache_nvdec_${BENCHMARK_ID}"
  -q "$QUEUE"
  -n "$NCORES"
  -W "$WALLTIME"
  -R "span[hosts=1] rusage[mem=${MEM_GB_PER_SLOT}G]"
  -gpu "num=1:mode=shared:j_exclusive=no"
  -oo "${RUN_DIR}/%J.out"
  -eo "${RUN_DIR}/%J.err"
)
BSUB_COMMAND=(bsub "${BSUB_ARGS[@]}" bash "$JOB_SCRIPT")

printf 'mode=%s\n' "$([[ "$SUBMIT" == "1" ]] && printf submit || printf render-only)"
printf 'palette_commit=%s\n' "$EXPECTED_COMMIT"
printf 'benchmark_id=%s\n' "$BENCHMARK_ID"
printf 'clips=%s\n' "${#CLIP_IDS[@]}"
printf 'trial_orders=%s\n' "${TRIAL_ORDERS[*]}"
printf 'limit_rows_per_clip=%s\n' "$LIMIT_ROWS"
printf 'run_dir=%s\n' "$RUN_DIR"
printf 'temporary_public_root=%s\n' "$PUBLIC_ROOT"
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
    printf 'report=%s\n' "${RUN_DIR}/results/report.json"
    printf 'palette_commit=%s\n' "$EXPECTED_COMMIT"
  } >"$SUBMISSION_FILE"
  printf 'job_id=%s\n' "$job_id"
  printf 'submission_file=%s\n' "$SUBMISSION_FILE"
fi
