#!/usr/bin/env bash
set -euo pipefail
umask 0002

SOURCE_ANALYSIS_ZARR=""
CROP_RUN=""
SOURCE_VIDEO_PATH=""
MODEL_PATH=""
CACHE_MANIFEST_PATH=""
DESTINATION=""
CANARY_ID=""
PALETTE_REPO="${PALETTE_CLUSTER_WORKTREE:-$(pwd)}"
SUBMIT_HOST="${PALETTE_LSF_SUBMIT_HOST:-login1-citrus-poller}"
LOG_ROOT=""
QUEUE="gpu_l4"
NCORES=4
MEM_GB=64
WALLTIME="3:00"
BATCH_SIZE=256
CACHE_BATCH_SIZE=512
DEVICE="cuda:0"
SUBMIT=0

usage() {
  cat <<'USAGE'
Usage: submit_keypoint_v2_crop_canary_bsub.sh [required options]

Render or submit one commit-pinned representative crop-v2 -> raw keypoint-v2
-> keypoint-quality-v1 -> body-frame-v1 canary. Pixel cache construction,
inference, and derived publication run on node-local scratch. The durable flat
cache is published to NRS and the selector-ineligible artifacts are atomically
published below .palette_benchmarks.

Required:
  --source-analysis-zarr PATH
  --crop-run ID
  --source-video-path PATH
  --model-path PATH
  --cache-manifest-path PATH
  --destination PATH
  --canary-id ID

Options:
  --palette-repo PATH   Clean commit-pinned cluster-visible checkout
  --submit-host HOST    Citrus poller (default: login1-citrus-poller)
  --log-root PATH       Default: <destination-parent>/submissions
  --queue NAME          Default: gpu_l4
  --ncores N            Default: 4
  --mem-gb N            Default: 64
  --walltime H:MM       Default: 3:00
  --batch-size N        YOLO batch rows (default: 256)
  --cache-batch-size N  Cache materialization rows (default: 512)
  --device DEVICE       Default: cuda:0
  --submit              Submit; otherwise render only
  -h, --help
USAGE
}

fail() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --source-analysis-zarr) SOURCE_ANALYSIS_ZARR="$2"; shift 2;;
    --crop-run) CROP_RUN="$2"; shift 2;;
    --source-video-path) SOURCE_VIDEO_PATH="$2"; shift 2;;
    --model-path) MODEL_PATH="$2"; shift 2;;
    --cache-manifest-path) CACHE_MANIFEST_PATH="$2"; shift 2;;
    --destination) DESTINATION="$2"; shift 2;;
    --canary-id) CANARY_ID="$2"; shift 2;;
    --palette-repo) PALETTE_REPO="$2"; shift 2;;
    --submit-host) SUBMIT_HOST="$2"; shift 2;;
    --log-root) LOG_ROOT="$2"; shift 2;;
    --queue) QUEUE="$2"; shift 2;;
    --ncores) NCORES="$2"; shift 2;;
    --mem-gb) MEM_GB="$2"; shift 2;;
    --walltime) WALLTIME="$2"; shift 2;;
    --batch-size) BATCH_SIZE="$2"; shift 2;;
    --cache-batch-size) CACHE_BATCH_SIZE="$2"; shift 2;;
    --device) DEVICE="$2"; shift 2;;
    --submit) SUBMIT=1; shift;;
    -h|--help) usage; exit 0;;
    *) fail "unknown argument: $1";;
  esac
done

[[ -d "$SOURCE_ANALYSIS_ZARR" ]] || fail "source Zarr not found: $SOURCE_ANALYSIS_ZARR"
[[ -n "$CROP_RUN" ]] || fail "--crop-run is required"
[[ -f "$SOURCE_VIDEO_PATH" ]] || fail "source video not found: $SOURCE_VIDEO_PATH"
[[ -f "$MODEL_PATH" ]] || fail "model not found: $MODEL_PATH"
[[ -n "$CACHE_MANIFEST_PATH" ]] || fail "--cache-manifest-path is required"
[[ "$CACHE_MANIFEST_PATH" == /nrs/johnson/palette_staging/flat_roi_cache/* ]] || \
  fail "cache manifest must be below the NRS flat-cache namespace"
[[ -n "$DESTINATION" ]] || fail "--destination is required"
[[ "$DESTINATION" == */.palette_benchmarks/keypoint_storage/integration/* ]] || \
  fail "destination must be below the keypoint-storage integration namespace"
[[ ! -e "$DESTINATION" ]] || fail "destination already exists: $DESTINATION"
[[ "$CANARY_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || fail "unsafe canary ID"
for value in "$NCORES" "$MEM_GB" "$BATCH_SIZE" "$CACHE_BATCH_SIZE"; do
  [[ "$value" =~ ^[1-9][0-9]*$ ]] || fail "numeric options must be positive integers"
done
git -C "$PALETTE_REPO" rev-parse --git-dir >/dev/null 2>&1 || \
  fail "Palette checkout not found: $PALETTE_REPO"
[[ -x "$PALETTE_REPO/scripts/py" ]] || fail "Palette scripts/py is not executable"
[[ -f "$PALETTE_REPO/src/fisheye/diagnostics/publish_keypoint_v2_crop_canary.py" ]] || \
  fail "keypoint-v2 canary driver is missing from Palette checkout"
[[ -z "$(git -C "$PALETTE_REPO" status --porcelain --untracked-files=all)" ]] || \
  fail "Palette checkout must be clean"

if [[ -z "$LOG_ROOT" ]]; then
  LOG_ROOT="$(dirname -- "$DESTINATION")/submissions"
fi
RUN_DIR="${LOG_ROOT%/}/${CANARY_ID}"
[[ ! -e "$RUN_DIR" ]] || fail "submission directory exists: $RUN_DIR"
mkdir -p "$RUN_DIR"

EXPECTED_COMMIT="$(git -C "$PALETTE_REPO" rev-parse HEAD)"
JOB_SCRIPT="$RUN_DIR/run_keypoint_v2_canary.sh"
STATUS_FILE="$RUN_DIR/status.txt"
RESOURCE_FILE="$RUN_DIR/resource_usage.txt"
SUBMISSION_FILE="$RUN_DIR/submission.txt"

q_repo="$(printf '%q' "$PALETTE_REPO")"
q_source="$(printf '%q' "$SOURCE_ANALYSIS_ZARR")"
q_crop="$(printf '%q' "$CROP_RUN")"
q_video="$(printf '%q' "$SOURCE_VIDEO_PATH")"
q_model="$(printf '%q' "$MODEL_PATH")"
q_cache="$(printf '%q' "$CACHE_MANIFEST_PATH")"
q_destination="$(printf '%q' "$DESTINATION")"
q_canary="$(printf '%q' "$CANARY_ID")"
q_expected="$(printf '%q' "$EXPECTED_COMMIT")"
q_status="$(printf '%q' "$STATUS_FILE")"
q_resource="$(printf '%q' "$RESOURCE_FILE")"
q_device="$(printf '%q' "$DEVICE")"

cat >"$JOB_SCRIPT" <<JOBSCRIPT
#!/usr/bin/env bash
set -euo pipefail
umask 0002
PALETTE_REPO=${q_repo}
SOURCE_ANALYSIS_ZARR=${q_source}
CROP_RUN=${q_crop}
SOURCE_VIDEO_PATH=${q_video}
MODEL_PATH=${q_model}
CACHE_MANIFEST_PATH=${q_cache}
DESTINATION=${q_destination}
CANARY_ID=${q_canary}
EXPECTED_COMMIT=${q_expected}
STATUS_FILE=${q_status}
RESOURCE_FILE=${q_resource}
DEVICE=${q_device}

[[ -n "\${LSB_JOBID:-}" ]] || { printf 'Refusing execution outside LSF.\n' >&2; exit 2; }
cd "\${PALETTE_REPO}"
ACTUAL_COMMIT="\$(git rev-parse HEAD)"
[[ "\${ACTUAL_COMMIT}" == "\${EXPECTED_COMMIT}" ]] || { printf 'Palette commit mismatch.\n' >&2; exit 2; }
[[ -z "\$(git status --porcelain --untracked-files=all)" ]] || { printf 'Palette checkout is dirty.\n' >&2; exit 2; }

if [[ -d "/scratch/\${USER}" && -w "/scratch/\${USER}" ]]; then
  scratch_base="/scratch/\${USER}/\${LSB_JOBID}"
elif [[ -n "\${TMPDIR:-}" && -d "\${TMPDIR}" && -w "\${TMPDIR}" ]]; then
  scratch_base="\${TMPDIR}/palette/\${LSB_JOBID}"
else
  printf 'No writable node-local scratch root is available.\n' >&2
  exit 2
fi
case "\${scratch_base}" in /groups/*|/nrs/*) printf 'Refusing shared scratch.\n' >&2; exit 2;; esac
scratch_root="\${scratch_base}/keypoint_v2_crop_\${CANARY_ID}"
[[ ! -e "\${scratch_root}" ]] || { printf 'Scratch root exists.\n' >&2; exit 2; }
# Python may populate its bytecode cache while importing the canary module,
# before the driver gets a chance to assert that scratch_root is new.  Keep
# bytecode in a sibling so that importing cannot create the driver-owned root.
export PYTHONPYCACHEPREFIX="\${scratch_root}.pycache"
export MPLBACKEND=Agg
export PALETTE_DISABLE_REGISTRY_WRITES=1

cmd=(
  scripts/py -m fisheye.diagnostics.publish_keypoint_v2_crop_canary
  --source-analysis-zarr "\${SOURCE_ANALYSIS_ZARR}"
  --crop-run "\${CROP_RUN}"
  --source-video-path "\${SOURCE_VIDEO_PATH}"
  --model-path "\${MODEL_PATH}"
  --cache-manifest-path "\${CACHE_MANIFEST_PATH}"
  --destination "\${DESTINATION}"
  --scratch-root "\${scratch_root}"
  --device "\${DEVICE}"
  --batch-size ${BATCH_SIZE}
  --cache-batch-size ${CACHE_BATCH_SIZE}
)
set +e
/usr/bin/time -v -o "\${RESOURCE_FILE}" "\${cmd[@]}"
payload_rc=\$?
set -e

status_tmp="\${STATUS_FILE}.tmp.\$\$"
{
  if [[ "\${payload_rc}" == "0" ]]; then printf 'status=complete\n'; else printf 'status=failed\n'; fi
  printf 'completed_at_utc=%s\n' "\$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'host=%s\n' "\$(hostname)"
  printf 'job_id=%s\n' "\${LSB_JOBID}"
  printf 'palette_commit=%s\n' "\${ACTUAL_COMMIT}"
  printf 'destination=%s\n' "\${DESTINATION}"
  printf 'cache_manifest=%s\n' "\${CACHE_MANIFEST_PATH}"
  printf 'scratch_root=%s\n' "\${scratch_root}"
  printf 'resource_usage=%s\n' "\${RESOURCE_FILE}"
  printf 'payload_returncode=%s\n' "\${payload_rc}"
} >"\${status_tmp}"
mv "\${status_tmp}" "\${STATUS_FILE}"
if [[ "\${payload_rc}" == "0" ]]; then
  rm -rf -- "\${scratch_root}" "\${PYTHONPYCACHEPREFIX}"
fi
exit "\${payload_rc}"
JOBSCRIPT
chmod +x "$JOB_SCRIPT"

BSUB_COMMAND=(
  bsub
  -J "keypoint_v2_${CANARY_ID}"
  -q "$QUEUE"
  -n "$NCORES"
  -W "$WALLTIME"
  -R "span[hosts=1] rusage[mem=${MEM_GB}G]"
  -gpu "num=1"
  -oo "${RUN_DIR}/%J.out"
  -eo "${RUN_DIR}/%J.err"
  bash "$JOB_SCRIPT"
)

printf 'mode=%s\n' "$([[ "$SUBMIT" == "1" ]] && printf submit || printf render-only)"
printf 'palette_commit=%s\n' "$EXPECTED_COMMIT"
printf 'run_dir=%s\n' "$RUN_DIR"
printf 'job_script=%s\n' "$JOB_SCRIPT"
printf 'status_file=%s\n' "$STATUS_FILE"
printf 'cache_manifest=%s\n' "$CACHE_MANIFEST_PATH"
printf 'destination=%s\n' "$DESTINATION"
printf 'bsub_command='; printf '%q ' "${BSUB_COMMAND[@]}"; printf '\n'

if [[ "$SUBMIT" == "1" ]]; then
  if command -v bsub >/dev/null 2>&1; then
    submit_mode="local_bsub"
    submit_output="$("${BSUB_COMMAND[@]}")"
  else
    printf -v remote_command '%q ' "${BSUB_COMMAND[@]}"
    submit_mode="ssh_bsub"
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
    printf 'status_file=%s\n' "$STATUS_FILE"
    printf 'resource_file=%s\n' "$RESOURCE_FILE"
    printf 'palette_commit=%s\n' "$EXPECTED_COMMIT"
  } >"$SUBMISSION_FILE"
  printf 'job_id=%s\n' "$job_id"
  printf 'submission_file=%s\n' "$SUBMISSION_FILE"
fi
