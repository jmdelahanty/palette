#!/usr/bin/env bash
set -euo pipefail
umask 0002

PALETTE_REPO="${PALETTE_CLUSTER_WORKTREE:-$(pwd)}"
SUBMIT_HOST="${PALETTE_LSF_SUBMIT_HOST:-login1-citrus-poller}"
SOURCE_CROP_ZARR=""
CROP_RUN=""
SOURCE_KEYPOINT_ZARR=""
KEYPOINT_RUN=""
ROI_CACHE_MANIFEST=""
SUBJECT_MASK_MODEL=""
DESTINATION=""
BENCHMARK_ROOT=""
RUN_ID=""
QUEUE="gpu_l4"
NCORES=8
MEM_GB=64
WALLTIME="2:00"
BATCH_SIZE=128
FINALIZE_WORKERS=8
DEVICE="0"
HOST=""
RESUME_SCRATCH=""
RESUME_SOURCE_JOB_ID=""
RESUME_SOURCE_PALETTE_COMMIT=""
SUBMIT=0

usage() {
  cat <<'EOF'
Usage: submit_subject_mask_cache_pipeline_bsub.sh [required options]

Render or submit one commit-pinned cache -> raw subject masks -> refined dense
masks -> quality integration job. All large intermediates stay on node-local
scratch. Only terminal selector-ineligible benchmark stores are published.

Required:
  --palette-repo PATH
  --source-crop-zarr PATH
  --crop-run ID
  --source-keypoint-zarr PATH
  --keypoint-run ID
  --roi-cache-manifest PATH
  --subject-mask-model PATH
  --destination PATH
  --run-id ID

Options:
  --benchmark-root PATH     Default: destination grandparent
  --submit-host HOST        Default: login1-citrus-poller
  --queue NAME              Default: gpu_l4
  --ncores N                Default: 8
  --mem-gb N                Default: 64
  --walltime H:MM           Default: 2:00
  --batch-size N            Default: 128
  --finalize-workers N      Default: 8
  --device DEVICE           Default: 0
  --host HOST               Optional exact compute host
  --resume-scratch PATH     Resume a complete retained raw-inference scratch
  --resume-source-job-id ID Required with --resume-scratch
  --resume-source-commit ID Required with --resume-scratch
  --submit                  Submit; otherwise render only
  -h, --help
EOF
}

fail() { printf 'ERROR: %s\n' "$*" >&2; exit 2; }
positive() { [[ "$1" =~ ^[1-9][0-9]*$ ]]; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --palette-repo) PALETTE_REPO="$2"; shift 2 ;;
    --source-crop-zarr) SOURCE_CROP_ZARR="$2"; shift 2 ;;
    --crop-run) CROP_RUN="$2"; shift 2 ;;
    --source-keypoint-zarr) SOURCE_KEYPOINT_ZARR="$2"; shift 2 ;;
    --keypoint-run) KEYPOINT_RUN="$2"; shift 2 ;;
    --roi-cache-manifest) ROI_CACHE_MANIFEST="$2"; shift 2 ;;
    --subject-mask-model) SUBJECT_MASK_MODEL="$2"; shift 2 ;;
    --destination) DESTINATION="$2"; shift 2 ;;
    --benchmark-root) BENCHMARK_ROOT="$2"; shift 2 ;;
    --run-id) RUN_ID="$2"; shift 2 ;;
    --submit-host) SUBMIT_HOST="$2"; shift 2 ;;
    --queue) QUEUE="$2"; shift 2 ;;
    --ncores) NCORES="$2"; shift 2 ;;
    --mem-gb) MEM_GB="$2"; shift 2 ;;
    --walltime) WALLTIME="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --finalize-workers) FINALIZE_WORKERS="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --host) HOST="$2"; shift 2 ;;
    --resume-scratch) RESUME_SCRATCH="$2"; shift 2 ;;
    --resume-source-job-id) RESUME_SOURCE_JOB_ID="$2"; shift 2 ;;
    --resume-source-commit) RESUME_SOURCE_PALETTE_COMMIT="$2"; shift 2 ;;
    --submit) SUBMIT=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) fail "unknown argument: $1" ;;
  esac
done

[[ -x "$PALETTE_REPO/scripts/py" ]] || fail "invalid Palette worktree: $PALETTE_REPO"
[[ -d "$SOURCE_CROP_ZARR" ]] || fail "source crop Zarr not found"
[[ -d "$SOURCE_KEYPOINT_ZARR" ]] || fail "source keypoint Zarr not found"
[[ -f "$ROI_CACHE_MANIFEST" ]] || fail "ROI cache manifest not found"
[[ -f "$SUBJECT_MASK_MODEL" ]] || fail "subject-mask model not found"
[[ -n "$CROP_RUN" && -n "$KEYPOINT_RUN" ]] || fail "run IDs are required"
[[ -n "$DESTINATION" && -n "$RUN_ID" ]] || fail "destination and run ID are required"
[[ "$RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || fail "unsafe run ID"
positive "$NCORES" || fail "--ncores must be positive"
positive "$MEM_GB" || fail "--mem-gb must be positive"
positive "$BATCH_SIZE" || fail "--batch-size must be positive"
positive "$FINALIZE_WORKERS" || fail "--finalize-workers must be positive"
[[ ! -e "$DESTINATION" ]] || fail "destination already exists: $DESTINATION"
if [[ -n "$RESUME_SCRATCH" ]]; then
  [[ -n "$HOST" ]] || fail "--resume-scratch requires --host"
  [[ -n "$RESUME_SOURCE_JOB_ID" && -n "$RESUME_SOURCE_PALETTE_COMMIT" ]] || \
    fail "resume source job ID and commit are required"
  [[ "$RESUME_SCRATCH" == /scratch/*/[0-9]*/subject_mask_cache_pipeline_* ]] || \
    fail "unsafe resume scratch path"
fi

PALETTE_REPO="$(realpath "$PALETTE_REPO")"
SOURCE_CROP_ZARR="$(realpath "$SOURCE_CROP_ZARR")"
SOURCE_KEYPOINT_ZARR="$(realpath "$SOURCE_KEYPOINT_ZARR")"
ROI_CACHE_MANIFEST="$(realpath "$ROI_CACHE_MANIFEST")"
SUBJECT_MASK_MODEL="$(realpath "$SUBJECT_MASK_MODEL")"
DESTINATION="$(realpath -m "$DESTINATION")"
if [[ -z "$BENCHMARK_ROOT" ]]; then
  BENCHMARK_ROOT="$(dirname "$(dirname "$DESTINATION")")"
fi
BENCHMARK_ROOT="$(realpath -m "$BENCHMARK_ROOT")"
case "$DESTINATION/" in "$BENCHMARK_ROOT/"*) ;; *) fail "destination is outside benchmark root" ;; esac

EXPECTED_COMMIT="$(git -C "$PALETTE_REPO" rev-parse HEAD)"
[[ -z "$(git -C "$PALETTE_REPO" status --porcelain --untracked-files=all)" ]] || \
  fail "Palette worktree must be clean"
SUBMISSION_ROOT="$BENCHMARK_ROOT/submissions/$RUN_ID"
[[ ! -e "$SUBMISSION_ROOT" ]] || fail "submission root exists: $SUBMISSION_ROOT"

mkdir -p "$SUBMISSION_ROOT"
JOB_SCRIPT="$SUBMISSION_ROOT/job.sh"
STATUS_FILE="$SUBMISSION_ROOT/status.txt"
RESOURCE_FILE="$SUBMISSION_ROOT/resource_usage.txt"
STDOUT_LOG="$SUBMISSION_ROOT/%J.out"
STDERR_LOG="$SUBMISSION_ROOT/%J.err"

printf -v DRIVER_ARGS '%q ' \
  --source-crop-zarr "$SOURCE_CROP_ZARR" \
  --crop-run "$CROP_RUN" \
  --source-refined-keypoint-zarr "$SOURCE_KEYPOINT_ZARR" \
  --refined-keypoint-run "$KEYPOINT_RUN" \
  --roi-cache-manifest "$ROI_CACHE_MANIFEST" \
  --subject-mask-model "$SUBJECT_MASK_MODEL" \
  --destination "$DESTINATION" \
  --benchmark-root "$BENCHMARK_ROOT" \
  --batch-size "$BATCH_SIZE" \
  --finalize-workers "$FINALIZE_WORKERS" \
  --device "$DEVICE" \
  --keep-scratch
if [[ -n "$RESUME_SCRATCH" ]]; then
  printf -v RESUME_ARGS '%q ' \
    --resume-after-raw-inference \
    --resume-source-job-id "$RESUME_SOURCE_JOB_ID" \
    --resume-source-palette-commit "$RESUME_SOURCE_PALETTE_COMMIT"
  DRIVER_ARGS+="$RESUME_ARGS"
fi

cat >"$JOB_SCRIPT" <<EOF
#!/usr/bin/env bash
set -euo pipefail
umask 0002
[[ -n "\${LSB_JOBID:-}" ]] || { printf 'Refusing execution outside LSF.\n' >&2; exit 2; }
PALETTE_REPO=$(printf '%q' "$PALETTE_REPO")
EXPECTED_COMMIT=$(printf '%q' "$EXPECTED_COMMIT")
STATUS_FILE=$(printf '%q' "$STATUS_FILE")
RESOURCE_FILE=$(printf '%q' "$RESOURCE_FILE")
ACTUAL_COMMIT="\$(git -C "\$PALETTE_REPO" rev-parse HEAD)"
[[ "\$ACTUAL_COMMIT" == "\$EXPECTED_COMMIT" ]] || { printf 'Palette commit mismatch.\n' >&2; exit 2; }
[[ -z "\$(git -C "\$PALETTE_REPO" status --porcelain --untracked-files=all)" ]] || {
  printf 'Palette worktree is dirty.\n' >&2; exit 2;
}
scratch_user="\${USER:-\$(id -un)}"
resume_scratch=$(printf '%q' "$RESUME_SCRATCH")
if [[ -n "\$resume_scratch" ]]; then
  scratch_root="\$resume_scratch"
  [[ "\$scratch_root" == /scratch/*/[0-9]*/subject_mask_cache_pipeline_* ]] || {
    printf 'Unsafe resume scratch.\n' >&2; exit 2;
  }
  [[ -d "\$scratch_root" ]] || { printf 'Resume scratch is absent.\n' >&2; exit 2; }
else
  scratch_root="/scratch/\$scratch_user/\${LSB_JOBID}/subject_mask_cache_pipeline_$(printf '%q' "$RUN_ID")"
  [[ ! -e "\$scratch_root" ]] || { printf 'Scratch root exists.\n' >&2; exit 2; }
  mkdir -p "\$(dirname "\$scratch_root")"
fi
export PYTHONPYCACHEPREFIX="\$scratch_root.pycache"
export MPLBACKEND=Agg
export PALETTE_DISABLE_REGISTRY_WRITES=1
cd "\$PALETTE_REPO"
set +e
/usr/bin/time -v -o "\$RESOURCE_FILE" scripts/py -m \
  fisheye.diagnostics.benchmark_subject_mask_cache_pipeline \
  ${DRIVER_ARGS}--scratch-root "\$scratch_root"
payload_rc=\$?
set -e
status_tmp="\$STATUS_FILE.tmp.\$\$"
{
  if [[ "\$payload_rc" == 0 ]]; then printf 'status=complete\n'; else printf 'status=failed\n'; fi
  printf 'completed_at_utc=%s\n' "\$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'host=%s\n' "\$(hostname)"
  printf 'job_id=%s\n' "\$LSB_JOBID"
  printf 'palette_commit=%s\n' "\$ACTUAL_COMMIT"
  printf 'scratch_root=%s\n' "\$scratch_root"
  printf 'destination=%s\n' $(printf '%q' "$DESTINATION")
  printf 'payload_returncode=%s\n' "\$payload_rc"
} >"\$status_tmp"
mv "\$status_tmp" "\$STATUS_FILE"
if [[ "\$payload_rc" == 0 ]]; then
  rm -rf -- "\$scratch_root" "\$PYTHONPYCACHEPREFIX"
fi
exit "\$payload_rc"
EOF
chmod +x "$JOB_SCRIPT"

BSUB_COMMAND=(
  bsub -J "sm_cache_$RUN_ID" -q "$QUEUE" -n "$NCORES" -W "$WALLTIME"
  -M "$((MEM_GB * 1024))" -R "span[hosts=1] rusage[mem=${MEM_GB}G]"
  -gpu "num=1" -oo "$STDOUT_LOG" -eo "$STDERR_LOG" bash "$JOB_SCRIPT"
)
if [[ -n "$HOST" ]]; then
  BSUB_COMMAND=("${BSUB_COMMAND[@]:0:1}" -m "$HOST" "${BSUB_COMMAND[@]:1}")
fi

printf 'mode=%s\n' "$([[ "$SUBMIT" == 1 ]] && printf submit || printf render-only)"
printf 'palette_commit=%s\n' "$EXPECTED_COMMIT"
printf 'job_script=%s\n' "$JOB_SCRIPT"
printf 'destination=%s\n' "$DESTINATION"
printf 'bsub_command='; printf '%q ' "${BSUB_COMMAND[@]}"; printf '\n'
if [[ "$SUBMIT" == 1 ]]; then
  if command -v bsub >/dev/null 2>&1; then
    "${BSUB_COMMAND[@]}"
  else
    printf -v remote_command '%q ' "${BSUB_COMMAND[@]}"
    ssh "$SUBMIT_HOST" "$remote_command"
  fi
fi
