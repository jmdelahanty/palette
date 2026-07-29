#!/usr/bin/env bash
set -euo pipefail
umask 0002

SOURCE_ANALYSIS_ZARR=""
RECORDING_DIR=""
CROP_META=""
CROP_VIDEO=""
KEYPOINT_MODEL=""
SUBJECT_MASK_MODEL=""
PALETTE_REPO="${PALETTE_CLUSTER_WORKTREE:-$(pwd)}"
OUTPUT_ROOT="/groups/johnson/johnsonlab/jeremy/recordings/.palette_benchmarks/crop_pixel_materialization/workflows"
RUN_ID=""
ROW_COUNT=2048
BATCH_ROWS=256
MODEL_BATCH_ROWS=64
QUEUE="gpu_l4"
NCORES=4
MEM_GB=48
WALLTIME="2:00"
DEVICE="cuda:0"
INTERFACE_ONLY=0
APPLY=0

usage() {
  cat <<'EOF'
Usage: submit_crop_pixel_materialization_canary_bsub.sh [options]

Submit one selector-ineligible crop-pixel integration canary. The job creates a
modern acquisition crop snapshot, materializes a bounded keyed work package,
and runs the real keypoint and subject-mask shard consumers entirely on
node-local scratch. Only the receipt and logs are published to the benchmark
namespace.

Required:
  --source-analysis-zarr PATH  Read-only source archive used for recording identity
  --recording-dir PATH         Recording directory containing Orange sidecars
  --crop-meta PATH             Acquisition crop metadata CSV
  --crop-video PATH            Acquisition crop MP4
  --keypoint-model PATH        YOLO pose checkpoint (unless --interface-only)
  --subject-mask-model PATH    Unified subject-mask checkpoint (unless --interface-only)

Options:
  --palette-repo PATH          Commit-pinned compute-visible Palette checkout
  --output-root PATH           Benchmark namespace root
  --run-id ID                  Immutable workflow id (default: UTC timestamp)
  --row-count N                Contiguous crop rows to materialize (default: 2048)
  --batch-rows N               Pixel materialization/read batch (default: 256)
  --model-batch-rows N         Inference batch (default: 64)
  --queue NAME                 LSF queue (default: gpu_l4)
  --ncores N                   CPU slots (default: 4)
  --mem-gb N                   Memory request (default: 48)
  --walltime H:MM              Walltime (default: 2:00)
  --device DEVICE              Torch device (default: cuda:0)
  --interface-only             Validate both package consumer boundaries without models
  --apply                      Submit; without this flag print the exact plan only
  -h, --help                   Show this help
EOF
}

fail() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --source-analysis-zarr) SOURCE_ANALYSIS_ZARR="$2"; shift 2 ;;
    --recording-dir) RECORDING_DIR="$2"; shift 2 ;;
    --crop-meta) CROP_META="$2"; shift 2 ;;
    --crop-video) CROP_VIDEO="$2"; shift 2 ;;
    --keypoint-model) KEYPOINT_MODEL="$2"; shift 2 ;;
    --subject-mask-model) SUBJECT_MASK_MODEL="$2"; shift 2 ;;
    --palette-repo) PALETTE_REPO="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --run-id) RUN_ID="$2"; shift 2 ;;
    --row-count) ROW_COUNT="$2"; shift 2 ;;
    --batch-rows) BATCH_ROWS="$2"; shift 2 ;;
    --model-batch-rows) MODEL_BATCH_ROWS="$2"; shift 2 ;;
    --queue) QUEUE="$2"; shift 2 ;;
    --ncores) NCORES="$2"; shift 2 ;;
    --mem-gb) MEM_GB="$2"; shift 2 ;;
    --walltime) WALLTIME="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --interface-only) INTERFACE_ONLY=1; shift ;;
    --apply) APPLY=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) fail "unknown argument: $1" ;;
  esac
done

for pair in \
  "source-analysis-zarr:$SOURCE_ANALYSIS_ZARR" \
  "recording-dir:$RECORDING_DIR" \
  "crop-meta:$CROP_META" \
  "crop-video:$CROP_VIDEO"; do
  name="${pair%%:*}"
  value="${pair#*:}"
  [[ -n "$value" ]] || fail "--$name is required"
  [[ -e "$value" ]] || fail "--$name does not exist: $value"
done
if [[ "$INTERFACE_ONLY" -eq 0 ]]; then
  [[ -f "$KEYPOINT_MODEL" ]] || fail "--keypoint-model is required and must exist"
  [[ -f "$SUBJECT_MASK_MODEL" ]] || fail "--subject-mask-model is required and must exist"
fi
[[ "$ROW_COUNT" =~ ^[1-9][0-9]*$ ]] || fail "--row-count must be positive"
[[ "$BATCH_ROWS" =~ ^[1-9][0-9]*$ ]] || fail "--batch-rows must be positive"
[[ "$MODEL_BATCH_ROWS" =~ ^[1-9][0-9]*$ ]] || fail "--model-batch-rows must be positive"
[[ "$NCORES" =~ ^[1-9][0-9]*$ ]] || fail "--ncores must be positive"
[[ "$MEM_GB" =~ ^[1-9][0-9]*$ ]] || fail "--mem-gb must be positive"

PALETTE_REPO="$(realpath -- "$PALETTE_REPO")"
[[ -x "$PALETTE_REPO/scripts/py" ]] || fail "Palette checkout lacks scripts/py: $PALETTE_REPO"
EXPECTED_COMMIT="$(git -C "$PALETTE_REPO" rev-parse HEAD)"
[[ -z "$(git -C "$PALETTE_REPO" status --porcelain)" ]] || \
  fail "Palette checkout must be clean"
if [[ -z "$RUN_ID" ]]; then
  RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
fi
SAFE_RUN_ID="$(printf '%s' "$RUN_ID" | tr -c 'A-Za-z0-9_.-' '_')"
RUN_DIR="${OUTPUT_ROOT}/${SAFE_RUN_ID}"
[[ ! -e "$RUN_DIR" ]] || fail "benchmark run directory exists: $RUN_DIR"

DRIVER_ARGS=(
  --source-analysis-zarr "$SOURCE_ANALYSIS_ZARR"
  --recording-dir "$RECORDING_DIR"
  --crop-meta "$CROP_META"
  --crop-video "$CROP_VIDEO"
  --row-count "$ROW_COUNT"
  --batch-rows "$BATCH_ROWS"
  --model-batch-rows "$MODEL_BATCH_ROWS"
  --device "$DEVICE"
)
if [[ "$INTERFACE_ONLY" -eq 0 ]]; then
  DRIVER_ARGS+=(
    --keypoint-model "$KEYPOINT_MODEL"
    --subject-mask-model "$SUBJECT_MASK_MODEL"
  )
fi
printf -v DRIVER_ARGS_SHELL '%q ' "${DRIVER_ARGS[@]}"

printf 'palette_repo=%s\n' "$PALETTE_REPO"
printf 'palette_commit=%s\n' "$EXPECTED_COMMIT"
printf 'run_dir=%s\n' "$RUN_DIR"
printf 'row_count=%s\n' "$ROW_COUNT"
printf 'model_consumers=%s\n' "$((1 - INTERFACE_ONLY))"
if [[ "$APPLY" -eq 0 ]]; then
  printf 'dry_run=true; pass --apply to submit\n'
  printf 'command=%q -m fisheye.diagnostics.benchmark_crop_pixel_materialization_consumers %s\n' \
    "$PALETTE_REPO/scripts/py" "$DRIVER_ARGS_SHELL"
  exit 0
fi

mkdir -p "$RUN_DIR"
JOB_SCRIPT="$RUN_DIR/run_canary.sh"
STDOUT_LOG="$RUN_DIR/lsf.stdout.log"
STDERR_LOG="$RUN_DIR/lsf.stderr.log"
REPORT="$RUN_DIR/receipt.json"

cat >"$JOB_SCRIPT" <<EOF
#!/usr/bin/env bash
set -euo pipefail
umask 0002

PALETTE_REPO=$(printf '%q' "$PALETTE_REPO")
EXPECTED_COMMIT=$(printf '%q' "$EXPECTED_COMMIT")
RUN_DIR=$(printf '%q' "$RUN_DIR")
REPORT=$(printf '%q' "$REPORT")
actual_commit="\$(git -C "\$PALETTE_REPO" rev-parse HEAD)"
[[ "\$actual_commit" == "\$EXPECTED_COMMIT" ]] || {
  echo "Palette checkout moved: expected \$EXPECTED_COMMIT, found \$actual_commit" >&2
  exit 2
}
[[ -z "\$(git -C "\$PALETTE_REPO" status --porcelain)" ]] || {
  echo "Palette checkout is dirty" >&2
  exit 2
}

scratch_user="\${USER:-\$(id -un)}"
if [[ -d "/scratch/\$scratch_user" && -w "/scratch/\$scratch_user" ]]; then
  scratch_base="/scratch/\$scratch_user"
else
  scratch_base="\${TMPDIR:-/tmp}"
fi
LOCAL_SCRATCH="\$(mktemp -d "\$scratch_base/palette-crop-pixel-canary.XXXXXX")"
cleanup() {
  case "\$LOCAL_SCRATCH" in
    /scratch/"\$scratch_user"/palette-crop-pixel-canary.*|/tmp/palette-crop-pixel-canary.*)
      rm -rf -- "\$LOCAL_SCRATCH"
      ;;
    *)
      echo "Refusing unexpected scratch cleanup path: \$LOCAL_SCRATCH" >&2
      ;;
  esac
}
trap cleanup EXIT

cd "\$PALETTE_REPO"
export MPLBACKEND=Agg
export PALETTE_DISABLE_REGISTRY_WRITES=1
echo "host=\$(hostname)"
echo "lsb_jobid=\${LSB_JOBID:-none}"
echo "palette_repo=\$PALETTE_REPO"
echo "palette_commit=\$EXPECTED_COMMIT"
echo "local_scratch=\$LOCAL_SCRATCH"
echo "report=\$REPORT"

scripts/py -m fisheye.diagnostics.benchmark_crop_pixel_materialization_consumers \
  ${DRIVER_ARGS_SHELL}--scratch-root "\$LOCAL_SCRATCH/work" --output-json "\$REPORT"
EOF
chmod +x "$JOB_SCRIPT"

bsub \
  -q "$QUEUE" \
  -n "$NCORES" \
  -M "$((MEM_GB * 1024))" \
  -R "rusage[mem=${MEM_GB}GB] span[hosts=1]" \
  -gpu "num=1" \
  -W "$WALLTIME" \
  -J "crop_pixel_${SAFE_RUN_ID}" \
  -oo "$STDOUT_LOG" \
  -eo "$STDERR_LOG" \
  "$JOB_SCRIPT"

printf 'submitted=true\n'
printf 'job_script=%s\n' "$JOB_SCRIPT"
printf 'receipt=%s\n' "$REPORT"
