#!/usr/bin/env bash
set -euo pipefail

ZARR=""
VIDEO=""
MODEL=""
CONFIG="configs/fisheye/yolo_detect_config.yaml"
OUTPUT_DIR=""
QUEUE="gpu_l4"
NCORES=8
MEM_GB=120
GPUS=1
WALLTIME="2:00"
DECODE_BACKEND="auto"
BATCH_SIZE=16
RUN_ID=""
RUN_LABEL=""
CONF=""
IOU=""
MAX_DET=""
RESIZE_DIMS=()
DRY_RUN=0
OVERWRITE_ARTIFACT=0

usage() {
  cat <<'USAGE'
Usage: submit_detect_artifact_bsub.sh --zarr PATH --video PATH --model PATH --output-dir PATH [options]

Submit one detection job that writes predictions to job-local scratch, packages
the completed detect run group as palette_run_group_artifact/*.tar.gz, and then
copies the tarball plus summary JSON back to the requested shared output dir.

Required:
  --zarr PATH              Target canonical analysis Zarr path
  --video PATH             Input camera video path
  --model PATH             YOLO model path
  --output-dir PATH        Shared destination for tarball + summary JSON

Options:
  --config PATH            Detect config path (default: configs/fisheye/yolo_detect_config.yaml)
  --queue NAME             LSF queue (default: gpu_l4)
  --ncores N               CPU slots (default: 8)
  --mem-gb N               Memory request in GB (default: 120)
  --gpus N                 GPU count (default: 1)
  --walltime H:MM          LSF wall time (default: 2:00)
  --decode-backend NAME    Backend passed to artifact runner (default: auto)
  --batch-size N           Frames per inference batch (default: 16)
  --conf FLOAT             Optional confidence threshold
  --iou FLOAT              Optional IoU threshold
  --max-det N              Optional max detections per frame
  --resize-dims H W        Optional canonical inference size [h w]
  --run-id ID              Stable run id instead of UTC timestamp
  --run-label LABEL        Artifact basename; default is analysis zarr stem
  --overwrite-artifact     Allow replacement of same scratch artifact path
  --dry-run                Print files and submit command; do not submit
  -h, --help               Show this message
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --zarr) ZARR="$2"; shift 2;;
    --video) VIDEO="$2"; shift 2;;
    --model) MODEL="$2"; shift 2;;
    --output-dir) OUTPUT_DIR="$2"; shift 2;;
    --config) CONFIG="$2"; shift 2;;
    --queue) QUEUE="$2"; shift 2;;
    --ncores) NCORES="$2"; shift 2;;
    --mem-gb) MEM_GB="$2"; shift 2;;
    --gpus) GPUS="$2"; shift 2;;
    --walltime) WALLTIME="$2"; shift 2;;
    --decode-backend) DECODE_BACKEND="$2"; shift 2;;
    --batch-size) BATCH_SIZE="$2"; shift 2;;
    --conf) CONF="$2"; shift 2;;
    --iou) IOU="$2"; shift 2;;
    --max-det) MAX_DET="$2"; shift 2;;
    --resize-dims) RESIZE_DIMS=("$2" "$3"); shift 3;;
    --run-id) RUN_ID="$2"; shift 2;;
    --run-label) RUN_LABEL="$2"; shift 2;;
    --overwrite-artifact) OVERWRITE_ARTIFACT=1; shift;;
    --dry-run) DRY_RUN=1; shift;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown arg: $1" >&2; usage; exit 2;;
  esac
done

if [[ -z "$ZARR" || -z "$VIDEO" || -z "$MODEL" || -z "$OUTPUT_DIR" ]]; then
  echo "Missing required --zarr, --video, --model, or --output-dir" >&2
  usage
  exit 2
fi

if [[ "$DRY_RUN" != "1" ]]; then
  [[ -d "$ZARR" ]] || { echo "Zarr not found: $ZARR" >&2; exit 2; }
  [[ -f "$VIDEO" ]] || { echo "Video not found: $VIDEO" >&2; exit 2; }
  [[ -f "$MODEL" ]] || { echo "Model not found: $MODEL" >&2; exit 2; }
  [[ -f "$CONFIG" ]] || { echo "Config not found: $CONFIG" >&2; exit 2; }
fi

if [[ -z "$RUN_ID" ]]; then
  RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
fi
if [[ -z "$RUN_LABEL" ]]; then
  stem="$(basename "$ZARR")"
  RUN_LABEL="${stem%.zarr}"
fi
SAFE_LABEL="$(printf '%s' "$RUN_LABEL" | tr -c 'A-Za-z0-9_.-' '_')"
RUN_DIR="${OUTPUT_DIR}/detect_artifact_${RUN_ID}_${SAFE_LABEL}"

if [[ -e "$RUN_DIR" ]]; then
  echo "Run directory already exists: $RUN_DIR" >&2
  echo "Choose a different --run-id or --output-dir." >&2
  exit 2
fi
mkdir -p "$RUN_DIR"

ARTIFACT_ARGS=(
  "$VIDEO"
  --target-zarr "$ZARR"
  --model "$MODEL"
  --config "$CONFIG"
  --decode-backend "$DECODE_BACKEND"
  --batch-size "$BATCH_SIZE"
)
if [[ -n "$CONF" ]]; then ARTIFACT_ARGS+=(--conf "$CONF"); fi
if [[ -n "$IOU" ]]; then ARTIFACT_ARGS+=(--iou "$IOU"); fi
if [[ -n "$MAX_DET" ]]; then ARTIFACT_ARGS+=(--max-det "$MAX_DET"); fi
if [[ "${#RESIZE_DIMS[@]}" -gt 0 ]]; then ARTIFACT_ARGS+=(--resize-dims "${RESIZE_DIMS[@]}"); fi
if [[ "$OVERWRITE_ARTIFACT" == "1" ]]; then ARTIFACT_ARGS+=(--overwrite-artifact); fi

printf -v ARTIFACT_ARGS_SHELL '%q ' "${ARTIFACT_ARGS[@]}"

JOB_SCRIPT="${RUN_DIR}/run_detect_artifact.sh"
RUN_DIR_Q="$(printf '%q' "$RUN_DIR")"
SAFE_LABEL_Q="$(printf '%q' "$SAFE_LABEL")"

cat > "$JOB_SCRIPT" <<JOBSCRIPT
#!/usr/bin/env bash
set -euo pipefail

cd "$(pwd)"

RUN_DIR=${RUN_DIR_Q}
RUN_LABEL=${SAFE_LABEL_Q}
JOB_ID="\${LSB_JOBID:-manual}"

scratch_user="\${USER:-\$(id -un)}"
if [[ -n "\${LSB_JOBID:-}" && -d "/scratch/\${scratch_user}" ]]; then
  SCRATCH_ROOT="/scratch/\${scratch_user}/\${LSB_JOBID}"
else
  SCRATCH_ROOT="\${TMPDIR:-/tmp}/palette_detect_artifact_\${JOB_ID}"
fi
export PALETTE_JOB_CACHE="\${SCRATCH_ROOT}/palette_cache"
export MPLBACKEND=Agg
mkdir -p "\$PALETTE_JOB_CACHE" "\$SCRATCH_ROOT" "\$RUN_DIR"

ARTIFACT_DIR="\${SCRATCH_ROOT}/palette_run_group_artifact"
WORK_DIR="\${SCRATCH_ROOT}/work"
SCRATCH_TARBALL="\${SCRATCH_ROOT}/\${RUN_LABEL}.\${JOB_ID}.tar.gz"
SUMMARY_JSON="\${RUN_DIR}/\${RUN_LABEL}.\${JOB_ID}.summary.json"

echo "repo=\$(pwd)"
echo "host=\$(hostname)"
echo "job_id=\$JOB_ID"
echo "scratch_root=\$SCRATCH_ROOT"
echo "palette_job_cache=\$PALETTE_JOB_CACHE"
echo "artifact_dir=\$ARTIFACT_DIR"
echo "scratch_tarball=\$SCRATCH_TARBALL"
echo "summary_json=\$SUMMARY_JSON"

scripts/py -m fisheye.utils.run_detection_artifact ${ARTIFACT_ARGS_SHELL}--artifact-dir "\$ARTIFACT_DIR" --work-dir "\$WORK_DIR" --tarball-output "\$SCRATCH_TARBALL" > "\$SUMMARY_JSON"

FINAL_TARBALL="\${RUN_DIR}/\${RUN_LABEL}.\${JOB_ID}.tar.gz"
TRANSFER_JSON="\${RUN_DIR}/\${RUN_LABEL}.\${JOB_ID}.transfer.json"
copy_start_ns="\$(date +%s%N)"
cp "\$SCRATCH_TARBALL" "\$FINAL_TARBALL"
copy_end_ns="\$(date +%s%N)"
copy_seconds="\$(awk -v s="\$copy_start_ns" -v e="\$copy_end_ns" 'BEGIN { printf "%.6f", (e - s) / 1000000000 }')"
cat > "\$TRANSFER_JSON" <<TRANSFERJSON
{
  "schema_version": 1,
  "job_id": "\$JOB_ID",
  "scratch_tarball": "\$SCRATCH_TARBALL",
  "final_tarball": "\$FINAL_TARBALL",
  "summary_json": "\$SUMMARY_JSON",
  "copy_tarball_seconds": \$copy_seconds
}
TRANSFERJSON
echo "final_tarball=\$FINAL_TARBALL"
echo "transfer_json=\$TRANSFER_JSON"
echo "copy_tarball_seconds=\$copy_seconds"
JOBSCRIPT
chmod +x "$JOB_SCRIPT"

BSUB_ARGS=(
  -J "detect_artifact_${SAFE_LABEL}"
  -n "$NCORES"
  -gpu "num=${GPUS}"
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

echo "Run dir: $RUN_DIR"
echo "Job script: $JOB_SCRIPT"
echo "Expected tarball: ${RUN_DIR}/${SAFE_LABEL}.<JOBID>.tar.gz"
echo "Expected summary: ${RUN_DIR}/${SAFE_LABEL}.<JOBID>.summary.json"
echo "Artifact command: scripts/py -m fisheye.utils.run_detection_artifact ${ARTIFACT_ARGS_SHELL}--artifact-dir <scratch>/palette_run_group_artifact --work-dir <scratch>/work --tarball-output <scratch>/<label>.<JOBID>.tar.gz"
echo "Submit command: $BSUB_CMD"

if [[ "$DRY_RUN" == "1" ]]; then
  echo "Dry run only; no submission."
  exit 0
fi

if ! command -v bsub >/dev/null 2>&1; then
  echo "bsub not found in PATH. Is this an LSF cluster?" >&2
  exit 2
fi

bsub "${BSUB_ARGS[@]}" bash "$JOB_SCRIPT"
