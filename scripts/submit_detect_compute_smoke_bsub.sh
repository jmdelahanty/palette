#!/usr/bin/env bash
set -euo pipefail

VIDEO=""
MODEL=""
CONFIG="configs/fisheye/yolo_detect_config.yaml"
LOG_DIR=""
QUEUE="gpu_l4"
NCORES=8
MEM_GB=120
GPUS=1
WALLTIME="1:00"
DECODE_BACKEND="auto"
BATCH_SIZE=16
MAX_BATCHES=20
MAX_FRAMES=0
START_FRAME=0
DEVICE="auto"
PIPELINE_MODE="sequential"
PIPELINE_DEPTH=2
RUN_ID=""
RUN_LABEL=""
CONF=""
IOU=""
MAX_DET=""
RESIZE_WIDTH=""
RESIZE_HEIGHT=""
FORCE_FP32=0
DRY_RUN=0

usage() {
  cat <<'USAGE'
Usage: submit_detect_compute_smoke_bsub.sh --video PATH --model PATH [options]

Submit a bounded detection compute smoke as a single LSF GPU job. The wrapper
writes a job script first so output JSON paths can safely include $LSB_JOBID.

Required:
  --video PATH             Input video path
  --model PATH             YOLO model path

Options:
  --config PATH            Detect config path (default: configs/fisheye/yolo_detect_config.yaml)
  --log-dir PATH           Log/output directory (default: runs/diagnostics/detect_compute_smoke_bsub)
  --queue NAME             LSF queue (default: gpu_l4)
  --ncores N               CPU slots (default: 8)
  --mem-gb N               Memory request in GB (default: 120)
  --gpus N                 GPU count (default: 1)
  --walltime H:MM          LSF wall time (default: 1:00)
  --decode-backend NAME    Backend passed to compute smoke
                            (default: auto; prefers pynvvc_luma_rgb on CUDA)
  --batch-size N           Frames per inference batch (default: 16)
  --max-batches N          Max batches to process (default: 20)
  --max-frames N           Max frames to process; 0 means use max-batches (default: 0)
  --start-frame N          First frame index (default: 0)
  --device {auto,cuda,cpu} Inference device (default: auto)
  --pipeline-mode NAME     Execution mode (default: sequential; experimental: producer)
  --pipeline-depth N       Decoded-batch queue depth for producer mode (default: 2)
  --conf FLOAT             Optional confidence threshold
  --iou FLOAT              Optional IoU threshold
  --max-det N              Optional max detections per frame
  --resize W H             Optional explicit resize before inference
  --force-fp32             Disable FP16
  --run-id ID              Stable run id instead of UTC timestamp
  --run-label LABEL        Output basename; default is video stem
  --dry-run                Print files and submit command; do not submit
  -h, --help               Show this message
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --video) VIDEO="$2"; shift 2;;
    --model) MODEL="$2"; shift 2;;
    --config) CONFIG="$2"; shift 2;;
    --log-dir) LOG_DIR="$2"; shift 2;;
    --queue) QUEUE="$2"; shift 2;;
    --ncores) NCORES="$2"; shift 2;;
    --mem-gb) MEM_GB="$2"; shift 2;;
    --gpus) GPUS="$2"; shift 2;;
    --walltime) WALLTIME="$2"; shift 2;;
    --decode-backend) DECODE_BACKEND="$2"; shift 2;;
    --batch-size) BATCH_SIZE="$2"; shift 2;;
    --max-batches) MAX_BATCHES="$2"; shift 2;;
    --max-frames) MAX_FRAMES="$2"; shift 2;;
    --start-frame) START_FRAME="$2"; shift 2;;
    --device) DEVICE="$2"; shift 2;;
    --pipeline-mode) PIPELINE_MODE="$2"; shift 2;;
    --pipeline-depth) PIPELINE_DEPTH="$2"; shift 2;;
    --conf) CONF="$2"; shift 2;;
    --iou) IOU="$2"; shift 2;;
    --max-det) MAX_DET="$2"; shift 2;;
    --resize) RESIZE_WIDTH="$2"; RESIZE_HEIGHT="$3"; shift 3;;
    --force-fp32) FORCE_FP32=1; shift;;
    --run-id) RUN_ID="$2"; shift 2;;
    --run-label) RUN_LABEL="$2"; shift 2;;
    --dry-run) DRY_RUN=1; shift;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown arg: $1" >&2; usage; exit 2;;
  esac
done

if [[ -z "$VIDEO" ]]; then
  echo "Missing required --video PATH" >&2
  usage
  exit 2
fi
if [[ -z "$MODEL" ]]; then
  echo "Missing required --model PATH" >&2
  usage
  exit 2
fi

if [[ "$DRY_RUN" != "1" ]]; then
  if [[ ! -f "$VIDEO" ]]; then
    echo "Video file not found: $VIDEO" >&2
    exit 2
  fi
  if [[ ! -f "$MODEL" ]]; then
    if [[ -d "$MODEL" ]]; then
      echo "Model path is a directory, expected a model file: $MODEL" >&2
      echo "Did the path get split across lines before weights/best.pt?" >&2
    else
      echo "Model file not found: $MODEL" >&2
    fi
    exit 2
  fi
  if [[ ! -f "$CONFIG" ]]; then
    echo "Config file not found: $CONFIG" >&2
    exit 2
  fi
fi

if [[ -z "$LOG_DIR" ]]; then
  LOG_DIR="runs/diagnostics/detect_compute_smoke_bsub"
fi
if [[ -z "$RUN_ID" ]]; then
  RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
fi
if [[ -z "$RUN_LABEL" ]]; then
  stem="$(basename "$VIDEO")"
  RUN_LABEL="${stem%.*}"
fi
SAFE_LABEL="$(printf '%s' "$RUN_LABEL" | tr -c 'A-Za-z0-9_.-' '_')"
RUN_DIR="${LOG_DIR}/detect_compute_smoke_${RUN_ID}_${SAFE_LABEL}"

if [[ -e "$RUN_DIR" ]]; then
  echo "Run directory already exists: $RUN_DIR" >&2
  echo "Choose a different --run-id or --log-dir." >&2
  exit 2
fi
mkdir -p "$RUN_DIR"

SMOKE_ARGS=(
  "$VIDEO"
  --model "$MODEL"
  --config "$CONFIG"
  --decode-backend "$DECODE_BACKEND"
  --start-frame "$START_FRAME"
  --max-frames "$MAX_FRAMES"
  --max-batches "$MAX_BATCHES"
  --batch-size "$BATCH_SIZE"
  --device "$DEVICE"
  --pipeline-mode "$PIPELINE_MODE"
  --pipeline-depth "$PIPELINE_DEPTH"
)
if [[ -n "$CONF" ]]; then SMOKE_ARGS+=(--conf "$CONF"); fi
if [[ -n "$IOU" ]]; then SMOKE_ARGS+=(--iou "$IOU"); fi
if [[ -n "$MAX_DET" ]]; then SMOKE_ARGS+=(--max-det "$MAX_DET"); fi
if [[ -n "$RESIZE_WIDTH" || -n "$RESIZE_HEIGHT" ]]; then
  if [[ -z "$RESIZE_WIDTH" || -z "$RESIZE_HEIGHT" ]]; then
    echo "--resize requires width and height" >&2
    exit 2
  fi
  SMOKE_ARGS+=(--resize "$RESIZE_WIDTH" "$RESIZE_HEIGHT")
fi
if [[ "$FORCE_FP32" == "1" ]]; then SMOKE_ARGS+=(--force-fp32); fi

printf -v SMOKE_ARGS_SHELL '%q ' "${SMOKE_ARGS[@]}"

JOB_SCRIPT="${RUN_DIR}/run_detect_compute_smoke.sh"
RUN_DIR_Q="$(printf '%q' "$RUN_DIR")"
SAFE_LABEL_Q="$(printf '%q' "$SAFE_LABEL")"

cat > "$JOB_SCRIPT" <<JOBSCRIPT
#!/usr/bin/env bash
set -euo pipefail

cd "$(pwd)"

RUN_DIR=${RUN_DIR_Q}
RUN_LABEL=${SAFE_LABEL_Q}
JOB_ID="\${LSB_JOBID:-manual}"
OUTPUT_JSON="\${RUN_DIR}/\${RUN_LABEL}.\${JOB_ID}.json"

scratch_user="\${USER:-\$(id -un)}"
if [[ -n "\${LSB_JOBID:-}" && -d "/scratch/\${scratch_user}" ]]; then
  export PALETTE_JOB_CACHE="/scratch/\${scratch_user}/\${LSB_JOBID}/palette_cache"
else
  export PALETTE_JOB_CACHE="\${TMPDIR:-/tmp}/palette_cache"
fi
export MPLBACKEND=Agg
mkdir -p "\$PALETTE_JOB_CACHE" "\$RUN_DIR"

echo "repo=\$(pwd)"
echo "host=\$(hostname)"
echo "job_id=\$JOB_ID"
echo "palette_job_cache=\$PALETTE_JOB_CACHE"
echo "output_json=\$OUTPUT_JSON"

scripts/py -m fisheye.diagnostics.detect_compute_smoke ${SMOKE_ARGS_SHELL}--output-json "\$OUTPUT_JSON"
JOBSCRIPT
chmod +x "$JOB_SCRIPT"

BSUB_ARGS=(
  -J "detect_compute_smoke_${SAFE_LABEL}"
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
echo "Expected JSON: ${RUN_DIR}/${SAFE_LABEL}.<JOBID>.json"
echo "Output log: ${RUN_DIR}/<JOBID>.out"
echo "Error log: ${RUN_DIR}/<JOBID>.err"
echo "Smoke command: scripts/py -m fisheye.diagnostics.detect_compute_smoke ${SMOKE_ARGS_SHELL}--output-json <json>"
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
