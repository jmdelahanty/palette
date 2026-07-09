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
BACKEND_A="decord_gpu"
BACKEND_B="pynvvc_nv12_rgb"
BATCH_SIZE=16
DEVICE="auto"
MAX_BBOX_DIFF="0.01"
MAX_SCORE_DIFF="0.05"
RUN_ID=""
RUN_LABEL=""
FRAMES=()
FORCE_FP32=0
SKIP_ENV_CHECK=0
DRY_RUN=0

usage() {
  cat <<'USAGE'
Usage: submit_detect_decode_backend_parity_bsub.sh --video PATH --model PATH --frames N [N ...] [options]

Submit a fixed-frame decode-backend prediction parity job to LSF.

Required:
  --video PATH             Input video path
  --model PATH             YOLO model path
  --frames N [N ...]       Explicit frame indices to compare

Options:
  --config PATH            Detect config path (default: configs/fisheye/yolo_detect_config.yaml)
  --log-dir PATH           Log/output directory (default: runs/diagnostics/detect_decode_backend_parity_bsub)
  --queue NAME             LSF queue (default: gpu_l4)
  --ncores N               CPU slots (default: 8)
  --mem-gb N               Memory request in GB (default: 120)
  --gpus N                 GPU count (default: 1)
  --walltime H:MM          LSF wall time (default: 1:00)
  --backend-a NAME         Reference backend (default: decord_gpu)
  --backend-b NAME         Candidate backend (default: pynvvc_nv12_rgb)
  --batch-size N           Frames per decode/preprocess batch (default: 16)
  --device {auto,cuda,cpu} Inference device (default: auto)
  --max-bbox-diff FLOAT    Fail when normalized bbox drift exceeds this (default: 0.01)
  --max-score-diff FLOAT   Fail when score drift exceeds this (default: 0.05)
  --force-fp32             Disable FP16
  --skip-env-check         Do not run cluster env preflight inside the LSF job
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
    --backend-a) BACKEND_A="$2"; shift 2;;
    --backend-b) BACKEND_B="$2"; shift 2;;
    --batch-size) BATCH_SIZE="$2"; shift 2;;
    --device) DEVICE="$2"; shift 2;;
    --max-bbox-diff) MAX_BBOX_DIFF="$2"; shift 2;;
    --max-score-diff) MAX_SCORE_DIFF="$2"; shift 2;;
    --force-fp32) FORCE_FP32=1; shift;;
    --skip-env-check) SKIP_ENV_CHECK=1; shift;;
    --run-id) RUN_ID="$2"; shift 2;;
    --run-label) RUN_LABEL="$2"; shift 2;;
    --dry-run) DRY_RUN=1; shift;;
    --frames)
      shift
      while [[ $# -gt 0 && "$1" != --* ]]; do
        FRAMES+=("$1")
        shift
      done
      ;;
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
if [[ "${#FRAMES[@]}" -eq 0 ]]; then
  echo "Missing required --frames values" >&2
  usage
  exit 2
fi

if [[ "$DRY_RUN" != "1" ]]; then
  if [[ ! -f "$VIDEO" ]]; then
    echo "Video file not found: $VIDEO" >&2
    exit 2
  fi
  if [[ ! -f "$MODEL" ]]; then
    echo "Model file not found: $MODEL" >&2
    exit 2
  fi
  if [[ ! -f "$CONFIG" ]]; then
    echo "Config file not found: $CONFIG" >&2
    exit 2
  fi
fi

if [[ -z "$LOG_DIR" ]]; then
  LOG_DIR="runs/diagnostics/detect_decode_backend_parity_bsub"
fi
if [[ -z "$RUN_ID" ]]; then
  RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
fi
if [[ -z "$RUN_LABEL" ]]; then
  stem="$(basename "$VIDEO")"
  RUN_LABEL="${stem%.*}"
fi
SAFE_LABEL="$(printf '%s' "$RUN_LABEL" | tr -c 'A-Za-z0-9_.-' '_')"
RUN_DIR="${LOG_DIR}/detect_decode_backend_parity_${RUN_ID}_${SAFE_LABEL}"

if [[ -e "$RUN_DIR" ]]; then
  echo "Run directory already exists: $RUN_DIR" >&2
  echo "Choose a different --run-id or --log-dir." >&2
  exit 2
fi
mkdir -p "$RUN_DIR"

PARITY_ARGS=(
  "$VIDEO"
  --model "$MODEL"
  --config "$CONFIG"
  --backend-a "$BACKEND_A"
  --backend-b "$BACKEND_B"
  --batch-size "$BATCH_SIZE"
  --device "$DEVICE"
  --max-bbox-diff "$MAX_BBOX_DIFF"
  --max-score-diff "$MAX_SCORE_DIFF"
  --fail-on-count-mismatch
  --frames "${FRAMES[@]}"
)
if [[ "$FORCE_FP32" == "1" ]]; then PARITY_ARGS+=(--force-fp32); fi

printf -v PARITY_ARGS_SHELL '%q ' "${PARITY_ARGS[@]}"

REQUIRE_PYNVVC=0
if [[ "$BACKEND_A" == pynvvc_* || "$BACKEND_B" == pynvvc_* ]]; then
  REQUIRE_PYNVVC=1
fi
RUN_ENV_CHECK=1
if [[ "$SKIP_ENV_CHECK" == "1" ]]; then
  RUN_ENV_CHECK=0
fi

JOB_SCRIPT="${RUN_DIR}/run_detect_decode_backend_parity.sh"
RUN_DIR_Q="$(printf '%q' "$RUN_DIR")"
SAFE_LABEL_Q="$(printf '%q' "$SAFE_LABEL")"
RUN_ENV_CHECK_Q="$(printf '%q' "$RUN_ENV_CHECK")"
REQUIRE_PYNVVC_Q="$(printf '%q' "$REQUIRE_PYNVVC")"

cat > "$JOB_SCRIPT" <<JOBSCRIPT
#!/usr/bin/env bash
set -euo pipefail

cd "$(pwd)"

RUN_DIR=${RUN_DIR_Q}
RUN_LABEL=${SAFE_LABEL_Q}
RUN_ENV_CHECK=${RUN_ENV_CHECK_Q}
REQUIRE_PYNVVC=${REQUIRE_PYNVVC_Q}
JOB_ID="\${LSB_JOBID:-manual}"
OUTPUT_JSON="\${RUN_DIR}/\${RUN_LABEL}.\${JOB_ID}.json"

scratch_user="\${USER:-\$(id -un)}"
if [[ -n "\${LSB_JOBID:-}" && -d "/scratch/\${scratch_user}" && -w "/scratch/\${scratch_user}" && -x "/scratch/\${scratch_user}" ]]; then
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
echo "run_env_check=\$RUN_ENV_CHECK"
echo "require_pynvvc=\$REQUIRE_PYNVVC"

if [[ "\$RUN_ENV_CHECK" == "1" ]]; then
  if [[ "\$REQUIRE_PYNVVC" == "1" ]]; then
    scripts/validate_cluster_palette_env.sh --require-pynvvc
  else
    scripts/validate_cluster_palette_env.sh
  fi
else
  echo "Skipping environment preflight by request."
fi

scripts/py -m fisheye.diagnostics.compare_detect_decode_backend_predictions ${PARITY_ARGS_SHELL}--output-json "\$OUTPUT_JSON"
JOBSCRIPT
chmod +x "$JOB_SCRIPT"

BSUB_ARGS=(
  -J "detect_decode_backend_parity_${SAFE_LABEL}"
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
echo "Environment check: run=${RUN_ENV_CHECK} require_pynvvc=${REQUIRE_PYNVVC}"
echo "Parity command: scripts/py -m fisheye.diagnostics.compare_detect_decode_backend_predictions ${PARITY_ARGS_SHELL}--output-json <json>"
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
