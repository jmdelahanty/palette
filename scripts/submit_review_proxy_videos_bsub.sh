#!/usr/bin/env bash
set -euo pipefail

RECORDING_DIR=""
OUTPUT_DIR=""
PROXY_RUN_ID=""
LOG_DIR=""
QUEUE="gpu_l4"
NCORES=4
MEM_GB=32
GPUS=1
WALLTIME="2:00"
PROXY_WIDTH=1024
PROXY_HEIGHT=1024
ENCODER="h264_nvenc"
PRESET="veryfast"
CRF=23
HWACCEL="cuda"
SCALE_FLAGS="bilinear"
FFMPEG_BIN="ffmpeg"
FFPROBE_BIN="ffprobe"
LIMIT=""
NO_PROBE=0
OVERWRITE=0
DRY_RUN=0
RUN_ID=""
CLIP_IDS=()
CAMERA_SERIALS=()

usage() {
  cat <<'USAGE'
Usage: submit_review_proxy_videos_bsub.sh RECORDING_DIR [options]

Submit one LSF job that builds browser-review proxy MP4s for a clipped
recording. The job runs fisheye.utils.build_review_proxy_videos sequentially on
the compute node; it does not fan out one job per clip.

Required:
  RECORDING_DIR                 Recording folder containing recording_clip_index.json

Options:
  --output-dir PATH             Proxy output dir (default: <recording>/derived/review_proxy/video_detect/<proxy-run-id>)
  --proxy-run-id ID             Stable proxy run id (default: UTC timestamp)
  --queue NAME                  LSF queue (default: gpu_l4)
  --ncores N                    CPU slots (default: 4)
  --mem-gb N                    Memory request in GB (default: 32)
  --gpus N                      GPU count; 0 omits -gpu (default: 1)
  --walltime H:MM               LSF wall time (default: 2:00)
  --proxy-width N               Proxy width (default: 1024)
  --proxy-height N              Proxy height (default: 1024)
  --encoder NAME                FFmpeg H.264 encoder (default: h264_nvenc)
  --preset NAME                 FFmpeg encoder preset (default: veryfast; NVENC maps to p3)
  --crf N                       H.264 quality value (default: 23; NVENC uses -cq)
  --hwaccel NAME                FFmpeg input hwaccel (default: cuda; use none to disable)
  --scale-flags NAME            FFmpeg scale flags (default: bilinear)
  --ffmpeg-bin PATH             ffmpeg binary (default: ffmpeg)
  --ffprobe-bin PATH            ffprobe binary (default: ffprobe)
  --clip-id ID                  Limit to a clip id; may be repeated
  --camera-serial SERIAL        Limit to a camera serial; may be repeated
  --limit N                     Limit selected clip-camera rows for smoke testing
  --no-probe                    Use recording_clip_index metadata only; skip ffprobe
  --overwrite                   Overwrite existing proxy artifacts
  --run-id ID                   Stable LSF submission run id (default: UTC timestamp)
  --log-dir PATH                LSF logs/job script dir (default: <output-dir>/bsub_submission_<run-id>)
  --dry-run                     Print files and submit command; do not submit
  -h, --help                    Show this message
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --recording-dir) RECORDING_DIR="$2"; shift 2;;
    --output-dir) OUTPUT_DIR="$2"; shift 2;;
    --proxy-run-id) PROXY_RUN_ID="$2"; shift 2;;
    --queue) QUEUE="$2"; shift 2;;
    --ncores) NCORES="$2"; shift 2;;
    --mem-gb) MEM_GB="$2"; shift 2;;
    --gpus) GPUS="$2"; shift 2;;
    --walltime) WALLTIME="$2"; shift 2;;
    --proxy-width) PROXY_WIDTH="$2"; shift 2;;
    --proxy-height) PROXY_HEIGHT="$2"; shift 2;;
    --encoder) ENCODER="$2"; shift 2;;
    --preset) PRESET="$2"; shift 2;;
    --crf) CRF="$2"; shift 2;;
    --hwaccel) HWACCEL="$2"; shift 2;;
    --scale-flags) SCALE_FLAGS="$2"; shift 2;;
    --ffmpeg-bin) FFMPEG_BIN="$2"; shift 2;;
    --ffprobe-bin) FFPROBE_BIN="$2"; shift 2;;
    --clip-id) CLIP_IDS+=("$2"); shift 2;;
    --camera-serial) CAMERA_SERIALS+=("$2"); shift 2;;
    --limit) LIMIT="$2"; shift 2;;
    --no-probe) NO_PROBE=1; shift;;
    --overwrite) OVERWRITE=1; shift;;
    --run-id) RUN_ID="$2"; shift 2;;
    --log-dir) LOG_DIR="$2"; shift 2;;
    --dry-run) DRY_RUN=1; shift;;
    -h|--help) usage; exit 0;;
    --*) echo "Unknown arg: $1" >&2; usage; exit 2;;
    *)
      if [[ -z "$RECORDING_DIR" ]]; then
        RECORDING_DIR="$1"
        shift
      else
        echo "Unexpected positional arg: $1" >&2
        usage
        exit 2
      fi
      ;;
  esac
done

if [[ -z "$RECORDING_DIR" ]]; then
  echo "Missing required RECORDING_DIR" >&2
  usage
  exit 2
fi

if [[ "$DRY_RUN" != "1" ]]; then
  [[ -d "$RECORDING_DIR" ]] || { echo "Recording dir not found: $RECORDING_DIR" >&2; exit 2; }
  [[ -f "$RECORDING_DIR/recording_clip_index.json" ]] || {
    echo "recording_clip_index.json not found under: $RECORDING_DIR" >&2
    exit 2
  }
fi

if [[ -z "$PROXY_RUN_ID" ]]; then
  PROXY_RUN_ID="review_proxy_$(date -u +%Y%m%dT%H%M%SZ)"
fi
if [[ -z "$RUN_ID" ]]; then
  RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
fi
if [[ -z "$OUTPUT_DIR" ]]; then
  OUTPUT_DIR="${RECORDING_DIR}/derived/review_proxy/video_detect/${PROXY_RUN_ID}"
fi
if [[ -z "$LOG_DIR" ]]; then
  LOG_DIR="${OUTPUT_DIR}/bsub_submission_${RUN_ID}"
fi

SAFE_PROXY_RUN_ID="$(printf '%s' "$PROXY_RUN_ID" | tr -c 'A-Za-z0-9_.-' '_')"
mkdir -p "$LOG_DIR"

BUILDER_ARGS=(
  "$RECORDING_DIR"
  --output-dir "$OUTPUT_DIR"
  --proxy-run-id "$PROXY_RUN_ID"
  --proxy-width "$PROXY_WIDTH"
  --proxy-height "$PROXY_HEIGHT"
  --encoder "$ENCODER"
  --preset "$PRESET"
  --crf "$CRF"
  --hwaccel "$HWACCEL"
  --scale-flags "$SCALE_FLAGS"
  --ffmpeg-bin "$FFMPEG_BIN"
  --ffprobe-bin "$FFPROBE_BIN"
)
if [[ "$NO_PROBE" == "1" ]]; then BUILDER_ARGS+=(--no-probe); fi
if [[ "$OVERWRITE" == "1" ]]; then BUILDER_ARGS+=(--overwrite); fi
if [[ -n "$LIMIT" ]]; then BUILDER_ARGS+=(--limit "$LIMIT"); fi
for clip_id in "${CLIP_IDS[@]}"; do
  BUILDER_ARGS+=(--clip-id "$clip_id")
done
for camera_serial in "${CAMERA_SERIALS[@]}"; do
  BUILDER_ARGS+=(--camera-serial "$camera_serial")
done

printf -v BUILDER_ARGS_SHELL '%q ' "${BUILDER_ARGS[@]}"

JOB_SCRIPT="${LOG_DIR}/run_review_proxy_videos.sh"
LOG_DIR_Q="$(printf '%q' "$LOG_DIR")"
OUTPUT_DIR_Q="$(printf '%q' "$OUTPUT_DIR")"
PROXY_RUN_ID_Q="$(printf '%q' "$PROXY_RUN_ID")"
SAFE_PROXY_RUN_ID_Q="$(printf '%q' "$SAFE_PROXY_RUN_ID")"

cat > "$JOB_SCRIPT" <<JOBSCRIPT
#!/usr/bin/env bash
set -euo pipefail

cd "$(pwd)"

LOG_DIR=${LOG_DIR_Q}
OUTPUT_DIR=${OUTPUT_DIR_Q}
PROXY_RUN_ID=${PROXY_RUN_ID_Q}
SAFE_PROXY_RUN_ID=${SAFE_PROXY_RUN_ID_Q}
JOB_ID="\${LSB_JOBID:-manual}"
SUMMARY_JSON="\${LOG_DIR}/\${SAFE_PROXY_RUN_ID}.\${JOB_ID}.summary.json"

export MPLBACKEND=Agg
mkdir -p "\$LOG_DIR" "\$OUTPUT_DIR"

echo "repo=\$(pwd)"
echo "host=\$(hostname)"
echo "job_id=\$JOB_ID"
echo "output_dir=\$OUTPUT_DIR"
echo "summary_json=\$SUMMARY_JSON"
echo "proxy_run_id=\$PROXY_RUN_ID"

scripts/py -m fisheye.utils.build_review_proxy_videos ${BUILDER_ARGS_SHELL}--apply --json > "\$SUMMARY_JSON"

echo "summary_json=\$SUMMARY_JSON"
echo "manifest=\$OUTPUT_DIR/manifest.json"
JOBSCRIPT
chmod +x "$JOB_SCRIPT"

BSUB_ARGS=(
  -J "review_proxy_${SAFE_PROXY_RUN_ID}"
  -n "$NCORES"
  -W "$WALLTIME"
  -R "rusage[mem=${MEM_GB}G]"
  -oo "${LOG_DIR}/%J.out"
  -eo "${LOG_DIR}/%J.err"
)
if [[ -n "$QUEUE" ]]; then
  BSUB_ARGS+=(-q "$QUEUE")
fi
if [[ "$GPUS" != "0" ]]; then
  BSUB_ARGS+=(-gpu "num=${GPUS}")
fi

printf -v BSUB_ARGS_SHELL '%q ' "${BSUB_ARGS[@]}"
BSUB_CMD="bsub ${BSUB_ARGS_SHELL}bash $(printf '%q' "$JOB_SCRIPT")"

echo "Recording dir: $RECORDING_DIR"
echo "Output dir: $OUTPUT_DIR"
echo "Proxy run id: $PROXY_RUN_ID"
echo "Log dir: $LOG_DIR"
echo "Job script: $JOB_SCRIPT"
echo "Expected manifest: $OUTPUT_DIR/manifest.json"
echo "Expected summary: ${LOG_DIR}/${SAFE_PROXY_RUN_ID}.<JOBID>.summary.json"
echo "Builder command: scripts/py -m fisheye.utils.build_review_proxy_videos ${BUILDER_ARGS_SHELL}--apply --json > <summary>"
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
