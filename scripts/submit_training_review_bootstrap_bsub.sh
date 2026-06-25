#!/usr/bin/env bash
set -euo pipefail

ZARR=""
CROP_RUN=""
POSE_MODEL=""
REGISTRY="${PALETTE_REGISTRY_PATH:-/groups/johnson/johnsonlab/jeremy/registries/palette_registry.sqlite}"
RUN_ID=""
LOG_DIR=""
REPO_DIR=""
QUEUE="gpu_l4"
NCORES=4
MEM_GB=32
WALLTIME="1:00"
GPUS=1
SUBMIT=0

KEYPOINT_IMGSZ=512
SUBJECT_MODEL_INPUT_SIZE=512
MODEL_INPUT_TRANSFORM="auto"
POSE_SCHEMA="traditional_v2"
KEYPOINT_BATCH_SIZE=256
SUBJECT_BATCH_SIZE=128
KEYPOINT_DEVICE="0"
SUBJECT_DEVICE="cuda:0"

usage() {
  cat <<'USAGE'
Usage: submit_training_review_bootstrap_bsub.sh --zarr PATH --crop-run RUN --pose-model PATH --run-id ID [options]

Submit one LSF job that bootstraps reviewable training-Zarr surfaces:
keypoints, refined keypoints, subject masks, and refined subject masks.

Required:
  --zarr PATH               Training zarr path
  --crop-run RUN            crop_runs/<run> to use as native ROI source
  --pose-model PATH         YOLO pose model weights
  --run-id ID               Stable suffix for deterministic run names

Options:
  --registry PATH           Palette registry sqlite path
  --log-dir PATH            Submission log dir (default: <zarr recording logs>/training_review_bootstrap/<run-id>)
  --repo-dir PATH           Repo checkout visible to compute nodes (default: current directory)
  --queue NAME              LSF queue (default: gpu_l4)
  --ncores N                CPU slots (default: 4)
  --mem-gb N                Memory GB (default: 32)
  --walltime H:MM           Wall time (default: 1:00)
  --gpus N                  GPUs (default: 1)
  --keypoint-imgsz N        YOLO pose model input size (default: 512)
  --subject-model-input-size N
                            Subject-mask model input size (default: 512)
  --model-input-transform MODE
                            auto|identity|pad_to_size (default: auto)
  --pose-schema NAME        Pose schema (default: traditional_v2)
  --keypoint-batch-size N   Keypoint batch size (default: 256)
  --subject-batch-size N    Subject-mask batch size (default: 128)
  --keypoint-device DEVICE  Keypoint device (default: 0)
  --subject-device DEVICE   Subject-mask device (default: cuda:0)
  --submit                  Submit to LSF. Without this, print the command only.
  -h, --help                Show this message
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --zarr) ZARR="$2"; shift 2;;
    --crop-run) CROP_RUN="$2"; shift 2;;
    --pose-model) POSE_MODEL="$2"; shift 2;;
    --registry) REGISTRY="$2"; shift 2;;
    --run-id) RUN_ID="$2"; shift 2;;
    --log-dir) LOG_DIR="$2"; shift 2;;
    --repo-dir) REPO_DIR="$2"; shift 2;;
    --queue) QUEUE="$2"; shift 2;;
    --ncores) NCORES="$2"; shift 2;;
    --mem-gb) MEM_GB="$2"; shift 2;;
    --walltime) WALLTIME="$2"; shift 2;;
    --gpus) GPUS="$2"; shift 2;;
    --keypoint-imgsz) KEYPOINT_IMGSZ="$2"; shift 2;;
    --subject-model-input-size) SUBJECT_MODEL_INPUT_SIZE="$2"; shift 2;;
    --model-input-transform) MODEL_INPUT_TRANSFORM="$2"; shift 2;;
    --pose-schema) POSE_SCHEMA="$2"; shift 2;;
    --keypoint-batch-size) KEYPOINT_BATCH_SIZE="$2"; shift 2;;
    --subject-batch-size) SUBJECT_BATCH_SIZE="$2"; shift 2;;
    --keypoint-device) KEYPOINT_DEVICE="$2"; shift 2;;
    --subject-device) SUBJECT_DEVICE="$2"; shift 2;;
    --submit) SUBMIT=1; shift;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown arg: $1" >&2; usage; exit 2;;
  esac
done

if [[ -z "$ZARR" || -z "$CROP_RUN" || -z "$POSE_MODEL" || -z "$RUN_ID" ]]; then
  echo "Missing required --zarr, --crop-run, --pose-model, or --run-id." >&2
  usage >&2
  exit 2
fi
if [[ -z "$REPO_DIR" ]]; then
  REPO_DIR="$(pwd)"
fi
if [[ -z "$LOG_DIR" ]]; then
  ZARR_PARENT="$(dirname "$(dirname "$ZARR")")"
  LOG_DIR="${ZARR_PARENT%/}/logs/training_review_bootstrap/${RUN_ID}"
fi
mkdir -p "$LOG_DIR"

CMD=(
  scripts/py -m fisheye.utils.bootstrap_training_review_surfaces
  "$ZARR"
  --crop-run "$CROP_RUN"
  --pose-model "$POSE_MODEL"
  --registry "$REGISTRY"
  --run-id "$RUN_ID"
  --pose-schema "$POSE_SCHEMA"
  --keypoint-imgsz "$KEYPOINT_IMGSZ"
  --subject-model-input-size "$SUBJECT_MODEL_INPUT_SIZE"
  --model-input-transform "$MODEL_INPUT_TRANSFORM"
  --keypoint-batch-size "$KEYPOINT_BATCH_SIZE"
  --subject-batch-size "$SUBJECT_BATCH_SIZE"
  --keypoint-device "$KEYPOINT_DEVICE"
  --subject-device "$SUBJECT_DEVICE"
  --progress-jsonl "$LOG_DIR/finalize_progress.jsonl"
  --json
)

printf 'repo_dir=%q\n' "$REPO_DIR"
printf 'log_dir=%q\n' "$LOG_DIR"
printf 'command: cd %q &&' "$REPO_DIR"
printf ' %q' "${CMD[@]}"
printf '\n'

if [[ "$SUBMIT" != "1" ]]; then
  echo "Dry run only; pass --submit to submit."
  exit 0
fi

BSUB_CMD="cd $(printf '%q' "$REPO_DIR") &&"
for token in "${CMD[@]}"; do
  BSUB_CMD+=" $(printf '%q' "$token")"
done

GPU_REQ="num=${GPUS}:mode=exclusive_process"
bsub -q "$QUEUE" \
  -gpu "$GPU_REQ" \
  -n "$NCORES" \
  -W "$WALLTIME" \
  -R "rusage[mem=$((MEM_GB * 1024))]" \
  -J training_review_bootstrap \
  -oo "$LOG_DIR/%J.out" \
  -eo "$LOG_DIR/%J.err" \
  "$BSUB_CMD"
