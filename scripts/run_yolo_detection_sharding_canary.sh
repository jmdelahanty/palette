#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 3 ]]; then
  echo "Usage: $0 RUN_DIR SOURCE_VIDEO MODEL_PATH" >&2
  exit 2
fi

RUN_DIR="$1"
SOURCE_VIDEO="$2"
MODEL_PATH="$3"
REPO_ROOT="/groups/johnson/johnsonlab/jeremy/gitrepos/palette"
SCRATCH_ROOT="${TMPDIR:-/scratch/${USER}/${LSB_JOBID:-yolo_detection_sharding}}"
CLIP_PATH="${SCRATCH_ROOT}/yolo_detection_sharding_100s.mp4"
OUTPUT_ZARR="${RUN_DIR}/yolo_detection_sharding_ab.zarr"
REGULAR_RUN="detect_yolo_regular_100s"
SHARDED_RUN="detect_yolo_sharded_100s"

mkdir -p "$RUN_DIR" "$SCRATCH_ROOT"
if [[ -e "$OUTPUT_ZARR" ]]; then
  echo "Output already exists: $OUTPUT_ZARR" >&2
  exit 2
fi

echo "host=$(hostname)"
echo "job_id=${LSB_JOBID:-}"
echo "repo_commit=$(git -C "$REPO_ROOT" rev-parse HEAD)"
echo "source_video=$SOURCE_VIDEO"
echo "model_path=$MODEL_PATH"
echo "scratch_root=$SCRATCH_ROOT"

ffmpeg -hide_banner -loglevel error -y -i "$SOURCE_VIDEO" -t 100 -an -c:v copy "$CLIP_PATH"

cd "$REPO_ROOT"
/usr/bin/time -f $'elapsed_seconds=%e\nmaximum_rss_kib=%M' \
  -o "$RUN_DIR/regular.time.txt" \
  scripts/py -m fisheye.detection.detect_yolo \
    "$CLIP_PATH" \
    --model "$MODEL_PATH" \
    --output "$OUTPUT_ZARR" \
    --conf 0.4 \
    --iou 0.8 \
    --max-det 1 \
    --batch-size 16 \
    --resize-dims 640 640 \
    --decode-backend pynvvc_nv12_rgb \
    --run-name "$REGULAR_RUN"

/usr/bin/time -f $'elapsed_seconds=%e\nmaximum_rss_kib=%M' \
  -o "$RUN_DIR/sharded.time.txt" \
  scripts/py -m fisheye.detection.detect_yolo \
    "$CLIP_PATH" \
    --model "$MODEL_PATH" \
    --output "$OUTPUT_ZARR" \
    --conf 0.4 \
    --iou 0.8 \
    --max-det 1 \
    --batch-size 16 \
    --resize-dims 640 640 \
    --decode-backend pynvvc_nv12_rgb \
    --run-name "$SHARDED_RUN" \
    --detect-row-shard-rows 262144 \
    --detect-frame-shard-rows 262144

scripts/py -m fisheye.diagnostics.audit_yolo_detection_sharding \
  "$OUTPUT_ZARR" \
  --regular-run "$REGULAR_RUN" \
  --sharded-run "$SHARDED_RUN" \
  --output-json "$RUN_DIR/audit.json"
