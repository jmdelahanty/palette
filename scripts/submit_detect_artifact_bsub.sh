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
WORKFLOW_ID=""
RECORDING_ID=""
CLIP_ID=""
CLIP_INDEX=""
CAMERA_SERIAL=""
CONF=""
IOU=""
MAX_DET=""
DETECT_RUN_NAME=""
LATEST_POLICY="do_not_set_latest"
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
  --workflow-id ID         Optional workflow id recorded in logs/manifests
  --recording-id ID        Optional recording id recorded in logs/manifests
  --clip-id ID             Optional clip id, e.g. clip_000000
  --clip-index N           Optional zero-based clip index
  --camera-serial SERIAL   Optional camera serial for clip-camera provenance
  --detect-run-name NAME   Optional explicit detect run name inside the artifact
  --latest-policy POLICY   Importer latest policy recorded in manifest
                            (default: do_not_set_latest)
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
    --workflow-id) WORKFLOW_ID="$2"; shift 2;;
    --recording-id) RECORDING_ID="$2"; shift 2;;
    --clip-id) CLIP_ID="$2"; shift 2;;
    --clip-index) CLIP_INDEX="$2"; shift 2;;
    --camera-serial) CAMERA_SERIAL="$2"; shift 2;;
    --detect-run-name) DETECT_RUN_NAME="$2"; shift 2;;
    --latest-policy) LATEST_POLICY="$2"; shift 2;;
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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

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
  if [[ -n "$CLIP_ID" ]]; then
    RUN_LABEL="${RUN_LABEL}_${CLIP_ID}"
  fi
  if [[ -n "$CAMERA_SERIAL" ]]; then
    RUN_LABEL="${RUN_LABEL}_cam${CAMERA_SERIAL}"
  fi
fi
SAFE_LABEL="$(printf '%s' "$RUN_LABEL" | tr -c 'A-Za-z0-9_.-' '_')"
RUN_DIR="${OUTPUT_DIR}/detect_artifact_${RUN_ID}_${SAFE_LABEL}"

if [[ -e "$RUN_DIR" ]]; then
  echo "Run directory already exists: $RUN_DIR" >&2
  echo "Choose a different --run-id or --output-dir." >&2
  exit 2
fi
mkdir -p "$RUN_DIR"

scripts/py - "$RUN_DIR/submission_context.json" \
  "$ZARR" "$VIDEO" "$MODEL" "$CONFIG" "$OUTPUT_DIR" "$RUN_ID" "$RUN_LABEL" "$SAFE_LABEL" \
  "$WORKFLOW_ID" "$RECORDING_ID" "$CLIP_ID" "$CLIP_INDEX" "$CAMERA_SERIAL" "$DETECT_RUN_NAME" <<'PY'
import json
import sys
from pathlib import Path

(
    output_path,
    zarr,
    video,
    model,
    config,
    output_dir,
    run_id,
    run_label,
    safe_label,
    workflow_id,
    recording_id,
    clip_id,
    clip_index,
    camera_serial,
    detect_run_name,
) = sys.argv[1:]

def optional(value: str):
    return value if value else None

payload = {
    "schema_version": 1,
    "submission_kind": "detect_artifact_bsub",
    "target_zarr": zarr,
    "video": video,
    "model": model,
    "config": config,
    "output_dir": output_dir,
    "run_id": run_id,
    "run_label": run_label,
    "safe_label": safe_label,
    "workflow_id": optional(workflow_id),
    "recording_id": optional(recording_id),
    "clip_id": optional(clip_id),
    "clip_index": int(clip_index) if clip_index else None,
    "camera_serial": optional(camera_serial),
    "detect_run_name": optional(detect_run_name),
}
Path(output_path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

ARTIFACT_ARGS=(
  "$VIDEO"
  --target-zarr "$ZARR"
  --model "$MODEL"
  --config "$CONFIG"
  --decode-backend "$DECODE_BACKEND"
  --batch-size "$BATCH_SIZE"
  --latest-policy "$LATEST_POLICY"
)
if [[ -n "$CONF" ]]; then ARTIFACT_ARGS+=(--conf "$CONF"); fi
if [[ -n "$IOU" ]]; then ARTIFACT_ARGS+=(--iou "$IOU"); fi
if [[ -n "$MAX_DET" ]]; then ARTIFACT_ARGS+=(--max-det "$MAX_DET"); fi
if [[ "${#RESIZE_DIMS[@]}" -gt 0 ]]; then ARTIFACT_ARGS+=(--resize-dims "${RESIZE_DIMS[@]}"); fi
if [[ -n "$WORKFLOW_ID" ]]; then ARTIFACT_ARGS+=(--workflow-id "$WORKFLOW_ID"); fi
if [[ -n "$RECORDING_ID" ]]; then ARTIFACT_ARGS+=(--recording-id "$RECORDING_ID"); fi
if [[ -n "$CLIP_ID" ]]; then ARTIFACT_ARGS+=(--clip-id "$CLIP_ID"); fi
if [[ -n "$CLIP_INDEX" ]]; then ARTIFACT_ARGS+=(--clip-index "$CLIP_INDEX"); fi
if [[ -n "$CAMERA_SERIAL" ]]; then ARTIFACT_ARGS+=(--camera-serial "$CAMERA_SERIAL"); fi
if [[ -n "$DETECT_RUN_NAME" ]]; then ARTIFACT_ARGS+=(--run-name "$DETECT_RUN_NAME"); fi
if [[ "$OVERWRITE_ARTIFACT" == "1" ]]; then ARTIFACT_ARGS+=(--overwrite-artifact); fi

printf -v ARTIFACT_ARGS_SHELL '%q ' "${ARTIFACT_ARGS[@]}"

JOB_SCRIPT="${RUN_DIR}/run_detect_artifact.sh"
RUN_DIR_Q="$(printf '%q' "$RUN_DIR")"
SAFE_LABEL_Q="$(printf '%q' "$SAFE_LABEL")"
REPO_ROOT_Q="$(printf '%q' "$REPO_ROOT")"
WORKFLOW_ID_Q="$(printf '%q' "$WORKFLOW_ID")"
RECORDING_ID_Q="$(printf '%q' "$RECORDING_ID")"
CLIP_ID_Q="$(printf '%q' "$CLIP_ID")"
CLIP_INDEX_Q="$(printf '%q' "$CLIP_INDEX")"
CAMERA_SERIAL_Q="$(printf '%q' "$CAMERA_SERIAL")"

cat > "$JOB_SCRIPT" <<JOBSCRIPT
#!/usr/bin/env bash
set -euo pipefail

cd ${REPO_ROOT_Q}

RUN_DIR=${RUN_DIR_Q}
RUN_LABEL=${SAFE_LABEL_Q}
WORKFLOW_ID=${WORKFLOW_ID_Q}
RECORDING_ID=${RECORDING_ID_Q}
CLIP_ID=${CLIP_ID_Q}
CLIP_INDEX=${CLIP_INDEX_Q}
CAMERA_SERIAL=${CAMERA_SERIAL_Q}
JOB_ID="\${LSB_JOBID:-manual}"

scratch_user="\${USER:-\$(id -un)}"
if [[ -n "\${LSB_JOBID:-}" && -d "/scratch/\${scratch_user}" && -w "/scratch/\${scratch_user}" && -x "/scratch/\${scratch_user}" ]]; then
  SCRATCH_ROOT="/scratch/\${scratch_user}/\${LSB_JOBID}"
else
  SCRATCH_ROOT="\${TMPDIR:-/tmp}/palette_detect_artifact_\${JOB_ID}"
fi
export PALETTE_JOB_CACHE="\${SCRATCH_ROOT}/palette_cache"
export YOLO_CONFIG_DIR="\${PALETTE_JOB_CACHE}/ultralytics"
export MPLCONFIGDIR="\${PALETTE_JOB_CACHE}/matplotlib"
export MPLBACKEND=Agg
mkdir -p "\$PALETTE_JOB_CACHE" "\$YOLO_CONFIG_DIR" "\$MPLCONFIGDIR" "\$SCRATCH_ROOT" "\$RUN_DIR"

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
echo "workflow_id=\$WORKFLOW_ID"
echo "recording_id=\$RECORDING_ID"
echo "clip_id=\$CLIP_ID"
echo "clip_index=\$CLIP_INDEX"
echo "camera_serial=\$CAMERA_SERIAL"

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
  "workflow_id": "\$WORKFLOW_ID",
  "recording_id": "\$RECORDING_ID",
  "clip_id": "\$CLIP_ID",
  "clip_index": "\$CLIP_INDEX",
  "camera_serial": "\$CAMERA_SERIAL",
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
echo "Submission context: ${RUN_DIR}/submission_context.json"
if [[ -n "$CLIP_ID" || -n "$CAMERA_SERIAL" ]]; then
  echo "Clip context: workflow_id=${WORKFLOW_ID:-<none>} recording_id=${RECORDING_ID:-<none>} clip_id=${CLIP_ID:-<none>} clip_index=${CLIP_INDEX:-<none>} camera_serial=${CAMERA_SERIAL:-<none>}"
fi
if [[ -n "$DETECT_RUN_NAME" ]]; then
  echo "Detect run name: $DETECT_RUN_NAME"
fi
echo "Expected tarball: ${RUN_DIR}/${SAFE_LABEL}.<JOBID>.tar.gz"
echo "Expected summary: ${RUN_DIR}/${SAFE_LABEL}.<JOBID>.summary.json"
echo "Expected transfer log: ${RUN_DIR}/${SAFE_LABEL}.<JOBID>.transfer.json"
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
