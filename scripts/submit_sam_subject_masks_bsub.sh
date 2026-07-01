#!/usr/bin/env bash
set -euo pipefail

ZARR=""
CROP_RUN=""
KEYPOINT_RUN=""
KEYPOINT_GROUP="refined_keypoints_runs"
OUTPUT_RUN=""
SAM3_ROOT="${PALETTE_SAM3_ROOT:-/groups/johnson/johnsonlab/jeremy/gitrepos/sam3}"
CHECKPOINT=""
RUN_ID=""
LOG_DIR=""
REPO_DIR=""
PYTHON_BIN="${PALETTE_PYTHON:-}"
QUEUE="gpu_l4"
NCORES=4
MEM_GB=64
GPUS=1
WALLTIME="1:00"
BATCH_SIZE=8
APPLY=0
APPLY_LIMIT=""
ALLOW_FULL_APPLY=0
SUBMIT=0
PROFILE_TIMINGS=0
INCLUDE_INTERPOLATED=0
SINGLE_MASK=0
NO_BOX_PROMPT=0
NO_HF_DOWNLOAD=0
OVERWRITE=0
BOX_PROMPT_SOURCE="detect"
NEGATIVE_POINT_POLICY="border8"
NEGATIVE_POINT_MARGIN_FRACTION="0.05"
POSITIVE_KEYPOINT_LABELS=()

usage() {
  cat <<'USAGE'
Usage: submit_sam_subject_masks_bsub.sh --zarr PATH --crop-run RUN --keypoint-run RUN [options]

Submit one GPU LSF job for fisheye.utils.run_sam_subject_masks. By default this
is a dry-run that prints the bsub command and generated job script path.

Required:
  --zarr PATH                 Palette analysis/training zarr path
  --crop-run RUN              crop_runs/<run> consumed as ROI source
  --keypoint-run RUN          keypoints_runs/<run> or refined_keypoints_runs/<run>

Options:
  --keypoint-group GROUP      auto|keypoints_runs|refined_keypoints_runs (default: refined_keypoints_runs)
  --output-run RUN            subject_mask_runs/<run> name
  --sam3-root PATH            SAM3 checkout visible to compute nodes
                              (default: $PALETTE_SAM3_ROOT or /groups/johnson/johnsonlab/jeremy/gitrepos/sam3)
  --checkpoint PATH           Optional local SAM3 checkpoint path visible to compute nodes
  --run-id ID                 Stable submission id (default: UTC timestamp)
  --log-dir PATH              Log/run dir (default: <recording>/logs/sam_subject_masks_bsub/<run-id>)
  --repo-dir PATH             Palette repo visible to compute nodes (default: current repo)
  --python-bin PATH           Optional Python interpreter for scripts/py via PALETTE_PYTHON
  --queue NAME                LSF queue (default: gpu_l4)
  --ncores N                  CPU slots (default: 4)
  --mem-gb N                  Memory GB (default: 64)
  --gpus N                    GPUs (default: 1)
  --walltime H:MM             LSF wall time (default: 1:00)
  --batch-size N              SAM inference batch size (default: 8)
  --apply                     Run SAM inference and write subject_mask_runs/<run>
  --apply-limit N             Limit --apply to first N eligible rows; recommended for smokes
  --allow-full-apply          Permit --apply without --apply-limit
  --profile-timings           Record per-stage timing diagnostics
  --include-interpolated      Include rows with detection_source != 0
  --single-mask               Use multimask_output=False
  --no-box-prompt             Use point prompts only
  --box-prompt-source NAME    detect|pose_roi|roi_inset (default: detect)
  --negative-point-policy P   none|border4|border8 (default: border8)
  --negative-point-margin-fraction F
                              Border negative-point margin fraction (default: 0.05)
  --positive-keypoint-labels LABEL...
                              Optional labels to use as positive prompts; must be last or followed by --.
  --no-hf-download            Require --checkpoint; do not allow Hugging Face downloads
  --overwrite                 Replace existing output run
  --submit                    Submit to LSF. Without this, print commands only.
  -h, --help                  Show this message
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --zarr) ZARR="$2"; shift 2;;
    --crop-run) CROP_RUN="$2"; shift 2;;
    --keypoint-run) KEYPOINT_RUN="$2"; shift 2;;
    --keypoint-group) KEYPOINT_GROUP="$2"; shift 2;;
    --output-run) OUTPUT_RUN="$2"; shift 2;;
    --sam3-root) SAM3_ROOT="$2"; shift 2;;
    --checkpoint) CHECKPOINT="$2"; shift 2;;
    --run-id) RUN_ID="$2"; shift 2;;
    --log-dir) LOG_DIR="$2"; shift 2;;
    --repo-dir) REPO_DIR="$2"; shift 2;;
    --python-bin) PYTHON_BIN="$2"; shift 2;;
    --queue) QUEUE="$2"; shift 2;;
    --ncores) NCORES="$2"; shift 2;;
    --mem-gb) MEM_GB="$2"; shift 2;;
    --gpus) GPUS="$2"; shift 2;;
    --walltime) WALLTIME="$2"; shift 2;;
    --batch-size) BATCH_SIZE="$2"; shift 2;;
    --apply) APPLY=1; shift;;
    --apply-limit) APPLY_LIMIT="$2"; shift 2;;
    --allow-full-apply) ALLOW_FULL_APPLY=1; shift;;
    --profile-timings) PROFILE_TIMINGS=1; shift;;
    --include-interpolated) INCLUDE_INTERPOLATED=1; shift;;
    --single-mask) SINGLE_MASK=1; shift;;
    --no-box-prompt) NO_BOX_PROMPT=1; shift;;
    --box-prompt-source) BOX_PROMPT_SOURCE="$2"; shift 2;;
    --negative-point-policy) NEGATIVE_POINT_POLICY="$2"; shift 2;;
    --negative-point-margin-fraction) NEGATIVE_POINT_MARGIN_FRACTION="$2"; shift 2;;
    --positive-keypoint-labels)
      shift
      while [[ $# -gt 0 && "$1" != --* ]]; do
        POSITIVE_KEYPOINT_LABELS+=("$1")
        shift
      done
      ;;
    --no-hf-download) NO_HF_DOWNLOAD=1; shift;;
    --overwrite) OVERWRITE=1; shift;;
    --submit) SUBMIT=1; shift;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown arg: $1" >&2; usage; exit 2;;
  esac
done

if [[ -z "$ZARR" || -z "$CROP_RUN" || -z "$KEYPOINT_RUN" ]]; then
  echo "Missing required --zarr, --crop-run, or --keypoint-run." >&2
  usage >&2
  exit 2
fi
if [[ "$APPLY" == "1" && -z "$APPLY_LIMIT" && "$ALLOW_FULL_APPLY" != "1" ]]; then
  echo "--apply requires --apply-limit N unless --allow-full-apply is passed." >&2
  exit 2
fi
if [[ "$NO_HF_DOWNLOAD" == "1" && -z "$CHECKPOINT" ]]; then
  echo "--no-hf-download requires --checkpoint PATH." >&2
  exit 2
fi
if [[ -z "$REPO_DIR" ]]; then
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi
if [[ -z "$RUN_ID" ]]; then
  RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
fi
if [[ -z "$LOG_DIR" ]]; then
  ZARR_PARENT="$(dirname "$(dirname "$ZARR")")"
  LOG_DIR="${ZARR_PARENT%/}/logs/sam_subject_masks_bsub/${RUN_ID}"
fi
mkdir -p "$LOG_DIR"

if [[ "$SUBMIT" == "1" ]]; then
  [[ -d "$ZARR" ]] || { echo "Zarr not found: $ZARR" >&2; exit 2; }
  [[ -d "$SAM3_ROOT" ]] || { echo "SAM3 root not found: $SAM3_ROOT" >&2; exit 2; }
  if [[ -n "$PYTHON_BIN" ]]; then
    [[ -x "$PYTHON_BIN" ]] || { echo "Python interpreter not executable: $PYTHON_BIN" >&2; exit 2; }
  fi
  if [[ -n "$CHECKPOINT" ]]; then
    [[ -f "$CHECKPOINT" ]] || { echo "Checkpoint not found: $CHECKPOINT" >&2; exit 2; }
  fi
fi

CMD=(
  scripts/py -m fisheye.utils.run_sam_subject_masks
  "$ZARR"
  --crop-run "$CROP_RUN"
  --keypoint-run "$KEYPOINT_RUN"
  --keypoint-group "$KEYPOINT_GROUP"
  --sam3-root "$SAM3_ROOT"
  --batch-size "$BATCH_SIZE"
  --box-prompt-source "$BOX_PROMPT_SOURCE"
  --negative-point-policy "$NEGATIVE_POINT_POLICY"
  --negative-point-margin-fraction "$NEGATIVE_POINT_MARGIN_FRACTION"
  --json
)
if [[ -n "$OUTPUT_RUN" ]]; then CMD+=(--output-run "$OUTPUT_RUN"); fi
if [[ -n "$CHECKPOINT" ]]; then CMD+=(--checkpoint "$CHECKPOINT"); fi
if [[ "$APPLY" == "1" ]]; then CMD+=(--apply); fi
if [[ -n "$APPLY_LIMIT" ]]; then CMD+=(--apply-limit "$APPLY_LIMIT"); fi
if [[ "$PROFILE_TIMINGS" == "1" ]]; then CMD+=(--profile-timings); fi
if [[ "$INCLUDE_INTERPOLATED" == "1" ]]; then CMD+=(--include-interpolated); fi
if [[ "$SINGLE_MASK" == "1" ]]; then CMD+=(--single-mask); fi
if [[ "$NO_BOX_PROMPT" == "1" ]]; then CMD+=(--no-box-prompt); fi
if [[ "$NO_HF_DOWNLOAD" == "1" ]]; then CMD+=(--no-hf-download); fi
if [[ "$OVERWRITE" == "1" ]]; then CMD+=(--overwrite); fi
if [[ "${#POSITIVE_KEYPOINT_LABELS[@]}" -gt 0 ]]; then
  CMD+=(--positive-keypoint-labels "${POSITIVE_KEYPOINT_LABELS[@]}")
fi

SUMMARY_JSON="${LOG_DIR}/summary.json"
JOB_SCRIPT="${LOG_DIR}/run_sam_subject_masks.sh"
REPO_DIR_Q="$(printf '%q' "$REPO_DIR")"
PYTHON_BIN_Q="$(printf '%q' "$PYTHON_BIN")"
SUMMARY_JSON_Q="$(printf '%q' "$SUMMARY_JSON")"
printf -v CMD_SHELL '%q ' "${CMD[@]}"

cat > "$JOB_SCRIPT" <<JOBSCRIPT
#!/usr/bin/env bash
set -euo pipefail

cd ${REPO_DIR_Q}

if [[ -n ${PYTHON_BIN_Q} ]]; then
  export PALETTE_PYTHON=${PYTHON_BIN_Q}
fi

echo "repo=\$(pwd)"
echo "palette_python=\${PALETTE_PYTHON:-<scripts/py default>}"
echo "host=\$(hostname)"
echo "job_id=\${LSB_JOBID:-manual}"
echo "sam3_root=$(printf '%q' "$SAM3_ROOT")"
echo "checkpoint=$(printf '%q' "$CHECKPOINT")"
echo "summary_json=${SUMMARY_JSON_Q}"

${CMD_SHELL}> ${SUMMARY_JSON_Q}
echo "summary_json=${SUMMARY_JSON_Q}"
JOBSCRIPT
chmod +x "$JOB_SCRIPT"

BSUB_ARGS=(
  -J "sam_subject_masks_${RUN_ID}"
  -n "$NCORES"
  -gpu "num=${GPUS}:mode=exclusive_process"
  -W "$WALLTIME"
  -R "rusage[mem=${MEM_GB}G]"
  -oo "${LOG_DIR}/%J.out"
  -eo "${LOG_DIR}/%J.err"
)
if [[ -n "$QUEUE" ]]; then
  BSUB_ARGS+=(-q "$QUEUE")
fi

printf -v BSUB_ARGS_SHELL '%q ' "${BSUB_ARGS[@]}"
BSUB_CMD="bsub ${BSUB_ARGS_SHELL}bash $(printf '%q' "$JOB_SCRIPT")"

echo "Run dir: $LOG_DIR"
echo "Job script: $JOB_SCRIPT"
echo "Summary JSON: $SUMMARY_JSON"
echo "SAM3 root: $SAM3_ROOT"
if [[ -n "$PYTHON_BIN" ]]; then
  echo "Python: $PYTHON_BIN"
else
  echo "Python: <scripts/py default>"
fi
if [[ -n "$CHECKPOINT" ]]; then
  echo "Checkpoint: $CHECKPOINT"
else
  echo "Checkpoint: <none; SAM3 runtime may try its default/HF resolution>"
fi
echo "Command: cd $(printf '%q' "$REPO_DIR") && ${CMD_SHELL}> $(printf '%q' "$SUMMARY_JSON")"
echo "Submit command: $BSUB_CMD"

if [[ "$SUBMIT" != "1" ]]; then
  echo "Dry run only; pass --submit to submit."
  exit 0
fi
if ! command -v bsub >/dev/null 2>&1; then
  echo "bsub not found in PATH. Is this an LSF cluster login node?" >&2
  exit 2
fi
bsub "${BSUB_ARGS[@]}" bash "$JOB_SCRIPT"
