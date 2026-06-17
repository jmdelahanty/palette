#!/usr/bin/env bash
set -euo pipefail

ROOT="/nvme1/recordings"
REGISTRY="/nvme1/palette_registry.sqlite"
CONFIG="configs/fisheye/default.yaml"
PATH_CONTAINS=""
LOG_DIR=""
RUN_ID=""

DETECT_QUEUE="gpu_l4"
DETECT_GPU="num=1"
DETECT_NCORES=4
DETECT_MEM_GB=64
DETECT_BATCH_SIZE=4
DETECT_MAX_ACTIVE=2
DETECT_DECODE_BACKEND="pynvvc_nv12_rgb"
DETECT_RESIZE_DIMS=("640" "640")
DETECT_MODEL=""
DETECT_SET_ID=""
DETECT_TOP_K=5
DETECT_REQUIRE_UNIQUE=0
DETECT_INCLUDE_NON_SUCCESS=0
DETECT_REQUIRE_TUNING=0
DETECT_OVERWRITE=0

QUALITY_QUEUE="short"
QUALITY_NCORES=4
QUALITY_MEM_GB=16
QUALITY_WALLTIME="1:00"
QUALITY_THRESHOLD=100.0
QUALITY_THRESHOLD_MODE="scaled"
QUALITY_THRESHOLD_REFERENCE_WIDTH=640.0

REFINE_QUEUE="short"
REFINE_NCORES=4
REFINE_MEM_GB=16
REFINE_WALLTIME="1:00"
REFINE_SAVE_VISUALS=0

SUBMIT=0

usage() {
  cat <<'USAGE'
Usage: submit_detect_quality_refine_bsub.sh [options]

Submit a registry-discovered detect -> detect_quality -> refined_detect LSF chain.
The script plans by default; pass --submit to call bsub.

WARNING: this direct-write convenience chain runs detect_quality/refine_detect
against latest detect/quality state at postprocess time. For broad production
runs or concurrent writers, prefer submit_detect_artifact_quality_refine_bsub.sh,
which imports one deterministic detect run per recording and pins downstream
stages to explicit run names.

Discovery options:
  --root PATH                    Recording root (default: /nvme1/recordings)
  --registry PATH                Registry sqlite path (default: /nvme1/palette_registry.sqlite)
  --path-contains STR            Registry zarr_path substring filter
  --config PATH                  Pipeline config passed to detect/refine
  --log-dir PATH                 Submission run root (default: <root>/logs/detect_quality_refine_bsub)
  --run-id ID                    Stable run id

Detect job options:
  --detect-queue NAME            LSF queue for detect array (default: gpu_l4)
  --detect-gpu SPEC              LSF GPU spec for detect array (default: num=1)
  --detect-ncores N              Cores per detect array element (default: 4)
  --detect-mem-gb N              Memory per detect array element (default: 64)
  --detect-batch-size N          Zarrs per detect array element (default: 4)
  --detect-max-active N          Max concurrent detect array elements (default: 2)
  --detect-decode-backend NAME   Decode backend passed to run_detections_batch
                                  (default: pynvvc_nv12_rgb)
  --detect-resize-dims H W       Canonical inference size (default: 640 640)
  --detect-model PATH            Explicit detect model path
  --detect-set-id ID             Optional detect set filter for registry model resolution
  --detect-top-k N               Candidate provenance depth (default: 5)
  --detect-require-unique        Fail model resolution on tied top scores
  --detect-include-non-success   Include non-success runs in model resolution
  --detect-require-tuning        Skip zarrs without detection_tuning
  --detect-overwrite             Re-run detection even if detect output exists

Quality/refine job options:
  --quality-queue NAME           LSF queue for detect_quality (default: short)
  --quality-ncores N             Cores for detect_quality (default: 4)
  --quality-mem-gb N             Memory for detect_quality (default: 16)
  --quality-walltime HH:MM       Walltime for detect_quality (default: 1:00)
  --quality-threshold VALUE      Jump threshold (default: 100.0)
  --quality-threshold-mode MODE  scaled, pixels, or normalized (default: scaled)
  --quality-threshold-reference-width VALUE
                                  Reference width for scaled threshold (default: 640.0)
  --refine-queue NAME            LSF queue for refine_detect (default: short)
  --refine-ncores N              Cores for refine_detect (default: 4)
  --refine-mem-gb N              Memory for refine_detect (default: 16)
  --refine-walltime HH:MM        Walltime for refine_detect (default: 1:00)
  --refine-save-visuals          Ask refine_detect_batch to save visuals

Execution:
  --submit                       Actually submit jobs. Without this, dry-run only.
  -h, --help                     Show this message
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --root) ROOT="$2"; shift 2;;
    --registry) REGISTRY="$2"; shift 2;;
    --path-contains) PATH_CONTAINS="$2"; shift 2;;
    --config) CONFIG="$2"; shift 2;;
    --log-dir) LOG_DIR="$2"; shift 2;;
    --run-id) RUN_ID="$2"; shift 2;;
    --detect-queue) DETECT_QUEUE="$2"; shift 2;;
    --detect-gpu) DETECT_GPU="$2"; shift 2;;
    --detect-ncores) DETECT_NCORES="$2"; shift 2;;
    --detect-mem-gb) DETECT_MEM_GB="$2"; shift 2;;
    --detect-batch-size) DETECT_BATCH_SIZE="$2"; shift 2;;
    --detect-max-active) DETECT_MAX_ACTIVE="$2"; shift 2;;
    --detect-decode-backend) DETECT_DECODE_BACKEND="$2"; shift 2;;
    --detect-resize-dims) DETECT_RESIZE_DIMS=("$2" "$3"); shift 3;;
    --detect-model) DETECT_MODEL="$2"; shift 2;;
    --detect-set-id) DETECT_SET_ID="$2"; shift 2;;
    --detect-top-k) DETECT_TOP_K="$2"; shift 2;;
    --detect-require-unique) DETECT_REQUIRE_UNIQUE=1; shift;;
    --detect-include-non-success) DETECT_INCLUDE_NON_SUCCESS=1; shift;;
    --detect-require-tuning) DETECT_REQUIRE_TUNING=1; shift;;
    --detect-overwrite) DETECT_OVERWRITE=1; shift;;
    --quality-queue) QUALITY_QUEUE="$2"; shift 2;;
    --quality-ncores) QUALITY_NCORES="$2"; shift 2;;
    --quality-mem-gb) QUALITY_MEM_GB="$2"; shift 2;;
    --quality-walltime) QUALITY_WALLTIME="$2"; shift 2;;
    --quality-threshold) QUALITY_THRESHOLD="$2"; shift 2;;
    --quality-threshold-mode) QUALITY_THRESHOLD_MODE="$2"; shift 2;;
    --quality-threshold-reference-width) QUALITY_THRESHOLD_REFERENCE_WIDTH="$2"; shift 2;;
    --refine-queue) REFINE_QUEUE="$2"; shift 2;;
    --refine-ncores) REFINE_NCORES="$2"; shift 2;;
    --refine-mem-gb) REFINE_MEM_GB="$2"; shift 2;;
    --refine-walltime) REFINE_WALLTIME="$2"; shift 2;;
    --refine-save-visuals) REFINE_SAVE_VISUALS=1; shift;;
    --submit) SUBMIT=1; shift;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown arg: $1" >&2; usage; exit 2;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${SCRIPT_DIR}/lib/palette_lsf.sh"

if [[ -z "$RUN_ID" ]]; then
  RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
fi
if [[ -z "$LOG_DIR" ]]; then
  LOG_DIR="${ROOT}/logs/detect_quality_refine_bsub"
fi

RUN_DIR="${LOG_DIR}/detect_quality_refine_${RUN_ID}"
DETECT_RUN_DIR="${RUN_DIR}/detect_${RUN_ID}"
RECORDINGS_FILE="${DETECT_RUN_DIR}/recordings.txt"
QUALITY_LOG_DIR="${RUN_DIR}/detect_quality_logs"
REFINE_LOG_DIR="${RUN_DIR}/refine_detect_logs"

if [[ -e "$RUN_DIR" ]]; then
  echo "Run directory already exists: $RUN_DIR" >&2
  echo "Choose a different --run-id or remove the existing run directory." >&2
  exit 2
fi
mkdir -p "$RUN_DIR"

DETECT_CMD=(
  "${SCRIPT_DIR}/submit_detect_batches_bsub.sh"
  --root "$ROOT"
  --source registry
  --registry "$REGISTRY"
  --batch-size "$DETECT_BATCH_SIZE"
  --max-active "$DETECT_MAX_ACTIVE"
  --queue "$DETECT_QUEUE"
  --gpu "$DETECT_GPU"
  --ncores "$DETECT_NCORES"
  --mem-gb "$DETECT_MEM_GB"
  --config "$CONFIG"
  --log-dir "$RUN_DIR"
  --run-id "$RUN_ID"
  --top-k "$DETECT_TOP_K"
)
if [[ -n "$PATH_CONTAINS" ]]; then DETECT_CMD+=(--path-contains "$PATH_CONTAINS"); fi
if [[ -n "$DETECT_DECODE_BACKEND" ]]; then DETECT_CMD+=(--decode-backend "$DETECT_DECODE_BACKEND"); fi
if [[ "${#DETECT_RESIZE_DIMS[@]}" -gt 0 ]]; then DETECT_CMD+=(--resize-dims "${DETECT_RESIZE_DIMS[@]}"); fi
if [[ -n "$DETECT_MODEL" ]]; then DETECT_CMD+=(--model "$DETECT_MODEL"); fi
if [[ -n "$DETECT_SET_ID" ]]; then DETECT_CMD+=(--set-id "$DETECT_SET_ID"); fi
if [[ "$DETECT_REQUIRE_UNIQUE" == "1" ]]; then DETECT_CMD+=(--require-unique); fi
if [[ "$DETECT_INCLUDE_NON_SUCCESS" == "1" ]]; then DETECT_CMD+=(--include-non-success); fi
if [[ "$DETECT_REQUIRE_TUNING" == "1" ]]; then DETECT_CMD+=(--require-tuning); fi
if [[ "$DETECT_OVERWRITE" == "1" ]]; then DETECT_CMD+=(--overwrite); fi
if [[ "$SUBMIT" != "1" ]]; then DETECT_CMD+=(--dry-run); fi

DETECT_SUBMIT_LOG="${RUN_DIR}/detect_submit.log"
echo "Planning detect stage..."
"${DETECT_CMD[@]}" 2>&1 | tee "$DETECT_SUBMIT_LOG"

cat >&2 <<'WARNING'
Warning: direct detect_quality/refine_detect postprocess uses latest run
selection. Prefer submit_detect_artifact_quality_refine_bsub.sh for production
or concurrent writer workflows because it pins explicit detect and quality run
names after artifact import.
WARNING

if [[ ! -f "$RECORDINGS_FILE" ]]; then
  echo "Detect planner did not create recordings file: $RECORDINGS_FILE" >&2
  exit 2
fi

analysis_count=$(grep -cve '^[[:space:]]*$' "$RECORDINGS_FILE" || true)
if [[ "$analysis_count" == "0" ]]; then
  echo "No target zarrs discovered. Nothing to submit."
  exit 0
fi

QUALITY_SCRIPT="${RUN_DIR}/run_detect_quality.sh"
cat > "$QUALITY_SCRIPT" <<JOBSCRIPT
#!/usr/bin/env bash
set -euo pipefail
cd "$REPO_ROOT"
mapfile -t zarr_paths < "$RECORDINGS_FILE"
mkdir -p "$QUALITY_LOG_DIR"
for zarr_path in "\${zarr_paths[@]}"; do
  [[ -z "\$zarr_path" ]] && continue
  scripts/py -m fisheye.utils.detect_quality_batch "\$zarr_path" \\
    --apply \\
    --json \\
    --log-dir "$QUALITY_LOG_DIR" \\
    --threshold "$QUALITY_THRESHOLD" \\
    --threshold-mode "$QUALITY_THRESHOLD_MODE" \\
    --threshold-reference-width "$QUALITY_THRESHOLD_REFERENCE_WIDTH"
done
JOBSCRIPT
chmod +x "$QUALITY_SCRIPT"

REFINE_SCRIPT="${RUN_DIR}/run_refine_detect.sh"
cat > "$REFINE_SCRIPT" <<JOBSCRIPT
#!/usr/bin/env bash
set -euo pipefail
cd "$REPO_ROOT"
mapfile -t zarr_paths < "$RECORDINGS_FILE"
mkdir -p "$REFINE_LOG_DIR"
for zarr_path in "\${zarr_paths[@]}"; do
  [[ -z "\$zarr_path" ]] && continue
  cmd=(scripts/py -m fisheye.utils.refine_detect_batch "\$zarr_path" --apply --zarr-use analysis --config "$CONFIG" --log-dir "$REFINE_LOG_DIR")
  if [[ "$REFINE_SAVE_VISUALS" == "1" ]]; then
    cmd+=(--save-visuals)
  fi
  "\${cmd[@]}"
done
JOBSCRIPT
chmod +x "$REFINE_SCRIPT"

if [[ "$SUBMIT" == "1" ]]; then
  detect_jobid="$(palette_lsf_extract_jobid "$DETECT_SUBMIT_LOG")"
  if [[ -z "$detect_jobid" ]]; then
    echo "Could not parse detect job id from $DETECT_SUBMIT_LOG" >&2
    exit 2
  fi
else
  detect_jobid="<detect_jobid>"
fi

QUALITY_BSUB_ARGS=(
  bsub
  -J "detect_quality_${RUN_ID}"
  -n "$QUALITY_NCORES"
  -R "rusage[mem=${QUALITY_MEM_GB}G]"
  -W "$QUALITY_WALLTIME"
  -oo "${RUN_DIR}/detect_quality_%J.out"
  -eo "${RUN_DIR}/detect_quality_%J.err"
  -w "done(${detect_jobid})"
)
if [[ -n "$QUALITY_QUEUE" ]]; then QUALITY_BSUB_ARGS+=(-q "$QUALITY_QUEUE"); fi
QUALITY_BSUB_ARGS+=(bash "$QUALITY_SCRIPT")

QUALITY_SUBMIT_LOG="${RUN_DIR}/detect_quality_submit.log"
echo "Planning detect_quality stage..."
palette_lsf_submit_or_print "$SUBMIT" "$QUALITY_SUBMIT_LOG" "${QUALITY_BSUB_ARGS[@]}"

if [[ "$SUBMIT" == "1" ]]; then
  quality_jobid="$(palette_lsf_extract_jobid "$QUALITY_SUBMIT_LOG")"
  if [[ -z "$quality_jobid" ]]; then
    echo "Could not parse detect_quality job id from $QUALITY_SUBMIT_LOG" >&2
    exit 2
  fi
else
  quality_jobid="<detect_quality_jobid>"
fi

REFINE_BSUB_ARGS=(
  bsub
  -J "refine_detect_${RUN_ID}"
  -n "$REFINE_NCORES"
  -R "rusage[mem=${REFINE_MEM_GB}G]"
  -W "$REFINE_WALLTIME"
  -oo "${RUN_DIR}/refine_detect_%J.out"
  -eo "${RUN_DIR}/refine_detect_%J.err"
  -w "done(${quality_jobid})"
)
if [[ -n "$REFINE_QUEUE" ]]; then REFINE_BSUB_ARGS+=(-q "$REFINE_QUEUE"); fi
REFINE_BSUB_ARGS+=(bash "$REFINE_SCRIPT")

REFINE_SUBMIT_LOG="${RUN_DIR}/refine_detect_submit.log"
echo "Planning refined_detect stage..."
palette_lsf_submit_or_print "$SUBMIT" "$REFINE_SUBMIT_LOG" "${REFINE_BSUB_ARGS[@]}"

cat > "${RUN_DIR}/submission_summary.txt" <<SUMMARY
run_id=$RUN_ID
run_dir=$RUN_DIR
root=$ROOT
registry=$REGISTRY
path_contains=${PATH_CONTAINS:-<none>}
analysis_zarr_count=$analysis_count
detect_dependency_job=$detect_jobid
quality_dependency_job=$quality_jobid
recordings_file=$RECORDINGS_FILE
detect_submit_log=$DETECT_SUBMIT_LOG
quality_submit_log=$QUALITY_SUBMIT_LOG
refine_submit_log=$REFINE_SUBMIT_LOG
SUMMARY

echo "Run dir: $RUN_DIR"
echo "Targets: $analysis_count"
echo "Recordings file: $RECORDINGS_FILE"
echo "Summary: ${RUN_DIR}/submission_summary.txt"
if [[ "$SUBMIT" != "1" ]]; then
  echo "Dry run only; pass --submit to submit the dependency chain."
fi
