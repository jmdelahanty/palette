#!/usr/bin/env bash
set -euo pipefail

ROOT="/groups/johnson/johnsonlab/jeremy/recordings"
REGISTRY="${PALETTE_REGISTRY_PATH:-/groups/johnson/johnsonlab/jeremy/registries/palette_registry.sqlite}"
SOURCE="registry"
PATH_CONTAINS=""
QUEUE=""
NCORES=16
MEM_GB=32
MAX_ACTIVE=4
RUN_ID=""
RUN_LABEL_OVERRIDE=""
LOG_DIR=""
DRY_RUN=0
REPO_DIR=""
FORCE_FINALIZATION=0
OVERWRITE=0
STAGE_OUTPUT_TO_SCRATCH=1
STAGE_FINALIZATION_INPUT_TO_SCRATCH=1
OUTPUT_STAGING_DIR=""
KEEP_STAGED_OUTPUT=0
HANDOFF_PACKAGE_DIR="${PALETTE_SUBJECT_MASK_HANDOFF_DIR:-/nrs/ahrens/palette_staging/subject_mask_run_packages}"
FINALIZE_CHUNK_SIZE=256
FINALIZE_DENSE_MASK_ROW_CHUNK=256
FINALIZE_EXECUTION_BACKEND="process_shards"
FINALIZE_NUM_WORKERS="16"
FINALIZE_POSTCOMPUTE_BACKEND="process_shards"
FINALIZE_POSTCOMPUTE_CHUNK_SIZE=""
FINALIZE_POSTCOMPUTE_NUM_WORKERS=""
METRIC_LEVEL="cheap"
MASK_STORAGE="dense_uint8"
MASK_RLE_VALIDATION_MODE="invariants"
WRITE_EYE_GEOMETRY=1
WRITE_COMPONENT_CONTOURS=1
WRITE_SAMPLED_COMPONENT_CONTOURS=0
RETAIN_SOURCE_SEEDS=0
PROFILE_TIMINGS=1

usage() {
  cat <<'USAGE'
Usage: submit_subject_mask_finalization_batches_bsub.sh [options]

Submit CPU-only refined subject-mask finalization jobs. This is the direct
finalization-only companion to submit_subject_mask_batches_bsub.sh: it does not
submit a no-op GPU inference dependency. Each array task calls
fisheye.utils.run_subject_mask_batch_pipeline with --workflow-stage finalization.

Discovery:
  --root PATH               Recording root (default: /groups/.../recordings)
  --registry PATH           Registry sqlite path (default: $PALETTE_REGISTRY_PATH or PRFS registry)
  --source registry|filesystem
                            Discovery source (default: registry)
  --path-contains STR       Keep zarr paths containing this substring

Resources:
  --queue NAME              LSF queue (default: cluster default)
  --ncores N                CPU slots per finalization task (default: 16)
  --mem-gb N                Memory per task in GB (default: 32)
  --max-active N            Max concurrent array tasks (default: 4)

Finalizer:
  --run-label LABEL         Run label passed to the Python batch driver
  --finalize-chunk-size N   Refined finalizer chunk size (default: 256)
  --finalize-dense-mask-row-chunk N
                            Physical dense masks_roi row chunk (default: 256)
  --finalize-num-workers N|auto
                            Worker count (default: 16; auto => --ncores)
  --finalize-postcompute-chunk-size N
                            Rows per postcompute shard
  --finalize-postcompute-num-workers N
                            Worker count for postcompute shards
  --metric-level LEVEL      cheap|full (default: cheap)
  --mask-storage MODE       dense_uint8|dense_and_bitpacked|dense_and_rle|dense_bitpacked_and_rle
                            (default: dense_uint8)
  --mask-rle-validation-mode MODE
                            full|invariants|none (default: invariants)
  --no-write-eye-geometry   Do not write eye geometry
  --write-component-contours
                            Write full ragged contours (default during Crimson migration)
  --write-sampled-component-contours
                            Also write the fixed-K sampled contour cache
  --no-write-sampled-component-contours
                            Disable the default fixed-K sampled contour cache
  --retain-source-seeds     Retain source_seed_masks_roi debug arrays
  --force-finalization      Re-finalize even if target refined run exists
  --overwrite               Replace an existing target run when finalizing

Staging:
  --no-stage-output-to-scratch
                            Write directly to target Zarr instead of local staged output
  --no-stage-finalization-input-to-scratch
                            Symlink/copy less input into staged output
  --output-staging-dir PATH Override local staged output root
  --keep-staged-output      Keep local staged output after publish
  --handoff-package-dir PATH
                            NRS handoff package directory

General:
  --log-dir PATH            Submission logs (default: <root>/logs/subject_mask_finalization_bsub)
  --run-id ID               Stable run id; default UTC timestamp
  --repo-dir PATH           Palette checkout visible to compute nodes (default: current directory)
  --no-profile-timings      Disable workflow profile JSONL
  --dry-run                 Print manifest and bsub command; do not submit
  -h, --help                Show this message
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --root) ROOT="$2"; shift 2;;
    --registry) REGISTRY="$2"; shift 2;;
    --source) SOURCE="$2"; shift 2;;
    --path-contains) PATH_CONTAINS="$2"; shift 2;;
    --queue) QUEUE="$2"; shift 2;;
    --ncores) NCORES="$2"; shift 2;;
    --mem-gb) MEM_GB="$2"; shift 2;;
    --max-active) MAX_ACTIVE="$2"; shift 2;;
    --run-label) RUN_LABEL_OVERRIDE="$2"; shift 2;;
    --finalize-chunk-size) FINALIZE_CHUNK_SIZE="$2"; shift 2;;
    --finalize-dense-mask-row-chunk) FINALIZE_DENSE_MASK_ROW_CHUNK="$2"; shift 2;;
    --finalize-num-workers) FINALIZE_NUM_WORKERS="$2"; shift 2;;
    --finalize-postcompute-chunk-size) FINALIZE_POSTCOMPUTE_CHUNK_SIZE="$2"; shift 2;;
    --finalize-postcompute-num-workers) FINALIZE_POSTCOMPUTE_NUM_WORKERS="$2"; shift 2;;
    --metric-level) METRIC_LEVEL="$2"; shift 2;;
    --mask-storage) MASK_STORAGE="$2"; shift 2;;
    --mask-rle-validation-mode) MASK_RLE_VALIDATION_MODE="$2"; shift 2;;
    --no-write-eye-geometry) WRITE_EYE_GEOMETRY=0; shift;;
    --write-component-contours) WRITE_COMPONENT_CONTOURS=1; shift;;
    --no-write-component-contours) WRITE_COMPONENT_CONTOURS=0; shift;;
    --write-sampled-component-contours) WRITE_SAMPLED_COMPONENT_CONTOURS=1; shift;;
    --no-write-sampled-component-contours) WRITE_SAMPLED_COMPONENT_CONTOURS=0; shift;;
    --retain-source-seeds) RETAIN_SOURCE_SEEDS=1; shift;;
    --force-finalization) FORCE_FINALIZATION=1; shift;;
    --overwrite) OVERWRITE=1; shift;;
    --no-stage-output-to-scratch) STAGE_OUTPUT_TO_SCRATCH=0; shift;;
    --no-stage-finalization-input-to-scratch) STAGE_FINALIZATION_INPUT_TO_SCRATCH=0; shift;;
    --output-staging-dir) OUTPUT_STAGING_DIR="$2"; shift 2;;
    --keep-staged-output) KEEP_STAGED_OUTPUT=1; shift;;
    --handoff-package-dir) HANDOFF_PACKAGE_DIR="$2"; shift 2;;
    --log-dir) LOG_DIR="$2"; shift 2;;
    --run-id) RUN_ID="$2"; shift 2;;
    --repo-dir) REPO_DIR="$2"; shift 2;;
    --no-profile-timings) PROFILE_TIMINGS=0; shift;;
    --dry-run) DRY_RUN=1; shift;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown arg: $1" >&2; usage; exit 2;;
  esac
done

if [[ "$SOURCE" != "registry" && "$SOURCE" != "filesystem" ]]; then
  echo "--source must be registry or filesystem." >&2
  exit 2
fi
for name_value in "ncores:$NCORES" "mem-gb:$MEM_GB" "max-active:$MAX_ACTIVE" "finalize-chunk-size:$FINALIZE_CHUNK_SIZE" "finalize-dense-mask-row-chunk:$FINALIZE_DENSE_MASK_ROW_CHUNK"; do
  name="${name_value%%:*}"
  value="${name_value#*:}"
  if ! [[ "$value" =~ ^[0-9]+$ ]] || [[ "$value" -lt 1 ]]; then
    echo "--$name must be a positive integer." >&2
    exit 2
  fi
done
if [[ "$FINALIZE_NUM_WORKERS" == "auto" || -z "$FINALIZE_NUM_WORKERS" ]]; then
  FINALIZE_NUM_WORKERS="$NCORES"
fi
if ! [[ "$FINALIZE_NUM_WORKERS" =~ ^[0-9]+$ ]] || [[ "$FINALIZE_NUM_WORKERS" -lt 1 ]]; then
  echo "--finalize-num-workers must be a positive integer or auto." >&2
  exit 2
fi
if [[ -n "$FINALIZE_POSTCOMPUTE_CHUNK_SIZE" ]]; then
  if ! [[ "$FINALIZE_POSTCOMPUTE_CHUNK_SIZE" =~ ^[0-9]+$ ]] || [[ "$FINALIZE_POSTCOMPUTE_CHUNK_SIZE" -lt 1 ]]; then
    echo "--finalize-postcompute-chunk-size must be a positive integer." >&2
    exit 2
  fi
fi
if [[ -n "$FINALIZE_POSTCOMPUTE_NUM_WORKERS" ]]; then
  if ! [[ "$FINALIZE_POSTCOMPUTE_NUM_WORKERS" =~ ^[0-9]+$ ]] || [[ "$FINALIZE_POSTCOMPUTE_NUM_WORKERS" -lt 1 ]]; then
    echo "--finalize-postcompute-num-workers must be a positive integer." >&2
    exit 2
  fi
fi
if [[ "$FINALIZE_NUM_WORKERS" -gt "$NCORES" ]]; then
  echo "Warning: --finalize-num-workers (${FINALIZE_NUM_WORKERS}) exceeds --ncores (${NCORES}); this can oversubscribe the LSF allocation." >&2
fi
if [[ -z "$RUN_ID" ]]; then
  RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
fi
if [[ -z "$RUN_LABEL_OVERRIDE" ]]; then
  RUN_LABEL="subject_mask_finalization_${RUN_ID}"
else
  RUN_LABEL="$RUN_LABEL_OVERRIDE"
fi
if [[ -z "$LOG_DIR" ]]; then
  LOG_DIR="${ROOT%/}/logs/subject_mask_finalization_bsub"
fi
if [[ -z "$REPO_DIR" ]]; then
  REPO_DIR="$(pwd)"
fi

RUN_DIR="${LOG_DIR%/}/sm_finalize_${RUN_ID}"
if [[ -e "$RUN_DIR" ]]; then
  echo "Run directory already exists: $RUN_DIR" >&2
  echo "Choose a different --run-id or --log-dir." >&2
  exit 2
fi
mkdir -p "$RUN_DIR/progress" "$RUN_DIR/reports"

DISCOVER_ARGS=(--source "$SOURCE" --emit-paths --registry "$REGISTRY" --workflow-stage finalization --subject-output-parent subject_mask_runs)
DISCOVER_ARGS+=(--run-label "$RUN_LABEL")
if [[ -n "$PATH_CONTAINS" ]]; then DISCOVER_ARGS+=(--path-contains "$PATH_CONTAINS"); fi
if [[ "$FORCE_FINALIZATION" == "1" ]]; then DISCOVER_ARGS+=(--force-finalization); fi
(cd "$REPO_DIR" && scripts/py -m fisheye.utils.run_subject_mask_batch_pipeline "$ROOT" "${DISCOVER_ARGS[@]}") > "$RUN_DIR/targets.tsv"

job_count="$(wc -l < "$RUN_DIR/targets.tsv" | tr -d ' ')"
if [[ "$job_count" == "0" ]]; then
  echo "No subject-mask finalization targets selected."
  exit 0
fi

SUBJECT_ARGS=(
  --apply
  --registry "$REGISTRY"
  --run-label "$RUN_LABEL"
  --workflow-stage finalization
  --subject-output-parent subject_mask_runs
  --metric-level "$METRIC_LEVEL"
  --mask-storage "$MASK_STORAGE"
  --mask-rle-validation-mode "$MASK_RLE_VALIDATION_MODE"
  --finalize-chunk-size "$FINALIZE_CHUNK_SIZE"
  --finalize-dense-mask-row-chunk "$FINALIZE_DENSE_MASK_ROW_CHUNK"
  --finalize-execution-backend "$FINALIZE_EXECUTION_BACKEND"
  --finalize-num-workers "$FINALIZE_NUM_WORKERS"
  --finalize-postcompute-backend "$FINALIZE_POSTCOMPUTE_BACKEND"
  --progress-dir "$RUN_DIR/progress"
)
if [[ -n "$FINALIZE_POSTCOMPUTE_CHUNK_SIZE" ]]; then SUBJECT_ARGS+=(--finalize-postcompute-chunk-size "$FINALIZE_POSTCOMPUTE_CHUNK_SIZE"); fi
if [[ -n "$FINALIZE_POSTCOMPUTE_NUM_WORKERS" ]]; then SUBJECT_ARGS+=(--finalize-postcompute-num-workers "$FINALIZE_POSTCOMPUTE_NUM_WORKERS"); fi
if [[ "$WRITE_EYE_GEOMETRY" == "1" ]]; then SUBJECT_ARGS+=(--write-eye-geometry); else SUBJECT_ARGS+=(--no-write-eye-geometry); fi
if [[ "$WRITE_COMPONENT_CONTOURS" == "1" ]]; then SUBJECT_ARGS+=(--write-component-contours); else SUBJECT_ARGS+=(--no-write-component-contours); fi
if [[ "$WRITE_SAMPLED_COMPONENT_CONTOURS" == "1" ]]; then SUBJECT_ARGS+=(--write-sampled-component-contours); else SUBJECT_ARGS+=(--no-write-sampled-component-contours); fi
if [[ "$RETAIN_SOURCE_SEEDS" == "1" ]]; then SUBJECT_ARGS+=(--retain-source-seeds); fi
if [[ "$FORCE_FINALIZATION" == "1" ]]; then SUBJECT_ARGS+=(--force-finalization); fi
if [[ "$OVERWRITE" == "1" ]]; then SUBJECT_ARGS+=(--overwrite); fi
if [[ "$STAGE_OUTPUT_TO_SCRATCH" == "1" ]]; then SUBJECT_ARGS+=(--stage-output-to-scratch); fi
if [[ "$STAGE_FINALIZATION_INPUT_TO_SCRATCH" == "1" ]]; then SUBJECT_ARGS+=(--stage-finalization-input-to-scratch); fi
if [[ -n "$OUTPUT_STAGING_DIR" ]]; then SUBJECT_ARGS+=(--output-staging-dir "$OUTPUT_STAGING_DIR"); fi
if [[ "$KEEP_STAGED_OUTPUT" == "1" ]]; then SUBJECT_ARGS+=(--keep-staged-output); fi
if [[ -n "$HANDOFF_PACKAGE_DIR" ]]; then SUBJECT_ARGS+=(--handoff-package-dir "$HANDOFF_PACKAGE_DIR"); fi
if [[ "$PROFILE_TIMINGS" == "1" ]]; then SUBJECT_ARGS+=(--profile-timings); fi

{
  echo "SUBJECT_ARGS=("
  printf '  %q\n' "${SUBJECT_ARGS[@]}"
  echo ")"
  printf 'RUN_LABEL=%q\n' "$RUN_LABEL"
  printf 'REPO_DIR=%q\n' "$REPO_DIR"
} > "$RUN_DIR/subject_args.sh"

JOB_SCRIPT="${RUN_DIR}/run_one_finalization.sh"
cat > "$JOB_SCRIPT" <<'JOBSCRIPT'
#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="$1"
if [[ -z "${LSB_JOBINDEX:-}" ]]; then
  echo "LSB_JOBINDEX not set; are you running under bsub array?" >&2
  exit 2
fi
source "${RUN_DIR}/subject_args.sh"
zarr_path="$(sed -n "${LSB_JOBINDEX}p" "${RUN_DIR}/targets.tsv")"
if [[ -z "$zarr_path" ]]; then
  echo "No zarr path for array index ${LSB_JOBINDEX}" >&2
  exit 2
fi
cd "$REPO_DIR"

report_stem="$(basename "$zarr_path" .zarr)"
cmd=(scripts/py -m fisheye.utils.run_subject_mask_batch_pipeline "$zarr_path" "${SUBJECT_ARGS[@]}"
  --json-report "${RUN_DIR}/reports/${report_stem}_finalization.json"
  --markdown-report "${RUN_DIR}/reports/${report_stem}_finalization.md")

echo "host=$(hostname)"
echo "job_id=${LSB_JOBID:-}"
echo "job_index=${LSB_JOBINDEX:-}"
echo "zarr_path=${zarr_path}"
echo "run_label=${RUN_LABEL}"
printf '+ %q ' "${cmd[@]}"
printf '\n'
"${cmd[@]}"
JOBSCRIPT
chmod +x "$JOB_SCRIPT"

BSUB_ARGS=(-J "sm_finalize[1-${job_count}]%${MAX_ACTIVE}" -n "$NCORES" -R "rusage[mem=${MEM_GB}G]" -oo "${RUN_DIR}/finalize_%J_%I.out" -eo "${RUN_DIR}/finalize_%J_%I.err")
if [[ -n "$QUEUE" ]]; then BSUB_ARGS+=(-q "$QUEUE"); fi

format_bsub_cmd() {
  local cmd="bsub"
  local arg
  for arg in "${BSUB_ARGS[@]}"; do
    cmd+=" $(printf '%q' "$arg")"
  done
  cmd+=" bash $(printf '%q' "$JOB_SCRIPT") $(printf '%q' "$RUN_DIR")"
  printf '%s\n' "$cmd"
}

BSUB_CMD="$(format_bsub_cmd)"

echo "Run dir: $RUN_DIR"
echo "Source: $SOURCE"
echo "Root: $ROOT"
echo "Registry: $REGISTRY"
echo "Jobs: $job_count (one finalization task per selected zarr)"
echo "Max active: $MAX_ACTIVE"
echo "Queue: ${QUEUE:-<default>}"
echo "Resources: ncores=$NCORES mem_gb=$MEM_GB"
echo "Run label: $RUN_LABEL"
echo "Finalizer: chunk_size=$FINALIZE_CHUNK_SIZE workers=$FINALIZE_NUM_WORKERS backend=$FINALIZE_EXECUTION_BACKEND"
echo "Finalizer dense mask row chunk: $FINALIZE_DENSE_MASK_ROW_CHUNK"
echo "Finalizer postcompute: backend=$FINALIZE_POSTCOMPUTE_BACKEND chunk_size=${FINALIZE_POSTCOMPUTE_CHUNK_SIZE:-<finalizer default>} workers=${FINALIZE_POSTCOMPUTE_NUM_WORKERS:-<finalizer default>}"
echo "Refined mask storage: $MASK_STORAGE"
echo "Stage output to scratch: $STAGE_OUTPUT_TO_SCRATCH"
echo "Stage finalization input to scratch: $STAGE_FINALIZATION_INPUT_TO_SCRATCH"
echo "Handoff package dir: ${HANDOFF_PACKAGE_DIR:-<none>}"
echo "Profile timings: $PROFILE_TIMINGS"
echo "Manifest file: $RUN_DIR/targets.tsv"
echo "Submit command: $BSUB_CMD"

if [[ "$DRY_RUN" == "1" ]]; then
  echo "Dry run only; no submission."
  exit 0
fi

if ! command -v bsub >/dev/null 2>&1; then
  echo "bsub not found in PATH. Is this an LSF cluster login node?" >&2
  exit 2
fi

bsub "${BSUB_ARGS[@]}" bash "$JOB_SCRIPT" "$RUN_DIR"
