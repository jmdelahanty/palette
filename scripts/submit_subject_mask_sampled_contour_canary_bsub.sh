#!/usr/bin/env bash
set -euo pipefail

SOURCE_ANALYSIS_ZARR=""
SOURCE_REFINED_RUN=""
DESTINATION=""
CANARY_ID=""
CACHE_RUN_ID=""
REPO_DIR="${PALETTE_GROUPS_REPO:-/groups/johnson/johnsonlab/jeremy/gitrepos/palette}"
EXPECTED_COMMIT=""
LOG_ROOT="/groups/johnson/johnsonlab/jeremy/recordings/logs/subject_mask_sampled_contour_canaries"
SUBMIT_HOST="${PALETTE_LSF_SUBMIT_HOST:-login1-citrus-poller}"
QUEUE="local"
NCORES=4
MEM_GB=16
WALLTIME="8:00"
SOURCE_COMPUTE_BLOCK_BYTES=268435456
DRY_RUN=0

usage() {
  cat <<'USAGE'
Usage: submit_subject_mask_sampled_contour_canary_bsub.sh [required options] [options]

Stages one complete immutable refined subject-mask run from PRFS to node-local
scratch, derives and validates the fixed-K sampled-contour cache there, then
copies the selector-ineligible artifact to a hidden sibling and atomically
renames it into the benchmark namespace. The source archive, selectors, and
registries are never mutated.

Required:
  --source-analysis-zarr PATH  Source analysis Zarr containing the refined run.
  --source-refined-run ID      Complete refined_subject_masks_runs child.
  --destination PATH           New artifact path below .palette_benchmarks.
  --canary-id ID               Stable canary identifier.
  --cache-run-id ID            New subject_mask_cache_runs child identifier.

Options:
  --repo PATH                  Clean compute-visible Palette checkout.
  --expected-commit SHA        Require this exact Palette commit.
  --log-root PATH              PRFS log root.
  --submit-host HOST           Host used when bsub is unavailable locally.
  --queue NAME                 LSF queue (default: local).
  --ncores N                   CPU slots (default: 4).
  --mem-gb N                   Memory request in GiB (default: 16).
  --walltime H:MM              Walltime (default: 8:00).
  --source-compute-block-bytes N
                               Maximum dense source block (default: 256 MiB).
  --dry-run                    Validate and print without submitting.
  -h, --help                   Show this help.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --source-analysis-zarr) SOURCE_ANALYSIS_ZARR="$2"; shift 2;;
    --source-refined-run) SOURCE_REFINED_RUN="$2"; shift 2;;
    --destination) DESTINATION="$2"; shift 2;;
    --canary-id) CANARY_ID="$2"; shift 2;;
    --cache-run-id) CACHE_RUN_ID="$2"; shift 2;;
    --repo) REPO_DIR="$2"; shift 2;;
    --expected-commit) EXPECTED_COMMIT="$2"; shift 2;;
    --log-root) LOG_ROOT="$2"; shift 2;;
    --submit-host) SUBMIT_HOST="$2"; shift 2;;
    --queue) QUEUE="$2"; shift 2;;
    --ncores) NCORES="$2"; shift 2;;
    --mem-gb) MEM_GB="$2"; shift 2;;
    --walltime) WALLTIME="$2"; shift 2;;
    --source-compute-block-bytes) SOURCE_COMPUTE_BLOCK_BYTES="$2"; shift 2;;
    --dry-run) DRY_RUN=1; shift;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown argument: $1" >&2; usage; exit 2;;
  esac
done

for value in SOURCE_ANALYSIS_ZARR SOURCE_REFINED_RUN DESTINATION CANARY_ID CACHE_RUN_ID; do
  if [[ -z "${!value}" ]]; then
    echo "Missing required option for ${value}." >&2
    usage
    exit 2
  fi
done
for value in "$NCORES" "$MEM_GB" "$SOURCE_COMPUTE_BLOCK_BYTES"; do
  if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
    echo "Expected a positive integer, got: $value" >&2
    exit 2
  fi
done
for identifier in "$SOURCE_REFINED_RUN" "$CANARY_ID" "$CACHE_RUN_ID"; do
  if [[ "$identifier" == */* || "$identifier" == "." || "$identifier" == ".." ]]; then
    echo "Unsafe identifier: $identifier" >&2
    exit 2
  fi
done
if [[ "$DESTINATION" != */.palette_benchmarks/* ]]; then
  echo "--destination must be below .palette_benchmarks." >&2
  exit 2
fi
if [[ ! -d "$SOURCE_ANALYSIS_ZARR/refined_subject_masks_runs/$SOURCE_REFINED_RUN" ]]; then
  echo "Source refined run not found." >&2
  exit 2
fi
if [[ -e "$DESTINATION" ]]; then
  echo "Destination already exists: $DESTINATION" >&2
  exit 2
fi
if [[ ! -d "$REPO_DIR/.git" && ! -f "$REPO_DIR/.git" ]]; then
  echo "Palette checkout not found: $REPO_DIR" >&2
  exit 2
fi
if [[ -n "$(git -C "$REPO_DIR" status --porcelain)" ]]; then
  echo "Compute-visible Palette checkout must be clean: $REPO_DIR" >&2
  exit 2
fi
PALETTE_COMMIT="$(git -C "$REPO_DIR" rev-parse HEAD)"
if [[ -n "$EXPECTED_COMMIT" && "$PALETTE_COMMIT" != "$EXPECTED_COMMIT" ]]; then
  echo "Expected commit $EXPECTED_COMMIT but found $PALETTE_COMMIT." >&2
  exit 2
fi

RUN_DIR="${LOG_ROOT%/}/${CANARY_ID}"
if [[ -e "$RUN_DIR" ]]; then
  echo "Canary log directory already exists: $RUN_DIR" >&2
  exit 2
fi
mkdir -p "$RUN_DIR"

SETTINGS="$RUN_DIR/settings.sh"
{
  printf 'SOURCE_ANALYSIS_ZARR=%q\n' "$SOURCE_ANALYSIS_ZARR"
  printf 'SOURCE_REFINED_RUN=%q\n' "$SOURCE_REFINED_RUN"
  printf 'DESTINATION=%q\n' "$DESTINATION"
  printf 'CANARY_ID=%q\n' "$CANARY_ID"
  printf 'CACHE_RUN_ID=%q\n' "$CACHE_RUN_ID"
  printf 'REPO_DIR=%q\n' "$REPO_DIR"
  printf 'PALETTE_COMMIT=%q\n' "$PALETTE_COMMIT"
  printf 'SOURCE_COMPUTE_BLOCK_BYTES=%q\n' "$SOURCE_COMPUTE_BLOCK_BYTES"
} > "$SETTINGS"

JOB_SCRIPT="$RUN_DIR/run_canary.sh"
cat > "$JOB_SCRIPT" <<'JOBSCRIPT'
#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="$1"
source "$RUN_DIR/settings.sh"
USER_NAME="${USER:-unknown}"
JOB_ID="${LSB_JOBID:-manual}"
if [[ -d "/scratch/${USER_NAME}" && -w "/scratch/${USER_NAME}" ]]; then
  SCRATCH_BASE="/scratch/${USER_NAME}/${JOB_ID}/sampled_contour_canary"
else
  SCRATCH_BASE="${TMPDIR:-/tmp}/palette_sampled_contour_canary_${JOB_ID}"
fi
SCRATCH_WORK="$SCRATCH_BASE/work"

cleanup() {
  local status=$?
  trap - EXIT INT TERM
  if [[ -d "$SCRATCH_BASE" ]]; then
    rm -rf "$SCRATCH_BASE"
  fi
  if [[ "$status" -eq 0 ]]; then
    printf 'complete\n' > "$RUN_DIR/status.txt"
  else
    printf 'failed:%s\n' "$status" > "$RUN_DIR/status.txt"
  fi
  exit "$status"
}
trap cleanup EXIT INT TERM

mkdir -p "$SCRATCH_BASE"
printf 'running\n' > "$RUN_DIR/status.txt"
if [[ -n "$(git -C "$REPO_DIR" status --porcelain)" ]]; then
  echo "Palette checkout became dirty: $REPO_DIR" >&2
  exit 2
fi
ACTUAL_COMMIT="$(git -C "$REPO_DIR" rev-parse HEAD)"
if [[ "$ACTUAL_COMMIT" != "$PALETTE_COMMIT" ]]; then
  echo "Palette commit changed: expected $PALETTE_COMMIT, got $ACTUAL_COMMIT" >&2
  exit 2
fi

export PYTHONPYCACHEPREFIX="$SCRATCH_BASE/pycache"
export OMP_NUM_THREADS="$LSB_DJOB_NUMPROC"
export OPENCV_FOR_THREADS_NUM="$LSB_DJOB_NUMPROC"
cd "$REPO_DIR"
/usr/bin/time -v -o "$RUN_DIR/resource_usage.txt" \
  scripts/py -m fisheye.diagnostics.publish_subject_mask_sampled_contour_canary \
    --source-analysis-zarr "$SOURCE_ANALYSIS_ZARR" \
    --source-refined-run "$SOURCE_REFINED_RUN" \
    --destination "$DESTINATION" \
    --scratch-root "$SCRATCH_WORK" \
    --canary-id "$CANARY_ID" \
    --cache-run-id "$CACHE_RUN_ID" \
    --palette-commit "$PALETTE_COMMIT" \
    --source-compute-block-bytes "$SOURCE_COMPUTE_BLOCK_BYTES" \
    --compute-workers "$LSB_DJOB_NUMPROC" \
    | tee "$RUN_DIR/result.jsonl"
JOBSCRIPT
chmod +x "$JOB_SCRIPT"

cat > "$RUN_DIR/submission_manifest.json" <<JSON
{
  "schema_id": "palette.subject_mask.sampled_contour_canary_lsf_submission",
  "schema_version": 1,
  "canary_id": "$CANARY_ID",
  "cache_run_id": "$CACHE_RUN_ID",
  "source_analysis_zarr": "$SOURCE_ANALYSIS_ZARR",
  "source_refined_run": "$SOURCE_REFINED_RUN",
  "destination": "$DESTINATION",
  "palette_commit": "$PALETTE_COMMIT",
  "repo_dir": "$REPO_DIR",
  "source_compute_block_bytes": $SOURCE_COMPUTE_BLOCK_BYTES,
  "ncores": $NCORES,
  "memory_gib": $MEM_GB,
  "walltime": "$WALLTIME",
  "queue": "$QUEUE",
  "execution": "prfs_source_to_node_local_compute_to_atomic_prfs_publication"
}
JSON

BSUB_ARGS=(
  -J "sm_contour_${CANARY_ID}"
  -n "$NCORES"
  -W "$WALLTIME"
  -R "rusage[mem=${MEM_GB}G]"
  -oo "$RUN_DIR/%J.out"
  -eo "$RUN_DIR/%J.err"
)
if [[ -n "$QUEUE" ]]; then
  BSUB_ARGS+=(-q "$QUEUE")
fi

printf 'Palette commit: %s\n' "$PALETTE_COMMIT"
printf 'Run directory: %s\n' "$RUN_DIR"
printf 'Destination: %s\n' "$DESTINATION"
printf 'Command: bsub'
printf ' %q' "${BSUB_ARGS[@]}"
printf ' bash %q %q\n' "$JOB_SCRIPT" "$RUN_DIR"

if [[ "$DRY_RUN" == "1" ]]; then
  exit 0
fi
if command -v bsub >/dev/null 2>&1; then
  bsub "${BSUB_ARGS[@]}" bash "$JOB_SCRIPT" "$RUN_DIR"
elif [[ -n "$SUBMIT_HOST" ]]; then
  ssh "$SUBMIT_HOST" bsub "${BSUB_ARGS[@]}" bash "$JOB_SCRIPT" "$RUN_DIR"
else
  echo "bsub unavailable; pass --submit-host or submit from an LSF login node." >&2
  exit 2
fi
