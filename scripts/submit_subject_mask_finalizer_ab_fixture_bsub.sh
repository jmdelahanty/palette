#!/usr/bin/env bash
set -euo pipefail

SOURCE_ZARR=""
SOURCE_SHARD_RUN=""
TARGET_CROP_RUN=""
ASSIGNMENT_KEYPOINT_GROUP="refined_keypoints_runs"
ASSIGNMENT_KEYPOINTS_RUN=""
OUTPUT_ROOT=""
RUN_ID="subject_mask_finalizer_ab_fixture_$(date +%Y%m%d_%H%M%S)"
LOG_ROOT="/groups/johnson/johnsonlab/jeremy/recordings/logs/subject_mask_finalizer_ab_fixtures"
QUEUE=""
SUBMIT_HOST="${PALETTE_LSF_SUBMIT_HOST:-}"
NCORES=1
MEM_GB_PER_SLOT=8
WALLTIME="2:00"
INNER_CHUNK_ROWS=32
SHARD_ROWS=2048
DRY_RUN=0
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

usage() {
  cat <<'USAGE'
Usage: submit_subject_mask_finalizer_ab_fixture_bsub.sh --source-zarr PATH \
  --source-shard-run RUN --target-crop-run RUN --assignment-keypoints-run RUN \
  --output-root PATH [options]

Builds two benchmark-only, contract-complete PRFS Zarrs for one raw subject-mask
shard. All logical data are identical; mask_probs_roi is regular in A and uses
2,048-row indexed sharding in B. The job validates exact probability digests and
production-finalizer dry-run contracts. It does not mutate the source archive or
registry.

Options:
  --assignment-keypoint-group GROUP  refined_keypoints_runs|keypoints_runs.
  --run-id ID                        Run/log directory name.
  --log-root PATH                    PRFS log root.
  --queue NAME                       LSF queue.
  --submit-host HOST                 SSH host used when bsub is unavailable.
  --ncores N                         LSF slots (default: 1).
  --mem-gb-per-slot N                Client memory request per slot (default: 8).
  --walltime H:MM                    Walltime (default: 2:00).
  --inner-chunk-rows N               Inner probability rows (default: 32).
  --shard-rows N                     Outer storage shard rows (default: 2048).
  --dry-run                          Package but do not submit.
  -h, --help                         Show this help.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --source-zarr) SOURCE_ZARR="$2"; shift 2;;
    --source-shard-run) SOURCE_SHARD_RUN="$2"; shift 2;;
    --target-crop-run) TARGET_CROP_RUN="$2"; shift 2;;
    --assignment-keypoint-group) ASSIGNMENT_KEYPOINT_GROUP="$2"; shift 2;;
    --assignment-keypoints-run) ASSIGNMENT_KEYPOINTS_RUN="$2"; shift 2;;
    --output-root) OUTPUT_ROOT="$2"; shift 2;;
    --run-id) RUN_ID="$2"; shift 2;;
    --log-root) LOG_ROOT="$2"; shift 2;;
    --queue) QUEUE="$2"; shift 2;;
    --submit-host) SUBMIT_HOST="$2"; shift 2;;
    --ncores) NCORES="$2"; shift 2;;
    --mem-gb-per-slot) MEM_GB_PER_SLOT="$2"; shift 2;;
    --walltime) WALLTIME="$2"; shift 2;;
    --inner-chunk-rows) INNER_CHUNK_ROWS="$2"; shift 2;;
    --shard-rows) SHARD_ROWS="$2"; shift 2;;
    --dry-run) DRY_RUN=1; shift;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown argument: $1" >&2; usage; exit 2;;
  esac
done

if [[ -z "$SOURCE_ZARR" || -z "$SOURCE_SHARD_RUN" || -z "$TARGET_CROP_RUN" || -z "$ASSIGNMENT_KEYPOINTS_RUN" || -z "$OUTPUT_ROOT" ]]; then
  echo "All source, crop, keypoint, and output arguments are required." >&2
  usage
  exit 2
fi
if [[ "$ASSIGNMENT_KEYPOINT_GROUP" != "refined_keypoints_runs" && "$ASSIGNMENT_KEYPOINT_GROUP" != "keypoints_runs" ]]; then
  echo "Invalid assignment keypoint group: $ASSIGNMENT_KEYPOINT_GROUP" >&2
  exit 2
fi
if [[ ! -f "${SOURCE_ZARR%/}/zarr.json" ]]; then
  echo "Source Zarr not found: $SOURCE_ZARR" >&2
  exit 2
fi
for value in "$NCORES" "$MEM_GB_PER_SLOT" "$INNER_CHUNK_ROWS" "$SHARD_ROWS"; do
  if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
    echo "Expected a positive integer, got: $value" >&2
    exit 2
  fi
done

RUN_DIR="${LOG_ROOT%/}/${RUN_ID}"
if [[ -e "$RUN_DIR" || -e "$OUTPUT_ROOT" ]]; then
  echo "Run directory or fixture output already exists." >&2
  exit 2
fi
mkdir -p "$RUN_DIR/source_snapshot" "$RUN_DIR/reports"
cp -R --no-preserve=mode,ownership,timestamps "$REPO_ROOT/src/fisheye" "$RUN_DIR/source_snapshot/"
cp --no-preserve=mode,ownership,timestamps "$REPO_ROOT/scripts/py" "$RUN_DIR/palette_py"
chmod +x "$RUN_DIR/palette_py"
git -C "$REPO_ROOT" status --short > "$RUN_DIR/source_git_status.txt"
git -C "$REPO_ROOT" diff --binary > "$RUN_DIR/source_worktree.patch"

cat > "$RUN_DIR/settings.sh" <<SETTINGS
SOURCE_ZARR=$(printf '%q' "$SOURCE_ZARR")
SOURCE_SHARD_RUN=$(printf '%q' "$SOURCE_SHARD_RUN")
TARGET_CROP_RUN=$(printf '%q' "$TARGET_CROP_RUN")
ASSIGNMENT_KEYPOINT_GROUP=$(printf '%q' "$ASSIGNMENT_KEYPOINT_GROUP")
ASSIGNMENT_KEYPOINTS_RUN=$(printf '%q' "$ASSIGNMENT_KEYPOINTS_RUN")
OUTPUT_ROOT=$(printf '%q' "$OUTPUT_ROOT")
INNER_CHUNK_ROWS=$INNER_CHUNK_ROWS
SHARD_ROWS=$SHARD_ROWS
SETTINGS

cat > "$RUN_DIR/manifest.json" <<JSON
{
  "schema_id": "palette.subject_mask_finalizer_ab_fixture_submission.v1",
  "run_id": "$RUN_ID",
  "source_zarr": "$SOURCE_ZARR",
  "source_shard_run": "$SOURCE_SHARD_RUN",
  "target_crop_run": "$TARGET_CROP_RUN",
  "assignment_keypoint_group": "$ASSIGNMENT_KEYPOINT_GROUP",
  "assignment_keypoints_run": "$ASSIGNMENT_KEYPOINTS_RUN",
  "output_root": "$OUTPUT_ROOT",
  "inner_chunk_rows": $INNER_CHUNK_ROWS,
  "shard_rows": $SHARD_ROWS,
  "mutates_source_zarr": false,
  "mutates_registry": false
}
JSON

JOB_SCRIPT="$RUN_DIR/run_fixture.sh"
cat > "$JOB_SCRIPT" <<'JOBSCRIPT'
#!/usr/bin/env bash
set -euo pipefail
RUN_DIR="$1"
source "$RUN_DIR/settings.sh"
export PYTHONPATH="$RUN_DIR/source_snapshot${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export TBB_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "host=$(hostname)"
echo "job_id=${LSB_JOBID:-}"
/usr/bin/time -f $'elapsed_seconds=%e\nmaximum_rss_kib=%M' \
  -o "$RUN_DIR/reports/driver.time.txt" \
  "$RUN_DIR/palette_py" -m fisheye.diagnostics.build_subject_mask_finalizer_ab_fixture \
    "$SOURCE_ZARR" \
    --source-shard-run "$SOURCE_SHARD_RUN" \
    --target-crop-run "$TARGET_CROP_RUN" \
    --assignment-keypoint-group "$ASSIGNMENT_KEYPOINT_GROUP" \
    --assignment-keypoints-run "$ASSIGNMENT_KEYPOINTS_RUN" \
    --output-root "$OUTPUT_ROOT" \
    --inner-chunk-rows "$INNER_CHUNK_ROWS" \
    --shard-rows "$SHARD_ROWS" \
    --require-output-storage-tier prfs
cp --no-preserve=mode,ownership,timestamps "$OUTPUT_ROOT/fixture_manifest.json" "$RUN_DIR/reports/fixture_manifest.json"
JOBSCRIPT
chmod +x "$JOB_SCRIPT"

BSUB_ARGS=(-J "sm_finalizer_ab_fixture" -n "$NCORES" -W "$WALLTIME" -R "rusage[mem=${MEM_GB_PER_SLOT}G]" -oo "$RUN_DIR/%J.out" -eo "$RUN_DIR/%J.err")
if [[ -n "$QUEUE" ]]; then BSUB_ARGS+=(-q "$QUEUE"); fi
printf 'Run dir: %s\n' "$RUN_DIR"
printf 'Fixture output: %s\n' "$OUTPUT_ROOT"
printf 'Command: bsub'
for arg in "${BSUB_ARGS[@]}"; do printf ' %q' "$arg"; done
printf ' bash %q %q\n' "$JOB_SCRIPT" "$RUN_DIR"
if [[ "$DRY_RUN" == "1" ]]; then exit 0; fi
if command -v bsub >/dev/null 2>&1; then
  bsub "${BSUB_ARGS[@]}" bash "$JOB_SCRIPT" "$RUN_DIR"
elif [[ -n "$SUBMIT_HOST" ]]; then
  ssh "$SUBMIT_HOST" bsub "${BSUB_ARGS[@]}" bash "$JOB_SCRIPT" "$RUN_DIR"
else
  echo "bsub is unavailable locally; rerun with --submit-host HOST or submit on an LSF login node." >&2
  exit 127
fi
