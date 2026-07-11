#!/usr/bin/env bash
set -euo pipefail

ZARR_PATH=""
SOURCE_REFINED_RUN=""
RUN_ID="subject_mask_collection_worker_init_$(date +%Y%m%d_%H%M%S)"
LOG_ROOT="/groups/johnson/johnsonlab/jeremy/recordings/logs/subject_mask_collection_worker_init"
QUEUE=""
SUBMIT_HOST="${PALETTE_LSF_SUBMIT_HOST:-}"
NCORES=8
MEM_GB_PER_SLOT=2
WALLTIME="0:30"
NUM_WORKERS=8
SAMPLE_ROWS=1
HOLD_SECONDS=15
DRY_RUN=0
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

usage() {
  cat <<'USAGE'
Usage: submit_subject_mask_collection_worker_init_bsub.sh --zarr PATH --source-refined-run RUN [options]

Submits a read-only LSF smoke that builds one collection identity plan in the
parent, initializes process workers from that compact plan, resolves real
keypoint assignment context, samples probabilities, and reports process RSS.
It never creates a refined run or updates the registry.

Options:
  --run-id ID          Run/log directory name.
  --log-root PATH      PRFS log root.
  --queue NAME         LSF queue (default: cluster default).
  --submit-host HOST   SSH host used when bsub is unavailable locally.
  --ncores N           LSF slots (default: 8).
  --mem-gb-per-slot N  LSF rusage memory per slot in GiB (default: 2; with
                       eight slots the client request is approximately 16 GiB;
                       site esub/application policy may raise the effective request).
  --mem-gb N           Compatibility alias for --mem-gb-per-slot.
  --walltime H:MM      Walltime (default: 0:30).
  --num-workers N      Initialization workers (default: 8).
  --sample-rows N      Probability/keypoint rows read per worker (default: 1).
  --hold-seconds N     Seconds workers retain initialized state (default: 15).
  --dry-run            Package but do not submit.
  -h, --help           Show this help.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --zarr) ZARR_PATH="$2"; shift 2;;
    --source-refined-run) SOURCE_REFINED_RUN="$2"; shift 2;;
    --run-id) RUN_ID="$2"; shift 2;;
    --log-root) LOG_ROOT="$2"; shift 2;;
    --queue) QUEUE="$2"; shift 2;;
    --submit-host) SUBMIT_HOST="$2"; shift 2;;
    --ncores) NCORES="$2"; shift 2;;
    --mem-gb-per-slot) MEM_GB_PER_SLOT="$2"; shift 2;;
    --mem-gb) MEM_GB_PER_SLOT="$2"; shift 2;;
    --walltime) WALLTIME="$2"; shift 2;;
    --num-workers) NUM_WORKERS="$2"; shift 2;;
    --sample-rows) SAMPLE_ROWS="$2"; shift 2;;
    --hold-seconds) HOLD_SECONDS="$2"; shift 2;;
    --dry-run) DRY_RUN=1; shift;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown argument: $1" >&2; usage; exit 2;;
  esac
done

if [[ -z "$ZARR_PATH" || -z "$SOURCE_REFINED_RUN" ]]; then
  echo "--zarr and --source-refined-run are required." >&2
  usage
  exit 2
fi
if [[ ! -f "${ZARR_PATH%/}/zarr.json" ]]; then
  echo "Zarr root not found: $ZARR_PATH" >&2
  exit 2
fi
for value in "$NCORES" "$MEM_GB_PER_SLOT" "$NUM_WORKERS" "$SAMPLE_ROWS" "$HOLD_SECONDS"; do
  if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
    echo "Expected a positive integer, got: $value" >&2
    exit 2
  fi
done

RUN_DIR="${LOG_ROOT%/}/${RUN_ID}"
if [[ -e "$RUN_DIR" ]]; then
  echo "Run directory already exists: $RUN_DIR" >&2
  exit 2
fi
mkdir -p "$RUN_DIR/source_snapshot" "$RUN_DIR/reports"
cp -R --no-preserve=mode,ownership,timestamps "$REPO_ROOT/src/fisheye" "$RUN_DIR/source_snapshot/"
cp --no-preserve=mode,ownership,timestamps "$REPO_ROOT/scripts/py" "$RUN_DIR/palette_py"
chmod +x "$RUN_DIR/palette_py"
git -C "$REPO_ROOT" status --short > "$RUN_DIR/source_git_status.txt"
git -C "$REPO_ROOT" diff --binary > "$RUN_DIR/source_worktree.patch"

cat > "$RUN_DIR/settings.sh" <<SETTINGS
ZARR_PATH=$(printf '%q' "$ZARR_PATH")
SOURCE_REFINED_RUN=$(printf '%q' "$SOURCE_REFINED_RUN")
NUM_WORKERS=$NUM_WORKERS
SAMPLE_ROWS=$SAMPLE_ROWS
HOLD_SECONDS=$HOLD_SECONDS
SETTINGS

cat > "$RUN_DIR/manifest.json" <<JSON
{
  "schema_id": "palette.subject_mask_collection_worker_init_submission.v1",
  "run_id": "$RUN_ID",
  "zarr_path": "$ZARR_PATH",
  "source_refined_run": "$SOURCE_REFINED_RUN",
  "num_workers": $NUM_WORKERS,
  "client_requested_memory_gib_per_slot": $MEM_GB_PER_SLOT,
  "client_requested_approximate_total_memory_gib": $((NCORES * MEM_GB_PER_SLOT)),
  "sample_rows": $SAMPLE_ROWS,
  "hold_seconds": $HOLD_SECONDS,
  "mutates_zarr": false,
  "mutates_registry": false
}
JSON

JOB_SCRIPT="$RUN_DIR/run_benchmark.sh"
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
echo "allocated_slots=${LSB_DJOB_NUMPROC:-}"
/usr/bin/time -f $'elapsed_seconds=%e\nmaximum_rss_kib=%M' \
  -o "$RUN_DIR/reports/driver.time.txt" \
  "$RUN_DIR/palette_py" -m fisheye.diagnostics.benchmark_subject_mask_collection_worker_init \
    "$ZARR_PATH" \
    --source-refined-run "$SOURCE_REFINED_RUN" \
    --num-workers "$NUM_WORKERS" \
    --sample-rows "$SAMPLE_ROWS" \
    --hold-seconds "$HOLD_SECONDS" \
    --output-json "$RUN_DIR/reports/worker_init.json"
JOBSCRIPT
chmod +x "$JOB_SCRIPT"

BSUB_ARGS=(
  -J "sm_collection_init"
  -n "$NCORES"
  -W "$WALLTIME"
  -R "rusage[mem=${MEM_GB_PER_SLOT}G]"
  -oo "$RUN_DIR/%J.out"
  -eo "$RUN_DIR/%J.err"
)
if [[ -n "$QUEUE" ]]; then
  BSUB_ARGS+=(-q "$QUEUE")
fi

printf 'Run dir: %s\n' "$RUN_DIR"
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
