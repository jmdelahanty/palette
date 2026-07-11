#!/usr/bin/env bash
set -euo pipefail

INPUT_ROOT=""
RUN_ID="subject_mask_probability_sharding_$(date +%Y%m%d_%H%M%S)"
LOG_ROOT="/groups/johnson/johnsonlab/jeremy/recordings/logs/subject_mask_probability_sharding_benchmarks"
OUTPUT_BASE="/groups/johnson/johnsonlab/jeremy/recordings/.palette_benchmarks/lsf"
QUEUE=""
SUBMIT_HOST="${PALETTE_LSF_SUBMIT_HOST:-}"
NCORES=1
MEM_GB=16
WALLTIME="1:00"
READ_REPEATS=7
BATCH_ROWS=256
SAMPLE_ROWS=8192
INNER_CHUNK_ROWS=32
RANDOM_READ_COUNT=32
RANDOM_SEED=20260710
DRY_RUN=0
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

usage() {
  cat <<'USAGE'
Usage: submit_subject_mask_probability_sharding_benchmark_bsub.sh --input-root PATH [options]

Submits one CPU job that measures direct PRFS reads of an existing probability-
sharding benchmark set, stages its exact regular fixture to node-local scratch,
then writes regular and 512/2048/4096/8192-row variants from scratch back to a
job-specific PRFS directory. Each write uses a separate process so peak RSS is
attributable to one layout.

Options:
  --input-root PATH    Existing PRFS benchmark set containing regular.zarr and
                       benchmark_set.json (required).
  --run-id ID          Submission/run identifier.
  --log-root PATH      PRFS log root.
  --output-base PATH   PRFS parent for job-specific write-back outputs.
  --queue NAME         LSF queue (default: cluster default).
  --submit-host HOST   SSH host on which to invoke bsub when it is unavailable
                       locally (for example: login1-citrus-poller).
  --ncores N           Allocated CPU slots (default: 1).
  --mem-gb N           Memory request in GiB (default: 16).
  --walltime H:MM      LSF walltime (default: 1:00).
  --read-repeats N     Randomized direct-PRFS read rounds (default: 7).
  --batch-rows N       Rows per component read (default: 256).
  --sample-rows N      Source rows written per layout (default: 8192).
  --inner-chunk-rows N Inner probability chunk rows (default: 32).
  --random-read-count N Random one-row reads during each write build (default: 32).
  --random-seed N      Deterministic read/order seed (default: 20260710).
  --dry-run            Create the self-contained run directory but do not submit.
  -h, --help           Show this help.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --input-root) INPUT_ROOT="$2"; shift 2;;
    --run-id) RUN_ID="$2"; shift 2;;
    --log-root) LOG_ROOT="$2"; shift 2;;
    --output-base) OUTPUT_BASE="$2"; shift 2;;
    --queue) QUEUE="$2"; shift 2;;
    --submit-host) SUBMIT_HOST="$2"; shift 2;;
    --ncores) NCORES="$2"; shift 2;;
    --mem-gb) MEM_GB="$2"; shift 2;;
    --walltime) WALLTIME="$2"; shift 2;;
    --read-repeats) READ_REPEATS="$2"; shift 2;;
    --batch-rows) BATCH_ROWS="$2"; shift 2;;
    --sample-rows) SAMPLE_ROWS="$2"; shift 2;;
    --inner-chunk-rows) INNER_CHUNK_ROWS="$2"; shift 2;;
    --random-read-count) RANDOM_READ_COUNT="$2"; shift 2;;
    --random-seed) RANDOM_SEED="$2"; shift 2;;
    --dry-run) DRY_RUN=1; shift;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown argument: $1" >&2; usage; exit 2;;
  esac
done

if [[ -z "$INPUT_ROOT" ]]; then
  echo "--input-root is required." >&2
  usage
  exit 2
fi
if [[ ! -f "${INPUT_ROOT%/}/benchmark_set.json" || ! -d "${INPUT_ROOT%/}/regular.zarr" ]]; then
  echo "Input root is not a complete benchmark set: $INPUT_ROOT" >&2
  exit 2
fi
for value in "$NCORES" "$MEM_GB" "$READ_REPEATS" "$BATCH_ROWS" "$SAMPLE_ROWS" \
  "$INNER_CHUNK_ROWS" "$RANDOM_READ_COUNT" "$RANDOM_SEED"; do
  if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
    echo "Expected a positive integer, got: $value" >&2
    exit 2
  fi
done

RUN_DIR="${LOG_ROOT%/}/${RUN_ID}"
PRFS_OUTPUT_ROOT="${OUTPUT_BASE%/}/${RUN_ID}/writeback"
if [[ -e "$RUN_DIR" || -e "${OUTPUT_BASE%/}/${RUN_ID}" ]]; then
  echo "Run or output directory already exists for: $RUN_ID" >&2
  exit 2
fi
mkdir -p "$RUN_DIR/source_snapshot" "$RUN_DIR/reports"
cp -R --no-preserve=mode,ownership,timestamps "$REPO_ROOT/src/fisheye" "$RUN_DIR/source_snapshot/"
cp --no-preserve=mode,ownership,timestamps "$REPO_ROOT/scripts/py" "$RUN_DIR/palette_py"
chmod +x "$RUN_DIR/palette_py"
git -C "$REPO_ROOT" status --short > "$RUN_DIR/source_git_status.txt"
git -C "$REPO_ROOT" diff --binary > "$RUN_DIR/source_worktree.patch"
SOURCE_GIT_HEAD="$(git -C "$REPO_ROOT" rev-parse HEAD)"
SOURCE_GIT_BRANCH="$(git -C "$REPO_ROOT" rev-parse --abbrev-ref HEAD)"

cat > "$RUN_DIR/settings.sh" <<SETTINGS
INPUT_ROOT=$(printf '%q' "$INPUT_ROOT")
PRFS_OUTPUT_ROOT=$(printf '%q' "$PRFS_OUTPUT_ROOT")
READ_REPEATS=$READ_REPEATS
BATCH_ROWS=$BATCH_ROWS
SAMPLE_ROWS=$SAMPLE_ROWS
INNER_CHUNK_ROWS=$INNER_CHUNK_ROWS
RANDOM_READ_COUNT=$RANDOM_READ_COUNT
RANDOM_SEED=$RANDOM_SEED
SETTINGS

cat > "$RUN_DIR/manifest.json" <<JSON
{
  "schema_id": "palette.subject_mask_probability_sharding_lsf_submission.v1",
  "run_id": "$RUN_ID",
  "input_root": "$INPUT_ROOT",
  "prfs_output_root": "$PRFS_OUTPUT_ROOT",
  "read_repeats": $READ_REPEATS,
  "batch_rows": $BATCH_ROWS,
  "sample_rows": $SAMPLE_ROWS,
  "inner_chunk_rows": $INNER_CHUNK_ROWS,
  "random_read_count": $RANDOM_READ_COUNT,
  "random_seed": $RANDOM_SEED,
  "source_git_head": "$SOURCE_GIT_HEAD",
  "source_git_branch": "$SOURCE_GIT_BRANCH",
  "source_snapshot": "source_snapshot/fisheye",
  "execution_design": "direct_prfs_reads_then_local_scratch_to_prfs_writes"
}
JSON

JOB_SCRIPT="$RUN_DIR/run_benchmark.sh"
cat > "$JOB_SCRIPT" <<'JOBSCRIPT'
#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="$1"
source "$RUN_DIR/settings.sh"
PALETTE_PY="$RUN_DIR/palette_py"
export PYTHONPATH="$RUN_DIR/source_snapshot${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export TBB_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

USER_NAME="${USER:-unknown}"
JOB_ID="${LSB_JOBID:-manual}"
if [[ -d "/scratch/${USER_NAME}" && -w "/scratch/${USER_NAME}" && -x "/scratch/${USER_NAME}" ]]; then
  SCRATCH_ROOT="/scratch/${USER_NAME}/${JOB_ID}/probability_sharding_benchmark"
else
  SCRATCH_ROOT="${TMPDIR:-/tmp}/palette_probability_sharding_benchmark_${JOB_ID}"
fi
STAGED_SOURCE="$SCRATCH_ROOT/source_regular.zarr"

cleanup() {
  local status=$?
  trap - EXIT INT TERM
  if [[ -d "$SCRATCH_ROOT" ]]; then
    rm -rf "$SCRATCH_ROOT"
  fi
  exit "$status"
}
trap cleanup EXIT INT TERM
mkdir -p "$SCRATCH_ROOT" "$RUN_DIR/reports"

echo "host=$(hostname)"
echo "job_id=${LSB_JOBID:-}"
echo "allocated_slots=${LSB_DJOB_NUMPROC:-1}"
echo "scratch_root=$SCRATCH_ROOT"
echo "input_root=$INPUT_ROOT"
echo "prfs_output_root=$PRFS_OUTPUT_ROOT"

"$PALETTE_PY" -m fisheye.diagnostics.benchmark_subject_mask_probability_sharding_reads \
  "$INPUT_ROOT" \
  --repeats "$READ_REPEATS" \
  --batch-rows "$BATCH_ROWS" \
  --component 0 \
  --random-seed "$RANDOM_SEED" \
  --require-storage-tier prfs \
  --output-json "$RUN_DIR/reports/compute_direct_prfs_reads.json"

/usr/bin/time -f $'elapsed_seconds=%e\nmaximum_rss_kib=%M\nfilesystem_inputs=%I\nfilesystem_outputs=%O' \
  -o "$RUN_DIR/reports/prfs_to_scratch_stage.time.txt" \
  cp -R --no-preserve=mode,ownership,timestamps "$INPUT_ROOT/regular.zarr" "$STAGED_SOURCE"

common_args=(
  "$STAGED_SOURCE"
  --output-root "$PRFS_OUTPUT_ROOT"
  --sample-rows "$SAMPLE_ROWS"
  --inner-chunk-rows "$INNER_CHUNK_ROWS"
  --batch-rows "$BATCH_ROWS"
  --random-read-count "$RANDOM_READ_COUNT"
  --random-seed "$RANDOM_SEED"
  --require-source-storage-tier local
  --require-destination-storage-tier prfs
)

"$PALETTE_PY" -m fisheye.diagnostics.benchmark_subject_mask_probability_sharding \
  "${common_args[@]}" --layout regular
for shard_rows in 512 2048 4096 8192; do
  "$PALETTE_PY" -m fisheye.diagnostics.benchmark_subject_mask_probability_sharding \
    "${common_args[@]}" --layout sharded --shard-rows "$shard_rows"
done

"$PALETTE_PY" - "$INPUT_ROOT/regular.summary.json" \
  "$PRFS_OUTPUT_ROOT/benchmark_set.json" "$RUN_DIR/reports/writeback_validation.json" <<'PY'
import json
import sys
from pathlib import Path

reference = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
result = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
expected = str(reference["destination_sha256"])
variants = list(result["variants"])
errors = []
for variant in variants:
    if str(variant.get("source_sha256")) != expected:
        errors.append(f"{variant.get('variant')}: staged source digest mismatch")
    if str(variant.get("destination_sha256")) != expected:
        errors.append(f"{variant.get('variant')}: destination digest mismatch")
    if (variant.get("source_filesystem") or {}).get("storage_tier") != "local":
        errors.append(f"{variant.get('variant')}: source was not local")
    if (variant.get("destination_filesystem") or {}).get("storage_tier") != "prfs":
        errors.append(f"{variant.get('variant')}: destination was not PRFS")
summary = {
    "schema_id": "palette.subject_mask_probability_sharding_lsf_validation.v1",
    "expected_sha256": expected,
    "variant_count": len(variants),
    "all_exact": not errors and len(variants) == 5,
    "errors": errors,
}
Path(sys.argv[3]).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(summary, sort_keys=True))
if not summary["all_exact"]:
    raise SystemExit(1)
PY
JOBSCRIPT
chmod +x "$JOB_SCRIPT"

BSUB_ARGS=(
  -J "sm_prob_shard_bench"
  -n "$NCORES"
  -W "$WALLTIME"
  -R "rusage[mem=${MEM_GB}G]"
  -oo "$RUN_DIR/%J.out"
  -eo "$RUN_DIR/%J.err"
)
if [[ -n "$QUEUE" ]]; then
  BSUB_ARGS+=(-q "$QUEUE")
fi

printf 'Run dir: %s\n' "$RUN_DIR"
printf 'PRFS write-back root: %s\n' "$PRFS_OUTPUT_ROOT"
printf 'Command: bsub'
for arg in "${BSUB_ARGS[@]}"; do
  printf ' %q' "$arg"
done
printf ' bash %q %q\n' "$JOB_SCRIPT" "$RUN_DIR"
if [[ -n "$SUBMIT_HOST" ]]; then
  printf 'Submission host: %s\n' "$SUBMIT_HOST"
fi

if [[ "$DRY_RUN" == "1" ]]; then
  exit 0
fi
if command -v bsub >/dev/null 2>&1; then
  bsub "${BSUB_ARGS[@]}" bash "$JOB_SCRIPT" "$RUN_DIR"
elif [[ -n "$SUBMIT_HOST" ]]; then
  ssh "$SUBMIT_HOST" bsub "${BSUB_ARGS[@]}" bash "$JOB_SCRIPT" "$RUN_DIR"
else
  echo "bsub is unavailable locally; rerun with --submit-host HOST or submit on an LSF login node." >&2
  exit 127
fi
