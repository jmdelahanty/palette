#!/usr/bin/env bash
set -euo pipefail

ZARR_PATH=""
SUBJECT_RUN=""
ASSIGNMENT_KEYPOINT_GROUP="refined_keypoints_runs"
ASSIGNMENT_KEYPOINT_RUN=""
RUN_ID="subject_mask_finalizer_benchmark_$(date +%Y%m%d_%H%M%S)"
LOG_ROOT="/groups/johnson/johnsonlab/jeremy/recordings/logs/subject_mask_finalizer_benchmarks"
QUEUE=""
DEPENDENCY=""
NCORES=8
MEM_GB=48
WALLTIME="2:00"
MAX_ACTIVE=1
CHUNK_SIZE=256
DENSE_MASK_ROW_CHUNK=256
METRIC_LEVEL="cheap"
MASK_STORAGE="dense_uint8"
POSTCOMPUTE_BACKEND="process_shards"
POSTCOMPUTE_CHUNK_SIZE=256
POSTCOMPUTE_NUM_WORKERS=""
WRITE_EYE_GEOMETRY=1
WRITE_COMPONENT_CONTOURS=1
DRY_RUN=0
VARIANTS=()
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

usage() {
  cat <<'USAGE'
Usage: submit_subject_mask_finalizer_benchmark_bsub.sh --zarr PATH --subject-run RUN --assignment-keypoints-run RUN [options]

Submits local-only CPU finalizer benchmark variants. Each array task creates a
node-local staged zarr overlay, extracts subject_mask_runs/<RUN> from the
run's NRS handoff tar package when available, runs finalize_subject_masks with a
unique benchmark refined run name, writes JSON/Markdown logs, and deletes
scratch via an EXIT trap. Nothing is published to the canonical recording zarr.

Options:
  --zarr PATH                      Canonical analysis zarr path.
  --subject-run RUN                Existing subject_mask_runs/<RUN> to finalize.
  --assignment-keypoint-group NAME refined_keypoints_runs|keypoints_runs (default: refined_keypoints_runs)
  --assignment-keypoints-run RUN   Keypoint run used for eye assignment.
  --run-id ID                      Benchmark run id/log directory name.
  --log-root PATH                  Benchmark log root.
  --queue NAME                     LSF queue name (default: cluster default).
  --dependency EXPR                LSF dependency expression, e.g. 'ended(12345)'.
  --ncores N                       Cores/slots per variant job (default: 8).
  --mem-gb N                       Memory request in GB (default: 48; effective memory follows cluster slot model).
  --walltime H:MM                  LSF walltime (default: 2:00).
  --max-active N                   Max concurrent benchmark variants (default: 1).
  --chunk-size N                   Finalizer row chunk size (default: 256).
  --dense-mask-row-chunk N         Physical dense masks_roi row chunk (default: 256).
  --metric-level LEVEL             cheap|full (default: cheap).
  --mask-storage MODE              dense_uint8|dense_and_bitpacked|dense_and_rle|dense_bitpacked_and_rle
                                   (default: dense_uint8).
  --postcompute-backend MODE       serial|process_shards (default: process_shards).
  --postcompute-chunk-size N       Rows per postcompute shard (default: 256).
  --postcompute-num-workers N      Postcompute workers (default: variant worker count).
  --no-write-eye-geometry          Disable eye geometry output.
  --no-write-component-contours    Disable component contour output.
  --variant SPEC                   Add process_shards variant workers[:chunk_size].
                                   Repeatable. Default: 8 workers with --chunk-size.
  --dry-run                        Write scripts/manifests and print bsub command; do not submit.
  -h, --help                       Show this message.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --zarr) ZARR_PATH="$2"; shift 2;;
    --subject-run) SUBJECT_RUN="$2"; shift 2;;
    --assignment-keypoint-group) ASSIGNMENT_KEYPOINT_GROUP="$2"; shift 2;;
    --assignment-keypoints-run) ASSIGNMENT_KEYPOINT_RUN="$2"; shift 2;;
    --run-id) RUN_ID="$2"; shift 2;;
    --log-root) LOG_ROOT="$2"; shift 2;;
    --queue) QUEUE="$2"; shift 2;;
    --dependency) DEPENDENCY="$2"; shift 2;;
    --ncores) NCORES="$2"; shift 2;;
    --mem-gb) MEM_GB="$2"; shift 2;;
    --walltime) WALLTIME="$2"; shift 2;;
    --max-active) MAX_ACTIVE="$2"; shift 2;;
    --chunk-size) CHUNK_SIZE="$2"; shift 2;;
    --dense-mask-row-chunk) DENSE_MASK_ROW_CHUNK="$2"; shift 2;;
    --metric-level) METRIC_LEVEL="$2"; shift 2;;
    --mask-storage) MASK_STORAGE="$2"; shift 2;;
    --postcompute-backend) POSTCOMPUTE_BACKEND="$2"; shift 2;;
    --postcompute-chunk-size) POSTCOMPUTE_CHUNK_SIZE="$2"; shift 2;;
    --postcompute-num-workers) POSTCOMPUTE_NUM_WORKERS="$2"; shift 2;;
    --no-write-eye-geometry) WRITE_EYE_GEOMETRY=0; shift;;
    --no-write-component-contours) WRITE_COMPONENT_CONTOURS=0; shift;;
    --variant) VARIANTS+=("$2"); shift 2;;
    --dry-run) DRY_RUN=1; shift;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown arg: $1" >&2; usage; exit 2;;
  esac
done

if [[ -z "$ZARR_PATH" || -z "$SUBJECT_RUN" || -z "$ASSIGNMENT_KEYPOINT_RUN" ]]; then
  echo "--zarr, --subject-run, and --assignment-keypoints-run are required." >&2
  usage
  exit 2
fi
if [[ "$ASSIGNMENT_KEYPOINT_GROUP" != "refined_keypoints_runs" && "$ASSIGNMENT_KEYPOINT_GROUP" != "keypoints_runs" ]]; then
  echo "--assignment-keypoint-group must be refined_keypoints_runs or keypoints_runs." >&2
  exit 2
fi
if [[ "$METRIC_LEVEL" != "cheap" && "$METRIC_LEVEL" != "full" ]]; then
  echo "--metric-level must be cheap or full." >&2
  exit 2
fi
if [[ "$MASK_STORAGE" != "dense_uint8" && "$MASK_STORAGE" != "dense_and_bitpacked" && "$MASK_STORAGE" != "dense_and_rle" && "$MASK_STORAGE" != "dense_bitpacked_and_rle" ]]; then
  echo "--mask-storage must be dense_uint8, dense_and_bitpacked, dense_and_rle, or dense_bitpacked_and_rle." >&2
  exit 2
fi
if [[ "$POSTCOMPUTE_BACKEND" != "serial" && "$POSTCOMPUTE_BACKEND" != "process_shards" ]]; then
  echo "--postcompute-backend must be serial or process_shards." >&2
  exit 2
fi

if [[ "${#VARIANTS[@]}" -eq 0 ]]; then
  VARIANTS=("8")
fi

RUN_DIR="${LOG_ROOT%/}/${RUN_ID}"
if [[ -e "$RUN_DIR" ]]; then
  echo "Benchmark run dir already exists: $RUN_DIR" >&2
  exit 2
fi
mkdir -p "$RUN_DIR"
mkdir -p "$RUN_DIR/source_snapshot"
cp -a "$REPO_ROOT/src/fisheye" "$RUN_DIR/source_snapshot/"
cp "$REPO_ROOT/scripts/py" "$RUN_DIR/palette_py"
chmod +x "$RUN_DIR/palette_py"
git -C "$REPO_ROOT" status --short > "$RUN_DIR/source_git_status.txt"
git -C "$REPO_ROOT" diff --binary > "$RUN_DIR/source_worktree.patch"
SOURCE_GIT_HEAD="$(git -C "$REPO_ROOT" rev-parse HEAD)"
SOURCE_GIT_BRANCH="$(git -C "$REPO_ROOT" rev-parse --abbrev-ref HEAD)"
SOURCE_DIRTY_FILE_COUNT="$(wc -l < "$RUN_DIR/source_git_status.txt")"

VARIANTS_TSV="$RUN_DIR/variants.tsv"
: > "$VARIANTS_TSV"
idx=0
for spec in "${VARIANTS[@]}"; do
  IFS=: read -r workers variant_chunk <<<"$spec"
  variant_chunk="${variant_chunk:-$CHUNK_SIZE}"
  if [[ ! "$workers" =~ ^[1-9][0-9]*$ || ! "$variant_chunk" =~ ^[1-9][0-9]*$ ]]; then
    echo "Invalid process_shards variant $spec; expected workers[:chunk_size]." >&2
    exit 2
  fi
  idx=$((idx + 1))
  variant_id="$(printf 'v%02d_process_shards_w%s_c%s' "$idx" "$workers" "$variant_chunk")"
  printf '%s\t%s\t%s\n' "$variant_id" "$workers" "$variant_chunk" >> "$VARIANTS_TSV"
done

cat > "$RUN_DIR/manifest.json" <<JSON
{
  "schema": "palette_subject_mask_finalizer_benchmark_submission_v2",
  "run_id": "$RUN_ID",
  "zarr_path": "$ZARR_PATH",
  "subject_run": "$SUBJECT_RUN",
  "assignment_keypoint_group": "$ASSIGNMENT_KEYPOINT_GROUP",
  "assignment_keypoints_run": "$ASSIGNMENT_KEYPOINT_RUN",
  "metric_level": "$METRIC_LEVEL",
  "dense_mask_row_chunk": $DENSE_MASK_ROW_CHUNK,
  "mask_storage": "$MASK_STORAGE",
  "postcompute_backend": "$POSTCOMPUTE_BACKEND",
  "postcompute_chunk_size": $POSTCOMPUTE_CHUNK_SIZE,
  "postcompute_num_workers": ${POSTCOMPUTE_NUM_WORKERS:-null},
  "write_eye_geometry": $([[ "$WRITE_EYE_GEOMETRY" == "1" ]] && echo true || echo false),
  "write_component_contours": $([[ "$WRITE_COMPONENT_CONTOURS" == "1" ]] && echo true || echo false),
  "source_git_head": "$SOURCE_GIT_HEAD",
  "source_git_branch": "$SOURCE_GIT_BRANCH",
  "source_dirty_file_count": $SOURCE_DIRTY_FILE_COUNT,
  "source_snapshot": "source_snapshot/fisheye",
  "source_worktree_patch": "source_worktree.patch",
  "execution_backend": "process_shards",
  "variant_count": $idx
}
JSON

JOB_SCRIPT="$RUN_DIR/run_variant.sh"
cat > "$JOB_SCRIPT" <<'JOBSCRIPT'
#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="$1"
VARIANT_LINE="$(sed -n "${LSB_JOBINDEX:-1}p" "${RUN_DIR}/variants.tsv")"
if [[ -z "$VARIANT_LINE" ]]; then
  echo "No variant row for LSB_JOBINDEX=${LSB_JOBINDEX:-1}" >&2
  exit 2
fi
IFS=$'\t' read -r VARIANT_ID WORKERS CHUNK_SIZE <<<"$VARIANT_LINE"

source "${RUN_DIR}/settings.sh"
PALETTE_PY="${RUN_DIR}/palette_py"
export PYTHONPATH="${RUN_DIR}/source_snapshot${PYTHONPATH:+:${PYTHONPATH}}"

THREADS_PER_PROCESS=1
export OMP_NUM_THREADS="$THREADS_PER_PROCESS"
export MKL_NUM_THREADS="$THREADS_PER_PROCESS"
export OPENBLAS_NUM_THREADS="$THREADS_PER_PROCESS"
export TBB_NUM_THREADS="$THREADS_PER_PROCESS"
export OPENMP_NUM_THREADS="$THREADS_PER_PROCESS"
export NUM_MKL_THREADS="$THREADS_PER_PROCESS"
export NUMEXPR_NUM_THREADS="$THREADS_PER_PROCESS"

USER_NAME="${USER:-unknown}"
JOB_ID="${LSB_JOBID:-manual}"
JOB_INDEX="${LSB_JOBINDEX:-0}"
if [[ -d "/scratch/${USER_NAME}" && -w "/scratch/${USER_NAME}" && -x "/scratch/${USER_NAME}" ]]; then
  SCRATCH_ROOT="/scratch/${USER_NAME}/${JOB_ID}/finalizer_benchmark_${JOB_INDEX}"
else
  SCRATCH_ROOT="${TMPDIR:-/tmp}/palette_finalizer_benchmark_${JOB_ID}_${JOB_INDEX}"
fi

cleanup() {
  local status=$?
  trap - EXIT INT TERM
  if [[ -n "${SCRATCH_ROOT:-}" && -d "$SCRATCH_ROOT" ]]; then
    echo "Cleaning benchmark scratch: $SCRATCH_ROOT"
    rm -rf "$SCRATCH_ROOT"
  fi
  exit "$status"
}
trap cleanup EXIT INT TERM

mkdir -p "$SCRATCH_ROOT" "$RUN_DIR/reports"

echo "host=$(hostname)"
echo "job_id=${LSB_JOBID:-}"
echo "job_index=${LSB_JOBINDEX:-}"
echo "variant_id=$VARIANT_ID"
echo "backend=process_shards"
echo "workers=$WORKERS"
echo "chunk_size=$CHUNK_SIZE"
echo "dense_mask_row_chunk=$DENSE_MASK_ROW_CHUNK"
echo "mask_storage=$MASK_STORAGE"
echo "postcompute_backend=$POSTCOMPUTE_BACKEND"
echo "scratch_root=$SCRATCH_ROOT"

STAGED_ZARR="$(
  "$PALETTE_PY" - "$ZARR_PATH" "$SUBJECT_RUN" "$VARIANT_ID" "$SCRATCH_ROOT" <<'PY'
import sys
from pathlib import Path

from fisheye.utils.run_subject_mask_batch_pipeline import (
    ArchivePlan,
    _prepare_output_staging_zarr,
)

zarr_path = Path(sys.argv[1]).expanduser().resolve()
subject_run = sys.argv[2]
variant_id = sys.argv[3]
scratch_root = Path(sys.argv[4]).expanduser().resolve()
refined_run = f"refined_subject_masks_benchmark_{variant_id}"

ctx = _prepare_output_staging_zarr(
    zarr_path,
    plan=ArchivePlan(
        zarr_path=str(zarr_path),
        subject_run=subject_run,
        refined_run=refined_run,
        crop_run=None,
        assignment_keypoint_group=None,
        assignment_keypoint_run=None,
        has_subject_runs=True,
        has_refined_subject_runs=False,
        run_inference=False,
        run_finalization=True,
    ),
    staging_root=scratch_root / "staged_zarrs",
    overwrite=True,
    stage_finalization_input=True,
)
print(ctx.staged_zarr_path)
PY
)"

REFINED_RUN="refined_subject_masks_benchmark_${VARIANT_ID}"
JSON_OUT="${RUN_DIR}/reports/${VARIANT_ID}.json"
STDOUT_LOG="${RUN_DIR}/reports/${VARIANT_ID}.stdout"
PROGRESS_LOG="${RUN_DIR}/reports/${VARIANT_ID}.progress.jsonl"
RESOURCE_SUMMARY="${RUN_DIR}/reports/${VARIANT_ID}.resources.json"
RESOURCE_SAMPLES="${RUN_DIR}/reports/${VARIANT_ID}.resources.jsonl"

cmd=("$PALETTE_PY" -m fisheye.refinement.finalize_subject_masks "$STAGED_ZARR"
  --subject-run "$SUBJECT_RUN"
  --run-name "$REFINED_RUN"
  --components subject_body eyes_union swim_bladder
  --chunk-size "$CHUNK_SIZE"
  --dense-mask-row-chunk "$DENSE_MASK_ROW_CHUNK"
  --metric-level "$METRIC_LEVEL"
  --mask-storage "$MASK_STORAGE"
  --postcompute-backend "$POSTCOMPUTE_BACKEND"
  --postcompute-chunk-size "$POSTCOMPUTE_CHUNK_SIZE"
  --postcompute-num-workers "${POSTCOMPUTE_NUM_WORKERS:-$WORKERS}"
  --execution-backend process_shards
  --assignment-keypoint-group "$ASSIGNMENT_KEYPOINT_GROUP"
  --assignment-keypoints-run "$ASSIGNMENT_KEYPOINT_RUN"
  --json
  --progress-jsonl "$PROGRESS_LOG"
  --num-workers "$WORKERS"
  --defer-registry-status
  --overwrite)
if [[ "$WRITE_EYE_GEOMETRY" == "1" ]]; then
  cmd+=(--write-eye-geometry)
fi
if [[ "$WRITE_COMPONENT_CONTOURS" == "1" ]]; then
  cmd+=(--write-component-contours)
fi

printf '+ %q ' "${cmd[@]}"
printf '\n'
"$PALETTE_PY" -m fisheye.diagnostics.run_with_resource_telemetry \
  --summary-json "$RESOURCE_SUMMARY" \
  --samples-jsonl "$RESOURCE_SAMPLES" \
  --stdout-log "$STDOUT_LOG" \
  --requested-workers "$WORKERS" \
  --allocated-slots "${LSB_DJOB_NUMPROC:-$WORKERS}" \
  --sample-interval-seconds 2 \
  -- "${cmd[@]}"

"$PALETTE_PY" - "$STAGED_ZARR" "$REFINED_RUN" "$JSON_OUT" "$RESOURCE_SUMMARY" <<'PY'
import json
import sys
from pathlib import Path

import zarr

from fisheye.utils.run_subject_mask_batch_pipeline import _run_group_storage_stats

zarr_path = Path(sys.argv[1])
run_name = sys.argv[2]
out_path = Path(sys.argv[3])
resource_summary_path = Path(sys.argv[4])
root = zarr.open_group(str(zarr_path), mode="r", use_consolidated=False)
run = root[f"refined_subject_masks_runs/{run_name}"]
attrs = dict(run.attrs)
run_group_path = zarr_path / "refined_subject_masks_runs" / run_name
resource_telemetry = json.loads(resource_summary_path.read_text(encoding="utf-8"))
payload = {
    "refined_run": run_name,
    "staged_zarr_path": str(zarr_path),
    "summary_statistics": attrs.get("summary_statistics"),
    "mask_storage": attrs.get("smart_finalizer_mask_storage"),
    "mask_storage_encoding": attrs.get("mask_storage_encoding"),
    "mask_store_encodings": attrs.get("mask_store_encodings"),
    "masks_roi_materialized": attrs.get("masks_roi_materialized"),
    "smart_finalizer_execution_backend": attrs.get("smart_finalizer_execution_backend"),
    "smart_finalizer_timing_summary": attrs.get("smart_finalizer_timing_summary"),
    "worker_process_count": attrs.get("worker_process_count"),
    "worker_chunk_size": attrs.get("worker_chunk_size"),
    "resource_telemetry": resource_telemetry,
    "refined_output_storage_stats": _run_group_storage_stats(run_group_path),
}
out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(payload, sort_keys=True))
PY
JOBSCRIPT
chmod +x "$JOB_SCRIPT"

cat > "$RUN_DIR/settings.sh" <<SETTINGS
ZARR_PATH=$(printf '%q' "$ZARR_PATH")
SUBJECT_RUN=$(printf '%q' "$SUBJECT_RUN")
ASSIGNMENT_KEYPOINT_GROUP=$(printf '%q' "$ASSIGNMENT_KEYPOINT_GROUP")
ASSIGNMENT_KEYPOINT_RUN=$(printf '%q' "$ASSIGNMENT_KEYPOINT_RUN")
METRIC_LEVEL=$(printf '%q' "$METRIC_LEVEL")
DENSE_MASK_ROW_CHUNK=$DENSE_MASK_ROW_CHUNK
MASK_STORAGE=$(printf '%q' "$MASK_STORAGE")
POSTCOMPUTE_BACKEND=$(printf '%q' "$POSTCOMPUTE_BACKEND")
POSTCOMPUTE_CHUNK_SIZE=$POSTCOMPUTE_CHUNK_SIZE
POSTCOMPUTE_NUM_WORKERS=$(printf '%q' "$POSTCOMPUTE_NUM_WORKERS")
WRITE_EYE_GEOMETRY=$WRITE_EYE_GEOMETRY
WRITE_COMPONENT_CONTOURS=$WRITE_COMPONENT_CONTOURS
SETTINGS

BSUB_ARGS=(-J "sm_fin_bench[1-${idx}]%${MAX_ACTIVE}" -n "$NCORES" -W "$WALLTIME" -R "rusage[mem=${MEM_GB}G]" -oo "${RUN_DIR}/%J_%I.out" -eo "${RUN_DIR}/%J_%I.err")
if [[ -n "$QUEUE" ]]; then
  BSUB_ARGS+=(-q "$QUEUE")
fi
if [[ -n "$DEPENDENCY" ]]; then
  BSUB_ARGS+=(-w "$DEPENDENCY")
fi

printf 'Run dir: %s\n' "$RUN_DIR"
printf 'Variants: %s\n' "$idx"
printf 'Max active: %s\n' "$MAX_ACTIVE"
printf 'Zarr: %s\n' "$ZARR_PATH"
printf 'Subject run: %s\n' "$SUBJECT_RUN"
printf 'Command: bsub'
for arg in "${BSUB_ARGS[@]}"; do
  printf ' %q' "$arg"
done
printf ' bash %q %q\n' "$JOB_SCRIPT" "$RUN_DIR"

if [[ "$DRY_RUN" == "1" ]]; then
  exit 0
fi

bsub "${BSUB_ARGS[@]}" bash "$JOB_SCRIPT" "$RUN_DIR"
