#!/usr/bin/env bash
set -euo pipefail

FIXTURE_ROOT=""
RUN_ID="subject_mask_finalizer_layout_ab_$(date +%Y%m%d_%H%M%S)"
LOG_ROOT="/groups/johnson/johnsonlab/jeremy/recordings/logs/subject_mask_finalizer_layout_ab"
QUEUE=""
SUBMIT_HOST="${PALETTE_LSF_SUBMIT_HOST:-}"
NCORES=8
MEM_GB_PER_SLOT=4
WALLTIME="4:00"
WORKERS=8
CHUNK_SIZE=256
DENSE_MASK_ROW_CHUNK=128
REPEATS=2
DRY_RUN=0
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

usage() {
  cat <<'USAGE'
Usage: submit_subject_mask_finalizer_layout_ab_bsub.sh --fixture-root PATH [options]

Runs the complete subject-mask finalizer directly against the regular and
2,048-row-sharded PRFS fixtures. One compute-node job alternates AB/BA order so
each layout occupies each execution position once. Refined outputs are written
only inside the benchmark fixtures, resource telemetry is retained on PRFS, and
all corresponding output arrays are compared exactly after the timed runs.

Options:
  --run-id ID             Run/log directory name.
  --log-root PATH         PRFS log root.
  --queue NAME            LSF queue.
  --submit-host HOST      SSH host used when bsub is unavailable locally.
  --ncores N              LSF slots (default: 8).
  --mem-gb-per-slot N     Client memory request per slot (default: 4).
  --walltime H:MM         LSF walltime (default: 4:00).
  --workers N             Finalizer process workers (default: 8).
  --chunk-size N          Logical worker rows (default: 256).
  --dense-mask-row-chunk N  Refined masks_roi physical rows (default: 128).
  --repeats N             Runs per layout (default: 2; AB then BA).
  --dry-run               Package the job without submitting it.
  -h, --help              Show this help.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --fixture-root) FIXTURE_ROOT="$2"; shift 2;;
    --run-id) RUN_ID="$2"; shift 2;;
    --log-root) LOG_ROOT="$2"; shift 2;;
    --queue) QUEUE="$2"; shift 2;;
    --submit-host) SUBMIT_HOST="$2"; shift 2;;
    --ncores) NCORES="$2"; shift 2;;
    --mem-gb-per-slot) MEM_GB_PER_SLOT="$2"; shift 2;;
    --walltime) WALLTIME="$2"; shift 2;;
    --workers) WORKERS="$2"; shift 2;;
    --chunk-size) CHUNK_SIZE="$2"; shift 2;;
    --dense-mask-row-chunk) DENSE_MASK_ROW_CHUNK="$2"; shift 2;;
    --repeats) REPEATS="$2"; shift 2;;
    --dry-run) DRY_RUN=1; shift;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown argument: $1" >&2; usage; exit 2;;
  esac
done

if [[ -z "$FIXTURE_ROOT" ]]; then
  echo "--fixture-root is required." >&2
  usage
  exit 2
fi
FIXTURE_ROOT="${FIXTURE_ROOT%/}"
REGULAR_ZARR="$FIXTURE_ROOT/regular.zarr"
SHARDED_ZARR="$FIXTURE_ROOT/shard_02048.zarr"
for path in "$REGULAR_ZARR" "$SHARDED_ZARR"; do
  if [[ ! -f "$path/zarr.json" ]]; then
    echo "Fixture Zarr not found: $path" >&2
    exit 2
  fi
done
for value in "$NCORES" "$MEM_GB_PER_SLOT" "$WORKERS" "$CHUNK_SIZE" "$DENSE_MASK_ROW_CHUNK" "$REPEATS"; do
  if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
    echo "Expected a positive integer, got: $value" >&2
    exit 2
  fi
done
if (( CHUNK_SIZE % DENSE_MASK_ROW_CHUNK != 0 )); then
  echo "--chunk-size must be a multiple of --dense-mask-row-chunk for safe parallel Zarr writes." >&2
  exit 2
fi

RUN_DIR="${LOG_ROOT%/}/${RUN_ID}"
if [[ -e "$RUN_DIR" ]]; then
  echo "Run directory already exists: $RUN_DIR" >&2
  exit 2
fi
mkdir -p "$RUN_DIR/source_snapshot" "$RUN_DIR/reports"
SOURCE_GIT_HEAD="$(git -C "$REPO_ROOT" rev-parse HEAD)"
SOURCE_GIT_BRANCH="$(git -C "$REPO_ROOT" branch --show-current)"
git -C "$REPO_ROOT" archive --format=tar "$SOURCE_GIT_HEAD" src/fisheye \
  | tar --extract --directory "$RUN_DIR/source_snapshot" --strip-components=1 \
      --no-same-owner --no-same-permissions
git -C "$REPO_ROOT" bundle create "$RUN_DIR/source_git.bundle" HEAD
cp --no-preserve=mode,ownership,timestamps "$REPO_ROOT/scripts/py" "$RUN_DIR/palette_py"
chmod +x "$RUN_DIR/palette_py"

cat > "$RUN_DIR/settings.sh" <<SETTINGS
FIXTURE_ROOT=$(printf '%q' "$FIXTURE_ROOT")
REGULAR_ZARR=$(printf '%q' "$REGULAR_ZARR")
SHARDED_ZARR=$(printf '%q' "$SHARDED_ZARR")
WORKERS=$WORKERS
CHUNK_SIZE=$CHUNK_SIZE
DENSE_MASK_ROW_CHUNK=$DENSE_MASK_ROW_CHUNK
REPEATS=$REPEATS
SOURCE_GIT_HEAD=$(printf '%q' "$SOURCE_GIT_HEAD")
SETTINGS

cat > "$RUN_DIR/manifest.json" <<JSON
{
  "schema_id": "palette.subject_mask_finalizer_layout_ab_submission.v1",
  "run_id": "$RUN_ID",
  "fixture_root": "$FIXTURE_ROOT",
  "regular_zarr": "$REGULAR_ZARR",
  "sharded_zarr": "$SHARDED_ZARR",
  "workers": $WORKERS,
  "chunk_size": $CHUNK_SIZE,
  "dense_mask_row_chunk": $DENSE_MASK_ROW_CHUNK,
  "repeats_per_layout": $REPEATS,
  "order": "position_balanced_ab_ba",
  "probability_reads": "direct_prfs",
  "refined_writes": "direct_prfs_benchmark_fixture_only",
  "source_git_head": "$SOURCE_GIT_HEAD",
  "source_git_branch": "$SOURCE_GIT_BRANCH",
  "provenance_checkout": "node_local_clean_checkout_from_source_git.bundle",
  "mutates_registry": false
}
JSON

JOB_SCRIPT="$RUN_DIR/run_layout_ab.sh"
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
export OPENMP_NUM_THREADS=1
export NUM_MKL_THREADS=1
export NUMEXPR_NUM_THREADS=1

USER_NAME="${USER:-unknown}"
JOB_ID="${LSB_JOBID:-manual}"
if [[ -d "/scratch/${USER_NAME}" && -w "/scratch/${USER_NAME}" && -x "/scratch/${USER_NAME}" ]]; then
  PROVENANCE_CHECKOUT="/scratch/${USER_NAME}/${JOB_ID}/palette_finalizer_layout_ab_provenance"
else
  PROVENANCE_CHECKOUT="${TMPDIR:-/tmp}/palette_finalizer_layout_ab_provenance_${JOB_ID}"
fi
cleanup() {
  local status=$?
  trap - EXIT INT TERM
  if [[ -d "$PROVENANCE_CHECKOUT" ]]; then
    rm -rf "$PROVENANCE_CHECKOUT"
  fi
  exit "$status"
}
trap cleanup EXIT INT TERM
git clone --quiet "$RUN_DIR/source_git.bundle" "$PROVENANCE_CHECKOUT"
git -C "$PROVENANCE_CHECKOUT" checkout --quiet --detach "$SOURCE_GIT_HEAD"
cd "$PROVENANCE_CHECKOUT"

echo "host=$(hostname)"
echo "job_id=${LSB_JOBID:-}"
echo "allocated_slots=${LSB_DJOB_NUMPROC:-$WORKERS}"
echo "fixture_root=$FIXTURE_ROOT"
echo "provenance_git_head=$(git rev-parse HEAD)"
echo "provenance_git_dirty=$(test -n "$(git status --porcelain)" && echo true || echo false)"

for repeat in $(seq 1 "$REPEATS"); do
  if (( repeat % 2 == 1 )); then
    order=(regular sharded)
  else
    order=(sharded regular)
  fi
  for variant in "${order[@]}"; do
    if [[ "$variant" == "regular" ]]; then
      zarr_path="$REGULAR_ZARR"
    else
      zarr_path="$SHARDED_ZARR"
    fi
    repeat_padded="$(printf '%02d' "$repeat")"
    refined_run="refined_subject_masks_layout_ab_${variant}_r${repeat_padded}"
    prefix="$RUN_DIR/reports/${variant}_r${repeat_padded}"
    cmd=("$PALETTE_PY" -m fisheye.refinement.finalize_subject_masks "$zarr_path"
      --subject-run subject_masks_finalizer_ab_fixture
      --run-name "$refined_run"
      --components subject_body eye_left eye_right swim_bladder
      --chunk-size "$CHUNK_SIZE"
      --dense-mask-row-chunk "$DENSE_MASK_ROW_CHUNK"
      --metric-level cheap
      --mask-storage dense_uint8
      --execution-backend process_shards
      --num-workers "$WORKERS"
      --assignment-keypoint-group refined_keypoints_runs
      --assignment-keypoints-run refined_keypoints_finalizer_ab_fixture
      --progress-jsonl "${prefix}.progress.jsonl"
      --defer-registry-status
      --overwrite
      --json)

    echo "starting variant=$variant repeat=$repeat zarr=$zarr_path"
    "$PALETTE_PY" -m fisheye.diagnostics.run_with_resource_telemetry \
      --summary-json "${prefix}.resources.json" \
      --samples-jsonl "${prefix}.resources.jsonl" \
      --stdout-log "${prefix}.stdout" \
      --requested-workers "$WORKERS" \
      --allocated-slots "${LSB_DJOB_NUMPROC:-$WORKERS}" \
      --sample-interval-seconds 2 \
      -- "${cmd[@]}"

    "$PALETTE_PY" - "$zarr_path" "$refined_run" "${prefix}.json" "${prefix}.resources.json" <<'PY'
import json
import sys
from pathlib import Path

import zarr

root = zarr.open_group(sys.argv[1], mode="r", use_consolidated=False)
run = root[f"refined_subject_masks_runs/{sys.argv[2]}"]
resources = json.loads(Path(sys.argv[4]).read_text(encoding="utf-8"))
payload = {
    "zarr_path": sys.argv[1],
    "refined_run": sys.argv[2],
    "summary_statistics": run.attrs.get("summary_statistics"),
    "timing_summary": run.attrs.get("smart_finalizer_timing_summary"),
    "worker_process_count": run.attrs.get("worker_process_count"),
    "worker_chunk_size": run.attrs.get("worker_chunk_size"),
    "resource_telemetry": resources,
}
Path(sys.argv[3]).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
  done
done

echo "starting untimed exact output parity validation"
"$PALETTE_PY" - "$REGULAR_ZARR" "$SHARDED_ZARR" "$RUN_DIR/reports" "$REPEATS" <<'PY'
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import zarr


def arrays(group, prefix=""):
    result = {}
    for name, member in group.members():
        path = f"{prefix}/{name}" if prefix else name
        if isinstance(member, zarr.Array):
            result[path] = member
        elif isinstance(member, zarr.Group):
            result.update(arrays(member, path))
    return result


def equal(left, right):
    if left.dtype.kind in "fc":
        return bool(np.array_equal(left, right, equal_nan=True))
    return bool(np.array_equal(left, right))


regular_root = zarr.open_group(sys.argv[1], mode="r", use_consolidated=False)
sharded_root = zarr.open_group(sys.argv[2], mode="r", use_consolidated=False)
reports = Path(sys.argv[3])
repeats = int(sys.argv[4])
parity_started = time.perf_counter()
comparisons = []
all_mismatches = []
for repeat in range(1, repeats + 1):
    suffix = f"r{repeat:02d}"
    regular = arrays(
        regular_root[f"refined_subject_masks_runs/refined_subject_masks_layout_ab_regular_{suffix}"]
    )
    sharded = arrays(
        sharded_root[f"refined_subject_masks_runs/refined_subject_masks_layout_ab_sharded_{suffix}"]
    )
    mismatches = []
    if regular.keys() != sharded.keys():
        mismatches.append("array path sets differ")
    for path in sorted(regular.keys() & sharded.keys()):
        left = regular[path]
        right = sharded[path]
        if left.shape != right.shape or left.dtype != right.dtype:
            mismatches.append(f"{path}: shape/dtype differ")
            continue
        if left.ndim == 0:
            if not equal(np.asarray(left[()]), np.asarray(right[()])):
                mismatches.append(f"{path}: scalar differs")
            continue
        # Compare at up to 64 rows per read. This remains comfortably bounded
        # for dense four-component masks while avoiding repeated decoding of a
        # 128-row physical output chunk in four separate 32-row reads.
        row_step = min(64, max(1, int(left.chunks[0] or 1)))
        for start in range(0, int(left.shape[0]), row_step):
            stop = min(int(left.shape[0]), start + row_step)
            if not equal(np.asarray(left[start:stop]), np.asarray(right[start:stop])):
                mismatches.append(f"{path}: rows {start}:{stop} differ")
                break
    comparisons.append(
        {
            "repeat": repeat,
            "regular_array_count": len(regular),
            "sharded_array_count": len(sharded),
            "mismatch_count": len(mismatches),
            "mismatches": mismatches,
        }
    )
    all_mismatches.extend(f"repeat {repeat}: {item}" for item in mismatches)

summary = {
    "schema_id": "palette.subject_mask_finalizer_layout_ab_summary.v1",
    "parity": {
        "comparison_count": len(comparisons),
        "comparisons": comparisons,
        "mismatch_count": len(all_mismatches),
        "mismatches": all_mismatches,
        "duration_seconds": time.perf_counter() - parity_started,
    },
}
for variant in ("regular", "sharded"):
    payloads = [
        json.loads((reports / f"{variant}_r{repeat:02d}.json").read_text(encoding="utf-8"))
        for repeat in range(1, repeats + 1)
    ]
    durations = [float(item["summary_statistics"]["duration_seconds"]) for item in payloads]
    walls = [float(item["resource_telemetry"]["duration_seconds"]) for item in payloads]
    cpu_seconds = [float(item["resource_telemetry"]["cpu_seconds"]["total"]) for item in payloads]
    peak_rss = [int(item["resource_telemetry"]["peak_process_tree_rss_bytes"]) for item in payloads]
    summary[variant] = {
        "repeat_count": len(payloads),
        "duration_seconds": durations,
        "duration_median_seconds": statistics.median(durations),
        "resource_wall_seconds": walls,
        "resource_wall_median_seconds": statistics.median(walls),
        "cpu_seconds": cpu_seconds,
        "cpu_median_seconds": statistics.median(cpu_seconds),
        "peak_process_tree_rss_bytes": peak_rss,
        "peak_process_tree_rss_median_bytes": statistics.median(peak_rss),
    }
summary_path = reports / "summary.json"
summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(summary, sort_keys=True))
if all_mismatches:
    raise SystemExit(1)
PY
JOBSCRIPT
chmod +x "$JOB_SCRIPT"

BSUB_ARGS=(-J "sm_finalizer_layout_ab" -n "$NCORES" -W "$WALLTIME" -R "rusage[mem=${MEM_GB_PER_SLOT}G]" -oo "$RUN_DIR/%J.out" -eo "$RUN_DIR/%J.err")
if [[ -n "$QUEUE" ]]; then BSUB_ARGS+=(-q "$QUEUE"); fi
printf 'Run dir: %s\n' "$RUN_DIR"
printf 'Fixture root: %s\n' "$FIXTURE_ROOT"
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
