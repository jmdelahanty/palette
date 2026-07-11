#!/usr/bin/env bash
set -euo pipefail

ZARR_PATH=""
SUBJECT_RUN=""
ASSIGNMENT_KEYPOINT_GROUP="refined_keypoints_runs"
ASSIGNMENT_KEYPOINT_RUN=""
BASELINE_SOURCE=""
RUN_ID="subject_mask_finalizer_kernel_ab_$(date +%Y%m%d_%H%M%S)"
LOG_ROOT="/groups/johnson/johnsonlab/jeremy/recordings/logs/subject_mask_finalizer_benchmarks"
QUEUE=""
NCORES=8
MEM_GB=48
WALLTIME="1:00"
WORKERS=8
START_ROW=0
ROI_COUNT=4096
CHUNK_SIZE=256
DENSE_MASK_ROW_CHUNK=256
REPEATS=4
DRY_RUN=0
PROVENANCE_REPO="${PALETTE_GROUPS_REPO:-/groups/johnson/johnsonlab/jeremy/gitrepos/palette}"

usage() {
  cat <<'USAGE'
Usage: submit_subject_mask_finalizer_kernel_ab_bsub.sh --zarr PATH --subject-run RUN \
  --assignment-keypoints-run RUN --baseline-source PATH [options]

Submits one isolated CPU job that copies a real row window once, then runs the
saved baseline and current candidate source snapshots sequentially in alternating
AB/BA order. Each run records in-job process-tree telemetry. The job finishes by
comparing all baseline/candidate output arrays exactly. Nothing is published.

--baseline-source PATH should contain a complete `fisheye/` package directory.

Options:
  --assignment-keypoint-group NAME  refined_keypoints_runs|keypoints_runs.
  --run-id ID                       Benchmark run/log directory name.
  --log-root PATH                   Benchmark log root.
  --queue NAME                      LSF queue (default: cluster default).
  --ncores N                        Allocated slots (default: 8).
  --mem-gb N                        Memory request in GB (default: 48).
  --walltime H:MM                   LSF wall time (default: 1:00).
  --workers N                       process_shards workers (default: 8).
  --start-row N                     First source row (default: 0).
  --roi-count N                     Contiguous real rows (default: 4096).
  --chunk-size N                    Worker row chunk (default: 256).
  --dense-mask-row-chunk N          Physical dense row chunk (default: 256).
  --repeats N                       Alternating runs per source (default: 4; position-balanced).
  --provenance-repo PATH            Cluster-visible Palette Git checkout used for provenance.
  --dry-run                         Generate the self-contained run directory only.
  -h, --help                        Show this message.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --zarr) ZARR_PATH="$2"; shift 2;;
    --subject-run) SUBJECT_RUN="$2"; shift 2;;
    --assignment-keypoint-group) ASSIGNMENT_KEYPOINT_GROUP="$2"; shift 2;;
    --assignment-keypoints-run) ASSIGNMENT_KEYPOINT_RUN="$2"; shift 2;;
    --baseline-source) BASELINE_SOURCE="$2"; shift 2;;
    --run-id) RUN_ID="$2"; shift 2;;
    --log-root) LOG_ROOT="$2"; shift 2;;
    --queue) QUEUE="$2"; shift 2;;
    --ncores) NCORES="$2"; shift 2;;
    --mem-gb) MEM_GB="$2"; shift 2;;
    --walltime) WALLTIME="$2"; shift 2;;
    --workers) WORKERS="$2"; shift 2;;
    --start-row) START_ROW="$2"; shift 2;;
    --roi-count) ROI_COUNT="$2"; shift 2;;
    --chunk-size) CHUNK_SIZE="$2"; shift 2;;
    --dense-mask-row-chunk) DENSE_MASK_ROW_CHUNK="$2"; shift 2;;
    --repeats) REPEATS="$2"; shift 2;;
    --provenance-repo) PROVENANCE_REPO="$2"; shift 2;;
    --dry-run) DRY_RUN=1; shift;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown argument: $1" >&2; usage; exit 2;;
  esac
done

if [[ -z "$ZARR_PATH" || -z "$SUBJECT_RUN" || -z "$ASSIGNMENT_KEYPOINT_RUN" || -z "$BASELINE_SOURCE" ]]; then
  echo "--zarr, --subject-run, --assignment-keypoints-run, and --baseline-source are required." >&2
  usage
  exit 2
fi
if [[ ! -f "${BASELINE_SOURCE%/}/fisheye/__init__.py" ]]; then
  echo "Baseline source must contain fisheye/__init__.py: $BASELINE_SOURCE" >&2
  exit 2
fi
if [[ ! -d "${PROVENANCE_REPO%/}/.git" ]]; then
  echo "Provenance repository is not a Git checkout: $PROVENANCE_REPO" >&2
  exit 2
fi
if [[ -n "$(git -C "$PROVENANCE_REPO" status --porcelain)" ]]; then
  echo "Provenance repository must be clean before snapshotting: $PROVENANCE_REPO" >&2
  exit 2
fi
CANDIDATE_GIT_SHA="$(git -C "$PROVENANCE_REPO" rev-parse HEAD)"
for value in "$NCORES" "$MEM_GB" "$WORKERS" "$ROI_COUNT" "$CHUNK_SIZE" "$DENSE_MASK_ROW_CHUNK" "$REPEATS"; do
  if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
    echo "Expected a positive integer, got: $value" >&2
    exit 2
  fi
done

RUN_DIR="${LOG_ROOT%/}/${RUN_ID}"
if [[ -e "$RUN_DIR" ]]; then
  echo "Benchmark run directory already exists: $RUN_DIR" >&2
  exit 2
fi
mkdir -p "$RUN_DIR/baseline_source" "$RUN_DIR/candidate_source" "$RUN_DIR/reports"
cp -R --no-preserve=mode,ownership,timestamps \
  "${BASELINE_SOURCE%/}/fisheye" "$RUN_DIR/baseline_source/"
git -C "$PROVENANCE_REPO" archive --format=tar "$CANDIDATE_GIT_SHA" src/fisheye \
  | tar --extract --directory "$RUN_DIR/candidate_source" --strip-components=1 \
      --no-same-owner --no-same-permissions
cp --no-preserve=mode,ownership,timestamps \
  "$PROVENANCE_REPO/scripts/py" "$RUN_DIR/palette_py"
chmod +x "$RUN_DIR/palette_py"
git -C "$PROVENANCE_REPO" status --short > "$RUN_DIR/candidate_git_status.txt"
git -C "$PROVENANCE_REPO" diff --binary > "$RUN_DIR/candidate_worktree.patch"

cat > "$RUN_DIR/settings.sh" <<SETTINGS
PROVENANCE_REPO=$(printf '%q' "$PROVENANCE_REPO")
ZARR_PATH=$(printf '%q' "$ZARR_PATH")
SUBJECT_RUN=$(printf '%q' "$SUBJECT_RUN")
ASSIGNMENT_KEYPOINT_GROUP=$(printf '%q' "$ASSIGNMENT_KEYPOINT_GROUP")
ASSIGNMENT_KEYPOINT_RUN=$(printf '%q' "$ASSIGNMENT_KEYPOINT_RUN")
WORKERS=$WORKERS
START_ROW=$START_ROW
ROI_COUNT=$ROI_COUNT
CHUNK_SIZE=$CHUNK_SIZE
DENSE_MASK_ROW_CHUNK=$DENSE_MASK_ROW_CHUNK
REPEATS=$REPEATS
SETTINGS

cat > "$RUN_DIR/manifest.json" <<JSON
{
  "schema": "palette_subject_mask_finalizer_kernel_ab_submission_v1",
  "run_id": "$RUN_ID",
  "zarr_path": "$ZARR_PATH",
  "subject_run": "$SUBJECT_RUN",
  "assignment_keypoint_group": "$ASSIGNMENT_KEYPOINT_GROUP",
  "assignment_keypoints_run": "$ASSIGNMENT_KEYPOINT_RUN",
  "workers": $WORKERS,
  "start_row": $START_ROW,
  "roi_count": $ROI_COUNT,
  "chunk_size": $CHUNK_SIZE,
  "dense_mask_row_chunk": $DENSE_MASK_ROW_CHUNK,
  "repeats": $REPEATS,
  "provenance_repo": "$PROVENANCE_REPO",
  "candidate_git_sha": "$CANDIDATE_GIT_SHA",
  "order": "position_balanced_ab_ba_ab_ba",
  "copy_policy": "one_real_row_window_per_job",
  "write_eye_geometry": false,
    "write_component_contours": false,
    "write_sampled_component_contours": false
}
JSON

JOB_SCRIPT="$RUN_DIR/run_kernel_ab.sh"
cat > "$JOB_SCRIPT" <<'JOBSCRIPT'
#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="$1"
source "$RUN_DIR/settings.sh"
cd "$PROVENANCE_REPO"
PALETTE_PY="$RUN_DIR/palette_py"
CANDIDATE_PYTHONPATH="$RUN_DIR/candidate_source"

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
  SCRATCH_ROOT="/scratch/${USER_NAME}/${JOB_ID}/subject_mask_finalizer_kernel_ab"
else
  SCRATCH_ROOT="${TMPDIR:-/tmp}/palette_subject_mask_finalizer_kernel_ab_${JOB_ID}"
fi
STAGED_ZARR="$SCRATCH_ROOT/benchmark_window.zarr"

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
echo "allocated_slots=${LSB_DJOB_NUMPROC:-$WORKERS}"
echo "scratch_root=$SCRATCH_ROOT"

PYTHONPATH="$CANDIDATE_PYTHONPATH" "$PALETTE_PY" - \
  "$ZARR_PATH" "$SUBJECT_RUN" "$START_ROW" "$ROI_COUNT" "$STAGED_ZARR" \
  "$ASSIGNMENT_KEYPOINT_GROUP" "$ASSIGNMENT_KEYPOINT_RUN" <<'PY'
import sys
from pathlib import Path

from fisheye.diagnostics.benchmark_subject_mask_full_finalizer import _copy_benchmark_slice

summary = _copy_benchmark_slice(
    sys.argv[1],
    source_run=sys.argv[2],
    start_row=int(sys.argv[3]),
    roi_count=int(sys.argv[4]),
    temp_zarr_path=Path(sys.argv[5]),
    assignment_keypoint_group=sys.argv[6],
    assignment_keypoints_run=sys.argv[7],
)
print(summary)
PY

for repeat in $(seq 1 "$REPEATS"); do
  if (( repeat % 2 == 1 )); then
    order=(baseline candidate)
  else
    order=(candidate baseline)
  fi
  for variant in "${order[@]}"; do
    variant_source="$RUN_DIR/${variant}_source"
    refined_run="refined_subject_masks_kernel_ab_${variant}"
    prefix="$RUN_DIR/reports/${variant}_r$(printf '%02d' "$repeat")"
    cmd=(env "PYTHONPATH=$variant_source" "$PALETTE_PY"
      -m fisheye.refinement.finalize_subject_masks "$STAGED_ZARR"
      --subject-run "$SUBJECT_RUN"
      --run-name "$refined_run"
      --components subject_body eyes_union swim_bladder
      --chunk-size "$CHUNK_SIZE"
      --dense-mask-row-chunk "$DENSE_MASK_ROW_CHUNK"
      --metric-level cheap
      --mask-storage dense_uint8
      --no-write-eye-geometry
      --no-write-component-contours
      --no-write-sampled-component-contours
      --execution-backend process_shards
      --num-workers "$WORKERS"
      --assignment-keypoint-group "$ASSIGNMENT_KEYPOINT_GROUP"
      --assignment-keypoints-run "$ASSIGNMENT_KEYPOINT_RUN"
      --progress-jsonl "${prefix}.progress.jsonl"
      --defer-registry-status
      --overwrite
      --json)

    PYTHONPATH="$CANDIDATE_PYTHONPATH" "$PALETTE_PY" \
      -m fisheye.diagnostics.run_with_resource_telemetry \
      --summary-json "${prefix}.resources.json" \
      --samples-jsonl "${prefix}.resources.jsonl" \
      --stdout-log "${prefix}.stdout" \
      --requested-workers "$WORKERS" \
      --allocated-slots "${LSB_DJOB_NUMPROC:-$WORKERS}" \
      --sample-interval-seconds 2 \
      -- "${cmd[@]}"

    PYTHONPATH="$CANDIDATE_PYTHONPATH" "$PALETTE_PY" - \
      "$STAGED_ZARR" "$refined_run" "${prefix}.json" "${prefix}.resources.json" <<'PY'
import json
import sys
from pathlib import Path

import zarr

root = zarr.open_group(sys.argv[1], mode="r", use_consolidated=False)
run = root[f"refined_subject_masks_runs/{sys.argv[2]}"]
resources = json.loads(Path(sys.argv[4]).read_text(encoding="utf-8"))
payload = {
    "refined_run": sys.argv[2],
    "summary_statistics": run.attrs.get("summary_statistics"),
    "timing_summary": run.attrs.get("smart_finalizer_timing_summary"),
    "resource_telemetry": resources,
}
Path(sys.argv[3]).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
  done
done

PYTHONPATH="$CANDIDATE_PYTHONPATH" "$PALETTE_PY" - "$STAGED_ZARR" "$RUN_DIR/reports" <<'PY'
import json
import statistics
import sys
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


def array_equal(left, right):
    if left.dtype.kind in "fc":
        return bool(np.array_equal(left, right, equal_nan=True))
    return bool(np.array_equal(left, right))


root = zarr.open_group(sys.argv[1], mode="r", use_consolidated=False)
reports = Path(sys.argv[2])
baseline = arrays(root["refined_subject_masks_runs/refined_subject_masks_kernel_ab_baseline"])
candidate = arrays(root["refined_subject_masks_runs/refined_subject_masks_kernel_ab_candidate"])
mismatches = []
if baseline.keys() != candidate.keys():
    mismatches.append("array path sets differ")
for path in sorted(baseline.keys() & candidate.keys()):
    left = baseline[path]
    right = candidate[path]
    if left.shape != right.shape or left.dtype != right.dtype:
        mismatches.append(f"{path}: shape/dtype differ")
        continue
    if left.ndim == 0:
        if not array_equal(np.asarray(left[()]), np.asarray(right[()])):
            mismatches.append(f"{path}: scalar differs")
        continue
    rows = int(left.shape[0])
    row_chunk = min(256, int(left.chunks[0] or 1))
    for start in range(0, rows, row_chunk):
        stop = min(rows, start + row_chunk)
        if not array_equal(np.asarray(left[start:stop]), np.asarray(right[start:stop])):
            mismatches.append(f"{path}: rows {start}:{stop} differ")
            break

parity = {
    "schema": "palette_subject_mask_finalizer_kernel_ab_parity_v1",
    "baseline_array_count": len(baseline),
    "candidate_array_count": len(candidate),
    "mismatch_count": len(mismatches),
    "mismatches": mismatches,
}
(reports / "parity.json").write_text(json.dumps(parity, indent=2, sort_keys=True) + "\n", encoding="utf-8")

summary = {"schema": "palette_subject_mask_finalizer_kernel_ab_summary_v1", "parity": parity}
for variant in ("baseline", "candidate"):
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(reports.glob(f"{variant}_r??.json"))]
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
        "peak_rss_bytes": peak_rss,
        "peak_rss_median_bytes": statistics.median(peak_rss),
    }
(reports / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(summary, sort_keys=True))
if mismatches:
    raise SystemExit(1)
PY
JOBSCRIPT
chmod +x "$JOB_SCRIPT"

BSUB_ARGS=(-J "sm_fin_kernel_ab" -n "$NCORES" -W "$WALLTIME" -R "rusage[mem=${MEM_GB}G]" -oo "$RUN_DIR/%J.out" -eo "$RUN_DIR/%J.err")
if [[ -n "$QUEUE" ]]; then
  BSUB_ARGS+=(-q "$QUEUE")
fi

printf 'Run dir: %s\n' "$RUN_DIR"
printf 'Command: bsub'
for arg in "${BSUB_ARGS[@]}"; do
  printf ' %q' "$arg"
done
printf ' bash %q %q\n' "$JOB_SCRIPT" "$RUN_DIR"

if [[ "$DRY_RUN" == "1" ]]; then
  exit 0
fi
bsub "${BSUB_ARGS[@]}" bash "$JOB_SCRIPT" "$RUN_DIR"
