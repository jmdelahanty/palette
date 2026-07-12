#!/usr/bin/env bash
set -euo pipefail

FIXTURE_ROOT=""
RUN_ID="subject_mask_input_staging_workers_$(date +%Y%m%d_%H%M%S)"
LOG_ROOT="/groups/johnson/johnsonlab/jeremy/recordings/logs/subject_mask_input_staging_workers"
QUEUE=""
SUBMIT_HOST="${PALETTE_LSF_SUBMIT_HOST:-}"
NCORES=24
MEM_GB_PER_SLOT=4
WALLTIME="4:00"
REPEATS=2
CHUNK_SIZE=256
DENSE_MASK_ROW_CHUNK=256
WAIT_FOR_JOB=""
DRY_RUN=0
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

usage() {
  cat <<'USAGE'
Usage: submit_subject_mask_input_staging_workers_bsub.sh --fixture-root PATH [options]

Runs a position-balanced 2x2 subject-mask finalizer benchmark:

  direct PRFS input / 16 workers
  staged subject-mask input / 16 workers
  direct PRFS input / 24 workers
  staged subject-mask input / 24 workers

The selected subject_mask_runs input is either symlinked to PRFS or copied once
to node-local scratch. Remaining lineage inputs stay linked to PRFS. Complete
default refined surfaces are written only to node-local scratch and compared
exactly. Reports go to PRFS; fixture inputs and the registry remain read-only.

Options:
  --run-id ID             Unique run/log directory name.
  --log-root PATH         PRFS log root.
  --queue NAME            LSF queue.
  --submit-host HOST      SSH host used when bsub is unavailable locally.
  --ncores N              LSF slots (default: 24; must be at least 24).
  --mem-gb-per-slot N     Client memory request per slot (default: 4).
  --walltime H:MM         LSF walltime (default: 4:00).
  --repeats N             Position-balanced repeats (default: 2).
  --chunk-size N          Logical worker rows (default: 256).
  --dense-mask-row-chunk N  Dense masks_roi rows (default: 256).
  --wait-for-job JOBID     Start only after this LSF job finishes successfully.
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
    --repeats) REPEATS="$2"; shift 2;;
    --chunk-size) CHUNK_SIZE="$2"; shift 2;;
    --dense-mask-row-chunk) DENSE_MASK_ROW_CHUNK="$2"; shift 2;;
    --wait-for-job) WAIT_FOR_JOB="$2"; shift 2;;
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
SOURCE_ZARR="$FIXTURE_ROOT/shard_02048.zarr"
SUBJECT_RUN="subject_masks_finalizer_ab_fixture"
KEYPOINT_RUN="refined_keypoints_finalizer_ab_fixture"
if [[ ! -f "$SOURCE_ZARR/zarr.json" ]]; then
  echo "Sharded fixture Zarr not found: $SOURCE_ZARR" >&2
  exit 2
fi
if [[ ! -f "$SOURCE_ZARR/subject_mask_runs/$SUBJECT_RUN/zarr.json" ]]; then
  echo "Fixture subject-mask run not found: $SUBJECT_RUN" >&2
  exit 2
fi
for value in "$NCORES" "$MEM_GB_PER_SLOT" "$REPEATS" "$CHUNK_SIZE" "$DENSE_MASK_ROW_CHUNK"; do
  if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
    echo "Expected a positive integer, got: $value" >&2
    exit 2
  fi
done
if (( NCORES < 24 )); then
  echo "--ncores must be at least 24 for the 24-worker cases." >&2
  exit 2
fi
if (( CHUNK_SIZE % DENSE_MASK_ROW_CHUNK != 0 )); then
  echo "--chunk-size must be a multiple of --dense-mask-row-chunk." >&2
  exit 2
fi
if [[ -n "$WAIT_FOR_JOB" && ! "$WAIT_FOR_JOB" =~ ^[1-9][0-9]*$ ]]; then
  echo "--wait-for-job must be a positive LSF job ID." >&2
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
RUN_TAG="$(printf '%s' "$RUN_ID" | tr -c 'A-Za-z0-9_-' '_')"
git -C "$REPO_ROOT" archive --format=tar "$SOURCE_GIT_HEAD" src/fisheye \
  | tar --extract --directory "$RUN_DIR/source_snapshot" --strip-components=1 \
      --no-same-owner --no-same-permissions
git -C "$REPO_ROOT" bundle create "$RUN_DIR/source_git.bundle" HEAD
cp --no-preserve=mode,ownership,timestamps "$REPO_ROOT/scripts/py" "$RUN_DIR/palette_py"
chmod +x "$RUN_DIR/palette_py"

MATRIX_TSV="$RUN_DIR/matrix.tsv"
: > "$MATRIX_TSV"
for ((repeat = 1; repeat <= REPEATS; repeat++)); do
  if (( repeat % 2 == 1 )); then
    order=(direct_w16 staged_w16 direct_w24 staged_w24)
  else
    order=(staged_w24 direct_w24 staged_w16 direct_w16)
  fi
  for variant in "${order[@]}"; do
    placement="${variant%%_*}"
    workers="${variant##*_w}"
    printf '%s_r%02d\t%s\t%s\t%d\n' "$variant" "$repeat" "$placement" "$workers" "$repeat" >> "$MATRIX_TSV"
  done
done

cat > "$RUN_DIR/settings.sh" <<SETTINGS
SOURCE_ZARR=$(printf '%q' "$SOURCE_ZARR")
SUBJECT_RUN=$(printf '%q' "$SUBJECT_RUN")
KEYPOINT_RUN=$(printf '%q' "$KEYPOINT_RUN")
RUN_TAG=$(printf '%q' "$RUN_TAG")
CHUNK_SIZE=$CHUNK_SIZE
DENSE_MASK_ROW_CHUNK=$DENSE_MASK_ROW_CHUNK
SOURCE_GIT_HEAD=$(printf '%q' "$SOURCE_GIT_HEAD")
SETTINGS

cat > "$RUN_DIR/manifest.json" <<JSON
{
  "schema_id": "palette.subject_mask_input_staging_workers_submission.v1",
  "run_id": "$RUN_ID",
  "fixture_root": "$FIXTURE_ROOT",
  "source_zarr": "$SOURCE_ZARR",
  "subject_run": "$SUBJECT_RUN",
  "worker_counts": [16, 24],
  "input_placements": ["direct_prfs", "staged_subject_run"],
  "repeats": $REPEATS,
  "wait_for_job": $(if [[ -n "$WAIT_FOR_JOB" ]]; then printf '%s' "$WAIT_FOR_JOB"; else printf 'null'; fi),
  "case_order": "forward_then_reverse",
  "chunk_size": $CHUNK_SIZE,
  "dense_mask_row_chunk": $DENSE_MASK_ROW_CHUNK,
  "complete_default_surfaces": true,
  "source_git_head": "$SOURCE_GIT_HEAD",
  "source_git_branch": "$SOURCE_GIT_BRANCH",
  "outputs": "node_local_only_until_exact_parity",
  "mutates_fixture_zarr": false,
  "mutates_registry": false
}
JSON

JOB_SCRIPT="$RUN_DIR/run_input_staging_workers.sh"
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
  SCRATCH_ROOT="/scratch/${USER_NAME}/${JOB_ID}/subject_mask_input_staging_workers"
else
  SCRATCH_ROOT="${TMPDIR:-/tmp}/subject_mask_input_staging_workers_${JOB_ID}"
fi
PROVENANCE_CHECKOUT="$SCRATCH_ROOT/provenance_checkout"
STAGING_ROOT="$SCRATCH_ROOT/cases"
cleanup() {
  local status=$?
  trap - EXIT INT TERM
  if [[ -d "$SCRATCH_ROOT" ]]; then rm -rf "$SCRATCH_ROOT"; fi
  exit "$status"
}
trap cleanup EXIT INT TERM
mkdir -p "$STAGING_ROOT" "$RUN_DIR/reports"
git clone --quiet "$RUN_DIR/source_git.bundle" "$PROVENANCE_CHECKOUT"
git -C "$PROVENANCE_CHECKOUT" checkout --quiet --detach "$SOURCE_GIT_HEAD"
cd "$PROVENANCE_CHECKOUT"

echo "host=$(hostname)"
echo "job_id=${LSB_JOBID:-}"
echo "allocated_slots=${LSB_DJOB_NUMPROC:-24}"
echo "source_zarr=$SOURCE_ZARR"
echo "scratch_root=$SCRATCH_ROOT"
echo "provenance_git_head=$(git rev-parse HEAD)"

while IFS=$'\t' read -r CASE_ID PLACEMENT WORKERS REPEAT; do
  prefix="$RUN_DIR/reports/$CASE_ID"
  refined_run="refined_subject_masks_input_staging_${RUN_TAG}_${CASE_ID}"

  "$PALETTE_PY" - "$SOURCE_ZARR/subject_mask_runs/$SUBJECT_RUN" \
    "${prefix}.cache_eviction.json" <<'PY'
import json
import sys
from pathlib import Path
from fisheye.diagnostics.benchmark_subject_mask_probability_sharding_reads import _request_cache_eviction

payload = _request_cache_eviction(Path(sys.argv[1]))
Path(sys.argv[2]).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

  staged_zarr="$(
    "$PALETTE_PY" - "$SOURCE_ZARR" "$STAGING_ROOT/$CASE_ID.zarr" "$SUBJECT_RUN" \
      "$PLACEMENT" "${prefix}.staging.json" <<'PY'
import json
import os
import shutil
import sys
import time
from pathlib import Path

source = Path(sys.argv[1]).resolve()
target = Path(sys.argv[2])
subject_run = sys.argv[3]
placement = sys.argv[4]
report_path = Path(sys.argv[5])
started = time.perf_counter()
if target.exists() or target.is_symlink():
    if target.is_symlink() or target.is_file():
        target.unlink()
    else:
        shutil.rmtree(target)
target.mkdir(parents=True)
shutil.copy2(source / "zarr.json", target / "zarr.json")
for child in source.iterdir():
    if child.name in {"zarr.json", "refined_subject_masks_runs", "subject_mask_runs"}:
        continue
    os.symlink(child, target / child.name, target_is_directory=child.is_dir())

source_parent = source / "subject_mask_runs"
target_parent = target / "subject_mask_runs"
target_parent.mkdir()
shutil.copy2(source_parent / "zarr.json", target_parent / "zarr.json")
for child in source_parent.iterdir():
    if child.name == "zarr.json":
        continue
    destination = target_parent / child.name
    if child.name == subject_run and placement == "staged":
        shutil.copytree(child, destination, symlinks=True)
    else:
        os.symlink(child, destination, target_is_directory=child.is_dir())

copied_root = target_parent / subject_run if placement == "staged" else target
regular_files = 0
apparent_bytes = 0
allocated_bytes = 0
for path in copied_root.rglob("*"):
    if not path.is_file() or path.is_symlink():
        continue
    stat = path.stat()
    regular_files += 1
    apparent_bytes += int(stat.st_size)
    allocated_bytes += int(getattr(stat, "st_blocks", 0)) * 512
payload = {
    "schema_id": "palette.subject_mask_input_staging_case.v1",
    "placement": placement,
    "source_zarr": str(source),
    "staged_zarr": str(target),
    "duration_seconds": float(time.perf_counter() - started),
    "copied_regular_files": regular_files,
    "copied_apparent_bytes": apparent_bytes,
    "copied_allocated_bytes": allocated_bytes,
}
report_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(target)
PY
  )"

  cmd=("$PALETTE_PY" -m fisheye.refinement.finalize_subject_masks "$staged_zarr"
    --subject-run "$SUBJECT_RUN"
    --run-name "$refined_run"
    --components subject_body eye_left eye_right swim_bladder
    --chunk-size "$CHUNK_SIZE"
    --dense-mask-row-chunk "$DENSE_MASK_ROW_CHUNK"
    --metric-level cheap
    --mask-storage dense_uint8
    --write-eye-geometry
    --write-component-contours
    --write-sampled-component-contours
    --postcompute-backend process_shards
    --postcompute-chunk-size "$CHUNK_SIZE"
    --postcompute-num-workers "$WORKERS"
    --execution-backend process_shards
    --num-workers "$WORKERS"
    --assignment-keypoint-group refined_keypoints_runs
    --assignment-keypoints-run "$KEYPOINT_RUN"
    --progress-jsonl "${prefix}.progress.jsonl"
    --defer-registry-status
    --json)

  echo "starting case=$CASE_ID placement=$PLACEMENT workers=$WORKERS repeat=$REPEAT"
  "$PALETTE_PY" -m fisheye.diagnostics.run_with_resource_telemetry \
    --summary-json "${prefix}.resources.json" \
    --samples-jsonl "${prefix}.resources.jsonl" \
    --stdout-log "${prefix}.stdout" \
    --requested-workers "$WORKERS" \
    --allocated-slots "${LSB_DJOB_NUMPROC:-24}" \
    --sample-interval-seconds 2 \
    -- "${cmd[@]}"

  "$PALETTE_PY" - "$staged_zarr" "$refined_run" "$CASE_ID" "$PLACEMENT" \
    "$WORKERS" "$REPEAT" "${prefix}.json" "${prefix}.resources.json" \
    "${prefix}.staging.json" "${prefix}.cache_eviction.json" <<'PY'
import json
import sys
from pathlib import Path
import zarr

root = zarr.open_group(sys.argv[1], mode="r", use_consolidated=False)
run = root[f"refined_subject_masks_runs/{sys.argv[2]}"]
resources = json.loads(Path(sys.argv[8]).read_text(encoding="utf-8"))
staging = json.loads(Path(sys.argv[9]).read_text(encoding="utf-8"))
payload = {
    "schema_id": "palette.subject_mask_input_staging_workers_case.v1",
    "case_id": sys.argv[3],
    "placement": sys.argv[4],
    "workers": int(sys.argv[5]),
    "repeat": int(sys.argv[6]),
    "refined_run": sys.argv[2],
    "staged_zarr": sys.argv[1],
    "staging": staging,
    "cache_eviction": json.loads(Path(sys.argv[10]).read_text(encoding="utf-8")),
    "resource_telemetry": resources,
    "finalizer_seconds": float(resources["duration_seconds"]),
    "end_to_end_seconds": float(resources["duration_seconds"]) + float(staging["duration_seconds"]),
    "timing_summary": run.attrs.get("smart_finalizer_timing_summary"),
    "postcompute_summary": run.attrs.get("smart_finalizer_postcompute_summary"),
    "completion_status": run.attrs.get("palette_run_completion_status"),
    "surface_status": {
        "eye_geometry": run.attrs.get("eye_geometry_status"),
        "component_contours": run.attrs.get("component_contours_status"),
        "sampled_component_contours": run.attrs.get("sampled_component_contours_status"),
    },
}
Path(sys.argv[7]).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
  echo "completed case=$CASE_ID"
done < "$RUN_DIR/matrix.tsv"

echo "starting exact complete-surface parity validation"
"$PALETTE_PY" - "$RUN_DIR/matrix.tsv" "$RUN_DIR/reports" <<'PY'
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


def row_step(array):
    chunk = max(1, int(array.chunks[0] or 1))
    if array.ndim >= 3 and tuple(array.shape[-2:]) == (512, 512):
        return min(64, chunk)
    return min(65536, chunk)


matrix_path = Path(sys.argv[1])
reports = Path(sys.argv[2])
case_ids = [line.split("\t", 1)[0] for line in matrix_path.read_text().splitlines() if line]
payloads = {
    case: json.loads((reports / f"{case}.json").read_text(encoding="utf-8"))
    for case in case_ids
}
runs = {}
for case, payload in payloads.items():
    root = zarr.open_group(payload["staged_zarr"], mode="r", use_consolidated=False)
    runs[case] = arrays(root[f"refined_subject_masks_runs/{payload['refined_run']}"])

reference_case = case_ids[0]
reference = runs[reference_case]
started = time.perf_counter()
comparisons = []
all_mismatches = []
for case in case_ids[1:]:
    candidate = runs[case]
    mismatches = []
    if reference.keys() != candidate.keys():
        mismatches.append("array path sets differ")
    for path in sorted(reference.keys() & candidate.keys()):
        left = reference[path]
        right = candidate[path]
        if left.shape != right.shape or left.dtype != right.dtype:
            mismatches.append(f"{path}: shape/dtype differ")
            continue
        if left.ndim == 0:
            if not equal(np.asarray(left[()]), np.asarray(right[()])):
                mismatches.append(f"{path}: scalar differs")
            continue
        step = row_step(left)
        for start in range(0, int(left.shape[0]), step):
            stop = min(int(left.shape[0]), start + step)
            if not equal(np.asarray(left[start:stop]), np.asarray(right[start:stop])):
                mismatches.append(f"{path}: rows {start}:{stop} differ")
                break
    comparisons.append({
        "reference": reference_case,
        "candidate": case,
        "reference_array_count": len(reference),
        "candidate_array_count": len(candidate),
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
    })
    all_mismatches.extend(f"{case}: {item}" for item in mismatches)

variants = {}
for payload in payloads.values():
    key = f"{payload['placement']}_w{payload['workers']}"
    variants.setdefault(key, []).append(payload)
variant_summary = {}
for key, rows in variants.items():
    finalizer = [float(row["finalizer_seconds"]) for row in rows]
    end_to_end = [float(row["end_to_end_seconds"]) for row in rows]
    staging = [float(row["staging"]["duration_seconds"]) for row in rows]
    variant_summary[key] = {
        "run_count": len(rows),
        "finalizer_seconds": finalizer,
        "median_finalizer_seconds": float(statistics.median(finalizer)),
        "end_to_end_seconds": end_to_end,
        "median_end_to_end_seconds": float(statistics.median(end_to_end)),
        "staging_seconds": staging,
        "median_staging_seconds": float(statistics.median(staging)),
        "peak_process_tree_rss_bytes": max(
            int(row["resource_telemetry"]["peak_process_tree_rss_bytes"]) for row in rows
        ),
        "median_cpu_efficiency_percent": float(statistics.median(
            float(row["resource_telemetry"]["cpu_efficiency_percent_of_requested_workers"])
            for row in rows
        )),
    }

summary = {
    "schema_id": "palette.subject_mask_input_staging_workers_summary.v1",
    "case_order": case_ids,
    "cases": payloads,
    "variants": variant_summary,
    "parity": {
        "reference_case": reference_case,
        "comparison_count": len(comparisons),
        "comparisons": comparisons,
        "mismatch_count": len(all_mismatches),
        "mismatches": all_mismatches,
        "duration_seconds": float(time.perf_counter() - started),
    },
}
(reports / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(summary, sort_keys=True))
incomplete = [
    case for case, payload in payloads.items()
    if payload["completion_status"] != "complete"
    or payload["surface_status"]["eye_geometry"] != "computed"
    or payload["surface_status"]["component_contours"] != "computed"
    or payload["surface_status"]["sampled_component_contours"] != "computed"
]
if incomplete:
    raise SystemExit(f"Incomplete benchmark cases: {incomplete}")
if all_mismatches:
    raise SystemExit(1)
PY
JOBSCRIPT
chmod +x "$JOB_SCRIPT"

BSUB_ARGS=(-J "sm_stage_workers" -n "$NCORES" -W "$WALLTIME" -R "rusage[mem=${MEM_GB_PER_SLOT}G]" -oo "$RUN_DIR/%J.out" -eo "$RUN_DIR/%J.err")
if [[ -n "$QUEUE" ]]; then BSUB_ARGS+=(-q "$QUEUE"); fi
if [[ -n "$WAIT_FOR_JOB" ]]; then BSUB_ARGS+=(-w "done($WAIT_FOR_JOB)"); fi
printf 'Run dir: %s\n' "$RUN_DIR"
printf 'Source Git HEAD: %s\n' "$SOURCE_GIT_HEAD"
printf 'Command: bsub'
for arg in "${BSUB_ARGS[@]}"; do printf ' %q' "$arg"; done
printf ' bash %q %q\n' "$JOB_SCRIPT" "$RUN_DIR"
if [[ "$DRY_RUN" == "1" ]]; then exit 0; fi
if command -v bsub >/dev/null 2>&1; then
  bsub "${BSUB_ARGS[@]}" bash "$JOB_SCRIPT" "$RUN_DIR"
elif [[ -n "$SUBMIT_HOST" ]]; then
  printf -v REMOTE_COMMAND '%q ' bsub "${BSUB_ARGS[@]}" bash "$JOB_SCRIPT" "$RUN_DIR"
  ssh "$SUBMIT_HOST" "$REMOTE_COMMAND"
else
  echo "bsub is unavailable locally; rerun with --submit-host HOST or submit on an LSF login node." >&2
  exit 127
fi
