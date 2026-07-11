#!/usr/bin/env bash
set -euo pipefail

FIXTURE_ROOT=""
RUN_ID="subject_mask_complete_finalizer_matrix_$(date +%Y%m%d_%H%M%S)"
LOG_ROOT="/groups/johnson/johnsonlab/jeremy/recordings/logs/subject_mask_complete_finalizer_matrix"
QUEUE=""
SUBMIT_HOST="${PALETTE_LSF_SUBMIT_HOST:-}"
NCORES=16
MEM_GB_PER_SLOT=4
WALLTIME="4:00"
CHUNK_SIZE=256
DENSE_MASK_ROW_CHUNK=256
DRY_RUN=0
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

usage() {
  cat <<'USAGE'
Usage: submit_subject_mask_complete_finalizer_matrix_bsub.sh --fixture-root PATH [options]

Runs four complete-contract subject-mask finalizations in one LSF job:

  regular/8 workers -> sharded/8 -> sharded/16 -> regular/16

Raw probabilities and assignment context remain on PRFS. Each refined output is
written to a unique node-local overlay, then copied and atomically committed to
a benchmark-only PRFS Zarr through the production run-group publication helper.
Eye geometry, full ragged contours, and sampled contours are always enabled.
The job records resource telemetry and phase timings, inventories outputs, and
compares every corresponding output array exactly. It does not mutate either
fixture input Zarr or the registry.

Options:
  --run-id ID             Unique run/log directory name.
  --log-root PATH         PRFS log root.
  --queue NAME            LSF queue.
  --submit-host HOST      SSH host used when bsub is unavailable locally.
  --ncores N              LSF slots (default: 16; must be at least 16).
  --mem-gb-per-slot N     Client memory request per slot (default: 4).
  --walltime H:MM         LSF walltime (default: 4:00).
  --chunk-size N          Logical worker rows (default: 256).
  --dense-mask-row-chunk N  Dense masks_roi rows (default: 256).
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
    --chunk-size) CHUNK_SIZE="$2"; shift 2;;
    --dense-mask-row-chunk) DENSE_MASK_ROW_CHUNK="$2"; shift 2;;
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
for value in "$NCORES" "$MEM_GB_PER_SLOT" "$CHUNK_SIZE" "$DENSE_MASK_ROW_CHUNK"; do
  if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
    echo "Expected a positive integer, got: $value" >&2
    exit 2
  fi
done
if (( NCORES < 16 )); then
  echo "--ncores must be at least 16 for the 16-worker cases." >&2
  exit 2
fi
if (( CHUNK_SIZE % DENSE_MASK_ROW_CHUNK != 0 )); then
  echo "--chunk-size must be a multiple of --dense-mask-row-chunk for safe parallel writes." >&2
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
cat > "$MATRIX_TSV" <<MATRIX
regular_w08	regular	8
sharded_w08	sharded	8
sharded_w16	sharded	16
regular_w16	regular	16
MATRIX

PUBLISH_ZARR="$RUN_DIR/published_outputs.zarr"
cat > "$RUN_DIR/settings.sh" <<SETTINGS
FIXTURE_ROOT=$(printf '%q' "$FIXTURE_ROOT")
REGULAR_ZARR=$(printf '%q' "$REGULAR_ZARR")
SHARDED_ZARR=$(printf '%q' "$SHARDED_ZARR")
PUBLISH_ZARR=$(printf '%q' "$PUBLISH_ZARR")
RUN_TAG=$(printf '%q' "$RUN_TAG")
CHUNK_SIZE=$CHUNK_SIZE
DENSE_MASK_ROW_CHUNK=$DENSE_MASK_ROW_CHUNK
SOURCE_GIT_HEAD=$(printf '%q' "$SOURCE_GIT_HEAD")
SETTINGS

cat > "$RUN_DIR/manifest.json" <<JSON
{
  "schema_id": "palette.subject_mask_complete_finalizer_matrix_submission.v1",
  "run_id": "$RUN_ID",
  "fixture_root": "$FIXTURE_ROOT",
  "regular_zarr": "$REGULAR_ZARR",
  "sharded_zarr": "$SHARDED_ZARR",
  "published_output_zarr": "$PUBLISH_ZARR",
  "matrix_order": ["regular_w08", "sharded_w08", "sharded_w16", "regular_w16"],
  "chunk_size": $CHUNK_SIZE,
  "dense_mask_row_chunk": $DENSE_MASK_ROW_CHUNK,
  "write_eye_geometry": true,
  "write_component_contours": true,
  "write_sampled_component_contours": true,
  "postcompute_backend": "process_shards",
  "source_git_head": "$SOURCE_GIT_HEAD",
  "source_git_branch": "$SOURCE_GIT_BRANCH",
  "provenance_checkout": "node_local_clean_checkout_from_source_git.bundle",
  "raw_reads": "direct_prfs_via_node_local_overlay",
  "refined_writes": "node_local_then_production_helper_prfs_publish",
  "mutates_fixture_zarrs": false,
  "mutates_registry": false
}
JSON

JOB_SCRIPT="$RUN_DIR/run_complete_matrix.sh"
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
  SCRATCH_ROOT="/scratch/${USER_NAME}/${JOB_ID}/subject_mask_complete_finalizer_matrix"
else
  SCRATCH_ROOT="${TMPDIR:-/tmp}/subject_mask_complete_finalizer_matrix_${JOB_ID}"
fi
PROVENANCE_CHECKOUT="$SCRATCH_ROOT/provenance_checkout"
STAGING_ROOT="$SCRATCH_ROOT/staged_zarrs"
cleanup() {
  local status=$?
  trap - EXIT INT TERM
  if [[ -d "$SCRATCH_ROOT" ]]; then
    rm -rf "$SCRATCH_ROOT"
  fi
  exit "$status"
}
trap cleanup EXIT INT TERM
mkdir -p "$SCRATCH_ROOT" "$STAGING_ROOT" "$RUN_DIR/reports"
git clone --quiet "$RUN_DIR/source_git.bundle" "$PROVENANCE_CHECKOUT"
git -C "$PROVENANCE_CHECKOUT" checkout --quiet --detach "$SOURCE_GIT_HEAD"
cd "$PROVENANCE_CHECKOUT"

"$PALETTE_PY" - "$PUBLISH_ZARR" <<'PY'
import sys
import zarr

root = zarr.open_group(sys.argv[1], mode="w", use_consolidated=False)
root.attrs.update(
    {
        "schema_id": "palette.subject_mask_complete_finalizer_matrix_outputs.v1",
        "benchmark_only": True,
        "mutates_registry": False,
    }
)
root.require_group("refined_subject_masks_runs")
PY

echo "host=$(hostname)"
echo "job_id=${LSB_JOBID:-}"
echo "allocated_slots=${LSB_DJOB_NUMPROC:-16}"
echo "fixture_root=$FIXTURE_ROOT"
echo "scratch_root=$SCRATCH_ROOT"
echo "publish_zarr=$PUBLISH_ZARR"
echo "provenance_git_head=$(git rev-parse HEAD)"
echo "provenance_git_dirty=$(test -n "$(git status --porcelain)" && echo true || echo false)"

while IFS=$'\t' read -r CASE_ID LAYOUT WORKERS; do
  if [[ -z "$CASE_ID" ]]; then
    continue
  fi
  if [[ "$LAYOUT" == "regular" ]]; then
    source_zarr="$REGULAR_ZARR"
  else
    source_zarr="$SHARDED_ZARR"
  fi
  refined_run="refined_subject_masks_complete_matrix_${RUN_TAG}_${CASE_ID}"
  prefix="$RUN_DIR/reports/$CASE_ID"

  staged_zarr="$(
    "$PALETTE_PY" - "$source_zarr" "$STAGING_ROOT" "$CASE_ID" <<'PY'
import sys
from pathlib import Path

from fisheye.utils.finalize_subject_mask_clip_package import _stage_zarr_with_local_refined_parent

path = _stage_zarr_with_local_refined_parent(
    source_zarr=Path(sys.argv[1]),
    staging_root=Path(sys.argv[2]),
    staging_name=sys.argv[3],
    overwrite=True,
)
print(path)
PY
  )"

  cmd=("$PALETTE_PY" -m fisheye.refinement.finalize_subject_masks "$staged_zarr"
    --subject-run subject_masks_finalizer_ab_fixture
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
    --assignment-keypoints-run refined_keypoints_finalizer_ab_fixture
    --progress-jsonl "${prefix}.progress.jsonl"
    --defer-registry-status
    --json)

  echo "starting case=$CASE_ID layout=$LAYOUT workers=$WORKERS source=$source_zarr"
  "$PALETTE_PY" -m fisheye.diagnostics.run_with_resource_telemetry \
    --summary-json "${prefix}.resources.json" \
    --samples-jsonl "${prefix}.resources.jsonl" \
    --stdout-log "${prefix}.stdout" \
    --requested-workers "$WORKERS" \
    --allocated-slots "${LSB_DJOB_NUMPROC:-16}" \
    --sample-interval-seconds 2 \
    -- "${cmd[@]}"

  "$PALETTE_PY" - "$staged_zarr" "$refined_run" "$PUBLISH_ZARR" \
    "${prefix}.publish.json" <<'PY'
import json
import sys
import time
from pathlib import Path

import zarr

from fisheye.shared.zarr_run_completion import RUN_COMPLETION_STATUS_ATTR
from fisheye.utils.run_subject_mask_batch_pipeline import (
    _commit_run_group_publish,
    _prepare_run_group_publish,
    _run_group_storage_stats,
)

staged_zarr = Path(sys.argv[1])
run_name = sys.argv[2]
publish_zarr = Path(sys.argv[3])
out_path = Path(sys.argv[4])
source_parent = staged_zarr / "refined_subject_masks_runs"
target_parent = publish_zarr / "refined_subject_masks_runs"
started = time.perf_counter()
plan = _prepare_run_group_publish(
    staged_parent=source_parent,
    target_parent=target_parent,
    run_name=run_name,
    overwrite=False,
)
prepared_elapsed = time.perf_counter() - started
commit_started = time.perf_counter()
_commit_run_group_publish(plan)
commit_seconds = time.perf_counter() - commit_started
validation_started = time.perf_counter()
root = zarr.open_group(str(publish_zarr), mode="r", use_consolidated=False)
run = root[f"refined_subject_masks_runs/{run_name}"]
required = {
    "eye_geometry": run.attrs.get("eye_geometry_status") == "computed",
    "component_contours": run.attrs.get("component_contours_status") == "computed",
    "sampled_component_contours": run.attrs.get("sampled_component_contours_status") == "computed",
    "completion": run.attrs.get(RUN_COMPLETION_STATUS_ATTR) == "complete",
}
published_stats = _run_group_storage_stats(target_parent / run_name)
payload = {
    "schema_id": "palette.subject_mask_complete_finalizer_publish.v1",
    "run_name": run_name,
    "source_path": str(plan.source_path),
    "temporary_path": str(plan.tmp_path),
    "target_path": str(plan.target_path),
    "publish_backend": plan.publish_backend,
    "source_storage_stats": dict(plan.storage_stats),
    "prepare_total_seconds": float(prepared_elapsed),
    "copy_seconds": float(plan.copy_duration_seconds),
    "commit_seconds": float(commit_seconds),
    "validation_seconds": float(time.perf_counter() - validation_started),
    "published_storage_stats": published_stats,
    "required_surface_checks": required,
    "all_required_surfaces": all(required.values()),
}
out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
if not payload["all_required_surfaces"]:
    raise SystemExit(f"Required finalized surfaces missing: {required}")
PY

  "$PALETTE_PY" - "$PUBLISH_ZARR" "$refined_run" "$CASE_ID" "$LAYOUT" \
    "$WORKERS" "${prefix}.json" "${prefix}.resources.json" \
    "${prefix}.publish.json" <<'PY'
import json
import sys
from pathlib import Path

import zarr

root = zarr.open_group(sys.argv[1], mode="r", use_consolidated=False)
run = root[f"refined_subject_masks_runs/{sys.argv[2]}"]
payload = {
    "schema_id": "palette.subject_mask_complete_finalizer_case.v1",
    "case_id": sys.argv[3],
    "layout": sys.argv[4],
    "workers": int(sys.argv[5]),
    "refined_run": sys.argv[2],
    "summary_statistics": run.attrs.get("summary_statistics"),
    "timing_summary": run.attrs.get("smart_finalizer_timing_summary"),
    "postcompute_summary": run.attrs.get("smart_finalizer_postcompute_summary"),
    "worker_process_count": run.attrs.get("worker_process_count"),
    "worker_chunk_size": run.attrs.get("worker_chunk_size"),
    "surface_status": {
        "eye_geometry": run.attrs.get("eye_geometry_status"),
        "component_contours": run.attrs.get("component_contours_status"),
        "sampled_component_contours": run.attrs.get("sampled_component_contours_status"),
    },
    "resource_telemetry": json.loads(Path(sys.argv[7]).read_text(encoding="utf-8")),
    "publication": json.loads(Path(sys.argv[8]).read_text(encoding="utf-8")),
}
Path(sys.argv[6]).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

  rm -rf "$staged_zarr"
  echo "completed case=$CASE_ID"
done < "$RUN_DIR/matrix.tsv"

echo "starting exact complete-surface parity validation"
"$PALETTE_PY" - "$PUBLISH_ZARR" "$RUN_DIR/reports" "$RUN_TAG" <<'PY'
import json
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


publish_zarr = Path(sys.argv[1])
reports = Path(sys.argv[2])
run_tag = sys.argv[3]
root = zarr.open_group(str(publish_zarr), mode="r", use_consolidated=False)
cases = ["regular_w08", "sharded_w08", "sharded_w16", "regular_w16"]
runs = {
    case: arrays(root[f"refined_subject_masks_runs/refined_subject_masks_complete_matrix_{run_tag}_{case}"])
    for case in cases
}
reference_case = cases[0]
reference = runs[reference_case]
parity_started = time.perf_counter()
comparisons = []
all_mismatches = []
for case in cases[1:]:
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
    comparisons.append(
        {
            "reference": reference_case,
            "candidate": case,
            "reference_array_count": len(reference),
            "candidate_array_count": len(candidate),
            "mismatch_count": len(mismatches),
            "mismatches": mismatches,
        }
    )
    all_mismatches.extend(f"{case}: {item}" for item in mismatches)

case_payloads = {
    case: json.loads((reports / f"{case}.json").read_text(encoding="utf-8"))
    for case in cases
}
summary = {
    "schema_id": "palette.subject_mask_complete_finalizer_matrix_summary.v1",
    "case_order": cases,
    "cases": case_payloads,
    "parity": {
        "reference_case": reference_case,
        "comparison_count": len(comparisons),
        "comparisons": comparisons,
        "mismatch_count": len(all_mismatches),
        "mismatches": all_mismatches,
        "duration_seconds": float(time.perf_counter() - parity_started),
    },
}
(reports / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(summary, sort_keys=True))
if all_mismatches:
    raise SystemExit(1)
PY
JOBSCRIPT
chmod +x "$JOB_SCRIPT"

BSUB_ARGS=(-J "sm_complete_matrix" -n "$NCORES" -W "$WALLTIME" -R "rusage[mem=${MEM_GB_PER_SLOT}G]" -oo "$RUN_DIR/%J.out" -eo "$RUN_DIR/%J.err")
if [[ -n "$QUEUE" ]]; then BSUB_ARGS+=(-q "$QUEUE"); fi
printf 'Run dir: %s\n' "$RUN_DIR"
printf 'Published output: %s\n' "$PUBLISH_ZARR"
printf 'Source Git HEAD: %s\n' "$SOURCE_GIT_HEAD"
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
