#!/usr/bin/env bash
set -euo pipefail

ZARR=""
SOURCE_REFINED_RUN=""
REFERENCE_ZARR=""
REFERENCE_RUN=""
RUN_ID="subject_mask_full_collection_canary_$(date +%Y%m%d_%H%M%S)"
LOG_ROOT="/groups/johnson/johnsonlab/jeremy/recordings/logs/subject_mask_full_collection_canary"
QUEUE=""
SUBMIT_HOST="${PALETTE_LSF_SUBMIT_HOST:-}"
NCORES=16
MEM_GB_PER_SLOT=3
WALLTIME="4:00"
CHUNK_SIZE=256
DENSE_MASK_ROW_CHUNK=256
EXPECTED_ROWS=1169010
EXPECTED_SHARDS=22
REFERENCE_ROWS=54000
DRY_RUN=0
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

usage() {
  cat <<'USAGE'
Usage: submit_subject_mask_full_collection_canary_bsub.sh [options]

Run one benchmark-only full subject-mask shard-collection finalizer. Source
lineage is recovered from an existing refined run's zarr.json. The finalizer
reads the canonical crop/keypoint/raw-shard inputs, writes only to node-local
scratch, publishes atomically into an isolated PRFS Zarr, and never updates the
recording Zarr or registry.

Required:
  --zarr PATH                 Source analysis Zarr containing the 22 shards.
  --source-refined-run RUN    Historical refined run carrying collection lineage.
  --reference-zarr PATH       Corrected single-clip benchmark output Zarr.
  --reference-run RUN         Reference refined run within that Zarr.

Options:
  --run-id ID                 Unique canary/log directory name.
  --log-root PATH             PRFS log root.
  --queue NAME                LSF queue.
  --submit-host HOST          SSH host used when bsub is unavailable locally.
  --ncores N                  LSF slots and workers (default: 16).
  --mem-gb-per-slot N         LSF memory request per slot (default: 3).
  --walltime H:MM             LSF walltime (default: 4:00).
  --chunk-size N              Logical worker rows (default: 256).
  --dense-mask-row-chunk N    Dense masks_roi rows (default: 256).
  --expected-rows N           Required collection rows (default: 1169010).
  --expected-shards N         Required shard count (default: 22).
  --reference-rows N          Clip-prefix rows to compare (default: 54000).
  --dry-run                   Package the job without submitting it.
  -h, --help                  Show this help.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --zarr) ZARR="$2"; shift 2;;
    --source-refined-run) SOURCE_REFINED_RUN="$2"; shift 2;;
    --reference-zarr) REFERENCE_ZARR="$2"; shift 2;;
    --reference-run) REFERENCE_RUN="$2"; shift 2;;
    --run-id) RUN_ID="$2"; shift 2;;
    --log-root) LOG_ROOT="$2"; shift 2;;
    --queue) QUEUE="$2"; shift 2;;
    --submit-host) SUBMIT_HOST="$2"; shift 2;;
    --ncores) NCORES="$2"; shift 2;;
    --mem-gb-per-slot) MEM_GB_PER_SLOT="$2"; shift 2;;
    --walltime) WALLTIME="$2"; shift 2;;
    --chunk-size) CHUNK_SIZE="$2"; shift 2;;
    --dense-mask-row-chunk) DENSE_MASK_ROW_CHUNK="$2"; shift 2;;
    --expected-rows) EXPECTED_ROWS="$2"; shift 2;;
    --expected-shards) EXPECTED_SHARDS="$2"; shift 2;;
    --reference-rows) REFERENCE_ROWS="$2"; shift 2;;
    --dry-run) DRY_RUN=1; shift;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown argument: $1" >&2; usage; exit 2;;
  esac
done

for required in ZARR SOURCE_REFINED_RUN REFERENCE_ZARR REFERENCE_RUN; do
  if [[ -z "${!required}" ]]; then
    echo "--$(printf '%s' "$required" | tr 'A-Z_' 'a-z-') is required." >&2
    usage
    exit 2
  fi
done
ZARR="${ZARR%/}"
REFERENCE_ZARR="${REFERENCE_ZARR%/}"
SOURCE_METADATA="$ZARR/refined_subject_masks_runs/$SOURCE_REFINED_RUN/zarr.json"
REFERENCE_METADATA="$REFERENCE_ZARR/refined_subject_masks_runs/$REFERENCE_RUN/zarr.json"
for path in "$ZARR/zarr.json" "$SOURCE_METADATA" "$REFERENCE_ZARR/zarr.json" "$REFERENCE_METADATA"; do
  if [[ ! -f "$path" ]]; then
    echo "Required Zarr metadata not found: $path" >&2
    exit 2
  fi
done
for value in "$NCORES" "$MEM_GB_PER_SLOT" "$CHUNK_SIZE" "$DENSE_MASK_ROW_CHUNK" "$EXPECTED_ROWS" "$EXPECTED_SHARDS" "$REFERENCE_ROWS"; do
  if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
    echo "Expected a positive integer, got: $value" >&2
    exit 2
  fi
done
if (( CHUNK_SIZE % DENSE_MASK_ROW_CHUNK != 0 )); then
  echo "--chunk-size must be a multiple of --dense-mask-row-chunk." >&2
  exit 2
fi

TARGET_CROP_RUN="$(jq -er '.attributes.source_crop_run' "$SOURCE_METADATA")"
ASSIGNMENT_KEYPOINT_GROUP="$(jq -er '.attributes.assignment_keypoint_group' "$SOURCE_METADATA")"
ASSIGNMENT_KEYPOINTS_RUN="$(jq -er '.attributes.assignment_keypoints_run' "$SOURCE_METADATA")"
mapfile -t SHARD_RUNS < <(jq -er '.attributes.source_subject_mask_shard_runs[]' "$SOURCE_METADATA")
if (( ${#SHARD_RUNS[@]} != EXPECTED_SHARDS )); then
  echo "Expected $EXPECTED_SHARDS shard runs, found ${#SHARD_RUNS[@]}." >&2
  exit 2
fi
for run in "${SHARD_RUNS[@]}"; do
  if [[ ! -f "$ZARR/subject_mask_shard_runs/$run/zarr.json" ]]; then
    echo "Source shard metadata missing: $run" >&2
    exit 2
  fi
done
if [[ ! -f "$ZARR/crop_runs/$TARGET_CROP_RUN/zarr.json" ]]; then
  echo "Target crop run missing: $TARGET_CROP_RUN" >&2
  exit 2
fi
if [[ ! -f "$ZARR/$ASSIGNMENT_KEYPOINT_GROUP/$ASSIGNMENT_KEYPOINTS_RUN/zarr.json" ]]; then
  echo "Assignment keypoint run missing: $ASSIGNMENT_KEYPOINT_GROUP/$ASSIGNMENT_KEYPOINTS_RUN" >&2
  exit 2
fi

RUN_DIR="${LOG_ROOT%/}/$RUN_ID"
if [[ -e "$RUN_DIR" ]]; then
  echo "Run directory already exists: $RUN_DIR" >&2
  exit 2
fi
mkdir -p "$RUN_DIR/source_snapshot" "$RUN_DIR/reports"
SOURCE_GIT_HEAD="$(git -C "$REPO_ROOT" rev-parse HEAD)"
SOURCE_GIT_BRANCH="$(git -C "$REPO_ROOT" branch --show-current)"
RUN_TAG="$(printf '%s' "$RUN_ID" | tr -c 'A-Za-z0-9_-' '_')"
CANARY_RUN="refined_subject_masks_${RUN_TAG}"
PUBLISH_ZARR="$RUN_DIR/published_output.zarr"
git -C "$REPO_ROOT" archive --format=tar "$SOURCE_GIT_HEAD" src/fisheye \
  | tar --extract --directory "$RUN_DIR/source_snapshot" --strip-components=1 \
      --no-same-owner --no-same-permissions
git -C "$REPO_ROOT" bundle create "$RUN_DIR/source_git.bundle" HEAD
cp --no-preserve=mode,ownership,timestamps "$REPO_ROOT/scripts/py" "$RUN_DIR/palette_py"
chmod +x "$RUN_DIR/palette_py"
printf '%s\n' "${SHARD_RUNS[@]}" | jq -R . | jq -s '{shard_runs: .}' > "$RUN_DIR/shard_runs.json"

cat > "$RUN_DIR/settings.sh" <<SETTINGS
ZARR=$(printf '%q' "$ZARR")
SOURCE_REFINED_RUN=$(printf '%q' "$SOURCE_REFINED_RUN")
REFERENCE_ZARR=$(printf '%q' "$REFERENCE_ZARR")
REFERENCE_RUN=$(printf '%q' "$REFERENCE_RUN")
TARGET_CROP_RUN=$(printf '%q' "$TARGET_CROP_RUN")
ASSIGNMENT_KEYPOINT_GROUP=$(printf '%q' "$ASSIGNMENT_KEYPOINT_GROUP")
ASSIGNMENT_KEYPOINTS_RUN=$(printf '%q' "$ASSIGNMENT_KEYPOINTS_RUN")
CANARY_RUN=$(printf '%q' "$CANARY_RUN")
PUBLISH_ZARR=$(printf '%q' "$PUBLISH_ZARR")
CHUNK_SIZE=$CHUNK_SIZE
DENSE_MASK_ROW_CHUNK=$DENSE_MASK_ROW_CHUNK
EXPECTED_ROWS=$EXPECTED_ROWS
EXPECTED_SHARDS=$EXPECTED_SHARDS
REFERENCE_ROWS=$REFERENCE_ROWS
SOURCE_GIT_HEAD=$(printf '%q' "$SOURCE_GIT_HEAD")
SETTINGS

cat > "$RUN_DIR/manifest.json" <<JSON
{
  "schema_id": "palette.subject_mask_full_collection_canary_submission.v1",
  "run_id": "$RUN_ID",
  "source_zarr": "$ZARR",
  "source_refined_lineage_run": "$SOURCE_REFINED_RUN",
  "target_crop_run": "$TARGET_CROP_RUN",
  "assignment_keypoint_group": "$ASSIGNMENT_KEYPOINT_GROUP",
  "assignment_keypoints_run": "$ASSIGNMENT_KEYPOINTS_RUN",
  "source_shard_count": ${#SHARD_RUNS[@]},
  "expected_rows": $EXPECTED_ROWS,
  "canary_run": "$CANARY_RUN",
  "published_output_zarr": "$PUBLISH_ZARR",
  "reference_zarr": "$REFERENCE_ZARR",
  "reference_run": "$REFERENCE_RUN",
  "reference_rows": $REFERENCE_ROWS,
  "chunk_size": $CHUNK_SIZE,
  "dense_mask_row_chunk": $DENSE_MASK_ROW_CHUNK,
  "workers": $NCORES,
  "write_eye_geometry": true,
  "write_component_contours": true,
  "write_sampled_component_contours": true,
  "mask_storage": "dense_uint8",
  "source_git_head": "$SOURCE_GIT_HEAD",
  "source_git_branch": "$SOURCE_GIT_BRANCH",
  "raw_reads": "direct_prfs_via_node_local_overlay",
  "refined_writes": "node_local_then_production_helper_prfs_publish",
  "exact_validation": "all_source_vs_published_arrays_plus_clip_prefix_reference",
  "mutates_source_zarr": false,
  "mutates_registry": false
}
JSON

JOB_SCRIPT="$RUN_DIR/run_full_collection_canary.sh"
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
  SCRATCH_ROOT="/scratch/${USER_NAME}/${JOB_ID}/subject_mask_full_collection_canary"
else
  SCRATCH_ROOT="${TMPDIR:-/tmp}/subject_mask_full_collection_canary_${JOB_ID}"
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
        "schema_id": "palette.subject_mask_full_collection_canary_outputs.v1",
        "benchmark_only": True,
        "mutates_source_zarr": False,
        "mutates_registry": False,
    }
)
root.require_group("refined_subject_masks_runs")
PY

staged_zarr="$(
  "$PALETTE_PY" - "$ZARR" "$STAGING_ROOT" full_collection <<'PY'
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

echo "host=$(hostname)"
echo "job_id=${LSB_JOBID:-}"
echo "allocated_slots=${LSB_DJOB_NUMPROC:-}"
echo "source_zarr=$ZARR"
echo "scratch_root=$SCRATCH_ROOT"
echo "published_output=$PUBLISH_ZARR"
echo "canary_run=$CANARY_RUN"
echo "source_git_head=$(git rev-parse HEAD)"
echo "canonical_target_preexisting=$(test -e "$ZARR/refined_subject_masks_runs/$CANARY_RUN" && echo true || echo false)"

cmd=("$PALETTE_PY" -m fisheye.refinement.finalize_subject_masks "$staged_zarr"
  --subject-shard-runs-file "$RUN_DIR/shard_runs.json"
  --target-crop-run "$TARGET_CROP_RUN"
  --refined-run "$CANARY_RUN"
  --components subject_body eyes_union swim_bladder
  --chunk-size "$CHUNK_SIZE"
  --dense-mask-row-chunk "$DENSE_MASK_ROW_CHUNK"
  --metric-level cheap
  --mask-storage dense_uint8
  --write-eye-geometry
  --write-component-contours
  --write-sampled-component-contours
  --postcompute-backend process_shards
  --postcompute-chunk-size "$CHUNK_SIZE"
  --postcompute-num-workers "${LSB_DJOB_NUMPROC:-16}"
  --execution-backend process_shards
  --num-workers "${LSB_DJOB_NUMPROC:-16}"
  --assignment-keypoint-group "$ASSIGNMENT_KEYPOINT_GROUP"
  --assignment-keypoints-run "$ASSIGNMENT_KEYPOINTS_RUN"
  --progress-jsonl "$RUN_DIR/reports/finalizer.progress.jsonl"
  --defer-registry-status
  --json)

"$PALETTE_PY" -m fisheye.diagnostics.run_with_resource_telemetry \
  --summary-json "$RUN_DIR/reports/finalizer.resources.json" \
  --samples-jsonl "$RUN_DIR/reports/finalizer.resources.jsonl" \
  --stdout-log "$RUN_DIR/reports/finalizer.stdout" \
  --requested-workers "${LSB_DJOB_NUMPROC:-16}" \
  --allocated-slots "${LSB_DJOB_NUMPROC:-16}" \
  --sample-interval-seconds 5 \
  -- "${cmd[@]}"

"$PALETTE_PY" - "$staged_zarr" "$CANARY_RUN" "$PUBLISH_ZARR" \
  "$RUN_DIR/reports/publication.json" <<'PY'
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
payload = {
    "schema_id": "palette.subject_mask_full_collection_canary_publish.v1",
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
    "published_storage_stats": _run_group_storage_stats(target_parent / run_name),
    "required_surface_checks": required,
    "all_required_surfaces": all(required.values()),
}
out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
if not payload["all_required_surfaces"]:
    raise SystemExit(f"Required finalized surfaces missing: {required}")
PY

echo "starting exact source/publication and clip-prefix validation"
"$PALETTE_PY" - "$staged_zarr" "$PUBLISH_ZARR" "$CANARY_RUN" \
  "$REFERENCE_ZARR" "$REFERENCE_RUN" "$EXPECTED_ROWS" "$EXPECTED_SHARDS" \
  "$REFERENCE_ROWS" "$ZARR" "$RUN_DIR/reports/validation.json" <<'PY'
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


def compare_arrays(left_arrays, right_arrays):
    mismatches = []
    if left_arrays.keys() != right_arrays.keys():
        missing_left = sorted(right_arrays.keys() - left_arrays.keys())
        missing_right = sorted(left_arrays.keys() - right_arrays.keys())
        mismatches.append(f"array path sets differ: missing_left={missing_left}, missing_right={missing_right}")
    compared_chunks = 0
    for path in sorted(left_arrays.keys() & right_arrays.keys()):
        left = left_arrays[path]
        right = right_arrays[path]
        if left.shape != right.shape or left.dtype != right.dtype:
            mismatches.append(f"{path}: shape/dtype differ")
            continue
        if left.ndim == 0:
            compared_chunks += 1
            if not equal(np.asarray(left[()]), np.asarray(right[()])):
                mismatches.append(f"{path}: scalar differs")
            continue
        step = row_step(left)
        for start in range(0, int(left.shape[0]), step):
            stop = min(int(left.shape[0]), start + step)
            compared_chunks += 1
            if not equal(np.asarray(left[start:stop]), np.asarray(right[start:stop])):
                mismatches.append(f"{path}: rows {start}:{stop} differ")
                break
    return mismatches, compared_chunks


def compare_reference_prefix(reference, candidate, row_count):
    excluded = {
        "detection_indices",
        "detection_source",
        "frame_counts",
        "frame_indices",
        "source_clip_indices",
        "source_clip_local_frame_indices",
        "source_crop_row_ids",
        "source_detect_row_index",
        "source_frame_indices",
        "source_refined_row_ids",
    }
    mismatches = []
    compared = []
    skipped = []
    for path, left in sorted(reference.items()):
        if path in excluded or "source_row_fingerprint" in path or "source_row_stale" in path:
            skipped.append(path)
            continue
        right = candidate.get(path)
        if right is None:
            mismatches.append(f"{path}: missing from collection output")
            continue
        if left.dtype != right.dtype or left.ndim != right.ndim:
            mismatches.append(f"{path}: dtype/ndim differ")
            continue
        if path.endswith("/contours/points_xy"):
            if tuple(left.shape[1:]) != tuple(right.shape[1:]) or int(right.shape[0]) < int(left.shape[0]):
                mismatches.append(f"{path}: point-buffer shape incompatible")
                continue
            same = equal(np.asarray(left[:]), np.asarray(right[: int(left.shape[0])]))
        elif left.ndim > 0 and int(left.shape[0]) == row_count:
            if tuple(left.shape[1:]) != tuple(right.shape[1:]) or int(right.shape[0]) < row_count:
                mismatches.append(f"{path}: row-prefix shape incompatible")
                continue
            step = row_step(left)
            same = True
            for start in range(0, row_count, step):
                stop = min(row_count, start + step)
                if not equal(np.asarray(left[start:stop]), np.asarray(right[start:stop])):
                    mismatches.append(f"{path}: prefix rows {start}:{stop} differ")
                    same = False
                    break
        elif left.shape == right.shape:
            same = equal(np.asarray(left[:]), np.asarray(right[:]))
        else:
            skipped.append(path)
            continue
        compared.append(path)
        if not same and not any(item.startswith(f"{path}:") for item in mismatches):
            mismatches.append(f"{path}: clip-prefix values differ")
    return mismatches, compared, skipped


staged_zarr = Path(sys.argv[1])
publish_zarr = Path(sys.argv[2])
run_name = sys.argv[3]
reference_zarr = Path(sys.argv[4])
reference_run = sys.argv[5]
expected_rows = int(sys.argv[6])
expected_shards = int(sys.argv[7])
reference_rows = int(sys.argv[8])
canonical_zarr = Path(sys.argv[9])
out_path = Path(sys.argv[10])

started = time.perf_counter()
source_root = zarr.open_group(str(staged_zarr), mode="r", use_consolidated=False)
published_root = zarr.open_group(str(publish_zarr), mode="r", use_consolidated=False)
reference_root = zarr.open_group(str(reference_zarr), mode="r", use_consolidated=False)
source_run = source_root[f"refined_subject_masks_runs/{run_name}"]
published_run = published_root[f"refined_subject_masks_runs/{run_name}"]
reference_group = reference_root[f"refined_subject_masks_runs/{reference_run}"]
source_arrays = arrays(source_run)
published_arrays = arrays(published_run)
reference_arrays = arrays(reference_group)

publication_mismatches, compared_chunks = compare_arrays(source_arrays, published_arrays)
prefix_mismatches, prefix_compared, prefix_skipped = compare_reference_prefix(
    reference_arrays,
    published_arrays,
    reference_rows,
)
timing = published_run.attrs.get("smart_finalizer_timing_summary") or {}
plan = timing.get("collection_worker_index_plan") or published_run.attrs.get("collection_worker_index_plan") or {}
checks = {
    "row_count": int(published_run["masks_roi"].shape[0]) == expected_rows,
    "shard_count": len(published_run.attrs.get("source_subject_mask_shard_runs") or []) == expected_shards,
    "collection_schema": published_run.attrs.get("collection_finalizer_schema") == "palette_subject_mask_shard_collection_finalizer_v1",
    "global_identity_map_builds": int(plan.get("global_identity_map_builds", -1)) == 1,
    "worker_identity_map_rebuilds": int(plan.get("worker_identity_map_rebuilds", -1)) == 0,
    "canonical_target_absent": not (canonical_zarr / "refined_subject_masks_runs" / run_name).exists(),
    "publication_exact": not publication_mismatches,
    "reference_prefix_exact": not prefix_mismatches,
}
payload = {
    "schema_id": "palette.subject_mask_full_collection_canary_validation.v1",
    "run_name": run_name,
    "expected_rows": expected_rows,
    "expected_shards": expected_shards,
    "array_count": len(published_arrays),
    "publication_compared_chunks": compared_chunks,
    "publication_mismatch_count": len(publication_mismatches),
    "publication_mismatches": publication_mismatches,
    "reference_run": reference_run,
    "reference_rows": reference_rows,
    "reference_compared_array_count": len(prefix_compared),
    "reference_compared_arrays": prefix_compared,
    "reference_skipped_arrays": prefix_skipped,
    "reference_mismatch_count": len(prefix_mismatches),
    "reference_mismatches": prefix_mismatches,
    "collection_worker_index_plan": plan,
    "checks": checks,
    "all_checks_pass": all(checks.values()),
    "duration_seconds": float(time.perf_counter() - started),
}
out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(payload, sort_keys=True))
if not payload["all_checks_pass"]:
    raise SystemExit(1)
PY

"$PALETTE_PY" - "$RUN_DIR" "$CANARY_RUN" "$ZARR" <<'PY'
import json
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
payload = {
    "schema_id": "palette.subject_mask_full_collection_canary_summary.v1",
    "run_name": sys.argv[2],
    "source_zarr": sys.argv[3],
    "resource_telemetry": json.loads((run_dir / "reports/finalizer.resources.json").read_text()),
    "publication": json.loads((run_dir / "reports/publication.json").read_text()),
    "validation": json.loads((run_dir / "reports/validation.json").read_text()),
}
(run_dir / "reports/summary.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY

echo "canonical_target_postexisting=$(test -e "$ZARR/refined_subject_masks_runs/$CANARY_RUN" && echo true || echo false)"
echo "full collection canary complete"
JOBSCRIPT
chmod +x "$JOB_SCRIPT"

BSUB_ARGS=(-J "sm_full_collection_canary" -n "$NCORES" -W "$WALLTIME" -R "rusage[mem=${MEM_GB_PER_SLOT}G]" -oo "$RUN_DIR/%J.out" -eo "$RUN_DIR/%J.err")
if [[ -n "$QUEUE" ]]; then BSUB_ARGS+=(-q "$QUEUE"); fi
printf 'Run dir: %s\n' "$RUN_DIR"
printf 'Published output: %s\n' "$PUBLISH_ZARR"
printf 'Canary run: %s\n' "$CANARY_RUN"
printf 'Source Git HEAD: %s\n' "$SOURCE_GIT_HEAD"
printf 'Shard count: %s\n' "${#SHARD_RUNS[@]}"
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
