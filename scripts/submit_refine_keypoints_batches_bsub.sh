#!/usr/bin/env bash
set -euo pipefail

ROOT="/groups/johnson/johnsonlab/jeremy/recordings"
FILE_LIST=""
PATH_CONTAINS=""
LIMIT=0
REGISTRY=""

QUEUE="short"
NCORES=4
MEM_GB=16
WALLTIME="1:00"
MAX_ACTIVE=4

KEYPOINT_RUN=""
CONFIG="configs/fisheye/default.yaml"
CHUNK_SIZE=""
SCHEDULER=""
NUM_WORKERS=""
MEMORY_LIMIT=""
AUTO_REVIEW=0
AUTO_REVIEW_REVIEWER=""
AUTO_REVIEW_NOTES=""
AUTO_REVIEW_OVERWRITE=0
OVERWRITE=0
DEFER_REGISTRY=1
SUBMIT_REGISTRY_FINALIZER=1
REGISTRY_FINALIZER_QUEUE="short"
REGISTRY_FINALIZER_NCORES=1
REGISTRY_FINALIZER_MEM_GB=8
REGISTRY_FINALIZER_WALLTIME="1:00"

LOG_DIR=""
RUN_ID=""
DRY_RUN=0
REPO_DIR=""

usage() {
  cat <<'USAGE'
Usage: submit_refine_keypoints_batches_bsub.sh [options]

Submit one refined-keypoint LSF array task per analysis zarr. Each task runs
fisheye.utils.refine_keypoints_batch on exactly one zarr, so failures are
isolated per recording while --max-active caps shared filesystem pressure.

Discovery:
  --root PATH               Recording root to scan (default: /groups/.../recordings)
  --file-list PATH          Text file with analysis zarr paths; bypasses discovery
  --path-contains STR       Keep discovered zarr paths containing this substring
  --limit N                 Limit discovered paths after sorting; 0 means no limit
  --registry PATH           Registry sqlite path exported as PALETTE_REGISTRY_PATH in each task

Refinement:
  --keypoint-run NAME       Explicit keypoints_runs child to refine (default: latest)
  --config PATH             Config passed to refine_keypoints_batch
  --chunk-size N            Refinement chunk size override
  --scheduler NAME          Dask scheduler override: processes|threads|distributed
  --num-workers N           Worker count override
  --memory-limit VALUE      Worker memory limit override
  --overwrite               Do not skip when a matching refined run already exists
  --inline-registry         Let each array task write registry status directly.
                            Default is safer deferred registry finalization.
  --no-registry-finalizer   With deferred registry writes, do not submit the
                            dependent serial registry finalizer automatically

Auto-review:
  --auto-review-full-recording
                            Apply algorithmic full-recording review after refine
  --auto-review-reviewer ID Reviewer label for auto-review payload
  --auto-review-notes TEXT  Notes for auto-review payload
  --auto-review-overwrite-existing
                            Overwrite existing keypoint_review_status

Resources:
  --queue NAME              LSF queue (default: short)
  --ncores N                CPU slots per task (default: 4)
  --mem-gb N                Memory per task in GB (default: 16)
  --walltime H:MM           Wall time per task (default: 1:00)
  --max-active N            Max concurrent array tasks (default: 4)
  --registry-finalizer-queue NAME
                            Queue for serial registry finalizer (default: short)
  --registry-finalizer-ncores N
                            CPU slots for registry finalizer (default: 1)
  --registry-finalizer-mem-gb N
                            Memory for registry finalizer in GB (default: 8)
  --registry-finalizer-walltime H:MM
                            Wall time for registry finalizer (default: 1:00)

General:
  --log-dir PATH            Submission log dir (default: <root>/logs/refine_keypoints_batch/bsub_submissions)
  --run-id ID               Stable run id; default UTC timestamp
  --repo-dir PATH           Repository checkout visible to compute nodes (default: current directory)
  --dry-run                 Generate run dir/files and print bsub command; do not submit
  -h, --help                Show this message

Example:
  scripts/submit_refine_keypoints_batches_bsub.sh \
    --file-list /tmp/goodcopbadcop_20260614_zarrs.txt \
    --keypoint-run keypoints_goodcopbadcop_kpt5_traditional_v2_flat_cache_20260617 \
    --auto-review-full-recording \
    --max-active 4 \
    --dry-run
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --root) ROOT="$2"; shift 2;;
    --file-list) FILE_LIST="$2"; shift 2;;
    --path-contains) PATH_CONTAINS="$2"; shift 2;;
    --limit) LIMIT="$2"; shift 2;;
    --registry) REGISTRY="$2"; shift 2;;
    --queue) QUEUE="$2"; shift 2;;
    --ncores) NCORES="$2"; shift 2;;
    --mem-gb) MEM_GB="$2"; shift 2;;
    --walltime) WALLTIME="$2"; shift 2;;
    --max-active) MAX_ACTIVE="$2"; shift 2;;
    --keypoint-run) KEYPOINT_RUN="$2"; shift 2;;
    --config) CONFIG="$2"; shift 2;;
    --chunk-size) CHUNK_SIZE="$2"; shift 2;;
    --scheduler) SCHEDULER="$2"; shift 2;;
    --num-workers) NUM_WORKERS="$2"; shift 2;;
    --memory-limit) MEMORY_LIMIT="$2"; shift 2;;
    --overwrite) OVERWRITE=1; shift;;
    --inline-registry) DEFER_REGISTRY=0; shift;;
    --no-registry-finalizer) SUBMIT_REGISTRY_FINALIZER=0; shift;;
    --auto-review-full-recording) AUTO_REVIEW=1; shift;;
    --auto-review-reviewer) AUTO_REVIEW_REVIEWER="$2"; shift 2;;
    --auto-review-notes) AUTO_REVIEW_NOTES="$2"; shift 2;;
    --auto-review-overwrite-existing) AUTO_REVIEW_OVERWRITE=1; shift;;
    --log-dir) LOG_DIR="$2"; shift 2;;
    --run-id) RUN_ID="$2"; shift 2;;
    --repo-dir) REPO_DIR="$2"; shift 2;;
    --registry-finalizer-queue) REGISTRY_FINALIZER_QUEUE="$2"; shift 2;;
    --registry-finalizer-ncores) REGISTRY_FINALIZER_NCORES="$2"; shift 2;;
    --registry-finalizer-mem-gb) REGISTRY_FINALIZER_MEM_GB="$2"; shift 2;;
    --registry-finalizer-walltime) REGISTRY_FINALIZER_WALLTIME="$2"; shift 2;;
    --dry-run) DRY_RUN=1; shift;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown arg: $1" >&2; usage; exit 2;;
  esac
done

if [[ -z "$RUN_ID" ]]; then
  RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
fi
if [[ -z "$LOG_DIR" ]]; then
  LOG_DIR="${ROOT%/}/logs/refine_keypoints_batch/bsub_submissions"
fi
if [[ -z "$REPO_DIR" ]]; then
  REPO_DIR="$(pwd)"
fi

RUN_DIR="${LOG_DIR%/}/refine_keypoints_${RUN_ID}"
if [[ -e "$RUN_DIR" ]]; then
  echo "Run directory already exists: $RUN_DIR" >&2
  echo "Choose a different --run-id or --log-dir." >&2
  exit 2
fi
mkdir -p "$RUN_DIR"

TARGETS_FILE="${RUN_DIR}/zarr_paths.txt"
if [[ -n "$FILE_LIST" ]]; then
  if [[ ! -f "$FILE_LIST" ]]; then
    echo "File list not found: $FILE_LIST" >&2
    exit 2
  fi
  awk 'NF && $1 !~ /^#/' "$FILE_LIST" > "$TARGETS_FILE"
else
  scripts/py - "$ROOT" "$PATH_CONTAINS" "$LIMIT" "$TARGETS_FILE" <<'PY'
import sys
from pathlib import Path

root = Path(sys.argv[1]).expanduser()
path_contains = sys.argv[2]
limit = int(sys.argv[3])
output = Path(sys.argv[4])

paths = sorted(str(path.resolve()) for path in root.glob("*/zarr/*_analysis.zarr"))
if path_contains:
    paths = [path for path in paths if path_contains in path]
if limit > 0:
    paths = paths[:limit]
output.write_text("\n".join(paths) + ("\n" if paths else ""), encoding="utf-8")
PY
fi

target_count="$(wc -l < "$TARGETS_FILE" | tr -d ' ')"
if [[ "$target_count" == "0" ]]; then
  echo "No analysis zarr targets found."
  exit 0
fi

REFINE_ARGS=(--zarr-use analysis --config "$CONFIG" --apply --log-dir "${RUN_DIR}/task_logs")
if [[ -n "$KEYPOINT_RUN" ]]; then REFINE_ARGS+=(--keypoint-run "$KEYPOINT_RUN"); fi
if [[ -n "$CHUNK_SIZE" ]]; then REFINE_ARGS+=(--chunk-size "$CHUNK_SIZE"); fi
if [[ -n "$SCHEDULER" ]]; then REFINE_ARGS+=(--scheduler "$SCHEDULER"); fi
if [[ -n "$NUM_WORKERS" ]]; then REFINE_ARGS+=(--num-workers "$NUM_WORKERS"); fi
if [[ -n "$MEMORY_LIMIT" ]]; then REFINE_ARGS+=(--memory-limit "$MEMORY_LIMIT"); fi
if [[ "$OVERWRITE" == "1" ]]; then REFINE_ARGS+=(--no-skip-existing); fi
if [[ "$AUTO_REVIEW" == "1" ]]; then REFINE_ARGS+=(--auto-review-full-recording); fi
if [[ -n "$AUTO_REVIEW_REVIEWER" ]]; then REFINE_ARGS+=(--auto-review-reviewer "$AUTO_REVIEW_REVIEWER"); fi
if [[ -n "$AUTO_REVIEW_NOTES" ]]; then REFINE_ARGS+=(--auto-review-notes "$AUTO_REVIEW_NOTES"); fi
if [[ "$AUTO_REVIEW_OVERWRITE" == "1" ]]; then REFINE_ARGS+=(--auto-review-overwrite-existing); fi
printf -v REFINE_ARGS_SHELL '%q ' "${REFINE_ARGS[@]}"

JOB_SCRIPT="${RUN_DIR}/run_refine_keypoints_task.sh"
REPO_DIR_Q="$(printf '%q' "$REPO_DIR")"
REGISTRY_Q="$(printf '%q' "$REGISTRY")"
DEFER_REGISTRY_Q="$(printf '%q' "$DEFER_REGISTRY")"
cat > "$JOB_SCRIPT" <<JOBSCRIPT
#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="\$1"
TARGETS_FILE="\${RUN_DIR}/zarr_paths.txt"
REGISTRY_PATH=${REGISTRY_Q}
DEFER_REGISTRY=${DEFER_REGISTRY_Q}
if [[ -z "\${LSB_JOBINDEX:-}" ]]; then
  echo "LSB_JOBINDEX not set; are you running under bsub array?" >&2
  exit 2
fi
zarr_path="\$(sed -n "\${LSB_JOBINDEX}p" "\$TARGETS_FILE")"
if [[ -z "\$zarr_path" ]]; then
  echo "No zarr path for array index \${LSB_JOBINDEX}" >&2
  exit 2
fi

cd ${REPO_DIR_Q}
if [[ -n "\$REGISTRY_PATH" ]]; then
  export PALETTE_REGISTRY_PATH="\$REGISTRY_PATH"
  echo "registry_path=\$PALETTE_REGISTRY_PATH"
fi
if [[ "\$DEFER_REGISTRY" == "1" ]]; then
  export PALETTE_DISABLE_REGISTRY_WRITES=1
  echo "registry_write_mode=deferred"
else
  echo "registry_write_mode=inline"
fi
echo "job_id=\${LSB_JOBID:-unknown}"
echo "job_index=\${LSB_JOBINDEX}"
echo "host=\$(hostname)"
echo "zarr_path=\$zarr_path"
echo "started_utc=\$(date -u +%Y-%m-%dT%H:%M:%SZ)"

scripts/py -m fisheye.utils.refine_keypoints_batch "\$zarr_path" ${REFINE_ARGS_SHELL}

echo "finished_utc=\$(date -u +%Y-%m-%dT%H:%M:%SZ)"
JOBSCRIPT
chmod +x "$JOB_SCRIPT"

scripts/py - "$RUN_DIR/submission_manifest.json" "$ROOT" "$FILE_LIST" "$PATH_CONTAINS" "$LIMIT" \
  "$target_count" "$QUEUE" "$NCORES" "$MEM_GB" "$WALLTIME" "$MAX_ACTIVE" "$KEYPOINT_RUN" \
  "$CONFIG" "$AUTO_REVIEW" "$OVERWRITE" "$JOB_SCRIPT" "$TARGETS_FILE" "$REPO_DIR" "$REGISTRY" \
  "$DEFER_REGISTRY" "$SUBMIT_REGISTRY_FINALIZER" "$REGISTRY_FINALIZER_QUEUE" \
  "$REGISTRY_FINALIZER_NCORES" "$REGISTRY_FINALIZER_MEM_GB" "$REGISTRY_FINALIZER_WALLTIME" <<'PY'
import json
import sys
from pathlib import Path

(
    output,
    root,
    file_list,
    path_contains,
    limit,
    target_count,
    queue,
    ncores,
    mem_gb,
    walltime,
    max_active,
    keypoint_run,
    config,
    auto_review,
    overwrite,
    job_script,
    targets_file,
    repo_dir,
    registry,
    defer_registry,
    submit_registry_finalizer,
    registry_finalizer_queue,
    registry_finalizer_ncores,
    registry_finalizer_mem_gb,
    registry_finalizer_walltime,
) = sys.argv[1:]

payload = {
    "root": root,
    "file_list": file_list or None,
    "path_contains": path_contains or None,
    "limit": int(limit),
    "target_count": int(target_count),
    "queue": queue,
    "ncores": int(ncores),
    "mem_gb": int(mem_gb),
    "walltime": walltime,
    "max_active": int(max_active),
    "keypoint_run": keypoint_run or None,
    "config": config,
    "auto_review_full_recording": auto_review == "1",
    "overwrite_existing_refined": overwrite == "1",
    "job_script": job_script,
    "targets_file": targets_file,
    "repo_dir": repo_dir,
    "registry": registry or None,
    "registry_write_mode": "deferred" if defer_registry == "1" else "inline",
    "submit_registry_finalizer": submit_registry_finalizer == "1",
    "registry_finalizer": {
        "queue": registry_finalizer_queue,
        "ncores": int(registry_finalizer_ncores),
        "mem_gb": int(registry_finalizer_mem_gb),
        "walltime": registry_finalizer_walltime,
    },
}
Path(output).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

BSUB_ARGS=(
  -J "refine_kp[1-${target_count}]%${MAX_ACTIVE}"
  -q "$QUEUE"
  -n "$NCORES"
  -W "$WALLTIME"
  -R "rusage[mem=${MEM_GB}G]"
  -oo "${RUN_DIR}/%J_%I.out"
  -eo "${RUN_DIR}/%J_%I.err"
)

BSUB_CMD="bsub"
for arg in "${BSUB_ARGS[@]}"; do
  BSUB_CMD+=" $(printf '%q' "$arg")"
done
BSUB_CMD+=" bash"
BSUB_CMD+=" $(printf '%q' "$JOB_SCRIPT")"
BSUB_CMD+=" $(printf '%q' "$RUN_DIR")"

printf -v REFINE_CMD 'scripts/py -m fisheye.utils.refine_keypoints_batch <zarr> %s' "$REFINE_ARGS_SHELL"

echo "Run dir: $RUN_DIR"
echo "Targets: $target_count"
echo "Target list: $TARGETS_FILE"
echo "Repo dir: $REPO_DIR"
echo "Registry: ${REGISTRY:-<default from environment/repo>}"
echo "Queue: $QUEUE"
echo "Resources: ncores=$NCORES mem_gb=$MEM_GB walltime=$WALLTIME"
echo "Max active: $MAX_ACTIVE"
echo "Registry write mode: $([[ "$DEFER_REGISTRY" == "1" ]] && echo deferred || echo inline)"
echo "Per-target command: $REFINE_CMD"
echo "Submit command: $BSUB_CMD"

if [[ "$DRY_RUN" == "1" ]]; then
  echo "Dry run only; no submission."
  exit 0
fi

if ! command -v bsub >/dev/null 2>&1; then
  echo "bsub not found in PATH. Is this an LSF cluster?" >&2
  exit 2
fi

submit_output="$(bsub "${BSUB_ARGS[@]}" bash "$JOB_SCRIPT" "$RUN_DIR")"
echo "$submit_output"
array_jobid="$(printf '%s\n' "$submit_output" | sed -n 's/.*Job <\([0-9][0-9]*\)>.*/\1/p' | head -1)"
if [[ -n "$array_jobid" ]]; then
  echo "array_jobid=${array_jobid}"
fi

if [[ "$DEFER_REGISTRY" == "1" && "$SUBMIT_REGISTRY_FINALIZER" == "1" ]]; then
  if [[ -z "$array_jobid" ]]; then
    echo "Could not parse array job id; not submitting registry finalizer." >&2
    exit 1
  fi

  FINALIZER_SCRIPT="${RUN_DIR}/run_registry_finalizer.sh"
  RUN_DIR_Q="$(printf '%q' "$RUN_DIR")"
  REGISTRY_Q="$(printf '%q' "$REGISTRY")"
  KEYPOINT_RUN_Q="$(printf '%q' "$KEYPOINT_RUN")"
  cat > "$FINALIZER_SCRIPT" <<JOBSCRIPT
#!/usr/bin/env bash
set -euo pipefail

RUN_DIR=${RUN_DIR_Q}
REGISTRY_PATH=${REGISTRY_Q}
KEYPOINT_RUN=${KEYPOINT_RUN_Q}
JOB_ID="\${LSB_JOBID:-manual_refine_keypoints_registry_finalizer}"
REPORT_JSON="\${RUN_DIR}/registry_finalizer.\${JOB_ID}.json"

cd ${REPO_DIR_Q}
unset PALETTE_DISABLE_REGISTRY_WRITES
if [[ -n "\$REGISTRY_PATH" ]]; then
  export PALETTE_REGISTRY_PATH="\$REGISTRY_PATH"
fi

args=(fisheye.utils.finalize_refine_keypoints_batch_registry "\$RUN_DIR" --apply --output-json "\$REPORT_JSON")
if [[ -n "\$REGISTRY_PATH" ]]; then
  args+=(--registry "\$REGISTRY_PATH")
fi
if [[ -n "\$KEYPOINT_RUN" ]]; then
  args+=(--keypoint-run "\$KEYPOINT_RUN")
fi

echo "repo=\$(pwd)"
echo "host=\$(hostname)"
echo "job_id=\$JOB_ID"
echo "run_dir=\$RUN_DIR"
echo "registry=\${REGISTRY_PATH:-<default>}"
echo "keypoint_run=\${KEYPOINT_RUN:-<latest matching>}"
echo "report_json=\$REPORT_JSON"

scripts/py -m "\${args[@]}"
JOBSCRIPT
  chmod +x "$FINALIZER_SCRIPT"

  FINALIZER_BSUB_ARGS=(
    -J "refine_kp_registry_finalizer_${RUN_ID}"
    -q "$REGISTRY_FINALIZER_QUEUE"
    -n "$REGISTRY_FINALIZER_NCORES"
    -W "$REGISTRY_FINALIZER_WALLTIME"
    -R "rusage[mem=${REGISTRY_FINALIZER_MEM_GB}G]"
    -oo "${RUN_DIR}/registry_finalizer_%J.out"
    -eo "${RUN_DIR}/registry_finalizer_%J.err"
    -w "done(${array_jobid})"
  )
  printf -v FINALIZER_BSUB_ARGS_SHELL '%q ' "${FINALIZER_BSUB_ARGS[@]}"
  echo "Registry finalizer script: $FINALIZER_SCRIPT"
  echo "Registry finalizer dependency: done(${array_jobid})"
  echo "Registry finalizer submit command: bsub ${FINALIZER_BSUB_ARGS_SHELL}bash $(printf '%q' "$FINALIZER_SCRIPT")"

  finalizer_submit_output="$(bsub "${FINALIZER_BSUB_ARGS[@]}" bash "$FINALIZER_SCRIPT")"
  echo "$finalizer_submit_output"
  finalizer_jobid="$(printf '%s\n' "$finalizer_submit_output" | sed -n 's/.*Job <\([0-9][0-9]*\)>.*/\1/p' | head -1)"
  if [[ -n "$finalizer_jobid" ]]; then
    echo "registry_finalizer_jobid=${finalizer_jobid}"
  fi
elif [[ "$DEFER_REGISTRY" == "1" ]]; then
  echo "Registry writes were deferred. Run this finalizer manually after array completion:"
  echo "  scripts/py -m fisheye.utils.finalize_refine_keypoints_batch_registry $(printf '%q' "$RUN_DIR") --registry $(printf '%q' "$REGISTRY") --keypoint-run $(printf '%q' "$KEYPOINT_RUN") --apply"
fi
