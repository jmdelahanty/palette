#!/usr/bin/env bash
set -euo pipefail

ROOT="/groups/johnson/johnsonlab/jeremy/recordings"
REGISTRY="${PALETTE_REGISTRY_PATH:-/groups/johnson/johnsonlab/jeremy/registries/palette_registry.sqlite}"
FILE_LIST=""
PATH_CONTAINS=""
LIMIT=0
GROUPS_ONLY=1
ORDER="desc"

PUBLIC_CACHE_ROOT="/nrs/johnson/palette_staging/flat_roi_cache"
PUBLIC_CACHE_DIR=""
LOG_DIR=""
RUN_ID=""
WORKFLOW_ID=""
SOURCE_TYPE="refined"
SOURCE_PATH=""
SELECTION_POLICY="full_recording"
FORCE_NEW=0
DEFER_REGISTRY=1
SUBMIT_REGISTRY_FINALIZER=1

CROP_QUEUE="short"
CROP_NCORES=4
CROP_MEM_GB=32
CROP_WALLTIME="1:00"

CACHE_QUEUE="gpu_l4"
CACHE_NCORES=4
CACHE_MEM_GB=64
CACHE_GPUS=1
CACHE_WALLTIME="2:00"
CACHE_BATCH_SIZE=1024
CACHE_DECODE_BACKEND="pynvvc_luma"
ROI_LIVE_ACCELERATION="gpu"
ROI_LIVE_GPU_CHUNK_FRAMES=32
SHA256=0
OVERWRITE=0

REGISTRY_FINALIZER_QUEUE="short"
REGISTRY_FINALIZER_NCORES=1
REGISTRY_FINALIZER_MEM_GB=16
REGISTRY_FINALIZER_WALLTIME="1:00"

DRY_RUN=0

usage() {
  cat <<'USAGE'
Usage: submit_crop_flat_roi_cache_batches_bsub.sh [options]

Submit one crop-geometry + flat-ROI-cache LSF workflow per analysis zarr. This
is a thin fan-out wrapper around scripts/submit_crop_flat_roi_cache_bsub.sh.

Discovery:
  --registry PATH             Registry sqlite path for discovery (default: $PALETTE_REGISTRY_PATH or /groups/johnson/johnsonlab/jeremy/registries/palette_registry.sqlite)
  --root PATH                 Recording root/path prefix (default: /groups/.../recordings)
  --file-list PATH            Text file with analysis zarr paths; bypasses registry discovery
  --path-contains STR         Registry zarr_path substring filter
  --limit N                   Limit discovered registry paths after ordering; 0 means no limit
  --include-non-groups        Do not require zarr_path to start with /groups/
  --order asc|desc            Registry ordering by started_utc, recording_name (default: desc)

Crop/cache:
  --workflow-id ID            Shared workflow namespace under --public-cache-root
  --public-cache-root PATH    Shared cache root (default: /nrs/johnson/palette_staging/flat_roi_cache)
  --public-cache-dir PATH     Explicit shared cache publish dir; overrides root/workflow_id/roi_cache
  --source-type TYPE          Crop detection source type (default: refined)
  --source-path PATH          Explicit detection source path
  --selection-policy POLICY   Crop selection policy (default: full_recording)
  --force-new                 Force new crop runs
  --overwrite                 Overwrite existing published cache files
  --inline-registry           Let each crop job write registry status directly
                              (not recommended for multi-recording batches)
  --no-registry-finalizer     Defer crop-job registry writes but do not submit
                              the serial registry finalizer

Resources:
  --crop-queue NAME           LSF queue for crop geometry (default: short)
  --crop-ncores N             CPU slots for crop job (default: 4)
  --crop-mem-gb N             Crop memory GB (default: 32)
  --crop-walltime H:MM        Crop wall time (default: 1:00)
  --cache-queue NAME          LSF queue for cache job (default: gpu_l4)
  --cache-ncores N            CPU slots for cache job (default: 4)
  --cache-mem-gb N            Cache memory GB (default: 64)
  --cache-gpus N              GPU count for cache job (default: 1)
  --cache-walltime H:MM       Cache wall time (default: 2:00)
  --cache-batch-size N        ROI rows per cache-builder batch (default: 1024)
  --cache-decode-backend NAME auto|pynvvc_luma|read_slice (default: pynvvc_luma)
  --roi-live-acceleration N   cpu|gpu|auto for live ROI reads (default: gpu)
  --roi-live-gpu-chunk-frames N
  --sha256                    Record payload sha256 in each manifest
  --registry-finalizer-queue NAME
  --registry-finalizer-ncores N
  --registry-finalizer-mem-gb N
  --registry-finalizer-walltime H:MM

General:
  --log-dir PATH              Submission log dir (default: <root>/logs/crop_flat_roi_cache_bsub)
  --run-id ID                 Stable run id; default UTC timestamp
  --dry-run                   Generate plans/commands; do not call bsub
  -h, --help                  Show this message

Example:
  scripts/submit_crop_flat_roi_cache_batches_bsub.sh \
    --registry /groups/johnson/johnsonlab/jeremy/registries/palette_registry.sqlite \
    --path-contains GoodCopBadCop \
    --limit 12 \
    --workflow-id goodcopbadcop_crop_cache_20260617 \
    --dry-run
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --root) ROOT="$2"; shift 2;;
    --registry) REGISTRY="$2"; shift 2;;
    --file-list) FILE_LIST="$2"; shift 2;;
    --path-contains) PATH_CONTAINS="$2"; shift 2;;
    --limit) LIMIT="$2"; shift 2;;
    --include-non-groups) GROUPS_ONLY=0; shift;;
    --order) ORDER="$2"; shift 2;;
    --workflow-id) WORKFLOW_ID="$2"; shift 2;;
    --public-cache-root) PUBLIC_CACHE_ROOT="$2"; shift 2;;
    --public-cache-dir) PUBLIC_CACHE_DIR="$2"; shift 2;;
    --source-type) SOURCE_TYPE="$2"; shift 2;;
    --source-path) SOURCE_PATH="$2"; shift 2;;
    --selection-policy) SELECTION_POLICY="$2"; shift 2;;
    --force-new) FORCE_NEW=1; shift;;
    --inline-registry) DEFER_REGISTRY=0; SUBMIT_REGISTRY_FINALIZER=0; shift;;
    --no-registry-finalizer) SUBMIT_REGISTRY_FINALIZER=0; shift;;
    --crop-queue) CROP_QUEUE="$2"; shift 2;;
    --crop-ncores) CROP_NCORES="$2"; shift 2;;
    --crop-mem-gb) CROP_MEM_GB="$2"; shift 2;;
    --crop-walltime) CROP_WALLTIME="$2"; shift 2;;
    --cache-queue) CACHE_QUEUE="$2"; shift 2;;
    --cache-ncores) CACHE_NCORES="$2"; shift 2;;
    --cache-mem-gb) CACHE_MEM_GB="$2"; shift 2;;
    --cache-gpus) CACHE_GPUS="$2"; shift 2;;
    --cache-walltime) CACHE_WALLTIME="$2"; shift 2;;
    --cache-batch-size) CACHE_BATCH_SIZE="$2"; shift 2;;
    --cache-decode-backend) CACHE_DECODE_BACKEND="$2"; shift 2;;
    --roi-live-acceleration) ROI_LIVE_ACCELERATION="$2"; shift 2;;
    --roi-live-gpu-chunk-frames) ROI_LIVE_GPU_CHUNK_FRAMES="$2"; shift 2;;
    --sha256) SHA256=1; shift;;
    --registry-finalizer-queue) REGISTRY_FINALIZER_QUEUE="$2"; shift 2;;
    --registry-finalizer-ncores) REGISTRY_FINALIZER_NCORES="$2"; shift 2;;
    --registry-finalizer-mem-gb) REGISTRY_FINALIZER_MEM_GB="$2"; shift 2;;
    --registry-finalizer-walltime) REGISTRY_FINALIZER_WALLTIME="$2"; shift 2;;
    --overwrite) OVERWRITE=1; shift;;
    --log-dir) LOG_DIR="$2"; shift 2;;
    --run-id) RUN_ID="$2"; shift 2;;
    --dry-run) DRY_RUN=1; shift;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown arg: $1" >&2; usage; exit 2;;
  esac
done

case "$ORDER" in
  asc|desc) ;;
  *) echo "--order must be asc or desc" >&2; exit 2;;
esac

reject_recordings_cache_dir() {
  local cache_dir="$1"
  local recordings_root="$2"
  local resolved_cache_dir resolved_recordings_root
  resolved_cache_dir="$(realpath -m "$cache_dir")"
  resolved_recordings_root="$(realpath -m "$recordings_root")"
  if [[ "$resolved_cache_dir" == "$resolved_recordings_root" || "$resolved_cache_dir" == "$resolved_recordings_root"/* ]]; then
    echo "Refusing to publish disposable ROI caches under the recordings root:" >&2
    echo "  cache_dir=$resolved_cache_dir" >&2
    echo "  recordings_root=$resolved_recordings_root" >&2
    echo "Use --public-cache-root /nrs/johnson/palette_staging/flat_roi_cache or another non-recordings scratch/cache root." >&2
    exit 2
  fi
}

if [[ -z "$RUN_ID" ]]; then
  RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
fi
if [[ -z "$WORKFLOW_ID" ]]; then
  WORKFLOW_ID="crop_flat_roi_cache_${RUN_ID}"
fi
if [[ -z "$LOG_DIR" ]]; then
  LOG_DIR="${ROOT%/}/logs/crop_flat_roi_cache_bsub"
fi
if [[ -n "$PUBLIC_CACHE_DIR" ]]; then
  reject_recordings_cache_dir "$PUBLIC_CACHE_DIR" "$ROOT"
else
  reject_recordings_cache_dir "${PUBLIC_CACHE_ROOT%/}/${WORKFLOW_ID}/roi_cache" "$ROOT"
fi

RUN_ROOT="${LOG_DIR%/}/crop_flat_roi_cache_batch_${RUN_ID}"
if [[ -e "$RUN_ROOT" ]]; then
  echo "Run directory already exists: $RUN_ROOT" >&2
  echo "Choose a different --run-id or --log-dir." >&2
  exit 2
fi
mkdir -p "$RUN_ROOT"

PATHS_FILE="${RUN_ROOT}/zarr_paths.txt"

if [[ -n "$FILE_LIST" ]]; then
  if [[ ! -f "$FILE_LIST" ]]; then
    echo "File list not found: $FILE_LIST" >&2
    exit 2
  fi
  awk 'NF && $1 !~ /^#/' "$FILE_LIST" > "$PATHS_FILE"
else
  if [[ -z "$REGISTRY" ]]; then
    echo "Missing --registry PATH, or pass --file-list." >&2
    exit 2
  fi
  if [[ ! -f "$REGISTRY" ]]; then
    echo "Registry not found: $REGISTRY" >&2
    exit 2
  fi
  scripts/py - "$REGISTRY" "$ROOT" "$PATH_CONTAINS" "$LIMIT" "$GROUPS_ONLY" "$ORDER" "$PATHS_FILE" <<'PY'
import sqlite3
import sys
from pathlib import Path

registry, root, path_contains, limit_s, groups_only_s, order, output = sys.argv[1:]
limit = int(limit_s)
groups_only = groups_only_s == "1"
clauses = ["d.zarr_use = 'analysis'", "COALESCE(d.status, 'active') = 'active'"]
params = []
if root:
    root_prefix = str(Path(root).expanduser())
    clauses.append("d.zarr_path LIKE ?")
    params.append(root_prefix.rstrip("/") + "/%")
if path_contains:
    clauses.append("d.zarr_path LIKE ?")
    params.append(f"%{path_contains}%")
if groups_only:
    clauses.append("d.zarr_path LIKE '/groups/%'")
order_sql = "ASC" if order == "asc" else "DESC"
limit_sql = "LIMIT ?" if limit > 0 else ""
if limit > 0:
    params.append(limit)
sql = f"""
SELECT DISTINCT d.zarr_path, COALESCE(r.started_utc, '') AS started_utc, COALESCE(r.recording_name, d.zarr_path) AS recording_name
FROM datasets d
LEFT JOIN recordings r ON r.recording_id = d.recording_id
WHERE {' AND '.join(clauses)}
ORDER BY started_utc {order_sql}, recording_name ASC
{limit_sql};
"""
with sqlite3.connect(registry) as conn:
    rows = [row[0] for row in conn.execute(sql, params)]
Path(output).write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
PY
fi

target_count="$(wc -l < "$PATHS_FILE" | tr -d ' ')"
if [[ "$target_count" == "0" ]]; then
  echo "No zarr targets discovered."
  exit 0
fi

COMMANDS_FILE="${RUN_ROOT}/submit_commands.sh"
{
  echo "#!/usr/bin/env bash"
  echo "set -euo pipefail"
} > "$COMMANDS_FILE"
chmod +x "$COMMANDS_FILE"

echo "Run root: $RUN_ROOT"
echo "Targets: $target_count"
echo "Workflow id: $WORKFLOW_ID"
echo "Public cache root: $PUBLIC_CACHE_ROOT"
if [[ -n "$PUBLIC_CACHE_DIR" ]]; then
  echo "Public cache dir: $PUBLIC_CACHE_DIR"
else
  echo "Public cache dir: ${PUBLIC_CACHE_ROOT%/}/${WORKFLOW_ID}/roi_cache"
fi
echo "Target list: $PATHS_FILE"
echo "Defer registry writes: $DEFER_REGISTRY"
if [[ "$DEFER_REGISTRY" == "1" ]]; then
  echo "Submit registry finalizer: $SUBMIT_REGISTRY_FINALIZER"
  echo "Registry path: $REGISTRY"
fi

submitted=0
cache_jobids=()
while IFS= read -r zarr_path; do
  [[ -z "$zarr_path" ]] && continue
  stem="$(basename "$zarr_path")"
  run_label="${stem%.zarr}"
  args=(
    scripts/submit_crop_flat_roi_cache_bsub.sh
    --zarr "$zarr_path"
    --run-id "$RUN_ID"
    --run-label "$run_label"
    --workflow-id "$WORKFLOW_ID"
    --public-cache-root "$PUBLIC_CACHE_ROOT"
    --log-dir "${RUN_ROOT}/per_recording"
    --source-type "$SOURCE_TYPE"
    --selection-policy "$SELECTION_POLICY"
    --crop-queue "$CROP_QUEUE"
    --crop-ncores "$CROP_NCORES"
    --crop-mem-gb "$CROP_MEM_GB"
    --crop-walltime "$CROP_WALLTIME"
    --cache-queue "$CACHE_QUEUE"
    --cache-ncores "$CACHE_NCORES"
    --cache-mem-gb "$CACHE_MEM_GB"
    --cache-gpus "$CACHE_GPUS"
    --cache-walltime "$CACHE_WALLTIME"
    --cache-batch-size "$CACHE_BATCH_SIZE"
    --cache-decode-backend "$CACHE_DECODE_BACKEND"
    --roi-live-acceleration "$ROI_LIVE_ACCELERATION"
    --roi-live-gpu-chunk-frames "$ROI_LIVE_GPU_CHUNK_FRAMES"
  )
  if [[ -n "$PUBLIC_CACHE_DIR" ]]; then args+=(--public-cache-dir "$PUBLIC_CACHE_DIR"); fi
  if [[ -n "$SOURCE_PATH" ]]; then args+=(--source-path "$SOURCE_PATH"); fi
  if [[ "$FORCE_NEW" == "1" ]]; then args+=(--force-new); fi
  if [[ "$DEFER_REGISTRY" == "1" ]]; then args+=(--defer-registry); fi
  if [[ "$SHA256" == "1" ]]; then args+=(--sha256); fi
  if [[ "$OVERWRITE" == "1" ]]; then args+=(--overwrite); fi
  if [[ "$DRY_RUN" == "1" ]]; then args+=(--dry-run); fi

  printf '%q ' "${args[@]}" >> "$COMMANDS_FILE"
  printf '\n' >> "$COMMANDS_FILE"
  submit_output="$("${args[@]}")"
  echo "$submit_output"
  if [[ "$DRY_RUN" != "1" ]]; then
    cache_jobid="$(printf '%s\n' "$submit_output" | sed -n 's/^cache_jobid=//p' | tail -1)"
    if [[ -n "$cache_jobid" ]]; then
      cache_jobids+=("$cache_jobid")
    else
      echo "Warning: could not parse cache job id for $zarr_path" >&2
    fi
  fi
  submitted=$((submitted + 1))
done < "$PATHS_FILE"

echo "Submission commands: $COMMANDS_FILE"
if [[ "$DRY_RUN" == "1" ]]; then
  echo "Dry run only; no bsub jobs submitted."
else
  echo "Submitted crop/cache workflows: $submitted"
fi

if [[ "$DRY_RUN" != "1" && "$DEFER_REGISTRY" == "1" && "$SUBMIT_REGISTRY_FINALIZER" == "1" ]]; then
  if [[ "${#cache_jobids[@]}" == "0" ]]; then
    echo "No cache job ids were parsed; not submitting registry finalizer." >&2
    exit 1
  fi

  FINALIZER_SCRIPT="${RUN_ROOT}/run_registry_finalizer.sh"
  RUN_ROOT_Q="$(printf '%q' "$RUN_ROOT")"
  REGISTRY_Q="$(printf '%q' "$REGISTRY")"

  cat > "$FINALIZER_SCRIPT" <<JOBSCRIPT
#!/usr/bin/env bash
set -euo pipefail

cd "$(pwd)"

RUN_ROOT=${RUN_ROOT_Q}
REGISTRY=${REGISTRY_Q}
JOB_ID="\${LSB_JOBID:-manual_registry_finalizer}"
REPORT_JSON="\${RUN_ROOT}/registry_finalizer.\${JOB_ID}.json"

echo "repo=\$(pwd)"
echo "host=\$(hostname)"
echo "job_id=\$JOB_ID"
echo "run_root=\$RUN_ROOT"
echo "registry=\$REGISTRY"
echo "report_json=\$REPORT_JSON"

scripts/py -m fisheye.utils.finalize_crop_flat_roi_cache_batch_registry \\
  "\$RUN_ROOT" \\
  --registry "\$REGISTRY" \\
  --apply \\
  --output-json "\$REPORT_JSON"
JOBSCRIPT
  chmod +x "$FINALIZER_SCRIPT"

  dep_expr=""
  for jobid in "${cache_jobids[@]}"; do
    if [[ -n "$dep_expr" ]]; then
      dep_expr+="&&"
    fi
    dep_expr+="done(${jobid})"
  done

  FINALIZER_BSUB_ARGS=(
    -J "crop_cache_registry_finalizer_${RUN_ID}"
    -n "$REGISTRY_FINALIZER_NCORES"
    -W "$REGISTRY_FINALIZER_WALLTIME"
    -R "rusage[mem=${REGISTRY_FINALIZER_MEM_GB}G]"
    -oo "${RUN_ROOT}/registry_finalizer_%J.out"
    -eo "${RUN_ROOT}/registry_finalizer_%J.err"
    -w "$dep_expr"
  )
  if [[ -n "$REGISTRY_FINALIZER_QUEUE" ]]; then
    FINALIZER_BSUB_ARGS+=(-q "$REGISTRY_FINALIZER_QUEUE")
  fi

  printf -v FINALIZER_BSUB_ARGS_SHELL '%q ' "${FINALIZER_BSUB_ARGS[@]}"
  echo "bsub ${FINALIZER_BSUB_ARGS_SHELL}bash $(printf '%q' "$FINALIZER_SCRIPT")" >> "$COMMANDS_FILE"
  echo "Registry finalizer script: $FINALIZER_SCRIPT"
  echo "Registry finalizer dependency: $dep_expr"

  finalizer_submit_output="$(bsub "${FINALIZER_BSUB_ARGS[@]}" bash "$FINALIZER_SCRIPT")"
  echo "$finalizer_submit_output"
  finalizer_jobid="$(printf '%s\n' "$finalizer_submit_output" | sed -n 's/.*Job <\([0-9][0-9]*\)>.*/\1/p' | head -1)"
  if [[ -n "$finalizer_jobid" ]]; then
    echo "registry_finalizer_jobid=${finalizer_jobid}"
  fi
elif [[ "$DRY_RUN" != "1" && "$DEFER_REGISTRY" == "1" ]]; then
  echo "Registry writes were deferred. Run this finalizer manually after cache jobs complete:"
  echo "  scripts/py -m fisheye.utils.finalize_crop_flat_roi_cache_batch_registry $(printf '%q' "$RUN_ROOT") --registry $(printf '%q' "$REGISTRY") --apply"
fi
