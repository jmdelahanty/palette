#!/usr/bin/env bash
set -euo pipefail

ZARR_PATH=""
CONFIG=""
SOURCE_TYPE="refined"
SOURCE_PATH=""
SELECTION_POLICY=""
FORCE_NEW=0

PUBLIC_CACHE_ROOT="/misc/public/palette_cache"
PUBLIC_CACHE_DIR=""
LOG_DIR=""
RUN_ID=""
RUN_LABEL=""
WORKFLOW_ID=""

CROP_QUEUE="short"
CROP_NCORES=4
CROP_MEM_GB=32
CROP_WALLTIME="1:00"

CACHE_QUEUE="short"
CACHE_NCORES=4
CACHE_MEM_GB=64
CACHE_GPUS=0
CACHE_WALLTIME="2:00"
CACHE_BATCH_SIZE=1024
CACHE_CROP_RUN=""
ROI_LIVE_ACCELERATION="cpu"
ROI_LIVE_GPU_CHUNK_FRAMES=32
SHA256=0
OVERWRITE=0
DRY_RUN=0

usage() {
  cat <<'USAGE'
Usage: submit_crop_flat_roi_cache_bsub.sh --zarr PATH [options]

Submit a two-job LSF workflow:
  1. Create/update a geometry-only crop run in the Zarr.
  2. After the crop job succeeds, build a flat_bin_v1 ROI cache on node-local
     scratch and publish the completed .bin/.json pair to shared workflow cache.

The crop stage writes canonical ROI geometry. The flat ROI cache is disposable
workflow data for downstream pose/segmentation jobs.

Required:
  --zarr PATH                       Analysis Zarr archive

Crop options:
  --config PATH                     Optional crop config YAML
  --source-type TYPE                Detection source type (default: refined; use auto to allow fallback)
  --source-path PATH                Explicit detection source path
  --selection-policy POLICY         Selection policy passed to crop_batch
  --force-new                       Force a new crop run even if a matching one exists
  --crop-queue NAME                 LSF queue for crop job (default: short)
  --crop-ncores N                   CPU slots for crop job (default: 4)
  --crop-mem-gb N                   Memory for crop job in GB (default: 32)
  --crop-walltime H:MM              Wall time for crop job (default: 1:00)

Cache options:
  --cache-crop-run NAME             Explicit crop run for cache; default: latest_any after crop job
  --public-cache-root PATH          Shared cache root (default: /misc/public/palette_cache)
  --public-cache-dir PATH           Explicit publish dir; overrides root/workflow_id/roi_cache
  --workflow-id ID                  Workflow namespace under public-cache-root
  --cache-queue NAME                LSF queue for cache job (default: short; use e.g. gpu_l4 with --cache-gpus)
  --cache-ncores N                  CPU slots for cache job (default: 4)
  --cache-mem-gb N                  Memory for cache job in GB (default: 64)
  --cache-gpus N                    GPU count for cache job; 0 omits -gpu (default: 0)
  --cache-walltime H:MM             Wall time for cache job (default: 2:00)
  --cache-batch-size N              ROI rows per cache-builder batch (default: 1024)
  --roi-live-acceleration NAME      cpu|gpu|auto for geometry-only live ROI reads (default: cpu)
  --roi-live-gpu-chunk-frames N     GPU live-read frame chunk size (default: 32)
  --sha256                          Record payload sha256 in manifest
  --overwrite                       Overwrite existing published cache files

General:
  --log-dir PATH                    Log/output directory (default: runs/diagnostics/crop_flat_roi_cache_bsub)
  --run-id ID                       Stable run id instead of UTC timestamp
  --run-label LABEL                 Output basename; default is Zarr archive stem
  --dry-run                         Print files and submit commands; do not submit
  -h, --help                        Show this message
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --zarr) ZARR_PATH="$2"; shift 2;;
    --config) CONFIG="$2"; shift 2;;
    --source-type) SOURCE_TYPE="$2"; shift 2;;
    --source-path) SOURCE_PATH="$2"; shift 2;;
    --selection-policy) SELECTION_POLICY="$2"; shift 2;;
    --force-new) FORCE_NEW=1; shift;;
    --crop-queue) CROP_QUEUE="$2"; shift 2;;
    --crop-ncores) CROP_NCORES="$2"; shift 2;;
    --crop-mem-gb) CROP_MEM_GB="$2"; shift 2;;
    --crop-walltime) CROP_WALLTIME="$2"; shift 2;;
    --cache-crop-run) CACHE_CROP_RUN="$2"; shift 2;;
    --public-cache-root) PUBLIC_CACHE_ROOT="$2"; shift 2;;
    --public-cache-dir) PUBLIC_CACHE_DIR="$2"; shift 2;;
    --workflow-id) WORKFLOW_ID="$2"; shift 2;;
    --cache-queue) CACHE_QUEUE="$2"; shift 2;;
    --cache-ncores) CACHE_NCORES="$2"; shift 2;;
    --cache-mem-gb) CACHE_MEM_GB="$2"; shift 2;;
    --cache-gpus) CACHE_GPUS="$2"; shift 2;;
    --cache-walltime) CACHE_WALLTIME="$2"; shift 2;;
    --cache-batch-size) CACHE_BATCH_SIZE="$2"; shift 2;;
    --roi-live-acceleration) ROI_LIVE_ACCELERATION="$2"; shift 2;;
    --roi-live-gpu-chunk-frames) ROI_LIVE_GPU_CHUNK_FRAMES="$2"; shift 2;;
    --sha256) SHA256=1; shift;;
    --overwrite) OVERWRITE=1; shift;;
    --log-dir) LOG_DIR="$2"; shift 2;;
    --run-id) RUN_ID="$2"; shift 2;;
    --run-label) RUN_LABEL="$2"; shift 2;;
    --dry-run) DRY_RUN=1; shift;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown arg: $1" >&2; usage; exit 2;;
  esac
done

if [[ -z "$ZARR_PATH" ]]; then
  echo "Missing required --zarr PATH" >&2
  usage
  exit 2
fi

if [[ "$DRY_RUN" != "1" && ! -e "$ZARR_PATH" ]]; then
  echo "Zarr path not found: $ZARR_PATH" >&2
  exit 2
fi

if [[ -z "$LOG_DIR" ]]; then
  LOG_DIR="runs/diagnostics/crop_flat_roi_cache_bsub"
fi
if [[ -z "$RUN_ID" ]]; then
  RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
fi
if [[ -z "$RUN_LABEL" ]]; then
  stem="$(basename "$ZARR_PATH")"
  RUN_LABEL="${stem%.zarr}"
fi

SAFE_LABEL="$(printf '%s' "$RUN_LABEL" | tr -c 'A-Za-z0-9_.-' '_')"
SAFE_RUN_ID="$(printf '%s' "$RUN_ID" | tr -c 'A-Za-z0-9_.-' '_')"
if [[ -z "$WORKFLOW_ID" ]]; then
  WORKFLOW_ID="crop_flat_roi_cache_${SAFE_RUN_ID}_${SAFE_LABEL}"
fi
SAFE_WORKFLOW_ID="$(printf '%s' "$WORKFLOW_ID" | tr -c 'A-Za-z0-9_.-' '_')"

RUN_DIR="${LOG_DIR}/crop_flat_roi_cache_${SAFE_RUN_ID}_${SAFE_LABEL}"
if [[ -e "$RUN_DIR" ]]; then
  echo "Run directory already exists: $RUN_DIR" >&2
  echo "Choose a different --run-id or --log-dir." >&2
  exit 2
fi
mkdir -p "$RUN_DIR"

if [[ -z "$PUBLIC_CACHE_DIR" ]]; then
  PUBLIC_CACHE_DIR="${PUBLIC_CACHE_ROOT}/${SAFE_WORKFLOW_ID}/roi_cache"
fi

CROP_ARGS=(
  "$ZARR_PATH"
  --apply
  --zarr-use analysis
  --crop-storage-mode geometry_only
)
if [[ -n "$CONFIG" ]]; then CROP_ARGS+=(--config "$CONFIG"); fi
if [[ -n "$SOURCE_TYPE" ]]; then CROP_ARGS+=(--source-type "$SOURCE_TYPE"); fi
if [[ -n "$SOURCE_PATH" ]]; then CROP_ARGS+=(--source-path "$SOURCE_PATH"); fi
if [[ -n "$SELECTION_POLICY" ]]; then CROP_ARGS+=(--selection-policy "$SELECTION_POLICY"); fi
if [[ "$FORCE_NEW" == "1" ]]; then CROP_ARGS+=(--force-new); fi
printf -v CROP_ARGS_SHELL '%q ' "${CROP_ARGS[@]}"

BUILDER_ARGS=(
  "$ZARR_PATH"
  --batch-size "$CACHE_BATCH_SIZE"
  --roi-live-acceleration "$ROI_LIVE_ACCELERATION"
  --roi-live-gpu-chunk-frames "$ROI_LIVE_GPU_CHUNK_FRAMES"
)
if [[ -n "$CACHE_CROP_RUN" ]]; then BUILDER_ARGS+=(--crop-run "$CACHE_CROP_RUN"); fi
if [[ "$SHA256" == "1" ]]; then BUILDER_ARGS+=(--sha256); fi
if [[ "$OVERWRITE" == "1" ]]; then BUILDER_ARGS+=(--overwrite); fi
printf -v BUILDER_ARGS_SHELL '%q ' "${BUILDER_ARGS[@]}"

CROP_SCRIPT="${RUN_DIR}/run_crop_geometry.sh"
CACHE_SCRIPT="${RUN_DIR}/run_flat_roi_cache_publish.sh"
RUN_DIR_Q="$(printf '%q' "$RUN_DIR")"
PUBLIC_CACHE_DIR_Q="$(printf '%q' "$PUBLIC_CACHE_DIR")"
SAFE_LABEL_Q="$(printf '%q' "$SAFE_LABEL")"
OVERWRITE_Q="$(printf '%q' "$OVERWRITE")"
ZARR_PATH_Q="$(printf '%q' "$ZARR_PATH")"

cat > "$CROP_SCRIPT" <<JOBSCRIPT
#!/usr/bin/env bash
set -euo pipefail

cd "$(pwd)"

RUN_DIR=${RUN_DIR_Q}
RUN_LABEL=${SAFE_LABEL_Q}
ZARR_PATH=${ZARR_PATH_Q}
JOB_ID="\${LSB_JOBID:-manual_crop}"
STATUS_JSON="\${RUN_DIR}/\${RUN_LABEL}.crop.\${JOB_ID}.json"

scratch_user="\${USER:-\$(id -un)}"
if [[ -n "\${LSB_JOBID:-}" && -d "/scratch/\${scratch_user}" ]]; then
  export PALETTE_JOB_CACHE="/scratch/\${scratch_user}/\${LSB_JOBID}/palette_cache"
else
  export PALETTE_JOB_CACHE="\${TMPDIR:-/tmp}/palette_crop_geometry_\${JOB_ID}/palette_cache"
fi
export MPLBACKEND=Agg
mkdir -p "\$PALETTE_JOB_CACHE" "\$RUN_DIR"

echo "repo=\$(pwd)"
echo "host=\$(hostname)"
echo "job_id=\$JOB_ID"
echo "zarr=\$ZARR_PATH"
echo "palette_job_cache=\$PALETTE_JOB_CACHE"
echo "status_json=\$STATUS_JSON"

scripts/py -m fisheye.utils.crop_batch ${CROP_ARGS_SHELL}

scripts/py -c 'import json, os, socket, sys; from datetime import datetime, timezone; from pathlib import Path; import zarr
zarr_path = Path(sys.argv[1]).expanduser().resolve()
status_json = Path(sys.argv[2])
root = zarr.open_group(str(zarr_path), mode="r", use_consolidated=False)
crop_parent = root.get("crop_runs")
latest_any = crop_parent.attrs.get("latest_any") if crop_parent is not None else None
latest = crop_parent.attrs.get("latest") if crop_parent is not None else None
latest_materialized = crop_parent.attrs.get("latest_materialized") if crop_parent is not None else None
crop_attrs = {}
if crop_parent is not None and latest_any in crop_parent:
    group = crop_parent[latest_any]
    crop_attrs = {
        "crop_storage_mode": group.attrs.get("crop_storage_mode"),
        "crop_signature": group.attrs.get("crop_signature"),
        "detection_source_type": group.attrs.get("detection_source_type"),
        "detection_source_path": group.attrs.get("detection_source_path"),
        "total_detections": group.attrs.get("total_detections"),
    }
payload = {
    "status": "ok",
    "stage": "crop_geometry",
    "job_id": os.environ.get("LSB_JOBID"),
    "host": socket.gethostname(),
    "finished_at_utc": datetime.now(timezone.utc).isoformat(),
    "zarr_path": str(zarr_path),
    "latest_any": latest_any,
    "latest": latest,
    "latest_materialized": latest_materialized,
    "latest_any_attrs": crop_attrs,
}
status_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(f"status_json={status_json}")
' "\$ZARR_PATH" "\$STATUS_JSON"
JOBSCRIPT
chmod +x "$CROP_SCRIPT"

cat > "$CACHE_SCRIPT" <<JOBSCRIPT
#!/usr/bin/env bash
set -euo pipefail

cd "$(pwd)"

RUN_DIR=${RUN_DIR_Q}
PUBLIC_CACHE_DIR=${PUBLIC_CACHE_DIR_Q}
RUN_LABEL=${SAFE_LABEL_Q}
OVERWRITE=${OVERWRITE_Q}
JOB_ID="\${LSB_JOBID:-manual_cache}"
HOST="\$(hostname)"
STATUS_JSON="\${RUN_DIR}/\${RUN_LABEL}.cache.\${JOB_ID}.json"
PROGRESS_JSONL="\${RUN_DIR}/\${RUN_LABEL}.cache.\${JOB_ID}.progress.jsonl"

scratch_user="\${USER:-\$(id -un)}"
if [[ -n "\${LSB_JOBID:-}" && -d "/scratch/\${scratch_user}" ]]; then
  SCRATCH_ROOT="/scratch/\${scratch_user}/\${LSB_JOBID}"
else
  SCRATCH_ROOT="\${TMPDIR:-/tmp}/palette_flat_roi_cache_\${JOB_ID}"
fi
export PALETTE_JOB_CACHE="\${SCRATCH_ROOT}/palette_cache"
export MPLBACKEND=Agg

LOCAL_CACHE_DIR="\${PALETTE_JOB_CACHE}/flat_roi_cache"
LOCAL_MANIFEST="\${LOCAL_CACHE_DIR}/\${RUN_LABEL}.flat_roi_cache.json"
LOCAL_BIN="\${LOCAL_MANIFEST%.json}.bin"
FINAL_MANIFEST="\${PUBLIC_CACHE_DIR}/\$(basename "\${LOCAL_MANIFEST}")"
FINAL_BIN="\${PUBLIC_CACHE_DIR}/\$(basename "\${LOCAL_BIN}")"
TMP_BIN="\${FINAL_BIN}.tmp.\${JOB_ID}.\${HOST}"
TMP_MANIFEST="\${FINAL_MANIFEST}.tmp.\${JOB_ID}.\${HOST}"

cleanup_tmp() {
  rm -f "\${TMP_BIN}" "\${TMP_MANIFEST}"
}
trap cleanup_tmp EXIT

mkdir -p "\${LOCAL_CACHE_DIR}" "\${RUN_DIR}" "\${PUBLIC_CACHE_DIR}"

echo "repo=\$(pwd)"
echo "host=\${HOST}"
echo "job_id=\${JOB_ID}"
echo "palette_job_cache=\${PALETTE_JOB_CACHE}"
echo "local_manifest=\${LOCAL_MANIFEST}"
echo "public_manifest=\${FINAL_MANIFEST}"
echo "status_json=\${STATUS_JSON}"
echo "progress_jsonl=\${PROGRESS_JSONL}"

if [[ "\${OVERWRITE}" != "1" && ( -e "\${FINAL_MANIFEST}" || -e "\${FINAL_BIN}" ) ]]; then
  echo "Published cache already exists; pass --overwrite to replace:" >&2
  echo "  \${FINAL_MANIFEST}" >&2
  echo "  \${FINAL_BIN}" >&2
  exit 2
fi

scripts/py -m fisheye.utils.build_flat_roi_cache ${BUILDER_ARGS_SHELL}--manifest-path "\${LOCAL_MANIFEST}" --progress-jsonl "\${PROGRESS_JSONL}" --progress-stderr --progress-interval-s 30 --json > "\${RUN_DIR}/\${RUN_LABEL}.cache.\${JOB_ID}.manifest.build.json"

if [[ ! -s "\${LOCAL_MANIFEST}" ]]; then
  echo "Local manifest was not created: \${LOCAL_MANIFEST}" >&2
  exit 1
fi
if [[ ! -s "\${LOCAL_BIN}" ]]; then
  echo "Local binary payload was not created: \${LOCAL_BIN}" >&2
  exit 1
fi

cp "\${LOCAL_BIN}" "\${TMP_BIN}"
mv -f "\${TMP_BIN}" "\${FINAL_BIN}"

scripts/py -c 'import json, os, socket, sys; from datetime import datetime, timezone; from pathlib import Path
local_manifest = Path(sys.argv[1])
final_manifest = Path(sys.argv[2])
final_bin = Path(sys.argv[3])
tmp_manifest = Path(sys.argv[4])
payload = json.loads(local_manifest.read_text(encoding="utf-8"))
payload["manifest_path"] = str(final_manifest)
publisher = payload.setdefault("publisher", {})
publisher.update({
    "published_at_utc": datetime.now(timezone.utc).isoformat(),
    "publish_host": socket.gethostname(),
    "lsb_jobid": os.environ.get("LSB_JOBID"),
    "source_manifest_path": str(local_manifest),
    "published_manifest_path": str(final_manifest),
    "published_bin_path": str(final_bin),
    "published_bin_size_bytes": final_bin.stat().st_size,
    "publish_policy": "payload_first_manifest_last",
})
tmp_manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
' "\${LOCAL_MANIFEST}" "\${FINAL_MANIFEST}" "\${FINAL_BIN}" "\${TMP_MANIFEST}"
mv -f "\${TMP_MANIFEST}" "\${FINAL_MANIFEST}"

scripts/py -c 'import sys; from fisheye.shared.flat_roi_cache import open_flat_roi_cache
cache = open_flat_roi_cache(sys.argv[1])
try:
    print(f"published_cache_validated=1 shape={cache.shape} dtype={cache.dtype} bin={cache.bin_path}")
finally:
    cache.close()
' "\${FINAL_MANIFEST}"

scripts/py -c 'import json, os, socket, sys; from datetime import datetime, timezone; from pathlib import Path
final_manifest = Path(sys.argv[1])
final_bin = Path(sys.argv[2])
status_json = Path(sys.argv[3])
manifest = json.loads(final_manifest.read_text(encoding="utf-8"))
status = {
    "status": "ok",
    "stage": "flat_roi_cache_publish",
    "schema": manifest.get("schema"),
    "layout": manifest.get("layout"),
    "job_id": os.environ.get("LSB_JOBID"),
    "host": socket.gethostname(),
    "finished_at_utc": datetime.now(timezone.utc).isoformat(),
    "published_manifest": str(final_manifest),
    "published_bin": str(final_bin),
    "published_bin_size_bytes": final_bin.stat().st_size,
    "source": manifest.get("source"),
    "array": manifest.get("array"),
}
status_json.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(f"status_json={status_json}")
' "\${FINAL_MANIFEST}" "\${FINAL_BIN}" "\${STATUS_JSON}"
JOBSCRIPT
chmod +x "$CACHE_SCRIPT"

CROP_BSUB_ARGS=(
  -J "crop_geometry_${SAFE_LABEL}"
  -n "$CROP_NCORES"
  -W "$CROP_WALLTIME"
  -R "rusage[mem=${CROP_MEM_GB}G]"
  -oo "${RUN_DIR}/crop_%J.out"
  -eo "${RUN_DIR}/crop_%J.err"
)
if [[ -n "$CROP_QUEUE" ]]; then
  CROP_BSUB_ARGS+=(-q "$CROP_QUEUE")
fi

CACHE_BSUB_ARGS_BASE=(
  -J "flat_roi_cache_${SAFE_LABEL}"
  -n "$CACHE_NCORES"
  -W "$CACHE_WALLTIME"
  -R "rusage[mem=${CACHE_MEM_GB}G]"
  -oo "${RUN_DIR}/cache_%J.out"
  -eo "${RUN_DIR}/cache_%J.err"
)
if [[ -n "$CACHE_QUEUE" ]]; then
  CACHE_BSUB_ARGS_BASE+=(-q "$CACHE_QUEUE")
fi
if [[ "$CACHE_GPUS" != "0" ]]; then
  CACHE_BSUB_ARGS_BASE+=(-gpu "num=${CACHE_GPUS}")
fi

printf -v CROP_BSUB_ARGS_SHELL '%q ' "${CROP_BSUB_ARGS[@]}"
printf -v CACHE_BSUB_ARGS_BASE_SHELL '%q ' "${CACHE_BSUB_ARGS_BASE[@]}"
CROP_CMD="bsub ${CROP_BSUB_ARGS_SHELL}bash $(printf '%q' "$CROP_SCRIPT")"
CACHE_CMD_TEMPLATE="bsub ${CACHE_BSUB_ARGS_BASE_SHELL}-w done\\(<crop_jobid>\\) bash $(printf '%q' "$CACHE_SCRIPT")"

echo "Run dir: $RUN_DIR"
echo "Crop script: $CROP_SCRIPT"
echo "Cache script: $CACHE_SCRIPT"
echo "Crop status JSON: ${RUN_DIR}/${SAFE_LABEL}.crop.<JOBID>.json"
echo "Cache status JSON: ${RUN_DIR}/${SAFE_LABEL}.cache.<JOBID>.json"
echo "Public cache dir: $PUBLIC_CACHE_DIR"
echo "Expected manifest: ${PUBLIC_CACHE_DIR}/${SAFE_LABEL}.flat_roi_cache.json"
echo "Expected payload: ${PUBLIC_CACHE_DIR}/${SAFE_LABEL}.flat_roi_cache.bin"
echo "Crop command: scripts/py -m fisheye.utils.crop_batch ${CROP_ARGS_SHELL}"
echo "Cache builder command: scripts/py -m fisheye.utils.build_flat_roi_cache ${BUILDER_ARGS_SHELL}--manifest-path <scratch manifest> --json"
echo "Submit crop command: $CROP_CMD"
echo "Submit cache command template: $CACHE_CMD_TEMPLATE"

if [[ "$DRY_RUN" == "1" ]]; then
  echo "Dry run only; no submission."
  exit 0
fi

if ! command -v bsub >/dev/null 2>&1; then
  echo "bsub not found in PATH. Is this an LSF cluster?" >&2
  exit 2
fi

crop_submit_output="$(bsub "${CROP_BSUB_ARGS[@]}" bash "$CROP_SCRIPT")"
echo "$crop_submit_output"
crop_jobid="$(printf '%s\n' "$crop_submit_output" | sed -n 's/.*Job <\([0-9][0-9]*\)>.*/\1/p' | head -1)"
if [[ -z "$crop_jobid" ]]; then
  echo "Could not parse crop job id from bsub output." >&2
  exit 1
fi

CACHE_BSUB_ARGS=("${CACHE_BSUB_ARGS_BASE[@]}" -w "done(${crop_jobid})")
cache_submit_output="$(bsub "${CACHE_BSUB_ARGS[@]}" bash "$CACHE_SCRIPT")"
echo "$cache_submit_output"
cache_jobid="$(printf '%s\n' "$cache_submit_output" | sed -n 's/.*Job <\([0-9][0-9]*\)>.*/\1/p' | head -1)"

echo "crop_jobid=${crop_jobid}"
if [[ -n "$cache_jobid" ]]; then
  echo "cache_jobid=${cache_jobid}"
fi
echo "cache_dependency=done(${crop_jobid})"
