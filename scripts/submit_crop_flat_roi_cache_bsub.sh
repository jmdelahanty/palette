#!/usr/bin/env bash
set -euo pipefail

ZARR_PATH=""
CONFIG=""
SOURCE_TYPE="detect"
SOURCE_PATH=""
SELECTION_POLICY=""
FORCE_NEW=0
DEFER_REGISTRY=0

PUBLIC_CACHE_ROOT="/nrs/johnson/palette_staging/flat_roi_cache"
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
CACHE_DECODE_BACKEND="auto"
SHA256=0
OVERWRITE=0
DRY_RUN=0

usage() {
  cat <<'USAGE'
Usage: submit_crop_flat_roi_cache_bsub.sh --zarr PATH [options]

Submit a two-job LSF workflow:
  1. Create/update a future-canonical materialized crop run in the Zarr.
  2. After the crop job succeeds, build a flat_bin_v1 ROI cache on node-local
     scratch and publish the completed .bin/.json pair to shared workflow cache.

The crop stage writes canonical ROI pixels and geometry. The flat ROI cache is
disposable workflow data for downstream pose/segmentation jobs.

Required:
  --zarr PATH                       Analysis Zarr archive

Crop options:
  --config PATH                     Optional crop config YAML
  --source-type TYPE                Detection source type (default: detect)
  --source-path PATH                Explicit detection source path
  --selection-policy POLICY         Selection policy passed to crop_batch
  --force-new                       Force a new crop run even if a matching one exists
  --defer-registry                  Do not let the crop job write the central registry;
                                    use the batch registry finalizer instead
  --crop-queue NAME                 LSF queue for crop job (default: short)
  --crop-ncores N                   CPU slots for crop job (default: 4)
  --crop-mem-gb N                   Memory for crop job in GB (default: 32)
  --crop-walltime H:MM              Wall time for crop job (default: 1:00)

Cache options:
  --cache-crop-run NAME             Exact materialized crop run to validate/use; default:
                                    exact committed/reused run reported by crop_batch
  --public-cache-root PATH          Shared cache root (default: /nrs/johnson/palette_staging/flat_roi_cache)
  --public-cache-dir PATH           Explicit publish dir; overrides root/workflow_id/roi_cache
  --workflow-id ID                  Workflow namespace under public-cache-root
  --cache-queue NAME                LSF queue for cache job (default: short; use e.g. gpu_l4 with --cache-gpus)
  --cache-ncores N                  CPU slots for cache job (default: 4)
  --cache-mem-gb N                  Memory for cache job in GB (default: 64)
  --cache-gpus N                    GPU count for cache job; 0 omits -gpu (default: 0)
  --cache-walltime H:MM             Wall time for cache job (default: 2:00)
  --cache-batch-size N              ROI rows per cache-builder batch (default: 1024)
  --cache-decode-backend NAME       auto|pynvvc_luma|read_slice (default: auto; fast sequential PyNv when available)
  --roi-live-acceleration NAME      cpu|gpu|auto for ROI cache reads (default: cpu)
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
    --defer-registry) DEFER_REGISTRY=1; shift;;
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
    --cache-decode-backend) CACHE_DECODE_BACKEND="$2"; shift 2;;
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

if [[ ! -e "$ZARR_PATH" ]]; then
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

if [[ -z "$PUBLIC_CACHE_DIR" ]]; then
  PUBLIC_CACHE_DIR="${PUBLIC_CACHE_ROOT}/${SAFE_WORKFLOW_ID}/roi_cache"
fi

reject_recordings_cache_dir() {
  local cache_dir="$1"
  local recordings_root="/groups/johnson/johnsonlab/jeremy/recordings"
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
reject_recordings_cache_dir "$PUBLIC_CACHE_DIR"

mkdir -p "$RUN_DIR"

CROP_COMMON_ARGS=(
  "$ZARR_PATH"
  --zarr-use analysis
  --crop-storage-mode materialized
)
if [[ -n "$CONFIG" ]]; then CROP_COMMON_ARGS+=(--config "$CONFIG"); fi
if [[ -n "$SOURCE_TYPE" ]]; then CROP_COMMON_ARGS+=(--source-type "$SOURCE_TYPE"); fi
if [[ -n "$SOURCE_PATH" ]]; then CROP_COMMON_ARGS+=(--source-path "$SOURCE_PATH"); fi
if [[ -n "$SELECTION_POLICY" ]]; then CROP_COMMON_ARGS+=(--selection-policy "$SELECTION_POLICY"); fi
if [[ "$FORCE_NEW" == "1" ]]; then CROP_COMMON_ARGS+=(--force-new); fi
CROP_ARGS=("${CROP_COMMON_ARGS[@]}" --apply)
printf -v CROP_ARGS_SHELL '%q ' "${CROP_ARGS[@]}"

if [[ "$DRY_RUN" == "1" ]]; then
  echo "Running canonical crop preflight for dry-run planning:"
  scripts/py -m fisheye.utils.crop_batch \
    "${CROP_COMMON_ARGS[@]}" \
    --fail-on-invalid-plan
fi

BUILDER_ARGS=(
  "$ZARR_PATH"
  --batch-size "$CACHE_BATCH_SIZE"
  --decode-backend "$CACHE_DECODE_BACKEND"
  --roi-live-acceleration "$ROI_LIVE_ACCELERATION"
  --roi-live-gpu-chunk-frames "$ROI_LIVE_GPU_CHUNK_FRAMES"
)
if [[ "$SHA256" == "1" ]]; then BUILDER_ARGS+=(--sha256); fi
if [[ "$OVERWRITE" == "1" ]]; then BUILDER_ARGS+=(--overwrite); fi
printf -v BUILDER_ARGS_SHELL '%q ' "${BUILDER_ARGS[@]}"

CROP_SCRIPT="${RUN_DIR}/run_crop_materialized.sh"
CACHE_SCRIPT="${RUN_DIR}/run_flat_roi_cache_publish.sh"
RUN_DIR_Q="$(printf '%q' "$RUN_DIR")"
PUBLIC_CACHE_DIR_Q="$(printf '%q' "$PUBLIC_CACHE_DIR")"
SAFE_LABEL_Q="$(printf '%q' "$SAFE_LABEL")"
OVERWRITE_Q="$(printf '%q' "$OVERWRITE")"
DEFER_REGISTRY_Q="$(printf '%q' "$DEFER_REGISTRY")"
ZARR_PATH_Q="$(printf '%q' "$ZARR_PATH")"
CACHE_CROP_RUN_Q="$(printf '%q' "$CACHE_CROP_RUN")"

cat > "$CROP_SCRIPT" <<JOBSCRIPT
#!/usr/bin/env bash
set -euo pipefail

cd "$(pwd)"

RUN_DIR=${RUN_DIR_Q}
RUN_LABEL=${SAFE_LABEL_Q}
DEFER_REGISTRY=${DEFER_REGISTRY_Q}
ZARR_PATH=${ZARR_PATH_Q}
CACHE_CROP_RUN=${CACHE_CROP_RUN_Q}
JOB_ID="\${LSB_JOBID:-manual_crop}"
STATUS_JSON="\${RUN_DIR}/\${RUN_LABEL}.crop.\${JOB_ID}.json"
CROP_RESULT_JSON="\${RUN_DIR}/\${RUN_LABEL}.crop.\${JOB_ID}.result.json"

scratch_user="\${USER:-\$(id -un)}"
if [[ -n "\${LSB_JOBID:-}" && -d "/scratch/\${scratch_user}" && -w "/scratch/\${scratch_user}" && -x "/scratch/\${scratch_user}" ]]; then
  export PALETTE_JOB_CACHE="/scratch/\${scratch_user}/\${LSB_JOBID}/palette_cache"
else
  export PALETTE_JOB_CACHE="\${TMPDIR:-/tmp}/palette_crop_materialized_\${JOB_ID}/palette_cache"
fi
export MPLBACKEND=Agg
if [[ "\${DEFER_REGISTRY}" == "1" ]]; then
  export PALETTE_DISABLE_REGISTRY_WRITES=1
fi
mkdir -p "\$PALETTE_JOB_CACHE" "\$RUN_DIR"

echo "repo=\$(pwd)"
echo "host=\$(hostname)"
echo "job_id=\$JOB_ID"
echo "zarr=\$ZARR_PATH"
echo "palette_job_cache=\$PALETTE_JOB_CACHE"
echo "defer_registry=\$DEFER_REGISTRY"
echo "status_json=\$STATUS_JSON"
echo "crop_result_json=\$CROP_RESULT_JSON"

scripts/py -m fisheye.utils.crop_batch ${CROP_ARGS_SHELL}--result-json "\$CROP_RESULT_JSON"

scripts/py -c 'import json, os, socket, sys; from datetime import datetime, timezone; from pathlib import Path; import zarr
from fisheye.shared.observation_coordinate_publication import load_persisted_ordinary_crop_observation_geometry
zarr_path = Path(sys.argv[1]).expanduser().resolve()
status_json = Path(sys.argv[2])
requested_crop_run = sys.argv[3].strip()
crop_result_json = Path(sys.argv[4])
root = zarr.open_group(str(zarr_path), mode="r", use_consolidated=False)
crop_parent = root.get("crop_runs")
batch_result = json.loads(crop_result_json.read_text(encoding="utf-8"))
if batch_result.get("schema") != "palette.crop_batch_result.v1":
    raise RuntimeError("Unsupported or missing crop_batch result schema")
outcomes = batch_result.get("outcomes")
if not isinstance(outcomes, list) or len(outcomes) != 1:
    raise RuntimeError(f"Expected one exact crop_batch outcome; found {outcomes!r}")
outcome = outcomes[0]
if not isinstance(outcome, dict):
    raise RuntimeError("Crop batch outcome is not a mapping")
outcome_path = Path(str(outcome.get("zarr_path", ""))).expanduser().resolve()
if outcome_path != zarr_path or outcome.get("status") not in {"ok", "skipped"}:
    raise RuntimeError(f"Crop batch outcome does not authorize cache work: {outcome!r}")
if requested_crop_run:
    crop_run = requested_crop_run
    crop_run_selection = "explicit"
else:
    crop_run = str(outcome.get("crop_run") or "").strip()
    crop_run_selection = "crop_batch_result"
if crop_parent is None or not crop_run or crop_run not in crop_parent:
    raise RuntimeError(f"Exact materialized crop run is unavailable: {crop_run!r}")
load_persisted_ordinary_crop_observation_geometry(root, f"crop_runs/{crop_run}")
group = crop_parent[crop_run]
crop_attrs = {
    "crop_storage_mode": group.attrs.get("crop_storage_mode"),
    "crop_signature": group.attrs.get("crop_signature"),
    "detection_source_type": group.attrs.get("detection_source_type"),
    "detection_source_path": group.attrs.get("detection_source_path"),
    "total_detections": group.attrs.get("total_detections"),
}
payload = {
    "status": "ok",
    "stage": "crop_materialized",
    "job_id": os.environ.get("LSB_JOBID"),
    "host": socket.gethostname(),
    "finished_at_utc": datetime.now(timezone.utc).isoformat(),
    "zarr_path": str(zarr_path),
    "crop_run": crop_run,
    "crop_run_selection": crop_run_selection,
    "crop_batch_result_json": str(crop_result_json),
    "crop_run_attrs": crop_attrs,
}
status_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(f"status_json={status_json}")
' "\$ZARR_PATH" "\$STATUS_JSON" "\$CACHE_CROP_RUN" "\$CROP_RESULT_JSON"
JOBSCRIPT
chmod +x "$CROP_SCRIPT"

cat > "$CACHE_SCRIPT" <<JOBSCRIPT
#!/usr/bin/env bash
set -euo pipefail

cd "$(pwd)"

CROP_JOB_ID="\${1:?crop job id is required}"
RUN_DIR=${RUN_DIR_Q}
PUBLIC_CACHE_DIR=${PUBLIC_CACHE_DIR_Q}
RUN_LABEL=${SAFE_LABEL_Q}
OVERWRITE=${OVERWRITE_Q}
JOB_ID="\${LSB_JOBID:-manual_cache}"
HOST="\$(hostname)"
STATUS_JSON="\${RUN_DIR}/\${RUN_LABEL}.cache.\${JOB_ID}.json"
PROGRESS_JSONL="\${RUN_DIR}/\${RUN_LABEL}.cache.\${JOB_ID}.progress.jsonl"
CROP_STATUS_JSON="\${RUN_DIR}/\${RUN_LABEL}.crop.\${CROP_JOB_ID}.json"
CROP_RUN="\$(scripts/py -c 'import json, sys
from pathlib import Path
path = Path(sys.argv[1])
payload = json.loads(path.read_text(encoding="utf-8"))
if payload.get("status") != "ok" or payload.get("stage") != "crop_materialized":
    raise RuntimeError(f"Crop status is not a completed materialized publication: {payload!r}")
crop_run = payload.get("crop_run")
if not isinstance(crop_run, str) or not crop_run.strip():
    raise RuntimeError("Crop status does not contain one exact crop_run")
print(crop_run.strip())
' "\${CROP_STATUS_JSON}")"

scratch_user="\${USER:-\$(id -un)}"
if [[ -n "\${LSB_JOBID:-}" && -d "/scratch/\${scratch_user}" && -w "/scratch/\${scratch_user}" && -x "/scratch/\${scratch_user}" ]]; then
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
echo "crop_job_id=\${CROP_JOB_ID}"
echo "crop_status_json=\${CROP_STATUS_JSON}"
echo "crop_run=\${CROP_RUN}"
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

scripts/py -m fisheye.utils.build_flat_roi_cache ${BUILDER_ARGS_SHELL}--crop-run "\${CROP_RUN}" --manifest-path "\${LOCAL_MANIFEST}" --progress-jsonl "\${PROGRESS_JSONL}" --progress-stderr --progress-interval-s 30 --json > "\${RUN_DIR}/\${RUN_LABEL}.cache.\${JOB_ID}.manifest.build.json"

if [[ ! -s "\${LOCAL_MANIFEST}" ]]; then
  echo "Local manifest was not created: \${LOCAL_MANIFEST}" >&2
  exit 1
fi
if [[ ! -s "\${LOCAL_BIN}" ]]; then
  echo "Local binary payload was not created: \${LOCAL_BIN}" >&2
  exit 1
fi

PUBLISH_BIN_COPY_STARTED_NS="\$(date +%s%N)"
cp "\${LOCAL_BIN}" "\${TMP_BIN}"
PUBLISH_BIN_COPY_FINISHED_NS="\$(date +%s%N)"
mv -f "\${TMP_BIN}" "\${FINAL_BIN}"

scripts/py -c 'import json, os, socket, sys; from datetime import datetime, timezone; from pathlib import Path
local_manifest = Path(sys.argv[1])
final_manifest = Path(sys.argv[2])
final_bin = Path(sys.argv[3])
tmp_manifest = Path(sys.argv[4])
copy_started_ns = int(sys.argv[5])
copy_finished_ns = int(sys.argv[6])
payload = json.loads(local_manifest.read_text(encoding="utf-8"))
payload["manifest_path"] = str(final_manifest)
published_bin_size = final_bin.stat().st_size
copy_seconds = max(0.0, (copy_finished_ns - copy_started_ns) / 1_000_000_000.0)
publisher = payload.setdefault("publisher", {})
publisher.update({
    "published_at_utc": datetime.now(timezone.utc).isoformat(),
    "publish_host": socket.gethostname(),
    "lsb_jobid": os.environ.get("LSB_JOBID"),
    "source_manifest_path": str(local_manifest),
    "published_manifest_path": str(final_manifest),
    "published_bin_path": str(final_bin),
    "published_bin_size_bytes": published_bin_size,
    "payload_copy_started_epoch_ns": copy_started_ns,
    "payload_copy_finished_epoch_ns": copy_finished_ns,
    "payload_copy_seconds": copy_seconds,
    "payload_copy_mib_per_second": (
        (published_bin_size / (1024 * 1024)) / copy_seconds if copy_seconds > 0 else None
    ),
    "publish_policy": "payload_first_manifest_last",
})
tmp_manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
' "\${LOCAL_MANIFEST}" "\${FINAL_MANIFEST}" "\${FINAL_BIN}" "\${TMP_MANIFEST}" "\${PUBLISH_BIN_COPY_STARTED_NS}" "\${PUBLISH_BIN_COPY_FINISHED_NS}"
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
    "builder": manifest.get("builder"),
    "publisher": manifest.get("publisher"),
}
status_json.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(f"status_json={status_json}")
' "\${FINAL_MANIFEST}" "\${FINAL_BIN}" "\${STATUS_JSON}"
JOBSCRIPT
chmod +x "$CACHE_SCRIPT"

CROP_BSUB_ARGS=(
  -J "crop_materialized_${SAFE_LABEL}"
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
CACHE_CMD_TEMPLATE="bsub ${CACHE_BSUB_ARGS_BASE_SHELL}-w done\\(<crop_jobid>\\) bash $(printf '%q' "$CACHE_SCRIPT") <crop_jobid>"

echo "Run dir: $RUN_DIR"
echo "Crop script: $CROP_SCRIPT"
echo "Cache script: $CACHE_SCRIPT"
echo "Crop status JSON: ${RUN_DIR}/${SAFE_LABEL}.crop.<JOBID>.json"
echo "Cache status JSON: ${RUN_DIR}/${SAFE_LABEL}.cache.<JOBID>.json"
echo "Public cache dir: $PUBLIC_CACHE_DIR"
echo "Defer registry writes: $DEFER_REGISTRY"
echo "Expected manifest: ${PUBLIC_CACHE_DIR}/${SAFE_LABEL}.flat_roi_cache.json"
echo "Expected payload: ${PUBLIC_CACHE_DIR}/${SAFE_LABEL}.flat_roi_cache.bin"
echo "Crop command: scripts/py -m fisheye.utils.crop_batch ${CROP_ARGS_SHELL}"
echo "Cache builder command: scripts/py -m fisheye.utils.build_flat_roi_cache ${BUILDER_ARGS_SHELL}--crop-run <exact validated crop_run> --manifest-path <scratch manifest> --json"
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
cache_submit_output="$(bsub "${CACHE_BSUB_ARGS[@]}" bash "$CACHE_SCRIPT" "$crop_jobid")"
echo "$cache_submit_output"
cache_jobid="$(printf '%s\n' "$cache_submit_output" | sed -n 's/.*Job <\([0-9][0-9]*\)>.*/\1/p' | head -1)"

echo "crop_jobid=${crop_jobid}"
if [[ -n "$cache_jobid" ]]; then
  echo "cache_jobid=${cache_jobid}"
fi
echo "cache_dependency=done(${crop_jobid})"
