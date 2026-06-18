#!/usr/bin/env bash
set -euo pipefail

ZARR_PATH=""
CROP_RUN=""
PUBLIC_CACHE_ROOT="/misc/public/palette_cache"
PUBLIC_CACHE_DIR=""
LOG_DIR=""
QUEUE="short"
NCORES=4
MEM_GB=64
GPUS=0
WALLTIME="2:00"
BATCH_SIZE=1024
DECODE_BACKEND="auto"
ROI_LIVE_ACCELERATION="cpu"
ROI_LIVE_GPU_CHUNK_FRAMES=32
SHA256=0
OVERWRITE=0
RUN_ID=""
RUN_LABEL=""
WORKFLOW_ID=""
DRY_RUN=0

usage() {
  cat <<'USAGE'
Usage: submit_flat_roi_cache_bsub.sh --zarr PATH [options]

Submit one LSF job that builds a flat_bin_v1 ROI cache on node-local scratch,
then publishes the completed .bin payload and .json manifest to shared workflow
cache storage. The manifest is published last so downstream consumers do not see
an incomplete cache.

Required:
  --zarr PATH                    Analysis/training Zarr archive containing crop_runs

Options:
  --crop-run NAME                Crop run to cache (default: latest_any/latest)
  --public-cache-root PATH       Shared cache root (default: /misc/public/palette_cache)
  --public-cache-dir PATH        Explicit publish directory; overrides root/workflow_id/roi_cache
  --workflow-id ID               Workflow namespace under public-cache-root
  --log-dir PATH                 Log/output directory (default: runs/diagnostics/flat_roi_cache_bsub)
  --queue NAME                   LSF queue (default: short)
  --ncores N                     CPU slots (default: 4)
  --mem-gb N                     Memory request in GB (default: 64)
  --gpus N                       GPU count; 0 omits -gpu (default: 0)
  --walltime H:MM                LSF wall time (default: 2:00)
  --batch-size N                 ROI rows written per cache-builder batch (default: 1024)
  --decode-backend NAME          auto|pynvvc_luma|read_slice (default: auto; fast sequential PyNv when available)
  --roi-live-acceleration NAME   CropImageSource live-read acceleration: cpu|gpu|auto (default: cpu)
  --roi-live-gpu-chunk-frames N  GPU live-read frame chunk size (default: 32)
  --sha256                       Record payload sha256 in the manifest
  --overwrite                    Overwrite existing published manifest/payload
  --run-id ID                    Stable run id instead of UTC timestamp
  --run-label LABEL              Output basename; default is Zarr archive stem
  --dry-run                      Print files and submit command; do not submit
  -h, --help                     Show this message
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --zarr) ZARR_PATH="$2"; shift 2;;
    --crop-run) CROP_RUN="$2"; shift 2;;
    --public-cache-root) PUBLIC_CACHE_ROOT="$2"; shift 2;;
    --public-cache-dir) PUBLIC_CACHE_DIR="$2"; shift 2;;
    --workflow-id) WORKFLOW_ID="$2"; shift 2;;
    --log-dir) LOG_DIR="$2"; shift 2;;
    --queue) QUEUE="$2"; shift 2;;
    --ncores) NCORES="$2"; shift 2;;
    --mem-gb) MEM_GB="$2"; shift 2;;
    --gpus) GPUS="$2"; shift 2;;
    --walltime) WALLTIME="$2"; shift 2;;
    --batch-size) BATCH_SIZE="$2"; shift 2;;
    --decode-backend) DECODE_BACKEND="$2"; shift 2;;
    --roi-live-acceleration) ROI_LIVE_ACCELERATION="$2"; shift 2;;
    --roi-live-gpu-chunk-frames) ROI_LIVE_GPU_CHUNK_FRAMES="$2"; shift 2;;
    --sha256) SHA256=1; shift;;
    --overwrite) OVERWRITE=1; shift;;
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
  LOG_DIR="runs/diagnostics/flat_roi_cache_bsub"
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
  WORKFLOW_ID="flat_roi_cache_${SAFE_RUN_ID}_${SAFE_LABEL}"
fi
SAFE_WORKFLOW_ID="$(printf '%s' "$WORKFLOW_ID" | tr -c 'A-Za-z0-9_.-' '_')"

RUN_DIR="${LOG_DIR}/flat_roi_cache_${SAFE_RUN_ID}_${SAFE_LABEL}"
if [[ -e "$RUN_DIR" ]]; then
  echo "Run directory already exists: $RUN_DIR" >&2
  echo "Choose a different --run-id or --log-dir." >&2
  exit 2
fi
mkdir -p "$RUN_DIR"

if [[ -z "$PUBLIC_CACHE_DIR" ]]; then
  PUBLIC_CACHE_DIR="${PUBLIC_CACHE_ROOT}/${SAFE_WORKFLOW_ID}/roi_cache"
fi

BUILDER_ARGS=(
  "$ZARR_PATH"
  --batch-size "$BATCH_SIZE"
  --decode-backend "$DECODE_BACKEND"
  --roi-live-acceleration "$ROI_LIVE_ACCELERATION"
  --roi-live-gpu-chunk-frames "$ROI_LIVE_GPU_CHUNK_FRAMES"
)
if [[ -n "$CROP_RUN" ]]; then BUILDER_ARGS+=(--crop-run "$CROP_RUN"); fi
if [[ "$SHA256" == "1" ]]; then BUILDER_ARGS+=(--sha256); fi
if [[ "$OVERWRITE" == "1" ]]; then BUILDER_ARGS+=(--overwrite); fi

printf -v BUILDER_ARGS_SHELL '%q ' "${BUILDER_ARGS[@]}"

JOB_SCRIPT="${RUN_DIR}/run_flat_roi_cache_publish.sh"
RUN_DIR_Q="$(printf '%q' "$RUN_DIR")"
PUBLIC_CACHE_DIR_Q="$(printf '%q' "$PUBLIC_CACHE_DIR")"
SAFE_LABEL_Q="$(printf '%q' "$SAFE_LABEL")"
OVERWRITE_Q="$(printf '%q' "$OVERWRITE")"

cat > "$JOB_SCRIPT" <<JOBSCRIPT
#!/usr/bin/env bash
set -euo pipefail

cd "$(pwd)"

RUN_DIR=${RUN_DIR_Q}
PUBLIC_CACHE_DIR=${PUBLIC_CACHE_DIR_Q}
RUN_LABEL=${SAFE_LABEL_Q}
OVERWRITE=${OVERWRITE_Q}
JOB_ID="\${LSB_JOBID:-manual}"
HOST="\$(hostname)"
STATUS_JSON="\${RUN_DIR}/\${RUN_LABEL}.\${JOB_ID}.json"
PROGRESS_JSONL="\${RUN_DIR}/\${RUN_LABEL}.\${JOB_ID}.progress.jsonl"

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

scripts/py -m fisheye.utils.build_flat_roi_cache ${BUILDER_ARGS_SHELL}--manifest-path "\${LOCAL_MANIFEST}" --progress-jsonl "\${PROGRESS_JSONL}" --progress-stderr --progress-interval-s 30 --json > "\${RUN_DIR}/\${RUN_LABEL}.\${JOB_ID}.manifest.build.json"

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
local_manifest = Path(sys.argv[1])
final_manifest = Path(sys.argv[2])
final_bin = Path(sys.argv[3])
status_json = Path(sys.argv[4])
manifest = json.loads(final_manifest.read_text(encoding="utf-8"))
status = {
    "status": "ok",
    "schema": manifest.get("schema"),
    "layout": manifest.get("layout"),
    "job_id": os.environ.get("LSB_JOBID"),
    "host": socket.gethostname(),
    "finished_at_utc": datetime.now(timezone.utc).isoformat(),
    "local_manifest": str(local_manifest),
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
' "\${LOCAL_MANIFEST}" "\${FINAL_MANIFEST}" "\${FINAL_BIN}" "\${STATUS_JSON}"
JOBSCRIPT
chmod +x "$JOB_SCRIPT"

BSUB_ARGS=(
  -J "flat_roi_cache_${SAFE_LABEL}"
  -n "$NCORES"
  -W "$WALLTIME"
  -R "rusage[mem=${MEM_GB}G]"
  -oo "${RUN_DIR}/%J.out"
  -eo "${RUN_DIR}/%J.err"
)
if [[ -n "$QUEUE" ]]; then
  BSUB_ARGS+=(-q "$QUEUE")
fi
if [[ "$GPUS" != "0" ]]; then
  BSUB_ARGS+=(-gpu "num=${GPUS}")
fi

printf -v BSUB_ARGS_SHELL '%q ' "${BSUB_ARGS[@]}"
BSUB_CMD="bsub ${BSUB_ARGS_SHELL}bash $(printf '%q' "$JOB_SCRIPT")"

echo "Run dir: $RUN_DIR"
echo "Job script: $JOB_SCRIPT"
echo "Expected status JSON: ${RUN_DIR}/${SAFE_LABEL}.<JOBID>.json"
echo "Output log: ${RUN_DIR}/<JOBID>.out"
echo "Error log: ${RUN_DIR}/<JOBID>.err"
echo "Public cache dir: $PUBLIC_CACHE_DIR"
echo "Expected manifest: ${PUBLIC_CACHE_DIR}/${SAFE_LABEL}.flat_roi_cache.json"
echo "Expected payload: ${PUBLIC_CACHE_DIR}/${SAFE_LABEL}.flat_roi_cache.bin"
echo "Builder command: scripts/py -m fisheye.utils.build_flat_roi_cache ${BUILDER_ARGS_SHELL}--manifest-path <scratch manifest> --json"
echo "Submit command: $BSUB_CMD"

if [[ "$DRY_RUN" == "1" ]]; then
  echo "Dry run only; no submission."
  exit 0
fi

if ! command -v bsub >/dev/null 2>&1; then
  echo "bsub not found in PATH. Is this an LSF cluster?" >&2
  exit 2
fi

bsub "${BSUB_ARGS[@]}" bash "$JOB_SCRIPT"
