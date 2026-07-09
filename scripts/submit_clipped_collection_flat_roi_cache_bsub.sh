#!/usr/bin/env bash
set -euo pipefail

ZARR_PATH=""
COLLECTION_ID=""
RECORDING_FRAME_INDEX=""
CLIP_IDS=()
WORK_UNIT_IDS=()
PUBLIC_CACHE_ROOT="/misc/public/palette_cache"
PUBLIC_CACHE_DIR=""
LOG_DIR=""
RUN_ID=""
RUN_LABEL=""
QUEUE="gpu_l4"
NCORES=4
MEM_GB=64
GPUS=1
WALLTIME="2:00"
ROI_SIZE=()
LIMIT_ROWS=""
GPU_CHUNK_FRAMES=32
PROGRESS_INTERVAL_S=30
PROGRESS_EVERY_BATCHES=0
SHA256=0
OVERWRITE=0
DRY_RUN=0

usage() {
  cat <<'USAGE'
Usage: submit_clipped_collection_flat_roi_cache_bsub.sh --zarr PATH --collection-id ID [options]

Submit one LSF job that builds a finalized clipped refined-detect collection
flat ROI cache on node-local scratch, then publishes payload, row-index parquet,
and manifest to shared workflow cache storage in manifest-last order.

Required:
  --zarr PATH                       Analysis Zarr archive
  --collection-id ID                Finalized clipped refined-detect collection id

Cache options:
  --recording-frame-index PATH      Override recording_frame_index.parquet path
  --clip-id ID                      Restrict to one finalized collection clip id; repeatable
  --work-unit-id ID                 Restrict to one finalized collection work_unit_id; repeatable
  --public-cache-root PATH          Shared cache root (default: /misc/public/palette_cache)
  --public-cache-dir PATH           Explicit publish dir; overrides root/collection_id/roi_cache
  --roi-size H W                    ROI size in Palette order; default from archive policy
  --limit-rows N                    Debug/smoke limit on ROI rows
  --gpu-chunk-frames N              Sequential PyNv decode batch size per clip (default: 32)
  --sha256                          Record payload sha256 in manifest
  --overwrite                       Overwrite existing published cache files

LSF options:
  --queue NAME                      LSF queue (default: gpu_l4)
  --ncores N                        CPU slots (default: 4)
  --mem-gb N                        Memory request in GB (default: 64)
  --gpus N                          GPU count; 0 omits -gpu (default: 1)
  --walltime H:MM                   Wall time (default: 2:00)

Logging:
  --log-dir PATH                    Log/output directory (default: runs/diagnostics/clipped_collection_flat_roi_cache_bsub)
  --run-id ID                       Stable run id instead of UTC timestamp
  --run-label LABEL                 Output basename; default is collection id
  --progress-interval-s SECONDS     Progress interval passed to builder (default: 30)
  --progress-every-batches N        Emit progress every N batches; 0 disables count-based emission

General:
  --dry-run                         Print files and submit command; do not submit
  -h, --help                        Show this message
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --zarr) ZARR_PATH="$2"; shift 2;;
    --collection-id) COLLECTION_ID="$2"; shift 2;;
    --recording-frame-index) RECORDING_FRAME_INDEX="$2"; shift 2;;
    --clip-id) CLIP_IDS+=("$2"); shift 2;;
    --work-unit-id) WORK_UNIT_IDS+=("$2"); shift 2;;
    --public-cache-root) PUBLIC_CACHE_ROOT="$2"; shift 2;;
    --public-cache-dir) PUBLIC_CACHE_DIR="$2"; shift 2;;
    --roi-size) ROI_SIZE=("$2" "$3"); shift 3;;
    --limit-rows) LIMIT_ROWS="$2"; shift 2;;
    --gpu-chunk-frames) GPU_CHUNK_FRAMES="$2"; shift 2;;
    --sha256) SHA256=1; shift;;
    --overwrite) OVERWRITE=1; shift;;
    --queue) QUEUE="$2"; shift 2;;
    --ncores) NCORES="$2"; shift 2;;
    --mem-gb) MEM_GB="$2"; shift 2;;
    --gpus) GPUS="$2"; shift 2;;
    --walltime) WALLTIME="$2"; shift 2;;
    --log-dir) LOG_DIR="$2"; shift 2;;
    --run-id) RUN_ID="$2"; shift 2;;
    --run-label) RUN_LABEL="$2"; shift 2;;
    --progress-interval-s) PROGRESS_INTERVAL_S="$2"; shift 2;;
    --progress-every-batches) PROGRESS_EVERY_BATCHES="$2"; shift 2;;
    --dry-run) DRY_RUN=1; shift;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown arg: $1" >&2; usage; exit 2;;
  esac
done

if [[ -z "$ZARR_PATH" || -z "$COLLECTION_ID" ]]; then
  echo "Missing required --zarr PATH or --collection-id ID" >&2
  usage
  exit 2
fi

if [[ "$DRY_RUN" != "1" ]]; then
  [[ -d "$ZARR_PATH" ]] || { echo "Zarr path not found: $ZARR_PATH" >&2; exit 2; }
  if [[ -n "$RECORDING_FRAME_INDEX" ]]; then
    [[ -f "$RECORDING_FRAME_INDEX" ]] || { echo "Recording frame index not found: $RECORDING_FRAME_INDEX" >&2; exit 2; }
  fi
fi

if [[ -z "$LOG_DIR" ]]; then
  LOG_DIR="runs/diagnostics/clipped_collection_flat_roi_cache_bsub"
fi
if [[ -z "$RUN_ID" ]]; then
  RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
fi
if [[ -z "$RUN_LABEL" ]]; then
  RUN_LABEL="$COLLECTION_ID"
fi

SAFE_LABEL="$(printf '%s' "$RUN_LABEL" | tr -c 'A-Za-z0-9_.-' '_')"
SAFE_RUN_ID="$(printf '%s' "$RUN_ID" | tr -c 'A-Za-z0-9_.-' '_')"
SAFE_COLLECTION_ID="$(printf '%s' "$COLLECTION_ID" | tr -c 'A-Za-z0-9_.-' '_')"

RUN_DIR="${LOG_DIR}/clipped_collection_flat_roi_cache_${SAFE_RUN_ID}_${SAFE_LABEL}"
if [[ -e "$RUN_DIR" ]]; then
  echo "Run directory already exists: $RUN_DIR" >&2
  echo "Choose a different --run-id or --log-dir." >&2
  exit 2
fi
mkdir -p "$RUN_DIR"

if [[ -z "$PUBLIC_CACHE_DIR" ]]; then
  PUBLIC_CACHE_DIR="${PUBLIC_CACHE_ROOT}/${SAFE_COLLECTION_ID}/roi_cache"
fi

BUILDER_ARGS=(
  "$ZARR_PATH"
  --collection-id "$COLLECTION_ID"
  --gpu-chunk-frames "$GPU_CHUNK_FRAMES"
)
if [[ -n "$RECORDING_FRAME_INDEX" ]]; then BUILDER_ARGS+=(--recording-frame-index "$RECORDING_FRAME_INDEX"); fi
for clip_id in "${CLIP_IDS[@]}"; do BUILDER_ARGS+=(--clip-id "$clip_id"); done
for work_unit_id in "${WORK_UNIT_IDS[@]}"; do BUILDER_ARGS+=(--work-unit-id "$work_unit_id"); done
if [[ "${#ROI_SIZE[@]}" -gt 0 ]]; then BUILDER_ARGS+=(--roi-size "${ROI_SIZE[@]}"); fi
if [[ -n "$LIMIT_ROWS" ]]; then BUILDER_ARGS+=(--limit-rows "$LIMIT_ROWS"); fi
if [[ "$SHA256" == "1" ]]; then BUILDER_ARGS+=(--sha256); fi
if [[ "$OVERWRITE" == "1" ]]; then BUILDER_ARGS+=(--overwrite); fi
printf -v BUILDER_ARGS_SHELL '%q ' "${BUILDER_ARGS[@]}"

scripts/py - "$RUN_DIR/submission_context.json" \
  "$ZARR_PATH" "$COLLECTION_ID" "$RECORDING_FRAME_INDEX" "$PUBLIC_CACHE_DIR" "$RUN_ID" "$RUN_LABEL" \
  "$SAFE_LABEL" "$QUEUE" "$NCORES" "$MEM_GB" "$GPUS" "$WALLTIME" "$LIMIT_ROWS" "$GPU_CHUNK_FRAMES" \
  "$(IFS=,; echo "${CLIP_IDS[*]}")" "$(IFS=,; echo "${WORK_UNIT_IDS[*]}")" <<'PY'
import json
import sys
from pathlib import Path

(
    output_path,
    zarr_path,
    collection_id,
    recording_frame_index,
    public_cache_dir,
    run_id,
    run_label,
    safe_label,
    queue,
    ncores,
    mem_gb,
    gpus,
    walltime,
    limit_rows,
    gpu_chunk_frames,
    clip_ids_csv,
    work_unit_ids_csv,
) = sys.argv[1:]

def split_csv(value):
    return [item for item in value.split(",") if item]

payload = {
    "schema_version": 1,
    "submission_kind": "clipped_collection_flat_roi_cache_bsub",
    "zarr_path": zarr_path,
    "collection_id": collection_id,
    "recording_frame_index": recording_frame_index or None,
    "public_cache_dir": public_cache_dir,
    "run_id": run_id,
    "run_label": run_label,
    "safe_label": safe_label,
    "queue": queue or None,
    "ncores": int(ncores),
    "mem_gb": int(mem_gb),
    "gpus": int(gpus),
    "walltime": walltime,
    "limit_rows": int(limit_rows) if limit_rows else None,
    "gpu_chunk_frames": int(gpu_chunk_frames),
    "clip_ids": split_csv(clip_ids_csv),
    "work_unit_ids": split_csv(work_unit_ids_csv),
}
Path(output_path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

JOB_SCRIPT="${RUN_DIR}/run_clipped_collection_flat_roi_cache.sh"
RUN_DIR_Q="$(printf '%q' "$RUN_DIR")"
PUBLIC_CACHE_DIR_Q="$(printf '%q' "$PUBLIC_CACHE_DIR")"
SAFE_LABEL_Q="$(printf '%q' "$SAFE_LABEL")"
OVERWRITE_Q="$(printf '%q' "$OVERWRITE")"
COLLECTION_ID_Q="$(printf '%q' "$COLLECTION_ID")"

cat > "$JOB_SCRIPT" <<JOBSCRIPT
#!/usr/bin/env bash
set -euo pipefail

cd "$(pwd)"

RUN_DIR=${RUN_DIR_Q}
PUBLIC_CACHE_DIR=${PUBLIC_CACHE_DIR_Q}
RUN_LABEL=${SAFE_LABEL_Q}
OVERWRITE=${OVERWRITE_Q}
COLLECTION_ID=${COLLECTION_ID_Q}
JOB_ID="\${LSB_JOBID:-manual}"
HOST="\$(hostname)"
STATUS_JSON="\${RUN_DIR}/\${RUN_LABEL}.\${JOB_ID}.json"
PROGRESS_JSONL="\${RUN_DIR}/\${RUN_LABEL}.\${JOB_ID}.progress.jsonl"

scratch_user="\${USER:-\$(id -un)}"
if [[ -n "\${LSB_JOBID:-}" && -d "/scratch/\${scratch_user}" && -w "/scratch/\${scratch_user}" && -x "/scratch/\${scratch_user}" ]]; then
  SCRATCH_ROOT="/scratch/\${scratch_user}/\${LSB_JOBID}"
else
  SCRATCH_ROOT="\${TMPDIR:-/tmp}/palette_clipped_collection_flat_roi_cache_\${JOB_ID}"
fi
export PALETTE_JOB_CACHE="\${SCRATCH_ROOT}/palette_cache"
export MPLBACKEND=Agg

LOCAL_CACHE_DIR="\${PALETTE_JOB_CACHE}/clipped_collection_flat_roi_cache"
LOCAL_MANIFEST="\${LOCAL_CACHE_DIR}/\${RUN_LABEL}.flat_roi_cache.json"
LOCAL_BIN="\${LOCAL_MANIFEST%.json}.bin"
LOCAL_ROWS="\${LOCAL_MANIFEST%.json}.rows.parquet"
FINAL_MANIFEST="\${PUBLIC_CACHE_DIR}/\$(basename "\${LOCAL_MANIFEST}")"
FINAL_BIN="\${PUBLIC_CACHE_DIR}/\$(basename "\${LOCAL_BIN}")"
FINAL_ROWS="\${PUBLIC_CACHE_DIR}/\$(basename "\${LOCAL_ROWS}")"
TMP_BIN="\${FINAL_BIN}.tmp.\${JOB_ID}.\${HOST}"
TMP_ROWS="\${FINAL_ROWS}.tmp.\${JOB_ID}.\${HOST}"
TMP_MANIFEST="\${FINAL_MANIFEST}.tmp.\${JOB_ID}.\${HOST}"

cleanup_tmp() {
  rm -f "\${TMP_BIN}" "\${TMP_ROWS}" "\${TMP_MANIFEST}"
}
trap cleanup_tmp EXIT

mkdir -p "\${LOCAL_CACHE_DIR}" "\${RUN_DIR}" "\${PUBLIC_CACHE_DIR}"

echo "repo=\$(pwd)"
echo "host=\${HOST}"
echo "job_id=\${JOB_ID}"
echo "collection_id=\${COLLECTION_ID}"
echo "scratch_root=\${SCRATCH_ROOT}"
echo "palette_job_cache=\${PALETTE_JOB_CACHE}"
echo "local_manifest=\${LOCAL_MANIFEST}"
echo "local_payload=\${LOCAL_BIN}"
echo "local_row_index=\${LOCAL_ROWS}"
echo "public_manifest=\${FINAL_MANIFEST}"
echo "public_payload=\${FINAL_BIN}"
echo "public_row_index=\${FINAL_ROWS}"
echo "status_json=\${STATUS_JSON}"
echo "progress_jsonl=\${PROGRESS_JSONL}"

if [[ "\${OVERWRITE}" != "1" && ( -e "\${FINAL_MANIFEST}" || -e "\${FINAL_BIN}" || -e "\${FINAL_ROWS}" ) ]]; then
  echo "Published cache already exists; pass --overwrite to replace:" >&2
  echo "  \${FINAL_MANIFEST}" >&2
  echo "  \${FINAL_BIN}" >&2
  echo "  \${FINAL_ROWS}" >&2
  exit 2
fi

scripts/py -m fisheye.utils.build_clipped_collection_flat_roi_cache ${BUILDER_ARGS_SHELL}--manifest-path "\${LOCAL_MANIFEST}" --progress-jsonl "\${PROGRESS_JSONL}" --progress-stderr --progress-interval-s ${PROGRESS_INTERVAL_S} --progress-every-batches ${PROGRESS_EVERY_BATCHES} --json > "\${RUN_DIR}/\${RUN_LABEL}.\${JOB_ID}.manifest.build.json"

if [[ ! -s "\${LOCAL_MANIFEST}" ]]; then
  echo "Local manifest was not created: \${LOCAL_MANIFEST}" >&2
  exit 1
fi
if [[ ! -s "\${LOCAL_BIN}" ]]; then
  echo "Local binary payload was not created: \${LOCAL_BIN}" >&2
  exit 1
fi
if [[ ! -s "\${LOCAL_ROWS}" ]]; then
  echo "Local row-index parquet was not created: \${LOCAL_ROWS}" >&2
  exit 1
fi

PUBLISH_BIN_COPY_STARTED_NS="\$(date +%s%N)"
cp "\${LOCAL_BIN}" "\${TMP_BIN}"
PUBLISH_BIN_COPY_FINISHED_NS="\$(date +%s%N)"
mv -f "\${TMP_BIN}" "\${FINAL_BIN}"
PUBLISH_ROWS_COPY_STARTED_NS="\$(date +%s%N)"
cp "\${LOCAL_ROWS}" "\${TMP_ROWS}"
PUBLISH_ROWS_COPY_FINISHED_NS="\$(date +%s%N)"
mv -f "\${TMP_ROWS}" "\${FINAL_ROWS}"

scripts/py -c 'import json, os, socket, sys; from datetime import datetime, timezone; from pathlib import Path
local_manifest = Path(sys.argv[1])
final_manifest = Path(sys.argv[2])
final_bin = Path(sys.argv[3])
final_rows = Path(sys.argv[4])
tmp_manifest = Path(sys.argv[5])
bin_copy_started_ns = int(sys.argv[6])
bin_copy_finished_ns = int(sys.argv[7])
rows_copy_started_ns = int(sys.argv[8])
rows_copy_finished_ns = int(sys.argv[9])
payload = json.loads(local_manifest.read_text(encoding="utf-8"))
payload["manifest_path"] = str(final_manifest)
array = payload.setdefault("array", {})
array["bin_path"] = final_bin.name
row_index = payload.setdefault("row_index", {})
row_index["path"] = final_rows.name
published_bin_size = final_bin.stat().st_size
published_rows_size = final_rows.stat().st_size
bin_copy_seconds = max(0.0, (bin_copy_finished_ns - bin_copy_started_ns) / 1_000_000_000.0)
rows_copy_seconds = max(0.0, (rows_copy_finished_ns - rows_copy_started_ns) / 1_000_000_000.0)
publisher = payload.setdefault("publisher", {})
publisher.update({
    "published_at_utc": datetime.now(timezone.utc).isoformat(),
    "publish_host": socket.gethostname(),
    "lsb_jobid": os.environ.get("LSB_JOBID"),
    "source_manifest_path": str(local_manifest),
    "published_manifest_path": str(final_manifest),
    "published_bin_path": str(final_bin),
    "published_row_index_path": str(final_rows),
    "published_bin_size_bytes": published_bin_size,
    "published_row_index_size_bytes": published_rows_size,
    "payload_copy_started_epoch_ns": bin_copy_started_ns,
    "payload_copy_finished_epoch_ns": bin_copy_finished_ns,
    "payload_copy_seconds": bin_copy_seconds,
    "payload_copy_mib_per_second": (
        (published_bin_size / (1024 * 1024)) / bin_copy_seconds if bin_copy_seconds > 0 else None
    ),
    "row_index_copy_started_epoch_ns": rows_copy_started_ns,
    "row_index_copy_finished_epoch_ns": rows_copy_finished_ns,
    "row_index_copy_seconds": rows_copy_seconds,
    "row_index_copy_mib_per_second": (
        (published_rows_size / (1024 * 1024)) / rows_copy_seconds if rows_copy_seconds > 0 else None
    ),
    "publish_policy": "payload_and_row_index_first_manifest_last",
})
tmp_manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
' "\${LOCAL_MANIFEST}" "\${FINAL_MANIFEST}" "\${FINAL_BIN}" "\${FINAL_ROWS}" "\${TMP_MANIFEST}" "\${PUBLISH_BIN_COPY_STARTED_NS}" "\${PUBLISH_BIN_COPY_FINISHED_NS}" "\${PUBLISH_ROWS_COPY_STARTED_NS}" "\${PUBLISH_ROWS_COPY_FINISHED_NS}"
mv -f "\${TMP_MANIFEST}" "\${FINAL_MANIFEST}"

scripts/py -c 'import sys; from pathlib import Path; from fisheye.shared.flat_roi_cache import open_flat_roi_cache
manifest_path = Path(sys.argv[1])
row_index_path = Path(sys.argv[2])
cache = open_flat_roi_cache(manifest_path)
try:
    if not row_index_path.exists():
        raise FileNotFoundError(row_index_path)
    print(f"published_cache_validated=1 shape={cache.shape} dtype={cache.dtype} bin={cache.bin_path} row_index={row_index_path}")
finally:
    cache.close()
' "\${FINAL_MANIFEST}" "\${FINAL_ROWS}"

scripts/py -c 'import json, os, socket, sys; from datetime import datetime, timezone; from pathlib import Path
final_manifest = Path(sys.argv[1])
final_bin = Path(sys.argv[2])
final_rows = Path(sys.argv[3])
status_json = Path(sys.argv[4])
manifest = json.loads(final_manifest.read_text(encoding="utf-8"))
status = {
    "status": "ok",
    "stage": "clipped_collection_flat_roi_cache_publish",
    "schema": manifest.get("schema"),
    "layout": manifest.get("layout"),
    "job_id": os.environ.get("LSB_JOBID"),
    "host": socket.gethostname(),
    "finished_at_utc": datetime.now(timezone.utc).isoformat(),
    "published_manifest": str(final_manifest),
    "published_bin": str(final_bin),
    "published_row_index": str(final_rows),
    "published_bin_size_bytes": final_bin.stat().st_size,
    "published_row_index_size_bytes": final_rows.stat().st_size,
    "source": manifest.get("source"),
    "array": manifest.get("array"),
    "row_index": manifest.get("row_index"),
    "builder": manifest.get("builder"),
    "timing": manifest.get("timing"),
    "publisher": manifest.get("publisher"),
}
status_json.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(f"status_json={status_json}")
' "\${FINAL_MANIFEST}" "\${FINAL_BIN}" "\${FINAL_ROWS}" "\${STATUS_JSON}"
JOBSCRIPT
chmod +x "$JOB_SCRIPT"

BSUB_ARGS=(
  -J "clipped_roi_cache_${SAFE_LABEL}"
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
echo "Submission context: ${RUN_DIR}/submission_context.json"
echo "Expected status JSON: ${RUN_DIR}/${SAFE_LABEL}.<JOBID>.json"
echo "Expected progress JSONL: ${RUN_DIR}/${SAFE_LABEL}.<JOBID>.progress.jsonl"
echo "Output log: ${RUN_DIR}/<JOBID>.out"
echo "Error log: ${RUN_DIR}/<JOBID>.err"
echo "Public cache dir: $PUBLIC_CACHE_DIR"
echo "Expected manifest: ${PUBLIC_CACHE_DIR}/${SAFE_LABEL}.flat_roi_cache.json"
echo "Expected payload: ${PUBLIC_CACHE_DIR}/${SAFE_LABEL}.flat_roi_cache.bin"
echo "Expected row index: ${PUBLIC_CACHE_DIR}/${SAFE_LABEL}.flat_roi_cache.rows.parquet"
echo "Builder command: scripts/py -m fisheye.utils.build_clipped_collection_flat_roi_cache ${BUILDER_ARGS_SHELL}--manifest-path <scratch manifest> --progress-jsonl <progress> --progress-stderr --json"
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
