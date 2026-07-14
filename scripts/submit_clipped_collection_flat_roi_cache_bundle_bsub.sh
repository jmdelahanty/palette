#!/usr/bin/env bash
set -euo pipefail

ZARR_PATH=""
COLLECTION_ID=""
RECORDING_FRAME_INDEX=""
CLIP_IDS=()
WORK_UNIT_IDS=()
ALL_CLIPS=0
PUBLIC_CACHE_ROOT="/nrs/johnson/palette_staging/flat_roi_cache"
PUBLIC_CACHE_DIR=""
LOG_DIR=""
RUN_ID=""
RUN_LABEL=""
QUEUE="gpu_l4"
NCORES=8
MEM_GB=64
GPUS=1
GPU_RESOURCE=""
WALLTIME="4:00"
ROI_SIZE=()
LIMIT_ROWS=""
GPU_CHUNK_FRAMES=32
MAX_WORKERS=4
PROGRESS_INTERVAL_S=30
PROGRESS_EVERY_BATCHES=0
SHA256=0
OVERWRITE=0
DRY_RUN=0
RUN_DIRECT=0

usage() {
  cat <<'USAGE'
Usage: submit_clipped_collection_flat_roi_cache_bundle_bsub.sh --zarr PATH --collection-id ID (--clip-id ID... | --all-clips) [options]

Submit one LSF job that runs multiple clipped-collection flat ROI cache builders
concurrently on one GPU allocation. Each child worker builds exactly one clip
cache on node-local scratch, publishes its own payload/row-index/manifest in
manifest-last order, and the parent job writes a bundle summary.

This is intended for GPUs with multiple NVDEC engines, such as NVIDIA L4. It
starts multiple independent PyNvVideoCodec decoder sessions in one job without
requiring shared writes to a single flat cache.

Required:
  --zarr PATH                       Analysis Zarr archive
  --collection-id ID                Finalized clipped refined-detect collection id
  --clip-id ID                      Add one clip to this bundle; repeatable
  --all-clips                       Build one child cache for every clip in the collection

Selection:
  --recording-frame-index PATH      Override recording_frame_index.parquet path
  --work-unit-id ID                 Restrict each child to matching work_unit_id; repeatable

Cache options:
  --public-cache-root PATH          Shared cache root (default: /nrs/johnson/palette_staging/flat_roi_cache)
  --public-cache-dir PATH           Explicit publish dir; overrides root/collection_id/roi_cache
  --roi-size H W                    ROI size in Palette order; default from archive policy
  --limit-rows N                    Debug/smoke limit per child clip
  --gpu-chunk-frames N              Sequential PyNv decode batch size per child (default: 32)
  --sha256                          Record payload sha256 in child manifests
  --overwrite                       Overwrite existing published child cache files

Bundle options:
  --max-workers N                   Concurrent child builders inside this job (default: 4)

LSF options:
  --queue NAME                      LSF queue (default: gpu_l4)
  --ncores N                        CPU slots (default: 8)
  --mem-gb N                        Memory request in GB (default: 64)
  --gpus N                          GPU count; 0 omits -gpu (default: 1)
  --gpu-resource STRING             Raw LSF -gpu resource string; overrides --gpus.
                                    Needed for multi-process children if the
                                    cluster default is exclusive_process.
  --walltime H:MM                   Wall time (default: 4:00)

Logging:
  --log-dir PATH                    Log/output directory (default: runs/diagnostics/clipped_collection_flat_roi_cache_bundle_bsub)
  --run-id ID                       Stable run id instead of UTC timestamp
  --run-label LABEL                 Bundle label; child cache basenames append __<clip_id>
  --progress-interval-s SECONDS     Progress interval passed to each child builder (default: 30)
  --progress-every-batches N        Emit progress every N batches per child; 0 disables count-based emission

General:
  --dry-run                         Print files and submit command; do not submit
  --run-direct                      Run the generated worker script in the current
                                    LSF allocation instead of nesting another bsub
  -h, --help                        Show this message
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --zarr) ZARR_PATH="$2"; shift 2;;
    --collection-id) COLLECTION_ID="$2"; shift 2;;
    --recording-frame-index) RECORDING_FRAME_INDEX="$2"; shift 2;;
    --clip-id) CLIP_IDS+=("$2"); shift 2;;
    --all-clips) ALL_CLIPS=1; shift;;
    --work-unit-id) WORK_UNIT_IDS+=("$2"); shift 2;;
    --public-cache-root) PUBLIC_CACHE_ROOT="$2"; shift 2;;
    --public-cache-dir) PUBLIC_CACHE_DIR="$2"; shift 2;;
    --roi-size) ROI_SIZE=("$2" "$3"); shift 3;;
    --limit-rows) LIMIT_ROWS="$2"; shift 2;;
    --gpu-chunk-frames) GPU_CHUNK_FRAMES="$2"; shift 2;;
    --max-workers) MAX_WORKERS="$2"; shift 2;;
    --sha256) SHA256=1; shift;;
    --overwrite) OVERWRITE=1; shift;;
    --queue) QUEUE="$2"; shift 2;;
    --ncores) NCORES="$2"; shift 2;;
    --mem-gb) MEM_GB="$2"; shift 2;;
    --gpus) GPUS="$2"; shift 2;;
    --gpu-resource) GPU_RESOURCE="$2"; shift 2;;
    --walltime) WALLTIME="$2"; shift 2;;
    --log-dir) LOG_DIR="$2"; shift 2;;
    --run-id) RUN_ID="$2"; shift 2;;
    --run-label) RUN_LABEL="$2"; shift 2;;
    --progress-interval-s) PROGRESS_INTERVAL_S="$2"; shift 2;;
    --progress-every-batches) PROGRESS_EVERY_BATCHES="$2"; shift 2;;
    --dry-run) DRY_RUN=1; shift;;
    --run-direct) RUN_DIRECT=1; shift;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown arg: $1" >&2; usage; exit 2;;
  esac
done

if [[ "$DRY_RUN" == "1" && "$RUN_DIRECT" == "1" ]]; then
  echo "Use either --dry-run or --run-direct, not both." >&2
  exit 2
fi

if [[ -z "$ZARR_PATH" || -z "$COLLECTION_ID" ]]; then
  echo "Missing required --zarr PATH or --collection-id ID" >&2
  usage
  exit 2
fi
if [[ "$ALL_CLIPS" == "1" && "${#CLIP_IDS[@]}" -gt 0 ]]; then
  echo "Use either --all-clips or explicit --clip-id values, not both." >&2
  exit 2
fi
if [[ "$ALL_CLIPS" != "1" && "${#CLIP_IDS[@]}" -eq 0 ]]; then
  echo "Provide at least one --clip-id, or pass --all-clips." >&2
  exit 2
fi
if [[ "$MAX_WORKERS" -lt 1 ]]; then
  echo "--max-workers must be >= 1" >&2
  exit 2
fi

if [[ "$DRY_RUN" != "1" ]]; then
  [[ -d "$ZARR_PATH" ]] || { echo "Zarr path not found: $ZARR_PATH" >&2; exit 2; }
  if [[ -n "$RECORDING_FRAME_INDEX" ]]; then
    [[ -f "$RECORDING_FRAME_INDEX" ]] || { echo "Recording frame index not found: $RECORDING_FRAME_INDEX" >&2; exit 2; }
  fi
fi

if [[ "$ALL_CLIPS" == "1" ]]; then
  mapfile -t CLIP_IDS < <(
    scripts/py - "$ZARR_PATH" "$COLLECTION_ID" "$(IFS=,; echo "${WORK_UNIT_IDS[*]}")" <<'PY'
import sys
from pathlib import Path
from collections.abc import Mapping

import zarr

zarr_path, collection_id, work_units_csv = sys.argv[1:]
work_filter = {item for item in work_units_csv.split(",") if item}

def open_root(path: Path):
    try:
        return zarr.open_group(str(path), mode="r", use_consolidated=False)
    except TypeError:
        return zarr.open_group(str(path), mode="r")

root = open_root(Path(zarr_path))
collection = root["experiment_index"]["finalized_runs"][collection_id]
selected_runs = collection.attrs.get("selected_runs", [])
clip_ids = []
for row in selected_runs:
    if not isinstance(row, Mapping):
        continue
    clip_id = str(row.get("clip_id") or "").strip()
    work_unit_id = str(row.get("work_unit_id") or "").strip()
    if not clip_id:
        continue
    if work_filter and work_unit_id not in work_filter:
        continue
    if clip_id not in clip_ids:
        clip_ids.append(clip_id)
for clip_id in sorted(clip_ids):
    print(clip_id)
PY
  )
  if [[ "${#CLIP_IDS[@]}" -eq 0 ]]; then
    echo "--all-clips resolved no clip ids for collection: $COLLECTION_ID" >&2
    exit 2
  fi
fi

if [[ -z "$LOG_DIR" ]]; then
  LOG_DIR="runs/diagnostics/clipped_collection_flat_roi_cache_bundle_bsub"
fi
if [[ -z "$RUN_ID" ]]; then
  RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
fi
if [[ -z "$RUN_LABEL" ]]; then
  RUN_LABEL="${COLLECTION_ID}_bundle"
fi

SAFE_LABEL="$(printf '%s' "$RUN_LABEL" | tr -c 'A-Za-z0-9_.-' '_')"
SAFE_RUN_ID="$(printf '%s' "$RUN_ID" | tr -c 'A-Za-z0-9_.-' '_')"
SAFE_COLLECTION_ID="$(printf '%s' "$COLLECTION_ID" | tr -c 'A-Za-z0-9_.-' '_')"

RUN_DIR="${LOG_DIR}/clipped_collection_flat_roi_cache_bundle_${SAFE_RUN_ID}_${SAFE_LABEL}"
if [[ -e "$RUN_DIR" ]]; then
  echo "Run directory already exists: $RUN_DIR" >&2
  echo "Choose a different --run-id or --log-dir." >&2
  exit 2
fi
mkdir -p "$RUN_DIR"

if [[ -z "$PUBLIC_CACHE_DIR" ]]; then
  PUBLIC_CACHE_DIR="${PUBLIC_CACHE_ROOT}/${SAFE_COLLECTION_ID}/roi_cache"
fi

scripts/py - "$RUN_DIR/submission_context.json" \
  "$ZARR_PATH" "$COLLECTION_ID" "$RECORDING_FRAME_INDEX" "$PUBLIC_CACHE_DIR" "$RUN_ID" "$RUN_LABEL" \
  "$SAFE_LABEL" "$QUEUE" "$NCORES" "$MEM_GB" "$GPUS" "$WALLTIME" "$LIMIT_ROWS" "$GPU_CHUNK_FRAMES" \
  "$MAX_WORKERS" "$SHA256" "$OVERWRITE" "$GPU_RESOURCE" \
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
    max_workers,
    sha256,
    overwrite,
    gpu_resource,
    clip_ids_csv,
    work_unit_ids_csv,
) = sys.argv[1:]

def split_csv(value):
    return [item for item in value.split(",") if item]

payload = {
    "schema_version": 1,
    "submission_kind": "clipped_collection_flat_roi_cache_bundle_bsub",
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
    "gpu_resource": gpu_resource or None,
    "walltime": walltime,
    "limit_rows": int(limit_rows) if limit_rows else None,
    "gpu_chunk_frames": int(gpu_chunk_frames),
    "max_workers": int(max_workers),
    "sha256": bool(int(sha256)),
    "overwrite": bool(int(overwrite)),
    "clip_ids": split_csv(clip_ids_csv),
    "work_unit_ids": split_csv(work_unit_ids_csv),
}
Path(output_path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

JOB_SCRIPT="${RUN_DIR}/run_clipped_collection_flat_roi_cache_bundle.sh"
RUN_DIR_Q="$(printf '%q' "$RUN_DIR")"
PUBLIC_CACHE_DIR_Q="$(printf '%q' "$PUBLIC_CACHE_DIR")"
ZARR_PATH_Q="$(printf '%q' "$ZARR_PATH")"
COLLECTION_ID_Q="$(printf '%q' "$COLLECTION_ID")"
RECORDING_FRAME_INDEX_Q="$(printf '%q' "$RECORDING_FRAME_INDEX")"
SAFE_LABEL_Q="$(printf '%q' "$SAFE_LABEL")"
MAX_WORKERS_Q="$(printf '%q' "$MAX_WORKERS")"
LIMIT_ROWS_Q="$(printf '%q' "$LIMIT_ROWS")"
GPU_CHUNK_FRAMES_Q="$(printf '%q' "$GPU_CHUNK_FRAMES")"
PROGRESS_INTERVAL_S_Q="$(printf '%q' "$PROGRESS_INTERVAL_S")"
PROGRESS_EVERY_BATCHES_Q="$(printf '%q' "$PROGRESS_EVERY_BATCHES")"
SHA256_Q="$(printf '%q' "$SHA256")"
OVERWRITE_Q="$(printf '%q' "$OVERWRITE")"

printf -v CLIP_IDS_SHELL '%q ' "${CLIP_IDS[@]}"
if [[ "${#WORK_UNIT_IDS[@]}" -gt 0 ]]; then
  printf -v WORK_UNIT_IDS_SHELL '%q ' "${WORK_UNIT_IDS[@]}"
else
  WORK_UNIT_IDS_SHELL=""
fi
if [[ "${#ROI_SIZE[@]}" -gt 0 ]]; then
  printf -v ROI_SIZE_SHELL '%q ' "${ROI_SIZE[@]}"
else
  ROI_SIZE_SHELL=""
fi

cat > "$JOB_SCRIPT" <<JOBSCRIPT
#!/usr/bin/env bash
set -euo pipefail

cd "$(pwd)"

RUN_DIR=${RUN_DIR_Q}
PUBLIC_CACHE_DIR=${PUBLIC_CACHE_DIR_Q}
ZARR_PATH=${ZARR_PATH_Q}
COLLECTION_ID=${COLLECTION_ID_Q}
RECORDING_FRAME_INDEX=${RECORDING_FRAME_INDEX_Q}
RUN_LABEL=${SAFE_LABEL_Q}
MAX_WORKERS=${MAX_WORKERS_Q}
LIMIT_ROWS=${LIMIT_ROWS_Q}
GPU_CHUNK_FRAMES=${GPU_CHUNK_FRAMES_Q}
PROGRESS_INTERVAL_S=${PROGRESS_INTERVAL_S_Q}
PROGRESS_EVERY_BATCHES=${PROGRESS_EVERY_BATCHES_Q}
SHA256=${SHA256_Q}
OVERWRITE=${OVERWRITE_Q}
CLIP_IDS=(${CLIP_IDS_SHELL})
WORK_UNIT_IDS=(${WORK_UNIT_IDS_SHELL})
ROI_SIZE=(${ROI_SIZE_SHELL})

JOB_ID="\${LSB_JOBID:-manual}"
HOST="\$(hostname)"
BUNDLE_STATUS_JSON="\${RUN_DIR}/\${RUN_LABEL}.\${JOB_ID}.bundle.json"

scratch_user="\${USER:-\$(id -un)}"
if [[ -n "\${LSB_JOBID:-}" && -d "/scratch/\${scratch_user}" && -w "/scratch/\${scratch_user}" && -x "/scratch/\${scratch_user}" ]]; then
  SCRATCH_ROOT="/scratch/\${scratch_user}/\${LSB_JOBID}"
else
  SCRATCH_ROOT="\${TMPDIR:-/tmp}/palette_clipped_collection_flat_roi_cache_bundle_\${JOB_ID}"
fi
export PALETTE_JOB_CACHE="\${SCRATCH_ROOT}/palette_cache"
export MPLBACKEND=Agg

cleanup_job_cache() {
  if [[ -n "\${PALETTE_JOB_CACHE:-}" && "\${PALETTE_JOB_CACHE}" == "\${SCRATCH_ROOT}/palette_cache" ]]; then
    rm -rf "\${PALETTE_JOB_CACHE}"
  fi
}
trap cleanup_job_cache EXIT INT TERM

mkdir -p "\${PALETTE_JOB_CACHE}/clipped_collection_flat_roi_cache" "\${RUN_DIR}" "\${PUBLIC_CACHE_DIR}"

safe_component() {
  printf '%s' "\$1" | tr -c 'A-Za-z0-9_.-' '_'
}

run_one_clip() (
  set -euo pipefail
  local clip_id="\$1"
  local safe_clip worker_label local_cache_dir local_manifest local_bin local_rows final_manifest final_bin final_rows
  safe_clip="\$(safe_component "\${clip_id}")"
  worker_label="\${RUN_LABEL}__\${safe_clip}"
  local_cache_dir="\${PALETTE_JOB_CACHE}/clipped_collection_flat_roi_cache/\${safe_clip}"
  local_manifest="\${local_cache_dir}/\${worker_label}.flat_roi_cache.json"
  local_bin="\${local_manifest%.json}.bin"
  local_rows="\${local_manifest%.json}.rows.parquet"
  final_manifest="\${PUBLIC_CACHE_DIR}/\$(basename "\${local_manifest}")"
  final_bin="\${PUBLIC_CACHE_DIR}/\$(basename "\${local_bin}")"
  final_rows="\${PUBLIC_CACHE_DIR}/\$(basename "\${local_rows}")"
  local tmp_bin tmp_rows tmp_manifest progress_jsonl status_json
  tmp_bin="\${final_bin}.tmp.\${JOB_ID}.\${HOST}"
  tmp_rows="\${final_rows}.tmp.\${JOB_ID}.\${HOST}"
  tmp_manifest="\${final_manifest}.tmp.\${JOB_ID}.\${HOST}"
  progress_jsonl="\${RUN_DIR}/\${worker_label}.\${JOB_ID}.progress.jsonl"
  status_json="\${RUN_DIR}/\${worker_label}.\${JOB_ID}.json"

  cleanup_worker_tmp() {
    rm -f "\${tmp_bin}" "\${tmp_rows}" "\${tmp_manifest}"
  }
  trap cleanup_worker_tmp EXIT

  mkdir -p "\${local_cache_dir}" "\${PUBLIC_CACHE_DIR}"

  local builder_args=(
    "\${ZARR_PATH}"
    --collection-id "\${COLLECTION_ID}"
    --clip-id "\${clip_id}"
    --gpu-chunk-frames "\${GPU_CHUNK_FRAMES}"
  )
  if [[ -n "\${RECORDING_FRAME_INDEX}" ]]; then builder_args+=(--recording-frame-index "\${RECORDING_FRAME_INDEX}"); fi
  for work_unit_id in "\${WORK_UNIT_IDS[@]}"; do builder_args+=(--work-unit-id "\${work_unit_id}"); done
  if [[ "\${#ROI_SIZE[@]}" -gt 0 ]]; then builder_args+=(--roi-size "\${ROI_SIZE[@]}"); fi
  if [[ -n "\${LIMIT_ROWS}" ]]; then builder_args+=(--limit-rows "\${LIMIT_ROWS}"); fi
  if [[ "\${SHA256}" == "1" ]]; then builder_args+=(--sha256); fi
  if [[ "\${OVERWRITE}" == "1" ]]; then builder_args+=(--overwrite); fi

  echo "clip_id=\${clip_id}"
  echo "worker_label=\${worker_label}"
  echo "local_manifest=\${local_manifest}"
  echo "public_manifest=\${final_manifest}"

  if [[ "\${OVERWRITE}" != "1" && ( -e "\${final_manifest}" || -e "\${final_bin}" || -e "\${final_rows}" ) ]]; then
    echo "Published child cache already exists; pass --overwrite to replace: \${final_manifest}" >&2
    exit 2
  fi

  scripts/py -m fisheye.utils.build_clipped_collection_flat_roi_cache "\${builder_args[@]}" \
    --manifest-path "\${local_manifest}" \
    --progress-jsonl "\${progress_jsonl}" \
    --progress-stderr \
    --progress-interval-s "\${PROGRESS_INTERVAL_S}" \
    --progress-every-batches "\${PROGRESS_EVERY_BATCHES}" \
    --json > "\${RUN_DIR}/\${worker_label}.\${JOB_ID}.manifest.build.json"

  if [[ ! -s "\${local_manifest}" ]]; then
    echo "Local manifest was not created: \${local_manifest}" >&2
    exit 1
  fi
  if [[ ! -s "\${local_bin}" ]]; then
    echo "Local binary payload was not created: \${local_bin}" >&2
    exit 1
  fi
  if [[ ! -s "\${local_rows}" ]]; then
    echo "Local row-index parquet was not created: \${local_rows}" >&2
    exit 1
  fi

  local publish_bin_copy_started_ns publish_bin_copy_finished_ns publish_rows_copy_started_ns publish_rows_copy_finished_ns
  publish_bin_copy_started_ns="\$(date +%s%N)"
  cp "\${local_bin}" "\${tmp_bin}"
  publish_bin_copy_finished_ns="\$(date +%s%N)"
  mv -f "\${tmp_bin}" "\${final_bin}"
  publish_rows_copy_started_ns="\$(date +%s%N)"
  cp "\${local_rows}" "\${tmp_rows}"
  publish_rows_copy_finished_ns="\$(date +%s%N)"
  mv -f "\${tmp_rows}" "\${final_rows}"

  scripts/py -c 'import json, os, socket, sys; from datetime import datetime, timezone; from pathlib import Path
local_manifest = Path(sys.argv[1])
final_manifest = Path(sys.argv[2])
final_bin = Path(sys.argv[3])
final_rows = Path(sys.argv[4])
tmp_manifest = Path(sys.argv[5])
clip_id = sys.argv[6]
bin_copy_started_ns = int(sys.argv[7])
bin_copy_finished_ns = int(sys.argv[8])
rows_copy_started_ns = int(sys.argv[9])
rows_copy_finished_ns = int(sys.argv[10])
payload = json.loads(local_manifest.read_text(encoding="utf-8"))
payload["manifest_path"] = str(final_manifest)
array = payload.setdefault("array", {})
array["bin_path"] = final_bin.name
row_index = payload.setdefault("row_index", {})
row_index["path"] = final_rows.name
source = payload.setdefault("source", {})
source["bundle_child_clip_id"] = clip_id
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
    "bundle_child": True,
})
tmp_manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
' "\${local_manifest}" "\${final_manifest}" "\${final_bin}" "\${final_rows}" "\${tmp_manifest}" "\${clip_id}" "\${publish_bin_copy_started_ns}" "\${publish_bin_copy_finished_ns}" "\${publish_rows_copy_started_ns}" "\${publish_rows_copy_finished_ns}"
  mv -f "\${tmp_manifest}" "\${final_manifest}"

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
' "\${final_manifest}" "\${final_rows}"

  scripts/py -c 'import json, os, socket, sys; from datetime import datetime, timezone; from pathlib import Path
final_manifest = Path(sys.argv[1])
final_bin = Path(sys.argv[2])
final_rows = Path(sys.argv[3])
status_json = Path(sys.argv[4])
clip_id = sys.argv[5]
manifest = json.loads(final_manifest.read_text(encoding="utf-8"))
status = {
    "status": "ok",
    "stage": "clipped_collection_flat_roi_cache_bundle_child_publish",
    "clip_id": clip_id,
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
    "publisher": manifest.get("publisher"),
}
status_json.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(f"status_json={status_json}")
' "\${final_manifest}" "\${final_bin}" "\${final_rows}" "\${status_json}" "\${clip_id}"
)

declare -a ACTIVE_PIDS=()
declare -a ACTIVE_LABELS=()
declare -a ACTIVE_CLIPS=()
declare -i FAILURES=0

wait_oldest() {
  if [[ "\${#ACTIVE_PIDS[@]}" -eq 0 ]]; then
    return 0
  fi
  local pid label clip code
  pid="\${ACTIVE_PIDS[0]}"
  label="\${ACTIVE_LABELS[0]}"
  clip="\${ACTIVE_CLIPS[0]}"
  set +e
  wait "\${pid}"
  code="\$?"
  set -e
  echo "\${code}" > "\${RUN_DIR}/\${label}.\${JOB_ID}.exitcode"
  if [[ "\${code}" -ne 0 ]]; then
    echo "child failed clip_id=\${clip} label=\${label} exit_code=\${code}" >&2
    FAILURES+=1
  fi
  ACTIVE_PIDS=("\${ACTIVE_PIDS[@]:1}")
  ACTIVE_LABELS=("\${ACTIVE_LABELS[@]:1}")
  ACTIVE_CLIPS=("\${ACTIVE_CLIPS[@]:1}")
}

for clip_id in "\${CLIP_IDS[@]}"; do
  while [[ "\${#ACTIVE_PIDS[@]}" -ge "\${MAX_WORKERS}" ]]; do
    wait_oldest
  done
  safe_clip="\$(safe_component "\${clip_id}")"
  child_label="\${RUN_LABEL}__\${safe_clip}"
  echo "starting_child clip_id=\${clip_id} label=\${child_label}"
  run_one_clip "\${clip_id}" > "\${RUN_DIR}/\${child_label}.\${JOB_ID}.out" 2> "\${RUN_DIR}/\${child_label}.\${JOB_ID}.err" &
  ACTIVE_PIDS+=("\$!")
  ACTIVE_LABELS+=("\${child_label}")
  ACTIVE_CLIPS+=("\${clip_id}")
done

while [[ "\${#ACTIVE_PIDS[@]}" -gt 0 ]]; do
  wait_oldest
done

scripts/py -c 'import json, os, socket, sys; from datetime import datetime, timezone; from pathlib import Path
run_dir = Path(sys.argv[1])
bundle_status = Path(sys.argv[2])
run_label = sys.argv[3]
job_id = sys.argv[4]
max_workers = int(sys.argv[5])
requested_clips = [item for item in sys.argv[6].split(",") if item]
statuses = []
exitcodes = {}
for path in sorted(run_dir.glob(f"{run_label}__*.{job_id}.json")):
    try:
        statuses.append(json.loads(path.read_text(encoding="utf-8")))
    except Exception as exc:
        statuses.append({"status": "invalid_status_json", "path": str(path), "error": str(exc)})
for path in sorted(run_dir.glob(f"{run_label}__*.{job_id}.exitcode")):
    exitcodes[path.name] = int(path.read_text(encoding="utf-8").strip() or "1")
ok_count = sum(1 for item in statuses if item.get("status") == "ok")
payload_bytes = sum(int(item.get("published_bin_size_bytes") or 0) for item in statuses if item.get("status") == "ok")
rows = 0
for item in statuses:
    if item.get("status") != "ok":
        continue
    row_index = item.get("row_index") if isinstance(item.get("row_index"), dict) else {}
    rows += int(row_index.get("row_count") or 0)
summary = {
    "status": "ok" if ok_count == len(requested_clips) and len(exitcodes) == len(requested_clips) and all(v == 0 for v in exitcodes.values()) else "failed",
    "stage": "clipped_collection_flat_roi_cache_bundle_publish",
    "job_id": os.environ.get("LSB_JOBID"),
    "host": socket.gethostname(),
    "finished_at_utc": datetime.now(timezone.utc).isoformat(),
    "run_label": run_label,
    "max_workers": max_workers,
    "requested_clip_ids": requested_clips,
    "requested_child_count": len(requested_clips),
    "completed_child_count": ok_count,
    "failed_child_count": len(requested_clips) - ok_count,
    "total_rows": rows,
    "total_payload_bytes": payload_bytes,
    "exitcodes": exitcodes,
    "children": statuses,
}
bundle_status.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(f"bundle_status_json={bundle_status}")
print(json.dumps({"status": summary["status"], "children": ok_count, "requested": len(requested_clips), "rows": rows}, sort_keys=True))
' "\${RUN_DIR}" "\${BUNDLE_STATUS_JSON}" "\${RUN_LABEL}" "\${JOB_ID}" "\${MAX_WORKERS}" "\$(IFS=,; echo "\${CLIP_IDS[*]}")"

if [[ "\${FAILURES}" -ne 0 ]]; then
  exit 1
fi
JOBSCRIPT
chmod +x "$JOB_SCRIPT"

BSUB_ARGS=(
  -J "clipped_roi_bundle_${SAFE_LABEL}"
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
  if [[ -n "$GPU_RESOURCE" ]]; then
    BSUB_ARGS+=(-gpu "$GPU_RESOURCE")
  else
    BSUB_ARGS+=(-gpu "num=${GPUS}")
  fi
fi

printf -v BSUB_ARGS_SHELL '%q ' "${BSUB_ARGS[@]}"
BSUB_CMD="bsub ${BSUB_ARGS_SHELL}bash $(printf '%q' "$JOB_SCRIPT")"

echo "Run dir: $RUN_DIR"
echo "Job script: $JOB_SCRIPT"
echo "Submission context: ${RUN_DIR}/submission_context.json"
echo "Expected bundle status JSON: ${RUN_DIR}/${SAFE_LABEL}.<JOBID>.bundle.json"
echo "Output log: ${RUN_DIR}/<JOBID>.out"
echo "Error log: ${RUN_DIR}/<JOBID>.err"
echo "Public cache dir: $PUBLIC_CACHE_DIR"
echo "Clip ids (${#CLIP_IDS[@]}): ${CLIP_IDS[*]}"
echo "Max workers: $MAX_WORKERS"
echo "Child manifests: ${PUBLIC_CACHE_DIR}/${SAFE_LABEL}__<clip_id>.flat_roi_cache.json"
echo "Submit command: $BSUB_CMD"

if [[ "$DRY_RUN" == "1" ]]; then
  echo "Dry run only; no submission."
  exit 0
fi

if [[ "$RUN_DIRECT" == "1" ]]; then
  echo "Running generated bundle worker inside current allocation."
  bash "$JOB_SCRIPT"
  exit $?
fi

if ! command -v bsub >/dev/null 2>&1; then
  echo "bsub not found in PATH. Is this an LSF cluster?" >&2
  exit 2
fi

bsub "${BSUB_ARGS[@]}" bash "$JOB_SCRIPT"
