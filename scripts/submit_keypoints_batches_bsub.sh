#!/usr/bin/env bash
set -euo pipefail

ROOT="/groups/johnson/johnsonlab/jeremy/recordings"
BATCH_SIZE=10
MAX_ACTIVE=12
QUEUE=""
NCORES=4
MEM_GB=32
GPUS=0
REGISTRY="${PALETTE_REGISTRY_PATH:-/groups/johnson/johnsonlab/jeremy/registries/palette_registry.sqlite}"
SET_ID=""
TOP_K=5
REQUIRE_UNIQUE=0
INCLUDE_NON_SUCCESS=0
CROP_RUN=""
OUTPUT_PARENT=""
POSE_SCHEMA=""
BATCH_SIZE_KP=256
KEYPOINT_ROI_SHARD_ROWS=262144
KEYPOINT_ROI_SHARD_ROWS_SET=0
KEYPOINT_FRAME_SHARD_ROWS=262144
NO_KEYPOINT_SHARDING=0
DEVICE=""
IMGSZ=""
CONF=""
IOU=""
MAX_DET=""
MASK_THRESHOLD=""
ROI_CACHE_POLICY=""
ROI_CACHE_DIR=""
ROI_CACHE_MANIFEST=""
STAGE_ROI_CACHE_TO_SCRATCH=0
ROI_CACHE_STAGING_DIR=""
PROGRESS_JSONL_DIR=""
PROGRESS_EVERY_BATCHES=1
NO_PROGRESS_JSONL=0
CPU=0
OVERWRITE=0
LOG_DIR=""
RUN_ID_OVERRIDE=""
DRY_RUN=0
SOURCE="filesystem"
RIG_ID=""
ARENA_ID=""
CAMERA_ID_FILTER=""
PATH_CONTAINS=""

usage() {
  cat <<'USAGE'
Usage: submit_keypoints_batches_bsub.sh [options]

Options:
  --root PATH               Root recordings directory (default: /groups/johnson/johnsonlab/jeremy/recordings)
  --batch-size N            Analysis zarrs per batch job (default: 10)
  --max-active N            Max concurrent jobs in array (default: 12)
  --queue NAME              LSF queue name
  --ncores N                Cores per job (default: 4)
  --mem-gb N                Memory per job in GB (default: 32)
  --gpus N                  GPUs per job; when >0 requests LSF GPUs and defaults --device 0
  --registry PATH           Registry sqlite path (default: $PALETTE_REGISTRY_PATH or /groups/johnson/johnsonlab/jeremy/registries/palette_registry.sqlite)
  --set-id ID               Optional pose model set filter for registry model resolution
  --top-k N                 Candidate provenance depth (default: 5)
  --require-unique          Fail if top model scores tie
  --include-non-success     Include non-success runs in model resolution
  --crop-run NAME           Optional explicit crop run name
  --output-parent NAME      Optional keypoint output parent: keypoints_runs|keypoint_shard_runs
  --pose-schema NAME        Optional pose schema (for example: traditional_v2 for 5-keypoint models)
  --batch-size-kp N         Keypoint inference batch size (default: 256)
  --keypoint-roi-shard-rows N
                            Outer rows for indexed-sharded ROI arrays (default: 262144)
  --keypoint-frame-shard-rows N
                            Outer rows for indexed-sharded frame arrays (default: 262144)
  --no-keypoint-sharding    Use ordinary chunks for keypoint outputs
  --device DEVICE           Torch device override
  --imgsz N                 Pose inference image size override
  --conf FLOAT              Confidence threshold override
  --iou FLOAT               IoU threshold override
  --max-det N               Max detections override
  --mask-threshold FLOAT    Compatibility threshold override
  --roi-cache-policy POLICY ROI cache policy: never|auto|always
  --roi-cache-dir PATH      Scratch directory for temporary ROI caches
  --roi-cache-manifest PATH Explicit flat_bin_v1 ROI cache manifest (requires exactly one target)
  --stage-roi-cache-to-scratch
                            Copy --roi-cache-manifest/payload to node-local scratch before inference.
                            Recommended for large flat caches on GPU jobs.
  --roi-cache-staging-dir PATH
                            Override staging directory (default: /scratch/$USER/$LSB_JOBID when available)
  --progress-jsonl-dir PATH Directory for per-recording live progress JSONL logs
                            (default: <run-dir>/progress_jsonl)
  --progress-every-batches N
                            Write one progress event every N inference batches (default: 1)
  --no-progress-jsonl       Do not pass live progress JSONL paths to keypoint jobs
  --cpu                     Force CPU inference
  --overwrite               Run keypoints even if keypoints run already exists
  --log-dir PATH            Submission logs (default: <root>/logs/run_keypoints_batch/bsub_submissions)
  --run-id ID               Optional stable run id for deterministic reruns
  --dry-run                 Print manifests + commands; do not submit
  --source {filesystem,registry}  Discovery source for analysis zarrs (default: filesystem)
  --rig-id ID               Filter by rig_id (registry source only)
  --arena-id ID             Filter by arena_id (registry source only)
  --camera-id-filter ID     Filter by camera_id (registry source only)
  --path-contains STR       Substring match on zarr_path (registry source only)
  -h, --help                Show this message
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --root) ROOT="$2"; shift 2;;
    --batch-size) BATCH_SIZE="$2"; shift 2;;
    --max-active) MAX_ACTIVE="$2"; shift 2;;
    --queue) QUEUE="$2"; shift 2;;
    --ncores) NCORES="$2"; shift 2;;
    --mem-gb) MEM_GB="$2"; shift 2;;
    --gpus) GPUS="$2"; shift 2;;
    --registry) REGISTRY="$2"; shift 2;;
    --set-id) SET_ID="$2"; shift 2;;
    --top-k) TOP_K="$2"; shift 2;;
    --require-unique) REQUIRE_UNIQUE=1; shift;;
    --include-non-success) INCLUDE_NON_SUCCESS=1; shift;;
    --crop-run) CROP_RUN="$2"; shift 2;;
    --output-parent) OUTPUT_PARENT="$2"; shift 2;;
    --pose-schema) POSE_SCHEMA="$2"; shift 2;;
    --batch-size-kp) BATCH_SIZE_KP="$2"; shift 2;;
    --keypoint-roi-shard-rows) KEYPOINT_ROI_SHARD_ROWS="$2"; KEYPOINT_ROI_SHARD_ROWS_SET=1; shift 2;;
    --keypoint-frame-shard-rows) KEYPOINT_FRAME_SHARD_ROWS="$2"; shift 2;;
    --no-keypoint-sharding) NO_KEYPOINT_SHARDING=1; shift;;
    --device) DEVICE="$2"; shift 2;;
    --imgsz) IMGSZ="$2"; shift 2;;
    --conf) CONF="$2"; shift 2;;
    --iou) IOU="$2"; shift 2;;
    --max-det) MAX_DET="$2"; shift 2;;
    --mask-threshold) MASK_THRESHOLD="$2"; shift 2;;
    --roi-cache-policy) ROI_CACHE_POLICY="$2"; shift 2;;
    --roi-cache-dir) ROI_CACHE_DIR="$2"; shift 2;;
    --roi-cache-manifest) ROI_CACHE_MANIFEST="$2"; shift 2;;
    --stage-roi-cache-to-scratch) STAGE_ROI_CACHE_TO_SCRATCH=1; shift;;
    --roi-cache-staging-dir) ROI_CACHE_STAGING_DIR="$2"; shift 2;;
    --progress-jsonl-dir) PROGRESS_JSONL_DIR="$2"; shift 2;;
    --progress-every-batches) PROGRESS_EVERY_BATCHES="$2"; shift 2;;
    --no-progress-jsonl) NO_PROGRESS_JSONL=1; shift;;
    --cpu) CPU=1; shift;;
    --overwrite) OVERWRITE=1; shift;;
    --log-dir) LOG_DIR="$2"; shift 2;;
    --run-id) RUN_ID_OVERRIDE="$2"; shift 2;;
    --dry-run) DRY_RUN=1; shift;;
    --source) SOURCE="$2"; shift 2;;
    --rig-id) RIG_ID="$2"; shift 2;;
    --arena-id) ARENA_ID="$2"; shift 2;;
    --camera-id-filter) CAMERA_ID_FILTER="$2"; shift 2;;
    --path-contains) PATH_CONTAINS="$2"; shift 2;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown arg: $1"; usage; exit 2;;
  esac
done

if [[ -z "$LOG_DIR" ]]; then
  LOG_DIR="${ROOT}/logs/run_keypoints_batch/bsub_submissions"
fi

if [[ -n "$RUN_ID_OVERRIDE" ]]; then
  RUN_ID="$RUN_ID_OVERRIDE"
else
  RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
fi
RUN_DIR="${LOG_DIR}/kp_${RUN_ID}"

if [[ -e "$RUN_DIR" ]]; then
  echo "Run directory already exists: $RUN_DIR" >&2
  echo "Choose a different --run-id or remove the existing run directory." >&2
  exit 2
fi
if ! mkdir -p "$RUN_DIR" 2>/dev/null; then
  if [[ "$DRY_RUN" == "1" ]]; then
    FALLBACK_LOG_DIR="${TMPDIR:-/tmp}/palette/run_keypoints_batch/bsub_submissions"
    RUN_DIR="${FALLBACK_LOG_DIR}/kp_${RUN_ID}"
    if [[ -e "$RUN_DIR" ]]; then
      echo "Fallback run directory already exists: $RUN_DIR" >&2
      echo "Choose a different --run-id or remove the existing run directory." >&2
      exit 2
    fi
    mkdir -p "$RUN_DIR"
    echo "Warning: cannot write under ${LOG_DIR}; using fallback ${FALLBACK_LOG_DIR}" >&2
  else
    echo "Cannot create run directory: $RUN_DIR" >&2
    echo "Use --log-dir to choose a writable location." >&2
    exit 2
  fi
fi

if [[ "$SOURCE" == "registry" ]]; then
  # Use the Python batch module's --emit-paths mode for registry discovery.
  DISCOVER_ARGS=(--source registry --emit-paths --registry "$REGISTRY" --no-log)
  if [[ "$OVERWRITE" == "1" ]]; then DISCOVER_ARGS+=(--overwrite); fi
  if [[ -n "$RIG_ID" ]]; then DISCOVER_ARGS+=(--rig-id "$RIG_ID"); fi
  if [[ -n "$ARENA_ID" ]]; then DISCOVER_ARGS+=(--arena-id "$ARENA_ID"); fi
  if [[ -n "$CAMERA_ID_FILTER" ]]; then DISCOVER_ARGS+=(--camera-id-filter "$CAMERA_ID_FILTER"); fi
  if [[ -n "$PATH_CONTAINS" ]]; then DISCOVER_ARGS+=(--path-contains "$PATH_CONTAINS"); fi
  DISCOVER_ARGS+=("$ROOT")

  scripts/py -m fisheye.utils.run_keypoints_batch "${DISCOVER_ARGS[@]}" > "${RUN_DIR}/discovered_paths.txt"

  scripts/py - "${RUN_DIR}/discovered_paths.txt" "$BATCH_SIZE" "$RUN_DIR" "$SOURCE" <<'PY'
import json, math, sys
from pathlib import Path

paths_file = Path(sys.argv[1])
batch_size = int(sys.argv[2])
run_dir = Path(sys.argv[3])
source = sys.argv[4]

zarr_paths = [line.strip() for line in paths_file.read_text(encoding="utf-8").splitlines() if line.strip()]
run_dir.mkdir(parents=True, exist_ok=True)

recordings_file = run_dir / "recordings.txt"
recordings_file.write_text("\n".join(zarr_paths) + ("\n" if zarr_paths else ""), encoding="utf-8")

for idx in range(0, len(zarr_paths), batch_size):
    batch_idx = idx // batch_size + 1
    batch_file = run_dir / f"batch_{batch_idx:04d}.txt"
    batch_file.write_text("\n".join(zarr_paths[idx : idx + batch_size]) + "\n", encoding="utf-8")

summary = run_dir / "manifest_summary.json"
summary.write_text(json.dumps({
    "source": source,
    "analysis_zarr_count": len(zarr_paths),
    "batch_size": batch_size,
    "batch_count": math.ceil(len(zarr_paths) / batch_size) if zarr_paths else 0,
}, indent=2) + "\n", encoding="utf-8")
PY
else
  scripts/py - "$ROOT" "$BATCH_SIZE" "$RUN_DIR" <<'PY'
import math
import sys
from pathlib import Path

root = Path(sys.argv[1]).expanduser()
batch_size = int(sys.argv[2])
run_dir = Path(sys.argv[3])

zarr_paths = sorted(str(path.resolve()) for path in root.rglob("*_analysis.zarr"))
run_dir.mkdir(parents=True, exist_ok=True)

recordings_file = run_dir / "recordings.txt"
recordings_file.write_text("\n".join(zarr_paths) + ("\n" if zarr_paths else ""), encoding="utf-8")

for idx in range(0, len(zarr_paths), batch_size):
    batch_idx = idx // batch_size + 1
    batch_file = run_dir / f"batch_{batch_idx:04d}.txt"
    batch_file.write_text("\n".join(zarr_paths[idx : idx + batch_size]) + "\n", encoding="utf-8")

summary = run_dir / "manifest_summary.json"
summary.write_text(
    (
        "{\n"
        f'  "root": "{root}",\n'
        f'  "analysis_zarr_count": {len(zarr_paths)},\n'
        f'  "batch_size": {batch_size},\n'
        f'  "batch_count": {math.ceil(len(zarr_paths)/batch_size) if zarr_paths else 0}\n'
        "}\n"
    ),
    encoding="utf-8",
)
PY
fi

batch_count=$(find "$RUN_DIR" -maxdepth 1 -name 'batch_*.txt' | wc -l | tr -d ' ')
if [[ "$batch_count" == "0" ]]; then
  echo "No *_analysis.zarr targets found under $ROOT"
  exit 0
fi

analysis_count=$(wc -l < "$RUN_DIR/recordings.txt" | tr -d ' ')
if [[ -n "$ROI_CACHE_MANIFEST" && "$analysis_count" != "1" ]]; then
  echo "--roi-cache-manifest is only safe when exactly one analysis zarr target is selected." >&2
  echo "Selected targets: $analysis_count" >&2
  echo "Use a single-recording filter, or add a manifest resolver before running multi-recording batches." >&2
  exit 2
fi
if [[ "$STAGE_ROI_CACHE_TO_SCRATCH" == "1" && -z "$ROI_CACHE_MANIFEST" ]]; then
  echo "--stage-roi-cache-to-scratch requires --roi-cache-manifest." >&2
  exit 2
fi
if [[ "$CPU" == "1" && "$GPUS" != "0" ]]; then
  echo "--cpu cannot be combined with --gpus." >&2
  exit 2
fi
if [[ "$NO_KEYPOINT_SHARDING" == "1" && "$KEYPOINT_ROI_SHARD_ROWS_SET" == "1" ]]; then
  echo "--no-keypoint-sharding cannot be combined with --keypoint-roi-shard-rows." >&2
  exit 2
fi
if [[ "$NO_KEYPOINT_SHARDING" != "1" ]]; then
  if ! [[ "$KEYPOINT_ROI_SHARD_ROWS" =~ ^[1-9][0-9]*$ ]]; then
    echo "--keypoint-roi-shard-rows must be a positive integer." >&2
    exit 2
  fi
  if ! [[ "$KEYPOINT_FRAME_SHARD_ROWS" =~ ^[1-9][0-9]*$ ]]; then
    echo "--keypoint-frame-shard-rows must be a positive integer." >&2
    exit 2
  fi
fi
if [[ "$CPU" != "1" && -z "$DEVICE" && "$GPUS" != "0" ]]; then
  DEVICE="0"
fi
if [[ "$NO_PROGRESS_JSONL" != "1" && -z "$PROGRESS_JSONL_DIR" ]]; then
  PROGRESS_JSONL_DIR="${RUN_DIR}/progress_jsonl"
fi
if [[ "$NO_PROGRESS_JSONL" != "1" ]]; then
  mkdir -p "$PROGRESS_JSONL_DIR"
fi
if [[ -n "$ROI_CACHE_MANIFEST" && "$GPUS" != "0" && "$STAGE_ROI_CACHE_TO_SCRATCH" != "1" ]]; then
  {
    echo "Warning: GPU job will read --roi-cache-manifest directly from its source tier."
    echo "For large flat ROI caches, benchmarked policy recommends --stage-roi-cache-to-scratch."
  } >&2
fi

# Build per-recording args for run_keypoints_with_registry_model.
EXTRA_ARGS=(--registry "$REGISTRY" --top-k "$TOP_K")
if [[ -n "$SET_ID" ]]; then EXTRA_ARGS+=(--set-id "$SET_ID"); fi
if [[ "$REQUIRE_UNIQUE" == "1" ]]; then EXTRA_ARGS+=(--require-unique); fi
if [[ "$INCLUDE_NON_SUCCESS" == "1" ]]; then EXTRA_ARGS+=(--include-non-success); fi
if [[ -n "$CROP_RUN" ]]; then EXTRA_ARGS+=(--crop-run "$CROP_RUN"); fi
if [[ -n "$OUTPUT_PARENT" ]]; then EXTRA_ARGS+=(--output-parent "$OUTPUT_PARENT"); fi
if [[ -n "$POSE_SCHEMA" ]]; then EXTRA_ARGS+=(--pose-schema "$POSE_SCHEMA"); fi
if [[ "$NO_KEYPOINT_SHARDING" == "1" ]]; then
  EXTRA_ARGS+=(--no-keypoint-sharding)
else
  EXTRA_ARGS+=(--keypoint-roi-shard-rows "$KEYPOINT_ROI_SHARD_ROWS")
  EXTRA_ARGS+=(--keypoint-frame-shard-rows "$KEYPOINT_FRAME_SHARD_ROWS")
fi
if [[ -n "$DEVICE" ]]; then EXTRA_ARGS+=(--device "$DEVICE"); fi
if [[ -n "$IMGSZ" ]]; then EXTRA_ARGS+=(--imgsz "$IMGSZ"); fi
if [[ -n "$CONF" ]]; then EXTRA_ARGS+=(--conf "$CONF"); fi
if [[ -n "$IOU" ]]; then EXTRA_ARGS+=(--iou "$IOU"); fi
if [[ -n "$MAX_DET" ]]; then EXTRA_ARGS+=(--max-det "$MAX_DET"); fi
if [[ -n "$MASK_THRESHOLD" ]]; then EXTRA_ARGS+=(--mask-threshold "$MASK_THRESHOLD"); fi
if [[ -n "$ROI_CACHE_POLICY" ]]; then EXTRA_ARGS+=(--roi-cache-policy "$ROI_CACHE_POLICY"); fi
if [[ -n "$ROI_CACHE_DIR" ]]; then EXTRA_ARGS+=(--roi-cache-dir "$ROI_CACHE_DIR"); fi
if [[ -n "$ROI_CACHE_MANIFEST" ]]; then EXTRA_ARGS+=(--roi-cache-manifest "$ROI_CACHE_MANIFEST"); fi
if [[ "$STAGE_ROI_CACHE_TO_SCRATCH" == "1" ]]; then EXTRA_ARGS+=(--stage-roi-cache-to-scratch); fi
if [[ -n "$ROI_CACHE_STAGING_DIR" ]]; then EXTRA_ARGS+=(--roi-cache-staging-dir "$ROI_CACHE_STAGING_DIR"); fi
if [[ "$CPU" == "1" ]]; then EXTRA_ARGS+=(--cpu); fi
EXTRA_ARGS+=(--batch-size "$BATCH_SIZE_KP")

printf -v EXTRA_ARGS_SHELL '%q ' "${EXTRA_ARGS[@]}"
printf -v PROGRESS_JSONL_DIR_SHELL '%q' "$PROGRESS_JSONL_DIR"
printf -v PROGRESS_EVERY_BATCHES_SHELL '%q' "$PROGRESS_EVERY_BATCHES"
printf -v ROI_CACHE_STAGING_DIR_SHELL '%q' "$ROI_CACHE_STAGING_DIR"

scripts/py - "$RUN_DIR/manifest_summary.json" "$KEYPOINT_ROI_SHARD_ROWS" "$KEYPOINT_FRAME_SHARD_ROWS" "$NO_KEYPOINT_SHARDING" <<'PY'
import json
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
roi_rows = int(sys.argv[2])
frame_rows = int(sys.argv[3])
disabled = sys.argv[4] == "1"
payload = json.loads(summary_path.read_text(encoding="utf-8"))
payload["keypoint_storage"] = {
    "requested": {
        "keypoint_roi_shard_rows": None if disabled else roi_rows,
        "keypoint_frame_shard_rows": frame_rows,
        "no_keypoint_sharding": disabled,
    },
    "effective": {
        "keypoint_storage_layout": "regular_chunks_v1" if disabled else "indexed_sharding_v1",
        "keypoint_storage_policy": (
            "explicit_regular_chunks_override" if disabled else "default_indexed_sharding_v1"
        ),
        "keypoint_roi_shard_rows": None if disabled else roi_rows,
        "keypoint_frame_shard_rows": None if disabled else frame_rows,
    },
}
summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

JOB_SCRIPT="${RUN_DIR}/run_batch.sh"
cat > "$JOB_SCRIPT" <<JOBSCRIPT
#!/usr/bin/env bash
set -euo pipefail
RUN_DIR="\$1"
PROGRESS_JSONL_DIR=${PROGRESS_JSONL_DIR_SHELL}
PROGRESS_EVERY_BATCHES=${PROGRESS_EVERY_BATCHES_SHELL}
STAGE_ROI_CACHE_TO_SCRATCH=${STAGE_ROI_CACHE_TO_SCRATCH}
ROI_CACHE_STAGING_DIR=${ROI_CACHE_STAGING_DIR_SHELL}

cleanup_staged_roi_cache() {
  local status=\$?
  trap - EXIT INT TERM
  if [[ "\$STAGE_ROI_CACHE_TO_SCRATCH" == "1" ]]; then
    local user_name="\${USER:-\$(id -un)}"
    local job_id="\${LSB_JOBID:-}"
    local cleanup_dir=""
    if [[ -n "\$ROI_CACHE_STAGING_DIR" ]]; then
      cleanup_dir="\$ROI_CACHE_STAGING_DIR"
    elif [[ -n "\$job_id" ]]; then
      cleanup_dir="/scratch/\${user_name}/\${job_id}/palette_roi_cache_stage"
    fi
    local scratch_job_root="/scratch/\${user_name}/\${job_id}"
    if [[ -n "\$cleanup_dir" && -d "\$cleanup_dir" ]]; then
      if [[ -n "\$job_id" && "\$cleanup_dir" == "\$scratch_job_root/"* ]]; then
        echo "Cleaning staged ROI cache: \$cleanup_dir"
        rm -rf -- "\$cleanup_dir"
      else
        echo "Skipping staged ROI cache cleanup outside this LSF job scratch root: \$cleanup_dir" >&2
      fi
    fi
  fi
  exit "\$status"
}
trap cleanup_staged_roi_cache EXIT INT TERM

if [[ -z "\${LSB_JOBINDEX:-}" ]]; then
  echo "LSB_JOBINDEX not set; are you running under bsub array?" >&2
  exit 2
fi
BATCH_FILE="\${RUN_DIR}/batch_\$(printf '%04d' "\${LSB_JOBINDEX}").txt"
if [[ ! -f "\$BATCH_FILE" ]]; then
  echo "Missing batch file: \$BATCH_FILE" >&2
  exit 2
fi
mapfile -t zarr_paths < "\$BATCH_FILE"
if [[ "\${#zarr_paths[@]}" -eq 0 ]]; then
  echo "Empty batch file: \$BATCH_FILE"
  exit 0
fi
for zarr_path in "\${zarr_paths[@]}"; do
  [[ -z "\$zarr_path" ]] && continue
  # Derive recording_dir from zarr path (parent.parent if under zarr/ subdir).
  zarr_parent="\$(dirname "\$zarr_path")"
  if [[ "\$(basename "\$zarr_parent")" == "zarr" ]]; then
    recording_dir="\$(dirname "\$zarr_parent")"
  else
    recording_dir="\$zarr_parent"
  fi
  progress_args=()
  if [[ -n "\$PROGRESS_JSONL_DIR" ]]; then
    mkdir -p "\$PROGRESS_JSONL_DIR"
    safe_recording="\$(basename "\$recording_dir" | tr -c 'A-Za-z0-9_.-' '_')"
    progress_args=(--progress-jsonl "\${PROGRESS_JSONL_DIR}/\${safe_recording}.jsonl" --progress-every-batches "\$PROGRESS_EVERY_BATCHES")
  fi
  echo "=== Processing: \$recording_dir ==="
  scripts/py -m fisheye.utils.run_keypoints_with_registry_model --recording-dir "\$recording_dir" ${EXTRA_ARGS_SHELL} "\${progress_args[@]}" || {
    echo "FAILED: \$recording_dir" >&2
    continue
  }
done
JOBSCRIPT
chmod +x "$JOB_SCRIPT"

BSUB_ARGS=(-J "kp_batch[1-${batch_count}]%${MAX_ACTIVE}" -n "$NCORES" -R "rusage[mem=${MEM_GB}G]" -oo "${RUN_DIR}/%J_%I.out" -eo "${RUN_DIR}/%J_%I.err")
if [[ -n "$QUEUE" ]]; then
  BSUB_ARGS+=(-q "$QUEUE")
fi
if [[ "$GPUS" != "0" ]]; then
  BSUB_ARGS+=(-gpu "num=${GPUS}")
fi

BSUB_CMD="bsub"
for arg in "${BSUB_ARGS[@]}"; do
  BSUB_CMD+=" $(printf '%q' "$arg")"
done
BSUB_CMD+=" bash"
BSUB_CMD+=" $(printf '%q' "$JOB_SCRIPT")"
BSUB_CMD+=" $(printf '%q' "$RUN_DIR")"

printf -v KP_CMD 'scripts/py -m fisheye.utils.run_keypoints_with_registry_model --recording-dir <dir> %s' "$EXTRA_ARGS_SHELL"

echo "Run dir: $RUN_DIR"
echo "Source: $SOURCE"
echo "Root: $ROOT"
echo "Registry: $REGISTRY"
echo "Analysis zarrs: $analysis_count"
echo "Batch size: $BATCH_SIZE"
echo "Batches: $batch_count"
echo "Max active: $MAX_ACTIVE"
echo "Queue: ${QUEUE:-<default>}"
echo "Resources: ncores=$NCORES mem_gb=$MEM_GB gpus=$GPUS"
if [[ "$NO_KEYPOINT_SHARDING" == "1" ]]; then
  echo "Keypoint storage: regular_chunks_v1 (explicit opt-out)"
else
  echo "Keypoint storage: indexed_sharding_v1 roi_rows=$KEYPOINT_ROI_SHARD_ROWS frame_rows=$KEYPOINT_FRAME_SHARD_ROWS"
fi
echo "Manifest file: $RUN_DIR/recordings.txt"
echo "Batch files: $RUN_DIR/batch_*.txt"
echo "Progress JSONL dir: ${PROGRESS_JSONL_DIR:-<disabled>}"
echo "Per-recording command: $KP_CMD"
echo "Submit command: $BSUB_CMD"

if [[ "$DRY_RUN" == "1" ]]; then
  echo "Dry run only; no submission."
  exit 0
fi

if ! command -v bsub >/dev/null 2>&1; then
  echo "bsub not found in PATH. Is this an LSF cluster?" >&2
  exit 2
fi

bsub "${BSUB_ARGS[@]}" bash "$JOB_SCRIPT" "$RUN_DIR"
