#!/usr/bin/env bash
set -euo pipefail

ROOT="/groups/johnson/johnsonlab/jeremy/recordings"
BATCH_SIZE=10
MAX_ACTIVE=2
QUEUE=""
GPU_SPEC=""
NCORES=4
MEM_GB=16
CONFIG="configs/fisheye/default.yaml"
REGISTRY="${PALETTE_REGISTRY_PATH:-/groups/johnson/johnsonlab/jeremy/registries/palette_registry.sqlite}"
MODEL=""
DECODE_BACKEND=""
RESIZE_DIMS=()
DETECT_ROW_SHARD_ROWS=""
DETECT_FRAME_SHARD_ROWS=""
NO_DETECT_SHARDING=0
SCHEDULER="threads"
NUM_WORKERS=""
SET_ID=""
TOP_K=5
REQUIRE_UNIQUE=0
INCLUDE_NON_SUCCESS=0
REQUIRE_TUNING=0
OVERWRITE=0
LOG_DIR=""
RUN_ID_OVERRIDE=""
DRY_RUN=0
SOURCE="filesystem"
RIG_ID=""
ARENA_ID=""
CAMERA_ID=""
PATH_CONTAINS=""

usage() {
  cat <<'USAGE'
Usage: submit_detect_batches_bsub.sh [options]

Options:
  --root PATH               Root recordings directory (default: /groups/johnson/johnsonlab/jeremy/recordings)
  --batch-size N            Analysis zarrs per batch job (default: 10)
  --max-active N            Max concurrent jobs in array (default: 2)
  --queue NAME              LSF queue name
  --gpu SPEC                Optional LSF GPU resource spec, e.g. 'num=1'
  --ncores N                Cores per job (default: 4)
  --mem-gb N                Memory per job in GB (default: 16)
  --config PATH             Detect config path (default: configs/fisheye/default.yaml)
  --registry PATH           Registry sqlite path (default: $PALETTE_REGISTRY_PATH or /groups/johnson/johnsonlab/jeremy/registries/palette_registry.sqlite)
  --model PATH              Explicit detect model path; bypass registry model resolution
  --decode-backend NAME     Decode backend passed to run_detections_batch
  --resize-dims H W         Canonical inference size passed to run_detections_batch
  --detect-row-shard-rows N Override default detection-row outer shard rows
  --detect-frame-shard-rows N
                            Outer rows for frame-count arrays when sharding is enabled
  --no-detect-sharding      Use ordinary chunks for YOLO detection outputs
  --scheduler NAME          Legacy pass-through option for batch runner compatibility
  --num-workers N           Legacy pass-through option for batch runner compatibility
  --set-id ID               Optional detect set filter for registry model resolution
  --top-k N                 Candidate provenance depth (default: 5)
  --require-unique          Fail if top model scores tie
  --include-non-success     Include non-success runs in model resolution
  --require-tuning          Skip zarrs without detection_tuning
  --overwrite               Run detection even if detect_runs/latest exists
  --log-dir PATH            Submission logs (default: <root>/logs/run_detections_batch/bsub_submissions)
  --run-id ID               Optional stable run id for deterministic reruns
  --dry-run                 Print manifests + commands; do not submit
  --source {filesystem,registry}  Discovery source for analysis zarrs (default: filesystem)
  --rig-id ID               Filter by rig_id (registry source only)
  --arena-id ID             Filter by arena_id (registry source only)
  --camera-id ID            Filter by camera_id (registry source only)
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
    --gpu) GPU_SPEC="$2"; shift 2;;
    --ncores) NCORES="$2"; shift 2;;
    --mem-gb) MEM_GB="$2"; shift 2;;
    --config) CONFIG="$2"; shift 2;;
    --registry) REGISTRY="$2"; shift 2;;
    --model) MODEL="$2"; shift 2;;
    --decode-backend) DECODE_BACKEND="$2"; shift 2;;
    --resize-dims) RESIZE_DIMS=("$2" "$3"); shift 3;;
    --detect-row-shard-rows) DETECT_ROW_SHARD_ROWS="$2"; shift 2;;
    --detect-frame-shard-rows) DETECT_FRAME_SHARD_ROWS="$2"; shift 2;;
    --no-detect-sharding) NO_DETECT_SHARDING=1; shift;;
    --scheduler) SCHEDULER="$2"; shift 2;;
    --num-workers) NUM_WORKERS="$2"; shift 2;;
    --set-id) SET_ID="$2"; shift 2;;
    --top-k) TOP_K="$2"; shift 2;;
    --require-unique) REQUIRE_UNIQUE=1; shift;;
    --include-non-success) INCLUDE_NON_SUCCESS=1; shift;;
    --require-tuning) REQUIRE_TUNING=1; shift;;
    --overwrite) OVERWRITE=1; shift;;
    --log-dir) LOG_DIR="$2"; shift 2;;
    --run-id) RUN_ID_OVERRIDE="$2"; shift 2;;
    --dry-run) DRY_RUN=1; shift;;
    --source) SOURCE="$2"; shift 2;;
    --rig-id) RIG_ID="$2"; shift 2;;
    --arena-id) ARENA_ID="$2"; shift 2;;
    --camera-id) CAMERA_ID="$2"; shift 2;;
    --path-contains) PATH_CONTAINS="$2"; shift 2;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown arg: $1"; usage; exit 2;;
  esac
done

for value in "$DETECT_ROW_SHARD_ROWS" "$DETECT_FRAME_SHARD_ROWS"; do
  if [[ -n "$value" && ! "$value" =~ ^[1-9][0-9]*$ ]]; then
    echo "Shard row counts must be positive integers; got: $value" >&2
    exit 2
  fi
done
if [[ "$NO_DETECT_SHARDING" == "1" && ( -n "$DETECT_ROW_SHARD_ROWS" || -n "$DETECT_FRAME_SHARD_ROWS" ) ]]; then
  echo "--no-detect-sharding cannot be combined with explicit shard row counts" >&2
  exit 2
fi

if [[ -z "$LOG_DIR" ]]; then
  LOG_DIR="${ROOT}/logs/run_detections_batch/bsub_submissions"
fi

if [[ -n "$RUN_ID_OVERRIDE" ]]; then
  RUN_ID="$RUN_ID_OVERRIDE"
else
  RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
fi
RUN_DIR="${LOG_DIR}/detect_${RUN_ID}"

if [[ -e "$RUN_DIR" ]]; then
  echo "Run directory already exists: $RUN_DIR" >&2
  echo "Choose a different --run-id or remove the existing run directory." >&2
  exit 2
fi
if ! mkdir -p "$RUN_DIR" 2>/dev/null; then
  if [[ "$DRY_RUN" == "1" ]]; then
    FALLBACK_LOG_DIR="${TMPDIR:-/tmp}/palette/run_detections_batch/bsub_submissions"
    RUN_DIR="${FALLBACK_LOG_DIR}/detect_${RUN_ID}"
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
  if [[ -n "$CAMERA_ID" ]]; then DISCOVER_ARGS+=(--camera-id "$CAMERA_ID"); fi
  if [[ -n "$PATH_CONTAINS" ]]; then DISCOVER_ARGS+=(--path-contains "$PATH_CONTAINS"); fi
  DISCOVER_ARGS+=("$ROOT")

  scripts/py -m fisheye.utils.run_detections_batch "${DISCOVER_ARGS[@]}" > "${RUN_DIR}/discovered_paths.txt"

  scripts/py - "${RUN_DIR}/discovered_paths.txt" "$BATCH_SIZE" "$RUN_DIR" "$SOURCE" <<'PY'
import json, math, sys
from pathlib import Path

paths_file = Path(sys.argv[1])
batch_size = int(sys.argv[2])
run_dir = Path(sys.argv[3])
source = sys.argv[4]

raw_lines = [line.strip() for line in paths_file.read_text(encoding="utf-8").splitlines() if line.strip()]
zarr_paths = [line for line in raw_lines if line.endswith(".zarr")]
ignored_lines = [line for line in raw_lines if not line.endswith(".zarr")]
run_dir.mkdir(parents=True, exist_ok=True)

if ignored_lines:
    (run_dir / "discovery_ignored_lines.txt").write_text(
        "\n".join(ignored_lines) + "\n",
        encoding="utf-8",
    )

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

EXTRA_ARGS=(--apply --no-dask-progress --registry "$REGISTRY" --top-k "$TOP_K" --config "$CONFIG")
if [[ -n "$MODEL" ]]; then
  EXTRA_ARGS+=(--model "$MODEL")
fi
if [[ -n "$DECODE_BACKEND" ]]; then
  EXTRA_ARGS+=(--decode-backend "$DECODE_BACKEND")
fi
if [[ "${#RESIZE_DIMS[@]}" -gt 0 ]]; then
  EXTRA_ARGS+=(--resize-dims "${RESIZE_DIMS[@]}")
fi
if [[ -n "$DETECT_ROW_SHARD_ROWS" ]]; then
  EXTRA_ARGS+=(--detect-row-shard-rows "$DETECT_ROW_SHARD_ROWS")
fi
if [[ -n "$DETECT_FRAME_SHARD_ROWS" ]]; then
  EXTRA_ARGS+=(--detect-frame-shard-rows "$DETECT_FRAME_SHARD_ROWS")
fi
if [[ "$NO_DETECT_SHARDING" == "1" ]]; then
  EXTRA_ARGS+=(--no-detect-sharding)
fi
if [[ -n "$SCHEDULER" ]]; then
  EXTRA_ARGS+=(--scheduler "$SCHEDULER")
fi
if [[ -n "$NUM_WORKERS" ]]; then
  EXTRA_ARGS+=(--num-workers "$NUM_WORKERS")
fi
if [[ "$REQUIRE_TUNING" == "1" ]]; then
  EXTRA_ARGS+=(--require-tuning)
fi
if [[ "$OVERWRITE" == "1" ]]; then
  EXTRA_ARGS+=(--overwrite)
fi
if [[ -n "$SET_ID" ]]; then
  EXTRA_ARGS+=(--set-id "$SET_ID")
fi
if [[ "$REQUIRE_UNIQUE" == "1" ]]; then
  EXTRA_ARGS+=(--require-unique)
fi
if [[ "$INCLUDE_NON_SUCCESS" == "1" ]]; then
  EXTRA_ARGS+=(--include-non-success)
fi

printf -v EXTRA_ARGS_SHELL '%q ' "${EXTRA_ARGS[@]}"

JOB_SCRIPT="${RUN_DIR}/run_batch.sh"
cat > "$JOB_SCRIPT" <<JOBSCRIPT
#!/usr/bin/env bash
set -euo pipefail
RUN_DIR="\$1"
if [[ -z "\${LSB_JOBINDEX:-}" ]]; then
  echo "LSB_JOBINDEX not set; are you running under bsub array?" >&2
  exit 2
fi
BATCH_FILE="\${RUN_DIR}/batch_\$(printf '%04d' "\${LSB_JOBINDEX}").txt"
if [[ ! -f "\$BATCH_FILE" ]]; then
  echo "Missing batch file: \$BATCH_FILE" >&2
  exit 2
fi
mapfile -t recs < "\$BATCH_FILE"
if [[ "\${#recs[@]}" -eq 0 ]]; then
  echo "Empty batch file: \$BATCH_FILE"
  exit 0
fi
scripts/py -m fisheye.utils.run_detections_batch ${EXTRA_ARGS_SHELL}"\${recs[@]}"
JOBSCRIPT
chmod +x "$JOB_SCRIPT"

BSUB_ARGS=(-J "detect_batch[1-${batch_count}]%${MAX_ACTIVE}" -n "$NCORES" -R "rusage[mem=${MEM_GB}G]" -oo "${RUN_DIR}/%J_%I.out" -eo "${RUN_DIR}/%J_%I.err")
if [[ -n "$QUEUE" ]]; then
  BSUB_ARGS+=(-q "$QUEUE")
fi
if [[ -n "$GPU_SPEC" ]]; then
  BSUB_ARGS+=(-gpu "$GPU_SPEC")
fi

printf -v BSUB_ARGS_SHELL '%q ' "${BSUB_ARGS[@]}"
BSUB_CMD="bsub ${BSUB_ARGS_SHELL}bash "
BSUB_CMD+="$(printf '%q' "$JOB_SCRIPT") "
BSUB_CMD+="$(printf '%q' "$RUN_DIR")"

printf -v DETECT_CMD 'scripts/py -m fisheye.utils.run_detections_batch %s<batch_zarr_paths>' "$EXTRA_ARGS_SHELL"

echo "Run dir: $RUN_DIR"
echo "Source: $SOURCE"
echo "Root: $ROOT"
echo "Registry: $REGISTRY"
echo "Model: ${MODEL:-<registry resolution>}"
echo "Decode backend: ${DECODE_BACKEND:-<runner default>}"
if [[ "${#RESIZE_DIMS[@]}" -gt 0 ]]; then
  echo "Resize dims: ${RESIZE_DIMS[0]} ${RESIZE_DIMS[1]}"
else
  echo "Resize dims: <runner/config default>"
fi
if [[ "$NO_DETECT_SHARDING" == "1" ]]; then
  echo "Detect storage: regular chunks (explicit override)"
else
  echo "Detect row shard rows: ${DETECT_ROW_SHARD_ROWS:-<runner default: 131072>}"
  echo "Detect frame shard rows: ${DETECT_FRAME_SHARD_ROWS:-<runner default: 131072>}"
fi
echo "Analysis zarrs: $analysis_count"
echo "Batch size: $BATCH_SIZE"
echo "Batches: $batch_count"
echo "Max active: $MAX_ACTIVE"
echo "Queue: ${QUEUE:-<default>}"
echo "GPU: ${GPU_SPEC:-<none>}"
echo "Resources: ncores=$NCORES mem_gb=$MEM_GB"
echo "Manifest file: $RUN_DIR/recordings.txt"
echo "Batch files: $RUN_DIR/batch_*.txt"
echo "Per-batch command: $DETECT_CMD"
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
