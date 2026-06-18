#!/usr/bin/env bash
set -euo pipefail

ROOT="/groups/johnson/johnsonlab/jeremy/recordings"
BATCH_SIZE=10
MAX_ACTIVE=2
QUEUE=""
NCORES=4
MEM_GB=32
REGISTRY="${PALETTE_REGISTRY_PATH:-/groups/johnson/johnsonlab/jeremy/registries/palette_registry.sqlite}"
CONFIG=""
SOURCE_TYPE=""
SOURCE_PATH=""
PREFERRED_POLICY=""
SCHEDULER=""
NUM_WORKERS=""
ACCELERATION=""
EXTERNAL_WRITE_BACKEND=""
EXTERNAL_ROI_STORAGE=""
EXTERNAL_USE_SHARDING=""
EXTERNAL_ROI_CHUNK_SIZE=""
EXTERNAL_ROI_SHARD_SIZE=""
EXTERNAL_GPU_CHUNK_FRAMES=""
REQUIRE_KVIKIO=0
NO_GPU=0
FORCE_CPU=0
FORCE_NEW=0
VERBOSE=0
LOG_DIR=""
RUN_ID_OVERRIDE=""
DRY_RUN=0
SOURCE="filesystem"
RIG_ID=""
ARENA_ID=""
CAMERA_ID=""
PATH_CONTAINS=""
ZAR_USE="analysis"

usage() {
  cat <<'USAGE'
Usage: submit_crop_batches_bsub.sh [options]

Options:
  --root PATH               Root recordings directory (default: /groups/johnson/johnsonlab/jeremy/recordings)
  --batch-size N            Analysis zarrs per batch job (default: 10)
  --max-active N            Max concurrent jobs in array (default: 2)
  --queue NAME              LSF queue name
  --ncores N                Cores per job (default: 4)
  --mem-gb N                Memory per job in GB (default: 32)
  --registry PATH           Registry sqlite path (default: $PALETTE_REGISTRY_PATH or /groups/johnson/johnsonlab/jeremy/registries/palette_registry.sqlite)
  --config PATH             Crop config YAML path
  --source-type TYPE        Detection source type (detect/filtered/interpolated/manual/preferred/auto)
  --source-path PATH        Explicit detection source path
  --preferred-policy POLICY Policy for preferred/auto source selection
  --scheduler NAME          Dask scheduler (processes/threads/distributed)
  --num-workers N           Dask worker count
  --acceleration {auto,gpu,cpu}  GPU/CPU acceleration mode
  --external-write-backend {standard,kvikio}  Write backend for external-video
  --external-roi-storage {compressed,uncompressed}  ROI storage mode
  --external-use-sharding   Enable sharding for external-video ROI writes
  --external-roi-chunk-size N   Detection-axis chunk length
  --external-roi-shard-size N   Detection-axis shard length
  --external-gpu-chunk-frames N Frame count decoded per GPU crop chunk
  --require-kvikio          Fail if kvikIO GDS writes cannot be enabled
  --no-gpu                  Disable GPU
  --force-cpu               Force CPU
  --force-new               Always create new crop run (disables skip-existing)
  --verbose                 Verbose logging
  --zarr-use {analysis,training,any}  Zarr use filter (default: analysis)
  --log-dir PATH            Submission logs (default: <root>/logs/crop_batch/bsub_submissions)
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
    --ncores) NCORES="$2"; shift 2;;
    --mem-gb) MEM_GB="$2"; shift 2;;
    --registry) REGISTRY="$2"; shift 2;;
    --config) CONFIG="$2"; shift 2;;
    --source-type) SOURCE_TYPE="$2"; shift 2;;
    --source-path) SOURCE_PATH="$2"; shift 2;;
    --preferred-policy) PREFERRED_POLICY="$2"; shift 2;;
    --scheduler) SCHEDULER="$2"; shift 2;;
    --num-workers) NUM_WORKERS="$2"; shift 2;;
    --acceleration) ACCELERATION="$2"; shift 2;;
    --external-write-backend) EXTERNAL_WRITE_BACKEND="$2"; shift 2;;
    --external-roi-storage) EXTERNAL_ROI_STORAGE="$2"; shift 2;;
    --external-use-sharding) EXTERNAL_USE_SHARDING="1"; shift;;
    --external-roi-chunk-size) EXTERNAL_ROI_CHUNK_SIZE="$2"; shift 2;;
    --external-roi-shard-size) EXTERNAL_ROI_SHARD_SIZE="$2"; shift 2;;
    --external-gpu-chunk-frames) EXTERNAL_GPU_CHUNK_FRAMES="$2"; shift 2;;
    --require-kvikio) REQUIRE_KVIKIO=1; shift;;
    --no-gpu) NO_GPU=1; shift;;
    --force-cpu) FORCE_CPU=1; shift;;
    --force-new) FORCE_NEW=1; shift;;
    --verbose) VERBOSE=1; shift;;
    --zarr-use) ZAR_USE="$2"; shift 2;;
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

if [[ -z "$LOG_DIR" ]]; then
  LOG_DIR="${ROOT}/logs/crop_batch/bsub_submissions"
fi

if [[ -n "$RUN_ID_OVERRIDE" ]]; then
  RUN_ID="$RUN_ID_OVERRIDE"
else
  RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
fi
RUN_DIR="${LOG_DIR}/crop_${RUN_ID}"

if [[ -e "$RUN_DIR" ]]; then
  echo "Run directory already exists: $RUN_DIR" >&2
  echo "Choose a different --run-id or remove the existing run directory." >&2
  exit 2
fi
if ! mkdir -p "$RUN_DIR" 2>/dev/null; then
  if [[ "$DRY_RUN" == "1" ]]; then
    FALLBACK_LOG_DIR="${TMPDIR:-/tmp}/palette/crop_batch/bsub_submissions"
    RUN_DIR="${FALLBACK_LOG_DIR}/crop_${RUN_ID}"
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
  DISCOVER_ARGS=(--source registry --emit-paths --registry "$REGISTRY" --zarr-use "$ZAR_USE")
  if [[ "$FORCE_NEW" == "1" ]]; then DISCOVER_ARGS+=(--force-new); fi
  if [[ -n "$RIG_ID" ]]; then DISCOVER_ARGS+=(--rig-id "$RIG_ID"); fi
  if [[ -n "$ARENA_ID" ]]; then DISCOVER_ARGS+=(--arena-id "$ARENA_ID"); fi
  if [[ -n "$CAMERA_ID" ]]; then DISCOVER_ARGS+=(--camera-id "$CAMERA_ID"); fi
  if [[ -n "$PATH_CONTAINS" ]]; then DISCOVER_ARGS+=(--path-contains "$PATH_CONTAINS"); fi
  DISCOVER_ARGS+=("$ROOT")

  scripts/py -m fisheye.utils.crop_batch "${DISCOVER_ARGS[@]}" > "${RUN_DIR}/discovered_paths.txt"

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

# Build per-batch args for crop_batch.
EXTRA_ARGS=(--apply --zarr-use "$ZAR_USE")
if [[ -n "$CONFIG" ]]; then EXTRA_ARGS+=(--config "$CONFIG"); fi
if [[ -n "$SOURCE_TYPE" ]]; then EXTRA_ARGS+=(--source-type "$SOURCE_TYPE"); fi
if [[ -n "$SOURCE_PATH" ]]; then EXTRA_ARGS+=(--source-path "$SOURCE_PATH"); fi
if [[ -n "$PREFERRED_POLICY" ]]; then EXTRA_ARGS+=(--preferred-policy "$PREFERRED_POLICY"); fi
if [[ -n "$SCHEDULER" ]]; then EXTRA_ARGS+=(--scheduler "$SCHEDULER"); fi
if [[ -n "$NUM_WORKERS" ]]; then EXTRA_ARGS+=(--num-workers "$NUM_WORKERS"); fi
if [[ -n "$ACCELERATION" ]]; then EXTRA_ARGS+=(--acceleration "$ACCELERATION"); fi
if [[ -n "$EXTERNAL_WRITE_BACKEND" ]]; then EXTRA_ARGS+=(--external-write-backend "$EXTERNAL_WRITE_BACKEND"); fi
if [[ -n "$EXTERNAL_ROI_STORAGE" ]]; then EXTRA_ARGS+=(--external-roi-storage "$EXTERNAL_ROI_STORAGE"); fi
if [[ "$EXTERNAL_USE_SHARDING" == "1" ]]; then EXTRA_ARGS+=(--external-use-sharding); fi
if [[ -n "$EXTERNAL_ROI_CHUNK_SIZE" ]]; then EXTRA_ARGS+=(--external-roi-chunk-size "$EXTERNAL_ROI_CHUNK_SIZE"); fi
if [[ -n "$EXTERNAL_ROI_SHARD_SIZE" ]]; then EXTRA_ARGS+=(--external-roi-shard-size "$EXTERNAL_ROI_SHARD_SIZE"); fi
if [[ -n "$EXTERNAL_GPU_CHUNK_FRAMES" ]]; then EXTRA_ARGS+=(--external-gpu-chunk-frames "$EXTERNAL_GPU_CHUNK_FRAMES"); fi
if [[ "$REQUIRE_KVIKIO" == "1" ]]; then EXTRA_ARGS+=(--require-kvikio); fi
if [[ "$NO_GPU" == "1" ]]; then EXTRA_ARGS+=(--no-gpu); fi
if [[ "$FORCE_CPU" == "1" ]]; then EXTRA_ARGS+=(--force-cpu); fi
if [[ "$FORCE_NEW" == "1" ]]; then EXTRA_ARGS+=(--force-new); fi
if [[ "$VERBOSE" == "1" ]]; then EXTRA_ARGS+=(--verbose); fi

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
scripts/py -m fisheye.utils.crop_batch ${EXTRA_ARGS_SHELL}"\${recs[@]}"
JOBSCRIPT
chmod +x "$JOB_SCRIPT"

BSUB_ARGS=(-J "crop_batch[1-${batch_count}]%${MAX_ACTIVE}" -n "$NCORES" -R "rusage[mem=${MEM_GB}G]" -oo "${RUN_DIR}/%J_%I.out" -eo "${RUN_DIR}/%J_%I.err")
if [[ -n "$QUEUE" ]]; then
  BSUB_ARGS+=(-q "$QUEUE")
fi

printf -v BSUB_CMD 'bsub %q ' "${BSUB_ARGS[@]}"
BSUB_CMD+="bash "
BSUB_CMD+="$(printf '%q' "$JOB_SCRIPT") "
BSUB_CMD+="$(printf '%q' "$RUN_DIR")"

printf -v CROP_CMD 'scripts/py -m fisheye.utils.crop_batch %s<batch_zarr_paths>' "$EXTRA_ARGS_SHELL"

echo "Run dir: $RUN_DIR"
echo "Source: $SOURCE"
echo "Root: $ROOT"
echo "Registry: $REGISTRY"
echo "Analysis zarrs: $analysis_count"
echo "Batch size: $BATCH_SIZE"
echo "Batches: $batch_count"
echo "Max active: $MAX_ACTIVE"
echo "Queue: ${QUEUE:-<default>}"
echo "Resources: ncores=$NCORES mem_gb=$MEM_GB"
echo "Manifest file: $RUN_DIR/recordings.txt"
echo "Batch files: $RUN_DIR/batch_*.txt"
echo "Per-batch command: $CROP_CMD"
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
