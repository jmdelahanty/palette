#!/usr/bin/env bash
set -euo pipefail

ROOT="/nvme1/recordings"
BATCH_SIZE=10
MAX_ACTIVE=2
QUEUE=""
NCORES=4
MEM_GB=32
REGISTRY="/nvme1/palette_registry.sqlite"
SET_ID=""
TOP_K=5
REQUIRE_UNIQUE=0
INCLUDE_NON_SUCCESS=0
METHOD=""
CROP_RUN=""
KEYPOINTS_RUN=""
BATCH_SIZE_EM=128
DEVICE=""
IMGSZ=""
CONF=""
IOU=""
MAX_DET=""
MASK_THRESHOLD=""
ADAPTIVE_SCALE=""
ADAPTIVE_CAP=""
NO_RETINA_MASKS=0
PROTO_UPSAMPLE_FACTOR=""
LEGACY_MASKS=0
VERBOSE=0
LABEL_MODE=""
WRITE_BINARY_MASKS=0
NO_USE_CROP=0
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
Usage: submit_eye_masks_batches_bsub.sh [options]

Options:
  --root PATH               Root recordings directory (default: /nvme1/recordings)
  --batch-size N            Analysis zarrs per batch job (default: 10)
  --max-active N            Max concurrent jobs in array (default: 2)
  --queue NAME              LSF queue name
  --ncores N                Cores per job (default: 4)
  --mem-gb N                Memory per job in GB (default: 32)
  --registry PATH           Registry sqlite path (default: /nvme1/palette_registry.sqlite)
  --set-id ID               Optional eye mask model set filter for registry model resolution
  --top-k N                 Candidate provenance depth (default: 5)
  --require-unique          Fail if top model scores tie
  --include-non-success     Include non-success runs in model resolution
  --method {yolo,unet}      Eye mask method override
  --crop-run NAME           Optional explicit crop run name
  --keypoints-run NAME      Optional explicit keypoints run name
  --batch-size-em N         Eye mask inference batch size (default: 128)
  --device DEVICE           Torch device override
  --imgsz N                 YOLO image size override
  --conf FLOAT              YOLO confidence threshold override
  --iou FLOAT               YOLO IoU threshold override
  --max-det N               YOLO max detections override
  --mask-threshold FLOAT    YOLO mask threshold override
  --adaptive-scale FLOAT    YOLO adaptive scale override
  --adaptive-cap FLOAT      YOLO adaptive cap override
  --no-retina-masks         YOLO: disable retina masks
  --proto-upsample-factor N YOLO proto upsample factor
  --legacy-masks            YOLO legacy mask conversion
  --verbose                 YOLO verbose Ultralytics output
  --label-mode {union,lr}   U-Net label-mode override
  --write-binary-masks      U-Net: write thresholded masks
  --no-use-crop             U-Net: do not pass --use-crop
  --cpu                     Force CPU inference
  --overwrite               Run eye masks even if eye_masks run already exists
  --log-dir PATH            Submission logs (default: <root>/logs/run_eye_masks_batch/bsub_submissions)
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
    --registry) REGISTRY="$2"; shift 2;;
    --set-id) SET_ID="$2"; shift 2;;
    --top-k) TOP_K="$2"; shift 2;;
    --require-unique) REQUIRE_UNIQUE=1; shift;;
    --include-non-success) INCLUDE_NON_SUCCESS=1; shift;;
    --method) METHOD="$2"; shift 2;;
    --crop-run) CROP_RUN="$2"; shift 2;;
    --keypoints-run) KEYPOINTS_RUN="$2"; shift 2;;
    --batch-size-em) BATCH_SIZE_EM="$2"; shift 2;;
    --device) DEVICE="$2"; shift 2;;
    --imgsz) IMGSZ="$2"; shift 2;;
    --conf) CONF="$2"; shift 2;;
    --iou) IOU="$2"; shift 2;;
    --max-det) MAX_DET="$2"; shift 2;;
    --mask-threshold) MASK_THRESHOLD="$2"; shift 2;;
    --adaptive-scale) ADAPTIVE_SCALE="$2"; shift 2;;
    --adaptive-cap) ADAPTIVE_CAP="$2"; shift 2;;
    --no-retina-masks) NO_RETINA_MASKS=1; shift;;
    --proto-upsample-factor) PROTO_UPSAMPLE_FACTOR="$2"; shift 2;;
    --legacy-masks) LEGACY_MASKS=1; shift;;
    --verbose) VERBOSE=1; shift;;
    --label-mode) LABEL_MODE="$2"; shift 2;;
    --write-binary-masks) WRITE_BINARY_MASKS=1; shift;;
    --no-use-crop) NO_USE_CROP=1; shift;;
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
  LOG_DIR="${ROOT}/logs/run_eye_masks_batch/bsub_submissions"
fi

if [[ -n "$RUN_ID_OVERRIDE" ]]; then
  RUN_ID="$RUN_ID_OVERRIDE"
else
  RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
fi
RUN_DIR="${LOG_DIR}/em_${RUN_ID}"

if [[ -e "$RUN_DIR" ]]; then
  echo "Run directory already exists: $RUN_DIR" >&2
  echo "Choose a different --run-id or remove the existing run directory." >&2
  exit 2
fi
if ! mkdir -p "$RUN_DIR" 2>/dev/null; then
  if [[ "$DRY_RUN" == "1" ]]; then
    FALLBACK_LOG_DIR="${TMPDIR:-/tmp}/palette/run_eye_masks_batch/bsub_submissions"
    RUN_DIR="${FALLBACK_LOG_DIR}/em_${RUN_ID}"
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

  scripts/py -m fisheye.utils.run_eye_masks_batch "${DISCOVER_ARGS[@]}" > "${RUN_DIR}/discovered_paths.txt"

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

# Build per-recording args for run_eye_masks_with_registry_model.
EXTRA_ARGS=(--registry "$REGISTRY" --top-k "$TOP_K")
if [[ -n "$SET_ID" ]]; then EXTRA_ARGS+=(--set-id "$SET_ID"); fi
if [[ "$REQUIRE_UNIQUE" == "1" ]]; then EXTRA_ARGS+=(--require-unique); fi
if [[ "$INCLUDE_NON_SUCCESS" == "1" ]]; then EXTRA_ARGS+=(--include-non-success); fi
if [[ -n "$METHOD" ]]; then EXTRA_ARGS+=(--method "$METHOD"); fi
if [[ -n "$CROP_RUN" ]]; then EXTRA_ARGS+=(--crop-run "$CROP_RUN"); fi
if [[ -n "$KEYPOINTS_RUN" ]]; then EXTRA_ARGS+=(--keypoints-run "$KEYPOINTS_RUN"); fi
if [[ -n "$DEVICE" ]]; then EXTRA_ARGS+=(--device "$DEVICE"); fi
if [[ -n "$IMGSZ" ]]; then EXTRA_ARGS+=(--imgsz "$IMGSZ"); fi
if [[ -n "$CONF" ]]; then EXTRA_ARGS+=(--conf "$CONF"); fi
if [[ -n "$IOU" ]]; then EXTRA_ARGS+=(--iou "$IOU"); fi
if [[ -n "$MAX_DET" ]]; then EXTRA_ARGS+=(--max-det "$MAX_DET"); fi
if [[ -n "$MASK_THRESHOLD" ]]; then EXTRA_ARGS+=(--mask-threshold "$MASK_THRESHOLD"); fi
if [[ -n "$ADAPTIVE_SCALE" ]]; then EXTRA_ARGS+=(--adaptive-scale "$ADAPTIVE_SCALE"); fi
if [[ -n "$ADAPTIVE_CAP" ]]; then EXTRA_ARGS+=(--adaptive-cap "$ADAPTIVE_CAP"); fi
if [[ "$NO_RETINA_MASKS" == "1" ]]; then EXTRA_ARGS+=(--no-retina-masks); fi
if [[ -n "$PROTO_UPSAMPLE_FACTOR" ]]; then EXTRA_ARGS+=(--proto-upsample-factor "$PROTO_UPSAMPLE_FACTOR"); fi
if [[ "$LEGACY_MASKS" == "1" ]]; then EXTRA_ARGS+=(--legacy-masks); fi
if [[ "$VERBOSE" == "1" ]]; then EXTRA_ARGS+=(--verbose); fi
if [[ -n "$LABEL_MODE" ]]; then EXTRA_ARGS+=(--label-mode "$LABEL_MODE"); fi
if [[ "$WRITE_BINARY_MASKS" == "1" ]]; then EXTRA_ARGS+=(--write-binary-masks); fi
if [[ "$NO_USE_CROP" == "1" ]]; then EXTRA_ARGS+=(--no-use-crop); fi
if [[ "$CPU" == "1" ]]; then EXTRA_ARGS+=(--device cpu); fi
EXTRA_ARGS+=(--batch-size "$BATCH_SIZE_EM")

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
  echo "=== Processing: \$recording_dir ==="
  scripts/py -m fisheye.utils.run_eye_masks_with_registry_model --recording-dir "\$recording_dir" ${EXTRA_ARGS_SHELL}|| {
    echo "FAILED: \$recording_dir" >&2
    continue
  }
done
JOBSCRIPT
chmod +x "$JOB_SCRIPT"

BSUB_ARGS=(-J "em_batch[1-${batch_count}]%${MAX_ACTIVE}" -n "$NCORES" -R "rusage[mem=${MEM_GB}G]" -oo "${RUN_DIR}/%J_%I.out" -eo "${RUN_DIR}/%J_%I.err")
if [[ -n "$QUEUE" ]]; then
  BSUB_ARGS+=(-q "$QUEUE")
fi

printf -v BSUB_CMD 'bsub %q ' "${BSUB_ARGS[@]}"
BSUB_CMD+="bash "
BSUB_CMD+="$(printf '%q' "$JOB_SCRIPT") "
BSUB_CMD+="$(printf '%q' "$RUN_DIR")"

printf -v EM_CMD 'scripts/py -m fisheye.utils.run_eye_masks_with_registry_model --recording-dir <dir> %s' "$EXTRA_ARGS_SHELL"

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
echo "Per-recording command: $EM_CMD"
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
