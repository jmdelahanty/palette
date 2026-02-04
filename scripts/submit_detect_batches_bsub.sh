#!/usr/bin/env bash
set -euo pipefail

ROOT="/nvme1/recordings"
BATCH_SIZE=10
MAX_ACTIVE=2
QUEUE=""
NCORES=4
MEM_GB=16
CONFIG="configs/fisheye/default.yaml"
SCHEDULER="threads"
NUM_WORKERS=""
LOG_DIR=""
REQUIRE_TUNING=0
OVERWRITE=0
DRY_RUN=0

usage() {
  cat <<'USAGE'
Usage: submit_detect_batches_bsub.sh [options]

Options:
  --root PATH           Root recordings directory (default: /nvme1/recordings)
  --batch-size N        Recordings per batch job (default: 10)
  --max-active N        Max concurrent jobs in array (default: 2)
  --queue NAME          LSF queue name
  --ncores N            Cores per job (default: 4)
  --mem-gb N            Memory per job in GB (default: 16)
  --config PATH         Pipeline config (default: configs/fisheye/default.yaml)
  --scheduler NAME      Dask scheduler for detect (threads|processes|single-threaded)
  --num-workers N       Dask worker hint for detect
  --require-tuning      Skip zarrs without detection_tuning
  --overwrite           Run detection even if detect_runs/latest exists
  --log-dir PATH        Directory for batch logs (default: <root>/logs/run_detections_batch/bsub_submissions)
  --dry-run             Print what would be submitted
  -h, --help            Show this message
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
    --config) CONFIG="$2"; shift 2;;
    --scheduler) SCHEDULER="$2"; shift 2;;
    --num-workers) NUM_WORKERS="$2"; shift 2;;
    --require-tuning) REQUIRE_TUNING=1; shift;;
    --overwrite) OVERWRITE=1; shift;;
    --log-dir) LOG_DIR="$2"; shift 2;;
    --dry-run) DRY_RUN=1; shift;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown arg: $1"; usage; exit 2;;
  esac
done

if ! command -v bsub >/dev/null 2>&1; then
  echo "bsub not found in PATH. Is this an LSF cluster?" >&2
  exit 2
fi

if [[ -z "$LOG_DIR" ]]; then
  LOG_DIR="${ROOT}/logs/run_detections_batch/bsub_submissions"
fi

RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)_$$"
RUN_DIR="${LOG_DIR}/detect_${RUN_ID}"
mkdir -p "$RUN_DIR"

python - "$ROOT" "$BATCH_SIZE" "$RUN_DIR" <<'PY'
import sys
from pathlib import Path

root = Path(sys.argv[1]).expanduser()
batch_size = int(sys.argv[2])
run_dir = Path(sys.argv[3])

recordings = sorted({str(p.parent.parent) for p in root.rglob("raw/*.h5")})
run_dir.mkdir(parents=True, exist_ok=True)

recordings_file = run_dir / "recordings.txt"
recordings_file.write_text("\n".join(recordings) + ("\n" if recordings else ""))

for idx in range(0, len(recordings), batch_size):
    batch_idx = idx // batch_size + 1
    batch_file = run_dir / f"batch_{batch_idx}.txt"
    batch_file.write_text("\n".join(recordings[idx:idx + batch_size]) + "\n")
PY

batch_count=$(ls -1 "$RUN_DIR"/batch_*.txt 2>/dev/null | wc -l | tr -d ' ')
if [[ "$batch_count" == "0" ]]; then
  echo "No recordings found under $ROOT"
  exit 0
fi

EXTRA_ARGS=(--apply --recursive --no-dask-progress --scheduler "$SCHEDULER" --config "$CONFIG")
if [[ -n "$NUM_WORKERS" ]]; then
  EXTRA_ARGS+=(--num-workers "$NUM_WORKERS")
fi
if [[ "$REQUIRE_TUNING" == "1" ]]; then
  EXTRA_ARGS+=(--require-tuning)
fi
if [[ "$OVERWRITE" == "1" ]]; then
  EXTRA_ARGS+=(--overwrite)
fi

JOB_SCRIPT="${RUN_DIR}/run_batch.sh"
cat > "$JOB_SCRIPT" <<'BASH'
#!/usr/bin/env bash
set -euo pipefail
RUN_DIR="$1"
if [[ -z "${LSB_JOBINDEX:-}" ]]; then
  echo "LSB_JOBINDEX not set; are you running under bsub array?" >&2
  exit 2
fi
BATCH_FILE="${RUN_DIR}/batch_${LSB_JOBINDEX}.txt"
if [[ ! -f "$BATCH_FILE" ]]; then
  echo "Missing batch file: $BATCH_FILE" >&2
  exit 2
fi
mapfile -t recs < "$BATCH_FILE"
if [[ "${#recs[@]}" -eq 0 ]]; then
  echo "Empty batch file: $BATCH_FILE"
  exit 0
fi
python src/fisheye/utils/run_detections_batch.py "${EXTRA_ARGS[@]}" "${recs[@]}"
BASH
chmod +x "$JOB_SCRIPT"

BSUB_ARGS=(-J "detect_batch[1-${batch_count}]%${MAX_ACTIVE}" -n "$NCORES" -R "rusage[mem=${MEM_GB}G]" -oo "${RUN_DIR}/%J_%I.out" -eo "${RUN_DIR}/%J_%I.err")
if [[ -n "$QUEUE" ]]; then
  BSUB_ARGS+=(-q "$QUEUE")
fi

echo "Run dir: $RUN_DIR"
echo "Batches: $batch_count"
echo "bsub args: ${BSUB_ARGS[*]}"
echo "Detect args: ${EXTRA_ARGS[*]}"

if [[ "$DRY_RUN" == "1" ]]; then
  echo "Dry run only; no submission."
  exit 0
fi

bsub "${BSUB_ARGS[@]}" bash "$JOB_SCRIPT" "$RUN_DIR"
