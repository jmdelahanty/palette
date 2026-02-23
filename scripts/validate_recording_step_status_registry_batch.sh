#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/validate_recording_step_status_registry_batch.sh [options]

Batch wrapper for recording-step-status validation across many recordings.
Runs one-recording validation in parallel via:
  scripts/validate_recording_step_status_registry.sh

Options:
  --recordings-root DIR      Root containing recording directories
                             (default: /nvme1/recordings)
  --recording-dir DIR        Specific recording directory to include (repeatable)
  --registry PATH            Registry SQLite path
                             (default: /nvme1/palette_registry.sqlite)
  --zarr-use USE             Zarr use to validate: training|analysis
                             (default: training)
  --jobs N                   Parallel workers (default: 4)
  --skip-backfill            Pass through --skip-backfill to per-recording runs
  --tmp-root DIR             Temp parent directory (default: /tmp)
  --max-recordings N         Optional cap after filtering
  -h, --help                 Show this help
EOF
}

RECORDINGS_ROOT="/nvme1/recordings"
declare -a RECORDING_DIRS=()
REGISTRY="/nvme1/palette_registry.sqlite"
ZARR_USE="training"
JOBS="4"
SKIP_BACKFILL="0"
TMP_ROOT="/tmp"
MAX_RECORDINGS=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --recordings-root)
      RECORDINGS_ROOT="$2"
      shift 2
      ;;
    --recording-dir)
      RECORDING_DIRS+=("$2")
      shift 2
      ;;
    --registry)
      REGISTRY="$2"
      shift 2
      ;;
    --zarr-use)
      ZARR_USE="$2"
      shift 2
      ;;
    --jobs)
      JOBS="$2"
      shift 2
      ;;
    --skip-backfill)
      SKIP_BACKFILL="1"
      shift
      ;;
    --tmp-root)
      TMP_ROOT="$2"
      shift 2
      ;;
    --max-recordings)
      MAX_RECORDINGS="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if [[ "$ZARR_USE" != "training" && "$ZARR_USE" != "analysis" ]]; then
  echo "--zarr-use must be 'training' or 'analysis'." >&2
  exit 2
fi

if ! [[ "$JOBS" =~ ^[1-9][0-9]*$ ]]; then
  echo "--jobs must be a positive integer." >&2
  exit 2
fi

if [[ -n "$MAX_RECORDINGS" ]] && ! [[ "$MAX_RECORDINGS" =~ ^[1-9][0-9]*$ ]]; then
  echo "--max-recordings must be a positive integer when set." >&2
  exit 2
fi

if [[ ! -f "$REGISTRY" ]]; then
  echo "Registry not found: $REGISTRY" >&2
  exit 2
fi

VALIDATOR="scripts/validate_recording_step_status_registry.sh"
if [[ ! -x "$VALIDATOR" ]]; then
  echo "Expected executable validator not found: $VALIDATOR" >&2
  exit 2
fi

if ! command -v xargs >/dev/null 2>&1; then
  echo "xargs is required." >&2
  exit 2
fi

declare -a CANDIDATES=()
if (( ${#RECORDING_DIRS[@]} > 0 )); then
  for dir in "${RECORDING_DIRS[@]}"; do
    if [[ -d "$dir" ]]; then
      CANDIDATES+=("$(cd "$dir" && pwd)")
    else
      echo "Skipping missing --recording-dir: $dir" >&2
    fi
  done
else
  if [[ ! -d "$RECORDINGS_ROOT" ]]; then
    echo "Recordings root not found: $RECORDINGS_ROOT" >&2
    exit 2
  fi
  while IFS= read -r -d '' dir; do
    CANDIDATES+=("$dir")
  done < <(find "$RECORDINGS_ROOT" -mindepth 1 -maxdepth 1 -type d -print0 | sort -z)
fi

if (( ${#CANDIDATES[@]} == 0 )); then
  echo "No candidate recording directories found." >&2
  exit 1
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${TMP_ROOT%/}/recording_step_status_batch_${STAMP}"
LOG_DIR="$RUN_DIR/logs"
PASS_LIST="$RUN_DIR/passed.txt"
FAIL_LIST="$RUN_DIR/failed.txt"
SKIP_LIST="$RUN_DIR/skipped_missing_zarr.txt"
mkdir -p "$LOG_DIR"
: > "$PASS_LIST"
: > "$FAIL_LIST"
: > "$SKIP_LIST"

declare -a TARGETS=()
for rec in "${CANDIDATES[@]}"; do
  name="$(basename "$rec")"
  target_zarr="$rec/zarr/${name}_${ZARR_USE}.zarr"
  if [[ -d "$target_zarr" ]]; then
    TARGETS+=("$rec")
  else
    printf '%s\n' "$rec" >> "$SKIP_LIST"
  fi
done

if [[ -n "$MAX_RECORDINGS" ]] && (( ${#TARGETS[@]} > MAX_RECORDINGS )); then
  TARGETS=("${TARGETS[@]:0:MAX_RECORDINGS}")
fi

if (( ${#TARGETS[@]} == 0 )); then
  echo "No recordings contain ${ZARR_USE} zarr targets. See $SKIP_LIST" >&2
  exit 1
fi

echo "RUN_DIR=$RUN_DIR"
echo "REGISTRY=$REGISTRY"
echo "ZARR_USE=$ZARR_USE"
echo "JOBS=$JOBS"
echo "TOTAL_CANDIDATES=${#CANDIDATES[@]}"
echo "TOTAL_TARGETS=${#TARGETS[@]}"
echo "TOTAL_SKIPPED_MISSING_ZARR=$(wc -l < "$SKIP_LIST" | tr -d ' ')"

export VALIDATOR REGISTRY ZARR_USE SKIP_BACKFILL TMP_ROOT RUN_DIR LOG_DIR PASS_LIST FAIL_LIST

set +e
printf '%s\0' "${TARGETS[@]}" | xargs -0 -n1 -P "$JOBS" bash -c '
  rec="$1"
  name="$(basename "$rec")"
  safe_name="$(printf "%s" "$name" | tr "/ " "__")"
  log_path="$LOG_DIR/${safe_name}.log"
  cmd=("$VALIDATOR" --recording-dir "$rec" --registry "$REGISTRY" --zarr-use "$ZARR_USE" --tmp-root "$TMP_ROOT")
  if [[ "$SKIP_BACKFILL" == "1" ]]; then
    cmd+=(--skip-backfill)
  fi
  if "${cmd[@]}" >"$log_path" 2>&1; then
    printf "%s\n" "$rec" >> "$PASS_LIST"
    exit 0
  fi
  printf "%s\n" "$rec" >> "$FAIL_LIST"
  exit 1
' _
XARGS_RC=$?
set -e

PASS_COUNT="$(wc -l < "$PASS_LIST" | tr -d ' ')"
FAIL_COUNT="$(wc -l < "$FAIL_LIST" | tr -d ' ')"
SKIP_COUNT="$(wc -l < "$SKIP_LIST" | tr -d ' ')"

echo
echo "=== Batch Validation Summary ==="
echo "run_dir=$RUN_DIR"
echo "passed=$PASS_COUNT"
echo "failed=$FAIL_COUNT"
echo "skipped_missing_zarr=$SKIP_COUNT"
echo "pass_list=$PASS_LIST"
echo "fail_list=$FAIL_LIST"
echo "skip_list=$SKIP_LIST"
echo "logs_dir=$LOG_DIR"

if [[ "$FAIL_COUNT" != "0" ]]; then
  echo
  echo "Failed recordings:"
  cat "$FAIL_LIST"
  echo
  echo "Inspect logs under: $LOG_DIR"
  exit 1
fi

if [[ "$XARGS_RC" -ne 0 ]]; then
  echo "Batch run exited non-zero despite zero recorded failures." >&2
  exit 1
fi

echo "Batch validation passed."
