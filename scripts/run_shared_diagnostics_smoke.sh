#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/run_shared_diagnostics_smoke.sh [options]

Runs the unified video and H5 diagnostics against a shared real-data fixture and
writes outputs under the shared runs directory.

Options:
  --recording-dir DIR   Organized recording fixture to inspect
                        (default: $PALETTE_TEST_FIXTURES_ROOT/recordings/2026-01-28T19-36-18Z_arena_1_Feeding)
  --output-dir DIR      Explicit output directory
                        (default: $PALETTE_TEST_RUNS_ROOT/diagnostics/<timestamp>)
  --label NAME          Human-readable label appended to the default output dir
  -h, --help            Show this help
EOF
}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"

PALETTE_TEST_DATA_ROOT="${PALETTE_TEST_DATA_ROOT:-/nvme1/palette_test_data}"
PALETTE_TEST_FIXTURES_ROOT="${PALETTE_TEST_FIXTURES_ROOT:-$PALETTE_TEST_DATA_ROOT/fixtures}"
PALETTE_TEST_RUNS_ROOT="${PALETTE_TEST_RUNS_ROOT:-$PALETTE_TEST_DATA_ROOT/runs}"
DEFAULT_RECORDING_DIR="$PALETTE_TEST_FIXTURES_ROOT/recordings/2026-01-28T19-36-18Z_arena_1_Feeding"

RECORDING_DIR="$DEFAULT_RECORDING_DIR"
OUTPUT_DIR=""
LABEL=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --recording-dir)
      RECORDING_DIR="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --label)
      LABEL="$2"
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

if [[ ! -x "scripts/py" ]]; then
  echo "Expected executable wrapper not found: scripts/py" >&2
  exit 2
fi

if [[ ! -d "$RECORDING_DIR" ]]; then
  echo "Recording fixture not found: $RECORDING_DIR" >&2
  exit 2
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
if [[ -z "$OUTPUT_DIR" ]]; then
  if [[ -n "$LABEL" ]]; then
    OUTPUT_DIR="$PALETTE_TEST_RUNS_ROOT/diagnostics/${STAMP}_${LABEL}"
  else
    OUTPUT_DIR="$PALETTE_TEST_RUNS_ROOT/diagnostics/${STAMP}"
  fi
fi
mkdir -p "$OUTPUT_DIR"

section() {
  printf '
=== %s ===
' "$1"
}

section "Shared Fixture Diagnostics Smoke"
echo "recording_dir=$RECORDING_DIR"
echo "output_dir=$OUTPUT_DIR"

echo "$RECORDING_DIR" > "$OUTPUT_DIR/recording_dir.txt"

section "Video Diagnostics (text + jsonl)"
PYTHONPYCACHEPREFIX=/tmp/palette-pycache scripts/py -m fisheye.diagnostics.video batch   "$RECORDING_DIR"   --jsonl "$OUTPUT_DIR/video_batch.jsonl"   | tee "$OUTPUT_DIR/video_batch.txt"

section "Video Diagnostics (json)"
PYTHONPYCACHEPREFIX=/tmp/palette-pycache scripts/py -m fisheye.diagnostics.video batch   "$RECORDING_DIR"   --json   > "$OUTPUT_DIR/video_batch.json"

section "H5 Diagnostics (text)"
PYTHONPYCACHEPREFIX=/tmp/palette-pycache scripts/py -m fisheye.diagnostics.h5 report   "$RECORDING_DIR"   | tee "$OUTPUT_DIR/h5_report.txt"

section "H5 Diagnostics (json)"
PYTHONPYCACHEPREFIX=/tmp/palette-pycache scripts/py -m fisheye.diagnostics.h5 report   "$RECORDING_DIR"   --json   > "$OUTPUT_DIR/h5_report.json"

section "Artifacts"
find "$OUTPUT_DIR" -maxdepth 1 -type f -printf '%f
' | sort

echo "diagnostics_smoke_ok=1"
echo "artifact_dir=$OUTPUT_DIR"
