#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/run_organize_preflight_smoke.sh [options]

Clones the shared staging fixture into a fresh run directory, runs
organize_recordings with both unified diagnostics hooks enabled, and preserves
all artifacts under the shared runs root.

Options:
  --staging-fixture DIR  Shared staging batch fixture to clone
                         (default: $PALETTE_TEST_FIXTURES_ROOT/staging/2026_01_28_19_36_18_batch)
  --output-dir DIR       Explicit output directory
                         (default: $PALETTE_TEST_RUNS_ROOT/organize_preflight/<timestamp>)
  --label NAME           Human-readable label appended to the default output dir
  -h, --help             Show this help
EOF
}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"

PALETTE_TEST_DATA_ROOT="${PALETTE_TEST_DATA_ROOT:-/nvme1/palette_test_data}"
PALETTE_TEST_FIXTURES_ROOT="${PALETTE_TEST_FIXTURES_ROOT:-$PALETTE_TEST_DATA_ROOT/fixtures}"
PALETTE_TEST_RUNS_ROOT="${PALETTE_TEST_RUNS_ROOT:-$PALETTE_TEST_DATA_ROOT/runs}"
DEFAULT_STAGING_FIXTURE="$PALETTE_TEST_FIXTURES_ROOT/staging/2026_01_28_19_36_18_batch"

STAGING_FIXTURE="$DEFAULT_STAGING_FIXTURE"
OUTPUT_DIR=""
LABEL=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --staging-fixture)
      STAGING_FIXTURE="$2"
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

if [[ ! -d "$STAGING_FIXTURE" ]]; then
  echo "Staging fixture not found: $STAGING_FIXTURE" >&2
  exit 2
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
if [[ -z "$OUTPUT_DIR" ]]; then
  if [[ -n "$LABEL" ]]; then
    OUTPUT_DIR="$PALETTE_TEST_RUNS_ROOT/organize_preflight/${STAMP}_${LABEL}"
  else
    OUTPUT_DIR="$PALETTE_TEST_RUNS_ROOT/organize_preflight/${STAMP}"
  fi
fi

RUN_STAGING_ROOT="$OUTPUT_DIR/staging"
RUN_RECORDINGS_ROOT="$OUTPUT_DIR/recordings"
RUN_LOG_ROOT="$OUTPUT_DIR/logs"
mkdir -p "$RUN_STAGING_ROOT" "$RUN_RECORDINGS_ROOT" "$RUN_LOG_ROOT"

BATCH_NAME="$(basename "$STAGING_FIXTURE")"
RUN_BATCH="$RUN_STAGING_ROOT/$BATCH_NAME"
cp -a --reflink=auto "$STAGING_FIXTURE" "$RUN_BATCH"

section() {
  printf '
=== %s ===
' "$1"
}

section "Organize Preflight Smoke"
echo "staging_fixture=$STAGING_FIXTURE"
echo "run_batch=$RUN_BATCH"
echo "recordings_root=$RUN_RECORDINGS_ROOT"
echo "log_root=$RUN_LOG_ROOT"
echo "$STAGING_FIXTURE" > "$OUTPUT_DIR/staging_fixture.txt"
echo "$RUN_BATCH" > "$OUTPUT_DIR/run_batch.txt"

section "Run organize_recordings with diagnostics"
PYTHONPYCACHEPREFIX=/tmp/palette-pycache scripts/py -m fisheye.utils.organize_recordings   "$RUN_BATCH"   --recursive   --apply   --write-manifest   --rename-cams   --dest-root "$RUN_RECORDINGS_ROOT"   --log-dir "$RUN_LOG_ROOT"   --run-video-diagnostics   --run-h5-diagnostics   | tee "$OUTPUT_DIR/organize_output.txt"

section "Artifacts"
find "$OUTPUT_DIR" -maxdepth 2 -type f | sort

echo "organize_preflight_smoke_ok=1"
echo "artifact_dir=$OUTPUT_DIR"
