#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/validate_pose_detect_parity_smoke.sh [options]

Runs the pose/detect parity end-to-end smoke flow on one analysis zarr:
1) run keypoints via registry model resolution
2) run refined_keypoints for the produced keypoint run
3) rescan registry for the target zarr
4) verify keypoint quality/performance rows via registry_query

Options:
  --recording-dir DIR          Recording directory containing zarr/<name>_analysis.zarr
                               (required)
  --registry PATH              Registry SQLite path
                               (default: /nvme1/palette_registry.sqlite)
  --output-zarr PATH           Explicit analysis zarr path (default derived from recording dir)
  --keypoint-method NAME       Expected keypoint method in registry_query checks
                               (default: yolo_pose)
  --run-name NAME              Optional explicit keypoint run name
  --crop-run NAME              Optional explicit crop run override for keypoint stage
  --top-k N                    Candidate count persisted in model-resolution provenance
                               (default: 5)
  --cpu                        Force CPU inference for keypoint stage
  --tmp-root DIR               Temp parent directory (default: /tmp)
  -h, --help                   Show this help
EOF
}

RECORDING_DIR=""
REGISTRY="/nvme1/palette_registry.sqlite"
OUTPUT_ZARR=""
KEYPOINT_METHOD="yolo_pose"
RUN_NAME=""
CROP_RUN=""
TOP_K="5"
FORCE_CPU="0"
TMP_ROOT="/tmp"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --recording-dir)
      RECORDING_DIR="$2"
      shift 2
      ;;
    --registry)
      REGISTRY="$2"
      shift 2
      ;;
    --output-zarr)
      OUTPUT_ZARR="$2"
      shift 2
      ;;
    --keypoint-method)
      KEYPOINT_METHOD="$2"
      shift 2
      ;;
    --run-name)
      RUN_NAME="$2"
      shift 2
      ;;
    --crop-run)
      CROP_RUN="$2"
      shift 2
      ;;
    --top-k)
      TOP_K="$2"
      shift 2
      ;;
    --cpu)
      FORCE_CPU="1"
      shift
      ;;
    --tmp-root)
      TMP_ROOT="$2"
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

if [[ -z "$RECORDING_DIR" ]]; then
  echo "--recording-dir is required." >&2
  usage
  exit 2
fi

if [[ ! -x "scripts/py" ]]; then
  echo "Expected executable wrapper not found: scripts/py" >&2
  exit 2
fi

if [[ ! -f "$REGISTRY" ]]; then
  echo "Registry not found: $REGISTRY" >&2
  exit 2
fi

if ! command -v rg >/dev/null 2>&1; then
  echo "ripgrep (rg) is required for log extraction." >&2
  exit 2
fi

RECORDING_DIR="$(cd "$RECORDING_DIR" && pwd)"
if [[ -z "$OUTPUT_ZARR" ]]; then
  RECORDING_NAME="$(basename "$RECORDING_DIR")"
  OUTPUT_ZARR="$RECORDING_DIR/zarr/${RECORDING_NAME}_analysis.zarr"
fi

if [[ ! -d "$OUTPUT_ZARR" ]]; then
  echo "Analysis zarr not found: $OUTPUT_ZARR" >&2
  exit 2
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
TMP_DIR="${TMP_ROOT%/}/pose_parity_smoke_${STAMP}"
mkdir -p "$TMP_DIR"

section() {
  printf '\n=== %s ===\n' "$1"
}

echo "TMP_DIR=$TMP_DIR"
echo "REGISTRY=$REGISTRY"
echo "RECORDING_DIR=$RECORDING_DIR"
echo "OUTPUT_ZARR=$OUTPUT_ZARR"
echo "KEYPOINT_METHOD=$KEYPOINT_METHOD"
echo "TOP_K=$TOP_K"

section "Run Keypoints With Registry Model"
KEYPOINT_CMD=(
  scripts/py -m fisheye.utils.run_keypoints_with_registry_model
  --recording-dir "$RECORDING_DIR"
  --registry "$REGISTRY"
  --output "$OUTPUT_ZARR"
  --top-k "$TOP_K"
  --json
)
if [[ "$FORCE_CPU" == "1" ]]; then
  KEYPOINT_CMD+=(--cpu)
fi
if [[ -n "$RUN_NAME" ]]; then
  KEYPOINT_CMD+=(--run-name "$RUN_NAME")
fi
if [[ -n "$CROP_RUN" ]]; then
  KEYPOINT_CMD+=(--crop-run "$CROP_RUN")
fi
"${KEYPOINT_CMD[@]}" | tee "$TMP_DIR/keypoints.log"

KEYPOINT_RUN="$(rg '^  keypoint_run: ' "$TMP_DIR/keypoints.log" | tail -n 1 | sed 's/^  keypoint_run: //')"
if [[ -z "$KEYPOINT_RUN" ]]; then
  echo "Could not extract keypoint run name from keypoint command output." >&2
  exit 1
fi
echo "keypoint_run=$KEYPOINT_RUN"

section "Refine Keypoints For Produced Run"
scripts/py -m fisheye.refinement.refine_keypoints \
  "$OUTPUT_ZARR" \
  --config configs/fisheye/default.yaml \
  --keypoint-run "$KEYPOINT_RUN" \
  >"$TMP_DIR/refine.log" 2>&1
tail -n 40 "$TMP_DIR/refine.log"

section "Registry Scan (Target Zarr)"
scripts/py -m fisheye.registry.scan \
  --registry "$REGISTRY" \
  "$OUTPUT_ZARR" \
  >"$TMP_DIR/scan.log" 2>&1
cat "$TMP_DIR/scan.log"

TARGET_CONTAINS="$(basename "$OUTPUT_ZARR")"

section "Registry Query: Keypoint Performance Slice"
scripts/py -m fisheye.utils.registry_query \
  --registry "$REGISTRY" \
  --path-contains "$TARGET_CONTAINS" \
  --keypoint-method "$KEYPOINT_METHOD" \
  --keypoint-success-rate-min 0 \
  --json \
  >"$TMP_DIR/keypoint_perf.json"
cat "$TMP_DIR/keypoint_perf.json"

section "Registry Query: Keypoint Quality Slice"
scripts/py -m fisheye.utils.registry_query \
  --registry "$REGISTRY" \
  --path-contains "$TARGET_CONTAINS" \
  --keypoint-method "$KEYPOINT_METHOD" \
  --json \
  >"$TMP_DIR/keypoint_quality.json"
cat "$TMP_DIR/keypoint_quality.json"

section "Registry Query: Keypoint Group Summary"
scripts/py -m fisheye.utils.registry_query \
  --registry "$REGISTRY" \
  --path-contains "$TARGET_CONTAINS" \
  --keypoint-group-by method \
  --json \
  >"$TMP_DIR/keypoint_group_summary.json"
cat "$TMP_DIR/keypoint_group_summary.json"

PERF_ROWS="$(scripts/py -c "import json; print(len(json.load(open('$TMP_DIR/keypoint_perf.json', 'r', encoding='utf-8'))))")"
QUALITY_ROWS="$(scripts/py -c "import json; print(len(json.load(open('$TMP_DIR/keypoint_quality.json', 'r', encoding='utf-8'))))")"
SUMMARY_ROWS="$(scripts/py -c "import json; print(len(json.load(open('$TMP_DIR/keypoint_group_summary.json', 'r', encoding='utf-8'))))")"

PERF_PASS=0
QUALITY_PASS=0
SUMMARY_PASS=0

if [[ "$PERF_ROWS" -gt 0 ]]; then
  PERF_PASS=1
fi
if [[ "$QUALITY_ROWS" -gt 0 ]]; then
  QUALITY_PASS=1
fi
if [[ "$SUMMARY_ROWS" -gt 0 ]]; then
  SUMMARY_PASS=1
fi

section "Smoke Summary"
echo "keypoint_run_detected=1"
echo "performance_rows=$PERF_ROWS"
echo "quality_rows=$QUALITY_ROWS"
echo "group_summary_rows=$SUMMARY_ROWS"
echo "performance_query_has_rows=$PERF_PASS"
echo "quality_query_has_rows=$QUALITY_PASS"
echo "group_summary_has_rows=$SUMMARY_PASS"
echo "artifact_dir=$TMP_DIR"

if [[ "$PERF_PASS" -ne 1 || "$QUALITY_PASS" -ne 1 || "$SUMMARY_PASS" -ne 1 ]]; then
  echo "Smoke test failed: expected all query checks to return at least one row." >&2
  exit 1
fi

echo "Smoke test passed."
