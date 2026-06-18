#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/aggregate_training_data_cards.sh [options]

Refresh registry profile/quality rows, then aggregate training data cards for
all manifests under the datasets root.

By default this script:
1) refreshes profile/quality rows in the registry
2) aggregates detect/pose/eye-mask data cards + plots
3) writes per-manifest logs and a status CSV

Options:
  --registry PATH         Registry sqlite path
                          (default: $PALETTE_REGISTRY_PATH or /groups/johnson/johnsonlab/jeremy/registries/palette_registry.sqlite)
  --datasets-root DIR     Root containing dataset bundles
                          (default: /nvme1/training/datasets)
  --include-legacy-merged-detect
                          Include legacy detect manifests that contain only a
                          single *_merged dataset_id (default: skipped because
                          detection_data_profile_latest rows are unavailable)
  --skip-refresh          Skip registry refresh phase
  --relaxed               Add relaxed aggregate flags for stale/missing-profile scenarios
  --no-force-plots        Do not pass --force for pose/eye-mask plot regeneration
  --log-dir DIR           Directory for phase + per-manifest logs
                          (default: /tmp/aggregate_training_data_cards_logs_<timestamp>)
  --output-csv PATH       Output CSV path for per-manifest status
                          (default: /tmp/aggregate_training_data_cards_status_<timestamp>.csv)
  -h, --help              Show this help
EOF
}

REGISTRY="${PALETTE_REGISTRY_PATH:-/groups/johnson/johnsonlab/jeremy/registries/palette_registry.sqlite}"
DATASETS_ROOT="/nvme1/training/datasets"
SKIP_REFRESH=0
RELAXED=0
FORCE_PLOTS=1
SKIP_LEGACY_MERGED_DETECT=1
LOG_DIR=""
OUTPUT_CSV=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --registry)
      REGISTRY="$2"
      shift 2
      ;;
    --datasets-root)
      DATASETS_ROOT="$2"
      shift 2
      ;;
    --skip-refresh)
      SKIP_REFRESH=1
      shift
      ;;
    --include-legacy-merged-detect)
      SKIP_LEGACY_MERGED_DETECT=0
      shift
      ;;
    --relaxed)
      RELAXED=1
      shift
      ;;
    --no-force-plots)
      FORCE_PLOTS=0
      shift
      ;;
    --log-dir)
      LOG_DIR="$2"
      shift 2
      ;;
    --output-csv)
      OUTPUT_CSV="$2"
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

if [[ ! -e "$REGISTRY" ]]; then
  echo "Registry not found: $REGISTRY" >&2
  exit 2
fi

if [[ ! -d "$DATASETS_ROOT" ]]; then
  echo "Datasets root not found: $DATASETS_ROOT" >&2
  exit 2
fi

if [[ ! -x "scripts/py" ]]; then
  echo "Expected executable wrapper not found: scripts/py" >&2
  exit 2
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
if [[ -z "$LOG_DIR" ]]; then
  LOG_DIR="/tmp/aggregate_training_data_cards_logs_${STAMP}"
fi
if [[ -z "$OUTPUT_CSV" ]]; then
  OUTPUT_CSV="/tmp/aggregate_training_data_cards_status_${STAMP}.csv"
fi

mkdir -p "$LOG_DIR"
mkdir -p "$(dirname "$OUTPUT_CSV")"

csv_escape() {
  local s="$1"
  s="${s//$'\n'/ }"
  s="${s//$'\r'/ }"
  s="${s//\"/\"\"}"
  printf '"%s"' "$s"
}

append_csv_row() {
  local manifest="$1"
  local module="$2"
  local status="$3"
  local rc="$4"
  local summary="$5"
  {
    csv_escape "$manifest"; printf ','
    csv_escape "$module"; printf ','
    csv_escape "$status"; printf ','
    csv_escape "$rc"; printf ','
    csv_escape "$summary"; printf '\n'
  } >> "$OUTPUT_CSV"
}

is_legacy_merged_detect_manifest() {
  local manifest="$1"
  local base
  local dataset_count
  local first_dataset_id

  base="$(basename "$manifest")"
  if [[ "$base" != detect_*.manifest.json ]]; then
    return 1
  fi
  if ! command -v jq >/dev/null 2>&1; then
    return 1
  fi

  dataset_count="$(jq -r 'if (.datasets | type) == "array" then (.datasets | length) else -1 end' "$manifest" 2>/dev/null || echo -1)"
  first_dataset_id="$(jq -r '.datasets[0].dataset_id // ""' "$manifest" 2>/dev/null || echo "")"
  [[ "$dataset_count" == "1" && "$first_dataset_id" == *_merged ]]
}

log_stub_for_path() {
  local path="$1"
  local base
  local hash
  base="$(basename "$path")"
  hash="$(printf '%s' "$path" | sha256sum | awk '{print $1}' | cut -c1-12)"
  printf '%s_%s' "$base" "$hash"
}

printf '%s\n' "manifest,module,status,rc,summary" > "$OUTPUT_CSV"

echo "aggregate_training_data_cards.sh"
echo "  registry:      $REGISTRY"
echo "  datasets_root: $DATASETS_ROOT"
echo "  skip_refresh:  $SKIP_REFRESH"
echo "  skip_legacy_detect: $SKIP_LEGACY_MERGED_DETECT"
echo "  relaxed:       $RELAXED"
echo "  force_plots:   $FORCE_PLOTS"
echo "  log_dir:       $LOG_DIR"
echo "  output_csv:    $OUTPUT_CSV"

if [[ "$SKIP_REFRESH" -eq 0 ]]; then
  echo
  echo "[refresh 1/3] sync detection profile rows"
  scripts/py -m fisheye.utils.sync_detection_profile_registry \
    --registry "$REGISTRY" \
    --zarr-use any \
    --apply | tee "$LOG_DIR/01_sync_detection_profile.log"

  echo
  echo "[refresh 2/3] sync eye-mask profile rows"
  scripts/py -m fisheye.utils.sync_eye_mask_profile_registry \
    --registry "$REGISTRY" \
    --zarr-use any \
    --apply | tee "$LOG_DIR/02_sync_eye_mask_profile.log"

  echo
  echo "[refresh 3/3] refresh maintenance profile/quality rows"
  scripts/py -m fisheye.registry.maintenance \
    --registry "$REGISTRY" \
    --refresh-keypoint-profiles \
    --refresh-eye-mask-profiles \
    --refresh-keypoint-quality \
    --refresh-eye-mask-quality \
    --refresh-detect-quality | tee "$LOG_DIR/03_refresh_registry_rows.log"
fi

mapfile -t MANIFESTS < <(find "$DATASETS_ROOT" -maxdepth 2 -type f -name '*.manifest.json' | sort)
if [[ "${#MANIFESTS[@]}" -eq 0 ]]; then
  echo
  echo "No manifests found under $DATASETS_ROOT"
  exit 0
fi

total=0
ok=0
fail=0
skipped=0
declare -a FAIL_ROWS=()

for manifest in "${MANIFESTS[@]}"; do
  if [[ "$manifest" == */preflight.manifest.json ]]; then
    skipped=$((skipped + 1))
    continue
  fi

  base="$(basename "$manifest")"
  module=""
  kind=""
  cmd=()
  extra=()

  if [[ "$SKIP_LEGACY_MERGED_DETECT" -eq 1 ]] && is_legacy_merged_detect_manifest "$manifest"; then
    skipped=$((skipped + 1))
    echo
    echo "[SKIP legacy detect merged-only] $manifest"
    continue
  fi

  if [[ "$base" == detect_*.manifest.json ]]; then
    kind="detect"
    module="fisheye.utils.aggregate_detection_training_data_card"
    if [[ "$RELAXED" -eq 1 ]]; then
      extra+=(--allow-mtime-mismatch --allow-detection-type-mismatch)
    fi
  elif [[ "$base" == pose_*.manifest.json ]]; then
    kind="pose"
    module="fisheye.utils.aggregate_keypoint_training_data_card"
    if [[ "$FORCE_PLOTS" -eq 1 ]]; then
      extra+=(--force)
    fi
    if [[ "$RELAXED" -eq 1 ]]; then
      extra+=(--allow-profile-mtime-mismatch --allow-profile-fallback-scan)
    fi
  elif [[ "$base" == eye_mask_*.manifest.json ]]; then
    kind="eye_mask"
    module="fisheye.utils.aggregate_eye_mask_training_data_card"
    if [[ "$FORCE_PLOTS" -eq 1 ]]; then
      extra+=(--force)
    fi
    if [[ "$RELAXED" -eq 1 ]]; then
      extra+=(--allow-profile-mtime-mismatch --allow-profile-fallback-scan)
    fi
  else
    skipped=$((skipped + 1))
    continue
  fi

  total=$((total + 1))
  stub="$(log_stub_for_path "$manifest")"
  log_file="$LOG_DIR/${stub}.${kind}.log"

  cmd=(scripts/py -m "$module" --manifest "$manifest" --registry "$REGISTRY" "${extra[@]}")

  echo
  echo "[${total}] $manifest"
  echo "    module: $module"
  if [[ "${#extra[@]}" -gt 0 ]]; then
    echo "    extra:  ${extra[*]}"
  fi
  echo "    log:    $log_file"

  set +e
  "${cmd[@]}" >"$log_file" 2>&1
  rc=$?
  set -e

  if [[ "$rc" -eq 0 ]]; then
    ok=$((ok + 1))
    append_csv_row "$manifest" "$module" "ok" "$rc" "ok"
    echo "    -> [OK]"
    continue
  fi

  fail=$((fail + 1))
  summary="$(rg -m 1 '^Training data card aggregation failed:' "$log_file" || true)"
  if [[ -n "$summary" ]]; then
    summary="${summary#Training data card aggregation failed: }"
  fi
  if [[ -z "$summary" ]]; then
    summary="$(awk 'NF {print; exit}' "$log_file")"
  fi
  if [[ -z "$summary" ]]; then
    summary="exit_code=${rc}"
  fi
  append_csv_row "$manifest" "$module" "fail" "$rc" "$summary"
  FAIL_ROWS+=("$manifest :: $summary")
  echo "    -> [FAIL] rc=$rc"
  echo "       $summary"
done

echo
echo "Aggregate complete."
echo "  total:   $total"
echo "  ok:      $ok"
echo "  fail:    $fail"
echo "  skipped: $skipped"
echo "  csv:     $OUTPUT_CSV"
echo "  logs:    $LOG_DIR"

if [[ "${#FAIL_ROWS[@]}" -gt 0 ]]; then
  echo
  echo "Failures:"
  for row in "${FAIL_ROWS[@]}"; do
    echo "  - $row"
  done
  exit 1
fi

exit 0
