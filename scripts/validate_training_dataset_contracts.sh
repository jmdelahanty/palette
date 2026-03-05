#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/validate_training_dataset_contracts.sh [options]

Validate merged training datasets against the expected Zarr contract/spec.

It scans for:
  <datasets-root>/*/zarr/*_merged.zarr

Dataset classification:
  - detect_*   -> fisheye.utils.validate_detect_training_zarr
  - pose_*     -> fisheye.utils.validate_keypoint_training_zarr
  - eye_mask_* -> fisheye.utils.validate_eye_mask_training_zarr
  - otherwise  -> tries detect, keypoint, then eye-mask validator

Options:
  --datasets-root DIR     Root containing dataset bundles
                          (default: /nvme1/training/datasets)
  --timeout-seconds N     Per-dataset validator timeout in seconds
                          (default: 120)
  --output-csv PATH       Output CSV path
                          (default: /tmp/training_dataset_contract_audit.csv)
  --log-dir DIR           Directory for per-dataset validator logs
                          (default: /tmp/training_dataset_contract_audit_logs_<timestamp>)
  -h, --help              Show this help
EOF
}

DATASETS_ROOT="/nvme1/training/datasets"
TIMEOUT_SECONDS="120"
OUTPUT_CSV="/tmp/training_dataset_contract_audit.csv"
LOG_DIR=""
HEARTBEAT_SECONDS=15
CURRENT_PID=""
CURRENT_VALIDATOR=""
CURRENT_DATASET=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --datasets-root)
      DATASETS_ROOT="$2"
      shift 2
      ;;
    --timeout-seconds)
      TIMEOUT_SECONDS="$2"
      shift 2
      ;;
    --output-csv)
      OUTPUT_CSV="$2"
      shift 2
      ;;
    --log-dir)
      LOG_DIR="$2"
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

if [[ ! -d "$DATASETS_ROOT" ]]; then
  echo "Datasets root not found: $DATASETS_ROOT" >&2
  exit 2
fi

if [[ ! -x "scripts/py" ]]; then
  echo "Expected executable wrapper not found: scripts/py" >&2
  exit 2
fi

if ! command -v timeout >/dev/null 2>&1; then
  echo "'timeout' command is required but not found on PATH." >&2
  exit 2
fi

if ! [[ "$TIMEOUT_SECONDS" =~ ^[0-9]+$ ]] || [[ "$TIMEOUT_SECONDS" -le 0 ]]; then
  echo "--timeout-seconds must be a positive integer (got: $TIMEOUT_SECONDS)" >&2
  exit 2
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
if [[ -z "$LOG_DIR" ]]; then
  LOG_DIR="/tmp/training_dataset_contract_audit_logs_${STAMP}"
fi
mkdir -p "$LOG_DIR"
mkdir -p "$(dirname "$OUTPUT_CSV")"

on_interrupt() {
  echo
  echo "Interrupt received. Stopping current validator..."
  if [[ -n "${CURRENT_PID}" ]] && kill -0 "${CURRENT_PID}" 2>/dev/null; then
    kill -INT "${CURRENT_PID}" 2>/dev/null || true
    sleep 1
    kill -TERM "${CURRENT_PID}" 2>/dev/null || true
  fi
  echo "Aborted."
  exit 130
}

trap on_interrupt INT TERM

csv_escape() {
  local s="$1"
  s="${s//$'\n'/ }"
  s="${s//$'\r'/ }"
  s="${s//\"/\"\"}"
  printf '"%s"' "$s"
}

append_csv_row() {
  local kind="$1"
  local status="$2"
  local path="$3"
  local validator="$4"
  local summary="$5"
  {
    csv_escape "$kind"; printf ','
    csv_escape "$status"; printf ','
    csv_escape "$path"; printf ','
    csv_escape "$validator"; printf ','
    csv_escape "$summary"; printf '\n'
  } >> "$OUTPUT_CSV"
}

log_stub_for_path() {
  local path="$1"
  local base
  local hash
  base="$(basename "$path")"
  hash="$(printf '%s' "$path" | sha256sum | awk '{print $1}' | cut -c1-12)"
  printf '%s_%s' "$base" "$hash"
}

LAST_RC=0
LAST_SUMMARY=""

run_validator_capture() {
  local validator="$1"
  local zarr_path="$2"
  local log_file="$3"
  local start_epoch=0
  local now_epoch=0
  local last_heartbeat=0
  local elapsed=0
  local first_nonempty=""
  local last_nonempty=""
  local rc=0

  echo "    -> validator: ${validator}"
  echo "    -> timeout: ${TIMEOUT_SECONDS}s"
  echo "    -> log: ${log_file}"

  set +e
  timeout --foreground "${TIMEOUT_SECONDS}s" scripts/py -m "$validator" "$zarr_path" >"$log_file" 2>&1 &
  CURRENT_PID="$!"
  CURRENT_VALIDATOR="$validator"
  CURRENT_DATASET="$zarr_path"
  start_epoch="$(date +%s)"
  last_heartbeat="$start_epoch"

  while kill -0 "${CURRENT_PID}" 2>/dev/null; do
    sleep 1
    now_epoch="$(date +%s)"
    if (( now_epoch - last_heartbeat >= HEARTBEAT_SECONDS )); then
      elapsed=$((now_epoch - start_epoch))
      echo "       ... running (${elapsed}s elapsed)"
      last_heartbeat="$now_epoch"
    fi
  done

  wait "${CURRENT_PID}"
  rc=$?
  set -e

  CURRENT_PID=""
  CURRENT_VALIDATOR=""
  CURRENT_DATASET=""

  elapsed=$(( $(date +%s) - start_epoch ))
  echo "    -> exit_code: ${rc} (elapsed: ${elapsed}s)"

  LAST_RC="$rc"

  if [[ "$rc" -eq 0 ]]; then
    LAST_SUMMARY="ok"
    return
  fi

  if [[ "$rc" -eq 124 ]]; then
    LAST_SUMMARY="timeout>${TIMEOUT_SECONDS}s"
    return
  fi

  first_nonempty="$(awk 'NF {print; exit}' "$log_file")"
  if [[ "$first_nonempty" == "Traceback (most recent call last):" ]]; then
    first_nonempty="$(grep -E '^(RuntimeError|ValueError|KeyError|AssertionError|SystemExit|Exception|zarr\.errors\.)' "$log_file" | head -n1 || true)"
  fi
  if [[ "$first_nonempty" == ValueError:*validation\ failed:* ]]; then
    local first_bullet=""
    first_bullet="$(grep -E '^- ' "$log_file" | head -n1 || true)"
    if [[ -n "$first_bullet" ]]; then
      first_nonempty="${first_nonempty} ${first_bullet}"
    fi
  fi
  if [[ -z "$first_nonempty" ]]; then
    last_nonempty="$(awk 'NF {line=$0} END {print line}' "$log_file")"
    first_nonempty="$last_nonempty"
  fi
  if [[ -z "$first_nonempty" ]]; then
    LAST_SUMMARY="exit_code=${rc}"
  else
    LAST_SUMMARY="$first_nonempty"
  fi
}

printf '%s\n' "kind,status,path,validator,error_summary" > "$OUTPUT_CSV"

mapfile -t MERGED_ZARRS < <(find "$DATASETS_ROOT" -type d -path '*/zarr/*_merged.zarr' | sort)
if [[ "${#MERGED_ZARRS[@]}" -eq 0 ]]; then
  echo "No merged datasets found under: $DATASETS_ROOT"
  echo "CSV written: $OUTPUT_CSV"
  exit 0
fi
TOTAL_DATASETS="${#MERGED_ZARRS[@]}"

total=0
ok=0
fail=0
timeouts=0
declare -a FAIL_ROWS=()

for zarr_path in "${MERGED_ZARRS[@]}"; do
  total=$((total + 1))
  base="$(basename "$zarr_path")"
  stub="$(log_stub_for_path "$zarr_path")"

  echo
  echo "[${total}/${TOTAL_DATASETS}] ${zarr_path}"

  kind=""
  status=""
  validator_label=""
  summary=""

  if [[ "$base" == pose_* ]]; then
    kind="pose"
    validator_label="fisheye.utils.validate_keypoint_training_zarr"
    run_validator_capture "$validator_label" "$zarr_path" "$LOG_DIR/${stub}.pose.log"
    if [[ "$LAST_RC" -eq 0 ]]; then
      status="ok"
      summary=""
    else
      status="fail"
      summary="$LAST_SUMMARY"
      [[ "$LAST_RC" -eq 124 ]] && timeouts=$((timeouts + 1))
    fi
  elif [[ "$base" == detect_* ]]; then
    kind="detect"
    validator_label="fisheye.utils.validate_detect_training_zarr"
    run_validator_capture "$validator_label" "$zarr_path" "$LOG_DIR/${stub}.detect.log"
    if [[ "$LAST_RC" -eq 0 ]]; then
      status="ok"
      summary=""
    else
      status="fail"
      summary="$LAST_SUMMARY"
      [[ "$LAST_RC" -eq 124 ]] && timeouts=$((timeouts + 1))
    fi
  elif [[ "$base" == eye_mask_* ]]; then
    kind="eye_mask"
    validator_label="fisheye.utils.validate_eye_mask_training_zarr"
    run_validator_capture "$validator_label" "$zarr_path" "$LOG_DIR/${stub}.eye_mask.log"
    if [[ "$LAST_RC" -eq 0 ]]; then
      status="ok"
      summary=""
    else
      status="fail"
      summary="$LAST_SUMMARY"
      [[ "$LAST_RC" -eq 124 ]] && timeouts=$((timeouts + 1))
    fi
  else
    kind="unknown"
    local_detect_validator="fisheye.utils.validate_detect_training_zarr"
    local_kp_validator="fisheye.utils.validate_keypoint_training_zarr"
    local_eye_validator="fisheye.utils.validate_eye_mask_training_zarr"

    run_validator_capture "$local_detect_validator" "$zarr_path" "$LOG_DIR/${stub}.unknown.detect.log"
    rc_detect="$LAST_RC"
    summary_detect="$LAST_SUMMARY"
    if [[ "$rc_detect" -eq 0 ]]; then
      validator_label="$local_detect_validator"
      status="ok"
      summary=""
    else
      echo "    -> detect validator failed, trying keypoint validator"
      run_validator_capture "$local_kp_validator" "$zarr_path" "$LOG_DIR/${stub}.unknown.keypoint.log"
      rc_kp="$LAST_RC"
      summary_kp="$LAST_SUMMARY"
      if [[ "$rc_kp" -eq 0 ]]; then
        validator_label="$local_kp_validator"
        status="ok"
        summary="fallback_after_detect_fail: ${summary_detect}"
      else
        echo "    -> keypoint validator failed, trying eye-mask validator"
        run_validator_capture "$local_eye_validator" "$zarr_path" "$LOG_DIR/${stub}.unknown.eye_mask.log"
        rc_eye="$LAST_RC"
        summary_eye="$LAST_SUMMARY"
        if [[ "$rc_eye" -eq 0 ]]; then
          validator_label="$local_eye_validator"
          status="ok"
          summary="fallback_after_detect_keypoint_fail: detect=${summary_detect}; keypoint=${summary_kp}"
        else
          validator_label="${local_detect_validator};${local_kp_validator};${local_eye_validator}"
          status="fail"
          summary="detect: ${summary_detect} | keypoint: ${summary_kp} | eye_mask: ${summary_eye}"
          if [[ "$rc_detect" -eq 124 || "$rc_kp" -eq 124 || "$rc_eye" -eq 124 ]]; then
            timeouts=$((timeouts + 1))
          fi
        fi
      fi
    fi
  fi

  append_csv_row "$kind" "$status" "$zarr_path" "$validator_label" "$summary"

  if [[ "$status" == "ok" ]]; then
    ok=$((ok + 1))
    echo "[OK]   ${kind}  ${zarr_path}"
  else
    fail=$((fail + 1))
    FAIL_ROWS+=("${kind}|${zarr_path}|${summary}")
    echo "[FAIL] ${kind}  ${zarr_path} :: ${summary}"
  fi
done

echo
echo "Audit complete."
echo "  Total: ${total}"
echo "  Pass:  ${ok}"
echo "  Fail:  ${fail}"
echo "  Timeout-failures: ${timeouts}"
echo "  CSV:   ${OUTPUT_CSV}"
echo "  Logs:  ${LOG_DIR}"

if [[ "${#FAIL_ROWS[@]}" -gt 0 ]]; then
  echo
  echo "Failures:"
  for row in "${FAIL_ROWS[@]}"; do
    IFS='|' read -r row_kind row_path row_summary <<<"$row"
    echo "  - [${row_kind}] ${row_path} :: ${row_summary}"
  done
fi
