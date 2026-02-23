#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/ci_registry_migration_integrity_subset.sh [options] [-- <extra pytest args>]

Runs the targeted registry migration/integrity subset used for CI coverage.

Options:
  --smoke                  Run a reduced smoke subset (fast local check).
  -h, --help               Show this help.

Examples:
  scripts/ci_registry_migration_integrity_subset.sh
  scripts/ci_registry_migration_integrity_subset.sh --smoke
  scripts/ci_registry_migration_integrity_subset.sh -- --maxfail=1 -x
EOF
}

SMOKE="0"
declare -a PYTEST_EXTRA=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --smoke)
      SMOKE="1"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      PYTEST_EXTRA=("$@")
      break
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

run_pytest() {
  local label="$1"
  shift
  echo
  echo "=== $label ==="
  scripts/py -m pytest -q "$@" "${PYTEST_EXTRA[@]}"
}

if [[ "$SMOKE" == "1" ]]; then
  run_pytest "Registry ledger smoke" \
    tests/unit/fisheye/test_registry_status_ledger.py
  run_pytest "Registry maintenance smoke (schema + integrity)" \
    tests/unit/fisheye/test_registry_maintenance.py \
    -k "schema_has_recording_step_status_tables_and_views or integrity_flags_required_view_missing"
  echo
  echo "Smoke subset passed."
  exit 0
fi

run_pytest "Registry maintenance" \
  tests/unit/fisheye/test_registry_maintenance.py
run_pytest "Registry status ledger" \
  tests/unit/fisheye/test_registry_status_ledger.py
run_pytest "Recording step compare tool" \
  tests/unit/fisheye/test_check_recording_steps.py
run_pytest "Training registry views" \
  tests/unit/fisheye/test_check_training_registry.py

echo
echo "Registry migration/integrity subset passed."
