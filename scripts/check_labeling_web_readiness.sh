#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-/tmp/palette-pycache}"

echo "==> Checking production decision record"
"$ROOT/scripts/py" "$ROOT/scripts/check_labeling_production_decision_record.py" \
  --record "$ROOT/docs/web_labeling_production_decision_record.md"

echo "==> Compiling labeling web implementation and focused tests"
"$ROOT/scripts/check_labeling_web_static.sh"

echo "==> Running focused non-zarr labeling web tests"
"$ROOT/scripts/check_labeling_web_unit.sh"

if [[ -n "${PALETTE_LABELING_WEB_REAL_ZARR_SMOKE_SPEC:-}" ]]; then
  echo "==> Running real-zarr labeling web smoke from ${PALETTE_LABELING_WEB_REAL_ZARR_SMOKE_SPEC}"
  "$ROOT/scripts/py" -m pytest -p no:cacheprovider \
    "$ROOT/tests/integration/fisheye/test_labeling_web_real_zarr_smoke.py" -q
else
  echo "==> Skipping real-zarr smoke; set PALETTE_LABELING_WEB_REAL_ZARR_SMOKE_SPEC to enable it"
fi
