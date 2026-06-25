#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-/tmp/palette-pycache}"

exec "$ROOT/scripts/py" -m pytest -p no:cacheprovider \
  "$ROOT/tests/unit/fisheye/test_labeling_assignment_store.py" \
  "$ROOT/tests/unit/fisheye/test_labeling_signed_links.py" \
  "$ROOT/tests/unit/fisheye/test_labeling_web_security.py" \
  "$ROOT/tests/unit/fisheye/test_labeling_promotion_retry.py" \
  "$ROOT/tests/unit/fisheye/test_labeling_web_routes.py" \
  "$ROOT/tests/unit/fisheye/test_labeling_web_config.py" \
  -q
