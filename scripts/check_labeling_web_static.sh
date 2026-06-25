#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-/tmp/palette-pycache}"

exec "$ROOT/scripts/py" -m py_compile \
  "$ROOT/src/fisheye/labeling/__init__.py" \
  "$ROOT/src/fisheye/labeling/assignment_store.py" \
  "$ROOT/src/fisheye/labeling/web.py" \
  "$ROOT/src/fisheye/utils/labeling_work.py" \
  "$ROOT/tests/unit/fisheye/test_labeling_assignment_store.py" \
  "$ROOT/tests/unit/fisheye/test_labeling_signed_links.py" \
  "$ROOT/tests/unit/fisheye/test_labeling_web_security.py" \
  "$ROOT/tests/unit/fisheye/test_labeling_promotion_retry.py" \
  "$ROOT/tests/unit/fisheye/test_labeling_web_routes.py" \
  "$ROOT/tests/unit/fisheye/test_labeling_web_config.py" \
  "$ROOT/tests/integration/fisheye/test_labeling_web_real_zarr_smoke.py"
