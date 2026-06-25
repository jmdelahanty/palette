#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STORE="${1:-/tmp/palette_labeling_first_operator_test.sqlite}"
USER_NAME="${2:-alice}"
PORT="${3:-8795}"

if [[ ! -f "$STORE" ]]; then
  cat >&2 <<EOF
Labeling smoke store not found:
  $STORE

Create it first:
  scripts/setup_labeling_web_local_smoke_store.sh "$STORE"
EOF
  exit 2
fi

exec "$ROOT/scripts/py" -m fisheye.utils.labeling_work --store "$STORE" serve \
  --user "$USER_NAME" \
  --admin-user "$USER_NAME" \
  --host 127.0.0.1 \
  --port "$PORT" \
  --link-secret local-test-secret \
  --access-log
